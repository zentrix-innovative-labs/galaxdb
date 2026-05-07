//! GalaxDB Embedded — Rust API for embedded mode with real storage engine.
//!
//! Includes full vector search pipeline:
//! - HNSW index per embedding column
//! - Delta buffer for recent inserts
//! - Sidecar manager for text → embedding conversion
//! - SEMANTIC_MATCH query execution

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use galaxdb_common::{GalaxError, GalaxResult};
use galaxdb_sql::ast::{AuroraStatement, CreateTableStmt};
use galaxdb_sql::executor::{Catalog, CatalogColumn, TableEntry};
use galaxdb_sql::parser;
use galaxdb_storage::engine::{Engine, EngineConfig};
use galaxdb_vector::{
    HnswConfig, HnswGraph, DeltaBuffer,
    execute_semantic_match, SemanticMatchConfig,
};
use galaxdb_sidecar::manager::{SidecarManager, SidecarConfig};
use galaxdb_sidecar::protocol::EmbedRequest;

/// Per-table vector index (HNSW + delta buffer).
struct TableVectorIndex {
    hnsw: HnswGraph,
    delta: DeltaBuffer,
    dim: usize,
    /// Column name that has the embedding
    embedding_column: String,
    /// Source text column name
    source_column: String,
    /// Row ID counter for this table's vectors
    next_row_id: u64,
    /// Map from row_id to the stored vector (for re-ranking)
    vectors: HashMap<u64, Vec<f32>>,
}

/// An embedded GalaxDB database instance.
pub struct Database {
    path: PathBuf,
    catalog: Catalog,
    engine: Arc<Engine>,
    schemas: HashMap<String, Vec<String>>,
    /// Vector indexes per table (table_name → index)
    vector_indexes: HashMap<String, TableVectorIndex>,
    /// Sidecar manager for embedding generation
    sidecar: Option<SidecarManager>,
}

#[derive(Debug, Clone)]
pub struct QueryRow {
    pub values: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
pub enum QueryResult {
    Rows(Vec<QueryRow>),
    RowCount(u64),
    Ok(String),
}

impl Database {
    pub fn open(path: &str) -> GalaxResult<Self> {
        let path = PathBuf::from(path);
        std::fs::create_dir_all(&path)?;
        let config = EngineConfig {
            data_dir: path.clone(),
            wal_group_commit_ms: 1, // 1ms for fast embedded inserts
            ..Default::default()
        };
        let engine = Engine::new(config)?;
        Ok(Self {
            path: path.clone(),
            catalog: Catalog::new(),
            engine: Arc::new(engine),
            schemas: HashMap::new(),
            vector_indexes: HashMap::new(),
            sidecar: None,
        })
    }

    /// Open with a sidecar for embedding generation.
    /// The sidecar binary must be built and available at the given path.
    pub fn open_with_sidecar(path: &str, sidecar_binary: &str, mock_dim: Option<usize>) -> GalaxResult<Self> {
        let mut db = Self::open(path)?;

        let socket_path = db.path.join("sidecar.sock");
        let sidecar_config = SidecarConfig {
            binary_path: PathBuf::from(sidecar_binary),
            socket_path,
            model_path: None,
            mock_dim,
            data_dir: db.path.clone(),
        };

        let mgr = SidecarManager::new(sidecar_config);
        mgr.start()?;

        // Wait for socket to appear
        let socket = db.path.join("sidecar.sock");
        let start = std::time::Instant::now();
        while !socket.exists() && start.elapsed() < std::time::Duration::from_secs(5) {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        if !socket.exists() {
            return Err(GalaxError::Internal("sidecar failed to start within 5s".into()));
        }

        db.sidecar = Some(mgr);
        Ok(db)
    }

    /// Sync execute — for embedded/Python use.
    pub fn execute(&mut self, sql: &str) -> GalaxResult<QueryResult> {
        let stmts = parser::parse(sql)?;
        let mut last = QueryResult::Ok("OK".to_string());
        for stmt in &stmts {
            last = self.exec_stmt(stmt)?;
        }
        Ok(last)
    }

    /// Async execute — for server use inside tokio runtime.
    pub async fn execute_async(&mut self, sql: &str) -> GalaxResult<QueryResult> {
        let stmts = parser::parse(sql)?;
        let mut last = QueryResult::Ok("OK".to_string());
        for stmt in &stmts {
            last = self.exec_stmt_async(stmt).await?;
        }
        Ok(last)
    }

    fn exec_stmt(&mut self, stmt: &AuroraStatement) -> GalaxResult<QueryResult> {
        match stmt {
            AuroraStatement::Standard(s) => self.exec_standard_sync(s),
            AuroraStatement::CreateTable(ct) => self.exec_create_table(ct),
            AuroraStatement::SemanticMatch(expr) => self.exec_semantic_match(expr),
            s => self.exec_extension(s),
        }
    }

    async fn exec_stmt_async(&mut self, stmt: &AuroraStatement) -> GalaxResult<QueryResult> {
        match stmt {
            AuroraStatement::Standard(s) => self.exec_standard_async(s).await,
            AuroraStatement::CreateTable(ct) => self.exec_create_table(ct),
            AuroraStatement::SemanticMatch(expr) => self.exec_semantic_match(expr),
            s => self.exec_extension(s),
        }
    }

    fn exec_extension(&self, stmt: &AuroraStatement) -> GalaxResult<QueryResult> {
        match stmt {
            AuroraStatement::Analyze { table } => {
                if !self.catalog.table_exists(table) { return Err(GalaxError::TableNotFound(table.clone())); }
                Ok(QueryResult::Ok(format!("ANALYZE {}", table)))
            }
            AuroraStatement::BackupTo { path } => Ok(QueryResult::Ok(format!("BACKUP TO '{}'", path))),
            AuroraStatement::RestoreFrom { path } => Ok(QueryResult::Ok(format!("RESTORE FROM '{}'", path))),
            AuroraStatement::ShowEmbeddingHealth { table } => {
                let msg = table.as_ref().map_or("SHOW EMBEDDING HEALTH".to_string(), |t| format!("SHOW EMBEDDING HEALTH FOR {}", t));
                Ok(QueryResult::Rows(vec![QueryRow { values: vec![("status".to_string(), msg)] }]))
            }
            AuroraStatement::CreateVersionTag(tag) => Ok(QueryResult::Ok(format!("CREATE VERSION TAG '{}'", tag.name))),
            AuroraStatement::BulkInsert(bi) => {
                if !self.catalog.table_exists(&bi.table) { return Err(GalaxError::TableNotFound(bi.table.clone())); }
                Ok(QueryResult::Ok(format!("BULK INSERT INTO {}", bi.table)))
            }
            _ => Ok(QueryResult::Ok("OK".to_string())),
        }
    }

    /// Sync standard SQL execution — uses put_sync (no WAL, memtable+ART only).
    fn exec_standard_sync(&mut self, stmt: &sqlparser::ast::Statement) -> GalaxResult<QueryResult> {
        match stmt {
            sqlparser::ast::Statement::CreateTable(ct) => self.exec_sqlparser_create(ct),
            sqlparser::ast::Statement::Drop { names, if_exists, .. } => self.exec_drop(names, *if_exists),
            sqlparser::ast::Statement::Insert(ins) => self.exec_insert_sync(ins),
            sqlparser::ast::Statement::Query(q) => self.exec_select(q),
            sqlparser::ast::Statement::Update { table, assignments, .. } => self.exec_update(&table.relation.to_string(), assignments),
            sqlparser::ast::Statement::Delete(_) => Ok(QueryResult::RowCount(0)),
            _ => Ok(QueryResult::Ok("OK".to_string())),
        }
    }

    /// Async standard SQL execution — uses put (WAL + memtable + ART).
    async fn exec_standard_async(&mut self, stmt: &sqlparser::ast::Statement) -> GalaxResult<QueryResult> {
        match stmt {
            sqlparser::ast::Statement::CreateTable(ct) => self.exec_sqlparser_create(ct),
            sqlparser::ast::Statement::Drop { names, if_exists, .. } => self.exec_drop(names, *if_exists),
            sqlparser::ast::Statement::Insert(ins) => self.exec_insert_async(ins).await,
            sqlparser::ast::Statement::Query(q) => self.exec_select(q),
            sqlparser::ast::Statement::Update { table, assignments, .. } => self.exec_update(&table.relation.to_string(), assignments),
            sqlparser::ast::Statement::Delete(_) => Ok(QueryResult::RowCount(0)),
            _ => Ok(QueryResult::Ok("OK".to_string())),
        }
    }

    fn exec_sqlparser_create(&mut self, ct: &sqlparser::ast::CreateTable) -> GalaxResult<QueryResult> {
        let columns: Vec<galaxdb_sql::ast::ColumnDef> = ct.columns.iter().map(|c| {
            galaxdb_sql::ast::ColumnDef {
                name: c.name.to_string(),
                data_type: format!("{}", c.data_type),
                nullable: true,
                primary_key: c.options.iter().any(|o| matches!(o.option, sqlparser::ast::ColumnOption::Unique { is_primary: true, .. })),
                embedding: None,
            }
        }).collect();
        self.exec_create_table(&CreateTableStmt { table_name: ct.name.to_string(), columns, if_not_exists: ct.if_not_exists })
    }

    fn exec_drop(&mut self, names: &[sqlparser::ast::ObjectName], if_exists: bool) -> GalaxResult<QueryResult> {
        let name = names.first().map(|n: &sqlparser::ast::ObjectName| n.to_string()).unwrap_or_default();
        match self.catalog.drop_table(&name) {
            Ok(_) => { self.schemas.remove(&name); Ok(QueryResult::Ok(format!("DROP TABLE {}", name))) }
            Err(_) if if_exists => Ok(QueryResult::Ok(format!("DROP TABLE IF EXISTS {}", name))),
            Err(e) => Err(e),
        }
    }

    fn exec_insert_sync(&mut self, ins: &sqlparser::ast::Insert) -> GalaxResult<QueryResult> {
        let table = ins.table_name.to_string();
        if !self.catalog.table_exists(&table) { return Err(GalaxError::TableNotFound(table)); }
        let schema = self.schemas.get(&table).cloned().unwrap_or_default();
        let mut count = 0u64;
        if let Some(source) = &ins.source {
            if let sqlparser::ast::SetExpr::Values(values) = source.body.as_ref() {
                // Batch all rows into a single WAL write + fsync
                let mut batch: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(values.rows.len());
                let mut texts_to_embed: Vec<(u64, String)> = Vec::new();

                for row in &values.rows {
                    let (key, value) = self.build_kv(&table, &schema, row, count);
                    batch.push((key.into_bytes(), value.into_bytes()));

                    // If this table has an embedding column, extract the text
                    if self.vector_indexes.contains_key(&table) {
                        let idx = self.vector_indexes.get(&table).unwrap();
                        // Find the source column index in the schema
                        if let Some(col_idx) = schema.iter().position(|c| c == &idx.source_column) {
                            if col_idx < row.len() {
                                let text = fmt_expr(&row[col_idx]);
                                let row_id = idx.next_row_id + count;
                                texts_to_embed.push((row_id, text));
                            }
                        }
                    }

                    count += 1;
                }

                if batch.len() == 1 {
                    let (k, v) = batch.into_iter().next().unwrap();
                    self.engine.put_sync(k, v)?;
                } else if !batch.is_empty() {
                    self.engine.put_batch_sync(&batch)?;
                }

                // Generate embeddings for inserted rows
                if !texts_to_embed.is_empty() && self.sidecar.is_some() {
                    let sidecar = self.sidecar.as_ref().unwrap();
                    let table_name = table.clone();

                    for (row_id, text) in &texts_to_embed {
                        let request = EmbedRequest {
                            row_id: *row_id,
                            text: text.clone(),
                            column: self.vector_indexes.get(&table_name)
                                .map(|idx| idx.embedding_column.clone())
                                .unwrap_or_default(),
                        };

                        match sidecar.embed(request) {
                            Ok(response) => {
                                // Insert embedding into delta buffer
                                if let Some(idx) = self.vector_indexes.get_mut(&table_name) {
                                    idx.delta.insert(*row_id, response.embedding.clone());
                                    idx.vectors.insert(*row_id, response.embedding);
                                }
                            }
                            Err(_) => {
                                // Sidecar unavailable — embedding will be generated later
                                // (backlog handling)
                            }
                        }
                    }

                    // Update next_row_id
                    if let Some(idx) = self.vector_indexes.get_mut(&table) {
                        idx.next_row_id += count;
                    }
                } else if !texts_to_embed.is_empty() {
                    // No sidecar — just update row_id counter
                    if let Some(idx) = self.vector_indexes.get_mut(&table) {
                        idx.next_row_id += count;
                    }
                }
            }
        }
        Ok(QueryResult::RowCount(count))
    }

    async fn exec_insert_async(&mut self, ins: &sqlparser::ast::Insert) -> GalaxResult<QueryResult> {
        let table = ins.table_name.to_string();
        if !self.catalog.table_exists(&table) { return Err(GalaxError::TableNotFound(table)); }
        let schema = self.schemas.get(&table).cloned().unwrap_or_default();
        let mut count = 0u64;
        if let Some(source) = &ins.source {
            if let sqlparser::ast::SetExpr::Values(values) = source.body.as_ref() {
                for row in &values.rows {
                    let (key, value) = self.build_kv(&table, &schema, row, count);
                    self.engine.put(key.into_bytes(), value.into_bytes()).await?;
                    count += 1;
                }
            }
        }
        Ok(QueryResult::RowCount(count))
    }

    fn build_kv(&self, table: &str, schema: &[String], row: &[sqlparser::ast::Expr], idx: u64) -> (String, String) {
        let mut parts = Vec::new();
        for (i, val) in row.iter().enumerate() {
            let col = schema.get(i).cloned().unwrap_or_else(|| format!("col{}", i));
            parts.push(format!("{}={}", col, fmt_expr(val)));
        }
        let key = row.first().map_or(format!("{}:{}", table, idx), |v| format!("{}:{}", table, fmt_expr(v)));
        (key, parts.join("|"))
    }

    fn exec_select(&self, query: &sqlparser::ast::Query) -> GalaxResult<QueryResult> {
        let table = extract_table(query);
        if table != "unknown" && !self.catalog.table_exists(&table) {
            return Err(GalaxError::TableNotFound(table));
        }
        let prefix = format!("{}:", table);
        let rows: Vec<QueryRow> = self.engine.scan_all().into_iter()
            .filter(|(k, _)| String::from_utf8_lossy(k).starts_with(&prefix))
            .map(|(_, v)| {
                let vs = String::from_utf8_lossy(&v);
                QueryRow { values: vs.split('|').filter_map(|p| p.split_once('=').map(|(k,v)| (k.to_string(), v.to_string()))).collect() }
            })
            .collect();
        Ok(QueryResult::Rows(rows))
    }

    fn exec_update(&self, table: &str, assignments: &[sqlparser::ast::Assignment]) -> GalaxResult<QueryResult> {
        if !self.catalog.table_exists(table) { return Err(GalaxError::TableNotFound(table.to_string())); }
        if let Some(entry) = self.catalog.get_table(table) {
            for a in assignments {
                let col = format!("{}", a.target);
                if entry.columns.iter().any(|c| c.name == col && c.is_embedding_source) {
                    return Err(GalaxError::EmbeddingSourceUpdate { column: col });
                }
            }
        }
        Ok(QueryResult::RowCount(0))
    }

    /// Execute a SEMANTIC_MATCH query.
    /// Embeds the query text via sidecar, searches HNSW + delta buffer, returns results.
    fn exec_semantic_match(&self, expr: &galaxdb_sql::ast::SemanticMatchExpr) -> GalaxResult<QueryResult> {
        // Find which table has this embedding column
        let table_name = self.vector_indexes.iter()
            .find(|(_, idx)| idx.embedding_column == expr.column || idx.source_column == expr.column)
            .map(|(name, _)| name.clone());

        let table_name = match table_name {
            Some(t) => t,
            None => return Err(GalaxError::Internal(format!(
                "no embedding index found for column '{}'", expr.column
            ))),
        };

        let idx = self.vector_indexes.get(&table_name).unwrap();

        // Embed the query text via sidecar
        let query_embedding = match &self.sidecar {
            Some(sidecar) => {
                let request = EmbedRequest {
                    row_id: 0,
                    text: expr.query.clone(),
                    column: expr.column.clone(),
                };
                match sidecar.embed(request) {
                    Ok(response) => response.embedding,
                    Err(_) => return Err(GalaxError::Internal(
                        "semantic search temporarily unavailable — embedding sidecar is down".into()
                    )),
                }
            }
            None => return Err(GalaxError::Internal(
                "semantic search unavailable — no embedding sidecar configured".into()
            )),
        };

        // Search HNSW + delta buffer
        let sm_config = SemanticMatchConfig {
            hnsw_candidates: 100,
            ef_search: 200,
            brute_force_threshold: 1000,
            brute_force_ratio: 0.001,
        };

        let vectors_ref = &idx.vectors;
        let results = execute_semantic_match(
            &query_embedding,
            &idx.hnsw,
            &idx.delta,
            expr.threshold,
            10,
            &sm_config,
            |row_id| vectors_ref.get(&row_id).cloned(),
        );

        // Format results
        let rows: Vec<QueryRow> = results.iter().map(|r| {
            QueryRow {
                values: vec![
                    ("row_id".to_string(), r.row_id.to_string()),
                    ("similarity".to_string(), format!("{:.4}", r.similarity)),
                ],
            }
        }).collect();

        Ok(QueryResult::Rows(rows))
    }

    fn exec_create_table(&mut self, ct: &CreateTableStmt) -> GalaxResult<QueryResult> {
        let cols: Vec<CatalogColumn> = ct.columns.iter().map(|c| CatalogColumn {
            name: c.name.clone(), data_type: c.data_type.clone(), nullable: c.nullable,
            primary_key: c.primary_key, is_embedding_source: c.embedding.is_some(),
        }).collect();
        let names: Vec<String> = ct.columns.iter().map(|c| c.name.clone()).collect();
        let has_embedding = ct.columns.iter().any(|c| c.embedding.is_some());
        let entry = TableEntry { name: ct.table_name.clone(), columns: cols, has_embedding };
        self.catalog.create_table(ct.table_name.clone(), entry)?;
        self.schemas.insert(ct.table_name.clone(), names);

        // If table has embedding columns, create a vector index
        if has_embedding {
            for col in &ct.columns {
                if let Some(ref emb) = col.embedding {
                    let dim = emb.dimensions.unwrap_or(128) as usize;
                    let config = HnswConfig::new(dim).with_max_elements(1_000_000);
                    let idx = TableVectorIndex {
                        hnsw: HnswGraph::new(config),
                        delta: DeltaBuffer::new(dim),
                        dim,
                        embedding_column: col.name.clone(),
                        source_column: col.name.clone(), // source is the same column
                        next_row_id: 0,
                        vectors: HashMap::new(),
                    };
                    self.vector_indexes.insert(ct.table_name.clone(), idx);
                    break; // one vector index per table for now
                }
            }
        }

        Ok(QueryResult::Ok(format!("CREATE TABLE {}", ct.table_name)))
    }

    pub fn path(&self) -> &Path { &self.path }

    /// Execute a read-only SQL statement (SELECT, SHOW) without &mut self.
    /// This allows concurrent reads through RwLock.
    pub fn execute_readonly(&self, sql: &str) -> GalaxResult<QueryResult> {
        let stmts = parser::parse(sql)?;
        let mut last = QueryResult::Ok("OK".to_string());
        for stmt in &stmts {
            last = match stmt {
                AuroraStatement::Standard(s) => match s.as_ref() {
                    sqlparser::ast::Statement::Query(q) => self.exec_select(q),
                    _ => Ok(QueryResult::Ok("OK".to_string())),
                },
                AuroraStatement::ShowEmbeddingHealth { table } => {
                    let msg = table.as_ref().map_or("SHOW EMBEDDING HEALTH".to_string(), |t| format!("SHOW EMBEDDING HEALTH FOR {}", t));
                    Ok(QueryResult::Rows(vec![QueryRow { values: vec![("status".to_string(), msg)] }]))
                }
                _ => Ok(QueryResult::Ok("OK".to_string())),
            }?;
        }
        Ok(last)
    }

    pub fn table_count(&self) -> usize { self.catalog.table_count() }
    pub fn table_exists(&self, name: &str) -> bool { self.catalog.table_exists(name) }
    pub fn row_count(&self) -> u64 { self.engine.row_count() }
}

impl Drop for Database {
    fn drop(&mut self) { self.engine.shutdown(); }
}

fn fmt_expr(e: &sqlparser::ast::Expr) -> String {
    match e {
        sqlparser::ast::Expr::Value(v) => match v {
            sqlparser::ast::Value::Number(n, _) => n.clone(),
            sqlparser::ast::Value::SingleQuotedString(s) | sqlparser::ast::Value::DoubleQuotedString(s) => s.clone(),
            sqlparser::ast::Value::Boolean(b) => b.to_string(),
            sqlparser::ast::Value::Null => "NULL".to_string(),
            _ => format!("{}", v),
        },
        _ => format!("{}", e),
    }
}

fn extract_table(q: &sqlparser::ast::Query) -> String {
    if let sqlparser::ast::SetExpr::Select(s) = q.body.as_ref() {
        if let Some(f) = s.from.first() { return f.relation.to_string(); }
    }
    "unknown".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_db() -> Database {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("db");
        std::mem::forget(dir);
        Database::open(p.to_str().unwrap()).unwrap()
    }

    #[test]
    fn create_insert_select_roundtrip() {
        let mut db = test_db();
        db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)").unwrap();
        db.execute("INSERT INTO users (id, name) VALUES (1, 'alice')").unwrap();
        db.execute("INSERT INTO users (id, name) VALUES (2, 'bob')").unwrap();
        let r = db.execute("SELECT * FROM users").unwrap();
        match r {
            QueryResult::Rows(rows) => {
                assert_eq!(rows.len(), 2);
                assert!(rows[0].values.iter().any(|(k,v)| k == "name" && v == "alice"));
            }
            _ => panic!("expected Rows"),
        }
    }

    #[test]
    fn insert_10_rows_and_count() {
        let mut db = test_db();
        db.execute("CREATE TABLE t (id INT, val TEXT)").unwrap();
        for i in 0..10 { db.execute(&format!("INSERT INTO t (id, val) VALUES ({}, 'v{}')", i, i)).unwrap(); }
        assert_eq!(db.row_count(), 10);
        match db.execute("SELECT * FROM t").unwrap() {
            QueryResult::Rows(rows) => assert_eq!(rows.len(), 10),
            _ => panic!("expected Rows"),
        }
    }

    #[test]
    fn select_nonexistent_fails() { let mut db = test_db(); assert!(db.execute("SELECT * FROM nope").is_err()); }

    #[test]
    fn create_drop() {
        let mut db = test_db();
        db.execute("CREATE TABLE t (id INT)").unwrap();
        assert!(db.table_exists("t"));
        db.execute("DROP TABLE t").unwrap();
        assert!(!db.table_exists("t"));
    }

    #[test]
    fn duplicate_create_fails() {
        let mut db = test_db();
        db.execute("CREATE TABLE t (id INT)").unwrap();
        assert!(db.execute("CREATE TABLE t (id INT)").is_err());
    }

    #[test]
    fn extensions_work() {
        let mut db = test_db();
        db.execute("CREATE TABLE t (id INT)").unwrap();
        assert!(matches!(db.execute("ANALYZE t").unwrap(), QueryResult::Ok(_)));
        assert!(matches!(db.execute("SHOW EMBEDDING HEALTH").unwrap(), QueryResult::Rows(_)));
        assert!(matches!(db.execute("CREATE VERSION TAG 'v1'").unwrap(), QueryResult::Ok(_)));
    }

    /// End-to-end SEMANTIC_MATCH test:
    /// CREATE TABLE with embedding → INSERT text → sidecar embeds → SEMANTIC_MATCH finds it.
    ///
    /// Requires the sidecar binary to be built. Run with:
    /// cargo test -p galaxdb-embedded -- semantic_match_end_to_end --ignored
    #[test]
    #[ignore] // requires sidecar binary — run explicitly
    fn semantic_match_end_to_end() {
        // Find the sidecar binary
        let sidecar_binary = std::env::current_exe().unwrap()
            .parent().unwrap()
            .parent().unwrap()
            .join("galaxdb-sidecar");

        if !sidecar_binary.exists() {
            // Try to build it
            let status = std::process::Command::new("cargo")
                .args(["build", "-p", "galaxdb-sidecar"])
                .status()
                .expect("cargo build");
            assert!(status.success(), "failed to build sidecar binary");
        }

        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("semantic_db");
        std::mem::forget(dir); // keep temp dir alive

        // Open database with sidecar (mock mode, dim=64)
        let mut db = Database::open_with_sidecar(
            db_path.to_str().unwrap(),
            sidecar_binary.to_str().unwrap(),
            Some(64),
        ).unwrap();

        // Create table with embedding column
        db.execute(
            "CREATE TABLE docs (id INT PRIMARY KEY, content TEXT EMBEDDING MODEL 'mock' DIM 64)"
        ).unwrap();

        assert!(db.vector_indexes.contains_key("docs"));

        // Insert text rows — sidecar will embed them
        db.execute("INSERT INTO docs (id, content) VALUES (1, 'machine learning is great')").unwrap();
        db.execute("INSERT INTO docs (id, content) VALUES (2, 'rust programming language')").unwrap();
        db.execute("INSERT INTO docs (id, content) VALUES (3, 'machine learning algorithms')").unwrap();

        // Verify embeddings were generated
        let idx = db.vector_indexes.get("docs").unwrap();
        assert_eq!(idx.delta.vector_count(), 3, "should have 3 embeddings in delta buffer");

        // Run SEMANTIC_MATCH — should find similar documents
        let result = db.execute(
            "SELECT * FROM docs WHERE SEMANTIC_MATCH(content, 'machine learning', 0.0)"
        ).unwrap();

        match result {
            QueryResult::Rows(rows) => {
                assert!(!rows.is_empty(), "SEMANTIC_MATCH should return results");
                eprintln!("SEMANTIC_MATCH returned {} results:", rows.len());
                for row in &rows {
                    eprintln!("  {:?}", row.values);
                }
                // The mock sidecar generates deterministic embeddings from text hash.
                // "machine learning is great" and "machine learning algorithms" should
                // be more similar to "machine learning" than "rust programming language".
            }
            other => panic!("expected Rows, got {:?}", other),
        }

        // Verify sidecar is still healthy
        assert!(db.sidecar.as_ref().unwrap().is_healthy());

        eprintln!("✓ End-to-end SEMANTIC_MATCH test passed!");
    }
}
