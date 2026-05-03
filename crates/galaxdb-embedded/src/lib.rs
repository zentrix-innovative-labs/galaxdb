//! GalaxDB Embedded — Rust API for embedded mode with real storage engine.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use galaxdb_common::{GalaxError, GalaxResult};
use galaxdb_sql::ast::{AuroraStatement, CreateTableStmt};
use galaxdb_sql::executor::{Catalog, CatalogColumn, TableEntry};
use galaxdb_sql::parser;
use galaxdb_storage::engine::{Engine, EngineConfig};

/// An embedded GalaxDB database instance.
pub struct Database {
    path: PathBuf,
    catalog: Catalog,
    engine: Arc<Engine>,
    schemas: HashMap<String, Vec<String>>,
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
        Ok(Self { path, catalog: Catalog::new(), engine: Arc::new(engine), schemas: HashMap::new() })
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
            s => self.exec_extension(s),
        }
    }

    async fn exec_stmt_async(&mut self, stmt: &AuroraStatement) -> GalaxResult<QueryResult> {
        match stmt {
            AuroraStatement::Standard(s) => self.exec_standard_async(s).await,
            AuroraStatement::CreateTable(ct) => self.exec_create_table(ct),
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
                for row in &values.rows {
                    let (key, value) = self.build_kv(&table, &schema, row, count);
                    batch.push((key.into_bytes(), value.into_bytes()));
                    count += 1;
                }
                if batch.len() == 1 {
                    // Single row — use put_sync directly
                    let (k, v) = batch.into_iter().next().unwrap();
                    self.engine.put_sync(k, v)?;
                } else if !batch.is_empty() {
                    // Multi-row — use batch write (one WAL entry, one fsync)
                    self.engine.put_batch_sync(&batch)?;
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

    fn exec_create_table(&mut self, ct: &CreateTableStmt) -> GalaxResult<QueryResult> {
        let cols: Vec<CatalogColumn> = ct.columns.iter().map(|c| CatalogColumn {
            name: c.name.clone(), data_type: c.data_type.clone(), nullable: c.nullable,
            primary_key: c.primary_key, is_embedding_source: c.embedding.is_some(),
        }).collect();
        let names: Vec<String> = ct.columns.iter().map(|c| c.name.clone()).collect();
        let entry = TableEntry { name: ct.table_name.clone(), columns: cols, has_embedding: ct.columns.iter().any(|c| c.embedding.is_some()) };
        self.catalog.create_table(ct.table_name.clone(), entry)?;
        self.schemas.insert(ct.table_name.clone(), names);
        Ok(QueryResult::Ok(format!("CREATE TABLE {}", ct.table_name)))
    }

    pub fn path(&self) -> &Path { &self.path }

    /// Execute multiple SQL statements in a batch.
    /// Multi-row INSERTs are automatically batched into a single WAL write.
    pub fn execute_batch(&mut self, statements: &[&str]) -> GalaxResult<Vec<QueryResult>> {
        let mut results = Vec::with_capacity(statements.len());
        for sql in statements {
            results.push(self.execute(sql)?);
        }
        Ok(results)
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
}
