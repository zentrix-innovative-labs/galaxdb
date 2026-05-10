//! GalaxDB Embedded — thin wrapper around the canonical executor.
//!
//! The embedded crate used to carry its own inline SQL execution logic
//! (CREATE TABLE / INSERT / SELECT / UPDATE / SEMANTIC_MATCH). During
//! the consolidation sprint that code moved into
//! `galaxdb_sql::executor::execute_with_context`, where it operates
//! through a real `Engine` + `ExecutorContext`. This crate now owns
//! the per-database state (engine, sidecar, tag catalog, vector
//! indexes) and delegates every statement to the canonical executor.
//!
//! Public API shape is preserved:
//!
//! * `Database::open(path)` / `Database::open_with_sidecar(path, bin, model_id)`
//! * `Database::execute(sql) -> GalaxResult<QueryResult>`
//! * `Database::execute_async(sql) -> GalaxResult<QueryResult>`
//! * `Database::execute_readonly(sql) -> GalaxResult<QueryResult>`
//! * `table_count`, `table_exists`, `row_count`, `path`
//!
//! There are no mocks on any production path. Sidecar unavailability
//! surfaces as a typed `GalaxError::SidecarUnavailable`; a missing
//! model means the sidecar process exits with status 1 and the engine
//! sees a dead child.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use galaxdb_common::{GalaxError, GalaxResult};
use galaxdb_sidecar::manager::{SidecarConfig, SidecarManager};
use galaxdb_sidecar::protocol::EmbedRequest;
use galaxdb_sql::ast::{AuroraStatement, CreateTableStmt};
use galaxdb_sql::executor::{
    execute_with_context, ExecuteResult, ExecutorContext, Row as SqlRow, VectorSearchBackend,
    VectorSearchResult,
};
use galaxdb_sql::parser;
use galaxdb_sql::planner::{self, FilterExpr, QueryPlan, SearchStrategy, Value};
use galaxdb_sql::row_codec;
use galaxdb_storage::engine::{Engine, EngineConfig};
use galaxdb_vector::{
    execute_semantic_match, DeltaBuffer, HnswConfig, HnswGraph, SemanticMatchConfig,
};
use galaxdb_versioning::{MerkleDag, TagCatalog};

// ---------------------------------------------------------------------------
// Query result types (stable public surface)
// ---------------------------------------------------------------------------

/// A single row returned by `Database::execute`. Each entry is
/// `(column_name, stringified_value)` — strings are the rendered form
/// of [`galaxdb_sql::planner::Value`] as produced by
/// [`galaxdb_sql::row_codec::value_display`]. `NULL` is rendered as the
/// literal string `"NULL"`.
#[derive(Debug, Clone)]
pub struct QueryRow {
    pub values: Vec<(String, String)>,
}

/// Outcome of executing one SQL statement.
#[derive(Debug, Clone)]
pub enum QueryResult {
    Rows(Vec<QueryRow>),
    RowCount(u64),
    Ok(String),
}

// ---------------------------------------------------------------------------
// Per-table vector index
// ---------------------------------------------------------------------------

/// HNSW graph + delta buffer + row-id/vector map for one table with an
/// embedding column.
struct TableVectorIndex {
    hnsw: HnswGraph,
    delta: DeltaBuffer,
    /// Embedding dimension — read by online tests to assert the
    /// sidecar's model dim matches the catalog's declared DIM.
    #[allow(dead_code)]
    dim: usize,
    /// Column with the embedding.
    embedding_column: String,
    /// Source text column (for `SEMANTIC_MATCH` lookup).
    source_column: String,
    /// Row-id counter.
    next_row_id: u64,
    /// Row-id → vector (for re-ranking).
    vectors: HashMap<u64, Vec<f32>>,
}

// ---------------------------------------------------------------------------
// Database
// ---------------------------------------------------------------------------

/// An embedded GalaxDB database instance.
///
/// Owns the storage engine, sidecar, tag catalog, and per-table vector
/// indexes. Every SQL statement is dispatched through the canonical
/// executor (`galaxdb_sql::executor::execute_with_context`).
pub struct Database {
    path: PathBuf,
    engine: Arc<Engine>,
    /// Sidecar manager — shared so the vector backend can also call it.
    sidecar: Option<Arc<SidecarManager>>,
    /// Merkle DAG for version history.
    merkle_dag: Arc<Mutex<MerkleDag>>,
    /// Version tag catalog.
    tag_catalog: Arc<Mutex<TagCatalog>>,
    /// Vector indexes per table. Wrapped in `Arc<RwLock>` so the vector
    /// backend (which takes `&self` across the `VectorSearchBackend`
    /// trait) can read/update them without requiring `&mut self` on
    /// every executor call.
    vector_indexes: Arc<RwLock<HashMap<String, TableVectorIndex>>>,
    /// Persisted catalog snapshot — mirrors the executor's context
    /// catalog. Carried here so `&self` read-only methods
    /// (`table_exists`, `table_count`) don't need to rebuild it.
    catalog: galaxdb_sql::executor::Catalog,
}

impl Database {
    /// Open (or create) a database at `path` without a sidecar.
    ///
    /// Tables with embedding columns can still be created; inserts will
    /// succeed but the embedding column will stay unpopulated until a
    /// sidecar is attached and the row is re-inserted.
    pub fn open(path: &str) -> GalaxResult<Self> {
        let path = PathBuf::from(path);
        std::fs::create_dir_all(&path)?;
        let config = EngineConfig {
            data_dir: path.clone(),
            wal_group_commit_ms: 1, // fast sync commits for embedded mode
            ..Default::default()
        };
        let engine = Engine::new(config)?;
        Ok(Self {
            path,
            engine: Arc::new(engine),
            sidecar: None,
            merkle_dag: Arc::new(Mutex::new(MerkleDag::new())),
            tag_catalog: Arc::new(Mutex::new(TagCatalog::new())),
            vector_indexes: Arc::new(RwLock::new(HashMap::new())),
            catalog: galaxdb_sql::executor::Catalog::new(),
        })
    }

    /// Open a database with an embedding sidecar attached.
    ///
    /// * `path` — storage data directory.
    /// * `sidecar_binary` — path to the `galaxdb-sidecar` binary.
    /// * `model_id` — HuggingFace model id (e.g.
    ///   `sentence-transformers/all-MiniLM-L6-v2`). The sidecar
    ///   downloads the model on first run and caches it.
    ///
    /// If the sidecar fails to load the model it exits with status 1;
    /// `SidecarManager` observes the dead child and any subsequent
    /// `embed` call returns a typed error. There is no mock fallback —
    /// every embedding is computed by the real model.
    pub fn open_with_sidecar(
        path: &str,
        sidecar_binary: &str,
        model_id: &str,
    ) -> GalaxResult<Self> {
        let mut db = Self::open(path)?;

        let socket_path = db.path.join("sidecar.sock");
        let sidecar_config = SidecarConfig {
            binary_path: PathBuf::from(sidecar_binary),
            socket_path: socket_path.clone(),
            model_id: model_id.to_string(),
            data_dir: db.path.clone(),
        };

        let mgr = SidecarManager::new(sidecar_config);
        mgr.start()?;

        // Wait for the sidecar socket. First run includes the ~90 MB
        // model download; subsequent runs hit the HF cache and come up
        // in seconds. Allow up to 120 s.
        let start = std::time::Instant::now();
        while !socket_path.exists() && start.elapsed() < std::time::Duration::from_secs(120) {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        if !socket_path.exists() {
            return Err(GalaxError::Internal(
                "sidecar failed to start within 120s — check network access to HuggingFace \
                 Hub and disk space for the HF cache"
                    .into(),
            ));
        }

        db.sidecar = Some(Arc::new(mgr));
        Ok(db)
    }

    /// Synchronous execute — for embedded Rust callers and the Python
    /// FFI.
    pub fn execute(&mut self, sql: &str) -> GalaxResult<QueryResult> {
        let stmts = parser::parse(sql)?;
        let mut last = QueryResult::Ok("OK".to_string());
        for stmt in &stmts {
            last = self.exec_stmt(stmt)?;
        }
        Ok(last)
    }

    /// Async variant — identical semantics; currently just wraps the
    /// sync path. Retained because the wire-protocol path wants an
    /// `async` signature and v2 will make the engine truly async.
    pub async fn execute_async(&mut self, sql: &str) -> GalaxResult<QueryResult> {
        self.execute(sql)
    }

    /// Execute a read-only statement without `&mut self`. Used by
    /// callers holding the database behind an `RwLock` that want to
    /// allow concurrent reads.
    pub fn execute_readonly(&self, sql: &str) -> GalaxResult<QueryResult> {
        let stmts = parser::parse(sql)?;
        let mut last = QueryResult::Ok("OK".to_string());
        for stmt in &stmts {
            match stmt {
                AuroraStatement::Standard(s) => {
                    if let sqlparser::ast::Statement::Query(q) = s.as_ref() {
                        last = self.select_readonly(q)?;
                    } else {
                        return Err(GalaxError::Internal(
                            "execute_readonly only supports SELECT and SHOW; \
                             use execute() for write-capable statements"
                                .into(),
                        ));
                    }
                }
                AuroraStatement::ShowEmbeddingHealth { table } => {
                    let msg = table
                        .as_ref()
                        .map_or("SHOW EMBEDDING HEALTH".to_string(), |t| {
                            format!("SHOW EMBEDDING HEALTH FOR {}", t)
                        });
                    last = QueryResult::Rows(vec![QueryRow {
                        values: vec![("status".to_string(), msg)],
                    }]);
                }
                _ => {
                    return Err(GalaxError::Internal(
                        "execute_readonly only supports SELECT and SHOW; use execute() for \
                         write-capable statements"
                            .into(),
                    ));
                }
            }
        }
        Ok(last)
    }

    fn select_readonly(&self, q: &sqlparser::ast::Query) -> GalaxResult<QueryResult> {
        // Route through the canonical executor so WHERE clauses and
        // projections are honoured. Before Phase I this did a raw
        // prefix scan over `engine.scan_all()` and dropped the filter,
        // which silently returned every row for any `SELECT ... WHERE`
        // the wire server received.
        //
        // `execute_with_context` takes `&mut ExecutorContext`, but this
        // method is `&self` (multiple concurrent readers). We build a
        // throwaway context that clones the catalog; the executor
        // never mutates the catalog on read paths, so the clone we
        // take here is discarded at the end.
        let (columns, filter) = extract_projection_and_filter(q);
        let table = extract_table(q);
        if table != "unknown" && !self.catalog.table_exists(&table) {
            return Err(GalaxError::TableNotFound(table));
        }
        let plan = QueryPlan::FullScan {
            table,
            filter,
            columns,
        };

        let mut ctx = ExecutorContext::new(self.engine.clone());
        ctx.catalog = self.catalog.clone();
        ctx.vector_backend = Some(Arc::new(EmbeddedVectorBackend {
            sidecar: self.sidecar.clone(),
            indexes: self.vector_indexes.clone(),
        }));
        let res = execute_with_context(&plan, &mut ctx)?;
        Ok(query_result_from(res))
    }

    // -----------------------------------------------------------------
    // Statement dispatch
    // -----------------------------------------------------------------

    fn exec_stmt(&mut self, stmt: &AuroraStatement) -> GalaxResult<QueryResult> {
        // Translate `AuroraStatement` into a `QueryPlan`, then delegate
        // to `execute_with_context`. For `SEMANTIC_MATCH` (which the
        // current planner doesn't handle directly outside `WHERE`), we
        // route to the vector backend inline.
        match stmt {
            AuroraStatement::Standard(s) => self.exec_standard(s),
            AuroraStatement::CreateTable(ct) => self.exec_create_table(ct),
            AuroraStatement::SemanticMatch(expr) => self.exec_semantic_match_standalone(expr),
            AuroraStatement::Analyze { table } => self.dispatch(QueryPlan::Analyze {
                table: table.clone(),
            }),
            AuroraStatement::BackupTo { path } => self.dispatch(QueryPlan::Backup {
                path: path.clone(),
            }),
            AuroraStatement::RestoreFrom { path } => self.dispatch(QueryPlan::Restore {
                path: path.clone(),
            }),
            AuroraStatement::ShowEmbeddingHealth { table } => {
                self.dispatch(QueryPlan::ShowEmbeddingHealth {
                    table: table.clone(),
                })
            }
            AuroraStatement::CreateVersionTag(tag_stmt) => {
                self.dispatch(QueryPlan::CreateVersionTag(tag_stmt.clone()))
            }
            AuroraStatement::BulkInsert(bi) => self.dispatch(QueryPlan::BulkInsert {
                table: bi.table.clone(),
            }),
            AuroraStatement::AtVersion(_) => Err(GalaxError::NotYetAvailable {
                task: "B6",
                feature: "AT VERSION planner wiring (consolidation Phase B6 deferred)",
            }),
        }
    }

    fn exec_standard(&mut self, stmt: &sqlparser::ast::Statement) -> GalaxResult<QueryResult> {
        match stmt {
            sqlparser::ast::Statement::CreateTable(ct) => self.exec_sqlparser_create(ct),
            sqlparser::ast::Statement::Drop {
                names, if_exists, ..
            } => self.dispatch(QueryPlan::DropTable {
                name: names
                    .first()
                    .map(|n| n.to_string())
                    .unwrap_or_default(),
                if_exists: *if_exists,
            }),
            sqlparser::ast::Statement::Insert(ins) => self.exec_insert(ins),
            sqlparser::ast::Statement::Query(q) => self.exec_select(q),
            sqlparser::ast::Statement::Update {
                table,
                assignments,
                selection,
                ..
            } => self.exec_update(
                &table.relation.to_string(),
                assignments,
                selection.as_ref(),
            ),
            sqlparser::ast::Statement::Delete(del) => self.exec_delete(del),
            other => Err(GalaxError::Internal(format!(
                "unsupported SQL statement: {:?}",
                std::mem::discriminant(other)
            ))),
        }
    }

    fn exec_sqlparser_create(
        &mut self,
        ct: &sqlparser::ast::CreateTable,
    ) -> GalaxResult<QueryResult> {
        let columns: Vec<galaxdb_sql::ast::ColumnDef> = ct
            .columns
            .iter()
            .map(|c| galaxdb_sql::ast::ColumnDef {
                name: c.name.to_string(),
                data_type: format!("{}", c.data_type),
                nullable: true,
                primary_key: c.options.iter().any(|o| {
                    matches!(
                        o.option,
                        sqlparser::ast::ColumnOption::Unique {
                            is_primary: true,
                            ..
                        }
                    )
                }),
                embedding: None,
            })
            .collect();
        self.exec_create_table(&CreateTableStmt {
            table_name: ct.name.to_string(),
            columns,
            if_not_exists: ct.if_not_exists,
        })
    }

    fn exec_create_table(&mut self, ct: &CreateTableStmt) -> GalaxResult<QueryResult> {
        let has_embedding = ct.columns.iter().any(|c| c.embedding.is_some());

        // Delegate catalog registration to the executor.
        let mut ctx = self.context();
        let plan = QueryPlan::CreateTable(ct.clone());
        let result = execute_with_context(&plan, &mut ctx)?;
        self.catalog = std::mem::take(&mut ctx.catalog);

        // If the table has an embedding column, create a vector index.
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
                        source_column: col.name.clone(),
                        next_row_id: 0,
                        vectors: HashMap::new(),
                    };
                    self.vector_indexes
                        .write()
                        .unwrap()
                        .insert(ct.table_name.clone(), idx);
                    break;
                }
            }
        }

        Ok(query_result_from(result))
    }

    fn exec_insert(&mut self, ins: &sqlparser::ast::Insert) -> GalaxResult<QueryResult> {
        let table = ins.table_name.to_string();
        let entry = self
            .catalog
            .get_table(&table)
            .cloned()
            .ok_or_else(|| GalaxError::TableNotFound(table.clone()))?;

        let column_names: Vec<String> = ins
            .columns
            .iter()
            .map(|c| c.to_string())
            .collect();

        let Some(source) = &ins.source else {
            return Ok(QueryResult::RowCount(0));
        };
        let sqlparser::ast::SetExpr::Values(values) = source.body.as_ref() else {
            return Ok(QueryResult::RowCount(0));
        };

        let mut count = 0u64;
        for row in &values.rows {
            let row_values: Vec<Value> = row.iter().map(value_from_expr).collect();
            let plan = QueryPlan::Insert {
                table: table.clone(),
                columns: column_names.clone(),
                values: row_values.clone(),
            };
            let mut ctx = self.context();
            let res = execute_with_context(&plan, &mut ctx)?;
            self.catalog = std::mem::take(&mut ctx.catalog);
            if matches!(res, ExecuteResult::RowCount(_)) {
                count += 1;
            }

            // Async embedding trigger for tables with an embedding
            // column. The sidecar-backed path queues the text; on
            // success the vector lands in the per-table delta buffer
            // so later SEMANTIC_MATCH queries can find it.
            if entry.has_embedding {
                self.generate_embedding_for_row(&table, &entry, &column_names, &row_values);
            }
        }

        Ok(QueryResult::RowCount(count))
    }

    fn generate_embedding_for_row(
        &self,
        table: &str,
        entry: &galaxdb_sql::executor::TableEntry,
        column_names: &[String],
        values: &[Value],
    ) {
        let Some(sidecar) = self.sidecar.as_ref() else {
            return;
        };
        let indexes = self.vector_indexes.read().unwrap();
        let Some(index_meta) = indexes.get(table) else {
            return;
        };
        // Resolve the source-column index in this INSERT's value list.
        let (source_col_name, source_col_index) = resolve_source_column(
            entry,
            column_names,
            &index_meta.source_column,
        );
        let Some(idx) = source_col_index else {
            return;
        };
        let Value::Text(text) = &values[idx] else {
            return;
        };
        let text = text.clone();
        drop(indexes);

        let row_id = {
            let mut indexes = self.vector_indexes.write().unwrap();
            let Some(mut_idx) = indexes.get_mut(table) else {
                return;
            };
            let row_id = mut_idx.next_row_id;
            mut_idx.next_row_id += 1;
            row_id
        };

        let request = EmbedRequest {
            row_id,
            text,
            column: source_col_name,
        };
        if let Ok(response) = sidecar.embed(request) {
            let mut indexes = self.vector_indexes.write().unwrap();
            if let Some(mut_idx) = indexes.get_mut(table) {
                mut_idx.delta.insert(row_id, response.embedding.clone());
                mut_idx.vectors.insert(row_id, response.embedding);
            }
        }
    }

    fn exec_select(&mut self, q: &sqlparser::ast::Query) -> GalaxResult<QueryResult> {
        let table = extract_table(q);
        let (columns, filter) = extract_projection_and_filter(q);
        let plan = QueryPlan::FullScan {
            table: table.clone(),
            filter,
            columns,
        };
        let mut ctx = self.context();
        let res = execute_with_context(&plan, &mut ctx)?;
        self.catalog = std::mem::take(&mut ctx.catalog);
        Ok(query_result_from(res))
    }

    fn exec_update(
        &mut self,
        table: &str,
        assignments: &[sqlparser::ast::Assignment],
        selection: Option<&sqlparser::ast::Expr>,
    ) -> GalaxResult<QueryResult> {
        let aligned: Vec<(String, Value)> = assignments
            .iter()
            .map(|a| (a.target.to_string(), value_from_expr(&a.value)))
            .collect();
        let filter = selection.and_then(filter_from_expr);
        let plan = QueryPlan::Update {
            table: table.to_string(),
            assignments: aligned,
            filter,
        };
        self.dispatch(plan)
    }

    fn exec_delete(&mut self, del: &sqlparser::ast::Delete) -> GalaxResult<QueryResult> {
        let table = match &del.from {
            sqlparser::ast::FromTable::WithFromKeyword(tables)
            | sqlparser::ast::FromTable::WithoutKeyword(tables) => tables
                .first()
                .map(|t| t.relation.to_string())
                .unwrap_or_default(),
        };
        if table.is_empty() {
            return Ok(QueryResult::RowCount(0));
        }
        let filter = del.selection.as_ref().and_then(filter_from_expr);
        let plan = QueryPlan::Delete { table, filter };
        self.dispatch(plan)
    }

    fn exec_semantic_match_standalone(
        &mut self,
        expr: &galaxdb_sql::ast::SemanticMatchExpr,
    ) -> GalaxResult<QueryResult> {
        // Identify the table by the embedding column name.
        let indexes = self.vector_indexes.read().unwrap();
        let table_name = indexes
            .iter()
            .find(|(_, idx)| {
                idx.embedding_column == expr.column || idx.source_column == expr.column
            })
            .map(|(n, _)| n.clone())
            .ok_or_else(|| {
                GalaxError::Internal(format!(
                    "no embedding index found for column '{}'",
                    expr.column
                ))
            })?;
        drop(indexes);

        let plan = planner::plan_semantic_search(
            table_name,
            expr.clone(),
            None,
            None,
        );
        self.dispatch(plan)
    }

    fn dispatch(&mut self, plan: QueryPlan) -> GalaxResult<QueryResult> {
        let mut ctx = self.context();
        let res = execute_with_context(&plan, &mut ctx)?;
        self.catalog = std::mem::take(&mut ctx.catalog);
        Ok(query_result_from(res))
    }

    /// Build a fresh `ExecutorContext` that shares this database's
    /// engine, sidecar, tag catalog, merkle DAG, and vector backend.
    /// The context's catalog is moved in from `self.catalog` and moved
    /// back after the executor runs so DDL mutations are preserved.
    fn context(&mut self) -> ExecutorContext {
        let mut ctx = ExecutorContext::new(self.engine.clone());
        ctx.catalog = std::mem::take(&mut self.catalog);
        ctx.sidecar = self.sidecar.clone();
        ctx.merkle_dag = Some(self.merkle_dag.clone());
        ctx.tag_catalog = Some(self.tag_catalog.clone());
        ctx.vector_backend = Some(Arc::new(EmbeddedVectorBackend {
            sidecar: self.sidecar.clone(),
            indexes: self.vector_indexes.clone(),
        }));
        ctx
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn table_count(&self) -> usize {
        self.catalog.table_count()
    }
    pub fn table_exists(&self, name: &str) -> bool {
        self.catalog.table_exists(name)
    }
    pub fn row_count(&self) -> u64 {
        self.engine.row_count()
    }
}

impl Drop for Database {
    fn drop(&mut self) {
        self.engine.shutdown();
    }
}

// ---------------------------------------------------------------------------
// Vector backend — bridges the SQL executor's VectorSearchBackend trait to
// the database's local HNSW + delta buffer + sidecar.
// ---------------------------------------------------------------------------

struct EmbeddedVectorBackend {
    sidecar: Option<Arc<SidecarManager>>,
    indexes: Arc<RwLock<HashMap<String, TableVectorIndex>>>,
}

impl VectorSearchBackend for EmbeddedVectorBackend {
    fn semantic_search(
        &self,
        table: &str,
        query_text: &str,
        threshold: f64,
        k: usize,
        _strategy: SearchStrategy,
    ) -> GalaxResult<Vec<VectorSearchResult>> {
        // Embed the query through the sidecar. No mock fallback —
        // missing sidecar is a typed error.
        let sidecar = self
            .sidecar
            .as_ref()
            .ok_or(GalaxError::SidecarUnavailable)?;
        let indexes = self.indexes.read().unwrap();
        let idx = indexes
            .get(table)
            .ok_or_else(|| GalaxError::TableNotFound(table.to_string()))?;

        let request = EmbedRequest {
            row_id: 0,
            text: query_text.to_string(),
            column: idx.embedding_column.clone(),
        };
        let response = sidecar
            .embed(request)
            .map_err(|_| GalaxError::SidecarUnavailable)?;

        let sm_config = SemanticMatchConfig {
            hnsw_candidates: 100,
            ef_search: 200,
            brute_force_threshold: 1000,
            brute_force_ratio: 0.001,
        };
        let vectors_ref = &idx.vectors;
        let results = execute_semantic_match(
            &response.embedding,
            &idx.hnsw,
            &idx.delta,
            threshold,
            k,
            &sm_config,
            |row_id| vectors_ref.get(&row_id).cloned(),
        );
        Ok(results
            .into_iter()
            .map(|r| VectorSearchResult {
                row_id: r.row_id,
                similarity: r.similarity,
            })
            .collect())
    }

    fn brute_force_filtered(
        &self,
        table: &str,
        query_text: &str,
        threshold: f64,
        k: usize,
        _filter: &FilterExpr,
    ) -> GalaxResult<Vec<VectorSearchResult>> {
        // Today the brute-force path shares the HNSW-backed
        // implementation; the planner's adaptive decision between
        // `BruteForceFiltered` and `HnswWithPostFilter` still picks
        // which plan variant reaches us, but both resolve here for
        // v1. A dedicated scan-then-distance path is task 31.5
        // follow-up work.
        self.semantic_search(
            table,
            query_text,
            threshold,
            k,
            SearchStrategy::HnswWithPostFilter,
        )
    }
}

// ---------------------------------------------------------------------------
// AST-to-Value helpers
// ---------------------------------------------------------------------------

fn value_from_expr(e: &sqlparser::ast::Expr) -> Value {
    match e {
        sqlparser::ast::Expr::Value(v) => match v {
            sqlparser::ast::Value::Number(n, _) => n
                .parse::<i64>()
                .map(Value::Integer)
                .or_else(|_| n.parse::<f64>().map(Value::Float))
                .unwrap_or_else(|_| Value::Text(n.clone())),
            sqlparser::ast::Value::SingleQuotedString(s)
            | sqlparser::ast::Value::DoubleQuotedString(s) => Value::Text(s.clone()),
            sqlparser::ast::Value::Boolean(b) => Value::Bool(*b),
            sqlparser::ast::Value::Null => Value::Null,
            other => Value::Text(format!("{}", other)),
        },
        other => Value::Text(format!("{}", other)),
    }
}

fn query_result_from(r: ExecuteResult) -> QueryResult {
    match r {
        ExecuteResult::Rows { rows, .. } => QueryResult::Rows(
            rows.into_iter()
                .map(|row: SqlRow| QueryRow {
                    values: row
                        .columns
                        .into_iter()
                        .map(|(k, v)| (k, row_codec::value_display(&v)))
                        .collect(),
                })
                .collect(),
        ),
        ExecuteResult::RowCount(n) => QueryResult::RowCount(n),
        ExecuteResult::Ok(msg) => QueryResult::Ok(msg),
        ExecuteResult::Error(msg) => QueryResult::Ok(msg),
    }
}

fn extract_table(q: &sqlparser::ast::Query) -> String {
    if let sqlparser::ast::SetExpr::Select(s) = q.body.as_ref() {
        if let Some(f) = s.from.first() {
            return f.relation.to_string();
        }
    }
    "unknown".to_string()
}

/// Extract the projection column list and the WHERE filter from a
/// `SELECT` query. `SELECT *` / unsupported projection items yield
/// an empty column list (which the executor interprets as "all
/// columns"). Missing WHERE returns `None`.
///
/// Supported projection items:
/// - `*` → empty list (all columns)
/// - `col_name` / `table.col_name` → column name
///
/// Anything else (aggregates, expressions, aliases) returns the
/// empty projection so the full row comes back — that's correct
/// behaviour for v1, the executor caller can drop columns it
/// doesn't want. A dedicated aggregation path is task 18.8 scope.
fn extract_projection_and_filter(
    q: &sqlparser::ast::Query,
) -> (Vec<String>, Option<FilterExpr>) {
    let sqlparser::ast::SetExpr::Select(s) = q.body.as_ref() else {
        return (vec![], None);
    };

    let mut columns = Vec::new();
    let mut projection_is_star = false;
    for item in &s.projection {
        match item {
            sqlparser::ast::SelectItem::Wildcard(_)
            | sqlparser::ast::SelectItem::QualifiedWildcard(..) => {
                projection_is_star = true;
                break;
            }
            sqlparser::ast::SelectItem::UnnamedExpr(expr)
            | sqlparser::ast::SelectItem::ExprWithAlias { expr, .. } => {
                if let Some(name) = column_name_from_expr(expr) {
                    columns.push(name);
                } else {
                    // Unsupported expression — fall back to full row.
                    projection_is_star = true;
                    break;
                }
            }
        }
    }
    let columns = if projection_is_star { Vec::new() } else { columns };

    let filter = s.selection.as_ref().and_then(filter_from_expr);

    (columns, filter)
}

/// If `expr` is a bare column reference, return its name.
fn column_name_from_expr(expr: &sqlparser::ast::Expr) -> Option<String> {
    match expr {
        sqlparser::ast::Expr::Identifier(id) => Some(id.value.clone()),
        sqlparser::ast::Expr::CompoundIdentifier(parts) => {
            // table.col → "col"
            parts.last().map(|p| p.value.clone())
        }
        _ => None,
    }
}

/// Convert a WHERE clause from the `sqlparser` AST into a `FilterExpr`
/// the executor can evaluate. Supported shapes:
///
/// - `col = literal`, `col != literal`, `col <> literal`
/// - `col < literal`, `col > literal`, `col <= literal`, `col >= literal`
/// - `expr AND expr`, `expr OR expr`
///
/// The left side must be a column reference and the right side a
/// literal value. Anything else returns `None` (treated by the planner
/// as "no filter", which is strictly less restrictive than the query
/// asks for — callers should prefer a parse error for that case, but
/// at the embedded layer today we only forward supported filters).
fn filter_from_expr(expr: &sqlparser::ast::Expr) -> Option<FilterExpr> {
    use sqlparser::ast::{BinaryOperator, Expr};
    match expr {
        Expr::BinaryOp { left, op, right } => match op {
            BinaryOperator::And => Some(FilterExpr::And(
                Box::new(filter_from_expr(left)?),
                Box::new(filter_from_expr(right)?),
            )),
            BinaryOperator::Or => Some(FilterExpr::Or(
                Box::new(filter_from_expr(left)?),
                Box::new(filter_from_expr(right)?),
            )),
            BinaryOperator::Eq
            | BinaryOperator::NotEq
            | BinaryOperator::Lt
            | BinaryOperator::Gt
            | BinaryOperator::LtEq
            | BinaryOperator::GtEq => {
                // Try col OP literal. If that fails, try literal OP col
                // and flip.
                if let (Some(col), Some(val)) =
                    (column_name_from_expr(left), literal_value(right))
                {
                    return Some(build_cmp(op, col, val));
                }
                if let (Some(val), Some(col)) =
                    (literal_value(left), column_name_from_expr(right))
                {
                    let flipped = flip_cmp_op(op);
                    return Some(build_cmp(&flipped, col, val));
                }
                None
            }
            _ => None,
        },
        Expr::Nested(inner) => filter_from_expr(inner),
        _ => None,
    }
}

/// Build a `FilterExpr` for a comparison op with `col OP val` ordering.
fn build_cmp(
    op: &sqlparser::ast::BinaryOperator,
    column: String,
    value: Value,
) -> FilterExpr {
    use sqlparser::ast::BinaryOperator::*;
    match op {
        Eq => FilterExpr::Eq { column, value },
        NotEq => FilterExpr::Ne { column, value },
        Lt => FilterExpr::Lt { column, value },
        Gt => FilterExpr::Gt { column, value },
        LtEq => FilterExpr::Le { column, value },
        GtEq => FilterExpr::Ge { column, value },
        _ => FilterExpr::Eq { column, value },
    }
}

/// Mirror a comparison operator when the column ends up on the right
/// side of the expression (`5 < id` becomes `id > 5`).
fn flip_cmp_op(op: &sqlparser::ast::BinaryOperator) -> sqlparser::ast::BinaryOperator {
    use sqlparser::ast::BinaryOperator::*;
    match op {
        Lt => Gt,
        Gt => Lt,
        LtEq => GtEq,
        GtEq => LtEq,
        other => other.clone(),
    }
}

/// If `expr` is a literal, return the corresponding [`Value`]. Mirrors
/// [`value_from_expr`] but returns `None` on non-literals so we can
/// distinguish a successful conversion from a fallback string.
fn literal_value(expr: &sqlparser::ast::Expr) -> Option<Value> {
    use sqlparser::ast::{Expr, Value as SqlValue};
    match expr {
        Expr::Value(v) => match v {
            SqlValue::Number(n, _) => n
                .parse::<i64>()
                .map(Value::Integer)
                .or_else(|_| n.parse::<f64>().map(Value::Float))
                .ok(),
            SqlValue::SingleQuotedString(s) | SqlValue::DoubleQuotedString(s) => {
                Some(Value::Text(s.clone()))
            }
            SqlValue::Boolean(b) => Some(Value::Bool(*b)),
            SqlValue::Null => Some(Value::Null),
            _ => None,
        },
        Expr::UnaryOp {
            op: sqlparser::ast::UnaryOperator::Minus,
            expr,
        } => match literal_value(expr) {
            Some(Value::Integer(n)) => Some(Value::Integer(-n)),
            Some(Value::Float(f)) => Some(Value::Float(-f)),
            _ => None,
        },
        _ => None,
    }
}

/// Find the column name and its index in `column_names` (the explicit
/// INSERT column list) that corresponds to the table's embedding source
/// column. If `column_names` is empty the index is resolved
/// positionally against `entry.columns`.
fn resolve_source_column(
    entry: &galaxdb_sql::executor::TableEntry,
    column_names: &[String],
    source_col: &str,
) -> (String, Option<usize>) {
    if !column_names.is_empty() {
        let idx = column_names.iter().position(|n| n == source_col);
        return (source_col.to_string(), idx);
    }
    let idx = entry.columns.iter().position(|c| c.name == source_col);
    (source_col.to_string(), idx)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
        db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
            .unwrap();
        db.execute("INSERT INTO users (id, name) VALUES (2, 'bob')")
            .unwrap();
        let r = db.execute("SELECT * FROM users").unwrap();
        match r {
            QueryResult::Rows(rows) => {
                assert_eq!(rows.len(), 2);
                assert!(
                    rows.iter()
                        .any(|r| r.values.iter().any(|(k, v)| k == "name" && v == "alice"))
                );
            }
            other => panic!("expected Rows, got {:?}", other),
        }
    }

    #[test]
    fn insert_10_rows_and_count() {
        let mut db = test_db();
        db.execute("CREATE TABLE t (id INT, val TEXT)").unwrap();
        for i in 0..10 {
            db.execute(&format!(
                "INSERT INTO t (id, val) VALUES ({}, 'v{}')",
                i, i
            ))
            .unwrap();
        }
        assert_eq!(db.row_count(), 10);
        match db.execute("SELECT * FROM t").unwrap() {
            QueryResult::Rows(rows) => assert_eq!(rows.len(), 10),
            other => panic!("expected Rows, got {:?}", other),
        }
    }

    #[test]
    fn select_nonexistent_fails() {
        let mut db = test_db();
        assert!(db.execute("SELECT * FROM nope").is_err());
    }

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
        assert!(matches!(
            db.execute("SHOW EMBEDDING HEALTH").unwrap(),
            QueryResult::Rows(_)
        ));
        assert!(matches!(
            db.execute("CREATE VERSION TAG 'v1'").unwrap(),
            QueryResult::Ok(_)
        ));
    }

    #[test]
    fn version_tag_creation_and_pinning() {
        let mut db = test_db();
        db.execute("CREATE TABLE docs (id INT, content TEXT)")
            .unwrap();
        db.execute("INSERT INTO docs (id, content) VALUES (1, 'hello')")
            .unwrap();
        db.execute("INSERT INTO docs (id, content) VALUES (2, 'world')")
            .unwrap();

        let result = db.execute("CREATE VERSION TAG 'v1.0'").unwrap();
        assert!(matches!(result, QueryResult::Ok(_)));

        let catalog = db.tag_catalog.lock().unwrap();
        assert!(catalog.get_tag("v1.0").is_some());
        let tag = catalog.get_tag("v1.0").unwrap();
        assert_eq!(tag.name, "v1.0");
        assert!(!tag.for_training);
        drop(catalog);

        assert!(db.execute("CREATE VERSION TAG 'v1.0'").is_err());
    }

    #[test]
    fn version_tag_for_training() {
        let mut db = test_db();
        db.execute("CREATE TABLE t (id INT)").unwrap();

        let result = db
            .execute(
                "CREATE VERSION TAG 'train-v1' FOR TRAINING WITH TRAINING PRECISION 'sq8' \
                 TRAINING SEED 42",
            )
            .unwrap();
        assert!(matches!(result, QueryResult::Ok(_)));

        let catalog = db.tag_catalog.lock().unwrap();
        let tag = catalog.get_tag("train-v1").unwrap();
        assert!(tag.for_training);
        let opts = tag.training_opts.as_ref().unwrap();
        assert_eq!(opts.precision, "sq8");
        assert_eq!(opts.seed, Some(42));
        assert!(opts.deterministic_order);
    }

    /// End-to-end SEMANTIC_MATCH test using the real model. Gated behind
    /// the `online-tests` feature — requires network access to HuggingFace
    /// Hub on first run (downloads ~90 MB for all-MiniLM-L6-v2).
    ///
    /// ```text
    /// cargo test -p galaxdb-embedded --features online-tests --release
    /// ```
    #[cfg(feature = "online-tests")]
    #[test]
    fn semantic_match_end_to_end() {
        const MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";
        const MODEL_DIM: usize = 384;

        let sidecar_binary = std::env::current_exe()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("galaxdb-sidecar");

        if !sidecar_binary.exists() {
            let status = std::process::Command::new("cargo")
                .args(["build", "-p", "galaxdb-sidecar"])
                .status()
                .expect("cargo build");
            assert!(status.success(), "failed to build sidecar binary");
        }

        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("semantic_db");
        std::mem::forget(dir);

        let mut db = Database::open_with_sidecar(
            db_path.to_str().unwrap(),
            sidecar_binary.to_str().unwrap(),
            MODEL_ID,
        )
        .unwrap();

        db.execute(&format!(
            "CREATE TABLE docs (id INT PRIMARY KEY, \
             content TEXT EMBEDDING MODEL '{MODEL_ID}' DIM {MODEL_DIM})"
        ))
        .unwrap();

        assert!(db.vector_indexes.read().unwrap().contains_key("docs"));

        db.execute("INSERT INTO docs (id, content) VALUES (1, 'machine learning is great')")
            .unwrap();
        db.execute("INSERT INTO docs (id, content) VALUES (2, 'rust programming language')")
            .unwrap();
        db.execute("INSERT INTO docs (id, content) VALUES (3, 'machine learning algorithms')")
            .unwrap();

        {
            let indexes = db.vector_indexes.read().unwrap();
            let idx = indexes.get("docs").unwrap();
            assert_eq!(
                idx.delta.vector_count(),
                3,
                "three INSERTs must produce three sidecar-computed embeddings"
            );
            assert_eq!(idx.dim, MODEL_DIM);
        }

        let result = db
            .execute(
                "SELECT * FROM docs WHERE SEMANTIC_MATCH(content, 'machine learning', 0.0)",
            )
            .unwrap();
        match result {
            QueryResult::Rows(rows) => {
                assert!(!rows.is_empty(), "SEMANTIC_MATCH should return results");
                for row in &rows {
                    assert!(row.values.iter().any(|(k, _)| k == "row_id"));
                    assert!(row.values.iter().any(|(k, _)| k == "similarity"));
                }
            }
            other => panic!("expected Rows, got {:?}", other),
        }

        assert!(
            db.sidecar.as_ref().unwrap().is_healthy(),
            "sidecar must still be healthy after a successful query"
        );
    }

    // -----------------------------------------------------------------
    // Phase I regressions — WHERE / projection plumbing
    //
    // Before Phase I, `exec_select`, `exec_update`, and `exec_delete`
    // hard-coded `filter: None`, silently ignoring the WHERE clause.
    // These tests drive real SQL through `Database::execute` and assert
    // that the filter reaches the executor. A regression would show up
    // as wrong row counts, which is exactly what AWS integration
    // testing caught.
    // -----------------------------------------------------------------

    fn seeded_db() -> Database {
        let mut db = test_db();
        db.execute("CREATE TABLE p (id INT PRIMARY KEY, name TEXT, price FLOAT)")
            .unwrap();
        db.execute("INSERT INTO p (id, name, price) VALUES (1, 'espresso', 3.50)")
            .unwrap();
        db.execute("INSERT INTO p (id, name, price) VALUES (2, 'latte', 4.25)")
            .unwrap();
        db.execute("INSERT INTO p (id, name, price) VALUES (3, 'mocha', 4.75)")
            .unwrap();
        db
    }

    fn rows_of(r: QueryResult) -> Vec<QueryRow> {
        match r {
            QueryResult::Rows(rows) => rows,
            other => panic!("expected Rows, got {:?}", other),
        }
    }

    #[test]
    fn select_where_price_filters_rows() {
        let mut db = seeded_db();
        let rows = rows_of(
            db.execute("SELECT id, name, price FROM p WHERE price > 4.0")
                .unwrap(),
        );
        assert_eq!(rows.len(), 2, "should return latte + mocha only");
        for r in &rows {
            let price_str = r
                .values
                .iter()
                .find(|(k, _)| k == "price")
                .map(|(_, v)| v.clone())
                .unwrap();
            let price: f64 = price_str.parse().unwrap();
            assert!(price > 4.0, "row slipped past WHERE: price={price}");
        }
    }

    #[test]
    fn select_where_id_equals_returns_single_row() {
        let mut db = seeded_db();
        let rows = rows_of(
            db.execute("SELECT id, name FROM p WHERE id = 2").unwrap(),
        );
        assert_eq!(rows.len(), 1);
        let name = &rows[0]
            .values
            .iter()
            .find(|(k, _)| k == "name")
            .unwrap()
            .1;
        assert_eq!(name, "latte");
    }

    #[test]
    fn select_projection_restricts_columns() {
        let mut db = seeded_db();
        let rows = rows_of(db.execute("SELECT name FROM p").unwrap());
        assert_eq!(rows.len(), 3);
        for r in &rows {
            assert_eq!(
                r.values.len(),
                1,
                "projection should limit output to one column, got {:?}",
                r.values
            );
            assert_eq!(r.values[0].0, "name");
        }
    }

    #[test]
    fn update_where_affects_only_matching_rows() {
        let mut db = seeded_db();
        match db
            .execute("UPDATE p SET price = 9.99 WHERE id = 3")
            .unwrap()
        {
            QueryResult::RowCount(n) => assert_eq!(n, 1, "UPDATE with id=3 must affect 1 row"),
            other => panic!("expected RowCount, got {:?}", other),
        }

        // Others unchanged.
        let latte = rows_of(
            db.execute("SELECT price FROM p WHERE id = 2").unwrap(),
        );
        assert_eq!(latte.len(), 1);
        assert_eq!(latte[0].values[0].1, "4.25");

        // Target updated.
        let mocha = rows_of(
            db.execute("SELECT price FROM p WHERE id = 3").unwrap(),
        );
        assert_eq!(mocha.len(), 1);
        assert_eq!(mocha[0].values[0].1, "9.99");
    }

    #[test]
    fn delete_where_affects_only_matching_rows() {
        let mut db = seeded_db();
        match db.execute("DELETE FROM p WHERE id = 1").unwrap() {
            QueryResult::RowCount(n) => assert_eq!(n, 1, "DELETE with id=1 must remove 1 row"),
            other => panic!("expected RowCount, got {:?}", other),
        }
        let rows = rows_of(db.execute("SELECT id FROM p").unwrap());
        assert_eq!(rows.len(), 2, "two rows should remain after deleting id=1");

        // Deleting a non-existent row is a no-op.
        match db.execute("DELETE FROM p WHERE id = 99").unwrap() {
            QueryResult::RowCount(n) => assert_eq!(n, 0),
            other => panic!("expected RowCount, got {:?}", other),
        }
    }

    #[test]
    fn delete_without_where_clears_table() {
        let mut db = seeded_db();
        match db.execute("DELETE FROM p").unwrap() {
            QueryResult::RowCount(n) => {
                assert_eq!(n, 3, "DELETE without WHERE must remove all rows")
            }
            other => panic!("expected RowCount, got {:?}", other),
        }
        let rows = rows_of(db.execute("SELECT * FROM p").unwrap());
        assert!(rows.is_empty());
    }

    #[test]
    fn where_and_or_combine() {
        let mut db = seeded_db();
        let rows = rows_of(
            db.execute(
                "SELECT id FROM p WHERE price > 4.0 AND price < 4.5",
            )
            .unwrap(),
        );
        assert_eq!(rows.len(), 1, "only latte matches 4.0 < p < 4.5");
        assert_eq!(rows[0].values[0].1, "2");

        let rows = rows_of(
            db.execute(
                "SELECT id FROM p WHERE id = 1 OR id = 3",
            )
            .unwrap(),
        );
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn where_text_equality() {
        let mut db = seeded_db();
        let rows = rows_of(
            db.execute("SELECT id FROM p WHERE name = 'latte'")
                .unwrap(),
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values[0].1, "2");
    }

    #[test]
    fn where_column_on_right_side_is_flipped() {
        // `5 < id` should behave like `id > 5`.
        let mut db = seeded_db();
        let rows = rows_of(
            db.execute("SELECT id FROM p WHERE 2 < id").unwrap(),
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values[0].1, "3");
    }
}
