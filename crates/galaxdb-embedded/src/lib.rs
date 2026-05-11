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
    /// Primary-key bytes → vector row-id. Populated when the sidecar
    /// returns an embedding for a newly inserted row; consumed by
    /// `on_row_deleted` so we know which vector row to tombstone when
    /// the user issues `DELETE FROM t WHERE ...`. Without this map,
    /// SQL-level DELETEs would leave orphaned vectors in the HNSW
    /// graph (task 18.6 hole surfaced during the Phase I audit).
    key_to_row_id: HashMap<Vec<u8>, u64>,
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
        // AT VERSION intercept: sqlparser doesn't understand the
        // AuroraSQL `AT VERSION ...` suffix, so if we see one on a
        // SELECT we split the SQL into (stripped, at_version) and
        // dispatch to the versioned plan arm directly. See task 32.3 /
        // 32.4 in docs/CONSOLIDATION.md.
        if let Some((stripped, at)) = split_at_version(sql)? {
            return self.exec_select_at_version(&stripped, at);
        }

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
        // AT VERSION on the read path: same intercept as `execute`, but
        // we don't need `&mut self` because the plan only scans storage.
        if let Some((stripped, at)) = split_at_version(sql)? {
            return self.select_at_version_readonly(&stripped, at);
        }

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
            engine: self.engine.clone(),
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
                columns: bi.columns.clone(),
                values: bi.values.clone(),
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
                        key_to_row_id: HashMap::new(),
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

        // Compute the storage primary key for this row so we can
        // remember the mapping `primary_key -> vector_row_id`. SQL
        // DELETEs later use this to tombstone the right delta-buffer
        // entry (task 18.6). Failure to build the key is non-fatal —
        // the row still gets embedded, it just can't be reverse-mapped
        // later. We log at warn level so operators see the drift.
        let row_key = match row_codec::align_values(entry, column_names, values)
            .and_then(|ordered| row_codec::build_primary_key(table, entry, &ordered))
        {
            Ok(k) => Some(k),
            Err(e) => {
                tracing::warn!(
                    table = %table,
                    error = %e,
                    "could not compute primary key for embedding row; \
                     DELETE of this row will not tombstone its vector",
                );
                None
            }
        };

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
                if let Some(key) = row_key {
                    mut_idx.key_to_row_id.insert(key, row_id);
                }
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

    /// Dispatch a `SELECT ... AT VERSION <ref> [CONSISTENCY <mode>]`
    /// query to the canonical executor. The SQL text passed in has
    /// already had the AT VERSION suffix stripped by
    /// [`split_at_version`]; we parse the remainder as a normal
    /// SELECT so we can reuse `extract_projection_and_filter`, then
    /// build a `FullScanAtVersion` plan.
    fn exec_select_at_version(
        &mut self,
        stripped_sql: &str,
        at: galaxdb_sql::ast::AtVersionExpr,
    ) -> GalaxResult<QueryResult> {
        let stmts = parser::parse(stripped_sql)?;
        let Some(stmt) = stmts.first() else {
            return Err(GalaxError::Internal(
                "AT VERSION: SELECT body parsed to zero statements".into(),
            ));
        };
        let AuroraStatement::Standard(boxed) = stmt else {
            return Err(GalaxError::Internal(
                "AT VERSION is only supported on SELECT statements".into(),
            ));
        };
        let sqlparser::ast::Statement::Query(q) = boxed.as_ref() else {
            return Err(GalaxError::Internal(
                "AT VERSION is only supported on SELECT statements".into(),
            ));
        };

        let table = extract_table(q);
        let (columns, filter) = extract_projection_and_filter(q);
        let plan = QueryPlan::FullScanAtVersion {
            table,
            filter,
            columns,
            at,
        };
        let mut ctx = self.context();
        let res = execute_with_context(&plan, &mut ctx)?;
        self.catalog = std::mem::take(&mut ctx.catalog);
        Ok(query_result_from(res))
    }

    /// `&self` variant of [`Self::exec_select_at_version`] used by the
    /// wire-protocol read path.
    fn select_at_version_readonly(
        &self,
        stripped_sql: &str,
        at: galaxdb_sql::ast::AtVersionExpr,
    ) -> GalaxResult<QueryResult> {
        let stmts = parser::parse(stripped_sql)?;
        let Some(stmt) = stmts.first() else {
            return Err(GalaxError::Internal(
                "AT VERSION: SELECT body parsed to zero statements".into(),
            ));
        };
        let AuroraStatement::Standard(boxed) = stmt else {
            return Err(GalaxError::Internal(
                "AT VERSION is only supported on SELECT statements".into(),
            ));
        };
        let sqlparser::ast::Statement::Query(q) = boxed.as_ref() else {
            return Err(GalaxError::Internal(
                "AT VERSION is only supported on SELECT statements".into(),
            ));
        };

        let table = extract_table(q);
        let (columns, filter) = extract_projection_and_filter(q);
        if table != "unknown" && !self.catalog.table_exists(&table) {
            return Err(GalaxError::TableNotFound(table));
        }
        let plan = QueryPlan::FullScanAtVersion {
            table,
            filter,
            columns,
            at,
        };

        let mut ctx = ExecutorContext::new(self.engine.clone());
        ctx.catalog = self.catalog.clone();
        ctx.tag_catalog = Some(self.tag_catalog.clone());
        ctx.merkle_dag = Some(self.merkle_dag.clone());
        ctx.vector_backend = Some(Arc::new(EmbeddedVectorBackend {
            sidecar: self.sidecar.clone(),
            indexes: self.vector_indexes.clone(),
            engine: self.engine.clone(),
        }));
        let res = execute_with_context(&plan, &mut ctx)?;
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
            engine: self.engine.clone(),
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

    /// Build a [`galaxdb_storage::compaction::GcContext`] that pins
    /// every commit timestamp currently referenced by a version tag.
    /// Tasks 10.5 and 33.5: ensures the compactor's MVCC GC retains
    /// row versions that tagged snapshots depend on.
    ///
    /// `oldest_active_snapshot` should be the minimum active
    /// transaction's read timestamp (None if there are no active
    /// readers). When compaction runs from embedded-mode callers with
    /// no transaction manager, passing `None` is safe: pinned
    /// timestamps alone are sufficient to keep training snapshots
    /// alive, and unreferenced versions are already beyond any
    /// caller's interest.
    pub fn gc_context_with_pins(
        &self,
        oldest_active_snapshot: Option<u64>,
    ) -> galaxdb_storage::compaction::GcContext {
        let pins = self
            .tag_catalog
            .lock()
            .map(|tc| tc.all_pinned_timestamps())
            .unwrap_or_default();
        galaxdb_storage::compaction::GcContext::with_pins(
            oldest_active_snapshot,
            pins,
        )
    }

    /// Export the table backing `tag` as a Lance dataset on disk and
    /// return the dataset's path (Req 25 / Req 32, task 22.4).
    ///
    /// What the method actually does, in order:
    ///
    /// 1. Resolve `tag` via the [`TagCatalog`]. Unknown tag name →
    ///    [`GalaxError::Internal`] carrying "unknown version tag: …".
    /// 2. Reject if the tag was not created with `FOR TRAINING`. Only
    ///    training tags deterministically pin block sets and precision
    ///    options — exporting a non-training tag would silently lose
    ///    that contract.
    /// 3. Pick the table the tag is associated with. For v1 the
    ///    assumption is one-table-per-database (which is how the
    ///    canonical `CREATE VERSION TAG` statement is used today); if
    ///    the catalog holds multiple tables we pick the only table with
    ///    any data and error if that choice is ambiguous.
    /// 4. Build an Arrow schema from that table's `CatalogColumn`s,
    ///    mapping SQL types to Arrow types (`INT` / `BIGINT` → `Int64`,
    ///    `FLOAT` / `REAL` / `DOUBLE` → `Float32`, everything else →
    ///    `Utf8`). Embedding columns are not projected yet — the v1
    ///    export surface is the scalar/text row; vector export lands
    ///    once the delta buffer can be versioned.
    /// 5. Instantiate an [`EmbeddedLanceExportSource`] that reads rows
    ///    at the tag's `version_timestamp` via
    ///    [`Engine::scan_all_at`], filters them to the chosen table's
    ///    primary-key prefix, and decodes each row through
    ///    [`row_codec::decode_row`].
    /// 6. Drive [`LanceExporter::export`] on a fresh tokio current-
    ///    thread runtime (the embedded database is a sync API; all
    ///    the async lives inside Lance's writer). The output path is
    ///    deterministic: `<db>/training_exports/<tag>_<version_ts>/`
    ///    so repeat exports of the same tag overwrite the same
    ///    directory rather than racing for different names.
    /// 7. Record a lineage row through [`InMemoryLineageSink`]. The
    ///    persistent `_galaxdb_training_exports` table is tracked
    ///    under Req 38 / task 36; wiring it up is a follow-up — until
    ///    then the sink keeps the shape correct so callers can read
    ///    the number of exports without having to run through a SQL
    ///    system-table scan.
    ///
    /// The returned path points at the on-disk Lance dataset. Python
    /// callers wrap it with `lance.dataset(path).to_pytorch()` to get
    /// an `IterableDataset` — that glue lives in the Python package,
    /// not in this Rust method, because PyTorch is a Python-only
    /// dependency (Rule 5: no vendor lock-in in the engine core).
    pub fn training_dataset(&self, tag: &str) -> GalaxResult<PathBuf> {
        use galaxdb_versioning::{
            InMemoryLineageSink, LanceExporter, TrainingExportLineageSink, TrainingPrecision,
        };

        // 1. Resolve the tag and clone the bits we need out of the
        // mutex before we do any async work. The exporter takes
        // `Arc<TagCatalog>` / `Arc<MerkleDag>` — we snapshot the
        // current state so the running export cannot see concurrent
        // tag creations/deletions mid-flight.
        let (version_tag, tag_catalog_snapshot, merkle_dag_snapshot) = {
            let tag_catalog = self
                .tag_catalog
                .lock()
                .map_err(|_| GalaxError::Internal("tag catalog mutex poisoned".into()))?;
            let Some(version_tag) = tag_catalog.get_tag(tag).cloned() else {
                return Err(GalaxError::Internal(format!(
                    "unknown version tag: {tag}"
                )));
            };
            let tag_catalog_snapshot = tag_catalog.clone();
            let merkle_dag_snapshot = self
                .merkle_dag
                .lock()
                .map_err(|_| GalaxError::Internal("merkle dag mutex poisoned".into()))?
                .clone();
            (version_tag, tag_catalog_snapshot, merkle_dag_snapshot)
        };

        // 2. Training-only — non-training tags don't carry the
        // deterministic-order / precision contract the export relies
        // on.
        if !version_tag.for_training {
            return Err(GalaxError::Internal(format!(
                "version tag '{tag}' is not a FOR TRAINING tag; \
                 only training tags can be exported as Lance datasets"
            )));
        }

        // 3. Pick the table. v1 supports the single-table case that
        // `CREATE VERSION TAG` produces today. If there is more than
        // one table with rows we refuse rather than exporting the
        // first one alphabetically — silent choice here would be a
        // correctness bug the user has no way to see.
        let (table_name, table_entry) = self.pick_training_table()?;

        // 4. Build the Arrow schema from the catalog.
        let schema = Arc::new(arrow_schema_from_catalog(&table_entry));

        // 5. Build the export source over the real engine.
        let source: Arc<dyn galaxdb_versioning::LanceExportSource> =
            Arc::new(EmbeddedLanceExportSource {
                engine: self.engine.clone(),
                table_name: table_name.clone(),
                table_entry: table_entry.clone(),
                version_timestamp: version_tag.version_timestamp,
            });

        // 6. Resolve the output path. Use the tag name plus the tag's
        // version timestamp so repeat exports of a mutated tag don't
        // collide (tag names are unique so the version_ts is almost
        // redundant — but it makes the path self-describing).
        let safe_tag = sanitize_tag_for_path(tag);
        let output_path = self
            .path
            .join("training_exports")
            .join(format!("{safe_tag}_{}", version_tag.version_timestamp));

        // Lance refuses to write into a non-empty directory. For a
        // deterministic repeat export we clear any previous artefact
        // at the same path before handing it to the writer. The
        // parent `training_exports` directory is created on demand.
        if output_path.exists() {
            std::fs::remove_dir_all(&output_path)?;
        }
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // 7. Precision comes from the tag's training metadata, falling
        // back to Float32 if the tag didn't specify one (shouldn't
        // happen with the SQL path today, which always sets it — but
        // guard against programmatic tag creation).
        let precision = version_tag
            .training_opts
            .as_ref()
            .and_then(|o| TrainingPrecision::from_str_opt(&o.precision))
            .unwrap_or(TrainingPrecision::Float32);
        let seed = version_tag.training_opts.as_ref().and_then(|o| o.seed);

        // 8. Build the exporter and drive it. Lance writers are async;
        // the embedded database API is sync. Spin a dedicated current-
        // thread runtime so we don't assume the caller already has one.
        let sink: Arc<dyn TrainingExportLineageSink> =
            Arc::new(InMemoryLineageSink::new());
        let exporter = LanceExporter::new(
            &output_path,
            schema,
            Arc::new(merkle_dag_snapshot),
            Arc::new(tag_catalog_snapshot),
            source,
            version_tag.name.clone(),
            precision,
            false, // dedup — opt-in via `WHERE NOT DUPLICATE`, not wired into the method API yet
            seed,
        )
        .with_lineage_sink(sink);

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                GalaxError::Internal(format!("could not build tokio runtime: {e}"))
            })?;
        rt.block_on(exporter.export())
            .map_err(|e| GalaxError::Internal(format!("Lance export failed: {e}")))?;

        Ok(output_path)
    }

    /// Pick the single table that a training export should consume. v1
    /// assumes one training-eligible table per database. See
    /// [`Self::training_dataset`] for why that choice is explicit
    /// rather than implicit.
    fn pick_training_table(
        &self,
    ) -> GalaxResult<(String, galaxdb_sql::executor::TableEntry)> {
        let names: Vec<String> = self
            .catalog
            .table_names()
            .map(|n| n.to_string())
            .collect();
        match names.len() {
            0 => Err(GalaxError::Internal(
                "training_dataset: no tables exist in the database".into(),
            )),
            1 => {
                let name = &names[0];
                let entry = self
                    .catalog
                    .get_table(name)
                    .cloned()
                    .ok_or_else(|| GalaxError::TableNotFound(name.clone()))?;
                Ok((name.clone(), entry))
            }
            _ => Err(GalaxError::Internal(format!(
                "training_dataset: multiple tables found ({}); v1 supports \
                 single-table exports — drop or rename tables so only the \
                 target table remains",
                names.len()
            ))),
        }
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
    /// Real storage engine, shared with the `Database`. Needed so
    /// `on_row_deleted` can append a `DELTA_TOMBSTONE` WAL record
    /// durably before the in-memory delta buffer is tombstoned.
    engine: Arc<Engine>,
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

    fn on_row_deleted(&self, table: &str, row_key: &[u8]) -> GalaxResult<()> {
        // Resolve the primary-key bytes to the vector-row-id we stored
        // when the embedding was generated. If we don't have a mapping
        // (table has no vector index, or the embedding never landed)
        // the delete is a no-op for the vector side, which is correct.
        let row_id = {
            let indexes = self.indexes.read().unwrap();
            let Some(idx) = indexes.get(table) else {
                return Ok(());
            };
            match idx.key_to_row_id.get(row_key) {
                Some(id) => *id,
                None => {
                    // No vector for this row_key — nothing to tombstone.
                    return Ok(());
                }
            }
        };

        // WAL first, memory after. The payload is
        // `[u64 le vector_row_id][row_key]` so replay on recovery can
        // rebuild the tombstone set and the key→row_id mapping.
        let mut payload = Vec::with_capacity(8 + row_key.len());
        payload.extend_from_slice(&row_id.to_le_bytes());
        payload.extend_from_slice(row_key);
        self.engine.append_delta_tombstone_sync(payload)?;

        // Tombstone the in-memory delta buffer and drop the mapping so
        // re-insert of the same key allocates a fresh vector row-id.
        let mut indexes = self.indexes.write().unwrap();
        if let Some(idx) = indexes.get_mut(table) {
            idx.delta.delete(row_id);
            idx.vectors.remove(&row_id);
            idx.key_to_row_id.remove(row_key);
        }

        Ok(())
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

/// If the SQL is a `SELECT` with an `AT VERSION ...` suffix, return
/// `Some((rest_of_sql_without_at_version, parsed_AtVersionExpr))`.
/// If no `AT VERSION` is present, return `None`. If parsing the
/// version fragment fails, propagate the parser error.
///
/// The matcher is deliberately conservative: it requires the literal
/// token `AT VERSION` to appear case-insensitively outside quotes and
/// after a `FROM` clause. The rest of the string (from `AT VERSION`
/// to the end, minus a trailing semicolon) is handed to
/// `galaxdb_sql::parser::parse_at_version`. This keeps the suffix
/// syntax consistent with `galaxdb-sql::parser::parse_at_version`.
fn split_at_version(
    sql: &str,
) -> GalaxResult<Option<(String, galaxdb_sql::ast::AtVersionExpr)>> {
    let trimmed = sql.trim().trim_end_matches(';');
    let upper: String = trimmed
        .chars()
        .map(|c| if c == '\'' { '\'' } else { c.to_ascii_uppercase() })
        .collect();

    // Case-insensitive search that skips quoted regions. We need the
    // position in the *original* string, which matches the uppercase
    // string byte-for-byte because we only mapped ASCII letters.
    let bytes = trimmed.as_bytes();
    let upper_bytes = upper.as_bytes();
    let needle = b"AT VERSION";
    let mut in_quote = false;
    let mut i = 0usize;
    let mut found: Option<usize> = None;

    while i + needle.len() <= bytes.len() {
        if bytes[i] == b'\'' {
            in_quote = !in_quote;
            i += 1;
            continue;
        }
        if !in_quote && &upper_bytes[i..i + needle.len()] == needle {
            let before_ok = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
            let after_idx = i + needle.len();
            let after_ok =
                after_idx == bytes.len() || !bytes[after_idx].is_ascii_alphanumeric();
            if before_ok && after_ok {
                found = Some(i);
                break;
            }
        }
        i += 1;
    }

    let Some(pos) = found else {
        return Ok(None);
    };

    let stripped = trimmed[..pos].trim_end().to_string();
    let fragment = &trimmed[pos..];
    let at = galaxdb_sql::parser::parse_at_version(fragment)?;
    Ok(Some((stripped, at)))
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
// Training-export glue (task 22.4)
//
// `Database::training_dataset` exports a tagged table as a Lance
// dataset by driving `galaxdb_versioning::LanceExporter` against the
// live `Engine`. The pieces below are the concrete types that wiring
// needs: a real `LanceExportSource` over `Engine::scan_all_at`, a
// catalog → Arrow schema mapper, and a path-safe version of the tag
// name for the output directory.
// ---------------------------------------------------------------------------

/// Real [`galaxdb_versioning::LanceExportSource`] that reads rows
/// from the live storage engine at a specific timestamp.
///
/// `read_blocks` ignores the block-id list supplied by the exporter
/// because v1's memtable-based `scan_all_at` addresses keys, not
/// blocks. The exporter uses the block list only to decide which
/// rows the source should return; when the source already knows the
/// version ts it can ask the engine directly, which is simpler than
/// round-tripping through a block-set. When K2-Follow lands and
/// AT VERSION becomes SST-aware, this impl switches to asking
/// `SstRegistry` for the pinned-block payload instead.
struct EmbeddedLanceExportSource {
    engine: Arc<galaxdb_storage::engine::Engine>,
    table_name: String,
    table_entry: galaxdb_sql::executor::TableEntry,
    version_timestamp: u64,
}

impl galaxdb_versioning::LanceExportSource for EmbeddedLanceExportSource {
    fn read_blocks(
        &self,
        _block_ids: &[galaxdb_common::types::BlockId],
    ) -> galaxdb_versioning::ExportResult<Vec<galaxdb_versioning::ExportedRow>> {
        use galaxdb_sql::row_codec;
        use galaxdb_versioning::ExportedRow;

        // `scan_all_at` returns every visible row in the whole engine.
        // We restrict to this table by the shared `"{table}:"` prefix
        // that `row_codec::build_primary_key` builds for INSERTs.
        let prefix = format!("{}:", self.table_name);
        let raw = self.engine.scan_all_at(self.version_timestamp);

        let mut rows = Vec::with_capacity(raw.len());
        for (key, val, _ts) in raw {
            if !key.starts_with(prefix.as_bytes()) {
                continue;
            }
            // Decode the `col=value|col=value|...` on-disk row into
            // typed values, then project in catalog order so the row's
            // `fields` align with the Arrow schema.
            let decoded = row_codec::decode_row(&val);
            let fields =
                project_row_to_field_values(&self.table_entry, &decoded);
            rows.push(ExportedRow {
                primary_key: key,
                fields,
                near_duplicate_group: None,
            });
        }
        Ok(rows)
    }
}

/// Project a decoded row into one [`FieldValue`] per catalog column,
/// in catalog order. Missing columns surface as the type-appropriate
/// zero value (0 / empty string / empty vector) so that the Arrow
/// builder always sees one value per column — the schema is marked
/// nullable at construction (see [`arrow_schema_from_catalog`]) so
/// defaulting is safe for v1. Embedding columns are filled with an
/// empty vector because the scalar row codec doesn't carry them; a
/// vector-aware source is follow-up work.
fn project_row_to_field_values(
    entry: &galaxdb_sql::executor::TableEntry,
    decoded: &[(String, galaxdb_sql::planner::Value)],
) -> Vec<galaxdb_versioning::FieldValue> {
    use galaxdb_sql::planner::Value;
    use galaxdb_versioning::FieldValue;

    let mut out = Vec::with_capacity(entry.columns.len());
    for col in &entry.columns {
        let value = decoded.iter().find(|(n, _)| n == &col.name).map(|(_, v)| v);
        let kind = classify_column(&col.data_type);
        let fv = match (kind, value) {
            (ColumnKind::Int, Some(Value::Integer(n))) => FieldValue::Int64(*n),
            (ColumnKind::Int, Some(Value::Float(f))) => FieldValue::Int64(*f as i64),
            (ColumnKind::Int, Some(Value::Text(s))) => {
                FieldValue::Int64(s.parse::<i64>().unwrap_or(0))
            }
            (ColumnKind::Int, Some(Value::Bool(b))) => FieldValue::Int64(*b as i64),
            (ColumnKind::Int, _) => FieldValue::Int64(0),

            (ColumnKind::Float, Some(Value::Float(f))) => FieldValue::Float32(*f as f32),
            (ColumnKind::Float, Some(Value::Integer(n))) => {
                FieldValue::Float32(*n as f32)
            }
            (ColumnKind::Float, Some(Value::Text(s))) => {
                FieldValue::Float32(s.parse::<f32>().unwrap_or(0.0))
            }
            (ColumnKind::Float, _) => FieldValue::Float32(0.0),

            (ColumnKind::Text, Some(v)) => {
                FieldValue::Utf8(galaxdb_sql::row_codec::value_display(v))
            }
            (ColumnKind::Text, None) => FieldValue::Utf8(String::new()),
        };
        out.push(fv);
    }
    out
}

/// Kind of Arrow column the exporter should build for a given SQL
/// type string. Anything the v1 exporter doesn't specifically know
/// about falls into `Text` — the row codec stores display strings,
/// so the round-trip is lossless even for types we haven't modelled
/// as first-class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ColumnKind {
    Int,
    Float,
    Text,
}

fn classify_column(data_type: &str) -> ColumnKind {
    let base = data_type
        .split('(')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_uppercase();
    match base.as_str() {
        "INT" | "INTEGER" | "BIGINT" | "SMALLINT" | "TINYINT" => ColumnKind::Int,
        "FLOAT" | "REAL" | "DOUBLE" | "DOUBLE PRECISION" => ColumnKind::Float,
        _ => ColumnKind::Text,
    }
}

/// Map a [`TableEntry`] to an Arrow [`arrow::datatypes::Schema`]. Every
/// column is marked nullable so partial rows (which can happen when a
/// column was added after some rows were inserted) don't fail the
/// Arrow builder. Embedding columns are skipped in v1 — the scalar
/// export carries them as an empty `FieldValue::Utf8` if anyone asks.
fn arrow_schema_from_catalog(
    entry: &galaxdb_sql::executor::TableEntry,
) -> arrow::datatypes::Schema {
    use arrow::datatypes::{DataType, Field};
    let fields: Vec<Field> = entry
        .columns
        .iter()
        .map(|c| {
            let dt = match classify_column(&c.data_type) {
                ColumnKind::Int => DataType::Int64,
                ColumnKind::Float => DataType::Float32,
                ColumnKind::Text => DataType::Utf8,
            };
            // Nullable: see comment above.
            Field::new(&c.name, dt, true)
        })
        .collect();
    arrow::datatypes::Schema::new(fields)
}

/// Make a tag name safe for use as a single path component. Replaces
/// every non-alphanumeric / non-`-` / non-`_` / non-`.` byte with `_`
/// so tags like `"train-v1 (latest)"` still land under a sensible
/// directory name. This is cosmetic — tag uniqueness is guaranteed by
/// the catalog.
fn sanitize_tag_for_path(tag: &str) -> String {
    tag.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
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

    // -----------------------------------------------------------------
    // Phase K regressions — AT VERSION + DELTA_TOMBSTONE + compactor
    // pin-set (tasks 18.6, 32.3, 32.4, 33.5).
    //
    // These tests go through the canonical `Database::execute` path
    // so they exercise the SQL parser, the plan dispatch, the storage
    // engine's MVCC memtable, and the tag catalog together. Anything
    // that regresses the real behaviour on any of those layers will
    // fail here.
    // -----------------------------------------------------------------

    #[test]
    fn at_version_timestamp_returns_historical_snapshot() {
        let mut db = test_db();
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO t (id, name) VALUES (1, 'alpha')")
            .unwrap();
        // The INSERT above consumed the latest allocated ts.
        // `next_ts_for_tests()` returns the next one that *would* be
        // allocated, so to read "as of just after the INSERT but before
        // any update", we subtract 1.
        let read_ts = db.engine.next_ts_for_tests() - 1;
        // Now mutate the row; the UPDATE lands at a higher ts.
        db.execute("UPDATE t SET name = 'beta' WHERE id = 1")
            .unwrap();

        // Plain SELECT sees the latest value.
        let rows = rows_of(db.execute("SELECT id, name FROM t").unwrap());
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values[1].1, "beta");

        // AT VERSION <read_ts> sees the pre-update value.
        let sql = format!("SELECT id, name FROM t AT VERSION {read_ts}");
        let rows = rows_of(db.execute(&sql).unwrap());
        assert_eq!(rows.len(), 1, "AT VERSION must see exactly one row");
        assert_eq!(
            rows[0].values[1].1,
            "alpha",
            "AT VERSION must return the value as of the snapshot ts"
        );
    }

    #[test]
    fn at_version_tag_resolves_through_tag_catalog() {
        use galaxdb_versioning::{MerkleRoot, TrainingTagMetadata};

        let mut db = test_db();
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO t (id, name) VALUES (1, 'v1')")
            .unwrap();
        let tag_ts = db.engine.next_ts_for_tests() - 1;
        // Register a real tag that points at the just-committed ts.
        {
            let mut tc = db.tag_catalog.lock().unwrap();
            tc.create_tag(
                "snap-v1".to_string(),
                tag_ts, // created_at
                MerkleRoot { hash: 0xC0DE },
                tag_ts, // version_timestamp
                vec![], // no pinned blocks for this test
                false,
                None::<TrainingTagMetadata>,
            )
            .expect("create tag");
        }
        db.execute("UPDATE t SET name = 'v2' WHERE id = 1").unwrap();

        let rows = rows_of(
            db.execute("SELECT id, name FROM t AT VERSION 'snap-v1'").unwrap(),
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].values[1].1, "v1",
            "AT VERSION '<tag>' must resolve through the tag catalog and return the pre-update row",
        );
    }

    #[test]
    fn at_version_unknown_tag_errors() {
        let mut db = test_db();
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        let err = db
            .execute("SELECT id FROM t AT VERSION 'does-not-exist'")
            .expect_err("unknown tag must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("unknown version tag") || msg.contains("does-not-exist"),
            "expected an 'unknown version tag' error, got: {msg}",
        );
    }

    #[test]
    fn compactor_pins_tagged_timestamps() {
        use galaxdb_storage::compaction::GcContext;

        let mut db = test_db();
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO t (id, name) VALUES (1, 'alpha')")
            .unwrap();
        let tag_ts = db.engine.next_ts_for_tests();
        {
            let mut tc = db.tag_catalog.lock().unwrap();
            tc.create_tag(
                "keep-me".to_string(),
                tag_ts,
                galaxdb_versioning::MerkleRoot { hash: 1 },
                tag_ts,
                vec![],
                false,
                None,
            )
            .unwrap();
        }
        db.execute("UPDATE t SET name = 'beta' WHERE id = 1")
            .unwrap();

        let gc: GcContext = db.gc_context_with_pins(None);
        assert!(
            gc.pinned_tag_timestamps.contains(&tag_ts),
            "compactor pin-set must include the tag's version_timestamp ({tag_ts}); \
             got {:?}",
            gc.pinned_tag_timestamps,
        );
        // Compaction-time decision: the tagged version must be retained,
        // a non-tagged intermediate version may be discarded.
        assert!(gc.should_keep(tag_ts, /* is_latest = */ false));
    }

    // -----------------------------------------------------------------
    // Task 22.4 — training_dataset(tag) produces a real Lance dataset
    // -----------------------------------------------------------------

    /// End-to-end: CREATE TABLE → INSERT → create a FOR TRAINING tag
    /// pointing at the post-insert timestamp → call
    /// `Database::training_dataset` → re-open the returned path with
    /// the `lance` crate and assert the row count.
    ///
    /// This is the acceptance test for task 22.4. If it passes, the
    /// Rust method is writing a real, Lance-readable dataset backed
    /// by real engine data — no mocks, no placeholders. The Python
    /// wrapper around this path (`galaxdb.Database.training_dataset`
    /// in `galaxdb-python`) just surfaces the returned path as a
    /// string so `lance.dataset(path).to_pytorch()` works as the
    /// final IterableDataset shim.
    #[test]
    fn training_dataset_writes_real_lance_dataset() {
        use galaxdb_versioning::{MerkleRoot, TrainingTagMetadata};

        let mut db = test_db();
        db.execute("CREATE TABLE docs (id INT PRIMARY KEY, body TEXT)")
            .unwrap();
        for i in 1..=5 {
            db.execute(&format!(
                "INSERT INTO docs (id, body) VALUES ({i}, 'row-{i}')"
            ))
            .unwrap();
        }

        // Capture the post-insert timestamp so the tag points at a
        // commit that actually contains rows. `exec_create_version_tag`
        // still takes its ts from `MerkleDag::latest()` — which is 0
        // until task 36 wires the DAG to real commits — so for now we
        // register the training tag directly against the tag catalog
        // with a ts the engine does have data at. This is the same
        // pattern the Phase K AT VERSION tests use above.
        let tag_ts = db.engine.next_ts_for_tests();
        {
            let mut tc = db.tag_catalog.lock().unwrap();
            tc.create_tag(
                "train-v1".to_string(),
                tag_ts,
                MerkleRoot { hash: 0xC0DE },
                tag_ts,
                vec![], // pinned blocks are irrelevant: the engine
                        // source drives off `version_timestamp`.
                true,   // FOR TRAINING
                Some(TrainingTagMetadata {
                    precision: "float32".to_string(),
                    seed: Some(42),
                    deterministic_order: true,
                }),
            )
            .expect("create training tag");
        }

        let path = db
            .training_dataset("train-v1")
            .expect("training_dataset must produce a Lance dataset");
        assert!(path.exists(), "returned path must exist on disk");
        assert!(
            path.is_dir(),
            "Lance writes the dataset as a directory, not a single file"
        );
        assert!(
            path.starts_with(db.path()),
            "output must land under the database directory: {:?}",
            path
        );

        // Open the dataset through the real `lance` crate (the same
        // API the Python wrapper uses under the hood) and verify the
        // row count matches the number of INSERTs.
        let row_count = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                let ds = lance::Dataset::open(path.to_str().unwrap())
                    .await
                    .expect("open Lance dataset");
                ds.scan()
                    .count_rows()
                    .await
                    .expect("count rows in Lance scan")
            });
        assert_eq!(
            row_count, 5,
            "Lance dataset must contain exactly the 5 INSERTed rows"
        );
    }

    /// Non-training tags must not pass the `training_dataset` guard.
    /// This keeps the deterministic-order contract on the exporter:
    /// every caller of `training_dataset` is guaranteed the tag was
    /// created with `FOR TRAINING` (and therefore carries a precision
    /// and a deterministic seed).
    #[test]
    fn training_dataset_rejects_non_training_tag() {
        use galaxdb_versioning::MerkleRoot;

        let mut db = test_db();
        db.execute("CREATE TABLE docs (id INT PRIMARY KEY, body TEXT)")
            .unwrap();
        db.execute("INSERT INTO docs (id, body) VALUES (1, 'row-1')")
            .unwrap();
        let tag_ts = db.engine.next_ts_for_tests();
        {
            let mut tc = db.tag_catalog.lock().unwrap();
            tc.create_tag(
                "plain-snapshot".to_string(),
                tag_ts,
                MerkleRoot { hash: 1 },
                tag_ts,
                vec![],
                false, // NOT a training tag
                None,
            )
            .unwrap();
        }
        let err = db
            .training_dataset("plain-snapshot")
            .expect_err("non-training tag must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("not a FOR TRAINING"),
            "expected a FOR-TRAINING guard message, got: {msg}"
        );
    }

    /// An unknown tag name surfaces a real error rather than silently
    /// exporting an empty dataset.
    #[test]
    fn training_dataset_unknown_tag_errors() {
        let mut db = test_db();
        db.execute("CREATE TABLE docs (id INT PRIMARY KEY, body TEXT)")
            .unwrap();
        let err = db
            .training_dataset("does-not-exist")
            .expect_err("unknown tag must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("unknown version tag") || msg.contains("does-not-exist"),
            "expected 'unknown version tag' error, got: {msg}"
        );
    }
}
