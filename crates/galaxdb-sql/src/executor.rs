//! Query executor — executes query plans against the storage engine.
//!
//! The executor is the bridge between the SQL layer and the storage engine.
//! It translates query plans into storage operations (memtable writes, ART
//! lookups, PAX block reads, etc.).
//!
//! For SEMANTIC_MATCH queries, the executor delegates to a `VectorSearchBackend`
//! trait which abstracts the HNSW + delta buffer + sidecar pipeline.

use galaxdb_common::{GalaxError, GalaxResult};
use galaxdb_versioning::{MinHashDedup, SIGNATURE_BYTES};

use crate::ast::SemanticMatchExpr;
use crate::planner::*;

/// A single row returned by a query.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub columns: Vec<(String, Value)>,
}

/// Result of executing a query plan.
#[derive(Debug, Clone, PartialEq)]
pub enum ExecuteResult {
    /// Rows returned (SELECT, SHOW).
    Rows {
        columns: Vec<String>,
        rows: Vec<Row>,
    },
    /// Row count affected (INSERT, UPDATE, DELETE).
    RowCount(u64),
    /// DDL completed (CREATE TABLE, DROP TABLE).
    Ok(String),
    /// Error with message.
    Error(String),
}

/// Result from a vector search operation.
#[derive(Debug, Clone)]
pub struct VectorSearchResult {
    pub row_id: u64,
    pub similarity: f32,
}

/// Trait for the vector search backend (HNSW + delta buffer + sidecar).
///
/// The executor calls this to perform SEMANTIC_MATCH queries. The implementation
/// handles: query text → embedding (via sidecar), HNSW search, delta buffer
/// union, re-ranking, and threshold filtering.
pub trait VectorSearchBackend {
    /// Execute a semantic search: embed the query text, search HNSW + delta buffer,
    /// re-rank, apply threshold, return top-k results.
    ///
    /// Returns Err if the embedding sidecar is unavailable.
    fn semantic_search(
        &self,
        table: &str,
        query_text: &str,
        threshold: f64,
        k: usize,
        strategy: SearchStrategy,
    ) -> Result<Vec<VectorSearchResult>, String>;

    /// Execute a brute-force filtered search over a pre-filtered candidate set.
    fn brute_force_filtered(
        &self,
        table: &str,
        query_text: &str,
        threshold: f64,
        k: usize,
        filter: &FilterExpr,
    ) -> Result<Vec<VectorSearchResult>, String>;
}

/// A no-op vector backend that returns "sidecar unavailable" errors.
/// Used when no vector search is configured.
pub struct NoOpVectorBackend;

impl VectorSearchBackend for NoOpVectorBackend {
    fn semantic_search(
        &self, _table: &str, _query_text: &str, _threshold: f64, _k: usize, _strategy: SearchStrategy,
    ) -> Result<Vec<VectorSearchResult>, String> {
        Err("semantic search temporarily unavailable — embedding sidecar is down".to_string())
    }

    fn brute_force_filtered(
        &self, _table: &str, _query_text: &str, _threshold: f64, _k: usize, _filter: &FilterExpr,
    ) -> Result<Vec<VectorSearchResult>, String> {
        Err("semantic search temporarily unavailable — embedding sidecar is down".to_string())
    }
}

/// Catalog entry for a table.
#[derive(Debug, Clone)]
pub struct TableEntry {
    pub name: String,
    pub columns: Vec<CatalogColumn>,
    pub has_embedding: bool,
}

/// Column metadata in the catalog.
#[derive(Debug, Clone)]
pub struct CatalogColumn {
    pub name: String,
    pub data_type: String,
    pub nullable: bool,
    pub primary_key: bool,
    pub is_embedding_source: bool,
}

/// In-memory catalog tracking table metadata.
#[derive(Debug, Default)]
pub struct Catalog {
    tables: std::collections::HashMap<String, TableEntry>,
}

impl Catalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create_table(&mut self, name: String, entry: TableEntry) -> GalaxResult<()> {
        if self.tables.contains_key(&name) {
            return Err(GalaxError::TableAlreadyExists(name));
        }
        self.tables.insert(name, entry);
        Ok(())
    }

    pub fn drop_table(&mut self, name: &str) -> GalaxResult<TableEntry> {
        self.tables
            .remove(name)
            .ok_or_else(|| GalaxError::TableNotFound(name.to_string()))
    }

    pub fn get_table(&self, name: &str) -> Option<&TableEntry> {
        self.tables.get(name)
    }

    pub fn table_exists(&self, name: &str) -> bool {
        self.tables.contains_key(name)
    }

    pub fn table_count(&self) -> usize {
        self.tables.len()
    }
}

/// Execute a query plan against the catalog.
///
/// The `vector_backend` provides SEMANTIC_MATCH execution (HNSW + delta buffer + sidecar).
/// Pass `&NoOpVectorBackend` if vector search is not configured.
pub fn execute(plan: &QueryPlan, catalog: &mut Catalog, vector_backend: &dyn VectorSearchBackend) -> ExecuteResult {
    execute_with_policies(plan, catalog, vector_backend, None)
}

/// Execute a query plan with optional side-channel policies such as MinHash
/// near-duplicate signature computation on INSERT (task 35.2).
///
/// Callers that have no policy to apply should use [`execute`] instead.
pub fn execute_with_policies(
    plan: &QueryPlan,
    catalog: &mut Catalog,
    vector_backend: &dyn VectorSearchBackend,
    minhash_policy: Option<&MinHashPolicy>,
) -> ExecuteResult {
    match plan {
        QueryPlan::CreateTable(stmt) => execute_create_table(stmt, catalog),
        QueryPlan::DropTable { name, if_exists } => {
            execute_drop_table(name, *if_exists, catalog)
        }
        QueryPlan::Insert {
            table,
            columns,
            values,
        } => execute_insert(table, columns, values, catalog, minhash_policy),
        QueryPlan::Update {
            table,
            assignments,
            filter,
        } => execute_update(table, assignments, filter, catalog),
        QueryPlan::Delete { table, filter } => execute_delete(table, filter, catalog),
        QueryPlan::FullScan {
            table,
            filter,
            columns,
        } => execute_select(table, columns, filter, catalog),
        QueryPlan::Analyze { table } => {
            if !catalog.table_exists(table) {
                return ExecuteResult::Error(format!("table not found: {}", table));
            }
            ExecuteResult::Ok(format!("ANALYZE {}", table))
        }
        QueryPlan::Backup { path } => ExecuteResult::Ok(format!("BACKUP TO '{}'", path)),
        QueryPlan::Restore { path } => ExecuteResult::Ok(format!("RESTORE FROM '{}'", path)),
        QueryPlan::ShowEmbeddingHealth { table } => {
            let msg = match table {
                Some(t) => format!("SHOW EMBEDDING HEALTH FOR {}", t),
                None => "SHOW EMBEDDING HEALTH".to_string(),
            };
            ExecuteResult::Rows {
                columns: vec!["status".to_string()],
                rows: vec![Row {
                    columns: vec![("status".to_string(), Value::Text(msg))],
                }],
            }
        }
        QueryPlan::CreateVersionTag(stmt) => {
            ExecuteResult::Ok(format!("CREATE VERSION TAG '{}'", stmt.name))
        }
        QueryPlan::BulkInsert { table } => {
            if !catalog.table_exists(table) {
                return ExecuteResult::Error(format!("table not found: {}", table));
            }
            ExecuteResult::Ok(format!("BULK INSERT INTO {}", table))
        }
        QueryPlan::SemanticSearch {
            table,
            query_text,
            threshold,
            strategy,
            ..
        } => {
            if !catalog.table_exists(table) {
                return ExecuteResult::Error(format!("table not found: {}", table));
            }
            execute_semantic_search(table, query_text, *threshold, *strategy, vector_backend)
        }
        QueryPlan::HybridSearch {
            table,
            filter,
            semantic,
            strategy,
        } => {
            if !catalog.table_exists(table) {
                return ExecuteResult::Error(format!("table not found: {}", table));
            }
            execute_hybrid_search(table, semantic, filter, *strategy, vector_backend)
        }
        QueryPlan::PointLookup { table, .. } => {
            if !catalog.table_exists(table) {
                return ExecuteResult::Error(format!("table not found: {}", table));
            }
            ExecuteResult::Rows {
                columns: vec![],
                rows: vec![],
            }
        }
    }
}

fn execute_semantic_search(
    table: &str,
    query_text: &str,
    threshold: f64,
    strategy: SearchStrategy,
    vector_backend: &dyn VectorSearchBackend,
) -> ExecuteResult {
    match vector_backend.semantic_search(table, query_text, threshold, 10, strategy) {
        Ok(results) => {
            let rows: Vec<Row> = results
                .iter()
                .map(|r| Row {
                    columns: vec![
                        ("id".to_string(), Value::Integer(r.row_id as i64)),
                        ("score".to_string(), Value::Float(r.similarity as f64)),
                    ],
                })
                .collect();
            ExecuteResult::Rows {
                columns: vec!["id".to_string(), "score".to_string()],
                rows,
            }
        }
        Err(msg) => ExecuteResult::Error(msg),
    }
}

fn execute_hybrid_search(
    table: &str,
    semantic: &SemanticMatchExpr,
    filter: &FilterExpr,
    strategy: SearchStrategy,
    vector_backend: &dyn VectorSearchBackend,
) -> ExecuteResult {
    let result = match strategy {
        SearchStrategy::BruteForceFiltered => {
            vector_backend.brute_force_filtered(
                table,
                &semantic.query,
                semantic.threshold,
                10,
                filter,
            )
        }
        SearchStrategy::HnswWithPostFilter => {
            vector_backend.semantic_search(
                table,
                &semantic.query,
                semantic.threshold,
                10,
                strategy,
            )
        }
    };

    match result {
        Ok(results) => {
            let rows: Vec<Row> = results
                .iter()
                .map(|r| Row {
                    columns: vec![
                        ("id".to_string(), Value::Integer(r.row_id as i64)),
                        ("score".to_string(), Value::Float(r.similarity as f64)),
                    ],
                })
                .collect();
            ExecuteResult::Rows {
                columns: vec!["id".to_string(), "score".to_string()],
                rows,
            }
        }
        Err(msg) => ExecuteResult::Error(msg),
    }
}

fn execute_create_table(
    stmt: &crate::ast::CreateTableStmt,
    catalog: &mut Catalog,
) -> ExecuteResult {
    let columns: Vec<CatalogColumn> = stmt
        .columns
        .iter()
        .map(|c| CatalogColumn {
            name: c.name.clone(),
            data_type: c.data_type.clone(),
            nullable: c.nullable,
            primary_key: c.primary_key,
            is_embedding_source: c.embedding.is_some(),
        })
        .collect();

    let has_embedding = columns.iter().any(|c| c.is_embedding_source);

    let entry = TableEntry {
        name: stmt.table_name.clone(),
        columns,
        has_embedding,
    };

    match catalog.create_table(stmt.table_name.clone(), entry) {
        Ok(()) => ExecuteResult::Ok(format!("CREATE TABLE {}", stmt.table_name)),
        Err(e) => ExecuteResult::Error(format!("{}", e)),
    }
}

fn execute_drop_table(name: &str, if_exists: bool, catalog: &mut Catalog) -> ExecuteResult {
    match catalog.drop_table(name) {
        Ok(_) => ExecuteResult::Ok(format!("DROP TABLE {}", name)),
        Err(_) if if_exists => ExecuteResult::Ok(format!("DROP TABLE IF EXISTS {}", name)),
        Err(e) => ExecuteResult::Error(format!("{}", e)),
    }
}

fn execute_insert(
    table: &str,
    columns: &[String],
    values: &[Value],
    catalog: &Catalog,
    minhash_policy: Option<&MinHashPolicy>,
) -> ExecuteResult {
    if !catalog.table_exists(table) {
        return ExecuteResult::Error(format!("table not found: {}", table));
    }

    // Check column count matches
    if !columns.is_empty() && columns.len() != values.len() {
        return ExecuteResult::Error(format!(
            "column count ({}) does not match value count ({})",
            columns.len(),
            values.len()
        ));
    }

    // Task 35.2: compute MinHash signatures for TEXT columns and hand them
    // to the sink. When the caller didn't install a policy (legacy `execute`)
    // this path is skipped entirely.
    if let Some(policy) = minhash_policy {
        // Safe to unwrap: table_exists was checked above.
        let table_entry = catalog.get_table(table).unwrap();
        policy.compute_and_sink(table, table_entry, columns, values);
    }

    // In the full implementation, this would write to memtable + WAL + ART
    ExecuteResult::RowCount(1)
}

fn execute_update(
    table: &str,
    assignments: &[(String, Value)],
    _filter: &Option<FilterExpr>,
    catalog: &Catalog,
) -> ExecuteResult {
    if !catalog.table_exists(table) {
        return ExecuteResult::Error(format!("table not found: {}", table));
    }

    let table_entry = catalog.get_table(table).unwrap();

    // Check if any assignment targets an embedding source column (Req 15.5)
    for (col_name, _) in assignments {
        if let Some(col) = table_entry.columns.iter().find(|c| &c.name == col_name) {
            if col.is_embedding_source {
                return ExecuteResult::Error(format!(
                    "cannot update embedding source column '{}'; use DELETE + INSERT instead",
                    col_name
                ));
            }
        }
    }

    // In the full implementation, this would write new MVCC version
    ExecuteResult::RowCount(0)
}

fn execute_delete(
    table: &str,
    _filter: &Option<FilterExpr>,
    catalog: &Catalog,
) -> ExecuteResult {
    if !catalog.table_exists(table) {
        return ExecuteResult::Error(format!("table not found: {}", table));
    }
    // In the full implementation, this would write tombstone to memtable + WAL
    ExecuteResult::RowCount(0)
}

fn execute_select(
    table: &str,
    _columns: &[String],
    _filter: &Option<FilterExpr>,
    catalog: &Catalog,
) -> ExecuteResult {
    if !catalog.table_exists(table) {
        return ExecuteResult::Error(format!("table not found: {}", table));
    }
    // In the full implementation, this would do ART lookup or scan with zone-map pruning
    ExecuteResult::Rows {
        columns: vec![],
        rows: vec![],
    }
}

// ---------------------------------------------------------------------------
// Task 35.2 — MinHash write-path integration
// ---------------------------------------------------------------------------

/// Does `data_type` name a text-valued SQL type that should be MinHashed?
///
/// Matches `TEXT`, `VARCHAR`, `STRING`, and `CHAR` case-insensitively.
/// Parameterised forms like `VARCHAR(100)` or `CHAR(10)` are accepted — the
/// size parameter is ignored because it doesn't affect MinHash applicability.
pub fn is_text_column(data_type: &str) -> bool {
    // Strip any "(n)" suffix — we only care about the base type name.
    let base = match data_type.find('(') {
        Some(paren) => &data_type[..paren],
        None => data_type,
    };
    matches!(
        base.trim().to_ascii_uppercase().as_str(),
        "TEXT" | "VARCHAR" | "STRING" | "CHAR"
    )
}

/// One system-column write produced alongside a user row during INSERT.
///
/// Task 35.2 emits these for MinHash signatures on TEXT columns. Later tasks
/// (35.4 background grouping, 35.5 `WHERE NOT DUPLICATE`) read them back via
/// the storage engine. Storage integration wires these into PAX system
/// columns.
#[derive(Debug, Clone, PartialEq)]
pub struct SystemColumnWrite {
    /// Table the row belongs to.
    pub table: String,
    /// Which user-visible TEXT column the signature was computed from.
    pub user_column: String,
    /// System column name, e.g. `_minhash_signature__body`.
    pub signature_column: String,
    /// The 512-byte MinHash signature.
    pub signature: [u8; SIGNATURE_BYTES],
}

/// Receives system-column writes produced during INSERT execution.
///
/// Task 35.2 wires MinHash signatures through this trait. The concrete
/// storage integration (later tasks / `galaxdb-embedded`) implements it by
/// appending the signatures as PAX system columns. Tests use
/// [`InMemorySystemColumnSink`], a Vec-backed reference impl.
pub trait SystemColumnSink: Send + Sync {
    /// Record a single system-column write.
    fn write(&self, row: SystemColumnWrite);
}

/// In-memory reference implementation of [`SystemColumnSink`], used by tests
/// and by callers that want to buffer system-column writes without a full
/// storage backend.
#[derive(Debug, Default)]
pub struct InMemorySystemColumnSink {
    entries: std::sync::Mutex<Vec<SystemColumnWrite>>,
}

impl InMemorySystemColumnSink {
    /// Construct an empty sink.
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot the entries recorded so far.
    pub fn entries(&self) -> Vec<SystemColumnWrite> {
        self.entries.lock().unwrap().clone()
    }

    /// Number of entries recorded so far.
    pub fn len(&self) -> usize {
        self.entries.lock().unwrap().len()
    }

    /// Whether no entries have been recorded.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl SystemColumnSink for InMemorySystemColumnSink {
    fn write(&self, row: SystemColumnWrite) {
        self.entries.lock().unwrap().push(row);
    }
}

/// Write-path MinHash policy: computes a 512-byte MinHash signature for every
/// TEXT column on INSERT and forwards the result to a [`SystemColumnSink`].
///
/// Task 35.2 wires this into [`execute_with_policies`]. Non-TEXT columns and
/// `NULL` text values are skipped silently — they are not MinHash candidates.
pub struct MinHashPolicy {
    dedup: std::sync::Arc<MinHashDedup>,
    sink: std::sync::Arc<dyn SystemColumnSink>,
}

impl MinHashPolicy {
    /// Construct a new policy with a deterministic seed and a sink.
    ///
    /// Two policies built with the same seed produce byte-identical
    /// signatures for the same input text — see [`MinHashDedup::new`].
    pub fn new(seed: u64, sink: std::sync::Arc<dyn SystemColumnSink>) -> Self {
        Self {
            dedup: std::sync::Arc::new(MinHashDedup::new(seed)),
            sink,
        }
    }

    /// Compute MinHash signatures for every TEXT column in a row and forward
    /// them to the sink.
    ///
    /// * If `columns` is empty, `values` are assumed to be in table-definition
    ///   order — this matches `INSERT INTO t VALUES (...)` without a column
    ///   list.
    /// * If `columns` is non-empty, it is the user-provided column name
    ///   mapping and overrides the positional assumption.
    /// * Columns whose catalog `data_type` is not a TEXT type (per
    ///   [`is_text_column`]) are skipped.
    /// * Non-text `Value` variants (e.g. `Value::Null` for a nullable TEXT
    ///   column) are skipped.
    pub fn compute_and_sink(
        &self,
        table: &str,
        table_entry: &TableEntry,
        columns: &[String],
        values: &[Value],
    ) {
        // Build the (user_column_name, value) pairs.
        let pairs: Vec<(&str, &Value)> = if columns.is_empty() {
            // Positional: zip table-definition order against values. If the
            // caller supplied fewer values than columns (an error caught
            // elsewhere) we still emit for the prefix that does match — the
            // policy itself is best-effort.
            table_entry
                .columns
                .iter()
                .map(|c| c.name.as_str())
                .zip(values.iter())
                .collect()
        } else {
            columns
                .iter()
                .map(|c| c.as_str())
                .zip(values.iter())
                .collect()
        };

        for (user_column, value) in pairs {
            // Locate the catalog column for this name. Unknown columns are
            // skipped silently — they'll be reported by the validator path.
            let Some(col_meta) = table_entry
                .columns
                .iter()
                .find(|c| c.name == user_column)
            else {
                continue;
            };

            if !is_text_column(&col_meta.data_type) {
                continue;
            }

            let text = match value {
                Value::Text(s) => s,
                _ => continue, // NULL / non-text → nothing to hash
            };

            let signature = self.dedup.signature(text).to_bytes();

            self.sink.write(SystemColumnWrite {
                table: table.to_string(),
                user_column: user_column.to_string(),
                signature_column: format!("_minhash_signature__{user_column}"),
                signature,
            });
        }
    }
}
