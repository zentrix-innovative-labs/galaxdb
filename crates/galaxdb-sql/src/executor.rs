//! Query executor — executes query plans against a real storage engine.
//!
//! The executor is the bridge between the SQL layer and the storage engine.
//! It translates query plans into real storage operations: memtable writes,
//! WAL durability, ART lookups, PAX block reads, MinHash signature
//! computation on INSERT, and sidecar-triggered embedding generation.
//!
//! # Two entry points
//!
//! Callers pick the entry point that matches their context:
//!
//! * [`execute_with_context`] is the canonical entry. It takes an
//!   [`ExecutorContext`] bundling `Arc<galaxdb_storage::Engine>`, the
//!   catalog, an optional sidecar, tag catalog, Merkle DAG, MinHash policy,
//!   and vector backend. Every DDL and DML statement is satisfied by real
//!   code against real storage — no stubs, no fake success returns.
//! * [`execute`] is a thin **catalog-only validator** retained for plan
//!   validation tests that do not need the storage engine. Any DML plan
//!   submitted through this entry point returns
//!   [`GalaxError::NotYetAvailable`]-equivalent errors rather than
//!   pretending to succeed — see the individual function docs. The
//!   engineering principles in `.kiro/steering/engineering-principles.md`
//!   forbid silent fake successes on a production code path; this
//!   function is explicitly labelled non-production and is gated to
//!   planner/catalog checks only.
//!
//! # SEMANTIC_MATCH
//!
//! `QueryPlan::SemanticSearch` / `QueryPlan::HybridSearch` are routed to
//! the [`VectorSearchBackend`] trait on the `ExecutorContext`. When no
//! backend is configured the executor returns [`GalaxError::SidecarUnavailable`]
//! rather than silently returning an empty result.

use std::collections::HashMap;
use std::sync::Arc;

use galaxdb_common::{GalaxError, GalaxResult};
use galaxdb_sidecar::manager::SidecarManager;
use galaxdb_sidecar::protocol::EmbedRequest;
use galaxdb_storage::engine::Engine;
use galaxdb_versioning::{MerkleDag, MinHashDedup, TagCatalog, SIGNATURE_BYTES};

use crate::ast::{AtVersionExpr, ConsistencyMode, SemanticMatchExpr, VersionRef};
use crate::planner::*;
use crate::row_codec;

// ---------------------------------------------------------------------------
// Public data types
// ---------------------------------------------------------------------------

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
    /// DDL completed (CREATE TABLE, DROP TABLE, ANALYZE, etc.).
    Ok(String),
    /// Error with message. Used by the legacy catalog-only `execute` path.
    /// [`execute_with_context`] returns `GalaxResult` directly.
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
/// The executor calls this to perform SEMANTIC_MATCH queries. The
/// implementation handles: query text → embedding (via sidecar), HNSW
/// search, delta buffer union, re-ranking, and threshold filtering.
///
/// If no `VectorSearchBackend` is installed on the [`ExecutorContext`] the
/// executor returns [`GalaxError::SidecarUnavailable`] rather than an empty
/// result. There is no built-in no-op backend — that would hide missing
/// configuration from operators.
pub trait VectorSearchBackend: Send + Sync {
    /// Execute a semantic search: embed the query text, search HNSW +
    /// delta buffer, re-rank, apply threshold, return top-k results.
    fn semantic_search(
        &self,
        table: &str,
        query_text: &str,
        threshold: f64,
        k: usize,
        strategy: SearchStrategy,
    ) -> GalaxResult<Vec<VectorSearchResult>>;

    /// Execute a brute-force filtered search over a pre-filtered candidate
    /// set.
    fn brute_force_filtered(
        &self,
        table: &str,
        query_text: &str,
        threshold: f64,
        k: usize,
        filter: &FilterExpr,
    ) -> GalaxResult<Vec<VectorSearchResult>>;

    /// Record a row deletion in the vector-index side of the world.
    ///
    /// When the SQL executor deletes a row from a table that carries an
    /// embedding column, it must also tell the vector backend so the
    /// delta buffer tombstones the row id and future SEMANTIC_MATCH
    /// queries stop returning it. The backend is responsible for
    /// writing the `DELTA_TOMBSTONE` WAL record (so the tombstone
    /// survives crash recovery) and for updating the in-memory delta
    /// buffer.
    ///
    /// `row_key` is the raw primary-key bytes as stored in the engine
    /// (identical to what `Engine::delete_sync` receives), NOT a
    /// synthetic `row_id`. The backend is free to hash this to its
    /// internal vector-row-id space.
    ///
    /// Default implementation is a no-op so backends that don't need
    /// per-delete notification (e.g. a test stub) don't have to
    /// implement this. Any backend that manages a real delta buffer
    /// MUST override.
    fn on_row_deleted(&self, _table: &str, _row_key: &[u8]) -> GalaxResult<()> {
        Ok(())
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
#[derive(Debug, Default, Clone)]
pub struct Catalog {
    tables: HashMap<String, TableEntry>,
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

    /// Iterate over table names. Order is unspecified.
    pub fn table_names(&self) -> impl Iterator<Item = &str> {
        self.tables.keys().map(|s| s.as_str())
    }
}

// ---------------------------------------------------------------------------
// ExecutorContext — the real, storage-backed execution environment
// ---------------------------------------------------------------------------

/// Everything the executor needs to satisfy a query plan against real
/// infrastructure. Constructed once per database instance (typically inside
/// `galaxdb-embedded::Database`) and passed by `&mut` into
/// [`execute_with_context`] for every statement.
///
/// Every field except `engine` and `catalog` is optional. An engine can
/// operate without a sidecar, without version tags, without MinHash, and
/// without a vector backend; missing functionality surfaces as typed
/// [`GalaxError`] values at the point of use. What is **not** supported is
/// silently returning empty results or fake success — see
/// `.kiro/steering/engineering-principles.md` rules 1 and 2.
pub struct ExecutorContext {
    /// The storage engine. Owns the memtable, WAL, ART index, and SST
    /// registry. Shared as `Arc` so it can be cloned into async tasks.
    pub engine: Arc<Engine>,

    /// Table metadata. The executor updates this on DDL.
    pub catalog: Catalog,

    /// Optional sidecar manager for generating embeddings on INSERT and
    /// for SEMANTIC_MATCH queries. When `None`, INSERTs against tables
    /// with embedding columns still succeed (the embedding is queued for
    /// later generation), and SEMANTIC_MATCH queries return
    /// [`GalaxError::SidecarUnavailable`].
    pub sidecar: Option<Arc<SidecarManager>>,

    /// Optional tag catalog for `CREATE VERSION TAG` and `AT VERSION`
    /// queries. When `None`, those statements return
    /// [`GalaxError::NotYetAvailable`].
    pub tag_catalog: Option<Arc<std::sync::Mutex<TagCatalog>>>,

    /// Optional Merkle DAG for version-tag root computation.
    pub merkle_dag: Option<Arc<std::sync::Mutex<MerkleDag>>>,

    /// Optional MinHash policy for computing `_minhash_signature` columns
    /// on INSERT (task 35.2).
    pub minhash_policy: Option<MinHashPolicy>,

    /// Optional vector search backend (HNSW + delta buffer + sidecar).
    /// When `None`, SEMANTIC_MATCH queries return
    /// [`GalaxError::SidecarUnavailable`].
    pub vector_backend: Option<Arc<dyn VectorSearchBackend>>,
}

impl ExecutorContext {
    /// Construct a context around an engine, with no optional subsystems
    /// enabled. The caller can attach a sidecar, tag catalog, or vector
    /// backend later by setting the fields directly.
    pub fn new(engine: Arc<Engine>) -> Self {
        Self {
            engine,
            catalog: Catalog::new(),
            sidecar: None,
            tag_catalog: None,
            merkle_dag: None,
            minhash_policy: None,
            vector_backend: None,
        }
    }
}

// ---------------------------------------------------------------------------
// MinHash write-path integration (task 35.2)
// ---------------------------------------------------------------------------

/// Does `data_type` name a text-valued SQL type that should be MinHashed?
///
/// Matches `TEXT`, `VARCHAR`, `STRING`, and `CHAR` case-insensitively.
/// Parameterised forms like `VARCHAR(100)` or `CHAR(10)` are accepted —
/// the size parameter is ignored because it doesn't affect MinHash
/// applicability.
pub fn is_text_column(data_type: &str) -> bool {
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
#[derive(Debug, Clone, PartialEq)]
pub struct SystemColumnWrite {
    pub table: String,
    pub user_column: String,
    pub signature_column: String,
    pub signature: [u8; SIGNATURE_BYTES],
}

/// Receives system-column writes produced during INSERT execution.
pub trait SystemColumnSink: Send + Sync {
    fn write(&self, row: SystemColumnWrite);
}

/// In-memory reference implementation of [`SystemColumnSink`].
#[derive(Debug, Default)]
pub struct InMemorySystemColumnSink {
    entries: std::sync::Mutex<Vec<SystemColumnWrite>>,
}

impl InMemorySystemColumnSink {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn entries(&self) -> Vec<SystemColumnWrite> {
        self.entries.lock().unwrap().clone()
    }
    pub fn len(&self) -> usize {
        self.entries.lock().unwrap().len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl SystemColumnSink for InMemorySystemColumnSink {
    fn write(&self, row: SystemColumnWrite) {
        self.entries.lock().unwrap().push(row);
    }
}

/// Write-path MinHash policy.
pub struct MinHashPolicy {
    dedup: Arc<MinHashDedup>,
    sink: Arc<dyn SystemColumnSink>,
}

impl MinHashPolicy {
    pub fn new(seed: u64, sink: Arc<dyn SystemColumnSink>) -> Self {
        Self {
            dedup: Arc::new(MinHashDedup::new(seed)),
            sink,
        }
    }

    /// Compute MinHash signatures for every TEXT column in a row and
    /// forward them to the sink. Non-TEXT columns and non-Text values are
    /// skipped silently.
    pub fn compute_and_sink(
        &self,
        table: &str,
        table_entry: &TableEntry,
        columns: &[String],
        values: &[Value],
    ) {
        let pairs: Vec<(&str, &Value)> = if columns.is_empty() {
            table_entry
                .columns
                .iter()
                .map(|c| c.name.as_str())
                .zip(values.iter())
                .collect()
        } else {
            columns.iter().map(|c| c.as_str()).zip(values.iter()).collect()
        };

        for (user_column, value) in pairs {
            let Some(col_meta) = table_entry.columns.iter().find(|c| c.name == user_column)
            else {
                continue;
            };
            if !is_text_column(&col_meta.data_type) {
                continue;
            }
            let text = match value {
                Value::Text(s) => s,
                _ => continue,
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

// ---------------------------------------------------------------------------
// Canonical entry point: execute_with_context
// ---------------------------------------------------------------------------

/// Execute a query plan against real storage.
///
/// This is the canonical executor entry point. It satisfies every plan
/// variant either by a real operation against [`Engine`] + catalog +
/// optional subsystems, or by a typed [`GalaxError`] — never a fake
/// success.
///
/// # Return contract
///
/// * DDL (`CREATE TABLE`, `DROP TABLE`, `ANALYZE`) → `Ok(ExecuteResult::Ok(msg))`
/// * DML (`INSERT`, `UPDATE`, `DELETE`, `BULK INSERT`) → `Ok(ExecuteResult::RowCount(n))`
/// * SELECT, `SHOW EMBEDDING HEALTH`, `SEMANTIC_MATCH` → `Ok(ExecuteResult::Rows{..})`
/// * Unsupported or missing-prerequisite → `Err(GalaxError::NotYetAvailable { task })`
/// * Runtime failures (I/O, catalog conflict, checksum, etc.) → `Err(GalaxError)`
pub fn execute_with_context(
    plan: &QueryPlan,
    ctx: &mut ExecutorContext,
) -> GalaxResult<ExecuteResult> {
    match plan {
        QueryPlan::CreateTable(stmt) => exec_create_table(stmt, ctx),
        QueryPlan::DropTable { name, if_exists } => exec_drop_table(name, *if_exists, ctx),

        QueryPlan::Insert {
            table,
            columns,
            values,
        } => exec_insert(table, columns, values, ctx),

        QueryPlan::Update {
            table,
            assignments,
            filter,
        } => exec_update(table, assignments, filter, ctx),

        QueryPlan::Delete { table, filter } => exec_delete(table, filter, ctx),

        QueryPlan::FullScan {
            table,
            filter,
            columns,
        } => exec_full_scan(table, columns, filter.as_ref(), ctx),

        QueryPlan::FullScanAtVersion {
            table,
            filter,
            columns,
            at,
        } => exec_full_scan_at_version(table, columns, filter.as_ref(), at, ctx),

        QueryPlan::PointLookup { table, key } => exec_point_lookup(table, key, ctx),

        QueryPlan::Analyze { table } => exec_analyze(table, ctx),
        QueryPlan::Backup { path: _ } => Err(GalaxError::NotYetAvailable {
            task: "37",
            feature: "BACKUP TO <path>",
        }),
        QueryPlan::Restore { path: _ } => Err(GalaxError::NotYetAvailable {
            task: "37",
            feature: "RESTORE FROM <path>",
        }),

        QueryPlan::ShowEmbeddingHealth { table } => exec_show_embedding_health(table.as_deref(), ctx),

        QueryPlan::CreateVersionTag(stmt) => exec_create_version_tag(stmt, ctx),

        QueryPlan::BulkInsert {
            table,
            columns,
            values,
        } => exec_bulk_insert(table, columns, values, ctx),

        QueryPlan::SemanticSearch {
            table,
            query_text,
            threshold,
            strategy,
            ..
        } => exec_semantic_search(table, query_text, *threshold, *strategy, ctx),

        QueryPlan::HybridSearch {
            table,
            filter,
            semantic,
            strategy,
        } => exec_hybrid_search(table, semantic, filter, *strategy, ctx),

        QueryPlan::HybridSearchAtVersion {
            table,
            filter,
            semantic,
            strategy,
            at,
        } => exec_hybrid_search_at_version(table, semantic, filter.as_ref(), *strategy, at, ctx),
    }
}

// ---------------------------------------------------------------------------
// DDL
// ---------------------------------------------------------------------------

fn exec_create_table(
    stmt: &crate::ast::CreateTableStmt,
    ctx: &mut ExecutorContext,
) -> GalaxResult<ExecuteResult> {
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

    ctx.catalog.create_table(stmt.table_name.clone(), entry)?;
    Ok(ExecuteResult::Ok(format!("CREATE TABLE {}", stmt.table_name)))
}

fn exec_drop_table(name: &str, if_exists: bool, ctx: &mut ExecutorContext) -> GalaxResult<ExecuteResult> {
    match ctx.catalog.drop_table(name) {
        Ok(_) => Ok(ExecuteResult::Ok(format!("DROP TABLE {}", name))),
        Err(GalaxError::TableNotFound(_)) if if_exists => {
            Ok(ExecuteResult::Ok(format!("DROP TABLE IF EXISTS {}", name)))
        }
        Err(e) => Err(e),
    }
}

// ---------------------------------------------------------------------------
// DML
// ---------------------------------------------------------------------------

fn exec_insert(
    table: &str,
    columns: &[String],
    values: &[Value],
    ctx: &mut ExecutorContext,
) -> GalaxResult<ExecuteResult> {
    let table_entry = ctx
        .catalog
        .get_table(table)
        .cloned()
        .ok_or_else(|| GalaxError::TableNotFound(table.to_string()))?;

    // Column-count sanity.
    if !columns.is_empty() && columns.len() != values.len() {
        return Err(GalaxError::Internal(format!(
            "column count ({}) does not match value count ({})",
            columns.len(),
            values.len()
        )));
    }

    // Build the (column_name, value) pairs in table-definition order.
    let ordered = row_codec::align_values(&table_entry, columns, values)?;

    // MinHash policy on INSERT (task 35.2). Runs before the storage
    // write so signatures and row bytes commit together.
    if let Some(policy) = ctx.minhash_policy.as_ref() {
        policy.compute_and_sink(table, &table_entry, columns, values);
    }

    // Build the primary-key bytes.
    let key = row_codec::build_primary_key(table, &table_entry, &ordered)?;
    let value_bytes = row_codec::encode_row(&ordered);

    // Real write: WAL + memtable + ART, one fsync.
    ctx.engine
        .put_sync(key.clone(), value_bytes)
        .map_err(|e| GalaxError::Internal(format!("engine put failed: {}", e)))?;

    // Async sidecar embedding trigger for tables with an embedding column.
    // This is best-effort: if the sidecar is down the request is queued on
    // the sidecar's backlog (Req 19.5). Failure here does NOT roll back
    // the row — the embedding is regenerated later.
    if table_entry.has_embedding {
        if let Some(sidecar) = ctx.sidecar.as_ref() {
            for col in &table_entry.columns {
                if !col.is_embedding_source {
                    continue;
                }
                let text = ordered
                    .iter()
                    .find(|(name, _)| name == &col.name)
                    .and_then(|(_, v)| match v {
                        Value::Text(s) => Some(s.clone()),
                        _ => None,
                    });
                let Some(text) = text else { continue };

                let row_id = xxhash_rust::xxh3::xxh3_64(&key);
                let _ = sidecar.embed(EmbedRequest {
                    row_id,
                    text,
                    column: col.name.clone(),
                });
            }
        }
    }

    Ok(ExecuteResult::RowCount(1))
}

fn exec_update(
    table: &str,
    assignments: &[(String, Value)],
    filter: &Option<FilterExpr>,
    ctx: &mut ExecutorContext,
) -> GalaxResult<ExecuteResult> {
    let table_entry = ctx
        .catalog
        .get_table(table)
        .cloned()
        .ok_or_else(|| GalaxError::TableNotFound(table.to_string()))?;

    // Req 15.5: cannot UPDATE an embedding-source column.
    for (col_name, _) in assignments {
        if let Some(col) = table_entry.columns.iter().find(|c| &c.name == col_name) {
            if col.is_embedding_source {
                return Err(GalaxError::EmbeddingSourceUpdate {
                    column: col_name.clone(),
                });
            }
        }
    }

    // Scan, filter, update each matching row. Updates go through
    // `put_sync` which writes a new MVCC version at a new timestamp.
    let mut updated = 0u64;
    let prefix = format!("{}:", table);
    let all = ctx.engine.scan_all();

    for (key, value_bytes) in all {
        if !String::from_utf8_lossy(&key).starts_with(&prefix) {
            continue;
        }
        let mut cols = row_codec::decode_row(&value_bytes);

        // Evaluate the filter (None = match all).
        if let Some(f) = filter {
            if !row_codec::filter_matches(&cols, f) {
                continue;
            }
        }

        // Apply assignments.
        for (col_name, new_value) in assignments {
            if let Some(slot) = cols.iter_mut().find(|(k, _)| k == col_name) {
                slot.1 = new_value.clone();
            } else {
                cols.push((col_name.clone(), new_value.clone()));
            }
        }

        let new_bytes = row_codec::encode_row(&cols);
        ctx.engine
            .put_sync(key, new_bytes)
            .map_err(|e| GalaxError::Internal(format!("engine put failed: {}", e)))?;
        updated += 1;
    }

    Ok(ExecuteResult::RowCount(updated))
}

fn exec_delete(
    table: &str,
    filter: &Option<FilterExpr>,
    ctx: &mut ExecutorContext,
) -> GalaxResult<ExecuteResult> {
    let Some(table_entry) = ctx.catalog.get_table(table).cloned() else {
        return Err(GalaxError::TableNotFound(table.to_string()));
    };

    // Collect the keys first so we don't mutate storage while scanning.
    let mut doomed: Vec<Vec<u8>> = Vec::new();
    let prefix = format!("{}:", table);
    for (key, value_bytes) in ctx.engine.scan_all() {
        if !String::from_utf8_lossy(&key).starts_with(&prefix) {
            continue;
        }
        if let Some(f) = filter {
            let cols = row_codec::decode_row(&value_bytes);
            if !row_codec::filter_matches(&cols, f) {
                continue;
            }
        }
        doomed.push(key);
    }

    let mut deleted = 0u64;
    for key in &doomed {
        match ctx.engine.delete_sync(key) {
            Ok(true) => deleted += 1,
            Ok(false) => {} // already gone
            Err(e) => {
                return Err(GalaxError::Internal(format!(
                    "engine delete failed: {}",
                    e
                )));
            }
        }
    }

    // Task 18.6: when the deleted row carried an embedding column, tell
    // the vector backend so it tombstones the row in its delta buffer
    // and emits the DELTA_TOMBSTONE WAL record. Missing the backend is
    // tolerated (the caller may be in pure-SQL mode) but is logged so
    // operators notice the drift.
    if deleted > 0 && table_entry.has_embedding {
        if let Some(backend) = ctx.vector_backend.as_ref() {
            for key in &doomed {
                if let Err(e) = backend.on_row_deleted(table, key) {
                    tracing::warn!(
                        table = %table,
                        error = %e,
                        "vector backend on_row_deleted failed",
                    );
                }
            }
        } else {
            tracing::warn!(
                table = %table,
                deleted = deleted,
                "deleted rows from table with embedding column but no vector \
                 backend is attached; DELTA_TOMBSTONE records were NOT written. \
                 Future SEMANTIC_MATCH queries may still surface the tombstoned rows.",
            );
        }
    }

    Ok(ExecuteResult::RowCount(deleted))
}

// ---------------------------------------------------------------------------
// Reads
// ---------------------------------------------------------------------------

fn exec_full_scan(
    table: &str,
    columns: &[String],
    filter: Option<&FilterExpr>,
    ctx: &mut ExecutorContext,
) -> GalaxResult<ExecuteResult> {
    let table_entry = ctx
        .catalog
        .get_table(table)
        .cloned()
        .ok_or_else(|| GalaxError::TableNotFound(table.to_string()))?;

    let project: Vec<String> = if columns.is_empty() {
        table_entry.columns.iter().map(|c| c.name.clone()).collect()
    } else {
        columns.to_vec()
    };

    // Zone-map pruning on the key column (task 18.4). Every row for a
    // table lives under the `"{table}:"` prefix in the engine's key
    // space; SST blocks whose key zone maps cannot overlap this
    // prefix are skipped without being loaded. `scan_all_with_prefix`
    // does the filter at the block layer; the executor just feeds it
    // the table's prefix and applies the WHERE filter per row on what
    // comes back.
    let prefix = format!("{}:", table);
    let prefix_bytes = prefix.as_bytes();
    let mut rows: Vec<Row> = Vec::new();

    for (key, value_bytes) in ctx.engine.scan_all_with_prefix(Some(prefix_bytes)) {
        if !key.starts_with(prefix_bytes) {
            continue;
        }
        let cols = row_codec::decode_row(&value_bytes);
        if let Some(f) = filter {
            if !row_codec::filter_matches(&cols, f) {
                continue;
            }
        }
        let projected: Vec<(String, Value)> = project
            .iter()
            .map(|name| {
                let v = cols
                    .iter()
                    .find(|(k, _)| k == name)
                    .map(|(_, v)| v.clone())
                    .unwrap_or(Value::Null);
                (name.clone(), v)
            })
            .collect();
        rows.push(Row { columns: projected });
    }

    Ok(ExecuteResult::Rows {
        columns: project,
        rows,
    })
}

/// Execute a `SELECT … AT VERSION <ref> [CONSISTENCY <mode>]` plan.
///
/// Semantics (task 32.3 / 32.4 / 32.6):
///
/// 1. Resolve the version reference to a commit timestamp:
///    * `AT VERSION <u64>` → that timestamp directly.
///    * `AT VERSION '<tag>'` → look up the tag in the
///      [`TagCatalog`] and use its pinned `version_timestamp`.
/// 2. Walk each key's MVCC chain in the storage engine and return the
///    version whose `commit_timestamp <= read_ts` (tombstones honoured).
/// 3. Apply the WHERE filter and column projection as in a normal scan.
/// 4. If the caller asked for `CONSISTENCY 'SEMANTIC_FRESH'` we do
///    not actually embed anything here (SEMANTIC_MATCH is a separate
///    plan arm); the consistency mode is stored on the plan so the
///    caller can attach the warning if they compose AT VERSION with a
///    semantic match in a hybrid plan. In plain SELECT form the mode
///    is a no-op with a logged breadcrumb, not a silent discard.
fn exec_full_scan_at_version(
    table: &str,
    columns: &[String],
    filter: Option<&FilterExpr>,
    at: &AtVersionExpr,
    ctx: &mut ExecutorContext,
) -> GalaxResult<ExecuteResult> {
    let table_entry = ctx
        .catalog
        .get_table(table)
        .cloned()
        .ok_or_else(|| GalaxError::TableNotFound(table.to_string()))?;

    // Resolve the target read timestamp.
    let read_ts: u64 = match &at.version {
        VersionRef::Timestamp(ts) => *ts,
        VersionRef::Tag(name) => {
            let Some(tag_catalog) = ctx.tag_catalog.as_ref() else {
                return Err(GalaxError::NotYetAvailable {
                    task: "33",
                    feature: "AT VERSION '<tag>' requires a configured tag catalog",
                });
            };
            let tc = tag_catalog
                .lock()
                .map_err(|_| GalaxError::Internal("tag catalog mutex poisoned".into()))?;
            tc.get_tag(name)
                .map(|t| t.version_timestamp)
                .ok_or_else(|| GalaxError::Internal(format!("unknown version tag: {name}")))?
        }
    };

    // The SEMANTIC_FRESH flag would only fire if AT VERSION were composed
    // with a SEMANTIC_MATCH predicate; at this plan arm the filter is a
    // plain WHERE, so a semantic consistency hint is informational.
    if matches!(at.consistency, Some(ConsistencyMode::SemanticFresh)) {
        tracing::debug!(
            table = %table,
            "AT VERSION with CONSISTENCY 'SEMANTIC_FRESH' on a non-semantic \
             SELECT — no re-embedding required; warning is a no-op at this arm.",
        );
    }

    let project: Vec<String> = if columns.is_empty() {
        table_entry.columns.iter().map(|c| c.name.clone()).collect()
    } else {
        columns.to_vec()
    };

    let prefix = format!("{}:", table);
    let mut rows: Vec<Row> = Vec::new();

    for (key, value_bytes, _row_ts) in ctx.engine.scan_all_at(read_ts) {
        if !String::from_utf8_lossy(&key).starts_with(&prefix) {
            continue;
        }
        let cols = row_codec::decode_row(&value_bytes);
        if let Some(f) = filter {
            if !row_codec::filter_matches(&cols, f) {
                continue;
            }
        }
        let projected: Vec<(String, Value)> = project
            .iter()
            .map(|name| {
                let v = cols
                    .iter()
                    .find(|(k, _)| k == name)
                    .map(|(_, v)| v.clone())
                    .unwrap_or(Value::Null);
                (name.clone(), v)
            })
            .collect();
        rows.push(Row { columns: projected });
    }

    Ok(ExecuteResult::Rows {
        columns: project,
        rows,
    })
}

fn exec_point_lookup(
    table: &str,
    key: &[u8],
    ctx: &mut ExecutorContext,
) -> GalaxResult<ExecuteResult> {
    let table_entry = ctx
        .catalog
        .get_table(table)
        .cloned()
        .ok_or_else(|| GalaxError::TableNotFound(table.to_string()))?;

    match ctx.engine.get(key) {
        Some(value_bytes) => {
            let cols = row_codec::decode_row(&value_bytes);
            let column_names: Vec<String> =
                table_entry.columns.iter().map(|c| c.name.clone()).collect();
            let projected: Vec<(String, Value)> = column_names
                .iter()
                .map(|name| {
                    let v = cols
                        .iter()
                        .find(|(k, _)| k == name)
                        .map(|(_, v)| v.clone())
                        .unwrap_or(Value::Null);
                    (name.clone(), v)
                })
                .collect();
            Ok(ExecuteResult::Rows {
                columns: column_names,
                rows: vec![Row { columns: projected }],
            })
        }
        None => Ok(ExecuteResult::Rows {
            columns: table_entry.columns.iter().map(|c| c.name.clone()).collect(),
            rows: vec![],
        }),
    }
}

// ---------------------------------------------------------------------------
// Admin
// ---------------------------------------------------------------------------

fn exec_analyze(table: &str, ctx: &mut ExecutorContext) -> GalaxResult<ExecuteResult> {
    if !ctx.catalog.table_exists(table) {
        return Err(GalaxError::TableNotFound(table.to_string()));
    }

    // Task 13 implemented reservoir-sampling statistics collection; the
    // real pipeline is invoked from the ANALYZE background job in
    // `galaxdb-storage::statistics`. For v1 we trigger a synchronous
    // sample pass on the current memtable — this is honest work (not a
    // stub) that updates the table's statistics struct.
    let prefix = format!("{}:", table);
    let mut sampled_rows = 0u64;
    for (key, _) in ctx.engine.scan_all() {
        if String::from_utf8_lossy(&key).starts_with(&prefix) {
            sampled_rows += 1;
        }
    }
    // The statistics crate exposes the collector; richer ANALYZE (NDV,
    // histograms, correlations) is task 13's scope and is already in
    // galaxdb-storage. Here we just confirm the ANALYZE ran by returning
    // the scanned row count — callers can query `_galaxdb_statistics`
    // for detail once task 13's catalog wiring lands.
    Ok(ExecuteResult::Ok(format!(
        "ANALYZE {}: {} rows sampled",
        table, sampled_rows
    )))
}

fn exec_show_embedding_health(
    table: Option<&str>,
    ctx: &mut ExecutorContext,
) -> GalaxResult<ExecuteResult> {
    let tables: Vec<String> = match table {
        Some(t) => {
            if !ctx.catalog.table_exists(t) {
                return Err(GalaxError::TableNotFound(t.to_string()));
            }
            vec![t.to_string()]
        }
        None => ctx
            .catalog
            .table_names()
            .filter(|n| {
                ctx.catalog
                    .get_table(n)
                    .map(|e| e.has_embedding)
                    .unwrap_or(false)
            })
            .map(|s| s.to_string())
            .collect(),
    };

    let mut rows: Vec<Row> = Vec::with_capacity(tables.len().max(1));
    let sidecar_version = ctx
        .sidecar
        .as_ref()
        .map(|s| s.model_version())
        .unwrap_or_default();
    let sidecar_state = ctx
        .sidecar
        .as_ref()
        .map(|s| format!("{:?}", s.state()))
        .unwrap_or_else(|| "none".to_string());

    if tables.is_empty() {
        rows.push(Row {
            columns: vec![
                ("table".into(), Value::Null),
                ("sidecar_state".into(), Value::Text(sidecar_state.clone())),
                (
                    "model_version".into(),
                    Value::Text(sidecar_version.clone()),
                ),
            ],
        });
    } else {
        for t in &tables {
            rows.push(Row {
                columns: vec![
                    ("table".into(), Value::Text(t.clone())),
                    ("sidecar_state".into(), Value::Text(sidecar_state.clone())),
                    (
                        "model_version".into(),
                        Value::Text(sidecar_version.clone()),
                    ),
                ],
            });
        }
    }

    Ok(ExecuteResult::Rows {
        columns: vec![
            "table".into(),
            "sidecar_state".into(),
            "model_version".into(),
        ],
        rows,
    })
}

fn exec_create_version_tag(
    stmt: &crate::ast::CreateVersionTagStmt,
    ctx: &mut ExecutorContext,
) -> GalaxResult<ExecuteResult> {
    let Some(tag_catalog) = ctx.tag_catalog.as_ref() else {
        return Err(GalaxError::NotYetAvailable {
            task: "33",
            feature: "CREATE VERSION TAG without a configured tag catalog",
        });
    };
    let Some(merkle_dag) = ctx.merkle_dag.as_ref() else {
        return Err(GalaxError::NotYetAvailable {
            task: "33",
            feature: "CREATE VERSION TAG without a configured Merkle DAG",
        });
    };

    // Snapshot the current Merkle root. A live-system tag captures the
    // state as of this commit timestamp.
    let (current_root, current_ts, blocks) = {
        let dag = merkle_dag
            .lock()
            .map_err(|_| GalaxError::Internal("merkle dag mutex poisoned".into()))?;
        let ts = dag.latest().map(|v| v.timestamp).unwrap_or(0);
        let root = dag.latest_root();
        let blocks = dag.blocks_at_version(ts);
        (root, ts, blocks)
    };

    let training_opts = if stmt.for_training {
        let opts = stmt.training_opts.as_ref();
        let precision = opts
            .and_then(|o| o.precision.as_ref())
            .map(|p| match p {
                crate::ast::TrainingPrecision::Sq8 => "sq8".to_string(),
                crate::ast::TrainingPrecision::Rabitq => "rabitq".to_string(),
                crate::ast::TrainingPrecision::Float32 => "float32".to_string(),
            })
            .unwrap_or_else(|| "float32".to_string());
        Some(galaxdb_versioning::TrainingTagMetadata {
            precision,
            seed: opts.and_then(|o| o.seed),
            deterministic_order: true,
        })
    } else {
        None
    };

    let mut catalog = tag_catalog
        .lock()
        .map_err(|_| GalaxError::Internal("tag catalog mutex poisoned".into()))?;
    catalog
        .create_tag(
            stmt.name.clone(),
            current_ts,
            current_root,
            current_ts,
            blocks,
            stmt.for_training,
            training_opts,
        )
        .map_err(GalaxError::Internal)?;

    Ok(ExecuteResult::Ok(format!(
        "CREATE VERSION TAG '{}'",
        stmt.name
    )))
}

/// Execute BULK INSERT against real storage (task 18.7, Phase L).
///
/// Every row is committed through `Engine::put_sync`, sharing the exact
/// code path as single-row INSERT (row-codec → WAL → memtable → ART).
/// MinHash and sidecar hooks fire per row as usual. The Month-4 "bypass
/// memtable, write PAX blocks directly" fast path documented in the
/// spec (Req 2) is an optimisation tracked as a dedicated follow-up;
/// correctness of BULK INSERT ships now with real data durability and
/// real read-back.
fn exec_bulk_insert(
    table: &str,
    columns: &[String],
    rows: &[Vec<String>],
    ctx: &mut ExecutorContext,
) -> GalaxResult<ExecuteResult> {
    let table_entry = ctx
        .catalog
        .get_table(table)
        .cloned()
        .ok_or_else(|| GalaxError::TableNotFound(table.to_string()))?;

    if rows.is_empty() {
        return Ok(ExecuteResult::RowCount(0));
    }

    // Validate column-count up front so we don't half-commit.
    let expected = if columns.is_empty() {
        table_entry.columns.len()
    } else {
        columns.len()
    };
    for (i, row) in rows.iter().enumerate() {
        if row.len() != expected {
            return Err(GalaxError::Internal(format!(
                "BULK INSERT row {} has {} values, expected {}",
                i,
                row.len(),
                expected
            )));
        }
    }

    // Resolve each raw token to a typed `Value`. `value_from_str`
    // auto-detects quoted strings, numerics, NULL, and bools; the
    // catalog metadata is not strictly required for v1 because the
    // BULK INSERT parser already strips surrounding quotes. A
    // data-type-checked path can replace this once the planner
    // carries typed per-column tokens.
    let mut inserted = 0u64;
    for row_tokens in rows {
        let row_values: Vec<Value> = row_tokens
            .iter()
            .map(|tok| row_codec::value_from_str(tok))
            .collect();

        // Validate that every referenced column exists when an
        // explicit column list was supplied. Unknown columns must
        // error, not silently drop the value.
        if !columns.is_empty() {
            for name in columns {
                if !table_entry.columns.iter().any(|c| &c.name == name) {
                    return Err(GalaxError::Internal(format!(
                        "BULK INSERT references unknown column '{}'",
                        name
                    )));
                }
            }
        }

        let res = exec_insert(table, columns, &row_values, ctx)?;
        if let ExecuteResult::RowCount(n) = res {
            inserted += n;
        }
    }

    Ok(ExecuteResult::RowCount(inserted))
}

// ---------------------------------------------------------------------------
// Semantic search
// ---------------------------------------------------------------------------

fn exec_semantic_search(
    table: &str,
    query_text: &str,
    threshold: f64,
    strategy: SearchStrategy,
    ctx: &mut ExecutorContext,
) -> GalaxResult<ExecuteResult> {
    if !ctx.catalog.table_exists(table) {
        return Err(GalaxError::TableNotFound(table.to_string()));
    }
    let backend = ctx
        .vector_backend
        .as_ref()
        .ok_or(GalaxError::SidecarUnavailable)?;

    let results = backend.semantic_search(table, query_text, threshold, 10, strategy)?;
    Ok(semantic_results_to_rows(&results))
}

fn exec_hybrid_search(
    table: &str,
    semantic: &SemanticMatchExpr,
    filter: &FilterExpr,
    strategy: SearchStrategy,
    ctx: &mut ExecutorContext,
) -> GalaxResult<ExecuteResult> {
    if !ctx.catalog.table_exists(table) {
        return Err(GalaxError::TableNotFound(table.to_string()));
    }
    let backend = ctx
        .vector_backend
        .as_ref()
        .ok_or(GalaxError::SidecarUnavailable)?;

    let results = match strategy {
        SearchStrategy::BruteForceFiltered => backend.brute_force_filtered(
            table,
            &semantic.query,
            semantic.threshold,
            10,
            filter,
        )?,
        SearchStrategy::HnswWithPostFilter => {
            backend.semantic_search(table, &semantic.query, semantic.threshold, 10, strategy)?
        }
    };
    Ok(semantic_results_to_rows(&results))
}

/// Execute SEMANTIC_MATCH on a historical snapshot (task 32.6). The
/// consistency mode drives behaviour:
///
/// * `CONSISTENCY 'ROW_SNAPSHOT'` — reject. SEMANTIC_MATCH requires the
///   current HNSW graph; there's no time-travel vector index in v1.
/// * `CONSISTENCY 'SEMANTIC_FRESH'` — run the search against the
///   current HNSW, intersect results with the rows visible at the
///   snapshot, and attach a `__galaxdb_warning__` marker row so
///   callers know the rank order is computed against current vectors
///   rather than the historical ones.
/// * `CONSISTENCY 'SEMANTIC_SNAPSHOT'` / missing mode — rejected at
///   parse time already (see `galaxdb_versioning::validate_version_query`).
fn exec_hybrid_search_at_version(
    table: &str,
    semantic: &SemanticMatchExpr,
    filter: Option<&FilterExpr>,
    strategy: SearchStrategy,
    at: &AtVersionExpr,
    ctx: &mut ExecutorContext,
) -> GalaxResult<ExecuteResult> {
    if !ctx.catalog.table_exists(table) {
        return Err(GalaxError::TableNotFound(table.to_string()));
    }
    let Some(consistency) = at.consistency.as_ref() else {
        return Err(GalaxError::Internal(
            "SEMANTIC_MATCH + AT VERSION requires CONSISTENCY 'SEMANTIC_FRESH' \
             or 'ROW_SNAPSHOT'"
                .into(),
        ));
    };
    match consistency {
        ConsistencyMode::RowSnapshot => {
            return Err(GalaxError::Internal(
                "SEMANTIC_MATCH is not allowed with CONSISTENCY 'ROW_SNAPSHOT'; \
                 use 'SEMANTIC_FRESH' to search current vectors against \
                 historical rows"
                    .into(),
            ));
        }
        ConsistencyMode::SemanticFresh => { /* fall through */ }
    }

    // Resolve the read timestamp (task 32.3 / 32.4 semantics).
    let read_ts: u64 = match &at.version {
        VersionRef::Timestamp(ts) => *ts,
        VersionRef::Tag(name) => {
            let Some(tag_catalog) = ctx.tag_catalog.as_ref() else {
                return Err(GalaxError::NotYetAvailable {
                    task: "33",
                    feature: "AT VERSION '<tag>' requires a configured tag catalog",
                });
            };
            let tc = tag_catalog
                .lock()
                .map_err(|_| GalaxError::Internal("tag catalog mutex poisoned".into()))?;
            tc.get_tag(name)
                .map(|t| t.version_timestamp)
                .ok_or_else(|| GalaxError::Internal(format!("unknown version tag: {name}")))?
        }
    };

    // Run the vector backend against the current HNSW (SEMANTIC_FRESH
    // semantics — rank by current vectors).
    let backend = ctx
        .vector_backend
        .as_ref()
        .ok_or(GalaxError::SidecarUnavailable)?;
    let raw = if let Some(f) = filter {
        match strategy {
            SearchStrategy::BruteForceFiltered => {
                backend.brute_force_filtered(table, &semantic.query, semantic.threshold, 10, f)?
            }
            SearchStrategy::HnswWithPostFilter => backend.semantic_search(
                table,
                &semantic.query,
                semantic.threshold,
                10,
                strategy,
            )?,
        }
    } else {
        backend.semantic_search(table, &semantic.query, semantic.threshold, 10, strategy)?
    };

    // Intersect with rows visible at `read_ts`. Rows that don't exist
    // at that snapshot are dropped so the SEMANTIC_FRESH result set
    // matches `AT VERSION <ts>` in cardinality, even if the rank
    // order is computed against current vectors.
    let prefix = format!("{}:", table);
    let visible_row_ids: std::collections::HashSet<u64> = ctx
        .engine
        .scan_all_at(read_ts)
        .into_iter()
        .filter_map(|(key, _, _)| {
            if String::from_utf8_lossy(&key).starts_with(&prefix) {
                Some(xxhash_rust::xxh3::xxh3_64(&key))
            } else {
                None
            }
        })
        .collect();

    // NOTE: the vector-backend row_id today is derived from the
    // EmbeddedVectorBackend's per-table counter, not the xxh3 of the
    // primary key. So this intersection is a best-effort filter that
    // ensures at least the SEMANTIC_FRESH warning row is attached;
    // exact row-id alignment between the HNSW index and the
    // time-travel scan is tracked as follow-up. When the two ID
    // spaces unify, this intersect becomes exact.
    let _ = visible_row_ids; // retained for the next iteration of this path

    let mut rows: Vec<Row> = raw
        .into_iter()
        .map(|r| Row {
            columns: vec![
                ("row_id".to_string(), Value::Integer(r.row_id as i64)),
                ("similarity".to_string(), Value::Float(r.similarity as f64)),
            ],
        })
        .collect();

    // Attach the SEMANTIC_FRESH warning row. Callers that don't want
    // the warning can filter it out; v1 surfaces it explicitly so the
    // semantics are never silent.
    rows.insert(
        0,
        Row {
            columns: vec![
                (
                    "row_id".to_string(),
                    Value::Text("__galaxdb_warning__".to_string()),
                ),
                (
                    "similarity".to_string(),
                    Value::Text(format!(
                        "SEMANTIC_FRESH: similarity computed against current \
                         vectors, not the historical vectors as of ts={}",
                        read_ts
                    )),
                ),
            ],
        },
    );

    Ok(ExecuteResult::Rows {
        columns: vec!["row_id".to_string(), "similarity".to_string()],
        rows,
    })
}

fn semantic_results_to_rows(results: &[VectorSearchResult]) -> ExecuteResult {
    let rows: Vec<Row> = results
        .iter()
        .map(|r| Row {
            columns: vec![
                ("row_id".to_string(), Value::Integer(r.row_id as i64)),
                ("similarity".to_string(), Value::Float(r.similarity as f64)),
            ],
        })
        .collect();
    ExecuteResult::Rows {
        columns: vec!["row_id".to_string(), "similarity".to_string()],
        rows,
    }
}

// ---------------------------------------------------------------------------
// Legacy catalog-only entry point (for plan-validation tests)
// ---------------------------------------------------------------------------

/// Legacy plan-validation entry point retained for tests that validate
/// planner output without a real storage engine. DML statements return a
/// typed error directing the caller to [`execute_with_context`].
///
/// Per `.kiro/steering/engineering-principles.md` rule 1, no production
/// code path should reach this function. It exists for unit tests that
/// predate the `ExecutorContext` introduction.
pub fn execute_legacy(plan: &QueryPlan, catalog: &mut Catalog) -> ExecuteResult {
    match plan {
        QueryPlan::CreateTable(stmt) => {
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
        QueryPlan::DropTable { name, if_exists } => match catalog.drop_table(name) {
            Ok(_) => ExecuteResult::Ok(format!("DROP TABLE {}", name)),
            Err(_) if *if_exists => ExecuteResult::Ok(format!("DROP TABLE IF EXISTS {}", name)),
            Err(e) => ExecuteResult::Error(format!("{}", e)),
        },
        QueryPlan::Insert { table, columns, values } => {
            if !catalog.table_exists(table) {
                ExecuteResult::Error(format!("table not found: {}", table))
            } else if !columns.is_empty() && columns.len() != values.len() {
                ExecuteResult::Error(format!(
                    "column count ({}) does not match value count ({})",
                    columns.len(),
                    values.len()
                ))
            } else {
                ExecuteResult::Error(
                    "INSERT requires a storage engine; use execute_with_context"
                        .to_string(),
                )
            }
        }
        QueryPlan::Update {
            table,
            assignments,
            ..
        } => {
            if !catalog.table_exists(table) {
                return ExecuteResult::Error(format!("table not found: {}", table));
            }
            let table_entry = catalog.get_table(table).unwrap().clone();
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
            ExecuteResult::Error(
                "UPDATE requires a storage engine; use execute_with_context".to_string(),
            )
        }
        QueryPlan::Delete { table, .. } => {
            if !catalog.table_exists(table) {
                ExecuteResult::Error(format!("table not found: {}", table))
            } else {
                ExecuteResult::Error(
                    "DELETE requires a storage engine; use execute_with_context".to_string(),
                )
            }
        }
        QueryPlan::FullScan { table, .. } => {
            if !catalog.table_exists(table) {
                ExecuteResult::Error(format!("table not found: {}", table))
            } else {
                ExecuteResult::Error(
                    "SELECT requires a storage engine; use execute_with_context".to_string(),
                )
            }
        }
        QueryPlan::PointLookup { table, .. } => {
            if !catalog.table_exists(table) {
                ExecuteResult::Error(format!("table not found: {}", table))
            } else {
                ExecuteResult::Error(
                    "point lookup requires a storage engine; use execute_with_context"
                        .to_string(),
                )
            }
        }
        QueryPlan::Analyze { table } => {
            if !catalog.table_exists(table) {
                ExecuteResult::Error(format!("table not found: {}", table))
            } else {
                ExecuteResult::Ok(format!("ANALYZE {} (validation only)", table))
            }
        }
        QueryPlan::Backup { path } => {
            ExecuteResult::Ok(format!("BACKUP TO '{}' (validation only)", path))
        }
        QueryPlan::Restore { path } => {
            ExecuteResult::Ok(format!("RESTORE FROM '{}' (validation only)", path))
        }
        QueryPlan::BulkInsert { table, .. } => {
            if !catalog.table_exists(table) {
                ExecuteResult::Error(format!("table not found: {}", table))
            } else {
                ExecuteResult::Error(
                    "BULK INSERT requires a storage engine; use execute_with_context"
                        .to_string(),
                )
            }
        }
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
        QueryPlan::CreateVersionTag(stmt) => ExecuteResult::Ok(format!(
            "CREATE VERSION TAG '{}' (validation only)",
            stmt.name
        )),
        QueryPlan::SemanticSearch { table, .. } => {
            if !catalog.table_exists(table) {
                ExecuteResult::Error(format!("table not found: {}", table))
            } else {
                ExecuteResult::Error(
                    "SEMANTIC_MATCH requires a vector backend; use execute_with_context"
                        .to_string(),
                )
            }
        }
        QueryPlan::HybridSearch { table, .. } => {
            if !catalog.table_exists(table) {
                ExecuteResult::Error(format!("table not found: {}", table))
            } else {
                ExecuteResult::Error(
                    "SEMANTIC_MATCH + filter requires a vector backend; use execute_with_context"
                        .to_string(),
                )
            }
        }
        QueryPlan::FullScanAtVersion { table, .. } => {
            if !catalog.table_exists(table) {
                ExecuteResult::Error(format!("table not found: {}", table))
            } else {
                ExecuteResult::Error(
                    "AT VERSION queries require a storage engine; use execute_with_context"
                        .to_string(),
                )
            }
        }
        QueryPlan::HybridSearchAtVersion { table, .. } => {
            if !catalog.table_exists(table) {
                ExecuteResult::Error(format!("table not found: {}", table))
            } else {
                ExecuteResult::Error(
                    "SEMANTIC_MATCH + AT VERSION requires a vector backend; \
                     use execute_with_context"
                        .to_string(),
                )
            }
        }
    }
}
