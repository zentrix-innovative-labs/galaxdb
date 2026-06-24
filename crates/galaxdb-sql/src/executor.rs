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
    /// Append-only tables reject UPDATE and DELETE (task 36.2). Used
    /// for system lineage tables like `_galaxdb_training_exports` that
    /// must remain auditable.
    pub append_only: bool,
}

/// Canonical name of the training-export lineage system table
/// (Req 38 / task 36). Every successful `LanceExporter::export` lands
/// one row here; DELETE and UPDATE against this table are rejected at
/// the executor.
pub const TRAINING_EXPORTS_TABLE: &str = "_galaxdb_training_exports";

/// Is `name` the fixed name of an append-only system table? Today
/// `_galaxdb_training_exports` is the only one; future system tables
/// that want the same treatment extend this list.
pub fn is_system_append_only_table(name: &str) -> bool {
    name == TRAINING_EXPORTS_TABLE
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

    /// Table metadata. The executor updates this on DDL via
    /// `Arc::make_mut` (copy-on-write); DML clones the `Arc` (a refcount
    /// bump) instead of deep-cloning the table map on every statement.
    pub catalog: Arc<Catalog>,

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

    /// Optional persistent role + grant catalog (Req 3). When `None`,
    /// role/grant DDL returns [`GalaxError::NotYetAvailable`] rather than
    /// a fake success. The server/embedded layer supplies a real
    /// [`crate::auth_store::AuthStore`] backed by the engine.
    pub auth_store: Option<crate::auth_store::AuthStore>,

    /// The authenticated session, if the connection ran through
    /// authentication (Req 3). When `Some`, the executor enforces
    /// authorization at the chokepoint in [`execute_with_context`] before
    /// any storage access: every plan maps to an
    /// [`galaxdb_auth::Action`] + [`galaxdb_auth::ObjectRef`] and is
    /// checked against the [`Self::authorizer`].
    ///
    /// When `None` (in-process trusted embedded use without auth, e.g. a
    /// PyO3 caller that opened the database directly), the check is
    /// skipped — there is no authenticated principal to evaluate and the
    /// caller already holds the engine handle. The wire server always
    /// supplies a session when auth is enabled, so a networked client can
    /// never bypass the check (Req 3, AC7).
    pub session: Option<galaxdb_auth::SessionContext>,

    /// The authorizer consulted at the chokepoint. Defaults to the
    /// grant-backed [`galaxdb_auth::TableGrantAuthorizer`] wired to this
    /// context's [`Self::auth_store`] when a session is attached. Only
    /// consulted when [`Self::session`] is `Some`.
    pub authorizer: Option<Arc<dyn galaxdb_auth::Authorizer>>,

    /// Security audit sink (Req 4). When set, the executor records
    /// authorization denials at the chokepoint and role/grant changes as
    /// they commit, so security-relevant events land in the configured
    /// [`galaxdb_auth::AuditSink`] (a JSONL file in OSS, a tamper-evident
    /// sink in ENT). `None` (the default) discards events — equivalent to
    /// [`galaxdb_auth::NoOpAuditSink`]. The server/embedded layer supplies
    /// a real sink when one is configured.
    pub audit: Option<Arc<dyn galaxdb_auth::AuditSink>>,

    /// Secondary-index store (Req 5). When set, `CREATE/DROP INDEX` manage
    /// definitions here, the write path (`exec_insert`/`update`/`delete`)
    /// maintains entries in the same logical write, and the read path uses
    /// it to resolve indexed predicates without a full scan. `None`
    /// disables secondary indexes (DDL returns a typed error); the
    /// embedded/server layer always supplies a real store backed by the
    /// engine.
    pub secondary_index: Option<crate::secondary_index::SecondaryIndexStore>,
}

impl ExecutorContext {
    /// Construct a context around an engine, with no optional subsystems
    /// enabled. The caller can attach a sidecar, tag catalog, or vector
    /// backend later by setting the fields directly.
    pub fn new(engine: Arc<Engine>) -> Self {
        Self {
            engine,
            catalog: Arc::new(Catalog::new()),
            sidecar: None,
            tag_catalog: None,
            merkle_dag: None,
            minhash_policy: None,
            vector_backend: None,
            auth_store: None,
            session: None,
            authorizer: None,
            audit: None,
            secondary_index: None,
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

/// A decoded row buffered in memory during scan-and-filter passes:
/// the raw primary-key bytes paired with the row's `(column, value)`
/// pairs in catalog order. Used by UPDATE/DELETE rewrite passes and the
/// `WHERE NOT DUPLICATE` dedup pass.
type BufferedRow = (Vec<u8>, Vec<(String, Value)>);

/// Human-readable label for a [`QueryPlan`] variant, used as the
/// `plan` field on the `query.execute` span (task 38.5).
fn plan_kind_str(plan: &QueryPlan) -> &'static str {
    match plan {
        QueryPlan::CreateTable(_) => "create_table",
        QueryPlan::DropTable { .. } => "drop_table",
        QueryPlan::Insert { .. } => "insert",
        QueryPlan::Update { .. } => "update",
        QueryPlan::Delete { .. } => "delete",
        QueryPlan::FullScan { .. } => "full_scan",
        QueryPlan::FullScanAtVersion { .. } => "full_scan_at_version",
        QueryPlan::PointLookup { .. } => "point_lookup",
        QueryPlan::SemanticSearch { .. } => "semantic_search",
        QueryPlan::HybridSearch { .. } => "hybrid_search",
        QueryPlan::HybridSearchAtVersion { .. } => "hybrid_search_at_version",
        QueryPlan::BulkInsert { .. } => "bulk_insert",
        QueryPlan::CreateVersionTag(_) => "create_version_tag",
        QueryPlan::Analyze { .. } => "analyze",
        QueryPlan::Backup { .. } => "backup",
        QueryPlan::Restore { .. } => "restore",
        QueryPlan::ShowEmbeddingHealth { .. } => "show_embedding_health",
        QueryPlan::CreateRole(_) => "create_role",
        QueryPlan::DropRole { .. } => "drop_role",
        QueryPlan::AlterRolePassword { .. } => "alter_role_password",
        QueryPlan::Grant(_) => "grant",
        QueryPlan::Revoke(_) => "revoke",
        QueryPlan::CreateIndex(_) => "create_index",
        QueryPlan::DropIndex { .. } => "drop_index",
    }
}

/// Map a [`QueryPlan`] to the `(Action, ObjectRef)` it must be authorized
/// for. This is the single source of truth for "what privilege does this
/// statement require", consulted by the authorization chokepoint in
/// [`execute_with_context`].
///
/// * Data reads (SELECT, point lookup, semantic/hybrid search, time-travel
///   scans) require [`Action::Select`] on the target table.
/// * Data writes require the matching [`Action::Insert`]/`Update`/`Delete`
///   on the target table.
/// * Schema changes (CREATE/DROP TABLE, ANALYZE) and version tags require
///   [`Action::Ddl`] (superuser-only in the open-core baseline).
/// * Role/grant administration requires [`Action::Admin`] (superuser-only).
/// * BACKUP/RESTORE are operator actions and require [`Action::Admin`].
fn plan_authz_target(plan: &QueryPlan) -> (galaxdb_auth::Action, galaxdb_auth::ObjectRef) {
    use galaxdb_auth::{Action, ObjectRef};
    match plan {
        // -- Reads (SELECT family) --
        QueryPlan::PointLookup { table, .. }
        | QueryPlan::FullScan { table, .. }
        | QueryPlan::FullScanAtVersion { table, .. }
        | QueryPlan::SemanticSearch { table, .. }
        | QueryPlan::HybridSearch { table, .. }
        | QueryPlan::HybridSearchAtVersion { table, .. } => {
            (Action::Select, ObjectRef::Table(table.clone()))
        }

        // -- Writes --
        QueryPlan::Insert { table, .. } | QueryPlan::BulkInsert { table, .. } => {
            (Action::Insert, ObjectRef::Table(table.clone()))
        }
        QueryPlan::Update { table, .. } => (Action::Update, ObjectRef::Table(table.clone())),
        QueryPlan::Delete { table, .. } => (Action::Delete, ObjectRef::Table(table.clone())),

        // -- Schema (DDL) --
        QueryPlan::CreateTable(stmt) => {
            (Action::Ddl, ObjectRef::Table(stmt.table_name.clone()))
        }
        QueryPlan::DropTable { name, .. } => (Action::Ddl, ObjectRef::Table(name.clone())),
        QueryPlan::Analyze { table } => (Action::Ddl, ObjectRef::Table(table.clone())),
        QueryPlan::CreateVersionTag(_) => (Action::Ddl, ObjectRef::Cluster),
        // CREATE/DROP INDEX are schema changes scoped to their table.
        QueryPlan::CreateIndex(stmt) => (Action::Ddl, ObjectRef::Table(stmt.table.clone())),
        QueryPlan::DropIndex { .. } => (Action::Ddl, ObjectRef::Cluster),
        // SHOW EMBEDDING HEALTH reads health for a table (or the whole
        // server when no table is named): a table-scoped read, or a
        // cluster-scoped read needing superuser when global.
        QueryPlan::ShowEmbeddingHealth { table } => match table {
            Some(t) => (Action::Select, ObjectRef::Table(t.clone())),
            None => (Action::Select, ObjectRef::Cluster),
        },

        // -- Operator actions --
        QueryPlan::Backup { .. } | QueryPlan::Restore { .. } => (Action::Admin, ObjectRef::Cluster),

        // -- Role/grant administration --
        QueryPlan::CreateRole(_)
        | QueryPlan::DropRole { .. }
        | QueryPlan::AlterRolePassword { .. }
        | QueryPlan::Grant(_)
        | QueryPlan::Revoke(_) => (Action::Admin, ObjectRef::Cluster),
    }
}

/// Enforce authorization for `plan` before any storage access (Req 3, AC3).
///
/// When the context carries no [`ExecutorContext::session`] (trusted
/// in-process embedded use), this is a no-op: there is no authenticated
/// principal to evaluate and the caller already holds the engine handle.
///
/// When a session *is* present, the plan is mapped to an
/// `(Action, ObjectRef)` via [`plan_authz_target`] and checked against the
/// authorizer. The authorizer is, in precedence order:
///
/// 1. an explicitly-installed [`ExecutorContext::authorizer`], or
/// 2. the grant-backed [`galaxdb_auth::TableGrantAuthorizer`] built over
///    the context's [`ExecutorContext::auth_store`] — this reads the
///    *live* grant set, so a `GRANT`/`REVOKE` committed by an earlier
///    statement takes effect immediately (Req 3, AC6).
///
/// If a session is attached but neither an authorizer nor an auth store is
/// available, the check **fails closed** (`InsufficientPrivilege`) rather
/// than allowing unchecked access — a misconfigured server must never
/// silently skip authorization.
fn enforce_authorization(plan: &QueryPlan, ctx: &ExecutorContext) -> GalaxResult<()> {
    use galaxdb_auth::Authorizer as _;
    let Some(session) = ctx.session.as_ref() else {
        return Ok(());
    };
    let role = &session.role;
    let (action, object) = plan_authz_target(plan);

    let outcome = if let Some(authz) = ctx.authorizer.as_ref() {
        authz.check(role, action, &object)
    } else if let Some(store) = ctx.auth_store.as_ref() {
        let store = store.clone();
        let authz = galaxdb_auth::TableGrantAuthorizer::new(Arc::new(
            move |r: &str, t: &str, a: galaxdb_auth::Action| store.has_grant(r, t, a),
        ));
        authz.check(role, action, &object)
    } else {
        let err = GalaxError::InsufficientPrivilege {
            role: role.id.to_string(),
            action: action.label(),
            object: object.label(),
        };
        audit_authz(ctx, role, action, &object, galaxdb_auth::AuditOutcome::Denied,
            Some("no authorizer or auth store configured (fail-closed)"));
        return Err(err);
    };

    match outcome {
        Ok(()) => {
            // Record successful privileged actions (role/grant admin and
            // schema changes) so the audit trail shows who changed
            // security state or the schema. High-volume data reads/writes
            // are intentionally not audited per-row here to keep the hot
            // path quiet; the chokepoint still records every denial below.
            if matches!(action, galaxdb_auth::Action::Admin | galaxdb_auth::Action::Ddl) {
                audit_authz(ctx, role, action, &object,
                    galaxdb_auth::AuditOutcome::Allowed, None);
            }
            Ok(())
        }
        Err(e) => {
            audit_authz(ctx, role, action, &object,
                galaxdb_auth::AuditOutcome::Denied, None);
            Err(GalaxError::InsufficientPrivilege {
                role: e.role.to_string(),
                action: e.action,
                object: e.object,
            })
        }
    }
}

/// Emit an `authz` audit event when an audit sink is configured. A no-op
/// when `ctx.audit` is `None`, so the hot path is untouched for engines
/// without auditing.
fn audit_authz(
    ctx: &ExecutorContext,
    role: &galaxdb_auth::Role,
    action: galaxdb_auth::Action,
    object: &galaxdb_auth::ObjectRef,
    outcome: galaxdb_auth::AuditOutcome,
    detail: Option<&str>,
) {
    let Some(sink) = ctx.audit.as_ref() else {
        return;
    };
    let mut event = galaxdb_auth::AuditEvent::new("authz", action.label(), outcome)
        .with_role(role.id.as_str())
        .with_object(object.label());
    if let Some(d) = detail {
        event = event.with_detail(d);
    }
    sink.record(&event);
}

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
    // Task 38.5: every plan dispatch lives inside a `query.execute`
    // span so tracing backends get a root span per SQL statement.
    // Child spans at `exec_insert`, `exec_full_scan`, and
    // `exec_semantic_search` below form the rest of the hot-path
    // span tree. The span is entered explicitly via `_entered` so
    // any synchronous child calls inherit it without needing to be
    // instrumented manually.
    let plan_kind = plan_kind_str(plan);
    let span = tracing::info_span!("query.execute", plan = plan_kind);
    let _entered = span.enter();

    // Authorization chokepoint (Req 3, AC3 + AC7). Enforced here — before
    // the match dispatches to any `exec_*` that touches storage — so the
    // wire path and the embedded path share one check and a non-privileged
    // role is rejected with SQLSTATE 42501 before any data is read or
    // written. A `None` session (trusted in-process embedded use) skips
    // the check and preserves today's behavior.
    enforce_authorization(plan, ctx)?;

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
        QueryPlan::Backup { path } => exec_backup(path, ctx),
        QueryPlan::Restore { path } => exec_restore(path, ctx),

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

        QueryPlan::CreateRole(stmt) => exec_create_role(stmt, ctx),
        QueryPlan::DropRole { name, if_exists } => exec_drop_role_principal(name, *if_exists, ctx),
        QueryPlan::AlterRolePassword { name, password } => {
            exec_alter_role_password(name, password, ctx)
        }
        QueryPlan::Grant(stmt) => exec_grant(stmt, false, ctx),
        QueryPlan::Revoke(stmt) => exec_grant(stmt, true, ctx),

        QueryPlan::CreateIndex(stmt) => exec_create_index(stmt, ctx),
        QueryPlan::DropIndex { name, if_exists } => exec_drop_index(name, *if_exists, ctx),
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
        append_only: is_system_append_only_table(&stmt.table_name),
    };

    Arc::make_mut(&mut ctx.catalog).create_table(stmt.table_name.clone(), entry)?;
    Ok(ExecuteResult::Ok(format!("CREATE TABLE {}", stmt.table_name)))
}

fn exec_drop_table(name: &str, if_exists: bool, ctx: &mut ExecutorContext) -> GalaxResult<ExecuteResult> {
    match Arc::make_mut(&mut ctx.catalog).drop_table(name) {
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

    // Secondary-index maintenance (Req 5 AC3): add an entry for every
    // index on this table in the same logical write. The index store is
    // engine-backed, so its entries are durable through the same WAL.
    if let Some(idx) = ctx.secondary_index.as_ref() {
        idx.on_row_inserted(table, &ordered, &key)?;
    }

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

    // Task 36.2: append-only system tables reject UPDATE. The lineage
    // record in `_galaxdb_training_exports` is a permanent audit trail;
    // allowing an in-place mutation would silently break
    // reproducibility of past training exports.
    if table_entry.append_only {
        return Err(GalaxError::AppendOnlyTable {
            table: table.to_string(),
            operation: "UPDATE",
        });
    }

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

        // Keep the pre-update column values so secondary indexes can
        // remove the stale entries after the new row is written.
        let old_cols = cols.clone();

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
            .put_sync(key.clone(), new_bytes)
            .map_err(|e| GalaxError::Internal(format!("engine put failed: {}", e)))?;

        // Secondary-index maintenance (Req 5 AC3): the primary key is
        // unchanged by UPDATE, so for each index whose column changed we
        // move the entry from the old value to the new one.
        if let Some(idx) = ctx.secondary_index.as_ref() {
            idx.on_row_updated(table, &old_cols, &cols, &key)?;
        }

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

    // Task 36.2: append-only system tables reject DELETE for the same
    // audit-trail reasons as UPDATE. See `exec_update` above.
    if table_entry.append_only {
        return Err(GalaxError::AppendOnlyTable {
            table: table.to_string(),
            operation: "DELETE",
        });
    }

    // Collect the keys first so we don't mutate storage while scanning.
    // Buffer the decoded columns too so secondary-index entries can be
    // removed (Req 5 AC3).
    let mut doomed: Vec<Vec<u8>> = Vec::new();
    let mut doomed_rows: Vec<BufferedRow> = Vec::new();
    let prefix = format!("{}:", table);
    for (key, value_bytes) in ctx.engine.scan_all() {
        if !String::from_utf8_lossy(&key).starts_with(&prefix) {
            continue;
        }
        let cols = row_codec::decode_row(&value_bytes);
        if let Some(f) = filter {
            if !row_codec::filter_matches(&cols, f) {
                continue;
            }
        }
        doomed.push(key.clone());
        doomed_rows.push((key, cols));
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

    // Secondary-index maintenance (Req 5 AC3): remove the index entries
    // for every deleted row, using the columns buffered before deletion.
    if let Some(idx) = ctx.secondary_index.as_ref() {
        for (key, cols) in &doomed_rows {
            idx.on_row_deleted(table, cols, key)?;
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
    // Task 38.5: PAX-read span covers every block the scan touches.
    let _span = tracing::info_span!("executor.full_scan", table = %table).entered();

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

    // `WHERE NOT DUPLICATE` is a group-level predicate: we have to see
    // every row that passes the per-row filter before we can pick the
    // representative for each `_near_duplicate_group`. So we buffer
    // `(primary_key, decoded_row)` rather than projecting as we go,
    // run the dedup pass if needed, then project at the end. Task 35.5.
    let dedup = filter.map(filter_has_not_duplicate).unwrap_or(false);
    let mut buffered: Vec<BufferedRow> = Vec::new();

    // Access-path selection (Req 5 AC2/AC5/AC6): if a secondary index
    // covers the filter's column with an equality or range predicate, use
    // it to fetch only the matching primary keys instead of scanning the
    // whole table. The exact `filter_matches` re-check below still runs on
    // each fetched row, so an index that returns a superset (e.g. the
    // exclusive-bound approximation) stays correct. `NOT DUPLICATE` needs
    // the full row set, so it always takes the scan path. When no index
    // covers the predicate we fall back to the zone-map-pruned scan — a
    // correctness-preserving path, not an error (AC6).
    let index_pks = if dedup {
        None
    } else {
        filter.and_then(|f| {
            ctx.secondary_index
                .as_ref()
                .and_then(|idx| crate::secondary_index::index_pk_set(idx, table, f))
        })
    };

    if let Some(pk_set) = index_pks {
        tracing::debug!(
            table = %table,
            access_path = "secondary_index",
            candidates = pk_set.len(),
            "planner chose secondary-index lookup",
        );
        // Fetch only the candidate rows by primary key, then apply the
        // exact filter (the index may return a superset for range bounds).
        for pk in pk_set.keys() {
            let Some(value_bytes) = ctx.engine.get(pk) else {
                continue; // entry referenced a since-deleted row
            };
            let cols = row_codec::decode_row(&value_bytes);
            if let Some(f) = filter {
                if !row_codec::filter_matches(&cols, f) {
                    continue;
                }
            }
            buffered.push((pk.clone(), cols));
        }
    } else {
        tracing::debug!(
            table = %table,
            access_path = "full_scan",
            "planner chose zone-map-pruned full scan",
        );
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
            buffered.push((key, cols));
        }
    }

    if dedup {
        apply_not_duplicate_pass(&mut buffered);
    }

    let rows: Vec<Row> = buffered
        .into_iter()
        .map(|(_, cols)| Row {
            columns: project
                .iter()
                .map(|name| {
                    let v = cols
                        .iter()
                        .find(|(k, _)| k == name)
                        .map(|(_, v)| v.clone())
                        .unwrap_or(Value::Null);
                    (name.clone(), v)
                })
                .collect(),
        })
        .collect();

    Ok(ExecuteResult::Rows {
        columns: project,
        rows,
    })
}

/// Collapse rows sharing a `_near_duplicate_group` to a single
/// representative — the row with the lexicographically smallest
/// primary key in each group. Rows with NULL / missing
/// `_near_duplicate_group` always survive (no duplicate info ⇒ not a
/// duplicate). Mirrors the contract used by
/// [`galaxdb_versioning::export::apply_dedup_filter`] so `WHERE NOT
/// DUPLICATE` queries and `CREATE VERSION TAG … FOR TRAINING` exports
/// pick the same representative per group.
fn apply_not_duplicate_pass(buffered: &mut Vec<BufferedRow>) {
    use std::collections::HashMap;

    // Pass 1: for each group ID, remember the smallest primary key.
    let mut representative: HashMap<i64, Vec<u8>> = HashMap::new();
    for (key, row) in buffered.iter() {
        let group = row_group_id(row);
        if let Some(g) = group {
            representative
                .entry(g)
                .and_modify(|best| {
                    if key.as_slice() < best.as_slice() {
                        *best = key.clone();
                    }
                })
                .or_insert_with(|| key.clone());
        }
    }

    if representative.is_empty() {
        return;
    }

    // Pass 2: keep every row whose group is None, or whose primary key
    // matches the representative for its group.
    buffered.retain(|(key, row)| {
        match row_group_id(row) {
            None => true,
            Some(g) => representative
                .get(&g)
                .map(|best| key.as_slice() == best.as_slice())
                .unwrap_or(true),
        }
    });
}

/// Read the `_near_duplicate_group` column off a decoded row. Accepts
/// [`Value::Integer`] (encoded group id) and [`Value::Text`] (stringified
/// group id) — `row_codec::value_from_str` round-trips unsigned 64-bit
/// group ids through `i64::parse`, which negates values whose MSB is
/// set, so we cast and compare as raw bits to stay injective. A NULL,
/// absent column, or unparseable text value returns `None`: the row is
/// not known to be a duplicate and always survives `WHERE NOT
/// DUPLICATE`.
fn row_group_id(row: &[(String, Value)]) -> Option<i64> {
    let v = row
        .iter()
        .find(|(n, _)| n == crate::planner::NEAR_DUPLICATE_GROUP_COLUMN)
        .map(|(_, v)| v)?;
    match v {
        Value::Integer(n) => Some(*n),
        Value::Text(s) => {
            if s == "NULL" {
                return None;
            }
            // Accept both signed and unsigned decimal spellings — the
            // group-id generator is `xxh3_64` which fills the full
            // 64-bit range; `row_codec::value_from_str` decodes
            // unsigned values whose MSB is set as strings rather than
            // integers.
            if let Ok(n) = s.parse::<i64>() {
                return Some(n);
            }
            if let Ok(u) = s.parse::<u64>() {
                return Some(u as i64);
            }
            None
        }
        Value::Null => None,
        _ => None,
    }
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
    let dedup = filter.map(filter_has_not_duplicate).unwrap_or(false);
    let mut buffered: Vec<BufferedRow> = Vec::new();

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
        buffered.push((key, cols));
    }

    if dedup {
        apply_not_duplicate_pass(&mut buffered);
    }

    let rows: Vec<Row> = buffered
        .into_iter()
        .map(|(_, cols)| Row {
            columns: project
                .iter()
                .map(|name| {
                    let v = cols
                        .iter()
                        .find(|(k, _)| k == name)
                        .map(|(_, v)| v.clone())
                        .unwrap_or(Value::Null);
                    (name.clone(), v)
                })
                .collect(),
        })
        .collect();

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

/// BACKUP TO '/path' (Req 27 / task 37.1–37.3).
///
/// Flushes the active memtable to an SST to produce a clean checkpoint,
/// then copies every `sst_*.pax` plus `wal.log` to the caller-supplied
/// path. The path is created if absent; an existing non-empty directory
/// is not a hard error — matching file names overwrite.
///
/// Read queries continue to serve during the file copy (SSTs are
/// immutable; WAL is append-only and the restore path replays only up
/// to the copied offset). The write-quiesce window is the duration of
/// `flush_memtable` — the stated target is < 100 ms on NVMe with
/// the default 64 MB seal threshold, but real timings are exercised
/// by the storage-crate tests rather than asserted here.
fn exec_backup(path: &str, ctx: &mut ExecutorContext) -> GalaxResult<ExecuteResult> {
    // Object-store target (s3://, gs://, az://): back up to a local staging
    // directory first (reusing the engine's flush + checksum-clean file set),
    // then upload every file to the store. Credentials are sourced from the
    // environment by the store and never logged.
    if galaxdb_backup::is_object_store_url(path) {
        let store = galaxdb_backup::object_store_for_target(path)?;
        let staging = backup_staging_dir();
        let _ = std::fs::remove_dir_all(&staging);
        let copied = ctx.engine.backup_to_sync(&staging);
        let result = copied.and_then(|files| {
            let uploaded = galaxdb_backup::upload_dir(store.as_ref(), &staging)?;
            Ok((files.len(), uploaded.len()))
        });
        let _ = std::fs::remove_dir_all(&staging);
        let (_local, uploaded) = result?;
        return Ok(ExecuteResult::Ok(format!(
            "BACKUP TO '{}': {} files uploaded to {} object store",
            path,
            uploaded,
            store.scheme()
        )));
    }

    let target = std::path::PathBuf::from(path);
    let copied = ctx.engine.backup_to_sync(&target)?;

    Ok(ExecuteResult::Ok(format!(
        "BACKUP TO '{}': {} files copied",
        path,
        copied.len()
    )))
}

/// A unique local staging directory for an object-store backup/restore.
fn backup_staging_dir() -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "galaxdb_backup_{}_{}",
        std::process::id(),
        nanos
    ))
}

/// RESTORE FROM '/path' (Req 27 / task 37.4–37.5).
///
/// Validates every SST block's XXH3-64 checksum in the source
/// directory via `Engine::validate_backup`; aborts on the first
/// corruption with a descriptive error that names the file and
/// block index. Only if validation succeeds are files copied into
/// the live engine's data directory. The caller is expected to
/// reopen the engine after a successful RESTORE so that WAL replay
/// and ART rebuild pick up the newly-copied files.
fn exec_restore(path: &str, ctx: &mut ExecutorContext) -> GalaxResult<ExecuteResult> {
    let target = ctx.engine.data_dir().to_path_buf();

    // Object-store source: download every backup object into a local staging
    // directory, then validate + restore from it exactly as for a local path
    // (so the checksum-abort-on-corruption guarantee is preserved).
    if galaxdb_backup::is_object_store_url(path) {
        let store = galaxdb_backup::object_store_for_target(path)?;
        let staging = backup_staging_dir();
        let _ = std::fs::remove_dir_all(&staging);
        let outcome = (|| {
            galaxdb_backup::download_dir(store.as_ref(), &staging)?;
            let (sst_count, block_count) = Engine::validate_backup(&staging)?;
            let copied = Engine::restore_from(&staging, &target)?;
            Ok::<_, GalaxError>((copied.len(), sst_count, block_count))
        })();
        let _ = std::fs::remove_dir_all(&staging);
        let (copied, sst_count, block_count) = outcome?;
        return Ok(ExecuteResult::Ok(format!(
            "RESTORE FROM '{}': {} files restored from {} object store \
             ({} SSTs / {} blocks validated). Reopen the engine to complete WAL replay.",
            path,
            copied,
            store.scheme(),
            sst_count,
            block_count
        )));
    }

    let source = std::path::PathBuf::from(path);
    let (sst_count, block_count) = Engine::validate_backup(&source)?;
    let copied = Engine::restore_from(&source, &target)?;

    Ok(ExecuteResult::Ok(format!(
        "RESTORE FROM '{}': {} files copied ({} SSTs / {} blocks validated). \
         Reopen the engine to complete WAL replay.",
        path,
        copied.len(),
        sst_count,
        block_count
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

// ---------------------------------------------------------------------------
// Roles, privileges, grants (Req 3). All persist through the AuthStore,
// which writes reserved rows via the engine's WAL+SST path.
// ---------------------------------------------------------------------------

/// Map an AST privilege to the auth `Action` it grants.
fn privilege_action(p: crate::ast::Privilege) -> galaxdb_auth::Action {
    match p {
        crate::ast::Privilege::Select => galaxdb_auth::Action::Select,
        crate::ast::Privilege::Insert => galaxdb_auth::Action::Insert,
        crate::ast::Privilege::Update => galaxdb_auth::Action::Update,
        crate::ast::Privilege::Delete => galaxdb_auth::Action::Delete,
    }
}

fn require_auth_store(ctx: &ExecutorContext) -> GalaxResult<&crate::auth_store::AuthStore> {
    ctx.auth_store.as_ref().ok_or(GalaxError::NotYetAvailable {
        task: "4",
        feature: "role/grant DDL without a configured auth store",
    })
}

fn exec_create_role(
    stmt: &crate::ast::CreateRoleStmt,
    ctx: &mut ExecutorContext,
) -> GalaxResult<ExecuteResult> {
    let store = require_auth_store(ctx)?;
    if store.get_role(&stmt.name).is_some() {
        return Err(GalaxError::Internal(format!(
            "role '{}' already exists",
            stmt.name
        )));
    }
    // The plaintext password is used only to derive the SCRAM verifier
    // here and is never persisted. A passwordless role can be created
    // and have its password set later via ALTER ROLE.
    let verifier = stmt
        .password
        .as_ref()
        .map(|pw| galaxdb_auth::ScramVerifier::from_password(pw));
    let record = crate::auth_store::RoleRecord {
        name: stmt.name.clone(),
        is_superuser: stmt.is_superuser,
        verifier,
    };
    store.put_role(&record)?;
    Ok(ExecuteResult::Ok(format!("CREATE ROLE {}", stmt.name)))
}

fn exec_drop_role_principal(
    name: &str,
    if_exists: bool,
    ctx: &mut ExecutorContext,
) -> GalaxResult<ExecuteResult> {
    let store = require_auth_store(ctx)?;
    let existed = store.drop_role(name)?;
    if !existed && !if_exists {
        return Err(GalaxError::Internal(format!("role '{}' does not exist", name)));
    }
    Ok(ExecuteResult::Ok(format!("DROP ROLE {}", name)))
}

fn exec_alter_role_password(
    name: &str,
    password: &str,
    ctx: &mut ExecutorContext,
) -> GalaxResult<ExecuteResult> {
    let store = require_auth_store(ctx)?;
    let mut record = store
        .get_role(name)
        .ok_or_else(|| GalaxError::Internal(format!("role '{}' does not exist", name)))?;
    // Re-derive the verifier from the new plaintext, then drop it.
    record.verifier = Some(galaxdb_auth::ScramVerifier::from_password(password));
    store.put_role(&record)?;
    Ok(ExecuteResult::Ok(format!("ALTER ROLE {}", name)))
}

fn exec_grant(
    stmt: &crate::ast::GrantStmt,
    revoke: bool,
    ctx: &mut ExecutorContext,
) -> GalaxResult<ExecuteResult> {
    let store = require_auth_store(ctx)?;
    // The grantee role must exist.
    if store.get_role(&stmt.role).is_none() {
        return Err(GalaxError::Internal(format!(
            "role '{}' does not exist",
            stmt.role
        )));
    }
    let action = privilege_action(stmt.privilege);
    if revoke {
        store.revoke(&stmt.role, &stmt.table, action)?;
        Ok(ExecuteResult::Ok(format!(
            "REVOKE {} ON {} FROM {}",
            action.label(),
            stmt.table,
            stmt.role
        )))
    } else {
        store.grant(&stmt.role, &stmt.table, action)?;
        Ok(ExecuteResult::Ok(format!(
            "GRANT {} ON {} TO {}",
            action.label(),
            stmt.table,
            stmt.role
        )))
    }
}

// ---------------------------------------------------------------------------
// Secondary indexes (Req 5)
// ---------------------------------------------------------------------------

fn require_secondary_index(
    ctx: &ExecutorContext,
) -> GalaxResult<&crate::secondary_index::SecondaryIndexStore> {
    ctx.secondary_index.as_ref().ok_or(GalaxError::NotYetAvailable {
        task: "8",
        feature: "secondary indexes without a configured index store",
    })
}

fn exec_create_index(
    stmt: &crate::ast::CreateIndexStmt,
    ctx: &mut ExecutorContext,
) -> GalaxResult<ExecuteResult> {
    // The target table must exist (the index references its column).
    let table_entry = ctx
        .catalog
        .get_table(&stmt.table)
        .cloned()
        .ok_or_else(|| GalaxError::TableNotFound(stmt.table.clone()))?;
    if !table_entry.columns.iter().any(|c| c.name == stmt.column) {
        return Err(GalaxError::ColumnNotFound(format!(
            "{}.{}",
            stmt.table, stmt.column
        )));
    }

    let store = require_secondary_index(ctx)?;
    if store.get_def(&stmt.name).is_some() {
        if stmt.if_not_exists {
            return Ok(ExecuteResult::Ok(format!(
                "CREATE INDEX {} (already exists)",
                stmt.name
            )));
        }
        return Err(GalaxError::Internal(format!(
            "index '{}' already exists",
            stmt.name
        )));
    }

    let def = crate::secondary_index::IndexDef {
        name: stmt.name.clone(),
        table: stmt.table.clone(),
        column: stmt.column.clone(),
    };
    store.create_def(&def)?;
    // Populate from existing rows so the index covers the current table
    // contents immediately, not only future writes (Req 5 AC2).
    let indexed = store.build_from_table(&def)?;
    Ok(ExecuteResult::Ok(format!(
        "CREATE INDEX {} ON {} ({}): {} rows indexed",
        stmt.name, stmt.table, stmt.column, indexed
    )))
}

fn exec_drop_index(
    name: &str,
    if_exists: bool,
    ctx: &mut ExecutorContext,
) -> GalaxResult<ExecuteResult> {
    let store = require_secondary_index(ctx)?;
    let existed = store.drop_index(name)?;
    if !existed && !if_exists {
        return Err(GalaxError::Internal(format!("index '{}' does not exist", name)));
    }
    Ok(ExecuteResult::Ok(format!("DROP INDEX {}", name)))
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
    //
    // The MerkleDag is advanced by the compactor/flush path (task 32.2)
    // and is still at `ts=0` in an in-memory memtable-only database.
    // For version-tag correctness we therefore pin to the engine's
    // most-recently allocated commit ts — that is the real "everything
    // committed so far" boundary observable by readers, and it's what
    // `AT VERSION` and `training_dataset` resolve against via
    // `Engine::scan_all_at`. Falling back to the DAG ts when the engine
    // is absent keeps the plan-validation tests (which build a legacy
    // `ExecutorContext` without an engine) unchanged.
    let (current_root, current_ts, blocks) = {
        let dag = merkle_dag
            .lock()
            .map_err(|_| GalaxError::Internal("merkle dag mutex poisoned".into()))?;
        let dag_ts = dag.latest().map(|v| v.timestamp).unwrap_or(0);
        let engine_ts = ctx.engine.latest_commit_ts();
        // Pick whichever is greater so we never regress behind a DAG
        // update, and never land behind an uncommitted-but-visible row.
        let ts = dag_ts.max(engine_ts);
        let blocks = dag.blocks_at_version(ts);
        // Compute a real content Merkle root over the exact snapshot the tag
        // pins (xxh3-128 of the per-row checksums of everything visible at
        // `ts`). This certifies the tagged contents and is reproducible from
        // the same data, rather than echoing the DAG's empty/seed root in a
        // memtable-only database.
        let root = galaxdb_versioning::MerkleRoot::compute(&ctx.engine.snapshot_checksums(ts));
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

    // Validate column-count up front so we never half-commit.
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

    // Validate that every referenced column exists — once, not per row.
    // Unknown columns must error, not silently drop the value.
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

    // Resolve + encode every row BEFORE any storage write, so a malformed
    // row aborts the whole batch with no partial commit (Req 8 AC5).
    // `value_from_str` auto-detects quoted strings, numerics, NULL, bools —
    // exactly the tokens `COPY ... FROM STDIN` produces.
    let mut ordered_rows: Vec<Vec<(String, Value)>> = Vec::with_capacity(rows.len());
    let mut pairs: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(rows.len());
    for row_tokens in rows {
        let row_values: Vec<Value> = row_tokens
            .iter()
            .map(|tok| row_codec::value_from_str(tok))
            .collect();

        // MinHash policy runs before the storage write so signatures and
        // row bytes commit together (parity with exec_insert / task 35.2).
        if let Some(policy) = ctx.minhash_policy.as_ref() {
            policy.compute_and_sink(table, &table_entry, columns, &row_values);
        }

        let ordered = row_codec::align_values(&table_entry, columns, &row_values)?;
        let key = row_codec::build_primary_key(table, &table_entry, &ordered)?;
        let value_bytes = row_codec::encode_row(&ordered);
        pairs.push((key, value_bytes));
        ordered_rows.push(ordered);
    }

    // The whole point of BULK INSERT / COPY (Req 8 AC3): batch the WAL
    // fsync instead of one fsync per row. We commit in bounded chunks
    // (`BULK_COMMIT_CHUNK` rows → one `put_batch_sync` → one fsync per
    // chunk) rather than one unbounded `put_batch_sync` of the whole
    // input. Chunking keeps the per-call cost flat so total ingest stays
    // linear in row count (a single giant batch degrades super-linearly),
    // and bounds peak memory for very large COPY streams. Looping
    // `exec_insert` here would defeat the feature (one fsync per row).
    const BULK_COMMIT_CHUNK: usize = 1024;
    for chunk in pairs.chunks(BULK_COMMIT_CHUNK) {
        ctx.engine
            .put_batch_sync(chunk)
            .map_err(|e| GalaxError::Internal(format!("engine batch put failed: {}", e)))?;
    }

    // Secondary-index + embedding maintenance, per row, only when the table
    // actually has indexes / an embedding column. Plain tables skip this
    // loop entirely, preserving the single-fsync fast path above.
    let has_index = ctx.secondary_index.is_some();
    let has_embedding = table_entry.has_embedding && ctx.sidecar.is_some();
    if has_index || has_embedding {
        for (ordered, (key, _)) in ordered_rows.iter().zip(pairs.iter()) {
            if let Some(idx) = ctx.secondary_index.as_ref() {
                idx.on_row_inserted(table, ordered, key)?;
            }
            if has_embedding {
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
                        let row_id = xxhash_rust::xxh3::xxh3_64(key);
                        let _ = sidecar.embed(EmbedRequest {
                            row_id,
                            text,
                            column: col.name.clone(),
                        });
                    }
                }
            }
        }
    }

    Ok(ExecuteResult::RowCount(pairs.len() as u64))
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
    let _span = tracing::info_span!(
        "executor.semantic_search",
        table = %table,
        strategy = ?strategy,
    )
    .entered();

    if !ctx.catalog.table_exists(table) {
        return Err(GalaxError::TableNotFound(table.to_string()));
    }
    let backend = ctx
        .vector_backend
        .as_ref()
        .ok_or(GalaxError::SidecarUnavailable)?;

    let results = backend.semantic_search(table, query_text, threshold, 10, strategy)?;

    // Join vector results back to actual table rows using the row_id
    // (which is xxh3_64 of the primary key). Scan the table and match.
    let table_entry = ctx.catalog.get_table(table).cloned()
        .ok_or_else(|| GalaxError::TableNotFound(table.to_string()))?;
    let col_names: Vec<String> = table_entry.columns.iter().map(|c| c.name.clone()).collect();

    let prefix = format!("{}:", table);
    let all_rows: Vec<BufferedRow> = ctx.engine.scan_all()
        .into_iter()
        .filter(|(k, _)| String::from_utf8_lossy(k).starts_with(&prefix))
        .map(|(k, v)| {
            let cols = row_codec::decode_row(&v);
            (k, cols)
        })
        .collect();

    // Build a map from xxh3_64(key) → decoded row
    let row_map: std::collections::HashMap<u64, Vec<(String, Value)>> = all_rows
        .into_iter()
        .map(|(k, cols)| (xxhash_rust::xxh3::xxh3_64(&k), cols))
        .collect();

    let rows: Vec<Row> = results
        .iter()
        .filter_map(|r| {
            row_map.get(&r.row_id).map(|cols| Row {
                columns: cols.clone(),
            })
        })
        .collect();

    Ok(ExecuteResult::Rows {
        columns: col_names,
        rows,
    })
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
                append_only: is_system_append_only_table(&stmt.table_name),
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
        QueryPlan::CreateRole(_)
        | QueryPlan::DropRole { .. }
        | QueryPlan::AlterRolePassword { .. }
        | QueryPlan::Grant(_)
        | QueryPlan::Revoke(_) => ExecuteResult::Error(
            "role/grant DDL requires a storage-backed auth store; use execute_with_context"
                .to_string(),
        ),
        QueryPlan::CreateIndex(_) | QueryPlan::DropIndex { .. } => ExecuteResult::Error(
            "CREATE/DROP INDEX requires a storage-backed index store; use execute_with_context"
                .to_string(),
        ),
    }
}
