//! GalaxDB Query — the anti-corruption boundary for the relational /
//! analytical query layer (HTAP spec ADR-0001, Req 7).
//!
//! This crate is the **only** place in the workspace permitted to depend on
//! Apache DataFusion. Its public surface is GalaxDB-owned types exclusively —
//! no `datafusion::` type appears in any signature here, so the volatile
//! DataFusion API never leaks into the rest of the engine, the wire protocol,
//! or persisted formats. A CI containment guard (`tests/containment.rs`)
//! enforces this mechanically.
//!
//! Apache Arrow *does* appear in the public surface: it is the stable
//! columnar interchange the storage layer produces and the query layer
//! consumes (`ArrowSource`). Arrow is a separate, foundational format — the
//! anti-corruption boundary is specifically about the DataFusion query API.
//!
//! # Surface (task 2 skeleton)
//!
//! - [`ReadSnapshot`] — MVCC read point (latest / version tag / timestamp).
//! - [`ScanPredicate`] — GalaxDB-owned predicate IR (no `datafusion::Expr`).
//! - [`ScanRequest`] — projection + predicates + limit + snapshot for a scan.
//! - [`ArrowSource`] — what the query layer needs from storage (implemented in
//!   `galaxdb-embedded`, HTAP tasks 7/10).
//! - [`QueryBackend`] — the pluggable execution backend (the DataFusion impl
//!   lands in HTAP task 11; the trait is the Req 7.5 escape hatch).
//! - [`GalaxLogicalPlan`] / [`QueryContext`] — backend execution inputs.
//!
//! The concrete `DataFusionBackend` and the `ColumnType → Arrow` schema
//! mapping are added by later HTAP tasks; this module defines the contract.

use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;

use galaxdb_common::{GalaxResult, Timestamp};
use galaxdb_sql::planner::Value;

pub mod schema;

/// MVCC read point for a scan. The same snapshot mechanism backs both
/// `AT VERSION` time-travel and the read timestamp of an open transaction
/// (HTAP design §3.5, §3.6), so native and analytical scans observe a
/// consistent view.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ReadSnapshot {
    /// Read the latest committed data.
    #[default]
    Latest,
    /// Read as of a specific commit timestamp.
    AsOfTimestamp(Timestamp),
    /// Read as of a named version tag, resolved by the engine.
    AsOfTag(String),
}

/// Comparison operator for a [`ScanPredicate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredicateOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// A GalaxDB-owned predicate the query layer may push down to a storage
/// scan. This is deliberately **not** `datafusion::Expr`: the backend
/// translates a DataFusion filter into this IR (or reports the predicate as
/// not pushable) so storage never sees a DataFusion type.
#[derive(Debug, Clone, PartialEq)]
pub enum ScanPredicate {
    /// `column <op> value`.
    Compare {
        column: String,
        op: PredicateOp,
        value: Value,
    },
    /// `column IS NULL` / `column IS NOT NULL`.
    IsNull { column: String, negated: bool },
    /// Conjunction of predicates (all must hold).
    And(Vec<ScanPredicate>),
    /// Disjunction of predicates (any may hold).
    Or(Vec<ScanPredicate>),
}

/// A request to scan one table, carrying everything the storage layer needs
/// to satisfy projection / predicate / limit pushdown under an MVCC
/// snapshot (HTAP design §3.1). Mirrors the information DataFusion's
/// `TableProvider::scan` provides, but in GalaxDB-owned types.
#[derive(Debug, Clone)]
pub struct ScanRequest {
    /// Table to scan.
    pub table: String,
    /// Column indices to project, or `None` for all columns.
    pub projection: Option<Vec<usize>>,
    /// Predicates the backend asked storage to enforce. Storage reports
    /// which it enforced exactly vs. which still need a re-check
    /// (see [`ArrowSource::supports_predicate`]).
    pub filters: Vec<ScanPredicate>,
    /// Row limit hint, or `None` for unbounded.
    pub limit: Option<usize>,
    /// MVCC read point.
    pub snapshot: ReadSnapshot,
}

impl ScanRequest {
    /// A full scan of `table` at the latest snapshot.
    pub fn full(table: impl Into<String>) -> Self {
        ScanRequest {
            table: table.into(),
            projection: None,
            filters: Vec::new(),
            limit: None,
            snapshot: ReadSnapshot::Latest,
        }
    }
}

/// How exactly a storage source can enforce a pushed predicate. Mirrors
/// DataFusion's `TableProviderFilterPushDown`, but GalaxDB-owned so the
/// distinction never requires importing a DataFusion type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredicateSupport {
    /// Storage cannot evaluate it; the backend must apply it after the scan.
    Unsupported,
    /// Storage prunes with it but may return extra rows; backend re-checks.
    Inexact,
    /// Storage enforces it precisely; backend need not re-check.
    Exact,
}

/// A stream of Arrow record batches, each fallible. Boxed so trait objects
/// stay simple and the producer (storage) controls batching.
pub type BatchStream = Box<dyn Iterator<Item = GalaxResult<RecordBatch>> + Send>;

/// The result of executing a query: a stream of Arrow batches. The wire
/// layer converts these to PostgreSQL `DataRow`s (HTAP task 15).
pub type ResultStream = BatchStream;

/// What the query layer needs from storage. Implemented in
/// `galaxdb-embedded` over both the columnar PAX path and the legacy
/// `col=v|...` decode bridge (HTAP tasks 7/8/10), so both look identical to
/// a backend. Arrow-native by design: no per-row decode on the hot path.
pub trait ArrowSource: Send + Sync {
    /// The Arrow schema of `table` (logical SQL columns mapped to Arrow).
    fn schema(&self, table: &str) -> GalaxResult<SchemaRef>;

    /// Stream Arrow batches for `req`, honoring projection / pushed
    /// predicates / limit / MVCC snapshot.
    fn scan(&self, req: ScanRequest) -> GalaxResult<BatchStream>;

    /// Report how precisely storage can enforce `predicate` for `table`,
    /// so the backend knows whether to re-check it (HTAP Property 5).
    /// Defaults to `Unsupported` (always re-checked) — a safe default.
    fn supports_predicate(&self, _table: &str, _predicate: &ScanPredicate) -> PredicateSupport {
        PredicateSupport::Unsupported
    }

    /// Write Arrow batches into `table` (INSERT ... SELECT, CTAS). Returns
    /// the number of rows written.
    fn insert(&self, table: &str, batches: BatchStream) -> GalaxResult<u64>;
}

/// Execution context for a backend run: the read snapshot plus a place for
/// future session settings (memory budget, target partitions).
#[derive(Debug, Clone, Default)]
pub struct QueryContext {
    /// MVCC read point for every scan in this execution.
    pub snapshot: ReadSnapshot,
}

/// The body of a [`GalaxLogicalPlan`]. The HTAP planner (task 13) replaces
/// this with the full relational IR (joins / aggregates / subqueries /
/// windows / order / limit). The skeleton carries the analytical SQL text
/// that `galaxdb-sql` validated and classified as backend-bound, which is a
/// real, executable representation — not a placeholder stub.
#[derive(Debug, Clone, PartialEq)]
pub enum PlanBody {
    /// Validated analytical SQL to be planned and executed by the backend
    /// against the registered [`ArrowSource`]s.
    AnalyticalSql(String),
}

/// A GalaxDB-owned logical plan handed to a [`QueryBackend`]. Owns the set
/// of tables it reads so the backend can register their [`ArrowSource`]s
/// before execution.
#[derive(Debug, Clone, PartialEq)]
pub struct GalaxLogicalPlan {
    /// Tables this plan reads.
    pub referenced_tables: Vec<String>,
    /// The plan body.
    pub body: PlanBody,
}

/// The pluggable relational/analytical execution backend. The DataFusion
/// implementation (HTAP task 11) is confined to this crate; this trait is
/// the Req 7.5 escape hatch that lets an alternative backend be substituted
/// without changing any other crate's public surface.
pub trait QueryBackend: Send + Sync {
    /// Register a table's [`ArrowSource`] so the backend can scan it.
    fn register(&self, table: &str, source: Arc<dyn ArrowSource>) -> GalaxResult<()>;

    /// Execute a logical plan, returning a stream of Arrow result batches.
    fn execute(&self, plan: GalaxLogicalPlan, ctx: &QueryContext) -> GalaxResult<ResultStream>;
}
