//! Query planner — translates parsed AuroraSQL statements into query plans.
//!
//! The planner chooses execution strategies based on table statistics:
//! - Point lookups via ART index
//! - Full scans with zone-map pruning + Bloom filter checks
//! - Adaptive HNSW vs brute-force for semantic search (Req 22)

use crate::ast::*;

/// A query execution plan.
#[derive(Debug, Clone, PartialEq)]
pub enum QueryPlan {
    /// Point lookup by primary key.
    PointLookup { table: String, key: Vec<u8> },
    /// Full table scan with optional filter and zone-map pruning.
    FullScan {
        table: String,
        filter: Option<FilterExpr>,
        columns: Vec<String>,
    },
    /// Time-travel scan: a FullScan constrained to rows whose
    /// `commit_timestamp` is visible at the resolved version
    /// (timestamp or named tag). The `at` field carries the parsed
    /// `AT VERSION ...` fragment from the SQL.
    ///
    /// SEMANTIC_MATCH combined with `AT VERSION` is rejected at parse
    /// time unless the caller explicitly passes `CONSISTENCY
    /// 'SEMANTIC_FRESH'`, in which case the plan also carries the
    /// consistency mode so the executor can emit the SEMANTIC_FRESH
    /// warning in the result metadata (task 32.6).
    FullScanAtVersion {
        table: String,
        filter: Option<FilterExpr>,
        columns: Vec<String>,
        at: AtVersionExpr,
    },
    /// Semantic vector search.
    SemanticSearch {
        table: String,
        column: String,
        query_text: String,
        threshold: f64,
        strategy: SearchStrategy,
        /// Top-k bound from a SQL `LIMIT`. `None` uses the executor's
        /// default page size; `Some(n)` returns the n nearest matches.
        limit: Option<usize>,
    },
    /// Hybrid: structured filter + semantic search.
    HybridSearch {
        table: String,
        filter: FilterExpr,
        semantic: SemanticMatchExpr,
        strategy: SearchStrategy,
        /// Top-k bound from a SQL `LIMIT` (see `SemanticSearch::limit`).
        limit: Option<usize>,
    },
    /// Hybrid search over a historical snapshot (task 32.6 SEMANTIC_FRESH).
    /// SEMANTIC_MATCH combined with AT VERSION is only legal when the
    /// user opts into `CONSISTENCY 'SEMANTIC_FRESH'` or
    /// `'ROW_SNAPSHOT'`. The executor attaches a warning row to the
    /// result metadata for SEMANTIC_FRESH and rejects the combination
    /// if no consistency mode is set.
    HybridSearchAtVersion {
        table: String,
        filter: Option<FilterExpr>,
        semantic: SemanticMatchExpr,
        strategy: SearchStrategy,
        at: AtVersionExpr,
        /// Top-k bound from a SQL `LIMIT` (see `SemanticSearch::limit`).
        limit: Option<usize>,
    },
    /// INSERT a single row.
    Insert {
        table: String,
        columns: Vec<String>,
        values: Vec<Value>,
    },
    /// UPDATE rows matching a filter. Each assignment's value is a scalar
    /// expression evaluated per row against the old (pre-update) values —
    /// e.g. `SET bal = bal - 30` (HTAP: real expression evaluation, not a
    /// literal), per PostgreSQL UPDATE semantics.
    Update {
        table: String,
        assignments: Vec<(String, crate::scalar::ScalarExpr)>,
        filter: Option<FilterExpr>,
    },
    /// DELETE rows matching a filter.
    Delete {
        table: String,
        filter: Option<FilterExpr>,
    },
    /// BULK INSERT — write multiple rows. Phase L: real implementation.
    /// The executor loops `Engine::put_sync` per row (sharing the
    /// normal INSERT path's codec + sidecar trigger). The Month-4
    /// "direct PAX write, bypass memtable" fast path is an
    /// optimisation added later; correctness ships now.
    BulkInsert {
        table: String,
        columns: Vec<String>,
        /// One entry per row. Raw string tokens identical to what
        /// `sqlparser` emits for `INSERT … VALUES (…)`.
        values: Vec<Vec<String>>,
    },
    /// CREATE TABLE.
    CreateTable(CreateTableStmt),
    /// DROP TABLE.
    DropTable { name: String, if_exists: bool },
    /// CREATE VERSION TAG.
    CreateVersionTag(CreateVersionTagStmt),
    /// BACKUP TO path.
    Backup { path: String },
    /// RESTORE FROM path.
    Restore { path: String },
    /// ANALYZE table.
    Analyze { table: String },
    /// SHOW EMBEDDING HEALTH.
    ShowEmbeddingHealth { table: Option<String> },
    /// CREATE ROLE (Req 3).
    CreateRole(crate::ast::CreateRoleStmt),
    /// DROP ROLE.
    DropRole { name: String, if_exists: bool },
    /// ALTER ROLE ... PASSWORD.
    AlterRolePassword { name: String, password: String },
    /// GRANT privilege ON table TO role.
    Grant(crate::ast::GrantStmt),
    /// REVOKE privilege ON table FROM role.
    Revoke(crate::ast::GrantStmt),
    /// CREATE INDEX (Req 5).
    CreateIndex(crate::ast::CreateIndexStmt),
    /// DROP INDEX.
    DropIndex { name: String, if_exists: bool },
    /// ALTER TABLE ... SET STORAGE {COLUMNAR|LEGACY} (HTAP task 9).
    AlterTableSetStorage {
        table: String,
        mode: galaxdb_common::StorageMode,
    },
}

/// Search strategy chosen by the adaptive planner (Req 22).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchStrategy {
    /// HNSW graph traversal with post-filtering. Used when filter cardinality
    /// is moderate to high.
    HnswWithPostFilter,
    /// Brute-force scan over the filtered candidate set. Used when filter
    /// cardinality is very low (< 1000 rows or < 0.1% of table).
    BruteForceFiltered,
}

/// Canonical name of the system column populated by MinHash's
/// near-duplicate grouping job (task 35.4). A non-NULL value is the
/// group ID that row shares with its near-duplicate peers; a NULL or
/// missing value means "not known to be a duplicate" and the row
/// always passes `WHERE NOT DUPLICATE`.
pub const NEAR_DUPLICATE_GROUP_COLUMN: &str = "_near_duplicate_group";

/// Does `filter` contain a `NotDuplicate` predicate anywhere?
///
/// Walks the expression tree including `And` / `Or` children. This
/// decides whether the executor needs to buffer the row set and run
/// the group-level dedup pass in addition to per-row filtering.
pub fn filter_has_not_duplicate(filter: &FilterExpr) -> bool {
    match filter {
        FilterExpr::NotDuplicate => true,
        FilterExpr::And(a, b) | FilterExpr::Or(a, b) => {
            filter_has_not_duplicate(a) || filter_has_not_duplicate(b)
        }
        _ => false,
    }
}

/// A simple filter expression for WHERE clauses.
#[derive(Debug, Clone, PartialEq)]
pub enum FilterExpr {
    /// column = value
    Eq { column: String, value: Value },
    /// column < value
    Lt { column: String, value: Value },
    /// column > value
    Gt { column: String, value: Value },
    /// column <= value
    Le { column: String, value: Value },
    /// column >= value
    Ge { column: String, value: Value },
    /// column != value
    Ne { column: String, value: Value },
    /// expr AND expr
    And(Box<FilterExpr>, Box<FilterExpr>),
    /// expr OR expr
    Or(Box<FilterExpr>, Box<FilterExpr>),
    /// `WHERE NOT DUPLICATE` — exclude rows in near-duplicate groups,
    /// keeping one deterministic representative per group (Req 26,
    /// task 35.5). Rows with a non-NULL `_near_duplicate_group` column
    /// collapse to the row with the lexicographically smallest primary
    /// key in their group; rows with NULL / missing group pass through
    /// (no duplicate info ⇒ not a duplicate).
    ///
    /// The filter is a group-level predicate: it cannot be evaluated
    /// per-row without knowing every other row's group. The executor
    /// buffers the full row set for a scan once before applying the
    /// dedup pass — this matches the contract used by the Lance
    /// training exporter's `apply_dedup_filter` so `SELECT … WHERE NOT
    /// DUPLICATE` and `CREATE VERSION TAG … FOR TRAINING` export agree
    /// on which row represents each near-duplicate cluster.
    NotDuplicate,
}

/// A typed value in a query plan.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Integer(i64),
    Float(f64),
    Text(String),
    Bool(bool),
    Null,
    Blob(Vec<u8>),
    /// A 1-dimensional SQL array (e.g. `int[]`, `text[]`). Elements share
    /// the array's logical element type; the per-element `Value` carries
    /// the physical representation. Introduced with the HTAP type-system
    /// extension (Req 5.3). Multi-dimensional arrays are not represented.
    Array(Vec<Value>),
}

/// Statistics hint for the adaptive planner.
#[derive(Debug, Clone)]
pub struct PlannerStats {
    pub row_count: u64,
    pub filter_selectivity: f64,
}

impl PlannerStats {
    /// Estimate the number of rows matching the filter.
    pub fn estimated_cardinality(&self) -> u64 {
        (self.row_count as f64 * self.filter_selectivity) as u64
    }
}

/// Choose the search strategy based on filter cardinality (Req 22).
///
/// - If estimated matching rows < 1000 or < 0.1% of table → BruteForceFiltered
/// - Otherwise → HnswWithPostFilter
pub fn choose_search_strategy(stats: &PlannerStats) -> SearchStrategy {
    let cardinality = stats.estimated_cardinality();
    let fraction = stats.filter_selectivity;

    if cardinality < 1000 || fraction < 0.001 {
        SearchStrategy::BruteForceFiltered
    } else {
        SearchStrategy::HnswWithPostFilter
    }
}

/// Plan a CREATE TABLE statement.
pub fn plan_create_table(stmt: CreateTableStmt) -> QueryPlan {
    QueryPlan::CreateTable(stmt)
}

/// Plan a DROP TABLE statement.
pub fn plan_drop_table(name: String, if_exists: bool) -> QueryPlan {
    QueryPlan::DropTable { name, if_exists }
}

/// Plan an INSERT statement from sqlparser AST.
pub fn plan_insert(table: String, columns: Vec<String>, values: Vec<Value>) -> QueryPlan {
    QueryPlan::Insert {
        table,
        columns,
        values,
    }
}

/// Plan a DELETE statement.
pub fn plan_delete(table: String, filter: Option<FilterExpr>) -> QueryPlan {
    QueryPlan::Delete { table, filter }
}

/// Plan an UPDATE statement.
pub fn plan_update(
    table: String,
    assignments: Vec<(String, crate::scalar::ScalarExpr)>,
    filter: Option<FilterExpr>,
) -> QueryPlan {
    QueryPlan::Update {
        table,
        assignments,
        filter,
    }
}

/// Plan a SELECT statement (simplified — full SQL planning is complex).
pub fn plan_select(
    table: String,
    columns: Vec<String>,
    filter: Option<FilterExpr>,
) -> QueryPlan {
    QueryPlan::FullScan {
        table,
        filter,
        columns,
    }
}

/// Plan a semantic search with adaptive strategy selection.
///
/// `limit` is the SQL `LIMIT` count (top-k bound) when the query carries
/// one, or `None` to use the executor's default page size.
pub fn plan_semantic_search(
    table: String,
    semantic: SemanticMatchExpr,
    filter: Option<FilterExpr>,
    stats: Option<&PlannerStats>,
    limit: Option<usize>,
) -> QueryPlan {
    if let Some(filter) = filter {
        let strategy = stats
            .map(choose_search_strategy)
            .unwrap_or(SearchStrategy::HnswWithPostFilter);

        QueryPlan::HybridSearch {
            table,
            filter,
            semantic,
            strategy,
            limit,
        }
    } else {
        QueryPlan::SemanticSearch {
            table,
            column: semantic.column.clone(),
            query_text: semantic.query.clone(),
            threshold: semantic.threshold,
            strategy: SearchStrategy::HnswWithPostFilter,
            limit,
        }
    }
}
