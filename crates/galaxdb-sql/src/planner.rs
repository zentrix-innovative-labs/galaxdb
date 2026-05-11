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
    },
    /// Hybrid: structured filter + semantic search.
    HybridSearch {
        table: String,
        filter: FilterExpr,
        semantic: SemanticMatchExpr,
        strategy: SearchStrategy,
    },
    /// INSERT a single row.
    Insert {
        table: String,
        columns: Vec<String>,
        values: Vec<Value>,
    },
    /// UPDATE rows matching a filter.
    Update {
        table: String,
        assignments: Vec<(String, Value)>,
        filter: Option<FilterExpr>,
    },
    /// DELETE rows matching a filter.
    Delete {
        table: String,
        filter: Option<FilterExpr>,
    },
    /// BULK INSERT — bypass memtable, write PAX blocks directly.
    BulkInsert { table: String },
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
    assignments: Vec<(String, Value)>,
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
pub fn plan_semantic_search(
    table: String,
    semantic: SemanticMatchExpr,
    filter: Option<FilterExpr>,
    stats: Option<&PlannerStats>,
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
        }
    } else {
        QueryPlan::SemanticSearch {
            table,
            column: semantic.column.clone(),
            query_text: semantic.query.clone(),
            threshold: semantic.threshold,
            strategy: SearchStrategy::HnswWithPostFilter,
        }
    }
}
