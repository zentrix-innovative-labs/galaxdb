//! Statement classifier (HTAP query engine, task 14).
//!
//! Decides whether a parsed `SELECT` runs on GalaxDB's **native** path
//! (single-table filtered scan, point lookup, vector search, time-travel)
//! or the **analytical** path (joins, aggregates, GROUP BY, subqueries, set
//! operations, DISTINCT, HAVING, window functions, ORDER BY / LIMIT /
//! OFFSET), which is executed by DataFusion behind `galaxdb-query`
//! (design §2.1).
//!
//! The classification is conservative: anything that is not provably a
//! simple single-table scan is routed to DataFusion, which either executes
//! it or returns a typed error — never a silent wrong result.

use sqlparser::ast::{Expr, GroupByExpr, Query, SelectItem, SetExpr, TableFactor};

/// Where a statement should execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatementClass {
    /// GalaxDB's native executor (single-table scan / point lookup /
    /// vector / time-travel).
    Native,
    /// The DataFusion-backed relational/analytical engine.
    Analytical,
}

/// Classify a parsed `SELECT` query.
pub fn classify_query(q: &Query) -> StatementClass {
    // ORDER BY / LIMIT / OFFSET are not honored by the native scan path
    // (they parse but are ignored), so route them to DataFusion where they
    // execute correctly.
    if has_order_by(q) || q.limit.is_some() || q.offset.is_some() {
        return StatementClass::Analytical;
    }

    if body_is_native_single_table(q) {
        StatementClass::Native
    } else {
        StatementClass::Analytical
    }
}

/// The `LIMIT` count of `q`, if present and a plain non-negative integer
/// literal. The semantic vector path uses this to treat
/// `SEMANTIC_MATCH(...) LIMIT n` as a top-n search instead of silently
/// capping at the default page size. A non-literal limit (`LIMIT ?`,
/// `LIMIT a+b`) returns `None`, in which case the default applies.
pub fn query_limit(q: &Query) -> Option<usize> {
    q.limit.as_ref().and_then(expr_to_usize)
}

/// True when `q` has an analytical feature *beyond* a bare `LIMIT` / `OFFSET`
/// — a JOIN, aggregate, GROUP BY, DISTINCT, HAVING, window function,
/// ORDER BY, subquery, or set operation. A query that is analytical *only*
/// because it carries `LIMIT` / `OFFSET` returns `false`.
///
/// The semantic vector path uses this to keep `SEMANTIC_MATCH(...) LIMIT n`
/// on the similarity-ranked native path (applying the limit as a top-k
/// bound) rather than routing it to DataFusion, which would not preserve
/// similarity order without an explicit `ORDER BY`.
pub fn is_analytical_beyond_pagination(q: &Query) -> bool {
    has_order_by(q) || !body_is_native_single_table(q)
}

/// Is the query body a single-table SELECT with no join, subquery, DISTINCT,
/// HAVING, GROUP BY, aggregate, or window function? (Ignores ORDER BY / LIMIT
/// / OFFSET, which are handled separately.)
fn body_is_native_single_table(q: &Query) -> bool {
    match q.body.as_ref() {
        SetExpr::Select(select) => {
            // Multiple tables / any JOIN.
            if select.from.len() != 1 {
                return false;
            }
            if let Some(twj) = select.from.first() {
                if !twj.joins.is_empty() {
                    return false;
                }
                // Subquery / derived table / table-valued function in FROM.
                if !matches!(twj.relation, TableFactor::Table { .. }) {
                    return false;
                }
            } else {
                return false;
            }
            if select.distinct.is_some() || select.having.is_some() {
                return false;
            }
            match &select.group_by {
                GroupByExpr::Expressions(exprs, _) if !exprs.is_empty() => return false,
                GroupByExpr::All(_) => return false,
                _ => {}
            }
            // Aggregate or window function anywhere in the projection.
            if select.projection.iter().any(projection_is_analytical) {
                return false;
            }
            true
        }
        // A subquery body, set operation (UNION/…), VALUES, etc.
        _ => false,
    }
}

/// A single `ORDER BY` sort key over a bare column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortKey {
    pub column: String,
    pub descending: bool,
    /// Whether NULLs sort before non-NULLs. PostgreSQL default is NULLS LAST
    /// for ASC and NULLS FIRST for DESC; an explicit `NULLS FIRST/LAST`
    /// overrides. Null placement is absolute (not flipped by direction).
    pub nulls_first: bool,
}

/// Sort + pagination extracted from a query whose *only* analytical feature
/// is `ORDER BY` / `LIMIT` / `OFFSET` over a single table.
#[derive(Debug, Clone, PartialEq)]
pub struct SortLimit {
    pub order_by: Vec<SortKey>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// If `q` is a single-table SELECT whose only reason for being analytical is
/// `ORDER BY` (over bare columns) / `LIMIT` / `OFFSET` (with literal counts),
/// return the sort/pagination spec so a caller — e.g. the in-transaction read
/// path, where the DataFusion analytical engine cannot see the uncommitted
/// write buffer — can run the native scan and apply the ordering in memory.
///
/// Returns `None` when the query is already [`StatementClass::Native`] (no
/// sort/limit to apply), when it is analytical for some other reason (join,
/// aggregate, …), or when the `ORDER BY`/`LIMIT`/`OFFSET` uses a form we do
/// not evaluate natively (a computed sort expression, a non-literal limit).
/// A `None` result must never be treated as "no ordering" — the caller falls
/// back to rejecting the query rather than returning an unordered result.
pub fn simple_sort_limit(q: &Query) -> Option<SortLimit> {
    let has_sort_or_page = has_order_by(q) || q.limit.is_some() || q.offset.is_some();
    if !has_sort_or_page {
        return None;
    }
    if !body_is_native_single_table(q) {
        return None;
    }

    let mut order_by = Vec::new();
    if let Some(ob) = &q.order_by {
        for e in &ob.exprs {
            let column = match &e.expr {
                Expr::Identifier(id) => id.value.clone(),
                Expr::CompoundIdentifier(parts) => parts.last()?.value.clone(),
                // A computed ORDER BY expression (e.g. `ORDER BY a + b`) is
                // not evaluated natively — bail so the caller rejects it
                // rather than silently mis-sorting.
                _ => return None,
            };
            let descending = e.asc == Some(false);
            let nulls_first = e.nulls_first.unwrap_or(descending);
            order_by.push(SortKey {
                column,
                descending,
                nulls_first,
            });
        }
    }

    let limit = match &q.limit {
        Some(e) => Some(expr_to_usize(e)?),
        None => None,
    };
    let offset = match &q.offset {
        Some(o) => Some(expr_to_usize(&o.value)?),
        None => None,
    };

    Some(SortLimit {
        order_by,
        limit,
        offset,
    })
}

/// Parse a non-negative integer literal used in `LIMIT`/`OFFSET`.
fn expr_to_usize(e: &Expr) -> Option<usize> {
    match e {
        Expr::Value(sqlparser::ast::Value::Number(n, _)) => n.parse::<usize>().ok(),
        _ => None,
    }
}

/// sqlparser 0.50 models `ORDER BY` as `Option<OrderBy>`; treat a present,
/// non-empty clause as analytical.
fn has_order_by(q: &Query) -> bool {
    match &q.order_by {
        Some(ob) => !ob.exprs.is_empty(),
        None => false,
    }
}

fn projection_is_analytical(item: &SelectItem) -> bool {
    let expr = match item {
        SelectItem::UnnamedExpr(e) => e,
        SelectItem::ExprWithAlias { expr, .. } => expr,
        _ => return false,
    };
    expr_has_aggregate_or_window(expr)
}

/// Does `e` contain an aggregate or window function call?
fn expr_has_aggregate_or_window(e: &Expr) -> bool {
    match e {
        Expr::Function(f) => {
            if f.over.is_some() {
                return true;
            }
            let name = f.name.to_string().to_ascii_uppercase();
            matches!(
                name.as_str(),
                "COUNT" | "SUM" | "AVG" | "MIN" | "MAX" | "STDDEV" | "STDDEV_POP"
                    | "STDDEV_SAMP" | "VARIANCE" | "VAR_POP" | "VAR_SAMP" | "ARRAY_AGG"
                    | "STRING_AGG"
            )
        }
        Expr::BinaryOp { left, right, .. } => {
            expr_has_aggregate_or_window(left) || expr_has_aggregate_or_window(right)
        }
        Expr::UnaryOp { expr, .. } | Expr::Nested(expr) | Expr::Cast { expr, .. } => {
            expr_has_aggregate_or_window(expr)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlparser::dialect::PostgreSqlDialect;
    use sqlparser::parser::Parser;

    fn parse_query(sql: &str) -> Query {
        let mut stmts = Parser::parse_sql(&PostgreSqlDialect {}, sql).unwrap();
        match stmts.remove(0) {
            sqlparser::ast::Statement::Query(q) => *q,
            other => panic!("not a query: {other:?}"),
        }
    }

    #[test]
    fn plain_single_table_is_native_and_has_no_sort_limit() {
        let q = parse_query("SELECT id, name FROM users WHERE id = 1");
        assert_eq!(classify_query(&q), StatementClass::Native);
        assert_eq!(simple_sort_limit(&q), None);
    }

    #[test]
    fn single_table_order_by_is_extractable() {
        let q = parse_query("SELECT id FROM users ORDER BY id DESC LIMIT 5 OFFSET 2");
        assert_eq!(classify_query(&q), StatementClass::Analytical);
        let sl = simple_sort_limit(&q).expect("should be a simple sort/limit");
        assert_eq!(sl.order_by.len(), 1);
        assert_eq!(sl.order_by[0].column, "id");
        assert!(sl.order_by[0].descending);
        // DESC defaults to NULLS FIRST.
        assert!(sl.order_by[0].nulls_first);
        assert_eq!(sl.limit, Some(5));
        assert_eq!(sl.offset, Some(2));
    }

    #[test]
    fn asc_defaults_to_nulls_last() {
        let q = parse_query("SELECT id FROM users ORDER BY name ASC");
        let sl = simple_sort_limit(&q).unwrap();
        assert!(!sl.order_by[0].descending);
        assert!(!sl.order_by[0].nulls_first);
    }

    #[test]
    fn explicit_nulls_first_overrides_direction() {
        let q = parse_query("SELECT id FROM users ORDER BY name ASC NULLS FIRST");
        let sl = simple_sort_limit(&q).unwrap();
        assert!(sl.order_by[0].nulls_first);
    }

    #[test]
    fn multi_key_order_by() {
        let q = parse_query("SELECT * FROM t ORDER BY a ASC, b DESC");
        let sl = simple_sort_limit(&q).unwrap();
        assert_eq!(sl.order_by.len(), 2);
        assert_eq!(sl.order_by[0].column, "a");
        assert!(!sl.order_by[0].descending);
        assert_eq!(sl.order_by[1].column, "b");
        assert!(sl.order_by[1].descending);
    }

    #[test]
    fn limit_only_no_order_by() {
        let q = parse_query("SELECT * FROM t LIMIT 10");
        let sl = simple_sort_limit(&q).unwrap();
        assert!(sl.order_by.is_empty());
        assert_eq!(sl.limit, Some(10));
        assert_eq!(sl.offset, None);
    }

    #[test]
    fn join_with_order_by_is_not_simple() {
        let q = parse_query("SELECT * FROM a JOIN b ON a.id = b.id ORDER BY a.id");
        assert_eq!(classify_query(&q), StatementClass::Analytical);
        // Genuinely analytical (join) — cannot be handled by the native
        // sorted-scan path.
        assert_eq!(simple_sort_limit(&q), None);
    }

    #[test]
    fn aggregate_with_order_by_is_not_simple() {
        let q = parse_query("SELECT COUNT(*) FROM t ORDER BY 1");
        assert_eq!(simple_sort_limit(&q), None);
    }

    #[test]
    fn computed_order_by_expression_bails() {
        // ORDER BY over a computed expression is not evaluated natively.
        let q = parse_query("SELECT * FROM t ORDER BY a + b");
        assert_eq!(simple_sort_limit(&q), None);
    }

    #[test]
    fn compound_identifier_order_key_uses_last_segment() {
        let q = parse_query("SELECT * FROM t ORDER BY t.created_at DESC");
        let sl = simple_sort_limit(&q).unwrap();
        assert_eq!(sl.order_by[0].column, "created_at");
    }
}
