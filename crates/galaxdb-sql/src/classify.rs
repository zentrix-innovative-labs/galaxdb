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

    match q.body.as_ref() {
        SetExpr::Select(select) => {
            // Multiple tables / any JOIN.
            if select.from.len() != 1 {
                return StatementClass::Analytical;
            }
            if let Some(twj) = select.from.first() {
                if !twj.joins.is_empty() {
                    return StatementClass::Analytical;
                }
                // Subquery / derived table / table-valued function in FROM.
                if !matches!(twj.relation, TableFactor::Table { .. }) {
                    return StatementClass::Analytical;
                }
            }
            if select.distinct.is_some() || select.having.is_some() {
                return StatementClass::Analytical;
            }
            match &select.group_by {
                GroupByExpr::Expressions(exprs, _) if !exprs.is_empty() => {
                    return StatementClass::Analytical
                }
                GroupByExpr::All(_) => return StatementClass::Analytical,
                _ => {}
            }
            // Aggregate or window function anywhere in the projection.
            if select.projection.iter().any(projection_is_analytical) {
                return StatementClass::Analytical;
            }
            StatementClass::Native
        }
        // A subquery body, set operation (UNION/…), VALUES, etc.
        _ => StatementClass::Analytical,
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
