//! pg_catalog stubs for psycopg2 and SQLAlchemy compatibility.
//!
//! Minimal system tables to satisfy client library introspection:
//! - pg_catalog.pg_class (table listing)
//! - pg_catalog.pg_attribute (column metadata)
//! - pg_catalog.pg_type (type system)
//! - pg_catalog.pg_namespace (schema listing)
//! - pg_catalog.pg_database (database listing)
//!
//! Queries against unsupported pg_catalog tables return empty result sets.

use crate::messages::ColumnDesc;

/// A row in a pg_catalog result set (all text values).
pub type PgCatalogRow = Vec<Option<String>>;

/// Result of a pg_catalog query.
#[derive(Debug, Clone)]
pub struct PgCatalogResult {
    pub columns: Vec<ColumnDesc>,
    pub rows: Vec<PgCatalogRow>,
}

/// Check if a query targets pg_catalog and handle it.
/// Returns Some(result) if handled, None if not a pg_catalog query.
pub fn try_handle_pg_catalog(sql: &str) -> Option<PgCatalogResult> {
    let upper = sql.to_uppercase();

    if !upper.contains("PG_CATALOG") && !upper.contains("PG_CLASS")
        && !upper.contains("PG_ATTRIBUTE") && !upper.contains("PG_TYPE")
        && !upper.contains("PG_NAMESPACE") && !upper.contains("PG_DATABASE")
    {
        return None;
    }

    // Build the full stub table for the targeted catalog relation, then
    // refine it with the query's WHERE / projection / COUNT(*) so a query
    // like `SELECT oid FROM pg_type WHERE typname = 'int4'` returns exactly
    // the matching row/column instead of the whole table.
    let full = if upper.contains("PG_CLASS") {
        handle_pg_class()
    } else if upper.contains("PG_ATTRIBUTE") {
        handle_pg_attribute()
    } else if upper.contains("PG_TYPE") {
        handle_pg_type()
    } else if upper.contains("PG_NAMESPACE") {
        handle_pg_namespace()
    } else if upper.contains("PG_DATABASE") {
        handle_pg_database()
    } else {
        // Unsupported pg_catalog table — return empty result set
        return Some(PgCatalogResult {
            columns: vec![],
            rows: vec![],
        });
    };

    Some(refine_with_query(sql, full))
}

/// Apply the SELECT's WHERE, projection, and `COUNT(*)` to a fully-populated
/// stub table. Uses a real SQL parse; if the query shape is not one we can
/// faithfully evaluate (joins, casts, functions other than COUNT(*), or an
/// unsupported WHERE), we return the full table unchanged — a safe superset
/// for driver introspection, never a wrong-but-plausible subset.
fn refine_with_query(sql: &str, full: PgCatalogResult) -> PgCatalogResult {
    use sqlparser::ast::{SetExpr, Statement};
    use sqlparser::dialect::PostgreSqlDialect;
    use sqlparser::parser::Parser;

    let Ok(stmts) = Parser::parse_sql(&PostgreSqlDialect {}, sql) else {
        return full;
    };
    let Some(Statement::Query(q)) = stmts.into_iter().next() else {
        return full;
    };
    let SetExpr::Select(select) = q.body.as_ref() else {
        return full;
    };

    // 1. WHERE filtering (equality / AND over the stub columns).
    let mut rows = full.rows.clone();
    if let Some(pred) = &select.selection {
        match apply_where(pred, &full.columns, &rows) {
            Some(filtered) => rows = filtered,
            None => return full, // unsupported predicate → safe superset
        }
    }

    // 2. Projection: COUNT(*), explicit columns, or wildcard.
    apply_projection(&select.projection, &full.columns, rows).unwrap_or(full)
}

/// Column index (case-insensitive) for a name.
fn column_index(columns: &[ColumnDesc], name: &str) -> Option<usize> {
    columns
        .iter()
        .position(|c| c.name.eq_ignore_ascii_case(name))
}

/// Evaluate a supported WHERE predicate against the stub rows. Supports
/// `col = literal`, `col <> literal`, and `AND` of those. Returns `None` for
/// any other shape so the caller can safely return the full table.
fn apply_where(
    pred: &sqlparser::ast::Expr,
    columns: &[ColumnDesc],
    rows: &[PgCatalogRow],
) -> Option<Vec<PgCatalogRow>> {
    use sqlparser::ast::{BinaryOperator, Expr};

    match pred {
        Expr::BinaryOp { left, op, right } => match op {
            BinaryOperator::And => {
                let first = apply_where(left, columns, rows)?;
                apply_where(right, columns, &first)
            }
            BinaryOperator::Eq | BinaryOperator::NotEq => {
                let (col, lit) = binop_col_and_literal(left, right)?;
                let idx = column_index(columns, &col)?;
                let want_eq = matches!(op, BinaryOperator::Eq);
                Some(
                    rows.iter()
                        .filter(|row| {
                            let cell = row.get(idx).and_then(|v| v.as_deref()).unwrap_or("");
                            (cell == lit) == want_eq
                        })
                        .cloned()
                        .collect(),
                )
            }
            _ => None,
        },
        // Parenthesised predicate.
        Expr::Nested(inner) => apply_where(inner, columns, rows),
        _ => None,
    }
}

/// Extract `(column_name, literal_text)` from the two sides of a binary
/// comparison, in either order (`col = 'x'` or `'x' = col`).
fn binop_col_and_literal(
    left: &sqlparser::ast::Expr,
    right: &sqlparser::ast::Expr,
) -> Option<(String, String)> {
    let col = |e: &sqlparser::ast::Expr| match e {
        sqlparser::ast::Expr::Identifier(id) => Some(id.value.clone()),
        sqlparser::ast::Expr::CompoundIdentifier(parts) => {
            parts.last().map(|p| p.value.clone())
        }
        _ => None,
    };
    let lit = |e: &sqlparser::ast::Expr| match e {
        sqlparser::ast::Expr::Value(sqlparser::ast::Value::SingleQuotedString(s))
        | sqlparser::ast::Expr::Value(sqlparser::ast::Value::DoubleQuotedString(s)) => {
            Some(s.clone())
        }
        sqlparser::ast::Expr::Value(sqlparser::ast::Value::Number(n, _)) => Some(n.clone()),
        _ => None,
    };
    if let (Some(c), Some(l)) = (col(left), lit(right)) {
        return Some((c, l));
    }
    if let (Some(c), Some(l)) = (col(right), lit(left)) {
        return Some((c, l));
    }
    None
}

/// Apply the projection list. Returns `None` (caller falls back to the full
/// table) if any item is neither `*`, a bare column, nor `COUNT(*)`.
fn apply_projection(
    projection: &[sqlparser::ast::SelectItem],
    columns: &[ColumnDesc],
    rows: Vec<PgCatalogRow>,
) -> Option<PgCatalogResult> {
    use sqlparser::ast::{Expr, FunctionArguments, SelectItem};

    // COUNT(*) — a single aggregate item.
    if projection.len() == 1 {
        if let SelectItem::UnnamedExpr(Expr::Function(f))
        | SelectItem::ExprWithAlias {
            expr: Expr::Function(f),
            ..
        } = &projection[0]
        {
            if f.name.to_string().eq_ignore_ascii_case("count") {
                // COUNT(*) or COUNT(<one arg>) over the stub → row count.
                let is_countable = match &f.args {
                    FunctionArguments::List(list) => list.args.len() == 1,
                    _ => false,
                };
                if is_countable {
                    return Some(PgCatalogResult {
                        columns: vec![ColumnDesc::int4("count")],
                        rows: vec![vec![Some(rows.len().to_string())]],
                    });
                }
            }
        }
    }

    // Wildcard anywhere → all columns.
    if projection
        .iter()
        .any(|item| matches!(item, SelectItem::Wildcard(_) | SelectItem::QualifiedWildcard(..)))
    {
        return Some(PgCatalogResult {
            columns: columns.to_vec(),
            rows,
        });
    }

    // Explicit column list: every item must resolve to a known column.
    let mut indices = Vec::with_capacity(projection.len());
    let mut out_cols = Vec::with_capacity(projection.len());
    for item in projection {
        let name = match item {
            SelectItem::UnnamedExpr(Expr::Identifier(id)) => id.value.clone(),
            SelectItem::UnnamedExpr(Expr::CompoundIdentifier(parts)) => {
                parts.last()?.value.clone()
            }
            SelectItem::ExprWithAlias {
                expr: Expr::Identifier(id),
                ..
            } => id.value.clone(),
            _ => return None,
        };
        let idx = column_index(columns, &name)?;
        indices.push(idx);
        out_cols.push(columns[idx].clone());
    }

    let out_rows = rows
        .into_iter()
        .map(|row| indices.iter().map(|&i| row.get(i).cloned().flatten()).collect())
        .collect();

    Some(PgCatalogResult {
        columns: out_cols,
        rows: out_rows,
    })
}

fn handle_pg_class() -> PgCatalogResult {
    PgCatalogResult {
        columns: vec![
            ColumnDesc::int4("oid"),
            ColumnDesc::text("relname"),
            ColumnDesc::int4("relnamespace"),
            ColumnDesc::text("relkind"),
        ],
        rows: vec![],
    }
}

fn handle_pg_attribute() -> PgCatalogResult {
    PgCatalogResult {
        columns: vec![
            ColumnDesc::int4("attrelid"),
            ColumnDesc::text("attname"),
            ColumnDesc::int4("atttypid"),
            ColumnDesc::int4("attnum"),
            ColumnDesc::text("attnotnull"),
        ],
        rows: vec![],
    }
}

fn handle_pg_type() -> PgCatalogResult {
    // The scalar OIDs GalaxDB reports on the wire (task 22), matching
    // `galaxdb_sql::types::oid`. Drivers (psycopg/SQLAlchemy/tokio-postgres)
    // look types up here by oid or typname during connection setup.
    PgCatalogResult {
        columns: vec![
            ColumnDesc::int4("oid"),
            ColumnDesc::text("typname"),
            ColumnDesc::int4("typlen"),
            ColumnDesc::text("typtype"),
        ],
        rows: vec![
            vec![s("16"), s("bool"), s("1"), s("b")],
            vec![s("17"), s("bytea"), s("-1"), s("b")],
            vec![s("20"), s("int8"), s("8"), s("b")],
            vec![s("21"), s("int2"), s("2"), s("b")],
            vec![s("23"), s("int4"), s("4"), s("b")],
            vec![s("25"), s("text"), s("-1"), s("b")],
            vec![s("114"), s("json"), s("-1"), s("b")],
            vec![s("700"), s("float4"), s("4"), s("b")],
            vec![s("701"), s("float8"), s("8"), s("b")],
            vec![s("1043"), s("varchar"), s("-1"), s("b")],
            vec![s("1082"), s("date"), s("4"), s("b")],
            vec![s("1114"), s("timestamp"), s("8"), s("b")],
            vec![s("1184"), s("timestamptz"), s("8"), s("b")],
            vec![s("1700"), s("numeric"), s("-1"), s("b")],
            vec![s("2950"), s("uuid"), s("16"), s("b")],
            vec![s("3802"), s("jsonb"), s("-1"), s("b")],
        ],
    }
}

fn handle_pg_namespace() -> PgCatalogResult {
    PgCatalogResult {
        columns: vec![
            ColumnDesc::int4("oid"),
            ColumnDesc::text("nspname"),
        ],
        rows: vec![
            vec![s("11"), s("pg_catalog")],
            vec![s("2200"), s("public")],
        ],
    }
}

fn handle_pg_database() -> PgCatalogResult {
    PgCatalogResult {
        columns: vec![
            ColumnDesc::int4("oid"),
            ColumnDesc::text("datname"),
        ],
        rows: vec![
            vec![s("1"), s("galaxdb")],
        ],
    }
}

/// Helper to create Some(String).
fn s(val: &str) -> Option<String> {
    Some(val.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pg_class_query_returns_result() {
        let result = try_handle_pg_catalog("SELECT * FROM pg_catalog.pg_class").unwrap();
        assert_eq!(result.columns.len(), 4);
        assert_eq!(result.columns[0].name, "oid");
        assert_eq!(result.columns[1].name, "relname");
    }

    #[test]
    fn pg_attribute_query_returns_result() {
        let result = try_handle_pg_catalog("SELECT * FROM pg_catalog.pg_attribute").unwrap();
        assert_eq!(result.columns.len(), 5);
        assert_eq!(result.columns[1].name, "attname");
    }

    #[test]
    fn pg_type_returns_basic_types() {
        let result = try_handle_pg_catalog("SELECT * FROM pg_catalog.pg_type").unwrap();
        assert_eq!(result.columns.len(), 4);
        assert!(!result.rows.is_empty());
        // Should have int4, text, bool at minimum
        let type_names: Vec<&str> = result.rows.iter()
            .filter_map(|r| r[1].as_deref())
            .collect();
        assert!(type_names.contains(&"int4"));
        assert!(type_names.contains(&"text"));
        assert!(type_names.contains(&"bool"));
    }

    #[test]
    fn pg_namespace_returns_public_schema() {
        let result = try_handle_pg_catalog("SELECT * FROM pg_catalog.pg_namespace").unwrap();
        assert_eq!(result.rows.len(), 2);
        let names: Vec<&str> = result.rows.iter()
            .filter_map(|r| r[1].as_deref())
            .collect();
        assert!(names.contains(&"public"));
        assert!(names.contains(&"pg_catalog"));
    }

    #[test]
    fn pg_database_returns_galaxdb() {
        let result = try_handle_pg_catalog("SELECT * FROM pg_catalog.pg_database").unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][1].as_deref(), Some("galaxdb"));
    }

    #[test]
    fn unsupported_pg_catalog_returns_empty() {
        let result = try_handle_pg_catalog("SELECT * FROM pg_catalog.pg_settings").unwrap();
        assert!(result.columns.is_empty());
        assert!(result.rows.is_empty());
    }

    #[test]
    fn pg_type_where_filters_to_one_row() {
        let r = try_handle_pg_catalog("SELECT oid, typname FROM pg_catalog.pg_type WHERE typname = 'int4'").unwrap();
        assert_eq!(r.rows.len(), 1, "WHERE must filter to the single matching type");
        // projection is oid, typname in that order
        assert_eq!(r.columns.len(), 2);
        assert_eq!(r.columns[0].name, "oid");
        assert_eq!(r.columns[1].name, "typname");
        assert_eq!(r.rows[0][0].as_deref(), Some("23"));
        assert_eq!(r.rows[0][1].as_deref(), Some("int4"));
    }

    #[test]
    fn pg_type_count_star_returns_scalar() {
        let full = try_handle_pg_catalog("SELECT * FROM pg_catalog.pg_type").unwrap();
        let cnt = try_handle_pg_catalog("SELECT COUNT(*) FROM pg_catalog.pg_type").unwrap();
        assert_eq!(cnt.columns.len(), 1);
        assert_eq!(cnt.columns[0].name, "count");
        assert_eq!(cnt.rows.len(), 1);
        assert_eq!(cnt.rows[0][0].as_deref(), Some(full.rows.len().to_string().as_str()));
    }

    #[test]
    fn pg_type_count_star_with_where_counts_filtered() {
        let cnt = try_handle_pg_catalog("SELECT COUNT(*) FROM pg_type WHERE typname = 'uuid'").unwrap();
        assert_eq!(cnt.rows[0][0].as_deref(), Some("1"));
    }

    #[test]
    fn pg_type_projection_single_column() {
        let r = try_handle_pg_catalog("SELECT typname FROM pg_type WHERE oid = 25").unwrap();
        assert_eq!(r.columns.len(), 1);
        assert_eq!(r.columns[0].name, "typname");
        assert_eq!(r.rows.len(), 1);
        assert_eq!(r.rows[0][0].as_deref(), Some("text"));
    }

    #[test]
    fn pg_type_unmatched_where_returns_no_rows() {
        let r = try_handle_pg_catalog("SELECT oid FROM pg_type WHERE typname = 'nonesuch'").unwrap();
        assert!(r.rows.is_empty());
    }

    #[test]
    fn non_pg_catalog_returns_none() {
        let result = try_handle_pg_catalog("SELECT * FROM users");
        assert!(result.is_none());
    }

    #[test]
    fn case_insensitive_detection() {
        assert!(try_handle_pg_catalog("select * from PG_CATALOG.PG_TYPE").is_some());
        assert!(try_handle_pg_catalog("SELECT oid FROM pg_type WHERE typname = 'int4'").is_some());
    }
}
