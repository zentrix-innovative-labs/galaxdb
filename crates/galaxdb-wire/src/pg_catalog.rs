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

    // pg_class — table listing
    if upper.contains("PG_CLASS") {
        return Some(handle_pg_class());
    }

    // pg_attribute — column metadata
    if upper.contains("PG_ATTRIBUTE") {
        return Some(handle_pg_attribute());
    }

    // pg_type — type system
    if upper.contains("PG_TYPE") {
        return Some(handle_pg_type());
    }

    // pg_namespace — schema listing
    if upper.contains("PG_NAMESPACE") {
        return Some(handle_pg_namespace());
    }

    // pg_database — database listing
    if upper.contains("PG_DATABASE") {
        return Some(handle_pg_database());
    }

    // Unsupported pg_catalog table — return empty result set
    Some(PgCatalogResult {
        columns: vec![],
        rows: vec![],
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
