//! GalaxDB Embedded — Rust API for embedded mode.
//!
//! Provides `Database` struct that can be used directly from Rust or
//! exposed to Python via PyO3 (in the galaxdb-python package).
//!
//! ```ignore
//! let db = Database::open("/path/to/data")?;
//! db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")?;
//! db.execute("INSERT INTO users (id, name) VALUES (1, 'alice')")?;
//! let rows = db.execute("SELECT * FROM users")?;
//! ```

use galaxdb_common::{GalaxError, GalaxResult};
use galaxdb_sql::executor::{Catalog, ExecuteResult};
use galaxdb_sql::parser;
use galaxdb_sql::planner;
use galaxdb_sql::ast::AuroraStatement;

use std::path::{Path, PathBuf};

/// An embedded GalaxDB database instance.
///
/// This is the primary API for embedded mode. It manages the catalog,
/// parses SQL, plans queries, and executes them.
pub struct Database {
    path: PathBuf,
    catalog: Catalog,
}

/// Result row from a query.
#[derive(Debug, Clone)]
pub struct QueryRow {
    pub values: Vec<(String, String)>,
}

/// Result of executing a SQL statement.
#[derive(Debug, Clone)]
pub enum QueryResult {
    /// Rows returned (SELECT, SHOW).
    Rows(Vec<QueryRow>),
    /// Number of rows affected (INSERT, UPDATE, DELETE).
    RowCount(u64),
    /// DDL or command completed.
    Ok(String),
}

impl Database {
    /// Open or create a database at the given path.
    pub fn open(path: &str) -> GalaxResult<Self> {
        let path = PathBuf::from(path);
        std::fs::create_dir_all(&path)?;

        Ok(Self {
            path,
            catalog: Catalog::new(),
        })
    }

    /// Execute a SQL statement and return the result.
    pub fn execute(&mut self, sql: &str) -> GalaxResult<QueryResult> {
        let statements = parser::parse(sql)?;

        let mut last_result = QueryResult::Ok("OK".to_string());

        for stmt in &statements {
            let plan = self.plan_statement(stmt)?;
            let exec_result = galaxdb_sql::executor::execute(&plan, &mut self.catalog);

            last_result = match exec_result {
                ExecuteResult::Rows { columns: _, rows } => {
                    let query_rows: Vec<QueryRow> = rows
                        .iter()
                        .map(|row| QueryRow {
                            values: row
                                .columns
                                .iter()
                                .map(|(k, v)| (k.clone(), format_value(v)))
                                .collect(),
                        })
                        .collect();
                    QueryResult::Rows(query_rows)
                }
                ExecuteResult::RowCount(n) => QueryResult::RowCount(n),
                ExecuteResult::Ok(msg) => QueryResult::Ok(msg),
                ExecuteResult::Error(msg) => {
                    return Err(GalaxError::Internal(msg));
                }
            };
        }

        Ok(last_result)
    }

    /// Get the database path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get the number of tables in the catalog.
    pub fn table_count(&self) -> usize {
        self.catalog.table_count()
    }

    /// Check if a table exists.
    pub fn table_exists(&self, name: &str) -> bool {
        self.catalog.table_exists(name)
    }

    /// Plan a single statement.
    fn plan_statement(&self, stmt: &AuroraStatement) -> GalaxResult<planner::QueryPlan> {
        match stmt {
            AuroraStatement::Standard(boxed_stmt) => {
                match boxed_stmt.as_ref() {
                    sqlparser::ast::Statement::CreateTable(ct) => {
                        let columns = ct.columns.iter().map(|c| {
                            galaxdb_sql::ast::ColumnDef {
                                name: c.name.to_string(),
                                data_type: format!("{}", c.data_type),
                                nullable: true,
                                primary_key: false,
                                embedding: None,
                            }
                        }).collect();
                        Ok(planner::plan_create_table(galaxdb_sql::ast::CreateTableStmt {
                            table_name: ct.name.to_string(),
                            columns,
                            if_not_exists: ct.if_not_exists,
                        }))
                    }
                    sqlparser::ast::Statement::Drop { names, if_exists, .. } => {
                        let name = names.first()
                            .map(|n: &sqlparser::ast::ObjectName| n.to_string())
                            .unwrap_or_default();
                        Ok(planner::plan_drop_table(name, *if_exists))
                    }
                    sqlparser::ast::Statement::Insert(ins) => {
                        let table = ins.table_name.to_string();
                        Ok(planner::plan_insert(table, vec![], vec![]))
                    }
                    sqlparser::ast::Statement::Query(q) => {
                        // Try to extract table name from FROM clause
                        let table = extract_table_from_query(q);
                        Ok(planner::plan_select(table, vec![], None))
                    }
                    sqlparser::ast::Statement::Update { table, .. } => {
                        let table_name = table.relation.to_string();
                        Ok(planner::plan_update(table_name, vec![], None))
                    }
                    sqlparser::ast::Statement::Delete(del) => {
                        let table_name = format!("{:?}", del.from);
                        // Extract just the table name from the debug output
                        let table_name = table_name
                            .split('"')
                            .nth(1)
                            .unwrap_or("unknown")
                            .to_string();
                        Ok(planner::plan_delete(table_name, None))
                    }
                    _ => Ok(planner::QueryPlan::Analyze { table: "noop".to_string() }),
                }
            }
            AuroraStatement::CreateTable(ct) => Ok(planner::plan_create_table(ct.clone())),
            AuroraStatement::Analyze { table } => {
                Ok(planner::QueryPlan::Analyze { table: table.clone() })
            }
            AuroraStatement::BackupTo { path } => {
                Ok(planner::QueryPlan::Backup { path: path.clone() })
            }
            AuroraStatement::RestoreFrom { path } => {
                Ok(planner::QueryPlan::Restore { path: path.clone() })
            }
            AuroraStatement::ShowEmbeddingHealth { table } => {
                Ok(planner::QueryPlan::ShowEmbeddingHealth { table: table.clone() })
            }
            AuroraStatement::CreateVersionTag(tag) => {
                Ok(planner::QueryPlan::CreateVersionTag(tag.clone()))
            }
            _ => Ok(planner::QueryPlan::Analyze { table: "noop".to_string() }),
        }
    }
}

fn format_value(v: &galaxdb_sql::planner::Value) -> String {
    match v {
        galaxdb_sql::planner::Value::Integer(i) => i.to_string(),
        galaxdb_sql::planner::Value::Float(f) => f.to_string(),
        galaxdb_sql::planner::Value::Text(s) => s.clone(),
        galaxdb_sql::planner::Value::Bool(b) => b.to_string(),
        galaxdb_sql::planner::Value::Null => "NULL".to_string(),
        galaxdb_sql::planner::Value::Blob(b) => format!("\\x{}", hex_encode(b)),
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Extract table name from a SELECT query's FROM clause.
fn extract_table_from_query(query: &sqlparser::ast::Query) -> String {
    if let sqlparser::ast::SetExpr::Select(select) = query.body.as_ref() {
        if let Some(from) = select.from.first() {
            return from.relation.to_string();
        }
    }
    "unknown".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_creates_directory() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_db");
        let db = Database::open(db_path.to_str().unwrap()).unwrap();
        assert!(db_path.exists());
        assert_eq!(db.table_count(), 0);
    }

    #[test]
    fn create_table_and_check_exists() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = Database::open(dir.path().join("db").to_str().unwrap()).unwrap();

        let result = db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)").unwrap();
        assert!(matches!(result, QueryResult::Ok(_)));
        assert!(db.table_exists("users"));
        assert_eq!(db.table_count(), 1);
    }

    #[test]
    fn create_and_drop_table() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = Database::open(dir.path().join("db").to_str().unwrap()).unwrap();

        db.execute("CREATE TABLE test (id INT)").unwrap();
        assert!(db.table_exists("test"));

        db.execute("DROP TABLE test").unwrap();
        assert!(!db.table_exists("test"));
    }

    #[test]
    fn create_duplicate_table_fails() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = Database::open(dir.path().join("db").to_str().unwrap()).unwrap();

        db.execute("CREATE TABLE t (id INT)").unwrap();
        let result = db.execute("CREATE TABLE t (id INT)");
        assert!(result.is_err());
    }

    #[test]
    fn drop_nonexistent_table_fails() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = Database::open(dir.path().join("db").to_str().unwrap()).unwrap();

        let result = db.execute("DROP TABLE nope");
        assert!(result.is_err());
    }

    #[test]
    fn execute_analyze() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = Database::open(dir.path().join("db").to_str().unwrap()).unwrap();

        db.execute("CREATE TABLE t (id INT)").unwrap();
        let result = db.execute("ANALYZE t").unwrap();
        assert!(matches!(result, QueryResult::Ok(_)));
    }

    #[test]
    fn execute_show_embedding_health() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = Database::open(dir.path().join("db").to_str().unwrap()).unwrap();

        let result = db.execute("SHOW EMBEDDING HEALTH").unwrap();
        assert!(matches!(result, QueryResult::Rows(_)));
    }

    #[test]
    fn execute_create_version_tag() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = Database::open(dir.path().join("db").to_str().unwrap()).unwrap();

        let result = db.execute("CREATE VERSION TAG 'v1.0'").unwrap();
        assert!(matches!(result, QueryResult::Ok(_)));
    }

    #[test]
    fn execute_backup_restore() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = Database::open(dir.path().join("db").to_str().unwrap()).unwrap();

        let result = db.execute("BACKUP TO '/tmp/backup'").unwrap();
        assert!(matches!(result, QueryResult::Ok(_)));

        let result = db.execute("RESTORE FROM '/tmp/backup'").unwrap();
        assert!(matches!(result, QueryResult::Ok(_)));
    }

    #[test]
    fn invalid_sql_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = Database::open(dir.path().join("db").to_str().unwrap()).unwrap();

        let result = db.execute("SELECTT * FROM nowhere");
        assert!(result.is_err());
    }

    #[test]
    fn empty_sql_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = Database::open(dir.path().join("db").to_str().unwrap()).unwrap();

        let result = db.execute("");
        assert!(result.is_err());
    }

    #[test]
    fn database_path_accessor() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("mydb");
        let db = Database::open(db_path.to_str().unwrap()).unwrap();
        assert_eq!(db.path(), db_path);
    }
}
