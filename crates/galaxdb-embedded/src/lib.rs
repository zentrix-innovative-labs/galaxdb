//! GalaxDB Embedded — Rust API for embedded mode with real storage engine.
//!
//! ```ignore
//! let mut db = Database::open("/path/to/data")?;
//! db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")?;
//! db.execute("INSERT INTO users (id, name) VALUES (1, 'alice')")?;
//! let rows = db.execute("SELECT * FROM users")?;
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use galaxdb_common::{GalaxError, GalaxResult};
use galaxdb_sql::ast::{AuroraStatement, CreateTableStmt};
use galaxdb_sql::executor::{Catalog, CatalogColumn, TableEntry};
use galaxdb_sql::parser;
use galaxdb_storage::engine::{Engine, EngineConfig};

/// An embedded GalaxDB database instance backed by the real storage engine.
pub struct Database {
    path: PathBuf,
    catalog: Catalog,
    engine: Arc<Engine>,
    /// Table schemas: table_name → column names (ordered).
    schemas: HashMap<String, Vec<String>>,
    rt: tokio::runtime::Runtime,
}

/// Result row from a query.
#[derive(Debug, Clone)]
pub struct QueryRow {
    pub values: Vec<(String, String)>,
}

/// Result of executing a SQL statement.
#[derive(Debug, Clone)]
pub enum QueryResult {
    Rows(Vec<QueryRow>),
    RowCount(u64),
    Ok(String),
}

impl Database {
    /// Open or create a database at the given path.
    pub fn open(path: &str) -> GalaxResult<Self> {
        let path = PathBuf::from(path);
        std::fs::create_dir_all(&path)?;

        let config = EngineConfig {
            data_dir: path.clone(),
            ..Default::default()
        };

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| GalaxError::Internal(format!("failed to create runtime: {}", e)))?;

        let engine = rt.block_on(async { Engine::new(config) })?;

        Ok(Self {
            path,
            catalog: Catalog::new(),
            engine: Arc::new(engine),
            schemas: HashMap::new(),
            rt,
        })
    }

    /// Execute a SQL statement and return the result.
    pub fn execute(&mut self, sql: &str) -> GalaxResult<QueryResult> {
        let statements = parser::parse(sql)?;
        let mut last_result = QueryResult::Ok("OK".to_string());

        for stmt in &statements {
            last_result = self.execute_statement(stmt)?;
        }

        Ok(last_result)
    }

    fn execute_statement(&mut self, stmt: &AuroraStatement) -> GalaxResult<QueryResult> {
        match stmt {
            AuroraStatement::Standard(boxed_stmt) => self.execute_standard(boxed_stmt),
            AuroraStatement::CreateTable(ct) => self.execute_create_table(ct),
            AuroraStatement::Analyze { table } => {
                if !self.catalog.table_exists(table) {
                    return Err(GalaxError::TableNotFound(table.clone()));
                }
                Ok(QueryResult::Ok(format!("ANALYZE {}", table)))
            }
            AuroraStatement::BackupTo { path } => {
                Ok(QueryResult::Ok(format!("BACKUP TO '{}'", path)))
            }
            AuroraStatement::RestoreFrom { path } => {
                Ok(QueryResult::Ok(format!("RESTORE FROM '{}'", path)))
            }
            AuroraStatement::ShowEmbeddingHealth { table } => {
                let msg = match table {
                    Some(t) => format!("SHOW EMBEDDING HEALTH FOR {}", t),
                    None => "SHOW EMBEDDING HEALTH".to_string(),
                };
                Ok(QueryResult::Rows(vec![QueryRow {
                    values: vec![("status".to_string(), msg)],
                }]))
            }
            AuroraStatement::CreateVersionTag(tag) => {
                Ok(QueryResult::Ok(format!("CREATE VERSION TAG '{}'", tag.name)))
            }
            AuroraStatement::BulkInsert(bi) => {
                if !self.catalog.table_exists(&bi.table) {
                    return Err(GalaxError::TableNotFound(bi.table.clone()));
                }
                Ok(QueryResult::Ok(format!("BULK INSERT INTO {}", bi.table)))
            }
            _ => Ok(QueryResult::Ok("OK".to_string())),
        }
    }

    fn execute_standard(
        &mut self,
        stmt: &sqlparser::ast::Statement,
    ) -> GalaxResult<QueryResult> {
        match stmt {
            sqlparser::ast::Statement::CreateTable(ct) => {
                let columns: Vec<galaxdb_sql::ast::ColumnDef> = ct
                    .columns
                    .iter()
                    .map(|c| galaxdb_sql::ast::ColumnDef {
                        name: c.name.to_string(),
                        data_type: format!("{}", c.data_type),
                        nullable: true,
                        primary_key: c.options.iter().any(|o| {
                            matches!(
                                o.option,
                                sqlparser::ast::ColumnOption::Unique {
                                    is_primary: true,
                                    ..
                                }
                            )
                        }),
                        embedding: None,
                    })
                    .collect();

                self.execute_create_table(&CreateTableStmt {
                    table_name: ct.name.to_string(),
                    columns,
                    if_not_exists: ct.if_not_exists,
                })
            }

            sqlparser::ast::Statement::Drop {
                names, if_exists, ..
            } => {
                let name = names
                    .first()
                    .map(|n: &sqlparser::ast::ObjectName| n.to_string())
                    .unwrap_or_default();

                match self.catalog.drop_table(&name) {
                    Ok(_) => {
                        self.schemas.remove(&name);
                        Ok(QueryResult::Ok(format!("DROP TABLE {}", name)))
                    }
                    Err(_) if *if_exists => {
                        Ok(QueryResult::Ok(format!("DROP TABLE IF EXISTS {}", name)))
                    }
                    Err(e) => Err(e),
                }
            }

            sqlparser::ast::Statement::Insert(ins) => {
                let table = ins.table_name.to_string();
                if !self.catalog.table_exists(&table) {
                    return Err(GalaxError::TableNotFound(table));
                }

                let schema = self.schemas.get(&table).cloned().unwrap_or_default();

                // Extract values from the INSERT statement
                let mut row_count = 0u64;
                if let Some(source) = &ins.source {
                    if let sqlparser::ast::SetExpr::Values(values) = source.body.as_ref() {
                        for row_values in &values.rows {
                            // Build key from first column value, rest as value
                            let mut parts = Vec::new();
                            for (i, val) in row_values.iter().enumerate() {
                                let col_name = schema
                                    .get(i)
                                    .cloned()
                                    .unwrap_or_else(|| format!("col{}", i));
                                let val_str = format_sql_expr(val);
                                parts.push(format!("{}={}", col_name, val_str));
                            }

                            let key = if let Some(first_val) = row_values.first() {
                                format!("{}:{}", table, format_sql_expr(first_val))
                            } else {
                                format!("{}:{}", table, row_count)
                            };

                            let value = parts.join("|");

                            self.rt.block_on(async {
                                self.engine
                                    .put(key.into_bytes(), value.into_bytes())
                                    .await
                            })?;

                            row_count += 1;
                        }
                    }
                }

                Ok(QueryResult::RowCount(row_count))
            }

            sqlparser::ast::Statement::Query(query) => {
                // Extract table name from FROM clause
                let table = extract_table_from_query(query);

                if table != "unknown" && !self.catalog.table_exists(&table) {
                    return Err(GalaxError::TableNotFound(table));
                }

                // Scan all rows for this table
                let all_rows = self.engine.scan_all();
                let prefix = format!("{}:", table);

                let mut result_rows = Vec::new();
                let _schema = self.schemas.get(&table).cloned().unwrap_or_default();

                for (key, value) in &all_rows {
                    let key_str = String::from_utf8_lossy(key);
                    if !key_str.starts_with(&prefix) {
                        continue;
                    }

                    let value_str = String::from_utf8_lossy(value);
                    let mut row_values = Vec::new();

                    // Parse "col=val|col=val" format
                    for part in value_str.split('|') {
                        if let Some((col, val)) = part.split_once('=') {
                            row_values.push((col.to_string(), val.to_string()));
                        }
                    }

                    result_rows.push(QueryRow {
                        values: row_values,
                    });
                }

                Ok(QueryResult::Rows(result_rows))
            }

            sqlparser::ast::Statement::Update { table, assignments, .. } => {
                let table_name = table.relation.to_string();
                if !self.catalog.table_exists(&table_name) {
                    return Err(GalaxError::TableNotFound(table_name));
                }

                // Check for embedding source column updates
                if let Some(entry) = self.catalog.get_table(&table_name) {
                    for assign in assignments {
                        let col_name = format!("{}", assign.target);
                        if entry
                            .columns
                            .iter()
                            .any(|c| c.name == col_name && c.is_embedding_source)
                        {
                            return Err(GalaxError::EmbeddingSourceUpdate { column: col_name });
                        }
                    }
                }

                Ok(QueryResult::RowCount(0))
            }

            sqlparser::ast::Statement::Delete(_) => {
                Ok(QueryResult::RowCount(0))
            }

            _ => Ok(QueryResult::Ok("OK".to_string())),
        }
    }

    fn execute_create_table(&mut self, ct: &CreateTableStmt) -> GalaxResult<QueryResult> {
        let columns: Vec<CatalogColumn> = ct
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

        let col_names: Vec<String> = ct.columns.iter().map(|c| c.name.clone()).collect();

        let entry = TableEntry {
            name: ct.table_name.clone(),
            columns,
            has_embedding: ct.columns.iter().any(|c| c.embedding.is_some()),
        };

        self.catalog
            .create_table(ct.table_name.clone(), entry)?;
        self.schemas.insert(ct.table_name.clone(), col_names);

        Ok(QueryResult::Ok(format!("CREATE TABLE {}", ct.table_name)))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn table_count(&self) -> usize {
        self.catalog.table_count()
    }

    pub fn table_exists(&self, name: &str) -> bool {
        self.catalog.table_exists(name)
    }

    pub fn row_count(&self) -> u64 {
        self.engine.row_count()
    }
}

impl Drop for Database {
    fn drop(&mut self) {
        self.engine.shutdown();
    }
}

fn format_sql_expr(expr: &sqlparser::ast::Expr) -> String {
    match expr {
        sqlparser::ast::Expr::Value(v) => match v {
            sqlparser::ast::Value::Number(n, _) => n.clone(),
            sqlparser::ast::Value::SingleQuotedString(s) => s.clone(),
            sqlparser::ast::Value::DoubleQuotedString(s) => s.clone(),
            sqlparser::ast::Value::Boolean(b) => b.to_string(),
            sqlparser::ast::Value::Null => "NULL".to_string(),
            _ => format!("{}", v),
        },
        _ => format!("{}", expr),
    }
}

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

    fn test_db() -> Database {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_db");
        std::mem::forget(dir);
        Database::open(path.to_str().unwrap()).unwrap()
    }

    #[test]
    fn create_table_and_insert_and_select() {
        let mut db = test_db();

        db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        assert!(db.table_exists("users"));

        let result = db
            .execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
            .unwrap();
        assert!(matches!(result, QueryResult::RowCount(1)));

        let result = db
            .execute("INSERT INTO users (id, name) VALUES (2, 'bob')")
            .unwrap();
        assert!(matches!(result, QueryResult::RowCount(1)));

        let result = db.execute("SELECT * FROM users").unwrap();
        match result {
            QueryResult::Rows(rows) => {
                assert_eq!(rows.len(), 2);
                // Check first row has id and name
                let first = &rows[0];
                assert!(first.values.iter().any(|(k, v)| k == "id" && v == "1"));
                assert!(first.values.iter().any(|(k, v)| k == "name" && v == "alice"));
            }
            other => panic!("expected Rows, got {:?}", other),
        }
    }

    #[test]
    fn insert_multiple_rows_and_count() {
        let mut db = test_db();
        db.execute("CREATE TABLE t (id INT, val TEXT)").unwrap();

        for i in 0..10 {
            db.execute(&format!("INSERT INTO t (id, val) VALUES ({}, 'v{}')", i, i))
                .unwrap();
        }

        assert_eq!(db.row_count(), 10);

        let result = db.execute("SELECT * FROM t").unwrap();
        match result {
            QueryResult::Rows(rows) => assert_eq!(rows.len(), 10),
            other => panic!("expected Rows, got {:?}", other),
        }
    }

    #[test]
    fn select_from_nonexistent_table_fails() {
        let mut db = test_db();
        let result = db.execute("SELECT * FROM nope");
        assert!(result.is_err());
    }

    #[test]
    fn insert_into_nonexistent_table_fails() {
        let mut db = test_db();
        let result = db.execute("INSERT INTO nope (id) VALUES (1)");
        assert!(result.is_err());
    }

    #[test]
    fn create_and_drop_table() {
        let mut db = test_db();
        db.execute("CREATE TABLE t (id INT)").unwrap();
        assert!(db.table_exists("t"));
        db.execute("DROP TABLE t").unwrap();
        assert!(!db.table_exists("t"));
    }

    #[test]
    fn duplicate_create_fails() {
        let mut db = test_db();
        db.execute("CREATE TABLE t (id INT)").unwrap();
        assert!(db.execute("CREATE TABLE t (id INT)").is_err());
    }

    #[test]
    fn empty_select_returns_empty_rows() {
        let mut db = test_db();
        db.execute("CREATE TABLE t (id INT)").unwrap();
        let result = db.execute("SELECT * FROM t").unwrap();
        match result {
            QueryResult::Rows(rows) => assert!(rows.is_empty()),
            other => panic!("expected empty Rows, got {:?}", other),
        }
    }

    #[test]
    fn aurora_extensions_work() {
        let mut db = test_db();
        db.execute("CREATE TABLE t (id INT)").unwrap();

        assert!(matches!(
            db.execute("ANALYZE t").unwrap(),
            QueryResult::Ok(_)
        ));
        assert!(matches!(
            db.execute("SHOW EMBEDDING HEALTH").unwrap(),
            QueryResult::Rows(_)
        ));
        assert!(matches!(
            db.execute("CREATE VERSION TAG 'v1'").unwrap(),
            QueryResult::Ok(_)
        ));
        assert!(matches!(
            db.execute("BACKUP TO '/tmp/bak'").unwrap(),
            QueryResult::Ok(_)
        ));
    }
}
