//! Query executor — executes query plans against the storage engine.
//!
//! The executor is the bridge between the SQL layer and the storage engine.
//! It translates query plans into storage operations (memtable writes, ART
//! lookups, PAX block reads, etc.).

use galaxdb_common::{GalaxError, GalaxResult};

use crate::planner::*;

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
    /// DDL completed (CREATE TABLE, DROP TABLE).
    Ok(String),
    /// Error with message.
    Error(String),
}

/// Catalog entry for a table.
#[derive(Debug, Clone)]
pub struct TableEntry {
    pub name: String,
    pub columns: Vec<CatalogColumn>,
    pub has_embedding: bool,
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
#[derive(Debug, Default)]
pub struct Catalog {
    tables: std::collections::HashMap<String, TableEntry>,
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
}

/// Execute a query plan against the catalog.
///
/// This is a simplified executor that handles DDL and validates DML.
/// Full storage engine integration (memtable writes, ART lookups, etc.)
/// will be wired in when the engine facade is built.
pub fn execute(plan: &QueryPlan, catalog: &mut Catalog) -> ExecuteResult {
    match plan {
        QueryPlan::CreateTable(stmt) => execute_create_table(stmt, catalog),
        QueryPlan::DropTable { name, if_exists } => {
            execute_drop_table(name, *if_exists, catalog)
        }
        QueryPlan::Insert {
            table,
            columns,
            values,
        } => execute_insert(table, columns, values, catalog),
        QueryPlan::Update {
            table,
            assignments,
            filter,
        } => execute_update(table, assignments, filter, catalog),
        QueryPlan::Delete { table, filter } => execute_delete(table, filter, catalog),
        QueryPlan::FullScan {
            table,
            filter,
            columns,
        } => execute_select(table, columns, filter, catalog),
        QueryPlan::Analyze { table } => {
            if !catalog.table_exists(table) {
                return ExecuteResult::Error(format!("table not found: {}", table));
            }
            ExecuteResult::Ok(format!("ANALYZE {}", table))
        }
        QueryPlan::Backup { path } => ExecuteResult::Ok(format!("BACKUP TO '{}'", path)),
        QueryPlan::Restore { path } => ExecuteResult::Ok(format!("RESTORE FROM '{}'", path)),
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
        QueryPlan::CreateVersionTag(stmt) => {
            ExecuteResult::Ok(format!("CREATE VERSION TAG '{}'", stmt.name))
        }
        QueryPlan::BulkInsert { table } => {
            if !catalog.table_exists(table) {
                return ExecuteResult::Error(format!("table not found: {}", table));
            }
            ExecuteResult::Ok(format!("BULK INSERT INTO {}", table))
        }
        QueryPlan::SemanticSearch { table, .. } | QueryPlan::HybridSearch { table, .. } => {
            if !catalog.table_exists(table) {
                return ExecuteResult::Error(format!("table not found: {}", table));
            }
            // Semantic search requires the embedding sidecar (Month 3)
            ExecuteResult::Rows {
                columns: vec!["id".to_string(), "score".to_string()],
                rows: vec![],
            }
        }
        QueryPlan::PointLookup { table, .. } => {
            if !catalog.table_exists(table) {
                return ExecuteResult::Error(format!("table not found: {}", table));
            }
            ExecuteResult::Rows {
                columns: vec![],
                rows: vec![],
            }
        }
    }
}

fn execute_create_table(
    stmt: &crate::ast::CreateTableStmt,
    catalog: &mut Catalog,
) -> ExecuteResult {
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
    };

    match catalog.create_table(stmt.table_name.clone(), entry) {
        Ok(()) => ExecuteResult::Ok(format!("CREATE TABLE {}", stmt.table_name)),
        Err(e) => ExecuteResult::Error(format!("{}", e)),
    }
}

fn execute_drop_table(name: &str, if_exists: bool, catalog: &mut Catalog) -> ExecuteResult {
    match catalog.drop_table(name) {
        Ok(_) => ExecuteResult::Ok(format!("DROP TABLE {}", name)),
        Err(_) if if_exists => ExecuteResult::Ok(format!("DROP TABLE IF EXISTS {}", name)),
        Err(e) => ExecuteResult::Error(format!("{}", e)),
    }
}

fn execute_insert(
    table: &str,
    columns: &[String],
    values: &[Value],
    catalog: &Catalog,
) -> ExecuteResult {
    if !catalog.table_exists(table) {
        return ExecuteResult::Error(format!("table not found: {}", table));
    }

    // Check column count matches
    if !columns.is_empty() && columns.len() != values.len() {
        return ExecuteResult::Error(format!(
            "column count ({}) does not match value count ({})",
            columns.len(),
            values.len()
        ));
    }

    // In the full implementation, this would write to memtable + WAL + ART
    ExecuteResult::RowCount(1)
}

fn execute_update(
    table: &str,
    assignments: &[(String, Value)],
    _filter: &Option<FilterExpr>,
    catalog: &Catalog,
) -> ExecuteResult {
    if !catalog.table_exists(table) {
        return ExecuteResult::Error(format!("table not found: {}", table));
    }

    let table_entry = catalog.get_table(table).unwrap();

    // Check if any assignment targets an embedding source column (Req 15.5)
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

    // In the full implementation, this would write new MVCC version
    ExecuteResult::RowCount(0)
}

fn execute_delete(
    table: &str,
    _filter: &Option<FilterExpr>,
    catalog: &Catalog,
) -> ExecuteResult {
    if !catalog.table_exists(table) {
        return ExecuteResult::Error(format!("table not found: {}", table));
    }
    // In the full implementation, this would write tombstone to memtable + WAL
    ExecuteResult::RowCount(0)
}

fn execute_select(
    table: &str,
    _columns: &[String],
    _filter: &Option<FilterExpr>,
    catalog: &Catalog,
) -> ExecuteResult {
    if !catalog.table_exists(table) {
        return ExecuteResult::Error(format!("table not found: {}", table));
    }
    // In the full implementation, this would do ART lookup or scan with zone-map pruning
    ExecuteResult::Rows {
        columns: vec![],
        rows: vec![],
    }
}
