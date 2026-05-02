//! PostgreSQL wire protocol server.
//!
//! Accepts TCP connections, performs the startup handshake, and handles
//! simple query protocol (Q message) by routing to the SQL parser/executor.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::io::{AsyncWriteExt, BufReader, BufWriter};
use tokio::net::{TcpListener, TcpStream};

use galaxdb_common::GalaxError;
use galaxdb_sql::executor::{Catalog, ExecuteResult};
use galaxdb_sql::parser;
use galaxdb_sql::planner;
use galaxdb_sql::ast::AuroraStatement;

use crate::messages::*;

/// Wire protocol server configuration.
#[derive(Debug, Clone)]
pub struct WireServerConfig {
    /// Address to bind (e.g., "0.0.0.0:5433").
    pub listen_addr: String,
    /// Maximum concurrent connections (default 1000).
    pub max_connections: usize,
}

impl Default for WireServerConfig {
    fn default() -> Self {
        Self {
            listen_addr: "0.0.0.0:5433".to_string(),
            max_connections: 1000,
        }
    }
}

/// The wire protocol server.
pub struct WireServer {
    config: WireServerConfig,
    active_connections: Arc<AtomicUsize>,
}

impl WireServer {
    /// Create a new wire server.
    pub fn new(config: WireServerConfig) -> Self {
        Self {
            config,
            active_connections: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Get the number of active connections.
    pub fn active_connections(&self) -> usize {
        self.active_connections.load(Ordering::Relaxed)
    }

    /// Get the max connections limit.
    pub fn max_connections(&self) -> usize {
        self.config.max_connections
    }

    /// Start the server and listen for connections.
    pub async fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let listener = TcpListener::bind(&self.config.listen_addr).await?;
        tracing::info!(addr = %self.config.listen_addr, "wire protocol server listening");

        loop {
            let (stream, addr) = listener.accept().await?;

            let current = self.active_connections.load(Ordering::Relaxed);
            if current >= self.config.max_connections {
                tracing::warn!(
                    addr = %addr,
                    current,
                    max = self.config.max_connections,
                    "rejecting connection: too many connections"
                );
                // Send error and close
                let mut writer = BufWriter::new(stream);
                let _ = write_error_response(&mut writer, "53300", "too many connections").await;
                let _ = writer.flush().await;
                continue;
            }

            self.active_connections.fetch_add(1, Ordering::Relaxed);
            let counter = self.active_connections.clone();

            tokio::spawn(async move {
                if let Err(e) = handle_connection(stream).await {
                    tracing::debug!(addr = %addr, error = %e, "connection closed");
                }
                counter.fetch_sub(1, Ordering::Relaxed);
            });
        }
    }
}

/// Handle a single client connection.
async fn handle_connection(stream: TcpStream) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (read_half, write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);
    let mut writer = BufWriter::new(write_half);

    // Step 1: Read startup message
    let startup = read_startup(&mut reader).await?;

    if startup.protocol_version != PROTOCOL_VERSION {
        write_error_response(
            &mut writer,
            "08P01",
            &format!(
                "unsupported protocol version: {}",
                startup.protocol_version
            ),
        )
        .await?;
        writer.flush().await?;
        return Ok(());
    }

    // Step 2: Send AuthenticationOk
    write_auth_ok(&mut writer).await?;

    // Step 3: Send ParameterStatus messages
    write_parameter_status(&mut writer, "server_version", "16.0.0-galaxdb").await?;
    write_parameter_status(&mut writer, "server_encoding", "UTF8").await?;
    write_parameter_status(&mut writer, "client_encoding", "UTF8").await?;
    write_parameter_status(&mut writer, "DateStyle", "ISO, MDY").await?;
    write_parameter_status(&mut writer, "integer_datetimes", "on").await?;
    write_parameter_status(&mut writer, "standard_conforming_strings", "on").await?;

    // Step 4: Send BackendKeyData
    let process_id = std::process::id() as i32;
    write_backend_key_data(&mut writer, process_id, 0).await?;

    // Step 5: Send ReadyForQuery
    write_ready_for_query(&mut writer, b'I').await?;
    writer.flush().await?;

    // Step 6: Query loop
    let mut catalog = Catalog::new();

    loop {
        let sql = match read_query(&mut reader).await {
            Ok(q) => q,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        };

        tracing::debug!(sql = %sql, "received query");

        // Parse
        let statements = match parser::parse(&sql) {
            Ok(stmts) => stmts,
            Err(GalaxError::SqlParse { position, message }) => {
                write_error_response(
                    &mut writer,
                    "42601",
                    &format!("syntax error at position {}: {}", position, message),
                )
                .await?;
                write_ready_for_query(&mut writer, b'I').await?;
                writer.flush().await?;
                continue;
            }
            Err(e) => {
                write_error_response(&mut writer, "XX000", &format!("{}", e)).await?;
                write_ready_for_query(&mut writer, b'I').await?;
                writer.flush().await?;
                continue;
            }
        };

        // Execute each statement
        for stmt in &statements {
            match execute_statement(stmt, &mut catalog, &mut writer).await {
                Ok(()) => {}
                Err(e) => {
                    write_error_response(&mut writer, "XX000", &format!("{}", e)).await?;
                }
            }
        }

        write_ready_for_query(&mut writer, b'I').await?;
        writer.flush().await?;
    }

    Ok(())
}

/// Execute a single statement and write the response.
async fn execute_statement<W: AsyncWriteExt + Unpin>(
    stmt: &AuroraStatement,
    catalog: &mut Catalog,
    writer: &mut W,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Plan the statement
    let plan = match stmt {
        AuroraStatement::Standard(boxed_stmt) => {
            // Convert standard SQL to a plan based on statement type
            match boxed_stmt.as_ref() {
                sqlparser::ast::Statement::Query(_) => {
                    planner::plan_select("unknown".to_string(), vec![], None)
                }
                sqlparser::ast::Statement::Insert(_) => {
                    planner::plan_insert("unknown".to_string(), vec![], vec![])
                }
                sqlparser::ast::Statement::Update { .. } => {
                    planner::plan_update("unknown".to_string(), vec![], None)
                }
                sqlparser::ast::Statement::Delete(_) => {
                    planner::plan_delete("unknown".to_string(), None)
                }
                sqlparser::ast::Statement::Drop { names, if_exists, .. } => {
                    let name = names.first().map(|n: &sqlparser::ast::ObjectName| n.to_string()).unwrap_or_default();
                    planner::plan_drop_table(name, *if_exists)
                }
                sqlparser::ast::Statement::CreateTable(ct) => {
                    let table_name = ct.name.to_string();
                    let columns = ct.columns.iter().map(|c| {
                        galaxdb_sql::ast::ColumnDef {
                            name: c.name.to_string(),
                            data_type: format!("{}", c.data_type),
                            nullable: true,
                            primary_key: false,
                            embedding: None,
                        }
                    }).collect();
                    planner::plan_create_table(galaxdb_sql::ast::CreateTableStmt {
                        table_name,
                        columns,
                        if_not_exists: ct.if_not_exists,
                    })
                }
                _ => {
                    write_command_complete(writer, "OK").await?;
                    return Ok(());
                }
            }
        }
        AuroraStatement::CreateTable(ct) => planner::plan_create_table(ct.clone()),
        AuroraStatement::Analyze { table } => planner::QueryPlan::Analyze { table: table.clone() },
        AuroraStatement::BackupTo { path } => planner::QueryPlan::Backup { path: path.clone() },
        AuroraStatement::RestoreFrom { path } => planner::QueryPlan::Restore { path: path.clone() },
        AuroraStatement::ShowEmbeddingHealth { table } => {
            planner::QueryPlan::ShowEmbeddingHealth { table: table.clone() }
        }
        AuroraStatement::CreateVersionTag(tag) => {
            planner::QueryPlan::CreateVersionTag(tag.clone())
        }
        AuroraStatement::BulkInsert(bi) => {
            planner::QueryPlan::BulkInsert { table: bi.table.clone() }
        }
        _ => {
            write_command_complete(writer, "OK").await?;
            return Ok(());
        }
    };

    // Execute
    let result = galaxdb_sql::executor::execute(&plan, catalog);

    // Write response
    match result {
        ExecuteResult::Rows { columns, rows } => {
            let col_descs: Vec<ColumnDesc> = columns.iter().map(|c| ColumnDesc::text(c)).collect();
            write_row_description(writer, &col_descs).await?;

            for row in &rows {
                let values: Vec<Option<&str>> = row
                    .columns
                    .iter()
                    .map(|(_, v)| match v {
                        galaxdb_sql::planner::Value::Text(s) => Some(s.as_str()),
                        galaxdb_sql::planner::Value::Integer(_) => None, // simplified
                        _ => None,
                    })
                    .collect();
                write_data_row(writer, &values).await?;
            }

            write_command_complete(writer, &format!("SELECT {}", rows.len())).await?;
        }
        ExecuteResult::RowCount(n) => {
            write_command_complete(writer, &format!("OK {}", n)).await?;
        }
        ExecuteResult::Ok(msg) => {
            write_command_complete(writer, &msg).await?;
        }
        ExecuteResult::Error(msg) => {
            write_error_response(writer, "42000", &msg).await?;
        }
    }

    Ok(())
}
