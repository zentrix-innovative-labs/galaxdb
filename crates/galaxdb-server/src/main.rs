//! GalaxDB Server — standalone server accepting PostgreSQL wire protocol connections.
//!
//! Usage:
//!   galaxdb-server                          # listen on 0.0.0.0:5433
//!   galaxdb-server --port 5434              # custom port
//!   galaxdb-server --data-dir /tmp/galaxdb  # custom data directory

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::io::{AsyncWriteExt, BufReader, BufWriter};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::RwLock;

use galaxdb_embedded::Database;
use galaxdb_wire::messages::*;
use galaxdb_wire::pg_catalog;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("galaxdb=info".parse().unwrap()),
        )
        .init();

    let port = std::env::args()
        .position(|a| a == "--port")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(5433);

    let data_dir = std::env::args()
        .position(|a| a == "--data-dir")
        .and_then(|i| std::env::args().nth(i + 1))
        .unwrap_or_else(|| {
            let dir = tempfile::tempdir().expect("failed to create temp dir");
            let path = dir.path().to_string_lossy().to_string();
            std::mem::forget(dir);
            path
        });

    let addr = format!("0.0.0.0:{}", port);
    let listener = TcpListener::bind(&addr).await.expect("failed to bind");

    tracing::info!(addr = %addr, data_dir = %data_dir, "GalaxDB server started");
    eprintln!("GalaxDB server listening on {}", addr);

    let db = Arc::new(RwLock::new(
        Database::open(&data_dir).expect("failed to open database"),
    ));
    let active = Arc::new(AtomicUsize::new(0));
    let max_connections: usize = 1000;

    loop {
        let (stream, peer) = match listener.accept().await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(error = %e, "accept failed");
                continue;
            }
        };

        let current = active.load(Ordering::Relaxed);
        if current >= max_connections {
            tracing::warn!(peer = %peer, "rejecting: too many connections");
            let mut w = BufWriter::new(stream);
            let _ = write_error_response(&mut w, "53300", "too many connections").await;
            let _ = w.flush().await;
            continue;
        }

        active.fetch_add(1, Ordering::Relaxed);
        let db = db.clone();
        let counter = active.clone();

        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, db).await {
                tracing::debug!(peer = %peer, error = %e, "connection closed");
            }
            counter.fetch_sub(1, Ordering::Relaxed);
        });
    }
}

async fn handle_connection(
    stream: TcpStream,
    db: Arc<RwLock<Database>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (read_half, write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);
    let mut writer = BufWriter::new(write_half);

    // Startup handshake
    let startup = read_startup(&mut reader).await?;
    if startup.protocol_version != PROTOCOL_VERSION {
        write_error_response(&mut writer, "08P01", "unsupported protocol version").await?;
        writer.flush().await?;
        return Ok(());
    }

    write_auth_ok(&mut writer).await?;
    write_parameter_status(&mut writer, "server_version", "16.0.0-galaxdb").await?;
    write_parameter_status(&mut writer, "server_encoding", "UTF8").await?;
    write_parameter_status(&mut writer, "client_encoding", "UTF8").await?;
    write_parameter_status(&mut writer, "DateStyle", "ISO, MDY").await?;
    write_parameter_status(&mut writer, "integer_datetimes", "on").await?;
    write_parameter_status(&mut writer, "standard_conforming_strings", "on").await?;
    write_backend_key_data(&mut writer, std::process::id() as i32, 0).await?;
    write_ready_for_query(&mut writer, b'I').await?;
    writer.flush().await?;

    // Query loop
    loop {
        let sql = match read_query(&mut reader).await {
            Ok(q) => q,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        };

        // Check pg_catalog queries first
        if let Some(pg_result) = pg_catalog::try_handle_pg_catalog(&sql) {
            write_row_description(&mut writer, &pg_result.columns).await?;
            for row in &pg_result.rows {
                let values: Vec<Option<&str>> =
                    row.iter().map(|v| v.as_deref()).collect();
                write_data_row(&mut writer, &values).await?;
            }
            write_command_complete(
                &mut writer,
                &format!("SELECT {}", pg_result.rows.len()),
            )
            .await?;
            write_ready_for_query(&mut writer, b'I').await?;
            writer.flush().await?;
            continue;
        }

        // Execute SQL — use write lock for DDL/DML, read lock for SELECT
        let upper = sql.trim().to_uppercase();
        let is_read = upper.starts_with("SELECT") || upper.starts_with("SHOW");

        let result = if is_read {
            // Read path — multiple concurrent readers allowed
            let db = db.read().await;
            db.execute_readonly(&sql)
        } else {
            // Write path — exclusive access
            let mut db = db.write().await;
            db.execute_async(&sql).await
        };

        match result {
            Ok(galaxdb_embedded::QueryResult::Rows(rows)) => {
                // Build column descriptions from first row
                let col_descs: Vec<ColumnDesc> = if let Some(first) = rows.first() {
                    first
                        .values
                        .iter()
                        .map(|(name, _)| ColumnDesc::text(name))
                        .collect()
                } else {
                    vec![]
                };

                write_row_description(&mut writer, &col_descs).await?;

                for row in &rows {
                    let values: Vec<Option<&str>> =
                        row.values.iter().map(|(_, v)| Some(v.as_str())).collect();
                    write_data_row(&mut writer, &values).await?;
                }

                write_command_complete(
                    &mut writer,
                    &format!("SELECT {}", rows.len()),
                )
                .await?;
            }
            Ok(galaxdb_embedded::QueryResult::RowCount(n)) => {
                write_command_complete(&mut writer, &format!("OK {}", n)).await?;
            }
            Ok(galaxdb_embedded::QueryResult::Ok(msg)) => {
                write_command_complete(&mut writer, &msg).await?;
            }
            Err(e) => {
                write_error_response(&mut writer, "42000", &format!("{}", e)).await?;
            }
        }

        write_ready_for_query(&mut writer, b'I').await?;
        writer.flush().await?;
    }

    Ok(())
}
