//! GalaxDB Server library — the accept loop and connection handler,
//! extracted from `main.rs` so integration tests can drive a real TCP
//! listener against a temp directory.
//!
//! The binary (`src/main.rs`) is a thin wrapper around [`run`] that
//! parses CLI args and installs a tracing subscriber.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::io::{AsyncWriteExt, BufReader, BufWriter};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::RwLock;

use galaxdb_embedded::Database;
use galaxdb_wire::messages::*;
use galaxdb_wire::pg_catalog;

/// Server configuration.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Bind address, e.g. `"0.0.0.0:5433"`.
    pub bind_addr: String,
    /// Data directory passed to [`Database::open`].
    pub data_dir: String,
    /// Maximum concurrent client connections.
    pub max_connections: usize,
    /// Optional path to the `galaxdb-sidecar` binary. When set the
    /// server opens the database in sidecar mode so embedding columns
    /// are populated by the real model. When absent the database opens
    /// without a sidecar — scalar SQL works, semantic search returns
    /// `SidecarUnavailable`.
    pub sidecar_binary: Option<String>,
    /// HuggingFace model id for the sidecar (e.g.
    /// `"sentence-transformers/all-MiniLM-L6-v2"`). Required when
    /// `sidecar_binary` is set; ignored otherwise.
    pub model_id: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: "0.0.0.0:5433".to_string(),
            data_dir: String::new(),
            max_connections: 1000,
            sidecar_binary: None,
            model_id: None,
        }
    }
}

/// Bind the wire-protocol listener and return `(local_addr, join_handle)`.
///
/// The listener is bound synchronously — the returned `SocketAddr` is
/// the real local address (useful when `bind_addr` uses port 0). The
/// accept loop runs on the provided tokio runtime; caller can abort
/// it by dropping the handle.
pub async fn start(
    config: ServerConfig,
) -> std::io::Result<(std::net::SocketAddr, tokio::task::JoinHandle<()>)> {
    let listener = TcpListener::bind(&config.bind_addr).await?;
    let local_addr = listener.local_addr()?;

    tracing::info!(
        addr = %local_addr,
        data_dir = %config.data_dir,
        "GalaxDB server started"
    );

    let db = Arc::new(RwLock::new(
        // Task 40.1: spawn sidecar when configured. The sidecar binary
        // and model id are optional — scalar SQL works without them.
        if let (Some(sidecar_bin), Some(model)) = (
            config.sidecar_binary.as_deref(),
            config.model_id.as_deref(),
        ) {
            tracing::info!(
                sidecar = %sidecar_bin,
                model = %model,
                "opening database with embedding sidecar"
            );
            Database::open_with_sidecar(&config.data_dir, sidecar_bin, model)
                .expect("failed to open database with sidecar")
        } else {
            Database::open(&config.data_dir).expect("failed to open database")
        },
    ));
    let active = Arc::new(AtomicUsize::new(0));
    let max_connections = config.max_connections;

    // Task 38.3: mirror the live connection count into the observe
    // crate's Prometheus gauge so /metrics reports it accurately.
    // Eagerly register all metrics so the first scrape is complete.
    galaxdb_observe::register_all_metrics();
    let metrics = galaxdb_observe::metrics();
    metrics.connections_active.set(0);

    let handle = tokio::spawn(async move {
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

            let new_count = active.fetch_add(1, Ordering::Relaxed) + 1;
            metrics.connections_active.set(new_count as i64);
            let db = db.clone();
            let counter = active.clone();
            let metrics = metrics.clone();

            tokio::spawn(async move {
                if let Err(e) = handle_connection(stream, db).await {
                    tracing::debug!(peer = %peer, error = %e, "connection closed");
                }
                let new_count = counter.fetch_sub(1, Ordering::Relaxed) - 1;
                metrics.connections_active.set(new_count as i64);
            });
        }
    });

    Ok((local_addr, handle))
}

async fn handle_connection(
    stream: TcpStream,
    db: Arc<RwLock<Database>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (read_half, write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);
    let mut writer = BufWriter::new(write_half);

    // Startup handshake.
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

    // Query loop.
    loop {
        let sql = match read_query(&mut reader).await {
            Ok(q) => q,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        };

        // Task 38.6: SQL commenter format — extract a W3C traceparent
        // from the `/* traceparent='...' */` suffix if the client
        // attached one. When present the query span logs the trace
        // and span ids so downstream backends can stitch together the
        // full distributed trace. Task 38.5: the child spans emitted
        // by the executor (`sql.parse`, `query.execute`,
        // `executor.full_scan`, `executor.semantic_search`) run
        // inside the `wire.query` span entered below, so tracing
        // backends that support `trace_id`/`span_id` fields link the
        // whole tree back to the client's traceparent.
        let traceparent = galaxdb_observe::extract_traceparent_from_sql(&sql);
        let wire_span = match traceparent.as_ref() {
            Some(tp) => tracing::info_span!(
                "wire.query",
                trace_id = %tp.trace_id,
                parent_span_id = %tp.span_id,
                sampled = tp.sampled,
            ),
            None => tracing::info_span!("wire.query"),
        };
        let _wire_entered = wire_span.enter();
        galaxdb_observe::metrics().queries_total.inc();

        // Catalog queries first (psycopg2 / SQLAlchemy reflection).
        if let Some(pg_result) = pg_catalog::try_handle_pg_catalog(&sql) {
            write_row_description(&mut writer, &pg_result.columns).await?;
            for row in &pg_result.rows {
                let values: Vec<Option<&str>> = row.iter().map(|v| v.as_deref()).collect();
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

        // Execute SQL — use write lock for DDL/DML, read lock for SELECT.
        //
        // Every code path here calls into the synchronous storage
        // engine, which can block on `WalWriter::append_sync`'s
        // `tokio::sync::oneshot::blocking_recv`. Blocking primitives
        // are forbidden inside a tokio runtime worker, so we offload
        // both the lock acquisition AND the executor call to
        // `spawn_blocking`. Moving the acquisition inside the blocking
        // task is what lets the group-commit wait actually block. The
        // AWS integration run on i-0b2dec9226f62db65 caught the
        // non-offloaded version panicking on INSERT.
        let upper = sql.trim().to_uppercase();
        let is_read = upper.starts_with("SELECT") || upper.starts_with("SHOW");

        let sql_owned = sql.clone();
        let db_clone = db.clone();
        let result = tokio::task::spawn_blocking(move || {
            if is_read {
                let guard = db_clone.blocking_read();
                guard.execute_readonly(&sql_owned)
            } else {
                let mut guard = db_clone.blocking_write();
                guard.execute(&sql_owned)
            }
        })
        .await
        .map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, format!("worker panic: {e}"))
        })?;

        match result {
            Ok(galaxdb_embedded::QueryResult::Rows(rows)) => {
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
