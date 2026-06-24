//! GalaxDB Server library — the accept loop and connection handler,
//! extracted from `main.rs` so integration tests can drive a real TCP
//! listener against a temp directory.
//!
//! The binary (`src/main.rs`) is a thin wrapper around [`run`] that
//! parses CLI args and installs a tracing subscriber.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::RwLock;

use galaxdb_auth::{
    AuditEvent, AuditOutcome, AuditSink, Authenticator, AuthStep, FileAuditSink, NoOpAuditSink,
    Role, ScramAuthenticator, SessionContext,
};
use galaxdb_embedded::Database;
use galaxdb_wire::messages::*;
use galaxdb_wire::pg_catalog;
use galaxdb_wire::tls::{self, Prologue, ReexportedTlsAcceptor as TlsAcceptor, TlsMode};

pub mod tuning;

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
    /// Authentication mode (task 6, Req 1).
    ///
    /// * `true` — every connection must authenticate via SCRAM-SHA-256
    ///   before running statements. At first startup with an empty auth
    ///   catalog the server provisions the initial superuser from
    ///   `GALAXDB_INITIAL_SUPERUSER` / `GALAXDB_INITIAL_SUPERUSER_PASSWORD`
    ///   (or [`Self::initial_superuser`]); if those are absent it refuses
    ///   to start rather than shipping a default password.
    /// * `false` — trusted-local mode: connections skip authentication and
    ///   run as the configured superuser. A startup warning names that
    ///   auth is disabled (never silent). This is the loopback/development
    ///   default that preserves v1 behavior.
    pub auth_enabled: bool,
    /// The superuser name connections run as in trusted-local mode
    /// (`auth_enabled = false`). Defaults to `galaxdb`.
    pub trusted_local_user: String,
    /// Initial superuser `(name, password)` to provision at first startup
    /// when `auth_enabled` and the catalog is empty. When `None` the
    /// server reads `GALAXDB_INITIAL_SUPERUSER` /
    /// `GALAXDB_INITIAL_SUPERUSER_PASSWORD`. Carrying it in config is for
    /// tests; production uses the env vars so the password never lands in
    /// a config file in plaintext.
    pub initial_superuser: Option<(String, String)>,
    /// TLS negotiation mode (task 7, Req 2). `disable` never offers TLS;
    /// `allow` (default) offers it on `SSLRequest` but also serves
    /// plaintext; `require` rejects any connection that does not negotiate
    /// TLS first.
    pub tls_mode: TlsMode,
    /// Path to the PEM server certificate chain (leaf first). Required
    /// when `tls_mode` is `allow` (to actually offer TLS) or `require`.
    pub tls_cert_path: Option<String>,
    /// Path to the PEM server private key (PKCS#8 / SEC1 / PKCS#1).
    pub tls_key_path: Option<String>,
    /// Optional path to a JSONL security audit log (Req 4). When set, the
    /// server records authentication outcomes, authorization denials, and
    /// role/grant/DDL changes via a [`galaxdb_auth::FileAuditSink`]. When
    /// `None`, audit events are discarded (no-op).
    pub audit_log_path: Option<String>,
    /// Auto-tuned configuration (Req 12). At startup the server probes the
    /// host (RAM + CPU) and derives buffer-pool / memtable / compaction
    /// concurrency, unless an explicit override is set here. Defaults to
    /// auto-tune enabled with no overrides.
    pub auto_tune: galaxdb_common::AutoTuneConfig,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: "0.0.0.0:5433".to_string(),
            data_dir: String::new(),
            max_connections: 1000,
            sidecar_binary: None,
            model_id: None,
            auth_enabled: false,
            trusted_local_user: "galaxdb".to_string(),
            initial_superuser: None,
            tls_mode: TlsMode::Disable,
            tls_cert_path: None,
            tls_key_path: None,
            audit_log_path: None,
            auto_tune: galaxdb_common::AutoTuneConfig::default(),
        }
    }
}

/// Per-connection authentication policy, resolved once at startup and
/// shared (cheaply cloned) into every connection handler.
#[derive(Clone)]
struct AuthPolicy {
    /// Whether SCRAM authentication is required.
    enabled: bool,
    /// The superuser name used in trusted-local mode (`enabled = false`).
    trusted_local_user: String,
    /// Audit sink for authentication outcomes (login success/failure).
    /// Shared with the `Database` so auth and authz events land in the
    /// same place.
    audit: Arc<dyn AuditSink>,
}

/// Resolve the initial-superuser credential from config or environment.
/// Config takes precedence (used by tests); production sets the env vars
/// so the password never lands in a config file.
fn resolve_initial_superuser(config: &ServerConfig) -> Option<(String, String)> {
    if let Some((n, p)) = config.initial_superuser.clone() {
        return Some((n, p));
    }
    match (
        std::env::var("GALAXDB_INITIAL_SUPERUSER"),
        std::env::var("GALAXDB_INITIAL_SUPERUSER_PASSWORD"),
    ) {
        (Ok(n), Ok(p)) if !n.is_empty() && !p.is_empty() => Some((n, p)),
        _ => None,
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

    // Task 4 (Req 4): build the security audit sink. A JSONL file when
    // configured, else a no-op. Shared between the auth path (login
    // events) and the Database (authz/role-change events) so all security
    // events land in one place.
    let audit_sink: Arc<dyn AuditSink> = match config.audit_log_path.as_deref() {
        Some(path) => {
            let sink = FileAuditSink::open(path)
                .unwrap_or_else(|e| panic!("failed to open audit log '{path}': {e}"));
            tracing::info!(path = %path, "security audit log enabled (JSONL)");
            Arc::new(sink)
        }
        None => Arc::new(NoOpAuditSink),
    };

    // Task 15 (Req 12): probe the host and resolve the effective auto-tune
    // configuration, then log it with the source of each value (auto-derived
    // vs overridden vs static-default). The derived buffer-pool and memtable
    // sizes are applied to the engine via open_with_tuning. The
    // compaction-concurrency value is reported for operator visibility; the
    // OSS engine does not yet run a background compaction driver that would
    // consume it (tracked in docs/CONSOLIDATION.md), so it is surfaced but
    // not yet wired to a runtime compactor.
    let tuning = tuning::resolve_tuning(&config.auto_tune);
    tracing::info!("{}", tuning.describe());
    let memtable_bytes = tuning.memtable_size_bytes.value;
    let sst_cache_bytes = tuning.buffer_pool_bytes.value;

    let db = Arc::new(RwLock::new(
        // Task 40.1: spawn sidecar when configured. The sidecar binary
        // and model id are optional — scalar SQL works without them.
        {
            let mut base = Database::open_with_tuning(
                &config.data_dir,
                memtable_bytes,
                sst_cache_bytes,
            )
            .expect("failed to open database");
            if let (Some(sidecar_bin), Some(model)) = (
                config.sidecar_binary.as_deref(),
                config.model_id.as_deref(),
            ) {
                tracing::info!(
                    sidecar = %sidecar_bin,
                    model = %model,
                    "opening database with embedding sidecar"
                );
                base.attach_sidecar(sidecar_bin, model)
                    .expect("failed to attach embedding sidecar");
            }
            // Attach the audit sink so the executor records authz denials
            // and role/grant/DDL changes (Req 4).
            base.with_audit_sink(audit_sink.clone())
        },
    ));

    // Task 6 (Req 1): resolve the authentication policy and, when auth is
    // enabled, provision the initial superuser on a fresh catalog. If auth
    // is enabled, the catalog is empty, and no initial superuser is
    // configured, refuse to start — never ship a default password.
    //
    // Provisioning writes through the WAL (`put_sync` → `blocking_recv`),
    // which must not run on a tokio runtime thread, so the catalog check +
    // provisioning are offloaded to `spawn_blocking`.
    if config.auth_enabled {
        let db_for_setup = db.clone();
        let initial = resolve_initial_superuser(&config);
        tokio::task::spawn_blocking(move || {
            let guard = db_for_setup.blocking_read();
            if !guard.any_role_exists() {
                match initial {
                    Some((name, password)) => {
                        guard
                            .provision_superuser(&name, &password)
                            .expect("failed to provision initial superuser");
                        tracing::info!(
                            superuser = %name,
                            "auth enabled: provisioned initial superuser on empty catalog"
                        );
                    }
                    None => {
                        panic!(
                            "auth is enabled but the auth catalog is empty and no initial \
                             superuser is configured. Set GALAXDB_INITIAL_SUPERUSER and \
                             GALAXDB_INITIAL_SUPERUSER_PASSWORD (or ServerConfig.initial_superuser). \
                             Refusing to start with no superuser — GalaxDB never ships a default \
                             password."
                        );
                    }
                }
            }
        })
        .await
        .expect("initial superuser provisioning task panicked");
        tracing::info!("authentication: SCRAM-SHA-256 required on every connection");
    } else {
        tracing::warn!(
            user = %config.trusted_local_user,
            "AUTHENTICATION IS DISABLED (trusted-local mode): every connection runs as the \
             superuser '{}' WITHOUT verifying a password. Enable auth for any networked \
             deployment.",
            config.trusted_local_user,
        );
    }

    let auth_policy = AuthPolicy {
        enabled: config.auth_enabled,
        trusted_local_user: config.trusted_local_user.clone(),
        audit: audit_sink.clone(),
    };

    // Task 7 (Req 2): build the TLS acceptor when TLS is offered. In
    // `allow`/`require` modes a cert+key are mandatory; a misconfiguration
    // is a hard startup error (never silently downgraded to plaintext).
    let tls_acceptor: Option<TlsAcceptor> = match config.tls_mode {
        TlsMode::Disable => None,
        TlsMode::Allow | TlsMode::Require => {
            let cert = config.tls_cert_path.as_deref().unwrap_or_else(|| {
                panic!(
                    "tls_mode is '{}' but no tls_cert_path is configured",
                    config.tls_mode.label()
                )
            });
            let key = config.tls_key_path.as_deref().unwrap_or_else(|| {
                panic!(
                    "tls_mode is '{}' but no tls_key_path is configured",
                    config.tls_mode.label()
                )
            });
            let server_config = tls::load_server_config(cert, key)
                .expect("failed to load TLS certificate/key");
            tracing::info!(
                mode = config.tls_mode.label(),
                cert = %cert,
                "TLS enabled (rustls, TLS 1.2/1.3)"
            );
            Some(tls::acceptor(server_config))
        }
    };
    let tls_mode = config.tls_mode;

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

            // Disable Nagle's algorithm. The PostgreSQL wire protocol is
            // request/response with many small backend messages (especially
            // the extended-query path: BindComplete, DataRow,
            // CommandComplete sent across separate writes). With Nagle on,
            // these small segments collide with the peer's delayed-ACK
            // timer and stall ~40 ms per round trip — which caps prepared
            // single-row throughput at ~24 rows/s. Real PostgreSQL sets
            // TCP_NODELAY for the same reason. Best-effort: a failure here
            // is non-fatal, the connection still works (just slower).
            if let Err(e) = stream.set_nodelay(true) {
                tracing::warn!(peer = %peer, error = %e, "could not set TCP_NODELAY");
            }

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
            let auth_policy = auth_policy.clone();
            let tls_acceptor = tls_acceptor.clone();

            tokio::spawn(async move {
                if let Err(e) =
                    handle_prologue(stream, db, auth_policy, tls_acceptor, tls_mode).await
                {
                    tracing::debug!(peer = %peer, error = %e, "connection closed");
                }
                let new_count = counter.fetch_sub(1, Ordering::Relaxed) - 1;
                metrics.connections_active.set(new_count as i64);
            });
        }
    });

    Ok((local_addr, handle))
}

/// TCP-level connection prologue (task 7, Req 2): handle the optional
/// PostgreSQL `SSLRequest`/`GSSENCRequest` negotiation that precedes the
/// StartupMessage, then hand the (possibly TLS-wrapped) stream to the
/// generic [`serve_connection`].
///
/// The 8-byte prologue is read directly off the raw socket rather than
/// through a `BufReader`: PostgreSQL clients wait for the server's single
/// `S`/`N` byte before sending anything further, so there is no pipelined
/// data to strand in a buffer when we upgrade to TLS.
async fn handle_prologue(
    mut stream: TcpStream,
    db: Arc<RwLock<Database>>,
    auth_policy: AuthPolicy,
    tls_acceptor: Option<TlsAcceptor>,
    tls_mode: TlsMode,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    loop {
        match tls::peek_ssl_request(&mut stream).await? {
            Prologue::SslRequest => match &tls_acceptor {
                Some(acceptor) => {
                    // Accept TLS: reply 'S', complete the rustls handshake,
                    // and run the rest of the protocol on the encrypted
                    // stream (SCRAM therefore runs inside TLS — Req 2 AC6).
                    tls::write_negotiation_reply(&mut stream, true).await?;
                    let tls_stream = acceptor.accept(stream).await?;
                    return serve_connection(tls_stream, None, db, auth_policy).await;
                }
                None => {
                    // TLS not configured: decline with 'N'. The client then
                    // either continues in plaintext (libpq `sslmode=prefer`)
                    // or disconnects (`sslmode=require`). Read the fresh
                    // StartupMessage off the plaintext stream.
                    tls::write_negotiation_reply(&mut stream, false).await?;
                    return serve_connection(stream, None, db, auth_policy).await;
                }
            },
            Prologue::GssEncRequest => {
                // GSSAPI encryption is not supported; decline and read the
                // next prologue packet (the client falls back to SSLRequest
                // or a plaintext StartupMessage).
                tls::write_negotiation_reply(&mut stream, false).await?;
                continue;
            }
            Prologue::StartupHead { length, code } => {
                // A plaintext StartupMessage arrived with no prior TLS.
                if tls_mode == TlsMode::Require {
                    // Req 2 AC3: reject — TLS is mandatory.
                    let mut writer = BufWriter::new(&mut stream);
                    write_error_response(
                        &mut writer,
                        "08P01",
                        "TLS is required: reconnect with TLS enabled (e.g. sslmode=require)",
                    )
                    .await?;
                    writer.flush().await?;
                    return Ok(());
                }
                return serve_connection(stream, Some((length, code)), db, auth_policy).await;
            }
        }
    }
}

/// Serve a client connection over an established byte stream — either a
/// plaintext `TcpStream` or a `TlsStream<TcpStream>`. Generic over the
/// stream type so the auth handshake and query loop are written once and
/// shared by both transports (Req 2 AC describing the generic handler).
///
/// `startup_head` carries the 8-byte StartupMessage head (length +
/// protocol version) when it was already consumed by the plaintext
/// prologue peek; `None` means read the StartupMessage fresh (the case
/// after a TLS handshake or a declined `SSLRequest`).
async fn serve_connection<S>(
    stream: S,
    startup_head: Option<(i32, i32)>,
    db: Arc<RwLock<Database>>,
    auth_policy: AuthPolicy,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
where
    S: AsyncReadExt + AsyncWriteExt + Unpin + Send,
{
    let (read_half, write_half) = tokio::io::split(stream);
    let mut reader = BufReader::new(read_half);
    let mut writer = BufWriter::new(write_half);

    // Startup handshake. After a TLS upgrade or a declined SSLRequest the
    // StartupMessage is read fresh; after a plaintext prologue peek the
    // 8-byte head was already consumed and is supplied here.
    let startup = match startup_head {
        Some((length, code)) => read_startup_after_head(&mut reader, length, code).await?,
        None => read_startup(&mut reader).await?,
    };
    if startup.protocol_version != PROTOCOL_VERSION {
        write_error_response(&mut writer, "08P01", "unsupported protocol version").await?;
        writer.flush().await?;
        return Ok(());
    }

    // The role name the client claims via the startup `user` parameter.
    let user_param = startup
        .params
        .iter()
        .find(|(k, _)| k == "user")
        .map(|(_, v)| v.clone());

    // Task 6 / Req 1: authenticate before doing anything else. On success
    // we hold the authenticated `SessionContext`; on failure the client
    // has already been sent `28P01` and the connection is closed.
    let session = match authenticate(&mut reader, &mut writer, &db, &auth_policy, user_param).await
    {
        Ok(s) => s,
        Err(AuthOutcome::Rejected) => {
            // 28P01 already written by `authenticate`.
            writer.flush().await?;
            return Ok(());
        }
        Err(AuthOutcome::Io(e)) => return Err(e),
    };

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

    // Query loop — dispatches both the simple (`Q`) and extended
    // (Parse/Bind/Describe/Execute/Close/Sync) query protocols (Req 6).
    // Per-connection prepared statements and portals live here; they are
    // dropped when the connection closes. `ReadyForQuery` is sent after a
    // simple query and after `Sync` (extended protocol), never mid-series.
    let mut prepared: std::collections::HashMap<String, PreparedStatement> =
        std::collections::HashMap::new();
    let mut portals: std::collections::HashMap<String, Portal> =
        std::collections::HashMap::new();

    loop {
        let msg = match read_message(&mut reader).await {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        };
        match msg {
            FrontendMessage::Terminate => break,

            FrontendMessage::Query(sql) => {
                // COPY ... FROM STDIN / TO STDOUT is a wire sub-protocol,
                // not a normal statement — intercept it before execution.
                if let Some(cmd) = galaxdb_wire::copy::parse_copy(&sql) {
                    run_copy(&mut reader, &mut writer, &db, &session, cmd).await?;
                } else {
                    run_simple_query(&mut writer, &db, &session, &sql).await?;
                }
                write_ready_for_query(&mut writer, b'I').await?;
                writer.flush().await?;
            }

            // ── Extended query protocol ────────────────────────────
            FrontendMessage::Parse {
                statement,
                query,
                param_types,
            } => {
                // COPY is a wire sub-protocol, not a parseable statement —
                // detect it at Parse and store the command for Execute.
                if let Some(cmd) = galaxdb_wire::copy::parse_copy(&query) {
                    prepared.insert(statement, PreparedStatement::Copy(cmd));
                    write_parse_complete(&mut writer).await?;
                    writer.flush().await?;
                    continue;
                }
                // Parse the template ONCE here; Bind/Execute reuse this AST
                // (no re-parse per execution — Req 7). A parse error is
                // reported now via ErrorResponse.
                let db_clone = db.clone();
                let q = query.clone();
                let prepared_result = tokio::task::spawn_blocking(move || {
                    db_clone.blocking_read().prepare(&q)
                })
                .await
                .map_err(|e| std::io::Error::other(format!("prepare worker panic: {e}")))?;
                match prepared_result {
                    Ok(template) => {
                        prepared.insert(
                            statement,
                            PreparedStatement::Normal {
                                template,
                                param_types,
                            },
                        );
                        write_parse_complete(&mut writer).await?;
                    }
                    Err(e) => {
                        write_error_response(&mut writer, e.sqlstate(), &format!("{}", e)).await?;
                    }
                }
                writer.flush().await?;
            }

            FrontendMessage::Bind {
                portal,
                statement,
                param_formats,
                params,
                result_formats,
            } => {
                match prepared.get(&statement) {
                    None => {
                        write_error_response(
                            &mut writer,
                            "26000",
                            &format!("prepared statement \"{statement}\" does not exist"),
                        )
                        .await?;
                    }
                    Some(PreparedStatement::Copy(_)) => {
                        // COPY takes no parameters; the portal just refers
                        // back to the prepared COPY command.
                        portals.insert(
                            portal,
                            Portal {
                                statement: statement.clone(),
                                values: Vec::new(),
                            },
                        );
                        write_bind_complete(&mut writer).await?;
                    }
                    Some(PreparedStatement::Normal { param_types, .. }) => {
                        match bind_portal(param_types, &statement, &param_formats, &params, result_formats) {
                            Ok(p) => {
                                portals.insert(portal, p);
                                write_bind_complete(&mut writer).await?;
                            }
                            Err(msg) => {
                                write_error_response(&mut writer, "22023", &msg).await?;
                            }
                        }
                    }
                }
                writer.flush().await?;
            }

            FrontendMessage::Describe { kind, name } => {
                if kind == b'S' {
                    match prepared.get(&name) {
                        None => {
                            write_error_response(
                                &mut writer,
                                "26000",
                                &format!("prepared statement \"{name}\" does not exist"),
                            )
                            .await?;
                        }
                        Some(PreparedStatement::Copy(_)) => {
                            // COPY has no bind parameters and no result rows.
                            write_parameter_description(&mut writer, &[]).await?;
                            write_no_data(&mut writer).await?;
                        }
                        Some(PreparedStatement::Normal { template, param_types }) => {
                            let oids = resolve_param_oids(param_types, template.param_count);
                            write_parameter_description(&mut writer, &oids).await?;
                            write_describe_rows(&mut writer, template.columns.clone()).await?;
                        }
                    }
                } else {
                    // Describe portal: RowDescription | NoData (columns come
                    // from the portal's backing prepared statement).
                    let columns = portals.get(&name).and_then(|p| match prepared.get(&p.statement) {
                        Some(PreparedStatement::Normal { template, .. }) => template.columns.clone(),
                        _ => None,
                    });
                    if portals.contains_key(&name) {
                        write_describe_rows(&mut writer, columns).await?;
                    } else {
                        write_error_response(
                            &mut writer,
                            "34000",
                            &format!("portal \"{name}\" does not exist"),
                        )
                        .await?;
                    }
                }
                writer.flush().await?;
            }

            FrontendMessage::Execute { portal, .. } => {
                let Some(p) = portals.get(&portal) else {
                    write_error_response(
                        &mut writer,
                        "34000",
                        &format!("portal \"{portal}\" does not exist"),
                    )
                    .await?;
                    writer.flush().await?;
                    continue;
                };
                match prepared.get(&p.statement) {
                    None => {
                        write_error_response(
                            &mut writer,
                            "26000",
                            &format!("prepared statement \"{}\" does not exist", p.statement),
                        )
                        .await?;
                    }
                    Some(PreparedStatement::Copy(cmd)) => {
                        // COPY drives its own sub-protocol on the live
                        // reader/writer (CopyInResponse / CopyData / ...).
                        let cmd = cmd.clone();
                        run_copy(&mut reader, &mut writer, &db, &session, cmd).await?;
                    }
                    Some(PreparedStatement::Normal { template, .. }) => {
                        // Bind the parsed template + run it — no re-parse
                        // (Req 7). The client already received the
                        // RowDescription from Describe, so emit DataRows +
                        // CommandComplete only.
                        let result =
                            run_bound_execute(&db, &session, template, p.values.clone()).await?;
                        write_query_result(&mut writer, result, false).await?;
                    }
                }
                // No ReadyForQuery here — that waits for Sync.
                writer.flush().await?;
            }

            FrontendMessage::Close { kind, name } => {
                if kind == b'S' {
                    prepared.remove(&name);
                } else {
                    portals.remove(&name);
                }
                write_close_complete(&mut writer).await?;
                writer.flush().await?;
            }

            FrontendMessage::Sync => {
                write_ready_for_query(&mut writer, b'I').await?;
                writer.flush().await?;
            }

            FrontendMessage::Flush => {
                writer.flush().await?;
            }
        }
    }

    Ok(())
}

/// A statement prepared via `Parse`. Either a normal parse-once template,
/// or a COPY command (which is a wire sub-protocol, not a parseable
/// statement) detected at Parse time.
enum PreparedStatement {
    Normal {
        template: galaxdb_embedded::PreparedTemplate,
        /// Client-declared parameter type OIDs (0 = unspecified).
        param_types: Vec<i32>,
    },
    Copy(galaxdb_wire::copy::CopyCommand),
}

/// A portal produced by `Bind`: the backing prepared-statement name and
/// the decoded parameter values to bind at `Execute`.
struct Portal {
    statement: String,
    values: Vec<galaxdb_embedded::BoundValue>,
}

/// Decode the bound parameters of a `Bind` into typed values, producing a
/// portal that references its prepared statement (Req 6 AC5: text+binary).
fn bind_portal(
    param_types: &[i32],
    statement_name: &str,
    param_formats: &[i16],
    params: &[Option<Vec<u8>>],
    _result_formats: Vec<i16>,
) -> Result<Portal, String> {
    // Per PostgreSQL: 0 format codes → all text; 1 code → applies to all;
    // otherwise one code per parameter.
    let fmt_for = |i: usize| -> i16 {
        match param_formats.len() {
            0 => 0,
            1 => param_formats[0],
            _ => *param_formats.get(i).unwrap_or(&0),
        }
    };
    let mut values = Vec::with_capacity(params.len());
    for (i, val) in params.iter().enumerate() {
        let type_oid = param_types.get(i).copied().unwrap_or(0);
        let v = galaxdb_wire::param_codec::param_to_bound_value(val.as_deref(), fmt_for(i), type_oid)?;
        values.push(v);
    }
    Ok(Portal {
        statement: statement_name.to_string(),
        values,
    })
}

/// Resolve the parameter type OIDs to report in `ParameterDescription`:
/// use the client-declared types from `Parse` where given, padding any
/// inferred parameters with `text` (the executor binds text literals for
/// every supported type).
fn resolve_param_oids(param_types: &[i32], param_count: usize) -> Vec<i32> {
    let n = param_types.len().max(param_count);
    (0..n)
        .map(|i| match param_types.get(i) {
            Some(&oid) if oid != 0 => oid,
            _ => galaxdb_wire::param_codec::oid::TEXT,
        })
        .collect()
}

/// Emit a `RowDescription` for the resolved columns (text-typed, matching
/// the simple-query result path), or `NoData` when the statement returns
/// no rows.
async fn write_describe_rows<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
    columns: Option<Vec<String>>,
) -> std::io::Result<()> {
    match columns {
        Some(cols) if !cols.is_empty() => {
            let descs: Vec<ColumnDesc> = cols.iter().map(|c| ColumnDesc::text(c)).collect();
            write_row_description(writer, &descs).await
        }
        // Empty column list or a non-row statement → NoData.
        _ => write_no_data(writer).await,
    }
}

/// Run a simple-query (`Q`) statement: pg_catalog passthrough, then the
/// engine, writing RowDescription + DataRows + CommandComplete (or an
/// ErrorResponse). Does NOT write ReadyForQuery — the caller does.
async fn run_simple_query<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
    db: &Arc<RwLock<Database>>,
    session: &SessionContext,
    sql: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Build the trace span and record the SQL-commenter traceparent (if
    // any) into it. We do NOT hold the span's `Entered` guard across the
    // `.await`s below: the engine runs on the blocking pool (a different
    // thread), so a held guard would neither parent those spans nor be
    // `Send` for the spawned connection task.
    let traceparent = galaxdb_observe::extract_traceparent_from_sql(sql);
    let wire_span = match traceparent.as_ref() {
        Some(tp) => tracing::info_span!(
            "wire.query",
            trace_id = %tp.trace_id,
            parent_span_id = %tp.span_id,
            sampled = tp.sampled,
        ),
        None => tracing::info_span!("wire.query"),
    };
    wire_span.in_scope(|| galaxdb_observe::metrics().queries_total.inc());

    if let Some(pg_result) = pg_catalog::try_handle_pg_catalog(sql) {
        write_row_description(writer, &pg_result.columns).await?;
        for row in &pg_result.rows {
            let values: Vec<Option<&str>> = row.iter().map(|v| v.as_deref()).collect();
            write_data_row(writer, &values).await?;
        }
        write_command_complete(writer, &format!("SELECT {}", pg_result.rows.len())).await?;
        return Ok(());
    }

    let result = run_engine(db, session, sql).await?;
    write_query_result(writer, result, true).await
}

/// Drive the COPY sub-protocol (Req 8): `COPY t FROM STDIN` ingests text
/// rows through the bulk-insert path (not one INSERT per row); `COPY t TO
/// STDOUT` streams the table's rows back as text `CopyData`. Text format
/// only (AC4). Does NOT write ReadyForQuery — the caller does.
async fn run_copy<R, W>(
    reader: &mut R,
    writer: &mut W,
    db: &Arc<RwLock<Database>>,
    session: &SessionContext,
    cmd: galaxdb_wire::copy::CopyCommand,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
where
    R: AsyncReadExt + Unpin,
    W: AsyncWriteExt + Unpin,
{
    use galaxdb_wire::copy::CopyDirection;
    galaxdb_observe::metrics().queries_total.inc();

    // Resolve the table's full column set (also validates the table
    // exists, before we tell the client to start streaming).
    let table = cmd.table.clone();
    let all_cols = {
        let db2 = db.clone();
        let t = table.clone();
        tokio::task::spawn_blocking(move || db2.blocking_read().table_columns(&t))
            .await
            .map_err(|e| std::io::Error::other(format!("copy worker panic: {e}")))?
    };
    let all_cols = match all_cols {
        Ok(c) => c,
        Err(e) => {
            write_error_response(writer, e.sqlstate(), &format!("{e}")).await?;
            return Ok(());
        }
    };
    let num_cols = if cmd.columns.is_empty() {
        all_cols.len()
    } else {
        cmd.columns.len()
    };

    match cmd.direction {
        CopyDirection::In => {
            write_copy_in_response(writer, num_cols as u16).await?;
            writer.flush().await?;

            // Accumulate the text stream across CopyData frames.
            let mut buf: Vec<u8> = Vec::new();
            loop {
                match read_copy_in_message(reader).await? {
                    CopyInMessage::Data(mut d) => buf.append(&mut d),
                    CopyInMessage::Done => break,
                    CopyInMessage::Fail(msg) => {
                        write_error_response(
                            writer,
                            "57014",
                            &format!("COPY from stdin failed: {msg}"),
                        )
                        .await?;
                        return Ok(());
                    }
                }
            }

            // Split the text into rows (text format, Req 8 AC4).
            let text = String::from_utf8_lossy(&buf);
            let mut values: Vec<Vec<String>> = Vec::new();
            for line in text.split('\n') {
                let line = line.strip_suffix('\r').unwrap_or(line);
                if line.is_empty() || line == "\\." {
                    continue;
                }
                values.push(galaxdb_wire::copy::decode_text_row(line));
            }
            let n = values.len();

            let db2 = db.clone();
            let sess = session.clone();
            let t = table.clone();
            let cols = cmd.columns.clone();
            let result = tokio::task::spawn_blocking(move || {
                db2.blocking_write()
                    .bulk_insert_with_session(&t, cols, values, Some(sess))
            })
            .await
            .map_err(|e| std::io::Error::other(format!("copy worker panic: {e}")))?;
            match result {
                Ok(_) => write_command_complete(writer, &format!("COPY {n}")).await?,
                Err(e) => write_error_response(writer, e.sqlstate(), &format!("{e}")).await?,
            }
        }
        CopyDirection::Out => {
            // Project the requested columns (or all) and stream the rows.
            let projection = if cmd.columns.is_empty() {
                "*".to_string()
            } else {
                cmd.columns.join(", ")
            };
            let sql = format!("SELECT {projection} FROM {table}");
            let result = run_engine(db, session, &sql).await?;
            match result {
                Ok(galaxdb_embedded::QueryResult::Rows(rows)) => {
                    let ncols = rows.first().map(|r| r.values.len()).unwrap_or(num_cols);
                    write_copy_out_response(writer, ncols as u16).await?;
                    for row in &rows {
                        let cells: Vec<(bool, &str)> = row
                            .values
                            .iter()
                            .map(|(_, v)| (v == "NULL", v.as_str()))
                            .collect();
                        let mut line = galaxdb_wire::copy::encode_text_row(&cells);
                        line.push('\n');
                        write_copy_data(writer, line.as_bytes()).await?;
                    }
                    write_copy_done(writer).await?;
                    write_command_complete(writer, &format!("COPY {}", rows.len())).await?;
                }
                Ok(_) => {
                    write_copy_out_response(writer, num_cols as u16).await?;
                    write_copy_done(writer).await?;
                    write_command_complete(writer, "COPY 0").await?;
                }
                Err(e) => write_error_response(writer, e.sqlstate(), &format!("{e}")).await?,
            }
        }
    }
    Ok(())
}
/// pool (Req 7: no re-parse). The lock is chosen by the template's
/// read/write classification, matching the simple-query path.
async fn run_bound_execute(
    db: &Arc<RwLock<Database>>,
    session: &SessionContext,
    template: &galaxdb_embedded::PreparedTemplate,
    values: Vec<galaxdb_embedded::BoundValue>,
) -> Result<galaxdb_common::GalaxResult<galaxdb_embedded::QueryResult>, std::io::Error> {
    galaxdb_observe::metrics().queries_total.inc();
    let db_clone = db.clone();
    let session_clone = session.clone();
    let template = template.clone();
    tokio::task::spawn_blocking(move || {
        if template.is_read {
            let guard = db_clone.blocking_read();
            guard.execute_bound_readonly_with_session(&template, &values, Some(session_clone))
        } else {
            // Use the shared read lock for DML prepared statements so
            // concurrent clients can batch their WAL fsyncs (same as
            // run_engine's DML path).
            let guard = db_clone.blocking_read();
            guard.execute_bound_dml_concurrent(&template, &values, Some(session_clone))
        }
    })
    .await
    .map_err(|e| std::io::Error::other(format!("worker panic: {e}")))
}

/// Execute one statement against the engine on the blocking pool, choosing
/// a read or write lock by statement kind (same rule as the v1 loop).
/// DML (INSERT/UPDATE/DELETE) uses the concurrent path with `blocking_read()`
/// so multiple clients can issue writes simultaneously and share WAL fsyncs
/// through group commit — the same pattern that gives PostgreSQL its
/// multi-client throughput.
async fn run_engine(
    db: &Arc<RwLock<Database>>,
    session: &SessionContext,
    sql: &str,
) -> Result<galaxdb_common::GalaxResult<galaxdb_embedded::QueryResult>, std::io::Error> {
    let upper = sql.trim().to_uppercase();
    let is_read = upper.starts_with("SELECT") || upper.starts_with("SHOW");
    // DML (INSERT/UPDATE/DELETE) is safe on the shared read lock because:
    //   - The storage engine (Arc<Engine>) is internally thread-safe.
    //   - DML never mutates the schema catalog.
    //   - Concurrent reads allow WAL group-commit to batch multiple clients'
    //     fsyncs into a single durable write (matching PostgreSQL's model).
    let is_dml = upper.starts_with("INSERT")
        || upper.starts_with("UPDATE")
        || upper.starts_with("DELETE")
        || upper.starts_with("COPY");
    let sql_owned = sql.to_string();
    let db_clone = db.clone();
    let session_clone = session.clone();
    tokio::task::spawn_blocking(move || {
        if is_read {
            let guard = db_clone.blocking_read();
            guard.execute_readonly_with_session(&sql_owned, Some(session_clone))
        } else if is_dml {
            // Shared read lock — concurrent writers allowed.
            let guard = db_clone.blocking_read();
            guard.execute_dml_concurrent(&sql_owned, Some(session_clone))
        } else {
            // DDL (CREATE/DROP TABLE, CREATE INDEX, GRANT, etc.) needs
            // the exclusive write lock so catalog mutations are visible.
            let mut guard = db_clone.blocking_write();
            guard.execute_with_session(&sql_owned, Some(session_clone))
        }
    })
    .await
    .map_err(|e| std::io::Error::other(format!("worker panic: {e}")))
}

/// Write an engine result to the wire. `with_row_description` controls
/// whether a RowDescription precedes the DataRows (true for simple query,
/// false for extended Execute, which already sent it via Describe).
async fn write_query_result<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
    result: galaxdb_common::GalaxResult<galaxdb_embedded::QueryResult>,
    with_row_description: bool,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    match result {
        Ok(galaxdb_embedded::QueryResult::Rows(rows)) => {
            if with_row_description {
                let col_descs: Vec<ColumnDesc> = if let Some(first) = rows.first() {
                    first
                        .values
                        .iter()
                        .map(|(name, _)| ColumnDesc::text(name))
                        .collect()
                } else {
                    vec![]
                };
                write_row_description(writer, &col_descs).await?;
            }
            for row in &rows {
                let values: Vec<Option<&str>> =
                    row.values.iter().map(|(_, v)| Some(v.as_str())).collect();
                write_data_row(writer, &values).await?;
            }
            write_command_complete(writer, &format!("SELECT {}", rows.len())).await?;
        }
        Ok(galaxdb_embedded::QueryResult::RowCount(n)) => {
            write_command_complete(writer, &format!("OK {}", n)).await?;
        }
        Ok(galaxdb_embedded::QueryResult::Ok(msg)) => {
            write_command_complete(writer, &msg).await?;
        }
        Err(e) => {
            write_error_response(writer, e.sqlstate(), &format!("{}", e)).await?;
        }
    }
    Ok(())
}

/// Why authentication did not yield a session.
enum AuthOutcome {
    /// The client failed authentication; a `28P01` ErrorResponse has
    /// already been written to the connection and it should be closed.
    Rejected,
    /// A transport error occurred while reading/writing the handshake.
    Io(Box<dyn std::error::Error + Send + Sync>),
}

impl From<std::io::Error> for AuthOutcome {
    fn from(e: std::io::Error) -> Self {
        AuthOutcome::Io(Box::new(e))
    }
}

/// Run the authentication phase of the connection prologue (task 6,
/// Req 1).
///
/// * Trusted-local mode (`auth_policy.enabled == false`): no SASL exchange;
///   the connection runs as the configured superuser. The startup warning
///   that auth is disabled was already logged once at server start.
/// * SCRAM mode: advertise `AuthenticationSASL(SCRAM-SHA-256)`, drive the
///   two-round exchange against the verifier stored for the claimed role
///   in the auth catalog, and on success return the authenticated
///   `SessionContext`. The role's superuser flag comes from the catalog.
///
/// On any failure (unknown role, bad proof, malformed message) the client
/// is sent `ErrorResponse 28P01` and [`AuthOutcome::Rejected`] is returned;
/// the plaintext password is never seen by the server (only the proof is).
async fn authenticate<R, W>(
    reader: &mut R,
    writer: &mut W,
    db: &Arc<RwLock<Database>>,
    auth_policy: &AuthPolicy,
    user_param: Option<String>,
) -> Result<SessionContext, AuthOutcome>
where
    R: AsyncReadExt + Unpin,
    W: AsyncWriteExt + Unpin,
{
    if !auth_policy.enabled {
        // Trusted-local: run as the configured superuser without a
        // credential check (documented, never silent — see the startup
        // warning). This is the v1-compatible loopback/dev mode.
        return Ok(SessionContext::new(Role::superuser(
            auth_policy.trusted_local_user.clone(),
        )));
    }

    // Build a SCRAM authenticator backed by the persistent auth catalog.
    // Lookups read a fresh snapshot from the engine each time, so a role
    // created/altered by another connection is immediately visible.
    let store = {
        let guard = db.read().await;
        guard.auth_store()
    };
    let verifier_store = store.clone();
    let superuser_store = store.clone();
    let authenticator = ScramAuthenticator::new(
        Arc::new(move |name: &str| verifier_store.verifier_for(name)),
        Arc::new(move |name: &str| superuser_store.is_superuser(name)),
    );

    let mechanisms = authenticator.mechanisms();
    write_auth_sasl(writer, mechanisms).await?;
    writer.flush().await?;

    // Round 1: client selects a mechanism and sends client-first.
    let initial = read_sasl_initial_response(reader).await?;
    if !mechanisms.contains(&initial.mechanism.as_str()) {
        audit_login(auth_policy, user_param.as_deref(), AuditOutcome::Denied);
        reject_auth(
            writer,
            &format!("unsupported SASL mechanism '{}'", initial.mechanism),
        )
        .await?;
        return Err(AuthOutcome::Rejected);
    }

    let mut state = authenticator.begin(user_param.as_deref());
    let server_first = match authenticator.step(&mut state, &initial.initial_response) {
        AuthStep::Continue(data) => data,
        AuthStep::Fail(e) => {
            tracing::debug!(error = %e, "SCRAM round 1 failed");
            audit_login(auth_policy, user_param.as_deref(), AuditOutcome::Denied);
            reject_auth(writer, "authentication failed").await?;
            return Err(AuthOutcome::Rejected);
        }
        AuthStep::Success { .. } => {
            // SCRAM never succeeds in one round; treat as protocol error.
            reject_auth(writer, "unexpected SASL success in round 1").await?;
            return Err(AuthOutcome::Rejected);
        }
    };
    write_auth_sasl_continue(writer, &server_first).await?;
    writer.flush().await?;

    // Round 2: client sends client-final (proof); server verifies.
    let client_final = read_sasl_response(reader).await?;
    match authenticator.step(&mut state, &client_final) {
        AuthStep::Success {
            role,
            final_message,
        } => {
            if let Some(server_final) = final_message {
                write_auth_sasl_final(writer, &server_final).await?;
            }
            tracing::info!(role = %role.id, superuser = role.is_superuser, "authenticated");
            auth_policy.audit.record(
                &AuditEvent::new("auth", "login", AuditOutcome::Allowed)
                    .with_role(role.id.as_str()),
            );
            Ok(SessionContext::new(role))
        }
        AuthStep::Fail(e) => {
            tracing::debug!(error = %e, "SCRAM round 2 failed");
            audit_login(auth_policy, user_param.as_deref(), AuditOutcome::Denied);
            reject_auth(writer, "authentication failed").await?;
            Err(AuthOutcome::Rejected)
        }
        AuthStep::Continue(_) => {
            reject_auth(writer, "unexpected extra SASL round").await?;
            Err(AuthOutcome::Rejected)
        }
    }
}

/// Record a failed login. The role is the claimed startup `user` (which
/// may not exist); the generic outcome avoids confirming whether the role
/// exists.
fn audit_login(auth_policy: &AuthPolicy, claimed_user: Option<&str>, outcome: AuditOutcome) {
    let mut event = AuditEvent::new("auth", "login", outcome);
    if let Some(u) = claimed_user {
        event = event.with_role(u);
    }
    auth_policy.audit.record(&event);
}

/// Send `ErrorResponse 28P01` (invalid_password). The message is generic
/// so the server never leaks whether the role exists or the password was
/// merely wrong.
async fn reject_auth<W: AsyncWriteExt + Unpin>(writer: &mut W, _detail: &str) -> std::io::Result<()> {
    write_error_response(writer, "28P01", "password authentication failed").await?;
    writer.flush().await
}
