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

use galaxdb_auth::{Authenticator, AuthStep, Role, ScramAuthenticator, SessionContext};
use galaxdb_embedded::Database;
use galaxdb_wire::messages::*;
use galaxdb_wire::pg_catalog;
use galaxdb_wire::tls::{self, Prologue, ReexportedTlsAcceptor as TlsAcceptor, TlsMode};

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
        let session_clone = session.clone();
        let result = tokio::task::spawn_blocking(move || {
            if is_read {
                let guard = db_clone.blocking_read();
                guard.execute_readonly_with_session(&sql_owned, Some(session_clone))
            } else {
                let mut guard = db_clone.blocking_write();
                guard.execute_with_session(&sql_owned, Some(session_clone))
            }
        })
        .await
        .map_err(|e| {
            std::io::Error::other(format!("worker panic: {e}"))
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
                // Render the typed engine error to its PostgreSQL SQLSTATE
                // so standard clients classify it correctly — e.g. a
                // failed authorization surfaces as `42501`
                // (insufficient_privilege), not a generic syntax error.
                write_error_response(&mut writer, e.sqlstate(), &format!("{}", e)).await?;
            }
        }

        write_ready_for_query(&mut writer, b'I').await?;
        writer.flush().await?;
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
            Ok(SessionContext::new(role))
        }
        AuthStep::Fail(e) => {
            tracing::debug!(error = %e, "SCRAM round 2 failed");
            reject_auth(writer, "authentication failed").await?;
            Err(AuthOutcome::Rejected)
        }
        AuthStep::Continue(_) => {
            reject_auth(writer, "unexpected extra SASL round").await?;
            Err(AuthOutcome::Rejected)
        }
    }
}

/// Send `ErrorResponse 28P01` (invalid_password). The message is generic
/// so the server never leaks whether the role exists or the password was
/// merely wrong.
async fn reject_auth<W: AsyncWriteExt + Unpin>(writer: &mut W, _detail: &str) -> std::io::Result<()> {
    write_error_response(writer, "28P01", "password authentication failed").await?;
    writer.flush().await
}
