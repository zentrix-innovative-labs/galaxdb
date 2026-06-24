//! GalaxDB Server — standalone server accepting PostgreSQL wire protocol connections.
//!
//! Usage:
//!   galaxdb-server                          # listen on 0.0.0.0:5433
//!   galaxdb-server --port 5434              # custom port
//!   galaxdb-server --data-dir /tmp/galaxdb  # custom data directory
//!   galaxdb-server --sidecar /path/to/galaxdb-sidecar --model sentence-transformers/all-MiniLM-L6-v2
//!   galaxdb-server --observe-port 9090      # HTTP /health + /metrics port (default 9090)
//!
//! The accept loop + connection handler live in [`galaxdb_server`] (lib.rs)
//! so integration tests can drive a real TCP listener against a temp dir.
//!
//! Task 40.1: tokio main, wire protocol + HTTP observability, sidecar spawn,
//! graceful shutdown on SIGTERM/SIGINT.

use galaxdb_server::{start, ServerConfig};

#[tokio::main]
async fn main() {
    // Task 38.4: structured JSON logging via tracing-subscriber.
    // GALAXDB_LOG_LEVEL env var controls the level (default: info).
    galaxdb_observe::init_logging();

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

    // Optional sidecar binary + model for embedding support.
    let sidecar_binary = std::env::args()
        .position(|a| a == "--sidecar")
        .and_then(|i| std::env::args().nth(i + 1));
    let model_id = std::env::args()
        .position(|a| a == "--model")
        .and_then(|i| std::env::args().nth(i + 1));

    // HTTP observability port (task 40.1 / task 38.1-38.2).
    let observe_port = std::env::args()
        .position(|a| a == "--observe-port")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(9090);

    // Task 6 (Req 1): authentication mode. SCRAM-SHA-256 auth is enabled
    // with `--auth` or `GALAXDB_AUTH=1`. When enabled the server provisions
    // the initial superuser from GALAXDB_INITIAL_SUPERUSER[_PASSWORD] on a
    // fresh catalog and refuses to start if neither is set (never ships a
    // default password). Without it the server runs in trusted-local mode
    // (v1-compatible) and logs a warning that auth is disabled.
    let auth_enabled = std::env::args().any(|a| a == "--auth")
        || matches!(std::env::var("GALAXDB_AUTH").as_deref(), Ok("1") | Ok("true"));
    let trusted_local_user = std::env::var("GALAXDB_TRUSTED_LOCAL_USER")
        .unwrap_or_else(|_| "galaxdb".to_string());

    // Task 7 (Req 2): TLS configuration. Mode comes from --tls-mode or
    // GALAXDB_TLS_MODE (disable|allow|require, default disable); cert and
    // key paths from --tls-cert/--tls-key or GALAXDB_TLS_CERT/_KEY.
    let tls_mode_str = std::env::args()
        .position(|a| a == "--tls-mode")
        .and_then(|i| std::env::args().nth(i + 1))
        .or_else(|| std::env::var("GALAXDB_TLS_MODE").ok())
        .unwrap_or_else(|| "disable".to_string());
    let tls_mode = galaxdb_wire::tls::TlsMode::parse(&tls_mode_str)
        .unwrap_or_else(|e| panic!("invalid --tls-mode: {e}"));
    let tls_cert_path = std::env::args()
        .position(|a| a == "--tls-cert")
        .and_then(|i| std::env::args().nth(i + 1))
        .or_else(|| std::env::var("GALAXDB_TLS_CERT").ok());
    let tls_key_path = std::env::args()
        .position(|a| a == "--tls-key")
        .and_then(|i| std::env::args().nth(i + 1))
        .or_else(|| std::env::var("GALAXDB_TLS_KEY").ok());

    // Task 4 (Req 4): optional JSONL security audit log.
    let audit_log_path = std::env::args()
        .position(|a| a == "--audit-log")
        .and_then(|i| std::env::args().nth(i + 1))
        .or_else(|| std::env::var("GALAXDB_AUDIT_LOG").ok());

    // Task 15 (Req 12): auto-tune is on by default. Operators can disable it
    // with GALAXDB_AUTOTUNE=off, or override any single derived value with
    // GALAXDB_BUFFER_POOL_BYTES / GALAXDB_MEMTABLE_BYTES /
    // GALAXDB_COMPACTION_CONCURRENCY (an explicit value always wins, AC2).
    let auto_tune = {
        let enabled = std::env::var("GALAXDB_AUTOTUNE")
            .map(|v| !matches!(v.trim().to_ascii_lowercase().as_str(), "off" | "0" | "false"))
            .unwrap_or(true);
        let parse_u64 = |k: &str| std::env::var(k).ok().and_then(|v| v.trim().parse::<u64>().ok());
        let parse_usize =
            |k: &str| std::env::var(k).ok().and_then(|v| v.trim().parse::<usize>().ok());
        galaxdb_common::AutoTuneConfig {
            enabled,
            buffer_pool_bytes: parse_u64("GALAXDB_BUFFER_POOL_BYTES"),
            memtable_size_bytes: parse_u64("GALAXDB_MEMTABLE_BYTES"),
            compaction_concurrency: parse_usize("GALAXDB_COMPACTION_CONCURRENCY"),
        }
    };

    let cfg = ServerConfig {
        bind_addr: format!("0.0.0.0:{port}"),
        data_dir,
        max_connections: 1000,
        sidecar_binary,
        model_id,
        auth_enabled,
        trusted_local_user,
        // Read from env in `start()` via resolve_initial_superuser; leave
        // None here so the password never sits in an argv-derived struct.
        initial_superuser: None,
        tls_mode,
        tls_cert_path,
        tls_key_path,
        audit_log_path,
        auto_tune,
    };

    // Task 40.1: start the HTTP observability server (/health + /metrics)
    // alongside the wire-protocol server. Both run concurrently on the
    // same tokio runtime.
    let observe_cfg = galaxdb_observe::ObserveConfig {
        bind_addr: format!("0.0.0.0:{observe_port}"),
    };
    let (observe_addr, _observe_handle) = galaxdb_observe::start_http(observe_cfg)
        .await
        .expect("failed to bind HTTP observability server");
    tracing::info!(addr = %observe_addr, "HTTP observability server listening (/health, /metrics)");
    eprintln!("GalaxDB observability listening on {observe_addr}");

    let (addr, handle) = start(cfg).await.expect("failed to bind");
    tracing::info!(addr = %addr, "GalaxDB wire-protocol server listening");
    eprintln!("GalaxDB server listening on {addr}");

    // Task 40.1: graceful shutdown on SIGTERM / SIGINT.
    let shutdown = async {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};
            let mut sigterm = signal(SignalKind::terminate())
                .expect("failed to install SIGTERM handler");
            let mut sigint = signal(SignalKind::interrupt())
                .expect("failed to install SIGINT handler");
            tokio::select! {
                _ = sigterm.recv() => tracing::info!("received SIGTERM"),
                _ = sigint.recv()  => tracing::info!("received SIGINT"),
            }
        }
        #[cfg(not(unix))]
        {
            tokio::signal::ctrl_c()
                .await
                .expect("failed to install Ctrl-C handler");
            tracing::info!("received Ctrl-C");
        }
    };

    tokio::select! {
        _ = handle => {
            tracing::warn!("accept loop exited unexpectedly");
        }
        _ = shutdown => {
            tracing::info!("shutting down gracefully");
            // Give in-flight connections up to 5 s to finish.
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        }
    }

    tracing::info!("GalaxDB server stopped");
}
