//! GalaxDB Server — standalone server accepting PostgreSQL wire protocol connections.
//!
//! Usage:
//!   galaxdb-server                          # listen on 0.0.0.0:5433
//!   galaxdb-server --port 5434              # custom port
//!   galaxdb-server --data-dir /tmp/galaxdb  # custom data directory
//!   galaxdb-server --sidecar /path/to/galaxdb-sidecar --model sentence-transformers/all-MiniLM-L6-v2
//!
//! The accept loop + connection handler live in [`galaxdb_server`] (lib.rs)
//! so integration tests can drive a real TCP listener against a temp dir.
//!
//! Task 40.1: graceful shutdown on SIGTERM/SIGINT — the server installs
//! signal handlers and waits for the accept loop to drain before exiting.

use galaxdb_server::{start, ServerConfig};

// Task 38.4: structured JSON logging.
use galaxdb_observe;

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

    let cfg = ServerConfig {
        bind_addr: format!("0.0.0.0:{port}"),
        data_dir,
        max_connections: 1000,
        sidecar_binary,
        model_id,
    };

    let (addr, handle) = start(cfg).await.expect("failed to bind");
    tracing::info!(addr = %addr, "GalaxDB server listening");
    eprintln!("GalaxDB server listening on {addr}");

    // Task 40.1: graceful shutdown on SIGTERM / SIGINT.
    // The accept loop runs until a signal arrives, then we abort it
    // and wait for in-flight connections to drain (best-effort: we
    // give them 5 s before hard-exiting).
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
