//! GalaxDB Server — standalone server accepting PostgreSQL wire protocol connections.
//!
//! Usage:
//!   galaxdb-server                          # listen on 0.0.0.0:5433
//!   galaxdb-server --port 5434              # custom port
//!   galaxdb-server --data-dir /tmp/galaxdb  # custom data directory
//!
//! The accept loop + connection handler live in [`galaxdb_server`] (lib.rs)
//! so integration tests can drive a real TCP listener against a temp dir.

use galaxdb_server::{start, ServerConfig};

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

    let cfg = ServerConfig {
        bind_addr: format!("0.0.0.0:{port}"),
        data_dir,
        max_connections: 1000,
    };

    let (addr, handle) = start(cfg).await.expect("failed to bind");
    eprintln!("GalaxDB server listening on {addr}");

    // Block until the accept loop exits (it won't, short of panic).
    let _ = handle.await;
}
