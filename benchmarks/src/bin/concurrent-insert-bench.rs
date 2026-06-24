//! Concurrent wire INSERT throughput benchmark (v2-phase1 / Req 7).
//!
//! Measures single-row INSERT throughput with N concurrent clients, all
//! hammering the same GalaxDB server simultaneously. This is the same
//! model as `pgbench -c N` — concurrent clients share WAL fsyncs through
//! the group-commit path, so throughput scales with concurrency.
//!
//! Also runs the equivalent PostgreSQL test on the same port if
//! --pg-port is supplied, for a direct side-by-side comparison.
//!
//! Reproduce:
//! ```
//! cargo run --release -p galaxdb-benchmarks --bin concurrent-insert-bench \
//!     --rows 5000 --clients 1,4,8,16
//! ```

use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use galaxdb_server::{start, ServerConfig};
use tokio::sync::Barrier;
use tokio_postgres::NoTls;

#[derive(Parser, Debug)]
#[command(about = "Concurrent wire INSERT throughput: GalaxDB vs PostgreSQL")]
struct Args {
    /// Rows each client inserts.
    #[arg(long, default_value_t = 2000)]
    rows: usize,

    /// Comma-separated list of client counts to sweep, e.g. 1,4,8,16
    #[arg(long, default_value = "1,4,8,16")]
    clients: String,

    /// Optional PostgreSQL port to run the same test against for comparison.
    /// PostgreSQL must already be running with a 'bench' database.
    #[arg(long)]
    pg_port: Option<u16>,
}

async fn run_concurrent(conn_str: &str, n_clients: usize, rows_per_client: usize) -> f64 {
    let barrier = Arc::new(Barrier::new(n_clients));
    let mut handles = Vec::new();
    let t_start = Instant::now();

    for worker in 0..n_clients {
        let cs = conn_str.to_string();
        let b = barrier.clone();
        handles.push(tokio::spawn(async move {
            let (client, conn) = tokio_postgres::connect(&cs, NoTls).await.expect("connect");
            tokio::spawn(async move { let _ = conn.await; });

            let table = format!("bench_{worker}");
            client.simple_query(&format!("DROP TABLE IF EXISTS {table}")).await.expect("drop");
            client.simple_query(&format!(
                "CREATE TABLE {table} (id INTEGER PRIMARY KEY, val TEXT)"
            )).await.expect("create");

            // Prepared statement (parse once), matching the PostgreSQL path.
            // GalaxDB advertises TEXT param oids, so bind values as strings.
            let stmt = client.prepare(&format!(
                "INSERT INTO {table} (id, val) VALUES ($1, $2)"
            )).await.expect("prepare");

            // All clients start at the same instant
            b.wait().await;

            for i in 0..rows_per_client {
                let id = i.to_string();
                let val = format!("w{worker}-r{i}");
                client.execute(&stmt, &[&id, &val]).await.expect("insert");
            }
        }));
    }

    for h in handles { h.await.expect("join"); }
    let elapsed = t_start.elapsed();
    let total = n_clients * rows_per_client;
    total as f64 / elapsed.as_secs_f64()
}

async fn run_pg_concurrent(port: u16, n_clients: usize, rows_per_client: usize) -> f64 {
    // Drop and recreate tables for a clean run
    let conn_str = format!(
        "host=127.0.0.1 port={port} user=postgres dbname=bench sslmode=disable"
    );
    let (setup, conn) = tokio_postgres::connect(&conn_str, NoTls).await.expect("pg connect");
    tokio::spawn(async move { let _ = conn.await; });
    for w in 0..n_clients {
        let _ = setup.simple_query(&format!("DROP TABLE IF EXISTS pgbench_{w}")).await;
        setup.simple_query(&format!(
            "CREATE TABLE pgbench_{w} (id INTEGER PRIMARY KEY, val TEXT)"
        )).await.expect("pg create");
    }
    drop(setup);

    let barrier = Arc::new(Barrier::new(n_clients));
    let mut handles = Vec::new();
    let t_start = Instant::now();
    for worker in 0..n_clients {
        let b = barrier.clone();
        let cs = conn_str.clone();
        handles.push(tokio::spawn(async move {
            let (client, conn) = tokio_postgres::connect(&cs, NoTls).await.expect("pg connect w");
            tokio::spawn(async move { let _ = conn.await; });
            let stmt = client.prepare(&format!(
                "INSERT INTO pgbench_{worker} (id, val) VALUES ($1, $2)"
            )).await.expect("pg prepare");
            b.wait().await;
            for i in 0..rows_per_client {
                let id = i as i32;
                let val = format!("w{worker}-r{i}");
                client.execute(&stmt, &[&id, &val]).await.expect("pg insert");
            }
        }));
    }
    for h in handles { h.await.expect("pg join"); }
    let elapsed = t_start.elapsed();
    let total = n_clients * rows_per_client;
    total as f64 / elapsed.as_secs_f64()
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let client_counts: Vec<usize> = args.clients.split(',')
        .map(|s| s.trim().parse().expect("bad client count"))
        .collect();

    let data_dir = tempfile::tempdir().expect("tempdir");
    let cfg = ServerConfig {
        bind_addr: "127.0.0.1:0".to_string(),
        data_dir: data_dir.path().to_string_lossy().to_string(),
        max_connections: 64,
        ..Default::default()
    };
    let (addr, _handle) = start(cfg).await.expect("server bind");
    let conn_str = format!(
        "host=127.0.0.1 port={} user=galaxdb dbname=galaxdb sslmode=disable",
        addr.port()
    );

    println!("=== GalaxDB concurrent INSERT benchmark ===");
    println!("rows per client: {}", args.rows);
    println!();
    println!("{:<10} {:>14}  {:>14}", "clients", "GalaxDB TPS", "PostgreSQL TPS");
    println!("{}", "-".repeat(42));

    for &n in &client_counts {
        let galaxdb_tps = run_concurrent(&conn_str, n, args.rows).await;
        let pg_tps = if let Some(port) = args.pg_port {
            run_pg_concurrent(port, n, args.rows).await
        } else {
            0.0
        };
        if args.pg_port.is_some() {
            println!("{:<10} {:>14.0}  {:>14.0}", n, galaxdb_tps, pg_tps);
        } else {
            println!("{:<10} {:>14.0}", n, galaxdb_tps);
        }
    }

    println!();
    println!("Reproduce:");
    println!("  cargo run --release -p galaxdb-benchmarks --bin concurrent-insert-bench \\");
    println!("    --rows {} --clients {}", args.rows, args.clients);
}
