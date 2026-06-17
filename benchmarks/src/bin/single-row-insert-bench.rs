//! Single-row wire INSERT throughput benchmark (v2-phase1 task 10.4 / Req 7).
//!
//! Measures the effect of the statement cache + prepared-statement
//! parse-once path on single-row INSERT throughput over the real
//! PostgreSQL wire protocol against a real `galaxdb-server`:
//!
//! * **before** — the simple query protocol (`Q`), which re-parses every
//!   statement (the ~1–2 ms sqlparser cost per row the paper's 454 rows/s
//!   ceiling came from).
//! * **after** — the extended query protocol: the statement is prepared
//!   ONCE and each row is an `Execute` that binds parameters into the
//!   cached AST (no re-parse).
//!
//! Per `.kiro/steering/engineering-principles.md` §4, this prints numbers
//! when run on named hardware against `--release`; no number is published
//! in docs until it has actually been run. Run with:
//!
//! ```bash
//! cargo run --release -p galaxdb-benchmarks --bin single-row-insert-bench -- --rows 20000
//! ```

use std::time::Instant;

use clap::Parser;
use galaxdb_server::{start, ServerConfig};
use tokio_postgres::types::Type;
use tokio_postgres::NoTls;

#[derive(Parser, Debug)]
#[command(about = "Single-row wire INSERT throughput: simple (re-parse) vs prepared (parse-once)")]
struct Args {
    /// Number of single-row INSERTs per phase.
    #[arg(long, default_value_t = 20_000)]
    rows: usize,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let data_dir = tempfile::tempdir().expect("tempdir");

    let cfg = ServerConfig {
        bind_addr: "127.0.0.1:0".to_string(),
        data_dir: data_dir.path().to_string_lossy().to_string(),
        max_connections: 16,
        ..Default::default()
    };
    let (addr, _handle) = start(cfg).await.expect("server bind");
    let conn_str = format!(
        "host=127.0.0.1 port={} user=galaxdb dbname=galaxdb sslmode=disable",
        addr.port()
    );

    let (client, connection) = tokio_postgres::connect(&conn_str, NoTls)
        .await
        .expect("connect");
    tokio::spawn(async move {
        let _ = connection.await;
    });

    println!("=== GalaxDB single-row INSERT benchmark ===");
    println!("rows per phase: {}", args.rows);
    println!("(reproduce: cargo run --release -p galaxdb-benchmarks --bin single-row-insert-bench -- --rows {})", args.rows);

    // ── Phase A: simple protocol (re-parse every row) ──────────────
    client
        .simple_query("CREATE TABLE bench_simple (id INTEGER PRIMARY KEY, name TEXT)")
        .await
        .expect("create simple table");

    let start_a = Instant::now();
    for i in 0..args.rows {
        client
            .simple_query(&format!(
                "INSERT INTO bench_simple (id, name) VALUES ({i}, 'row-{i}')"
            ))
            .await
            .expect("simple insert");
    }
    let elapsed_a = start_a.elapsed();
    let tps_a = args.rows as f64 / elapsed_a.as_secs_f64();

    // ── Phase B: extended protocol (prepare once, execute many) ─────
    client
        .simple_query("CREATE TABLE bench_prepared (id INTEGER PRIMARY KEY, name TEXT)")
        .await
        .expect("create prepared table");
    let stmt = client
        .prepare_typed(
            "INSERT INTO bench_prepared (id, name) VALUES ($1, $2)",
            &[Type::INT4, Type::TEXT],
        )
        .await
        .expect("prepare insert");

    let start_b = Instant::now();
    for i in 0..args.rows {
        let id = i as i32;
        let name = format!("row-{i}");
        client.execute(&stmt, &[&id, &name]).await.expect("prepared insert");
    }
    let elapsed_b = start_b.elapsed();
    let tps_b = args.rows as f64 / elapsed_b.as_secs_f64();

    println!();
    println!("simple   (re-parse each):   {elapsed_a:?}  =>  {tps_a:.0} rows/sec");
    println!("prepared (parse-once):      {elapsed_b:?}  =>  {tps_b:.0} rows/sec");
    if tps_a > 0.0 {
        println!("speedup (prepared / simple): {:.2}x", tps_b / tps_a);
    }

    // Sanity: both tables must hold all rows (correctness, not just speed).
    let count_rows = |msgs: Vec<tokio_postgres::SimpleQueryMessage>| {
        msgs.into_iter()
            .filter(|m| matches!(m, tokio_postgres::SimpleQueryMessage::Row(_)))
            .count()
    };
    let a = count_rows(client.simple_query("SELECT id FROM bench_simple").await.unwrap());
    let b = count_rows(client.simple_query("SELECT id FROM bench_prepared").await.unwrap());
    assert_eq!(a, args.rows, "simple phase lost rows");
    assert_eq!(b, args.rows, "prepared phase lost rows");
    println!("\nverified: both phases persisted all {} rows.", args.rows);
}
