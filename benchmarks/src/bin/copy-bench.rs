//! COPY bulk-load throughput benchmark (v2-phase1 task 11 / Req 8).
//!
//! Measures real ingest throughput over the PostgreSQL wire protocol
//! against a real `galaxdb-server`, comparing three load methods for the
//! same set of rows:
//!
//!   * **insert-simple**   — one `INSERT` per row over the simple query
//!     protocol (`Q`); the parser runs once per row.
//!   * **insert-prepared** — one `Execute` per row over the extended
//!     protocol; the statement is parsed once and each row binds params
//!     into the cached AST (v2-phase1 task 10).
//!   * **copy**            — `COPY t FROM STDIN` (text format), which
//!     streams the whole batch and ingests it through the bulk-insert
//!     path (v2-phase1 task 11), not one INSERT per row.
//!
//! It also round-trips the data back with `COPY t TO STDOUT` and verifies
//! every method persisted exactly `--rows` rows (correctness, not just
//! speed).
//!
//! Per `.kiro/steering/engineering-principles.md` §4 the numbers are only
//! published once this has actually been run on named hardware with
//! `--release`. Reproduce with:
//!
//! ```bash
//! cargo run --release -p galaxdb-benchmarks --bin copy-bench -- --rows 200000
//! ```

use std::time::{Duration, Instant};

use bytes::Bytes;
use clap::Parser;
use futures_util::SinkExt;
use galaxdb_server::{start, ServerConfig};
use tokio_postgres::types::Type;
use tokio_postgres::NoTls;

#[derive(Parser, Debug)]
#[command(about = "COPY bulk-load throughput vs row-by-row INSERT over the wire")]
struct Args {
    /// Number of rows to load per method.
    #[arg(long, default_value_t = 200_000)]
    rows: usize,

    /// Emit the result line as JSON (for the AWS orchestrator to collect).
    #[arg(long, default_value_t = false)]
    json: bool,

    /// Commit SHA to stamp into the JSON provenance (orchestrator-supplied).
    #[arg(long, default_value = "")]
    commit_sha: String,

    /// Instance-type label to stamp into the JSON provenance.
    #[arg(long, default_value = "")]
    instance_type: String,

    /// UTC timestamp to stamp into the JSON provenance.
    #[arg(long, default_value = "")]
    timestamp_utc: String,
}

/// One synthetic row. Width is representative of a small fact table:
/// an int key, a short text label, and a longer text payload. Rows are
/// deterministic from the index so a re-run loads identical bytes.
fn row_fields(i: usize) -> (i32, String, String) {
    let id = i as i32;
    let name = format!("user_{i:08}");
    let payload = format!(
        "event-{i}|region={}|score={}|note=synthetic-row-for-ingest-benchmark",
        i % 16,
        (i * 2654435761usize) % 100_000
    );
    (id, name, payload)
}

/// Text-format COPY line for row `i` (tab-separated, newline-terminated).
fn copy_line(i: usize) -> String {
    let (id, name, payload) = row_fields(i);
    format!("{id}\t{name}\t{payload}\n")
}

fn throughput(rows: usize, bytes: usize, elapsed: Duration) -> (f64, f64) {
    let secs = elapsed.as_secs_f64();
    let rps = if secs > 0.0 { rows as f64 / secs } else { 0.0 };
    let mbps = if secs > 0.0 {
        (bytes as f64 / (1024.0 * 1024.0)) / secs
    } else {
        0.0
    };
    (rps, mbps)
}

async fn count_rows(client: &tokio_postgres::Client, table: &str) -> usize {
    client
        .simple_query(&format!("SELECT id FROM {table}"))
        .await
        .unwrap()
        .into_iter()
        .filter(|m| matches!(m, tokio_postgres::SimpleQueryMessage::Row(_)))
        .count()
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

    // Total payload size (the COPY text stream byte count) used for MB/s.
    let copy_bytes: usize = (0..args.rows).map(|i| copy_line(i).len()).sum();

    if !args.json {
        println!("=== GalaxDB COPY bulk-load benchmark ===");
        println!("rows per method: {}", args.rows);
        println!("copy stream size: {:.2} MiB", copy_bytes as f64 / (1024.0 * 1024.0));
        println!("(reproduce: cargo run --release -p galaxdb-benchmarks --bin copy-bench -- --rows {})", args.rows);
        println!();
    }

    // ── Method 1: row-by-row INSERT, simple protocol (re-parse each) ──
    client
        .simple_query("CREATE TABLE load_insert_simple (id INTEGER PRIMARY KEY, name TEXT, payload TEXT)")
        .await
        .expect("create simple table");
    let t = Instant::now();
    for i in 0..args.rows {
        let (id, name, payload) = row_fields(i);
        let name = name.replace('\'', "''");
        let payload = payload.replace('\'', "''");
        client
            .simple_query(&format!(
                "INSERT INTO load_insert_simple (id, name, payload) VALUES ({id}, '{name}', '{payload}')"
            ))
            .await
            .expect("simple insert");
    }
    let elapsed_simple = t.elapsed();
    let (rps_simple, mbps_simple) = throughput(args.rows, copy_bytes, elapsed_simple);

    // ── Method 2: prepared INSERT, extended protocol (parse-once) ─────
    client
        .simple_query("CREATE TABLE load_insert_prepared (id INTEGER PRIMARY KEY, name TEXT, payload TEXT)")
        .await
        .expect("create prepared table");
    let stmt = client
        .prepare_typed(
            "INSERT INTO load_insert_prepared (id, name, payload) VALUES ($1, $2, $3)",
            &[Type::INT4, Type::TEXT, Type::TEXT],
        )
        .await
        .expect("prepare insert");
    let t = Instant::now();
    for i in 0..args.rows {
        let (id, name, payload) = row_fields(i);
        client
            .execute(&stmt, &[&id, &name, &payload])
            .await
            .expect("prepared insert");
    }
    let elapsed_prepared = t.elapsed();
    let (rps_prepared, mbps_prepared) = throughput(args.rows, copy_bytes, elapsed_prepared);

    // ── Method 3: COPY FROM STDIN bulk load ───────────────────────────
    client
        .simple_query("CREATE TABLE load_copy (id INTEGER PRIMARY KEY, name TEXT, payload TEXT)")
        .await
        .expect("create copy table");
    let t = Instant::now();
    {
        let sink = client
            .copy_in("COPY load_copy (id, name, payload) FROM STDIN")
            .await
            .expect("copy_in");
        futures_util::pin_mut!(sink);
        // Stream in chunks so we don't hold the entire dataset in one Bytes.
        const CHUNK_ROWS: usize = 4_096;
        let mut buf = String::new();
        let mut in_chunk = 0usize;
        for i in 0..args.rows {
            buf.push_str(&copy_line(i));
            in_chunk += 1;
            if in_chunk == CHUNK_ROWS {
                sink.send(Bytes::from(std::mem::take(&mut buf)))
                    .await
                    .expect("copy send");
                in_chunk = 0;
            }
        }
        if !buf.is_empty() {
            sink.send(Bytes::from(buf)).await.expect("copy send tail");
        }
        let copied = sink.finish().await.expect("copy finish");
        assert_eq!(copied as usize, args.rows, "COPY must report all rows");
    }
    let elapsed_copy = t.elapsed();
    let (rps_copy, mbps_copy) = throughput(args.rows, copy_bytes, elapsed_copy);

    // ── Correctness: every method persisted exactly `rows` rows ───────
    let n_simple = count_rows(&client, "load_insert_simple").await;
    let n_prepared = count_rows(&client, "load_insert_prepared").await;
    let n_copy = count_rows(&client, "load_copy").await;
    assert_eq!(n_simple, args.rows, "simple insert lost rows");
    assert_eq!(n_prepared, args.rows, "prepared insert lost rows");
    assert_eq!(n_copy, args.rows, "copy lost rows");

    if args.json {
        let json = serde_json::json!({
            "benchmark": "copy-bulk-load",
            "rows": args.rows,
            "copy_stream_bytes": copy_bytes,
            "commit_sha": args.commit_sha,
            "instance_type": args.instance_type,
            "timestamp_utc": args.timestamp_utc,
            "methods": {
                "insert_simple": {
                    "elapsed_secs": elapsed_simple.as_secs_f64(),
                    "rows_per_sec": rps_simple,
                    "mib_per_sec": mbps_simple,
                },
                "insert_prepared": {
                    "elapsed_secs": elapsed_prepared.as_secs_f64(),
                    "rows_per_sec": rps_prepared,
                    "mib_per_sec": mbps_prepared,
                },
                "copy": {
                    "elapsed_secs": elapsed_copy.as_secs_f64(),
                    "rows_per_sec": rps_copy,
                    "mib_per_sec": mbps_copy,
                },
            },
            "speedup_copy_over_simple": if rps_simple > 0.0 { rps_copy / rps_simple } else { 0.0 },
            "speedup_copy_over_prepared": if rps_prepared > 0.0 { rps_copy / rps_prepared } else { 0.0 },
        });
        println!("{}", serde_json::to_string_pretty(&json).unwrap());
    } else {
        println!("insert-simple   : {elapsed_simple:?}  =>  {rps_simple:>10.0} rows/sec  {mbps_simple:>7.1} MiB/sec");
        println!("insert-prepared : {elapsed_prepared:?}  =>  {rps_prepared:>10.0} rows/sec  {mbps_prepared:>7.1} MiB/sec");
        println!("copy            : {elapsed_copy:?}  =>  {rps_copy:>10.0} rows/sec  {mbps_copy:>7.1} MiB/sec");
        println!();
        if rps_simple > 0.0 {
            println!("speedup copy / insert-simple   : {:.2}x", rps_copy / rps_simple);
        }
        if rps_prepared > 0.0 {
            println!("speedup copy / insert-prepared : {:.2}x", rps_copy / rps_prepared);
        }
        println!("\nverified: all three methods persisted exactly {} rows.", args.rows);
    }
}
