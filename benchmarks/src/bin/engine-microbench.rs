//! Storage-engine write-path microbenchmark (diagnostic).
//!
//! Isolates the engine from the wire protocol and the SQL layer to answer
//! one question with facts: where does write time go, and how does per-row
//! cost scale with the number of rows already resident?
//!
//! Measures, against a real `galaxdb_storage::Engine` on the real data dir:
//!   * `put_sync` (one WAL fsync per row) — latency per row + scaling.
//!   * `put_batch_sync` (one WAL fsync per call) — per-row cost vs batch size.
//!
//! Reproduce:
//! ```bash
//! cargo run --release -p galaxdb-benchmarks --bin engine-microbench -- --max 200000
//! ```

use std::time::Instant;

use clap::Parser;
use galaxdb_storage::engine::{Engine, EngineConfig};

#[derive(Parser, Debug)]
struct Args {
    /// Largest row count to test (powers stepped from 1k).
    #[arg(long, default_value_t = 200_000)]
    max: usize,
    /// WAL group-commit window in ms (engine default is 10).
    #[arg(long, default_value_t = 10)]
    group_commit_ms: u64,
}

fn row(i: usize) -> (Vec<u8>, Vec<u8>) {
    let key = format!("bench:{i:012}").into_bytes();
    let val = format!("user_{i:08}|event-{i}|region={}|payload-for-microbench", i % 16)
        .into_bytes();
    (key, val)
}

fn new_engine(dir: &std::path::Path, group_commit_ms: u64) -> Engine {
    let cfg = EngineConfig {
        data_dir: dir.to_path_buf(),
        wal_group_commit_ms: group_commit_ms,
        ..Default::default()
    };
    Engine::new(cfg).expect("engine open")
}

fn main() {
    let args = Args::parse();

    // --- 1. put_sync per-row latency, measured in windows as the engine
    //        grows, so we can see whether per-op cost is flat or rising. ---
    println!("=== put_sync (one fsync per row), group_commit_ms={} ===", args.group_commit_ms);
    {
        let dir = tempfile::tempdir().unwrap();
        let engine = new_engine(dir.path(), args.group_commit_ms);
        let window = 2000usize;
        let mut i = 0usize;
        while i < args.max.min(20_000) {
            let start = Instant::now();
            for _ in 0..window {
                let (k, v) = row(i);
                engine.put_sync(k, v).expect("put_sync");
                i += 1;
            }
            let el = start.elapsed();
            let per = el.as_secs_f64() * 1e6 / window as f64;
            println!(
                "  rows {:>7}..{:>7}: {:>8.0} rows/s, {:>7.1} us/row",
                i - window,
                i,
                window as f64 / el.as_secs_f64(),
                per
            );
        }
    }

    // --- 2. put_batch_sync: per-row cost as a function of how many rows are
    //        ALREADY resident (fresh engine, single growing memtable). One
    //        batch call per window so the WAL fsync is amortised; any rise in
    //        us/row is the in-memory (memtable + ART) path, not the fsync. ---
    println!("\n=== put_batch_sync (one fsync per 2000-row call), cumulative ===");
    {
        let dir = tempfile::tempdir().unwrap();
        let engine = new_engine(dir.path(), args.group_commit_ms);
        let window = 2000usize;
        let mut i = 0usize;
        while i < args.max {
            let batch: Vec<(Vec<u8>, Vec<u8>)> = (i..i + window).map(row).collect();
            let start = Instant::now();
            engine.put_batch_sync(&batch).expect("put_batch_sync");
            let el = start.elapsed();
            i += window;
            let per = el.as_secs_f64() * 1e6 / window as f64;
            println!(
                "  resident {:>8}: {:>9.0} rows/s, {:>7.1} us/row",
                i,
                window as f64 / el.as_secs_f64(),
                per
            );
        }
    }

    // --- 3. Single large batch in ONE call (no cumulative prior state) to
    //        separate batch-size effects from resident-size effects. ---
    println!("\n=== put_batch_sync single call, varying batch size (fresh engine each) ===");
    for &n in &[1000usize, 2000, 5000, 10000, 50000, 100000] {
        if n > args.max {
            break;
        }
        let dir = tempfile::tempdir().unwrap();
        let engine = new_engine(dir.path(), args.group_commit_ms);
        let batch: Vec<(Vec<u8>, Vec<u8>)> = (0..n).map(row).collect();
        let start = Instant::now();
        engine.put_batch_sync(&batch).expect("put_batch_sync");
        let el = start.elapsed();
        let per = el.as_secs_f64() * 1e6 / n as f64;
        println!(
            "  batch {:>7}: {:>9.0} rows/s, {:>7.1} us/row, total {:?}",
            n,
            n as f64 / el.as_secs_f64(),
            per,
            el
        );
    }
}
