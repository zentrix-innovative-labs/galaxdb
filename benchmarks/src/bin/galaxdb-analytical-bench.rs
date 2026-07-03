//! Analytical query benchmark (HTAP task 27).
//!
//! Exercises GalaxDB's relational/analytical query path (the DataFusion
//! backend behind `galaxdb-query`, over real SST-backed columnar storage) on
//! two workload shapes:
//!   * ClickBench-style single-table aggregation (COUNT, GROUP BY, filtered
//!     SUM, top-N).
//!   * TPC-H-style star join + group-by aggregation across two tables.
//!
//! The dataset is generated **deterministically** from a fixed seed, so the
//! run is fully reproducible from the named command + parameters (row count,
//! seed). It is written through the normal `CREATE TABLE` + `INSERT` path and
//! flushed to real SSTs, so queries hit the same columnar scan the server
//! uses — no spike MemTable, no synthetic in-memory shortcut.
//!
//! HONESTY (engineering-principles §4): this is a **synthetic** workload for
//! measuring GalaxDB's own analytical throughput and tracking regressions. It
//! is deliberately NOT labelled "ClickBench" or "TPC-H": those are specific
//! public datasets, and their official numbers require loading those exact
//! datasets. The numbers here are reproducible from this binary's seed and
//! must not be compared against other systems' ClickBench/TPC-H results.
//! Every field in the output JSON is either measured locally or passed in by
//! the orchestrator (commit, instance type, timestamp); nothing is faked.

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;

use galaxdb_embedded::{Database, QueryResult};

#[derive(Parser, Debug)]
#[command(
    name = "galaxdb-analytical-bench",
    about = "Analytical (aggregation + join) benchmark on GalaxDB (HTAP task 27)"
)]
struct Cli {
    /// Output path for the provenance JSON.
    #[arg(long)]
    output: PathBuf,

    /// Commit SHA of this build. Passed in by the orchestration script.
    #[arg(long, default_value = "unknown")]
    commit_sha: String,

    /// EC2 instance type (e.g. "c6id.4xlarge"). Passed in by the orchestrator.
    #[arg(long, default_value = "local")]
    instance_type: String,

    /// UTC timestamp stamped by the orchestration script.
    #[arg(long, default_value = "unknown")]
    timestamp_utc: String,

    /// Number of fact-table rows to generate.
    #[arg(long, default_value_t = 1_000_000)]
    rows: usize,

    /// Deterministic RNG seed (part of the dataset's reproducible identity).
    #[arg(long, default_value_t = 0x6761_6c61_7864_62u64)]
    seed: u64,

    /// Iterations per query; the reported latency is the min + median.
    #[arg(long, default_value_t = 5)]
    iterations: usize,
}

#[derive(Serialize)]
struct Report {
    schema_version: u32,
    commit_sha: String,
    timestamp_utc: String,
    instance: Instance,
    cpu: Cpu,
    ram_gb: u64,
    dataset: Dataset,
    load: Load,
    queries: Vec<QueryResultJson>,
}

#[derive(Serialize)]
struct Instance {
    r#type: String,
}

#[derive(Serialize)]
struct Cpu {
    model: String,
    cores: usize,
    arch: String,
}

#[derive(Serialize)]
struct Dataset {
    /// Deliberately NOT "clickbench"/"tpch": a deterministic synthetic
    /// workload, reproducible from (generator, rows, seed).
    name: String,
    generator: String,
    fact_rows: usize,
    dim_rows: usize,
    seed: u64,
}

#[derive(Serialize)]
struct Load {
    load_ms: u128,
    flush_ms: u128,
}

#[derive(Serialize)]
struct QueryResultJson {
    id: String,
    shape: String,
    sql: String,
    rows_returned: usize,
    iterations: usize,
    min_ms: f64,
    median_ms: f64,
}

const N_REGIONS: usize = 8;
const N_USERS: usize = 10_000;
const REGION_NAMES: [&str; N_REGIONS] = [
    "north", "south", "east", "west", "central", "alpine", "coastal", "delta",
];

fn read_cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|s| s.trim().to_string())
        })
        .unwrap_or_else(|| "unknown (not on Linux)".to_string())
}

fn read_ram_gb() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("MemTotal"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|s| s.parse::<u64>().ok())
                .map(|kb| kb / (1024 * 1024))
        })
        .unwrap_or(0)
}

fn core_count() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0)
}

/// Row count of a `SELECT` result (analytical queries return `Rows`).
fn row_count(db: &mut Database, sql: &str) -> usize {
    match db.execute(sql).expect("query failed") {
        QueryResult::Rows(rows) => rows.len(),
        other => panic!("expected Rows from {sql}, got {other:?}"),
    }
}

/// Time `sql` over `iters` runs; return `(rows, min_ms, median_ms)`.
fn time_query(db: &mut Database, sql: &str, iters: usize) -> (usize, f64, f64) {
    let mut samples = Vec::with_capacity(iters);
    let mut rows = 0;
    for _ in 0..iters {
        let t0 = Instant::now();
        rows = row_count(db, sql);
        samples.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = samples[0];
    let median = samples[samples.len() / 2];
    (rows, min, median)
}

fn main() {
    let cli = Cli::parse();
    eprintln!(
        "[analytical-bench] generating {} fact rows (seed={:#x})",
        cli.rows, cli.seed
    );

    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path().join("db").to_str().unwrap()).expect("open db");

    // Schema: a wide fact table `events` + a small dimension table `regions`.
    db.execute(
        "CREATE TABLE events (id INT PRIMARY KEY, user_id INT, region TEXT, \
         device TEXT, amount INT, hour INT)",
    )
    .unwrap();
    db.execute("CREATE TABLE regions (code TEXT PRIMARY KEY, name TEXT, tier INT)")
        .unwrap();
    for (i, r) in REGION_NAMES.iter().enumerate() {
        db.execute(&format!(
            "INSERT INTO regions (code, name, tier) VALUES ('{r}', 'Region {r}', {})",
            i % 3
        ))
        .unwrap();
    }

    // Deterministic fact-table generation, loaded in multi-row INSERT batches.
    let devices = ["mobile", "desktop", "tablet"];
    let mut rng = SmallRng::seed_from_u64(cli.seed);
    let load_start = Instant::now();
    const BATCH: usize = 500;
    let mut id = 0usize;
    while id < cli.rows {
        let end = (id + BATCH).min(cli.rows);
        let mut values = Vec::with_capacity(end - id);
        for row_id in id..end {
            let user_id = rng.gen_range(0..N_USERS);
            let region = REGION_NAMES[rng.gen_range(0..N_REGIONS)];
            let device = devices[rng.gen_range(0..devices.len())];
            let amount = rng.gen_range(1..1000);
            let hour = rng.gen_range(0..24);
            values.push(format!(
                "({row_id}, {user_id}, '{region}', '{device}', {amount}, {hour})"
            ));
        }
        db.execute(&format!(
            "INSERT INTO events (id, user_id, region, device, amount, hour) VALUES {}",
            values.join(", ")
        ))
        .unwrap();
        id = end;
    }
    let load_ms = load_start.elapsed().as_millis();

    let flush_start = Instant::now();
    db.flush().expect("flush");
    let flush_ms = flush_start.elapsed().as_millis();
    eprintln!("[analytical-bench] loaded + flushed in {load_ms}+{flush_ms} ms; running queries");

    // Workload: ClickBench-style single-table aggregation + a TPC-H-style
    // star join. All route through the DataFusion analytical path.
    let workload: &[(&str, &str, &str)] = &[
        ("q1_count", "single-table aggregate", "SELECT COUNT(*) AS n FROM events"),
        (
            "q2_group_by_region",
            "single-table GROUP BY",
            "SELECT region, COUNT(*) AS n FROM events GROUP BY region ORDER BY n DESC",
        ),
        (
            "q3_filtered_sum",
            "single-table filtered aggregate",
            "SELECT SUM(amount) AS total FROM events WHERE amount > 500",
        ),
        (
            "q4_top_users",
            "single-table top-N",
            "SELECT user_id, SUM(amount) AS spend FROM events \
             GROUP BY user_id ORDER BY spend DESC LIMIT 10",
        ),
        (
            "q5_group_by_two_dims",
            "single-table multi-key GROUP BY",
            "SELECT region, device, COUNT(*) AS n FROM events \
             GROUP BY region, device ORDER BY region, device",
        ),
        (
            "q6_star_join",
            "two-table star join + aggregate",
            "SELECT r.name, COUNT(*) AS n, SUM(e.amount) AS revenue \
             FROM events e JOIN regions r ON e.region = r.code \
             GROUP BY r.name ORDER BY revenue DESC",
        ),
    ];

    let queries: Vec<QueryResultJson> = workload
        .iter()
        .map(|(id, shape, sql)| {
            let (rows, min_ms, median_ms) = time_query(&mut db, sql, cli.iterations);
            eprintln!("[analytical-bench] {id:24} rows={rows:<6} min={min_ms:.2}ms median={median_ms:.2}ms");
            QueryResultJson {
                id: id.to_string(),
                shape: shape.to_string(),
                sql: sql.to_string(),
                rows_returned: rows,
                iterations: cli.iterations,
                min_ms,
                median_ms,
            }
        })
        .collect();

    let report = Report {
        schema_version: 1,
        commit_sha: cli.commit_sha,
        timestamp_utc: cli.timestamp_utc,
        instance: Instance { r#type: cli.instance_type },
        cpu: Cpu {
            model: read_cpu_model(),
            cores: core_count(),
            arch: std::env::consts::ARCH.to_string(),
        },
        ram_gb: read_ram_gb(),
        dataset: Dataset {
            name: "synthetic-analytical-v1".to_string(),
            generator: "galaxdb-analytical-bench deterministic SmallRng".to_string(),
            fact_rows: cli.rows,
            dim_rows: N_REGIONS,
            seed: cli.seed,
        },
        load: Load { load_ms, flush_ms },
        queries,
    };

    let json = serde_json::to_string_pretty(&report).expect("serialize report");
    std::fs::write(&cli.output, &json).expect("write output");
    eprintln!("[analytical-bench] wrote {}", cli.output.display());
}
