//! Phase G SIFT1M benchmark binary.
//!
//! Consumes the canonical SIFT1M .fvecs / .ivecs files and emits a
//! provenance-rich JSON report covering:
//!   - commit SHA, instance type, dataset SHA256, CPU model, RAM, timestamp
//!   - HNSW build time
//!   - recall@10, mean latency, p99 latency across the ef_search sweep
//!
//! Invoked by scripts/aws-integration-run.sh on the real AWS c6id.4xlarge
//! instance. This binary produces no numbers on its own — it only
//! processes the dataset the orchestrator downloads and hash-verifies.
//!
//! Rule: every field in the output JSON is either measured locally
//! (timing, recall, /proc/*) or passed in by the orchestrator (commit,
//! instance type, dataset hash). Nothing is faked.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use serde::Serialize;

use galaxdb_vector::{HnswConfig, HnswGraph};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "galaxdb-sift-bench",
    about = "SIFT1M recall + ef_search sweep on GalaxDB HNSW (Phase G)"
)]
struct Cli {
    /// Directory containing sift_base.fvecs, sift_query.fvecs,
    /// sift_groundtruth.ivecs (the tar.gz unpacks to a `sift/` subdir).
    #[arg(long)]
    dataset: PathBuf,

    /// Output path for the provenance JSON.
    #[arg(long)]
    output: PathBuf,

    /// Commit SHA of this build. Passed in by the orchestration script.
    #[arg(long)]
    commit_sha: String,

    /// EC2 instance type (e.g. "c6id.4xlarge"). Passed in by the
    /// orchestration script — we never guess.
    #[arg(long)]
    instance_type: String,

    /// SHA256 of sift.tar.gz as verified by the orchestration script.
    #[arg(long)]
    dataset_sha256: String,

    /// UTC timestamp stamped by the orchestration script.
    #[arg(long)]
    timestamp_utc: String,

    /// HNSW M parameter (graph connectivity).
    #[arg(long, default_value_t = 16)]
    m: usize,

    /// HNSW ef_construction parameter.
    #[arg(long, default_value_t = 200)]
    ef_construction: usize,

    /// Comma-separated ef_search values to sweep.
    #[arg(long, default_value = "10,50,100,200")]
    ef_search_sweep: String,

    /// Number of queries to evaluate per ef (defaults to all 10k).
    #[arg(long, default_value_t = 10_000)]
    num_queries: usize,

    /// k in recall@k.
    #[arg(long, default_value_t = 10)]
    k: usize,
}

// ---------------------------------------------------------------------------
// JSON schema — the exact provenance contract
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct Report {
    schema_version: u32,
    commit_sha: String,
    timestamp_utc: String,
    instance: InstanceInfo,
    cpu: CpuInfo,
    ram_gb: u64,
    dataset: DatasetInfo,
    hnsw_config: HnswConfigJson,
    build: BuildInfo,
    search: SearchInfo,
}

#[derive(Serialize)]
struct InstanceInfo {
    r#type: String,
}

#[derive(Serialize)]
struct CpuInfo {
    model: String,
    cores: usize,
    arch: String,
}

#[derive(Serialize)]
struct DatasetInfo {
    name: String,
    size: usize,
    dim: usize,
    sha256: String,
    source_url: String,
}

#[derive(Serialize)]
struct HnswConfigJson {
    m: usize,
    ef_construction: usize,
}

#[derive(Serialize)]
struct BuildInfo {
    build_time_ms: u64,
    build_rate_vec_per_sec: f64,
}

#[derive(Serialize)]
struct SearchInfo {
    k: usize,
    num_queries_evaluated: usize,
    ef_search_sweep: Vec<EfPoint>,
}

#[derive(Serialize)]
struct EfPoint {
    ef: usize,
    recall_at_k: f64,
    mean_latency_us: f64,
    p99_latency_us: u64,
}

// ---------------------------------------------------------------------------
// .fvecs / .ivecs readers
//
// Format (both): for each vector, [i32 dim little-endian] followed by
// `dim` × [f32 little-endian] for .fvecs or [i32 little-endian] for .ivecs.
//
// This binary reads the whole file into memory — SIFT1M is small enough
// (base 512 MB, queries 5 MB, groundtruth 4 MB) that streaming would only
// add complexity.
// ---------------------------------------------------------------------------

fn read_fvecs(path: &Path) -> std::io::Result<Vec<Vec<f32>>> {
    let mut buf = Vec::new();
    File::open(path)?.read_to_end(&mut buf)?;
    let mut out = Vec::new();
    let mut off = 0;
    while off < buf.len() {
        if off + 4 > buf.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("truncated dim prefix in {}", path.display()),
            ));
        }
        let dim = i32::from_le_bytes(buf[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        let need = dim * 4;
        if off + need > buf.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("truncated vector payload in {}", path.display()),
            ));
        }
        let mut v = Vec::with_capacity(dim);
        for _ in 0..dim {
            v.push(f32::from_le_bytes(buf[off..off + 4].try_into().unwrap()));
            off += 4;
        }
        out.push(v);
    }
    Ok(out)
}

fn read_ivecs(path: &Path) -> std::io::Result<Vec<Vec<i32>>> {
    let mut buf = Vec::new();
    File::open(path)?.read_to_end(&mut buf)?;
    let mut out = Vec::new();
    let mut off = 0;
    while off < buf.len() {
        if off + 4 > buf.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("truncated dim prefix in {}", path.display()),
            ));
        }
        let dim = i32::from_le_bytes(buf[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        let need = dim * 4;
        if off + need > buf.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("truncated vector payload in {}", path.display()),
            ));
        }
        let mut v = Vec::with_capacity(dim);
        for _ in 0..dim {
            v.push(i32::from_le_bytes(buf[off..off + 4].try_into().unwrap()));
            off += 4;
        }
        out.push(v);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Hardware discovery (Linux only — this binary is designed to run on the
// c6id.4xlarge Ubuntu instance).
// ---------------------------------------------------------------------------

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
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();

    let base_path = cli.dataset.join("sift_base.fvecs");
    let query_path = cli.dataset.join("sift_query.fvecs");
    let gt_path = cli.dataset.join("sift_groundtruth.ivecs");

    eprintln!("[sift-bench] loading base vectors from {}", base_path.display());
    let base = read_fvecs(&base_path).expect("read sift_base.fvecs");
    eprintln!("[sift-bench] loading query vectors from {}", query_path.display());
    let queries = read_fvecs(&query_path).expect("read sift_query.fvecs");
    eprintln!("[sift-bench] loading ground truth from {}", gt_path.display());
    let gt = read_ivecs(&gt_path).expect("read sift_groundtruth.ivecs");

    assert!(!base.is_empty(), "empty sift_base.fvecs");
    assert!(!queries.is_empty(), "empty sift_query.fvecs");
    assert_eq!(
        queries.len(),
        gt.len(),
        "query count {} != ground truth count {}",
        queries.len(),
        gt.len()
    );

    let num_base = base.len();
    let dim = base[0].len();
    let num_queries = cli.num_queries.min(queries.len());
    eprintln!(
        "[sift-bench] base={} dim={} queries={} ground-truth={}",
        num_base,
        dim,
        num_queries,
        gt[0].len()
    );

    // ---- build ----
    let config = HnswConfig::new(dim)
        .with_m(cli.m)
        .with_ef_construction(cli.ef_construction)
        .with_max_elements(num_base);
    let mut hnsw = HnswGraph::new(config);
    let entries: Vec<(u64, Vec<f32>)> = base
        .into_iter()
        .enumerate()
        .map(|(i, v)| (i as u64, v))
        .collect();

    eprintln!(
        "[sift-bench] building HNSW (M={}, ef_construction={}) over {} vectors",
        cli.m, cli.ef_construction, num_base
    );
    let build_start = Instant::now();
    hnsw.insert_parallel(&entries);
    let build_elapsed = build_start.elapsed();
    let build_time_ms = build_elapsed.as_millis() as u64;
    let build_rate = num_base as f64 / build_elapsed.as_secs_f64();
    eprintln!(
        "[sift-bench] build complete: {} ms ({:.0} vec/sec)",
        build_time_ms, build_rate
    );

    // ---- ef sweep ----
    let ef_values: Vec<usize> = cli
        .ef_search_sweep
        .split(',')
        .map(|s| s.trim().parse::<usize>().expect("ef_search_sweep: non-integer"))
        .collect();

    let mut sweep = Vec::with_capacity(ef_values.len());
    for ef in ef_values {
        let effective_ef = ef.max(cli.k);
        let mut recall_sum = 0.0f64;
        let mut latencies_us: Vec<u64> = Vec::with_capacity(num_queries);

        for qi in 0..num_queries {
            let q = &queries[qi];
            let t0 = Instant::now();
            let results = hnsw.search(q, cli.k, effective_ef);
            let us = t0.elapsed().as_micros() as u64;
            latencies_us.push(us);

            // Ground truth is a list of the true nearest neighbor ids.
            // Recall@k = |gt_top_k ∩ results| / k.
            let found: std::collections::HashSet<u64> =
                results.iter().map(|(id, _)| *id).collect();
            let truth: std::collections::HashSet<u64> =
                gt[qi].iter().take(cli.k).map(|&id| id as u64).collect();
            recall_sum += found.intersection(&truth).count() as f64 / cli.k as f64;
        }

        let recall = recall_sum / num_queries as f64;
        let mean_latency_us =
            latencies_us.iter().sum::<u64>() as f64 / num_queries as f64;
        latencies_us.sort_unstable();
        let p99_idx = ((num_queries as f64) * 0.99) as usize;
        let p99_latency_us = latencies_us[p99_idx.min(num_queries - 1)];

        eprintln!(
            "[sift-bench] ef={:>4}: recall@{}={:.4}  mean_latency={:.1}µs  p99={}µs",
            ef, cli.k, recall, mean_latency_us, p99_latency_us
        );

        sweep.push(EfPoint {
            ef,
            recall_at_k: recall,
            mean_latency_us,
            p99_latency_us,
        });
    }

    // ---- assemble report ----
    let report = Report {
        schema_version: 1,
        commit_sha: cli.commit_sha,
        timestamp_utc: cli.timestamp_utc,
        instance: InstanceInfo {
            r#type: cli.instance_type,
        },
        cpu: CpuInfo {
            model: read_cpu_model(),
            cores: core_count(),
            arch: std::env::consts::ARCH.to_string(),
        },
        ram_gb: read_ram_gb(),
        dataset: DatasetInfo {
            name: "SIFT1M".to_string(),
            size: num_base,
            dim,
            sha256: cli.dataset_sha256,
            source_url: "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz".to_string(),
        },
        hnsw_config: HnswConfigJson {
            m: cli.m,
            ef_construction: cli.ef_construction,
        },
        build: BuildInfo {
            build_time_ms,
            build_rate_vec_per_sec: build_rate,
        },
        search: SearchInfo {
            k: cli.k,
            num_queries_evaluated: num_queries,
            ef_search_sweep: sweep,
        },
    };

    let json = serde_json::to_string_pretty(&report).expect("serialize report");
    std::fs::write(&cli.output, &json).expect("write output");
    eprintln!("[sift-bench] wrote {}", cli.output.display());
    println!("{}", json);
}
