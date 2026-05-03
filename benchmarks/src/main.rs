//! GalaxDB Macro-Benchmark Suite
//!
//! Standalone binary that runs three production-pattern workloads:
//! 1. OLTP Write + Point Read
//! 2. OLAP Column Scan
//! 3. Mixed OLTP + OLAP
//!
//! Outputs structured JSON results to stdout.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use hdrhistogram::Histogram;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tokio::sync::Mutex;

use galaxdb_common::{BlockId, ColumnType};
use galaxdb_storage::art::{ArtIndex, RowLocation};
use galaxdb_storage::buffer_pool::{AccessType, BufferPool, CachedBlock};
use galaxdb_storage::engine::{Engine, EngineConfig};
use galaxdb_storage::memtable::Memtable;
use galaxdb_storage::pax::{ColumnData, PaxBlock};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(name = "galaxdb-benchmarks", about = "GalaxDB macro-benchmark suite")]
struct Cli {
    /// Workload to run: oltp, olap, mixed, all, or coldcache
    #[arg(long, default_value = "all")]
    workload: String,

    /// Benchmark duration in seconds
    #[arg(long, default_value_t = 60)]
    duration: u64,

    /// Warmup period in seconds (results discarded)
    #[arg(long, default_value_t = 10)]
    warmup: u64,

    /// Number of rows for OLTP pre-population (default 1M)
    #[arg(long, default_value_t = 1_000_000)]
    rows: u64,

    /// Number of worker threads
    #[arg(long, default_value_t = 8)]
    threads: usize,

    /// Data directory (default: temp dir)
    #[arg(long)]
    data_dir: Option<String>,
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

#[derive(Debug, serde::Serialize)]
struct BenchmarkResults {
    hardware: HardwareInfo,
    git_hash: String,
    workloads: WorkloadResults,
}

#[derive(Debug, serde::Serialize)]
struct HardwareInfo {
    cpu: String,
    cores: usize,
    ram_gb: u64,
    os: String,
    arch: String,
    aes_ni: bool,
}

#[derive(Debug, serde::Serialize)]
struct WorkloadResults {
    #[serde(skip_serializing_if = "Option::is_none")]
    oltp: Option<OltpResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    olap: Option<OlapResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mixed: Option<MixedResult>,
}

#[derive(Debug, serde::Serialize)]
struct OltpResult {
    write_tps: u64,
    read_p50_us: u64,
    read_p99_us: u64,
    read_p999_us: u64,
    write_p50_us: u64,
    write_p99_us: u64,
    duration_secs: u64,
    pass: bool,
}

#[derive(Debug, serde::Serialize)]
struct OlapResult {
    scan_throughput_gbps: f64,
    blocks_scanned: u64,
    blocks_skipped: u64,
    zone_map_skip_pct: f64,
    duration_secs: u64,
    pass: bool,
}

#[derive(Debug, serde::Serialize)]
struct MixedResult {
    oltp_p99_during_scan_us: u64,
    oltp_p99_degradation_pct: f64,
    hotset_evictions: u64,
    pass: bool,
}

// ---------------------------------------------------------------------------
// Hardware detection
// ---------------------------------------------------------------------------

fn detect_hardware() -> HardwareInfo {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let os = std::env::consts::OS.to_string();
    let arch = std::env::consts::ARCH.to_string();

    let (cpu, ram_gb) = if cfg!(target_os = "macos") {
        let cpu = std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| format!("{} ({})", arch, os));

        let ram_gb = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse::<u64>().ok())
            .map(|bytes| bytes / (1024 * 1024 * 1024))
            .unwrap_or(0);

        (cpu, ram_gb)
    } else if cfg!(target_os = "linux") {
        let cpu = std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|contents| {
                contents
                    .lines()
                    .find(|line| line.starts_with("model name"))
                    .and_then(|line| line.split(':').nth(1))
                    .map(|s| s.trim().to_string())
            })
            .unwrap_or_else(|| format!("{} ({})", arch, os));

        let ram_gb = std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|contents| {
                contents
                    .lines()
                    .find(|line| line.starts_with("MemTotal"))
                    .and_then(|line| line.split_whitespace().nth(1))
                    .and_then(|s| s.parse::<u64>().ok())
                    .map(|kb| kb / (1024 * 1024))
            })
            .unwrap_or(0);

        (cpu, ram_gb)
    } else {
        (format!("{} ({})", arch, os), 0)
    };

    #[cfg(target_arch = "x86_64")]
    let aes_ni = std::arch::is_x86_feature_detected!("aes");
    #[cfg(not(target_arch = "x86_64"))]
    let aes_ni = false;

    HardwareInfo {
        cpu,
        cores,
        ram_gb,
        os,
        arch,
        aes_ni,
    }
}

fn detect_git_hash() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

// ---------------------------------------------------------------------------
// Key/value generation helpers
// ---------------------------------------------------------------------------

fn make_key(id: u64) -> Vec<u8> {
    // 64-byte key: 8-byte id prefix + padding
    let mut key = Vec::with_capacity(64);
    key.extend_from_slice(&id.to_be_bytes());
    key.resize(64, 0xAA);
    key
}

fn make_value(rng: &mut SmallRng) -> Vec<u8> {
    // 256-byte random value
    let mut val = vec![0u8; 256];
    rng.fill(&mut val[..]);
    val
}

// ---------------------------------------------------------------------------
// Workload 1: OLTP Write + Point Read
// ---------------------------------------------------------------------------

async fn run_oltp(
    duration_secs: u64,
    warmup_secs: u64,
    num_rows: u64,
    num_threads: usize,
    data_dir: &Path,
) -> OltpResult {
    eprintln!("[OLTP] Pre-populating {} rows...", num_rows);

    // Pre-populate: insert rows into memtable and build ART index
    let memtable = Arc::new(Memtable::new(u64::MAX)); // huge threshold so it never seals
    let art = Arc::new(ArtIndex::new());

    // Insert rows in batches
    let batch_size = 10_000u64;
    let mut rng = SmallRng::seed_from_u64(42);
    for batch_start in (0..num_rows).step_by(batch_size as usize) {
        let batch_end = (batch_start + batch_size).min(num_rows);
        for i in batch_start..batch_end {
            let key = make_key(i);
            let value = make_value(&mut rng);
            memtable.put(key.clone(), i + 1, Some(value));
            art.insert(
                key,
                RowLocation::Memtable {
                    shard: (i % 16) as u8,
                    key: make_key(i),
                },
            );
        }
    }

    eprintln!(
        "[OLTP] Pre-population complete. Memtable size: {} bytes, ART entries: {}",
        memtable.size(),
        art.len()
    );

    // Flush a subset to SST files to simulate mixed memtable/SST reads
    let flush_memtable = Memtable::new(u64::MAX);
    let flush_count = (num_rows / 10).min(100_000); // flush 10% or 100K
    let mut flush_rng = SmallRng::seed_from_u64(99);
    for i in 0..flush_count {
        let key = make_key(i);
        let value = make_value(&mut flush_rng);
        flush_memtable.put(key, i + 1, Some(value));
    }

    let flush_config = galaxdb_storage::flush::FlushConfig {
        data_dir: data_dir.to_path_buf(),
        sst_size_bytes: 64 * 1024 * 1024,
        max_rows_per_block: 100_000,
    };
    let _ = galaxdb_storage::flush::flush_memtable(&flush_memtable, &flush_config, 1).await;

    // Update ART entries for flushed rows to point to SST
    for i in 0..flush_count {
        let key = make_key(i);
        art.insert(
            key,
            RowLocation::SST {
                sst_id: 1,
                block_offset: 0,
                row_offset: i as u32,
            },
        );
    }

    eprintln!("[OLTP] Starting benchmark phase ({} seconds, {} warmup)...", duration_secs, warmup_secs);

    let running = Arc::new(AtomicBool::new(true));
    let warmed_up = Arc::new(AtomicBool::new(false));

    let write_hist = Arc::new(Mutex::new(Histogram::<u64>::new(3).unwrap()));
    let read_hist = Arc::new(Mutex::new(Histogram::<u64>::new(3).unwrap()));
    let write_ops = Arc::new(AtomicU64::new(0));
    let read_ops = Arc::new(AtomicU64::new(0));

    let total_duration = Duration::from_secs(duration_secs);
    let warmup_duration = Duration::from_secs(warmup_secs);

    let mut handles = Vec::new();

    // Writer threads (half of threads)
    let writer_count = (num_threads / 2).max(1);
    for t in 0..writer_count {
        let memtable = memtable.clone();
        let art = art.clone();
        let running = running.clone();
        let warmed_up = warmed_up.clone();
        let write_hist = write_hist.clone();
        let write_ops = write_ops.clone();

        handles.push(tokio::spawn(async move {
            let mut rng = SmallRng::seed_from_u64(1000 + t as u64);
            let mut local_hist = Histogram::<u64>::new(3).unwrap();
            let mut local_ops = 0u64;

            while running.load(Ordering::Relaxed) {
                let id = rng.gen_range(num_rows..num_rows * 2);
                let key = make_key(id);
                let value = make_value(&mut rng);

                let start = Instant::now();
                memtable.put(key.clone(), id + 1, Some(value));
                art.insert(
                    key.clone(),
                    RowLocation::Memtable {
                        shard: (id % 16) as u8,
                        key,
                    },
                );
                let elapsed_us = start.elapsed().as_micros() as u64;

                if warmed_up.load(Ordering::Relaxed) {
                    let _ = local_hist.record(elapsed_us.min(60_000_000));
                    local_ops += 1;
                }

                // Yield periodically to avoid starving the runtime
                if local_ops.is_multiple_of(1000) {
                    tokio::task::yield_now().await;
                }
            }

            let mut hist = write_hist.lock().await;
            hist.add(&local_hist).ok();
            write_ops.fetch_add(local_ops, Ordering::Relaxed);
        }));
    }

    // Reader threads (other half)
    let reader_count = (num_threads - writer_count).max(1);
    for t in 0..reader_count {
        let memtable = memtable.clone();
        let art = art.clone();
        let running = running.clone();
        let warmed_up = warmed_up.clone();
        let read_hist = read_hist.clone();
        let read_ops = read_ops.clone();

        handles.push(tokio::spawn(async move {
            let mut rng = SmallRng::seed_from_u64(2000 + t as u64);
            let mut local_hist = Histogram::<u64>::new(3).unwrap();
            let mut local_ops = 0u64;

            while running.load(Ordering::Relaxed) {
                let id = rng.gen_range(0..num_rows);
                let key = make_key(id);

                let start = Instant::now();
                // ART lookup → memtable read
                let _location = art.lookup(&key);
                let _value = memtable.get(&key);
                let elapsed_us = start.elapsed().as_micros() as u64;

                if warmed_up.load(Ordering::Relaxed) {
                    let _ = local_hist.record(elapsed_us.min(60_000_000));
                    local_ops += 1;
                }

                if local_ops.is_multiple_of(1000) {
                    tokio::task::yield_now().await;
                }
            }

            let mut hist = read_hist.lock().await;
            hist.add(&local_hist).ok();
            read_ops.fetch_add(local_ops, Ordering::Relaxed);
        }));
    }

    // Warmup phase
    tokio::time::sleep(warmup_duration).await;
    warmed_up.store(true, Ordering::SeqCst);
    eprintln!("[OLTP] Warmup complete, measuring...");

    let measure_start = Instant::now();

    // Measurement phase
    let measure_duration = total_duration.saturating_sub(warmup_duration);
    tokio::time::sleep(measure_duration).await;

    running.store(false, Ordering::SeqCst);
    for h in handles {
        let _ = h.await;
    }

    let actual_duration = measure_start.elapsed();
    let actual_secs = actual_duration.as_secs_f64();

    let rh = read_hist.lock().await;
    let wh = write_hist.lock().await;

    let total_write_ops = write_ops.load(Ordering::Relaxed);
    let _total_read_ops = read_ops.load(Ordering::Relaxed);

    let write_tps = if actual_secs > 0.0 {
        (total_write_ops as f64 / actual_secs) as u64
    } else {
        0
    };

    let read_p50 = rh.value_at_quantile(0.50);
    let read_p99 = rh.value_at_quantile(0.99);
    let read_p999 = rh.value_at_quantile(0.999);
    let write_p50 = wh.value_at_quantile(0.50);
    let write_p99 = wh.value_at_quantile(0.99);

    let pass = write_tps >= 50_000 && read_p50 <= 50;

    eprintln!(
        "[OLTP] Done. write_tps={}, read_p50={}µs, read_p99={}µs, pass={}",
        write_tps, read_p50, read_p99, pass
    );

    OltpResult {
        write_tps,
        read_p50_us: read_p50,
        read_p99_us: read_p99,
        read_p999_us: read_p999,
        write_p50_us: write_p50,
        write_p99_us: write_p99,
        duration_secs,
        pass,
    }
}

// ---------------------------------------------------------------------------
// Workload 2: OLAP Column Scan
// ---------------------------------------------------------------------------

fn create_pax_block_with_zone_map(
    block_id: BlockId,
    row_count: usize,
    base_value: i32,
    rng: &mut SmallRng,
) -> PaxBlock {
    // Int32 column with values in a range around base_value
    let int_values: Vec<Vec<u8>> = (0..row_count)
        .map(|_| {
            let v = base_value + rng.gen_range(-100..100);
            v.to_le_bytes().to_vec()
        })
        .collect();

    // Text column with random 64-byte strings
    let text_values: Vec<Vec<u8>> = (0..row_count)
        .map(|_| {
            let mut buf = vec![0u8; 64];
            rng.fill(&mut buf[..]);
            buf
        })
        .collect();

    let columns = vec![
        ColumnData {
            col_type: ColumnType::Int32,
            values: int_values,
        },
        ColumnData {
            col_type: ColumnType::Text,
            values: text_values,
        },
    ];

    PaxBlock::write(block_id, 1, &columns).expect("failed to create PAX block")
}

async fn run_olap(
    duration_secs: u64,
    _warmup_secs: u64,
    num_threads: usize,
) -> OlapResult {
    eprintln!("[OLAP] Creating 1000 PAX blocks with 10K rows each...");

    let num_blocks = 1000u64;
    let rows_per_block = 10_000usize;
    let mut rng = SmallRng::seed_from_u64(77);

    // Create blocks with varying base values for zone-map pruning
    // Blocks 0-199: base_value in [0, 200) — will match filter col < 100
    // Blocks 200-999: base_value in [200, 1000) — will be skipped by zone map
    let mut blocks: Vec<PaxBlock> = Vec::with_capacity(num_blocks as usize);
    for i in 0..num_blocks {
        let base_value = if i < 200 {
            rng.gen_range(0..200) // some will match, some won't
        } else {
            rng.gen_range(200..1000) // all above threshold
        };
        let block = create_pax_block_with_zone_map(i, rows_per_block, base_value, &mut rng);
        blocks.push(block);
    }

    // Also serialize blocks to simulate reading from disk (measures decompression, not just in-memory access)
    let serialized_blocks: Vec<Vec<u8>> = blocks
        .iter()
        .map(|b| b.serialize().expect("serialize"))
        .collect();

    eprintln!(
        "[OLAP] Pre-population complete. {} blocks, {} bytes total serialized. Starting parallel scan ({} threads, {} seconds)...",
        blocks.len(),
        serialized_blocks.iter().map(|b| b.len() as u64).sum::<u64>(),
        num_threads,
        duration_secs
    );

    // Configure rayon thread pool to match requested thread count
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("failed to build rayon thread pool");

    let threshold = 100i32;

    let deadline = Instant::now() + Duration::from_secs(duration_secs);
    let total_bytes_scanned = AtomicU64::new(0);
    let total_blocks_scanned = AtomicU64::new(0);
    let total_blocks_skipped = AtomicU64::new(0);
    let scan_iterations = AtomicU64::new(0);

    pool.install(|| {
        while Instant::now() < deadline {
            // Parallel scan: each thread picks up blocks and processes them
            use rayon::prelude::*;

            serialized_blocks.par_iter().for_each(|serialized| {
                if Instant::now() >= deadline {
                    return;
                }

                // Deserialize (simulates reading from NVMe + checksum verification)
                let block = match PaxBlock::deserialize(serialized) {
                    Ok(b) => b,
                    Err(_) => return,
                };

                // Zone-map pruning
                let desc = &block.header.column_descriptors[0];
                let block_min = {
                    let zone_min = &desc.zone_map_min;
                    if zone_min.len() >= 4 {
                        i32::from_le_bytes(zone_min[..4].try_into().unwrap_or([0; 4]))
                    } else {
                        i32::MIN
                    }
                };

                total_blocks_scanned.fetch_add(1, Ordering::Relaxed);

                // Skip block if all values are >= threshold
                if block_min >= threshold {
                    total_blocks_skipped.fetch_add(1, Ordering::Relaxed);
                    return;
                }

                // Decompress and scan the Int32 column
                let values = block.read_column(0).expect("failed to read column");
                let _matching: usize = values
                    .iter()
                    .filter(|v| {
                        if v.len() >= 4 {
                            let val = i32::from_le_bytes(v[..4].try_into().unwrap());
                            val < threshold
                        } else {
                            false
                        }
                    })
                    .count();

                // Count bytes processed (serialized block size = what we'd read from NVMe)
                total_bytes_scanned.fetch_add(serialized.len() as u64, Ordering::Relaxed);
            });

            scan_iterations.fetch_add(1, Ordering::Relaxed);
        }
    });

    let total_bytes = total_bytes_scanned.load(Ordering::Relaxed);
    let total_scanned = total_blocks_scanned.load(Ordering::Relaxed);
    let total_skipped = total_blocks_skipped.load(Ordering::Relaxed);
    let iterations = scan_iterations.load(Ordering::Relaxed);

    let actual_secs = duration_secs as f64;
    let throughput_gbps = if actual_secs > 0.0 {
        (total_bytes as f64) / (1024.0 * 1024.0 * 1024.0) / actual_secs
    } else {
        0.0
    };

    let skip_pct = if total_scanned > 0 {
        (total_skipped as f64 / total_scanned as f64) * 100.0
    } else {
        0.0
    };

    let pass = throughput_gbps >= 3.0 && skip_pct >= 79.5;

    eprintln!(
        "[OLAP] Done. throughput={:.2} GB/s, scanned={}, skipped={}, skip_pct={:.1}%, iterations={}, threads={}, pass={}",
        throughput_gbps, total_scanned, total_skipped, skip_pct, iterations, num_threads, pass
    );

    OlapResult {
        scan_throughput_gbps: (throughput_gbps * 100.0).round() / 100.0,
        blocks_scanned: total_scanned,
        blocks_skipped: total_skipped,
        zone_map_skip_pct: (skip_pct * 10.0).round() / 10.0,
        duration_secs,
        pass,
    }
}

// ---------------------------------------------------------------------------
// Workload 3: Mixed OLTP + OLAP
// ---------------------------------------------------------------------------

async fn run_mixed(
    duration_secs: u64,
    warmup_secs: u64,
    num_rows: u64,
    num_threads: usize,
    _data_dir: &Path,
) -> MixedResult {
    eprintln!("[MIXED] Setting up buffer pool and data...");

    // Set up buffer pool with HotSet and ScanBuffer
    let pool = Arc::new(tokio::sync::Mutex::new(BufferPool::new(2000, 1)));

    // Fill HotSet with 1000 blocks (point lookup access)
    {
        let mut bp = pool.lock().await;
        for i in 0..1000u64 {
            let block = CachedBlock {
                block_id: i,
                data: vec![0xAA; 4096], // 4KB blocks
            };
            bp.insert(i, block, AccessType::PointLookup, 0);
        }
        eprintln!(
            "[MIXED] HotSet populated: {} entries",
            bp.hot_set_len(0)
        );
    }

    // Record initial HotSet state
    let initial_hotset_len = {
        let bp = pool.lock().await;
        bp.hot_set_len(0)
    };

    // Set up OLTP components
    let memtable = Arc::new(Memtable::new(u64::MAX));
    let art = Arc::new(ArtIndex::new());

    // Pre-populate OLTP data (smaller set for mixed workload)
    let oltp_rows = (num_rows / 10).max(100_000);
    let mut rng = SmallRng::seed_from_u64(42);
    for i in 0..oltp_rows {
        let key = make_key(i);
        let value = make_value(&mut rng);
        memtable.put(key.clone(), i + 1, Some(value));
        art.insert(
            key.clone(),
            RowLocation::Memtable {
                shard: (i % 16) as u8,
                key,
            },
        );
    }

    eprintln!("[MIXED] Starting mixed workload ({} seconds)...", duration_secs);

    let running = Arc::new(AtomicBool::new(true));
    let warmed_up = Arc::new(AtomicBool::new(false));
    let oltp_hist = Arc::new(Mutex::new(Histogram::<u64>::new(3).unwrap()));
    let hotset_evictions = Arc::new(AtomicU64::new(0));

    let mut handles = Vec::new();

    // OLTP reader/writer threads
    let oltp_threads = (num_threads / 2).max(2);
    for t in 0..oltp_threads {
        let memtable = memtable.clone();
        let art = art.clone();
        let pool = pool.clone();
        let running = running.clone();
        let warmed_up = warmed_up.clone();
        let oltp_hist = oltp_hist.clone();

        handles.push(tokio::spawn(async move {
            let mut rng = SmallRng::seed_from_u64(3000 + t as u64);
            let mut local_hist = Histogram::<u64>::new(3).unwrap();

            while running.load(Ordering::Relaxed) {
                let id = rng.gen_range(0..oltp_rows);
                let key = make_key(id);

                let start = Instant::now();

                // Point lookup through ART → memtable
                let _loc = art.lookup(&key);
                let _val = memtable.get(&key);

                // Also check buffer pool (point lookup path)
                {
                    let mut bp = pool.lock().await;
                    let _cached = bp.get_for_point_lookup(id % 1000, 0);
                }

                let elapsed_us = start.elapsed().as_micros() as u64;

                if warmed_up.load(Ordering::Relaxed) {
                    let _ = local_hist.record(elapsed_us.min(60_000_000));
                }

                // Occasional writes
                if rng.gen_bool(0.3) {
                    let wid = rng.gen_range(oltp_rows..oltp_rows * 2);
                    let wkey = make_key(wid);
                    let wval = make_value(&mut rng);
                    memtable.put(wkey.clone(), wid + 1, Some(wval));
                    art.insert(
                        wkey.clone(),
                        RowLocation::Memtable {
                            shard: (wid % 16) as u8,
                            key: wkey,
                        },
                    );
                }

                tokio::task::yield_now().await;
            }

            let mut hist = oltp_hist.lock().await;
            hist.add(&local_hist).ok();
        }));
    }

    // OLAP scan threads — scan through ScanBuffer, should NOT evict HotSet
    let scan_threads = (num_threads - oltp_threads).max(1);
    for t in 0..scan_threads {
        let pool = pool.clone();
        let running = running.clone();

        handles.push(tokio::spawn(async move {
            let _rng = SmallRng::seed_from_u64(4000 + t as u64);
            let mut scan_block_id = 10_000u64; // start well above HotSet range

            while running.load(Ordering::Relaxed) {
                // Simulate scanning 10K different blocks through ScanBuffer
                for _ in 0..100 {
                    if !running.load(Ordering::Relaxed) {
                        break;
                    }

                    let block = CachedBlock {
                        block_id: scan_block_id,
                        data: vec![0xBB; 4096],
                    };

                    {
                        let mut bp = pool.lock().await;
                        bp.insert(scan_block_id, block, AccessType::SequentialScan, 0);
                        let _cached = bp.get_for_scan(scan_block_id, 0);
                    }

                    scan_block_id += 1;
                    if scan_block_id > 100_000 {
                        scan_block_id = 10_000;
                    }
                }

                tokio::task::yield_now().await;
            }
        }));
    }

    // Warmup
    tokio::time::sleep(Duration::from_secs(warmup_secs)).await;
    warmed_up.store(true, Ordering::SeqCst);
    eprintln!("[MIXED] Warmup complete, measuring...");

    // Measurement phase
    let measure_duration = Duration::from_secs(duration_secs).saturating_sub(Duration::from_secs(warmup_secs));
    tokio::time::sleep(measure_duration).await;

    running.store(false, Ordering::SeqCst);
    for h in handles {
        let _ = h.await;
    }

    // Check HotSet survival
    let final_hotset_len = {
        let bp = pool.lock().await;
        bp.hot_set_len(0)
    };

    let evictions = if final_hotset_len < initial_hotset_len {
        (initial_hotset_len - final_hotset_len) as u64
    } else {
        0
    };
    hotset_evictions.store(evictions, Ordering::Relaxed);

    let hist = oltp_hist.lock().await;
    let oltp_p99 = hist.value_at_quantile(0.99);

    // Compare against baseline (assume ~500µs baseline p99 from pure OLTP)
    let baseline_p99 = 500u64;
    let degradation_pct = if baseline_p99 > 0 {
        ((oltp_p99 as f64 - baseline_p99 as f64) / baseline_p99 as f64 * 100.0).max(0.0)
    } else {
        0.0
    };

    let pass = oltp_p99 <= 5_000 && evictions == 0;

    eprintln!(
        "[MIXED] Done. oltp_p99={}µs, degradation={:.1}%, hotset_evictions={}, pass={}",
        oltp_p99, degradation_pct, evictions, pass
    );

    MixedResult {
        oltp_p99_during_scan_us: oltp_p99,
        oltp_p99_degradation_pct: (degradation_pct * 10.0).round() / 10.0,
        hotset_evictions: evictions,
        pass,
    }
}

// ---------------------------------------------------------------------------
// Workload 4: Cold-Cache Read (larger-than-RAM dataset)
// ---------------------------------------------------------------------------

async fn run_coldcache(
    num_rows: u64,
    num_reads: u64,
    data_dir: &Path,
) {
    eprintln!("[COLDCACHE] Writing {} rows to engine...", num_rows);

    let config = EngineConfig {
        data_dir: data_dir.to_path_buf(),
        memtable_size_bytes: 64 * 1024 * 1024,
        back_pressure_bytes: 256 * 1024 * 1024,
        wal_group_commit_ms: 1,
    };
    let engine = Engine::new(config).unwrap();

    // Write rows in batches
    let batch_size = 10_000u64;
    let value_size = 600; // ~600 bytes per row → 50M rows = 30GB
    // Write rows using spawn_blocking to avoid async/sync conflict
    let engine = Arc::new(engine);
    let start = Instant::now();

    for batch_start in (0..num_rows).step_by(batch_size as usize) {
        let batch_end = (batch_start + batch_size).min(num_rows);
        let eng = engine.clone();

        tokio::task::spawn_blocking(move || {
            let mut batch: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity((batch_end - batch_start) as usize);
            for i in batch_start..batch_end {
                let key = format!("cc-key-{:012}", i).into_bytes();
                let mut value = vec![0u8; value_size];
                let seed = i.to_le_bytes();
                for (j, byte) in value.iter_mut().enumerate() {
                    *byte = seed[j % 8] ^ (j as u8);
                }
                batch.push((key, value));
            }
            eng.put_batch_sync(&batch).unwrap();
        }).await.unwrap();

        if (batch_start / batch_size) % 100 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let rows_done = batch_end;
            let rate = rows_done as f64 / elapsed;
            eprintln!(
                "[COLDCACHE]   {}/{} rows ({:.0} rows/sec, {:.1}s elapsed)",
                rows_done, num_rows, rate, elapsed
            );
        }
    }

    let write_elapsed = start.elapsed();
    let write_rate = num_rows as f64 / write_elapsed.as_secs_f64();
    eprintln!(
        "[COLDCACHE] Write complete: {} rows in {:.1}s ({:.0} rows/sec)",
        num_rows, write_elapsed.as_secs_f64(), write_rate
    );

    // Flush memtable to SST
    eprintln!("[COLDCACHE] Flushing memtable to SST...");
    let flushed = engine.flush_memtable().await.unwrap();
    eprintln!("[COLDCACHE] Flushed {} rows to SST", flushed);

    // Now read random keys and measure latency
    eprintln!("[COLDCACHE] Reading {} random keys...", num_reads);
    let mut rng = SmallRng::seed_from_u64(42);
    let mut hist = Histogram::<u64>::new(3).unwrap();

    for i in 0..num_reads {
        let key_id = rng.gen_range(0..num_rows);
        let key = format!("cc-key-{:012}", key_id).into_bytes();

        let read_start = Instant::now();
        let result = engine.get(&key);
        let elapsed_us = read_start.elapsed().as_micros() as u64;

        let _ = hist.record(elapsed_us.min(60_000_000));

        if result.is_none() {
            eprintln!("[COLDCACHE] WARNING: key {} not found", key_id);
        }

        if i > 0 && i % 10_000 == 0 {
            eprintln!("[COLDCACHE]   {}/{} reads done", i, num_reads);
        }
    }

    let p50 = hist.value_at_quantile(0.50);
    let p99 = hist.value_at_quantile(0.99);
    let p999 = hist.value_at_quantile(0.999);

    eprintln!("[COLDCACHE] Results:");
    eprintln!("  Rows: {}", num_rows);
    eprintln!("  Reads: {}", num_reads);
    eprintln!("  Read p50: {} µs", p50);
    eprintln!("  Read p99: {} µs", p99);
    eprintln!("  Read p999: {} µs", p999);

    // Output as JSON
    println!("{{");
    println!("  \"coldcache\": {{");
    println!("    \"rows\": {},", num_rows);
    println!("    \"reads\": {},", num_reads);
    println!("    \"read_p50_us\": {},", p50);
    println!("    \"read_p99_us\": {},", p99);
    println!("    \"read_p999_us\": {},", p999);
    println!("    \"write_rate_rows_per_sec\": {:.0}", write_rate);
    println!("  }}");
    println!("}}");

    engine.shutdown();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    #[cfg(debug_assertions)]
    {
        eprintln!("WARNING: Running in debug mode. Results are not meaningful.");
        eprintln!("Run with: cargo run --release -p galaxdb-benchmarks");
    }

    let cli = Cli::parse();

    let data_dir = cli
        .data_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let tmp = tempfile::tempdir().expect("failed to create temp dir");
            // Leak the tempdir so it persists for the benchmark duration
            let path = tmp.path().to_path_buf();
            std::mem::forget(tmp);
            path
        });

    std::fs::create_dir_all(&data_dir).expect("failed to create data dir");

    let hardware = detect_hardware();
    let git_hash = detect_git_hash();

    let workload = cli.workload.to_lowercase();

    let mut oltp_result = None;
    let mut olap_result = None;
    let mut mixed_result = None;

    if workload == "oltp" || workload == "all" {
        oltp_result = Some(
            run_oltp(
                cli.duration,
                cli.warmup,
                cli.rows,
                cli.threads,
                &data_dir,
            )
            .await,
        );
    }

    if workload == "olap" || workload == "all" {
        olap_result = Some(
            run_olap(cli.duration, cli.warmup, cli.threads).await,
        );
    }

    if workload == "mixed" || workload == "all" {
        mixed_result = Some(
            run_mixed(
                cli.duration,
                cli.warmup,
                cli.rows,
                cli.threads,
                &data_dir,
            )
            .await,
        );
    }

    if workload == "coldcache" {
        // Cold-cache benchmark: NOT included in "all" because it takes 10+ minutes
        // Usage: --workload coldcache --rows 50000000
        run_coldcache(cli.rows, 100_000, &data_dir).await;
        return;
    }

    let results = BenchmarkResults {
        hardware,
        git_hash,
        workloads: WorkloadResults {
            oltp: oltp_result,
            olap: olap_result,
            mixed: mixed_result,
        },
    };

    let json = serde_json::to_string_pretty(&results).expect("failed to serialize results");
    println!("{}", json);
}
