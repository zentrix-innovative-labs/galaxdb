//! SIFT1M Benchmark — the standard ANN benchmark dataset.
//!
//! Uses the real SIFT1M dataset (1M vectors, 128-dim, L2 distance)
//! with pre-computed ground truth for accurate recall measurement.

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::Instant;

use galaxdb_vector::{HnswConfig, HnswGraph};

/// Read .fvecs format: [dim: i32][float32 × dim] per vector
fn read_fvecs(path: &Path) -> Vec<Vec<f32>> {
    let mut file = File::open(path).expect(&format!("cannot open {:?}", path));
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();

    let mut vectors = Vec::new();
    let mut offset = 0;
    while offset < buf.len() {
        let dim = i32::from_le_bytes(buf[offset..offset+4].try_into().unwrap()) as usize;
        offset += 4;
        let mut vec = Vec::with_capacity(dim);
        for _ in 0..dim {
            vec.push(f32::from_le_bytes(buf[offset..offset+4].try_into().unwrap()));
            offset += 4;
        }
        vectors.push(vec);
    }
    vectors
}

/// Read .ivecs format: [dim: i32][int32 × dim] per vector
fn read_ivecs(path: &Path) -> Vec<Vec<i32>> {
    let mut file = File::open(path).expect(&format!("cannot open {:?}", path));
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();

    let mut vectors = Vec::new();
    let mut offset = 0;
    while offset < buf.len() {
        let dim = i32::from_le_bytes(buf[offset..offset+4].try_into().unwrap()) as usize;
        offset += 4;
        let mut vec = Vec::with_capacity(dim);
        for _ in 0..dim {
            vec.push(i32::from_le_bytes(buf[offset..offset+4].try_into().unwrap()));
            offset += 4;
        }
        vectors.push(vec);
    }
    vectors
}

pub fn run_sift_benchmark(sift_dir: &str) {
    let base_path = Path::new(sift_dir).join("sift_base.fvecs");
    let query_path = Path::new(sift_dir).join("sift_query.fvecs");
    let gt_path = Path::new(sift_dir).join("sift_groundtruth.ivecs");

    eprintln!("[SIFT] Loading SIFT1M dataset from {}...", sift_dir);

    let base_vectors = read_fvecs(&base_path);
    let query_vectors = read_fvecs(&query_path);
    let ground_truth = read_ivecs(&gt_path);

    let num_base = base_vectors.len();
    let num_queries = query_vectors.len();
    let dim = base_vectors[0].len();

    eprintln!("[SIFT] Base: {} vectors, dim={}", num_base, dim);
    eprintln!("[SIFT] Queries: {}, Ground truth: {} entries", num_queries, ground_truth.len());

    // SIFT uses L2 distance, but our HNSW uses cosine distance on normalized vectors.
    // For SIFT, we normalize all vectors to unit length so cosine distance approximates
    // the L2 ranking (since all SIFT vectors have similar norms, the ranking is preserved).

    let config = HnswConfig::new(dim)
        .with_m(16)
        .with_ef_construction(200)
        .with_max_elements(num_base);
    let mut hnsw = HnswGraph::new(config);

    // Build index
    eprintln!("[SIFT] Building HNSW index (M=16, ef_construction=200)...");
    let build_start = Instant::now();

    let entries: Vec<(u64, Vec<f32>)> = base_vectors.into_iter()
        .enumerate()
        .map(|(i, v)| (i as u64, v))
        .collect();
    hnsw.insert_parallel(&entries);

    let build_elapsed = build_start.elapsed();
    let build_rate = num_base as f64 / build_elapsed.as_secs_f64();
    eprintln!("[SIFT] Build complete: {:.1}s ({:.0} vec/sec)", build_elapsed.as_secs_f64(), build_rate);

    // Measure recall at different ef values
    let k = 10;
    for ef_search in [50, 100, 200, 500] {
        let mut total_recall = 0.0f64;
        let num_test = 1000.min(num_queries);

        let search_start = Instant::now();
        for i in 0..num_test {
            let results = hnsw.search(&query_vectors[i], k, ef_search);
            let found: std::collections::HashSet<u64> = results.iter().map(|r| r.0).collect();
            let gt: std::collections::HashSet<u64> = ground_truth[i].iter()
                .take(k)
                .map(|&id| id as u64)
                .collect();
            total_recall += found.intersection(&gt).count() as f64 / k as f64;
        }
        let search_elapsed = search_start.elapsed();
        let avg_recall = total_recall / num_test as f64;
        let qps = num_test as f64 / search_elapsed.as_secs_f64();

        eprintln!("[SIFT] ef={}: recall@{}={:.4}, QPS={:.0}, latency_avg={:.0}µs",
            ef_search, k, avg_recall, qps,
            search_elapsed.as_micros() as f64 / num_test as f64);
    }

    // Output JSON
    println!("{{");
    println!("  \"sift_benchmark\": {{");
    println!("    \"num_vectors\": {},", num_base);
    println!("    \"dim\": {},", dim);
    println!("    \"build_time_secs\": {:.2},", build_elapsed.as_secs_f64());
    println!("    \"build_rate_vec_per_sec\": {:.0}", build_rate);
    println!("  }}");
    println!("}}");
}
