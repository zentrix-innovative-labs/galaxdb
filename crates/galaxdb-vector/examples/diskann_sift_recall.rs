//! DiskANN SIFT1M recall harness (v0.7, inventory 8.17).
//!
//! Reproducible command (run on the AWS benchmark instance where SIFT1M lives):
//!
//! ```text
//! cargo run --release -p galaxdb-vector --example diskann_sift_recall -- \
//!     <base.fvecs> <query.fvecs> <groundtruth.ivecs> [k] [l_search] [R] [L_build]
//! ```
//!
//! SIFT1M (ANN-benchmarks / TEXMEX): 1,000,000 128-dim base vectors, 10,000
//! queries, exact L2 ground truth. This builds a DiskANN (Vamana) index with the
//! vectors + graph on disk, then reports recall@k against the provided ground
//! truth — the honest, dataset-named number required by the no-faked-benchmarks
//! rule. Random-vector recall is never reported for HNSW/DiskANN.
//!
//! `.fvecs`: each vector is `[dim: i32-le][dim × f32-le]`.
//! `.ivecs`: same framing with i32 payload (ground-truth neighbor ids).

use std::path::Path;
use std::time::Instant;

use galaxdb_vector::diskann::{DiskAnnConfig, DiskAnnIndex, Metric};

fn read_fvecs(path: &Path) -> Vec<Vec<f32>> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    let mut out = Vec::new();
    let mut i = 0usize;
    while i + 4 <= bytes.len() {
        let dim = i32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]) as usize;
        i += 4;
        let mut v = Vec::with_capacity(dim);
        for _ in 0..dim {
            let f = f32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
            v.push(f);
            i += 4;
        }
        out.push(v);
    }
    out
}

fn read_ivecs(path: &Path) -> Vec<Vec<u32>> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    let mut out = Vec::new();
    let mut i = 0usize;
    while i + 4 <= bytes.len() {
        let dim = i32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]) as usize;
        i += 4;
        let mut v = Vec::with_capacity(dim);
        for _ in 0..dim {
            let x = u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
            v.push(x);
            i += 4;
        }
        out.push(v);
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: {} <base.fvecs> <query.fvecs> <groundtruth.ivecs> [k=10] [l_search=100] [R=64] [L_build=125]",
            args[0]
        );
        std::process::exit(2);
    }
    let base = read_fvecs(Path::new(&args[1]));
    let queries = read_fvecs(Path::new(&args[2]));
    let truth = read_ivecs(Path::new(&args[3]));
    let k: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(10);
    let l_search: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(100);
    let r: usize = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(64);
    let l_build: usize = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(125);

    let dim = base[0].len();
    println!(
        "SIFT recall harness: base={} queries={} dim={dim} k={k} L_search={l_search} R={r} L_build={l_build}",
        base.len(),
        queries.len()
    );

    let entries: Vec<(u64, Vec<f32>)> = base
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u64, v.clone()))
        .collect();

    let dir = std::env::temp_dir().join("galaxdb_diskann_sift");
    std::fs::create_dir_all(&dir).unwrap();
    let idx_path = dir.join("sift.gdan");

    let cfg = DiskAnnConfig::new(dim)
        .with_metric(Metric::L2)
        .with_r(r)
        .with_l_build(l_build);

    let t0 = Instant::now();
    let index = DiskAnnIndex::build(&idx_path, &entries, cfg).unwrap();
    println!("build: {:?} ({} points on disk)", t0.elapsed(), index.len());

    let t1 = Instant::now();
    let mut hit = 0usize;
    let mut total = 0usize;
    for (qi, q) in queries.iter().enumerate() {
        let got = index.search(q, k, Some(l_search)).unwrap();
        let gt: std::collections::HashSet<u64> =
            truth[qi].iter().take(k).map(|&x| x as u64).collect();
        for (id, _) in &got {
            if gt.contains(id) {
                hit += 1;
            }
        }
        total += k;
    }
    let recall = hit as f64 / total as f64;
    let qps = queries.len() as f64 / t1.elapsed().as_secs_f64();
    println!("recall@{k} = {recall:.4}  |  {qps:.0} queries/sec  |  search {:?}", t1.elapsed());
}
