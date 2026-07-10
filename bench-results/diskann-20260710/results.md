# DiskANN (Vamana) disk-resident ANN — recall verification

**Feature:** v0.7 Task 6 (inventory 8.17) — Vamana graph + FreshDiskANN delta,
graph and full-precision vectors on disk, bounded in-memory node cache.

## What is verified here (real brute-force ground truth, `--release`)

Recall is measured against exact brute-force nearest neighbors on real (clustered,
non-uniform) data — not random-vector recall, which is never reported for graph
indexes. Run: `cargo test --release -p galaxdb-vector diskann`.

| Test | Data | Metric | k | Verified recall |
|---|---|---|---|---|
| `build_search_high_recall_vs_brute_force` | 2,000 × 32d, 40 clusters | cosine | 10 | ≥ 0.90 |
| `l2_metric_build_and_search` | 400 × 12d, 10 clusters | L2 | 10 | ≥ 0.85 |
| `reopen_from_disk_searches_without_rebuild` | 500 × 16d | cosine | 5 | exact-match self-recall from disk |
| `incremental_insert_is_findable_before_consolidate` | 300 × 16d + delta | cosine | 3 | new point findable pre- and post-consolidate |
| `delete_excludes_from_results` | 300 × 16d | cosine | 5 | tombstoned id excluded |
| `open_refuses_too_new_format` | — | — | — | `GDAN` version+1 → `FormatTooNew` |

- Hardware: Intel Core i7-7820HQ, macOS 13.7.8, rustc 1.96.0, `--release`.
- Date: 2026-07-10.

## SIFT1M full-scale number (deferred to AWS instance)

The documented SIFT1M recall@10 / QPS target must be measured on the standard
1M-vector dataset. The reproducible harness is shipped:

```bash
cargo run --release -p galaxdb-vector --example diskann_sift_recall -- \
    sift_base.fvecs sift_query.fvecs sift_groundtruth.ivecs 10 100 64 125
```

**Status (2026-07-10):** the algorithm is verified correct (recall vs exact
ground truth above) and disk-resident; the SIFT1M-scale number is **deferred**
because the current `VamanaBuilder` is single-threaded and a 1M-point build must
run on the AWS benchmark instance (`i-0b2dec9226f62db65`) with a parallel build
pass. Per the no-faked-benchmarks rule, no SIFT1M number is published until that
run completes. This deferral is recorded in `feature-inventory.md`.
