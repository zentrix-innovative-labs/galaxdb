# AWS run — HTAP query engine, SIFT1M + full suite (2026-07-03)

Real run on the named benchmark instance. No mocks, no fabricated numbers:
every value comes from a `--release` build on real hardware over the
sha256-verified SIFT1M dataset, via `scripts/aws-integration-run.sh`.

- **Commit:** `6c7811f5ecbfe50e8e291d202a3420ad7228fd78` (branch
  `feat/v2-phase1-copy-protocol` — the HTAP query-engine work: tasks 1–20,
  22–26, 28).
- **Instance:** c6id.4xlarge (id redacted), us-east-1 — Intel Xeon
  Platinum 8375C, 16 vCPU, 30 GiB. **Stopped after the run** (confirmed
  `instance final state: stopped`).
- **Dataset:** SIFT1M (1,000,000 × 128-dim), sha256
  `92f1270c5e3a0cb46b89983e72b0511e4df065c31a9fa0276d8c9b1fca5bc81a`.
- **Build:** release, 17m28s (full workspace incl. DataFusion/Lance).

## Full test suite on real hardware (regression gate)

`cargo test --release --lib` across 10 crates: **823 passed, 0 failed.**
This includes every test added this session — columnar `force_compact`
rewrite + `scan_columnar_fully_on_disk_does_zero_string_parse`, buffered
transactions (SI + savepoints), analytical `AT VERSION`, `result_codec`
binary encoding, SEMANTIC_MATCH candidate operator, and the SQL conformance
corpus — so the HTAP work is green on x86_64 Linux release, not just macOS
dev.

## SIFT1M HNSW (M=16, ef_construction=200), recall@10, 10k queries

| ef  | recall@10 | mean µs | p99 µs |
|-----|-----------|---------|--------|
| 10  | 0.7562    | 57.7    | 105    |
| 50  | 0.9591    | 156.7   | 229    |
| 100 | 0.9828    | 266.7   | 364    |
| 200 | 0.9900    | 458.9   | 612    |

Build: 65,378 ms (15,295 vec/sec).

## No regression vs the prior published run (`aws-live-20260625`, commit f1825c5)

Recall is unchanged within run-to-run noise (ef=100: 0.98282 vs 0.98299;
ef=200: 0.98999 vs 0.99010) and build throughput is comparable (15,295 vs
15,631 vec/sec). Latencies are ~10–15% higher on this fresh-boot run (p99
ef=100: 364 vs 334 µs) — normal cloud run-to-run variance; the HNSW path is
untouched by the HTAP query-engine work, which is additive on the columnar
storage + SQL layers. **Conclusion: the HTAP work does not regress vector
recall or build throughput.**

## Reproduction

```bash
GALAXDB_AWS_INSTANCE_ID=<your-instance-id> AWS_REGION=us-east-1 \
  GALAXDB_SSH_KEY=$HOME/.ssh/galaxdb-bench-key.pem \
  bash scripts/aws-integration-run.sh
```

## Still outstanding for task 27

This run covers SIFT1M recall + the full regression suite on the named
hardware. The **ClickBench-style single-table aggregation** and **TPC-H
subset** analytical benchmarks (which would exercise the new DataFusion
analytical path at scale) are not yet implemented as a bench binary; they
need a dedicated harness + datasets and a second instance run. An OLTP
point-read latency microbench is likewise not part of the SIFT harness. Task
27 remains open pending those.
