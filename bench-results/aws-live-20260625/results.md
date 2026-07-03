# GalaxDB live benchmark run — 2026-06-25

All numbers below are from a single live session on real hardware. No mocks,
no fabricated values. Every write-path benchmark drives a **real
`galaxdb-server`** over the PostgreSQL wire protocol with `tokio-postgres`
clients (server + DB up); the engine microbench drives a real
`galaxdb_storage::Engine`; SIFT1M drives the real HNSW index over the
verified ANN-benchmarks dataset.

## Hardware / provenance

- Instance: AWS `c6id.4xlarge` (id redacted), us-east-1
- CPU: Intel Xeon Platinum 8375C @ 2.90 GHz, 16 vCPU
- RAM: 30 GiB
- Storage: instance-store NVMe (~885 GiB) at `/mnt/nvme`, xfs, `noatime`
- Build: `cargo build --release` (7m08s), Rust stable
- Commit: `f1825c5642061ca0901cfb31bda94e23b28d42fb`
- PostgreSQL comparison: PostgreSQL 16.14, **data directory relocated to the
  same NVMe** (`/mnt/nvme/pgdata/main`) so the comparison is apples-to-apples
  on identical storage; `fsync=on`.

## Correctness — full test suite (`--release`, on the instance)

730 tests pass, 0 failed: common 14, crypto 40, storage 341 + compaction_driver 5
+ mvcc_timestamps 3, sql 157, embedded 58, vector 50, wire 62. (Credential-gated
cloud-KMS/Vault integration tests skip cleanly.)

## Live wire-protocol scenario (real server + psql)

A real `galaxdb-server --port 5433 --data-dir /mnt/nvme/livedb` (IoUring
scheduler, `/health` = ok) driven by `psql`:

- `CREATE TABLE`, `INSERT`, `SELECT ... WHERE`, `UPDATE`, `DELETE` — correct
  MVCC results (deleted/updated rows reflected).
- `CREATE VERSION TAG 'snap'` then `UPDATE` then `SELECT ... AT VERSION 'snap'`
  returns the pre-update value (`orig`) while the live read returns `updated`
  — time-travel correct.
- `... JOIN ...` → `ERROR: feature not supported: JOIN not supported` (SQLSTATE 0A000).
- `SELECT COUNT(*)` → `ERROR: feature not supported: aggregate functions not supported`.

## Engine write path (`engine-microbench --max 200000`)

- `put_sync` (one WAL fsync per row): ~26,500 rows/s, ~37.7 µs/row (stable across 200k rows).
- `put_batch_sync` (one fsync per call): ~1.4–2.1 M rows/s, ~0.5–0.7 µs/row.

## Wire INSERT (`single-row-insert-bench --rows 20000`)

- simple protocol (re-parse each): 8,525 rows/s
- prepared (parse-once): 8,332 rows/s

## COPY bulk load (`copy-bench --rows 200000`, all methods verified to persist exactly 200000 rows)

- insert-simple: 7,328 rows/s (0.7 MiB/s)
- insert-prepared: 7,539 rows/s (0.7 MiB/s)
- **COPY: 190,287 rows/s (17.1 MiB/s) — 25.97× faster than row-by-row INSERT**

## Concurrent wire INSERT vs PostgreSQL (`concurrent-insert-bench --rows 5000 --clients 1,4,8,16 --pg-port 5432`, both on the same NVMe)

| clients | GalaxDB TPS | PostgreSQL 16 TPS |
|--------:|------------:|------------------:|
| 1       | 10,450      | 11,891            |
| 4       | 30,468      | 34,298            |
| 8       | 36,632      | 54,432            |
| 16      | 37,448      | 84,747            |

Honest reading: GalaxDB is competitive at low concurrency (1–4 clients); PostgreSQL
scales better past 8 clients. (Consistent with the prior corrected baseline.)

## SIFT1M ANN recall (`galaxdb-sift-bench`, dataset sha256 `92f1270c…5bc81a`)

1,000,000 × 128-dim base, 10,000 queries. HNSW M=16, ef_construction=200.
Build: 63,974 ms (15,631 vec/s).

| ef_search | recall@10 | mean latency | p99 latency |
|----------:|----------:|-------------:|------------:|
| 10        | 0.7590    | 50.0 µs      | 92 µs       |
| 50        | 0.9588    | 143.3 µs     | 208 µs      |
| 100       | 0.9830    | 246.6 µs     | 334 µs      |
| 200       | 0.9901    | 432.1 µs     | 572 µs      |

Full provenance JSON: `sift_bench.json` in this directory.

## Reproduce

```bash
# on the c6id.4xlarge, workspace at /mnt/nvme/galaxdb, dataset at /mnt/nvme/datasets/sift/sift
cargo build --release -p galaxdb-server -p galaxdb-benchmarks
./target/release/engine-microbench --max 200000
./target/release/single-row-insert-bench --rows 20000
./target/release/copy-bench --rows 200000
./target/release/concurrent-insert-bench --rows 5000 --clients 1,4,8,16 --pg-port 5432
./target/release/galaxdb-sift-bench --dataset /mnt/nvme/datasets/sift/sift \
    --commit-sha <sha> --instance-type c6id.4xlarge \
    --dataset-sha256 92f1270c5e3a0cb46b89983e72b0511e4df065c31a9fa0276d8c9b1fca5bc81a \
    --timestamp-utc <utc> --output sift_bench.json
```
