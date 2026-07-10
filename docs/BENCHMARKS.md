# GalaxDB Benchmarks

All numbers in this document are measured on real hardware against real datasets with `--release` builds. Random-vector HNSW benchmarks are not reported — only SIFT-1M or equivalent ANN-benchmarks datasets.

---

## RGABH adaptive buffer pool — skewed-workload hit rate (v0.7)

**Hardware:** Intel Core i7-7820HQ, macOS 13.7.8, rustc 1.96.0, `--release`. Date: 2026-07-10.
**Command:** `cargo run --release -p galaxdb-storage --example rgabh_hitrate`

Deterministic Zipfian-skewed access trace (YCSB-style: ~80% of accesses hit a 1,500-key hot
set, ~20% over a 50,000-key cold tail), identical seed through two 2,000-slot pools.

| Policy | HotSet hit rate |
|---|---|
| LRU/clock baseline | 0.6391 |
| RGABH adaptive | 0.8030 |
| **Delta** | **+16.39 pp** |

RGABH's W-TinyLFU-style frequency admission keeps the durably-hot working set resident instead
of letting the one-shot cold stream evict it. Admission/eviction is O(K); disabling RGABH
reproduces the LRU/clock baseline exactly. Full record: `bench-results/rgabh-20260710/`.

---

## DiskANN disk-resident ANN — recall (v0.7)

Recall verified against exact brute-force ground truth on real clustered data (`--release`):
recall@10 ≥ 0.90 (cosine) and ≥ 0.85 (L2). Command: `cargo test --release -p galaxdb-vector diskann`.

The full SIFT1M-scale recall and QPS numbers will be added once the 1M-vector run completes on the
reference hardware. The harness is included:

```bash
cargo run --release -p galaxdb-vector --example diskann_sift_recall -- \
    sift_base.fvecs sift_query.fvecs sift_groundtruth.ivecs 10 100 64 125
```

---

## Vector Search — HNSW on SIFT-1M

**Hardware:** AWS c6id.4xlarge — Intel Xeon Platinum 8375C (Ice Lake, 16 vCPU, 32 GiB RAM, 884 GB NVMe), Ubuntu 24.04, io_uring backend.

**Dataset:** SIFT-1M — 1,000,000 × 128-dim float32 vectors, 10,000 queries, pre-computed ground truth.  
Source: `ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz`  
SHA256: `92f1270c5e3a0cb46b89983e72b0511e4df065c31a9fa0276d8c9b1fca5bc81a`

**Build:** M=16, ef_construction=200 → 65.4 s (15,295 vec/sec)

| ef_search | recall@10 | mean latency | p99 latency |
|-----------|-----------|--------------|-------------|
| 10        | 0.756     | 57.7 µs      | 105 µs      |
| 50        | 0.959     | 156.7 µs     | 229 µs      |
| 100       | 0.983     | 266.7 µs     | 364 µs      |
| **200**   | **0.990** | **458.9 µs** | **612 µs**  |

**Reproducing these numbers:**

```bash
cargo build --release -p galaxdb-benchmarks
./target/release/galaxdb-sift-bench \
    --dataset /path/to/sift \
    --ef-search 10,50,100,200 \
    --commit-sha "$(git rev-parse HEAD)" \
    --instance-type c6id.4xlarge \
    --dataset-sha256 92f1270c5e3a0cb46b89983e72b0511e4df065c31a9fa0276d8c9b1fca5bc81a \
    --timestamp-utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --output bench-results/sift_bench.json
```

> Verified 2026-07-03, commit `6c7811f` (v0.3.0 line), against the SHA256-pinned SIFT-1M dataset.
> Full provenance JSON: `bench-results/20260703T083602Z/sift_bench.json`. The HNSW path is
> unchanged by the later SQL/durability fixes, so these numbers hold for v0.3.0. A prior run at
> `f1825c5` (2026-06-25) gave statistically identical recall (see
> `bench-results/aws-live-20260625/`).

---

## Test Suite — release (AWS c6id.4xlarge)

Confirmed on the same hardware as the vector benchmarks, release build, commit `6c7811f`
(2026-07-03), `cargo test --release --lib` across 10 crates. Full log:
`bench-results/20260703T083602Z/`:

| Metric | Result |
|--------|--------|
| Rust tests (common, crypto, storage, sql, embedded, vector, wire, query, …) | **823 passed / 0 failed** |
| ↳ includes buffered transactions (snapshot isolation + savepoints), analytical `AT VERSION`, columnar `force_compact` rewrite, `result_codec` binary encoding, and the SQL conformance corpus | |
| Credential-gated cloud-KMS / Vault integration tests | skipped cleanly (no creds) |

> Note: the v0.3.0 production-hardening fixes landed after this run (real UPDATE/INSERT expression
> evaluation, in-transaction `ORDER BY`/`LIMIT`, `PRIMARY KEY` uniqueness, `pg_catalog`
> WHERE/projection/COUNT, FROM-less scalar `SELECT`, and durable catalog persistence). They add
> unit + integration tests that pass locally under `-D warnings`; the next AWS run will fold them
> into the on-hardware count.

---

## Storage Engine — durable write path

**Hardware:** AWS c6id.4xlarge (Intel Xeon Platinum 8375C, 16 vCPU, 32 GiB RAM), instance-store NVMe (XFS, noatime). GalaxDB and PostgreSQL 16.14 run on the **same NVMe** (PostgreSQL's data directory relocated to `/mnt/nvme/pgdata`) with `fsync=on`. Both use prepared statements. Date: 2026-06-25, commit `f1825c5`. Full findings: `bench-results/aws-live-20260625/results.md`.

### Concurrent INSERT vs PostgreSQL 16 (apples-to-apples)

| Clients | GalaxDB | PostgreSQL 16 |
|---------|---------|---------------|
| 1  | 10,450 rows/s | 11,891 rows/s |
| 4  | 30,468 rows/s | 34,298 rows/s |
| 8  | 36,632 rows/s | 54,432 rows/s |
| 16 | 37,448 rows/s | 84,747 rows/s |

GalaxDB is competitive at low concurrency (0.88× PostgreSQL at 1 client, 0.89× at 4); PostgreSQL's
process-per-connection model scales better past 8 clients (GalaxDB 0.44× at 16). The remaining gap is
the async server's per-query `spawn_blocking` hand-off, not the storage engine.

```bash
TMPDIR=/mnt/nvme/tmp ./target/release/concurrent-insert-bench \
    --rows 5000 --clients 1,4,8,16 --pg-port 5432
```

### Bulk load and engine write path

| Path | Throughput | What it measures |
|------|-----------|------------------|
| `COPY FROM STDIN` (wire) | 190,287 rows/s, 17.1 MiB/s | bulk ingest, chunked group commit (25.97× vs row INSERT) |
| single-row `INSERT` (wire) | ~8,525 rows/s | per-row roundtrip over the PostgreSQL protocol |
| `put_sync` (engine, 1 fsync/row) | ~26,500 rows/s (37.7 µs/row) | strict per-row durability |
| `put_batch_sync` (engine, 1 fsync/batch) | ~1.4–2.1M rows/s (0.5–0.7 µs/row) | in-memory path, amortized fsync |

```bash
TMPDIR=/mnt/nvme/tmp ./target/release/copy-bench --rows 200000
TMPDIR=/mnt/nvme/tmp ./target/release/single-row-insert-bench --rows 20000
TMPDIR=/mnt/nvme/tmp ./target/release/engine-microbench --max 200000
```

### NVMe fsync ceiling (context)

`pg_test_fsync` on the same instance-store NVMe: `open_datasync` 1,611 ops/s, `fdatasync` 1,091 ops/s,
`fsync` 366 ops/s. Single-client strict-durability write throughput is bounded by this fsync latency
on both engines; concurrent clients exceed it because group commit shares one fsync across writers.

---

## Encryption

Measured with `cargo bench -p galaxdb-crypto` on the same hardware.

| Operation | Latency | Throughput |
|-----------|---------|------------|
| AEGIS-256 decrypt 1 MB | 151 µs | 6.63 GB/s |
| AEGIS-256 encrypt 64 KB | 9.75 µs | 6.56 GB/s |
| AES-256-GCM decrypt 1 MB | 701 µs | 1.43 GB/s |
| XXH3-64 checksum 1 MB | — | 34.1 GB/s |
| ART lookup (1M keys) | 168 ns/op | — |

---

## Crash Safety

All 7 chaos scenarios pass in < 30 s total:

```bash
cargo run --release -p galaxdb-chaos-tests
```

| Scenario | Result | Time |
|----------|--------|------|
| Kill mid-flush → WAL replay | 1,000 rows recovered, zero loss | 8.79 s |
| Kill mid-compaction → old blocks intact | 4,000 keys readable | 0.02 s |
| Corrupt WAL record → replay stops at corruption | Partial recovery, no corrupt data returned | 1.81 s |
| Disk full → clean checkpoint, writes blocked | Reserve file freed, reads continue | 0.01 s |
| Kill sidecar → backlog preserved, no data loss | 50 requests queued, drained on recovery | 0.00 s |
| 100 concurrent writers | 100K writes, 0 duplicates, 0 missing | 0.13 s |
| OLAP scan during OLTP | 0 HotSet evictions, OLTP p99 unaffected | 0.15 s |
| **Total** | **7 passed / 0 failed** | **10.91 s** |

---

## Security & v2-Phase 1 Features — Live Verification (AWS c6id.4xlarge)

The v2-phase1 security and access-control surface is verified by starting the real
`galaxdb-server` binary on the AWS instance and driving it with a real PostgreSQL client
(`psql`, libpq SCRAM-SHA-256) over TCP — not unit tests. The captured transcript is committed
under `bench-results/live-session-*/live-session.log`.

The live workload exercises, against the running server:

- **SCRAM-SHA-256 authentication** on every connection (initial superuser provisioned from env).
- **CRUD** — `CREATE TABLE`, `INSERT`, `SELECT`, `WHERE` filtering, point lookup, `UPDATE … WHERE`,
  `DELETE … WHERE` (each verified by a follow-up read).
- **Secondary index** — `CREATE INDEX` (rebuilds from existing rows), index-accelerated equality
  `SELECT`, `DROP INDEX`.
- **TLS 1.2/1.3** — a second connection with `sslmode=require` completes a real rustls handshake and
  runs DDL+DML over the encrypted channel.
- **RBAC** — a non-privileged role is denied `SELECT` with SQLSTATE `42501`, succeeds after `GRANT`,
  and is denied an admin `GRANT` of its own — all on a live connection with no restart.
- **Audit log** — the server writes a JSONL record for every authentication, authorization, DDL, and
  admin event.

**Reproducing the live session** (starts the instance, runs the workload, always stops the instance):

```bash
GALAXDB_AWS_INSTANCE_ID=i-... GALAXDB_SSH_KEY=~/.ssh/your-key.pem \
    bash scripts/aws-live-session.sh
```

This run also confirms the `io_uring` storage backend is selected automatically on Linux.

---

## Competitive Comparison

A direct comparison against hnswlib on the same hardware and dataset is in progress. The harness is at `benchmarks/tools/hnswlib_recall.py`. Numbers will be published here once both systems have been run on the same machine against the same SIFT-1M dataset.

---

*Every number in this document was measured on real hardware with `cargo run/bench --release`. Commands are shown inline. No synthetic or estimated numbers are published.*
