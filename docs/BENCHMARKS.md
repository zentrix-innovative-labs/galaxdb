# GalaxDB Benchmarks

All numbers in this document are measured on real hardware against real datasets with `--release` builds. Random-vector HNSW benchmarks are not reported — only SIFT-1M or equivalent ANN-benchmarks datasets.

---

## Vector Search — HNSW on SIFT-1M

**Hardware:** AWS c6id.4xlarge — Intel Xeon Platinum 8375C (Ice Lake, 16 vCPU, 32 GiB RAM, 884 GB NVMe), Ubuntu 24.04, io_uring backend.

**Dataset:** SIFT-1M — 1,000,000 × 128-dim float32 vectors, 10,000 queries, pre-computed ground truth.  
Source: `ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz`  
SHA256: `92f1270c5e3a0cb46b89983e72b0511e4df065c31a9fa0276d8c9b1fca5bc81a`

**Build:** M=16, ef_construction=200 → 66.2 s (15,114 vec/sec)

| ef_search | recall@10 | mean latency | p99 latency |
|-----------|-----------|--------------|-------------|
| 10        | 0.762     | 57.6 µs      | 101 µs      |
| 50        | 0.959     | 158.1 µs     | 228 µs      |
| 100       | 0.983     | 266.5 µs     | 364 µs      |
| **200**   | **0.990** | **459.4 µs** | **616 µs**  |

**Reproducing these numbers:**

```bash
cargo run --release -p galaxdb-benchmarks -- \
    --sift-dir /path/to/sift \
    --ef-sweep 10,50,100,200 \
    --output bench-results/sift_bench.json
```

---

## Test Suite — v1 Release (AWS c6id.4xlarge)

Confirmed on the same hardware as the vector benchmarks, release build:

| Metric | Result |
|--------|--------|
| Rust unit tests | **740 passed / 0 failed** |
| Chaos scenarios | **7 passed / 0 failed** |
| Total chaos suite time | **10.91 s** (all recovery scenarios < 30 s each) |

---

## Storage Engine — durable write path

**Hardware:** AWS c6id.4xlarge (Intel Xeon Platinum 8375C, 16 vCPU, 32 GiB RAM), instance-store NVMe (XFS, noatime). GalaxDB and PostgreSQL 16 run on the **same NVMe** with `synchronous_commit=on` / `fdatasync`. Both use prepared statements. Date: 2026-06-18. Full findings: `bench-results/wal-inline-20260618/results.md`.

### Concurrent INSERT vs PostgreSQL 16 (apples-to-apples)

| Clients | GalaxDB | PostgreSQL 16 |
|---------|---------|---------------|
| 1  | 10,269 rows/s | 11,810 rows/s |
| 4  | 30,637 rows/s | 34,431 rows/s |
| 8  | 36,062 rows/s | 53,397 rows/s |
| 16 | 36,264 rows/s | 82,232 rows/s |

GalaxDB is competitive at low concurrency (0.87× PostgreSQL at 1 client, 0.89× at 4); PostgreSQL's
process-per-connection model scales better past 8 clients (GalaxDB 0.44× at 16). The remaining gap is
the async server's per-query `spawn_blocking` hand-off (24.5% futex in a `perf` profile of the
16-client run), not the storage engine.

```bash
TMPDIR=/mnt/nvme/tmp cargo run --release -p galaxdb-benchmarks --bin concurrent-insert-bench \
    -- --rows 2000 --clients 1,4,8,16 --pg-port 5432
```

### Bulk load and engine write path

| Path | Throughput | What it measures |
|------|-----------|------------------|
| `COPY FROM STDIN` (wire) | 129,368 rows/s, 11.2 MiB/s | bulk ingest, chunked group commit |
| `put_sync` (engine, 1 fsync/row) | 24,517 rows/s (40.8 µs/row) | strict per-row durability |
| `put_batch_sync` (engine, 1 fsync/batch) | ~1.9M rows/s (0.5 µs/row) | in-memory path, amortized fsync |

```bash
TMPDIR=/mnt/nvme/tmp cargo run --release -p galaxdb-benchmarks --bin copy-bench -- --rows 5000
TMPDIR=/mnt/nvme/tmp cargo run --release -p galaxdb-benchmarks --bin engine-microbench -- --max 4000
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
