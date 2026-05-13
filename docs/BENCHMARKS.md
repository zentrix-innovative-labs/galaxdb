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

## Storage Engine

**Hardware:** AWS c6id.4xlarge, NVMe storage.

### Write throughput

| Workload | Throughput | Durability |
|----------|-----------|------------|
| OLTP — 16 threads, 1M rows, 60 s | **258,555 TPS** | Relaxed (group commit) |
| Embedded INSERT — batched 100/stmt | **20,267 rows/sec** | Relaxed |
| Wire INSERT — 4 clients | **454 rows/sec** | Strict |

### Read latency (warm cache)

| Metric | Value |
|--------|-------|
| Point read p50 | 3 µs |
| Point read p99 | 47 µs |

### Read latency (cold cache, 50M rows × 600 B ≈ 30 GB)

| Metric | Value |
|--------|-------|
| Missing keys | 0 / 100,000 |
| Read p50 | 147 µs |
| Read p99 | 308 µs |

### OLAP scan throughput

| Metric | Value |
|--------|-------|
| Scan throughput | 4.49 GB/s |
| Zone-map skip rate | 80% |

### Mixed OLTP + OLAP (HotSet/ScanBuffer isolation)

| Metric | Value |
|--------|-------|
| OLTP p99 during concurrent scan | 191 µs |
| p99 degradation vs OLTP-alone | 0% |
| HotSet evictions caused by scan | 0 |

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

## Competitive Comparison

A direct comparison against hnswlib on the same hardware and dataset is in progress. The harness is at `benchmarks/tools/hnswlib_recall.py`. Numbers will be published here once both systems have been run on the same machine against the same SIFT-1M dataset.

---

*Every number in this document was measured on real hardware with `cargo run/bench --release`. Commands are shown inline. No synthetic or estimated numbers are published.*
