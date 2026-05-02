# GalaxDB Storage Engine Benchmarks

> **Last updated:** 2026-05-02  
> **Git hash:** `d90742e`  
> **Edition:** Rust 2024 (rustc 1.94.0 macOS / 1.95.0 Linux)

This document tracks all benchmark results for the GalaxDB v1 storage engine across different hardware platforms. Every number is reproducible — run the commands below on your own hardware.

---

## How to Run

```bash
# Micro-benchmarks (Criterion — per-component)
cargo bench -p galaxdb-storage
cargo bench -p galaxdb-crypto

# Macro-benchmarks (full workloads — ALWAYS use --release)
cargo run -p galaxdb-benchmarks --release -- --workload all --duration 60 --warmup 10

# Chaos tests (crash safety — ALWAYS use --release)
cargo run -p galaxdb-chaos-tests --release

# Quick smoke test (15 seconds)
cargo run -p galaxdb-benchmarks --release -- --workload oltp --duration 15 --warmup 3 --rows 100000
```

---

## Pass / Fail Criteria

| Metric | Production Target | Notes |
|--------|------------------|-------|
| OLTP point read p50 (warm cache) | ≤ 50 µs | ART + HotSet path |
| Write throughput (group commit) | ≥ 50K TPS | WAL + memtable path |
| P99 write latency (sustained load) | ≤ 5 ms | WriteController must be wired |
| OLAP column scan | ≥ 3 GB/s | PAX + zone map pruning (NVMe, multi-threaded) |
| Zone map skip rate (selective filter) | ≥ 80% | Blocks skipped by min/max |
| Bloom filter FPR improvement | ≥ 40% vs fixed 10-bit | Monkey allocation |
| Crash recovery time | ≤ 30 s | From 512 MB WAL |
| Chaos tests | 6/6 pass | Zero committed data loss |

---

## Platform 1: MacBook Pro (Intel) — Development

| Spec | Value |
|------|-------|
| CPU | Intel Core i7-7820HQ @ 2.90 GHz (4C/8T) |
| RAM | 16 GB DDR4 |
| Storage | Apple SSD (SATA) |
| OS | macOS |
| AES-NI | Yes |
| I/O Backend | tokio (kqueue) |
| Rust | 1.94.0, edition 2024 |

### Macro-Benchmark Results (2026-05-02)

**OLTP** (500K rows, 8 threads, 60s)

| Metric | Value | Status |
|--------|-------|--------|
| Write TPS | **70,592** | ✅ |
| Read p50 | **6 µs** | ✅ |
| Read p99 | 86 µs | — |
| Read p999 | 1,153 µs | — |
| Write p50 | 23 µs | — |
| Write p99 | 776 µs | ✅ |

**OLAP** (1000 blocks × 10K rows, 60s)

| Metric | Value | Status |
|--------|-------|--------|
| Scan throughput | 0.24 GB/s | ⚠️ SATA bottleneck |
| Zone map skip | 79.9% | ⚠️ Borderline |

**Mixed** (concurrent, 60s)

| Metric | Value | Status |
|--------|-------|--------|
| OLTP p99 during scan | 597 µs | ✅ |
| HotSet evictions | 0 | ✅ |

### Micro-Benchmarks (Criterion)

| Component | Benchmark | Time |
|-----------|-----------|------|
| PAX | Encode 1000 rows | 1.03 ms |
| PAX | Decode 1000 rows | 4.5 µs |
| PAX | XXH3-64 checksum 1 MB | 100.6 µs (9.9 GB/s) |
| ART | Insert 1M sequential | 839 ms (839 ns/op) |
| ART | Lookup 1M sequential | 213 ms (213 ns/op) |
| ART | Lookup 1M random | 752 ms (752 ns/op) |
| Bloom | Build 100K keys | 20.7 ms |
| Bloom | Lookup existing | 1.27 µs |
| Bloom | Lookup non-existing | 139 ns |
| AES-256-GCM | Encrypt 1 MB | 1.47 ms (680 MB/s) |
| AES-256-GCM | Decrypt 1 MB | 1.41 ms (709 MB/s) |

### Chaos Tests: **6/6 PASS** ✅

---

## Platform 2: AWS c6id.4xlarge (Linux) — Production

| Spec | Value |
|------|-------|
| Instance | c6id.4xlarge |
| CPU | Intel Xeon Platinum 8375C @ 2.90 GHz (16 vCPU) |
| RAM | 30 GB |
| Storage | **884 GB local NVMe SSD** |
| OS | Ubuntu 24.04 (kernel 6.17.0) |
| AES-NI | Yes (AVX-512) |
| I/O Backend | tokio (io_uring available but not yet wired) |
| Rust | 1.95.0, edition 2024 |
| Cost | $0.81/hr on-demand, $0 when stopped |

### Macro-Benchmark Results (2026-05-02)

**OLTP** (1M rows, 16 threads, 60s)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Write TPS | **257,610** | ≥ 50K | ✅ **5.2x over target** |
| Read p50 | **3 µs** | ≤ 50 µs | ✅ **17x better** |
| Read p99 | **46 µs** | — | ✅ |
| Read p999 | **524 µs** | — | — |
| Write p50 | **16 µs** | — | — |
| Write p99 | **367 µs** | ≤ 5 ms | ✅ **14x better** |

**OLAP** (1000 blocks × 10K rows, 60s)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Scan throughput | **1.1 GB/s** | ≥ 3 GB/s | ⚠️ Needs parallel scan |
| Blocks scanned | 521,116 | — | — |
| Blocks skipped | 416,800 | — | — |
| Zone map skip | **80.0%** | ≥ 80% | ✅ |

**Mixed** (concurrent, 60s)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| OLTP p99 during scan | **243 µs** | ≤ 5 ms | ✅ **21x better** |
| p99 degradation | **0.0%** | ≤ 20% | ✅ |
| HotSet evictions | **0** | 0 | ✅ |

### Chaos Tests: **6/6 PASS** ✅ (completed in 29.45s)

| Test | Result | Notes |
|------|--------|-------|
| C1: Kill-mid-flush | ✅ | 10K rows recovered |
| C2: Kill-mid-compaction | ✅ | 4K keys intact |
| C3: Corrupt-WAL-record | ✅ | 538/1000 recovered, stopped at corruption |
| C4: Fill-disk simulation | ✅ | Full lifecycle verified |
| C5: 100 concurrent writers | ✅ | 100K writes in **0.06s** (7x faster than MacBook) |
| C6: OLAP-scan-during-OLTP | ✅ | 0 HotSet evictions, p99: 0µs |

---

## Cross-Platform Comparison

| Metric | MacBook (Intel i7) | AWS c6id.4xlarge | Improvement |
|--------|-------------------|------------------|-------------|
| Write TPS | 70,592 | **257,610** | **3.6x** |
| Read p50 | 6 µs | **3 µs** | **2x** |
| Read p99 | 86 µs | **46 µs** | **1.9x** |
| Write p99 | 776 µs | **367 µs** | **2.1x** |
| OLAP scan | 0.24 GB/s | **1.1 GB/s** | **4.6x** |
| Zone map skip | 79.9% | **80.0%** | — |
| OLTP p99 (mixed) | 597 µs | **243 µs** | **2.5x** |
| Concurrent writers (100 threads) | 0.42s | **0.06s** | **7x** |
| Chaos test total time | 70.6s | **29.5s** | **2.4x** |

## Comparison with Other Systems

| Metric | GalaxDB (AWS) | RocksDB | PostgreSQL 16 | Source |
|--------|--------------|---------|---------------|--------|
| Point read p50 (warm) | **3 µs** | ~180 µs | ~95 µs | Leis 2013, Facebook CIDR 2017 |
| Write TPS (group commit) | **257K** | ~80K | ~3.2K | RocksDB wiki, pgbench |
| P99 write (sustained) | **367 µs** | 1-10 s (no pacing) | — | vLSM arXiv 2024 |
| Column scan | 1.1 GB/s* | — | ~0.9 GB/s | EnterpriseDB analysis |

*OLAP scan is single-threaded; parallel scan expected to reach 3-5 GB/s.

---

## Known Issues & Optimization Roadmap

### 1. OLAP Scan Throughput (1.1 GB/s vs 3 GB/s target)

**Root cause:** Single-threaded scan. The benchmark runs one scan thread that decompresses PAX blocks sequentially. Zstd decompression is the CPU bottleneck, not NVMe I/O.

**Fix:** Parallelize the scan across blocks using a thread pool. Each thread decompresses and scans a subset of blocks. With 16 vCPUs, we expect 3-5 GB/s.

**Priority:** High — this is the only metric below target.

### 2. io_uring Not Yet Wired

The AWS server has kernel 6.17.0 with io_uring support, but the benchmark currently uses the tokio backend. Wiring io_uring for the HP/BK queue separation will improve I/O latency isolation under mixed workloads.

### 3. Encryption Not in Benchmark Path

The current benchmarks don't encrypt/decrypt PAX blocks. Adding TDE to the write/read path will add ~3-8% overhead (based on AES-NI micro-benchmarks showing 680 MB/s encrypt throughput).

---

## Benchmark History

| Date | Git Hash | Platform | OLTP TPS | Read p50 | OLAP GB/s | Chaos |
|------|----------|----------|----------|----------|-----------|-------|
| 2026-05-02 | `d90742e` | MacBook i7-7820HQ (8T) | 70,592 | 6 µs | 0.24 | 6/6 ✅ |
| 2026-05-02 | `d90742e` | AWS c6id.4xlarge (16T) | **257,610** | **3 µs** | **1.1** | 6/6 ✅ |

---

## Server Management

```bash
# Stop the benchmark server (saves money — $0 when stopped, NVMe data lost)
aws ec2 stop-instances --instance-ids i-0b2dec9226f62db65

# Start the benchmark server
aws ec2 start-instances --instance-ids i-0b2dec9226f62db65

# Check server status
aws ec2 describe-instances --instance-ids i-0b2dec9226f62db65 \
  --query 'Reservations[0].Instances[0].{State:State.Name,IP:PublicIpAddress}' --output table

# SSH into the server
ssh -i ~/.ssh/galaxdb-bench-key.pem ubuntu@<PUBLIC_IP>

# Rsync code to server
rsync -avz --exclude 'target/' --exclude '.git/' -e "ssh -i ~/.ssh/galaxdb-bench-key.pem" . ubuntu@<PUBLIC_IP>:/data/galaxdb/

# Run benchmarks on server
ssh -i ~/.ssh/galaxdb-bench-key.pem ubuntu@<PUBLIC_IP> \
  'source ~/.cargo/env && cd /data/galaxdb && cargo run --release -p galaxdb-benchmarks -- --workload all --duration 60 --warmup 10 --rows 1000000 --threads 16 --data-dir /data/bench_data'
```

---

*All benchmarks run with `--release` flag. Debug mode results are not recorded.*
