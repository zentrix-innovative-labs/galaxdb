# GalaxDB Storage Engine

An AI-native storage engine that unifies transactional row storage, columnar analytics, and vector similarity search into a single Rust binary.

**Language:** Rust (edition 2024) | **License:** Apache 2.0

---

## What It Does

GalaxDB's storage engine handles everything between "application writes a row" and "bits hit NVMe":

1. **Sub-microsecond point reads** via an Adaptive Radix Tree primary key index
2. **250K+ write TPS** with group-commit WAL and concurrent memtable
3. **4+ GB/s analytical scans** with columnar PAX blocks and zone-map pruning
4. **Zero data loss on crash** — WAL replay, checksum verification, atomic compaction
5. **Encryption at rest** — AES-256-GCM on every block and WAL record
6. **Write stall prevention** — auto-tuned RateLimiter + WriteController

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Write Path                         │
│  Client → WAL (LZ4 + XXH3-64) → Memtable (SkipMap) │
│           ↓ (>1KB values)                            │
│         Blob Log (content-addressed)                 │
├─────────────────────────────────────────────────────┤
│                   Flush Path                         │
│  Sealed Memtable → PAX Blocks → SST Files → NVMe    │
│  (AES-256-GCM encryption before write)               │
├─────────────────────────────────────────────────────┤
│                   Read Path                          │
│  ART Index → Bloom Filter → Buffer Pool → PAX Block  │
│  (HotSet for OLTP, ScanBuffer for OLAP)              │
├─────────────────────────────────────────────────────┤
│                   Background                         │
│  Lazy Leveling Compaction + MVCC GC                  │
│  RateLimiter (token bucket) + WriteController        │
│  Disk-full handler (32MB reserve file)               │
└─────────────────────────────────────────────────────┘
```

---

## Performance

All numbers measured on AWS c6id.4xlarge (Intel Xeon Platinum 8375C, 16 vCPU, 32 GiB RAM, 884 GB NVMe), Ubuntu 24.04, release build. See [BENCHMARKS.md](BENCHMARKS.md) for full details and reproduction commands.

### OLTP (1M rows, 16 threads, 60 s)

| Metric | GalaxDB | RocksDB | PostgreSQL 16 | SQLite |
|--------|---------|---------|---------------|--------|
| **Write TPS** | **258,555** | ~80,000 | ~3,200 | ~50,000 |
| **Read p50** | **3 µs** | ~180 µs | ~95 µs | ~50 µs |
| **Read p99** | **47 µs** | ~500 µs | ~300 µs | ~200 µs |
| **Write p99** | **377 µs** | 1–10 s* | — | — |

*RocksDB without write pacing (vLSM, arXiv 2024)

### OLAP Column Scan (10M rows, 16 threads, 60 s)

| Metric | GalaxDB | PostgreSQL 16 | DuckDB |
|--------|---------|---------------|--------|
| **Scan throughput** | **4.49 GB/s** | ~0.9 GB/s | ~5–10 GB/s |
| **Zone-map skip rate** | **80%** | N/A (heap scan) | similar |

### Mixed OLTP + OLAP (concurrent, 60 s)

| Metric | Result |
|--------|--------|
| OLTP p99 during concurrent scan | **191 µs** — no degradation |
| HotSet evictions from scan storm | **0** — ScanBuffer isolation works |

### Crash Safety

| Scenario | Result |
|----------|--------|
| Kill mid-flush → WAL replay | ✅ Zero data loss |
| Kill mid-compaction → old blocks intact | ✅ Zero data loss |
| Corrupt WAL record → replay stops at corruption | ✅ No silent corruption |
| Disk full → clean checkpoint → writes blocked | ✅ Reads continue |
| 100 concurrent writers | ✅ 0 duplicates, 0 missing |
| OLAP scan during OLTP | ✅ 0 HotSet evictions |

---

## How GalaxDB Compares

### vs PostgreSQL + pgvector

PostgreSQL is a general-purpose relational database. pgvector is a bolt-on extension for vector similarity search.

| Capability | GalaxDB | PostgreSQL + pgvector |
|---|---|---|
| SQL queries | ✅ Full AuroraSQL | ✅ Standard SQL |
| Vector search | ✅ HNSW built-in, recall@10=0.990 | ⚠️ pgvector (slower, lower recall) |
| Embeddings | ✅ Local model, no API cost | ❌ External API required |
| Time-travel queries | ✅ `AT VERSION` | ❌ |
| Training export | ✅ Lance format, one SQL command | ❌ |
| Near-dedup | ✅ MinHash LSH | ❌ |
| Write throughput | **258K TPS** | ~3.2K TPS |
| Scan throughput | **4.49 GB/s** | ~0.9 GB/s |
| Embedded mode | ✅ (like SQLite) | ❌ |

### vs Pinecone / Weaviate

Dedicated vector databases. No SQL, no training export, no time-travel.

| Capability | GalaxDB | Pinecone | Weaviate |
|---|---|---|---|
| SQL queries | ✅ | ❌ | Partial |
| Vector search | ✅ HNSW | ✅ | ✅ |
| Embeddings | ✅ Local | ❌ External | ✅ |
| Time-travel | ✅ | ❌ | ❌ |
| Training export | ✅ Lance | ❌ | ❌ |
| Self-hosted | ✅ | ❌ | ✅ |
| Wire protocol | PostgreSQL | REST | REST/gRPC |

### vs DuckDB

DuckDB is an excellent analytical database. It doesn't do vector search, embeddings, or training exports.

| Capability | GalaxDB | DuckDB |
|---|---|---|
| OLAP scan | 4.49 GB/s | ~5–10 GB/s |
| OLTP writes | **258K TPS** | ~50K TPS |
| Vector search | ✅ | ❌ |
| Embeddings | ✅ | ❌ |
| Training export | ✅ | ❌ |
| Embedded mode | ✅ | ✅ |

DuckDB wins on pure analytical throughput. GalaxDB wins on the full AI/ML workload.

---

## The 12 Components

### 1. PAX Block Format

Column-oriented storage blocks. Every byte of GalaxDB data lives in this format.

```
┌─────────────────────────────────────────┐
│ Header: magic (0x47414C41), block_id,   │
│   row_count, column descriptors,        │
│   zone maps (min/max per column)        │
├─────────────────────────────────────────┤
│ Column 0: FastPFOR (delta + bitpack)    │
│ Column 1: Zstandard L3                  │
│ Column 2: Raw (embeddings)              │
├─────────────────────────────────────────┤
│ Row Offset Table                        │
├─────────────────────────────────────────┤
│ Footer: XXH3-64 checksum               │
└─────────────────────────────────────────┘
```

- Fixed-width integers: delta encoding + bit-packing (FastPFOR) — ~4× compression
- Variable-width (TEXT, BLOB): Zstandard level 3 — ~3–5× compression
- Embedding columns: no compression (quantization handles it)
- XXH3-64 checksum verified on every read. Corrupt blocks rejected immediately.

**Reference:** PAX layout from Ailamaki et al. (VLDB 2001). Zone maps from Netezza/Redshift.

### 2. Write-Ahead Log

```
[type: u8][seq_no: u64][length: u32][xxh3_checksum: u64][lz4_payload]
```

- 6 record types: ROW_PUT, ROW_DELETE, DELTA_INSERT, DELTA_TOMBSTONE, CHECKPOINT, BLOB_REF
- Group commit: batches writes over 10 ms window, single fsync per batch → 250K+ TPS
- DURABILITY STRICT: fsync per commit → ~5K TPS (for financial workloads)
- Recovery: replay from last checkpoint, verify XXH3-64 per record, stop at first corruption

**Reference:** PostgreSQL WAL design with XXH3-64 replacing CRC-32 (3× faster hashing).

### 3. Memtable (crossbeam-skiplist, 16-shard)

Lock-free concurrent skip map with per-key MVCC version chains.

- 16 shards via `xxh3_64(key) % 16` — eliminates cross-shard contention
- Seal at 64 MB — atomically swap to new empty memtable
- Back-pressure at 256 MB — block writers when sealed-but-unflushed bytes exceed limit

**Reference:** Wang et al., "Building a Bw-Tree Takes More Than Just Buzz Words" (SIGMOD 2018).

### 4. ART Primary Key Index

Adaptive Radix Tree (Leis et al., ICDE 2013) with Node4/Node16/Node48/Node256 and path compression.

- 213 ns/lookup (sequential keys, warm cache)
- 752 ns/lookup (random keys, warm cache)
- O(k) lookup where k is key length, independent of dataset size

**Reference:** Leis et al., "The Adaptive Radix Tree" (ICDE 2013).

### 5. Bloom Filters with Monkey Allocation

Per-SST Bloom filters with false-positive rates optimally allocated across LSM levels.

```
FPR(level_i) = budget × (ratio^(L-i)) / Σ(ratio^(L-j))
```

Larger, colder levels get more bits per key. 40–80% fewer false positives than fixed allocation at the same memory budget.

**Reference:** Dayan et al., "Monkey: Optimal Navigable Key-Value Store" (ACM TODS 2018).

### 6. NUMA-Aware Buffer Pool

Dual-region buffer pool with NUMA-local allocation.

- HotSet (70% RAM): LRU eviction for OLTP point lookups
- ScanBuffer (30% RAM): Clock-sweep eviction for OLAP sequential scans
- Isolation guarantee: ScanBuffer NEVER evicts a HotSet-resident block

This is why OLTP p99 doesn't degrade during concurrent OLAP scans.

**Reference:** PostgreSQL 18 NUMA analysis (EnterpriseDB). Clock-sweep from FreeBSD VM subsystem.

### 7. Lazy Leveling Compaction + MVCC GC

LSM-tree with tiered upper levels and leveled bottom level (Dostoevsky design).

- L0: tiered (up to 4 files before compaction trigger)
- L1–L3: tiered (multiple sorted runs per level)
- L4 (bottom): leveled (single sorted run)
- MVCC GC during merge: discard versions not needed by active snapshots or pinned tags

**Reference:** Dayan & Idreos, "Dostoevsky" (SIGMOD 2018).

### 8. KV Separation at WAL Time

Values > 1 KB written to a content-addressed blob log during WAL construction.

- Multi-queue parallel writers (4 queues, round-robin)
- Content addressing: XXH3-128 hash → deduplication
- Transparent read: blob references detected and fetched automatically
- GC: compact blob files when discardable space > 50%

**Reference:** Li et al., "BVLSM" (arXiv 2025). WiscKey (Lu et al., FAST 2016).

### 9. AES-256-GCM Encryption (TDE)

Every PAX block and WAL record encrypted before hitting storage.

- Pluggable key management via `KeyProvider` trait — no vendor lock-in:
  - `LocalKeyProvider` — 32-byte key file
  - `EnvKeyProvider` — hex key from `GALAXDB_MASTER_KEY` env var
  - `ExternalCommandKeyProvider` — delegate to any shell command (AWS CLI, gcloud, az, vault CLI, custom HSM)
  - `HashicorpVaultKeyProvider` — Vault Transit engine over rustls
- Counter-based 96-bit nonces: 4-byte random prefix + 8-byte atomic counter
- AES-NI accelerated: 680 MB/s encrypt, 709 MB/s decrypt

### 10. Write Stall Mitigation

**RateLimiter (token bucket):**
- Calibrates to 70% of NVMe write bandwidth at startup
- Lowers ceiling by 30% when HP-queue P99 exceeds 1.5× baseline for 3 consecutive windows
- SILK-style flush pre-emption: flush I/O takes priority over compaction under back-pressure

**WriteController (admission control):**
- Soft limit (32 GB pending compaction): proportional slowdown
- Hard limit (64 GB): block all writes until compaction catches up

**Reference:** vLSM (Xanthakis et al., arXiv 2024). SILK (Balmau et al., USENIX ATC 2019).

### 11. Disk-Full Handling

- 32 MB reserve file pre-allocated at startup
- On disk-full: delete reserve → clean checkpoint → block writes → reads continue
- Recovery: operator frees space → reserve recreated → writes resume
- Zero data corruption — all committed data safe

### 12. Statistics Collection

Background ANALYZE with HyperLogLog NDV estimation, equi-height histograms, and multi-column correlation statistics. Used by the adaptive query planner for HNSW-vs-brute-force decisions.

**Reference:** PostgreSQL extended statistics. HyperLogLog (Flajolet et al., 2007).

---

## Research Basis

| Decision | Paper | Year | Venue |
|----------|-------|------|-------|
| ART primary key index | Leis et al., "The Adaptive Radix Tree" | 2013 | ICDE |
| Monkey Bloom allocation | Dayan et al., "Monkey" | 2018 | ACM TODS |
| Lazy Leveling compaction | Dayan & Idreos, "Dostoevsky" | 2018 | SIGMOD |
| crossbeam-skiplist (not Bw-Tree) | Wang et al., "Building a Bw-Tree..." | 2018 | SIGMOD |
| KV separation at WAL time | Li et al., "BVLSM" | 2025 | arXiv |
| Write stall mitigation | Xanthakis et al., "vLSM" | 2024 | arXiv |
| SILK flush pre-emption | Balmau et al., "SILK" | 2019 | USENIX ATC |
| KV separation (original) | Lu et al., "WiscKey" | 2016 | FAST |
| RaBitQ quantization | Gao et al. | 2024 | SIGMOD |
| PAX block format | Ailamaki et al. | 2001 | VLDB |
| HyperLogLog NDV | Flajolet et al. | 2007 | — |
| Snapshot Isolation | Berenson et al. | 1995 | SIGMOD |
| MinHash LSH | Broder | 1997 | — |

---

*Every number is reproducible. Every design decision has a citation. See [BENCHMARKS.md](BENCHMARKS.md) for reproduction commands.*
