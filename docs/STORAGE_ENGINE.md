# GalaxDB Storage Engine

An AI-native storage engine that unifies transactional row storage, columnar analytics, and vector similarity search into a single Rust binary.

**Language:** Rust (edition 2024) | **License:** Apache 2.0

---

## What It Does

GalaxDB's storage engine handles everything between "application writes a row" and "bits hit NVMe":

1. **O(k) point reads** via an Adaptive Radix Tree primary key index
2. **Durable writes competitive with PostgreSQL 16** on the same NVMe; ~1.9M rows/s in-memory path when fsync is amortized across a batch
3. **Columnar analytics** with PAX blocks and zone-map pruning
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

All numbers measured on AWS c6id.4xlarge (Intel Xeon Platinum 8375C, 16 vCPU, 32 GiB RAM, 884 GB NVMe), Ubuntu 24.04, release build. GalaxDB and PostgreSQL 16 run on the **same instance-store NVMe**, `synchronous_commit=on` / `fdatasync`, both with prepared statements. See [BENCHMARKS.md](BENCHMARKS.md) for full details and reproduction commands.

### Durable write path (concurrent INSERT vs PostgreSQL 16)

| Clients | GalaxDB | PostgreSQL 16 |
|---------|---------|---------------|
| 1  | 10,450 rows/s | 11,891 rows/s |
| 4  | 30,468 rows/s | 34,298 rows/s |
| 8  | 36,632 rows/s | 54,432 rows/s |
| 16 | 37,448 rows/s | 84,747 rows/s |

GalaxDB is competitive at low concurrency; PostgreSQL scales better past 8 clients. The gap is the
async server's per-query thread hand-off, not the storage engine — the engine's in-memory path
(`put_batch_sync`) sustains ~1.9M rows/s.

### Bulk load and engine micro-path

| Path | Throughput |
|------|-----------|
| `COPY FROM STDIN` (wire) | 190,287 rows/s (17.1 MiB/s) |
| `put_sync` (engine, 1 fsync/row) | 26,500 rows/s (37.7 µs/row) |
| `put_batch_sync` (engine, 1 fsync/batch) | ~1.9M rows/s (0.5 µs/row) |

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
| Durable write throughput | competitive (same NVMe, see above) | baseline |
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
- Pre-allocated, zero-filled WAL written in place (PostgreSQL segment model) so `fdatasync` flushes only dirty data pages (~37 µs on NVMe), not extent/inode metadata.
- Group commit batches concurrent writers into a single fsync; engine `put_sync` reaches 26,500 commits/s (one fsync per row), `put_batch_sync` ~1.9M rows/s (one fsync per batch).
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

- Warm-cache point lookups in the low hundreds of nanoseconds (O(k) in key length, independent of dataset size)
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
- AES-NI / AVX2 accelerated; see [BENCHMARKS.md](BENCHMARKS.md) for measured cipher throughput (`cargo bench -p galaxdb-crypto`)

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
