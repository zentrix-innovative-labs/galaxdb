# GalaxDB Architecture Specification  
## Final Version 4.2 — Production‑Ready, AI‑Native, Distributed, Self‑Improving  

**Status:** Design locked. All 27 research‑backed findings resolved.  
**Target:** v1 — 4 months, 2–3 Rust engineers. v2 — 12–18 months, expanded team.

---

## Table of Contents

1. [Vision & Design Principles](#1-vision--design-principles)  
2. [Architecture Overview](#2-architecture-overview)  
3. [v1 Core System](#3-v1-core-system)  
   - 3.1 Storage Engine  
   - 3.2 Vector Index & Quantization  
   - 3.3 Versioning & Semantic Search Semantics  
   - 3.4 Embedding Inference Sidecar & Model‑Version Tracking  
   - 3.5 AuroraSQL Language  
   - 3.6 PostgreSQL Wire Compatibility  
   - 3.7 Consistency Model  
   - 3.8 Training Data Path & AI Workloads  
   - 3.9 Deployment Modes & Platform Support  
   - 3.10 Binary Footprint & Module Tiers  
   - 3.11 Durability & Crash Recovery Contract  
   - 3.12 Transparent Data Encryption (TDE) & Security  
   - 3.13 Observability  
   - 3.14 Backup & Restore  
   - 3.15 v1 Limitations  
4. [v2 Full System](#4-v2-full-system)  
   - 4.1 RGABH‑Driven Adaptive Storage  
   - 4.2 Advanced Indexing (DiskANN, FreshDiskANN, SPANN)  
   - 4.3 Distributed Clustering & Global Transactions  
   - 4.4 Active Learning & Feedback Loop  
   - 4.5 Semantic Snapshot Guarantees  
   - 4.6 Full PostgreSQL Protocol & BI Integration  
   - 4.7 GPU‑Direct & Hardware Acceleration  
   - 4.8 Federated Queries & Privacy  
   - 4.9 Plugin Marketplace  
   - 4.10 Semantic Caching  
   - 4.11 Multi‑Tenancy  
5. [Implementation Roadmap](#5-implementation-roadmap)  
6. [Appendices](#6-appendices)

---

## 1. Vision & Design Principles

GalaxDB is the **AI‑native database** that unifies transactional, analytical, and vector workloads into a single engine. It eliminates the five‑database spaghetti and actively improves the AI built on top of it – including built‑in training data optimization, near‑duplicate detection, semantic caching, and zero‑copy model feeding.

**Non‑Negotiable Principles:**
- **Unified Data Atom** – relational fields, embeddings, binaries, and lineage in one row.  
- **Honest Semantics** – limitations are documented; silent incorrectness is never allowed.  
- **Small to Planet Scale** – same binary from laptop to million‑node cluster.  
- **AI‑First** – embeddings, versioned snapshots, feedback loops, and training‑aware optimizations are first‑class.  
- **Falsifiable Claims** – every performance number is stated with measurable conditions and reproducible benchmarks.

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                  AuroraSQL Language                  │
│       (PostgreSQL wire protocol + AI extensions)     │
├──────────────────────────────────────────────────────┤
│           Query Optimizer, Planner & Executor        │
├──────────────┬───────────────┬───────────────────────┤
│  LSM + PAX   │  Mutable ANN  │ Embedding Sidecar    │
│  Store       │ (mmap + delta │ (Unix Socket,         │
│              │   + SQ8)     │  persistent backlog)  │
├──────────────┴───────────────┴───────────────────────┤
│       io_uring I/O Scheduler (HP/BK queues)          │
│         [Linux only; tokio on macOS/Windows]         │
├──────────────────────────────────────────────────────┤
│  Storage (NVMe, blob store, object store)            │
└──────────────────────────────────────────────────────┘
```
v1 implements all layers above as a single‑node embedded/standalone system.  
v2 adds distributed clustering, RGABH adaptive tiering, and hardware acceleration.

---

## 3. v1 Core System

### 3.1 Storage Engine

#### 3.1.1 LSM‑Tree with PAX Blocks
- **Write path**: lock‑free `crossbeam‑skiplist‑mvcc` memtable (replaces Bw‑Tree, per CMU SIGMOD 2018 correctness analysis). The crate provides per‑key version chains with atomic read‑check‑write operations, eliminating MVCC race conditions. Fallback: 16‑shard per‑key `Mutex` wrapper around `crossbeam‑skiplist`.  
  **Implementation guideline:** Values read from the memtable must be copied out of the entry handle immediately, and the handle dropped before any async operation. Holding entry handles across `.await` boundaries prevents epoch‑based memory reclamation from freeing garbage entries from concurrent writes. A clippy lint enforces this rule.
- **Flush:** Buffer sealed at 64 MB, backpressure at 256 MB.
- **Compaction**: **Lazy Leveling** (upper levels tiered, bottom level leveled). Compaction integrates **MVCC garbage collection** – retains only versions needed by oldest active snapshot or pinned tag.
- **Read path**: ART (Adaptive Radix Tree) primary key index (Leis et al., ICDE 2013); Bloom filters with **Monkey allocation** (Dayan et al., TODS 2018) minimize false‑positive sum across levels; zone‑map pruning per block.
- **Buffer pool**: partitioned into **HotSet** (70 % RAM, LRU) and **ScanBuffer** (30 %, clock‑sweep). **NUMA‑aware** – worker threads allocate buffer frames from the local NUMA node. The HNSW delta buffer is allocated on the NUMA node closest to the GPU if GPU training is enabled.
- **Compression** (§3.1.2): Fixed‑width columns → delta + bit‑packing (FastPFOR); variable‑width → Zstandard level 3; embeddings not further compressed (quantization already handles size); WAL → LZ4.
- **KV separation** (§3.1.3): Values > 1 KB stored in content‑addressed blob log; PAX blocks hold only 32‑byte hash. KV separation occurs at **WAL write time**, not flush time. Blob garbage collection runs when discardable space ratio exceeds 50 %.
- **Statistics** (§3.1.7): Background `ANALYZE` collects per‑column NDV, equi‑height histogram, null fraction, plus **multi‑column correlation statistics** (PostgreSQL extended statistics model). Used by planner for filter selectivity and HNSW‑vs‑brute choice.

#### 3.1.2 KV Separation at WAL Time (BVLSM)
For values exceeding the 1 KB threshold, the value is written directly to the blob log during WAL entry construction. The in‑memory skiplist stores only the 32‑byte content hash and blob offset. This prevents large values from consuming the 256 MB back‑pressure budget and eliminates repeated writes of large values during compaction. The blob log uses multi‑queue parallel writes, following the BVLSM design (Li et al., arXiv 2025). Garbage collection compacts blob files when the discardable space ratio exceeds 50 %.

#### 3.1.3 Compaction Write Stall Mitigation

**Problem.** Under sustained write load, LSM compaction can fall behind. RocksDB’s default write‑stalls block all user writes, producing P99 latencies in the order of seconds even at moderate throughput (Xanthakis et al., arXiv 2024).

**GalaxDB’s three‑mechanism design:**

1. **Auto‑tuned RateLimiter** – Controls aggregate compaction + flush I/O bandwidth. Uses an auto‑tuned token‑bucket limiter that dynamically adjusts the allowable compaction write rate based on observed compaction debt and write pressure. The upper bound is configurable (default: 70 % of measured NVMe write bandwidth, calibrated at startup). During OLTP peak load, if the io_uring HP‑queue latency exceeds 1.5× the idle baseline for three consecutive 100 ms measurement windows, the limiter ceiling is temporarily lowered by 30 %.

2. **WriteController – User‑Write Throttle** – Manages whether incoming user writes are accepted at full speed, gradually slowed, or stopped entirely, using two configurable thresholds:
   - `soft_pending_compaction_bytes_limit` (default 32 GB) → writes **slowed** to `delayed_write_rate` (16 MB/s).
   - `hard_pending_compaction_bytes_limit` (default 64 GB) → writes **stopped** until pending bytes fall below the hard limit.  
   The WriteController operates on 1 ms intervals, applying gradual slowdown proportional to the excess.

3. **vLSM Structural Improvements (Month 4 hardening)** – Smaller SST size (8 MB default, configurable; 64 MB in Month 1, tuned in Month 4), combined with elimination of tiering compaction at L0 and SILK‑style dynamic bandwidth pre‑emption for flush operations. These changes reduce cumulative write stalls by up to 60 % and improve P99 latency by orders of magnitude (vLSM, arXiv 2024; SILK, USENIX ATC 2019).

---

### 3.2 Vector Index & Quantization

#### 3.2.1 Mutable ANN
Architecture: **mmap’d HNSW base graph + WAL‑backed delta buffer (exact k‑NN)**.
- Inserts go to delta buffer (same WAL as LSM, record type `DELTA_INSERT`).
- Query searches both base graph and delta buffer, union + re‑rank.
- Merge when delta exceeds `max(10k, 1% of total_indexed)`. Merge uses **atomic rename** (shadow file + `rename()`) — crash‑safe, no downtime.
- Deletes write tombstones; emergency merge if tombstones > 20 % of indexed vectors.
- Filter‑aware traversal: v1 uses adaptive planner fallback (brute‑force when filter cardinality very low); v2 adds ACORN‑style in‑graph filtering.
- **Crash safety**: delta buffer replayed from WAL in batches of 1,000 on recovery; base graph intact due to atomic rename.

#### 3.2.2 Quantization
**Platform‑aware quantization defaults:**
- **x86‑64 with AVX2 / AVX‑512** → SQ8 (int8 scalar). 4× compression, SIMD‑accelerated, no training.
- **ARM64 (Apple Silicon, Graviton)** → **FP16** (half‑precision float) as default. 2× compression, natively accelerated on ARM NEON. SQ8 available as opt‑in; throughput approximately 3× lower than AVX2.
- **RaBitQ** (32× compression, random rotation + binary quantization) available as opt‑in on both platforms; ARM64 port uses NEON‑optimized kernels and FP16 precision.
- **Binary quantization** (32× compression, Hamming distance via popcount) for latency‑critical workloads.  
**PQ completely removed** — RaBitQ dominates it at every compression ratio (Gao et al., SIGMOD 2024/2025).

---

### 3.3 Versioning & Semantic Search Semantics

- **Merkle DAG:** every write creates a PAX block with commit timestamp. Merkle tree over block hashes gives a version root. `AT VERSION` queries filter blocks. Named tags are GC‑exempt (pinned blocks).
- **Three consistency modes** for `AT VERSION` + `SEMANTIC_MATCH`:
  - `ROW_SNAPSHOT` (default) — historical row data, **rejects** `SEMANTIC_MATCH`.
  - `SEMANTIC_FRESH` — opt‑in; current HNSW index against historical rows. Warning in result metadata.
  - `SEMANTIC_SNAPSHOT` (v2) — versioned index for exact historical vector search.
- **Training reproducibility**: tags with `FOR TRAINING` guarantee deterministic block order (primary key sort) and store a shuffle seed.

---

### 3.4 Embedding Inference Sidecar & Model‑Version Tracking

- **Sidecar**: standalone Rust binary (ONNX Runtime), Unix socket.
- **Lifecycle**: parent PID monitoring (platform‑specific), heartbeat, exponential backoff on crash.
- **Back‑pressure**: 10k in‑flight queue; overflow stored in persistent backlog table (`_galaxdb_embedding_backlog`). Backlog uses `DURABILITY STRICT` regardless of session setting.
- **Model‑version tracking**: each embedded row carries `_embedding_model_version`. When sidecar model changes, all rows with old version are marked stale and re‑embedded via backlog. `SHOW EMBEDDING HEALTH` reports version distribution and progress.

---

### 3.5 AuroraSQL Language

PostgreSQL simple query protocol extended with:
- `CREATE TABLE ... (col TEXT EMBEDDING MODEL 'name' DIM ...)`
- `SEMANTIC_MATCH(col, 'query', threshold)`
- `AT VERSION timestamp/tag` with consistency modes
- `FEEDBACK ... SET ... SOURCE 'model'` (v2)
- `ORDER BY ACTIVE_LEARNING(target)` (v2)
- `CREATE VERSION TAG 'name' [FOR TRAINING [WITH TRAINING PRECISION 'sq8'|'rabitq'|'float32'] [TRAINING SEED n]]`
- `BULK INSERT` for direct PAX‑block writes
- `SHOW EMBEDDING HEALTH` (model version distribution)

---

### 3.6 PostgreSQL Wire Compatibility (Tier 1)

- Simple query protocol (`Q` message).
- Basic DDL/DML.
- `pg_catalog` stubs sufficient for `psycopg2` and SQLAlchemy (simple mode).
- Extended protocol, `COPY`, full `information_schema` → v2.

---

### 3.7 Consistency Model

- **Snapshot Isolation** (SI) for row data — no dirty reads, no non‑repeatable reads, no phantoms. Write‑skew anomalies possible.
- **Eventually fresh semantic index** — `_embedding_stale` flag reliable because embedding writes follow standard LSM update path.
- v2 adds **Serializable Snapshot Isolation** (SSI) via anti‑dependency tracking (Cahill et al., SIGMOD 2008).

---

### 3.8 Training Data Path & AI Workloads

- **Lance materialization**: `FOR TRAINING` exports Lance‑format dataset; Python API `galaxdb.training_dataset(tag)` → PyTorch `IterableDataset`, zero‑copy.
- **Training precision**: `float32`, `sq8`, `rabitq` — reduces I/O volume 4–32×.
- **Near‑dedup**: MinHash LSH signatures computed in Rust on write path (128‑hash, 512 B). `WHERE NOT DUPLICATE` filter excludes near‑duplicates; background job periodically refreshes groups.
- **Curriculum learning**: `ORDER BY ACTIVE_LEARNING` (v2) combined with training export.
- **Lineage**: system table `_galaxdb_training_exports` records every export (tag, filter, precision, dedup flag, curriculum mode, row count, export timestamp, hash). Compliant with EU AI Act Article 13.

---

### 3.9 Deployment Modes & Platform Support

| Mode | Platforms | Production? | I/O Backend |
|------|-----------|-------------|-------------|
| Embedded | Linux 5.10+, macOS (ARM64/x86-64), Windows (x86-64) | Linux only | io_uring (Linux), tokio (macOS, Windows) |
| Standalone server | Linux, macOS | Linux only | io_uring (Linux), tokio (macOS) |
| Clustered | v2 only | Linux 5.10+ | io_uring |

**Production deployment target:** Linux kernel 5.10+ (LTS), io_uring enabled, `seccomp=unconfined` for the GalaxDB process. The io_uring HP/BK queue scheduler, P99 latency guarantees, and NVMe bandwidth saturation benchmarks apply only to Linux production deployments. macOS and Windows are **development and testing platforms**; they use tokio's native async I/O (`kqueue`/`IOCP`) and do not provide the HP/BK queue isolation guarantees.

**Connections:** async Rust tasks (tokio). Max connections configurable (default 1,000). Queue‑and‑reject above threshold. No external pooler required for most workloads.

---

### 3.10 Binary Footprint & Module Tiers

- Core engine: < 70 MB (Rust statically linked).
- Full (sidecar + default model): < 350 MB.
- Tiers: minimal (core), standard (core + Python), full (+ sidecar).

---

### 3.11 Durability & Crash Recovery Contract

- **WAL fsync**: group commit default (10 ms); `DURABILITY STRICT` / `RELAXED` per‑connection.
- **Checkpoint**: every 60 s or when WAL exceeds 512 MB. Recovery < 30 s.
- **Block integrity**: XXH3‑64 checksum + magic number `0x47414C41`.
- **WAL record header**: `[type][seq_no][length][xxh3_checksum][payload]`. Replay skips corrupt records; stops at first checksum failure.
- **Disk full**: 32 MB reserve file; write blocking after clean checkpoint.

---

### 3.12 Transparent Data Encryption (TDE) & Security

- AES‑256‑GCM encryption of every PAX block and all WAL records.
- AES‑NI hardware acceleration (3‑8 % CPU overhead).
- Key management via AWS KMS (v1); additional providers in v2.
- TLS 1.3 for all wire protocol connections.
- **io_uring Security Note:** io_uring has been the source of security vulnerabilities in containerised and sandboxed environments. In Docker default seccomp profiles, gVisor, or AWS Lambda, io_uring may be blocked. Set `GALAXDB_IO_BACKEND=tokio` to use standard async I/O. Performance guarantees are void in this mode. Production deployments should use dedicated instances with `seccomp=unconfined` for the GalaxDB process.
- Compliance: GDPR, HIPAA, SOC 2 ready.

---

### 3.13 Observability

- Embedded HTTP server (`/health`, `/metrics` in Prometheus format).
- Structured JSON logging with configurable level.
- OpenTelemetry trace context propagation: every query log line includes `traceparent` header (W3C format). Queries spanning HNSW, delta buffer, and sidecar emit child spans. SQL commenter format carries trace context in the wire protocol.

---

### 3.14 Backup & Restore

- **Backup**: `BACKUP TO '/path'` — acquires brief write‑quiesce (< 100 ms) to flush memtable and create clean Merkle root; copies PAX blocks + WAL to target. Reads continue; new writes queue and resume immediately after copy begins.
- **Restore**: `RESTORE FROM '/path'` — validates block checksums, replays WAL, rebuilds ART primary key index and HNSW graph.
- v2: continuous backup to S3 using Merkle root manifests.

---

### 3.15 v1 Limitations (Explicit)

- No distributed clustering; single‑node only.
- Snapshot Isolation only (write‑skew possible).
- Simple wire protocol only.
- `UPDATE` of embedded column source value blocked (workaround: `DELETE` + `INSERT`).
- No versioned vector indexes for `AT VERSION` (semantic snapshots v2).
- No secondary indexes: non‑primary‑key lookups use zone‑map pruning + brute‑force scan of candidate blocks. `CREATE INDEX` planned for v2.
- RaBitQ optional if implementation time overruns; default is SQ8/FP16.
- macOS/Windows platforms are for development only; production performance guarantees are Linux‑io_uring specific.

---

## 4. v2 Full System

### 4.1 RGABH‑Driven Adaptive Storage
Multi‑timescale EMA gradients (`short_heat`, `long_heat`, `training_heat`) drive buffer‑pool admission, speculative prefetch, and automated storage tiering (NVMe → S3 → Glacier). Auto‑tuned thresholds keep NVMe utilisation at 80 %. Codebook refresh triggered by drift detection.

### 4.2 Advanced Indexing
- **DiskANN / FreshDiskANN** — disk‑resident mutable search for 50M–1B+ vectors.
- **SPANN** — inverted index for web‑scale (> 100B vectors).
- Index selection via DDL (`USING diskann`); auto‑selection by dataset size.

### 4.3 Distributed Clustering & Global Transactions
- OLTP sharding: consistent hash on primary key.
- ANN sharding: IVF coarse quantizer with coexisting quantizers during retraining.
- Raft replication; read‑only columnar replicas for HTAP.
- Default **causal consistency** (HLC). Opt‑in **strict serializability** (2PC + SSI).

### 4.4 Active Learning & Feedback Loop
- `_galaxdb_predictions` table; `FEEDBACK` SQL.
- Background uncertainty scoring (`_al_uncertainty`).
- Cold‑start: random → cluster‑then‑sample → uncertainty.
- Drift detection triggers quantizer retraining and re‑embedding.

### 4.5 Semantic Snapshot Guarantees
- `CONSISTENCY 'SEMANTIC_SNAPSHOT'` uses versioned HNSW/DiskANN index for exact historical vector search.
- Index snapshots linked in Merkle DAG.

### 4.6 Full PostgreSQL Protocol & BI Integration
- Extended query protocol, `COPY`, full `information_schema`.
- BI tool compatibility (Tableau, Metabase, DataGrip).

### 4.7 GPU‑Direct & Hardware Acceleration
- Training scans DMA directly from NVMe to GPU memory.
- `training_heat` callback feeds RGABH without polluting OLTP prefetch.
- Optional FPGA/SmartNIC for distance computation offload.

### 4.8 Federated Queries & Privacy
- Ownership policies per data atom.
- Differential‑privacy‑budgeted federated aggregation across organisational silos.

### 4.9 Plugin Marketplace
- `EmbeddingModel` trait with batch interface:
  ```rust
  trait EmbeddingModel {
      fn embed_batch(&self, texts: &[&str]) -> Vec<Vec<f32>>;
      fn model_id(&self) -> &str;
      fn dimensions(&self) -> usize;
      fn max_batch_size(&self) -> usize { 32 }
  }
  ```
- Sandboxed registry; revenue share for commercial plugins.

### 4.10 Semantic Caching
- `CREATE SEMANTIC CACHE FOR TABLE … SIMILARITY … TTL …`
- System‑managed cache table; automatic `SEMANTIC_MATCH` lookup before query execution.
- Reduces LLM API costs by 50–70 % for repetitive or near‑duplicate queries.

### 4.11 Multi‑Tenancy
- **Schema‑level isolation** (recommended): `CREATE TENANT 'name'` provisions dedicated schema; per‑schema encryption keys; no cross‑tenant compaction.
- **Row‑level isolation** via RLS for lightweight scenarios.

---

## 5. Implementation Roadmap

### v1 — 4 Months, 2–3 Engineers

| Month | Deliverable |
|-------|-------------|
| **1** | LSM+PAX store, crossbeam‑skiplist‑mvcc memtable, ART index, Monkey Bloom, NUMA buffer pool, Lazy Leveling, WAL with XXH3‑64, checkpoint, TDE, statistics, io_uring/tokio I/O abstraction |
| **2** | SQL parser (AuroraSQL extensions), PostgreSQL simple wire protocol, Python client, DDL/DML |
| **3** | mmap HNSW + delta buffer, SQ8/FP16 quantization, embedding sidecar, backlog, `SEMANTIC_MATCH`, adaptive planner |
| **4** | Merkle DAG, `AT VERSION` + guardrails, version tags, Lance training export, near‑dedup, backup/restore, observability, RateLimiter + WriteController, chaos tests, public demo |

### v2 — 12–18 Months, Expanded Team

| Phase | Duration | Focus |
|-------|----------|-------|
| **1** | 3–4 mo | RGABH adaptive tiering |
| **2** | 4–5 mo | Distributed clustering (Raft, IVF+HNSW, 2PC) |
| **3** | 3–4 mo | Advanced indexing (DiskANN, FreshDiskANN, SPANN), GPU Direct |
| **4** | 2–3 mo | Active learning, feedback loop, semantic caching, full PostgreSQL protocol, federated queries, multi‑tenancy, plugin marketplace |

---

## 6. Appendices

### A. Complete Audit Trail (All 27 Findings Resolved)

| # | Finding | Source | Resolution |
|---|---------|--------|------------|
| 1 | Bloom filter sub‑optimal | Monkey (Dayan et al., TODS 2018) | Monkey‑level allocation |
| 2 | NUMA blind spot | PostgreSQL 18 NUMA, EnterpriseDB | Per‑NUMA‑node HotSet |
| 3 | MinHash bottleneck in sidecar | FED (Son et al., 2025) | Moved to Rust write path |
| 4 | No encryption at rest | GDPR/HIPAA, OpenSSL AES‑NI | AES‑256‑GCM TDE, WAL encryption |
| 5 | Connection pooling undefined | PostgreSQL protocol, RocksDB memtable | Async Rust tasks, explicit limits |
| 6 | Bw‑Tree correctness risk | CMU OpenBw‑Tree (SIGMOD 2018) | Replaced with crossbeam‑skiplist‑mvcc |
| 7 | WAL record no checksums | PostgreSQL WAL CRC‑32 | XXH3‑64 per WAL record header |
| 8 | Statistics undersized | UC Riverside framework, System R | Multi‑column correlation stats + ANALYZE |
| 9 | No training data lineage | EU AI Act (2026) | `_galaxdb_training_exports` table |
| 10 | Embedding model version untracked | Bloomberg RAG failure study | `_embedding_model_version` + re‑embed |
| 11 | Primary key index unspecified | ART (Leis et al., ICDE 2013) | ART as primary key index |
| 12 | Semantic caching absent | Production RAG cost studies | `CREATE SEMANTIC CACHE` DDL |
| 13 | Multi‑tenancy undesigned | Salesforce, PostgreSQL RLS | Schema‑level isolation, `CREATE TENANT` |
| 14 | Observability stack missing | Kubernetes, Prometheus | `/health`, `/metrics`, OTel tracing |
| 15 | Backup/restore uncovered | Production DR requirements | `BACKUP TO`/`RESTORE FROM` |
| 16 | Write stalls not mitigated | vLSM, RocksDB WriteController, SILK | RateLimiter + WriteController + small SSTs |
| 17 | ARM64 SQ8 9× slower | ARM NEON vs AVX2 analysis | Platform‑aware quantization: FP16 default ARM |
| 18 | KV separation at flush, not WAL | BVLSM (Li et al., arXiv 2025) | Moved to WAL‑time separation |
| 19 | OTel deferred to v2 | OpenTelemetry SQL commenter standard | OTel trace context in v1 |
| 20 | Secondary index missing | PostgreSQL `CREATE INDEX` | Zone‑map v1, B‑tree v2 |
| 21 | Backup race in embedded mode | PostgreSQL `pg_start_backup()` | Write‑quiesce <100ms before backup |
| 22 | Rate limiter fixed 50% unfounded | RocksDB auto‑tuned limiter | Auto‑tuned RateLimiter with HP‑queue feedback |
| 23 | vLSM structural changes deferred | vLSM, SILK | Small SST (8MB) as Month 4 hardening |
| 24 | io_uring Linux‑only, spec claimed cross‑platform | io_uring kernel docs, Google security report 2022 | Platform‑specific backends; Linux 5.10+ production |
| 25 | crossbeam‑skiplist multi‑op race corrupts MVCC | crossbeam documentation, LSM concurrency analysis | `crossbeam‑skiplist‑mvcc` or per‑shard Mutex |
| 26 | io_uring security exposure in containers | Google bug bounty 2022 | Security note in §3.12; `GALAXDB_IO_BACKEND=tokio` |
| 27 | Epoch reclamation memory leak under long queries | crossbeam epoch reclamation docs | Memtable read guideline: drop Entry before async |

### B. Research Basis (Selected Papers)
- **Monkey** (Dayan et al., ACM TODS 2018) – Bloom filter allocation.
- **OpenBw‑Tree** (Wang et al., SIGMOD 2018) – Bw‑Tree correctness gap.
- **ART** (Leis et al., ICDE 2013) – Adaptive Radix Tree.
- **WiscKey** (Lu et al., FAST 2016) – KV separation.
- **vLSM** (Xanthakis et al., arXiv 2024) – Compaction chain reduction.
- **SILK** (Balmau et al., USENIX ATC 2019) – Dynamic I/O bandwidth pre‑emption.
- **BVLSM** (Li et al., arXiv 2025) – WAL‑time KV separation.
- **RaBitQ** (Gao et al., SIGMOD 2024/2025) – Random binary quantization.
- **DiskANN** (Jayaram Subramanya et al., NeurIPS 2019) / **FreshDiskANN** (2024) / **SPANN** (Chen et al., NeurIPS 2021).
- **PostgreSQL SSI** (Cahill et al., SIGMOD 2008).
- **MinHash LSH** (Broder, 1997; FED, Son et al., 2025).
- **EU AI Act** – Article 13, training data lineage.
- **Google io_uring security report** (2022) – io_uring vulnerabilities and containment.

### C. Platform‑Aware Quantization Reference

| Platform | Default | SIMD | Compression | Notes |
|----------|---------|------|-------------|-------|
| x86‑64 (AVX2/AVX‑512) | SQ8 (int8) | AVX‑512/AVX2 | 4× | Production‑ready |
| ARM64 (Apple Silicon, Graviton) | FP16 | NEON | 2× | SQ8 opt‑in; throughput ~3× lower |
| Both | RaBitQ (opt‑in) | AVX2 / NEON | 32× | Requires GalaxDB RaBitQ extension |

---

*GalaxDB is open source under Apache 2.0. This specification is the authoritative, production‑ready reference. Implementation of Month 1 may commence.*