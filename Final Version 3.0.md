# GalaxDB Architecture Specification
## Final Version 3.0 — Research‑Hardened, AI‑Training‑Aware

**Status:** Design locked. All prior audit findings resolved. New AI‑training optimizations integrated.  
**Target:** v1 — 4 months, 2–3 Rust engineers. v2 — 12–18 months, expanded team.

---

## Table of Contents

1. [Vision & Design Principles](#1-vision--design-principles)
2. [Architecture Overview](#2-architecture-overview)
3. [v1 Core System](#3-v1-core-system)
   - 3.1 Storage Engine
   - 3.2 Vector Index & Quantization
   - 3.3 Versioning & Semantic Search Semantics
   - 3.4 Embedding Inference Sidecar
   - 3.5 AuroraSQL Language
   - 3.6 PostgreSQL Wire Compatibility
   - 3.7 Consistency Model
   - 3.8 Training Data Path & AI Workloads
   - 3.9 Deployment Modes & Platform Support
   - 3.10 Binary Footprint & Module Tiers
   - 3.11 Durability & Crash Recovery Contract
   - 3.12 v1 Limitations
4. [v2 Full System](#4-v2-full-system)
   - 4.1 RGABH-Driven Adaptive Storage
   - 4.2 Advanced Indexing (DiskANN, FreshDiskANN, SPANN)
   - 4.3 Distributed Clustering & Global Transactions
   - 4.4 Active Learning & Feedback Loop
   - 4.5 Semantic Snapshot Guarantees
   - 4.6 Full PostgreSQL Protocol & BI Integration
   - 4.7 GPU-Direct & Hardware Acceleration
   - 4.8 Federated Queries & Privacy
   - 4.9 Plugin Marketplace
5. [Implementation Roadmap](#5-implementation-roadmap)
6. [Appendices](#6-appendices)

---

## 1. Vision & Design Principles

GalaxDB is the **AI‑native database** that unifies transactional, analytical, and vector workloads into a single engine. It eliminates the five‑database spaghetti and actively improves the AI built on top of it — including built‑in training data optimization, near‑duplicate detection, and zero‑copy model feeding.

**Core Principles:**
1. **Unified, not just integrated.** One data atom carries relational fields, embeddings, binaries, and provenance lineage.
2. **Honest semantics above all.** Limitations are documented as clearly as capabilities; silent incorrectness is never allowed.
3. **Start small, scale seamlessly.** A ~70 MB embedded binary that grows into a million‑node global cluster without changing the data model.
4. **AI‑first architecture.** Embeddings, versioned snapshots, feedback loops, and training‑aware optimizations are first‑class primitives.
5. **Falsifiable claims.** Every performance number is stated with measurable conditions and reproducible benchmarks.

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────┐
│              AuroraSQL Language                   │
│    (PostgreSQL wire protocol + AI extensions)     │
├──────────────────────────────────────────────────┤
│        Query Optimizer, Planner & Executor        │
├─────────────┬──────────────┬─────────────────────┤
│ LSM + PAX   │ Mutable ANN  │ Embedding Sidecar   │
│ Store       │ (mmap graph  │ (Unix Socket,       │
│             │  + delta buf)│  persistent backlog) │
├─────────────┴──────────────┴─────────────────────┤
│    io_uring I/O Scheduler (HP/BK queues)          │
├──────────────────────────────────────────────────┤
│  Storage (NVMe, blob store, object store)         │
└──────────────────────────────────────────────────┘
```

- **v1** implements the bottom three layers as a single‑node embedded/standalone system.
- **v2** extends with distributed clustering, RGABH adaptive tiering, active learning, and hardware acceleration.

---

## 3. v1 Core System

### 3.1 Storage Engine

#### LSM‑Tree with PAX Blocks
Data is organized into **PAX (Partition Attributes Across) blocks** of ~1,000 rows.

**Write Path:**
- Rows accumulate in a lock‑free Bw‑Tree memory buffer (max 256 MB, backpressure at 256 MB).
- At 64 MB, buffer is sealed and flushed as an immutable PAX block. Flush includes fsync before commit acknowledgment.
- A WAL records transactions to guarantee atomicity of multi‑block operations.

**Compaction:**
- **Lazy Leveling:** upper levels (L0–L3) use tiered compaction; bottom level (L4) uses leveled compaction (standard RocksDB configuration). This minimizes write amplification while maintaining point‑read performance.
- Compaction doubles as MVCC garbage collection: only the latest version needed by the oldest active snapshot or referenced by a version tag is kept; older versions are dropped. No separate GC pass.

**Read Paths:**
- **Point query:** sparse primary key index → block + offset → single random read. Bloom filters (10 bits/key) skip blocks where the key is absent.
- **Column scan:** sequential reads of needed column chunks, with zone‑map pruning (min/max per column) to skip irrelevant blocks.

**Buffer Pool:**
- `HotSet` (70% RAM): LRU eviction for OLTP‑hot blocks. Pinned by version tag retention.
- `ScanBuffer` (30% RAM): clock‑sweep for OLAP scans. Cannot evict HotSet‑resident blocks.

**Compression (§3.1.1):**  
Data regions are compressed by type to balance speed and size:
- Fixed‑width columns: delta encoding + bitpacking (FastPFOR) → 8–16× compression.
- TEXT/JSONB/BLOB: Zstandard level 3.
- Embedding columns (float32/SQ8): no general‑purpose compression; quantization already handles it.
- WAL records: LZ4 for low latency.

The PAX block header stores a codec ID byte: 0=none, 1=delta+bitpack, 2=zstd, 3=lz4.

**KV Separation for Large Values (§3.1.5):**  
Columns exceeding 1 KB (configurable) are stored in a content‑addressed blob store. The PAX block stores only the 32‑byte content hash. Compaction never rewrites blobs, eliminating write amplification for image/PDF datasets.

**Bulk Insert Path:**  
For large imports (training data), `BULK INSERT` writes sorted rows directly to PAX blocks, bypassing the Bw‑Tree buffer. This avoids backpressure and maximizes write throughput.

#### Durability & Crash Recovery (§3.11)

**WAL fsync Policy:**
- Default: group commit with 10 ms fsync window (configurable). Achieves 50k–150k TPS with minimal data‑loss window.
- `DURABILITY STRICT` connection parameter: fsync per commit. Zero data loss.
- `DURABILITY RELAXED`: asynchronous WAL flush. Suitable for analytics workloads.
- The embedding backlog table always uses `STRICT` durability, regardless of session setting.

**Checkpoint & WAL Size:**
- Checkpoint every 60 seconds or when WAL exceeds 512 MB (`max_wal_size`), whichever comes first.
- If a checkpoint fails, retry with exponential backoff; after 3 consecutive failures, block writes to prevent unbounded WAL growth.
- Recovery time < 30 seconds with WAL ≤ 512 MB.

**Block Integrity:**
- XXH3‑64 checksums (faster and more robust than CRC‑32).
- Magic number `0x47414C41` at the start of every block to catch torn writes.

**Disk Full Handling:**
- WAL write fails → error returned, transaction rolled back.
- PAX block flush fails → backpressure, metric `_disk_full` set, reads continue.
- 32 MB reserve file pre‑allocated; deleted on disk full to allow a clean checkpoint before writes are blocked.

---

### 3.2 Vector Index & Quantization

**Mutable ANN: Base Graph + Delta Buffer**

- **Base graph:** HNSW index stored as an mmap’d file (`galax_index.hnsw`). Immutable between merges.
- **Delta buffer:** in‑memory flat exact k‑NN index, backed by the **same WAL** as the LSM (record type `DELTA_INSERT`).
- **Merge policy:** when delta exceeds `max(10,000, total_indexed × 0.01)`, background job rebuilds the base graph via atomic rename (shadow file + `rename()`). Only one merge at a time.

**Crash Safety:**  
During merge, the old graph remains active. On crash, the rename hasn’t occurred; the old graph is intact. The delta buffer WAL replays un‑merged entries in batches of 1,000 (to avoid memory spikes). Queries are available after the first batch, with the background continuing replay.

**Quantization:**
- **v1 default:** SQ8 (int8 scalar quantization). 4× compression, SIMD‑accelerated, no training.
- **v1 opt‑in high compression:** RaBitQ (random rotation + binary quantization). 32× compression, SIMD‑fast, no codebook training required; implementation complexity may delay it to v2 launch.
- **v2 extreme latency:** Binary quantization (popcount Hamming distance).
- **PQ completely removed.** RaBitQ dominates PQ at every compression ratio.

**Filter‑Aware Traversal:**  
Combined `SEMANTIC_MATCH` + strict `WHERE` uses ACORN‑style graph traversal (disconnected‑safe) in v2; v1 ships an adaptive plan: brute‑force scan when filter cardinality is very small.

**Tombstone Policy:**  
`DELETE` writes a tombstone to the delta buffer. When tombstones exceed 20% of indexed vectors, an emergency merge is triggered.

---

### 3.3 Versioning & Semantic Search Semantics

**Merkle DAG:**  
Every write produces a PAX block with a commit timestamp. Merkle tree over block hashes gives a version root. `AT VERSION timestamp_or_tag` filters blocks accordingly. Named tags are GC‑exempt.

**Consistency Modes for Historical Semantic Search:**

| Mode | Behavior | Availability |
|------|----------|--------------|
| `ROW_SNAPSHOT` (default) | Returns row data from the version, **rejects** `SEMANTIC_MATCH`. | v1 |
| `SEMANTIC_FRESH` | Uses current HNSW index against historical rows. Explicit opt‑in. | v1 |
| `SEMANTIC_SNAPSHOT` | Uses a versioned index for exact historical vector search. | v2 |

Without an explicit consistency hint, combining `AT VERSION` and `SEMANTIC_MATCH` raises an error.

**Training Reproducibility:**
- Tags created with `FOR TRAINING` guarantee deterministic block order (primary key sort) and store a shuffle seed.
- Byte‑identical Arrow exports; tags are retention‑exempt.

---

### 3.4 Embedding Inference Sidecar

**Architecture:**
- Standalone Rust binary (ONNX Runtime), Unix socket / named pipe communication.
- Lifecycle management: parent‑PID monitoring across platforms (Linux `prctl`, macOS `kqueue`, Windows named pipe heartbeat).

**Back‑pressure & Durability:**
- Queue depth 10,000. Overflow rows go to persistent backlog table `_andromeda_embedding_backlog` (always flush with `STRICT` durability).
- Backlog scanner retries when sidecar capacity recovers; no data loss.
- Sidecar crash → degraded mode (semantic search still works on already‑indexed vectors), automatic restart with exponential backoff.

---

### 3.5 AuroraSQL Language

PostgreSQL‑compatible SQL extended with:

- `EMBEDDING MODEL` in DDL (auto‑embedding).
- `SEMANTIC_MATCH(column, 'query', threshold)`.
- `AT VERSION timestamp/tag` with consistency modes.
- `FEEDBACK` SQL (v2).
- `ORDER BY ACTIVE_LEARNING(target)` (v2).
- `CREATE VERSION TAG 'name' [WITH TRAINING SEED n] [FOR TRAINING]`.
- `BULK INSERT` for high‑throughput loading.

**Training‑Aware DDL Extensions:**
```sql
CREATE VERSION TAG 'q4_run' FOR TRAINING WITH TRAINING PRECISION 'sq8', TRAINING SEED 42;
```
- Precision `sq8`, `rabitq`, or `float32` controls the quantization of exported embeddings.
- Training‑specific exports materialize a Lance‑format dataset (zero‑copy for PyTorch).

**Near‑Duplicate Detection (v1 opt‑in):**
- System column `_near_duplicate_of`.
- `WHERE NOT DUPLICATE` filters out rows that are > 80% Jaccard similar (MinHash LSH computed automatically for TEXT/BLOB columns).
- Background job periodically refreshes duplicate groups.

---

### 3.6 PostgreSQL Wire Compatibility (Tier 1)

- Simple query protocol only; basic DDL/DML, `pg_catalog` stubs for `psycopg2` and SQLAlchemy (simple mode).
- Extended protocol, `COPY`, full `information_schema` are v2.

---

### 3.7 Consistency Model

**v1 provides Snapshot Isolation (SI), not strict serializability.**  
- Guarantees no dirty reads, no non‑repeatable reads, no phantoms.
- Write‑skew is theoretically possible (e.g., doctors‑on‑call problem). Strict serializable (SSI) is available in v2 via a `SERIALIZABLE` connection parameter.
- Semantic search freshness is eventually consistent (asynchronous embedding pipeline). The `_embedding_stale` flag is reliable because embedding writes follow the standard LSM path, ensuring flag and data are seen atomically.

---

### 3.8 Training Data Path & AI Workloads

**Arrow Flight Export:**  
`execute_arrow(query)` returns an Arrow RecordBatch stream; in v1 this is an in‑memory IPC path without gRPC.

**Lance Materialization for Training:**  
When a version tag is created with `FOR TRAINING`, the relevant columns are materialized into a **Lance‑format dataset** (Arrow columnar, memory‑mappable). A Python method `galaxdb.training_dataset(tag)` returns a PyTorch `IterableDataset` that reads Lance files with zero deserialization overhead.

**Training Precision:**  
`TRAINING PRECISION 'sq8'` materializes int8 vectors; `rabitq` produces binary‑quantized vectors. Both significantly reduce I/O during training, cutting GPU idle time.

**Near‑Duplicate Exclusion for Training:**  
`WHERE NOT DUPLICATE` can be used in training exports to remove near‑duplicates, reducing training cost by 15–30% (empirical estimate).

**Curriculum Learning Export:**  
`ORDER BY ACTIVE_LEARNING('label')` (v2) combined with `FOR TRAINING` will sort by uncertainty, enabling curriculum learning. In v1, a simple difficulty metric can be approximated via `_near_duplicate_of IS NOT NULL` or manual labeling.

**Reproducibility:**  
- Tags created with `WITH TRAINING SEED n` store the seed.
- Block order is deterministic (sorted by primary key) for training exports.

---

### 3.9 Deployment Modes & Platform Support

| Mode | Platforms | Description |
|------|-----------|-------------|
| Embedded | Linux, macOS, Windows | `import galaxdb`; sidecar spawned as child process. |
| Standalone server | Linux, macOS | `galaxdb --server` on port 5432. |
| Clustered | v2 only | Distributed with Raft and sharding. |

Cross‑process ownership: platform‑specific mechanisms (prctl, kqueue, named pipe heartbeat).

---

### 3.10 Binary Footprint & Module Tiers

- Core engine: **< 70 MB** (Rust, statically linked).
- Full installation with sidecar and default model: **< 350 MB**.
- Tiers: minimal (engine only), standard (plus Python client), full (sidecar + model).

---

### 3.11 Durability & Crash Recovery Contract (Consolidated)

All critical durability and recovery policies are unified here:

| Aspect | Policy |
|--------|--------|
| **WAL fsync** | Group commit by default (10 ms); `DURABILITY STRICT` per‑connection. |
| **Checkpoint** | Every 60 s or 512 MB WAL; on failure, backpressure and retry. |
| **Block integrity** | XXH3‑64 checksum + magic number. |
| **Delta buffer WAL** | Same WAL as LSM, distinct record type; batch replay on recovery. |
| **HNSW crash safety** | Atomic rename for merges; delta buffer replay in batches of 1,000. |
| **Sidecar state** | Stateless; all pending work in durable backlog table (`STRICT` durability). |
| **Disk full** | Pre‑allocated 32 MB reserve file; write blocking after clean checkpoint. |

---

### 3.12 v1 Limitations (Explicit)

- No distributed clustering; single‑node only.
- No SSI (Snapshot Isolation only); write‑skew possible.
- No GPU Direct; no versioned vector indexes for `AT VERSION`.
- Simple wire protocol only; `UPDATE` of embedded column source values blocked.
- RaBitQ may be optional if not completed; SQ8 is default.
- Near‑dedup and Lance export are v1 features but may be marked “beta” if necessary.

---

## 4. v2 Full System

### 4.1 RGABH-Driven Adaptive Storage
Multi‑timescale EMA gradients (`short_heat`, `long_heat`, `training_heat`) drive buffer pool admission, prefetch, and automatic storage tiering (NVMe → S3 → Glacier). Codebook refresh triggered by drift detection.

### 4.2 Advanced Indexing
In addition to HNSW, v2 supports:
- **DiskANN / FreshDiskANN**: disk‑resident, mutable, for datasets beyond RAM.
- **SPANN**: inverted index for web‑scale (> 100B vectors).
- Index selection via DDL (`USING diskann`). Auto‑selection by dataset size.

### 4.3 Distributed Clustering
- Sharding for OLTP (consistent hash) and ANN (IVF coarse quantizer with coexisting quantizers during retraining).
- Causal consistency via HLC; strict serializability with 2PC opt‑in.
- Read‑replicas for HTAP scale‑out (Raft log shipping, zero‑ETL).

### 4.4 Active Learning & Feedback
- `_andromeda_predictions` table; `FEEDBACK` SQL; `ORDER BY ACTIVE_LEARNING` pre‑computed via background job.
- Cold‑start stages: random → cluster‑then‑sample → uncertainty sampling.
- Drift detection triggers quantizer/codebook refresh.

### 4.5 Semantic Snapshot Guarantees
- `CONSISTENCY 'SEMANTIC_SNAPSHOT'` uses versioned HNSW index.
- Byte‑identical semantic results at historical versions.

### 4.6 Full PostgreSQL Protocol
Extended query protocol, `COPY`, full `information_schema`, BI tool compatibility.

### 4.7 GPU‑Direct & Hardware Acceleration
Training data flows directly from NVMe to GPU; `training_heat` feeds RGABH.

### 4.8 Federated Queries & Privacy
Differential‑privacy‑budgeted federated aggregation across organizational silos.

### 4.9 Plugin Marketplace
`EmbeddingModel` trait extended with batch interface:
```rust
trait EmbeddingModel {
    fn embed_batch(&self, texts: &[&str]) -> Vec<Vec<f32>>;
    fn model_id(&self) -> &str;
    fn dimensions(&self) -> usize;
    fn max_batch_size(&self) -> usize { 32 }
}
```
Third‑party plugins for models, active‑learning strategies, etc. Revenue share.

---

## 5. Implementation Roadmap

### v1 — 4 Months, 2–3 Engineers

| Month | Deliverable |
|-------|-------------|
| **1** | Core LSM with Lazy Leveling, compression, KV separation, WAL, checkpoint, Bw‑Tree buffer, bulk insert, HotSet/ScanBuffer, Bloom filters, zone maps, readahead. |
| **2** | SQL layer, AuroraSQL extensions, simple wire protocol, Python embedded mode, DDL/DML. |
| **3** | Mutable ANN (mmap + delta buffer), SQ8 quantization, embedding sidecar, backlog system, `SEMANTIC_MATCH` with adaptive planning, filter‑aware fallback. |
| **4** | Merkle DAG, `AT VERSION` with guardrails, version tags, Lance training export, near‑dedup, training precision, hardening, chaos tests, public demo. |

### v2 — 12–18 Months
- **Phase 1:** RGABH + adaptive tiering.
- **Phase 2:** Distributed clustering + advanced index backends.
- **Phase 3:** Active learning, feedback, drift detection.
- **Phase 4:** Full PostgreSQL protocol, GPU Direct, federated queries, plugin marketplace.

---

## 6. Appendices

### A. Audit Trail (Major Issues Resolved)
All previously identified loopholes—PQ choice, HNSW crash safety, consistency model, compression, WAL fsync, block integrity, etc.—are resolved in this version. PQ eliminated; RaBitQ introduced; MVCC GC via compaction; near‑dedup and training optimizations added.

### B. Key Performance Optimizations
- Bloom filters, zone map pruning, sequential scan readahead.
- Batch embedding interface.
- Lance materialization for zero‑copy training.
- Training precision to slash I/O.

### C. Glossary
- **PAX** – Partition Attributes Across (hybrid row/column block)
- **SQ8** – int8 scalar quantization
- **RaBitQ** – Random Binary Quantization (SIGMOD 2024/2025)
- **LSH** – Locality Sensitive Hashing (MinHash for dedup)
- **Lance** – Columnar format designed for ML training

### D. References
(Key papers cited throughout the document: Dostoevsky, WiscKey, P‑HNSW, RaBitQ, DiskANN, FreshDiskANN, SPANN, PostgreSQL SSI, etc.)

---

*This is the authoritative GalaxDB specification. Implementation begins immediately against this document.*