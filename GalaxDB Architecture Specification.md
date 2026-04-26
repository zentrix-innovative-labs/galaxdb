# GalaxDB Architecture Specification  
**Final Version 2.0**  
*Covering v1 (MVP) and v2 (Full AI‑Native Vision)*

**Authors:** GalaxDB Engineering  
**Status:** Design Locked. Implementation‑ready for v1.

---

## Table of Contents
1. [Vision & Design Principles](#1-vision--design-principles)  
2. [Architecture Overview](#2-architecture-overview)  
3. [v1 Core System (Buildable MVP)](#3-v1-core-system)  
   - 3.1 Storage Engine  
   - 3.2 Vector Index  
   - 3.3 Versioning (Merkle DAG)  
   - 3.4 Embedding Inference Sidecar  
   - 3.5 AuroraSQL Language  
   - 3.6 PostgreSQL Wire Compatibility  
   - 3.7 Consistency Model  
   - 3.8 Training Data Path  
   - 3.9 Deployment Modes  
   - 3.10 Binary Footprint  
   - 3.11 v1 Limitations & Explicit Constraints  
4. [v2 Full System (AI‑Native Vision)](#4-v2-full-system)  
   - 4.1 RGABH‑Driven Adaptive Storage  
   - 4.2 Distributed Clustering & Global Transactions  
   - 4.3 Active Learning & Feedback Loop  
   - 4.4 Full PostgreSQL Protocol & BI Integration  
   - 4.5 GPU‑Direct & Hardware Acceleration  
   - 4.6 Federated Queries & Privacy  
   - 4.7 Plugin Marketplace  
5. [Implementation Roadmap](#5-implementation-roadmap)  
6. [Appendices](#6-appendices)  

---

## 1. Vision & Design Principles

GalaxDB is the **AI‑native database** that unifies transactional, analytical, vector, and graph data into a single engine. It eliminates the five‑database spaghetti and actively improves the AI built on top of it.

**Core Principles (apply to all versions):**

- **Unified, not just integrated.** One data atom carries relational fields, embeddings, binaries, and lineage.
- **Honesty in semantics.** Every feature’s limitations are documented as clearly as its capabilities.
- **Start small, scale seamlessly.** A 50 MB embedded binary that grows into a million‑node global cluster without changing the data model.
- **AI‑first architecture.** Embeddings, versioned snapshots, and feedback loops are first‑class, not bolted on.

---

## 2. Architecture Overview

GalaxDB’s architecture is layered, with each version adding capabilities:

```
┌────────────────────────────────────────────┐
│                AuroraSQL                    │
│     (PostgreSQL wire protocol + AI SQL)     │
├────────────────────────────────────────────┤
│          Query Optimizer & Executor         │
├───────────┬───────────┬────────────────────┤
│  LSM +   │  HNSW +   │  Embedding Sidecar  │
│ PAX Store│  PQ Index │  (Unix Socket)      │
├───────────┴───────────┴────────────────────┤
│  io_uring I/O Scheduler (HP/BK queues)      │
├────────────────────────────────────────────┤
│  Storage (NVMe, S3, Glacier) – tiered v2   │
└────────────────────────────────────────────┘
```

- **v1** implements the bottom three layers as a single‑node embedded/standalone system.
- **v2** extends the upper layers with distributed clustering, adaptive tiering (RGABH), and active learning.

---

## 3. v1 Core System

### 3.1 Storage Engine

**LSM‑Tree with PAX Blocks**

- **Data Atom**: A single row is stored in a PAX (Partition Attributes Across) block of ~1,000 rows.
- **Write path**: Rows accumulate in a lock‑free Bw‑Tree memory buffer. At 64 MB, the buffer is sealed and flushed as an immutable PAX block.
- **Compaction**: Background compactor merges blocks into larger sorted runs (LSM levels). Old blocks are reclaimed after version GC.
- **Reads**:
  - Point query: sparse index → block + offset → single random read.
  - Column scan: sequential I/O on column chunks.
- **Buffer Pool**:
  - `HotSet` (70% RAM): LRU eviction for OLTP‑hot blocks.
  - `ScanBuffer` (30% RAM): clock‑sweep for OLAP scans; cannot evict a HotSet‑resident block.
  - No gradient‑based (RGABH) logic in v1.

**PAX Block Physical Layout**  
(See v1 spec §3.2)

### 3.2 Vector Index

**HNSW Graph on Memory‑Mapped File**

- The HNSW graph is stored as a **separate mmap'd file** (`.hnsw`) to avoid LSM multi‑level lookup penalty. Traversal uses direct memory access.
- **PAX blocks** hold the PQ‑compressed vectors. Search: HNSW finds candidate nodes → read PQ codes from LSM → top‑K re‑rank using raw floats from PAX blocks.

**Product Quantization (PQ)**
- Codebook trained on 10k–1M vectors (skip PQ below 10k). Static in v1.
- Embeddings stored as PQ codes; HNSW operates on codes.

**Insert Path & Index Freshness**
1. `INSERT` → text sent to sidecar → row written with `_embedding_stale=true`.
2. Sidecar returns embedding → background worker updates the row and adds vector to an in‑memory **flat index** (exact k‑NN).
3. `SEMANTIC_MATCH` searches HNSW + flat index → union → re‑rank.
4. When flat index size exceeds `max(10,000, total_indexed × 0.01)`, trigger a **double‑buffered HNSW rebuild**. Only one rebuild runs at a time.

**DML Restrictions (v1)**
- `UPDATE` of an embedded column’s source value is **blocked**.
- All other `UPDATE`/`DELETE` on rows with embeddings work normally (tomestones).

**`AT VERSION` and Vector Search**  
`AT VERSION` provides historical row data, but the HNSW index is not versioned. The query uses the current index. This is a documented limitation.

### 3.3 Versioning (Merkle DAG)

- Every write creates a PAX block with a commit timestamp. A Merkle tree over block hashes yields a version root.
- `AT VERSION` filters blocks where `commit_time ≤ target_version`. Named tags can be created.
- Versioning covers **row data only**. Embeddings and system catalogs are not versioned.

### 3.4 Embedding Inference Sidecar

**Architecture**
- Separate process loading ONNX model, communicates via Unix socket.
- CLI/Python spawns it as a child; parent PID monitoring ensures cleanup.

**Lifecycle**
- **Crash recovery**: Exponential backoff restart; DB continues serving (embeddings remain stale).
- **Back‑pressure**: 10k in‑flight queue limit. Overflow rows are written to `_GalaxDB_embedding_backlog`, reprocessed later by a low‑priority scanner. No data loss.

### 3.5 AuroraSQL Language

PostgreSQL‑compatible SQL extended with:
- `EMBEDDING MODEL` in DDL for auto‑embedding.
- `SEMANTIC_MATCH(column, 'query', threshold)` – hybrid search.
- `AT VERSION timestamp/tag` – time travel.
- Active learning and `FEEDBACK` are **v2 only**.

### 3.6 PostgreSQL Wire Compatibility (Tier 1)

- Simple query protocol (`Q` message).
- Basic DDL/DML, pg_catalog stubs for `psycopg2` & SQLAlchemy.
- No extended query protocol, `COPY`, or full `information_schema` in v1.

### 3.7 Consistency Model

**v1: Strict Serializability**  
Single‑node, single‑writer LSM provides true serializable isolation. Snapshot isolation via versioned blocks.

**v2: Causal Consistency (Distributed)**  
When clustering is added, the guarantee will be downgraded to causal consistency with bounded staleness (HLC‑based). This is an explicit design decision.

### 3.8 Training Data Path

- `SELECT … AT VERSION 'tag'` can be materialized as Arrow RecordBatch stream via Arrow Flight.
- No GPU Direct storage in v1; data flows through CPU memory.

### 3.9 Deployment Modes

| Mode | Description |
|------|-------------|
| Embedded | `import GalaxDB` – in‑process, no server. |
| Standalone | `GalaxDB --server` – listens on port 5432. |
| Clustered | **v2 only** |

### 3.10 Binary Footprint

- Core engine: **< 50 MB** (Rust, statically linked).
- Full install with inference sidecar + model: **< 500 MB**.

### 3.11 v1 Limitations & Explicit Constraints

- No distributed transactions, no clustering.
- No active learning, feedback loops.
- No GPU Direct.
- HNSW index not versioned for `AT VERSION`.
- PQ codebook static.
- No `UPDATE` of embedded column source values.
- Simple query protocol only.
- All these are clearly documented and will be addressed in v2.

---

## 4. v2 Full System

The following sections describe the complete AI‑Native vision, building upon the v1 foundation. All design decisions are informed by the rigorous review of the v1 spec.

### 4.1 RGABH‑Driven Adaptive Storage

**Row‑Gradient‑Aggregated Block Hotness** replaces static buffer pool policies.  
- Each row maintains a multi‑timescale EMA gradient: `short_heat` (OLTP bursts), `long_heat` (sustained importance), `training_heat` (GPU access).  
- Block hotness = sum of row gradients.  
- **Buffer pool**: Admission and eviction use block hotness, not LRU. Partitioned HotSet holds blocks above a dynamic threshold.  
- **Speculative prefetch**: Pre‑fetch blocks whose `short_heat` velocity is high.  
- **Adaptive tiering**: Hotness thresholds `T_hot`/`T_cold` auto‑tune to keep NVMe at 80% utilization. Blocks migrate automatically between NVMe, object store, and glacier. A quiescent metadata sweep handles cold‑data demotion.  
- PQ codebook refresh is triggered by the same drift detection used for IVF re‑training.

### 4.2 Distributed Clustering & Global Transactions

**Sharding Strategy**
- **OLTP**: Consistent hash on primary key for uniform distribution.
- **ANN**: Global IVF coarse quantizer routes queries to 1–2 relevant shards, then per‑shard HNSW searches. No separate physical shard maps.
- IVF quantizer is retrained when embedding drift is detected, with coexisting quantizers (no immediate data migration).

**Consistency**
- Hybrid Logical Clocks (HLC) provide **causal consistency with bounded staleness** as the default.
- Strict serializability for cross‑shard transactions will be added later via a 2PC protocol (like Percolator/TiDB).

**Replication**
- Raft groups within each shard.
- Read‑only replicas for HTAP scale‑out: identical binary, Raft‑shipped logs, zero‑ETL.
- The distributed system can serve billions of rows and 100k+ TPS with proper resource allocation.

### 4.3 Active Learning & Feedback Loop

- `_GalaxDB_predictions` table: applications insert `(row_id, model_id, prediction, actual)`.
- **Drift detector** monitors accuracy, triggers PQ/index retraining and alerts.
- **Active learning**: `ORDER BY ACTIVE_LEARNING` pre‑computes uncertainty scores as a column; background job refreshes them on model update.
- **FEEDBACK SQL** appends delta updates, preserving lineage.
- Cold‑start: random → cluster‑then‑sample → uncertainty sampling.

### 4.4 Full PostgreSQL Protocol & BI Integration

- Extended query protocol (`Parse`/`Bind`/`Execute`) for prepared statements.
- `COPY` support, full `information_schema`, enough catalog tables for Tableau, Metabase, DataGrip.
- Service‑side cursors for large result sets.

### 4.5 GPU‑Direct & Hardware Acceleration

- **GPUDirect Storage**: Training scans bypass CPU, DMA directly from NVMe to GPU memory.
- `report_training_access` callback feeds `training_heat` into RGABH without affecting OLTP prefetch.
- Optional FPGA/SmartNIC offload for vector distance computation near storage.

### 4.6 Federated Queries & Privacy

- Atoms carry ownership policies. Federated queries aggregate across organizations with differential privacy budgets managed transparently.
- Secure aggregation built into the query planner.

### 4.7 Plugin Marketplace

- Defined `EmbeddingModel` trait in v1; v2 adds plugin system for models, active‑learning strategies, and domain adapters. Revenue‑share model.

---

## 5. Implementation Roadmap

### v1 (4 months, 2–3 engineers)
*See v1 spec §14 for detailed month‑by‑month plan.*
- Month 1: Core LSM storage engine with PAX blocks.
- Month 2: SQL layer, PostgreSQL wire protocol, Python embedded mode.
- Month 3: mmap’d HNSW, PQ, embedding sidecar, insert path with flat index.
- Month 4: Merkle DAG versioning, `AT VERSION`, HNSW rebuild, Arrow Flight, hardening.

### v2 (estimated 12–18 months)
- Phase 1: RGABH adaptive storage + tiering.
- Phase 2: Distributed clustering with IVF+HNSW and Raft.
- Phase 3: Active learning, FEEDBACK, drift detection.
- Phase 4: Full PostgreSQL protocol, GPU Direct, federated queries, plugins.

---

## 6. Appendices

### A. Key Design Decisions & Rationale
- HNSW in separate mmap file: performance (avoids LSM overhead).
- UPDATE restriction on embedded columns only: pragmatic v1 scope.
- Stale row backlog (not drop): prevents silent data loss.
- PQ minimum 10k vectors: codebook quality.
- Flat index rebuild threshold `max(10k, 1%)`: prevents rebuild storms on large tables.
- Causal consistency for distributed v2: avoids exaggerated claims.

### B. Glossary
- **PAX** – Partition Attributes Across (hybrid row/column block layout).
- **PQ** – Product Quantization (vector compression).
- **RGABH** – Row‑Gradient‑Aggregated Block Hotness.
- **HLC** – Hybrid Logical Clock.

### C. References
- hnswlib, FAISS, Milvus for vector index design.
- TiDB/TiFlash for HTAP read‑replica architecture.
- RocksDB/LevelDB for LSM compaction strategies.

---

*This document is the authoritative reference for GalaxDB. All future implementation decisions must be traceable to the principles and constraints laid out herein.*