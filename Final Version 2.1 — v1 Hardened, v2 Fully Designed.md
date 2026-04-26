# Andromeda Architecture Specification
## Final Version 2.1 — v1 Hardened, v2 Fully Designed

**Status:** Design locked. All 20 critical/high issues from the full design review resolved. Twelve v1.1 loopholes identified in the external audit closed. Ten v2 performance upgrades incorporated.

**Target:** v1 — 4 months, 2–3 Rust engineers. v2 — 12–18 months.

---

## Table of Contents

1. [Vision & Design Principles](#1-vision--design-principles)
2. [Architecture Overview](#2-architecture-overview)
3. [v1 Core System — Corrected & Hardened](#3-v1-core-system)
   - 3.1 Storage Engine
   - 3.2 Vector Index & Mutable ANN
   - 3.3 Versioning & Semantic Search Semantics
   - 3.4 Embedding Inference Sidecar
   - 3.5 AuroraSQL Language
   - 3.6 PostgreSQL Wire Compatibility
   - 3.7 Consistency Model — Revised
   - 3.8 Training Data Path
   - 3.9 Deployment Modes & Platform Support
   - 3.10 Binary Footprint & Module Tiers
   - 3.11 Durability & Crash Recovery Contract
   - 3.12 v1 Limitations — Explicit
4. [v2 Full System — AI-Native Vision](#4-v2-full-system)
   - 4.1 RGABH-Driven Adaptive Storage
   - 4.2 Mutable ANN with Merge Policy
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

Andromeda is the **AI-native database** that unifies transactional, analytical, vector, and graph data into a single engine. It eliminates the five-database spaghetti and actively improves the AI built on top of it.

**Core Principles (apply to all versions):**

1. **Unified, not just integrated.** One data atom carries relational fields, embeddings, binaries, and lineage.
2. **Honest semantics above all.** Every feature's limitations are documented as clearly as its capabilities. Users must never be surprised by silent incorrectness.
3. **Start small, scale seamlessly.** A ~50 MB embedded binary that grows into a million-node global cluster without changing the data model.
4. **AI-first architecture.** Embeddings, versioned snapshots, and feedback loops are first-class primitives, not bolt-on extensions.
5. **Falsifiable claims.** Every performance number, every consistency guarantee, every durability promise is stated with measurable conditions.

---

## 2. Architecture Overview

Andromeda's architecture is layered, with each version adding capabilities:

```
┌──────────────────────────────────────────────────┐
│              AuroraSQL Language                   │
│    (PostgreSQL wire protocol + AI extensions)     │
├──────────────────────────────────────────────────┤
│        Query Optimizer, Planner & Executor        │
├─────────────┬──────────────┬─────────────────────┤
│ LSM + PAX   │ Mutable ANN  │ Embedding Sidecar   │
│ Store       │ (mmap graph  │ (Unix Socket,       │
│             │  + flat buf) │  persistent backlog) │
├─────────────┴──────────────┴─────────────────────┤
│    io_uring I/O Scheduler (HP/BK queues)          │
├──────────────────────────────────────────────────┤
│  Storage (NVMe, object store, glacier — tiered v2)│
└──────────────────────────────────────────────────┘
```

- **v1** implements the bottom three layers as a single-node embedded/standalone system.
- **v2** extends with distributed clustering, RGABH adaptive tiering, active learning, and hardware acceleration.

---

## 3. v1 Core System — Corrected & Hardened

*This section replaces the v1.1 spec. All audit findings are addressed inline.*

### 3.1 Storage Engine

**LSM-Tree with PAX Blocks**

- **Data Atom**: A single row is stored in a PAX (Partition Attributes Across) block of ~1,000 rows.
- **Write path**: Rows accumulate in a lock-free Bw-Tree memory buffer. At 64 MB, the buffer is sealed and flushed as an immutable PAX block. Flush is synchronous with fsync before acknowledging commit.
- **Compaction**: Background compactor merges blocks into larger sorted runs (LSM levels). Old blocks are reclaimed only after version retention windows expire and all active snapshots release references.
- **Reads**:
  - Point query: sparse index → block + offset → single random read.
  - Column scan: sequential I/O on column chunks within block.
- **Buffer Pool**:
  - `HotSet` (70% RAM): LRU eviction for OLTP‑hot blocks. Can be pinned by version tag retention.
  - `ScanBuffer` (30% RAM): clock‑sweep for OLAP scans; **cannot evict a HotSet‑resident block**. ScanBuffer blocks never promote to HotSet based on scan access alone.
  - No gradient‑based (RGABH) logic in v1.

**PAX Block Physical Layout**

| Section | Contents |
|---------|----------|
| Header | Block ID, row count, column descriptors, min/max per column, compression info, checksum |
| Column chunks | Fixed‑width columns stored contiguously; variable‑width (TEXT, BLOB) with length prefixes |
| Embedding column | Dense float32 arrays, optionally PQ‑compressed (see §3.2) |
| Row offset table | Byte offset of each row within the block |

All block data is CRC‑32 checksummed. The format is self‑describing and versioned (format version byte in header).

**Durability Contract:**
- Committed writes survive OS crash / power loss.
- Flush-on-commit: each transaction's PAX block is fsync'd before acknowledgment.
- Write‑ahead log (WAL) for atomicity of multi‑block transactions: WAL entries written and fsync'd before data block flush. WAL is replayed on startup.
- Checkpoint: periodic full flush with WAL truncation, bound recovery time to < 30 seconds on cold start.

### 3.2 Vector Index & Mutable ANN

*This section completely replaces the v1.0/v1.1 vector index design, incorporating audit findings #4, #5, #6, #7 and performance upgrades #1, #2, #4, #5, #6.*

**Architecture: Graph + Flat Buffer with Merge Policy**

The HNSW index is treated as **mutable** from the start, avoiding the brittle "immutable after initial build" model. The design uses two structures that coexist and periodically merge:

1. **Base graph** — an HNSW index covering all vectors at the time of last merge. Stored as a memory‑mapped file (`.hnsw`). Read‑only between merges.
2. **Delta buffer** — a small mutable flat index (exact k‑NN) holding vectors inserted/updated since the last merge. Stored in‑memory, with a persistent write‑ahead log for durability.

**Query Path:**
- `SEMANTIC_MATCH` probes **both** the base graph and the delta buffer.
- Results are **unioned** and re‑ranked against raw vectors from the LSM store for final exact scores.
- When the delta buffer size exceeds `max(10,000, total_indexed × 0.01)`, a **background merge** is triggered: a new HNSW graph is built incorporating all vectors (base + delta), and the result atomically replaces the old base graph via double‑buffering.
- During merge, queries continue against the old base graph + delta buffer. Once the new graph is ready, an atomic pointer swap routes new queries to the new graph. The old graph is freed when all in‑flight queries release references.

**Dynamic Search Depth (ef):**
- ef is **not a fixed static value**. It is bounded dynamically:
  - Default ef = 100.
  - If query latency exceeds a configured SLO (e.g., 5 ms), ef is reduced for subsequent queries.
  - If recall is measured below target (via periodic sampling against ground truth), ef is increased for that workload.
  - This is a simple feedback controller, not a machine‑learning model — a proportional controller on latency and recall error signals.

**mmap Behavior Disclaimer:**
The spec no longer claims "zero I/O" for HNSW traversal. Pointer dereference on an mmap'd region triggers **page faults** when the underlying page is not resident in the OS page cache. Under memory pressure or cold start, traversal incurs disk reads. The mmap design **avoids explicit I/O system calls** in the hot path, but physical I/O still occurs. We document this honestly.

**Candidate Scoring Optimization:**
During graph traversal, candidate PQ codes are read from the mmap'd graph payload (which stores a copy of the PQ code alongside each node's adjacency list). This avoids LSM random reads during traversal. Only the final top‑K candidates require raw vector reads from PAX blocks for exact re‑ranking. This is performance upgrade #5 (co‑located candidate scoring).

**Tombstone Policy (Deletes):**
- `DELETE` on a row with an embedding inserts a tombstone record in the delta buffer.
- Tombstone is effective immediately: the row is excluded from all queries.
- During the next merge:
  - All tombstones are removed from the new base graph.
  - A `_tombstone_count` metric tracks accumulation. If tombstones exceed 20% of total indexed vectors (configurable), an emergency merge is triggered regardless of delta buffer size.
- This is audit fix #7 (explicit tombstone lifecycle) and performance upgrade #4.

**Filter‑Aware Traversal:**
When a `SEMANTIC_MATCH` is combined with a strict `WHERE` filter (e.g., `WHERE price < 100 AND status = 'active'`), the graph traversal uses a **filter‑aware mode**:
- During graph neighbor exploration, nodes that fail the filter are skipped as candidates *but still traversed* (their edges are followed) to prevent the search from becoming disconnected in the graph. This is the ACORN‑style strategy referenced in performance upgrade #3.
- If the filtered candidate count is very small (below a threshold), the executor may fall back to a brute‑force scan over the filtered subset, bypassing the graph entirely (performance upgrade #2: adaptive query planning).

**PQ Codebook:**
- Trained on 10k–1M vectors (skip PQ entirely below 10k). Static in v1.
- **Drift monitoring**: a background thread periodically samples raw vectors and recomputes quantization error against the codebook. If error increases beyond a threshold, a warning is logged and a metric is emitted. No automatic refresh in v1 (that requires RGABH in v2). This addresses audit #8 (codebook drift risk).

### 3.3 Versioning & Semantic Search Semantics

*This section replaces §4.4 and §5 of the v1.1 spec, addressing audit findings #1, #9, #10 of the external audit and critical issues from the design review.*

**Merkle DAG Versioning:**
- Every write creates a PAX block with a commit timestamp and a monotonic version sequence number.
- A Merkle tree over block hashes forms the system state; each commit produces a **version root**.
- `AT VERSION` queries filter blocks where `commit_time ≤ target_version`.
- Named version tags can be created: `CREATE VERSION TAG 'q2_snapshot'`.

**Version Retention & Garbage Collection:**
- Default retention: 30 days (not 7 — see rationale below). Configurable per table.
- **Pinned tags are retention‑exempt.** A named version tag explicitly prevents GC of the blocks it references. This is audit fix #9: reproducible training snapshots cannot disappear.
- `DROP VERSION TAG` releases the pin. An explicit `EXPIRE VERSION` DDL allows timed auto‑expiry.
- Tags can be exported as Arrow Flight datasets (§3.8). Exported tags are listed in system catalog `_andromeda_versions`.

**Semantic Search at Historical Versions:**

*This fully resolves audit finding #1 — the most critical correctness issue in v1.1.*

v1 introduces **three consistency modes** for combining `SEMANTIC_MATCH` with `AT VERSION`. The default prevents silent incorrectness.

| Mode | Behavior | Use Case |
|------|----------|----------|
| `ROW_SNAPSHOT` (default) | Uses historical row data but **rejects** `SEMANTIC_MATCH` unless vector index can satisfy the version. **In v1, this means `SEMANTIC_MATCH` with `AT VERSION` raises an error.** | Offline training, audit, evaluation. Guarantees row-level correctness. |
| `SEMANTIC_FRESH` | Uses current index against historical row data. **Explicitly opt‑in.** Query returns a warning in the result metadata. | Exploratory queries where approximate vector results are acceptable. |
| `SEMANTIC_SNAPSHOT` | **v2 only.** Uses a versioned index to provide exact historical vector search. | Reproducible semantic audit. |

**Default behavior in v1:** A query combining `AT VERSION` and `SEMANTIC_MATCH` returns:

```
ERROR: SEMANTIC_MATCH with AT VERSION requires consistency mode.
HINT: Use CONSISTENCY 'SEMANTIC_FRESH' to search against current index
      with historical row data. Full versioned vector search is available
      in v2 with CONSISTENCY 'SEMANTIC_SNAPSHOT'.
```

This replaces the previous "documented limitation" with an active guardrail. Users cannot accidentally poison training data with anachronistic vector results.

**Implementation detail:** The version‑aware guardrail adds a check in the query planner: when an `AT VERSION` clause is present and a `SEMANTIC_MATCH` predicate references an embedding column, the planner checks the query's consistency hint. If absent, the error is raised before execution.

### 3.4 Embedding Inference Sidecar

*Addresses audit findings #6 (pending rows), #10 (cross‑platform), and design review bug #3 (dropped requests).*

**Architecture:**
- The sidecar is a **standalone binary** (Rust, using ONNX Runtime) that loads a sentence‑transformer model.
- Communication: Unix domain socket on Linux/macOS; named pipe on Windows. Protocol: simple length‑prefixed message framing with JSON payloads.
- The core database engine has no ML framework dependency.

**Lifecycle & Cross‑Platform Ownership:**

| Platform | Parent Death Detection |
|----------|------------------------|
| Linux | `prctl(PR_SET_PDEATHSIG)` — sidecar terminates on parent exit |
| macOS | `kqueue` monitoring parent PID — sidecar terminates on parent exit |
| Windows | Named pipe heartbeat — sidecar polls parent; if pipe breaks, sidecar exits |

Addresses audit #10.

**Back‑pressure & Data Loss Prevention:**
- Embedding request queue: 10,000 in‑flight items, bounded.
- When queue is full, new requests are written to the **persistent stale‑row backlog table** `_andromeda_embedding_backlog`.
- A low‑priority background scanner drains this table, re‑submitting requests when the sidecar has capacity.
- **No data is silently dropped.** Rows in the backlog remain stale (excluded from semantic search) but are guaranteed eventual processing.
- The metric `_embedding_backlog_depth` is exposed for monitoring.

Addresses design review critical issue #3.

**Crash Recovery:**
- Heartbeat: sidecar sends ping every 5 seconds.
- Missed heartbeats: if 3 consecutive pings are missed, database enters degraded mode.
- Degraded mode: writes proceed, embeddings marked stale, `SEMANTIC_MATCH` operates only on already‑indexed vectors.
- Restart: exponential backoff (1s, 2s, 4s, 8s, up to 60s max). On successful restart, the backlog scanner resumes processing.
- All writes committed during degraded mode are intact; embedding staleness is transient.

**Durability Contract for Embeddings:**
- Once the sidecar acknowledges an embedding computation and the background worker writes it to the LSM (flushed + fsync'd), the embedding is durable.
- Stale rows in the backlog are not durable — they are persisted in the backlog table (which is part of the LSM) but their embedding column remains `NULL` until processed.
- On crash, the backlog table survives in the WAL and LSM; recovery replays it and resumes processing.

### 3.5 AuroraSQL Language

PostgreSQL‑compatible SQL extended with:

**DDL:**
```sql
CREATE TABLE products (
    id          BIGINT PRIMARY KEY,
    title       TEXT,
    description TEXT EMBEDDING MODEL 'all-MiniLM-L6-v2' DIM 384,
    price       DECIMAL,
    created_at  TIMESTAMP
);
```

**SEMANTIC_MATCH:**
```sql
SEMANTIC_MATCH(column, 'query text', threshold)
-- optional consistency hint:
SEMANTIC_MATCH(column, 'query text', threshold) CONSISTENCY 'SEMANTIC_FRESH'
```

**AT VERSION (with consistency mode for semantic queries):**
```sql
SELECT * FROM products
AT VERSION 'q4_snapshot'
CONSISTENCY 'ROW_SNAPSHOT'  -- blocks SEMANTIC_MATCH (default)
WHERE price < 100;

-- Explicit fresh‑semantic mode:
SELECT * FROM products
AT VERSION 'q4_snapshot'
CONSISTENCY 'SEMANTIC_FRESH'
WHERE SEMANTIC_MATCH(description, 'camping', 0.7);
```

**Version Tag Management:**
```sql
CREATE VERSION TAG 'q4_snapshot' [FOR table_list] [EXPIRE AFTER interval];
DROP VERSION TAG 'q4_snapshot';
EXPIRE VERSION 'q4_snapshot' AFTER INTERVAL '90 days';
```

**Active learning and FEEDBACK are v2 only.**

### 3.6 PostgreSQL Wire Compatibility

*Addresses audit findings #2, #12.*

**v1: Tier 1 — Simple Query Protocol Only**

| Supported | Not Supported in v1 |
|-----------|---------------------|
| Simple query protocol (`Q` message) | Extended query protocol (Parse/Bind/Execute) |
| Basic DDL/DML | Prepared statements with parameters |
| pg_catalog stubs for `psycopg2`, SQLAlchemy | `COPY`, `SET` with session state |
| `SELECT`, `INSERT`, `UPDATE`, `DELETE` | Cursors, portals |
| Basic authentication (password, trust) | `information_schema` (full) |

**Compatibility statement (revised):**
> v1 supports the PostgreSQL simple query protocol. Most Python ORMs and clients (`psycopg2` in simple‑query mode, SQLAlchemy) work correctly. Clients requiring extended query protocol (prepared statements, parameterized execution) will receive a protocol‑level error with instructions to use simple query mode. Full extended protocol support is planned for v2.

Addresses audit #2: the claim is now scoped and honest.

### 3.7 Consistency Model — Revised

*Addresses audit finding #3 and design review consistency discussion.*

**v1 Consistency Guarantees (Revised):**

| Concern | Guarantee |
|---------|-----------|
| Row data (CRUD) | **Strict serializable.** Single‑node, single‑writer LSM provides true serializable isolation via snapshot isolation over versioned blocks. |
| Semantic search (index) | **Eventually fresh.** Embeddings are populated asynchronously. A row may be visible to non‑semantic queries but excluded from `SEMANTIC_MATCH` until its embedding is computed and indexed. |
| Semantic search + AT VERSION | **Guarded.** By default, rejected unless explicit consistency mode is chosen (§3.3). |

The previous v1.1 statement "v1 provides strict serializable isolation for all reads and writes" was **incorrect** because it did not account for the asynchronous embedding pipeline. The revised statement separates row‑data consistency (serializable) from semantic‑index freshness (eventual). This is audit fix #3.

**Timing guarantees:**
- A committed row is visible to non‑semantic queries immediately (within the same transaction's snapshot).
- The embedding for that row is typically available within 10–500 ms (depending on sidecar load and model latency).
- The `_embedding_stale` system column is readable by applications to check status.

### 3.8 Training Data Path

**Arrow Flight Export:**
- `SELECT … AT VERSION 'tag'` materializes as an Arrow RecordBatch stream via an embedded Arrow Flight server.
- Python API: `db.execute_arrow(query)` returns an iterator of `pyarrow.RecordBatch`.
- No GPU Direct Storage in v1; data flows through CPU memory.

**Reproducibility Contract:**
- A named version tag guarantees that repeated exports with the same tag produce identical data (byte‑identical Arrow batches).
- Tags are pin‑protected from garbage collection (§3.3).

### 3.9 Deployment Modes & Platform Support

*Addresses audit #10 (cross‑platform).*

| Mode | Platforms | Description |
|------|-----------|-------------|
| Embedded | Linux (x86‑64, ARM64), macOS (x86‑64, ARM64), Windows (x86‑64) | `import andromeda` — in‑process, no server. Sidecar spawned as child process. |
| Standalone server | Linux, macOS | `andromeda --server` — listens on port 5432. |
| Clustered | v2 only | Distributed mode with Raft, multi‑node. |

**Platform‑specific notes:**
- `prctl(PR_SET_PDEATHSIG)` — Linux only.
- `kqueue` parent monitoring — macOS.
- Named pipe heartbeat — Windows.
- The platform abstraction layer is in the sidecar manager module (pure Rust with `#[cfg(target_os)]` conditional compilation).

### 3.10 Binary Footprint & Module Tiers

*Addresses audit finding #11 — aggressive size target.*

**Revised Footprint:**

| Component | Size | Notes |
|-----------|------|-------|
| Core engine (Rust, statically linked) | **< 64 MB** | LSM, PAX, HNSW, SQL, wire protocol |
| Embedding sidecar (ONNX Runtime) | ~150 MB | Separate binary, downloaded on demand |
| Default embedding model | ~90 MB | all‑MiniLM‑L6‑v2 ONNX, downloaded on demand |
| Python client wheel | ~5 MB | Pure Python + compiled Rust extension |
| **Full install (all optional components)** | **< 350 MB** | When all features enabled |

**Installation tiers:**

1. **Minimal:** `andromeda-core` — database engine only. No embedding. ~60 MB.
2. **Standard:** `andromeda` — engine + embedded mode Python client. ~65 MB.
3. **Full:** `andromeda-full` — engine + sidecar + default model. ~300 MB.

Users choose tier at install time. The sidecar and model are fetched lazily if needed.

Addresses audit #11: size target is now realistic, with explicit module tiers.

### 3.11 Durability & Crash Recovery Contract

*This is the fully specified durability contract requested in the audit synthesis — covering data, embeddings, and sidecar state.*

**Data Durability:**
- Committed row data is durable after fsync on the PAX block and WAL entry.
- Crash recovery: on startup, replay WAL from last checkpoint. Maximum recovery time: < 30 seconds (bounded by checkpoint frequency).
- Power loss: committed data survives. Uncommitted transactions (no fsync) are rolled back.

**Embedding Durability:**
- A computed embedding is durable once its PAX block is flushed and fsync'd.
- Stale rows in the backlog table `_andromeda_embedding_backlog` are durable as rows (the row data is persisted) but the embedding column is `NULL` until processed.
- On crash, the backlog table is recovered from the LSM/WAL. The background scanner resumes processing.
- No embedding work is lost: either the embedding was written to a PAX block (durable) or the row is in the backlog (will be retried).

**Sidecar State:**
- The sidecar is **stateless**. It carries no durable state. On crash, any in‑flight embedding requests are retried from the backlog.
- The database tracks which rows have been submitted to the sidecar via the backlog table. On recovery, all unacknowledged submissions are re‑submitted.
- **Idempotency:** The sidecar's embedding computation is idempotent for the same input text. Re‑computing the same embedding after a crash produces the same result.

**Fault Injection Testing:**
- v1 ships with a chaos test harness: kill sidecar mid‑request, kill database mid‑flush, corrupt WAL block, fill disk.
- Pass criteria: no committed data loss, no silent row disappearance from semantic search, recovery within time bound.

### 3.12 v1 Limitations — Explicit

- No distributed clustering, no Raft, no multi‑node.
- No RGABH adaptive storage; buffer pool uses LRU.
- No active learning, no FEEDBACK SQL.
- No GPU Direct.
- HNSW index not versioned for `AT VERSION`.
- PQ codebook static; drift monitored but not auto‑refreshed.
- No `UPDATE` of embedded column source values.
- Simple query protocol only.
- Semantic search is eventually fresh, not transactionally consistent.
- Platform support: Linux primary, macOS/Windows with feature parity but less performance tuning.

---

## 4. v2 Full System — AI-Native Vision

*All v2 features are designed in detail but implemented after v1 ships. The design decisions below are informed by the full design review, external audit, and performance upgrade recommendations.*

### 4.1 RGABH-Driven Adaptive Storage

Building on the v1 foundation, v2 introduces **Row‑Gradient‑Aggregated Block Hotness** (RGABH) as the central nervous system of the storage hierarchy.

**Gradient Structure (per row):**
```
gradient = short_heat        // OLTP point‑read bursts → drives prefetch
         + γ × long_heat     // sustained importance → drives buffer pool admission
         + δ × training_heat // GPU training access → drives storage tiering
```

- `short_heat`: EMA with 30s half‑life. Incremented on OLTP point reads.
- `long_heat`: EMA with 10‑minute half‑life. Incremented on sustained access and model feedback.
- `training_heat`: EMA with 1‑hour half‑life. Incremented by GPU training access callbacks.

**Block Hotness & Buffer Pool:**
- Block hotness = Σ gradient(row) for all rows in the block.
- HotSet admission: blocks with hotness above dynamic threshold `T_hot`.
- Eviction: evict the block with lowest hotness.
- **Speculative prefetch**: blocks with rising `short_heat` velocity are prefetched from NVMe into HotSet.

**Adaptive Storage Tiering:**
- `T_hot` and `T_cold` thresholds auto‑tune to maintain 80% NVMe utilization.
- Blocks migrate automatically: Hot (NVMe) → Warm (NVMe) → Cold (object store) → Frozen (glacier).
- Quiescent metadata sweep (hourly) detects cooled‑off blocks and triggers demotion.

**PQ Codebook Refresh:**
- Drift detector monitors quantization error. When error exceeds threshold, a new codebook is trained on the current snapshot.
- Codebooks are versioned alongside data in the Merkle DAG.
- Old codebooks retained for `AT VERSION` queries.

### 4.2 Mutable ANN with Merge Policy

v2 enhances the v1 graph+delta design with:

- **Incremental merge**: instead of full rebuild, new HNSW layers are merged incrementally (similar to DiskANN's merge strategy).
- **Tombstone budget**: explicit per‑index tombstones limit; emergency merge triggered at threshold.
- **Quantization lifecycle**: per‑shard minimum training set, drift alarms, controlled refresh windows.
- **Recall‑latency SLO**: dynamic ef adjustment based on measured recall vs. latency target.
- **Filter‑aware graph traversal** (ACORN‑style): disconnected‑safe traversal for strict filters.

Performance upgrades #1, #6, #7 incorporated.

### 4.3 Distributed Clustering & Global Transactions

- **OLTP sharding**: consistent hash on primary key.
- **ANN sharding**: global IVF coarse quantizer routes queries to 1–2 shards; per‑shard HNSW fine‑grained search.
- **Consistency**: HLC‑based causal consistency with bounded staleness as default. Strict serializability via 2PC (Percolator‑style) for cross‑shard transactions.
- **Read replicas**: Raft‑shipped columnar replicas for HTAP scale‑out. Zero‑ETL, same binary.
- **IVF retraining**: coexisting quantizers during retraining; no immediate data migration.

### 4.4 Active Learning & Feedback Loop

- `_andromeda_predictions` table: applications insert `(row_id, model_id, prediction, actual)`.
- Drift detector: monitors accuracy, triggers PQ/index retraining.
- `ORDER BY ACTIVE_LEARNING`: pre‑computed uncertainty scores as a real column; background job refreshes on model update.
- `FEEDBACK` SQL: append‑only delta updates, preserving Merkle lineage.
- Cold‑start strategy: random → cluster‑then‑sample → uncertainty sampling (three‑stage, automatic transition).

### 4.5 Semantic Snapshot Guarantees

v2 delivers the fully correct versioned semantic search that v1 guards against:

- **`CONSISTENCY 'SEMANTIC_SNAPSHOT'`**: uses a versioned HNSW index built from the same snapshot as the row data.
- **Index snapshots**: each named version tag can include a reference to the HNSW index state at that point.
- **Cost**: additional storage for index snapshots; user opts in per tag.
- **Training reproducibility**: a tagged snapshot with `SEMANTIC_SNAPSHOT` consistency produces byte‑identical results on repeated export.

Audit fix #1 (fully resolved for v2) and performance upgrade #9.

### 4.6 Full PostgreSQL Protocol & BI Integration

- Extended query protocol (Parse/Bind/Execute) for prepared statements.
- `COPY` support, full `information_schema`, complete `pg_catalog`.
- Compatibility with Tableau, Metabase, DataGrip, DBeaver.
- Service‑side cursors for large result sets.

### 4.7 GPU-Direct & Hardware Acceleration

- Training scans bypass CPU: DMA directly from NVMe to GPU memory via GPUDirect Storage.
- `report_training_access` callback feeds `training_heat` into RGABH.
- Optional FPGA/SmartNIC offload for vector distance computation near storage.

### 4.8 Federated Queries & Privacy

- Data atoms carry ownership policies.
- Federated queries aggregate across organizations with differential privacy budgets.
- Secure aggregation built into query planner.

### 4.9 Plugin Marketplace

- `EmbeddingModel` trait defined in v1; v2 adds plugin system for custom models, active‑learning strategies, domain adapters.
- Revenue‑share model for third‑party plugins.

---

## 5. Implementation Roadmap

### v1 (4 months, 2–3 Rust engineers)

| Month | Deliverables |
|-------|-------------|
| **1** | Core LSM storage engine with PAX blocks, buffer pool (HotSet/ScanBuffer), WAL, checkpoint, crash recovery. Rust API only. |
| **2** | AuroraSQL parser (`sqlparser-rs` + extensions), PostgreSQL simple query protocol, Python embedded mode, DDL/DML end‑to‑end. |
| **3** | mmap'd HNSW base graph + delta flat buffer with merge, PQ codebook training (min‑10k threshold), embedding sidecar with backlog durability, `SEMANTIC_MATCH` with union+re‑rank. |
| **4** | Merkle DAG versioning, `AT VERSION` with semantic guardrails, version tags with pinning, Arrow Flight export, compatibility testing (psycopg2, SQLAlchemy), fault‑injection pass, public demo notebook. |

### v2 (12–18 months, expanded team)

| Phase | Deliverables |
|-------|-------------|
| **1** | RGABH adaptive storage: multi‑timescale EMA gradients, block hotness, auto‑tuned tiering (NVMe → S3 → Glacier), quiescent sweep, codebook drift detection + refresh. |
| **2** | Distributed clustering: consistent hash sharding, IVF+HNSW two‑level index, HLC causal consistency, Raft replication, read replicas for HTAP. |
| **3** | Active learning engine: uncertainty scoring, `FEEDBACK` SQL, drift detector, `_andromeda_predictions` table, cold‑start bootstrap. |
| **4** | Full PostgreSQL extended query protocol, BI tool compatibility, GPU Direct, federated queries, plugin marketplace. |

---

## 6. Appendices

### A. Audit Trail — Issues Resolved from v1.1

| # | Issue | Resolution |
|---|-------|------------|
| 1 | `AT VERSION` + `SEMANTIC_MATCH` silent incorrectness | Guardrail: default rejection, explicit consistency modes (§3.3) |
| 2 | PostgreSQL compatibility overstatement | Scoped to simple query protocol, documented limitations (§3.6) |
| 3 | Consistency claim too broad for async embeddings | Revised to row‑serializable + semantic‑eventual (§3.7) |
| 4 | mmap traversal "zero I/O" claim | Corrected to "no explicit I/O syscalls; page faults incur disk reads" (§3.2) |
| 5 | Candidate scoring random‑read heavy | PQ codes co‑located in graph payload; only final re‑rank reads LSM (§3.2) |
| 6 | Flat index latency bomb | Replaced with bounded delta buffer + merge trigger (§3.2) |
| 7 | Missing tombstone lifecycle | Explicit tombstone policy with count‑based emergency merge (§3.2) |
| 8 | Static codebook drift risk | Drift monitoring metric + warning; auto‑refresh in v2 (§3.2) |
| 9 | Version retention vs. reproducibility | Pinned tags exempt from GC; 30‑day default retention (§3.3) |
| 10 | Linux‑only parent death | Cross‑platform: prctl (Linux), kqueue (macOS), named pipe heartbeat (Windows) (§3.4) |
| 11 | Aggressive binary size target | Raised to 64MB core, with explicit module tiers (§3.10) |
| 12 | Timeline plausibility | Protocol scope intentionally shallow (simple query only); 4‑month plan (§5) |

### B. Performance Upgrades Incorporated

| # | Upgrade | Where |
|---|---------|-------|
| 1 | Delta buffer + merge instead of large mutable flat index | §3.2 |
| 2 | Adaptive query planning (flat scan vs. graph traversal) | §3.2 |
| 3 | Filter‑aware graph traversal (ACORN‑style) | §3.2 |
| 4 | Tombstone budget + emergency merge | §3.2 |
| 5 | Co‑located PQ codes in graph payload | §3.2 |
| 6 | Dynamic ef bounded by latency SLO + recall target | §3.2 |
| 7 | Quantization lifecycle (min training set, drift alarms, refresh) | §3.2 (v1 monitoring), §4.1 (v2 refresh) |
| 8 | Semantic snapshot guardrails | §3.3 (v1 guard), §4.5 (v2 snapshots) |
| 9 | Per‑query consistency modes | §3.3 |
| 10 | Reproducibility mode with pinned tags | §3.3, §3.8 |

### C. Key Design Principles (Non‑Negotiable)
- No silent incorrectness: guard, don't just document.
- Honest performance claims: all numbers qualified with conditions.
- Graceful degradation: overload → backlog, not data loss.
- Version everything: data, codebooks, indices — linked in Merkle DAG.
- Ship v1: the best spec is one that becomes a working system.

### D. References
- HNSW paper: https://arxiv.org/abs/1603.09320
- DiskANN: https://www.microsoft.com/en-us/research/publication/diskann
- TiDB/TiFlash HTAP architecture
- PostgreSQL wire protocol: https://www.postgresql.org/docs/current/protocol-overview.html
- hnswlib: https://github.com/nmslib/hnswlib
- pgvector: https://github.com/pgvector/pgvector
- Weaviate vector index docs: https://docs.weaviate.io
- Qdrant indexing docs: https://qdrant.tech/documentation/concepts/indexing/
- FAISS guidelines: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
- ANN-Benchmarks: https://github.com/erikbern/ann-benchmarks

---

*This document is the authoritative reference for Andromeda v1 and v2. Every design decision is traceable to a specific issue in the audit trail. The v1 specification is frozen; implementation begins against §3.*