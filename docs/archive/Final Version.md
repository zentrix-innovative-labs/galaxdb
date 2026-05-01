# GalaxDB Architecture Specification
## Final Version — v1 Hardened, v2 Fully Designed

**Status:** Design locked. All design review issues resolved. All audit loopholes closed. Implementation-ready for v1.

**Target:** v1 — 4 months, 2–3 Rust engineers. v2 — 12–18 months, expanded team.

---

## Table of Contents

1. [Vision & Design Principles](#1-vision--design-principles)
2. [Architecture Overview](#2-architecture-overview)
3. [v1 Core System](#3-v1-core-system)
   - 3.1 Storage Engine
   - 3.2 Vector Index & Mutable ANN
   - 3.3 Versioning & Semantic Search Semantics
   - 3.4 Embedding Inference Sidecar
   - 3.5 AuroraSQL Language
   - 3.6 PostgreSQL Wire Compatibility
   - 3.7 Consistency Model
   - 3.8 Training Data Path
   - 3.9 Deployment Modes & Platform Support
   - 3.10 Binary Footprint & Module Tiers
   - 3.11 Durability & Crash Recovery Contract
   - 3.12 v1 Limitations
4. [v2 Full System](#4-v2-full-system)
   - 4.1 RGABH-Driven Adaptive Storage
   - 4.2 Mutable ANN with Incremental Merge
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

GalaxDB is the **AI-native database** that unifies transactional, analytical, vector, and graph data into a single engine. It eliminates the five-database spaghetti and actively improves the AI built on top of it.

### Core Principles

1. **Unified, not just integrated.** One data atom carries relational fields, embeddings, binaries, and provenance lineage. No separate systems to synchronize.

2. **Honest semantics above all.** Every feature's limitations are documented as clearly as its capabilities. Users must never be surprised by silent incorrectness. Guardrails prevent misuse; documentation explains edge cases.

3. **Start small, scale seamlessly.** A ~60 MB embedded binary for a laptop that grows into a million-node global cluster without changing the data model or application code.

4. **AI-first architecture.** Embeddings, versioned snapshots, and feedback loops are first-class primitives designed into the storage engine and query planner, not bolt-on extensions.

5. **Falsifiable claims.** Every performance number, every consistency guarantee, every durability promise is stated with measurable conditions and a reproducible benchmark harness.

---

## 2. Architecture Overview

GalaxDB's architecture is layered. Each version adds capabilities while preserving the same foundational model.

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
│  Storage (NVMe, object store, glacier — tiered v2)│
└──────────────────────────────────────────────────┘
```

- **v1** implements the bottom three layers as a single-node embedded/standalone system.
- **v2** extends with distributed clustering, RGABH adaptive tiering, active learning, and hardware acceleration.
- **Every layer** uses the same Data Atom model and the same Merkle DAG versioning.

---

## 3. v1 Core System

### 3.1 Storage Engine

#### LSM-Tree with PAX Blocks

GalaxDB stores data in **PAX (Partition Attributes Across) blocks** — each block holds approximately 1,000 rows with columns stored contiguously within the block.

**Write Path:**
1. Incoming rows accumulate in a lock-free Bw-Tree memory buffer.
2. When the buffer reaches 64 MB, it is sealed and flushed as an immutable PAX block.
3. Flush includes fsync before acknowledging the commit.
4. A write-ahead log (WAL) records the transaction for atomicity across multiple blocks.

**Compaction:**
- A background compactor merges smaller blocks into larger sorted runs (standard LSM leveling).
- Compaction always produces new blocks; old blocks are reclaimed only after all active snapshots release references and version retention windows expire.

**Read Paths:**
- **Point query:** Sparse primary key index → block ID + row offset → single small random read.
- **Column scan:** Sequential reads of the needed column chunks from blocks in sorted order.

**Buffer Pool:**
The buffer pool is split into two isolated partitions:

| Partition | Share | Eviction | Purpose |
|-----------|-------|----------|---------|
| **HotSet** | 70% of RAM | LRU | OLTP point-read blocks. Can be pinned by version tag retention. |
| **ScanBuffer** | 30% of RAM | Clock-sweep | OLAP scan-prefetched blocks. **Never evicts a HotSet-resident block.** |

ScanBuffer blocks never promote to HotSet based on scan access alone. This prevents a large analytical query from flushing the transactional working set.

#### PAX Block Physical Layout

| Section | Contents |
|---------|----------|
| Header | Block ID, row count, column descriptors, min/max per column, compression metadata, CRC-32 checksum |
| Column chunks | Fixed-width columns stored contiguously; variable-width (TEXT, BLOB) with length prefixes |
| Embedding column | Dense float32 arrays, optionally PQ-compressed (see §3.2) |
| Row offset table | Byte offset of each row within the block |

All blocks are self-describing with a format version byte. This enables forward compatibility as the format evolves.

---

### 3.2 Vector Index & Mutable ANN

The v1 vector index is **mutable from the start**, avoiding the brittle "immutable after initial build" model that would make incremental writes impractical.

#### Architecture: Base Graph + Delta Buffer

Two structures coexist and periodically merge:

1. **Base graph** — An HNSW index covering all vectors at the time of the last merge. Stored as a **memory-mapped file** (`.hnsw`) on disk. Read-only between merges. Graph traversal uses direct memory access via mmap.

2. **Delta buffer** — A small in-memory flat index (exact k-NN) holding vectors inserted or updated since the last merge. Backed by a persistent write-ahead log for durability across crashes.

The delta buffer uses the **same unified WAL** as the LSM store, with a distinct `DELTA_INSERT` record type. Recovery replays both row-store and delta-buffer state from the same log sequence.

#### Query Path

1. `SEMANTIC_MATCH` probes **both** the base graph and the delta buffer.
2. Results are **unioned** and re-ranked against raw vectors from LSM PAX blocks for final exact cosine scores.
3. Candidate PQ codes are **co-located in the graph payload** alongside adjacency lists, so graph traversal never touches the LSM. Only the final top-K re-ranking reads raw vectors from PAX blocks.

#### Merge Policy

When the delta buffer size exceeds `max(10,000, total_indexed_vectors × 0.01)`:

1. A background job builds a new HNSW graph incorporating all vectors from both base and delta.
2. The new graph is built in a **shadow file**. Queries continue against the old base + delta.
3. When ready, an atomic pointer swap routes new queries to the new graph.
4. The old graph is freed when all in-flight queries release their references (reference-counted, `Arc`-like semantics).
5. Only one merge runs at a time; subsequent triggers are queued and deduplicated.

#### Dynamic Search Depth (ef)

`ef` is not a fixed static value. A proportional feedback controller adjusts it:

- Default ef = 100.
- If query p99 latency exceeds the configured SLO (default 5 ms), ef is reduced.
- If measured recall drops below target (periodic sampling against ground truth), ef is increased.
- The controller uses exponential smoothing to avoid oscillation.

#### mmap Behavior (Honest Semantics)

The spec does not claim "zero I/O" for HNSW traversal. Pointer dereference on an mmap'd region triggers **page faults** when the underlying page is not resident in the OS page cache. Under memory pressure or cold start, traversal incurs physical disk reads. The mmap design **avoids explicit read() system calls** in the hot path, which reduces CPU overhead, but does not eliminate I/O.

#### Filter-Aware Traversal

When a `SEMANTIC_MATCH` is combined with a strict `WHERE` clause:

- During graph neighbor exploration, nodes that fail the filter are **skipped as candidates** but **still traversed** (their edges are followed). This prevents the search from becoming disconnected in the graph — the ACORN-style strategy.
- If the filtered candidate set is small (below an adaptive threshold), the executor may fall back to a brute-force scan over the filtered subset, bypassing the graph entirely. This is **adaptive query planning**: the optimizer estimates cardinality and chooses the cheaper path.

#### Tombstone Policy

- `DELETE` on a row with an embedding inserts a tombstone record in the delta buffer. The row is immediately excluded from all queries.
- During the next merge, all accumulated tombstones are removed from the new base graph.
- A `_tombstone_count` metric tracks accumulation. If tombstones exceed 20% of total indexed vectors (configurable), an **emergency merge** is triggered regardless of delta buffer size.

#### Product Quantization (PQ)

To reduce storage and I/O for embeddings:

- A codebook is trained using k-means on the table's vectors.
- **Training policy:**
  - If total vectors < 10,000: skip PQ entirely. Store raw float32 vectors.
  - If between 10,000 and 1,000,000: train PQ on all available vectors.
  - If > 1,000,000: randomly sample 1,000,000 vectors for training.
- Each vector is stored as a short PQ code alongside its raw floats in the PAX block.
- The HNSW graph references PQ codes for approximate search; raw floats are only read for final re-ranking.

**Bootstrap transition behavior:** Before the PQ codebook exists (`< 10,000` vectors), graph payload entries store raw float32 vectors instead of PQ codes. After codebook training completes, a background rewrite populates PQ codes for existing nodes. During this transition, each node carries a `has_pq_code` flag; query execution falls back to raw-float scoring when the flag is false.

**Drift monitoring (v1):** A background thread periodically samples raw vectors and recomputes quantization error against the codebook. If error increases beyond a configurable threshold, a warning is logged and a metric is emitted. Automatic codebook refresh requires v2 (RGABH-driven drift detection).

#### DML on Embedded Columns

| Operation | v1 Support |
|-----------|-----------|
| `INSERT` with embedded column | ✅ Row written; embedding populated asynchronously |
| `UPDATE non_embedded_column` | ✅ Works normally |
| `UPDATE embedded_column_source` | ❌ Blocked. Workaround: `DELETE` + `INSERT` |
| `DELETE` | ✅ Tombstone written; row excluded immediately |

This constraint is specifically and only on the source column of an embedding. All other DML is fully functional. The restriction will be lifted in v2.

---

### 3.3 Versioning & Semantic Search Semantics

#### Merkle DAG Versioning

- Every write creates a PAX block with a commit timestamp and a monotonic version sequence number.
- A Merkle tree over block hashes forms the system state; each commit produces a **version root**.
- `AT VERSION timestamp_or_tag` filters blocks where `commit_time ≤ target_version`.
- Named version tags: `CREATE VERSION TAG 'q4_snapshot'`. Tags are **GC-exempt** — they pin the blocks they reference until explicitly dropped.

#### Version Retention

- Default retention: **30 days**. Configurable per table.
- **Pinned tags are retention-exempt.** A named version tag prevents garbage collection of its referenced blocks.
- `DROP VERSION TAG` releases the pin.
- `EXPIRE VERSION 'tag' AFTER INTERVAL '90 days'` sets a timed auto-expiry.
- Tags are listed in the system catalog `_GalaxDB_versions`.

#### Semantic Search at Historical Versions

v1 introduces **three consistency modes** for combining `SEMANTIC_MATCH` with `AT VERSION`. The default **prevents silent incorrectness** — it actively rejects the ambiguous query.

| Mode | Behavior | Availability |
|------|----------|--------------|
| `ROW_SNAPSHOT` (default) | Uses historical row data. **Rejects** `SEMANTIC_MATCH` unless a versioned index can satisfy it. | v1 |
| `SEMANTIC_FRESH` | Uses the current HNSW index against historical row data. **Explicit opt-in.** Results carry a warning in metadata. | v1 |
| `SEMANTIC_SNAPSHOT` | Uses a versioned HNSW index for exact historical vector search. | v2 only |

**Default behavior in v1:** A query combining `AT VERSION` and `SEMANTIC_MATCH` without an explicit consistency mode returns:

```
ERROR: SEMANTIC_MATCH with AT VERSION requires a consistency mode.
HINT: Use CONSISTENCY 'SEMANTIC_FRESH' to search against the current index
      with historical row data. Full versioned vector search is available
      in v2 with CONSISTENCY 'SEMANTIC_SNAPSHOT'.
```

**Implementation:** The query planner checks for the combination of `AT VERSION` and `SEMANTIC_MATCH`. If present without a consistency hint, the error is raised before execution begins. This is a compile-time guard, not a runtime surprise.

#### Version Tag DDL

```sql
-- Create a pinned, GC-exempt snapshot
CREATE VERSION TAG 'q4_training_snapshot';

-- Create a snapshot with auto-expiry
CREATE VERSION TAG 'weekly_backup' EXPIRE AFTER INTERVAL '90 days';

-- Release a tag (unblocks GC of its blocks)
DROP VERSION TAG 'weekly_backup';

-- Add or change expiry on an existing tag
EXPIRE VERSION 'q4_training_snapshot' AFTER INTERVAL '180 days';
```

---

### 3.4 Embedding Inference Sidecar

#### Architecture

- The sidecar is a **standalone Rust binary** using ONNX Runtime to load and serve a sentence-transformer model.
- Communication with the database engine is over a **Unix domain socket** (Linux/macOS) or **named pipe** (Windows).
- Protocol: length-prefixed message framing with JSON payloads.
- The core database engine has **no ML framework dependency**. If the sidecar is unavailable, the database continues operating (in degraded semantic mode).

#### Lifecycle & Cross-Platform Ownership

| Platform | Parent Death Detection |
|----------|------------------------|
| Linux | `prctl(PR_SET_PDEATHSIG)` — sidecar terminates when parent exits |
| macOS | `kqueue` monitoring parent PID — sidecar terminates when parent exits |
| Windows | Named pipe heartbeat — sidecar polls parent; if pipe breaks, sidecar exits |

When the `GalaxDB` process starts (CLI or embedded), it spawns the sidecar as a child process. When the parent exits (cleanly or via crash), the sidecar terminates automatically.

#### Back-Pressure & Data Loss Prevention

- The embedding request queue has a hard limit of **10,000 in-flight items**.
- When the queue is full, new requests are **not dropped**. Instead, the row ID is written to the **persistent backlog table** `_GalaxDB_embedding_backlog`.
- A low-priority background scanner periodically drains this table, re-submitting requests when the sidecar has capacity.
- **No data is silently lost.** Rows in the backlog remain stale (excluded from semantic search) but are guaranteed eventual processing.
- The metric `_embedding_backlog_depth` is exposed for monitoring.

#### Crash Recovery

- **Heartbeat:** Sidecar sends a ping every 5 seconds. If 3 consecutive pings are missed, the database enters degraded mode.
- **Degraded mode:** Writes proceed normally. Embeddings are marked stale. `SEMANTIC_MATCH` operates only on already-indexed vectors.
- **Restart:** Exponential backoff (1s, 2s, 4s, 8s, up to 60s max). On successful restart, the backlog scanner resumes.
- All committed writes during degraded mode are intact. Embedding staleness is transient.

#### Durability Contract for Embeddings

- Once the sidecar acknowledges an embedding computation and the background worker writes it to a PAX block (flushed + fsync'd), the embedding is **durable**.
- Stale rows in the backlog table are **durable as row data**, but the embedding column is **not yet materialized**: it remains `NULL` until the sidecar processes the request and the embedding is written back to the row. No data loss occurs; the backlog record itself is fully durable.
- On crash, the backlog table survives in the WAL and LSM. Recovery replays it and resumes processing.
- **Idempotency:** The sidecar's embedding computation is deterministic for the same input text. Re-computing after a crash produces the same result.

---

### 3.5 AuroraSQL Language

GalaxDB v1 supports PostgreSQL-compatible SQL extended with AI-native primitives.

#### DDL with Auto-Embedding

```sql
CREATE TABLE products (
    id          BIGINT PRIMARY KEY,
    title       TEXT,
    description TEXT EMBEDDING MODEL 'all-MiniLM-L6-v2' DIM 384,
    image       BLOB  EMBEDDING MODEL 'clip-vit-base' DIM 512,
    price       DECIMAL,
    created_at  TIMESTAMP DEFAULT NOW()
);
```

- `EMBEDDING MODEL` specifies the model name known to the sidecar. The column is automatically populated on `INSERT`.
- `DIM` is optional; if omitted, the sidecar is queried for the model's output dimension.
- A table may have multiple embedding columns.

#### SEMANTIC_MATCH

```sql
SELECT title, price, similarity
FROM products
WHERE price < 100
  AND SEMANTIC_MATCH(description, 'lightweight camping tent', 0.7)
ORDER BY similarity DESC
LIMIT 20;
```

- `threshold` is minimum cosine similarity (float 0–1).
- Combines with standard `WHERE` clauses. The planner pushes down filters where beneficial.
- Returns a synthetic `similarity` column in the result set.

#### AT VERSION with Consistency Modes

```sql
-- Row-level time travel (no semantic search)
SELECT * FROM products
AT VERSION 'q4_snapshot'
WHERE price < 100;

-- Semantic search with explicit fresh-index opt-in
SELECT * FROM products
AT VERSION 'q4_snapshot'
CONSISTENCY 'SEMANTIC_FRESH'
WHERE SEMANTIC_MATCH(description, 'camping gear', 0.7);
```

#### Version Tag Management

```sql
CREATE VERSION TAG 'q4_snapshot';
CREATE VERSION TAG 'weekly' EXPIRE AFTER INTERVAL '90 days';
DROP VERSION TAG 'weekly';
EXPIRE VERSION 'q4_snapshot' AFTER INTERVAL '180 days';
```

Active learning and `FEEDBACK` SQL are **v2 only**.

---

### 3.6 PostgreSQL Wire Compatibility

v1 implements **Tier 1** of the PostgreSQL wire protocol. The compatibility claim is scoped and honest.

#### Supported (v1)

| Feature | Notes |
|---------|-------|
| Simple query protocol (`Q` message) | Single SQL string per message |
| Basic DDL | `CREATE TABLE`, `DROP TABLE` |
| Basic DML | `INSERT`, `SELECT`, `UPDATE`, `DELETE` (within constraints) |
| pg_catalog stubs | `pg_class`, `pg_attribute`, `pg_type` — sufficient for `psycopg2` (simple mode) and SQLAlchemy introspection |
| Basic authentication | Password, trust |

#### Not Supported in v1

| Feature | v2 Plan |
|---------|---------|
| Extended query protocol (Parse/Bind/Execute) | v2 |
| Prepared statements with parameters | v2 |
| `COPY` | v2 |
| `SET` (session runtime parameters) | v2 |
| Cursors, portals | v2 |
| Full `information_schema` | v2 |

**Note:** The simple `SET` command (e.g., `SET search_path TO …`) is not supported in v1. All runtime configuration is provided via connection string parameters or startup flags.

**Compatibility statement:** v1 supports the PostgreSQL simple query protocol. Most Python ORMs and clients (`psycopg2` in simple-query mode, SQLAlchemy) work correctly. Clients requiring extended query protocol will receive a protocol-level error with instructions to use simple query mode. Full extended protocol support is planned for v2.

---

### 3.7 Consistency Model

The v1 consistency model is split into two separate concerns, because the row store and the semantic index have different freshness properties.

| Concern | Guarantee | Details |
|---------|-----------|---------|
| **Row data (CRUD)** | **Strict serializable** | Single-node, single-writer LSM provides true serializable isolation via snapshot isolation over versioned blocks. Readers never block writers. |
| **Semantic search** | **Eventually fresh** | Embeddings are populated asynchronously. A newly inserted row is visible to non-semantic queries immediately but may be excluded from `SEMANTIC_MATCH` until its embedding is computed and indexed (typically 10–500 ms). |
| **Semantic search + AT VERSION** | **Guarded** | By default, the combination is rejected unless the user provides an explicit consistency mode (§3.3). |

**Timing guarantees:**
- Committed row: visible immediately (within the transaction's snapshot) for non-semantic queries.
- Embedding: typically available within 10–500 ms, depending on sidecar load and model latency.
- The system column `_embedding_stale` is readable by applications: `SELECT _embedding_stale FROM products WHERE id = 42`.

**Reader-visible atomicity rule:** Embedding materialization uses the same LSM update path as any other row update. The embedding value and `_embedding_stale` transition (`true -> false`) are written in the same row version. Readers therefore observe a consistent pair: either stale+NULL embedding or fresh+materialized embedding, never a mixed state.

---

### 3.8 Training Data Path

#### Arrow IPC Export (v1)

- A `SELECT … AT VERSION 'tag'` query can be materialized as an **Arrow RecordBatch stream** via the in-process `execute_arrow()` API.
- Python API: `db.execute_arrow(query)` returns an iterator of `pyarrow.RecordBatch` objects, suitable for direct ingestion into PyTorch, TensorFlow, or JAX.
- v1 does not expose Arrow Flight over gRPC; this keeps the embedded binary lean and avoids network-protocol complexity during initial delivery.
- No GPU Direct Storage in v1; data flows through CPU memory.

#### Reproducibility Contract

- A named version tag guarantees that **repeated exports with the same tag produce byte-identical Arrow batches**.
- Tags are pin-protected from garbage collection (§3.3).
- The Arrow schema is stable per tag: column order and types match the table definition at tag creation time.

---

### 3.9 Deployment Modes & Platform Support

| Mode | Platforms | Description |
|------|-----------|-------------|
| **Embedded** | Linux (x86-64, ARM64), macOS (x86-64, ARM64), Windows (x86-64) | `import GalaxDB` in Python; library runs in-process. Sidecar spawned as child. |
| **Standalone server** | Linux, macOS | `GalaxDB --server` listens on localhost:5432 (PostgreSQL wire protocol). |
| **Clustered** | Linux (x86-64) | v2 only. Multi-node with Raft consensus, distributed ANN. |

Platform-specific sidecar ownership is handled via conditional compilation (`#[cfg(target_os)]`):
- Linux: `prctl(PR_SET_PDEATHSIG)`
- macOS: `kqueue`
- Windows: named pipe heartbeat

---

### 3.10 Binary Footprint & Module Tiers

| Installation Tier | Contents | Size |
|-------------------|----------|------|
| **Minimal** (`GalaxDB-core`) | Database engine: LSM, PAX, HNSW, SQL, wire protocol | **< 64 MB** |
| **Standard** (`GalaxDB`) | Core + Python embedded client | **< 70 MB** |
| **Full** (`GalaxDB-full`) | Standard + embedding sidecar + default model | **< 350 MB** |

Users select the tier at install time. The sidecar and model are fetched lazily when first needed if not pre-installed. The `pip install GalaxDB-db` package defaults to the Standard tier with lazy fetch for the sidecar.

---

### 3.11 Durability & Crash Recovery Contract

#### Data Durability
- Committed row data is durable after fsync on the PAX block and WAL entry.
- Crash recovery: on startup, replay the unified WAL from the last checkpoint. Maximum recovery time is < 30 seconds, bounded by checkpoint frequency (configurable, default every 60 seconds).
- Power loss: committed data survives. Uncommitted transactions (no fsync acknowledgment) are rolled back.

WAL replay includes all record families (row updates, compaction metadata, and vector delta-buffer records such as `DELTA_INSERT`) in a single ordered recovery path.

#### Embedding Durability
- A computed embedding is durable once its PAX block is flushed and fsync'd.
- Stale rows in the backlog table are **durable as row data**, but the embedding column is **not yet materialized**: it remains `NULL` until the sidecar processes the request and the embedding is written back to the row. No data loss occurs; the backlog record itself is fully durable.
- On crash, the backlog table is recovered from the LSM/WAL. The background scanner resumes processing.
- No embedding work is lost: either the embedding was written to a PAX block (durable) or the row is in the backlog (will be retried).

#### Sidecar State
- The sidecar is **stateless**. It carries no durable state of its own.
- On sidecar crash, all in-flight embedding requests are retried from the backlog table.
- The database tracks which rows have been submitted to the sidecar. On recovery, all unacknowledged submissions are re-submitted.
- Embedding computation is idempotent for the same input text.

#### Fault Injection Testing
v1 ships with a chaos test harness covering:
- Kill sidecar mid-request
- Kill database mid-flush
- Corrupt a WAL block
- Fill the disk
- Power cycle

**Pass criteria:** No committed data loss. No silent row disappearance from semantic search. Recovery within time bound. Backlog fully reprocessed after sidecar restart.

---

### 3.12 v1 Limitations

These are explicit, documented constraints for v1. Each maps to a v2 feature.

| Limitation | v2 Resolution |
|------------|---------------|
| No distributed clustering | Raft + IVF+HNSW in v2 |
| No RGABH adaptive storage | Multi-timescale gradient engine in v2 |
| No active learning or FEEDBACK SQL | Uncertainty scoring + feedback loop in v2 |
| No GPU Direct | GPUDirect Storage in v2 |
| HNSW index not versioned for `AT VERSION` | Index snapshots in v2 |
| PQ codebook static (drift monitored, not refreshed) | Auto-refresh on drift in v2 |
| No `UPDATE` of embedded column source values | Full DML on embeddings in v2 |
| Simple query protocol only | Extended protocol in v2 |
| Semantic search is eventually fresh | Transactional embedding option in v2 |

---

## 4. v2 Full System

All v2 features are designed in detail but implemented after v1 ships. The decisions below are informed by the full design review, external audit, and performance upgrade analysis.

### 4.1 RGABH-Driven Adaptive Storage

**Row-Gradient-Aggregated Block Hotness (RGABH)** replaces static buffer pool policies with a gradient-based adaptive system that responds to actual workload patterns.

#### Per-Row Gradient Structure

```
gradient = short_heat        // OLTP point-read bursts → drives prefetch
         + γ × long_heat     // sustained importance → drives buffer pool admission
         + δ × training_heat // GPU training access → drives storage tiering only
```

- **`short_heat`** — EMA with 30-second half-life. Incremented on OLTP point reads. Drives speculative prefetch.
- **`long_heat`** — EMA with 10-minute half-life. Incremented on sustained access and model feedback. Drives HotSet admission.
- **`training_heat`** — EMA with 1-hour half-life. Incremented by GPU training access callbacks (`report_training_access`). Drives storage tiering only; **excluded** from prefetch to prevent batch-training false positives.

#### Block Aggregation

- **Block hotness** = Σ gradient(row) for all rows in the block. Recalculated incrementally via a dirty-block queue (only blocks with gradient changes are rescored).
- A quiescent metadata sweep (hourly) handles cold data that never enters the dirty queue, applying lazy decay and triggering tier demotion.
- On LSM compaction, block hotness is recomputed as the exact sum of constituent rows' current gradients.

#### Buffer Pool (v2)

- HotSet admission: blocks with hotness above the dynamic threshold `T_hot`.
- Eviction: evict the block with the lowest hotness (not LRU).
- Speculative prefetch: blocks with rising `short_heat` velocity are prefetched from NVMe into HotSet via the low-priority I/O queue.

#### Adaptive Storage Tiering

| Tier | Condition | Storage |
|------|-----------|---------|
| **Hot** | hotness > `T_hot` | NVMe |
| **Warm** | `T_cold` < hotness ≤ `T_hot` | NVMe, eligible for background move |
| **Cold** | hotness ≤ `T_cold` | Object storage (S3) with metadata stub |
| **Frozen** | hotness = 0 for > 7 days | Glacier, no stub |

**Auto-tuning:** `T_hot` and `T_cold` are controlled by a feedback loop targeting 80% NVMe utilization. If utilization > 85%, `T_hot` is raised 5% (fewer blocks in Hot). If < 75%, `T_hot` is lowered 5%. Adjustments are clamped to ±20% per cycle to prevent oscillation. `T_cold` tracks `T_hot` at a configurable ratio (default 0.3).

#### PQ Codebook Lifecycle (v2)

- The same drift detector that triggers IVF quantizer retraining (see §4.3) also triggers PQ codebook refresh.
- New codebooks are trained on the current snapshot, versioned in the Merkle DAG, and linked to the data versions they cover.
- Old codebooks are retained for `AT VERSION` queries with `CONSISTENCY 'SEMANTIC_SNAPSHOT'`.

---

### 4.2 Mutable ANN with Incremental Merge

v2 enhances the v1 graph+delta design with:

- **Incremental merge:** Instead of a full rebuild from scratch, new vectors are merged into the existing HNSW graph in layers. This is similar to DiskANN's incremental merge strategy and reduces the cost of index maintenance from O(N log N) to O(K log N) where K is the number of new vectors.
- **Tombstone budget:** Explicit per-index tombstone limit (configurable, default 20% of indexed vectors). Emergency merge triggered at threshold.
- **Recall-latency SLO:** The dynamic ef controller uses both latency and measured recall signals. If recall drops below target while latency is within budget, ef increases. If latency exceeds SLO, ef decreases regardless of recall.
- **Filter-aware traversal:** ACORN-style disconnected-safe filtered search. The planner chooses graph traversal vs. brute-force based on filter cardinality estimation.
- **Quantization lifecycle:** Minimum training set per shard, drift alarms, controlled refresh windows with coexisting codebooks during transition.

---

### 4.3 Distributed Clustering & Global Transactions

#### Sharding Strategy

- **OLTP:** Consistent hash on primary key distributes rows uniformly across shards. Each shard owns its LSM store.
- **ANN:** A global IVF coarse quantizer is trained over the embedding space. At query time, the quantizer routes the query to 1–2 most relevant shards. Within each target shard, fine-grained HNSW search executes.

#### IVF Quantizer Management
- Trained at shard creation time.
- Retrained when embedding distribution drift is detected (same drift detector as PQ codebook refresh).
- During retraining, old and new quantizers coexist. New writes use the new quantizer; queries consult both and union the candidate shard sets.
- Data migration happens lazily during the next LSM compaction of affected shards.

#### Consistency
- **Default: Causal consistency with bounded staleness** via Hybrid Logical Clocks (HLC). Provides read-your-writes, monotonic reads, and consistent prefix.
- **Strict serializability:** Available via a 2PC protocol (Percolator-style) for cross-shard transactions. Opt-in per transaction due to the latency cost.

#### HTAP Scale-Out
- Read-only replicas within each Raft group handle analytical scans.
- Replicas receive updates via Raft log shipping — identical binary, zero ETL.
- The SQL planner routes OLAP queries to replicas when freshness constraints allow.

---

### 4.4 Active Learning & Feedback Loop

#### Prediction Tracking
- Applications insert prediction outcomes into the system table `_GalaxDB_predictions`:

```sql
INSERT INTO _GalaxDB_predictions (row_id, model_id, prediction, actual, timestamp)
VALUES (42, 'fraud_v3', 'legitimate', 'fraud', NOW());
```

- The drift detector reads this table to monitor model accuracy over time.

#### Active Learning
- `ORDER BY ACTIVE_LEARNING(target_column, strategy)` retrieves the rows that would most improve the model if labeled next.
- Uncertainty scores are **pre-computed by a background job** and stored as the real column `_al_uncertainty`. The SQL clause becomes a simple `ORDER BY _al_uncertainty DESC` — indexable and O(log n).
- **Cold-start strategy:** Three stages, automatic transition:
  - **< 50 labeled rows:** Random sampling.
  - **50–200 labeled rows:** Cluster-then-sample (k-means over embeddings, sample from each cluster).
  - **> 200 labeled rows:** Uncertainty sampling (margin sampling, entropy, or configurable strategy).

#### FEEDBACK SQL

```sql
FEEDBACK products
SET label = 'defective',
    confidence = 0.95
WHERE id = 4921
SOURCE 'quality_model_v3'
  AT PREDICTION_TIME '2025-06-15T14:05:00Z';
```

- Feedback is appended as a delta record, preserving the original value in the Merkle DAG.
- The row's gradient is boosted, influencing RGABH tiering and active learning.

#### Drift Detection
- Monitors accuracy in `_GalaxDB_predictions` over sliding windows.
- When accuracy drops below a threshold for a data segment:
  - Alerts are emitted.
  - Affected rows' gradients are boosted (increasing their hotness and surfacing them for relabeling).
  - PQ codebook and IVF quantizer retraining are triggered if embedding distribution drift is detected.

---

### 4.5 Semantic Snapshot Guarantees

v2 delivers the fully correct versioned semantic search that v1 guards against misuse:

- **`CONSISTENCY 'SEMANTIC_SNAPSHOT'`** — uses a versioned HNSW index built from the same snapshot as the row data.
- **Index snapshots:** When a version tag is created with `WITH SEMANTIC SNAPSHOT`, the current HNSW index state is also versioned and linked in the Merkle DAG.
- **Cost:** Additional storage for index snapshots. User opts in per tag.
- **Reproducibility:** A tagged snapshot with `SEMANTIC_SNAPSHOT` consistency produces **byte-identical** results on repeated export — including the similarity scores and result ordering.

```sql
-- Create a fully reproducible training snapshot
CREATE VERSION TAG 'q4_train' WITH SEMANTIC SNAPSHOT;

-- Export with guaranteed reproducibility
SELECT * FROM products
AT VERSION 'q4_train'
CONSISTENCY 'SEMANTIC_SNAPSHOT'
WHERE SEMANTIC_MATCH(description, 'camping gear', 0.5);
```

---

### 4.6 Full PostgreSQL Protocol & BI Integration

- Extended query protocol: Parse, Bind, Execute, Describe, Sync.
- `COPY` for bulk ingest and export.
- Full `information_schema` and `pg_catalog` sufficient for Tableau, Metabase, DataGrip, DBeaver.
- Server-side cursors and portals for large result sets.
- `SET` for session-level runtime parameters.
- Arrow Flight network export (gRPC) for remote high-throughput dataset transfer.

---

### 4.7 GPU-Direct & Hardware Acceleration

- **GPUDirect Storage:** Training scans bypass CPU entirely. Data flows directly from NVMe to GPU memory via DMA.
- `report_training_access` callbacks feed `training_heat` into RGABH without polluting the OLTP prefetch signal.
- Optional FPGA/SmartNIC offload for vector distance computation at the storage node level, reducing data movement for large ANN scans.

---

### 4.8 Federated Queries & Privacy

- Data atoms carry **ownership policies** (row-level security with organizational boundaries).
- Federated queries aggregate across organizations: `SELECT COUNT(*) FROM patients WHERE diagnosis = 'X'` executes across hospital databases.
- **Differential privacy** budgets are managed transparently by the query planner. Each federated query consumes budget; the planner enforces limits.
- Secure aggregation: model updates are computed without raw data leaving organizational boundaries.

---

### 4.9 Plugin Marketplace

- The `EmbeddingModel` trait (defined in v1) is the plugin interface:

```rust
trait EmbeddingModel {
  fn embed(&self, text: &str) -> Vec<f32> {
    self.embed_batch(&[text]).remove(0)
  }
  fn embed_batch(&self, texts: &[&str]) -> Vec<Vec<f32>>;
  fn model_id(&self) -> &str;
  fn dimensions(&self) -> usize;
  fn max_batch_size(&self) -> usize { 32 }
}
```

The sidecar uses `embed_batch` for throughput on bulk ingestion. The default `embed` implementation preserves compatibility for simple single-item callers.

- v2 adds a plugin registry and sandboxed execution environment.
- Third-party developers can publish models, active-learning strategies, and domain-specific data processors.
- Revenue-share model for commercial plugins.
- Users can swap models with a configuration change: `ALTER TABLE products SET EMBEDDING MODEL 'custom-finbert-v2' FOR COLUMN description`.

---

## 5. Implementation Roadmap

### v1 — 4 Months, 2–3 Rust Engineers

| Month | Deliverable | Key Milestone |
|-------|-------------|---------------|
| **1** | Core LSM storage engine | PAX blocks, WAL, checkpoint, crash recovery, Bw-Tree buffer, HotSet/ScanBuffer pool, compaction. Tested entirely via Rust API. |
| **2** | SQL layer + client integration | `sqlparser-rs` with AuroraSQL extensions, PostgreSQL simple query protocol, Python embedded mode, pg_catalog stubs, basic CRUD end-to-end. |
| **3** | Vector index + embedding sidecar | mmap'd HNSW base graph + delta buffer with merge policy, PQ codebook training (min-10k threshold), sidecar lifecycle + backlog durability, `SEMANTIC_MATCH` with union+re-rank, adaptive planner fallback (graph vs brute-force under tight filters). |
| **4** | Versioning + hardening | Merkle DAG, `AT VERSION` with consistency guardrails, version tags with pinning, ACORN-style disconnected-safe in-graph filtering hardening, Arrow IPC export API, compatibility testing (psycopg2, SQLAlchemy), chaos test harness, public demo notebook. |

### v2 — 12–18 Months, Expanded Team

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| **1** | 3–4 months | RGABH adaptive storage: multi-timescale EMA gradients, block hotness, auto-tuned tiering (NVMe → S3 → Glacier), quiescent metadata sweep, PQ codebook drift detection + auto-refresh. |
| **2** | 4–5 months | Distributed clustering: consistent hash sharding, IVF+HNSW two-level index, IVF coexisting quantizers, HLC causal consistency, Raft replication, read replicas for HTAP, 2PC for strict serializability. |
| **3** | 3–4 months | Active learning engine: `_GalaxDB_predictions` table, uncertainty scoring background job, `FEEDBACK` SQL, drift detector, three-stage cold-start bootstrap. |
| **4** | 2–3 months | Full PostgreSQL extended protocol, BI tool compatibility, GPU Direct Storage, federated queries with differential privacy, plugin marketplace launch. |

---

## 6. Appendices

### A. Audit Trail — All Issues Resolved

| # | Source | Issue | Resolution |
|---|--------|-------|------------|
| 1 | Design review | HLC doesn't give strict serializability | v1: single-node serializable. v2: causal + opt-in 2PC |
| 2 | Design review | Scatter-gather ANN returns wrong top-K | IVF coarse quantizer routes to 1–2 shards; per-shard HNSW |
| 3 | Design review | GPUDirect bypasses buffer pool | `training_heat` callback feeds RGABH without polluting prefetch |
| 4 | Design review | HNSW degrades under mutations | Mutable ANN: base graph + delta buffer with merge policy |
| 5 | Design review | PQ codebook drifts | Drift monitoring v1; auto-refresh v2 |
| 6 | Design review | OLTP sharding vs ANN sharding conflict | Consistent hash for OLTP; IVF router for ANN |
| 7–12 | External audit | See Appendix B | All 12 findings resolved in v2.1 spec |

### B. External Audit Findings — Resolution Map

| Finding | Resolution | Location |
|---------|------------|----------|
| `AT VERSION` + `SEMANTIC_MATCH` silent incorrectness | Guardrail: default rejection, explicit consistency modes | §3.3 |
| PostgreSQL compatibility overstatement | Scoped to simple query protocol | §3.6 |
| Consistency claim too broad | Split: row-serializable + semantic-eventual | §3.7 |
| mmap "zero I/O" claim | Corrected to "no explicit syscalls; page faults possible" | §3.2 |
| Candidate scoring random-read heavy | PQ codes co-located in graph payload | §3.2 |
| Flat index latency bomb | Delta buffer + merge, not unbounded flat index | §3.2 |
| Missing tombstone lifecycle | Tombstone budget + emergency merge | §3.2 |
| Static codebook drift | Drift monitoring + metric v1; auto-refresh v2 | §3.2, §4.1 |
| Version retention vs reproducibility | Pinned tags GC-exempt; 30-day default | §3.3 |
| Linux-only parent death | Cross-platform: prctl, kqueue, named pipe | §3.4, §3.9 |
| Aggressive binary size target | 64 MB core with module tiers | §3.10 |
| Timeline plausibility | Shallow protocol scope; 4 months | §5 |

### C. Performance Upgrades Incorporated

| # | Upgrade | v1 or v2 |
|---|---------|----------|
| 1 | Delta buffer + merge instead of unbounded flat index | v1 |
| 2 | Adaptive query planning (flat scan vs graph traversal) | v1 |
| 3 | Filter-aware graph traversal (ACORN-style) | v1 |
| 4 | Tombstone budget + emergency merge | v1 |
| 5 | Co-located PQ codes in graph payload | v1 |
| 6 | Dynamic ef bounded by latency SLO + recall | v1 |
| 7 | Quantization lifecycle (drift alarms, refresh) | v1/v2 |
| 8 | Semantic snapshot guardrails (consistency modes) | v1 |
| 9 | Per-query consistency modes | v1 |
| 10 | Reproducibility mode with pinned + exempt tags | v1 |

### D. Glossary

| Term | Definition |
|------|------------|
| **PAX** | Partition Attributes Across — hybrid row/column block layout |
| **PQ** | Product Quantization — lossy vector compression via subspace clustering |
| **HNSW** | Hierarchical Navigable Small World — graph-based approximate nearest neighbor index |
| **RGABH** | Row-Gradient-Aggregated Block Hotness — adaptive storage management via per-row utility signals |
| **IVF** | Inverted File — coarse quantizer for shard-level ANN routing |
| **HLC** | Hybrid Logical Clock — distributed timestamp mechanism for causal consistency |
| **LSM** | Log-Structured Merge tree — write-optimized storage structure |
| **WAL** | Write-Ahead Log — durability mechanism ensuring committed transactions survive crashes |

### E. References

- **HNSW paper:** Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" — https://arxiv.org/abs/1603.09320
- **DiskANN:** Microsoft Research, "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node"
- **TiDB/TiFlash:** HTAP architecture with Raft-shipped columnar replicas
- **PostgreSQL wire protocol:** https://www.postgresql.org/docs/current/protocol-overview.html
- **hnswlib:** https://github.com/nmslib/hnswlib
- **pgvector:** https://github.com/pgvector/pgvector
- **Weaviate:** https://docs.weaviate.io
- **Qdrant:** https://qdrant.tech/documentation/concepts/indexing/
- **FAISS:** https://github.com/facebookresearch/faiss
- **ANN-Benchmarks:** https://github.com/erikbern/ann-benchmarks

### F. Design Principles (Non-Negotiable)

1. **No silent incorrectness.** Guard, don't just document. The default path is always safe.
2. **Honest performance claims.** Every number qualified with conditions and reproducible.
3. **Graceful degradation.** Overload produces backlog, not data loss. Degraded mode is explicit.
4. **Version everything.** Data, codebooks, indices — all linked in the Merkle DAG.
5. **Ship v1.** The best specification is one that becomes a working system.

---

*This document is the authoritative reference for GalaxDB v1 and v2. Every design decision is traceable to a specific issue in the audit trail (Appendices A–B). The v1 specification (§3) is frozen. Implementation begins against §3 immediately.*