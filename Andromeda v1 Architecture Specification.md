You've caught the exact failure modes that would have surfaced during implementation. I accept all seven findings. Here's the corrected v1 specification with every issue resolved at its source.

---

## Andromeda v1 Architecture Specification — Corrected

**Version 1.1 — Buildable MVP, Hardened**  
**Target: 4 months, 2–3 Rust engineers**  
**Status: Design locked. All known correctness issues resolved.**

---

### Changelog from v1.0

| Issue | Resolution |
|-------|------------|
| HNSW stored in LSM column family | Moved to memory-mapped flat file (§4.1) |
| UPDATE/DELETE constraint too broad | Constrained to embedded column source value only (§4.3, §4.5) |
| Dropped embedding requests = data loss | Replaced with persistent backlog table (§6.2) |
| PQ codebook for small tables | Minimum vector threshold for PQ (§4.2) |
| Flat index rebuild threshold | Relative threshold: max(10k, 1% of total) (§4.3) |
| Arrow Flight undefined | Added §10 for training data path |
| Timeline unrealistic | Adjusted to 4 months (§13) |

---

## 1. Introduction

Andromeda v1 is a **single‑node, embedded, hybrid database** that unifies OLTP row storage, columnar analytics, and vector similarity search into one engine. It is the first step toward the full AI‑native vision — a system that replaces PostgreSQL, a vector extension, and a feature store with a single ~50 MB binary.

This document describes exactly what will be built, what is explicitly deferred, and every design decision that carries a correctness or performance consequence.

---

## 2. Design Principles

1. **Single‑node first.** No clustering, no Raft, no distributed transactions. v1 is an embedded library like SQLite, with an optional standalone server.
2. **Unified storage.** OLTP rows, analytical columns, and dense embeddings live together in a single LSM tree of PAX blocks. The HNSW graph is a separate memory‑mapped file for performance.
3. **Honest semantics.** Every `AT VERSION` query, every `SEMANTIC_MATCH`, every insert behaviour is explicitly defined, including limitations.
4. **AI‑ready, not AI‑only.** The engine can run any standard SQL workload; vector search and versioning are extensions, not the whole product.
5. **Simple integration.** PostgreSQL wire protocol (simple query) so that `psycopg2`, SQLAlchemy, and other tools work out of the box.

---

## 3. Storage Engine

### 3.1 LSM Tree with PAX Blocks

Data is organized into **PAX (Partition Attributes Across) blocks** — each block holds ~1,000 rows, with columns stored contiguously inside.

- **Write path:** Incoming rows accumulate in a memory buffer (lock‑free Bw‑Tree). When the buffer reaches 64 MB, it is sealed and flushed as an immutable PAX block.
- **Sorted runs:** Flushed blocks are periodically merged by a background compactor into larger sorted runs (LSM levels). Compaction always produces new blocks; old blocks are garbage‑collected after no active snapshot references them.
- **Read path:**
  - **Point query** → sparse index (primary key → block + offset) → single random read.
  - **Column scan** → sequential reads of the needed column chunks from blocks in sorted order.
- **Buffer pool:** Split into **HotSet** (70% of RAM) and **ScanBuffer** (30%).
  - HotSet holds blocks recently hit by OLTP point reads (LRU eviction).
  - ScanBuffer holds blocks prefetched for OLAP scans, using a clock‑sweep policy, and is never allowed to evict a block that belongs to a HotSet‑resident key.
  - In v1, there is **no value‑gradient (RGABH)** — admission and eviction are strictly access‑pattern‑based.

### 3.2 PAX Block Format (Physical Layout)

| Section | Contents |
|---------|----------|
| Header | Block ID, row count, column descriptors, min/max per column, compression info |
| Column chunks | Fixed‑width columns stored contiguously; variable‑width columns (TEXT, BLOB) stored with length prefixes |
| Embedding column | Dense float32 arrays, optionally compressed with Product Quantization (see §4.2) |
| Row offset table | Byte offset of each row within the block |

All block data is checksummed. The format is self‑describing and versioned.

---

## 4. Vector Index

### 4.1 HNSW Graph — Memory‑Mapped File

The HNSW graph is stored as a **memory‑mapped flat file** on disk, completely separate from the LSM tree.

- **Rationale:** HNSW graph traversal requires 50–200 random node accesses per query. Each access in an LSM tree would trigger bloom‑filter checks and multi‑level lookups — catastrophically slow. Production systems (hnswlib, FAISS, Milvus, Weaviate) all use direct memory‑mapped graph storage.
- **Implementation:** The graph is a single `.hnsw` file containing contiguous arrays of adjacency lists. It is `mmap`'d into the process address space at startup. Traversal uses native pointer dereferences with zero I/O overhead.
- **PAX blocks** store the PQ‑compressed embedding vectors (see §4.2). During search, the HNSW traversal uses the mmap'd graph to identify candidate nodes; the PQ codes for those candidates are read from the LSM tree for approximate scoring. The top‑K candidates are then re‑ranked against raw vectors stored in the PAX blocks for final exact scores.

### 4.2 Product Quantization (PQ)

To reduce memory and I/O for embedding storage, embeddings may be compressed using **Product Quantization**:

- A codebook (k‑means centroids) is trained on the vectors in the table.
- Each vector is stored as a short code (e.g., 16 bytes for a 768‑dim vector).
- The HNSW graph edges reference PQ‑compressed codes during approximate search; raw floats are only read for final re‑ranking.

**Codebook training policy:**
- If total vectors < 10,000: **Skip PQ entirely.** Store raw float32 vectors. The training set is too small to produce a useful codebook.
- If total vectors is between 10,000 and 1,000,000: Train PQ on all available vectors.
- If total vectors > 1,000,000: Randomly sample 1,000,000 vectors for training.
- In v1, the codebook is **static** after initial training. No drift detection, no codebook refresh.

### 4.3 Insert Behaviour and Index Freshness

v1 treats the HNSW index as **immutable after initial build** except for new rows. The write path is:

1. `INSERT` with an embedded column → the text is sent to the embedding sidecar.
2. The row is written to the PAX block with the embedding columns marked as stale (`_embedding_stale = true`).
3. Once the sidecar returns the embedding, the background worker:
   - Updates the row's embedding in the LSM (a new PAX block pointing to the same row data with the embedding filled).
   - Adds the vector to a **pending flat index** (in‑memory exact k‑NN).
4. `SEMANTIC_MATCH` queries search **both** the HNSW index (old rows) and the flat index (new rows), then union and re‑rank the results.
5. The flat index triggers a **background HNSW rebuild** when its size exceeds `max(10,000, total_indexed_vectors × 0.01)`. This relative threshold prevents pathological rebuild frequency on large tables (10M indexed rows → rebuild at 100,000 new rows, not 10,000).
6. The rebuild uses **double‑buffering** — the old index stays alive for in‑flight queries; new queries transparently switch to the new index when ready. Only one rebuild can run at a time; subsequent triggers are queued and deduplicated.

### 4.4 `AT VERSION` and Vector Search

The Merkle DAG versioning system (see §5) provides historical snapshots of row data. However, the HNSW index is **not versioned** in v1. A query like:

```sql
SELECT * FROM products AT VERSION 'last_month'
WHERE SEMANTIC_MATCH(description, 'camping gear', 0.7);
```

will:
- Retrieve row data from the historical snapshot (correct semantics).
- Use the **current HNSW index** for similarity search and ranking.

This means:
- Rows that did not exist at `last_month` may appear in results if they are similar enough.
- Rows that existed at `last_month` but have since been deleted may not appear.

**This is documented behaviour, not a bug.** Users who need fully versioned vector search must wait for v2 (index snapshots).

### 4.5 UPDATE and DELETE on Embedded Columns

The only DML restriction in v1 is on the embedded column's **source value**:

| Operation | Supported? |
|-----------|-----------|
| `UPDATE products SET price = 99.99 WHERE id = 42` | ✅ Works normally |
| `UPDATE products SET status = 'archived' WHERE id = 42` | ✅ Works normally |
| `DELETE FROM products WHERE id = 42` | ✅ Writes a tombstone, row excluded from queries |
| `UPDATE products SET description = 'new text' WHERE id = 42` | ❌ Blocked in v1. Re-embedding requires index update not yet supported. Workaround: `DELETE` + `INSERT`. |

This constraint applies specifically and only to the source column of an embedding. All other column DML is fully functional.

---

## 5. Versioning (Merkle DAG)

Andromeda provides Git‑like time‑travel for row data:

- Every write creates a new PAX block with a commit timestamp.
- A Merkle tree over block hashes forms the system state; each commit produces a **version root**.
- `AT VERSION` queries filter blocks where `block.commit_time ≤ target_version`.
- Named version tags can be created: `CREATE VERSION TAG 'q2_snapshot'`.
- Version data is retained until garbage‑collected; compaction automatically prunes versions older than a configurable retention window (default 7 days).

Versioning applies to **row data only**. Embeddings, the HNSW index, and system catalogs are not versioned in v1.

---

## 6. Embedding Inference Sidecar

Embedding computation is offloaded to a separate process to keep the database lightweight and ML‑framework‑independent.

### 6.1 Architecture

- The sidecar is a **standalone binary** (Python or Rust) that loads a sentence‑transformer model via ONNX Runtime.
- Communication with the database is over a **Unix domain socket** (gRPC or simple length‑prefixed messages).

### 6.2 Lifecycle

- **Starting:** When the `andromeda` process starts (either CLI or embedded via Python), it spawns the sidecar as a child process.
- **Process ownership:** The sidecar monitors its parent's PID; if the parent dies, the sidecar terminates (`prctl(PR_SET_PDEATHSIG)` on Linux).
- **Health:** The sidecar sends a heartbeat (ping) every 5 seconds. If the database misses 3 consecutive pings, it considers the sidecar crashed.
- **Crash recovery:** On crash, embedding work is marked as failed. The database continues serving queries normally (embeddings remain stale). The sidecar is restarted with exponential backoff (1s, 2s, 4s, up to 60s). During recovery, new writes still proceed but their embeddings stay stale until the sidecar returns.
- **Back‑pressure:** The embedding request queue has a hard limit of 10,000 in‑flight items. When the queue is full, new embedding requests are **not dropped** — instead, the row ID is written to a persistent **stale‑row backlog table** (`_andromeda_embedding_backlog`). A low‑priority background scanner periodically drains this table, re‑submitting embedding requests when the sidecar has capacity. The row remains stale until processed, but it is never permanently lost. The metric `_embedding_queue_full` increments to track how often back‑pressure is triggered, representing delayed indexing, not data loss.

---

## 7. SQL Language (AuroraSQL)

Andromeda v1 supports a **superset of PostgreSQL‑compatible SQL** with the following custom extensions.

### 7.1 DDL Extensions

```sql
CREATE TABLE products (
    id          BIGINT PRIMARY KEY,
    title       TEXT,
    description TEXT EMBEDDING MODEL 'all-MiniLM-L6-v2' DIM 384,
    price       DECIMAL,
    created_at  TIMESTAMP
);
```

- `EMBEDDING MODEL` specifies the model name known to the sidecar. The column will be automatically populated on INSERT.
- `DIM` is optional; if omitted, the sidecar is queried for the model's output dimension.
- A table may have multiple embedding columns.

### 7.2 `SEMANTIC_MATCH` Predicate

```sql
SEMANTIC_MATCH(column, 'query text', threshold)
```

- `threshold` is a minimum similarity score (cosine similarity, float 0–1).
- Can be combined with standard `WHERE` clauses.
- Internally: computes the query embedding via sidecar, searches HNSW + pending flat index, filters by threshold, returns results with a synthetic `similarity` column.

### 7.3 `AT VERSION` Clause

```sql
SELECT * FROM table AT VERSION timestamp_or_tag ...
```

- Returns rows as they existed at that point in time.
- Works with any `SELECT`, including joins and aggregations (on versioned tables).
- Limitations explicitly documented: HNSW index is not versioned; system tables are not versioned.

### 7.4 Active Learning and FEEDBACK

**Not present in v1.** Deferred to v2.

---

## 8. PostgreSQL Wire Compatibility

Andromeda v1 implements **Tier 1** of the PostgreSQL wire protocol:

- Simple query protocol (`Q` message).
- Basic DDL (`CREATE TABLE`, `DROP TABLE`), DML (`INSERT`, `SELECT`, `DELETE` with normal columns), and `SET` for runtime parameters.
- Minimal `pg_catalog` stubs: `pg_class`, `pg_attribute`, `pg_type` are present with enough columns for `psycopg2` and SQLAlchemy to connect and introspect tables.

**Not supported in v1:** extended query protocol (prepared statements, parameterized queries via `Parse`/`Bind`), `COPY`, full `information_schema`. These will be added in v2.

---

## 9. Consistency Model

v1 is a **single‑node, single‑writer** database. It provides **strict serializable isolation** for all reads and writes by construction — there is no concurrent transaction execution. Readers never block writers, and vice versa, because the LSM tree provides snapshot isolation via versioned blocks.

This guarantee applies to v1 only. The distributed (clustered) mode planned for v2 will downgrade to causal consistency with bounded staleness; this will be explicitly documented when the distributed feature ships.

---

## 10. Training Data Path

A `SELECT` query can be executed against a named version tag and materialized as an **Arrow RecordBatch** stream for training consumption:

```python
import andromeda

db = andromeda.Database("mydata")
reader = db.execute_arrow("""
    SELECT title, description, embedding
    FROM products AT VERSION 'q2_train'
""")
for batch in reader:
    # batch is a pyarrow.RecordBatch, zero‑copy into PyTorch
    ...
```

- The Arrow Flight server is embedded in the Andromeda process and exposes versioned snapshots via a simple `do_get` endpoint.
- **No GPU Direct Storage in v1.** Training data flows through CPU memory. GPU‑Direct (direct NVMe → GPU DMA) is deferred to v2.

---

## 11. Binary Size

- Core storage engine (Rust, statically linked): **under 50 MB**.
- Full installation including the ML inference sidecar (ONNX Runtime + default model): **under 500 MB**.

The sidecar and model are downloaded on first use or bundled in the official distribution package.

---

## 12. Deployment Modes

| Mode | Description |
|------|-------------|
| **Embedded** | `import andromeda` in Python; library starts in‑process. No replication, no background cluster tasks. The sidecar is spawned as a child process. |
| **Standalone server** | `andromeda --server` starts a PostgreSQL‑wire‑compatible server on localhost:5432. Same binary, same capabilities. |
| **Clustered** | Not in v1. |

---

## 13. What Is Intentionally Excluded from v1

These features are designed but deferred to v2 or later:

- Distributed clustering (Raft, scatter‑gather ANN, Go control plane).
- RGABH (Row‑Gradient‑Aggregated Block Hotness) and full adaptive tiering.
- Active learning (`ORDER BY ACTIVE_LEARNING`), `FEEDBACK` SQL, drift detection.
- `UPDATE` of embedded column source values.
- Versioned vector indexes.
- Full PostgreSQL extended query protocol.
- GPU‑Direct storage access.
- Plugin marketplace.

---

## 14. Implementation Sequencing (4‑Month Plan)

### Month 1 — Core Storage Engine
- LSM tree with PAX blocks: write path (Bw‑Tree buffer, flush), basic compaction, primary key sparse index.
- Point reads and full column scans. No SQL yet. Tested entirely via a Rust API.
- Buffer pool: HotSet/ScanBuffer split, LRU eviction.

### Month 2 — SQL Layer + Client Integration
- `sqlparser-rs` integration with AuroraSQL extensions (`SEMANTIC_MATCH`, `AT VERSION`, `EMBEDDING MODEL` in DDL).
- PostgreSQL simple query protocol (`Q` message).
- Python embedded mode: `import andromeda`, basic CRUD, `pg_catalog` stubs.
- End‑to‑end working: create tables, insert rows, query with standard SQL.

### Month 3 — Vector Index + Embedding Sidecar
- mmap‑based HNSW index (separate `.hnsw` file).
- PQ codebook training with the minimum‑vector threshold logic.
- Embedding sidecar: Unix socket communication, lifecycle management, persistent backlog table.
- Insert path: flat index, staleness flag, `SEMANTIC_MATCH` integration with union+re‑rank.

### Month 4 — Versioning + Hardening
- Merkle DAG versioning: commit timestamps, version roots, named tags, `AT VERSION` query filtering.
- HNSW double‑buffer rebuild trigger with relative threshold logic.
- Crash recovery, sidecar restart with backoff, back‑pressure backlog drain.
- Arrow Flight training data path.
- Compatibility testing: `psycopg2`, SQLAlchemy.
- Public demo notebook: "5‑minute hybrid search + time travel."

---

## 15. Appendix: v2 Design Principles (Preview)

The decisions made for v2 are recorded separately but include:

- **RGABH‑driven storage:** block hotness from multi‑timescale EMA, driving adaptive tiering (NVMe/S3/Glacier), prefetch, and buffer pool admission.
- **Distributed clustering:** consistent hash sharding for OLTP, IVF+HNSW two‑level index for ANN, causal consistency via HLC.
- **Active learning engine:** background uncertainty scoring, `FEEDBACK` ingestion, drift‑triggered PQ codebook refresh.
- **Full wire protocol:** extended query, `COPY`, `pg_catalog` maturity.
- **Go control plane:** orchestration, Kubernetes operator, cloud API.

These principles are locked but not detailed until v1 ships.

---

*End of v1 Architecture Specification.*