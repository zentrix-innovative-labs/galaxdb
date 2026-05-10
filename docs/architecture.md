# Architecture

GalaxDB is structured as a workspace of Rust crates, each responsible for a specific subsystem.

---

## Crate Map

```
galaxdb/
├── crates/
│   ├── galaxdb-common/      # Shared types, errors, config
│   ├── galaxdb-storage/     # LSM storage engine (core)
│   ├── galaxdb-io/          # I/O abstraction (io_uring / tokio)
│   ├── galaxdb-crypto/      # TDE encryption (AEGIS-256, AES-256-GCM)
│   ├── galaxdb-vector/      # HNSW index, delta buffer, quantizers
│   ├── galaxdb-sql/         # SQL parser, planner, executor
│   ├── galaxdb-wire/        # PostgreSQL wire protocol
│   ├── galaxdb-sidecar/     # Embedding sidecar (Candle inference)
│   ├── galaxdb-embedded/    # Embedded database API
│   ├── galaxdb-versioning/  # Merkle DAG, version tags
│   ├── galaxdb-observe/     # Metrics, health, tracing
│   └── galaxdb-server/      # Standalone server binary
├── galaxdb-python/          # Python bindings (PyO3)
├── benchmarks/              # Macro-benchmark suite
└── tests/chaos/             # Chaos/fault-injection tests
```

---

## Storage Engine (galaxdb-storage)

### Write Path

```
INSERT → SQL Parser → Executor → WAL (group commit) → Memtable → ART Index
                                                    ↓
                                              Seal at 64MB
                                                    ↓
                                         Flush to SST (PAX blocks)
                                                    ↓
                                         Compaction (Lazy Leveling)
```

### Read Path

```
SELECT → SQL Parser → Executor → ART Lookup → Memtable (if present)
                                            → Buffer Pool → SST Block Read
                                                         → Bloom Filter Check
                                                         → Zone Map Pruning
```

### Key Components

| Component | Description |
|-----------|-------------|
| **WAL** | Write-ahead log with LZ4 compression, XXH3-64 checksums, group commit |
| **Memtable** | 16-shard crossbeam-skiplist with MVCC version chains |
| **PAX Blocks** | Columnar storage format with per-column compression |
| **ART Index** | Adaptive Radix Tree for O(k) primary key lookups |
| **Bloom Filters** | Per-SST with Monkey-optimal FPR allocation |
| **Buffer Pool** | NUMA-aware with HotSet (LRU) + ScanBuffer (clock-sweep) |
| **Compaction** | Lazy Leveling (L0-L3 tiered, L4 leveled) with MVCC GC |
| **Blob Log** | KV separation for values > 1KB |
| **TDE** | Transparent data encryption (AEGIS-256 for blocks, AES-256-GCM for WAL) |
| **RateLimiter** | Auto-tuned token bucket calibrated to NVMe bandwidth |
| **WriteController** | Proportional slowdown between soft/hard compaction limits |

### I/O Subsystem

On Linux 5.10+, GalaxDB uses io_uring with two separate submission queues:
- **HP queue** — user-facing reads (point lookups, HNSW search)
- **BK queue** — background writes (flush, compaction)

On macOS/Windows, falls back to tokio async I/O.

---

## Vector Search (galaxdb-vector)

### HNSW Index

Hierarchical Navigable Small World graph for approximate nearest neighbor search.

| Parameter | Default | Description |
|-----------|---------|-------------|
| M | 16 | Max edges per node (32 at layer 0) |
| ef_construction | 200 | Beam width during index build |
| ef_search | 100-200 | Beam width during query (tunable per query) |

**Performance (SIFT1M, 1M × 128-dim):**
- Build: 14,728 vec/sec (beats hnswlib by 8%)
- Recall@10: 0.99 at ef=200
- Search: 281µs at ef=100

### Delta Buffer

In-memory flat index for vectors inserted after the last HNSW build. Searched with brute-force and unioned with HNSW results.

Merge trigger: `max(10,000, total_indexed × 0.01)` vectors in delta buffer.

### Quantization

| Method | Compression | Use Case |
|--------|-------------|----------|
| SQ8 | 4× | Training export, memory reduction |
| FP16 | 2× | ARM64 inference |
| RaBitQ | 32× | Extreme compression (opt-in) |

---

## Embedding Sidecar (galaxdb-sidecar)

Standalone process for text → embedding conversion using sentence-transformer models.

### Architecture

```
Engine ←→ Unix Socket ←→ Sidecar (Candle inference)
                              ↓
                    HuggingFace Model Hub
                    (downloads on first use)
```

### Lifecycle

1. Engine spawns sidecar as child process
2. Sidecar downloads model (cached in `~/.cache/huggingface/`)
3. Sidecar listens on Unix socket
4. Engine sends embed requests, receives embeddings
5. If sidecar crashes: 3 missed heartbeats → degraded mode → exponential backoff restart
6. Overflow requests go to backlog table (drained when sidecar recovers)

### Protocol

Length-prefixed JSON over Unix socket:

```json
// Request
{"type": "EmbedRequest", "row_id": 1, "text": "hello world", "column": "content"}

// Response
{"type": "EmbedResponse", "row_id": 1, "embedding": [0.01, -0.04, ...], "model_version": "all-MiniLM-L6-v2"}
```

### Model Version Tracking

Each row stores `_embedding_model_version`. When the model changes:
1. Existing embeddings are marked `_embedding_stale = true`
2. Background re-embedding is triggered
3. `SHOW EMBEDDING HEALTH` reports stale count and version distribution

---

## SQL Layer (galaxdb-sql)

### Parser

Extends sqlparser-rs with AuroraSQL extensions:
- `EMBEDDING MODEL 'name' DIM n` in CREATE TABLE
- `SEMANTIC_MATCH(col, 'query', threshold)` predicate
- `AT VERSION timestamp_or_tag`
- `CREATE VERSION TAG 'name' [FOR TRAINING]`
- `BULK INSERT`, `SHOW EMBEDDING HEALTH`, `BACKUP TO`, `RESTORE FROM`

### Query Planner

Adaptive strategy selection based on table statistics:
- Point lookup (primary key) → ART index
- Full scan → zone-map pruning + Bloom filter
- Semantic search → HNSW + delta buffer
- Hybrid (filter + semantic) → cardinality-based routing

### Executor

Trait-based architecture:
- `VectorSearchBackend` — abstracts HNSW + sidecar for testability
- Catalog tracks table metadata, embedding columns, statistics

---

## Wire Protocol (galaxdb-wire)

PostgreSQL v3 simple query protocol:
- StartupMessage → AuthenticationOk → ReadyForQuery
- Query → RowDescription → DataRow* → CommandComplete → ReadyForQuery
- ErrorResponse with SQLSTATE codes
- TLS 1.3 via rustls
- Max 1000 concurrent connections

---

## Encryption (galaxdb-crypto)

### At-Rest Encryption (TDE)

| Layer | Algorithm | Throughput |
|-------|-----------|------------|
| PAX blocks (SST files) | AEGIS-256 | 6.63 GB/s decrypt |
| WAL records | AES-256-GCM | 1.43 GB/s decrypt |

### Key Management

Pluggable via `KeyProvider` trait:
- `LocalKeyProvider` — key from file (development)
- `EnvKeyProvider` — key from environment variable
- `ExternalCommandKeyProvider` — delegate to any KMS CLI (AWS, GCP, Azure, Vault, custom)
- `HashicorpVaultKeyProvider` — Vault Transit engine (feature = "vault")

---

## Crash Safety

GalaxDB guarantees:
- No committed data loss on crash (WAL + fsync)
- Recovery in < 30 seconds (WAL replay from last checkpoint)
- Atomic SST file creation (write to temp, fsync, rename)
- Reserve file for disk-full handling (32MB pre-allocated)

Verified by 6 chaos tests:
1. Kill mid-flush → WAL replay recovers all data
2. Kill mid-compaction → old SSTs intact
3. Corrupt WAL → replay stops at corruption, valid data preserved
4. Disk full → clean checkpoint, reads continue
5. 100 concurrent writers → zero duplicates, zero missing
6. OLAP scan during OLTP → zero HotSet evictions
