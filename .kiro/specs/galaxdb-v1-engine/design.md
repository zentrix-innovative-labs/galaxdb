# Design Document — GalaxDB v1 Engine

## Overview

GalaxDB v1 is a single-node, AI-native database engine written in Rust. It unifies transactional row storage (LSM+PAX), approximate nearest neighbor vector search (mutable HNSW), and versioned training data export (Merkle DAG + Lance format) into a single binary. The engine exposes a PostgreSQL-compatible wire protocol extended with AuroraSQL syntax for embedding columns, semantic search, versioning, and training export.

### Design Principles (from Final Version 4.2)

1. **Unified Data Atom** — relational fields, embeddings, binaries, and lineage in one row.
2. **Honest Semantics** — limitations documented; silent incorrectness never allowed.
3. **AI-First** — embeddings, versioned snapshots, and training-aware optimizations are first-class.
4. **Falsifiable Claims** — every performance number stated with measurable conditions.

### Platform Support

| Mode | Platforms | I/O Backend | Production? |
|------|-----------|-------------|-------------|
| Embedded | Linux 5.10+, macOS, Windows | io_uring (Linux), tokio (macOS/Windows) | Linux only |
| Standalone server | Linux, macOS | io_uring (Linux), tokio (macOS) | Linux only |

Performance guarantees (P99 latency, NVMe bandwidth saturation) apply only to Linux production deployments with io_uring on NVMe storage.

---

## Architecture

### System Architecture Diagram

```
┌──────────────────────────────────────────────────────┐
│                  AuroraSQL Language                   │
│       (PostgreSQL wire protocol + AI extensions)      │
├──────────────────────────────────────────────────────┤
│           Query Optimizer, Planner & Executor         │
├──────────────┬───────────────┬───────────────────────┤
│  LSM + PAX   │  Mutable ANN  │ Embedding Sidecar     │
│  Store       │ (mmap + delta │ (Unix Socket,          │
│              │   + SQ8)      │  persistent backlog)   │
├──────────────┴───────────────┴───────────────────────┤
│       io_uring I/O Scheduler (HP/BK queues)           │
│         [Linux only; tokio on macOS/Windows]          │
├──────────────────────────────────────────────────────┤
│  Storage (NVMe, blob store)                           │
└──────────────────────────────────────────────────────┘
```

### Rust Workspace Structure

```
galaxdb/
├── Cargo.toml                    # Workspace root
├── crates/
│   ├── galaxdb-storage/          # LSM, PAX, WAL, Memtable, ART, Bloom, Buffer Pool, Blob Log, Compactor
│   ├── galaxdb-vector/           # HNSW, Delta Buffer, Quantizer (SQ8/FP16/RaBitQ)
│   ├── galaxdb-sql/              # SQL Parser (sqlparser-rs + AuroraSQL), Query Planner, Query Executor
│   ├── galaxdb-wire/             # PostgreSQL simple query wire protocol, pg_catalog stubs
│   ├── galaxdb-versioning/       # Merkle DAG, Version Tags, Lance Exporter, MinHash Dedup
│   ├── galaxdb-sidecar/          # Embedding Sidecar binary (ort crate, Unix socket, backlog)
│   ├── galaxdb-crypto/           # TDE (AES-256-GCM), TLS 1.3, AWS KMS integration
│   ├── galaxdb-io/               # I/O abstraction: io_uring (Linux) / tokio (macOS/Windows)
│   ├── galaxdb-observe/          # HTTP /health + /metrics, Prometheus, OTel tracing, JSON logging
│   ├── galaxdb-server/           # Standalone server binary, connection management
│   ├── galaxdb-embedded/         # Embedded library API (Python FFI via PyO3)
│   └── galaxdb-common/           # Shared types, config, error types
├── galaxdb-python/               # Python client package (PyO3 + Lance + PyTorch integration)
└── tests/
    ├── integration/              # End-to-end integration tests
    └── chaos/                    # Chaos test harness (sidecar kill, WAL corruption, disk-full)
```

### Key Crate Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `crossbeam-skiplist` | latest | Lock-free concurrent skiplist for Memtable |
| `sqlparser` | latest | SQL parsing foundation (PostgreSQL dialect) |
| `tokio` | 1.x | Async runtime, connection management |
| `io-uring` | latest | Linux io_uring bindings (conditional) |
| `ort` | 2.x | ONNX Runtime bindings for embedding sidecar |
| `lance` | 1.x | Lance columnar format for training export |
| `xxhash-rust` | latest | XXH3-64 checksums for WAL and PAX blocks |
| `lz4_flex` | latest | LZ4 compression for WAL payloads |
| `zstd` | latest | Zstandard compression for PAX variable-width columns |
| `aes-gcm` | latest | AES-256-GCM encryption for TDE |
| `aws-sdk-kms` | latest | AWS KMS key management |
| `rustls` | latest | TLS 1.3 for wire protocol |
| `prometheus` | latest | Prometheus metrics exposition |
| `tracing` + `tracing-subscriber` | latest | Structured JSON logging |
| `opentelemetry` | latest | OTel W3C trace context propagation |
| `pyo3` | latest | Python FFI for embedded mode and client |
| `memmap2` | latest | Memory-mapped files for HNSW base graph |
| `half` | latest | FP16 half-precision for ARM64 quantization |


---

## Component Designs

### 1. Storage Engine — `galaxdb-storage` (Reqs 1–8, 31, 36)

#### 1.1 Memtable (Req 1)

**Data Structure:** `crossbeam-skiplist::SkipMap<Vec<u8>, VersionChain>` wrapped with 16-shard `Mutex` for per-key MVCC concurrency.

```rust
pub struct Memtable {
    shards: [Mutex<SkipMap<Vec<u8>, VersionedValue>>; 16],
    size: AtomicU64,          // current byte size
    sealed: AtomicBool,       // true when size >= 64 MB
}

pub struct VersionedValue {
    timestamp: u64,           // MVCC commit timestamp
    value: Option<Vec<u8>>,   // None = tombstone
    prev: Option<Box<VersionedValue>>, // version chain
}
```

**Shard selection:** `shard_index = xxh3_64(primary_key) % 16`. This distributes keys uniformly across shards.

**Seal threshold:** 64 MB. When `size` crosses this, the memtable is atomically swapped to a new empty one. The sealed memtable is enqueued for flush.

**Back-pressure:** A `Semaphore` with capacity 256 MB tracks total sealed-but-unflushed bytes. Writers `acquire()` before writing; if total exceeds 256 MB, writers block until flush releases permits.

**Epoch safety rule:** All reads from the skiplist copy the value bytes out of the `Entry` handle immediately. The handle is dropped before any `.await`. A `#[deny(clippy::await_holding_lock)]`-style custom lint enforces this.

#### 1.2 PAX Block Format (Req 2)

```
┌─────────────────────────────────────────────┐
│ PAX Block Header (fixed size)               │
│  magic: u32 = 0x47414C41                    │
│  format_version: u8                         │
│  block_id: u64                              │
│  row_count: u32                             │
│  commit_timestamp: u64                      │
│  column_count: u16                          │
│  column_descriptors: [ColumnDesc; N]        │
│    - col_type, offset, compressed_len       │
│    - zone_map_min, zone_map_max             │
│  checksum: u64 (XXH3-64 over entire block)  │
├─────────────────────────────────────────────┤
│ Column Chunk 0 (fixed-width: FastPFOR)      │
│ Column Chunk 1 (variable-width: Zstd L3)    │
│ Column Chunk 2 (embedding: raw quantized)   │
│ ...                                         │
├─────────────────────────────────────────────┤
│ Row Offset Table                            │
│  [u32; row_count] byte offsets              │
└─────────────────────────────────────────────┘
```

**Compression strategy per column type:**
- Fixed-width integers: delta encoding + bit-packing (FastPFOR algorithm)
- Variable-width (TEXT, BLOB, JSON): Zstandard level 3
- Embedding columns: no additional compression (quantization handles it)
- The codec ID is stored per column descriptor: `0=none, 1=fastpfor, 2=zstd`

**Integrity:** On every read, the engine recomputes XXH3-64 over the block bytes and compares against the stored checksum. Mismatch → block rejected, error returned.

#### 1.3 ART Primary Key Index (Req 3)

```rust
pub struct ArtIndex {
    tree: AdaptiveRadixTree<Vec<u8>, RowLocation>,
}

pub enum RowLocation {
    Memtable { shard: u8, key: Vec<u8> },
    SST { sst_id: u64, block_offset: u64, row_offset: u32 },
}
```

The ART is an in-memory index rebuilt on crash recovery by scanning all SST block headers and replaying the WAL. It maps every primary key to its current location (memtable or SST). Point lookups go directly through the ART without scanning SST files.

**Implementation:** Custom Rust ART implementation following Leis et al. (ICDE 2013) with Node4, Node16, Node48, Node256 node types and path compression.

#### 1.4 Bloom Filters with Monkey Allocation (Req 4)

Each SST file carries a Bloom filter. The false-positive rate (FPR) per level is allocated using the Monkey-optimal strategy:

```
FPR(level_i) = total_fpr_budget * (size_ratio^(L-i)) / sum(size_ratio^(L-j) for j in 0..L)
```

Where `L` is the number of levels and `size_ratio` is the LSM size ratio (default 10). This concentrates Bloom filter memory on the larger, colder levels where false positives are most expensive.

**Memory budget:** Configurable, default 10 bits per key across all levels.

#### 1.5 NUMA-Aware Buffer Pool (Req 5)

```rust
pub struct BufferPool {
    hot_set: NumaPartitioned<LruCache<BlockId, CachedBlock>>,   // 70% RAM
    scan_buffer: NumaPartitioned<ClockSweep<BlockId, CachedBlock>>, // 30% RAM
}

pub struct NumaPartitioned<T> {
    per_node: Vec<T>,  // one instance per NUMA node
}
```

**Allocation:** Worker threads detect their NUMA node via `libnuma` (Linux) or fall back to a single partition on macOS/Windows. Buffer frames are allocated from the local NUMA node's pool.

**Routing:** Point lookups place blocks in HotSet. Sequential scans place blocks in ScanBuffer. ScanBuffer never evicts a HotSet-resident block.

#### 1.6 Lazy Leveling Compaction with MVCC GC (Req 6)

**LSM structure:**
- L0: flushed memtables (tiered, up to 4 files before compaction trigger)
- L1–L3: tiered compaction (multiple sorted runs per level)
- L4 (bottom): leveled compaction (single sorted run)

**MVCC GC during compaction:** For each key encountered during merge, the compactor checks:
1. Is this version needed by the oldest active snapshot? → keep
2. Is this version referenced by any pinned Version_Tag? → keep
3. Otherwise → discard

**SST size:** 64 MB initially (Month 1), configurable down to 8 MB in Month 4 hardening (Req 36).

**Month 4 vLSM improvements (Req 36):**
- L0 switches from tiered to leveled compaction
- SILK-style flush pre-emption: when memtable back-pressure is high, flush I/O gets priority over compaction I/O via the RateLimiter

#### 1.7 Write-Ahead Log (Req 7)

**Record format:**
```
┌──────┬──────────┬────────┬──────────────┬─────────────────┐
│ type │ seq_no   │ length │ xxh3_checksum│ lz4_payload     │
│ u8   │ u64      │ u32    │ u64          │ [u8; length]    │
└──────┴──────────┴────────┴──────────────┴─────────────────┘
```

**Record types:**
- `0x01` ROW_PUT — row insert/update
- `0x02` ROW_DELETE — tombstone
- `0x03` DELTA_INSERT — vector delta buffer insert
- `0x04` DELTA_TOMBSTONE — vector delete
- `0x05` CHECKPOINT — checkpoint marker
- `0x06` BLOB_REF — blob log reference for KV-separated values

**Group commit:** A background task collects WAL writes into batches over a configurable window (default 10 ms), then issues a single `fsync`. Connections with `DURABILITY STRICT` bypass the batch and fsync immediately. `DURABILITY RELAXED` uses the batch window.

**Checkpoint:** Triggered when WAL exceeds 512 MB or 60 seconds since last checkpoint. Flushes the active memtable, writes a CHECKPOINT record, and truncates the WAL up to that point.

**Recovery:** On startup, replay from the last CHECKPOINT record. For each record, verify XXH3-64 checksum. Skip corrupt records. Stop at first checksum failure. Recovery target: < 30 seconds.

#### 1.8 KV Separation — Blob Log (Req 8)

```rust
pub struct BlobLog {
    writers: Vec<BlobWriter>,  // multi-queue parallel writers (4 queues default)
    index: HashMap<[u8; 32], BlobRef>,  // content-hash → (file_id, offset, length)
}

pub struct BlobRef {
    file_id: u64,
    offset: u64,
    length: u32,
}
```

**Write path:** During WAL entry construction, if `value.len() > 1024`, the value is written to the blob log immediately. The memtable stores only the 32-byte XXH3-128 content hash + BlobRef. This happens at WAL time, not flush time (BVLSM pattern).

**GC:** A background task scans blob files. When discardable space (values whose keys have been compacted away) exceeds 50% of a file, the live values are copied to a new file and the old file is deleted.

#### 1.9 Disk Full Handling (Req 31)

At startup, the engine pre-allocates a 32 MB reserve file (`_galaxdb_reserve`). On disk-full:
1. Delete the reserve file to free 32 MB
2. Perform a clean checkpoint (flush memtable, write checkpoint record)
3. Block all writes
4. Emit `_disk_full` metric and log error
5. No data corruption — all committed data is safe on disk


---

### 2. Transparent Data Encryption — `galaxdb-crypto` (Req 9)

#### 2.1 Block-Level Encryption

Every PAX block and WAL record is encrypted with AES-256-GCM before writing to disk. The encryption sits between the storage engine and the I/O scheduler — the I/O layer only sees ciphertext.

```rust
pub struct TdeModule {
    kms_client: aws_sdk_kms::Client,
    data_key: CachedDataKey,  // DEK cached in memory, rotated periodically
}

pub struct CachedDataKey {
    plaintext: [u8; 32],     // AES-256 key
    encrypted: Vec<u8>,       // KMS-encrypted copy stored in WAL header
    created_at: Instant,
}
```

**Key hierarchy:**
- Master key: stored in AWS KMS, never leaves KMS
- Data Encryption Key (DEK): generated via `kms:GenerateDataKey`, cached in engine memory
- Each PAX block and WAL record gets a unique 96-bit nonce (counter-based)

**Performance:** AES-NI hardware acceleration on x86-64 and ARM64 (ARMv8 crypto extensions). Target: < 8% CPU overhead.

#### 2.2 TLS 1.3 Wire Encryption

All PostgreSQL wire protocol connections use TLS 1.3 via `rustls`. The engine generates a self-signed certificate on first startup or accepts a user-provided certificate via config.

---

### 3. Statistics Collection — `galaxdb-storage` (Req 10)

```rust
pub struct TableStatistics {
    row_count: u64,
    columns: HashMap<String, ColumnStats>,
    multi_column: Vec<CorrelationStats>,
    last_analyzed: Option<Timestamp>,
}

pub struct ColumnStats {
    ndv: u64,                          // number of distinct values
    null_fraction: f64,
    histogram: EquiHeightHistogram,    // configurable bucket count, default 100
}

pub struct CorrelationStats {
    columns: Vec<String>,
    correlation_matrix: Vec<f64>,      // PostgreSQL extended statistics model
}
```

**ANALYZE execution:** Runs as a background tokio task. Samples PAX blocks (reservoir sampling for large tables), computes NDV via HyperLogLog, builds equi-height histograms, and computes multi-column correlations. Does not block user queries — reads snapshot-consistent data.

**Usage:** The query planner uses these statistics for filter selectivity estimation and for choosing between HNSW graph traversal vs brute-force scan in hybrid queries.

---

### 4. I/O Abstraction Layer — `galaxdb-io` (Req 11)

```rust
pub trait IoScheduler: Send + Sync {
    async fn read(&self, file: &Path, offset: u64, len: usize, priority: IoPriority) -> Result<Vec<u8>>;
    async fn write(&self, file: &Path, offset: u64, data: &[u8], priority: IoPriority) -> Result<()>;
    async fn fsync(&self, file: &Path) -> Result<()>;
    fn latency_report(&self) -> LatencyReport;
}

pub enum IoPriority { High, Background }

pub struct LatencyReport {
    pub hp_p99_us: u64,
    pub bk_p99_us: u64,
    pub hp_idle_baseline_us: u64,
}
```

**Implementations:**
- `IoUringScheduler` — Linux 5.10+. Two separate io_uring instances: HP queue for user-facing reads/writes, BK queue for compaction and flush. The HP queue latency is monitored every 100 ms; if P99 exceeds 1.5× idle baseline for 3 consecutive windows, the scheduler signals the RateLimiter.
- `TokioScheduler` — macOS, Windows, or when `GALAXDB_IO_BACKEND=tokio`. Uses `tokio::fs` with no queue separation. No HP/BK latency guarantees.

**Selection:** At startup, check platform + env var. Linux 5.10+ with io_uring available and `GALAXDB_IO_BACKEND != tokio` → `IoUringScheduler`. Otherwise → `TokioScheduler`.

---

### 5. SQL Layer — `galaxdb-sql` (Reqs 12, 14, 15, 16, 22)

#### 5.1 SQL Parser (Req 12)

Built on `sqlparser-rs` with a custom `AuroraSqlDialect` that extends the PostgreSQL dialect:

```rust
pub enum AuroraStatement {
    // Standard SQL (delegated to sqlparser)
    Standard(sqlparser::ast::Statement),
    // AuroraSQL extensions
    CreateTableWithEmbedding { table: CreateTable, embedding_cols: Vec<EmbeddingColDef> },
    SemanticMatch { column: String, query: String, threshold: f64 },
    AtVersion { version: VersionRef, consistency: ConsistencyMode },
    CreateVersionTag { name: String, training: Option<TrainingOpts> },
    BulkInsert { table: String, rows: Vec<Vec<Value>> },
    ShowEmbeddingHealth { table: Option<String> },
    BackupTo { path: String },
    RestoreFrom { path: String },
    Analyze { table: String },
}

pub struct EmbeddingColDef {
    pub column_name: String,
    pub model_name: String,
    pub dimensions: Option<u32>,
}

pub enum ConsistencyMode { RowSnapshot, SemanticFresh }

pub struct TrainingOpts {
    pub precision: Option<TrainingPrecision>,  // sq8, rabitq, float32
    pub seed: Option<u64>,
}
```

**Error handling:** On parse failure, return error with byte offset position and descriptive message.

#### 5.2 Query Planner & Executor (Reqs 14, 15, 22)

```rust
pub enum QueryPlan {
    PointLookup { table: String, key: Value },
    FullScan { table: String, filter: Option<Expr>, zone_map_pruning: bool },
    SemanticSearch { table: String, column: String, query_embedding: Vec<f32>,
                     threshold: f64, strategy: SearchStrategy },
    HybridSearch { structured_filter: Expr, semantic: SemanticSearch, strategy: SearchStrategy },
    Insert { table: String, row: Row },
    Update { table: String, key: Value, updates: Vec<(String, Value)> },
    Delete { table: String, key: Value },
    BulkInsert { table: String, blocks: Vec<PaxBlock> },
    CreateTable { def: TableDef },
    DropTable { name: String },
    CreateVersionTag { tag: VersionTagDef },
    Backup { path: String },
    Restore { path: String },
    Analyze { table: String },
    ShowEmbeddingHealth { table: Option<String> },
}

pub enum SearchStrategy {
    HnswWithPostFilter,   // moderate-to-high cardinality filters
    BruteForceFiltered,   // very low cardinality (high selectivity)
}
```

**Adaptive planner logic (Req 22):**
1. Estimate filter cardinality using `TableStatistics`
2. If estimated matching rows < 1000 (or < 0.1% of table) → `BruteForceFiltered`
3. Otherwise → `HnswWithPostFilter`
4. Log chosen strategy via tracing span

**DDL execution (Req 14):**
- `CREATE TABLE`: allocate table metadata in catalog, initialize per-table ART index, create memtable. If embedding columns present, register with sidecar manager.
- `DROP TABLE`: remove catalog entry, schedule SST file deletion, remove ART entries, remove HNSW index file.

**DML execution (Req 15):**
- `INSERT`: write to memtable + WAL, update ART, trigger async embedding for embedding columns
- `SELECT`: ART lookup (point) or scan with zone-map pruning + Bloom filter checks
- `UPDATE`: write new MVCC version. If target column is embedding source → reject with error suggesting DELETE+INSERT
- `DELETE`: write tombstone to memtable + WAL, write DELTA_TOMBSTONE for vector index
- `BULK INSERT`: bypass memtable, write sorted rows directly as PAX blocks

#### 5.3 Snapshot Isolation (Req 16)

```rust
pub struct TransactionManager {
    next_timestamp: AtomicU64,
    active_snapshots: RwLock<BTreeSet<u64>>,  // timestamps of active readers
}

pub struct Snapshot {
    read_timestamp: u64,
    write_set: Vec<(Vec<u8>, u64)>,  // (key, write_timestamp) for conflict detection
}
```

**Guarantees:**
- Each transaction gets a monotonically increasing `read_timestamp`
- Reads see only versions with `commit_ts <= read_timestamp`
- Write-write conflict: if two transactions write the same key, the second to commit is aborted
- No dirty reads, no non-repeatable reads, no phantoms
- Write-skew is possible (documented limitation, SSI deferred to v2)


---

### 6. Wire Protocol — `galaxdb-wire` (Reqs 13, 33, 34)

#### 6.1 PostgreSQL Simple Query Protocol (Req 13)

```rust
pub struct WireServer {
    listener: TcpListener,
    tls_config: Arc<rustls::ServerConfig>,
    max_connections: usize,           // default 1000
    active_connections: AtomicUsize,
}
```

**Message flow:**
1. Client connects → TLS 1.3 handshake via `rustls`
2. `StartupMessage` → engine responds with `AuthenticationOk`, `ParameterStatus` (server_version, server_encoding, etc.), `BackendKeyData`, `ReadyForQuery`
3. Client sends `Query` (Q message) with SQL string
4. Engine parses → plans → executes → responds with `RowDescription` + `DataRow`* + `CommandComplete` + `ReadyForQuery`
5. On error: `ErrorResponse` with SQLSTATE code + message

Each connection is a separate `tokio::spawn` task. When `active_connections >= max_connections`, new connections get `ErrorResponse` with `53300` (too_many_connections) and the socket is closed.

#### 6.2 pg_catalog Stubs (Req 33)

Minimal system tables to satisfy psycopg2 and SQLAlchemy introspection:

| Table | Columns Implemented | Purpose |
|-------|-------------------|---------|
| `pg_catalog.pg_class` | oid, relname, relnamespace, relkind | Table listing |
| `pg_catalog.pg_attribute` | attrelid, attname, atttypid, attnum, attnotnull | Column metadata |
| `pg_catalog.pg_type` | oid, typname, typlen, typtype | Type system |
| `pg_catalog.pg_namespace` | oid, nspname | Schema listing |
| `pg_catalog.pg_database` | oid, datname | Database listing |

Queries against unsupported `pg_catalog` tables return empty result sets (not errors).

#### 6.3 Connection Management (Req 34)

- Each connection = one `tokio::spawn` async task
- `AtomicUsize` counter tracks active connections
- Configurable max (default 1000) via startup config
- No external pooler needed — tokio's task scheduler handles multiplexing

---

### 7. Vector Index — `galaxdb-vector` (Reqs 17, 18, 21)

#### 7.1 HNSW Base Graph + Delta Buffer (Req 17)

```rust
pub struct VectorIndex {
    base_graph: Arc<MmapHnswGraph>,     // mmap'd .hnsw file, read-only between merges
    delta_buffer: RwLock<DeltaBuffer>,   // in-memory exact k-NN
    merge_trigger: MergeTrigger,
}

pub struct MmapHnswGraph {
    mmap: memmap2::Mmap,
    metadata: HnswMetadata,  // M, ef_construction, max_level, entry_point
}

pub struct DeltaBuffer {
    vectors: Vec<(u64, Vec<f32>)>,       // (row_id, raw_vector)
    tombstones: HashSet<u64>,            // deleted row_ids
    quantized: Vec<(u64, Vec<u8>)>,      // quantized copies for fast scan
}

pub struct MergeTrigger {
    threshold: usize,  // max(10_000, total_indexed * 0.01)
    tombstone_emergency: f64,  // 0.20 (20%)
}
```

**Query path:**
1. Search HNSW base graph → top-K candidates with approximate scores
2. Search delta buffer (exact brute-force k-NN) → candidates
3. Union both candidate sets
4. Re-rank by exact cosine similarity against raw vectors from PAX blocks
5. Apply similarity threshold filter
6. Return final results

**Merge process:**
1. Build new HNSW graph in shadow file (`.hnsw.new`) incorporating base + delta vectors, excluding tombstones
2. `fsync` the shadow file
3. `rename(".hnsw.new", ".hnsw")` — atomic, crash-safe
4. Clear delta buffer
5. Old mmap is released when all in-flight queries complete (Arc reference counting)

**Emergency merge:** If `tombstones.len() > 0.20 * total_indexed`, trigger merge regardless of delta buffer size.

**Crash recovery:** Delta buffer entries are WAL-backed (record types `DELTA_INSERT`, `DELTA_TOMBSTONE`). On recovery, replay WAL delta records in batches of 1000 to rebuild the delta buffer. The base graph file is always consistent due to atomic rename.

#### 7.2 Platform-Aware Quantization (Req 18)

```rust
pub trait Quantizer: Send + Sync {
    fn quantize(&self, vector: &[f32]) -> Vec<u8>;
    fn dequantize(&self, quantized: &[u8]) -> Vec<f32>;
    fn distance(&self, a: &[u8], b: &[u8]) -> f32;  // SIMD-accelerated
    fn compression_ratio(&self) -> f32;
}

pub struct Sq8Quantizer { /* AVX2/AVX-512 SIMD kernels */ }
pub struct Fp16Quantizer { /* ARM NEON kernels */ }
pub struct RabitqQuantizer { /* random rotation matrix + binary quantization */ }
```

**Platform detection at startup:**
```rust
fn select_default_quantizer() -> Box<dyn Quantizer> {
    if cfg!(target_arch = "x86_64") && is_x86_feature_detected!("avx2") {
        Box::new(Sq8Quantizer::new())       // 4× compression
    } else if cfg!(target_arch = "aarch64") {
        Box::new(Fp16Quantizer::new())      // 2× compression, NEON-native
    } else {
        Box::new(Sq8Quantizer::new())       // fallback
    }
}
```

**RaBitQ (opt-in):** User enables via config. Applies random orthogonal rotation matrix then binary quantization. 32× compression. Available on both x86-64 (AVX2) and ARM64 (NEON).

#### 7.3 SEMANTIC_MATCH Execution (Req 21)

```rust
pub async fn execute_semantic_match(
    sidecar: &SidecarClient,
    vector_index: &VectorIndex,
    query_text: &str,
    threshold: f64,
    filter: Option<&Expr>,
    stats: &TableStatistics,
) -> Result<Vec<ScoredRow>> {
    // 1. Embed query text via sidecar
    let query_vec = sidecar.embed(query_text).await?;

    // 2. Choose strategy based on filter cardinality (Req 22)
    let strategy = if let Some(filter) = filter {
        let cardinality = stats.estimate_cardinality(filter);
        if cardinality < 1000 || cardinality as f64 / stats.row_count as f64 < 0.001 {
            SearchStrategy::BruteForceFiltered
        } else {
            SearchStrategy::HnswWithPostFilter
        }
    } else {
        SearchStrategy::HnswWithPostFilter
    };

    // 3. Execute search
    match strategy {
        SearchStrategy::HnswWithPostFilter => {
            let candidates = vector_index.search_hnsw_and_delta(&query_vec, top_k * 2);
            candidates.into_iter()
                .filter(|c| c.similarity >= threshold)
                .filter(|c| filter.map_or(true, |f| evaluate_filter(f, &c.row)))
                .collect()
        }
        SearchStrategy::BruteForceFiltered => {
            let filtered_rows = storage.scan_with_filter(filter);
            brute_force_knn(&query_vec, &filtered_rows, threshold)
        }
    }
}
```

If the sidecar is unavailable (degraded mode), return `ErrorResponse` with message "semantic search temporarily unavailable — embedding sidecar is down".

---

### 8. Embedding Sidecar — `galaxdb-sidecar` (Reqs 19, 20, 39)

#### 8.1 Sidecar Binary Architecture (Req 19)

```rust
// Separate binary: galaxdb-sidecar
pub struct EmbeddingSidecar {
    ort_session: ort::Session,           // ONNX Runtime session
    model_id: String,
    model_version: String,
    dimensions: usize,
    socket: UnixListener,
    in_flight: AtomicUsize,              // current in-flight count
    max_in_flight: usize,               // 10,000
}
```

**Communication protocol (Unix socket):**
```
Request:  [u32 length][JSON: {"row_id": u64, "text": "...", "column": "..."}]
Response: [u32 length][JSON: {"row_id": u64, "embedding": [f32; dim], "model_version": "..."}]
```

**Parent PID monitoring:**
- Linux: `prctl(PR_SET_PDEATHSIG, SIGTERM)` — kernel sends SIGTERM when parent dies
- macOS: `kqueue` with `EVFILT_PROC` + `NOTE_EXIT` on parent PID
- Windows: named pipe heartbeat (not in v1 production scope)

**Heartbeat:** Sidecar sends ping every 5 seconds. Engine expects response within 2 seconds. 3 missed pings → engine enters degraded mode.

**Crash recovery:** Engine restarts sidecar with exponential backoff: 1s, 2s, 4s, 8s, 16s, 32s, 60s (capped). During restart, writes continue but embeddings stay stale.

#### 8.2 Back-Pressure & Backlog (Req 19)

```rust
pub struct EmbeddingManager {
    sidecar_client: SidecarClient,
    in_flight_semaphore: Semaphore,      // capacity 10,000
    backlog_writer: BacklogWriter,
}
```

When `in_flight >= 10,000`:
1. New embedding requests are written to `_galaxdb_embedding_backlog` system table
2. The backlog table uses `DURABILITY STRICT` regardless of session setting
3. A background scanner drains the backlog when sidecar capacity recovers
4. Backlog is drained FIFO before accepting new in-flight requests

#### 8.3 Model-Version Tracking (Req 20)

Each embedded row stores `_embedding_model_version: String` alongside the embedding vector. When the sidecar reports a new model version:

1. Engine scans all rows where `_embedding_model_version != current_version`
2. Sets `_embedding_stale = true` on those rows (Req 39)
3. Enqueues them into the backlog table for re-embedding
4. `SHOW EMBEDDING HEALTH` queries the catalog to report version distribution and re-embedding progress

#### 8.4 Embedding Staleness Tracking (Req 39)

Every row with an embedding column carries two system columns:
- `_embedding_stale: bool` — true when embedding is pending or outdated
- `_embedding_model_version: String` — model version that produced the embedding

These are written through the standard LSM update path (same WAL, same MVCC), ensuring the flag and embedding value are always consistent from a reader's perspective.


---

### 9. Versioning & Training — `galaxdb-versioning` (Reqs 23, 24, 25, 26, 38)

#### 9.1 Merkle DAG (Req 23)

```rust
pub struct MerkleDag {
    roots: BTreeMap<u64, MerkleRoot>,    // commit_timestamp → root
    tags: HashMap<String, VersionTag>,
}

pub struct MerkleRoot {
    timestamp: u64,
    root_hash: [u8; 32],                // XXH3-128 over child block hashes
    block_hashes: Vec<[u8; 32]>,        // hashes of PAX blocks in this version
}
```

**On commit:** Each committed PAX block's XXH3-64 checksum is collected. A Merkle tree is computed over these hashes to produce a version root. The root is stored in the Merkle DAG index.

**AT VERSION query execution:**
1. Resolve version reference (timestamp or tag name) to a `MerkleRoot`
2. Filter PAX blocks: include only blocks with `commit_timestamp <= version_timestamp`
3. For tag references, use the tag's stored Merkle root to identify exact block set

**Semantic guardrails:**
- `AT VERSION` + `SEMANTIC_MATCH` with no consistency mode → `ROW_SNAPSHOT` (default) → reject `SEMANTIC_MATCH` with error: "SEMANTIC_MATCH with AT VERSION requires consistency mode"
- `AT VERSION` + `CONSISTENCY 'SEMANTIC_FRESH'` → search current HNSW index against historical rows, include warning in result metadata
- `CONSISTENCY 'SEMANTIC_SNAPSHOT'` → return error "SEMANTIC_SNAPSHOT is a v2 feature"

#### 9.2 Version Tags (Req 24)

```rust
pub struct VersionTag {
    name: String,
    merkle_root: MerkleRoot,
    created_at: Timestamp,
    training: Option<TrainingTagMetadata>,
    pinned_blocks: Vec<BlockId>,         // GC-exempt
}

pub struct TrainingTagMetadata {
    precision: TrainingPrecision,        // float32, sq8, rabitq
    seed: Option<u64>,                   // shuffle seed for reproducibility
    deterministic_order: bool,           // true → primary key sort
}
```

When a tag is created:
- The current Merkle root is captured
- All referenced PAX blocks are marked as pinned (GC-exempt) in the compactor's pin set
- `FOR TRAINING` tags additionally store precision, seed, and guarantee deterministic block order (primary key sort)

#### 9.3 Lance Training Export (Req 25)

```rust
pub struct LanceExporter {
    lance_writer: lance::dataset::WriteParams,
}

impl LanceExporter {
    pub async fn export(
        &self,
        tag: &VersionTag,
        storage: &StorageEngine,
        filter: Option<&Expr>,
        dedup: bool,
    ) -> Result<PathBuf> {
        let blocks = storage.get_blocks_for_version(&tag.merkle_root);
        // Sort by primary key for deterministic order
        let sorted_blocks = sort_by_primary_key(blocks);

        let mut writer = lance::dataset::Dataset::write(/* ... */);
        for block in sorted_blocks {
            let rows = block.read_rows()?;
            let rows = if dedup { filter_duplicates(rows) } else { rows };
            let batch = to_arrow_batch(rows, &tag.training.precision)?;
            writer.write(batch)?;
        }
        // Record lineage (Req 38)
        self.record_export_lineage(tag, filter, dedup)?;
        Ok(output_path)
    }
}
```

**Training precision:** When exporting, embeddings are quantized to the requested precision:
- `float32`: raw 4-byte floats (no conversion)
- `sq8`: int8 scalar quantization (4× I/O reduction)
- `rabitq`: binary quantization (32× I/O reduction)

**Python integration:** `galaxdb.training_dataset(tag)` calls the export, then wraps the Lance dataset as a PyTorch `IterableDataset` with zero-copy memory-mapped access.

#### 9.4 MinHash Near-Duplicate Detection (Req 26)

```rust
pub struct MinHashDedup {
    num_hashes: usize,        // 128
    signature_bytes: usize,   // 512
}

impl MinHashDedup {
    pub fn compute_signature(&self, text: &str) -> [u8; 512] {
        // 128 independent hash functions over character n-grams
        // Each hash selects the minimum hash value → 4 bytes per hash → 512 bytes total
    }

    pub fn jaccard_estimate(&self, sig_a: &[u8; 512], sig_b: &[u8; 512]) -> f64 {
        // Count matching hash values / 128
    }
}
```

**Write path integration:** On every INSERT of a row with TEXT columns, the MinHash signature is computed in the Rust write path (not in the sidecar — per Finding #3 in the audit trail) and stored as a system column `_minhash_signature`.

**Background refresh:** A periodic background job groups rows with Jaccard similarity > 0.8 and marks them in a `_near_duplicate_group` system column.

**Query filter:** `WHERE NOT DUPLICATE` excludes rows that are in a near-duplicate group (keeps one representative per group).

#### 9.5 Training Data Lineage (Req 38)

```rust
pub struct TrainingExportRecord {
    tag_name: String,
    filter_expression: Option<String>,
    precision: TrainingPrecision,
    dedup_flag: bool,
    curriculum_mode: Option<String>,
    row_count: u64,
    export_timestamp: Timestamp,
    content_hash: [u8; 32],
}
```

The `_galaxdb_training_exports` system table is append-only. Every training export inserts a record. User queries cannot DELETE from this table. This satisfies EU AI Act Article 13 lineage requirements.

---

### 10. Backup & Restore — `galaxdb-versioning` (Req 27)

```rust
pub struct BackupModule {
    storage: Arc<StorageEngine>,
    merkle_dag: Arc<MerkleDag>,
}

impl BackupModule {
    pub async fn backup(&self, target_path: &Path) -> Result<()> {
        // 1. Acquire write-quiesce (< 100 ms)
        //    - Flush active memtable
        //    - Create clean Merkle root
        //    - Reads continue during quiesce
        let quiesce_guard = self.storage.acquire_write_quiesce().await?;

        // 2. Copy PAX blocks + WAL + blob log to target
        //    - New writes resume immediately after copy begins
        self.copy_data_files(target_path).await?;

        drop(quiesce_guard); // releases write quiesce
        Ok(())
    }

    pub async fn restore(&self, source_path: &Path) -> Result<()> {
        // 1. Validate all block checksums
        self.validate_checksums(source_path)?;
        // 2. Copy files to data directory
        self.copy_data_files_from(source_path).await?;
        // 3. Replay WAL
        self.replay_wal()?;
        // 4. Rebuild ART index from SST files
        self.rebuild_art_index()?;
        // 5. Rebuild HNSW graph from PAX block embeddings
        self.rebuild_hnsw_index()?;
        Ok(())
    }
}
```

**Write-quiesce:** Implemented as a `RwLock` — backup takes a write lock (blocking new writes for < 100 ms to flush), then downgrades to allow writes to resume while the copy proceeds. Reads are never blocked.

---

### 11. Observability — `galaxdb-observe` (Req 28)

```rust
pub struct ObservabilityModule {
    http_server: axum::Router,           // /health, /metrics
    metrics_registry: prometheus::Registry,
    tracer: opentelemetry::global::BoxedTracer,
}
```

#### 11.1 HTTP Endpoints

- `GET /health` → `{"status": "ok", "sidecar": "connected"|"degraded", "uptime_seconds": N}`
- `GET /metrics` → Prometheus text exposition format

#### 11.2 Metrics Exported

| Metric | Type | Description |
|--------|------|-------------|
| `galaxdb_buffer_pool_hot_set_usage` | Gauge | HotSet utilization % |
| `galaxdb_buffer_pool_scan_buffer_usage` | Gauge | ScanBuffer utilization % |
| `galaxdb_embedding_queue_depth` | Gauge | In-flight embedding requests |
| `galaxdb_embedding_backlog_depth` | Gauge | Rows in backlog table |
| `galaxdb_checkpoint_last_duration_ms` | Gauge | Last checkpoint duration |
| `galaxdb_compaction_pending_bytes` | Gauge | Pending compaction bytes |
| `galaxdb_wal_write_latency_us` | Histogram | WAL write latency |
| `galaxdb_hnsw_recall_estimate` | Gauge | Estimated recall from sampling |
| `galaxdb_connections_active` | Gauge | Active client connections |
| `galaxdb_disk_full` | Gauge | 1 if disk-full condition active |
| `galaxdb_sidecar_status` | Gauge | 1=connected, 0=degraded |

#### 11.3 Tracing

OpenTelemetry W3C `traceparent` propagation. Every query creates a root span. Child spans for:
- SQL parsing
- Query planning (logs chosen strategy)
- HNSW search
- Delta buffer search
- Embedding sidecar call
- PAX block reads

SQL commenter format carries trace context through the wire protocol: `/* traceparent=00-... */`.

#### 11.4 Logging

Structured JSON via `tracing-subscriber` with `tracing-subscriber::fmt::json()`. Configurable level via `GALAXDB_LOG_LEVEL` env var (default: `info`). Every log line includes `traceparent` when available.

---

### 12. Write Stall Mitigation (Reqs 29, 30)

#### 12.1 RateLimiter (Req 29)

```rust
pub struct RateLimiter {
    token_bucket: TokenBucket,
    max_rate: u64,                       // bytes/sec, calibrated at startup
    current_ceiling: AtomicU64,
    hp_latency_monitor: LatencyMonitor,
}
```

**Startup calibration:** Measure NVMe sequential write bandwidth with a 1 MB test write. Set `max_rate = measured_bandwidth * 0.70`.

**Dynamic adjustment:** The I/O scheduler reports HP-queue P99 latency every 100 ms. If P99 exceeds 1.5× idle baseline for 3 consecutive windows:
- Lower `current_ceiling` by 30%
- When latency returns to normal, restore to previous ceiling

The compactor and flush tasks acquire tokens from this bucket before issuing I/O.

#### 12.2 WriteController (Req 30)

```rust
pub struct WriteController {
    soft_limit: u64,          // default 32 GB
    hard_limit: u64,          // default 64 GB
    delayed_write_rate: u64,  // default 16 MB/s
    check_interval: Duration, // 1 ms
}
```

**Logic (runs every 1 ms):**
1. Read `compaction_pending_bytes` from compactor
2. If `pending < soft_limit` → full speed (no throttle)
3. If `soft_limit <= pending < hard_limit` → throttle writes proportionally: `rate = delayed_write_rate * (hard_limit - pending) / (hard_limit - soft_limit)`
4. If `pending >= hard_limit` → block all writes until pending drops below hard_limit

---

### 13. Deployment — `galaxdb-server`, `galaxdb-embedded` (Req 35)

#### 13.1 Standalone Server Mode

```rust
// galaxdb-server binary
#[tokio::main]
async fn main() {
    let config = Config::from_args_and_env();
    let engine = Engine::open(config.data_dir).await?;
    let wire_server = WireServer::bind(config.listen_addr, engine).await?;
    let http_server = ObservabilityModule::bind(config.metrics_addr).await?;

    // Spawn sidecar if embedding models configured
    if config.has_embedding_models() {
        SidecarManager::spawn(config.sidecar_path).await?;
    }

    tokio::select! {
        _ = wire_server.serve() => {},
        _ = http_server.serve() => {},
        _ = signal::ctrl_c() => { engine.shutdown().await; },
    }
}
```

**Binary size targets:**
- Core engine (`galaxdb-server`): < 70 MB statically linked
- Full install (+ sidecar + default ONNX model): < 350 MB

#### 13.2 Embedded Mode

```rust
// galaxdb-embedded crate, exposed via PyO3
#[pyclass]
pub struct Database {
    engine: Arc<Engine>,
    sidecar: Option<SidecarManager>,
}

#[pymethods]
impl Database {
    #[new]
    fn new(path: &str) -> PyResult<Self> { /* open engine, optionally spawn sidecar */ }
    fn execute(&self, sql: &str) -> PyResult<Vec<PyRow>> { /* parse, plan, execute */ }
    fn training_dataset(&self, tag: &str) -> PyResult<PyObject> { /* Lance export → PyTorch IterableDataset */ }
}
```

#### 13.3 Python Client (Req 32)

The Python client (`galaxdb-python` package) provides two modes:
1. **Embedded:** `import galaxdb; db = galaxdb.Database("./mydata")` — in-process via PyO3
2. **Remote:** `import galaxdb; db = galaxdb.connect("host=localhost port=5432")` — PostgreSQL wire protocol

Both expose:
- `db.execute(sql)` → list of rows
- `db.training_dataset(tag)` → PyTorch `IterableDataset` backed by Lance
- Compatible with Python 3.9+

---

### 14. Chaos Testing (Req 37)

```rust
// tests/chaos/mod.rs
pub struct ChaosHarness {
    engine: Engine,
    sidecar: SidecarManager,
}

impl ChaosHarness {
    /// Kill sidecar mid-request, verify engine recovers and drains backlog
    pub async fn test_sidecar_kill(&self) { /* ... */ }

    /// Kill engine mid-flush, verify recovery produces consistent state
    pub async fn test_engine_kill_mid_flush(&self) { /* ... */ }

    /// Corrupt WAL records, verify recovery skips corrupt and recovers valid data
    pub async fn test_wal_corruption(&self) { /* ... */ }

    /// Fill disk, verify clean shutdown without data corruption
    pub async fn test_disk_full(&self) { /* ... */ }

    /// Verify all recovery scenarios complete in < 30 seconds
    pub async fn test_recovery_time(&self) { /* ... */ }
}
```

Each test:
1. Sets up a populated database with known data
2. Injects the fault (kill process, corrupt bytes, fill disk)
3. Restarts the engine
4. Verifies: no committed data lost, recovery < 30s, backlog drained after sidecar restart

---

## Data Models

### PAX Block On-Disk Format

```
Offset  Size    Field
0       4       magic (0x47414C41)
4       1       format_version
5       8       block_id
13      4       row_count
17      8       commit_timestamp
25      2       column_count
27      var     column_descriptors[column_count]
                  - col_type: u8
                  - codec: u8 (0=none, 1=fastpfor, 2=zstd)
                  - offset: u32
                  - compressed_len: u32
                  - zone_map_min: var
                  - zone_map_max: var
var     var     column_chunks[column_count]
var     var     row_offset_table[row_count] (u32 each)
EOF-8   8       xxh3_64_checksum (over bytes 0..EOF-8)
```

### WAL Record Format

```
Offset  Size    Field
0       1       record_type (0x01=PUT, 0x02=DELETE, 0x03=DELTA_INSERT, etc.)
1       8       sequence_number
9       4       payload_length (after LZ4 compression)
13      8       xxh3_64_checksum (over uncompressed payload)
21      var     lz4_compressed_payload
```

### System Tables

| Table | Columns | Purpose |
|-------|---------|---------|
| `_galaxdb_embedding_backlog` | row_id, table_name, column_name, text_content, enqueued_at | Overflow embedding requests |
| `_galaxdb_training_exports` | tag_name, filter_expr, precision, dedup, curriculum, row_count, exported_at, content_hash | Training lineage (append-only) |
| `_galaxdb_versions` | tag_name, merkle_root_hash, created_at, training_opts, pinned | Version tag catalog |

### System Columns (per row with embedding)

| Column | Type | Description |
|--------|------|-------------|
| `_embedding_stale` | bool | true if embedding is pending or model version changed |
| `_embedding_model_version` | String | model version that produced the embedding |
| `_minhash_signature` | [u8; 512] | MinHash LSH signature for near-dedup |
| `_near_duplicate_group` | Option<u64> | group ID if row is a near-duplicate |

---

## Requirement Traceability Matrix

| Requirement | Design Section | Components |
|-------------|---------------|------------|
| Req 1: Memtable Write Path | §1.1 | galaxdb-storage |
| Req 2: PAX Block Format | §1.2 | galaxdb-storage |
| Req 3: ART Primary Key Index | §1.3 | galaxdb-storage |
| Req 4: Bloom Filter Monkey | §1.4 | galaxdb-storage |
| Req 5: NUMA Buffer Pool | §1.5 | galaxdb-storage |
| Req 6: Lazy Leveling Compaction | §1.6 | galaxdb-storage |
| Req 7: WAL | §1.7 | galaxdb-storage |
| Req 8: KV Separation BVLSM | §1.8 | galaxdb-storage |
| Req 9: TDE Encryption | §2 | galaxdb-crypto |
| Req 10: Statistics Collection | §3 | galaxdb-storage |
| Req 11: I/O Abstraction | §4 | galaxdb-io |
| Req 12: SQL Parser AuroraSQL | §5.1 | galaxdb-sql |
| Req 13: PG Wire Protocol | §6.1 | galaxdb-wire |
| Req 14: DDL Execution | §5.2 | galaxdb-sql |
| Req 15: DML Execution | §5.2 | galaxdb-sql |
| Req 16: Snapshot Isolation | §5.3 | galaxdb-sql |
| Req 17: HNSW Vector Index | §7.1 | galaxdb-vector |
| Req 18: Platform Quantization | §7.2 | galaxdb-vector |
| Req 19: Embedding Sidecar | §8.1, §8.2 | galaxdb-sidecar |
| Req 20: Model-Version Tracking | §8.3 | galaxdb-sidecar |
| Req 21: SEMANTIC_MATCH | §7.3 | galaxdb-vector, galaxdb-sql |
| Req 22: Adaptive Planner | §5.2, §7.3 | galaxdb-sql |
| Req 23: Merkle DAG Versioning | §9.1 | galaxdb-versioning |
| Req 24: Version Tags | §9.2 | galaxdb-versioning |
| Req 25: Lance Training Export | §9.3 | galaxdb-versioning |
| Req 26: MinHash Near-Dedup | §9.4 | galaxdb-versioning |
| Req 27: Backup & Restore | §10 | galaxdb-versioning |
| Req 28: Observability | §11 | galaxdb-observe |
| Req 29: RateLimiter | §12.1 | galaxdb-storage |
| Req 30: WriteController | §12.2 | galaxdb-storage |
| Req 31: Disk Full Handling | §1.9 | galaxdb-storage |
| Req 32: Python Client | §13.3 | galaxdb-python |
| Req 33: pg_catalog Stubs | §6.2 | galaxdb-wire |
| Req 34: Connection Management | §6.3 | galaxdb-wire |
| Req 35: Deployment Modes | §13.1, §13.2 | galaxdb-server, galaxdb-embedded |
| Req 36: vLSM Improvements | §1.6 | galaxdb-storage |
| Req 37: Chaos Testing | §14 | tests/chaos |
| Req 38: Training Lineage | §9.5 | galaxdb-versioning |
| Req 39: Embedding Staleness | §8.4 | galaxdb-sidecar |
