# Implementation Tasks — GalaxDB v1 Engine

## Month 1: Core Storage Engine

- [x] 1. Scaffold Rust workspace and shared types
  - [x] 1.1 Create Cargo workspace with crates: galaxdb-common, galaxdb-storage, galaxdb-io, galaxdb-crypto, galaxdb-vector, galaxdb-sql, galaxdb-wire, galaxdb-versioning, galaxdb-sidecar, galaxdb-observe, galaxdb-server, galaxdb-embedded
  - [x] 1.2 Define shared error types, config structs, and type aliases in galaxdb-common (TableId, BlockId, RowId, Timestamp, ColumnType enum)
  - [x] 1.3 Add workspace dependencies: crossbeam-skiplist, xxhash-rust, lz4_flex, zstd, tokio, memmap2, aes-gcm, aws-sdk-kms, rustls, prometheus, tracing, tracing-subscriber, opentelemetry, half
  - [x] 1.4 Set up CI with `cargo build --release`, `cargo test`, `cargo clippy` in GitHub Actions

- [x] 2. Implement I/O abstraction layer — galaxdb-io (Req 11)
  - [x] 2.1 Define `IoScheduler` trait with async read/write/fsync methods and `IoPriority` enum (High, Background)
  - [x] 2.2 Implement `TokioScheduler` using `tokio::fs` for macOS/Windows fallback
  - [x] 2.3 Implement `IoUringScheduler` with separate HP and BK io_uring submission queues (Linux 5.10+ only, behind `#[cfg(target_os = "linux")]`)
  - [x] 2.4 Add HP-queue latency monitoring (100 ms windows) and `LatencyReport` struct for RateLimiter feedback
  - [x] 2.5 Add startup detection: check platform + `GALAXDB_IO_BACKEND` env var to select scheduler implementation
  - [x] 2.6 Write unit tests for both scheduler implementations

- [x] 3. Implement WAL — galaxdb-storage (Req 7)
  - [x] 3.1 Define WAL record format: `[type: u8][seq_no: u64][length: u32][xxh3_checksum: u64][lz4_payload: bytes]`
  - [x] 3.2 Define record types: ROW_PUT (0x01), ROW_DELETE (0x02), DELTA_INSERT (0x03), DELTA_TOMBSTONE (0x04), CHECKPOINT (0x05), BLOB_REF (0x06)
  - [x] 3.3 Implement WAL writer with LZ4 compression and XXH3-64 checksums per record
  - [x] 3.4 Implement group commit: background task batches writes over configurable interval (default 10 ms), single fsync per batch
  - [x] 3.5 Implement DURABILITY STRICT mode: bypass group commit, fsync each commit individually
  - [x] 3.6 Implement DURABILITY RELAXED mode: use group commit batch window
  - [x] 3.7 Implement checkpoint trigger: when WAL exceeds 512 MB or 60 seconds since last checkpoint
  - [x] 3.8 Implement WAL recovery: replay from last CHECKPOINT, verify XXH3-64 per record, skip corrupt records, stop at first checksum failure
  - [x] 3.9 Write tests: WAL write/read round-trip, group commit batching, checkpoint trigger, recovery with corrupt records, recovery time < 30s

- [x] 4. Implement Memtable — galaxdb-storage (Req 1)
  - [x] 4.1 Implement `Memtable` struct with 16-shard `Mutex<SkipMap>` using crossbeam-skiplist, shard selection via `xxh3_64(key) % 16`
  - [x] 4.2 Implement `VersionedValue` with MVCC version chains (timestamp, value, prev pointer)
  - [x] 4.3 Implement insert/update: acquire shard mutex, write versioned value, update AtomicU64 size counter
  - [x] 4.4 Implement seal logic: when size >= 64 MB, atomically swap to new empty memtable, enqueue sealed for flush
  - [x] 4.5 Implement back-pressure: Semaphore with 256 MB capacity tracking sealed-but-unflushed bytes, block writers when exceeded
  - [x] 4.6 Implement read path: copy value bytes out of Entry handle immediately, drop handle before any async boundary (epoch safety)
  - [x] 4.7 Write tests: concurrent writes to different shards (no contention), same-key serialization, seal threshold, back-pressure blocking, epoch safety

- [x] 5. Implement PAX block format — galaxdb-storage (Req 2)
  - [x] 5.1 Define PAX block header: magic (0x47414C41), format_version, block_id, row_count, commit_timestamp, column_count, column_descriptors (col_type, codec, offset, compressed_len, zone_map_min, zone_map_max)
  - [x] 5.2 Implement column chunk writer: fixed-width columns with delta encoding + bit-packing (FastPFOR), variable-width with Zstandard L3, embedding columns uncompressed
  - [x] 5.3 Implement row offset table at end of block
  - [x] 5.4 Implement XXH3-64 checksum computation over entire block, stored at block footer
  - [x] 5.5 Implement block reader with checksum + magic number verification, reject on mismatch
  - [x] 5.6 Implement zone map extraction (min/max per column) during block write
  - [x] 5.7 Write tests: write/read round-trip, checksum verification, corrupt block rejection, compression correctness per column type

- [x] 6. Implement memtable flush to SST — galaxdb-storage (Reqs 1, 2)
  - [x] 6.1 Implement flush pipeline: sealed memtable → sort by primary key → write PAX blocks → write to disk via IoScheduler → TDE encryption
  - [x] 6.2 Integrate WAL: write CHECKPOINT record after successful flush, truncate WAL
  - [x] 6.3 Write tests: flush produces valid PAX blocks, checkpoint advances WAL truncation point

- [x] 7. Implement ART primary key index — galaxdb-storage (Req 3)
  - [x] 7.1 Implement Adaptive Radix Tree with Node4, Node16, Node48, Node256 node types and path compression (Leis et al., ICDE 2013)
  - [x] 7.2 Implement `RowLocation` enum: Memtable { shard, key } or SST { sst_id, block_offset, row_offset }
  - [x] 7.3 Implement insert/lookup/delete operations on ART
  - [x] 7.4 Implement ART rebuild from SST files + WAL replay for crash recovery
  - [x] 7.5 Write tests: insert/lookup correctness, rebuild from SSTs, concurrent read/write safety

- [x] 8. Implement Bloom filters with Monkey allocation — galaxdb-storage (Req 4)
  - [x] 8.1 Implement per-SST Bloom filter construction with configurable bits-per-key
  - [x] 8.2 Implement Monkey-optimal FPR allocation: `FPR(level_i) = budget * (ratio^(L-i)) / sum(ratio^(L-j))` across LSM levels
  - [x] 8.3 Integrate Bloom filter check into point read path: consult filter before disk read, skip SST on negative
  - [x] 8.4 Write tests: false-positive rate within Monkey-allocated budget, correct skip behavior

- [x] 9. Implement NUMA-aware buffer pool — galaxdb-storage (Req 5)
  - [x] 9.1 Implement `BufferPool` with HotSet (70% RAM, LRU eviction) and ScanBuffer (30% RAM, clock-sweep eviction)
  - [x] 9.2 Implement `NumaPartitioned<T>` wrapper: detect NUMA node via libnuma (Linux) or single partition fallback (macOS/Windows)
  - [x] 9.3 Implement routing: point lookups → HotSet, sequential scans → ScanBuffer
  - [x] 9.4 Implement eviction constraint: ScanBuffer never evicts a HotSet-resident block
  - [x] 9.5 Write tests: LRU eviction correctness, clock-sweep correctness, NUMA allocation on Linux, cross-partition isolation

- [x] 10. Implement Lazy Leveling compaction with MVCC GC — galaxdb-storage (Req 6)
  - [x] 10.1 Implement LSM level structure: L0 tiered (up to 4 files), L1-L3 tiered, L4 leveled
  - [x] 10.2 Implement compaction trigger: L0 file count threshold, level size ratio threshold
  - [x] 10.3 Implement merge iterator: merge sorted runs, apply MVCC GC (discard versions not needed by active snapshots or pinned tags)
  - [x] 10.4 Implement compaction output: write new SST files (64 MB default), build Bloom filters, update ART index
  - [x] 10.5 Implement pinned tag awareness: retain all versions referenced by any pinned VersionTag
  - [x] 10.6 Write tests: compaction produces correct merged output, MVCC GC discards old versions, pinned versions retained

- [x] 11. Implement KV separation — Blob Log (Req 8)
  - [x] 11.1 Implement `BlobLog` with multi-queue parallel writers (4 queues default)
  - [x] 11.2 Implement WAL-time separation: during WAL entry construction, if value > 1 KB, write to blob log, store 32-byte content hash + BlobRef in memtable
  - [x] 11.3 Implement transparent blob fetch on read: detect BlobRef in PAX block, fetch from blob log
  - [x] 11.4 Implement blob GC: background task compacts blob files when discardable space > 50%
  - [x] 11.5 Write tests: large value separation, transparent read, GC reclaims space

- [x] 12. Implement TDE encryption — galaxdb-crypto (Req 9)
  - [x] 12.1 Implement `TdeModule` with AES-256-GCM encryption/decryption using `aes-gcm` crate
  - [x] 12.2 Implement pluggable key management via `KeyProvider` trait with LocalKeyProvider, EnvKeyProvider, and AwsKmsKeyProvider (stub behind feature flag)
  - [x] 12.3 Implement counter-based 96-bit nonce generation per block/record
  - [x] 12.4 Integrate TDE into PAX block write path: encrypt before IoScheduler write
  - [x] 12.5 Integrate TDE into WAL write path: encrypt each record before disk write
  - [x] 12.6 Implement TDE into read path: decrypt after IoScheduler read
  - [x] 12.7 Write tests: encrypt/decrypt round-trip, AES-NI acceleration detection, nonce uniqueness

- [x] 13. Implement statistics collection — galaxdb-storage (Req 10)
  - [x] 13.1 Define `TableStatistics`, `ColumnStats` (NDV, null_fraction, equi-height histogram), `CorrelationStats` structs
  - [x] 13.2 Implement ANALYZE command: background tokio task, reservoir sampling of PAX blocks, HyperLogLog for NDV, histogram construction
  - [x] 13.3 Implement multi-column correlation statistics following PostgreSQL extended statistics model
  - [x] 13.4 Store statistics in catalog, expose to query planner for selectivity estimation
  - [x] 13.5 Write tests: NDV accuracy, histogram bucket distribution, correlation computation

- [x] 14. Implement disk full handling — galaxdb-storage (Req 31)
  - [x] 14.1 Pre-allocate 32 MB reserve file (`_galaxdb_reserve`) at engine startup
  - [x] 14.2 On disk-full detection: delete reserve file, perform clean checkpoint, block writes, emit metric
  - [x] 14.3 Write tests: disk-full simulation, clean checkpoint before stop, no data corruption

- [x] 15. Implement write stall mitigation — RateLimiter (Req 29)
  - [x] 15.1 Implement auto-tuned token-bucket `RateLimiter`: calibrate max rate at startup (70% of NVMe write bandwidth)
  - [x] 15.2 Integrate with IoScheduler latency reports: lower ceiling by 30% when HP-queue P99 exceeds 1.5× baseline for 3 consecutive 100 ms windows
  - [x] 15.3 Restore ceiling when latency returns to normal
  - [x] 15.4 Integrate with compactor and flush tasks: acquire tokens before I/O
  - [x] 15.5 Write tests: rate limiting under load, dynamic ceiling adjustment

- [x] 16. Implement write stall mitigation — WriteController (Req 30)
  - [x] 16.1 Implement `WriteController` with soft limit (32 GB default) and hard limit (64 GB default)
  - [x] 16.2 Implement 1 ms check interval: read pending compaction bytes, apply proportional slowdown between soft and hard limits
  - [x] 16.3 Implement hard stop: block all writes when pending >= hard limit
  - [x] 16.4 Implement recovery: restore full throughput when pending < soft limit
  - [x] 16.5 Write tests: gradual slowdown, hard stop, recovery to full speed


## Month 2: SQL Layer & Wire Protocol

- [x] 17. Implement SQL parser with AuroraSQL extensions — galaxdb-sql (Req 12)
  - [x] 17.1 Add `sqlparser` crate dependency, create `AuroraSqlDialect` extending PostgreSQL dialect
  - [x] 17.2 Implement parsing for standard DDL: CREATE TABLE, DROP TABLE, ALTER TABLE
  - [x] 17.3 Implement parsing for standard DML: INSERT, SELECT, UPDATE, DELETE
  - [x] 17.4 Implement parsing for `EMBEDDING MODEL 'name' DIM n` in CREATE TABLE column definitions
  - [x] 17.5 Implement parsing for `SEMANTIC_MATCH(col, 'query', threshold)` predicate
  - [x] 17.6 Implement parsing for `AT VERSION timestamp_or_tag` with optional `CONSISTENCY 'ROW_SNAPSHOT'|'SEMANTIC_FRESH'`
  - [x] 17.7 Implement parsing for `CREATE VERSION TAG 'name' [FOR TRAINING [WITH TRAINING PRECISION 'sq8'|'rabitq'|'float32'] [TRAINING SEED n]]`
  - [x] 17.8 Implement parsing for `BULK INSERT`, `SHOW EMBEDDING HEALTH`, `BACKUP TO`, `RESTORE FROM`, `ANALYZE`
  - [x] 17.9 Implement descriptive parse error messages with byte offset position
  - [x] 17.10 Write tests: parse every AuroraSQL extension, error messages with positions, round-trip standard SQL

- [x] 18. Implement query planner and executor — galaxdb-sql (Reqs 14, 15, 22)
  - [x] 18.1 Define `QueryPlan` enum: PointLookup, FullScan, SemanticSearch, HybridSearch, Insert, Update, Delete, BulkInsert, CreateTable, DropTable, CreateVersionTag, Backup, Restore, Analyze, ShowEmbeddingHealth
  - [x] 18.2 Implement DDL executor: CREATE TABLE (allocate catalog entry, init ART index, create memtable, register embedding columns with sidecar), DROP TABLE (remove catalog, schedule file deletion)
  - [x] 18.3 Implement INSERT executor: write to memtable + WAL, update ART, trigger async embedding for embedding columns
  - [x] 18.4 Implement SELECT executor: ART point lookup or full scan with zone-map pruning + Bloom filter checks
  - [x] 18.5 Implement UPDATE executor: write new MVCC version; reject if target column is embedding source (return error with DELETE+INSERT suggestion)
  - [x] 18.6 Implement DELETE executor: write tombstone to memtable + WAL, write DELTA_TOMBSTONE for vector index
  - [x] 18.7 Implement BULK INSERT executor: bypass memtable, write sorted rows directly as PAX blocks
  - [x] 18.8 Implement adaptive query planner: estimate filter cardinality from statistics, choose BruteForceFiltered (< 1000 rows or < 0.1%) vs HnswWithPostFilter, log chosen strategy
  - [x] 18.9 Write tests: DDL create/drop, INSERT/SELECT/UPDATE/DELETE round-trip, UPDATE-of-embedding-source rejection, BULK INSERT, adaptive planner strategy selection

- [x] 19. Implement snapshot isolation — galaxdb-sql (Req 16)
  - [x] 19.1 Implement `TransactionManager` with monotonic timestamp assignment and active snapshot tracking
  - [x] 19.2 Implement snapshot read: filter MVCC versions where `commit_ts <= read_timestamp`
  - [x] 19.3 Implement write-write conflict detection: abort second writer on same key
  - [x] 19.4 Document write-skew limitation (SSI deferred to v2)
  - [x] 19.5 Write tests: no dirty reads, no non-repeatable reads, no phantoms, write-write conflict abort

- [x] 20. Implement PostgreSQL wire protocol — galaxdb-wire (Req 13)
  - [x] 20.1 Implement TCP listener with TLS 1.3 via rustls
  - [x] 20.2 Implement startup handshake: StartupMessage → AuthenticationOk, ParameterStatus, BackendKeyData, ReadyForQuery
  - [x] 20.3 Implement simple query protocol: parse Q message → route to SQL parser → execute → RowDescription + DataRow + CommandComplete + ReadyForQuery
  - [x] 20.4 Implement ErrorResponse with SQLSTATE codes for parse errors, execution errors, and connection errors
  - [x] 20.5 Implement connection as async tokio task with AtomicUsize connection counter
  - [x] 20.6 Implement max connection limit (default 1000): reject with SQLSTATE 53300 when exceeded
  - [x] 20.7 Write tests: startup handshake, simple query round-trip, error responses, connection limit enforcement

- [x] 21. Implement pg_catalog stubs — galaxdb-wire (Req 33)
  - [x] 21.1 Implement `pg_catalog.pg_class` (oid, relname, relnamespace, relkind) populated from catalog
  - [x] 21.2 Implement `pg_catalog.pg_attribute` (attrelid, attname, atttypid, attnum, attnotnull)
  - [x] 21.3 Implement `pg_catalog.pg_type` (oid, typname, typlen, typtype)
  - [x] 21.4 Implement `pg_catalog.pg_namespace` (oid, nspname) and `pg_catalog.pg_database` (oid, datname)
  - [x] 21.5 Implement fallback: queries against unsupported pg_catalog tables return empty result set (not error)
  - [x] 21.6 Write tests: psycopg2 connection handshake succeeds, SQLAlchemy table reflection works

- [x] 22. Implement Python client — galaxdb-python (Req 32)
  - [x] 22.1 Create galaxdb-python package with PyO3 bindings for embedded mode (`galaxdb.Database(path)`)
  - [ ] 22.2 Implement remote mode: connect via PostgreSQL wire protocol (`galaxdb.connect(connstring)`)
  - [x] 22.3 Implement `db.execute(sql)` returning list of row dicts
  - [ ] 22.4 Implement `db.training_dataset(tag)` returning PyTorch IterableDataset backed by Lance
  - [x] 22.5 Ensure Python 3.9+ compatibility
  - [ ] 22.6 Write tests: embedded mode CRUD, remote mode CRUD, training_dataset returns valid IterableDataset

## Month 3: Vector Index & Embedding Sidecar

- [x] 23. Implement HNSW base graph — galaxdb-vector (Req 17)
  - [x] 23.1 Implement HNSW graph construction algorithm (insert with layer selection, neighbor selection with heuristic pruning)
  - [x] 23.2 Implement mmap'd graph file format: metadata header (M, ef_construction, max_level, entry_point) + adjacency lists + quantized vector payloads
  - [x] 23.3 Implement graph search: greedy beam search from entry point through layers, ef parameter for search width
  - [x] 23.4 Implement cosine similarity distance computation with SIMD acceleration
  - [x] 23.5 Write tests: graph construction correctness, recall@10 >= 0.95 on SIFT-1M equivalent, mmap read-only access

- [x] 24. Implement delta buffer — galaxdb-vector (Req 17)
  - [x] 24.1 Implement `DeltaBuffer` with in-memory vector storage, tombstone set, and quantized copies
  - [x] 24.2 Implement WAL integration: DELTA_INSERT and DELTA_TOMBSTONE record types written to unified WAL
  - [x] 24.3 Implement exact brute-force k-NN search over delta buffer
  - [x] 24.4 Implement union + re-rank: combine HNSW candidates with delta buffer candidates, re-rank by exact cosine similarity from PAX blocks
  - [x] 24.5 Write tests: delta insert/search, tombstone exclusion, union+re-rank correctness

- [x] 25. Implement HNSW merge — galaxdb-vector (Req 17)
  - [x] 25.1 Implement merge trigger: `max(10_000, total_indexed * 0.01)` threshold check
  - [x] 25.2 Implement emergency merge trigger: tombstones > 20% of indexed vectors
  - [x] 25.3 Implement shadow file merge: build new HNSW graph in `.hnsw.new`, fsync, atomic `rename()` to `.hnsw`
  - [x] 25.4 Implement Arc-based reference counting for old graph: release when all in-flight queries complete
  - [x] 25.5 Implement crash recovery: replay WAL delta records in batches of 1000 to rebuild delta buffer
  - [x] 25.6 Write tests: merge produces correct graph, atomic rename crash safety, recovery rebuilds delta buffer

- [x] 26. Implement platform-aware quantization — galaxdb-vector (Req 18)
  - [x] 26.1 Define `Quantizer` trait: quantize, dequantize, distance (SIMD-accelerated), compression_ratio
  - [x] 26.2 Implement `Sq8Quantizer`: int8 scalar quantization with AVX2/AVX-512 SIMD distance kernels (4× compression)
  - [x] 26.3 Implement `Fp16Quantizer`: half-precision float with ARM NEON SIMD distance kernels (2× compression)
  - [x] 26.4 Implement `RabitqQuantizer` (opt-in): random orthogonal rotation matrix + binary quantization (32× compression)
  - [x] 26.5 Implement platform detection at startup: x86-64+AVX2 → SQ8, ARM64 → FP16, configurable override
  - [x] 26.6 Write tests: quantize/dequantize round-trip accuracy, distance computation correctness, platform detection

- [-] 27. Implement embedding sidecar binary — galaxdb-sidecar (Req 19)
  - [ ] 27.1 Create galaxdb-sidecar binary crate with `ort` (ONNX Runtime) dependency
  - [ ] 27.2 Implement ONNX model loading and session creation for sentence-transformer models
  - [ ] 27.3 Implement Unix socket server with length-prefixed JSON protocol (request: row_id + text, response: row_id + embedding + model_version)
  - [ ] 27.4 Implement parent PID monitoring: Linux `prctl(PR_SET_PDEATHSIG)`, macOS `kqueue` EVFILT_PROC
  - [ ] 27.5 Implement heartbeat: ping every 5 seconds, engine expects response within 2 seconds
  - [ ] 27.6 Write tests: model loading, embedding generation correctness, Unix socket communication, parent death detection

- [ ] 28. Implement sidecar manager in engine — galaxdb-sidecar (Req 19)
  - [ ] 28.1 Implement `SidecarManager`: spawn sidecar as child process, manage lifecycle
  - [ ] 28.2 Implement crash detection: 3 missed heartbeats → enter degraded mode
  - [ ] 28.3 Implement restart with exponential backoff: 1s, 2s, 4s, 8s, 16s, 32s, 60s cap
  - [ ] 28.4 Implement `EmbeddingManager` with in-flight semaphore (capacity 10,000)
  - [ ] 28.5 Implement backlog overflow: when semaphore full, write to `_galaxdb_embedding_backlog` system table with DURABILITY STRICT
  - [ ] 28.6 Implement backlog drain: background scanner drains FIFO when sidecar capacity recovers
  - [ ] 28.7 Write tests: sidecar spawn/kill/restart, backlog overflow and drain, degraded mode behavior

- [ ] 29. Implement model-version tracking — galaxdb-sidecar (Req 20)
  - [ ] 29.1 Store `_embedding_model_version` system column with each embedded row
  - [ ] 29.2 Implement model version change detection: compare sidecar-reported version with stored versions
  - [ ] 29.3 Implement stale marking: set `_embedding_stale = true` on rows with old model version, enqueue for re-embedding
  - [ ] 29.4 Implement `SHOW EMBEDDING HEALTH`: query catalog for version distribution and re-embedding progress
  - [ ] 29.5 Write tests: version tracking on insert, stale marking on model change, SHOW EMBEDDING HEALTH output

- [ ] 30. Implement embedding staleness tracking — galaxdb-sidecar (Req 39)
  - [ ] 30.1 Add `_embedding_stale: bool` system column to rows with embedding columns
  - [ ] 30.2 Set `_embedding_stale = true` on INSERT (before embedding generated), clear on embedding completion
  - [ ] 30.3 Set `_embedding_stale = true` on model version change for affected rows
  - [ ] 30.4 Write staleness flag through standard LSM update path (same WAL, same MVCC) for reader consistency
  - [ ] 30.5 Write tests: stale flag lifecycle (insert → stale → embedded → not stale), model change → stale

- [ ] 31. Implement SEMANTIC_MATCH execution — galaxdb-sql, galaxdb-vector (Req 21)
  - [ ] 31.1 Implement query-time embedding: send query text to sidecar, receive query vector
  - [ ] 31.2 Implement HNSW + delta buffer search with union + re-rank pipeline
  - [ ] 31.3 Implement similarity threshold filtering on results
  - [ ] 31.4 Implement sidecar-unavailable error: return "semantic search temporarily unavailable" when degraded
  - [ ] 31.5 Integrate adaptive planner: use statistics to choose BruteForceFiltered vs HnswWithPostFilter (Req 22)
  - [ ] 31.6 Write tests: SEMANTIC_MATCH returns correct results, threshold filtering, sidecar-down error, adaptive strategy selection


## Month 4: Versioning, Training, Hardening & Observability

- [ ] 32. Implement Merkle DAG versioning — galaxdb-versioning (Req 23)
  - [ ] 32.1 Implement `MerkleDag` struct: BTreeMap of commit_timestamp → MerkleRoot, with root_hash computed as XXH3-128 over child block hashes
  - [ ] 32.2 Implement version root creation on each commit: collect PAX block checksums, compute Merkle tree
  - [ ] 32.3 Implement `AT VERSION timestamp` query: filter PAX blocks where commit_timestamp <= target
  - [ ] 32.4 Implement `AT VERSION tag_name` query: resolve tag to MerkleRoot, return exact block set
  - [ ] 32.5 Implement semantic guardrail: AT VERSION + SEMANTIC_MATCH without consistency mode → reject with error message
  - [ ] 32.6 Implement CONSISTENCY 'SEMANTIC_FRESH': search current HNSW against historical rows, include warning in result metadata
  - [ ] 32.7 Implement CONSISTENCY 'SEMANTIC_SNAPSHOT' rejection: return error "v2 feature"
  - [ ] 32.8 Write tests: version root computation, AT VERSION query correctness, semantic guardrail rejection, SEMANTIC_FRESH warning

- [ ] 33. Implement version tags — galaxdb-versioning (Req 24)
  - [ ] 33.1 Implement `CREATE VERSION TAG 'name'`: capture current MerkleRoot, mark referenced blocks as GC-exempt (pinned)
  - [ ] 33.2 Implement `FOR TRAINING` tag: store TrainingTagMetadata (precision, seed, deterministic_order=true with primary key sort)
  - [ ] 33.3 Implement `WITH TRAINING PRECISION 'sq8'|'rabitq'|'float32'` storage in tag metadata
  - [ ] 33.4 Implement `TRAINING SEED n` storage in tag metadata
  - [ ] 33.5 Integrate with compactor: pinned blocks are never GC'd regardless of MVCC age
  - [ ] 33.6 Implement `_galaxdb_versions` system table for tag catalog
  - [ ] 33.7 Write tests: tag creation, GC exemption, FOR TRAINING metadata, compactor respects pins

- [ ] 34. Implement Lance training export — galaxdb-versioning (Req 25)
  - [ ] 34.1 Add `lance` crate dependency, implement `LanceExporter` struct
  - [ ] 34.2 Implement export pipeline: read blocks for tagged version → sort by primary key → convert to Arrow batches → write Lance dataset
  - [ ] 34.3 Implement training precision conversion: float32 passthrough, sq8 quantization, rabitq quantization during export
  - [ ] 34.4 Implement dedup integration: apply `WHERE NOT DUPLICATE` filter during export if dedup flag set
  - [ ] 34.5 Implement lineage recording: insert record into `_galaxdb_training_exports` on each export (Req 38)
  - [ ] 34.6 Write tests: Lance export produces valid dataset, precision conversion correctness, dedup filtering, lineage record created

- [ ] 35. Implement MinHash near-duplicate detection — galaxdb-versioning (Req 26)
  - [ ] 35.1 Implement `MinHashDedup`: 128 independent hash functions over character n-grams, 512-byte signature per row
  - [ ] 35.2 Integrate into write path: compute MinHash signature on INSERT for TEXT columns, store as `_minhash_signature` system column
  - [ ] 35.3 Implement Jaccard similarity estimation from signatures
  - [ ] 35.4 Implement background refresh job: group rows with Jaccard > 0.8, populate `_near_duplicate_group` column
  - [ ] 35.5 Implement `WHERE NOT DUPLICATE` query filter: exclude rows in near-duplicate groups (keep one representative)
  - [ ] 35.6 Write tests: signature computation, Jaccard estimation accuracy, duplicate grouping, WHERE NOT DUPLICATE filtering

- [ ] 36. Implement training data lineage — galaxdb-versioning (Req 38)
  - [ ] 36.1 Create `_galaxdb_training_exports` system table: tag_name, filter_expr, precision, dedup, curriculum, row_count, exported_at, content_hash
  - [ ] 36.2 Make table append-only: reject DELETE/UPDATE queries against this table
  - [ ] 36.3 Insert lineage record on every training export
  - [ ] 36.4 Write tests: lineage record creation, append-only enforcement, content hash correctness

- [ ] 37. Implement backup and restore — galaxdb-versioning (Req 27)
  - [ ] 37.1 Implement `BACKUP TO '/path'`: acquire write-quiesce (< 100 ms), flush memtable, create clean Merkle root
  - [ ] 37.2 Implement concurrent backup copy: reads continue during quiesce, writes resume after copy begins
  - [ ] 37.3 Implement file copy: PAX blocks + WAL + blob log files to target path
  - [ ] 37.4 Implement `RESTORE FROM '/path'`: validate all block checksums, copy files, replay WAL, rebuild ART index, rebuild HNSW graph
  - [ ] 37.5 Implement restore abort on checksum failure: report corrupted block and stop
  - [ ] 37.6 Write tests: backup/restore round-trip, write-quiesce < 100 ms, reads during backup, checksum failure abort

- [ ] 38. Implement observability — galaxdb-observe (Req 28)
  - [ ] 38.1 Implement embedded HTTP server (axum) with `/health` endpoint returning JSON status
  - [ ] 38.2 Implement `/metrics` endpoint with Prometheus text exposition format
  - [ ] 38.3 Register all metrics: buffer_pool_hot_set_usage, buffer_pool_scan_buffer_usage, embedding_queue_depth, embedding_backlog_depth, checkpoint_last_duration_ms, compaction_pending_bytes, wal_write_latency_us, hnsw_recall_estimate, connections_active, disk_full, sidecar_status
  - [ ] 38.4 Implement structured JSON logging via tracing-subscriber with configurable level (GALAXDB_LOG_LEVEL env var)
  - [ ] 38.5 Implement OpenTelemetry W3C traceparent propagation: root span per query, child spans for SQL parse, plan, HNSW search, delta search, sidecar call, PAX reads
  - [ ] 38.6 Implement SQL commenter format for trace context in wire protocol
  - [ ] 38.7 Write tests: /health returns correct status, /metrics returns valid Prometheus format, trace spans created for query execution

- [ ] 39. Implement vLSM structural improvements (Month 4 hardening) — galaxdb-storage (Req 36)
  - [ ] 39.1 Make SST size configurable, change default from 64 MB to 8 MB
  - [ ] 39.2 Switch L0 from tiered to leveled compaction
  - [ ] 39.3 Implement SILK-style flush pre-emption: prioritize flush I/O over compaction I/O when memtable back-pressure is high
  - [ ] 39.4 Write tests: smaller SSTs reduce write stalls, L0 leveled compaction correctness, flush pre-emption under load

- [ ] 40. Implement deployment modes — galaxdb-server, galaxdb-embedded (Req 35)
  - [ ] 40.1 Implement standalone server binary: tokio main, bind wire protocol + HTTP observability, spawn sidecar, graceful shutdown on SIGTERM/SIGINT
  - [ ] 40.2 Implement embedded mode: PyO3 `Database` class with `new(path)`, `execute(sql)`, `training_dataset(tag)` methods
  - [ ] 40.3 Implement platform-specific I/O backend selection: io_uring on Linux, tokio on macOS/Windows
  - [ ] 40.4 Verify binary size: core < 70 MB, full < 350 MB (with sidecar + model)
  - [ ] 40.5 Write tests: server starts and accepts connections, embedded mode CRUD from Python, binary size check

- [ ] 41. Implement chaos tests — tests/chaos (Req 37)
  - [ ] 41.1 Implement test harness: set up populated database with known data, inject faults, verify recovery
  - [ ] 41.2 Test: kill sidecar mid-request → engine recovers sidecar, drains backlog, no data loss
  - [ ] 41.3 Test: kill engine mid-flush → recovery produces consistent state, no committed data lost
  - [ ] 41.4 Test: corrupt WAL records → recovery skips corrupt records, recovers all valid data
  - [ ] 41.5 Test: fill disk → engine performs clean checkpoint, blocks writes, no data corruption
  - [ ] 41.6 Test: all recovery scenarios complete in < 30 seconds
  - [ ] 41.7 Write integration test suite that runs all chaos scenarios in CI

- [ ] 42. End-to-end integration tests
  - [ ] 42.1 Test: psycopg2 connects, creates table with embedding column, inserts rows, queries with SEMANTIC_MATCH
  - [ ] 42.2 Test: SQLAlchemy connects, reflects table metadata via pg_catalog stubs
  - [ ] 42.3 Test: CREATE VERSION TAG FOR TRAINING → export Lance dataset → verify PyTorch IterableDataset
  - [ ] 42.4 Test: AT VERSION query with ROW_SNAPSHOT (no SEMANTIC_MATCH), SEMANTIC_FRESH (with warning)
  - [ ] 42.5 Test: SHOW EMBEDDING HEALTH returns correct model version distribution
  - [ ] 42.6 Test: WHERE NOT DUPLICATE filters near-duplicates in training export
  - [ ] 42.7 Test: BACKUP TO / RESTORE FROM round-trip with data verification
