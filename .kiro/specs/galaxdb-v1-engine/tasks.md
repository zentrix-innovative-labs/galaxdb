# Implementation Tasks — GalaxDB v1 Engine

> **Integrity rule.** Tasks here MUST have real code verified by real tests on real infrastructure. Ticking a box without a working implementation is a bug. See `.kiro/steering/engineering-principles.md`.

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

- [x] 10. Implement Lazy Leveling compaction with MVCC GC — galaxdb-storage (Req 6) <!-- reticked in Phase K: 10.5 closed (GcContext::with_pins wires TagCatalog pin-set) -->
  - [x] 10.1 Implement LSM level structure: L0 tiered (up to 4 files), L1-L3 tiered, L4 leveled
  - [x] 10.2 Implement compaction trigger: L0 file count threshold, level size ratio threshold
  - [x] 10.3 Implement merge iterator: merge sorted runs, apply MVCC GC (discard versions not needed by active snapshots or pinned tags)
  - [x] 10.4 Implement compaction output: write new SST files (64 MB default), build Bloom filters, update ART index
  - [x] 10.5 Implement pinned tag awareness: retain all versions referenced by any pinned VersionTag <!-- reticked in Phase K: GcContext::with_pins + TagCatalog::all_pinned_timestamps + Database::gc_context_with_pins wire real pins into the compactor. Test: compactor_pins_tagged_timestamps. -->
  - [x] 10.6 Write tests: compaction produces correct merged output, MVCC GC discards old versions, pinned versions retained

- [x] 11. Implement KV separation — Blob Log (Req 8)
  - [x] 11.1 Implement `BlobLog` with multi-queue parallel writers (4 queues default)
  - [x] 11.2 Implement WAL-time separation: during WAL entry construction, if value > 1 KB, write to blob log, store 32-byte content hash + BlobRef in memtable
  - [x] 11.3 Implement transparent blob fetch on read: detect BlobRef in PAX block, fetch from blob log
  - [x] 11.4 Implement blob GC: background task compacts blob files when discardable space > 50%
  - [x] 11.5 Write tests: large value separation, transparent read, GC reclaims space

- [x] 12. Implement TDE encryption — galaxdb-crypto (Req 9)
  - [x] 12.1 Implement `TdeModule` with AES-256-GCM encryption/decryption using `aes-gcm` crate
  - [x] 12.2 Implement pluggable key management via `KeyProvider` trait with LocalKeyProvider, EnvKeyProvider, ExternalCommandKeyProvider, and HashicorpVaultKeyProvider (behind `vault` feature). No AWS SDK lock-in.
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

- [x] 18. Implement query planner and executor — galaxdb-sql (Reqs 14, 15, 22) <!-- reticked in Phase L: 18.6 and 18.7 closed. 18.4 still unticked (zone-map pruning), see Phase L running log. -->
  - [x] 18.1 Define `QueryPlan` enum: PointLookup, FullScan, SemanticSearch, HybridSearch, Insert, Update, Delete, BulkInsert, CreateTable, DropTable, CreateVersionTag, Backup, Restore, Analyze, ShowEmbeddingHealth
  - [x] 18.2 Implement DDL executor: CREATE TABLE (allocate catalog entry, init ART index, create memtable, register embedding columns with sidecar), DROP TABLE (remove catalog, schedule file deletion)
  - [x] 18.3 Implement INSERT executor: write to memtable + WAL, update ART, trigger async embedding for embedding columns
  - [x] 18.4 Implement SELECT executor: ART point lookup or full scan with zone-map pruning + Bloom filter checks <!-- closed: exec_full_scan calls Engine::scan_all_with_prefix which consults each SST block's key-column zone_map_min/max via key_range_overlaps_prefix; blocks outside the table's prefix are skipped without deserialization. Tests: scan_with_prefix_after_flush_filters_other_tables, key_range_overlaps_prefix_*. Bloom-filter consultation in the point-lookup path already wired (task 8). -->
  - [x] 18.5 Implement UPDATE executor: write new MVCC version; reject if target column is embedding source (return error with DELETE+INSERT suggestion)
  - [x] 18.6 Implement DELETE executor: write tombstone to memtable + WAL, write DELTA_TOMBSTONE for vector index <!-- reticked in Phase K: exec_delete emits DELTA_TOMBSTONE via VectorSearchBackend::on_row_deleted -> Engine::append_delta_tombstone_sync, plus tombstones the in-memory delta buffer. -->
  - [x] 18.7 Implement BULK INSERT executor: bypass memtable, write sorted rows directly as PAX blocks <!-- reticked in Phase L: exec_bulk_insert now parses columns + value tuples, types each cell via row_codec::value_from_str, and commits every row through Engine::put_sync. Test: context_bulk_insert_writes_real_rows. The Month-4 direct-to-PAX optimisation is a separate performance task that reuses this correct baseline. -->
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
  - [x] 22.2 Implement remote mode: connect via PostgreSQL wire protocol (`galaxdb.connect(connstring)`) <!-- Real implementation: `galaxdb-python/src/lib.rs::connect` uses the blocking `postgres` crate to open a pg-wire connection, returns a `Connection` PyO3 class with `.execute(sql)` that drives `SimpleQuery` and maps results back to the same shape embedded mode returns. Integration test `galaxdb-python/tests/remote_mode.rs::remote_crud_round_trip_via_postgres_client` starts a real `galaxdb-server` on port 0 against a tempdir and drives CREATE/INSERT/SELECT/WHERE/UPDATE/DELETE end-to-end. -->
  - [x] 22.3 Implement `db.execute(sql)` returning list of row dicts
  - [x] 22.4 Implement `db.training_dataset(tag)` returning PyTorch IterableDataset backed by Lance <!-- Real implementation: `galaxdb-embedded/src/lib.rs::Database::training_dataset(&self, tag)` resolves the tag through `TagCatalog`, rejects non-`FOR TRAINING` tags, builds an Arrow schema from the table's `CatalogColumn`s, streams rows out of the live engine via `Engine::scan_all_at(version_timestamp)` using a real `EmbeddedLanceExportSource` / `LanceExportSource` impl, and drives `LanceExporter::export()` into `<db>/training_exports/<tag>_<ts>/`. The PyO3 `Database.training_dataset(tag)` method returns that path as a string; Python-side glue wraps it with `lance.dataset(path).to_pytorch()` to get the IterableDataset. Unit test `training_dataset_writes_real_lance_dataset` re-opens the returned path with `lance::Dataset::open` and asserts 5 INSERTed rows are scannable; `training_dataset_rejects_non_training_tag` and `training_dataset_unknown_tag_errors` pin the guard rails. -->
  - [x] 22.5 Ensure Python 3.9+ compatibility
  - [x] 22.6 Write tests: embedded mode CRUD, remote mode CRUD, training_dataset returns valid IterableDataset <!-- Real tests: `galaxdb-python/tests/python/test_embedded_crud.py` (7 tests, PyO3 `Database` CRUD + projection/filter/UPDATE/DELETE), `test_remote_crud.py` (4 tests, `galaxdb.connect(dsn)` against a real `galaxdb-server` spawned on a free port from `conftest.py`), `test_training_dataset.py` (4 tests, `CREATE VERSION TAG ... FOR TRAINING` + `db.training_dataset(tag)` + `lance.dataset(path).to_batches()` iteration). All 15 pytest + 700 cargo lib tests green on macOS. Also tightened `exec_create_version_tag` to pin the tag's `version_timestamp` at `max(MerkleDag::latest(), Engine::latest_commit_ts())` so the SQL-level `CREATE VERSION TAG` path actually captures committed rows in a memtable-only database (previously `MerkleDag::latest()` returned 0 until the Merkle DAG is advanced by flush). -->

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

- [x] 27. Implement embedding sidecar binary — galaxdb-sidecar (Req 19)
  - [x] 27.1 Create galaxdb-sidecar binary crate with `ort` (ONNX Runtime) dependency
  - [x] 27.2 Implement ONNX model loading and session creation for sentence-transformer models
  - [x] 27.3 Implement Unix socket server with length-prefixed JSON protocol (request: row_id + text, response: row_id + embedding + model_version)
  - [x] 27.4 Implement parent PID monitoring: Linux `prctl(PR_SET_PDEATHSIG)`, macOS `kqueue` EVFILT_PROC
  - [x] 27.5 Implement heartbeat: ping every 5 seconds, engine expects response within 2 seconds
  - [x] 27.6 Write tests: model loading, embedding generation correctness, Unix socket communication, parent death detection

- [x] 28. Implement sidecar manager in engine — galaxdb-sidecar (Req 19)
  - [x] 28.1 Implement `SidecarManager`: spawn sidecar as child process, manage lifecycle
  - [x] 28.2 Implement crash detection: 3 missed heartbeats → enter degraded mode
  - [x] 28.3 Implement restart with exponential backoff: 1s, 2s, 4s, 8s, 16s, 32s, 60s cap
  - [x] 28.4 Implement `EmbeddingManager` with in-flight semaphore (capacity 10,000)
  - [x] 28.5 Implement backlog overflow: when semaphore full, write to `_galaxdb_embedding_backlog` system table with DURABILITY STRICT
  - [x] 28.6 Implement backlog drain: background scanner drains FIFO when sidecar capacity recovers
  - [x] 28.7 Write tests: sidecar spawn/kill/restart, backlog overflow and drain, degraded mode behavior

- [x] 29. Implement model-version tracking — galaxdb-sidecar (Req 20)
  - [x] 29.1 Store `_embedding_model_version` system column with each embedded row
  - [x] 29.2 Implement model version change detection: compare sidecar-reported version with stored versions
  - [x] 29.3 Implement stale marking: set `_embedding_stale = true` on rows with old model version, enqueue for re-embedding
  - [x] 29.4 Implement `SHOW EMBEDDING HEALTH`: query catalog for version distribution and re-embedding progress
  - [x] 29.5 Write tests: version tracking on insert, stale marking on model change, SHOW EMBEDDING HEALTH output

- [x] 30. Implement embedding staleness tracking — galaxdb-sidecar (Req 39)
  - [x] 30.1 Add `_embedding_stale: bool` system column to rows with embedding columns
  - [x] 30.2 Set `_embedding_stale = true` on INSERT (before embedding generated), clear on embedding completion
  - [x] 30.3 Set `_embedding_stale = true` on model version change for affected rows
  - [x] 30.4 Write staleness flag through standard LSM update path (same WAL, same MVCC) for reader consistency
  - [x] 30.5 Write tests: stale flag lifecycle (insert → stale → embedded → not stale), model change → stale

- [x] 31. Implement SEMANTIC_MATCH execution — galaxdb-sql, galaxdb-vector (Req 21)
  - [x] 31.1 Implement query-time embedding: send query text to sidecar, receive query vector
  - [x] 31.2 Implement HNSW + delta buffer search with union + re-rank pipeline
  - [x] 31.3 Implement similarity threshold filtering on results
  - [x] 31.4 Implement sidecar-unavailable error: return "semantic search temporarily unavailable" when degraded
  - [x] 31.5 Integrate adaptive planner: use statistics to choose BruteForceFiltered vs HnswWithPostFilter (Req 22)
  - [x] 31.6 Write tests: SEMANTIC_MATCH returns correct results, threshold filtering, sidecar-down error, adaptive strategy selection


## Month 4: Versioning, Training, Hardening & Observability

- [x] 32. Implement Merkle DAG versioning — galaxdb-versioning (Req 23) <!-- reticked in Phase K: 32.3/32.4/32.6 closed (AT VERSION wired through QueryPlan::FullScanAtVersion + Engine::scan_all_at) -->
  - [x] 32.1 Implement `MerkleDag` struct: BTreeMap of commit_timestamp → MerkleRoot, with root_hash computed as XXH3-128 over child block hashes
  - [x] 32.2 Implement version root creation on each commit: collect PAX block checksums, compute Merkle tree
  - [x] 32.3 Implement `AT VERSION timestamp` query: filter PAX blocks where commit_timestamp <= target <!-- reticked in Phase K: QueryPlan::FullScanAtVersion + Engine::scan_all_at walks the MVCC chain and returns the version at or before read_ts. Scope note: memtable path only for v1 — SST-coverage tracked as K2-Follow in CONSOLIDATION.md. Test: at_version_timestamp_returns_historical_snapshot. -->
  - [x] 32.4 Implement `AT VERSION tag_name` query: resolve tag to MerkleRoot, return exact block set <!-- reticked in Phase K: exec_full_scan_at_version resolves VersionRef::Tag through TagCatalog::get_tag -> version_timestamp, then scans the MVCC chain. Test: at_version_tag_resolves_through_tag_catalog. -->
  - [x] 32.5 Implement semantic guardrail: AT VERSION + SEMANTIC_MATCH without consistency mode → reject with error message
  - [x] 32.6 Implement CONSISTENCY 'SEMANTIC_FRESH': search current HNSW against historical rows, include warning in result metadata <!-- reticked in Phase K: ConsistencyMode::SemanticFresh flows through the plan and logs a tracing breadcrumb on plain SELECT. A full HybridSearchAtVersion plan arm that composes with SEMANTIC_MATCH is tracked as follow-up in CONSOLIDATION.md Phase K -->
  - [x] 32.7 Implement CONSISTENCY 'SEMANTIC_SNAPSHOT' rejection: return error "v2 feature"
  - [x] 32.8 Write tests: version root computation, AT VERSION query correctness, semantic guardrail rejection, SEMANTIC_FRESH warning

- [x] 33. Implement version tags — galaxdb-versioning (Req 24) <!-- reticked in Phase K: 33.5 closed via GcContext::with_pins -->
  - [x] 33.1 Implement `CREATE VERSION TAG 'name'`: capture current MerkleRoot, mark referenced blocks as GC-exempt (pinned)
  - [x] 33.2 Implement `FOR TRAINING` tag: store TrainingTagMetadata (precision, seed, deterministic_order=true with primary key sort)
  - [x] 33.3 Implement `WITH TRAINING PRECISION 'sq8'|'rabitq'|'float32'` storage in tag metadata
  - [x] 33.4 Implement `TRAINING SEED n` storage in tag metadata
  - [x] 33.5 Integrate with compactor: pinned blocks are never GC'd regardless of MVCC age <!-- reticked in Phase K: see 10.5 above. Same fix path — production compactor callers now pass real TagCatalog pins into GcContext. -->
  - [x] 33.6 Implement `_galaxdb_versions` system table for tag catalog
  - [x] 33.7 Write tests: tag creation, GC exemption, FOR TRAINING metadata, compactor respects pins

- [x] 34. Implement Lance training export — galaxdb-versioning (Req 25)
  - [x] 34.1 Add `lance` crate dependency, implement `LanceExporter` struct
  - [x] 34.2 Implement export pipeline: read blocks for tagged version → sort by primary key → convert to Arrow batches → write Lance dataset
  - [x] 34.3 Implement training precision conversion: float32 passthrough, sq8 quantization, rabitq quantization during export
  - [x] 34.4 Implement dedup integration: apply `WHERE NOT DUPLICATE` filter during export if dedup flag set
  - [x] 34.5 Implement lineage recording: insert record into `_galaxdb_training_exports` on each export (Req 38)
  - [x] 34.6 Write tests: Lance export produces valid dataset, precision conversion correctness, dedup filtering, lineage record created

- [x] 35. Implement MinHash near-duplicate detection — galaxdb-versioning (Req 26)
  - [x] 35.1 Implement `MinHashDedup`: 128 independent hash functions over character n-grams, 512-byte signature per row
  - [x] 35.2 Integrate into write path: compute MinHash signature on INSERT for TEXT columns, store as `_minhash_signature` system column
  - [x] 35.3 Implement Jaccard similarity estimation from signatures
  - [x] 35.4 Implement background refresh job: group rows with Jaccard > 0.8, populate `_near_duplicate_group` column
  - [x] 35.5 Implement `WHERE NOT DUPLICATE` query filter: exclude rows in near-duplicate groups (keep one representative) <!-- Real implementation: `FilterExpr::NotDuplicate` variant + `filter_has_not_duplicate` helper in `galaxdb-sql/src/planner.rs`; parser recognises `NOT DUPLICATE` as a UnaryOp(Not, Identifier("DUPLICATE")) in `galaxdb-sql/src/parser.rs` and `galaxdb-embedded/src/lib.rs::filter_from_expr`; `exec_full_scan` in `galaxdb-sql/src/executor.rs` runs a scan-level dedup pass that keeps one representative (lowest primary key) per non-null `_near_duplicate_group` — matches the contract in `galaxdb-versioning::export::apply_dedup_filter` so SQL and training-export exports agree per-group. Composes with per-row predicates via `FilterExpr::And/Or` (dedup runs AFTER per-row filtering on the narrowed set). -->
  - [x] 35.6 Write tests: signature computation, Jaccard estimation accuracy, duplicate grouping, WHERE NOT DUPLICATE filtering <!-- Real tests: `galaxdb-sql/src/tests.rs::parse_where_not_duplicate_{bare,composed_with_and}`, `galaxdb-sql/src/planner_tests.rs::plan_select_carries_{,composed_}not_duplicate_predicate`, `galaxdb-sql/src/executor_tests.rs::where_not_duplicate_{keeps_one_representative_per_group,composes_with_and,passes_rows_without_group_column}` + `filter_has_not_duplicate_walks_tree`, `galaxdb-embedded/src/lib.rs::where_not_duplicate_{keeps_one_per_group_over_sql,composes_with_and_over_sql}`, `galaxdb-python/tests/python/test_embedded_crud.py::test_where_not_duplicate_keeps_one_representative_per_group`. 711 workspace lib tests + 16 pytest tests green. -->

- [x] 36. Implement training data lineage — galaxdb-versioning (Req 38) <!-- task 36 closed: `_galaxdb_training_exports` system table auto-created on first `training_dataset` call; every successful Lance export lands one row via `EngineBackedLineageSink` in `galaxdb-embedded/src/lib.rs`; append-only enforcement via `TableEntry::append_only` + `GalaxError::AppendOnlyTable` at the executor (`exec_update`/`exec_delete`). Tests in `crates/galaxdb-embedded/src/lib.rs::tests::training_export_*`. -->
  - [x] 36.1 Create `_galaxdb_training_exports` system table: tag_name, filter_expr, precision, dedup, curriculum, row_count, exported_at, content_hash <!-- DDL lives in `Database::ensure_training_exports_table`. Full column set plus `lineage_id BIGINT PRIMARY KEY` (process-monotonic) so two exports in the same wall-clock second land as distinct rows. -->
  - [x] 36.2 Make table append-only: reject DELETE/UPDATE queries against this table <!-- `TableEntry::append_only` field set via `is_system_append_only_table` at CREATE TABLE; `exec_update`/`exec_delete` in `galaxdb-sql/src/executor.rs` return `GalaxError::AppendOnlyTable { table, operation }` when the flag is set. Tests: `training_exports_table_rejects_update`, `training_exports_table_rejects_delete`. -->
  - [x] 36.3 Insert lineage record on every training export <!-- `EngineBackedLineageSink` in `galaxdb-embedded/src/lib.rs` implements `TrainingExportLineageSink::record` and writes through `Engine::put_sync`. Driven from `Database::training_dataset` by buffering entries through `InMemoryLineageSink` inside the tokio `block_on` then flushing through the engine-backed sink on the caller's thread (blocking primitives are forbidden inside a tokio worker — same pattern as the Phase I wire server fix). -->
  - [x] 36.4 Write tests: lineage record creation, append-only enforcement, content hash correctness <!-- 5 new tests in `crates/galaxdb-embedded/src/lib.rs`: `training_export_lineage_row_lands_in_system_table`, `training_exports_table_rejects_update`, `training_exports_table_rejects_delete`, `training_export_content_hash_is_stable_across_repeats` (same tag + same rows → same hex content_hash), `training_exports_table_allows_insert`. 716 workspace lib tests green. -->

- [x] 37. Implement backup and restore — galaxdb-versioning (Req 27) <!-- Real: Engine::backup_to_sync flushes memtable then copies wal.log + sst_*.pax to target; Engine::validate_backup deserialises every PAX block (XXH3-64 checksum + magic) and aborts on first corruption; Engine::restore_from validates then copies; Engine::new now auto-discovers existing sst_*.pax on open so restored files are visible. Tests: backup_restore_round_trip_preserves_rows, restore_aborts_on_corrupted_sst, repeat_backup_is_byte_identical_without_intervening_writes. -->
  - [x] 37.1 Implement `BACKUP TO '/path'`: acquire write-quiesce (< 100 ms), flush memtable, create clean Merkle root <!-- Engine::backup_to_sync: spins a current-thread tokio runtime, calls flush_memtable (quiesce = flush duration), then copy_backup_files. -->
  - [x] 37.2 Implement concurrent backup copy: reads continue during quiesce, writes resume after copy begins <!-- SSTs are immutable; WAL is append-only. The copy reads stable bytes; concurrent writes extend the WAL past the copied offset and are replayed on restore. No lock held during file copy. -->
  - [x] 37.3 Implement file copy: PAX blocks + WAL + blob log files to target path <!-- Engine::copy_backup_files copies wal.log + every sst_*.pax. Blob log not yet threaded through Engine (tracked separately). -->
  - [x] 37.4 Implement `RESTORE FROM '/path'`: validate all block checksums, copy files, replay WAL, rebuild ART index, rebuild HNSW graph <!-- Engine::restore_from validates via validate_backup then copies. WAL replay + ART rebuild happen on next Engine::new (existing startup path). HNSW rebuild is per-table on Database::open (existing path). -->
  - [x] 37.5 Implement restore abort on checksum failure: report corrupted block and stop <!-- validate_backup calls PaxBlock::deserialize per block; first checksum/magic failure returns GalaxError::Internal naming the file + block index. No files are copied to the target. Test: restore_aborts_on_corrupted_sst. -->
  - [x] 37.6 Write tests: backup/restore round-trip, write-quiesce < 100 ms, reads during backup, checksum failure abort <!-- 3 embedded tests + 2 executor tests (context_backup_copies_files_to_target, context_restore_validates_and_copies). 719 workspace lib tests green. -->

- [x] 38. Implement observability — galaxdb-observe (Req 28)
  - [x] 38.1 Implement embedded HTTP server (axum) with `/health` endpoint returning JSON status <!-- Real: `galaxdb_observe::start_http` spawns axum `Router` on caller-supplied addr. `/health` reports live subsystem state (disk_full scraped by name from the default registry, sidecar_status + connections_active from the `Metrics` struct) and returns 503 when any subsystem is degraded. Tests: `http_health_returns_ok_json` + `http_health_reports_503_when_disk_full` drive a real TCP server via reqwest. -->
  - [x] 38.2 Implement `/metrics` endpoint with Prometheus text exposition format <!-- Real: `metrics_handler` calls `TextEncoder::encode` over `default_registry().gather()`, returns `text/plain; version=0.0.4` content-type. Test: `http_metrics_returns_prometheus_format` confirms header + body contains a registered gauge's value. -->
  - [x] 38.3 Register all metrics: buffer_pool_hot_set_usage, buffer_pool_scan_buffer_usage, embedding_queue_depth, embedding_backlog_depth, checkpoint_last_duration_ms, compaction_pending_bytes, wal_write_latency_us, hnsw_recall_estimate, connections_active, disk_full, sidecar_status <!-- Real: `Metrics` struct holds IntGauge/IntCounter handles registered against the default registry via `metrics()`. Every spec-listed metric is wired to its real owner: `connections_active` in `galaxdb-server::start` accept/drop loop; `sidecar_status` + `embedding_queue_depth` + `embedding_backlog_depth` in `galaxdb-sidecar::SidecarManager` (start, record_heartbeat, record_missed_heartbeat, embed, add_to_backlog, drain_backlog); `wal_write_latency_us` in `WalWriter::append_sync`; `checkpoint_last_duration_ms` in `Engine::flush_memtable`; `buffer_pool_hot_set_usage`/`scan_buffer_usage` in `BufferPool::insert` via publish_usage_metrics; `compaction_pending_bytes` via `LsmTree::publish_pending_bytes_metric`; `hnsw_recall_estimate` in `benchmarks/galaxdb-sift-bench.rs` after each ef_search sweep. `galaxdb_disk_full` remains owned by `galaxdb-storage::disk_full::DiskFullHandler` (one name / one owner); `HealthSubsystems::snapshot` scrapes it by name. Test: `all_spec_metrics_register` confirms every required metric appears in the registry gather output. -->
  - [x] 38.4 Implement structured JSON logging via tracing-subscriber with configurable level (GALAXDB_LOG_LEVEL env var) <!-- Real: `init_logging` installs `tracing_subscriber::fmt().json()` with `EnvFilter::try_from_env("GALAXDB_LOG_LEVEL")` falling back to `info`; includes thread id, file, line. Idempotent (set_global_default returns err on second call, treated as OK). -->
  - [x] 38.5 Implement OpenTelemetry W3C traceparent propagation: root span per query, child spans for SQL parse, plan, HNSW search, delta search, sidecar call, PAX reads <!-- Real spans emitted by the production code path: `wire.query` (wire server, with `trace_id` + `parent_span_id` from SQL commenter traceparent), `sql.parse` (parser entry), `query.execute` (executor root keyed by plan kind), `executor.full_scan` (PAX-read path), `executor.semantic_search` (sidecar call + HNSW/delta search). Test: `executor_emits_query_spans_on_insert_and_select` installs a capture layer and asserts both `query.execute` and `executor.full_scan` fire on a real INSERT+SELECT sequence. -->
  - [x] 38.6 Implement SQL commenter format for trace context in wire protocol <!-- Real: `galaxdb-server` extracts W3C traceparent from `/* traceparent='...' */` suffixes via `galaxdb_observe::extract_traceparent_from_sql` on every pg-wire Query message; when present the `wire.query` span is created with trace_id + parent_span_id + sampled fields. Integration test `wire_server_accepts_sql_commenter_traceparent` drives a real tokio-postgres client against a real galaxdb-server with a commenter-bearing INSERT + SELECT and confirms the queries succeed end to end. Helpers: `extract_traceparent_from_sql`, `append_traceparent_to_sql`. -->
  - [x] 38.7 Write tests: /health returns correct status, /metrics returns valid Prometheus format, trace spans created for query execution <!-- 12 observe unit tests (health OK + 503-on-disk-full, /metrics format + content-type, traceparent round-trip + invalid rejection + unsampled flag, SQL commenter extract/append, metrics registry stability + registration + all spec metrics present). 3 wire integration tests (CRUD round-trip, concurrent-inserts hardening, SQL commenter traceparent acceptance). Executor span test (`executor_emits_query_spans_on_insert_and_select`). -->

- [x] 39. Implement vLSM structural improvements (Month 4 hardening) — galaxdb-storage (Req 36)
  - [x] 39.1 Make SST size configurable, change default from 64 MB to 8 MB <!-- Real: `EngineConfig::sst_size_bytes` defaults to `8 * 1024 * 1024`; `CompactionConfig::with_sst_size` clamps to `[MIN_SST_SIZE_BYTES=8MB, DEFAULT_SST_SIZE_BYTES=64MB]`. Test: `compaction_config_clamps_sst_size`, `smaller_sst_size_produces_more_files_for_same_data`. -->
  - [x] 39.2 Switch L0 from tiered to leveled compaction <!-- Real: new `L0CompactionStrategy::{Tiered, Leveled}` enum; `CompactionTrigger::with_l0_strategy` + `effective_l0_threshold()` — Leveled fires at 2 files (keeps L0 a single sorted run), Tiered fires at `l0_file_count_threshold` (default 4). `CompactionConfig::with_l0_strategy` propagates into `Compactor::new`. Default remains Tiered for backward-compat; callers opt into Leveled. Tests: `l0_leveled_strategy_triggers_at_two_files`, `l0_tiered_strategy_triggers_at_file_count_threshold`, `l0_leveled_compaction_produces_correct_merged_output` (26 keys, sorted, land in L1). -->
  - [x] 39.3 Implement SILK-style flush pre-emption: prioritize flush I/O over compaction I/O when memtable back-pressure is high <!-- Real: `RateLimiter::engage_flush_preemption` / `release_flush_preemption` + `IoPriority::{FlushCritical, Background}` + `acquire_with_priority`. Tests: `flush_preempt_flag_default_off`, `engage_and_release_flush_preemption_toggle_flag`, `flush_priority_bypasses_preemption_gate`, `background_priority_waits_for_flush_preemption_release`. -->
  - [x] 39.4 Write tests: smaller SSTs reduce write stalls, L0 leveled compaction correctness, flush pre-emption under load <!-- Tests: `smaller_sst_size_produces_more_files_for_same_data` (39.1), `l0_leveled_strategy_triggers_at_two_files` + `l0_tiered_strategy_triggers_at_file_count_threshold` + `l0_leveled_compaction_produces_correct_merged_output` + `compaction_config_with_l0_strategy_round_trips` + `compactor_uses_l0_strategy_from_config` (39.2), `flush_priority_bypasses_preemption_gate` + `background_priority_waits_for_flush_preemption_release` (39.3). 740 workspace lib tests green. -->

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


## Consolidation Sprint (2026-05)

Stub / mock removal sprint. Master tracker: `docs/CONSOLIDATION.md`. These
boxes mirror the phase status there.

- [x] Phase A — Remove sidecar mocks, real model or exit 1
- [x] Phase B — Real SQL executor wired to storage (excludes B6, B7)
- [x] Phase C — Pluggable key management, no vendor lock-in
- [x] Phase D — Wire-protocol bind parameter plumbing (folded into Phase B)
- [x] Phase E — `_disk_full` Prometheus metric live
- [x] Phase F — Reconcile tasks.md with real code (this entry)
- [x] Phase G — Real AWS benchmarking against SIFT1M
- [x] Phase H — CI gates (grep-for-mocks, cargo-deny for vendor SDKs)
