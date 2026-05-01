# Requirements Document — GalaxDB v1 Engine

## Introduction

GalaxDB v1 is a single-node, AI-native database engine written entirely in Rust. It unifies transactional row storage, vector similarity search, and versioned training data export into a single binary. v1 targets a 4-month build by 2–3 Rust engineers, delivering an LSM+PAX storage engine, mutable HNSW vector index, embedding inference sidecar, PostgreSQL-compatible wire protocol, Merkle DAG versioning, and Lance-format training export. The engine runs on Linux (production, io_uring), macOS, and Windows (development, tokio). All performance guarantees apply only to Linux production deployments with io_uring on NVMe storage.

## Glossary

- **Engine**: The GalaxDB v1 database engine process (Rust binary).
- **Storage_Engine**: The LSM-tree + PAX block subsystem responsible for durable row storage, compaction, and read path.
- **Memtable**: The in-memory write buffer backed by a crossbeam-skiplist with per-key Mutex sharding (16 shards) for MVCC concurrency.
- **PAX_Block**: A column-oriented storage block containing rows for a set of primary keys, with per-block zone maps and checksums.
- **ART_Index**: The Adaptive Radix Tree primary key index (Leis et al., ICDE 2013).
- **Bloom_Filter**: Per-SST Bloom filters with Monkey-optimal allocation across LSM levels (Dayan et al., TODS 2018).
- **Buffer_Pool**: The NUMA-aware memory pool partitioned into HotSet (70% LRU) and ScanBuffer (30% clock-sweep).
- **Compactor**: The background compaction subsystem implementing Lazy Leveling with MVCC garbage collection.
- **WAL**: Write-Ahead Log with XXH3-64 checksums, LZ4 compression, and group commit.
- **Blob_Log**: Content-addressed value log for KV-separated large values (>1 KB), following the BVLSM pattern.
- **TDE_Module**: Transparent Data Encryption module using AES-256-GCM with AWS KMS key management.
- **Statistics_Collector**: Background ANALYZE subsystem collecting per-column NDV, histograms, null fraction, and multi-column correlation statistics.
- **IO_Scheduler**: The I/O abstraction layer providing io_uring (Linux) or tokio (macOS/Windows) backends with HP (high-priority) and BK (background) queues.
- **SQL_Parser**: The SQL parsing frontend built on sqlparser-rs with AuroraSQL extensions.
- **Wire_Protocol**: The PostgreSQL simple query protocol (Q message) implementation.
- **Query_Executor**: The component that plans and executes parsed SQL statements.
- **HNSW_Index**: The mmap'd HNSW base graph for approximate nearest neighbor search.
- **Delta_Buffer**: The WAL-backed exact k-NN buffer for recent vector inserts not yet merged into the HNSW base graph.
- **Quantizer**: Platform-aware vector quantization module (SQ8 on x86-64, FP16 on ARM64, RaBitQ opt-in).
- **Embedding_Sidecar**: Standalone Rust binary running ONNX Runtime for embedding inference, communicating via Unix socket.
- **Backlog_Table**: Persistent table `_galaxdb_embedding_backlog` storing overflow embedding requests.
- **Merkle_DAG**: The versioning subsystem maintaining a Merkle tree over PAX block hashes for point-in-time queries.
- **Version_Tag**: A named, GC-exempt reference to a Merkle DAG root, optionally annotated with `FOR TRAINING`.
- **Lance_Exporter**: The subsystem that materializes versioned snapshots into Lance columnar format for ML training.
- **MinHash_Dedup**: Near-duplicate detection using MinHash LSH signatures (128-hash, 512 bytes per row).
- **Backup_Module**: The subsystem implementing `BACKUP TO` / `RESTORE FROM` with write-quiesce and checksum validation.
- **Observability_Module**: Embedded HTTP server exposing `/health`, `/metrics` (Prometheus), structured JSON logging, and OpenTelemetry tracing.
- **RateLimiter**: Auto-tuned token-bucket limiter controlling aggregate compaction + flush I/O bandwidth.
- **WriteController**: User-write throttle managing write admission based on pending compaction bytes.
- **Python_Client**: Python library providing connection, query execution, and `galaxdb.training_dataset(tag)` for PyTorch integration.
- **SST**: Sorted String Table file on disk, the immutable output of memtable flush or compaction.
- **Zone_Map**: Per-PAX-block min/max metadata for each column, used for scan pruning.

---

## Requirements

### Requirement 1: Memtable Write Path

**User Story:** As a database user, I want writes to be accepted into a concurrent in-memory buffer, so that write throughput is maximized without blocking on disk I/O.

#### Acceptance Criteria

1. WHEN a row is inserted or updated, THE Memtable SHALL accept the write into a crossbeam-skiplist with 16-shard per-key Mutex MVCC concurrency control.
2. WHEN the active Memtable reaches 64 MB, THE Storage_Engine SHALL seal the Memtable and create a new active Memtable for subsequent writes.
3. WHILE the total sealed-but-unflushed Memtable size exceeds 256 MB, THE Storage_Engine SHALL apply back-pressure by blocking new writes until flush progress reduces the total below 256 MB.
4. WHEN a value read from the Memtable is returned to a caller, THE Memtable SHALL copy the value out of the entry handle and drop the handle before any async (.await) operation, so that epoch-based memory reclamation is not blocked.
5. WHEN multiple concurrent writers insert rows with different primary keys, THE Memtable SHALL allow those writes to proceed without mutual exclusion across shards.
6. WHEN two concurrent writers attempt to update the same primary key, THE Memtable SHALL serialize those updates through the per-key Mutex for that key's shard.

### Requirement 2: PAX Block Storage Format

**User Story:** As a database user, I want row data stored in a columnar PAX format, so that analytical scans over individual columns are efficient while row-level access remains possible.

#### Acceptance Criteria

1. WHEN the Storage_Engine flushes a sealed Memtable to disk, THE Storage_Engine SHALL write the data as PAX_Blocks with column-oriented layout within each block.
2. THE Storage_Engine SHALL store a zone map (min/max per column) in each PAX_Block header.
3. THE Storage_Engine SHALL compute and store an XXH3-64 checksum and the magic number 0x47414C41 in each PAX_Block header.
4. WHEN reading a PAX_Block from disk, THE Storage_Engine SHALL verify the XXH3-64 checksum and magic number before returning data, and reject the block if verification fails.
5. WHEN a PAX_Block contains fixed-width columns, THE Storage_Engine SHALL compress those columns using delta encoding plus bit-packing (FastPFOR).
6. WHEN a PAX_Block contains variable-width columns, THE Storage_Engine SHALL compress those columns using Zstandard at level 3.
7. THE Storage_Engine SHALL NOT further compress embedding columns within PAX_Blocks, because quantization already handles size reduction.

### Requirement 3: ART Primary Key Index

**User Story:** As a database user, I want point lookups by primary key to be fast, so that OLTP read latency is minimized.

#### Acceptance Criteria

1. THE ART_Index SHALL maintain an Adaptive Radix Tree mapping primary keys to their location (memtable pointer or SST block offset).
2. WHEN a row is inserted into the Memtable, THE ART_Index SHALL be updated to reflect the new row's location.
3. WHEN a point lookup by primary key is issued, THE ART_Index SHALL return the row location without scanning SST files.
4. WHEN the Engine recovers from a crash, THE ART_Index SHALL be rebuilt from the persisted SST files and replayed WAL entries.

### Requirement 4: Bloom Filter with Monkey Allocation

**User Story:** As a database user, I want point reads to avoid unnecessary disk I/O on SST files that do not contain the target key, so that read latency is reduced.

#### Acceptance Criteria

1. THE Storage_Engine SHALL maintain a Bloom filter for each SST file.
2. THE Bloom_Filter SHALL allocate false-positive rates across LSM levels using the Monkey-optimal allocation strategy (Dayan et al., TODS 2018), minimizing the total false-positive sum given a fixed total memory budget.
3. WHEN a point read is issued, THE Storage_Engine SHALL consult the Bloom_Filter for each candidate SST before performing a disk read, and skip SSTs where the Bloom_Filter indicates the key is absent.
4. WHEN a new SST is created by flush or compaction, THE Storage_Engine SHALL build a Bloom_Filter for that SST using the Monkey-allocated false-positive rate for its level.

### Requirement 5: NUMA-Aware Buffer Pool

**User Story:** As a database operator, I want the buffer pool to be NUMA-aware, so that memory access latency is minimized on multi-socket systems.

#### Acceptance Criteria

1. THE Buffer_Pool SHALL be partitioned into a HotSet region (70% of allocated RAM, LRU eviction) and a ScanBuffer region (30% of allocated RAM, clock-sweep eviction).
2. WHEN a worker thread allocates a buffer frame, THE Buffer_Pool SHALL allocate from the NUMA node local to that worker thread.
3. WHEN the HotSet is full and a new block must be loaded, THE Buffer_Pool SHALL evict the least-recently-used block from the HotSet.
4. WHEN the ScanBuffer is full and a new scan block must be loaded, THE Buffer_Pool SHALL evict a block selected by clock-sweep from the ScanBuffer.
5. WHEN a block is accessed by a point lookup, THE Buffer_Pool SHALL place that block in the HotSet.
6. WHEN a block is accessed by a sequential scan, THE Buffer_Pool SHALL place that block in the ScanBuffer.

### Requirement 6: Lazy Leveling Compaction with MVCC GC

**User Story:** As a database operator, I want compaction to balance write amplification and read performance while reclaiming space from obsolete MVCC versions, so that storage is used efficiently.

#### Acceptance Criteria

1. THE Compactor SHALL implement Lazy Leveling: upper LSM levels use tiered compaction and the bottom level uses leveled compaction.
2. WHEN the Compactor merges SSTs during compaction, THE Compactor SHALL remove MVCC versions that are no longer needed by any active snapshot or pinned Version_Tag.
3. WHEN a Version_Tag is pinned, THE Compactor SHALL retain all MVCC versions referenced by that tag's Merkle root, regardless of age.
4. THE Compactor SHALL use 64 MB SST files in Month 1, configurable down to 8 MB in Month 4 hardening.

### Requirement 7: Write-Ahead Log (WAL)

**User Story:** As a database user, I want all committed writes to survive process crashes, so that durability is guaranteed.

#### Acceptance Criteria

1. WHEN a write transaction commits, THE WAL SHALL persist the write record to stable storage before acknowledging the commit to the client.
2. THE WAL SHALL format each record as `[type][seq_no][length][xxh3_checksum][payload]`.
3. THE WAL SHALL compress payloads using LZ4.
4. THE WAL SHALL support group commit with a configurable flush interval (default 10 ms).
5. WHEN a connection sets `DURABILITY STRICT`, THE WAL SHALL fsync each commit individually without group commit batching.
6. WHEN a connection sets `DURABILITY RELAXED`, THE WAL SHALL use group commit with the configured interval.
7. WHEN the WAL size exceeds 512 MB or 60 seconds have elapsed since the last checkpoint, THE Engine SHALL trigger a checkpoint that flushes the current Memtable and advances the WAL truncation point.
8. WHEN the Engine recovers from a crash, THE WAL SHALL replay records from the last checkpoint, skipping any record whose XXH3-64 checksum fails verification, and stopping at the first checksum failure.
9. THE Engine SHALL achieve crash recovery in less than 30 seconds under normal operating conditions.

### Requirement 8: KV Separation at WAL Time (BVLSM)

**User Story:** As a database operator, I want large values separated from the LSM tree at write time, so that the memtable back-pressure budget is not consumed by large values and compaction does not repeatedly rewrite them.

#### Acceptance Criteria

1. WHEN a value exceeds 1 KB, THE Storage_Engine SHALL write the value directly to the Blob_Log during WAL entry construction, storing only the 32-byte content hash and blob offset in the Memtable.
2. WHEN a value is 1 KB or smaller, THE Storage_Engine SHALL store the value inline in the Memtable and PAX_Block.
3. THE Blob_Log SHALL use multi-queue parallel writes following the BVLSM design.
4. WHEN the discardable space ratio in a Blob_Log file exceeds 50%, THE Storage_Engine SHALL trigger garbage collection to compact that blob file.
5. WHEN a row with a blob-separated value is read, THE Storage_Engine SHALL transparently fetch the value from the Blob_Log using the stored hash and offset.

### Requirement 9: Transparent Data Encryption (TDE)

**User Story:** As a database operator, I want all data encrypted at rest, so that the system meets GDPR and HIPAA compliance requirements.

#### Acceptance Criteria

1. THE TDE_Module SHALL encrypt every PAX_Block using AES-256-GCM before writing to disk.
2. THE TDE_Module SHALL encrypt every WAL record using AES-256-GCM before writing to disk.
3. THE TDE_Module SHALL use AES-NI hardware acceleration when available, targeting less than 8% CPU overhead.
4. THE TDE_Module SHALL retrieve and manage encryption keys via AWS KMS.
5. WHEN a PAX_Block or WAL record is read from disk, THE TDE_Module SHALL decrypt the data before returning it to the caller.
6. THE Wire_Protocol SHALL enforce TLS 1.3 for all client connections.

### Requirement 10: Statistics Collection

**User Story:** As a query optimizer, I want accurate table and column statistics, so that query plans select optimal access paths.

#### Acceptance Criteria

1. WHEN the `ANALYZE` command is executed on a table, THE Statistics_Collector SHALL compute per-column number of distinct values (NDV), equi-height histogram, and null fraction.
2. THE Statistics_Collector SHALL compute multi-column correlation statistics following the PostgreSQL extended statistics model.
3. THE Statistics_Collector SHALL run as a background task that does not block user queries.
4. THE Query_Executor SHALL use collected statistics for filter selectivity estimation and for choosing between HNSW graph traversal and brute-force scan.


### Requirement 11: I/O Abstraction Layer

**User Story:** As a database operator, I want the engine to use io_uring on Linux for maximum I/O performance while falling back to tokio on other platforms, so that the engine runs on all development platforms.

#### Acceptance Criteria

1. WHEN running on Linux 5.10+ with io_uring available, THE IO_Scheduler SHALL use io_uring with separate HP (high-priority) and BK (background) submission queues.
2. WHEN running on macOS or Windows, THE IO_Scheduler SHALL use tokio's native async I/O (kqueue or IOCP).
3. WHEN the environment variable `GALAXDB_IO_BACKEND` is set to `tokio`, THE IO_Scheduler SHALL use tokio regardless of platform.
4. THE IO_Scheduler SHALL present a unified async interface to the Storage_Engine, so that upper layers do not depend on the specific I/O backend.
5. WHEN the io_uring HP-queue latency exceeds 1.5× the idle baseline for three consecutive 100 ms measurement windows, THE IO_Scheduler SHALL report this condition to the RateLimiter.

### Requirement 12: SQL Parser with AuroraSQL Extensions

**User Story:** As a database user, I want to interact with GalaxDB using SQL with AI-native extensions, so that I can use familiar syntax for both relational and vector operations.

#### Acceptance Criteria

1. THE SQL_Parser SHALL parse standard SQL DDL statements: `CREATE TABLE`, `DROP TABLE`, `ALTER TABLE`.
2. THE SQL_Parser SHALL parse standard SQL DML statements: `INSERT`, `SELECT`, `UPDATE`, `DELETE`.
3. THE SQL_Parser SHALL parse the AuroraSQL extension `CREATE TABLE ... (col TEXT EMBEDDING MODEL 'name' DIM n)` for declaring embedding columns.
4. THE SQL_Parser SHALL parse the AuroraSQL extension `SEMANTIC_MATCH(col, 'query', threshold)` for vector similarity search.
5. THE SQL_Parser SHALL parse the AuroraSQL extension `AT VERSION timestamp_or_tag` with optional consistency mode (`ROW_SNAPSHOT`, `SEMANTIC_FRESH`).
6. THE SQL_Parser SHALL parse `CREATE VERSION TAG 'name' [FOR TRAINING [WITH TRAINING PRECISION 'sq8'|'rabitq'|'float32'] [TRAINING SEED n]]`.
7. THE SQL_Parser SHALL parse `BULK INSERT` for direct PAX-block writes.
8. THE SQL_Parser SHALL parse `SHOW EMBEDDING HEALTH` for model version distribution reporting.
9. THE SQL_Parser SHALL parse `BACKUP TO '/path'` and `RESTORE FROM '/path'`.
10. THE SQL_Parser SHALL parse `ANALYZE table_name`.
11. THE SQL_Parser SHALL be built on the `sqlparser-rs` crate, extending its PostgreSQL dialect.
12. IF the SQL_Parser encounters a syntax error, THEN THE SQL_Parser SHALL return a descriptive error message including the position of the error.

### Requirement 13: PostgreSQL Wire Protocol (Simple Query)

**User Story:** As a database user, I want to connect to GalaxDB using standard PostgreSQL client libraries, so that I can use existing tools like psycopg2 and SQLAlchemy.

#### Acceptance Criteria

1. THE Wire_Protocol SHALL implement the PostgreSQL simple query protocol (Q message flow: Query → RowDescription → DataRow → CommandComplete → ReadyForQuery).
2. THE Wire_Protocol SHALL implement the PostgreSQL startup handshake (StartupMessage, AuthenticationOk, ParameterStatus, BackendKeyData, ReadyForQuery).
3. THE Wire_Protocol SHALL support basic DDL and DML operations through the simple query protocol.
4. THE Wire_Protocol SHALL expose `pg_catalog` stubs sufficient for psycopg2 and SQLAlchemy to establish connections and perform simple queries.
5. THE Wire_Protocol SHALL support up to 1000 concurrent connections (configurable).
6. WHEN the connection count exceeds the configured maximum, THE Wire_Protocol SHALL reject new connections with an appropriate error message.
7. THE Engine SHALL manage connections as async Rust tasks using tokio.

### Requirement 14: DDL Execution (CREATE TABLE, DROP TABLE)

**User Story:** As a database user, I want to create and drop tables with standard and embedding columns, so that I can define my data schema.

#### Acceptance Criteria

1. WHEN a `CREATE TABLE` statement is executed, THE Query_Executor SHALL create the table metadata, initialize the ART_Index, and prepare the Memtable for writes.
2. WHEN a `CREATE TABLE` includes an `EMBEDDING MODEL 'name' DIM n` column, THE Query_Executor SHALL register the column for automatic embedding generation via the Embedding_Sidecar.
3. WHEN a `DROP TABLE` statement is executed, THE Query_Executor SHALL remove the table metadata, release associated storage, and remove the ART_Index entries.
4. IF a `CREATE TABLE` specifies a table name that already exists, THEN THE Query_Executor SHALL return an error.
5. IF a `DROP TABLE` specifies a table name that does not exist, THEN THE Query_Executor SHALL return an error.

### Requirement 15: DML Execution (INSERT, SELECT, UPDATE, DELETE)

**User Story:** As a database user, I want to insert, query, update, and delete rows, so that I can manage my data.

#### Acceptance Criteria

1. WHEN an `INSERT` statement is executed, THE Query_Executor SHALL write the row to the Memtable and WAL, update the ART_Index, and trigger embedding generation for any embedding columns.
2. WHEN a `SELECT` statement is executed, THE Query_Executor SHALL read from the Memtable and SST files, applying zone-map pruning and Bloom_Filter checks, and return matching rows.
3. WHEN an `UPDATE` statement is executed, THE Query_Executor SHALL write a new MVCC version of the row to the Memtable and WAL.
4. WHEN a `DELETE` statement is executed, THE Query_Executor SHALL write a tombstone to the Memtable and WAL.
5. WHEN an `UPDATE` targets a column that is the source for an embedding column, THE Query_Executor SHALL reject the update with an error message explaining the limitation and suggesting DELETE + INSERT as a workaround.
6. WHEN a `BULK INSERT` statement is executed, THE Query_Executor SHALL write rows directly as PAX_Blocks, bypassing the Memtable for bulk loading efficiency.
7. WHEN a `SELECT` includes a `WHERE` clause, THE Query_Executor SHALL use zone-map pruning to skip PAX_Blocks whose min/max ranges do not intersect the predicate.

### Requirement 16: Snapshot Isolation

**User Story:** As a database user, I want transactions to see a consistent snapshot of the database, so that concurrent reads and writes do not produce dirty reads, non-repeatable reads, or phantom rows.

#### Acceptance Criteria

1. WHEN a transaction begins, THE Engine SHALL assign a snapshot timestamp that determines the visible set of MVCC versions.
2. THE Engine SHALL guarantee no dirty reads: a transaction SHALL NOT see uncommitted writes from other transactions.
3. THE Engine SHALL guarantee no non-repeatable reads: a row read twice within the same transaction SHALL return the same value.
4. THE Engine SHALL guarantee no phantom reads: a range scan executed twice within the same transaction SHALL return the same set of rows.
5. WHEN two concurrent transactions write to the same key, THE Engine SHALL detect the write-write conflict and abort one transaction.
6. THE Engine SHALL document that write-skew anomalies are possible under Snapshot Isolation and that SSI is deferred to v2.

### Requirement 17: Mutable HNSW Vector Index

**User Story:** As a database user, I want to perform approximate nearest neighbor search on embedding columns, so that I can find semantically similar rows.

#### Acceptance Criteria

1. THE HNSW_Index SHALL maintain an mmap'd base graph on disk for approximate nearest neighbor search.
2. WHEN a new embedding is generated, THE Delta_Buffer SHALL store the vector with a `DELTA_INSERT` record in the WAL.
3. WHEN a vector similarity query is executed, THE Engine SHALL search both the HNSW_Index base graph and the Delta_Buffer, then union and re-rank the results.
4. WHEN the Delta_Buffer size exceeds `max(10000, 1% of total_indexed)` vectors, THE Engine SHALL trigger a merge of the Delta_Buffer into the HNSW_Index base graph.
5. WHEN a merge is performed, THE HNSW_Index SHALL use atomic rename (shadow file + rename()) to ensure crash safety and zero downtime.
6. WHEN a vector is deleted, THE HNSW_Index SHALL write a tombstone, and trigger an emergency merge if tombstones exceed 20% of indexed vectors.
7. WHEN the Engine recovers from a crash, THE Delta_Buffer SHALL be replayed from the WAL in batches of 1000 entries.

### Requirement 18: Platform-Aware Quantization

**User Story:** As a database operator, I want vector quantization to automatically use the best encoding for the CPU architecture, so that vector search performance is optimized per platform.

#### Acceptance Criteria

1. WHEN running on x86-64 with AVX2 or AVX-512, THE Quantizer SHALL default to SQ8 (int8 scalar quantization) providing 4× compression with SIMD acceleration.
2. WHEN running on ARM64 (Apple Silicon, Graviton), THE Quantizer SHALL default to FP16 (half-precision float) providing 2× compression with NEON acceleration.
3. WHERE the user opts in to RaBitQ quantization, THE Quantizer SHALL apply random rotation plus binary quantization providing 32× compression.
4. WHERE the user opts in to SQ8 on ARM64, THE Quantizer SHALL apply SQ8 with the documented caveat that throughput is approximately 3× lower than AVX2.
5. THE Quantizer SHALL detect the CPU architecture at startup and select the default quantization scheme without user configuration.

### Requirement 19: Embedding Inference Sidecar

**User Story:** As a database user, I want embeddings generated automatically when I insert text into an embedding column, so that I do not need an external embedding service.

#### Acceptance Criteria

1. THE Embedding_Sidecar SHALL run as a standalone Rust binary using ONNX Runtime (`ort` crate) for model inference.
2. THE Embedding_Sidecar SHALL communicate with the Engine via a Unix socket.
3. THE Embedding_Sidecar SHALL monitor the parent Engine process ID and terminate if the parent exits.
4. WHEN the Embedding_Sidecar crashes, THE Engine SHALL restart the sidecar with exponential backoff.
5. THE Embedding_Sidecar SHALL maintain a heartbeat protocol with the Engine.
6. WHILE the Embedding_Sidecar has more than 10,000 in-flight embedding requests, THE Embedding_Sidecar SHALL overflow new requests to the Backlog_Table (`_galaxdb_embedding_backlog`).
7. THE Backlog_Table SHALL use `DURABILITY STRICT` regardless of the session's durability setting.
8. WHEN the Embedding_Sidecar recovers capacity, THE Embedding_Sidecar SHALL drain the Backlog_Table before accepting new in-flight requests.

### Requirement 20: Model-Version Tracking

**User Story:** As a database operator, I want the system to track which embedding model version produced each row's embedding, so that model upgrades do not silently corrupt search quality.

#### Acceptance Criteria

1. THE Engine SHALL store an `_embedding_model_version` metadata field with each embedded row.
2. WHEN the Embedding_Sidecar model changes, THE Engine SHALL mark all rows with the old model version as stale.
3. WHEN rows are marked stale, THE Engine SHALL enqueue them for re-embedding via the Backlog_Table.
4. WHEN the `SHOW EMBEDDING HEALTH` command is executed, THE Engine SHALL report the distribution of model versions across rows and the re-embedding progress.

### Requirement 21: SEMANTIC_MATCH Query

**User Story:** As a database user, I want to find rows semantically similar to a query string, so that I can perform AI-powered search.

#### Acceptance Criteria

1. WHEN a `SELECT` includes `SEMANTIC_MATCH(col, 'query', threshold)`, THE Query_Executor SHALL embed the query string via the Embedding_Sidecar, then search the HNSW_Index and Delta_Buffer for vectors within the similarity threshold.
2. THE Query_Executor SHALL union results from the HNSW_Index and Delta_Buffer and re-rank by similarity score.
3. WHEN a `SELECT` includes both structured `WHERE` predicates and `SEMANTIC_MATCH`, THE Query_Executor SHALL use the adaptive planner to choose between filtered brute-force scan (when filter cardinality is very low) and HNSW graph traversal with post-filtering.
4. IF the Embedding_Sidecar is unavailable, THEN THE Query_Executor SHALL return an error indicating that semantic search is temporarily unavailable.

### Requirement 22: Adaptive Query Planner

**User Story:** As a database user, I want the query planner to automatically choose the best execution strategy for hybrid queries, so that performance is optimal regardless of filter selectivity.

#### Acceptance Criteria

1. WHEN a hybrid query combines structured filters with SEMANTIC_MATCH, THE Query_Executor SHALL estimate the filter cardinality using Statistics_Collector data.
2. WHEN the estimated filter cardinality is very low (high selectivity), THE Query_Executor SHALL choose brute-force scan over the filtered candidate set rather than HNSW graph traversal.
3. WHEN the estimated filter cardinality is moderate to high, THE Query_Executor SHALL choose HNSW graph traversal with post-filtering.
4. THE Query_Executor SHALL log the chosen plan path for observability.


### Requirement 23: Merkle DAG Versioning

**User Story:** As a data scientist, I want to query the database at any historical point in time, so that I can reproduce past results and audit data changes.

#### Acceptance Criteria

1. WHEN a write transaction commits, THE Merkle_DAG SHALL record the PAX_Block with its commit timestamp and compute a Merkle tree hash over the block hashes to produce a version root.
2. WHEN a `SELECT ... AT VERSION timestamp` query is executed, THE Query_Executor SHALL filter PAX_Blocks to return only rows visible at the specified timestamp.
3. WHEN a `SELECT ... AT VERSION tag_name` query is executed, THE Query_Executor SHALL resolve the tag to its Merkle root and return rows visible at that version.
4. WHEN `AT VERSION` is used with `SEMANTIC_MATCH` and no consistency mode is specified, THE Query_Executor SHALL default to `ROW_SNAPSHOT` mode and reject the `SEMANTIC_MATCH` clause.
5. WHEN `AT VERSION` is used with `CONSISTENCY 'SEMANTIC_FRESH'`, THE Query_Executor SHALL search the current HNSW_Index against historical rows and include a warning in the result metadata.
6. THE Engine SHALL NOT support `CONSISTENCY 'SEMANTIC_SNAPSHOT'` in v1; requests for this mode SHALL return an error indicating it is a v2 feature.

### Requirement 24: Version Tags

**User Story:** As a data scientist, I want to create named version tags, so that I can reference specific dataset snapshots for training and reproducibility.

#### Acceptance Criteria

1. WHEN `CREATE VERSION TAG 'name'` is executed, THE Engine SHALL create a named reference to the current Merkle DAG root.
2. WHEN a Version_Tag is created, THE Compactor SHALL treat all PAX_Blocks referenced by that tag as GC-exempt (pinned).
3. WHEN `CREATE VERSION TAG 'name' FOR TRAINING` is executed, THE Engine SHALL guarantee deterministic block order (primary key sort) and store a shuffle seed with the tag metadata.
4. WHEN `CREATE VERSION TAG 'name' FOR TRAINING WITH TRAINING PRECISION 'sq8'` is executed, THE Engine SHALL record the requested training precision with the tag.
5. WHEN `CREATE VERSION TAG 'name' FOR TRAINING TRAINING SEED n` is executed, THE Engine SHALL store the shuffle seed n for reproducible training data ordering.

### Requirement 25: Lance Training Export

**User Story:** As a machine learning engineer, I want to export versioned snapshots in Lance format, so that I can feed training data directly to PyTorch with zero-copy.

#### Acceptance Criteria

1. WHEN a `FOR TRAINING` tagged version is exported, THE Lance_Exporter SHALL materialize the snapshot as a Lance-format dataset.
2. THE Lance_Exporter SHALL support training precision options: `float32`, `sq8`, and `rabitq`, reducing I/O volume by 4–32× for quantized precisions.
3. THE Python_Client SHALL provide a `galaxdb.training_dataset(tag)` function that returns a PyTorch `IterableDataset` with zero-copy access to the Lance data.
4. WHEN a training export is performed, THE Engine SHALL record the export in the `_galaxdb_training_exports` system table with: tag, filter, precision, dedup flag, curriculum mode, row count, export timestamp, and hash.

### Requirement 26: Near-Duplicate Detection (MinHash LSH)

**User Story:** As a data scientist, I want to detect and filter near-duplicate rows, so that training data quality is improved.

#### Acceptance Criteria

1. WHEN a row is written that contains text content, THE MinHash_Dedup SHALL compute a MinHash LSH signature (128-hash, 512 bytes) in the Rust write path.
2. WHEN a `SELECT ... WHERE NOT DUPLICATE` filter is used, THE Query_Executor SHALL exclude rows identified as near-duplicates based on their MinHash signatures.
3. THE Engine SHALL run a background job that periodically refreshes near-duplicate groups.

### Requirement 27: Backup and Restore

**User Story:** As a database operator, I want to create consistent backups and restore from them, so that I can recover from data loss or migrate data.

#### Acceptance Criteria

1. WHEN `BACKUP TO '/path'` is executed, THE Backup_Module SHALL acquire a brief write-quiesce (less than 100 ms) to flush the Memtable and create a clean Merkle root.
2. WHILE the write-quiesce is held, THE Engine SHALL continue serving read queries.
3. WHEN the write-quiesce completes, THE Backup_Module SHALL copy PAX_Blocks and WAL to the target path, and new writes SHALL resume immediately.
4. WHEN `RESTORE FROM '/path'` is executed, THE Backup_Module SHALL validate all block checksums, replay the WAL, and rebuild the ART_Index and HNSW_Index.
5. IF a block checksum fails during restore, THEN THE Backup_Module SHALL report the corrupted block and abort the restore.

### Requirement 28: Observability

**User Story:** As a database operator, I want health checks, metrics, and distributed tracing, so that I can monitor the engine in production.

#### Acceptance Criteria

1. THE Observability_Module SHALL expose an HTTP endpoint at `/health` returning the engine's health status.
2. THE Observability_Module SHALL expose an HTTP endpoint at `/metrics` returning metrics in Prometheus exposition format.
3. THE Observability_Module SHALL emit structured JSON log lines with configurable log level.
4. THE Observability_Module SHALL propagate OpenTelemetry trace context (W3C `traceparent` format) across query execution, including child spans for HNSW search, Delta_Buffer search, and Embedding_Sidecar calls.
5. THE Observability_Module SHALL carry trace context in SQL commenter format through the Wire_Protocol.
6. THE Observability_Module SHALL export metrics for: buffer pool pressure, embedding queue depth, checkpoint status, compaction debt, WAL write latency, and HNSW recall estimates.

### Requirement 29: Write Stall Mitigation — RateLimiter

**User Story:** As a database operator, I want compaction I/O to be rate-limited dynamically, so that compaction does not starve user-facing I/O.

#### Acceptance Criteria

1. THE RateLimiter SHALL control aggregate compaction and flush I/O bandwidth using an auto-tuned token-bucket algorithm.
2. THE RateLimiter SHALL set the upper bound to 70% of measured NVMe write bandwidth, calibrated at startup.
3. WHEN the IO_Scheduler reports that HP-queue latency exceeds 1.5× the idle baseline for three consecutive 100 ms windows, THE RateLimiter SHALL temporarily lower the compaction ceiling by 30%.
4. WHEN HP-queue latency returns to normal, THE RateLimiter SHALL restore the compaction ceiling to its previous level.

### Requirement 30: Write Stall Mitigation — WriteController

**User Story:** As a database operator, I want user writes to be throttled gracefully when compaction falls behind, so that the system degrades predictably rather than stalling.

#### Acceptance Criteria

1. WHEN pending compaction bytes exceed the `soft_pending_compaction_bytes_limit` (default 32 GB), THE WriteController SHALL slow user writes to `delayed_write_rate` (default 16 MB/s).
2. WHEN pending compaction bytes exceed the `hard_pending_compaction_bytes_limit` (default 64 GB), THE WriteController SHALL stop user writes until pending bytes fall below the hard limit.
3. THE WriteController SHALL operate on 1 ms intervals, applying gradual slowdown proportional to the excess above the soft limit.
4. WHEN pending compaction bytes fall below the soft limit, THE WriteController SHALL restore full write throughput.

### Requirement 31: Disk Full Handling

**User Story:** As a database operator, I want the engine to handle disk-full conditions gracefully, so that data is not corrupted when storage is exhausted.

#### Acceptance Criteria

1. THE Engine SHALL pre-allocate a 32 MB reserve file at startup.
2. WHEN available disk space is exhausted, THE Engine SHALL delete the reserve file to free space for a clean checkpoint.
3. WHEN the reserve file has been consumed, THE Engine SHALL block all writes and perform a clean checkpoint before stopping.
4. THE Engine SHALL NOT corrupt existing data when disk space is exhausted.

### Requirement 32: Python Client

**User Story:** As a Python developer, I want a native Python client library, so that I can connect to GalaxDB and use it from Python applications and ML pipelines.

#### Acceptance Criteria

1. THE Python_Client SHALL connect to the Engine using the PostgreSQL wire protocol.
2. THE Python_Client SHALL support executing DDL and DML statements and returning results.
3. THE Python_Client SHALL provide `galaxdb.training_dataset(tag)` returning a PyTorch `IterableDataset` backed by Lance-format data.
4. THE Python_Client SHALL be compatible with Python 3.9+.

### Requirement 33: pg_catalog Stubs

**User Story:** As a database user, I want psycopg2 and SQLAlchemy to connect without errors, so that I can use standard Python database tooling.

#### Acceptance Criteria

1. THE Engine SHALL expose `pg_catalog` system tables sufficient for psycopg2 to complete its connection handshake and introspection queries.
2. THE Engine SHALL expose `pg_catalog` system tables sufficient for SQLAlchemy (simple mode) to reflect table metadata.
3. WHEN a query references an unsupported `pg_catalog` table, THE Engine SHALL return an empty result set rather than an error.

### Requirement 34: Connection Management

**User Story:** As a database operator, I want the engine to manage connections efficiently with configurable limits, so that the system remains stable under load.

#### Acceptance Criteria

1. THE Engine SHALL manage each client connection as an async Rust task using tokio.
2. THE Engine SHALL support a configurable maximum connection count (default 1000).
3. WHEN the connection count reaches the configured maximum, THE Engine SHALL reject new connections with an appropriate PostgreSQL error code.
4. THE Engine SHALL NOT require an external connection pooler for workloads within the connection limit.

### Requirement 35: Deployment Modes

**User Story:** As a developer, I want to use GalaxDB as either an embedded library or a standalone server, so that I can choose the deployment model that fits my application.

#### Acceptance Criteria

1. THE Engine SHALL support an embedded mode where the database runs in-process, available on Linux, macOS, and Windows.
2. THE Engine SHALL support a standalone server mode where the database runs as a separate process accepting wire protocol connections, available on Linux and macOS.
3. WHILE running in embedded mode on macOS or Windows, THE Engine SHALL use tokio for I/O and document that production performance guarantees do not apply.
4. THE Engine core binary SHALL be less than 70 MB (statically linked Rust).
5. THE Engine full binary (core + sidecar + default model) SHALL be less than 350 MB.

### Requirement 36: vLSM Structural Improvements (Month 4)

**User Story:** As a database operator, I want reduced write stalls and improved P99 latency under sustained write load, so that the system performs predictably.

#### Acceptance Criteria

1. WHEN Month 4 hardening is applied, THE Storage_Engine SHALL reduce the default SST size from 64 MB to 8 MB (configurable).
2. THE Compactor SHALL eliminate tiering compaction at L0, using only leveled compaction at L0.
3. THE Compactor SHALL implement SILK-style dynamic bandwidth pre-emption for flush operations, prioritizing flushes over compaction when memtable back-pressure is high.

### Requirement 37: Chaos Testing

**User Story:** As a database developer, I want automated chaos tests, so that durability and recovery guarantees are validated under adverse conditions.

#### Acceptance Criteria

1. THE Engine SHALL include chaos tests that kill the Embedding_Sidecar mid-request and verify that the Engine recovers the sidecar and drains the backlog.
2. THE Engine SHALL include chaos tests that kill the Engine process mid-flush and verify that recovery produces a consistent state with no data loss for committed transactions.
3. THE Engine SHALL include chaos tests that simulate WAL corruption and verify that recovery skips corrupt records and recovers all valid data.
4. THE Engine SHALL include chaos tests that simulate disk-full conditions and verify that the Engine performs a clean shutdown without data corruption.
5. THE Engine SHALL include chaos tests that verify recovery completes in less than 30 seconds.

### Requirement 38: Training Data Lineage

**User Story:** As a compliance officer, I want a complete audit trail of all training data exports, so that the system satisfies EU AI Act Article 13 requirements.

#### Acceptance Criteria

1. THE Engine SHALL maintain a `_galaxdb_training_exports` system table.
2. WHEN a training export is performed, THE Engine SHALL insert a record into `_galaxdb_training_exports` containing: tag name, filter expression, precision, dedup flag, curriculum mode, row count, export timestamp, and content hash.
3. THE `_galaxdb_training_exports` table SHALL be append-only and not deletable by user queries.

### Requirement 39: Embedding Staleness Tracking

**User Story:** As a database user, I want to know which rows have stale embeddings, so that I can understand the freshness of semantic search results.

#### Acceptance Criteria

1. THE Engine SHALL maintain an `_embedding_stale` flag for each row with an embedding column.
2. WHEN a row's embedding is generated or re-generated, THE Engine SHALL clear the `_embedding_stale` flag.
3. WHEN the Embedding_Sidecar model version changes, THE Engine SHALL set the `_embedding_stale` flag on all rows embedded with the old model version.
4. THE Engine SHALL update the `_embedding_stale` flag through the standard LSM update path, ensuring the flag is durable and consistent.
