//! Storage Engine facade — the unified API that the SQL executor calls.
//!
//! Connects: Memtable + WAL + ART Index + Flush + Buffer Pool + Compaction
//! into a single coherent interface for reading and writing rows.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock, Weak};
use std::time::Duration;

use galaxdb_common::{GalaxError, GalaxResult, Timestamp};
use galaxdb_io::{IoScheduler, IoPriority};

use crate::art::{ArtIndex, RowLocation};
use crate::columnar::{ColumnarRegistration, RowColumnSplitter};
use crate::flush::{self, FlushConfig};
use crate::memtable::MemtableManager;
use crate::write_controller::{WriteAdmission, WriteController, WriteControllerConfig};

use crate::wal::{DurabilityMode, WalRecordType, WalWriter, WalWriterConfig};

/// Configuration for the storage engine.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub data_dir: PathBuf,
    pub memtable_size_bytes: u64,
    pub back_pressure_bytes: u64,
    pub wal_group_commit_ms: u64,
    /// Target SST file size in bytes (default: 8 MB).
    /// Smaller SSTs improve point read latency on cold cache (less data to
    /// read + decompress per lookup). Larger SSTs improve write throughput
    /// and reduce file count.
    pub sst_size_bytes: u64,
    /// Maximum rows per SST block (default: 100,000).
    pub max_rows_per_sst: usize,
    /// Maximum SST cache size in bytes (default: 1 GB).
    /// Controls how much decompressed column data is kept in memory.
    /// Set to a small value (e.g., 10 MB) for cold-cache benchmarks.
    pub sst_cache_bytes: u64,
    /// Number of worker threads the runtime compaction driver uses to
    /// build output SSTs in parallel (default: 4). Clamped at run time to
    /// `1..=number_of_timestamp_buckets`. Fed by auto-tune (Req 12).
    pub compaction_concurrency: usize,
    /// Trigger threshold: when the number of on-disk SST files reaches
    /// this value, a flush schedules a background compaction that merges
    /// them (default: 4). A larger value trades read amplification for
    /// fewer compactions.
    pub l0_compaction_trigger: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("galaxdb_data"),
            memtable_size_bytes: 64 * 1024 * 1024,
            back_pressure_bytes: 256 * 1024 * 1024,
            wal_group_commit_ms: 10,
            sst_size_bytes: 8 * 1024 * 1024,   // 8 MB — smaller SSTs for fast point reads
            max_rows_per_sst: 100_000,
            sst_cache_bytes: 1024 * 1024 * 1024, // 1 GB
            compaction_concurrency: 4,
            l0_compaction_trigger: 4,
        }
    }
}

/// A single row stored in the engine.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredRow {
    pub key: Vec<u8>,
    pub columns: Vec<(String, Vec<u8>)>,
    pub timestamp: Timestamp,
}

/// Tracks a single SST file for the read path.
///
/// Each SST file contains multiple small PAX blocks (~100 rows, ~64KB each)
/// with a block index at the end. Following RocksDB's BlockBasedTable pattern,
/// a point read uses the block index to locate the exact block, then does a
/// single targeted `pread` of ~64KB from NVMe (~18µs) instead of reading the
/// entire SST file.
struct SstEntry {
    path: PathBuf,
    /// Block index loaded from the SST footer. Maps block_offset → (file_offset, length).
    /// Kept in memory for O(1) block lookup during point reads.
    block_index: crate::sst::SstBlockIndex,
    /// Whether this SST was written with AEGIS-256 encryption.
    #[cfg(feature = "aegis-tde")]
    encrypted: bool,
}

/// Registry of all SST files on disk.
struct SstRegistry {
    entries: HashMap<u64, SstEntry>,
}

impl SstRegistry {
    fn with_cache_limit(_max_bytes: u64) -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Register an SST file by reading its block index from the footer.
    /// The block index is small (12 bytes per block) and kept in memory.
    fn register(&mut self, sst_id: u64, path: PathBuf) {
        // Read the SST file to extract the block index from the footer.
        // The block index is at the end of the file and is typically < 1KB.
        let block_index = if let Ok(data) = std::fs::read(&path) {
            crate::sst::SstBlockIndex::from_file_data(&data).unwrap_or_else(|_| {
                // Fallback for legacy single-block SSTs (no footer)
                let mut idx = crate::sst::SstBlockIndex::new();
                idx.add_block(0, data.len() as u32);
                idx
            })
        } else {
            crate::sst::SstBlockIndex::new()
        };

        self.entries.insert(sst_id, SstEntry {
            path,
            block_index,
            #[cfg(feature = "aegis-tde")]
            encrypted: false,
        });
    }

    #[cfg(feature = "aegis-tde")]
    fn register_encrypted(
        &mut self,
        sst_id: u64,
        path: PathBuf,
        _tde: &galaxdb_crypto::AegisTdeModule,
    ) {
        let block_index = if let Ok(data) = std::fs::read(&path) {
            crate::sst::SstBlockIndex::from_file_data(&data).unwrap_or_else(|_| {
                let mut idx = crate::sst::SstBlockIndex::new();
                idx.add_block(0, data.len() as u32);
                idx
            })
        } else {
            crate::sst::SstBlockIndex::new()
        };

        self.entries.insert(sst_id, SstEntry {
            path,
            block_index,
            encrypted: true,
        });
    }

    /// Read a single value by doing a targeted block read.
    ///
    /// Uses the block index to find the exact file offset and length of the
    /// PAX block containing the target row, then reads ONLY that block from
    /// disk via the IoScheduler HP queue. This is one NVMe read of ~64KB
    /// (~18µs) instead of reading the entire SST file (~8MB, ~2ms).
    #[cfg(feature = "aegis-tde")]
    fn read_value(
        &self,
        sst_id: u64,
        block_offset: u64,
        row_offset: u32,
        tde: Option<&galaxdb_crypto::AegisTdeModule>,
        io: &dyn IoScheduler,
    ) -> Option<Vec<u8>> {
        let entry = self.entries.get(&sst_id)?;
        let block_info = entry.block_index.get_block(block_offset)?;

        // Targeted pread: one NVMe read of ~62KB via io_uring HP queue
        let block_bytes = io.read_sync(
            &entry.path,
            block_info.file_offset,
            block_info.block_len as usize,
            IoPriority::High,
        ).ok()?;

        // Decrypt if needed
        let data = if entry.encrypted {
            if let Some(module) = tde {
                module.decrypt(&block_bytes).ok()?
            } else {
                block_bytes
            }
        } else {
            block_bytes
        };

        // Zero-copy row extraction: parses minimal header, slices directly
        // into column data, scans length prefixes to target row.
        // No PaxBlock struct allocation, no 62KB memcpy.
        crate::pax::read_value_from_raw_block(&data, 1, row_offset).ok()
    }

    #[cfg(not(feature = "aegis-tde"))]
    fn read_value(
        &self,
        sst_id: u64,
        block_offset: u64,
        row_offset: u32,
        io: &dyn IoScheduler,
    ) -> Option<Vec<u8>> {
        let entry = self.entries.get(&sst_id)?;
        let block_info = entry.block_index.get_block(block_offset)?;

        // Targeted pread: one NVMe read of ~62KB
        let block_bytes = io.read_sync(
            &entry.path,
            block_info.file_offset,
            block_info.block_len as usize,
            IoPriority::High,
        ).ok()?;

        // Zero-copy row extraction
        crate::pax::read_value_from_raw_block(&block_bytes, 1, row_offset).ok()
    }
}

/// Supplies the set of commit timestamps that runtime compaction must
/// never garbage-collect, because a version tag (or active snapshot) can
/// still read the row versions committed at or before them.
///
/// The engine does not own the tag catalog (it lives in the embedded
/// `Database` layer), so the catalog owner installs an implementation via
/// [`Engine::set_pin_source`]. When no pin source is installed, runtime
/// compaction keeps only the latest version of each key (plus live
/// tombstone handling) — correct for a deployment with no version tags.
pub trait PinSource: Send + Sync {
    /// Current set of pinned commit timestamps. May be empty.
    fn pinned_timestamps(&self) -> Vec<Timestamp>;
}

/// Outcome of a runtime compaction run, returned for observability + tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactionStats {
    /// Number of SST files merged (deleted at the end).
    pub input_ssts: usize,
    /// Number of SST files written (registered).
    pub output_ssts: usize,
    /// Number of distinct live keys retained.
    pub keys_retained: usize,
    /// Number of (key, version) pairs dropped by MVCC GC.
    pub versions_gc_dropped: usize,
}

/// The storage engine — unified read/write API.
pub struct Engine {
    config: EngineConfig,
    memtable_mgr: MemtableManager,
    art: Arc<ArtIndex>,
    wal: Arc<WalWriter>,
    sst_registry: RwLock<SstRegistry>,
    next_timestamp: AtomicU64,
    next_sst_id: AtomicU64,
    row_count: AtomicU64,
    /// I/O scheduler — routes reads/writes through io_uring HP/BK queues on
    /// Linux, or tokio::fs on macOS/Windows. All SST reads and flush writes
    /// go through this scheduler.
    io_scheduler: Arc<dyn IoScheduler>,
    /// Serializes runtime compaction runs so at most one merge executes at
    /// a time (its internal build phase still fans out across
    /// `compaction_concurrency` threads).
    compaction_lock: Mutex<()>,
    /// Optional source of pinned timestamps for MVCC GC during compaction.
    pin_source: RwLock<Option<Arc<dyn PinSource>>>,
    /// User-write admission throttle driven by the pending-compaction
    /// backlog (v1 WriteController design). Writes consult it so that if
    /// compaction falls behind, ingest is slowed/blocked instead of letting
    /// the SST backlog grow without bound.
    write_controller: WriteController,
    /// Wake signal + shutdown flag for the optional background compaction
    /// worker. `(wake, shutdown)` guarded by the mutex; the condvar wakes
    /// the worker when a flush signals new work or on shutdown.
    compaction_wake: Arc<(Mutex<(bool, bool)>, Condvar)>,
    /// Whether a background compaction worker is running. When `false`,
    /// `flush_memtable` compacts inline (used by raw-`Engine` callers and
    /// tests); when `true`, the flush only signals the worker so it never
    /// blocks on a merge.
    bg_worker_active: AtomicBool,
    /// Per-table columnar storage registrations (HTAP ADR-0002). The SQL
    /// layer registers a [`RowColumnSplitter`] per columnar table's key
    /// prefix; flush consults this to append one typed PAX column per SQL
    /// column. Empty by default, so legacy tables are unaffected.
    columnar: RwLock<Vec<ColumnarRegistration>>,
    /// Optional AEGIS-256 TDE module for encrypting/decrypting PAX blocks.
    #[cfg(feature = "aegis-tde")]
    tde: Option<Arc<galaxdb_crypto::AegisTdeModule>>,
}

impl Engine {
    /// Open or create a storage engine at the given path.
    pub fn new(config: EngineConfig) -> GalaxResult<Self> {
        std::fs::create_dir_all(&config.data_dir)?;

        let wal_config = WalWriterConfig {
            wal_path: config.data_dir.join("wal.log"),
            group_commit_interval: Duration::from_millis(config.wal_group_commit_ms),
            checkpoint_size_bytes: 512 * 1024 * 1024,
            checkpoint_interval: Duration::from_secs(60),
            preallocate_bytes: crate::wal::DEFAULT_WAL_PREALLOCATE_BYTES,
        };
        let wal_path_for_replay = config.data_dir.join("wal.log");

        let wal = Arc::new(WalWriter::new(wal_config).map_err(GalaxError::Io)?);

        let memtable_mgr = MemtableManager::new(
            config.memtable_size_bytes,
            config.back_pressure_bytes,
        );
        let art = Arc::new(ArtIndex::new());

        let sst_cache_bytes = config.sst_cache_bytes;

        // Select I/O scheduler: io_uring on Linux 5.10+, tokio elsewhere.
        // This routes all SST reads through the HP queue and flush writes
        // through the BK queue, providing I/O isolation between OLTP and
        // background workloads.
        let io_scheduler: Arc<dyn IoScheduler> = Arc::from(
            galaxdb_io::select_scheduler()
                .map_err(|e| GalaxError::Internal(format!("failed to select I/O scheduler: {}", e)))?
        );
        tracing::info!(backend = ?io_scheduler.backend(), "storage engine I/O scheduler selected");

        // Discover existing `sst_<id>.pax` files in the data
        // directory and register them with the SST registry. This is
        // what makes RESTORE FROM (task 37) + normal engine restart
        // actually see the SSTs on disk. Without this, a restart
        // (or a freshly-restored data dir) would return empty scans
        // even though the files are right there. The SST id is
        // parsed from the filename; `next_sst_id` is advanced past
        // every discovered id so fresh flushes don't collide.
        let mut registry = SstRegistry::with_cache_limit(sst_cache_bytes);
        let mut max_sst_id: u64 = 0;
        for entry in std::fs::read_dir(&config.data_dir).map_err(GalaxError::Io)? {
            let entry = entry.map_err(GalaxError::Io)?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            // Match `sst_<id>.pax`.
            if !(name.starts_with("sst_") && name.ends_with(".pax")) {
                continue;
            }
            let id_str = &name[4..name.len() - 4];
            let Ok(sst_id) = id_str.parse::<u64>() else {
                tracing::warn!(file = %name, "skipping SST with unparsable id during open");
                continue;
            };
            registry.register(sst_id, path);
            if sst_id > max_sst_id {
                max_sst_id = sst_id;
            }
        }
        let next_sst_id_start = max_sst_id.saturating_add(1).max(1);

        // WAL replay (crash recovery). `WalWriter::new` opens the WAL with
        // O_APPEND and does not truncate, so any records written before a
        // restart or crash are still on disk. `recover_wal` reads from the
        // last CHECKPOINT, verifying the XXH3-64 checksum per record and
        // stopping at the first corruption (records before it are kept).
        // We apply each recovered record to the memtable + ART so a row
        // that was committed to the WAL but not yet flushed to an SST is
        // visible again after reopen.
        //
        // This is what makes the engine actually durable across restart
        // and underpins the crash-safety guarantee: a `put_sync` that
        // returned Ok is recoverable even if the process dies before the
        // next flush. Without this step a clean restart silently lost any
        // WAL-only (not-yet-flushed) rows.
        //
        // Timestamps: recovered rows are re-applied under fresh monotonic
        // timestamps allocated from `next_timestamp`. MVCC ordering among
        // recovered rows is preserved because `recover_wal` returns records
        // in WAL (write) order, and we replay them in that order.
        //
        // Payload format: production `RowPut` records carry the single-row
        // `encode_kv` layout (`[key_len:u32][key][value]`), decoded by
        // `decode_kv`. `RowDelete` records carry the raw key bytes. A
        // record whose payload does not decode is logged and skipped
        // rather than aborting recovery, mirroring the WAL's
        // stop-at-corruption contract without discarding valid prior rows.
        let recovered = match crate::wal::recover_wal(&wal_path_for_replay) {
            Ok((records, _next_seq)) => records,
            Err(e) => {
                tracing::error!(error = %e, "WAL recovery read failed; starting with empty memtable");
                Vec::new()
            }
        };

        let mut next_ts: u64 = 1;
        let mut replayed_puts: u64 = 0;
        let mut replayed_deletes: u64 = 0;
        for record in &recovered {
            match record.record_type {
                WalRecordType::RowPut => {
                    let Some((key, value)) = decode_kv(&record.payload) else {
                        tracing::warn!(
                            seq = record.seq_no,
                            "skipping RowPut WAL record with undecodable payload during replay"
                        );
                        continue;
                    };
                    let ts = next_ts;
                    next_ts += 1;
                    let active = memtable_mgr.active();
                    active.put(key.clone(), ts, Some(value));
                    let shard = (xxhash_rust::xxh3::xxh3_64(&key) % 16) as u8;
                    art.insert(key.clone(), RowLocation::Memtable { shard, key });
                    replayed_puts += 1;
                }
                WalRecordType::RowDelete => {
                    let key = record.payload.clone();
                    let ts = next_ts;
                    next_ts += 1;
                    let active = memtable_mgr.active();
                    active.put(key.clone(), ts, None); // tombstone
                    let shard = (xxhash_rust::xxh3::xxh3_64(&key) % 16) as u8;
                    art.insert(key.clone(), RowLocation::Memtable { shard, key });
                    replayed_deletes += 1;
                }
                // Vector delta records are replayed by the vector layer
                // (galaxdb-vector) from the same WAL, not here. Checkpoints
                // carry no row payload. BlobRef records belong to the
                // KV-separation (blob log) layer and are not row data; the
                // blob log owns their replay. None of these affect the
                // row memtable, so the row-replay path skips them.
                WalRecordType::DeltaInsert
                | WalRecordType::DeltaTombstone
                | WalRecordType::Checkpoint
                | WalRecordType::BlobRef => {}
            }
        }

        if replayed_puts > 0 || replayed_deletes > 0 {
            tracing::info!(
                puts = replayed_puts,
                deletes = replayed_deletes,
                "replayed WAL records into memtable on open"
            );
        }

        Ok(Self {
            config,
            memtable_mgr,
            art,
            wal,
            sst_registry: RwLock::new(registry),
            // Resume timestamp allocation past every recovered row so new
            // writes never reuse a recovered row's logical timestamp.
            next_timestamp: AtomicU64::new(next_ts.max(1)),
            next_sst_id: AtomicU64::new(next_sst_id_start),
            row_count: AtomicU64::new(replayed_puts),
            io_scheduler,
            compaction_lock: Mutex::new(()),
            pin_source: RwLock::new(None),
            write_controller: WriteController::new(WriteControllerConfig::default()),
            compaction_wake: Arc::new((Mutex::new((false, false)), Condvar::new())),
            bg_worker_active: AtomicBool::new(false),
            columnar: RwLock::new(Vec::new()),
            #[cfg(feature = "aegis-tde")]
            tde: None,
        })
    }

    /// Allocate a new monotonic timestamp.
    fn next_ts(&self) -> Timestamp {
        self.next_timestamp.fetch_add(1, Ordering::SeqCst)
    }

    /// Peek at the next timestamp without consuming it. Test-only —
    /// provides a stable "snapshot boundary" that lets AT VERSION
    /// tests take a snapshot between two writes. Not exposed on
    /// production code paths because callers should use real
    /// `TagCatalog` snapshots rather than synthetic timestamp peeks.
    #[doc(hidden)]
    pub fn next_ts_for_tests(&self) -> Timestamp {
        self.next_timestamp.load(Ordering::SeqCst)
    }

    /// Return the timestamp most recently allocated by `next_ts`, i.e.
    /// the commit ts of the most recent write that has landed in the
    /// engine.
    ///
    /// Zero if nothing has been written yet. Callers use this to pin
    /// a training snapshot at "everything committed so far" — the
    /// same boundary an external observer sees after
    /// [`Self::put_sync`] / [`Self::delete_sync`] returns.
    pub fn latest_commit_ts(&self) -> Timestamp {
        self.next_timestamp
            .load(Ordering::SeqCst)
            .saturating_sub(1)
    }

    /// Insert a row. Writes to WAL + memtable + ART index.
    pub async fn put(&self, key: Vec<u8>, value: Vec<u8>) -> GalaxResult<Timestamp> {
        let ts = self.next_ts();

        // Write to WAL first (durability)
        let payload = encode_kv(&key, &value);
        self.wal
            .append(WalRecordType::RowPut, payload, DurabilityMode::Relaxed)
            .await
            .map_err(GalaxError::Io)?;

        // Write to memtable
        let active = self.memtable_mgr.active();
        active.put(key.clone(), ts, Some(value));

        // Update ART index
        let shard = (xxhash_rust::xxh3::xxh3_64(&key) % 16) as u8;
        self.art.insert(
            key.clone(),
            RowLocation::Memtable {
                shard,
                key: key.clone(),
            },
        );

        self.row_count.fetch_add(1, Ordering::Relaxed);
        Ok(ts)
    }

    /// Insert a row synchronously (memtable + ART + WAL sync).
    /// Use this from sync contexts (embedded mode, Python FFI).
    pub fn put_sync(&self, key: Vec<u8>, value: Vec<u8>) -> GalaxResult<Timestamp> {
        self.admit_write();
        let ts = self.next_ts();

        // Write to WAL first (durability — sync fsync)
        let payload = encode_kv(&key, &value);
        self.wal
            .append_sync(WalRecordType::RowPut, payload)
            .map_err(GalaxError::Io)?;

        // Write to memtable
        let active = self.memtable_mgr.active();
        active.put(key.clone(), ts, Some(value));

        // Update ART index
        let shard = (xxhash_rust::xxh3::xxh3_64(&key) % 16) as u8;
        self.art.insert(
            key.clone(),
            RowLocation::Memtable {
                shard,
                key: key.clone(),
            },
        );

        self.row_count.fetch_add(1, Ordering::Relaxed);
        Ok(ts)
    }

    /// Insert multiple rows in a single batch (one WAL entry, one fsync).
    /// This is the fast path for multi-row INSERT statements.
    pub fn put_batch_sync(&self, rows: &[(Vec<u8>, Vec<u8>)]) -> GalaxResult<u64> {
        if rows.is_empty() {
            return Ok(0);
        }
        self.admit_write();

        // Build a single WAL payload containing all rows
        let mut batch_payload = Vec::with_capacity(rows.len() * 128);
        let row_count = rows.len() as u32;
        batch_payload.extend_from_slice(&row_count.to_le_bytes());
        for (key, value) in rows {
            batch_payload.extend_from_slice(&(key.len() as u32).to_le_bytes());
            batch_payload.extend_from_slice(key);
            batch_payload.extend_from_slice(&(value.len() as u32).to_le_bytes());
            batch_payload.extend_from_slice(value);
        }

        // Single WAL write + single fsync for the entire batch
        self.wal
            .append_sync(WalRecordType::RowPut, batch_payload)
            .map_err(GalaxError::Io)?;

        // Write all rows to memtable + ART
        let active = self.memtable_mgr.active();
        for (key, value) in rows {
            let ts = self.next_ts();
            active.put(key.clone(), ts, Some(value.clone()));

            let shard = (xxhash_rust::xxh3::xxh3_64(key) % 16) as u8;
            self.art.insert(
                key.clone(),
                RowLocation::Memtable {
                    shard,
                    key: key.clone(),
                },
            );
        }

        let count = rows.len() as u64;
        self.row_count.fetch_add(count, Ordering::Relaxed);
        Ok(count)
    }

    /// Get a row by primary key. Checks memtable first, then SST files on disk.
    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        // Hold the SST registry read lock across the ART lookup AND the SST
        // read. Runtime compaction performs its ART relocation + registry
        // swap under the registry *write* lock (same registry-before-ART
        // lock order), so a reader observes either the fully pre-compaction
        // state (old ART entry + old SST present) or the fully
        // post-compaction state (new ART entry + new SST present) — never a
        // relocated-but-deleted in-between that would surface a false miss.
        let registry = self.sst_registry.read().ok()?;
        let location = self.art.lookup(key)?;

        match &location {
            RowLocation::Memtable { .. } => {
                // Read from memtable (checks active + sealed)
                match self.memtable_mgr.get(key) {
                    Some(Some(value)) => Some(value),
                    Some(None) => None, // tombstone
                    None => None,
                }
            }
            RowLocation::SST { sst_id, block_offset, row_offset } => {
                // Read from SST file via IoScheduler (io_uring HP queue on Linux)
                #[cfg(feature = "aegis-tde")]
                {
                    registry.read_value(
                        *sst_id,
                        *block_offset,
                        *row_offset,
                        self.tde.as_deref(),
                        self.io_scheduler.as_ref(),
                    )
                }
                #[cfg(not(feature = "aegis-tde"))]
                {
                    registry.read_value(
                        *sst_id,
                        *block_offset,
                        *row_offset,
                        self.io_scheduler.as_ref(),
                    )
                }
            }
        }
    }

    /// Register a table's key prefix for columnar storage (HTAP ADR-0002).
    ///
    /// After registration, flush writes one typed PAX column per SQL column
    /// for rows under `prefix`, in addition to the legacy `[key, value, ts]`
    /// columns (which keep every existing read/compaction path working). The
    /// SQL layer calls this on CREATE TABLE for a `Columnar`-mode table and
    /// on catalog reload. Re-registering the same prefix replaces the
    /// splitter.
    pub fn register_columnar_table(&self, prefix: Vec<u8>, splitter: Arc<dyn RowColumnSplitter>) {
        let mut regs = self.columnar.write().expect("columnar registry lock");
        regs.retain(|r| r.prefix != prefix);
        regs.push(ColumnarRegistration { prefix, splitter });
    }

    /// Remove a columnar registration (e.g. on DROP TABLE). No-op if absent.
    pub fn unregister_columnar_table(&self, prefix: &[u8]) {
        let mut regs = self.columnar.write().expect("columnar registry lock");
        regs.retain(|r| r.prefix != prefix);
    }

    /// Snapshot the current columnar registrations for a flush/compaction run.
    fn columnar_registrations(&self) -> Vec<ColumnarRegistration> {
        self.columnar
            .read()
            .map(|r| r.clone())
            .unwrap_or_default()
    }

    /// Flush the active memtable to an SST file on disk.
    /// Updates ART entries to point to the SST instead of the memtable.
    pub async fn flush_memtable(&self) -> GalaxResult<u64> {
        let active = self.memtable_mgr.active();
        let entries = active.iter_all();

        if entries.is_empty() {
            return Ok(0);
        }

        let flush_start = std::time::Instant::now();

        let sst_id = self.next_sst_id.fetch_add(1, Ordering::SeqCst);
        let flush_config = FlushConfig {
            data_dir: self.config.data_dir.clone(),
            sst_size_bytes: self.config.sst_size_bytes,
            max_rows_per_block: self.config.max_rows_per_sst,
        };

        let result = {
            let regs = self.columnar_registrations();
            #[cfg(feature = "aegis-tde")]
            {
                if let Some(tde) = &self.tde {
                    flush::flush_memtable_encrypted(&active, &flush_config, sst_id, tde, self.io_scheduler.as_ref(), &regs).await?
                } else {
                    flush::flush_memtable(&active, &flush_config, sst_id, self.io_scheduler.as_ref(), &regs).await?
                }
            }
            #[cfg(not(feature = "aegis-tde"))]
            {
                flush::flush_memtable(&active, &flush_config, sst_id, self.io_scheduler.as_ref(), &regs).await?
            }
        };

        // Register SST files and update ART entries using the block_map.
        // The flush pipeline now packs multiple small PAX blocks (~100 rows, ~64KB)
        // into each SST file with a block index at the end. The ART stores
        // (sst_id, block_offset, row_offset) so point reads can pread just the
        // specific block instead of the entire SST file.
        {
            let mut registry = self.sst_registry.write()
                .map_err(|_| GalaxError::Internal("sst registry lock".to_string()))?;

            // Register each SST file (loads block index from footer)
            for (sst_idx, sst_path) in result.sst_paths.iter().enumerate() {
                let file_sst_id = if sst_idx == 0 {
                    sst_id
                } else {
                    self.next_sst_id.fetch_add(1, Ordering::SeqCst)
                };

                #[cfg(feature = "aegis-tde")]
                {
                    if let Some(tde) = &self.tde {
                        registry.register_encrypted(file_sst_id, sst_path.clone(), tde);
                    } else {
                        registry.register(file_sst_id, sst_path.clone());
                    }
                }
                #[cfg(not(feature = "aegis-tde"))]
                {
                    registry.register(file_sst_id, sst_path.clone());
                }
            }

            // Update ART entries using block_map from the flush result.
            // Each entry in block_map tells us: which SST, which block within it,
            // and how many rows. We map this to ART's (sst_id, block_offset, row_offset).
            let mut global_row_idx = 0usize;
            for block_info in &result.block_map {
                // Resolve the sst_id for this block's SST file
                let file_sst_id = if block_info.sst_index == 0 {
                    sst_id
                } else {
                    // The sst_ids were allocated sequentially during registration above
                    sst_id + block_info.sst_index as u64
                };

                for local_row in 0..block_info.row_count {
                    if global_row_idx < entries.len() {
                        let (key, _) = &entries[global_row_idx];
                        self.art.insert(
                            key.clone(),
                            RowLocation::SST {
                                sst_id: file_sst_id,
                                block_offset: block_info.block_index as u64,
                                row_offset: local_row as u32,
                            },
                        );
                        global_row_idx += 1;
                    }
                }
            }
        }

        // Seal the active memtable and swap in a new one.
        // This must go through MemtableManager so it properly:
        // 1. Marks the memtable as sealed
        // 2. Adds it to the sealed queue
        // 3. Swaps in a new empty active memtable
        // Without this, subsequent writes would be silently dropped
        // because Memtable::put() rejects writes to sealed memtables.
        self.memtable_mgr.seal_active();
        self.memtable_mgr.on_flush_complete(active.size());

        // Task 38.3: publish flush (a.k.a. checkpoint) duration in ms.
        let elapsed_ms = flush_start.elapsed().as_millis() as i64;
        galaxdb_observe::metrics()
            .checkpoint_last_duration_ms
            .set(elapsed_ms);

        // Runtime compaction (Req 12 knob consumer): once the on-disk SST
        // count reaches the configured L0 trigger, merge the SSTs so file
        // count and read amplification stay bounded instead of growing with
        // every flush. When a background worker is running it does the merge
        // off the flush path (so flush never blocks); otherwise we compact
        // inline. A compaction failure must not fail the flush — the rows
        // are already durably on disk — so it is logged, not propagated.
        if self.bg_worker_active.load(Ordering::Relaxed) {
            self.signal_compaction();
        } else if let Err(e) = self.maybe_compact() {
            tracing::warn!(error = %e, "runtime compaction after flush failed");
        }
        self.refresh_compaction_backlog();

        Ok(result.rows_flushed as u64)
    }

    /// Delete a row by primary key. Writes tombstone to WAL + memtable.
    pub async fn delete(&self, key: &[u8]) -> GalaxResult<bool> {
        // Check if key exists
        if self.art.lookup(key).is_none() {
            return Ok(false);
        }

        let ts = self.next_ts();

        // Write tombstone to WAL
        self.wal
            .append(
                WalRecordType::RowDelete,
                key.to_vec(),
                DurabilityMode::Relaxed,
            )
            .await
            .map_err(GalaxError::Io)?;

        // Write tombstone to memtable
        let active = self.memtable_mgr.active();
        active.put(key.to_vec(), ts, None);

        // Remove from ART
        self.art.delete(key);

        Ok(true)
    }

    /// Delete a row by primary key, synchronously (sync fsync).
    ///
    /// Mirror of [`Engine::delete`] for callers that run outside a tokio
    /// runtime (embedded mode, the SQL executor's DELETE arm). Writes
    /// the tombstone WAL record via `append_sync`, then updates the
    /// memtable and ART. Returns `Ok(true)` if the key was present,
    /// `Ok(false)` if there was nothing to delete.
    pub fn delete_sync(&self, key: &[u8]) -> GalaxResult<bool> {
        if self.art.lookup(key).is_none() {
            return Ok(false);
        }

        let ts = self.next_ts();

        self.wal
            .append_sync(WalRecordType::RowDelete, key.to_vec())
            .map_err(GalaxError::Io)?;

        let active = self.memtable_mgr.active();
        active.put(key.to_vec(), ts, None);

        self.art.delete(key);

        Ok(true)
    }

    /// Append a `DELTA_INSERT` record to the WAL so the vector delta
    /// buffer is rebuildable on crash recovery. Task 18.3 / 24.2.
    ///
    /// The payload is the application-defined encoding of `(row_id,
    /// vector_bytes)`. This method does NOT decode or validate it; it
    /// only guarantees durability. Recovery code in
    /// `galaxdb-vector::delta_buffer` interprets the payload.
    pub fn append_delta_insert_sync(&self, payload: Vec<u8>) -> GalaxResult<u64> {
        self.wal
            .append_sync(WalRecordType::DeltaInsert, payload)
            .map_err(GalaxError::Io)
    }

    /// Append a `DELTA_TOMBSTONE` record to the WAL. Task 18.6 / 24.2.
    ///
    /// Called by the SQL executor's DELETE arm (via the vector backend
    /// trait's `on_row_deleted` hook) when a row in a table with an
    /// embedding column is removed. The payload format mirrors
    /// `append_delta_insert_sync`; recovery will replay tombstones
    /// into the in-memory delta buffer so tombstoned rows stay hidden
    /// from SEMANTIC_MATCH results across restarts.
    pub fn append_delta_tombstone_sync(&self, payload: Vec<u8>) -> GalaxResult<u64> {
        self.wal
            .append_sync(WalRecordType::DeltaTombstone, payload)
            .map_err(GalaxError::Io)
    }

    /// Scan all rows with optional key-range filtering via SST
    /// zone-map pruning. Tasks 18.4 / 5.6.
    ///
    /// When `key_prefix` is `Some`, SST blocks whose `[zone_map_min,
    /// zone_map_max]` on the key column cannot overlap keys starting
    /// with the prefix are skipped entirely — saving a full block
    /// deserialization and column read. The memtable path can't use
    /// zone maps (no per-memtable-shard min/max tracking yet) so
    /// memtable scan still iterates every entry and matches prefix
    /// per-key.
    ///
    /// When `key_prefix` is `None`, this is equivalent to `scan_all`.
    pub fn scan_all_with_prefix(
        &self,
        key_prefix: Option<&[u8]>,
    ) -> Vec<(Vec<u8>, Vec<u8>)> {
        // Track the per-key winning version with its MVCC timestamp so that
        // when the same key appears in multiple SST blocks (e.g. an update
        // flushed after the original insert, before they are compacted) the
        // newest version wins regardless of the registry's iteration order.
        let mut out: HashMap<Vec<u8>, (Vec<u8>, Timestamp)> = HashMap::new();
        // Keys whose newest visible version is a tombstone — kept so a later
        // (older-ts) block cannot resurrect a deleted key.
        let mut tombstoned: HashMap<Vec<u8>, Timestamp> = HashMap::new();

        if let Ok(registry) = self.sst_registry.read() {
            for (_sst_id, entry) in registry.entries.iter() {
                let Ok(data) = std::fs::read(&entry.path) else {
                    continue;
                };
                for block_entry in &entry.block_index.entries {
                    let start = block_entry.file_offset as usize;
                    let end = start + block_entry.block_len as usize;
                    if end > data.len() {
                        continue;
                    }
                    let block_bytes = &data[start..end];
                    let block = match crate::pax::PaxBlock::deserialize(block_bytes) {
                        Ok(b) => b,
                        Err(_) => continue,
                    };

                    // Zone-map pruning against the key column (col 0).
                    if let Some(prefix) = key_prefix {
                        if block.header.column_descriptors.is_empty() {
                            continue;
                        }
                        let desc = &block.header.column_descriptors[0];
                        if !key_range_overlaps_prefix(
                            &desc.zone_map_min,
                            &desc.zone_map_max,
                            prefix,
                        ) {
                            continue; // Block can't contain any matching key.
                        }
                    }

                    let header_ts = block.header.commit_timestamp;
                    let keys = match block.read_column(0) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    let vals = match block.read_column(1) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    let row_ts: Vec<Timestamp> = if block.header.column_count >= 3 {
                        match block.read_column(crate::flush::ROW_TS_COLUMN) {
                            Ok(col) => col
                                .iter()
                                .map(|b| crate::flush::decode_row_ts(b).unwrap_or(header_ts))
                                .collect(),
                            Err(_) => vec![header_ts; keys.len()],
                        }
                    } else {
                        vec![header_ts; keys.len()]
                    };
                    for (idx, (k, v)) in keys.into_iter().zip(vals).enumerate() {
                        if let Some(prefix) = key_prefix {
                            if !k.starts_with(prefix) {
                                continue;
                            }
                        }
                        let ts = row_ts.get(idx).copied().unwrap_or(header_ts);
                        // A newer version (live or tombstone) for this key
                        // always wins over an older one.
                        let live_ts = out.get(&k).map(|(_, t)| *t);
                        let tomb_ts = tombstoned.get(&k).copied();
                        let newest_seen = live_ts.max(tomb_ts).unwrap_or(0);
                        if ts < newest_seen {
                            continue;
                        }
                        if v.is_empty() {
                            out.remove(&k);
                            tombstoned.insert(k, ts);
                        } else {
                            tombstoned.remove(&k);
                            out.insert(k, (v, ts));
                        }
                    }
                }
            }
        }

        let mut out: HashMap<Vec<u8>, Vec<u8>> =
            out.into_iter().map(|(k, (v, _))| (k, v)).collect();

        let active = self.memtable_mgr.active();
        let mem_entries = match key_prefix {
            // Bounded prefix range scan — O(log n + matches) per shard,
            // not a full O(table-size) iteration. This is what keeps
            // per-row secondary-index/grant prefix lookups off the
            // O(n)-per-call path (otherwise ingest is O(n^2)).
            Some(prefix) => active.iter_prefix(prefix),
            None => active.iter_all(),
        };
        // The active memtable always holds versions newer than any SST, so
        // its entries override the SST-resolved values unconditionally.
        for (key, versioned) in mem_entries {
            match versioned.value {
                Some(v) => {
                    out.insert(key, v);
                }
                None => {
                    out.remove(&key);
                }
            }
        }

        let mut results: Vec<(Vec<u8>, Vec<u8>)> = out.into_iter().collect();
        results.sort_by(|a, b| a.0.cmp(&b.0));
        results
    }

    /// Scan all rows (for SELECT * without filter). Returns keys and
    /// values from the memtable AND all registered SST files, sorted
    /// by key in ascending order.
    ///
    /// Rows flushed to SST are returned with their stored values; memtable
    /// rows override SST rows for the same key (because the ART index
    /// already points flushed keys at SST, so they wouldn't be in the
    /// active memtable anyway — but a freshly-updated key is). Tombstones
    /// are honoured in both layers.
    pub fn scan_all(&self) -> Vec<(Vec<u8>, Vec<u8>)> {
        self.scan_all_with_prefix(None)
    }

    /// Scan all rows visible at `read_ts` (AT VERSION timestamp support,
    /// task 32.3).
    ///
    /// Walks each key's MVCC chain in the memtable, returning the latest
    /// version with `timestamp <= read_ts`. Also scans every registered
    /// SST file: each SST block carries a `commit_timestamp` in its
    /// header, and the flush pipeline only writes blocks containing
    /// committed-and-sealed data, so every key in an SST with
    /// `block.commit_timestamp <= read_ts` is visible at `read_ts`.
    /// Tombstones in both layers are honoured.
    pub fn scan_all_at(&self, read_ts: Timestamp) -> Vec<(Vec<u8>, Vec<u8>, Timestamp)> {
        let mut out: HashMap<Vec<u8>, (Vec<u8>, Timestamp)> = HashMap::new();

        // SSTs first. For each block, skip if its commit_timestamp is
        // strictly greater than read_ts — those rows committed in the
        // future relative to the caller's snapshot and must not be
        // visible.
        if let Ok(registry) = self.sst_registry.read() {
            for (_sst_id, entry) in registry.entries.iter() {
                let Ok(data) = std::fs::read(&entry.path) else {
                    continue;
                };
                for block_entry in &entry.block_index.entries {
                    let start = block_entry.file_offset as usize;
                    let end = start + block_entry.block_len as usize;
                    if end > data.len() {
                        continue;
                    }
                    let block_bytes = &data[start..end];
                    let block = match crate::pax::PaxBlock::deserialize(block_bytes) {
                        Ok(b) => b,
                        Err(_) => continue,
                    };
                    let header_ts = block.header.commit_timestamp;
                    // Block-level fast skip: the header timestamp is the
                    // newest MVCC commit in the block, so if it is already
                    // beyond the snapshot, no row in the block is visible.
                    if header_ts > read_ts && block.header.column_count < 3 {
                        // Legacy 2-column block carries no per-row timestamps;
                        // the header is the only timestamp, so skip wholesale.
                        continue;
                    }
                    let keys = match block.read_column(0) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    let vals = match block.read_column(1) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    // Per-row MVCC timestamps (column 2) when present; legacy
                    // two-column blocks fall back to the block header ts.
                    let row_ts: Vec<Timestamp> = if block.header.column_count >= 3 {
                        match block.read_column(crate::flush::ROW_TS_COLUMN) {
                            Ok(col) => col
                                .iter()
                                .map(|b| crate::flush::decode_row_ts(b).unwrap_or(header_ts))
                                .collect(),
                            Err(_) => vec![header_ts; keys.len()],
                        }
                    } else {
                        vec![header_ts; keys.len()]
                    };
                    for (idx, (k, v)) in keys.into_iter().zip(vals).enumerate() {
                        let ts = row_ts.get(idx).copied().unwrap_or(header_ts);
                        // Rows committed after the snapshot are invisible.
                        if ts > read_ts {
                            continue;
                        }
                        // Merge by taking the latest visible version per key.
                        let existing_ts = out.get(&k).map(|(_, ts)| *ts).unwrap_or(0);
                        if ts >= existing_ts {
                            if v.is_empty() {
                                out.remove(&k);
                            } else {
                                out.insert(k, (v, ts));
                            }
                        }
                    }
                }
            }
        }

        // Memtable — walks MVCC chains and returns the version at or
        // before read_ts. Overrides SSTs for the same key when the
        // memtable has a newer-but-still-visible version.
        let active = self.memtable_mgr.active();
        for (key, versioned) in active.iter_all() {
            if let Some((maybe_val, ts)) = versioned.get_at_with_ts(read_ts) {
                if ts >= out.get(&key).map(|(_, t)| *t).unwrap_or(0) {
                    match maybe_val {
                        Some(v) => {
                            out.insert(key, (v, ts));
                        }
                        None => {
                            out.remove(&key);
                        }
                    }
                }
            }
        }

        let mut results: Vec<(Vec<u8>, Vec<u8>, Timestamp)> = out
            .into_iter()
            .map(|(k, (v, ts))| (k, v, ts))
            .collect();
        results.sort_by(|a, b| a.0.cmp(&b.0));
        results
    }

    /// Column-major scan of a columnar table visible at `read_ts` (HTAP
    /// task 7 — the reader of the columnar write path).
    ///
    /// Returns the projected SQL columns of every live row, reading typed
    /// PAX columns **directly** from columnar SST blocks (no per-row string
    /// parse — the OLAP fast path) and decoding the row blob via the
    /// registered splitter only for memtable rows and legacy/non-columnar
    /// blocks (the migration bridge, HTAP task 8). Rows come out sorted by
    /// primary key.
    ///
    /// * `prefix` — the table's key prefix (`"table:"`); must have a
    ///   columnar registration (else an error).
    /// * `projection` — indices into the table's SQL columns; empty = all.
    /// * `predicates` — conjuncts used only for **zone-map block pruning**
    ///   (an I/O optimization). Pruning is MVCC-safe: a pruned block's rows
    ///   still record their key+timestamp so an older matching version
    ///   elsewhere cannot resurface. Row-level filtering is the caller's
    ///   job (DataFusion re-checks), so predicates are treated as inexact.
    pub fn scan_columnar(
        &self,
        prefix: &[u8],
        projection: &[usize],
        predicates: &[crate::columnar::ColumnPredicate],
        read_ts: Timestamp,
    ) -> GalaxResult<crate::columnar::ColumnarBatch> {
        use crate::columnar::{data_column_index, is_valid_marker, validity_column_index, ColumnarBatch};
        use galaxdb_common::ColumnType;

        let reg = {
            let regs = self
                .columnar
                .read()
                .map_err(|_| GalaxError::Internal("columnar registry lock".into()))?;
            regs.iter().find(|r| r.prefix == prefix).cloned()
        }
        .ok_or_else(|| {
            GalaxError::Internal(format!(
                "no columnar registration for prefix {}",
                String::from_utf8_lossy(prefix)
            ))
        })?;

        let splitter = reg.splitter.as_ref();
        let col_types = splitter.column_types();
        let n = col_types.len();
        let proj: Vec<usize> = if projection.is_empty() {
            (0..n).collect()
        } else {
            projection.to_vec()
        };
        for &p in &proj {
            if p >= n {
                return Err(GalaxError::Internal(format!(
                    "projection index {p} out of range (table has {n} columns)"
                )));
            }
        }
        let columnar_col_count = crate::columnar::FIRST_DATA_COLUMN + 2 * n;

        // A staged row outcome, merged per key by max timestamp.
        enum Staged {
            Tombstone,
            /// In a zone-map-pruned block: definitely fails the predicate,
            /// but its timestamp must still suppress older versions.
            Excluded,
            Cells(Vec<Option<Vec<u8>>>),
        }
        let mut winners: BTreeMap<Vec<u8>, (Timestamp, Staged)> = BTreeMap::new();
        let consider = |winners: &mut BTreeMap<Vec<u8>, (Timestamp, Staged)>,
                        key: Vec<u8>,
                        ts: Timestamp,
                        staged: Staged| {
            match winners.get(&key) {
                Some((cur, _)) if *cur > ts => {}
                _ => {
                    winners.insert(key, (ts, staged));
                }
            }
        };
        let project = |full: &[Option<Vec<u8>>]| -> Vec<Option<Vec<u8>>> {
            proj.iter().map(|&c| full.get(c).cloned().flatten()).collect()
        };

        if let Ok(registry) = self.sst_registry.read() {
            for (_sst_id, entry) in registry.entries.iter() {
                let Ok(data) = std::fs::read(&entry.path) else {
                    continue;
                };
                for be in &entry.block_index.entries {
                    let start = be.file_offset as usize;
                    let end = start + be.block_len as usize;
                    if end > data.len() {
                        continue;
                    }
                    let block = match crate::pax::PaxBlock::deserialize(&data[start..end]) {
                        Ok(b) => b,
                        Err(_) => continue,
                    };
                    let header_ts = block.header.commit_timestamp;
                    let Ok(keys) = block.read_column(0) else { continue };
                    let Ok(vals) = block.read_column(1) else { continue };
                    let row_ts: Vec<Timestamp> = if block.header.column_count >= 3 {
                        block
                            .read_column(crate::flush::ROW_TS_COLUMN)
                            .map(|col| {
                                col.iter()
                                    .map(|b| crate::flush::decode_row_ts(b).unwrap_or(header_ts))
                                    .collect()
                            })
                            .unwrap_or_else(|_| vec![header_ts; keys.len()])
                    } else {
                        vec![header_ts; keys.len()]
                    };

                    let is_columnar = block.header.column_count as usize == columnar_col_count;

                    // Zone-map block pruning (only meaningful for columnar
                    // blocks, which carry per-SQL-column zone maps).
                    let pruned = is_columnar
                        && !predicates.is_empty()
                        && predicates.iter().any(|p| {
                            match block
                                .header
                                .column_descriptors
                                .get(data_column_index(p.column))
                            {
                                Some(d) => !crate::pax::zone_map_can_match(
                                    &col_types[p.column],
                                    &d.zone_map_min,
                                    &d.zone_map_max,
                                    p.op,
                                    &p.value,
                                ),
                                None => false,
                            }
                        });

                    // Pre-read projected typed columns + validity (skip the
                    // heavy data reads when the block is pruned).
                    let mut proj_data: Vec<Vec<Vec<u8>>> = Vec::new();
                    let mut proj_valid: Vec<Vec<Vec<u8>>> = Vec::new();
                    let use_columnar = is_columnar && !pruned && {
                        let mut ok = true;
                        for &c in &proj {
                            match (
                                block.read_column(data_column_index(c)),
                                block.read_column(validity_column_index(c)),
                            ) {
                                (Ok(d), Ok(v)) => {
                                    proj_data.push(d);
                                    proj_valid.push(v);
                                }
                                _ => {
                                    ok = false;
                                    break;
                                }
                            }
                        }
                        ok && proj_data.len() == proj.len()
                    };

                    for (idx, key) in keys.iter().enumerate() {
                        if !key.starts_with(prefix) {
                            continue;
                        }
                        let ts = row_ts.get(idx).copied().unwrap_or(header_ts);
                        if ts > read_ts {
                            continue;
                        }
                        let is_tomb = vals.get(idx).map(|v| v.is_empty()).unwrap_or(true);
                        if is_tomb {
                            consider(&mut winners, key.clone(), ts, Staged::Tombstone);
                            continue;
                        }
                        if pruned {
                            consider(&mut winners, key.clone(), ts, Staged::Excluded);
                            continue;
                        }
                        let cells = if use_columnar {
                            (0..proj.len())
                                .map(|j| {
                                    let valid = proj_valid[j]
                                        .get(idx)
                                        .map(|b| is_valid_marker(b))
                                        .unwrap_or(false);
                                    if valid {
                                        proj_data[j].get(idx).cloned()
                                    } else {
                                        None
                                    }
                                })
                                .collect()
                        } else {
                            match splitter.split(&vals[idx]) {
                                Some(full) => project(&full),
                                None => continue,
                            }
                        };
                        consider(&mut winners, key.clone(), ts, Staged::Cells(cells));
                    }
                }
            }
        }

        // Memtable rows (sealed-but-unflushed are out of scope here,
        // mirroring scan_all_at): decode the blob via the splitter (bridge).
        let active = self.memtable_mgr.active();
        for (key, versioned) in active.iter_all() {
            if !key.starts_with(prefix) {
                continue;
            }
            if let Some((maybe_val, ts)) = versioned.get_at_with_ts(read_ts) {
                match maybe_val {
                    Some(v) => {
                        if let Some(full) = splitter.split(&v) {
                            consider(&mut winners, key.clone(), ts, Staged::Cells(project(&full)));
                        }
                    }
                    None => consider(&mut winners, key.clone(), ts, Staged::Tombstone),
                }
            }
        }

        // Assemble column-major output (BTreeMap → sorted by key).
        let mut columns: Vec<(ColumnType, Vec<Option<Vec<u8>>>)> =
            proj.iter().map(|&c| (col_types[c].clone(), Vec::new())).collect();
        let mut num_rows = 0;
        for (_key, (_ts, staged)) in winners {
            if let Staged::Cells(cells) = staged {
                for (j, cell) in cells.into_iter().enumerate() {
                    columns[j].1.push(cell);
                }
                num_rows += 1;
            }
        }
        Ok(ColumnarBatch { num_rows, columns })
    }

    /// Get the total number of rows (approximate).
    pub fn row_count(&self) -> u64 {
        self.row_count.load(Ordering::Relaxed)
    }

    /// Per-row content checksums for the committed snapshot visible at
    /// `read_ts`, one `xxh3_64(key ‖ value)` per live row.
    ///
    /// This is the raw material for a real version-tag Merkle root: the
    /// caller folds these with `MerkleRoot::compute`, which sorts and
    /// hashes them into a 128-bit digest. Because it is derived from the
    /// exact rows `scan_all_at` (and therefore `AT VERSION`) returns, the
    /// root is reproducible and actually certifies the tagged snapshot's
    /// contents — not a placeholder constant.
    pub fn snapshot_checksums(&self, read_ts: Timestamp) -> Vec<u64> {
        self.scan_all_at(read_ts)
            .into_iter()
            .map(|(k, v, _ts)| {
                let mut buf = Vec::with_capacity(k.len() + v.len() + 1);
                buf.extend_from_slice(&k);
                buf.push(0xff); // separator so (k‖v) is unambiguous
                buf.extend_from_slice(&v);
                xxhash_rust::xxh3::xxh3_64(&buf)
            })
            .collect()
    }

    /// Get the ART index entry count.
    pub fn index_count(&self) -> usize {
        self.art.len()
    }

    /// Get the data directory path.
    pub fn data_dir(&self) -> &Path {
        &self.config.data_dir
    }

    /// Back up the engine's on-disk state to `target_dir` (Req 27 /
    /// task 37).
    ///
    /// Flushes the active memtable to an SST (so the WAL reflects a
    /// clean checkpoint), then copies every `sst_*.pax` file and
    /// `wal.log` to `target_dir`. The target directory is created
    /// on demand; if it already contains files with matching names
    /// they're overwritten — callers who need retention should write
    /// to a fresh directory per backup (e.g. `backups/<ts>/`).
    ///
    /// The write-quiesce window is the duration of `flush_memtable`
    /// plus the per-file copy. For the memtable sizes the v1 engine
    /// targets (64 MB default seal threshold) this is well under
    /// 100 ms on NVMe. No quiesce is imposed during the file copy
    /// itself because SST files are immutable once written and the
    /// WAL file is append-only — concurrent writes simply extend the
    /// WAL beyond the offset we captured, which the restore path
    /// replays on next open without double-counting.
    ///
    /// Returns the list of files copied on success.
    pub async fn backup_to(&self, target_dir: &Path) -> GalaxResult<Vec<PathBuf>> {
        std::fs::create_dir_all(target_dir).map_err(GalaxError::Io)?;

        // 1. Flush active memtable so the SST set reflects every
        // acknowledged write. This is the "clean Merkle root" half of
        // task 37.1 — we can't write PAX blocks that aren't on disk.
        self.flush_memtable().await?;

        // 2. Enumerate the source files. Everything the engine owns
        // lives directly under `data_dir`:
        //   - `wal.log` (single append-only file)
        //   - `sst_<id>.pax` files written by flush+compaction
        //   - `_galaxdb_reserve` (disk-full handler's reserve; safe
        //     to skip — the target engine allocates its own reserve)
        // Blob logs live in a subdirectory owned by upstream
        // `BlobLog`; today the production `Engine` does not yet
        // thread blob log through, so there is nothing to copy here.
        // When that wiring lands this loop grows one branch.
        Self::copy_backup_files(&self.config.data_dir, target_dir)
    }

    /// Sync variant of [`Self::backup_to`] for callers (like the
    /// sync executor) that don't already own a tokio runtime. Spins
    /// a dedicated current-thread runtime for the `flush_memtable`
    /// call and discards it when the backup returns.
    pub fn backup_to_sync(&self, target_dir: &Path) -> GalaxResult<Vec<PathBuf>> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                GalaxError::Internal(format!("BACKUP: tokio runtime build failed: {e}"))
            })?;
        rt.block_on(self.backup_to(target_dir))
    }

    /// Shared helper: copy `wal.log` + every `sst_*.pax` file from
    /// `src_dir` to `target_dir`. Used by both backup and restore.
    fn copy_backup_files(
        src_dir: &Path,
        target_dir: &Path,
    ) -> GalaxResult<Vec<PathBuf>> {
        std::fs::create_dir_all(target_dir).map_err(GalaxError::Io)?;
        let mut copied: Vec<PathBuf> = Vec::new();
        for entry in std::fs::read_dir(src_dir).map_err(GalaxError::Io)? {
            let entry = entry.map_err(GalaxError::Io)?;
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let is_backup_target = name == "wal.log"
                || (name.starts_with("sst_") && name.ends_with(".pax"));
            if !is_backup_target {
                continue;
            }
            if !path.is_file() {
                continue;
            }
            let dst = target_dir.join(name);
            std::fs::copy(&path, &dst).map_err(GalaxError::Io)?;
            copied.push(dst);
        }
        Ok(copied)
    }

    /// Validate every `sst_*.pax` file under `target_dir` by parsing
    /// its block index and deserialising each block. The PAX block
    /// reader checks the XXH3-64 checksum and the magic number on
    /// every block (task 37.5), so a corrupted block surfaces as
    /// `GalaxError::Internal` carrying the filename and block index.
    ///
    /// The WAL is not separately validated here because `recover_wal`
    /// already stops at the first checksum failure and skips the
    /// remainder — the restore path calls it on next engine open.
    ///
    /// Returns `(sst_files_checked, total_blocks_validated)` on
    /// success.
    pub fn validate_backup(target_dir: &Path) -> GalaxResult<(usize, usize)> {
        let mut sst_count = 0usize;
        let mut block_count = 0usize;
        for entry in std::fs::read_dir(target_dir).map_err(GalaxError::Io)? {
            let entry = entry.map_err(GalaxError::Io)?;
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if !(name.starts_with("sst_") && name.ends_with(".pax")) {
                continue;
            }
            let data = std::fs::read(&path).map_err(GalaxError::Io)?;
            let idx = crate::sst::SstBlockIndex::from_file_data(&data).map_err(|e| {
                GalaxError::Internal(format!(
                    "RESTORE: corrupt SST block index in {}: {}",
                    path.display(),
                    e
                ))
            })?;
            for (block_idx, entry) in idx.entries.iter().enumerate() {
                let start = entry.file_offset as usize;
                let end = start + entry.block_len as usize;
                if end > data.len() {
                    return Err(GalaxError::Internal(format!(
                        "RESTORE: block {} in {} overruns file ({} > {})",
                        block_idx,
                        path.display(),
                        end,
                        data.len()
                    )));
                }
                crate::pax::PaxBlock::deserialize(&data[start..end]).map_err(|e| {
                    GalaxError::Internal(format!(
                        "RESTORE: corrupt block {} in {}: {}",
                        block_idx,
                        path.display(),
                        e
                    ))
                })?;
                block_count += 1;
            }
            sst_count += 1;
        }
        Ok((sst_count, block_count))
    }

    /// Restore a backup from `source_dir` into `target_dir`. Expected
    /// to be called on a *fresh* data directory — restoring into a
    /// populated engine is not supported in v1.
    ///
    /// Steps (task 37.4 / 37.5):
    /// 1. Validate every SST block's checksum in `source_dir` via
    ///    [`Engine::validate_backup`]. Abort on the first failure
    ///    without touching `target_dir`.
    /// 2. Create `target_dir` and copy every `sst_*.pax` and
    ///    `wal.log` across.
    ///
    /// WAL replay, ART rebuild, and HNSW rebuild all run when the
    /// caller subsequently opens the restored directory with
    /// [`Engine::new`] — those recovery paths are the ones used on
    /// ordinary startup, so restore is a file-level operation plus
    /// a reopen. Callers that want a single-call experience should
    /// drop the existing engine and construct a new one pointing at
    /// `target_dir`.
    pub fn restore_from(source_dir: &Path, target_dir: &Path) -> GalaxResult<Vec<PathBuf>> {
        // 1. Validate first — abort cleanly before any file copy.
        Self::validate_backup(source_dir)?;

        // 2. Copy files via the shared helper used by `backup_to`.
        Self::copy_backup_files(source_dir, target_dir)
    }

    /// Get the I/O backend in use (IoUring or Tokio).
    pub fn io_backend(&self) -> galaxdb_io::IoBackend {
        self.io_scheduler.backend()
    }

    /// Enable AEGIS-256 TDE encryption for PAX blocks.
    ///
    /// When enabled, all new SST files written by `flush_memtable()` will be
    /// encrypted with AEGIS-256, and reads from SST files will be decrypted.
    #[cfg(feature = "aegis-tde")]
    pub fn enable_tde(&mut self, tde: galaxdb_crypto::AegisTdeModule) {
        self.tde = Some(Arc::new(tde));
    }

    /// Check if TDE is enabled.
    #[cfg(feature = "aegis-tde")]
    pub fn tde_enabled(&self) -> bool {
        self.tde.is_some()
    }

    /// Shutdown the engine cleanly.
    pub fn shutdown(&self) {
        self.wal.shutdown();
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.wal.shutdown();
    }
}

/// Encode a key-value pair for WAL storage.
fn encode_kv(key: &[u8], value: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(4 + key.len() + value.len());
    buf.extend_from_slice(&(key.len() as u32).to_le_bytes());
    buf.extend_from_slice(key);
    buf.extend_from_slice(value);
    buf
}

/// Zone-map pruning helper (task 18.4).
///
/// Returns true if the byte range `[zone_min, zone_max]` could contain
/// a key starting with `prefix`. False means the caller can skip the
/// whole SST block without loading it.
///
/// Logic: the block's keys are bytewise-sorted, so the block contains a
/// key with `prefix` iff `prefix <= zone_max` AND any key of the form
/// `prefix..` is >= zone_min. Concretely:
///
/// * If `zone_max < prefix` bytewise, the block's largest key is
///   strictly less than the prefix — skip.
/// * If `zone_min > prefix_upper_bound`, every key exceeds the prefix
///   namespace — skip. (prefix_upper_bound is the smallest byte
///   string strictly greater than any `prefix..`; computed by
///   incrementing the last byte or pushing 0xFF.)
/// * Otherwise the block might contain a matching key — keep it.
fn key_range_overlaps_prefix(zone_min: &[u8], zone_max: &[u8], prefix: &[u8]) -> bool {
    if zone_min.is_empty() && zone_max.is_empty() {
        // Empty zone map — block is empty or zone maps weren't
        // computed. Be safe: keep it.
        return true;
    }
    // Case 1: zone_max strictly precedes prefix lexicographically.
    if zone_max < prefix {
        return false;
    }
    // Case 2: zone_min exceeds every possible `prefix..` key. The
    // upper bound of the prefix namespace is `prefix` with trailing
    // 0xFF bytes, or equivalently, a key whose first `prefix.len()`
    // bytes are >= prefix and differ at some position.
    //
    // The cleanest check: zone_min starts with bytes > prefix at the
    // first differing position, and is not itself a prefix extension.
    let common = zone_min
        .iter()
        .zip(prefix.iter())
        .take_while(|(a, b)| a == b)
        .count();
    if common < prefix.len() && common < zone_min.len() {
        // They differ inside the prefix range. If zone_min is larger
        // at the first differing byte AND zone_min doesn't extend
        // the prefix, the block's keys are all past the prefix.
        if zone_min[common] > prefix[common] {
            return false;
        }
    }
    true
}

/// Decode a key-value pair from WAL payload.
pub fn decode_kv(payload: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
    if payload.len() < 4 {
        return None;
    }
    let key_len = u32::from_le_bytes(payload[..4].try_into().ok()?) as usize;
    if payload.len() < 4 + key_len {
        return None;
    }
    let key = payload[4..4 + key_len].to_vec();
    let value = payload[4 + key_len..].to_vec();
    Some((key, value))
}

/// One key's version history read out of the input SSTs: `(timestamp,
/// value-or-tombstone)` pairs, oldest first after sorting.
type VersionList = Vec<(Timestamp, Option<Vec<u8>>)>;
/// A surviving row queued for output: `(key, mvcc_timestamp,
/// value-or-tombstone)`. Multiple versions of one key (latest + pinned)
/// appear as separate rows; per-row timestamps are written to the SST's
/// timestamp column so one block can hold rows committed at different times.
type SurvivorRow = (Vec<u8>, Timestamp, Option<Vec<u8>>);

/// One output SST produced by compaction: its raw on-disk bytes and the ART
/// relocation targets for the rows that are the *latest* version of their
/// key (only those need a primary-key index entry — older pinned versions
/// are reachable solely through `scan_all_at`).
struct CompactionOutput {
    bytes: Vec<u8>,
    art_targets: Vec<(Vec<u8>, u32, u32)>,
}

/// Encrypt a block for compaction output when a TDE module is active,
/// mirroring the flush pipeline. The block-index footer is always written
/// in the clear (appended after the encrypted blocks), so re-registration
/// can read it without the key.
#[cfg(feature = "aegis-tde")]
fn compaction_encrypt(
    bytes: &[u8],
    tde: Option<&galaxdb_crypto::AegisTdeModule>,
) -> GalaxResult<Vec<u8>> {
    match tde {
        Some(m) => m.encrypt(bytes),
        None => Ok(bytes.to_vec()),
    }
}

#[cfg(not(feature = "aegis-tde"))]
fn compaction_encrypt(bytes: &[u8], _tde: Option<&()>) -> GalaxResult<Vec<u8>> {
    Ok(bytes.to_vec())
}

/// Build the output SST(s) for one contiguous run of surviving rows
/// (sorted by key, then timestamp). Rows are packed into PAX blocks
/// (≤ `max_rows_per_block` each) and SST files (≤ `sst_size_bytes`) using
/// the same three-column format the flush pipeline writes (key, value,
/// per-row MVCC timestamp), so the read path is identical. A single block
/// can hold rows committed at different timestamps because visibility is
/// resolved per row from the timestamp column. The block header timestamp
/// is the newest row in the block (a real MVCC value), used as a fast
/// upper bound on reads.
#[allow(clippy::too_many_arguments)]
fn build_run(
    rows: &[SurvivorRow],
    latest_ts: &HashMap<Vec<u8>, Timestamp>,
    sst_size_bytes: u64,
    max_rows_per_block: usize,
    registrations: &[ColumnarRegistration],
    #[cfg(feature = "aegis-tde")] tde: Option<&galaxdb_crypto::AegisTdeModule>,
) -> GalaxResult<Vec<CompactionOutput>> {
    use crate::pax::{CodecId, ColumnData, PaxBlock};
    use crate::sst::SstBlockIndex;
    use galaxdb_common::ColumnType;

    let max_rows = max_rows_per_block.max(1);
    let mut outputs: Vec<CompactionOutput> = Vec::new();
    let mut cur_data: Vec<u8> = Vec::new();
    let mut cur_index = SstBlockIndex::new();
    let mut cur_block_count: u32 = 0;
    let mut cur_targets: Vec<(Vec<u8>, u32, u32)> = Vec::new();
    let mut block_id: u64 = 1;

    let mut i = 0;
    while i < rows.len() {
        let mut end = (i + max_rows).min(rows.len());
        // Keep each block to a single table so a columnar block carries one
        // schema. If the run at `i` belongs to a columnar table, stop the
        // block at the first row of a different table; if it is legacy
        // (unregistered), stop before the first row that IS columnar.
        match crate::columnar::registration_for(registrations, &rows[i].0) {
            Some(reg) => {
                if let Some(off) =
                    rows[i..end].iter().position(|(k, _, _)| !reg.matches(k))
                {
                    if off > 0 {
                        end = i + off;
                    }
                }
            }
            None => {
                if let Some(off) = rows[i..end]
                    .iter()
                    .position(|(k, _, _)| crate::columnar::registration_for(registrations, k).is_some())
                {
                    if off > 0 {
                        end = i + off;
                    }
                }
            }
        }
        let chunk = &rows[i..end];

        let mut key_col: Vec<Vec<u8>> = Vec::with_capacity(chunk.len());
        let mut val_col: Vec<Vec<u8>> = Vec::with_capacity(chunk.len());
        let mut ts_col: Vec<Vec<u8>> = Vec::with_capacity(chunk.len());
        let mut block_max_ts: Timestamp = 0;
        for (k, ts, v) in chunk {
            key_col.push(k.clone());
            val_col.push(v.clone().unwrap_or_default());
            ts_col.push(ts.to_le_bytes().to_vec());
            block_max_ts = block_max_ts.max(*ts);
        }
        let mut cols = vec![
            ColumnData { col_type: ColumnType::Blob, values: key_col },
            ColumnData { col_type: ColumnType::Blob, values: val_col },
            ColumnData { col_type: ColumnType::Blob, values: ts_col },
        ];
        let mut codecs = vec![CodecId::Zstd, CodecId::None, CodecId::None];
        // Preserve the columnar layout across compaction: if this block's
        // rows belong to a columnar table, append the typed per-SQL-column
        // chunks, exactly as the flush pipeline does (HTAP ADR-0002). The
        // chunk boundary above already keeps each block to one table, so a
        // single registration applies to every row in the chunk.
        if let Some(reg) = crate::columnar::registration_for(registrations, &chunk[0].0) {
            let values: Vec<Option<Vec<u8>>> =
                chunk.iter().map(|(_, _, v)| v.clone()).collect();
            if let Some((data_cols, data_codecs)) =
                crate::columnar::columnar_data_columns(&values, reg.splitter.as_ref())
            {
                cols.extend(data_cols);
                codecs.extend(data_codecs);
            }
        }
        let pax = PaxBlock::write_with_codecs(
            block_id,
            block_max_ts,
            &cols,
            &codecs,
        )?;
        block_id += 1;
        let block_bytes = pax.serialize()?;
        #[cfg(feature = "aegis-tde")]
        let enc = compaction_encrypt(&block_bytes, tde)?;
        #[cfg(not(feature = "aegis-tde"))]
        let enc = compaction_encrypt(&block_bytes, None)?;

        let offset = cur_data.len() as u64;
        cur_index.add_block(offset, enc.len() as u32);
        // Only the latest version of each key gets an ART entry; an older
        // pinned version of the same key (also in this run) is reachable
        // only through scan_all_at.
        for (row_off, (k, ts, _)) in chunk.iter().enumerate() {
            if latest_ts.get(k) == Some(ts) {
                cur_targets.push((k.clone(), cur_block_count, row_off as u32));
            }
        }
        cur_data.extend_from_slice(&enc);
        cur_block_count += 1;
        i = end;

        if cur_data.len() as u64 >= sst_size_bytes {
            let index_offset = cur_data.len() as u64;
            let footer = cur_index.serialize_with_footer(index_offset);
            cur_data.extend_from_slice(&footer);
            outputs.push(CompactionOutput {
                bytes: std::mem::take(&mut cur_data),
                art_targets: std::mem::take(&mut cur_targets),
            });
            cur_index = SstBlockIndex::new();
            cur_block_count = 0;
        }
    }

    if !cur_data.is_empty() {
        let index_offset = cur_data.len() as u64;
        let footer = cur_index.serialize_with_footer(index_offset);
        cur_data.extend_from_slice(&footer);
        outputs.push(CompactionOutput {
            bytes: cur_data,
            art_targets: cur_targets,
        });
    }

    Ok(outputs)
}

impl Engine {
    /// Install the source of pinned timestamps used by compaction's MVCC
    /// garbage collector (see [`PinSource`]). The embedded `Database`
    /// layer, which owns the version-tag catalog, calls this after open.
    pub fn set_pin_source(&self, src: Arc<dyn PinSource>) {
        *self.pin_source.write().expect("pin_source lock") = Some(src);
    }

    /// Number of SST files currently registered. Observability + tests.
    pub fn sst_count(&self) -> usize {
        self.sst_registry
            .read()
            .map(|r| r.entries.len())
            .unwrap_or(0)
    }

    /// Wake the background compaction worker (if any) to consider a merge.
    fn signal_compaction(&self) {
        let (lock, cvar) = &*self.compaction_wake;
        if let Ok(mut g) = lock.lock() {
            g.0 = true; // wake
            cvar.notify_one();
        }
    }

    /// Recompute the pending-compaction backlog and publish it to the
    /// [`WriteController`]. The estimate is the total on-disk SST bytes once
    /// the file count is past the L0 trigger (i.e. work the compactor still
    /// owes); below the trigger there is no backlog. This is what lets write
    /// admission slow down if compaction falls behind.
    pub fn refresh_compaction_backlog(&self) {
        let pending = {
            match self.sst_registry.read() {
                Ok(reg) => {
                    if reg.entries.len() <= self.config.l0_compaction_trigger {
                        0
                    } else {
                        reg.entries
                            .values()
                            .filter_map(|e| std::fs::metadata(&e.path).ok())
                            .map(|m| m.len())
                            .sum()
                    }
                }
                Err(_) => 0,
            }
        };
        self.write_controller.update_pending_bytes(pending);
    }

    /// Block the calling (synchronous) writer as the [`WriteController`]
    /// dictates: proceed immediately below the soft limit, sleep for a
    /// proportional delay between the soft and hard limits, and spin-wait in
    /// short sleeps while writes are hard-stopped at/above the hard limit.
    /// Returns once the write is admitted.
    fn admit_write(&self) {
        loop {
            match self.write_controller.check_write() {
                WriteAdmission::Proceed => return,
                WriteAdmission::Delay(d) => {
                    std::thread::sleep(d);
                    return;
                }
                WriteAdmission::Block => {
                    std::thread::sleep(Duration::from_millis(1));
                    // Re-check; a background compaction will reduce the
                    // backlog and unblock us.
                }
            }
        }
    }

    /// Borrow the write controller (observability + tests).
    pub fn write_controller(&self) -> &WriteController {
        &self.write_controller
    }

    /// Start a background compaction worker so flushes never block on a
    /// merge. Idempotent. The worker holds only a [`Weak`] reference to the
    /// engine, so it imposes no liveness of its own: once the last strong
    /// `Arc<Engine>` is dropped (or [`shutdown_background_compaction`] is
    /// called) the worker exits. Call after wrapping the engine in an `Arc`
    /// (the embedded `Database` does this at open).
    pub fn start_background_compaction(self: &Arc<Self>) {
        if self
            .bg_worker_active
            .swap(true, Ordering::SeqCst)
        {
            return; // already running
        }
        let weak: Weak<Engine> = Arc::downgrade(self);
        let wake = self.compaction_wake.clone();
        std::thread::Builder::new()
            .name("galaxdb-compaction".to_string())
            .spawn(move || {
                let (lock, cvar) = &*wake;
                loop {
                    // Wait for a wake signal or a periodic timeout (so we
                    // also notice the engine being dropped).
                    let shutdown = {
                        let mut g = match lock.lock() {
                            Ok(g) => g,
                            Err(_) => break,
                        };
                        if !g.0 && !g.1 {
                            let (ng, _timeout) = cvar
                                .wait_timeout(g, Duration::from_millis(500))
                                .expect("compaction condvar");
                            g = ng;
                        }
                        g.0 = false; // consume the wake
                        g.1 // shutdown?
                    };
                    if shutdown {
                        break;
                    }
                    // Upgrade only for the duration of one maybe_compact; if
                    // the engine is gone, exit.
                    let Some(engine) = weak.upgrade() else {
                        break;
                    };
                    if let Err(e) = engine.maybe_compact() {
                        tracing::warn!(error = %e, "background compaction failed");
                    }
                    engine.refresh_compaction_backlog();
                }
            })
            .expect("spawn compaction worker");
    }

    /// Signal the background compaction worker to stop. The worker finishes
    /// any in-flight merge and exits. Safe to call even if no worker runs.
    pub fn shutdown_background_compaction(&self) {
        let (lock, cvar) = &*self.compaction_wake;
        if let Ok(mut g) = lock.lock() {
            g.1 = true; // shutdown
            cvar.notify_all();
        }
        self.bg_worker_active.store(false, Ordering::SeqCst);
    }

    /// Run compaction iff the on-disk SST count has reached the configured
    /// L0 trigger. Uses `try_lock` so a flush that triggers it never blocks
    /// behind an already-running compaction. Returns whether it ran.
    pub fn maybe_compact(&self) -> GalaxResult<bool> {
        let count = self
            .sst_registry
            .read()
            .map_err(|_| GalaxError::Internal("sst registry lock".into()))?
            .entries
            .len();
        if count < self.config.l0_compaction_trigger {
            return Ok(false);
        }
        match self.compaction_lock.try_lock() {
            Ok(_guard) => {
                self.compact_inner()?;
                Ok(true)
            }
            Err(_) => Ok(false), // another compaction is already running
        }
    }

    /// Merge every registered SST into a minimal set, applying MVCC GC.
    /// Blocks until it can acquire the compaction lock. Intended for
    /// explicit callers and tests; the flush path uses [`maybe_compact`].
    pub fn compact(&self) -> GalaxResult<CompactionStats> {
        let _guard = self
            .compaction_lock
            .lock()
            .map_err(|_| GalaxError::Internal("compaction lock poisoned".into()))?;
        self.compact_inner()
    }

    /// Decrypt an SST block read during compaction when it was written
    /// encrypted. Plaintext blocks (and builds without the TDE feature)
    /// pass through unchanged.
    #[cfg(feature = "aegis-tde")]
    fn compaction_decrypt(&self, raw: &[u8], encrypted: bool) -> GalaxResult<Vec<u8>> {
        match (encrypted, self.tde.as_deref()) {
            (true, Some(m)) => m.decrypt(raw),
            _ => Ok(raw.to_vec()),
        }
    }

    /// Core compaction, run while holding the compaction lock.
    fn compact_inner(&self) -> GalaxResult<CompactionStats> {
        // 1. Snapshot the input set (ids + paths + per-file encrypted flag).
        let inputs: Vec<(u64, PathBuf, bool)> = {
            let reg = self
                .sst_registry
                .read()
                .map_err(|_| GalaxError::Internal("sst registry lock".into()))?;
            reg.entries
                .iter()
                .map(|(id, e)| {
                    #[cfg(feature = "aegis-tde")]
                    let enc = e.encrypted;
                    #[cfg(not(feature = "aegis-tde"))]
                    let enc = {
                        let _ = e;
                        false
                    };
                    (*id, e.path.clone(), enc)
                })
                .collect()
        };
        if inputs.len() < 2 {
            return Ok(CompactionStats {
                input_ssts: inputs.len(),
                output_ssts: inputs.len(),
                keys_retained: 0,
                versions_gc_dropped: 0,
            });
        }
        let input_ids: HashSet<u64> = inputs.iter().map(|(id, _, _)| *id).collect();

        // 2. Read every (key -> [(ts, value|tombstone)]) from the inputs.
        let mut versions: BTreeMap<Vec<u8>, VersionList> = BTreeMap::new();
        for (_, path, encrypted) in &inputs {
            let data = std::fs::read(path).map_err(GalaxError::Io)?;
            let index = crate::sst::SstBlockIndex::from_file_data(&data).unwrap_or_else(|_| {
                let mut idx = crate::sst::SstBlockIndex::new();
                idx.add_block(0, data.len() as u32);
                idx
            });
            for be in &index.entries {
                let start = be.file_offset as usize;
                let end = start + be.block_len as usize;
                if end > data.len() {
                    continue;
                }
                #[cfg(feature = "aegis-tde")]
                let block_bytes = self.compaction_decrypt(&data[start..end], *encrypted)?;
                #[cfg(not(feature = "aegis-tde"))]
                let block_bytes = {
                    let _ = encrypted;
                    data[start..end].to_vec()
                };
                let block = match crate::pax::PaxBlock::deserialize(&block_bytes) {
                    Ok(b) => b,
                    Err(_) => continue,
                };
                let header_ts = block.header.commit_timestamp;
                let keys = match block.read_column(0) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let vals = match block.read_column(1) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                // Per-row MVCC timestamps (column 2) when present; legacy
                // two-column SSTs fall back to the block header timestamp.
                let row_ts: Vec<Timestamp> = if block.header.column_count >= 3 {
                    match block.read_column(crate::flush::ROW_TS_COLUMN) {
                        Ok(col) => col
                            .iter()
                            .map(|b| crate::flush::decode_row_ts(b).unwrap_or(header_ts))
                            .collect(),
                        Err(_) => vec![header_ts; keys.len()],
                    }
                } else {
                    vec![header_ts; keys.len()]
                };
                for (idx, (k, v)) in keys.into_iter().zip(vals).enumerate() {
                    let ts = row_ts.get(idx).copied().unwrap_or(header_ts);
                    versions
                        .entry(k)
                        .or_default()
                        .push((ts, if v.is_empty() { None } else { Some(v) }));
                }
            }
        }

        // 3. MVCC GC. Keep, per key: the latest version (always), plus the
        //    newest version at or before each pinned timestamp so AT VERSION
        //    reads still resolve. A key whose latest version is a tombstone
        //    and that no pin needs history for is dropped entirely (it is
        //    deleted and this is the bottom level — nothing below can
        //    resurrect it).
        let mut pins: Vec<Timestamp> = self
            .pin_source
            .read()
            .expect("pin_source lock")
            .as_ref()
            .map(|p| p.pinned_timestamps())
            .unwrap_or_default();
        pins.sort_unstable();
        pins.dedup();

        let mut survivors: Vec<SurvivorRow> = Vec::new();
        let mut latest_ts: HashMap<Vec<u8>, Timestamp> = HashMap::new();
        let mut keys_retained = 0usize;
        let mut versions_gc_dropped = 0usize;

        for (key, mut vers) in versions {
            vers.sort_by_key(|(ts, _)| *ts);
            // Collapse duplicate timestamps (last wins) — defensive; each
            // write gets a distinct MVCC timestamp.
            vers.dedup_by_key(|(ts, _)| *ts);
            let n = vers.len();
            let latest_idx = n - 1;
            let mut keep = vec![false; n];
            keep[latest_idx] = true;
            for &p in &pins {
                if let Some(idx) = vers.iter().rposition(|(ts, _)| *ts <= p) {
                    keep[idx] = true;
                }
            }
            let latest_is_tombstone = vers[latest_idx].1.is_none();
            let only_latest_kept = keep
                .iter()
                .enumerate()
                .all(|(i, &k)| !k || i == latest_idx);
            if latest_is_tombstone && only_latest_kept {
                versions_gc_dropped += n;
                continue;
            }
            keys_retained += 1;
            latest_ts.insert(key.clone(), vers[latest_idx].0);
            for (i, (ts, val)) in vers.into_iter().enumerate() {
                if keep[i] {
                    survivors.push((key.clone(), ts, val));
                } else {
                    versions_gc_dropped += 1;
                }
            }
        }

        // Sort by (key, then timestamp) so each output block holds ascending
        // keys (the read path + zone maps assume this); multiple kept
        // versions of one key sit adjacently. Per-row timestamps live in the
        // SST timestamp column, so one block can hold rows committed at
        // different times — no need to split output by timestamp.
        survivors.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        // 4. Build the output SSTs, splitting the sorted survivor run across
        //    `compaction_concurrency` worker threads (the Req 12 knob). Each
        //    worker owns a contiguous slice, so their outputs never conflict.
        let mut all_outputs: Vec<CompactionOutput> = Vec::new();

        if !survivors.is_empty() {
            let n_workers = self
                .config
                .compaction_concurrency
                .max(1)
                .min(survivors.len());
            let chunk_size = survivors.len().div_ceil(n_workers).max(1);
            let latest_ref = &latest_ts;
            let sst_size = self.config.sst_size_bytes;
            let max_rows = self.config.max_rows_per_sst;
            // Columnar registrations preserved across compaction (HTAP
            // ADR-0002): build_run re-emits typed per-SQL-column chunks so
            // the columnar layout survives merges instead of reverting to
            // legacy 3-column blocks.
            let regs = self.columnar_registrations();
            let regs_ref = regs.as_slice();
            #[cfg(feature = "aegis-tde")]
            let tde_arc = self.tde.clone();

            std::thread::scope(|scope| -> GalaxResult<()> {
                let mut handles = Vec::new();
                for chunk in survivors.chunks(chunk_size) {
                    #[cfg(feature = "aegis-tde")]
                    let tde_arc = tde_arc.clone();
                    let handle = scope.spawn(move || -> GalaxResult<Vec<CompactionOutput>> {
                        #[cfg(feature = "aegis-tde")]
                        {
                            build_run(chunk, latest_ref, sst_size, max_rows, regs_ref, tde_arc.as_deref())
                        }
                        #[cfg(not(feature = "aegis-tde"))]
                        {
                            build_run(chunk, latest_ref, sst_size, max_rows, regs_ref)
                        }
                    });
                    handles.push(handle);
                }
                for handle in handles {
                    let outs = handle
                        .join()
                        .map_err(|_| GalaxError::Internal("compaction worker panicked".into()))??;
                    all_outputs.extend(outs);
                }
                Ok(())
            })?;
        }

        // 5. Commit. Under the registry write lock (same registry→ART lock
        //    order the read path uses): write + register each output SST,
        //    relocate the ART entry for each latest-version key (only if it
        //    still points into an input SST — never clobbering a concurrent
        //    write/flush), then drop the input registry entries. Old files
        //    are unlinked after the lock is released; by then no reader can
        //    reach them through the registry.
        let output_ssts = all_outputs.len();
        {
            let mut reg = self
                .sst_registry
                .write()
                .map_err(|_| GalaxError::Internal("sst registry lock".into()))?;
            for out in &all_outputs {
                let sid = self.next_sst_id.fetch_add(1, Ordering::SeqCst);
                // Draw the on-disk filename id from the same global counter
                // the flush pipeline uses, so a compaction output file never
                // collides with (and is never mistaken for / deleted as) a
                // flush-written SST. The registry key stays `sid`
                // (next_sst_id space), mirroring the flush path.
                let fid = flush::allocate_block_id();
                let path = self.config.data_dir.join(format!("sst_{}.pax", fid));
                std::fs::write(&path, &out.bytes).map_err(GalaxError::Io)?;
                #[cfg(feature = "aegis-tde")]
                {
                    if let Some(tde) = &self.tde {
                        reg.register_encrypted(sid, path.clone(), tde);
                    } else {
                        reg.register(sid, path.clone());
                    }
                }
                #[cfg(not(feature = "aegis-tde"))]
                {
                    reg.register(sid, path.clone());
                }
                for (key, block_index, row_offset) in &out.art_targets {
                    self.art.relocate_if_points_to(
                        key,
                        &input_ids,
                        RowLocation::SST {
                            sst_id: sid,
                            block_offset: *block_index as u64,
                            row_offset: *row_offset,
                        },
                    );
                }
            }
            for id in &input_ids {
                reg.entries.remove(id);
            }
        }

        // 6. Delete the merged input files (registry no longer references them).
        for (_, path, _) in &inputs {
            let _ = std::fs::remove_file(path);
        }

        Ok(CompactionStats {
            input_ssts: inputs.len(),
            output_ssts,
            keys_retained,
            versions_gc_dropped,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_engine() -> Engine {
        let dir = tempfile::tempdir().unwrap();
        let config = EngineConfig {
            data_dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        // Leak the tempdir so it persists for the test
        std::mem::forget(dir);
        Engine::new(config).unwrap()
    }

    /// A test [`RowColumnSplitter`] that splits a row value of the form
    /// `id_le(8 bytes) ++ name_bytes` into an `Int64` column and a `Text`
    /// column. Stands in for the real SQL-layer splitter (which uses the
    /// row codec + type system); this exercises the storage columnar path
    /// in isolation, in a `#[cfg(test)]` block (engineering-principles §1).
    struct IdNameSplitter;
    impl crate::columnar::RowColumnSplitter for IdNameSplitter {
        fn column_types(&self) -> Vec<galaxdb_common::ColumnType> {
            vec![galaxdb_common::ColumnType::Int64, galaxdb_common::ColumnType::Text]
        }
        fn split(&self, value: &[u8]) -> Option<Vec<Option<Vec<u8>>>> {
            if value.len() < 8 {
                return None;
            }
            Some(vec![Some(value[0..8].to_vec()), Some(value[8..].to_vec())])
        }
    }

    fn id_name_value(id: i64, name: &str) -> Vec<u8> {
        let mut v = id.to_le_bytes().to_vec();
        v.extend_from_slice(name.as_bytes());
        v
    }

    #[tokio::test]
    async fn columnar_flush_appends_typed_columns_and_keeps_base_columns() {
        let engine = test_engine();
        engine.register_columnar_table(b"t:".to_vec(), Arc::new(IdNameSplitter));

        // Three rows of the columnar table `t`.
        let rows = [(1i64, "alice"), (2, "bob"), (3, "carol")];
        for (id, name) in rows {
            let key = format!("t:{id}").into_bytes();
            engine.put_sync(key, id_name_value(id, name)).unwrap();
        }
        engine.flush_memtable().await.unwrap();

        // Base columns must still serve point reads + scans unchanged.
        assert_eq!(engine.get(b"t:1"), Some(id_name_value(1, "alice")));
        assert_eq!(engine.scan_all().len(), 3);

        // Inspect the on-disk SST: the columnar block must carry the base
        // [key, value, ts] columns PLUS [id(Int64), name(Text)].
        let mut checked_columnar = false;
        for entry in std::fs::read_dir(engine.data_dir()).unwrap() {
            let path = entry.unwrap().path();
            let name = path.file_name().unwrap().to_str().unwrap().to_string();
            if !(name.starts_with("sst_") && name.ends_with(".pax")) {
                continue;
            }
            let data = std::fs::read(&path).unwrap();
            let index = crate::sst::SstBlockIndex::from_file_data(&data).unwrap();
            for be in &index.entries {
                let start = be.file_offset as usize;
                let end = start + be.block_len as usize;
                let block = crate::pax::PaxBlock::deserialize(&data[start..end]).unwrap();
                assert_eq!(
                    block.header.column_count, 7,
                    "columnar block = key,value,ts + (id,id_valid) + (name,name_valid)"
                );
                let ids = block.read_column(crate::columnar::data_column_index(0)).unwrap();
                let names = block
                    .read_column(crate::columnar::data_column_index(1))
                    .unwrap();
                // Row order within the block is primary-key sorted: t:1,t:2,t:3.
                assert_eq!(i64::from_le_bytes(ids[0].clone().try_into().unwrap()), 1);
                assert_eq!(names[0], b"alice");
                assert_eq!(i64::from_le_bytes(ids[2].clone().try_into().unwrap()), 3);
                assert_eq!(names[2], b"carol");
                // Validity companions: all present.
                let id_valid = block
                    .read_column(crate::columnar::validity_column_index(0))
                    .unwrap();
                assert_eq!(id_valid[0], vec![1u8]);
                checked_columnar = true;
            }
        }
        assert!(checked_columnar, "expected at least one columnar SST block");
    }

    #[tokio::test]
    async fn unregistered_table_stays_three_column_legacy() {
        let engine = test_engine();
        // No columnar registration → legacy 3-column blocks, unchanged.
        engine.put_sync(b"u:1".to_vec(), b"hello".to_vec()).unwrap();
        engine.flush_memtable().await.unwrap();

        assert_eq!(engine.get(b"u:1"), Some(b"hello".to_vec()));
        for entry in std::fs::read_dir(engine.data_dir()).unwrap() {
            let path = entry.unwrap().path();
            let name = path.file_name().unwrap().to_str().unwrap().to_string();
            if !(name.starts_with("sst_") && name.ends_with(".pax")) {
                continue;
            }
            let data = std::fs::read(&path).unwrap();
            let index = crate::sst::SstBlockIndex::from_file_data(&data).unwrap();
            for be in &index.entries {
                let start = be.file_offset as usize;
                let end = start + be.block_len as usize;
                let block = crate::pax::PaxBlock::deserialize(&data[start..end]).unwrap();
                assert_eq!(block.header.column_count, 3, "legacy block = key,value,ts");
            }
        }
    }

    #[tokio::test]
    async fn columnar_layout_survives_compaction() {
        let engine = test_engine();
        engine.register_columnar_table(b"t:".to_vec(), Arc::new(IdNameSplitter));

        // Two flushes → two SSTs, each a columnar block.
        engine.put_sync(b"t:1".to_vec(), id_name_value(1, "alice")).unwrap();
        engine.put_sync(b"t:2".to_vec(), id_name_value(2, "bob")).unwrap();
        engine.flush_memtable().await.unwrap();
        engine.put_sync(b"t:3".to_vec(), id_name_value(3, "carol")).unwrap();
        engine.put_sync(b"t:4".to_vec(), id_name_value(4, "dave")).unwrap();
        engine.flush_memtable().await.unwrap();

        // Explicitly merge.
        engine.compact().unwrap();

        // Data still correct after compaction.
        assert_eq!(engine.get(b"t:1"), Some(id_name_value(1, "alice")));
        assert_eq!(engine.get(b"t:4"), Some(id_name_value(4, "dave")));
        assert_eq!(engine.scan_all().len(), 4);

        // Every columnar block in the (post-compaction) SSTs must still carry
        // the appended typed columns — the layout is preserved, not reverted.
        let mut saw_columnar_block = false;
        for entry in std::fs::read_dir(engine.data_dir()).unwrap() {
            let path = entry.unwrap().path();
            let name = path.file_name().unwrap().to_str().unwrap().to_string();
            if !(name.starts_with("sst_") && name.ends_with(".pax")) {
                continue;
            }
            let data = std::fs::read(&path).unwrap();
            let Ok(index) = crate::sst::SstBlockIndex::from_file_data(&data) else {
                continue;
            };
            for be in &index.entries {
                let start = be.file_offset as usize;
                let end = start + be.block_len as usize;
                let block = crate::pax::PaxBlock::deserialize(&data[start..end]).unwrap();
                // All rows here belong to columnar table `t`, so every block
                // that holds rows must be a 5-column columnar block.
                if block.header.row_count > 0 {
                    assert_eq!(
                        block.header.column_count, 7,
                        "compacted columnar block must keep base + (data,validity) pairs"
                    );
                    saw_columnar_block = true;
                }
            }
        }
        assert!(saw_columnar_block, "expected a columnar block after compaction");
    }

    #[tokio::test]
    async fn scan_columnar_reads_typed_columns_projection_and_memtable() {
        let engine = test_engine();
        engine.register_columnar_table(b"t:".to_vec(), Arc::new(IdNameSplitter));

        for (id, name) in [(1i64, "alice"), (2, "bob"), (3, "carol")] {
            engine
                .put_sync(format!("t:{id}").into_bytes(), id_name_value(id, name))
                .unwrap();
        }
        engine.flush_memtable().await.unwrap();
        // A memtable-only row (exercises the splitter bridge for unflushed data).
        engine.put_sync(b"t:4".to_vec(), id_name_value(4, "dave")).unwrap();
        // An MVCC override of t:1 still in the memtable (newer version wins).
        engine.put_sync(b"t:1".to_vec(), id_name_value(1, "alice2")).unwrap();

        let batch = engine.scan_columnar(b"t:", &[], &[], u64::MAX).unwrap();
        assert_eq!(batch.num_rows, 4);
        assert_eq!(batch.columns.len(), 2);
        let ids: Vec<i64> = batch.columns[0]
            .1
            .iter()
            .map(|c| i64::from_le_bytes(c.clone().unwrap().try_into().unwrap()))
            .collect();
        assert_eq!(ids, vec![1, 2, 3, 4]); // sorted by key
        let names: Vec<String> = batch.columns[1]
            .1
            .iter()
            .map(|c| String::from_utf8(c.clone().unwrap()).unwrap())
            .collect();
        assert_eq!(names, vec!["alice2", "bob", "carol", "dave"]); // memtable override wins

        // Projection: only the name column.
        let only_name = engine.scan_columnar(b"t:", &[1], &[], u64::MAX).unwrap();
        assert_eq!(only_name.columns.len(), 1);
        assert_eq!(only_name.num_rows, 4);
        assert_eq!(only_name.columns[0].0, galaxdb_common::ColumnType::Text);
    }

    #[tokio::test]
    async fn scan_columnar_prunes_blocks_by_zone_map() {
        use crate::columnar::ColumnPredicate;
        use crate::pax::PruneOp;

        let engine = test_engine();
        engine.register_columnar_table(b"t:".to_vec(), Arc::new(IdNameSplitter));
        for (id, name) in [(10i64, "a"), (20, "b"), (30, "c")] {
            engine
                .put_sync(format!("t:{id}").into_bytes(), id_name_value(id, name))
                .unwrap();
        }
        engine.flush_memtable().await.unwrap();

        // id > 100: the block's id range [10,30] cannot match → pruned → 0 rows.
        let prune = ColumnPredicate {
            column: 0,
            op: PruneOp::Gt,
            value: 100i64.to_le_bytes().to_vec(),
        };
        let pruned = engine
            .scan_columnar(b"t:", &[], std::slice::from_ref(&prune), u64::MAX)
            .unwrap();
        assert_eq!(pruned.num_rows, 0);

        // id > 5: range [10,30] can match → not pruned → all 3 rows returned.
        let keep = ColumnPredicate {
            column: 0,
            op: PruneOp::Gt,
            value: 5i64.to_le_bytes().to_vec(),
        };
        let kept = engine
            .scan_columnar(b"t:", &[], std::slice::from_ref(&keep), u64::MAX)
            .unwrap();
        assert_eq!(kept.num_rows, 3);
    }

    #[tokio::test]
    async fn scan_columnar_bridges_legacy_blocks() {
        // HTAP task 8: a block written BEFORE the table was registered
        // columnar is a legacy 3-column block. scan_columnar must still read
        // it by decoding the row blob via the splitter (the migration
        // bridge), so existing data dirs keep working.
        let engine = test_engine();
        engine.put_sync(b"t:1".to_vec(), id_name_value(1, "alice")).unwrap();
        engine.flush_memtable().await.unwrap(); // legacy 3-column block

        // Register only now, then scan: the legacy block is bridged.
        engine.register_columnar_table(b"t:".to_vec(), Arc::new(IdNameSplitter));
        let batch = engine.scan_columnar(b"t:", &[], &[], u64::MAX).unwrap();
        assert_eq!(batch.num_rows, 1);
        assert_eq!(
            i64::from_le_bytes(batch.columns[0].1[0].clone().unwrap().try_into().unwrap()),
            1
        );
        assert_eq!(batch.columns[1].1[0].clone().unwrap(), b"alice");
    }

    #[tokio::test]
    async fn put_and_get_roundtrip() {
        let engine = test_engine();
        engine.put(b"key1".to_vec(), b"value1".to_vec()).await.unwrap();

        let result = engine.get(b"key1");
        assert_eq!(result, Some(b"value1".to_vec()));
    }

    #[tokio::test]
    async fn get_nonexistent_returns_none() {
        let engine = test_engine();
        assert_eq!(engine.get(b"nope"), None);
    }

    #[tokio::test]
    async fn put_multiple_keys() {
        let engine = test_engine();
        for i in 0..100u32 {
            let key = format!("key-{:04}", i).into_bytes();
            let value = format!("value-{:04}", i).into_bytes();
            engine.put(key, value).await.unwrap();
        }

        assert_eq!(engine.row_count(), 100);
        assert_eq!(engine.index_count(), 100);

        for i in 0..100u32 {
            let key = format!("key-{:04}", i).into_bytes();
            let expected = format!("value-{:04}", i).into_bytes();
            assert_eq!(engine.get(&key), Some(expected));
        }
    }

    #[tokio::test]
    async fn delete_removes_key() {
        let engine = test_engine();
        engine.put(b"key1".to_vec(), b"value1".to_vec()).await.unwrap();
        assert!(engine.get(b"key1").is_some());

        let deleted = engine.delete(b"key1").await.unwrap();
        assert!(deleted);
        assert_eq!(engine.get(b"key1"), None);
    }

    #[tokio::test]
    async fn delete_nonexistent_returns_false() {
        let engine = test_engine();
        let deleted = engine.delete(b"nope").await.unwrap();
        assert!(!deleted);
    }

    #[tokio::test]
    async fn overwrite_key() {
        let engine = test_engine();
        engine.put(b"key1".to_vec(), b"v1".to_vec()).await.unwrap();
        engine.put(b"key1".to_vec(), b"v2".to_vec()).await.unwrap();

        assert_eq!(engine.get(b"key1"), Some(b"v2".to_vec()));
    }

    #[test]
    fn key_range_overlaps_prefix_rejects_below_prefix() {
        // zone_max "aaz" is strictly less than prefix "bb" → skip.
        assert!(!key_range_overlaps_prefix(b"aaa", b"aaz", b"bb"));
    }

    #[test]
    fn key_range_overlaps_prefix_rejects_above_prefix() {
        // zone_min "cc..." starts past "bb..." → skip.
        assert!(!key_range_overlaps_prefix(b"cc1", b"cc9", b"bb"));
    }

    #[test]
    fn key_range_overlaps_prefix_accepts_overlap() {
        // zone_min "bb5" starts with "bb" → keep.
        assert!(key_range_overlaps_prefix(b"bb5", b"cc9", b"bb"));
    }

    #[test]
    fn key_range_overlaps_prefix_accepts_straddle() {
        // zone_min "aa" is below, zone_max "cc" is above — block straddles the prefix namespace.
        assert!(key_range_overlaps_prefix(b"aa", b"cc", b"bb"));
    }

    #[test]
    fn key_range_overlaps_prefix_accepts_empty_zone_map() {
        // Legacy blocks without zone maps → conservatively keep.
        assert!(key_range_overlaps_prefix(b"", b"", b"bb"));
    }

    #[tokio::test]
    async fn scan_with_prefix_after_flush_filters_other_tables() {
        // Insert rows for two tables (different prefixes), flush, scan
        // with prefix "t1:" — only t1 rows should come back, and the
        // SST block for "t2:" should be skipped via zone-map pruning.
        let engine = test_engine();
        for i in 0..50u32 {
            let key = format!("t1:{:04}", i).into_bytes();
            let value = format!("t1v{}", i).into_bytes();
            engine.put(key, value).await.unwrap();
        }
        for i in 0..50u32 {
            let key = format!("t2:{:04}", i).into_bytes();
            let value = format!("t2v{}", i).into_bytes();
            engine.put(key, value).await.unwrap();
        }
        engine.flush_memtable().await.unwrap();

        let t1_rows = engine.scan_all_with_prefix(Some(b"t1:"));
        assert_eq!(t1_rows.len(), 50, "prefix scan must find all t1 rows");
        for (k, _) in &t1_rows {
            assert!(
                k.starts_with(b"t1:"),
                "unexpected key in t1 prefix scan: {:?}",
                k
            );
        }

        let t2_rows = engine.scan_all_with_prefix(Some(b"t2:"));
        assert_eq!(t2_rows.len(), 50);
        for (k, _) in &t2_rows {
            assert!(k.starts_with(b"t2:"));
        }
    }

    #[tokio::test]
    async fn scan_all_at_sees_memtable_and_sst() {
        // AT VERSION must see rows from both the memtable MVCC chain
        // and flushed SSTs. Write, flush, write more, then snapshot
        // each state.
        let engine = test_engine();
        engine.put(b"k1".to_vec(), b"v1".to_vec()).await.unwrap();
        engine.put(b"k2".to_vec(), b"v2".to_vec()).await.unwrap();
        engine.flush_memtable().await.unwrap();
        // At this point both keys live in an SST block.
        let ts_after_flush = engine.next_ts_for_tests() - 1;

        engine.put(b"k3".to_vec(), b"v3".to_vec()).await.unwrap();
        let ts_after_k3 = engine.next_ts_for_tests() - 1;

        // Snapshot at the flush boundary — must see k1 and k2 from SST,
        // but not k3 (written later and still in memtable).
        let rows = engine.scan_all_at(ts_after_flush);
        let keys: Vec<&[u8]> = rows.iter().map(|(k, _, _)| k.as_slice()).collect();
        assert_eq!(keys, vec![b"k1".as_slice(), b"k2".as_slice()]);

        // Snapshot after k3 — must see all three.
        let rows = engine.scan_all_at(ts_after_k3);
        let keys: Vec<&[u8]> = rows.iter().map(|(k, _, _)| k.as_slice()).collect();
        assert_eq!(keys, vec![b"k1".as_slice(), b"k2".as_slice(), b"k3".as_slice()]);
    }

    #[tokio::test]
    async fn scan_all_returns_live_rows() {
        let engine = test_engine();
        engine.put(b"a".to_vec(), b"1".to_vec()).await.unwrap();
        engine.put(b"b".to_vec(), b"2".to_vec()).await.unwrap();
        engine.put(b"c".to_vec(), b"3".to_vec()).await.unwrap();

        let rows = engine.scan_all();
        assert_eq!(rows.len(), 3);

        // Should be sorted by key
        let keys: Vec<&[u8]> = rows.iter().map(|(k, _)| k.as_slice()).collect();
        assert_eq!(keys, vec![b"a".as_slice(), b"b".as_slice(), b"c".as_slice()]);
    }

    #[tokio::test]
    async fn scan_excludes_deleted_rows() {
        let engine = test_engine();
        engine.put(b"a".to_vec(), b"1".to_vec()).await.unwrap();
        engine.put(b"b".to_vec(), b"2".to_vec()).await.unwrap();
        engine.delete(b"a").await.unwrap();

        let rows = engine.scan_all();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].0, b"b");
    }

    #[tokio::test]
    async fn flush_and_read_from_sst() {
        let engine = test_engine();

        // Write 100 rows
        for i in 0..100u32 {
            let key = format!("sst-key-{:04}", i).into_bytes();
            let value = format!("sst-value-{:04}", i).into_bytes();
            engine.put(key, value).await.unwrap();
        }

        assert_eq!(engine.row_count(), 100);

        // Verify reads work from memtable
        assert_eq!(
            engine.get(b"sst-key-0050"),
            Some(b"sst-value-0050".to_vec())
        );

        // Flush to SST
        let flushed = engine.flush_memtable().await.unwrap();
        assert_eq!(flushed, 100);

        // Reads should now come from SST (ART points to SST location)
        let result = engine.get(b"sst-key-0050");
        assert_eq!(result, Some(b"sst-value-0050".to_vec()));

        // Verify all 100 rows are readable from SST
        for i in 0..100u32 {
            let key = format!("sst-key-{:04}", i).into_bytes();
            let expected = format!("sst-value-{:04}", i).into_bytes();
            assert_eq!(engine.get(&key), Some(expected), "failed at key {}", i);
        }
    }

    #[tokio::test]
    async fn concurrent_puts() {
        let engine = Arc::new(test_engine());
        let mut handles = Vec::new();

        for t in 0..8 {
            let eng = engine.clone();
            handles.push(tokio::spawn(async move {
                for i in 0..100u32 {
                    let key = format!("t{}-key-{:04}", t, i).into_bytes();
                    let value = format!("t{}-val-{:04}", t, i).into_bytes();
                    eng.put(key, value).await.unwrap();
                }
            }));
        }

        for h in handles {
            h.await.unwrap();
        }

        assert_eq!(engine.row_count(), 800);
        assert_eq!(engine.index_count(), 800);
    }

    #[test]
    fn encode_decode_kv_roundtrip() {
        let key = b"mykey";
        let value = b"myvalue";
        let encoded = encode_kv(key, value);
        let (dk, dv) = decode_kv(&encoded).unwrap();
        assert_eq!(dk, key);
        assert_eq!(dv, value);
    }

    #[test]
    fn decode_kv_empty_value() {
        let encoded = encode_kv(b"key", b"");
        let (dk, dv) = decode_kv(&encoded).unwrap();
        assert_eq!(dk, b"key");
        assert!(dv.is_empty());
    }

    #[test]
    fn decode_kv_invalid_returns_none() {
        assert!(decode_kv(&[]).is_none());
        assert!(decode_kv(&[0, 0, 0]).is_none());
    }

    #[tokio::test]
    async fn repeated_flush_preserves_all_keys() {
        let engine = test_engine();

        // Phase 1: Write 100 rows, flush
        for i in 0..100u32 {
            let key = format!("phase1-key-{:04}", i).into_bytes();
            let value = format!("phase1-value-{:04}", i).into_bytes();
            engine.put(key, value).await.unwrap();
        }
        let flushed1 = engine.flush_memtable().await.unwrap();
        assert_eq!(flushed1, 100);

        // Phase 2: Write 100 MORE rows (different keys), flush again
        for i in 0..100u32 {
            let key = format!("phase2-key-{:04}", i).into_bytes();
            let value = format!("phase2-value-{:04}", i).into_bytes();
            engine.put(key, value).await.unwrap();
        }
        let flushed2 = engine.flush_memtable().await.unwrap();
        assert_eq!(flushed2, 100);

        // Phase 3: Write 50 MORE rows, flush a third time
        for i in 0..50u32 {
            let key = format!("phase3-key-{:04}", i).into_bytes();
            let value = format!("phase3-value-{:04}", i).into_bytes();
            engine.put(key, value).await.unwrap();
        }
        let flushed3 = engine.flush_memtable().await.unwrap();
        assert_eq!(flushed3, 50);

        // ALL 250 keys must be readable
        let mut missing = Vec::new();
        for i in 0..100u32 {
            let key = format!("phase1-key-{:04}", i).into_bytes();
            if engine.get(&key).is_none() {
                missing.push(format!("phase1-key-{:04}", i));
            }
        }
        for i in 0..100u32 {
            let key = format!("phase2-key-{:04}", i).into_bytes();
            if engine.get(&key).is_none() {
                missing.push(format!("phase2-key-{:04}", i));
            }
        }
        for i in 0..50u32 {
            let key = format!("phase3-key-{:04}", i).into_bytes();
            if engine.get(&key).is_none() {
                missing.push(format!("phase3-key-{:04}", i));
            }
        }

        assert!(
            missing.is_empty(),
            "missing {} keys after repeated flushes: {:?}",
            missing.len(),
            &missing[..missing.len().min(10)]
        );

        // Verify correct values
        assert_eq!(
            engine.get(b"phase1-key-0050"),
            Some(b"phase1-value-0050".to_vec())
        );
        assert_eq!(
            engine.get(b"phase2-key-0099"),
            Some(b"phase2-value-0099".to_vec())
        );
        assert_eq!(
            engine.get(b"phase3-key-0049"),
            Some(b"phase3-value-0049".to_vec())
        );
    }

    #[cfg(feature = "aegis-tde")]
    #[tokio::test]
    async fn flush_and_read_with_aegis_tde() {
        use galaxdb_crypto::{AegisTdeModule, LocalKeyProvider};

        let dir = tempfile::tempdir().unwrap();
        let config = EngineConfig {
            data_dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        std::mem::forget(dir);

        let mut engine = Engine::new(config).unwrap();

        // Enable AEGIS-256 TDE
        let key_provider = LocalKeyProvider::from_key([0xABu8; 32]);
        let tde = AegisTdeModule::new(&key_provider).unwrap();
        engine.enable_tde(tde);
        assert!(engine.tde_enabled());

        // Write 100 rows
        for i in 0..100u32 {
            let key = format!("tde-key-{:04}", i).into_bytes();
            let value = format!("tde-value-{:04}", i).into_bytes();
            engine.put(key, value).await.unwrap();
        }

        // Verify reads from memtable work
        assert_eq!(
            engine.get(b"tde-key-0050"),
            Some(b"tde-value-0050".to_vec())
        );

        // Flush to encrypted SST
        let flushed = engine.flush_memtable().await.unwrap();
        assert_eq!(flushed, 100);

        // Reads should now come from encrypted SST (decrypted transparently)
        for i in 0..100u32 {
            let key = format!("tde-key-{:04}", i).into_bytes();
            let expected = format!("tde-value-{:04}", i).into_bytes();
            assert_eq!(engine.get(&key), Some(expected), "failed at key {}", i);
        }

        // Verify the SST file on disk is actually encrypted (not readable as PAX)
        let sst_files: Vec<_> = std::fs::read_dir(engine.data_dir())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "pax"))
            .collect();
        assert!(!sst_files.is_empty(), "should have SST files");

        for sst_file in &sst_files {
            let raw_data = std::fs::read(sst_file.path()).unwrap();
            // Raw data should NOT be a valid PAX block (it's encrypted)
            assert!(
                crate::pax::PaxBlock::deserialize(&raw_data).is_err(),
                "encrypted SST should not be deserializable as plain PAX"
            );
        }
    }
}
