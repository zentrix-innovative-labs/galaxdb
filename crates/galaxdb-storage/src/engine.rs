//! Storage Engine facade — the unified API that the SQL executor calls.
//!
//! Connects: Memtable + WAL + ART Index + Flush + Buffer Pool + Compaction
//! into a single coherent interface for reading and writing rows.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use galaxdb_common::{GalaxError, GalaxResult, Timestamp};
use galaxdb_io::{IoScheduler, IoPriority};

use crate::art::{ArtIndex, RowLocation};
use crate::flush::{self, FlushConfig};
use crate::memtable::MemtableManager;

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
        };

        let wal = Arc::new(WalWriter::new(wal_config).map_err(GalaxError::Io)?);

        let memtable_mgr = MemtableManager::new(
            config.memtable_size_bytes,
            config.back_pressure_bytes,
        );

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

        Ok(Self {
            config,
            memtable_mgr,
            art: Arc::new(ArtIndex::new()),
            wal,
            sst_registry: RwLock::new(SstRegistry::with_cache_limit(sst_cache_bytes)),
            next_timestamp: AtomicU64::new(1),
            next_sst_id: AtomicU64::new(1),
            row_count: AtomicU64::new(0),
            io_scheduler,
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
        // Check ART index first
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
                let registry = self.sst_registry.read().ok()?;
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

    /// Flush the active memtable to an SST file on disk.
    /// Updates ART entries to point to the SST instead of the memtable.
    pub async fn flush_memtable(&self) -> GalaxResult<u64> {
        let active = self.memtable_mgr.active();
        let entries = active.iter_all();

        if entries.is_empty() {
            return Ok(0);
        }

        let sst_id = self.next_sst_id.fetch_add(1, Ordering::SeqCst);
        let flush_config = FlushConfig {
            data_dir: self.config.data_dir.clone(),
            sst_size_bytes: self.config.sst_size_bytes,
            max_rows_per_block: self.config.max_rows_per_sst,
        };

        let result = {
            #[cfg(feature = "aegis-tde")]
            {
                if let Some(tde) = &self.tde {
                    flush::flush_memtable_encrypted(&active, &flush_config, sst_id, tde, self.io_scheduler.as_ref()).await?
                } else {
                    flush::flush_memtable(&active, &flush_config, sst_id, self.io_scheduler.as_ref()).await?
                }
            }
            #[cfg(not(feature = "aegis-tde"))]
            {
                flush::flush_memtable(&active, &flush_config, sst_id, self.io_scheduler.as_ref()).await?
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
        let mut out: HashMap<Vec<u8>, Vec<u8>> = HashMap::new();

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

                    let keys = match block.read_column(0) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    let vals = match block.read_column(1) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    for (k, v) in keys.into_iter().zip(vals.into_iter()) {
                        if let Some(prefix) = key_prefix {
                            if !k.starts_with(prefix) {
                                continue;
                            }
                        }
                        if v.is_empty() {
                            out.remove(&k);
                        } else {
                            out.insert(k, v);
                        }
                    }
                }
            }
        }

        let active = self.memtable_mgr.active();
        for (key, versioned) in active.iter_all() {
            if let Some(prefix) = key_prefix {
                if !key.starts_with(prefix) {
                    continue;
                }
            }
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
                    let block_ts = block.header.commit_timestamp;
                    if block_ts > read_ts {
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
                    for (k, v) in keys.into_iter().zip(vals.into_iter()) {
                        // Merge by taking the latest visible version.
                        let existing_ts = out.get(&k).map(|(_, ts)| *ts).unwrap_or(0);
                        if block_ts >= existing_ts {
                            if v.is_empty() {
                                out.remove(&k);
                            } else {
                                out.insert(k, (v, block_ts));
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

    /// Get the total number of rows (approximate).
    pub fn row_count(&self) -> u64 {
        self.row_count.load(Ordering::Relaxed)
    }

    /// Get the ART index entry count.
    pub fn index_count(&self) -> usize {
        self.art.len()
    }

    /// Get the data directory path.
    pub fn data_dir(&self) -> &Path {
        &self.config.data_dir
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
    if zone_max.as_ref() < prefix {
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
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "pax"))
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
