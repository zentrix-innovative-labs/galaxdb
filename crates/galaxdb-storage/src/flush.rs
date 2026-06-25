//! Memtable flush pipeline for GalaxDB.
//!
//! Converts a sealed memtable into SST files (PAX blocks on disk) and
//! integrates with the WAL for checkpoint and truncation.
//!
//! ## Flush Pipeline
//!
//! 1. Get all entries from the sealed memtable (sorted by primary key via `iter_all()`)
//! 2. Group entries into PAX blocks (respecting the configured SST target size)
//! 3. Encrypt each PAX block with AEGIS-256 (if TDE is enabled)
//! 4. Write each encrypted PAX block to disk as an SST file
//! 5. After successful flush, write a CHECKPOINT record to the WAL
//! 6. Truncate the WAL to the checkpoint point
//! 7. Notify the MemtableManager that flush is complete (releases back-pressure)
//!
//! ## TDE Encryption
//!
//! When the `aegis-tde` feature is enabled, PAX blocks are encrypted with
//! AEGIS-256 before writing to disk. AEGIS-256 achieves 6-10 GB/s on modern
//! CPUs with AES-NI — 4-8× faster than AES-256-GCM.
//! WAL records continue to use AES-256-GCM (append-only sequential writes).

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use galaxdb_common::{BlockId, ColumnType, GalaxError, GalaxResult, Timestamp};
use galaxdb_io::{IoScheduler, IoPriority};

use crate::memtable::{Memtable, VersionedValue};
use crate::columnar::{registration_for, ColumnarRegistration, RowColumnSplitter};
use crate::pax::{CodecId, ColumnData, PaxBlock};
use crate::wal::WalWriter;

/// Configuration for the flush pipeline.
#[derive(Debug, Clone)]
pub struct FlushConfig {
    /// Directory where SST files are written.
    pub data_dir: PathBuf,
    /// Target SST file size in bytes (default: 8 MB).
    /// Multiple PAX blocks are packed into each SST file.
    pub sst_size_bytes: u64,
    /// Maximum number of rows per PAX block within an SST (default: 100).
    /// Smaller blocks = faster point reads (one NVMe read per block).
    /// With 100 rows × ~625 bytes = ~62KB per block, a cold point read
    /// loads ~64KB from NVMe = ~18µs at 3.5 GB/s.
    pub max_rows_per_block: usize,
}

impl Default for FlushConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("galaxdb_data"),
            sst_size_bytes: 8 * 1024 * 1024,  // 8 MB
            max_rows_per_block: 100,           // ~64KB per block for fast point reads
        }
    }
}

/// Global block ID counter for generating unique SST file names.
static NEXT_BLOCK_ID: AtomicU64 = AtomicU64::new(1);

/// Set the next block ID (useful for recovery or testing).
pub fn set_next_block_id(id: u64) {
    NEXT_BLOCK_ID.store(id, Ordering::SeqCst);
}

/// Get the next block ID without incrementing.
pub fn peek_next_block_id() -> u64 {
    NEXT_BLOCK_ID.load(Ordering::SeqCst)
}

/// Allocate a new unique block ID.
///
/// Public so the runtime compaction driver names its output SST files
/// from the *same* monotonic counter the flush pipeline uses, guaranteeing
/// compaction output never collides with a flush-written `sst_<id>.pax`
/// path (the registry key stays in the engine's `next_sst_id` space, just
/// like flush — only the on-disk filename is drawn from this counter).
pub fn allocate_block_id() -> BlockId {
    NEXT_BLOCK_ID.fetch_add(1, Ordering::SeqCst)
}

/// Result of a successful flush operation.
#[derive(Debug)]
pub struct FlushResult {
    /// Paths to the SST files written.
    pub sst_paths: Vec<PathBuf>,
    /// Block IDs of the PAX blocks written.
    pub block_ids: Vec<BlockId>,
    /// Total number of rows flushed.
    pub rows_flushed: usize,
    /// Total bytes written to disk.
    pub bytes_written: u64,
    /// The WAL checkpoint sequence number (if WAL integration was used).
    pub checkpoint_seq_no: Option<u64>,
    /// Per-SST block metadata: (sst_index, block_index_within_sst, rows_in_block).
    /// Used by the Engine to update ART entries with correct block_offset values.
    pub block_map: Vec<SstBlockInfo>,
}

/// Metadata about one PAX block within an SST file.
#[derive(Debug, Clone)]
pub struct SstBlockInfo {
    /// Index of the SST file in `sst_paths`.
    pub sst_index: usize,
    /// Block index within the SST file (used as `block_offset` in ART).
    pub block_index: u32,
    /// Number of rows in this block.
    pub row_count: usize,
}

/// Encrypt a PAX block using AEGIS-256 TDE.
///
/// When the `aegis-tde` feature is enabled and a TDE module is provided,
/// the block data is encrypted with AEGIS-256 (6-10 GB/s on AES-NI hardware).
/// When TDE is not configured, the data is returned unchanged.
#[cfg(feature = "aegis-tde")]
fn encrypt_block_data(
    data: &[u8],
    tde: Option<&galaxdb_crypto::AegisTdeModule>,
) -> GalaxResult<Vec<u8>> {
    match tde {
        Some(module) => module.encrypt(data),
        None => Ok(data.to_vec()),
    }
}

#[cfg(not(feature = "aegis-tde"))]
fn encrypt_block_data(
    data: &[u8],
    _tde: Option<&()>,
) -> GalaxResult<Vec<u8>> {
    Ok(data.to_vec())
}

/// Decrypt a PAX block using AEGIS-256 TDE.
///
/// Counterpart to `encrypt_block_data`. Used when reading SST files from disk.
#[cfg(feature = "aegis-tde")]
pub fn decrypt_block_data(
    data: &[u8],
    tde: Option<&galaxdb_crypto::AegisTdeModule>,
) -> GalaxResult<Vec<u8>> {
    match tde {
        Some(module) => module.decrypt(data),
        None => Ok(data.to_vec()),
    }
}

#[cfg(not(feature = "aegis-tde"))]
pub fn decrypt_block_data(
    data: &[u8],
    _tde: Option<&()>,
) -> GalaxResult<Vec<u8>> {
    Ok(data.to_vec())
}

/// Convert memtable entries into columnar ColumnData for PAX block writing.
///
/// Each row is stored as three columns:
/// - Column 0: key (Blob)
/// - Column 1: value (Blob; empty for a tombstone)
/// - Column 2: the row's MVCC commit timestamp (Blob, 8-byte little-endian)
///
/// The per-row timestamp column is what makes `AT VERSION <ts>` correct at
/// an arbitrary snapshot: a single flush packs rows committed at different
/// MVCC timestamps into one block, so a single block-level timestamp cannot
/// express per-row visibility. Readers that predate this column (legacy
/// two-column SSTs) fall back to the block header's `commit_timestamp`.
fn entries_to_columns(entries: &[(Vec<u8>, VersionedValue)]) -> Vec<ColumnData> {
    let mut key_values: Vec<Vec<u8>> = Vec::with_capacity(entries.len());
    let mut val_values: Vec<Vec<u8>> = Vec::with_capacity(entries.len());
    let mut ts_values: Vec<Vec<u8>> = Vec::with_capacity(entries.len());

    for (key, versioned) in entries {
        key_values.push(key.clone());
        // For tombstones, store an empty byte vector.
        // The actual tombstone semantics are tracked by the version chain;
        // at the SST level we just need the key present.
        val_values.push(versioned.value.clone().unwrap_or_default());
        ts_values.push(versioned.timestamp.to_le_bytes().to_vec());
    }

    vec![
        ColumnData {
            col_type: ColumnType::Blob,
            values: key_values,
        },
        ColumnData {
            col_type: ColumnType::Blob,
            values: val_values,
        },
        ColumnData {
            col_type: ColumnType::Blob,
            values: ts_values,
        },
    ]
}

/// The per-row MVCC timestamp column index in an SST PAX block written by
/// [`entries_to_columns`]. Blocks with fewer columns are legacy two-column
/// SSTs and fall back to the block header timestamp.
pub const ROW_TS_COLUMN: usize = 2;

/// Decode a row's MVCC commit timestamp from the 8-byte little-endian
/// encoding written into [`ROW_TS_COLUMN`]. A malformed or short value
/// yields `None` so the caller can fall back to the block header.
pub fn decode_row_ts(bytes: &[u8]) -> Option<Timestamp> {
    if bytes.len() == 8 {
        Some(u64::from_le_bytes(bytes.try_into().ok()?))
    } else {
        None
    }
}

/// Split sorted entries into chunks that respect the SST size target.
///
/// Each chunk will become one PAX block / SST file. When a chunk's rows
/// belong to a table registered for columnar storage, the chunk is also cut
/// at the table's key-prefix boundary so a single block never mixes a
/// columnar table's rows with another table's — a PAX block's columns are
/// aligned across all its rows, so per-SQL-column chunks require one schema
/// per block (HTAP ADR-0002).
fn split_into_blocks<'a>(
    entries: &'a [(Vec<u8>, VersionedValue)],
    config: &FlushConfig,
    registrations: &[ColumnarRegistration],
) -> Vec<&'a [(Vec<u8>, VersionedValue)]> {
    if entries.is_empty() {
        return Vec::new();
    }

    // Index of the matching registration for a key (None = unregistered).
    let reg_of = |key: &[u8]| -> Option<usize> {
        registrations.iter().position(|r| r.matches(key))
    };

    let mut blocks = Vec::new();
    let mut start = 0;
    let mut current_size: u64 = 0;

    for (i, (key, versioned)) in entries.iter().enumerate() {
        let _ = key;
        let entry_size = entries[i].0.len() as u64
            + versioned.value.as_ref().map_or(0, |v| v.len()) as u64
            + 16; // overhead

        current_size += entry_size;

        let row_count = i - start + 1;
        // Cut the block when the NEXT row crosses a columnar table boundary,
        // so each block holds rows of at most one registered table.
        let next_crosses_table = i + 1 < entries.len()
            && reg_of(&entries[i + 1].0) != reg_of(&entries[start].0);
        let should_split = ((current_size >= config.sst_size_bytes
            || row_count >= config.max_rows_per_block)
            && row_count > 0)
            || next_crosses_table;

        if should_split && i + 1 < entries.len() {
            blocks.push(&entries[start..=i]);
            start = i + 1;
            current_size = 0;
        }
    }

    // Don't forget the last chunk
    if start < entries.len() {
        blocks.push(&entries[start..]);
    }

    blocks
}

/// Build the PAX columns and per-column codecs for one block.
///
/// For a block whose rows belong to a columnar-registered table, this
/// returns the base `[key, value, ts]` columns plus one typed column per
/// SQL column (HTAP ADR-0002). For every other block — and as a safe
/// fallback if any row fails to split — it returns the legacy three-column
/// layout, so no data is ever lost or mis-shaped.
fn build_block_columns(
    chunk: &[(Vec<u8>, VersionedValue)],
    registrations: &[ColumnarRegistration],
) -> (Vec<ColumnData>, Vec<CodecId>) {
    let legacy = || {
        (
            entries_to_columns(chunk),
            vec![CodecId::Zstd, CodecId::None, CodecId::None],
        )
    };
    let Some(first) = chunk.first() else {
        return legacy();
    };
    let Some(reg) = registration_for(registrations, &first.0) else {
        return legacy();
    };
    match entries_to_columns_columnar(chunk, reg.splitter.as_ref()) {
        Some(res) => res,
        None => {
            tracing::warn!(
                prefix = ?String::from_utf8_lossy(&reg.prefix),
                "columnar split failed for a block; writing legacy 3-column block (scan falls back to decode bridge)"
            );
            legacy()
        }
    }
}

/// Build base + per-SQL-column PAX columns for a columnar block. Returns
/// `None` if any row's value cannot be split into the expected number of
/// columns, so the caller falls back to the legacy layout.
fn entries_to_columns_columnar(
    chunk: &[(Vec<u8>, VersionedValue)],
    splitter: &dyn RowColumnSplitter,
) -> Option<(Vec<ColumnData>, Vec<CodecId>)> {
    let mut base = entries_to_columns(chunk); // [key, value, ts]
    let values: Vec<Option<Vec<u8>>> = chunk.iter().map(|(_, v)| v.value.clone()).collect();
    let (data_cols, data_codecs) = crate::columnar::columnar_data_columns(&values, splitter)?;

    let mut codecs = vec![CodecId::Zstd, CodecId::None, CodecId::None];
    codecs.extend(data_codecs);
    base.extend(data_cols);
    Some((base, codecs))
}

/// Flush a sealed memtable to SST files on disk.
///
/// This is the core flush pipeline:
/// 1. Extract all entries from the memtable (already sorted by key)
/// 2. Split entries into chunks based on SST size target
/// 3. For each chunk, create a PAX block, encrypt with AEGIS-256 (if TDE enabled), and write to disk
/// 4. Return the flush result with paths and metadata
///
/// This function does NOT handle WAL checkpoint or MemtableManager notification.
/// Use [`flush_memtable_with_wal`] for the full pipeline.
///
/// # TDE Encryption
///
/// Pass `Some(&aegis_module)` to encrypt PAX blocks with AEGIS-256 before writing.
/// Pass `None` to write unencrypted blocks (for testing or when TDE is disabled).
#[cfg(feature = "aegis-tde")]
pub async fn flush_memtable(
    memtable: &Memtable,
    config: &FlushConfig,
    commit_timestamp: Timestamp,
    io: &dyn IoScheduler,
    registrations: &[ColumnarRegistration],
) -> GalaxResult<FlushResult> {
    flush_memtable_inner(memtable, config, commit_timestamp, None, io, registrations).await
}

/// Flush a sealed memtable with AEGIS-256 TDE encryption.
#[cfg(feature = "aegis-tde")]
pub async fn flush_memtable_encrypted(
    memtable: &Memtable,
    config: &FlushConfig,
    commit_timestamp: Timestamp,
    tde: &galaxdb_crypto::AegisTdeModule,
    io: &dyn IoScheduler,
    registrations: &[ColumnarRegistration],
) -> GalaxResult<FlushResult> {
    flush_memtable_inner(memtable, config, commit_timestamp, Some(tde), io, registrations).await
}

#[cfg(not(feature = "aegis-tde"))]
pub async fn flush_memtable(
    memtable: &Memtable,
    config: &FlushConfig,
    commit_timestamp: Timestamp,
    io: &dyn IoScheduler,
    registrations: &[ColumnarRegistration],
) -> GalaxResult<FlushResult> {
    flush_memtable_inner(memtable, config, commit_timestamp, None, io, registrations).await
}

#[cfg(feature = "aegis-tde")]
async fn flush_memtable_inner(
    memtable: &Memtable,
    config: &FlushConfig,
    commit_timestamp: Timestamp,
    tde: Option<&galaxdb_crypto::AegisTdeModule>,
    io: &dyn IoScheduler,
    registrations: &[ColumnarRegistration],
) -> GalaxResult<FlushResult> {
    // Step 1: Get all entries sorted by primary key
    let entries = memtable.iter_all();

    if entries.is_empty() {
        return Ok(FlushResult {
            sst_paths: Vec::new(),
            block_ids: Vec::new(),
            rows_flushed: 0,
            bytes_written: 0,
            checkpoint_seq_no: None,
            block_map: Vec::new(),
        });
    }

    // Ensure the data directory exists
    tokio::fs::create_dir_all(&config.data_dir)
        .await
        .map_err(GalaxError::Io)?;

    // Step 2: Split entries into small block-sized chunks (~100 rows each).
    // Following RocksDB's BlockBasedTable pattern: each SST file contains
    // multiple small data blocks with a block index at the end. A point read
    // loads one block (~64KB) from NVMe instead of the entire SST file.
    let chunks = split_into_blocks(&entries, config, registrations);

    let mut sst_paths: Vec<PathBuf> = Vec::new();
    let mut block_ids: Vec<BlockId> = Vec::new();
    let mut block_map: Vec<SstBlockInfo> = Vec::new();
    let mut total_bytes: u64 = 0;
    let total_rows = entries.len();

    // Step 3: Pack multiple PAX blocks into SST files with block indexes.
    // Each SST file holds blocks until it reaches sst_size_bytes.
    let mut current_sst_data: Vec<u8> = Vec::new();
    let mut current_sst_index = crate::sst::SstBlockIndex::new();
    let mut current_sst_block_count: u32 = 0;
    let sst_index_in_result = std::cell::Cell::new(0usize);

    for chunk in &chunks {
        let block_id = allocate_block_id();

        // Build PAX columns for this block: legacy [key, value, ts], plus
        // one typed column per SQL column when the block belongs to a
        // columnar-registered table (HTAP ADR-0002). Codecs: key=Zstd,
        // value=None (fast single-row reads), ts=None, then a per-type codec
        // for each appended data column.
        let (columns, codecs) = build_block_columns(chunk, registrations);
        // The block header timestamp is the newest MVCC commit in this
        // block (a real row timestamp, not a flush sequence number). Per-row
        // visibility comes from the timestamp column; the header is a fast
        // upper bound and the legacy fallback. `commit_timestamp` (the flush
        // argument) is used only when a chunk somehow has no rows.
        let block_ts = chunk
            .iter()
            .map(|(_, v)| v.timestamp)
            .max()
            .unwrap_or(commit_timestamp);
        let pax_block = PaxBlock::write_with_codecs(block_id, block_ts, &columns, &codecs)?;
        let block_bytes = pax_block.serialize()?;
        let encrypted_bytes = encrypt_block_data(&block_bytes, tde)?;

        // Record block position within the SST file
        let block_file_offset = current_sst_data.len() as u64;
        let block_len = encrypted_bytes.len() as u32;
        current_sst_index.add_block(block_file_offset, block_len);

        // Track block metadata for ART updates
        block_map.push(SstBlockInfo {
            sst_index: sst_index_in_result.get(),
            block_index: current_sst_block_count,
            row_count: chunk.len(),
        });

        current_sst_data.extend_from_slice(&encrypted_bytes);
        block_ids.push(block_id);
        current_sst_block_count += 1;

        // Check if current SST has reached size limit
        if current_sst_data.len() as u64 >= config.sst_size_bytes {
            // Append block index + footer
            let index_offset = current_sst_data.len() as u64;
            let index_footer = current_sst_index.serialize_with_footer(index_offset);
            current_sst_data.extend_from_slice(&index_footer);

            // Write SST file to disk via IoScheduler BK queue
            let sst_id = allocate_block_id();
            let sst_filename = format!("sst_{}.pax", sst_id);
            let sst_path = config.data_dir.join(&sst_filename);
            io.write(&sst_path, 0, &current_sst_data, IoPriority::Background).await?;

            total_bytes += current_sst_data.len() as u64;
            sst_paths.push(sst_path);

            // Reset for next SST
            current_sst_data.clear();
            current_sst_index = crate::sst::SstBlockIndex::new();
            current_sst_block_count = 0;
            sst_index_in_result.set(sst_index_in_result.get() + 1);
        }
    }

    // Flush remaining blocks to a final SST file
    if !current_sst_data.is_empty() {
        let index_offset = current_sst_data.len() as u64;
        let index_footer = current_sst_index.serialize_with_footer(index_offset);
        current_sst_data.extend_from_slice(&index_footer);

        let sst_id = allocate_block_id();
        let sst_filename = format!("sst_{}.pax", sst_id);
        let sst_path = config.data_dir.join(&sst_filename);
        io.write(&sst_path, 0, &current_sst_data, IoPriority::Background).await?;

        total_bytes += current_sst_data.len() as u64;
        sst_paths.push(sst_path);
    }

    Ok(FlushResult {
        sst_paths,
        block_ids,
        rows_flushed: total_rows,
        bytes_written: total_bytes,
        checkpoint_seq_no: None,
        block_map,
    })
}

#[cfg(not(feature = "aegis-tde"))]
async fn flush_memtable_inner(
    memtable: &Memtable,
    config: &FlushConfig,
    commit_timestamp: Timestamp,
    _tde: Option<&()>,
    io: &dyn IoScheduler,
    registrations: &[ColumnarRegistration],
) -> GalaxResult<FlushResult> {
    // Step 1: Get all entries sorted by primary key
    let entries = memtable.iter_all();

    if entries.is_empty() {
        return Ok(FlushResult {
            sst_paths: Vec::new(),
            block_ids: Vec::new(),
            rows_flushed: 0,
            bytes_written: 0,
            checkpoint_seq_no: None,
            block_map: Vec::new(),
        });
    }

    // Ensure the data directory exists
    tokio::fs::create_dir_all(&config.data_dir)
        .await
        .map_err(GalaxError::Io)?;

    // Step 2: Split entries into small block-sized chunks (~100 rows each).
    let chunks = split_into_blocks(&entries, config, registrations);

    let mut sst_paths: Vec<PathBuf> = Vec::new();
    let mut block_ids: Vec<BlockId> = Vec::new();
    let mut block_map: Vec<SstBlockInfo> = Vec::new();
    let mut total_bytes: u64 = 0;
    let total_rows = entries.len();

    // Step 3: Pack multiple PAX blocks into SST files with block indexes.
    let mut current_sst_data: Vec<u8> = Vec::new();
    let mut current_sst_index = crate::sst::SstBlockIndex::new();
    let mut current_sst_block_count: u32 = 0;
    let sst_index_in_result = std::cell::Cell::new(0usize);

    for chunk in &chunks {
        let block_id = allocate_block_id();

        // Legacy [key, value, ts] plus one typed column per SQL column when
        // the block belongs to a columnar-registered table (HTAP ADR-0002).
        let (columns, codecs) = build_block_columns(chunk, registrations);
        let block_ts = chunk
            .iter()
            .map(|(_, v)| v.timestamp)
            .max()
            .unwrap_or(commit_timestamp);
        let pax_block = PaxBlock::write_with_codecs(block_id, block_ts, &columns, &codecs)?;
        let block_bytes = pax_block.serialize()?;

        let block_file_offset = current_sst_data.len() as u64;
        let block_len = block_bytes.len() as u32;
        current_sst_index.add_block(block_file_offset, block_len);

        block_map.push(SstBlockInfo {
            sst_index: sst_index_in_result.get(),
            block_index: current_sst_block_count,
            row_count: chunk.len(),
        });

        current_sst_data.extend_from_slice(&block_bytes);
        block_ids.push(block_id);
        current_sst_block_count += 1;

        if current_sst_data.len() as u64 >= config.sst_size_bytes {
            let index_offset = current_sst_data.len() as u64;
            let index_footer = current_sst_index.serialize_with_footer(index_offset);
            current_sst_data.extend_from_slice(&index_footer);

            let sst_id = allocate_block_id();
            let sst_filename = format!("sst_{}.pax", sst_id);
            let sst_path = config.data_dir.join(&sst_filename);
            io.write(&sst_path, 0, &current_sst_data, IoPriority::Background).await?;

            total_bytes += current_sst_data.len() as u64;
            sst_paths.push(sst_path);

            current_sst_data.clear();
            current_sst_index = crate::sst::SstBlockIndex::new();
            current_sst_block_count = 0;
            sst_index_in_result.set(sst_index_in_result.get() + 1);
        }
    }

    if !current_sst_data.is_empty() {
        let index_offset = current_sst_data.len() as u64;
        let index_footer = current_sst_index.serialize_with_footer(index_offset);
        current_sst_data.extend_from_slice(&index_footer);

        let sst_id = allocate_block_id();
        let sst_filename = format!("sst_{}.pax", sst_id);
        let sst_path = config.data_dir.join(&sst_filename);
        io.write(&sst_path, 0, &current_sst_data, IoPriority::Background).await?;

        total_bytes += current_sst_data.len() as u64;
        sst_paths.push(sst_path);
    }

    Ok(FlushResult {
        sst_paths,
        block_ids,
        rows_flushed: total_rows,
        bytes_written: total_bytes,
        checkpoint_seq_no: None,
        block_map,
    })
}

/// Flush a sealed memtable with full WAL integration.
///
/// This is the complete flush pipeline:
/// 1. Flush the memtable to SST files via [`flush_memtable`]
/// 2. Write a CHECKPOINT record to the WAL
/// 3. Truncate the WAL to the checkpoint point
///
/// The caller is responsible for calling `MemtableManager::on_flush_complete()`
/// after this function returns successfully.
pub async fn flush_memtable_with_wal(
    memtable: &Memtable,
    config: &FlushConfig,
    commit_timestamp: Timestamp,
    wal_writer: &WalWriter,
    io: &dyn IoScheduler,
) -> GalaxResult<FlushResult> {
    // Step 1: Flush memtable to SST files (no columnar registrations on the
    // WAL-checkpoint convenience path; the engine's flush threads them).
    let mut result = flush_memtable(memtable, config, commit_timestamp, io, &[]).await?;

    // Step 2: Write CHECKPOINT record to WAL
    let checkpoint_seq_no = wal_writer
        .write_checkpoint()
        .await
        .map_err(GalaxError::Io)?;

    // Step 3: Truncate WAL to checkpoint
    wal_writer
        .truncate_to_checkpoint()
        .await
        .map_err(GalaxError::Io)?;

    result.checkpoint_seq_no = Some(checkpoint_seq_no);

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memtable::Memtable;
    use crate::pax::PaxBlock;
    use crate::wal::{WalWriter, WalWriterConfig};
    use galaxdb_io::TokioScheduler;
    use std::time::Duration;

    /// Helper: create a TokioScheduler for tests.
    fn test_io() -> TokioScheduler {
        TokioScheduler::new()
    }

    /// Helper: create a memtable with some test data.
    fn create_test_memtable(num_entries: usize) -> Memtable {
        // Use a very large threshold so we can control sealing manually
        let memtable = Memtable::new(1024 * 1024 * 1024);
        for i in 0..num_entries {
            let key = format!("key-{:06}", i).into_bytes();
            let value = format!("value-{:06}", i).into_bytes();
            memtable.put(key, i as u64 + 1, Some(value));
        }
        memtable
    }

    #[tokio::test]
    async fn flush_empty_memtable_produces_no_files() {
        let dir = tempfile::tempdir().unwrap();
        let config = FlushConfig {
            data_dir: dir.path().to_path_buf(),
            sst_size_bytes: 64 * 1024 * 1024,
            max_rows_per_block: 1_000_000,
        };

        let memtable = Memtable::new(1024 * 1024 * 1024);
        let result = flush_memtable(&memtable, &config, 100, &test_io(), &[]).await.unwrap();

        assert!(result.sst_paths.is_empty());
        assert!(result.block_ids.is_empty());
        assert_eq!(result.rows_flushed, 0);
        assert_eq!(result.bytes_written, 0);
        assert!(result.checkpoint_seq_no.is_none());
    }

    #[tokio::test]
    async fn flush_produces_valid_pax_blocks() {
        let dir = tempfile::tempdir().unwrap();
        let config = FlushConfig {
            data_dir: dir.path().to_path_buf(),
            sst_size_bytes: 64 * 1024 * 1024,
            max_rows_per_block: 1_000_000,
        };

        let num_entries = 100;
        let memtable = create_test_memtable(num_entries);
        memtable.seal();

        let result = flush_memtable(&memtable, &config, 42, &test_io(), &[]).await.unwrap();

        assert_eq!(result.rows_flushed, num_entries);
        assert!(!result.sst_paths.is_empty());
        assert!(!result.block_ids.is_empty());
        assert!(result.bytes_written > 0);

        // Read back each SST file and verify blocks via the block index
        for sst_path in &result.sst_paths {
            assert!(sst_path.exists());

            let file_data = tokio::fs::read(sst_path).await.unwrap();
            let block_index = crate::sst::SstBlockIndex::from_file_data(&file_data)
                .expect("SST file should have a valid block index");

            assert!(block_index.block_count() > 0, "SST should have at least one block");

            for entry in &block_index.entries {
                let start = entry.file_offset as usize;
                let end = start + entry.block_len as usize;
                let block_bytes = &file_data[start..end];

                let block = PaxBlock::deserialize(block_bytes)
                    .expect("each block in SST should be a valid PAX block");

                // Verify block metadata. The header timestamp is now the
                // newest MVCC commit in the block (real row timestamps
                // 1..=100), not the flush argument (42).
                assert_eq!(block.header.column_count, 3); // key + value + ts
                assert!(block.header.row_count > 0);

                // Verify we can read back the columns
                let keys = block.read_column(0).unwrap();
                let values = block.read_column(1).unwrap();
                let timestamps = block.read_column(2).unwrap();
                assert_eq!(keys.len(), block.header.row_count as usize);
                assert_eq!(values.len(), block.header.row_count as usize);
                assert_eq!(timestamps.len(), block.header.row_count as usize);

                // The header timestamp equals the max per-row timestamp.
                let max_ts = timestamps
                    .iter()
                    .map(|b| super::decode_row_ts(b).unwrap())
                    .max()
                    .unwrap();
                assert_eq!(block.header.commit_timestamp, max_ts);

                // Verify keys are sorted
                for window in keys.windows(2) {
                    assert!(window[0] <= window[1], "keys should be sorted");
                }
            }
        }
    }

    #[tokio::test]
    async fn flush_splits_large_memtable_into_multiple_blocks() {
        let dir = tempfile::tempdir().unwrap();
        // Use a very small SST size to force splitting
        let config = FlushConfig {
            data_dir: dir.path().to_path_buf(),
            sst_size_bytes: 512, // Very small — will force multiple blocks
            max_rows_per_block: 10,
        };

        let num_entries = 50;
        let memtable = create_test_memtable(num_entries);
        memtable.seal();

        let result = flush_memtable(&memtable, &config, 1, &test_io(), &[]).await.unwrap();

        // Should have multiple SST files
        assert!(
            result.sst_paths.len() > 1,
            "expected multiple SST files, got {}",
            result.sst_paths.len()
        );
        assert_eq!(result.rows_flushed, num_entries);

        // Verify total row count across all blocks in all SSTs
        let mut total_rows = 0;
        for sst_path in &result.sst_paths {
            let data = tokio::fs::read(sst_path).await.unwrap();
            let block_index = crate::sst::SstBlockIndex::from_file_data(&data).unwrap();
            for entry in &block_index.entries {
                let start = entry.file_offset as usize;
                let end = start + entry.block_len as usize;
                let block = PaxBlock::deserialize(&data[start..end]).unwrap();
                total_rows += block.header.row_count as usize;
            }
        }
        assert_eq!(total_rows, num_entries);
    }

    #[tokio::test]
    async fn flush_sst_file_naming_convention() {
        let dir = tempfile::tempdir().unwrap();
        let config = FlushConfig {
            data_dir: dir.path().to_path_buf(),
            sst_size_bytes: 64 * 1024 * 1024,
            max_rows_per_block: 1_000_000,
        };

        let memtable = create_test_memtable(10);
        let result = flush_memtable(&memtable, &config, 1, &test_io(), &[]).await.unwrap();

        // SST files should follow the sst_N.pax naming convention
        for path in &result.sst_paths {
            let filename = path.file_name().unwrap().to_str().unwrap();
            assert!(filename.starts_with("sst_"), "SST filename should start with sst_");
            assert!(filename.ends_with(".pax"), "SST filename should end with .pax");
        }
    }

    #[tokio::test]
    async fn flush_handles_tombstones() {
        let dir = tempfile::tempdir().unwrap();
        let config = FlushConfig {
            data_dir: dir.path().to_path_buf(),
            sst_size_bytes: 64 * 1024 * 1024,
            max_rows_per_block: 1_000_000,
        };

        let memtable = Memtable::new(1024 * 1024 * 1024);
        // Insert a live value
        memtable.put(b"key-a".to_vec(), 1, Some(b"value-a".to_vec()));
        // Insert a tombstone
        memtable.put(b"key-b".to_vec(), 2, None);
        // Insert another live value
        memtable.put(b"key-c".to_vec(), 3, Some(b"value-c".to_vec()));

        let result = flush_memtable(&memtable, &config, 10, &test_io(), &[]).await.unwrap();

        assert_eq!(result.rows_flushed, 3);
        assert_eq!(result.sst_paths.len(), 1);

        // Read back and verify via block index
        let data = tokio::fs::read(&result.sst_paths[0]).await.unwrap();
        let block_index = crate::sst::SstBlockIndex::from_file_data(&data).unwrap();
        assert!(block_index.block_count() > 0);

        // Read the first block (all 3 rows should be in one block with max_rows_per_block=1M)
        let entry = &block_index.entries[0];
        let block_bytes = &data[entry.file_offset as usize..(entry.file_offset as usize + entry.block_len as usize)];
        let block = PaxBlock::deserialize(block_bytes).unwrap();
        assert_eq!(block.header.row_count, 3);

        let keys = block.read_column(0).unwrap();
        let values = block.read_column(1).unwrap();

        // key-b should have an empty value (tombstone)
        assert_eq!(keys[1], b"key-b");
        assert!(values[1].is_empty(), "tombstone value should be empty");
    }

    #[tokio::test]
    async fn flush_with_wal_writes_checkpoint_and_truncates() {
        let dir = tempfile::tempdir().unwrap();
        let data_dir = dir.path().join("data");
        let wal_path = dir.path().join("wal.log");

        let flush_config = FlushConfig {
            data_dir: data_dir.clone(),
            sst_size_bytes: 64 * 1024 * 1024,
            max_rows_per_block: 1_000_000,
        };

        let wal_config = WalWriterConfig {
            wal_path: wal_path.clone(),
            group_commit_interval: Duration::from_millis(5),
            checkpoint_size_bytes: 512 * 1024 * 1024,
            checkpoint_interval: Duration::from_secs(60),
            preallocate_bytes: 262144,
        };

        let wal_writer = WalWriter::new(wal_config).unwrap();

        // Write some WAL records (simulating writes before flush)
        use crate::wal::{DurabilityMode, WalRecordType};
        for i in 0..5 {
            wal_writer
                .append(
                    WalRecordType::RowPut,
                    format!("pre-flush-{}", i).into_bytes(),
                    DurabilityMode::Strict,
                )
                .await
                .unwrap();
        }

        let wal_size_before = wal_writer.current_size();
        assert!(wal_size_before > 0);

        // Create and flush a memtable
        let memtable = create_test_memtable(50);
        memtable.seal();

        let result = flush_memtable_with_wal(&memtable, &flush_config, 100, &wal_writer, &test_io())
            .await
            .unwrap();

        // Verify checkpoint was written
        assert!(result.checkpoint_seq_no.is_some());
        let cp_seq = result.checkpoint_seq_no.unwrap();
        assert!(cp_seq > 0);

        // Verify WAL was truncated (size should be smaller)
        let wal_size_after = wal_writer.current_size();
        assert!(
            wal_size_after < wal_size_before,
            "WAL should be truncated after checkpoint: before={}, after={}",
            wal_size_before,
            wal_size_after
        );

        // Verify SST files were written
        assert!(!result.sst_paths.is_empty());
        assert_eq!(result.rows_flushed, 50);

        wal_writer.shutdown();
    }

    #[tokio::test]
    async fn checkpoint_advances_wal_truncation_point() {
        let dir = tempfile::tempdir().unwrap();
        let data_dir = dir.path().join("data");
        let wal_path = dir.path().join("wal.log");

        let flush_config = FlushConfig {
            data_dir: data_dir.clone(),
            sst_size_bytes: 64 * 1024 * 1024,
            max_rows_per_block: 1_000_000,
        };

        let wal_config = WalWriterConfig {
            wal_path: wal_path.clone(),
            group_commit_interval: Duration::from_millis(5),
            checkpoint_size_bytes: 512 * 1024 * 1024,
            checkpoint_interval: Duration::from_secs(60),
            preallocate_bytes: 262144,
        };

        let wal_writer = WalWriter::new(wal_config).unwrap();

        use crate::wal::{DurabilityMode, WalRecordType};

        // Phase 1: Write records, flush, checkpoint
        for i in 0..10 {
            wal_writer
                .append(
                    WalRecordType::RowPut,
                    format!("phase1-{}", i).into_bytes(),
                    DurabilityMode::Strict,
                )
                .await
                .unwrap();
        }

        let memtable1 = create_test_memtable(20);
        memtable1.seal();
        let result1 = flush_memtable_with_wal(&memtable1, &flush_config, 50, &wal_writer, &test_io())
            .await
            .unwrap();
        let cp1 = result1.checkpoint_seq_no.unwrap();

        // Phase 2: Write more records, flush again
        for i in 0..5 {
            wal_writer
                .append(
                    WalRecordType::RowPut,
                    format!("phase2-{}", i).into_bytes(),
                    DurabilityMode::Strict,
                )
                .await
                .unwrap();
        }

        let memtable2 = create_test_memtable(10);
        memtable2.seal();
        let result2 = flush_memtable_with_wal(&memtable2, &flush_config, 100, &wal_writer, &test_io())
            .await
            .unwrap();
        let cp2 = result2.checkpoint_seq_no.unwrap();

        // Second checkpoint should have a higher sequence number
        assert!(
            cp2 > cp1,
            "second checkpoint seq_no ({}) should be > first ({})",
            cp2,
            cp1
        );

        // WAL should only contain records from after the last checkpoint
        wal_writer.shutdown();

        let (recovered, _) = crate::wal::recover_wal(&wal_path).unwrap();
        // After the second checkpoint+truncate, there should be no records
        // after the checkpoint (we didn't write any after the second flush)
        assert!(
            recovered.is_empty(),
            "no records should remain after the last checkpoint, got {}",
            recovered.len()
        );
    }

    #[test]
    fn split_into_blocks_respects_max_rows() {
        let config = FlushConfig {
            data_dir: PathBuf::from("/tmp"),
            sst_size_bytes: u64::MAX, // Don't split by size
            max_rows_per_block: 5,
        };

        let entries: Vec<(Vec<u8>, VersionedValue)> = (0..12)
            .map(|i| {
                (
                    format!("key-{:03}", i).into_bytes(),
                    VersionedValue::new(i as u64, Some(format!("val-{}", i).into_bytes())),
                )
            })
            .collect();

        let blocks = split_into_blocks(&entries, &config, &[]);

        // 12 entries with max 5 per block → 3 blocks (5, 5, 2)
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].len(), 5);
        assert_eq!(blocks[1].len(), 5);
        assert_eq!(blocks[2].len(), 2);
    }

    #[test]
    fn split_into_blocks_single_block_when_small() {
        let config = FlushConfig {
            data_dir: PathBuf::from("/tmp"),
            sst_size_bytes: 64 * 1024 * 1024,
            max_rows_per_block: 1_000_000,
        };

        let entries: Vec<(Vec<u8>, VersionedValue)> = (0..10)
            .map(|i| {
                (
                    format!("key-{:03}", i).into_bytes(),
                    VersionedValue::new(i as u64, Some(b"small".to_vec())),
                )
            })
            .collect();

        let blocks = split_into_blocks(&entries, &config, &[]);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].len(), 10);
    }

    #[test]
    fn split_into_blocks_empty_entries() {
        let config = FlushConfig::default();
        let entries: Vec<(Vec<u8>, VersionedValue)> = Vec::new();
        let blocks = split_into_blocks(&entries, &config, &[]);
        assert!(blocks.is_empty());
    }

    #[test]
    fn entries_to_columns_converts_correctly() {
        let entries = vec![
            (
                b"key-a".to_vec(),
                VersionedValue::new(1, Some(b"val-a".to_vec())),
            ),
            (
                b"key-b".to_vec(),
                VersionedValue::new(2, None), // tombstone
            ),
            (
                b"key-c".to_vec(),
                VersionedValue::new(3, Some(b"val-c".to_vec())),
            ),
        ];

        let columns = entries_to_columns(&entries);
        assert_eq!(columns.len(), 3);

        // Key column
        assert_eq!(columns[0].col_type, ColumnType::Blob);
        assert_eq!(columns[0].values.len(), 3);
        assert_eq!(columns[0].values[0], b"key-a");
        assert_eq!(columns[0].values[1], b"key-b");
        assert_eq!(columns[0].values[2], b"key-c");

        // Value column
        assert_eq!(columns[1].col_type, ColumnType::Blob);
        assert_eq!(columns[1].values.len(), 3);
        assert_eq!(columns[1].values[0], b"val-a");
        assert!(columns[1].values[1].is_empty()); // tombstone → empty
        assert_eq!(columns[1].values[2], b"val-c");

        // Timestamp column (2): per-row MVCC commit timestamps, 8-byte LE.
        assert_eq!(columns[2].col_type, ColumnType::Blob);
        assert_eq!(columns[2].values.len(), 3);
        assert_eq!(decode_row_ts(&columns[2].values[0]), Some(1));
        assert_eq!(decode_row_ts(&columns[2].values[1]), Some(2));
        assert_eq!(decode_row_ts(&columns[2].values[2]), Some(3));
    }
}
