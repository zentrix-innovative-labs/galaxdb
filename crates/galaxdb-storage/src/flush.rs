//! Memtable flush pipeline for GalaxDB.
//!
//! Converts a sealed memtable into SST files (PAX blocks on disk) and
//! integrates with the WAL for checkpoint and truncation.
//!
//! ## Flush Pipeline
//!
//! 1. Get all entries from the sealed memtable (sorted by primary key via `iter_all()`)
//! 2. Group entries into PAX blocks (respecting the configured SST target size)
//! 3. Write each PAX block to disk as an SST file
//! 4. After successful flush, write a CHECKPOINT record to the WAL
//! 5. Truncate the WAL to the checkpoint point
//! 6. Notify the MemtableManager that flush is complete (releases back-pressure)
//!
//! ## TDE Encryption
//!
//! TDE encryption is not yet implemented (Task 12). The flush pipeline includes
//! a hook/placeholder for encryption but writes unencrypted data for now.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use galaxdb_common::{BlockId, ColumnType, GalaxError, GalaxResult, Timestamp};

use crate::memtable::{Memtable, VersionedValue};
use crate::pax::{ColumnData, PaxBlock};
use crate::wal::WalWriter;

/// Configuration for the flush pipeline.
#[derive(Debug, Clone)]
pub struct FlushConfig {
    /// Directory where SST files are written.
    pub data_dir: PathBuf,
    /// Target SST file size in bytes (default: 64 MB).
    /// Each PAX block is written as a separate SST file.
    pub sst_size_bytes: u64,
    /// Maximum number of rows per PAX block.
    /// This is derived from `sst_size_bytes` and average row size,
    /// but we also enforce a hard cap for safety.
    pub max_rows_per_block: usize,
}

impl Default for FlushConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("galaxdb_data"),
            sst_size_bytes: 64 * 1024 * 1024, // 64 MB
            max_rows_per_block: 1_000_000,     // safety cap
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
fn allocate_block_id() -> BlockId {
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
}

/// Placeholder hook for TDE encryption.
///
/// In v1, this is a no-op that returns the data unchanged. When Task 12
/// (TDE encryption) is implemented, this will encrypt the PAX block bytes
/// using AES-256-GCM before writing to disk.
fn encrypt_block_data(data: &[u8]) -> GalaxResult<Vec<u8>> {
    // TODO(Task 12): Encrypt with AES-256-GCM via TdeModule
    Ok(data.to_vec())
}

/// Convert memtable entries into columnar ColumnData for PAX block writing.
///
/// For v1, each row is treated as having two columns:
/// - Column 0: key (Blob type)
/// - Column 1: value (Blob type)
///
/// Tombstones (entries where the latest value is `None`) are stored with
/// an empty value column to preserve the deletion marker in the SST.
fn entries_to_columns(entries: &[(Vec<u8>, VersionedValue)]) -> Vec<ColumnData> {
    let mut key_values: Vec<Vec<u8>> = Vec::with_capacity(entries.len());
    let mut val_values: Vec<Vec<u8>> = Vec::with_capacity(entries.len());

    for (key, versioned) in entries {
        key_values.push(key.clone());
        // For tombstones, store an empty byte vector.
        // The actual tombstone semantics are tracked by the version chain;
        // at the SST level we just need the key present.
        val_values.push(versioned.value.clone().unwrap_or_default());
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
    ]
}

/// Split sorted entries into chunks that respect the SST size target.
///
/// Each chunk will become one PAX block / SST file.
fn split_into_blocks<'a>(
    entries: &'a [(Vec<u8>, VersionedValue)],
    config: &FlushConfig,
) -> Vec<&'a [(Vec<u8>, VersionedValue)]> {
    if entries.is_empty() {
        return Vec::new();
    }

    let mut blocks = Vec::new();
    let mut start = 0;
    let mut current_size: u64 = 0;

    for (i, (key, versioned)) in entries.iter().enumerate() {
        let entry_size = key.len() as u64
            + versioned.value.as_ref().map_or(0, |v| v.len()) as u64
            + 16; // overhead

        current_size += entry_size;

        let row_count = i - start + 1;
        let should_split = (current_size >= config.sst_size_bytes
            || row_count >= config.max_rows_per_block)
            && row_count > 0;

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

/// Flush a sealed memtable to SST files on disk.
///
/// This is the core flush pipeline:
/// 1. Extract all entries from the memtable (already sorted by key)
/// 2. Split entries into chunks based on SST size target
/// 3. For each chunk, create a PAX block and write it to disk
/// 4. Return the flush result with paths and metadata
///
/// This function does NOT handle WAL checkpoint or MemtableManager notification.
/// Use [`flush_memtable_with_wal`] for the full pipeline.
pub async fn flush_memtable(
    memtable: &Memtable,
    config: &FlushConfig,
    commit_timestamp: Timestamp,
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
        });
    }

    // Ensure the data directory exists
    tokio::fs::create_dir_all(&config.data_dir)
        .await
        .map_err(GalaxError::Io)?;

    // Step 2: Split entries into block-sized chunks
    let chunks = split_into_blocks(&entries, config);

    let mut sst_paths = Vec::with_capacity(chunks.len());
    let mut block_ids = Vec::with_capacity(chunks.len());
    let mut total_bytes: u64 = 0;
    let total_rows = entries.len();

    // Step 3: Write each chunk as a PAX block to disk
    for chunk in chunks {
        let block_id = allocate_block_id();
        let columns = entries_to_columns(chunk);

        // Create the PAX block
        let pax_block = PaxBlock::write(block_id, commit_timestamp, &columns)?;

        // Serialize the block
        let block_bytes = pax_block.serialize()?;

        // Apply TDE encryption (placeholder — currently a no-op)
        let encrypted_bytes = encrypt_block_data(&block_bytes)?;

        // Write to disk
        let sst_filename = format!("sst_{}.pax", block_id);
        let sst_path = config.data_dir.join(&sst_filename);

        tokio::fs::write(&sst_path, &encrypted_bytes)
            .await
            .map_err(GalaxError::Io)?;

        total_bytes += encrypted_bytes.len() as u64;
        sst_paths.push(sst_path);
        block_ids.push(block_id);
    }

    Ok(FlushResult {
        sst_paths,
        block_ids,
        rows_flushed: total_rows,
        bytes_written: total_bytes,
        checkpoint_seq_no: None,
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
) -> GalaxResult<FlushResult> {
    // Step 1: Flush memtable to SST files
    let mut result = flush_memtable(memtable, config, commit_timestamp).await?;

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
    use std::time::Duration;

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
        let result = flush_memtable(&memtable, &config, 100).await.unwrap();

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

        let result = flush_memtable(&memtable, &config, 42).await.unwrap();

        assert_eq!(result.rows_flushed, num_entries);
        assert!(!result.sst_paths.is_empty());
        assert!(!result.block_ids.is_empty());
        assert!(result.bytes_written > 0);

        // Read back each SST file and verify it's a valid PAX block
        for sst_path in &result.sst_paths {
            assert!(sst_path.exists());

            let data = tokio::fs::read(sst_path).await.unwrap();
            let block = PaxBlock::deserialize(&data)
                .expect("SST file should contain a valid PAX block");

            // Verify block metadata
            assert_eq!(block.header.commit_timestamp, 42);
            assert_eq!(block.header.column_count, 2); // key + value columns
            assert!(block.header.row_count > 0);

            // Verify we can read back the columns
            let keys = block.read_column(0).unwrap();
            let values = block.read_column(1).unwrap();
            assert_eq!(keys.len(), block.header.row_count as usize);
            assert_eq!(values.len(), block.header.row_count as usize);

            // Verify keys are sorted
            for window in keys.windows(2) {
                assert!(window[0] <= window[1], "keys should be sorted");
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

        let result = flush_memtable(&memtable, &config, 1).await.unwrap();

        // Should have multiple SST files
        assert!(
            result.sst_paths.len() > 1,
            "expected multiple SST files, got {}",
            result.sst_paths.len()
        );
        assert_eq!(result.rows_flushed, num_entries);

        // Verify total row count across all blocks
        let mut total_rows = 0;
        for sst_path in &result.sst_paths {
            let data = tokio::fs::read(sst_path).await.unwrap();
            let block = PaxBlock::deserialize(&data).unwrap();
            total_rows += block.header.row_count as usize;
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
        let result = flush_memtable(&memtable, &config, 1).await.unwrap();

        for (path, block_id) in result.sst_paths.iter().zip(result.block_ids.iter()) {
            let filename = path.file_name().unwrap().to_str().unwrap();
            let expected = format!("sst_{}.pax", block_id);
            assert_eq!(filename, expected);
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

        let result = flush_memtable(&memtable, &config, 10).await.unwrap();

        assert_eq!(result.rows_flushed, 3);
        assert_eq!(result.sst_paths.len(), 1);

        // Read back and verify
        let data = tokio::fs::read(&result.sst_paths[0]).await.unwrap();
        let block = PaxBlock::deserialize(&data).unwrap();
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

        let result = flush_memtable_with_wal(&memtable, &flush_config, 100, &wal_writer)
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
        let result1 = flush_memtable_with_wal(&memtable1, &flush_config, 50, &wal_writer)
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
        let result2 = flush_memtable_with_wal(&memtable2, &flush_config, 100, &wal_writer)
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

        let blocks = split_into_blocks(&entries, &config);

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

        let blocks = split_into_blocks(&entries, &config);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].len(), 10);
    }

    #[test]
    fn split_into_blocks_empty_entries() {
        let config = FlushConfig::default();
        let entries: Vec<(Vec<u8>, VersionedValue)> = Vec::new();
        let blocks = split_into_blocks(&entries, &config);
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
        assert_eq!(columns.len(), 2);

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
    }
}
