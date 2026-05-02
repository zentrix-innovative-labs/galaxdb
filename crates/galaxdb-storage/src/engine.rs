//! Storage Engine facade — the unified API that the SQL executor calls.
//!
//! Connects: Memtable + WAL + ART Index + Flush + Buffer Pool + Compaction
//! into a single coherent interface for reading and writing rows.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use galaxdb_common::{GalaxError, GalaxResult, Timestamp};

use crate::art::{ArtIndex, RowLocation};
use crate::memtable::MemtableManager;
use crate::wal::{DurabilityMode, WalRecordType, WalWriter, WalWriterConfig};

/// Configuration for the storage engine.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub data_dir: PathBuf,
    pub memtable_size_bytes: u64,
    pub back_pressure_bytes: u64,
    pub wal_group_commit_ms: u64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("galaxdb_data"),
            memtable_size_bytes: 64 * 1024 * 1024,
            back_pressure_bytes: 256 * 1024 * 1024,
            wal_group_commit_ms: 10,
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

/// The storage engine — unified read/write API.
pub struct Engine {
    config: EngineConfig,
    memtable_mgr: MemtableManager,
    art: Arc<ArtIndex>,
    wal: Arc<WalWriter>,
    next_timestamp: AtomicU64,
    row_count: AtomicU64,
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

        let wal = WalWriter::new(wal_config).map_err(GalaxError::Io)?;

        let memtable_mgr = MemtableManager::new(
            config.memtable_size_bytes,
            config.back_pressure_bytes,
        );

        Ok(Self {
            config,
            memtable_mgr,
            art: Arc::new(ArtIndex::new()),
            wal: Arc::new(wal),
            next_timestamp: AtomicU64::new(1),
            row_count: AtomicU64::new(0),
        })
    }

    /// Allocate a new monotonic timestamp.
    fn next_ts(&self) -> Timestamp {
        self.next_timestamp.fetch_add(1, Ordering::SeqCst)
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

    /// Insert a row synchronously (memtable + ART, WAL skipped).
    /// Use this from sync contexts (embedded mode, Python FFI).
    /// For full durability, use the async `put` method.
    pub fn put_sync(&self, key: Vec<u8>, value: Vec<u8>) -> GalaxResult<Timestamp> {
        let ts = self.next_ts();

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

    /// Get a row by primary key. Reads from memtable (via ART index).
    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        // Check ART index first
        let _location = self.art.lookup(key)?;

        // Read from memtable (checks active + sealed)
        match self.memtable_mgr.get(key) {
            Some(Some(value)) => Some(value),
            Some(None) => None, // tombstone
            None => None,       // not found
        }
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

    /// Scan all rows (for SELECT * without filter).
    /// Returns keys and values from the memtable.
    pub fn scan_all(&self) -> Vec<(Vec<u8>, Vec<u8>)> {
        let active = self.memtable_mgr.active();
        let entries = active.iter_all();

        entries
            .into_iter()
            .filter_map(|(key, versioned)| {
                versioned.value.map(|v| (key, v))
            })
            .collect()
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
}
