//! Memtable — concurrent in-memory write buffer for GalaxDB.
//!
//! The memtable uses a 16-shard design where each shard contains a
//! `crossbeam_skiplist::SkipMap`. Shard selection is determined by
//! `xxh3_64(key) % 16`, distributing keys uniformly across shards.
//!
//! Each shard's `Mutex` protects MVCC version chain updates for keys
//! in that shard. The `SkipMap` itself is lock-free for concurrent reads.
//!
//! ## Seal & Back-pressure
//!
//! When the active memtable's byte size reaches the configured threshold
//! (default 64 MB), it is sealed and a new empty memtable is swapped in.
//! A `tokio::sync::Semaphore` tracks total sealed-but-unflushed bytes;
//! writers block when this exceeds the back-pressure limit (default 256 MB).
//!
//! ## Epoch Safety
//!
//! All reads copy value bytes out of the `Entry` handle immediately and
//! drop the handle before any async boundary, ensuring epoch-based memory
//! reclamation in crossbeam is not blocked.

mod versioned_value;

#[cfg(test)]
mod tests;

pub use versioned_value::VersionedValue;

use crossbeam_skiplist::SkipMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::Semaphore;
use xxhash_rust::xxh3::xxh3_64;

/// Number of shards used to distribute keys across independent skip maps.
const NUM_SHARDS: usize = 16;

/// Default seal threshold: 64 MB.
const DEFAULT_SEAL_THRESHOLD: u64 = 64 * 1024 * 1024;

/// Default back-pressure capacity: 256 MB.
const DEFAULT_BACK_PRESSURE_BYTES: u64 = 256 * 1024 * 1024;

/// A single shard containing a concurrent skip map protected by a mutex
/// for MVCC version chain updates.
pub struct Shard {
    /// The concurrent skip map for this shard. The SkipMap itself is lock-free
    /// for reads; the Mutex serializes version chain mutations for keys in
    /// this shard.
    map: SkipMap<Vec<u8>, VersionedValue>,
    /// Mutex that serializes write operations (insert/update) to version chains
    /// within this shard.
    write_lock: Mutex<()>,
}

impl Shard {
    fn new() -> Self {
        Self {
            map: SkipMap::new(),
            write_lock: Mutex::new(()),
        }
    }
}

/// The in-memory write buffer backed by 16 sharded concurrent skip maps.
///
/// Writers hash the key to select a shard, acquire the shard's write lock,
/// and insert/update the versioned value. Readers can access the skip map
/// without acquiring the write lock (lock-free reads).
pub struct Memtable {
    /// 16 shards, each with its own SkipMap and write Mutex.
    shards: Vec<Shard>,
    /// Current approximate byte size of all entries in this memtable.
    size: AtomicU64,
    /// Whether this memtable has been sealed (no more writes accepted).
    sealed: AtomicBool,
    /// The seal threshold in bytes.
    seal_threshold: u64,
}

impl Memtable {
    /// Creates a new empty memtable with the given seal threshold.
    pub fn new(seal_threshold: u64) -> Self {
        let shards = (0..NUM_SHARDS).map(|_| Shard::new()).collect();
        Self {
            shards,
            size: AtomicU64::new(0),
            sealed: AtomicBool::new(false),
            seal_threshold,
        }
    }

    /// Creates a new memtable with the default 64 MB seal threshold.
    pub fn with_default_threshold() -> Self {
        Self::new(DEFAULT_SEAL_THRESHOLD)
    }

    /// Returns the shard index for the given key.
    #[inline]
    fn shard_index(key: &[u8]) -> usize {
        (xxh3_64(key) % NUM_SHARDS as u64) as usize
    }

    /// Inserts or updates a key with a new versioned value.
    ///
    /// Acquires the shard's write lock to serialize version chain updates
    /// for keys in the same shard. If the key already exists, the new value
    /// is prepended to the version chain.
    ///
    /// Returns `true` if the memtable should be sealed after this write
    /// (i.e., size crossed the seal threshold).
    pub fn put(&self, key: Vec<u8>, timestamp: u64, value: Option<Vec<u8>>) -> bool {
        if self.sealed.load(Ordering::Acquire) {
            // Sealed memtables do not accept writes.
            // The caller should have already swapped to a new memtable.
            return false;
        }

        let shard_idx = Self::shard_index(&key);
        let shard = &self.shards[shard_idx];

        // Estimate the size contribution of this write.
        let entry_size = key.len() as u64
            + value.as_ref().map_or(0, |v| v.len() as u64)
            + 8 // timestamp
            + 8; // pointer overhead estimate

        // Acquire the shard write lock to serialize version chain updates.
        let _guard = shard.write_lock.lock().expect("shard lock poisoned");

        // Check if the key already exists in this shard's skip map.
        if let Some(entry) = shard.map.get(&key) {
            // Key exists — prepend new version to the chain.
            let existing = entry.value().clone();
            drop(entry); // Drop the Entry handle immediately (epoch safety).

            let new_value = VersionedValue {
                timestamp,
                value,
                prev: Some(Box::new(existing)),
            };
            shard.map.insert(key, new_value);
        } else {
            // New key — create a fresh version chain.
            let new_value = VersionedValue {
                timestamp,
                value,
                prev: None,
            };
            shard.map.insert(key, new_value);
        }

        // Update the size counter.
        let new_size = self.size.fetch_add(entry_size, Ordering::AcqRel) + entry_size;

        // Check if we crossed the seal threshold.
        new_size >= self.seal_threshold
    }

    /// Reads the latest value for the given key, copying bytes out immediately.
    ///
    /// The `Entry` handle from the skip map is dropped before returning,
    /// ensuring epoch-based memory reclamation is not blocked across async
    /// boundaries.
    ///
    /// Returns `None` if the key is not found. Returns `Some(None)` if the
    /// latest version is a tombstone. Returns `Some(Some(bytes))` for a
    /// live value.
    pub fn get(&self, key: &[u8]) -> Option<Option<Vec<u8>>> {
        let shard_idx = Self::shard_index(key);
        let shard = &self.shards[shard_idx];

        // Look up the key in the skip map.
        let entry = shard.map.get(key)?;

        // Copy the value bytes out immediately (epoch safety).
        let result = entry.value().value.clone();

        // Drop the Entry handle before returning.
        drop(entry);

        Some(result)
    }

    /// Reads the value for the given key at a specific timestamp (MVCC read).
    ///
    /// Walks the version chain to find the latest version with
    /// `timestamp <= read_ts`. Returns `None` if no such version exists.
    pub fn get_at(&self, key: &[u8], read_ts: u64) -> Option<Option<Vec<u8>>> {
        let shard_idx = Self::shard_index(key);
        let shard = &self.shards[shard_idx];

        let entry = shard.map.get(key)?;

        // Walk the version chain to find the right version.
        let versioned = entry.value();
        let result = versioned.get_at(read_ts);

        // Drop the Entry handle before returning.
        drop(entry);

        result
    }

    /// Seals this memtable, preventing further writes.
    pub fn seal(&self) {
        self.sealed.store(true, Ordering::Release);
    }

    /// Returns whether this memtable is sealed.
    pub fn is_sealed(&self) -> bool {
        self.sealed.load(Ordering::Acquire)
    }

    /// Returns the current approximate byte size of this memtable.
    pub fn size(&self) -> u64 {
        self.size.load(Ordering::Acquire)
    }

    /// Returns the number of shards (always 16).
    pub fn num_shards(&self) -> usize {
        NUM_SHARDS
    }

    /// Returns the seal threshold in bytes.
    pub fn seal_threshold(&self) -> u64 {
        self.seal_threshold
    }

    /// Returns an iterator over all key-value pairs across all shards.
    ///
    /// Values are copied out of the skip map entries immediately.
    /// This is primarily used during flush to SST.
    pub fn iter_all(&self) -> Vec<(Vec<u8>, VersionedValue)> {
        let mut entries = Vec::new();
        for shard in &self.shards {
            for entry in shard.map.iter() {
                let key = entry.key().clone();
                let value = entry.value().clone();
                // Drop entry handle immediately.
                drop(entry);
                entries.push((key, value));
            }
        }
        // Sort by key for ordered iteration (needed for flush).
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        entries
    }
}

/// Manages the active memtable and a queue of sealed memtables awaiting flush.
///
/// When the active memtable reaches the seal threshold, it is atomically
/// swapped to a new empty memtable and the sealed one is enqueued for flush.
///
/// Back-pressure is enforced via a `tokio::sync::Semaphore` that tracks
/// total sealed-but-unflushed bytes. Writers must acquire permits before
/// writing; if the total exceeds the configured limit, writers block until
/// flush progress releases permits.
pub struct MemtableManager {
    /// The currently active (writable) memtable.
    active: Mutex<Arc<Memtable>>,
    /// Queue of sealed memtables awaiting flush, in seal order.
    sealed_queue: Mutex<Vec<Arc<Memtable>>>,
    /// Semaphore tracking available back-pressure capacity in bytes.
    /// Initialized with `back_pressure_bytes` permits. Each sealed memtable
    /// consumes permits equal to its byte size. Writers block when no
    /// permits are available.
    back_pressure: Arc<Semaphore>,
    /// The configured back-pressure limit in bytes.
    back_pressure_bytes: u64,
    /// The seal threshold for new memtables.
    seal_threshold: u64,
}

impl MemtableManager {
    /// Creates a new `MemtableManager` with the given thresholds.
    pub fn new(seal_threshold: u64, back_pressure_bytes: u64) -> Self {
        Self {
            active: Mutex::new(Arc::new(Memtable::new(seal_threshold))),
            sealed_queue: Mutex::new(Vec::new()),
            back_pressure: Arc::new(Semaphore::new(back_pressure_bytes as usize)),
            back_pressure_bytes,
            seal_threshold,
        }
    }

    /// Creates a new `MemtableManager` with default thresholds (64 MB seal, 256 MB back-pressure).
    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_SEAL_THRESHOLD, DEFAULT_BACK_PRESSURE_BYTES)
    }

    /// Returns a reference to the currently active memtable.
    pub fn active(&self) -> Arc<Memtable> {
        self.active.lock().expect("active lock poisoned").clone()
    }

    /// Writes a key-value pair to the active memtable.
    ///
    /// If the write causes the memtable to cross the seal threshold,
    /// the memtable is sealed and swapped to a new empty one.
    ///
    /// This method acquires back-pressure permits before writing. If the
    /// total sealed-but-unflushed bytes exceed the limit, this call blocks
    /// until flush progress frees capacity.
    pub async fn put(&self, key: Vec<u8>, timestamp: u64, value: Option<Vec<u8>>) {
        let entry_size = key.len()
            + value.as_ref().map_or(0, |v| v.len())
            + 16; // timestamp + overhead

        // Acquire back-pressure permits for this write's size contribution.
        // This blocks if sealed-but-unflushed bytes exceed the limit.
        let permit = self
            .back_pressure
            .clone()
            .acquire_many_owned(entry_size as u32)
            .await
            .expect("semaphore closed");

        // We release the permit immediately — the back-pressure accounting
        // happens at seal time, not per-write. Individual write permits are
        // just to ensure we don't exceed capacity.
        drop(permit);

        let should_seal = {
            let active = self.active.lock().expect("active lock poisoned");
            active.put(key, timestamp, value)
        };

        if should_seal {
            self.seal_active();
        }
    }

    /// Reads the latest value for the given key from the active memtable.
    ///
    /// If not found in the active memtable, searches sealed memtables
    /// in reverse order (most recently sealed first).
    pub fn get(&self, key: &[u8]) -> Option<Option<Vec<u8>>> {
        // Check active memtable first.
        {
            let active = self.active.lock().expect("active lock poisoned");
            if let Some(result) = active.get(key) {
                return Some(result);
            }
        }

        // Check sealed memtables in reverse order.
        let sealed = self.sealed_queue.lock().expect("sealed lock poisoned");
        for memtable in sealed.iter().rev() {
            if let Some(result) = memtable.get(key) {
                return Some(result);
            }
        }

        None
    }

    /// Reads the value for the given key at a specific MVCC timestamp.
    pub fn get_at(&self, key: &[u8], read_ts: u64) -> Option<Option<Vec<u8>>> {
        // Check active memtable first.
        {
            let active = self.active.lock().expect("active lock poisoned");
            if let Some(result) = active.get_at(key, read_ts) {
                return Some(result);
            }
        }

        // Check sealed memtables in reverse order.
        let sealed = self.sealed_queue.lock().expect("sealed lock poisoned");
        for memtable in sealed.iter().rev() {
            if let Some(result) = memtable.get_at(key, read_ts) {
                return Some(result);
            }
        }

        None
    }

    /// Seals the active memtable and swaps in a new empty one.
    ///
    /// The sealed memtable is added to the flush queue. Back-pressure
    /// permits equal to the sealed memtable's size are consumed from
    /// the semaphore by forgetting the acquired permit (so it stays
    /// consumed until flush releases it).
    ///
    /// This is public so the Engine can trigger a seal before flushing
    /// the memtable to SST files.
    pub fn seal_active(&self) {
        let mut active_guard = self.active.lock().expect("active lock poisoned");
        let mut sealed_guard = self.sealed_queue.lock().expect("sealed lock poisoned");

        // Only seal if the active memtable hasn't already been sealed
        // (another thread may have beaten us to it).
        if active_guard.is_sealed() {
            return;
        }

        let old_active = active_guard.clone();
        old_active.seal();

        let sealed_size = old_active.size();
        let consume = sealed_size.min(u32::MAX as u64) as u32;

        // Consume back-pressure permits for the sealed memtable's size.
        // We forget the permit so it stays consumed — future writers will
        // block if total sealed-but-unflushed bytes exceed the limit.
        if let Ok(permit) = self.back_pressure.clone().try_acquire_many_owned(consume) {
            permit.forget();
        }

        // Enqueue the sealed memtable for flush.
        sealed_guard.push(old_active);

        // Swap in a new empty memtable.
        *active_guard = Arc::new(Memtable::new(self.seal_threshold));
    }

    /// Called by the flush subsystem after a sealed memtable has been
    /// successfully flushed to disk. Removes the memtable from the
    /// sealed queue and releases back-pressure permits.
    pub fn on_flush_complete(&self, flushed_size: u64) {
        let mut sealed_guard = self.sealed_queue.lock().expect("sealed lock poisoned");
        if !sealed_guard.is_empty() {
            sealed_guard.remove(0);
        }

        // Release back-pressure permits that were consumed at seal time.
        self.back_pressure.add_permits(
            flushed_size.min(u32::MAX as u64) as usize,
        );
    }

    /// Returns the number of sealed memtables awaiting flush.
    pub fn sealed_count(&self) -> usize {
        self.sealed_queue.lock().expect("sealed lock poisoned").len()
    }

    /// Returns the total size of all sealed memtables in bytes.
    pub fn sealed_total_bytes(&self) -> u64 {
        self.sealed_queue
            .lock()
            .expect("sealed lock poisoned")
            .iter()
            .map(|m| m.size())
            .sum()
    }

    /// Returns the back-pressure semaphore's available permits.
    pub fn available_back_pressure(&self) -> usize {
        self.back_pressure.available_permits()
    }

    /// Returns the configured back-pressure limit in bytes.
    pub fn back_pressure_bytes(&self) -> u64 {
        self.back_pressure_bytes
    }

    /// Returns the seal threshold in bytes.
    pub fn seal_threshold(&self) -> u64 {
        self.seal_threshold
    }

    /// Returns the next sealed memtable to flush (if any), without removing it.
    pub fn peek_sealed(&self) -> Option<Arc<Memtable>> {
        self.sealed_queue
            .lock()
            .expect("sealed lock poisoned")
            .first()
            .cloned()
    }
}
