//! Blob Log — KV separation for large values (BVLSM pattern).
//!
//! When a value exceeds the blob threshold (default 1 KB), it is written
//! directly to the blob log during WAL entry construction. The memtable
//! stores only a content hash + [`BlobRef`] instead of the full value.
//!
//! ## Multi-Queue Parallel Writers
//!
//! The blob log uses multiple writer queues (default 4) for parallel writes.
//! Each write is assigned to a queue via round-robin, and each queue writes
//! to its own blob file. This follows the BVLSM design for high write
//! throughput.
//!
//! ## Garbage Collection
//!
//! A background GC task scans blob files. When discardable space (values
//! whose keys have been compacted away) exceeds 50% of a file, the live
//! values are copied to a new file and the old file is deleted.
//!
//! ## Content Addressing
//!
//! Values are content-addressed using XXH3-128 (16 bytes). The hash is
//! stored alongside the [`BlobRef`] in the memtable value slot.

#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, RwLock};

use xxhash_rust::xxh3::{xxh3_64, xxh3_128};

/// Default number of parallel writer queues.
pub const DEFAULT_NUM_QUEUES: usize = 4;

/// Default blob threshold in bytes. Values larger than this are separated.
pub const DEFAULT_BLOB_THRESHOLD: usize = 1024;

/// GC trigger: compact a blob file when discardable space exceeds this ratio.
pub const GC_DISCARD_RATIO: f64 = 0.50;

/// Magic bytes prepended to each blob entry on disk for validation.
const BLOB_ENTRY_MAGIC: u32 = 0x424C4F42; // "BLOB"

/// Size of a serialized [`BlobRef`] in bytes.
/// file_id (8) + offset (8) + length (4) = 20 bytes.
pub const BLOB_REF_SIZE: usize = 20;

/// Size of the content hash in bytes (XXH3-128).
pub const CONTENT_HASH_SIZE: usize = 16;

/// Total size of the blob reference stored in the memtable:
/// content_hash (16) + BlobRef (20) = 36 bytes.
/// We use a 1-byte prefix tag (0xFF) to distinguish blob refs from inline values.
pub const BLOB_MARKER_SIZE: usize = 1 + CONTENT_HASH_SIZE + BLOB_REF_SIZE;

/// Prefix byte used to identify a blob reference in a memtable value slot.
/// This byte value (0xFF) is chosen because it is unlikely to appear as the
/// first byte of a normal value at exactly the right length.
pub const BLOB_REF_TAG: u8 = 0xFF;

/// A reference to a value stored in the blob log.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlobRef {
    /// The blob file ID that contains this value.
    pub file_id: u64,
    /// Byte offset within the blob file where the entry starts.
    pub offset: u64,
    /// Length of the value in bytes (uncompressed).
    pub length: u32,
}

impl BlobRef {
    /// Serialize this BlobRef to a fixed-size byte array.
    pub fn to_bytes(&self) -> [u8; BLOB_REF_SIZE] {
        let mut buf = [0u8; BLOB_REF_SIZE];
        buf[0..8].copy_from_slice(&self.file_id.to_le_bytes());
        buf[8..16].copy_from_slice(&self.offset.to_le_bytes());
        buf[16..20].copy_from_slice(&self.length.to_le_bytes());
        buf
    }

    /// Deserialize a BlobRef from bytes.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < BLOB_REF_SIZE {
            return None;
        }
        let file_id = u64::from_le_bytes(data[0..8].try_into().ok()?);
        let offset = u64::from_le_bytes(data[8..16].try_into().ok()?);
        let length = u32::from_le_bytes(data[16..20].try_into().ok()?);
        Some(Self {
            file_id,
            offset,
            length,
        })
    }
}

/// Compute the content hash for a value using XXH3-128.
///
/// Returns a 16-byte hash that serves as the content address.
pub fn content_hash(data: &[u8]) -> [u8; CONTENT_HASH_SIZE] {
    let hash = xxh3_128(data);
    hash.to_le_bytes()
}

/// Encode a blob reference for storage in the memtable value slot.
///
/// Format: `[BLOB_REF_TAG (1 byte)][content_hash (16 bytes)][BlobRef (20 bytes)]`
///
/// Total: 37 bytes.
pub fn encode_blob_ref(hash: &[u8; CONTENT_HASH_SIZE], blob_ref: &BlobRef) -> Vec<u8> {
    let mut buf = Vec::with_capacity(BLOB_MARKER_SIZE);
    buf.push(BLOB_REF_TAG);
    buf.extend_from_slice(hash);
    buf.extend_from_slice(&blob_ref.to_bytes());
    buf
}

/// Check if a value stored in the memtable is a blob reference.
///
/// Returns `Some((content_hash, BlobRef))` if the value is a blob reference,
/// or `None` if it is an inline value.
pub fn decode_blob_ref(value: &[u8]) -> Option<([u8; CONTENT_HASH_SIZE], BlobRef)> {
    if value.len() != BLOB_MARKER_SIZE {
        return None;
    }
    if value[0] != BLOB_REF_TAG {
        return None;
    }
    let mut hash = [0u8; CONTENT_HASH_SIZE];
    hash.copy_from_slice(&value[1..1 + CONTENT_HASH_SIZE]);
    let blob_ref = BlobRef::from_bytes(&value[1 + CONTENT_HASH_SIZE..])?;
    Some((hash, blob_ref))
}

/// Check if a value should be separated into the blob log.
pub fn should_separate(value: &[u8], threshold: usize) -> bool {
    value.len() > threshold
}

/// On-disk format for a single blob entry:
/// ```text
/// [magic: u32][length: u32][content_hash: 16 bytes][value: length bytes][checksum: u64]
/// ```
const BLOB_ENTRY_HEADER_SIZE: usize = 4 + 4 + CONTENT_HASH_SIZE; // 24 bytes
const BLOB_ENTRY_FOOTER_SIZE: usize = 8; // checksum

/// A single writer queue that writes to its own blob file.
struct BlobWriter {
    /// The file ID for this writer's current blob file.
    file_id: u64,
    /// The file handle.
    file: File,
    /// Current write offset in the file.
    offset: u64,
    /// Path to the blob file.
    #[allow(dead_code)]
    path: PathBuf,
}

impl BlobWriter {
    /// Create a new blob writer for the given file ID.
    fn new(blob_dir: &Path, file_id: u64) -> io::Result<Self> {
        fs::create_dir_all(blob_dir)?;
        let path = blob_dir.join(format!("blob_{}.dat", file_id));
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&path)?;
        let offset = file.metadata()?.len();
        Ok(Self {
            file_id,
            file,
            offset,
            path,
        })
    }

    /// Write a value to this blob file. Returns the BlobRef.
    fn write_value(&mut self, value: &[u8], hash: &[u8; CONTENT_HASH_SIZE]) -> io::Result<BlobRef> {
        let entry_offset = self.offset;
        let length = value.len() as u32;

        // Build the entry
        let mut entry = Vec::with_capacity(BLOB_ENTRY_HEADER_SIZE + value.len() + BLOB_ENTRY_FOOTER_SIZE);
        entry.extend_from_slice(&BLOB_ENTRY_MAGIC.to_le_bytes());
        entry.extend_from_slice(&length.to_le_bytes());
        entry.extend_from_slice(hash);
        entry.extend_from_slice(value);

        // Compute checksum over everything before the checksum field
        let checksum = xxh3_64(&entry);
        entry.extend_from_slice(&checksum.to_le_bytes());

        self.file.write_all(&entry)?;
        self.file.flush()?;
        self.offset += entry.len() as u64;

        Ok(BlobRef {
            file_id: self.file_id,
            offset: entry_offset,
            length,
        })
    }
}

/// Metadata tracked per blob file for GC decisions.
#[derive(Debug, Clone)]
pub struct BlobFileStats {
    /// Total bytes of live (referenced) values in this file.
    pub live_bytes: u64,
    /// Total bytes of all values (live + dead) in this file.
    pub total_bytes: u64,
    /// Number of live entries.
    pub live_count: u64,
    /// Number of total entries.
    pub total_count: u64,
    /// Path to the blob file.
    pub path: PathBuf,
    /// File ID.
    pub file_id: u64,
}

impl BlobFileStats {
    /// Returns the fraction of space that is discardable (dead).
    pub fn discard_ratio(&self) -> f64 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        1.0 - (self.live_bytes as f64 / self.total_bytes as f64)
    }

    /// Returns true if this file should be garbage collected.
    pub fn needs_gc(&self) -> bool {
        self.discard_ratio() > GC_DISCARD_RATIO
    }
}

/// The blob log with multi-queue parallel writers.
///
/// Provides content-addressed storage for large values separated from the
/// LSM tree. Writers are distributed across multiple queues (files) using
/// round-robin assignment.
pub struct BlobLog {
    /// Directory where blob files are stored.
    blob_dir: PathBuf,
    /// The parallel writer queues, each protected by its own mutex.
    writers: Vec<Mutex<BlobWriter>>,
    /// Round-robin counter for writer assignment.
    next_writer: AtomicU64,
    /// Content-hash → BlobRef index for deduplication and lookup.
    index: RwLock<HashMap<[u8; CONTENT_HASH_SIZE], BlobRef>>,
    /// Next file ID for new blob files.
    next_file_id: AtomicU64,
    /// Set of live blob refs (used by GC to determine what's still referenced).
    live_refs: RwLock<HashMap<BlobRef, u64>>, // BlobRef → reference count
    /// Blob threshold in bytes.
    threshold: usize,
}

impl BlobLog {
    /// Create a new BlobLog with the specified number of writer queues.
    pub fn new(blob_dir: PathBuf, num_queues: usize, threshold: usize) -> io::Result<Self> {
        let num_queues = if num_queues == 0 { DEFAULT_NUM_QUEUES } else { num_queues };

        let mut writers = Vec::with_capacity(num_queues);
        for i in 0..num_queues {
            let file_id = i as u64;
            writers.push(Mutex::new(BlobWriter::new(&blob_dir, file_id)?));
        }

        Ok(Self {
            blob_dir,
            writers,
            next_writer: AtomicU64::new(0),
            index: RwLock::new(HashMap::new()),
            next_file_id: AtomicU64::new(num_queues as u64),
            live_refs: RwLock::new(HashMap::new()),
            threshold,
        })
    }

    /// Create a new BlobLog with default settings (4 queues, 1 KB threshold).
    pub fn with_defaults(blob_dir: PathBuf) -> io::Result<Self> {
        Self::new(blob_dir, DEFAULT_NUM_QUEUES, DEFAULT_BLOB_THRESHOLD)
    }

    /// Returns the blob threshold in bytes.
    pub fn threshold(&self) -> usize {
        self.threshold
    }

    /// Check if a value should be separated into the blob log.
    pub fn should_separate(&self, value: &[u8]) -> bool {
        should_separate(value, self.threshold)
    }

    /// Write a value to the blob log.
    ///
    /// Computes the content hash, checks for deduplication, and writes to
    /// the next writer queue via round-robin. Returns the content hash and
    /// BlobRef.
    ///
    /// If the value is already stored (same content hash), returns the
    /// existing BlobRef without writing again.
    pub fn write(&self, value: &[u8]) -> io::Result<([u8; CONTENT_HASH_SIZE], BlobRef)> {
        let hash = content_hash(value);

        // Check for deduplication
        {
            let index = self.index.read().expect("index lock poisoned");
            if let Some(existing_ref) = index.get(&hash) {
                // Value already exists — increment reference count
                let mut live = self.live_refs.write().expect("live_refs lock poisoned");
                *live.entry(*existing_ref).or_insert(0) += 1;
                return Ok((hash, *existing_ref));
            }
        }

        // Select writer via round-robin
        let writer_idx = (self.next_writer.fetch_add(1, Ordering::Relaxed)
            % self.writers.len() as u64) as usize;

        let blob_ref = {
            let mut writer = self.writers[writer_idx].lock().expect("writer lock poisoned");
            writer.write_value(value, &hash)?
        };

        // Update the index
        {
            let mut index = self.index.write().expect("index lock poisoned");
            index.insert(hash, blob_ref);
        }

        // Track as live reference
        {
            let mut live = self.live_refs.write().expect("live_refs lock poisoned");
            *live.entry(blob_ref).or_insert(0) += 1;
        }

        Ok((hash, blob_ref))
    }

    /// Read a value from the blob log using a BlobRef.
    ///
    /// Opens the blob file, seeks to the entry offset, reads and validates
    /// the entry (magic number, checksum, content hash), and returns the
    /// value bytes.
    pub fn read(&self, blob_ref: &BlobRef) -> io::Result<Vec<u8>> {
        let path = self.blob_dir.join(format!("blob_{}.dat", blob_ref.file_id));
        let mut file = File::open(&path)?;
        file.seek(SeekFrom::Start(blob_ref.offset))?;

        // Read header
        let mut header = [0u8; BLOB_ENTRY_HEADER_SIZE];
        file.read_exact(&mut header)?;

        // Validate magic
        let magic = u32::from_le_bytes(header[0..4].try_into().unwrap());
        if magic != BLOB_ENTRY_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid blob entry magic: {:#010x}", magic),
            ));
        }

        let length = u32::from_le_bytes(header[4..8].try_into().unwrap());
        if length != blob_ref.length {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "blob length mismatch: expected {}, got {}",
                    blob_ref.length, length
                ),
            ));
        }

        // Read value
        let mut value = vec![0u8; length as usize];
        file.read_exact(&mut value)?;

        // Read and verify checksum
        let mut checksum_bytes = [0u8; 8];
        file.read_exact(&mut checksum_bytes)?;
        let stored_checksum = u64::from_le_bytes(checksum_bytes);

        // Recompute checksum over header + value
        let mut check_data = Vec::with_capacity(BLOB_ENTRY_HEADER_SIZE + value.len());
        check_data.extend_from_slice(&header);
        check_data.extend_from_slice(&value);
        let computed_checksum = xxh3_64(&check_data);

        if stored_checksum != computed_checksum {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "blob checksum mismatch: expected {:#018x}, got {:#018x}",
                    stored_checksum, computed_checksum
                ),
            ));
        }

        Ok(value)
    }

    /// Read a value transparently: if the value is a blob reference, fetch
    /// from the blob log; otherwise return the inline value as-is.
    pub fn read_transparent(&self, value: &[u8]) -> io::Result<Vec<u8>> {
        if let Some((_hash, blob_ref)) = decode_blob_ref(value) {
            self.read(&blob_ref)
        } else {
            Ok(value.to_vec())
        }
    }

    /// Mark a BlobRef as no longer live (e.g., after compaction removes the key).
    pub fn mark_dead(&self, blob_ref: &BlobRef) {
        let mut live = self.live_refs.write().expect("live_refs lock poisoned");
        if let Some(count) = live.get_mut(blob_ref) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                live.remove(blob_ref);
            }
        }
    }

    /// Collect statistics for all blob files for GC decision-making.
    pub fn collect_file_stats(&self) -> io::Result<Vec<BlobFileStats>> {
        let live_refs = self.live_refs.read().expect("live_refs lock poisoned");

        // Group live refs by file_id
        let mut live_by_file: HashMap<u64, (u64, u64)> = HashMap::new(); // file_id → (live_bytes, live_count)
        for (blob_ref, count) in live_refs.iter() {
            if *count > 0 {
                let entry = live_by_file.entry(blob_ref.file_id).or_insert((0, 0));
                let entry_size = BLOB_ENTRY_HEADER_SIZE as u64
                    + blob_ref.length as u64
                    + BLOB_ENTRY_FOOTER_SIZE as u64;
                entry.0 += entry_size;
                entry.1 += 1;
            }
        }

        // Scan blob files
        let mut stats = Vec::new();
        let entries = fs::read_dir(&self.blob_dir)?;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_none_or(|ext| ext != "dat") {
                continue;
            }

            let filename = path.file_stem().unwrap_or_default().to_string_lossy();
            let file_id: u64 = filename
                .strip_prefix("blob_")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);

            let total_bytes = entry.metadata()?.len();
            let (live_bytes, live_count) = live_by_file.get(&file_id).copied().unwrap_or((0, 0));

            // Estimate total count from file size (approximate)
            let avg_entry_size = if live_count > 0 {
                live_bytes / live_count
            } else {
                // Rough estimate
                BLOB_ENTRY_HEADER_SIZE as u64 + 2048 + BLOB_ENTRY_FOOTER_SIZE as u64
            };
            let total_count = if avg_entry_size > 0 {
                total_bytes / avg_entry_size
            } else {
                0
            };

            stats.push(BlobFileStats {
                live_bytes,
                total_bytes,
                live_count,
                total_count,
                path,
                file_id,
            });
        }

        Ok(stats)
    }

    /// Run garbage collection on blob files that exceed the discard ratio.
    ///
    /// For each file where discardable space > 50%, copies live values to
    /// a new file and deletes the old one. Returns the number of files
    /// compacted.
    pub fn run_gc(&self) -> io::Result<usize> {
        let file_stats = self.collect_file_stats()?;
        let mut compacted = 0;

        for stats in &file_stats {
            if !stats.needs_gc() {
                continue;
            }

            // Collect live blob refs for this file
            let live_refs_for_file: Vec<BlobRef> = {
                let live = self.live_refs.read().expect("live_refs lock poisoned");
                live.iter()
                    .filter(|(br, count)| br.file_id == stats.file_id && **count > 0)
                    .map(|(br, _)| *br)
                    .collect()
            };

            if live_refs_for_file.is_empty() {
                // No live values — just delete the file
                let _ = fs::remove_file(&stats.path);
                // Remove from index
                let mut index = self.index.write().expect("index lock poisoned");
                index.retain(|_, v| v.file_id != stats.file_id);
                compacted += 1;
                continue;
            }

            // Create a new blob file for the compacted data
            let new_file_id = self.next_file_id.fetch_add(1, Ordering::SeqCst);
            let mut new_writer = BlobWriter::new(&self.blob_dir, new_file_id)?;

            // Copy live values to the new file
            let mut new_refs: Vec<([u8; CONTENT_HASH_SIZE], BlobRef, BlobRef)> = Vec::new(); // (hash, old_ref, new_ref)

            for old_ref in &live_refs_for_file {
                // Read the value from the old file
                let value = match self.read(old_ref) {
                    Ok(v) => v,
                    Err(_) => continue, // Skip corrupt entries
                };

                let hash = content_hash(&value);
                let new_ref = new_writer.write_value(&value, &hash)?;
                new_refs.push((hash, *old_ref, new_ref));
            }

            // Update the index and live refs atomically
            {
                let mut index = self.index.write().expect("index lock poisoned");
                let mut live = self.live_refs.write().expect("live_refs lock poisoned");

                for (hash, old_ref, new_ref) in &new_refs {
                    // Update index
                    index.insert(*hash, *new_ref);

                    // Transfer live ref count
                    let count = live.remove(old_ref).unwrap_or(1);
                    live.insert(*new_ref, count);
                }
            }

            // Delete the old file
            let _ = fs::remove_file(&stats.path);
            compacted += 1;
        }

        Ok(compacted)
    }

    /// Returns the number of writer queues.
    pub fn num_queues(&self) -> usize {
        self.writers.len()
    }

    /// Returns the number of entries in the content-hash index.
    pub fn index_size(&self) -> usize {
        self.index.read().expect("index lock poisoned").len()
    }

    /// Returns the number of live blob references.
    pub fn live_ref_count(&self) -> usize {
        self.live_refs.read().expect("live_refs lock poisoned").len()
    }

    /// Returns the blob directory path.
    pub fn blob_dir(&self) -> &Path {
        &self.blob_dir
    }

    /// Look up a BlobRef by content hash.
    pub fn lookup_by_hash(&self, hash: &[u8; CONTENT_HASH_SIZE]) -> Option<BlobRef> {
        let index = self.index.read().expect("index lock poisoned");
        index.get(hash).copied()
    }
}
