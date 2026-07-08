//! SST file format with block index for fast point reads.
//!
//! An SST file contains multiple small PAX blocks (~64KB each) packed together,
//! with a block index at the end that maps block IDs to file offsets. This allows
//! point reads to load a single ~64KB block from NVMe (~18µs) instead of the
//! entire SST file (~2ms for 8MB).
//!
//! ## On-Disk Layout
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │ PAX Block 0 (serialized, ~64KB)         │
//! ├─────────────────────────────────────────┤
//! │ PAX Block 1 (serialized, ~64KB)         │
//! ├─────────────────────────────────────────┤
//! │ ...                                     │
//! ├─────────────────────────────────────────┤
//! │ PAX Block N (serialized, ~64KB)         │
//! ├─────────────────────────────────────────┤
//! │ Block Index                             │
//! │   block_count: u32                      │
//! │   [block_offset: u64, block_len: u32]   │
//! │   × block_count                         │
//! ├─────────────────────────────────────────┤
//! │ Footer                                  │
//! │   index_offset: u64                     │
//! │   block_count: u32                      │
//! │   magic: u32 = 0x53535446 ("SSTF")      │
//! └─────────────────────────────────────────┘
//! ```
//!
//! ## Point Read Path
//!
//! 1. ART lookup → `RowLocation::SST { sst_id, block_offset, row_offset }`
//! 2. `SstRegistry` has the block index in memory (loaded at registration)
//! 3. Look up `block_offset` in the index → `(file_offset, block_len)`
//! 4. `IoScheduler::read(file, file_offset, block_len, High)` — one NVMe read (~18µs for 64KB)
//! 5. `PaxBlock::deserialize(block_data)` → `read_column_row(1, row_offset)`

use galaxdb_common::{GalaxError, GalaxResult};

/// Legacy (pre-v0.5) SST footer magic — a footer with no explicit format
/// version. Files written before format versioning still open via this path
/// and are treated as format version 1.
pub const SST_FOOTER_MAGIC: u32 = 0x53535446; // "SSTF"

/// Versioned SST footer magic (v0.5+): the footer carries an explicit
/// `format_version` before the magic. A distinct magic lets a reader tell a
/// versioned footer from a legacy one unambiguously (the version field sits
/// where a legacy footer had none).
pub const SST_FOOTER_MAGIC_V2: u32 = 0x53535456; // "SSTV"

/// Legacy footer size: index_offset(8) + block_count(4) + magic(4) = 16 bytes.
pub const SST_FOOTER_SIZE: usize = 16;

/// Versioned footer size: index_offset(8) + block_count(4) + format_version(2)
/// + magic(4) = 18 bytes.
pub const SST_FOOTER_SIZE_V2: usize = 18;

/// Entry in the block index: where each PAX block lives in the SST file.
#[derive(Debug, Clone)]
pub struct BlockIndexEntry {
    /// Byte offset of this block within the SST file.
    pub file_offset: u64,
    /// Serialized length of this block in bytes.
    pub block_len: u32,
}

/// Block index for an SST file. Kept in memory for fast point reads.
#[derive(Debug, Clone)]
pub struct SstBlockIndex {
    /// One entry per PAX block in the SST file, in order.
    pub entries: Vec<BlockIndexEntry>,
}

impl Default for SstBlockIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl SstBlockIndex {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Add a block to the index.
    pub fn add_block(&mut self, file_offset: u64, block_len: u32) {
        self.entries.push(BlockIndexEntry { file_offset, block_len });
    }

    /// Look up a block by its index (block_offset from ART).
    pub fn get_block(&self, block_idx: u64) -> Option<&BlockIndexEntry> {
        self.entries.get(block_idx as usize)
    }

    /// Number of blocks in this SST.
    pub fn block_count(&self) -> usize {
        self.entries.len()
    }

    /// Serialize the block index + footer to bytes.
    pub fn serialize_with_footer(&self, index_offset: u64) -> Vec<u8> {
        let block_count = self.entries.len() as u32;
        // Index: block_count(4) + entries × (file_offset(8) + block_len(4))
        let index_size = 4 + self.entries.len() * 12;
        let mut buf = Vec::with_capacity(index_size + SST_FOOTER_SIZE);

        // Block index
        buf.extend_from_slice(&block_count.to_le_bytes());
        for entry in &self.entries {
            buf.extend_from_slice(&entry.file_offset.to_le_bytes());
            buf.extend_from_slice(&entry.block_len.to_le_bytes());
        }

        // Versioned footer (v0.5+): index_offset | block_count | format_version | magic.
        buf.extend_from_slice(&index_offset.to_le_bytes());
        buf.extend_from_slice(&block_count.to_le_bytes());
        buf.extend_from_slice(&galaxdb_common::format::SST.current_write.to_le_bytes());
        buf.extend_from_slice(&SST_FOOTER_MAGIC_V2.to_le_bytes());

        buf
    }

    /// Parse the trailing footer, accepting both the versioned (v0.5+) and the
    /// legacy layouts. Returns `(index_offset, block_count, footer_size)`.
    ///
    /// A versioned footer's `format_version` is range-checked against the
    /// engine's supported range (typed `FormatTooOld` / `FormatTooNew`, the
    /// rollback-safety refusal). A legacy footer is treated as format v1.
    fn parse_footer(data: &[u8]) -> GalaxResult<(u64, u32, usize)> {
        if data.len() < SST_FOOTER_SIZE {
            return Err(GalaxError::Internal("SST file too small for footer".into()));
        }
        let end = data.len();
        let trailing_magic = u32::from_le_bytes(
            data[end - 4..end]
                .try_into()
                .map_err(|_| GalaxError::Internal("bad footer magic".into()))?,
        );
        match trailing_magic {
            SST_FOOTER_MAGIC_V2 => {
                if data.len() < SST_FOOTER_SIZE_V2 {
                    return Err(GalaxError::Internal("SST file too small for versioned footer".into()));
                }
                let fs = end - SST_FOOTER_SIZE_V2;
                let index_offset = u64::from_le_bytes(data[fs..fs + 8].try_into().unwrap());
                let block_count = u32::from_le_bytes(data[fs + 8..fs + 12].try_into().unwrap());
                let format_version = u16::from_le_bytes(data[fs + 12..fs + 14].try_into().unwrap());
                galaxdb_common::format::SST.check(format_version)?;
                Ok((index_offset, block_count, SST_FOOTER_SIZE_V2))
            }
            SST_FOOTER_MAGIC => {
                // Legacy footer: no explicit version → implicitly format v1.
                let fs = end - SST_FOOTER_SIZE;
                let index_offset = u64::from_le_bytes(data[fs..fs + 8].try_into().unwrap());
                let block_count = u32::from_le_bytes(data[fs + 8..fs + 12].try_into().unwrap());
                Ok((index_offset, block_count, SST_FOOTER_SIZE))
            }
            other => Err(GalaxError::Internal(format!(
                "SST footer magic mismatch: expected {:#x} or {:#x}, got {:#x}",
                SST_FOOTER_MAGIC_V2, SST_FOOTER_MAGIC, other
            ))),
        }
    }

    /// Deserialize a block index from SST file data.
    /// Reads the footer first to find the index offset, then reads the index.
    pub fn from_file_data(data: &[u8]) -> GalaxResult<Self> {
        // Accepts both the versioned (v0.5+) and legacy footer layouts.
        let (index_offset, block_count, footer_size) = Self::parse_footer(data)?;
        let index_offset = index_offset as usize;
        let block_count = block_count as usize;
        let footer_start = data.len() - footer_size;

        // Read block index
        let mut offset = index_offset + 4; // skip block_count (already read from footer)
        let mut entries = Vec::with_capacity(block_count);

        for _ in 0..block_count {
            if offset + 12 > footer_start {
                return Err(GalaxError::Internal("block index extends into footer".into()));
            }
            let file_offset = u64::from_le_bytes(
                data[offset..offset + 8].try_into()
                    .map_err(|_| GalaxError::Internal("bad block index entry".into()))?
            );
            let block_len = u32::from_le_bytes(
                data[offset + 8..offset + 12].try_into()
                    .map_err(|_| GalaxError::Internal("bad block index entry".into()))?
            );
            entries.push(BlockIndexEntry { file_offset, block_len });
            offset += 12;
        }

        Ok(Self { entries })
    }

    /// Read just the footer from a file to get block count and index offset.
    /// This reads only 16 bytes from the end of the file.
    pub fn read_footer(data: &[u8]) -> GalaxResult<(u64, u32)> {
        let (index_offset, block_count, _footer_size) = Self::parse_footer(data)?;
        Ok((index_offset, block_count))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_index_roundtrip() {
        let mut index = SstBlockIndex::new();
        index.add_block(0, 1024);
        index.add_block(1024, 2048);
        index.add_block(3072, 512);

        let index_offset = 3584u64; // after all blocks
        let serialized = index.serialize_with_footer(index_offset);

        // Simulate a full SST file: blocks + index + footer
        let mut file_data = vec![0u8; index_offset as usize]; // fake block data
        file_data.extend_from_slice(&serialized);

        let recovered = SstBlockIndex::from_file_data(&file_data).unwrap();
        assert_eq!(recovered.block_count(), 3);
        assert_eq!(recovered.entries[0].file_offset, 0);
        assert_eq!(recovered.entries[0].block_len, 1024);
        assert_eq!(recovered.entries[1].file_offset, 1024);
        assert_eq!(recovered.entries[1].block_len, 2048);
        assert_eq!(recovered.entries[2].file_offset, 3072);
        assert_eq!(recovered.entries[2].block_len, 512);
    }

    /// Build a legacy (pre-v0.5) 16-byte SSTF footer by hand and confirm a
    /// current reader still opens it (backward-compat read, Req 5.1).
    #[test]
    fn legacy_footer_still_reads() {
        let mut index = SstBlockIndex::new();
        index.add_block(0, 1024);
        index.add_block(1024, 2048);
        let index_offset = 3072u64;

        // Hand-serialize the index + a *legacy* footer (no format_version).
        let mut file_data = vec![0u8; index_offset as usize];
        file_data.extend_from_slice(&(index.entries.len() as u32).to_le_bytes());
        for e in &index.entries {
            file_data.extend_from_slice(&e.file_offset.to_le_bytes());
            file_data.extend_from_slice(&e.block_len.to_le_bytes());
        }
        file_data.extend_from_slice(&index_offset.to_le_bytes());
        file_data.extend_from_slice(&(index.entries.len() as u32).to_le_bytes());
        file_data.extend_from_slice(&SST_FOOTER_MAGIC.to_le_bytes()); // legacy SSTF

        let recovered = SstBlockIndex::from_file_data(&file_data).unwrap();
        assert_eq!(recovered.block_count(), 2);
        assert_eq!(recovered.entries[1].block_len, 2048);
    }

    /// A versioned footer whose format_version exceeds what this engine writes
    /// must be refused with a typed FormatTooNew (rollback safety, Req 5.2).
    #[test]
    fn future_footer_version_refused() {
        let mut index = SstBlockIndex::new();
        index.add_block(0, 512);
        let index_offset = 512u64;

        let mut file_data = vec![0u8; index_offset as usize];
        file_data.extend_from_slice(&index.serialize_with_footer(index_offset));

        // Bump the format_version (the 2 bytes just before the 4-byte magic).
        let end = file_data.len();
        let future = galaxdb_common::format::SST.current_write + 1;
        file_data[end - 6..end - 4].copy_from_slice(&future.to_le_bytes());

        match SstBlockIndex::from_file_data(&file_data).err() {
            Some(GalaxError::FormatTooNew {
                artifact,
                found,
                current,
            }) => {
                assert_eq!(artifact, "SST");
                assert_eq!(found, future);
                assert_eq!(current, galaxdb_common::format::SST.current_write);
            }
            other => panic!("expected FormatTooNew, got {other:?}"),
        }
    }

    #[test]
    fn get_block_by_index() {
        let mut index = SstBlockIndex::new();
        index.add_block(0, 100);
        index.add_block(100, 200);

        assert!(index.get_block(0).is_some());
        assert_eq!(index.get_block(0).unwrap().block_len, 100);
        assert_eq!(index.get_block(1).unwrap().file_offset, 100);
        assert!(index.get_block(2).is_none());
    }

    #[test]
    fn bad_magic_rejected() {
        let data = vec![0u8; 20]; // all zeros, wrong magic
        assert!(SstBlockIndex::from_file_data(&data).is_err());
    }

    #[test]
    fn too_small_rejected() {
        let data = vec![0u8; 10];
        assert!(SstBlockIndex::from_file_data(&data).is_err());
    }
}
