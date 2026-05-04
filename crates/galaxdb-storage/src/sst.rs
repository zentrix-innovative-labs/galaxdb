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

/// SST file footer magic number.
pub const SST_FOOTER_MAGIC: u32 = 0x53535446; // "SSTF"

/// Footer size: index_offset(8) + block_count(4) + magic(4) = 16 bytes.
pub const SST_FOOTER_SIZE: usize = 16;

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

        // Footer
        buf.extend_from_slice(&index_offset.to_le_bytes());
        buf.extend_from_slice(&block_count.to_le_bytes());
        buf.extend_from_slice(&SST_FOOTER_MAGIC.to_le_bytes());

        buf
    }

    /// Deserialize a block index from SST file data.
    /// Reads the footer first to find the index offset, then reads the index.
    pub fn from_file_data(data: &[u8]) -> GalaxResult<Self> {
        if data.len() < SST_FOOTER_SIZE {
            return Err(GalaxError::Internal("SST file too small for footer".into()));
        }

        // Read footer (last 16 bytes)
        let footer_start = data.len() - SST_FOOTER_SIZE;
        let index_offset = u64::from_le_bytes(
            data[footer_start..footer_start + 8].try_into()
                .map_err(|_| GalaxError::Internal("bad footer index_offset".into()))?
        ) as usize;
        let block_count = u32::from_le_bytes(
            data[footer_start + 8..footer_start + 12].try_into()
                .map_err(|_| GalaxError::Internal("bad footer block_count".into()))?
        ) as usize;
        let magic = u32::from_le_bytes(
            data[footer_start + 12..footer_start + 16].try_into()
                .map_err(|_| GalaxError::Internal("bad footer magic".into()))?
        );

        if magic != SST_FOOTER_MAGIC {
            return Err(GalaxError::Internal(format!(
                "SST footer magic mismatch: expected {:#x}, got {:#x}",
                SST_FOOTER_MAGIC, magic
            )));
        }

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
        if data.len() < SST_FOOTER_SIZE {
            return Err(GalaxError::Internal("data too small for SST footer".into()));
        }
        let footer_start = data.len() - SST_FOOTER_SIZE;
        let index_offset = u64::from_le_bytes(
            data[footer_start..footer_start + 8].try_into()
                .map_err(|_| GalaxError::Internal("bad footer".into()))?
        );
        let block_count = u32::from_le_bytes(
            data[footer_start + 8..footer_start + 12].try_into()
                .map_err(|_| GalaxError::Internal("bad footer".into()))?
        );
        let magic = u32::from_le_bytes(
            data[footer_start + 12..footer_start + 16].try_into()
                .map_err(|_| GalaxError::Internal("bad footer".into()))?
        );
        if magic != SST_FOOTER_MAGIC {
            return Err(GalaxError::Internal("not an SST file (bad magic)".into()));
        }
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
