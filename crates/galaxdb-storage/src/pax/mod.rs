//! PAX Block Format for GalaxDB.
//!
//! Implements column-oriented storage blocks with per-column compression,
//! zone maps (min/max per column), and XXH3-64 integrity checksums.
//!
//! # Block Layout
//!
//! ```text
//! ┌─────────────────────────────────────────────┐
//! │ PAX Block Header (fixed-size prefix)        │
//! │  magic: u32 = 0x47414C41                    │
//! │  format_version: u8                         │
//! │  block_id: u64                              │
//! │  row_count: u32                             │
//! │  commit_timestamp: u64                      │
//! │  column_count: u16                          │
//! │  column_descriptors: [ColumnDesc; N]        │
//! ├─────────────────────────────────────────────┤
//! │ Column Chunk 0 (fixed-width: FastPFOR)      │
//! │ Column Chunk 1 (variable-width: Zstd L3)    │
//! │ Column Chunk 2 (embedding: raw)             │
//! │ ...                                         │
//! ├─────────────────────────────────────────────┤
//! │ Row Offset Table                            │
//! │  [u32; row_count] byte offsets              │
//! ├─────────────────────────────────────────────┤
//! │ Block Footer                                │
//! │  checksum: u64 (XXH3-64 over entire block)  │
//! └─────────────────────────────────────────────┘
//! ```
//!
//! # Compression Strategy
//!
//! - Fixed-width integers: delta encoding + bit-packing (simplified FastPFOR)
//! - Variable-width (TEXT, BLOB, JSON): Zstandard level 3
//! - Embedding columns: no additional compression (quantization handles it)
//! - Codec IDs: `0=none, 1=fastpfor, 2=zstd`

mod codec;

#[cfg(test)]
mod tests;

use std::io::{Cursor, Read, Write};

use galaxdb_common::{BlockId, ColumnType, GalaxError, GalaxResult, Timestamp};
use xxhash_rust::xxh3::xxh3_64;

/// Magic number identifying a PAX block: ASCII "GALA" = 0x47414C41.
pub const PAX_MAGIC: u32 = 0x47414C41;

/// Current format version for PAX blocks.
pub const PAX_FORMAT_VERSION: u8 = 1;

/// Codec identifiers stored in column descriptors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CodecId {
    /// No compression.
    None = 0,
    /// Delta encoding + bit-packing (simplified FastPFOR).
    FastPFor = 1,
    /// Zstandard level 3.
    Zstd = 2,
}

impl CodecId {
    /// Convert a raw byte to a `CodecId`.
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::None),
            1 => Some(Self::FastPFor),
            2 => Some(Self::Zstd),
            _ => None,
        }
    }

    /// Select the appropriate codec for a given column type.
    pub fn for_column_type(col_type: &ColumnType) -> Self {
        if col_type.is_fixed_width() {
            CodecId::FastPFor
        } else if col_type.is_variable_width() {
            CodecId::Zstd
        } else {
            // Embedding columns: no additional compression
            CodecId::None
        }
    }
}

/// Zone map storing min/max values for a column as raw bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZoneMap {
    /// Minimum value in the column (raw bytes, comparable).
    pub min: Vec<u8>,
    /// Maximum value in the column (raw bytes, comparable).
    pub max: Vec<u8>,
}

/// Descriptor for a single column within a PAX block header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnDescriptor {
    /// The column's data type.
    pub col_type: ColumnType,
    /// Compression codec used for this column's chunk.
    pub codec: CodecId,
    /// Byte offset of this column's chunk data within the block (from start of column data area).
    pub offset: u64,
    /// Compressed length of this column's chunk in bytes.
    pub compressed_len: u32,
    /// Zone map: min value.
    pub zone_map_min: Vec<u8>,
    /// Zone map: max value.
    pub zone_map_max: Vec<u8>,
}

/// PAX block header containing metadata for the block.
#[derive(Debug, Clone)]
pub struct PaxBlockHeader {
    /// Magic number (must be `PAX_MAGIC`).
    pub magic: u32,
    /// Format version.
    pub format_version: u8,
    /// Unique block identifier.
    pub block_id: BlockId,
    /// Number of rows in this block.
    pub row_count: u32,
    /// Commit timestamp for MVCC.
    pub commit_timestamp: Timestamp,
    /// Number of columns.
    pub column_count: u16,
    /// Per-column descriptors.
    pub column_descriptors: Vec<ColumnDescriptor>,
}

/// A complete PAX block with header, column data, row offsets, and checksum.
#[derive(Debug, Clone)]
pub struct PaxBlock {
    /// Block header with metadata and column descriptors.
    pub header: PaxBlockHeader,
    /// Compressed column chunk data (concatenated).
    pub column_data: Vec<u8>,
    /// Row offset table: byte offsets for reconstructing rows.
    pub row_offsets: Vec<u32>,
    /// XXH3-64 checksum over the entire block (excluding the checksum itself).
    pub checksum: u64,
}

/// Input column data for writing a PAX block.
#[derive(Debug, Clone)]
pub struct ColumnData {
    /// The column's data type.
    pub col_type: ColumnType,
    /// Raw values for each row. For fixed-width types, each entry is the
    /// fixed-size byte representation. For variable-width types, each entry
    /// is the raw bytes of the value.
    pub values: Vec<Vec<u8>>,
}

impl PaxBlock {
    /// Write a new PAX block from column data.
    ///
    /// This compresses each column according to its type, computes zone maps,
    /// builds the row offset table, and computes the XXH3-64 checksum.
    pub fn write(
        block_id: BlockId,
        commit_timestamp: Timestamp,
        columns: &[ColumnData],
    ) -> GalaxResult<Self> {
        // Use default codec selection per column type
        let codecs: Vec<CodecId> = columns.iter()
            .map(|col| CodecId::for_column_type(&col.col_type))
            .collect();
        Self::write_with_codecs(block_id, commit_timestamp, columns, &codecs)
    }

    /// Write a PAX block with explicit codec selection per column.
    ///
    /// This allows the caller to override the default codec for specific columns.
    /// For example, the KV storage flush path uses `CodecId::None` for the value
    /// column to enable fast single-row point reads without decompressing the
    /// entire column.
    pub fn write_with_codecs(
        block_id: BlockId,
        commit_timestamp: Timestamp,
        columns: &[ColumnData],
        codecs: &[CodecId],
    ) -> GalaxResult<Self> {
        if columns.is_empty() {
            return Err(GalaxError::Internal("PAX block must have at least one column".into()));
        }
        if codecs.len() != columns.len() {
            return Err(GalaxError::Internal(format!(
                "codec count ({}) must match column count ({})",
                codecs.len(), columns.len()
            )));
        }

        let row_count = columns[0].values.len();
        for col in columns {
            if col.values.len() != row_count {
                return Err(GalaxError::Internal(
                    "all columns must have the same number of rows".into(),
                ));
            }
        }

        let column_count = columns.len() as u16;
        let mut column_descriptors = Vec::with_capacity(columns.len());
        let mut column_data = Vec::new();

        for (col_idx, col) in columns.iter().enumerate() {
            let codec = codecs[col_idx];
            let zone_map = extract_zone_map(&col.col_type, &col.values);
            let offset = column_data.len() as u64;

            let compressed = codec::compress(&col.col_type, codec, &col.values)?;
            let compressed_len = compressed.len() as u32;
            column_data.extend_from_slice(&compressed);

            column_descriptors.push(ColumnDescriptor {
                col_type: col.col_type.clone(),
                codec,
                offset,
                compressed_len,
                zone_map_min: zone_map.min,
                zone_map_max: zone_map.max,
            });
        }

        // Build row offset table: cumulative byte offsets per row across all columns.
        // Each entry is the starting byte offset of that row's data in the
        // uncompressed column layout.
        let row_offsets = build_row_offsets(columns, row_count);

        let header = PaxBlockHeader {
            magic: PAX_MAGIC,
            format_version: PAX_FORMAT_VERSION,
            block_id,
            row_count: row_count as u32,
            commit_timestamp,
            column_count,
            column_descriptors,
        };

        // Serialize everything except the checksum, then compute checksum
        let mut block_bytes = Vec::new();
        serialize_header(&header, &mut block_bytes)?;
        block_bytes.extend_from_slice(&column_data);
        serialize_row_offsets(&row_offsets, &mut block_bytes)?;

        let checksum = xxh3_64(&block_bytes);

        Ok(PaxBlock {
            header,
            column_data,
            row_offsets,
            checksum,
        })
    }

    /// Serialize the entire PAX block to bytes.
    pub fn serialize(&self) -> GalaxResult<Vec<u8>> {
        let mut buf = Vec::new();
        serialize_header(&self.header, &mut buf)?;
        buf.extend_from_slice(&self.column_data);
        serialize_row_offsets(&self.row_offsets, &mut buf)?;
        // Footer: checksum
        buf.extend_from_slice(&self.checksum.to_le_bytes());
        Ok(buf)
    }

    /// Deserialize a PAX block from bytes, verifying magic number and checksum.
    ///
    /// Returns `Err(GalaxError::InvalidMagic)` if the magic number doesn't match,
    /// and `Err(GalaxError::ChecksumMismatch)` if the checksum doesn't verify.
    pub fn deserialize(data: &[u8]) -> GalaxResult<Self> {
        if data.len() < 4 {
            return Err(GalaxError::Internal("PAX block too small".into()));
        }

        // The last 8 bytes are the checksum
        if data.len() < 8 {
            return Err(GalaxError::Internal("PAX block too small for checksum".into()));
        }

        let checksum_offset = data.len() - 8;
        let stored_checksum = u64::from_le_bytes(
            data[checksum_offset..].try_into().map_err(|_| {
                GalaxError::Internal("failed to read checksum bytes".into())
            })?,
        );

        // Verify checksum over everything except the checksum itself
        let block_data = &data[..checksum_offset];
        let computed_checksum = xxh3_64(block_data);

        if computed_checksum != stored_checksum {
            return Err(GalaxError::ChecksumMismatch {
                expected: stored_checksum,
                actual: computed_checksum,
            });
        }

        // Verify magic number
        let mut cursor = Cursor::new(block_data);
        let header = deserialize_header(&mut cursor)?;

        if header.magic != PAX_MAGIC {
            return Err(GalaxError::InvalidMagic(header.magic));
        }

        // Read column data
        let column_data_start = cursor.position() as usize;
        let total_column_data_len: u64 = header
            .column_descriptors
            .iter()
            .map(|d| d.compressed_len as u64)
            .sum();
        let column_data_end = column_data_start + total_column_data_len as usize;

        if column_data_end > block_data.len() {
            return Err(GalaxError::Internal("column data extends beyond block".into()));
        }

        let column_data = block_data[column_data_start..column_data_end].to_vec();

        // Read row offset table
        let row_offset_start = column_data_end;
        let row_offset_bytes = header.row_count as usize * 4;
        let row_offset_end = row_offset_start + row_offset_bytes;

        if row_offset_end > block_data.len() {
            return Err(GalaxError::Internal("row offset table extends beyond block".into()));
        }

        let mut row_offsets = Vec::with_capacity(header.row_count as usize);
        for i in 0..header.row_count as usize {
            let start = row_offset_start + i * 4;
            let offset = u32::from_le_bytes(
                block_data[start..start + 4].try_into().map_err(|_| {
                    GalaxError::Internal("failed to read row offset".into())
                })?,
            );
            row_offsets.push(offset);
        }

        Ok(PaxBlock {
            header,
            column_data,
            row_offsets,
            checksum: stored_checksum,
        })
    }

    /// Decompress a specific column's data from this block.
    ///
    /// Returns the raw values for each row in the column.
    pub fn read_column(&self, col_index: usize) -> GalaxResult<Vec<Vec<u8>>> {
        if col_index >= self.header.column_descriptors.len() {
            return Err(GalaxError::Internal(format!(
                "column index {} out of range (block has {} columns)",
                col_index,
                self.header.column_descriptors.len()
            )));
        }

        let desc = &self.header.column_descriptors[col_index];
        let start = desc.offset as usize;
        let end = start + desc.compressed_len as usize;

        if end > self.column_data.len() {
            return Err(GalaxError::Internal("column chunk extends beyond column data".into()));
        }

        let compressed = &self.column_data[start..end];
        codec::decompress(&desc.col_type, desc.codec, compressed, self.header.row_count)
    }

    /// Read a single row from a specific column.
    ///
    /// For uncompressed columns (CodecId::None), this scans length prefixes
    /// to the target row without decompressing the entire column — much faster
    /// for point lookups.
    ///
    /// For compressed columns (Zstd, FastPFor), falls back to decompressing
    /// the entire column and extracting the target row.
    pub fn read_column_row(&self, col_index: usize, row_offset: u32) -> GalaxResult<Vec<u8>> {
        if col_index >= self.header.column_descriptors.len() {
            return Err(GalaxError::Internal(format!(
                "column index {} out of range (block has {} columns)",
                col_index,
                self.header.column_descriptors.len()
            )));
        }

        if row_offset >= self.header.row_count {
            return Err(GalaxError::Internal(format!(
                "row offset {} out of range (block has {} rows)",
                row_offset, self.header.row_count
            )));
        }

        let desc = &self.header.column_descriptors[col_index];
        let start = desc.offset as usize;
        let end = start + desc.compressed_len as usize;

        if end > self.column_data.len() {
            return Err(GalaxError::Internal("column chunk extends beyond column data".into()));
        }

        let column_bytes = &self.column_data[start..end];

        match desc.codec {
            CodecId::None => {
                // Fast path: scan length prefixes to target row
                codec::decompress_none_single_row(column_bytes, row_offset)
            }
            _ => {
                // Slow path: decompress entire column, extract target row
                let values = codec::decompress(&desc.col_type, desc.codec, column_bytes, self.header.row_count)?;
                values.into_iter().nth(row_offset as usize)
                    .ok_or_else(|| GalaxError::Internal("row offset out of range after decompression".into()))
            }
        }
    }
}

/// Zero-copy extraction of a single value from raw PAX block bytes.
///
/// This is the fast path for point reads. Instead of deserializing the entire
/// PAX block into a `PaxBlock` struct (which allocates Vecs for column data,
/// row offsets, zone maps, etc.), this function:
///
/// 1. Verifies the XXH3-64 checksum (~1.8µs for 62KB)
/// 2. Parses just enough of the header to find the target column's offset and codec
/// 3. Slices directly into the raw bytes (zero-copy) to get the column data
/// 4. Scans length prefixes to the target row (for CodecId::None)
/// 5. Copies only the target row's value bytes
///
/// This eliminates the ~62KB memcpy and multiple heap allocations that
/// `PaxBlock::deserialize` + `read_column_row` would do.
///
/// Target: < 10µs for a single row extraction from a 62KB block.
pub fn read_value_from_raw_block(
    block_data: &[u8],
    col_index: usize,
    row_offset: u32,
) -> GalaxResult<Vec<u8>> {
    if block_data.len() < 8 {
        return Err(GalaxError::Internal("block too small".into()));
    }

    // Step 1: Verify checksum
    let checksum_offset = block_data.len() - 8;
    let stored_checksum = u64::from_le_bytes(
        block_data[checksum_offset..].try_into()
            .map_err(|_| GalaxError::Internal("bad checksum bytes".into()))?
    );
    let computed_checksum = xxh3_64(&block_data[..checksum_offset]);
    if computed_checksum != stored_checksum {
        return Err(GalaxError::ChecksumMismatch {
            expected: stored_checksum,
            actual: computed_checksum,
        });
    }

    let data = &block_data[..checksum_offset];

    // Step 2: Parse minimal header — just enough to find the target column.
    // Fixed header: magic(4) + version(1) + block_id(8) + row_count(4) + timestamp(8) + col_count(2) = 27 bytes
    if data.len() < 27 {
        return Err(GalaxError::Internal("block too small for header".into()));
    }

    // magic(4) + version(1) + block_id(8) = 13 bytes before row_count
    let row_count = u32::from_le_bytes(data[13..17].try_into().unwrap());
    // row_count(4) + timestamp(8) = 12 more bytes, then col_count at offset 25
    let column_count = u16::from_le_bytes(data[25..27].try_into().unwrap()) as usize;

    if col_index >= column_count {
        return Err(GalaxError::Internal("column index out of range".into()));
    }
    if row_offset >= row_count {
        return Err(GalaxError::Internal("row offset out of range".into()));
    }

    // Step 3: Skip through column descriptors to find the target column.
    // Each descriptor: col_type(1) [+ embedding_dims(4)] + codec(1) + offset(8) + compressed_len(4) + zone_min_len(4) + zone_min + zone_max_len(4) + zone_max
    let mut pos = 27; // after fixed header
    let mut target_offset: u64 = 0;
    let mut target_len: u32 = 0;
    let mut target_codec: u8 = 0;
    let mut header_end_pos: usize = 0;

    for col in 0..column_count {
        if pos >= data.len() {
            return Err(GalaxError::Internal("header truncated".into()));
        }

        let col_type_byte = data[pos];
        pos += 1;

        // Embedding type has extra 4 bytes for dimensions
        if col_type_byte == 14 {
            pos += 4;
        }

        // codec: u8
        let codec = data[pos];
        pos += 1;

        // offset: u64
        let offset = u64::from_le_bytes(
            data[pos..pos + 8].try_into()
                .map_err(|_| GalaxError::Internal("bad column offset".into()))?
        );
        pos += 8;

        // compressed_len: u32
        let compressed_len = u32::from_le_bytes(
            data[pos..pos + 4].try_into()
                .map_err(|_| GalaxError::Internal("bad compressed_len".into()))?
        );
        pos += 4;

        // zone_map_min: skip
        let min_len = u32::from_le_bytes(
            data[pos..pos + 4].try_into()
                .map_err(|_| GalaxError::Internal("bad zone_min_len".into()))?
        ) as usize;
        pos += 4 + min_len;

        // zone_map_max: skip
        let max_len = u32::from_le_bytes(
            data[pos..pos + 4].try_into()
                .map_err(|_| GalaxError::Internal("bad zone_max_len".into()))?
        ) as usize;
        pos += 4 + max_len;

        if col == col_index {
            target_offset = offset;
            target_len = compressed_len;
            target_codec = codec;
        }

        // After the last column descriptor, pos is the start of column data
        if col == column_count - 1 {
            header_end_pos = pos;
        }
    }

    // Step 4: Slice directly into the column data (zero-copy)
    let col_data_start = header_end_pos + target_offset as usize;
    let col_data_end = col_data_start + target_len as usize;

    if col_data_end > data.len() {
        return Err(GalaxError::Internal("column data extends beyond block".into()));
    }

    let column_bytes = &data[col_data_start..col_data_end];

    // Step 5: Extract the target row
    match CodecId::from_u8(target_codec) {
        Some(CodecId::None) => {
            // Fast path: scan length prefixes to target row (zero-copy until final copy)
            codec::decompress_none_single_row(column_bytes, row_offset)
        }
        Some(codec_id) => {
            // Slow path: decompress entire column
            let col_type = ColumnType::Blob; // KV storage always uses Blob
            let values = codec::decompress(&col_type, codec_id, column_bytes, row_count)?;
            values.into_iter().nth(row_offset as usize)
                .ok_or_else(|| GalaxError::Internal("row not found after decompression".into()))
        }
        None => Err(GalaxError::Internal(format!("unknown codec: {}", target_codec))),
    }
}

/// Extract zone map (min/max) for a column from its raw values.
fn extract_zone_map(col_type: &ColumnType, values: &[Vec<u8>]) -> ZoneMap {
    if values.is_empty() {
        return ZoneMap {
            min: Vec::new(),
            max: Vec::new(),
        };
    }

    // For all types, compare raw bytes lexicographically.
    // For fixed-width numeric types stored in little-endian, we need to
    // compare the actual numeric values. For simplicity and correctness,
    // we compare the raw byte representations which works for unsigned LE
    // integers and for variable-width types (lexicographic).
    //
    // For signed integers, we convert to a comparable byte form.
    let comparable_values: Vec<Vec<u8>> = values
        .iter()
        .map(|v| to_comparable_bytes(col_type, v))
        .collect();

    let min_idx = comparable_values
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let max_idx = comparable_values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0);

    ZoneMap {
        min: values[min_idx].clone(),
        max: values[max_idx].clone(),
    }
}

/// Convert a value to a byte representation that sorts correctly via
/// lexicographic comparison. For unsigned integers and variable-width types,
/// this is a no-op. For signed integers, we flip the sign bit.
fn to_comparable_bytes(col_type: &ColumnType, value: &[u8]) -> Vec<u8> {
    match col_type {
        // Signed integers: flip the sign bit for correct lexicographic ordering
        ColumnType::Int8 if value.len() == 1 => {
            vec![value[0] ^ 0x80]
        }
        ColumnType::Int16 if value.len() == 2 => {
            let val = i16::from_le_bytes([value[0], value[1]]);
            let unsigned = (val as u16) ^ 0x8000;
            unsigned.to_be_bytes().to_vec()
        }
        ColumnType::Int32 if value.len() == 4 => {
            let val = i32::from_le_bytes(value.try_into().unwrap());
            let unsigned = (val as u32) ^ 0x8000_0000;
            unsigned.to_be_bytes().to_vec()
        }
        ColumnType::Int64 if value.len() == 8 => {
            let val = i64::from_le_bytes(value.try_into().unwrap());
            let unsigned = (val as u64) ^ 0x8000_0000_0000_0000;
            unsigned.to_be_bytes().to_vec()
        }
        // Unsigned integers: convert to big-endian for correct lexicographic ordering
        ColumnType::UInt16 if value.len() == 2 => {
            let val = u16::from_le_bytes(value.try_into().unwrap());
            val.to_be_bytes().to_vec()
        }
        ColumnType::UInt32 if value.len() == 4 => {
            let val = u32::from_le_bytes(value.try_into().unwrap());
            val.to_be_bytes().to_vec()
        }
        ColumnType::UInt64 if value.len() == 8 => {
            let val = u64::from_le_bytes(value.try_into().unwrap());
            val.to_be_bytes().to_vec()
        }
        // Float types: IEEE 754 comparable encoding
        ColumnType::Float32 if value.len() == 4 => {
            let bits = u32::from_le_bytes(value.try_into().unwrap());
            let comparable = if bits & 0x8000_0000 != 0 {
                !bits // negative: flip all bits
            } else {
                bits ^ 0x8000_0000 // positive: flip sign bit
            };
            comparable.to_be_bytes().to_vec()
        }
        ColumnType::Float64 if value.len() == 8 => {
            let bits = u64::from_le_bytes(value.try_into().unwrap());
            let comparable = if bits & 0x8000_0000_0000_0000 != 0 {
                !bits
            } else {
                bits ^ 0x8000_0000_0000_0000
            };
            comparable.to_be_bytes().to_vec()
        }
        // Everything else: use raw bytes directly (UInt8, Boolean, Text, Blob, Json, Embedding)
        _ => value.to_vec(),
    }
}

/// Build the row offset table. Each entry is the cumulative byte size of
/// all column values for rows up to (but not including) that row index.
fn build_row_offsets(columns: &[ColumnData], row_count: usize) -> Vec<u32> {
    let mut offsets = Vec::with_capacity(row_count);
    let mut cumulative: u32 = 0;

    for row_idx in 0..row_count {
        offsets.push(cumulative);
        let row_size: u32 = columns
            .iter()
            .map(|col| col.values[row_idx].len() as u32)
            .sum();
        cumulative = cumulative.saturating_add(row_size);
    }

    offsets
}

// --- Serialization helpers ---

/// Serialize a column type to a single byte.
fn column_type_to_u8(col_type: &ColumnType) -> u8 {
    match col_type {
        ColumnType::Int8 => 0,
        ColumnType::Int16 => 1,
        ColumnType::Int32 => 2,
        ColumnType::Int64 => 3,
        ColumnType::UInt8 => 4,
        ColumnType::UInt16 => 5,
        ColumnType::UInt32 => 6,
        ColumnType::UInt64 => 7,
        ColumnType::Float32 => 8,
        ColumnType::Float64 => 9,
        ColumnType::Text => 10,
        ColumnType::Blob => 11,
        ColumnType::Json => 12,
        ColumnType::Boolean => 13,
        ColumnType::Embedding(_) => 14,
    }
}

/// Deserialize a column type from a byte and optional dimension.
fn column_type_from_u8(value: u8, embedding_dims: u32) -> Option<ColumnType> {
    match value {
        0 => Some(ColumnType::Int8),
        1 => Some(ColumnType::Int16),
        2 => Some(ColumnType::Int32),
        3 => Some(ColumnType::Int64),
        4 => Some(ColumnType::UInt8),
        5 => Some(ColumnType::UInt16),
        6 => Some(ColumnType::UInt32),
        7 => Some(ColumnType::UInt64),
        8 => Some(ColumnType::Float32),
        9 => Some(ColumnType::Float64),
        10 => Some(ColumnType::Text),
        11 => Some(ColumnType::Blob),
        12 => Some(ColumnType::Json),
        13 => Some(ColumnType::Boolean),
        14 => Some(ColumnType::Embedding(embedding_dims)),
        _ => None,
    }
}

/// Serialize the PAX block header to a writer.
fn serialize_header(header: &PaxBlockHeader, buf: &mut Vec<u8>) -> GalaxResult<()> {
    buf.write_all(&header.magic.to_le_bytes())
        .map_err(GalaxError::Io)?;
    buf.write_all(&[header.format_version])
        .map_err(GalaxError::Io)?;
    buf.write_all(&header.block_id.to_le_bytes())
        .map_err(GalaxError::Io)?;
    buf.write_all(&header.row_count.to_le_bytes())
        .map_err(GalaxError::Io)?;
    buf.write_all(&header.commit_timestamp.to_le_bytes())
        .map_err(GalaxError::Io)?;
    buf.write_all(&header.column_count.to_le_bytes())
        .map_err(GalaxError::Io)?;

    for desc in &header.column_descriptors {
        // col_type: u8
        buf.write_all(&[column_type_to_u8(&desc.col_type)])
            .map_err(GalaxError::Io)?;

        // For Embedding type, write the dimensions as u32
        if let ColumnType::Embedding(dims) = &desc.col_type {
            buf.write_all(&dims.to_le_bytes())
                .map_err(GalaxError::Io)?;
        }

        // codec: u8
        buf.write_all(&[desc.codec as u8])
            .map_err(GalaxError::Io)?;

        // offset: u64
        buf.write_all(&desc.offset.to_le_bytes())
            .map_err(GalaxError::Io)?;

        // compressed_len: u32
        buf.write_all(&desc.compressed_len.to_le_bytes())
            .map_err(GalaxError::Io)?;

        // zone_map_min: u32 length + bytes
        let min_len = desc.zone_map_min.len() as u32;
        buf.write_all(&min_len.to_le_bytes())
            .map_err(GalaxError::Io)?;
        buf.write_all(&desc.zone_map_min)
            .map_err(GalaxError::Io)?;

        // zone_map_max: u32 length + bytes
        let max_len = desc.zone_map_max.len() as u32;
        buf.write_all(&max_len.to_le_bytes())
            .map_err(GalaxError::Io)?;
        buf.write_all(&desc.zone_map_max)
            .map_err(GalaxError::Io)?;
    }

    Ok(())
}

/// Deserialize a PAX block header from a cursor.
fn deserialize_header(cursor: &mut Cursor<&[u8]>) -> GalaxResult<PaxBlockHeader> {
    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];
    let mut buf2 = [0u8; 2];
    let mut buf1 = [0u8; 1];

    // magic: u32
    cursor.read_exact(&mut buf4).map_err(GalaxError::Io)?;
    let magic = u32::from_le_bytes(buf4);

    // format_version: u8
    cursor.read_exact(&mut buf1).map_err(GalaxError::Io)?;
    let format_version = buf1[0];

    // block_id: u64
    cursor.read_exact(&mut buf8).map_err(GalaxError::Io)?;
    let block_id = u64::from_le_bytes(buf8);

    // row_count: u32
    cursor.read_exact(&mut buf4).map_err(GalaxError::Io)?;
    let row_count = u32::from_le_bytes(buf4);

    // commit_timestamp: u64
    cursor.read_exact(&mut buf8).map_err(GalaxError::Io)?;
    let commit_timestamp = u64::from_le_bytes(buf8);

    // column_count: u16
    cursor.read_exact(&mut buf2).map_err(GalaxError::Io)?;
    let column_count = u16::from_le_bytes(buf2);

    let mut column_descriptors = Vec::with_capacity(column_count as usize);
    for _ in 0..column_count {
        // col_type: u8
        cursor.read_exact(&mut buf1).map_err(GalaxError::Io)?;
        let col_type_byte = buf1[0];

        // If embedding, read dimensions
        let embedding_dims = if col_type_byte == 14 {
            cursor.read_exact(&mut buf4).map_err(GalaxError::Io)?;
            u32::from_le_bytes(buf4)
        } else {
            0
        };

        let col_type = column_type_from_u8(col_type_byte, embedding_dims).ok_or_else(|| {
            GalaxError::Internal(format!("unknown column type byte: {}", col_type_byte))
        })?;

        // codec: u8
        cursor.read_exact(&mut buf1).map_err(GalaxError::Io)?;
        let codec = CodecId::from_u8(buf1[0]).ok_or_else(|| {
            GalaxError::Internal(format!("unknown codec byte: {}", buf1[0]))
        })?;

        // offset: u64
        cursor.read_exact(&mut buf8).map_err(GalaxError::Io)?;
        let offset = u64::from_le_bytes(buf8);

        // compressed_len: u32
        cursor.read_exact(&mut buf4).map_err(GalaxError::Io)?;
        let compressed_len = u32::from_le_bytes(buf4);

        // zone_map_min: u32 length + bytes
        cursor.read_exact(&mut buf4).map_err(GalaxError::Io)?;
        let min_len = u32::from_le_bytes(buf4) as usize;
        let mut zone_map_min = vec![0u8; min_len];
        cursor.read_exact(&mut zone_map_min).map_err(GalaxError::Io)?;

        // zone_map_max: u32 length + bytes
        cursor.read_exact(&mut buf4).map_err(GalaxError::Io)?;
        let max_len = u32::from_le_bytes(buf4) as usize;
        let mut zone_map_max = vec![0u8; max_len];
        cursor.read_exact(&mut zone_map_max).map_err(GalaxError::Io)?;

        column_descriptors.push(ColumnDescriptor {
            col_type,
            codec,
            offset,
            compressed_len,
            zone_map_min,
            zone_map_max,
        });
    }

    Ok(PaxBlockHeader {
        magic,
        format_version,
        block_id,
        row_count,
        commit_timestamp,
        column_count,
        column_descriptors,
    })
}

/// Serialize the row offset table to a buffer.
fn serialize_row_offsets(offsets: &[u32], buf: &mut Vec<u8>) -> GalaxResult<()> {
    for &offset in offsets {
        buf.write_all(&offset.to_le_bytes())
            .map_err(GalaxError::Io)?;
    }
    Ok(())
}
