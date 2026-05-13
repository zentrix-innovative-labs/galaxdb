//! Compression codecs for PAX column chunks.
//!
//! - **FastPFOR** (codec 1): delta encoding + bit-packing for fixed-width integers.
//! - **Zstd** (codec 2): Zstandard level 3 for variable-width data.
//! - **None** (codec 0): raw passthrough for embedding columns.

use galaxdb_common::{ColumnType, GalaxError, GalaxResult};

use super::CodecId;

/// Compress column values using the specified codec.
pub fn compress(
    col_type: &ColumnType,
    codec: CodecId,
    values: &[Vec<u8>],
) -> GalaxResult<Vec<u8>> {
    match codec {
        CodecId::None => compress_none(values),
        CodecId::FastPFor => compress_fastpfor(col_type, values),
        CodecId::Zstd => compress_zstd(values),
    }
}

/// Decompress column data using the specified codec.
pub fn decompress(
    col_type: &ColumnType,
    codec: CodecId,
    data: &[u8],
    row_count: u32,
) -> GalaxResult<Vec<Vec<u8>>> {
    match codec {
        CodecId::None => decompress_none(col_type, data, row_count),
        CodecId::FastPFor => decompress_fastpfor(col_type, data, row_count),
        CodecId::Zstd => decompress_zstd(data, row_count),
    }
}

// --- No compression (codec 0) ---

/// No compression: length-prefix each value and concatenate.
fn compress_none(values: &[Vec<u8>]) -> GalaxResult<Vec<u8>> {
    let mut buf = Vec::new();
    for val in values {
        let len = val.len() as u32;
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(val);
    }
    Ok(buf)
}

/// Decompress no-compression data: read length-prefixed values.
fn decompress_none(
    _col_type: &ColumnType,
    data: &[u8],
    row_count: u32,
) -> GalaxResult<Vec<Vec<u8>>> {
    let mut values = Vec::with_capacity(row_count as usize);
    let mut offset = 0;

    for _ in 0..row_count {
        if offset + 4 > data.len() {
            return Err(GalaxError::Internal(
                "unexpected end of uncompressed column data".into(),
            ));
        }
        let len = u32::from_le_bytes(
            data[offset..offset + 4]
                .try_into()
                .map_err(|_| GalaxError::Internal("failed to read value length".into()))?,
        ) as usize;
        offset += 4;

        if offset + len > data.len() {
            return Err(GalaxError::Internal(
                "value extends beyond column data".into(),
            ));
        }
        values.push(data[offset..offset + len].to_vec());
        offset += len;
    }

    Ok(values)
}

/// Read a single row from uncompressed (CodecId::None) column data.
///
/// Scans through length-prefixed values to reach the target row.
/// Much faster than decompressing the entire column when only one row is needed.
pub fn decompress_none_single_row(
    data: &[u8],
    target_row: u32,
) -> GalaxResult<Vec<u8>> {
    let mut offset = 0;

    for row in 0..=target_row {
        if offset + 4 > data.len() {
            return Err(GalaxError::Internal(
                "unexpected end of uncompressed column data".into(),
            ));
        }
        let len = u32::from_le_bytes(
            data[offset..offset + 4]
                .try_into()
                .map_err(|_| GalaxError::Internal("failed to read value length".into()))?,
        ) as usize;
        offset += 4;

        if offset + len > data.len() {
            return Err(GalaxError::Internal(
                "value extends beyond column data".into(),
            ));
        }

        if row == target_row {
            return Ok(data[offset..offset + len].to_vec());
        }
        offset += len;
    }

    Err(GalaxError::Internal("target row not reached".into()))
}

// --- FastPFOR: delta encoding + bit-packing (codec 1) ---

/// Compress fixed-width integer values using delta encoding + bit-packing.
///
/// Format:
/// ```text
/// [byte_width: u8]
/// [first_value: [u8; byte_width]]  (stored raw for delta base)
/// [max_bits: u8]                   (bits needed per delta)
/// [packed_deltas: bytes]           (bit-packed delta values)
/// ```
///
/// Delta encoding stores differences between consecutive values. Bit-packing
/// stores each delta using only the minimum number of bits needed.
fn compress_fastpfor(col_type: &ColumnType, values: &[Vec<u8>]) -> GalaxResult<Vec<u8>> {
    let byte_width = col_type.byte_size().ok_or_else(|| {
        GalaxError::Internal("FastPFOR requires fixed-width column type".into())
    })?;

    if values.is_empty() {
        return Ok(vec![byte_width as u8]);
    }

    // Convert values to u64 for uniform delta computation
    let u64_values: Vec<u64> = values
        .iter()
        .map(|v| bytes_to_u64(col_type, v))
        .collect();

    // Compute deltas (zigzag-encoded for signed types)
    let is_signed = matches!(
        col_type,
        ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32 | ColumnType::Int64
    );

    let deltas: Vec<u64> = if u64_values.len() <= 1 {
        Vec::new()
    } else {
        u64_values
            .windows(2)
            .map(|w| {
                if is_signed {
                    // Zigzag encode the signed difference
                    let diff = (w[1] as i64).wrapping_sub(w[0] as i64);
                    zigzag_encode(diff)
                } else {
                    // For unsigned, store the raw difference (wrapping)
                    w[1].wrapping_sub(w[0])
                }
            })
            .collect()
    };

    // Find the maximum number of bits needed
    let max_bits = if deltas.is_empty() {
        0u8
    } else {
        let max_delta = deltas.iter().copied().max().unwrap_or(0);
        if max_delta == 0 {
            0
        } else {
            64 - max_delta.leading_zeros() as u8
        }
    };

    // Build output
    let mut buf = Vec::new();
    buf.push(byte_width as u8);

    // Store first value raw
    buf.extend_from_slice(&values[0]);

    // Store max_bits
    buf.push(max_bits);

    // Bit-pack the deltas
    if max_bits > 0 && !deltas.is_empty() {
        let packed = bitpack_encode(&deltas, max_bits);
        // Store packed data length for safe decoding
        let packed_len = packed.len() as u32;
        buf.extend_from_slice(&packed_len.to_le_bytes());
        buf.extend_from_slice(&packed);
    }

    Ok(buf)
}

/// Decompress FastPFOR-encoded data back to individual values.
fn decompress_fastpfor(
    col_type: &ColumnType,
    data: &[u8],
    row_count: u32,
) -> GalaxResult<Vec<Vec<u8>>> {
    if data.is_empty() {
        return Err(GalaxError::Internal("empty FastPFOR data".into()));
    }

    let byte_width = data[0] as usize;
    let mut offset = 1;

    if row_count == 0 {
        return Ok(Vec::new());
    }

    // Read first value
    if offset + byte_width > data.len() {
        return Err(GalaxError::Internal("FastPFOR: missing first value".into()));
    }
    let first_value_bytes = data[offset..offset + byte_width].to_vec();
    let first_u64 = bytes_to_u64(col_type, &first_value_bytes);
    offset += byte_width;

    if row_count == 1 {
        return Ok(vec![first_value_bytes]);
    }

    // Read max_bits
    if offset >= data.len() {
        return Err(GalaxError::Internal("FastPFOR: missing max_bits".into()));
    }
    let max_bits = data[offset];
    offset += 1;

    let is_signed = matches!(
        col_type,
        ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32 | ColumnType::Int64
    );

    let delta_count = (row_count - 1) as usize;

    let deltas = if max_bits == 0 {
        vec![0u64; delta_count]
    } else {
        // Read packed data length
        if offset + 4 > data.len() {
            return Err(GalaxError::Internal("FastPFOR: missing packed length".into()));
        }
        let packed_len = u32::from_le_bytes(
            data[offset..offset + 4]
                .try_into()
                .map_err(|_| GalaxError::Internal("failed to read packed length".into()))?,
        ) as usize;
        offset += 4;

        if offset + packed_len > data.len() {
            return Err(GalaxError::Internal(
                "FastPFOR: packed data extends beyond buffer".into(),
            ));
        }

        let packed = &data[offset..offset + packed_len];
        bitpack_decode(packed, max_bits, delta_count)
    };

    // Reconstruct values from deltas
    let mut u64_values = Vec::with_capacity(row_count as usize);
    u64_values.push(first_u64);

    for delta in &deltas {
        let prev = *u64_values.last().unwrap();
        let next = if is_signed {
            let signed_delta = zigzag_decode(*delta);
            (prev as i64).wrapping_add(signed_delta) as u64
        } else {
            prev.wrapping_add(*delta)
        };
        u64_values.push(next);
    }

    // Convert back to byte representations
    let values: Vec<Vec<u8>> = u64_values
        .iter()
        .map(|&v| u64_to_bytes(col_type, v, byte_width))
        .collect();

    Ok(values)
}

// --- Zstandard level 3 (codec 2) ---

/// Compress variable-width values using Zstandard level 3.
///
/// Format: length-prefix each value, concatenate, then Zstd-compress the whole thing.
/// The output is: [uncompressed_len: u32][zstd_compressed_data]
fn compress_zstd(values: &[Vec<u8>]) -> GalaxResult<Vec<u8>> {
    // First, serialize values with length prefixes
    let mut raw = Vec::new();
    for val in values {
        let len = val.len() as u32;
        raw.extend_from_slice(&len.to_le_bytes());
        raw.extend_from_slice(val);
    }

    // Compress with Zstandard level 3
    let compressed = zstd::encode_all(raw.as_slice(), 3)
        .map_err(|e| GalaxError::Internal(format!("Zstd compression failed: {}", e)))?;

    // Output: uncompressed length + compressed data
    let mut buf = Vec::new();
    let uncompressed_len = raw.len() as u32;
    buf.extend_from_slice(&uncompressed_len.to_le_bytes());
    buf.extend_from_slice(&compressed);

    Ok(buf)
}

/// Decompress Zstd-compressed variable-width data.
fn decompress_zstd(data: &[u8], row_count: u32) -> GalaxResult<Vec<Vec<u8>>> {
    if data.len() < 4 {
        return Err(GalaxError::Internal("Zstd data too small".into()));
    }

    let _uncompressed_len = u32::from_le_bytes(
        data[0..4]
            .try_into()
            .map_err(|_| GalaxError::Internal("failed to read uncompressed length".into()))?,
    );

    let compressed = &data[4..];
    let raw = zstd::decode_all(compressed)
        .map_err(|e| GalaxError::Internal(format!("Zstd decompression failed: {}", e)))?;

    // Parse length-prefixed values
    let mut values = Vec::with_capacity(row_count as usize);
    let mut offset = 0;

    for _ in 0..row_count {
        if offset + 4 > raw.len() {
            return Err(GalaxError::Internal(
                "unexpected end of Zstd-decompressed data".into(),
            ));
        }
        let len = u32::from_le_bytes(
            raw[offset..offset + 4]
                .try_into()
                .map_err(|_| GalaxError::Internal("failed to read value length".into()))?,
        ) as usize;
        offset += 4;

        if offset + len > raw.len() {
            return Err(GalaxError::Internal(
                "value extends beyond decompressed data".into(),
            ));
        }
        values.push(raw[offset..offset + len].to_vec());
        offset += len;
    }

    Ok(values)
}

// --- Utility functions ---

/// Convert raw bytes to u64 based on column type (for delta encoding).
fn bytes_to_u64(col_type: &ColumnType, bytes: &[u8]) -> u64 {
    match col_type {
        ColumnType::Int8 | ColumnType::UInt8 | ColumnType::Boolean => {
            if bytes.is_empty() { 0 } else { bytes[0] as u64 }
        }
        ColumnType::Int16 | ColumnType::UInt16 => {
            if bytes.len() < 2 {
                0
            } else {
                u16::from_le_bytes(bytes[..2].try_into().unwrap()) as u64
            }
        }
        ColumnType::Int32 | ColumnType::UInt32 | ColumnType::Float32 => {
            if bytes.len() < 4 {
                0
            } else {
                u32::from_le_bytes(bytes[..4].try_into().unwrap()) as u64
            }
        }
        ColumnType::Int64 | ColumnType::UInt64 | ColumnType::Float64 => {
            if bytes.len() < 8 {
                0
            } else {
                u64::from_le_bytes(bytes[..8].try_into().unwrap())
            }
        }
        _ => 0, // Should not be called for non-fixed-width types
    }
}

/// Convert a u64 back to the byte representation for a column type.
fn u64_to_bytes(col_type: &ColumnType, value: u64, byte_width: usize) -> Vec<u8> {
    match col_type {
        ColumnType::Int8 | ColumnType::UInt8 | ColumnType::Boolean => {
            vec![value as u8]
        }
        ColumnType::Int16 | ColumnType::UInt16 => {
            (value as u16).to_le_bytes().to_vec()
        }
        ColumnType::Int32 | ColumnType::UInt32 | ColumnType::Float32 => {
            (value as u32).to_le_bytes().to_vec()
        }
        ColumnType::Int64 | ColumnType::UInt64 | ColumnType::Float64 => {
            value.to_le_bytes().to_vec()
        }
        _ => {
            // Fallback: truncate to byte_width
            let full = value.to_le_bytes();
            full[..byte_width].to_vec()
        }
    }
}

/// Zigzag encode a signed i64 to an unsigned u64.
/// Maps: 0 → 0, -1 → 1, 1 → 2, -2 → 3, 2 → 4, ...
fn zigzag_encode(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

/// Zigzag decode an unsigned u64 back to a signed i64.
fn zigzag_decode(value: u64) -> i64 {
    ((value >> 1) as i64) ^ (-((value & 1) as i64))
}

/// Bit-pack an array of u64 values using `bits` bits per value.
fn bitpack_encode(values: &[u64], bits: u8) -> Vec<u8> {
    if bits == 0 || values.is_empty() {
        return Vec::new();
    }

    let bits = bits as usize;
    let total_bits = values.len() * bits;
    let total_bytes = total_bits.div_ceil(8);
    let mut output = vec![0u8; total_bytes];

    let mut bit_offset = 0usize;
    for &val in values {
        let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
        let masked_val = val & mask;

        // Write `bits` bits starting at `bit_offset`
        let mut remaining_bits = bits;
        let mut current_val = masked_val;
        let mut current_bit = bit_offset;

        while remaining_bits > 0 {
            let byte_idx = current_bit / 8;
            let bit_in_byte = current_bit % 8;
            let bits_available = 8 - bit_in_byte;
            let bits_to_write = remaining_bits.min(bits_available);

            let byte_mask = ((1u64 << bits_to_write) - 1) as u8;
            output[byte_idx] |= ((current_val as u8) & byte_mask) << bit_in_byte;

            current_val >>= bits_to_write;
            current_bit += bits_to_write;
            remaining_bits -= bits_to_write;
        }

        bit_offset += bits;
    }

    output
}

/// Decode bit-packed values.
fn bitpack_decode(data: &[u8], bits: u8, count: usize) -> Vec<u64> {
    if bits == 0 || count == 0 {
        return vec![0u64; count];
    }

    let bits = bits as usize;
    let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
    let mut values = Vec::with_capacity(count);

    let mut bit_offset = 0usize;
    for _ in 0..count {
        let mut value: u64 = 0;
        let mut remaining_bits = bits;
        let mut current_bit = bit_offset;
        let mut value_shift = 0;

        while remaining_bits > 0 {
            let byte_idx = current_bit / 8;
            let bit_in_byte = current_bit % 8;
            let bits_available = 8 - bit_in_byte;
            let bits_to_read = remaining_bits.min(bits_available);

            if byte_idx < data.len() {
                let byte_mask = ((1u64 << bits_to_read) - 1) as u8;
                let extracted = (data[byte_idx] >> bit_in_byte) & byte_mask;
                value |= (extracted as u64) << value_shift;
            }

            current_bit += bits_to_read;
            value_shift += bits_to_read;
            remaining_bits -= bits_to_read;
        }

        values.push(value & mask);
        bit_offset += bits;
    }

    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zigzag_round_trip() {
        for val in [-100i64, -1, 0, 1, 100, i64::MIN, i64::MAX] {
            assert_eq!(zigzag_decode(zigzag_encode(val)), val);
        }
    }

    #[test]
    fn bitpack_round_trip() {
        let values = vec![0u64, 1, 2, 3, 7, 5, 6, 4];
        let bits = 3u8;
        let packed = bitpack_encode(&values, bits);
        let decoded = bitpack_decode(&packed, bits, values.len());
        assert_eq!(decoded, values);
    }

    #[test]
    fn bitpack_round_trip_large_values() {
        let values = vec![255u64, 128, 0, 64, 192, 1, 254, 127];
        let bits = 8u8;
        let packed = bitpack_encode(&values, bits);
        let decoded = bitpack_decode(&packed, bits, values.len());
        assert_eq!(decoded, values);
    }

    #[test]
    fn bitpack_zero_bits() {
        let values = vec![0u64; 10];
        let packed = bitpack_encode(&values, 0);
        assert!(packed.is_empty());
        let decoded = bitpack_decode(&packed, 0, 10);
        assert_eq!(decoded, values);
    }
}
