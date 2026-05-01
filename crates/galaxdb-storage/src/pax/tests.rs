//! Tests for the PAX block format.
//!
//! Covers: write/read round-trip, checksum verification, corrupt block
//! rejection, and compression correctness per column type.

use galaxdb_common::ColumnType;

use super::*;

/// Helper: create a simple Int32 column with sequential values.
fn make_int32_column(values: &[i32]) -> ColumnData {
    ColumnData {
        col_type: ColumnType::Int32,
        values: values.iter().map(|v| v.to_le_bytes().to_vec()).collect(),
    }
}

/// Helper: create a Text column.
fn make_text_column(values: &[&str]) -> ColumnData {
    ColumnData {
        col_type: ColumnType::Text,
        values: values.iter().map(|v| v.as_bytes().to_vec()).collect(),
    }
}

/// Helper: create an Embedding column with f32 vectors.
fn make_embedding_column(dims: u32, vectors: &[Vec<f32>]) -> ColumnData {
    ColumnData {
        col_type: ColumnType::Embedding(dims),
        values: vectors
            .iter()
            .map(|v| {
                v.iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect::<Vec<u8>>()
            })
            .collect(),
    }
}

// --- Write/Read Round-Trip Tests ---

#[test]
fn round_trip_single_int32_column() {
    let col = make_int32_column(&[10, 20, 30, 40, 50]);
    let block = PaxBlock::write(1, 1000, &[col.clone()]).unwrap();

    let serialized = block.serialize().unwrap();
    let deserialized = PaxBlock::deserialize(&serialized).unwrap();

    assert_eq!(deserialized.header.magic, PAX_MAGIC);
    assert_eq!(deserialized.header.format_version, PAX_FORMAT_VERSION);
    assert_eq!(deserialized.header.block_id, 1);
    assert_eq!(deserialized.header.row_count, 5);
    assert_eq!(deserialized.header.commit_timestamp, 1000);
    assert_eq!(deserialized.header.column_count, 1);

    let read_values = deserialized.read_column(0).unwrap();
    assert_eq!(read_values, col.values);
}

#[test]
fn round_trip_multiple_column_types() {
    let int_col = make_int32_column(&[100, 200, 300]);
    let text_col = make_text_column(&["hello", "world", "test"]);
    let embed_col = make_embedding_column(
        3,
        &[
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ],
    );

    let block = PaxBlock::write(42, 5000, &[int_col.clone(), text_col.clone(), embed_col.clone()])
        .unwrap();

    let serialized = block.serialize().unwrap();
    let deserialized = PaxBlock::deserialize(&serialized).unwrap();

    assert_eq!(deserialized.header.column_count, 3);
    assert_eq!(deserialized.header.row_count, 3);

    // Verify each column round-trips correctly
    let read_ints = deserialized.read_column(0).unwrap();
    assert_eq!(read_ints, int_col.values);

    let read_texts = deserialized.read_column(1).unwrap();
    assert_eq!(read_texts, text_col.values);

    let read_embeds = deserialized.read_column(2).unwrap();
    assert_eq!(read_embeds, embed_col.values);
}

#[test]
fn round_trip_empty_rows() {
    // A block with zero rows should still serialize/deserialize correctly
    let col = ColumnData {
        col_type: ColumnType::Int32,
        values: vec![],
    };
    let block = PaxBlock::write(1, 100, &[col]).unwrap();
    let serialized = block.serialize().unwrap();
    let deserialized = PaxBlock::deserialize(&serialized).unwrap();

    assert_eq!(deserialized.header.row_count, 0);
    assert!(deserialized.row_offsets.is_empty());
}

// --- Checksum Verification Tests ---

#[test]
fn checksum_verification_passes_for_valid_block() {
    let col = make_int32_column(&[1, 2, 3]);
    let block = PaxBlock::write(1, 100, &[col]).unwrap();
    let serialized = block.serialize().unwrap();

    // Should deserialize without error
    let result = PaxBlock::deserialize(&serialized);
    assert!(result.is_ok());
}

#[test]
fn corrupt_block_rejected_with_checksum_mismatch() {
    let col = make_int32_column(&[1, 2, 3]);
    let block = PaxBlock::write(1, 100, &[col]).unwrap();
    let mut serialized = block.serialize().unwrap();

    // Corrupt a byte in the middle of the block (not the checksum)
    let mid = serialized.len() / 2;
    serialized[mid] ^= 0xFF;

    let result = PaxBlock::deserialize(&serialized);
    assert!(result.is_err());
    match result.unwrap_err() {
        GalaxError::ChecksumMismatch { .. } => {} // Expected
        other => panic!("expected ChecksumMismatch, got: {:?}", other),
    }
}

#[test]
fn corrupt_magic_number_rejected() {
    let col = make_int32_column(&[1, 2, 3]);
    let block = PaxBlock::write(1, 100, &[col]).unwrap();
    let mut serialized = block.serialize().unwrap();

    // Corrupt the magic number (first 4 bytes) and fix the checksum
    serialized[0] = 0x00;
    serialized[1] = 0x00;
    serialized[2] = 0x00;
    serialized[3] = 0x00;

    // Recompute checksum over the corrupted data (so checksum passes but magic fails)
    let checksum_offset = serialized.len() - 8;
    let new_checksum = xxh3_64(&serialized[..checksum_offset]);
    serialized[checksum_offset..].copy_from_slice(&new_checksum.to_le_bytes());

    let result = PaxBlock::deserialize(&serialized);
    assert!(result.is_err());
    match result.unwrap_err() {
        GalaxError::InvalidMagic(magic) => {
            assert_ne!(magic, PAX_MAGIC);
        }
        other => panic!("expected InvalidMagic, got: {:?}", other),
    }
}

// --- Compression Correctness Tests ---

#[test]
fn fixed_width_uses_fastpfor_codec() {
    let col = make_int32_column(&[10, 20, 30, 40, 50]);
    let block = PaxBlock::write(1, 100, &[col]).unwrap();

    assert_eq!(block.header.column_descriptors[0].codec, CodecId::FastPFor);
}

#[test]
fn variable_width_uses_zstd_codec() {
    let col = make_text_column(&["hello", "world"]);
    let block = PaxBlock::write(1, 100, &[col]).unwrap();

    assert_eq!(block.header.column_descriptors[0].codec, CodecId::Zstd);
}

#[test]
fn embedding_uses_none_codec() {
    let col = make_embedding_column(2, &[vec![1.0, 2.0], vec![3.0, 4.0]]);
    let block = PaxBlock::write(1, 100, &[col]).unwrap();

    assert_eq!(block.header.column_descriptors[0].codec, CodecId::None);
}

#[test]
fn fastpfor_compresses_sequential_integers() {
    // Sequential integers should compress well with delta encoding
    let values: Vec<i32> = (0..100).collect();
    let col = make_int32_column(&values);
    let block = PaxBlock::write(1, 100, &[col.clone()]).unwrap();

    // Verify round-trip
    let serialized = block.serialize().unwrap();
    let deserialized = PaxBlock::deserialize(&serialized).unwrap();
    let read_values = deserialized.read_column(0).unwrap();
    assert_eq!(read_values, col.values);

    // The compressed column data should be smaller than raw data
    let raw_size = 100 * 4; // 100 i32 values
    let compressed_size = block.header.column_descriptors[0].compressed_len;
    assert!(
        (compressed_size as usize) < raw_size,
        "FastPFOR should compress sequential integers: compressed={}, raw={}",
        compressed_size,
        raw_size
    );
}

#[test]
fn zstd_compresses_repetitive_text() {
    // Repetitive text should compress well with Zstd
    let values: Vec<&str> = (0..50).map(|_| "the quick brown fox jumps over the lazy dog").collect();
    let col = make_text_column(&values);
    let block = PaxBlock::write(1, 100, &[col.clone()]).unwrap();

    // Verify round-trip
    let serialized = block.serialize().unwrap();
    let deserialized = PaxBlock::deserialize(&serialized).unwrap();
    let read_values = deserialized.read_column(0).unwrap();
    assert_eq!(read_values, col.values);
}

#[test]
fn fastpfor_handles_negative_deltas() {
    // Decreasing values produce negative deltas
    let values: Vec<i32> = vec![100, 50, 25, 10, 5, 1, 0, -10, -100];
    let col = make_int32_column(&values);
    let block = PaxBlock::write(1, 100, &[col.clone()]).unwrap();

    let serialized = block.serialize().unwrap();
    let deserialized = PaxBlock::deserialize(&serialized).unwrap();
    let read_values = deserialized.read_column(0).unwrap();
    assert_eq!(read_values, col.values);
}

#[test]
fn fastpfor_handles_single_value() {
    let col = make_int32_column(&[42]);
    let block = PaxBlock::write(1, 100, &[col.clone()]).unwrap();

    let serialized = block.serialize().unwrap();
    let deserialized = PaxBlock::deserialize(&serialized).unwrap();
    let read_values = deserialized.read_column(0).unwrap();
    assert_eq!(read_values, col.values);
}

// --- Zone Map Tests ---

#[test]
fn zone_map_captures_min_max_for_int32() {
    let col = make_int32_column(&[50, 10, 90, 30, 70]);
    let block = PaxBlock::write(1, 100, &[col]).unwrap();

    let desc = &block.header.column_descriptors[0];
    // Min should be 10, max should be 90
    let min_val = i32::from_le_bytes(desc.zone_map_min.clone().try_into().unwrap());
    let max_val = i32::from_le_bytes(desc.zone_map_max.clone().try_into().unwrap());
    assert_eq!(min_val, 10);
    assert_eq!(max_val, 90);
}

#[test]
fn zone_map_captures_min_max_for_text() {
    let col = make_text_column(&["banana", "apple", "cherry"]);
    let block = PaxBlock::write(1, 100, &[col]).unwrap();

    let desc = &block.header.column_descriptors[0];
    assert_eq!(desc.zone_map_min, b"apple");
    assert_eq!(desc.zone_map_max, b"cherry");
}

#[test]
fn zone_map_preserved_through_serialization() {
    let col = make_int32_column(&[5, 1, 9, 3, 7]);
    let block = PaxBlock::write(1, 100, &[col]).unwrap();
    let serialized = block.serialize().unwrap();
    let deserialized = PaxBlock::deserialize(&serialized).unwrap();

    let orig_desc = &block.header.column_descriptors[0];
    let read_desc = &deserialized.header.column_descriptors[0];
    assert_eq!(orig_desc.zone_map_min, read_desc.zone_map_min);
    assert_eq!(orig_desc.zone_map_max, read_desc.zone_map_max);
}

// --- Row Offset Table Tests ---

#[test]
fn row_offsets_are_correct() {
    let col1 = make_int32_column(&[1, 2, 3]); // 4 bytes each
    let col2 = make_text_column(&["hi", "hey", "hello"]); // 2, 3, 5 bytes

    let block = PaxBlock::write(1, 100, &[col1, col2]).unwrap();

    // Row 0: offset 0, size = 4 + 2 = 6
    // Row 1: offset 6, size = 4 + 3 = 7
    // Row 2: offset 13, size = 4 + 5 = 9
    assert_eq!(block.row_offsets, vec![0, 6, 13]);
}

// --- Additional Column Type Tests ---

#[test]
fn round_trip_int64_column() {
    let values: Vec<i64> = vec![i64::MIN, -1, 0, 1, i64::MAX];
    let col = ColumnData {
        col_type: ColumnType::Int64,
        values: values.iter().map(|v| v.to_le_bytes().to_vec()).collect(),
    };
    let block = PaxBlock::write(1, 100, &[col.clone()]).unwrap();
    let serialized = block.serialize().unwrap();
    let deserialized = PaxBlock::deserialize(&serialized).unwrap();
    let read_values = deserialized.read_column(0).unwrap();
    assert_eq!(read_values, col.values);
}

#[test]
fn round_trip_uint32_column() {
    let values: Vec<u32> = vec![0, 1, 100, 1000, u32::MAX];
    let col = ColumnData {
        col_type: ColumnType::UInt32,
        values: values.iter().map(|v| v.to_le_bytes().to_vec()).collect(),
    };
    let block = PaxBlock::write(1, 100, &[col.clone()]).unwrap();
    let serialized = block.serialize().unwrap();
    let deserialized = PaxBlock::deserialize(&serialized).unwrap();
    let read_values = deserialized.read_column(0).unwrap();
    assert_eq!(read_values, col.values);
}

#[test]
fn round_trip_boolean_column() {
    let col = ColumnData {
        col_type: ColumnType::Boolean,
        values: vec![vec![1], vec![0], vec![1], vec![1], vec![0]],
    };
    let block = PaxBlock::write(1, 100, &[col.clone()]).unwrap();
    let serialized = block.serialize().unwrap();
    let deserialized = PaxBlock::deserialize(&serialized).unwrap();
    let read_values = deserialized.read_column(0).unwrap();
    assert_eq!(read_values, col.values);
}

#[test]
fn round_trip_blob_column() {
    let col = ColumnData {
        col_type: ColumnType::Blob,
        values: vec![
            vec![0x00, 0xFF, 0xAB],
            vec![0x01, 0x02],
            vec![0xDE, 0xAD, 0xBE, 0xEF],
        ],
    };
    let block = PaxBlock::write(1, 100, &[col.clone()]).unwrap();
    let serialized = block.serialize().unwrap();
    let deserialized = PaxBlock::deserialize(&serialized).unwrap();
    let read_values = deserialized.read_column(0).unwrap();
    assert_eq!(read_values, col.values);
}

#[test]
fn round_trip_json_column() {
    let col = ColumnData {
        col_type: ColumnType::Json,
        values: vec![
            b"{\"key\": \"value\"}".to_vec(),
            b"[1, 2, 3]".to_vec(),
            b"null".to_vec(),
        ],
    };
    let block = PaxBlock::write(1, 100, &[col.clone()]).unwrap();
    let serialized = block.serialize().unwrap();
    let deserialized = PaxBlock::deserialize(&serialized).unwrap();
    let read_values = deserialized.read_column(0).unwrap();
    assert_eq!(read_values, col.values);
}

#[test]
fn round_trip_float64_column() {
    let values: Vec<f64> = vec![-1.5, 0.0, 1.5, f64::MIN, f64::MAX];
    let col = ColumnData {
        col_type: ColumnType::Float64,
        values: values.iter().map(|v| v.to_le_bytes().to_vec()).collect(),
    };
    let block = PaxBlock::write(1, 100, &[col.clone()]).unwrap();
    let serialized = block.serialize().unwrap();
    let deserialized = PaxBlock::deserialize(&serialized).unwrap();
    let read_values = deserialized.read_column(0).unwrap();
    assert_eq!(read_values, col.values);
}

#[test]
fn error_on_mismatched_row_counts() {
    let col1 = make_int32_column(&[1, 2, 3]);
    let col2 = make_text_column(&["a", "b"]); // Different row count

    let result = PaxBlock::write(1, 100, &[col1, col2]);
    assert!(result.is_err());
}

#[test]
fn error_on_empty_columns() {
    let result = PaxBlock::write(1, 100, &[]);
    assert!(result.is_err());
}

#[test]
fn error_on_column_index_out_of_range() {
    let col = make_int32_column(&[1, 2, 3]);
    let block = PaxBlock::write(1, 100, &[col]).unwrap();
    let result = block.read_column(5);
    assert!(result.is_err());
}

#[test]
fn block_too_small_rejected() {
    let result = PaxBlock::deserialize(&[0u8; 4]);
    assert!(result.is_err());
}
