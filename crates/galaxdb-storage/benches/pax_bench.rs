//! PAX block encode/decode benchmarks.
//!
//! Measures encode, decode, roundtrip, XXH3-64 checksum, and zone map extraction
//! for realistic 1000-row blocks with Int32 + Text + Blob columns.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use galaxdb_common::ColumnType;
use galaxdb_storage::pax::{ColumnData, PaxBlock};
use xxhash_rust::xxh3::xxh3_64;

/// Build realistic column data: 1000 rows with Int32, Text, and Blob columns.
fn build_test_columns(row_count: usize) -> Vec<ColumnData> {
    let mut int_values = Vec::with_capacity(row_count);
    let mut text_values = Vec::with_capacity(row_count);
    let mut blob_values = Vec::with_capacity(row_count);

    for i in 0..row_count {
        // Int32 column: sequential values
        int_values.push((i as i32).to_le_bytes().to_vec());

        // Text column: realistic variable-length strings (30-80 bytes)
        let text = format!(
            "user_{:06}_record_data_payload_with_some_realistic_length_{}",
            i,
            i % 100
        );
        text_values.push(text.into_bytes());

        // Blob column: 128-byte binary payloads
        let mut blob = vec![0u8; 128];
        for (j, byte) in blob.iter_mut().enumerate() {
            *byte = ((i * 7 + j * 13) % 256) as u8;
        }
        blob_values.push(blob);
    }

    vec![
        ColumnData {
            col_type: ColumnType::Int32,
            values: int_values,
        },
        ColumnData {
            col_type: ColumnType::Text,
            values: text_values,
        },
        ColumnData {
            col_type: ColumnType::Blob,
            values: blob_values,
        },
    ]
}

fn pax_encode(c: &mut Criterion) {
    let columns = build_test_columns(1000);

    c.bench_function("pax_block_encode_1000_rows", |b| {
        b.iter(|| {
            let block = PaxBlock::write(
                black_box(1),
                black_box(1000),
                black_box(&columns),
            )
            .unwrap();
            black_box(block);
        });
    });
}

fn pax_decode(c: &mut Criterion) {
    let columns = build_test_columns(1000);
    let block = PaxBlock::write(1, 1000, &columns).unwrap();
    let serialized = block.serialize().unwrap();

    c.bench_function("pax_block_decode_1000_rows", |b| {
        b.iter(|| {
            let decoded = PaxBlock::deserialize(black_box(&serialized)).unwrap();
            black_box(decoded);
        });
    });
}

fn pax_roundtrip(c: &mut Criterion) {
    let columns = build_test_columns(1000);

    c.bench_function("pax_block_encode_decode_roundtrip", |b| {
        b.iter(|| {
            let block = PaxBlock::write(
                black_box(1),
                black_box(1000),
                black_box(&columns),
            )
            .unwrap();
            let serialized = block.serialize().unwrap();
            let decoded = PaxBlock::deserialize(black_box(&serialized)).unwrap();
            black_box(decoded);
        });
    });
}

fn xxh3_checksum_1mb(c: &mut Criterion) {
    // Simulate a 1 MB block
    let data: Vec<u8> = (0..1_048_576).map(|i| (i % 256) as u8).collect();

    c.bench_function("xxh3_64_checksum_1mb", |b| {
        b.iter(|| {
            let hash = xxh3_64(black_box(&data));
            black_box(hash);
        });
    });
}

fn zone_map_extraction(c: &mut Criterion) {
    // Build a 1000-row block and measure how long it takes to encode
    // (which includes zone map extraction internally).
    // To isolate zone map cost, we encode and then read back column data
    // which triggers decompression + zone map verification.
    let columns = build_test_columns(1000);
    let block = PaxBlock::write(1, 1000, &columns).unwrap();
    let serialized = block.serialize().unwrap();

    c.bench_function("pax_zone_map_extraction_1000_rows", |b| {
        b.iter(|| {
            let decoded = PaxBlock::deserialize(black_box(&serialized)).unwrap();
            // Access zone maps from all column descriptors
            for desc in &decoded.header.column_descriptors {
                black_box(&desc.zone_map_min);
                black_box(&desc.zone_map_max);
            }
            // Also decompress each column to verify zone maps are correct
            for col_idx in 0..decoded.header.column_count as usize {
                let col_data = decoded.read_column(col_idx).unwrap();
                black_box(col_data);
            }
        });
    });
}

criterion_group!(
    benches,
    pax_encode,
    pax_decode,
    pax_roundtrip,
    xxh3_checksum_1mb,
    zone_map_extraction,
);
criterion_main!(benches);
