//! Adaptive Radix Tree (ART) benchmarks.
//!
//! Measures insert, lookup, and delete throughput for sequential and random
//! key distributions at 1M scale.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use galaxdb_storage::art::{ArtIndex, RowLocation};
use std::time::Duration;

const NUM_KEYS: usize = 1_000_000;

/// Generate a sequential 16-byte key from an index.
fn sequential_key(i: usize) -> Vec<u8> {
    // 16-byte key: 8-byte prefix + 8-byte big-endian index
    let mut key = vec![b'k', b'e', b'y', b'_', b's', b'e', b'q', b'_'];
    key.extend_from_slice(&(i as u64).to_be_bytes());
    key
}

/// Generate pseudo-random 16-byte keys using a simple LCG.
/// Pre-generates all keys to avoid measuring RNG in the benchmark.
fn random_keys(count: usize) -> Vec<Vec<u8>> {
    let mut keys = Vec::with_capacity(count);
    let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
    for _ in 0..count {
        // LCG: state = state * 6364136223846793005 + 1442695040888963407
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let mut key = vec![b'k', b'e', b'y', b'_', b'r', b'n', b'd', b'_'];
        key.extend_from_slice(&state.to_be_bytes());
        keys.push(key);
    }
    keys
}

fn make_location(i: usize) -> RowLocation {
    RowLocation::SST {
        sst_id: (i / 1000) as u64,
        block_offset: (i % 1000) as u64,
        row_offset: (i % 256) as u32,
    }
}

fn art_insert_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("art_insert");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    group.bench_function("insert_1m_sequential", |b| {
        b.iter(|| {
            let index = ArtIndex::new();
            for i in 0..NUM_KEYS {
                index.insert(sequential_key(i), make_location(i));
            }
            black_box(index.len());
        });
    });

    group.finish();
}

fn art_insert_random(c: &mut Criterion) {
    let keys = random_keys(NUM_KEYS);

    let mut group = c.benchmark_group("art_insert");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    group.bench_function("insert_1m_random", |b| {
        b.iter(|| {
            let index = ArtIndex::new();
            for (i, key) in keys.iter().enumerate() {
                index.insert(key.clone(), make_location(i));
            }
            black_box(index.len());
        });
    });

    group.finish();
}

fn art_lookup_sequential(c: &mut Criterion) {
    // Pre-populate the tree
    let index = ArtIndex::new();
    for i in 0..NUM_KEYS {
        index.insert(sequential_key(i), make_location(i));
    }

    let mut group = c.benchmark_group("art_lookup");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    group.bench_function("lookup_1m_sequential_warm", |b| {
        b.iter(|| {
            for i in 0..NUM_KEYS {
                let result = index.lookup(&sequential_key(i));
                black_box(result);
            }
        });
    });

    group.finish();
}

fn art_lookup_random(c: &mut Criterion) {
    let keys = random_keys(NUM_KEYS);

    // Pre-populate the tree with random keys
    let index = ArtIndex::new();
    for (i, key) in keys.iter().enumerate() {
        index.insert(key.clone(), make_location(i));
    }

    let mut group = c.benchmark_group("art_lookup");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    group.bench_function("lookup_1m_random_warm", |b| {
        b.iter(|| {
            for key in &keys {
                let result = index.lookup(key);
                black_box(result);
            }
        });
    });

    group.finish();
}

fn art_delete(c: &mut Criterion) {
    let keys = random_keys(NUM_KEYS);

    let mut group = c.benchmark_group("art_delete");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    group.bench_function("delete_100k_keys", |b| {
        b.iter_with_setup(
            || {
                // Setup: build a tree with 1M keys
                let index = ArtIndex::new();
                for (i, key) in keys.iter().enumerate() {
                    index.insert(key.clone(), make_location(i));
                }
                index
            },
            |index| {
                // Benchmark: delete the first 100K keys
                for key in &keys[..100_000] {
                    let result = index.delete(key);
                    black_box(result);
                }
                black_box(index.len());
            },
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    art_insert_sequential,
    art_insert_random,
    art_lookup_sequential,
    art_lookup_random,
    art_delete,
);
criterion_main!(benches);
