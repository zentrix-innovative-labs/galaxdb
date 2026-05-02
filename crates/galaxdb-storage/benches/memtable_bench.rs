//! Memtable benchmarks.
//!
//! Measures insert throughput, read throughput, concurrent multi-shard insert,
//! and seal + swap latency.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use galaxdb_storage::memtable::Memtable;
use std::sync::Arc;
use std::time::Duration;

const NUM_ENTRIES: usize = 100_000;

/// Generate a deterministic 64-byte key.
fn make_key(i: usize) -> Vec<u8> {
    let mut key = Vec::with_capacity(64);
    key.extend_from_slice(b"memtable_bench_key_");
    key.extend_from_slice(&(i as u64).to_be_bytes());
    // Pad to 64 bytes with a deterministic pattern
    while key.len() < 64 {
        key.push((i % 256) as u8);
    }
    key
}

/// Generate a deterministic 256-byte value.
fn make_value(i: usize) -> Vec<u8> {
    let mut value = vec![0u8; 256];
    let seed = (i as u64).to_le_bytes();
    for (j, byte) in value.iter_mut().enumerate() {
        *byte = seed[j % 8] ^ (j as u8);
    }
    value
}

fn memtable_insert(c: &mut Criterion) {
    let keys: Vec<Vec<u8>> = (0..NUM_ENTRIES).map(make_key).collect();
    let values: Vec<Vec<u8>> = (0..NUM_ENTRIES).map(make_value).collect();

    let mut group = c.benchmark_group("memtable_insert");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));

    group.bench_function("insert_100k_64b_key_256b_value", |b| {
        b.iter(|| {
            // Use a very large threshold so we don't trigger seal during the benchmark
            let memtable = Memtable::new(u64::MAX);
            for (i, (key, value)) in keys.iter().zip(values.iter()).enumerate() {
                memtable.put(key.clone(), i as u64, Some(value.clone()));
            }
            black_box(memtable.size());
        });
    });

    group.finish();
}

fn memtable_read(c: &mut Criterion) {
    let keys: Vec<Vec<u8>> = (0..NUM_ENTRIES).map(make_key).collect();
    let values: Vec<Vec<u8>> = (0..NUM_ENTRIES).map(make_value).collect();

    // Pre-populate the memtable
    let memtable = Memtable::new(u64::MAX);
    for (i, (key, value)) in keys.iter().zip(values.iter()).enumerate() {
        memtable.put(key.clone(), i as u64, Some(value.clone()));
    }

    let mut group = c.benchmark_group("memtable_read");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));

    group.bench_function("read_100k_latest", |b| {
        b.iter(|| {
            for key in &keys {
                let result = memtable.get(black_box(key));
                black_box(result);
            }
        });
    });

    group.finish();
}

fn memtable_concurrent_insert(c: &mut Criterion) {
    let num_threads = 8usize;
    let entries_per_thread = NUM_ENTRIES / num_threads;

    // Pre-generate keys and values for each thread
    let thread_data: Vec<Vec<(Vec<u8>, Vec<u8>)>> = (0..num_threads)
        .map(|t| {
            let base = t * entries_per_thread;
            (0..entries_per_thread)
                .map(|i| {
                    let idx = base + i;
                    (make_key(idx), make_value(idx))
                })
                .collect()
        })
        .collect();

    let mut group = c.benchmark_group("memtable_concurrent");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));

    group.bench_function("concurrent_insert_8_threads", |b| {
        b.iter(|| {
            let memtable = Arc::new(Memtable::new(u64::MAX));
            let mut handles = Vec::with_capacity(num_threads);

            for (t, data) in thread_data.iter().enumerate() {
                let mt = Arc::clone(&memtable);
                let data = data.clone();
                handles.push(std::thread::spawn(move || {
                    for (i, (key, value)) in data.into_iter().enumerate() {
                        mt.put(key, (t * entries_per_thread + i) as u64, Some(value));
                    }
                }));
            }

            for handle in handles {
                handle.join().unwrap();
            }

            black_box(memtable.size());
        });
    });

    group.finish();
}

fn memtable_seal_swap(c: &mut Criterion) {
    let keys: Vec<Vec<u8>> = (0..1000).map(make_key).collect();
    let values: Vec<Vec<u8>> = (0..1000).map(make_value).collect();

    c.bench_function("memtable_seal_latency", |b| {
        b.iter_with_setup(
            || {
                // Setup: create and populate a memtable
                let memtable = Memtable::new(u64::MAX);
                for (i, (key, value)) in keys.iter().zip(values.iter()).enumerate() {
                    memtable.put(key.clone(), i as u64, Some(value.clone()));
                }
                memtable
            },
            |memtable| {
                // Benchmark: seal the memtable
                memtable.seal();
                black_box(memtable.is_sealed());
            },
        );
    });
}

criterion_group!(
    benches,
    memtable_insert,
    memtable_read,
    memtable_concurrent_insert,
    memtable_seal_swap,
);
criterion_main!(benches);
