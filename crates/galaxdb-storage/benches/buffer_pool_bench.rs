//! Buffer pool benchmarks.
//!
//! Measures HotSet insert + lookup throughput, ScanBuffer eviction overhead,
//! and mixed OLTP+OLAP concurrent access patterns.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use galaxdb_storage::buffer_pool::{AccessType, BufferPool, CachedBlock};
use std::time::Duration;

const POOL_CAPACITY: usize = 10_000;
const LOOKUP_COUNT: usize = 100_000;

/// Create a CachedBlock with a realistic 4KB payload.
fn make_block(id: u64) -> CachedBlock {
    let mut data = vec![0u8; 4096];
    let seed = id.to_le_bytes();
    for (i, byte) in data.iter_mut().enumerate() {
        *byte = seed[i % 8] ^ (i as u8);
    }
    CachedBlock {
        block_id: id,
        data,
    }
}

fn hotset_insert_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_pool_hotset");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("insert_and_lookup_100k", |b| {
        b.iter(|| {
            let mut pool = BufferPool::new(POOL_CAPACITY, 1);

            // Fill the HotSet (70% of 10K = 7K slots)
            for i in 0..7_000u64 {
                pool.insert(i, make_block(i), AccessType::PointLookup, 0);
            }

            // Lookup 100K times, cycling through existing blocks
            for i in 0..LOOKUP_COUNT {
                let block_id = (i % 7_000) as u64;
                let result = pool.get_for_point_lookup(block_id, 0);
                black_box(result);
            }
        });
    });

    group.finish();
}

fn scanbuffer_eviction(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_pool_scanbuffer");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("insert_beyond_capacity_eviction", |b| {
        b.iter(|| {
            let mut pool = BufferPool::new(POOL_CAPACITY, 1);

            // ScanBuffer capacity is 30% of 10K = 3K slots.
            // Insert 10K blocks to force heavy eviction.
            for i in 0..10_000u64 {
                pool.insert(i, make_block(i), AccessType::SequentialScan, 0);
            }

            black_box(pool.scan_buffer_len(0));
        });
    });

    group.finish();
}

fn mixed_oltp_olap(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_pool_mixed");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));

    group.bench_function("concurrent_point_lookup_and_scan", |b| {
        b.iter(|| {
            let mut pool = BufferPool::new(POOL_CAPACITY, 1);

            // Phase 1: Populate HotSet with OLTP blocks (IDs 0..5000)
            for i in 0..5_000u64 {
                pool.insert(i, make_block(i), AccessType::PointLookup, 0);
            }

            // Phase 2: Simulate concurrent OLAP scan that inserts blocks
            // into ScanBuffer (IDs 100_000..110_000) — these should NOT
            // evict HotSet blocks.
            for i in 100_000..110_000u64 {
                pool.insert(i, make_block(i), AccessType::SequentialScan, 0);
            }

            // Phase 3: Verify HotSet blocks survived the scan storm.
            // Do 50K point lookups on the original OLTP blocks.
            let mut hits = 0u64;
            for i in 0..50_000u64 {
                let block_id = i % 5_000;
                if pool.get_for_point_lookup(block_id, 0).is_some() {
                    hits += 1;
                }
            }

            // Also do scan lookups to exercise the ScanBuffer path
            for i in 100_000..105_000u64 {
                let result = pool.get_for_scan(i, 0);
                black_box(result);
            }

            black_box(hits);
            black_box(pool.hot_set_len(0));
            black_box(pool.scan_buffer_len(0));
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    hotset_insert_lookup,
    scanbuffer_eviction,
    mixed_oltp_olap,
);
criterion_main!(benches);
