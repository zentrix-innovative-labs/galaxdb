//! Bloom filter benchmarks.
//!
//! Measures filter build time, lookup throughput for existing and non-existing
//! keys, Monkey FPR allocation computation, and FPR comparison between Monkey
//! and fixed allocation strategies.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use galaxdb_storage::bloom::{BloomFilter, MonkeyAllocator};

const NUM_KEYS: usize = 100_000;

/// Generate a deterministic 32-byte key from an index.
fn make_key(i: usize) -> Vec<u8> {
    let mut key = Vec::with_capacity(32);
    key.extend_from_slice(b"bloom_bench_key_");
    key.extend_from_slice(&(i as u64).to_be_bytes());
    key.extend_from_slice(&((i as u64).wrapping_mul(0x517cc1b727220a95)).to_be_bytes());
    key
}

/// Generate a key that was NOT inserted into the filter.
fn make_absent_key(i: usize) -> Vec<u8> {
    let mut key = Vec::with_capacity(32);
    key.extend_from_slice(b"absent_key______");
    key.extend_from_slice(&(i as u64).to_be_bytes());
    key.extend_from_slice(&((i as u64).wrapping_mul(0x9e3779b97f4a7c15)).to_be_bytes());
    key
}

fn bloom_build(c: &mut Criterion) {
    let keys: Vec<Vec<u8>> = (0..NUM_KEYS).map(make_key).collect();

    c.bench_function("bloom_filter_build_100k", |b| {
        b.iter(|| {
            let mut filter = BloomFilter::new(NUM_KEYS, 0.01);
            for key in &keys {
                filter.insert(black_box(key));
            }
            black_box(filter.num_bits());
        });
    });
}

fn bloom_lookup_existing(c: &mut Criterion) {
    let keys: Vec<Vec<u8>> = (0..NUM_KEYS).map(make_key).collect();
    let mut filter = BloomFilter::new(NUM_KEYS, 0.01);
    for key in &keys {
        filter.insert(key);
    }

    c.bench_function("bloom_filter_lookup_existing", |b| {
        b.iter(|| {
            // Lookup a single existing key (cycle through to avoid branch prediction)
            let key = &keys[black_box(42_000)];
            let found = filter.may_contain(black_box(key));
            black_box(found);
        });
    });
}

fn bloom_lookup_nonexisting(c: &mut Criterion) {
    let keys: Vec<Vec<u8>> = (0..NUM_KEYS).map(make_key).collect();
    let mut filter = BloomFilter::new(NUM_KEYS, 0.01);
    for key in &keys {
        filter.insert(key);
    }

    let absent = make_absent_key(999_999);

    c.bench_function("bloom_filter_lookup_nonexisting", |b| {
        b.iter(|| {
            let found = filter.may_contain(black_box(&absent));
            black_box(found);
        });
    });
}

fn monkey_fpr_computation(c: &mut Criterion) {
    c.bench_function("monkey_allocate_fpr_5_levels", |b| {
        b.iter(|| {
            let allocator = MonkeyAllocator::new(black_box(10), black_box(10));
            let fprs = allocator.allocate_all(black_box(5));
            black_box(fprs);
        });
    });
}

fn bloom_fpr_comparison(c: &mut Criterion) {
    // Compare actual FPR between Monkey-optimal and fixed 10-bit allocation.
    // This benchmark measures the time to build filters and compute empirical FPR.
    let num_levels = 5usize;
    let keys_per_level = 10_000usize;
    let test_keys = 50_000usize;

    // Pre-generate keys for each level
    let level_keys: Vec<Vec<Vec<u8>>> = (0..num_levels)
        .map(|level| {
            (0..keys_per_level)
                .map(|i| {
                    let mut key = Vec::with_capacity(32);
                    key.extend_from_slice(b"lvl_");
                    key.extend_from_slice(&(level as u32).to_be_bytes());
                    key.extend_from_slice(b"_key_");
                    key.extend_from_slice(&(i as u64).to_be_bytes());
                    // Pad to 32 bytes
                    key.resize(32, 0);
                    key
                })
                .collect()
        })
        .collect();

    // Pre-generate test keys (absent from all levels)
    let absent_keys: Vec<Vec<u8>> = (0..test_keys)
        .map(|i| {
            let mut key = Vec::with_capacity(32);
            key.extend_from_slice(b"test_absent_____");
            key.extend_from_slice(&(i as u64).to_be_bytes());
            key.resize(32, 0);
            key
        })
        .collect();

    c.bench_function("bloom_fpr_monkey_vs_fixed_comparison", |b| {
        b.iter(|| {
            let allocator = MonkeyAllocator::new(10, 10);
            let monkey_fprs = allocator.allocate_all(num_levels);

            let mut monkey_total_fp = 0usize;
            let mut fixed_total_fp = 0usize;

            for level in 0..num_levels {
                // Build Monkey-optimal filter
                let mut monkey_filter = BloomFilter::new(keys_per_level, monkey_fprs[level]);
                for key in &level_keys[level] {
                    monkey_filter.insert(key);
                }

                // Build fixed 10-bit filter
                let mut fixed_filter = BloomFilter::with_bits_per_key(keys_per_level, 10);
                for key in &level_keys[level] {
                    fixed_filter.insert(key);
                }

                // Count false positives on absent keys
                for key in &absent_keys {
                    if monkey_filter.may_contain(key) {
                        monkey_total_fp += 1;
                    }
                    if fixed_filter.may_contain(key) {
                        fixed_total_fp += 1;
                    }
                }
            }

            black_box(monkey_total_fp);
            black_box(fixed_total_fp);
        });
    });
}

criterion_group!(
    benches,
    bloom_build,
    bloom_lookup_existing,
    bloom_lookup_nonexisting,
    monkey_fpr_computation,
    bloom_fpr_comparison,
);
criterion_main!(benches);
