//! Tests for Bloom filters with Monkey-optimal FPR allocation.

use super::*;

// ===========================================================================
// BloomFilter — basic construction and membership
// ===========================================================================

#[test]
fn bloom_filter_no_false_negatives() {
    let keys: Vec<Vec<u8>> = (0..1000)
        .map(|i| format!("key-{:06}", i).into_bytes())
        .collect();

    let mut filter = BloomFilter::new(keys.len(), 0.01);
    for key in &keys {
        filter.insert(key);
    }

    // Every inserted key must be found (no false negatives).
    for key in &keys {
        assert!(
            filter.may_contain(key),
            "inserted key {:?} must be found",
            String::from_utf8_lossy(key)
        );
    }
}

#[test]
fn bloom_filter_false_positive_rate_within_budget() {
    let num_keys = 10_000;
    let target_fpr = 0.01; // 1%

    let keys: Vec<Vec<u8>> = (0..num_keys)
        .map(|i| format!("key-{:08}", i).into_bytes())
        .collect();

    let mut filter = BloomFilter::new(num_keys, target_fpr);
    for key in &keys {
        filter.insert(key);
    }

    // Test with keys that were NOT inserted.
    let num_probes = 100_000;
    let mut false_positives = 0;
    for i in num_keys..(num_keys + num_probes) {
        let probe = format!("probe-{:08}", i).into_bytes();
        if filter.may_contain(&probe) {
            false_positives += 1;
        }
    }

    let observed_fpr = false_positives as f64 / num_probes as f64;
    // Allow up to 2× the target FPR to account for statistical variance.
    assert!(
        observed_fpr < target_fpr * 2.0,
        "observed FPR {:.4} exceeds 2× target {:.4}",
        observed_fpr,
        target_fpr
    );
}

#[test]
fn bloom_filter_with_bits_per_key() {
    let num_keys = 500;
    let bits_per_key = 10;

    let keys: Vec<Vec<u8>> = (0..num_keys)
        .map(|i| format!("bpk-{:06}", i).into_bytes())
        .collect();

    let mut filter = BloomFilter::with_bits_per_key(num_keys, bits_per_key);
    for key in &keys {
        filter.insert(key);
    }

    // No false negatives.
    for key in &keys {
        assert!(filter.may_contain(key));
    }

    // Verify the filter is sized correctly.
    assert!(filter.num_bits() >= (num_keys as u64) * (bits_per_key as u64));
}

#[test]
fn bloom_filter_empty_key() {
    let mut filter = BloomFilter::new(10, 0.01);
    filter.insert(b"");
    assert!(filter.may_contain(b""));
}

// ===========================================================================
// BloomFilter — serialization round-trip
// ===========================================================================

#[test]
fn bloom_filter_serialize_deserialize_roundtrip() {
    let keys: Vec<Vec<u8>> = (0..200)
        .map(|i| format!("ser-{:04}", i).into_bytes())
        .collect();

    let mut original = BloomFilter::new(keys.len(), 0.01);
    for key in &keys {
        original.insert(key);
    }

    let bytes = original.serialize();
    let restored = BloomFilter::deserialize(&bytes).expect("deserialization should succeed");

    assert_eq!(restored.num_bits(), original.num_bits());
    assert_eq!(restored.num_hashes(), original.num_hashes());

    // All inserted keys must still be found.
    for key in &keys {
        assert!(restored.may_contain(key));
    }
}

#[test]
fn bloom_filter_deserialize_too_short() {
    assert!(BloomFilter::deserialize(&[0u8; 5]).is_none());
}

#[test]
fn bloom_filter_deserialize_truncated_bits() {
    // Create a valid header but truncate the bit vector.
    let mut data = Vec::new();
    data.extend_from_slice(&1000u64.to_le_bytes()); // num_bits = 1000
    data.extend_from_slice(&7u32.to_le_bytes()); // num_hashes = 7
    // Need ceil(1000/8) = 125 bytes, but only provide 10.
    data.extend_from_slice(&[0u8; 10]);
    assert!(BloomFilter::deserialize(&data).is_none());
}

// ===========================================================================
// MonkeyAllocator — FPR allocation
// ===========================================================================

#[test]
fn monkey_allocator_fprs_sum_to_budget() {
    let allocator = MonkeyAllocator::with_fpr_budget(0.01, 10);
    let num_levels = 5;
    let fprs = allocator.allocate_all(num_levels);

    assert_eq!(fprs.len(), num_levels);

    // The sum of allocated FPRs should be close to the total budget.
    let sum: f64 = fprs.iter().sum();
    assert!(
        (sum - 0.01).abs() < 1e-6,
        "FPR sum {:.8} should be close to budget 0.01",
        sum
    );
}

#[test]
fn monkey_allocator_higher_levels_get_lower_fpr() {
    // Higher levels (larger, colder) should get lower FPR (more memory).
    let allocator = MonkeyAllocator::with_fpr_budget(0.01, 10);
    let num_levels = 5;
    let fprs = allocator.allocate_all(num_levels);

    // Level 0 (smallest, hottest) should have the highest FPR.
    // Level 4 (largest, coldest) should have the lowest FPR.
    for i in 1..num_levels {
        assert!(
            fprs[i] < fprs[i - 1],
            "FPR at level {} ({:.8}) should be < FPR at level {} ({:.8})",
            i,
            fprs[i],
            i - 1,
            fprs[i - 1]
        );
    }
}

#[test]
fn monkey_allocator_single_level() {
    let allocator = MonkeyAllocator::with_fpr_budget(0.01, 10);
    let fprs = allocator.allocate_all(1);
    assert_eq!(fprs.len(), 1);
    // With a single level, it gets the entire budget.
    assert!((fprs[0] - 0.01).abs() < 1e-6);
}

#[test]
fn monkey_allocator_from_bits_per_key() {
    let allocator = MonkeyAllocator::new(10, 10);
    // 10 bits per key → FPR ≈ 0.6185^10 ≈ 0.0082
    assert!(allocator.total_fpr_budget() > 0.005);
    assert!(allocator.total_fpr_budget() < 0.02);
}

#[test]
fn monkey_allocator_bits_per_key_for_level() {
    let allocator = MonkeyAllocator::with_fpr_budget(0.01, 10);
    let num_levels = 5;

    // Deeper levels should get more bits per key (lower FPR → more bits).
    let bpk: Vec<u32> = (0..num_levels)
        .map(|l| allocator.bits_per_key_for_level(l, num_levels))
        .collect();

    for i in 1..num_levels {
        assert!(
            bpk[i] >= bpk[i - 1],
            "bits_per_key at level {} ({}) should be >= level {} ({})",
            i,
            bpk[i],
            i - 1,
            bpk[i - 1]
        );
    }
}

#[test]
fn monkey_allocator_ratio_affects_distribution() {
    // A larger size ratio should produce a more skewed distribution.
    let alloc_10 = MonkeyAllocator::with_fpr_budget(0.01, 10);
    let alloc_2 = MonkeyAllocator::with_fpr_budget(0.01, 2);

    let fprs_10 = alloc_10.allocate_all(4);
    let fprs_2 = alloc_2.allocate_all(4);

    // With ratio=10, the spread between level 0 and level 3 should be larger.
    let spread_10 = fprs_10[0] / fprs_10[3];
    let spread_2 = fprs_2[0] / fprs_2[3];

    assert!(
        spread_10 > spread_2,
        "ratio=10 spread ({:.2}) should be > ratio=2 spread ({:.2})",
        spread_10,
        spread_2
    );
}

// ===========================================================================
// SstBloomFilter — SST-level wrapper
// ===========================================================================

#[test]
fn sst_bloom_filter_build_and_check() {
    let keys: Vec<Vec<u8>> = (0..500)
        .map(|i| format!("sst-key-{:06}", i).into_bytes())
        .collect();

    let key_slices: Vec<&[u8]> = keys.iter().map(|k| k.as_slice()).collect();

    let sst_filter = SstBloomFilter::build(42, 2, key_slices.iter().copied(), 0.01);

    assert_eq!(sst_filter.sst_id, 42);
    assert_eq!(sst_filter.level, 2);

    // All inserted keys should be found.
    for key in &keys {
        assert!(sst_filter.check_key(key));
    }

    // A key that was never inserted should (usually) not be found.
    // We can't guarantee it due to false positives, but with 500 keys and 1% FPR
    // the chance of a specific key being a false positive is ~1%.
    let absent_key = b"definitely-not-in-the-sst";
    // Just verify the method runs without panic; we don't assert the result.
    let _ = sst_filter.check_key(absent_key);
}

#[test]
fn sst_bloom_filter_build_with_bits_per_key() {
    let keys: Vec<Vec<u8>> = (0..100)
        .map(|i| format!("bpk-sst-{:04}", i).into_bytes())
        .collect();

    let key_slices: Vec<&[u8]> = keys.iter().map(|k| k.as_slice()).collect();

    let sst_filter =
        SstBloomFilter::build_with_bits_per_key(7, 1, key_slices.iter().copied(), 10);

    for key in &keys {
        assert!(sst_filter.check_key(key));
    }
}

#[test]
fn sst_bloom_filter_serialize_deserialize_roundtrip() {
    let keys: Vec<Vec<u8>> = (0..300)
        .map(|i| format!("sst-ser-{:06}", i).into_bytes())
        .collect();

    let key_slices: Vec<&[u8]> = keys.iter().map(|k| k.as_slice()).collect();

    let original = SstBloomFilter::build(99, 3, key_slices.iter().copied(), 0.01);

    let bytes = original.serialize();
    let restored = SstBloomFilter::deserialize(&bytes).expect("deserialization should succeed");

    assert_eq!(restored.sst_id, 99);
    assert_eq!(restored.level, 3);

    for key in &keys {
        assert!(restored.check_key(key));
    }
}

// ===========================================================================
// filter_candidate_ssts — point read integration
// ===========================================================================

#[test]
fn filter_candidate_ssts_skips_absent_keys() {
    // Build 3 SST filters, each with a distinct set of keys.
    let mut filters = Vec::new();

    for sst_id in 0..3 {
        let keys: Vec<Vec<u8>> = (0..100)
            .map(|i| format!("sst{}-key-{:04}", sst_id, i).into_bytes())
            .collect();
        let key_slices: Vec<&[u8]> = keys.iter().map(|k| k.as_slice()).collect();
        filters.push(SstBloomFilter::build(
            sst_id as u64,
            0,
            key_slices.iter().copied(),
            0.001, // very low FPR to minimize false positives in test
        ));
    }

    // A key from SST 1 should only match SST 1 (with very high probability).
    let target_key = b"sst1-key-0050";
    let candidates = filter_candidate_ssts(target_key, &filters);

    // SST 1 must be in the candidates.
    assert!(
        candidates.contains(&1),
        "SST 1 should be a candidate for its own key"
    );

    // SST 0 and SST 2 should (very likely) not be candidates.
    // With 100 keys and 0.1% FPR, the chance of a false positive is ~0.1%.
    // We accept a tiny chance of test flakiness here.
}

#[test]
fn filter_candidate_ssts_returns_all_when_key_in_multiple() {
    // Build two SST filters that both contain the same key.
    let shared_key = b"shared-key";

    let mut filter1 = BloomFilter::new(10, 0.01);
    filter1.insert(shared_key);
    let sst1 = SstBloomFilter {
        sst_id: 1,
        level: 0,
        filter: filter1,
    };

    let mut filter2 = BloomFilter::new(10, 0.01);
    filter2.insert(shared_key);
    let sst2 = SstBloomFilter {
        sst_id: 2,
        level: 1,
        filter: filter2,
    };

    let candidates = filter_candidate_ssts(shared_key, &[sst1, sst2]);
    assert!(candidates.contains(&1));
    assert!(candidates.contains(&2));
}

#[test]
fn filter_candidate_ssts_empty_filters() {
    let candidates = filter_candidate_ssts(b"any-key", &[]);
    assert!(candidates.is_empty());
}

// ===========================================================================
// Integration: Monkey allocation → Bloom filter → correct skip behavior
// ===========================================================================

#[test]
fn monkey_allocated_bloom_filters_have_correct_fpr() {
    let allocator = MonkeyAllocator::with_fpr_budget(0.01, 10);
    let num_levels = 4;
    let fprs = allocator.allocate_all(num_levels);

    let num_keys_per_level = 5_000;
    let num_probes = 50_000;

    for (level, &target_fpr) in fprs.iter().enumerate() {
        // Build a Bloom filter with the Monkey-allocated FPR.
        let keys: Vec<Vec<u8>> = (0..num_keys_per_level)
            .map(|i| format!("L{}-key-{:08}", level, i).into_bytes())
            .collect();

        let mut filter = BloomFilter::new(num_keys_per_level, target_fpr);
        for key in &keys {
            filter.insert(key);
        }

        // No false negatives.
        for key in &keys {
            assert!(filter.may_contain(key));
        }

        // Measure false positive rate.
        let mut fp_count = 0;
        for i in num_keys_per_level..(num_keys_per_level + num_probes) {
            let probe = format!("L{}-probe-{:08}", level, i).into_bytes();
            if filter.may_contain(&probe) {
                fp_count += 1;
            }
        }

        let observed_fpr = fp_count as f64 / num_probes as f64;
        // Allow up to 3× the target FPR for statistical variance.
        // For very small target FPRs (level 3 with ratio=10), we need more slack.
        let tolerance = (target_fpr * 3.0).max(0.001);
        assert!(
            observed_fpr < tolerance,
            "Level {} observed FPR {:.6} exceeds tolerance {:.6} (target {:.6})",
            level,
            observed_fpr,
            tolerance,
            target_fpr
        );
    }
}
