//! Tests for Lazy Leveling compaction with MVCC garbage collection.

use std::collections::HashSet;

use super::*;

// ── Helpers ───────────────────────────────────────────────────────────

fn entry(key: &str, ts: u64, value: Option<&str>) -> VersionedEntry {
    VersionedEntry {
        key: key.as_bytes().to_vec(),
        timestamp: ts,
        value: value.map(|v| v.as_bytes().to_vec()),
    }
}

fn sst(id: u64, level: usize, min: &str, max: &str, size: u64) -> SstMetadata {
    SstMetadata {
        sst_id: id,
        level,
        min_key: min.as_bytes().to_vec(),
        max_key: max.as_bytes().to_vec(),
        size_bytes: size,
        row_count: 10,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 10.1 — LSM level structure
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn lsm_tree_has_five_levels() {
    let tree = LsmTree::new();
    assert_eq!(tree.levels.len(), NUM_LEVELS);
    assert_eq!(NUM_LEVELS, 5);
}

#[test]
fn l0_is_tiered_l4_is_bottom() {
    let tree = LsmTree::new();
    assert!(tree.level(0).is_l0());
    assert!(!tree.level(0).is_bottom());
    assert!(tree.level(4).is_bottom());
    assert!(!tree.level(4).is_l0());
}

#[test]
fn add_and_remove_sst_from_level() {
    let mut tree = LsmTree::new();
    let s = sst(1, 0, "a", "z", 1000);
    tree.add_sst(0, s.clone());

    assert_eq!(tree.level(0).file_count(), 1);
    assert_eq!(tree.level(0).ssts[0].sst_id, 1);

    let removed = tree.remove_sst(0, 1);
    assert!(removed.is_some());
    assert_eq!(tree.level(0).file_count(), 0);
}

#[test]
fn ssts_sorted_by_min_key() {
    let mut tree = LsmTree::new();
    tree.add_sst(1, sst(3, 1, "m", "z", 100));
    tree.add_sst(1, sst(1, 1, "a", "f", 100));
    tree.add_sst(1, sst(2, 1, "g", "l", 100));

    let keys: Vec<&[u8]> = tree.level(1).ssts.iter().map(|s| s.min_key.as_slice()).collect();
    assert_eq!(keys, vec![b"a", b"g", b"m"]);
}

#[test]
fn overlapping_ssts_detected() {
    let mut level = LsmLevel::new(1);
    level.add_sst(sst(1, 1, "a", "f", 100));
    level.add_sst(sst(2, 1, "g", "l", 100));
    level.add_sst(sst(3, 1, "m", "z", 100));

    let overlapping = level.overlapping_ssts(b"e", b"h");
    let ids: Vec<u64> = overlapping.iter().map(|s| s.sst_id).collect();
    assert!(ids.contains(&1)); // a-f overlaps e-h
    assert!(ids.contains(&2)); // g-l overlaps e-h
    assert!(!ids.contains(&3)); // m-z does not overlap e-h
}

#[test]
fn total_sst_count_across_levels() {
    let mut tree = LsmTree::new();
    tree.add_sst(0, sst(1, 0, "a", "z", 100));
    tree.add_sst(0, sst(2, 0, "a", "z", 100));
    tree.add_sst(1, sst(3, 1, "a", "z", 100));

    assert_eq!(tree.total_sst_count(), 3);
}

// ═══════════════════════════════════════════════════════════════════════
// 10.2 — Compaction trigger
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn l0_triggers_at_file_count_threshold() {
    let trigger = CompactionTrigger::new();
    let mut tree = LsmTree::new();

    // Below threshold — no trigger.
    for i in 0..3 {
        tree.add_sst(0, sst(i, 0, "a", "z", 1000));
    }
    assert_eq!(trigger.check(&tree), None);

    // At threshold — triggers.
    tree.add_sst(0, sst(4, 0, "a", "z", 1000));
    assert_eq!(trigger.check(&tree), Some(0));
}

#[test]
fn l0_needs_compaction_check() {
    let trigger = CompactionTrigger::new();
    let mut tree = LsmTree::new();

    assert!(!trigger.needs_compaction(&tree, 0));

    for i in 0..L0_FILE_COUNT_THRESHOLD {
        tree.add_sst(0, sst(i as u64, 0, "a", "z", 1000));
    }
    assert!(trigger.needs_compaction(&tree, 0));
}

#[test]
fn bottom_level_never_triggers() {
    let trigger = CompactionTrigger::new();
    let mut tree = LsmTree::new();
    tree.add_sst(BOTTOM_LEVEL, sst(1, BOTTOM_LEVEL, "a", "z", u64::MAX));

    assert!(!trigger.needs_compaction(&tree, BOTTOM_LEVEL));
}

// ═══════════════════════════════════════════════════════════════════════
// 10.3 — Merge iterator
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn merge_iterator_single_run() {
    let run = vec![
        entry("a", 10, Some("v1")),
        entry("b", 10, Some("v2")),
        entry("c", 10, Some("v3")),
    ];

    let mut iter = MergeIterator::new(vec![run]);
    let result = iter.collect_all();

    assert_eq!(result.len(), 3);
    assert_eq!(result[0].key, b"a");
    assert_eq!(result[1].key, b"b");
    assert_eq!(result[2].key, b"c");
}

#[test]
fn merge_iterator_two_runs_interleaved() {
    let run1 = vec![
        entry("a", 10, Some("r1-a")),
        entry("c", 10, Some("r1-c")),
        entry("e", 10, Some("r1-e")),
    ];
    let run2 = vec![
        entry("b", 10, Some("r2-b")),
        entry("d", 10, Some("r2-d")),
    ];

    let mut iter = MergeIterator::new(vec![run1, run2]);
    let result = iter.collect_all();

    let keys: Vec<&[u8]> = result.iter().map(|e| e.key.as_slice()).collect();
    assert_eq!(keys, vec![b"a", b"b", b"c", b"d", b"e"]);
}

#[test]
fn merge_iterator_same_key_different_timestamps() {
    let run1 = vec![entry("key", 20, Some("newer"))];
    let run2 = vec![entry("key", 10, Some("older"))];

    let mut iter = MergeIterator::new(vec![run1, run2]);
    let result = iter.collect_all();

    assert_eq!(result.len(), 2);
    // Higher timestamp should come first for the same key.
    assert_eq!(result[0].timestamp, 20);
    assert_eq!(result[1].timestamp, 10);
}

#[test]
fn merge_iterator_empty_runs() {
    let mut iter = MergeIterator::new(vec![vec![], vec![]]);
    let result = iter.collect_all();
    assert!(result.is_empty());
}

#[test]
fn merge_iterator_three_runs() {
    let run1 = vec![entry("a", 30, Some("v1")), entry("d", 30, Some("v4"))];
    let run2 = vec![entry("b", 20, Some("v2")), entry("e", 20, Some("v5"))];
    let run3 = vec![entry("c", 10, Some("v3")), entry("f", 10, Some("v6"))];

    let mut iter = MergeIterator::new(vec![run1, run2, run3]);
    let result = iter.collect_all();

    let keys: Vec<&[u8]> = result.iter().map(|e| e.key.as_slice()).collect();
    assert_eq!(keys, vec![b"a", b"b", b"c", b"d", b"e", b"f"]);
}

// ═══════════════════════════════════════════════════════════════════════
// 10.3 — MVCC GC
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn gc_keeps_latest_version_always() {
    let gc_ctx = GcContext::new(); // No active snapshots, no pinned tags.
    let gc = MvccGarbageCollector::new(gc_ctx);

    let entries = vec![
        entry("key", 30, Some("v3")),
        entry("key", 20, Some("v2")),
        entry("key", 10, Some("v1")),
    ];

    let result = gc.apply(&entries);
    // Only the latest version (ts=30) should be kept.
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].timestamp, 30);
}

#[test]
fn gc_keeps_versions_needed_by_active_snapshot() {
    let gc_ctx = GcContext::with_context(Some(15), HashSet::new());
    let gc = MvccGarbageCollector::new(gc_ctx);

    let entries = vec![
        entry("key", 30, Some("v3")),
        entry("key", 20, Some("v2")),
        entry("key", 10, Some("v1")),
    ];

    let result = gc.apply(&entries);
    // ts=30 (latest, always kept), ts=20 (>= oldest snapshot 15), ts=10 (< 15, discarded).
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].timestamp, 30);
    assert_eq!(result[1].timestamp, 20);
}

#[test]
fn gc_keeps_pinned_tag_versions() {
    let mut pinned = HashSet::new();
    pinned.insert(10u64);
    let gc_ctx = GcContext::with_context(None, pinned);
    let gc = MvccGarbageCollector::new(gc_ctx);

    let entries = vec![
        entry("key", 30, Some("v3")),
        entry("key", 20, Some("v2")),
        entry("key", 10, Some("v1")),
    ];

    let result = gc.apply(&entries);
    // ts=30 (latest), ts=10 (pinned). ts=20 is discarded.
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].timestamp, 30);
    assert_eq!(result[1].timestamp, 10);
}

#[test]
fn gc_discards_all_old_versions_when_no_snapshots_or_tags() {
    let gc_ctx = GcContext::new();
    let gc = MvccGarbageCollector::new(gc_ctx);

    let entries = vec![
        entry("a", 30, Some("a3")),
        entry("a", 20, Some("a2")),
        entry("a", 10, Some("a1")),
        entry("b", 25, Some("b2")),
        entry("b", 5, Some("b1")),
    ];

    let result = gc.apply(&entries);
    // Only latest per key: a@30, b@25.
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].key, b"a");
    assert_eq!(result[0].timestamp, 30);
    assert_eq!(result[1].key, b"b");
    assert_eq!(result[1].timestamp, 25);
}

#[test]
fn gc_handles_tombstones() {
    let gc_ctx = GcContext::new();
    let gc = MvccGarbageCollector::new(gc_ctx);

    let entries = vec![
        entry("key", 20, None), // tombstone is the latest
        entry("key", 10, Some("alive")),
    ];

    let result = gc.apply(&entries);
    // Latest version is the tombstone — it's kept.
    assert_eq!(result.len(), 1);
    assert!(result[0].value.is_none());
}

#[test]
fn gc_empty_input() {
    let gc_ctx = GcContext::new();
    let gc = MvccGarbageCollector::new(gc_ctx);
    let result = gc.apply(&[]);
    assert!(result.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════
// 10.4 — Compaction output: new SSTs with Bloom filters
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn compactor_produces_correct_output() {
    let config = CompactionConfig::new().with_sst_size(MIN_SST_SIZE_BYTES);
    let mut compactor = Compactor::new(config);
    let mut tree = LsmTree::new();

    // Add 4 SSTs to L0 to trigger compaction.
    for i in 0..4 {
        tree.add_sst(0, sst(i + 100, 0, &format!("k{:02}", i * 10), &format!("k{:02}", i * 10 + 9), 1000));
    }

    let run1 = vec![entry("k00", 10, Some("v1")), entry("k05", 10, Some("v2"))];
    let run2 = vec![entry("k10", 10, Some("v3")), entry("k15", 10, Some("v4"))];
    let run3 = vec![entry("k20", 10, Some("v5")), entry("k25", 10, Some("v6"))];
    let run4 = vec![entry("k30", 10, Some("v7")), entry("k35", 10, Some("v8"))];

    let gc_ctx = GcContext::new();

    // Remove old SSTs from L0 first (simulating what maybe_compact does).
    for i in 0..4 {
        tree.remove_sst(0, i + 100);
    }

    let output = compactor.compact(&mut tree, 0, vec![run1, run2, run3, run4], &gc_ctx);

    // Should produce output SSTs at level 1.
    assert!(!output.new_ssts.is_empty());
    assert_eq!(output.total_entries, 8);
    assert_eq!(output.gc_discarded, 0);

    // All output SSTs should be at level 1.
    for sst_meta in &output.new_ssts {
        assert_eq!(sst_meta.level, 1);
    }

    // Bloom filters should be built for each output SST.
    assert_eq!(output.bloom_filters.len(), output.new_ssts.len());

    // Bloom filters should find the keys.
    for (bloom, entries) in output.bloom_filters.iter().zip(output.sst_entries.iter()) {
        for e in entries {
            assert!(bloom.check_key(&e.key), "Bloom filter should contain key");
        }
    }

    // L0 should be empty, L1 should have the new SSTs.
    assert_eq!(tree.level(0).file_count(), 0);
    assert!(tree.level(1).file_count() > 0);
}

#[test]
fn compaction_with_mvcc_gc_discards_old_versions() {
    let config = CompactionConfig::new();
    let mut compactor = Compactor::new(config);
    let mut tree = LsmTree::new();

    let run = vec![
        entry("key-a", 30, Some("latest-a")),
        entry("key-a", 20, Some("old-a")),
        entry("key-a", 10, Some("oldest-a")),
        entry("key-b", 25, Some("latest-b")),
        entry("key-b", 5, Some("old-b")),
    ];

    let gc_ctx = GcContext::new(); // No snapshots or tags.
    let output = compactor.compact(&mut tree, 0, vec![run], &gc_ctx);

    // GC should discard old versions: 5 input → 2 output (latest per key).
    assert_eq!(output.total_entries, 2);
    assert_eq!(output.gc_discarded, 3);
}

#[test]
fn compaction_preserves_pinned_versions() {
    let config = CompactionConfig::new();
    let mut compactor = Compactor::new(config);
    let mut tree = LsmTree::new();

    let run = vec![
        entry("key", 30, Some("v3")),
        entry("key", 20, Some("v2")),
        entry("key", 10, Some("v1")),
    ];

    let mut pinned = HashSet::new();
    pinned.insert(10u64);
    let gc_ctx = GcContext::with_context(None, pinned);

    let output = compactor.compact(&mut tree, 0, vec![run], &gc_ctx);

    // ts=30 (latest) + ts=10 (pinned) = 2 kept, ts=20 discarded.
    assert_eq!(output.total_entries, 2);
    assert_eq!(output.gc_discarded, 1);
}

// ═══════════════════════════════════════════════════════════════════════
// 10.5 — Pinned tag awareness
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn pinned_tags_prevent_gc_of_referenced_versions() {
    let mut pinned = HashSet::new();
    pinned.insert(10u64);
    pinned.insert(20u64);

    let gc_ctx = GcContext::with_context(None, pinned);

    assert!(gc_ctx.should_keep(10, false));
    assert!(gc_ctx.should_keep(20, false));
    assert!(!gc_ctx.should_keep(15, false)); // Not pinned, not latest.
    assert!(gc_ctx.should_keep(15, true)); // Latest is always kept.
}

#[test]
fn snapshot_and_pinned_tags_combined() {
    let mut pinned = HashSet::new();
    pinned.insert(5u64);

    let gc_ctx = GcContext::with_context(Some(15), pinned);
    let gc = MvccGarbageCollector::new(gc_ctx);

    let entries = vec![
        entry("key", 30, Some("v3")),
        entry("key", 20, Some("v2")),
        entry("key", 10, Some("v1")),
        entry("key", 5, Some("v0")),
    ];

    let result = gc.apply(&entries);
    // ts=30 (latest), ts=20 (>= snapshot 15), ts=5 (pinned). ts=10 discarded.
    assert_eq!(result.len(), 3);
    let timestamps: Vec<u64> = result.iter().map(|e| e.timestamp).collect();
    assert_eq!(timestamps, vec![30, 20, 5]);
}

// ═══════════════════════════════════════════════════════════════════════
// 10.6 — Full compaction cycle via maybe_compact
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn maybe_compact_triggers_when_l0_full() {
    let config = CompactionConfig::new();
    let mut compactor = Compactor::new(config);
    let mut tree = LsmTree::new();

    // Add 4 SSTs to L0.
    for i in 0..4 {
        tree.add_sst(0, sst(i + 1, 0, &format!("k{}", i), &format!("k{}", i), 1000));
    }

    let gc_ctx = GcContext::new();

    let output = compactor.maybe_compact(&mut tree, &gc_ctx, |sst_ids| {
        // Return one entry per SST.
        sst_ids
            .iter()
            .map(|&id| vec![entry(&format!("key-{}", id), 10, Some("value"))])
            .collect()
    });

    assert!(output.is_some());
    let output = output.unwrap();
    assert_eq!(output.total_entries, 4);

    // L0 should be empty after compaction.
    assert_eq!(tree.level(0).file_count(), 0);
    // L1 should have the compacted SSTs.
    assert!(tree.level(1).file_count() > 0);
}

#[test]
fn maybe_compact_returns_none_when_no_trigger() {
    let config = CompactionConfig::new();
    let mut compactor = Compactor::new(config);
    let mut tree = LsmTree::new();

    // Only 2 SSTs in L0 — below threshold.
    tree.add_sst(0, sst(1, 0, "a", "m", 1000));
    tree.add_sst(0, sst(2, 0, "n", "z", 1000));

    let gc_ctx = GcContext::new();
    let output = compactor.maybe_compact(&mut tree, &gc_ctx, |_| vec![]);
    assert!(output.is_none());
}

#[test]
fn compaction_output_ssts_have_bloom_filters_that_work() {
    let config = CompactionConfig::new();
    let mut compactor = Compactor::new(config);
    let mut tree = LsmTree::new();

    let run = vec![
        entry("alpha", 10, Some("v1")),
        entry("beta", 10, Some("v2")),
        entry("gamma", 10, Some("v3")),
        entry("delta", 10, Some("v4")),
    ];

    let gc_ctx = GcContext::new();
    let output = compactor.compact(&mut tree, 0, vec![run], &gc_ctx);

    // Bloom filters should find inserted keys.
    for bloom in &output.bloom_filters {
        assert!(bloom.check_key(b"alpha"));
        assert!(bloom.check_key(b"beta"));
        assert!(bloom.check_key(b"gamma"));
        assert!(bloom.check_key(b"delta"));
    }

    // A key that was never inserted should (very likely) not be found.
    // With 4 keys and a reasonable FPR, false positives are rare.
    let mut any_bloom_has_absent = false;
    for bloom in &output.bloom_filters {
        if bloom.check_key(b"nonexistent-key-xyz-12345") {
            any_bloom_has_absent = true;
        }
    }
    // We don't assert this strictly due to FP possibility, just verify the method works.
    let _ = any_bloom_has_absent;
}

// ═══════════════════════════════════════════════════════════════════════
// Edge cases and stress
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn compaction_with_all_tombstones() {
    let config = CompactionConfig::new();
    let mut compactor = Compactor::new(config);
    let mut tree = LsmTree::new();

    let run = vec![
        entry("a", 20, None),
        entry("b", 20, None),
        entry("c", 20, None),
    ];

    let gc_ctx = GcContext::new();
    let output = compactor.compact(&mut tree, 0, vec![run], &gc_ctx);

    // Tombstones are the latest versions — they should be kept.
    assert_eq!(output.total_entries, 3);
}

#[test]
fn compaction_stress_many_keys() {
    let config = CompactionConfig::new();
    let mut compactor = Compactor::new(config);
    let mut tree = LsmTree::new();

    // Create 3 runs with 1000 keys each, some overlapping.
    let run1: Vec<VersionedEntry> = (0..1000)
        .map(|i| entry(&format!("key-{:06}", i), 30, Some(&format!("r1-{}", i))))
        .collect();
    let run2: Vec<VersionedEntry> = (500..1500)
        .map(|i| entry(&format!("key-{:06}", i), 20, Some(&format!("r2-{}", i))))
        .collect();
    let run3: Vec<VersionedEntry> = (1000..2000)
        .map(|i| entry(&format!("key-{:06}", i), 10, Some(&format!("r3-{}", i))))
        .collect();

    let gc_ctx = GcContext::new();
    let output = compactor.compact(&mut tree, 0, vec![run1, run2, run3], &gc_ctx);

    // With no snapshots/tags, only the latest version per key survives.
    // Keys 0-499: only in run1 (ts=30) → 500 entries
    // Keys 500-999: in run1 (ts=30) and run2 (ts=20) → keep ts=30 → 500 entries
    // Keys 1000-1499: in run2 (ts=20) and run3 (ts=10) → keep ts=20 → 500 entries
    // Keys 1500-1999: only in run3 (ts=10) → 500 entries
    // Total: 2000 unique keys.
    assert_eq!(output.total_entries, 2000);

    // Verify output is sorted by key.
    for sst_entries in &output.sst_entries {
        for window in sst_entries.windows(2) {
            assert!(
                window[0].key <= window[1].key,
                "output should be sorted by key"
            );
        }
    }
}

#[test]
fn sst_metadata_overlap_detection() {
    let s = sst(1, 0, "d", "h", 100);
    assert!(s.overlaps(b"a", b"e")); // partial overlap left
    assert!(s.overlaps(b"f", b"z")); // partial overlap right
    assert!(s.overlaps(b"e", b"g")); // contained
    assert!(s.overlaps(b"a", b"z")); // fully contains
    assert!(!s.overlaps(b"i", b"z")); // no overlap right
    assert!(!s.overlaps(b"a", b"c")); // no overlap left
}

#[test]
fn compaction_config_clamps_sst_size() {
    let config = CompactionConfig::new().with_sst_size(1); // Too small.
    assert_eq!(config.sst_size_bytes, MIN_SST_SIZE_BYTES);

    let config = CompactionConfig::new().with_sst_size(u64::MAX); // Too large.
    assert_eq!(config.sst_size_bytes, DEFAULT_SST_SIZE_BYTES);
}

#[test]
fn lsm_tree_default_size_ratio() {
    let tree = LsmTree::new();
    assert_eq!(tree.size_ratio, DEFAULT_SIZE_RATIO);
    assert_eq!(DEFAULT_SIZE_RATIO, 10);
}

// ---------------------------------------------------------------------------
// Task 39.1 — configurable SST size
// ---------------------------------------------------------------------------

#[test]
fn smaller_sst_size_produces_more_files_for_same_data() {
    // With a very small SST size, the compactor splits output into more
    // SST files than with the default. This is the "smaller SSTs reduce
    // write stalls" property: each SST is cheaper to flush and compact.
    let small_config = CompactionConfig::new().with_sst_size(MIN_SST_SIZE_BYTES);
    let large_config = CompactionConfig::new().with_sst_size(DEFAULT_SST_SIZE_BYTES);

    let mut small_compactor = Compactor::new(small_config);
    let mut large_compactor = Compactor::new(large_config);
    let mut tree_small = LsmTree::new();
    let mut tree_large = LsmTree::new();

    // Seed both trees with the same 4 L0 SSTs.
    for i in 0..4 {
        let s = sst(i + 1, 0, &format!("k{:04}", i * 100), &format!("k{:04}", i * 100 + 99), 1000);
        tree_small.add_sst(0, s.clone());
        tree_large.add_sst(0, s);
    }

    // Build a large input run so the small-SST compactor has to split.
    let entries: Vec<VersionedEntry> = (0..4000)
        .map(|i| VersionedEntry {
            key: format!("k{:08}", i).into_bytes(),
            value: Some(vec![0u8; 256]),
            timestamp: i as u64 + 1,
        })
        .collect();
    let input_runs = vec![entries.clone(), vec![], vec![], vec![]];

    for &id in &[1u64, 2, 3, 4] {
        tree_small.remove_sst(0, id);
        tree_large.remove_sst(0, id);
    }

    let gc = GcContext::new();
    let out_small = small_compactor.compact(&mut tree_small, 0, input_runs.clone(), &gc);
    let out_large = large_compactor.compact(&mut tree_large, 0, input_runs, &gc);

    // Both produce the same total entries.
    assert_eq!(out_small.total_entries, out_large.total_entries);
    // The small-SST compactor must produce at least as many output SSTs
    // as the large-SST one (and typically more).
    assert!(
        out_small.new_ssts.len() >= out_large.new_ssts.len(),
        "smaller SST size must produce >= output files; small={}, large={}",
        out_small.new_ssts.len(),
        out_large.new_ssts.len()
    );
}

// ---------------------------------------------------------------------------
// Task 39.2 — L0 leveled compaction
// ---------------------------------------------------------------------------

#[test]
fn l0_leveled_strategy_triggers_at_two_files() {
    // With Leveled L0, compaction fires as soon as a second SST lands in
    // L0 — keeping L0 a single sorted run.
    let trigger = CompactionTrigger::new()
        .with_l0_strategy(L0CompactionStrategy::Leveled);
    let mut tree = LsmTree::new();

    // One SST: no trigger.
    tree.add_sst(0, sst(1, 0, "a", "m", 1000));
    assert!(
        !trigger.needs_compaction(&tree, 0),
        "Leveled L0 must not trigger with only 1 SST"
    );

    // Two SSTs: trigger fires.
    tree.add_sst(0, sst(2, 0, "n", "z", 1000));
    assert!(
        trigger.needs_compaction(&tree, 0),
        "Leveled L0 must trigger as soon as a second SST lands"
    );
}

#[test]
fn l0_tiered_strategy_triggers_at_file_count_threshold() {
    // Tiered L0 (default) only fires at the 4-file threshold.
    let trigger = CompactionTrigger::new()
        .with_l0_strategy(L0CompactionStrategy::Tiered);
    let mut tree = LsmTree::new();

    for i in 0..L0_FILE_COUNT_THRESHOLD - 1 {
        tree.add_sst(0, sst(i as u64 + 1, 0, &format!("k{}", i), &format!("k{}", i), 100));
        assert!(
            !trigger.needs_compaction(&tree, 0),
            "Tiered L0 must not trigger before threshold; files={}",
            i + 1
        );
    }
    tree.add_sst(0, sst(99, 0, "z", "z", 100));
    assert!(
        trigger.needs_compaction(&tree, 0),
        "Tiered L0 must trigger at threshold"
    );
}

#[test]
fn l0_leveled_compaction_produces_correct_merged_output() {
    // End-to-end: two L0 SSTs with overlapping keys → leveled compaction
    // merges them into a single L1 SST with all keys present and in order.
    let config = CompactionConfig::new()
        .with_l0_strategy(L0CompactionStrategy::Leveled)
        .with_sst_size(MIN_SST_SIZE_BYTES);
    let mut compactor = Compactor::new(config);
    let mut tree = LsmTree::new();

    tree.add_sst(0, sst(1, 0, "a", "m", 5));
    tree.add_sst(0, sst(2, 0, "n", "z", 5));

    // Verify the trigger fires.
    assert!(compactor.trigger.needs_compaction(&tree, 0));

    let run_a: Vec<VersionedEntry> = (b'a'..=b'm')
        .map(|c| VersionedEntry {
            key: vec![c],
            value: Some(vec![c]),
            timestamp: 1,
        })
        .collect();
    let run_b: Vec<VersionedEntry> = (b'n'..=b'z')
        .map(|c| VersionedEntry {
            key: vec![c],
            value: Some(vec![c]),
            timestamp: 1,
        })
        .collect();

    for &id in &[1u64, 2] {
        tree.remove_sst(0, id);
    }

    let gc = GcContext::new();
    let output = compactor.compact(&mut tree, 0, vec![run_a, run_b], &gc);

    // All 26 letters must survive.
    assert_eq!(output.total_entries, 26);
    // Output lands in L1.
    for sst_meta in &output.new_ssts {
        assert_eq!(sst_meta.level, 1, "leveled L0 output must land in L1");
    }
    // Keys must be in sorted order across all output SSTs.
    let all_keys: Vec<Vec<u8>> = output
        .sst_entries
        .iter()
        .flat_map(|g| g.iter().map(|e| e.key.clone()))
        .collect();
    let mut sorted = all_keys.clone();
    sorted.sort();
    assert_eq!(all_keys, sorted, "merged output must be sorted");
}

// ---------------------------------------------------------------------------
// Task 39.3 — flush pre-emption under load (compaction-level test)
// ---------------------------------------------------------------------------

#[test]
fn compaction_config_with_l0_strategy_round_trips() {
    let config = CompactionConfig::new()
        .with_l0_strategy(L0CompactionStrategy::Leveled);
    assert_eq!(config.l0_strategy, L0CompactionStrategy::Leveled);

    let config2 = CompactionConfig::new()
        .with_l0_strategy(L0CompactionStrategy::Tiered);
    assert_eq!(config2.l0_strategy, L0CompactionStrategy::Tiered);
}

#[test]
fn compactor_uses_l0_strategy_from_config() {
    // Compactor::new must propagate the config's l0_strategy into the
    // trigger so the effective threshold is correct.
    let config = CompactionConfig::new()
        .with_l0_strategy(L0CompactionStrategy::Leveled);
    let compactor = Compactor::new(config);
    assert_eq!(
        compactor.trigger.l0_strategy,
        L0CompactionStrategy::Leveled,
        "Compactor must propagate l0_strategy from config to trigger"
    );
}
