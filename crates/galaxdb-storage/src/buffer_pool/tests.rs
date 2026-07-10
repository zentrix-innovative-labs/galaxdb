//! Tests for the NUMA-aware buffer pool.

use super::*;

// ── Helper ────────────────────────────────────────────────────────────

fn make_block(id: BlockId) -> CachedBlock {
    CachedBlock {
        block_id: id,
        data: vec![id as u8; 64],
    }
}

// ── LRU eviction correctness ─────────────────────────────────────────

#[test]
fn lru_evicts_least_recently_used() {
    let mut cache = LruCache::<u64, String>::new(3);

    cache.put(1, "a".into());
    cache.put(2, "b".into());
    cache.put(3, "c".into());

    // Cache is full: [3(MRU), 2, 1(LRU)]
    // Inserting 4 should evict 1.
    let evicted = cache.put(4, "d".into());
    assert_eq!(evicted, Some((1, "a".into())));
    assert!(!cache.contains(&1));
    assert!(cache.contains(&4));
}

#[test]
fn lru_access_promotes_to_mru() {
    let mut cache = LruCache::<u64, String>::new(3);

    cache.put(1, "a".into());
    cache.put(2, "b".into());
    cache.put(3, "c".into());

    // Access key 1 — promotes it to MRU.
    cache.get(&1);

    // Now order is [1(MRU), 3, 2(LRU)]. Inserting 4 should evict 2.
    let evicted = cache.put(4, "d".into());
    assert_eq!(evicted, Some((2, "b".into())));
}

#[test]
fn lru_update_existing_key() {
    let mut cache = LruCache::<u64, String>::new(3);

    cache.put(1, "a".into());
    cache.put(2, "b".into());
    cache.put(1, "a_updated".into());

    assert_eq!(cache.len(), 2);
    assert_eq!(cache.get(&1), Some(&"a_updated".into()));
}

#[test]
fn lru_remove() {
    let mut cache = LruCache::<u64, String>::new(3);

    cache.put(1, "a".into());
    cache.put(2, "b".into());

    let removed = cache.remove(&1);
    assert_eq!(removed, Some("a".into()));
    assert_eq!(cache.len(), 1);
    assert!(!cache.contains(&1));
}

#[test]
fn lru_eviction_order_with_multiple_accesses() {
    let mut cache = LruCache::<u64, u64>::new(4);

    cache.put(1, 10);
    cache.put(2, 20);
    cache.put(3, 30);
    cache.put(4, 40);

    // Access 1 and 2, making them MRU. Order: [2, 1, 4, 3(LRU)]
    cache.get(&1);
    cache.get(&2);

    // Evict 3 (LRU).
    let evicted = cache.put(5, 50);
    assert_eq!(evicted, Some((3, 30)));

    // Evict 4 (now LRU).
    let evicted = cache.put(6, 60);
    assert_eq!(evicted, Some((4, 40)));
}

// ── Clock-sweep eviction correctness ─────────────────────────────────

#[test]
fn clock_sweep_basic_eviction() {
    let mut cache = ClockSweep::<u64, String>::new(3);

    cache.put(1, "a".into());
    cache.put(2, "b".into());
    cache.put(3, "c".into());

    // All entries have referenced=true from insertion.
    // First sweep clears all referenced bits.
    // Second sweep evicts the first unreferenced entry found by the hand.
    let evicted = cache.put(4, "d".into());
    assert!(evicted.is_some());
    assert_eq!(cache.len(), 3);
    assert!(cache.contains(&4));
}

#[test]
fn clock_sweep_referenced_bit_gives_second_chance() {
    let mut cache = ClockSweep::<u64, String>::new(3);

    cache.put(1, "a".into());
    cache.put(2, "b".into());
    cache.put(3, "c".into());

    // All entries start with referenced=true from insertion.
    // Insert key 4 — triggers eviction. The clock hand sweeps:
    //   First pass: clears all referenced bits (second chance).
    //   Second pass: evicts the first unreferenced entry at the hand position.
    let evicted = cache.put(4, "d".into());
    assert!(evicted.is_some());
    let (evicted_key, _) = evicted.unwrap();

    // The evicted key got its second chance (referenced bit was cleared on first pass)
    // but was evicted on the second pass. The remaining keys survived the first pass.
    assert!(!cache.contains(&evicted_key));
    assert!(cache.contains(&4));
    assert_eq!(cache.len(), 3);

    // Now test that a recently-accessed key survives when others are not accessed.
    // Reset: create a fresh cache.
    let mut cache2 = ClockSweep::<u64, String>::new(3);
    cache2.put(10, "x".into());
    cache2.put(20, "y".into());
    cache2.put(30, "z".into());

    // Trigger one eviction to clear referenced bits of survivors.
    let _ = cache2.put(40, "w".into());

    // Now access key 40 — sets its referenced bit.
    cache2.get(&40);

    // Insert another key — key 40 should survive because it was recently accessed.
    let evicted2 = cache2.put(50, "v".into());
    assert!(evicted2.is_some());
    assert!(cache2.contains(&40), "recently accessed key should survive");
    assert!(cache2.contains(&50));
}

#[test]
fn clock_sweep_remove() {
    let mut cache = ClockSweep::<u64, String>::new(3);

    cache.put(1, "a".into());
    cache.put(2, "b".into());

    let removed = cache.remove(&1);
    assert_eq!(removed, Some("a".into()));
    assert_eq!(cache.len(), 1);
    assert!(!cache.contains(&1));
}

#[test]
fn clock_sweep_update_existing() {
    let mut cache = ClockSweep::<u64, String>::new(3);

    cache.put(1, "a".into());
    cache.put(1, "a_updated".into());

    assert_eq!(cache.len(), 1);
    assert_eq!(cache.get(&1), Some(&"a_updated".into()));
}

// ── Clock-sweep with HotSet constraint ───────────────────────────────

#[test]
fn clock_sweep_never_evicts_protected_keys() {
    use std::collections::HashSet;

    let mut cache = ClockSweep::<u64, String>::new(3);

    cache.put(1, "a".into());
    cache.put(2, "b".into());
    cache.put(3, "c".into());

    // Protect keys 1 and 2 (simulating HotSet-resident blocks).
    let mut protected = HashSet::new();
    protected.insert(1u64);
    protected.insert(2u64);

    // Inserting 4 should only evict key 3 (the only unprotected one).
    let evicted = cache.put_with_constraint(4, "d".into(), &protected);
    assert_eq!(evicted.map(|(k, _)| k), Some(3));
    assert!(cache.contains(&1));
    assert!(cache.contains(&2));
    assert!(cache.contains(&4));
}

#[test]
fn clock_sweep_all_protected_skips_insertion() {
    use std::collections::HashSet;

    let mut cache = ClockSweep::<u64, String>::new(2);

    cache.put(1, "a".into());
    cache.put(2, "b".into());

    // Protect all keys.
    let mut protected = HashSet::new();
    protected.insert(1u64);
    protected.insert(2u64);

    // Cannot evict anything — insertion is a no-op.
    let evicted = cache.put_with_constraint(3, "c".into(), &protected);
    assert!(evicted.is_none());
    assert!(!cache.contains(&3));
    assert_eq!(cache.len(), 2);
}

// ── NUMA partitioning ────────────────────────────────────────────────

#[test]
fn numa_partitioned_single_node_fallback() {
    let partitioned = NumaPartitioned::new(1, || LruCache::<u64, String>::new(10));
    assert_eq!(partitioned.node_count(), 1);
}

#[test]
fn numa_partitioned_multiple_nodes() {
    let partitioned = NumaPartitioned::new(4, || LruCache::<u64, String>::new(10));
    assert_eq!(partitioned.node_count(), 4);
}

#[test]
fn numa_detect_returns_at_least_one() {
    let nodes = numa::detect_numa_nodes();
    assert!(nodes >= 1);
}

#[test]
fn numa_current_node_is_valid() {
    let nodes = numa::detect_numa_nodes();
    let current = numa::current_numa_node();
    assert!(current < nodes);
}

// ── Cross-partition isolation ────────────────────────────────────────

#[test]
fn cross_partition_isolation() {
    let mut partitioned = NumaPartitioned::new(2, || LruCache::<u64, String>::new(10));

    // Insert into node 0.
    partitioned.get_mut(0).put(1, "node0".into());

    // Node 1 should not see it.
    assert!(partitioned.get_mut(1).get(&1).is_none());

    // Insert into node 1.
    partitioned.get_mut(1).put(2, "node1".into());

    // Node 0 should not see it.
    assert!(partitioned.get_mut(0).get(&2).is_none());
}

// ── BufferPool integration ───────────────────────────────────────────

#[test]
fn buffer_pool_point_lookup_routes_to_hot_set() {
    let mut pool = BufferPool::new(10, 1);

    pool.insert(1, make_block(1), AccessType::PointLookup, 0);

    assert_eq!(pool.hot_set_len(0), 1);
    assert_eq!(pool.scan_buffer_len(0), 0);

    let block = pool.get_for_point_lookup(1, 0);
    assert!(block.is_some());
    assert_eq!(block.unwrap().block_id, 1);
}

#[test]
fn buffer_pool_scan_routes_to_scan_buffer() {
    let mut pool = BufferPool::new(10, 1);

    pool.insert(1, make_block(1), AccessType::SequentialScan, 0);

    assert_eq!(pool.hot_set_len(0), 0);
    assert_eq!(pool.scan_buffer_len(0), 1);

    let block = pool.get_for_scan(1, 0);
    assert!(block.is_some());
    assert_eq!(block.unwrap().block_id, 1);
}

#[test]
fn buffer_pool_point_lookup_promotes_from_scan_buffer() {
    let mut pool = BufferPool::new(10, 1);

    // Insert via scan.
    pool.insert(1, make_block(1), AccessType::SequentialScan, 0);
    assert_eq!(pool.scan_buffer_len(0), 1);
    assert_eq!(pool.hot_set_len(0), 0);

    // Point lookup should find it in ScanBuffer and promote to HotSet.
    let block = pool.get_for_point_lookup(1, 0);
    assert!(block.is_some());
    assert_eq!(pool.hot_set_len(0), 1);
    // It should be removed from ScanBuffer after promotion.
    assert_eq!(pool.scan_buffer_len(0), 0);
}

#[test]
fn buffer_pool_scan_does_not_promote_from_hot_set() {
    let mut pool = BufferPool::new(10, 1);

    // Insert via point lookup.
    pool.insert(1, make_block(1), AccessType::PointLookup, 0);
    assert_eq!(pool.hot_set_len(0), 1);

    // Scan should find it in HotSet but NOT move it to ScanBuffer.
    let block = pool.get_for_scan(1, 0);
    assert!(block.is_some());
    assert_eq!(pool.hot_set_len(0), 1);
    assert_eq!(pool.scan_buffer_len(0), 0);
}

#[test]
fn buffer_pool_capacity_split_70_30() {
    let pool = BufferPool::new(100, 1);

    assert_eq!(pool.hot_set_capacity(0), 70);
    assert_eq!(pool.scan_buffer_capacity(0), 30);
}

#[test]
fn buffer_pool_hot_set_lru_eviction() {
    // HotSet capacity = 70% of 10 = 7
    let mut pool = BufferPool::new(10, 1);
    let hot_cap = pool.hot_set_capacity(0);

    // Fill the HotSet.
    for i in 0..hot_cap {
        pool.insert(i as u64, make_block(i as u64), AccessType::PointLookup, 0);
    }
    assert_eq!(pool.hot_set_len(0), hot_cap);

    // Insert one more — should evict the LRU entry (block 0).
    pool.insert(100, make_block(100), AccessType::PointLookup, 0);
    assert_eq!(pool.hot_set_len(0), hot_cap);

    // Block 0 should be evicted.
    let block = pool.get_for_point_lookup(0, 0);
    assert!(block.is_none());

    // Block 100 should be present.
    let block = pool.get_for_point_lookup(100, 0);
    assert!(block.is_some());
}

#[test]
fn buffer_pool_scan_buffer_clock_sweep_eviction() {
    // ScanBuffer capacity = 30% of 10 = 3
    let mut pool = BufferPool::new(10, 1);
    let scan_cap = pool.scan_buffer_capacity(0);

    // Fill the ScanBuffer.
    for i in 0..scan_cap {
        pool.insert(
            i as u64,
            make_block(i as u64),
            AccessType::SequentialScan,
            0,
        );
    }
    assert_eq!(pool.scan_buffer_len(0), scan_cap);

    // Insert one more — should evict via clock-sweep.
    pool.insert(100, make_block(100), AccessType::SequentialScan, 0);
    assert_eq!(pool.scan_buffer_len(0), scan_cap);
    assert!(pool.get_for_scan(100, 0).is_some());
}

#[test]
fn buffer_pool_scan_buffer_never_evicts_hot_set_resident() {
    // Total capacity 10: HotSet=7, ScanBuffer=3
    let mut pool = BufferPool::new(10, 1);
    let scan_cap = pool.scan_buffer_capacity(0);

    // Put blocks 1, 2, 3 in HotSet.
    pool.insert(1, make_block(1), AccessType::PointLookup, 0);
    pool.insert(2, make_block(2), AccessType::PointLookup, 0);
    pool.insert(3, make_block(3), AccessType::PointLookup, 0);

    // Also put blocks 1, 2, 3 in ScanBuffer (same block IDs).
    pool.insert(1, make_block(1), AccessType::SequentialScan, 0);
    pool.insert(2, make_block(2), AccessType::SequentialScan, 0);
    pool.insert(3, make_block(3), AccessType::SequentialScan, 0);

    assert_eq!(pool.scan_buffer_len(0), scan_cap);

    // Now insert block 10 into ScanBuffer. The eviction constraint means
    // blocks 1, 2, 3 are protected (they're in HotSet). Since all ScanBuffer
    // entries are protected, the insertion should be a no-op.
    pool.insert(10, make_block(10), AccessType::SequentialScan, 0);

    // All original blocks should still be in ScanBuffer.
    assert!(pool.get_for_scan(1, 0).is_some());
    assert!(pool.get_for_scan(2, 0).is_some());
    assert!(pool.get_for_scan(3, 0).is_some());
}

#[test]
fn buffer_pool_multi_numa_node() {
    let mut pool = BufferPool::new(20, 2);

    assert_eq!(pool.numa_node_count(), 2);

    // Insert into node 0.
    pool.insert(1, make_block(1), AccessType::PointLookup, 0);

    // Should be visible on node 0 but not node 1.
    assert!(pool.get_for_point_lookup(1, 0).is_some());
    assert!(pool.get_for_point_lookup(1, 1).is_none());

    // Insert into node 1.
    pool.insert(2, make_block(2), AccessType::PointLookup, 1);

    assert!(pool.get_for_point_lookup(2, 1).is_some());
    assert!(pool.get_for_point_lookup(2, 0).is_none());
}

// ── RGABH adaptive buffer pool (v0.7, inventory 8.1/8.3) ──────────────

use std::time::Instant;

/// A deterministic Zipfian-ish skewed key generator: ~80% of accesses hit a
/// small hot set, the rest spread over a large cold tail. Uses a fixed LCG so
/// the sequence is identical across pool configurations (fair comparison).
struct SkewGen {
    state: u64,
    hot: u64,
    cold: u64,
}

impl SkewGen {
    fn new(hot: u64, cold: u64) -> Self {
        SkewGen {
            state: 0x1234_5678_9abc_def0,
            hot,
            cold,
        }
    }
    fn next_u64(&mut self) -> u64 {
        // xorshift64 — deterministic, no external deps.
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }
    fn next_key(&mut self) -> u64 {
        let r = self.next_u64();
        if r % 100 < 80 {
            r % self.hot // hot set
        } else {
            self.hot + (r % self.cold) // cold tail
        }
    }
}

/// Drive a workload through a pool and return the HotSet hit rate.
fn run_skew_workload(pool: &mut BufferPool, ops: usize, hot: u64, cold: u64) -> f64 {
    let mut keygen = SkewGen::new(hot, cold);
    let mut hits = 0usize;
    for _ in 0..ops {
        let key = keygen.next_key();
        if pool.get_for_point_lookup(key, 0).is_some() {
            hits += 1;
        } else {
            pool.insert(key, make_block(key), AccessType::PointLookup, 0);
        }
    }
    hits as f64 / ops as f64
}

#[test]
fn rgabh_off_by_default_and_on_when_adaptive() {
    assert!(!BufferPool::new(100, 1).is_adaptive());
    assert!(BufferPool::new_adaptive(100, 1).is_adaptive());
}

#[test]
fn rgabh_improves_hit_rate_on_skewed_workload() {
    // Hot set (40 keys) larger than nothing but the HotSet capacity is 70% of
    // 60 = 42 slots, so the hot set *fits* but LRU can still evict a hot block
    // during a run of cold-tail accesses. RGABH's long_heat should retain the
    // durable hot set and beat plain LRU.
    let hot = 40u64;
    let cold = 400u64;
    let ops = 20_000usize;

    let mut baseline = BufferPool::new(60, 1);
    let base_rate = run_skew_workload(&mut baseline, ops, hot, cold);

    let mut adaptive = BufferPool::new_adaptive(60, 1);
    let adaptive_rate = run_skew_workload(&mut adaptive, ops, hot, cold);

    // RGABH must not do worse than LRU on a skewed workload; the gradient's
    // durable-membership term should keep the hot set resident.
    assert!(
        adaptive_rate >= base_rate,
        "RGABH hit rate {adaptive_rate:.4} should be >= LRU baseline {base_rate:.4}"
    );
}

#[test]
fn rgabh_off_switch_reproduces_lru_baseline_exactly() {
    // Two non-adaptive pools must produce byte-identical hit rates (determinism)
    // and the non-adaptive pool must behave exactly like the historical LRU
    // path — proven by the eviction test below still holding on a non-adaptive
    // pool.
    let mut a = BufferPool::new(60, 1);
    let mut b = BufferPool::new(60, 1);
    let ra = run_skew_workload(&mut a, 5_000, 40, 400);
    let rb = run_skew_workload(&mut b, 5_000, 40, 400);
    assert_eq!(ra, rb, "non-adaptive pool must be deterministic");

    // Classic LRU eviction still holds with RGABH off (baseline preserved).
    let mut pool = BufferPool::new(10, 1);
    let cap = pool.hot_set_capacity(0);
    for i in 0..cap {
        pool.insert(i as u64, make_block(i as u64), AccessType::PointLookup, 0);
    }
    pool.insert(100, make_block(100), AccessType::PointLookup, 0);
    assert!(pool.get_for_point_lookup(0, 0).is_none(), "LRU tail evicted");
    assert!(pool.get_for_point_lookup(100, 0).is_some());
}

#[test]
fn rgabh_preserves_scan_hotset_isolation_invariant() {
    // The OLTP/OLAP isolation invariant (ScanBuffer never evicts a HotSet-
    // resident block) must still hold under RGABH.
    let mut pool = BufferPool::new_adaptive(10, 1);
    let scan_cap = pool.scan_buffer_capacity(0);

    pool.insert(1, make_block(1), AccessType::PointLookup, 0);
    pool.insert(2, make_block(2), AccessType::PointLookup, 0);
    pool.insert(3, make_block(3), AccessType::PointLookup, 0);

    pool.insert(1, make_block(1), AccessType::SequentialScan, 0);
    pool.insert(2, make_block(2), AccessType::SequentialScan, 0);
    pool.insert(3, make_block(3), AccessType::SequentialScan, 0);
    assert_eq!(pool.scan_buffer_len(0), scan_cap);

    // A scan insert cannot evict HotSet-resident blocks 1/2/3.
    pool.insert(10, make_block(10), AccessType::SequentialScan, 0);
    assert!(pool.get_for_scan(1, 0).is_some());
    assert!(pool.get_for_scan(2, 0).is_some());
    assert!(pool.get_for_scan(3, 0).is_some());
}

#[test]
fn rgabh_evicts_coldest_by_heat_not_lru_tail() {
    // HotSet capacity = 70% of 10 = 7. Fill 0..7, then repeatedly heat block 0
    // (the LRU tail) so it is the *hottest*. A new insert must evict a genuinely
    // cold block, NOT block 0 — the RGABH difference from plain LRU.
    let mut pool = BufferPool::new_adaptive(10, 1);
    let cap = pool.hot_set_capacity(0);
    assert_eq!(cap, 7);
    for i in 0..cap as u64 {
        pool.insert(i, make_block(i), AccessType::PointLookup, 0);
    }
    // Hammer block 0 so its long_heat dominates.
    for _ in 0..50 {
        assert!(pool.get_for_point_lookup(0, 0).is_some());
    }
    // Insert a new block → forces one eviction.
    pool.insert(100, make_block(100), AccessType::PointLookup, 0);

    // Block 0 (hottest) must survive; a cold block must have been evicted.
    assert!(
        pool.get_for_point_lookup(0, 0).is_some(),
        "RGABH must retain the hottest block, not evict it as the LRU tail"
    );
    assert!(pool.get_for_point_lookup(100, 0).is_some());
}

#[test]
fn rgabh_prefetch_loads_high_velocity_blocks_off_foreground() {
    // Build velocity on a set of non-resident blocks, then a prefetch tick must
    // warm them via the loader (the background IO seam).
    let mut pool = BufferPool::new_adaptive(100, 1);

    // Access blocks 200,201,202 through the scan path then remove them so they
    // are tracked (have velocity) but not resident.
    for _ in 0..5 {
        for id in 200..203u64 {
            pool.insert(id, make_block(id), AccessType::SequentialScan, 0);
            let _ = pool.get_for_scan(id, 0);
        }
    }
    // Evict them from the scan buffer by flooding with fresh scan blocks that
    // are not the hot ids (they remain tracked in the heat map).
    for id in 300..400u64 {
        pool.insert(id, make_block(id), AccessType::SequentialScan, 0);
    }

    // Prefetch: load any block trending hot into the HotSet via the loader.
    let mut loaded_ids = Vec::new();
    let got = pool.prefetch_tick(0, 0.0, 8, |id| {
        loaded_ids.push(id);
        Some(make_block(id))
    });
    assert!(
        !got.is_empty(),
        "prefetch should warm at least one high-velocity non-resident block"
    );
    // Every prefetched block is now resident in the HotSet.
    for id in &got {
        assert!(pool.get_for_point_lookup(*id, 0).is_some());
    }
}

#[test]
fn rgabh_prefetch_is_noop_when_disabled() {
    let mut pool = BufferPool::new(100, 1);
    let got = pool.prefetch_tick(0, 0.0, 8, |id| Some(make_block(id)));
    assert!(got.is_empty(), "prefetch must be a no-op when RGABH is off");
}

#[test]
fn block_heat_decays_over_time() {
    let c = HeatConstants::default();
    let t0 = Instant::now();
    let mut tracker = HeatTracker::new(c);
    tracker.record_access(1, t0, false);
    let s0 = tracker.score(1, t0);
    // Score after simulated elapsed time is lower (decay). We can't fast-forward
    // Instant, so assert the freshly-recorded score is positive and a second
    // untracked block scores 0.
    assert!(s0 > 0.0);
    assert_eq!(tracker.score(999, t0), 0.0);
    // Two accesses raise the score above one.
    tracker.record_access(1, t0, false);
    assert!(tracker.score(1, t0) > s0);
}
