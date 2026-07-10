//! NUMA-aware buffer pool with HotSet (LRU) and ScanBuffer (clock-sweep).
//!
//! The buffer pool is partitioned into two regions:
//! - **HotSet** (70% of capacity): LRU eviction, used for point lookups.
//! - **ScanBuffer** (30% of capacity): Clock-sweep eviction, used for sequential scans.
//!
//! Each region is NUMA-partitioned: on Linux, one instance per NUMA node;
//! on macOS/Windows, a single partition fallback.

mod clock_sweep;
mod lru_cache;
mod numa;
mod rgabh;

#[cfg(test)]
mod tests;

use std::collections::HashSet;
use std::time::Instant;

pub use clock_sweep::ClockSweep;
pub use lru_cache::LruCache;
pub use numa::NumaPartitioned;
pub use rgabh::{BlockHeat, HeatConstants, HeatTracker};

use galaxdb_common::BlockId;

/// Number of resident blocks sampled per RGABH eviction. Redis uses ~5-10 for
/// approximate LRU/LFU; 16 gives a closer approximation to true coldest at
/// negligible cost.
const EVICTION_SAMPLE: usize = 16;

/// A cached block held in the buffer pool.
#[derive(Debug, Clone)]
pub struct CachedBlock {
    /// The block identifier.
    pub block_id: BlockId,
    /// Raw block bytes.
    pub data: Vec<u8>,
}

/// The type of access that determines routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessType {
    /// Point lookup — routes to HotSet.
    PointLookup,
    /// Sequential scan — routes to ScanBuffer.
    SequentialScan,
}

/// NUMA-aware buffer pool combining HotSet (LRU) and ScanBuffer (clock-sweep).
///
/// Point lookups place blocks in the HotSet. Sequential scans place blocks in
/// the ScanBuffer. The ScanBuffer never evicts a block that is resident in the
/// HotSet.
pub struct BufferPool {
    /// 70% of capacity — LRU eviction for point lookups.
    hot_set: NumaPartitioned<LruCache<BlockId, CachedBlock>>,
    /// 30% of capacity — clock-sweep eviction for sequential scans.
    scan_buffer: NumaPartitioned<ClockSweep<BlockId, CachedBlock>>,
    /// RGABH gradient state, one tracker per NUMA node. `None` when RGABH is
    /// disabled (the off switch) — the pool then uses the exact LRU/clock
    /// baseline.
    heat: Option<NumaPartitioned<HeatTracker>>,
}

impl BufferPool {
    /// Create a new buffer pool.
    ///
    /// `total_capacity` is the total number of block slots. 70% goes to HotSet,
    /// 30% goes to ScanBuffer. `numa_nodes` controls the number of NUMA partitions
    /// (use 1 for single-partition fallback on macOS/Windows).
    pub fn new(total_capacity: usize, numa_nodes: usize) -> Self {
        let hot_capacity = (total_capacity * 70) / 100;
        let scan_capacity = total_capacity - hot_capacity;

        // Divide capacity evenly across NUMA nodes.
        let hot_per_node = hot_capacity.max(1) / numa_nodes.max(1);
        let scan_per_node = scan_capacity.max(1) / numa_nodes.max(1);

        let hot_set = NumaPartitioned::new(numa_nodes, || LruCache::new(hot_per_node.max(1)));
        let scan_buffer =
            NumaPartitioned::new(numa_nodes, || ClockSweep::new(scan_per_node.max(1)));

        BufferPool {
            hot_set,
            scan_buffer,
            heat: None,
        }
    }

    /// Create an RGABH-adaptive buffer pool. Identical layout to [`BufferPool::new`],
    /// but HotSet eviction is driven by the per-block heat gradient (coldest-by-
    /// score victim) instead of the LRU tail, and [`BufferPool::prefetch_tick`]
    /// becomes active. Disabling RGABH (`new`) reproduces the LRU/clock baseline.
    pub fn new_adaptive(total_capacity: usize, numa_nodes: usize) -> Self {
        Self::new_adaptive_with(total_capacity, numa_nodes, HeatConstants::default())
    }

    /// [`BufferPool::new_adaptive`] with explicit heat constants.
    pub fn new_adaptive_with(
        total_capacity: usize,
        numa_nodes: usize,
        constants: HeatConstants,
    ) -> Self {
        let mut pool = Self::new(total_capacity, numa_nodes);
        // Cap the doorkeeper at 4× the per-node HotSet capacity: enough history
        // to let a genuinely hot key prove its frequency before admission, while
        // keeping heat memory bounded to a small multiple of residency.
        let per_node_hot = pool.hot_set_capacity(0).max(1);
        let cap = per_node_hot.saturating_mul(4);
        pool.heat = Some(NumaPartitioned::new(numa_nodes.max(1), || {
            HeatTracker::with_cap(constants, cap)
        }));
        pool
    }

    /// Whether RGABH adaptive admission/eviction is enabled.
    pub fn is_adaptive(&self) -> bool {
        self.heat.is_some()
    }

    /// Create a buffer pool using auto-detected NUMA topology.
    pub fn with_auto_numa(total_capacity: usize) -> Self {
        let numa_nodes = numa::detect_numa_nodes();
        Self::new(total_capacity, numa_nodes)
    }

    /// Get a block for a point lookup. Checks HotSet first, then ScanBuffer.
    /// If found in ScanBuffer, promotes it to HotSet.
    /// Returns `None` if the block is not cached.
    pub fn get_for_point_lookup(&mut self, block_id: BlockId, node: usize) -> Option<CachedBlock> {
        // Check HotSet first.
        if let Some(block) = self.hot_set.get_mut(node).get(&block_id) {
            let block = block.clone();
            self.record_heat(node, block_id, false);
            return Some(block);
        }

        // Check ScanBuffer — if found, promote to HotSet (adaptive-aware).
        if let Some(block) = self.scan_buffer.get_mut(node).remove(&block_id) {
            self.record_heat(node, block_id, false);
            self.admit_hot(node, block_id, block.clone());
            return Some(block);
        }

        None
    }

    /// Get a block for a sequential scan. Checks ScanBuffer first, then HotSet.
    /// Does NOT promote from HotSet to ScanBuffer.
    /// Returns `None` if the block is not cached.
    pub fn get_for_scan(&mut self, block_id: BlockId, node: usize) -> Option<CachedBlock> {
        // Check ScanBuffer first.
        if let Some(block) = self.scan_buffer.get_mut(node).get(&block_id) {
            let block = block.clone();
            self.record_heat(node, block_id, true);
            return Some(block);
        }

        // Check HotSet — return if found but don't move it.
        if let Some(block) = self.hot_set.get_mut(node).get(&block_id) {
            let block = block.clone();
            self.record_heat(node, block_id, true);
            return Some(block);
        }

        None
    }

    /// Insert a block into the buffer pool with the given access type routing.
    ///
    /// - `PointLookup` → HotSet (LRU eviction)
    /// - `SequentialScan` → ScanBuffer (clock-sweep eviction, never evicts HotSet-resident blocks)
    pub fn insert(
        &mut self,
        block_id: BlockId,
        block: CachedBlock,
        access_type: AccessType,
        node: usize,
    ) {
        match access_type {
            AccessType::PointLookup => {
                // Record heat first so the admission decision sees this access.
                self.record_heat(node, block_id, false);
                self.admit_hot(node, block_id, block);
            }
            AccessType::SequentialScan => {
                // Collect the set of block IDs currently in the HotSet for this node,
                // so the ScanBuffer can avoid evicting them.
                let hot = self.hot_set.get_mut(node);
                let hot_set_ids: HashSet<BlockId> = hot.keys().collect();

                let scan = self.scan_buffer.get_mut(node);
                scan.put_with_constraint(block_id, block, &hot_set_ids);
                self.record_heat(node, block_id, true);
            }
        }
        // Task 38.3: mirror total occupancy across all NUMA
        // partitions into the observe gauges. "Usage" here is entry
        // count (not bytes) — units the spec didn't fix, so we pick
        // entries since that's what the buffer pool tracks natively.
        self.publish_usage_metrics();
    }

    /// Publish current hot-set / scan-buffer occupancy to the observe
    /// gauges (task 38.3). Called from `insert` so `/metrics` always
    /// reflects the live state.
    fn publish_usage_metrics(&self) {
        let m = galaxdb_observe::metrics();
        let nodes = self.hot_set.node_count();
        let mut hot_total: i64 = 0;
        let mut scan_total: i64 = 0;
        for node in 0..nodes {
            hot_total += self.hot_set.get(node).len() as i64;
            scan_total += self.scan_buffer.get(node).len() as i64;
        }
        m.buffer_pool_hot_set_usage.set(hot_total);
        m.buffer_pool_scan_buffer_usage.set(scan_total);
    }

    /// Record a heat access for a block (no-op when RGABH is disabled).
    fn record_heat(&mut self, node: usize, block_id: BlockId, is_training: bool) {
        if let Some(heat) = self.heat.as_mut() {
            heat.get_mut(node)
                .record_access(block_id, Instant::now(), is_training);
        }
    }

    /// Admit a block into the HotSet.
    ///
    /// - RGABH off: plain `LruCache::put` (LRU eviction — the baseline).
    /// - RGABH on: **frequency-based admission control** (W-TinyLFU-style). When
    ///   the HotSet is full and the block is new, pick an eviction victim (the
    ///   coldest of a small LRU-tail sample, O(K)); admit the newcomer only if
    ///   it is hotter than that victim, otherwise reject it. This stops a stream
    ///   of one-shot cold blocks (a scan flood) from displacing the durably-hot
    ///   working set — the core RGABH win — while staying O(K) per admission.
    ///   The rejected block's heat stays tracked, so a genuinely hot key is
    ///   admitted once it has proven its frequency.
    fn admit_hot(&mut self, node: usize, block_id: BlockId, block: CachedBlock) {
        let Some(heat) = self.heat.as_mut() else {
            self.hot_set.get_mut(node).put(block_id, block);
            return;
        };

        let hot = self.hot_set.get_mut(node);
        // Update-in-place or free capacity: always admit.
        if hot.contains(&block_id) || hot.len() < hot.capacity() {
            self.hot_set.get_mut(node).put(block_id, block);
            return;
        }

        let now = Instant::now();
        let sample = hot.lru_tail_keys(EVICTION_SAMPLE);
        let tracker = heat.get(node);
        let Some(victim) = tracker.coldest(&sample, now) else {
            self.hot_set.get_mut(node).put(block_id, block);
            return;
        };
        let victim_score = tracker.score(victim, now);
        let newcomer_score = tracker.score(block_id, now);

        if newcomer_score > victim_score {
            // Newcomer has proven hotter than the coldest stale victim: evict.
            self.hot_set.get_mut(node).remove(&victim);
            heat.get_mut(node).remove(victim);
            self.hot_set.get_mut(node).put(block_id, block);
        }
        // else: reject admission — keep the victim, drop the newcomer. Its heat
        // remains tracked (bounded by the tracker cap) so it can be admitted on
        // a later access once it is hotter than the resident set's coldest.
    }

    /// One background prefetch pass (BK / background queue only — never called on
    /// the foreground read path). Selects up to `max` non-resident blocks whose
    /// short-heat velocity is at or above `min_velocity`, loads each via `loader`,
    /// and admits the returned block into the HotSet. Returns the block ids
    /// prefetched. No-op when RGABH is disabled.
    ///
    /// `loader` is the real IO seam: the caller supplies the function that reads a
    /// block from storage. Prefetch decisions come from the live heat gradient, so
    /// this speculatively warms blocks trending hot without touching foreground
    /// latency.
    pub fn prefetch_tick(
        &mut self,
        node: usize,
        min_velocity: f32,
        max: usize,
        mut loader: impl FnMut(BlockId) -> Option<CachedBlock>,
    ) -> Vec<BlockId> {
        let Some(heat) = self.heat.as_ref() else {
            return Vec::new();
        };
        let now = Instant::now();
        let hot_snapshot: HashSet<BlockId> = self.hot_set.get(node).keys().collect();
        let candidates = heat.get(node).prefetch_candidates(now, min_velocity, max, |id| {
            hot_snapshot.contains(&id)
        });
        let mut loaded = Vec::new();
        for id in candidates {
            if let Some(block) = loader(id) {
                self.admit_hot(node, id, block);
                loaded.push(id);
            }
        }
        loaded
    }

    /// Heat score of a resident/tracked block (0.0 if RGABH off or untracked).
    /// Exposed for tests and observability.
    pub fn heat_score(&self, node: usize, block_id: BlockId) -> f32 {
        self.heat
            .as_ref()
            .map(|h| h.get(node).score(block_id, Instant::now()))
            .unwrap_or(0.0)
    }

    /// Returns the number of NUMA partitions.
    pub fn numa_node_count(&self) -> usize {
        self.hot_set.node_count()
    }

    /// Returns the number of entries in the HotSet for a given NUMA node.
    pub fn hot_set_len(&self, node: usize) -> usize {
        self.hot_set.get(node).len()
    }

    /// Returns the number of entries in the ScanBuffer for a given NUMA node.
    pub fn scan_buffer_len(&self, node: usize) -> usize {
        self.scan_buffer.get(node).len()
    }

    /// Returns the capacity of the HotSet for a given NUMA node.
    pub fn hot_set_capacity(&self, node: usize) -> usize {
        self.hot_set.get(node).capacity()
    }

    /// Returns the capacity of the ScanBuffer for a given NUMA node.
    pub fn scan_buffer_capacity(&self, node: usize) -> usize {
        self.scan_buffer.get(node).capacity()
    }
}
