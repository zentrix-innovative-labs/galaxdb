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

#[cfg(test)]
mod tests;

use std::collections::HashSet;

pub use clock_sweep::ClockSweep;
pub use lru_cache::LruCache;
pub use numa::NumaPartitioned;

use galaxdb_common::BlockId;

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
        }
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
        let hot = self.hot_set.get_mut(node);

        // Check HotSet first.
        if let Some(block) = hot.get(&block_id) {
            return Some(block.clone());
        }

        // Check ScanBuffer — if found, promote to HotSet.
        let scan = self.scan_buffer.get_mut(node);
        if let Some(block) = scan.remove(&block_id) {
            hot.put(block_id, block.clone());
            return Some(block);
        }

        None
    }

    /// Get a block for a sequential scan. Checks ScanBuffer first, then HotSet.
    /// Does NOT promote from HotSet to ScanBuffer.
    /// Returns `None` if the block is not cached.
    pub fn get_for_scan(&mut self, block_id: BlockId, node: usize) -> Option<CachedBlock> {
        let scan = self.scan_buffer.get_mut(node);

        // Check ScanBuffer first.
        if let Some(block) = scan.get(&block_id) {
            return Some(block.clone());
        }

        // Check HotSet — return if found but don't move it.
        let hot = self.hot_set.get_mut(node);
        if let Some(block) = hot.get(&block_id) {
            return Some(block.clone());
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
                let hot = self.hot_set.get_mut(node);
                hot.put(block_id, block);
            }
            AccessType::SequentialScan => {
                // Collect the set of block IDs currently in the HotSet for this node,
                // so the ScanBuffer can avoid evicting them.
                let hot = self.hot_set.get_mut(node);
                let hot_set_ids: HashSet<BlockId> = hot.keys().collect();

                let scan = self.scan_buffer.get_mut(node);
                scan.put_with_constraint(block_id, block, &hot_set_ids);
            }
        }
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
