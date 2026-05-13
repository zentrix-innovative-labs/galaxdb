//! Merkle DAG — content-addressed version history for time-travel queries.
//!
//! Each commit creates a MerkleRoot: the XXH3-128 hash over all PAX block
//! checksums included in that version. This enables:
//! - `AT VERSION timestamp` — filter blocks by commit_timestamp
//! - `AT VERSION 'tag_name'` — resolve tag to exact block set
//! - Integrity verification — detect corruption by recomputing root hash
//!
//! The DAG is append-only: new commits reference previous roots.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3::xxh3_128;

/// 128-bit Merkle root hash (XXH3-128).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MerkleRoot {
    pub hash: u128,
}

impl MerkleRoot {
    /// Compute a Merkle root from a set of block checksums.
    /// The checksums are sorted for deterministic ordering, then hashed together.
    pub fn compute(block_checksums: &[u64]) -> Self {
        let mut sorted = block_checksums.to_vec();
        sorted.sort();

        // Concatenate all checksums as bytes and hash
        let mut data = Vec::with_capacity(sorted.len() * 8);
        for checksum in &sorted {
            data.extend_from_slice(&checksum.to_le_bytes());
        }

        let hash = xxh3_128(&data);
        Self { hash }
    }

    /// Empty root (no blocks).
    pub fn empty() -> Self {
        Self { hash: 0 }
    }

    /// Check if this is an empty root.
    pub fn is_empty(&self) -> bool {
        self.hash == 0
    }
}

impl std::fmt::Display for MerkleRoot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:032x}", self.hash)
    }
}

/// A single version entry in the Merkle DAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionEntry {
    /// Commit timestamp (monotonically increasing).
    pub timestamp: u64,
    /// Merkle root hash for this version.
    pub root: MerkleRoot,
    /// Block checksums included in this version.
    pub block_checksums: Vec<u64>,
    /// Block IDs included in this version (for AT VERSION queries).
    pub block_ids: Vec<u64>,
    /// Parent version timestamp (0 for first commit).
    pub parent_timestamp: u64,
}

/// The Merkle DAG — ordered history of all committed versions.
///
/// Supports:
/// - Adding new versions (on commit)
/// - Looking up versions by timestamp (AT VERSION timestamp)
/// - Looking up the latest version
/// - Filtering blocks visible at a given timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleDag {
    /// All versions ordered by timestamp.
    versions: BTreeMap<u64, VersionEntry>,
}

impl MerkleDag {
    /// Create a new empty Merkle DAG.
    pub fn new() -> Self {
        Self {
            versions: BTreeMap::new(),
        }
    }

    /// Record a new version (called on each successful commit/flush).
    ///
    /// `block_checksums` are the XXH3-64 checksums of all PAX blocks in this version.
    /// `block_ids` are the corresponding block IDs for AT VERSION filtering.
    pub fn commit(
        &mut self,
        timestamp: u64,
        block_checksums: Vec<u64>,
        block_ids: Vec<u64>,
    ) -> MerkleRoot {
        let root = MerkleRoot::compute(&block_checksums);
        let parent_timestamp = self.versions.keys().next_back().copied().unwrap_or(0);

        let entry = VersionEntry {
            timestamp,
            root,
            block_checksums,
            block_ids,
            parent_timestamp,
        };

        self.versions.insert(timestamp, entry);
        root
    }

    /// Get the version entry at a specific timestamp.
    pub fn get_version(&self, timestamp: u64) -> Option<&VersionEntry> {
        self.versions.get(&timestamp)
    }

    /// Get the latest version.
    pub fn latest(&self) -> Option<&VersionEntry> {
        self.versions.values().next_back()
    }

    /// Get the latest Merkle root.
    pub fn latest_root(&self) -> MerkleRoot {
        self.latest().map(|v| v.root).unwrap_or(MerkleRoot::empty())
    }

    /// Get all block IDs visible at a given timestamp.
    /// Returns blocks from all versions with commit_timestamp <= target.
    pub fn blocks_at_version(&self, timestamp: u64) -> Vec<u64> {
        let mut all_blocks = Vec::new();
        for (ts, entry) in &self.versions {
            if *ts <= timestamp {
                all_blocks.extend_from_slice(&entry.block_ids);
            }
        }
        all_blocks.sort();
        all_blocks.dedup();
        all_blocks
    }

    /// Get the version entry closest to (but not exceeding) a timestamp.
    pub fn version_at_or_before(&self, timestamp: u64) -> Option<&VersionEntry> {
        self.versions.range(..=timestamp).next_back().map(|(_, v)| v)
    }

    /// Get all version timestamps.
    pub fn timestamps(&self) -> Vec<u64> {
        self.versions.keys().copied().collect()
    }

    /// Number of versions in the DAG.
    pub fn version_count(&self) -> usize {
        self.versions.len()
    }

    /// Verify integrity: recompute root hash and compare.
    pub fn verify(&self, timestamp: u64) -> bool {
        if let Some(entry) = self.versions.get(&timestamp) {
            let recomputed = MerkleRoot::compute(&entry.block_checksums);
            recomputed == entry.root
        } else {
            false
        }
    }

    /// Get all versions (for serialization/persistence).
    pub fn all_versions(&self) -> Vec<&VersionEntry> {
        self.versions.values().collect()
    }
}

impl Default for MerkleDag {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_dag() {
        let dag = MerkleDag::new();
        assert_eq!(dag.version_count(), 0);
        assert!(dag.latest().is_none());
        assert!(dag.latest_root().is_empty());
    }

    #[test]
    fn single_commit() {
        let mut dag = MerkleDag::new();
        let checksums = vec![111, 222, 333];
        let block_ids = vec![1, 2, 3];

        let root = dag.commit(1000, checksums.clone(), block_ids.clone());

        assert!(!root.is_empty());
        assert_eq!(dag.version_count(), 1);
        assert_eq!(dag.latest().unwrap().timestamp, 1000);
        assert_eq!(dag.latest_root(), root);
        assert_eq!(dag.blocks_at_version(1000), vec![1, 2, 3]);
    }

    #[test]
    fn multiple_commits() {
        let mut dag = MerkleDag::new();

        dag.commit(100, vec![11, 22], vec![1, 2]);
        dag.commit(200, vec![33, 44], vec![3, 4]);
        dag.commit(300, vec![55], vec![5]);

        assert_eq!(dag.version_count(), 3);

        // Blocks at version 200 should include versions 100 and 200
        let blocks = dag.blocks_at_version(200);
        assert_eq!(blocks, vec![1, 2, 3, 4]);

        // Blocks at version 300 should include all
        let blocks = dag.blocks_at_version(300);
        assert_eq!(blocks, vec![1, 2, 3, 4, 5]);

        // Blocks at version 150 should only include version 100
        let blocks = dag.blocks_at_version(150);
        assert_eq!(blocks, vec![1, 2]);
    }

    #[test]
    fn deterministic_root_hash() {
        // Same checksums in different order should produce same root
        let root1 = MerkleRoot::compute(&[100, 200, 300]);
        let root2 = MerkleRoot::compute(&[300, 100, 200]);
        assert_eq!(root1, root2);
    }

    #[test]
    fn different_checksums_different_root() {
        let root1 = MerkleRoot::compute(&[100, 200, 300]);
        let root2 = MerkleRoot::compute(&[100, 200, 301]);
        assert_ne!(root1, root2);
    }

    #[test]
    fn verify_integrity() {
        let mut dag = MerkleDag::new();
        dag.commit(1000, vec![11, 22, 33], vec![1, 2, 3]);

        assert!(dag.verify(1000));
        assert!(!dag.verify(9999)); // non-existent version
    }

    #[test]
    fn parent_timestamp_tracking() {
        let mut dag = MerkleDag::new();
        dag.commit(100, vec![1], vec![1]);
        dag.commit(200, vec![2], vec![2]);

        let v1 = dag.get_version(100).unwrap();
        assert_eq!(v1.parent_timestamp, 0); // first commit has no parent

        let v2 = dag.get_version(200).unwrap();
        assert_eq!(v2.parent_timestamp, 100);
    }

    #[test]
    fn version_at_or_before() {
        let mut dag = MerkleDag::new();
        dag.commit(100, vec![1], vec![1]);
        dag.commit(200, vec![2], vec![2]);
        dag.commit(300, vec![3], vec![3]);

        let v = dag.version_at_or_before(250).unwrap();
        assert_eq!(v.timestamp, 200);

        let v = dag.version_at_or_before(300).unwrap();
        assert_eq!(v.timestamp, 300);

        assert!(dag.version_at_or_before(50).is_none());
    }
}
