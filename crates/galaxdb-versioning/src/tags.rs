//! Version Tags — named snapshots for time-travel and training export.
//!
//! A version tag captures a MerkleRoot at a point in time. Tagged versions:
//! - Are GC-exempt (pinned blocks never compacted away)
//! - Can be queried with `AT VERSION 'tag_name'`
//! - Can be exported for training with `FOR TRAINING` metadata

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::merkle::MerkleRoot;

/// Training-specific metadata for a version tag.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingTagMetadata {
    /// Quantization precision for export: "float32", "sq8", "rabitq"
    pub precision: String,
    /// Random seed for deterministic ordering
    pub seed: Option<u64>,
    /// Whether to sort by primary key for deterministic iteration
    pub deterministic_order: bool,
}

/// A named version tag.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionTag {
    /// Tag name (user-provided, unique)
    pub name: String,
    /// Timestamp when the tag was created
    pub created_at: u64,
    /// The Merkle root this tag points to
    pub root: MerkleRoot,
    /// The commit timestamp this tag references
    pub version_timestamp: u64,
    /// Block IDs pinned by this tag (GC-exempt)
    pub pinned_blocks: Vec<u64>,
    /// Whether this is a training tag
    pub for_training: bool,
    /// Training-specific metadata (if for_training=true)
    pub training_opts: Option<TrainingTagMetadata>,
}

/// Version tag catalog — manages all named tags.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TagCatalog {
    tags: HashMap<String, VersionTag>,
}

impl TagCatalog {
    pub fn new() -> Self {
        Self { tags: HashMap::new() }
    }

    /// Create a new version tag.
    pub fn create_tag(
        &mut self,
        name: String,
        created_at: u64,
        root: MerkleRoot,
        version_timestamp: u64,
        pinned_blocks: Vec<u64>,
        for_training: bool,
        training_opts: Option<TrainingTagMetadata>,
    ) -> Result<&VersionTag, String> {
        if self.tags.contains_key(&name) {
            return Err(format!("version tag '{}' already exists", name));
        }

        let tag = VersionTag {
            name: name.clone(),
            created_at,
            root,
            version_timestamp,
            pinned_blocks,
            for_training,
            training_opts,
        };

        self.tags.insert(name.clone(), tag);
        Ok(self.tags.get(&name).unwrap())
    }

    /// Get a tag by name.
    pub fn get_tag(&self, name: &str) -> Option<&VersionTag> {
        self.tags.get(name)
    }

    /// Check if a block is pinned by any tag (GC-exempt).
    pub fn is_block_pinned(&self, block_id: u64) -> bool {
        self.tags.values().any(|tag| tag.pinned_blocks.contains(&block_id))
    }

    /// Get all pinned block IDs across all tags.
    pub fn all_pinned_blocks(&self) -> Vec<u64> {
        let mut blocks: Vec<u64> = self.tags.values()
            .flat_map(|tag| tag.pinned_blocks.iter().copied())
            .collect();
        blocks.sort();
        blocks.dedup();
        blocks
    }

    /// Get every commit timestamp currently pinned by a version tag.
    /// Compaction consumes this set via
    /// `galaxdb_storage::compaction::GcContext::with_pins` so MVCC GC
    /// retains the exact versions that tagged snapshots reference
    /// (tasks 33.5 and 10.5). Duplicates are removed; the iteration
    /// order is unspecified.
    pub fn all_pinned_timestamps(&self) -> Vec<u64> {
        let mut stamps: Vec<u64> = self
            .tags
            .values()
            .map(|tag| tag.version_timestamp)
            .collect();
        stamps.sort();
        stamps.dedup();
        stamps
    }

    /// List all tag names.
    pub fn list_tags(&self) -> Vec<&str> {
        self.tags.keys().map(|s| s.as_str()).collect()
    }

    /// Number of tags.
    pub fn tag_count(&self) -> usize {
        self.tags.len()
    }

    /// Delete a tag (unpins its blocks).
    pub fn delete_tag(&mut self, name: &str) -> Option<VersionTag> {
        self.tags.remove(name)
    }
}

/// Consistency mode for AT VERSION + SEMANTIC_MATCH queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsistencyMode {
    /// ROW_SNAPSHOT: no SEMANTIC_MATCH allowed (default for AT VERSION)
    RowSnapshot,
    /// SEMANTIC_FRESH: search current HNSW against historical rows (with warning)
    SemanticFresh,
    /// SEMANTIC_SNAPSHOT: not implemented (v2 feature)
    SemanticSnapshot,
}

/// Result of resolving an AT VERSION query.
#[derive(Debug, Clone)]
pub struct VersionResolution {
    /// Block IDs visible at this version
    pub block_ids: Vec<u64>,
    /// The Merkle root for verification
    pub root: MerkleRoot,
    /// Timestamp of the resolved version
    pub timestamp: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_get_tag() {
        let mut catalog = TagCatalog::new();
        let root = MerkleRoot { hash: 12345 };

        catalog.create_tag(
            "v1.0".to_string(), 1000, root, 999,
            vec![1, 2, 3], false, None,
        ).unwrap();

        let tag = catalog.get_tag("v1.0").unwrap();
        assert_eq!(tag.name, "v1.0");
        assert_eq!(tag.root, root);
        assert_eq!(tag.pinned_blocks, vec![1, 2, 3]);
        assert!(!tag.for_training);
    }

    #[test]
    fn duplicate_tag_fails() {
        let mut catalog = TagCatalog::new();
        let root = MerkleRoot { hash: 1 };

        catalog.create_tag("v1".to_string(), 100, root, 99, vec![], false, None).unwrap();
        let result = catalog.create_tag("v1".to_string(), 200, root, 199, vec![], false, None);
        assert!(result.is_err());
    }

    #[test]
    fn training_tag_metadata() {
        let mut catalog = TagCatalog::new();
        let root = MerkleRoot { hash: 999 };

        catalog.create_tag(
            "train-v1".to_string(), 1000, root, 999,
            vec![1, 2, 3], true,
            Some(TrainingTagMetadata {
                precision: "sq8".to_string(),
                seed: Some(42),
                deterministic_order: true,
            }),
        ).unwrap();

        let tag = catalog.get_tag("train-v1").unwrap();
        assert!(tag.for_training);
        let opts = tag.training_opts.as_ref().unwrap();
        assert_eq!(opts.precision, "sq8");
        assert_eq!(opts.seed, Some(42));
        assert!(opts.deterministic_order);
    }

    #[test]
    fn pinned_blocks() {
        let mut catalog = TagCatalog::new();
        let root = MerkleRoot { hash: 1 };

        catalog.create_tag("a".to_string(), 100, root, 99, vec![1, 2, 3], false, None).unwrap();
        catalog.create_tag("b".to_string(), 200, root, 199, vec![3, 4, 5], false, None).unwrap();

        assert!(catalog.is_block_pinned(1));
        assert!(catalog.is_block_pinned(3));
        assert!(catalog.is_block_pinned(5));
        assert!(!catalog.is_block_pinned(99));

        let all = catalog.all_pinned_blocks();
        assert_eq!(all, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn delete_tag_unpins() {
        let mut catalog = TagCatalog::new();
        let root = MerkleRoot { hash: 1 };

        catalog.create_tag("x".to_string(), 100, root, 99, vec![10, 20], false, None).unwrap();
        assert!(catalog.is_block_pinned(10));

        catalog.delete_tag("x");
        assert!(!catalog.is_block_pinned(10));
    }
}
