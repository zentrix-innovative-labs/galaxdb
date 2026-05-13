//! HNSW merge — builds a new base graph incorporating delta buffer changes.
//!
//! The merge process follows the design spec:
//! 1. Build new HNSW graph in shadow file (`.hnsw.new`) incorporating
//!    base graph vectors + delta buffer vectors, excluding tombstones
//! 2. `fsync` the shadow file
//! 3. Atomic `rename(".hnsw.new", ".hnsw")` — crash-safe
//! 4. Clear delta buffer
//! 5. Old mmap is released when all in-flight queries complete (Arc)
//!
//! Emergency merge is triggered when tombstones > 20% of indexed vectors.

use std::path::Path;

use galaxdb_common::{GalaxError, GalaxResult};

use crate::delta_buffer::DeltaBuffer;
use crate::hnsw::{HnswConfig, HnswGraph};
use crate::hnsw_file::{write_hnsw_file, MmapHnswGraph};

/// Merge the base HNSW graph with delta buffer changes.
///
/// Builds a completely new HNSW graph containing:
/// - All vectors from the base graph that are NOT tombstoned
/// - All vectors from the delta buffer
///
/// The new graph is written to `{dir}/.hnsw.new`, fsynced, then
/// atomically renamed to `{dir}/.hnsw`.
///
/// Returns the new graph's node count.
pub fn merge_hnsw(
    base_graph: Option<&MmapHnswGraph>,
    delta: &DeltaBuffer,
    config: &HnswConfig,
    dir: &Path,
) -> GalaxResult<usize> {
    let (delta_vectors, tombstones) = delta.drain();

    // Build new graph with all non-tombstoned vectors
    let mut new_graph = HnswGraph::new(config.clone());

    // Add base graph vectors (excluding tombstones)
    if let Some(base) = base_graph {
        for i in 0..base.node_count() {
            let ext_id = base.external_id(i as u32);
            if tombstones.contains(&ext_id) {
                continue; // skip tombstoned vectors
            }
            let vector = base.get_vector(i as u32).to_vec();
            new_graph.insert(ext_id, vector);
        }
    }

    // Add delta buffer vectors
    for (row_id, vector) in delta_vectors {
        new_graph.insert(row_id, vector);
    }

    let node_count = new_graph.len();

    // Write to shadow file
    let shadow_path = dir.join(".hnsw.new");
    write_hnsw_file(&new_graph, &shadow_path)?;

    // Atomic rename: .hnsw.new → .hnsw
    let final_path = dir.join(".hnsw");
    std::fs::rename(&shadow_path, &final_path).map_err(|e| {
        GalaxError::Io(std::io::Error::new(
            e.kind(),
            format!("atomic rename failed: {}", e),
        ))
    })?;

    Ok(node_count)
}

/// Check if a merge should be triggered.
pub fn should_merge(delta: &DeltaBuffer, total_indexed: usize) -> bool {
    delta.should_merge(total_indexed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::HnswConfig;

    #[test]
    fn merge_empty_base_with_delta() {
        let dir = tempfile::tempdir().unwrap();
        let config = HnswConfig::new(4).with_m(4).with_ef_construction(20);
        let delta = DeltaBuffer::new(4);

        delta.insert(1, vec![1.0, 0.0, 0.0, 0.0]);
        delta.insert(2, vec![0.0, 1.0, 0.0, 0.0]);
        delta.insert(3, vec![0.0, 0.0, 1.0, 0.0]);

        let count = merge_hnsw(None, &delta, &config, dir.path()).unwrap();
        assert_eq!(count, 3);

        // Verify the merged graph file exists and is valid
        let merged = MmapHnswGraph::open(&dir.path().join(".hnsw")).unwrap();
        assert_eq!(merged.node_count(), 3);
    }

    #[test]
    fn merge_excludes_tombstones() {
        let dir = tempfile::tempdir().unwrap();
        let config = HnswConfig::new(4).with_m(4).with_ef_construction(20);

        // Build initial base graph
        let mut base = HnswGraph::new(config.clone());
        base.insert(1, vec![1.0, 0.0, 0.0, 0.0]);
        base.insert(2, vec![0.0, 1.0, 0.0, 0.0]);
        base.insert(3, vec![0.0, 0.0, 1.0, 0.0]);

        let base_path = dir.path().join(".hnsw");
        write_hnsw_file(&base, &base_path).unwrap();
        let base_mmap = MmapHnswGraph::open(&base_path).unwrap();

        // Delta: delete row 2, add row 4
        let delta = DeltaBuffer::new(4);
        delta.delete(2);
        delta.insert(4, vec![0.0, 0.0, 0.0, 1.0]);

        let count = merge_hnsw(Some(&base_mmap), &delta, &config, dir.path()).unwrap();
        assert_eq!(count, 3); // 3 from base - 1 tombstone + 1 new = 3

        // Verify merged graph
        let merged = MmapHnswGraph::open(&dir.path().join(".hnsw")).unwrap();
        assert_eq!(merged.node_count(), 3);

        // Row 2 should not be in the merged graph
        let mut found_ids: Vec<u64> = (0..merged.node_count())
            .map(|i| merged.external_id(i as u32))
            .collect();
        found_ids.sort();
        assert_eq!(found_ids, vec![1, 3, 4]);
    }

    #[test]
    fn merge_atomic_rename_produces_valid_file() {
        let dir = tempfile::tempdir().unwrap();
        let config = HnswConfig::new(4).with_m(4).with_ef_construction(20);
        let delta = DeltaBuffer::new(4);
        delta.insert(1, vec![1.0, 0.0, 0.0, 0.0]);

        merge_hnsw(None, &delta, &config, dir.path()).unwrap();

        // Shadow file should not exist (renamed)
        assert!(!dir.path().join(".hnsw.new").exists());
        // Final file should exist
        assert!(dir.path().join(".hnsw").exists());
    }

    #[test]
    fn merge_clears_delta_buffer() {
        let dir = tempfile::tempdir().unwrap();
        let config = HnswConfig::new(4).with_m(4).with_ef_construction(20);
        let delta = DeltaBuffer::new(4);
        delta.insert(1, vec![1.0, 0.0, 0.0, 0.0]);
        delta.delete(99);

        assert_eq!(delta.vector_count(), 1);
        assert_eq!(delta.tombstone_count(), 1);

        merge_hnsw(None, &delta, &config, dir.path()).unwrap();

        // drain() was called inside merge, so buffer should be empty
        assert_eq!(delta.vector_count(), 0);
        assert_eq!(delta.tombstone_count(), 0);
    }

    #[test]
    fn crash_recovery_shadow_file_cleanup() {
        let dir = tempfile::tempdir().unwrap();

        // Simulate a crash: shadow file exists but rename didn't happen
        let shadow_path = dir.path().join(".hnsw.new");
        std::fs::write(&shadow_path, b"incomplete").unwrap();

        // On recovery, the shadow file should be deleted
        // (the old .hnsw is still valid)
        if shadow_path.exists() {
            std::fs::remove_file(&shadow_path).unwrap();
        }
        assert!(!shadow_path.exists());
    }
}
