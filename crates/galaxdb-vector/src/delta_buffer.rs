//! Delta buffer for recent vector inserts and deletes.
//!
//! The delta buffer holds vectors that have been inserted or deleted since
//! the last HNSW merge. During search, the delta buffer is searched with
//! exact brute-force k-NN, and results are unioned with HNSW candidates
//! before re-ranking.
//!
//! Delta buffer entries are WAL-backed (DELTA_INSERT, DELTA_TOMBSTONE record
//! types) for crash recovery. On recovery, WAL delta records are replayed
//! to rebuild the buffer.
//!
//! Merge is triggered when:
//! - Delta buffer size exceeds max(10_000, total_indexed × 0.01)
//! - Tombstones exceed 20% of total indexed vectors (emergency merge)

use std::collections::HashSet;
use std::sync::RwLock;

use crate::distance::cosine_distance;

/// A vector entry in the delta buffer.
#[derive(Debug, Clone)]
struct DeltaEntry {
    /// External row ID from the storage engine.
    row_id: u64,
    /// The raw f32 vector.
    vector: Vec<f32>,
}

/// Thread-safe delta buffer for recent vector changes.
///
/// Supports concurrent reads (brute-force search) and exclusive writes
/// (insert/delete) via RwLock.
pub struct DeltaBuffer {
    /// Recently inserted vectors, not yet in the base HNSW graph.
    inner: RwLock<DeltaBufferInner>,
    /// Vector dimensionality.
    dim: usize,
}

struct DeltaBufferInner {
    /// Inserted vectors pending merge into HNSW.
    vectors: Vec<DeltaEntry>,
    /// Row IDs that have been deleted. These are excluded from search
    /// results even if they exist in the base HNSW graph.
    tombstones: HashSet<u64>,
}

/// Result of a delta buffer search.
#[derive(Debug, Clone)]
pub struct DeltaSearchResult {
    pub row_id: u64,
    pub distance: f32,
}

impl DeltaBuffer {
    /// Create a new empty delta buffer.
    pub fn new(dim: usize) -> Self {
        Self {
            inner: RwLock::new(DeltaBufferInner {
                vectors: Vec::new(),
                tombstones: HashSet::new(),
            }),
            dim,
        }
    }

    /// Insert a vector into the delta buffer.
    ///
    /// The vector will be found by brute-force search until the next
    /// HNSW merge incorporates it into the base graph.
    pub fn insert(&self, row_id: u64, vector: Vec<f32>) {
        assert_eq!(vector.len(), self.dim, "vector dimension mismatch");
        let mut inner = self.inner.write().expect("delta buffer write lock");
        // Remove from tombstones if re-inserting a deleted row
        inner.tombstones.remove(&row_id);
        inner.vectors.push(DeltaEntry { row_id, vector });
    }

    /// Mark a row as deleted (tombstone).
    ///
    /// The tombstone ensures this row_id is excluded from search results
    /// even if it still exists in the base HNSW graph. The tombstone is
    /// cleared during the next HNSW merge.
    pub fn delete(&self, row_id: u64) {
        let mut inner = self.inner.write().expect("delta buffer write lock");
        inner.tombstones.insert(row_id);
        // Also remove from pending vectors if it was recently inserted
        inner.vectors.retain(|e| e.row_id != row_id);
    }

    /// Exact brute-force k-NN search over the delta buffer.
    ///
    /// Returns the k nearest vectors to the query, excluding tombstoned rows.
    /// This is O(n) where n is the delta buffer size — acceptable because
    /// the buffer is small (< 10K vectors between merges).
    pub fn search(&self, query: &[f32], k: usize) -> Vec<DeltaSearchResult> {
        assert_eq!(query.len(), self.dim, "query dimension mismatch");
        let inner = self.inner.read().expect("delta buffer read lock");

        let mut results: Vec<DeltaSearchResult> = inner
            .vectors
            .iter()
            .filter(|e| !inner.tombstones.contains(&e.row_id))
            .map(|e| DeltaSearchResult {
                row_id: e.row_id,
                distance: cosine_distance(query, &e.vector),
            })
            .collect();

        // Sort by distance (nearest first) and take top-k
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        results.truncate(k);
        results
    }

    /// Check if a row_id is tombstoned (deleted).
    pub fn is_tombstoned(&self, row_id: u64) -> bool {
        let inner = self.inner.read().expect("delta buffer read lock");
        inner.tombstones.contains(&row_id)
    }

    /// Number of vectors in the delta buffer (excluding tombstones).
    pub fn vector_count(&self) -> usize {
        let inner = self.inner.read().expect("delta buffer read lock");
        inner.vectors.len()
    }

    /// Number of tombstones.
    pub fn tombstone_count(&self) -> usize {
        let inner = self.inner.read().expect("delta buffer read lock");
        inner.tombstones.len()
    }

    /// Check if a merge should be triggered.
    ///
    /// Merge triggers:
    /// 1. Delta buffer size >= max(10_000, total_indexed × 0.01)
    /// 2. Tombstones > 20% of total_indexed (emergency merge)
    pub fn should_merge(&self, total_indexed: usize) -> bool {
        let inner = self.inner.read().expect("delta buffer read lock");
        let threshold = 10_000usize.max(total_indexed / 100);
        let emergency = inner.tombstones.len() as f64 > total_indexed as f64 * 0.20;
        inner.vectors.len() >= threshold || emergency
    }

    /// Drain the delta buffer, returning all vectors and tombstones.
    ///
    /// Called during HNSW merge to incorporate delta entries into the
    /// new base graph. After draining, the buffer is empty.
    pub fn drain(&self) -> (Vec<(u64, Vec<f32>)>, HashSet<u64>) {
        let mut inner = self.inner.write().expect("delta buffer write lock");
        let vectors: Vec<(u64, Vec<f32>)> = inner
            .vectors
            .drain(..)
            .map(|e| (e.row_id, e.vector))
            .collect();
        let tombstones = std::mem::take(&mut inner.tombstones);
        (vectors, tombstones)
    }

    /// Get all vectors (for merge). Does not drain.
    pub fn vectors(&self) -> Vec<(u64, Vec<f32>)> {
        let inner = self.inner.read().expect("delta buffer read lock");
        inner.vectors.iter().map(|e| (e.row_id, e.vector.clone())).collect()
    }

    /// Get all tombstones (for merge). Does not drain.
    pub fn tombstones(&self) -> HashSet<u64> {
        let inner = self.inner.read().expect("delta buffer read lock");
        inner.tombstones.clone()
    }

    /// Clear the buffer (after successful merge).
    pub fn clear(&self) {
        let mut inner = self.inner.write().expect("delta buffer write lock");
        inner.vectors.clear();
        inner.tombstones.clear();
    }
}

/// Union HNSW candidates with delta buffer candidates, then re-rank.
///
/// This is the core search pipeline:
/// 1. HNSW search returns approximate candidates with distances
/// 2. Delta buffer brute-force search returns exact candidates
/// 3. Union both sets, deduplicate by row_id
/// 4. Re-rank by exact cosine distance against the query
/// 5. Return top-k
///
/// The `fetch_vector` callback retrieves the raw vector for a row_id
/// from PAX blocks (for re-ranking HNSW candidates with exact distances).
pub fn union_and_rerank<F>(
    hnsw_candidates: &[(u64, f32)],
    delta_candidates: &[DeltaSearchResult],
    tombstones: &HashSet<u64>,
    query: &[f32],
    k: usize,
    fetch_vector: F,
) -> Vec<(u64, f32)>
where
    F: Fn(u64) -> Option<Vec<f32>>,
{
    let mut seen = HashSet::new();
    let mut all_candidates: Vec<(u64, f32)> = Vec::new();

    // Add HNSW candidates (re-rank with exact distance if vector available)
    for &(row_id, approx_dist) in hnsw_candidates {
        if tombstones.contains(&row_id) || seen.contains(&row_id) {
            continue;
        }
        seen.insert(row_id);

        // Re-rank: fetch raw vector and compute exact cosine distance
        let exact_dist = if let Some(vec) = fetch_vector(row_id) {
            cosine_distance(query, &vec)
        } else {
            approx_dist // fallback to approximate if vector not available
        };
        all_candidates.push((row_id, exact_dist));
    }

    // Add delta buffer candidates (already exact distances)
    for result in delta_candidates {
        if tombstones.contains(&result.row_id) || seen.contains(&result.row_id) {
            continue;
        }
        seen.insert(result.row_id);
        all_candidates.push((result.row_id, result.distance));
    }

    // Sort by distance (nearest first) and take top-k
    all_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    all_candidates.truncate(k);
    all_candidates
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_search() {
        let buf = DeltaBuffer::new(3);
        buf.insert(1, vec![1.0, 0.0, 0.0]);
        buf.insert(2, vec![0.0, 1.0, 0.0]);
        buf.insert(3, vec![0.9, 0.1, 0.0]);

        let results = buf.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].row_id, 1); // exact match
        assert!(results[0].distance < 0.01);
        assert_eq!(results[1].row_id, 3); // closest
    }

    #[test]
    fn tombstone_excludes_from_search() {
        let buf = DeltaBuffer::new(3);
        buf.insert(1, vec![1.0, 0.0, 0.0]);
        buf.insert(2, vec![0.9, 0.1, 0.0]);
        buf.delete(1); // tombstone the exact match

        let results = buf.search(&[1.0, 0.0, 0.0], 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].row_id, 2); // only non-tombstoned result
    }

    #[test]
    fn delete_removes_pending_insert() {
        let buf = DeltaBuffer::new(3);
        buf.insert(1, vec![1.0, 0.0, 0.0]);
        assert_eq!(buf.vector_count(), 1);

        buf.delete(1);
        assert_eq!(buf.vector_count(), 0);
        assert_eq!(buf.tombstone_count(), 1);
    }

    #[test]
    fn reinsert_after_delete() {
        let buf = DeltaBuffer::new(3);
        buf.insert(1, vec![1.0, 0.0, 0.0]);
        buf.delete(1);
        buf.insert(1, vec![0.0, 1.0, 0.0]); // re-insert with different vector

        assert_eq!(buf.vector_count(), 1);
        assert_eq!(buf.tombstone_count(), 0); // tombstone cleared
        assert!(!buf.is_tombstoned(1));

        let results = buf.search(&[0.0, 1.0, 0.0], 1);
        assert_eq!(results[0].row_id, 1);
        assert!(results[0].distance < 0.01);
    }

    #[test]
    fn drain_empties_buffer() {
        let buf = DeltaBuffer::new(3);
        buf.insert(1, vec![1.0, 0.0, 0.0]);
        buf.insert(2, vec![0.0, 1.0, 0.0]);
        buf.delete(3);

        let (vectors, tombstones) = buf.drain();
        assert_eq!(vectors.len(), 2);
        assert_eq!(tombstones.len(), 1);
        assert!(tombstones.contains(&3));

        assert_eq!(buf.vector_count(), 0);
        assert_eq!(buf.tombstone_count(), 0);
    }

    #[test]
    fn should_merge_threshold() {
        let buf = DeltaBuffer::new(3);
        // With total_indexed=1_000_000, threshold = max(10_000, 10_000) = 10_000
        assert!(!buf.should_merge(1_000_000));

        // Insert 10_000 vectors
        for i in 0..10_000 {
            buf.insert(i, vec![i as f32, 0.0, 0.0]);
        }
        assert!(buf.should_merge(1_000_000));
    }

    #[test]
    fn should_merge_emergency_tombstones() {
        let buf = DeltaBuffer::new(3);
        // With total_indexed=100, emergency at > 20 tombstones
        for i in 0..21 {
            buf.delete(i);
        }
        assert!(buf.should_merge(100));
    }

    #[test]
    fn union_and_rerank_deduplicates() {
        let hnsw = vec![(1u64, 0.1f32), (2, 0.2), (3, 0.3)];
        let delta = vec![
            DeltaSearchResult { row_id: 2, distance: 0.15 }, // duplicate
            DeltaSearchResult { row_id: 4, distance: 0.05 }, // new
        ];
        let tombstones = HashSet::new();
        let query = [1.0, 0.0, 0.0];

        let results = union_and_rerank(
            &hnsw,
            &delta,
            &tombstones,
            &query,
            3,
            |_| None, // no re-ranking vectors available
        );

        assert_eq!(results.len(), 3);
        // Should be sorted by distance
        assert!(results[0].1 <= results[1].1);
        assert!(results[1].1 <= results[2].1);
        // row_id 4 should be first (distance 0.05)
        assert_eq!(results[0].0, 4);
    }

    #[test]
    fn union_and_rerank_excludes_tombstones() {
        let hnsw = vec![(1u64, 0.1f32), (2, 0.2)];
        let delta = vec![];
        let mut tombstones = HashSet::new();
        tombstones.insert(1); // tombstone row 1

        let results = union_and_rerank(
            &hnsw,
            &delta,
            &tombstones,
            &[1.0, 0.0, 0.0],
            5,
            |_| None,
        );

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 2); // row 1 excluded
    }

    #[test]
    fn search_empty_buffer() {
        let buf = DeltaBuffer::new(3);
        let results = buf.search(&[1.0, 0.0, 0.0], 5);
        assert!(results.is_empty());
    }
}
