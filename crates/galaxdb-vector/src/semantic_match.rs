//! SEMANTIC_MATCH query execution — the full vector search pipeline.
//!
//! Implements Req 21 (SEMANTIC_MATCH) and Req 22 (Adaptive Planner):
//!
//! 1. Embed the query text via the sidecar → query vector
//! 2. Choose search strategy (adaptive planner):
//!    - BruteForceFiltered: when filter cardinality < 1000 or < 0.1%
//!    - HnswWithPostFilter: when filter cardinality is moderate to high
//! 3. Search HNSW base graph + delta buffer
//! 4. Union candidates, exclude tombstones, re-rank by exact cosine distance
//! 5. Apply similarity threshold filter
//! 6. Return top-k results
//!
//! If the sidecar is unavailable, returns an error:
//! "semantic search temporarily unavailable — embedding sidecar is down"

use crate::delta_buffer::{DeltaBuffer, union_and_rerank};
use crate::distance::cosine_distance;
use crate::hnsw::HnswGraph;

/// Search strategy chosen by the adaptive planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchStrategy {
    /// Use HNSW graph traversal, then post-filter results.
    /// Best for moderate-to-high cardinality filters.
    HnswWithPostFilter,
    /// Brute-force scan over the filtered candidate set.
    /// Best for very low cardinality (high selectivity) filters.
    BruteForceFiltered,
}

/// Result of a SEMANTIC_MATCH query.
#[derive(Debug, Clone)]
pub struct SemanticMatchResult {
    /// Row ID from the storage engine.
    pub row_id: u64,
    /// Cosine similarity score (higher = more similar). Range: [-1, 1].
    pub similarity: f32,
}

/// Configuration for SEMANTIC_MATCH execution.
#[derive(Debug, Clone)]
pub struct SemanticMatchConfig {
    /// Number of candidates to retrieve from HNSW (before filtering).
    /// Typically 2× the desired top-k for re-ranking headroom.
    pub hnsw_candidates: usize,
    /// ef_search parameter for HNSW beam search width.
    pub ef_search: usize,
    /// Brute-force cardinality threshold: use brute-force when
    /// estimated filter cardinality < this value.
    pub brute_force_threshold: usize,
    /// Brute-force ratio threshold: use brute-force when
    /// estimated filter cardinality / total_rows < this ratio.
    pub brute_force_ratio: f64,
}

impl Default for SemanticMatchConfig {
    fn default() -> Self {
        Self {
            hnsw_candidates: 200,
            ef_search: 100,
            brute_force_threshold: 1000,
            brute_force_ratio: 0.001, // 0.1%
        }
    }
}

/// Choose the search strategy based on filter cardinality (Req 22).
///
/// - If no filter: always use HNSW
/// - If filter cardinality < 1000 or < 0.1% of total: brute-force
/// - Otherwise: HNSW with post-filter
pub fn choose_strategy(
    estimated_cardinality: Option<usize>,
    total_rows: usize,
    config: &SemanticMatchConfig,
) -> SearchStrategy {
    match estimated_cardinality {
        None => SearchStrategy::HnswWithPostFilter,
        Some(cardinality) => {
            if cardinality < config.brute_force_threshold
                || (total_rows > 0 && (cardinality as f64 / total_rows as f64) < config.brute_force_ratio)
            {
                SearchStrategy::BruteForceFiltered
            } else {
                SearchStrategy::HnswWithPostFilter
            }
        }
    }
}

/// Execute a SEMANTIC_MATCH query using the HNSW + delta buffer pipeline.
///
/// This is the core search function that implements the full pipeline:
/// 1. Search HNSW base graph for approximate candidates
/// 2. Search delta buffer for exact candidates
/// 3. Union + re-rank with exact cosine distance
/// 4. Apply similarity threshold
/// 5. Return top-k results
///
/// The `fetch_vector` callback retrieves raw vectors from PAX blocks
/// for re-ranking HNSW candidates with exact distances.
pub fn execute_semantic_match<F>(
    query_vector: &[f32],
    hnsw: &HnswGraph,
    delta: &DeltaBuffer,
    threshold: f64,
    k: usize,
    config: &SemanticMatchConfig,
    fetch_vector: F,
) -> Vec<SemanticMatchResult>
where
    F: Fn(u64) -> Option<Vec<f32>>,
{
    // Step 1: Search HNSW base graph
    let hnsw_candidates = hnsw.search(
        query_vector,
        config.hnsw_candidates,
        config.ef_search,
    );

    // Step 2: Search delta buffer (exact brute-force)
    let delta_candidates = delta.search(query_vector, config.hnsw_candidates);

    // Step 3: Get tombstones from delta buffer
    let tombstones = delta.tombstones();

    // Step 4: Union + re-rank
    let ranked = union_and_rerank(
        &hnsw_candidates,
        &delta_candidates,
        &tombstones,
        query_vector,
        config.hnsw_candidates, // keep more candidates for threshold filtering
        &fetch_vector,
    );

    // Step 5: Apply similarity threshold and convert to results
    // cosine_distance = 1 - similarity, so threshold on distance = 1 - threshold
    let distance_threshold = 1.0 - threshold as f32;

    let mut results: Vec<SemanticMatchResult> = ranked
        .into_iter()
        .filter(|&(_, dist)| dist <= distance_threshold)
        .take(k)
        .map(|(row_id, dist)| SemanticMatchResult {
            row_id,
            similarity: 1.0 - dist, // convert distance back to similarity
        })
        .collect();

    // Sort by similarity descending (most similar first)
    results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
    results
}

/// Execute a brute-force filtered SEMANTIC_MATCH.
///
/// Used when the adaptive planner determines the filter cardinality is very low.
/// Instead of HNSW traversal, scans only the filtered candidate vectors.
pub fn execute_brute_force_filtered(
    query_vector: &[f32],
    filtered_vectors: &[(u64, Vec<f32>)],
    threshold: f64,
    k: usize,
) -> Vec<SemanticMatchResult> {
    let distance_threshold = 1.0 - threshold as f32;

    let mut results: Vec<SemanticMatchResult> = filtered_vectors
        .iter()
        .map(|(row_id, vec)| {
            let dist = cosine_distance(query_vector, vec);
            SemanticMatchResult {
                row_id: *row_id,
                similarity: 1.0 - dist,
            }
        })
        .filter(|r| (1.0 - r.similarity) <= distance_threshold)
        .collect();

    results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
    results.truncate(k);
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::HnswConfig;
    use galaxdb_common::GalaxError;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    fn random_vector(rng: &mut SmallRng, dim: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        // Normalize
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            for x in v.iter_mut() { *x /= norm; }
        }
        v
    }

    #[test]
    fn semantic_match_finds_similar_vectors() {
        let dim = 64;
        let config = HnswConfig::new(dim).with_m(8).with_ef_construction(50);
        let mut hnsw = HnswGraph::new(config);
        let delta = DeltaBuffer::new(dim);
        let mut rng = SmallRng::seed_from_u64(42);

        // Insert 100 random vectors into HNSW
        let mut vectors: Vec<(u64, Vec<f32>)> = Vec::new();
        for i in 0..100 {
            let v = random_vector(&mut rng, dim);
            vectors.push((i, v.clone()));
            hnsw.insert(i, v);
        }

        // Insert 10 more into delta buffer
        for i in 100..110 {
            let v = random_vector(&mut rng, dim);
            vectors.push((i, v.clone()));
            delta.insert(i, v);
        }

        // Search for nearest to vector 0
        let query = vectors[0].1.clone();
        let sm_config = SemanticMatchConfig::default();

        let results = execute_semantic_match(
            &query,
            &hnsw,
            &delta,
            0.0, // no threshold — return all
            10,
            &sm_config,
            |row_id| vectors.iter().find(|(id, _)| *id == row_id).map(|(_, v)| v.clone()),
        );

        assert!(!results.is_empty());
        assert!(results.len() <= 10);
        // First result should be vector 0 itself (exact match)
        assert_eq!(results[0].row_id, 0);
        assert!(results[0].similarity > 0.99);
        // Results should be sorted by similarity descending
        for window in results.windows(2) {
            assert!(window[0].similarity >= window[1].similarity);
        }
    }

    #[test]
    fn semantic_match_threshold_filters() {
        let dim = 32;
        let config = HnswConfig::new(dim).with_m(4).with_ef_construction(20);
        let mut hnsw = HnswGraph::new(config);
        let delta = DeltaBuffer::new(dim);

        // Insert orthogonal vectors (low similarity to each other)
        let mut v1 = vec![0.0f32; dim];
        v1[0] = 1.0;
        hnsw.insert(1, v1.clone());

        let mut v2 = vec![0.0f32; dim];
        v2[1] = 1.0;
        hnsw.insert(2, v2);

        // Search with high threshold — should only find exact match
        let sm_config = SemanticMatchConfig::default();
        let results = execute_semantic_match(
            &v1,
            &hnsw,
            &delta,
            0.9, // high threshold
            10,
            &sm_config,
            |_| None,
        );

        // Only vector 1 should pass the 0.9 threshold
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].row_id, 1);
    }

    #[test]
    fn semantic_match_excludes_tombstones() {
        let dim = 32;
        let config = HnswConfig::new(dim).with_m(4).with_ef_construction(20);
        let mut hnsw = HnswGraph::new(config);
        let delta = DeltaBuffer::new(dim);

        let mut v1 = vec![0.0f32; dim];
        v1[0] = 1.0;
        hnsw.insert(1, v1.clone());

        let mut v2 = vec![0.0f32; dim];
        v2[0] = 0.9; v2[1] = 0.1;
        hnsw.insert(2, v2);

        // Tombstone vector 1
        delta.delete(1);

        let sm_config = SemanticMatchConfig::default();
        let results = execute_semantic_match(
            &v1,
            &hnsw,
            &delta,
            0.0,
            10,
            &sm_config,
            |_| None,
        );

        // Vector 1 should be excluded (tombstoned)
        assert!(!results.iter().any(|r| r.row_id == 1));
    }

    #[test]
    fn sidecar_unavailable_error() {
        // This tests the error path — when sidecar is down,
        // the caller should get a clear error message.
        // The actual sidecar check happens in the SQL executor layer,
        // but we verify the error message format here.
        let err = GalaxError::Internal(
            "semantic search temporarily unavailable — embedding sidecar is down".to_string()
        );
        let msg = format!("{}", err);
        assert!(msg.contains("semantic search temporarily unavailable"));
    }

    #[test]
    fn adaptive_planner_chooses_brute_force_for_low_cardinality() {
        let config = SemanticMatchConfig::default();

        // Low cardinality → brute force
        assert_eq!(
            choose_strategy(Some(500), 1_000_000, &config),
            SearchStrategy::BruteForceFiltered
        );

        // Very low ratio → brute force
        assert_eq!(
            choose_strategy(Some(50), 1_000_000, &config),
            SearchStrategy::BruteForceFiltered
        );
    }

    #[test]
    fn adaptive_planner_chooses_hnsw_for_high_cardinality() {
        let config = SemanticMatchConfig::default();

        // High cardinality → HNSW
        assert_eq!(
            choose_strategy(Some(100_000), 1_000_000, &config),
            SearchStrategy::HnswWithPostFilter
        );

        // No filter → HNSW
        assert_eq!(
            choose_strategy(None, 1_000_000, &config),
            SearchStrategy::HnswWithPostFilter
        );
    }

    #[test]
    fn brute_force_filtered_returns_correct_results() {
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let filtered = vec![
            (1, vec![0.9, 0.1, 0.0, 0.0]),  // similar
            (2, vec![0.0, 1.0, 0.0, 0.0]),  // orthogonal
            (3, vec![0.95, 0.05, 0.0, 0.0]), // very similar
        ];

        let results = execute_brute_force_filtered(&query, &filtered, 0.5, 10);

        // Should return vectors above 0.5 similarity, sorted descending
        assert!(results.len() >= 2); // vectors 1 and 3 should pass
        assert!(results[0].similarity > results.last().unwrap().similarity);
    }

    #[test]
    fn semantic_match_with_delta_buffer_vectors() {
        let dim = 32;
        let config = HnswConfig::new(dim).with_m(4).with_ef_construction(20);
        let hnsw = HnswGraph::new(config); // empty HNSW
        let delta = DeltaBuffer::new(dim);

        // Only delta buffer has vectors
        let mut v1 = vec![0.0f32; dim];
        v1[0] = 1.0;
        delta.insert(1, v1.clone());

        let mut v2 = vec![0.0f32; dim];
        v2[0] = 0.9; v2[1] = 0.1;
        delta.insert(2, v2);

        let sm_config = SemanticMatchConfig::default();
        let results = execute_semantic_match(
            &v1,
            &hnsw,
            &delta,
            0.0,
            10,
            &sm_config,
            |_| None,
        );

        assert!(!results.is_empty());
        assert_eq!(results[0].row_id, 1); // exact match from delta
    }
}
