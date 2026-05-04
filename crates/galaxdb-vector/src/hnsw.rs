//! HNSW (Hierarchical Navigable Small World) graph index.
//!
//! Implementation follows Malkov & Yashunin (2018):
//! "Efficient and robust approximate nearest neighbor search using
//! Hierarchical Navigable Small World graphs"
//! https://arxiv.org/abs/1603.09320
//!
//! Key parameters:
//! - M: max edges per node (upper layers). Default 16.
//! - M0: max edges per node (layer 0). Default 2*M = 32.
//! - ef_construction: search width during insertion. Default 200.
//! - mL: level generation factor = 1/ln(M).
//!
//! The graph is built incrementally. Each inserted vector is assigned a
//! random maximum layer from a geometric distribution. Insertion navigates
//! from the top layer down, wiring edges at each layer using the diversity
//! heuristic (Algorithm 4 from the paper).

use std::collections::{BinaryHeap, HashSet};
use std::cmp::Ordering as CmpOrdering;

use crate::distance::cosine_distance;

/// HNSW graph configuration.
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Max edges per node in upper layers. Default: 16.
    pub m: usize,
    /// Max edges per node in layer 0. Default: 2*M.
    pub m0: usize,
    /// Search width during construction. Default: 200.
    pub ef_construction: usize,
    /// Level generation factor: 1/ln(M).
    pub ml: f64,
    /// Vector dimensionality.
    pub dim: usize,
}

impl HnswConfig {
    pub fn new(dim: usize) -> Self {
        let m = 16;
        Self {
            m,
            m0: m * 2,
            ef_construction: 200,
            ml: 1.0 / (m as f64).ln(),
            dim,
        }
    }

    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self.m0 = m * 2;
        self.ml = 1.0 / (m as f64).ln();
        self
    }

    pub fn with_ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Max edges for a given layer.
    fn max_edges(&self, layer: usize) -> usize {
        if layer == 0 { self.m0 } else { self.m }
    }
}

/// A node in the HNSW graph.
#[derive(Debug, Clone)]
struct HnswNode {
    /// The vector data (f32 × dim).
    vector: Vec<f32>,
    /// External ID (e.g., row_id from the storage engine).
    external_id: u64,
    /// Maximum layer this node is present in.
    max_layer: usize,
    /// Adjacency lists per layer. neighbors[layer] = vec of node indices.
    neighbors: Vec<Vec<u32>>,
}

/// Candidate entry for the search heaps.
#[derive(Debug, Clone)]
struct Candidate {
    distance: f32,
    node_idx: u32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl Eq for Candidate {}

/// Min-heap ordering (smallest distance first).
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        // Reverse for min-heap (BinaryHeap is max-heap by default)
        other.distance.partial_cmp(&self.distance).unwrap_or(CmpOrdering::Equal)
    }
}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

/// Max-heap candidate (for the result set W — farthest first).
#[derive(Debug, Clone)]
struct FarCandidate {
    distance: f32,
    node_idx: u32,
}

impl PartialEq for FarCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl Eq for FarCandidate {}

impl Ord for FarCandidate {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        self.distance.partial_cmp(&other.distance).unwrap_or(CmpOrdering::Equal)
    }
}
impl PartialOrd for FarCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

/// In-memory HNSW graph index.
///
/// Supports incremental insertion and k-NN search.
/// For persistence, use `HnswFile` (mmap'd format).
pub struct HnswGraph {
    config: HnswConfig,
    nodes: Vec<HnswNode>,
    /// Index of the entry point node (top of the hierarchy).
    entry_point: Option<u32>,
    /// Current maximum layer in the graph.
    max_layer: usize,
}

impl HnswGraph {
    /// Create a new empty HNSW graph.
    pub fn new(config: HnswConfig) -> Self {
        Self {
            config,
            nodes: Vec::new(),
            entry_point: None,
            max_layer: 0,
        }
    }

    /// Number of vectors in the graph.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Insert a vector into the graph.
    ///
    /// Follows Algorithm 1 from Malkov & Yashunin (2018):
    /// 1. Assign random max layer from geometric distribution
    /// 2. Navigate from top layer down to max_layer+1 (coarse, ef=1)
    /// 3. At each layer from max_layer down to 0, search with ef_construction
    ///    and wire edges using the diversity heuristic
    pub fn insert(&mut self, external_id: u64, vector: Vec<f32>) {
        assert_eq!(vector.len(), self.config.dim, "vector dimension mismatch");

        let node_idx = self.nodes.len() as u32;

        // Assign random max layer (geometric distribution)
        let node_layer = self.random_layer();

        // Create the node with empty neighbor lists
        let node = HnswNode {
            vector,
            external_id,
            max_layer: node_layer,
            neighbors: vec![Vec::new(); node_layer + 1],
        };

        // Push the node first so it's accessible during edge wiring.
        self.nodes.push(node);

        // First node — just set as entry point, no edges to wire
        if self.entry_point.is_none() {
            self.entry_point = Some(node_idx);
            self.max_layer = node_layer;
            return;
        }

        let mut ep_idx = self.entry_point.unwrap();

        // Phase 1: Navigate from top layer down to node_layer+1 (coarse, ef=1)
        // Find the closest node at each layer as the entry point for the next
        for layer in (node_layer + 1..=self.max_layer).rev() {
            let candidates = self.search_layer(&self.nodes[node_idx as usize].vector, ep_idx, 1, layer);
            if let Some(nearest) = candidates.first() {
                ep_idx = nearest.node_idx;
            }
        }

        // Phase 2: Insert at layers min(max_layer, node_layer) down to 0
        let insert_from = node_layer.min(self.max_layer);
        for layer in (0..=insert_from).rev() {
            let query_vec = self.nodes[node_idx as usize].vector.clone();
            let candidates = self.search_layer(
                &query_vec,
                ep_idx,
                self.config.ef_construction,
                layer,
            );

            // Select neighbors using diversity heuristic (Algorithm 4)
            let max_edges = self.config.max_edges(layer);
            let selected = self.select_neighbors_heuristic(&query_vec, &candidates, max_edges);

            // Wire bidirectional edges
            self.nodes[node_idx as usize].neighbors[layer] = selected.iter().map(|c| c.node_idx).collect();

            for neighbor in &selected {
                let n_idx = neighbor.node_idx as usize;
                self.nodes[n_idx].neighbors[layer].push(node_idx);

                // Prune neighbor's edges if over capacity
                let n_max = self.config.max_edges(layer);
                if self.nodes[n_idx].neighbors[layer].len() > n_max {
                    self.prune_neighbors(n_idx, layer, n_max);
                }
            }

            if let Some(nearest) = candidates.first() {
                ep_idx = nearest.node_idx;
            }
        }

        // Update entry point if new node is in a higher layer
        if node_layer > self.max_layer {
            self.entry_point = Some(node_idx);
            self.max_layer = node_layer;
        }
    }

    /// Search for the k nearest neighbors of a query vector.
    ///
    /// Follows Algorithm 5 from the paper:
    /// 1. Navigate from top layer down to layer 1 (coarse, ef=1)
    /// 2. Search layer 0 with ef=ef_search
    /// 3. Return top-k from the result set
    ///
    /// Returns (external_id, distance) pairs sorted by distance (nearest first).
    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<(u64, f32)> {
        assert_eq!(query.len(), self.config.dim, "query dimension mismatch");

        if self.entry_point.is_none() {
            return Vec::new();
        }

        let mut ep_idx = self.entry_point.unwrap();
        let ef = ef_search.max(k);

        // Phase 1: Coarse navigation from top to layer 1 (ef=1)
        for layer in (1..=self.max_layer).rev() {
            let candidates = self.search_layer(query, ep_idx, 1, layer);
            if let Some(nearest) = candidates.first() {
                ep_idx = nearest.node_idx;
            }
        }

        // Phase 2: Fine search at layer 0 with ef candidates
        let candidates = self.search_layer(query, ep_idx, ef, 0);

        // Return top-k results
        candidates
            .into_iter()
            .take(k)
            .map(|c| (self.nodes[c.node_idx as usize].external_id, c.distance))
            .collect()
    }

    /// Search within a single layer (Algorithm 2 from the paper).
    ///
    /// Best-first beam search with beam width `ef`.
    /// Returns candidates sorted by distance (nearest first).
    fn search_layer(
        &self,
        query: &[f32],
        entry_point: u32,
        ef: usize,
        layer: usize,
    ) -> Vec<Candidate> {
        let ep_dist = cosine_distance(query, &self.nodes[entry_point as usize].vector);

        let mut visited = HashSet::new();
        visited.insert(entry_point);

        // C = min-heap of candidates (nearest first)
        let mut candidates = BinaryHeap::new();
        candidates.push(Candidate { distance: ep_dist, node_idx: entry_point });

        // W = max-heap of results (farthest first, capped at ef)
        let mut results = BinaryHeap::new();
        results.push(FarCandidate { distance: ep_dist, node_idx: entry_point });

        while let Some(closest) = candidates.pop() {
            // Termination: if closest candidate is farther than farthest result, stop
            let farthest_dist = results.peek().map_or(f32::MAX, |f| f.distance);
            if closest.distance > farthest_dist {
                break;
            }

            // Expand neighbors of the closest candidate
            let node = &self.nodes[closest.node_idx as usize];
            if layer < node.neighbors.len() {
                for &neighbor_idx in &node.neighbors[layer] {
                    if visited.contains(&neighbor_idx) {
                        continue;
                    }
                    visited.insert(neighbor_idx);

                    let dist = cosine_distance(query, &self.nodes[neighbor_idx as usize].vector);
                    let farthest_dist = results.peek().map_or(f32::MAX, |f| f.distance);

                    if dist < farthest_dist || results.len() < ef {
                        candidates.push(Candidate { distance: dist, node_idx: neighbor_idx });
                        results.push(FarCandidate { distance: dist, node_idx: neighbor_idx });

                        if results.len() > ef {
                            results.pop(); // remove farthest
                        }
                    }
                }
            }
        }

        // Convert results to sorted vec (nearest first)
        let mut result_vec: Vec<Candidate> = results
            .into_iter()
            .map(|f| Candidate { distance: f.distance, node_idx: f.node_idx })
            .collect();
        result_vec.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(CmpOrdering::Equal));
        result_vec
    }

    /// Select neighbors using the diversity heuristic (Algorithm 4).
    ///
    /// Instead of simply picking the M nearest candidates, this heuristic
    /// ensures directional diversity: a candidate is only selected if it is
    /// closer to the query than to any already-selected neighbor. This prevents
    /// all edges from pointing in the same direction.
    fn select_neighbors_heuristic(
        &self,
        _query: &[f32],
        candidates: &[Candidate],
        max_neighbors: usize,
    ) -> Vec<Candidate> {
        let mut selected: Vec<Candidate> = Vec::with_capacity(max_neighbors);

        for candidate in candidates {
            if selected.len() >= max_neighbors {
                break;
            }

            let dist_to_query = candidate.distance;

            // Check if this candidate is closer to query than to any selected neighbor
            let mut is_diverse = true;
            for existing in &selected {
                let dist_to_existing = cosine_distance(
                    &self.nodes[candidate.node_idx as usize].vector,
                    &self.nodes[existing.node_idx as usize].vector,
                );
                if dist_to_existing < dist_to_query {
                    is_diverse = false;
                    break;
                }
            }

            if is_diverse {
                selected.push(candidate.clone());
            }
        }

        // If diversity heuristic didn't fill all slots, add nearest remaining
        if selected.len() < max_neighbors {
            for candidate in candidates {
                if selected.len() >= max_neighbors {
                    break;
                }
                if !selected.iter().any(|s| s.node_idx == candidate.node_idx) {
                    selected.push(candidate.clone());
                }
            }
        }

        selected
    }

    /// Prune a node's neighbor list to max_edges using the diversity heuristic.
    fn prune_neighbors(&mut self, node_idx: usize, layer: usize, max_edges: usize) {
        let node_vec = self.nodes[node_idx].vector.clone();
        let neighbor_indices: Vec<u32> = self.nodes[node_idx].neighbors[layer].clone();

        // Build candidates from current neighbors
        let mut candidates: Vec<Candidate> = neighbor_indices
            .iter()
            .map(|&n_idx| {
                let dist = cosine_distance(&node_vec, &self.nodes[n_idx as usize].vector);
                Candidate { distance: dist, node_idx: n_idx }
            })
            .collect();
        candidates.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(CmpOrdering::Equal));

        let selected = self.select_neighbors_heuristic(&node_vec, &candidates, max_edges);
        self.nodes[node_idx].neighbors[layer] = selected.iter().map(|c| c.node_idx).collect();
    }

    /// Generate a random layer for a new node using geometric distribution.
    /// P(layer = l) = (1 - p) * p^l where p = e^(-1/mL)
    fn random_layer(&self) -> usize {
        use rand::Rng;
        let uniform: f64 = rand::thread_rng().gen_range(0.0001..1.0);
        let layer = (-uniform.ln() * self.config.ml).floor() as usize;
        layer
    }

    /// Get the vector for a node by its index.
    pub fn get_vector(&self, node_idx: u32) -> Option<&[f32]> {
        self.nodes.get(node_idx as usize).map(|n| n.vector.as_slice())
    }

    /// Get the external ID for a node by its index.
    pub fn get_external_id(&self, node_idx: u32) -> Option<u64> {
        self.nodes.get(node_idx as usize).map(|n| n.external_id)
    }

    /// Get the max layer for a node.
    pub fn node_max_layer(&self, node_idx: u32) -> usize {
        self.nodes.get(node_idx as usize).map_or(0, |n| n.max_layer)
    }

    /// Get the neighbors of a node at a specific layer.
    pub fn get_neighbors(&self, node_idx: u32, layer: usize) -> &[u32] {
        self.nodes
            .get(node_idx as usize)
            .and_then(|n| n.neighbors.get(layer))
            .map_or(&[], |v| v.as_slice())
    }

    /// Get the config.
    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    /// Get the entry point node index.
    pub fn entry_point(&self) -> Option<u32> {
        self.entry_point
    }

    /// Get the current max layer.
    pub fn max_layer(&self) -> usize {
        self.max_layer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    fn random_vector(rng: &mut SmallRng, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    #[test]
    fn insert_single_vector() {
        let config = HnswConfig::new(4);
        let mut graph = HnswGraph::new(config);
        graph.insert(1, vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(graph.len(), 1);
        assert!(graph.entry_point().is_some());
    }

    #[test]
    fn insert_and_search_exact() {
        let config = HnswConfig::new(4).with_m(4).with_ef_construction(50);
        let mut graph = HnswGraph::new(config);

        // Insert 3 known vectors
        graph.insert(1, vec![1.0, 0.0, 0.0, 0.0]);
        graph.insert(2, vec![0.0, 1.0, 0.0, 0.0]);
        graph.insert(3, vec![0.9, 0.1, 0.0, 0.0]); // closest to [1,0,0,0]

        // Search for nearest to [1,0,0,0]
        let results = graph.search(&[1.0, 0.0, 0.0, 0.0], 2, 10);
        assert_eq!(results.len(), 2);
        // First result should be vector 1 (exact match)
        assert_eq!(results[0].0, 1);
        assert!(results[0].1 < 0.01);
        // Second should be vector 3 (closest)
        assert_eq!(results[1].0, 3);
    }

    #[test]
    fn search_empty_graph() {
        let config = HnswConfig::new(4);
        let graph = HnswGraph::new(config);
        let results = graph.search(&[1.0, 0.0, 0.0, 0.0], 5, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn insert_100_vectors_and_search() {
        let config = HnswConfig::new(32).with_m(8).with_ef_construction(100);
        let mut graph = HnswGraph::new(config);
        let mut rng = SmallRng::seed_from_u64(42);

        let mut vectors: Vec<Vec<f32>> = Vec::new();
        for i in 0..100 {
            let v = random_vector(&mut rng, 32);
            vectors.push(v.clone());
            graph.insert(i as u64, v);
        }

        assert_eq!(graph.len(), 100);

        // Search for nearest to the first vector
        let results = graph.search(&vectors[0], 5, 50);
        assert_eq!(results.len(), 5);
        // First result should be the vector itself
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 < 0.01);
    }

    #[test]
    fn recall_at_10_on_1000_vectors() {
        // This tests the quality of the HNSW index.
        // We insert 1000 random 128-dim vectors and check that
        // HNSW search finds at least 95% of the true 10 nearest neighbors.
        let dim = 128;
        let n = 1000;
        let k = 10;
        let ef_search = 100;

        let config = HnswConfig::new(dim).with_m(16).with_ef_construction(200);
        let mut graph = HnswGraph::new(config);
        let mut rng = SmallRng::seed_from_u64(123);

        let mut vectors: Vec<Vec<f32>> = Vec::new();
        for i in 0..n {
            let v = random_vector(&mut rng, dim);
            vectors.push(v.clone());
            graph.insert(i as u64, v);
        }

        // Pick 50 random queries and measure recall
        let num_queries = 50;
        let mut total_recall = 0.0;

        for _ in 0..num_queries {
            let query = random_vector(&mut rng, dim);

            // Brute-force ground truth
            let mut distances: Vec<(u64, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (i as u64, cosine_distance(&query, v)))
                .collect();
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let ground_truth: HashSet<u64> = distances.iter().take(k).map(|d| d.0).collect();

            // HNSW search
            let results = graph.search(&query, k, ef_search);
            let found: HashSet<u64> = results.iter().map(|r| r.0).collect();

            let recall = ground_truth.intersection(&found).count() as f64 / k as f64;
            total_recall += recall;
        }

        let avg_recall = total_recall / num_queries as f64;
        assert!(
            avg_recall >= 0.95,
            "recall@{} should be >= 0.95, got {:.3} (over {} queries)",
            k, avg_recall, num_queries
        );
    }

    #[test]
    fn layer_distribution_is_geometric() {
        let config = HnswConfig::new(4);
        let graph = HnswGraph::new(config);

        let mut layer_counts = [0u32; 10];
        for _ in 0..10000 {
            let layer = graph.random_layer();
            if layer < 10 {
                layer_counts[layer] += 1;
            }
        }

        // Layer 0 should have ~93.8% of nodes (for M=16)
        let layer0_pct = layer_counts[0] as f64 / 10000.0;
        assert!(
            layer0_pct > 0.90 && layer0_pct < 0.97,
            "layer 0 should have ~93.8%, got {:.1}%",
            layer0_pct * 100.0
        );

        // Layer 1 should have ~5-7%
        let layer1_pct = layer_counts[1] as f64 / 10000.0;
        assert!(
            layer1_pct > 0.03 && layer1_pct < 0.10,
            "layer 1 should have ~6%, got {:.1}%",
            layer1_pct * 100.0
        );
    }

    #[test]
    fn diversity_heuristic_produces_spread_neighbors() {
        // Insert vectors in a cluster + one outlier.
        // The diversity heuristic should select the outlier even though
        // it's farther, because it provides directional diversity.
        let config = HnswConfig::new(2).with_m(3).with_ef_construction(50);
        let mut graph = HnswGraph::new(config);

        // Cluster of similar vectors
        graph.insert(0, vec![1.0, 0.0]);
        graph.insert(1, vec![0.99, 0.01]);
        graph.insert(2, vec![0.98, 0.02]);
        graph.insert(3, vec![0.97, 0.03]);
        // Outlier in a different direction
        graph.insert(4, vec![0.0, 1.0]);

        // The entry point's neighbors should include the outlier
        // for directional diversity, not just the 3 nearest cluster members
        assert_eq!(graph.len(), 5);
    }

    #[test]
    fn recall_at_10_on_10000_vectors() {
        // Test recall at 10K scale to catch scaling bugs
        let dim = 64; // smaller dim for faster test
        let n = 10_000;
        let k = 10;
        let ef_search = 100;
        let num_queries = 20;

        let config = HnswConfig::new(dim).with_m(16).with_ef_construction(200);
        let mut graph = HnswGraph::new(config);
        let mut rng = SmallRng::seed_from_u64(123);

        let mut vectors: Vec<Vec<f32>> = Vec::new();
        for i in 0..n {
            let v = random_vector(&mut rng, dim);
            vectors.push(v.clone());
            graph.insert(i as u64, v);
        }

        let mut total_recall = 0.0;
        for _ in 0..num_queries {
            let query = random_vector(&mut rng, dim);
            let mut distances: Vec<(u64, f32)> = vectors.iter().enumerate()
                .map(|(i, v)| (i as u64, cosine_distance(&query, v)))
                .collect();
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let ground_truth: HashSet<u64> = distances.iter().take(k).map(|d| d.0).collect();

            let results = graph.search(&query, k, ef_search);
            let found: HashSet<u64> = results.iter().map(|r| r.0).collect();
            total_recall += ground_truth.intersection(&found).count() as f64 / k as f64;
        }

        let avg_recall = total_recall / num_queries as f64;
        eprintln!("10K recall@{}: {:.4}", k, avg_recall);
        assert!(
            avg_recall >= 0.90,
            "recall@{} at 10K should be >= 0.90, got {:.4}",
            k, avg_recall
        );
    }
}
