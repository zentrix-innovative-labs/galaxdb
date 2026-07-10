//! DiskANN — disk-resident approximate nearest-neighbor index (v0.7, inventory 8.17).
//!
//! Implements the Vamana graph algorithm (Jayaram Subramanya et al., NeurIPS 2019)
//! with a FreshDiskANN-style mutable delta (arXiv 2105.09613): the base graph plus
//! full-precision vectors live on disk as fixed-size node records so beam search
//! reads only the nodes it visits through a bounded in-memory cache, while recent
//! inserts and deletes are held in an in-memory delta and folded into every search
//! until a `consolidate()` rewrites the base.
//!
//! ## Why Vamana on disk
//!
//! HNSW (the default index) keeps its whole graph in RAM. DiskANN targets datasets
//! larger than memory: a single flat graph (degree `R`) with a long search list
//! (`L`) and the `alpha`-scaled robust-prune gives HNSW-class recall while keeping
//! the graph on SSD. Fixed-size node records (`[dim×f32 vector][u32 degree][R×u32
//! neighbors]`) make any node reachable with one `seek + read`.
//!
//! ## On-disk layout (magic `GDAN`, versioned via `galaxdb-common::format`)
//!
//! ```text
//! [FormatHeader 16B] [Meta] [external_ids: n×u64] [node_records: n×record_size]
//! Meta   = dim u32, num_points u32, r u32, entry_point u32, metric u8, pad 3B
//! record = [dim×f32 vector][degree u32][R×u32 neighbor ids]  (fixed size)
//! ```
//!
//! HNSW remains the default; DiskANN is selected per column/table.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use parking_lot::Mutex;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use galaxdb_common::error::{GalaxError, GalaxResult};
use galaxdb_common::format::{self, FormatHeader, FORMAT_HEADER_SIZE};

use crate::distance::{cosine_distance_normalized, l2_distance_squared, normalize};

/// Distance metric for the index. Cosine (default, for normalized embeddings) or
/// squared-L2 (for datasets like SIFT whose ground truth is L2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// Cosine distance on unit-normalized vectors (`1 - dot`).
    Cosine,
    /// Squared Euclidean distance.
    L2,
}

impl Metric {
    fn as_u8(self) -> u8 {
        match self {
            Metric::Cosine => 0,
            Metric::L2 => 1,
        }
    }
    fn from_u8(b: u8) -> GalaxResult<Self> {
        match b {
            0 => Ok(Metric::Cosine),
            1 => Ok(Metric::L2),
            other => Err(GalaxError::Internal(format!("unknown DiskANN metric {other}"))),
        }
    }
    #[inline]
    fn distance(self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metric::Cosine => cosine_distance_normalized(a, b),
            Metric::L2 => l2_distance_squared(a, b),
        }
    }
}

/// Build/search parameters for a Vamana graph.
#[derive(Debug, Clone, Copy)]
pub struct DiskAnnConfig {
    /// Vector dimensionality.
    pub dim: usize,
    /// Maximum out-degree `R` of the graph.
    pub r: usize,
    /// Search-list size `L` used during construction (larger = better graph).
    pub l_build: usize,
    /// Robust-prune slack `alpha` (> 1.0 keeps longer-range edges for recall).
    pub alpha: f32,
    /// Default search-list size `L` for queries (>= k).
    pub l_search: usize,
    /// Distance metric.
    pub metric: Metric,
}

impl DiskAnnConfig {
    /// Sensible defaults for embedding-scale data (cosine).
    pub fn new(dim: usize) -> Self {
        DiskAnnConfig {
            dim,
            r: 64,
            l_build: 100,
            alpha: 1.2,
            l_search: 100,
            metric: Metric::Cosine,
        }
    }
    pub fn with_metric(mut self, m: Metric) -> Self {
        self.metric = m;
        self
    }
    pub fn with_r(mut self, r: usize) -> Self {
        self.r = r;
        self
    }
    pub fn with_l_build(mut self, l: usize) -> Self {
        self.l_build = l;
        self
    }
    pub fn with_alpha(mut self, a: f32) -> Self {
        self.alpha = a;
        self
    }
}

/// A decoded on-disk node: its vector and its neighbor ids.
#[derive(Clone)]
struct Node {
    vector: Vec<f32>,
    neighbors: Vec<u32>,
}

/// A pending insert held in memory until consolidation.
#[derive(Clone)]
struct DeltaEntry {
    external_id: u64,
    vector: Vec<f32>,
}

/// Disk-resident Vamana index with a FreshDiskANN mutable delta.
pub struct DiskAnnIndex {
    path: PathBuf,
    config: DiskAnnConfig,
    /// Number of points in the base (on-disk) graph.
    num_points: u32,
    /// Entry point (medoid) node id in the base graph.
    entry_point: u32,
    /// external_id for each base node id.
    external_ids: Vec<u64>,
    /// Byte offset where node records begin.
    records_offset: u64,
    /// Fixed size of one node record.
    record_size: usize,
    /// Bounded in-memory node cache (base graph).
    cache: Mutex<NodeCache>,
    /// In-memory inserts not yet consolidated (FreshDiskANN delta).
    delta: Vec<DeltaEntry>,
    /// Tombstoned external ids (deleted; excluded from results).
    tombstones: HashSet<u64>,
}

/// A tiny bounded LRU-ish node cache (insertion-order eviction; good enough for
/// beam search locality, keeps memory bounded regardless of graph size).
struct NodeCache {
    map: HashMap<u32, Node>,
    order: std::collections::VecDeque<u32>,
    capacity: usize,
}

impl NodeCache {
    fn new(capacity: usize) -> Self {
        NodeCache {
            map: HashMap::new(),
            order: std::collections::VecDeque::new(),
            capacity: capacity.max(1),
        }
    }
    fn get(&mut self, id: u32) -> Option<Node> {
        self.map.get(&id).cloned()
    }
    fn put(&mut self, id: u32, node: Node) {
        // Present already → update in place, insertion order unchanged.
        if self.map.insert(id, node).is_some() {
            return;
        }
        // New entry: track its order and evict the oldest if over capacity.
        self.order.push_back(id);
        if self.map.len() > self.capacity {
            if let Some(evict) = self.order.pop_front() {
                self.map.remove(&evict);
            }
        }
    }
}

/// A candidate (node id, distance) used in search/prune queues.
#[derive(Clone, Copy)]
struct Cand {
    id: u32,
    dist: f32,
}

impl DiskAnnIndex {
    /// Build a Vamana graph from `entries` and write it to `path`.
    ///
    /// Two-pass construction: pass 1 with `alpha = 1.0` (tight), pass 2 with the
    /// configured `alpha` (adds long-range edges for recall). Vectors are
    /// normalized in place when the metric is cosine.
    pub fn build(
        path: impl AsRef<Path>,
        entries: &[(u64, Vec<f32>)],
        config: DiskAnnConfig,
    ) -> GalaxResult<Self> {
        let path = path.as_ref().to_path_buf();
        let n = entries.len();
        if n == 0 {
            return Err(GalaxError::Internal("DiskANN build needs >= 1 point".into()));
        }
        let dim = config.dim;
        for (_, v) in entries {
            if v.len() != dim {
                return Err(GalaxError::Internal(format!(
                    "DiskANN vector dim {} != config dim {dim}",
                    v.len()
                )));
            }
        }

        // Materialize vectors (normalized for cosine) and external ids.
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut external_ids: Vec<u64> = Vec::with_capacity(n);
        for (ext, v) in entries {
            let mut vv = v.clone();
            if config.metric == Metric::Cosine {
                normalize(&mut vv);
            }
            vectors.push(vv);
            external_ids.push(*ext);
        }

        let mut builder = VamanaBuilder::new(&vectors, config);
        builder.build();
        let entry_point = builder.medoid;
        let graph = builder.graph;

        let record_size = Self::record_size(dim, config.r);
        let records_offset = (FORMAT_HEADER_SIZE + META_SIZE + n * 8) as u64;

        // Serialize.
        let mut buf: Vec<u8> = Vec::with_capacity(records_offset as usize + n * record_size);
        buf.extend_from_slice(&format::DISKANN.header().to_bytes());
        // Meta.
        buf.extend_from_slice(&(dim as u32).to_le_bytes());
        buf.extend_from_slice(&(n as u32).to_le_bytes());
        buf.extend_from_slice(&(config.r as u32).to_le_bytes());
        buf.extend_from_slice(&entry_point.to_le_bytes());
        buf.push(config.metric.as_u8());
        buf.extend_from_slice(&[0u8; 3]); // pad
        // external ids.
        for e in &external_ids {
            buf.extend_from_slice(&e.to_le_bytes());
        }
        // node records.
        for id in 0..n {
            let v = &vectors[id];
            for f in v {
                buf.extend_from_slice(&f.to_le_bytes());
            }
            let nbrs = &graph[id];
            buf.extend_from_slice(&(nbrs.len() as u32).to_le_bytes());
            for j in 0..config.r {
                let nb = nbrs.get(j).copied().unwrap_or(0);
                buf.extend_from_slice(&nb.to_le_bytes());
            }
        }

        format::atomic_replace(&path, &buf).map_err(GalaxError::Io)?;

        Ok(DiskAnnIndex {
            path,
            config,
            num_points: n as u32,
            entry_point,
            external_ids,
            records_offset,
            record_size,
            cache: Mutex::new(NodeCache::new(DEFAULT_CACHE_NODES)),
            delta: Vec::new(),
            tombstones: HashSet::new(),
        })
    }

    /// Open an existing DiskANN index, validating the `GDAN` format version
    /// (too-old / too-new refused via the shared format gate).
    pub fn open(path: impl AsRef<Path>) -> GalaxResult<Self> {
        let path = path.as_ref().to_path_buf();
        let mut f = File::open(&path).map_err(GalaxError::Io)?;

        let mut hdr = [0u8; FORMAT_HEADER_SIZE];
        f.read_exact(&mut hdr).map_err(GalaxError::Io)?;
        let header = FormatHeader::from_bytes(&hdr, format::DISKANN.magic)?;
        format::DISKANN.check(header.format_version)?; // typed too-old/too-new

        let mut meta = [0u8; META_SIZE];
        f.read_exact(&mut meta).map_err(GalaxError::Io)?;
        let dim = u32::from_le_bytes([meta[0], meta[1], meta[2], meta[3]]) as usize;
        let num_points = u32::from_le_bytes([meta[4], meta[5], meta[6], meta[7]]);
        let r = u32::from_le_bytes([meta[8], meta[9], meta[10], meta[11]]) as usize;
        let entry_point = u32::from_le_bytes([meta[12], meta[13], meta[14], meta[15]]);
        let metric = Metric::from_u8(meta[16])?;

        let n = num_points as usize;
        let mut id_bytes = vec![0u8; n * 8];
        f.read_exact(&mut id_bytes).map_err(GalaxError::Io)?;
        let mut external_ids = Vec::with_capacity(n);
        for i in 0..n {
            let mut b = [0u8; 8];
            b.copy_from_slice(&id_bytes[i * 8..i * 8 + 8]);
            external_ids.push(u64::from_le_bytes(b));
        }

        let mut config = DiskAnnConfig::new(dim);
        config.r = r;
        config.metric = metric;

        Ok(DiskAnnIndex {
            path,
            config,
            num_points,
            entry_point,
            external_ids,
            records_offset: (FORMAT_HEADER_SIZE + META_SIZE + n * 8) as u64,
            record_size: Self::record_size(dim, r),
            cache: Mutex::new(NodeCache::new(DEFAULT_CACHE_NODES)),
            delta: Vec::new(),
            tombstones: HashSet::new(),
        })
    }

    /// Total live points (base minus tombstones plus delta).
    pub fn len(&self) -> usize {
        (self.num_points as usize) + self.delta.len()
            - self
                .tombstones
                .iter()
                .filter(|t| self.external_ids.contains(t))
                .count()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn config(&self) -> &DiskAnnConfig {
        &self.config
    }

    fn record_size(dim: usize, r: usize) -> usize {
        dim * 4 + 4 + r * 4
    }

    /// Read node `id`'s record from disk (through the cache).
    fn read_node(&self, id: u32) -> GalaxResult<Node> {
        if let Some(n) = self.cache.lock().get(id) {
            return Ok(n);
        }
        let mut f = File::open(&self.path).map_err(GalaxError::Io)?;
        let offset = self.records_offset + id as u64 * self.record_size as u64;
        f.seek(SeekFrom::Start(offset)).map_err(GalaxError::Io)?;
        let mut rec = vec![0u8; self.record_size];
        f.read_exact(&mut rec).map_err(GalaxError::Io)?;

        let dim = self.config.dim;
        let mut vector = Vec::with_capacity(dim);
        for i in 0..dim {
            let mut b = [0u8; 4];
            b.copy_from_slice(&rec[i * 4..i * 4 + 4]);
            vector.push(f32::from_le_bytes(b));
        }
        let mut p = dim * 4;
        let degree = u32::from_le_bytes([rec[p], rec[p + 1], rec[p + 2], rec[p + 3]]) as usize;
        p += 4;
        let mut neighbors = Vec::with_capacity(degree);
        for _ in 0..degree {
            let nb = u32::from_le_bytes([rec[p], rec[p + 1], rec[p + 2], rec[p + 3]]);
            neighbors.push(nb);
            p += 4;
        }
        let node = Node { vector, neighbors };
        self.cache.lock().put(id, node.clone());
        Ok(node)
    }

    /// Greedy beam search over the base graph. Returns up to `l` visited
    /// candidates sorted by distance (nearest first).
    fn greedy_search_base(&self, query: &[f32], l: usize) -> GalaxResult<Vec<Cand>> {
        let metric = self.config.metric;
        if self.num_points == 0 {
            return Ok(Vec::new());
        }
        let ep = self.entry_point;
        let ep_node = self.read_node(ep)?;
        let ep_dist = metric.distance(query, &ep_node.vector);

        // Working list of best-L candidates + a visited set.
        let mut list: Vec<Cand> = vec![Cand { id: ep, dist: ep_dist }];
        let mut visited: HashSet<u32> = HashSet::new();
        let mut expanded: HashSet<u32> = HashSet::new();
        visited.insert(ep);

        loop {
            // Pick the closest not-yet-expanded candidate.
            let next = list
                .iter()
                .filter(|c| !expanded.contains(&c.id))
                .min_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal))
                .copied();
            let Some(cur) = next else { break };
            expanded.insert(cur.id);

            let node = self.read_node(cur.id)?;
            for &nb in &node.neighbors {
                if visited.insert(nb) {
                    let nb_node = self.read_node(nb)?;
                    let d = metric.distance(query, &nb_node.vector);
                    list.push(Cand { id: nb, dist: d });
                }
            }
            // Truncate to best L.
            list.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
            list.truncate(l.max(1));
        }
        Ok(list)
    }

    /// Search for the `k` nearest neighbors. Merges the disk base-graph result
    /// with the in-memory delta (brute-forced — the delta is small until
    /// consolidation) and excludes tombstoned ids. `l_search` overrides the
    /// configured search-list size when `Some`.
    pub fn search(&self, query: &[f32], k: usize, l_search: Option<usize>) -> GalaxResult<Vec<(u64, f32)>> {
        let metric = self.config.metric;
        let mut q = query.to_vec();
        if metric == Metric::Cosine {
            normalize(&mut q);
        }
        let l = l_search.unwrap_or(self.config.l_search).max(k);

        let mut out: Vec<(u64, f32)> = Vec::new();

        // Base graph.
        for c in self.greedy_search_base(&q, l)? {
            let ext = self.external_ids[c.id as usize];
            if !self.tombstones.contains(&ext) {
                out.push((ext, c.dist));
            }
        }
        // Delta (recent inserts).
        for d in &self.delta {
            if self.tombstones.contains(&d.external_id) {
                continue;
            }
            let dist = metric.distance(&q, &d.vector);
            out.push((d.external_id, dist));
        }

        // Dedup by external id keeping the smallest distance, then top-k.
        out.sort_by(|a, b| {
            a.0.cmp(&b.0).then(
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal),
            )
        });
        out.dedup_by_key(|(id, _)| *id);
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        out.truncate(k);
        Ok(out)
    }

    /// FreshDiskANN incremental insert: append to the in-memory delta. The point
    /// is immediately findable via `search`; `consolidate()` later folds it into
    /// the disk graph. Re-inserting a tombstoned id revives it.
    pub fn insert(&mut self, external_id: u64, vector: Vec<f32>) -> GalaxResult<()> {
        if vector.len() != self.config.dim {
            return Err(GalaxError::Internal(format!(
                "DiskANN insert dim {} != index dim {}",
                vector.len(),
                self.config.dim
            )));
        }
        let mut v = vector;
        if self.config.metric == Metric::Cosine {
            normalize(&mut v);
        }
        self.tombstones.remove(&external_id);
        // Replace any existing delta entry for the same id.
        self.delta.retain(|d| d.external_id != external_id);
        self.delta.push(DeltaEntry { external_id, vector: v });
        Ok(())
    }

    /// Mark an external id deleted. It is excluded from results immediately and
    /// dropped from the base at the next `consolidate()`.
    pub fn delete(&mut self, external_id: u64) {
        self.delta.retain(|d| d.external_id != external_id);
        self.tombstones.insert(external_id);
    }

    /// Number of un-consolidated delta inserts.
    pub fn delta_len(&self) -> usize {
        self.delta.len()
    }

    /// Rebuild the on-disk base graph from all live points (base ∪ delta minus
    /// tombstones), then clear the delta and tombstones. Mirrors the HNSW
    /// delta-buffer merge contract. After this, search is pure disk again.
    pub fn consolidate(&mut self) -> GalaxResult<()> {
        // Gather live (external_id, vector) pairs.
        let mut entries: Vec<(u64, Vec<f32>)> = Vec::new();
        for id in 0..self.num_points as usize {
            let ext = self.external_ids[id];
            if self.tombstones.contains(&ext) {
                continue;
            }
            // Skip if a delta entry supersedes it (re-insert).
            if self.delta.iter().any(|d| d.external_id == ext) {
                continue;
            }
            let node = self.read_node(id as u32)?;
            entries.push((ext, node.vector));
        }
        for d in &self.delta {
            if !self.tombstones.contains(&d.external_id) {
                entries.push((d.external_id, d.vector.clone()));
            }
        }

        if entries.is_empty() {
            return Err(GalaxError::Internal(
                "DiskANN consolidate would empty the index".into(),
            ));
        }

        // Rebuild. Vectors are already normalized (cosine) from insert/build, so
        // pass them through a config that does not re-normalize twice — build
        // normalizes again which is idempotent for unit vectors.
        let rebuilt = DiskAnnIndex::build(&self.path, &entries, self.config)?;
        self.num_points = rebuilt.num_points;
        self.entry_point = rebuilt.entry_point;
        self.external_ids = rebuilt.external_ids;
        self.records_offset = rebuilt.records_offset;
        self.record_size = rebuilt.record_size;
        self.cache = Mutex::new(NodeCache::new(DEFAULT_CACHE_NODES));
        self.delta.clear();
        self.tombstones.clear();
        Ok(())
    }
}

/// Meta block size after the format header (bytes).
const META_SIZE: usize = 20; // dim(4)+n(4)+r(4)+ep(4)+metric(1)+pad(3)
/// Bounded node-cache size (records). Keeps memory ~ cache_nodes × record_size
/// regardless of graph size — the "bounded in-memory node cache" of Req 6.1.
const DEFAULT_CACHE_NODES: usize = 4096;

/// In-memory Vamana graph builder.
struct VamanaBuilder<'a> {
    vectors: &'a [Vec<f32>],
    config: DiskAnnConfig,
    graph: Vec<Vec<u32>>,
    medoid: u32,
}

impl<'a> VamanaBuilder<'a> {
    fn new(vectors: &'a [Vec<f32>], config: DiskAnnConfig) -> Self {
        let n = vectors.len();
        VamanaBuilder {
            vectors,
            config,
            graph: vec![Vec::new(); n],
            medoid: 0,
        }
    }

    #[inline]
    fn dist(&self, a: u32, b: u32) -> f32 {
        self.config
            .metric
            .distance(&self.vectors[a as usize], &self.vectors[b as usize])
    }
    #[inline]
    fn dist_v(&self, q: &[f32], b: u32) -> f32 {
        self.config.metric.distance(q, &self.vectors[b as usize])
    }

    fn compute_medoid(&self) -> u32 {
        let n = self.vectors.len();
        let dim = self.config.dim;
        let mut centroid = vec![0.0f32; dim];
        for v in self.vectors {
            for (c, x) in centroid.iter_mut().zip(v) {
                *c += *x;
            }
        }
        for c in &mut centroid {
            *c /= n as f32;
        }
        // Node closest to the centroid.
        let mut best = 0u32;
        let mut best_d = f32::MAX;
        for (i, _) in self.vectors.iter().enumerate() {
            let d = self.dist_v(&centroid, i as u32);
            if d < best_d {
                best_d = d;
                best = i as u32;
            }
        }
        best
    }

    fn build(&mut self) {
        let n = self.vectors.len();
        if n == 1 {
            self.medoid = 0;
            return;
        }
        self.medoid = self.compute_medoid();

        // Initialize a random R-regular-ish graph so search has somewhere to go.
        let mut rng = SmallRng::seed_from_u64(0xDA_11_A5_5E);
        let r = self.config.r;
        for i in 0..n {
            let mut nbrs = Vec::with_capacity(r);
            let mut seen = HashSet::new();
            seen.insert(i as u32);
            let target = r.min(n - 1);
            while nbrs.len() < target {
                let j = rng.gen_range(0..n) as u32;
                if seen.insert(j) {
                    nbrs.push(j);
                }
            }
            self.graph[i] = nbrs;
        }

        // Two passes: alpha = 1.0 then the configured alpha.
        let order: Vec<u32> = {
            let mut o: Vec<u32> = (0..n as u32).collect();
            // deterministic shuffle
            for i in (1..o.len()).rev() {
                let j = rng.gen_range(0..=i);
                o.swap(i, j);
            }
            o
        };
        for &alpha in &[1.0f32, self.config.alpha] {
            for &p in &order {
                self.insert_point(p, alpha);
            }
        }
    }

    /// Vamana per-point step: greedy-search from the medoid to `p`, robust-prune
    /// the visited set to `p`'s new neighbors, then add back-edges with prune.
    fn insert_point(&mut self, p: u32, alpha: f32) {
        let l = self.config.l_build;
        let visited = self.greedy_search_visited(&self.vectors[p as usize], l);
        let candidates: Vec<u32> = visited.into_iter().filter(|&v| v != p).collect();
        let new_nbrs = self.robust_prune(p, candidates, alpha);
        self.graph[p as usize] = new_nbrs.clone();

        for j in new_nbrs {
            if j == p {
                continue;
            }
            if !self.graph[j as usize].contains(&p) {
                self.graph[j as usize].push(p);
            }
            if self.graph[j as usize].len() > self.config.r {
                let cand = self.graph[j as usize].clone();
                self.graph[j as usize] = self.robust_prune(j, cand, alpha);
            }
        }
    }

    /// Greedy search returning the set of visited node ids (for construction).
    fn greedy_search_visited(&self, query: &[f32], l: usize) -> Vec<u32> {
        let mut list: Vec<Cand> = vec![Cand {
            id: self.medoid,
            dist: self.dist_v(query, self.medoid),
        }];
        let mut visited: HashSet<u32> = HashSet::new();
        let mut expanded: HashSet<u32> = HashSet::new();
        let mut all_visited: Vec<u32> = vec![self.medoid];
        visited.insert(self.medoid);

        loop {
            let next = list
                .iter()
                .filter(|c| !expanded.contains(&c.id))
                .min_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal))
                .copied();
            let Some(cur) = next else { break };
            expanded.insert(cur.id);

            for &nb in &self.graph[cur.id as usize] {
                if visited.insert(nb) {
                    all_visited.push(nb);
                    let d = self.dist_v(query, nb);
                    list.push(Cand { id: nb, dist: d });
                }
            }
            list.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
            list.truncate(l.max(1));
        }
        all_visited
    }

    /// Robust prune (DiskANN): from `candidates`, greedily keep the closest to
    /// `p`, discarding any candidate that is `alpha`× closer to an already-kept
    /// neighbor than to `p` — yields a diverse, navigable neighborhood of <= R.
    fn robust_prune(&self, p: u32, candidates: Vec<u32>, alpha: f32) -> Vec<u32> {
        let mut pool: Vec<Cand> = candidates
            .into_iter()
            .filter(|&c| c != p)
            .map(|c| Cand { id: c, dist: self.dist(p, c) })
            .collect();
        // include current neighbors
        for &c in &self.graph[p as usize] {
            if c != p && !pool.iter().any(|x| x.id == c) {
                pool.push(Cand { id: c, dist: self.dist(p, c) });
            }
        }
        pool.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));

        let mut result: Vec<u32> = Vec::with_capacity(self.config.r);
        let mut removed = vec![false; pool.len()];
        for i in 0..pool.len() {
            if removed[i] {
                continue;
            }
            let pstar = pool[i];
            result.push(pstar.id);
            if result.len() >= self.config.r {
                break;
            }
            for j in (i + 1)..pool.len() {
                if removed[j] {
                    continue;
                }
                let v = pool[j];
                // dist(p*, v) vs dist(p, v)
                let d_star_v = self.dist(pstar.id, v.id);
                if alpha * d_star_v <= v.dist {
                    removed[j] = true;
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic clustered vectors: `clusters` groups, each around a random
    /// center, so nearest-neighbor structure is real (not uniform noise).
    fn clustered(n: usize, dim: usize, clusters: usize, seed: u64) -> Vec<(u64, Vec<f32>)> {
        let mut rng = SmallRng::seed_from_u64(seed);
        let centers: Vec<Vec<f32>> = (0..clusters)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        (0..n)
            .map(|i| {
                let c = &centers[i % clusters];
                let v: Vec<f32> = c
                    .iter()
                    .map(|x| x + rng.gen_range(-0.1..0.1))
                    .collect();
                (i as u64, v)
            })
            .collect()
    }

    fn brute_force(entries: &[(u64, Vec<f32>)], query: &[f32], k: usize, metric: Metric) -> Vec<u64> {
        let mut q = query.to_vec();
        if metric == Metric::Cosine {
            normalize(&mut q);
        }
        let mut scored: Vec<(u64, f32)> = entries
            .iter()
            .map(|(id, v)| {
                let mut vv = v.clone();
                if metric == Metric::Cosine {
                    normalize(&mut vv);
                }
                (*id, metric.distance(&q, &vv))
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.into_iter().take(k).map(|(id, _)| id).collect()
    }

    #[test]
    fn build_search_high_recall_vs_brute_force() {
        let dim = 32;
        let n = 2000;
        let entries = clustered(n, dim, 40, 0xBEEF);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("idx.gdan");
        let config = DiskAnnConfig::new(dim).with_r(48).with_l_build(128);
        let index = DiskAnnIndex::build(&path, &entries, config).unwrap();
        assert_eq!(index.len(), n);

        // Measure recall@10 over 50 queries against brute-force ground truth.
        let k = 10;
        let mut total = 0usize;
        let mut hit = 0usize;
        for probe in (0..n).step_by(n / 50) {
            let q = &entries[probe].1;
            let truth: HashSet<u64> = brute_force(&entries, q, k, Metric::Cosine).into_iter().collect();
            let got = index.search(q, k, Some(128)).unwrap();
            for (id, _) in &got {
                if truth.contains(id) {
                    hit += 1;
                }
            }
            total += k;
        }
        let recall = hit as f64 / total as f64;
        assert!(
            recall >= 0.90,
            "DiskANN recall@{k} = {recall:.3} should be >= 0.90 vs brute force"
        );
    }

    #[test]
    fn reopen_from_disk_searches_without_rebuild() {
        let dim = 16;
        let entries = clustered(500, dim, 10, 0x1234);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("idx.gdan");
        let config = DiskAnnConfig::new(dim).with_r(32);
        {
            let _ = DiskAnnIndex::build(&path, &entries, config).unwrap();
        }
        // Reopen purely from disk (no entries passed) and search.
        let reopened = DiskAnnIndex::open(&path).unwrap();
        assert_eq!(reopened.len(), 500);
        let q = &entries[42].1;
        let got = reopened.search(q, 5, None).unwrap();
        assert!(!got.is_empty());
        assert!(
            got.iter().any(|(id, _)| *id == 42),
            "exact-match probe should recover itself from the disk index"
        );
    }

    #[test]
    fn incremental_insert_is_findable_before_consolidate() {
        let dim = 16;
        let entries = clustered(300, dim, 8, 0x77);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("idx.gdan");
        let mut index = DiskAnnIndex::build(&path, &entries, DiskAnnConfig::new(dim)).unwrap();

        // Insert a brand new vector; it must be findable immediately (delta).
        let new_id = 99999u64;
        let new_vec: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
        index.insert(new_id, new_vec.clone()).unwrap();
        assert_eq!(index.delta_len(), 1);

        let got = index.search(&new_vec, 3, None).unwrap();
        assert!(
            got.iter().any(|(id, _)| *id == new_id),
            "freshly inserted vector must be findable before consolidation"
        );

        // Consolidate folds it into the disk graph; still findable, delta empty.
        index.consolidate().unwrap();
        assert_eq!(index.delta_len(), 0);
        let got2 = index.search(&new_vec, 3, None).unwrap();
        assert!(got2.iter().any(|(id, _)| *id == new_id));
    }

    #[test]
    fn delete_excludes_from_results() {
        let dim = 16;
        let entries = clustered(300, dim, 8, 0x88);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("idx.gdan");
        let mut index = DiskAnnIndex::build(&path, &entries, DiskAnnConfig::new(dim)).unwrap();

        let victim = entries[10].0;
        let q = &entries[10].1;
        // Present before delete.
        assert!(index.search(q, 5, None).unwrap().iter().any(|(id, _)| *id == victim));
        index.delete(victim);
        // Absent after delete.
        assert!(!index.search(q, 5, None).unwrap().iter().any(|(id, _)| *id == victim));
    }

    #[test]
    fn open_refuses_too_new_format() {
        let dim = 8;
        let entries = clustered(50, dim, 4, 0x99);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("idx.gdan");
        DiskAnnIndex::build(&path, &entries, DiskAnnConfig::new(dim)).unwrap();

        // Corrupt the on-disk format version to current+1 (a "from the future"
        // file) and confirm open refuses with FormatTooNew.
        let mut bytes = std::fs::read(&path).unwrap();
        let bumped = format::DISKANN.current_write + 1;
        bytes[4..6].copy_from_slice(&bumped.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();

        match DiskAnnIndex::open(&path) {
            Err(GalaxError::FormatTooNew { found, .. }) => assert_eq!(found, bumped),
            Err(other) => panic!("expected FormatTooNew, got {other:?}"),
            Ok(_) => panic!("expected FormatTooNew, got Ok"),
        }
    }

    #[test]
    fn l2_metric_build_and_search() {
        let dim = 12;
        let entries = clustered(400, dim, 10, 0xABCD);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("idx_l2.gdan");
        let config = DiskAnnConfig::new(dim).with_metric(Metric::L2).with_r(32);
        let index = DiskAnnIndex::build(&path, &entries, config).unwrap();

        let k = 10;
        let mut hit = 0usize;
        let mut total = 0usize;
        for probe in (0..400).step_by(20) {
            let q = &entries[probe].1;
            let truth: HashSet<u64> = brute_force(&entries, q, k, Metric::L2).into_iter().collect();
            for (id, _) in index.search(q, k, Some(128)).unwrap() {
                if truth.contains(&id) {
                    hit += 1;
                }
            }
            total += k;
        }
        let recall = hit as f64 / total as f64;
        assert!(recall >= 0.85, "L2 recall@{k} = {recall:.3} should be >= 0.85");
    }
}
