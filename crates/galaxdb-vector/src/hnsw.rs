use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicUsize, AtomicU32, Ordering as AtomicOrdering};
use std::cell::UnsafeCell;

use parking_lot::{RwLock, Mutex};
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use rayon::prelude::*;

use crate::distance::{cosine_distance_normalized, normalize};

#[derive(Debug, Clone, Copy)]
pub struct HnswConfig {
    pub m: usize,
    pub m0: usize,
    pub ef_construction: usize,
    pub dim: usize,
    pub max_elements: usize,
}

impl HnswConfig {
    pub fn new(dim: usize) -> Self {
        Self {
            m: 16,
            m0: 32,
            ef_construction: 200,
            dim,
            max_elements: 1_000_000,
        }
    }
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self.m0 = m * 2;
        self
    }
    pub fn with_ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }
    pub fn with_max_elements(mut self, max: usize) -> Self {
        self.max_elements = max;
        self
    }
    pub fn max_edges(&self, layer: usize) -> usize {
        if layer == 0 { self.m0 } else { self.m }
    }
}

struct VisitedList {
    visited: Vec<u32>,
    visited_gen: u32,
}

impl VisitedList {
    fn new(size: usize) -> Self {
        Self {
            visited: vec![0; size],
            visited_gen: 1,
        }
    }

    fn mark_visited(&mut self, node: u32) -> bool {
        if self.visited[node as usize] == self.visited_gen {
            true
        } else {
            self.visited[node as usize] = self.visited_gen;
            false
        }
    }

    fn next_gen(&mut self) {
        if self.visited_gen == u32::MAX {
            self.visited.fill(0);
            self.visited_gen = 1;
        } else {
            self.visited_gen += 1;
        }
    }
}

#[derive(Clone, Copy)]
struct Candidate {
    id: u32,
    dist: f32,
}
impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool { self.dist == other.dist && self.id == other.id }
}
impl Eq for Candidate {}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering on distance so a max-`BinaryHeap` of
        // `Candidate` behaves as a min-heap by distance (nearest first).
        // `dist` is f32 (only PartialOrd), so NaN collapses to Equal.
        other.dist.partial_cmp(&self.dist).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy)]
struct MaxCandidate {
    id: u32,
    dist: f32,
}
impl PartialEq for MaxCandidate {
    fn eq(&self, other: &Self) -> bool { self.dist == other.dist && self.id == other.id }
}
impl Eq for MaxCandidate {}
impl Ord for MaxCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Normal ordering on distance: a max-`BinaryHeap` of
        // `MaxCandidate` keeps the farthest result on top so it can be
        // evicted when the result set exceeds `ef`.
        self.dist.partial_cmp(&other.dist).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for MaxCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct ConcurrentVec<T> {
    inner: UnsafeCell<Vec<T>>,
}
unsafe impl<T: Send> Send for ConcurrentVec<T> {}
unsafe impl<T: Sync> Sync for ConcurrentVec<T> {}

impl<T> ConcurrentVec<T> {
    fn new() -> Self {
        Self { inner: UnsafeCell::new(Vec::new()) }
    }
    fn resize(&mut self, new_len: usize, value: T) where T: Clone {
        self.inner.get_mut().resize(new_len, value);
    }
    fn as_mut_slice(&mut self) -> &mut [T] {
        self.inner.get_mut().as_mut_slice()
    }
    /// Returns a mutable slice into the interior buffer from a shared
    /// reference. This is deliberate interior mutability: `ConcurrentVec`
    /// hands out disjoint mutable slices to different node ranges from
    /// parallel build threads, with per-node `RwLock`s (in `HnswGraph`)
    /// serialising writes to any given node. The clippy lint flags the
    /// `&self -> &mut` shape, which is exactly the intended (unsafe)
    /// contract here.
    #[allow(clippy::mut_from_ref)]
    unsafe fn get_slice_mut(&self, start: usize, len: usize) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut((*self.inner.get()).as_mut_ptr().add(start), len) }
    }
    unsafe fn as_mut_ptr(&self) -> *mut T {
        unsafe { (*self.inner.get()).as_mut_ptr() }
    }
    fn get(&self) -> &[T] {
        unsafe { &*self.inner.get() }
    }
}

pub struct HnswGraph {
    config: HnswConfig,
    vectors: ConcurrentVec<f32>,
    external_ids: ConcurrentVec<u64>,
    node_max_layers: ConcurrentVec<usize>,

    neighbors0: ConcurrentVec<u32>,
    neighbors0_counts: ConcurrentVec<u16>,
    neighbors_upper: ConcurrentVec<Vec<Vec<u32>>>,

    node_locks: RwLock<Vec<RwLock<()>>>,

    entry_point: RwLock<Option<u32>>,
    max_layer: AtomicUsize,

    visited_pool: Mutex<Vec<VisitedList>>,
    total_gens: AtomicU32,
    
    len: AtomicUsize,
}

impl HnswGraph {
    pub fn new(config: HnswConfig) -> Self {
        Self {
            config,
            vectors: ConcurrentVec::new(),
            external_ids: ConcurrentVec::new(),
            node_max_layers: ConcurrentVec::new(),
            neighbors0: ConcurrentVec::new(),
            neighbors0_counts: ConcurrentVec::new(),
            neighbors_upper: ConcurrentVec::new(),
            node_locks: RwLock::new(Vec::new()),
            entry_point: RwLock::new(None),
            max_layer: AtomicUsize::new(0),
            visited_pool: Mutex::new(Vec::new()),
            total_gens: AtomicU32::new(0),
            len: AtomicUsize::new(0),
        }
    }

    pub fn len(&self) -> usize { self.len.load(AtomicOrdering::Acquire) }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
    pub fn config(&self) -> &HnswConfig { &self.config }
    pub fn entry_point(&self) -> Option<u32> { *self.entry_point.read() }
    pub fn max_layer(&self) -> usize { self.max_layer.load(AtomicOrdering::Acquire) }
    pub fn node_max_layer(&self, id: u32) -> usize { self.node_max_layers.get()[id as usize] }
    pub fn visited_gen(&self) -> u32 { self.total_gens.load(AtomicOrdering::Relaxed) }

    pub fn get_external_id(&self, id: u32) -> Option<u64> {
        if id as usize >= self.len() { None } else { Some(self.external_ids.get()[id as usize]) }
    }

    pub fn get_vector(&self, id: u32) -> Option<&[f32]> {
        let start = id as usize * self.config.dim;
        let end = start + self.config.dim;
        if end <= self.vectors.get().len() {
            Some(&self.vectors.get()[start..end])
        } else {
            None
        }
    }

    pub fn get_neighbors(&self, id: u32, layer: usize) -> Vec<u32> {
        if layer == 0 {
            // Lock-free read for layer 0: fixed-size pre-allocated slots.
            // Safe because writes only overwrite within the pre-allocated m0 slots
            // and atomically update the count.
            let m0 = self.config.m0;
            let count = unsafe { *self.neighbors0_counts.as_mut_ptr().add(id as usize) } as usize;
            if count == 0 { return Vec::new(); }
            let slice = unsafe { self.neighbors0.get_slice_mut(id as usize * m0, count) };
            slice.to_vec()
        } else {
            // Upper layers use Vec<u32> which requires locking for safe concurrent access
            let guards = self.node_locks.read();
            let _lock = guards[id as usize].read();
            let upper = unsafe { &*self.neighbors_upper.as_mut_ptr().add(id as usize) };
            if layer - 1 < upper.len() {
                upper[layer - 1].clone()
            } else {
                Vec::new()
            }
        }
    }

    fn get_visited_list(&self) -> VisitedList {
        self.total_gens.fetch_add(1, AtomicOrdering::Relaxed);
        let mut pool = self.visited_pool.lock();
        if let Some(mut list) = pool.pop() {
            list.next_gen();
            list
        } else {
            VisitedList::new(self.config.max_elements.max(1_000_000))
        }
    }
    
    fn put_visited_list(&self, list: VisitedList) {
        self.visited_pool.lock().push(list);
    }

    fn search_layer(
        &self,
        query: &[f32],
        entry_point: u32,
        ep_dist: f32,
        ef: usize,
        layer: usize,
        visited: &mut VisitedList,
    ) -> BinaryHeap<MaxCandidate> {
        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();

        candidates.push(Candidate { id: entry_point, dist: ep_dist });
        results.push(MaxCandidate { id: entry_point, dist: ep_dist });
        visited.mark_visited(entry_point);

        let m0 = self.config.m0;

        while let Some(c) = candidates.pop() {
            let farthest_dist = results.peek().unwrap().dist;
            if results.len() >= ef && c.dist > farthest_dist {
                break;
            }

            // Inline neighbor access to avoid Vec allocation in hot path
            if layer == 0 {
                let count = unsafe { *self.neighbors0_counts.as_mut_ptr().add(c.id as usize) } as usize;
                let nb_slice = unsafe { self.neighbors0.get_slice_mut(c.id as usize * m0, count) };
                for &nb in nb_slice.iter() {
                    if !visited.mark_visited(nb) {
                        let d = cosine_distance_normalized(query, self.get_vector(nb).unwrap());
                        let farthest_dist = results.peek().unwrap().dist;
                        if results.len() < ef || d < farthest_dist {
                            candidates.push(Candidate { id: nb, dist: d });
                            results.push(MaxCandidate { id: nb, dist: d });
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            } else {
                let neighbors = self.get_neighbors(c.id, layer);
                for &nb in &neighbors {
                    if !visited.mark_visited(nb) {
                        let d = cosine_distance_normalized(query, self.get_vector(nb).unwrap());
                        let farthest_dist = results.peek().unwrap().dist;
                        if results.len() < ef || d < farthest_dist {
                            candidates.push(Candidate { id: nb, dist: d });
                            results.push(MaxCandidate { id: nb, dist: d });
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }
        results
    }

    fn select_neighbors_heuristic(&self, candidates: &[Candidate], m_max: usize, _query: &[f32]) -> Vec<u32> {
        let mut selected = Vec::with_capacity(m_max);
        let mut pruned = Vec::new();
        
        for c in candidates {
            if selected.len() >= m_max {
                break;
            }
            
            let c_vec = self.get_vector(c.id).unwrap();
            let mut is_good = true;
            for &s_id in &selected {
                let s_vec = self.get_vector(s_id).unwrap();
                let dist_c_s = cosine_distance_normalized(c_vec, s_vec);
                if dist_c_s < c.dist {
                    is_good = false;
                    break;
                }
            }
            
            if is_good {
                selected.push(c.id);
            } else {
                pruned.push(c.id);
            }
        }
        
        let mut iter = pruned.into_iter();
        while selected.len() < m_max {
            if let Some(p) = iter.next() {
                selected.push(p);
            } else {
                break;
            }
        }
        
        selected
    }

    fn set_neighbors(&self, id: u32, layer: usize, neighbors: &[u32]) {
        let guards = self.node_locks.read();
        let _lock = guards[id as usize].write();
        
        if layer == 0 {
            let count_ptr = unsafe { self.neighbors0_counts.as_mut_ptr().add(id as usize) };
            let slice = unsafe { self.neighbors0.get_slice_mut(id as usize * self.config.m0, self.config.m0) };
            for (i, &n) in neighbors.iter().enumerate() {
                slice[i] = n;
            }
            unsafe { *count_ptr = neighbors.len() as u16; }
        } else {
            let upper = unsafe { &mut *self.neighbors_upper.as_mut_ptr().add(id as usize) };
            upper[layer - 1] = neighbors.to_vec();
        }
    }

    fn add_edge_bidirectional(&self, target_node: u32, new_nb: u32, layer: usize) {
        let m_max = self.config.max_edges(layer);
        
        let guards = self.node_locks.read();
        let _lock = guards[target_node as usize].write();
        
        let mut neighbors = if layer == 0 {
            let count = unsafe { *self.neighbors0_counts.as_mut_ptr().add(target_node as usize) };
            let slice = unsafe { self.neighbors0.get_slice_mut(target_node as usize * self.config.m0, count as usize) };
            slice.to_vec()
        } else {
            let upper = unsafe { &*self.neighbors_upper.as_mut_ptr().add(target_node as usize) };
            upper[layer - 1].clone()
        };
        
        if !neighbors.contains(&new_nb) {
            neighbors.push(new_nb);
            
            if neighbors.len() > m_max {
                let target_vec = self.get_vector(target_node).unwrap();
                let mut candidates: Vec<Candidate> = neighbors.into_iter().map(|n| {
                    let d = cosine_distance_normalized(target_vec, self.get_vector(n).unwrap());
                    Candidate { id: n, dist: d }
                }).collect();
                candidates.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
                
                neighbors = if layer == 0 {
                    self.select_neighbors_heuristic(&candidates, m_max, target_vec)
                } else {
                    candidates.into_iter().take(m_max).map(|c| c.id).collect()
                };
            }
            
            if layer == 0 {
                let count_ptr = unsafe { self.neighbors0_counts.as_mut_ptr().add(target_node as usize) };
                let slice = unsafe { self.neighbors0.get_slice_mut(target_node as usize * self.config.m0, self.config.m0) };
                for (i, &n) in neighbors.iter().enumerate() {
                    slice[i] = n;
                }
                unsafe { *count_ptr = neighbors.len() as u16; }
            } else {
                let upper = unsafe { &mut *self.neighbors_upper.as_mut_ptr().add(target_node as usize) };
                upper[layer - 1] = neighbors;
            }
        }
    }

    fn insert_internal(&self, id: u32) {
        let mut visited = self.get_visited_list();
        let query = self.get_vector(id).unwrap();
        let node_layer = self.node_max_layers.get()[id as usize];
        
        let mut curr_ep = *self.entry_point.read();
        let max_layer = self.max_layer.load(AtomicOrdering::Acquire);
        
        if curr_ep.is_none() {
            let mut write_ep = self.entry_point.write();
            if write_ep.is_none() {
                *write_ep = Some(id);
                self.max_layer.store(node_layer, AtomicOrdering::Release);
                self.put_visited_list(visited);
                return;
            }
            curr_ep = *write_ep;
        }
        
        let mut curr_node = curr_ep.unwrap();
        let mut curr_dist = cosine_distance_normalized(query, self.get_vector(curr_node).unwrap());
        
        for layer in (node_layer + 1 ..= max_layer).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                let neighbors = self.get_neighbors(curr_node, layer);
                for &nb in &neighbors {
                    let d = cosine_distance_normalized(query, self.get_vector(nb).unwrap());
                    if d < curr_dist {
                        curr_dist = d;
                        curr_node = nb;
                        changed = true;
                    }
                }
            }
        }
        
        let search_start_layer = max_layer.min(node_layer);
        let mut ep_nodes = vec![Candidate { id: curr_node, dist: curr_dist }];
        
        for layer in (0 ..= search_start_layer).rev() {
            visited.next_gen(); // CLEAR VISITED LIST BETWEEN LAYERS
            
            let mut candidates = BinaryHeap::new();
            let mut results = BinaryHeap::new();
            
            for &ep in &ep_nodes {
                candidates.push(ep);
                results.push(MaxCandidate { id: ep.id, dist: ep.dist });
                visited.mark_visited(ep.id);
            }
            
            while let Some(c) = candidates.pop() {
                let farthest = results.peek().unwrap().dist;
                if results.len() >= self.config.ef_construction && c.dist > farthest {
                    break;
                }
                
                let neighbors = self.get_neighbors(c.id, layer);
                for &nb in &neighbors {
                    if !visited.mark_visited(nb) {
                        let d = cosine_distance_normalized(query, self.get_vector(nb).unwrap());
                        let farthest = results.peek().unwrap().dist;
                        if results.len() < self.config.ef_construction || d < farthest {
                            candidates.push(Candidate { id: nb, dist: d });
                            results.push(MaxCandidate { id: nb, dist: d });
                            if results.len() > self.config.ef_construction {
                                results.pop();
                            }
                        }
                    }
                }
            }
            
            let mut best_candidates: Vec<Candidate> = results.into_iter()
                .map(|mc| Candidate { id: mc.id, dist: mc.dist })
                .collect();
            best_candidates.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
            
            let m_max = self.config.max_edges(layer);
            let selected_neighbors = if layer == 0 {
                self.select_neighbors_heuristic(&best_candidates, m_max, query)
            } else {
                best_candidates.iter().take(m_max).map(|c| c.id).collect()
            };
            
            self.set_neighbors(id, layer, &selected_neighbors);
            
            for &nb in &selected_neighbors {
                self.add_edge_bidirectional(nb, id, layer);
            }
            
            ep_nodes = best_candidates;
        }
        
        if node_layer > max_layer {
            let mut write_ep = self.entry_point.write();
            if node_layer > self.max_layer.load(AtomicOrdering::Acquire) {
                *write_ep = Some(id);
                self.max_layer.store(node_layer, AtomicOrdering::Release);
            }
        }
        
        self.put_visited_list(visited);
    }

    pub fn insert(&mut self, external_id: u64, vector: Vec<f32>) {
        let start_id = self.len.load(AtomicOrdering::SeqCst);
        let end_id = start_id + 1;
        
        self.vectors.resize(end_id * self.config.dim, 0.0);
        self.external_ids.resize(end_id, 0);
        self.node_max_layers.resize(end_id, 0);
        self.neighbors0.resize(end_id * self.config.m0, 0);
        self.neighbors0_counts.resize(end_id, 0);
        self.neighbors_upper.resize(end_id, Vec::new());
        self.node_locks.get_mut().resize_with(end_id, || RwLock::new(()));
        
        let v_slice = unsafe { self.vectors.get_slice_mut(start_id * self.config.dim, self.config.dim) };
        v_slice.copy_from_slice(&vector);
        normalize(v_slice);
        
        self.external_ids.as_mut_slice()[start_id] = external_id;
        
        let ml = 1.0 / (self.config.m as f64).ln();
        let mut rng = SmallRng::seed_from_u64(start_id as u64 + 0x12345);
        let uniform: f64 = rng.gen_range(0.0001..1.0);
        let l = (-uniform.ln() * ml).floor() as usize;
        self.node_max_layers.as_mut_slice()[start_id] = l;
        
        if l > 0 {
            self.neighbors_upper.as_mut_slice()[start_id] = vec![Vec::new(); l];
        }
        
        self.insert_internal(start_id as u32);
        self.len.fetch_add(1, AtomicOrdering::SeqCst);
    }

    pub fn insert_parallel(&mut self, entries: &[(u64, Vec<f32>)]) {
        if entries.is_empty() { return; }
        
        let start_id = self.len.load(AtomicOrdering::SeqCst);
        let end_id = start_id + entries.len();
        
        self.vectors.resize(end_id * self.config.dim, 0.0);
        self.external_ids.resize(end_id, 0);
        self.node_max_layers.resize(end_id, 0);
        self.neighbors0.resize(end_id * self.config.m0, 0);
        self.neighbors0_counts.resize(end_id, 0);
        self.neighbors_upper.resize(end_id, Vec::new());
        self.node_locks.get_mut().resize_with(end_id, || RwLock::new(()));
        
        let vectors_slice = self.vectors.as_mut_slice();
        let external_ids_slice = self.external_ids.as_mut_slice();
        let node_max_layers_slice = self.node_max_layers.as_mut_slice();
        let neighbors_upper_slice = self.neighbors_upper.as_mut_slice();
        
        let ml = 1.0 / (self.config.m as f64).ln();
        let dim = self.config.dim;
        
        vectors_slice[start_id * dim .. end_id * dim].par_chunks_mut(dim)
            .zip(&mut external_ids_slice[start_id .. end_id])
            .zip(&mut node_max_layers_slice[start_id .. end_id])
            .zip(&mut neighbors_upper_slice[start_id .. end_id])
            .enumerate()
            .for_each(|(i, (((v_slice, ext_id), max_layer), upper))| {
                let id = start_id + i;
                let (e, vec) = &entries[i];
                
                *ext_id = *e;
                v_slice.copy_from_slice(vec);
                normalize(v_slice);
                
                let mut rng = SmallRng::seed_from_u64(id as u64 + 0x12345);
                let uniform: f64 = rng.gen_range(0.0001..1.0);
                let l = (-uniform.ln() * ml).floor() as usize;
                *max_layer = l;
                
                if l > 0 {
                    *upper = vec![Vec::new(); l];
                }
            });
            
        let ids: Vec<u32> = (start_id as u32 .. end_id as u32).collect();
        let seed_count = 1000.min(ids.len());
        
        for &id in &ids[..seed_count] {
            self.insert_internal(id);
            self.len.fetch_add(1, AtomicOrdering::SeqCst);
        }
        
        if seed_count < ids.len() {
            ids[seed_count..].par_iter().for_each(|&id| {
                self.insert_internal(id);
            });
            self.len.fetch_add(ids.len() - seed_count, AtomicOrdering::SeqCst);
        }
    }

    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<(u64, f32)> {
        let ep = match *self.entry_point.read() {
            Some(e) => e,
            None => return Vec::new(),
        };
        
        let mut q_vec = query.to_vec();
        normalize(&mut q_vec);
        
        let mut curr_ep = ep;
        let mut curr_dist = cosine_distance_normalized(&q_vec, self.get_vector(curr_ep).unwrap());
        
        let max_layer = self.max_layer.load(AtomicOrdering::Acquire);
        
        for layer in (1 ..= max_layer).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                let neighbors = self.get_neighbors(curr_ep, layer);
                for &nb in &neighbors {
                    let d = cosine_distance_normalized(&q_vec, self.get_vector(nb).unwrap());
                    if d < curr_dist {
                        curr_dist = d;
                        curr_ep = nb;
                        changed = true;
                    }
                }
            }
        }
        
        let mut visited = self.get_visited_list();
        let ef = ef_search.max(k);
        let results = self.search_layer(&q_vec, curr_ep, curr_dist, ef, 0, &mut visited);
        self.put_visited_list(visited);
        
        let mut final_results: Vec<(u64, f32)> = results.into_iter()
            .map(|c| (self.external_ids.get()[c.id as usize], c.dist))
            .collect();
            
        final_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        final_results.truncate(k);
        final_results
    }
}
