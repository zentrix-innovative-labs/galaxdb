//! Mmap'd HNSW graph file format for persistent storage.
//!
//! The base HNSW graph is stored on disk and memory-mapped read-only.
//! This allows the graph to be loaded instantly without deserialization,
//! and the OS manages paging — only accessed pages are loaded into RAM.
//!
//! ## On-Disk Layout
//!
//! ```text
//! ┌─────────────────────────────────────────────┐
//! │ Header (fixed size: 64 bytes)               │
//! │   magic: u32 = 0x484E5357 ("HNSW")          │
//! │   version: u32 = 1                          │
//! │   dim: u32                                  │
//! │   m: u32                                    │
//! │   m0: u32                                   │
//! │   ef_construction: u32                      │
//! │   max_layer: u32                            │
//! │   entry_point: u32                          │
//! │   node_count: u64                           │
//! │   vectors_offset: u64                       │
//! │   adjacency_offset: u64                     │
//! │   padding to 64 bytes                       │
//! ├─────────────────────────────────────────────┤
//! │ Node metadata (node_count × 16 bytes)       │
//! │   external_id: u64                          │
//! │   max_layer: u32                            │
//! │   _reserved: u32                            │
//! ├─────────────────────────────────────────────┤
//! │ Vectors (node_count × dim × 4 bytes)        │
//! │   Contiguous f32 arrays, one per node       │
//! ├─────────────────────────────────────────────┤
//! │ Adjacency lists                             │
//! │   Per node, per layer:                      │
//! │     neighbor_count: u16                     │
//! │     neighbors: [u32; neighbor_count]        │
//! └─────────────────────────────────────────────┘
//! ```

use std::io::Write;
use std::path::Path;

use galaxdb_common::{GalaxError, GalaxResult};
use memmap2::Mmap;

use crate::hnsw::HnswGraph;

/// Magic number for HNSW files.
const HNSW_MAGIC: u32 = 0x484E5357; // "HNSW"
/// File format version.
const HNSW_VERSION: u32 = 1;
/// Header size in bytes.
const HEADER_SIZE: usize = 64;
/// Per-node metadata size: external_id(8) + max_layer(4) + reserved(4) = 16.
const NODE_META_SIZE: usize = 16;

/// Metadata from the HNSW file header.
#[derive(Debug, Clone)]
pub struct HnswFileHeader {
    pub dim: u32,
    pub m: u32,
    pub m0: u32,
    pub ef_construction: u32,
    pub max_layer: u32,
    pub entry_point: u32,
    pub node_count: u64,
    pub vectors_offset: u64,
    pub adjacency_offset: u64,
}

/// Write an in-memory HNSW graph to a file.
///
/// The file can later be mmap'd read-only for fast loading.
pub fn write_hnsw_file(graph: &HnswGraph, path: &Path) -> GalaxResult<()> {
    let mut file = std::fs::File::create(path)?;
    let config = graph.config();
    let node_count = graph.len() as u64;
    let dim = config.dim as u32;

    // Calculate section offsets
    let node_meta_offset = HEADER_SIZE as u64;
    let vectors_offset = node_meta_offset + node_count * NODE_META_SIZE as u64;
    let adjacency_offset = vectors_offset + node_count * dim as u64 * 4;

    // Write header (64 bytes)
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(&HNSW_MAGIC.to_le_bytes());
    header[4..8].copy_from_slice(&HNSW_VERSION.to_le_bytes());
    header[8..12].copy_from_slice(&dim.to_le_bytes());
    header[12..16].copy_from_slice(&(config.m as u32).to_le_bytes());
    header[16..20].copy_from_slice(&(config.m0 as u32).to_le_bytes());
    header[20..24].copy_from_slice(&(config.ef_construction as u32).to_le_bytes());
    header[24..28].copy_from_slice(&(graph.max_layer() as u32).to_le_bytes());
    header[28..32].copy_from_slice(&graph.entry_point().unwrap_or(0).to_le_bytes());
    header[32..40].copy_from_slice(&node_count.to_le_bytes());
    header[40..48].copy_from_slice(&vectors_offset.to_le_bytes());
    header[48..56].copy_from_slice(&adjacency_offset.to_le_bytes());
    // bytes 56..64 are padding (zeros)
    file.write_all(&header)?;

    // Write node metadata
    for i in 0..graph.len() {
        let ext_id = graph.get_external_id(i as u32).unwrap_or(0);
        let _vector = graph.get_vector(i as u32).unwrap();
        // We need max_layer from the node — get it from the graph
        // For now, infer from the adjacency structure
        let max_layer = graph.node_max_layer(i as u32);
        let mut meta = [0u8; NODE_META_SIZE];
        meta[0..8].copy_from_slice(&ext_id.to_le_bytes());
        meta[8..12].copy_from_slice(&(max_layer as u32).to_le_bytes());
        // bytes 12..16 reserved
        file.write_all(&meta)?;
    }

    // Write vectors (contiguous f32 arrays)
    for i in 0..graph.len() {
        let vector = graph.get_vector(i as u32).unwrap();
        for &val in vector {
            file.write_all(&val.to_le_bytes())?;
        }
    }

    // Write adjacency lists
    for i in 0..graph.len() {
        let max_layer = graph.node_max_layer(i as u32);
        for layer in 0..=max_layer {
            let neighbors = graph.get_neighbors(i as u32, layer);
            let count = neighbors.len() as u16;
            file.write_all(&count.to_le_bytes())?;
            for n in &neighbors {
                file.write_all(&n.to_le_bytes())?;
            }
        }
    }

    file.sync_all()?;
    Ok(())
}

/// Read-only mmap'd HNSW graph file.
///
/// The graph data is memory-mapped from disk. Only accessed pages are
/// loaded into RAM by the OS. This allows instant loading of large graphs
/// without explicit deserialization.
pub struct MmapHnswGraph {
    mmap: Mmap,
    header: HnswFileHeader,
}

impl MmapHnswGraph {
    /// Open an HNSW graph file for read-only access via mmap.
    pub fn open(path: &Path) -> GalaxResult<Self> {
        let file = std::fs::File::open(path)?;
        // Safety: the file is opened read-only and we don't modify it
        let mmap = unsafe {
            Mmap::map(&file).map_err(GalaxError::Io)?
        };

        if mmap.len() < HEADER_SIZE {
            return Err(GalaxError::Internal("HNSW file too small for header".into()));
        }

        // Parse header
        let data = &mmap[..];
        let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
        if magic != HNSW_MAGIC {
            return Err(GalaxError::Internal(format!(
                "not an HNSW file: magic {:#x}, expected {:#x}", magic, HNSW_MAGIC
            )));
        }
        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if version != HNSW_VERSION {
            return Err(GalaxError::Internal(format!(
                "unsupported HNSW version: {}, expected {}", version, HNSW_VERSION
            )));
        }

        let header = HnswFileHeader {
            dim: u32::from_le_bytes(data[8..12].try_into().unwrap()),
            m: u32::from_le_bytes(data[12..16].try_into().unwrap()),
            m0: u32::from_le_bytes(data[16..20].try_into().unwrap()),
            ef_construction: u32::from_le_bytes(data[20..24].try_into().unwrap()),
            max_layer: u32::from_le_bytes(data[24..28].try_into().unwrap()),
            entry_point: u32::from_le_bytes(data[28..32].try_into().unwrap()),
            node_count: u64::from_le_bytes(data[32..40].try_into().unwrap()),
            vectors_offset: u64::from_le_bytes(data[40..48].try_into().unwrap()),
            adjacency_offset: u64::from_le_bytes(data[48..56].try_into().unwrap()),
        };

        Ok(Self { mmap, header })
    }

    /// Get the file header metadata.
    pub fn header(&self) -> &HnswFileHeader {
        &self.header
    }

    /// Number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.header.node_count as usize
    }

    /// Get the external ID for a node.
    pub fn external_id(&self, node_idx: u32) -> u64 {
        let offset = HEADER_SIZE + node_idx as usize * NODE_META_SIZE;
        u64::from_le_bytes(self.mmap[offset..offset + 8].try_into().unwrap())
    }

    /// Get the max layer for a node.
    pub fn node_max_layer(&self, node_idx: u32) -> usize {
        let offset = HEADER_SIZE + node_idx as usize * NODE_META_SIZE + 8;
        u32::from_le_bytes(self.mmap[offset..offset + 4].try_into().unwrap()) as usize
    }

    /// Get a vector by node index. Returns a slice into the mmap'd data (zero-copy).
    pub fn get_vector(&self, node_idx: u32) -> &[f32] {
        let dim = self.header.dim as usize;
        let offset = self.header.vectors_offset as usize + node_idx as usize * dim * 4;
        let bytes = &self.mmap[offset..offset + dim * 4];
        // Safety: f32 is 4 bytes, properly aligned in the file
        unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const f32, dim)
        }
    }

    /// Get the neighbors of a node at a specific layer.
    ///
    /// This requires scanning the adjacency section from the start for the
    /// target node, since adjacency lists are variable-length.
    /// For production use, a node-to-offset index should be built at load time.
    pub fn get_neighbors(&self, node_idx: u32, layer: usize) -> Vec<u32> {
        // Scan to the target node's adjacency data
        let mut offset = self.header.adjacency_offset as usize;

        for i in 0..=node_idx as usize {
            let node_max_layer = self.node_max_layer(i as u32);
            for l in 0..=node_max_layer {
                let count = u16::from_le_bytes(
                    self.mmap[offset..offset + 2].try_into().unwrap()
                ) as usize;
                offset += 2;

                if i == node_idx as usize && l == layer {
                    let mut neighbors = Vec::with_capacity(count);
                    for _ in 0..count {
                        let n = u32::from_le_bytes(
                            self.mmap[offset..offset + 4].try_into().unwrap()
                        );
                        neighbors.push(n);
                        offset += 4;
                    }
                    return neighbors;
                }

                offset += count * 4; // skip neighbor data
            }
        }

        Vec::new()
    }

    /// Get the entry point node index.
    pub fn entry_point(&self) -> u32 {
        self.header.entry_point
    }

    /// Get the maximum layer in the graph.
    pub fn max_layer(&self) -> usize {
        self.header.max_layer as usize
    }

    /// Get the dimensionality.
    pub fn dim(&self) -> usize {
        self.header.dim as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::{HnswConfig, HnswGraph};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    fn random_vector(rng: &mut SmallRng, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    #[test]
    fn write_and_read_roundtrip() {
        let dim = 32;
        let config = HnswConfig::new(dim).with_m(8).with_ef_construction(50);
        let mut graph = HnswGraph::new(config);
        let mut rng = SmallRng::seed_from_u64(42);

        for i in 0..100 {
            let v = random_vector(&mut rng, dim);
            graph.insert(i as u64, v);
        }

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.hnsw");

        // Write
        write_hnsw_file(&graph, &path).unwrap();

        // Read via mmap
        let mmap_graph = MmapHnswGraph::open(&path).unwrap();

        assert_eq!(mmap_graph.node_count(), 100);
        assert_eq!(mmap_graph.dim(), dim);
        assert_eq!(mmap_graph.header().m, 8);

        // Verify vectors match
        for i in 0..100u32 {
            let orig = graph.get_vector(i).unwrap();
            let loaded = mmap_graph.get_vector(i);
            assert_eq!(orig.len(), loaded.len());
            for (a, b) in orig.iter().zip(loaded.iter()) {
                assert!((a - b).abs() < 1e-7, "vector mismatch at node {}", i);
            }
        }

        // Verify external IDs match
        for i in 0..100u32 {
            assert_eq!(
                graph.get_external_id(i).unwrap(),
                mmap_graph.external_id(i),
                "external_id mismatch at node {}", i
            );
        }

        // Verify entry point
        assert_eq!(
            graph.entry_point().unwrap(),
            mmap_graph.entry_point()
        );
    }

    #[test]
    fn mmap_vectors_are_zero_copy() {
        let dim = 4;
        let config = HnswConfig::new(dim).with_m(4).with_ef_construction(20);
        let mut graph = HnswGraph::new(config);
        graph.insert(0, vec![1.0, 2.0, 3.0, 4.0]);
        graph.insert(1, vec![5.0, 6.0, 7.0, 8.0]);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.hnsw");
        write_hnsw_file(&graph, &path).unwrap();

        let mmap_graph = MmapHnswGraph::open(&path).unwrap();

        // get_vector returns a slice into the mmap — no allocation
        // Vectors are normalized on insert, so compare against normalized values
        let mut expected0 = vec![1.0f32, 2.0, 3.0, 4.0];
        crate::distance::normalize(&mut expected0);
        let v0 = mmap_graph.get_vector(0);
        assert_eq!(v0, expected0.as_slice());

        let mut expected1 = vec![5.0f32, 6.0, 7.0, 8.0];
        crate::distance::normalize(&mut expected1);
        let v1 = mmap_graph.get_vector(1);
        assert_eq!(v1, expected1.as_slice());
    }

    #[test]
    fn bad_magic_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.hnsw");
        std::fs::write(&path, [0u8; 64]).unwrap();
        assert!(MmapHnswGraph::open(&path).is_err());
    }

    #[test]
    fn too_small_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.hnsw");
        std::fs::write(&path, [0u8; 10]).unwrap();
        assert!(MmapHnswGraph::open(&path).is_err());
    }
}
