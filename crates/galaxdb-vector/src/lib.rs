//! GalaxDB Vector — HNSW index, Delta Buffer, Quantizer (SQ8/FP16/RaBitQ).
//!
//! This crate implements the vector search subsystem:
//! - **HNSW graph**: Hierarchical Navigable Small World index for approximate
//!   nearest neighbor search (Malkov & Yashunin 2018)
//! - **Distance computation**: Cosine similarity with AVX2+FMA SIMD acceleration
//! - **Delta buffer**: In-memory buffer for recent inserts/deletes (Month 3)
//! - **Quantization**: SQ8, FP16, RaBitQ for memory-efficient storage (Month 3)

pub mod distance;
pub mod hnsw;
pub mod hnsw_file;

pub use distance::{cosine_distance, cosine_similarity, normalize};
pub use hnsw::{HnswConfig, HnswGraph};
pub use hnsw_file::{MmapHnswGraph, write_hnsw_file};
