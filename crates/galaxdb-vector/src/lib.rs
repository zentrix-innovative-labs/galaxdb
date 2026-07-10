//! GalaxDB Vector — HNSW index, Delta Buffer, Quantizer (SQ8/FP16/RaBitQ).
//!
//! This crate implements the vector search subsystem:
//! - **HNSW graph**: Hierarchical Navigable Small World index for approximate
//!   nearest neighbor search (Malkov & Yashunin 2018)
//! - **Distance computation**: Cosine similarity with AVX2+FMA SIMD acceleration
//! - **Delta buffer**: In-memory buffer for recent inserts/deletes (Month 3)
//! - **Quantization**: SQ8, FP16, RaBitQ for memory-efficient storage (Month 3)

pub mod delta_buffer;
pub mod diskann;
pub mod distance;
pub mod hnsw;
pub mod hnsw_file;
pub mod merge;
pub mod quantizer;
pub mod semantic_match;

pub use delta_buffer::{DeltaBuffer, DeltaSearchResult, union_and_rerank};
pub use diskann::{DiskAnnConfig, DiskAnnIndex, Metric as DiskAnnMetric};
pub use distance::{cosine_distance, cosine_similarity, normalize};
pub use hnsw::{HnswConfig, HnswGraph};
pub use hnsw_file::{MmapHnswGraph, write_hnsw_file};
pub use merge::merge_hnsw;
pub use quantizer::{Quantizer, Sq8Quantizer, Fp16Quantizer, RabitqQuantizer, select_default_quantizer};
pub use semantic_match::{
    SemanticMatchConfig, SemanticMatchResult, SearchStrategy,
    execute_semantic_match, execute_brute_force_filtered, choose_strategy,
};
