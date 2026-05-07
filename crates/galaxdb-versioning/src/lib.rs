//! GalaxDB Versioning — Merkle DAG, Version Tags, Lance Exporter, MinHash Dedup.

pub mod merkle;
pub mod tags;
pub mod guardrails;

pub use merkle::{MerkleDag, MerkleRoot, VersionEntry};
pub use tags::{TagCatalog, VersionTag, TrainingTagMetadata, ConsistencyMode, VersionResolution};
pub use guardrails::{validate_version_query, SEMANTIC_FRESH_WARNING};
