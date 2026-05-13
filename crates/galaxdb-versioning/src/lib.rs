//! GalaxDB Versioning — Merkle DAG, Version Tags, Lance Exporter, MinHash Dedup.

pub mod merkle;
pub mod tags;
pub mod guardrails;
pub mod export;
pub mod minhash;
pub mod dedup_grouping;

pub use merkle::{MerkleDag, MerkleRoot, VersionEntry};
pub use tags::{TagCatalog, VersionTag, TrainingTagMetadata, ConsistencyMode, VersionResolution};
pub use guardrails::{validate_version_query, SEMANTIC_FRESH_WARNING};
pub use export::{
    ExportError, ExportResult, ExportStats, ExportedRow, FieldValue, InMemoryLineageSink,
    LanceExportSource, LanceExporter, TrainingExportLineage, TrainingExportLineageSink,
    TrainingPrecision,
};
pub use minhash::{
    estimate_jaccard, jaccard_estimate_from_bytes, MinHashDedup, MinHashSignature, NUM_HASHES,
    SIGNATURE_BYTES,
};
pub use dedup_grouping::{
    group_near_duplicates, DedupRowId, NearDuplicateGrouping, BANDS,
    NEAR_DUPLICATE_JACCARD_THRESHOLD, ROWS_PER_BAND,
};
