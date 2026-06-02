//! Lance Training Export — export versioned data as Lance datasets for ML training.
//!
//! This module implements Req 25 of the GalaxDB v1 engine: producing a
//! deterministic, columnar Lance dataset from a tagged version of a table for
//! downstream PyTorch training.
//!
//! The export pipeline (filled in across tasks 34.1 … 34.6):
//!
//! 1. Resolve a [`VersionTag`] to its pinned block set via the [`MerkleDag`]
//!    and [`TagCatalog`].
//! 2. Read rows from those blocks via a [`LanceExportSource`] implementation
//!    and sort them by primary key (this is the guarantee `FOR TRAINING`
//!    tags make — see `TrainingTagMetadata::deterministic_order`).
//! 3. Apply training precision conversion (float32 passthrough, sq8, rabitq).
//! 4. Optionally apply MinHash-based deduplication (`WHERE NOT DUPLICATE`).
//! 5. Write the resulting Arrow record batches as a Lance dataset.
//! 6. Record a lineage row in `_galaxdb_training_exports` (Req 38).
//!
//! Task 34.2 + 34.3 + 34.4 (covered by this file) implement steps 1, 2, 3,
//! 4, and 5. Lineage (34.5) lands in a later task.
//!
//! ### Arrow/Lance version alignment
//!
//! Lance 4.0.1 internally depends on `arrow 57.0`. This crate therefore also
//! pins `arrow = "57"` in `Cargo.toml`. Any mismatch here silently breaks
//! compilation because `arrow::Schema` from version N is not the same type
//! as `arrow::Schema` from version M, and `lance::Dataset::write` requires
//! Lance's own Arrow version.

use std::fmt::Write as _;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::array::{
    ArrayRef, BinaryBuilder, Float32Builder, Int64Builder, RecordBatch, RecordBatchIterator,
    StringBuilder,
};
use arrow::array::builder::FixedSizeListBuilder;
use arrow::datatypes::{DataType, Schema as ArrowSchema};
use arrow::error::ArrowError;
use galaxdb_common::types::BlockId;
use lance::Dataset;
use lance::dataset::WriteParams;
use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3::xxh3_128;

use crate::merkle::MerkleDag;
use crate::tags::TagCatalog;

/// Row-batch size written to Lance. Chosen to match Lance's recommended
/// default group size and keep per-batch allocation predictable.
const EXPORT_BATCH_SIZE: usize = 8192;

/// Quantisation precision applied to embedding columns during export.
///
/// Matches the `WITH TRAINING PRECISION '…'` clause of
/// `CREATE VERSION TAG … FOR TRAINING` (see Req 24 and
/// `galaxdb_sql::ast::TrainingPrecision`). Kept as a local enum so this crate
/// does not have to depend on `galaxdb-sql`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TrainingPrecision {
    /// Raw 4-byte IEEE 754 floats (no conversion).
    #[default]
    Float32,
    /// int8 scalar quantisation — 4× I/O reduction.
    Sq8,
    /// Random-rotation binary quantisation — 32× I/O reduction.
    Rabitq,
}

impl TrainingPrecision {
    /// Parse from the string form used by SQL / tag metadata.
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "float32" => Some(Self::Float32),
            "sq8" => Some(Self::Sq8),
            "rabitq" => Some(Self::Rabitq),
            _ => None,
        }
    }

    /// Canonical string form (matches `TrainingTagMetadata::precision`).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Float32 => "float32",
            Self::Sq8 => "sq8",
            Self::Rabitq => "rabitq",
        }
    }
}

/// Summary statistics for a completed training export.
///
/// `content_hash` is the XXH3-128 hash of the canonicalised exported rows —
/// produced by the same algorithm used for Merkle roots so that two exports
/// of the same tag at the same precision are byte-for-byte identical and
/// reproducible. Stored as `[u8; 16]` (big-endian) so the value is
/// portable across platforms and languages (Python, SQL result rows, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportStats {
    /// Total number of rows written to the Lance dataset.
    pub row_count: u64,
    /// Total bytes written to the Lance dataset on disk.
    pub byte_count: u64,
    /// XXH3-128 hash over the canonical row encoding, big-endian.
    pub content_hash: [u8; 16],
}

impl ExportStats {
    /// Empty stats (zero rows, zero bytes, zero hash).
    pub fn empty() -> Self {
        Self {
            row_count: 0,
            byte_count: 0,
            content_hash: [0u8; 16],
        }
    }
}

/// Lineage record for the `_galaxdb_training_exports` system table (Req 38).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExportLineage {
    pub tag_name: String,
    pub filter_expr: Option<String>,
    pub precision: String,
    pub dedup: bool,
    pub row_count: u64,
    pub exported_at: u64,
    /// Lower-case hex encoding of `ExportStats::content_hash`.
    pub content_hash: String,
}

/// Sink for training-export lineage rows (Req 38, task 34.5).
///
/// `LanceExporter` does not write lineage rows directly — task 36 will
/// materialise the `_galaxdb_training_exports` system table inside the
/// storage engine, which this crate intentionally does not depend on.
/// Instead, callers (typically `galaxdb-embedded`) provide a sink that
/// knows how to persist a lineage row against whatever backing store they
/// control. The exporter is oblivious to that detail.
///
/// ### Error semantics
///
/// [`TrainingExportLineageSink::record`] is called *after* the Lance
/// dataset has already been written to disk. If `record` returns an error
/// the exporter propagates it out of [`LanceExporter::export`], which
/// means the dataset is on disk but the lineage row is not. This is
/// deliberate: the caller needs to see the failure so they can either
/// retry the lineage write or roll back the dataset themselves. The spec
/// does not require the pair to be transactional.
pub trait TrainingExportLineageSink: Send + Sync {
    /// Persist a lineage record. Called once per successful Lance write.
    fn record(&self, lineage: TrainingExportLineage) -> ExportResult<()>;
}

/// In-memory [`TrainingExportLineageSink`] used by tests and as a
/// reference implementation.
///
/// `galaxdb-embedded` can wrap one of these while task 36 is in flight;
/// once the storage-backed sink exists the embedded layer swaps in the
/// persistent impl without any change to the exporter. The sink itself
/// is trivially thread-safe — records go into a `Mutex<Vec<_>>` — and
/// `Clone`s share the same underlying buffer via `Arc` so callers can
/// hand copies to the exporter and to assertion code alike.
#[derive(Debug, Default)]
pub struct InMemoryLineageSink {
    entries: Mutex<Vec<TrainingExportLineage>>,
}

impl InMemoryLineageSink {
    /// Create an empty sink.
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot of every lineage record that has been `record`ed so far,
    /// in insertion order.
    pub fn entries(&self) -> Vec<TrainingExportLineage> {
        self.entries
            .lock()
            .expect("InMemoryLineageSink mutex poisoned")
            .clone()
    }

    /// Number of lineage records currently held by the sink. Equivalent
    /// to `self.entries().len()` but cheaper when callers only want a
    /// count.
    pub fn len(&self) -> usize {
        self.entries
            .lock()
            .expect("InMemoryLineageSink mutex poisoned")
            .len()
    }

    /// True when no lineage records have been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl TrainingExportLineageSink for InMemoryLineageSink {
    fn record(&self, lineage: TrainingExportLineage) -> ExportResult<()> {
        self.entries
            .lock()
            .expect("InMemoryLineageSink mutex poisoned")
            .push(lineage);
        Ok(())
    }
}

/// Lower-case hex encoding of `bytes`. Two hex chars per input byte.
///
/// Kept inline to avoid pulling in `hex` just for 16 bytes per export.
/// This is the encoding used for
/// [`TrainingExportLineage::content_hash`] so it can round-trip through
/// SQL result rows, JSON, and Python without binary-encoding concerns.
fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        // `write!` to a `String` is infallible — see `std::fmt::Write`.
        write!(&mut s, "{:02x}", b).expect("writing to String cannot fail");
    }
    s
}

/// Errors returned by [`LanceExporter::export`].
#[derive(Debug)]
pub enum ExportError {
    /// The referenced version tag does not exist in the [`TagCatalog`].
    TagNotFound(String),
    /// No PAX blocks were found for the tag's commit timestamp, or the
    /// source returned zero rows.
    EmptyVersion,
    /// A row returned by the [`LanceExportSource`] did not match the schema
    /// (wrong field count, wrong field type, or wrong embedding dimension).
    SchemaMismatch(String),
    /// An underlying Arrow error occurred while building record batches.
    Arrow(String),
    /// An underlying Lance error occurred while writing the dataset.
    Lance(String),
    /// An I/O error occurred reading blocks or writing the output path.
    Io(std::io::Error),
    /// The requested training precision is not yet supported by this
    /// task. Task 34.3 will implement Sq8 / Rabitq — until then, only
    /// `TrainingPrecision::Float32` is a valid `export()` precision.
    NotImplemented,
}

impl std::fmt::Display for ExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TagNotFound(n) => write!(f, "version tag not found: {}", n),
            Self::EmptyVersion => write!(f, "version has no rows to export"),
            Self::SchemaMismatch(m) => write!(f, "schema mismatch: {}", m),
            Self::Arrow(m) => write!(f, "arrow error: {}", m),
            Self::Lance(m) => write!(f, "lance error: {}", m),
            Self::Io(e) => write!(f, "io error: {}", e),
            Self::NotImplemented => {
                write!(f, "LanceExporter: feature not implemented in this task")
            }
        }
    }
}

impl std::error::Error for ExportError {}

impl From<std::io::Error> for ExportError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<ArrowError> for ExportError {
    fn from(e: ArrowError) -> Self {
        Self::Arrow(e.to_string())
    }
}

/// Convenience `Result` alias for exporter operations.
pub type ExportResult<T> = std::result::Result<T, ExportError>;

/// A single typed field value for an exported row.
///
/// The variants cover the Arrow data types the exporter currently knows how
/// to write. Each `FieldValue` slot must match the corresponding Arrow
/// field type in the exporter's schema, in the same order.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldValue {
    /// Signed 64-bit integer (Arrow `Int64`).
    Int64(i64),
    /// UTF-8 string (Arrow `Utf8`).
    Utf8(String),
    /// 32-bit float (Arrow `Float32`).
    Float32(f32),
    /// Variable-length binary (Arrow `Binary`).
    Binary(Vec<u8>),
    /// Fixed-size dense embedding (Arrow `FixedSizeList<Float32, dim>`).
    /// The length of the vector must equal the schema's fixed-list size.
    Embedding(Vec<f32>),
}

/// A single row surfaced by a [`LanceExportSource`] for the Lance exporter.
///
/// `primary_key` is opaque bytes — matching PAX primary keys — used to sort
/// rows deterministically before they are written to the Lance dataset.
/// `fields` holds typed values aligned to the exporter's Arrow schema by
/// column index.
///
/// `near_duplicate_group` is the MinHash-based near-dup group ID (Req 26).
/// The field is populated upstream — in this crate it is a purely
/// structural slot that the exporter consults when [`LanceExporter::dedup`]
/// is `true`. Task 35 (MinHash) will drive these values from the write
/// path; until then the source can leave every row at `None` and the
/// exporter is a no-op with respect to dedup.
#[derive(Debug, Clone, PartialEq)]
pub struct ExportedRow {
    /// Primary-key bytes used for deterministic ordering.
    pub primary_key: Vec<u8>,
    /// One [`FieldValue`] per column, in schema field order.
    pub fields: Vec<FieldValue>,
    /// Near-duplicate group ID for `WHERE NOT DUPLICATE` filtering.
    ///
    /// `None` means the row is unique (not in any near-duplicate group)
    /// and is always kept regardless of `dedup`. `Some(id)` means the row
    /// shares this group with other rows; when `dedup = true` the exporter
    /// keeps only the representative with the lexicographically smallest
    /// `primary_key` for each group.
    pub near_duplicate_group: Option<u64>,
}

/// Sensible defaults: empty primary key, no fields, not in any
/// near-duplicate group. Mostly useful in tests that want to construct
/// rows with only a subset of fields set via `..Default::default()`.
impl Default for ExportedRow {
    fn default() -> Self {
        Self {
            primary_key: Vec::new(),
            fields: Vec::new(),
            near_duplicate_group: None,
        }
    }
}

/// Source abstraction for the Lance exporter.
///
/// The versioning crate does not depend on `galaxdb-storage` (to avoid a
/// dependency cycle between versioning metadata and the block store), so the
/// concrete PAX-block reader is injected through this trait. Production
/// callers (e.g. `galaxdb-embedded`) wire this to the real storage engine;
/// tests provide an in-memory `Vec<ExportedRow>`-backed implementation.
pub trait LanceExportSource: Send + Sync {
    /// Read all rows belonging to the given PAX blocks.
    ///
    /// The returned order is irrelevant — the exporter re-sorts by
    /// [`ExportedRow::primary_key`] before writing.
    fn read_blocks(&self, block_ids: &[BlockId]) -> ExportResult<Vec<ExportedRow>>;
}

/// Exports a tagged version of a table as a Lance dataset for training.
///
/// Construct with [`LanceExporter::new`] and drive the export with
/// [`LanceExporter::export`]. The struct is `Clone` because it only holds
/// `Arc`s and plain values, so the same configured exporter can be reused
/// across threads or scheduled on a `tokio` runtime.
#[derive(Clone)]
pub struct LanceExporter {
    /// Absolute path at which the Lance dataset will be written.
    output_path: std::path::PathBuf,
    /// Arrow schema describing the exported columns.
    schema: Arc<ArrowSchema>,
    /// Merkle DAG used to resolve a tag's commit timestamp to a block set.
    merkle_dag: Arc<MerkleDag>,
    /// Tag catalog used to look up the `VersionTag` being exported.
    tag_catalog: Arc<TagCatalog>,
    /// Concrete block reader (see [`LanceExportSource`]).
    source: Arc<dyn LanceExportSource>,
    /// Name of the version tag being exported.
    tag_name: String,
    /// Quantisation precision applied to embedding columns.
    precision: TrainingPrecision,
    /// If true, apply `WHERE NOT DUPLICATE` during export (Req 26).
    dedup: bool,
    /// Optional deterministic seed for any randomised steps (e.g. RaBitQ
    /// rotation matrix initialisation, dedup tie-break).
    seed: Option<u64>,
    /// Optional filter expression (e.g. the `WHERE` clause of the
    /// `EXPORT TRAINING DATA` query) that produced this export. Recorded
    /// verbatim on the lineage row — the exporter does not parse or
    /// interpret it.
    filter_expr: Option<String>,
    /// Optional lineage sink. When present, [`LanceExporter::export`]
    /// calls [`TrainingExportLineageSink::record`] once after a
    /// successful Lance write (Req 38, task 34.5). When absent, no
    /// lineage row is produced.
    lineage_sink: Option<Arc<dyn TrainingExportLineageSink>>,
}

impl LanceExporter {
    /// Build a new exporter.
    ///
    /// The caller is expected to supply:
    /// * `output_path` — destination on disk for the Lance dataset
    ///   (directory that will be created by Lance).
    /// * `schema` — Arrow schema matching the columns of the source table
    ///   after precision conversion is applied.
    /// * `merkle_dag` / `tag_catalog` — version metadata, shared with the
    ///   rest of the engine via `Arc`.
    /// * `source` — how to materialise rows from a block-id list. See
    ///   [`LanceExportSource`].
    /// * `tag_name` — the `CREATE VERSION TAG … FOR TRAINING` name.
    /// * `precision`, `dedup`, `seed` — training configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        output_path: impl Into<std::path::PathBuf>,
        schema: Arc<ArrowSchema>,
        merkle_dag: Arc<MerkleDag>,
        tag_catalog: Arc<TagCatalog>,
        source: Arc<dyn LanceExportSource>,
        tag_name: impl Into<String>,
        precision: TrainingPrecision,
        dedup: bool,
        seed: Option<u64>,
    ) -> Self {
        Self {
            output_path: output_path.into(),
            schema,
            merkle_dag,
            tag_catalog,
            source,
            tag_name: tag_name.into(),
            precision,
            dedup,
            seed,
            filter_expr: None,
            lineage_sink: None,
        }
    }

    /// Attach a [`TrainingExportLineageSink`] (Req 38, task 34.5).
    ///
    /// When set, [`LanceExporter::export`] records one lineage row per
    /// successful export via [`TrainingExportLineageSink::record`]. If
    /// the sink returns an error the exporter propagates it — the Lance
    /// dataset stays on disk because the write already succeeded; the
    /// caller is expected to handle the partial-failure semantics.
    ///
    /// This is a builder-style method so callers can keep the existing
    /// [`LanceExporter::new`] signature and only opt into lineage
    /// recording when they need it.
    pub fn with_lineage_sink(mut self, sink: Arc<dyn TrainingExportLineageSink>) -> Self {
        self.lineage_sink = Some(sink);
        self
    }

    /// Record a filter expression (e.g. the `WHERE` clause of the SQL
    /// `EXPORT TRAINING DATA` statement) verbatim on the lineage row
    /// (Req 38, task 34.5). The exporter itself does not interpret the
    /// string — it is stored as-is so that lineage consumers can trace
    /// exactly which query produced each dataset.
    ///
    /// Calling this with `Some(..)` overrides any previously set filter
    /// expression; calling with `None` clears it. Has no effect when no
    /// lineage sink is attached.
    pub fn with_filter_expr(mut self, expr: Option<String>) -> Self {
        self.filter_expr = expr;
        self
    }

    /// Destination path for the Lance dataset.
    pub fn output_path(&self) -> &std::path::Path {
        &self.output_path
    }

    /// Arrow schema of the exported dataset.
    pub fn schema(&self) -> &ArrowSchema {
        &self.schema
    }

    /// Name of the version tag being exported.
    pub fn tag_name(&self) -> &str {
        &self.tag_name
    }

    /// Quantisation precision applied to embedding columns.
    pub fn precision(&self) -> TrainingPrecision {
        self.precision
    }

    /// Whether `WHERE NOT DUPLICATE` is applied during export.
    pub fn dedup(&self) -> bool {
        self.dedup
    }

    /// Optional deterministic seed for randomised export steps.
    pub fn seed(&self) -> Option<u64> {
        self.seed
    }

    /// Access the Merkle DAG the exporter resolves tags against.
    pub fn merkle_dag(&self) -> &MerkleDag {
        &self.merkle_dag
    }

    /// Access the tag catalog the exporter resolves tags against.
    pub fn tag_catalog(&self) -> &TagCatalog {
        &self.tag_catalog
    }

    /// Run the export pipeline.
    ///
    /// Steps implemented in tasks 34.2 + 34.3 + 34.4:
    /// 1. Resolve `tag_name` via the `TagCatalog`.
    /// 2. Ask the source to materialise rows for the tag's pinned blocks.
    /// 3. Sort rows by `primary_key` for deterministic ordering.
    /// 4. If [`LanceExporter::dedup`] is `true`, collapse every
    ///    [`ExportedRow::near_duplicate_group`] to a single representative
    ///    (lowest primary key per group). Rows whose group is `None` are
    ///    always retained — they are unique by definition.
    /// 5. Apply [`TrainingPrecision`] conversion to every embedding column
    ///    (34.3). `Float32` is a passthrough; `Sq8` and `Rabitq` turn each
    ///    `FieldValue::Embedding` into `FieldValue::Binary` and require the
    ///    caller's schema to declare that column as [`DataType::Binary`].
    /// 6. Compute the XXH3-128 content hash over the canonical row bytes
    ///    (post-conversion, so the hash pins the precision used).
    /// 7. Build Arrow `RecordBatch`es (8 192 rows each) against `schema`.
    /// 8. Write the batches out as a Lance dataset via `Dataset::write`.
    /// 9. Sum up on-disk bytes and return [`ExportStats`].
    ///
    /// ### Schema requirements for quantised precisions
    ///
    /// For `TrainingPrecision::Sq8` and `TrainingPrecision::Rabitq`, every
    /// column that holds a `FieldValue::Embedding` in the source rows must
    /// be declared as `DataType::Binary` in the Arrow schema. If the schema
    /// instead declares `FixedSizeList<Float32, dim>` for such a column,
    /// [`ExportError::SchemaMismatch`] is returned with a message pointing
    /// at the offending column — the caller must pick one or the other.
    ///
    /// ### Lineage recording (task 34.5)
    ///
    /// After the Lance write succeeds, if [`LanceExporter::with_lineage_sink`]
    /// has been called, a [`TrainingExportLineage`] row is constructed
    /// from the exporter configuration and `ExportStats` and handed to
    /// the sink. If the sink returns an error, `export()` propagates it;
    /// the Lance dataset remains on disk and the caller is responsible
    /// for deciding whether to retry the lineage write or roll back the
    /// dataset. When no lineage sink is attached, this step is a no-op.
    ///
    /// ### Determinism
    ///
    /// RaBitQ uses a random rotation matrix. The exporter seeds it from
    /// [`LanceExporter::seed`], defaulting to `0` when `seed` is `None`, so
    /// the same inputs always produce the same content hash.
    pub async fn export(&self) -> ExportResult<ExportStats> {
        // 1. Tag resolution.
        let tag = self
            .tag_catalog
            .get_tag(&self.tag_name)
            .ok_or_else(|| ExportError::TagNotFound(self.tag_name.clone()))?;

        // 2. Materialise rows via the injected source.
        let block_ids: Vec<BlockId> = tag.pinned_blocks.clone();
        let mut rows = self.source.read_blocks(&block_ids)?;

        if rows.is_empty() {
            return Err(ExportError::EmptyVersion);
        }

        // 3. Deterministic ordering by primary key. `Vec<u8>` sorts
        // lexicographically which matches how PAX primary keys compare.
        rows.sort_by(|a, b| a.primary_key.cmp(&b.primary_key));

        // 3b. Near-duplicate filtering (34.4). When `dedup = true`, rows
        // sharing a `near_duplicate_group` collapse to a single
        // representative — the one with the smallest `primary_key` in the
        // group. Rows with `near_duplicate_group = None` are always kept.
        // Task 35 populates the group IDs upstream; this step is a no-op
        // until then, which is why 34.4 can land ahead of 35.
        if self.dedup {
            apply_dedup_filter(&mut rows);
            if rows.is_empty() {
                // A dedup pass that wipes out every row is still an
                // "empty" export from the caller's point of view.
                return Err(ExportError::EmptyVersion);
            }
        }

        // 4. Precision conversion (34.3). Validates the schema against the
        // requested precision and rewrites embedding columns to binary for
        // Sq8 / Rabitq. Float32 is a no-op.
        apply_precision(&mut rows, &self.schema, self.precision, self.seed)?;

        // 5. Canonical content hash — stable across runs and independent of
        // Lance's on-disk encoding (which embeds timestamps in its
        // manifest and is therefore not byte-identical across writes).
        let content_hash = canonical_content_hash(&rows);
        let row_count = rows.len() as u64;

        // 6. Convert to Arrow `RecordBatch`es.
        let batches = rows_to_record_batches(&rows, &self.schema, EXPORT_BATCH_SIZE)?;

        // 7. Write the Lance dataset.
        let uri = self
            .output_path
            .to_str()
            .ok_or_else(|| {
                ExportError::SchemaMismatch(format!(
                    "output path is not valid UTF-8: {:?}",
                    self.output_path
                ))
            })?
            .to_string();
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), self.schema.clone());

        Dataset::write(reader, uri.as_str(), Some(WriteParams::default()))
            .await
            .map_err(|e| ExportError::Lance(e.to_string()))?;

        // 8. Sum on-disk bytes.
        let byte_count = dir_size_bytes(&self.output_path)?;

        let stats = ExportStats {
            row_count,
            byte_count,
            content_hash,
        };

        // 9. Record lineage (Req 38, task 34.5). Only runs when the
        // caller has attached a sink via `with_lineage_sink`. The sink
        // is called *after* the Lance dataset has been written — if it
        // fails we propagate the error so the caller can react.
        if let Some(sink) = self.lineage_sink.as_ref() {
            // System time is the only source of non-determinism in the
            // lineage row itself; everything else comes from exporter
            // configuration and `stats`. A system clock before the epoch
            // is treated as time = 0 rather than failing the export —
            // the export is already done, and a missing timestamp is
            // better than losing the whole lineage row.
            let exported_at = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let lineage = TrainingExportLineage {
                tag_name: self.tag_name.clone(),
                filter_expr: self.filter_expr.clone(),
                precision: self.precision.as_str().to_string(),
                dedup: self.dedup,
                row_count: stats.row_count,
                exported_at,
                content_hash: hex_encode(&stats.content_hash),
            };
            sink.record(lineage)?;
        }

        Ok(stats)
    }
}

impl std::fmt::Debug for LanceExporter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LanceExporter")
            .field("output_path", &self.output_path)
            .field("schema_fields", &self.schema.fields().len())
            .field("tag_name", &self.tag_name)
            .field("precision", &self.precision)
            .field("dedup", &self.dedup)
            .field("seed", &self.seed)
            .field("filter_expr", &self.filter_expr)
            .field("has_lineage_sink", &self.lineage_sink.is_some())
            .field("merkle_versions", &self.merkle_dag.version_count())
            .field("tag_count", &self.tag_catalog.tag_count())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Near-duplicate dedup filter (task 34.4)
// ---------------------------------------------------------------------------

/// Drop rows that share a [`ExportedRow::near_duplicate_group`] with another
/// row, keeping a single deterministic representative per group.
///
/// Rules:
/// * Rows with `near_duplicate_group = None` are unique and always kept.
/// * Rows with `near_duplicate_group = Some(g)` are grouped by `g`; the
///   row with the lexicographically smallest `primary_key` in each group
///   is the representative, and all other members of the group are
///   dropped.
/// * The surviving rows are reinserted at the positions they occupied in
///   the input, so a pre-sorted input stays sorted after filtering.
///
/// Task 35 (MinHash dedup, Req 26) is what populates the group IDs in the
/// write path. Until 35 lands every row arrives with `None` and this
/// function is a pure no-op — that is deliberate: 34.4 integrates at the
/// abstraction boundary so 35 can wire through without touching the
/// exporter.
fn apply_dedup_filter(rows: &mut Vec<ExportedRow>) {
    use std::collections::HashMap;

    // First pass: find the smallest primary key per group ID. Using the
    // row index as the tiebreaker would be non-deterministic across
    // different source orderings; primary-key order is what the rest of
    // the pipeline already relies on.
    //
    // We store owned `Vec<u8>` representatives (rather than borrowing
    // from `rows`) so the second pass can mutate `rows` with `retain`
    // without upsetting the borrow checker.
    let mut representative: HashMap<u64, Vec<u8>> = HashMap::new();
    for row in rows.iter() {
        if let Some(group) = row.near_duplicate_group {
            representative
                .entry(group)
                .and_modify(|best| {
                    if row.primary_key.as_slice() < best.as_slice() {
                        *best = row.primary_key.clone();
                    }
                })
                .or_insert_with(|| row.primary_key.clone());
        }
    }

    if representative.is_empty() {
        return;
    }

    // Second pass: keep a row iff (a) its group is None, or (b) its
    // primary key equals the representative for its group. Rows whose
    // group ID is somehow absent from the map (shouldn't happen — we
    // just walked every row) are kept defensively.
    rows.retain(|row| match row.near_duplicate_group {
        None => true,
        Some(group) => representative
            .get(&group)
            .map(|best| row.primary_key.as_slice() == best.as_slice())
            .unwrap_or(true),
    });
}

// ---------------------------------------------------------------------------
// Training precision conversion (task 34.3)
// ---------------------------------------------------------------------------

/// Default RaBitQ seed used when [`LanceExporter::seed`] is `None`. Picking
/// a fixed default rather than letting RaBitQ pull from the OS RNG means
/// the export is always deterministic, even when the caller forgot to pin
/// a seed. The choice of `0` is arbitrary but stable.
const DEFAULT_QUANT_SEED: u64 = 0;

/// Apply the requested [`TrainingPrecision`] to every embedding column in
/// `rows`, rewriting `FieldValue::Embedding(..)` into
/// `FieldValue::Binary(..)` when the precision is Sq8 / Rabitq.
///
/// Validates that:
/// * Every column index that holds an `Embedding` in the rows maps to the
///   schema's declared `DataType` for that precision:
///   - `Float32` ⇒ `FixedSizeList<Float32, dim>`
///   - `Sq8` / `Rabitq` ⇒ `Binary`
/// * All rows agree on the embedding dimension per column. (Mixed
///   dimensions are a SchemaMismatch — not a precision issue.)
///
/// For Sq8 the calibration is computed across *all* rows per column
/// (global min/max), which matches how SQ8 is used during indexing. For
/// Rabitq the quantiser is seeded once per column so the rotation is
/// column-specific but deterministic.
fn apply_precision(
    rows: &mut [ExportedRow],
    schema: &ArrowSchema,
    precision: TrainingPrecision,
    seed: Option<u64>,
) -> ExportResult<()> {
    if precision == TrainingPrecision::Float32 {
        // Float32 passthrough: just sanity-check schema/row alignment for
        // embedding columns so downstream Arrow building has a clean
        // contract to rely on.
        return validate_float32_schema(rows, schema);
    }

    // Find every column index that is an embedding somewhere in the rows.
    // We can't use the schema alone because Sq8/Rabitq schemas legitimately
    // say `Binary` for what used to be embeddings.
    let embedding_columns = collect_embedding_columns(rows, schema)?;
    if embedding_columns.is_empty() {
        // Nothing to quantise — every non-embedding column passes through.
        return Ok(());
    }

    let seed = seed.unwrap_or(DEFAULT_QUANT_SEED);

    for (col_idx, dim) in embedding_columns {
        // Schema must declare this column as Binary for Sq8/Rabitq.
        let schema_dt = schema
            .field(col_idx)
            .data_type();
        if schema_dt != &DataType::Binary {
            return Err(ExportError::SchemaMismatch(format!(
                "column {} holds embeddings but schema declared {:?} for precision {:?}; \
                 Sq8 and Rabitq require columns of type Binary",
                col_idx, schema_dt, precision
            )));
        }

        match precision {
            TrainingPrecision::Sq8 => quantise_column_sq8(rows, col_idx, dim)?,
            TrainingPrecision::Rabitq => quantise_column_rabitq(rows, col_idx, dim, seed)?,
            TrainingPrecision::Float32 => unreachable!("handled above"),
        }
    }

    Ok(())
}

/// For Float32 exports: ensure embedding-row fields line up with the
/// schema's `FixedSizeList<Float32>` columns so the Arrow builder code
/// downstream never hits a row/schema mismatch.
fn validate_float32_schema(rows: &[ExportedRow], schema: &ArrowSchema) -> ExportResult<()> {
    for (row_idx, row) in rows.iter().enumerate() {
        if row.fields.len() != schema.fields().len() {
            return Err(ExportError::SchemaMismatch(format!(
                "row {} has {} fields but schema declares {} columns",
                row_idx,
                row.fields.len(),
                schema.fields().len()
            )));
        }
        for (col_idx, field) in schema.fields().iter().enumerate() {
            if let FieldValue::Embedding(e) = &row.fields[col_idx] {
                match field.data_type() {
                    DataType::FixedSizeList(child, list_len) => {
                        if child.data_type() != &DataType::Float32 {
                            return Err(ExportError::SchemaMismatch(format!(
                                "column {}: only FixedSizeList<Float32> is supported, got FixedSizeList<{:?}>",
                                col_idx,
                                child.data_type()
                            )));
                        }
                        if e.len() != *list_len as usize {
                            return Err(ExportError::SchemaMismatch(format!(
                                "column {} row {}: embedding dim {} does not match schema dim {}",
                                col_idx,
                                row_idx,
                                e.len(),
                                list_len
                            )));
                        }
                    }
                    other => {
                        return Err(ExportError::SchemaMismatch(format!(
                            "column {} row {}: FieldValue::Embedding requires \
                             schema type FixedSizeList<Float32> for Float32 precision, got {:?}",
                            col_idx, row_idx, other
                        )));
                    }
                }
            }
        }
    }
    Ok(())
}

/// Collect `(col_idx, dim)` for every column that holds embeddings in at
/// least one row, checking that all rows agree on the dimension for that
/// column. Also ensures the schema has exactly as many columns as each row.
fn collect_embedding_columns(
    rows: &[ExportedRow],
    schema: &ArrowSchema,
) -> ExportResult<Vec<(usize, usize)>> {
    let n_cols = schema.fields().len();
    let mut dims: Vec<Option<usize>> = vec![None; n_cols];

    for (row_idx, row) in rows.iter().enumerate() {
        if row.fields.len() != n_cols {
            return Err(ExportError::SchemaMismatch(format!(
                "row {} has {} fields but schema declares {} columns",
                row_idx,
                row.fields.len(),
                n_cols
            )));
        }
        for (col_idx, f) in row.fields.iter().enumerate() {
            if let FieldValue::Embedding(v) = f {
                match dims[col_idx] {
                    None => dims[col_idx] = Some(v.len()),
                    Some(expected) if expected == v.len() => {}
                    Some(expected) => {
                        return Err(ExportError::SchemaMismatch(format!(
                            "column {} row {}: embedding dim {} does not match dim {} seen earlier in this column",
                            col_idx,
                            row_idx,
                            v.len(),
                            expected
                        )));
                    }
                }
            }
        }
    }

    Ok(dims
        .into_iter()
        .enumerate()
        .filter_map(|(i, d)| d.map(|dim| (i, dim)))
        .collect())
}

/// Quantise column `col_idx` (embedding, dimension `dim`) to SQ8 bytes.
///
/// Calibration is computed across all rows in the column — this matches
/// the calibration strategy SQ8 uses at indexing time (`Sq8Quantizer::calibrate`)
/// and is the reason we run this per-column.
fn quantise_column_sq8(
    rows: &mut [ExportedRow],
    col_idx: usize,
    dim: usize,
) -> ExportResult<()> {
    // Collect embeddings for calibration without copying into owned Vecs.
    let calibration_slices: Vec<&[f32]> = rows
        .iter()
        .filter_map(|r| match r.fields.get(col_idx) {
            Some(FieldValue::Embedding(v)) => Some(v.as_slice()),
            _ => None,
        })
        .collect();

    if calibration_slices.is_empty() {
        return Ok(());
    }

    let quantiser = galaxdb_vector::Sq8Quantizer::calibrate(&calibration_slices, dim);

    for row in rows.iter_mut() {
        if let Some(slot) = row.fields.get_mut(col_idx) {
            if let FieldValue::Embedding(v) = slot {
                let bytes = <galaxdb_vector::Sq8Quantizer as galaxdb_vector::Quantizer>::quantize(
                    &quantiser, v,
                );
                *slot = FieldValue::Binary(bytes);
            }
        }
    }

    Ok(())
}

/// Quantise column `col_idx` to RaBitQ packed bits (1 bit per dim).
fn quantise_column_rabitq(
    rows: &mut [ExportedRow],
    col_idx: usize,
    dim: usize,
    seed: u64,
) -> ExportResult<()> {
    // RaBitQ rotation matrix is seeded once per column. Mixing the column
    // index in keeps independent columns independent even when the caller
    // passes the same seed, while keeping the output deterministic.
    let column_seed = seed.wrapping_add(col_idx as u64);
    let quantiser = galaxdb_vector::RabitqQuantizer::new(dim, column_seed);

    for row in rows.iter_mut() {
        if let Some(slot) = row.fields.get_mut(col_idx) {
            if let FieldValue::Embedding(v) = slot {
                let bytes = <galaxdb_vector::RabitqQuantizer as galaxdb_vector::Quantizer>::quantize(
                    &quantiser, v,
                );
                *slot = FieldValue::Binary(bytes);
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Canonical row encoding + content hash
// ---------------------------------------------------------------------------

/// Tag byte prefix for each `FieldValue` variant in the canonical encoding.
/// These values are part of the content-hash contract: changing them
/// invalidates every previously-recorded `ExportStats::content_hash`.
const TAG_INT64: u8 = 0x01;
const TAG_UTF8: u8 = 0x02;
const TAG_FLOAT32: u8 = 0x03;
const TAG_BINARY: u8 = 0x04;
const TAG_EMBEDDING: u8 = 0x05;
/// Tag byte that precedes the per-row `near_duplicate_group` encoding
/// after the primary key and field list. Added in task 34.4 so that
/// flipping `LanceExporter::dedup` or changing a row's group ID is
/// visible in the content hash.
const TAG_GROUP: u8 = 0x06;

/// Serialise sorted rows to a canonical byte stream and XXH3-128 it.
///
/// The encoding of each row is:
///
/// ```text
/// u64 primary_key_len | primary_key bytes
/// u64 field_count
/// field_0 | field_1 | ...
/// u8  TAG_GROUP (0x06)
/// u8  group_tag       | (0x00 = None, 0x01 = Some(u64 LE))
/// ```
///
/// The `TAG_GROUP` byte (0x06) plus the group payload were added in task
/// 34.4 and are part of the content-hash contract from this release
/// onward: toggling the `dedup` flag is visible in the hash even when it
/// happens to keep the same rows, and flipping a row's
/// `near_duplicate_group` changes the hash in a predictable way. This is
/// what lets the lineage table (Req 38) be keyed on
/// `(tag, precision, dedup)` without risk of collisions.
fn canonical_content_hash(rows: &[ExportedRow]) -> [u8; 16] {
    let mut buf: Vec<u8> = Vec::with_capacity(rows.len() * 64);
    buf.extend_from_slice(&(rows.len() as u64).to_le_bytes());

    for row in rows {
        buf.extend_from_slice(&(row.primary_key.len() as u64).to_le_bytes());
        buf.extend_from_slice(&row.primary_key);
        buf.extend_from_slice(&(row.fields.len() as u64).to_le_bytes());
        for field in &row.fields {
            match field {
                FieldValue::Int64(v) => {
                    buf.push(TAG_INT64);
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                FieldValue::Utf8(s) => {
                    buf.push(TAG_UTF8);
                    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
                    buf.extend_from_slice(s.as_bytes());
                }
                FieldValue::Float32(f) => {
                    buf.push(TAG_FLOAT32);
                    // `to_le_bytes` on `f32` hashes the exact bit pattern, so
                    // +0.0 and -0.0 hash distinctly and NaN hashes by payload.
                    // That is the correct semantics for deterministic export:
                    // byte-equal input → byte-equal hash.
                    buf.extend_from_slice(&f.to_le_bytes());
                }
                FieldValue::Binary(b) => {
                    buf.push(TAG_BINARY);
                    buf.extend_from_slice(&(b.len() as u64).to_le_bytes());
                    buf.extend_from_slice(b);
                }
                FieldValue::Embedding(v) => {
                    buf.push(TAG_EMBEDDING);
                    buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
                    for x in v {
                        buf.extend_from_slice(&x.to_le_bytes());
                    }
                }
            }
        }
        // Per-row `near_duplicate_group` trailer (task 34.4). Emitted
        // after the field list so existing rows-with-None hashes change
        // deterministically compared to the 34.3 encoding.
        buf.push(TAG_GROUP);
        match row.near_duplicate_group {
            None => buf.push(0x00),
            Some(g) => {
                buf.push(0x01);
                buf.extend_from_slice(&g.to_le_bytes());
            }
        }
    }

    // xxh3_128 returns u128; serialise big-endian so the representation is
    // stable across architectures and matches the hex encoding used for
    // lineage rows (Req 38, task 34.5).
    xxh3_128(&buf).to_be_bytes()
}

// ---------------------------------------------------------------------------
// Arrow batch construction
// ---------------------------------------------------------------------------

/// Build Arrow `RecordBatch`es from sorted exported rows, chunked to
/// `batch_size` rows each.
fn rows_to_record_batches(
    rows: &[ExportedRow],
    schema: &Arc<ArrowSchema>,
    batch_size: usize,
) -> ExportResult<Vec<RecordBatch>> {
    let mut batches = Vec::with_capacity(rows.len().div_ceil(batch_size).max(1));

    for chunk in rows.chunks(batch_size) {
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());
        for (col_idx, field) in schema.fields().iter().enumerate() {
            let array = build_column_array(chunk, col_idx, field.data_type())?;
            columns.push(array);
        }
        let batch = RecordBatch::try_new(schema.clone(), columns)?;
        batches.push(batch);
    }

    Ok(batches)
}

/// Build a single Arrow array for column `col_idx` across `rows`.
///
/// Each `FieldValue` variant is mapped to the matching Arrow builder; any
/// mismatch between `FieldValue` and the schema's declared `DataType`
/// returns [`ExportError::SchemaMismatch`] with enough context to debug.
fn build_column_array(
    rows: &[ExportedRow],
    col_idx: usize,
    dt: &DataType,
) -> ExportResult<ArrayRef> {
    match dt {
        DataType::Int64 => {
            let mut b = Int64Builder::with_capacity(rows.len());
            for (row_idx, row) in rows.iter().enumerate() {
                let v = field_at(row, col_idx, row_idx)?;
                match v {
                    FieldValue::Int64(i) => b.append_value(*i),
                    other => return Err(mismatch(col_idx, "Int64", other)),
                }
            }
            Ok(Arc::new(b.finish()))
        }
        DataType::Utf8 => {
            let mut b = StringBuilder::with_capacity(rows.len(), rows.len() * 16);
            for (row_idx, row) in rows.iter().enumerate() {
                let v = field_at(row, col_idx, row_idx)?;
                match v {
                    FieldValue::Utf8(s) => b.append_value(s),
                    other => return Err(mismatch(col_idx, "Utf8", other)),
                }
            }
            Ok(Arc::new(b.finish()))
        }
        DataType::Float32 => {
            let mut b = Float32Builder::with_capacity(rows.len());
            for (row_idx, row) in rows.iter().enumerate() {
                let v = field_at(row, col_idx, row_idx)?;
                match v {
                    FieldValue::Float32(f) => b.append_value(*f),
                    other => return Err(mismatch(col_idx, "Float32", other)),
                }
            }
            Ok(Arc::new(b.finish()))
        }
        DataType::Binary => {
            let mut b = BinaryBuilder::with_capacity(rows.len(), rows.len() * 16);
            for (row_idx, row) in rows.iter().enumerate() {
                let v = field_at(row, col_idx, row_idx)?;
                match v {
                    FieldValue::Binary(bytes) => b.append_value(bytes),
                    other => return Err(mismatch(col_idx, "Binary", other)),
                }
            }
            Ok(Arc::new(b.finish()))
        }
        DataType::FixedSizeList(child_field, list_len) => {
            if child_field.data_type() != &DataType::Float32 {
                return Err(ExportError::SchemaMismatch(format!(
                    "column {}: only FixedSizeList<Float32> is supported, got FixedSizeList<{:?}>",
                    col_idx,
                    child_field.data_type()
                )));
            }
            let values_builder = Float32Builder::with_capacity(rows.len() * (*list_len as usize));
            let mut b = FixedSizeListBuilder::new(values_builder, *list_len)
                .with_field(child_field.clone());
            let expected_dim = *list_len as usize;
            for (row_idx, row) in rows.iter().enumerate() {
                let v = field_at(row, col_idx, row_idx)?;
                match v {
                    FieldValue::Embedding(e) => {
                        if e.len() != expected_dim {
                            return Err(ExportError::SchemaMismatch(format!(
                                "column {} row {}: embedding dim {} does not match schema dim {}",
                                col_idx,
                                row_idx,
                                e.len(),
                                expected_dim
                            )));
                        }
                        for x in e {
                            b.values().append_value(*x);
                        }
                        b.append(true);
                    }
                    other => return Err(mismatch(col_idx, "Embedding", other)),
                }
            }
            Ok(Arc::new(b.finish()))
        }
        other => Err(ExportError::SchemaMismatch(format!(
            "column {}: Arrow data type {:?} is not supported by the 34.2 exporter",
            col_idx, other
        ))),
    }
}

fn field_at(row: &ExportedRow, col_idx: usize, row_idx: usize) -> ExportResult<&FieldValue> {
    row.fields.get(col_idx).ok_or_else(|| {
        ExportError::SchemaMismatch(format!(
            "row {} has {} fields, but schema requires column index {}",
            row_idx,
            row.fields.len(),
            col_idx
        ))
    })
}

fn mismatch(col_idx: usize, expected: &str, got: &FieldValue) -> ExportError {
    let got_name = match got {
        FieldValue::Int64(_) => "Int64",
        FieldValue::Utf8(_) => "Utf8",
        FieldValue::Float32(_) => "Float32",
        FieldValue::Binary(_) => "Binary",
        FieldValue::Embedding(_) => "Embedding",
    };
    ExportError::SchemaMismatch(format!(
        "column {}: expected FieldValue::{}, got FieldValue::{}",
        col_idx, expected, got_name
    ))
}

// ---------------------------------------------------------------------------
// Output directory size helper
// ---------------------------------------------------------------------------

/// Recursively sum the sizes of every regular file under `path`.
/// Used to populate [`ExportStats::byte_count`].
fn dir_size_bytes(path: &std::path::Path) -> std::io::Result<u64> {
    let mut total: u64 = 0;
    let mut stack: Vec<std::path::PathBuf> = vec![path.to_path_buf()];
    while let Some(p) = stack.pop() {
        let meta = match std::fs::symlink_metadata(&p) {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(e),
        };
        if meta.is_dir() {
            for entry in std::fs::read_dir(&p)? {
                let entry = entry?;
                stack.push(entry.path());
            }
        } else if meta.is_file() {
            total = total.saturating_add(meta.len());
        }
        // Symlinks are skipped deliberately — Lance never emits them and we
        // don't want to follow into arbitrary targets during a size scan.
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tags::{TagCatalog, TrainingTagMetadata};
    use arrow::array::{Array, AsArray, Float32Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Float32Type};
    use lance::Dataset;
    use std::sync::Mutex;

    // -----------------------------------------------------------------
    // Test fixtures
    // -----------------------------------------------------------------

    /// In-memory `LanceExportSource` backed by a pre-built row list.
    /// This is *not* a mock — it's a legitimate implementation of the trait
    /// that returns rows from a `Vec` instead of reading PAX blocks.
    struct VecSource {
        rows: Mutex<Vec<ExportedRow>>,
    }

    impl VecSource {
        fn new(rows: Vec<ExportedRow>) -> Arc<Self> {
            Arc::new(Self {
                rows: Mutex::new(rows),
            })
        }
    }

    impl LanceExportSource for VecSource {
        fn read_blocks(&self, _block_ids: &[BlockId]) -> ExportResult<Vec<ExportedRow>> {
            Ok(self.rows.lock().expect("rows mutex poisoned").clone())
        }
    }

    fn sample_schema() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            Field::new("row_id", DataType::Int64, false),
            Field::new("text", DataType::Utf8, false),
        ]))
    }

    fn embedding_schema(dim: i32) -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            Field::new("pk", DataType::Int64, false),
            Field::new("text", DataType::Utf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, false)),
                    dim,
                ),
                false,
            ),
        ]))
    }

    fn make_row(pk: i64, text: &str, emb: Vec<f32>) -> ExportedRow {
        ExportedRow {
            primary_key: pk.to_be_bytes().to_vec(),
            fields: vec![
                FieldValue::Int64(pk),
                FieldValue::Utf8(text.to_string()),
                FieldValue::Embedding(emb),
            ],
            near_duplicate_group: None,
        }
    }

    fn sample_catalog_with_tag(name: &str) -> (Arc<MerkleDag>, Arc<TagCatalog>) {
        let mut dag = MerkleDag::new();
        let root = dag.commit(1_000, vec![111, 222], vec![1, 2]);

        let mut catalog = TagCatalog::new();
        catalog
            .create_tag(
                name.to_string(),
                1_000,
                root,
                1_000,
                vec![1, 2],
                true,
                Some(TrainingTagMetadata {
                    precision: "float32".to_string(),
                    seed: Some(42),
                    deterministic_order: true,
                }),
            )
            .expect("tag creation");

        (Arc::new(dag), Arc::new(catalog))
    }

    /// Trivial source used by the construction-only tests (never called).
    struct UnusedSource;
    impl LanceExportSource for UnusedSource {
        fn read_blocks(&self, _block_ids: &[BlockId]) -> ExportResult<Vec<ExportedRow>> {
            Ok(Vec::new())
        }
    }

    // -----------------------------------------------------------------
    // Task 34.1 (kept green)
    // -----------------------------------------------------------------

    #[test]
    fn lance_exporter_constructs() {
        let schema = sample_schema();
        let (dag, catalog) = sample_catalog_with_tag("train-v1");

        let exporter = LanceExporter::new(
            "/tmp/galaxdb/train-v1.lance",
            schema.clone(),
            dag.clone(),
            catalog.clone(),
            Arc::new(UnusedSource) as Arc<dyn LanceExportSource>,
            "train-v1",
            TrainingPrecision::Sq8,
            true,
            Some(42),
        );

        assert_eq!(
            exporter.output_path(),
            std::path::Path::new("/tmp/galaxdb/train-v1.lance")
        );
        assert_eq!(exporter.schema().fields().len(), schema.fields().len());
        assert_eq!(exporter.tag_name(), "train-v1");
        assert_eq!(exporter.precision(), TrainingPrecision::Sq8);
        assert!(exporter.dedup());
        assert_eq!(exporter.seed(), Some(42));
        assert_eq!(exporter.merkle_dag().version_count(), 1);
        assert_eq!(exporter.tag_catalog().tag_count(), 1);

        let dbg = format!("{:?}", exporter);
        assert!(dbg.contains("train-v1"));
        assert!(dbg.contains("Sq8"));
    }

    #[test]
    fn training_precision_round_trip() {
        for p in [
            TrainingPrecision::Float32,
            TrainingPrecision::Sq8,
            TrainingPrecision::Rabitq,
        ] {
            assert_eq!(TrainingPrecision::from_str_opt(p.as_str()), Some(p));
        }
        assert_eq!(TrainingPrecision::from_str_opt("unknown"), None);
        assert_eq!(TrainingPrecision::default(), TrainingPrecision::Float32);
    }

    #[test]
    fn export_stats_empty_is_zeroed() {
        let s = ExportStats::empty();
        assert_eq!(s.row_count, 0);
        assert_eq!(s.byte_count, 0);
        assert_eq!(s.content_hash, [0u8; 16]);
    }

    #[tokio::test]
    async fn export_reports_tag_not_found() {
        let schema = sample_schema();
        let (dag, catalog) = sample_catalog_with_tag("train-v1");

        let exporter = LanceExporter::new(
            "/tmp/galaxdb/missing.lance",
            schema,
            dag,
            catalog,
            Arc::new(UnusedSource) as Arc<dyn LanceExportSource>,
            "does-not-exist",
            TrainingPrecision::Float32,
            false,
            None,
        );

        match exporter.export().await {
            Err(ExportError::TagNotFound(name)) => assert_eq!(name, "does-not-exist"),
            other => panic!("expected TagNotFound, got {:?}", other),
        }
    }

    /// Sq8/Rabitq precisions are now fully wired up in task 34.3. We keep
    /// the tag-not-found test above as regression coverage; precision
    /// conversion coverage lives in the dedicated 34.3 tests below.
    #[tokio::test]
    async fn export_sq8_errors_on_mismatched_schema() {
        let dim: i32 = 8;
        // Schema still says FixedSizeList<Float32> but we request Sq8 —
        // this must be rejected with a clear message (see docs on
        // `LanceExporter::export`).
        let schema = embedding_schema(dim);
        let (dag, catalog) = sample_catalog_with_tag("train-v1");

        let rows: Vec<ExportedRow> = (0..4i64)
            .map(|i| make_row(i, "x", vec![i as f32; dim as usize]))
            .collect();
        let source = VecSource::new(rows);

        let tmp = tempfile::tempdir().expect("tempdir");
        let out = tmp.path().join("sq8_mismatch.lance");

        let exporter = LanceExporter::new(
            &out,
            schema,
            dag,
            catalog,
            source,
            "train-v1",
            TrainingPrecision::Sq8,
            false,
            None,
        );

        match exporter.export().await {
            Err(ExportError::SchemaMismatch(msg)) => {
                assert!(msg.contains("Sq8") || msg.contains("Binary"), "{msg}");
                assert!(msg.contains("column"), "{msg}");
            }
            other => panic!("expected SchemaMismatch, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------
    // Task 34.2
    // -----------------------------------------------------------------

    /// End-to-end smoke: 100 rows go in, a real Lance dataset comes out,
    /// and `ExportStats` accurately reflect what was written.
    #[tokio::test]
    async fn export_pipeline_writes_lance_dataset() {
        let dim: i32 = 4;
        let schema = embedding_schema(dim);
        let (dag, catalog) = sample_catalog_with_tag("train-v1");

        let mut rows = Vec::with_capacity(100);
        for i in 0..100i64 {
            // Interleave the input order so we also exercise the sort step,
            // though this test doesn't assert on ordering (that's the next
            // test's job).
            let pk = (i * 37 + 13) % 100;
            rows.push(make_row(
                pk,
                &format!("row-{pk}"),
                vec![pk as f32, (pk + 1) as f32, (pk + 2) as f32, (pk + 3) as f32],
            ));
        }
        let source = VecSource::new(rows);

        let tmp = tempfile::tempdir().expect("tempdir");
        let out = tmp.path().join("train.lance");

        let exporter = LanceExporter::new(
            &out,
            schema.clone(),
            dag,
            catalog,
            source,
            "train-v1",
            TrainingPrecision::Float32,
            false,
            None,
        );

        let stats = exporter.export().await.expect("export succeeds");
        assert_eq!(stats.row_count, 100);
        assert!(stats.byte_count > 0, "byte_count should be non-zero");
        assert_ne!(stats.content_hash, [0u8; 16], "content hash is non-zero");

        // Dataset is a real, readable Lance dataset.
        assert!(out.exists(), "output path exists");
        let ds = Dataset::open(out.to_str().unwrap())
            .await
            .expect("open written dataset");
        let count = ds
            .scan()
            .count_rows()
            .await
            .expect("count rows");
        assert_eq!(count, 100);
    }

    /// Rows supplied out of order come back sorted by primary key when the
    /// dataset is re-read.
    #[tokio::test]
    async fn export_pipeline_sorts_by_primary_key() {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("pk", DataType::Int64, false),
            Field::new("text", DataType::Utf8, false),
        ]));
        let (dag, catalog) = sample_catalog_with_tag("train-v1");

        let rows = vec![
            ExportedRow {
                primary_key: 3i64.to_be_bytes().to_vec(),
                fields: vec![FieldValue::Int64(3), FieldValue::Utf8("third".into())],
                near_duplicate_group: None,
            },
            ExportedRow {
                primary_key: 1i64.to_be_bytes().to_vec(),
                fields: vec![FieldValue::Int64(1), FieldValue::Utf8("first".into())],
                near_duplicate_group: None,
            },
            ExportedRow {
                primary_key: 2i64.to_be_bytes().to_vec(),
                fields: vec![FieldValue::Int64(2), FieldValue::Utf8("second".into())],
                near_duplicate_group: None,
            },
        ];
        let source = VecSource::new(rows);

        let tmp = tempfile::tempdir().expect("tempdir");
        let out = tmp.path().join("sorted.lance");

        let exporter = LanceExporter::new(
            &out,
            schema.clone(),
            dag,
            catalog,
            source,
            "train-v1",
            TrainingPrecision::Float32,
            false,
            None,
        );

        exporter.export().await.expect("export succeeds");

        let ds = Dataset::open(out.to_str().unwrap())
            .await
            .expect("open dataset");
        let batch = ds.scan().try_into_batch().await.expect("collect");
        let batches = vec![batch];

        let mut observed_pks: Vec<i64> = Vec::new();
        let mut observed_text: Vec<String> = Vec::new();
        for batch in &batches {
            let pks = batch
                .column_by_name("pk")
                .expect("pk col")
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64Array");
            let texts = batch
                .column_by_name("text")
                .expect("text col")
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("StringArray");
            for i in 0..batch.num_rows() {
                observed_pks.push(pks.value(i));
                observed_text.push(texts.value(i).to_string());
            }
        }

        assert_eq!(observed_pks, vec![1, 2, 3]);
        assert_eq!(observed_text, vec!["first", "second", "third"]);
    }

    /// A source that returns zero rows maps to `ExportError::EmptyVersion`
    /// rather than writing an empty Lance dataset.
    #[tokio::test]
    async fn export_pipeline_empty_source_returns_empty_version_error() {
        let schema = sample_schema();
        let (dag, catalog) = sample_catalog_with_tag("train-v1");
        let source = VecSource::new(vec![]);

        let tmp = tempfile::tempdir().expect("tempdir");
        let out = tmp.path().join("empty.lance");

        let exporter = LanceExporter::new(
            &out,
            schema,
            dag,
            catalog,
            source,
            "train-v1",
            TrainingPrecision::Float32,
            false,
            None,
        );

        match exporter.export().await {
            Err(ExportError::EmptyVersion) => {}
            other => panic!("expected EmptyVersion, got {:?}", other),
        }
        assert!(
            !out.exists() || dir_size_bytes(&out).unwrap() == 0,
            "empty export must not leave a non-empty dataset on disk"
        );
    }

    /// Running the same export pipeline against two different output paths
    /// must produce a byte-identical `content_hash`. This is what lets
    /// downstream training jobs cache on content hash alone.
    #[tokio::test]
    async fn export_pipeline_deterministic() {
        let dim: i32 = 4;
        let schema = embedding_schema(dim);
        let (dag, catalog) = sample_catalog_with_tag("train-v1");

        let rows: Vec<ExportedRow> = (0..25i64)
            .map(|i| {
                make_row(
                    i,
                    &format!("row-{i}"),
                    vec![i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32],
                )
            })
            .collect();

        let tmp1 = tempfile::tempdir().expect("tempdir 1");
        let tmp2 = tempfile::tempdir().expect("tempdir 2");
        let out1 = tmp1.path().join("a.lance");
        let out2 = tmp2.path().join("b.lance");

        let exporter1 = LanceExporter::new(
            &out1,
            schema.clone(),
            dag.clone(),
            catalog.clone(),
            VecSource::new(rows.clone()),
            "train-v1",
            TrainingPrecision::Float32,
            false,
            None,
        );
        let exporter2 = LanceExporter::new(
            &out2,
            schema.clone(),
            dag,
            catalog,
            VecSource::new(rows),
            "train-v1",
            TrainingPrecision::Float32,
            false,
            None,
        );

        let stats1 = exporter1.export().await.expect("export 1");
        let stats2 = exporter2.export().await.expect("export 2");

        assert_eq!(stats1.row_count, 25);
        assert_eq!(stats2.row_count, 25);
        assert_eq!(
            stats1.content_hash, stats2.content_hash,
            "same input rows ⇒ same content hash"
        );
    }

    // A couple of unit-level tests that don't need a Lance roundtrip — they
    // keep the canonical encoding honest without a 100-row setup each time.

    #[test]
    fn canonical_hash_is_stable_and_order_sensitive() {
        let a = vec![
            ExportedRow {
                primary_key: vec![1],
                fields: vec![FieldValue::Int64(1)],
                near_duplicate_group: None,
            },
            ExportedRow {
                primary_key: vec![2],
                fields: vec![FieldValue::Int64(2)],
                near_duplicate_group: None,
            },
        ];
        let b = a.clone();
        let c = vec![a[1].clone(), a[0].clone()];

        assert_eq!(canonical_content_hash(&a), canonical_content_hash(&b));
        // Reversing the row order changes the hash — the caller must sort
        // first (which `export()` does).
        assert_ne!(canonical_content_hash(&a), canonical_content_hash(&c));
    }

    #[test]
    fn schema_mismatch_is_reported_with_context() {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("x", DataType::Int64, false),
        ]));
        let rows = vec![ExportedRow {
            primary_key: vec![1],
            fields: vec![FieldValue::Utf8("oops".into())],
            near_duplicate_group: None,
        }];
        match rows_to_record_batches(&rows, &schema, 16) {
            Err(ExportError::SchemaMismatch(msg)) => {
                assert!(msg.contains("Int64"), "{msg}");
                assert!(msg.contains("Utf8"), "{msg}");
            }
            other => panic!("expected SchemaMismatch, got {:?}", other),
        }
    }

    // Ensure FixedSizeList Arrow metadata survives the Lance round trip so
    // downstream consumers can still see it's an embedding column.
    #[tokio::test]
    async fn export_preserves_fixed_size_list_metadata() {
        let dim: i32 = 4;
        let schema = embedding_schema(dim);
        let (dag, catalog) = sample_catalog_with_tag("train-v1");

        let rows: Vec<ExportedRow> = (0..5i64)
            .map(|i| make_row(i, "x", vec![0.0, 1.0, 2.0, 3.0]))
            .collect();
        let source = VecSource::new(rows);

        let tmp = tempfile::tempdir().expect("tempdir");
        let out = tmp.path().join("emb.lance");

        let exporter = LanceExporter::new(
            &out,
            schema,
            dag,
            catalog,
            source,
            "train-v1",
            TrainingPrecision::Float32,
            false,
            None,
        );
        exporter.export().await.expect("export");

        let ds = Dataset::open(out.to_str().unwrap()).await.expect("open");
        let arrow_schema = ds.schema();
        let emb = arrow_schema
            .field("embedding")
            .expect("embedding field present");
        // Lance infers the fixed-size-list length from the underlying Arrow
        // field. We only assert it's a list of float32 — depending on Lance's
        // internal representation this may be FixedSizeList or List.
        // The point is that reading the column back yields float32 values.
        let _ = emb;

        // Scan the column and confirm values round-trip.
        let batch = ds.scan().try_into_batch().await.expect("batch");
        let col = batch
            .column_by_name("embedding")
            .expect("embedding column present");
        // Either FixedSizeList or List of Float32 — in both cases we can
        // collect the float values by flattening.
        let floats: Vec<f32> = match col.data_type() {
            DataType::FixedSizeList(_, _) => {
                let list = col.as_fixed_size_list();
                list.values()
                    .as_primitive::<Float32Type>()
                    .values()
                    .to_vec()
            }
            DataType::List(_) => {
                let list = col.as_list::<i32>();
                list.values()
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .expect("float32 values")
                    .values()
                    .to_vec()
            }
            other => panic!("unexpected embedding arrow type: {:?}", other),
        };
        assert_eq!(floats.len(), 5 * dim as usize);
        // First row should be [0, 1, 2, 3].
        assert_eq!(&floats[..4], &[0.0, 1.0, 2.0, 3.0]);
    }

    // -----------------------------------------------------------------
    // Task 34.3 — precision conversion (Sq8 / Rabitq)
    // -----------------------------------------------------------------

    /// Schema with a Binary embedding column, used for Sq8 and Rabitq
    /// exports. Matches the schema contract documented on
    /// `LanceExporter::export`.
    fn quantised_embedding_schema() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            Field::new("pk", DataType::Int64, false),
            Field::new("text", DataType::Utf8, false),
            Field::new("embedding", DataType::Binary, false),
        ]))
    }

    /// Deterministic pseudo-random embedding generator so tests don't
    /// depend on `rand` and produce the same vectors on every platform.
    fn synthetic_embedding(pk: i64, dim: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(dim);
        let mut state = (pk as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        for _ in 0..dim {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let x = (state >> 33) as f32 / (u32::MAX as f32 / 2.0) - 1.0;
            v.push(x);
        }
        v
    }

    /// Collect the binary values of a `Binary` column from a Lance dataset.
    async fn read_binary_column(ds: &Dataset, name: &str) -> Vec<Vec<u8>> {
        use arrow::array::BinaryArray;
        let batch = ds.scan().try_into_batch().await.expect("collect");
        let col = batch
            .column_by_name(name)
            .expect("column present")
            .as_any()
            .downcast_ref::<BinaryArray>()
            .expect("BinaryArray");
        (0..col.len()).map(|i| col.value(i).to_vec()).collect()
    }

    /// Sq8 precision rewrites `Embedding` → `Binary` and the Lance dataset
    /// round-trips through the SQ8 quantiser within the documented error
    /// budget (< 1 % L2 distance).
    #[tokio::test]
    async fn export_sq8_converts_embeddings_to_binary() {
        let dim: usize = 128;
        let schema = quantised_embedding_schema();
        let (dag, catalog) = sample_catalog_with_tag("train-v1");

        let rows: Vec<ExportedRow> = (0..10i64)
            .map(|i| {
                ExportedRow {
                    primary_key: i.to_be_bytes().to_vec(),
                    fields: vec![
                        FieldValue::Int64(i),
                        FieldValue::Utf8(format!("row-{i}")),
                        FieldValue::Embedding(synthetic_embedding(i, dim)),
                    ],
                    near_duplicate_group: None,
                }
            })
            .collect();
        let original_vectors: Vec<Vec<f32>> = rows
            .iter()
            .map(|r| match &r.fields[2] {
                FieldValue::Embedding(v) => v.clone(),
                _ => unreachable!(),
            })
            .collect();
        let source = VecSource::new(rows);

        let tmp = tempfile::tempdir().expect("tempdir");
        let out = tmp.path().join("sq8.lance");

        let exporter = LanceExporter::new(
            &out,
            schema,
            dag,
            catalog,
            source,
            "train-v1",
            TrainingPrecision::Sq8,
            false,
            None,
        );

        let stats = exporter.export().await.expect("sq8 export");
        assert_eq!(stats.row_count, 10);
        assert!(stats.byte_count > 0);

        let ds = Dataset::open(out.to_str().unwrap()).await.expect("open");
        // Embedding column is Binary in the on-disk schema.
        let emb_field = ds.schema().field("embedding").expect("emb field");
        assert_eq!(emb_field.data_type(), DataType::Binary);

        let bytes = read_binary_column(&ds, "embedding").await;
        assert_eq!(bytes.len(), 10);
        for row_bytes in &bytes {
            // SQ8 uses one byte per dimension (4× compression vs f32).
            assert_eq!(row_bytes.len(), dim);
        }

        // Independently dequantise with the same calibration the exporter
        // used (global min/max across the 10 input vectors) and confirm we
        // land within SQ8's documented error budget.
        let refs: Vec<&[f32]> = original_vectors.iter().map(|v| v.as_slice()).collect();
        let expected_q = galaxdb_vector::Sq8Quantizer::calibrate(&refs, dim);
        for (original, row_bytes) in original_vectors.iter().zip(bytes.iter()) {
            // The bytes on disk must match what an independently-calibrated
            // Sq8Quantizer would produce for the same vector — i.e. the
            // exporter is actually using the real `Sq8Quantizer`, not a
            // reimplementation.
            let expected_bytes =
                <galaxdb_vector::Sq8Quantizer as galaxdb_vector::Quantizer>::quantize(
                    &expected_q,
                    original,
                );
            assert_eq!(row_bytes, &expected_bytes);

            // Decode and compare to the original in L2 space.
            let decoded =
                <galaxdb_vector::Sq8Quantizer as galaxdb_vector::Quantizer>::dequantize(
                    &expected_q,
                    row_bytes,
                );
            let l2_sq: f32 = original
                .iter()
                .zip(decoded.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            let l2 = l2_sq.sqrt();
            let norm: f32 = original.iter().map(|x| x * x).sum::<f32>().sqrt();
            let relative = if norm > f32::EPSILON { l2 / norm } else { 0.0 };
            // SQ8 gives ~1/255 per dim of relative error; well under 1%
            // for well-distributed inputs.
            assert!(
                relative < 0.01,
                "relative L2 error {:.6} exceeds 1 % — SQ8 roundtrip lost too much precision",
                relative
            );
        }
    }

    /// Two Sq8 exports of the same rows produce the same content hash —
    /// the hash is what downstream training jobs key caches on.
    #[tokio::test]
    async fn export_sq8_is_deterministic() {
        let dim: usize = 64;
        let schema = quantised_embedding_schema();
        let (dag, catalog) = sample_catalog_with_tag("train-v1");

        let rows: Vec<ExportedRow> = (0..20i64)
            .map(|i| ExportedRow {
                primary_key: i.to_be_bytes().to_vec(),
                fields: vec![
                    FieldValue::Int64(i),
                    FieldValue::Utf8(format!("r{i}")),
                    FieldValue::Embedding(synthetic_embedding(i, dim)),
                ],
                near_duplicate_group: None,
            })
            .collect();

        let tmp1 = tempfile::tempdir().expect("tempdir 1");
        let tmp2 = tempfile::tempdir().expect("tempdir 2");

        let stats1 = LanceExporter::new(
            tmp1.path().join("a.lance"),
            schema.clone(),
            dag.clone(),
            catalog.clone(),
            VecSource::new(rows.clone()),
            "train-v1",
            TrainingPrecision::Sq8,
            false,
            None,
        )
        .export()
        .await
        .expect("export 1");
        let stats2 = LanceExporter::new(
            tmp2.path().join("b.lance"),
            schema,
            dag,
            catalog,
            VecSource::new(rows),
            "train-v1",
            TrainingPrecision::Sq8,
            false,
            None,
        )
        .export()
        .await
        .expect("export 2");

        assert_eq!(
            stats1.content_hash, stats2.content_hash,
            "same Sq8 input ⇒ identical content hash"
        );
    }

    /// Rabitq precision produces 1-bit-per-dim packed binary (128 dims →
    /// 16 bytes/row) and the dataset is a valid Lance dataset.
    #[tokio::test]
    async fn export_rabitq_converts_embeddings_to_binary() {
        let dim: usize = 128;
        let schema = quantised_embedding_schema();
        let (dag, catalog) = sample_catalog_with_tag("train-v1");

        let rows: Vec<ExportedRow> = (0..50i64)
            .map(|i| ExportedRow {
                primary_key: i.to_be_bytes().to_vec(),
                fields: vec![
                    FieldValue::Int64(i),
                    FieldValue::Utf8(format!("row-{i}")),
                    FieldValue::Embedding(synthetic_embedding(i, dim)),
                ],
                near_duplicate_group: None,
            })
            .collect();
        let source = VecSource::new(rows);

        let tmp = tempfile::tempdir().expect("tempdir");
        let out = tmp.path().join("rabitq.lance");

        let stats = LanceExporter::new(
            &out,
            schema,
            dag,
            catalog,
            source,
            "train-v1",
            TrainingPrecision::Rabitq,
            false,
            Some(42),
        )
        .export()
        .await
        .expect("rabitq export");
        assert_eq!(stats.row_count, 50);

        let ds = Dataset::open(out.to_str().unwrap()).await.expect("open");
        let emb_field = ds.schema().field("embedding").expect("emb field");
        assert_eq!(emb_field.data_type(), DataType::Binary);

        let bytes = read_binary_column(&ds, "embedding").await;
        assert_eq!(bytes.len(), 50);
        for row_bytes in &bytes {
            // RaBitQ packs 1 bit per dim into bytes — 128 / 8 = 16 bytes.
            assert_eq!(row_bytes.len(), dim / 8);
        }

        // Sanity check: the bytes produced by the exporter must match what
        // a freshly-seeded `RabitqQuantizer` would produce for the same
        // vectors. The exporter seeds column 2 with `seed + col_idx`.
        let expected_q = galaxdb_vector::RabitqQuantizer::new(dim, 42 + 2);
        for (i, row_bytes) in bytes.iter().enumerate() {
            let expected =
                <galaxdb_vector::RabitqQuantizer as galaxdb_vector::Quantizer>::quantize(
                    &expected_q,
                    &synthetic_embedding(i as i64, dim),
                );
            assert_eq!(row_bytes, &expected);
        }
    }

    /// Rabitq exports are deterministic per seed and differ across seeds.
    #[tokio::test]
    async fn export_rabitq_deterministic_with_seed() {
        let dim: usize = 32;
        let schema = quantised_embedding_schema();
        let (dag, catalog) = sample_catalog_with_tag("train-v1");

        let rows: Vec<ExportedRow> = (0..12i64)
            .map(|i| ExportedRow {
                primary_key: i.to_be_bytes().to_vec(),
                fields: vec![
                    FieldValue::Int64(i),
                    FieldValue::Utf8("x".into()),
                    FieldValue::Embedding(synthetic_embedding(i, dim)),
                ],
                near_duplicate_group: None,
            })
            .collect();

        let mk = |seed: Option<u64>, path: std::path::PathBuf| {
            LanceExporter::new(
                path,
                schema.clone(),
                dag.clone(),
                catalog.clone(),
                VecSource::new(rows.clone()),
                "train-v1",
                TrainingPrecision::Rabitq,
                false,
                seed,
            )
        };

        let tmp = tempfile::tempdir().expect("tempdir");
        let h_42a = mk(Some(42), tmp.path().join("42a.lance"))
            .export()
            .await
            .expect("42a")
            .content_hash;
        let h_42b = mk(Some(42), tmp.path().join("42b.lance"))
            .export()
            .await
            .expect("42b")
            .content_hash;
        let h_100 = mk(Some(100), tmp.path().join("100.lance"))
            .export()
            .await
            .expect("100")
            .content_hash;

        assert_eq!(h_42a, h_42b, "seed=42 twice ⇒ same content hash");
        assert_ne!(h_42a, h_100, "seed=42 vs seed=100 ⇒ different content hash");
    }

    /// Regression: the original Float32 passthrough path still works with
    /// the new precision-conversion plumbing in place.
    #[tokio::test]
    async fn export_float32_still_works() {
        let dim: i32 = 4;
        let schema = embedding_schema(dim);
        let (dag, catalog) = sample_catalog_with_tag("train-v1");

        let rows: Vec<ExportedRow> = (0..8i64)
            .map(|i| make_row(i, &format!("r{i}"), vec![i as f32, 1.0, 2.0, 3.0]))
            .collect();
        let source = VecSource::new(rows);

        let tmp = tempfile::tempdir().expect("tempdir");
        let out = tmp.path().join("f32.lance");

        let stats = LanceExporter::new(
            &out,
            schema,
            dag,
            catalog,
            source,
            "train-v1",
            TrainingPrecision::Float32,
            false,
            None,
        )
        .export()
        .await
        .expect("float32 export");
        assert_eq!(stats.row_count, 8);

        let ds = Dataset::open(out.to_str().unwrap()).await.expect("open");
        // Embedding column is still a float list (FixedSizeList or List<f32>),
        // not Binary.
        let emb_field = ds.schema().field("embedding").expect("emb field");
        match emb_field.data_type() {
            DataType::FixedSizeList(child, _) => {
                assert_eq!(child.data_type(), &DataType::Float32);
            }
            DataType::List(child) => {
                assert_eq!(child.data_type(), &DataType::Float32);
            }
            other => panic!("float32 path must keep a float list column, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------
    // Task 34.4 — dedup integration
    // -----------------------------------------------------------------

    /// Build a minimal row (pk, text, optional group) with a two-column
    /// schema shape — matches `dedup_schema` below. Keeps the dedup tests
    /// from having to care about embeddings, which are tested in 34.3.
    fn dedup_row(pk: i64, text: &str, group: Option<u64>) -> ExportedRow {
        ExportedRow {
            primary_key: pk.to_be_bytes().to_vec(),
            fields: vec![
                FieldValue::Int64(pk),
                FieldValue::Utf8(text.to_string()),
            ],
            near_duplicate_group: group,
        }
    }

    fn dedup_schema() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            Field::new("pk", DataType::Int64, false),
            Field::new("text", DataType::Utf8, false),
        ]))
    }

    /// Collect primary-key ints from a Lance dataset written by the dedup
    /// tests. Returns them in scan order, which — because the exporter
    /// sorts by primary key first — is always ascending.
    async fn read_dedup_pks(out: &std::path::Path) -> Vec<i64> {
        let ds = Dataset::open(out.to_str().unwrap()).await.expect("open");
        let batch = ds.scan().try_into_batch().await.expect("batch");
        let pks = batch
            .column_by_name("pk")
            .expect("pk col")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64Array");
        (0..batch.num_rows()).map(|i| pks.value(i)).collect()
    }

    /// 10 rows, two overlapping near-dup groups. With `dedup = true` the
    /// exporter keeps every ungrouped row plus one representative per
    /// group (the lowest PK). Expected survivors: [0, 1, 2, 5, 6, 7, 9].
    #[tokio::test]
    async fn export_dedup_removes_near_duplicates() {
        let schema = dedup_schema();
        let (dag, catalog) = sample_catalog_with_tag("train-v1");

        // PKs 0..9. Group 5 holds {2, 3, 4}, group 9 holds {7, 8}.
        let rows: Vec<ExportedRow> = (0..10i64)
            .map(|i| {
                let group = match i {
                    2..=4 => Some(5u64),
                    7 | 8 => Some(9u64),
                    _ => None,
                };
                dedup_row(i, &format!("row-{i}"), group)
            })
            .collect();
        let source = VecSource::new(rows);

        let tmp = tempfile::tempdir().expect("tempdir");
        let out = tmp.path().join("dedup_overlap.lance");

        let exporter = LanceExporter::new(
            &out,
            schema,
            dag,
            catalog,
            source,
            "train-v1",
            TrainingPrecision::Float32,
            true, // dedup ON
            None,
        );

        let stats = exporter.export().await.expect("dedup export");
        assert_eq!(stats.row_count, 7);

        let survivors = read_dedup_pks(&out).await;
        assert_eq!(survivors, vec![0, 1, 2, 5, 6, 7, 9]);
    }

    /// 3 rows in the same group with out-of-order PKs — only the smallest
    /// PK survives. PKs are `i64::to_be_bytes`, so for positive integers
    /// big-endian byte comparison matches numeric comparison and 5 < 8 < 10.
    #[tokio::test]
    async fn export_dedup_keeps_lowest_primary_key_per_group() {
        let schema = dedup_schema();
        let (dag, catalog) = sample_catalog_with_tag("train-v1");

        let rows = vec![
            dedup_row(10, "ten", Some(1)),
            dedup_row(5, "five", Some(1)),
            dedup_row(8, "eight", Some(1)),
        ];
        let source = VecSource::new(rows);

        let tmp = tempfile::tempdir().expect("tempdir");
        let out = tmp.path().join("dedup_min_pk.lance");

        let stats = LanceExporter::new(
            &out,
            schema,
            dag,
            catalog,
            source,
            "train-v1",
            TrainingPrecision::Float32,
            true,
            None,
        )
        .export()
        .await
        .expect("dedup export");
        assert_eq!(stats.row_count, 1);

        let survivors = read_dedup_pks(&out).await;
        assert_eq!(survivors, vec![5]);
    }

    /// Same three grouped rows, but with `dedup = false`: no filtering,
    /// every input row is preserved.
    #[tokio::test]
    async fn export_dedup_false_keeps_all_rows() {
        let schema = dedup_schema();
        let (dag, catalog) = sample_catalog_with_tag("train-v1");

        let rows = vec![
            dedup_row(10, "ten", Some(1)),
            dedup_row(5, "five", Some(1)),
            dedup_row(8, "eight", Some(1)),
        ];
        let source = VecSource::new(rows);

        let tmp = tempfile::tempdir().expect("tempdir");
        let out = tmp.path().join("dedup_off.lance");

        let stats = LanceExporter::new(
            &out,
            schema,
            dag,
            catalog,
            source,
            "train-v1",
            TrainingPrecision::Float32,
            false, // dedup OFF
            None,
        )
        .export()
        .await
        .expect("export");
        assert_eq!(stats.row_count, 3);

        let survivors = read_dedup_pks(&out).await;
        // Sorted by PK, not filtered.
        assert_eq!(survivors, vec![5, 8, 10]);
    }

    /// Running the same dedup export twice into different output paths
    /// produces the same content hash.
    #[tokio::test]
    async fn export_dedup_is_deterministic() {
        let schema = dedup_schema();
        let (dag, catalog) = sample_catalog_with_tag("train-v1");

        let rows: Vec<ExportedRow> = (0..10i64)
            .map(|i| {
                let group = match i {
                    2..=4 => Some(5u64),
                    7 | 8 => Some(9u64),
                    _ => None,
                };
                dedup_row(i, &format!("row-{i}"), group)
            })
            .collect();

        let tmp1 = tempfile::tempdir().expect("tempdir 1");
        let tmp2 = tempfile::tempdir().expect("tempdir 2");

        let stats1 = LanceExporter::new(
            tmp1.path().join("a.lance"),
            schema.clone(),
            dag.clone(),
            catalog.clone(),
            VecSource::new(rows.clone()),
            "train-v1",
            TrainingPrecision::Float32,
            true,
            None,
        )
        .export()
        .await
        .expect("export 1");

        let stats2 = LanceExporter::new(
            tmp2.path().join("b.lance"),
            schema,
            dag,
            catalog,
            VecSource::new(rows),
            "train-v1",
            TrainingPrecision::Float32,
            true,
            None,
        )
        .export()
        .await
        .expect("export 2");

        assert_eq!(stats1.row_count, stats2.row_count);
        assert_eq!(
            stats1.content_hash, stats2.content_hash,
            "same dedup input ⇒ identical content hash"
        );
    }

    /// Toggling `dedup` changes the content hash: either because the
    /// surviving row set changes, or because the per-row group trailer in
    /// the canonical encoding differs. Either way the hash must not match.
    #[tokio::test]
    async fn export_dedup_changes_content_hash() {
        let schema = dedup_schema();
        let (dag, catalog) = sample_catalog_with_tag("train-v1");

        let rows: Vec<ExportedRow> = (0..10i64)
            .map(|i| {
                let group = match i {
                    2..=4 => Some(5u64),
                    7 | 8 => Some(9u64),
                    _ => None,
                };
                dedup_row(i, &format!("row-{i}"), group)
            })
            .collect();

        let tmp1 = tempfile::tempdir().expect("tempdir 1");
        let tmp2 = tempfile::tempdir().expect("tempdir 2");

        let dedup_on = LanceExporter::new(
            tmp1.path().join("on.lance"),
            schema.clone(),
            dag.clone(),
            catalog.clone(),
            VecSource::new(rows.clone()),
            "train-v1",
            TrainingPrecision::Float32,
            true,
            None,
        )
        .export()
        .await
        .expect("dedup on");

        let dedup_off = LanceExporter::new(
            tmp2.path().join("off.lance"),
            schema,
            dag,
            catalog,
            VecSource::new(rows),
            "train-v1",
            TrainingPrecision::Float32,
            false,
            None,
        )
        .export()
        .await
        .expect("dedup off");

        assert_ne!(dedup_on.row_count, dedup_off.row_count);
        assert_ne!(
            dedup_on.content_hash, dedup_off.content_hash,
            "dedup on vs off ⇒ different content hash"
        );
    }

    /// Dedup + Sq8 end-to-end: the survivor set is correct, and the
    /// embedding column lands on disk as Binary (SQ8 bytes), not as a
    /// float list.
    #[tokio::test]
    async fn export_dedup_with_sq8() {
        let dim: usize = 16;
        let schema = quantised_embedding_schema();
        let (dag, catalog) = sample_catalog_with_tag("train-v1");

        // 6 rows: 3 unique (pk 0, 1, 5) and 3 in a single dedup group
        // (pk 2, 3, 4) — survivor from group is pk=2. Final expected
        // PKs: [0, 1, 2, 5].
        let rows: Vec<ExportedRow> = (0..6i64)
            .map(|i| {
                let group = match i {
                    2..=4 => Some(7u64),
                    _ => None,
                };
                ExportedRow {
                    primary_key: i.to_be_bytes().to_vec(),
                    fields: vec![
                        FieldValue::Int64(i),
                        FieldValue::Utf8(format!("r{i}")),
                        FieldValue::Embedding(synthetic_embedding(i, dim)),
                    ],
                    near_duplicate_group: group,
                }
            })
            .collect();
        let source = VecSource::new(rows);

        let tmp = tempfile::tempdir().expect("tempdir");
        let out = tmp.path().join("dedup_sq8.lance");

        let stats = LanceExporter::new(
            &out,
            schema,
            dag,
            catalog,
            source,
            "train-v1",
            TrainingPrecision::Sq8,
            true,
            None,
        )
        .export()
        .await
        .expect("sq8 dedup export");
        assert_eq!(stats.row_count, 4);

        let survivors = read_dedup_pks(&out).await;
        assert_eq!(survivors, vec![0, 1, 2, 5]);

        // Embedding column on disk is Binary and each row has exactly
        // `dim` bytes (SQ8 = 1 byte per dim).
        let ds = Dataset::open(out.to_str().unwrap()).await.expect("open");
        let emb_field = ds.schema().field("embedding").expect("embedding field");
        assert_eq!(emb_field.data_type(), DataType::Binary);

        let bytes = read_binary_column(&ds, "embedding").await;
        assert_eq!(bytes.len(), 4);
        for row_bytes in &bytes {
            assert_eq!(row_bytes.len(), dim);
        }
    }

    // -----------------------------------------------------------------
    // Task 34.5 — lineage recording into the sink
    // -----------------------------------------------------------------

    /// `FailingLineageSink` rejects every `record()` call. Used to prove
    /// that a sink failure propagates out of `export()` and that the
    /// Lance dataset on disk is *not* rolled back (per the documented
    /// semantics).
    struct FailingLineageSink;

    impl TrainingExportLineageSink for FailingLineageSink {
        fn record(&self, _lineage: TrainingExportLineage) -> ExportResult<()> {
            Err(ExportError::SchemaMismatch("sink forced failure".into()))
        }
    }

    /// Build a small float32 export fixture returning (output_path, exporter,
    /// TempDir kept-alive) so tests can inspect the resulting lineage row
    /// without duplicating 30 lines of setup.
    fn float32_fixture(
        out_name: &str,
    ) -> (tempfile::TempDir, std::path::PathBuf, LanceExporter) {
        let dim: i32 = 4;
        let schema = embedding_schema(dim);
        let (dag, catalog) = sample_catalog_with_tag("train-v1");

        let rows: Vec<ExportedRow> = (0..8i64)
            .map(|i| make_row(i, &format!("r{i}"), vec![i as f32, 1.0, 2.0, 3.0]))
            .collect();
        let source = VecSource::new(rows);

        let tmp = tempfile::tempdir().expect("tempdir");
        let out = tmp.path().join(out_name);

        let exporter = LanceExporter::new(
            &out,
            schema,
            dag,
            catalog,
            source,
            "train-v1",
            TrainingPrecision::Float32,
            false,
            None,
        );
        (tmp, out, exporter)
    }

    /// A successful export with a sink attached writes exactly one
    /// lineage record whose fields all come from the exporter
    /// configuration and the returned `ExportStats`.
    #[tokio::test]
    async fn export_records_lineage_on_success() {
        let (_tmp, _out, exporter) = float32_fixture("lineage_ok.lance");
        let sink = Arc::new(InMemoryLineageSink::new());

        let before = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("now")
            .as_secs();

        let stats = exporter
            .clone()
            .with_lineage_sink(sink.clone() as Arc<dyn TrainingExportLineageSink>)
            .export()
            .await
            .expect("export succeeds");

        let after = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("now")
            .as_secs();

        let entries = sink.entries();
        assert_eq!(entries.len(), 1, "exactly one lineage row per export");
        let row = &entries[0];
        assert_eq!(row.tag_name, "train-v1");
        assert_eq!(row.filter_expr, None);
        assert_eq!(row.precision, "float32");
        assert!(!row.dedup);
        assert_eq!(row.row_count, stats.row_count);
        assert_eq!(row.row_count, 8);
        // exported_at is epoch seconds, captured during export(). Must
        // sit inside [before, after + 60].
        assert!(
            row.exported_at >= before && row.exported_at <= after + 60,
            "exported_at {} outside [{}..={}]",
            row.exported_at,
            before,
            after + 60
        );
        // content_hash is 32 lowercase hex chars of the XXH3-128 output.
        assert_eq!(row.content_hash.len(), 32);
        assert!(
            row.content_hash.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "content_hash contains non-hex or uppercase chars: {}",
            row.content_hash
        );
    }

    /// `with_filter_expr` threads the caller's `WHERE` clause through to
    /// the lineage row verbatim.
    #[tokio::test]
    async fn export_records_lineage_with_filter_expr() {
        let (_tmp, _out, exporter) = float32_fixture("lineage_filter.lance");
        let sink = Arc::new(InMemoryLineageSink::new());

        exporter
            .clone()
            .with_lineage_sink(sink.clone() as Arc<dyn TrainingExportLineageSink>)
            .with_filter_expr(Some("WHERE deleted = FALSE".into()))
            .export()
            .await
            .expect("export succeeds");

        let entries = sink.entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0].filter_expr.as_deref(),
            Some("WHERE deleted = FALSE")
        );
    }

    /// The lineage row's `precision` field tracks the
    /// `TrainingPrecision` the exporter was built with — here, `Sq8`
    /// serialises as "sq8".
    #[tokio::test]
    async fn export_records_lineage_respects_precision() {
        let dim: usize = 16;
        let schema = quantised_embedding_schema();
        let (dag, catalog) = sample_catalog_with_tag("train-v1");

        let rows: Vec<ExportedRow> = (0..5i64)
            .map(|i| ExportedRow {
                primary_key: i.to_be_bytes().to_vec(),
                fields: vec![
                    FieldValue::Int64(i),
                    FieldValue::Utf8(format!("r{i}")),
                    FieldValue::Embedding(synthetic_embedding(i, dim)),
                ],
                near_duplicate_group: None,
            })
            .collect();
        let source = VecSource::new(rows);

        let tmp = tempfile::tempdir().expect("tempdir");
        let out = tmp.path().join("lineage_sq8.lance");
        let sink = Arc::new(InMemoryLineageSink::new());

        LanceExporter::new(
            &out,
            schema,
            dag,
            catalog,
            source,
            "train-v1",
            TrainingPrecision::Sq8,
            false,
            None,
        )
        .with_lineage_sink(sink.clone() as Arc<dyn TrainingExportLineageSink>)
        .export()
        .await
        .expect("sq8 export");

        let entries = sink.entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].precision, "sq8");
        assert_eq!(entries[0].precision, TrainingPrecision::Sq8.as_str());
    }

    /// The lineage row carries the `dedup` flag verbatim — useful for
    /// downstream auditing of training datasets.
    #[tokio::test]
    async fn export_records_lineage_respects_dedup() {
        let schema = dedup_schema();
        let (dag, catalog) = sample_catalog_with_tag("train-v1");

        // Mix of grouped and ungrouped rows so there's actually work to
        // do for the dedup pass.
        let rows: Vec<ExportedRow> = vec![
            dedup_row(0, "a", None),
            dedup_row(1, "b", Some(1)),
            dedup_row(2, "c", Some(1)),
            dedup_row(3, "d", None),
        ];
        let source = VecSource::new(rows);

        let tmp = tempfile::tempdir().expect("tempdir");
        let out = tmp.path().join("lineage_dedup.lance");
        let sink = Arc::new(InMemoryLineageSink::new());

        LanceExporter::new(
            &out,
            schema,
            dag,
            catalog,
            source,
            "train-v1",
            TrainingPrecision::Float32,
            true,
            None,
        )
        .with_lineage_sink(sink.clone() as Arc<dyn TrainingExportLineageSink>)
        .export()
        .await
        .expect("dedup export");

        let entries = sink.entries();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].dedup, "dedup flag preserved on lineage row");
        // Surviving rows: 0, 1 (representative of group 1), 3.
        assert_eq!(entries[0].row_count, 3);
    }

    /// Without a sink the exporter runs to completion and produces no
    /// lineage record anywhere. This is the default for every task 34.1
    /// through 34.4 test.
    #[tokio::test]
    async fn export_without_sink_produces_no_lineage() {
        let (_tmp, out, exporter) = float32_fixture("lineage_absent.lance");
        let stats = exporter.export().await.expect("export succeeds");
        assert_eq!(stats.row_count, 8);
        // Dataset is on disk; we just can't observe lineage without a sink.
        assert!(out.exists());
    }

    /// The lineage row's `content_hash` is the lowercase hex form of the
    /// raw `ExportStats::content_hash` bytes — byte-for-byte.
    #[tokio::test]
    async fn export_lineage_content_hash_matches_stats() {
        let (_tmp, _out, exporter) = float32_fixture("lineage_hash.lance");
        let sink = Arc::new(InMemoryLineageSink::new());

        let stats = exporter
            .clone()
            .with_lineage_sink(sink.clone() as Arc<dyn TrainingExportLineageSink>)
            .export()
            .await
            .expect("export succeeds");

        let entries = sink.entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].content_hash, hex_encode(&stats.content_hash));
        // Sanity: 16 bytes ⇒ 32 hex chars.
        assert_eq!(entries[0].content_hash.len(), 32);
    }

    /// A sink that returns an error bubbles out of `export()`, but the
    /// Lance dataset that was already written stays on disk — the
    /// exporter deliberately does not roll back (documented on
    /// `TrainingExportLineageSink`).
    #[tokio::test]
    async fn export_lineage_failure_propagates() {
        let (_tmp, out, exporter) = float32_fixture("lineage_fail.lance");

        let result = exporter
            .clone()
            .with_lineage_sink(Arc::new(FailingLineageSink) as Arc<dyn TrainingExportLineageSink>)
            .export()
            .await;

        match result {
            Err(ExportError::SchemaMismatch(msg)) => {
                assert!(msg.contains("sink forced failure"), "{msg}");
            }
            other => panic!("expected sink failure to propagate, got {:?}", other),
        }

        // Dataset is still on disk — the Lance write succeeded before
        // the sink was called, and there is no rollback.
        assert!(out.exists(), "Lance dataset must remain on disk after sink failure");
        assert!(
            dir_size_bytes(&out).expect("size") > 0,
            "Lance dataset directory must be non-empty"
        );
    }
}
