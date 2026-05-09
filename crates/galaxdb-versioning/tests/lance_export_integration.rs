//! Integration tests for the Lance training exporter (Req 25, task 34.6).
//!
//! These tests live outside the crate and exercise the exporter strictly
//! through its **public API**. Where 34.1 … 34.5 each added unit tests
//! living inside `src/export.rs` with access to private helpers, this
//! module is the acceptance gate for the public interface the downstream
//! `galaxdb-embedded` crate and training jobs actually consume.
//!
//! The four acceptance dimensions for task 34.6 map one-to-one onto the
//! test groups in this file:
//!
//! 1. **Lance export produces valid dataset** — `lance_export_*`
//!    end-to-end write/read.
//! 2. **Precision conversion correctness** — `precision_*` for float32,
//!    sq8, rabitq.
//! 3. **Dedup filtering** — `dedup_*` on / off.
//! 4. **Lineage record created** — `lineage_*` with and without a sink.
//!
//! Plus one cross-cutting `full_pipeline_end_to_end` test that combines
//! dedup + sq8 + lineage.
//!
//! ### Test doubles
//!
//! No mocks. `VecSource` is a legitimate in-file implementation of
//! [`LanceExportSource`] that returns a pre-built `Vec<ExportedRow>` in
//! response to any block-id list — the equivalent of a `Vec`-backed
//! storage engine. This is how `galaxdb-embedded` would plug a real PAX
//! block reader into the exporter in production.

use std::sync::Arc;

use arrow::array::{
    Array, AsArray, BinaryArray, Float32Array, Int64Array,
};
use arrow::datatypes::{DataType, Field, Float32Type, Schema as ArrowSchema};
use galaxdb_common::types::BlockId;
use galaxdb_versioning::{
    ExportError, ExportResult, ExportStats, ExportedRow, FieldValue, InMemoryLineageSink,
    LanceExportSource, LanceExporter, MerkleDag, TagCatalog, TrainingExportLineageSink,
    TrainingPrecision, TrainingTagMetadata,
};
use lance::Dataset;
use tempfile::tempdir;

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

/// In-memory [`LanceExportSource`] backed by a `Vec<ExportedRow>`. Not a
/// mock — a legitimate alternative trait implementation. Production
/// callers plug in a reader that materialises PAX blocks; tests plug in
/// this. The block-id list is intentionally ignored: the catalog below
/// pins `[1, 2]` for every test tag, and the source always returns its
/// full row buffer regardless of which subset the exporter asks for.
struct VecSource {
    rows: Vec<ExportedRow>,
}

impl VecSource {
    fn new(rows: Vec<ExportedRow>) -> Arc<Self> {
        Arc::new(Self { rows })
    }
}

impl LanceExportSource for VecSource {
    fn read_blocks(&self, _block_ids: &[BlockId]) -> ExportResult<Vec<ExportedRow>> {
        Ok(self.rows.clone())
    }
}

/// Build a tag catalog pinning a single `train-v1` tag, together with a
/// minimal [`MerkleDag`] that references the same block set. The
/// returned `Arc`s are what the exporter API expects.
fn sample_catalog() -> (Arc<MerkleDag>, Arc<TagCatalog>) {
    let mut dag = MerkleDag::new();
    let root = dag.commit(1_000, vec![111, 222], vec![1, 2]);

    let mut catalog = TagCatalog::new();
    catalog
        .create_tag(
            "train-v1".to_string(),
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

/// Arrow schema with an `Int64` pk, a `Utf8` text column, and a
/// `FixedSizeList<Float32, dim>` embedding — used by the float32 tests.
fn fixed_size_list_schema(dim: i32) -> Arc<ArrowSchema> {
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

/// Arrow schema where the embedding column is `Binary` — used by the
/// sq8 and rabitq precision tests.
fn binary_embedding_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("pk", DataType::Int64, false),
        Field::new("text", DataType::Utf8, false),
        Field::new("embedding", DataType::Binary, false),
    ]))
}

/// Two-column schema used by the dedup tests. No embeddings — dedup is
/// orthogonal to precision, so we keep it simple.
fn two_col_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("pk", DataType::Int64, false),
        Field::new("text", DataType::Utf8, false),
    ]))
}

/// Deterministic pseudo-random embedding generator — avoids a `rand`
/// dev-dep and gives byte-identical vectors across platforms.
fn synthetic_embedding(seed: i64, dim: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(dim);
    let mut state = (seed as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(1);
    for _ in 0..dim {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let x = (state >> 33) as f32 / (u32::MAX as f32 / 2.0) - 1.0;
        v.push(x);
    }
    v
}

/// Build a single embedding row: `(pk, "row-{pk}", embedding)` with the
/// group ID defaulting to `None`.
fn make_embedding_row(pk: i64, emb: Vec<f32>) -> ExportedRow {
    ExportedRow {
        primary_key: pk.to_be_bytes().to_vec(),
        fields: vec![
            FieldValue::Int64(pk),
            FieldValue::Utf8(format!("row-{pk}")),
            FieldValue::Embedding(emb),
        ],
        near_duplicate_group: None,
    }
}

/// Two-column (pk, text) row with an optional dedup group.
fn make_dedup_row(pk: i64, group: Option<u64>) -> ExportedRow {
    ExportedRow {
        primary_key: pk.to_be_bytes().to_vec(),
        fields: vec![
            FieldValue::Int64(pk),
            FieldValue::Utf8(format!("row-{pk}")),
        ],
        near_duplicate_group: group,
    }
}

/// Read the `pk` column out of a Lance dataset in scan order, which —
/// because the exporter always sorts by primary key before writing — is
/// always ascending for positive `i64` PKs.
async fn read_pk_column(out: &std::path::Path) -> Vec<i64> {
    let ds = Dataset::open(out.to_str().unwrap())
        .await
        .expect("open dataset");
    let batch = ds.scan().try_into_batch().await.expect("scan batch");
    let pks = batch
        .column_by_name("pk")
        .expect("pk column present")
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("Int64Array");
    (0..batch.num_rows()).map(|i| pks.value(i)).collect()
}

/// Read every row of a `Binary` column as a `Vec<Vec<u8>>`.
async fn read_binary_column(out: &std::path::Path, column: &str) -> Vec<Vec<u8>> {
    let ds = Dataset::open(out.to_str().unwrap())
        .await
        .expect("open dataset");
    let batch = ds.scan().try_into_batch().await.expect("scan batch");
    let col = batch
        .column_by_name(column)
        .expect("column present")
        .as_any()
        .downcast_ref::<BinaryArray>()
        .expect("BinaryArray");
    (0..col.len()).map(|i| col.value(i).to_vec()).collect()
}

// ---------------------------------------------------------------------------
// Group 1 — Lance export produces valid dataset (end-to-end)
// ---------------------------------------------------------------------------

/// Writing 256 rows with an int64 PK, a utf8 text column, and a
/// `FixedSizeList<Float32, 32>` embedding produces a Lance dataset that
/// can be re-opened via the public `lance::Dataset::open` API and
/// round-trips the row count. This is the smoke test for the overall
/// public contract.
#[tokio::test]
async fn lance_export_produces_valid_dataset_end_to_end() {
    let dim: i32 = 32;
    let schema = fixed_size_list_schema(dim);
    let (dag, catalog) = sample_catalog();

    // 256 rows, deliberately shuffled so the sort step is also exercised.
    let mut rows = Vec::with_capacity(256);
    for i in 0..256i64 {
        let pk = (i * 17 + 5) % 256;
        rows.push(make_embedding_row(
            pk,
            synthetic_embedding(pk, dim as usize),
        ));
    }
    let source = VecSource::new(rows);

    let tmp = tempdir().expect("tempdir");
    let out = tmp.path().join("train.lance");

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

    let stats: ExportStats = exporter.export().await.expect("export succeeds");
    assert_eq!(stats.row_count, 256, "all 256 rows written");
    assert!(stats.byte_count > 0, "byte_count reflects on-disk bytes");
    assert_ne!(
        stats.content_hash, [0u8; 16],
        "content hash is non-zero after writing rows"
    );

    // The output path is a populated directory on disk.
    assert!(out.exists(), "dataset directory exists");
    assert!(
        out.is_dir(),
        "Lance writes the dataset as a directory, not a single file"
    );
    let entries: Vec<_> = std::fs::read_dir(&out)
        .expect("read_dir")
        .filter_map(Result::ok)
        .collect();
    assert!(
        !entries.is_empty(),
        "dataset directory must contain at least one file"
    );

    // Re-open the Lance dataset via the public API and confirm the
    // scan-side row count matches.
    let ds = Dataset::open(out.to_str().unwrap())
        .await
        .expect("open written dataset");
    let count = ds.scan().count_rows().await.expect("count_rows");
    assert_eq!(count, 256, "Dataset::scan().count_rows() matches stats");
}

// ---------------------------------------------------------------------------
// Group 2 — Precision conversion correctness
// ---------------------------------------------------------------------------

/// `TrainingPrecision::Float32` is a passthrough: the embedding column
/// stays a float list on disk, and the first row's values survive
/// byte-for-byte (same bit pattern via `to_le_bytes`).
#[tokio::test]
async fn precision_float32_preserves_embedding_bytes() {
    let dim: i32 = 64;
    let schema = fixed_size_list_schema(dim);
    let (dag, catalog) = sample_catalog();

    let rows: Vec<ExportedRow> = (0..64i64)
        .map(|i| make_embedding_row(i, synthetic_embedding(i, dim as usize)))
        .collect();
    // Capture the sorted-order first embedding up front for comparison
    // after the round trip.
    let mut sorted = rows.clone();
    sorted.sort_by(|a, b| a.primary_key.cmp(&b.primary_key));
    let first_input = match &sorted[0].fields[2] {
        FieldValue::Embedding(v) => v.clone(),
        _ => unreachable!("fixture row must have an embedding"),
    };

    let source = VecSource::new(rows);
    let tmp = tempdir().expect("tempdir");
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
    assert_eq!(stats.row_count, 64);

    let ds = Dataset::open(out.to_str().unwrap())
        .await
        .expect("open dataset");
    let emb_field = ds
        .schema()
        .field("embedding")
        .expect("embedding field present");
    // Lance may internally flatten a FixedSizeList<Float32> into
    // List<Float32>. The contract is that it round-trips as *some* list
    // of float32 — both variants are accepted here.
    match emb_field.data_type() {
        DataType::FixedSizeList(child, _) => {
            assert_eq!(child.data_type(), &DataType::Float32);
        }
        DataType::List(child) => {
            assert_eq!(child.data_type(), &DataType::Float32);
        }
        other => panic!(
            "float32 export must keep a float list column, got {:?}",
            other
        ),
    }

    // Pull the raw float values out of the column regardless of the
    // underlying list variant and check that the first row's 64 floats
    // match the input bit-for-bit.
    let batch = ds.scan().try_into_batch().await.expect("batch");
    let col = batch
        .column_by_name("embedding")
        .expect("embedding column present");
    let floats: Vec<f32> = match col.data_type() {
        DataType::FixedSizeList(_, _) => col
            .as_fixed_size_list()
            .values()
            .as_primitive::<Float32Type>()
            .values()
            .to_vec(),
        DataType::List(_) => col
            .as_list::<i32>()
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("float32 values")
            .values()
            .to_vec(),
        other => panic!("unexpected embedding arrow type: {:?}", other),
    };
    assert_eq!(floats.len(), 64 * dim as usize);
    assert_eq!(
        &floats[..dim as usize],
        first_input.as_slice(),
        "first row embedding bytes preserved verbatim under Float32"
    );
}

/// `TrainingPrecision::Sq8` lands the embedding column on disk as
/// `Binary` with exactly `dim` bytes per row (1 byte per dim, 4×
/// compression vs f32).
#[tokio::test]
async fn precision_sq8_writes_binary_one_byte_per_dim() {
    let dim: usize = 64;
    let schema = binary_embedding_schema();
    let (dag, catalog) = sample_catalog();

    let rows: Vec<ExportedRow> = (0..64i64)
        .map(|i| make_embedding_row(i, synthetic_embedding(i, dim)))
        .collect();
    let source = VecSource::new(rows);

    let tmp = tempdir().expect("tempdir");
    let out = tmp.path().join("sq8.lance");

    let stats = LanceExporter::new(
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
    .export()
    .await
    .expect("sq8 export");
    assert_eq!(stats.row_count, 64);

    let ds = Dataset::open(out.to_str().unwrap())
        .await
        .expect("open dataset");
    let emb_field = ds.schema().field("embedding").expect("embedding field");
    assert_eq!(
        emb_field.data_type(),
        DataType::Binary,
        "SQ8 precision writes the embedding column as Binary"
    );

    let bytes = read_binary_column(&out, "embedding").await;
    assert_eq!(bytes.len(), 64);
    for row_bytes in &bytes {
        assert_eq!(
            row_bytes.len(),
            dim,
            "SQ8 packs 1 byte per dimension"
        );
    }
}

/// `TrainingPrecision::Rabitq` lands the embedding column on disk as
/// `Binary` with exactly `dim / 8` bytes per row (1 bit per dim, 32×
/// compression vs f32).
#[tokio::test]
async fn precision_rabitq_writes_binary_one_bit_per_dim() {
    let dim: usize = 64;
    let schema = binary_embedding_schema();
    let (dag, catalog) = sample_catalog();

    let rows: Vec<ExportedRow> = (0..64i64)
        .map(|i| make_embedding_row(i, synthetic_embedding(i, dim)))
        .collect();
    let source = VecSource::new(rows);

    let tmp = tempdir().expect("tempdir");
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
        Some(7),
    )
    .export()
    .await
    .expect("rabitq export");
    assert_eq!(stats.row_count, 64);

    let ds = Dataset::open(out.to_str().unwrap())
        .await
        .expect("open dataset");
    let emb_field = ds.schema().field("embedding").expect("embedding field");
    assert_eq!(
        emb_field.data_type(),
        DataType::Binary,
        "RaBitQ precision writes the embedding column as Binary"
    );

    let bytes = read_binary_column(&out, "embedding").await;
    assert_eq!(bytes.len(), 64);
    for row_bytes in &bytes {
        assert_eq!(
            row_bytes.len(),
            dim / 8,
            "RaBitQ packs 1 bit per dimension into bytes"
        );
    }
}

// ---------------------------------------------------------------------------
// Group 3 — Dedup filtering
// ---------------------------------------------------------------------------

/// Build a 20-row fixture where the 5 rows with PKs 10..15 all share
/// dedup group `42`; every other row has `near_duplicate_group = None`.
/// With dedup ON the 5 grouped rows collapse to a single representative
/// (PK 10) — 20 − 4 = 16 survivors.
fn dedup_fixture_20_with_one_group_of_5() -> Vec<ExportedRow> {
    (0..20i64)
        .map(|i| {
            let group = if (10..15).contains(&i) {
                Some(42u64)
            } else {
                None
            };
            make_dedup_row(i, group)
        })
        .collect()
}

/// With `dedup = false` every input row survives — even the ones that
/// share a near-duplicate group.
#[tokio::test]
async fn dedup_off_preserves_all_rows() {
    let schema = two_col_schema();
    let (dag, catalog) = sample_catalog();

    let rows = dedup_fixture_20_with_one_group_of_5();
    assert_eq!(rows.len(), 20, "fixture sanity check");
    let source = VecSource::new(rows);

    let tmp = tempdir().expect("tempdir");
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
    assert_eq!(
        stats.row_count, 20,
        "dedup=false retains every row, including grouped ones"
    );

    let pks = read_pk_column(&out).await;
    assert_eq!(pks, (0..20).collect::<Vec<i64>>());
}

/// With `dedup = true` the 5 rows in a single group collapse to one
/// representative (lowest PK = 10), leaving 16 rows: `[0..10] ∪ [10]
/// ∪ [15..20]`.
#[tokio::test]
async fn dedup_on_collapses_groups() {
    let schema = two_col_schema();
    let (dag, catalog) = sample_catalog();

    let rows = dedup_fixture_20_with_one_group_of_5();
    let source = VecSource::new(rows);

    let tmp = tempdir().expect("tempdir");
    let out = tmp.path().join("dedup_on.lance");

    let stats = LanceExporter::new(
        &out,
        schema,
        dag,
        catalog,
        source,
        "train-v1",
        TrainingPrecision::Float32,
        true, // dedup ON
        None,
    )
    .export()
    .await
    .expect("export");
    assert_eq!(
        stats.row_count, 16,
        "5-row near-dup group collapses to 1 (20 − 4 = 16)"
    );

    let pks = read_pk_column(&out).await;
    // Survivors: PKs 0..=9 are ungrouped; PK 10 is the lowest in group
    // 42 and is kept as the representative; PKs 11..=14 are dropped as
    // near-duplicates; PKs 15..=19 are ungrouped and kept.
    let expected: Vec<i64> = (0..=10).chain(15..20).collect();
    assert_eq!(
        pks, expected,
        "survivor PKs: 0..=10 (10 is the group rep) plus 15..=19"
    );
}

// ---------------------------------------------------------------------------
// Group 4 — Lineage record created
// ---------------------------------------------------------------------------

/// When an `InMemoryLineageSink` is attached, a successful export
/// records *exactly one* lineage entry whose fields all match the
/// exporter configuration and the returned `ExportStats`.
#[tokio::test]
async fn export_with_sink_emits_exactly_one_record() {
    let dim: i32 = 16;
    let schema = fixed_size_list_schema(dim);
    let (dag, catalog) = sample_catalog();

    let rows: Vec<ExportedRow> = (0..8i64)
        .map(|i| make_embedding_row(i, synthetic_embedding(i, dim as usize)))
        .collect();
    let source = VecSource::new(rows);

    let tmp = tempdir().expect("tempdir");
    let out = tmp.path().join("lineage_ok.lance");

    let sink = Arc::new(InMemoryLineageSink::new());
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
    .with_lineage_sink(sink.clone() as Arc<dyn TrainingExportLineageSink>)
    .export()
    .await
    .expect("export succeeds");

    let entries = sink.entries();
    assert_eq!(entries.len(), 1, "exactly one lineage row per export");

    let row = &entries[0];
    assert_eq!(row.tag_name, "train-v1");
    assert_eq!(row.filter_expr, None);
    assert_eq!(row.precision, "float32");
    assert_eq!(row.precision, TrainingPrecision::Float32.as_str());
    assert!(!row.dedup);
    assert_eq!(row.row_count, stats.row_count);
    assert_eq!(row.row_count, 8);
    // content_hash is the lowercase-hex form of the stats content_hash.
    let expected_hex = stats
        .content_hash
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>();
    assert_eq!(row.content_hash, expected_hex);
    assert_eq!(row.content_hash.len(), 32, "16 bytes ⇒ 32 hex chars");
}

/// Without a sink attached, `export()` still succeeds and produces a
/// valid on-disk dataset — there is simply no lineage side effect to
/// observe. This matches the documented default behaviour before Req 38
/// is enabled by the caller.
#[tokio::test]
async fn export_without_sink_is_a_no_op() {
    let dim: i32 = 8;
    let schema = fixed_size_list_schema(dim);
    let (dag, catalog) = sample_catalog();

    let rows: Vec<ExportedRow> = (0..4i64)
        .map(|i| make_embedding_row(i, synthetic_embedding(i, dim as usize)))
        .collect();
    let source = VecSource::new(rows);

    let tmp = tempdir().expect("tempdir");
    let out = tmp.path().join("lineage_absent.lance");

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
    .expect("export succeeds");
    assert_eq!(stats.row_count, 4);
    assert!(out.exists(), "dataset is still on disk");

    // We can't directly observe "no lineage row" without a sink — the
    // assertion here is that `export()` doesn't require one, and the
    // next test (`full_pipeline_end_to_end`) re-confirms that attaching
    // a sink produces exactly one record, so the two together pin down
    // the behaviour.
}

// ---------------------------------------------------------------------------
// Cross-cutting — full pipeline combining all four dimensions
// ---------------------------------------------------------------------------

/// Combine dedup=true + precision=Sq8 + lineage sink attached into a
/// single export. Verifies, in order:
/// 1. The Lance dataset is a valid directory readable via
///    `Dataset::open`.
/// 2. The embedding column landed on disk as `Binary` (Sq8 bytes).
/// 3. Dedup collapsed the near-dup group — 6 input rows, 3 in one
///    group, ⇒ 4 survivors.
/// 4. Exactly one lineage record was emitted, with `precision = "sq8"`,
///    `dedup = true`, and `row_count = 4` matching the post-dedup
///    count.
#[tokio::test]
async fn full_pipeline_end_to_end() {
    let dim: usize = 16;
    let schema = binary_embedding_schema();
    let (dag, catalog) = sample_catalog();

    // 6 rows: PKs 0, 1, 5 are unique; PKs 2, 3, 4 share group 7. After
    // dedup the survivors are [0, 1, 2, 5] — 4 rows total.
    let rows: Vec<ExportedRow> = (0..6i64)
        .map(|i| {
            let group = if (2..5).contains(&i) { Some(7u64) } else { None };
            ExportedRow {
                primary_key: i.to_be_bytes().to_vec(),
                fields: vec![
                    FieldValue::Int64(i),
                    FieldValue::Utf8(format!("row-{i}")),
                    FieldValue::Embedding(synthetic_embedding(i, dim)),
                ],
                near_duplicate_group: group,
            }
        })
        .collect();
    let source = VecSource::new(rows);

    let tmp = tempdir().expect("tempdir");
    let out = tmp.path().join("full_pipeline.lance");
    let sink = Arc::new(InMemoryLineageSink::new());

    let stats = LanceExporter::new(
        &out,
        schema,
        dag,
        catalog,
        source,
        "train-v1",
        TrainingPrecision::Sq8,
        true,
        Some(123),
    )
    .with_lineage_sink(sink.clone() as Arc<dyn TrainingExportLineageSink>)
    .with_filter_expr(Some("WHERE deleted = FALSE".into()))
    .export()
    .await
    .expect("full pipeline export");

    // (1) Valid Lance dataset on disk.
    assert!(out.exists() && out.is_dir());
    let ds = Dataset::open(out.to_str().unwrap())
        .await
        .expect("open dataset");
    let scan_count = ds.scan().count_rows().await.expect("count_rows");

    // (2) Embedding column is Binary, SQ8 = 1 byte/dim.
    let emb_field = ds.schema().field("embedding").expect("embedding field");
    assert_eq!(emb_field.data_type(), DataType::Binary);
    let bytes = read_binary_column(&out, "embedding").await;
    for row_bytes in &bytes {
        assert_eq!(row_bytes.len(), dim);
    }

    // (3) Dedup collapsed the group. Expected survivors: PKs [0, 1, 2, 5].
    assert_eq!(stats.row_count, 4, "post-dedup row count");
    assert_eq!(scan_count as u64, 4, "Lance scan count matches");
    let pks = read_pk_column(&out).await;
    assert_eq!(pks, vec![0, 1, 2, 5]);

    // (4) Exactly one lineage record, with every field tracking the
    // exporter configuration.
    let entries = sink.entries();
    assert_eq!(entries.len(), 1, "one lineage row per successful export");
    let lineage = &entries[0];
    assert_eq!(lineage.tag_name, "train-v1");
    assert_eq!(lineage.precision, "sq8");
    assert!(lineage.dedup);
    assert_eq!(lineage.row_count, 4);
    assert_eq!(
        lineage.filter_expr.as_deref(),
        Some("WHERE deleted = FALSE")
    );
    // Hex-encoded content_hash must match the raw stats bytes.
    let expected_hex: String = stats
        .content_hash
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect();
    assert_eq!(lineage.content_hash, expected_hex);
}

// ---------------------------------------------------------------------------
// Error-path smoke test — ensures the public `ExportError` variants are
// visible and pattern-matchable from outside the crate. Not one of the
// four acceptance dimensions but a cheap regression gate against
// accidentally un-`pub`-ing an error variant.
// ---------------------------------------------------------------------------

/// Asking the exporter for a tag that does not exist surfaces
/// `ExportError::TagNotFound` verbatim.
#[tokio::test]
async fn missing_tag_surfaces_tag_not_found_error() {
    let schema = two_col_schema();
    let (dag, catalog) = sample_catalog();

    let source = VecSource::new(vec![make_dedup_row(0, None)]);
    let tmp = tempdir().expect("tempdir");
    let out = tmp.path().join("missing.lance");

    let exporter = LanceExporter::new(
        &out,
        schema,
        dag,
        catalog,
        source,
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
