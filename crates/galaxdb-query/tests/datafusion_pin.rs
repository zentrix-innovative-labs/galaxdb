//! Smoke test: the pinned DataFusion (`=52.5.0`) resolves, links, and runs.
//!
//! This is not the `DataFusionBackend` (HTAP task 11) — it is a minimal
//! end-to-end proof that the exact-pinned dependency is usable from this
//! crate, so the pin (Req 7.2) is verified rather than asserted. It also
//! anchors the API surface the backend will build on (`SessionContext`,
//! `MemTable`, Arrow `RecordBatch`).

use std::sync::Arc;

use datafusion::arrow::array::{Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::prelude::SessionContext;

#[tokio::test]
async fn pinned_datafusion_runs_a_group_by() {
    // Build a tiny Arrow table: (id, category).
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("category", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
            Arc::new(StringArray::from(vec!["a", "b", "a", "a"])),
        ],
    )
    .expect("record batch");

    let ctx = SessionContext::new();
    let table = MemTable::try_new(schema, vec![vec![batch]]).expect("memtable");
    ctx.register_table("t", Arc::new(table)).expect("register");

    // A real aggregate query through the pinned engine.
    let df = ctx
        .sql("SELECT category, COUNT(*) AS n FROM t GROUP BY category ORDER BY category")
        .await
        .expect("plan");
    let results = df.collect().await.expect("collect");

    let total: usize = results.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 2, "two distinct categories expected");

    // Verify the 'a' bucket has count 3.
    let first = &results[0];
    let cats = first
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let counts = first
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(cats.value(0), "a");
    assert_eq!(counts.value(0), 3);
}
