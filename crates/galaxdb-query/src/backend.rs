//! DataFusion-backed [`QueryBackend`] (HTAP tasks 10–11).
//!
//! This is the only module that drives Apache DataFusion. A
//! [`GalaxTableProvider`] adapts a GalaxDB [`ArrowSource`] to DataFusion's
//! `TableProvider`; [`DataFusionBackend`] registers those providers in a
//! fresh `SessionContext` per query and runs the analytical SQL, returning
//! Arrow result batches. No DataFusion type escapes this crate (Req 7.1);
//! DataFusion error text is wrapped in [`GalaxError::Query`] (Req 7.3).

use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;

use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::datasource::MemTable;
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::logical_expr::{Expr, TableType};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionContext;

use galaxdb_common::{GalaxError, GalaxResult};

use crate::{
    ArrowSource, GalaxLogicalPlan, PlanBody, QueryBackend, QueryContext, ReadSnapshot,
    ResultStream, ScanRequest,
};

/// Wrap a DataFusion error as a GalaxDB-owned [`GalaxError::Query`]. The
/// underlying text is kept for diagnostics; sanitizing it to a stable
/// SQLSTATE-coded message is HTAP task 12.
fn query_err(e: DataFusionError) -> GalaxError {
    GalaxError::Query(e.to_string())
}

/// Adapts a GalaxDB [`ArrowSource`] to a DataFusion `TableProvider`.
///
/// `scan` materializes the table's current snapshot as Arrow via the
/// source, then delegates projection / filter / limit to an in-memory
/// `MemTable`. Pushing projection and predicates down into
/// `Engine::scan_columnar` is a later optimization (the storage scan already
/// supports both); correctness here does not depend on it.
struct GalaxTableProvider {
    table: String,
    source: Arc<dyn ArrowSource>,
    schema: SchemaRef,
    snapshot: ReadSnapshot,
}

impl std::fmt::Debug for GalaxTableProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GalaxTableProvider")
            .field("table", &self.table)
            .field("snapshot", &self.snapshot)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl TableProvider for GalaxTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        let req = ScanRequest {
            table: self.table.clone(),
            projection: None,
            filters: Vec::new(),
            limit: None,
            snapshot: self.snapshot.clone(),
        };
        let batches: Vec<RecordBatch> = self
            .source
            .scan(req)
            .map_err(|e| DataFusionError::External(Box::new(e)))?
            .collect::<GalaxResult<Vec<_>>>()
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let mem = MemTable::try_new(self.schema.clone(), vec![batches])?;
        mem.scan(state, projection, filters, limit).await
    }
}

/// The DataFusion implementation of [`QueryBackend`].
#[derive(Default)]
pub struct DataFusionBackend {
    sources: Mutex<HashMap<String, Arc<dyn ArrowSource>>>,
}

impl DataFusionBackend {
    /// Create an empty backend with no registered tables.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl QueryBackend for DataFusionBackend {
    fn register(&self, table: &str, source: Arc<dyn ArrowSource>) -> GalaxResult<()> {
        self.sources
            .lock()
            .map_err(|_| GalaxError::Internal("query backend registry lock".into()))?
            .insert(table.to_string(), source);
        Ok(())
    }

    async fn execute(
        &self,
        plan: GalaxLogicalPlan,
        ctx: &QueryContext,
    ) -> GalaxResult<ResultStream> {
        let session = SessionContext::new();

        // Register a provider for each referenced table. Scope the lock so
        // it is released before the await below.
        {
            let sources = self
                .sources
                .lock()
                .map_err(|_| GalaxError::Internal("query backend registry lock".into()))?;
            for table in &plan.referenced_tables {
                let source = sources.get(table).ok_or_else(|| {
                    GalaxError::Query(format!("table '{table}' is not registered with the query engine"))
                })?;
                let schema = source.schema(table)?;
                let provider = GalaxTableProvider {
                    table: table.clone(),
                    source: source.clone(),
                    schema,
                    snapshot: ctx.snapshot.clone(),
                };
                session
                    .register_table(table.as_str(), Arc::new(provider))
                    .map_err(query_err)?;
            }
        }

        let PlanBody::AnalyticalSql(sql) = &plan.body;
        let df = session.sql(sql).await.map_err(query_err)?;
        let batches = df.collect().await.map_err(query_err)?;
        Ok(Box::new(batches.into_iter().map(Ok)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use crate::{BatchStream, PlanBody};

    /// A fixed in-memory [`ArrowSource`] for backend tests.
    struct StaticSource {
        schema: SchemaRef,
        batch: RecordBatch,
    }
    impl ArrowSource for StaticSource {
        fn schema(&self, _table: &str) -> GalaxResult<SchemaRef> {
            Ok(self.schema.clone())
        }
        fn scan(&self, _req: ScanRequest) -> GalaxResult<BatchStream> {
            Ok(Box::new(std::iter::once(Ok(self.batch.clone()))))
        }
        fn insert(&self, _t: &str, _b: BatchStream) -> GalaxResult<u64> {
            Err(GalaxError::FeatureNotSupported("insert".into()))
        }
    }

    fn static_source(fields: Vec<(&str, Vec<i64>)>) -> Arc<dyn ArrowSource> {
        let arrow_fields: Vec<Field> = fields
            .iter()
            .map(|(n, _)| Field::new(*n, DataType::Int64, false))
            .collect();
        let schema = Arc::new(Schema::new(arrow_fields));
        let arrays: Vec<Arc<dyn arrow::array::Array>> = fields
            .iter()
            .map(|(_, v)| Arc::new(Int64Array::from(v.clone())) as Arc<dyn arrow::array::Array>)
            .collect();
        let batch = RecordBatch::try_new(schema.clone(), arrays).unwrap();
        Arc::new(StaticSource { schema, batch })
    }

    #[tokio::test]
    async fn join_group_by_through_backend() {
        let backend = DataFusionBackend::new();
        // users(id, age); orders(user_id, amount)
        backend
            .register("users", static_source(vec![("id", vec![1, 2]), ("age", vec![30, 40])]))
            .unwrap();
        backend
            .register(
                "orders",
                static_source(vec![("user_id", vec![1, 1, 2]), ("amount", vec![5, 7, 9])]),
            )
            .unwrap();

        let plan = GalaxLogicalPlan {
            referenced_tables: vec!["users".into(), "orders".into()],
            body: PlanBody::AnalyticalSql(
                "SELECT users.age AS age, COUNT(*) AS n \
                 FROM users JOIN orders ON users.id = orders.user_id \
                 GROUP BY users.age ORDER BY users.age"
                    .into(),
            ),
        };

        let batches: Vec<RecordBatch> = backend
            .execute(plan, &QueryContext::default())
            .await
            .unwrap()
            .collect::<GalaxResult<Vec<_>>>()
            .unwrap();

        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 2, "two age groups");
        let first = &batches[0];
        let ages = first.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let counts = first.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        // age 30 → 2 orders (users 1), age 40 → 1 order (user 2)
        assert_eq!(ages.value(0), 30);
        assert_eq!(counts.value(0), 2);
        assert_eq!(ages.value(1), 40);
        assert_eq!(counts.value(1), 1);
    }

    #[tokio::test]
    async fn unregistered_table_is_typed_error() {
        let backend = DataFusionBackend::new();
        let plan = GalaxLogicalPlan {
            referenced_tables: vec!["ghost".into()],
            body: PlanBody::AnalyticalSql("SELECT * FROM ghost".into()),
        };
        let result = backend.execute(plan, &QueryContext::default()).await;
        assert!(matches!(result, Err(GalaxError::Query(_))));
    }

    // ---- Full-stack integration: real engine → columnar → Arrow → JOIN ----

    use galaxdb_common::ColumnType;
    use galaxdb_sql::SqlType;
    use galaxdb_storage::columnar::RowColumnSplitter;
    use galaxdb_storage::engine::{Engine, EngineConfig};
    use crate::source::EngineArrowSource;

    /// Splits a 16-byte value (`a_le(8) ++ b_le(8)`) into two Int64 columns.
    struct TwoIntSplitter;
    impl RowColumnSplitter for TwoIntSplitter {
        fn column_types(&self) -> Vec<ColumnType> {
            vec![ColumnType::Int64, ColumnType::Int64]
        }
        fn split(&self, v: &[u8]) -> Option<Vec<Option<Vec<u8>>>> {
            if v.len() < 16 {
                return None;
            }
            Some(vec![Some(v[0..8].to_vec()), Some(v[8..16].to_vec())])
        }
    }

    fn two_int_value(a: i64, b: i64) -> Vec<u8> {
        let mut v = a.to_le_bytes().to_vec();
        v.extend_from_slice(&b.to_le_bytes());
        v
    }

    #[tokio::test]
    async fn join_group_by_over_real_columnar_engine() {
        let dir = tempfile::tempdir().unwrap();
        let engine = Arc::new(
            Engine::new(EngineConfig {
                data_dir: dir.path().to_path_buf(),
                wal_group_commit_ms: 1,
                ..Default::default()
            })
            .unwrap(),
        );
        engine.register_columnar_table(b"users:".to_vec(), Arc::new(TwoIntSplitter));
        engine.register_columnar_table(b"orders:".to_vec(), Arc::new(TwoIntSplitter));

        // users(id, age): (1,30),(2,40)
        engine.put_sync(b"users:1".to_vec(), two_int_value(1, 30)).unwrap();
        engine.put_sync(b"users:2".to_vec(), two_int_value(2, 40)).unwrap();
        // orders(user_id, amount): user 1 has two, user 2 has one
        engine.put_sync(b"orders:1".to_vec(), two_int_value(1, 5)).unwrap();
        engine.put_sync(b"orders:2".to_vec(), two_int_value(1, 7)).unwrap();
        engine.put_sync(b"orders:3".to_vec(), two_int_value(2, 9)).unwrap();
        engine.flush_memtable().await.unwrap(); // typed columnar SST blocks

        let backend = DataFusionBackend::new();
        backend
            .register(
                "users",
                Arc::new(EngineArrowSource::new(
                    engine.clone(),
                    b"users:".to_vec(),
                    vec![("id".into(), SqlType::Int8), ("age".into(), SqlType::Int8)],
                )),
            )
            .unwrap();
        backend
            .register(
                "orders",
                Arc::new(EngineArrowSource::new(
                    engine.clone(),
                    b"orders:".to_vec(),
                    vec![("user_id".into(), SqlType::Int8), ("amount".into(), SqlType::Int8)],
                )),
            )
            .unwrap();

        let plan = GalaxLogicalPlan {
            referenced_tables: vec!["users".into(), "orders".into()],
            body: PlanBody::AnalyticalSql(
                "SELECT users.age AS age, COUNT(*) AS n, SUM(orders.amount) AS total \
                 FROM users JOIN orders ON users.id = orders.user_id \
                 GROUP BY users.age ORDER BY users.age"
                    .into(),
            ),
        };
        let batches: Vec<RecordBatch> = backend
            .execute(plan, &QueryContext::default())
            .await
            .unwrap()
            .collect::<GalaxResult<Vec<_>>>()
            .unwrap();

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 2);
        let b0 = &batches[0];
        let ages = b0.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let counts = b0.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        let totals = b0.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(ages.value(0), 30);
        assert_eq!(counts.value(0), 2);
        assert_eq!(totals.value(0), 12); // 5 + 7
        assert_eq!(ages.value(1), 40);
        assert_eq!(counts.value(1), 1);
        assert_eq!(totals.value(1), 9);
    }
}
