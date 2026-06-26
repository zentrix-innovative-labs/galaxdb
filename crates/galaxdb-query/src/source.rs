//! [`ArrowSource`] backed by the storage engine's columnar scan (HTAP
//! tasks 7/10). Bridges `Engine::scan_columnar` (column-major typed bytes)
//! to Arrow `RecordBatch`es the DataFusion `TableProvider` consumes.

use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;

use galaxdb_common::{GalaxError, GalaxResult, Timestamp};
use galaxdb_sql::SqlType;
use galaxdb_storage::engine::Engine;

use crate::arrow_batch::{arrow_schema, columnar_batch_to_record_batch};
use crate::{ArrowSource, BatchStream, ReadSnapshot, ScanRequest};

/// An [`ArrowSource`] for one columnar table, reading through
/// `Engine::scan_columnar`.
pub struct EngineArrowSource {
    engine: Arc<Engine>,
    /// Primary-key prefix (`"table:"`) the table's rows share.
    prefix: Vec<u8>,
    /// Logical columns (name + type) in declaration order.
    fields: Vec<(String, SqlType)>,
}

impl EngineArrowSource {
    /// Build a source for `table` whose rows are keyed under `prefix` with
    /// the given logical column list (declaration order).
    pub fn new(engine: Arc<Engine>, prefix: Vec<u8>, fields: Vec<(String, SqlType)>) -> Self {
        Self { engine, prefix, fields }
    }

    /// Resolve a [`ReadSnapshot`] to the timestamp upper bound used by the
    /// storage scan. Tag resolution is the embedded layer's job, so an
    /// unresolved tag here is a typed error rather than a wrong snapshot.
    fn resolve_ts(snapshot: &ReadSnapshot) -> GalaxResult<Timestamp> {
        match snapshot {
            ReadSnapshot::Latest => Ok(Timestamp::MAX),
            ReadSnapshot::AsOfTimestamp(ts) => Ok(*ts),
            ReadSnapshot::AsOfTag(tag) => Err(GalaxError::FeatureNotSupported(format!(
                "version tag '{tag}' must be resolved to a timestamp before the columnar scan"
            ))),
        }
    }

    /// The projected logical fields for a projection (or all fields).
    fn projected_fields(&self, projection: &Option<Vec<usize>>) -> Vec<(String, SqlType)> {
        match projection {
            Some(idxs) => idxs.iter().filter_map(|&i| self.fields.get(i).cloned()).collect(),
            None => self.fields.clone(),
        }
    }
}

impl ArrowSource for EngineArrowSource {
    fn schema(&self, _table: &str) -> GalaxResult<SchemaRef> {
        Ok(arrow_schema(&self.fields))
    }

    fn scan(&self, req: ScanRequest) -> GalaxResult<BatchStream> {
        let read_ts = Self::resolve_ts(&req.snapshot)?;
        let projection: Vec<usize> = req
            .projection
            .clone()
            .unwrap_or_else(|| (0..self.fields.len()).collect());

        // Predicate pushdown is not yet wired (the backend reports filters
        // as unsupported, so DataFusion re-checks them); pass none here.
        let batch = self
            .engine
            .scan_columnar(&self.prefix, &projection, &[], read_ts)?;

        let fields = self.projected_fields(&req.projection);
        let record_batch: RecordBatch = columnar_batch_to_record_batch(&batch, &fields)?;
        Ok(Box::new(std::iter::once(Ok(record_batch))))
    }

    fn insert(&self, _table: &str, _batches: BatchStream) -> GalaxResult<u64> {
        Err(GalaxError::FeatureNotSupported(
            "INSERT ... SELECT through the Arrow source is not yet supported".into(),
        ))
    }
}
