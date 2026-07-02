//! `SEMANTIC_MATCH` as an analytical operand (HTAP task 16, ADR-0004).
//!
//! A scalar UDF cannot host an efficient HNSW top-k — it would run per row.
//! Instead, the native vector backend computes the candidate set **once**
//! (top-k with the paper's adaptive strategy), and that candidate set is
//! surfaced to DataFusion as an ordinary Arrow table so joins / aggregates /
//! GROUP BY over "the semantically matched rows" execute in the analytical
//! engine. This module owns the boundary type — [`VectorCandidateProvider`]
//! — that the embedded HNSW backend implements, and the [`ArrowSource`]
//! adapter that feeds the candidate batch into the backend. DataFusion's
//! physical layer (a `MemoryExec` over the materialized batch) is the
//! physical operator; no vector type crosses into DataFusion.
//!
//! The candidate set is the matched subset of a base table's rows — the base
//! table's own columns plus a trailing `similarity` (`Float64`) column — so
//! the analytical SQL (with the `SEMANTIC_MATCH(...)` predicate stripped)
//! runs over exactly the matched rows and may reference `similarity` in its
//! projection / ORDER BY.

use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;

use galaxdb_common::GalaxResult;

use crate::{ArrowSource, BatchStream, PredicateSupport, ScanPredicate, ScanRequest};

/// Produces the semantic-match candidate set for one base table.
///
/// Implemented in `galaxdb-embedded` over the native HNSW + delta-buffer
/// backend: it embeds the query text, runs the top-k search with the chosen
/// strategy, resolves the matching rows, and returns them as Arrow batches
/// whose schema is the base table's columns followed by a `similarity`
/// `Float64` column. Called once per analytical query (not per row).
pub trait VectorCandidateProvider: Send + Sync {
    /// The Arrow schema of the candidate batches: the base table's columns
    /// in declaration order, then a non-null `similarity` `Float64` column.
    fn schema(&self) -> SchemaRef;

    /// Compute and return the candidate rows (the matched subset + their
    /// similarity scores). Invoked once when the analytical scan runs.
    fn candidates(&self) -> GalaxResult<Vec<RecordBatch>>;
}

/// Adapts a [`VectorCandidateProvider`] to an [`ArrowSource`] so the
/// DataFusion backend can register it as the source for the base table an
/// analytical query applies `SEMANTIC_MATCH` to. The backend then plans the
/// query (joins/aggregates/GROUP BY) over the matched rows exactly as it
/// would over a normal table.
pub struct SemanticCandidateSource {
    provider: Arc<dyn VectorCandidateProvider>,
}

impl SemanticCandidateSource {
    /// Wrap a candidate provider as an Arrow source.
    pub fn new(provider: Arc<dyn VectorCandidateProvider>) -> Self {
        Self { provider }
    }
}

impl ArrowSource for SemanticCandidateSource {
    fn schema(&self, _table: &str) -> GalaxResult<SchemaRef> {
        Ok(self.provider.schema())
    }

    fn scan(&self, _req: ScanRequest) -> GalaxResult<BatchStream> {
        // The candidate set is materialized once here; projection / filter /
        // limit are applied by the backend's `MemTable` wrapper, mirroring
        // the `EngineArrowSource` path. Snapshot is irrelevant — candidates
        // are already resolved against the query's snapshot by the provider.
        let batches = self.provider.candidates()?;
        Ok(Box::new(batches.into_iter().map(Ok)))
    }

    fn supports_predicate(&self, _table: &str, _predicate: &ScanPredicate) -> PredicateSupport {
        // The candidate rows are re-checked by the backend against any
        // residual relational predicate (the SEMANTIC_MATCH predicate itself
        // is already consumed by producing this set).
        PredicateSupport::Unsupported
    }

    fn insert(&self, _table: &str, _batches: BatchStream) -> GalaxResult<u64> {
        Err(galaxdb_common::GalaxError::FeatureNotSupported(
            "cannot INSERT into a SEMANTIC_MATCH candidate set".into(),
        ))
    }
}

use arrow::datatypes::{DataType, Field, Schema};
use galaxdb_sql::planner::Value;
use galaxdb_sql::SqlType;

/// The Arrow schema of a candidate batch: the base table's `columns` mapped
/// to their logical Arrow types, followed by a non-null `similarity`
/// `Float64` column (HTAP task 16).
pub fn candidate_schema(columns: &[(String, SqlType)]) -> SchemaRef {
    let mut fields: Vec<Field> = columns
        .iter()
        .map(|(name, ty)| Field::new(name, crate::schema::sql_type_to_arrow(ty), true))
        .collect();
    fields.push(Field::new("similarity", DataType::Float64, false));
    Arc::new(Schema::new(fields))
}

/// Build a candidate [`RecordBatch`] from decoded `Value` rows plus their
/// similarity scores (HTAP task 16). `columns` names the base table's
/// columns and their logical types; `rows` is one `Vec<Option<Value>>` per
/// matched row (aligned to `columns`); `similarities` is aligned to `rows`.
/// The scalar types GalaxDB's executor produces — int/float/bool/text and
/// their NULLs — are built into typed Arrow arrays; a column whose logical
/// type has no scalar Arrow builder here (e.g. arrays, decimals) is a typed
/// [`GalaxError::FeatureNotSupported`] rather than a silent wrong column.
pub fn build_candidate_batch(
    columns: &[(String, SqlType)],
    rows: &[Vec<Option<Value>>],
    similarities: &[f64],
) -> GalaxResult<RecordBatch> {
    use arrow::array::{
        ArrayRef, BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array,
        Int64Array, StringArray,
    };

    let schema = candidate_schema(columns);
    let n = rows.len();
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(columns.len() + 1);

    // Helper: pull column `c` from every row as an iterator of `&Option<Value>`.
    for (c, (_name, ty)) in columns.iter().enumerate() {
        let cell = |r: usize| rows[r].get(c).unwrap_or(&None);
        let array: ArrayRef = match ty {
            SqlType::Int2 => Arc::new(
                (0..n)
                    .map(|r| value_as_i64(cell(r)).map(|v| v as i16))
                    .collect::<Int16Array>(),
            ),
            SqlType::Int4 => Arc::new(
                (0..n)
                    .map(|r| value_as_i64(cell(r)).map(|v| v as i32))
                    .collect::<Int32Array>(),
            ),
            SqlType::Int8 => Arc::new(
                (0..n).map(|r| value_as_i64(cell(r))).collect::<Int64Array>(),
            ),
            SqlType::Float4 => Arc::new(
                (0..n)
                    .map(|r| value_as_f64(cell(r)).map(|v| v as f32))
                    .collect::<Float32Array>(),
            ),
            SqlType::Float8 => Arc::new(
                (0..n).map(|r| value_as_f64(cell(r))).collect::<Float64Array>(),
            ),
            SqlType::Bool => Arc::new(
                (0..n).map(|r| value_as_bool(cell(r))).collect::<BooleanArray>(),
            ),
            SqlType::Text
            | SqlType::Varchar(_)
            | SqlType::Json
            | SqlType::Jsonb
            | SqlType::Uuid
            | SqlType::Date
            | SqlType::Timestamp
            | SqlType::TimestampTz => Arc::new(
                (0..n)
                    .map(|r| value_as_text(cell(r)))
                    .collect::<StringArray>(),
            ),
            other => {
                return Err(galaxdb_common::GalaxError::FeatureNotSupported(format!(
                    "SEMANTIC_MATCH candidate column of type {other:?} is not supported \
                     in the analytical path yet"
                )))
            }
        };
        arrays.push(array);
    }
    arrays.push(Arc::new(Float64Array::from(similarities.to_vec())));

    RecordBatch::try_new(schema, arrays)
        .map_err(|e| galaxdb_common::GalaxError::Internal(format!("candidate batch: {e}")))
}

fn value_as_i64(v: &Option<Value>) -> Option<i64> {
    match v {
        Some(Value::Integer(n)) => Some(*n),
        Some(Value::Float(f)) => Some(*f as i64),
        Some(Value::Text(s)) => s.trim().parse().ok(),
        _ => None,
    }
}

fn value_as_f64(v: &Option<Value>) -> Option<f64> {
    match v {
        Some(Value::Float(f)) => Some(*f),
        Some(Value::Integer(n)) => Some(*n as f64),
        Some(Value::Text(s)) => s.trim().parse().ok(),
        _ => None,
    }
}

fn value_as_bool(v: &Option<Value>) -> Option<bool> {
    match v {
        Some(Value::Bool(b)) => Some(*b),
        Some(Value::Text(s)) => match s.trim() {
            "t" | "true" | "TRUE" | "1" => Some(true),
            "f" | "false" | "FALSE" | "0" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

fn value_as_text(v: &Option<Value>) -> Option<String> {
    match v {
        None | Some(Value::Null) => None,
        Some(other) => Some(galaxdb_sql::row_codec::value_display(other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Float64Array, Int64Array, StringArray};

    #[test]
    fn candidate_schema_appends_similarity() {
        let cols = vec![
            ("id".to_string(), SqlType::Int8),
            ("name".to_string(), SqlType::Text),
        ];
        let schema = candidate_schema(&cols);
        assert_eq!(schema.fields().len(), 3);
        assert_eq!(schema.field(2).name(), "similarity");
        assert_eq!(schema.field(2).data_type(), &DataType::Float64);
        assert!(!schema.field(2).is_nullable());
    }

    #[test]
    fn build_candidate_batch_types_and_nulls() {
        let cols = vec![
            ("id".to_string(), SqlType::Int8),
            ("name".to_string(), SqlType::Text),
        ];
        let rows = vec![
            vec![Some(Value::Integer(1)), Some(Value::Text("a".into()))],
            vec![Some(Value::Integer(2)), None],
        ];
        let sims = vec![0.9, 0.5];
        let batch = build_candidate_batch(&cols, &rows, &sims).unwrap();
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 3);

        let ids = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(ids.value(0), 1);
        assert_eq!(ids.value(1), 2);
        let names = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(names.value(0), "a");
        assert!(names.is_null(1));
        let s = batch.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(s.value(0), 0.9);
    }

    #[test]
    fn unsupported_candidate_column_type_errors() {
        let cols = vec![(
            "tags".to_string(),
            SqlType::Array(Box::new(SqlType::Int4)),
        )];
        let rows = vec![vec![None]];
        assert!(build_candidate_batch(&cols, &rows, &[0.5]).is_err());
    }
}
