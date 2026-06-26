//! Convert a storage [`ColumnarBatch`] into an Arrow [`RecordBatch`] (HTAP
//! task 7 — the "emit Arrow" half of the columnar read path).
//!
//! The storage layer returns physical column bytes (little-endian for
//! fixed-width types, raw bytes for variable-width) plus per-cell nullness.
//! This module builds the Arrow array of each column's **logical** type
//! (`SqlType::to_arrow` via `crate::schema`) directly from those bytes, with
//! Arrow's native null buffer — so DataFusion consumes the data with zero
//! per-row string parse.

use std::sync::Arc;

use arrow::array::{
    ArrayRef, BinaryArray, BooleanArray, Date32Array, FixedSizeBinaryArray, Float32Array,
    Float64Array, Int16Array, Int32Array, Int64Array, RecordBatch, StringArray,
    TimestampMicrosecondArray,
};
use arrow::datatypes::{Field, Schema, SchemaRef};

use galaxdb_common::{GalaxError, GalaxResult};
use galaxdb_sql::SqlType;
use galaxdb_storage::columnar::ColumnarBatch;

use crate::schema::sql_type_to_arrow;

/// Build an Arrow schema (all fields nullable) from the projected logical
/// column list.
pub fn arrow_schema(fields: &[(String, SqlType)]) -> SchemaRef {
    let arrow_fields: Vec<Field> = fields
        .iter()
        .map(|(name, ty)| Field::new(name, sql_type_to_arrow(ty), true))
        .collect();
    Arc::new(Schema::new(arrow_fields))
}

/// Convert a [`ColumnarBatch`] to a [`RecordBatch`]. `fields` is the
/// projected column list (name + logical type) in the same order as
/// `batch.columns`.
pub fn columnar_batch_to_record_batch(
    batch: &ColumnarBatch,
    fields: &[(String, SqlType)],
) -> GalaxResult<RecordBatch> {
    if batch.columns.len() != fields.len() {
        return Err(GalaxError::Internal(format!(
            "columnar batch has {} columns but {} fields were given",
            batch.columns.len(),
            fields.len()
        )));
    }
    let schema = arrow_schema(fields);
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(fields.len());
    for ((_, ty), (_, cells)) in fields.iter().zip(batch.columns.iter()) {
        arrays.push(build_array(ty, cells)?);
    }
    RecordBatch::try_new(schema, arrays)
        .map_err(|e| GalaxError::Internal(format!("arrow record batch: {e}")))
}

/// Decode a fixed-width little-endian integer cell.
fn le_i64(b: &[u8]) -> GalaxResult<i64> {
    b.try_into()
        .map(i64::from_le_bytes)
        .map_err(|_| GalaxError::Internal("columnar: bad i64 cell width".into()))
}
fn le_i32(b: &[u8]) -> GalaxResult<i32> {
    b.try_into()
        .map(i32::from_le_bytes)
        .map_err(|_| GalaxError::Internal("columnar: bad i32 cell width".into()))
}
fn le_i16(b: &[u8]) -> GalaxResult<i16> {
    b.try_into()
        .map(i16::from_le_bytes)
        .map_err(|_| GalaxError::Internal("columnar: bad i16 cell width".into()))
}
fn le_f32(b: &[u8]) -> GalaxResult<f32> {
    b.try_into()
        .map(f32::from_le_bytes)
        .map_err(|_| GalaxError::Internal("columnar: bad f32 cell width".into()))
}
fn le_f64(b: &[u8]) -> GalaxResult<f64> {
    b.try_into()
        .map(f64::from_le_bytes)
        .map_err(|_| GalaxError::Internal("columnar: bad f64 cell width".into()))
}

/// Build one Arrow array of `ty`'s logical type from physical cell bytes.
fn build_array(ty: &SqlType, cells: &[Option<Vec<u8>>]) -> GalaxResult<ArrayRef> {
    // Helper to map each present cell through `f`, propagating nulls.
    macro_rules! map_cells {
        ($f:expr) => {{
            cells
                .iter()
                .map(|c| c.as_deref().map($f).transpose())
                .collect::<GalaxResult<Vec<Option<_>>>>()?
        }};
    }

    let arr: ArrayRef = match ty {
        SqlType::Int2 => Arc::new(Int16Array::from(map_cells!(le_i16))),
        SqlType::Int4 => Arc::new(Int32Array::from(map_cells!(le_i32))),
        SqlType::Int8 => Arc::new(Int64Array::from(map_cells!(le_i64))),
        SqlType::Float4 => Arc::new(Float32Array::from(map_cells!(le_f32))),
        SqlType::Float8 => Arc::new(Float64Array::from(map_cells!(le_f64))),
        SqlType::Bool => {
            let v: Vec<Option<bool>> = cells
                .iter()
                .map(|c| c.as_deref().map(|b| b.first().copied() == Some(1)))
                .collect();
            Arc::new(BooleanArray::from(v))
        }
        SqlType::Text | SqlType::Varchar(_) | SqlType::Json | SqlType::Jsonb => {
            let v: Vec<Option<String>> = cells
                .iter()
                .map(|c| c.as_deref().map(|b| String::from_utf8_lossy(b).into_owned()))
                .collect();
            Arc::new(StringArray::from(v))
        }
        SqlType::Bytea => {
            let v: Vec<Option<&[u8]>> = cells.iter().map(|c| c.as_deref()).collect();
            Arc::new(BinaryArray::from(v))
        }
        SqlType::Date => Arc::new(Date32Array::from(map_cells!(le_i32))),
        SqlType::Timestamp => Arc::new(TimestampMicrosecondArray::from(map_cells!(le_i64))),
        SqlType::TimestampTz => Arc::new(
            TimestampMicrosecondArray::from(map_cells!(le_i64)).with_timezone("UTC"),
        ),
        SqlType::Uuid => {
            let iter = cells.iter().map(|c| c.as_deref());
            FixedSizeBinaryArray::try_from_sparse_iter_with_size(iter, 16)
                .map(|a| Arc::new(a) as ArrayRef)
                .map_err(|e| GalaxError::Internal(format!("uuid arrow array: {e}")))?
        }
        // Decimal and array Arrow building are deferred (physical Text/Blob);
        // surfaced explicitly rather than producing a mis-typed array.
        SqlType::Numeric { .. } => {
            return Err(GalaxError::FeatureNotSupported(
                "NUMERIC columns in the Arrow scan path are not yet supported".into(),
            ))
        }
        SqlType::Array(_) => {
            return Err(GalaxError::FeatureNotSupported(
                "array columns in the Arrow scan path are not yet supported".into(),
            ))
        }
    };
    Ok(arr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Array;
    use galaxdb_common::ColumnType;

    fn batch(columns: Vec<(ColumnType, Vec<Option<Vec<u8>>>)>) -> ColumnarBatch {
        let num_rows = columns.first().map(|(_, c)| c.len()).unwrap_or(0);
        ColumnarBatch { num_rows, columns }
    }

    #[test]
    fn builds_int_text_with_nulls() {
        let b = batch(vec![
            (
                ColumnType::Int64,
                vec![Some(1i64.to_le_bytes().to_vec()), None, Some(3i64.to_le_bytes().to_vec())],
            ),
            (
                ColumnType::Text,
                vec![Some(b"a".to_vec()), Some(b"b".to_vec()), None],
            ),
        ]);
        let fields = vec![("id".to_string(), SqlType::Int8), ("name".to_string(), SqlType::Text)];
        let rb = columnar_batch_to_record_batch(&b, &fields).unwrap();
        assert_eq!(rb.num_rows(), 3);

        let ids = rb.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(ids.value(0), 1);
        assert!(ids.is_null(1));
        assert_eq!(ids.value(2), 3);

        let names = rb.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(names.value(0), "a");
        assert!(names.is_null(2));
    }

    #[test]
    fn builds_date_and_timestamp() {
        let b = batch(vec![
            (ColumnType::Int32, vec![Some(10957i32.to_le_bytes().to_vec())]),
            (ColumnType::Int64, vec![Some(1_000_000i64.to_le_bytes().to_vec())]),
        ]);
        let fields = vec![
            ("d".to_string(), SqlType::Date),
            ("t".to_string(), SqlType::Timestamp),
        ];
        let rb = columnar_batch_to_record_batch(&b, &fields).unwrap();
        let d = rb.column(0).as_any().downcast_ref::<Date32Array>().unwrap();
        assert_eq!(d.value(0), 10957);
        let t = rb
            .column(1)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap();
        assert_eq!(t.value(0), 1_000_000);
    }

    #[test]
    fn numeric_is_explicit_unsupported_not_wrong() {
        let b = batch(vec![(ColumnType::Text, vec![Some(b"1.5".to_vec())])]);
        let fields = vec![(
            "n".to_string(),
            SqlType::Numeric { precision: None, scale: None },
        )];
        let err = columnar_batch_to_record_batch(&b, &fields).unwrap_err();
        assert_eq!(err.sqlstate(), "0A000");
    }
}
