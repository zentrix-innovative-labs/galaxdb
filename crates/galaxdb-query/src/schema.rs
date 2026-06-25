//! Arrow type mapping for the query layer (HTAP design §3.3, the
//! `→ Arrow` half of the single-source-of-truth type system, kept in this
//! crate so `arrow` does not leak into `galaxdb-sql`).
//!
//! Two mappings:
//! - [`column_type_to_arrow`] — the **physical** [`ColumnType`] storage
//!   actually emits (what `ArrowSource::scan` produces today).
//! - [`sql_type_to_arrow`] — the **logical** [`SqlType`] the query layer
//!   should present to clients (e.g. a timestamp is a logical
//!   `Timestamp(µs)` even though it is physically an `Int64`). Bridging the
//!   two (casting Int64→Timestamp on scan) is the columnar scan path's job
//!   (HTAP task 7).

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};

use galaxdb_common::ColumnType;
use galaxdb_sql::SqlType;

/// Map a **physical** storage [`ColumnType`] to its Arrow `DataType`.
pub fn column_type_to_arrow(col: &ColumnType) -> DataType {
    match col {
        ColumnType::Int8 => DataType::Int8,
        ColumnType::Int16 => DataType::Int16,
        ColumnType::Int32 => DataType::Int32,
        ColumnType::Int64 => DataType::Int64,
        ColumnType::UInt8 => DataType::UInt8,
        ColumnType::UInt16 => DataType::UInt16,
        ColumnType::UInt32 => DataType::UInt32,
        ColumnType::UInt64 => DataType::UInt64,
        ColumnType::Float32 => DataType::Float32,
        ColumnType::Float64 => DataType::Float64,
        ColumnType::Text | ColumnType::Json => DataType::Utf8,
        ColumnType::Blob => DataType::Binary,
        ColumnType::Boolean => DataType::Boolean,
        ColumnType::Embedding(dim) => DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, false)),
            *dim as i32,
        ),
    }
}

/// Map a **logical** [`SqlType`] to the Arrow `DataType` the query layer
/// presents to clients. Temporal/uuid/numeric types get their true logical
/// Arrow types here (vs. their physical encodings).
pub fn sql_type_to_arrow(ty: &SqlType) -> DataType {
    match ty {
        SqlType::Int2 => DataType::Int16,
        SqlType::Int4 => DataType::Int32,
        SqlType::Int8 => DataType::Int64,
        SqlType::Float4 => DataType::Float32,
        SqlType::Float8 => DataType::Float64,
        SqlType::Bool => DataType::Boolean,
        SqlType::Text | SqlType::Varchar(_) => DataType::Utf8,
        // JSON is carried as UTF-8 (logical JSON), matching PG text transfer.
        SqlType::Json | SqlType::Jsonb => DataType::Utf8,
        SqlType::Bytea => DataType::Binary,
        SqlType::Uuid => DataType::FixedSizeBinary(16),
        SqlType::Date => DataType::Date32,
        SqlType::Timestamp => DataType::Timestamp(TimeUnit::Microsecond, None),
        SqlType::TimestampTz => {
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
        }
        SqlType::Numeric { precision, scale } => {
            // Arrow Decimal128: precision <= 38, scale fits in i8.
            let p = precision.unwrap_or(38).min(38);
            let s = scale.unwrap_or(0) as i8;
            DataType::Decimal128(p, s)
        }
        SqlType::Array(elem) => DataType::List(Arc::new(Field::new(
            "item",
            sql_type_to_arrow(elem),
            true,
        ))),
    }
}

/// Build an Arrow [`SchemaRef`] from a logical column list
/// `(name, SqlType, nullable)`.
pub fn build_schema(columns: &[(String, SqlType, bool)]) -> SchemaRef {
    let fields: Vec<Field> = columns
        .iter()
        .map(|(name, ty, nullable)| Field::new(name, sql_type_to_arrow(ty), *nullable))
        .collect();
    Arc::new(Schema::new(fields))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_column_types_map() {
        assert_eq!(column_type_to_arrow(&ColumnType::Int32), DataType::Int32);
        assert_eq!(column_type_to_arrow(&ColumnType::Int64), DataType::Int64);
        assert_eq!(column_type_to_arrow(&ColumnType::Float32), DataType::Float32);
        assert_eq!(column_type_to_arrow(&ColumnType::Text), DataType::Utf8);
        assert_eq!(column_type_to_arrow(&ColumnType::Json), DataType::Utf8);
        assert_eq!(column_type_to_arrow(&ColumnType::Blob), DataType::Binary);
        assert_eq!(column_type_to_arrow(&ColumnType::Boolean), DataType::Boolean);
        match column_type_to_arrow(&ColumnType::Embedding(128)) {
            DataType::FixedSizeList(field, dim) => {
                assert_eq!(dim, 128);
                assert_eq!(field.data_type(), &DataType::Float32);
            }
            other => panic!("expected FixedSizeList, got {other:?}"),
        }
    }

    #[test]
    fn logical_sql_types_map() {
        assert_eq!(sql_type_to_arrow(&SqlType::Int4), DataType::Int32);
        assert_eq!(sql_type_to_arrow(&SqlType::Float8), DataType::Float64);
        assert_eq!(sql_type_to_arrow(&SqlType::Bool), DataType::Boolean);
        assert_eq!(sql_type_to_arrow(&SqlType::Text), DataType::Utf8);
        assert_eq!(sql_type_to_arrow(&SqlType::Bytea), DataType::Binary);
        assert_eq!(sql_type_to_arrow(&SqlType::Date), DataType::Date32);
        assert_eq!(
            sql_type_to_arrow(&SqlType::Timestamp),
            DataType::Timestamp(TimeUnit::Microsecond, None)
        );
        assert_eq!(
            sql_type_to_arrow(&SqlType::Uuid),
            DataType::FixedSizeBinary(16)
        );
        assert_eq!(
            sql_type_to_arrow(&SqlType::Numeric { precision: Some(10), scale: Some(2) }),
            DataType::Decimal128(10, 2)
        );
    }

    #[test]
    fn timestamptz_carries_utc() {
        match sql_type_to_arrow(&SqlType::TimestampTz) {
            DataType::Timestamp(TimeUnit::Microsecond, Some(tz)) => assert_eq!(&*tz, "UTC"),
            other => panic!("expected tz timestamp, got {other:?}"),
        }
    }

    #[test]
    fn array_maps_to_list() {
        let ty = SqlType::Array(Box::new(SqlType::Int4));
        match sql_type_to_arrow(&ty) {
            DataType::List(field) => assert_eq!(field.data_type(), &DataType::Int32),
            other => panic!("expected List, got {other:?}"),
        }
    }

    #[test]
    fn build_schema_from_columns() {
        let cols = vec![
            ("id".to_string(), SqlType::Int8, false),
            ("name".to_string(), SqlType::Text, true),
            ("created".to_string(), SqlType::Timestamp, true),
        ];
        let schema = build_schema(&cols);
        assert_eq!(schema.fields().len(), 3);
        assert_eq!(schema.field(0).name(), "id");
        assert!(!schema.field(0).is_nullable());
        assert_eq!(schema.field(1).data_type(), &DataType::Utf8);
        assert!(schema.field(2).is_nullable());
    }
}
