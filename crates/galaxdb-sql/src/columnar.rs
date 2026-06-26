//! SQL-layer columnar splitter (HTAP query engine, ADR-0002).
//!
//! Implements [`galaxdb_storage::columnar::RowColumnSplitter`] over the
//! catalog schema so the storage engine can lay a columnar table's rows out
//! as one typed PAX column per SQL column. The storage engine stays
//! schema-agnostic; this is the one place that knows both the row codec and
//! the logical type system.
//!
//! # Type coercion happens here, not on INSERT
//!
//! Inserted literals are stored as syntactic `Value`s (a DATE/UUID/NUMERIC
//! column holds `Value::Text`, an INT holds `Value::Integer`). Rather than
//! change INSERT semantics globally, this splitter coerces each decoded
//! value to its column's [`SqlType`] physical encoding at flush time via
//! [`crate::types::parse_value`]. So the columnar columns are type-faithful
//! while the stored `value` blob and the point-read/wire paths are
//! untouched. If a value cannot be coerced to its column type (malformed
//! data), `split` returns `None` and the flush writes that block in legacy
//! form — no data loss, never a silently wrong column (engineering-
//! principles §2).

use galaxdb_common::ColumnType;
use galaxdb_storage::columnar::RowColumnSplitter;

use crate::executor::TableEntry;
use crate::planner::Value;
use crate::row_codec;
use crate::types::{self, SqlType};

/// A [`RowColumnSplitter`] built from a table's catalog schema.
pub struct CatalogRowSplitter {
    /// `(column_name, sql_type)` in declaration order.
    columns: Vec<(String, SqlType)>,
}

impl CatalogRowSplitter {
    /// Build a splitter from a catalog [`TableEntry`].
    ///
    /// Returns `None` if the table has no columns or if **any** column's
    /// declared type name is not recognized by [`SqlType::from_sql_name`].
    /// A partial column set would misalign the columnar layout, so an
    /// unrecognized type means the whole table stays legacy (correctness
    /// over partial columnarization).
    pub fn from_table_entry(entry: &TableEntry) -> Option<Self> {
        if entry.columns.is_empty() {
            return None;
        }
        let mut columns = Vec::with_capacity(entry.columns.len());
        for c in &entry.columns {
            let ty = SqlType::from_sql_name(&c.data_type).ok()?;
            columns.push((c.name.clone(), ty));
        }
        Some(Self { columns })
    }
}

impl RowColumnSplitter for CatalogRowSplitter {
    fn column_types(&self) -> Vec<ColumnType> {
        self.columns.iter().map(|(_, t)| t.to_column_type()).collect()
    }

    fn split(&self, value: &[u8]) -> Option<Vec<Option<Vec<u8>>>> {
        let decoded = row_codec::decode_row(value);
        let mut out = Vec::with_capacity(self.columns.len());
        for (name, ty) in &self.columns {
            let cell = decoded.iter().find(|(n, _)| n == name).map(|(_, v)| v);
            match cell {
                None | Some(Value::Null) => out.push(None), // missing / NULL
                Some(v) => out.push(Some(physical_bytes(v, ty)?)),
            }
        }
        Some(out)
    }
}

/// Encode a decoded value to its column's physical byte representation,
/// coercing through the logical [`SqlType`]. Returns `None` if the value
/// cannot be represented as the column's type (so the caller falls back to
/// a legacy block rather than writing a mis-typed column).
fn physical_bytes(v: &Value, ty: &SqlType) -> Option<Vec<u8>> {
    // Coerce via the canonical text form + the type system's parser, so a
    // DATE/UUID/TIMESTAMP/NUMERIC stored as text lands in its physical form.
    let text = row_codec::value_display(v);
    let coerced = types::parse_value(&text, ty).ok()?;
    encode_physical(&coerced, &ty.to_column_type())
}

/// Encode a value already coerced to `ct` into the little-endian / raw byte
/// form the columnar PAX column stores. Width-exact for fixed types so the
/// fixed-width codecs stay consistent.
fn encode_physical(v: &Value, ct: &ColumnType) -> Option<Vec<u8>> {
    Some(match (ct, v) {
        (ColumnType::Int16, Value::Integer(n)) => (*n as i16).to_le_bytes().to_vec(),
        (ColumnType::Int32, Value::Integer(n)) => (*n as i32).to_le_bytes().to_vec(),
        (ColumnType::Int64, Value::Integer(n)) => n.to_le_bytes().to_vec(),
        (ColumnType::Float32, Value::Float(f)) => (*f as f32).to_le_bytes().to_vec(),
        (ColumnType::Float64, Value::Float(f)) => f.to_le_bytes().to_vec(),
        (ColumnType::Boolean, Value::Bool(b)) => vec![*b as u8],
        (ColumnType::Text, Value::Text(s)) | (ColumnType::Json, Value::Text(s)) => {
            s.as_bytes().to_vec()
        }
        (ColumnType::Blob, Value::Blob(b)) => b.clone(),
        // The coerced value's variant did not match the expected physical
        // type — treat as un-columnarizable for this row.
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::{CatalogColumn, TableEntry};

    fn entry(cols: &[(&str, &str)]) -> TableEntry {
        TableEntry {
            name: "t".into(),
            columns: cols
                .iter()
                .map(|(n, ty)| CatalogColumn {
                    name: n.to_string(),
                    data_type: ty.to_string(),
                    nullable: true,
                    primary_key: false,
                    is_embedding_source: false,
                })
                .collect(),
            has_embedding: false,
            append_only: false,
            storage_mode: galaxdb_common::StorageMode::Columnar,
        }
    }

    #[test]
    fn column_types_map_to_physical() {
        let e = entry(&[("id", "BIGINT"), ("name", "TEXT"), ("created", "DATE")]);
        let s = CatalogRowSplitter::from_table_entry(&e).unwrap();
        assert_eq!(
            s.column_types(),
            vec![ColumnType::Int64, ColumnType::Text, ColumnType::Int32]
        );
    }

    #[test]
    fn unknown_type_disables_columnar() {
        let e = entry(&[("id", "BIGINT"), ("weird", "MONEYBAGS")]);
        assert!(CatalogRowSplitter::from_table_entry(&e).is_none());
    }

    #[test]
    fn split_coerces_typed_columns() {
        let e = entry(&[("id", "BIGINT"), ("name", "TEXT"), ("created", "DATE")]);
        let s = CatalogRowSplitter::from_table_entry(&e).unwrap();

        // Stored row as the executor would write it (DATE held as text).
        let row = vec![
            ("id".to_string(), Value::Integer(42)),
            ("name".to_string(), Value::Text("alice".into())),
            ("created".to_string(), Value::Text("2000-01-01".into())),
        ];
        let blob = row_codec::encode_row(&row);

        let cells = s.split(&blob).unwrap();
        assert_eq!(cells.len(), 3);
        // id → 8-byte LE 42
        assert_eq!(cells[0], Some(42i64.to_le_bytes().to_vec()));
        // name → raw utf8
        assert_eq!(cells[1], Some(b"alice".to_vec()));
        // created → 4-byte LE days since epoch (2000-01-01 = day 10957)
        assert_eq!(cells[2], Some(10957i32.to_le_bytes().to_vec()));
    }

    #[test]
    fn split_marks_null_and_missing_cells() {
        let e = entry(&[("id", "BIGINT"), ("name", "TEXT")]);
        let s = CatalogRowSplitter::from_table_entry(&e).unwrap();

        // name explicitly NULL; (a missing column would also be None).
        let row = vec![
            ("id".to_string(), Value::Integer(1)),
            ("name".to_string(), Value::Null),
        ];
        let blob = row_codec::encode_row(&row);
        let cells = s.split(&blob).unwrap();
        assert_eq!(cells[0], Some(1i64.to_le_bytes().to_vec()));
        assert_eq!(cells[1], None);
    }

    #[test]
    fn split_aborts_on_uncoercible_value() {
        // 'alice' cannot be an INT → split returns None (legacy fallback).
        let e = entry(&[("n", "INTEGER")]);
        let s = CatalogRowSplitter::from_table_entry(&e).unwrap();
        let row = vec![("n".to_string(), Value::Text("alice".into()))];
        let blob = row_codec::encode_row(&row);
        assert!(s.split(&blob).is_none());
    }
}
