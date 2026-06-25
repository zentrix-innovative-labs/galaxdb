//! Core type aliases and enums shared across all GalaxDB crates.

use serde::{Deserialize, Serialize};

/// Unique identifier for a table in the catalog.
pub type TableId = u64;

/// Unique identifier for a PAX block on disk.
pub type BlockId = u64;

/// Unique identifier for a row within a table.
pub type RowId = u64;

/// Logical timestamp used for MVCC versioning and commit ordering.
pub type Timestamp = u64;

/// How a SQL table's rows are physically laid out in PAX storage
/// (HTAP query engine, ADR-0002).
///
/// This is per-table catalog metadata. It selects the write/scan path the
/// storage engine uses for the table; it does not change the logical schema
/// or query results (see HTAP Property 3: the two modes return identical
/// results, differing only in performance).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum StorageMode {
    /// Each row is one opaque `col=v|...` UTF-8 blob keyed by primary key
    /// (the format every table used before the columnar path). Scanned via
    /// the decode-on-scan bridge. This is the default until the columnar
    /// write path (HTAP task 5) lands, and remains the format of any table
    /// created by an earlier build.
    #[default]
    Legacy,
    /// One typed PAX column per SQL column, so analytical scans read Arrow
    /// directly with no per-row string parse and predicates push down to
    /// per-column zone maps (HTAP tasks 5–7). OLTP point reads still resolve
    /// a single row via the ART.
    Columnar,
}

/// Describes the data type of a column in a GalaxDB table.
///
/// Covers standard integer, float, text, and binary types, plus an
/// `Embedding` variant that carries the vector dimensionality.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ColumnType {
    /// Signed 8-bit integer.
    Int8,
    /// Signed 16-bit integer.
    Int16,
    /// Signed 32-bit integer.
    Int32,
    /// Signed 64-bit integer.
    Int64,
    /// Unsigned 8-bit integer.
    UInt8,
    /// Unsigned 16-bit integer.
    UInt16,
    /// Unsigned 32-bit integer.
    UInt32,
    /// Unsigned 64-bit integer.
    UInt64,
    /// 32-bit IEEE 754 floating point.
    Float32,
    /// 64-bit IEEE 754 floating point.
    Float64,
    /// Variable-length UTF-8 text.
    Text,
    /// Variable-length binary data.
    Blob,
    /// JSON document stored as text.
    Json,
    /// Boolean value.
    Boolean,
    /// Dense embedding vector with the given number of dimensions.
    Embedding(u32),
}

impl ColumnType {
    /// Returns `true` if this column type has a fixed byte width.
    pub fn is_fixed_width(&self) -> bool {
        matches!(
            self,
            ColumnType::Int8
                | ColumnType::Int16
                | ColumnType::Int32
                | ColumnType::Int64
                | ColumnType::UInt8
                | ColumnType::UInt16
                | ColumnType::UInt32
                | ColumnType::UInt64
                | ColumnType::Float32
                | ColumnType::Float64
                | ColumnType::Boolean
        )
    }

    /// Returns `true` if this column type is variable-width (Text, Blob, Json).
    pub fn is_variable_width(&self) -> bool {
        matches!(
            self,
            ColumnType::Text | ColumnType::Blob | ColumnType::Json
        )
    }

    /// Returns `true` if this column type is an embedding vector.
    pub fn is_embedding(&self) -> bool {
        matches!(self, ColumnType::Embedding(_))
    }

    /// Returns the fixed byte size for fixed-width types, or `None` for
    /// variable-width and embedding types.
    pub fn byte_size(&self) -> Option<usize> {
        match self {
            ColumnType::Int8 | ColumnType::UInt8 | ColumnType::Boolean => Some(1),
            ColumnType::Int16 | ColumnType::UInt16 => Some(2),
            ColumnType::Int32 | ColumnType::UInt32 | ColumnType::Float32 => Some(4),
            ColumnType::Int64 | ColumnType::UInt64 | ColumnType::Float64 => Some(8),
            ColumnType::Text | ColumnType::Blob | ColumnType::Json => None,
            ColumnType::Embedding(dims) => Some(*dims as usize * 4), // f32 per dimension
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_width_types_report_correct_sizes() {
        assert_eq!(ColumnType::Int8.byte_size(), Some(1));
        assert_eq!(ColumnType::Int16.byte_size(), Some(2));
        assert_eq!(ColumnType::Int32.byte_size(), Some(4));
        assert_eq!(ColumnType::Int64.byte_size(), Some(8));
        assert_eq!(ColumnType::UInt8.byte_size(), Some(1));
        assert_eq!(ColumnType::UInt16.byte_size(), Some(2));
        assert_eq!(ColumnType::UInt32.byte_size(), Some(4));
        assert_eq!(ColumnType::UInt64.byte_size(), Some(8));
        assert_eq!(ColumnType::Float32.byte_size(), Some(4));
        assert_eq!(ColumnType::Float64.byte_size(), Some(8));
        assert_eq!(ColumnType::Boolean.byte_size(), Some(1));
    }

    #[test]
    fn variable_width_types_return_none_for_size() {
        assert_eq!(ColumnType::Text.byte_size(), None);
        assert_eq!(ColumnType::Blob.byte_size(), None);
        assert_eq!(ColumnType::Json.byte_size(), None);
    }

    #[test]
    fn embedding_type_reports_correct_size() {
        assert_eq!(ColumnType::Embedding(128).byte_size(), Some(512));
        assert_eq!(ColumnType::Embedding(768).byte_size(), Some(3072));
    }

    #[test]
    fn type_classification_is_correct() {
        assert!(ColumnType::Int32.is_fixed_width());
        assert!(!ColumnType::Int32.is_variable_width());
        assert!(!ColumnType::Int32.is_embedding());

        assert!(!ColumnType::Text.is_fixed_width());
        assert!(ColumnType::Text.is_variable_width());
        assert!(!ColumnType::Text.is_embedding());

        assert!(!ColumnType::Embedding(128).is_fixed_width());
        assert!(!ColumnType::Embedding(128).is_variable_width());
        assert!(ColumnType::Embedding(128).is_embedding());
    }
}
