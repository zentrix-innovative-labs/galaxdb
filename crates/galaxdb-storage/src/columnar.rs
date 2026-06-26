//! Schema-aware columnar storage path (HTAP query engine, ADR-0002).
//!
//! The storage engine is schema-agnostic: it stores `(key, value, ts)` rows
//! and knows nothing about SQL columns. To gain real OLAP — one typed PAX
//! column per SQL column, with per-column codecs and zone maps — the engine
//! needs to lay a row's value out by column at flush time. It does this
//! **without** learning the SQL type system, by holding a per-table
//! [`RowColumnSplitter`] the SQL layer registers. Storage only calls
//! `split` and drops the resulting bytes into typed PAX columns.
//!
//! # Additive, backward-compatible layout
//!
//! A columnar block keeps the existing `[key, value, ts]` columns unchanged
//! and **appends**, for each SQL column, a typed data column immediately
//! followed by a validity companion column. Every existing read,
//! compaction, and recovery path reads columns 0–2 exactly as before, so
//! they are untouched and stay correct; the analytical scan path (HTAP task
//! 7) reads the appended columns as Arrow with no per-row string parse. The
//! `value` blob remains the canonical row for point-read reconstruction
//! (the PAX philosophy from the paper §3.1: one block serves both a point
//! read and a column scan).
//!
//! ## NULLs (Arrow semantics)
//!
//! GalaxDB follows Apache Arrow's null model: a value is either present or
//! NULL, tracked by a per-column validity signal where `1 = valid`. Arrow
//! keeps a packed bitmap; Parquet keeps RLE-encoded definition levels. PAX
//! columns are **row-aligned** (each column has exactly `row_count`
//! entries), so a single packed bitmap cannot be one column. GalaxDB stores
//! validity as a row-aligned **1-byte-per-row companion column** (`1` =
//! valid, `0` = NULL) compressed with Zstd — which crushes the all-valid
//! common case to a few bytes (Parquet-style savings) while preserving
//! Arrow's exact semantics. The scan path packs these bytes into an Arrow
//! `NullBuffer`. A NULL slot in the data column carries a width-exact zero
//! placeholder so fixed-width codecs stay consistent; the validity column,
//! not the placeholder, is the source of truth for nullness.
//!
//! If `split` returns `None` for a row (malformed value), the block is
//! written in legacy form (no appended columns) so no data is ever lost or
//! silently mis-shaped — the analytical path then falls back to the
//! decode-on-scan bridge for that block (HTAP task 8).

use std::sync::Arc;

use galaxdb_common::ColumnType;

/// Splits a stored row value into its per-column cells, in declared column
/// order, for columnar PAX flush.
///
/// Implemented by the SQL layer (which owns the row codec and the type
/// system) and registered on the engine per table-key-prefix. Each element
/// of the returned vector is one column's cell for one row: `Some(bytes)`
/// for a present value (physical encoding matching the corresponding
/// [`ColumnType`] in [`column_types`] — little-endian for fixed-width types,
/// raw bytes for variable-width), or `None` for SQL `NULL`.
///
/// [`column_types`]: RowColumnSplitter::column_types
pub trait RowColumnSplitter: Send + Sync {
    /// The physical column types, in order, that [`split`] produces one cell
    /// for. Used to type the PAX columns, pick codecs, and size NULL
    /// placeholders.
    ///
    /// [`split`]: RowColumnSplitter::split
    fn column_types(&self) -> Vec<ColumnType>;

    /// Split one row's stored value into one cell per column, in the same
    /// order as [`column_types`]: `Some(bytes)` for a present value, `None`
    /// for SQL NULL. Returns `None` (the whole result) if the value cannot
    /// be split (e.g. a malformed legacy row), in which case the flush
    /// writes the row without the appended columns — never a panic, never a
    /// silently wrong column set.
    ///
    /// [`column_types`]: RowColumnSplitter::column_types
    fn split(&self, value: &[u8]) -> Option<Vec<Option<Vec<u8>>>>;
}

/// A per-table columnar registration held by the engine: the table's
/// primary-key prefix (`"table:"` bytes) and the splitter for its rows.
#[derive(Clone)]
pub struct ColumnarRegistration {
    /// Key prefix shared by every row of the table (e.g. `b"users:"`).
    pub prefix: Vec<u8>,
    /// Splitter that turns a row value into per-column bytes.
    pub splitter: Arc<dyn RowColumnSplitter>,
}

impl ColumnarRegistration {
    /// Does `key` belong to this registration's table?
    pub fn matches(&self, key: &[u8]) -> bool {
        key.starts_with(&self.prefix)
    }
}

/// Find the registration whose prefix matches `key`, if any. Used by the
/// flush pipeline to decide whether a block is columnar.
pub fn registration_for<'a>(
    registrations: &'a [ColumnarRegistration],
    key: &[u8],
) -> Option<&'a ColumnarRegistration> {
    registrations.iter().find(|r| r.matches(key))
}

/// The number of fixed leading PAX columns every SST block carries before
/// any appended per-SQL-column columns: `key` (0), `value` (1), `ts` (2).
/// Appended columnar data starts at index [`FIRST_DATA_COLUMN`].
pub const FIRST_DATA_COLUMN: usize = 3;

/// Appended columns come in (data, validity) pairs, one pair per SQL column,
/// so SQL column `k` (0-based) has its typed data column at
/// [`data_column_index`]`(k)` and its validity companion at
/// [`validity_column_index`]`(k)`.
pub const DATA_VALIDITY_STRIDE: usize = 2;

/// PAX column index of SQL column `k`'s typed data column.
pub fn data_column_index(k: usize) -> usize {
    FIRST_DATA_COLUMN + DATA_VALIDITY_STRIDE * k
}

/// PAX column index of SQL column `k`'s validity companion column.
pub fn validity_column_index(k: usize) -> usize {
    data_column_index(k) + 1
}

/// Interpret a validity companion cell: `1` (the first byte) means the
/// value is present, anything else (incl. empty) means SQL NULL.
pub fn is_valid_marker(cell: &[u8]) -> bool {
    cell.first().copied() == Some(VALID)
}

/// A pushed-down predicate for a columnar scan: `column <op> value`, where
/// `column` indexes the table's SQL columns (declaration order) and `value`
/// is the physical byte encoding of the comparison constant (same encoding
/// the column stores). Used for zone-map block pruning during the scan.
#[derive(Debug, Clone)]
pub struct ColumnPredicate {
    /// Index into the table's SQL columns.
    pub column: usize,
    /// Comparison operator.
    pub op: crate::pax::PruneOp,
    /// Physical bytes of the comparison constant.
    pub value: Vec<u8>,
}

/// A column-major batch produced by a columnar scan (HTAP task 7): one
/// entry per projected SQL column, each a row-aligned vector of physical
/// value bytes (`Some`) or SQL NULL (`None`). All columns have the same
/// length, `num_rows`. The query layer turns this into an Arrow
/// `RecordBatch` with no per-row string parse.
#[derive(Debug, Clone)]
pub struct ColumnarBatch {
    /// Number of rows (length of every column vector).
    pub num_rows: usize,
    /// `(physical type, per-row cells)` per projected column, in projection
    /// order.
    pub columns: Vec<(ColumnType, Vec<Option<Vec<u8>>>)>,
}

use crate::pax::{CodecId, ColumnData};

/// The single-byte validity markers stored per row in a validity companion
/// column: `1` = value present, `0` = SQL NULL (Arrow semantics).
const VALID: u8 = 1;
const NULL: u8 = 0;

/// Width-exact placeholder bytes stored in a NULL slot of a typed data
/// column so fixed-width codecs stay width-consistent. The validity column,
/// not these bytes, is the source of truth for nullness.
fn placeholder_bytes(ct: &ColumnType) -> Vec<u8> {
    match ct.byte_size() {
        Some(n) => vec![0u8; n], // fixed-width / embedding: zeroed
        None => Vec::new(),      // variable-width (Text/Blob/Json): empty
    }
}

/// Build the appended per-SQL-column PAX columns (and their codecs) for one
/// block of rows that all belong to the same columnar table, given each
/// row's stored value (`None` for a tombstone).
///
/// For each SQL column it emits two row-aligned columns: the typed data
/// column (NULL slots carry width-exact placeholders) immediately followed
/// by a 1-byte-per-row validity column (`1` = valid, `0` = NULL), Zstd-
/// compressed. Shared by the flush pipeline and the compaction output
/// builder so both produce identical columnar layouts. Returns `None` if
/// the splitter has no columns or any row fails to split, so the caller
/// falls back to the legacy three-column block (no data loss, never a
/// panic).
pub fn columnar_data_columns(
    values: &[Option<Vec<u8>>],
    splitter: &dyn RowColumnSplitter,
) -> Option<(Vec<ColumnData>, Vec<CodecId>)> {
    let col_types = splitter.column_types();
    let n = col_types.len();
    if n == 0 {
        return None;
    }

    // Per SQL column: the typed value bytes and the per-row validity marker.
    let mut data_cols: Vec<Vec<Vec<u8>>> = vec![Vec::with_capacity(values.len()); n];
    let mut valid_cols: Vec<Vec<Vec<u8>>> = vec![Vec::with_capacity(values.len()); n];

    for v in values {
        match v {
            Some(b) => {
                let parts = splitter.split(b)?;
                if parts.len() != n {
                    return None;
                }
                for (i, cell) in parts.into_iter().enumerate() {
                    match cell {
                        Some(bytes) => {
                            data_cols[i].push(bytes);
                            valid_cols[i].push(vec![VALID]);
                        }
                        None => {
                            data_cols[i].push(placeholder_bytes(&col_types[i]));
                            valid_cols[i].push(vec![NULL]);
                        }
                    }
                }
            }
            None => {
                // Tombstone: every column is NULL/placeholder, row-aligned.
                for i in 0..n {
                    data_cols[i].push(placeholder_bytes(&col_types[i]));
                    valid_cols[i].push(vec![NULL]);
                }
            }
        }
    }

    let mut cols = Vec::with_capacity(n * DATA_VALIDITY_STRIDE);
    let mut codecs = Vec::with_capacity(n * DATA_VALIDITY_STRIDE);
    for (i, ct) in col_types.into_iter().enumerate() {
        codecs.push(CodecId::for_column_type(&ct));
        cols.push(ColumnData { col_type: ct, values: std::mem::take(&mut data_cols[i]) });
        // Validity companion: Blob, Zstd-compressed (all-valid → tiny).
        codecs.push(CodecId::Zstd);
        cols.push(ColumnData {
            col_type: ColumnType::Blob,
            values: std::mem::take(&mut valid_cols[i]),
        });
    }
    Some((cols, codecs))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TwoColSplitter;
    impl RowColumnSplitter for TwoColSplitter {
        fn column_types(&self) -> Vec<ColumnType> {
            vec![ColumnType::Int64, ColumnType::Text]
        }
        // value layout: first 8 bytes present-int (or marker 0xFF.. for null),
        // remainder is the name; an empty name encodes a NULL name.
        fn split(&self, value: &[u8]) -> Option<Vec<Option<Vec<u8>>>> {
            if value.len() < 8 {
                return None;
            }
            let id_bytes = value[0..8].to_vec();
            let id = if id_bytes == vec![0xFFu8; 8] {
                None
            } else {
                Some(id_bytes)
            };
            let name = &value[8..];
            let name_cell = if name.is_empty() {
                None
            } else {
                Some(name.to_vec())
            };
            Some(vec![id, name_cell])
        }
    }

    #[test]
    fn emits_data_and_validity_pairs() {
        let mut v1 = 7i64.to_le_bytes().to_vec();
        v1.extend_from_slice(b"alice");
        let values = vec![Some(v1)];
        let (cols, codecs) = columnar_data_columns(&values, &TwoColSplitter).unwrap();
        // 2 SQL columns → 4 appended columns (data, validity) × 2.
        assert_eq!(cols.len(), 4);
        assert_eq!(codecs.len(), 4);
        assert_eq!(cols[0].col_type, ColumnType::Int64);
        assert_eq!(cols[1].col_type, ColumnType::Blob); // validity
        assert_eq!(cols[2].col_type, ColumnType::Text);
        assert_eq!(cols[3].col_type, ColumnType::Blob); // validity
        assert_eq!(cols[1].values[0], vec![VALID]);
        assert_eq!(cols[3].values[0], vec![VALID]);
    }

    #[test]
    fn null_cell_gets_placeholder_and_zero_validity() {
        // id present, name NULL (empty).
        let v = 9i64.to_le_bytes().to_vec(); // no name → NULL name
        let values = vec![Some(v)];
        let (cols, _) = columnar_data_columns(&values, &TwoColSplitter).unwrap();
        // name data column: placeholder empty, validity 0.
        assert_eq!(cols[2].values[0], Vec::<u8>::new());
        assert_eq!(cols[3].values[0], vec![NULL]);
        // id present.
        assert_eq!(cols[1].values[0], vec![VALID]);

        // Now a NULL id (0xFF marker), present name.
        let mut v2 = vec![0xFFu8; 8];
        v2.extend_from_slice(b"bob");
        let (cols2, _) = columnar_data_columns(&[Some(v2)], &TwoColSplitter).unwrap();
        // id data column: width-exact 8-byte zero placeholder, validity 0.
        assert_eq!(cols2[0].values[0], vec![0u8; 8]);
        assert_eq!(cols2[1].values[0], vec![NULL]);
        assert_eq!(cols2[3].values[0], vec![VALID]);
    }

    #[test]
    fn tombstone_row_is_all_null() {
        let values = vec![None];
        let (cols, _) = columnar_data_columns(&values, &TwoColSplitter).unwrap();
        assert_eq!(cols[1].values[0], vec![NULL]);
        assert_eq!(cols[3].values[0], vec![NULL]);
        assert_eq!(cols[0].values[0], vec![0u8; 8]); // int placeholder
    }

    #[test]
    fn index_helpers() {
        assert_eq!(data_column_index(0), 3);
        assert_eq!(validity_column_index(0), 4);
        assert_eq!(data_column_index(1), 5);
        assert_eq!(validity_column_index(1), 6);
    }
}
