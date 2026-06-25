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
//! and **appends** the per-SQL-column chunks `[c0, c1, ... cN]`. Every
//! existing read, compaction, and recovery path reads columns 0–2 exactly
//! as before, so they are untouched and stay correct; the analytical scan
//! path (HTAP task 7) reads columns `3..3+N` as Arrow with no per-row
//! string parse. The `value` blob remains the canonical row for point-read
//! reconstruction (the PAX philosophy from the paper §3.1: one block serves
//! both a point read and a column scan).
//!
//! If `split` returns `None` for a row (malformed value), the block is
//! written in legacy form (no appended columns) so no data is ever lost or
//! silently mis-shaped — the analytical path then falls back to the
//! decode-on-scan bridge for that block (HTAP task 8).

use std::sync::Arc;

use galaxdb_common::ColumnType;

/// Splits a stored row value into its per-column physical byte encodings,
/// in declared column order, for columnar PAX flush.
///
/// Implemented by the SQL layer (which owns the row codec and the type
/// system) and registered on the engine per table-key-prefix. Each returned
/// byte vector is the physical encoding of one column's value for one row,
/// matching the corresponding [`ColumnType`] in [`column_types`]:
/// fixed-width types use little-endian bytes; variable-width types
/// (`Text`/`Blob`/`Json`) use their raw bytes.
///
/// [`column_types`]: RowColumnSplitter::column_types
pub trait RowColumnSplitter: Send + Sync {
    /// The physical column types, in order, that [`split`] produces one
    /// byte vector for. Used to type the PAX columns and pick codecs.
    ///
    /// [`split`]: RowColumnSplitter::split
    fn column_types(&self) -> Vec<ColumnType>;

    /// Split one row's stored value into one byte vector per column, in the
    /// same order as [`column_types`]. Returns `None` if the value cannot be
    /// split (e.g. a malformed legacy row), in which case the flush writes
    /// the row without the appended columns — never a panic, never a
    /// silently wrong column set.
    ///
    /// [`column_types`]: RowColumnSplitter::column_types
    fn split(&self, value: &[u8]) -> Option<Vec<Vec<u8>>>;
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
/// any appended per-SQL-column chunks: `key` (0), `value` (1), `ts` (2).
/// Appended columnar data, when present, starts at index
/// [`FIRST_DATA_COLUMN`].
pub const FIRST_DATA_COLUMN: usize = 3;

use crate::pax::{CodecId, ColumnData};

/// Build the appended per-SQL-column PAX columns (and their codecs) for one
/// block of rows that all belong to the same columnar table, given each
/// row's stored value (`None` for a tombstone).
///
/// Shared by the flush pipeline and the compaction output builder so both
/// produce identical columnar layouts. Returns `None` if the splitter has
/// no columns or any row fails to split, so the caller falls back to the
/// legacy three-column block (no data loss, never a panic).
pub fn columnar_data_columns(
    values: &[Option<Vec<u8>>],
    splitter: &dyn RowColumnSplitter,
) -> Option<(Vec<ColumnData>, Vec<CodecId>)> {
    let col_types = splitter.column_types();
    let n = col_types.len();
    if n == 0 {
        return None;
    }

    let mut data_cols: Vec<Vec<Vec<u8>>> = vec![Vec::with_capacity(values.len()); n];
    for v in values {
        let parts = match v {
            Some(b) => splitter.split(b)?,
            None => vec![Vec::new(); n], // tombstone: empty, row-aligned cell
        };
        if parts.len() != n {
            return None;
        }
        for (i, part) in parts.into_iter().enumerate() {
            data_cols[i].push(part);
        }
    }

    let mut codecs = Vec::with_capacity(n);
    let mut cols = Vec::with_capacity(n);
    for (ct, vals) in col_types.into_iter().zip(data_cols) {
        codecs.push(CodecId::for_column_type(&ct));
        cols.push(ColumnData { col_type: ct, values: vals });
    }
    Some((cols, codecs))
}
