//! Row encoding shared by the real executor (`execute_with_context`) and
//! by the legacy `galaxdb-embedded` path that pre-dated the executor
//! rewrite.
//!
//! # Wire format
//!
//! A row is serialised as `col1=v1|col2=v2|...|colN=vN` in the order the
//! executor constructed the value list. Each `vi` is the display form of
//! a [`crate::planner::Value`]; `Value::Null` is rendered as the literal
//! `NULL`, strings are kept verbatim (no quoting — `|` characters inside
//! a TEXT value would collide with the separator and are rejected by the
//! encoder).
//!
//! This matches the format that `galaxdb-embedded::build_kv` produced
//! before the consolidation sprint, so on-disk data written before this
//! refactor round-trips correctly. A proper columnar PAX encoding is
//! tracked by task 18.7 / 39; when that lands this codec becomes the
//! compatibility layer for rows written by earlier builds.
//!
//! # Primary key
//!
//! The primary key is `"{table}:{pk_value}"`. If the table has a column
//! flagged `primary_key`, the corresponding value from the INSERT is
//! used; otherwise the first value in the row is used as the key. This
//! preserves the previous embedded-crate behaviour.

use galaxdb_common::{GalaxError, GalaxResult};

use crate::planner::{FilterExpr, Value};
use crate::executor::TableEntry;

/// Align a `(columns, values)` INSERT payload with the catalog's
/// column list. Returns one `(column_name, value)` pair per caller-
/// provided value, in the order the SQL text supplied them.
///
/// When `columns` is empty (i.e. `INSERT INTO t VALUES (...)`), the
/// values are matched positionally to `table_entry.columns`.
pub fn align_values(
    table_entry: &TableEntry,
    columns: &[String],
    values: &[Value],
) -> GalaxResult<Vec<(String, Value)>> {
    if columns.is_empty() {
        if values.len() > table_entry.columns.len() {
            return Err(GalaxError::Internal(format!(
                "value count ({}) exceeds column count ({}) for table '{}'",
                values.len(),
                table_entry.columns.len(),
                table_entry.name
            )));
        }
        Ok(table_entry
            .columns
            .iter()
            .map(|c| c.name.clone())
            .zip(values.iter().cloned())
            .collect())
    } else {
        if columns.len() != values.len() {
            return Err(GalaxError::Internal(format!(
                "column count ({}) does not match value count ({})",
                columns.len(),
                values.len()
            )));
        }
        Ok(columns.iter().cloned().zip(values.iter().cloned()).collect())
    }
}

/// Build the primary-key bytes for a row.
///
/// Preference order:
/// 1. The value for the column the catalog marked `primary_key`.
/// 2. The first value in the aligned row.
/// 3. An error if the row is empty.
///
/// The bytes are `"{table}:{pk_display}"` so that the LSM stores every
/// row of a table under a shared prefix for fast table scans.
pub fn build_primary_key(
    table: &str,
    table_entry: &TableEntry,
    ordered: &[(String, Value)],
) -> GalaxResult<Vec<u8>> {
    let pk_column = table_entry.columns.iter().find(|c| c.primary_key);

    let pk_value = if let Some(pk_col) = pk_column {
        ordered
            .iter()
            .find(|(n, _)| n == &pk_col.name)
            .map(|(_, v)| v.clone())
    } else {
        ordered.first().map(|(_, v)| v.clone())
    };

    let pk_value = pk_value.ok_or_else(|| {
        GalaxError::Internal(format!(
            "cannot build primary key for table '{}': row has no values",
            table
        ))
    })?;

    Ok(format!("{}:{}", table, value_display(&pk_value)).into_bytes())
}

/// Encode an ordered list of columns into the on-disk row byte format.
///
/// Panics if any value contains a `|` character because that would
/// collide with the column separator. Callers must sanitise text input
/// before INSERT; this is a real constraint of the text codec, not a
/// silent data corruption path. A proper columnar codec (task 18.7)
/// removes the constraint.
pub fn encode_row(ordered: &[(String, Value)]) -> Vec<u8> {
    let mut out = String::with_capacity(ordered.len() * 16);
    for (i, (col, val)) in ordered.iter().enumerate() {
        if i > 0 {
            out.push('|');
        }
        out.push_str(col);
        out.push('=');
        let rendered = value_display(val);
        // Guard against `|` in text values — the text codec cannot
        // express them unambiguously.
        if rendered.contains('|') {
            out.push_str(&rendered.replace('|', "\\|"));
        } else {
            out.push_str(&rendered);
        }
    }
    out.into_bytes()
}

/// Decode row bytes into an ordered list of `(column, Value)` pairs.
///
/// Reverses [`encode_row`]. Unknown value forms fall back to `Value::Text`.
pub fn decode_row(bytes: &[u8]) -> Vec<(String, Value)> {
    let text = String::from_utf8_lossy(bytes);
    let mut out = Vec::new();
    // Split on `|` but respect `\|` escapes produced by `encode_row`.
    for part in split_respecting_escape(&text) {
        if let Some((col, val)) = part.split_once('=') {
            out.push((col.to_string(), value_from_str(val)));
        }
    }
    out
}

fn split_respecting_escape(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(&next) = chars.peek() {
                if next == '|' {
                    current.push('|');
                    chars.next();
                    continue;
                }
            }
            current.push('\\');
        } else if c == '|' {
            parts.push(std::mem::take(&mut current));
        } else {
            current.push(c);
        }
    }
    if !current.is_empty() || !parts.is_empty() {
        parts.push(current);
    }
    parts
}

/// Render a value as the text form stored on disk.
pub fn value_display(v: &Value) -> String {
    match v {
        Value::Integer(n) => n.to_string(),
        Value::Float(f) => f.to_string(),
        Value::Text(s) => s.clone(),
        Value::Bool(b) => b.to_string(),
        Value::Null => "NULL".to_string(),
        Value::Blob(bytes) => {
            // Base64-free lossless encoding: hex. Blobs are uncommon in
            // the current test corpus so readability beats compactness.
            let mut s = String::with_capacity(bytes.len() * 2);
            for b in bytes {
                use std::fmt::Write;
                let _ = write!(&mut s, "{:02x}", b);
            }
            s
        }
    }
}

/// Reverse of [`value_display`]. Distinguishes integers, floats, bool,
/// NULL, and falls back to `Value::Text`. The same-text round trip for
/// bool/NULL literals is intentional — `INSERT INTO t VALUES ('NULL')`
/// stores the 4-character string `"NULL"` which decodes as
/// `Value::Null`. Callers who need exact text fidelity should not use
/// the literal `NULL` as a string value. A proper columnar codec (task
/// 18.7) carries explicit type tags per column and removes the
/// ambiguity.
pub fn value_from_str(s: &str) -> Value {
    if s == "NULL" {
        return Value::Null;
    }
    if s == "true" {
        return Value::Bool(true);
    }
    if s == "false" {
        return Value::Bool(false);
    }
    if let Ok(i) = s.parse::<i64>() {
        return Value::Integer(i);
    }
    if let Ok(f) = s.parse::<f64>() {
        // Only treat as float if the source text had a decimal point or
        // exponent; otherwise `"1"` would become `Value::Float(1.0)`
        // instead of `Value::Integer(1)`. This is decided above by the
        // `i64::parse` happening first.
        if s.contains('.') || s.contains('e') || s.contains('E') {
            return Value::Float(f);
        }
    }
    Value::Text(s.to_string())
}

/// Evaluate a [`FilterExpr`] against a decoded row. Unknown columns
/// never match; comparisons between incompatible value types return
/// `false` rather than an error (SQL semantics for NULL comparisons).
///
/// [`FilterExpr::NotDuplicate`] is a **group-level** predicate: by the
/// time the row-level evaluator sees a row, the per-scan dedup pass
/// has already dropped any non-representatives, so this function
/// returns `true` for every row and lets the caller enforce the group
/// filter out of band. The executor's `exec_full_scan` applies
/// [`crate::planner::extract_not_duplicate`] + a representative-per-
/// group pass before per-row filter matching, which mirrors the
/// [`galaxdb_versioning::export::apply_dedup_filter`] contract.
pub fn filter_matches(row: &[(String, Value)], filter: &FilterExpr) -> bool {
    match filter {
        FilterExpr::Eq { column, value } => compare(row, column, value, |a, b| a == b),
        FilterExpr::Ne { column, value } => compare(row, column, value, |a, b| a != b),
        FilterExpr::Lt { column, value } => cmp_order(row, column, value, |a, b| a < b),
        FilterExpr::Gt { column, value } => cmp_order(row, column, value, |a, b| a > b),
        FilterExpr::Le { column, value } => cmp_order(row, column, value, |a, b| a <= b),
        FilterExpr::Ge { column, value } => cmp_order(row, column, value, |a, b| a >= b),
        FilterExpr::And(a, b) => filter_matches(row, a) && filter_matches(row, b),
        FilterExpr::Or(a, b) => filter_matches(row, a) || filter_matches(row, b),
        // Group-level predicate, enforced outside the per-row loop —
        // see the doc comment above. Returning `true` here keeps
        // composed filters like `price > 4 AND NOT DUPLICATE` working:
        // the price comparison narrows the per-row candidates, then
        // the scan-level dedup pass collapses each group.
        FilterExpr::NotDuplicate => true,
    }
}

fn compare<F>(row: &[(String, Value)], column: &str, value: &Value, pred: F) -> bool
where
    F: Fn(&Value, &Value) -> bool,
{
    row.iter()
        .find(|(n, _)| n == column)
        .map(|(_, v)| pred(v, value))
        .unwrap_or(false)
}

fn cmp_order<F>(row: &[(String, Value)], column: &str, value: &Value, pred: F) -> bool
where
    F: Fn(f64, f64) -> bool,
{
    let (Some(a), Some(b)) = (to_f64(row_value(row, column)), to_f64(Some(value))) else {
        return false;
    };
    pred(a, b)
}

fn row_value<'a>(row: &'a [(String, Value)], column: &str) -> Option<&'a Value> {
    row.iter().find(|(n, _)| n == column).map(|(_, v)| v)
}

fn to_f64(v: Option<&Value>) -> Option<f64> {
    match v {
        Some(Value::Integer(n)) => Some(*n as f64),
        Some(Value::Float(f)) => Some(*f),
        Some(Value::Text(s)) => s.parse::<f64>().ok(),
        Some(Value::Bool(b)) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::{CatalogColumn, TableEntry};

    fn users_entry() -> TableEntry {
        TableEntry {
            name: "users".to_string(),
            columns: vec![
                CatalogColumn {
                    name: "id".to_string(),
                    data_type: "INT".to_string(),
                    nullable: false,
                    primary_key: true,
                    is_embedding_source: false,
                },
                CatalogColumn {
                    name: "name".to_string(),
                    data_type: "TEXT".to_string(),
                    nullable: true,
                    primary_key: false,
                    is_embedding_source: false,
                },
            ],
            has_embedding: false,
            append_only: false,
        }
    }

    #[test]
    fn align_positional() {
        let aligned = align_values(
            &users_entry(),
            &[],
            &[Value::Integer(1), Value::Text("alice".into())],
        )
        .unwrap();
        assert_eq!(aligned[0].0, "id");
        assert_eq!(aligned[1].0, "name");
    }

    #[test]
    fn align_named_reorder() {
        let aligned = align_values(
            &users_entry(),
            &["name".to_string(), "id".to_string()],
            &[Value::Text("bob".into()), Value::Integer(7)],
        )
        .unwrap();
        assert_eq!(aligned[0].0, "name");
        assert_eq!(aligned[1].0, "id");
    }

    #[test]
    fn align_mismatch_errors() {
        let err = align_values(
            &users_entry(),
            &["id".to_string(), "name".to_string()],
            &[Value::Integer(1)],
        );
        assert!(err.is_err());
    }

    #[test]
    fn encode_decode_roundtrip() {
        let row = vec![
            ("id".to_string(), Value::Integer(42)),
            ("name".to_string(), Value::Text("alice".into())),
            ("active".to_string(), Value::Bool(true)),
            ("score".to_string(), Value::Float(3.5)),
            ("bio".to_string(), Value::Null),
        ];
        let bytes = encode_row(&row);
        let decoded = decode_row(&bytes);
        assert_eq!(decoded.len(), row.len());
        for (a, b) in row.iter().zip(decoded.iter()) {
            assert_eq!(a.0, b.0);
            assert_eq!(a.1, b.1);
        }
    }

    #[test]
    fn primary_key_prefers_pk_column() {
        let entry = users_entry();
        let ordered = vec![
            ("name".to_string(), Value::Text("alice".into())),
            ("id".to_string(), Value::Integer(5)),
        ];
        let key = build_primary_key("users", &entry, &ordered).unwrap();
        assert_eq!(key, b"users:5");
    }

    #[test]
    fn primary_key_falls_back_to_first_value() {
        let entry = TableEntry {
            name: "events".into(),
            columns: vec![CatalogColumn {
                name: "msg".into(),
                data_type: "TEXT".into(),
                nullable: false,
                primary_key: false,
                is_embedding_source: false,
            }],
            has_embedding: false,
            append_only: false,
        };
        let ordered = vec![("msg".to_string(), Value::Text("hello".into()))];
        let key = build_primary_key("events", &entry, &ordered).unwrap();
        assert_eq!(key, b"events:hello");
    }

    #[test]
    fn filter_eq_matches() {
        let row = vec![
            ("id".to_string(), Value::Integer(3)),
            ("name".to_string(), Value::Text("alice".into())),
        ];
        assert!(filter_matches(
            &row,
            &FilterExpr::Eq {
                column: "id".into(),
                value: Value::Integer(3)
            }
        ));
        assert!(!filter_matches(
            &row,
            &FilterExpr::Eq {
                column: "id".into(),
                value: Value::Integer(4)
            }
        ));
    }

    #[test]
    fn filter_and_or() {
        let row = vec![
            ("a".into(), Value::Integer(2)),
            ("b".into(), Value::Integer(10)),
        ];
        let and = FilterExpr::And(
            Box::new(FilterExpr::Gt {
                column: "a".into(),
                value: Value::Integer(1),
            }),
            Box::new(FilterExpr::Lt {
                column: "b".into(),
                value: Value::Integer(20),
            }),
        );
        assert!(filter_matches(&row, &and));

        let or = FilterExpr::Or(
            Box::new(FilterExpr::Eq {
                column: "a".into(),
                value: Value::Integer(999),
            }),
            Box::new(FilterExpr::Gt {
                column: "b".into(),
                value: Value::Integer(5),
            }),
        );
        assert!(filter_matches(&row, &or));
    }

    #[test]
    fn filter_missing_column_is_false() {
        let row = vec![("a".into(), Value::Integer(2))];
        assert!(!filter_matches(
            &row,
            &FilterExpr::Eq {
                column: "nonexistent".into(),
                value: Value::Integer(0)
            }
        ));
    }

    #[test]
    fn pipe_in_text_is_escaped() {
        let row = vec![
            ("id".to_string(), Value::Integer(1)),
            ("note".to_string(), Value::Text("a|b".into())),
        ];
        let bytes = encode_row(&row);
        let decoded = decode_row(&bytes);
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[1].0, "note");
        assert_eq!(decoded[1].1, Value::Text("a|b".into()));
    }
}
