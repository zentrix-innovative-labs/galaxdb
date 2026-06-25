//! Secondary indexes (Requirement 5).
//!
//! A secondary index maps a non-primary-key column value to the set of
//! primary keys holding that value, so equality and range predicates on
//! that column resolve without a full table scan (AC2).
//!
//! ## Why a reserved-key store (not a separate ART)
//!
//! Index definitions and entries are stored as ordinary engine rows under
//! dedicated sentinel key-prefixes — the same pattern the [`crate::auth_store`]
//! uses for roles/grants. This buys three things for free:
//!
//! * **Durability + crash recovery (AC4).** Every entry is written through
//!   [`Engine::put_sync`] / removed through [`Engine::delete_sync`], so the
//!   index survives restart through the normal WAL + SST replay path with
//!   no separate rebuild step. (The engine's own ART primary index is a
//!   point-lookup structure with no range/scan API, so it can't back a
//!   secondary index without modification anyway.)
//! * **MVCC consistency (AC7).** Entries are tombstoned and versioned by
//!   the same engine machinery as base rows, so a current-snapshot read
//!   through the index sees exactly what a full scan would.
//! * **Sorted scans.** [`Engine::scan_all_with_prefix`] returns rows sorted
//!   by key; with an order-preserving value encoding the index entries come
//!   back in value order, which is what makes range predicates work.
//!
//! ## Key layout
//!
//! ```text
//! def:   b"\x00galaxdb_secidx\x00def\x00"  + index_name                          -> IndexDef bytes
//! entry: b"\x00galaxdb_secidx\x00ent\x00"  + index_name "\x00" + ENC(value) + TERM + pk -> pk
//! ```
//!
//! `ENC(value)` is an **order-preserving** encoding (see [`encode_index_value`])
//! so byte-lexicographic order equals value order. Zero bytes inside the
//! encoded value are escaped (`0x00` -> `0x00 0xFF`) and the value is
//! terminated by `0x00 0x00`, which sorts before any escaped byte so a
//! shorter value sorts first — preserving order through the terminator.
//! The primary key follows the terminator verbatim.

use std::collections::HashMap;
use std::sync::Arc;

use galaxdb_common::{GalaxError, GalaxResult};
use galaxdb_storage::engine::Engine;

use crate::planner::Value;
use crate::row_codec;

const DEF_PREFIX: &[u8] = b"\x00galaxdb_secidx\x00def\x00";
const ENTRY_PREFIX: &[u8] = b"\x00galaxdb_secidx\x00ent\x00";

/// A secondary-index definition recorded in the catalog (AC4: survives
/// restart). The index *contents* are stored as separate entry rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexDef {
    /// Index name (unique across the database).
    pub name: String,
    /// Table the index is defined on.
    pub table: String,
    /// Column being indexed.
    pub column: String,
}

impl IndexDef {
    fn to_bytes(&self) -> Vec<u8> {
        // name_len:u16 LE | name | table_len:u16 LE | table | column
        let mut out = Vec::new();
        out.extend_from_slice(&(self.name.len() as u16).to_le_bytes());
        out.extend_from_slice(self.name.as_bytes());
        out.extend_from_slice(&(self.table.len() as u16).to_le_bytes());
        out.extend_from_slice(self.table.as_bytes());
        out.extend_from_slice(self.column.as_bytes());
        out
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 2 {
            return None;
        }
        let name_len = u16::from_le_bytes([bytes[0], bytes[1]]) as usize;
        let mut pos = 2;
        if bytes.len() < pos + name_len + 2 {
            return None;
        }
        let name = String::from_utf8(bytes[pos..pos + name_len].to_vec()).ok()?;
        pos += name_len;
        let table_len = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]) as usize;
        pos += 2;
        if bytes.len() < pos + table_len {
            return None;
        }
        let table = String::from_utf8(bytes[pos..pos + table_len].to_vec()).ok()?;
        pos += table_len;
        let column = String::from_utf8(bytes[pos..].to_vec()).ok()?;
        Some(IndexDef {
            name,
            table,
            column,
        })
    }
}

/// Order-preserving encoding of a column value for use as the sortable
/// portion of an index key.
///
/// Requirements:
/// * byte-lexicographic order of the output equals the natural order of
///   the value (so `scan_all_with_prefix` returns entries in value order
///   and range predicates can be answered by a bounded scan), and
/// * the output contains no unescaped `0x00`, so a `0x00 0x00` terminator
///   unambiguously ends it.
///
/// Encodings:
/// * Integers: `b'i'` tag + big-endian `u64` with the sign bit flipped, so
///   negative < positive and both sort correctly as unsigned bytes.
/// * Floats: `b'f'` tag + an order-preserving transform of the IEEE-754
///   bits (flip sign bit if positive, flip all bits if negative).
/// * Bool: `b'b'` + `0`/`1`.
/// * Text: `b't'` + UTF-8 bytes (lexicographic — matches SQL text order).
/// * Null: `b'0'` (sorts before every non-null tag, matching SQL `NULL`
///   ordering "nulls first" for ascending scans).
/// * Blob: `b'x'` + raw bytes.
///
/// Distinct type tags keep different types from colliding; within a type
/// the body is order-preserving. After tagging, all `0x00` bytes are
/// escaped so the terminator stays unambiguous.
pub fn encode_index_value(value: &Value) -> Vec<u8> {
    let mut raw = Vec::new();
    encode_index_value_raw(value, &mut raw);
    escape_zeros(&raw)
}

/// Append the pre-escape (unescaped) order/identity encoding of `value` to
/// `raw`. Escaping is applied once by [`encode_index_value`]; this lets
/// composite values (arrays) nest element encodings without double-escaping.
fn encode_index_value_raw(value: &Value, raw: &mut Vec<u8>) {
    match value {
        Value::Null => raw.push(b'0'),
        Value::Integer(n) => {
            raw.push(b'i');
            // Flip the sign bit so the two's-complement ordering becomes
            // unsigned-lexicographic ordering.
            let u = (*n as u64) ^ 0x8000_0000_0000_0000;
            raw.extend_from_slice(&u.to_be_bytes());
        }
        Value::Float(f) => {
            raw.push(b'f');
            let bits = f.to_bits();
            // Order-preserving total ordering of IEEE-754: if the sign bit
            // is set (negative), flip all bits; else flip just the sign bit.
            let ordered = if bits & 0x8000_0000_0000_0000 != 0 {
                !bits
            } else {
                bits ^ 0x8000_0000_0000_0000
            };
            raw.extend_from_slice(&ordered.to_be_bytes());
        }
        Value::Bool(b) => {
            raw.push(b'b');
            raw.push(if *b { 1 } else { 0 });
        }
        Value::Text(s) => {
            raw.push(b't');
            raw.extend_from_slice(s.as_bytes());
        }
        Value::Blob(bytes) => {
            raw.push(b'x');
            raw.extend_from_slice(bytes);
        }
        Value::Array(items) => {
            // Deterministic, collision-free encoding for equality lookups:
            // tag `b'a'`, element count, then a 4-byte big-endian length
            // prefix + the raw element encoding for each item. The length
            // prefixes make concatenated variable-width elements (text,
            // blob) unambiguous, so `{1,2}` and `{12}` never collide. This
            // is identity-preserving (the common array-index use); it is not
            // lexicographically order-preserving across arrays of differing
            // length, which array columns do not rely on for range scans.
            raw.push(b'a');
            raw.extend_from_slice(&(items.len() as u32).to_be_bytes());
            for item in items {
                let mut elem = Vec::new();
                encode_index_value_raw(item, &mut elem);
                raw.extend_from_slice(&(elem.len() as u32).to_be_bytes());
                raw.extend_from_slice(&elem);
            }
        }
    }
}

/// Escape `0x00` as `0x00 0xFF` so that a `0x00 0x00` sequence is reserved
/// as a value terminator and never appears inside an encoded value. The
/// escape byte `0xFF` sorts after everything, so escaping does not disturb
/// the relative order of values that share a prefix.
fn escape_zeros(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() + 2);
    for &b in raw {
        out.push(b);
        if b == 0 {
            out.push(0xFF);
        }
    }
    out
}

fn def_key(name: &str) -> Vec<u8> {
    let mut k = DEF_PREFIX.to_vec();
    k.extend_from_slice(name.as_bytes());
    k
}

/// The per-index entry-key prefix: everything up to and including the
/// index-name separator. All entries for one index share this prefix.
fn entry_prefix_for_index(index_name: &str) -> Vec<u8> {
    let mut k = ENTRY_PREFIX.to_vec();
    k.extend_from_slice(index_name.as_bytes());
    k.push(0);
    k
}

/// The entry-key prefix for a specific *value* within an index: the index
/// prefix + encoded value + the `0x00 0x00` terminator. A prefix scan with
/// this returns exactly the primary keys holding that value (equality).
fn entry_value_prefix(index_name: &str, value: &Value) -> Vec<u8> {
    let mut k = entry_prefix_for_index(index_name);
    k.extend_from_slice(&encode_index_value(value));
    k.push(0);
    k.push(0); // value terminator
    k
}

/// The full entry key: value prefix + the primary key bytes.
fn entry_key(index_name: &str, value: &Value, pk: &[u8]) -> Vec<u8> {
    let mut k = entry_value_prefix(index_name, value);
    k.extend_from_slice(pk);
    k
}

/// Extract the trailing primary key from a full entry key, given the index
/// name. Returns `None` if the key isn't a well-formed entry for that index.
fn pk_from_entry_key(index_name: &str, key: &[u8]) -> Option<Vec<u8>> {
    let index_prefix = entry_prefix_for_index(index_name);
    let rest = key.strip_prefix(index_prefix.as_slice())?;
    // Find the `0x00 0x00` terminator that separates the encoded value
    // from the primary key. Escaped zeros are `0x00 0xFF`, so a real
    // terminator is the first `0x00` followed by `0x00`.
    let mut i = 0;
    while i + 1 < rest.len() {
        if rest[i] == 0 {
            if rest[i + 1] == 0 {
                // Terminator at i..i+2; PK is everything after.
                return Some(rest[i + 2..].to_vec());
            }
            // Escaped zero (0x00 0xFF): skip both bytes.
            i += 2;
        } else {
            i += 1;
        }
    }
    None
}

/// The durable secondary-index store, backed by the storage engine.
///
/// Cheap to clone (holds an `Arc<Engine>`). All reads/writes go through the
/// engine, so the index reflects committed state after restart with no
/// rebuild step (AC4).
#[derive(Clone)]
pub struct SecondaryIndexStore {
    engine: Arc<Engine>,
}

impl SecondaryIndexStore {
    /// Wrap an engine handle.
    pub fn new(engine: Arc<Engine>) -> Self {
        SecondaryIndexStore { engine }
    }

    // ---- Definitions ----

    /// Create an index definition. Errors if one with the same name
    /// already exists.
    pub fn create_def(&self, def: &IndexDef) -> GalaxResult<()> {
        if self.get_def(&def.name).is_some() {
            return Err(GalaxError::Internal(format!(
                "index '{}' already exists",
                def.name
            )));
        }
        self.engine
            .put_sync(def_key(&def.name), def.to_bytes())
            .map(|_| ())
            .map_err(|e| GalaxError::Internal(format!("secondary index create_def: {e}")))
    }

    /// Fetch an index definition by name.
    pub fn get_def(&self, name: &str) -> Option<IndexDef> {
        let bytes = self.engine.get(&def_key(name))?;
        IndexDef::from_bytes(&bytes)
    }

    /// Drop an index: remove its definition and every entry. Returns
    /// `true` if the index existed.
    pub fn drop_index(&self, name: &str) -> GalaxResult<bool> {
        let Some(_def) = self.get_def(name) else {
            return Ok(false);
        };
        // Delete every entry row for this index.
        let prefix = entry_prefix_for_index(name);
        for (key, _) in self.engine.scan_all_with_prefix(Some(&prefix)) {
            if key.starts_with(&prefix) {
                self.engine
                    .delete_sync(&key)
                    .map_err(|e| GalaxError::Internal(format!("secondary index drop entry: {e}")))?;
            }
        }
        self.engine
            .delete_sync(&def_key(name))
            .map_err(|e| GalaxError::Internal(format!("secondary index drop_def: {e}")))?;
        Ok(true)
    }

    /// All index definitions on a given table.
    pub fn defs_for_table(&self, table: &str) -> Vec<IndexDef> {
        self.engine
            .scan_all_with_prefix(Some(DEF_PREFIX))
            .into_iter()
            .filter_map(|(key, _)| {
                if key.starts_with(DEF_PREFIX) {
                    self.engine.get(&key).and_then(|b| IndexDef::from_bytes(&b))
                } else {
                    None
                }
            })
            .filter(|d| d.table == table)
            .collect()
    }

    /// The single index covering `column` on `table`, if any.
    pub fn def_for_column(&self, table: &str, column: &str) -> Option<IndexDef> {
        self.defs_for_table(table)
            .into_iter()
            .find(|d| d.column == column)
    }

    // ---- Entry maintenance ----

    /// Insert an index entry `(value -> pk)` (idempotent).
    pub fn insert_entry(&self, index_name: &str, value: &Value, pk: &[u8]) -> GalaxResult<()> {
        let key = entry_key(index_name, value, pk);
        self.engine
            .put_sync(key, pk.to_vec())
            .map(|_| ())
            .map_err(|e| GalaxError::Internal(format!("secondary index insert_entry: {e}")))
    }

    /// Remove an index entry `(value -> pk)` (idempotent).
    pub fn delete_entry(&self, index_name: &str, value: &Value, pk: &[u8]) -> GalaxResult<()> {
        let key = entry_key(index_name, value, pk);
        self.engine
            .delete_sync(&key)
            .map(|_| ())
            .map_err(|e| GalaxError::Internal(format!("secondary index delete_entry: {e}")))
    }

    /// Look up the primary keys holding `value` on the given index
    /// (equality predicate, AC2). Returns the PKs in sorted order.
    pub fn lookup_eq(&self, index_name: &str, value: &Value) -> Vec<Vec<u8>> {
        let prefix = entry_value_prefix(index_name, value);
        self.engine
            .scan_all_with_prefix(Some(&prefix))
            .into_iter()
            .filter(|(k, _)| k.starts_with(&prefix))
            .filter_map(|(k, _)| pk_from_entry_key(index_name, &k))
            .collect()
    }

    /// Look up the primary keys whose indexed value lies in the inclusive
    /// range `[low, high]` (range predicate, AC2). Either bound may be
    /// `None` for an open range. Relies on the order-preserving value
    /// encoding so a single sorted scan answers the range. NULLs are
    /// excluded from range results (SQL semantics: NULL is unordered).
    pub fn lookup_range(
        &self,
        index_name: &str,
        low: Option<&Value>,
        high: Option<&Value>,
    ) -> Vec<Vec<u8>> {
        let index_prefix = entry_prefix_for_index(index_name);
        // Encoded bounds (without the value terminator) for comparison
        // against each entry's encoded-value portion.
        let low_enc = low.map(encode_index_value);
        let high_enc = high.map(encode_index_value);

        let mut out = Vec::new();
        for (key, _) in self.engine.scan_all_with_prefix(Some(&index_prefix)) {
            if !key.starts_with(&index_prefix) {
                continue;
            }
            let rest = &key[index_prefix.len()..];
            // Split the entry's encoded value from the trailing PK at the
            // `0x00 0x00` terminator.
            let Some((enc_value, _pk)) = split_value_and_pk(rest) else {
                continue;
            };
            // Exclude NULLs from range scans (tag b'0', escaped to the
            // single byte 0x30 — never matches a numeric/text bound).
            if enc_value.first() == Some(&b'0') {
                continue;
            }
            if let Some(lo) = &low_enc {
                if enc_value.as_slice() < lo.as_slice() {
                    continue;
                }
            }
            if let Some(hi) = &high_enc {
                if enc_value.as_slice() > hi.as_slice() {
                    continue;
                }
            }
            if let Some(pk) = pk_from_entry_key(index_name, &key) {
                out.push(pk);
            }
        }
        out
    }

    /// Maintain every index on `table` for an inserted row: add one entry
    /// per index for the row's value in that index's column.
    pub fn on_row_inserted(
        &self,
        table: &str,
        ordered: &[(String, Value)],
        pk: &[u8],
    ) -> GalaxResult<()> {
        for def in self.defs_for_table(table) {
            if let Some(value) = column_value(ordered, &def.column) {
                self.insert_entry(&def.name, value, pk)?;
            }
        }
        Ok(())
    }

    /// Maintain every index on `table` for a deleted row: remove the entry
    /// for each index's column value.
    pub fn on_row_deleted(
        &self,
        table: &str,
        ordered: &[(String, Value)],
        pk: &[u8],
    ) -> GalaxResult<()> {
        for def in self.defs_for_table(table) {
            if let Some(value) = column_value(ordered, &def.column) {
                self.delete_entry(&def.name, value, pk)?;
            }
        }
        Ok(())
    }

    /// Maintain every index on `table` for an updated row: for each index
    /// whose column changed, remove the old entry and add the new one.
    /// The primary key is unchanged by UPDATE (it identifies the row).
    pub fn on_row_updated(
        &self,
        table: &str,
        old_ordered: &[(String, Value)],
        new_ordered: &[(String, Value)],
        pk: &[u8],
    ) -> GalaxResult<()> {
        for def in self.defs_for_table(table) {
            let old_v = column_value(old_ordered, &def.column);
            let new_v = column_value(new_ordered, &def.column);
            if old_v != new_v {
                if let Some(ov) = old_v {
                    self.delete_entry(&def.name, ov, pk)?;
                }
                if let Some(nv) = new_v {
                    self.insert_entry(&def.name, nv, pk)?;
                }
            }
        }
        Ok(())
    }

    /// Rebuild an index from the current contents of its base table. Used
    /// by `CREATE INDEX` on a non-empty table so existing rows are
    /// covered immediately (AC2 from creation, not only for future rows).
    pub fn build_from_table(&self, def: &IndexDef) -> GalaxResult<u64> {
        let table_prefix = format!("{}:", def.table);
        let mut count = 0u64;
        for (key, value_bytes) in self
            .engine
            .scan_all_with_prefix(Some(table_prefix.as_bytes()))
        {
            if !key.starts_with(table_prefix.as_bytes()) {
                continue;
            }
            let cols = row_codec::decode_row(&value_bytes);
            if let Some(value) = column_value(&cols, &def.column) {
                self.insert_entry(&def.name, value, &key)?;
                count += 1;
            }
        }
        Ok(count)
    }
}

/// Find a column's value in a decoded/ordered row.
fn column_value<'a>(ordered: &'a [(String, Value)], column: &str) -> Option<&'a Value> {
    ordered.iter().find(|(c, _)| c == column).map(|(_, v)| v)
}

/// Split an entry-key remainder (after the index prefix) into the encoded
/// value (without terminator) and the trailing primary key, at the
/// `0x00 0x00` terminator. Mirrors [`pk_from_entry_key`] but also returns
/// the value portion.
fn split_value_and_pk(rest: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
    let mut i = 0;
    while i + 1 < rest.len() {
        if rest[i] == 0 {
            if rest[i + 1] == 0 {
                return Some((rest[..i].to_vec(), rest[i + 2..].to_vec()));
            }
            i += 2; // escaped zero
        } else {
            i += 1;
        }
    }
    None
}

/// Resolve the set of primary keys for an indexable equality/range filter,
/// returned as a `HashMap` for O(1) membership during scan intersection.
/// `None` means "no usable index for this filter".
pub fn index_pk_set(
    store: &SecondaryIndexStore,
    table: &str,
    filter: &crate::planner::FilterExpr,
) -> Option<HashMap<Vec<u8>, ()>> {
    use crate::planner::FilterExpr;
    let (column, pks) = match filter {
        FilterExpr::Eq { column, value } => {
            let def = store.def_for_column(table, column)?;
            (column, store.lookup_eq(&def.name, value))
        }
        FilterExpr::Gt { column, value } => {
            let def = store.def_for_column(table, column)?;
            // Exclusive low bound: scan >= value then drop equals handled
            // by the executor's exact filter re-check, so an inclusive
            // scan is a safe superset.
            (column, store.lookup_range(&def.name, Some(value), None))
        }
        FilterExpr::Ge { column, value } => {
            let def = store.def_for_column(table, column)?;
            (column, store.lookup_range(&def.name, Some(value), None))
        }
        FilterExpr::Lt { column, value } => {
            let def = store.def_for_column(table, column)?;
            (column, store.lookup_range(&def.name, None, Some(value)))
        }
        FilterExpr::Le { column, value } => {
            let def = store.def_for_column(table, column)?;
            (column, store.lookup_range(&def.name, None, Some(value)))
        }
        _ => return None,
    };
    let _ = column;
    Some(pks.into_iter().map(|pk| (pk, ())).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use galaxdb_storage::engine::{Engine, EngineConfig};

    fn test_engine() -> Arc<Engine> {
        let dir = tempfile::tempdir().unwrap();
        let config = EngineConfig {
            data_dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        std::mem::forget(dir);
        Arc::new(Engine::new(config).unwrap())
    }

    #[test]
    fn index_value_encoding_preserves_integer_order() {
        // -5 < 0 < 7 < 100 must hold byte-lexicographically.
        let enc = |n: i64| encode_index_value(&Value::Integer(n));
        assert!(enc(-5) < enc(0));
        assert!(enc(0) < enc(7));
        assert!(enc(7) < enc(100));
        assert!(enc(-100) < enc(-5));
    }

    #[test]
    fn index_value_encoding_preserves_float_order() {
        let enc = |f: f64| encode_index_value(&Value::Float(f));
        assert!(enc(-1.5) < enc(0.0));
        assert!(enc(0.0) < enc(2.5));
        assert!(enc(2.5) < enc(100.25));
    }

    #[test]
    fn index_value_encoding_preserves_text_order() {
        let enc = |s: &str| encode_index_value(&Value::Text(s.to_string()));
        assert!(enc("alice") < enc("bob"));
        assert!(enc("a") < enc("aa"));
    }

    #[test]
    fn index_def_byte_roundtrip() {
        let def = IndexDef {
            name: "idx_age".into(),
            table: "users".into(),
            column: "age".into(),
        };
        assert_eq!(IndexDef::from_bytes(&def.to_bytes()).unwrap(), def);
    }

    #[test]
    fn pk_extraction_handles_zero_bytes_in_value() {
        // A text value containing a literal NUL must round-trip: the
        // escape keeps the terminator unambiguous.
        let idx = "idx_x";
        let value = Value::Text("a\u{0}b".to_string());
        let pk = b"users:42";
        let key = entry_key(idx, &value, pk);
        assert_eq!(pk_from_entry_key(idx, &key).unwrap(), pk.to_vec());
    }

    #[test]
    fn create_lookup_drop_roundtrip() {
        let store = SecondaryIndexStore::new(test_engine());
        let def = IndexDef {
            name: "idx_city".into(),
            table: "people".into(),
            column: "city".into(),
        };
        store.create_def(&def).unwrap();
        assert!(store.create_def(&def).is_err(), "duplicate index errors");
        assert_eq!(store.def_for_column("people", "city"), Some(def.clone()));

        // Two people in NYC, one in LA.
        store
            .insert_entry("idx_city", &Value::Text("NYC".into()), b"people:1")
            .unwrap();
        store
            .insert_entry("idx_city", &Value::Text("NYC".into()), b"people:2")
            .unwrap();
        store
            .insert_entry("idx_city", &Value::Text("LA".into()), b"people:3")
            .unwrap();

        let mut nyc = store.lookup_eq("idx_city", &Value::Text("NYC".into()));
        nyc.sort();
        assert_eq!(nyc, vec![b"people:1".to_vec(), b"people:2".to_vec()]);
        assert_eq!(
            store.lookup_eq("idx_city", &Value::Text("LA".into())),
            vec![b"people:3".to_vec()]
        );

        // Drop removes def + entries.
        assert!(store.drop_index("idx_city").unwrap());
        assert!(store.def_for_column("people", "city").is_none());
        assert!(store
            .lookup_eq("idx_city", &Value::Text("NYC".into()))
            .is_empty());
    }

    #[test]
    fn range_lookup_returns_values_in_bounds() {
        let store = SecondaryIndexStore::new(test_engine());
        store
            .create_def(&IndexDef {
                name: "idx_age".into(),
                table: "u".into(),
                column: "age".into(),
            })
            .unwrap();
        for (age, pk) in [(20i64, "u:a"), (30, "u:b"), (40, "u:c"), (50, "u:d")] {
            store
                .insert_entry("idx_age", &Value::Integer(age), pk.as_bytes())
                .unwrap();
        }
        // [30, 45] -> ages 30 and 40.
        let mut got = store.lookup_range(
            "idx_age",
            Some(&Value::Integer(30)),
            Some(&Value::Integer(45)),
        );
        got.sort();
        assert_eq!(got, vec![b"u:b".to_vec(), b"u:c".to_vec()]);

        // Open upper bound: >= 40 -> 40 and 50.
        let mut hi = store.lookup_range("idx_age", Some(&Value::Integer(40)), None);
        hi.sort();
        assert_eq!(hi, vec![b"u:c".to_vec(), b"u:d".to_vec()]);
    }

    #[test]
    fn entries_survive_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_path_buf();
        {
            let engine = Arc::new(
                Engine::new(EngineConfig {
                    data_dir: path.clone(),
                    ..Default::default()
                })
                .unwrap(),
            );
            let store = SecondaryIndexStore::new(engine.clone());
            store
                .create_def(&IndexDef {
                    name: "idx_k".into(),
                    table: "t".into(),
                    column: "k".into(),
                })
                .unwrap();
            store
                .insert_entry("idx_k", &Value::Integer(7), b"t:1")
                .unwrap();
            engine.shutdown();
        }
        // Reopen: WAL replay restores the definition and the entry (AC4).
        let engine = Arc::new(
            Engine::new(EngineConfig {
                data_dir: path,
                ..Default::default()
            })
            .unwrap(),
        );
        let store = SecondaryIndexStore::new(engine);
        assert!(store.get_def("idx_k").is_some());
        assert_eq!(
            store.lookup_eq("idx_k", &Value::Integer(7)),
            vec![b"t:1".to_vec()]
        );
    }

    #[test]
    fn update_maintains_entries() {
        let store = SecondaryIndexStore::new(test_engine());
        store
            .create_def(&IndexDef {
                name: "idx_status".into(),
                table: "orders".into(),
                column: "status".into(),
            })
            .unwrap();
        let pk = b"orders:1";
        let old = vec![("status".to_string(), Value::Text("pending".into()))];
        let new = vec![("status".to_string(), Value::Text("shipped".into()))];
        store.on_row_inserted("orders", &old, pk).unwrap();
        assert_eq!(
            store.lookup_eq("idx_status", &Value::Text("pending".into())),
            vec![pk.to_vec()]
        );
        store.on_row_updated("orders", &old, &new, pk).unwrap();
        assert!(store
            .lookup_eq("idx_status", &Value::Text("pending".into()))
            .is_empty());
        assert_eq!(
            store.lookup_eq("idx_status", &Value::Text("shipped".into())),
            vec![pk.to_vec()]
        );
    }
}
