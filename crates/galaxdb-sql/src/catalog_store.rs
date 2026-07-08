//! Durable catalog persistence.
//!
//! Table definitions (the [`Catalog`](crate::executor::Catalog)) were
//! historically in-memory only: `CREATE TABLE` mutated a `HashMap` that was
//! rebuilt empty on every `Database::open`. That made a server restart lose
//! every schema — durable row data in the WAL/SSTs became unreadable because
//! the engine no longer knew the table existed. This module fixes that by
//! persisting each table's [`TableEntry`] to the storage engine under a
//! reserved key prefix, through the same WAL-backed `put_sync` as row data,
//! and reloading it on open.
//!
//! ## Key layout
//!
//! Catalog entries live under [`CATALOG_KEY_PREFIX`], which begins with a
//! `0x00` byte. User row keys are `"{table}:{pk}"` where the table name is a
//! SQL identifier (ASCII, never starting with `0x00`), so the catalog
//! namespace is disjoint from every user table and from full-table scans
//! (which always scan the `"{table}:"` prefix).
//!
//! ## Encoding
//!
//! A small, explicit, versioned line-oriented format (`GXCAT1`). Table and
//! column names and SQL type names never contain newlines, so a
//! newline-delimited layout is unambiguous. A malformed record decodes to
//! `None` and is skipped rather than panicking — a corrupt catalog entry can
//! never crash open.

use std::sync::Arc;

use galaxdb_common::{GalaxError, GalaxResult, StorageMode};
use galaxdb_storage::engine::Engine;

use crate::executor::{CatalogColumn, TableEntry};

/// Reserved key prefix for persisted catalog entries. Leading `0x00` keeps it
/// disjoint from every `"{table}:"` user-row namespace.
pub const CATALOG_KEY_PREFIX: &[u8] = b"\x00__galaxdb_catalog__:";

/// The catalog record family tag. The on-disk first line is
/// `"{CATALOG_MAGIC}{version}"` (e.g. `GXCAT1`), fusing magic + format version.
pub const CATALOG_MAGIC: &str = "GXCAT";

/// Current catalog record format version (the `1` in `GXCAT1`). Kept in step
/// with `galaxdb_common::format::CATALOG.current_write`.
pub const CATALOG_FORMAT_VERSION: u16 = 1;

/// Extract the format version from a catalog record's first line, if the line
/// is a recognized `GXCAT<version>` tag. Returns `None` for bytes that are not
/// a catalog record at all (foreign/corrupt), which callers skip; a recognized
/// tag with an unparsable version yields `None` as well (treated as malformed).
fn catalog_record_version(bytes: &[u8]) -> Option<u16> {
    let text = std::str::from_utf8(bytes).ok()?;
    let first_line = text.split('\n').next()?;
    let digits = first_line.strip_prefix(CATALOG_MAGIC)?;
    digits.parse::<u16>().ok()
}

/// The storage key a table's catalog entry is persisted under.
pub fn catalog_key(table: &str) -> Vec<u8> {
    let mut k = CATALOG_KEY_PREFIX.to_vec();
    k.extend_from_slice(table.as_bytes());
    k
}

/// Serialize a [`TableEntry`] into the `GXCAT1` byte format.
pub fn encode_table_entry(entry: &TableEntry) -> Vec<u8> {
    let mut lines: Vec<String> = Vec::with_capacity(6 + entry.columns.len() * 5);
    lines.push(format!("{CATALOG_MAGIC}{CATALOG_FORMAT_VERSION}"));
    lines.push(entry.name.clone());
    lines.push(bool_str(entry.has_embedding).to_string());
    lines.push(bool_str(entry.append_only).to_string());
    lines.push(storage_str(entry.storage_mode).to_string());
    lines.push(entry.columns.len().to_string());
    for c in &entry.columns {
        lines.push(c.name.clone());
        lines.push(c.data_type.clone());
        lines.push(bool_str(c.nullable).to_string());
        lines.push(bool_str(c.primary_key).to_string());
        lines.push(bool_str(c.is_embedding_source).to_string());
    }
    lines.join("\n").into_bytes()
}

/// Parse a `GXCAT1` record back into a [`TableEntry`]. Returns `None` on any
/// malformed input.
pub fn decode_table_entry(bytes: &[u8]) -> Option<TableEntry> {
    let text = std::str::from_utf8(bytes).ok()?;
    let mut lines = text.split('\n');

    if lines.next()? != "GXCAT1" {
        return None;
    }
    let name = lines.next()?.to_string();
    let has_embedding = parse_bool(lines.next()?)?;
    let append_only = parse_bool(lines.next()?)?;
    let storage_mode = parse_storage(lines.next()?)?;
    let num_columns: usize = lines.next()?.parse().ok()?;

    let mut columns = Vec::with_capacity(num_columns);
    for _ in 0..num_columns {
        let cname = lines.next()?.to_string();
        let data_type = lines.next()?.to_string();
        let nullable = parse_bool(lines.next()?)?;
        let primary_key = parse_bool(lines.next()?)?;
        let is_embedding_source = parse_bool(lines.next()?)?;
        columns.push(CatalogColumn {
            name: cname,
            data_type,
            nullable,
            primary_key,
            is_embedding_source,
        });
    }

    Some(TableEntry {
        name,
        columns,
        has_embedding,
        append_only,
        storage_mode,
    })
}

/// Persist (create or overwrite) a table's catalog entry durably.
pub fn persist_table_entry(engine: &Engine, entry: &TableEntry) -> GalaxResult<()> {
    engine
        .put_sync(catalog_key(&entry.name), encode_table_entry(entry))
        .map(|_| ())
        .map_err(|e| GalaxError::Internal(format!("failed to persist catalog entry: {e}")))
}

/// Remove a table's catalog entry (on DROP TABLE).
pub fn remove_table_entry(engine: &Engine, table: &str) -> GalaxResult<()> {
    engine
        .delete_sync(&catalog_key(table))
        .map(|_| ())
        .map_err(|e| GalaxError::Internal(format!("failed to remove catalog entry: {e}")))
}

/// Load every persisted table entry from the engine.
///
/// Version safety (Req 5.2, rollback safety): a catalog record whose format
/// version is **newer** than this engine writes is refused with a typed
/// [`GalaxError::FormatTooNew`] rather than silently skipped — otherwise a
/// rollback onto newer-written catalog rows would make tables *vanish* instead
/// of failing loudly. Genuinely malformed / foreign records (no `GXCAT` tag)
/// are still skipped, never fatal, so a single corrupt entry can't block open.
pub fn load_all(engine: &Arc<Engine>) -> GalaxResult<Vec<TableEntry>> {
    let mut out = Vec::new();
    for (_k, v) in engine
        .scan_all_with_prefix(Some(CATALOG_KEY_PREFIX))
        .into_iter()
        .filter(|(k, _)| k.starts_with(CATALOG_KEY_PREFIX))
    {
        // Gate on the record's declared version before trusting its layout.
        if let Some(version) = catalog_record_version(&v) {
            galaxdb_common::format::CATALOG.check(version)?;
        }
        if let Some(entry) = decode_table_entry(&v) {
            out.push(entry);
        }
    }
    Ok(out)
}

fn bool_str(b: bool) -> &'static str {
    if b {
        "1"
    } else {
        "0"
    }
}

fn parse_bool(s: &str) -> Option<bool> {
    match s {
        "1" => Some(true),
        "0" => Some(false),
        _ => None,
    }
}

fn storage_str(m: StorageMode) -> &'static str {
    match m {
        StorageMode::Columnar => "C",
        StorageMode::Legacy => "L",
    }
}

fn parse_storage(s: &str) -> Option<StorageMode> {
    match s {
        "C" => Some(StorageMode::Columnar),
        "L" => Some(StorageMode::Legacy),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> TableEntry {
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
                    name: "bio".to_string(),
                    data_type: "VARCHAR(255)".to_string(),
                    nullable: true,
                    primary_key: false,
                    is_embedding_source: true,
                },
            ],
            has_embedding: true,
            append_only: false,
            storage_mode: StorageMode::Columnar,
        }
    }

    #[test]
    fn round_trips() {
        let e = sample();
        let bytes = encode_table_entry(&e);
        let back = decode_table_entry(&bytes).expect("decode");
        assert_eq!(back.name, e.name);
        assert_eq!(back.has_embedding, e.has_embedding);
        assert_eq!(back.append_only, e.append_only);
        assert_eq!(back.storage_mode, e.storage_mode);
        assert_eq!(back.columns.len(), 2);
        assert_eq!(back.columns[1].name, "bio");
        assert_eq!(back.columns[1].data_type, "VARCHAR(255)");
        assert!(back.columns[1].nullable);
        assert!(back.columns[0].primary_key);
        assert!(back.columns[1].is_embedding_source);
    }

    #[test]
    fn malformed_returns_none() {
        assert!(decode_table_entry(b"not a catalog record").is_none());
        assert!(decode_table_entry(b"GXCAT1\nusers\n1").is_none());
    }

    #[test]
    fn version_is_extracted_from_record() {
        // A real encoded entry advertises the current version.
        let bytes = encode_table_entry(&sample());
        assert_eq!(catalog_record_version(&bytes), Some(CATALOG_FORMAT_VERSION));
        // A future-versioned tag parses to its number.
        assert_eq!(catalog_record_version(b"GXCAT2\nusers\n..."), Some(2));
        // Non-catalog bytes are unrecognized (skipped by load_all, never fatal).
        assert_eq!(catalog_record_version(b"random bytes"), None);
    }

    #[test]
    fn newer_catalog_version_is_refused_by_gate() {
        // The load gate uses format::CATALOG.check on the parsed version. A
        // newer version must be a typed FormatTooNew (rollback safety), not a
        // silent skip that would make the table vanish.
        let ver = catalog_record_version(b"GXCAT2\nusers\n...").unwrap();
        match galaxdb_common::format::CATALOG.check(ver) {
            Err(galaxdb_common::GalaxError::FormatTooNew {
                artifact, found, ..
            }) => {
                assert_eq!(artifact, "catalog");
                assert_eq!(found, 2);
            }
            other => panic!("expected FormatTooNew, got {other:?}"),
        }
        // The current version passes the gate.
        assert!(galaxdb_common::format::CATALOG
            .check(CATALOG_FORMAT_VERSION)
            .is_ok());
    }

    #[test]
    fn key_is_disjoint_from_user_rows() {
        let k = catalog_key("users");
        assert!(k.starts_with(CATALOG_KEY_PREFIX));
        // A user row key "users:1" never starts with the 0x00 prefix.
        assert!(!b"users:1".starts_with(CATALOG_KEY_PREFIX));
    }
}
