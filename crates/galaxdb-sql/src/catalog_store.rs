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

/// The storage key a table's catalog entry is persisted under.
pub fn catalog_key(table: &str) -> Vec<u8> {
    let mut k = CATALOG_KEY_PREFIX.to_vec();
    k.extend_from_slice(table.as_bytes());
    k
}

/// Serialize a [`TableEntry`] into the `GXCAT1` byte format.
pub fn encode_table_entry(entry: &TableEntry) -> Vec<u8> {
    let mut lines: Vec<String> = Vec::with_capacity(6 + entry.columns.len() * 5);
    lines.push("GXCAT1".to_string());
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

/// Load every persisted table entry from the engine. Malformed records are
/// skipped (logged by the caller if desired), never fatal.
pub fn load_all(engine: &Arc<Engine>) -> Vec<TableEntry> {
    engine
        .scan_all_with_prefix(Some(CATALOG_KEY_PREFIX))
        .into_iter()
        .filter(|(k, _)| k.starts_with(CATALOG_KEY_PREFIX))
        .filter_map(|(_, v)| decode_table_entry(&v))
        .collect()
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
    fn key_is_disjoint_from_user_rows() {
        let k = catalog_key("users");
        assert!(k.starts_with(CATALOG_KEY_PREFIX));
        // A user row key "users:1" never starts with the 0x00 prefix.
        assert!(!b"users:1".starts_with(CATALOG_KEY_PREFIX));
    }
}
