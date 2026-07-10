//! Semantic-cache configuration store (v0.7, inventory 8.11).
//!
//! `CREATE SEMANTIC CACHE FOR TABLE t SIMILARITY f TTL n` persists a
//! per-table cache configuration so the cache stays enabled across restart.
//! The cached *entries* are in-memory (rebuilt naturally); only the config
//! is durable.
//!
//! Storage follows the same engine-backed reserved-prefix pattern as
//! [`crate::secondary_index`] / [`crate::auth_store`]: config rows are
//! ordinary engine rows under a sentinel prefix, so they survive restart
//! through the normal WAL + SST replay path with no separate rebuild.
//!
//! ```text
//! key:   b"\x00galaxdb_semcache\x00" + table  -> CacheConfig bytes
//! value: similarity: f32 LE | ttl_secs: u32 LE
//! ```

use std::sync::Arc;

use galaxdb_common::GalaxResult;
use galaxdb_storage::engine::Engine;

const PREFIX: &[u8] = b"\x00galaxdb_semcache\x00";

/// Persisted per-table semantic-cache configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CacheConfig {
    /// Cosine-similarity threshold in (0.0, 1.0] for a cache hit.
    pub similarity: f32,
    /// Time-to-live in seconds.
    pub ttl_secs: u32,
}

impl CacheConfig {
    fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8);
        out.extend_from_slice(&self.similarity.to_le_bytes());
        out.extend_from_slice(&self.ttl_secs.to_le_bytes());
        out
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 8 {
            return None;
        }
        let similarity = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let ttl_secs = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        Some(CacheConfig {
            similarity,
            ttl_secs,
        })
    }
}

fn config_key(table: &str) -> Vec<u8> {
    let mut k = PREFIX.to_vec();
    k.extend_from_slice(table.as_bytes());
    k
}

/// Engine-backed store of semantic-cache configurations.
pub struct SemanticCacheStore {
    engine: Arc<Engine>,
}

impl SemanticCacheStore {
    pub fn new(engine: Arc<Engine>) -> Self {
        Self { engine }
    }

    /// Persist (create or replace) the config for `table` (Req 1.2, 1.4:
    /// CREATE on an already-cached table replaces its config).
    pub fn put(&self, table: &str, config: CacheConfig) -> GalaxResult<()> {
        self.engine
            .put_sync(config_key(table), config.to_bytes())
            .map(|_| ())
    }

    /// Look up the config for `table`, if a cache is configured.
    pub fn get(&self, table: &str) -> Option<CacheConfig> {
        self.engine
            .get(&config_key(table))
            .and_then(|bytes| CacheConfig::from_bytes(&bytes))
    }

    /// Remove the config for `table` (`DROP SEMANTIC CACHE`). Returns
    /// whether a config existed.
    pub fn drop_config(&self, table: &str) -> GalaxResult<bool> {
        let existed = self.get(table).is_some();
        if existed {
            self.engine.delete_sync(&config_key(table))?;
        }
        Ok(existed)
    }

    /// Load every configured table's cache config (used on open to
    /// initialize the in-memory caches).
    pub fn load_all(&self) -> Vec<(String, CacheConfig)> {
        let mut out = Vec::new();
        for (key, value) in self.engine.scan_all_with_prefix(Some(PREFIX)) {
            if !key.starts_with(PREFIX) {
                continue;
            }
            let table = match String::from_utf8(key[PREFIX.len()..].to_vec()) {
                Ok(t) => t,
                Err(_) => continue,
            };
            if let Some(cfg) = CacheConfig::from_bytes(&value) {
                out.push((table, cfg));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_bytes_roundtrip() {
        let c = CacheConfig {
            similarity: 0.92,
            ttl_secs: 3600,
        };
        let b = c.to_bytes();
        assert_eq!(CacheConfig::from_bytes(&b), Some(c));
    }
}
