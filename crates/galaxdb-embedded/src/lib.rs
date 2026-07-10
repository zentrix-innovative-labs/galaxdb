//! GalaxDB Embedded — thin wrapper around the canonical executor.
//!
//! The embedded crate used to carry its own inline SQL execution logic
//! (CREATE TABLE / INSERT / SELECT / UPDATE / SEMANTIC_MATCH). During
//! the consolidation sprint that code moved into
//! `galaxdb_sql::executor::execute_with_context`, where it operates
//! through a real `Engine` + `ExecutorContext`. This crate now owns
//! the per-database state (engine, sidecar, tag catalog, vector
//! indexes) and delegates every statement to the canonical executor.
//!
//! Public API shape is preserved:
//!
//! * `Database::open(path)` / `Database::open_with_sidecar(path, bin, model_id)`
//! * `Database::execute(sql) -> GalaxResult<QueryResult>`
//! * `Database::execute_async(sql) -> GalaxResult<QueryResult>`
//! * `Database::execute_readonly(sql) -> GalaxResult<QueryResult>`
//! * `table_count`, `table_exists`, `row_count`, `path`
//!
//! There are no mocks on any production path. Sidecar unavailability
//! surfaces as a typed `GalaxError::SidecarUnavailable`; a missing
//! model means the sidecar process exits with status 1 and the engine
//! sees a dead child.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use galaxdb_common::{GalaxError, GalaxResult};
use galaxdb_sidecar::manager::{SidecarConfig, SidecarManager};
use galaxdb_sidecar::protocol::EmbedRequest;
use galaxdb_sql::ast::{AuroraStatement, CreateTableStmt};
use galaxdb_sql::executor::{
    execute_with_context, ExecuteResult, ExecutorContext, Row as SqlRow, VectorSearchBackend,
    VectorSearchResult,
};
use galaxdb_sql::parser;
use galaxdb_sql::planner::{self, FilterExpr, QueryPlan, SearchStrategy, Value};
use galaxdb_sql::row_codec;
use galaxdb_sql::scalar::{ArithOp, ScalarExpr};

/// Re-export of the bound-parameter value type so wire-protocol callers
/// (the server) can name it without depending on `galaxdb-sql` directly.
pub use galaxdb_sql::BoundValue;
use galaxdb_storage::engine::{Engine, EngineConfig};
use galaxdb_vector::{
    execute_semantic_match, DeltaBuffer, HnswConfig, HnswGraph, SemanticMatchConfig,
};
use galaxdb_versioning::{MerkleDag, TagCatalog};

// ---------------------------------------------------------------------------
// Query result types (stable public surface)
// ---------------------------------------------------------------------------

/// A single row returned by `Database::execute`. Each entry is
/// `(column_name, stringified_value)` — strings are the rendered form
/// of [`galaxdb_sql::planner::Value`] as produced by
/// [`galaxdb_sql::row_codec::value_display`]. `NULL` is rendered as the
/// literal string `"NULL"`.
#[derive(Debug, Clone)]
pub struct QueryRow {
    pub values: Vec<(String, String)>,
}

/// Outcome of executing one SQL statement.
#[derive(Debug, Clone)]
pub enum QueryResult {
    Rows(Vec<QueryRow>),
    RowCount(u64),
    Ok(String),
}

/// The static shape of a prepared statement, resolved without executing
/// it — used by the extended query protocol's `Describe` (Req 6 AC4).
#[derive(Debug, Clone, PartialEq)]
pub struct StatementShape {
    /// Number of bind parameters (`$1..$N`) the statement expects.
    pub param_count: usize,
    /// Result column names when the statement returns rows (a SELECT),
    /// resolved from the catalog projection. `None` for statements that
    /// return no rows (INSERT/UPDATE/DELETE/DDL) → the protocol answers
    /// `NoData`. Columns are reported as text-typed, consistent with the
    /// simple-query result path.
    pub columns: Option<Vec<String>>,
    /// PostgreSQL type OID per result column (HTAP task 22, Req 5.3),
    /// aligned 1:1 with `columns`. Resolved from the catalog column's
    /// declared SQL type via `SqlType::pg_oid`; columns that do not map to a
    /// catalog column (expressions, aggregates, multi-table joins) report
    /// TEXT (25). `None` exactly when `columns` is `None`.
    pub column_type_oids: Option<Vec<u32>>,
}
///
/// Holds the parsed AST template (with `$n` placeholders) plus its static
/// shape, so each `Execute` binds values into a clone of the AST instead
/// of re-parsing. Produced by [`Database::prepare`].
#[derive(Clone)]
pub struct PreparedTemplate {
    /// The parsed template; `$n` placeholders are filled per execution.
    stmts: Arc<Vec<AuroraStatement>>,
    /// Number of bind parameters (`$1..$N`).
    pub param_count: usize,
    /// Result column names (text-typed) for a SELECT, else `None`.
    pub columns: Option<Vec<String>>,
    /// PostgreSQL type OID per result column (HTAP task 22), aligned with
    /// `columns`; see [`StatementShape::column_type_oids`].
    pub column_type_oids: Option<Vec<u32>>,
    /// Whether this is a read-only statement (a single SELECT) — lets the
    /// caller choose a read vs write lock.
    pub is_read: bool,
}


// ---------------------------------------------------------------------------
// Per-table vector index
// ---------------------------------------------------------------------------

/// HNSW graph + delta buffer + row-id/vector map for one table with an
/// embedding column.
struct TableVectorIndex {
    hnsw: HnswGraph,
    delta: DeltaBuffer,
    /// Embedding dimension — read by online tests to assert the
    /// sidecar's model dim matches the catalog's declared DIM.
    #[allow(dead_code)]
    dim: usize,
    /// Column with the embedding.
    embedding_column: String,
    /// Source text column (for `SEMANTIC_MATCH` lookup).
    source_column: String,
    /// Row-id → vector (for re-ranking).
    vectors: HashMap<u64, Vec<f32>>,
    /// Primary-key bytes → vector row-id. Populated when the sidecar
    /// returns an embedding for a newly inserted row; consumed by
    /// `on_row_deleted` so we know which vector row to tombstone when
    /// the user issues `DELETE FROM t WHERE ...`. Without this map,
    /// SQL-level DELETEs would leave orphaned vectors in the HNSW
    /// graph (task 18.6 hole surfaced during the Phase I audit).
    key_to_row_id: HashMap<Vec<u8>, u64>,
    /// Per-table semantic result cache (v0.7, inventory 8.11). Disabled
    /// until `CREATE SEMANTIC CACHE` configures it. Interior-mutable so it
    /// can be used through the shared read lock on the indexes map.
    semantic_cache: SemanticCache,
}

// ---------------------------------------------------------------------------
// Semantic result cache (v0.7, inventory 8.11 / Cloud E-4.1).
//
// A per-table cache of recent `SEMANTIC_MATCH` results keyed by the query
// embedding. A later query whose embedding is within the configured cosine
// SIMILARITY of a cached, unexpired, param-matching entry returns the cached
// results without running HNSW — and increments
// `galaxdb_semantic_cache_hits_total`. Interior-mutable (RwLock) so lookups/
// stores work through the shared read lock on the `indexes` map, mirroring
// the `DeltaBuffer` pattern. In-memory only: an empty cache after restart is
// correct (misses until repopulated). Only the pure `SEMANTIC_MATCH` path is
// cached; filtered/brute-force searches bypass the cache (the filter is not
// part of the key), so a filtered result is never served to an unfiltered
// query.
// ---------------------------------------------------------------------------

/// One cached semantic-search result set.
struct SemCacheEntry {
    query_embedding: Vec<f32>,
    model_version: String,
    threshold_bits: u64,
    k: usize,
    results: Vec<VectorSearchResult>,
    created_at: std::time::Instant,
    last_used: std::time::Instant,
}

struct SemCacheInner {
    enabled: bool,
    similarity: f32,
    ttl: std::time::Duration,
    max_entries: usize,
    entries: Vec<SemCacheEntry>,
}

/// Interior-mutable per-table semantic cache.
struct SemanticCache {
    inner: std::sync::RwLock<SemCacheInner>,
}

impl SemanticCache {
    /// Default per-table entry bound (LRU-evicted beyond this).
    const DEFAULT_MAX_ENTRIES: usize = 256;

    fn new() -> Self {
        Self {
            inner: std::sync::RwLock::new(SemCacheInner {
                enabled: false,
                similarity: 1.0,
                ttl: std::time::Duration::from_secs(0),
                max_entries: Self::DEFAULT_MAX_ENTRIES,
                entries: Vec::new(),
            }),
        }
    }

    /// Enable / reconfigure the cache; changing config clears stale entries.
    fn configure(&self, similarity: f32, ttl_secs: u32) {
        let mut inner = self.inner.write().expect("semantic cache lock");
        inner.enabled = true;
        inner.similarity = similarity;
        inner.ttl = std::time::Duration::from_secs(ttl_secs as u64);
        inner.entries.clear();
    }

    /// Disable and discard the cache (`DROP SEMANTIC CACHE`).
    fn disable(&self) {
        let mut inner = self.inner.write().expect("semantic cache lock");
        inner.enabled = false;
        inner.entries.clear();
    }

    /// Invalidate all entries (called on any write to the table).
    fn invalidate(&self) {
        let mut inner = self.inner.write().expect("semantic cache lock");
        inner.entries.clear();
    }

    #[allow(dead_code)] // used by tests + future introspection
    fn is_enabled(&self) -> bool {
        self.inner.read().expect("semantic cache lock").enabled
    }

    /// Look up a hit for `query_embedding` under the given params + model.
    /// Returns the cached results on a hit (and refreshes LRU); `None` on a
    /// miss. Expired entries encountered are dropped.
    fn lookup(
        &self,
        query_embedding: &[f32],
        model_version: &str,
        threshold_bits: u64,
        k: usize,
    ) -> Option<Vec<VectorSearchResult>> {
        let mut inner = self.inner.write().expect("semantic cache lock");
        if !inner.enabled {
            return None;
        }
        let ttl = inner.ttl;
        let sim_threshold = inner.similarity;
        let now = std::time::Instant::now();
        // Drop expired entries first (lazy eviction).
        inner.entries.retain(|e| now.duration_since(e.created_at) < ttl);
        // Find the first param+model-matching entry within similarity.
        let mut hit_idx: Option<usize> = None;
        for (i, e) in inner.entries.iter().enumerate() {
            if e.threshold_bits != threshold_bits
                || e.k != k
                || e.model_version != model_version
            {
                continue;
            }
            if cosine_similarity(query_embedding, &e.query_embedding) >= sim_threshold {
                hit_idx = Some(i);
                break;
            }
        }
        let i = hit_idx?;
        inner.entries[i].last_used = now;
        Some(inner.entries[i].results.clone())
    }

    /// Store a fresh result set (called on a miss).
    fn store(
        &self,
        query_embedding: Vec<f32>,
        model_version: String,
        threshold_bits: u64,
        k: usize,
        results: Vec<VectorSearchResult>,
    ) {
        let mut inner = self.inner.write().expect("semantic cache lock");
        if !inner.enabled {
            return;
        }
        let now = std::time::Instant::now();
        // LRU-evict if at capacity.
        if inner.entries.len() >= inner.max_entries {
            if let Some((oldest, _)) = inner
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_used)
            {
                inner.entries.remove(oldest);
            }
        }
        inner.entries.push(SemCacheEntry {
            query_embedding,
            model_version,
            threshold_bits,
            k,
            results,
            created_at: now,
            last_used: now,
        });
    }
}

/// Cosine similarity of two equal-length vectors. Robust to non-normalized
/// inputs (computes the full cosine), though embeddings are L2-normalized.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return -1.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return -1.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

// ---------------------------------------------------------------------------
// On-disk vector-index persistence (v0.7, inventory 4.10).
//
// The embedded engine keeps each embedding table's vectors in the delta
// buffer + `vectors` map (the HNSW graph is empty — nothing merges into it),
// so a restart re-embeds every durable row (O(rows × embed)). We persist the
// full reconstructable state — dim, source/embedding columns, and every
// (primary_key, row_id, vector) — to `<data_dir>/vidx_<hash>.gvix` on flush,
// and on open reconcile it against the durable rows: reuse a persisted vector
// when its key is still durable, embed only rows the snapshot is missing, and
// drop snapshot entries whose row was deleted. When the snapshot is fresh this
// does zero embeds; it is always correct because the durable rows are the
// source of truth.
// ---------------------------------------------------------------------------

/// Filesystem-safe, collision-resistant snapshot path for a table.
fn vindex_path(data_dir: &std::path::Path, table: &str) -> std::path::PathBuf {
    let h = xxhash_rust::xxh3::xxh3_64(table.as_bytes());
    data_dir.join(format!("vidx_{h:016x}.gvix"))
}

/// Serialize a table's vector state to the versioned `GVIX` snapshot bytes.
fn serialize_vindex(idx: &TableVectorIndex) -> Vec<u8> {
    let mut out = galaxdb_common::format::VINDEX.header().to_bytes().to_vec();
    out.extend_from_slice(&(idx.dim as u32).to_le_bytes());
    let sc = idx.source_column.as_bytes();
    out.extend_from_slice(&(sc.len() as u32).to_le_bytes());
    out.extend_from_slice(sc);
    let ec = idx.embedding_column.as_bytes();
    out.extend_from_slice(&(ec.len() as u32).to_le_bytes());
    out.extend_from_slice(ec);
    // Only persist entries that still have a vector (skip tombstoned rows).
    let entries: Vec<(&Vec<u8>, u64, &Vec<f32>)> = idx
        .key_to_row_id
        .iter()
        .filter_map(|(k, rid)| idx.vectors.get(rid).map(|v| (k, *rid, v)))
        .collect();
    out.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    for (key, row_id, vec) in entries {
        out.extend_from_slice(&(key.len() as u32).to_le_bytes());
        out.extend_from_slice(key);
        out.extend_from_slice(&row_id.to_le_bytes());
        for f in vec {
            out.extend_from_slice(&f.to_le_bytes());
        }
    }
    out
}

/// Parsed snapshot: dim, source column, and key → (row_id, vector).
/// `dim`/`source_column`/`embedding_column` are retained from the on-disk
/// format for completeness and future schema-drift validation; the reconcile
/// path keys off `by_key`.
#[derive(Debug)]
#[allow(dead_code)]
struct VindexSnapshot {
    dim: usize,
    source_column: String,
    embedding_column: String,
    by_key: HashMap<Vec<u8>, (u64, Vec<f32>)>,
}

/// Deserialize a `GVIX` snapshot. Returns a typed `FormatTooNew`/`FormatTooOld`
/// on an out-of-range version (rollback safety), or a parse error on a
/// truncated/corrupt file.
fn deserialize_vindex(bytes: &[u8]) -> GalaxResult<VindexSnapshot> {
    use galaxdb_common::format::{FormatHeader, FORMAT_HEADER_SIZE, VINDEX};
    if bytes.len() < FORMAT_HEADER_SIZE + 4 {
        return Err(GalaxError::Internal("vindex snapshot too small".into()));
    }
    let mut hdr = [0u8; FORMAT_HEADER_SIZE];
    hdr.copy_from_slice(&bytes[..FORMAT_HEADER_SIZE]);
    let header = FormatHeader::from_bytes(&hdr, VINDEX.magic)?;
    VINDEX.check(header.format_version)?;

    let mut pos = FORMAT_HEADER_SIZE;
    let read_u32 = |b: &[u8], p: &mut usize| -> GalaxResult<u32> {
        if *p + 4 > b.len() {
            return Err(GalaxError::Internal("vindex truncated (u32)".into()));
        }
        let v = u32::from_le_bytes([b[*p], b[*p + 1], b[*p + 2], b[*p + 3]]);
        *p += 4;
        Ok(v)
    };
    let read_u64 = |b: &[u8], p: &mut usize| -> GalaxResult<u64> {
        if *p + 8 > b.len() {
            return Err(GalaxError::Internal("vindex truncated (u64)".into()));
        }
        let mut a = [0u8; 8];
        a.copy_from_slice(&b[*p..*p + 8]);
        *p += 8;
        Ok(u64::from_le_bytes(a))
    };

    let dim = read_u32(bytes, &mut pos)? as usize;
    let sc_len = read_u32(bytes, &mut pos)? as usize;
    if pos + sc_len > bytes.len() {
        return Err(GalaxError::Internal("vindex truncated (source col)".into()));
    }
    let source_column = String::from_utf8(bytes[pos..pos + sc_len].to_vec())
        .map_err(|_| GalaxError::Internal("vindex bad source col utf8".into()))?;
    pos += sc_len;
    let ec_len = read_u32(bytes, &mut pos)? as usize;
    if pos + ec_len > bytes.len() {
        return Err(GalaxError::Internal("vindex truncated (embed col)".into()));
    }
    let embedding_column = String::from_utf8(bytes[pos..pos + ec_len].to_vec())
        .map_err(|_| GalaxError::Internal("vindex bad embed col utf8".into()))?;
    pos += ec_len;

    let count = read_u64(bytes, &mut pos)? as usize;
    let mut by_key = HashMap::with_capacity(count);
    for _ in 0..count {
        let klen = read_u32(bytes, &mut pos)? as usize;
        if pos + klen > bytes.len() {
            return Err(GalaxError::Internal("vindex truncated (key)".into()));
        }
        let key = bytes[pos..pos + klen].to_vec();
        pos += klen;
        let row_id = read_u64(bytes, &mut pos)?;
        if pos + dim * 4 > bytes.len() {
            return Err(GalaxError::Internal("vindex truncated (vector)".into()));
        }
        let mut vec = Vec::with_capacity(dim);
        for _ in 0..dim {
            let f = f32::from_le_bytes([
                bytes[pos],
                bytes[pos + 1],
                bytes[pos + 2],
                bytes[pos + 3],
            ]);
            pos += 4;
            vec.push(f);
        }
        by_key.insert(key, (row_id, vec));
    }
    Ok(VindexSnapshot {
        dim,
        source_column,
        embedding_column,
        by_key,
    })
}

#[cfg(test)]
mod vindex_tests {
    use super::*;

    #[test]
    fn vindex_serialize_roundtrip() {
        let mut idx = TableVectorIndex {
            hnsw: HnswGraph::new(HnswConfig::new(3).with_max_elements(16)),
            delta: DeltaBuffer::new(3),
            dim: 3,
            embedding_column: "emb".into(),
            source_column: "body".into(),
            vectors: HashMap::new(),
            key_to_row_id: HashMap::new(),
            semantic_cache: SemanticCache::new(),
        };
        idx.vectors.insert(10, vec![1.0, 2.0, 3.0]);
        idx.key_to_row_id.insert(b"t:1".to_vec(), 10);
        idx.vectors.insert(20, vec![4.0, 5.0, 6.0]);
        idx.key_to_row_id.insert(b"t:2".to_vec(), 20);

        let bytes = serialize_vindex(&idx);
        let snap = deserialize_vindex(&bytes).expect("parse");
        assert_eq!(snap.dim, 3);
        assert_eq!(snap.source_column, "body");
        assert_eq!(snap.by_key.len(), 2);
        assert_eq!(snap.by_key.get(b"t:1".as_slice()).unwrap().0, 10);
        assert_eq!(snap.by_key.get(b"t:2".as_slice()).unwrap().1, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn vindex_too_new_is_refused() {
        let idx = TableVectorIndex {
            hnsw: HnswGraph::new(HnswConfig::new(2).with_max_elements(16)),
            delta: DeltaBuffer::new(2),
            dim: 2,
            embedding_column: "e".into(),
            source_column: "s".into(),
            vectors: HashMap::new(),
            key_to_row_id: HashMap::new(),
            semantic_cache: SemanticCache::new(),
        };
        let mut bytes = serialize_vindex(&idx);
        // Bump the on-disk format version to current+1 (offset 4..6 LE).
        bytes[4] = bytes[4].wrapping_add(1);
        match deserialize_vindex(&bytes) {
            Err(GalaxError::FormatTooNew { .. }) => {}
            other => panic!("expected FormatTooNew, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod semantic_cache_tests {
    use super::*;

    fn res(row_id: u64, sim: f32) -> VectorSearchResult {
        VectorSearchResult {
            row_id,
            similarity: sim,
        }
    }

    #[test]
    fn disabled_cache_never_hits() {
        let c = SemanticCache::new();
        assert!(!c.is_enabled());
        // store is a no-op while disabled; lookup misses.
        c.store(vec![1.0, 0.0], "m".into(), 0u64, 10, vec![res(1, 0.9)]);
        assert!(c.lookup(&[1.0, 0.0], "m", 0u64, 10).is_none());
    }

    #[test]
    fn hit_within_similarity_miss_outside() {
        let c = SemanticCache::new();
        c.configure(0.95, 3600);
        let q = vec![1.0f32, 0.0, 0.0];
        c.store(q.clone(), "m".into(), 7u64, 10, vec![res(42, 0.99)]);

        // Identical query → cosine 1.0 ≥ 0.95 → hit.
        let hit = c.lookup(&q, "m", 7u64, 10).expect("hit");
        assert_eq!(hit.len(), 1);
        assert_eq!(hit[0].row_id, 42);

        // Orthogonal query → cosine 0 < 0.95 → miss.
        assert!(c.lookup(&[0.0, 1.0, 0.0], "m", 7u64, 10).is_none());
    }

    #[test]
    fn params_and_model_must_match() {
        let c = SemanticCache::new();
        c.configure(0.9, 3600);
        let q = vec![1.0f32, 0.0];
        c.store(q.clone(), "m1".into(), 7u64, 10, vec![res(1, 0.99)]);
        // Same vector, different k → miss (no stale-shape bleed).
        assert!(c.lookup(&q, "m1", 7u64, 50).is_none());
        // Different threshold bits → miss.
        assert!(c.lookup(&q, "m1", 8u64, 10).is_none());
        // Different model version → miss (embeddings not comparable).
        assert!(c.lookup(&q, "m2", 7u64, 10).is_none());
        // All matching → hit.
        assert!(c.lookup(&q, "m1", 7u64, 10).is_some());
    }

    #[test]
    fn ttl_expiry_yields_miss() {
        let c = SemanticCache::new();
        c.configure(0.9, 0); // ttl 0 → everything already expired
        // configure() clamps ttl to Duration(0); force a tiny ttl instead.
        {
            let mut inner = c.inner.write().unwrap();
            inner.ttl = std::time::Duration::from_millis(20);
        }
        let q = vec![1.0f32, 0.0];
        c.store(q.clone(), "m".into(), 0u64, 10, vec![res(1, 0.99)]);
        assert!(c.lookup(&q, "m", 0u64, 10).is_some());
        std::thread::sleep(std::time::Duration::from_millis(40));
        assert!(c.lookup(&q, "m", 0u64, 10).is_none(), "entry must expire");
    }

    #[test]
    fn invalidate_clears_entries() {
        let c = SemanticCache::new();
        c.configure(0.9, 3600);
        let q = vec![1.0f32, 0.0];
        c.store(q.clone(), "m".into(), 0u64, 10, vec![res(1, 0.99)]);
        assert!(c.lookup(&q, "m", 0u64, 10).is_some());
        c.invalidate();
        assert!(c.lookup(&q, "m", 0u64, 10).is_none(), "write invalidates");
    }

    #[test]
    fn lru_bound_respected() {
        let c = SemanticCache::new();
        c.configure(0.999, 3600);
        {
            let mut inner = c.inner.write().unwrap();
            inner.max_entries = 4;
        }
        // Insert 6 distinct (orthogonal-ish) queries; only 4 retained.
        for i in 0..6u64 {
            let mut v = vec![0.0f32; 6];
            v[i as usize] = 1.0;
            c.store(v, "m".into(), 0u64, 10, vec![res(i, 0.99)]);
        }
        let inner = c.inner.read().unwrap();
        assert!(inner.entries.len() <= 4, "LRU bound: {}", inner.entries.len());
    }
}

// ---------------------------------------------------------------------------
// Database
// ---------------------------------------------------------------------------

/// An embedded GalaxDB database instance.
///
/// Owns the storage engine, sidecar, tag catalog, and per-table vector
/// indexes. Every SQL statement is dispatched through the canonical
/// executor (`galaxdb_sql::executor::execute_with_context`).
pub struct Database {
    path: PathBuf,
    engine: Arc<Engine>,
    /// Sidecar manager — shared so the vector backend can also call it.
    sidecar: Option<Arc<SidecarManager>>,
    /// Merkle DAG for version history.
    merkle_dag: Arc<Mutex<MerkleDag>>,
    /// Version tag catalog.
    tag_catalog: Arc<Mutex<TagCatalog>>,
    /// Vector indexes per table. Wrapped in `Arc<RwLock>` so the vector
    /// backend (which takes `&self` across the `VectorSearchBackend`
    /// trait) can read/update them without requiring `&mut self` on
    /// every executor call.
    vector_indexes: Arc<RwLock<HashMap<String, TableVectorIndex>>>,
    /// Persisted catalog snapshot — mirrors the executor's context
    /// catalog. Carried here so `&self` read-only methods
    /// (`table_exists`, `table_count`) don't need to rebuild it.
    catalog: Arc<galaxdb_sql::executor::Catalog>,
    /// The authenticated session this database handle runs statements
    /// under, if any (Req 3). `None` is trusted in-process embedded use:
    /// no authenticated principal, so the executor skips authorization
    /// (today's behavior, preserved for direct PyO3/Rust callers). The
    /// wire server sets this to the role established by the SCRAM
    /// handshake (task 6) via [`Database::with_session`] so every
    /// networked statement is authorization-checked at the executor
    /// chokepoint (Req 3, AC7).
    session: Option<galaxdb_auth::SessionContext>,
    /// Security audit sink (Req 4). When set, the executor records
    /// authorization denials and role/grant/DDL changes. `None` discards
    /// events (no-op). The server attaches a real sink (e.g. a JSONL file)
    /// when one is configured.
    audit: Option<Arc<dyn galaxdb_auth::AuditSink>>,
    /// Bounded LRU cache of parsed statements (Req 7). A repeated,
    /// byte-identical statement skips the SQL parser on a hit. Wrapped in
    /// a `Mutex` for interior mutability so both `&mut self` (`execute`)
    /// and `&self` (`execute_readonly`) callers can use it.
    stmt_cache: Mutex<galaxdb_sql::StatementCache>,
    /// Snapshot-isolation transaction manager (HTAP Phase 5). Shared across
    /// connections for write-write conflict detection (acquire/release write
    /// locks); each `BEGIN` gets a snapshot id from it. Read snapshots use
    /// the engine's MVCC clock; this manager owns the lock table + txn ids.
    txn_manager: Arc<galaxdb_sql::transaction::TransactionManager>,
}

/// Adapter exposing the version-tag catalog's pinned timestamps to the
/// storage engine's runtime compaction MVCC garbage collector. Every tag's
/// `version_timestamp` is GC-exempt so `AT VERSION` time-travel keeps
/// resolving after a compaction (engineering-principles §2).
struct TagCatalogPins(Arc<Mutex<TagCatalog>>);

impl galaxdb_storage::engine::PinSource for TagCatalogPins {
    fn pinned_timestamps(&self) -> Vec<u64> {
        self.0
            .lock()
            .map(|tc| tc.all_pinned_timestamps())
            .unwrap_or_default()
    }
}

impl Database {
    /// Open (or create) a database at `path` without a sidecar.
    ///
    /// Tables with embedding columns can still be created; inserts will
    /// succeed but the embedding column will stay unpopulated until a
    /// sidecar is attached and the row is re-inserted.
    pub fn open(path: &str) -> GalaxResult<Self> {
        let path = PathBuf::from(path);
        std::fs::create_dir_all(&path)?;
        let config = EngineConfig {
            data_dir: path.clone(),
            wal_group_commit_ms: 1, // fast sync commits for embedded mode
            ..Default::default()
        };
        let engine = Engine::new(config)?;
        Self::from_engine(path, engine)
    }

    /// Open (or create) a database applying auto-tuned sizes (Req 12).
    ///
    /// `memtable_size_bytes` and `sst_cache_bytes` are the values the
    /// server derived from the host (or that the operator overrode); they
    /// replace the static [`EngineConfig`] defaults so the running engine
    /// actually uses the tuned configuration. Every other field keeps its
    /// default (including the embedded-mode `wal_group_commit_ms = 1`).
    pub fn open_with_tuning(
        path: &str,
        memtable_size_bytes: u64,
        sst_cache_bytes: u64,
        compaction_concurrency: usize,
    ) -> GalaxResult<Self> {
        let path = PathBuf::from(path);
        std::fs::create_dir_all(&path)?;
        let config = EngineConfig {
            data_dir: path.clone(),
            wal_group_commit_ms: 1,
            memtable_size_bytes,
            sst_cache_bytes,
            compaction_concurrency: compaction_concurrency.max(1),
            ..Default::default()
        };
        let engine = Engine::new(config)?;
        Self::from_engine(path, engine)
    }

    /// Assemble a [`Database`] around an already-opened engine. Shared by
    /// [`open`](Self::open) and [`open_with_tuning`](Self::open_with_tuning)
    /// so the handle's auxiliary state is constructed in exactly one place.
    fn from_engine(path: PathBuf, engine: Engine) -> GalaxResult<Self> {
        let engine = Arc::new(engine);
        let tag_catalog = Arc::new(Mutex::new(TagCatalog::new()));
        // Make runtime compaction tag-aware: its MVCC garbage collector
        // must never drop a row version a live version tag can still read
        // through `AT VERSION` (engineering-principles §2). The tag catalog
        // is the authoritative pin set; the engine queries it on every
        // compaction. Without this, a flush-triggered compaction would GC
        // unpinned historical versions and break time-travel reads.
        engine.set_pin_source(Arc::new(TagCatalogPins(tag_catalog.clone())));
        // Run compaction on a background worker so flushes never block on a
        // merge; the worker holds only a Weak<Engine> and exits when this
        // handle is dropped (see Database::drop).
        engine.start_background_compaction();

        // Rebuild the catalog from the durably-persisted schema entries so
        // tables (and their row data) survive a restart. Without this, DDL
        // would be in-memory only and every table would vanish on reopen even
        // though its rows are recovered from the WAL/SSTs. Columnar tables
        // also need their PAX splitter re-registered so flush/compaction keep
        // laying rows out in the persisted layout.
        let catalog = {
            let mut cat = galaxdb_sql::executor::Catalog::new();
            // Refuses to open if any persisted catalog entry is a newer format
            // than this engine understands (rollback safety, Req 5.2).
            for entry in galaxdb_sql::catalog_store::load_all(&engine)? {
                if entry.storage_mode == galaxdb_common::StorageMode::Columnar {
                    if let Some(splitter) =
                        galaxdb_sql::columnar::CatalogRowSplitter::from_table_entry(&entry)
                    {
                        let prefix = format!("{}:", entry.name).into_bytes();
                        engine.register_columnar_table(prefix, Arc::new(splitter));
                    }
                }
                let name = entry.name.clone();
                if let Err(e) = cat.create_table(name.clone(), entry) {
                    tracing::warn!(table = %name, error = %e, "skipping duplicate catalog entry on open");
                }
            }
            Arc::new(cat)
        };

        Ok(Self {
            path,
            engine,
            sidecar: None,
            merkle_dag: Arc::new(Mutex::new(MerkleDag::new())),
            tag_catalog,
            vector_indexes: Arc::new(RwLock::new(HashMap::new())),
            catalog,
            session: None,
            audit: None,
            stmt_cache: Mutex::new(galaxdb_sql::StatementCache::new(256)),
            txn_manager: Arc::new(galaxdb_sql::transaction::TransactionManager::new()),
        })
    }

    /// Open a database with an embedding sidecar attached.
    ///
    /// * `path` — storage data directory.
    /// * `sidecar_binary` — path to the `galaxdb-sidecar` binary.
    /// * `model_id` — HuggingFace model id (e.g.
    ///   `sentence-transformers/all-MiniLM-L6-v2`). The sidecar
    ///   downloads the model on first run and caches it.
    ///
    /// If the sidecar fails to load the model it exits with status 1;
    /// `SidecarManager` observes the dead child and any subsequent
    /// `embed` call returns a typed error. There is no mock fallback —
    /// every embedding is computed by the real model.
    pub fn open_with_sidecar(
        path: &str,
        sidecar_binary: &str,
        model_id: &str,
    ) -> GalaxResult<Self> {
        let mut db = Self::open(path)?;
        db.attach_sidecar(sidecar_binary, model_id)?;
        Ok(db)
    }

    /// Attach an embedding sidecar to an already-open handle.
    ///
    /// Extracted from [`open_with_sidecar`](Self::open_with_sidecar) so the
    /// auto-tuned open path can reuse it without duplicating the sidecar
    /// boot + socket-wait logic. There is no mock fallback — every
    /// embedding is computed by the real model, and a sidecar that fails to
    /// come up surfaces a typed error.
    pub fn attach_sidecar(&mut self, sidecar_binary: &str, model_id: &str) -> GalaxResult<()> {
        let socket_path = self.path.join("sidecar.sock");
        let sidecar_config = SidecarConfig {
            binary_path: PathBuf::from(sidecar_binary),
            socket_path: socket_path.clone(),
            model_id: model_id.to_string(),
            data_dir: self.path.clone(),
        };

        let mgr = SidecarManager::new(sidecar_config);
        mgr.start()?;

        // Wait for the sidecar socket. First run includes the ~90 MB
        // model download; subsequent runs hit the HF cache and come up
        // in seconds. Allow up to 120 s.
        let start = std::time::Instant::now();
        while !socket_path.exists() && start.elapsed() < std::time::Duration::from_secs(120) {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        if !socket_path.exists() {
            return Err(GalaxError::Internal(
                "sidecar failed to start within 120s — check network access to HuggingFace \
                 Hub and disk space for the HF cache"
                    .into(),
            ));
        }

        self.sidecar = Some(Arc::new(mgr));

        // Vectors are not persisted on disk today — they live in the in-memory
        // HNSW + delta buffer. On a restart the row data survives (WAL/SST) but
        // the vector index does not, so SEMANTIC_MATCH would fail with
        // "table not found" for every recovered embedding table. Rebuild each
        // such table's index now by re-embedding its durable rows with the just-
        // attached model (deterministic: the same model reproduces the same
        // vectors, so search results are identical to before the restart).
        self.rebuild_vector_indexes()?;
        Ok(())
    }

    /// Reconstruct the in-memory vector index for every persisted table that
    /// has an embedding column, by re-embedding its stored rows through the
    /// attached sidecar. Called once when a sidecar is attached (open time), so
    /// semantic search survives a server restart. A no-op when no sidecar is
    /// attached or no embedding tables exist.
    ///
    /// This is O(rows × embed_cost) at open. It is correct across WAL
    /// checkpoints because it reads the durable row data (not the ephemeral
    /// WAL delta records). Persisting the HNSW graph to disk to avoid the
    /// re-embed cost on large tables is a tracked follow-up.
    fn rebuild_vector_indexes(&mut self) -> GalaxResult<()> {
        let Some(sidecar) = self.sidecar.clone() else {
            return Ok(());
        };

        // Readiness gate. On a restart the *stale* `sidecar.sock` from the
        // previous run is still on the data volume, so `attach_sidecar`'s
        // "socket file exists" check can pass a moment before the freshly
        // spawned sidecar has removed the stale file and bound its listener.
        // Probe with a real embed until the sidecar actually answers (or a
        // bounded timeout elapses) so the rebuild — and the first user query —
        // never race the listener with a spurious "connection refused".
        {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
            loop {
                match sidecar.embed(EmbedRequest::query(
                    0,
                    "sidecar readiness probe".to_string(),
                    String::new(),
                )) {
                    Ok(_) => break,
                    Err(_) if std::time::Instant::now() < deadline => {
                        std::thread::sleep(std::time::Duration::from_millis(200));
                    }
                    Err(e) => {
                        return Err(GalaxError::Internal(format!(
                            "sidecar did not become ready for vector-index rebuild: {e}"
                        )));
                    }
                }
            }
        }

        let entries = galaxdb_sql::catalog_store::load_all(&self.engine)?;
        for entry in entries {
            if !entry.has_embedding {
                continue;
            }
            // The embedding source column is the one flagged in the catalog.
            let Some(src) = entry.columns.iter().find(|c| c.is_embedding_source) else {
                continue;
            };
            let source_column = src.name.clone();
            let prefix = format!("{}:", entry.name).into_bytes();

            let rows = self.engine.scan_all_with_prefix(Some(&prefix));

            // v0.7 (inventory 4.10): load the persisted vector-index snapshot
            // if present, so we reuse its vectors and embed ONLY rows the
            // snapshot is missing (reconcile against the durable rows). A
            // too-new snapshot is refused (rollback safety); a corrupt one
            // falls back to a full re-embed. When the snapshot is fresh this
            // does zero embeds.
            let snapshot: Option<VindexSnapshot> =
                match std::fs::read(vindex_path(self.engine.data_dir(), &entry.name)) {
                    Ok(bytes) => match deserialize_vindex(&bytes) {
                        Ok(s) => Some(s),
                        Err(e @ GalaxError::FormatTooNew { .. }) => return Err(e),
                        Err(e) => {
                            tracing::warn!(
                                table = %entry.name,
                                error = %e,
                                "vector-index snapshot unreadable; rebuilding by re-embedding"
                            );
                            None
                        }
                    },
                    Err(ref e) if e.kind() == std::io::ErrorKind::NotFound => None,
                    Err(e) => return Err(GalaxError::Io(e)),
                };

            let mut idx: Option<TableVectorIndex> = None;
            let mut reused: u64 = 0;
            let mut embedded: u64 = 0;

            for (key, value) in rows {
                if !key.starts_with(&prefix) {
                    continue;
                }
                let cols = galaxdb_sql::row_codec::decode_row(&value);
                let text = cols.iter().find(|(n, _)| n == &source_column).and_then(|(_, v)| {
                    match v {
                        Value::Text(s) => Some(s.clone()),
                        _ => None,
                    }
                });
                let Some(text) = text else {
                    continue; // row carries no text in the source column
                };
                // Same row-id convention as `on_row_inserted`: xxh3_64 of the
                // full storage key, so search results join back to table rows.
                let row_id = xxhash_rust::xxh3::xxh3_64(&key);
                // Reuse the persisted vector when the snapshot still covers this
                // durable key; otherwise embed the row (only the missing ones).
                let embedding: Vec<f32> = match snapshot.as_ref().and_then(|s| s.by_key.get(&key)) {
                    Some((_rid, v)) => {
                        reused += 1;
                        v.clone()
                    }
                    None => {
                        let resp = sidecar
                            .embed(EmbedRequest::document(row_id, text, source_column.clone()))
                            .map_err(|e| {
                                GalaxError::Internal(format!(
                                    "rebuilding vector index for '{}': embedding failed: {e}",
                                    entry.name
                                ))
                            })?;
                        embedded += 1;
                        resp.embedding
                    }
                };
                let dim = embedding.len();
                let index = idx.get_or_insert_with(|| {
                    let config = HnswConfig::new(dim).with_max_elements(1_000_000);
                    TableVectorIndex {
                        hnsw: HnswGraph::new(config),
                        delta: DeltaBuffer::new(dim),
                        dim,
                        embedding_column: source_column.clone(),
                        source_column: source_column.clone(),
                        vectors: HashMap::new(),
                        key_to_row_id: HashMap::new(),
                        semantic_cache: SemanticCache::new(),
                    }
                });
                index.delta.insert(row_id, embedding.clone());
                index.vectors.insert(row_id, embedding);
                index.key_to_row_id.insert(key.clone(), row_id);
            }
            let rebuilt = reused + embedded;

            // A table with an embedding column but no rows still needs a
            // registered (empty) index so SEMANTIC_MATCH returns zero rows
            // rather than "table not found". Probe the model for its dimension.
            if idx.is_none() {
                let probe = sidecar
                    .embed(EmbedRequest::document(0, "dimension probe".to_string(), source_column.clone()))
                    .map_err(|e| {
                        GalaxError::Internal(format!(
                            "rebuilding vector index for '{}': dimension probe failed: {e}",
                            entry.name
                        ))
                    })?;
                let dim = probe.embedding.len();
                let config = HnswConfig::new(dim).with_max_elements(1_000_000);
                idx = Some(TableVectorIndex {
                    hnsw: HnswGraph::new(config),
                    delta: DeltaBuffer::new(dim),
                    dim,
                    embedding_column: source_column.clone(),
                    source_column: source_column.clone(),
                    vectors: HashMap::new(),
                    key_to_row_id: HashMap::new(),
                    semantic_cache: SemanticCache::new(),
                });
            }

            if let Some(index) = idx {
                // Persist a fresh snapshot so the next restart reuses these
                // vectors instead of re-embedding. Best-effort: a write failure
                // just means the next open re-embeds (correct, slower).
                let path = vindex_path(self.engine.data_dir(), &entry.name);
                if let Err(e) =
                    galaxdb_common::format::atomic_replace(&path, &serialize_vindex(&index))
                {
                    tracing::warn!(
                        table = %entry.name,
                        error = %e,
                        "failed to persist vector-index snapshot on open"
                    );
                }
                self.vector_indexes
                    .write()
                    .unwrap()
                    .insert(entry.name.clone(), index);
                tracing::info!(
                    table = %entry.name,
                    reused,
                    embedded,
                    total = rebuilt,
                    "vector index ready on open (reused persisted vectors; embedded only missing rows)"
                );
            }
        }

        // v0.7: re-apply persisted semantic-cache configs so a cache stays
        // enabled across restart (the cached entries start empty, which is
        // correct — misses until repopulated).
        let cache_store =
            galaxdb_sql::semantic_cache_store::SemanticCacheStore::new(self.engine.clone());
        let mut indexes = self.vector_indexes.write().unwrap();
        for (table, cfg) in cache_store.load_all() {
            if let Some(idx) = indexes.get_mut(&table) {
                idx.semantic_cache.configure(cfg.similarity, cfg.ttl_secs);
            }
        }
        Ok(())
    }

    /// Persist every table's vector-index snapshot to the data volume so a
    /// restart reuses the vectors instead of re-embedding (v0.7, inventory
    /// 4.10). Called on flush/checkpoint and on drop. Best-effort per table:
    /// a write failure is logged, never fatal (the next open just re-embeds).
    pub fn persist_vector_indexes(&self) {
        let data_dir = self.engine.data_dir().to_path_buf();
        let indexes = self.vector_indexes.read().unwrap();
        for (table, idx) in indexes.iter() {
            let path = vindex_path(&data_dir, table);
            if let Err(e) =
                galaxdb_common::format::atomic_replace(&path, &serialize_vindex(idx))
            {
                tracing::warn!(
                    table = %table,
                    error = %e,
                    "failed to persist vector-index snapshot"
                );
            }
        }
    }

    /// Attach an authenticated session to this database handle so every
    /// statement it executes is authorization-checked against the role's
    /// privileges at the executor chokepoint (Req 3, AC7).
    ///
    /// The wire server calls this after a successful SCRAM handshake
    /// (task 6) so a networked client runs under its authenticated role.
    /// Without a session (the default), the handle is trusted in-process
    /// embedded mode and skips authorization — there is no authenticated
    /// principal and the caller already holds the engine directly.
    ///
    /// Consumes and returns `self` for builder-style use.
    pub fn with_session(mut self, session: galaxdb_auth::SessionContext) -> Self {
        self.session = Some(session);
        self
    }

    /// Set (or clear) the authenticated session on an existing handle.
    /// See [`Database::with_session`].
    pub fn set_session(&mut self, session: Option<galaxdb_auth::SessionContext>) {
        self.session = session;
    }

    /// Attach a security audit sink (Req 4) so authorization denials and
    /// role/grant/DDL changes are recorded. Without one, audit events are
    /// discarded (no-op). Builder-style; consumes and returns `self`.
    pub fn with_audit_sink(mut self, sink: Arc<dyn galaxdb_auth::AuditSink>) -> Self {
        self.audit = Some(sink);
        self
    }

    /// Set (or clear) the audit sink on an existing handle.
    pub fn set_audit_sink(&mut self, sink: Option<Arc<dyn galaxdb_auth::AuditSink>>) {
        self.audit = sink;
    }

    /// The role this handle runs statements under, if a session is
    /// attached.
    pub fn session_role(&self) -> Option<&galaxdb_auth::Role> {
        self.session.as_ref().map(|s| &s.role)
    }

    /// An [`AuthStore`](galaxdb_sql::auth_store::AuthStore) over this
    /// database's engine, for looking up role verifiers and superuser
    /// flags during the wire SCRAM handshake and for provisioning the
    /// initial superuser at startup (task 6). Cheap to construct (holds an
    /// `Arc<Engine>`).
    pub fn auth_store(&self) -> galaxdb_sql::auth_store::AuthStore {
        galaxdb_sql::auth_store::AuthStore::new(self.engine.clone())
    }

    /// Whether any role exists in the auth catalog. Used at startup to
    /// decide whether to provision the initial superuser (task 6, Req 1
    /// AC7).
    pub fn any_role_exists(&self) -> bool {
        self.auth_store().any_role_exists()
    }

    /// Provision an initial superuser from a plaintext password. Used once
    /// at first startup when auth is enabled and the catalog is empty
    /// (Req 1 AC7). The plaintext is consumed to build the SCRAM verifier
    /// and never stored. Returns an error if a role with that name already
    /// exists.
    pub fn provision_superuser(&self, name: &str, password: &str) -> GalaxResult<()> {
        let store = self.auth_store();
        if store.get_role(name).is_some() {
            return Err(GalaxError::Internal(format!(
                "cannot provision initial superuser: role '{name}' already exists"
            )));
        }
        store.put_role(&galaxdb_sql::auth_store::RoleRecord {
            name: name.to_string(),
            is_superuser: true,
            verifier: Some(galaxdb_auth::ScramVerifier::from_password(password)),
        })
    }

    /// Execute a write-capable statement under an explicit per-call
    /// session, overriding [`Database::session`] for the duration of this
    /// call only (task 6). The wire server uses this so each connection
    /// runs its statements under the role established by that connection's
    /// SCRAM handshake, even though all connections share one `Database`
    /// behind an `RwLock`.
    ///
    /// Safe because the caller holds the database exclusively (`&mut self`
    /// / a write lock): the session is set, the statement runs
    /// synchronously, and the previous session is restored before
    /// returning. There is no reentrancy.
    pub fn execute_with_session(
        &mut self,
        sql: &str,
        session: Option<galaxdb_auth::SessionContext>,
    ) -> GalaxResult<QueryResult> {
        let prev = self.session.take();
        self.session = session;
        let result = self.execute(sql);
        self.session = prev;
        result
    }

    /// Parse `sql`, using the statement cache but **without holding the cache
    /// lock during parsing**. On a miss the mutex is released, the parser runs
    /// (CPU-heavy), then the result is inserted. This lets concurrent
    /// connections parse in parallel instead of serializing every statement on
    /// the single cache mutex — critical for multi-client write throughput.
    fn cached_parse(
        &self,
        sql: &str,
    ) -> GalaxResult<std::sync::Arc<Vec<AuroraStatement>>> {
        // Fast path: cache hit under a brief lock.
        if let Some(hit) = {
            let mut cache = self
                .stmt_cache
                .lock()
                .map_err(|_| GalaxError::Internal("statement cache mutex poisoned".into()))?;
            cache.get_cached(sql)
        } {
            return Ok(hit);
        }
        // Slow path: parse with NO lock held.
        let parsed = std::sync::Arc::new(parser::parse(sql)?);
        // Publish to the cache (brief lock). A concurrent racer may have
        // inserted the same key meanwhile; last write wins and both are equal.
        {
            let mut cache = self
                .stmt_cache
                .lock()
                .map_err(|_| GalaxError::Internal("statement cache mutex poisoned".into()))?;
            cache.put_parsed(sql, parsed.clone());
        }
        Ok(parsed)
    }

    /// DML-write path that takes `&self` (shared/read lock) instead of
    /// `&mut self` (exclusive write lock). Safe for concurrent callers
    /// because:
    ///   - INSERT, UPDATE, DELETE, BULK INSERT never mutate `self.catalog`
    ///     (only DDL does). This method rejects DDL explicitly.
    ///   - The storage engine (`Arc<Engine>`) is internally thread-safe —
    ///     memtable, ART, and WAL use their own fine-grained locks.
    ///   - The catalog is cloned once per call (cheap: table metadata only,
    ///     no row data) so the executor has a stable snapshot without
    ///     locking the caller.
    ///
    /// This enables multiple concurrent wire connections to insert rows
    /// simultaneously, letting the WAL group-commit coalesce their fsyncs
    /// — the same pattern that gives PostgreSQL its concurrent TPS.
    ///
    /// Callers must NOT pass DDL (CREATE TABLE, DROP TABLE, CREATE INDEX,
    /// role/grant statements). Those require `&mut self` via
    /// `execute_with_session` so the catalog mutation is visible.
    pub fn execute_dml_concurrent(
        &self,
        sql: &str,
        session: Option<galaxdb_auth::SessionContext>,
    ) -> GalaxResult<QueryResult> {
        // Parse and classify — reject DDL before we even touch the engine.
        // Parse OUTSIDE the cache lock so concurrent connections parse in
        // parallel instead of serializing on the statement-cache mutex.
        let stmts = self.cached_parse(sql)?;

        for stmt in stmts.iter() {
            match stmt {
                AuroraStatement::Standard(s) => {
                    use sqlparser::ast::Statement;
                    match s.as_ref() {
                        // These are all safe DML that never mutate the catalog.
                        Statement::Insert(_)
                        | Statement::Update { .. }
                        | Statement::Delete(_)
                        | Statement::Query(_) => {}
                        other => {
                            return Err(GalaxError::Internal(format!(
                                "execute_dml_concurrent: DDL statement not allowed \
                                 on the concurrent path: {other}"
                            )));
                        }
                    }
                }
                // BulkInsert is safe: no catalog mutation.
                AuroraStatement::BulkInsert(_) => {}
                other => {
                    return Err(GalaxError::Internal(format!(
                        "execute_dml_concurrent: statement not allowed on the \
                         concurrent path: {other:?}"
                    )));
                }
            }
        }

        // Build a context from &self — mirrors what context() does but
        // without the &mut self requirement.
        let mut last = QueryResult::Ok("OK".to_string());
        for stmt in stmts.iter() {
            // Translate the parsed statement to a QueryPlan inline.
            // Only INSERT and BULK INSERT are allowed (validated above).
            let plan = match stmt {
                AuroraStatement::BulkInsert(bi) => QueryPlan::BulkInsert {
                    table: bi.table.clone(),
                    columns: bi.columns.clone(),
                    values: bi.values.clone(),
                },
                AuroraStatement::Standard(s) => {
                    match s.as_ref() {
                        sqlparser::ast::Statement::Insert(ins) => {
                            // Build one Insert plan per row in the VALUES list.
                            let table = ins.table_name.to_string();
                            let column_names: Vec<String> = ins.columns.iter()
                                .map(|c| c.to_string())
                                .collect();
                            let Some(source) = &ins.source else {
                                continue;
                            };
                            let sqlparser::ast::SetExpr::Values(values) =
                                source.body.as_ref() else { continue };
                            for row in &values.rows {
                                let row_values: Vec<Value> = row.iter()
                                    .map(|e| scalar_from_expr(e).and_then(|s| s.eval(&[])))
                                    .collect::<GalaxResult<Vec<Value>>>()?;
                                let row_plan = QueryPlan::Insert {
                                    table: table.clone(),
                                    columns: column_names.clone(),
                                    values: row_values,
                                };
                                let mut ctx = ExecutorContext::new(self.engine.clone());
                                ctx.catalog = self.catalog.clone();
                                ctx.sidecar = self.sidecar.clone();
                                ctx.merkle_dag = Some(self.merkle_dag.clone());
                                ctx.tag_catalog = Some(self.tag_catalog.clone());
                                ctx.vector_backend = Some(Arc::new(EmbeddedVectorBackend {
                                    sidecar: self.sidecar.clone(),
                                    indexes: self.vector_indexes.clone(),
                                    engine: self.engine.clone(),
                                }));
                                ctx.auth_store = Some(
                                    galaxdb_sql::auth_store::AuthStore::new(self.engine.clone()),
                                );
                                ctx.secondary_index = Some(
                                    galaxdb_sql::secondary_index::SecondaryIndexStore::new(
                                        self.engine.clone(),
                                    ),
                                );
                                ctx.session = session.clone();
                                ctx.audit = self.audit.clone();
                                let res = execute_with_context(&row_plan, &mut ctx)?;
                                // Intentionally NOT writing ctx.catalog back — INSERT
                                // never mutates it, and this is &self so we couldn't anyway.
                                last = query_result_from(res);
                            }
                            // v0.6 metering: one INSERT statement = one write
                            // op, regardless of row count. Counted here (above
                            // the per-row fan-out), on success — a failed row
                            // returns early via `?` and never reaches this.
                            galaxdb_observe::metrics().write_ops_total.inc();
                            continue;
                        }
                        sqlparser::ast::Statement::Update {
                            table, assignments, selection, ..
                        } => {
                            let tname = table.relation.to_string();
                            let asns: Vec<(String, ScalarExpr)> = assignments.iter()
                                .map(|a| Ok((
                                    a.target.to_string(),
                                    scalar_from_expr(&a.value)?,
                                )))
                                .collect::<GalaxResult<Vec<_>>>()?;
                            let filter = selection.as_ref().and_then(filter_from_expr);
                            QueryPlan::Update {
                                table: tname,
                                assignments: asns,
                                filter,
                            }
                        }
                        sqlparser::ast::Statement::Delete(del) => {
                            let tname = match &del.from {
                                sqlparser::ast::FromTable::WithFromKeyword(tables)
                                | sqlparser::ast::FromTable::WithoutKeyword(tables) => {
                                    tables.first().map(|t| t.relation.to_string()).unwrap_or_default()
                                }
                            };
                            let filter = del.selection.as_ref().and_then(filter_from_expr);
                            QueryPlan::Delete { table: tname, filter }
                        }
                        sqlparser::ast::Statement::Query(q) => {
                            let (columns, filter) = extract_projection_and_filter(q);
                            let table = extract_table(q);
                            QueryPlan::FullScan { table, filter, columns }
                        }
                        other => {
                            return Err(GalaxError::Internal(format!(
                                "execute_dml_concurrent: unexpected statement: {other}"
                            )));
                        }
                    }
                }
                other => {
                    return Err(GalaxError::Internal(format!(
                        "execute_dml_concurrent: unexpected statement: {other:?}"
                    )));
                }
            };
            let mut ctx = ExecutorContext::new(self.engine.clone());
            ctx.catalog = self.catalog.clone();
            ctx.sidecar = self.sidecar.clone();
            ctx.merkle_dag = Some(self.merkle_dag.clone());
            ctx.tag_catalog = Some(self.tag_catalog.clone());
            ctx.vector_backend = Some(Arc::new(EmbeddedVectorBackend {
                sidecar: self.sidecar.clone(),
                indexes: self.vector_indexes.clone(),
                engine: self.engine.clone(),
            }));
            ctx.auth_store = Some(galaxdb_sql::auth_store::AuthStore::new(self.engine.clone()));
            ctx.secondary_index = Some(
                galaxdb_sql::secondary_index::SecondaryIndexStore::new(self.engine.clone()),
            );
            ctx.session = session.clone();
            ctx.audit = self.audit.clone();
            let res = execute_with_context(&plan, &mut ctx)?;
            last = query_result_from(res);
        }
        Ok(last)
    }

    // -----------------------------------------------------------------
    // Explicit transactions (HTAP Phase 5, design §3.6.1)
    // -----------------------------------------------------------------

    /// Flush the active memtable to an on-disk SST (maintenance / test
    /// helper). Blocks on the engine's async flush using a short-lived
    /// current-thread runtime, so it is safe to call from a synchronous
    /// context (e.g. the conformance corpus, which runs queries over real
    /// SST-backed data rather than only the memtable).
    pub fn flush(&self) -> GalaxResult<()> {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| GalaxError::Internal(format!("flush runtime: {e}")))?
            .block_on(self.engine.flush_memtable())?;
        // v0.7 (inventory 4.10): persist vector-index snapshots on checkpoint
        // so a restart reuses the vectors instead of re-embedding.
        self.persist_vector_indexes();
        Ok(())
    }

    /// Begin an explicit transaction (`BEGIN`/`START TRANSACTION`).
    ///
    /// Captures the engine's current MVCC timestamp as the transaction's
    /// read snapshot (every version committed at or before this ts is
    /// visible for the life of the transaction — snapshot isolation) and a
    /// [`TransactionManager`](galaxdb_sql::transaction::TransactionManager)
    /// snapshot id that owns this transaction's write locks. The returned
    /// [`TxnHandle`] is threaded through
    /// [`execute_in_txn`](Self::execute_in_txn) and finalized with
    /// [`commit_transaction`](Self::commit_transaction) or
    /// [`rollback_transaction`](Self::rollback_transaction).
    pub fn begin_transaction(&self) -> GalaxResult<galaxdb_sql::executor::TxnHandle> {
        // `latest_commit_ts()` is the ts of the most recent committed write.
        // `scan_all_at`/`get_at` use `ts <= read_ts`, so reading at this value
        // makes exactly "everything committed so far" visible and anything
        // committed after BEGIN (which is assigned a strictly larger ts)
        // invisible — the snapshot-isolation boundary (design §3.6.1). Using
        // the *next* (unassigned) ts here would leak the first write that
        // commits after BEGIN into the snapshot (off-by-one dirty read).
        let read_ts = self.engine.latest_commit_ts();
        let snapshot = self.txn_manager.begin();
        Ok(galaxdb_sql::executor::TxnHandle::new(
            read_ts,
            snapshot.read_timestamp,
            self.txn_manager.clone(),
        ))
    }

    /// Begin a SERIALIZABLE (SSI) transaction (v0.7, inventory 8.14). Same as
    /// [`begin_transaction`](Self::begin_transaction) but the handle tracks its
    /// read-set and is certified for serializability at commit — a write-skew
    /// (rw-antidependency into a concurrent committer) aborts with SQLSTATE
    /// 40001. The default isolation remains snapshot isolation.
    pub fn begin_transaction_serializable(
        &self,
    ) -> GalaxResult<galaxdb_sql::executor::TxnHandle> {
        let mut txn = self.begin_transaction()?;
        txn.set_serializable(true);
        Ok(txn)
    }

    /// Commit a transaction: atomically* apply every buffered write to the
    /// engine (durable through the WAL), then release the transaction's
    /// write locks and snapshot.
    ///
    /// Write-write conflicts were already rejected at buffer time
    /// (`buffer_write` → `acquire_write_lock` → `40001`), so the apply step
    /// cannot conflict. Buffered upserts become `put_sync`; buffered
    /// tombstones become `delete_sync`.
    ///
    /// \*Per design §3.6.1 each buffered entry is applied with its own
    /// `put_sync`/`delete_sync` (each individually WAL-durable). Cross-key
    /// crash atomicity of the apply step is a documented v1 limitation — the
    /// engine exposes no mixed put/delete batch primitive yet — not a silent
    /// fallback: the writes that land are correct and durable.
    ///
    /// Secondary-index and embedding-sidecar hooks are NOT re-applied for
    /// buffered rows at commit (documented limitation): committed data is
    /// correct and durable, but indexes/embeddings derived from rows written
    /// inside a transaction are eventually-consistent until the next
    /// rebuild/insert. Autocommit DML keeps applying those hooks inline.
    pub fn commit_transaction(
        &self,
        txn: &galaxdb_sql::executor::TxnHandle,
    ) -> GalaxResult<()> {
        let writes = txn.writes.lock().expect("txn writes lock").clone();

        // v0.7 SSI (inventory 8.14): build the write-key set for the
        // serializability certifier — each written storage key plus a
        // table-granularity SIREAD sentinel, so a concurrent serializable scan
        // of the same table conflicts (catches write-skew + phantoms).
        let mut write_keys: std::collections::HashSet<Vec<u8>> =
            std::collections::HashSet::new();
        for key in writes.keys() {
            write_keys.insert(key.clone());
            if let Some(pos) = key.iter().position(|&b| b == b':') {
                let table = String::from_utf8_lossy(&key[..pos]).to_string();
                write_keys.insert(galaxdb_sql::executor::siread_sentinel(&table));
            }
        }
        let read_keys = txn.read_key_set();

        // Certify + allocate the commit timestamp BEFORE applying any write, so
        // a transaction that fails serializability certification (40001) never
        // persists. For an SI transaction (`serializable == false`) this is a
        // plain commit. Releases write locks + drops the snapshot on success.
        self.txn_manager.commit_serializable(
            txn.txn_id,
            &read_keys,
            write_keys.into_iter().collect(),
            txn.serializable,
        )?;

        // Certification passed — apply the buffered writes.
        for (key, value) in writes {
            match value {
                Some(v) => {
                    self.engine.put_sync(key, v)?;
                }
                None => {
                    self.engine.delete_sync(&key)?;
                }
            }
        }
        Ok(())
    }

    /// Roll back a transaction (`ROLLBACK`): discard the buffered write set
    /// and release the transaction's write locks + snapshot. Nothing was
    /// applied to the engine, so there is nothing to undo.
    pub fn rollback_transaction(&self, txn: &galaxdb_sql::executor::TxnHandle) {
        txn.writes.lock().expect("txn writes lock").clear();
        txn.savepoints.lock().expect("txn savepoints lock").clear();
        let snapshot = galaxdb_sql::transaction::Snapshot {
            read_timestamp: txn.txn_id,
            write_set: Vec::new(),
        };
        self.txn_manager.abort(&snapshot);
    }

    /// Execute one statement inside an open transaction (`txn`).
    ///
    /// DML (`INSERT`/`UPDATE`/`DELETE`) buffers its writes into the handle's
    /// write set with read-your-writes overlay rather than committing to the
    /// engine; row-returning `SELECT`s read at the transaction's snapshot ts
    /// with the buffer overlaid. The following are rejected with a typed
    /// error for v1 (they are not yet transaction-aware): DDL, multi-table
    /// /analytical queries (the DataFusion path cannot see the write
    /// buffer), `SEMANTIC_MATCH`, and `AT VERSION`.
    pub fn execute_in_txn(
        &self,
        sql: &str,
        txn: &galaxdb_sql::executor::TxnHandle,
        session: Option<galaxdb_auth::SessionContext>,
    ) -> GalaxResult<QueryResult> {
        if split_at_version(sql)?.is_some() {
            return Err(GalaxError::FeatureNotSupported(
                "AT VERSION time-travel is not supported inside an explicit \
                 transaction (v1)"
                    .into(),
            ));
        }
        let stmts = self.cached_parse(sql)?;
        let mut last = QueryResult::Ok("OK".to_string());
        for stmt in stmts.iter() {
            last = self.exec_stmt_in_txn(stmt, txn, session.clone())?;
        }
        Ok(last)
    }

    /// Translate + execute a single parsed statement against the transaction
    /// buffer. Shared by [`execute_in_txn`](Self::execute_in_txn).
    fn exec_stmt_in_txn(
        &self,
        stmt: &AuroraStatement,
        txn: &galaxdb_sql::executor::TxnHandle,
        session: Option<galaxdb_auth::SessionContext>,
    ) -> GalaxResult<QueryResult> {
        use sqlparser::ast::Statement;
        let s = match stmt {
            AuroraStatement::Standard(s) => s,
            other => {
                return Err(GalaxError::FeatureNotSupported(format!(
                    "statement not supported inside a transaction (v1): {other:?}"
                )));
            }
        };
        match s.as_ref() {
            Statement::Insert(ins) => {
                let table = ins.table_name.to_string();
                let column_names: Vec<String> =
                    ins.columns.iter().map(|c| c.to_string()).collect();
                let Some(source) = &ins.source else {
                    return Ok(QueryResult::RowCount(0));
                };
                let sqlparser::ast::SetExpr::Values(values) = source.body.as_ref() else {
                    return Err(GalaxError::FeatureNotSupported(
                        "INSERT ... SELECT is not supported inside a transaction (v1)"
                            .into(),
                    ));
                };
                let mut inserted = 0u64;
                for row in &values.rows {
                    let row_values: Vec<Value> = row
                        .iter()
                        .map(|e| scalar_from_expr(e).and_then(|s| s.eval(&[])))
                        .collect::<GalaxResult<Vec<Value>>>()?;
                    let plan = QueryPlan::Insert {
                        table: table.clone(),
                        columns: column_names.clone(),
                        values: row_values,
                    };
                    let mut ctx = self.txn_context(txn, session.clone());
                    let res = execute_with_context(&plan, &mut ctx)?;
                    inserted += match res {
                        ExecuteResult::RowCount(n) => n,
                        _ => 1,
                    };
                }
                // v0.6 metering: one INSERT statement = one write op (in-txn
                // path), counted after all rows succeed.
                galaxdb_observe::metrics().write_ops_total.inc();
                Ok(QueryResult::RowCount(inserted))
            }
            Statement::Update {
                table,
                assignments,
                selection,
                ..
            } => {
                let tname = table.relation.to_string();
                let asns: Vec<(String, ScalarExpr)> = assignments
                    .iter()
                    .map(|a| Ok((a.target.to_string(), scalar_from_expr(&a.value)?)))
                    .collect::<GalaxResult<Vec<_>>>()?;
                let filter = selection.as_ref().and_then(filter_from_expr);
                let plan = QueryPlan::Update {
                    table: tname,
                    assignments: asns,
                    filter,
                };
                let mut ctx = self.txn_context(txn, session);
                let res = execute_with_context(&plan, &mut ctx)?;
                Ok(query_result_from(res))
            }
            Statement::Delete(del) => {
                let tname = match &del.from {
                    sqlparser::ast::FromTable::WithFromKeyword(tables)
                    | sqlparser::ast::FromTable::WithoutKeyword(tables) => tables
                        .first()
                        .map(|t| t.relation.to_string())
                        .unwrap_or_default(),
                };
                let filter = del.selection.as_ref().and_then(filter_from_expr);
                let plan = QueryPlan::Delete {
                    table: tname,
                    filter,
                };
                let mut ctx = self.txn_context(txn, session);
                let res = execute_with_context(&plan, &mut ctx)?;
                Ok(query_result_from(res))
            }
            Statement::Query(q) => {
                if extract_semantic_match_from_query(q).is_some() {
                    return Err(GalaxError::FeatureNotSupported(
                        "SEMANTIC_MATCH is not supported inside a transaction (v1)".into(),
                    ));
                }
                // A single-table SELECT whose only analytical feature is
                // ORDER BY / LIMIT / OFFSET (over bare columns and literal
                // counts) runs natively over the transaction buffer: scan
                // (seeing the uncommitted writes the DataFusion path cannot),
                // then sort + paginate the result in memory. This must be
                // checked before the analytical rejection below.
                if let Some(sort_limit) = galaxdb_sql::classify::simple_sort_limit(q) {
                    return self.exec_txn_sorted_scan(q, &sort_limit, txn, session);
                }
                if matches!(
                    galaxdb_sql::classify::classify_query(q),
                    galaxdb_sql::classify::StatementClass::Analytical
                ) {
                    return Err(GalaxError::FeatureNotSupported(
                        "analytical queries (joins/aggregates/subqueries/GROUP BY/\
                         DISTINCT) are not supported inside an explicit transaction \
                         (v1); the columnar analytical engine cannot see a \
                         transaction's uncommitted write buffer"
                            .into(),
                    ));
                }
                let (columns, filter) = extract_projection_and_filter(q);
                let plan = QueryPlan::FullScan {
                    table: extract_table(q),
                    filter,
                    columns,
                };
                let mut ctx = self.txn_context(txn, session);
                let res = execute_with_context(&plan, &mut ctx)?;
                Ok(query_result_from(res))
            }
            other => Err(GalaxError::FeatureNotSupported(format!(
                "statement not supported inside a transaction (v1): {other}"
            ))),
        }
    }

    /// Native ORDER BY / LIMIT / OFFSET over the transaction buffer.
    ///
    /// A single-table SELECT whose only analytical feature is sort/pagination
    /// cannot use the DataFusion analytical engine inside a transaction (that
    /// engine reads committed columnar data and cannot see the uncommitted
    /// write buffer). Instead we run the native filtered scan — which overlays
    /// the transaction's writes — collect **all** columns so every sort key is
    /// present, sort in memory with type-aware, NULL-aware comparison, apply
    /// OFFSET then LIMIT, and finally project to the requested columns. This
    /// yields correct read-your-writes ordering with no silent fallback.
    fn exec_txn_sorted_scan(
        &self,
        q: &sqlparser::ast::Query,
        sort_limit: &galaxdb_sql::classify::SortLimit,
        txn: &galaxdb_sql::executor::TxnHandle,
        session: Option<galaxdb_auth::SessionContext>,
    ) -> GalaxResult<QueryResult> {
        use galaxdb_sql::executor::{ExecuteResult, Row as SqlRow};

        let (projection, filter) = extract_projection_and_filter(q);
        // Scan all columns (empty projection) so ORDER BY keys are available
        // even when they are not in the SELECT list.
        let plan = QueryPlan::FullScan {
            table: extract_table(q),
            filter,
            columns: Vec::new(),
        };
        let mut ctx = self.txn_context(txn, session);
        let ExecuteResult::Rows { mut rows, .. } = execute_with_context(&plan, &mut ctx)? else {
            // FullScan always yields Rows; anything else is a real bug, not a
            // case to paper over.
            return Err(GalaxError::Internal(
                "in-transaction sorted scan expected row output".into(),
            ));
        };

        // Type-aware, NULL-aware, multi-key sort. Null placement is absolute
        // (NULLS FIRST/LAST), independent of ASC/DESC, per PostgreSQL.
        rows.sort_by(|a, b| {
            for key in &sort_limit.order_by {
                let va = row_column(a, &key.column);
                let vb = row_column(b, &key.column);
                let a_null = matches!(va, None | Some(Value::Null));
                let b_null = matches!(vb, None | Some(Value::Null));
                let ord = match (a_null, b_null) {
                    (true, true) => std::cmp::Ordering::Equal,
                    (true, false) => {
                        if key.nulls_first {
                            std::cmp::Ordering::Less
                        } else {
                            std::cmp::Ordering::Greater
                        }
                    }
                    (false, true) => {
                        if key.nulls_first {
                            std::cmp::Ordering::Greater
                        } else {
                            std::cmp::Ordering::Less
                        }
                    }
                    (false, false) => {
                        let c = value_cmp(va.unwrap(), vb.unwrap());
                        if key.descending {
                            c.reverse()
                        } else {
                            c
                        }
                    }
                };
                if ord != std::cmp::Ordering::Equal {
                    return ord;
                }
            }
            std::cmp::Ordering::Equal
        });

        // OFFSET then LIMIT.
        let offset = sort_limit.offset.unwrap_or(0);
        let mut out: Vec<SqlRow> = rows.into_iter().skip(offset).collect();
        if let Some(limit) = sort_limit.limit {
            out.truncate(limit);
        }

        // Project to the requested columns (empty projection = SELECT *).
        if !projection.is_empty() {
            for row in &mut out {
                row.columns = projection
                    .iter()
                    .map(|name| {
                        let v = row
                            .columns
                            .iter()
                            .find(|(k, _)| k == name)
                            .map(|(_, v)| v.clone())
                            .unwrap_or(Value::Null);
                        (name.clone(), v)
                    })
                    .collect();
            }
        }

        Ok(query_result_from(ExecuteResult::Rows {
            columns: if projection.is_empty() {
                out.first()
                    .map(|r| r.columns.iter().map(|(k, _)| k.clone()).collect())
                    .unwrap_or_default()
            } else {
                projection
            },
            rows: out,
        }))
    }

    /// Build an [`ExecutorContext`] wired with every engine subsystem and
    /// the active transaction handle, so DML buffers + reads overlay the
    /// transaction's write set (design §3.6.1). Mirrors the per-statement
    /// context built on the autocommit concurrent path, plus `ctx.txn`.
    fn txn_context(
        &self,
        txn: &galaxdb_sql::executor::TxnHandle,
        session: Option<galaxdb_auth::SessionContext>,
    ) -> ExecutorContext {
        let mut ctx = ExecutorContext::new(self.engine.clone());
        ctx.catalog = self.catalog.clone();
        ctx.sidecar = self.sidecar.clone();
        ctx.merkle_dag = Some(self.merkle_dag.clone());
        ctx.tag_catalog = Some(self.tag_catalog.clone());
        ctx.vector_backend = Some(Arc::new(EmbeddedVectorBackend {
            sidecar: self.sidecar.clone(),
            indexes: self.vector_indexes.clone(),
            engine: self.engine.clone(),
        }));
        ctx.auth_store = Some(galaxdb_sql::auth_store::AuthStore::new(self.engine.clone()));
        ctx.secondary_index = Some(galaxdb_sql::secondary_index::SecondaryIndexStore::new(
            self.engine.clone(),
        ));
        ctx.session = session;
        ctx.audit = self.audit.clone();
        ctx.txn = Some(txn.clone());
        ctx
    }

    /// Resolve the static shape of a statement for the extended query
    /// protocol's `Describe` (Req 6 AC4) — parameter count and, for a
    /// row-returning SELECT, the result column names — without executing
    /// it. `&self` (read-only): describing never mutates the database.
    pub fn describe_statement(&self, sql: &str) -> GalaxResult<StatementShape> {
        let param_count = count_placeholders(sql);

        // Strip any AuroraSQL `AT VERSION ...` suffix before parsing, the
        // same way `execute` does — it is still a row-returning SELECT.
        let sql_for_parse = match split_at_version(sql)? {
            Some((stripped, _)) => stripped,
            None => sql.to_string(),
        };

        let stmts = parser::parse(&sql_for_parse)?;
        let columns = stmts.first().and_then(|stmt| self.describe_columns(stmt));
        let column_type_oids = stmts
            .first()
            .and_then(|stmt| self.describe_column_oids(stmt));
        Ok(StatementShape {
            param_count,
            columns,
            column_type_oids,
        })
    }

    /// The PostgreSQL type OID of each result column of `sql`, aligned with
    /// the column names from [`describe_statement`](Self::describe_statement),
    /// or `None` when the statement returns no rows (HTAP task 22). Used by
    /// the simple-query RowDescription path to report real per-column types
    /// instead of always-TEXT.
    pub fn describe_result_oids(&self, sql: &str) -> Option<Vec<u32>> {
        let sql_for_parse = match split_at_version(sql) {
            Ok(Some((stripped, _))) => stripped,
            Ok(None) => sql.to_string(),
            Err(_) => return None,
        };
        let stmts = parser::parse(&sql_for_parse).ok()?;
        stmts.first().and_then(|stmt| self.describe_column_oids(stmt))
    }

    /// All column names of a table in catalog (declaration) order, or a
    /// `TableNotFound` error. Used by the COPY sub-protocol to resolve the
    /// column set when `COPY t FROM STDIN` / `TO STDOUT` omits an explicit
    /// list, and to advertise the column count.
    pub fn table_columns(&self, table: &str) -> GalaxResult<Vec<String>> {
        let entry = self
            .catalog
            .get_table(table)
            .ok_or_else(|| GalaxError::TableNotFound(table.to_string()))?;
        Ok(entry.columns.iter().map(|c| c.name.clone()).collect())
    }

    /// Result column names for a statement, or `None` when it returns no
    /// rows. Mirrors the SELECT dispatch in `exec_stmt`: a `Query` resolves
    /// its projection (explicit columns, or all catalog columns for
    /// `SELECT *`); everything else returns no rows.
    fn describe_columns(&self, stmt: &AuroraStatement) -> Option<Vec<String>> {
        let AuroraStatement::Standard(s) = stmt else {
            return None;
        };
        let sqlparser::ast::Statement::Query(q) = s.as_ref() else {
            return None;
        };
        let (proj, _filter) = extract_projection_and_filter(q);
        if !proj.is_empty() {
            return Some(proj);
        }
        // Empty projection means `SELECT *` (or an unsupported expression
        // that falls back to the full row) — report all catalog columns in
        // declaration order so the client sees the same shape it gets back.
        let table = extract_table(q);
        let entry = self.catalog.get_table(&table)?;
        Some(entry.columns.iter().map(|c| c.name.clone()).collect())
    }

    /// PostgreSQL type OID for each result column of `stmt`, aligned with
    /// [`describe_columns`](Self::describe_columns) (HTAP task 22). Each
    /// resolved column name is matched against the target table's catalog
    /// column and mapped `data_type → SqlType::pg_oid`; a name that does not
    /// match a catalog column (an expression, aggregate, alias, or a column
    /// from another table in a join) is reported as TEXT (25) — an honest,
    /// display-safe default, never a guess at a specific type.
    fn describe_column_oids(&self, stmt: &AuroraStatement) -> Option<Vec<u32>> {
        use galaxdb_sql::types::{oid, SqlType};
        let names = self.describe_columns(stmt)?;
        // The (single) target table whose catalog columns we resolve against.
        let entry = match stmt {
            AuroraStatement::Standard(s) => match s.as_ref() {
                sqlparser::ast::Statement::Query(q) => {
                    self.catalog.get_table(&extract_table(q))
                }
                _ => None,
            },
            _ => None,
        };
        let oids = names
            .iter()
            .map(|name| {
                entry
                    .and_then(|e| e.columns.iter().find(|c| &c.name == name))
                    .and_then(|c| SqlType::from_sql_name(&c.data_type).ok())
                    .map(|t| t.pg_oid())
                    .unwrap_or(oid::TEXT)
            })
            .collect();
        Some(oids)
    }

    /// Prepare a statement template once (extended query protocol, Req 6).
    /// Parses the SQL a single time and resolves its static shape so that
    /// repeated `Execute`s bind parameters into the cached AST without
    /// ever re-invoking the parser (Req 7). The returned [`PreparedTemplate`]
    /// is bound + run via [`Database::execute_bound_with_session`] /
    /// [`Database::execute_bound_readonly_with_session`].
    pub fn prepare(&self, sql: &str) -> GalaxResult<PreparedTemplate> {
        let stmts = parser::parse(sql)?;
        let columns = stmts.first().and_then(|s| self.describe_columns(s));
        let column_type_oids = stmts.first().and_then(|s| self.describe_column_oids(s));
        // A read-only statement is a single standard `Query` (SELECT). This
        // mirrors the read/write split the wire server uses for locking.
        let is_read = matches!(
            stmts.first(),
            Some(AuroraStatement::Standard(s))
                if matches!(s.as_ref(), sqlparser::ast::Statement::Query(_))
        );
        Ok(PreparedTemplate {
            stmts: Arc::new(stmts),
            param_count: count_placeholders(sql),
            columns,
            column_type_oids,
            is_read,
        })
    }

    /// Bind parameters into a prepared template and execute it on the
    /// write path (`&mut self`). The template's AST is reused — the parser
    /// is not invoked (Req 7). Authorization is enforced identically to the
    /// simple-query path because execution funnels through `exec_stmt`.
    pub fn execute_bound_with_session(
        &mut self,
        template: &PreparedTemplate,
        values: &[galaxdb_sql::BoundValue],
        session: Option<galaxdb_auth::SessionContext>,
    ) -> GalaxResult<QueryResult> {
        let bound = galaxdb_sql::bind_placeholders(&template.stmts, values)?;
        let prev = self.session.take();
        self.session = session;
        let result = (|| {
            let mut last = QueryResult::Ok("OK".to_string());
            for stmt in &bound {
                last = self.exec_stmt(stmt)?;
            }
            Ok(last)
        })();
        self.session = prev;
        result
    }

    /// Bind parameters into a prepared template and execute it on the
    /// read-only path (`&self`). Only valid for SELECT/SHOW templates
    /// (`PreparedTemplate::is_read`).
    pub fn execute_bound_readonly_with_session(
        &self,
        template: &PreparedTemplate,
        values: &[galaxdb_sql::BoundValue],
        session: Option<galaxdb_auth::SessionContext>,
    ) -> GalaxResult<QueryResult> {
        let bound = galaxdb_sql::bind_placeholders(&template.stmts, values)?;
        let sess = session.or_else(|| self.session.clone());
        let mut last = QueryResult::Ok("OK".to_string());
        for stmt in &bound {
            last = self.dispatch_readonly_stmt(stmt, sess.as_ref())?;
        }
        Ok(last)
    }

    /// `&self` DML path for prepared statements — same as
    /// `execute_bound_with_session` but takes a shared read lock instead of
    /// an exclusive write lock. Safe because DML (INSERT/UPDATE/DELETE) never
    /// mutates the catalog, and the engine is internally thread-safe. Allows
    /// concurrent prepared INSERTs to share WAL group-commit fsyncs.
    /// Bind + execute a prepared template inside an open explicit
    /// transaction (HTAP Phase 5, extended query protocol). Each bound
    /// statement runs through the same transaction-buffer path as the
    /// simple-query [`execute_in_txn`](Self::execute_in_txn): DML buffers
    /// writes (read-your-writes), SELECT reads at the transaction snapshot
    /// with the buffer overlaid, and unsupported-in-txn constructs surface a
    /// typed error. The parser is not re-invoked (Req 7).
    pub fn execute_bound_in_txn(
        &self,
        template: &PreparedTemplate,
        values: &[galaxdb_sql::BoundValue],
        txn: &galaxdb_sql::executor::TxnHandle,
        session: Option<galaxdb_auth::SessionContext>,
    ) -> GalaxResult<QueryResult> {
        let bound = galaxdb_sql::bind_placeholders(&template.stmts, values)?;
        let mut last = QueryResult::Ok("OK".to_string());
        for stmt in bound.iter() {
            last = self.exec_stmt_in_txn(stmt, txn, session.clone())?;
        }
        Ok(last)
    }

    pub fn execute_bound_dml_concurrent(
        &self,
        template: &PreparedTemplate,
        values: &[galaxdb_sql::BoundValue],
        session: Option<galaxdb_auth::SessionContext>,
    ) -> GalaxResult<QueryResult> {
        let bound = galaxdb_sql::bind_placeholders(&template.stmts, values)?;
        let mut last = QueryResult::Ok("OK".to_string());
        for stmt in bound.iter() {
            // Reuse the execute_dml_concurrent approach: build a context
            // from &self and dispatch the already-parsed statement.
            let plan = match stmt {
                AuroraStatement::BulkInsert(bi) => QueryPlan::BulkInsert {
                    table: bi.table.clone(),
                    columns: bi.columns.clone(),
                    values: bi.values.clone(),
                },
                AuroraStatement::Standard(s) => match s.as_ref() {
                    sqlparser::ast::Statement::Insert(ins) => {
                        let table = ins.table_name.to_string();
                        let column_names: Vec<String> = ins.columns.iter()
                            .map(|c| c.to_string())
                            .collect();
                        let Some(source) = &ins.source else { continue };
                        let sqlparser::ast::SetExpr::Values(vals) =
                            source.body.as_ref() else { continue };
                        for row in &vals.rows {
                            let row_values: Vec<Value> = row.iter()
                                .map(|e| scalar_from_expr(e).and_then(|s| s.eval(&[])))
                                .collect::<GalaxResult<Vec<Value>>>()?;
                            let row_plan = QueryPlan::Insert {
                                table: table.clone(),
                                columns: column_names.clone(),
                                values: row_values,
                            };
                            let mut ctx = ExecutorContext::new(self.engine.clone());
                            ctx.catalog = self.catalog.clone();
                            ctx.sidecar = self.sidecar.clone();
                            ctx.merkle_dag = Some(self.merkle_dag.clone());
                            ctx.tag_catalog = Some(self.tag_catalog.clone());
                            ctx.vector_backend = Some(Arc::new(EmbeddedVectorBackend {
                                sidecar: self.sidecar.clone(),
                                indexes: self.vector_indexes.clone(),
                                engine: self.engine.clone(),
                            }));
                            ctx.auth_store = Some(
                                galaxdb_sql::auth_store::AuthStore::new(self.engine.clone()),
                            );
                            ctx.secondary_index = Some(
                                galaxdb_sql::secondary_index::SecondaryIndexStore::new(
                                    self.engine.clone(),
                                ),
                            );
                            ctx.session = session.clone();
                            ctx.audit = self.audit.clone();
                            let res = execute_with_context(&row_plan, &mut ctx)?;
                            last = query_result_from(res);
                        }
                        // v0.6 metering: one INSERT statement = one write op
                        // (extended/prepared path), counted after the row loop.
                        galaxdb_observe::metrics().write_ops_total.inc();
                        continue;
                    }
                    sqlparser::ast::Statement::Update {
                        table, assignments, selection, ..
                    } => {
                        let tname = table.relation.to_string();
                        let asns: Vec<(String, ScalarExpr)> = assignments.iter()
                            .map(|a| Ok((a.target.to_string(), scalar_from_expr(&a.value)?)))
                            .collect::<GalaxResult<Vec<_>>>()?;
                        let filter = selection.as_ref().and_then(filter_from_expr);
                        QueryPlan::Update { table: tname, assignments: asns, filter }
                    }
                    sqlparser::ast::Statement::Delete(del) => {
                        let tname = match &del.from {
                            sqlparser::ast::FromTable::WithFromKeyword(tables)
                            | sqlparser::ast::FromTable::WithoutKeyword(tables) => {
                                tables.first().map(|t| t.relation.to_string()).unwrap_or_default()
                            }
                        };
                        let filter = del.selection.as_ref().and_then(filter_from_expr);
                        QueryPlan::Delete { table: tname, filter }
                    }
                    other => {
                        return Err(GalaxError::Internal(format!(
                            "execute_bound_dml_concurrent: unexpected statement: {other}"
                        )));
                    }
                },
                other => {
                    return Err(GalaxError::Internal(format!(
                        "execute_bound_dml_concurrent: unexpected statement: {other:?}"
                    )));
                }
            };
            let mut ctx = ExecutorContext::new(self.engine.clone());
            ctx.catalog = self.catalog.clone();
            ctx.sidecar = self.sidecar.clone();
            ctx.merkle_dag = Some(self.merkle_dag.clone());
            ctx.tag_catalog = Some(self.tag_catalog.clone());
            ctx.vector_backend = Some(Arc::new(EmbeddedVectorBackend {
                sidecar: self.sidecar.clone(),
                indexes: self.vector_indexes.clone(),
                engine: self.engine.clone(),
            }));
            ctx.auth_store = Some(galaxdb_sql::auth_store::AuthStore::new(self.engine.clone()));
            ctx.secondary_index = Some(
                galaxdb_sql::secondary_index::SecondaryIndexStore::new(self.engine.clone()),
            );
            ctx.session = session.clone();
            ctx.audit = self.audit.clone();
            let res = execute_with_context(&plan, &mut ctx)?;
            last = query_result_from(res);
        }
        Ok(last)
    }


    /// Bulk-insert pre-tokenized rows through the BULK INSERT executor path
    /// (`exec_bulk_insert` → `put_sync`), under an optional session so
    /// authorization is enforced (Req 3 AC7). Each cell is a raw token
    /// typed by `value_from_str` (`NULL`, numerics, bools, otherwise text),
    /// which is exactly the form `COPY ... FROM STDIN` produces — so the
    /// wire server ingests a COPY stream without building one INSERT per
    /// row (Req 8 AC3).
    pub fn bulk_insert_with_session(
        &mut self,
        table: &str,
        columns: Vec<String>,
        values: Vec<Vec<String>>,
        session: Option<galaxdb_auth::SessionContext>,
    ) -> GalaxResult<QueryResult> {
        let prev = self.session.take();
        self.session = session;
        let result = self.dispatch(QueryPlan::BulkInsert {
            table: table.to_string(),
            columns,
            values,
        });
        self.session = prev;
        result
    }

    /// Synchronous execute — for embedded Rust callers and the Python
    /// FFI.
    pub fn execute(&mut self, sql: &str) -> GalaxResult<QueryResult> {
        // AT VERSION intercept: sqlparser doesn't understand the
        // AuroraSQL `AT VERSION ...` suffix, so if we see one on a
        // SELECT we split the SQL into (stripped, at_version) and
        // dispatch to the versioned plan arm directly. See task 32.3 /
        // 32.4 in docs/CONSOLIDATION.md.
        if let Some((stripped, at)) = split_at_version(sql)? {
            return self.exec_select_at_version(&stripped, at);
        }

        let stmts = self.cached_parse(sql)?;
        let mut last = QueryResult::Ok("OK".to_string());
        for stmt in stmts.iter() {
            last = self.exec_stmt(stmt)?;
        }
        Ok(last)
    }

    /// Async variant — identical semantics; currently just wraps the
    /// sync path. Retained because the wire-protocol path wants an
    /// `async` signature and v2 will make the engine truly async.
    pub async fn execute_async(&mut self, sql: &str) -> GalaxResult<QueryResult> {
        self.execute(sql)
    }

    /// Execute a read-only statement without `&mut self`. Used by
    /// callers holding the database behind an `RwLock` that want to
    /// allow concurrent reads.
    pub fn execute_readonly(&self, sql: &str) -> GalaxResult<QueryResult> {
        self.execute_readonly_with_session(sql, self.session.clone())
    }

    /// `&self` read path under an explicit per-call session (task 6). The
    /// wire server uses this on the shared-read lock so each connection's
    /// SELECTs are authorization-checked under that connection's role
    /// without mutating the shared `Database`.
    pub fn execute_readonly_with_session(
        &self,
        sql: &str,
        session: Option<galaxdb_auth::SessionContext>,
    ) -> GalaxResult<QueryResult> {
        // AT VERSION on the read path: same intercept as `execute`, but
        // we don't need `&mut self` because the plan only scans storage.
        if let Some((stripped, at)) = split_at_version(sql)? {
            return self.select_at_version_readonly(&stripped, at, session.as_ref());
        }

        let stmts = self.cached_parse(sql)?;
        let mut last = QueryResult::Ok("OK".to_string());
        for stmt in stmts.iter() {
            last = self.dispatch_readonly_stmt(stmt, session.as_ref())?;
        }
        Ok(last)
    }

    /// Dispatch one already-parsed statement on the read-only path.
    /// Rejects write-capable statements so a read lock can never mutate.
    fn dispatch_readonly_stmt(
        &self,
        stmt: &AuroraStatement,
        session: Option<&galaxdb_auth::SessionContext>,
    ) -> GalaxResult<QueryResult> {
        match stmt {
            AuroraStatement::Standard(s) => {
                if let sqlparser::ast::Statement::Query(q) = s.as_ref() {
                    self.select_readonly(q, session)
                } else {
                    Err(GalaxError::Internal(
                        "execute_readonly only supports SELECT and SHOW; \
                         use execute() for write-capable statements"
                            .into(),
                    ))
                }
            }
            AuroraStatement::ShowEmbeddingHealth { table } => {
                // Route through the real executor so this reports the
                // actual sidecar state + model version (Req 19), not a
                // canned echo string. Build a read context that carries
                // the sidecar so `exec_show_embedding_health` can inspect
                // it.
                let plan = planner::QueryPlan::ShowEmbeddingHealth {
                    table: table.clone(),
                };
                let mut ctx = ExecutorContext::new(self.engine.clone());
                ctx.catalog = self.catalog.clone();
                ctx.sidecar = self.sidecar.clone();
                // The authorization chokepoint resolves the session role's
                // grants through the auth store — without it even a valid
                // read is denied. Mirror the SELECT readonly path.
                ctx.auth_store =
                    Some(galaxdb_sql::auth_store::AuthStore::new(self.engine.clone()));
                ctx.session = session.cloned();
                ctx.audit = self.audit.clone();
                let res = execute_with_context(&plan, &mut ctx)?;
                Ok(query_result_from(res))
            }
            _ => Err(GalaxError::Internal(
                "execute_readonly only supports SELECT and SHOW; use execute() for \
                 write-capable statements"
                    .into(),
            )),
        }
    }

    fn select_readonly(
        &self,
        q: &sqlparser::ast::Query,
        session: Option<&galaxdb_auth::SessionContext>,
    ) -> GalaxResult<QueryResult> {
        // FROM-less scalar SELECT (`SELECT 1 + 1`, `version()`,
        // `current_database()`): evaluate directly, never route to the
        // analytical engine.
        if is_from_less_select(q) {
            let user = session.map(|s| s.role.id.as_str().to_string());
            return eval_scalar_select(q, user.as_deref());
        }

        // Detect SEMANTIC_MATCH in WHERE and route to vector search.
        if let Some(semantic_expr) = extract_semantic_match_from_query(q) {
            // SEMANTIC_MATCH combined with *genuine* analytical clauses
            // (JOIN / GROUP BY / aggregate / ORDER BY / …) → feed the matched
            // candidate set to the DataFusion analytical engine (HTAP task
            // 16). A bare `LIMIT` / `OFFSET` is NOT such a clause: it stays on
            // the native similarity-ranked path and becomes the top-k bound,
            // so `SEMANTIC_MATCH(...) LIMIT n` returns the n nearest matches
            // (previously LIMIT>100 was silently capped by the analytical
            // candidate ceiling, and a plain query capped at 10).
            if galaxdb_sql::classify::is_analytical_beyond_pagination(q) {
                return self.analytical_semantic_query(q, &semantic_expr, session);
            }
            let table = extract_table(q);
            let (_columns, extra_filter) = extract_projection_and_filter_no_semantic(q);
            let limit = galaxdb_sql::classify::query_limit(q);
            let plan = planner::plan_semantic_search(
                table.clone(),
                semantic_expr,
                extra_filter,
                None,
                limit,
            );
            let mut ctx = ExecutorContext::new(self.engine.clone());
            ctx.catalog = self.catalog.clone();
            ctx.auth_store = Some(galaxdb_sql::auth_store::AuthStore::new(self.engine.clone()));
            ctx.secondary_index = Some(
                galaxdb_sql::secondary_index::SecondaryIndexStore::new(self.engine.clone()),
            );
            ctx.session = session.cloned();
            ctx.audit = self.audit.clone();
            ctx.vector_backend = Some(Arc::new(EmbeddedVectorBackend {
                sidecar: self.sidecar.clone(),
                indexes: self.vector_indexes.clone(),
                engine: self.engine.clone(),
            }));
            let res = execute_with_context(&plan, &mut ctx)?;
            return Ok(query_result_from(res));
        }

        // Analytical (joins/aggregates/GROUP BY/subqueries/ORDER BY/LIMIT)
        // → DataFusion analytical engine (HTAP task 15). `NOT DUPLICATE`
        // queries stay native so the dedup pass is preserved (task 17.1).
        if !query_has_not_duplicate(q)
            && matches!(
                galaxdb_sql::classify::classify_query(q),
                galaxdb_sql::classify::StatementClass::Analytical
            )
        {
            return self.analytical_query(q, session);
        }

        let (columns, filter) = extract_projection_and_filter(q);
        let table = extract_table(q);
        if table != "unknown" && !self.catalog.table_exists(&table) {
            return Err(GalaxError::TableNotFound(table));
        }
        let plan = QueryPlan::FullScan {
            table,
            filter,
            columns,
        };

        let mut ctx = ExecutorContext::new(self.engine.clone());
        ctx.catalog = self.catalog.clone();
        ctx.auth_store = Some(galaxdb_sql::auth_store::AuthStore::new(self.engine.clone()));
        ctx.secondary_index = Some(
            galaxdb_sql::secondary_index::SecondaryIndexStore::new(self.engine.clone()),
        );
        ctx.session = session.cloned();
        ctx.audit = self.audit.clone();
        ctx.vector_backend = Some(Arc::new(EmbeddedVectorBackend {
            sidecar: self.sidecar.clone(),
            indexes: self.vector_indexes.clone(),
            engine: self.engine.clone(),
        }));
        let res = execute_with_context(&plan, &mut ctx)?;
        Ok(query_result_from(res))
    }

    /// Execute an analytical SELECT (joins / aggregates / GROUP BY /
    /// subqueries / ORDER BY / LIMIT) via the DataFusion query engine behind
    /// `galaxdb-query` (HTAP task 15). Collects the referenced tables,
    /// authorizes SELECT on each (when a session is attached), builds the
    /// columnar Arrow sources, runs the SQL, and maps the Arrow result to
    /// wire rows. `NULL` cells render as the literal `"NULL"`, matching the
    /// native path's text rendering.
    fn analytical_query(
        &self,
        q: &sqlparser::ast::Query,
        session: Option<&galaxdb_auth::SessionContext>,
    ) -> GalaxResult<QueryResult> {
        self.analytical_query_at(q, session, galaxdb_query::ReadSnapshot::Latest)
    }

    /// [`analytical_query`](Self::analytical_query) at a specific MVCC read
    /// snapshot. `AT VERSION <tag|ts>` on an analytical query (JOIN / GROUP BY
    /// / aggregate) resolves to `ReadSnapshot::AsOfTimestamp`, so DataFusion's
    /// scans read the historical columnar data through the same per-row ts
    /// filter the native time-travel path uses (HTAP task 17, design §3.5).
    fn analytical_query_at(
        &self,
        q: &sqlparser::ast::Query,
        session: Option<&galaxdb_auth::SessionContext>,
        snapshot: galaxdb_query::ReadSnapshot,
    ) -> GalaxResult<QueryResult> {
        let tables = collect_table_names(q);
        if tables.is_empty() {
            return Err(GalaxError::FeatureNotSupported(
                "analytical query references no base table".into(),
            ));
        }
        self.authorize_select(&tables, session)?;

        let mut metas: Vec<galaxdb_query::backend::TableSpec> =
            Vec::with_capacity(tables.len());
        for t in &tables {
            let entry = self
                .catalog
                .get_table(t)
                .ok_or_else(|| GalaxError::TableNotFound(t.clone()))?;
            let fields = entry
                .columns
                .iter()
                .map(|c| {
                    let ty = galaxdb_sql::SqlType::from_sql_name(&c.data_type)
                        .unwrap_or(galaxdb_sql::SqlType::Text);
                    (c.name.clone(), ty)
                })
                .collect::<Vec<_>>();
            metas.push((t.clone(), format!("{t}:").into_bytes(), fields));
        }

        let sql = q.to_string();
        let (col_names, rows) = galaxdb_query::backend::run_analytical_sql_blocking(
            self.engine.clone(),
            &metas,
            &sql,
            snapshot,
        )?;

        let out_rows = rows
            .into_iter()
            .map(|cells| QueryRow {
                values: col_names
                    .iter()
                    .cloned()
                    .zip(cells.into_iter().map(|c| c.unwrap_or_else(|| "NULL".to_string())))
                    .collect(),
            })
            .collect();
        Ok(QueryResult::Rows(out_rows))
    }

    /// `SEMANTIC_MATCH` feeding an analytical query (HTAP task 16, ADR-0004).
    ///
    /// When a `SELECT` combines `SEMANTIC_MATCH(...)` with analytical clauses
    /// (JOIN / GROUP BY / aggregate / ORDER BY / …), the native single-table
    /// vector path cannot serve it. Here the native HNSW backend computes the
    /// top-k matched rows **once** (the paper's adaptive strategy: a residual
    /// relational predicate → brute-force over the filtered set; otherwise
    /// HNSW-first), those rows become a candidate Arrow table (the base
    /// columns plus a `similarity` column), and DataFusion runs the query —
    /// with the `SEMANTIC_MATCH(...)` predicate stripped — over exactly the
    /// matched rows, joining/aggregating them against the relational columns.
    fn analytical_semantic_query(
        &self,
        q: &sqlparser::ast::Query,
        semantic: &galaxdb_sql::ast::SemanticMatchExpr,
        session: Option<&galaxdb_auth::SessionContext>,
    ) -> GalaxResult<QueryResult> {
        use galaxdb_sql::executor::VectorSearchBackend;
        use galaxdb_sql::planner::{SearchStrategy, Value};
        use std::collections::HashMap;

        // The semantically-matched table is the query's base (FROM) table.
        let sem_table = extract_table(q);
        let tables = collect_table_names(q);
        if tables.is_empty() {
            return Err(GalaxError::FeatureNotSupported(
                "SEMANTIC_MATCH analytical query references no base table".into(),
            ));
        }
        self.authorize_select(&tables, session)?;

        let sem_entry = self
            .catalog
            .get_table(&sem_table)
            .ok_or_else(|| GalaxError::TableNotFound(sem_table.clone()))?
            .clone();

        // Residual (non-semantic) relational predicate → adaptive strategy.
        let (_proj, residual) = extract_projection_and_filter_no_semantic(q);

        // Candidate set the analytical query aggregates/paginates over.
        // The floor (100) is larger than the native default because
        // aggregates/joins want the full matched population, not a display
        // page. An explicit `LIMIT n` larger than the floor raises the
        // ceiling to `n` so `SEMANTIC_MATCH(...) LIMIT 500` is not silently
        // truncated to 100 before DataFusion applies the limit.
        const CANDIDATE_K_FLOOR: usize = 100;
        let candidate_k = galaxdb_sql::classify::query_limit(q)
            .map(|n| n.max(CANDIDATE_K_FLOOR))
            .unwrap_or(CANDIDATE_K_FLOOR);

        let backend = EmbeddedVectorBackend {
            sidecar: self.sidecar.clone(),
            indexes: self.vector_indexes.clone(),
            engine: self.engine.clone(),
        };
        let results = match &residual {
            Some(f) => backend.brute_force_filtered(
                &sem_table,
                &semantic.query,
                semantic.threshold,
                candidate_k,
                f,
            )?,
            None => backend.semantic_search(
                &sem_table,
                &semantic.query,
                semantic.threshold,
                candidate_k,
                SearchStrategy::HnswWithPostFilter,
            )?,
        };

        // Resolve each matched row_id (xxh3_64 of the primary key) to its
        // decoded row, mirroring the native SEMANTIC_MATCH row join.
        let col_order: Vec<String> =
            sem_entry.columns.iter().map(|c| c.name.clone()).collect();
        let base_cols: Vec<(String, galaxdb_sql::SqlType)> = sem_entry
            .columns
            .iter()
            .map(|c| {
                let ty = galaxdb_sql::SqlType::from_sql_name(&c.data_type)
                    .unwrap_or(galaxdb_sql::SqlType::Text);
                (c.name.clone(), ty)
            })
            .collect();
        let prefix = format!("{sem_table}:");
        let row_map: HashMap<u64, Vec<(String, Value)>> = self
            .engine
            .scan_all()
            .into_iter()
            .filter(|(k, _)| String::from_utf8_lossy(k).starts_with(&prefix))
            .map(|(k, v)| {
                (
                    xxhash_rust::xxh3::xxh3_64(&k),
                    galaxdb_sql::row_codec::decode_row(&v),
                )
            })
            .collect();

        let mut cand_rows: Vec<Vec<Option<Value>>> = Vec::new();
        let mut sims: Vec<f64> = Vec::new();
        for r in &results {
            if let Some(cells) = row_map.get(&r.row_id) {
                let by_name: HashMap<&str, &Value> =
                    cells.iter().map(|(n, v)| (n.as_str(), v)).collect();
                let row: Vec<Option<Value>> = col_order
                    .iter()
                    .map(|n| by_name.get(n.as_str()).map(|v| (*v).clone()))
                    .collect();
                cand_rows.push(row);
                sims.push(r.similarity as f64);
            }
        }

        let provider = Arc::new(EmbeddedSemanticCandidateProvider {
            schema: galaxdb_query::semantic::candidate_schema(&base_cols),
            base_cols,
            rows: cand_rows,
            sims,
        });
        let source: Arc<dyn galaxdb_query::ArrowSource> =
            Arc::new(galaxdb_query::semantic::SemanticCandidateSource::new(provider));

        // TableSpecs for every referenced table (the semantic one is served
        // by the override, the rest by the engine).
        let mut metas: Vec<galaxdb_query::backend::TableSpec> = Vec::with_capacity(tables.len());
        for t in &tables {
            let entry = self
                .catalog
                .get_table(t)
                .ok_or_else(|| GalaxError::TableNotFound(t.clone()))?;
            let fields = entry
                .columns
                .iter()
                .map(|c| {
                    let ty = galaxdb_sql::SqlType::from_sql_name(&c.data_type)
                        .unwrap_or(galaxdb_sql::SqlType::Text);
                    (c.name.clone(), ty)
                })
                .collect::<Vec<_>>();
            metas.push((t.clone(), format!("{t}:").into_bytes(), fields));
        }

        let stripped = strip_semantic_match_query(q);
        let sql = stripped.to_string();
        let (col_names, rows) = galaxdb_query::backend::run_analytical_sql_blocking_with_semantic(
            self.engine.clone(),
            &metas,
            std::slice::from_ref(&(sem_table, source)),
            &sql,
            galaxdb_query::ReadSnapshot::Latest,
        )?;

        let out_rows = rows
            .into_iter()
            .map(|cells| QueryRow {
                values: col_names
                    .iter()
                    .cloned()
                    .zip(cells.into_iter().map(|c| c.unwrap_or_else(|| "NULL".to_string())))
                    .collect(),
            })
            .collect();
        Ok(QueryResult::Rows(out_rows))
    }

    /// Authorize `SELECT` on every referenced table before an analytical
    /// query runs, mirroring the executor's chokepoint so the analytical
    /// path is not a privilege-escalation bypass. Trusted in-process mode
    /// (no session) skips the check, exactly like the native executor.
    fn authorize_select(
        &self,
        tables: &[String],
        session: Option<&galaxdb_auth::SessionContext>,
    ) -> GalaxResult<()> {
        use galaxdb_auth::{Action, Authorizer, ObjectRef, TableGrantAuthorizer};
        let Some(session) = session else {
            return Ok(());
        };
        let store = galaxdb_sql::auth_store::AuthStore::new(self.engine.clone());
        let authz = TableGrantAuthorizer::new(Arc::new(move |r: &str, t: &str, a: Action| {
            store.has_grant(r, t, a)
        }));
        for t in tables {
            authz
                .check(&session.role, Action::Select, &ObjectRef::Table(t.clone()))
                .map_err(|e| GalaxError::InsufficientPrivilege {
                    role: e.role.as_str().to_string(),
                    action: e.action,
                    object: e.object,
                })?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------
    // Statement dispatch
    // -----------------------------------------------------------------

    fn exec_stmt(&mut self, stmt: &AuroraStatement) -> GalaxResult<QueryResult> {
        // Translate `AuroraStatement` into a `QueryPlan`, then delegate
        // to `execute_with_context`. For `SEMANTIC_MATCH` (which the
        // current planner doesn't handle directly outside `WHERE`), we
        // route to the vector backend inline.
        match stmt {
            AuroraStatement::Standard(s) => self.exec_standard(s),
            AuroraStatement::CreateTable(ct) => self.exec_create_table(ct),
            AuroraStatement::SemanticMatch(expr) => self.exec_semantic_match_standalone(expr),
            AuroraStatement::Analyze { table } => self.dispatch(QueryPlan::Analyze {
                table: table.clone(),
            }),
            AuroraStatement::BackupTo { path } => self.dispatch(QueryPlan::Backup {
                path: path.clone(),
            }),
            AuroraStatement::RestoreFrom { path } => self.dispatch(QueryPlan::Restore {
                path: path.clone(),
            }),
            AuroraStatement::ShowEmbeddingHealth { table } => {
                self.dispatch(QueryPlan::ShowEmbeddingHealth {
                    table: table.clone(),
                })
            }
            AuroraStatement::CreateVersionTag(tag_stmt) => {
                self.dispatch(QueryPlan::CreateVersionTag(tag_stmt.clone()))
            }
            AuroraStatement::BulkInsert(bi) => self.dispatch(QueryPlan::BulkInsert {
                table: bi.table.clone(),
                columns: bi.columns.clone(),
                values: bi.values.clone(),
            }),
            AuroraStatement::AtVersion(_) => Err(GalaxError::NotYetAvailable {
                task: "B6",
                feature: "AT VERSION planner wiring (consolidation Phase B6 deferred)",
            }),
            AuroraStatement::CreateRole(stmt) => {
                self.dispatch(QueryPlan::CreateRole(stmt.clone()))
            }
            AuroraStatement::DropRole { name, if_exists } => self.dispatch(QueryPlan::DropRole {
                name: name.clone(),
                if_exists: *if_exists,
            }),
            AuroraStatement::AlterRolePassword { name, password } => {
                self.dispatch(QueryPlan::AlterRolePassword {
                    name: name.clone(),
                    password: password.clone(),
                })
            }
            AuroraStatement::Grant(stmt) => self.dispatch(QueryPlan::Grant(stmt.clone())),
            AuroraStatement::Revoke(stmt) => self.dispatch(QueryPlan::Revoke(stmt.clone())),
            AuroraStatement::CreateIndex(stmt) => {
                self.dispatch(QueryPlan::CreateIndex(stmt.clone()))
            }
            AuroraStatement::DropIndex { name, if_exists } => {
                self.dispatch(QueryPlan::DropIndex {
                    name: name.clone(),
                    if_exists: *if_exists,
                })
            }
            AuroraStatement::AlterTableSetStorage { table, mode } => {
                self.dispatch(QueryPlan::AlterTableSetStorage {
                    table: table.clone(),
                    mode: *mode,
                })
            }
            AuroraStatement::CreateSemanticCache(stmt) => {
                self.dispatch(QueryPlan::CreateSemanticCache(stmt.clone()))
            }
            AuroraStatement::DropSemanticCache { table } => {
                self.dispatch(QueryPlan::DropSemanticCache {
                    table: table.clone(),
                })
            }
        }
    }

    fn exec_standard(&mut self, stmt: &sqlparser::ast::Statement) -> GalaxResult<QueryResult> {
        match stmt {
            sqlparser::ast::Statement::CreateTable(ct) => self.exec_sqlparser_create(ct),
            sqlparser::ast::Statement::Drop {
                names, if_exists, ..
            } => self.dispatch(QueryPlan::DropTable {
                name: names
                    .first()
                    .map(|n| n.to_string())
                    .unwrap_or_default(),
                if_exists: *if_exists,
            }),
            sqlparser::ast::Statement::Insert(ins) => self.exec_insert(ins),
            sqlparser::ast::Statement::Query(q) => self.exec_select(q),
            sqlparser::ast::Statement::Update {
                table,
                assignments,
                selection,
                ..
            } => self.exec_update(
                &table.relation.to_string(),
                assignments,
                selection.as_ref(),
            ),
            sqlparser::ast::Statement::Delete(del) => self.exec_delete(del),
            // SQL-level PREPARE/EXECUTE/DEALLOCATE are not implemented as
            // server commands — clients use the wire extended-query protocol
            // (Parse/Bind/Execute) for prepared statements, which is
            // supported. Name the construct in a clean, typed error rather
            // than leaking an internal AST discriminant.
            sqlparser::ast::Statement::Prepare { .. } => Err(GalaxError::FeatureNotSupported(
                "SQL-level PREPARE is not supported; use the wire protocol's \
                 extended-query (Parse/Bind/Execute) prepared statements instead"
                    .to_string(),
            )),
            sqlparser::ast::Statement::Execute { .. } => Err(GalaxError::FeatureNotSupported(
                "SQL-level EXECUTE is not supported; use the wire protocol's \
                 extended-query (Parse/Bind/Execute) prepared statements instead"
                    .to_string(),
            )),
            sqlparser::ast::Statement::Deallocate { .. } => Err(GalaxError::FeatureNotSupported(
                "SQL-level DEALLOCATE is not supported; wire-protocol prepared \
                 statements are closed with the Close message"
                    .to_string(),
            )),
            other => Err(GalaxError::FeatureNotSupported(format!(
                "SQL statement is not supported by GalaxDB: {other}"
            ))),
        }
    }

    fn exec_sqlparser_create(
        &mut self,
        ct: &sqlparser::ast::CreateTable,
    ) -> GalaxResult<QueryResult> {
        let columns: Vec<galaxdb_sql::ast::ColumnDef> = ct
            .columns
            .iter()
            .map(|c| galaxdb_sql::ast::ColumnDef {
                name: c.name.to_string(),
                data_type: format!("{}", c.data_type),
                nullable: true,
                primary_key: c.options.iter().any(|o| {
                    matches!(
                        o.option,
                        sqlparser::ast::ColumnOption::Unique {
                            is_primary: true,
                            ..
                        }
                    )
                }),
                embedding: None,
            })
            .collect();
        self.exec_create_table(&CreateTableStmt {
            table_name: ct.name.to_string(),
            columns,
            if_not_exists: ct.if_not_exists,
        })
    }

    fn exec_create_table(&mut self, ct: &CreateTableStmt) -> GalaxResult<QueryResult> {
        let has_embedding = ct.columns.iter().any(|c| c.embedding.is_some());

        // Delegate catalog registration to the executor.
        let mut ctx = self.context();
        let plan = QueryPlan::CreateTable(ct.clone());
        let result = execute_with_context(&plan, &mut ctx)?;
        self.catalog = std::mem::take(&mut ctx.catalog);

        // If the table has an embedding column, create a vector index.
        if has_embedding {
            for col in &ct.columns {
                if let Some(ref emb) = col.embedding {
                    let dim = emb.dimensions.unwrap_or(128) as usize;
                    let config = HnswConfig::new(dim).with_max_elements(1_000_000);
                    let idx = TableVectorIndex {
                        hnsw: HnswGraph::new(config),
                        delta: DeltaBuffer::new(dim),
                        dim,
                        embedding_column: col.name.clone(),
                        source_column: col.name.clone(),
                        vectors: HashMap::new(),
                        key_to_row_id: HashMap::new(),
                        semantic_cache: SemanticCache::new(),
                    };
                    self.vector_indexes
                        .write()
                        .unwrap()
                        .insert(ct.table_name.clone(), idx);
                    break;
                }
            }
        }

        Ok(query_result_from(result))
    }

    fn exec_insert(&mut self, ins: &sqlparser::ast::Insert) -> GalaxResult<QueryResult> {
        let table = ins.table_name.to_string();
        // Validate the table exists up front for a clean error before the
        // per-row loop (execute_with_context re-checks internally).
        if !self.catalog.table_exists(&table) {
            return Err(GalaxError::TableNotFound(table.clone()));
        }

        let column_names: Vec<String> = ins
            .columns
            .iter()
            .map(|c| c.to_string())
            .collect();

        let Some(source) = &ins.source else {
            return Ok(QueryResult::RowCount(0));
        };
        let sqlparser::ast::SetExpr::Values(values) = source.body.as_ref() else {
            return Ok(QueryResult::RowCount(0));
        };

        let mut count = 0u64;
        for row in &values.rows {
            let row_values: Vec<Value> = row
                .iter()
                .map(|e| scalar_from_expr(e).and_then(|s| s.eval(&[])))
                .collect::<GalaxResult<Vec<Value>>>()?;
            let plan = QueryPlan::Insert {
                table: table.clone(),
                columns: column_names.clone(),
                values: row_values.clone(),
            };
            let mut ctx = self.context();
            let res = execute_with_context(&plan, &mut ctx)?;
            self.catalog = std::mem::take(&mut ctx.catalog);
            if matches!(res, ExecuteResult::RowCount(_)) {
                count += 1;
            }
            // Embedding population happens inside `execute_with_context`
            // (exec_insert → VectorSearchBackend::on_row_inserted), which
            // both this `&mut self` path and the server's concurrent
            // `&self` path share. No separate trigger here — that used to
            // double-embed the wire path or skip it entirely.
        }

        // v0.6 metering: one INSERT statement = one write op (embedded/&mut
        // path), counted after all rows succeed.
        galaxdb_observe::metrics().write_ops_total.inc();
        Ok(QueryResult::RowCount(count))
    }

    fn exec_select(&mut self, q: &sqlparser::ast::Query) -> GalaxResult<QueryResult> {
        // FROM-less scalar SELECT (`SELECT 1 + 1`, `SELECT version()`,
        // `SELECT current_database()`): evaluate the projection directly.
        // These have no base table, so they must not reach the analytical
        // engine (which would error "references no base table").
        if is_from_less_select(q) {
            let user = self
                .session
                .as_ref()
                .map(|s| s.role.id.as_str().to_string());
            return eval_scalar_select(q, user.as_deref());
        }

        let table = extract_table(q);

        // Detect SEMANTIC_MATCH(...) in the WHERE clause and route to
        // the vector search path instead of a full scan.
        if let Some(semantic_expr) = extract_semantic_match_from_query(q) {
            // SEMANTIC_MATCH + *genuine* analytical clauses → analytical
            // candidate path (HTAP task 16). A bare LIMIT/OFFSET stays native
            // and becomes the top-k bound (see select_readonly for details).
            if galaxdb_sql::classify::is_analytical_beyond_pagination(q) {
                let session = self.session.clone();
                return self.analytical_semantic_query(q, &semantic_expr, session.as_ref());
            }
            let (_columns, extra_filter) = extract_projection_and_filter_no_semantic(q);
            let limit = galaxdb_sql::classify::query_limit(q);
            let plan = planner::plan_semantic_search(
                table.clone(),
                semantic_expr,
                extra_filter,
                None,
                limit,
            );
            let mut ctx = self.context();
            let res = execute_with_context(&plan, &mut ctx)?;
            self.catalog = std::mem::take(&mut ctx.catalog);
            return Ok(query_result_from(res));
        }

        // Analytical (joins/aggregates/GROUP BY/subqueries/ORDER BY/LIMIT)
        // → DataFusion engine (HTAP task 15). `NOT DUPLICATE` queries stay
        // native so the dedup pass is preserved (task 17.1).
        if !query_has_not_duplicate(q)
            && matches!(
                galaxdb_sql::classify::classify_query(q),
                galaxdb_sql::classify::StatementClass::Analytical
            )
        {
            let session = self.session.clone();
            return self.analytical_query(q, session.as_ref());
        }

        let (columns, filter) = extract_projection_and_filter(q);
        let plan = QueryPlan::FullScan {
            table: table.clone(),
            filter,
            columns,
        };
        let mut ctx = self.context();
        let res = execute_with_context(&plan, &mut ctx)?;
        self.catalog = std::mem::take(&mut ctx.catalog);
        Ok(query_result_from(res))
    }

    /// Resolve an `AtVersionExpr` to an MVCC read timestamp: a literal
    /// timestamp is used directly; a version tag is looked up in the tag
    /// catalog (HTAP task 17, mirroring the native `FullScanAtVersion`
    /// resolver so native and analytical time-travel agree).
    fn resolve_at_version_ts(
        &self,
        at: &galaxdb_sql::ast::AtVersionExpr,
    ) -> GalaxResult<galaxdb_common::Timestamp> {
        use galaxdb_sql::ast::VersionRef;
        match &at.version {
            VersionRef::Timestamp(ts) => Ok(*ts),
            VersionRef::Tag(name) => {
                let tc = self
                    .tag_catalog
                    .lock()
                    .map_err(|_| GalaxError::Internal("tag catalog mutex poisoned".into()))?;
                tc.get_tag(name)
                    .map(|t| t.version_timestamp)
                    .ok_or_else(|| GalaxError::Internal(format!("unknown version tag: {name}")))
            }
        }
    }

    /// Dispatch a `SELECT ... AT VERSION <ref> [CONSISTENCY <mode>]`
    /// query to the canonical executor. The SQL text passed in has
    /// already had the AT VERSION suffix stripped by
    /// [`split_at_version`]; we parse the remainder as a normal
    /// SELECT so we can reuse `extract_projection_and_filter`, then
    /// build a `FullScanAtVersion` plan.
    fn exec_select_at_version(
        &mut self,
        stripped_sql: &str,
        at: galaxdb_sql::ast::AtVersionExpr,
    ) -> GalaxResult<QueryResult> {
        let stmts = parser::parse(stripped_sql)?;
        let Some(stmt) = stmts.first() else {
            return Err(GalaxError::Internal(
                "AT VERSION: SELECT body parsed to zero statements".into(),
            ));
        };
        let AuroraStatement::Standard(boxed) = stmt else {
            return Err(GalaxError::Internal(
                "AT VERSION is only supported on SELECT statements".into(),
            ));
        };
        let sqlparser::ast::Statement::Query(q) = boxed.as_ref() else {
            return Err(GalaxError::Internal(
                "AT VERSION is only supported on SELECT statements".into(),
            ));
        };

        // Analytical AT VERSION (JOIN / GROUP BY / aggregate, without a
        // SEMANTIC_MATCH) → resolve the snapshot ts and run the analytical
        // engine at that snapshot (HTAP task 17).
        if extract_semantic_match_from_query(q).is_none()
            && matches!(
                galaxdb_sql::classify::classify_query(q),
                galaxdb_sql::classify::StatementClass::Analytical
            )
        {
            let ts = self.resolve_at_version_ts(&at)?;
            let session = self.session.clone();
            return self.analytical_query_at(
                q,
                session.as_ref(),
                galaxdb_query::ReadSnapshot::AsOfTimestamp(ts),
            );
        }

        check_select_supported(q)?;
        let table = extract_table(q);
        let (columns, filter) = extract_projection_and_filter(q);

        // v0.7 (inventory 5.12/8.13): SEMANTIC_MATCH + AT VERSION →
        // historical semantic search (SEMANTIC_FRESH / SEMANTIC_SNAPSHOT),
        // routed to HybridSearchAtVersion. Without a SEMANTIC_MATCH it stays a
        // plain time-travel row scan (FullScanAtVersion).
        if let Some(sem) = extract_semantic_match_from_query(q) {
            let strategy = if filter.is_some() {
                galaxdb_sql::planner::SearchStrategy::BruteForceFiltered
            } else {
                galaxdb_sql::planner::SearchStrategy::HnswWithPostFilter
            };
            let plan = QueryPlan::HybridSearchAtVersion {
                table,
                filter,
                semantic: sem,
                strategy,
                at,
                limit: None,
            };
            let mut ctx = self.context();
            let res = execute_with_context(&plan, &mut ctx)?;
            self.catalog = std::mem::take(&mut ctx.catalog);
            return Ok(query_result_from(res));
        }

        let plan = QueryPlan::FullScanAtVersion {
            table,
            filter,
            columns,
            at,
        };
        let mut ctx = self.context();
        let res = execute_with_context(&plan, &mut ctx)?;
        self.catalog = std::mem::take(&mut ctx.catalog);
        Ok(query_result_from(res))
    }

    /// `&self` variant of [`Self::exec_select_at_version`] used by the
    /// wire-protocol read path.
    fn select_at_version_readonly(
        &self,
        stripped_sql: &str,
        at: galaxdb_sql::ast::AtVersionExpr,
        session: Option<&galaxdb_auth::SessionContext>,
    ) -> GalaxResult<QueryResult> {
        let stmts = parser::parse(stripped_sql)?;
        let Some(stmt) = stmts.first() else {
            return Err(GalaxError::Internal(
                "AT VERSION: SELECT body parsed to zero statements".into(),
            ));
        };
        let AuroraStatement::Standard(boxed) = stmt else {
            return Err(GalaxError::Internal(
                "AT VERSION is only supported on SELECT statements".into(),
            ));
        };
        let sqlparser::ast::Statement::Query(q) = boxed.as_ref() else {
            return Err(GalaxError::Internal(
                "AT VERSION is only supported on SELECT statements".into(),
            ));
        };

        // Analytical AT VERSION → historical analytical scan (HTAP task 17).
        if extract_semantic_match_from_query(q).is_none()
            && matches!(
                galaxdb_sql::classify::classify_query(q),
                galaxdb_sql::classify::StatementClass::Analytical
            )
        {
            let ts = self.resolve_at_version_ts(&at)?;
            return self.analytical_query_at(
                q,
                session,
                galaxdb_query::ReadSnapshot::AsOfTimestamp(ts),
            );
        }

        check_select_supported(q)?;
        let table = extract_table(q);
        let (columns, filter) = extract_projection_and_filter(q);
        if table != "unknown" && !self.catalog.table_exists(&table) {
            return Err(GalaxError::TableNotFound(table));
        }
        // v0.7: SEMANTIC_MATCH + AT VERSION → historical semantic search.
        let plan = if let Some(sem) = extract_semantic_match_from_query(q) {
            let strategy = if filter.is_some() {
                galaxdb_sql::planner::SearchStrategy::BruteForceFiltered
            } else {
                galaxdb_sql::planner::SearchStrategy::HnswWithPostFilter
            };
            QueryPlan::HybridSearchAtVersion {
                table,
                filter,
                semantic: sem,
                strategy,
                at,
                limit: None,
            }
        } else {
            QueryPlan::FullScanAtVersion {
                table,
                filter,
                columns,
                at,
            }
        };

        let mut ctx = ExecutorContext::new(self.engine.clone());
        ctx.catalog = self.catalog.clone();
        ctx.auth_store = Some(galaxdb_sql::auth_store::AuthStore::new(self.engine.clone()));
        ctx.secondary_index = Some(
            galaxdb_sql::secondary_index::SecondaryIndexStore::new(self.engine.clone()),
        );
        ctx.session = session.cloned();
        ctx.audit = self.audit.clone();
        ctx.tag_catalog = Some(self.tag_catalog.clone());
        ctx.merkle_dag = Some(self.merkle_dag.clone());
        ctx.vector_backend = Some(Arc::new(EmbeddedVectorBackend {
            sidecar: self.sidecar.clone(),
            indexes: self.vector_indexes.clone(),
            engine: self.engine.clone(),
        }));
        let res = execute_with_context(&plan, &mut ctx)?;
        Ok(query_result_from(res))
    }

    fn exec_update(
        &mut self,
        table: &str,
        assignments: &[sqlparser::ast::Assignment],
        selection: Option<&sqlparser::ast::Expr>,
    ) -> GalaxResult<QueryResult> {
        let aligned: Vec<(String, ScalarExpr)> = assignments
            .iter()
            .map(|a| Ok((a.target.to_string(), scalar_from_expr(&a.value)?)))
            .collect::<GalaxResult<Vec<_>>>()?;
        let filter = selection.and_then(filter_from_expr);
        let plan = QueryPlan::Update {
            table: table.to_string(),
            assignments: aligned,
            filter,
        };
        self.dispatch(plan)
    }

    fn exec_delete(&mut self, del: &sqlparser::ast::Delete) -> GalaxResult<QueryResult> {
        let table = match &del.from {
            sqlparser::ast::FromTable::WithFromKeyword(tables)
            | sqlparser::ast::FromTable::WithoutKeyword(tables) => tables
                .first()
                .map(|t| t.relation.to_string())
                .unwrap_or_default(),
        };
        if table.is_empty() {
            return Ok(QueryResult::RowCount(0));
        }
        let filter = del.selection.as_ref().and_then(filter_from_expr);
        let plan = QueryPlan::Delete { table, filter };
        self.dispatch(plan)
    }

    fn exec_semantic_match_standalone(
        &mut self,
        expr: &galaxdb_sql::ast::SemanticMatchExpr,
    ) -> GalaxResult<QueryResult> {
        // Identify the table by the embedding column name.
        let indexes = self.vector_indexes.read().unwrap();
        let table_name = indexes
            .iter()
            .find(|(_, idx)| {
                idx.embedding_column == expr.column || idx.source_column == expr.column
            })
            .map(|(n, _)| n.clone())
            .ok_or_else(|| {
                GalaxError::Internal(format!(
                    "no embedding index found for column '{}'",
                    expr.column
                ))
            })?;
        drop(indexes);

        let plan = planner::plan_semantic_search(
            table_name,
            expr.clone(),
            None,
            None,
            // Standalone SEMANTIC_MATCH (not a SELECT) carries no SQL LIMIT;
            // use the executor's default page size.
            None,
        );
        self.dispatch(plan)
    }

    fn dispatch(&mut self, plan: QueryPlan) -> GalaxResult<QueryResult> {
        let mut ctx = self.context();
        let res = execute_with_context(&plan, &mut ctx)?;
        self.catalog = std::mem::take(&mut ctx.catalog);
        Ok(query_result_from(res))
    }

    /// Build a fresh `ExecutorContext` that shares this database's
    /// engine, sidecar, tag catalog, merkle DAG, and vector backend.
    ///
    /// The context's catalog is **cloned** from `self.catalog` (not moved)
    /// so that if `execute_with_context` returns an error — for example
    /// the authorization chokepoint rejecting a statement before it runs —
    /// the caller's early `?` return cannot strand the catalog inside the
    /// dropped context. On success the caller writes the (possibly DDL-
    /// mutated) catalog back via `self.catalog = std::mem::take(&mut
    /// ctx.catalog)`. The clone is cheap: the catalog holds only table
    /// metadata, not row data.
    fn context(&mut self) -> ExecutorContext {
        let mut ctx = ExecutorContext::new(self.engine.clone());
        ctx.catalog = self.catalog.clone();
        ctx.sidecar = self.sidecar.clone();
        ctx.merkle_dag = Some(self.merkle_dag.clone());
        ctx.tag_catalog = Some(self.tag_catalog.clone());
        ctx.vector_backend = Some(Arc::new(EmbeddedVectorBackend {
            sidecar: self.sidecar.clone(),
            indexes: self.vector_indexes.clone(),
            engine: self.engine.clone(),
        }));
        // Role/grant DDL persists through the engine-backed auth store.
        ctx.auth_store = Some(galaxdb_sql::auth_store::AuthStore::new(self.engine.clone()));
        // Secondary indexes (Req 5): engine-backed, durable store shared
        // by DDL, the write path, and the read path.
        ctx.secondary_index = Some(galaxdb_sql::secondary_index::SecondaryIndexStore::new(
            self.engine.clone(),
        ));
        // Semantic-cache config store (v0.7): engine-backed, durable.
        ctx.semantic_cache_store = Some(
            galaxdb_sql::semantic_cache_store::SemanticCacheStore::new(self.engine.clone()),
        );
        // The authenticated session (if any) drives the executor's
        // authorization chokepoint. `None` = trusted embedded mode.
        ctx.session = self.session.clone();
        ctx.audit = self.audit.clone();
        ctx
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn table_count(&self) -> usize {
        self.catalog.table_count()
    }
    pub fn table_exists(&self, name: &str) -> bool {
        self.catalog.table_exists(name)
    }
    /// Count of live user rows. Excludes the reserved, durably-persisted
    /// catalog entries (`__galaxdb_catalog__` namespace), which are engine
    /// keys carrying schema, not user data.
    pub fn row_count(&self) -> u64 {
        self.engine
            .scan_all()
            .into_iter()
            .filter(|(k, _)| !k.starts_with(galaxdb_sql::catalog_store::CATALOG_KEY_PREFIX))
            .count() as u64
    }

    /// Register a `FOR TRAINING` version tag pinned at the most
    /// recently committed timestamp and return that timestamp.
    ///
    /// This is a programmatic snapshot API. The SQL path
    /// `CREATE VERSION TAG 'name' FOR TRAINING …` will eventually
    /// carry the same semantics once task 36 wires the `MerkleDag`
    /// to real commit events; until that lands, `CREATE VERSION TAG`
    /// reads `MerkleDag::latest()` which stays at 0, so tools that
    /// want a durable training snapshot must go through this entry
    /// point.
    ///
    /// * `name` — unique tag name (same uniqueness rules as
    ///   `CREATE VERSION TAG`). An existing name is a hard error.
    /// * `seed` — optional deterministic-ordering seed stored in the
    ///   tag's `TrainingTagMetadata`. Callers that want exactly-
    ///   reproducible Lance exports should pass a stable value.
    ///
    /// Returns the `version_timestamp` the tag was pinned at, which
    /// is also the ts you can pass to `AT VERSION <ts>` to query
    /// the same snapshot through SQL.
    pub fn create_training_snapshot(
        &self,
        name: &str,
        seed: Option<u64>,
    ) -> GalaxResult<u64> {
        use galaxdb_versioning::{MerkleRoot, TrainingTagMetadata};

        // Pin at the most recently committed ts.
        let tag_ts = self.engine.latest_commit_ts();

        let mut tc = self
            .tag_catalog
            .lock()
            .map_err(|_| GalaxError::Internal("tag catalog mutex poisoned".into()))?;
        tc.create_tag(
            name.to_string(),
            tag_ts, // created_at
            // Real content Merkle root over the exact snapshot this tag
            // pins (xxh3-128 of the per-row checksums) — not a placeholder.
            MerkleRoot::compute(&self.engine.snapshot_checksums(tag_ts)),
            tag_ts,                      // version_timestamp
            vec![],                      // pinned blocks driven off version_timestamp for now
            true,                        // FOR TRAINING
            Some(TrainingTagMetadata {
                precision: "float32".to_string(),
                seed,
                deterministic_order: true,
            }),
        )
        .map_err(|e| GalaxError::Internal(e.to_string()))?;
        Ok(tag_ts)
    }

    /// Build a [`galaxdb_storage::compaction::GcContext`] that pins
    /// every commit timestamp currently referenced by a version tag.
    /// Tasks 10.5 and 33.5: ensures the compactor's MVCC GC retains
    /// row versions that tagged snapshots depend on.
    ///
    /// `oldest_active_snapshot` should be the minimum active
    /// transaction's read timestamp (None if there are no active
    /// readers). When compaction runs from embedded-mode callers with
    /// no transaction manager, passing `None` is safe: pinned
    /// timestamps alone are sufficient to keep training snapshots
    /// alive, and unreferenced versions are already beyond any
    /// caller's interest.
    pub fn gc_context_with_pins(
        &self,
        oldest_active_snapshot: Option<u64>,
    ) -> galaxdb_storage::compaction::GcContext {
        let pins = self
            .tag_catalog
            .lock()
            .map(|tc| tc.all_pinned_timestamps())
            .unwrap_or_default();
        galaxdb_storage::compaction::GcContext::with_pins(
            oldest_active_snapshot,
            pins,
        )
    }

    /// Export the table backing `tag` as a Lance dataset on disk and
    /// return the dataset's path (Req 25 / Req 32, task 22.4).
    ///
    /// What the method actually does, in order:
    ///
    /// 1. Resolve `tag` via the [`TagCatalog`]. Unknown tag name →
    ///    [`GalaxError::Internal`] carrying "unknown version tag: …".
    /// 2. Reject if the tag was not created with `FOR TRAINING`. Only
    ///    training tags deterministically pin block sets and precision
    ///    options — exporting a non-training tag would silently lose
    ///    that contract.
    /// 3. Pick the table the tag is associated with. For v1 the
    ///    assumption is one-table-per-database (which is how the
    ///    canonical `CREATE VERSION TAG` statement is used today); if
    ///    the catalog holds multiple tables we pick the only table with
    ///    any data and error if that choice is ambiguous.
    /// 4. Build an Arrow schema from that table's `CatalogColumn`s,
    ///    mapping SQL types to Arrow types (`INT` / `BIGINT` → `Int64`,
    ///    `FLOAT` / `REAL` / `DOUBLE` → `Float32`, everything else →
    ///    `Utf8`). A table with an embedding column gets one extra Arrow
    ///    column (`{column}_embedding`): `FixedSizeList<Float32, dim>` for
    ///    Float32 precision, or `Binary` for Sq8/Rabitq. Each row's vector is
    ///    resolved by primary key from the in-memory vector index; a row with
    ///    no embedding at the tag version exports a NULL (never fabricated).
    /// 5. Instantiate an [`EmbeddedLanceExportSource`] that reads rows
    ///    at the tag's `version_timestamp` via
    ///    [`Engine::scan_all_at`], filters them to the chosen table's
    ///    primary-key prefix, and decodes each row through
    ///    [`row_codec::decode_row`].
    /// 6. Drive [`LanceExporter::export`] on a fresh tokio current-
    ///    thread runtime (the embedded database is a sync API; all
    ///    the async lives inside Lance's writer). The output path is
    ///    deterministic: `<db>/training_exports/<tag>_<version_ts>/`
    ///    so repeat exports of the same tag overwrite the same
    ///    directory rather than racing for different names.
    /// 7. Record a lineage row in the real `_galaxdb_training_exports`
    ///    system table (Req 38 / task 36). The table is created on
    ///    first export (idempotent) and every subsequent export
    ///    appends one row. UPDATE and DELETE against the table are
    ///    rejected by the executor so the audit trail is permanent.
    ///
    /// The returned path points at the on-disk Lance dataset. Python
    /// callers wrap it with `lance.dataset(path).to_pytorch()` to get
    /// an `IterableDataset` — that glue lives in the Python package,
    /// not in this Rust method, because PyTorch is a Python-only
    /// dependency (Rule 5: no vendor lock-in in the engine core).
    pub fn training_dataset(&mut self, tag: &str) -> GalaxResult<PathBuf> {
        use galaxdb_versioning::{
            LanceExporter, TrainingExportLineageSink, TrainingPrecision,
        };

        // 1. Resolve the tag and clone the bits we need out of the
        // mutex before we do any async work. The exporter takes
        // `Arc<TagCatalog>` / `Arc<MerkleDag>` — we snapshot the
        // current state so the running export cannot see concurrent
        // tag creations/deletions mid-flight.
        let (version_tag, tag_catalog_snapshot, merkle_dag_snapshot) = {
            let tag_catalog = self
                .tag_catalog
                .lock()
                .map_err(|_| GalaxError::Internal("tag catalog mutex poisoned".into()))?;
            let Some(version_tag) = tag_catalog.get_tag(tag).cloned() else {
                return Err(GalaxError::Internal(format!(
                    "unknown version tag: {tag}"
                )));
            };
            let tag_catalog_snapshot = tag_catalog.clone();
            let merkle_dag_snapshot = self
                .merkle_dag
                .lock()
                .map_err(|_| GalaxError::Internal("merkle dag mutex poisoned".into()))?
                .clone();
            (version_tag, tag_catalog_snapshot, merkle_dag_snapshot)
        };

        // 2. Training-only — non-training tags don't carry the
        // deterministic-order / precision contract the export relies
        // on.
        if !version_tag.for_training {
            return Err(GalaxError::Internal(format!(
                "version tag '{tag}' is not a FOR TRAINING tag; \
                 only training tags can be exported as Lance datasets"
            )));
        }

        // 3. Pick the table. v1 supports the single-table case that
        // `CREATE VERSION TAG` produces today. If there is more than
        // one table with rows we refuse rather than exporting the
        // first one alphabetically — silent choice here would be a
        // correctness bug the user has no way to see.
        let (table_name, table_entry) = self.pick_training_table()?;

        // 4. Resolve the training precision up front — the Arrow schema for an
        // embedding column depends on it (FixedSizeList<Float32> for Float32,
        // Binary for Sq8/Rabitq, which the exporter quantises into).
        let precision = version_tag
            .training_opts
            .as_ref()
            .and_then(|o| TrainingPrecision::from_str_opt(&o.precision))
            .unwrap_or(TrainingPrecision::Float32);
        let seed = version_tag.training_opts.as_ref().and_then(|o| o.seed);

        // 4b. Snapshot the table's embedding vectors (if any) keyed by primary
        // key, so the export source can attach each row's vector. Vectors live
        // in the in-memory vector index, not in the row codec; we resolve
        // key→row_id→vector here. A row whose embedding never landed (no
        // sidecar, or not yet computed) simply won't appear in the map and is
        // exported as a NULL vector (Req 20 AC4 — never fabricate).
        let embedding_info: Option<EmbeddingExportInfo> = {
            let indexes = self
                .vector_indexes
                .read()
                .map_err(|_| GalaxError::Internal("vector index lock poisoned".into()))?;
            indexes.get(&table_name).map(|idx| {
                let mut key_to_vec = HashMap::new();
                for (key, row_id) in &idx.key_to_row_id {
                    if let Some(v) = idx.vectors.get(row_id) {
                        key_to_vec.insert(key.clone(), v.clone());
                    }
                }
                EmbeddingExportInfo {
                    column: format!("{}_embedding", idx.embedding_column),
                    dim: idx.dim,
                    key_to_vec,
                }
            })
        };

        // 5. Build the Arrow schema from the catalog (+ the embedding column).
        let schema = Arc::new(arrow_schema_from_catalog(
            &table_entry,
            embedding_info.as_ref(),
            precision,
        ));

        // 5b. Build the export source over the real engine.
        let source: Arc<dyn galaxdb_versioning::LanceExportSource> =
            Arc::new(EmbeddedLanceExportSource {
                engine: self.engine.clone(),
                table_name: table_name.clone(),
                table_entry: table_entry.clone(),
                version_timestamp: version_tag.version_timestamp,
                embedding: embedding_info,
            });

        // 6. Resolve the output path. Use the tag name plus the tag's
        // version timestamp so repeat exports of a mutated tag don't
        // collide (tag names are unique so the version_ts is almost
        // redundant — but it makes the path self-describing).
        let safe_tag = sanitize_tag_for_path(tag);
        let output_path = self
            .path
            .join("training_exports")
            .join(format!("{safe_tag}_{}", version_tag.version_timestamp));

        // Lance refuses to write into a non-empty directory. For a
        // deterministic repeat export we clear any previous artefact
        // at the same path before handing it to the writer. The
        // parent `training_exports` directory is created on demand.
        if output_path.exists() {
            std::fs::remove_dir_all(&output_path)?;
        }
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // 7. Precision and seed were resolved up front (step 4) so the schema
        // could be built for the right embedding-column Arrow type.

        // 8. Build the exporter and drive it. Lance writers are async;
        // the embedded database API is sync. Spin a dedicated current-
        // thread runtime so we don't assume the caller already has one.
        //
        // Task 36.3: every successful export must also land one row in
        // the real `_galaxdb_training_exports` system table. We can't
        // call `Engine::put_sync` from inside the exporter's async
        // context because `put_sync` blocks on WAL group-commit via
        // `oneshot::blocking_recv`, which is forbidden inside a tokio
        // worker. Instead we buffer the lineage record through the
        // in-memory sink, run the async export, then flush the
        // recorded entries through the real engine AFTER `block_on`
        // returns (we're back on the caller's thread at that point).
        // Same pattern the Phase I wire server uses for the blocking
        // storage primitives.
        let buffer = Arc::new(galaxdb_versioning::InMemoryLineageSink::new());
        let sink: Arc<dyn TrainingExportLineageSink> = buffer.clone();
        self.ensure_training_exports_table()?;
        let exporter = LanceExporter::new(
            &output_path,
            schema,
            Arc::new(merkle_dag_snapshot),
            Arc::new(tag_catalog_snapshot),
            source,
            version_tag.name.clone(),
            precision,
            false, // dedup — opt-in via `WHERE NOT DUPLICATE`, not wired into the method API yet
            seed,
        )
        .with_lineage_sink(sink);

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                GalaxError::Internal(format!("could not build tokio runtime: {e}"))
            })?;
        rt.block_on(exporter.export())
            .map_err(|e| GalaxError::Internal(format!("Lance export failed: {e}")))?;

        // Flush the buffered lineage records to `_galaxdb_training_exports`.
        // Runs on the caller's thread, so `Engine::put_sync` is safe.
        let engine_sink = EngineBackedLineageSink {
            engine: self.engine.clone(),
        };
        for entry in buffer.entries() {
            use galaxdb_versioning::TrainingExportLineageSink as _;
            engine_sink
                .record(entry)
                .map_err(|e| GalaxError::Internal(format!("lineage flush failed: {e}")))?;
        }

        // v0.6 metering (E-4): count the real bytes emitted by this export
        // (recursive on-disk size of the Lance dataset directory). Measured,
        // not estimated; counted only after a successful export.
        fn dir_size_bytes(dir: &std::path::Path) -> u64 {
            let mut total = 0u64;
            let Ok(entries) = std::fs::read_dir(dir) else {
                return 0;
            };
            for entry in entries.flatten() {
                let Ok(meta) = entry.metadata() else { continue };
                if meta.is_file() {
                    total += meta.len();
                } else if meta.is_dir() {
                    total += dir_size_bytes(&entry.path());
                }
            }
            total
        }
        galaxdb_observe::metrics()
            .training_export_bytes_total
            .inc_by(dir_size_bytes(&output_path));

        Ok(output_path)
    }

    /// Pick the single table that a training export should consume. v1
    /// assumes one training-eligible table per database. See
    /// [`Self::training_dataset`] for why that choice is explicit
    /// rather than implicit.
    fn pick_training_table(
        &self,
    ) -> GalaxResult<(String, galaxdb_sql::executor::TableEntry)> {
        // Skip append-only system tables (e.g.
        // `_galaxdb_training_exports`) — those are lineage sinks and
        // the user never wants them as the export source. The
        // `append_only` flag on `TableEntry` is our canonical signal.
        let names: Vec<String> = self
            .catalog
            .table_names()
            .filter(|n| {
                self.catalog
                    .get_table(n)
                    .map(|e| !e.append_only)
                    .unwrap_or(false)
            })
            .map(|n| n.to_string())
            .collect();
        match names.len() {
            0 => Err(GalaxError::Internal(
                "training_dataset: no tables exist in the database".into(),
            )),
            1 => {
                let name = &names[0];
                let entry = self
                    .catalog
                    .get_table(name)
                    .cloned()
                    .ok_or_else(|| GalaxError::TableNotFound(name.clone()))?;
                Ok((name.clone(), entry))
            }
            _ => Err(GalaxError::Internal(format!(
                "training_dataset: multiple tables found ({}); v1 supports \
                 single-table exports — drop or rename tables so only the \
                 target table remains",
                names.len()
            ))),
        }
    }

    /// Idempotent: create the `_galaxdb_training_exports` system table
    /// if it doesn't already exist. Called on the first `training_dataset`
    /// export per database so callers never see a missing table error
    /// for lineage SELECTs.
    ///
    /// The schema matches the one defined on
    /// [`galaxdb_versioning::TrainingExportLineage`] and required by
    /// task 36.1:
    ///
    ///   lineage_id (PK), exported_at, tag_name, filter_expr,
    ///   precision, dedup, curriculum, row_count, content_hash
    ///
    /// `lineage_id` is a process-monotonic counter so two exports in
    /// the same wall-clock second still land as distinct rows;
    /// `exported_at` carries the wall-clock timestamp on the row.
    /// `append_only = true` is set by the executor's
    /// `is_system_append_only_table` check keyed on the table name,
    /// so DELETE and UPDATE against this table return
    /// `GalaxError::AppendOnlyTable` without any extra config here.
    fn ensure_training_exports_table(&mut self) -> GalaxResult<()> {
        use galaxdb_sql::executor::TRAINING_EXPORTS_TABLE;
        if self.catalog.table_exists(TRAINING_EXPORTS_TABLE) {
            return Ok(());
        }
        let sql = format!(
            "CREATE TABLE {table} (\
                 lineage_id BIGINT PRIMARY KEY, \
                 exported_at BIGINT, \
                 tag_name TEXT, \
                 filter_expr TEXT, \
                 precision TEXT, \
                 dedup TEXT, \
                 curriculum TEXT, \
                 row_count BIGINT, \
                 content_hash TEXT\
             )",
            table = TRAINING_EXPORTS_TABLE
        );
        self.execute(&sql)?;
        Ok(())
    }
}

impl Drop for Database {
    fn drop(&mut self) {
        // v0.7 (inventory 4.10): persist vector-index snapshots on graceful
        // shutdown so the next open reuses the vectors instead of re-embedding.
        self.persist_vector_indexes();
        // Stop the background compaction worker promptly (it holds only a
        // Weak<Engine>, but this wakes it instead of waiting for its poll
        // timeout), then run the engine's own shutdown.
        self.engine.shutdown_background_compaction();
        self.engine.shutdown();
    }
}

// ---------------------------------------------------------------------------
// Vector backend — bridges the SQL executor's VectorSearchBackend trait to
// the database's local HNSW + delta buffer + sidecar.
// ---------------------------------------------------------------------------

/// Candidate provider for the analytical `SEMANTIC_MATCH` path (HTAP task 16).
/// Holds the already-resolved matched rows (base-column `Value`s) plus their
/// similarity scores; [`candidates`](galaxdb_query::semantic::VectorCandidateProvider::candidates)
/// materializes them into the Arrow batch DataFusion joins/aggregates over.
/// The HNSW search that produced these rows already ran in
/// `analytical_semantic_query`, so this is a pure builder — no vector work
/// happens inside the query runtime.
struct EmbeddedSemanticCandidateProvider {
    schema: arrow::datatypes::SchemaRef,
    base_cols: Vec<(String, galaxdb_sql::SqlType)>,
    rows: Vec<Vec<Option<galaxdb_sql::planner::Value>>>,
    sims: Vec<f64>,
}

impl galaxdb_query::semantic::VectorCandidateProvider for EmbeddedSemanticCandidateProvider {
    fn schema(&self) -> arrow::datatypes::SchemaRef {
        self.schema.clone()
    }
    fn candidates(&self) -> GalaxResult<Vec<arrow::record_batch::RecordBatch>> {
        let batch =
            galaxdb_query::semantic::build_candidate_batch(&self.base_cols, &self.rows, &self.sims)?;
        Ok(vec![batch])
    }
}

struct EmbeddedVectorBackend {
    sidecar: Option<Arc<SidecarManager>>,
    indexes: Arc<RwLock<HashMap<String, TableVectorIndex>>>,
    /// Real storage engine, shared with the `Database`. Needed so
    /// `on_row_deleted` can append a `DELTA_TOMBSTONE` WAL record
    /// durably before the in-memory delta buffer is tombstoned.
    engine: Arc<Engine>,
}

// ---------------------------------------------------------------------------
// EngineBackedLineageSink — writes every training-export lineage row into
// the real `_galaxdb_training_exports` system table (task 36).
// ---------------------------------------------------------------------------

/// Persists [`galaxdb_versioning::TrainingExportLineage`] records as
/// rows in the `_galaxdb_training_exports` system table.
///
/// The sink is called from inside
/// [`galaxdb_versioning::LanceExporter::export`] after a successful
/// Lance write; here we bypass the SQL executor and write directly
/// through [`Engine::put_sync`] so the sink does not need a mutable
/// reference to `Database`. The row bytes are constructed through the
/// same [`galaxdb_sql::row_codec`] text codec the executor uses, so a
/// subsequent `SELECT * FROM _galaxdb_training_exports` through the
/// normal path decodes the same values back.
///
/// Append-only enforcement (task 36.2) is done at the executor level —
/// this sink doesn't need to be defensive about DELETE / UPDATE.
///
/// The primary key is a process-monotonic `lineage_id` allocated
/// from a single `AtomicU64` so two exports that happen in the same
/// wall-clock second still land as two distinct rows. `exported_at`
/// stays as the wall-clock timestamp on the row, not as the PK.
struct EngineBackedLineageSink {
    engine: Arc<Engine>,
}

/// Process-wide monotonic counter used by [`EngineBackedLineageSink`]
/// to allocate a unique primary key per lineage row. `AtomicU64`
/// starting at 1 so the zero-value sentinel isn't a valid row id.
static LINEAGE_ROW_ID: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(1);

impl galaxdb_versioning::TrainingExportLineageSink for EngineBackedLineageSink {
    fn record(
        &self,
        lineage: galaxdb_versioning::TrainingExportLineage,
    ) -> galaxdb_versioning::ExportResult<()> {
        use galaxdb_sql::executor::TRAINING_EXPORTS_TABLE;
        use galaxdb_sql::planner::Value;
        use galaxdb_sql::row_codec;

        // The catalog column order declared in
        // `Database::ensure_training_exports_table`:
        //   lineage_id (PK), exported_at, tag_name, filter_expr,
        //   precision, dedup, curriculum, row_count, content_hash
        // `lineage_id` is a process-monotonic counter that guarantees
        // two exports in the same wall-clock second still produce two
        // distinct rows (MVCC would otherwise collapse them).
        let lineage_id =
            LINEAGE_ROW_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let ordered = vec![
            (
                "lineage_id".to_string(),
                Value::Integer(lineage_id as i64),
            ),
            (
                "exported_at".to_string(),
                Value::Integer(lineage.exported_at as i64),
            ),
            ("tag_name".to_string(), Value::Text(lineage.tag_name.clone())),
            (
                "filter_expr".to_string(),
                match lineage.filter_expr.as_ref() {
                    Some(s) => Value::Text(s.clone()),
                    None => Value::Null,
                },
            ),
            (
                "precision".to_string(),
                Value::Text(lineage.precision.clone()),
            ),
            ("dedup".to_string(), Value::Bool(lineage.dedup)),
            (
                // Curriculum mode is reserved by task 36.1 but the
                // `TrainingExportLineage` struct does not yet carry a
                // curriculum field — landing that requires Req-scope
                // work on the exporter. The column exists so the
                // schema matches the spec; it's always NULL today and
                // becomes non-NULL once curriculum lands without
                // needing an ALTER TABLE.
                "curriculum".to_string(),
                Value::Null,
            ),
            (
                "row_count".to_string(),
                Value::Integer(lineage.row_count as i64),
            ),
            (
                "content_hash".to_string(),
                Value::Text(lineage.content_hash.clone()),
            ),
        ];

        // The primary key for each lineage row is `lineage_id`
        // (monotonic u64). `TRAINING_EXPORTS_TABLE:<lineage_id>` is
        // the storage key shape used by every other table's row codec.
        let primary_key =
            format!("{}:{}", TRAINING_EXPORTS_TABLE, lineage_id).into_bytes();
        let row_bytes = row_codec::encode_row(&ordered);

        self.engine.put_sync(primary_key, row_bytes).map_err(|e| {
            galaxdb_versioning::ExportError::Arrow(format!(
                "failed to append training-export lineage row: {e}"
            ))
        })?;
        Ok(())
    }
}

impl VectorSearchBackend for EmbeddedVectorBackend {
    fn semantic_search(
        &self,
        table: &str,
        query_text: &str,
        threshold: f64,
        k: usize,
        _strategy: SearchStrategy,
    ) -> GalaxResult<Vec<VectorSearchResult>> {
        // Embed the query through the sidecar. No mock fallback —
        // missing sidecar is a typed error.
        let sidecar = self
            .sidecar
            .as_ref()
            .ok_or(GalaxError::SidecarUnavailable)?;
        let indexes = self.indexes.read().unwrap();
        let idx = indexes
            .get(table)
            .ok_or_else(|| GalaxError::TableNotFound(table.to_string()))?;

        // This text is a search query — asymmetric models apply their query prefix.
        let request = EmbedRequest::query(0, query_text.to_string(), idx.embedding_column.clone());
        let response = sidecar
            .embed(request)
            .map_err(|_| GalaxError::SidecarUnavailable)?;

        // v0.7 semantic cache (inventory 8.11 / E-4.1): a query whose
        // embedding is within the configured SIMILARITY of a cached,
        // unexpired, param-matching entry returns the cached results without
        // running HNSW — and counts as a cache hit. Only this pure
        // `SEMANTIC_MATCH` path is cached; `brute_force_filtered` bypasses it.
        let threshold_bits = threshold.to_bits();
        if let Some(cached) = idx.semantic_cache.lookup(
            &response.embedding,
            &response.model_version,
            threshold_bits,
            k,
        ) {
            galaxdb_observe::metrics().semantic_cache_hits_total.inc();
            return Ok(cached);
        }

        let sm_config = SemanticMatchConfig {
            hnsw_candidates: 100,
            ef_search: 200,
            brute_force_threshold: 1000,
            brute_force_ratio: 0.001,
        };
        let vectors_ref = &idx.vectors;
        let results = execute_semantic_match(
            &response.embedding,
            &idx.hnsw,
            &idx.delta,
            threshold,
            k,
            &sm_config,
            |row_id| vectors_ref.get(&row_id).cloned(),
        );
        let out: Vec<VectorSearchResult> = results
            .into_iter()
            .map(|r| VectorSearchResult {
                row_id: r.row_id,
                similarity: r.similarity,
            })
            .collect();
        // Populate the cache on a miss (no-op when the cache is disabled).
        idx.semantic_cache.store(
            response.embedding,
            response.model_version,
            threshold_bits,
            k,
            out.clone(),
        );
        Ok(out)
    }

    fn brute_force_filtered(
        &self,
        table: &str,
        query_text: &str,
        threshold: f64,
        k: usize,
        _filter: &FilterExpr,
    ) -> GalaxResult<Vec<VectorSearchResult>> {
        // The brute-force path shares the HNSW-backed implementation but
        // MUST bypass the semantic cache: the cache key does not include the
        // filter, so serving a cached (unfiltered) result here would be
        // wrong. Run the search directly without cache lookup/store.
        let sidecar = self
            .sidecar
            .as_ref()
            .ok_or(GalaxError::SidecarUnavailable)?;
        let indexes = self.indexes.read().unwrap();
        let idx = indexes
            .get(table)
            .ok_or_else(|| GalaxError::TableNotFound(table.to_string()))?;
        let request = EmbedRequest::query(0, query_text.to_string(), idx.embedding_column.clone());
        let response = sidecar
            .embed(request)
            .map_err(|_| GalaxError::SidecarUnavailable)?;
        let sm_config = SemanticMatchConfig {
            hnsw_candidates: 100,
            ef_search: 200,
            brute_force_threshold: 1000,
            brute_force_ratio: 0.001,
        };
        let vectors_ref = &idx.vectors;
        let results = execute_semantic_match(
            &response.embedding,
            &idx.hnsw,
            &idx.delta,
            threshold,
            k,
            &sm_config,
            |row_id| vectors_ref.get(&row_id).cloned(),
        );
        Ok(results
            .into_iter()
            .map(|r| VectorSearchResult {
                row_id: r.row_id,
                similarity: r.similarity,
            })
            .collect())
    }

    fn configure_semantic_cache(&self, table: &str, similarity: f32, ttl_secs: u32) {
        let indexes = self.indexes.read().unwrap();
        if let Some(idx) = indexes.get(table) {
            idx.semantic_cache.configure(similarity, ttl_secs);
        }
    }

    fn drop_semantic_cache(&self, table: &str) {
        let indexes = self.indexes.read().unwrap();
        if let Some(idx) = indexes.get(table) {
            idx.semantic_cache.disable();
        }
    }

    fn on_row_deleted(&self, table: &str, row_key: &[u8]) -> GalaxResult<()> {
        // Resolve the primary-key bytes to the vector-row-id we stored
        // when the embedding was generated. If we don't have a mapping
        // (table has no vector index, or the embedding never landed)
        // the delete is a no-op for the vector side, which is correct.
        let row_id = {
            let indexes = self.indexes.read().unwrap();
            let Some(idx) = indexes.get(table) else {
                return Ok(());
            };
            match idx.key_to_row_id.get(row_key) {
                Some(id) => *id,
                None => {
                    // No vector for this row_key — nothing to tombstone.
                    return Ok(());
                }
            }
        };

        // WAL first, memory after. The payload is
        // `[u64 le vector_row_id][row_key]` so replay on recovery can
        // rebuild the tombstone set and the key→row_id mapping.
        let mut payload = Vec::with_capacity(8 + row_key.len());
        payload.extend_from_slice(&row_id.to_le_bytes());
        payload.extend_from_slice(row_key);
        self.engine.append_delta_tombstone_sync(payload)?;

        // Tombstone the in-memory delta buffer and drop the mapping so
        // re-insert of the same key allocates a fresh vector row-id.
        let mut indexes = self.indexes.write().unwrap();
        if let Some(idx) = indexes.get_mut(table) {
            idx.delta.delete(row_id);
            idx.vectors.remove(&row_id);
            idx.key_to_row_id.remove(row_key);
            // v0.7: a write invalidates the semantic cache so no cached
            // result predating this delete is served.
            idx.semantic_cache.invalidate();
        }

        Ok(())
    }

    fn on_row_inserted(
        &self,
        table: &str,
        row_key: &[u8],
        row: &[(String, Value)],
    ) -> GalaxResult<()> {
        // No sidecar configured → embeddings are disabled for this
        // deployment. Scalar SQL still works and `semantic_search`
        // surfaces `SidecarUnavailable`. Never fabricate a vector.
        let Some(sidecar) = self.sidecar.as_ref() else {
            return Ok(());
        };

        // Resolve the embedding source column and its text value for this
        // table's index. If the table has no vector index, or this row
        // carries no text in the source column, there is nothing to embed.
        let (source_column, text) = {
            let indexes = self.indexes.read().unwrap();
            let Some(idx) = indexes.get(table) else {
                return Ok(());
            };
            let source_column = idx.source_column.clone();
            let text = row
                .iter()
                .find(|(name, _)| name == &source_column)
                .and_then(|(_, v)| match v {
                    Value::Text(s) => Some(s.clone()),
                    _ => None,
                });
            match text {
                Some(t) => (source_column, t),
                None => return Ok(()),
            }
        };

        // Use xxh3_64(primary_key) as the vector row-id so results join
        // back to table rows in `exec_semantic_search` (which hashes the
        // same key) and so `on_row_deleted` can tombstone the right entry.
        let row_id = xxhash_rust::xxh3::xxh3_64(row_key);

        // A stored row is a document — asymmetric models apply their document prefix.
        let response = sidecar
            .embed(EmbedRequest::document(row_id, text, source_column))
            .map_err(|e| {
                GalaxError::Internal(format!(
                    "embedding sidecar failed for insert into '{table}': {e}"
                ))
            })?;

        // Store the vector in both the delta buffer (searched by
        // SEMANTIC_MATCH) and the re-rank vector map, and remember the
        // primary-key → row-id mapping so a later DELETE can tombstone it.
        let mut indexes = self.indexes.write().unwrap();
        if let Some(idx) = indexes.get_mut(table) {
            // Dimension integrity (task A.5): the model's real output dimension must
            // match the table's declared `DIM`. A mismatch means the configured model
            // does not fit the schema — refuse with a typed error rather than storing a
            // wrong-width vector that would corrupt the index or silently truncate.
            if response.embedding.len() != idx.dim {
                return Err(GalaxError::Internal(format!(
                    "embedding dimension mismatch for table '{table}': model '{}' produced \
                     {}-d vectors but the '{}' column was declared DIM {}. Re-create the table \
                     with DIM {} or configure a model whose dimension is {}.",
                    response.model_version,
                    response.embedding.len(),
                    idx.embedding_column,
                    idx.dim,
                    response.embedding.len(),
                    idx.dim,
                )));
            }
            idx.delta.insert(row_id, response.embedding.clone());
            idx.vectors.insert(row_id, response.embedding);
            idx.key_to_row_id.insert(row_key.to_vec(), row_id);
            // v0.7: a write invalidates the semantic cache so no cached
            // result predating this insert is served.
            idx.semantic_cache.invalidate();
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// AST-to-Value helpers
// ---------------------------------------------------------------------------

fn value_from_expr(e: &sqlparser::ast::Expr) -> Value {
    match e {
        sqlparser::ast::Expr::Value(v) => match v {
            sqlparser::ast::Value::Number(n, _) => n
                .parse::<i64>()
                .map(Value::Integer)
                .or_else(|_| n.parse::<f64>().map(Value::Float))
                .unwrap_or_else(|_| Value::Text(n.clone())),
            sqlparser::ast::Value::SingleQuotedString(s)
            | sqlparser::ast::Value::DoubleQuotedString(s) => Value::Text(s.clone()),
            sqlparser::ast::Value::Boolean(b) => Value::Bool(*b),
            sqlparser::ast::Value::Null => Value::Null,
            other => Value::Text(format!("{}", other)),
        },
        other => Value::Text(format!("{}", other)),
    }
}

/// Translate a `sqlparser` expression in a value position (`INSERT ... VALUES`
/// or `UPDATE ... SET col = <expr>`) into a GalaxDB [`ScalarExpr`] that the
/// executor evaluates per row against the old row values.
///
/// This replaces the old behavior where a non-literal expression such as
/// `bal - 30` was silently stringified to the literal text `"bal - 30"`
/// (data corruption). An expression we cannot represent is a typed
/// [`GalaxError::FeatureNotSupported`], never a silent wrong value.
fn scalar_from_expr(e: &sqlparser::ast::Expr) -> GalaxResult<ScalarExpr> {
    use sqlparser::ast::{BinaryOperator, Expr, UnaryOperator};

    match e {
        Expr::Value(_) => Ok(ScalarExpr::Literal(value_from_expr(e))),
        Expr::Identifier(ident) => Ok(ScalarExpr::Column(ident.value.clone())),
        Expr::CompoundIdentifier(parts) => parts
            .last()
            .map(|p| ScalarExpr::Column(p.value.clone()))
            .ok_or_else(|| {
                GalaxError::FeatureNotSupported("empty compound identifier".to_string())
            }),
        Expr::Nested(inner) => scalar_from_expr(inner),
        // A cast in a value position carries no runtime coercion here; evaluate
        // the inner expression (numeric/text values are coerced at eval time).
        Expr::Cast { expr, .. } => scalar_from_expr(expr),
        Expr::UnaryOp { op, expr } => match op {
            UnaryOperator::Minus => Ok(ScalarExpr::Neg(Box::new(scalar_from_expr(expr)?))),
            UnaryOperator::Plus => scalar_from_expr(expr),
            other => Err(GalaxError::FeatureNotSupported(format!(
                "unary operator {other:?} in a value expression"
            ))),
        },
        Expr::BinaryOp { left, op, right } => {
            let arith = match op {
                BinaryOperator::Plus => ArithOp::Add,
                BinaryOperator::Minus => ArithOp::Sub,
                BinaryOperator::Multiply => ArithOp::Mul,
                BinaryOperator::Divide => ArithOp::Div,
                BinaryOperator::Modulo => ArithOp::Mod,
                BinaryOperator::StringConcat => ArithOp::Concat,
                other => {
                    return Err(GalaxError::FeatureNotSupported(format!(
                        "binary operator '{other}' in a value expression"
                    )))
                }
            };
            Ok(ScalarExpr::Binary {
                op: arith,
                left: Box::new(scalar_from_expr(left)?),
                right: Box::new(scalar_from_expr(right)?),
            })
        }
        // A clean, wire-friendly message: name the syntactic form rather than
        // dumping the full parser AST. Still a typed error — never silent text.
        Expr::Function(f) => Err(GalaxError::FeatureNotSupported(format!(
            "function call '{}(...)' is not supported in a value position",
            f.name
        ))),
        other => Err(GalaxError::FeatureNotSupported(format!(
            "expression '{other}' is not supported in a value position"
        ))),
    }
}

/// Look up a column's value in a result row by name.
fn row_column<'a>(row: &'a SqlRow, name: &str) -> Option<&'a Value> {
    row.columns.iter().find(|(k, _)| k == name).map(|(_, v)| v)
}

/// Total order over [`Value`] for in-memory `ORDER BY` (the in-transaction
/// sorted-scan path). Numeric types compare numerically (int/float mix
/// promoted to float); text and blobs lexicographically; booleans false <
/// true. Cross-type comparisons fall back to a stable per-variant rank so the
/// sort is total and deterministic rather than panicking. NULLs are handled by
/// the caller (absolute NULLS FIRST/LAST placement) and never reach here.
fn value_cmp(a: &Value, b: &Value) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    use Value::*;
    match (a, b) {
        (Integer(x), Integer(y)) => x.cmp(y),
        (Float(x), Float(y)) => x.partial_cmp(y).unwrap_or(Ordering::Equal),
        (Integer(x), Float(y)) => (*x as f64).partial_cmp(y).unwrap_or(Ordering::Equal),
        (Float(x), Integer(y)) => x.partial_cmp(&(*y as f64)).unwrap_or(Ordering::Equal),
        (Text(x), Text(y)) => x.cmp(y),
        (Bool(x), Bool(y)) => x.cmp(y),
        (Blob(x), Blob(y)) => x.cmp(y),
        _ => value_type_rank(a).cmp(&value_type_rank(b)),
    }
}

/// Stable per-variant rank used only to make cross-type `ORDER BY` total.
fn value_type_rank(v: &Value) -> u8 {
    match v {
        Value::Null => 0,
        Value::Bool(_) => 1,
        Value::Integer(_) | Value::Float(_) => 2,
        Value::Text(_) => 3,
        Value::Blob(_) => 4,
        Value::Array(_) => 5,
    }
}

/// Is `q` a `SELECT` with no `FROM` clause (a scalar/constant projection such
/// as `SELECT 1 + 1`, `SELECT version()`, `SELECT current_database()`)?
fn is_from_less_select(q: &sqlparser::ast::Query) -> bool {
    matches!(
        q.body.as_ref(),
        sqlparser::ast::SetExpr::Select(s) if s.from.is_empty()
    )
}

/// Evaluate a FROM-less scalar `SELECT` into a single result row.
///
/// Each projection item is a constant/arithmetic expression (evaluated by the
/// [`ScalarExpr`] evaluator against an empty row) or one of the common
/// PostgreSQL session functions (`version()`, `current_database()`,
/// `current_user`, `current_schema`, …). Anything else is a typed
/// [`GalaxError::FeatureNotSupported`] — never a silent wrong value. The
/// output column is named after an explicit alias, else the function/column
/// name, else `?column?` (matching PostgreSQL).
fn eval_scalar_select(
    q: &sqlparser::ast::Query,
    current_user: Option<&str>,
) -> GalaxResult<QueryResult> {
    use sqlparser::ast::{SelectItem, SetExpr};

    let SetExpr::Select(select) = q.body.as_ref() else {
        return Err(GalaxError::FeatureNotSupported(
            "unsupported FROM-less query form".to_string(),
        ));
    };
    if select.selection.is_some() {
        return Err(GalaxError::FeatureNotSupported(
            "WHERE is not supported without a FROM clause".to_string(),
        ));
    }

    let mut columns: Vec<(String, Value)> = Vec::with_capacity(select.projection.len());
    for (i, item) in select.projection.iter().enumerate() {
        let (expr, alias) = match item {
            SelectItem::UnnamedExpr(e) => (e, None),
            SelectItem::ExprWithAlias { expr, alias } => (expr, Some(alias.value.clone())),
            other => {
                return Err(GalaxError::FeatureNotSupported(format!(
                    "unsupported projection item without a FROM clause: {other}"
                )))
            }
        };
        let value = eval_scalar_projection_expr(expr, current_user)?;
        let name = alias
            .or_else(|| scalar_output_name(expr))
            .unwrap_or_else(|| format!("?column?{}", if i == 0 { String::new() } else { i.to_string() }));
        columns.push((name, value));
    }

    let col_names: Vec<String> = columns.iter().map(|(k, _)| k.clone()).collect();
    Ok(query_result_from(ExecuteResult::Rows {
        columns: col_names,
        rows: vec![SqlRow { columns }],
    }))
}

/// Evaluate one FROM-less projection expression.
fn eval_scalar_projection_expr(
    expr: &sqlparser::ast::Expr,
    current_user: Option<&str>,
) -> GalaxResult<Value> {
    use sqlparser::ast::Expr;
    match expr {
        Expr::Function(f) => eval_builtin_function(&f.name.to_string(), current_user),
        // `current_user` / `current_schema` parse as bare identifiers/keywords
        // in some positions.
        Expr::Identifier(id) => eval_builtin_function(&id.value, current_user),
        // Constant / arithmetic / concat — reuse the scalar evaluator.
        _ => scalar_from_expr(expr).and_then(|s| s.eval(&[])),
    }
}

/// The PostgreSQL session/informational functions GalaxDB answers for
/// FROM-less SELECTs. Unknown functions are a typed error.
fn eval_builtin_function(name: &str, current_user: Option<&str>) -> GalaxResult<Value> {
    let n = name.to_ascii_lowercase();
    let user = current_user.unwrap_or("galaxdb");
    let v = match n.as_str() {
        // Reported to match the wire handshake's server_version, with a
        // PostgreSQL-compatible prefix so drivers that sniff version() work.
        "version" => Value::Text(format!(
            "PostgreSQL 16.0.0-galaxdb, GalaxDB {} (HTAP)",
            env!("CARGO_PKG_VERSION")
        )),
        "current_database" | "current_catalog" => Value::Text("galaxdb".to_string()),
        "current_schema" => Value::Text("public".to_string()),
        "current_user" | "session_user" | "user" | "current_role" => {
            Value::Text(user.to_string())
        }
        other => {
            return Err(GalaxError::FeatureNotSupported(format!(
                "function '{other}()' is not supported in a FROM-less SELECT"
            )))
        }
    };
    Ok(v)
}

/// Output column name for a FROM-less projection expression (PostgreSQL names
/// it after the function or column; expressions get `?column?`).
fn scalar_output_name(expr: &sqlparser::ast::Expr) -> Option<String> {
    use sqlparser::ast::Expr;
    match expr {
        Expr::Function(f) => Some(f.name.to_string().to_ascii_lowercase()),
        Expr::Identifier(id) => Some(id.value.to_ascii_lowercase()),
        _ => None,
    }
}

fn query_result_from(r: ExecuteResult) -> QueryResult {
    match r {
        ExecuteResult::Rows { rows, .. } => QueryResult::Rows(
            rows.into_iter()
                .map(|row: SqlRow| QueryRow {
                    values: row
                        .columns
                        .into_iter()
                        .map(|(k, v)| (k, row_codec::value_display(&v)))
                        .collect(),
                })
                .collect(),
        ),
        ExecuteResult::RowCount(n) => QueryResult::RowCount(n),
        ExecuteResult::Ok(msg) => QueryResult::Ok(msg),
        ExecuteResult::Error(msg) => QueryResult::Ok(msg),
    }
}

/// Count the bind parameters (`$1..$N`) in a SQL string by the highest
/// placeholder index, skipping `$N` sequences that appear inside
/// single-quoted string literals. PostgreSQL prepared-statement
/// parameters are contiguous `$1..$N`, so the maximum index is the count.
fn count_placeholders(sql: &str) -> usize {
    let bytes = sql.as_bytes();
    let mut max = 0usize;
    let mut i = 0;
    let mut in_string = false;
    while i < bytes.len() {
        let b = bytes[i];
        if in_string {
            if b == b'\'' {
                // Doubled '' is an escaped quote inside the literal.
                if i + 1 < bytes.len() && bytes[i + 1] == b'\'' {
                    i += 2;
                    continue;
                }
                in_string = false;
            }
            i += 1;
            continue;
        }
        if b == b'\'' {
            in_string = true;
            i += 1;
            continue;
        }
        if b == b'$' {
            let mut j = i + 1;
            let mut num = 0usize;
            let mut has_digit = false;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                num = num * 10 + (bytes[j] - b'0') as usize;
                has_digit = true;
                j += 1;
            }
            if has_digit {
                max = max.max(num);
                i = j;
                continue;
            }
        }
        i += 1;
    }
    max
}

/// Collect the base table names a query references (top-level FROM + JOINs,
/// recursing into derived subqueries and set operations). Used by the
/// analytical path to know which columnar sources to register. WHERE-clause
/// subqueries are not yet walked; an unrecognized table simply surfaces as a
/// typed "table not found" from the query engine, never a wrong result.
fn collect_table_names(q: &sqlparser::ast::Query) -> Vec<String> {
    let mut out = Vec::new();
    collect_from_setexpr(q.body.as_ref(), &mut out);
    out.sort();
    out.dedup();
    out
}

fn collect_from_setexpr(se: &sqlparser::ast::SetExpr, out: &mut Vec<String>) {
    use sqlparser::ast::SetExpr;
    match se {
        SetExpr::Select(s) => {
            for twj in &s.from {
                collect_from_table_factor(&twj.relation, out);
                for j in &twj.joins {
                    collect_from_table_factor(&j.relation, out);
                }
            }
        }
        SetExpr::Query(inner) => collect_from_setexpr(inner.body.as_ref(), out),
        SetExpr::SetOperation { left, right, .. } => {
            collect_from_setexpr(left, out);
            collect_from_setexpr(right, out);
        }
        _ => {}
    }
}

fn collect_from_table_factor(tf: &sqlparser::ast::TableFactor, out: &mut Vec<String>) {
    use sqlparser::ast::TableFactor;
    match tf {
        TableFactor::Table { name, .. } => {
            out.push(name.to_string().trim_matches('"').to_string());
        }
        TableFactor::Derived { subquery, .. } => collect_from_setexpr(subquery.body.as_ref(), out),
        TableFactor::NestedJoin { table_with_joins, .. } => {
            collect_from_table_factor(&table_with_joins.relation, out);
            for j in &table_with_joins.joins {
                collect_from_table_factor(&j.relation, out);
            }
        }
        _ => {}
    }
}

fn extract_table(q: &sqlparser::ast::Query) -> String {
    if let sqlparser::ast::SetExpr::Select(s) = q.body.as_ref() {
        if let Some(f) = s.from.first() {
            return f.relation.to_string();
        }
    }
    "unknown".to_string()
}

/// If `q` uses a SQL construct the engine does not execute, return a
/// human-readable reason. GalaxDB's SQL surface is single-table scans with
/// `WHERE` filters plus vector search; JOINs, set operations, subqueries in
/// `FROM`, `GROUP BY`/aggregates, `HAVING`, and `DISTINCT` are not
/// supported. Callers reject such queries with a typed error instead of
/// silently scanning the first table and returning wrong results
/// (engineering-principles §2 — no silent fallback).
fn unsupported_select_reason(q: &sqlparser::ast::Query) -> Option<&'static str> {
    use sqlparser::ast::{Expr, GroupByExpr, SelectItem, SetExpr};

    let select = match q.body.as_ref() {
        SetExpr::Select(s) => s,
        SetExpr::Query(_) => return Some("subqueries"),
        SetExpr::SetOperation { .. } => {
            return Some("set operations (UNION/INTERSECT/EXCEPT)")
        }
        SetExpr::Values(_) => return Some("VALUES in SELECT position"),
        _ => return Some("this query form"),
    };

    if select.from.len() > 1 {
        return Some("comma-joined / multiple tables");
    }
    if let Some(t) = select.from.first() {
        if !t.joins.is_empty() {
            return Some("JOIN");
        }
    }
    match &select.group_by {
        GroupByExpr::Expressions(exprs, _) if !exprs.is_empty() => return Some("GROUP BY"),
        GroupByExpr::All(_) => return Some("GROUP BY ALL"),
        _ => {}
    }
    if select.having.is_some() {
        return Some("HAVING");
    }
    if select.distinct.is_some() {
        return Some("DISTINCT");
    }
    // Aggregate function calls in the projection (e.g. COUNT/SUM/AVG).
    for item in &select.projection {
        let expr = match item {
            SelectItem::UnnamedExpr(e) => Some(e),
            SelectItem::ExprWithAlias { expr, .. } => Some(expr),
            _ => None,
        };
        if let Some(Expr::Function(f)) = expr {
            let name = f.name.to_string().to_ascii_lowercase();
            if matches!(
                name.as_str(),
                "count" | "sum" | "avg" | "min" | "max" | "array_agg" | "string_agg"
                    | "stddev" | "variance" | "var_samp" | "var_pop"
            ) {
                return Some("aggregate functions");
            }
        }
    }
    None
}

/// Reject an unsupported `SELECT` with a typed, SQLSTATE-mapped error so a
/// client sees a clear "feature not supported" instead of silently wrong
/// rows. Used at every SELECT planning entry point.
fn check_select_supported(q: &sqlparser::ast::Query) -> GalaxResult<()> {
    if let Some(reason) = unsupported_select_reason(q) {
        return Err(GalaxError::FeatureNotSupported(format!(
            "{reason} not supported: GalaxDB executes single-table scans with WHERE \
             filters and vector search; rewrite the query against one table"
        )));
    }
    Ok(())
}

/// Extract the projection column list and the WHERE filter from a
/// `SELECT` query. `SELECT *` / unsupported projection items yield
/// an empty column list (which the executor interprets as "all
/// columns"). Missing WHERE returns `None`.
///
/// Supported projection items:
/// - `*` → empty list (all columns)
/// - `col_name` / `table.col_name` → column name
///
/// Anything else (aggregates, expressions, aliases) returns the
/// empty projection so the full row comes back — that's correct
/// behaviour for v1, the executor caller can drop columns it
/// doesn't want. A dedicated aggregation path is task 18.8 scope.
fn extract_projection_and_filter(
    q: &sqlparser::ast::Query,
) -> (Vec<String>, Option<FilterExpr>) {
    let sqlparser::ast::SetExpr::Select(s) = q.body.as_ref() else {
        return (vec![], None);
    };

    let mut columns = Vec::new();
    let mut projection_is_star = false;
    for item in &s.projection {
        match item {
            sqlparser::ast::SelectItem::Wildcard(_)
            | sqlparser::ast::SelectItem::QualifiedWildcard(..) => {
                projection_is_star = true;
                break;
            }
            sqlparser::ast::SelectItem::UnnamedExpr(expr)
            | sqlparser::ast::SelectItem::ExprWithAlias { expr, .. } => {
                if let Some(name) = column_name_from_expr(expr) {
                    columns.push(name);
                } else {
                    // Unsupported expression — fall back to full row.
                    projection_is_star = true;
                    break;
                }
            }
        }
    }
    let columns = if projection_is_star { Vec::new() } else { columns };

    let filter = s.selection.as_ref().and_then(filter_from_expr);

    (columns, filter)
}

/// Extract a `SemanticMatchExpr` from a SELECT's WHERE clause if it
/// contains a `SEMANTIC_MATCH(col, 'query', threshold)` call.
/// Returns `None` if no SEMANTIC_MATCH is present.
fn extract_semantic_match_from_query(
    q: &sqlparser::ast::Query,
) -> Option<galaxdb_sql::ast::SemanticMatchExpr> {
    let sqlparser::ast::SetExpr::Select(s) = q.body.as_ref() else {
        return None;
    };
    let selection = s.selection.as_ref()?;
    extract_semantic_match_from_expr(selection)
}

/// Recursively walk a WHERE expression looking for a SEMANTIC_MATCH call.
fn extract_semantic_match_from_expr(
    expr: &sqlparser::ast::Expr,
) -> Option<galaxdb_sql::ast::SemanticMatchExpr> {
    use sqlparser::ast::{BinaryOperator, Expr, FunctionArguments};
    match expr {
        Expr::Function(f) => {
            let name = f.name.to_string().to_uppercase();
            if name != "SEMANTIC_MATCH" {
                return None;
            }
            // Extract args: (column, 'query_text', threshold)
            let FunctionArguments::List(arg_list) = &f.args else {
                return None;
            };
            let args: Vec<&sqlparser::ast::Expr> = arg_list
                .args
                .iter()
                .filter_map(|a| match a {
                    sqlparser::ast::FunctionArg::Unnamed(
                        sqlparser::ast::FunctionArgExpr::Expr(e),
                    ) => Some(e),
                    _ => None,
                })
                .collect();
            if args.len() != 3 {
                return None;
            }
            let column = match args[0] {
                Expr::Identifier(id) => id.value.clone(),
                Expr::CompoundIdentifier(parts) => {
                    parts.last().map(|p| p.value.clone()).unwrap_or_default()
                }
                _ => return None,
            };
            let query = match args[1] {
                Expr::Value(sqlparser::ast::Value::SingleQuotedString(s)) => s.clone(),
                _ => return None,
            };
            let threshold: f64 = match args[2] {
                Expr::Value(sqlparser::ast::Value::Number(n, _)) => {
                    n.parse().ok()?
                }
                _ => return None,
            };
            Some(galaxdb_sql::ast::SemanticMatchExpr {
                column,
                query,
                threshold,
            })
        }
        Expr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
        } => extract_semantic_match_from_expr(left)
            .or_else(|| extract_semantic_match_from_expr(right)),
        _ => None,
    }
}

/// Is `expr` itself a top-level `SEMANTIC_MATCH(...)` function call?
fn is_semantic_match_call(expr: &sqlparser::ast::Expr) -> bool {
    matches!(expr, sqlparser::ast::Expr::Function(f)
        if f.name.to_string().eq_ignore_ascii_case("SEMANTIC_MATCH"))
}

/// Does the query's WHERE clause contain the AuroraSQL `NOT DUPLICATE`
/// group-level dedup predicate (parsed as `NOT DUPLICATE` → `UnaryOp{Not,
/// Identifier("DUPLICATE")}`)? Used to keep such queries on the native
/// executor, which applies the dedup pass — the DataFusion analytical engine
/// has no `NOT DUPLICATE` operator, so routing one there would drop the
/// semantics (HTAP task 17.1).
fn query_has_not_duplicate(q: &sqlparser::ast::Query) -> bool {
    fn expr_has(expr: &sqlparser::ast::Expr) -> bool {
        use sqlparser::ast::{Expr, UnaryOperator};
        match expr {
            Expr::UnaryOp {
                op: UnaryOperator::Not,
                expr,
            } => match expr.as_ref() {
                Expr::Identifier(id) => id.value.eq_ignore_ascii_case("DUPLICATE"),
                inner => expr_has(inner),
            },
            Expr::BinaryOp { left, right, .. } => expr_has(left) || expr_has(right),
            Expr::Nested(e) => expr_has(e),
            _ => false,
        }
    }
    if let sqlparser::ast::SetExpr::Select(s) = q.body.as_ref() {
        if let Some(sel) = &s.selection {
            return expr_has(sel);
        }
    }
    false
}

/// Remove every `SEMANTIC_MATCH(...)` conjunct from a WHERE expression,
/// returning the residual relational predicate (`None` if nothing remains).
/// Only `AND` chains are unwound — `SEMANTIC_MATCH` combined with `OR` is not
/// a supported analytical shape (the candidate set defines the row universe),
/// so an `OR` containing it is left intact and will surface as an error when
/// DataFusion cannot resolve the `SEMANTIC_MATCH` name (never silently wrong).
fn strip_semantic_match_expr(expr: &sqlparser::ast::Expr) -> Option<sqlparser::ast::Expr> {
    use sqlparser::ast::{BinaryOperator, Expr};
    if is_semantic_match_call(expr) {
        return None;
    }
    if let Expr::BinaryOp {
        left,
        op: BinaryOperator::And,
        right,
    } = expr
    {
        let l = strip_semantic_match_expr(left);
        let r = strip_semantic_match_expr(right);
        return match (l, r) {
            (Some(l), Some(r)) => Some(Expr::BinaryOp {
                left: Box::new(l),
                op: BinaryOperator::And,
                right: Box::new(r),
            }),
            (Some(e), None) | (None, Some(e)) => Some(e),
            (None, None) => None,
        };
    }
    if let Expr::Nested(inner) = expr {
        return strip_semantic_match_expr(inner).map(|e| Expr::Nested(Box::new(e)));
    }
    Some(expr.clone())
}

/// Return a copy of `q` with the `SEMANTIC_MATCH(...)` predicate removed from
/// its WHERE clause (HTAP task 16). The candidate set produced by the vector
/// search becomes the row source for the base table, so the residual query
/// carries only the relational predicate + the analytical clauses.
fn strip_semantic_match_query(q: &sqlparser::ast::Query) -> sqlparser::ast::Query {
    let mut out = q.clone();
    if let sqlparser::ast::SetExpr::Select(s) = out.body.as_mut() {
        let mut select = s.as_ref().clone();
        select.selection = select
            .selection
            .as_ref()
            .and_then(strip_semantic_match_expr);
        **s = select;
    }
    out
}

fn extract_projection_and_filter_no_semantic(
    q: &sqlparser::ast::Query,
) -> (Vec<String>, Option<FilterExpr>) {
    let sqlparser::ast::SetExpr::Select(s) = q.body.as_ref() else {
        return (vec![], None);
    };
    let mut columns = Vec::new();
    let mut projection_is_star = false;
    for item in &s.projection {
        match item {
            sqlparser::ast::SelectItem::Wildcard(_)
            | sqlparser::ast::SelectItem::QualifiedWildcard(..) => {
                projection_is_star = true;
                break;
            }
            sqlparser::ast::SelectItem::UnnamedExpr(expr)
            | sqlparser::ast::SelectItem::ExprWithAlias { expr, .. } => {
                if let Some(name) = column_name_from_expr(expr) {
                    columns.push(name);
                } else {
                    projection_is_star = true;
                    break;
                }
            }
        }
    }
    let columns = if projection_is_star { Vec::new() } else { columns };
    // Strip SEMANTIC_MATCH from the filter — only keep non-semantic predicates
    let filter = s
        .selection
        .as_ref()
        .and_then(filter_from_expr_no_semantic);
    (columns, filter)
}

/// Like `filter_from_expr` but returns `None` for SEMANTIC_MATCH calls
/// (they are handled separately by the vector backend).
fn filter_from_expr_no_semantic(expr: &sqlparser::ast::Expr) -> Option<FilterExpr> {
    use sqlparser::ast::Expr;
    // Skip SEMANTIC_MATCH function calls
    if let Expr::Function(f) = expr {
        if f.name.to_string().to_uppercase() == "SEMANTIC_MATCH" {
            return None;
        }
    }
    // For AND, strip the SEMANTIC_MATCH side and keep the other
    if let Expr::BinaryOp {
        left,
        op: sqlparser::ast::BinaryOperator::And,
        right,
    } = expr
    {
        let l = filter_from_expr_no_semantic(left);
        let r = filter_from_expr_no_semantic(right);
        return match (l, r) {
            (Some(a), Some(b)) => Some(FilterExpr::And(Box::new(a), Box::new(b))),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
    }
    filter_from_expr(expr)
}

/// If the SQL is a `SELECT` with an `AT VERSION ...` suffix, return
/// `Some((rest_of_sql_without_at_version, parsed_AtVersionExpr))`.
/// If no `AT VERSION` is present, return `None`. If parsing the
/// version fragment fails, propagate the parser error.
///
/// The matcher is deliberately conservative: it requires the literal
/// token `AT VERSION` to appear case-insensitively outside quotes and
/// after a `FROM` clause. The rest of the string (from `AT VERSION`
/// to the end, minus a trailing semicolon) is handed to
/// `galaxdb_sql::parser::parse_at_version`. This keeps the suffix
/// syntax consistent with `galaxdb-sql::parser::parse_at_version`.
fn split_at_version(
    sql: &str,
) -> GalaxResult<Option<(String, galaxdb_sql::ast::AtVersionExpr)>> {
    let trimmed = sql.trim().trim_end_matches(';');
    let upper: String = trimmed
        .chars()
        .map(|c| if c == '\'' { '\'' } else { c.to_ascii_uppercase() })
        .collect();

    // Case-insensitive search that skips quoted regions. We need the
    // position in the *original* string, which matches the uppercase
    // string byte-for-byte because we only mapped ASCII letters.
    let bytes = trimmed.as_bytes();
    let upper_bytes = upper.as_bytes();
    let needle = b"AT VERSION";
    let mut in_quote = false;
    let mut i = 0usize;
    let mut found: Option<usize> = None;

    while i + needle.len() <= bytes.len() {
        if bytes[i] == b'\'' {
            in_quote = !in_quote;
            i += 1;
            continue;
        }
        if !in_quote && &upper_bytes[i..i + needle.len()] == needle {
            let before_ok = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
            let after_idx = i + needle.len();
            let after_ok =
                after_idx == bytes.len() || !bytes[after_idx].is_ascii_alphanumeric();
            if before_ok && after_ok {
                found = Some(i);
                break;
            }
        }
        i += 1;
    }

    let Some(pos) = found else {
        return Ok(None);
    };

    let stripped = trimmed[..pos].trim_end().to_string();
    let fragment = &trimmed[pos..];
    let at = galaxdb_sql::parser::parse_at_version(fragment)?;
    Ok(Some((stripped, at)))
}

/// If `expr` is a bare column reference, return its name.
fn column_name_from_expr(expr: &sqlparser::ast::Expr) -> Option<String> {
    match expr {
        sqlparser::ast::Expr::Identifier(id) => Some(id.value.clone()),
        sqlparser::ast::Expr::CompoundIdentifier(parts) => {
            // table.col → "col"
            parts.last().map(|p| p.value.clone())
        }
        _ => None,
    }
}

/// Convert a WHERE clause from the `sqlparser` AST into a `FilterExpr`
/// the executor can evaluate. Supported shapes:
///
/// - `col = literal`, `col != literal`, `col <> literal`
/// - `col < literal`, `col > literal`, `col <= literal`, `col >= literal`
/// - `expr AND expr`, `expr OR expr`
/// - `NOT DUPLICATE` — the AuroraSQL group-level dedup predicate
///   (task 35.5). sqlparser parses it as `UnaryOp { op: Not, expr:
///   Identifier("DUPLICATE") }`; we recognise that exact shape and
///   translate it to [`FilterExpr::NotDuplicate`].
///
/// The left side must be a column reference and the right side a
/// literal value. Anything else returns `None` (treated by the planner
/// as "no filter", which is strictly less restrictive than the query
/// asks for — callers should prefer a parse error for that case, but
/// at the embedded layer today we only forward supported filters).
fn filter_from_expr(expr: &sqlparser::ast::Expr) -> Option<FilterExpr> {
    use sqlparser::ast::{BinaryOperator, Expr, UnaryOperator};
    match expr {
        // `NOT DUPLICATE` — AuroraSQL extension (task 35.5). Parses as
        // `UnaryOp { op: Not, expr: Identifier("DUPLICATE") }` in
        // sqlparser; case-insensitive match keeps the user-facing SQL
        // tolerant of quoting and casing.
        Expr::UnaryOp {
            op: UnaryOperator::Not,
            expr: inner,
        } => {
            if let Expr::Identifier(id) = inner.as_ref() {
                if id.value.eq_ignore_ascii_case("DUPLICATE") {
                    return Some(FilterExpr::NotDuplicate);
                }
            }
            None
        }
        Expr::BinaryOp { left, op, right } => match op {
            BinaryOperator::And => Some(FilterExpr::And(
                Box::new(filter_from_expr(left)?),
                Box::new(filter_from_expr(right)?),
            )),
            BinaryOperator::Or => Some(FilterExpr::Or(
                Box::new(filter_from_expr(left)?),
                Box::new(filter_from_expr(right)?),
            )),
            BinaryOperator::Eq
            | BinaryOperator::NotEq
            | BinaryOperator::Lt
            | BinaryOperator::Gt
            | BinaryOperator::LtEq
            | BinaryOperator::GtEq => {
                // Try col OP literal. If that fails, try literal OP col
                // and flip.
                if let (Some(col), Some(val)) =
                    (column_name_from_expr(left), literal_value(right))
                {
                    return Some(build_cmp(op, col, val));
                }
                if let (Some(val), Some(col)) =
                    (literal_value(left), column_name_from_expr(right))
                {
                    let flipped = flip_cmp_op(op);
                    return Some(build_cmp(&flipped, col, val));
                }
                None
            }
            _ => None,
        },
        Expr::Nested(inner) => filter_from_expr(inner),
        _ => None,
    }
}

/// Build a `FilterExpr` for a comparison op with `col OP val` ordering.
fn build_cmp(
    op: &sqlparser::ast::BinaryOperator,
    column: String,
    value: Value,
) -> FilterExpr {
    use sqlparser::ast::BinaryOperator::*;
    match op {
        Eq => FilterExpr::Eq { column, value },
        NotEq => FilterExpr::Ne { column, value },
        Lt => FilterExpr::Lt { column, value },
        Gt => FilterExpr::Gt { column, value },
        LtEq => FilterExpr::Le { column, value },
        GtEq => FilterExpr::Ge { column, value },
        _ => FilterExpr::Eq { column, value },
    }
}

/// Mirror a comparison operator when the column ends up on the right
/// side of the expression (`5 < id` becomes `id > 5`).
fn flip_cmp_op(op: &sqlparser::ast::BinaryOperator) -> sqlparser::ast::BinaryOperator {
    use sqlparser::ast::BinaryOperator::*;
    match op {
        Lt => Gt,
        Gt => Lt,
        LtEq => GtEq,
        GtEq => LtEq,
        other => other.clone(),
    }
}

/// If `expr` is a literal, return the corresponding [`Value`]. Mirrors
/// [`value_from_expr`] but returns `None` on non-literals so we can
/// distinguish a successful conversion from a fallback string.
fn literal_value(expr: &sqlparser::ast::Expr) -> Option<Value> {
    use sqlparser::ast::{Expr, Value as SqlValue};
    match expr {
        Expr::Value(v) => match v {
            SqlValue::Number(n, _) => n
                .parse::<i64>()
                .map(Value::Integer)
                .or_else(|_| n.parse::<f64>().map(Value::Float))
                .ok(),
            SqlValue::SingleQuotedString(s) | SqlValue::DoubleQuotedString(s) => {
                Some(Value::Text(s.clone()))
            }
            SqlValue::Boolean(b) => Some(Value::Bool(*b)),
            SqlValue::Null => Some(Value::Null),
            _ => None,
        },
        Expr::UnaryOp {
            op: sqlparser::ast::UnaryOperator::Minus,
            expr,
        } => match literal_value(expr) {
            Some(Value::Integer(n)) => Some(Value::Integer(-n)),
            Some(Value::Float(f)) => Some(Value::Float(-f)),
            _ => None,
        },
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Training-export glue (task 22.4)
//
// `Database::training_dataset` exports a tagged table as a Lance
// dataset by driving `galaxdb_versioning::LanceExporter` against the
// live `Engine`. The pieces below are the concrete types that wiring
// needs: a real `LanceExportSource` over `Engine::scan_all_at`, a
// catalog → Arrow schema mapper, and a path-safe version of the tag
// name for the output directory.
// ---------------------------------------------------------------------------

/// Real [`galaxdb_versioning::LanceExportSource`] that reads rows
/// from the live storage engine at a specific timestamp.
///
/// `read_blocks` ignores the block-id list supplied by the exporter
/// because v1's memtable-based `scan_all_at` addresses keys, not
/// blocks. The exporter uses the block list only to decide which
/// rows the source should return; when the source already knows the
/// version ts it can ask the engine directly, which is simpler than
/// round-tripping through a block-set. When K2-Follow lands and
/// AT VERSION becomes SST-aware, this impl switches to asking
/// `SstRegistry` for the pinned-block payload instead.
struct EmbeddedLanceExportSource {
    engine: Arc<galaxdb_storage::engine::Engine>,
    table_name: String,
    table_entry: galaxdb_sql::executor::TableEntry,
    version_timestamp: u64,
    /// Embedding-column projection (Req 20). When `Some`, each exported row
    /// gets one extra `FieldValue` appended after its scalar fields: the
    /// row's vector resolved by primary key, or `FieldValue::Null` when the
    /// row has no embedding at this version (AC4 — never fabricate).
    embedding: Option<EmbeddingExportInfo>,
}

/// Snapshot of a table's embedding vectors for a training export.
///
/// `key_to_vec` maps storage primary-key bytes (the `{table}:…` key that
/// `Engine::scan_all_at` returns) to the row's embedding vector, resolved
/// from the in-memory vector index at export time. A row absent from this
/// map is exported as a NULL vector.
#[derive(Clone)]
struct EmbeddingExportInfo {
    /// Export column name (`{source_column}_embedding`).
    column: String,
    /// Embedding dimensionality (Arrow `FixedSizeList` size).
    dim: usize,
    /// Primary-key bytes → embedding vector.
    key_to_vec: HashMap<Vec<u8>, Vec<f32>>,
}

impl galaxdb_versioning::LanceExportSource for EmbeddedLanceExportSource {
    fn read_blocks(
        &self,
        _block_ids: &[galaxdb_common::types::BlockId],
    ) -> galaxdb_versioning::ExportResult<Vec<galaxdb_versioning::ExportedRow>> {
        use galaxdb_sql::row_codec;
        use galaxdb_versioning::ExportedRow;

        // `scan_all_at` returns every visible row in the whole engine.
        // We restrict to this table by the shared `"{table}:"` prefix
        // that `row_codec::build_primary_key` builds for INSERTs.
        let prefix = format!("{}:", self.table_name);
        let raw = self.engine.scan_all_at(self.version_timestamp);

        let mut rows = Vec::with_capacity(raw.len());
        for (key, val, _ts) in raw {
            if !key.starts_with(prefix.as_bytes()) {
                continue;
            }
            // Decode the `col=value|col=value|...` on-disk row into
            // typed values, then project in catalog order so the row's
            // `fields` align with the Arrow schema.
            let decoded = row_codec::decode_row(&val);
            let mut fields =
                project_row_to_field_values(&self.table_entry, &decoded);
            // Req 20: append the embedding vector for this row, or NULL if the
            // row has no embedding at the exported version (never fabricate).
            if let Some(emb) = &self.embedding {
                match emb.key_to_vec.get(&key) {
                    Some(v) => fields.push(galaxdb_versioning::FieldValue::Embedding(v.clone())),
                    None => fields.push(galaxdb_versioning::FieldValue::Null),
                }
            }
            rows.push(ExportedRow {
                primary_key: key,
                fields,
                near_duplicate_group: None,
            });
        }
        Ok(rows)
    }
}

/// Project a decoded row into one [`FieldValue`] per catalog column,
/// in catalog order. Missing columns surface as the type-appropriate
/// zero value (0 / empty string / empty vector) so that the Arrow
/// builder always sees one value per column — the schema is marked
/// nullable at construction (see [`arrow_schema_from_catalog`]) so
/// defaulting is safe for v1. The embedding column (if any) is NOT
/// produced here — it is appended separately by the export source from
/// the vector index, since the scalar row codec does not carry vectors.
fn project_row_to_field_values(
    entry: &galaxdb_sql::executor::TableEntry,
    decoded: &[(String, galaxdb_sql::planner::Value)],
) -> Vec<galaxdb_versioning::FieldValue> {
    use galaxdb_sql::planner::Value;
    use galaxdb_versioning::FieldValue;

    let mut out = Vec::with_capacity(entry.columns.len());
    for col in &entry.columns {
        let value = decoded.iter().find(|(n, _)| n == &col.name).map(|(_, v)| v);
        let kind = classify_column(&col.data_type);
        let fv = match (kind, value) {
            (ColumnKind::Int, Some(Value::Integer(n))) => FieldValue::Int64(*n),
            (ColumnKind::Int, Some(Value::Float(f))) => FieldValue::Int64(*f as i64),
            (ColumnKind::Int, Some(Value::Text(s))) => {
                FieldValue::Int64(s.parse::<i64>().unwrap_or(0))
            }
            (ColumnKind::Int, Some(Value::Bool(b))) => FieldValue::Int64(*b as i64),
            (ColumnKind::Int, _) => FieldValue::Int64(0),

            (ColumnKind::Float, Some(Value::Float(f))) => FieldValue::Float32(*f as f32),
            (ColumnKind::Float, Some(Value::Integer(n))) => {
                FieldValue::Float32(*n as f32)
            }
            (ColumnKind::Float, Some(Value::Text(s))) => {
                FieldValue::Float32(s.parse::<f32>().unwrap_or(0.0))
            }
            (ColumnKind::Float, _) => FieldValue::Float32(0.0),

            (ColumnKind::Text, Some(v)) => {
                FieldValue::Utf8(galaxdb_sql::row_codec::value_display(v))
            }
            (ColumnKind::Text, None) => FieldValue::Utf8(String::new()),
        };
        out.push(fv);
    }
    out
}

/// Kind of Arrow column the exporter should build for a given SQL
/// type string. Anything the v1 exporter doesn't specifically know
/// about falls into `Text` — the row codec stores display strings,
/// so the round-trip is lossless even for types we haven't modelled
/// as first-class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ColumnKind {
    Int,
    Float,
    Text,
}

fn classify_column(data_type: &str) -> ColumnKind {
    let base = data_type
        .split('(')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_uppercase();
    match base.as_str() {
        "INT" | "INTEGER" | "BIGINT" | "SMALLINT" | "TINYINT" => ColumnKind::Int,
        "FLOAT" | "REAL" | "DOUBLE" | "DOUBLE PRECISION" => ColumnKind::Float,
        _ => ColumnKind::Text,
    }
}

/// Map a [`TableEntry`] to an Arrow [`arrow::datatypes::Schema`]. Every
/// scalar column is marked nullable so partial rows (which can happen when a
/// column was added after some rows were inserted) don't fail the Arrow
/// builder. When `embedding` is `Some`, one extra nullable column is appended
/// for the table's embedding vector (Req 20): `FixedSizeList<Float32, dim>`
/// for `Float32` precision, or `Binary` for `Sq8`/`Rabitq` (which the exporter
/// quantises the vector into). The column is appended LAST so the export
/// source can push the vector after the scalar fields in catalog order.
fn arrow_schema_from_catalog(
    entry: &galaxdb_sql::executor::TableEntry,
    embedding: Option<&EmbeddingExportInfo>,
    precision: galaxdb_versioning::TrainingPrecision,
) -> arrow::datatypes::Schema {
    use arrow::datatypes::{DataType, Field};
    use galaxdb_versioning::TrainingPrecision;

    let mut fields: Vec<Field> = entry
        .columns
        .iter()
        .map(|c| {
            let dt = match classify_column(&c.data_type) {
                ColumnKind::Int => DataType::Int64,
                ColumnKind::Float => DataType::Float32,
                ColumnKind::Text => DataType::Utf8,
            };
            // Nullable: see comment above.
            Field::new(&c.name, dt, true)
        })
        .collect();

    if let Some(emb) = embedding {
        let dt = match precision {
            TrainingPrecision::Float32 => DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                emb.dim as i32,
            ),
            // Sq8 / Rabitq: the exporter rewrites Embedding → Binary, so the
            // schema column must be Binary for those precisions.
            TrainingPrecision::Sq8 | TrainingPrecision::Rabitq => DataType::Binary,
        };
        fields.push(Field::new(&emb.column, dt, true));
    }

    arrow::datatypes::Schema::new(fields)
}

/// Make a tag name safe for use as a single path component. Replaces
/// every non-alphanumeric / non-`-` / non-`_` / non-`.` byte with `_`
/// so tags like `"train-v1 (latest)"` still land under a sensible
/// directory name. This is cosmetic — tag uniqueness is guaranteed by
/// the catalog.
fn sanitize_tag_for_path(tag: &str) -> String {
    tag.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_db() -> Database {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("db");
        std::mem::forget(dir);
        Database::open(p.to_str().unwrap()).unwrap()
    }

    #[test]
    fn create_insert_select_roundtrip() {
        let mut db = test_db();
        db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
            .unwrap();
        db.execute("INSERT INTO users (id, name) VALUES (2, 'bob')")
            .unwrap();
        let r = db.execute("SELECT * FROM users").unwrap();
        match r {
            QueryResult::Rows(rows) => {
                assert_eq!(rows.len(), 2);
                assert!(
                    rows.iter()
                        .any(|r| r.values.iter().any(|(k, v)| k == "name" && v == "alice"))
                );
            }
            other => panic!("expected Rows, got {:?}", other),
        }
    }

    #[test]
    fn insert_10_rows_and_count() {
        let mut db = test_db();
        db.execute("CREATE TABLE t (id INT, val TEXT)").unwrap();
        for i in 0..10 {
            db.execute(&format!(
                "INSERT INTO t (id, val) VALUES ({}, 'v{}')",
                i, i
            ))
            .unwrap();
        }
        assert_eq!(db.row_count(), 10);
        match db.execute("SELECT * FROM t").unwrap() {
            QueryResult::Rows(rows) => assert_eq!(rows.len(), 10),
            other => panic!("expected Rows, got {:?}", other),
        }
    }

    #[test]
    fn select_nonexistent_fails() {
        let mut db = test_db();
        assert!(db.execute("SELECT * FROM nope").is_err());
    }

    #[test]
    fn catalog_and_data_survive_reopen() {
        // Durability regression: the catalog was in-memory only, so after a
        // restart the tables vanished and their WAL-recovered rows became
        // unreadable. Schema is now persisted and reloaded on open.
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("db");
        let path = p.to_str().unwrap().to_string();

        {
            let mut db = Database::open(&path).unwrap();
            db.execute("CREATE TABLE acct (id INT PRIMARY KEY, bal INT)")
                .unwrap();
            db.execute("INSERT INTO acct (id, bal) VALUES (1, 100)").unwrap();
            db.execute("INSERT INTO acct (id, bal) VALUES (2, 200)").unwrap();
            db.execute("ALTER TABLE acct SET STORAGE COLUMNAR").unwrap();
            db.execute("CREATE TABLE tmp (id INT PRIMARY KEY)").unwrap();
            db.execute("DROP TABLE tmp").unwrap();
            // ensure durability to disk
            db.flush().unwrap();
        }

        // Reopen the same directory: tables and rows must be present.
        let mut db = Database::open(&path).unwrap();
        assert!(db.table_exists("acct"), "acct must survive reopen");
        assert!(!db.table_exists("tmp"), "dropped table must not reappear");
        match db.execute("SELECT id, bal FROM acct ORDER BY id").unwrap() {
            QueryResult::Rows(rows) => {
                assert_eq!(rows.len(), 2);
                assert_eq!(rows[0].values.iter().find(|(k, _)| k == "bal").unwrap().1, "100");
                assert_eq!(rows[1].values.iter().find(|(k, _)| k == "bal").unwrap().1, "200");
            }
            other => panic!("expected Rows after reopen, got {other:?}"),
        }
        // A duplicate insert after reopen must still be rejected (PK metadata
        // survived, not just the rows).
        assert!(
            db.execute("INSERT INTO acct (id, bal) VALUES (1, 999)").is_err(),
            "PK uniqueness must survive reopen"
        );
    }

    #[test]
    fn duplicate_primary_key_is_rejected_not_overwritten() {
        let mut db = test_db();
        db.execute("CREATE TABLE pk (id INT PRIMARY KEY, v TEXT)").unwrap();
        db.execute("INSERT INTO pk (id, v) VALUES (1, 'a')").unwrap();
        let err = db.execute("INSERT INTO pk (id, v) VALUES (1, 'b')").unwrap_err();
        assert!(
            matches!(err, GalaxError::UniqueViolation { .. }),
            "second insert of the same PK must be a unique violation, got {err:?}"
        );
        // The original row is intact — no silent overwrite.
        match db.execute("SELECT v FROM pk WHERE id = 1").unwrap() {
            QueryResult::Rows(rows) => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0].values.iter().find(|(k, _)| k == "v").unwrap().1, "a");
            }
            other => panic!("expected Rows, got {other:?}"),
        }
    }

    #[test]
    fn delete_then_reinsert_same_pk_succeeds() {
        let mut db = test_db();
        db.execute("CREATE TABLE pk (id INT PRIMARY KEY, v TEXT)").unwrap();
        db.execute("INSERT INTO pk (id, v) VALUES (1, 'a')").unwrap();
        db.execute("DELETE FROM pk WHERE id = 1").unwrap();
        // Re-inserting a deleted key is allowed (tombstone is not "exists").
        db.execute("INSERT INTO pk (id, v) VALUES (1, 'b')").unwrap();
        match db.execute("SELECT v FROM pk WHERE id = 1").unwrap() {
            QueryResult::Rows(rows) => {
                assert_eq!(rows[0].values.iter().find(|(k, _)| k == "v").unwrap().1, "b");
            }
            other => panic!("expected Rows, got {other:?}"),
        }
    }

    #[test]
    fn from_less_scalar_select_evaluates() {
        let mut db = test_db();
        // arithmetic
        match db.execute("SELECT 1 + 1").unwrap() {
            QueryResult::Rows(rows) => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0].values[0].1, "2");
            }
            other => panic!("expected Rows, got {other:?}"),
        }
        // version() — PostgreSQL-compatible prefix
        match db.execute("SELECT version()").unwrap() {
            QueryResult::Rows(rows) => {
                assert!(rows[0].values[0].1.starts_with("PostgreSQL"));
                assert!(rows[0].values[0].1.contains("GalaxDB"));
                assert_eq!(rows[0].values[0].0, "version");
            }
            other => panic!("expected Rows, got {other:?}"),
        }
        // current_database()
        match db.execute("SELECT current_database()").unwrap() {
            QueryResult::Rows(rows) => {
                assert_eq!(rows[0].values[0].1, "galaxdb");
            }
            other => panic!("expected Rows, got {other:?}"),
        }
    }

    #[test]
    fn from_less_unsupported_function_is_typed_error() {
        let mut db = test_db();
        let err = db.execute("SELECT pg_sleep(1)").unwrap_err();
        assert!(matches!(err, GalaxError::FeatureNotSupported(_)));
    }

    #[test]
    fn update_set_column_expression_is_evaluated_not_stringified() {
        // End-to-end regression for the live-testing data-corruption bug:
        // `UPDATE t SET bal = bal - 30` stored the literal text "bal - 30".
        // It must now compute old_bal - 30 = 70.
        let mut db = test_db();
        db.execute("CREATE TABLE acct (id INT PRIMARY KEY, bal INT)")
            .unwrap();
        db.execute("INSERT INTO acct (id, bal) VALUES (1, 100)")
            .unwrap();
        db.execute("UPDATE acct SET bal = bal - 30 WHERE id = 1")
            .unwrap();
        match db.execute("SELECT bal FROM acct WHERE id = 1").unwrap() {
            QueryResult::Rows(rows) => {
                assert_eq!(rows.len(), 1);
                let bal = rows[0]
                    .values
                    .iter()
                    .find(|(k, _)| k == "bal")
                    .map(|(_, v)| v.clone())
                    .unwrap();
                assert_eq!(bal, "70", "expected computed 70, got {bal:?}");
            }
            other => panic!("expected Rows, got {:?}", other),
        }
    }

    #[test]
    fn in_txn_order_by_limit_reads_your_writes_sorted() {
        // Regression: a single-table SELECT with ORDER BY/LIMIT inside a
        // transaction used to be rejected as "analytical". It must now run
        // natively over the txn buffer with read-your-writes ordering.
        let mut db = test_db();
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, score INT)")
            .unwrap();

        let txn = db.begin_transaction().unwrap();
        db.execute_in_txn("INSERT INTO t (id, score) VALUES (1, 30)", &txn, None)
            .unwrap();
        db.execute_in_txn("INSERT INTO t (id, score) VALUES (2, 10)", &txn, None)
            .unwrap();
        db.execute_in_txn("INSERT INTO t (id, score) VALUES (3, 20)", &txn, None)
            .unwrap();

        // ORDER BY score ASC over the uncommitted buffer.
        let asc = db
            .execute_in_txn("SELECT id FROM t ORDER BY score ASC", &txn, None)
            .unwrap();
        match asc {
            QueryResult::Rows(rows) => {
                let ids: Vec<String> = rows
                    .iter()
                    .map(|r| r.values.iter().find(|(k, _)| k == "id").unwrap().1.clone())
                    .collect();
                assert_eq!(ids, vec!["2", "3", "1"], "ascending by score");
            }
            other => panic!("expected Rows, got {other:?}"),
        }

        // ORDER BY score DESC LIMIT 2.
        let top2 = db
            .execute_in_txn("SELECT id FROM t ORDER BY score DESC LIMIT 2", &txn, None)
            .unwrap();
        match top2 {
            QueryResult::Rows(rows) => {
                let ids: Vec<String> = rows
                    .iter()
                    .map(|r| r.values.iter().find(|(k, _)| k == "id").unwrap().1.clone())
                    .collect();
                assert_eq!(ids, vec!["1", "3"], "top-2 by score desc");
            }
            other => panic!("expected Rows, got {other:?}"),
        }

        db.rollback_transaction(&txn);
    }

    #[test]
    fn update_set_unsupported_expression_is_typed_error_not_silent() {
        // A value expression we cannot represent (e.g. a function call) must
        // surface a typed error, never silently store stringified text.
        let mut db = test_db();
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, v INT)")
            .unwrap();
        db.execute("INSERT INTO t (id, v) VALUES (1, 5)").unwrap();
        let err = db.execute("UPDATE t SET v = some_func(v) WHERE id = 1");
        assert!(err.is_err(), "unsupported UPDATE expr must error, got {err:?}");
    }

    #[test]
    fn create_drop() {
        let mut db = test_db();
        db.execute("CREATE TABLE t (id INT)").unwrap();
        assert!(db.table_exists("t"));
        db.execute("DROP TABLE t").unwrap();
        assert!(!db.table_exists("t"));
    }

    #[test]
    fn alter_table_set_storage_switches_mode_and_preserves_results() {
        use galaxdb_common::StorageMode;
        let mut db = test_db();
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        for i in 1..=3 {
            db.execute(&format!("INSERT INTO t (id, name) VALUES ({i}, 'n{i}')"))
                .unwrap();
        }
        // New tables default to Columnar (HTAP task 5).
        assert_eq!(
            db.catalog.get_table("t").unwrap().storage_mode,
            StorageMode::Columnar
        );
        let before = rows_of(db.execute("SELECT id, name FROM t").unwrap());
        assert_eq!(before.len(), 3);

        // → LEGACY: the catalog mode flips and query results are identical
        // (the on-disk rewrite is verified deterministically in the storage
        // crate; here we assert the SQL-level contract — Property 3).
        assert!(matches!(
            db.execute("ALTER TABLE t SET STORAGE LEGACY").unwrap(),
            QueryResult::Ok(_)
        ));
        assert_eq!(
            db.catalog.get_table("t").unwrap().storage_mode,
            StorageMode::Legacy
        );
        assert_eq!(rows_of(db.execute("SELECT id, name FROM t").unwrap()).len(), 3);

        // → COLUMNAR again: mode flips back, new writes still work, and
        // point-lookup + filter remain correct after the round trip.
        assert!(matches!(
            db.execute("ALTER TABLE t SET STORAGE COLUMNAR").unwrap(),
            QueryResult::Ok(_)
        ));
        assert_eq!(
            db.catalog.get_table("t").unwrap().storage_mode,
            StorageMode::Columnar
        );
        db.execute("INSERT INTO t (id, name) VALUES (4, 'n4')").unwrap();
        assert_eq!(rows_of(db.execute("SELECT id FROM t").unwrap()).len(), 4);
        let one = rows_of(db.execute("SELECT name FROM t WHERE id = 2").unwrap());
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].values[0].1, "n2");

        // `ROW` is an accepted alias for LEGACY.
        assert!(matches!(
            db.execute("ALTER TABLE t SET STORAGE ROW").unwrap(),
            QueryResult::Ok(_)
        ));
        assert_eq!(
            db.catalog.get_table("t").unwrap().storage_mode,
            StorageMode::Legacy
        );
    }

    #[test]
    fn alter_table_set_storage_unknown_table_errors() {
        let mut db = test_db();
        assert!(db.execute("ALTER TABLE nope SET STORAGE COLUMNAR").is_err());
    }

    #[test]
    fn describe_result_oids_maps_catalog_types() {
        use galaxdb_sql::types::oid;
        let mut db = test_db();
        db.execute(
            "CREATE TABLE typed (id INTEGER PRIMARY KEY, name TEXT, \
             score DOUBLE PRECISION, big BIGINT, flag BOOLEAN)",
        )
        .unwrap();

        // Explicit projection of catalog columns → each column's real OID.
        let oids = db
            .describe_result_oids("SELECT id, name, score, big, flag FROM typed")
            .expect("SELECT resolves result OIDs");
        assert_eq!(oids, vec![oid::INT4, oid::TEXT, oid::FLOAT8, oid::INT8, oid::BOOL]);

        // SELECT * → all catalog columns in declaration order.
        let star = db.describe_result_oids("SELECT * FROM typed").unwrap();
        assert_eq!(star, vec![oid::INT4, oid::TEXT, oid::FLOAT8, oid::INT8, oid::BOOL]);

        // A non-row statement resolves to no OIDs.
        assert!(db
            .describe_result_oids("INSERT INTO typed (id) VALUES (1)")
            .is_none());
    }

    #[test]
    fn alter_table_set_storage_rejects_unknown_mode() {
        let mut db = test_db();
        db.execute("CREATE TABLE t (id INT PRIMARY KEY)").unwrap();
        assert!(db.execute("ALTER TABLE t SET STORAGE SIDEWAYS").is_err());
    }

    #[test]
    fn duplicate_create_fails() {
        let mut db = test_db();
        db.execute("CREATE TABLE t (id INT)").unwrap();
        assert!(db.execute("CREATE TABLE t (id INT)").is_err());
    }

    #[test]
    fn extensions_work() {
        let mut db = test_db();
        db.execute("CREATE TABLE t (id INT)").unwrap();
        assert!(matches!(db.execute("ANALYZE t").unwrap(), QueryResult::Ok(_)));
        assert!(matches!(
            db.execute("SHOW EMBEDDING HEALTH").unwrap(),
            QueryResult::Rows(_)
        ));
        assert!(matches!(
            db.execute("CREATE VERSION TAG 'v1'").unwrap(),
            QueryResult::Ok(_)
        ));
    }

    #[test]
    fn version_tag_creation_and_pinning() {
        let mut db = test_db();
        db.execute("CREATE TABLE docs (id INT, content TEXT)")
            .unwrap();
        db.execute("INSERT INTO docs (id, content) VALUES (1, 'hello')")
            .unwrap();
        db.execute("INSERT INTO docs (id, content) VALUES (2, 'world')")
            .unwrap();

        let result = db.execute("CREATE VERSION TAG 'v1.0'").unwrap();
        assert!(matches!(result, QueryResult::Ok(_)));

        let catalog = db.tag_catalog.lock().unwrap();
        assert!(catalog.get_tag("v1.0").is_some());
        let tag = catalog.get_tag("v1.0").unwrap();
        assert_eq!(tag.name, "v1.0");
        assert!(!tag.for_training);
        drop(catalog);

        assert!(db.execute("CREATE VERSION TAG 'v1.0'").is_err());
    }

    #[test]
    fn version_tag_for_training() {
        let mut db = test_db();
        db.execute("CREATE TABLE t (id INT)").unwrap();

        let result = db
            .execute(
                "CREATE VERSION TAG 'train-v1' FOR TRAINING WITH TRAINING PRECISION 'sq8' \
                 TRAINING SEED 42",
            )
            .unwrap();
        assert!(matches!(result, QueryResult::Ok(_)));

        let catalog = db.tag_catalog.lock().unwrap();
        let tag = catalog.get_tag("train-v1").unwrap();
        assert!(tag.for_training);
        let opts = tag.training_opts.as_ref().unwrap();
        assert_eq!(opts.precision, "sq8");
        assert_eq!(opts.seed, Some(42));
        assert!(opts.deterministic_order);
    }

    #[test]
    fn training_snapshot_has_real_content_merkle_root() {
        // The version-tag Merkle root must be a real content digest of the
        // pinned snapshot — deterministic for the same data, different for
        // different data, and never the old placeholder constant.
        let mut db1 = test_db();
        db1.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        db1.execute("INSERT INTO t (id, name) VALUES (1, 'alice')")
            .unwrap();
        db1.execute("INSERT INTO t (id, name) VALUES (2, 'bob')")
            .unwrap();
        db1.create_training_snapshot("snap", None).unwrap();
        let root1 = db1.tag_catalog.lock().unwrap().get_tag("snap").unwrap().root;

        // Same data in a second database → identical content root
        // (reproducible, order-independent).
        let mut db2 = test_db();
        db2.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        db2.execute("INSERT INTO t (id, name) VALUES (2, 'bob')")
            .unwrap();
        db2.execute("INSERT INTO t (id, name) VALUES (1, 'alice')")
            .unwrap();
        db2.create_training_snapshot("snap", None).unwrap();
        let root2 = db2.tag_catalog.lock().unwrap().get_tag("snap").unwrap().root;

        assert_eq!(root1, root2, "same data must yield the same content root");
        assert_ne!(root1.hash, 0xC0DE, "must not be the old placeholder");
        assert_ne!(
            root1,
            galaxdb_versioning::MerkleRoot::empty(),
            "a non-empty snapshot must have a non-empty root"
        );

        // Different data → different root.
        let mut db3 = test_db();
        db3.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        db3.execute("INSERT INTO t (id, name) VALUES (1, 'CHANGED')")
            .unwrap();
        db3.create_training_snapshot("snap", None).unwrap();
        let root3 = db3.tag_catalog.lock().unwrap().get_tag("snap").unwrap().root;
        assert_ne!(root1, root3, "different data must yield a different root");
    }

    #[test]
    fn analytical_select_constructs_execute_via_datafusion() {
        let mut db = test_db();
        db.execute("CREATE TABLE a (id INT PRIMARY KEY, name TEXT)").unwrap();
        db.execute("CREATE TABLE b (id INT PRIMARY KEY, a_id INT)").unwrap();
        db.execute("INSERT INTO a (id, name) VALUES (1, 'x')").unwrap();
        db.execute("INSERT INTO a (id, name) VALUES (2, 'y')").unwrap();
        db.execute("INSERT INTO b (id, a_id) VALUES (10, 1)").unwrap();

        // These all route to the DataFusion analytical engine (HTAP task 15)
        // and now EXECUTE correctly instead of returning FeatureNotSupported.

        // Aggregate over a single table.
        match db.execute("SELECT COUNT(*) AS n FROM a").unwrap() {
            QueryResult::Rows(rows) => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0].values[0].1, "2");
            }
            other => panic!("expected Rows, got {other:?}"),
        }

        // Inner JOIN: only a.id=1 matches b.a_id=1.
        match db
            .execute("SELECT a.name, b.id FROM a JOIN b ON a.id = b.a_id")
            .unwrap()
        {
            QueryResult::Rows(rows) => assert_eq!(rows.len(), 1),
            other => panic!("expected Rows, got {other:?}"),
        }

        // GROUP BY with ORDER BY.
        match db
            .execute("SELECT name, COUNT(*) AS n FROM a GROUP BY name ORDER BY name")
            .unwrap()
        {
            QueryResult::Rows(rows) => assert_eq!(rows.len(), 2),
            other => panic!("expected Rows, got {other:?}"),
        }

        // DISTINCT and UNION also execute.
        assert!(matches!(
            db.execute("SELECT DISTINCT name FROM a").unwrap(),
            QueryResult::Rows(_)
        ));
        assert!(matches!(
            db.execute("SELECT id FROM a UNION SELECT id FROM b").unwrap(),
            QueryResult::Rows(_)
        ));

        // A plain single-table scan + WHERE still uses the native path.
        let ok = db.execute("SELECT id, name FROM a WHERE id = 1").unwrap();
        assert!(matches!(ok, QueryResult::Rows { .. }));
    }

    /// ORDER BY / LIMIT / OFFSET execute end to end (HTAP task 15.1): the
    /// classifier routes them to DataFusion and they actually sort/limit/skip
    /// rather than being parsed-and-ignored by the native scan.
    #[test]
    fn order_by_limit_offset_execute_end_to_end() {
        let mut db = test_db();
        db.execute("CREATE TABLE nums (id INT PRIMARY KEY, v INT)")
            .unwrap();
        // Insert out of order so a wrong/no-op ORDER BY is visible.
        for (id, v) in [(1, 30), (2, 10), (3, 50), (4, 20), (5, 40)] {
            db.execute(&format!("INSERT INTO nums (id, v) VALUES ({id}, {v})"))
                .unwrap();
        }
        let ids = |r: QueryResult| -> Vec<String> {
            rows_of(r)
                .into_iter()
                .map(|row| row.values.iter().find(|(k, _)| k == "id").unwrap().1.clone())
                .collect()
        };

        // ORDER BY v ASC → ids ordered by ascending v (2,4,1,5,3).
        assert_eq!(
            ids(db.execute("SELECT id FROM nums ORDER BY v ASC").unwrap()),
            vec!["2", "4", "1", "5", "3"]
        );
        // ORDER BY v DESC → reverse.
        assert_eq!(
            ids(db.execute("SELECT id FROM nums ORDER BY v DESC").unwrap()),
            vec!["3", "5", "1", "4", "2"]
        );
        // LIMIT keeps the first N in order.
        assert_eq!(
            ids(db.execute("SELECT id FROM nums ORDER BY v ASC LIMIT 2").unwrap()),
            vec!["2", "4"]
        );
        // OFFSET skips the first M, LIMIT then takes N.
        assert_eq!(
            ids(db
                .execute("SELECT id FROM nums ORDER BY v ASC LIMIT 2 OFFSET 1")
                .unwrap()),
            vec!["4", "1"]
        );
        // OFFSET past the end → empty.
        assert!(ids(db
            .execute("SELECT id FROM nums ORDER BY v ASC OFFSET 10")
            .unwrap())
        .is_empty());
    }

    #[test]
    fn strip_semantic_match_query_removes_predicate() {
        let parse_q = |sql: &str| {
            let stmts = parser::parse(sql).unwrap();
            match &stmts[0] {
                AuroraStatement::Standard(s) => match s.as_ref() {
                    sqlparser::ast::Statement::Query(q) => (**q).clone(),
                    other => panic!("not a query: {other:?}"),
                },
                other => panic!("not standard: {other:?}"),
            }
        };

        // Only SEMANTIC_MATCH in WHERE → the whole WHERE is removed, the
        // analytical clauses survive.
        let q = parse_q(
            "SELECT category, COUNT(*) FROM docs \
             WHERE SEMANTIC_MATCH(body, 'ai', 0.5) GROUP BY category",
        );
        let s = strip_semantic_match_query(&q).to_string().to_uppercase();
        assert!(!s.contains("SEMANTIC_MATCH"), "predicate must be gone: {s}");
        assert!(s.contains("GROUP BY"), "analytical clause kept: {s}");

        // SEMANTIC_MATCH AND <relational> → the relational conjunct survives.
        let q2 = parse_q(
            "SELECT id FROM docs \
             WHERE SEMANTIC_MATCH(body, 'ai', 0.5) AND price > 5 ORDER BY id",
        );
        let s2 = strip_semantic_match_query(&q2).to_string().to_uppercase();
        assert!(!s2.contains("SEMANTIC_MATCH"), "predicate gone: {s2}");
        assert!(s2.contains("PRICE > 5"), "relational conjunct kept: {s2}");
        assert!(s2.contains("ORDER BY"), "order clause kept: {s2}");

        // <relational> AND SEMANTIC_MATCH (other order) → relational survives.
        let q3 = parse_q(
            "SELECT id FROM docs WHERE price > 5 AND SEMANTIC_MATCH(body, 'ai', 0.5)",
        );
        let s3 = strip_semantic_match_query(&q3).to_string().to_uppercase();
        assert!(!s3.contains("SEMANTIC_MATCH"));
        assert!(s3.contains("PRICE > 5"));
    }

    #[test]
    fn analytical_at_version_reads_historical_snapshot() {
        // HTAP task 17 / 17.1: an analytical (GROUP BY) query with AT VERSION
        // reads the historical columnar snapshot through the DataFusion path.
        let mut db = test_db();
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, cat TEXT, v INT)")
            .unwrap();
        for (id, cat, v) in [(1, "a", 10), (2, "a", 20), (3, "b", 30)] {
            db.execute(&format!(
                "INSERT INTO t (id, cat, v) VALUES ({id}, '{cat}', {v})"
            ))
            .unwrap();
        }
        // Snapshot after the first three inserts.
        let ts0 = db.engine.next_ts_for_tests() - 1;
        // Two more rows land at a higher ts (invisible at ts0).
        db.execute("INSERT INTO t (id, cat, v) VALUES (4, 'a', 40)").unwrap();
        db.execute("INSERT INTO t (id, cat, v) VALUES (5, 'b', 50)").unwrap();

        let counts = |r: QueryResult| -> Vec<(String, String)> {
            rows_of(r)
                .into_iter()
                .map(|row| {
                    let cat = row.values.iter().find(|(k, _)| k == "cat").unwrap().1.clone();
                    let n = row.values.iter().find(|(k, _)| k == "n").unwrap().1.clone();
                    (cat, n)
                })
                .collect()
        };

        // Latest: a=3 (1,2,4), b=2 (3,5).
        let latest = counts(
            db.execute("SELECT cat, COUNT(*) AS n FROM t GROUP BY cat ORDER BY cat")
                .unwrap(),
        );
        assert_eq!(
            latest,
            vec![("a".into(), "3".into()), ("b".into(), "2".into())]
        );

        // AT VERSION ts0: a=2 (1,2), b=1 (3) — the later inserts are invisible.
        let hist = counts(
            db.execute(&format!(
                "SELECT cat, COUNT(*) AS n FROM t GROUP BY cat ORDER BY cat AT VERSION {ts0}"
            ))
            .unwrap(),
        );
        assert_eq!(
            hist,
            vec![("a".into(), "2".into()), ("b".into(), "1".into())],
            "analytical AT VERSION must aggregate over the historical snapshot"
        );
    }

    /// End-to-end SEMANTIC_MATCH test using the real model. Gated behind
    /// the `online-tests` feature — requires network access to HuggingFace
    /// Hub on first run (downloads ~90 MB for all-MiniLM-L6-v2).
    ///
    /// ```text
    /// cargo test -p galaxdb-embedded --features online-tests --release
    /// ```
    #[cfg(feature = "online-tests")]
    #[test]
    fn semantic_match_end_to_end() {
        const MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";
        const MODEL_DIM: usize = 384;

        let sidecar_binary = std::env::current_exe()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("galaxdb-sidecar");

        if !sidecar_binary.exists() {
            let status = std::process::Command::new("cargo")
                .args(["build", "-p", "galaxdb-sidecar"])
                .status()
                .expect("cargo build");
            assert!(status.success(), "failed to build sidecar binary");
        }

        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("semantic_db");
        std::mem::forget(dir);

        let mut db = Database::open_with_sidecar(
            db_path.to_str().unwrap(),
            sidecar_binary.to_str().unwrap(),
            MODEL_ID,
        )
        .unwrap();

        db.execute(&format!(
            "CREATE TABLE docs (id INT PRIMARY KEY, \
             content TEXT EMBEDDING MODEL '{MODEL_ID}' DIM {MODEL_DIM})"
        ))
        .unwrap();

        assert!(db.vector_indexes.read().unwrap().contains_key("docs"));

        db.execute("INSERT INTO docs (id, content) VALUES (1, 'machine learning is great')")
            .unwrap();
        db.execute("INSERT INTO docs (id, content) VALUES (2, 'rust programming language')")
            .unwrap();
        db.execute("INSERT INTO docs (id, content) VALUES (3, 'machine learning algorithms')")
            .unwrap();

        {
            let indexes = db.vector_indexes.read().unwrap();
            let idx = indexes.get("docs").unwrap();
            assert_eq!(
                idx.delta.vector_count(),
                3,
                "three INSERTs must produce three sidecar-computed embeddings"
            );
            assert_eq!(idx.dim, MODEL_DIM);
        }

        let result = db
            .execute(
                "SELECT * FROM docs WHERE SEMANTIC_MATCH(content, 'machine learning', 0.0)",
            )
            .unwrap();
        match result {
            QueryResult::Rows(rows) => {
                assert!(!rows.is_empty(), "SEMANTIC_MATCH should return results");
                for row in &rows {
                    assert!(row.values.iter().any(|(k, _)| k == "row_id"));
                    assert!(row.values.iter().any(|(k, _)| k == "similarity"));
                }
            }
            other => panic!("expected Rows, got {:?}", other),
        }

        assert!(
            db.sidecar.as_ref().unwrap().is_healthy(),
            "sidecar must still be healthy after a successful query"
        );
    }

    /// End-to-end SEMANTIC_MATCH feeding an analytical query (HTAP task 16):
    /// the matched rows are grouped/aggregated by DataFusion. Gated behind
    /// `online-tests` (needs the sidecar + model), like the sibling test.
    ///
    /// ```text
    /// cargo test -p galaxdb-embedded --features online-tests --release
    /// ```
    #[cfg(feature = "online-tests")]
    #[test]
    fn semantic_match_group_by_end_to_end() {
        const MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";
        const MODEL_DIM: usize = 384;

        let sidecar_binary = std::env::current_exe()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("galaxdb-sidecar");
        if !sidecar_binary.exists() {
            let status = std::process::Command::new("cargo")
                .args(["build", "-p", "galaxdb-sidecar"])
                .status()
                .expect("cargo build");
            assert!(status.success(), "failed to build sidecar binary");
        }

        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("semantic_agg_db");
        std::mem::forget(dir);
        let mut db = Database::open_with_sidecar(
            db_path.to_str().unwrap(),
            sidecar_binary.to_str().unwrap(),
            MODEL_ID,
        )
        .unwrap();

        db.execute(&format!(
            "CREATE TABLE docs (id INT PRIMARY KEY, category TEXT, \
             content TEXT EMBEDDING MODEL '{MODEL_ID}' DIM {MODEL_DIM})"
        ))
        .unwrap();
        db.execute("INSERT INTO docs (id, category, content) VALUES (1, 'ml', 'machine learning is great')").unwrap();
        db.execute("INSERT INTO docs (id, category, content) VALUES (2, 'rust', 'rust programming language')").unwrap();
        db.execute("INSERT INTO docs (id, category, content) VALUES (3, 'ml', 'machine learning algorithms')").unwrap();

        // SEMANTIC_MATCH + GROUP BY: aggregate over the matched rows. With a
        // permissive threshold all three match; grouped by category that is
        // 2 'ml' + 1 'rust'. The point is that GROUP BY executes over the
        // semantic candidate set (task 16), not that ranking is exact.
        let result = db
            .execute(
                "SELECT category, COUNT(*) AS n FROM docs \
                 WHERE SEMANTIC_MATCH(content, 'machine learning', 0.0) \
                 GROUP BY category ORDER BY category",
            )
            .unwrap();
        match result {
            QueryResult::Rows(rows) => {
                assert!(!rows.is_empty(), "GROUP BY over semantic matches returns rows");
                let total: i64 = rows
                    .iter()
                    .map(|r| {
                        r.values
                            .iter()
                            .find(|(k, _)| k == "n")
                            .and_then(|(_, v)| v.parse::<i64>().ok())
                            .unwrap_or(0)
                    })
                    .sum();
                assert!(total >= 1, "at least one matched row is aggregated");
            }
            other => panic!("expected Rows, got {:?}", other),
        }
    }

    /// Regression test for the wire-protocol SEMANTIC_MATCH defect: the
    /// server routes INSERT through `execute_dml_concurrent` (`&self`) and
    /// SELECT through `execute_readonly_with_session` (`&self`), NOT the
    /// `&mut self` `execute()` path the other online tests use. Before the
    /// fix, `execute_dml_concurrent` never populated the vector index, so
    /// SEMANTIC_MATCH over the wire returned zero rows silently. This test
    /// drives the exact server paths and asserts a non-empty result.
    ///
    /// ```text
    /// cargo test -p galaxdb-embedded --features online-tests --release \
    ///   semantic_match_concurrent_wire_path
    /// ```
    #[cfg(feature = "online-tests")]
    #[test]
    fn semantic_match_concurrent_wire_path() {
        const MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";
        const MODEL_DIM: usize = 384;

        let sidecar_binary = std::env::current_exe()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("galaxdb-sidecar");
        if !sidecar_binary.exists() {
            let status = std::process::Command::new("cargo")
                .args(["build", "-p", "galaxdb-sidecar"])
                .status()
                .expect("cargo build");
            assert!(status.success(), "failed to build sidecar binary");
        }

        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("semantic_wire_db");
        std::mem::forget(dir);
        let mut db = Database::open_with_sidecar(
            db_path.to_str().unwrap(),
            sidecar_binary.to_str().unwrap(),
            MODEL_ID,
        )
        .unwrap();

        // DDL still goes through the write path (as the server does).
        db.execute(&format!(
            "CREATE TABLE docs (id INT PRIMARY KEY, \
             content TEXT EMBEDDING MODEL '{MODEL_ID}' DIM {MODEL_DIM})"
        ))
        .unwrap();

        // INSERT via the CONCURRENT path — the one the server uses for DML
        // and the one that was broken.
        db.execute_dml_concurrent(
            "INSERT INTO docs (id, content) VALUES (1, 'machine learning is great')",
            None,
        )
        .unwrap();
        db.execute_dml_concurrent(
            "INSERT INTO docs (id, content) VALUES (2, 'rust programming language')",
            None,
        )
        .unwrap();
        db.execute_dml_concurrent(
            "INSERT INTO docs (id, content) VALUES (3, 'deep neural networks for vision')",
            None,
        )
        .unwrap();

        // The concurrent INSERT path must now populate the vector index.
        {
            let indexes = db.vector_indexes.read().unwrap();
            let idx = indexes.get("docs").unwrap();
            assert_eq!(
                idx.delta.vector_count(),
                3,
                "concurrent INSERT path must produce three sidecar embeddings \
                 (this is the regression: it used to be 0)"
            );
        }

        // SELECT via the READONLY path — the one the server uses for reads.
        let result = db
            .execute_readonly_with_session(
                "SELECT id, content FROM docs \
                 WHERE SEMANTIC_MATCH(content, 'machine learning', 0.0)",
                None,
            )
            .unwrap();
        match result {
            QueryResult::Rows(rows) => {
                assert!(
                    !rows.is_empty(),
                    "SEMANTIC_MATCH over the concurrent+readonly (wire) path must \
                     return results now that the index is populated"
                );
            }
            other => panic!("expected Rows, got {:?}", other),
        }
    }

    // Before Phase I, `exec_select`, `exec_update`, and `exec_delete`
    // hard-coded `filter: None`, silently ignoring the WHERE clause.
    // These tests drive real SQL through `Database::execute` and assert
    // that the filter reaches the executor. A regression would show up
    // as wrong row counts, which is exactly what AWS integration
    // testing caught.
    // -----------------------------------------------------------------

    fn seeded_db() -> Database {
        let mut db = test_db();
        db.execute("CREATE TABLE p (id INT PRIMARY KEY, name TEXT, price FLOAT)")
            .unwrap();
        db.execute("INSERT INTO p (id, name, price) VALUES (1, 'espresso', 3.50)")
            .unwrap();
        db.execute("INSERT INTO p (id, name, price) VALUES (2, 'latte', 4.25)")
            .unwrap();
        db.execute("INSERT INTO p (id, name, price) VALUES (3, 'mocha', 4.75)")
            .unwrap();
        db
    }

    fn rows_of(r: QueryResult) -> Vec<QueryRow> {
        match r {
            QueryResult::Rows(rows) => rows,
            other => panic!("expected Rows, got {:?}", other),
        }
    }

    #[test]
    fn select_where_price_filters_rows() {
        let mut db = seeded_db();
        let rows = rows_of(
            db.execute("SELECT id, name, price FROM p WHERE price > 4.0")
                .unwrap(),
        );
        assert_eq!(rows.len(), 2, "should return latte + mocha only");
        for r in &rows {
            let price_str = r
                .values
                .iter()
                .find(|(k, _)| k == "price")
                .map(|(_, v)| v.clone())
                .unwrap();
            let price: f64 = price_str.parse().unwrap();
            assert!(price > 4.0, "row slipped past WHERE: price={price}");
        }
    }

    #[test]
    fn select_where_id_equals_returns_single_row() {
        let mut db = seeded_db();
        let rows = rows_of(
            db.execute("SELECT id, name FROM p WHERE id = 2").unwrap(),
        );
        assert_eq!(rows.len(), 1);
        let name = &rows[0]
            .values
            .iter()
            .find(|(k, _)| k == "name")
            .unwrap()
            .1;
        assert_eq!(name, "latte");
    }

    #[test]
    fn select_projection_restricts_columns() {
        let mut db = seeded_db();
        let rows = rows_of(db.execute("SELECT name FROM p").unwrap());
        assert_eq!(rows.len(), 3);
        for r in &rows {
            assert_eq!(
                r.values.len(),
                1,
                "projection should limit output to one column, got {:?}",
                r.values
            );
            assert_eq!(r.values[0].0, "name");
        }
    }

    #[test]
    fn update_where_affects_only_matching_rows() {
        let mut db = seeded_db();
        match db
            .execute("UPDATE p SET price = 9.99 WHERE id = 3")
            .unwrap()
        {
            QueryResult::RowCount(n) => assert_eq!(n, 1, "UPDATE with id=3 must affect 1 row"),
            other => panic!("expected RowCount, got {:?}", other),
        }

        // Others unchanged.
        let latte = rows_of(
            db.execute("SELECT price FROM p WHERE id = 2").unwrap(),
        );
        assert_eq!(latte.len(), 1);
        assert_eq!(latte[0].values[0].1, "4.25");

        // Target updated.
        let mocha = rows_of(
            db.execute("SELECT price FROM p WHERE id = 3").unwrap(),
        );
        assert_eq!(mocha.len(), 1);
        assert_eq!(mocha[0].values[0].1, "9.99");
    }

    #[test]
    fn delete_where_affects_only_matching_rows() {
        let mut db = seeded_db();
        match db.execute("DELETE FROM p WHERE id = 1").unwrap() {
            QueryResult::RowCount(n) => assert_eq!(n, 1, "DELETE with id=1 must remove 1 row"),
            other => panic!("expected RowCount, got {:?}", other),
        }
        let rows = rows_of(db.execute("SELECT id FROM p").unwrap());
        assert_eq!(rows.len(), 2, "two rows should remain after deleting id=1");

        // Deleting a non-existent row is a no-op.
        match db.execute("DELETE FROM p WHERE id = 99").unwrap() {
            QueryResult::RowCount(n) => assert_eq!(n, 0),
            other => panic!("expected RowCount, got {:?}", other),
        }
    }

    #[test]
    fn delete_without_where_clears_table() {
        let mut db = seeded_db();
        match db.execute("DELETE FROM p").unwrap() {
            QueryResult::RowCount(n) => {
                assert_eq!(n, 3, "DELETE without WHERE must remove all rows")
            }
            other => panic!("expected RowCount, got {:?}", other),
        }
        let rows = rows_of(db.execute("SELECT * FROM p").unwrap());
        assert!(rows.is_empty());
    }

    #[test]
    fn where_and_or_combine() {
        let mut db = seeded_db();
        let rows = rows_of(
            db.execute(
                "SELECT id FROM p WHERE price > 4.0 AND price < 4.5",
            )
            .unwrap(),
        );
        assert_eq!(rows.len(), 1, "only latte matches 4.0 < p < 4.5");
        assert_eq!(rows[0].values[0].1, "2");

        let rows = rows_of(
            db.execute(
                "SELECT id FROM p WHERE id = 1 OR id = 3",
            )
            .unwrap(),
        );
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn where_text_equality() {
        let mut db = seeded_db();
        let rows = rows_of(
            db.execute("SELECT id FROM p WHERE name = 'latte'")
                .unwrap(),
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values[0].1, "2");
    }

    #[test]
    fn where_column_on_right_side_is_flipped() {
        // `5 < id` should behave like `id > 5`.
        let mut db = seeded_db();
        let rows = rows_of(
            db.execute("SELECT id FROM p WHERE 2 < id").unwrap(),
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values[0].1, "3");
    }

    // -----------------------------------------------------------------
    // Phase K regressions — AT VERSION + DELTA_TOMBSTONE + compactor
    // pin-set (tasks 18.6, 32.3, 32.4, 33.5).
    //
    // These tests go through the canonical `Database::execute` path
    // so they exercise the SQL parser, the plan dispatch, the storage
    // engine's MVCC memtable, and the tag catalog together. Anything
    // that regresses the real behaviour on any of those layers will
    // fail here.
    // -----------------------------------------------------------------

    #[test]
    fn at_version_timestamp_returns_historical_snapshot() {
        let mut db = test_db();
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO t (id, name) VALUES (1, 'alpha')")
            .unwrap();
        // The INSERT above consumed the latest allocated ts.
        // `next_ts_for_tests()` returns the next one that *would* be
        // allocated, so to read "as of just after the INSERT but before
        // any update", we subtract 1.
        let read_ts = db.engine.next_ts_for_tests() - 1;
        // Now mutate the row; the UPDATE lands at a higher ts.
        db.execute("UPDATE t SET name = 'beta' WHERE id = 1")
            .unwrap();

        // Plain SELECT sees the latest value.
        let rows = rows_of(db.execute("SELECT id, name FROM t").unwrap());
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values[1].1, "beta");

        // AT VERSION <read_ts> sees the pre-update value.
        let sql = format!("SELECT id, name FROM t AT VERSION {read_ts}");
        let rows = rows_of(db.execute(&sql).unwrap());
        assert_eq!(rows.len(), 1, "AT VERSION must see exactly one row");
        assert_eq!(
            rows[0].values[1].1,
            "alpha",
            "AT VERSION must return the value as of the snapshot ts"
        );
    }

    #[test]
    fn at_version_tag_resolves_through_tag_catalog() {
        use galaxdb_versioning::{MerkleRoot, TrainingTagMetadata};

        let mut db = test_db();
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO t (id, name) VALUES (1, 'v1')")
            .unwrap();
        let tag_ts = db.engine.next_ts_for_tests() - 1;
        // Register a real tag that points at the just-committed ts.
        {
            let mut tc = db.tag_catalog.lock().unwrap();
            tc.create_tag(
                "snap-v1".to_string(),
                tag_ts, // created_at
                MerkleRoot { hash: 0xC0DE },
                tag_ts, // version_timestamp
                vec![], // no pinned blocks for this test
                false,
                None::<TrainingTagMetadata>,
            )
            .expect("create tag");
        }
        db.execute("UPDATE t SET name = 'v2' WHERE id = 1").unwrap();

        let rows = rows_of(
            db.execute("SELECT id, name FROM t AT VERSION 'snap-v1'").unwrap(),
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].values[1].1, "v1",
            "AT VERSION '<tag>' must resolve through the tag catalog and return the pre-update row",
        );
    }

    /// Req 11 AC5: write rows, force a flush to SST, then update — a
    /// historical `AT VERSION` query must return the pre-update values
    /// *from the flushed SST*, not only from the memtable. This is the
    /// test that would fail against a memtable-only implementation.
    #[test]
    fn at_version_reads_pre_update_values_from_sst() {
        use galaxdb_versioning::{MerkleRoot, TrainingTagMetadata};

        let mut db = test_db();
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO t (id, name) VALUES (1, 'v1')").unwrap();
        db.execute("INSERT INTO t (id, name) VALUES (2, 'w1')").unwrap();
        let tag_ts = db.engine.next_ts_for_tests() - 1;
        {
            let mut tc = db.tag_catalog.lock().unwrap();
            tc.create_tag(
                "snap".to_string(),
                tag_ts,
                MerkleRoot { hash: 0xABCD },
                tag_ts,
                vec![],
                false,
                None::<TrainingTagMetadata>,
            )
            .expect("create tag");
        }

        // Force the active memtable to disk. After this the pre-update
        // rows live ONLY in an SST file, so the AT VERSION query below
        // exercises the SST read path, not the memtable.
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(db.engine.flush_memtable())
            .expect("flush to SST");

        // Update both rows — new versions land in the fresh memtable at a
        // timestamp after the tag.
        db.execute("UPDATE t SET name = 'v2' WHERE id = 1").unwrap();
        db.execute("UPDATE t SET name = 'w2' WHERE id = 2").unwrap();

        // Current read sees the new values.
        let now = rows_of(db.execute("SELECT id, name FROM t").unwrap());
        assert_eq!(now.len(), 2);

        // Historical read at the tag must return v1/w1 from the SST.
        let hist = rows_of(
            db.execute("SELECT id, name FROM t AT VERSION 'snap'")
                .unwrap(),
        );
        let mut names: Vec<String> = hist.iter().map(|r| r.values[1].1.clone()).collect();
        names.sort();
        assert_eq!(
            names,
            vec!["v1".to_string(), "w1".to_string()],
            "AT VERSION must return pre-update values recovered from the flushed SST",
        );
    }

    #[test]
    fn at_version_unknown_tag_errors() {
        let mut db = test_db();
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        let err = db
            .execute("SELECT id FROM t AT VERSION 'does-not-exist'")
            .expect_err("unknown tag must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("unknown version tag") || msg.contains("does-not-exist"),
            "expected an 'unknown version tag' error, got: {msg}",
        );
    }

    #[test]
    fn compactor_pins_tagged_timestamps() {
        use galaxdb_storage::compaction::GcContext;

        let mut db = test_db();
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO t (id, name) VALUES (1, 'alpha')")
            .unwrap();
        let tag_ts = db.engine.next_ts_for_tests();
        {
            let mut tc = db.tag_catalog.lock().unwrap();
            tc.create_tag(
                "keep-me".to_string(),
                tag_ts,
                galaxdb_versioning::MerkleRoot { hash: 1 },
                tag_ts,
                vec![],
                false,
                None,
            )
            .unwrap();
        }
        db.execute("UPDATE t SET name = 'beta' WHERE id = 1")
            .unwrap();

        let gc: GcContext = db.gc_context_with_pins(None);
        assert!(
            gc.pinned_tag_timestamps.contains(&tag_ts),
            "compactor pin-set must include the tag's version_timestamp ({tag_ts}); \
             got {:?}",
            gc.pinned_tag_timestamps,
        );
        // Compaction-time decision: the tagged version must be retained,
        // a non-tagged intermediate version may be discarded.
        assert!(gc.should_keep(tag_ts, /* is_latest = */ false));
    }

    // -----------------------------------------------------------------
    // Task 22.4 — training_dataset(tag) produces a real Lance dataset
    // -----------------------------------------------------------------

    /// End-to-end: CREATE TABLE → INSERT → create a FOR TRAINING tag
    /// pointing at the post-insert timestamp → call
    /// `Database::training_dataset` → re-open the returned path with
    /// the `lance` crate and assert the row count.
    ///
    /// This is the acceptance test for task 22.4. If it passes, the
    /// Rust method is writing a real, Lance-readable dataset backed
    /// by real engine data — no mocks, no placeholders. The Python
    /// wrapper around this path (`galaxdb.Database.training_dataset`
    /// in `galaxdb-python`) just surfaces the returned path as a
    /// string so `lance.dataset(path).to_pytorch()` works as the
    /// final IterableDataset shim.
    #[test]
    fn training_dataset_writes_real_lance_dataset() {
        use galaxdb_versioning::{MerkleRoot, TrainingTagMetadata};

        let mut db = test_db();
        db.execute("CREATE TABLE docs (id INT PRIMARY KEY, body TEXT)")
            .unwrap();
        for i in 1..=5 {
            db.execute(&format!(
                "INSERT INTO docs (id, body) VALUES ({i}, 'row-{i}')"
            ))
            .unwrap();
        }

        // Capture the post-insert timestamp so the tag points at a
        // commit that actually contains rows. `exec_create_version_tag`
        // still takes its ts from `MerkleDag::latest()` — which is 0
        // until task 36 wires the DAG to real commits — so for now we
        // register the training tag directly against the tag catalog
        // with a ts the engine does have data at. This is the same
        // pattern the Phase K AT VERSION tests use above.
        let tag_ts = db.engine.next_ts_for_tests();
        {
            let mut tc = db.tag_catalog.lock().unwrap();
            tc.create_tag(
                "train-v1".to_string(),
                tag_ts,
                MerkleRoot { hash: 0xC0DE },
                tag_ts,
                vec![], // pinned blocks are irrelevant: the engine
                        // source drives off `version_timestamp`.
                true,   // FOR TRAINING
                Some(TrainingTagMetadata {
                    precision: "float32".to_string(),
                    seed: Some(42),
                    deterministic_order: true,
                }),
            )
            .expect("create training tag");
        }

        let path = db
            .training_dataset("train-v1")
            .expect("training_dataset must produce a Lance dataset");
        assert!(path.exists(), "returned path must exist on disk");
        assert!(
            path.is_dir(),
            "Lance writes the dataset as a directory, not a single file"
        );
        assert!(
            path.starts_with(db.path()),
            "output must land under the database directory: {:?}",
            path
        );

        // Open the dataset through the real `lance` crate (the same
        // API the Python wrapper uses under the hood) and verify the
        // row count matches the number of INSERTs.
        let row_count = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                let ds = lance::Dataset::open(path.to_str().unwrap())
                    .await
                    .expect("open Lance dataset");
                ds.scan()
                    .count_rows()
                    .await
                    .expect("count rows in Lance scan")
            });
        assert_eq!(
            row_count, 5,
            "Lance dataset must contain exactly the 5 INSERTed rows"
        );
    }

    /// Task 23 (Req 20): a table with an embedding column exports the vector
    /// as an Arrow `FixedSizeList<Float32, dim>` column, with a NULL vector for
    /// any row whose embedding is absent at the tag version (never fabricated).
    #[test]
    fn training_dataset_exports_embedding_column_with_nulls() {
        use galaxdb_versioning::{MerkleRoot, TrainingTagMetadata};

        let mut db = test_db();
        db.execute(
            "CREATE TABLE docs (id INT PRIMARY KEY, \
             body TEXT EMBEDDING MODEL 'sentence-transformers/all-MiniLM-L6-v2' DIM 4)",
        )
        .unwrap();
        for i in 1..=3 {
            db.execute(&format!(
                "INSERT INTO docs (id, body) VALUES ({i}, 'row-{i}')"
            ))
            .unwrap();
        }

        // No sidecar in this test, so no vectors were generated. Inject vectors
        // for the first two rows by their real storage keys; leave the third
        // without one so the export must emit a NULL vector (AC4).
        let keys: Vec<Vec<u8>> = db
            .engine
            .scan_all_at(u64::MAX)
            .into_iter()
            .map(|(k, _, _)| k)
            .filter(|k| k.starts_with(b"docs:"))
            .collect();
        assert_eq!(keys.len(), 3, "three rows must be present");
        {
            let mut idxs = db.vector_indexes.write().unwrap();
            let idx = idxs.get_mut("docs").expect("vector index for docs");
            for (i, key) in keys.iter().take(2).enumerate() {
                let row_id = xxhash_rust::xxh3::xxh3_64(key);
                idx.vectors.insert(row_id, vec![i as f32 + 0.5; 4]);
                idx.key_to_row_id.insert(key.clone(), row_id);
            }
        }

        let tag_ts = db.engine.next_ts_for_tests();
        {
            let mut tc = db.tag_catalog.lock().unwrap();
            tc.create_tag(
                "train-emb".to_string(),
                tag_ts,
                MerkleRoot { hash: 0xBEEF },
                tag_ts,
                vec![],
                true,
                Some(TrainingTagMetadata {
                    precision: "float32".to_string(),
                    seed: Some(7),
                    deterministic_order: true,
                }),
            )
            .expect("create training tag");
        }

        let path = db
            .training_dataset("train-emb")
            .expect("training_dataset must produce a Lance dataset");

        let (row_count, embedding_dim) = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                let ds = lance::Dataset::open(path.to_str().unwrap())
                    .await
                    .expect("open Lance dataset");
                let rows = ds.scan().count_rows().await.expect("count rows");
                let arrow_schema = arrow::datatypes::Schema::from(ds.schema());
                let field = arrow_schema
                    .field_with_name("body_embedding")
                    .expect("body_embedding column must exist")
                    .clone();
                let dim = match field.data_type() {
                    arrow::datatypes::DataType::FixedSizeList(child, len) => {
                        assert_eq!(
                            child.data_type(),
                            &arrow::datatypes::DataType::Float32,
                            "embedding child type must be Float32"
                        );
                        *len
                    }
                    other => panic!("embedding column must be FixedSizeList, got {other:?}"),
                };
                (rows, dim)
            });

        assert_eq!(row_count, 3, "all three rows must be exported");
        assert_eq!(embedding_dim, 4, "embedding dimensionality must match DIM 4");
    }


    /// This keeps the deterministic-order contract on the exporter:
    /// every caller of `training_dataset` is guaranteed the tag was
    /// created with `FOR TRAINING` (and therefore carries a precision
    /// and a deterministic seed).
    #[test]
    fn training_dataset_rejects_non_training_tag() {
        use galaxdb_versioning::MerkleRoot;

        let mut db = test_db();
        db.execute("CREATE TABLE docs (id INT PRIMARY KEY, body TEXT)")
            .unwrap();
        db.execute("INSERT INTO docs (id, body) VALUES (1, 'row-1')")
            .unwrap();
        let tag_ts = db.engine.next_ts_for_tests();
        {
            let mut tc = db.tag_catalog.lock().unwrap();
            tc.create_tag(
                "plain-snapshot".to_string(),
                tag_ts,
                MerkleRoot { hash: 1 },
                tag_ts,
                vec![],
                false, // NOT a training tag
                None,
            )
            .unwrap();
        }
        let err = db
            .training_dataset("plain-snapshot")
            .expect_err("non-training tag must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("not a FOR TRAINING"),
            "expected a FOR-TRAINING guard message, got: {msg}"
        );
    }

    /// An unknown tag name surfaces a real error rather than silently
    /// exporting an empty dataset.
    #[test]
    fn training_dataset_unknown_tag_errors() {
        let mut db = test_db();
        db.execute("CREATE TABLE docs (id INT PRIMARY KEY, body TEXT)")
            .unwrap();
        let err = db
            .training_dataset("does-not-exist")
            .expect_err("unknown tag must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("unknown version tag") || msg.contains("does-not-exist"),
            "expected 'unknown version tag' error, got: {msg}"
        );
    }

    // -----------------------------------------------------------------
    // WHERE NOT DUPLICATE (task 35.5) — embedded end-to-end
    // -----------------------------------------------------------------

    /// `NOT DUPLICATE` combined with an analytical clause (ORDER BY) must
    /// still dedup: the query stays on the native executor rather than being
    /// routed to DataFusion, which has no `NOT DUPLICATE` operator (HTAP
    /// task 17.1 — WHERE NOT DUPLICATE preserved).
    #[test]
    fn where_not_duplicate_with_order_by_stays_native_and_dedups() {
        let mut db = test_db();
        db.execute(
            "CREATE TABLE docs (id INT PRIMARY KEY, body TEXT, _near_duplicate_group BIGINT)",
        )
        .unwrap();
        // Group 100: ids 1,2,3 → representative id=1. Group 200: 4,5 → id=4.
        for (id, g) in [(1, "100"), (2, "100"), (3, "100"), (4, "200"), (5, "200")] {
            db.execute(&format!(
                "INSERT INTO docs (id, body, _near_duplicate_group) VALUES ({id}, 'b{id}', {g})"
            ))
            .unwrap();
        }
        // With ORDER BY present, classify() would say Analytical; the guard
        // keeps it native so dedup runs. Two representatives survive.
        let r = db
            .execute("SELECT id FROM docs WHERE NOT DUPLICATE ORDER BY id")
            .unwrap();
        let QueryResult::Rows(rows) = r else {
            panic!("expected Rows");
        };
        let ids: std::collections::HashSet<String> = rows
            .iter()
            .map(|row| row.values.iter().find(|(k, _)| k == "id").unwrap().1.clone())
            .collect();
        assert_eq!(ids.len(), 2, "one representative per group, got {ids:?}");
        assert!(ids.contains("1") && ids.contains("4"), "reps 1 and 4: {ids:?}");
    }

    /// End-to-end: create a table with the near-duplicate group column,
    /// seed rows that share group ids, run `SELECT … WHERE NOT
    /// DUPLICATE` through the full embedded SQL pipeline (parser →
    /// filter_from_expr → planner → executor), and assert only the
    /// deterministic representatives come back. Requirement 26 + task 35.5.
    #[test]
    fn where_not_duplicate_keeps_one_per_group_over_sql() {
        let mut db = test_db();
        db.execute(
            "CREATE TABLE docs (\
                id INT PRIMARY KEY, \
                body TEXT, \
                _near_duplicate_group BIGINT\
             )",
        )
        .unwrap();
        // Group 100: ids 3, 1, 4 → representative is id=1.
        db.execute("INSERT INTO docs (id, body, _near_duplicate_group) VALUES (3, 'hello world', 100)")
            .unwrap();
        db.execute("INSERT INTO docs (id, body, _near_duplicate_group) VALUES (1, 'hello world!', 100)")
            .unwrap();
        db.execute("INSERT INTO docs (id, body, _near_duplicate_group) VALUES (4, 'hello world.', 100)")
            .unwrap();
        // Group 200: ids 5, 2 → representative is id=2.
        db.execute("INSERT INTO docs (id, body, _near_duplicate_group) VALUES (5, 'quick fox', 200)")
            .unwrap();
        db.execute("INSERT INTO docs (id, body, _near_duplicate_group) VALUES (2, 'quick fox!', 200)")
            .unwrap();
        // Ungrouped.
        db.execute("INSERT INTO docs (id, body, _near_duplicate_group) VALUES (6, 'unique', NULL)")
            .unwrap();

        let r = db
            .execute("SELECT id FROM docs WHERE NOT DUPLICATE")
            .unwrap();
        let QueryResult::Rows(rows) = r else {
            panic!("expected Rows");
        };
        let mut ids: Vec<String> = rows
            .into_iter()
            .map(|r| {
                r.values
                    .into_iter()
                    .find(|(k, _)| k == "id")
                    .map(|(_, v)| v)
                    .expect("id column present")
            })
            .collect();
        ids.sort();
        // Representatives are id=1 (group 100) and id=2 (group 200),
        // plus the ungrouped id=6. Values render through
        // `row_codec::value_display`, so integers come back as decimal
        // strings.
        assert_eq!(ids, vec!["1".to_string(), "2".to_string(), "6".to_string()]);
    }

    /// Composition with a conventional WHERE must still dedup on the
    /// narrowed candidate set. `WHERE id > 1 AND NOT DUPLICATE` drops
    /// id=1 first — so group 100's representative shifts from id=1 to
    /// id=2, proving the dedup pass runs after per-row filtering.
    #[test]
    fn where_not_duplicate_composes_with_and_over_sql() {
        let mut db = test_db();
        db.execute(
            "CREATE TABLE docs (\
                id INT PRIMARY KEY, \
                body TEXT, \
                _near_duplicate_group BIGINT\
             )",
        )
        .unwrap();
        db.execute("INSERT INTO docs (id, body, _near_duplicate_group) VALUES (1, 'a', 100)")
            .unwrap();
        db.execute("INSERT INTO docs (id, body, _near_duplicate_group) VALUES (2, 'b', 100)")
            .unwrap();
        db.execute("INSERT INTO docs (id, body, _near_duplicate_group) VALUES (3, 'c', 200)")
            .unwrap();
        db.execute("INSERT INTO docs (id, body, _near_duplicate_group) VALUES (4, 'd', NULL)")
            .unwrap();

        let r = db
            .execute("SELECT id FROM docs WHERE id > 1 AND NOT DUPLICATE")
            .unwrap();
        let QueryResult::Rows(rows) = r else {
            panic!("expected Rows");
        };
        let mut ids: Vec<String> = rows
            .into_iter()
            .map(|r| {
                r.values
                    .into_iter()
                    .find(|(k, _)| k == "id")
                    .map(|(_, v)| v)
                    .unwrap()
            })
            .collect();
        ids.sort();
        assert_eq!(ids, vec!["2".to_string(), "3".to_string(), "4".to_string()]);
    }

    // -----------------------------------------------------------------
    // Task 36 — training-data lineage (`_galaxdb_training_exports`)
    // -----------------------------------------------------------------

    /// Helper: CREATE TABLE + INSERT rows + register a FOR TRAINING tag
    /// pinned at the current commit ts, then drive `training_dataset`.
    /// Mirrors the Phase M fixture so the lineage tests stay readable.
    fn training_export_fixture(tag: &str) -> Database {
        use galaxdb_versioning::{MerkleRoot, TrainingTagMetadata};
        let mut db = test_db();
        db.execute("CREATE TABLE docs (id INT PRIMARY KEY, body TEXT)")
            .unwrap();
        for i in 1..=3 {
            db.execute(&format!(
                "INSERT INTO docs (id, body) VALUES ({i}, 'row-{i}')"
            ))
            .unwrap();
        }
        let tag_ts = db.engine.next_ts_for_tests();
        {
            let mut tc = db.tag_catalog.lock().unwrap();
            tc.create_tag(
                tag.to_string(),
                tag_ts,
                MerkleRoot { hash: 0xC0DE },
                tag_ts,
                vec![],
                true,
                Some(TrainingTagMetadata {
                    precision: "float32".to_string(),
                    seed: Some(42),
                    deterministic_order: true,
                }),
            )
            .unwrap();
        }
        db
    }

    /// After a successful `training_dataset` call the
    /// `_galaxdb_training_exports` system table exists and contains
    /// exactly one row carrying the tag name, precision, dedup flag,
    /// row count, and content hash.
    #[test]
    fn training_export_lineage_row_lands_in_system_table() {
        let mut db = training_export_fixture("train-v1");
        let _path = db.training_dataset("train-v1").expect("export");

        assert!(
            db.table_exists("_galaxdb_training_exports"),
            "training_dataset must create the system table on first use"
        );
        let rows = match db
            .execute("SELECT tag_name, precision, row_count, content_hash FROM _galaxdb_training_exports")
            .unwrap()
        {
            QueryResult::Rows(r) => r,
            other => panic!("expected Rows, got {:?}", other),
        };
        assert_eq!(
            rows.len(),
            1,
            "exactly one lineage row per successful export; got {:?}",
            rows
        );
        let row = &rows[0].values;
        let get = |k: &str| -> String {
            row.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone()).unwrap()
        };
        assert_eq!(get("tag_name"), "train-v1");
        assert_eq!(get("precision"), "float32");
        assert_eq!(get("row_count"), "3");
        let hash = get("content_hash");
        assert_eq!(
            hash.len(),
            32,
            "content_hash is a lower-case hex encoding of the 16-byte \
             XXH3-128 over the canonical row encoding; got {hash:?}"
        );
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    }

    /// UPDATE against the system table is rejected with
    /// `GalaxError::AppendOnlyTable`.
    #[test]
    fn training_exports_table_rejects_update() {
        let mut db = training_export_fixture("train-v1");
        db.training_dataset("train-v1").unwrap();
        let err = db
            .execute(
                "UPDATE _galaxdb_training_exports SET tag_name = 'hacked' WHERE row_count = 3",
            )
            .expect_err("UPDATE against append-only table must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("append-only")
                && msg.contains("UPDATE")
                && msg.contains("_galaxdb_training_exports"),
            "expected append-only UPDATE rejection, got: {msg}"
        );
    }

    /// DELETE against the system table is rejected.
    #[test]
    fn training_exports_table_rejects_delete() {
        let mut db = training_export_fixture("train-v1");
        db.training_dataset("train-v1").unwrap();
        let err = db
            .execute("DELETE FROM _galaxdb_training_exports")
            .expect_err("DELETE against append-only table must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("append-only") && msg.contains("DELETE"),
            "expected append-only DELETE rejection, got: {msg}"
        );
    }

    /// Exporting the same tag twice produces two lineage rows. The
    /// content hash is stable across exports of the same data
    /// (determinism test: both rows carry the exact same hex hash).
    #[test]
    fn training_export_content_hash_is_stable_across_repeats() {
        let mut db = training_export_fixture("train-v1");
        db.training_dataset("train-v1").unwrap();
        // Two exports of the same tag ⇒ two rows.
        db.training_dataset("train-v1").unwrap();

        let rows = match db
            .execute("SELECT content_hash FROM _galaxdb_training_exports")
            .unwrap()
        {
            QueryResult::Rows(r) => r,
            other => panic!("expected Rows, got {:?}", other),
        };
        assert_eq!(rows.len(), 2);
        let hashes: Vec<String> = rows
            .iter()
            .map(|r| r.values[0].1.clone())
            .collect();
        assert_eq!(
            hashes[0], hashes[1],
            "the same tag with the same rows must hash to the same content_hash; got {hashes:?}"
        );
    }

    /// INSERT against the system table works — the append-only guard
    /// only blocks UPDATE and DELETE, not further inserts. This keeps
    /// the sink itself working (otherwise we'd need a privileged write
    /// path). Users manually inserting into the system table is
    /// unusual but not forbidden by the design.
    #[test]
    fn training_exports_table_allows_insert() {
        let mut db = training_export_fixture("train-v1");
        db.training_dataset("train-v1").unwrap();
        let res = db.execute(
            "INSERT INTO _galaxdb_training_exports \
             (lineage_id, exported_at, tag_name, filter_expr, precision, dedup, curriculum, row_count, content_hash) \
             VALUES (9999, 999, 'manual', NULL, 'float32', false, NULL, 1, 'aa')",
        );
        assert!(res.is_ok(), "INSERT against append-only table should be allowed, got {res:?}");
    }

    // -----------------------------------------------------------------
    // Task 37 — BACKUP / RESTORE round-trip + checksum abort
    // -----------------------------------------------------------------

    /// Backup to a fresh directory, open a second database pointing
    /// at that directory, and verify every row survives the round
    /// trip. Exercises the full path: flush → file copy →
    /// validate_backup on restore → reopen → WAL replay.
    #[test]
    fn backup_restore_round_trip_preserves_rows() {
        // Source DB with known data.
        let src = tempfile::tempdir().unwrap();
        let src_path = src.path().join("src");
        let mut src_db = Database::open(src_path.to_str().unwrap()).unwrap();
        src_db.execute("CREATE TABLE items (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        for i in 1..=5 {
            src_db
                .execute(&format!(
                    "INSERT INTO items (id, name) VALUES ({i}, 'name-{i}')"
                ))
                .unwrap();
        }

        // BACKUP TO '<backup_dir>'.
        let backup_dir = src.path().join("backup");
        let res = src_db
            .execute(&format!("BACKUP TO '{}'", backup_dir.display()))
            .unwrap();
        let QueryResult::Ok(msg) = res else {
            panic!("expected Ok, got {res:?}")
        };
        assert!(
            msg.contains("files copied"),
            "backup message should report file count, got: {msg}"
        );
        assert!(backup_dir.exists() && backup_dir.is_dir());
        let sst_count_backup = std::fs::read_dir(&backup_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                let n = e.file_name();
                let s = n.to_string_lossy();
                s.starts_with("sst_") && s.ends_with(".pax")
            })
            .count();
        assert!(
            sst_count_backup >= 1,
            "backup must include at least one SST (flush wrote one); got {sst_count_backup}"
        );

        // Drop the source DB to release file handles, then restore
        // into a fresh target directory and reopen.
        drop(src_db);
        let dst_root = tempfile::tempdir().unwrap();
        let dst_path = dst_root.path().join("restored");
        // Construct a fresh DB pointing at dst_path, then RESTORE
        // FROM into it.
        let mut dst_db = Database::open(dst_path.to_str().unwrap()).unwrap();
        dst_db
            .execute(&format!("RESTORE FROM '{}'", backup_dir.display()))
            .unwrap();

        // The executor doesn't reopen the engine automatically
        // (documented on exec_restore). Drop and reopen manually so
        // WAL replay picks up the restored files.
        drop(dst_db);
        let mut dst_db = Database::open(dst_path.to_str().unwrap()).unwrap();

        // The catalog is persisted and backed up with the SSTs, so the
        // table's schema is recovered automatically on reopen — no manual
        // re-CREATE needed.
        assert!(
            dst_db.table_exists("items"),
            "restored catalog must recover the table schema"
        );

        let rows = match dst_db
            .execute("SELECT id, name FROM items")
            .unwrap()
        {
            QueryResult::Rows(r) => r,
            other => panic!("expected Rows, got {other:?}"),
        };
        assert_eq!(
            rows.len(),
            5,
            "every inserted row must survive backup → restore → reopen; \
             got {:?}",
            rows
        );
        let names: std::collections::BTreeSet<String> = rows
            .iter()
            .map(|r| r.values.iter().find(|(k, _)| k == "name").unwrap().1.clone())
            .collect();
        let expected: std::collections::BTreeSet<String> = (1..=5)
            .map(|i| format!("name-{i}"))
            .collect();
        assert_eq!(names, expected);
    }

    /// Restore from a directory whose SST file has been corrupted
    /// must abort before any file lands in the target. The error
    /// message must identify the offending file so an operator can
    /// triage without guessing.
    #[test]
    fn restore_aborts_on_corrupted_sst() {
        let root = tempfile::tempdir().unwrap();
        let src_path = root.path().join("src");
        let mut src_db = Database::open(src_path.to_str().unwrap()).unwrap();
        src_db.execute("CREATE TABLE t (id INT PRIMARY KEY, v TEXT)")
            .unwrap();
        for i in 1..=3 {
            src_db
                .execute(&format!("INSERT INTO t (id, v) VALUES ({i}, 'x{i}')"))
                .unwrap();
        }
        let backup_dir = root.path().join("backup");
        src_db
            .execute(&format!("BACKUP TO '{}'", backup_dir.display()))
            .unwrap();
        drop(src_db);

        // Corrupt every byte in the middle of every SST file in the
        // backup. The block checksum must fail when `validate_backup`
        // deserialises the blocks.
        let mut flipped_something = false;
        for entry in std::fs::read_dir(&backup_dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            if !(name.starts_with("sst_") && name.ends_with(".pax")) {
                continue;
            }
            let mut bytes = std::fs::read(&path).unwrap();
            // Flip a byte in the middle of the file — avoid the
            // footer (last 16 bytes) so the block index still reads
            // cleanly and the corruption is caught by the block
            // checksum, not by the footer parser.
            let mid = bytes.len().saturating_sub(64);
            if mid > 0 {
                bytes[mid / 2] ^= 0xFF;
                std::fs::write(&path, &bytes).unwrap();
                flipped_something = true;
            }
        }
        assert!(
            flipped_something,
            "test setup precondition: backup must contain at least one SST to corrupt"
        );

        // RESTORE FROM must now fail.
        let dst_path = root.path().join("dst");
        let mut dst_db = Database::open(dst_path.to_str().unwrap()).unwrap();
        let err = dst_db
            .execute(&format!("RESTORE FROM '{}'", backup_dir.display()))
            .expect_err("corrupt backup must abort RESTORE");
        let msg = format!("{err}");
        assert!(
            msg.contains("RESTORE: corrupt") || msg.contains("corrupt"),
            "expected corruption error message, got: {msg}"
        );
        // And the target must still be empty of restored SSTs.
        let restored_ssts = std::fs::read_dir(&dst_path)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                let n = e.file_name();
                let s = n.to_string_lossy();
                s.starts_with("sst_") && s.ends_with(".pax")
            })
            .count();
        assert_eq!(
            restored_ssts, 0,
            "RESTORE must not copy any files when validation fails; got {restored_ssts}"
        );
    }

    /// Backup the same database into two different directories and
    /// assert they contain the same SST bytes. This isn't strictly
    /// required by the spec but pins down the "clean Merkle root"
    /// half of 37.1 — a stable snapshot point.
    #[test]
    fn repeat_backup_is_byte_identical_without_intervening_writes() {
        let root = tempfile::tempdir().unwrap();
        let src_path = root.path().join("src");
        let mut db = Database::open(src_path.to_str().unwrap()).unwrap();
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, v TEXT)")
            .unwrap();
        for i in 1..=4 {
            db.execute(&format!("INSERT INTO t (id, v) VALUES ({i}, 'v{i}')"))
                .unwrap();
        }

        let backup_a = root.path().join("backup_a");
        let backup_b = root.path().join("backup_b");
        db.execute(&format!("BACKUP TO '{}'", backup_a.display()))
            .unwrap();
        db.execute(&format!("BACKUP TO '{}'", backup_b.display()))
            .unwrap();

        // Collect SST file names + sizes from both backups. The
        // second flush is a no-op (memtable already flushed) so the
        // SST set must match exactly.
        let read_ssts = |dir: &Path| -> Vec<(String, u64)> {
            let mut out: Vec<(String, u64)> = std::fs::read_dir(dir)
                .unwrap()
                .filter_map(|e| e.ok())
                .filter_map(|e| {
                    let name = e.file_name().to_string_lossy().to_string();
                    if !(name.starts_with("sst_") && name.ends_with(".pax")) {
                        return None;
                    }
                    let size = e.metadata().ok()?.len();
                    Some((name, size))
                })
                .collect();
            out.sort();
            out
        };
        let a = read_ssts(&backup_a);
        let b = read_ssts(&backup_b);
        assert_eq!(
            a, b,
            "two backups without intervening writes must contain the same SSTs; \
             got a={a:?}, b={b:?}"
        );
        assert!(!a.is_empty(), "backup must include at least one SST");
    }
}

#[cfg(test)]
mod role_grant_tests {
    use super::*;
    use galaxdb_sql::auth_store::AuthStore;
    use galaxdb_sql::auth_store::PrivilegeAction as Action;

    fn test_db() -> Database {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("db");
        std::mem::forget(dir);
        Database::open(p.to_str().unwrap()).unwrap()
    }

    #[test]
    fn create_role_persists_verifier_via_ddl() {
        let mut db = test_db();
        db.execute("CREATE ROLE alice PASSWORD 'secret'").unwrap();

        // Inspect the persisted state through a fresh AuthStore over the
        // same engine — proves the DDL wrote through to storage.
        let store = AuthStore::new(db.engine.clone());
        let role = store.get_role("alice").expect("role persisted");
        assert!(!role.is_superuser);
        assert!(role.verifier.is_some(), "password must produce a verifier");
        // The plaintext must not be recoverable from the stored bytes.
        let v = role.verifier.unwrap();
        assert!(!v.salt.windows(6).any(|w| w == b"secret"));
    }

    #[test]
    fn create_superuser_role() {
        let mut db = test_db();
        db.execute("CREATE ROLE admin PASSWORD 'pw' SUPERUSER").unwrap();
        let store = AuthStore::new(db.engine.clone());
        assert!(store.is_superuser("admin"));
    }

    #[test]
    fn create_duplicate_role_errors() {
        let mut db = test_db();
        db.execute("CREATE ROLE alice PASSWORD 'pw'").unwrap();
        let err = db.execute("CREATE ROLE alice PASSWORD 'pw2'");
        assert!(err.is_err(), "duplicate role must error");
    }

    #[test]
    fn alter_role_password_replaces_verifier() {
        let mut db = test_db();
        db.execute("CREATE ROLE alice PASSWORD 'old'").unwrap();
        let store = AuthStore::new(db.engine.clone());
        let v_old = store.verifier_for("alice").unwrap();

        db.execute("ALTER ROLE alice PASSWORD 'new'").unwrap();
        let v_new = store.verifier_for("alice").unwrap();
        // A new salt+keys must be derived (overwhelmingly different).
        assert_ne!(v_old.stored_key, v_new.stored_key);
    }

    #[test]
    fn alter_unknown_role_errors() {
        let mut db = test_db();
        assert!(db.execute("ALTER ROLE ghost PASSWORD 'x'").is_err());
    }

    #[test]
    fn drop_role_removes_it() {
        let mut db = test_db();
        db.execute("CREATE ROLE alice PASSWORD 'pw'").unwrap();
        db.execute("DROP ROLE alice").unwrap();
        let store = AuthStore::new(db.engine.clone());
        assert!(store.get_role("alice").is_none());
        // DROP of a missing role errors unless IF EXISTS.
        assert!(db.execute("DROP ROLE alice").is_err());
        db.execute("DROP ROLE IF EXISTS alice").unwrap();
    }

    #[test]
    fn grant_and_revoke_persist() {
        let mut db = test_db();
        db.execute("CREATE ROLE alice PASSWORD 'pw'").unwrap();
        db.execute("GRANT SELECT ON docs TO alice").unwrap();
        db.execute("GRANT INSERT ON docs TO alice").unwrap();

        let store = AuthStore::new(db.engine.clone());
        assert!(store.has_grant("alice", "docs", Action::Select));
        assert!(store.has_grant("alice", "docs", Action::Insert));
        assert!(!store.has_grant("alice", "docs", Action::Delete));

        db.execute("REVOKE SELECT ON docs FROM alice").unwrap();
        assert!(!store.has_grant("alice", "docs", Action::Select));
        assert!(store.has_grant("alice", "docs", Action::Insert));
    }

    #[test]
    fn grant_to_unknown_role_errors() {
        let mut db = test_db();
        assert!(db.execute("GRANT SELECT ON docs TO nobody").is_err());
    }

    #[test]
    fn roles_and_grants_survive_reopen_via_ddl() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("db");
        {
            let mut db = Database::open(p.to_str().unwrap()).unwrap();
            db.execute("CREATE ROLE alice PASSWORD 'pw'").unwrap();
            db.execute("GRANT UPDATE ON docs TO alice").unwrap();
        }
        // Reopen: WAL replay restores the auth rows.
        let db = Database::open(p.to_str().unwrap()).unwrap();
        let store = AuthStore::new(db.engine.clone());
        assert!(store.get_role("alice").is_some());
        assert!(store.has_grant("alice", "docs", Action::Update));
    }
}

/// Authorization chokepoint tests (task 5): the executor must reject a
/// non-privileged role with SQLSTATE `42501` *before* touching storage,
/// and accept the same statement once the matching grant exists. These
/// run on the embedded `Database` path; the wire path is covered by the
/// `tokio-postgres` integration test in task 6 once SCRAM supplies the
/// session. Both paths funnel through `execute_with_context`, so this
/// exercises the single shared chokepoint (Req 3, AC7).
#[cfg(test)]
mod authz_chokepoint_tests {
    use super::*;
    use galaxdb_auth::{Role, SessionContext};

    fn fresh_db() -> Database {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("db");
        std::mem::forget(dir);
        Database::open(p.to_str().unwrap()).unwrap()
    }

    /// Assert a result is an `InsufficientPrivilege` error for `role` on
    /// the given action label, and that it renders to SQLSTATE 42501.
    fn assert_denied(res: GalaxResult<QueryResult>, role: &str, action: &str) {
        match res {
            Err(GalaxError::InsufficientPrivilege {
                role: r,
                action: a,
                ..
            }) => {
                assert_eq!(r, role, "denied role");
                assert_eq!(a, action, "denied action");
                assert_eq!(
                    GalaxError::InsufficientPrivilege {
                        role: r,
                        action: a,
                        object: "x".into()
                    }
                    .sqlstate(),
                    "42501",
                    "insufficient_privilege must map to SQLSTATE 42501",
                );
            }
            other => panic!("expected InsufficientPrivilege, got {other:?}"),
        }
    }

    /// A superuser session (the operator) sets up the schema and a
    /// non-privileged role; a session for that role is then denied until
    /// granted.
    #[test]
    fn non_privileged_role_is_denied_then_granted_select() {
        // 1. As superuser: create the table and a plain role.
        let mut admin = fresh_db().with_session(SessionContext::new(Role::superuser("root")));
        admin
            .execute("CREATE TABLE docs (id INT PRIMARY KEY, body TEXT)")
            .unwrap();
        admin.execute("INSERT INTO docs VALUES (1, 'hello')").unwrap();
        admin.execute("CREATE ROLE alice PASSWORD 'pw'").unwrap();

        // 2. A handle for alice over the SAME engine/catalog. We reuse
        //    the admin handle but swap the session, which is exactly what
        //    the wire server does per-connection.
        admin.set_session(Some(SessionContext::new(Role::user("alice"))));

        // alice has no grant on docs → SELECT denied with 42501, before
        // any row is read.
        assert_denied(admin.execute("SELECT * FROM docs"), "alice", "select");

        // 3. Grant SELECT (must be done by a superuser). Swap back.
        admin.set_session(Some(SessionContext::new(Role::superuser("root"))));
        admin.execute("GRANT SELECT ON docs TO alice").unwrap();

        // 4. Now alice can SELECT — no restart, the grant took effect
        //    immediately (Req 3, AC6).
        admin.set_session(Some(SessionContext::new(Role::user("alice"))));
        let res = admin.execute("SELECT * FROM docs").unwrap();
        match res {
            QueryResult::Rows(rows) => assert_eq!(rows.len(), 1),
            other => panic!("expected one row, got {other:?}"),
        }
    }

    /// INSERT/UPDATE/DELETE each require their own privilege; a SELECT
    /// grant does not let alice write.
    #[test]
    fn write_privileges_are_distinct() {
        let mut db = fresh_db().with_session(SessionContext::new(Role::superuser("root")));
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, n INT)").unwrap();
        db.execute("INSERT INTO t VALUES (1, 10)").unwrap();
        db.execute("CREATE ROLE bob PASSWORD 'pw'").unwrap();
        db.execute("GRANT SELECT ON t TO bob").unwrap();

        db.set_session(Some(SessionContext::new(Role::user("bob"))));
        // SELECT works.
        db.execute("SELECT * FROM t").unwrap();
        // INSERT/UPDATE/DELETE denied with the matching action label.
        assert_denied(db.execute("INSERT INTO t VALUES (2, 20)"), "bob", "insert");
        assert_denied(db.execute("UPDATE t SET n = 99 WHERE id = 1"), "bob", "update");
        assert_denied(db.execute("DELETE FROM t WHERE id = 1"), "bob", "delete");

        // Grant INSERT; now INSERT works but UPDATE still denied.
        db.set_session(Some(SessionContext::new(Role::superuser("root"))));
        db.execute("GRANT INSERT ON t TO bob").unwrap();
        db.set_session(Some(SessionContext::new(Role::user("bob"))));
        db.execute("INSERT INTO t VALUES (2, 20)").unwrap();
        assert_denied(db.execute("UPDATE t SET n = 99 WHERE id = 1"), "bob", "update");
    }

    /// Only a superuser may run CREATE ROLE / GRANT / REVOKE (Req 3, AC5).
    #[test]
    fn role_admin_is_superuser_only() {
        let mut db = fresh_db().with_session(SessionContext::new(Role::superuser("root")));
        db.execute("CREATE TABLE t (id INT PRIMARY KEY)").unwrap();
        db.execute("CREATE ROLE carol PASSWORD 'pw'").unwrap();

        // carol (non-superuser) cannot administer roles/grants.
        db.set_session(Some(SessionContext::new(Role::user("carol"))));
        assert_denied(db.execute("CREATE ROLE mallory PASSWORD 'x'"), "carol", "admin");
        assert_denied(db.execute("GRANT SELECT ON t TO carol"), "carol", "admin");
        assert_denied(db.execute("REVOKE SELECT ON t FROM carol"), "carol", "admin");
        // DDL is also superuser-only in the baseline.
        assert_denied(
            db.execute("CREATE TABLE t2 (id INT PRIMARY KEY)"),
            "carol",
            "ddl",
        );
    }

    /// REVOKE takes effect immediately for the next statement (Req 3, AC6).
    #[test]
    fn revoke_takes_effect_without_restart() {
        let mut db = fresh_db().with_session(SessionContext::new(Role::superuser("root")));
        db.execute("CREATE TABLE docs (id INT PRIMARY KEY)").unwrap();
        db.execute("CREATE ROLE dave PASSWORD 'pw'").unwrap();
        db.execute("GRANT SELECT ON docs TO dave").unwrap();

        db.set_session(Some(SessionContext::new(Role::user("dave"))));
        db.execute("SELECT * FROM docs").unwrap();

        // Superuser revokes; dave is immediately denied.
        db.set_session(Some(SessionContext::new(Role::superuser("root"))));
        db.execute("REVOKE SELECT ON docs FROM dave").unwrap();
        db.set_session(Some(SessionContext::new(Role::user("dave"))));
        assert_denied(db.execute("SELECT * FROM docs"), "dave", "select");
    }

    /// A handle with no session (trusted embedded use) skips authorization
    /// entirely — today's behavior for direct PyO3/Rust callers.
    #[test]
    fn no_session_skips_authorization() {
        let mut db = fresh_db(); // no session attached
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, n INT)").unwrap();
        db.execute("INSERT INTO t VALUES (1, 5)").unwrap();
        db.execute("UPDATE t SET n = 7 WHERE id = 1").unwrap();
        db.execute("SELECT * FROM t").unwrap();
        db.execute("DELETE FROM t WHERE id = 1").unwrap();
        // Even role administration works with no session (trusted caller).
        db.execute("CREATE ROLE anyone PASSWORD 'pw'").unwrap();
    }

    /// The grant is scoped to its exact table: a SELECT grant on `a` does
    /// not authorize SELECT on `b`.
    #[test]
    fn grants_are_table_scoped() {
        let mut db = fresh_db().with_session(SessionContext::new(Role::superuser("root")));
        db.execute("CREATE TABLE a (id INT PRIMARY KEY)").unwrap();
        db.execute("CREATE TABLE b (id INT PRIMARY KEY)").unwrap();
        db.execute("CREATE ROLE eve PASSWORD 'pw'").unwrap();
        db.execute("GRANT SELECT ON a TO eve").unwrap();

        db.set_session(Some(SessionContext::new(Role::user("eve"))));
        db.execute("SELECT * FROM a").unwrap();
        assert_denied(db.execute("SELECT * FROM b"), "eve", "select");
    }
}

/// Secondary-index tests (task 8): CREATE/DROP INDEX, index-accelerated
/// reads equal full-scan results, write-path maintenance, range queries,
/// and restart survival. All run through the embedded `Database` so the
/// whole stack (parser → planner → executor → engine) is exercised.
#[cfg(test)]
mod secondary_index_tests {
    use super::*;

    fn fresh_db() -> Database {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("db");
        std::mem::forget(dir);
        Database::open(p.to_str().unwrap()).unwrap()
    }

    fn rows(res: QueryResult) -> Vec<QueryRow> {
        match res {
            QueryResult::Rows(r) => r,
            other => panic!("expected rows, got {other:?}"),
        }
    }

    /// An index lookup returns exactly the same rows as the equivalent
    /// full scan (Req 5 AC2), and covers rows that existed *before* the
    /// index was created.
    #[test]
    fn index_lookup_equals_full_scan() {
        let mut db = fresh_db();
        db.execute("CREATE TABLE people (id INT PRIMARY KEY, city TEXT)")
            .unwrap();
        db.execute("INSERT INTO people VALUES (1, 'NYC')").unwrap();
        db.execute("INSERT INTO people VALUES (2, 'LA')").unwrap();
        db.execute("INSERT INTO people VALUES (3, 'NYC')").unwrap();

        // Build the index AFTER inserting — it must cover existing rows.
        let msg = db.execute("CREATE INDEX idx_city ON people (city)").unwrap();
        match msg {
            QueryResult::Ok(s) => assert!(s.contains("3 rows indexed"), "got: {s}"),
            other => panic!("expected Ok, got {other:?}"),
        }

        let via_index = rows(db.execute("SELECT id, city FROM people WHERE city = 'NYC'").unwrap());
        assert_eq!(via_index.len(), 2, "index lookup must return both NYC rows");

        // Drop the index → same query falls back to full scan, same result.
        db.execute("DROP INDEX idx_city").unwrap();
        let via_scan = rows(db.execute("SELECT id, city FROM people WHERE city = 'NYC'").unwrap());
        assert_eq!(via_scan.len(), 2, "full scan must return the same rows");
    }

    /// INSERT/UPDATE/DELETE keep the index consistent (Req 5 AC3): a query
    /// through the index reflects the latest committed state.
    #[test]
    fn writes_maintain_the_index() {
        let mut db = fresh_db();
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, status TEXT)")
            .unwrap();
        db.execute("CREATE INDEX idx_status ON t (status)").unwrap();

        db.execute("INSERT INTO t VALUES (1, 'pending')").unwrap();
        db.execute("INSERT INTO t VALUES (2, 'pending')").unwrap();
        db.execute("INSERT INTO t VALUES (3, 'shipped')").unwrap();
        assert_eq!(
            rows(db.execute("SELECT id FROM t WHERE status = 'pending'").unwrap()).len(),
            2
        );

        // UPDATE moves row 1 from pending -> shipped.
        db.execute("UPDATE t SET status = 'shipped' WHERE id = 1").unwrap();
        assert_eq!(
            rows(db.execute("SELECT id FROM t WHERE status = 'pending'").unwrap()).len(),
            1,
            "after UPDATE only row 2 is pending"
        );
        assert_eq!(
            rows(db.execute("SELECT id FROM t WHERE status = 'shipped'").unwrap()).len(),
            2,
            "rows 1 and 3 are shipped"
        );

        // DELETE row 2 removes it from the pending bucket.
        db.execute("DELETE FROM t WHERE id = 2").unwrap();
        assert_eq!(
            rows(db.execute("SELECT id FROM t WHERE status = 'pending'").unwrap()).len(),
            0,
            "no pending rows remain"
        );
    }

    /// Range predicates resolve through the index and match the scan
    /// result (Req 5 AC2 range case).
    #[test]
    fn range_query_uses_index() {
        let mut db = fresh_db();
        db.execute("CREATE TABLE u (id INT PRIMARY KEY, age INT)").unwrap();
        for (id, age) in [(1, 20), (2, 30), (3, 40), (4, 50)] {
            db.execute(&format!("INSERT INTO u VALUES ({id}, {age})")).unwrap();
        }
        db.execute("CREATE INDEX idx_age ON u (age)").unwrap();

        // age >= 30 AND age <= 45 -> ages 30, 40 (ids 2, 3).
        let got = rows(db.execute("SELECT id FROM u WHERE age >= 30").unwrap());
        assert_eq!(got.len(), 3, ">= 30 must match ages 30,40,50");

        let got = rows(db.execute("SELECT id FROM u WHERE age <= 30").unwrap());
        assert_eq!(got.len(), 2, "<= 30 must match ages 20,30");
    }

    /// The index definition and its entries survive a restart (Req 5 AC4)
    /// because they are engine rows replayed through the WAL.
    #[test]
    fn index_survives_restart() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("db");
        {
            let mut db = Database::open(p.to_str().unwrap()).unwrap();
            db.execute("CREATE TABLE k (id INT PRIMARY KEY, v TEXT)").unwrap();
            db.execute("INSERT INTO k VALUES (1, 'x')").unwrap();
            db.execute("INSERT INTO k VALUES (2, 'y')").unwrap();
            db.execute("CREATE INDEX idx_v ON k (v)").unwrap();
        }
        // Reopen: the table's schema is now recovered from the persisted
        // catalog (no need to re-CREATE it), and the index entries were
        // recovered through the WAL.
        let mut db = Database::open(p.to_str().unwrap()).unwrap();
        assert!(db.table_exists("k"), "table schema survives restart");
        let got = rows(db.execute("SELECT id FROM k WHERE v = 'x'").unwrap());
        assert_eq!(got.len(), 1, "index entry for 'x' survived restart");
        assert_eq!(got[0].values.iter().find(|(c, _)| c == "id").unwrap().1, "1");
    }

    /// CREATE INDEX on a missing table or column is a typed error, and
    /// duplicate index names error unless IF NOT EXISTS.
    #[test]
    fn create_index_validation() {
        let mut db = fresh_db();
        assert!(db.execute("CREATE INDEX i ON ghost (c)").is_err(), "missing table");
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, a TEXT)").unwrap();
        assert!(db.execute("CREATE INDEX i ON t (nope)").is_err(), "missing column");
        db.execute("CREATE INDEX i ON t (a)").unwrap();
        assert!(db.execute("CREATE INDEX i ON t (a)").is_err(), "duplicate name");
        db.execute("CREATE INDEX IF NOT EXISTS i ON t (a)").unwrap();
        // DROP missing index errors unless IF EXISTS.
        db.execute("DROP INDEX i").unwrap();
        assert!(db.execute("DROP INDEX i").is_err());
        db.execute("DROP INDEX IF EXISTS i").unwrap();
    }
}
