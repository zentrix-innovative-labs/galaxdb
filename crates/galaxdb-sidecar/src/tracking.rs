//! Model-version tracking and embedding staleness (Reqs 20, 39).
//!
//! Each row with an embedding column carries two system metadata fields:
//! - `_embedding_model_version: String` — model version that produced the embedding
//! - `_embedding_stale: bool` — true when embedding is pending or outdated
//!
//! These are written through the standard LSM update path (same WAL, same MVCC),
//! ensuring the flag and embedding value are always consistent from a reader's
//! perspective.
//!
//! Lifecycle:
//! 1. INSERT with embedding column → `_embedding_stale = true`, `_embedding_model_version = ""`
//! 2. Sidecar generates embedding → `_embedding_stale = false`, `_embedding_model_version = "v1.0"`
//! 3. Model version changes → `_embedding_stale = true` on all rows with old version
//! 4. Re-embedding completes → `_embedding_stale = false`, `_embedding_model_version = "v2.0"`

use std::collections::HashMap;
use std::sync::RwLock;

/// Embedding metadata for a single row.
#[derive(Debug, Clone)]
pub struct EmbeddingMeta {
    /// Model version that produced the current embedding.
    pub model_version: String,
    /// Whether the embedding is stale (pending generation or model changed).
    pub stale: bool,
}

impl EmbeddingMeta {
    /// Create metadata for a newly inserted row (stale, no version yet).
    pub fn new_pending() -> Self {
        Self {
            model_version: String::new(),
            stale: true,
        }
    }

    /// Create metadata after embedding generation completes.
    pub fn completed(model_version: String) -> Self {
        Self {
            model_version,
            stale: false,
        }
    }

    /// Mark as stale (model version changed).
    pub fn mark_stale(&mut self) {
        self.stale = true;
    }
}

/// Tracks embedding metadata for all rows across all tables.
///
/// This is the in-memory representation. The actual persistence goes through
/// the standard LSM path (WAL + memtable + PAX blocks) via system columns.
pub struct EmbeddingTracker {
    /// Per-table, per-row embedding metadata.
    /// Key: (table_name, row_id) → EmbeddingMeta
    inner: RwLock<HashMap<(String, u64), EmbeddingMeta>>,
    /// Current model version from the sidecar.
    current_model_version: RwLock<String>,
}

impl EmbeddingTracker {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(HashMap::new()),
            current_model_version: RwLock::new(String::new()),
        }
    }

    /// Record that a row was inserted with an embedding column.
    /// Sets `_embedding_stale = true` until the sidecar generates the embedding.
    pub fn on_insert(&self, table: &str, row_id: u64) {
        let mut inner = self.inner.write().unwrap();
        inner.insert(
            (table.to_string(), row_id),
            EmbeddingMeta::new_pending(),
        );
    }

    /// Record that an embedding was generated for a row.
    /// Sets `_embedding_stale = false` and records the model version.
    pub fn on_embedding_complete(&self, table: &str, row_id: u64, model_version: &str) {
        let mut inner = self.inner.write().unwrap();
        inner.insert(
            (table.to_string(), row_id),
            EmbeddingMeta::completed(model_version.to_string()),
        );
    }

    /// Record that the sidecar model version has changed.
    /// Marks all rows with the old version as stale and returns the count.
    pub fn on_model_version_change(&self, new_version: &str) -> usize {
        let old_version = {
            let mut ver = self.current_model_version.write().unwrap();
            let old = ver.clone();
            *ver = new_version.to_string();
            old
        };

        if old_version.is_empty() || old_version == new_version {
            return 0;
        }

        let mut inner = self.inner.write().unwrap();
        let mut stale_count = 0;

        for meta in inner.values_mut() {
            if meta.model_version == old_version && !meta.stale {
                meta.mark_stale();
                stale_count += 1;
            }
        }

        stale_count
    }

    /// Get the embedding metadata for a row.
    pub fn get_meta(&self, table: &str, row_id: u64) -> Option<EmbeddingMeta> {
        let inner = self.inner.read().unwrap();
        inner.get(&(table.to_string(), row_id)).cloned()
    }

    /// Check if a row's embedding is stale.
    pub fn is_stale(&self, table: &str, row_id: u64) -> bool {
        self.get_meta(table, row_id).map_or(false, |m| m.stale)
    }

    /// Get all stale row IDs for a table (for re-embedding queue).
    pub fn stale_rows(&self, table: &str) -> Vec<u64> {
        let inner = self.inner.read().unwrap();
        inner.iter()
            .filter(|((t, _), meta)| t == table && meta.stale)
            .map(|((_, row_id), _)| *row_id)
            .collect()
    }

    /// Get the current model version.
    pub fn current_model_version(&self) -> String {
        self.current_model_version.read().unwrap().clone()
    }

    /// Generate a health report for SHOW EMBEDDING HEALTH.
    ///
    /// Returns: (total_rows, stale_count, version_distribution)
    pub fn health_report(&self, table: Option<&str>) -> EmbeddingHealthReport {
        let inner = self.inner.read().unwrap();

        let mut total = 0usize;
        let mut stale = 0usize;
        let mut versions: HashMap<String, usize> = HashMap::new();

        for ((t, _), meta) in inner.iter() {
            if let Some(filter_table) = table {
                if t != filter_table {
                    continue;
                }
            }
            total += 1;
            if meta.stale {
                stale += 1;
            }
            let ver = if meta.model_version.is_empty() {
                "pending".to_string()
            } else {
                meta.model_version.clone()
            };
            *versions.entry(ver).or_insert(0) += 1;
        }

        EmbeddingHealthReport {
            total_rows: total,
            stale_count: stale,
            fresh_count: total - stale,
            version_distribution: versions,
            current_model_version: self.current_model_version(),
        }
    }

    /// Remove tracking for a deleted row.
    pub fn on_delete(&self, table: &str, row_id: u64) {
        let mut inner = self.inner.write().unwrap();
        inner.remove(&(table.to_string(), row_id));
    }

    /// Total tracked rows.
    pub fn total_tracked(&self) -> usize {
        self.inner.read().unwrap().len()
    }
}

/// Report from SHOW EMBEDDING HEALTH.
#[derive(Debug, Clone)]
pub struct EmbeddingHealthReport {
    pub total_rows: usize,
    pub stale_count: usize,
    pub fresh_count: usize,
    pub version_distribution: HashMap<String, usize>,
    pub current_model_version: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_marks_stale() {
        let tracker = EmbeddingTracker::new();
        tracker.on_insert("docs", 1);

        assert!(tracker.is_stale("docs", 1));
        let meta = tracker.get_meta("docs", 1).unwrap();
        assert!(meta.stale);
        assert!(meta.model_version.is_empty());
    }

    #[test]
    fn embedding_complete_clears_stale() {
        let tracker = EmbeddingTracker::new();
        tracker.on_insert("docs", 1);
        assert!(tracker.is_stale("docs", 1));

        tracker.on_embedding_complete("docs", 1, "v1.0");
        assert!(!tracker.is_stale("docs", 1));

        let meta = tracker.get_meta("docs", 1).unwrap();
        assert_eq!(meta.model_version, "v1.0");
        assert!(!meta.stale);
    }

    #[test]
    fn model_version_change_marks_stale() {
        let tracker = EmbeddingTracker::new();

        // Set initial version
        tracker.on_model_version_change("v1.0");

        // Insert and complete some rows
        tracker.on_insert("docs", 1);
        tracker.on_embedding_complete("docs", 1, "v1.0");
        tracker.on_insert("docs", 2);
        tracker.on_embedding_complete("docs", 2, "v1.0");
        tracker.on_insert("docs", 3);
        // Row 3 is still pending (stale)

        assert!(!tracker.is_stale("docs", 1));
        assert!(!tracker.is_stale("docs", 2));
        assert!(tracker.is_stale("docs", 3));

        // Model version changes to v2.0
        let stale_count = tracker.on_model_version_change("v2.0");
        assert_eq!(stale_count, 2); // rows 1 and 2 marked stale (row 3 was already stale)

        assert!(tracker.is_stale("docs", 1));
        assert!(tracker.is_stale("docs", 2));
        assert!(tracker.is_stale("docs", 3));
    }

    #[test]
    fn stale_rows_returns_correct_ids() {
        let tracker = EmbeddingTracker::new();
        tracker.on_insert("docs", 1);
        tracker.on_insert("docs", 2);
        tracker.on_embedding_complete("docs", 2, "v1.0");
        tracker.on_insert("docs", 3);

        let stale = tracker.stale_rows("docs");
        assert_eq!(stale.len(), 2); // rows 1 and 3
        assert!(stale.contains(&1));
        assert!(stale.contains(&3));
        assert!(!stale.contains(&2));
    }

    #[test]
    fn health_report_correct() {
        let tracker = EmbeddingTracker::new();
        tracker.on_model_version_change("v1.0");

        tracker.on_insert("docs", 1);
        tracker.on_embedding_complete("docs", 1, "v1.0");
        tracker.on_insert("docs", 2);
        tracker.on_embedding_complete("docs", 2, "v1.0");
        tracker.on_insert("docs", 3); // pending

        let report = tracker.health_report(Some("docs"));
        assert_eq!(report.total_rows, 3);
        assert_eq!(report.stale_count, 1); // row 3
        assert_eq!(report.fresh_count, 2);
        assert_eq!(report.current_model_version, "v1.0");
        assert_eq!(*report.version_distribution.get("v1.0").unwrap_or(&0), 2);
        assert_eq!(*report.version_distribution.get("pending").unwrap_or(&0), 1);
    }

    #[test]
    fn health_report_after_model_change() {
        let tracker = EmbeddingTracker::new();
        tracker.on_model_version_change("v1.0");

        tracker.on_insert("docs", 1);
        tracker.on_embedding_complete("docs", 1, "v1.0");
        tracker.on_insert("docs", 2);
        tracker.on_embedding_complete("docs", 2, "v1.0");

        // Model changes
        tracker.on_model_version_change("v2.0");

        // Re-embed row 1
        tracker.on_embedding_complete("docs", 1, "v2.0");

        let report = tracker.health_report(Some("docs"));
        assert_eq!(report.total_rows, 2);
        assert_eq!(report.stale_count, 1); // row 2 still stale
        assert_eq!(report.fresh_count, 1);
        assert_eq!(report.current_model_version, "v2.0");
        assert_eq!(*report.version_distribution.get("v2.0").unwrap_or(&0), 1);
        assert_eq!(*report.version_distribution.get("v1.0").unwrap_or(&0), 1);
    }

    #[test]
    fn delete_removes_tracking() {
        let tracker = EmbeddingTracker::new();
        tracker.on_insert("docs", 1);
        assert_eq!(tracker.total_tracked(), 1);

        tracker.on_delete("docs", 1);
        assert_eq!(tracker.total_tracked(), 0);
        assert!(tracker.get_meta("docs", 1).is_none());
    }

    #[test]
    fn health_report_filters_by_table() {
        let tracker = EmbeddingTracker::new();
        tracker.on_insert("docs", 1);
        tracker.on_insert("images", 2);

        let docs_report = tracker.health_report(Some("docs"));
        assert_eq!(docs_report.total_rows, 1);

        let all_report = tracker.health_report(None);
        assert_eq!(all_report.total_rows, 2);
    }

    #[test]
    fn same_version_change_is_noop() {
        let tracker = EmbeddingTracker::new();
        tracker.on_model_version_change("v1.0");
        tracker.on_insert("docs", 1);
        tracker.on_embedding_complete("docs", 1, "v1.0");

        // Same version — should not mark anything stale
        let stale_count = tracker.on_model_version_change("v1.0");
        assert_eq!(stale_count, 0);
        assert!(!tracker.is_stale("docs", 1));
    }
}
