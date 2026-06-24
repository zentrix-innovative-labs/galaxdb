//! Real-scenario integration tests for the engine runtime compaction driver.
//!
//! These drive a real [`Engine`] against a real on-disk data directory (no
//! mocks): they write through the WAL + memtable, flush to real SST files,
//! and let the flush-triggered compaction merge them. The assertions cover
//! the four properties that make compaction a correct, useful feature:
//!
//! 1. Under sustained updates the on-disk SST count reaches a bounded
//!    steady state instead of growing with every flush.
//! 2. Point reads return the latest value of every live key after merges.
//! 3. Deleted keys stay deleted (tombstone GC at the bottom level).
//! 4. Version tags pin historical versions so `AT VERSION` time-travel
//!    still resolves across a compaction (MVCC GC honors the pin set).

use std::sync::Arc;

use galaxdb_common::Timestamp;
use galaxdb_storage::engine::{Engine, EngineConfig, PinSource};

fn small_config(dir: &std::path::Path) -> EngineConfig {
    EngineConfig {
        data_dir: dir.to_path_buf(),
        // Large memtable so flushes happen only when we ask, giving the
        // test deterministic control over how many SSTs exist.
        memtable_size_bytes: 1 << 30,
        back_pressure_bytes: 1 << 31,
        wal_group_commit_ms: 1,
        l0_compaction_trigger: 4,
        compaction_concurrency: 2,
        ..Default::default()
    }
}

/// A fixed pin set, standing in for the version-tag catalog the embedded
/// `Database` layer owns in production.
struct FixedPins(Vec<Timestamp>);
impl PinSource for FixedPins {
    fn pinned_timestamps(&self) -> Vec<Timestamp> {
        self.0.clone()
    }
}

#[tokio::test]
async fn compaction_bounds_sst_count_under_repeated_updates() {
    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::new(small_config(dir.path())).unwrap();

    let n_keys = 50usize;
    let rounds = 24usize;

    // Each round rewrites the SAME key set and flushes. Without compaction
    // this would leave ~24 SST files (one per flush). With flush-triggered
    // compaction + MVCC GC (no pins) only the latest version of each key
    // survives, so the file count must stay bounded near the L0 trigger.
    for round in 0..rounds {
        for k in 0..n_keys {
            let key = format!("k{k:04}").into_bytes();
            let val = format!("v{k}-r{round}").into_bytes();
            engine.put(key, val).await.unwrap();
        }
        engine.flush_memtable().await.unwrap();
    }

    let final_count = engine.sst_count();
    assert!(
        final_count <= 5,
        "SST count should reach a bounded steady state, got {final_count} after {rounds} flushes"
    );
    assert!(
        final_count < rounds,
        "compaction must collapse the {rounds} per-flush SSTs (got {final_count})"
    );

    // Every key must read back its latest value through the point-read path
    // (ART -> relocated SST location).
    for k in 0..n_keys {
        let key = format!("k{k:04}").into_bytes();
        let expected = format!("v{k}-r{}", rounds - 1).into_bytes();
        assert_eq!(
            engine.get(&key),
            Some(expected),
            "key k{k:04} must return its latest value after compaction"
        );
    }
}

#[tokio::test]
async fn compaction_preserves_latest_values_and_reclaims_deletes() {
    let dir = tempfile::tempdir().unwrap();
    // High trigger so auto-compaction never fires; we call compact() to test
    // the explicit path deterministically.
    let mut cfg = small_config(dir.path());
    cfg.l0_compaction_trigger = 10_000;
    let engine = Engine::new(cfg).unwrap();

    // Round 1: insert k0..k10 = v1, flush.
    for k in 0..10 {
        engine
            .put(format!("k{k}").into_bytes(), format!("v1-{k}").into_bytes())
            .await
            .unwrap();
    }
    engine.flush_memtable().await.unwrap();

    // Round 2: update k0..k4 = v2, delete k5, flush.
    for k in 0..5 {
        engine
            .put(format!("k{k}").into_bytes(), format!("v2-{k}").into_bytes())
            .await
            .unwrap();
    }
    engine.delete(b"k5").await.unwrap();
    engine.flush_memtable().await.unwrap();

    assert!(engine.sst_count() >= 2, "should have multiple SSTs pre-compaction");

    let stats = engine.compact().unwrap();
    assert!(stats.input_ssts >= 2);
    assert!(stats.output_ssts >= 1);
    assert!(
        stats.versions_gc_dropped > 0,
        "superseded versions / deleted key should be GC'd"
    );

    // Latest values for updated keys.
    for k in 0..5 {
        assert_eq!(
            engine.get(format!("k{k}").into_bytes().as_slice()),
            Some(format!("v2-{k}").into_bytes())
        );
    }
    // Untouched keys keep their round-1 value.
    for k in 6..10 {
        assert_eq!(
            engine.get(format!("k{k}").into_bytes().as_slice()),
            Some(format!("v1-{k}").into_bytes())
        );
    }
    // Deleted key stays gone.
    assert_eq!(engine.get(b"k5"), None);
}

#[tokio::test]
async fn compaction_preserves_pinned_versions_for_time_travel() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = small_config(dir.path());
    cfg.l0_compaction_trigger = 10_000; // explicit compaction only
    let engine = Engine::new(cfg).unwrap();

    // v1 for two keys, flush. Capture the commit timestamp of v1.
    let ts_a1 = engine.put(b"a".to_vec(), b"a-v1".to_vec()).await.unwrap();
    let _ts_b1 = engine.put(b"b".to_vec(), b"b-v1".to_vec()).await.unwrap();
    engine.flush_memtable().await.unwrap();

    // Pin the v1 snapshot timestamp, then overwrite with v2 and flush.
    engine.set_pin_source(Arc::new(FixedPins(vec![ts_a1])));
    engine.put(b"a".to_vec(), b"a-v2".to_vec()).await.unwrap();
    engine.put(b"b".to_vec(), b"b-v2".to_vec()).await.unwrap();
    engine.flush_memtable().await.unwrap();

    let stats = engine.compact().unwrap();
    // Two distinct surviving timestamps (v1 pinned + v2 latest) → the
    // parallel bucket build produces at least the v2 + pinned-v1 outputs.
    assert!(stats.output_ssts >= 1);
    assert_eq!(stats.keys_retained, 2);

    // Time-travel read at the pinned snapshot still sees v1.
    let at_v1 = engine.scan_all_at(ts_a1);
    let a_v1 = at_v1.iter().find(|(k, _, _)| k == b"a").map(|(_, v, _)| v.clone());
    assert_eq!(a_v1, Some(b"a-v1".to_vec()), "pinned historical version must survive compaction");

    // Current reads see v2.
    assert_eq!(engine.get(b"a"), Some(b"a-v2".to_vec()));
    assert_eq!(engine.get(b"b"), Some(b"b-v2".to_vec()));
    let now = engine.scan_all_at(Timestamp::MAX);
    let a_now = now.iter().find(|(k, _, _)| k == b"a").map(|(_, v, _)| v.clone());
    assert_eq!(a_now, Some(b"a-v2".to_vec()));
}

#[tokio::test]
async fn compaction_is_noop_below_two_ssts() {
    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::new(small_config(dir.path())).unwrap();
    engine.put(b"only".to_vec(), b"one".to_vec()).await.unwrap();
    engine.flush_memtable().await.unwrap();
    // One SST → nothing to merge.
    let stats = engine.compact().unwrap();
    assert_eq!(stats.input_ssts, 1);
    assert_eq!(stats.output_ssts, 1);
    assert_eq!(engine.get(b"only"), Some(b"one".to_vec()));
}
