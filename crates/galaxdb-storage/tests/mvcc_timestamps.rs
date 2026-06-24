//! Real-scenario tests for per-row MVCC commit timestamps in SSTs.
//!
//! Before this fix the engine stamped a whole flushed block with the flush
//! sequence number, so `AT VERSION <ts>` could not express partial
//! visibility *within* a block — a snapshot landing in the middle of a
//! flushed batch returned rows that committed after it. These tests drive a
//! real on-disk engine (real WAL + flush + SST reads) and assert correct
//! per-row visibility.

use galaxdb_storage::engine::{Engine, EngineConfig};

fn cfg(dir: &std::path::Path) -> EngineConfig {
    EngineConfig {
        data_dir: dir.to_path_buf(),
        memtable_size_bytes: 1 << 30,
        back_pressure_bytes: 1 << 31,
        wal_group_commit_ms: 1,
        // Keep everything in one SST/block and never auto-compact, so the
        // test exercises a single block with rows at distinct timestamps.
        l0_compaction_trigger: 10_000,
        ..Default::default()
    }
}

#[tokio::test]
async fn at_version_has_per_row_visibility_within_one_flushed_block() {
    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::new(cfg(dir.path())).unwrap();

    // Three rows committed at three increasing MVCC timestamps, all flushed
    // together into a single SST block.
    let ts_a = engine.put(b"a".to_vec(), b"va".to_vec()).await.unwrap();
    let ts_b = engine.put(b"b".to_vec(), b"vb".to_vec()).await.unwrap();
    let ts_c = engine.put(b"c".to_vec(), b"vc".to_vec()).await.unwrap();
    assert!(ts_a < ts_b && ts_b < ts_c, "timestamps must be increasing");
    engine.flush_memtable().await.unwrap();

    // Snapshot at ts_b: a and b are visible, c is NOT (it committed later) —
    // even though all three live in the same block. This is the case a
    // single block-level timestamp could not represent.
    let at_b = engine.scan_all_at(ts_b);
    let keys_at_b: Vec<Vec<u8>> = at_b.iter().map(|(k, _, _)| k.clone()).collect();
    assert_eq!(keys_at_b, vec![b"a".to_vec(), b"b".to_vec()]);

    // Snapshot before any write sees nothing.
    let at_zero = engine.scan_all_at(ts_a - 1);
    assert!(at_zero.is_empty(), "no rows visible before the first commit");

    // Snapshot at/after ts_c sees all three.
    let at_c = engine.scan_all_at(ts_c);
    assert_eq!(at_c.len(), 3);

    // Each returned row carries its real MVCC commit timestamp.
    let a_ts = at_c.iter().find(|(k, _, _)| k == b"a").map(|(_, _, t)| *t);
    assert_eq!(a_ts, Some(ts_a));
}

#[tokio::test]
async fn at_version_resolves_updates_across_flushes() {
    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::new(cfg(dir.path())).unwrap();

    let ts_v1 = engine.put(b"k".to_vec(), b"v1".to_vec()).await.unwrap();
    engine.flush_memtable().await.unwrap();
    let ts_v2 = engine.put(b"k".to_vec(), b"v2".to_vec()).await.unwrap();
    engine.flush_memtable().await.unwrap();

    // At v1's snapshot the old value is visible; at v2's the new one.
    let v_at_1 = engine
        .scan_all_at(ts_v1)
        .into_iter()
        .find(|(k, _, _)| k == b"k")
        .map(|(_, v, _)| v);
    assert_eq!(v_at_1, Some(b"v1".to_vec()));

    let v_at_2 = engine
        .scan_all_at(ts_v2)
        .into_iter()
        .find(|(k, _, _)| k == b"k")
        .map(|(_, v, _)| v);
    assert_eq!(v_at_2, Some(b"v2".to_vec()));

    // Point read returns the latest.
    assert_eq!(engine.get(b"k"), Some(b"v2".to_vec()));
}

#[tokio::test]
async fn scan_all_returns_newest_version_regardless_of_sst_order() {
    // Two flushes put an old then a new value for the same key into two
    // separate SSTs. scan_all must return the newest version irrespective
    // of the registry's (hash-map) iteration order.
    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::new(cfg(dir.path())).unwrap();

    for i in 0..20 {
        engine
            .put(format!("k{i}").into_bytes(), b"old".to_vec())
            .await
            .unwrap();
    }
    engine.flush_memtable().await.unwrap();
    for i in 0..20 {
        engine
            .put(format!("k{i}").into_bytes(), b"new".to_vec())
            .await
            .unwrap();
    }
    engine.flush_memtable().await.unwrap();

    let rows = engine.scan_all();
    assert_eq!(rows.len(), 20, "one row per key, newest version");
    for (_k, v) in &rows {
        assert_eq!(v, b"new", "scan_all must return the newest version");
    }
}
