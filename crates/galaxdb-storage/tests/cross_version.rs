//! Cross-version integration test (v0.5 Workstream B, task B.7).
//!
//! Proves the rollback-safety contract Cloud's fleet-upgrade tooling depends on, end to end
//! through the real `Engine` open path — no mocks:
//!
//! 1. **Forward read.** Data written by the current engine is read cleanly after reopen
//!    (rows survive across a simulated restart via SST + WAL).
//! 2. **Newer-format refusal.** If an on-disk artifact is bumped to a version newer than this
//!    engine supports (simulating "written by a newer engine"), reopening is **refused** with a
//!    typed error rather than a best-effort read that could corrupt data — and the file is left
//!    intact, so a real rollback (restore snapshot + previous binary) is safe.
//!
//! FORMAT_VERSION is a compile-time constant, so a genuine two-binary vN/vN+1 test isn't
//! expressible in one process; tampering the on-disk version to `current + 1` is the faithful
//! in-process equivalent of "an older binary meets newer data".

use galaxdb_storage::engine::{Engine, EngineConfig};

fn config(dir: &std::path::Path) -> EngineConfig {
    EngineConfig {
        data_dir: dir.to_path_buf(),
        ..Default::default()
    }
}

fn find_sst(dir: &std::path::Path) -> std::path::PathBuf {
    std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            n.starts_with("sst_") && n.ends_with(".pax")
        })
        .expect("an SST file should exist after flush")
}

#[tokio::test]
async fn forward_read_then_refuse_future_sst() {
    let dir = tempfile::tempdir().unwrap();

    // Write rows and flush them to an SST (current format), then drop the engine.
    {
        let engine = Engine::new(config(dir.path())).unwrap();
        for i in 0..20 {
            engine
                .put_sync(format!("k{i:02}").into_bytes(), format!("v{i}").into_bytes())
                .unwrap();
        }
        engine.flush_memtable().await.unwrap();
    }

    // (1) Forward read: a fresh engine reads the current-format data cleanly.
    {
        let engine = Engine::new(config(dir.path())).unwrap();
        assert_eq!(engine.get(b"k00"), Some(b"v0".to_vec()));
        assert_eq!(engine.get(b"k19"), Some(b"v19".to_vec()));
    }

    // (2) Simulate "written by a newer engine": bump the SST footer's format
    // version. Versioned footer tail is [.. version(2) magic(4)], so the version
    // sits at len-6..len-4.
    let sst = find_sst(dir.path());
    let original = std::fs::read(&sst).unwrap();
    let mut tampered = original.clone();
    let n = tampered.len();
    let future = galaxdb_common::format::SST.current_write + 1;
    tampered[n - 6..n - 4].copy_from_slice(&future.to_le_bytes());
    std::fs::write(&sst, &tampered).unwrap();

    // Reopen must be refused with a typed FormatTooNew — never a silent misread.
    let err = Engine::new(config(dir.path()))
        .err()
        .expect("opening an SST newer than this engine must be refused");
    match err {
        galaxdb_common::GalaxError::FormatTooNew {
            artifact,
            found,
            current,
        } => {
            assert_eq!(artifact, "SST");
            assert_eq!(found, future);
            assert_eq!(current, galaxdb_common::format::SST.current_write);
        }
        other => panic!("expected FormatTooNew, got {other:?}"),
    }

    // The SST file is untouched by the refusal (rollback stays safe).
    assert!(sst.exists());
    assert_eq!(std::fs::read(&sst).unwrap(), tampered);
}

#[tokio::test]
async fn refuse_future_wal_superblock_on_open() {
    let dir = tempfile::tempdir().unwrap();

    {
        let engine = Engine::new(config(dir.path())).unwrap();
        engine.put_sync(b"key".to_vec(), b"val".to_vec()).unwrap();
        // No flush: the row lives in the WAL, which now carries a superblock.
    }

    // Bump the WAL superblock format version (header bytes 4..6) beyond current.
    let wal = dir.path().join("wal.log");
    let mut bytes = std::fs::read(&wal).unwrap();
    assert_eq!(&bytes[0..4], &galaxdb_common::format::WAL.magic);
    let future = galaxdb_common::format::WAL.current_write + 1;
    bytes[4..6].copy_from_slice(&future.to_le_bytes());
    std::fs::write(&wal, &bytes).unwrap();

    // Reopen must be refused (WalWriter::new range-checks the superblock).
    let err = Engine::new(config(dir.path()))
        .err()
        .expect("opening a WAL newer than this engine must be refused");
    assert!(
        err.to_string().contains("newer"),
        "expected a too-new refusal, got: {err}"
    );
}
