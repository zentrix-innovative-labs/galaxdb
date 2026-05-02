//! Tests for disk-full handling.

use super::*;
use std::fs;

/// Small reserve size used in tests (4 KB) to keep things fast.
const TEST_RESERVE_SIZE: u64 = 4 * 1024;

// ---------------------------------------------------------------
// 14.1 — Pre-allocate reserve file at startup
// ---------------------------------------------------------------

#[test]
fn init_creates_reserve_file_with_correct_size() {
    let dir = tempfile::tempdir().unwrap();
    let handler = DiskFullHandler::init(dir.path(), TEST_RESERVE_SIZE).unwrap();

    let meta = fs::metadata(handler.reserve_path()).unwrap();
    assert_eq!(meta.len(), TEST_RESERVE_SIZE);
}

#[test]
fn init_creates_data_dir_if_missing() {
    let dir = tempfile::tempdir().unwrap();
    let nested = dir.path().join("a").join("b").join("c");

    let handler = DiskFullHandler::init(&nested, TEST_RESERVE_SIZE).unwrap();
    assert!(handler.reserve_path().exists());
}

#[test]
fn init_not_in_disk_full_mode() {
    let dir = tempfile::tempdir().unwrap();
    let handler = DiskFullHandler::init(dir.path(), TEST_RESERVE_SIZE).unwrap();
    assert!(!handler.is_disk_full());
}

#[test]
fn reserve_file_name_is_correct() {
    let dir = tempfile::tempdir().unwrap();
    let handler = DiskFullHandler::init(dir.path(), TEST_RESERVE_SIZE).unwrap();
    assert_eq!(
        handler.reserve_path().file_name().unwrap().to_str().unwrap(),
        "_galaxdb_reserve"
    );
}

#[test]
fn reserve_size_accessor_returns_configured_value() {
    let dir = tempfile::tempdir().unwrap();
    let handler = DiskFullHandler::init(dir.path(), 1234).unwrap();
    assert_eq!(handler.reserve_size(), 1234);
}

// ---------------------------------------------------------------
// 14.2 — Disk-full detection: delete reserve, block writes, emit metric
// ---------------------------------------------------------------

#[test]
fn handle_disk_full_deletes_reserve_file() {
    let dir = tempfile::tempdir().unwrap();
    let handler = DiskFullHandler::init(dir.path(), TEST_RESERVE_SIZE).unwrap();

    assert!(handler.reserve_path().exists());
    handler.handle_disk_full().unwrap();
    assert!(!handler.reserve_path().exists());
}

#[test]
fn handle_disk_full_sets_flag() {
    let dir = tempfile::tempdir().unwrap();
    let handler = DiskFullHandler::init(dir.path(), TEST_RESERVE_SIZE).unwrap();

    assert!(!handler.is_disk_full());
    handler.handle_disk_full().unwrap();
    assert!(handler.is_disk_full());
}

#[test]
fn handle_disk_full_is_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    let handler = DiskFullHandler::init(dir.path(), TEST_RESERVE_SIZE).unwrap();

    handler.handle_disk_full().unwrap();
    // Second call should not error even though the file is already gone.
    handler.handle_disk_full().unwrap();
    assert!(handler.is_disk_full());
}

#[test]
fn writes_blocked_after_disk_full() {
    let dir = tempfile::tempdir().unwrap();
    let handler = DiskFullHandler::init(dir.path(), TEST_RESERVE_SIZE).unwrap();

    handler.handle_disk_full().unwrap();

    // Simulate a write check — the caller should consult is_disk_full().
    assert!(handler.is_disk_full());
    // In the real engine this would return GalaxError::DiskFull.
}

// ---------------------------------------------------------------
// 14.3 — Recovery, clean checkpoint simulation, no data corruption
// ---------------------------------------------------------------

#[test]
fn recover_recreates_reserve_file_and_unblocks_writes() {
    let dir = tempfile::tempdir().unwrap();
    let handler = DiskFullHandler::init(dir.path(), TEST_RESERVE_SIZE).unwrap();

    handler.handle_disk_full().unwrap();
    assert!(handler.is_disk_full());
    assert!(!handler.reserve_path().exists());

    handler.recover().unwrap();
    assert!(!handler.is_disk_full());
    assert!(handler.reserve_path().exists());

    let meta = fs::metadata(handler.reserve_path()).unwrap();
    assert_eq!(meta.len(), TEST_RESERVE_SIZE);
}

#[test]
fn recover_is_noop_when_not_in_disk_full_mode() {
    let dir = tempfile::tempdir().unwrap();
    let handler = DiskFullHandler::init(dir.path(), TEST_RESERVE_SIZE).unwrap();

    // Not in disk-full mode — recover should succeed silently.
    handler.recover().unwrap();
    assert!(!handler.is_disk_full());
    assert!(handler.reserve_path().exists());
}

#[test]
fn full_lifecycle_init_diskfull_recover() {
    let dir = tempfile::tempdir().unwrap();
    let handler = DiskFullHandler::init(dir.path(), TEST_RESERVE_SIZE).unwrap();

    // Phase 1: Normal operation.
    assert!(!handler.is_disk_full());
    assert!(handler.reserve_path().exists());

    // Phase 2: Disk full detected.
    handler.handle_disk_full().unwrap();
    assert!(handler.is_disk_full());
    assert!(!handler.reserve_path().exists());

    // Phase 3: Operator frees space, recover.
    handler.recover().unwrap();
    assert!(!handler.is_disk_full());
    assert!(handler.reserve_path().exists());
    assert_eq!(
        fs::metadata(handler.reserve_path()).unwrap().len(),
        TEST_RESERVE_SIZE
    );
}

#[test]
fn disk_full_simulation_no_data_corruption() {
    // Simulate the full disk-full flow including a "checkpoint" step.
    //
    // We write some data to a file before the disk-full event, trigger
    // the handler, write a "checkpoint" marker, and verify that both the
    // original data and the checkpoint are intact.

    let dir = tempfile::tempdir().unwrap();
    let data_file = dir.path().join("data.bin");

    // Pre-condition: write some committed data.
    let committed_data = b"committed-row-data-abc-123";
    fs::write(&data_file, committed_data).unwrap();

    // Init handler.
    let handler = DiskFullHandler::init(dir.path(), TEST_RESERVE_SIZE).unwrap();

    // Trigger disk-full.
    handler.handle_disk_full().unwrap();
    assert!(handler.is_disk_full());

    // Perform a "clean checkpoint" — in the real engine this flushes the
    // memtable and writes a WAL checkpoint record. Here we simulate by
    // writing a checkpoint marker file using the space freed by the
    // reserve file deletion.
    let checkpoint_file = dir.path().join("checkpoint.bin");
    let checkpoint_data = b"checkpoint-record";
    fs::write(&checkpoint_file, checkpoint_data).unwrap();

    // Verify: committed data is intact.
    assert_eq!(fs::read(&data_file).unwrap(), committed_data);

    // Verify: checkpoint was written successfully.
    assert_eq!(fs::read(&checkpoint_file).unwrap(), checkpoint_data);

    // Verify: writes are blocked.
    assert!(handler.is_disk_full());
}

#[test]
fn clean_checkpoint_before_stop() {
    // Verify that after handle_disk_full, there is enough free space
    // (from the deleted reserve) to write a checkpoint-sized payload.

    let dir = tempfile::tempdir().unwrap();
    let handler = DiskFullHandler::init(dir.path(), TEST_RESERVE_SIZE).unwrap();

    // Confirm reserve file exists and has the right size.
    assert_eq!(
        fs::metadata(handler.reserve_path()).unwrap().len(),
        TEST_RESERVE_SIZE
    );

    handler.handle_disk_full().unwrap();

    // The reserve file is gone — we should be able to write up to
    // TEST_RESERVE_SIZE bytes for a checkpoint.
    let checkpoint_path = dir.path().join("wal_checkpoint");
    let checkpoint_payload = vec![0xCDu8; TEST_RESERVE_SIZE as usize];
    fs::write(&checkpoint_path, &checkpoint_payload).unwrap();

    // Read back and verify integrity.
    let read_back = fs::read(&checkpoint_path).unwrap();
    assert_eq!(read_back.len(), TEST_RESERVE_SIZE as usize);
    assert!(read_back.iter().all(|&b| b == 0xCD));
}
