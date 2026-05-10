//! Tests for disk-full handling.

use super::*;
use std::fs;
use std::sync::Mutex;

/// Small reserve size used in tests (4 KB) to keep things fast.
const TEST_RESERVE_SIZE: u64 = 4 * 1024;

/// Tests that read the process-wide `galaxdb_disk_full` Prometheus gauge must
/// not run concurrently — a second test flipping the flag can race with the
/// first test's assertion. This is the standard Rust-test pattern for
/// serialising access to a singleton. We only guard the gauge-reading tests;
/// tests that only check the local `AtomicBool` flag are isolated per handler
/// and can stay parallel.
static GAUGE_SERIAL: Mutex<()> = Mutex::new(());

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

// ---------------------------------------------------------------
// Phase E — `galaxdb_disk_full` Prometheus metric
// ---------------------------------------------------------------
//
// The process-wide `IntGauge` is shared across every `DiskFullHandler`
// instance. The `GAUGE_SERIAL` mutex above serialises any test that asserts
// on the gauge's value so a concurrent flip from a sibling test cannot race
// the check.

/// Look up the current value of `galaxdb_disk_full` from the default
/// Prometheus registry exposed by `galaxdb-observe`. This is the value a
/// real Prometheus scraper would read from `/metrics`, so asserting on it
/// guarantees E2 (registration with the observe registry) is wired up.
fn gauge_value_from_default_registry() -> i64 {
    let registry = galaxdb_observe::default_registry();
    for family in registry.gather() {
        if family.get_name() == "galaxdb_disk_full" {
            let metrics = family.get_metric();
            assert_eq!(
                metrics.len(),
                1,
                "galaxdb_disk_full should be a single gauge, found {} metrics",
                metrics.len()
            );
            return metrics[0].get_gauge().get_value() as i64;
        }
    }
    panic!(
        "galaxdb_disk_full metric was not registered with galaxdb_observe::default_registry()"
    );
}

#[test]
fn disk_full_gauge_sets_to_one_when_tripped() {
    let _lock = GAUGE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let handler = DiskFullHandler::init(dir.path(), TEST_RESERVE_SIZE).unwrap();

    // Starting state: gauge is 0 (set by `init`).
    assert_eq!(handler.disk_full_gauge(), 0);
    assert_eq!(gauge_value_from_default_registry(), 0);

    // Trip disk-full. Gauge must read 1 through both paths — the handler's
    // accessor and the default Prometheus registry.
    handler.handle_disk_full().unwrap();
    assert_eq!(
        handler.disk_full_gauge(),
        1,
        "handler accessor must read 1 after handle_disk_full"
    );
    assert_eq!(
        gauge_value_from_default_registry(),
        1,
        "default registry must scrape 1 after handle_disk_full"
    );

    // Reset for the next test that shares this singleton.
    handler.recover().unwrap();
}

#[test]
fn disk_full_gauge_sets_to_zero_after_recovery() {
    let _lock = GAUGE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let handler = DiskFullHandler::init(dir.path(), TEST_RESERVE_SIZE).unwrap();

    handler.handle_disk_full().unwrap();
    assert_eq!(handler.disk_full_gauge(), 1);
    assert_eq!(gauge_value_from_default_registry(), 1);

    handler.recover().unwrap();
    assert_eq!(
        handler.disk_full_gauge(),
        0,
        "handler accessor must read 0 after recover"
    );
    assert_eq!(
        gauge_value_from_default_registry(),
        0,
        "default registry must scrape 0 after recover"
    );
}

#[test]
fn disk_full_gauge_is_registered_with_default_registry() {
    let _lock = GAUGE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    // Constructing any handler must register the gauge once — confirm the
    // metric appears in the default registry's metric families and carries
    // the canonical help string.
    let dir = tempfile::tempdir().unwrap();
    let _handler = DiskFullHandler::init(dir.path(), TEST_RESERVE_SIZE).unwrap();

    let registry = galaxdb_observe::default_registry();
    let family = registry
        .gather()
        .into_iter()
        .find(|f| f.get_name() == "galaxdb_disk_full")
        .expect("galaxdb_disk_full must be registered with the default Prometheus registry");
    assert_eq!(
        family.get_help(),
        "Set to 1 while the storage engine is in disk-full recovery mode, 0 otherwise."
    );
}
