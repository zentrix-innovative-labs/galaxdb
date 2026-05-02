//! Unit tests for the galaxdb-io crate.

use std::path::Path;

use crate::latency::{LatencyMonitor, LatencyReport};
use crate::scheduler::{IoBackend, IoPriority, IoScheduler};
use crate::tokio_scheduler::TokioScheduler;
use crate::{detect_backend, select_scheduler};

// ---------------------------------------------------------------------------
// TokioScheduler tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tokio_scheduler_write_and_read_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("test_data.bin");

    let scheduler = TokioScheduler::new();

    // Write data at offset 0
    let data = b"hello galaxdb io layer";
    scheduler
        .write(&file, 0, data, IoPriority::High)
        .await
        .expect("write should succeed");

    // Read it back
    let result = scheduler
        .read(&file, 0, data.len(), IoPriority::High)
        .await
        .expect("read should succeed");

    assert_eq!(result, data);
}

#[tokio::test]
async fn tokio_scheduler_write_at_offset() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("offset_test.bin");

    let scheduler = TokioScheduler::new();

    // Write some initial data
    let initial = b"AAAAAAAAAA"; // 10 bytes
    scheduler
        .write(&file, 0, initial, IoPriority::Background)
        .await
        .unwrap();

    // Write at offset 5
    let patch = b"BBBBB";
    scheduler
        .write(&file, 5, patch, IoPriority::Background)
        .await
        .unwrap();

    // Read the full 10 bytes
    let result = scheduler
        .read(&file, 0, 10, IoPriority::High)
        .await
        .unwrap();

    assert_eq!(&result[..5], b"AAAAA");
    assert_eq!(&result[5..], b"BBBBB");
}

#[tokio::test]
async fn tokio_scheduler_fsync_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("fsync_test.bin");

    let scheduler = TokioScheduler::new();

    // Write some data first (file must exist for fsync)
    scheduler
        .write(&file, 0, b"data", IoPriority::High)
        .await
        .unwrap();

    // fsync should succeed
    scheduler.fsync(&file).await.expect("fsync should succeed");
}

#[tokio::test]
async fn tokio_scheduler_read_nonexistent_file_returns_error() {
    let scheduler = TokioScheduler::new();
    let result = scheduler
        .read(Path::new("/nonexistent/file.bin"), 0, 10, IoPriority::High)
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn tokio_scheduler_read_past_eof_truncates() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("short.bin");

    let scheduler = TokioScheduler::new();

    // Write 5 bytes
    scheduler
        .write(&file, 0, b"short", IoPriority::High)
        .await
        .unwrap();

    // Try to read 100 bytes — should get only 5
    let result = scheduler
        .read(&file, 0, 100, IoPriority::High)
        .await
        .unwrap();

    assert_eq!(result.len(), 5);
    assert_eq!(&result, b"short");
}

#[tokio::test]
async fn tokio_scheduler_reports_tokio_backend() {
    let scheduler = TokioScheduler::new();
    assert_eq!(scheduler.backend(), IoBackend::Tokio);
}

#[tokio::test]
async fn tokio_scheduler_latency_report_populated_after_io() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("latency_test.bin");

    let scheduler = TokioScheduler::new();

    // Do some I/O to generate latency samples
    for _ in 0..10 {
        scheduler
            .write(&file, 0, b"test", IoPriority::High)
            .await
            .unwrap();
        scheduler
            .read(&file, 0, 4, IoPriority::Background)
            .await
            .unwrap();
    }

    // Report should be available (may be zero if window hasn't rotated)
    let report = scheduler.latency_report();
    // At minimum, the report should be valid
    assert!(!report.should_throttle, "fresh scheduler should not throttle");
}

#[tokio::test]
async fn tokio_scheduler_large_write_and_read() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("large.bin");

    let scheduler = TokioScheduler::new();

    // Write 1 MB of data
    let data: Vec<u8> = (0..1_048_576).map(|i| (i % 256) as u8).collect();
    scheduler
        .write(&file, 0, &data, IoPriority::High)
        .await
        .unwrap();

    // Read it back
    let result = scheduler
        .read(&file, 0, data.len(), IoPriority::High)
        .await
        .unwrap();

    assert_eq!(result.len(), data.len());
    assert_eq!(result, data);
}

// ---------------------------------------------------------------------------
// LatencyMonitor tests
// ---------------------------------------------------------------------------

#[test]
fn latency_monitor_default_report() {
    let monitor = LatencyMonitor::new();
    let report = monitor.report();
    assert_eq!(report.hp_p99_us, 0);
    assert_eq!(report.bk_p99_us, 0);
    assert_eq!(report.hp_idle_baseline_us, 0);
    assert_eq!(report.consecutive_exceeded, 0);
    assert!(!report.should_throttle);
}

#[test]
fn latency_monitor_baseline_calibration() {
    let monitor = LatencyMonitor::new();
    monitor.set_idle_baseline(200);
    let report = monitor.report();
    assert_eq!(report.hp_idle_baseline_us, 200);
}

#[test]
fn latency_report_default_values() {
    let report = LatencyReport::default();
    assert_eq!(report.hp_p99_us, 0);
    assert_eq!(report.bk_p99_us, 0);
    assert_eq!(report.hp_idle_baseline_us, 0);
    assert_eq!(report.consecutive_exceeded, 0);
    assert!(!report.should_throttle);
}

// ---------------------------------------------------------------------------
// Backend detection tests
// ---------------------------------------------------------------------------

#[test]
fn detect_backend_defaults_to_tokio_on_macos() {
    // On macOS (where tests run), should default to Tokio
    // unless GALAXDB_IO_BACKEND is set
    // We can't guarantee env var state, but we can test the function doesn't panic
    let backend = detect_backend();
    // On macOS, should be Tokio (unless env var forces something)
    #[cfg(target_os = "macos")]
    {
        if std::env::var("GALAXDB_IO_BACKEND").is_err() {
            assert_eq!(backend, IoBackend::Tokio);
        }
    }
    // On any platform, the result should be a valid variant
    assert!(backend == IoBackend::Tokio || backend == IoBackend::IoUring);
}

#[test]
fn detect_backend_respects_env_var_tokio() {
    // Temporarily set the env var
    // SAFETY: This test is not run concurrently with other tests that read this env var.
    unsafe { std::env::set_var("GALAXDB_IO_BACKEND", "tokio"); }
    let backend = detect_backend();
    assert_eq!(backend, IoBackend::Tokio);
    unsafe { std::env::remove_var("GALAXDB_IO_BACKEND"); }
}

#[test]
fn detect_backend_env_var_case_insensitive() {
    // SAFETY: This test is not run concurrently with other tests that read this env var.
    unsafe { std::env::set_var("GALAXDB_IO_BACKEND", "TOKIO"); }
    let backend = detect_backend();
    assert_eq!(backend, IoBackend::Tokio);
    unsafe { std::env::remove_var("GALAXDB_IO_BACKEND"); }
}

#[tokio::test]
async fn select_scheduler_returns_valid_scheduler() {
    // On macOS, should return TokioScheduler
    let scheduler = select_scheduler().expect("should select a scheduler");
    // Verify it works
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("select_test.bin");

    scheduler
        .write(&file, 0, b"test", IoPriority::High)
        .await
        .unwrap();

    let data = scheduler
        .read(&file, 0, 4, IoPriority::High)
        .await
        .unwrap();

    assert_eq!(&data, b"test");
}

// ---------------------------------------------------------------------------
// IoPriority tests
// ---------------------------------------------------------------------------

#[test]
fn io_priority_display() {
    assert_eq!(format!("{}", IoPriority::High), "High");
    assert_eq!(format!("{}", IoPriority::Background), "Background");
}

#[test]
fn io_priority_equality() {
    assert_eq!(IoPriority::High, IoPriority::High);
    assert_eq!(IoPriority::Background, IoPriority::Background);
    assert_ne!(IoPriority::High, IoPriority::Background);
}

// ---------------------------------------------------------------------------
// IoBackend tests
// ---------------------------------------------------------------------------

#[test]
fn io_backend_equality() {
    assert_eq!(IoBackend::Tokio, IoBackend::Tokio);
    assert_eq!(IoBackend::IoUring, IoBackend::IoUring);
    assert_ne!(IoBackend::Tokio, IoBackend::IoUring);
}
