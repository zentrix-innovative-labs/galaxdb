//! Tests for the WriteController module.

use super::*;
use std::time::Instant;

const GB: u64 = 1024 * 1024 * 1024;

/// Helper to create a WriteController with small limits for fast testing.
/// soft = 100 bytes, hard = 200 bytes, max_delay = 50 ms.
fn test_controller() -> WriteController {
    let config = WriteControllerConfig {
        soft_limit_bytes: 100,
        hard_limit_bytes: 200,
        max_delay_ms: 50,
    };
    WriteController::new(config)
}

// ── Default configuration tests ────────────────────────────────────

#[test]
fn default_config_has_correct_limits() {
    let config = WriteControllerConfig::default();
    assert_eq!(config.soft_limit_bytes, 32 * GB);
    assert_eq!(config.hard_limit_bytes, 64 * GB);
    assert_eq!(config.max_delay_ms, 100);
}

#[test]
fn new_controller_starts_with_zero_pending() {
    let controller = test_controller();
    assert_eq!(controller.pending_compaction_bytes(), 0);
    assert!(!controller.is_throttled());
    assert!(!controller.is_stopped());
}

// ── Slowdown factor tests ──────────────────────────────────────────

#[test]
fn slowdown_factor_zero_below_soft_limit() {
    let controller = test_controller();
    controller.update_pending_bytes(0);
    assert!((controller.slowdown_factor() - 0.0).abs() < f64::EPSILON);

    controller.update_pending_bytes(50);
    assert!((controller.slowdown_factor() - 0.0).abs() < f64::EPSILON);

    controller.update_pending_bytes(99);
    assert!((controller.slowdown_factor() - 0.0).abs() < f64::EPSILON);

    // Exactly at soft limit — still 0.0 (soft limit is the boundary).
    controller.update_pending_bytes(100);
    assert!((controller.slowdown_factor() - 0.0).abs() < f64::EPSILON);
}

#[test]
fn slowdown_factor_linear_between_limits() {
    let controller = test_controller();
    // soft=100, hard=200, range=100

    // 25% into the range
    controller.update_pending_bytes(125);
    assert!((controller.slowdown_factor() - 0.25).abs() < 0.001);

    // 50% into the range
    controller.update_pending_bytes(150);
    assert!((controller.slowdown_factor() - 0.50).abs() < 0.001);

    // 75% into the range
    controller.update_pending_bytes(175);
    assert!((controller.slowdown_factor() - 0.75).abs() < 0.001);
}

#[test]
fn slowdown_factor_one_at_hard_limit() {
    let controller = test_controller();
    controller.update_pending_bytes(200);
    assert!((controller.slowdown_factor() - 1.0).abs() < f64::EPSILON);
}

#[test]
fn slowdown_factor_one_above_hard_limit() {
    let controller = test_controller();
    controller.update_pending_bytes(300);
    assert!((controller.slowdown_factor() - 1.0).abs() < f64::EPSILON);
}

// ── State query tests ──────────────────────────────────────────────

#[test]
fn is_throttled_reflects_soft_limit() {
    let controller = test_controller();

    controller.update_pending_bytes(50);
    assert!(!controller.is_throttled());

    controller.update_pending_bytes(100);
    assert!(controller.is_throttled());

    controller.update_pending_bytes(150);
    assert!(controller.is_throttled());
}

#[test]
fn is_stopped_reflects_hard_limit() {
    let controller = test_controller();

    controller.update_pending_bytes(150);
    assert!(!controller.is_stopped());

    controller.update_pending_bytes(200);
    assert!(controller.is_stopped());

    controller.update_pending_bytes(300);
    assert!(controller.is_stopped());
}

#[test]
fn update_pending_bytes_stores_value() {
    let controller = test_controller();
    controller.update_pending_bytes(42);
    assert_eq!(controller.pending_compaction_bytes(), 42);

    controller.update_pending_bytes(999);
    assert_eq!(controller.pending_compaction_bytes(), 999);
}

#[test]
fn soft_and_hard_limit_accessors() {
    let controller = test_controller();
    assert_eq!(controller.soft_limit(), 100);
    assert_eq!(controller.hard_limit(), 200);
}

// ── reduce_pending_bytes tests ─────────────────────────────────────

#[test]
fn reduce_pending_bytes_subtracts() {
    let controller = test_controller();
    controller.update_pending_bytes(500);
    controller.reduce_pending_bytes(200);
    assert_eq!(controller.pending_compaction_bytes(), 300);
}

#[test]
fn reduce_pending_bytes_clamps_to_zero() {
    let controller = test_controller();
    controller.update_pending_bytes(100);
    controller.reduce_pending_bytes(200); // More than current
    assert_eq!(controller.pending_compaction_bytes(), 0);
}

#[test]
fn reduce_pending_bytes_from_zero() {
    let controller = test_controller();
    controller.reduce_pending_bytes(50);
    assert_eq!(controller.pending_compaction_bytes(), 0);
}

// ── check_write / WriteAdmission tests ─────────────────────────────

#[test]
fn check_write_proceed_below_soft_limit() {
    let controller = test_controller();
    controller.update_pending_bytes(0);
    assert_eq!(controller.check_write(), WriteAdmission::Proceed);

    controller.update_pending_bytes(50);
    assert_eq!(controller.check_write(), WriteAdmission::Proceed);

    controller.update_pending_bytes(99);
    assert_eq!(controller.check_write(), WriteAdmission::Proceed);
}

#[test]
fn check_write_delay_between_limits() {
    let controller = test_controller();
    // soft=100, hard=200, max_delay=50ms

    // 50% into the range → factor=0.5 → delay=25ms
    controller.update_pending_bytes(150);
    match controller.check_write() {
        WriteAdmission::Delay(d) => {
            assert_eq!(d, Duration::from_millis(25));
        }
        other => panic!("Expected Delay, got {:?}", other),
    }

    // 25% into the range → factor=0.25 → delay=12ms
    controller.update_pending_bytes(125);
    match controller.check_write() {
        WriteAdmission::Delay(d) => {
            assert_eq!(d, Duration::from_millis(12));
        }
        other => panic!("Expected Delay, got {:?}", other),
    }

    // 75% into the range → factor=0.75 → delay=37ms
    controller.update_pending_bytes(175);
    match controller.check_write() {
        WriteAdmission::Delay(d) => {
            assert_eq!(d, Duration::from_millis(37));
        }
        other => panic!("Expected Delay, got {:?}", other),
    }
}

#[test]
fn check_write_block_at_hard_limit() {
    let controller = test_controller();
    controller.update_pending_bytes(200);
    assert_eq!(controller.check_write(), WriteAdmission::Block);
}

#[test]
fn check_write_block_above_hard_limit() {
    let controller = test_controller();
    controller.update_pending_bytes(500);
    assert_eq!(controller.check_write(), WriteAdmission::Block);
}

// ── Admit write async tests ────────────────────────────────────────

#[tokio::test]
async fn admit_write_returns_immediately_below_soft_limit() {
    let controller = test_controller();
    controller.update_pending_bytes(50);

    let start = Instant::now();
    controller.admit_write().await;
    let elapsed = start.elapsed();

    // Should be nearly instant (< 5 ms).
    assert!(
        elapsed.as_millis() < 5,
        "Expected immediate return, got {}ms",
        elapsed.as_millis()
    );
}

#[tokio::test]
async fn admit_write_returns_immediately_at_zero_pending() {
    let controller = test_controller();

    let start = Instant::now();
    controller.admit_write().await;
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 5,
        "Expected immediate return, got {}ms",
        elapsed.as_millis()
    );
}

#[tokio::test]
async fn admit_write_applies_proportional_delay() {
    let controller = test_controller();
    // 50% into the range → factor = 0.5 → delay = 0.5 * 50ms = 25ms
    controller.update_pending_bytes(150);

    let start = Instant::now();
    controller.admit_write().await;
    let elapsed = start.elapsed();

    // Should have waited approximately 25 ms (allow tolerance).
    assert!(
        elapsed.as_millis() >= 15,
        "Expected ~25ms delay, got {}ms",
        elapsed.as_millis()
    );
    assert!(
        elapsed.as_millis() < 60,
        "Expected ~25ms delay, got {}ms",
        elapsed.as_millis()
    );
}

#[tokio::test]
async fn admit_write_blocks_at_hard_limit_then_recovers() {
    let controller = test_controller();
    controller.update_pending_bytes(200); // At hard limit

    // Spawn a task that will lower pending after a short delay.
    let unblock = tokio::spawn({
        let pending_ref = &controller.pending_compaction_bytes;
        let ptr = pending_ref as *const AtomicU64 as usize;
        async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            // Safety: the controller outlives this task (we join below).
            let atomic = unsafe { &*(ptr as *const AtomicU64) };
            atomic.store(50, Ordering::Relaxed); // Drop below soft limit
        }
    });

    let start = Instant::now();
    controller.admit_write().await;
    let elapsed = start.elapsed();

    unblock.await.unwrap();

    // Should have been blocked for ~20ms then returned quickly.
    assert!(
        elapsed.as_millis() >= 15,
        "Expected blocking for ~20ms, got {}ms",
        elapsed.as_millis()
    );
}

#[tokio::test]
async fn admit_write_hard_stop_with_partial_recovery() {
    // Start at hard limit, recover to between soft and hard.
    let controller = test_controller();
    controller.update_pending_bytes(250); // Above hard limit

    let pending_ref = &controller.pending_compaction_bytes;
    let ptr = pending_ref as *const AtomicU64 as usize;
    let unblock = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(15)).await;
        // Drop to 150 — between soft and hard, factor = 0.5
        let atomic = unsafe { &*(ptr as *const AtomicU64) };
        atomic.store(150, Ordering::Relaxed);
    });

    let start = Instant::now();
    controller.admit_write().await;
    let elapsed = start.elapsed();

    unblock.await.unwrap();

    // Should have blocked ~15ms, then applied proportional delay (~25ms).
    // Total ~40ms.
    assert!(
        elapsed.as_millis() >= 15,
        "Expected at least 15ms blocking, got {}ms",
        elapsed.as_millis()
    );
}

// ── Recovery tests ─────────────────────────────────────────────────

#[tokio::test]
async fn recovery_restores_full_throughput() {
    let controller = test_controller();

    // Start throttled.
    controller.update_pending_bytes(150);
    assert!(controller.is_throttled());

    // Recover below soft limit.
    controller.update_pending_bytes(50);
    assert!(!controller.is_throttled());
    assert!(!controller.is_stopped());
    assert!((controller.slowdown_factor() - 0.0).abs() < f64::EPSILON);

    // Writes should be immediate.
    let start = Instant::now();
    controller.admit_write().await;
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 5,
        "Expected immediate return after recovery, got {}ms",
        elapsed.as_millis()
    );
}

#[tokio::test]
async fn recovery_from_hard_stop_to_below_soft() {
    let controller = test_controller();

    // Hard stop.
    controller.update_pending_bytes(200);
    assert!(controller.is_stopped());

    // Recover fully.
    controller.update_pending_bytes(10);
    assert!(!controller.is_stopped());
    assert!(!controller.is_throttled());

    let start = Instant::now();
    controller.admit_write().await;
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 5,
        "Expected immediate return after full recovery, got {}ms",
        elapsed.as_millis()
    );
}

#[tokio::test]
async fn recovery_via_reduce_pending_bytes() {
    let controller = test_controller();

    // Start at hard stop.
    controller.update_pending_bytes(250);
    assert!(controller.is_stopped());
    assert_eq!(controller.check_write(), WriteAdmission::Block);

    // Compaction completes, reducing pending bytes below soft limit.
    controller.reduce_pending_bytes(200); // 250 - 200 = 50
    assert_eq!(controller.pending_compaction_bytes(), 50);
    assert!(!controller.is_stopped());
    assert!(!controller.is_throttled());
    assert_eq!(controller.check_write(), WriteAdmission::Proceed);

    // Writes should be immediate.
    let start = Instant::now();
    controller.admit_write().await;
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 5,
        "Expected immediate return after reduce_pending_bytes recovery, got {}ms",
        elapsed.as_millis()
    );
}

// ── Edge case tests ────────────────────────────────────────────────

#[test]
fn equal_soft_and_hard_limits() {
    let config = WriteControllerConfig {
        soft_limit_bytes: 100,
        hard_limit_bytes: 100,
        max_delay_ms: 50,
    };
    let controller = WriteController::new(config);

    controller.update_pending_bytes(50);
    assert!((controller.slowdown_factor() - 0.0).abs() < f64::EPSILON);

    // At the limit — both soft and hard are the same.
    // `is_stopped` returns true (pending >= hard_limit), so admit_write blocks.
    controller.update_pending_bytes(100);
    assert!(controller.is_stopped());

    // Above the limit — clearly a hard stop.
    controller.update_pending_bytes(101);
    assert!((controller.slowdown_factor() - 1.0).abs() < f64::EPSILON);
    assert!(controller.is_stopped());
}

#[test]
fn slowdown_factor_with_large_values() {
    let config = WriteControllerConfig::default();
    let controller = WriteController::new(config);

    // Midpoint between 32 GB and 64 GB = 48 GB
    controller.update_pending_bytes(48 * GB);
    assert!((controller.slowdown_factor() - 0.5).abs() < 0.001);
}

#[test]
fn check_write_at_exact_soft_limit_is_proceed() {
    let controller = test_controller();
    // At exactly the soft limit, pending <= soft_limit, so Proceed.
    controller.update_pending_bytes(100);
    assert_eq!(controller.check_write(), WriteAdmission::Proceed);
}

#[test]
fn check_write_just_above_soft_limit() {
    let controller = test_controller();
    // 1 byte above soft limit → factor = 1/100 = 0.01 → delay = 0.01 * 50 = 0ms (truncated)
    // So this should be Proceed since delay rounds to 0.
    controller.update_pending_bytes(101);
    // factor = 1/100 = 0.01, delay = 0.01 * 50 = 0.5 → truncated to 0
    assert_eq!(controller.check_write(), WriteAdmission::Proceed);

    // A bit further: 3 bytes above → factor = 3/100 = 0.03 → delay = 1.5 → 1ms
    controller.update_pending_bytes(103);
    match controller.check_write() {
        WriteAdmission::Delay(d) => {
            assert_eq!(d, Duration::from_millis(1));
        }
        other => panic!("Expected Delay, got {:?}", other),
    }
}

#[test]
fn update_pending_alias_works() {
    let controller = test_controller();
    controller.update_pending(42);
    assert_eq!(controller.pending_compaction_bytes(), 42);
}
