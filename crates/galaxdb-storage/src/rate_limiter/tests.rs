//! Tests for the RateLimiter module.

use super::*;
use galaxdb_io::LatencyReport;

/// Helper to create a RateLimiter with a known bandwidth for testing.
fn test_limiter(bandwidth: u64) -> RateLimiter {
    let config = RateLimiterConfig {
        max_bandwidth_bytes_per_sec: bandwidth,
        calibration_fraction: 0.70,
        throttle_reduction_fraction: 0.30,
    };
    RateLimiter::new(config)
}

/// Helper to create a LatencyReport that triggers throttling.
fn throttle_report() -> LatencyReport {
    LatencyReport {
        hp_p99_us: 300,
        bk_p99_us: 500,
        hp_idle_baseline_us: 100,
        consecutive_exceeded: 3,
        should_throttle: true,
    }
}

/// Helper to create a LatencyReport that indicates normal latency.
fn normal_report() -> LatencyReport {
    LatencyReport {
        hp_p99_us: 100,
        bk_p99_us: 200,
        hp_idle_baseline_us: 100,
        consecutive_exceeded: 0,
        should_throttle: false,
    }
}

// ── Calibration tests ──────────────────────────────────────────────

#[test]
fn new_limiter_uses_default_calibration() {
    let limiter = test_limiter(1_000_000_000); // 1 GB/s
    // 70% of 1 GB/s = 700 MB/s
    assert_eq!(limiter.max_rate(), 700_000_000);
    assert_eq!(limiter.current_ceiling(), 700_000_000);
    assert!(!limiter.is_throttled());
}

#[test]
fn calibrate_sets_rate_to_70_percent_of_measured() {
    let limiter = test_limiter(1_000_000_000);
    // Re-calibrate with a different measured bandwidth
    limiter.calibrate(2_000_000_000); // 2 GB/s
    // 70% of 2 GB/s = 1.4 GB/s
    assert_eq!(limiter.max_rate(), 1_400_000_000);
    assert_eq!(limiter.current_ceiling(), 1_400_000_000);
    assert!(!limiter.is_throttled());
}

#[test]
fn calibrate_resets_throttled_state() {
    let limiter = test_limiter(1_000_000_000);
    // Throttle first
    limiter.adjust_from_latency(&throttle_report());
    assert!(limiter.is_throttled());

    // Calibrate should reset throttled state
    limiter.calibrate(1_000_000_000);
    assert!(!limiter.is_throttled());
    assert_eq!(limiter.current_ceiling(), 700_000_000);
}

#[test]
fn calibrate_with_zero_bandwidth() {
    let limiter = test_limiter(0);
    assert_eq!(limiter.max_rate(), 0);
    assert_eq!(limiter.current_ceiling(), 0);
}

// ── Dynamic ceiling adjustment tests ───────────────────────────────

#[test]
fn adjust_from_latency_lowers_ceiling_on_throttle() {
    let limiter = test_limiter(1_000_000_000);
    // max_rate = 700_000_000
    limiter.adjust_from_latency(&throttle_report());

    assert!(limiter.is_throttled());
    // Ceiling should be 70% of max_rate = 700_000_000 * 0.70 = 490_000_000
    let expected = 700_000_000 - (700_000_000.0 * 0.30) as u64;
    assert_eq!(limiter.current_ceiling(), expected);
}

#[test]
fn adjust_from_latency_does_not_double_throttle() {
    let limiter = test_limiter(1_000_000_000);
    limiter.adjust_from_latency(&throttle_report());
    let ceiling_after_first = limiter.current_ceiling();

    // Second throttle should not reduce further
    limiter.adjust_from_latency(&throttle_report());
    assert_eq!(limiter.current_ceiling(), ceiling_after_first);
}

#[test]
fn adjust_from_latency_restores_ceiling_on_normal() {
    let limiter = test_limiter(1_000_000_000);
    // Throttle
    limiter.adjust_from_latency(&throttle_report());
    assert!(limiter.is_throttled());
    let throttled_ceiling = limiter.current_ceiling();
    assert!(throttled_ceiling < 700_000_000);

    // Restore
    limiter.adjust_from_latency(&normal_report());
    assert!(!limiter.is_throttled());
    assert_eq!(limiter.current_ceiling(), 700_000_000);
}

#[test]
fn adjust_from_latency_noop_when_already_normal() {
    let limiter = test_limiter(1_000_000_000);
    // Not throttled, normal report should be a no-op
    limiter.adjust_from_latency(&normal_report());
    assert!(!limiter.is_throttled());
    assert_eq!(limiter.current_ceiling(), 700_000_000);
}

#[test]
fn throttle_restore_cycle() {
    let limiter = test_limiter(1_000_000_000);

    // Cycle 1: throttle → restore
    limiter.adjust_from_latency(&throttle_report());
    assert!(limiter.is_throttled());
    limiter.adjust_from_latency(&normal_report());
    assert!(!limiter.is_throttled());
    assert_eq!(limiter.current_ceiling(), 700_000_000);

    // Cycle 2: throttle → restore again
    limiter.adjust_from_latency(&throttle_report());
    assert!(limiter.is_throttled());
    let expected = 700_000_000 - (700_000_000.0 * 0.30) as u64;
    assert_eq!(limiter.current_ceiling(), expected);
    limiter.adjust_from_latency(&normal_report());
    assert!(!limiter.is_throttled());
    assert_eq!(limiter.current_ceiling(), 700_000_000);
}

#[test]
fn report_with_consecutive_below_3_does_not_throttle() {
    let limiter = test_limiter(1_000_000_000);
    let report = LatencyReport {
        hp_p99_us: 300,
        bk_p99_us: 500,
        hp_idle_baseline_us: 100,
        consecutive_exceeded: 2,
        should_throttle: false, // Only 2 consecutive, not 3
    };
    limiter.adjust_from_latency(&report);
    assert!(!limiter.is_throttled());
    assert_eq!(limiter.current_ceiling(), 700_000_000);
}

// ── Token acquisition tests (async) ───────────────────────────────

#[tokio::test]
async fn acquire_zero_bytes_returns_immediately() {
    let limiter = test_limiter(1_000_000_000);
    // Should not block
    limiter.acquire(0).await;
}

#[tokio::test]
async fn acquire_small_amount_succeeds_immediately() {
    let limiter = test_limiter(1_000_000_000);
    // 700 MB/s rate, acquiring 1 KB should be instant
    limiter.acquire(1024).await;
}

#[tokio::test]
async fn acquire_under_load_respects_rate() {
    // Use a very low rate to make timing observable
    let config = RateLimiterConfig {
        max_bandwidth_bytes_per_sec: 10_000, // 10 KB/s
        calibration_fraction: 1.0,           // Use full bandwidth for simplicity
        throttle_reduction_fraction: 0.30,
    };
    let limiter = RateLimiter::new(config);

    // Acquire all available tokens (1 second worth = 10_000 bytes)
    limiter.acquire(10_000).await;

    // Now acquiring more should require waiting
    let start = Instant::now();
    limiter.acquire(5_000).await; // Need 0.5 seconds worth of tokens
    let elapsed = start.elapsed();

    // Should have waited approximately 0.5 seconds (allow generous tolerance)
    assert!(
        elapsed.as_millis() >= 200,
        "Expected wait of ~500ms, got {}ms",
        elapsed.as_millis()
    );
}

#[tokio::test]
async fn acquire_with_throttled_ceiling() {
    let config = RateLimiterConfig {
        max_bandwidth_bytes_per_sec: 100_000, // 100 KB/s
        calibration_fraction: 1.0,
        throttle_reduction_fraction: 0.30,
    };
    let limiter = RateLimiter::new(config);

    // Throttle the limiter
    limiter.adjust_from_latency(&throttle_report());
    assert!(limiter.is_throttled());
    // Ceiling should now be 70_000 bytes/sec
    assert_eq!(limiter.current_ceiling(), 70_000);

    // Acquire should still work, just at the reduced rate
    limiter.acquire(1024).await;
}

// ── Config tests ───────────────────────────────────────────────────

#[test]
fn default_config_values() {
    let config = RateLimiterConfig::default();
    assert_eq!(config.max_bandwidth_bytes_per_sec, 1_000_000_000);
    assert!((config.calibration_fraction - 0.70).abs() < f64::EPSILON);
    assert!((config.throttle_reduction_fraction - 0.30).abs() < f64::EPSILON);
}

#[test]
fn custom_config_fractions() {
    let config = RateLimiterConfig {
        max_bandwidth_bytes_per_sec: 500_000_000,
        calibration_fraction: 0.50,
        throttle_reduction_fraction: 0.20,
    };
    let limiter = RateLimiter::new(config);
    // 50% of 500 MB/s = 250 MB/s
    assert_eq!(limiter.max_rate(), 250_000_000);

    // Throttle: reduce by 20%
    limiter.adjust_from_latency(&throttle_report());
    let expected = 250_000_000 - (250_000_000.0 * 0.20) as u64;
    assert_eq!(limiter.current_ceiling(), expected);
}
