//! HP-queue latency monitoring with 100 ms sliding windows.
//!
//! The [`LatencyMonitor`] tracks per-operation latencies for both HP and BK
//! queues. Every 100 ms window it computes P99 latency. If the HP P99 exceeds
//! 1.5× the idle baseline for 3 consecutive windows, it signals the
//! RateLimiter via the [`LatencyReport`].

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// Latency report for RateLimiter feedback.
///
/// The RateLimiter uses this to decide whether to throttle background I/O.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LatencyReport {
    /// P99 latency of HP (high-priority) queue in microseconds.
    pub hp_p99_us: u64,
    /// P99 latency of BK (background) queue in microseconds.
    pub bk_p99_us: u64,
    /// Idle baseline latency of HP queue in microseconds.
    pub hp_idle_baseline_us: u64,
    /// Number of consecutive windows where HP P99 exceeded 1.5× baseline.
    pub consecutive_exceeded: u32,
    /// Whether the RateLimiter should be signaled (3+ consecutive exceeded windows).
    pub should_throttle: bool,
}

/// Tracks latency samples and computes windowed P99 statistics.
///
/// Used by both `IoUringScheduler` and `TokioScheduler` (though the Tokio
/// variant doesn't separate HP/BK queues, it still records latencies for
/// observability).
pub struct LatencyMonitor {
    /// HP queue latency samples for the current 100 ms window (in microseconds).
    hp_samples: Mutex<Vec<u64>>,
    /// BK queue latency samples for the current 100 ms window (in microseconds).
    bk_samples: Mutex<Vec<u64>>,
    /// Start time of the current measurement window.
    window_start: Mutex<Instant>,
    /// Latest computed HP P99 (microseconds).
    hp_p99_us: AtomicU64,
    /// Latest computed BK P99 (microseconds).
    bk_p99_us: AtomicU64,
    /// Idle baseline for HP queue (microseconds). Set during calibration.
    hp_idle_baseline_us: AtomicU64,
    /// Number of consecutive windows where HP P99 > 1.5× baseline.
    consecutive_exceeded: AtomicU64,
    /// Whether the idle baseline has been calibrated.
    baseline_calibrated: Mutex<bool>,
}

impl LatencyMonitor {
    /// Create a new latency monitor.
    pub fn new() -> Self {
        Self {
            hp_samples: Mutex::new(Vec::with_capacity(1024)),
            bk_samples: Mutex::new(Vec::with_capacity(1024)),
            window_start: Mutex::new(Instant::now()),
            hp_p99_us: AtomicU64::new(0),
            bk_p99_us: AtomicU64::new(0),
            hp_idle_baseline_us: AtomicU64::new(0),
            consecutive_exceeded: AtomicU64::new(0),
            baseline_calibrated: Mutex::new(false),
        }
    }

    /// Record a latency sample for the given priority queue.
    pub fn record(&self, priority: super::IoPriority, latency_us: u64) {
        let samples = match priority {
            super::IoPriority::High => &self.hp_samples,
            super::IoPriority::Background => &self.bk_samples,
        };
        if let Ok(mut s) = samples.lock() {
            s.push(latency_us);
        }

        // Check if the current window has elapsed (100 ms)
        self.maybe_rotate_window();
    }

    /// Set the idle baseline from calibration measurements.
    pub fn set_idle_baseline(&self, baseline_us: u64) {
        self.hp_idle_baseline_us.store(baseline_us, Ordering::Relaxed);
        if let Ok(mut calibrated) = self.baseline_calibrated.lock() {
            *calibrated = true;
        }
    }

    /// Get the current latency report.
    pub fn report(&self) -> LatencyReport {
        let hp_p99 = self.hp_p99_us.load(Ordering::Relaxed);
        let bk_p99 = self.bk_p99_us.load(Ordering::Relaxed);
        let baseline = self.hp_idle_baseline_us.load(Ordering::Relaxed);
        let consecutive = self.consecutive_exceeded.load(Ordering::Relaxed) as u32;

        LatencyReport {
            hp_p99_us: hp_p99,
            bk_p99_us: bk_p99,
            hp_idle_baseline_us: baseline,
            consecutive_exceeded: consecutive,
            should_throttle: consecutive >= 3,
        }
    }

    /// Check if the 100 ms window has elapsed and rotate if so.
    fn maybe_rotate_window(&self) {
        let should_rotate = {
            let start = self.window_start.lock().unwrap();
            start.elapsed().as_millis() >= 100
        };

        if should_rotate {
            self.rotate_window();
        }
    }

    /// Rotate the measurement window: compute P99, check threshold, reset samples.
    fn rotate_window(&self) {
        let mut window_start = self.window_start.lock().unwrap();

        // Double-check after acquiring lock
        if window_start.elapsed().as_millis() < 100 {
            return;
        }

        // Compute HP P99
        let hp_p99 = {
            let mut samples = self.hp_samples.lock().unwrap();
            let p99 = compute_p99(&mut samples);
            samples.clear();
            p99
        };

        // Compute BK P99
        let bk_p99 = {
            let mut samples = self.bk_samples.lock().unwrap();
            let p99 = compute_p99(&mut samples);
            samples.clear();
            p99
        };

        self.hp_p99_us.store(hp_p99, Ordering::Relaxed);
        self.bk_p99_us.store(bk_p99, Ordering::Relaxed);

        // Check if HP P99 exceeds 1.5× idle baseline
        let baseline = self.hp_idle_baseline_us.load(Ordering::Relaxed);
        let calibrated = self.baseline_calibrated.lock().map(|c| *c).unwrap_or(false);

        if calibrated && baseline > 0 && hp_p99 > 0 {
            let threshold = baseline + baseline / 2; // 1.5× baseline
            if hp_p99 > threshold {
                let prev = self.consecutive_exceeded.fetch_add(1, Ordering::Relaxed);
                if prev + 1 >= 3 {
                    tracing::warn!(
                        hp_p99_us = hp_p99,
                        baseline_us = baseline,
                        consecutive = prev + 1,
                        "HP-queue latency exceeded 1.5× baseline for 3+ consecutive windows"
                    );
                }
            } else {
                self.consecutive_exceeded.store(0, Ordering::Relaxed);
            }
        }

        // Reset window
        *window_start = Instant::now();
    }
}

impl Default for LatencyMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute P99 from a mutable slice of samples. Returns 0 if empty.
fn compute_p99(samples: &mut [u64]) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    samples.sort_unstable();
    let idx = ((samples.len() as f64) * 0.99).ceil() as usize;
    let idx = idx.min(samples.len()) - 1;
    samples[idx]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IoPriority;

    #[test]
    fn compute_p99_empty_returns_zero() {
        let mut samples = vec![];
        assert_eq!(compute_p99(&mut samples), 0);
    }

    #[test]
    fn compute_p99_single_sample() {
        let mut samples = vec![42];
        assert_eq!(compute_p99(&mut samples), 42);
    }

    #[test]
    fn compute_p99_hundred_samples() {
        let mut samples: Vec<u64> = (1..=100).collect();
        // P99 of 1..=100 should be 99 (index 98 in 0-based after sort)
        let p99 = compute_p99(&mut samples);
        assert!(p99 >= 99, "P99 should be >= 99, got {p99}");
    }

    #[test]
    fn latency_report_default_no_throttle() {
        let monitor = LatencyMonitor::new();
        let report = monitor.report();
        assert_eq!(report.hp_p99_us, 0);
        assert_eq!(report.bk_p99_us, 0);
        assert!(!report.should_throttle);
    }

    #[test]
    fn set_idle_baseline() {
        let monitor = LatencyMonitor::new();
        monitor.set_idle_baseline(100);
        let report = monitor.report();
        assert_eq!(report.hp_idle_baseline_us, 100);
    }

    #[test]
    fn record_samples_without_panic() {
        let monitor = LatencyMonitor::new();
        for i in 0..100 {
            monitor.record(IoPriority::High, i * 10);
            monitor.record(IoPriority::Background, i * 20);
        }
        // Should not panic
    }
}
