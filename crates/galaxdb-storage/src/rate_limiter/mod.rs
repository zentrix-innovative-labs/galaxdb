//! Auto-tuned token-bucket RateLimiter for compaction + flush I/O bandwidth control.
//!
//! The [`RateLimiter`] controls aggregate compaction and flush I/O bandwidth
//! using a token-bucket algorithm. At startup it calibrates its max rate to
//! 70% of the configured NVMe write bandwidth. It dynamically adjusts the
//! ceiling based on HP-queue latency feedback from the [`LatencyReport`].
//!
//! When the HP-queue P99 latency exceeds 1.5× the idle baseline for 3
//! consecutive 100 ms windows, the ceiling is lowered by 30%. When latency
//! returns to normal, the ceiling is restored.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use galaxdb_io::LatencyReport;

/// Configuration for the RateLimiter.
#[derive(Debug, Clone)]
pub struct RateLimiterConfig {
    /// Maximum NVMe write bandwidth in bytes/sec.
    /// The RateLimiter will calibrate to 70% of this value.
    pub max_bandwidth_bytes_per_sec: u64,
    /// Fraction of max bandwidth to use as the ceiling (default 0.70).
    pub calibration_fraction: f64,
    /// Fraction to reduce the ceiling by when throttling (default 0.30).
    pub throttle_reduction_fraction: f64,
}

impl Default for RateLimiterConfig {
    fn default() -> Self {
        Self {
            // Default: 1 GB/s NVMe write bandwidth
            max_bandwidth_bytes_per_sec: 1_000_000_000,
            calibration_fraction: 0.70,
            throttle_reduction_fraction: 0.30,
        }
    }
}

/// Internal token bucket state.
struct TokenBucket {
    /// Available tokens (bytes).
    available: f64,
    /// Maximum tokens that can accumulate.
    capacity: f64,
    /// Rate at which tokens are added (bytes/sec).
    rate: f64,
    /// Last time tokens were refilled.
    last_refill: Instant,
}

impl TokenBucket {
    fn new(rate: f64) -> Self {
        Self {
            available: rate, // Start with 1 second worth of tokens
            capacity: rate,  // Cap at 1 second worth
            rate,
            last_refill: Instant::now(),
        }
    }

    /// Refill tokens based on elapsed time.
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        if elapsed > 0.0 {
            self.available = (self.available + self.rate * elapsed).min(self.capacity);
            self.last_refill = now;
        }
    }

    /// Try to consume `bytes` tokens. Returns the wait time in seconds
    /// if there aren't enough tokens, or 0.0 if tokens were consumed.
    fn try_acquire(&mut self, bytes: u64) -> f64 {
        self.refill();
        let needed = bytes as f64;
        if self.available >= needed {
            self.available -= needed;
            0.0
        } else {
            let deficit = needed - self.available;
            if self.rate > 0.0 {
                deficit / self.rate
            } else {
                // Rate is zero — effectively blocked forever; return a large wait.
                1.0
            }
        }
    }

    /// Consume tokens after waiting. Called after the wait completes.
    fn consume_after_wait(&mut self, bytes: u64) {
        self.refill();
        self.available -= bytes as f64;
        // available can go negative briefly; next refill will recover.
    }

    /// Update the token rate and capacity.
    fn set_rate(&mut self, rate: f64) {
        self.rate = rate;
        self.capacity = rate; // 1 second worth of burst
        // Don't reset available — let existing tokens drain naturally.
        if self.available > self.capacity {
            self.available = self.capacity;
        }
    }
}

/// Auto-tuned token-bucket RateLimiter controlling aggregate compaction + flush
/// I/O bandwidth.
///
/// # Usage
///
/// ```ignore
/// let config = RateLimiterConfig::default();
/// let limiter = RateLimiter::new(config);
/// limiter.calibrate(1_000_000_000); // 1 GB/s measured bandwidth
///
/// // Before performing I/O in compaction or flush:
/// limiter.acquire(block_size).await;
/// // ... perform I/O ...
///
/// // Periodically feed latency reports:
/// let report = io_scheduler.latency_report();
/// limiter.adjust_from_latency(&report);
/// ```
pub struct RateLimiter {
    /// The token bucket controlling I/O rate.
    bucket: Mutex<TokenBucket>,
    /// The calibrated max rate (bytes/sec) — 70% of NVMe bandwidth.
    max_rate: AtomicU64,
    /// The current ceiling (bytes/sec) — may be reduced during throttling.
    current_ceiling: AtomicU64,
    /// Whether the ceiling is currently throttled (reduced by 30%).
    is_throttled: Mutex<bool>,
    /// Configuration.
    config: RateLimiterConfig,
}

impl RateLimiter {
    /// Create a new RateLimiter with the given configuration.
    ///
    /// The limiter starts with the calibrated rate based on
    /// `config.max_bandwidth_bytes_per_sec * config.calibration_fraction`.
    pub fn new(config: RateLimiterConfig) -> Self {
        let initial_rate =
            (config.max_bandwidth_bytes_per_sec as f64 * config.calibration_fraction) as u64;

        Self {
            bucket: Mutex::new(TokenBucket::new(initial_rate as f64)),
            max_rate: AtomicU64::new(initial_rate),
            current_ceiling: AtomicU64::new(initial_rate),
            is_throttled: Mutex::new(false),
            config,
        }
    }

    /// Calibrate the max rate at startup based on measured NVMe write bandwidth.
    ///
    /// Sets `max_rate = measured_bandwidth * calibration_fraction` (default 70%).
    pub fn calibrate(&self, measured_bandwidth_bytes_per_sec: u64) {
        let rate =
            (measured_bandwidth_bytes_per_sec as f64 * self.config.calibration_fraction) as u64;

        tracing::info!(
            measured_bw = measured_bandwidth_bytes_per_sec,
            calibrated_rate = rate,
            fraction = self.config.calibration_fraction,
            "RateLimiter calibrated"
        );

        self.max_rate.store(rate, Ordering::Relaxed);
        self.current_ceiling.store(rate, Ordering::Relaxed);

        let mut bucket = self.bucket.lock().unwrap();
        bucket.set_rate(rate as f64);

        let mut throttled = self.is_throttled.lock().unwrap();
        *throttled = false;
    }

    /// Acquire `bytes` worth of tokens before performing I/O.
    ///
    /// If insufficient tokens are available, this method sleeps until
    /// enough tokens have accumulated. Compaction and flush tasks should
    /// call this before each I/O operation.
    pub async fn acquire(&self, bytes: u64) {
        if bytes == 0 {
            return;
        }

        let wait_secs = {
            let mut bucket = self.bucket.lock().unwrap();
            bucket.try_acquire(bytes)
        };

        if wait_secs <= 0.0 {
            return; // Tokens acquired successfully.
        }

        // Wait for tokens to accumulate.
        let wait_duration =
            std::time::Duration::from_secs_f64(wait_secs.min(1.0));
        tokio::time::sleep(wait_duration).await;

        // After waiting, consume the tokens.
        {
            let mut bucket = self.bucket.lock().unwrap();
            bucket.consume_after_wait(bytes);
        }
    }

    /// Adjust the ceiling based on a latency report from the IoScheduler.
    ///
    /// - If `report.should_throttle` is true (HP P99 exceeded 1.5× baseline
    ///   for 3+ consecutive windows), lower the ceiling by 30%.
    /// - If `report.should_throttle` is false and the ceiling was previously
    ///   lowered, restore it to the calibrated max rate.
    pub fn adjust_from_latency(&self, report: &LatencyReport) {
        if report.should_throttle {
            self.lower_ceiling();
        } else {
            self.restore_ceiling();
        }
    }

    /// Lower the ceiling by the configured throttle reduction fraction (default 30%).
    fn lower_ceiling(&self) {
        let mut throttled = self.is_throttled.lock().unwrap();
        if *throttled {
            // Already throttled — don't reduce further.
            return;
        }

        let max_rate = self.max_rate.load(Ordering::Relaxed);
        let reduction = (max_rate as f64 * self.config.throttle_reduction_fraction) as u64;
        let new_ceiling = max_rate.saturating_sub(reduction);

        tracing::warn!(
            max_rate,
            new_ceiling,
            reduction_pct = self.config.throttle_reduction_fraction * 100.0,
            "RateLimiter lowering ceiling due to HP-queue latency"
        );

        self.current_ceiling.store(new_ceiling, Ordering::Relaxed);
        {
            let mut bucket = self.bucket.lock().unwrap();
            bucket.set_rate(new_ceiling as f64);
        }
        *throttled = true;
    }

    /// Restore the ceiling to the calibrated max rate.
    fn restore_ceiling(&self) {
        let mut throttled = self.is_throttled.lock().unwrap();
        if !*throttled {
            // Not throttled — nothing to restore.
            return;
        }

        let max_rate = self.max_rate.load(Ordering::Relaxed);

        tracing::info!(
            restored_rate = max_rate,
            "RateLimiter restoring ceiling — HP-queue latency normal"
        );

        self.current_ceiling.store(max_rate, Ordering::Relaxed);
        {
            let mut bucket = self.bucket.lock().unwrap();
            bucket.set_rate(max_rate as f64);
        }
        *throttled = false;
    }

    /// Get the current ceiling in bytes/sec.
    pub fn current_ceiling(&self) -> u64 {
        self.current_ceiling.load(Ordering::Relaxed)
    }

    /// Get the calibrated max rate in bytes/sec.
    pub fn max_rate(&self) -> u64 {
        self.max_rate.load(Ordering::Relaxed)
    }

    /// Check whether the limiter is currently in a throttled state.
    pub fn is_throttled(&self) -> bool {
        *self.is_throttled.lock().unwrap()
    }
}

#[cfg(test)]
mod tests;
