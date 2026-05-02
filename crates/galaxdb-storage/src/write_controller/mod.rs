//! WriteController — User-write throttle based on pending compaction bytes.
//!
//! The [`WriteController`] manages write admission to prevent write stalls
//! when compaction falls behind. It monitors pending compaction bytes and
//! applies proportional slowdown between a soft limit and a hard limit.
//!
//! - Below the soft limit (default 32 GB): full write throughput.
//! - Between soft and hard limits: proportional slowdown — writes are
//!   delayed by an amount that increases linearly from 0 ms at the soft
//!   limit to `max_delay_ms` (default 100 ms) at the hard limit.
//! - At or above the hard limit (default 64 GB): all writes are blocked
//!   until pending bytes drop below the hard limit.
//! - When pending bytes fall below the soft limit: full throughput is
//!   restored immediately.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Write admission decision returned by [`WriteController::check_write`].
///
/// The caller inspects this value to decide how to proceed:
/// - [`Proceed`](WriteAdmission::Proceed) — write immediately at full speed.
/// - [`Delay`](WriteAdmission::Delay) — sleep for the given duration before writing.
/// - [`Block`](WriteAdmission::Block) — the write is rejected; the caller should
///   retry after a short wait (the controller polls every 1 ms internally).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WriteAdmission {
    /// Write may proceed immediately — pending bytes are below the soft limit.
    Proceed,
    /// Write should be delayed by the specified duration (proportional slowdown).
    Delay(Duration),
    /// Write is blocked — pending bytes are at or above the hard limit.
    /// The caller should retry (e.g. after 1 ms).
    Block,
}

/// Configuration for the [`WriteController`].
#[derive(Debug, Clone)]
pub struct WriteControllerConfig {
    /// Soft limit for pending compaction bytes (default 32 GB).
    /// Above this threshold, writes are proportionally slowed.
    pub soft_limit_bytes: u64,
    /// Hard limit for pending compaction bytes (default 64 GB).
    /// At or above this threshold, all writes are blocked.
    pub hard_limit_bytes: u64,
    /// Maximum delay applied to a single write admission when the
    /// slowdown factor reaches 1.0 (i.e. pending bytes are at the
    /// hard limit). Default: 100 ms.
    pub max_delay_ms: u64,
}

impl Default for WriteControllerConfig {
    fn default() -> Self {
        Self {
            soft_limit_bytes: 32 * 1024 * 1024 * 1024, // 32 GB
            hard_limit_bytes: 64 * 1024 * 1024 * 1024, // 64 GB
            max_delay_ms: 100,
        }
    }
}

/// User-write throttle that manages write admission based on pending
/// compaction bytes.
///
/// The compaction subsystem calls [`WriteController::update_pending_bytes`]
/// to report the current pending compaction byte count, and
/// [`WriteController::reduce_pending_bytes`] after a compaction job
/// completes. User-facing write paths call [`WriteController::check_write`]
/// or [`WriteController::admit_write`] before each write.
///
/// # Example
///
/// ```ignore
/// let config = WriteControllerConfig::default();
/// let controller = WriteController::new(config);
///
/// // Compaction subsystem updates pending bytes periodically:
/// controller.update_pending_bytes(40_000_000_000); // 40 GB
///
/// // Write path checks admission before each write:
/// match controller.check_write() {
///     WriteAdmission::Proceed => { /* write immediately */ }
///     WriteAdmission::Delay(d) => {
///         tokio::time::sleep(d).await;
///         /* then write */
///     }
///     WriteAdmission::Block => { /* retry later */ }
/// }
///
/// // Or use the convenience async method:
/// controller.admit_write().await;
/// // ... perform the write ...
/// ```
pub struct WriteController {
    /// Current pending compaction bytes, updated by the compaction subsystem.
    pending_compaction_bytes: AtomicU64,
    /// Configuration (limits and max delay).
    config: WriteControllerConfig,
}

impl WriteController {
    /// Create a new `WriteController` with the given configuration.
    pub fn new(config: WriteControllerConfig) -> Self {
        Self {
            pending_compaction_bytes: AtomicU64::new(0),
            config,
        }
    }

    /// Set the pending compaction byte count to an absolute value.
    ///
    /// Called by the compaction subsystem whenever the pending byte
    /// estimate changes (typically every 1 ms or after each compaction
    /// job completes).
    pub fn update_pending_bytes(&self, bytes: u64) {
        self.pending_compaction_bytes.store(bytes, Ordering::Relaxed);
    }

    /// Reduce the pending compaction byte count by `bytes`.
    ///
    /// Called by the compaction subsystem after a compaction job completes
    /// to reflect the reduction in pending work. The count is clamped to
    /// zero (never goes negative).
    pub fn reduce_pending_bytes(&self, bytes: u64) {
        self.pending_compaction_bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_sub(bytes))
            })
            .ok();
    }

    /// Alias for [`update_pending_bytes`](Self::update_pending_bytes) for
    /// backward compatibility.
    pub fn update_pending(&self, bytes: u64) {
        self.update_pending_bytes(bytes);
    }

    /// Get the current pending compaction bytes.
    pub fn pending_compaction_bytes(&self) -> u64 {
        self.pending_compaction_bytes.load(Ordering::Relaxed)
    }

    /// Check whether a write should proceed, be delayed, or be blocked.
    ///
    /// Returns a [`WriteAdmission`] decision based on the current pending
    /// compaction bytes relative to the configured soft and hard limits:
    ///
    /// - `pending < soft_limit` → [`Proceed`](WriteAdmission::Proceed)
    /// - `soft_limit <= pending < hard_limit` → [`Delay(d)`](WriteAdmission::Delay)
    ///   where `d` is proportional to the excess above the soft limit
    /// - `pending >= hard_limit` → [`Block`](WriteAdmission::Block)
    pub fn check_write(&self) -> WriteAdmission {
        let pending = self.pending_compaction_bytes.load(Ordering::Relaxed);

        if pending < self.config.soft_limit_bytes {
            return WriteAdmission::Proceed;
        }

        if pending >= self.config.hard_limit_bytes {
            return WriteAdmission::Block;
        }

        // Between soft and hard limits — proportional slowdown.
        let factor = self.compute_slowdown_factor(pending);
        let delay_ms = (factor * self.config.max_delay_ms as f64) as u64;

        if delay_ms == 0 {
            WriteAdmission::Proceed
        } else {
            WriteAdmission::Delay(Duration::from_millis(delay_ms))
        }
    }

    /// Compute the slowdown factor based on current pending compaction bytes.
    ///
    /// Returns a value in `[0.0, 1.0]`:
    /// - `0.0` when pending bytes are at or below the soft limit (no slowdown).
    /// - Linearly increases to `1.0` as pending bytes approach the hard limit.
    /// - `1.0` when pending bytes are at or above the hard limit (full stop).
    pub fn slowdown_factor(&self) -> f64 {
        let pending = self.pending_compaction_bytes.load(Ordering::Relaxed);
        self.compute_slowdown_factor(pending)
    }

    /// Admit a write operation, applying proportional delay or blocking
    /// as needed based on the current pending compaction bytes.
    ///
    /// - Below soft limit: returns immediately (full throughput).
    /// - Between soft and hard limits: sleeps for a proportional delay.
    /// - At or above hard limit: blocks (polls every 1 ms) until pending
    ///   bytes drop below the hard limit, then applies any remaining
    ///   proportional delay.
    pub async fn admit_write(&self) {
        let pending = self.pending_compaction_bytes.load(Ordering::Relaxed);

        if pending < self.config.soft_limit_bytes {
            // Below soft limit — full speed.
            return;
        }

        if pending >= self.config.hard_limit_bytes {
            // Hard stop — block until pending drops below hard limit.
            tracing::warn!(
                pending_bytes = pending,
                hard_limit = self.config.hard_limit_bytes,
                "WriteController: hard stop — blocking all writes"
            );

            loop {
                tokio::time::sleep(Duration::from_millis(1)).await;
                let current = self.pending_compaction_bytes.load(Ordering::Relaxed);
                if current < self.config.hard_limit_bytes {
                    // Dropped below hard limit. Check if we still need
                    // proportional delay.
                    let factor = self.compute_slowdown_factor(current);
                    if factor > 0.0 {
                        let delay_ms = (factor * self.config.max_delay_ms as f64) as u64;
                        if delay_ms > 0 {
                            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                        }
                    }
                    return;
                }
            }
        }

        // Between soft and hard limits — proportional slowdown.
        let factor = self.compute_slowdown_factor(pending);
        let delay_ms = (factor * self.config.max_delay_ms as f64) as u64;

        if delay_ms > 0 {
            tracing::debug!(
                pending_bytes = pending,
                slowdown_factor = factor,
                delay_ms,
                "WriteController: proportional slowdown"
            );
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
        }
    }

    /// Returns `true` if writes are currently hard-stopped (pending >= hard limit).
    pub fn is_stopped(&self) -> bool {
        self.pending_compaction_bytes.load(Ordering::Relaxed) >= self.config.hard_limit_bytes
    }

    /// Returns `true` if writes are currently being slowed down
    /// (pending >= soft limit).
    pub fn is_throttled(&self) -> bool {
        self.pending_compaction_bytes.load(Ordering::Relaxed) >= self.config.soft_limit_bytes
    }

    /// Get the soft limit in bytes.
    pub fn soft_limit(&self) -> u64 {
        self.config.soft_limit_bytes
    }

    /// Get the hard limit in bytes.
    pub fn hard_limit(&self) -> u64 {
        self.config.hard_limit_bytes
    }

    /// Internal: compute the slowdown factor for a given pending byte count.
    fn compute_slowdown_factor(&self, pending: u64) -> f64 {
        if pending <= self.config.soft_limit_bytes {
            return 0.0;
        }
        if pending >= self.config.hard_limit_bytes {
            return 1.0;
        }
        // Linear interpolation between soft and hard limits.
        let range = self.config.hard_limit_bytes - self.config.soft_limit_bytes;
        if range == 0 {
            return 1.0;
        }
        let excess = pending - self.config.soft_limit_bytes;
        excess as f64 / range as f64
    }
}

#[cfg(test)]
mod tests;
