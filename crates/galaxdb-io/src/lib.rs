//! GalaxDB I/O — I/O abstraction: io_uring (Linux) / tokio (macOS/Windows).
//!
//! This crate provides a unified async I/O interface ([`IoScheduler`]) that
//! upper layers use without caring about the underlying backend. On Linux 5.10+
//! with io_uring available, the [`IoUringScheduler`] uses separate HP and BK
//! submission queues. On macOS, Windows, or when `GALAXDB_IO_BACKEND=tokio`,
//! the [`TokioScheduler`] uses `tokio::fs`.
//!
//! The [`select_scheduler`] function performs startup detection and returns the
//! appropriate implementation.

mod latency;
mod scheduler;
mod tokio_scheduler;

#[cfg(target_os = "linux")]
mod uring_scheduler;

pub use latency::{LatencyMonitor, LatencyReport};
pub use scheduler::{IoBackend, IoPriority, IoScheduler};
pub use tokio_scheduler::TokioScheduler;

#[cfg(target_os = "linux")]
pub use uring_scheduler::IoUringScheduler;

use galaxdb_common::GalaxResult;

/// Detect the platform and environment to select the appropriate I/O scheduler.
///
/// Selection logic:
/// 1. If `GALAXDB_IO_BACKEND=tokio` → [`TokioScheduler`]
/// 2. If Linux 5.10+ with io_uring available → [`IoUringScheduler`] (Linux only)
/// 3. Otherwise → [`TokioScheduler`]
pub fn select_scheduler() -> GalaxResult<Box<dyn IoScheduler>> {
    let backend = detect_backend();
    tracing::info!(?backend, "selected I/O backend");

    match backend {
        IoBackend::Tokio => Ok(Box::new(TokioScheduler::new())),
        #[cfg(target_os = "linux")]
        IoBackend::IoUring => Ok(Box::new(IoUringScheduler::new()?)),
        #[cfg(not(target_os = "linux"))]
        IoBackend::IoUring => {
            tracing::warn!("io_uring requested but not available on this platform, falling back to tokio");
            Ok(Box::new(TokioScheduler::new()))
        }
    }
}

/// Detect which I/O backend should be used based on platform and environment.
pub fn detect_backend() -> IoBackend {
    // Check env var override first
    if let Ok(val) = std::env::var("GALAXDB_IO_BACKEND") {
        if val.eq_ignore_ascii_case("tokio") {
            return IoBackend::Tokio;
        }
    }

    // Platform detection
    #[cfg(target_os = "linux")]
    {
        if is_io_uring_available() {
            return IoBackend::IoUring;
        }
    }

    IoBackend::Tokio
}

/// Check if io_uring is available on the current Linux system (kernel 5.10+).
#[cfg(target_os = "linux")]
fn is_io_uring_available() -> bool {
    use std::fs;

    // Check kernel version >= 5.10
    if let Ok(version) = fs::read_to_string("/proc/version") {
        if let Some(ver_str) = version.split_whitespace().nth(2) {
            let parts: Vec<&str> = ver_str.split('.').collect();
            if parts.len() >= 2 {
                if let (Ok(major), Ok(minor)) = (
                    parts[0].parse::<u32>(),
                    parts[1].parse::<u32>(),
                ) {
                    return major > 5 || (major == 5 && minor >= 10);
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests;
