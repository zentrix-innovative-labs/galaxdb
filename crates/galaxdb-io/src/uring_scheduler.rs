//! IoUringScheduler — Linux 5.10+ io_uring-based I/O scheduler.
//!
//! This module is only compiled on Linux (`#[cfg(target_os = "linux")]`).
//! It uses two separate io_uring instances:
//! - HP queue: for user-facing reads/writes (high priority)
//! - BK queue: for compaction and flush (background)
//!
//! The HP queue latency is monitored every 100 ms. If P99 exceeds 1.5× the
//! idle baseline for 3 consecutive windows, the scheduler signals the
//! RateLimiter to throttle background I/O.
//!
//! NOTE: This is a stub implementation. The actual io_uring integration
//! requires the `io-uring` crate dependency which is currently commented out
//! in the workspace Cargo.toml. When enabled, this module will use real
//! io_uring submission/completion queues.

use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::time::Instant;

use galaxdb_common::{GalaxError, GalaxResult};
use tokio::fs::OpenOptions;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

use crate::latency::{LatencyMonitor, LatencyReport};
use crate::scheduler::{IoBackend, IoPriority, IoScheduler};

/// io_uring-based I/O scheduler with separate HP and BK submission queues.
///
/// On Linux 5.10+, this scheduler uses two io_uring instances to isolate
/// user-facing I/O from background compaction/flush I/O. The HP queue is
/// monitored for latency spikes that trigger RateLimiter throttling.
///
/// Currently uses tokio::fs as a placeholder until the `io-uring` crate
/// dependency is enabled.
pub struct IoUringScheduler {
    /// Latency monitor for HP/BK queue separation.
    latency: LatencyMonitor,
}

impl IoUringScheduler {
    /// Create a new IoUringScheduler.
    ///
    /// In the full implementation, this would initialize two io_uring instances
    /// (HP and BK queues) and calibrate the idle baseline.
    pub fn new() -> GalaxResult<Self> {
        // TODO: Initialize actual io_uring instances when io-uring crate is enabled.
        // For now, use tokio::fs as a fallback on Linux.
        tracing::info!("IoUringScheduler created (stub: using tokio::fs until io-uring crate is enabled)");

        Ok(Self {
            latency: LatencyMonitor::new(),
        })
    }

    /// Calibrate the idle baseline by performing a series of no-op I/O
    /// operations and measuring latency.
    pub fn calibrate_baseline(&self, baseline_us: u64) {
        self.latency.set_idle_baseline(baseline_us);
    }
}

impl IoScheduler for IoUringScheduler {
    fn read<'a>(
        &'a self,
        file: &'a Path,
        offset: u64,
        len: usize,
        priority: IoPriority,
    ) -> Pin<Box<dyn Future<Output = GalaxResult<Vec<u8>>> + Send + 'a>> {
        // TODO: Route to HP or BK io_uring instance based on priority.
        // For now, use tokio::fs for both.
        Box::pin(async move {
            let start = Instant::now();

            let mut f = tokio::fs::File::open(file).await?;
            f.seek(std::io::SeekFrom::Start(offset)).await?;

            let mut buf = vec![0u8; len];
            let mut total_read = 0;
            while total_read < len {
                let n = f.read(&mut buf[total_read..]).await?;
                if n == 0 {
                    buf.truncate(total_read);
                    break;
                }
                total_read += n;
            }

            let elapsed_us = start.elapsed().as_micros() as u64;
            self.latency.record(priority, elapsed_us);

            Ok(buf)
        })
    }

    fn write<'a>(
        &'a self,
        file: &'a Path,
        offset: u64,
        data: &'a [u8],
        priority: IoPriority,
    ) -> Pin<Box<dyn Future<Output = GalaxResult<()>> + Send + 'a>> {
        // TODO: Route to HP or BK io_uring instance based on priority.
        Box::pin(async move {
            let start = Instant::now();

            let mut f = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(false)
                .open(file)
                .await?;

            f.seek(std::io::SeekFrom::Start(offset)).await?;
            f.write_all(data).await?;

            let elapsed_us = start.elapsed().as_micros() as u64;
            self.latency.record(priority, elapsed_us);

            Ok(())
        })
    }

    fn fsync<'a>(
        &'a self,
        file: &'a Path,
    ) -> Pin<Box<dyn Future<Output = GalaxResult<()>> + Send + 'a>> {
        Box::pin(async move {
            let f = tokio::fs::File::open(file).await?;
            f.sync_all().await?;
            Ok(())
        })
    }

    fn latency_report(&self) -> LatencyReport {
        self.latency.report()
    }

    fn backend(&self) -> IoBackend {
        IoBackend::IoUring
    }
}
