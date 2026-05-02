//! TokioScheduler — async I/O using `tokio::fs`.
//!
//! Used on macOS, Windows, or when `GALAXDB_IO_BACKEND=tokio`. This
//! implementation does not separate HP and BK queues — all I/O goes through
//! tokio's thread pool. Latency is still recorded for observability but there
//! are no HP/BK latency guarantees.

use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::time::Instant;

use galaxdb_common::GalaxResult;
use tokio::fs::OpenOptions;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

use crate::latency::{LatencyMonitor, LatencyReport};
use crate::scheduler::{IoBackend, IoPriority, IoScheduler};

/// Tokio-based I/O scheduler for macOS/Windows fallback.
///
/// Uses `tokio::fs` for all I/O operations. No queue separation between
/// HP and BK priorities — both go through the same tokio thread pool.
pub struct TokioScheduler {
    latency: LatencyMonitor,
}

impl TokioScheduler {
    /// Create a new TokioScheduler.
    pub fn new() -> Self {
        Self {
            latency: LatencyMonitor::new(),
        }
    }
}

impl Default for TokioScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl IoScheduler for TokioScheduler {
    fn read<'a>(
        &'a self,
        file: &'a Path,
        offset: u64,
        len: usize,
        priority: IoPriority,
    ) -> Pin<Box<dyn Future<Output = GalaxResult<Vec<u8>>> + Send + 'a>> {
        Box::pin(async move {
            let start = Instant::now();

            let mut f = tokio::fs::File::open(file).await?;
            f.seek(std::io::SeekFrom::Start(offset)).await?;

            let mut buf = vec![0u8; len];
            let mut total_read = 0;
            while total_read < len {
                let n = f.read(&mut buf[total_read..]).await?;
                if n == 0 {
                    // EOF reached before reading `len` bytes — truncate
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
        IoBackend::Tokio
    }
}
