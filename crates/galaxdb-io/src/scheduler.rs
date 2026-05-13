//! Core I/O scheduler trait and types.

use std::fmt;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;

use galaxdb_common::GalaxResult;

use crate::LatencyReport;

/// I/O priority level for scheduling reads and writes.
///
/// - `High` — user-facing operations (point lookups, query reads).
/// - `Background` — compaction, flush, and other background I/O.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IoPriority {
    /// High-priority: user-facing reads/writes.
    High,
    /// Background: compaction, flush, maintenance I/O.
    Background,
}

impl fmt::Display for IoPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IoPriority::High => write!(f, "High"),
            IoPriority::Background => write!(f, "Background"),
        }
    }
}

/// Which I/O backend is in use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoBackend {
    /// tokio::fs — macOS, Windows, or forced via env var.
    Tokio,
    /// io_uring — Linux 5.10+ with separate HP/BK queues.
    IoUring,
}

/// Unified async I/O interface for the storage engine.
///
/// Upper layers call these methods without knowing whether the underlying
/// backend is io_uring or tokio. Implementations must be `Send + Sync` so
/// they can be shared across tokio tasks.
pub trait IoScheduler: Send + Sync {
    /// Read `len` bytes from `file` starting at `offset`.
    fn read<'a>(
        &'a self,
        file: &'a Path,
        offset: u64,
        len: usize,
        priority: IoPriority,
    ) -> Pin<Box<dyn Future<Output = GalaxResult<Vec<u8>>> + Send + 'a>>;

    /// Write `data` to `file` starting at `offset`.
    fn write<'a>(
        &'a self,
        file: &'a Path,
        offset: u64,
        data: &'a [u8],
        priority: IoPriority,
    ) -> Pin<Box<dyn Future<Output = GalaxResult<()>> + Send + 'a>>;

    /// Flush file data to stable storage.
    fn fsync<'a>(
        &'a self,
        file: &'a Path,
    ) -> Pin<Box<dyn Future<Output = GalaxResult<()>> + Send + 'a>>;

    /// Synchronous read — for use from non-async contexts (embedded mode,
    /// point lookups from Engine::get()).
    ///
    /// Reads `len` bytes from `file` starting at `offset`.
    /// Default implementation uses std::fs pread-style access.
    /// IoUringScheduler overrides this with io_uring submit+wait.
    fn read_sync(
        &self,
        file: &Path,
        offset: u64,
        len: usize,
        priority: IoPriority,
    ) -> GalaxResult<Vec<u8>> {
        let _ = priority;
        use std::io::{Read, Seek, SeekFrom};
        let mut f = std::fs::File::open(file)?;
        f.seek(SeekFrom::Start(offset))?;
        let mut buf = vec![0u8; len];
        let mut total = 0;
        while total < len {
            let n = f.read(&mut buf[total..])?;
            if n == 0 { break; }
            total += n;
        }
        buf.truncate(total);
        Ok(buf)
    }

    /// Return the latest latency report for RateLimiter feedback.
    fn latency_report(&self) -> LatencyReport;

    /// Return which backend this scheduler uses.
    fn backend(&self) -> IoBackend;
}
