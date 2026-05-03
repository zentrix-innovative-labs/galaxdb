//! IoUringScheduler — Linux 5.10+ io_uring-based I/O scheduler.
//!
//! Uses two separate io_uring instances:
//! - HP queue: for user-facing reads/writes (high priority)
//! - BK queue: for compaction and flush (background)
//!
//! The HP queue latency is monitored every 100 ms. If P99 exceeds 1.5× the
//! idle baseline for 3 consecutive windows, the scheduler signals the
//! RateLimiter to throttle background I/O.

use std::fs::{File, OpenOptions};
use std::future::Future;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::pin::Pin;
use std::sync::Mutex;
use std::time::Instant;

use galaxdb_common::{GalaxError, GalaxResult};
use io_uring::{opcode, types, IoUring};

use crate::latency::{LatencyMonitor, LatencyReport};
use crate::scheduler::{IoBackend, IoPriority, IoScheduler};

/// Queue depth for each io_uring instance.
const QUEUE_DEPTH: u32 = 256;

/// A single io_uring instance with its own submission and completion queues.
struct UringQueue {
    /// The io_uring instance.
    ring: IoUring,
}

impl UringQueue {
    fn new() -> io::Result<Self> {
        let ring = IoUring::builder()
            .build(QUEUE_DEPTH)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("io_uring init: {}", e)))?;
        Ok(Self { ring })
    }

    /// Submit a read operation and wait for completion.
    fn read_sync(&mut self, fd: i32, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        let read_op = opcode::Read::new(types::Fd(fd), buf.as_mut_ptr(), buf.len() as u32)
            .offset(offset as _)
            .build()
            .user_data(0x01);

        // Safety: the buffer lives until we get the completion
        unsafe {
            self.ring
                .submission()
                .push(&read_op)
                .map_err(|_| io::Error::new(io::ErrorKind::Other, "io_uring submission full"))?;
        }

        self.ring.submit_and_wait(1)?;

        let cqe = self.ring.completion().next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "no io_uring completion"))?;

        let result = cqe.result();
        if result < 0 {
            Err(io::Error::from_raw_os_error(-result))
        } else {
            Ok(result as usize)
        }
    }

    /// Submit a write operation and wait for completion.
    fn write_sync(&mut self, fd: i32, buf: &[u8], offset: u64) -> io::Result<usize> {
        let write_op = opcode::Write::new(types::Fd(fd), buf.as_ptr(), buf.len() as u32)
            .offset(offset as _)
            .build()
            .user_data(0x02);

        unsafe {
            self.ring
                .submission()
                .push(&write_op)
                .map_err(|_| io::Error::new(io::ErrorKind::Other, "io_uring submission full"))?;
        }

        self.ring.submit_and_wait(1)?;

        let cqe = self.ring.completion().next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "no io_uring completion"))?;

        let result = cqe.result();
        if result < 0 {
            Err(io::Error::from_raw_os_error(-result))
        } else {
            Ok(result as usize)
        }
    }

    /// Submit an fsync operation and wait for completion.
    fn fsync_sync(&mut self, fd: i32) -> io::Result<()> {
        let fsync_op = opcode::Fsync::new(types::Fd(fd))
            .build()
            .user_data(0x03);

        unsafe {
            self.ring
                .submission()
                .push(&fsync_op)
                .map_err(|_| io::Error::new(io::ErrorKind::Other, "io_uring submission full"))?;
        }

        self.ring.submit_and_wait(1)?;

        let cqe = self.ring.completion().next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "no io_uring completion"))?;

        let result = cqe.result();
        if result < 0 {
            Err(io::Error::from_raw_os_error(-result))
        } else {
            Ok(())
        }
    }
}

/// io_uring-based I/O scheduler with separate HP and BK submission queues.
///
/// HP queue handles user-facing reads/writes (point lookups, query reads).
/// BK queue handles compaction, flush, and other background I/O.
/// The two queues are completely independent io_uring instances, providing
/// I/O isolation between OLTP and background workloads.
pub struct IoUringScheduler {
    /// High-priority queue for user-facing I/O.
    hp_queue: Mutex<UringQueue>,
    /// Background queue for compaction and flush I/O.
    bk_queue: Mutex<UringQueue>,
    /// Latency monitor for HP queue feedback to RateLimiter.
    latency: LatencyMonitor,
}

impl IoUringScheduler {
    /// Create a new IoUringScheduler with two io_uring instances.
    pub fn new() -> GalaxResult<Self> {
        let hp_queue = UringQueue::new().map_err(|e| {
            GalaxError::Internal(format!("failed to create HP io_uring queue: {}", e))
        })?;
        let bk_queue = UringQueue::new().map_err(|e| {
            GalaxError::Internal(format!("failed to create BK io_uring queue: {}", e))
        })?;

        tracing::info!(
            queue_depth = QUEUE_DEPTH,
            "IoUringScheduler created with HP and BK queues"
        );

        Ok(Self {
            hp_queue: Mutex::new(hp_queue),
            bk_queue: Mutex::new(bk_queue),
            latency: LatencyMonitor::new(),
        })
    }

    /// Calibrate the idle baseline for HP queue latency monitoring.
    pub fn calibrate_baseline(&self, baseline_us: u64) {
        self.latency.set_idle_baseline(baseline_us);
    }

    /// Get the appropriate queue based on priority.
    fn get_queue(&self, priority: IoPriority) -> &Mutex<UringQueue> {
        match priority {
            IoPriority::High => &self.hp_queue,
            IoPriority::Background => &self.bk_queue,
        }
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
        Box::pin(async move {
            let start = Instant::now();

            // Open the file
            let f = File::open(file)?;
            let fd = f.as_raw_fd();

            let mut buf = vec![0u8; len];
            let queue = self.get_queue(priority);

            // Use io_uring for the read
            let bytes_read = {
                let mut q = queue.lock().expect("uring queue lock poisoned");
                q.read_sync(fd, &mut buf, offset)?
            };

            buf.truncate(bytes_read);

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

            // Open/create the file
            let f = OpenOptions::new()
                .write(true)
                .create(true)
                .open(file)?;
            let fd = f.as_raw_fd();

            let queue = self.get_queue(priority);

            // Use io_uring for the write
            {
                let mut q = queue.lock().expect("uring queue lock poisoned");
                let mut written = 0;
                while written < data.len() {
                    let n = q.write_sync(fd, &data[written..], offset + written as u64)?;
                    if n == 0 {
                        return Err(GalaxError::Io(io::Error::new(
                            io::ErrorKind::WriteZero,
                            "io_uring write returned 0",
                        )));
                    }
                    written += n;
                }
            }

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
            let f = File::open(file)?;
            let fd = f.as_raw_fd();

            // Use HP queue for fsync (it's typically called from the write path)
            let mut q = self.hp_queue.lock().expect("uring queue lock poisoned");
            q.fsync_sync(fd)?;

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
