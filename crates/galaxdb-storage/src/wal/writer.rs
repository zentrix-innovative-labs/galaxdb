//! WAL writer with group commit, durability modes, checkpoint, and recovery.
//!
//! The [`WalWriter`] manages a single WAL file on disk. It supports two
//! durability modes:
//!
//! - **STRICT**: each `append` call fsyncs immediately (no batching).
//! - **RELAXED**: writes are batched and a background task fsyncs once per
//!   configurable interval (default 10 ms).
//!
//! Checkpoint is triggered when the WAL exceeds a size threshold (default
//! 512 MB) or a time threshold (default 60 s). Recovery replays from the
//! last CHECKPOINT record, verifying checksums per record.

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{mpsc, oneshot, Mutex as TokioMutex, Notify};
use tracing;

use super::record::{WalRecord, WalRecordType};

/// Configuration for the WAL writer.
#[derive(Debug, Clone)]
pub struct WalWriterConfig {
    /// Path to the WAL file.
    pub wal_path: PathBuf,
    /// Group commit flush interval (default: 10 ms).
    pub group_commit_interval: Duration,
    /// Checkpoint trigger: WAL size in bytes (default: 512 MB).
    pub checkpoint_size_bytes: u64,
    /// Checkpoint trigger: seconds since last checkpoint (default: 60).
    pub checkpoint_interval: Duration,
}

impl Default for WalWriterConfig {
    fn default() -> Self {
        Self {
            wal_path: PathBuf::from("galaxdb_data/wal.log"),
            group_commit_interval: Duration::from_millis(10),
            checkpoint_size_bytes: 512 * 1024 * 1024,
            checkpoint_interval: Duration::from_secs(60),
        }
    }
}

/// Durability mode for WAL writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurabilityMode {
    /// Fsync each commit individually — no group commit batching.
    Strict,
    /// Use group commit with the configured batch window.
    Relaxed,
}

/// Information about the last checkpoint.
#[derive(Debug, Clone)]
pub struct CheckpointInfo {
    /// Sequence number of the checkpoint record.
    pub seq_no: u64,
    /// Byte offset in the WAL file where the checkpoint record starts.
    pub offset: u64,
    /// When the checkpoint was created.
    pub timestamp: Instant,
}

/// A write request sent to the group commit background task.
struct GroupCommitRequest {
    /// The serialized record bytes to write.
    data: Vec<u8>,
    /// Channel to notify the caller when the write (and fsync) is complete.
    done: oneshot::Sender<io::Result<()>>,
}

/// The WAL writer manages appending records, group commit, and checkpointing.
pub struct WalWriter {
    /// Configuration.
    config: WalWriterConfig,
    /// Next sequence number (monotonically increasing).
    next_seq_no: AtomicU64,
    /// Current WAL file size in bytes.
    current_size: AtomicU64,
    /// The file handle for async (STRICT mode) writes.
    file: TokioMutex<BufWriter<File>>,
    /// The file handle for sync writes (embedded mode).
    sync_file: std::sync::Mutex<BufWriter<File>>,
    /// Channel to send writes to the group commit background task.
    group_commit_tx: mpsc::UnboundedSender<GroupCommitRequest>,
    /// Last checkpoint info.
    last_checkpoint: TokioMutex<Option<CheckpointInfo>>,
    /// Whether the group commit task is running.
    running: Arc<AtomicBool>,
    /// Notify handle to wake the group commit task for immediate flush.
    flush_notify: Arc<Notify>,
}

impl WalWriter {
    /// Create a new WAL writer, opening (or creating) the WAL file.
    ///
    /// This also spawns the group commit background task.
    pub fn new(config: WalWriterConfig) -> io::Result<Self> {
        // Ensure parent directory exists
        if let Some(parent) = config.wal_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&config.wal_path)?;

        let file_size = file.metadata()?.len();

        let file_for_group = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&config.wal_path)?;

        let file_for_sync = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&config.wal_path)?;

        let running = Arc::new(AtomicBool::new(true));
        let flush_notify = Arc::new(Notify::new());

        let (tx, rx) = mpsc::unbounded_channel();

        // Spawn the group commit background task on a dedicated OS thread
        // with its own tokio runtime. This ensures the WAL works in both
        // embedded mode (no external runtime) and server mode (existing runtime).
        let group_commit_interval = config.group_commit_interval;
        let running_clone = running.clone();
        let flush_notify_clone = flush_notify.clone();

        std::thread::Builder::new()
            .name("galaxdb-wal-group-commit".to_string())
            .spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_time()
                    .build()
                    .expect("failed to create WAL group commit runtime");
                rt.block_on(async move {
                    group_commit_task(
                        file_for_group,
                        rx,
                        group_commit_interval,
                        running_clone,
                        flush_notify_clone,
                    )
                    .await;
                });
            })
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        Ok(Self {
            config,
            next_seq_no: AtomicU64::new(1),
            current_size: AtomicU64::new(file_size),
            file: TokioMutex::new(BufWriter::new(file)),
            sync_file: std::sync::Mutex::new(BufWriter::new(file_for_sync)),
            group_commit_tx: tx,
            last_checkpoint: TokioMutex::new(None),
            running,
            flush_notify,
        })
    }

    /// Append a record to the WAL with the given durability mode.
    ///
    /// - **STRICT**: writes and fsyncs immediately under a lock.
    /// - **RELAXED**: sends the write to the group commit task and waits for
    ///   the next batch fsync.
    pub async fn append(
        &self,
        record_type: WalRecordType,
        payload: Vec<u8>,
        durability: DurabilityMode,
    ) -> io::Result<u64> {
        let seq_no = self.next_seq_no.fetch_add(1, Ordering::SeqCst);
        let record = WalRecord::new(record_type, seq_no, payload);
        let data = record.serialize();
        let data_len = data.len() as u64;

        match durability {
            DurabilityMode::Strict => {
                let mut file = self.file.lock().await;
                file.write_all(&data)?;
                file.flush()?;
                file.get_mut().sync_all()?;
                self.current_size.fetch_add(data_len, Ordering::SeqCst);
            }
            DurabilityMode::Relaxed => {
                let (done_tx, done_rx) = oneshot::channel();
                self.group_commit_tx
                    .send(GroupCommitRequest {
                        data,
                        done: done_tx,
                    })
                    .map_err(|_| {
                        io::Error::new(io::ErrorKind::BrokenPipe, "group commit task stopped")
                    })?;

                done_rx.await.map_err(|_| {
                    io::Error::new(io::ErrorKind::BrokenPipe, "group commit response lost")
                })??;

                self.current_size.fetch_add(data_len, Ordering::SeqCst);
            }
        }

        Ok(seq_no)
    }

    /// Append a record synchronously (no tokio runtime required).
    ///
    /// Sends the write to the group commit background thread and blocks
    /// until the batch fsync completes. This gives group commit benefits
    /// (batched fsyncs) without requiring a tokio runtime in the caller.
    pub fn append_sync(
        &self,
        record_type: WalRecordType,
        payload: Vec<u8>,
    ) -> io::Result<u64> {
        let seq_no = self.next_seq_no.fetch_add(1, Ordering::SeqCst);
        let record = WalRecord::new(record_type, seq_no, payload);
        let data = record.serialize();
        let data_len = data.len() as u64;

        // Send to group commit thread via channel
        let (done_tx, done_rx) = oneshot::channel();
        self.group_commit_tx
            .send(GroupCommitRequest {
                data,
                done: done_tx,
            })
            .map_err(|_| {
                io::Error::new(io::ErrorKind::BrokenPipe, "group commit task stopped")
            })?;

        // Block-wait for the group commit to complete
        // The group commit thread has its own tokio runtime, so this is safe
        done_rx.blocking_recv().map_err(|_| {
            io::Error::new(io::ErrorKind::BrokenPipe, "group commit response lost")
        })??;

        self.current_size.fetch_add(data_len, Ordering::SeqCst);
        Ok(seq_no)
    }

    /// Return the next sequence number that will be assigned.
    pub fn next_seq_no(&self) -> u64 {
        self.next_seq_no.load(Ordering::SeqCst)
    }

    /// Return the current WAL file size in bytes.
    pub fn current_size(&self) -> u64 {
        self.current_size.load(Ordering::SeqCst)
    }

    /// Check whether a checkpoint should be triggered based on size or time.
    pub async fn should_checkpoint(&self) -> bool {
        let size_exceeded = self.current_size() >= self.config.checkpoint_size_bytes;

        let time_exceeded = {
            let last_cp = self.last_checkpoint.lock().await;
            match last_cp.as_ref() {
                Some(info) => info.timestamp.elapsed() >= self.config.checkpoint_interval,
                None => {
                    // No checkpoint yet — trigger if WAL has any data
                    self.current_size() > 0
                }
            }
        };

        size_exceeded || time_exceeded
    }

    /// Write a CHECKPOINT record and update the checkpoint state.
    ///
    /// The caller is responsible for flushing the memtable before calling this.
    /// After writing the checkpoint record, the WAL can be truncated up to this
    /// point (truncation is handled by `truncate_to_checkpoint`).
    pub async fn write_checkpoint(&self) -> io::Result<u64> {
        let offset = self.current_size();
        let seq_no = self
            .append(WalRecordType::Checkpoint, Vec::new(), DurabilityMode::Strict)
            .await?;

        let mut last_cp = self.last_checkpoint.lock().await;
        *last_cp = Some(CheckpointInfo {
            seq_no,
            offset,
            timestamp: Instant::now(),
        });

        tracing::info!(seq_no, offset, "WAL checkpoint written");
        Ok(seq_no)
    }

    /// Truncate the WAL file, keeping only records after the last checkpoint.
    ///
    /// This rewrites the WAL file to contain only the checkpoint record and
    /// any records that follow it.
    pub async fn truncate_to_checkpoint(&self) -> io::Result<()> {
        let last_cp = self.last_checkpoint.lock().await;
        let cp_info = match last_cp.as_ref() {
            Some(info) => info.clone(),
            None => return Ok(()), // No checkpoint to truncate to
        };
        drop(last_cp);

        // Read all records from the checkpoint offset onward
        let remaining = {
            let mut f = File::open(&self.config.wal_path)?;
            f.seek(SeekFrom::Start(cp_info.offset))?;
            let mut buf = Vec::new();
            std::io::Read::read_to_end(&mut f, &mut buf)?;
            buf
        };

        // Rewrite the WAL file with only the remaining data
        {
            let mut file = self.file.lock().await;
            let inner = file.get_mut();

            // Truncate and rewrite
            inner.set_len(0)?;
            inner.seek(SeekFrom::Start(0))?;
            inner.write_all(&remaining)?;
            inner.sync_all()?;

            // Reset the BufWriter
            *file = BufWriter::new(inner.try_clone()?);
        }

        let new_size = remaining.len() as u64;
        self.current_size.store(new_size, Ordering::SeqCst);

        // Update checkpoint offset to 0 since we truncated
        let mut last_cp = self.last_checkpoint.lock().await;
        if let Some(ref mut info) = *last_cp {
            info.offset = 0;
        }

        tracing::info!(
            new_size,
            checkpoint_seq_no = cp_info.seq_no,
            "WAL truncated to checkpoint"
        );

        Ok(())
    }

    /// Shut down the group commit background task.
    pub fn shutdown(&self) {
        self.running.store(false, Ordering::SeqCst);
        self.flush_notify.notify_one();
    }

    /// Get the last checkpoint info.
    pub async fn last_checkpoint(&self) -> Option<CheckpointInfo> {
        self.last_checkpoint.lock().await.clone()
    }
}

impl Drop for WalWriter {
    fn drop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        self.flush_notify.notify_one();
    }
}

/// Background task that batches WAL writes and fsyncs once per interval.
async fn group_commit_task(
    file: File,
    mut rx: mpsc::UnboundedReceiver<GroupCommitRequest>,
    interval: Duration,
    running: Arc<AtomicBool>,
    flush_notify: Arc<Notify>,
) {
    let mut writer = BufWriter::new(file);
    let mut pending: Vec<oneshot::Sender<io::Result<()>>> = Vec::new();

    loop {
        // Wait for either the interval or a flush notification
        let deadline = tokio::time::sleep(interval);
        tokio::pin!(deadline);

        // Collect writes until the interval expires or we're notified
        loop {
            tokio::select! {
                biased;
                req = rx.recv() => {
                    match req {
                        Some(req) => {
                            let result = writer.write_all(&req.data);
                            if let Err(e) = result {
                                let _ = req.done.send(Err(io::Error::new(e.kind(), e.to_string())));
                            } else {
                                pending.push(req.done);
                            }
                        }
                        None => {
                            // Channel closed — flush remaining and exit
                            let _ = flush_and_notify(&mut writer, &mut pending);
                            return;
                        }
                    }
                }
                _ = &mut deadline => {
                    break;
                }
                _ = flush_notify.notified() => {
                    if !running.load(Ordering::SeqCst) {
                        let _ = flush_and_notify(&mut writer, &mut pending);
                        return;
                    }
                    break;
                }
            }
        }

        // Flush and fsync the batch
        if !pending.is_empty() {
            let _ = flush_and_notify(&mut writer, &mut pending);
        }

        if !running.load(Ordering::SeqCst) {
            return;
        }
    }
}

/// Flush the writer, fsync, and notify all pending callers.
fn flush_and_notify(
    writer: &mut BufWriter<File>,
    pending: &mut Vec<oneshot::Sender<io::Result<()>>>,
) -> io::Result<()> {
    let result = writer.flush().and_then(|_| writer.get_mut().sync_all());

    let status = match &result {
        Ok(()) => Ok(()),
        Err(e) => Err(io::Error::new(e.kind(), e.to_string())),
    };

    for sender in pending.drain(..) {
        let send_result = match &status {
            Ok(()) => Ok(()),
            Err(e) => Err(io::Error::new(e.kind(), e.to_string())),
        };
        let _ = sender.send(send_result);
    }

    result
}

/// Recover WAL records from the last CHECKPOINT.
///
/// Reads the entire WAL file, finds the last CHECKPOINT record, then replays
/// all records after it. For each record, the XXH3-64 checksum is verified.
/// Corrupt records cause recovery to stop (records before the corruption are
/// returned).
///
/// Returns the recovered records and the sequence number to resume from.
#[allow(dead_code)]
pub fn recover_wal(wal_path: &Path) -> io::Result<(Vec<WalRecord>, u64)> {
    let file = match File::open(wal_path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            return Ok((Vec::new(), 1));
        }
        Err(e) => return Err(e),
    };

    let metadata = file.metadata()?;
    if metadata.len() == 0 {
        return Ok((Vec::new(), 1));
    }

    let mut reader = BufReader::new(file);

    // First pass: read all records and find the last checkpoint
    let mut all_records: Vec<(WalRecord, usize)> = Vec::new(); // (record, index)
    let mut last_checkpoint_idx: Option<usize> = None;
    let mut max_seq_no: u64 = 0;

    loop {
        match WalRecord::deserialize(&mut reader) {
            Ok(Some(record)) => {
                let idx = all_records.len();
                if record.seq_no > max_seq_no {
                    max_seq_no = record.seq_no;
                }
                if record.record_type == WalRecordType::Checkpoint {
                    last_checkpoint_idx = Some(idx);
                }
                all_records.push((record, idx));
            }
            Ok(None) => break, // Clean EOF
            Err(e) if e.kind() == io::ErrorKind::InvalidData => {
                // Checksum failure or corrupt record — stop here
                tracing::warn!(error = %e, "WAL recovery: stopping at corrupt record");
                break;
            }
            Err(e) => return Err(e),
        }
    }

    // Replay from after the last checkpoint (or from the beginning if none)
    let start_idx = match last_checkpoint_idx {
        Some(idx) => idx + 1, // Skip the checkpoint record itself
        None => 0,
    };

    let recovered: Vec<WalRecord> = all_records
        .into_iter()
        .skip(start_idx)
        .map(|(record, _)| record)
        .collect();

    let next_seq_no = max_seq_no + 1;

    tracing::info!(
        recovered_count = recovered.len(),
        next_seq_no,
        had_checkpoint = last_checkpoint_idx.is_some(),
        "WAL recovery complete"
    );

    Ok((recovered, next_seq_no))
}
