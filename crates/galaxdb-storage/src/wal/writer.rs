//! WAL writer: pre-allocated file + inline two-lock group commit.
//!
//! This is PostgreSQL's WAL model adapted to a single recycled, pre-allocated
//! file, implemented in safe Rust:
//!
//! 1. **Pre-allocation.** The file is filled with real zero blocks and
//!    fdatasync'd once at open. Because the file size and extent map never
//!    change during steady-state writes, each commit's `fdatasync` only
//!    flushes dirty *data* pages — no per-write extent allocation or inode
//!    metadata journaling. Measured on AWS c6id NVMe: an inline
//!    write+fdatasync on a pre-allocated file is ~0.037 ms (≈25k/s), versus
//!    ~2.7 ms when appending to a growing file.
//!
//! 2. **Inline commit, no thread handoff.** The committing thread writes its
//!    record and fdatasyncs itself (PostgreSQL backends do the same). A
//!    cross-thread channel hand-off would add two context switches (~0.8 ms)
//!    per commit and was the dominant cost in the previous design.
//!
//! 3. **Two-lock group commit.** An *insert* lock guards appending bytes to
//!    the file and advancing `write_offset`; a *flush* lock guards the
//!    `fdatasync` and advancing `flush_offset`. A committer appends under the
//!    insert lock, then takes the flush lock — but first checks whether
//!    `flush_offset` already covers its bytes (another committer's fdatasync
//!    flushed them). This is PostgreSQL's WALInsertLock + WALWriteLock: under
//!    concurrency, one fdatasync makes many commits durable.
//!
//! A zero `type` byte is an invalid record discriminant, so recovery stops at
//! the zero padding without needing an explicit end marker.

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tokio::sync::Mutex as TokioMutex;

use galaxdb_common::format::{self, FormatHeader, FORMAT_HEADER_SIZE};

use super::record::{WalRecord, WalRecordType};

/// Size of the WAL superblock (a `format::WAL` header) written at offset 0 of a
/// versioned WAL file. Records begin immediately after it.
const WAL_HEADER_SIZE: u64 = FORMAT_HEADER_SIZE as u64;

/// Map a typed format error to an `io::Error` for the WAL's `io::Result` API.
fn format_err_to_io(e: galaxdb_common::GalaxError) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e.to_string())
}

/// Determine where WAL records begin in an existing file.
///
/// Returns `WAL_HEADER_SIZE` if the file opens with a valid, in-range versioned
/// superblock (`format::WAL`), or `0` for a legacy/headerless file (records from
/// offset 0, treated as format v1) and for a missing/empty file. A superblock
/// whose version is out of range surfaces a typed `FormatTooOld`/`FormatTooNew`
/// (mapped to `InvalidData`) — the rollback-safety refusal.
fn wal_data_start(path: &Path) -> io::Result<u64> {
    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(e),
    };
    let mut magic = [0u8; 4];
    match file.read_exact(&mut magic) {
        Ok(()) => {}
        // Fewer than 4 bytes: empty/tiny file → no header.
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(0),
        Err(e) => return Err(e),
    }
    if magic != format::WAL.magic {
        // Legacy records (first byte is a record type 0x01–0x06) or a zero-fill
        // prefix — either way, no versioned superblock. Read from offset 0.
        return Ok(0);
    }
    // Versioned superblock: parse + range-check the full 16-byte header.
    let mut buf = [0u8; FORMAT_HEADER_SIZE];
    file.seek(SeekFrom::Start(0))?;
    file.read_exact(&mut buf)?;
    let header = FormatHeader::from_bytes(&buf, format::WAL.magic).map_err(format_err_to_io)?;
    format::WAL.check(header.format_version).map_err(format_err_to_io)?;
    Ok(WAL_HEADER_SIZE)
}

/// Default WAL pre-allocation size: 64 MiB written as real zero blocks.
pub const DEFAULT_WAL_PREALLOCATE_BYTES: u64 = 64 * 1024 * 1024;

/// Configuration for the WAL writer.
#[derive(Debug, Clone)]
pub struct WalWriterConfig {
    /// Path to the WAL file.
    pub wal_path: PathBuf,
    /// Retained for API/config compatibility. The writer flushes inline with
    /// no fixed wait, so this is not used in the hot path.
    pub group_commit_interval: Duration,
    /// Checkpoint trigger: WAL size in bytes (default: 512 MB).
    pub checkpoint_size_bytes: u64,
    /// Checkpoint trigger: seconds since last checkpoint (default: 60).
    pub checkpoint_interval: Duration,
    /// Bytes to pre-allocate (zero-fill); the file grows by this much when
    /// it would otherwise overflow.
    pub preallocate_bytes: u64,
}

impl Default for WalWriterConfig {
    fn default() -> Self {
        Self {
            wal_path: PathBuf::from("galaxdb_data/wal.log"),
            group_commit_interval: Duration::from_millis(10),
            checkpoint_size_bytes: 512 * 1024 * 1024,
            checkpoint_interval: Duration::from_secs(60),
            preallocate_bytes: DEFAULT_WAL_PREALLOCATE_BYTES,
        }
    }
}

/// Durability mode. Both modes are durable-on-return; retained for API
/// compatibility (the inline group commit makes them behave identically).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurabilityMode {
    /// Fsync each commit (durable on return).
    Strict,
    /// Group-committed (durable on return; batched with concurrent writers).
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

/// State guarded by the *insert* lock: the write fd and where the next record
/// goes. Sequential writes keep the fd cursor equal to `write_offset`.
struct WriteState {
    file: File,
    write_offset: u64,
    file_len: u64,
    /// Byte offset where records begin: `WAL_HEADER_SIZE` for a versioned file,
    /// `0` for a legacy/headerless one. Legacy files migrate to versioned on
    /// the next `truncate_to_checkpoint` rewrite.
    data_start: u64,
}

/// The WAL writer.
pub struct WalWriter {
    config: WalWriterConfig,
    next_seq_no: AtomicU64,
    /// Bytes appended to the file (may not yet be durable).
    write_offset: AtomicU64,
    /// Bytes that are durable (fdatasync'd).
    flush_offset: AtomicU64,
    /// Insert lock: append bytes + advance `write_offset`.
    write_state: Mutex<WriteState>,
    /// Flush lock: holds a second fd used only for `fdatasync`. fsync/fdatasync
    /// act on the inode, so syncing through this fd flushes the writer fd's
    /// dirty pages too.
    sync_file: Mutex<File>,
    /// Last checkpoint info.
    last_checkpoint: TokioMutex<Option<CheckpointInfo>>,
    /// How much to grow the file by when extending.
    prealloc_chunk: u64,
}

/// Write `count` real zero bytes at `from_offset` (allocates concrete blocks,
/// unlike `set_len` which makes a sparse hole). Leaves the cursor past the
/// written region; callers seek afterwards.
fn zero_fill(file: &mut File, from_offset: u64, count: u64) -> io::Result<()> {
    file.seek(SeekFrom::Start(from_offset))?;
    let chunk = vec![0u8; 1024 * 1024];
    let mut remaining = count;
    while remaining > 0 {
        let n = remaining.min(chunk.len() as u64) as usize;
        file.write_all(&chunk[..n])?;
        remaining -= n as u64;
    }
    Ok(())
}

/// Scan an existing WAL to find the logical end (offset just past the last
/// valid record). Stops at the first invalid/zero record or EOF.
fn scan_logical_end(path: &Path, data_start: u64) -> io::Result<u64> {
    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(data_start),
        Err(e) => return Err(e),
    };
    // Skip the superblock (if any) so scanning starts at the first record.
    if data_start > 0 {
        file.seek(SeekFrom::Start(data_start))?;
    }
    let mut reader = BufReader::new(file);
    let mut offset: u64 = data_start;
    loop {
        match WalRecord::deserialize(&mut reader) {
            Ok(Some(rec)) => offset += rec.serialize().len() as u64,
            Ok(None) => break,
            Err(_) => break,
        }
    }
    Ok(offset)
}

impl WalWriter {
    /// Open (or create) and pre-allocate the WAL file.
    pub fn new(config: WalWriterConfig) -> io::Result<Self> {
        if let Some(parent) = config.wal_path.parent() {
            fs::create_dir_all(parent)?;
        }

        // A brand-new (missing or zero-length) WAL gets a versioned superblock;
        // an existing file is opened as-is (versioned if it already has a header,
        // else legacy from offset 0).
        let is_fresh = match fs::metadata(&config.wal_path) {
            Ok(m) => m.len() == 0,
            Err(e) if e.kind() == io::ErrorKind::NotFound => true,
            Err(e) => return Err(e),
        };
        let mut data_start = if is_fresh {
            WAL_HEADER_SIZE
        } else {
            wal_data_start(&config.wal_path)?
        };

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false) // never truncate on open — WAL recovery reads prior records
            .open(&config.wal_path)?;

        if is_fresh {
            // Stamp the superblock at offset 0; records begin at WAL_HEADER_SIZE.
            let header = FormatHeader::new(format::WAL.magic, format::WAL.current_write).to_bytes();
            file.seek(SeekFrom::Start(0))?;
            file.write_all(&header)?;
            file.sync_all()?;
            data_start = WAL_HEADER_SIZE;
        }

        let logical_end = scan_logical_end(&config.wal_path, data_start)?;

        // Pre-allocate real zero blocks up to at least `preallocate_bytes`.
        let cur_len = file.metadata()?.len();
        let want_len = logical_end.max(config.preallocate_bytes);
        if cur_len < want_len {
            zero_fill(&mut file, cur_len, want_len - cur_len)?;
            file.sync_all()?;
        }
        let file_len = file.metadata()?.len();
        // Position cursor at the logical end: new records overwrite zeros.
        file.seek(SeekFrom::Start(logical_end))?;

        // Second fd, used only for fdatasync.
        let sync_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&config.wal_path)?;

        let prealloc_chunk = config.preallocate_bytes.max(1024 * 1024);

        Ok(Self {
            config,
            next_seq_no: AtomicU64::new(1),
            write_offset: AtomicU64::new(logical_end),
            flush_offset: AtomicU64::new(logical_end),
            write_state: Mutex::new(WriteState {
                file,
                write_offset: logical_end,
                file_len,
                data_start,
            }),
            sync_file: Mutex::new(sync_file),
            last_checkpoint: TokioMutex::new(None),
            prealloc_chunk,
        })
    }

    /// Core inline append: write the record under the insert lock, then make
    /// it durable under the flush lock (coalescing with concurrent writers).
    fn do_append(&self, record_type: WalRecordType, payload: Vec<u8>) -> io::Result<u64> {
        let seq_no = self.next_seq_no.fetch_add(1, Ordering::SeqCst);
        let data = WalRecord::new(record_type, seq_no, payload).serialize();
        let len = data.len() as u64;

        // ── Insert phase ──────────────────────────────────────────────
        let my_end = {
            let mut ws = self.write_state.lock().unwrap();
            let off = ws.write_offset;
            let file_len = ws.file_len;
            if off + len > file_len {
                let grow_to = (off + len).max(file_len + self.prealloc_chunk);
                zero_fill(&mut ws.file, file_len, grow_to - file_len)?;
                ws.file.sync_all()?; // size changed: one metadata flush (rare)
                ws.file_len = grow_to;
                ws.file.seek(SeekFrom::Start(off))?;
            }
            ws.file.write_all(&data)?;
            let my_end = off + len;
            ws.write_offset = my_end;
            self.write_offset.store(my_end, Ordering::SeqCst);
            my_end
        };

        // ── Flush phase (coalesced) ───────────────────────────────────
        if self.flush_offset.load(Ordering::SeqCst) < my_end {
            let sf = self.sync_file.lock().unwrap();
            if self.flush_offset.load(Ordering::SeqCst) < my_end {
                // Everything appended so far becomes durable in one fdatasync.
                let target = self.write_offset.load(Ordering::SeqCst);
                sf.sync_data()?;
                self.flush_offset.store(target, Ordering::SeqCst);
            }
        }
        Ok(seq_no)
    }

    /// Append a record (async wrapper). Durable on return.
    pub async fn append(
        &self,
        record_type: WalRecordType,
        payload: Vec<u8>,
        _durability: DurabilityMode,
    ) -> io::Result<u64> {
        self.do_append(record_type, payload)
    }

    /// Append a record synchronously. Durable on return.
    pub fn append_sync(&self, record_type: WalRecordType, payload: Vec<u8>) -> io::Result<u64> {
        let start = std::time::Instant::now();
        let seq_no = self.do_append(record_type, payload)?;
        galaxdb_observe::metrics()
            .wal_write_latency_us
            .set(start.elapsed().as_micros() as i64);
        Ok(seq_no)
    }

    /// Next sequence number that will be assigned.
    pub fn next_seq_no(&self) -> u64 {
        self.next_seq_no.load(Ordering::SeqCst)
    }

    /// Current durable logical WAL size in bytes.
    pub fn current_size(&self) -> u64 {
        self.flush_offset.load(Ordering::SeqCst)
    }

    /// Whether a checkpoint should be triggered by size or time.
    pub async fn should_checkpoint(&self) -> bool {
        let size_exceeded = self.current_size() >= self.config.checkpoint_size_bytes;
        let time_exceeded = {
            let last_cp = self.last_checkpoint.lock().await;
            match last_cp.as_ref() {
                Some(info) => info.timestamp.elapsed() >= self.config.checkpoint_interval,
                None => self.current_size() > 0,
            }
        };
        size_exceeded || time_exceeded
    }

    /// Write a CHECKPOINT record and record the checkpoint state.
    pub async fn write_checkpoint(&self) -> io::Result<u64> {
        let offset = self.current_size();
        let seq_no = self.do_append(WalRecordType::Checkpoint, Vec::new())?;
        let mut last_cp = self.last_checkpoint.lock().await;
        *last_cp = Some(CheckpointInfo {
            seq_no,
            offset,
            timestamp: Instant::now(),
        });
        tracing::info!(seq_no, offset, "WAL checkpoint written");
        Ok(seq_no)
    }

    /// Truncate the WAL, keeping the checkpoint record and everything after
    /// it (rewritten from offset 0), then re-pre-allocate.
    pub async fn truncate_to_checkpoint(&self) -> io::Result<()> {
        let cp_info = {
            let last_cp = self.last_checkpoint.lock().await;
            match last_cp.as_ref() {
                Some(info) => info.clone(),
                None => return Ok(()),
            }
        };

        // Hold both locks for the rewrite: no appends or flushes during it.
        // Scoped in an explicit block so both std MutexGuards are released
        // before the `.await` below (clippy::await_holding_lock).
        {
            let mut ws = self.write_state.lock().unwrap();
            let _sf = self.sync_file.lock().unwrap();

            let logical_end = ws.write_offset;
            let len = logical_end.saturating_sub(cp_info.offset);
            let mut remaining = vec![0u8; len as usize];
            ws.file.seek(SeekFrom::Start(cp_info.offset))?;
            ws.file.read_exact(&mut remaining)?;

            // Rewrite the file: superblock at offset 0, then the retained
            // records. This is also the migration point for a legacy WAL —
            // after a truncate it is always versioned.
            let header = FormatHeader::new(format::WAL.magic, format::WAL.current_write).to_bytes();
            ws.file.set_len(0)?;
            ws.file.seek(SeekFrom::Start(0))?;
            ws.file.write_all(&header)?;
            ws.file.write_all(&remaining)?;
            ws.data_start = WAL_HEADER_SIZE;
            let new_end = WAL_HEADER_SIZE + remaining.len() as u64;
            let want = new_end.max(self.prealloc_chunk);
            if want > new_end {
                zero_fill(&mut ws.file, new_end, want - new_end)?;
            }
            ws.file.sync_all()?;
            ws.file.seek(SeekFrom::Start(new_end))?;
            ws.file_len = want;
            ws.write_offset = new_end;
            self.write_offset.store(new_end, Ordering::SeqCst);
            self.flush_offset.store(new_end, Ordering::SeqCst);
        }

        let mut last_cp = self.last_checkpoint.lock().await;
        if let Some(ref mut info) = *last_cp {
            // The retained checkpoint record now sits just past the superblock.
            info.offset = WAL_HEADER_SIZE;
        }
        tracing::info!(checkpoint_seq_no = cp_info.seq_no, "WAL truncated to checkpoint");
        Ok(())
    }

    /// No-op now (no background thread); kept for API compatibility.
    pub fn shutdown(&self) {}

    /// Get the last checkpoint info.
    pub async fn last_checkpoint(&self) -> Option<CheckpointInfo> {
        self.last_checkpoint.lock().await.clone()
    }
}

/// Recover WAL records from the last CHECKPOINT.
///
/// Reads the WAL, finds the last CHECKPOINT, replays everything after it.
/// Each record's XXH3-64 checksum is verified; recovery stops at the first
/// failure or at the zero padding (a zero record-type byte is invalid), so
/// the pre-allocated tail terminates replay cleanly.
#[allow(dead_code)]
pub fn recover_wal(wal_path: &Path) -> io::Result<(Vec<WalRecord>, u64)> {
    // Skip the versioned superblock (if present); a legacy WAL reads from 0.
    // A too-new superblock is refused here as a typed error (rollback safety).
    let data_start = wal_data_start(wal_path)?;
    let mut file = match File::open(wal_path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok((Vec::new(), 1)),
        Err(e) => return Err(e),
    };
    if file.metadata()?.len() <= data_start {
        return Ok((Vec::new(), 1));
    }
    if data_start > 0 {
        file.seek(SeekFrom::Start(data_start))?;
    }

    let mut reader = BufReader::new(file);
    let mut all_records: Vec<WalRecord> = Vec::new();
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
                all_records.push(record);
            }
            Ok(None) => break,
            Err(e) if e.kind() == io::ErrorKind::InvalidData => {
                tracing::debug!(error = %e, "WAL recovery: stopping at end of valid records");
                break;
            }
            Err(e) => return Err(e),
        }
    }

    let start_idx = match last_checkpoint_idx {
        Some(idx) => idx + 1,
        None => 0,
    };
    let recovered: Vec<WalRecord> = all_records.into_iter().skip(start_idx).collect();
    let next_seq_no = max_seq_no + 1;

    tracing::info!(
        recovered_count = recovered.len(),
        next_seq_no,
        had_checkpoint = last_checkpoint_idx.is_some(),
        "WAL recovery complete"
    );
    Ok((recovered, next_seq_no))
}
