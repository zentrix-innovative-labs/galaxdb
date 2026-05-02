//! Disk-full handling for GalaxDB.
//!
//! At engine startup, a reserve file (`_galaxdb_reserve`) is pre-allocated.
//! When a disk-full condition is detected the handler:
//!
//! 1. Deletes the reserve file to free space.
//! 2. Performs a clean checkpoint (flush memtable, write checkpoint record).
//! 3. Blocks all subsequent writes.
//! 4. Emits the `_disk_full` metric and logs an error.
//!
//! No data corruption occurs — all committed data remains safe on disk.
//!
//! ## Recovery
//!
//! After an operator frees disk space, [`DiskFullHandler::recover`] re-creates
//! the reserve file and unblocks writes.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use tracing::{error, info, warn};

use galaxdb_common::{GalaxError, GalaxResult};

/// Default reserve file name.
const RESERVE_FILE_NAME: &str = "_galaxdb_reserve";

/// Handles disk-full detection and recovery for the storage engine.
///
/// The handler pre-allocates a reserve file at startup. On disk-full the
/// reserve is deleted to free enough space for a clean checkpoint, and all
/// writes are blocked until the operator frees space and calls [`recover`].
///
/// [`recover`]: DiskFullHandler::recover
pub struct DiskFullHandler {
    /// Path to the reserve file.
    reserve_path: PathBuf,
    /// Size of the reserve file in bytes.
    reserve_size: u64,
    /// `true` when the engine is in disk-full mode (writes blocked).
    disk_full: AtomicBool,
}

impl DiskFullHandler {
    /// Create a new handler **and** pre-allocate the reserve file.
    ///
    /// `data_dir` is the engine's data directory and `reserve_size` is the
    /// number of bytes to pre-allocate (default 32 MB from `StorageConfig`).
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the reserve file cannot be created.
    pub fn init(data_dir: &Path, reserve_size: u64) -> GalaxResult<Self> {
        let reserve_path = data_dir.join(RESERVE_FILE_NAME);

        // Ensure the data directory exists.
        std::fs::create_dir_all(data_dir)?;

        // Pre-allocate the reserve file filled with zeros.
        Self::create_reserve_file(&reserve_path, reserve_size)?;

        info!(
            path = %reserve_path.display(),
            size_bytes = reserve_size,
            "pre-allocated disk-full reserve file"
        );

        Ok(Self {
            reserve_path,
            reserve_size,
            disk_full: AtomicBool::new(false),
        })
    }

    /// Handle a disk-full event.
    ///
    /// 1. Deletes the reserve file to free space.
    /// 2. Sets the disk-full flag so that [`is_disk_full`] returns `true`.
    /// 3. Logs an error and emits the `_disk_full` metric conceptually.
    ///
    /// The caller is responsible for performing a clean checkpoint (flush
    /// memtable + write checkpoint record) after this method returns.
    ///
    /// [`is_disk_full`]: DiskFullHandler::is_disk_full
    pub fn handle_disk_full(&self) -> GalaxResult<()> {
        // Avoid double-handling.
        if self.disk_full.load(Ordering::SeqCst) {
            warn!("disk-full handler invoked but engine is already in disk-full mode");
            return Ok(());
        }

        // Step 1: Delete the reserve file to free space.
        if self.reserve_path.exists() {
            std::fs::remove_file(&self.reserve_path).map_err(|e| {
                error!(path = %self.reserve_path.display(), error = %e, "failed to delete reserve file");
                GalaxError::Io(e)
            })?;
            info!(
                path = %self.reserve_path.display(),
                freed_bytes = self.reserve_size,
                "deleted reserve file to free space for clean checkpoint"
            );
        } else {
            warn!(
                path = %self.reserve_path.display(),
                "reserve file does not exist; cannot free additional space"
            );
        }

        // Step 2: Set the disk-full flag (blocks writes).
        self.disk_full.store(true, Ordering::SeqCst);

        // Step 3: Log error and emit metric.
        error!("disk full detected — all writes are now blocked until space is freed");

        // The `_disk_full` metric would be incremented here via the
        // observability module. For now we rely on the tracing log line
        // and the `is_disk_full()` flag which the metrics collector can
        // poll.

        Ok(())
    }

    /// Returns `true` when the engine is in disk-full mode and writes
    /// should be blocked.
    pub fn is_disk_full(&self) -> bool {
        self.disk_full.load(Ordering::SeqCst)
    }

    /// Recover from a disk-full condition.
    ///
    /// Re-creates the reserve file and clears the disk-full flag so that
    /// writes can resume. The operator should only call this after freeing
    /// sufficient disk space.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the reserve file cannot be re-created (e.g.
    /// the disk is still full).
    pub fn recover(&self) -> GalaxResult<()> {
        if !self.disk_full.load(Ordering::SeqCst) {
            info!("recover called but engine is not in disk-full mode");
            return Ok(());
        }

        // Re-create the reserve file.
        Self::create_reserve_file(&self.reserve_path, self.reserve_size)?;

        // Clear the flag — writes are unblocked.
        self.disk_full.store(false, Ordering::SeqCst);

        info!(
            path = %self.reserve_path.display(),
            size_bytes = self.reserve_size,
            "re-created reserve file; writes unblocked"
        );

        Ok(())
    }

    /// Returns the path to the reserve file.
    pub fn reserve_path(&self) -> &Path {
        &self.reserve_path
    }

    /// Returns the configured reserve file size in bytes.
    pub fn reserve_size(&self) -> u64 {
        self.reserve_size
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Create (or overwrite) the reserve file with `size` zero-bytes.
    fn create_reserve_file(path: &Path, size: u64) -> GalaxResult<()> {
        use std::io::Write;

        let mut file = std::fs::File::create(path)?;

        // Write in 64 KB chunks to avoid a single huge allocation.
        const CHUNK: usize = 64 * 1024;
        let zeros = vec![0u8; CHUNK];
        let mut remaining = size;

        while remaining > 0 {
            let to_write = std::cmp::min(remaining, CHUNK as u64) as usize;
            file.write_all(&zeros[..to_write])?;
            remaining -= to_write as u64;
        }

        file.sync_all()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
