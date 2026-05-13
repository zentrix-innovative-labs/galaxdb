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
//!
//! ## Metric
//!
//! The handler owns a process-wide `prometheus::IntGauge` named
//! `galaxdb_disk_full`. It reads `1` while the engine is in disk-full mode
//! and `0` while the engine is in normal operation. The gauge is registered
//! with the default Prometheus registry exposed by `galaxdb-observe` so
//! that the `/metrics` endpoint can scrape it.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use prometheus::{IntGauge, Registry};
use tracing::{error, info, warn};

use galaxdb_common::{GalaxError, GalaxResult};

/// Default reserve file name.
const RESERVE_FILE_NAME: &str = "_galaxdb_reserve";

/// Metric name for the disk-full gauge.
const DISK_FULL_METRIC_NAME: &str = "galaxdb_disk_full";

/// Help string published alongside the `galaxdb_disk_full` metric.
const DISK_FULL_METRIC_HELP: &str =
    "Set to 1 while the storage engine is in disk-full recovery mode, 0 otherwise.";

/// Process-wide instance of the `galaxdb_disk_full` gauge.
///
/// Registration with the Prometheus registry must be idempotent — a single
/// process may construct multiple [`DiskFullHandler`] instances (one per
/// `Engine`) and Prometheus rejects duplicate registration with `AlreadyReg`.
/// The `OnceLock` guarantees a single registration attempt; every handler
/// then clones the `IntGauge` (cheap, it is an `Arc` internally).
static DISK_FULL_GAUGE: OnceLock<IntGauge> = OnceLock::new();

/// Register the `galaxdb_disk_full` gauge with `registry` exactly once per
/// process.
///
/// The `OnceLock` makes registration idempotent: the first caller creates the
/// gauge and registers it with the provided Prometheus registry; every
/// subsequent caller (other `DiskFullHandler` instances, other crates that
/// want to read the same signal) receives a clone of the same `IntGauge`
/// handle. Cloning an `IntGauge` is cheap — internally it is an `Arc` — and
/// every clone observes the same counter value, so `/metrics` scrapes stay
/// consistent across callers.
///
/// Any Prometheus error other than the `OnceLock`-guarded first registration
/// is a hard failure. Surfacing it as a panic matches the engineering-
/// principles rule that forbids silent fallbacks: if we cannot register the
/// metric there is no correct fallback, and handing back a detached gauge
/// that `/metrics` cannot see would be exactly the kind of silent fake
/// behaviour the rule exists to prevent.
fn get_or_register_disk_full_gauge(registry: &Registry) -> IntGauge {
    DISK_FULL_GAUGE
        .get_or_init(|| {
            let gauge = IntGauge::new(DISK_FULL_METRIC_NAME, DISK_FULL_METRIC_HELP)
                .expect("galaxdb_disk_full gauge construction must not fail");
            registry
                .register(Box::new(gauge.clone()))
                .unwrap_or_else(|err| {
                    panic!(
                        "failed to register {DISK_FULL_METRIC_NAME} with the Prometheus \
                         registry: {err}"
                    )
                });
            gauge
        })
        .clone()
}

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
    /// Prometheus gauge emitting `1` while in disk-full mode, `0` otherwise.
    gauge: IntGauge,
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

        // Wire up the Prometheus gauge against the process-wide registry.
        // Registration is idempotent — the `OnceLock` guarantees a single
        // register call regardless of how many `DiskFullHandler` instances
        // are created.
        let gauge = get_or_register_disk_full_gauge(galaxdb_observe::default_registry());
        gauge.set(0);

        Ok(Self {
            reserve_path,
            reserve_size,
            disk_full: AtomicBool::new(false),
            gauge,
        })
    }

    /// Handle a disk-full event.
    ///
    /// 1. Deletes the reserve file to free space.
    /// 2. Sets the disk-full flag so that [`is_disk_full`] returns `true`.
    /// 3. Sets the `galaxdb_disk_full` gauge to `1` and logs an error.
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

        // Step 3: Publish the metric and log the error. Prometheus scrapers
        // reading `/metrics` will now see `galaxdb_disk_full 1`.
        self.gauge.set(1);
        error!("disk full detected — all writes are now blocked until space is freed");

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
        // Publish the metric back to the healthy state.
        self.gauge.set(0);

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

    /// Return the current value of the `galaxdb_disk_full` Prometheus gauge.
    ///
    /// Exposed so callers (and tests) can read the same metric that Prometheus
    /// scrapers see without having to parse `/metrics` output.
    pub fn disk_full_gauge(&self) -> i64 {
        self.gauge.get()
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
