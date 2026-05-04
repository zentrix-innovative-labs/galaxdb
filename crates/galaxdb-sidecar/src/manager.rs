//! Sidecar Manager — engine-side lifecycle management for the embedding sidecar.
//!
//! Responsibilities:
//! - Spawn the sidecar binary as a child process
//! - Monitor heartbeats (3 missed → degraded mode)
//! - Restart with exponential backoff on crash (1s, 2s, 4s, 8s, 16s, 32s, 60s cap)
//! - Route embedding requests to the sidecar via Unix socket
//! - Track in-flight count with semaphore (capacity 10,000)
//! - Overflow to backlog when semaphore is full
//!
//! Per design spec Section 8.1 and 8.2, Requirements 19.

use std::io::{BufReader, BufWriter};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use galaxdb_common::{GalaxError, GalaxResult};

use crate::protocol::*;

/// Sidecar connection state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SidecarState {
    /// Sidecar is running and responding to heartbeats.
    Healthy,
    /// Sidecar has missed heartbeats — semantic search unavailable.
    Degraded,
    /// Sidecar is not running (initial state or after shutdown).
    Stopped,
    /// Sidecar is restarting after a crash.
    Restarting,
}

/// Configuration for the sidecar manager.
#[derive(Debug, Clone)]
pub struct SidecarConfig {
    /// Path to the sidecar binary.
    pub binary_path: PathBuf,
    /// Unix socket path for communication.
    pub socket_path: PathBuf,
    /// Model path (ONNX file) or None for mock mode.
    pub model_path: Option<PathBuf>,
    /// Mock embedding dimensions (for testing without a model).
    pub mock_dim: Option<usize>,
    /// Data directory (for backlog table).
    pub data_dir: PathBuf,
}

/// Engine-side sidecar manager.
///
/// Manages the lifecycle of the embedding sidecar process and provides
/// the `embed()` method for generating embeddings.
pub struct SidecarManager {
    config: SidecarConfig,
    /// Current state of the sidecar.
    state: Arc<std::sync::RwLock<SidecarState>>,
    /// Child process handle.
    child: Mutex<Option<Child>>,
    /// Number of consecutive missed heartbeats.
    missed_heartbeats: AtomicU32,
    /// Current restart attempt (for exponential backoff).
    restart_attempt: AtomicU32,
    /// Number of in-flight embedding requests.
    in_flight: AtomicUsize,
    /// Whether the manager has been shut down.
    shutdown: AtomicBool,
    /// Current model version reported by the sidecar.
    model_version: Mutex<String>,
    /// Backlog: embedding requests that couldn't be sent (sidecar down or at capacity).
    backlog: Mutex<Vec<EmbedRequest>>,
}

impl SidecarManager {
    /// Create a new sidecar manager (does not start the sidecar yet).
    pub fn new(config: SidecarConfig) -> Self {
        Self {
            config,
            state: Arc::new(std::sync::RwLock::new(SidecarState::Stopped)),
            child: Mutex::new(None),
            missed_heartbeats: AtomicU32::new(0),
            restart_attempt: AtomicU32::new(0),
            in_flight: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            model_version: Mutex::new(String::new()),
            backlog: Mutex::new(Vec::new()),
        }
    }

    /// Start the sidecar process.
    pub fn start(&self) -> GalaxResult<()> {
        let mut child_guard = self.child.lock()
            .map_err(|_| GalaxError::Internal("child lock poisoned".into()))?;

        if child_guard.is_some() {
            return Ok(()); // already running
        }

        let mut cmd = Command::new(&self.config.binary_path);
        cmd.arg("--socket").arg(&self.config.socket_path);
        cmd.arg("--parent-pid").arg(std::process::id().to_string());

        if let Some(dim) = self.config.mock_dim {
            cmd.arg("--mock-dim").arg(dim.to_string());
        }
        if let Some(ref model) = self.config.model_path {
            cmd.arg("--model").arg(model);
        }

        let child = cmd.spawn().map_err(|e| {
            GalaxError::Internal(format!("failed to spawn sidecar: {}", e))
        })?;

        *child_guard = Some(child);
        *self.state.write().unwrap() = SidecarState::Healthy;
        self.missed_heartbeats.store(0, Ordering::Relaxed);
        self.restart_attempt.store(0, Ordering::Relaxed);

        Ok(())
    }

    /// Stop the sidecar process.
    pub fn stop(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
        let mut child_guard = self.child.lock().unwrap();
        if let Some(ref mut child) = *child_guard {
            let _ = child.kill();
            let _ = child.wait();
        }
        *child_guard = None;
        *self.state.write().unwrap() = SidecarState::Stopped;
    }

    /// Get the current sidecar state.
    pub fn state(&self) -> SidecarState {
        *self.state.read().unwrap()
    }

    /// Check if the sidecar is healthy (accepting requests).
    pub fn is_healthy(&self) -> bool {
        self.state() == SidecarState::Healthy
    }

    /// Check if the sidecar is degraded (not accepting semantic search).
    pub fn is_degraded(&self) -> bool {
        matches!(self.state(), SidecarState::Degraded | SidecarState::Stopped | SidecarState::Restarting)
    }

    /// Send an embedding request to the sidecar.
    ///
    /// If the sidecar is down or at capacity, the request is added to the backlog.
    /// Returns the embedding response or an error.
    pub fn embed(&self, request: EmbedRequest) -> GalaxResult<EmbedResponse> {
        if self.is_degraded() {
            // Sidecar is down — add to backlog
            self.add_to_backlog(request);
            return Err(GalaxError::Internal(
                "semantic search temporarily unavailable — embedding sidecar is down".into()
            ));
        }

        // Check in-flight capacity
        let current = self.in_flight.fetch_add(1, Ordering::Relaxed);
        if current >= MAX_IN_FLIGHT {
            self.in_flight.fetch_sub(1, Ordering::Relaxed);
            // Over capacity — add to backlog
            self.add_to_backlog(request);
            return Err(GalaxError::Internal(
                "embedding backlog: sidecar at capacity".into()
            ));
        }

        // Connect and send request
        let result = self.send_embed_request(&request);
        self.in_flight.fetch_sub(1, Ordering::Relaxed);

        match result {
            Ok(response) => {
                // Update model version
                let mut ver = self.model_version.lock().unwrap();
                *ver = response.model_version.clone();
                Ok(response)
            }
            Err(e) => {
                // Connection failed — sidecar may have crashed
                self.add_to_backlog(request);
                Err(e)
            }
        }
    }

    /// Send an embed request via Unix socket.
    fn send_embed_request(&self, request: &EmbedRequest) -> GalaxResult<EmbedResponse> {
        let stream = UnixStream::connect(&self.config.socket_path)
            .map_err(|e| GalaxError::Internal(format!("sidecar connect failed: {}", e)))?;
        stream.set_read_timeout(Some(Duration::from_secs(30))).ok();

        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut writer = BufWriter::new(stream);

        let msg = SidecarMessage::EmbedRequest(request.clone());
        write_message(&mut writer, &msg)
            .map_err(|e| GalaxError::Internal(format!("sidecar write failed: {}", e)))?;

        let response = read_message(&mut reader)
            .map_err(|e| GalaxError::Internal(format!("sidecar read failed: {}", e)))?;

        match response {
            SidecarMessage::EmbedResponse(resp) => Ok(resp),
            SidecarMessage::Error { message } => {
                Err(GalaxError::Internal(format!("sidecar error: {}", message)))
            }
            _ => Err(GalaxError::Internal("unexpected sidecar response".into())),
        }
    }

    /// Add a request to the backlog.
    fn add_to_backlog(&self, request: EmbedRequest) {
        let mut backlog = self.backlog.lock().unwrap();
        backlog.push(request);
    }

    /// Get the current backlog size.
    pub fn backlog_size(&self) -> usize {
        self.backlog.lock().unwrap().len()
    }

    /// Drain the backlog by sending pending requests to the sidecar.
    ///
    /// Called when the sidecar recovers capacity. Processes requests FIFO.
    /// Returns the number of successfully processed requests.
    pub fn drain_backlog(&self) -> usize {
        if self.is_degraded() {
            return 0;
        }

        let mut backlog = self.backlog.lock().unwrap();
        let mut processed = 0;

        while let Some(request) = backlog.first().cloned() {
            match self.send_embed_request(&request) {
                Ok(_response) => {
                    backlog.remove(0);
                    processed += 1;
                }
                Err(_) => break, // sidecar not ready, stop draining
            }
        }

        processed
    }

    /// Record a missed heartbeat. After 3 consecutive misses, enter degraded mode.
    pub fn record_missed_heartbeat(&self) {
        let missed = self.missed_heartbeats.fetch_add(1, Ordering::Relaxed) + 1;
        if missed >= 3 {
            *self.state.write().unwrap() = SidecarState::Degraded;
        }
    }

    /// Record a successful heartbeat. Resets the missed counter.
    pub fn record_heartbeat(&self) {
        self.missed_heartbeats.store(0, Ordering::Relaxed);
        if self.state() == SidecarState::Degraded {
            *self.state.write().unwrap() = SidecarState::Healthy;
        }
    }

    /// Attempt to restart the sidecar after a crash.
    ///
    /// Uses exponential backoff: 1s, 2s, 4s, 8s, 16s, 32s, 60s (capped).
    pub fn restart(&self) -> GalaxResult<()> {
        if self.shutdown.load(Ordering::Relaxed) {
            return Ok(());
        }

        *self.state.write().unwrap() = SidecarState::Restarting;

        // Kill any existing process
        {
            let mut child_guard = self.child.lock().unwrap();
            if let Some(ref mut child) = *child_guard {
                let _ = child.kill();
                let _ = child.wait();
            }
            *child_guard = None;
        }

        // Exponential backoff
        let attempt = self.restart_attempt.fetch_add(1, Ordering::Relaxed) as usize;
        let delay_secs = if attempt < RESTART_BACKOFF.len() {
            RESTART_BACKOFF[attempt]
        } else {
            *RESTART_BACKOFF.last().unwrap()
        };

        std::thread::sleep(Duration::from_secs(delay_secs));

        // Try to start
        self.start()
    }

    /// Check if the sidecar process is still alive.
    pub fn is_process_alive(&self) -> bool {
        let mut child_guard = self.child.lock().unwrap();
        if let Some(ref mut child) = *child_guard {
            match child.try_wait() {
                Ok(None) => true,  // still running
                Ok(Some(_)) => false, // exited
                Err(_) => false,
            }
        } else {
            false
        }
    }

    /// Get the current model version.
    pub fn model_version(&self) -> String {
        self.model_version.lock().unwrap().clone()
    }

    /// Get the in-flight count.
    pub fn in_flight_count(&self) -> usize {
        self.in_flight.load(Ordering::Relaxed)
    }
}

impl Drop for SidecarManager {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::time::Instant;

    fn test_config(socket_path: &Path) -> SidecarConfig {
        // Find the sidecar binary in the target directory
        let binary = std::env::current_exe().unwrap()
            .parent().unwrap() // deps/
            .parent().unwrap() // debug/ or release/
            .join("galaxdb-sidecar");

        // Build it if it doesn't exist
        if !binary.exists() {
            let status = std::process::Command::new("cargo")
                .args(["build", "-p", "galaxdb-sidecar"])
                .status()
                .expect("cargo build");
            assert!(status.success(), "failed to build sidecar binary");
        }

        SidecarConfig {
            binary_path: binary,
            socket_path: socket_path.to_path_buf(),
            model_path: None,
            mock_dim: Some(64),
            data_dir: socket_path.parent().unwrap().to_path_buf(),
        }
    }

    /// Wait for the sidecar socket to appear.
    fn wait_for_socket(path: &Path, timeout: Duration) -> bool {
        let start = Instant::now();
        while start.elapsed() < timeout {
            if path.exists() {
                std::thread::sleep(Duration::from_millis(50));
                return true;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        false
    }

    #[test]
    fn manager_starts_and_stops() {
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("test.sock");
        let config = test_config(&socket);
        let mgr = SidecarManager::new(config);

        assert_eq!(mgr.state(), SidecarState::Stopped);

        mgr.start().unwrap();
        assert!(wait_for_socket(&socket, Duration::from_secs(5)), "socket should appear");
        assert_eq!(mgr.state(), SidecarState::Healthy);
        assert!(mgr.is_process_alive());

        mgr.stop();
        assert_eq!(mgr.state(), SidecarState::Stopped);
    }

    #[test]
    fn manager_embed_request() {
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("test_embed.sock");
        let config = test_config(&socket);
        let mgr = SidecarManager::new(config);

        mgr.start().unwrap();
        assert!(wait_for_socket(&socket, Duration::from_secs(5)));

        let response = mgr.embed(EmbedRequest {
            row_id: 1,
            text: "hello world".to_string(),
            column: "emb".to_string(),
        }).unwrap();

        assert_eq!(response.row_id, 1);
        assert_eq!(response.embedding.len(), 64);
        assert!(!response.model_version.is_empty());

        mgr.stop();
    }

    #[test]
    fn manager_degraded_mode() {
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("test_degraded.sock");
        let config = test_config(&socket);
        let mgr = SidecarManager::new(config);

        // Don't start the sidecar — it should be in Stopped state
        assert!(mgr.is_degraded());

        // Embed should fail and add to backlog
        let result = mgr.embed(EmbedRequest {
            row_id: 1,
            text: "test".to_string(),
            column: "emb".to_string(),
        });
        assert!(result.is_err());
        assert_eq!(mgr.backlog_size(), 1);
    }

    #[test]
    fn manager_heartbeat_tracking() {
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("test_hb.sock");
        let config = test_config(&socket);
        let mgr = SidecarManager::new(config);

        mgr.start().unwrap();
        assert!(wait_for_socket(&socket, Duration::from_secs(5)));
        assert!(mgr.is_healthy());

        // Miss 2 heartbeats — still healthy
        mgr.record_missed_heartbeat();
        mgr.record_missed_heartbeat();
        assert!(mgr.is_healthy());

        // Miss 3rd — degraded
        mgr.record_missed_heartbeat();
        assert_eq!(mgr.state(), SidecarState::Degraded);

        // Successful heartbeat — back to healthy
        mgr.record_heartbeat();
        assert!(mgr.is_healthy());

        mgr.stop();
    }

    #[test]
    fn manager_backlog_drain() {
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("test_drain.sock");
        let config = test_config(&socket);
        let mgr = SidecarManager::new(config);

        // Add requests to backlog while sidecar is down
        mgr.add_to_backlog(EmbedRequest {
            row_id: 1, text: "a".to_string(), column: "emb".to_string(),
        });
        mgr.add_to_backlog(EmbedRequest {
            row_id: 2, text: "b".to_string(), column: "emb".to_string(),
        });
        assert_eq!(mgr.backlog_size(), 2);

        // Start sidecar and wait for it to be ready
        mgr.start().unwrap();
        assert!(wait_for_socket(&socket, Duration::from_secs(5)), "socket should appear");

        // Verify sidecar is actually accepting connections before draining
        let test_result = mgr.send_embed_request(&EmbedRequest {
            row_id: 99, text: "test".to_string(), column: "emb".to_string(),
        });
        assert!(test_result.is_ok(), "sidecar should be accepting connections");

        // Drain backlog
        let processed = mgr.drain_backlog();
        assert_eq!(processed, 2);
        assert_eq!(mgr.backlog_size(), 0);

        mgr.stop();
    }
}
