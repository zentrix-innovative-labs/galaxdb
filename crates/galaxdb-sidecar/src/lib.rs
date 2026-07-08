//! GalaxDB Embedding Sidecar — shared library for protocol types and client.
//!
//! This crate provides:
//! - `protocol`: Wire protocol types and serialization for Unix socket communication
//! - The sidecar binary (`galaxdb-sidecar`) uses these types for the server side
//! - The engine uses these types via `SidecarClient` for the client side
//!
//! The sidecar uses Unix sockets and is only available on Unix platforms
//! (Linux, macOS). On Windows, the manager module is replaced with a stub
//! that returns `SidecarUnavailable` for all operations.

// Protocol types are platform-independent (pure serialization).
pub mod protocol;

// Manager and tracking use Unix sockets — Unix only.
#[cfg(unix)]
pub mod manager;
#[cfg(unix)]
pub mod tracking;

// Multi-architecture embedding model registry + loaders (candle). Used by the sidecar
// binary (Unix-only). Gated to Unix to keep the non-Unix stub build minimal.
#[cfg(unix)]
pub mod models;

// On non-Unix platforms, provide stub types so dependents compile.
#[cfg(not(unix))]
pub mod manager {
    use std::path::PathBuf;
    use galaxdb_common::{GalaxError, GalaxResult};
    use crate::protocol::{EmbedRequest, EmbedResponse};

    pub const DEFAULT_MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum SidecarState { Stopped }

    #[derive(Debug, Clone)]
    pub struct SidecarConfig {
        pub binary_path: PathBuf,
        pub socket_path: PathBuf,
        pub model_id: String,
        pub data_dir: PathBuf,
    }

    pub struct SidecarManager;

    impl SidecarManager {
        pub fn new(_config: SidecarConfig) -> Self { Self }
        pub fn start(&self) -> GalaxResult<()> { Ok(()) }
        pub fn stop(&self) {}
        pub fn state(&self) -> SidecarState { SidecarState::Stopped }
        pub fn is_healthy(&self) -> bool { false }
        pub fn is_degraded(&self) -> bool { true }
        pub fn embed(&self, _request: EmbedRequest) -> GalaxResult<EmbedResponse> {
            Err(GalaxError::SidecarUnavailable)
        }
        pub fn backlog_size(&self) -> usize { 0 }
        pub fn drain_backlog(&self) -> usize { 0 }
        pub fn model_version(&self) -> String { String::new() }
        pub fn in_flight_count(&self) -> usize { 0 }
        pub fn record_missed_heartbeat(&self) {}
        pub fn record_heartbeat(&self) {}
        pub fn is_process_alive(&self) -> bool { false }
    }
}

#[cfg(not(unix))]
pub mod tracking {
    // Stub — no-op on Windows
}
