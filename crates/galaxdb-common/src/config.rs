//! Engine configuration structs.
//!
//! `GalaxConfig` is the top-level configuration loaded at startup. Sub-configs
//! are broken out per subsystem so individual crates can accept only the slice
//! they need.

use serde::{Deserialize, Serialize};

/// Top-level GalaxDB engine configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GalaxConfig {
    /// Storage engine settings.
    pub storage: StorageConfig,
    /// Wire protocol / networking settings.
    pub server: ServerConfig,
    /// Encryption settings.
    pub crypto: CryptoConfig,
    /// Observability settings.
    pub observe: ObserveConfig,
    /// Embedding sidecar settings.
    pub sidecar: SidecarConfig,
}

/// Storage engine configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Directory for data files.
    pub data_dir: String,
    /// Memtable seal threshold in bytes (default: 64 MB).
    pub memtable_size_bytes: u64,
    /// Maximum sealed-but-unflushed memtable bytes before back-pressure (default: 256 MB).
    pub back_pressure_bytes: u64,
    /// SST file target size in bytes (default: 64 MB, configurable down to 8 MB).
    pub sst_size_bytes: u64,
    /// WAL group commit interval in milliseconds (default: 10).
    pub wal_group_commit_ms: u64,
    /// WAL checkpoint trigger size in bytes (default: 512 MB).
    pub wal_checkpoint_size_bytes: u64,
    /// WAL checkpoint trigger interval in seconds (default: 60).
    pub wal_checkpoint_interval_secs: u64,
    /// KV separation threshold in bytes (default: 1024).
    pub blob_threshold_bytes: u32,
    /// Number of blob log writer queues (default: 4).
    pub blob_writer_queues: u32,
    /// Bloom filter bits per key across all levels (default: 10).
    pub bloom_bits_per_key: u32,
    /// Buffer pool hot set fraction (default: 0.70).
    pub buffer_pool_hot_fraction: f64,
    /// LSM size ratio (default: 10).
    pub lsm_size_ratio: u32,
    /// Reserve file size for disk-full handling in bytes (default: 32 MB).
    pub reserve_file_bytes: u64,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            data_dir: "galaxdb_data".to_string(),
            memtable_size_bytes: 64 * 1024 * 1024,           // 64 MB
            back_pressure_bytes: 256 * 1024 * 1024,          // 256 MB
            sst_size_bytes: 64 * 1024 * 1024,                // 64 MB
            wal_group_commit_ms: 10,
            wal_checkpoint_size_bytes: 512 * 1024 * 1024,    // 512 MB
            wal_checkpoint_interval_secs: 60,
            blob_threshold_bytes: 1024,                       // 1 KB
            blob_writer_queues: 4,
            bloom_bits_per_key: 10,
            buffer_pool_hot_fraction: 0.70,
            lsm_size_ratio: 10,
            reserve_file_bytes: 32 * 1024 * 1024,            // 32 MB
        }
    }
}

/// Wire protocol and server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Address to bind the PostgreSQL wire protocol listener.
    pub listen_addr: String,
    /// Maximum number of concurrent connections (default: 1000).
    pub max_connections: usize,
    /// Address to bind the HTTP observability server.
    pub http_addr: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            listen_addr: "0.0.0.0:5433".to_string(),
            max_connections: 1000,
            http_addr: "0.0.0.0:9090".to_string(),
        }
    }
}

/// Encryption configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CryptoConfig {
    /// Whether TDE is enabled (default: false for development).
    pub tde_enabled: bool,
    /// AWS KMS key ARN for data encryption key management.
    pub kms_key_arn: Option<String>,
    /// Path to TLS certificate file (PEM). If `None`, a self-signed cert is generated.
    pub tls_cert_path: Option<String>,
    /// Path to TLS private key file (PEM).
    pub tls_key_path: Option<String>,
}

/// Observability configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObserveConfig {
    /// Log level (default: "info"). Overridden by GALAXDB_LOG_LEVEL env var.
    pub log_level: String,
    /// Whether to enable OpenTelemetry trace export.
    pub otel_enabled: bool,
    /// OpenTelemetry collector endpoint.
    pub otel_endpoint: Option<String>,
}

impl Default for ObserveConfig {
    fn default() -> Self {
        Self {
            log_level: "info".to_string(),
            otel_enabled: false,
            otel_endpoint: None,
        }
    }
}

/// Embedding sidecar configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SidecarConfig {
    /// Path to the sidecar binary.
    pub binary_path: Option<String>,
    /// Path to the ONNX model file.
    pub model_path: Option<String>,
    /// Maximum in-flight embedding requests before backlog overflow (default: 10_000).
    pub max_in_flight: usize,
    /// Heartbeat interval in seconds (default: 5).
    pub heartbeat_interval_secs: u64,
    /// Heartbeat timeout in seconds (default: 2).
    pub heartbeat_timeout_secs: u64,
}

impl Default for SidecarConfig {
    fn default() -> Self {
        Self {
            binary_path: None,
            model_path: None,
            max_in_flight: 10_000,
            heartbeat_interval_secs: 5,
            heartbeat_timeout_secs: 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_expected_values() {
        let config = GalaxConfig::default();
        assert_eq!(config.storage.memtable_size_bytes, 64 * 1024 * 1024);
        assert_eq!(config.storage.back_pressure_bytes, 256 * 1024 * 1024);
        assert_eq!(config.storage.sst_size_bytes, 64 * 1024 * 1024);
        assert_eq!(config.storage.wal_group_commit_ms, 10);
        assert_eq!(config.storage.blob_threshold_bytes, 1024);
        assert_eq!(config.server.max_connections, 1000);
        assert!(!config.crypto.tde_enabled);
        assert_eq!(config.sidecar.max_in_flight, 10_000);
    }

    #[test]
    fn config_serializes_to_json() {
        let config = GalaxConfig::default();
        let json = serde_json::to_string(&config).expect("serialize");
        let _: GalaxConfig = serde_json::from_str(&json).expect("deserialize");
    }
}
