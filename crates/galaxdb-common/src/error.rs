//! Unified error types for GalaxDB.
//!
//! All crates return `GalaxResult<T>` from fallible operations, keeping
//! error handling consistent across the engine.

use thiserror::Error;

/// Convenience alias used throughout GalaxDB.
pub type GalaxResult<T> = Result<T, GalaxError>;

/// Top-level error type covering every failure mode in the engine.
#[derive(Debug, Error)]
pub enum GalaxError {
    // -- Storage errors --
    /// An I/O operation failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A PAX block or WAL record failed checksum verification.
    #[error("checksum mismatch: expected {expected:#x}, got {actual:#x}")]
    ChecksumMismatch { expected: u64, actual: u64 },

    /// A PAX block has an invalid magic number.
    #[error("invalid magic number: expected 0x47414C41, got {0:#x}")]
    InvalidMagic(u32),

    /// The WAL contains a corrupt record during recovery.
    #[error("corrupt WAL record at sequence {seq_no}")]
    CorruptWalRecord { seq_no: u64 },

    // -- Capacity / resource errors --
    /// The disk is full; writes are blocked.
    #[error("disk full: writes are blocked until space is freed")]
    DiskFull,

    /// Back-pressure limit reached; the caller should retry.
    #[error("write back-pressure: sealed memtable bytes exceed limit")]
    BackPressure,

    /// Maximum connection count reached.
    #[error("too many connections (SQLSTATE 53300)")]
    TooManyConnections,

    // -- SQL / query errors --
    /// SQL parse error with byte offset.
    #[error("SQL parse error at position {position}: {message}")]
    SqlParse { position: usize, message: String },

    /// A referenced table does not exist.
    #[error("table not found: {0}")]
    TableNotFound(String),

    /// A table with the given name already exists.
    #[error("table already exists: {0}")]
    TableAlreadyExists(String),

    /// A column referenced in a query does not exist.
    #[error("column not found: {0}")]
    ColumnNotFound(String),

    /// An UPDATE targeted an embedding-source column, which is not allowed.
    #[error("cannot update embedding source column '{column}'; use DELETE + INSERT instead")]
    EmbeddingSourceUpdate { column: String },

    /// A DELETE or UPDATE targeted an append-only system table. Append-
    /// only tables (e.g. `_galaxdb_training_exports`, Req 38 / task 36)
    /// reject any mutation beyond INSERT so the lineage they record
    /// remains auditable.
    #[error("table '{table}' is append-only and does not support {operation}")]
    AppendOnlyTable {
        table: String,
        operation: &'static str,
    },

    /// Write-write conflict under snapshot isolation.
    #[error("write-write conflict on key; transaction aborted")]
    WriteConflict,

    // -- Authorization errors --
    /// A role attempted an action it lacks the privilege for. Rendered to
    /// PostgreSQL SQLSTATE `42501` (insufficient_privilege) on the wire,
    /// raised by the executor authorization chokepoint *before* any data
    /// is read or written (Requirement 3, AC3).
    #[error("permission denied: role '{role}' may not {action} on {object}")]
    InsufficientPrivilege {
        /// The role that was denied.
        role: String,
        /// The action it attempted (`select`/`insert`/`update`/`delete`/`ddl`/`admin`).
        action: &'static str,
        /// The object it targeted (e.g. `table:docs` or `cluster`).
        object: String,
    },

    // -- Versioning errors --
    /// The requested version tag does not exist.
    #[error("version tag not found: {0}")]
    VersionTagNotFound(String),

    /// SEMANTIC_SNAPSHOT consistency mode is not supported in v1.
    #[error("CONSISTENCY 'SEMANTIC_SNAPSHOT' is a v2 feature and is not supported")]
    SemanticSnapshotNotSupported,

    /// AT VERSION + SEMANTIC_MATCH without an explicit consistency mode.
    #[error("AT VERSION with SEMANTIC_MATCH requires an explicit CONSISTENCY mode")]
    SemanticConsistencyRequired,

    // -- Encryption errors --
    /// An encryption or decryption operation failed.
    #[error("encryption error: {0}")]
    Encryption(String),

    /// AWS KMS key management error.
    #[error("KMS error: {0}")]
    Kms(String),

    // -- Sidecar / embedding errors --
    /// The embedding sidecar is unavailable.
    #[error("semantic search temporarily unavailable — embedding sidecar is down")]
    SidecarUnavailable,

    /// An embedding request failed.
    #[error("embedding error: {0}")]
    Embedding(String),

    // -- Backup / restore errors --
    /// A backup or restore operation failed.
    #[error("backup/restore error: {0}")]
    BackupRestore(String),

    // -- Execution paths that are scheduled for a later task --
    /// The requested feature has not been implemented yet. Carries the
    /// task ID from `.kiro/specs/galaxdb-v1-engine/tasks.md` that will
    /// land it. This is deliberately a typed error rather than a fake
    /// `Ok` return — see the engineering principles in
    /// `.kiro/steering/engineering-principles.md`.
    #[error("feature not yet available (tracked by task {task}): {feature}")]
    NotYetAvailable {
        /// The task identifier, e.g. `"37"` or `"40.3"`.
        task: &'static str,
        /// Human-readable description of what the caller asked for.
        feature: &'static str,
    },

    // -- Generic catch-all --
    /// An internal error that doesn't fit other categories.
    #[error("internal error: {0}")]
    Internal(String),
}

impl GalaxError {
    /// The PostgreSQL SQLSTATE code that best describes this error, used
    /// by the wire protocol's `ErrorResponse`. Codes follow the
    /// PostgreSQL error-code appendix so standard clients
    /// (`psql`, `tokio-postgres`, JDBC, …) classify failures correctly.
    ///
    /// Errors without a more specific class fall back to `XX000`
    /// (internal_error) rather than a misleading syntax-error code.
    pub fn sqlstate(&self) -> &'static str {
        match self {
            // Class 42 — syntax error or access rule violation.
            GalaxError::SqlParse { .. } => "42601", // syntax_error
            GalaxError::InsufficientPrivilege { .. } => "42501", // insufficient_privilege
            GalaxError::TableNotFound(_) => "42P01", // undefined_table
            GalaxError::TableAlreadyExists(_) => "42P07", // duplicate_table
            GalaxError::ColumnNotFound(_) => "42703", // undefined_column

            // Class 53 — insufficient resources.
            GalaxError::TooManyConnections => "53300", // too_many_connections
            GalaxError::DiskFull => "53100",           // disk_full
            GalaxError::BackPressure => "53200",       // out_of_memory (back-pressure)

            // Class 40 — transaction rollback.
            GalaxError::WriteConflict => "40001", // serialization_failure

            // Class 0A — feature not supported.
            GalaxError::SemanticSnapshotNotSupported
            | GalaxError::SemanticConsistencyRequired
            | GalaxError::NotYetAvailable { .. } => "0A000", // feature_not_supported

            // Class 42 — restricted/generated column update (the embedding
            // source column behaves like a GENERATED ALWAYS column).
            GalaxError::EmbeddingSourceUpdate { .. } => "428C9", // generated_always

            // Class 58 — system error (I/O, corruption).
            GalaxError::Io(_)
            | GalaxError::ChecksumMismatch { .. }
            | GalaxError::InvalidMagic(_)
            | GalaxError::CorruptWalRecord { .. } => "58030", // io_error

            // Everything else (Internal, AppendOnlyTable, Encryption, Kms,
            // Sidecar, Embedding, Backup, Version tag): internal error.
            _ => "XX000",
        }
    }
}
