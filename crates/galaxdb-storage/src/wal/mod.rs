//! Write-Ahead Log (WAL) for GalaxDB.
//!
//! Provides crash-safe durability for all write operations. Each record is
//! LZ4-compressed and protected by an XXH3-64 checksum. The WAL supports
//! two durability modes:
//!
//! - **STRICT** — fsync per commit (bypasses group commit).
//! - **RELAXED** — group commit with a configurable batch window (default 10 ms).
//!
//! Checkpoint triggers when the WAL exceeds 512 MB or 60 seconds since the
//! last checkpoint. Recovery replays from the last CHECKPOINT record, verifying
//! checksums and stopping at the first failure.

mod record;
mod writer;

#[cfg(test)]
mod tests;

pub use record::{WalRecord, WalRecordType};
pub use writer::{DurabilityMode, WalWriter, WalWriterConfig, CheckpointInfo};
