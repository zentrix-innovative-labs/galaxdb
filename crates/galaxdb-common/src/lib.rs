//! GalaxDB Common — Shared types, config structs, and error types.
//!
//! This crate provides the foundational types used across all GalaxDB crates,
//! including type aliases for identifiers, column type definitions, configuration
//! structs, and a unified error type.

pub mod autotune;
pub mod config;
pub mod error;
pub mod types;

// Re-export commonly used items at crate root for convenience.
pub use autotune::{AutoTuneConfig, EffectiveTuning, SystemResources, TuningSource};
pub use config::GalaxConfig;
pub use error::{GalaxError, GalaxResult};
pub use types::{BlockId, ColumnType, RowId, TableId, Timestamp};
