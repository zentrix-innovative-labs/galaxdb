//! GalaxDB Crypto — TDE with AES-256-GCM + AEGIS-256, pluggable key management.
//!
//! Two encryption backends:
//! - **AES-256-GCM** — for WAL records (append-only sequential writes)
//! - **AEGIS-256** — for PAX blocks (10-15 GB/s, random-access friendly)
//!
//! Key management is pluggable via the [`KeyProvider`] trait.

pub mod aegis_tde;
pub mod key_provider;
pub mod nonce;
pub mod tde;

pub use aegis_tde::AegisTdeModule;
pub use key_provider::{EnvKeyProvider, KeyProvider, LocalKeyProvider};
pub use nonce::NonceGenerator;
pub use tde::{CachedDataKey, TdeModule};

#[cfg(feature = "aws-kms")]
pub use key_provider::AwsKmsKeyProvider;
