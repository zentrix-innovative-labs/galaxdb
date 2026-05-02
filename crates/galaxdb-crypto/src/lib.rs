//! GalaxDB Crypto — TDE (AES-256-GCM), TLS 1.3, pluggable key management.
//!
//! This crate provides transparent data encryption for PAX blocks and WAL records
//! using AES-256-GCM. Key management is pluggable via the [`KeyProvider`] trait,
//! with built-in implementations for local file, environment variable, and
//! (behind a feature flag) AWS KMS.
//!
//! The `aes-gcm` crate automatically uses AES-NI on x86-64 and ARMv8 crypto
//! extensions on ARM64, targeting < 8% CPU overhead.

pub mod key_provider;
pub mod nonce;
pub mod tde;

pub use key_provider::{EnvKeyProvider, KeyProvider, LocalKeyProvider};
pub use nonce::NonceGenerator;
pub use tde::{CachedDataKey, TdeModule};

#[cfg(feature = "aws-kms")]
pub use key_provider::AwsKmsKeyProvider;
