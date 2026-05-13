//! GalaxDB Crypto — TDE with AES-256-GCM + AEGIS-256, pluggable key management.
//!
//! Two encryption backends:
//! - **AES-256-GCM** — for WAL records (append-only sequential writes)
//! - **AEGIS-256** — for PAX blocks (10-15 GB/s, random-access friendly)
//!
//! Key management is pluggable via the [`KeyProvider`] trait. Four real
//! providers are shipped:
//!
//! * [`LocalKeyProvider`] — 32-byte master key file (dev/self-hosted).
//! * [`EnvKeyProvider`] — hex-encoded key in an environment variable
//!   (containers).
//! * [`ExternalCommandKeyProvider`] — delegates to any KMS-CLI wrapper
//!   (AWS CLI, gcloud, az, vault CLI, custom HSM scripts). No SDK lock-in.
//! * [`HashicorpVaultKeyProvider`] — HashiCorp Vault Transit engine via
//!   rustls, opt-in behind the `vault` Cargo feature.
//!
//! See [`KeyProviderSpec::parse`] for the startup-time configuration
//! syntax (`local:`, `env:`, `command:`, `vault:`).

pub mod aegis_tde;
pub mod key_provider;
pub mod nonce;
pub mod tde;

pub use aegis_tde::AegisTdeModule;
pub use key_provider::{
    EnvKeyProvider, ExternalCommandKeyProvider, KeyProvider, KeyProviderSpec, LocalKeyProvider,
};
pub use nonce::NonceGenerator;
pub use tde::{CachedDataKey, TdeModule};

#[cfg(feature = "vault")]
pub use key_provider::HashicorpVaultKeyProvider;
