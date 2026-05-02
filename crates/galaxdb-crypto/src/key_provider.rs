//! Pluggable key management — no vendor lock-in.
//!
//! The [`KeyProvider`] trait abstracts key generation and decryption so that
//! GalaxDB can run with a local key file (dev/self-hosted), an environment
//! variable (containers), or AWS KMS (production, behind feature flag).

use aes_gcm::aead::Aead;
use aes_gcm::{Aes256Gcm, KeyInit, Nonce};
use galaxdb_common::GalaxResult;
use rand::RngCore;
use std::path::Path;

/// Trait for pluggable key management — no vendor lock-in.
pub trait KeyProvider: Send + Sync {
    /// Generate or retrieve a data encryption key (DEK).
    /// Returns `(plaintext_key, encrypted_key_for_storage)`.
    fn generate_data_key(&self) -> GalaxResult<(Vec<u8>, Vec<u8>)>;

    /// Decrypt a previously encrypted DEK.
    fn decrypt_data_key(&self, encrypted_key: &[u8]) -> GalaxResult<Vec<u8>>;

    /// Provider name for logging/config.
    fn provider_name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Helper: encrypt/decrypt a DEK with a 32-byte master key using AES-256-GCM
// ---------------------------------------------------------------------------

/// Encrypt a plaintext DEK with the given master key.
/// The output is `nonce (12 bytes) || ciphertext+tag`.
fn encrypt_dek_with_master(master_key: &[u8; 32], plaintext_dek: &[u8]) -> GalaxResult<Vec<u8>> {
    let cipher = Aes256Gcm::new_from_slice(master_key)
        .map_err(|e| galaxdb_common::GalaxError::Encryption(format!("cipher init: {e}")))?;

    let mut nonce_bytes = [0u8; 12];
    rand::thread_rng().fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);

    let ciphertext = cipher
        .encrypt(nonce, plaintext_dek)
        .map_err(|e| galaxdb_common::GalaxError::Encryption(format!("DEK encrypt: {e}")))?;

    // nonce || ciphertext+tag
    let mut out = Vec::with_capacity(12 + ciphertext.len());
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ciphertext);
    Ok(out)
}

/// Decrypt an encrypted DEK (nonce || ciphertext+tag) with the given master key.
fn decrypt_dek_with_master(master_key: &[u8; 32], encrypted_dek: &[u8]) -> GalaxResult<Vec<u8>> {
    if encrypted_dek.len() < 12 {
        return Err(galaxdb_common::GalaxError::Encryption(
            "encrypted DEK too short".to_string(),
        ));
    }

    let cipher = Aes256Gcm::new_from_slice(master_key)
        .map_err(|e| galaxdb_common::GalaxError::Encryption(format!("cipher init: {e}")))?;

    let nonce = Nonce::from_slice(&encrypted_dek[..12]);
    let ciphertext = &encrypted_dek[12..];

    cipher
        .decrypt(nonce, ciphertext)
        .map_err(|e| galaxdb_common::GalaxError::Encryption(format!("DEK decrypt: {e}")))
}

// ---------------------------------------------------------------------------
// LocalKeyProvider — reads a 32-byte master key from a local file
// ---------------------------------------------------------------------------

/// Reads a 32-byte master key from a local file. DEK is encrypted with the
/// master key using AES-256-GCM. Good for development and self-hosted
/// deployments.
pub struct LocalKeyProvider {
    master_key: [u8; 32],
}

impl LocalKeyProvider {
    /// Create a new `LocalKeyProvider` by reading a 32-byte key from `path`.
    ///
    /// The file must contain exactly 32 bytes (raw binary).
    pub fn from_file(path: &Path) -> GalaxResult<Self> {
        let data = std::fs::read(path).map_err(|e| {
            galaxdb_common::GalaxError::Encryption(format!(
                "failed to read key file {}: {e}",
                path.display()
            ))
        })?;

        if data.len() != 32 {
            return Err(galaxdb_common::GalaxError::Encryption(format!(
                "key file must be exactly 32 bytes, got {}",
                data.len()
            )));
        }

        let mut master_key = [0u8; 32];
        master_key.copy_from_slice(&data);
        Ok(Self { master_key })
    }

    /// Create a `LocalKeyProvider` directly from a 32-byte key (useful for tests).
    pub fn from_key(master_key: [u8; 32]) -> Self {
        Self { master_key }
    }
}

impl KeyProvider for LocalKeyProvider {
    fn generate_data_key(&self) -> GalaxResult<(Vec<u8>, Vec<u8>)> {
        let mut plaintext = vec![0u8; 32];
        rand::thread_rng().fill_bytes(&mut plaintext);

        let encrypted = encrypt_dek_with_master(&self.master_key, &plaintext)?;
        Ok((plaintext, encrypted))
    }

    fn decrypt_data_key(&self, encrypted_key: &[u8]) -> GalaxResult<Vec<u8>> {
        decrypt_dek_with_master(&self.master_key, encrypted_key)
    }

    fn provider_name(&self) -> &str {
        "local-file"
    }
}

// ---------------------------------------------------------------------------
// EnvKeyProvider — reads a hex-encoded 32-byte master key from env var
// ---------------------------------------------------------------------------

/// Reads a hex-encoded 32-byte master key from the `GALAXDB_MASTER_KEY`
/// environment variable. Good for containerized deployments.
pub struct EnvKeyProvider {
    master_key: [u8; 32],
}

impl EnvKeyProvider {
    /// The default environment variable name.
    pub const ENV_VAR: &'static str = "GALAXDB_MASTER_KEY";

    /// Create a new `EnvKeyProvider` by reading from the default env var.
    pub fn from_env() -> GalaxResult<Self> {
        Self::from_env_var(Self::ENV_VAR)
    }

    /// Create a new `EnvKeyProvider` by reading from a custom env var name.
    pub fn from_env_var(var_name: &str) -> GalaxResult<Self> {
        let hex_str = std::env::var(var_name).map_err(|_| {
            galaxdb_common::GalaxError::Encryption(format!(
                "environment variable {var_name} not set"
            ))
        })?;

        let bytes = hex_decode(&hex_str).map_err(|e| {
            galaxdb_common::GalaxError::Encryption(format!(
                "invalid hex in {var_name}: {e}"
            ))
        })?;

        if bytes.len() != 32 {
            return Err(galaxdb_common::GalaxError::Encryption(format!(
                "{var_name} must decode to exactly 32 bytes, got {}",
                bytes.len()
            )));
        }

        let mut master_key = [0u8; 32];
        master_key.copy_from_slice(&bytes);
        Ok(Self { master_key })
    }

    /// Create an `EnvKeyProvider` directly from a 32-byte key (useful for tests).
    pub fn from_key(master_key: [u8; 32]) -> Self {
        Self { master_key }
    }
}

impl KeyProvider for EnvKeyProvider {
    fn generate_data_key(&self) -> GalaxResult<(Vec<u8>, Vec<u8>)> {
        let mut plaintext = vec![0u8; 32];
        rand::thread_rng().fill_bytes(&mut plaintext);

        let encrypted = encrypt_dek_with_master(&self.master_key, &plaintext)?;
        Ok((plaintext, encrypted))
    }

    fn decrypt_data_key(&self, encrypted_key: &[u8]) -> GalaxResult<Vec<u8>> {
        decrypt_dek_with_master(&self.master_key, encrypted_key)
    }

    fn provider_name(&self) -> &str {
        "env-var"
    }
}

// ---------------------------------------------------------------------------
// AwsKmsKeyProvider — stub behind feature flag
// ---------------------------------------------------------------------------

/// Stub AWS KMS key provider. Only available when the `aws-kms` feature is
/// enabled. Without the feature, this struct exists but all operations return
/// an error explaining that the feature must be enabled.
#[cfg(feature = "aws-kms")]
pub struct AwsKmsKeyProvider {
    _key_arn: String,
}

#[cfg(feature = "aws-kms")]
impl AwsKmsKeyProvider {
    /// Create a new `AwsKmsKeyProvider` with the given KMS key ARN.
    ///
    /// Note: This is a stub. A real implementation would initialise the
    /// AWS SDK KMS client here.
    pub fn new(key_arn: String) -> Self {
        Self { _key_arn: key_arn }
    }
}

#[cfg(feature = "aws-kms")]
impl KeyProvider for AwsKmsKeyProvider {
    fn generate_data_key(&self) -> GalaxResult<(Vec<u8>, Vec<u8>)> {
        // Stub: a real implementation would call kms:GenerateDataKey
        Err(galaxdb_common::GalaxError::Kms(
            "AWS KMS GenerateDataKey not yet implemented — this is a stub".to_string(),
        ))
    }

    fn decrypt_data_key(&self, _encrypted_key: &[u8]) -> GalaxResult<Vec<u8>> {
        // Stub: a real implementation would call kms:Decrypt
        Err(galaxdb_common::GalaxError::Kms(
            "AWS KMS Decrypt not yet implemented — this is a stub".to_string(),
        ))
    }

    fn provider_name(&self) -> &str {
        "aws-kms"
    }
}

// ---------------------------------------------------------------------------
// Minimal hex decode (avoids adding a `hex` crate dependency)
// ---------------------------------------------------------------------------

fn hex_decode(s: &str) -> Result<Vec<u8>, String> {
    let s = s.trim();
    if s.len() % 2 != 0 {
        return Err("odd number of hex characters".to_string());
    }
    (0..s.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&s[i..i + 2], 16)
                .map_err(|e| format!("invalid hex at position {i}: {e}"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_decode_valid() {
        assert_eq!(hex_decode("deadbeef").unwrap(), vec![0xde, 0xad, 0xbe, 0xef]);
        assert_eq!(hex_decode("00ff").unwrap(), vec![0x00, 0xff]);
    }

    #[test]
    fn hex_decode_odd_length() {
        assert!(hex_decode("abc").is_err());
    }

    #[test]
    fn local_key_provider_round_trip() {
        let master = [0x42u8; 32];
        let provider = LocalKeyProvider::from_key(master);
        assert_eq!(provider.provider_name(), "local-file");

        let (plaintext, encrypted) = provider.generate_data_key().unwrap();
        assert_eq!(plaintext.len(), 32);
        assert_ne!(plaintext, encrypted); // encrypted is longer (nonce + tag)

        let decrypted = provider.decrypt_data_key(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn env_key_provider_round_trip() {
        let master = [0x77u8; 32];
        let provider = EnvKeyProvider::from_key(master);
        assert_eq!(provider.provider_name(), "env-var");

        let (plaintext, encrypted) = provider.generate_data_key().unwrap();
        let decrypted = provider.decrypt_data_key(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn decrypt_with_wrong_key_fails() {
        let provider_a = LocalKeyProvider::from_key([0x11u8; 32]);
        let provider_b = LocalKeyProvider::from_key([0x22u8; 32]);

        let (_plaintext, encrypted) = provider_a.generate_data_key().unwrap();
        assert!(provider_b.decrypt_data_key(&encrypted).is_err());
    }

    #[test]
    fn decrypt_truncated_data_fails() {
        let provider = LocalKeyProvider::from_key([0x33u8; 32]);
        assert!(provider.decrypt_data_key(&[0u8; 5]).is_err());
    }

    #[test]
    fn local_key_provider_from_file_wrong_size() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad_key");
        std::fs::write(&path, &[0u8; 16]).unwrap(); // 16 bytes, not 32
        assert!(LocalKeyProvider::from_file(&path).is_err());
    }

    #[test]
    fn local_key_provider_from_file_valid() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("good_key");
        std::fs::write(&path, &[0xAAu8; 32]).unwrap();
        let provider = LocalKeyProvider::from_file(&path).unwrap();

        let (pt, enc) = provider.generate_data_key().unwrap();
        let dec = provider.decrypt_data_key(&enc).unwrap();
        assert_eq!(pt, dec);
    }
}
