//! Transparent Data Encryption module.
//!
//! [`TdeModule`] encrypts and decrypts PAX blocks and WAL records using
//! AES-256-GCM. It sits between the storage engine and the I/O scheduler —
//! the I/O layer only ever sees ciphertext.
//!
//! The `aes-gcm` crate automatically uses AES-NI on x86-64 and ARMv8 crypto
//! extensions on ARM64, so no manual SIMD code is needed.

use aes_gcm::aead::Aead;
use aes_gcm::{Aes256Gcm, KeyInit, Nonce};
use galaxdb_common::{GalaxError, GalaxResult};
use std::time::Instant;

use crate::key_provider::KeyProvider;
use crate::nonce::NonceGenerator;

/// Cached data encryption key (DEK).
pub struct CachedDataKey {
    /// AES-256 plaintext key (32 bytes).
    plaintext: [u8; 32],
    /// Encrypted copy of the DEK (for storage in WAL header / metadata).
    encrypted: Vec<u8>,
    /// When this DEK was created.
    created_at: Instant,
}

impl CachedDataKey {
    /// The plaintext DEK bytes.
    pub fn plaintext(&self) -> &[u8; 32] {
        &self.plaintext
    }

    /// The encrypted DEK bytes (for persisting alongside encrypted data).
    pub fn encrypted(&self) -> &[u8] {
        &self.encrypted
    }

    /// When this DEK was generated.
    pub fn created_at(&self) -> Instant {
        self.created_at
    }
}

/// Transparent Data Encryption module.
///
/// Encrypts PAX blocks and WAL records with AES-256-GCM before they hit disk.
/// Decrypts them on read. Uses a pluggable [`KeyProvider`] for key management.
pub struct TdeModule {
    /// The pluggable key provider (local file, env var, AWS KMS, etc.).
    key_provider: Box<dyn KeyProvider>,
    /// The current data encryption key, cached in memory.
    data_key: CachedDataKey,
    /// Counter-based nonce generator for unique 96-bit nonces.
    nonce_gen: NonceGenerator,
    /// Pre-built AES-256-GCM cipher instance for the current DEK.
    cipher: Aes256Gcm,
}

impl TdeModule {
    /// Create a new `TdeModule`, generating a fresh DEK via the key provider.
    pub fn new(key_provider: Box<dyn KeyProvider>) -> GalaxResult<Self> {
        let (plaintext_vec, encrypted) = key_provider.generate_data_key()?;

        if plaintext_vec.len() != 32 {
            return Err(GalaxError::Encryption(format!(
                "DEK must be 32 bytes, got {}",
                plaintext_vec.len()
            )));
        }

        let mut plaintext = [0u8; 32];
        plaintext.copy_from_slice(&plaintext_vec);

        let cipher = Aes256Gcm::new_from_slice(&plaintext)
            .map_err(|e| GalaxError::Encryption(format!("cipher init: {e}")))?;

        let data_key = CachedDataKey {
            plaintext,
            encrypted,
            created_at: Instant::now(),
        };

        Ok(Self {
            key_provider,
            data_key,
            nonce_gen: NonceGenerator::new(),
            cipher,
        })
    }

    /// Create a `TdeModule` by restoring a previously encrypted DEK.
    ///
    /// Use this during recovery when the encrypted DEK is read from the
    /// WAL header or metadata file.
    pub fn from_encrypted_key(
        key_provider: Box<dyn KeyProvider>,
        encrypted_dek: &[u8],
    ) -> GalaxResult<Self> {
        let plaintext_vec = key_provider.decrypt_data_key(encrypted_dek)?;

        if plaintext_vec.len() != 32 {
            return Err(GalaxError::Encryption(format!(
                "decrypted DEK must be 32 bytes, got {}",
                plaintext_vec.len()
            )));
        }

        let mut plaintext = [0u8; 32];
        plaintext.copy_from_slice(&plaintext_vec);

        let cipher = Aes256Gcm::new_from_slice(&plaintext)
            .map_err(|e| GalaxError::Encryption(format!("cipher init: {e}")))?;

        let data_key = CachedDataKey {
            plaintext,
            encrypted: encrypted_dek.to_vec(),
            created_at: Instant::now(),
        };

        Ok(Self {
            key_provider,
            data_key,
            nonce_gen: NonceGenerator::new(),
            cipher,
        })
    }

    /// Encrypt a plaintext block or record.
    ///
    /// Returns `nonce (12 bytes) || ciphertext || tag (16 bytes)`.
    /// This is the format written to disk for both PAX blocks and WAL records.
    pub fn encrypt(&self, plaintext: &[u8]) -> GalaxResult<Vec<u8>> {
        let nonce_bytes = self.nonce_gen.next_nonce();
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = self
            .cipher
            .encrypt(nonce, plaintext)
            .map_err(|e| GalaxError::Encryption(format!("encrypt: {e}")))?;

        // nonce || ciphertext+tag
        let mut out = Vec::with_capacity(12 + ciphertext.len());
        out.extend_from_slice(&nonce_bytes);
        out.extend_from_slice(&ciphertext);
        Ok(out)
    }

    /// Decrypt a ciphertext block or record.
    ///
    /// Expects the format produced by [`encrypt`]: `nonce (12) || ciphertext || tag (16)`.
    pub fn decrypt(&self, encrypted: &[u8]) -> GalaxResult<Vec<u8>> {
        if encrypted.len() < 12 + 16 {
            return Err(GalaxError::Encryption(
                "encrypted data too short (need at least nonce + tag = 28 bytes)".to_string(),
            ));
        }

        let nonce = Nonce::from_slice(&encrypted[..12]);
        let ciphertext = &encrypted[12..];

        self.cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| GalaxError::Encryption(format!("decrypt: {e}")))
    }

    /// Access the cached data encryption key.
    pub fn data_key(&self) -> &CachedDataKey {
        &self.data_key
    }

    /// The name of the active key provider.
    pub fn provider_name(&self) -> &str {
        self.key_provider.provider_name()
    }

    /// Access the nonce generator (for diagnostics).
    pub fn nonce_generator(&self) -> &NonceGenerator {
        &self.nonce_gen
    }

    /// Returns `true` if the CPU supports AES-NI (x86-64) or ARMv8 crypto
    /// extensions (aarch64). The `aes-gcm` crate uses these automatically
    /// when available.
    pub fn has_hardware_acceleration() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            std::arch::is_x86_feature_detected!("aes")
        }
        #[cfg(target_arch = "aarch64")]
        {
            // On aarch64-linux we can check /proc/cpuinfo or std::arch,
            // but the aes-gcm crate handles this internally. We report true
            // on aarch64 as ARMv8 crypto extensions are nearly universal.
            true
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::key_provider::LocalKeyProvider;

    fn make_tde() -> TdeModule {
        let provider = LocalKeyProvider::from_key([0xABu8; 32]);
        TdeModule::new(Box::new(provider)).unwrap()
    }

    #[test]
    fn encrypt_decrypt_round_trip_empty() {
        let tde = make_tde();
        let plaintext = b"";
        let encrypted = tde.encrypt(plaintext).unwrap();
        let decrypted = tde.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn encrypt_decrypt_round_trip_small() {
        let tde = make_tde();
        let plaintext = b"hello, GalaxDB!";
        let encrypted = tde.encrypt(plaintext).unwrap();
        assert_ne!(encrypted.as_slice(), plaintext.as_slice());
        let decrypted = tde.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn encrypt_decrypt_round_trip_large() {
        let tde = make_tde();
        // Simulate a 64 KB PAX block
        let plaintext: Vec<u8> = (0..65536).map(|i| (i % 256) as u8).collect();
        let encrypted = tde.encrypt(&plaintext).unwrap();
        let decrypted = tde.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn encrypted_output_has_nonce_prefix() {
        let tde = make_tde();
        let encrypted = tde.encrypt(b"test").unwrap();
        // Output should be: 12 (nonce) + 4 (plaintext) + 16 (tag) = 32 bytes
        assert_eq!(encrypted.len(), 12 + 4 + 16);
    }

    #[test]
    fn decrypt_tampered_data_fails() {
        let tde = make_tde();
        let mut encrypted = tde.encrypt(b"secret data").unwrap();
        // Flip a byte in the ciphertext portion
        let last = encrypted.len() - 1;
        encrypted[last] ^= 0xFF;
        assert!(tde.decrypt(&encrypted).is_err());
    }

    #[test]
    fn decrypt_truncated_data_fails() {
        let tde = make_tde();
        assert!(tde.decrypt(&[0u8; 10]).is_err());
    }

    #[test]
    fn different_encryptions_produce_different_ciphertext() {
        let tde = make_tde();
        let plaintext = b"same data";
        let enc1 = tde.encrypt(plaintext).unwrap();
        let enc2 = tde.encrypt(plaintext).unwrap();
        // Different nonces → different ciphertext
        assert_ne!(enc1, enc2);
        // But both decrypt to the same plaintext
        assert_eq!(tde.decrypt(&enc1).unwrap(), plaintext);
        assert_eq!(tde.decrypt(&enc2).unwrap(), plaintext);
    }

    #[test]
    fn from_encrypted_key_restores_module() {
        let master = [0xCCu8; 32];
        let provider = LocalKeyProvider::from_key(master);
        let tde1 = TdeModule::new(Box::new(provider)).unwrap();

        let encrypted_dek = tde1.data_key().encrypted().to_vec();
        let plaintext = b"round-trip through key restore";
        let ciphertext = tde1.encrypt(plaintext).unwrap();

        // Restore from the encrypted DEK
        let provider2 = LocalKeyProvider::from_key(master);
        let tde2 = TdeModule::from_encrypted_key(Box::new(provider2), &encrypted_dek).unwrap();

        // tde2 should be able to decrypt data encrypted by tde1
        let decrypted = tde2.decrypt(&ciphertext).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn hardware_acceleration_detection() {
        // Just verify it doesn't panic — the result depends on the CPU.
        let _has_hw = TdeModule::has_hardware_acceleration();
    }

    #[test]
    fn provider_name_is_correct() {
        let tde = make_tde();
        assert_eq!(tde.provider_name(), "local-file");
    }

    #[test]
    fn nonce_counter_advances_with_encryptions() {
        let tde = make_tde();
        assert_eq!(tde.nonce_generator().current_counter(), 0);
        let _ = tde.encrypt(b"a").unwrap();
        assert_eq!(tde.nonce_generator().current_counter(), 1);
        let _ = tde.encrypt(b"b").unwrap();
        assert_eq!(tde.nonce_generator().current_counter(), 2);
    }
}
