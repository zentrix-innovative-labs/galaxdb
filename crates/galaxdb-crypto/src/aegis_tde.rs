//! AEGIS-256 Transparent Data Encryption module.
//!
//! AEGIS-256 achieves 10-15 GB/s per core on modern CPUs with AES-NI —
//! 3-4x faster than AES-256-GCM. Used for PAX block encryption.
//! WAL continues to use AES-256-GCM (append-only sequential writes).

use aegis::aegis256;
use galaxdb_common::{GalaxError, GalaxResult};

use crate::key_provider::KeyProvider;

/// AEGIS-256 tag size: 32 bytes (256-bit authentication tag).
const TAG_BYTES: usize = 32;
/// AEGIS-256 nonce size: 32 bytes.
const NONCE_SIZE: usize = 32;

/// AEGIS-256 TDE module for PAX block encryption.
pub struct AegisTdeModule {
    key: aegis256::Key,
    encrypted_key: Vec<u8>,
    nonce_counter: std::sync::atomic::AtomicU64,
    nonce_prefix: [u8; 24],
}

impl AegisTdeModule {
    /// Create a new AEGIS-256 TDE module with a fresh key.
    pub fn new(key_provider: &dyn KeyProvider) -> GalaxResult<Self> {
        let (plaintext_vec, encrypted_key) = key_provider.generate_data_key()?;
        if plaintext_vec.len() != 32 {
            return Err(GalaxError::Encryption(format!(
                "AEGIS-256 key must be 32 bytes, got {}", plaintext_vec.len()
            )));
        }
        let mut key = [0u8; 32];
        key.copy_from_slice(&plaintext_vec);

        let mut nonce_prefix = [0u8; 24];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut nonce_prefix);

        Ok(Self {
            key,
            encrypted_key,
            nonce_counter: std::sync::atomic::AtomicU64::new(0),
            nonce_prefix,
        })
    }

    /// Create from an existing encrypted key (for recovery).
    pub fn from_encrypted_key(
        key_provider: &dyn KeyProvider,
        encrypted_key: &[u8],
    ) -> GalaxResult<Self> {
        let plaintext_vec = key_provider.decrypt_data_key(encrypted_key)?;
        if plaintext_vec.len() != 32 {
            return Err(GalaxError::Encryption(format!(
                "decrypted key must be 32 bytes, got {}", plaintext_vec.len()
            )));
        }
        let mut key = [0u8; 32];
        key.copy_from_slice(&plaintext_vec);

        let mut nonce_prefix = [0u8; 24];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut nonce_prefix);

        Ok(Self {
            key,
            encrypted_key: encrypted_key.to_vec(),
            nonce_counter: std::sync::atomic::AtomicU64::new(0),
            nonce_prefix,
        })
    }

    fn next_nonce(&self) -> aegis256::Nonce {
        let counter = self.nonce_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut nonce = [0u8; NONCE_SIZE];
        nonce[..24].copy_from_slice(&self.nonce_prefix);
        nonce[24..32].copy_from_slice(&counter.to_be_bytes());
        nonce
    }

    /// Encrypt a plaintext block.
    /// Returns `nonce (32) || ciphertext || tag (32)`.
    pub fn encrypt(&self, plaintext: &[u8]) -> GalaxResult<Vec<u8>> {
        let nonce = self.next_nonce();

        // Create a new cipher instance for each encryption (AEGIS consumes self)
        let cipher = aegis256::Aegis256::<TAG_BYTES>::new(&self.key, &nonce);
        let (ciphertext, tag) = cipher.encrypt(plaintext, &[]);

        let mut out = Vec::with_capacity(NONCE_SIZE + ciphertext.len() + TAG_BYTES);
        out.extend_from_slice(&nonce);
        out.extend_from_slice(&ciphertext);
        out.extend_from_slice(&tag);
        Ok(out)
    }

    /// Decrypt a ciphertext block.
    /// Expects: `nonce (32) || ciphertext || tag (32)`.
    pub fn decrypt(&self, encrypted: &[u8]) -> GalaxResult<Vec<u8>> {
        if encrypted.len() < NONCE_SIZE + TAG_BYTES {
            return Err(GalaxError::Encryption("AEGIS-256 data too short".to_string()));
        }

        let nonce: aegis256::Nonce = encrypted[..NONCE_SIZE].try_into()
            .map_err(|_| GalaxError::Encryption("invalid nonce".to_string()))?;
        let tag_start = encrypted.len() - TAG_BYTES;
        let ciphertext = &encrypted[NONCE_SIZE..tag_start];
        let tag: aegis256::Tag<TAG_BYTES> = encrypted[tag_start..].try_into()
            .map_err(|_| GalaxError::Encryption("invalid tag".to_string()))?;

        let cipher = aegis256::Aegis256::<TAG_BYTES>::new(&self.key, &nonce);
        cipher.decrypt(ciphertext, &tag, &[])
            .map_err(|_| GalaxError::Encryption("AEGIS-256 authentication failed".to_string()))
    }

    pub fn encrypted_key(&self) -> &[u8] { &self.encrypted_key }

    pub fn has_hardware_acceleration() -> bool {
        #[cfg(target_arch = "x86_64")]
        { std::arch::is_x86_feature_detected!("aes") }
        #[cfg(not(target_arch = "x86_64"))]
        { false }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::key_provider::LocalKeyProvider;

    fn make_aegis() -> AegisTdeModule {
        let provider = LocalKeyProvider::from_key([0xABu8; 32]);
        AegisTdeModule::new(&provider).unwrap()
    }

    #[test]
    fn encrypt_decrypt_roundtrip_empty() {
        let a = make_aegis();
        let ct = a.encrypt(b"").unwrap();
        let pt = a.decrypt(&ct).unwrap();
        assert_eq!(pt, b"");
    }

    #[test]
    fn encrypt_decrypt_roundtrip_small() {
        let a = make_aegis();
        let ct = a.encrypt(b"hello AEGIS").unwrap();
        let pt = a.decrypt(&ct).unwrap();
        assert_eq!(pt, b"hello AEGIS");
    }

    #[test]
    fn encrypt_decrypt_roundtrip_1mb() {
        let a = make_aegis();
        let data: Vec<u8> = (0..1_048_576).map(|i| (i % 256) as u8).collect();
        let ct = a.encrypt(&data).unwrap();
        let pt = a.decrypt(&ct).unwrap();
        assert_eq!(pt, data);
    }

    #[test]
    fn different_nonces() {
        let a = make_aegis();
        let ct1 = a.encrypt(b"same").unwrap();
        let ct2 = a.encrypt(b"same").unwrap();
        assert_ne!(ct1, ct2);
        assert_eq!(a.decrypt(&ct1).unwrap(), b"same");
        assert_eq!(a.decrypt(&ct2).unwrap(), b"same");
    }

    #[test]
    fn tampered_fails() {
        let a = make_aegis();
        let mut ct = a.encrypt(b"secret").unwrap();
        ct[40] ^= 0xFF;
        assert!(a.decrypt(&ct).is_err());
    }

    #[test]
    fn truncated_fails() {
        let a = make_aegis();
        assert!(a.decrypt(&[0u8; 10]).is_err());
    }

    #[test]
    fn from_encrypted_key_restores() {
        let master = [0xCCu8; 32];
        let p = LocalKeyProvider::from_key(master);
        let a1 = AegisTdeModule::new(&p).unwrap();
        let ek = a1.encrypted_key().to_vec();
        let ct = a1.encrypt(b"test").unwrap();

        let p2 = LocalKeyProvider::from_key(master);
        let a2 = AegisTdeModule::from_encrypted_key(&p2, &ek).unwrap();
        assert_eq!(a2.decrypt(&ct).unwrap(), b"test");
    }
}
