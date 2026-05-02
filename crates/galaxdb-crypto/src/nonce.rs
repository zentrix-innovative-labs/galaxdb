//! Counter-based 96-bit nonce generation.
//!
//! Each nonce is 12 bytes: a 4-byte random prefix (generated once at
//! construction time) concatenated with an 8-byte monotonic counter.
//! The random prefix prevents nonce reuse across engine restarts; the
//! counter ensures uniqueness within a single run.

use rand::RngCore;
use std::sync::atomic::{AtomicU64, Ordering};

/// Thread-safe counter-based 96-bit nonce generator.
///
/// Nonce layout (12 bytes / 96 bits):
/// ```text
/// [ random_prefix: 4 bytes ][ counter: 8 bytes (big-endian) ]
/// ```
pub struct NonceGenerator {
    /// Random prefix generated once at construction time.
    prefix: [u8; 4],
    /// Monotonically increasing counter.
    counter: AtomicU64,
}

impl NonceGenerator {
    /// Create a new `NonceGenerator` with a random 4-byte prefix.
    pub fn new() -> Self {
        let mut prefix = [0u8; 4];
        rand::thread_rng().fill_bytes(&mut prefix);
        Self {
            prefix,
            counter: AtomicU64::new(0),
        }
    }

    /// Create a `NonceGenerator` with a specific prefix and starting counter
    /// (useful for deterministic tests).
    pub fn with_prefix_and_counter(prefix: [u8; 4], start: u64) -> Self {
        Self {
            prefix,
            counter: AtomicU64::new(start),
        }
    }

    /// Generate the next unique 12-byte nonce.
    ///
    /// This is safe to call from multiple threads concurrently.
    pub fn next_nonce(&self) -> [u8; 12] {
        let count = self.counter.fetch_add(1, Ordering::Relaxed);
        let mut nonce = [0u8; 12];
        nonce[..4].copy_from_slice(&self.prefix);
        nonce[4..12].copy_from_slice(&count.to_be_bytes());
        nonce
    }

    /// Return the current counter value (for diagnostics / testing).
    pub fn current_counter(&self) -> u64 {
        self.counter.load(Ordering::Relaxed)
    }

    /// Return the random prefix (for diagnostics / testing).
    pub fn prefix(&self) -> [u8; 4] {
        self.prefix
    }
}

impl Default for NonceGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn nonces_are_unique() {
        let gen = NonceGenerator::new();
        let mut seen = HashSet::new();
        for _ in 0..10_000 {
            let nonce = gen.next_nonce();
            assert!(seen.insert(nonce), "duplicate nonce detected");
        }
    }

    #[test]
    fn nonce_has_correct_layout() {
        let prefix = [0xDE, 0xAD, 0xBE, 0xEF];
        let gen = NonceGenerator::with_prefix_and_counter(prefix, 0);

        let n0 = gen.next_nonce();
        assert_eq!(&n0[..4], &prefix);
        assert_eq!(&n0[4..12], &0u64.to_be_bytes());

        let n1 = gen.next_nonce();
        assert_eq!(&n1[..4], &prefix);
        assert_eq!(&n1[4..12], &1u64.to_be_bytes());
    }

    #[test]
    fn counter_increments() {
        let gen = NonceGenerator::with_prefix_and_counter([0; 4], 42);
        assert_eq!(gen.current_counter(), 42);
        let _ = gen.next_nonce();
        assert_eq!(gen.current_counter(), 43);
    }

    #[test]
    fn concurrent_nonce_uniqueness() {
        use std::sync::Arc;
        let gen = Arc::new(NonceGenerator::new());
        let mut handles = Vec::new();

        for _ in 0..4 {
            let g = Arc::clone(&gen);
            handles.push(std::thread::spawn(move || {
                let mut nonces = Vec::with_capacity(1000);
                for _ in 0..1000 {
                    nonces.push(g.next_nonce());
                }
                nonces
            }));
        }

        let mut all = HashSet::new();
        for h in handles {
            for nonce in h.join().unwrap() {
                assert!(all.insert(nonce), "duplicate nonce across threads");
            }
        }
        assert_eq!(all.len(), 4000);
    }
}
