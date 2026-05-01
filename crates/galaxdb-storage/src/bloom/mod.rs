//! Bloom filters with Monkey-optimal FPR allocation for GalaxDB.
//!
//! Each SST file carries a Bloom filter to avoid unnecessary disk reads during
//! point lookups. The false-positive rate (FPR) per LSM level is allocated using
//! the Monkey-optimal strategy (Dayan et al., TODS 2018), which concentrates
//! Bloom filter memory on larger, colder levels where false positives are most
//! expensive.
//!
//! ## Implementation Details
//!
//! - **Hash function:** Double hashing using XXH3-64. Two base hashes `h1` and
//!   `h2` are derived from a single XXH3-64 call (upper and lower 32 bits),
//!   then `h_i(x) = h1 + i * h2` for each of `k` hash functions.
//! - **Serialization:** The filter is stored as a header (num_bits, num_hashes)
//!   followed by the raw bit vector bytes, suitable for embedding in SST files.
//! - **Monkey allocation:** `MonkeyAllocator` computes the optimal FPR for each
//!   LSM level given a total memory budget (bits per key).

#[cfg(test)]
mod tests;

use xxhash_rust::xxh3::xxh3_64;

// ---------------------------------------------------------------------------
// BloomFilter
// ---------------------------------------------------------------------------

/// A standard Bloom filter using double-hashing over XXH3-64.
#[derive(Debug, Clone)]
pub struct BloomFilter {
    /// The bit vector backing the filter.
    bits: Vec<u8>,
    /// Total number of bits in the filter.
    num_bits: u64,
    /// Number of hash functions (k).
    num_hashes: u32,
}

impl BloomFilter {
    /// Create a new Bloom filter sized for `num_keys` keys at the given
    /// `false_positive_rate`.
    ///
    /// # Panics
    ///
    /// Panics if `num_keys` is 0 or `false_positive_rate` is not in (0, 1).
    pub fn new(num_keys: usize, false_positive_rate: f64) -> Self {
        assert!(num_keys > 0, "num_keys must be > 0");
        assert!(
            false_positive_rate > 0.0 && false_positive_rate < 1.0,
            "false_positive_rate must be in (0, 1)"
        );

        // Optimal number of bits: m = -n * ln(p) / (ln(2)^2)
        let m = (-(num_keys as f64) * false_positive_rate.ln() / (2.0_f64.ln().powi(2)))
            .ceil()
            .max(8.0) as u64;

        // Optimal number of hash functions: k = (m / n) * ln(2)
        let k = ((m as f64 / num_keys as f64) * 2.0_f64.ln())
            .round()
            .max(1.0) as u32;

        let byte_len = m.div_ceil(8) as usize;

        Self {
            bits: vec![0u8; byte_len],
            num_bits: m,
            num_hashes: k,
        }
    }

    /// Create a Bloom filter with an explicit bits-per-key setting.
    ///
    /// The number of hash functions is derived optimally: `k = bits_per_key * ln(2)`.
    pub fn with_bits_per_key(num_keys: usize, bits_per_key: u32) -> Self {
        assert!(num_keys > 0, "num_keys must be > 0");
        assert!(bits_per_key > 0, "bits_per_key must be > 0");

        let m = (num_keys as u64) * (bits_per_key as u64);
        let m = m.max(8);

        let k = ((bits_per_key as f64) * 2.0_f64.ln())
            .round()
            .max(1.0) as u32;

        let byte_len = m.div_ceil(8) as usize;

        Self {
            bits: vec![0u8; byte_len],
            num_bits: m,
            num_hashes: k,
        }
    }

    /// Insert a key into the Bloom filter.
    pub fn insert(&mut self, key: &[u8]) {
        let (h1, h2) = self.hash_pair(key);
        for i in 0..self.num_hashes {
            let bit_index = self.bit_index(h1, h2, i);
            self.set_bit(bit_index);
        }
    }

    /// Check whether a key *might* be in the set.
    ///
    /// Returns `true` if the key might be present (possible false positive),
    /// or `false` if the key is definitely absent.
    pub fn may_contain(&self, key: &[u8]) -> bool {
        let (h1, h2) = self.hash_pair(key);
        for i in 0..self.num_hashes {
            let bit_index = self.bit_index(h1, h2, i);
            if !self.get_bit(bit_index) {
                return false;
            }
        }
        true
    }

    /// Return the number of bits in this filter.
    pub fn num_bits(&self) -> u64 {
        self.num_bits
    }

    /// Return the number of hash functions.
    pub fn num_hashes(&self) -> u32 {
        self.num_hashes
    }

    /// Serialize the Bloom filter to bytes.
    ///
    /// Format: `[num_bits: u64][num_hashes: u32][bit_vector: bytes]`
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(12 + self.bits.len());
        buf.extend_from_slice(&self.num_bits.to_le_bytes());
        buf.extend_from_slice(&self.num_hashes.to_le_bytes());
        buf.extend_from_slice(&self.bits);
        buf
    }

    /// Deserialize a Bloom filter from bytes.
    ///
    /// Returns `None` if the data is too short or inconsistent.
    pub fn deserialize(data: &[u8]) -> Option<Self> {
        if data.len() < 12 {
            return None;
        }

        let num_bits = u64::from_le_bytes(data[0..8].try_into().ok()?);
        let num_hashes = u32::from_le_bytes(data[8..12].try_into().ok()?);

        let expected_byte_len = num_bits.div_ceil(8) as usize;
        if data.len() < 12 + expected_byte_len {
            return None;
        }

        let bits = data[12..12 + expected_byte_len].to_vec();

        Some(Self {
            bits,
            num_bits,
            num_hashes,
        })
    }

    // -- Internal helpers --

    /// Compute two base hashes from a key using XXH3-64.
    ///
    /// We derive h1 and h2 by hashing the key with two different seeds.
    fn hash_pair(&self, key: &[u8]) -> (u64, u64) {
        let h1 = xxh3_64(key);
        // Use a different seed by appending a fixed byte to get h2.
        // This is a common approach for double hashing with a single hash function.
        let h2 = xxh3_64(&[key, &[0x47]].concat());
        (h1, h2)
    }

    /// Compute the bit index for the i-th hash function using double hashing.
    fn bit_index(&self, h1: u64, h2: u64, i: u32) -> u64 {
        h1.wrapping_add((i as u64).wrapping_mul(h2)) % self.num_bits
    }

    fn set_bit(&mut self, index: u64) {
        let byte_idx = (index / 8) as usize;
        let bit_idx = (index % 8) as u8;
        self.bits[byte_idx] |= 1 << bit_idx;
    }

    fn get_bit(&self, index: u64) -> bool {
        let byte_idx = (index / 8) as usize;
        let bit_idx = (index % 8) as u8;
        (self.bits[byte_idx] >> bit_idx) & 1 == 1
    }
}

// ---------------------------------------------------------------------------
// MonkeyAllocator
// ---------------------------------------------------------------------------

/// Monkey-optimal FPR allocator for LSM-tree Bloom filters.
///
/// Allocates false-positive rates across LSM levels using the formula:
///
/// ```text
/// FPR(level_i) = total_fpr_budget * (size_ratio^(L-i)) / sum(size_ratio^(L-j) for j in 0..L)
/// ```
///
/// This concentrates Bloom filter memory on larger, colder levels where false
/// positives are most expensive.
#[derive(Debug, Clone)]
pub struct MonkeyAllocator {
    /// Total FPR budget across all levels.
    total_fpr_budget: f64,
    /// LSM size ratio (default 10).
    size_ratio: u32,
}

impl MonkeyAllocator {
    /// Create a new Monkey allocator.
    ///
    /// # Arguments
    ///
    /// * `bits_per_key` — Total memory budget in bits per key (default 10).
    /// * `size_ratio` — LSM size ratio (default 10).
    pub fn new(bits_per_key: u32, size_ratio: u32) -> Self {
        // Convert bits-per-key to a total FPR budget.
        // For a standard Bloom filter: FPR ≈ (1 - e^(-k*n/m))^k
        // With optimal k: FPR ≈ 2^(-m/n * ln(2)) = 2^(-bits_per_key * ln(2))
        // ≈ 0.6185^bits_per_key
        let total_fpr_budget = 0.6185_f64.powi(bits_per_key as i32);

        Self {
            total_fpr_budget,
            size_ratio,
        }
    }

    /// Create a Monkey allocator with an explicit total FPR budget.
    pub fn with_fpr_budget(total_fpr_budget: f64, size_ratio: u32) -> Self {
        assert!(
            total_fpr_budget > 0.0 && total_fpr_budget < 1.0,
            "total_fpr_budget must be in (0, 1)"
        );
        Self {
            total_fpr_budget,
            size_ratio,
        }
    }

    /// Return the total FPR budget.
    pub fn total_fpr_budget(&self) -> f64 {
        self.total_fpr_budget
    }

    /// Return the size ratio.
    pub fn size_ratio(&self) -> u32 {
        self.size_ratio
    }

    /// Compute the optimal FPR for a given level.
    ///
    /// # Arguments
    ///
    /// * `level` — The LSM level index (0-based).
    /// * `num_levels` — Total number of LSM levels (L).
    ///
    /// # Returns
    ///
    /// The allocated FPR for this level, clamped to [1e-10, 0.5] for safety.
    pub fn fpr_for_level(&self, level: usize, num_levels: usize) -> f64 {
        assert!(num_levels > 0, "num_levels must be > 0");
        assert!(level < num_levels, "level must be < num_levels");

        let ratio = self.size_ratio as f64;
        let l = num_levels;

        // Denominator: sum(ratio^(L-j) for j in 0..L)
        let denominator: f64 = (0..l).map(|j| ratio.powi((l - j) as i32)).sum();

        // Numerator: ratio^(L - level)
        let numerator = ratio.powi((l - level) as i32);

        let fpr = self.total_fpr_budget * numerator / denominator;

        // Clamp to a safe range
        fpr.clamp(1e-10, 0.5)
    }

    /// Compute FPRs for all levels at once.
    ///
    /// Returns a vector of length `num_levels` where index `i` is the FPR
    /// for level `i`.
    pub fn allocate_all(&self, num_levels: usize) -> Vec<f64> {
        (0..num_levels)
            .map(|level| self.fpr_for_level(level, num_levels))
            .collect()
    }

    /// Compute the optimal bits-per-key for a given level's FPR.
    ///
    /// Uses the formula: `bits_per_key = -log2(fpr) / ln(2)`
    /// which is the inverse of the optimal Bloom filter sizing.
    pub fn bits_per_key_for_level(&self, level: usize, num_levels: usize) -> u32 {
        let fpr = self.fpr_for_level(level, num_levels);
        // m/n = -ln(fpr) / (ln(2)^2)
        let bpk = -fpr.ln() / (2.0_f64.ln().powi(2));
        bpk.ceil().max(1.0) as u32
    }
}

// ---------------------------------------------------------------------------
// SstBloomFilter — wraps a BloomFilter with SST metadata
// ---------------------------------------------------------------------------

/// A Bloom filter associated with a specific SST file.
///
/// This struct wraps a `BloomFilter` with SST-level metadata and provides
/// the `check_key()` method used in the point read path.
#[derive(Debug, Clone)]
pub struct SstBloomFilter {
    /// The SST file identifier.
    pub sst_id: u64,
    /// The LSM level this SST belongs to.
    pub level: usize,
    /// The underlying Bloom filter.
    filter: BloomFilter,
}

impl SstBloomFilter {
    /// Build a Bloom filter for an SST file from its keys.
    ///
    /// # Arguments
    ///
    /// * `sst_id` — The SST file identifier.
    /// * `level` — The LSM level this SST belongs to.
    /// * `keys` — Iterator over all keys in the SST.
    /// * `false_positive_rate` — The target FPR (from Monkey allocation).
    pub fn build<'a, I>(sst_id: u64, level: usize, keys: I, false_positive_rate: f64) -> Self
    where
        I: ExactSizeIterator<Item = &'a [u8]>,
    {
        let num_keys = keys.len();
        let mut filter = BloomFilter::new(num_keys.max(1), false_positive_rate);
        for key in keys {
            filter.insert(key);
        }
        Self {
            sst_id,
            level,
            filter,
        }
    }

    /// Build a Bloom filter for an SST file using bits-per-key.
    pub fn build_with_bits_per_key<'a, I>(
        sst_id: u64,
        level: usize,
        keys: I,
        bits_per_key: u32,
    ) -> Self
    where
        I: ExactSizeIterator<Item = &'a [u8]>,
    {
        let num_keys = keys.len();
        let mut filter = BloomFilter::with_bits_per_key(num_keys.max(1), bits_per_key);
        for key in keys {
            filter.insert(key);
        }
        Self {
            sst_id,
            level,
            filter,
        }
    }

    /// Check whether a key *might* exist in this SST.
    ///
    /// Returns `true` if the key might be present (consult the SST),
    /// or `false` if the key is definitely absent (skip the SST).
    pub fn check_key(&self, key: &[u8]) -> bool {
        self.filter.may_contain(key)
    }

    /// Serialize the SST Bloom filter (including metadata) to bytes.
    ///
    /// Format: `[sst_id: u64][level: u32][filter_bytes]`
    pub fn serialize(&self) -> Vec<u8> {
        let filter_bytes = self.filter.serialize();
        let mut buf = Vec::with_capacity(12 + filter_bytes.len());
        buf.extend_from_slice(&self.sst_id.to_le_bytes());
        buf.extend_from_slice(&(self.level as u32).to_le_bytes());
        buf.extend_from_slice(&filter_bytes);
        buf
    }

    /// Deserialize an SST Bloom filter from bytes.
    pub fn deserialize(data: &[u8]) -> Option<Self> {
        if data.len() < 12 {
            return None;
        }

        let sst_id = u64::from_le_bytes(data[0..8].try_into().ok()?);
        let level = u32::from_le_bytes(data[8..12].try_into().ok()?) as usize;
        let filter = BloomFilter::deserialize(&data[12..])?;

        Some(Self {
            sst_id,
            level,
            filter,
        })
    }
}

// ---------------------------------------------------------------------------
// Point read integration helper
// ---------------------------------------------------------------------------

/// Check a key against a set of SST Bloom filters and return only the SST IDs
/// that might contain the key (i.e., where the Bloom filter did not rule it out).
///
/// This is the core integration point for the read path: before performing any
/// disk reads, the caller passes candidate SSTs through this function to skip
/// those that definitely do not contain the target key.
pub fn filter_candidate_ssts(key: &[u8], filters: &[SstBloomFilter]) -> Vec<u64> {
    filters
        .iter()
        .filter(|f| f.check_key(key))
        .map(|f| f.sst_id)
        .collect()
}
