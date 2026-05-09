//! MinHash near-duplicate detection (Req 26, design §9.4).
//!
//! A row's text content is reduced to a 512-byte MinHash LSH signature:
//! 128 independent hash functions applied to the set of character
//! trigrams (3-character shingles) of the input, taking the minimum
//! hash per function. Signatures can then be compared to estimate
//! Jaccard similarity in O(128) time.
//!
//! # Signature Size
//!
//! The design document (§9.4, requirement 26.1) fixes the on-disk
//! signature size at 512 bytes per row. With 128 hash functions this
//! yields 128 × 4 = 512 bytes, so each hash value is stored as a
//! `u32` (the low 32 bits of the 64-bit minimum). 32-bit per-hash
//! precision is standard in MinHash LSH practice and is more than
//! sufficient for Jaccard estimation when 128 functions are used.
//!
//! # Hash Family
//!
//! We use a universal family of pairwise-independent hashes:
//!
//! ```text
//! h_i(x) = ((a_i * x + b_i) mod p) for p = 2^61 - 1 (Mersenne prime)
//! ```
//!
//! where `(a_i, b_i)` are 128 random pairs drawn deterministically
//! from a SplitMix64 PRNG seeded by the caller. `a_i` is forced to
//! be non-zero (universality requirement).
//!
//! # Shingling Strategy
//!
//! Input text is decomposed at the **Unicode scalar (`char`) level**,
//! not the byte level, so multi-byte codepoints (emoji, CJK) are
//! never split mid-sequence.
//!
//! | Input length (chars) | Shingles emitted                                   |
//! |----------------------|----------------------------------------------------|
//! | 0                    | none — signature is all `u32::MAX` (sentinel)      |
//! | 1 or 2               | one shingle: the text right-padded with `'\0'`     |
//! | ≥ 3                  | `len - 2` overlapping trigrams (character windows) |
//!
//! The all-`u32::MAX` sentinel for empty strings is the natural MinHash
//! "no observations" state: comparing two empty strings yields a Jaccard
//! estimate of 1.0 (they match in every slot), which is correct — both
//! documents have the same (empty) shingle set.

use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3::xxh3_64;

/// Number of independent hash functions per signature.
pub const NUM_HASHES: usize = 128;

/// On-disk signature size in bytes: `NUM_HASHES * 4` = 512.
pub const SIGNATURE_BYTES: usize = NUM_HASHES * 4;

/// Shingle width in Unicode scalar values (trigrams).
pub const SHINGLE_WIDTH: usize = 3;

/// Mersenne prime `2^61 - 1`, used as the modulus for the universal
/// hash family. Fits in a `u64` and guarantees `a * x + b` fits in
/// `u128` without overflow for any `u64` inputs.
const MERSENNE_61: u64 = (1u64 << 61) - 1;

// ---------------------------------------------------------------------------
// MinHashSignature
// ---------------------------------------------------------------------------

/// A 512-byte MinHash signature: 128 × `u32` hash values.
///
/// Two signatures can be compared with
/// [`MinHashSignature::jaccard_estimate`] to produce an unbiased
/// estimate of the Jaccard similarity between the underlying shingle
/// sets, accurate to roughly `±1/sqrt(128) ≈ 0.088`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MinHashSignature(pub [u32; NUM_HASHES]);

impl MinHashSignature {
    /// Construct a signature from 128 raw `u32` slots.
    #[inline]
    pub fn new(slots: [u32; NUM_HASHES]) -> Self {
        Self(slots)
    }

    /// Return a reference to the raw slots.
    #[inline]
    pub fn slots(&self) -> &[u32; NUM_HASHES] {
        &self.0
    }

    /// Serialize the signature as a 512-byte little-endian buffer.
    pub fn to_bytes(&self) -> [u8; SIGNATURE_BYTES] {
        let mut out = [0u8; SIGNATURE_BYTES];
        for (i, slot) in self.0.iter().enumerate() {
            let off = i * 4;
            out[off..off + 4].copy_from_slice(&slot.to_le_bytes());
        }
        out
    }

    /// Deserialize a signature from a 512-byte little-endian buffer.
    pub fn from_bytes(bytes: &[u8; SIGNATURE_BYTES]) -> Self {
        let mut slots = [0u32; NUM_HASHES];
        for (i, slot) in slots.iter_mut().enumerate() {
            let off = i * 4;
            *slot = u32::from_le_bytes([
                bytes[off],
                bytes[off + 1],
                bytes[off + 2],
                bytes[off + 3],
            ]);
        }
        Self(slots)
    }

    /// Estimate Jaccard similarity against another signature as the
    /// fraction of 128 slots that agree.
    ///
    /// # Statistical Guarantees
    ///
    /// For two shingle sets `A` and `B` with true Jaccard similarity
    /// `J(A, B) = |A ∩ B| / |A ∪ B|`, this estimator returns an
    /// **unbiased** estimate of `J(A, B)`:
    ///
    /// ```text
    /// E[estimate] = J(A, B)
    /// ```
    ///
    /// Each of the 128 slots is an independent Bernoulli(J) trial
    /// (collision probability equals true Jaccard under a universal
    /// hash family), so the estimator is a sample mean with
    /// **variance ≤ J(1 − J) / 128**. The **one-sigma standard error**
    /// is therefore bounded by
    ///
    /// ```text
    /// sqrt(0.25 / 128) ≈ 0.044
    /// ```
    ///
    /// at the worst case (`J = 0.5`). By Chebyshev/Hoeffding bounds,
    /// 128 hash functions yield roughly **±0.1 accuracy ~95% of the
    /// time** for any true similarity level, which is the accuracy
    /// budget this codebase relies on.
    ///
    /// # When to Use MinHash vs Exact Jaccard
    ///
    /// Use MinHash when exact set intersection is too expensive —
    /// i.e. when the shingle sets are large or comparisons must run
    /// at scan speed over many rows, because this call is `O(128)`
    /// regardless of set size. For small sets (a few dozen shingles)
    /// computing the exact Jaccard `|A ∩ B| / |A ∪ B|` over
    /// `HashSet<_>` is cheaper and gives a zero-error answer; prefer
    /// that path in correctness-critical contexts where the ±0.1
    /// estimator budget is unacceptable.
    ///
    /// The result is always in `[0.0, 1.0]`.
    pub fn jaccard_estimate(&self, other: &Self) -> f64 {
        let mut matches = 0usize;
        for i in 0..NUM_HASHES {
            if self.0[i] == other.0[i] {
                matches += 1;
            }
        }
        matches as f64 / NUM_HASHES as f64
    }
}

/// Estimate the Jaccard similarity between two MinHash signatures.
///
/// Free-function convenience wrapper over
/// [`MinHashSignature::jaccard_estimate`]. Intended for call sites
/// (e.g. the background near-duplicate grouping job in task 35.4 and
/// the `WHERE NOT DUPLICATE` query operator in 35.5) that prefer the
/// `estimate_jaccard(a, b)` spelling over method-call syntax.
///
/// # Accuracy
///
/// The estimate is unbiased (E[estimate] = true Jaccard) with
/// one-sigma standard error ≤ `sqrt(0.25 / 128) ≈ 0.044`, giving
/// roughly ±0.1 accuracy ~95 % of the time. See
/// [`MinHashSignature::jaccard_estimate`] for the full statistical
/// analysis.
#[inline]
pub fn estimate_jaccard(a: &MinHashSignature, b: &MinHashSignature) -> f64 {
    a.jaccard_estimate(b)
}

/// Estimate Jaccard similarity directly from two 512-byte serialized
/// signatures, without materializing a [`MinHashSignature`].
///
/// This is the zero-allocation hot path used by query operators that
/// scan `_minhash_signature` bytes straight out of a column store:
/// each 4-byte little-endian `u32` slot is read in place from both
/// buffers via [`u32::from_le_bytes`] and compared. No heap
/// allocation, no intermediate copy.
///
/// The result is algebraically identical to
/// [`MinHashSignature::jaccard_estimate`] on the deserialized
/// signatures — see `jaccard_from_bytes_matches_method` in the test
/// suite, which pins this equivalence.
///
/// Statistical properties match [`MinHashSignature::jaccard_estimate`]:
/// the estimate is unbiased, with one-sigma standard error
/// ≤ `sqrt(0.25 / 128) ≈ 0.044`.
#[inline]
pub fn jaccard_estimate_from_bytes(
    a: &[u8; SIGNATURE_BYTES],
    b: &[u8; SIGNATURE_BYTES],
) -> f64 {
    let mut matches = 0usize;
    // Stride 4 bytes at a time — 128 iterations, no allocation.
    let mut i = 0usize;
    while i < SIGNATURE_BYTES {
        let slot_a = u32::from_le_bytes([a[i], a[i + 1], a[i + 2], a[i + 3]]);
        let slot_b = u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
        if slot_a == slot_b {
            matches += 1;
        }
        i += 4;
    }
    matches as f64 / NUM_HASHES as f64
}

// Manual Serialize/Deserialize: emit as a 512-byte array so the on-disk
// representation is compact and version-stable regardless of serde's
// const-generic support. This matches the bytes produced by `to_bytes`.
impl Serialize for MinHashSignature {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        let bytes = self.to_bytes();
        ser.serialize_bytes(&bytes)
    }
}

impl<'de> Deserialize<'de> for MinHashSignature {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = MinHashSignature;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(f, "a 512-byte MinHash signature")
            }

            fn visit_bytes<E: serde::de::Error>(self, v: &[u8]) -> Result<Self::Value, E> {
                if v.len() != SIGNATURE_BYTES {
                    return Err(E::invalid_length(v.len(), &self));
                }
                let mut buf = [0u8; SIGNATURE_BYTES];
                buf.copy_from_slice(v);
                Ok(MinHashSignature::from_bytes(&buf))
            }

            fn visit_byte_buf<E: serde::de::Error>(self, v: Vec<u8>) -> Result<Self::Value, E> {
                self.visit_bytes(&v)
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let mut buf = [0u8; SIGNATURE_BYTES];
                for (i, slot) in buf.iter_mut().enumerate() {
                    *slot = seq
                        .next_element::<u8>()?
                        .ok_or_else(|| serde::de::Error::invalid_length(i, &self))?;
                }
                Ok(MinHashSignature::from_bytes(&buf))
            }
        }
        de.deserialize_bytes(Visitor)
    }
}

// ---------------------------------------------------------------------------
// MinHashDedup
// ---------------------------------------------------------------------------

/// Computes 512-byte MinHash signatures for text rows using 128
/// pairwise-independent hash functions over character trigrams.
///
/// A single `MinHashDedup` instance can be shared across many rows
/// (and threads — it is `Send + Sync`). Two instances constructed
/// with the same seed are guaranteed to produce byte-identical
/// signatures for every input.
#[derive(Debug, Clone)]
pub struct MinHashDedup {
    /// Random `(a, b)` coefficient pairs, one per hash function.
    /// Each `a` is guaranteed non-zero and `< MERSENNE_61`.
    /// Each `b` is in `[0, MERSENNE_61)`.
    pairs: [(u64, u64); NUM_HASHES],
}

impl MinHashDedup {
    /// Construct a deduper with a deterministic seed. Two instances
    /// built from the same seed produce identical signatures.
    pub fn new(seed: u64) -> Self {
        let mut state = seed;
        let mut pairs = [(0u64, 0u64); NUM_HASHES];
        for pair in pairs.iter_mut() {
            let mut a = splitmix64_next(&mut state) % MERSENNE_61;
            if a == 0 {
                // Universality requires a != 0. Collisions are
                // astronomically rare but we handle them for robustness.
                a = 1;
            }
            let b = splitmix64_next(&mut state) % MERSENNE_61;
            *pair = (a, b);
        }
        Self { pairs }
    }

    /// Compute the MinHash signature for `text`.
    ///
    /// See the module-level docs for the shingling strategy for empty,
    /// short (`< 3` chars), and unicode inputs.
    pub fn signature(&self, text: &str) -> MinHashSignature {
        let chars: Vec<char> = text.chars().collect();

        // Empty input → sentinel signature.
        if chars.is_empty() {
            return MinHashSignature([u32::MAX; NUM_HASHES]);
        }

        let shingle_hashes = shingle_hashes(&chars);

        let mut mins = [u64::MAX; NUM_HASHES];
        for &ngram_hash in &shingle_hashes {
            for (i, &(a, b)) in self.pairs.iter().enumerate() {
                let h = universal_hash(ngram_hash, a, b);
                if h < mins[i] {
                    mins[i] = h;
                }
            }
        }

        let mut slots = [0u32; NUM_HASHES];
        for (i, &m) in mins.iter().enumerate() {
            // Low 32 bits of each 61-bit minimum is the stored hash.
            slots[i] = m as u32;
        }
        MinHashSignature(slots)
    }
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// Produce the shingle-hash sequence for a character vector.
///
/// * `chars.len() >= 3` → `chars.len() - 2` overlapping trigrams.
/// * `chars.len() == 1 || 2` → one shingle, right-padded with `'\0'`.
/// * `chars.len() == 0` → caller handles (empty → sentinel signature).
fn shingle_hashes(chars: &[char]) -> Vec<u64> {
    debug_assert!(!chars.is_empty());

    if chars.len() < SHINGLE_WIDTH {
        let mut padded = [0u8; SHINGLE_WIDTH * 4]; // 3 chars × max 4 UTF-8 bytes
        let mut written = 0;
        for i in 0..SHINGLE_WIDTH {
            let c = if i < chars.len() { chars[i] } else { '\0' };
            let n = c.len_utf8();
            c.encode_utf8(&mut padded[written..written + n]);
            written += n;
        }
        return vec![xxh3_64(&padded[..written])];
    }

    chars
        .windows(SHINGLE_WIDTH)
        .map(hash_char_window)
        .collect()
}

/// Hash a 3-char window at the codepoint level (UTF-8 encoded).
#[inline]
fn hash_char_window(window: &[char]) -> u64 {
    let mut buf = [0u8; SHINGLE_WIDTH * 4]; // 3 chars × max 4 UTF-8 bytes
    let mut written = 0;
    for &c in window {
        let n = c.len_utf8();
        c.encode_utf8(&mut buf[written..written + n]);
        written += n;
    }
    xxh3_64(&buf[..written])
}

/// Evaluate `h(x) = ((a * x + b) mod p)` with `p = 2^61 - 1`.
///
/// `u128` arithmetic ensures the multiply cannot overflow.
#[inline]
fn universal_hash(x: u64, a: u64, b: u64) -> u64 {
    let x = (x as u128) % (MERSENNE_61 as u128);
    let m = (a as u128).wrapping_mul(x).wrapping_add(b as u128);
    (m % (MERSENNE_61 as u128)) as u64
}

/// SplitMix64 — a tiny, deterministic PRNG. Seeded from the caller's
/// `u64` seed; each call advances the state and returns 64 random bits.
#[inline]
fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_deterministic_with_seed() {
        let a = MinHashDedup::new(42);
        let b = MinHashDedup::new(42);
        let text = "the quick brown fox jumps over the lazy dog";
        assert_eq!(a.signature(text), b.signature(text));

        // And identical internal pairs:
        assert_eq!(a.pairs, b.pairs);
    }

    #[test]
    fn different_seeds_produce_different_signatures() {
        let a = MinHashDedup::new(1);
        let b = MinHashDedup::new(2);
        let text = "the quick brown fox jumps over the lazy dog";
        let sig_a = a.signature(text);
        let sig_b = b.signature(text);

        let differing = (0..NUM_HASHES)
            .filter(|i| sig_a.0[*i] != sig_b.0[*i])
            .count();
        // With independent seeds, ~all 128 slots should differ.
        // Require at least 64 (50 %) to differ to keep this a stable
        // probabilistic assertion.
        assert!(
            differing >= NUM_HASHES / 2,
            "only {differing}/128 slots differed between seeds 1 and 2"
        );
    }

    #[test]
    fn signature_is_512_bytes_and_round_trips() {
        let dedup = MinHashDedup::new(7);
        let sig = dedup.signature("hello world, this is a test of minhash");
        let bytes = sig.to_bytes();
        assert_eq!(bytes.len(), SIGNATURE_BYTES);
        assert_eq!(bytes.len(), 512);

        let decoded = MinHashSignature::from_bytes(&bytes);
        assert_eq!(sig, decoded);
    }

    #[test]
    fn identical_text_has_identical_signatures() {
        let dedup = MinHashDedup::new(99);
        let s1 = dedup.signature("hello world");
        let s2 = dedup.signature("hello world");
        assert_eq!(s1, s2);
    }

    #[test]
    fn short_text_is_handled() {
        let dedup = MinHashDedup::new(0);

        // 1-char, 2-char inputs must not panic and must produce a full
        // 512-byte signature.
        let s_a = dedup.signature("a");
        let s_ab = dedup.signature("ab");
        assert_eq!(s_a.to_bytes().len(), 512);
        assert_eq!(s_ab.to_bytes().len(), 512);

        // Different short inputs produce different signatures (the
        // padded shingle differs, so all 128 min values differ).
        assert_ne!(s_a, s_ab);
    }

    #[test]
    fn unicode_text_works() {
        let dedup = MinHashDedup::new(123);

        // Emoji and CJK — multi-byte codepoints. The shingle window
        // is at the char level, so slicing into a codepoint is
        // impossible. No panic is the primary assertion.
        let s_emoji = dedup.signature("🚀 galaxdb rocks 🌌");
        let s_cjk = dedup.signature("銀河データベース、高速検索");
        let s_mixed = dedup.signature("ümlaut naïve café");

        // Round-trip works for unicode inputs.
        assert_eq!(
            s_emoji,
            MinHashSignature::from_bytes(&s_emoji.to_bytes())
        );
        assert_eq!(s_cjk, MinHashSignature::from_bytes(&s_cjk.to_bytes()));
        assert_eq!(
            s_mixed,
            MinHashSignature::from_bytes(&s_mixed.to_bytes())
        );

        // And distinct unicode texts have distinct signatures.
        assert_ne!(s_emoji, s_cjk);
        assert_ne!(s_cjk, s_mixed);
    }

    #[test]
    fn empty_text_produces_sentinel_signature() {
        let dedup = MinHashDedup::new(0);
        let sig = dedup.signature("");
        assert_eq!(sig.0, [u32::MAX; NUM_HASHES]);

        // Round-trip still works.
        assert_eq!(sig, MinHashSignature::from_bytes(&sig.to_bytes()));

        // Two empties compare as identical (Jaccard = 1.0, consistent
        // with "both documents have the empty shingle set").
        let sig2 = dedup.signature("");
        assert_eq!(sig.jaccard_estimate(&sig2), 1.0);
    }

    #[test]
    fn jaccard_estimate_bounds() {
        let dedup = MinHashDedup::new(2024);
        let s = dedup.signature("the quick brown fox jumps over the lazy dog");
        // A signature is always fully self-similar.
        assert_eq!(s.jaccard_estimate(&s), 1.0);

        // Two very different strings should have low Jaccard.
        let a = dedup.signature("the quick brown fox jumps over the lazy dog");
        let b = dedup.signature("completely unrelated content about quantum physics");
        let j = a.jaccard_estimate(&b);
        assert!(
            (0.0..=1.0).contains(&j),
            "jaccard estimate {j} out of [0,1]"
        );
        assert!(j < 0.5, "expected low jaccard for unrelated texts, got {j}");
    }

    // -----------------------------------------------------------------
    // Task 35.3: Jaccard similarity estimator tests
    // -----------------------------------------------------------------

    /// Exact Jaccard over the module's char-trigram shingling strategy.
    ///
    /// Matches the rules documented on `shingle_hashes`:
    /// * `len < 3`  → one shingle: the text right-padded with `'\0'`.
    /// * `len >= 3` → `len - 2` overlapping character windows.
    fn exact_trigram_jaccard(a: &str, b: &str) -> f64 {
        fn shingle_set(s: &str) -> std::collections::HashSet<String> {
            let chars: Vec<char> = s.chars().collect();
            let mut set = std::collections::HashSet::new();

            if chars.is_empty() {
                return set;
            }

            if chars.len() < SHINGLE_WIDTH {
                // Pad to SHINGLE_WIDTH with '\0'.
                let mut padded: Vec<char> = chars.clone();
                while padded.len() < SHINGLE_WIDTH {
                    padded.push('\0');
                }
                set.insert(padded.into_iter().collect::<String>());
                return set;
            }

            for w in chars.windows(SHINGLE_WIDTH) {
                set.insert(w.iter().collect::<String>());
            }
            set
        }

        let a_set = shingle_set(a);
        let b_set = shingle_set(b);

        if a_set.is_empty() && b_set.is_empty() {
            // Two empty shingle sets → Jaccard is conventionally 1.
            return 1.0;
        }

        let inter = a_set.intersection(&b_set).count() as f64;
        let union = a_set.union(&b_set).count() as f64;
        inter / union
    }

    #[test]
    fn jaccard_identity_is_one() {
        let dedup = MinHashDedup::new(42);
        let sig = dedup.signature("arbitrary text for self-comparison");
        assert_eq!(sig.jaccard_estimate(&sig), 1.0);
        assert_eq!(estimate_jaccard(&sig, &sig), 1.0);
    }

    #[test]
    fn jaccard_disjoint_is_near_zero() {
        let dedup = MinHashDedup::new(42);
        let a = dedup.signature("aaaaaaaa");
        let b = dedup.signature("zzzzzzzz");
        let j = a.jaccard_estimate(&b);
        assert!(
            j < 0.05,
            "disjoint trigram sets should yield near-zero Jaccard, got {j}"
        );
    }

    #[test]
    fn jaccard_identical_text_is_one() {
        let dedup = MinHashDedup::new(7);
        let text = "same text hashed twice should collide in every slot";
        let s1 = dedup.signature(text);
        let s2 = dedup.signature(text);
        assert_eq!(s1.jaccard_estimate(&s2), 1.0);
    }

    #[test]
    fn jaccard_near_duplicate_is_high() {
        let dedup = MinHashDedup::new(13);
        let a = dedup.signature("The quick brown fox jumps over the lazy dog.");
        let b = dedup.signature("The quick brown fox jumps over the lazy dog");
        let j = a.jaccard_estimate(&b);
        assert!(
            j > 0.8,
            "near-duplicate texts should have high Jaccard, got {j}"
        );
    }

    #[test]
    fn jaccard_from_bytes_matches_method() {
        let dedup = MinHashDedup::new(2025);
        let texts = [
            "the quick brown fox jumps over the lazy dog",
            "pack my box with five dozen liquor jugs",
            "the five boxing wizards jump quickly",
            "how vexingly quick daft zebras jump",
            "sphinx of black quartz, judge my vow",
        ];
        let sigs: Vec<_> = texts.iter().map(|t| dedup.signature(t)).collect();

        for (i, s1) in sigs.iter().enumerate() {
            for (j, s2) in sigs.iter().enumerate() {
                let via_bytes =
                    jaccard_estimate_from_bytes(&s1.to_bytes(), &s2.to_bytes());
                let via_method = s1.jaccard_estimate(s2);
                assert_eq!(
                    via_bytes, via_method,
                    "bytes path disagrees with method path at ({i}, {j})"
                );
            }
        }
    }

    #[test]
    fn jaccard_symmetric() {
        let dedup = MinHashDedup::new(314);
        let pairs = [
            ("alpha beta gamma", "alpha beta delta"),
            ("short text", "shorter"),
            ("", "non-empty"),
            ("unicode 🚀 rocks", "unicode 🚀 rolls"),
            (
                "a longer sentence used to generate many shingles",
                "another longer sentence sharing several shingles",
            ),
        ];

        for (a, b) in pairs {
            let sa = dedup.signature(a);
            let sb = dedup.signature(b);
            assert_eq!(
                sa.jaccard_estimate(&sb),
                sb.jaccard_estimate(&sa),
                "estimator not symmetric for ({a:?}, {b:?})"
            );
        }
    }

    #[test]
    fn jaccard_convergence_sample() {
        // 10 varied texts → 45 unordered pairs. Assert at least 40 of the
        // 45 MinHash estimates land within 0.15 of the exact trigram
        // Jaccard. Deterministic seed 0xC0FFEE.
        let dedup = MinHashDedup::new(0xC0FFEE);

        let texts = [
            "the quick brown fox jumps over the lazy dog",
            "the quick brown fox jumps over the lazy cat",
            "pack my box with five dozen liquor jugs",
            "pack my crate with five dozen liquor jugs",
            "sphinx of black quartz, judge my vow",
            "how vexingly quick daft zebras jump",
            "the five boxing wizards jump quickly",
            "lorem ipsum dolor sit amet consectetur",
            "lorem ipsum dolor sit amet adipiscing",
            "completely unrelated quantum chromodynamics lecture",
        ];

        let sigs: Vec<_> = texts.iter().map(|t| dedup.signature(t)).collect();

        let mut pairs = 0usize;
        let mut within_tol = 0usize;
        for i in 0..texts.len() {
            for j in (i + 1)..texts.len() {
                pairs += 1;
                let est = sigs[i].jaccard_estimate(&sigs[j]);
                let exact = exact_trigram_jaccard(texts[i], texts[j]);
                if (est - exact).abs() <= 0.15 {
                    within_tol += 1;
                }
            }
        }

        assert_eq!(pairs, 45, "expected 45 unordered pairs");
        assert!(
            within_tol >= 40,
            "only {within_tol}/45 pairs within 0.15 of true Jaccard"
        );
    }
}
