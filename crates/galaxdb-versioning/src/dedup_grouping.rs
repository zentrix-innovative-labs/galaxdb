//! Near-duplicate grouping via MinHash LSH banding + union-find (Req 26, design §9.4, task 35.4).
//!
//! The background refresh job for `_near_duplicate_group` operates on the
//! output of `MinHashDedup` — 512-byte / 128-slot signatures attached to
//! rows. This module implements the pure-data side of that job: given a
//! set of `(row_id, MinHashSignature)` pairs it returns, for each row,
//! either `None` (row is unique, no near-duplicate peer exists) or
//! `Some(group_id)` (row belongs to a near-duplicate cluster).
//!
//! Write-path integration (task 35.2) and the `WHERE NOT DUPLICATE` query
//! operator (task 35.5) live elsewhere; both consume the results produced
//! here.
//!
//! # Algorithm
//!
//! Two signatures with Jaccard similarity ≥ `NEAR_DUPLICATE_JACCARD_THRESHOLD`
//! (0.8 by default, per Req 26) are considered a near-duplicate pair. A
//! naïve O(n²) scan of all pairs is too expensive once n is large, so we
//! use **MinHash LSH banding** to retrieve candidate pairs in sub-quadratic
//! expected time, followed by a **union-find** (disjoint-set) structure
//! to cluster transitive chains of candidates into groups.
//!
//! ## Banding
//!
//! Each 128-slot signature is split into `BANDS = 32` bands of
//! `ROWS_PER_BAND = 4` consecutive slots. Each band is then hashed
//! (xxh3_64 over 16 bytes — 4 × 4 little-endian `u32`) to produce a 64-bit
//! `band_hash`. We populate an LSH bucket map keyed by
//! `(band_index, band_hash)`: all rows whose signatures share the same
//! band hash for the same band index are candidate near-duplicates for
//! that band.
//!
//! Under MinHash, the probability that two signatures agree on all 4
//! slots of a single band equals `J^4` (where `J` is the true Jaccard).
//! The probability that they collide on **at least one** of the 32 bands
//! — i.e. are retrieved as candidates — is
//!
//! ```text
//! 1 - (1 - J^4)^32
//! ```
//!
//! This S-curve has its crossover point (probability 0.5) at
//! `J ≈ (1/32)^(1/4) ≈ 0.42`, so pairs with `J ≥ 0.8` are retrieved with
//! probability ≈ `1 - (1 - 0.4096)^32 ≈ 1 - 6 × 10⁻⁸` — effectively
//! certain — while pairs with `J ≤ 0.4` are retrieved only about 50% of
//! the time and pairs with `J ≤ 0.2` are retrieved only about 4% of the
//! time. False positives returned by LSH are cheap: we verify every
//! candidate pair with the exact estimator [`estimate_jaccard`] before
//! unioning, so low-J candidates are harmless.
//!
//! ## Clustering
//!
//! Candidate pairs that pass the threshold check are fed into a
//! union-find with path compression + union by rank. After processing
//! all buckets, every connected component of size ≥ 2 is a group. The
//! group's `u64` ID is `xxh3_64` applied to the lexicographically
//! smallest `row_id` in the component, which is deterministic and
//! reorder-invariant across runs.
//!
//! Rows in singleton components (no near-duplicate peer found) receive
//! `None` as their assignment.

use std::collections::{BTreeMap, HashMap};

use xxhash_rust::xxh3::xxh3_64;

use crate::minhash::{MinHashSignature, NUM_HASHES, estimate_jaccard};

// ---------------------------------------------------------------------------
// Public constants
// ---------------------------------------------------------------------------

/// Jaccard threshold above which two signatures are considered
/// near-duplicates (Req 26.3, design §9.4). Pairs with an exact MinHash
/// estimate ≥ this value get clustered into the same group.
pub const NEAR_DUPLICATE_JACCARD_THRESHOLD: f64 = 0.8;

/// Number of MinHash LSH bands. Combined with [`ROWS_PER_BAND`] this
/// must equal [`NUM_HASHES`] = 128 so that the entire signature is
/// partitioned. 32 bands × 4 rows gives an LSH S-curve crossover at
/// `J ≈ 0.42`, tuned so that the 0.8 threshold is hit with near-certainty
/// while keeping false-positive bucket traffic modest.
pub const BANDS: usize = 32;

/// Number of consecutive MinHash slots per band. See [`BANDS`].
pub const ROWS_PER_BAND: usize = 4;

// Compile-time invariant: the banding must exactly cover the signature.
const _: () = {
    assert!(
        BANDS * ROWS_PER_BAND == NUM_HASHES,
        "BANDS * ROWS_PER_BAND must equal NUM_HASHES (128)"
    );
};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A row identifier for near-duplicate grouping.
///
/// Opaque bytes — matches the `primary_key` shape used elsewhere in this
/// crate (see `ExportedRow::primary_key`). This module never parses the
/// bytes; it only compares them lexicographically and uses them as keys
/// in hash maps.
pub type DedupRowId = Vec<u8>;

/// Output of [`group_near_duplicates`].
///
/// Holds both the per-row assignment (row_id → Option<group_id> in
/// input order) and the inverse map (group_id → sorted row_ids, only
/// groups of size ≥ 2). The dual representation lets callers walk
/// rows in their original ingest order while also iterating groups
/// directly when, e.g., emitting diagnostic dumps.
#[derive(Debug, Clone)]
pub struct NearDuplicateGrouping {
    assignments: Vec<(DedupRowId, Option<u64>)>,
    groups: BTreeMap<u64, Vec<DedupRowId>>,
}

impl NearDuplicateGrouping {
    /// Per-row assignment in the same order as the input slice to
    /// [`group_near_duplicates`]. Rows with no near-duplicate peer
    /// receive `None`.
    #[inline]
    pub fn assignments(&self) -> &[(DedupRowId, Option<u64>)] {
        &self.assignments
    }

    /// All near-duplicate groups. Keyed by group ID, valued by the
    /// group's row IDs in lexicographic order. Only groups of size
    /// ≥ 2 appear here.
    #[inline]
    pub fn groups(&self) -> &BTreeMap<u64, Vec<DedupRowId>> {
        &self.groups
    }

    /// Look up the group ID assigned to a given row. Returns `None`
    /// if the row isn't in the grouping or if the row is unique.
    ///
    /// Linear scan over `assignments`. Callers processing large
    /// batches should build their own index from [`Self::assignments`].
    pub fn group_of(&self, row_id: &[u8]) -> Option<u64> {
        self.assignments
            .iter()
            .find(|(rid, _)| rid.as_slice() == row_id)
            .and_then(|(_, g)| *g)
    }

    /// Total number of rows assigned to some group. Equals
    /// `sum(group_size)` over all groups.
    pub fn grouped_row_count(&self) -> usize {
        self.groups.values().map(|v| v.len()).sum()
    }

    /// Number of distinct near-duplicate groups (each of size ≥ 2).
    #[inline]
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Group rows by near-duplicate similarity using MinHash LSH banding
/// plus union-find.
///
/// `threshold` is compared against the exact MinHash estimator
/// ([`estimate_jaccard`]) for every LSH-retrieved candidate pair. The
/// canonical production threshold is [`NEAR_DUPLICATE_JACCARD_THRESHOLD`].
///
/// The returned [`NearDuplicateGrouping`] is deterministic: identical
/// input pairs (same row_ids, same signatures, same threshold) always
/// produce identical assignments and groups, regardless of input order.
/// Group IDs are derived via `xxh3_64` of the smallest row_id in the
/// component, so different runs with the same data converge on the
/// same u64 IDs.
///
/// Complexity is `O(n · BANDS + |pairs| · α(n))` where `|pairs|` is
/// the number of LSH-retrieved candidate pairs. For well-spread inputs
/// this is sub-quadratic; for adversarial inputs where every row shares
/// a band with every other it degrades to `O(n²)`.
pub fn group_near_duplicates(
    rows: &[(DedupRowId, MinHashSignature)],
    threshold: f64,
) -> NearDuplicateGrouping {
    let n = rows.len();
    if n == 0 {
        return NearDuplicateGrouping {
            assignments: Vec::new(),
            groups: BTreeMap::new(),
        };
    }

    // --- Step 1: Populate LSH buckets ---------------------------------
    //
    // For each signature, compute 32 band hashes and append the row
    // index to each bucket it lands in. Keying on (band_idx, band_hash)
    // rather than band_hash alone prevents cross-band collisions from
    // merging rows that only happen to share a hash value in different
    // bands.
    let mut buckets: HashMap<(usize, u64), Vec<usize>> = HashMap::new();
    for (i, (_, sig)) in rows.iter().enumerate() {
        let bands = band_hashes(sig);
        for (band_idx, band_hash) in bands.iter().enumerate() {
            buckets
                .entry((band_idx, *band_hash))
                .or_default()
                .push(i);
        }
    }

    // --- Step 2: Verify candidate pairs and union ---------------------
    //
    // For each bucket of size ≥ 2, consider every pair. Skip pairs
    // already in the same union-find component (they've been verified
    // via another band already). Verify with the exact MinHash
    // estimator — LSH is a filter, not a final answer.
    let mut uf = UnionFind::new(n);
    for bucket in buckets.values() {
        if bucket.len() < 2 {
            continue;
        }
        for a_idx in 0..bucket.len() {
            let a = bucket[a_idx];
            for &b in &bucket[a_idx + 1..] {
                if uf.find(a) == uf.find(b) {
                    // Same component via an earlier pair.
                    continue;
                }
                let j = estimate_jaccard(&rows[a].1, &rows[b].1);
                if j >= threshold {
                    uf.union(a, b);
                }
            }
        }
    }

    // --- Step 3: Collect components -----------------------------------
    let mut components: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let root = uf.find(i);
        components.entry(root).or_default().push(i);
    }

    // --- Step 4: Assign group IDs to components of size ≥ 2 -----------
    let mut row_to_group: Vec<Option<u64>> = vec![None; n];
    let mut groups: BTreeMap<u64, Vec<DedupRowId>> = BTreeMap::new();
    for members in components.into_values() {
        if members.len() < 2 {
            continue;
        }

        // Sort member row_ids lexicographically — this determines both
        // the representative (smallest) and the stored group ordering.
        let mut sorted_ids: Vec<DedupRowId> =
            members.iter().map(|&i| rows[i].0.clone()).collect();
        sorted_ids.sort();

        let group_id = xxh3_64(&sorted_ids[0]);
        for &i in &members {
            row_to_group[i] = Some(group_id);
        }
        groups.insert(group_id, sorted_ids);
    }

    // --- Step 5: Assemble input-order assignments ---------------------
    let assignments: Vec<(DedupRowId, Option<u64>)> = rows
        .iter()
        .enumerate()
        .map(|(i, (id, _))| (id.clone(), row_to_group[i]))
        .collect();

    NearDuplicateGrouping {
        assignments,
        groups,
    }
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// Compute the 32 band hashes for a signature.
///
/// Each band is 4 consecutive `u32` slots (16 bytes little-endian)
/// fed to `xxh3_64`.
fn band_hashes(sig: &MinHashSignature) -> [u64; BANDS] {
    let slots = sig.slots();
    let mut result = [0u64; BANDS];
    for (band_idx, out) in result.iter_mut().enumerate() {
        let start = band_idx * ROWS_PER_BAND;
        let mut buf = [0u8; ROWS_PER_BAND * 4];
        for row in 0..ROWS_PER_BAND {
            let off = row * 4;
            buf[off..off + 4].copy_from_slice(&slots[start + row].to_le_bytes());
        }
        *out = xxh3_64(&buf);
    }
    result
}

/// Disjoint-set data structure with path compression and union by rank.
/// Amortised `O(α(n))` per operation — effectively constant for any
/// practical `n`.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    /// Find the root of `x`, compressing the path.
    fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        // Path compression: rewire every node on the x→root chain to
        // point directly at the root.
        let mut cur = x;
        while self.parent[cur] != root {
            let next = self.parent[cur];
            self.parent[cur] = root;
            cur = next;
        }
        root
    }

    /// Union the components containing `a` and `b`. No-op if they're
    /// already in the same component.
    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => self.parent[ra] = rb,
            std::cmp::Ordering::Greater => self.parent[rb] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] = self.rank[ra].saturating_add(1);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minhash::MinHashDedup;

    fn id(s: &str) -> DedupRowId {
        s.as_bytes().to_vec()
    }

    #[test]
    fn no_rows_produces_empty_grouping() {
        let rows: Vec<(DedupRowId, MinHashSignature)> = Vec::new();
        let grouping = group_near_duplicates(&rows, NEAR_DUPLICATE_JACCARD_THRESHOLD);
        assert!(grouping.assignments().is_empty());
        assert!(grouping.groups().is_empty());
        assert_eq!(grouping.group_count(), 0);
        assert_eq!(grouping.grouped_row_count(), 0);
    }

    #[test]
    fn unique_rows_produce_no_groups() {
        let dedup = MinHashDedup::new(42);
        // Five texts with no lexical overlap — pairwise Jaccard is
        // effectively zero, so nothing should group.
        let texts = [
            "Photosynthesis converts sunlight into energy in plants",
            "The Himalayan range is home to many endemic species",
            "Quantum chromodynamics describes strong nuclear forces",
            "Renaissance painters used linear perspective techniques",
            "The Fibonacci sequence appears in natural spirals",
        ];
        let rows: Vec<_> = texts
            .iter()
            .enumerate()
            .map(|(i, t)| (id(&format!("row-{i}")), dedup.signature(t)))
            .collect();

        let grouping = group_near_duplicates(&rows, NEAR_DUPLICATE_JACCARD_THRESHOLD);

        assert_eq!(grouping.group_count(), 0);
        assert_eq!(grouping.grouped_row_count(), 0);
        for (_, g) in grouping.assignments() {
            assert!(g.is_none(), "expected all rows to be unique, got {g:?}");
        }
    }

    #[test]
    fn identical_duplicates_are_grouped() {
        let dedup = MinHashDedup::new(42);
        let text = "The quick brown fox jumps over the lazy dog";
        let rows = vec![
            (id("a"), dedup.signature(text)),
            (id("b"), dedup.signature(text)),
            (id("c"), dedup.signature(text)),
        ];

        let grouping = group_near_duplicates(&rows, NEAR_DUPLICATE_JACCARD_THRESHOLD);

        assert_eq!(grouping.group_count(), 1);
        assert_eq!(grouping.grouped_row_count(), 3);

        // All three assignments point at the same group ID.
        let gid = grouping.assignments()[0].1.expect("a should be grouped");
        for (_, g) in grouping.assignments() {
            assert_eq!(*g, Some(gid));
        }

        // Group contents are all three row_ids, lexicographically sorted.
        let stored = grouping.groups().get(&gid).expect("group should exist");
        assert_eq!(stored, &vec![id("a"), id("b"), id("c")]);
    }

    #[test]
    fn near_duplicates_are_grouped() {
        let dedup = MinHashDedup::new(42);
        // Three near-identical phrasings of the pangram. Pairwise
        // Jaccard should exceed 0.8 for all three edges.
        let texts = [
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog.",
            "The quick brown fox jumps over the lazy dog!",
        ];
        let sigs: Vec<_> = texts.iter().map(|t| dedup.signature(t)).collect();

        // Sanity: all pairwise Jaccard estimates really are ≥ threshold.
        for i in 0..sigs.len() {
            for j in (i + 1)..sigs.len() {
                let jaccard = estimate_jaccard(&sigs[i], &sigs[j]);
                assert!(
                    jaccard >= NEAR_DUPLICATE_JACCARD_THRESHOLD,
                    "setup: pair ({i}, {j}) Jaccard {jaccard} below threshold"
                );
            }
        }

        let rows = vec![
            (id("a"), sigs[0]),
            (id("b"), sigs[1]),
            (id("c"), sigs[2]),
        ];
        let grouping = group_near_duplicates(&rows, NEAR_DUPLICATE_JACCARD_THRESHOLD);

        assert_eq!(grouping.group_count(), 1);
        assert_eq!(grouping.grouped_row_count(), 3);

        let gid = grouping.group_of(b"a").expect("a should be grouped");
        assert_eq!(grouping.group_of(b"b"), Some(gid));
        assert_eq!(grouping.group_of(b"c"), Some(gid));
    }

    #[test]
    fn low_jaccard_pairs_are_not_grouped() {
        let dedup = MinHashDedup::new(42);
        let s_a = dedup.signature("Photosynthesis converts sunlight into cellular energy");
        let s_b = dedup.signature("The Fibonacci sequence appears in natural spirals");

        // Sanity: these two have low Jaccard.
        let j = estimate_jaccard(&s_a, &s_b);
        assert!(
            j < NEAR_DUPLICATE_JACCARD_THRESHOLD,
            "setup: expected low Jaccard, got {j}"
        );

        let rows = vec![(id("a"), s_a), (id("b"), s_b)];
        let grouping = group_near_duplicates(&rows, NEAR_DUPLICATE_JACCARD_THRESHOLD);

        assert_eq!(grouping.group_count(), 0);
        assert_eq!(grouping.group_of(b"a"), None);
        assert_eq!(grouping.group_of(b"b"), None);
    }

    #[test]
    fn group_ids_are_stable_across_reordering() {
        let dedup = MinHashDedup::new(42);
        let texts = [
            ("x1", "The quick brown fox jumps over the lazy dog"),
            ("x2", "The quick brown fox jumps over the lazy dog."),
            ("y1", "Photosynthesis converts sunlight into cellular energy"),
            ("y2", "Photosynthesis converts sunlight into cellular energy!"),
            ("z1", "The Fibonacci sequence appears in natural spirals"),
        ];

        let mut rows1: Vec<_> = texts
            .iter()
            .map(|(k, t)| (id(k), dedup.signature(t)))
            .collect();

        // Second input: same data, reversed order.
        let mut rows2 = rows1.clone();
        rows2.reverse();

        // And a third: a rotation of the input, to be thorough.
        let mut rows3 = rows1.clone();
        rows3.rotate_left(2);

        let g1 = group_near_duplicates(&rows1, NEAR_DUPLICATE_JACCARD_THRESHOLD);
        let g2 = group_near_duplicates(&rows2, NEAR_DUPLICATE_JACCARD_THRESHOLD);
        let g3 = group_near_duplicates(&rows3, NEAR_DUPLICATE_JACCARD_THRESHOLD);

        // The `groups()` map is input-order invariant.
        assert_eq!(g1.groups(), g2.groups());
        assert_eq!(g1.groups(), g3.groups());

        // But the `assignments()` order follows the input order.
        let order1: Vec<_> = g1.assignments().iter().map(|(k, _)| k.clone()).collect();
        let order2: Vec<_> = g2.assignments().iter().map(|(k, _)| k.clone()).collect();
        rows1.reverse(); // becomes rows2 ordering
        let expected_order2: Vec<_> = rows1.iter().map(|(k, _)| k.clone()).collect();
        assert_ne!(order1, order2, "reverse input should produce reverse order");
        assert_eq!(order2, expected_order2);
    }

    #[test]
    fn banding_catches_high_jaccard_pairs() {
        let dedup = MinHashDedup::new(42);
        let mut rows = Vec::with_capacity(50);

        // 5 near-duplicates: common base, vary only terminal punctuation.
        let base = "The quick brown fox jumps over the lazy dog in the evening light";
        let punctuations = [".", "!", "?", ";", ","];
        for (i, p) in punctuations.iter().enumerate() {
            let text = format!("{base}{p}");
            let rid = id(&format!("dup-{i}"));
            rows.push((rid, dedup.signature(&text)));
        }

        // 45 unique rows: 30-character alphabetic strings seeded from `i`.
        // Constructed with a SplitMix64-style recurrence so trigram
        // overlap between any two is minimal.
        for i in 0..45usize {
            let mut s = String::with_capacity(30);
            let mut state = (i as u64)
                .wrapping_mul(0x9E3779B97F4A7C15)
                ^ 0xDEADBEEF_CAFEBABEu64;
            for _ in 0..30 {
                state = state.wrapping_mul(0xD6E8FEB86659FD93);
                state = state.wrapping_add(0x1234_5678_90AB_CDEF);
                let byte = (state >> 16) as u8 % 26;
                s.push((b'a' + byte) as char);
            }
            let rid = id(&format!("uniq-{i}"));
            rows.push((rid, dedup.signature(&s)));
        }

        let grouping = group_near_duplicates(&rows, NEAR_DUPLICATE_JACCARD_THRESHOLD);

        // Exactly one group, containing all 5 near-dupes.
        assert_eq!(grouping.group_count(), 1);
        assert_eq!(grouping.grouped_row_count(), 5);

        let (gid, members) = grouping.groups().iter().next().expect("one group");
        assert_eq!(members.len(), 5);
        for i in 0..5 {
            assert_eq!(grouping.group_of(format!("dup-{i}").as_bytes()), Some(*gid));
        }

        // All 45 unique rows must be ungrouped.
        for i in 0..45 {
            let rid = format!("uniq-{i}");
            assert_eq!(
                grouping.group_of(rid.as_bytes()),
                None,
                "row {rid} should not be grouped"
            );
        }
    }

    #[test]
    fn transitive_near_duplicates_cluster_together() {
        // A chain A — B — C where J(A,B) ≥ threshold, J(B,C) ≥ threshold,
        // but J(A,C) is weaker because both perturbations compound.
        // Union-find must still cluster all three into one group.
        //
        // Construction: a ~130-character base, then single-word
        // substitutions at two different positions — each substitution
        // affects only ~10 of ~130 trigrams so pairwise J stays well
        // above 0.8, while the compound change (A vs C) removes roughly
        // twice as many shingles and pulls J(A,C) closer to the
        // threshold.
        let dedup = MinHashDedup::new(42);
        let text_a = "The quick brown fox jumped cleverly over the lazy sleeping dog in the large garden near the old wooden fence at sunrise yesterday.";
        let text_b = "The quick brown fox jumped swiftly over the lazy sleeping dog in the large garden near the old wooden fence at sunrise yesterday.";
        let text_c = "The quick brown fox jumped swiftly over the lazy sleeping dog in the large meadow near the old wooden fence at sunrise yesterday.";

        let sig_a = dedup.signature(text_a);
        let sig_b = dedup.signature(text_b);
        let sig_c = dedup.signature(text_c);

        let jab = estimate_jaccard(&sig_a, &sig_b);
        let jbc = estimate_jaccard(&sig_b, &sig_c);

        assert!(
            jab >= NEAR_DUPLICATE_JACCARD_THRESHOLD,
            "setup: J(A,B) = {jab} should be ≥ threshold"
        );
        assert!(
            jbc >= NEAR_DUPLICATE_JACCARD_THRESHOLD,
            "setup: J(B,C) = {jbc} should be ≥ threshold"
        );

        let rows = vec![
            (id("a"), sig_a),
            (id("b"), sig_b),
            (id("c"), sig_c),
        ];
        let grouping = group_near_duplicates(&rows, NEAR_DUPLICATE_JACCARD_THRESHOLD);

        // Transitivity: a single group of 3, regardless of whether the
        // direct A-C edge clears threshold.
        assert_eq!(
            grouping.group_count(),
            1,
            "expected one transitive group, got {}",
            grouping.group_count()
        );
        assert_eq!(grouping.grouped_row_count(), 3);

        let gid = grouping.group_of(b"a").expect("a grouped");
        assert_eq!(grouping.group_of(b"b"), Some(gid));
        assert_eq!(grouping.group_of(b"c"), Some(gid));
    }

    #[test]
    fn threshold_respected() {
        let dedup = MinHashDedup::new(42);
        let texts = [
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog.",
            "The quick brown fox jumps over the lazy dog!",
        ];
        let rows: Vec<_> = texts
            .iter()
            .enumerate()
            .map(|(i, t)| (id(&format!("r{i}")), dedup.signature(t)))
            .collect();

        // Very high threshold: MinHash estimates will fall short of
        // 0.99 due to the ±~0.04 estimator noise, so most pairs aren't
        // confirmed and at least one row stays ungrouped.
        let strict = group_near_duplicates(&rows, 0.99);
        assert!(
            strict.grouped_row_count() < 3,
            "with threshold 0.99, not all 3 rows should be grouped; got {} grouped",
            strict.grouped_row_count()
        );

        // Low threshold: aggressive clustering. All three should fall
        // into a single group.
        let loose = group_near_duplicates(&rows, 0.5);
        assert_eq!(loose.group_count(), 1);
        assert_eq!(loose.grouped_row_count(), 3);
    }
}
