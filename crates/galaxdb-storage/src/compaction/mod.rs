//! Lazy Leveling compaction with MVCC garbage collection for GalaxDB.
//!
//! ## LSM Structure (Req 6)
//!
//! - **L0**: Flushed memtables (tiered, up to 4 files before compaction trigger)
//! - **L1–L3**: Tiered compaction (multiple sorted runs per level)
//! - **L4** (bottom): Leveled compaction (single sorted run)
//!
//! ## MVCC GC During Compaction
//!
//! For each key encountered during merge, the compactor checks:
//! 1. Is this version needed by the oldest active snapshot? → keep
//! 2. Is this version referenced by any pinned `VersionTag`? → keep
//! 3. Otherwise → discard
//!
//! ## SST Size
//!
//! 64 MB initially (Month 1), configurable down to 8 MB in Month 4 hardening (Req 36).

#[cfg(test)]
mod tests;

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use galaxdb_common::Timestamp;

use crate::bloom::{MonkeyAllocator, SstBloomFilter};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of LSM levels (L0 through L4).
pub const NUM_LEVELS: usize = 5;

/// Maximum number of SST files in L0 before compaction is triggered.
pub const L0_FILE_COUNT_THRESHOLD: usize = 4;

/// Default size ratio between adjacent levels for tiered compaction.
pub const DEFAULT_SIZE_RATIO: u32 = 10;

/// Default SST file size in bytes (64 MB for Month 1).
pub const DEFAULT_SST_SIZE_BYTES: u64 = 64 * 1024 * 1024;

/// Minimum SST file size in bytes (8 MB, configurable in Month 4).
pub const MIN_SST_SIZE_BYTES: u64 = 8 * 1024 * 1024;

/// The bottom level index (leveled compaction).
pub const BOTTOM_LEVEL: usize = 4;

// ---------------------------------------------------------------------------
// SstMetadata
// ---------------------------------------------------------------------------

/// Metadata for a single SST file in the LSM tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SstMetadata {
    /// Unique SST file identifier.
    pub sst_id: u64,
    /// The LSM level this SST belongs to.
    pub level: usize,
    /// Smallest key in this SST.
    pub min_key: Vec<u8>,
    /// Largest key in this SST.
    pub max_key: Vec<u8>,
    /// Size of the SST file in bytes.
    pub size_bytes: u64,
    /// Number of rows in this SST.
    pub row_count: u64,
}

impl SstMetadata {
    /// Returns true if this SST's key range overlaps with the given range.
    pub fn overlaps(&self, min: &[u8], max: &[u8]) -> bool {
        self.min_key <= max.to_vec() && self.max_key >= min.to_vec()
    }
}

// ---------------------------------------------------------------------------
// LsmLevel
// ---------------------------------------------------------------------------

/// Represents a single LSM level with its SST file metadata.
#[derive(Debug, Clone)]
pub struct LsmLevel {
    /// The level index (0–4).
    pub level: usize,
    /// SST files at this level, sorted by min_key.
    pub ssts: Vec<SstMetadata>,
}

impl LsmLevel {
    /// Creates a new empty level.
    pub fn new(level: usize) -> Self {
        Self {
            level,
            ssts: Vec::new(),
        }
    }

    /// Returns the total size of all SSTs at this level in bytes.
    pub fn total_size_bytes(&self) -> u64 {
        self.ssts.iter().map(|s| s.size_bytes).sum()
    }

    /// Returns the number of SST files at this level.
    pub fn file_count(&self) -> usize {
        self.ssts.len()
    }

    /// Adds an SST to this level, maintaining sorted order by min_key.
    pub fn add_sst(&mut self, sst: SstMetadata) {
        let pos = self
            .ssts
            .binary_search_by(|probe| probe.min_key.cmp(&sst.min_key))
            .unwrap_or_else(|e| e);
        self.ssts.insert(pos, sst);
    }

    /// Removes an SST by its ID.
    pub fn remove_sst(&mut self, sst_id: u64) -> Option<SstMetadata> {
        if let Some(pos) = self.ssts.iter().position(|s| s.sst_id == sst_id) {
            Some(self.ssts.remove(pos))
        } else {
            None
        }
    }

    /// Returns true if this is the bottom level (leveled compaction).
    pub fn is_bottom(&self) -> bool {
        self.level == BOTTOM_LEVEL
    }

    /// Returns true if this is L0 (tiered, file-count triggered).
    pub fn is_l0(&self) -> bool {
        self.level == 0
    }

    /// Returns SSTs whose key ranges overlap with the given range.
    pub fn overlapping_ssts(&self, min_key: &[u8], max_key: &[u8]) -> Vec<&SstMetadata> {
        self.ssts
            .iter()
            .filter(|s| s.overlaps(min_key, max_key))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// LsmTree
// ---------------------------------------------------------------------------

/// Manages all LSM levels and tracks SST files per level.
#[derive(Debug, Clone)]
pub struct LsmTree {
    /// The five LSM levels (L0–L4).
    pub levels: Vec<LsmLevel>,
    /// Size ratio between adjacent levels (default 10).
    pub size_ratio: u32,
}

impl LsmTree {
    /// Creates a new empty LSM tree with the default size ratio.
    pub fn new() -> Self {
        Self::with_size_ratio(DEFAULT_SIZE_RATIO)
    }

    /// Creates a new empty LSM tree with a custom size ratio.
    pub fn with_size_ratio(size_ratio: u32) -> Self {
        let levels = (0..NUM_LEVELS).map(LsmLevel::new).collect();
        Self { levels, size_ratio }
    }

    /// Returns a reference to the specified level.
    pub fn level(&self, idx: usize) -> &LsmLevel {
        &self.levels[idx]
    }

    /// Returns a mutable reference to the specified level.
    pub fn level_mut(&mut self, idx: usize) -> &mut LsmLevel {
        &mut self.levels[idx]
    }

    /// Adds an SST to the specified level.
    pub fn add_sst(&mut self, level: usize, sst: SstMetadata) {
        self.levels[level].add_sst(sst);
    }

    /// Removes an SST from the specified level by ID.
    pub fn remove_sst(&mut self, level: usize, sst_id: u64) -> Option<SstMetadata> {
        self.levels[level].remove_sst(sst_id)
    }

    /// Returns the total number of SST files across all levels.
    pub fn total_sst_count(&self) -> usize {
        self.levels.iter().map(|l| l.file_count()).sum()
    }
}

impl Default for LsmTree {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CompactionTrigger
// ---------------------------------------------------------------------------

/// Determines when compaction should be triggered.
#[derive(Debug, Clone)]
pub struct CompactionTrigger {
    /// Maximum number of SST files in L0 before triggering compaction.
    pub l0_file_count_threshold: usize,
    /// Size ratio threshold for level-based compaction triggers.
    pub size_ratio: u32,
}

impl CompactionTrigger {
    /// Creates a new trigger with default thresholds.
    pub fn new() -> Self {
        Self {
            l0_file_count_threshold: L0_FILE_COUNT_THRESHOLD,
            size_ratio: DEFAULT_SIZE_RATIO,
        }
    }

    /// Creates a trigger with custom thresholds.
    pub fn with_thresholds(l0_file_count_threshold: usize, size_ratio: u32) -> Self {
        Self {
            l0_file_count_threshold,
            size_ratio,
        }
    }

    /// Checks if compaction should be triggered for any level.
    ///
    /// Returns `Some(level)` for the first level that needs compaction,
    /// or `None` if no compaction is needed.
    pub fn check(&self, tree: &LsmTree) -> Option<usize> {
        // Check L0 file count threshold first (highest priority).
        if tree.level(0).file_count() >= self.l0_file_count_threshold {
            return Some(0);
        }

        // Check size ratio for L1–L3 (tiered levels).
        // A level needs compaction when its total size exceeds
        // size_ratio × the next level's target size.
        for level_idx in 1..BOTTOM_LEVEL {
            let current_size = tree.level(level_idx).total_size_bytes();
            let next_level_size = tree.level(level_idx + 1).total_size_bytes();

            // Target size for the next level is size_ratio × current level's size.
            // Trigger compaction when current level is "full" relative to the ratio.
            // A simple heuristic: compact when current_size > next_level_size / size_ratio
            // (i.e., the current level has accumulated enough data).
            if current_size > 0 {
                let target = if next_level_size > 0 {
                    next_level_size / self.size_ratio as u64
                } else {
                    // If the next level is empty, use a base target.
                    // For tiered levels, trigger when we have enough sorted runs.
                    DEFAULT_SST_SIZE_BYTES
                };

                if current_size >= target {
                    return Some(level_idx);
                }
            }
        }

        None
    }

    /// Checks if a specific level needs compaction.
    pub fn needs_compaction(&self, tree: &LsmTree, level: usize) -> bool {
        if level == 0 {
            return tree.level(0).file_count() >= self.l0_file_count_threshold;
        }

        if level >= BOTTOM_LEVEL {
            return false; // Bottom level doesn't trigger compaction to a lower level.
        }

        let current_size = tree.level(level).total_size_bytes();
        let next_level_size = tree.level(level + 1).total_size_bytes();

        if current_size == 0 {
            return false;
        }

        let target = if next_level_size > 0 {
            next_level_size / self.size_ratio as u64
        } else {
            DEFAULT_SST_SIZE_BYTES
        };

        current_size >= target
    }
}

impl Default for CompactionTrigger {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// VersionedEntry — a key-value pair with MVCC timestamp
// ---------------------------------------------------------------------------

/// A single versioned key-value entry used during merge iteration.
///
/// Each entry represents one version of a key at a specific timestamp.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VersionedEntry {
    /// The primary key.
    pub key: Vec<u8>,
    /// The MVCC commit timestamp.
    pub timestamp: Timestamp,
    /// The value bytes, or `None` for a tombstone.
    pub value: Option<Vec<u8>>,
}

// ---------------------------------------------------------------------------
// GcContext — MVCC garbage collection context
// ---------------------------------------------------------------------------

/// Context for MVCC garbage collection during compaction.
///
/// Holds the oldest active snapshot timestamp and a set of pinned tag
/// timestamps. Versions are retained if they are needed by any active
/// snapshot or referenced by any pinned tag.
#[derive(Debug, Clone)]
pub struct GcContext {
    /// The oldest active snapshot timestamp. Versions with timestamps
    /// >= this value must be retained (they may be visible to active readers).
    pub oldest_active_snapshot: Option<Timestamp>,
    /// Set of timestamps pinned by version tags. All versions at these
    /// timestamps must be retained regardless of age.
    pub pinned_tag_timestamps: HashSet<Timestamp>,
}

impl GcContext {
    /// Creates a new GC context with no active snapshots or pinned tags.
    pub fn new() -> Self {
        Self {
            oldest_active_snapshot: None,
            pinned_tag_timestamps: HashSet::new(),
        }
    }

    /// Creates a GC context with the given oldest snapshot and pinned tags.
    pub fn with_context(
        oldest_active_snapshot: Option<Timestamp>,
        pinned_tag_timestamps: HashSet<Timestamp>,
    ) -> Self {
        Self {
            oldest_active_snapshot,
            pinned_tag_timestamps,
        }
    }

    /// Build a GC context that pins every timestamp currently
    /// referenced by a version tag (task 33.5 / 10.5). The tag
    /// catalog records each tag's `version_timestamp`; compactor
    /// callers pass the full set here so MVCC garbage collection
    /// retains any row version that tagged snapshots depend on.
    ///
    /// `pinned_timestamps` is typically produced by iterating a
    /// `galaxdb_versioning::TagCatalog::list_tags()` result and
    /// collecting each tag's `version_timestamp`. Passed as a plain
    /// slice so the `galaxdb-storage` crate does not take a
    /// dependency on `galaxdb-versioning` (cycle avoidance).
    ///
    /// `oldest_active_snapshot` comes from the MVCC transaction
    /// manager. When `None`, only pinned versions are retained;
    /// otherwise versions `>= oldest_active_snapshot` are also
    /// retained for in-flight readers.
    pub fn with_pins(
        oldest_active_snapshot: Option<Timestamp>,
        pinned_timestamps: impl IntoIterator<Item = Timestamp>,
    ) -> Self {
        Self {
            oldest_active_snapshot,
            pinned_tag_timestamps: pinned_timestamps.into_iter().collect(),
        }
    }

    /// Determines whether a specific version should be kept.
    ///
    /// A version is kept if:
    /// 1. It is the latest version for its key (always keep at least one).
    /// 2. Its timestamp >= oldest active snapshot (needed by active readers).
    /// 3. Its timestamp is in the pinned tag set.
    ///
    /// The `is_latest` flag indicates whether this is the most recent version
    /// for the key.
    pub fn should_keep(&self, timestamp: Timestamp, is_latest: bool) -> bool {
        // Always keep the latest version for each key.
        if is_latest {
            return true;
        }

        // Keep if needed by the oldest active snapshot.
        if let Some(oldest) = self.oldest_active_snapshot {
            if timestamp >= oldest {
                return true;
            }
        }

        // Keep if referenced by a pinned version tag.
        if self.pinned_tag_timestamps.contains(&timestamp) {
            return true;
        }

        false
    }
}

impl Default for GcContext {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// MvccGarbageCollector
// ---------------------------------------------------------------------------

/// Applies MVCC garbage collection to a stream of versioned entries.
///
/// Given a sorted sequence of `(key, timestamp, value)` entries (sorted by
/// key ascending, then timestamp descending), the collector filters out
/// versions that are no longer needed.
pub struct MvccGarbageCollector {
    gc_context: GcContext,
}

impl MvccGarbageCollector {
    /// Creates a new MVCC garbage collector with the given context.
    pub fn new(gc_context: GcContext) -> Self {
        Self { gc_context }
    }

    /// Applies GC to a sorted list of versioned entries.
    ///
    /// Input must be sorted by (key ASC, timestamp DESC).
    /// Returns the filtered list with obsolete versions removed.
    pub fn apply(&self, entries: &[VersionedEntry]) -> Vec<VersionedEntry> {
        if entries.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(entries.len());
        let mut current_key: Option<&[u8]> = None;
        let mut is_first_for_key = true;

        for entry in entries {
            // Detect key boundary.
            let new_key = match current_key {
                Some(k) => k != entry.key.as_slice(),
                None => true,
            };

            if new_key {
                current_key = Some(&entry.key);
                is_first_for_key = true;
            }

            let keep = self
                .gc_context
                .should_keep(entry.timestamp, is_first_for_key);

            if keep {
                result.push(entry.clone());
            }

            is_first_for_key = false;
        }

        result
    }
}

// ---------------------------------------------------------------------------
// MergeIterator
// ---------------------------------------------------------------------------

/// An entry in the merge heap, tracking which source run it came from.
#[derive(Debug, Clone, Eq, PartialEq)]
struct HeapEntry {
    /// The versioned entry.
    entry: VersionedEntry,
    /// Index of the source sorted run.
    run_index: usize,
    /// Position within the source run (for advancing).
    position: usize,
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: smallest key first, then highest timestamp first (descending).
        // BinaryHeap is a max-heap, so we reverse the comparison.
        match other.entry.key.cmp(&self.entry.key) {
            Ordering::Equal => {
                // Same key: higher timestamp should come first (descending).
                self.entry.timestamp.cmp(&other.entry.timestamp)
            }
            ord => ord,
        }
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Merges multiple sorted runs into a single sorted stream.
///
/// Each sorted run is a `Vec<VersionedEntry>` sorted by (key ASC, timestamp DESC).
/// The merge iterator produces entries in the same order, suitable for
/// MVCC GC and output to new SST files.
pub struct MergeIterator {
    /// The source sorted runs.
    runs: Vec<Vec<VersionedEntry>>,
    /// Min-heap for k-way merge.
    heap: BinaryHeap<HeapEntry>,
}

impl MergeIterator {
    /// Creates a new merge iterator over the given sorted runs.
    ///
    /// Each run must be sorted by (key ASC, timestamp DESC).
    pub fn new(runs: Vec<Vec<VersionedEntry>>) -> Self {
        let mut heap = BinaryHeap::new();

        // Seed the heap with the first entry from each run.
        for (run_index, run) in runs.iter().enumerate() {
            if let Some(entry) = run.first() {
                heap.push(HeapEntry {
                    entry: entry.clone(),
                    run_index,
                    position: 0,
                });
            }
        }

        Self { runs, heap }
    }

    /// Collects all remaining entries into a vector.
    pub fn collect_all(&mut self) -> Vec<VersionedEntry> {
        let mut result = Vec::new();
        while let Some(entry) = self.next_entry() {
            result.push(entry);
        }
        result
    }

    /// Merges all runs and applies MVCC GC, returning the filtered result.
    pub fn merge_with_gc(mut self, gc: &MvccGarbageCollector) -> Vec<VersionedEntry> {
        let merged = self.collect_all();
        gc.apply(&merged)
    }

    /// Returns the next entry in sorted order, or `None` if exhausted.
    fn next_entry(&mut self) -> Option<VersionedEntry> {
        let heap_entry = self.heap.pop()?;

        // Advance the source run.
        let next_pos = heap_entry.position + 1;
        if next_pos < self.runs[heap_entry.run_index].len() {
            let next_entry = self.runs[heap_entry.run_index][next_pos].clone();
            self.heap.push(HeapEntry {
                entry: next_entry,
                run_index: heap_entry.run_index,
                position: next_pos,
            });
        }

        Some(heap_entry.entry)
    }
}

// ---------------------------------------------------------------------------
// CompactionConfig
// ---------------------------------------------------------------------------

/// Configuration for the compaction process.
#[derive(Debug, Clone)]
pub struct CompactionConfig {
    /// Target SST file size in bytes (default 64 MB).
    pub sst_size_bytes: u64,
    /// Bits per key for Bloom filter construction.
    pub bloom_bits_per_key: u32,
    /// LSM size ratio for Monkey allocation.
    pub size_ratio: u32,
}

impl CompactionConfig {
    /// Creates a new compaction config with defaults.
    pub fn new() -> Self {
        Self {
            sst_size_bytes: DEFAULT_SST_SIZE_BYTES,
            bloom_bits_per_key: 10,
            size_ratio: DEFAULT_SIZE_RATIO,
        }
    }

    /// Creates a config with a custom SST size.
    ///
    /// The SST size is clamped to the range [8 MB, 64 MB].
    pub fn with_sst_size(mut self, sst_size_bytes: u64) -> Self {
        self.sst_size_bytes = sst_size_bytes.clamp(MIN_SST_SIZE_BYTES, DEFAULT_SST_SIZE_BYTES);
        self
    }
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CompactionOutput
// ---------------------------------------------------------------------------

/// Represents the output of a compaction operation: a set of new SST files
/// with their metadata and Bloom filters.
#[derive(Debug)]
pub struct CompactionOutput {
    /// Metadata for each new SST file produced.
    pub new_ssts: Vec<SstMetadata>,
    /// Bloom filters for each new SST file.
    pub bloom_filters: Vec<SstBloomFilter>,
    /// The merged and GC'd entries grouped by output SST.
    pub sst_entries: Vec<Vec<VersionedEntry>>,
    /// Total number of entries after GC.
    pub total_entries: usize,
    /// Number of entries discarded by MVCC GC.
    pub gc_discarded: usize,
}

// ---------------------------------------------------------------------------
// Compactor
// ---------------------------------------------------------------------------

/// Orchestrates the compaction process for the LSM tree.
///
/// The compactor:
/// 1. Selects SSTs to compact based on triggers
/// 2. Merges sorted runs using `MergeIterator`
/// 3. Applies MVCC GC via `MvccGarbageCollector`
/// 4. Produces new SST files with Bloom filters
pub struct Compactor {
    /// Compaction configuration.
    config: CompactionConfig,
    /// Compaction trigger checker.
    trigger: CompactionTrigger,
    /// Next SST ID for output files.
    next_sst_id: u64,
}

impl Compactor {
    /// Creates a new compactor with the given configuration.
    pub fn new(config: CompactionConfig) -> Self {
        Self {
            trigger: CompactionTrigger::with_thresholds(
                L0_FILE_COUNT_THRESHOLD,
                config.size_ratio,
            ),
            config,
            next_sst_id: 1,
        }
    }

    /// Creates a compactor with a starting SST ID (useful for recovery).
    pub fn with_start_sst_id(mut self, start_id: u64) -> Self {
        self.next_sst_id = start_id;
        self
    }

    /// Returns a reference to the compaction config.
    pub fn config(&self) -> &CompactionConfig {
        &self.config
    }

    /// Returns a reference to the trigger.
    pub fn trigger(&self) -> &CompactionTrigger {
        &self.trigger
    }

    /// Allocates a new unique SST ID.
    fn allocate_sst_id(&mut self) -> u64 {
        let id = self.next_sst_id;
        self.next_sst_id += 1;
        id
    }

    /// Performs compaction on the given level of the LSM tree.
    ///
    /// For L0: merges all L0 SSTs into L1.
    /// For L1–L3 (tiered): merges all SSTs at this level into the next level.
    /// For L4 (bottom/leveled): merges overlapping SSTs within the level.
    ///
    /// The `input_runs` parameter provides the actual sorted data for each
    /// SST being compacted. In the current implementation, this is in-memory
    /// sorted key-value pairs (actual SST file reading will be integrated later).
    ///
    /// Returns the compaction output with new SST metadata and Bloom filters.
    pub fn compact(
        &mut self,
        tree: &mut LsmTree,
        source_level: usize,
        input_runs: Vec<Vec<VersionedEntry>>,
        gc_context: &GcContext,
    ) -> CompactionOutput {
        let target_level = if source_level < BOTTOM_LEVEL {
            source_level + 1
        } else {
            BOTTOM_LEVEL
        };

        // Count total input entries for GC stats.
        let total_input: usize = input_runs.iter().map(|r| r.len()).sum();

        // Step 1: Merge all input runs.
        let mut merge_iter = MergeIterator::new(input_runs);
        let merged = merge_iter.collect_all();

        // Step 2: Apply MVCC GC.
        let gc = MvccGarbageCollector::new(gc_context.clone());
        let gc_result = gc.apply(&merged);
        let gc_discarded = total_input.saturating_sub(gc_result.len());

        // Step 3: Split into output SSTs based on target size.
        let sst_groups = self.split_into_ssts(&gc_result);

        // Step 4: Build metadata and Bloom filters for each output SST.
        let monkey = MonkeyAllocator::new(self.config.bloom_bits_per_key, self.config.size_ratio);
        let target_fpr = monkey.fpr_for_level(target_level, NUM_LEVELS);

        let mut new_ssts = Vec::with_capacity(sst_groups.len());
        let mut bloom_filters = Vec::with_capacity(sst_groups.len());
        let mut sst_entries = Vec::with_capacity(sst_groups.len());
        let total_entries = gc_result.len();

        for group in &sst_groups {
            if group.is_empty() {
                continue;
            }

            let sst_id = self.allocate_sst_id();
            let min_key = group.first().unwrap().key.clone();
            let max_key = group.last().unwrap().key.clone();
            let size_bytes: u64 = group
                .iter()
                .map(|e| {
                    e.key.len() as u64
                        + e.value.as_ref().map_or(0, |v| v.len() as u64)
                        + 16 // timestamp + overhead
                })
                .sum();

            let metadata = SstMetadata {
                sst_id,
                level: target_level,
                min_key,
                max_key,
                size_bytes,
                row_count: group.len() as u64,
            };

            // Build Bloom filter for this SST.
            let keys: Vec<&[u8]> = group.iter().map(|e| e.key.as_slice()).collect();
            let bloom = SstBloomFilter::build(
                sst_id,
                target_level,
                keys.iter().copied(),
                target_fpr,
            );

            new_ssts.push(metadata);
            bloom_filters.push(bloom);
            sst_entries.push(group.clone());
        }

        // Step 5: Remove old SSTs from the source level in the tree.
        // (The caller is responsible for removing the specific SSTs that were compacted.)

        // Step 6: Add new SSTs to the target level.
        for sst in &new_ssts {
            tree.add_sst(target_level, sst.clone());
        }

        CompactionOutput {
            new_ssts,
            bloom_filters,
            sst_entries,
            total_entries,
            gc_discarded,
        }
    }

    /// Splits a sorted list of entries into groups, each targeting the
    /// configured SST file size.
    fn split_into_ssts(&self, entries: &[VersionedEntry]) -> Vec<Vec<VersionedEntry>> {
        if entries.is_empty() {
            return Vec::new();
        }

        let mut groups: Vec<Vec<VersionedEntry>> = Vec::new();
        let mut current_group: Vec<VersionedEntry> = Vec::new();
        let mut current_size: u64 = 0;

        for entry in entries {
            let entry_size = entry.key.len() as u64
                + entry.value.as_ref().map_or(0, |v| v.len() as u64)
                + 16; // timestamp + overhead

            // Start a new group if adding this entry would exceed the target size,
            // but only if the current group is non-empty.
            if current_size + entry_size > self.config.sst_size_bytes && !current_group.is_empty() {
                groups.push(std::mem::take(&mut current_group));
                current_size = 0;
            }

            current_size += entry_size;
            current_group.push(entry.clone());
        }

        if !current_group.is_empty() {
            groups.push(current_group);
        }

        groups
    }

    /// Performs a full compaction cycle: checks triggers and compacts if needed.
    ///
    /// Returns `Some(CompactionOutput)` if compaction was performed, or `None`
    /// if no compaction was needed.
    ///
    /// The `get_run_data` callback is called with the SST IDs to compact and
    /// should return the sorted entries for each SST.
    pub fn maybe_compact<F>(
        &mut self,
        tree: &mut LsmTree,
        gc_context: &GcContext,
        get_run_data: F,
    ) -> Option<CompactionOutput>
    where
        F: FnOnce(&[u64]) -> Vec<Vec<VersionedEntry>>,
    {
        let level = self.trigger.check(tree)?;

        // Collect SST IDs from the source level.
        let sst_ids: Vec<u64> = tree.level(level).ssts.iter().map(|s| s.sst_id).collect();

        if sst_ids.is_empty() {
            return None;
        }

        // Get the actual data for these SSTs.
        let input_runs = get_run_data(&sst_ids);

        // Remove old SSTs from the source level.
        for &sst_id in &sst_ids {
            tree.remove_sst(level, sst_id);
        }

        // Perform compaction.
        let output = self.compact(tree, level, input_runs, gc_context);

        Some(output)
    }
}
