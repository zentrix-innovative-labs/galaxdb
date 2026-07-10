//! RGABH — Reinforcement-Gradient Adaptive Block Heat (single-node).
//!
//! Adds a per-block "heat" gradient to the buffer pool so admission, eviction,
//! and speculative prefetch adapt to the live access distribution instead of a
//! fixed LRU/clock policy. This is the OSS single-node form of RGABH
//! (inventory 8.1 / 8.3): the cluster-wide reinforcement variant is Enterprise.
//!
//! ## Gradient
//!
//! Each tracked block carries three exponentially-decaying moving averages plus
//! a last-access timestamp:
//!
//! - `short_heat` — fast decay; reacts to bursts. Its post-decay value is the
//!   *velocity* signal the prefetcher keys on.
//! - `long_heat` — slow decay; captures a block's durable working-set membership,
//!   so a hot block is not evicted during a transient flood of cold accesses
//!   (the failure mode of plain LRU under a mixed OLTP/scan workload).
//! - `training_heat` — bumped only by scan / training-export reads, so
//!   analytical passes influence retention without masquerading as OLTP heat.
//!
//! On each access the three EMAs are first *decayed* by the elapsed wall-clock
//! time since `last_access` (continuous-time decay, so the result is independent
//! of access cadence), then an impulse of `1.0` is added to the relevant series.
//!
//! ## Eviction
//!
//! When the adaptive buffer pool must evict, it evicts the resident block with
//! the lowest `score()` (a weighted sum of the three decayed EMAs) rather than
//! the LRU tail. This generalizes the clock-sweep "referenced" bit from a single
//! bit to a real-valued gradient.
//!
//! ## Off switch
//!
//! RGABH is opt-in. With it disabled the buffer pool uses the exact LRU/clock
//! baseline (`HeatTracker` is simply never consulted), so a non-adaptive pool is
//! byte-for-byte the pre-RGABH behavior.

use std::collections::HashMap;
use std::time::Instant;

use galaxdb_common::BlockId;

/// Tunable decay/weight constants for the heat gradient.
///
/// Decay values are per-second multipliers applied as `heat *= decay^elapsed_secs`.
/// A smaller decay forgets faster. Defaults are chosen so `short_heat` half-lives
/// in ~1s (burst detection), `long_heat` in ~60s (working-set memory), and
/// `training_heat` in ~30s.
#[derive(Debug, Clone, Copy)]
pub struct HeatConstants {
    /// Per-second decay multiplier for `short_heat` (fast).
    pub short_decay: f32,
    /// Per-second decay multiplier for `long_heat` (slow).
    pub long_decay: f32,
    /// Per-second decay multiplier for `training_heat`.
    pub training_decay: f32,
    /// Weight of `short_heat` in the eviction score.
    pub short_weight: f32,
    /// Weight of `long_heat` in the eviction score.
    pub long_weight: f32,
    /// Weight of `training_heat` in the eviction score.
    pub training_weight: f32,
}

impl Default for HeatConstants {
    fn default() -> Self {
        // half-life h ⇒ decay = 0.5^(1/h). short: ~1s, long: ~60s, training: ~30s.
        HeatConstants {
            short_decay: 0.5,             // 0.5^(1/1)
            long_decay: 0.988_5,          // ≈ 0.5^(1/60)
            training_decay: 0.977_1,      // ≈ 0.5^(1/30)
            short_weight: 1.0,
            long_weight: 2.0,             // durable membership dominates a single burst
            training_weight: 1.5,
        }
    }
}

/// Per-block heat gradient state.
#[derive(Debug, Clone)]
pub struct BlockHeat {
    /// Fast-decay EMA (burst / velocity signal).
    pub short_heat: f32,
    /// Slow-decay EMA (durable working-set membership).
    pub long_heat: f32,
    /// Scan / training-export EMA.
    pub training_heat: f32,
    /// Wall-clock time of the most recent access.
    pub last_access: Instant,
}

impl BlockHeat {
    fn new(now: Instant, is_training: bool) -> Self {
        BlockHeat {
            short_heat: 1.0,
            long_heat: 1.0,
            training_heat: if is_training { 1.0 } else { 0.0 },
            last_access: now,
        }
    }

    /// Decay all three EMAs to `now`, without adding an impulse.
    fn decay_to(&mut self, now: Instant, c: &HeatConstants) {
        let elapsed = now.saturating_duration_since(self.last_access).as_secs_f32();
        if elapsed <= 0.0 {
            return;
        }
        self.short_heat *= c.short_decay.powf(elapsed);
        self.long_heat *= c.long_decay.powf(elapsed);
        self.training_heat *= c.training_decay.powf(elapsed);
        self.last_access = now;
    }

    /// Decay to `now`, then add a `+1.0` access impulse to the relevant series.
    fn touch(&mut self, now: Instant, is_training: bool, c: &HeatConstants) {
        self.decay_to(now, c);
        self.short_heat += 1.0;
        self.long_heat += 1.0;
        if is_training {
            self.training_heat += 1.0;
        }
        self.last_access = now;
    }

    /// Weighted eviction score at `now` (higher = hotter = keep).
    pub fn score(&self, now: Instant, c: &HeatConstants) -> f32 {
        let mut h = self.clone();
        h.decay_to(now, c);
        c.short_weight * h.short_heat
            + c.long_weight * h.long_heat
            + c.training_weight * h.training_heat
    }

    /// Prefetch velocity: the decayed short-heat at `now`.
    pub fn velocity(&self, now: Instant, c: &HeatConstants) -> f32 {
        let mut h = self.clone();
        h.decay_to(now, c);
        h.short_heat
    }
}

/// Tracks the heat gradient for every block the adaptive pool has seen.
///
/// Bounded: entries are dropped as soon as their block leaves the pool
/// (`remove`), so the map never outgrows the buffer-pool residency plus
/// in-flight prefetch candidates.
pub struct HeatTracker {
    map: HashMap<BlockId, BlockHeat>,
    constants: HeatConstants,
    /// Upper bound on tracked blocks. The tracker keeps heat for resident blocks
    /// plus recently-rejected candidates (the admission "doorkeeper"); when it
    /// exceeds this cap it prunes the coldest entries back down. `usize::MAX`
    /// disables pruning (unbounded — only used in isolated unit tests).
    max_tracked: usize,
}

impl HeatTracker {
    /// New tracker with the given constants and a default cap (unbounded until
    /// [`HeatTracker::with_cap`] is used; the buffer pool sets a real cap).
    pub fn new(constants: HeatConstants) -> Self {
        HeatTracker {
            map: HashMap::new(),
            constants,
            max_tracked: usize::MAX,
        }
    }

    /// New tracker with an explicit tracked-block cap (bounds the doorkeeper).
    pub fn with_cap(constants: HeatConstants, max_tracked: usize) -> Self {
        HeatTracker {
            map: HashMap::new(),
            constants,
            max_tracked: max_tracked.max(1),
        }
    }

    /// New tracker with default constants.
    pub fn with_defaults() -> Self {
        Self::new(HeatConstants::default())
    }

    /// Record an access to `block_id` at `now`. `is_training` routes the impulse
    /// to `training_heat` as well (scan / training-export reads).
    pub fn record_access(&mut self, block_id: BlockId, now: Instant, is_training: bool) {
        match self.map.get_mut(&block_id) {
            Some(h) => h.touch(now, is_training, &self.constants),
            None => {
                self.map.insert(block_id, BlockHeat::new(now, is_training));
                if self.map.len() > self.max_tracked {
                    self.prune(now);
                }
            }
        }
    }

    /// Prune the tracker back to ~75% of its cap by dropping the coldest entries.
    /// Amortized cheap (runs only when the cap is exceeded).
    fn prune(&mut self, now: Instant) {
        let target = (self.max_tracked * 3) / 4;
        if self.map.len() <= target {
            return;
        }
        let mut scored: Vec<(BlockId, f32)> = self
            .map
            .iter()
            .map(|(id, h)| (*id, h.score(now, &self.constants)))
            .collect();
        // Sort hottest first, keep the top `target`, drop the rest.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (id, _) in scored.into_iter().skip(target) {
            self.map.remove(&id);
        }
    }

    /// Current score for a block (0.0 if untracked).
    pub fn score(&self, block_id: BlockId, now: Instant) -> f32 {
        self.map
            .get(&block_id)
            .map(|h| h.score(now, &self.constants))
            .unwrap_or(0.0)
    }

    /// Among `candidates`, return the block with the lowest score (the coldest,
    /// i.e. the eviction victim). Untracked candidates score 0.0 and are evicted
    /// first. Returns `None` for an empty candidate set.
    pub fn coldest(&self, candidates: &[BlockId], now: Instant) -> Option<BlockId> {
        candidates
            .iter()
            .min_by(|a, b| {
                let sa = self.score(**a, now);
                let sb = self.score(**b, now);
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
    }

    /// Candidate blocks to speculatively prefetch: tracked blocks whose velocity
    /// (decayed short-heat) is at or above `min_velocity`, excluding those already
    /// resident (`is_resident`). Ordered hottest-first, capped at `max`.
    ///
    /// This is a pure decision function — the caller performs the actual IO on
    /// the background (BK) queue, never the foreground path.
    pub fn prefetch_candidates(
        &self,
        now: Instant,
        min_velocity: f32,
        max: usize,
        mut is_resident: impl FnMut(BlockId) -> bool,
    ) -> Vec<BlockId> {
        let mut scored: Vec<(BlockId, f32)> = self
            .map
            .iter()
            .filter(|(id, _)| !is_resident(**id))
            .map(|(id, h)| (*id, h.velocity(now, &self.constants)))
            .filter(|(_, v)| *v >= min_velocity)
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(max);
        scored.into_iter().map(|(id, _)| id).collect()
    }

    /// Drop tracking state for a block that has left the pool. Keeps the map
    /// bounded by residency.
    pub fn remove(&mut self, block_id: BlockId) {
        self.map.remove(&block_id);
    }

    /// Number of tracked blocks (for tests / observability).
    pub fn tracked_len(&self) -> usize {
        self.map.len()
    }
}
