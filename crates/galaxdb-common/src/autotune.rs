//! Auto-tuned configuration derivation (Requirement 12).
//!
//! At startup the server probes the host for total RAM and logical CPU count,
//! then derives sensible defaults for the buffer-pool size, memtable size, and
//! compaction concurrency. The probe itself lives in `galaxdb-server` (it needs
//! an OS-specific crate); everything in this module is pure data + arithmetic so
//! it can be unit-tested without touching the host.
//!
//! Design rules (Requirement 12):
//! - AC1: derive buffer-pool size, memtable size, and compaction concurrency
//!   from detected RAM and CPU count.
//! - AC2: an explicit config value ALWAYS overrides the derived value.
//! - AC3: the effective configuration is reported with the source of each value
//!   (derived vs overridden vs static default) so the operator can log it.
//! - AC4: cross-platform; a [`SystemResources`] with a missing metric (zero) is
//!   replaced by a documented conservative fallback via [`SystemResources::sanitized`].
//! - AC5: derived values are clamped so they never violate the v1
//!   RateLimiter/WriteController write-stall invariants (see the clamp bounds
//!   and [`AutoTuneConfig::INVARIANT_BACK_PRESSURE_BYTES`]).

use serde::{Deserialize, Serialize};

/// Detected (or fallback) host resources used to derive default sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemResources {
    /// Total physical RAM in bytes. `0` means "could not detect".
    pub total_ram_bytes: u64,
    /// Number of logical CPUs. `0` means "could not detect".
    pub logical_cpus: usize,
}

impl SystemResources {
    /// Conservative RAM fallback when the platform does not expose the metric
    /// (Requirement 12 AC4): assume a small 4 GiB host so derived sizes stay modest.
    pub const FALLBACK_RAM_BYTES: u64 = 4 * 1024 * 1024 * 1024;
    /// Conservative CPU fallback when the platform does not expose the metric.
    pub const FALLBACK_CPUS: usize = 4;

    /// The documented fallback resources (used when no metric is available).
    pub fn fallback() -> Self {
        Self {
            total_ram_bytes: Self::FALLBACK_RAM_BYTES,
            logical_cpus: Self::FALLBACK_CPUS,
        }
    }

    /// Replace any missing (zero) metric with its documented fallback. Derivation
    /// always runs against sanitized resources so a probe miss never yields a
    /// zero-sized buffer pool or zero compaction threads.
    pub fn sanitized(self) -> Self {
        Self {
            total_ram_bytes: if self.total_ram_bytes == 0 {
                Self::FALLBACK_RAM_BYTES
            } else {
                self.total_ram_bytes
            },
            logical_cpus: if self.logical_cpus == 0 {
                Self::FALLBACK_CPUS
            } else {
                self.logical_cpus
            },
        }
    }
}

/// Where an effective configuration value came from (for the startup log, AC3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TuningSource {
    /// Auto-derived from detected system resources.
    Derived,
    /// Explicitly set by the operator (overrides the derived value).
    Overridden,
    /// Auto-tune disabled and no override given: the static built-in default.
    StaticDefault,
}

impl TuningSource {
    /// Short human-readable label for the startup log line.
    pub fn label(self) -> &'static str {
        match self {
            TuningSource::Derived => "auto-derived",
            TuningSource::Overridden => "overridden",
            TuningSource::StaticDefault => "static-default",
        }
    }
}

/// A resolved value together with where it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedValue<T> {
    pub value: T,
    pub source: TuningSource,
}

/// The three values auto-tuning manages, after resolving overrides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectiveTuning {
    pub buffer_pool_bytes: ResolvedValue<u64>,
    pub memtable_size_bytes: ResolvedValue<u64>,
    pub compaction_concurrency: ResolvedValue<usize>,
}

impl EffectiveTuning {
    /// A single-line, operator-facing description of the effective tuning and
    /// the source of each value (Requirement 12 AC3).
    pub fn describe(&self) -> String {
        format!(
            "auto-tune: buffer_pool={} MiB ({}), memtable={} MiB ({}), compaction_concurrency={} ({})",
            self.buffer_pool_bytes.value / (1024 * 1024),
            self.buffer_pool_bytes.source.label(),
            self.memtable_size_bytes.value / (1024 * 1024),
            self.memtable_size_bytes.source.label(),
            self.compaction_concurrency.value,
            self.compaction_concurrency.source.label(),
        )
    }
}

/// The purely-derived values (before override resolution).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DerivedTuning {
    pub buffer_pool_bytes: u64,
    pub memtable_size_bytes: u64,
    pub compaction_concurrency: usize,
}

impl DerivedTuning {
    /// Derive defaults from host resources (Requirement 12 AC1), with all values
    /// clamped to ranges that preserve the write-stall-mitigation invariants (AC5).
    pub fn derive(resources: SystemResources) -> Self {
        let res = resources.sanitized();

        // Buffer pool: 25% of RAM, clamped to [64 MiB, 16 GiB]. Conservative so we
        // never starve the OS page cache or the memtable/WAL working set.
        let buffer_pool_bytes = (res.total_ram_bytes / 4).clamp(
            AutoTuneConfig::MIN_BUFFER_POOL_BYTES,
            AutoTuneConfig::MAX_BUFFER_POOL_BYTES,
        );

        // Memtable: 1/64 of RAM, clamped to [16 MiB, 128 MiB]. The 128 MiB ceiling
        // keeps at least two memtables under the 256 MiB sealed-but-unflushed
        // back-pressure budget so the WriteController back-pressure invariant holds
        // (AC5): 2 * memtable_max <= INVARIANT_BACK_PRESSURE_BYTES.
        let memtable_size_bytes = (res.total_ram_bytes / 64).clamp(
            AutoTuneConfig::MIN_MEMTABLE_BYTES,
            AutoTuneConfig::MAX_MEMTABLE_BYTES,
        );

        // Compaction concurrency: ~1/4 of cores, clamped to [1, 8]. Leaving the
        // majority of cores for foreground query + flush work preserves the SILK
        // flush-pre-emption invariant (compaction must never starve flush).
        let compaction_concurrency = (res.logical_cpus / 4).clamp(
            AutoTuneConfig::MIN_COMPACTION_CONCURRENCY,
            AutoTuneConfig::MAX_COMPACTION_CONCURRENCY,
        );

        Self {
            buffer_pool_bytes,
            memtable_size_bytes,
            compaction_concurrency,
        }
    }
}

/// Auto-tune configuration. When `enabled`, missing values are derived from the
/// host; any `Some(..)` override always wins (Requirement 12 AC2).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AutoTuneConfig {
    /// Master switch (default: `true`).
    pub enabled: bool,
    /// Explicit buffer-pool size override in bytes.
    #[serde(default)]
    pub buffer_pool_bytes: Option<u64>,
    /// Explicit memtable size override in bytes.
    #[serde(default)]
    pub memtable_size_bytes: Option<u64>,
    /// Explicit compaction concurrency override.
    #[serde(default)]
    pub compaction_concurrency: Option<usize>,
}

impl Default for AutoTuneConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            buffer_pool_bytes: None,
            memtable_size_bytes: None,
            compaction_concurrency: None,
        }
    }
}

impl AutoTuneConfig {
    // Clamp bounds (also the documented invariant guards for AC5).
    pub const MIN_BUFFER_POOL_BYTES: u64 = 64 * 1024 * 1024; // 64 MiB
    pub const MAX_BUFFER_POOL_BYTES: u64 = 16 * 1024 * 1024 * 1024; // 16 GiB
    pub const MIN_MEMTABLE_BYTES: u64 = 16 * 1024 * 1024; // 16 MiB
    pub const MAX_MEMTABLE_BYTES: u64 = 128 * 1024 * 1024; // 128 MiB
    pub const MIN_COMPACTION_CONCURRENCY: usize = 1;
    pub const MAX_COMPACTION_CONCURRENCY: usize = 8;

    /// The v1 sealed-but-unflushed back-pressure budget the memtable ceiling must
    /// respect (StorageConfig default); used by the AC5 invariant assertion.
    pub const INVARIANT_BACK_PRESSURE_BYTES: u64 = 256 * 1024 * 1024;

    // Static defaults used when auto-tune is disabled and no override is given.
    pub const STATIC_BUFFER_POOL_BYTES: u64 = 256 * 1024 * 1024; // 256 MiB
    pub const STATIC_MEMTABLE_BYTES: u64 = 64 * 1024 * 1024; // 64 MiB (matches StorageConfig)
    pub const STATIC_COMPACTION_CONCURRENCY: usize = 4;

    /// Resolve the effective tuning against detected resources, honoring overrides
    /// (AC2) and recording the source of each value (AC3).
    ///
    /// The AC5 invariant (two full memtables fit under the back-pressure budget)
    /// is enforced at compile time by the const assertion below this `impl`. against detected resources, honoring overrides
    /// (AC2) and recording the source of each value (AC3).
    pub fn resolve(&self, resources: SystemResources) -> EffectiveTuning {
        let derived = DerivedTuning::derive(resources);

        let buffer_pool_bytes = self.resolve_u64(
            self.buffer_pool_bytes,
            derived.buffer_pool_bytes,
            Self::STATIC_BUFFER_POOL_BYTES,
        );
        let memtable_size_bytes = self.resolve_u64(
            self.memtable_size_bytes,
            derived.memtable_size_bytes,
            Self::STATIC_MEMTABLE_BYTES,
        );
        let compaction_concurrency = self.resolve_usize(
            self.compaction_concurrency,
            derived.compaction_concurrency,
            Self::STATIC_COMPACTION_CONCURRENCY,
        );

        EffectiveTuning {
            buffer_pool_bytes,
            memtable_size_bytes,
            compaction_concurrency,
        }
    }

    fn resolve_u64(
        &self,
        override_value: Option<u64>,
        derived: u64,
        static_default: u64,
    ) -> ResolvedValue<u64> {
        match override_value {
            Some(value) => ResolvedValue {
                value,
                source: TuningSource::Overridden,
            },
            None if self.enabled => ResolvedValue {
                value: derived,
                source: TuningSource::Derived,
            },
            None => ResolvedValue {
                value: static_default,
                source: TuningSource::StaticDefault,
            },
        }
    }

    fn resolve_usize(
        &self,
        override_value: Option<usize>,
        derived: usize,
        static_default: usize,
    ) -> ResolvedValue<usize> {
        match override_value {
            Some(value) => ResolvedValue {
                value,
                source: TuningSource::Overridden,
            },
            None if self.enabled => ResolvedValue {
                value: derived,
                source: TuningSource::Derived,
            },
            None => ResolvedValue {
                value: static_default,
                source: TuningSource::StaticDefault,
            },
        }
    }
}

/// Compile-time guard for Requirement 12 AC5: two full memtables must fit under
/// the sealed-but-unflushed back-pressure budget, so the WriteController
/// back-pressure can never deadlock waiting on a single oversized memtable.
const _MEMTABLE_CEILING_INVARIANT: () = assert!(
    2 * AutoTuneConfig::MAX_MEMTABLE_BYTES <= AutoTuneConfig::INVARIANT_BACK_PRESSURE_BYTES
);

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn derive_scales_with_resources() {
        // 32 GiB / 16 cores (the c6id.4xlarge benchmark host).
        let res = SystemResources {
            total_ram_bytes: 32 * GIB,
            logical_cpus: 16,
        };
        let d = DerivedTuning::derive(res);
        assert_eq!(d.buffer_pool_bytes, 8 * GIB); // 25% of 32 GiB
        assert_eq!(d.memtable_size_bytes, AutoTuneConfig::MAX_MEMTABLE_BYTES); // 32 GiB/64 = 512 MiB → clamped to 128 MiB
        assert_eq!(d.compaction_concurrency, 4); // 16/4
    }

    #[test]
    fn derive_clamps_small_host_to_minimums() {
        let res = SystemResources {
            total_ram_bytes: GIB,
            logical_cpus: 1,
        };
        let d = DerivedTuning::derive(res);
        // 1 GiB/4 = 256 MiB ≥ 64 MiB min, so not clamped low here.
        assert_eq!(d.buffer_pool_bytes, 256 * 1024 * 1024);
        // 1 GiB/64 = 16 MiB == min.
        assert_eq!(d.memtable_size_bytes, AutoTuneConfig::MIN_MEMTABLE_BYTES);
        // 1/4 = 0 → clamped to min 1.
        assert_eq!(d.compaction_concurrency, 1);
    }

    #[test]
    fn derive_clamps_huge_host_to_maximums() {
        let res = SystemResources {
            total_ram_bytes: 512 * GIB,
            logical_cpus: 128,
        };
        let d = DerivedTuning::derive(res);
        assert_eq!(d.buffer_pool_bytes, AutoTuneConfig::MAX_BUFFER_POOL_BYTES); // 128 GiB → clamped 16 GiB
        assert_eq!(d.memtable_size_bytes, AutoTuneConfig::MAX_MEMTABLE_BYTES);
        assert_eq!(d.compaction_concurrency, AutoTuneConfig::MAX_COMPACTION_CONCURRENCY); // 32 → clamped 8
    }

    #[test]
    fn ac4_zero_metrics_fall_back() {
        let res = SystemResources {
            total_ram_bytes: 0,
            logical_cpus: 0,
        }
        .sanitized();
        assert_eq!(res.total_ram_bytes, SystemResources::FALLBACK_RAM_BYTES);
        assert_eq!(res.logical_cpus, SystemResources::FALLBACK_CPUS);
    }

    #[test]
    fn ac2_explicit_override_always_wins() {
        let cfg = AutoTuneConfig {
            enabled: true,
            buffer_pool_bytes: Some(123 * 1024 * 1024),
            memtable_size_bytes: None,
            compaction_concurrency: Some(7),
        };
        let res = SystemResources {
            total_ram_bytes: 32 * GIB,
            logical_cpus: 16,
        };
        let eff = cfg.resolve(res);
        assert_eq!(eff.buffer_pool_bytes.value, 123 * 1024 * 1024);
        assert_eq!(eff.buffer_pool_bytes.source, TuningSource::Overridden);
        assert_eq!(eff.compaction_concurrency.value, 7);
        assert_eq!(eff.compaction_concurrency.source, TuningSource::Overridden);
        // The un-overridden memtable is derived.
        assert_eq!(eff.memtable_size_bytes.source, TuningSource::Derived);
    }

    #[test]
    fn disabled_uses_static_defaults_unless_overridden() {
        let cfg = AutoTuneConfig {
            enabled: false,
            buffer_pool_bytes: None,
            memtable_size_bytes: Some(99 * 1024 * 1024),
            compaction_concurrency: None,
        };
        let res = SystemResources {
            total_ram_bytes: 32 * GIB,
            logical_cpus: 16,
        };
        let eff = cfg.resolve(res);
        assert_eq!(
            eff.buffer_pool_bytes.value,
            AutoTuneConfig::STATIC_BUFFER_POOL_BYTES
        );
        assert_eq!(eff.buffer_pool_bytes.source, TuningSource::StaticDefault);
        assert_eq!(eff.memtable_size_bytes.value, 99 * 1024 * 1024);
        assert_eq!(eff.memtable_size_bytes.source, TuningSource::Overridden);
        assert_eq!(
            eff.compaction_concurrency.value,
            AutoTuneConfig::STATIC_COMPACTION_CONCURRENCY
        );
        assert_eq!(
            eff.compaction_concurrency.source,
            TuningSource::StaticDefault
        );
    }

    #[test]
    fn describe_mentions_each_source() {
        let cfg = AutoTuneConfig::default();
        let res = SystemResources {
            total_ram_bytes: 32 * GIB,
            logical_cpus: 16,
        };
        let text = cfg.resolve(res).describe();
        assert!(text.contains("buffer_pool="));
        assert!(text.contains("memtable="));
        assert!(text.contains("compaction_concurrency="));
        assert!(text.contains("auto-derived"));
    }

    #[test]
    fn config_round_trips_through_json() {
        let cfg = AutoTuneConfig::default();
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: AutoTuneConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cfg, back);
    }
}
