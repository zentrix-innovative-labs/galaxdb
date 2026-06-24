//! Startup host probe for auto-tuned configuration (Requirement 12).
//!
//! The pure derivation/clamp/override arithmetic lives in
//! [`galaxdb_common::autotune`]; this module supplies the one piece that
//! needs the host: a probe of total physical RAM and logical CPU count.
//!
//! - CPU count comes from the std library
//!   ([`std::thread::available_parallelism`]) — no extra dependency.
//! - Total RAM comes from the `sysinfo` crate (cross-platform: macOS,
//!   Linux, Windows). When a metric is unavailable the probe records `0`,
//!   which [`SystemResources::sanitized`] later replaces with the
//!   documented conservative fallback (Requirement 12 AC4). There is no
//!   silent substitution of a fabricated host size — a zero is an honest
//!   "could not detect" that the documented fallback then covers.

use galaxdb_common::{AutoTuneConfig, EffectiveTuning, SystemResources};

/// Probe the host for total physical RAM (bytes) and logical CPU count.
///
/// Either metric may come back as `0` if the platform does not expose it;
/// callers should run the result through [`SystemResources::sanitized`]
/// (which [`resolve_tuning`] does) so a missed probe maps to the
/// documented fallback rather than a zero-sized buffer pool.
pub fn probe_system_resources() -> SystemResources {
    let total_ram_bytes = {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        sys.total_memory() // bytes (sysinfo >= 0.30 reports bytes, not KiB)
    };

    let logical_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);

    SystemResources {
        total_ram_bytes,
        logical_cpus,
    }
}

/// Probe the host, resolve the operator's [`AutoTuneConfig`] against it
/// (overrides always win, Requirement 12 AC2), and return the effective
/// tuning with a per-value source for logging (AC3).
pub fn resolve_tuning(cfg: &AutoTuneConfig) -> EffectiveTuning {
    cfg.resolve(probe_system_resources())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_reports_real_host_resources() {
        // Real probe against the test host — no mock. The machine running
        // the test has RAM and at least one CPU, so both must be non-zero.
        let res = probe_system_resources();
        assert!(
            res.total_ram_bytes > 0,
            "total RAM probe returned 0 on a host that clearly has memory"
        );
        assert!(
            res.logical_cpus > 0,
            "logical CPU probe returned 0 on a host that clearly has cores"
        );
    }

    #[test]
    fn resolve_tuning_applies_probe_and_describes_sources() {
        let eff = resolve_tuning(&AutoTuneConfig::default());
        // Default config enables auto-tune, so the un-overridden values are
        // derived from the (sanitized) probe.
        let text = eff.describe();
        assert!(text.contains("buffer_pool="));
        assert!(text.contains("memtable="));
        assert!(text.contains("compaction_concurrency="));
        // Derived values must respect the documented clamp bounds.
        assert!(eff.buffer_pool_bytes.value >= AutoTuneConfig::MIN_BUFFER_POOL_BYTES);
        assert!(eff.buffer_pool_bytes.value <= AutoTuneConfig::MAX_BUFFER_POOL_BYTES);
        assert!(eff.memtable_size_bytes.value >= AutoTuneConfig::MIN_MEMTABLE_BYTES);
        assert!(eff.memtable_size_bytes.value <= AutoTuneConfig::MAX_MEMTABLE_BYTES);
        assert!(eff.compaction_concurrency.value >= AutoTuneConfig::MIN_COMPACTION_CONCURRENCY);
        assert!(eff.compaction_concurrency.value <= AutoTuneConfig::MAX_COMPACTION_CONCURRENCY);
    }

    #[test]
    fn explicit_overrides_beat_the_probe() {
        let cfg = AutoTuneConfig {
            enabled: true,
            buffer_pool_bytes: Some(512 * 1024 * 1024),
            memtable_size_bytes: Some(32 * 1024 * 1024),
            compaction_concurrency: Some(3),
        };
        let eff = resolve_tuning(&cfg);
        assert_eq!(eff.buffer_pool_bytes.value, 512 * 1024 * 1024);
        assert_eq!(eff.memtable_size_bytes.value, 32 * 1024 * 1024);
        assert_eq!(eff.compaction_concurrency.value, 3);
    }
}
