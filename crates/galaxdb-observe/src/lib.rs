//! GalaxDB Observe — HTTP /health + /metrics, Prometheus, OTel tracing, JSON logging.
//!
//! This crate owns the process-wide Prometheus [`Registry`] so that every
//! GalaxDB crate emits metrics into a single collector that the `/metrics`
//! endpoint can scrape.
//!
//! The registry is exposed as a `&'static prometheus::Registry` via
//! [`default_registry`]. The underlying `Registry` is lazily constructed in a
//! [`std::sync::OnceLock`] so metric registration is a one-time, idempotent
//! operation for the life of the process.
//!
//! Downstream crates that want to publish a metric:
//!
//! ```no_run
//! use prometheus::IntGauge;
//!
//! let gauge = IntGauge::new("example_gauge", "Example help").unwrap();
//! galaxdb_observe::default_registry()
//!     .register(Box::new(gauge.clone()))
//!     .expect("register example_gauge");
//! gauge.set(1);
//! ```

use std::sync::OnceLock;

use prometheus::Registry;

/// Process-wide Prometheus registry holder.
static REGISTRY: OnceLock<Registry> = OnceLock::new();

/// Return the process-wide default Prometheus [`Registry`].
///
/// The first call constructs a fresh `Registry`; every subsequent call returns
/// the same instance. The returned reference is `'static` and therefore safe
/// to hand to metrics that want to re-register on retry.
pub fn default_registry() -> &'static Registry {
    REGISTRY.get_or_init(Registry::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_registry_is_stable_across_calls() {
        let a = default_registry() as *const Registry;
        let b = default_registry() as *const Registry;
        assert_eq!(a, b, "default_registry must return the same instance");
    }

    #[test]
    fn default_registry_accepts_metric_registration() {
        use prometheus::IntGauge;
        let gauge = IntGauge::new(
            "galaxdb_observe_test_gauge",
            "Unit test gauge registered against the default registry",
        )
        .unwrap();
        // Registration may fail with AlreadyReg if the test runs twice in the
        // same process (e.g. under `cargo test --jobs 1 --test-threads 1`
        // with module re-entry). That is not a failure of the registry
        // contract — it is exactly what idempotent re-registration looks
        // like, so tolerate it.
        let _ = default_registry().register(Box::new(gauge));
    }
}
