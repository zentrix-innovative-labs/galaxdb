//! The top-level provider bundle that the server accepts at startup.
//!
//! [`Providers`] holds one slot for each pluggable subsystem:
//!
//! * `security` — [`galaxdb_auth::SecurityProviders`] (authentication,
//!   authorization, audit).
//! * `key_spec` — `Option<`[`galaxdb_crypto::KeyProviderSpec`]`>`: encryption
//!   key management. `None` means TDE is not configured for this deployment.
//! * `cluster` — `Arc<dyn ClusterCoordinator>`: topology and routing.
//!
//! Assemble a custom bundle with [`Providers::new`], or call
//! [`Providers::single_node_default`] for the all-OSS default that needs no
//! configuration.

use std::sync::Arc;

use galaxdb_auth::SecurityProviders;
use galaxdb_crypto::KeyProviderSpec;

use crate::coordinator::{ClusterCoordinator, single_node_arc};

/// The complete set of pluggable provider implementations the server uses.
///
/// # Stability
///
/// This struct and its fields form part of the **semver-stable extension
/// boundary**. Enterprise editions construct a [`Providers`] with their own
/// implementations and pass it into [`crate::ServerConfig::providers`]; the
/// engine never names an enterprise type.
///
/// Adding new *optional* fields (wrapped in `Option`) is a semver-compatible
/// change. Removing or changing existing fields is a semver-major change.
#[derive(Clone)]
pub struct Providers {
    /// Security subsystem: authentication, authorization, and audit.
    pub security: SecurityProviders,
    /// Key-management specification for transparent data encryption.
    /// `None` when TDE is not configured for this deployment.
    pub key_spec: Option<KeyProviderSpec>,
    /// Cluster coordinator: topology queries and query routing.
    pub cluster: Arc<dyn ClusterCoordinator + Send + Sync>,
}

impl Providers {
    /// Build a [`Providers`] bundle from explicit implementations.
    ///
    /// Use this constructor when at least one slot needs a non-default
    /// implementation (e.g. the enterprise edition's OIDC authenticator or
    /// Raft coordinator).
    pub fn new(
        security: SecurityProviders,
        key_spec: Option<KeyProviderSpec>,
        cluster: Arc<dyn ClusterCoordinator + Send + Sync>,
    ) -> Self {
        Providers {
            security,
            key_spec,
            cluster,
        }
    }

    /// Build the all-OSS default bundle: trusted-local security providers,
    /// no TDE key spec (TDE off), and a [`crate::SingleNodeCoordinator`].
    ///
    /// This is the bundle used by the open-source binary when no custom
    /// [`Providers`] is supplied. It is fully functional for standalone
    /// deployments; enterprise editions supply their own bundle.
    pub fn single_node_default() -> Self {
        Providers {
            security: SecurityProviders::open_source_default(),
            key_spec: None,
            cluster: single_node_arc(),
        }
    }
}

impl std::fmt::Debug for Providers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Providers")
            .field("security", &self.security)
            .field("key_spec", &self.key_spec)
            .field("cluster_coordinator", &self.cluster.name())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coordinator::NodeRole;

    #[test]
    fn providers_default_builds_without_panic() {
        let p = Providers::single_node_default();
        // Security defaults are OSS.
        assert_eq!(p.security.authenticator.name(), "trusted-local");
        assert_eq!(p.security.authorizer.name(), "superuser-bypass");
        assert_eq!(p.security.audit.name(), "noop");
        // TDE is off by default.
        assert!(p.key_spec.is_none());
        // Cluster is single-node.
        assert_eq!(p.cluster.name(), "single-node");
        assert!(p.cluster.is_leader());
        assert_eq!(p.cluster.get_node_role(), NodeRole::Standalone);
    }

    #[test]
    fn providers_new_accepts_custom_bundle() {
        use std::sync::Arc;
        use crate::coordinator::SingleNodeCoordinator;

        let p = Providers::new(
            SecurityProviders::open_source_default(),
            None,
            Arc::new(SingleNodeCoordinator),
        );
        assert_eq!(p.cluster.name(), "single-node");
        assert!(p.key_spec.is_none());
    }

    #[test]
    fn providers_debug_includes_coordinator_name() {
        let p = Providers::single_node_default();
        let dbg = format!("{p:?}");
        assert!(dbg.contains("single-node"), "expected coordinator name in debug: {dbg}");
    }
}
