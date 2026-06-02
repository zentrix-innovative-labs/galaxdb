//! The security provider bundle — the registration seam.
//!
//! [`SecurityProviders`] holds one [`Authenticator`], one [`Authorizer`],
//! and one [`AuditSink`]. The server assembles it at startup and passes it
//! into the engine. The open-source `Default` is fully functional and
//! secure on its own; the enterprise edition constructs a bundle with its
//! own implementations and passes that into the same entry point — so the
//! engine never names an enterprise type.

use std::sync::Arc;

use crate::audit::{AuditSink, NoOpAuditSink};
use crate::authn::{Authenticator, TrustedLocalAuthenticator};
use crate::authz::{Authorizer, SuperuserBypassAuthorizer};

/// The default superuser name used by the open-source trusted-local
/// bundle when no explicit configuration is provided.
pub const DEFAULT_SUPERUSER: &str = "galaxdb";

/// A bundle of the three security extension implementations the engine
/// consults. Cloned cheaply (everything is an `Arc`).
#[derive(Clone)]
pub struct SecurityProviders {
    /// Verifies client credentials.
    pub authenticator: Arc<dyn Authenticator>,
    /// Decides whether a role may perform an action on an object.
    pub authorizer: Arc<dyn Authorizer>,
    /// Records security-relevant events.
    pub audit: Arc<dyn AuditSink>,
}

impl SecurityProviders {
    /// Build a bundle from explicit implementations.
    pub fn new(
        authenticator: Arc<dyn Authenticator>,
        authorizer: Arc<dyn Authorizer>,
        audit: Arc<dyn AuditSink>,
    ) -> Self {
        SecurityProviders {
            authenticator,
            authorizer,
            audit,
        }
    }

    /// The open-source default bundle: trusted-local authentication (the
    /// documented loopback/dev mode), the superuser-bypass authorizer, and
    /// no-op audit.
    ///
    /// This is functional and secure for local/trusted use. Networked
    /// deployments replace the authenticator with SCRAM (a later task);
    /// the enterprise edition replaces all three. The server logs a
    /// warning when the trusted-local authenticator is active so it can
    /// never be enabled silently.
    pub fn open_source_default() -> Self {
        SecurityProviders {
            authenticator: Arc::new(TrustedLocalAuthenticator::new(DEFAULT_SUPERUSER)),
            authorizer: Arc::new(SuperuserBypassAuthorizer),
            audit: Arc::new(NoOpAuditSink),
        }
    }
}

impl Default for SecurityProviders {
    fn default() -> Self {
        Self::open_source_default()
    }
}

impl std::fmt::Debug for SecurityProviders {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SecurityProviders")
            .field("authenticator", &self.authenticator.name())
            .field("authorizer", &self.authorizer.name())
            .field("audit", &self.audit.name())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::{AuditEvent, AuditOutcome, FileAuditSink};
    use crate::authn::{AuthStep, Role};
    use crate::authz::{Action, ObjectRef};

    #[test]
    fn default_bundle_is_all_open_source() {
        let p = SecurityProviders::default();
        assert_eq!(p.authenticator.name(), "trusted-local");
        assert_eq!(p.authorizer.name(), "superuser-bypass");
        assert_eq!(p.audit.name(), "noop");
    }

    #[test]
    fn default_bundle_authenticates_and_authorizes_end_to_end() {
        let p = SecurityProviders::default();

        // Authenticate via the trusted-local authenticator.
        let mut state = p.authenticator.begin(Some("anyone"));
        let role = match p.authenticator.step(&mut state, b"") {
            AuthStep::Success { role, .. } => role,
            other => panic!("expected success, got {other:?}"),
        };
        assert!(role.is_superuser);

        // The resulting superuser passes an admin check.
        assert!(p
            .authorizer
            .check(&role, Action::Admin, &ObjectRef::Cluster)
            .is_ok());

        // A plain user is denied admin under the default authorizer.
        let alice = Role::user("alice");
        assert!(p
            .authorizer
            .check(&alice, Action::Admin, &ObjectRef::Cluster)
            .is_err());
    }

    #[test]
    fn bundle_accepts_custom_implementations() {
        // Demonstrates the seam: a different AuditSink can be injected
        // without changing engine code. (Here we use the bundled file
        // sink; the enterprise edition would inject its own.)
        let dir = tempfile::tempdir().unwrap();
        let sink = Arc::new(FileAuditSink::open(dir.path().join("a.jsonl")).unwrap());
        let p = SecurityProviders::new(
            Arc::new(TrustedLocalAuthenticator::new("root")),
            Arc::new(SuperuserBypassAuthorizer),
            sink,
        );
        assert_eq!(p.audit.name(), "file");
        p.audit
            .record(&AuditEvent::new("auth", "login", AuditOutcome::Allowed));

        // Debug shows the selected implementations by name.
        let dbg = format!("{p:?}");
        assert!(dbg.contains("trusted-local"));
        assert!(dbg.contains("file"));
    }
}
