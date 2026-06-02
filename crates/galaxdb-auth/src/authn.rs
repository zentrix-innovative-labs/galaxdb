//! Authentication seam.
//!
//! [`Authenticator`] abstracts credential verification. It is driven as a
//! step machine so it can model multi-round exchanges (SASL/SCRAM) without
//! the wire layer needing to know the mechanism details:
//!
//! ```text
//! mechanisms() -> ["SCRAM-SHA-256"]              (advertise)
//! begin(role_hint) -> AuthState                   (start a session)
//! step(&mut AuthState, client_msg) -> AuthStep    (drive each round)
//! ```
//!
//! A single-round mechanism (e.g. trusted-local) returns
//! [`AuthStep::Success`] from the first `step`.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Stable identifier for a role (database user / principal).
///
/// A `RoleId` is the role's name. It is a newtype so the rest of the
/// engine cannot accidentally pass an arbitrary string where an
/// authenticated principal is required.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RoleId(String);

impl RoleId {
    /// Construct a role id from its name.
    pub fn new(name: impl Into<String>) -> Self {
        RoleId(name.into())
    }

    /// The role name as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for RoleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A role record: a principal plus whether it is the superuser.
///
/// Credentials (the SCRAM verifier) are stored in the auth catalog, not
/// here — this is the in-memory identity the rest of the engine carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Role {
    /// The role's identifier.
    pub id: RoleId,
    /// Whether this role bypasses authorization checks and may administer
    /// roles and grants.
    pub is_superuser: bool,
}

impl Role {
    /// A non-superuser role with the given name.
    pub fn user(name: impl Into<String>) -> Self {
        Role {
            id: RoleId::new(name),
            is_superuser: false,
        }
    }

    /// A superuser role with the given name.
    pub fn superuser(name: impl Into<String>) -> Self {
        Role {
            id: RoleId::new(name),
            is_superuser: true,
        }
    }
}

/// The authenticated context attached to a connection after a successful
/// handshake. Carried into the executor so authorization can be enforced
/// uniformly across the wire and embedded paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionContext {
    /// The authenticated role.
    pub role: Role,
}

impl SessionContext {
    /// Build a session context for an authenticated role.
    pub fn new(role: Role) -> Self {
        SessionContext { role }
    }

    /// Whether the session's role is a superuser.
    pub fn is_superuser(&self) -> bool {
        self.role.is_superuser
    }
}

/// Why authentication failed. Maps to PostgreSQL SQLSTATE `28P01`
/// (invalid_password) at the wire layer.
#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum AuthError {
    /// The supplied role does not exist.
    #[error("authentication failed: unknown role")]
    UnknownRole,
    /// The supplied credential did not verify.
    #[error("authentication failed: invalid credentials")]
    InvalidCredentials,
    /// The client sent a malformed authentication message.
    #[error("authentication failed: malformed message: {0}")]
    Malformed(String),
    /// The client requested a mechanism this authenticator does not offer.
    #[error("authentication failed: unsupported mechanism '{0}'")]
    UnsupportedMechanism(String),
}

/// The result of one authentication step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthStep {
    /// More rounds are needed. The bytes are the server's reply to send
    /// to the client (e.g. a SASL continue message).
    Continue(Vec<u8>),
    /// Authentication succeeded; the connection runs as `role`. The
    /// optional bytes are a final server message to send (e.g. SASL
    /// server-final) before `AuthenticationOk`.
    Success {
        /// The authenticated role.
        role: Role,
        /// A final message to send to the client, if the mechanism has one.
        final_message: Option<Vec<u8>>,
    },
    /// Authentication failed; the connection must be rejected and closed.
    Fail(AuthError),
}

/// Opaque per-connection authentication state. Mechanisms store their
/// in-progress exchange data here between [`Authenticator::step`] calls.
#[derive(Debug, Default)]
pub struct AuthState {
    /// Mechanism-private scratch space (e.g. the SCRAM server nonce and
    /// the stored verifier fields). Empty for single-round mechanisms.
    pub scratch: Vec<u8>,
    /// The role name the client claims, captured at `begin`.
    pub role_hint: Option<String>,
}

/// Verifies client credentials. The open core bundles
/// [`TrustedLocalAuthenticator`]; SCRAM-SHA-256 is added in a later task;
/// the enterprise edition adds SSO/OIDC. The engine selects one
/// implementation at startup via [`crate::SecurityProviders`].
pub trait Authenticator: Send + Sync {
    /// The SASL mechanism names this authenticator offers, in preference
    /// order. Used to build the `AuthenticationSASL` advertisement.
    fn mechanisms(&self) -> &[&str];

    /// Begin an authentication exchange for a connection. `role_hint` is
    /// the role name from the startup message's `user` parameter, if any.
    fn begin(&self, role_hint: Option<&str>) -> AuthState;

    /// Drive one round. `client_msg` is the latest client bytes (the SASL
    /// initial/continue response). For a single-round mechanism the first
    /// call returns [`AuthStep::Success`].
    fn step(&self, state: &mut AuthState, client_msg: &[u8]) -> AuthStep;

    /// A short, stable name for logging and metrics.
    fn name(&self) -> &str;
}

/// The trusted-local authenticator: authenticates every connection as the
/// configured superuser **without** checking a credential.
///
/// This is the real implementation behind the documented "trusted local
/// access" mode (Requirement 1, AC6) for loopback/development use. It is
/// not a mock — it has well-defined behavior (always authenticate as the
/// named superuser) and the engine logs a startup warning when it is
/// selected so it can never be enabled silently. Networked deployments
/// use the SCRAM authenticator instead.
pub struct TrustedLocalAuthenticator {
    superuser_name: String,
}

impl TrustedLocalAuthenticator {
    /// Build a trusted-local authenticator that authenticates as the given
    /// superuser name.
    pub fn new(superuser_name: impl Into<String>) -> Self {
        TrustedLocalAuthenticator {
            superuser_name: superuser_name.into(),
        }
    }
}

impl Authenticator for TrustedLocalAuthenticator {
    fn mechanisms(&self) -> &[&str] {
        // Trusted-local performs no SASL exchange. The wire layer, seeing
        // an empty mechanism list, sends `AuthenticationOk` directly.
        &[]
    }

    fn begin(&self, role_hint: Option<&str>) -> AuthState {
        AuthState {
            scratch: Vec::new(),
            role_hint: role_hint.map(str::to_owned),
        }
    }

    fn step(&self, _state: &mut AuthState, _client_msg: &[u8]) -> AuthStep {
        AuthStep::Success {
            role: Role::superuser(self.superuser_name.clone()),
            final_message: None,
        }
    }

    fn name(&self) -> &str {
        "trusted-local"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_id_roundtrips_and_displays() {
        let id = RoleId::new("alice");
        assert_eq!(id.as_str(), "alice");
        assert_eq!(id.to_string(), "alice");
        assert_eq!(id, RoleId::new("alice"));
        assert_ne!(id, RoleId::new("bob"));
    }

    #[test]
    fn superuser_and_user_roles_differ_in_privilege() {
        assert!(Role::superuser("admin").is_superuser);
        assert!(!Role::user("alice").is_superuser);
    }

    #[test]
    fn session_context_reports_superuser() {
        assert!(SessionContext::new(Role::superuser("admin")).is_superuser());
        assert!(!SessionContext::new(Role::user("alice")).is_superuser());
    }

    #[test]
    fn trusted_local_authenticates_as_superuser_in_one_step() {
        let auth = TrustedLocalAuthenticator::new("galaxdb");
        assert!(auth.mechanisms().is_empty());
        let mut state = auth.begin(Some("ignored"));
        match auth.step(&mut state, b"") {
            AuthStep::Success { role, final_message } => {
                assert!(role.is_superuser);
                assert_eq!(role.id.as_str(), "galaxdb");
                assert!(final_message.is_none());
            }
            other => panic!("expected success, got {other:?}"),
        }
        assert_eq!(auth.name(), "trusted-local");
    }
}
