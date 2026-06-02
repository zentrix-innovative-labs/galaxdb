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

/// Looks up the stored SCRAM verifier for a role name, returning `None`
/// if the role does not exist. The engine supplies this backed by the
/// `_galaxdb_roles` catalog; tests supply an in-memory map.
pub type VerifierLookup =
    dyn Fn(&str) -> Option<crate::scram::ScramVerifier> + Send + Sync;

/// SCRAM-SHA-256 authenticator (RFC 5802 / 7677). Verifies a client's
/// password proof against the stored verifier for the claimed role,
/// without ever seeing the plaintext password. This is the mechanism
/// PostgreSQL clients use by default.
///
/// Whether the authenticated role is a superuser is decided by the
/// `is_superuser` lookup the engine supplies (backed by `_galaxdb_roles`).
pub struct ScramAuthenticator {
    verifier_lookup: std::sync::Arc<VerifierLookup>,
    superuser_lookup: std::sync::Arc<dyn Fn(&str) -> bool + Send + Sync>,
    server_nonce_len: usize,
}

impl ScramAuthenticator {
    /// Build a SCRAM authenticator from a verifier lookup (role name →
    /// stored verifier) and a superuser lookup (role name → is_superuser).
    pub fn new(
        verifier_lookup: std::sync::Arc<VerifierLookup>,
        superuser_lookup: std::sync::Arc<dyn Fn(&str) -> bool + Send + Sync>,
    ) -> Self {
        ScramAuthenticator {
            verifier_lookup,
            superuser_lookup,
            server_nonce_len: 18,
        }
    }
}

impl Authenticator for ScramAuthenticator {
    fn mechanisms(&self) -> &[&str] {
        &["SCRAM-SHA-256"]
    }

    fn begin(&self, role_hint: Option<&str>) -> AuthState {
        AuthState {
            scratch: Vec::new(),
            role_hint: role_hint.map(str::to_owned),
        }
    }

    fn step(&self, state: &mut AuthState, client_msg: &[u8]) -> AuthStep {
        use crate::scram;

        // The exchange is two server steps. We track which one we're on by
        // whether `scratch` already holds our saved context.
        if state.scratch.is_empty() {
            // --- Round 1: receive client-first, send server-first. ---
            let client_first = match std::str::from_utf8(client_msg) {
                Ok(s) => s,
                Err(_) => return AuthStep::Fail(AuthError::Malformed("client-first not UTF-8".into())),
            };
            let (username, client_nonce, client_first_bare) =
                match scram::parse_client_first(client_first) {
                    Ok(t) => t,
                    Err(e) => return AuthStep::Fail(AuthError::Malformed(e.to_string())),
                };

            let verifier = match (self.verifier_lookup)(&username) {
                Some(v) => v,
                None => return AuthStep::Fail(AuthError::UnknownRole),
            };

            let server_nonce = scram::generate_nonce(self.server_nonce_len);
            let combined_nonce = format!("{client_nonce}{server_nonce}");
            let server_first = scram::server_first_message(&combined_nonce, &verifier);

            // Save context needed in round 2, newline-delimited:
            //   username \n client_first_bare \n server_first
            let saved = format!("{username}\n{client_first_bare}\n{server_first}");
            state.scratch = saved.into_bytes();

            AuthStep::Continue(server_first.into_bytes())
        } else {
            // --- Round 2: receive client-final, verify, send server-final. ---
            let saved = String::from_utf8_lossy(&state.scratch).into_owned();
            let mut parts = saved.splitn(3, '\n');
            let username = parts.next().unwrap_or("");
            let client_first_bare = parts.next().unwrap_or("");
            let server_first = parts.next().unwrap_or("");

            let verifier = match (self.verifier_lookup)(username) {
                Some(v) => v,
                None => return AuthStep::Fail(AuthError::UnknownRole),
            };

            let client_final = match std::str::from_utf8(client_msg) {
                Ok(s) => s,
                Err(_) => return AuthStep::Fail(AuthError::Malformed("client-final not UTF-8".into())),
            };
            let (final_nonce, proof, client_final_without_proof) =
                match scram::parse_client_final(client_final) {
                    Ok(t) => t,
                    Err(e) => return AuthStep::Fail(AuthError::Malformed(e.to_string())),
                };

            // The client must echo our combined nonce (it is embedded in
            // server_first as `r=<combined>,...`).
            let expected_nonce = server_first
                .strip_prefix("r=")
                .and_then(|s| s.split(',').next())
                .unwrap_or("");
            if final_nonce != expected_nonce {
                return AuthStep::Fail(AuthError::Malformed("nonce mismatch".into()));
            }

            let auth_message =
                format!("{client_first_bare},{server_first},{client_final_without_proof}");
            match scram::verify_and_server_final(&verifier, &auth_message, &proof) {
                Ok(server_final) => {
                    let is_super = (self.superuser_lookup)(username);
                    let role = if is_super {
                        Role::superuser(username)
                    } else {
                        Role::user(username)
                    };
                    AuthStep::Success {
                        role,
                        final_message: Some(server_final.into_bytes()),
                    }
                }
                Err(_) => AuthStep::Fail(AuthError::InvalidCredentials),
            }
        }
    }

    fn name(&self) -> &str {
        "scram-sha-256"
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

    // --- ScramAuthenticator integration over the trait ---

    use crate::scram::ScramVerifier;
    use base64::Engine as _;
    use base64::engine::general_purpose::STANDARD as B64;
    use std::sync::Arc;

    /// Reference client proof for driving the server authenticator.
    fn client_proof_for(password: &str, verifier: &ScramVerifier, auth_message: &str) -> Vec<u8> {
        use hmac::{Hmac, Mac, KeyInit};
        use sha2::{Digest, Sha256};
        type H = Hmac<Sha256>;
        let mut salted = [0u8; 32];
        pbkdf2::pbkdf2::<H>(password.as_bytes(), &verifier.salt, verifier.iterations, &mut salted)
            .unwrap();
        let mut mac = H::new_from_slice(&salted).unwrap();
        mac.update(b"Client Key");
        let client_key: [u8; 32] = mac.finalize().into_bytes().into();
        let stored_key: [u8; 32] = {
            let mut h = Sha256::new();
            h.update(client_key);
            h.finalize().into()
        };
        let mut mac = H::new_from_slice(&stored_key).unwrap();
        mac.update(auth_message.as_bytes());
        let client_sig: [u8; 32] = mac.finalize().into_bytes().into();
        (0..32).map(|i| client_key[i] ^ client_sig[i]).collect()
    }

    fn scram_auth_with(role: &'static str, password: &'static str, is_super: bool) -> ScramAuthenticator {
        let verifier = ScramVerifier::from_password_with(role, vec![5u8; 16], 4096);
        let v = verifier.clone();
        let role_name = role.to_string();
        let pw_role = role_name.clone();
        let _ = password;
        ScramAuthenticator::new(
            Arc::new(move |name: &str| if name == pw_role { Some(v.clone()) } else { None }),
            Arc::new(move |name: &str| name == role_name && is_super),
        )
    }

    /// Drive the full two-round SCRAM exchange through the Authenticator
    /// trait, acting as the client.
    fn run_scram(auth: &ScramAuthenticator, user: &str, password: &str) -> AuthStep {
        let verifier = ScramVerifier::from_password_with(user, vec![5u8; 16], 4096);
        let mut state = auth.begin(Some(user));

        let client_nonce = "clientNONCE123456";
        let client_first = format!("n,,n={user},r={client_nonce}");
        let client_first_bare = format!("n={user},r={client_nonce}");

        let server_first_bytes = match auth.step(&mut state, client_first.as_bytes()) {
            AuthStep::Continue(b) => b,
            other => return other, // UnknownRole etc. surface here
        };
        let server_first = String::from_utf8(server_first_bytes).unwrap();
        let combined = server_first
            .strip_prefix("r=")
            .and_then(|s| s.split(',').next())
            .unwrap()
            .to_string();

        let without_proof = format!("c=biws,r={combined}");
        let auth_message = format!("{client_first_bare},{server_first},{without_proof}");
        let proof = client_proof_for(password, &verifier, &auth_message);
        let client_final = format!("{without_proof},p={}", B64.encode(&proof));

        auth.step(&mut state, client_final.as_bytes())
    }

    #[test]
    fn scram_authenticator_succeeds_with_correct_password() {
        let auth = scram_auth_with("alice", "alice", false);
        match run_scram(&auth, "alice", "alice") {
            AuthStep::Success { role, final_message } => {
                assert_eq!(role.id.as_str(), "alice");
                assert!(!role.is_superuser);
                let sf = String::from_utf8(final_message.unwrap()).unwrap();
                assert!(sf.starts_with("v="));
            }
            other => panic!("expected success, got {other:?}"),
        }
        assert_eq!(auth.mechanisms(), &["SCRAM-SHA-256"]);
        assert_eq!(auth.name(), "scram-sha-256");
    }

    #[test]
    fn scram_authenticator_assigns_superuser_flag() {
        let auth = scram_auth_with("admin", "admin", true);
        match run_scram(&auth, "admin", "admin") {
            AuthStep::Success { role, .. } => assert!(role.is_superuser),
            other => panic!("expected success, got {other:?}"),
        }
    }

    #[test]
    fn scram_authenticator_rejects_wrong_password() {
        let auth = scram_auth_with("alice", "alice", false);
        match run_scram(&auth, "alice", "WRONG") {
            AuthStep::Fail(AuthError::InvalidCredentials) => {}
            other => panic!("expected InvalidCredentials, got {other:?}"),
        }
    }

    #[test]
    fn scram_authenticator_rejects_unknown_role() {
        let auth = scram_auth_with("alice", "alice", false);
        // The client claims a role the lookup doesn't know.
        let mut state = auth.begin(Some("bob"));
        let client_first = "n,,n=bob,r=clientNONCE123456";
        match auth.step(&mut state, client_first.as_bytes()) {
            AuthStep::Fail(AuthError::UnknownRole) => {}
            other => panic!("expected UnknownRole, got {other:?}"),
        }
    }
}
