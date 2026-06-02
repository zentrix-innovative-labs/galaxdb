//! GalaxDB authentication, authorization, and audit.
//!
//! This crate defines the three security extension seams that the engine
//! consults and that the enterprise edition plugs into:
//!
//! * [`Authenticator`] — verifies a client's credentials. The open-source
//!   bundle ships [`TrustedLocalAuthenticator`] (the documented
//!   trusted-local / loopback mode) here; SCRAM-SHA-256 lands in a later
//!   task, and the enterprise edition adds SSO/OIDC.
//! * [`Authorizer`] — decides whether a role may perform an action on an
//!   object. The open-source bundle ships [`SuperuserBypassAuthorizer`]
//!   here; table-level GRANT/REVOKE lands in a later task, and the
//!   enterprise edition adds fine-grained RBAC.
//! * [`AuditSink`] — records security-relevant events. The open-source
//!   bundle ships [`NoOpAuditSink`] and [`FileAuditSink`]; the enterprise
//!   edition adds a tamper-evident sink.
//!
//! [`SecurityProviders`] bundles one of each and is assembled at server
//! startup. The engine never names an enterprise type — the bundle is the
//! seam.
//!
//! # Stability
//!
//! The three traits above are a **stable extension API**. Breaking changes
//! follow semantic versioning so the enterprise edition can pin to a
//! compatible version of this crate.
//!
//! # No mocks
//!
//! Every bundled implementation here has real, well-defined semantics.
//! There is no implementation that ignores its inputs to fake a result.
//! `SuperuserBypassAuthorizer` genuinely checks whether the session role
//! is the superuser; `TrustedLocalAuthenticator` genuinely authenticates
//! as the built-in superuser and logs that authentication is disabled.

pub mod audit;
pub mod authn;
pub mod authz;
pub mod providers;

pub use audit::{AuditEvent, AuditOutcome, AuditSink, FileAuditSink, NoOpAuditSink};
pub use authn::{
    AuthError, AuthStep, Authenticator, Role, RoleId, SessionContext, TrustedLocalAuthenticator,
};
pub use authz::{Action, AuthzError, Authorizer, ObjectRef, SuperuserBypassAuthorizer};
pub use providers::SecurityProviders;
