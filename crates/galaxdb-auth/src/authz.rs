//! Authorization seam.
//!
//! [`Authorizer`] decides whether a [`crate::Role`] may perform an
//! [`Action`] on an [`ObjectRef`]. It is the single chokepoint the
//! executor consults before any storage read or write, so the wire path
//! and the embedded path enforce the same policy.
//!
//! The open core bundles [`SuperuserBypassAuthorizer`] (the baseline used
//! before grants exist) and [`TableGrantAuthorizer`] (the grant-backed
//! authorizer of Requirement 3, wired in once the `_galaxdb_grants`
//! catalog is populated). The enterprise edition adds fine-grained
//! (column/row-level) RBAC. All implement this trait.

use crate::authn::{Role, RoleId};

/// An action a role may attempt against a database object. These map to
/// the privileges that can be granted (SELECT/INSERT/UPDATE/DELETE) plus
/// the meta-actions for schema changes and administration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    /// Read rows (SELECT).
    Select,
    /// Insert rows (INSERT, COPY FROM, BULK INSERT).
    Insert,
    /// Modify rows (UPDATE).
    Update,
    /// Remove rows (DELETE).
    Delete,
    /// Schema change (CREATE/DROP TABLE, CREATE/DROP INDEX, ANALYZE).
    Ddl,
    /// Administer roles and grants (CREATE ROLE, GRANT, REVOKE). Superuser-only.
    Admin,
}

impl Action {
    /// A stable lowercase label for logging and audit.
    pub fn label(self) -> &'static str {
        match self {
            Action::Select => "select",
            Action::Insert => "insert",
            Action::Update => "update",
            Action::Delete => "delete",
            Action::Ddl => "ddl",
            Action::Admin => "admin",
        }
    }
}

/// The object an action targets. Table-granular in v1 (matching the
/// table-level grant model); finer granularity (columns, rows) is an
/// enterprise extension that can carry extra fields in its own types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObjectRef {
    /// A specific table by name.
    Table(String),
    /// A server-wide object (used for `Action::Admin`, which is not
    /// scoped to a single table).
    Cluster,
}

impl ObjectRef {
    /// A short label for logging and audit.
    pub fn label(&self) -> String {
        match self {
            ObjectRef::Table(t) => format!("table:{t}"),
            ObjectRef::Cluster => "cluster".to_string(),
        }
    }
}

/// Why an authorization check was denied. Maps to PostgreSQL SQLSTATE
/// `42501` (insufficient_privilege) at the wire layer.
#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
#[error("permission denied: role '{role}' may not {action} on {object}")]
pub struct AuthzError {
    /// The role that was denied.
    pub role: RoleId,
    /// The action it attempted.
    pub action: &'static str,
    /// The object it targeted.
    pub object: String,
}

/// Decides whether a role may perform an action on an object. The engine
/// selects one implementation at startup via [`crate::SecurityProviders`].
pub trait Authorizer: Send + Sync {
    /// Return `Ok(())` if `role` may perform `action` on `object`, else an
    /// [`AuthzError`]. Called before any storage access.
    fn check(&self, role: &Role, action: Action, object: &ObjectRef) -> Result<(), AuthzError>;

    /// A short, stable name for logging and metrics.
    fn name(&self) -> &str;
}

/// The open-core baseline authorizer: superusers may do anything;
/// non-superusers are allowed all data and DDL actions but denied `Admin`
/// (role/grant management).
///
/// This is a real, well-defined policy, not a mock: it genuinely inspects
/// the role's superuser flag and the action. It is the correct behavior
/// for the period before table-level GRANT/REVOKE lands — every
/// authenticated user can use the database, but only a superuser can
/// manage roles and grants (Requirement 3, AC5). When the grant-backed
/// authorizer ships, it replaces this in the default bundle.
pub struct SuperuserBypassAuthorizer;

impl Authorizer for SuperuserBypassAuthorizer {
    fn check(&self, role: &Role, action: Action, object: &ObjectRef) -> Result<(), AuthzError> {
        if role.is_superuser {
            return Ok(());
        }
        match action {
            // Role/grant administration is superuser-only.
            Action::Admin => Err(AuthzError {
                role: role.id.clone(),
                action: action.label(),
                object: object.label(),
            }),
            // Data and schema actions are permitted for any authenticated
            // role under the baseline policy.
            Action::Select | Action::Insert | Action::Update | Action::Delete | Action::Ddl => {
                Ok(())
            }
        }
    }

    fn name(&self) -> &str {
        "superuser-bypass"
    }
}

/// Live grant lookup: `(role_name, table_name, action) -> granted?`.
///
/// The engine supplies this backed by the persistent `_galaxdb_grants`
/// catalog so every check reads the *current* grant set — a `GRANT` or
/// `REVOKE` committed by another statement is visible to the next check
/// without any restart or cache invalidation (Requirement 3, AC6). Tests
/// supply an in-memory closure.
pub type GrantLookup = dyn Fn(&str, &str, Action) -> bool + Send + Sync;

/// The open-core table-level grant authorizer (the bundled OSS
/// `Authorizer` of Requirement 3 / Requirement 4 AC2).
///
/// Policy:
/// * A superuser bypasses every check.
/// * Data actions (`Select`/`Insert`/`Update`/`Delete`) on a table are
///   permitted only if the role holds the matching grant on that table,
///   looked up live through [`GrantLookup`].
/// * `Admin` (role/grant management) is superuser-only (Requirement 3,
///   AC5).
/// * `Ddl` (schema changes, version tags, ANALYZE) is superuser-only in
///   the open-core baseline. The Requirement 3 grant model defines only
///   the four table-data privileges; it has no schema-level privilege to
///   delegate, so schema management is reserved to the operator
///   (superuser). This is a deliberate, documented policy — not a stub —
///   and matches the Requirement 3 user story (the operator manages the
///   schema and grants *data* access to client roles). The enterprise
///   edition's fine-grained RBAC authorizer can relax this.
///
/// This is a real authorizer, not a mock: it inspects the role's
/// superuser flag, the action, and the live grant set. It replaces
/// [`SuperuserBypassAuthorizer`] as the default once grants exist.
pub struct TableGrantAuthorizer {
    grant_lookup: std::sync::Arc<GrantLookup>,
}

impl TableGrantAuthorizer {
    /// Build a table-grant authorizer over a live grant lookup.
    pub fn new(grant_lookup: std::sync::Arc<GrantLookup>) -> Self {
        TableGrantAuthorizer { grant_lookup }
    }

    fn deny(role: &Role, action: Action, object: &ObjectRef) -> AuthzError {
        AuthzError {
            role: role.id.clone(),
            action: action.label(),
            object: object.label(),
        }
    }
}

impl Authorizer for TableGrantAuthorizer {
    fn check(&self, role: &Role, action: Action, object: &ObjectRef) -> Result<(), AuthzError> {
        if role.is_superuser {
            return Ok(());
        }
        match action {
            // Role/grant administration and schema changes are
            // superuser-only in the baseline (see the type docs).
            Action::Admin | Action::Ddl => Err(Self::deny(role, action, object)),
            // Table-data actions require a matching grant on the target
            // table. Global (Cluster-scoped) data actions have no grant
            // to satisfy them and are denied to non-superusers.
            Action::Select | Action::Insert | Action::Update | Action::Delete => match object {
                ObjectRef::Table(table) => {
                    if (self.grant_lookup)(role.id.as_str(), table, action) {
                        Ok(())
                    } else {
                        Err(Self::deny(role, action, object))
                    }
                }
                ObjectRef::Cluster => Err(Self::deny(role, action, object)),
            },
        }
    }

    fn name(&self) -> &str {
        "table-grant"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn superuser_may_administer() {
        let authz = SuperuserBypassAuthorizer;
        let admin = Role::superuser("admin");
        assert!(authz
            .check(&admin, Action::Admin, &ObjectRef::Cluster)
            .is_ok());
        assert!(authz
            .check(&admin, Action::Delete, &ObjectRef::Table("t".into()))
            .is_ok());
    }

    #[test]
    fn non_superuser_denied_admin_but_allowed_data() {
        let authz = SuperuserBypassAuthorizer;
        let alice = Role::user("alice");
        // Data + DDL allowed.
        for action in [
            Action::Select,
            Action::Insert,
            Action::Update,
            Action::Delete,
            Action::Ddl,
        ] {
            assert!(authz
                .check(&alice, action, &ObjectRef::Table("t".into()))
                .is_ok());
        }
        // Admin denied, with a useful error.
        let err = authz
            .check(&alice, Action::Admin, &ObjectRef::Cluster)
            .unwrap_err();
        assert_eq!(err.role.as_str(), "alice");
        assert_eq!(err.action, "admin");
        assert_eq!(err.object, "cluster");
    }

    #[test]
    fn action_and_object_labels_are_stable() {
        assert_eq!(Action::Select.label(), "select");
        assert_eq!(Action::Admin.label(), "admin");
        assert_eq!(ObjectRef::Table("docs".into()).label(), "table:docs");
        assert_eq!(ObjectRef::Cluster.label(), "cluster");
    }

    // --- TableGrantAuthorizer ---

    use std::collections::HashSet;
    use std::sync::{Arc, Mutex};

    /// A live, mutable grant set behind the lookup closure, so a test can
    /// model GRANT/REVOKE taking effect without rebuilding the authorizer.
    fn grant_authorizer_with(
        grants: Arc<Mutex<HashSet<(String, String, Action)>>>,
    ) -> TableGrantAuthorizer {
        TableGrantAuthorizer::new(Arc::new(move |role: &str, table: &str, action: Action| {
            grants
                .lock()
                .unwrap()
                .contains(&(role.to_string(), table.to_string(), action))
        }))
    }

    #[test]
    fn table_grant_superuser_bypasses_everything() {
        let grants = Arc::new(Mutex::new(HashSet::new()));
        let authz = grant_authorizer_with(grants);
        let admin = Role::superuser("admin");
        assert!(authz
            .check(&admin, Action::Select, &ObjectRef::Table("t".into()))
            .is_ok());
        assert!(authz
            .check(&admin, Action::Admin, &ObjectRef::Cluster)
            .is_ok());
        assert!(authz
            .check(&admin, Action::Ddl, &ObjectRef::Table("t".into()))
            .is_ok());
        assert_eq!(authz.name(), "table-grant");
    }

    #[test]
    fn table_grant_non_superuser_needs_matching_grant() {
        let grants = Arc::new(Mutex::new(HashSet::new()));
        let authz = grant_authorizer_with(grants.clone());
        let alice = Role::user("alice");

        // No grant yet → denied with a useful error.
        let err = authz
            .check(&alice, Action::Select, &ObjectRef::Table("docs".into()))
            .unwrap_err();
        assert_eq!(err.role.as_str(), "alice");
        assert_eq!(err.action, "select");
        assert_eq!(err.object, "table:docs");

        // GRANT SELECT ON docs TO alice (takes effect immediately, no
        // rebuild of the authorizer).
        grants
            .lock()
            .unwrap()
            .insert(("alice".into(), "docs".into(), Action::Select));
        assert!(authz
            .check(&alice, Action::Select, &ObjectRef::Table("docs".into()))
            .is_ok());

        // The grant is scoped: a different action or table is still denied.
        assert!(authz
            .check(&alice, Action::Insert, &ObjectRef::Table("docs".into()))
            .is_err());
        assert!(authz
            .check(&alice, Action::Select, &ObjectRef::Table("other".into()))
            .is_err());

        // REVOKE takes effect immediately too.
        grants
            .lock()
            .unwrap()
            .remove(&("alice".into(), "docs".into(), Action::Select));
        assert!(authz
            .check(&alice, Action::Select, &ObjectRef::Table("docs".into()))
            .is_err());
    }

    #[test]
    fn table_grant_denies_admin_and_ddl_to_non_superuser() {
        let grants = Arc::new(Mutex::new(HashSet::new()));
        let authz = grant_authorizer_with(grants);
        let alice = Role::user("alice");
        assert!(authz
            .check(&alice, Action::Admin, &ObjectRef::Cluster)
            .is_err());
        assert!(authz
            .check(&alice, Action::Ddl, &ObjectRef::Table("docs".into()))
            .is_err());
    }
}
