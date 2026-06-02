//! Authorization seam.
//!
//! [`Authorizer`] decides whether a [`crate::Role`] may perform an
//! [`Action`] on an [`ObjectRef`]. It is the single chokepoint the
//! executor consults before any storage read or write, so the wire path
//! and the embedded path enforce the same policy.
//!
//! The open core bundles [`SuperuserBypassAuthorizer`] here. The
//! table-level GRANT/REVOKE authorizer (backed by the `_galaxdb_grants`
//! catalog) lands in a later task; the enterprise edition adds
//! fine-grained (column/row-level) RBAC. All implement this trait.

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
}
