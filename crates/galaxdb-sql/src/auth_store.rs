//! Persistent authentication catalog (the Auth_Catalog of Requirement 3).
//!
//! Roles and grants are stored as reserved rows in the storage engine
//! under dedicated key prefixes, so they:
//!
//! * survive restart and crash recovery through the normal WAL + SST path
//!   (every write goes through [`Engine::put_sync`] / [`Engine::delete_sync`]),
//! * are **not** exposed as an ordinary SQL table, so a client cannot
//!   `DELETE FROM _galaxdb_roles` to wipe authentication or `UPDATE` a
//!   verifier — the only way to mutate them is `CREATE/DROP/ALTER ROLE`
//!   and `GRANT/REVOKE`, which are authorization-checked.
//!
//! Why not the append-only system-table pattern (as `_galaxdb_training_exports`
//! uses)? Because auth needs `ALTER ROLE ... PASSWORD` (update) and
//! `DROP ROLE` / `REVOKE` (delete), which append-only tables forbid. A
//! dedicated reserved-key store gives full CRUD with the same durability
//! while keeping the records off the SQL surface.
//!
//! ## Key layout
//!
//! ```text
//! role:  b"\x00galaxdb_auth\x00role\x00"  + role_name        -> RoleRecord bytes
//! grant: b"\x00galaxdb_auth\x00grant\x00" + role "\x00" table "\x00" priv -> b"1"
//! ```
//!
//! The `\x00galaxdb_auth\x00` sentinel prefix cannot collide with a user
//! table's row keys (those are `"<table>:<pk>"` ASCII, no leading NUL).

use std::sync::Arc;

use galaxdb_auth::{Action, ScramVerifier};
use galaxdb_common::{GalaxError, GalaxResult};
use galaxdb_storage::engine::Engine;

const ROLE_PREFIX: &[u8] = b"\x00galaxdb_auth\x00role\x00";
const GRANT_PREFIX: &[u8] = b"\x00galaxdb_auth\x00grant\x00";

/// A stored role: its name, whether it is a superuser, and (optionally)
/// its SCRAM credential. A role may exist without a password (e.g. created
/// then password set later); authentication against a passwordless role
/// fails closed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoleRecord {
    /// Role name.
    pub name: String,
    /// Whether this role bypasses authorization and may administer roles/grants.
    pub is_superuser: bool,
    /// SCRAM verifier, if a password has been set.
    pub verifier: Option<ScramVerifier>,
}

impl RoleRecord {
    fn to_bytes(&self) -> Vec<u8> {
        // [is_superuser:u8][name_len:u16 LE][name][has_verifier:u8][verifier?]
        let mut out = Vec::new();
        out.push(self.is_superuser as u8);
        out.extend_from_slice(&(self.name.len() as u16).to_le_bytes());
        out.extend_from_slice(self.name.as_bytes());
        match &self.verifier {
            Some(v) => {
                out.push(1);
                out.extend_from_slice(&v.to_bytes());
            }
            None => out.push(0),
        }
        out
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 4 {
            return None;
        }
        let is_superuser = bytes[0] != 0;
        let name_len = u16::from_le_bytes([bytes[1], bytes[2]]) as usize;
        let name_start = 3;
        let name_end = name_start + name_len;
        if bytes.len() < name_end + 1 {
            return None;
        }
        let name = String::from_utf8(bytes[name_start..name_end].to_vec()).ok()?;
        let has_verifier = bytes[name_end] != 0;
        let verifier = if has_verifier {
            Some(ScramVerifier::from_bytes(&bytes[name_end + 1..])?)
        } else {
            None
        };
        Some(RoleRecord {
            name,
            is_superuser,
            verifier,
        })
    }
}

fn role_key(name: &str) -> Vec<u8> {
    let mut k = ROLE_PREFIX.to_vec();
    k.extend_from_slice(name.as_bytes());
    k
}

fn grant_key(role: &str, table: &str, privilege: Action) -> Vec<u8> {
    let mut k = GRANT_PREFIX.to_vec();
    k.extend_from_slice(role.as_bytes());
    k.push(0);
    k.extend_from_slice(table.as_bytes());
    k.push(0);
    k.extend_from_slice(privilege.label().as_bytes());
    k
}

/// The persistent role + grant catalog, backed by the storage engine.
///
/// Cheap to clone (holds an `Arc<Engine>`). All reads go through the
/// engine's ART/memtable/SST path, so they reflect committed state after
/// restart.
#[derive(Clone)]
pub struct AuthStore {
    engine: Arc<Engine>,
}

impl AuthStore {
    /// Wrap an engine handle.
    pub fn new(engine: Arc<Engine>) -> Self {
        AuthStore { engine }
    }

    // ---- Roles ----

    /// Create or replace a role record.
    pub fn put_role(&self, record: &RoleRecord) -> GalaxResult<()> {
        self.engine
            .put_sync(role_key(&record.name), record.to_bytes())
            .map(|_| ())
            .map_err(|e| GalaxError::Internal(format!("auth store put_role: {e}")))
    }

    /// Fetch a role by name, or `None` if it does not exist.
    pub fn get_role(&self, name: &str) -> Option<RoleRecord> {
        let bytes = self.engine.get(&role_key(name))?;
        RoleRecord::from_bytes(&bytes)
    }

    /// Remove a role and all of its grants. Returns `true` if the role
    /// existed.
    pub fn drop_role(&self, name: &str) -> GalaxResult<bool> {
        let existed = self.get_role(name).is_some();
        if existed {
            self.engine
                .delete_sync(&role_key(name))
                .map_err(|e| GalaxError::Internal(format!("auth store drop_role: {e}")))?;
            // Best-effort cascade: drop every grant held by this role.
            for (r, t, p) in self.list_grants() {
                if r == name {
                    let _ = self.engine.delete_sync(&grant_key(&r, &t, p));
                }
            }
        }
        Ok(existed)
    }

    /// Whether any role exists. Used at startup to decide whether to
    /// provision the initial superuser.
    pub fn any_role_exists(&self) -> bool {
        let rows = self.engine.scan_all_with_prefix(Some(ROLE_PREFIX));
        !rows.is_empty()
    }

    /// Convenience: is this role a superuser? `false` if it doesn't exist.
    pub fn is_superuser(&self, name: &str) -> bool {
        self.get_role(name).map(|r| r.is_superuser).unwrap_or(false)
    }

    /// Convenience: the stored SCRAM verifier for a role, if any.
    pub fn verifier_for(&self, name: &str) -> Option<ScramVerifier> {
        self.get_role(name).and_then(|r| r.verifier)
    }

    // ---- Grants ----

    /// Grant a privilege on a table to a role (idempotent).
    pub fn grant(&self, role: &str, table: &str, privilege: Action) -> GalaxResult<()> {
        self.engine
            .put_sync(grant_key(role, table, privilege), vec![1])
            .map(|_| ())
            .map_err(|e| GalaxError::Internal(format!("auth store grant: {e}")))
    }

    /// Revoke a privilege (idempotent — revoking a non-existent grant is
    /// a no-op success).
    pub fn revoke(&self, role: &str, table: &str, privilege: Action) -> GalaxResult<()> {
        self.engine
            .delete_sync(&grant_key(role, table, privilege))
            .map(|_| ())
            .map_err(|e| GalaxError::Internal(format!("auth store revoke: {e}")))
    }

    /// Whether a role holds a specific privilege on a table.
    pub fn has_grant(&self, role: &str, table: &str, privilege: Action) -> bool {
        self.engine.get(&grant_key(role, table, privilege)).is_some()
    }

    /// List every `(role, table, privilege)` grant currently stored.
    pub fn list_grants(&self) -> Vec<(String, String, Action)> {
        let rows = self.engine.scan_all_with_prefix(Some(GRANT_PREFIX));
        let mut out = Vec::new();
        for (key, _val) in rows {
            if let Some(parsed) = parse_grant_key(&key) {
                out.push(parsed);
            }
        }
        out
    }
}

fn parse_grant_key(key: &[u8]) -> Option<(String, String, Action)> {
    let rest = key.strip_prefix(GRANT_PREFIX)?;
    // rest = role \x00 table \x00 priv
    let mut parts = rest.split(|&b| b == 0);
    let role = std::str::from_utf8(parts.next()?).ok()?.to_string();
    let table = std::str::from_utf8(parts.next()?).ok()?.to_string();
    let priv_str = std::str::from_utf8(parts.next()?).ok()?;
    let action = action_from_label(priv_str)?;
    Some((role, table, action))
}

fn action_from_label(label: &str) -> Option<Action> {
    match label {
        "select" => Some(Action::Select),
        "insert" => Some(Action::Insert),
        "update" => Some(Action::Update),
        "delete" => Some(Action::Delete),
        "ddl" => Some(Action::Ddl),
        "admin" => Some(Action::Admin),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use galaxdb_storage::engine::{Engine, EngineConfig};

    fn test_engine() -> Arc<Engine> {
        let dir = tempfile::tempdir().unwrap();
        let config = EngineConfig {
            data_dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        // Leak the tempdir so files survive for the test's engine lifetime.
        std::mem::forget(dir);
        Arc::new(Engine::new(config).unwrap())
    }

    #[test]
    fn role_record_byte_roundtrip_with_and_without_verifier() {
        let with = RoleRecord {
            name: "alice".into(),
            is_superuser: false,
            verifier: Some(ScramVerifier::from_password_with("pw", vec![1u8; 16], 4096)),
        };
        let back = RoleRecord::from_bytes(&with.to_bytes()).unwrap();
        assert_eq!(with, back);

        let without = RoleRecord {
            name: "admin".into(),
            is_superuser: true,
            verifier: None,
        };
        let back = RoleRecord::from_bytes(&without.to_bytes()).unwrap();
        assert_eq!(without, back);
    }

    #[test]
    fn put_get_drop_role() {
        let store = AuthStore::new(test_engine());
        assert!(!store.any_role_exists());
        assert!(store.get_role("alice").is_none());

        let rec = RoleRecord {
            name: "alice".into(),
            is_superuser: false,
            verifier: Some(ScramVerifier::from_password_with("alice", vec![2u8; 16], 4096)),
        };
        store.put_role(&rec).unwrap();
        assert!(store.any_role_exists());
        assert_eq!(store.get_role("alice").unwrap(), rec);
        assert!(store.verifier_for("alice").is_some());
        assert!(!store.is_superuser("alice"));

        assert!(store.drop_role("alice").unwrap());
        assert!(store.get_role("alice").is_none());
        assert!(!store.drop_role("alice").unwrap(), "second drop is false");
    }

    #[test]
    fn grant_check_revoke() {
        let store = AuthStore::new(test_engine());
        assert!(!store.has_grant("alice", "docs", Action::Select));

        store.grant("alice", "docs", Action::Select).unwrap();
        store.grant("alice", "docs", Action::Insert).unwrap();
        assert!(store.has_grant("alice", "docs", Action::Select));
        assert!(store.has_grant("alice", "docs", Action::Insert));
        assert!(!store.has_grant("alice", "docs", Action::Delete));

        let grants = store.list_grants();
        assert_eq!(grants.len(), 2);
        assert!(grants.iter().any(|(r, t, p)| r == "alice" && t == "docs" && *p == Action::Select));

        store.revoke("alice", "docs", Action::Select).unwrap();
        assert!(!store.has_grant("alice", "docs", Action::Select));
        assert!(store.has_grant("alice", "docs", Action::Insert));
        // Revoking a non-existent grant is a no-op success.
        store.revoke("alice", "docs", Action::Delete).unwrap();
    }

    #[test]
    fn drop_role_cascades_grants() {
        let store = AuthStore::new(test_engine());
        store.grant("bob", "t1", Action::Select).unwrap();
        store.grant("bob", "t2", Action::Update).unwrap();
        store.put_role(&RoleRecord { name: "bob".into(), is_superuser: false, verifier: None }).unwrap();

        store.drop_role("bob").unwrap();
        assert!(!store.has_grant("bob", "t1", Action::Select));
        assert!(!store.has_grant("bob", "t2", Action::Update));
    }

    #[test]
    fn roles_survive_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_path_buf();

        {
            let config = EngineConfig {
                data_dir: path.clone(),
                ..Default::default()
            };
            let engine = Arc::new(Engine::new(config).unwrap());
            let store = AuthStore::new(engine.clone());
            store.put_role(&RoleRecord {
                name: "persisted".into(),
                is_superuser: true,
                verifier: Some(ScramVerifier::from_password_with("pw", vec![9u8; 16], 4096)),
            }).unwrap();
            store.grant("persisted", "docs", Action::Select).unwrap();
            engine.shutdown();
        }

        // Reopen against the same data dir: WAL replay must restore the role.
        let config = EngineConfig {
            data_dir: path,
            ..Default::default()
        };
        let engine = Arc::new(Engine::new(config).unwrap());
        let store = AuthStore::new(engine);
        let role = store.get_role("persisted").expect("role survives restart");
        assert!(role.is_superuser);
        assert!(role.verifier.is_some());
        assert!(store.has_grant("persisted", "docs", Action::Select));
    }
}
