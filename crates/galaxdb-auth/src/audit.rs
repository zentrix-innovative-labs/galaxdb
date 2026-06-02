//! Audit seam.
//!
//! [`AuditSink`] records security-relevant events (authentication
//! attempts, authorization decisions, role/grant changes). The open core
//! bundles [`NoOpAuditSink`] (default) and [`FileAuditSink`] (appends
//! JSON lines to a file). The enterprise edition adds a tamper-evident,
//! hash-chained sink behind this same trait.

use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;

/// Whether the audited action was permitted or denied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AuditOutcome {
    /// The action was allowed.
    Allowed,
    /// The action was denied.
    Denied,
}

/// A single security-relevant event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Event kind, e.g. `"auth"`, `"authz"`, `"role_change"`.
    pub kind: String,
    /// The role involved, if known (the role name).
    pub role: Option<String>,
    /// The action attempted, e.g. `"select"`, `"login"`.
    pub action: String,
    /// The object targeted, e.g. `"table:docs"`.
    pub object: Option<String>,
    /// Whether the action was allowed or denied.
    pub outcome: AuditOutcome,
    /// Optional human-readable detail.
    pub detail: Option<String>,
}

impl AuditEvent {
    /// Construct an event with the required fields; optional fields empty.
    pub fn new(kind: impl Into<String>, action: impl Into<String>, outcome: AuditOutcome) -> Self {
        AuditEvent {
            kind: kind.into(),
            role: None,
            action: action.into(),
            object: None,
            outcome,
            detail: None,
        }
    }

    /// Set the role.
    pub fn with_role(mut self, role: impl Into<String>) -> Self {
        self.role = Some(role.into());
        self
    }

    /// Set the object.
    pub fn with_object(mut self, object: impl Into<String>) -> Self {
        self.object = Some(object.into());
        self
    }

    /// Set the detail.
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }
}

/// Records security events. The engine selects one implementation at
/// startup via [`crate::SecurityProviders`].
pub trait AuditSink: Send + Sync {
    /// Record one event. Implementations must not panic; an I/O failure
    /// should be handled internally (logged) rather than propagated into
    /// the request path.
    fn record(&self, event: &AuditEvent);

    /// A short, stable name for logging.
    fn name(&self) -> &str;
}

/// Discards every event. The default when no audit is configured.
///
/// This is intentionally a no-op and is named as such — it is the
/// explicit "auditing disabled" choice, not a placeholder standing in for
/// a real sink.
#[derive(Debug, Default)]
pub struct NoOpAuditSink;

impl AuditSink for NoOpAuditSink {
    fn record(&self, _event: &AuditEvent) {}

    fn name(&self) -> &str {
        "noop"
    }
}

/// Appends each event as a JSON line to a file (JSONL). Real local audit
/// for self-hosted open-source deployments. The enterprise tamper-evident
/// sink (hash-chained, exportable) is a separate implementation of the
/// same trait.
pub struct FileAuditSink {
    file: Mutex<File>,
    path: String,
}

impl FileAuditSink {
    /// Open (creating if needed, appending if present) an audit file.
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path_ref = path.as_ref();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path_ref)?;
        Ok(FileAuditSink {
            file: Mutex::new(file),
            path: path_ref.display().to_string(),
        })
    }
}

impl AuditSink for FileAuditSink {
    fn record(&self, event: &AuditEvent) {
        // Serialize to a single JSON line. On failure, log rather than
        // propagate — auditing must never break the request path, but a
        // failure to audit is itself noteworthy.
        let line = match serde_json::to_string(event) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!(error = %e, "failed to serialize audit event");
                return;
            }
        };
        let mut guard = match self.file.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Err(e) = writeln!(guard, "{line}") {
            tracing::error!(error = %e, path = %self.path, "failed to write audit event");
        }
    }

    fn name(&self) -> &str {
        "file"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn audit_event_builder_sets_fields() {
        let e = AuditEvent::new("authz", "select", AuditOutcome::Denied)
            .with_role("alice")
            .with_object("table:docs")
            .with_detail("no grant");
        assert_eq!(e.kind, "authz");
        assert_eq!(e.role.as_deref(), Some("alice"));
        assert_eq!(e.object.as_deref(), Some("table:docs"));
        assert_eq!(e.outcome, AuditOutcome::Denied);
        assert_eq!(e.detail.as_deref(), Some("no grant"));
    }

    #[test]
    fn noop_sink_discards() {
        let sink = NoOpAuditSink;
        // Just must not panic.
        sink.record(&AuditEvent::new("auth", "login", AuditOutcome::Allowed));
        assert_eq!(sink.name(), "noop");
    }

    #[test]
    fn file_sink_appends_json_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");
        let sink = FileAuditSink::open(&path).unwrap();
        sink.record(
            &AuditEvent::new("auth", "login", AuditOutcome::Allowed).with_role("admin"),
        );
        sink.record(
            &AuditEvent::new("authz", "delete", AuditOutcome::Denied)
                .with_role("alice")
                .with_object("table:docs"),
        );
        drop(sink);

        let mut contents = String::new();
        File::open(&path)
            .unwrap()
            .read_to_string(&mut contents)
            .unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 2);

        // Each line round-trips as an AuditEvent.
        let first: AuditEvent = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(first.action, "login");
        assert_eq!(first.outcome, AuditOutcome::Allowed);
        let second: AuditEvent = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(second.action, "delete");
        assert_eq!(second.outcome, AuditOutcome::Denied);
        assert_eq!(second.role.as_deref(), Some("alice"));
    }

    #[test]
    fn file_sink_appends_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");
        {
            let sink = FileAuditSink::open(&path).unwrap();
            sink.record(&AuditEvent::new("auth", "login", AuditOutcome::Allowed));
        }
        {
            let sink = FileAuditSink::open(&path).unwrap();
            sink.record(&AuditEvent::new("auth", "logout", AuditOutcome::Allowed));
        }
        let mut contents = String::new();
        File::open(&path)
            .unwrap()
            .read_to_string(&mut contents)
            .unwrap();
        assert_eq!(contents.lines().count(), 2, "second open must append, not truncate");
    }
}
