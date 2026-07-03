//! Task 16.6 — CI assertion: the OSS workspace dependency graph contains no
//! `galaxdb-enterprise*` crate (Requirement 13 AC2/AC3).
//!
//! This is the belt to `cargo deny check bans`' braces: `deny.toml` bans the
//! enterprise crate names, and this test independently parses the resolved
//! workspace `Cargo.lock` and fails if any package whose name starts with
//! `galaxdb-enterprise` has entered the graph. It runs as part of the normal
//! `cargo test` suite, so an accidental OSS→ENT edge fails fast in CI even if
//! cargo-deny is not invoked.

use std::path::PathBuf;

/// Resolve the workspace `Cargo.lock` from this crate's manifest dir
/// (`crates/galaxdb-cluster`) by walking up to the workspace root.
fn workspace_lockfile() -> PathBuf {
    // CARGO_MANIFEST_DIR = <workspace>/crates/galaxdb-cluster
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent() // crates/
        .and_then(|p| p.parent()) // <workspace>
        .expect("galaxdb-cluster must live two levels below the workspace root");
    workspace_root.join("Cargo.lock")
}

#[test]
fn oss_graph_has_no_enterprise_crate() {
    let lock = workspace_lockfile();
    let contents = std::fs::read_to_string(&lock)
        .unwrap_or_else(|e| panic!("cannot read workspace lockfile {}: {e}", lock.display()));

    // Cargo.lock lists each package as a `name = "..."` line under `[[package]]`.
    let offenders: Vec<&str> = contents
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            let rest = trimmed.strip_prefix("name = \"")?;
            let name = rest.strip_suffix('"')?;
            if name.starts_with("galaxdb-enterprise") {
                Some(name)
            } else {
                None
            }
        })
        .collect();

    assert!(
        offenders.is_empty(),
        "the open-source dependency graph must not contain any galaxdb-enterprise* crate, \
         but Cargo.lock lists: {offenders:?}. The OSS core never depends on enterprise code \
         (Requirement 13 AC2/AC3)."
    );
}
