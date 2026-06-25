//! DataFusion containment guard (HTAP spec Req 7.1, Property 6).
//!
//! The single hard rule of the anti-corruption boundary: **no crate other
//! than `galaxdb-query` may depend on or reference DataFusion.** This test
//! walks the workspace and fails if any other crate's `Cargo.toml` declares
//! a `datafusion*` dependency, or if any other crate's Rust source contains
//! a `datafusion::` / `use datafusion` reference.
//!
//! Mirrors the project's existing no-mocks grep guard. It runs in CI on
//! every change, so a regression that leaks DataFusion out of this crate
//! breaks the build immediately rather than at some later integration point.

use std::fs;
use std::path::{Path, PathBuf};

/// Crate directory names allowed to reference DataFusion.
const ALLOWED_CRATES: &[&str] = &["galaxdb-query"];

/// Walk up from this test file to the workspace root (the dir holding the
/// top-level `Cargo.toml` with `[workspace]`).
fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")); // crates/galaxdb-query
    loop {
        let manifest = dir.join("Cargo.toml");
        if manifest.exists() {
            if let Ok(text) = fs::read_to_string(&manifest) {
                if text.contains("[workspace]") {
                    return dir;
                }
            }
        }
        if !dir.pop() {
            panic!("could not locate workspace root from CARGO_MANIFEST_DIR");
        }
    }
}

/// Is `path` inside one of the allowed crate directories?
fn is_allowed(path: &Path) -> bool {
    path.components().any(|c| {
        c.as_os_str()
            .to_str()
            .map(|s| ALLOWED_CRATES.contains(&s))
            .unwrap_or(false)
    })
}

/// Should this directory be skipped entirely (build output, vcs, vendored)?
fn is_skippable_dir(name: &str) -> bool {
    matches!(name, "target" | ".git" | ".kiro" | "node_modules" | ".venv")
}

fn collect_files(dir: &Path, ext: &str, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !is_skippable_dir(name) {
                collect_files(&path, ext, out);
            }
        } else if path.extension().and_then(|e| e.to_str()) == Some(ext) {
            out.push(path);
        }
    }
}

#[test]
fn no_datafusion_dependency_outside_galaxdb_query() {
    let root = workspace_root();
    let mut manifests = Vec::new();
    collect_files(&root, "toml", &mut manifests);

    let mut violations = Vec::new();
    for manifest in manifests {
        if manifest.file_name().and_then(|n| n.to_str()) != Some("Cargo.toml") {
            continue;
        }
        if is_allowed(&manifest) {
            continue;
        }
        let Ok(text) = fs::read_to_string(&manifest) else {
            continue;
        };
        for (lineno, line) in text.lines().enumerate() {
            let trimmed = line.trim_start();
            // Skip comments so the documented "NOT yet a dependency" note in
            // galaxdb-query and any explanatory comments never trip the guard.
            if trimmed.starts_with('#') {
                continue;
            }
            // A dependency declaration starts with `datafusion` at the start
            // of a line (e.g. `datafusion = ...` or `datafusion-foo = ...`).
            if trimmed.starts_with("datafusion") {
                violations.push(format!(
                    "{}:{}: {}",
                    manifest.display(),
                    lineno + 1,
                    line.trim()
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "DataFusion dependency found outside galaxdb-query (Req 7.1):\n{}",
        violations.join("\n")
    );
}

#[test]
fn no_datafusion_reference_in_other_crate_sources() {
    let root = workspace_root();
    let crates_dir = root.join("crates");
    let mut sources = Vec::new();
    collect_files(&crates_dir, "rs", &mut sources);

    let mut violations = Vec::new();
    for src in sources {
        if is_allowed(&src) {
            continue;
        }
        let Ok(text) = fs::read_to_string(&src) else {
            continue;
        };
        for (lineno, line) in text.lines().enumerate() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") || trimmed.starts_with('*') {
                continue; // doc/comment lines may legitimately say "datafusion"
            }
            if line.contains("datafusion::") || line.contains("use datafusion") {
                violations.push(format!("{}:{}: {}", src.display(), lineno + 1, line.trim()));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "DataFusion reference found in non-galaxdb-query source (Req 7.1):\n{}",
        violations.join("\n")
    );
}
