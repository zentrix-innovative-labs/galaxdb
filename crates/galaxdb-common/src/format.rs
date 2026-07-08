//! On-disk format versioning (v0.5 Workstream B).
//!
//! Every persistent artifact (WAL, SST, PAX block, blob log, catalog, vector index) carries a
//! small, explicit header so a binary knows *exactly* what layout it is reading and never
//! guesses from byte patterns. This module owns:
//!
//! - a per-artifact **magic** + **format version** header ([`FormatHeader`]) with
//!   read/write/byte helpers;
//! - the informational **writer engine version** stamped into new files ([`WriterVersion`]);
//! - the supported **version range** per artifact ([`FormatSupport`]) and the single
//!   [`check_version`] gate that turns an out-of-range version into a typed
//!   [`GalaxError::FormatTooOld`] / [`GalaxError::FormatTooNew`];
//! - the engine-wide [`FORMAT_VERSION`], tracked **separately from the crate/semver version**
//!   so a release can be classified *patch* (format unchanged) vs *minor/major* (format bumped).
//!
//! ## Why it lives in `galaxdb-common`
//!
//! The versioned artifacts span three crates — `galaxdb-storage` (WAL/SST/PAX/blob),
//! `galaxdb-sql` (catalog), and `galaxdb-vector` (HNSW index). Putting the shared header +
//! range table here (the foundation crate everyone already depends on) lets all of them use
//! one implementation with no dependency cycle. (The design sketch suggested
//! `galaxdb-storage/src/format.rs`; common is the cycle-free home for the shared pieces.)
//!
//! ## Header wire layout (fixed 16 bytes, little-endian)
//!
//! ```text
//! ┌────────────┬───────────────┬───────────────────────────┬──────────┐
//! │ magic [4]  │ format_ver u16│ writer maj/min/patch 3×u16│ resv [2] │
//! └────────────┴───────────────┴───────────────────────────┴──────────┘
//!   0..4          4..6            6..12                       12..14  (14..16 = 0)
//! ```
//!
//! The layout itself is format-version-independent: a reader can always parse the header and
//! then dispatch on `format_version`. `writer_version` is informational (diagnostics + the
//! release-classification signal Cloud consumes); it is never used for compatibility decisions.

use std::io::{self, Read, Write};

use crate::error::{GalaxError, GalaxResult};

/// Engine-wide current on-disk format version.
///
/// Bump this **only** when a persisted layout actually changes. It is deliberately independent
/// of the crate/semver version so a semver patch can be recognized as format-compatible.
pub const FORMAT_VERSION: u16 = 1;

/// Serialized size of a [`FormatHeader`], in bytes.
pub const FORMAT_HEADER_SIZE: usize = 16;

/// The writer engine's semantic version, stamped into newly written headers (informational).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WriterVersion {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
}

impl WriterVersion {
    /// The version of the engine build that is running now, parsed from the crate's
    /// `CARGO_PKG_VERSION`. Non-numeric pre-release/build suffixes are ignored.
    pub fn current() -> Self {
        Self::parse(env!("CARGO_PKG_VERSION"))
    }

    /// Parse `"MAJOR.MINOR.PATCH"` leniently; missing/garbage components become 0.
    pub fn parse(s: &str) -> Self {
        let mut it = s.split('.');
        let major = it.next().and_then(parse_leading_u16).unwrap_or(0);
        let minor = it.next().and_then(parse_leading_u16).unwrap_or(0);
        let patch = it.next().and_then(parse_leading_u16).unwrap_or(0);
        Self {
            major,
            minor,
            patch,
        }
    }
}

/// Parse the leading run of ASCII digits (so `"0-rc1"` → `0`, `"12+meta"` → `12`).
fn parse_leading_u16(s: &str) -> Option<u16> {
    let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

/// A parsed on-disk format header: which artifact family (`magic`), the layout version, and
/// the engine version that wrote it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatHeader {
    pub magic: [u8; 4],
    pub format_version: u16,
    pub writer_version: WriterVersion,
}

impl FormatHeader {
    /// Build a header for a fresh write: the given magic + format version, stamped with the
    /// current engine version.
    pub fn new(magic: [u8; 4], format_version: u16) -> Self {
        Self {
            magic,
            format_version,
            writer_version: WriterVersion::current(),
        }
    }

    /// Serialize to the fixed 16-byte layout.
    pub fn to_bytes(&self) -> [u8; FORMAT_HEADER_SIZE] {
        let mut b = [0u8; FORMAT_HEADER_SIZE];
        b[0..4].copy_from_slice(&self.magic);
        b[4..6].copy_from_slice(&self.format_version.to_le_bytes());
        b[6..8].copy_from_slice(&self.writer_version.major.to_le_bytes());
        b[8..10].copy_from_slice(&self.writer_version.minor.to_le_bytes());
        b[10..12].copy_from_slice(&self.writer_version.patch.to_le_bytes());
        // b[12..16] reserved (zero).
        b
    }

    /// Parse from the fixed 16-byte layout. Verifies the magic matches `expected_magic`
    /// (returns [`GalaxError::InvalidMagic`] on mismatch). Does **not** range-check the
    /// version — call [`check_version`] with the artifact's [`FormatSupport`] for that.
    pub fn from_bytes(bytes: &[u8; FORMAT_HEADER_SIZE], expected_magic: [u8; 4]) -> GalaxResult<Self> {
        let magic = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if magic != expected_magic {
            return Err(GalaxError::InvalidMagic(u32::from_le_bytes(magic)));
        }
        let format_version = u16::from_le_bytes([bytes[4], bytes[5]]);
        let writer_version = WriterVersion {
            major: u16::from_le_bytes([bytes[6], bytes[7]]),
            minor: u16::from_le_bytes([bytes[8], bytes[9]]),
            patch: u16::from_le_bytes([bytes[10], bytes[11]]),
        };
        Ok(Self {
            magic,
            format_version,
            writer_version,
        })
    }

    /// Write the header to a writer.
    pub fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_all(&self.to_bytes())
    }

    /// Read and parse a header from a reader, verifying the magic.
    pub fn read_from<R: Read>(r: &mut R, expected_magic: [u8; 4]) -> GalaxResult<Self> {
        let mut buf = [0u8; FORMAT_HEADER_SIZE];
        r.read_exact(&mut buf).map_err(GalaxError::Io)?;
        Self::from_bytes(&buf, expected_magic)
    }
}

/// The supported format-version range for one artifact family.
///
/// `min_readable` ..= `current_write`: the engine reads any version in this inclusive range and
/// writes `current_write`. A version below `min_readable` is [`GalaxError::FormatTooOld`]; a
/// version above `current_write` is [`GalaxError::FormatTooNew`] (the rollback-safety refusal).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatSupport {
    /// Human-readable artifact name used in error messages.
    pub artifact: &'static str,
    /// 4-byte magic identifying this artifact family on disk.
    pub magic: [u8; 4],
    /// Oldest format version this engine can read.
    pub min_readable: u16,
    /// Newest format version this engine reads and the version it writes.
    pub current_write: u16,
}

impl FormatSupport {
    /// Classify `found` against this range, returning a typed error when out of range.
    ///
    /// - `found < min_readable` → [`GalaxError::FormatTooOld`]
    /// - `found > current_write` → [`GalaxError::FormatTooNew`] (refuse; never best-effort read)
    /// - otherwise → `Ok(())`
    pub fn check(&self, found: u16) -> GalaxResult<()> {
        if found < self.min_readable {
            Err(GalaxError::FormatTooOld {
                artifact: self.artifact,
                found,
                min_readable: self.min_readable,
            })
        } else if found > self.current_write {
            Err(GalaxError::FormatTooNew {
                artifact: self.artifact,
                found,
                current: self.current_write,
            })
        } else {
            Ok(())
        }
    }

    /// Build a fresh write header for this artifact at `current_write`.
    pub fn header(&self) -> FormatHeader {
        FormatHeader::new(self.magic, self.current_write)
    }
}

/// Free-function form of [`FormatSupport::check`] for call sites that hold the parts separately.
pub fn check_version(support: &FormatSupport, found: u16) -> GalaxResult<()> {
    support.check(found)
}

// ---------------------------------------------------------------------------
// Per-artifact support table.
//
// Magic bytes are ASCII tags read left-to-right on disk. Where an artifact already had an
// ad-hoc magic before v0.5, B.3 reconciles the reader; these constants are the single source
// of truth going forward. All artifacts start at format v1 (== FORMAT_VERSION); min_readable
// is 1 until a real N-1 format exists.
// ---------------------------------------------------------------------------

/// Write-ahead log file superblock.
pub const WAL: FormatSupport = FormatSupport {
    artifact: "WAL",
    magic: *b"GWAL",
    min_readable: 1,
    current_write: 1,
};

/// SST (sorted string table) file.
pub const SST: FormatSupport = FormatSupport {
    artifact: "SST",
    magic: *b"GSST",
    min_readable: 1,
    current_write: 1,
};

/// PAX columnar block.
pub const PAX: FormatSupport = FormatSupport {
    artifact: "PAX block",
    magic: *b"GPAX",
    min_readable: 1,
    current_write: 1,
};

/// Blob log (KV-separated large values).
pub const BLOB: FormatSupport = FormatSupport {
    artifact: "blob log",
    magic: *b"GBLB",
    min_readable: 1,
    current_write: 1,
};

/// Table catalog entry.
pub const CATALOG: FormatSupport = FormatSupport {
    artifact: "catalog",
    magic: *b"GCAT",
    min_readable: 1,
    current_write: 1,
};

/// HNSW vector-index file.
pub const HNSW: FormatSupport = FormatSupport {
    artifact: "HNSW index",
    magic: *b"GHNS",
    min_readable: 1,
    current_write: 1,
};

// ---------------------------------------------------------------------------
// Crash-safe upgrade-on-open migration (B.5).
//
// When an on-disk artifact is at an older-but-supported format and the engine
// wants it at the current format, it is migrated crash-safely: write a new file
// alongside → fsync it → atomically rename over the target → fsync the
// directory. A crash before the rename leaves the original fully intact; after
// the rename the target is fully the new contents. There is never a torn file,
// so a rollback (previous binary + restored snapshot) is always safe.
// ---------------------------------------------------------------------------

use std::path::Path;

/// Crash-safe, atomic file replacement.
///
/// Writes `new_contents` to a temporary sibling of `target`, fsyncs it, renames
/// it over `target` (atomic on POSIX and Windows-with-ReplaceFile semantics via
/// `std::fs::rename`), then fsyncs the parent directory so the rename survives a
/// power loss. On any crash: before the rename the original `target` is
/// untouched; after it, `target` is exactly `new_contents`.
pub fn atomic_replace(target: &Path, new_contents: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let dir = target.parent().filter(|p| !p.as_os_str().is_empty());
    let dir = dir.unwrap_or_else(|| Path::new("."));
    let stem = target
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "artifact".to_string());
    // Unique temp name (pid + monotonic counter) so concurrent replacers of
    // different targets never collide.
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp = dir.join(format!(".{stem}.tmp-{}-{n}", std::process::id()));

    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(new_contents)?;
        f.sync_all()?;
    }
    // Atomic swap. On failure, clean up the temp so it doesn't linger.
    if let Err(e) = std::fs::rename(&tmp, target) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    // Make the rename durable. Directory fsync is best-effort on platforms that
    // don't support it; the rename atomicity guarantee still holds.
    if let Ok(dir_file) = std::fs::File::open(dir) {
        let _ = dir_file.sync_all();
    }
    Ok(())
}

/// Upgrade a [`FormatHeader`]-prefixed file to the current write version if it
/// is at an older-but-supported version, crash-safely (via [`atomic_replace`]).
///
/// - version `> current_write` → [`GalaxError::FormatTooNew`] (refuse).
/// - version `< min_readable` → [`GalaxError::FormatTooOld`].
/// - `min_readable ..< current_write` → run `migrate(old_version, full_bytes)`
///   to produce the new file bytes (which must begin with a current-version
///   header) and atomically install them; returns `Ok(true)`.
/// - version `== current_write` → no-op, `Ok(false)`.
pub fn upgrade_on_open<F>(
    target: &Path,
    support: &FormatSupport,
    migrate: F,
) -> GalaxResult<bool>
where
    F: FnOnce(u16, &[u8]) -> GalaxResult<Vec<u8>>,
{
    let bytes = std::fs::read(target).map_err(GalaxError::Io)?;
    if bytes.len() < FORMAT_HEADER_SIZE {
        return Err(GalaxError::Internal(format!(
            "{} file too small for a format header",
            support.artifact
        )));
    }
    let mut hdr = [0u8; FORMAT_HEADER_SIZE];
    hdr.copy_from_slice(&bytes[..FORMAT_HEADER_SIZE]);
    let header = FormatHeader::from_bytes(&hdr, support.magic)?;
    support.check(header.format_version)?; // typed too-old / too-new

    if header.format_version < support.current_write {
        let new_bytes = migrate(header.format_version, &bytes)?;
        atomic_replace(target, &new_bytes).map_err(GalaxError::Io)?;
        Ok(true)
    } else {
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_roundtrips_through_bytes() {
        let h = FormatHeader::new(*b"GSST", 1);
        let bytes = h.to_bytes();
        assert_eq!(bytes.len(), FORMAT_HEADER_SIZE);
        let parsed = FormatHeader::from_bytes(&bytes, *b"GSST").unwrap();
        assert_eq!(parsed, h);
        assert_eq!(parsed.format_version, 1);
    }

    #[test]
    fn header_roundtrips_through_reader_writer() {
        let h = FormatHeader::new(*b"GPAX", 7);
        let mut buf = Vec::new();
        h.write_to(&mut buf).unwrap();
        let mut cursor = std::io::Cursor::new(&buf);
        let parsed = FormatHeader::read_from(&mut cursor, *b"GPAX").unwrap();
        assert_eq!(parsed, h);
    }

    #[test]
    fn wrong_magic_is_rejected() {
        let bytes = FormatHeader::new(*b"GSST", 1).to_bytes();
        let err = FormatHeader::from_bytes(&bytes, *b"GPAX").unwrap_err();
        assert!(matches!(err, GalaxError::InvalidMagic(_)));
    }

    #[test]
    fn byte_layout_is_stable() {
        // Pin the exact on-disk bytes so a future refactor can't silently shift fields.
        let h = FormatHeader {
            magic: *b"GWAL",
            format_version: 0x0102,
            writer_version: WriterVersion {
                major: 0x0304,
                minor: 0x0506,
                patch: 0x0708,
            },
        };
        let b = h.to_bytes();
        assert_eq!(&b[0..4], b"GWAL");
        assert_eq!(&b[4..6], &[0x02, 0x01]); // 0x0102 LE
        assert_eq!(&b[6..8], &[0x04, 0x03]);
        assert_eq!(&b[8..10], &[0x06, 0x05]);
        assert_eq!(&b[10..12], &[0x08, 0x07]);
        assert_eq!(&b[12..16], &[0, 0, 0, 0]);
    }

    #[test]
    fn version_range_check_classifies_correctly() {
        let support = FormatSupport {
            artifact: "test",
            magic: *b"GTST",
            min_readable: 2,
            current_write: 4,
        };
        // In range.
        assert!(support.check(2).is_ok());
        assert!(support.check(3).is_ok());
        assert!(support.check(4).is_ok());
        // Too old.
        match support.check(1) {
            Err(GalaxError::FormatTooOld {
                found, min_readable, ..
            }) => {
                assert_eq!(found, 1);
                assert_eq!(min_readable, 2);
            }
            other => panic!("expected FormatTooOld, got {other:?}"),
        }
        // Too new (rollback-safety refusal).
        match support.check(5) {
            Err(GalaxError::FormatTooNew { found, current, .. }) => {
                assert_eq!(found, 5);
                assert_eq!(current, 4);
            }
            other => panic!("expected FormatTooNew, got {other:?}"),
        }
    }

    #[test]
    fn current_writer_version_parses() {
        // Whatever the crate version is, it parses without panicking and major is sane.
        let v = WriterVersion::current();
        // 0.x today; just assert the parse produced *some* structured value.
        let _ = (v.major, v.minor, v.patch);
        assert_eq!(WriterVersion::parse("1.2.3"), WriterVersion { major: 1, minor: 2, patch: 3 });
        assert_eq!(WriterVersion::parse("0.4.0-rc1"), WriterVersion { major: 0, minor: 4, patch: 0 });
        assert_eq!(WriterVersion::parse("7"), WriterVersion { major: 7, minor: 0, patch: 0 });
    }

    #[test]
    fn all_launch_artifacts_start_at_v1() {
        for s in [WAL, SST, PAX, BLOB, CATALOG, HNSW] {
            assert_eq!(s.min_readable, 1, "{}", s.artifact);
            assert_eq!(s.current_write, FORMAT_VERSION, "{}", s.artifact);
            assert!(s.check(FORMAT_VERSION).is_ok(), "{}", s.artifact);
        }
    }

    // ── B.5: crash-safe upgrade-on-open ───────────────────────────────────

    /// A synthetic artifact used only to exercise the migration mechanism with
    /// a real older→current version transform (production artifacts are all at
    /// v1 today, so there is nothing real to upgrade yet). #[cfg(test)] only.
    const SYNTH_V2: FormatSupport = FormatSupport {
        artifact: "synthetic",
        magic: *b"GTST",
        min_readable: 1,
        current_write: 2,
    };

    fn write_synth(path: &std::path::Path, version: u16, body: &[u8]) {
        let mut bytes = FormatHeader::new(*b"GTST", version).to_bytes().to_vec();
        bytes.extend_from_slice(body);
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn atomic_replace_swaps_contents() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("artifact.bin");
        std::fs::write(&path, b"OLD-CONTENTS").unwrap();

        atomic_replace(&path, b"NEW-CONTENTS-LONGER").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"NEW-CONTENTS-LONGER");

        // No stray temp files left behind in the directory.
        let strays: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp-"))
            .collect();
        assert!(strays.is_empty(), "temp file left behind: {strays:?}");
    }

    /// Crash BEFORE the rename: the original file must be fully intact and
    /// readable (recoverability). We simulate the pre-rename state by writing a
    /// temp sibling ourselves and confirming the target is untouched.
    #[test]
    fn crash_before_rename_leaves_original_intact() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("artifact.bin");
        std::fs::write(&path, b"ORIGINAL").unwrap();

        // Simulate: new contents staged in a temp, then the process dies.
        let tmp = dir.path().join(".artifact.bin.tmp-crashsim");
        std::fs::write(&tmp, b"HALF-WRITTEN-NEW").unwrap();
        // (no rename happened)

        // The target still holds the original, fully readable.
        assert_eq!(std::fs::read(&path).unwrap(), b"ORIGINAL");

        // A subsequent successful replace still works and wins.
        atomic_replace(&path, b"FINAL").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"FINAL");
    }

    /// Crash AFTER the rename: the target is fully the new contents (no torn
    /// state). `atomic_replace` returning Ok is exactly this post-rename state.
    #[test]
    fn crash_after_rename_shows_new_contents() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("artifact.bin");
        std::fs::write(&path, b"v1-bytes").unwrap();
        atomic_replace(&path, b"v2-bytes").unwrap();
        // Post-rename (== "crash right after rename"): fully new, never torn.
        assert_eq!(std::fs::read(&path).unwrap(), b"v2-bytes");
    }

    #[test]
    fn upgrade_on_open_migrates_older_to_current() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("synth.bin");
        write_synth(&path, 1, b"payload-v1");

        // Migrate v1 → v2: bump the header, keep the payload (contrived but real
        // transform). Returns Ok(true) since a migration happened.
        let migrated = upgrade_on_open(&path, &SYNTH_V2, |old, bytes| {
            assert_eq!(old, 1);
            let body = &bytes[FORMAT_HEADER_SIZE..];
            let mut out = FormatHeader::new(*b"GTST", 2).to_bytes().to_vec();
            out.extend_from_slice(body);
            Ok(out)
        })
        .unwrap();
        assert!(migrated);

        // The on-disk header is now v2 and the payload survived.
        let bytes = std::fs::read(&path).unwrap();
        let mut hdr = [0u8; FORMAT_HEADER_SIZE];
        hdr.copy_from_slice(&bytes[..FORMAT_HEADER_SIZE]);
        let header = FormatHeader::from_bytes(&hdr, *b"GTST").unwrap();
        assert_eq!(header.format_version, 2);
        assert_eq!(&bytes[FORMAT_HEADER_SIZE..], b"payload-v1");

        // Re-running is now a no-op (already current).
        let again = upgrade_on_open(&path, &SYNTH_V2, |_, _| panic!("should not migrate")).unwrap();
        assert!(!again);
    }

    #[test]
    fn upgrade_on_open_refuses_newer_format() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("synth.bin");
        write_synth(&path, 3, b"from-the-future"); // > current_write (2)

        match upgrade_on_open(&path, &SYNTH_V2, |_, _| panic!("must not migrate")) {
            Err(GalaxError::FormatTooNew { found, current, .. }) => {
                assert_eq!(found, 3);
                assert_eq!(current, 2);
            }
            other => panic!("expected FormatTooNew, got {other:?}"),
        }
        // The file is left untouched on refusal.
        let bytes = std::fs::read(&path).unwrap();
        let mut hdr = [0u8; FORMAT_HEADER_SIZE];
        hdr.copy_from_slice(&bytes[..FORMAT_HEADER_SIZE]);
        assert_eq!(
            FormatHeader::from_bytes(&hdr, *b"GTST").unwrap().format_version,
            3
        );
    }
}
