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
}
