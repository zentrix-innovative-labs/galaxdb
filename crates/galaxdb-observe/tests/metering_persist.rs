//! v0.6 E-4 metering: counter persistence (M.6).
//!
//! One `#[test]` in its own binary so the process-global counters are touched
//! only here (delta assertions can't be polluted by a parallel test). Proves:
//! absent file is a no-op, flush writes the exact current totals, load seeds
//! the live counters by the persisted amount (== "resume from persisted totals"
//! on a fresh-process restart), and a too-new file is refused with a typed
//! error rather than misread.

use galaxdb_common::format::FORMAT_HEADER_SIZE;
use galaxdb_observe::{flush_metering, load_metering, metrics, METERING, METERING_FILE};

fn decode_payload(bytes: &[u8], i: usize) -> u64 {
    let payload = &bytes[FORMAT_HEADER_SIZE..];
    let mut b = [0u8; 8];
    b.copy_from_slice(&payload[i * 8..i * 8 + 8]);
    u64::from_le_bytes(b)
}

#[test]
fn metering_persistence_roundtrip_and_refusal() {
    let m = metrics();

    // (1) Absent file → no-op; counters unchanged.
    let empty = tempfile::tempdir().unwrap();
    let before = m.read_ops_total.get();
    load_metering(empty.path()).unwrap();
    assert_eq!(
        m.read_ops_total.get(),
        before,
        "absent metering file must not change counters"
    );

    // (2) Flush writes the exact current cumulative totals.
    m.read_ops_total.inc_by(7);
    m.write_ops_total.inc_by(5);
    m.vector_ops_total.inc_by(3);
    m.embedding_ops_total.inc_by(11);
    m.near_dedup_rows_total.inc_by(13);
    m.training_export_bytes_total.inc_by(1024);

    let dir = tempfile::tempdir().unwrap();
    flush_metering(dir.path()).unwrap();
    let bytes = std::fs::read(dir.path().join(METERING_FILE)).unwrap();
    assert_eq!(decode_payload(&bytes, 0), m.read_ops_total.get());
    assert_eq!(decode_payload(&bytes, 1), m.write_ops_total.get());
    assert_eq!(decode_payload(&bytes, 2), m.vector_ops_total.get());
    assert_eq!(decode_payload(&bytes, 3), m.embedding_ops_total.get());
    assert_eq!(decode_payload(&bytes, 4), m.near_dedup_rows_total.get());
    assert_eq!(decode_payload(&bytes, 5), m.training_export_bytes_total.get());

    // (3) Load seeds the live counters by the persisted amount. On a real
    // fresh-process restart the counters start at 0, so after load they equal
    // the persisted totals exactly — "resume from persisted totals" (Req 5.2 /
    // Property 4). Here we assert the delta equals the persisted value.
    let persisted_r = decode_payload(&bytes, 0);
    let persisted_w = decode_payload(&bytes, 1);
    let r_before = m.read_ops_total.get();
    let w_before = m.write_ops_total.get();
    load_metering(dir.path()).unwrap();
    assert_eq!(
        m.read_ops_total.get(),
        r_before + persisted_r,
        "load must seed read_ops by the persisted total"
    );
    assert_eq!(
        m.write_ops_total.get(),
        w_before + persisted_w,
        "load must seed write_ops by the persisted total"
    );

    // (4) A too-new metering file is refused with a typed FormatTooNew — a
    // newer engine's totals are never misread (rollback safety).
    let newer = tempfile::tempdir().unwrap();
    let mut buf = METERING.header().to_bytes().to_vec();
    buf[4] = 2; // format_version LE low byte → 2 (> current_write = 1)
    buf[5] = 0;
    buf.extend_from_slice(&[0u8; 6 * 8]);
    std::fs::write(newer.path().join(METERING_FILE), &buf).unwrap();
    let err = load_metering(newer.path()).unwrap_err();
    assert!(
        matches!(err, galaxdb_common::GalaxError::FormatTooNew { .. }),
        "too-new metering file must be refused, got {err:?}"
    );
}
