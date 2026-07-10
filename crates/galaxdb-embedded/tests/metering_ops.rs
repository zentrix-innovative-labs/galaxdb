//! v0.6 E-4 metering: op-counter exactness on the embedded path.
//!
//! This lives in its own integration-test binary with a single `#[test]`
//! so the process-global Prometheus counters are touched only by this
//! test — no parallel test in the same binary can pollute the deltas we
//! assert. It proves the "one statement = one op" contract, including
//! that a multi-row INSERT is **one** write op (not one per row), and
//! that reads and vector-less writes land on the right counters.

use galaxdb_embedded::Database;

fn read_ops() -> u64 {
    galaxdb_observe::metrics().read_ops_total.get()
}
fn write_ops() -> u64 {
    galaxdb_observe::metrics().write_ops_total.get()
}

#[test]
fn op_counters_are_exact_and_statement_level() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("meter_db");
    let mut db = Database::open(path.to_str().unwrap()).unwrap();

    // DDL must NOT move any op counter.
    let r0 = read_ops();
    let w0 = write_ops();
    db.execute("CREATE TABLE t (id INT PRIMARY KEY, body TEXT)")
        .unwrap();
    assert_eq!(read_ops(), r0, "CREATE TABLE must not count as a read");
    assert_eq!(write_ops(), w0, "CREATE TABLE must not count as a write");

    // Single-row INSERT = 1 write op.
    let w1 = write_ops();
    db.execute("INSERT INTO t (id, body) VALUES (1, 'a')").unwrap();
    assert_eq!(write_ops(), w1 + 1, "single-row INSERT = 1 write op");

    // Multi-row INSERT of 3 rows = 1 write op (NOT 3). This is the whole
    // point of counting at the statement ingress above the row fan-out.
    let w2 = write_ops();
    db.execute("INSERT INTO t (id, body) VALUES (2,'b'),(3,'c'),(4,'d')")
        .unwrap();
    assert_eq!(
        write_ops(),
        w2 + 1,
        "3-row INSERT must be 1 write op, not 3 (statement-level)"
    );

    // UPDATE = 1 write op (affects multiple rows, still one op).
    let w3 = write_ops();
    db.execute("UPDATE t SET body = 'z' WHERE id >= 1").unwrap();
    assert_eq!(write_ops(), w3 + 1, "UPDATE = 1 write op regardless of rows");

    // DELETE = 1 write op.
    let w4 = write_ops();
    db.execute("DELETE FROM t WHERE id = 4").unwrap();
    assert_eq!(write_ops(), w4 + 1, "DELETE = 1 write op");

    // SELECT (full scan) = 1 read op, and moves no write op.
    let r1 = read_ops();
    let w5 = write_ops();
    let _ = db.execute("SELECT id, body FROM t").unwrap();
    assert_eq!(read_ops(), r1 + 1, "SELECT = 1 read op");
    assert_eq!(write_ops(), w5, "SELECT must not move the write counter");

    // Total writes issued: 1 + 1 + 1 + 1 = 4 (single insert, multi insert,
    // update, delete). Reads: at least 1 (the final SELECT).
    assert_eq!(write_ops(), w0 + 4, "exactly 4 write statements were issued");

    // near-dedup: `WHERE NOT DUPLICATE` runs the group-level pass over the
    // buffered candidate set, incrementing near_dedup_rows_total by the rows
    // it processes. Rows here have no `_near_duplicate_group` (all survive),
    // but the pass still consumes them — which is exactly what we meter.
    let nd0 = galaxdb_observe::metrics().near_dedup_rows_total.get();
    let _ = db.execute("SELECT id FROM t WHERE NOT DUPLICATE").unwrap();
    assert!(
        galaxdb_observe::metrics().near_dedup_rows_total.get() > nd0,
        "WHERE NOT DUPLICATE must increment near_dedup_rows_total by the rows processed"
    );

    // Capacity gauges: a flush refreshes rows_total and storage_bytes from
    // the real engine state. After inserting 4 rows and deleting 1, at least
    // some rows remain and the data dir has bytes on disk.
    db.flush().unwrap();
    let m = galaxdb_observe::metrics();
    assert!(
        m.rows_total.get() > 0,
        "rows_total gauge must reflect live rows after flush, got {}",
        m.rows_total.get()
    );
    assert!(
        m.storage_bytes.get() > 0,
        "storage_bytes gauge must reflect physical on-disk bytes after flush, got {}",
        m.storage_bytes.get()
    );

    // process_start_time_seconds is set once at startup to a real unix time.
    assert!(
        m.process_start_time_seconds.get() > 1_600_000_000,
        "process_start_time_seconds must be a real unix timestamp"
    );
}
