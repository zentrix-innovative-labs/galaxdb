//! v0.7 Serializable Snapshot Isolation (inventory 8.14) — end-to-end over the
//! real SQL/transaction path. Proves the classic write-skew anomaly is aborted
//! under SERIALIZABLE and (for reference) allowed under the default snapshot
//! isolation.

use galaxdb_embedded::Database;

fn make_db(dir: &std::path::Path) -> Database {
    let mut db = Database::open(dir.join("db").to_str().unwrap()).unwrap();
    db.execute("CREATE TABLE oncall (id INT PRIMARY KEY, flag INT)").unwrap();
    db.execute("INSERT INTO oncall (id, flag) VALUES (1, 1)").unwrap();
    db.execute("INSERT INTO oncall (id, flag) VALUES (2, 1)").unwrap();
    db
}

#[test]
fn ssi_write_skew_aborts_over_sql() {
    let dir = tempfile::tempdir().unwrap();
    let db = make_db(dir.path());

    // Two concurrent SERIALIZABLE transactions, the classic write-skew shape:
    // each reads the shared table then writes a different row based on it.
    let t1 = db.begin_transaction_serializable().unwrap();
    let t2 = db.begin_transaction_serializable().unwrap();

    // Both read the table (records a table-granularity SIREAD).
    db.execute_in_txn("SELECT id FROM oncall", &t1, None).unwrap();
    db.execute_in_txn("SELECT id FROM oncall", &t2, None).unwrap();

    // Each updates a different row (no write-write conflict).
    db.execute_in_txn("UPDATE oncall SET flag = 0 WHERE id = 1", &t1, None)
        .unwrap();
    db.execute_in_txn("UPDATE oncall SET flag = 0 WHERE id = 2", &t2, None)
        .unwrap();

    // T1 commits; T2 read a table T1 wrote after T2's snapshot → must abort.
    db.commit_transaction(&t1).unwrap();
    let err = db.commit_transaction(&t2).unwrap_err();
    assert!(
        matches!(err, galaxdb_common::GalaxError::WriteConflict),
        "SERIALIZABLE must abort the write-skew with 40001, got {err:?}"
    );
}

#[test]
fn si_default_allows_write_skew() {
    // Default (snapshot isolation) transactions do NOT track reads and both
    // commit — documents the SI limitation the SERIALIZABLE level removes.
    let dir = tempfile::tempdir().unwrap();
    let db = make_db(dir.path());

    let t1 = db.begin_transaction().unwrap();
    let t2 = db.begin_transaction().unwrap();
    db.execute_in_txn("SELECT id FROM oncall", &t1, None).unwrap();
    db.execute_in_txn("SELECT id FROM oncall", &t2, None).unwrap();
    db.execute_in_txn("UPDATE oncall SET flag = 0 WHERE id = 1", &t1, None)
        .unwrap();
    db.execute_in_txn("UPDATE oncall SET flag = 0 WHERE id = 2", &t2, None)
        .unwrap();
    db.commit_transaction(&t1).unwrap();
    // Under SI both commit (write-skew possible).
    db.commit_transaction(&t2).unwrap();
}

#[test]
fn ssi_no_false_abort_on_disjoint_tables() {
    // A SERIALIZABLE txn that reads/writes a table no one else touched must
    // commit — no spurious abort.
    let dir = tempfile::tempdir().unwrap();
    let mut db = make_db(dir.path());
    db.execute("CREATE TABLE other (id INT PRIMARY KEY, v INT)").unwrap();
    db.execute("INSERT INTO other (id, v) VALUES (1, 1)").unwrap();

    let t1 = db.begin_transaction_serializable().unwrap();
    let t2 = db.begin_transaction_serializable().unwrap();
    db.execute_in_txn("SELECT id FROM oncall", &t1, None).unwrap();
    db.execute_in_txn("SELECT id FROM other", &t2, None).unwrap();
    db.execute_in_txn("UPDATE oncall SET flag = 0 WHERE id = 1", &t1, None)
        .unwrap();
    db.execute_in_txn("UPDATE other SET v = 2 WHERE id = 1", &t2, None)
        .unwrap();
    db.commit_transaction(&t1).unwrap();
    // Different tables → no rw-antidependency → t2 commits.
    db.commit_transaction(&t2).unwrap();
}
