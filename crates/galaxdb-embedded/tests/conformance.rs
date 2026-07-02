//! SQL conformance + regression corpus (HTAP task 24).
//!
//! Each case is `(sql, expected)` run against a **real** `Database` — the
//! full parser → planner → classifier → (native executor | DataFusion
//! analytical engine) → storage stack — over data that has been flushed to
//! real on-disk SST blocks (not an in-memory spike table). The corpus is the
//! regression net for the query engine: relational, analytical, type-system,
//! and ordering behavior. It runs in CI on every change and is the suite the
//! `datafusion-bump` job (task 25) replays against a candidate DataFusion.
//!
//! Cases are grouped and easy to extend: add a `Case` to the relevant list.
//! `expected` is the result rows rendered as `|`-joined cells, row-major, in
//! the order the engine returned them (so ORDER BY cases assert ordering).

use galaxdb_embedded::{Database, QueryResult};

/// One conformance case: a SQL query and its expected rendered rows.
struct Case {
    sql: &'static str,
    /// Expected rows, each a `|`-joined string of the row's cell values in
    /// column order. Empty slice = zero rows expected.
    expected: &'static [&'static str],
}

/// Render a `QueryResult::Rows` to `|`-joined cells per row, preserving the
/// engine's row + column order.
fn render(result: QueryResult) -> Vec<String> {
    match result {
        QueryResult::Rows(rows) => rows
            .into_iter()
            .map(|r| {
                r.values
                    .iter()
                    .map(|(_, v)| v.clone())
                    .collect::<Vec<_>>()
                    .join("|")
            })
            .collect(),
        other => panic!("expected Rows, got {other:?}"),
    }
}

fn temp_db() -> Database {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("conformance_db");
    std::mem::forget(dir); // keep the dir alive for the test process
    Database::open(p.to_str().unwrap()).unwrap()
}

/// Seed a database with the fixture schema + data used by the corpus, then
/// flush so every subsequent query reads SST-backed columnar data.
fn seeded_db() -> Database {
    let mut db = temp_db();
    db.execute("CREATE TABLE emp (id INT PRIMARY KEY, name TEXT, dept TEXT, salary INT, active BOOLEAN)")
        .unwrap();
    let rows = [
        (1, "alice", "eng", 100, true),
        (2, "bob", "eng", 120, true),
        (3, "carol", "sales", 90, false),
        (4, "dave", "sales", 110, true),
        (5, "erin", "eng", 130, true),
        (6, "frank", "hr", 80, false),
    ];
    for (id, name, dept, salary, active) in rows {
        db.execute(&format!(
            "INSERT INTO emp (id, name, dept, salary, active) \
             VALUES ({id}, '{name}', '{dept}', {salary}, {active})"
        ))
        .unwrap();
    }
    db.execute("CREATE TABLE dept (dname TEXT PRIMARY KEY, floor INT)")
        .unwrap();
    for (dname, floor) in [("eng", 3), ("sales", 2), ("hr", 1)] {
        db.execute(&format!(
            "INSERT INTO dept (dname, floor) VALUES ('{dname}', {floor})"
        ))
        .unwrap();
    }
    // Flush to real SSTs so the corpus exercises the on-disk columnar path.
    db.flush().unwrap();
    db
}

fn run_cases(db: &mut Database, cases: &[Case]) {
    for c in cases {
        let got = render(db.execute(c.sql).unwrap());
        assert_eq!(
            got, c.expected,
            "\n  SQL: {}\n  expected: {:?}\n  got:      {:?}",
            c.sql, c.expected, got
        );
    }
}

#[test]
fn relational_native_cases() {
    let mut db = seeded_db();
    run_cases(
        &mut db,
        &[
            // Point lookup by primary key.
            Case { sql: "SELECT name FROM emp WHERE id = 3", expected: &["carol"] },
            // Projection + text equality.
            Case { sql: "SELECT id FROM emp WHERE name = 'bob'", expected: &["2"] },
            // Integer range.
            Case {
                sql: "SELECT id FROM emp WHERE salary >= 120",
                expected: &["2", "5"],
            },
            // AND / OR.
            Case {
                sql: "SELECT id FROM emp WHERE dept = 'eng' AND salary > 110",
                expected: &["2", "5"],
            },
            Case {
                sql: "SELECT id FROM emp WHERE id = 1 OR id = 6",
                expected: &["1", "6"],
            },
            // Boolean column filter.
            Case {
                sql: "SELECT id FROM emp WHERE active = false",
                expected: &["3", "6"],
            },
            // No match → empty.
            Case { sql: "SELECT id FROM emp WHERE salary > 1000", expected: &[] },
        ],
    );
}

#[test]
fn analytical_cases() {
    let mut db = seeded_db();
    run_cases(
        &mut db,
        &[
            // Aggregate.
            Case { sql: "SELECT COUNT(*) FROM emp", expected: &["6"] },
            Case { sql: "SELECT SUM(salary) FROM emp", expected: &["630"] },
            Case { sql: "SELECT MIN(salary), MAX(salary) FROM emp", expected: &["80|130"] },
            // GROUP BY + ORDER BY (deterministic order).
            Case {
                sql: "SELECT dept, COUNT(*) AS n FROM emp GROUP BY dept ORDER BY dept",
                expected: &["eng|3", "hr|1", "sales|2"],
            },
            // HAVING.
            Case {
                sql: "SELECT dept, COUNT(*) AS n FROM emp GROUP BY dept HAVING COUNT(*) > 1 ORDER BY dept",
                expected: &["eng|3", "sales|2"],
            },
            // ORDER BY DESC + LIMIT.
            Case {
                sql: "SELECT name FROM emp ORDER BY salary DESC LIMIT 2",
                expected: &["erin", "bob"],
            },
            // OFFSET.
            Case {
                sql: "SELECT id FROM emp ORDER BY id ASC LIMIT 2 OFFSET 2",
                expected: &["3", "4"],
            },
            // DISTINCT.
            Case {
                sql: "SELECT DISTINCT dept FROM emp ORDER BY dept",
                expected: &["eng", "hr", "sales"],
            },
            // JOIN + aggregate across two SST-backed tables.
            Case {
                sql: "SELECT d.floor, COUNT(*) AS n FROM emp e \
                      JOIN dept d ON e.dept = d.dname GROUP BY d.floor ORDER BY d.floor",
                expected: &["1|1", "2|2", "3|3"],
            },
            // UNION.
            Case {
                sql: "SELECT dept FROM emp WHERE dept = 'hr' \
                      UNION SELECT dname FROM dept WHERE dname = 'eng' ORDER BY 1",
                expected: &["eng", "hr"],
            },
        ],
    );
}

#[test]
fn type_and_edge_cases() {
    let mut db = temp_db();
    db.execute(
        "CREATE TABLE t (id INT PRIMARY KEY, big BIGINT, ratio DOUBLE PRECISION, \
         label TEXT, flag BOOLEAN)",
    )
    .unwrap();
    db.execute("INSERT INTO t (id, big, ratio, label, flag) VALUES (1, 9000000000, 1.5, 'x', true)")
        .unwrap();
    db.execute("INSERT INTO t (id, big, ratio, label, flag) VALUES (2, -42, 2.25, 'y', false)")
        .unwrap();
    db.flush().unwrap();
    run_cases(
        &mut db,
        &[
            // BIGINT round-trips.
            Case { sql: "SELECT big FROM t WHERE id = 1", expected: &["9000000000"] },
            // Negative integer.
            Case { sql: "SELECT big FROM t WHERE id = 2", expected: &["-42"] },
            // Float aggregate.
            Case { sql: "SELECT SUM(ratio) FROM t", expected: &["3.75"] },
            // Boolean render.
            Case {
                sql: "SELECT id FROM t WHERE flag = true",
                expected: &["1"],
            },
        ],
    );
}
