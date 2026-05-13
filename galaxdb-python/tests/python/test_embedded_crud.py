"""Embedded-mode CRUD coverage for the galaxdb Python module.

These tests drive `galaxdb.Database(path)` exactly as a Python user
would: open a database on a tempdir, issue SQL, assert the returned
dicts and row counts. There is no server, no tokio runtime, no wire
protocol — just the PyO3 FFI boundary on top of the real storage
engine.

Task 22.6 acceptance: the first of three pytest files covering
"embedded mode CRUD, remote mode CRUD, training_dataset returns
valid IterableDataset".
"""

from __future__ import annotations

from pathlib import Path

import pytest

import galaxdb


def test_open_empty_database(temp_db_dir: Path) -> None:
    db = galaxdb.Database(str(temp_db_dir))
    assert db.table_count == 0
    assert Path(db.path) == temp_db_dir
    assert not db.table_exists("nope")


def test_create_insert_select(temp_db_dir: Path) -> None:
    db = galaxdb.Database(str(temp_db_dir))

    status = db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
    assert isinstance(status, str)

    affected = db.execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
    assert affected == 1
    affected = db.execute("INSERT INTO users (id, name) VALUES (2, 'bob')")
    assert affected == 1

    rows = db.execute("SELECT id, name FROM users")
    assert isinstance(rows, list)
    assert len(rows) == 2
    assert all(isinstance(r, dict) for r in rows)

    names = {r["name"] for r in rows}
    assert names == {"alice", "bob"}

    assert db.table_count == 1
    assert db.table_exists("users")


def test_where_filter_restricts_rows(temp_db_dir: Path) -> None:
    db = galaxdb.Database(str(temp_db_dir))
    db.execute(
        "CREATE TABLE products (id INT PRIMARY KEY, name TEXT, price FLOAT)"
    )
    db.execute("INSERT INTO products (id, name, price) VALUES (1, 'espresso', 3.5)")
    db.execute("INSERT INTO products (id, name, price) VALUES (2, 'latte', 4.25)")
    db.execute("INSERT INTO products (id, name, price) VALUES (3, 'mocha', 4.75)")

    rows = db.execute("SELECT id, name FROM products WHERE price > 4.0")
    assert len(rows) == 2
    ids = {int(r["id"]) for r in rows}
    assert ids == {2, 3}

    rows = db.execute("SELECT id FROM products WHERE name = 'latte'")
    assert len(rows) == 1
    assert int(rows[0]["id"]) == 2


def test_update_affects_only_matching_rows(temp_db_dir: Path) -> None:
    db = galaxdb.Database(str(temp_db_dir))
    db.execute("CREATE TABLE t (id INT PRIMARY KEY, v INT)")
    db.execute("INSERT INTO t (id, v) VALUES (1, 10)")
    db.execute("INSERT INTO t (id, v) VALUES (2, 20)")
    db.execute("INSERT INTO t (id, v) VALUES (3, 30)")

    affected = db.execute("UPDATE t SET v = 99 WHERE id = 2")
    assert affected == 1

    rows = db.execute("SELECT v FROM t WHERE id = 2")
    assert int(rows[0]["v"]) == 99

    rows = db.execute("SELECT v FROM t WHERE id = 1")
    assert int(rows[0]["v"]) == 10


def test_delete_where_removes_only_matching_rows(temp_db_dir: Path) -> None:
    db = galaxdb.Database(str(temp_db_dir))
    db.execute("CREATE TABLE t (id INT PRIMARY KEY, v INT)")
    for i in range(1, 6):
        db.execute(f"INSERT INTO t (id, v) VALUES ({i}, {i * 10})")

    affected = db.execute("DELETE FROM t WHERE id = 3")
    assert affected == 1

    rows = db.execute("SELECT id FROM t")
    remaining_ids = sorted(int(r["id"]) for r in rows)
    assert remaining_ids == [1, 2, 4, 5]


def test_repr_is_informative(temp_db_dir: Path) -> None:
    db = galaxdb.Database(str(temp_db_dir))
    db.execute("CREATE TABLE t (id INT)")
    r = repr(db)
    assert "Database(" in r
    assert str(temp_db_dir) in r
    assert "tables=1" in r


def test_unknown_table_raises(temp_db_dir: Path) -> None:
    db = galaxdb.Database(str(temp_db_dir))
    with pytest.raises(RuntimeError) as excinfo:
        db.execute("SELECT * FROM does_not_exist")
    assert "does_not_exist" in str(excinfo.value).lower() or "not found" in str(
        excinfo.value
    ).lower()


def test_where_not_duplicate_keeps_one_representative_per_group(
    temp_db_dir: Path,
) -> None:
    """`WHERE NOT DUPLICATE` must collapse near-duplicate groups to a
    single representative (task 35.5 / Req 26). Seeds `docs` with two
    groups plus one ungrouped row and asserts the survivors.

    The `_near_duplicate_group` system column is what the Task 35.4
    background refresh job populates; here we write it directly via
    INSERT since we're testing the query-time filter, not the grouping
    job. Representative selection is "lowest primary key in each
    group", which matches what the Lance exporter's
    `apply_dedup_filter` uses so `WHERE NOT DUPLICATE` and `CREATE
    VERSION TAG ... FOR TRAINING` exports agree per-group.
    """
    db = galaxdb.Database(str(temp_db_dir))
    db.execute(
        "CREATE TABLE docs ("
        "id INT PRIMARY KEY, body TEXT, _near_duplicate_group BIGINT"
        ")"
    )
    # Group 100: ids 3, 1, 4 → representative is id=1.
    db.execute(
        "INSERT INTO docs (id, body, _near_duplicate_group)"
        " VALUES (3, 'hello world', 100)"
    )
    db.execute(
        "INSERT INTO docs (id, body, _near_duplicate_group)"
        " VALUES (1, 'hello world!', 100)"
    )
    db.execute(
        "INSERT INTO docs (id, body, _near_duplicate_group)"
        " VALUES (4, 'hello world.', 100)"
    )
    # Group 200: ids 5, 2 → representative is id=2.
    db.execute(
        "INSERT INTO docs (id, body, _near_duplicate_group)"
        " VALUES (5, 'quick fox', 200)"
    )
    db.execute(
        "INSERT INTO docs (id, body, _near_duplicate_group)"
        " VALUES (2, 'quick fox!', 200)"
    )
    # Ungrouped.
    db.execute(
        "INSERT INTO docs (id, body, _near_duplicate_group)"
        " VALUES (6, 'unique', NULL)"
    )

    rows = db.execute("SELECT id FROM docs WHERE NOT DUPLICATE")
    assert isinstance(rows, list)
    # Exactly three survivors: one representative per group plus the
    # ungrouped row.
    assert len(rows) == 3
    ids = sorted(int(r["id"]) for r in rows)
    assert ids == [1, 2, 6]

    # Composed with a per-row predicate: `id > 1 AND NOT DUPLICATE`
    # first drops id=1, then applies the dedup pass on the narrowed
    # candidate set:
    #   Group 100 survivors {3, 4} → representative id=3 (pk "docs:3" < "docs:4")
    #   Group 200 survivors {5, 2} → representative id=2 (pk "docs:2" < "docs:5")
    #   Ungrouped: id=6 survives
    # Final answer: [2, 3, 6]. This proves the dedup pass runs AFTER
    # the per-row filter, not before.
    rows = db.execute(
        "SELECT id FROM docs WHERE id > 1 AND NOT DUPLICATE"
    )
    ids = sorted(int(r["id"]) for r in rows)
    assert ids == [2, 3, 6]
