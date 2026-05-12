"""Remote-mode CRUD coverage for the galaxdb Python module.

Starts a real `galaxdb-server` on a free port (via the `running_server`
fixture in `conftest.py`), connects with `galaxdb.connect(dsn)`, and
drives a full CRUD sequence over the PostgreSQL wire protocol. The
server and data directory are cleaned up automatically after each test.

Task 22.6 acceptance: the second of three pytest files covering the
Python client's public API end-to-end.
"""

from __future__ import annotations

import galaxdb


def test_remote_connect_and_close(running_server) -> None:
    dsn, _data_dir = running_server
    conn = galaxdb.connect(dsn)
    try:
        assert conn.is_open
        r = repr(conn)
        assert "Connection(" in r
        assert "open" in r
    finally:
        conn.close()
    assert not conn.is_open


def test_remote_crud_round_trip(running_server) -> None:
    dsn, _data_dir = running_server
    conn = galaxdb.connect(dsn)
    try:
        conn.execute(
            "CREATE TABLE products (id INT PRIMARY KEY, name TEXT, price FLOAT)"
        )
        conn.execute("INSERT INTO products (id, name, price) VALUES (1, 'espresso', 3.5)")
        conn.execute("INSERT INTO products (id, name, price) VALUES (2, 'latte', 4.25)")
        conn.execute("INSERT INTO products (id, name, price) VALUES (3, 'mocha', 4.75)")

        rows = conn.execute("SELECT id, name, price FROM products")
        assert isinstance(rows, list)
        assert len(rows) == 3
        for r in rows:
            assert "id" in r and "name" in r and "price" in r

        rows = conn.execute(
            "SELECT id, name FROM products WHERE price > 4.0"
        )
        assert len(rows) == 2
        ids = {int(r["id"]) for r in rows}
        assert ids == {2, 3}

        conn.execute("UPDATE products SET price = 9.99 WHERE id = 3")
        rows = conn.execute("SELECT price FROM products WHERE id = 3")
        assert len(rows) == 1
        assert rows[0]["price"] == "9.99"

        conn.execute("DELETE FROM products WHERE id = 1")
        rows = conn.execute("SELECT id FROM products")
        remaining = sorted(int(r["id"]) for r in rows)
        assert remaining == [2, 3]
    finally:
        conn.close()


def test_execute_after_close_raises(running_server) -> None:
    import pytest

    dsn, _ = running_server
    conn = galaxdb.connect(dsn)
    conn.execute("CREATE TABLE x (id INT)")
    conn.close()
    with pytest.raises(RuntimeError) as excinfo:
        conn.execute("SELECT 1")
    assert "closed" in str(excinfo.value).lower()


def test_two_connections_to_same_server(running_server) -> None:
    dsn, _ = running_server
    a = galaxdb.connect(dsn)
    b = galaxdb.connect(dsn)
    try:
        a.execute("CREATE TABLE shared (id INT PRIMARY KEY, v TEXT)")
        a.execute("INSERT INTO shared (id, v) VALUES (1, 'hello')")

        rows = b.execute("SELECT id, v FROM shared")
        assert len(rows) == 1
        assert rows[0]["v"] == "hello"
    finally:
        a.close()
        b.close()
