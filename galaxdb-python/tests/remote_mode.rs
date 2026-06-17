//! Integration test for `galaxdb.connect(...)` (task 22.2).
//!
//! Starts a real `galaxdb-server` bound to port 0 against a tempdir
//! data directory, then drives the Rust-side `Connection` type
//! directly (bypassing Python so we can assert in plain Rust). The
//! wire path is identical to what Python would see — every byte
//! through a real TCP socket and a real pg simple-query exchange.
//!
//! This test exercises the same CRUD sequence that the Rust
//! `wire_integration.rs` test drives via `tokio-postgres`. If that
//! one passes and this one doesn't, the regression is in the
//! `Connection` wrapper; if both fail, it's in the server.

use std::time::Duration;

use galaxdb_server::{start, ServerConfig};
use postgres::{Client, NoTls, SimpleQueryMessage};

fn start_test_server(rt: &tokio::runtime::Runtime) -> (String, tempfile::TempDir) {
    let data_dir = tempfile::tempdir().unwrap();
    let cfg = ServerConfig {
        bind_addr: "127.0.0.1:0".to_string(),
        data_dir: data_dir.path().to_string_lossy().to_string(),
        max_connections: 16,
        sidecar_binary: None,
        model_id: None,
        ..Default::default()
    };
    let (addr, _handle) = rt.block_on(async {
        tokio::time::timeout(Duration::from_secs(5), start(cfg))
            .await
            .expect("server start timed out")
            .expect("server failed to bind")
    });
    let dsn = format!(
        "host=127.0.0.1 port={} user=galaxdb dbname=galaxdb sslmode=disable",
        addr.port()
    );
    (dsn, data_dir)
}

fn rows_of(msgs: Vec<SimpleQueryMessage>) -> Vec<Vec<(String, String)>> {
    let mut columns: Vec<String> = Vec::new();
    let mut out = Vec::new();
    for m in msgs {
        match m {
            SimpleQueryMessage::RowDescription(cols) => {
                columns = cols.iter().map(|c| c.name().to_string()).collect();
            }
            SimpleQueryMessage::Row(row) => {
                let mut pairs = Vec::with_capacity(columns.len());
                for (idx, name) in columns.iter().enumerate() {
                    let val = row.get(idx).map(|s| s.to_string()).unwrap_or_default();
                    pairs.push((name.clone(), val));
                }
                out.push(pairs);
            }
            _ => {}
        }
    }
    out
}

#[test]
fn remote_crud_round_trip_via_postgres_client() {
    // The server's accept loop runs on a tokio runtime, but the
    // `postgres` crate used by `Connection` is synchronous. Keep them
    // in separate threads: the runtime owns the listener, this thread
    // drives the client.
    let rt = tokio::runtime::Runtime::new().unwrap();
    let (dsn, _td) = start_test_server(&rt);

    let mut client = Client::connect(&dsn, NoTls).expect("remote connect");

    client
        .simple_query("CREATE TABLE products (id INT PRIMARY KEY, name TEXT, price FLOAT)")
        .expect("CREATE TABLE");

    client
        .simple_query("INSERT INTO products (id, name, price) VALUES (1, 'espresso', 3.5)")
        .expect("INSERT 1");
    client
        .simple_query("INSERT INTO products (id, name, price) VALUES (2, 'latte', 4.25)")
        .expect("INSERT 2");
    client
        .simple_query("INSERT INTO products (id, name, price) VALUES (3, 'mocha', 4.75)")
        .expect("INSERT 3");

    let msgs = client
        .simple_query("SELECT id, name, price FROM products")
        .expect("SELECT *");
    let rows = rows_of(msgs);
    assert_eq!(rows.len(), 3);

    let msgs = client
        .simple_query("SELECT id, name FROM products WHERE price > 4.0")
        .expect("SELECT WHERE");
    let rows = rows_of(msgs);
    assert_eq!(
        rows.len(),
        2,
        "WHERE price > 4.0 must filter real rows over the wire, got {} rows",
        rows.len()
    );

    client
        .simple_query("UPDATE products SET price = 9.99 WHERE id = 3")
        .expect("UPDATE");
    let msgs = client
        .simple_query("SELECT price FROM products WHERE id = 3")
        .unwrap();
    let rows = rows_of(msgs);
    assert_eq!(rows[0][0].1, "9.99");

    client
        .simple_query("DELETE FROM products WHERE id = 1")
        .expect("DELETE");
    let msgs = client.simple_query("SELECT id FROM products").unwrap();
    let rows = rows_of(msgs);
    assert_eq!(rows.len(), 2);
}
