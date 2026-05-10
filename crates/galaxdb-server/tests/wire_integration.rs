//! Integration test for the GalaxDB wire server.
//!
//! Starts a real `galaxdb-server` on port 0 with a tempdir data
//! directory, then connects with `tokio-postgres` and runs the same
//! CRUD sequence that caught Bug 2 on AWS (the WAL writer's
//! `blocking_recv` panicking inside the tokio runtime).
//!
//! A regression of either Bug 1 (WHERE ignored by `galaxdb-embedded`'s
//! exec_{select,update,delete}) or Bug 2 (WAL blocking inside tokio)
//! shows up as a failing row count or a panic, both caught below.

use std::time::Duration;

use galaxdb_server::{start, ServerConfig};
use tokio_postgres::NoTls;

async fn start_server() -> (String, tempfile::TempDir) {
    let data_dir = tempfile::tempdir().unwrap();
    let cfg = ServerConfig {
        bind_addr: "127.0.0.1:0".to_string(),
        data_dir: data_dir.path().to_string_lossy().to_string(),
        max_connections: 16,
    };

    let (addr, _handle) = start(cfg).await.expect("server failed to bind");
    let conn_str = format!(
        "host=127.0.0.1 port={} user=galaxdb dbname=galaxdb sslmode=disable",
        addr.port()
    );
    (conn_str, data_dir)
}

#[tokio::test]
async fn crud_round_trip_over_wire() {
    let (conn_str, _td) = start_server().await;

    // tokio-postgres returns both the client and a connection future;
    // we drive the connection on a side task.
    let (client, connection) = tokio::time::timeout(
        Duration::from_secs(5),
        tokio_postgres::connect(&conn_str, NoTls),
    )
    .await
    .expect("connect timed out")
    .expect("connect failed");

    tokio::spawn(async move {
        let _ = connection.await;
    });

    // DDL.
    client
        .simple_query(
            "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price FLOAT)",
        )
        .await
        .expect("create table failed");

    // Phase I Bug 2 regression: INSERT used to panic the server here
    // because `WalWriter::append_sync` calls `blocking_recv` on a
    // tokio worker. `spawn_blocking` in the wire handler is what keeps
    // this test green.
    client
        .simple_query("INSERT INTO products (id, name, price) VALUES (1, 'espresso', 3.50)")
        .await
        .expect("insert 1 failed (wire server panicked?)");
    client
        .simple_query("INSERT INTO products (id, name, price) VALUES (2, 'latte', 4.25)")
        .await
        .expect("insert 2 failed");
    client
        .simple_query("INSERT INTO products (id, name, price) VALUES (3, 'mocha', 4.75)")
        .await
        .expect("insert 3 failed");

    // Plain read — full table back.
    let msgs = client
        .simple_query("SELECT id, name, price FROM products")
        .await
        .expect("select failed");
    let rows: Vec<_> = msgs
        .into_iter()
        .filter_map(|m| match m {
            tokio_postgres::SimpleQueryMessage::Row(r) => Some(r),
            _ => None,
        })
        .collect();
    assert_eq!(rows.len(), 3, "expected 3 rows from plain SELECT");

    // Phase I Bug 1 regression: WHERE clause must actually filter.
    // Before the fix, `galaxdb-embedded::exec_select` hard-coded
    // `filter: None` and this returned all 3 rows.
    let msgs = client
        .simple_query("SELECT id, name FROM products WHERE price > 4.0")
        .await
        .unwrap();
    let rows: Vec<_> = msgs
        .into_iter()
        .filter_map(|m| match m {
            tokio_postgres::SimpleQueryMessage::Row(r) => Some(r),
            _ => None,
        })
        .collect();
    assert_eq!(
        rows.len(),
        2,
        "WHERE price > 4.0 must return 2 rows (latte + mocha), got {}",
        rows.len()
    );

    // Point lookup.
    let msgs = client
        .simple_query("SELECT id, name FROM products WHERE id = 2")
        .await
        .unwrap();
    let rows: Vec<_> = msgs
        .into_iter()
        .filter_map(|m| match m {
            tokio_postgres::SimpleQueryMessage::Row(r) => Some(r),
            _ => None,
        })
        .collect();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("name"), Some("latte"));

    // UPDATE with WHERE. Before Bug 1 was fixed, every row would be
    // mutated. Now only id=3 should change.
    client
        .simple_query("UPDATE products SET price = 9.99 WHERE id = 3")
        .await
        .unwrap();

    let msgs = client
        .simple_query("SELECT price FROM products WHERE id = 2")
        .await
        .unwrap();
    let latte_price = msgs
        .iter()
        .find_map(|m| match m {
            tokio_postgres::SimpleQueryMessage::Row(r) => r.get("price").map(|s| s.to_string()),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        latte_price, "4.25",
        "UPDATE WHERE id=3 must not change latte's price"
    );

    let msgs = client
        .simple_query("SELECT price FROM products WHERE id = 3")
        .await
        .unwrap();
    let mocha_price = msgs
        .iter()
        .find_map(|m| match m {
            tokio_postgres::SimpleQueryMessage::Row(r) => r.get("price").map(|s| s.to_string()),
            _ => None,
        })
        .unwrap();
    assert_eq!(mocha_price, "9.99");

    // DELETE with WHERE.
    client
        .simple_query("DELETE FROM products WHERE id = 1")
        .await
        .unwrap();

    let msgs = client
        .simple_query("SELECT id FROM products")
        .await
        .unwrap();
    let remaining: Vec<_> = msgs
        .into_iter()
        .filter_map(|m| match m {
            tokio_postgres::SimpleQueryMessage::Row(r) => Some(r),
            _ => None,
        })
        .collect();
    assert_eq!(
        remaining.len(),
        2,
        "after DELETE id=1 there must be 2 rows, got {}",
        remaining.len()
    );
}

#[tokio::test]
async fn many_concurrent_inserts_do_not_panic() {
    // Phase I Bug 2 hardening: drive inserts concurrently from several
    // clients so the spawn_blocking-based write path is exercised under
    // contention on the RwLock and the WAL group commit channel.
    let (conn_str, _td) = start_server().await;

    let mut handles = Vec::new();
    for worker in 0..4 {
        let cs = conn_str.clone();
        handles.push(tokio::spawn(async move {
            let (client, connection) = tokio_postgres::connect(&cs, NoTls).await.unwrap();
            tokio::spawn(async move {
                let _ = connection.await;
            });
            if worker == 0 {
                // one worker creates the table
                client
                    .simple_query("CREATE TABLE c (id INTEGER PRIMARY KEY, tag TEXT)")
                    .await
                    .unwrap();
            } else {
                // others wait briefly for the DDL to land
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
            for i in 0..10 {
                let id = worker * 100 + i;
                client
                    .simple_query(&format!(
                        "INSERT INTO c (id, tag) VALUES ({id}, 'w{worker}-i{i}')"
                    ))
                    .await
                    .expect("concurrent insert panicked the server");
            }
        }));
    }
    for h in handles {
        h.await.unwrap();
    }

    // Verify all 40 rows landed. Connect one more time and count.
    let (client, connection) = tokio_postgres::connect(&conn_str, NoTls).await.unwrap();
    tokio::spawn(async move {
        let _ = connection.await;
    });
    let msgs = client.simple_query("SELECT id FROM c").await.unwrap();
    let rows: Vec<_> = msgs
        .into_iter()
        .filter_map(|m| match m {
            tokio_postgres::SimpleQueryMessage::Row(r) => Some(r),
            _ => None,
        })
        .collect();
    assert_eq!(rows.len(), 40);
}
