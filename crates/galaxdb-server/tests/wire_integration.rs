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
        sidecar_binary: None,
        model_id: None,
        ..Default::default()
    };

    let (addr, _handle) = start(cfg).await.expect("server failed to bind");
    let conn_str = format!(
        "host=127.0.0.1 port={} user=galaxdb dbname=galaxdb sslmode=disable",
        addr.port()
    );
    (conn_str, data_dir)
}

/// Start a server with SCRAM-SHA-256 authentication enabled and an
/// initial superuser provisioned from config. Returns the bound port and
/// the tempdir (kept alive by the caller).
async fn start_auth_server(
    superuser: &str,
    password: &str,
) -> (u16, tempfile::TempDir) {
    let data_dir = tempfile::tempdir().unwrap();
    let cfg = ServerConfig {
        bind_addr: "127.0.0.1:0".to_string(),
        data_dir: data_dir.path().to_string_lossy().to_string(),
        max_connections: 16,
        sidecar_binary: None,
        model_id: None,
        auth_enabled: true,
        trusted_local_user: "galaxdb".to_string(),
        initial_superuser: Some((superuser.to_string(), password.to_string())),
        ..Default::default()
    };
    let (addr, _handle) = start(cfg).await.expect("auth server failed to bind");
    (addr.port(), data_dir)
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

/// Task 9 (Req 6): the extended query protocol — prepared statements with
/// bound parameters over `tokio-postgres`. `prepare_typed` drives
/// Parse(with param OIDs) + Describe + Sync; `query`/`execute` drive
/// Bind(binary/text params) + Execute + Sync. Results are text-typed
/// (consistent with the simple-query result path), so columns are read as
/// `String`. A regression in the dispatcher, the parameter codec, or the
/// describe path shows up as a connect/prepare/query error or a wrong row.
#[tokio::test]
async fn extended_protocol_prepared_statements() {
    use tokio_postgres::types::Type;

    let (conn_str, _td) = start_server().await;
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

    // Set up the table with the simple protocol (already covered above).
    client
        .simple_query("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, price FLOAT)")
        .await
        .expect("create table failed");

    // 1. Prepared INSERT with bound params (int4, text, float8). This is
    //    Parse(typed) + Bind(binary int4 / text / binary float8) + Execute.
    let insert = client
        .prepare_typed(
            "INSERT INTO items (id, name, price) VALUES ($1, $2, $3)",
            &[Type::INT4, Type::TEXT, Type::FLOAT8],
        )
        .await
        .expect("prepare insert failed");
    for (id, name, price) in [(1i32, "espresso", 3.50f64), (2, "latte", 4.25), (3, "mocha", 4.75)] {
        client
            .execute(&insert, &[&id, &name, &price])
            .await
            .unwrap_or_else(|e| panic!("prepared insert ({id},{name}) failed: {e}"));
    }

    // 2. Prepared SELECT with a bound int4 parameter — exercises Bind
    //    parameter substitution into the WHERE clause and the Describe
    //    RowDescription (2 columns). Results come back text-typed.
    let by_id = client
        .prepare_typed(
            "SELECT id, name FROM items WHERE id = $1",
            &[Type::INT4],
        )
        .await
        .expect("prepare select failed");

    let rows = client.query(&by_id, &[&2i32]).await.expect("query failed");
    assert_eq!(rows.len(), 1, "WHERE id = $1 must return exactly one row");
    assert_eq!(rows[0].get::<_, &str>("id"), "2");
    assert_eq!(rows[0].get::<_, &str>("name"), "latte");

    // A different bound value selects a different row — proves the param
    // is actually substituted, not constant-folded at prepare time.
    let rows = client.query(&by_id, &[&3i32]).await.expect("re-query failed");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get::<_, &str>("name"), "mocha");

    // 3. Prepared SELECT with a float8 range parameter.
    let dear = client
        .prepare_typed(
            "SELECT name FROM items WHERE price > $1",
            &[Type::FLOAT8],
        )
        .await
        .expect("prepare range select failed");
    let rows = client.query(&dear, &[&4.0f64]).await.expect("range query failed");
    let mut names: Vec<String> = rows.iter().map(|r| r.get::<_, &str>("name").to_string()).collect();
    names.sort();
    assert_eq!(names, vec!["latte".to_string(), "mocha".to_string()]);

    // 4. A text parameter is escaped correctly (no injection, quotes
    //    doubled). Insert a name with an apostrophe via a bound param,
    //    then read it back.
    client
        .execute(&insert, &[&4i32, &"o'brien", &9.99f64])
        .await
        .expect("insert with apostrophe failed");
    let rows = client
        .query(&by_id, &[&4i32])
        .await
        .expect("query apostrophe row failed");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get::<_, &str>("name"), "o'brien");

    // 5. Final count via the simple protocol confirms all 4 prepared
    //    inserts landed.
    let msgs = client.simple_query("SELECT id FROM items").await.unwrap();
    let count = msgs
        .into_iter()
        .filter(|m| matches!(m, tokio_postgres::SimpleQueryMessage::Row(_)))
        .count();
    assert_eq!(count, 4, "expected 4 rows after prepared inserts, got {count}");
}

/// Task 11 (Req 8): the COPY sub-protocol. `COPY FROM STDIN` bulk-loads
/// text rows through the bulk-insert path; `COPY TO STDOUT` streams them
/// back; a malformed row aborts cleanly without a partial commit.
#[tokio::test]
async fn copy_protocol_round_trip_and_malformed_abort() {
    use bytes::Bytes;
    use futures_util::SinkExt;
    use futures_util::TryStreamExt;

    let (conn_str, _td) = start_server().await;
    let (client, connection) = tokio_postgres::connect(&conn_str, NoTls)
        .await
        .expect("connect failed");
    tokio::spawn(async move {
        let _ = connection.await;
    });

    client
        .simple_query("CREATE TABLE c (id INTEGER PRIMARY KEY, name TEXT)")
        .await
        .expect("create table failed");

    // 1. COPY FROM STDIN — bulk load three rows across two CopyData frames.
    let sink = client
        .copy_in("COPY c (id, name) FROM STDIN")
        .await
        .expect("copy_in failed");
    futures_util::pin_mut!(sink);
    sink.send(Bytes::from_static(b"1\tespresso\n2\tlatte\n"))
        .await
        .expect("send frame 1");
    sink.send(Bytes::from_static(b"3\tmocha\n"))
        .await
        .expect("send frame 2");
    let copied = sink.finish().await.expect("copy_in finish failed");
    assert_eq!(copied, 3, "COPY FROM STDIN must report 3 rows");

    // Verify the rows landed.
    let rows = client
        .simple_query("SELECT id, name FROM c")
        .await
        .unwrap()
        .into_iter()
        .filter(|m| matches!(m, tokio_postgres::SimpleQueryMessage::Row(_)))
        .count();
    assert_eq!(rows, 3, "table must hold the 3 copied rows");

    // 2. COPY TO STDOUT — stream the rows back in text format.
    let stream = client
        .copy_out("COPY c (id, name) TO STDOUT")
        .await
        .expect("copy_out failed");
    let chunks: Vec<Bytes> = stream.try_collect().await.expect("copy_out collect failed");
    let dumped: Vec<u8> = chunks.concat();
    let text = String::from_utf8(dumped).expect("copy out text utf8");
    let lines: Vec<&str> = text.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(lines.len(), 3, "COPY TO STDOUT must stream 3 rows, got: {text:?}");
    assert!(text.contains("1\tespresso"), "missing row 1 in: {text:?}");
    assert!(text.contains("3\tmocha"), "missing row 3 in: {text:?}");

    // 3. Malformed row (wrong column count) aborts the COPY with an error
    //    and leaves the table unchanged (no partial commit, Req 8 AC5).
    let sink = client
        .copy_in("COPY c (id, name) FROM STDIN")
        .await
        .expect("copy_in (bad) failed");
    futures_util::pin_mut!(sink);
    // Three tab-separated cells for a two-column target.
    let _ = sink.send(Bytes::from_static(b"99\tbad\textra\n")).await;
    let result = sink.finish().await;
    assert!(result.is_err(), "malformed COPY row must abort with an error");

    // The table still has exactly the original 3 rows.
    let rows_after = client
        .simple_query("SELECT id FROM c")
        .await
        .unwrap()
        .into_iter()
        .filter(|m| matches!(m, tokio_postgres::SimpleQueryMessage::Row(_)))
        .count();
    assert_eq!(rows_after, 3, "malformed COPY must not partially commit");
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

/// Task 38.6: the server accepts a SQL statement carrying a W3C
/// traceparent via the SQL commenter `/* traceparent='...' */`
/// suffix. The query runs to completion (the commenter is stripped
/// at parse time), proving the wire server doesn't choke on the
/// trailing comment. The full-trace-span coverage (task 38.5) is
/// still pending; this test confirms the header is at least
/// tolerated end-to-end.
#[tokio::test]
async fn wire_server_accepts_sql_commenter_traceparent() {
    let (conn_str, _td) = start_server().await;
    let (client, connection) = tokio_postgres::connect(&conn_str, NoTls).await.unwrap();
    tokio::spawn(async move {
        let _ = connection.await;
    });

    client
        .simple_query("CREATE TABLE tp (id INTEGER PRIMARY KEY)")
        .await
        .expect("create table failed");

    // Real W3C traceparent (trace id + span id from the spec examples).
    let sql = concat!(
        "INSERT INTO tp (id) VALUES (1) ",
        "/* traceparent='00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01' */"
    );
    client
        .simple_query(sql)
        .await
        .expect("insert with SQL commenter must succeed");

    // Read-side commenter must also be tolerated.
    let sql = concat!(
        "SELECT id FROM tp ",
        "/* traceparent='00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01' */"
    );
    let msgs = client.simple_query(sql).await.expect("select with SQL commenter");
    let rows: Vec<_> = msgs
        .into_iter()
        .filter_map(|m| match m {
            tokio_postgres::SimpleQueryMessage::Row(r) => Some(r),
            _ => None,
        })
        .collect();
    assert_eq!(rows.len(), 1);
}

/// Task 40.4 / 40.5: verify the release binary exists and is under the
/// 70 MB "core" size limit. This test is skipped when the release binary
/// hasn't been built yet (e.g. in a dev `cargo test` run). It passes
/// automatically in CI where `cargo build --release` runs first.
///
/// The 70 MB limit is the spec's "core binary" gate — the full binary
/// including the sidecar + model is a separate 350 MB gate that requires
/// the sidecar to be built and the model to be downloaded.
#[test]
fn release_binary_size_under_70mb_when_built() {
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let binary = workspace_root.join("target/release/galaxdb-server");

    if !binary.exists() {
        // Not built yet — skip rather than fail. CI always builds first.
        eprintln!(
            "SKIP: release binary not found at {}; run `cargo build --release -p galaxdb-server` first",
            binary.display()
        );
        return;
    }

    let size_bytes = std::fs::metadata(&binary)
        .expect("stat release binary")
        .len();
    let size_mb = size_bytes as f64 / (1024.0 * 1024.0);

    println!(
        "galaxdb-server release binary: {:.1} MB ({} bytes)",
        size_mb, size_bytes
    );

    const CORE_LIMIT_MB: f64 = 70.0;
    assert!(
        size_mb < CORE_LIMIT_MB,
        "galaxdb-server release binary is {:.1} MB, exceeds the {:.0} MB core limit. \
         Check for accidental large static data or debug symbols leaking into the release build.",
        size_mb,
        CORE_LIMIT_MB
    );
}

/// Task 40.1 / 40.5: verify the HTTP observability server (/health + /metrics)
/// starts alongside the wire-protocol server and returns valid responses.
/// This is the "bind wire protocol + HTTP observability" acceptance criterion.
#[tokio::test]
async fn http_observability_starts_alongside_wire_server() {
    use galaxdb_observe::{start_http, ObserveConfig};

    // Start the HTTP observability server on a free port.
    let obs_cfg = ObserveConfig {
        bind_addr: "127.0.0.1:0".to_string(),
    };
    let (obs_addr, _obs_handle) = start_http(obs_cfg).await.unwrap();

    // /health must return 200 with a JSON body.
    let health_url = format!("http://{}/health", obs_addr);
    let resp = reqwest::get(&health_url).await.unwrap();
    assert_eq!(resp.status(), 200, "/health must return 200");
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body.get("status").is_some(), "/health must return a status field");

    // /metrics must return 200 with Prometheus text format.
    let metrics_url = format!("http://{}/metrics", obs_addr);
    let resp = reqwest::get(&metrics_url).await.unwrap();
    assert_eq!(resp.status(), 200, "/metrics must return 200");
    let ct = resp.headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap();
    assert!(ct.contains("text/plain"), "/metrics must return Prometheus text format");
}

// ---------------------------------------------------------------------------
// Task 6: SCRAM-SHA-256 authentication over the wire (Req 1)
// ---------------------------------------------------------------------------

/// A correct password authenticates and can run statements; the role is
/// the provisioned superuser so it may create tables and roles.
#[tokio::test]
async fn scram_correct_password_connects() {
    let (port, _td) = start_auth_server("admin", "s3cr3t").await;
    let conn_str = format!(
        "host=127.0.0.1 port={port} user=admin password=s3cr3t dbname=galaxdb sslmode=disable"
    );

    let (client, connection) = tokio::time::timeout(
        Duration::from_secs(5),
        tokio_postgres::connect(&conn_str, NoTls),
    )
    .await
    .expect("connect timed out")
    .expect("SCRAM connect with correct password must succeed");

    tokio::spawn(async move {
        let _ = connection.await;
    });

    // The authenticated superuser can run DDL + DML.
    client
        .simple_query("CREATE TABLE t (id INTEGER PRIMARY KEY, n INTEGER)")
        .await
        .expect("superuser DDL must succeed");
    client
        .simple_query("INSERT INTO t (id, n) VALUES (1, 10)")
        .await
        .expect("superuser INSERT must succeed");
    let msgs = client.simple_query("SELECT id, n FROM t").await.unwrap();
    let rows: Vec<_> = msgs
        .into_iter()
        .filter_map(|m| match m {
            tokio_postgres::SimpleQueryMessage::Row(r) => Some(r),
            _ => None,
        })
        .collect();
    assert_eq!(rows.len(), 1);
}

/// A wrong password is rejected with SQLSTATE `28P01` and no session is
/// established.
#[tokio::test]
async fn scram_wrong_password_is_rejected_28p01() {
    let (port, _td) = start_auth_server("admin", "right-password").await;
    let conn_str = format!(
        "host=127.0.0.1 port={port} user=admin password=WRONG-password dbname=galaxdb sslmode=disable"
    );

    let result = tokio::time::timeout(
        Duration::from_secs(5),
        tokio_postgres::connect(&conn_str, NoTls),
    )
    .await
    .expect("connect attempt timed out");

    let err = match result {
        Ok(_) => panic!("wrong password must fail to connect"),
        Err(e) => e,
    };
    // tokio-postgres surfaces the server's ErrorResponse; its SQLSTATE
    // must be 28P01 (invalid_password).
    let code = err
        .as_db_error()
        .map(|e| e.code().code().to_string())
        .unwrap_or_default();
    assert_eq!(code, "28P01", "wrong password must map to SQLSTATE 28P01, got {code:?} ({err})");
}

/// An unknown role is rejected with `28P01` (and the message does not
/// distinguish "no such role" from "wrong password", so it can't be used
/// to enumerate roles).
#[tokio::test]
async fn scram_unknown_role_is_rejected_28p01() {
    let (port, _td) = start_auth_server("admin", "pw").await;
    let conn_str = format!(
        "host=127.0.0.1 port={port} user=ghost password=anything dbname=galaxdb sslmode=disable"
    );

    let result = tokio::time::timeout(
        Duration::from_secs(5),
        tokio_postgres::connect(&conn_str, NoTls),
    )
    .await
    .expect("connect attempt timed out");

    let err = match result {
        Ok(_) => panic!("unknown role must fail to connect"),
        Err(e) => e,
    };
    let code = err
        .as_db_error()
        .map(|e| e.code().code().to_string())
        .unwrap_or_default();
    assert_eq!(code, "28P01", "unknown role must map to SQLSTATE 28P01, got {code:?} ({err})");
}

/// End-to-end authorization over the wire (task 5 + 6 together): the
/// superuser creates a non-privileged role and a table; that role is
/// denied SELECT with `42501` until granted, then succeeds — all without
/// a server restart. This is the wire-path half of task 5.6.
#[tokio::test]
async fn wire_authz_denied_then_granted() {
    let (port, _td) = start_auth_server("root", "rootpw").await;

    // 1. Superuser sets up the table, a row, and a plain role with a
    //    password.
    let admin_dsn = format!(
        "host=127.0.0.1 port={port} user=root password=rootpw dbname=galaxdb sslmode=disable"
    );
    let (admin, admin_conn) = tokio_postgres::connect(&admin_dsn, NoTls).await.unwrap();
    tokio::spawn(async move {
        let _ = admin_conn.await;
    });
    admin
        .simple_query("CREATE TABLE docs (id INTEGER PRIMARY KEY, body TEXT)")
        .await
        .unwrap();
    admin
        .simple_query("INSERT INTO docs (id, body) VALUES (1, 'hello')")
        .await
        .unwrap();
    admin
        .simple_query("CREATE ROLE alice PASSWORD 'alicepw'")
        .await
        .unwrap();

    // 2. alice connects (auth succeeds — she has a verifier) but has no
    //    grant on docs, so SELECT is denied with 42501.
    let alice_dsn = format!(
        "host=127.0.0.1 port={port} user=alice password=alicepw dbname=galaxdb sslmode=disable"
    );
    let (alice, alice_conn) = tokio_postgres::connect(&alice_dsn, NoTls).await.unwrap();
    tokio::spawn(async move {
        let _ = alice_conn.await;
    });
    let denied = alice.simple_query("SELECT id, body FROM docs").await;
    let err = denied.expect_err("alice without a grant must be denied");
    let code = err
        .as_db_error()
        .map(|e| e.code().code().to_string())
        .unwrap_or_default();
    assert_eq!(code, "42501", "ungranted SELECT must map to 42501, got {code:?} ({err})");

    // 3. Superuser grants SELECT.
    admin
        .simple_query("GRANT SELECT ON docs TO alice")
        .await
        .unwrap();

    // 4. alice can now SELECT — the grant took effect with no restart and
    //    on the same live connection.
    let msgs = alice
        .simple_query("SELECT id, body FROM docs")
        .await
        .expect("after GRANT, alice's SELECT must succeed");
    let rows: Vec<_> = msgs
        .into_iter()
        .filter_map(|m| match m {
            tokio_postgres::SimpleQueryMessage::Row(r) => Some(r),
            _ => None,
        })
        .collect();
    assert_eq!(rows.len(), 1, "alice must see the one row after GRANT");

    // 5. A non-superuser cannot administer roles/grants.
    let denied = alice.simple_query("GRANT SELECT ON docs TO alice").await;
    let err = denied.expect_err("alice may not GRANT");
    let code = err
        .as_db_error()
        .map(|e| e.code().code().to_string())
        .unwrap_or_default();
    assert_eq!(code, "42501", "non-superuser GRANT must map to 42501, got {code:?}");
}

// ---------------------------------------------------------------------------
// Task 7: TLS transport encryption (Req 2)
// ---------------------------------------------------------------------------

/// Generate a self-signed cert+key for `localhost`, write them to PEM
/// files in `dir`, and return the two paths. Test-only: the cert is not
/// from a real CA, so the client below uses an accept-any verifier.
fn write_test_cert(dir: &std::path::Path) -> (String, String) {
    let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
        .expect("generate self-signed cert");
    let cert_pem = cert.cert.pem();
    let key_pem = cert.key_pair.serialize_pem();
    let cert_path = dir.join("server.crt");
    let key_path = dir.join("server.key");
    std::fs::write(&cert_path, cert_pem).unwrap();
    std::fs::write(&key_path, key_pem).unwrap();
    (
        cert_path.to_string_lossy().to_string(),
        key_path.to_string_lossy().to_string(),
    )
}

/// A rustls client config that accepts any server certificate. TEST ONLY —
/// it disables authentication of the server, which is acceptable here
/// because the test only verifies that the TLS *channel* is established
/// and that SCRAM runs inside it, not that PKI validation works.
fn insecure_tls_connector() -> tokio_postgres_rustls::MakeRustlsConnect {
    use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
    use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
    use rustls::{DigitallySignedStruct, SignatureScheme};

    #[derive(Debug)]
    struct AcceptAny;
    impl ServerCertVerifier for AcceptAny {
        fn verify_server_cert(
            &self,
            _end_entity: &CertificateDer<'_>,
            _intermediates: &[CertificateDer<'_>],
            _server_name: &ServerName<'_>,
            _ocsp: &[u8],
            _now: UnixTime,
        ) -> Result<ServerCertVerified, rustls::Error> {
            Ok(ServerCertVerified::assertion())
        }
        fn verify_tls12_signature(
            &self,
            _message: &[u8],
            _cert: &CertificateDer<'_>,
            _dss: &DigitallySignedStruct,
        ) -> Result<HandshakeSignatureValid, rustls::Error> {
            Ok(HandshakeSignatureValid::assertion())
        }
        fn verify_tls13_signature(
            &self,
            _message: &[u8],
            _cert: &CertificateDer<'_>,
            _dss: &DigitallySignedStruct,
        ) -> Result<HandshakeSignatureValid, rustls::Error> {
            Ok(HandshakeSignatureValid::assertion())
        }
        fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
            vec![
                SignatureScheme::RSA_PKCS1_SHA256,
                SignatureScheme::ECDSA_NISTP256_SHA256,
                SignatureScheme::RSA_PSS_SHA256,
                SignatureScheme::ED25519,
            ]
        }
    }

    let config = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(std::sync::Arc::new(AcceptAny))
        .with_no_client_auth();
    tokio_postgres_rustls::MakeRustlsConnect::new(config)
}

/// `sslmode=require` against a TLS-enabled server: the client negotiates
/// TLS, completes the handshake, and runs SQL over the encrypted channel.
#[tokio::test]
async fn tls_require_connects_over_encrypted_channel() {
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    let data_dir = tempfile::tempdir().unwrap();
    let (cert_path, key_path) = write_test_cert(data_dir.path());

    let cfg = ServerConfig {
        bind_addr: "127.0.0.1:0".to_string(),
        data_dir: data_dir.path().to_string_lossy().to_string(),
        max_connections: 16,
        sidecar_binary: None,
        model_id: None,
        tls_mode: galaxdb_wire::tls::TlsMode::Require,
        tls_cert_path: Some(cert_path),
        tls_key_path: Some(key_path),
        ..Default::default()
    };
    let (addr, _handle) = start(cfg).await.expect("tls server failed to bind");

    let conn_str = format!(
        "host=localhost port={} user=galaxdb dbname=galaxdb sslmode=require",
        addr.port()
    );
    let connector = insecure_tls_connector();
    let (client, connection) = tokio::time::timeout(
        Duration::from_secs(5),
        tokio_postgres::connect(&conn_str, connector),
    )
    .await
    .expect("TLS connect timed out")
    .expect("TLS connect with sslmode=require must succeed");

    tokio::spawn(async move {
        let _ = connection.await;
    });

    // Run real SQL over the TLS channel.
    client
        .simple_query("CREATE TABLE secure_t (id INTEGER PRIMARY KEY, v TEXT)")
        .await
        .expect("DDL over TLS must succeed");
    client
        .simple_query("INSERT INTO secure_t (id, v) VALUES (1, 'tls-works')")
        .await
        .expect("INSERT over TLS must succeed");
    let msgs = client
        .simple_query("SELECT id, v FROM secure_t")
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
    assert_eq!(rows[0].get("v"), Some("tls-works"));
}

/// `require` mode must reject a plaintext StartupMessage that arrives
/// without a prior `SSLRequest` (Req 2 AC3). `tokio-postgres` with
/// `NoTls` + `sslmode=disable` sends a plaintext startup, which the server
/// must refuse.
#[tokio::test]
async fn tls_require_rejects_plaintext_startup() {
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    let data_dir = tempfile::tempdir().unwrap();
    let (cert_path, key_path) = write_test_cert(data_dir.path());

    let cfg = ServerConfig {
        bind_addr: "127.0.0.1:0".to_string(),
        data_dir: data_dir.path().to_string_lossy().to_string(),
        max_connections: 16,
        sidecar_binary: None,
        model_id: None,
        tls_mode: galaxdb_wire::tls::TlsMode::Require,
        tls_cert_path: Some(cert_path),
        tls_key_path: Some(key_path),
        ..Default::default()
    };
    let (addr, _handle) = start(cfg).await.expect("tls server failed to bind");

    // sslmode=disable forces a plaintext StartupMessage with no SSLRequest.
    let conn_str = format!(
        "host=127.0.0.1 port={} user=galaxdb dbname=galaxdb sslmode=disable",
        addr.port()
    );
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        tokio_postgres::connect(&conn_str, NoTls),
    )
    .await
    .expect("connect attempt timed out");

    assert!(
        result.is_err(),
        "require mode must reject a plaintext (no-TLS) connection"
    );
}

/// `allow` mode still serves plaintext clients that skip `SSLRequest`,
/// so existing non-TLS clients keep working (backward compatibility).
#[tokio::test]
async fn tls_allow_still_serves_plaintext() {
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    let data_dir = tempfile::tempdir().unwrap();
    let (cert_path, key_path) = write_test_cert(data_dir.path());

    let cfg = ServerConfig {
        bind_addr: "127.0.0.1:0".to_string(),
        data_dir: data_dir.path().to_string_lossy().to_string(),
        max_connections: 16,
        sidecar_binary: None,
        model_id: None,
        tls_mode: galaxdb_wire::tls::TlsMode::Allow,
        tls_cert_path: Some(cert_path),
        tls_key_path: Some(key_path),
        ..Default::default()
    };
    let (addr, _handle) = start(cfg).await.expect("server failed to bind");

    let conn_str = format!(
        "host=127.0.0.1 port={} user=galaxdb dbname=galaxdb sslmode=disable",
        addr.port()
    );
    let (client, connection) = tokio::time::timeout(
        Duration::from_secs(5),
        tokio_postgres::connect(&conn_str, NoTls),
    )
    .await
    .expect("connect timed out")
    .expect("allow mode must still accept plaintext clients");
    tokio::spawn(async move {
        let _ = connection.await;
    });
    client
        .simple_query("CREATE TABLE plain_t (id INTEGER PRIMARY KEY)")
        .await
        .expect("plaintext DDL under allow mode must succeed");
}

// ---------------------------------------------------------------------------
// Req 4: security audit sink wired end-to-end (JSONL file)
// ---------------------------------------------------------------------------

/// With an audit log configured, the server records authentication
/// outcomes, authorization denials, and role/grant changes to the JSONL
/// file. This proves the AuditSink seam is actually wired into the running
/// server (auth path + executor chokepoint), not just defined.
#[tokio::test]
async fn audit_log_records_auth_authz_and_admin_events() {
    let data_dir = tempfile::tempdir().unwrap();
    let audit_path = data_dir.path().join("audit.jsonl");
    let cfg = ServerConfig {
        bind_addr: "127.0.0.1:0".to_string(),
        data_dir: data_dir.path().to_string_lossy().to_string(),
        max_connections: 16,
        sidecar_binary: None,
        model_id: None,
        auth_enabled: true,
        trusted_local_user: "galaxdb".to_string(),
        initial_superuser: Some(("root".to_string(), "rootpw".to_string())),
        audit_log_path: Some(audit_path.to_string_lossy().to_string()),
        ..Default::default()
    };
    let (addr, _handle) = start(cfg).await.expect("server failed to bind");
    let port = addr.port();

    // Superuser creates a table (DDL → Allowed admin/ddl event), a role,
    // and grants nothing yet.
    let admin_dsn =
        format!("host=127.0.0.1 port={port} user=root password=rootpw dbname=galaxdb sslmode=disable");
    let (admin, admin_conn) = tokio_postgres::connect(&admin_dsn, NoTls).await.unwrap();
    tokio::spawn(async move {
        let _ = admin_conn.await;
    });
    admin
        .simple_query("CREATE TABLE docs (id INTEGER PRIMARY KEY, body TEXT)")
        .await
        .unwrap();
    admin
        .simple_query("CREATE ROLE alice PASSWORD 'alicepw'")
        .await
        .unwrap();

    // A wrong-password login (Denied auth event).
    let bad_dsn =
        format!("host=127.0.0.1 port={port} user=alice password=WRONG dbname=galaxdb sslmode=disable");
    let _ = tokio_postgres::connect(&bad_dsn, NoTls).await;

    // alice logs in (Allowed auth event) and is denied SELECT (Denied
    // authz event).
    let alice_dsn =
        format!("host=127.0.0.1 port={port} user=alice password=alicepw dbname=galaxdb sslmode=disable");
    let (alice, alice_conn) = tokio_postgres::connect(&alice_dsn, NoTls).await.unwrap();
    tokio::spawn(async move {
        let _ = alice_conn.await;
    });
    let _ = alice.simple_query("SELECT id FROM docs").await; // denied 42501

    // Give the file writes a moment to flush (synchronous, but the bad
    // login closes async).
    tokio::time::sleep(Duration::from_millis(200)).await;

    let contents = std::fs::read_to_string(&audit_path).expect("audit log must exist");
    let events: Vec<serde_json::Value> = contents
        .lines()
        .map(|l| serde_json::from_str(l).expect("each audit line is valid JSON"))
        .collect();
    assert!(!events.is_empty(), "audit log must contain events");

    let has = |kind: &str, action: &str, outcome: &str| {
        events.iter().any(|e| {
            e["kind"] == kind && e["action"] == action && e["outcome"] == outcome
        })
    };

    // Auth: root + alice logged in (allowed); alice's wrong password denied.
    assert!(has("auth", "login", "allowed"), "expected an allowed login event");
    assert!(has("auth", "login", "denied"), "expected a denied login event");
    // Authz: DDL by superuser allowed; alice's SELECT without a grant denied.
    assert!(has("authz", "ddl", "allowed"), "expected an allowed DDL event");
    assert!(
        has("authz", "admin", "allowed"),
        "expected an allowed admin event (CREATE ROLE)"
    );
    assert!(
        has("authz", "select", "denied"),
        "expected a denied SELECT event for alice"
    );
}

// ---------------------------------------------------------------------
// Explicit transactions over the wire (HTAP Phase 5, tasks 18/19/20).
// ---------------------------------------------------------------------

/// Collect the `Row` messages from a `simple_query` result.
fn simple_rows(
    msgs: Vec<tokio_postgres::SimpleQueryMessage>,
) -> Vec<tokio_postgres::SimpleQueryRow> {
    msgs.into_iter()
        .filter_map(|m| match m {
            tokio_postgres::SimpleQueryMessage::Row(r) => Some(r),
            _ => None,
        })
        .collect()
}

/// BEGIN buffers writes with read-your-writes; COMMIT makes them durable
/// and visible to later statements; ROLLBACK discards them entirely
/// (tasks 18/20 — the core transaction lifecycle).
#[tokio::test]
async fn transaction_commit_persists_and_rollback_discards() {
    let (conn_str, _td) = start_server().await;
    let (client, connection) = tokio_postgres::connect(&conn_str, NoTls).await.unwrap();
    tokio::spawn(async move {
        let _ = connection.await;
    });

    client
        .simple_query("CREATE TABLE acct (id INTEGER PRIMARY KEY, bal INTEGER)")
        .await
        .unwrap();
    client
        .simple_query("INSERT INTO acct (id, bal) VALUES (1, 100)")
        .await
        .unwrap();

    // COMMIT path: buffered UPDATE is visible read-your-writes inside the
    // txn, then durable after COMMIT.
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("UPDATE acct SET bal = 250 WHERE id = 1")
        .await
        .unwrap();
    let rows = simple_rows(
        client
            .simple_query("SELECT bal FROM acct WHERE id = 1")
            .await
            .unwrap(),
    );
    assert_eq!(
        rows[0].get("bal"),
        Some("250"),
        "read-your-writes: the buffered UPDATE must be visible inside its txn"
    );
    client.simple_query("COMMIT").await.unwrap();

    let rows = simple_rows(
        client
            .simple_query("SELECT bal FROM acct WHERE id = 1")
            .await
            .unwrap(),
    );
    assert_eq!(rows[0].get("bal"), Some("250"), "COMMIT must persist the write");

    // ROLLBACK path: a buffered UPDATE disappears after ROLLBACK.
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("UPDATE acct SET bal = 999 WHERE id = 1")
        .await
        .unwrap();
    client.simple_query("ROLLBACK").await.unwrap();

    let rows = simple_rows(
        client
            .simple_query("SELECT bal FROM acct WHERE id = 1")
            .await
            .unwrap(),
    );
    assert_eq!(
        rows[0].get("bal"),
        Some("250"),
        "ROLLBACK must discard the buffered write (value stays 250)"
    );
}

/// Snapshot isolation across two connections: an open transaction never
/// sees another connection's writes committed after it began (no dirty
/// reads, no non-repeatable reads), and sees them only after starting a
/// fresh transaction (task 20 — snapshot → scan).
#[tokio::test]
async fn transaction_snapshot_isolation_across_connections() {
    let (conn_str, _td) = start_server().await;

    let (a, a_conn) = tokio_postgres::connect(&conn_str, NoTls).await.unwrap();
    tokio::spawn(async move {
        let _ = a_conn.await;
    });
    let (b, b_conn) = tokio_postgres::connect(&conn_str, NoTls).await.unwrap();
    tokio::spawn(async move {
        let _ = b_conn.await;
    });

    a.simple_query("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)")
        .await
        .unwrap();
    a.simple_query("INSERT INTO t (id, v) VALUES (1, 10)")
        .await
        .unwrap();

    // A begins a transaction and takes a snapshot (sees v=10).
    a.simple_query("BEGIN").await.unwrap();
    let rows = simple_rows(a.simple_query("SELECT v FROM t WHERE id = 1").await.unwrap());
    assert_eq!(rows[0].get("v"), Some("10"));

    // B (autocommit) updates the row and inserts a new one AFTER A's snapshot.
    b.simple_query("UPDATE t SET v = 20 WHERE id = 1")
        .await
        .unwrap();
    b.simple_query("INSERT INTO t (id, v) VALUES (2, 99)")
        .await
        .unwrap();

    // A must still see its snapshot: v=10 and only one row (no dirty read,
    // no non-repeatable read, no phantom).
    let rows = simple_rows(a.simple_query("SELECT v FROM t WHERE id = 1").await.unwrap());
    assert_eq!(
        rows[0].get("v"),
        Some("10"),
        "A's snapshot must not see B's committed UPDATE (non-repeatable read)"
    );
    let all = simple_rows(a.simple_query("SELECT id FROM t").await.unwrap());
    assert_eq!(
        all.len(),
        1,
        "A's snapshot must not see B's inserted row (phantom read)"
    );

    a.simple_query("COMMIT").await.unwrap();

    // After committing, a fresh statement on A sees B's changes.
    let rows = simple_rows(a.simple_query("SELECT v FROM t WHERE id = 1").await.unwrap());
    assert_eq!(rows[0].get("v"), Some("20"));
    let all = simple_rows(a.simple_query("SELECT id FROM t").await.unwrap());
    assert_eq!(all.len(), 2, "after the txn ends, A sees the current state");
}

/// Two transactions writing the same key: the second writer is rejected
/// with SQLSTATE 40001 (serialization_failure) at write-buffer time
/// (task 18 — write-write conflict detection).
#[tokio::test]
async fn transaction_write_write_conflict_40001() {
    let (conn_str, _td) = start_server().await;

    let (a, a_conn) = tokio_postgres::connect(&conn_str, NoTls).await.unwrap();
    tokio::spawn(async move {
        let _ = a_conn.await;
    });
    let (b, b_conn) = tokio_postgres::connect(&conn_str, NoTls).await.unwrap();
    tokio::spawn(async move {
        let _ = b_conn.await;
    });

    a.simple_query("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)")
        .await
        .unwrap();
    a.simple_query("INSERT INTO t (id, v) VALUES (1, 0)")
        .await
        .unwrap();

    a.simple_query("BEGIN").await.unwrap();
    b.simple_query("BEGIN").await.unwrap();

    // A acquires the write lock on row 1 by buffering an UPDATE.
    a.simple_query("UPDATE t SET v = 1 WHERE id = 1")
        .await
        .unwrap();

    // B tries to write the same row → write-write conflict → 40001.
    let err = b
        .simple_query("UPDATE t SET v = 2 WHERE id = 1")
        .await
        .expect_err("second writer must conflict");
    let code = err
        .as_db_error()
        .map(|e| e.code().code().to_string())
        .unwrap_or_default();
    assert_eq!(code, "40001", "write-write conflict must map to 40001, got {code:?} ({err})");

    // B's transaction is now aborted; ROLLBACK ends it, then A commits.
    b.simple_query("ROLLBACK").await.unwrap();
    a.simple_query("COMMIT").await.unwrap();

    let rows = simple_rows(a.simple_query("SELECT v FROM t WHERE id = 1").await.unwrap());
    assert_eq!(rows[0].get("v"), Some("1"), "A's committed write must win");
}

/// SAVEPOINT / ROLLBACK TO SAVEPOINT: rolling back to a savepoint discards
/// writes made after it while keeping earlier ones; the transaction then
/// commits the surviving writes (task 19 — nested write-set markers).
#[tokio::test]
async fn transaction_savepoint_rollback_to() {
    let (conn_str, _td) = start_server().await;
    let (client, connection) = tokio_postgres::connect(&conn_str, NoTls).await.unwrap();
    tokio::spawn(async move {
        let _ = connection.await;
    });

    client
        .simple_query("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)")
        .await
        .unwrap();

    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("INSERT INTO t (id, v) VALUES (1, 1)")
        .await
        .unwrap();
    client.simple_query("SAVEPOINT sp1").await.unwrap();
    client
        .simple_query("INSERT INTO t (id, v) VALUES (2, 2)")
        .await
        .unwrap();

    // Inside the txn, both rows are visible (read-your-writes).
    let all = simple_rows(client.simple_query("SELECT id FROM t").await.unwrap());
    assert_eq!(all.len(), 2, "both buffered inserts visible before rollback-to");

    // Roll back to sp1: row 2 is discarded, row 1 survives.
    client.simple_query("ROLLBACK TO SAVEPOINT sp1").await.unwrap();
    let all = simple_rows(client.simple_query("SELECT id FROM t").await.unwrap());
    assert_eq!(all.len(), 1, "ROLLBACK TO must discard writes after the savepoint");

    client.simple_query("COMMIT").await.unwrap();

    // After commit only row 1 is durable.
    let all = simple_rows(client.simple_query("SELECT id FROM t").await.unwrap());
    assert_eq!(all.len(), 1);
    let rows = simple_rows(client.simple_query("SELECT v FROM t WHERE id = 1").await.unwrap());
    assert_eq!(rows[0].get("v"), Some("1"));
}
