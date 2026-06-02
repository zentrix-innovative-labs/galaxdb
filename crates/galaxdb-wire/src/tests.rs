//! Tests for the PostgreSQL wire protocol.

use crate::messages::*;
use std::io::Cursor;

// ── Message encoding tests ─────────────────────────────────────────

#[tokio::test]
async fn write_auth_ok_produces_correct_bytes() {
    let mut buf = Vec::new();
    write_auth_ok(&mut buf).await.unwrap();
    // R(1) + len(4) + auth_ok(4) = 9 bytes
    assert_eq!(buf.len(), 9);
    assert_eq!(buf[0], b'R');
}

#[tokio::test]
async fn write_parameter_status_produces_correct_bytes() {
    let mut buf = Vec::new();
    write_parameter_status(&mut buf, "server_version", "16.0").await.unwrap();
    assert_eq!(buf[0], b'S');
    // Should contain both key and value null-terminated
    assert!(buf.windows(14).any(|w| w == b"server_version"));
}

#[tokio::test]
async fn write_backend_key_data_produces_correct_bytes() {
    let mut buf = Vec::new();
    write_backend_key_data(&mut buf, 12345, 67890).await.unwrap();
    assert_eq!(buf[0], b'K');
    assert_eq!(buf.len(), 13); // K(1) + len(4) + pid(4) + key(4)
}

#[tokio::test]
async fn write_ready_for_query_idle() {
    let mut buf = Vec::new();
    write_ready_for_query(&mut buf, b'I').await.unwrap();
    assert_eq!(buf[0], b'Z');
    assert_eq!(buf.len(), 6); // Z(1) + len(4) + status(1)
    assert_eq!(buf[5], b'I');
}

#[tokio::test]
async fn write_row_description_single_column() {
    let mut buf = Vec::new();
    let cols = vec![ColumnDesc::text("name")];
    write_row_description(&mut buf, &cols).await.unwrap();
    assert_eq!(buf[0], b'T');
}

#[tokio::test]
async fn write_row_description_multiple_columns() {
    let mut buf = Vec::new();
    let cols = vec![
        ColumnDesc::int4("id"),
        ColumnDesc::text("name"),
        ColumnDesc::text("email"),
    ];
    write_row_description(&mut buf, &cols).await.unwrap();
    assert_eq!(buf[0], b'T');
}

#[tokio::test]
async fn write_data_row_with_values() {
    let mut buf = Vec::new();
    let values: Vec<Option<&str>> = vec![Some("1"), Some("alice"), None];
    write_data_row(&mut buf, &values).await.unwrap();
    assert_eq!(buf[0], b'D');
}

#[tokio::test]
async fn write_command_complete_select() {
    let mut buf = Vec::new();
    write_command_complete(&mut buf, "SELECT 5").await.unwrap();
    assert_eq!(buf[0], b'C');
    // Should contain the tag
    assert!(buf.windows(8).any(|w| w == b"SELECT 5"));
}

#[tokio::test]
async fn write_error_response_contains_sqlstate() {
    let mut buf = Vec::new();
    write_error_response(&mut buf, "42601", "syntax error").await.unwrap();
    assert_eq!(buf[0], b'E');
    // Should contain the SQLSTATE code
    assert!(buf.windows(5).any(|w| w == b"42601"));
}

// ── SASL / SCRAM message encoding + decoding (task 6) ──────────────

#[tokio::test]
async fn write_auth_sasl_advertises_mechanism() {
    let mut buf = Vec::new();
    write_auth_sasl(&mut buf, &["SCRAM-SHA-256"]).await.unwrap();
    assert_eq!(buf[0], b'R');
    // body code 10 (SASL) follows the 4-byte length.
    assert_eq!(&buf[5..9], &10i32.to_be_bytes());
    // mechanism name is present, null-terminated, with a final list terminator.
    assert!(buf.windows(13).any(|w| w == b"SCRAM-SHA-256"));
    assert_eq!(*buf.last().unwrap(), 0, "mechanism list must end with a 0 byte");
}

#[tokio::test]
async fn write_auth_sasl_continue_and_final_carry_code_and_data() {
    let mut cont = Vec::new();
    write_auth_sasl_continue(&mut cont, b"r=abc,s=salt,i=4096").await.unwrap();
    assert_eq!(cont[0], b'R');
    assert_eq!(&cont[5..9], &11i32.to_be_bytes()); // SASL continue
    assert!(cont.windows(2).any(|w| w == b"r="));

    let mut fin = Vec::new();
    write_auth_sasl_final(&mut fin, b"v=serversig").await.unwrap();
    assert_eq!(fin[0], b'R');
    assert_eq!(&fin[5..9], &12i32.to_be_bytes()); // SASL final
    assert!(fin.windows(2).any(|w| w == b"v="));
}

#[tokio::test]
async fn read_sasl_initial_response_round_trip() {
    // Build a frontend `p` SASLInitialResponse: mechanism + Int32 len + data.
    let mechanism = b"SCRAM-SHA-256\0";
    let payload = b"n,,n=,r=clientnonce";
    let mut body = Vec::new();
    body.extend_from_slice(mechanism);
    body.extend_from_slice(&(payload.len() as i32).to_be_bytes());
    body.extend_from_slice(payload);

    let mut msg = Vec::new();
    msg.push(b'p');
    msg.extend_from_slice(&((body.len() + 4) as i32).to_be_bytes());
    msg.extend_from_slice(&body);

    let mut cur = Cursor::new(msg);
    let parsed = read_sasl_initial_response(&mut cur).await.unwrap();
    assert_eq!(parsed.mechanism, "SCRAM-SHA-256");
    assert_eq!(parsed.initial_response, payload);
}

#[tokio::test]
async fn read_sasl_initial_response_handles_no_initial_data() {
    // resp_len == -1 means "no initial response".
    let mechanism = b"SCRAM-SHA-256\0";
    let mut body = Vec::new();
    body.extend_from_slice(mechanism);
    body.extend_from_slice(&(-1i32).to_be_bytes());

    let mut msg = Vec::new();
    msg.push(b'p');
    msg.extend_from_slice(&((body.len() + 4) as i32).to_be_bytes());
    msg.extend_from_slice(&body);

    let mut cur = Cursor::new(msg);
    let parsed = read_sasl_initial_response(&mut cur).await.unwrap();
    assert_eq!(parsed.mechanism, "SCRAM-SHA-256");
    assert!(parsed.initial_response.is_empty());
}

#[tokio::test]
async fn read_sasl_response_round_trip() {
    let payload = b"c=biws,r=combined,p=cHJvb2Y=";
    let mut msg = Vec::new();
    msg.push(b'p');
    msg.extend_from_slice(&((payload.len() + 4) as i32).to_be_bytes());
    msg.extend_from_slice(payload);

    let mut cur = Cursor::new(msg);
    let data = read_sasl_response(&mut cur).await.unwrap();
    assert_eq!(data, payload);
}

#[tokio::test]
async fn read_sasl_initial_response_rejects_wrong_message_type() {
    // A 'Q' message where a 'p' is expected must error, not misparse.
    let mut msg = Vec::new();
    msg.push(b'Q');
    msg.extend_from_slice(&8i32.to_be_bytes());
    msg.extend_from_slice(b"foo\0");
    let mut cur = Cursor::new(msg);
    assert!(read_sasl_initial_response(&mut cur).await.is_err());
}

// ── Startup message parsing ────────────────────────────────────────

#[tokio::test]
async fn read_startup_message() {
    // Build a startup message: len(4) + version(4) + "user\0postgres\0\0"
    let mut msg = Vec::new();
    let params = b"user\0postgres\0database\0galaxdb\0\0";
    let len = (8 + params.len()) as i32;
    msg.extend_from_slice(&len.to_be_bytes());
    msg.extend_from_slice(&PROTOCOL_VERSION.to_be_bytes());
    msg.extend_from_slice(params);

    let mut cursor = Cursor::new(msg);
    let startup = read_startup(&mut cursor).await.unwrap();

    assert_eq!(startup.protocol_version, PROTOCOL_VERSION);
    assert!(startup.params.iter().any(|(k, v)| k == "user" && v == "postgres"));
    assert!(startup.params.iter().any(|(k, v)| k == "database" && v == "galaxdb"));
}

// ── Query message parsing ──────────────────────────────────────────

#[tokio::test]
async fn read_query_message() {
    let sql = "SELECT 1";
    let mut msg = Vec::new();
    msg.push(b'Q');
    let len = (4 + sql.len() + 1) as i32; // +1 for null terminator
    msg.extend_from_slice(&len.to_be_bytes());
    msg.extend_from_slice(sql.as_bytes());
    msg.push(0);

    let mut cursor = Cursor::new(msg);
    let query = read_query(&mut cursor).await.unwrap();
    assert_eq!(query, "SELECT 1");
}

#[tokio::test]
async fn read_query_wrong_message_type() {
    let mut msg = Vec::new();
    msg.push(b'X'); // wrong type
    msg.extend_from_slice(&8i32.to_be_bytes());
    msg.extend_from_slice(b"test");

    let mut cursor = Cursor::new(msg);
    let result = read_query(&mut cursor).await;
    assert!(result.is_err());
}

// ── Column descriptor helpers ──────────────────────────────────────

#[test]
fn column_desc_text_has_correct_oid() {
    let col = ColumnDesc::text("name");
    assert_eq!(col.type_oid, 25); // TEXT
    assert_eq!(col.format_code, 0); // text format
}

#[test]
fn column_desc_int4_has_correct_oid() {
    let col = ColumnDesc::int4("id");
    assert_eq!(col.type_oid, 23); // INT4
    assert_eq!(col.type_size, 4);
}

// ── Connection limit ───────────────────────────────────────────────

#[test]
fn wire_server_tracks_connection_count() {
    let server = crate::server::WireServer::new(crate::server::WireServerConfig {
        listen_addr: "127.0.0.1:0".to_string(),
        max_connections: 100,
    });
    assert_eq!(server.active_connections(), 0);
    assert_eq!(server.max_connections(), 100);
}

// ── Full handshake simulation ──────────────────────────────────────

#[tokio::test]
async fn full_startup_handshake_response() {
    let mut buf = Vec::new();

    // Simulate server sending the full startup response
    write_auth_ok(&mut buf).await.unwrap();
    write_parameter_status(&mut buf, "server_version", "16.0.0-galaxdb").await.unwrap();
    write_parameter_status(&mut buf, "server_encoding", "UTF8").await.unwrap();
    write_backend_key_data(&mut buf, 1234, 5678).await.unwrap();
    write_ready_for_query(&mut buf, b'I').await.unwrap();

    // Verify the response contains all expected message types
    let msg_types: Vec<u8> = extract_message_types(&buf);
    assert!(msg_types.contains(&b'R')); // AuthenticationOk
    assert!(msg_types.contains(&b'S')); // ParameterStatus
    assert!(msg_types.contains(&b'K')); // BackendKeyData
    assert!(msg_types.contains(&b'Z')); // ReadyForQuery
}

#[tokio::test]
async fn query_response_flow() {
    let mut buf = Vec::new();

    // Simulate a SELECT response
    let cols = vec![ColumnDesc::int4("id"), ColumnDesc::text("name")];
    write_row_description(&mut buf, &cols).await.unwrap();
    write_data_row(&mut buf, &[Some("1"), Some("alice")]).await.unwrap();
    write_data_row(&mut buf, &[Some("2"), Some("bob")]).await.unwrap();
    write_command_complete(&mut buf, "SELECT 2").await.unwrap();
    write_ready_for_query(&mut buf, b'I').await.unwrap();

    let msg_types = extract_message_types(&buf);
    assert_eq!(msg_types[0], b'T'); // RowDescription
    assert_eq!(msg_types[1], b'D'); // DataRow
    assert_eq!(msg_types[2], b'D'); // DataRow
    assert_eq!(msg_types[3], b'C'); // CommandComplete
    assert_eq!(msg_types[4], b'Z'); // ReadyForQuery
}

#[tokio::test]
async fn error_response_flow() {
    let mut buf = Vec::new();

    write_error_response(&mut buf, "53300", "too many connections").await.unwrap();
    write_ready_for_query(&mut buf, b'I').await.unwrap();

    let msg_types = extract_message_types(&buf);
    assert_eq!(msg_types[0], b'E'); // ErrorResponse
    assert_eq!(msg_types[1], b'Z'); // ReadyForQuery
}

/// Extract message type bytes from a buffer of PostgreSQL messages.
fn extract_message_types(buf: &[u8]) -> Vec<u8> {
    let mut types = Vec::new();
    let mut pos = 0;
    while pos < buf.len() {
        let msg_type = buf[pos];
        types.push(msg_type);
        pos += 1;
        if pos + 4 > buf.len() {
            break;
        }
        let len = i32::from_be_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]) as usize;
        pos += len;
    }
    types
}
