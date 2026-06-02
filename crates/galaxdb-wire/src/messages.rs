//! PostgreSQL wire protocol message encoding/decoding.
//!
//! Implements the simple query protocol (Q message flow):
//! StartupMessage → AuthenticationOk → ParameterStatus → BackendKeyData →
//! ReadyForQuery → Query → RowDescription → DataRow → CommandComplete → ReadyForQuery

use std::io;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// PostgreSQL protocol version 3.0.
pub const PROTOCOL_VERSION: i32 = 196608; // 3 << 16 | 0

// ── Frontend (client → server) messages ────────────────────────────

/// Parsed startup message from the client.
#[derive(Debug, Clone)]
pub struct StartupMessage {
    pub protocol_version: i32,
    pub params: Vec<(String, String)>,
}

/// Read a startup message from the client.
pub async fn read_startup<R: AsyncReadExt + Unpin>(reader: &mut R) -> io::Result<StartupMessage> {
    let len = reader.read_i32().await? as usize;
    if !(8..=10240).contains(&len) {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid startup message length"));
    }

    let version = reader.read_i32().await?;

    // Read remaining bytes as null-terminated key-value pairs
    let remaining = len - 8;
    let mut buf = vec![0u8; remaining];
    reader.read_exact(&mut buf).await?;

    let mut params = Vec::new();
    let mut iter = buf.split(|&b| b == 0).map(|s| String::from_utf8_lossy(s).to_string());

    loop {
        let key = match iter.next() {
            Some(k) if !k.is_empty() => k,
            _ => break,
        };
        let value = iter.next().unwrap_or_default();
        params.push((key, value));
    }

    Ok(StartupMessage {
        protocol_version: version,
        params,
    })
}

/// Read a query message (Q) from the client.
/// Returns the SQL string.
pub async fn read_query<R: AsyncReadExt + Unpin>(reader: &mut R) -> io::Result<String> {
    let msg_type = reader.read_u8().await?;
    if msg_type != b'Q' {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("expected Q message, got '{}'", msg_type as char),
        ));
    }

    let len = reader.read_i32().await? as usize;
    if !(5..=10_000_000).contains(&len) {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid query length"));
    }

    let mut buf = vec![0u8; len - 4]; // len includes itself
    reader.read_exact(&mut buf).await?;

    // Remove trailing null byte
    if buf.last() == Some(&0) {
        buf.pop();
    }

    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

// ── Backend (server → client) messages ─────────────────────────────

/// Write AuthenticationOk (R) message.
pub async fn write_auth_ok<W: AsyncWriteExt + Unpin>(writer: &mut W) -> io::Result<()> {
    writer.write_u8(b'R').await?;
    writer.write_i32(8).await?; // length
    writer.write_i32(0).await?; // auth ok
    Ok(())
}

/// Write AuthenticationSASL (R, code 10): advertise the SASL mechanisms
/// the server offers. Body layout (after the `R` byte + length):
/// `Int32(10)` then each mechanism name null-terminated, then a final
/// zero byte terminating the list (PostgreSQL frontend/backend protocol).
pub async fn write_auth_sasl<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
    mechanisms: &[&str],
) -> io::Result<()> {
    let mut body = Vec::new();
    body.extend_from_slice(&10i32.to_be_bytes()); // SASL
    for m in mechanisms {
        body.extend_from_slice(m.as_bytes());
        body.push(0);
    }
    body.push(0); // terminate the mechanism list
    writer.write_u8(b'R').await?;
    writer.write_i32((body.len() + 4) as i32).await?;
    writer.write_all(&body).await?;
    Ok(())
}

/// Write AuthenticationSASLContinue (R, code 11): carries the server's
/// SASL challenge (the SCRAM `server-first-message`).
pub async fn write_auth_sasl_continue<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
    data: &[u8],
) -> io::Result<()> {
    writer.write_u8(b'R').await?;
    writer.write_i32((data.len() + 8) as i32).await?;
    writer.write_i32(11).await?; // SASL continue
    writer.write_all(data).await?;
    Ok(())
}

/// Write AuthenticationSASLFinal (R, code 12): carries the SCRAM
/// `server-final-message` (the server signature) sent just before
/// AuthenticationOk.
pub async fn write_auth_sasl_final<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
    data: &[u8],
) -> io::Result<()> {
    writer.write_u8(b'R').await?;
    writer.write_i32((data.len() + 8) as i32).await?;
    writer.write_i32(12).await?; // SASL final
    writer.write_all(data).await?;
    Ok(())
}

/// A frontend SASL `p` message split into its mechanism (only present on
/// the initial response) and the SASL payload bytes.
#[derive(Debug, Clone)]
pub struct SaslInitialResponse {
    /// The SASL mechanism the client selected (e.g. `SCRAM-SHA-256`).
    pub mechanism: String,
    /// The client's initial SASL response (the SCRAM `client-first-message`).
    pub initial_response: Vec<u8>,
}

/// Read a frontend SASLInitialResponse (`p`) message. Layout after the
/// `p` byte + length: a null-terminated mechanism name, an `Int32`
/// initial-response length (`-1` = none), then that many payload bytes.
pub async fn read_sasl_initial_response<R: AsyncReadExt + Unpin>(
    reader: &mut R,
) -> io::Result<SaslInitialResponse> {
    let msg_type = reader.read_u8().await?;
    if msg_type != b'p' {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("expected SASL 'p' message, got '{}'", msg_type as char),
        ));
    }
    let len = reader.read_i32().await? as usize;
    if !(5..=1_000_000).contains(&len) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid SASL initial-response length",
        ));
    }
    let mut buf = vec![0u8; len - 4];
    reader.read_exact(&mut buf).await?;

    // Null-terminated mechanism name.
    let nul = buf
        .iter()
        .position(|&b| b == 0)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "SASL: unterminated mechanism"))?;
    let mechanism = String::from_utf8_lossy(&buf[..nul]).to_string();
    let rest = &buf[nul + 1..];
    if rest.len() < 4 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "SASL: missing initial-response length",
        ));
    }
    let resp_len = i32::from_be_bytes([rest[0], rest[1], rest[2], rest[3]]);
    let initial_response = if resp_len < 0 {
        Vec::new()
    } else {
        let n = resp_len as usize;
        if rest.len() < 4 + n {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "SASL: initial-response shorter than declared",
            ));
        }
        rest[4..4 + n].to_vec()
    };
    Ok(SaslInitialResponse {
        mechanism,
        initial_response,
    })
}

/// Read a frontend SASLResponse (`p`) message. The entire payload after
/// the `p` byte + length is the SASL data (the SCRAM
/// `client-final-message`).
pub async fn read_sasl_response<R: AsyncReadExt + Unpin>(reader: &mut R) -> io::Result<Vec<u8>> {
    let msg_type = reader.read_u8().await?;
    if msg_type != b'p' {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("expected SASL 'p' message, got '{}'", msg_type as char),
        ));
    }
    let len = reader.read_i32().await? as usize;
    if !(4..=1_000_000).contains(&len) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid SASL response length",
        ));
    }
    let mut buf = vec![0u8; len - 4];
    reader.read_exact(&mut buf).await?;
    Ok(buf)
}

/// Write a ParameterStatus (S) message.
pub async fn write_parameter_status<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
    key: &str,
    value: &str,
) -> io::Result<()> {
    let len = 4 + key.len() + 1 + value.len() + 1;
    writer.write_u8(b'S').await?;
    writer.write_i32(len as i32).await?;
    writer.write_all(key.as_bytes()).await?;
    writer.write_u8(0).await?;
    writer.write_all(value.as_bytes()).await?;
    writer.write_u8(0).await?;
    Ok(())
}

/// Write BackendKeyData (K) message.
pub async fn write_backend_key_data<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
    process_id: i32,
    secret_key: i32,
) -> io::Result<()> {
    writer.write_u8(b'K').await?;
    writer.write_i32(12).await?;
    writer.write_i32(process_id).await?;
    writer.write_i32(secret_key).await?;
    Ok(())
}

/// Write ReadyForQuery (Z) message.
pub async fn write_ready_for_query<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
    status: u8, // 'I' = idle, 'T' = in transaction, 'E' = error
) -> io::Result<()> {
    writer.write_u8(b'Z').await?;
    writer.write_i32(5).await?;
    writer.write_u8(status).await?;
    Ok(())
}

/// Write RowDescription (T) message.
pub async fn write_row_description<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
    columns: &[ColumnDesc],
) -> io::Result<()> {
    // Calculate total length
    let mut body_len = 2i32; // field count (i16)
    for col in columns {
        body_len += col.name.len() as i32 + 1 + 18; // name + null + fixed fields
    }

    writer.write_u8(b'T').await?;
    writer.write_i32(body_len + 4).await?; // +4 for length field itself
    writer.write_i16(columns.len() as i16).await?;

    for col in columns {
        writer.write_all(col.name.as_bytes()).await?;
        writer.write_u8(0).await?;
        writer.write_i32(col.table_oid).await?;
        writer.write_i16(col.column_id).await?;
        writer.write_i32(col.type_oid).await?;
        writer.write_i16(col.type_size).await?;
        writer.write_i32(col.type_modifier).await?;
        writer.write_i16(col.format_code).await?; // 0 = text
    }

    Ok(())
}

/// Write a DataRow (D) message.
pub async fn write_data_row<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
    values: &[Option<&str>],
) -> io::Result<()> {
    let mut body_len = 2i32; // field count
    for val in values {
        body_len += 4; // length prefix
        if let Some(v) = val {
            body_len += v.len() as i32;
        }
    }

    writer.write_u8(b'D').await?;
    writer.write_i32(body_len + 4).await?;
    writer.write_i16(values.len() as i16).await?;

    for val in values {
        match val {
            Some(v) => {
                writer.write_i32(v.len() as i32).await?;
                writer.write_all(v.as_bytes()).await?;
            }
            None => {
                writer.write_i32(-1).await?; // NULL
            }
        }
    }

    Ok(())
}

/// Write CommandComplete (C) message.
pub async fn write_command_complete<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
    tag: &str,
) -> io::Result<()> {
    let len = 4 + tag.len() + 1;
    writer.write_u8(b'C').await?;
    writer.write_i32(len as i32).await?;
    writer.write_all(tag.as_bytes()).await?;
    writer.write_u8(0).await?;
    Ok(())
}

/// Write ErrorResponse (E) message.
pub async fn write_error_response<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
    sqlstate: &str,
    message: &str,
) -> io::Result<()> {
    // Fields: S (severity), V (severity non-localized), C (code), M (message)
    let mut body = Vec::new();
    body.push(b'S');
    body.extend_from_slice(b"ERROR\0");
    body.push(b'V');
    body.extend_from_slice(b"ERROR\0");
    body.push(b'C');
    body.extend_from_slice(sqlstate.as_bytes());
    body.push(0);
    body.push(b'M');
    body.extend_from_slice(message.as_bytes());
    body.push(0);
    body.push(0); // terminator

    writer.write_u8(b'E').await?;
    writer.write_i32((body.len() + 4) as i32).await?;
    writer.write_all(&body).await?;
    Ok(())
}

/// Column descriptor for RowDescription.
#[derive(Debug, Clone)]
pub struct ColumnDesc {
    pub name: String,
    pub table_oid: i32,
    pub column_id: i16,
    pub type_oid: i32,
    pub type_size: i16,
    pub type_modifier: i32,
    pub format_code: i16, // 0 = text
}

impl ColumnDesc {
    /// Create a text column descriptor.
    pub fn text(name: &str) -> Self {
        Self {
            name: name.to_string(),
            table_oid: 0,
            column_id: 0,
            type_oid: 25, // TEXT OID
            type_size: -1,
            type_modifier: -1,
            format_code: 0,
        }
    }

    /// Create an integer column descriptor.
    pub fn int4(name: &str) -> Self {
        Self {
            name: name.to_string(),
            table_oid: 0,
            column_id: 0,
            type_oid: 23, // INT4 OID
            type_size: 4,
            type_modifier: -1,
            format_code: 0,
        }
    }
}
