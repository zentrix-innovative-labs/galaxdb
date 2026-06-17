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
    read_startup_body(reader, len, version).await
}

/// Parse a StartupMessage when the 8-byte head (length + protocol
/// version) has already been read off the wire — used after the TLS/SSL
/// prologue peek in [`crate::tls::peek_ssl_request`], which consumes those
/// bytes to tell an `SSLRequest` from a real StartupMessage.
pub async fn read_startup_after_head<R: AsyncReadExt + Unpin>(
    reader: &mut R,
    length: i32,
    version: i32,
) -> io::Result<StartupMessage> {
    let len = length as usize;
    if !(8..=10240).contains(&len) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid startup message length",
        ));
    }
    read_startup_body(reader, len, version).await
}

/// Shared body parser for a StartupMessage: reads `len - 8` bytes of
/// null-terminated key/value parameter pairs.
async fn read_startup_body<R: AsyncReadExt + Unpin>(
    reader: &mut R,
    len: usize,
    version: i32,
) -> io::Result<StartupMessage> {
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

// ── Extended query protocol (Req 6) ────────────────────────────────
//
// The simple-query loop reads only `Q`. The extended protocol adds a
// message dispatcher that reads a one-byte type tag + Int32 length +
// body, and routes Parse/Bind/Describe/Execute/Close/Sync/Flush. To keep
// the dispatch in one place we read the whole frame into a buffer and
// parse the body synchronously here, rather than scattering async reads.

/// A decoded frontend (client → server) message. Covers both the simple
/// (`Query`) and extended (`Parse`/`Bind`/...) protocols plus `Terminate`.
#[derive(Debug, Clone, PartialEq)]
pub enum FrontendMessage {
    /// Simple query (`Q`).
    Query(String),
    /// Parse (`P`): prepare a named (or unnamed) statement.
    Parse {
        statement: String,
        query: String,
        /// Parameter type OIDs the client specified (0 = unspecified).
        param_types: Vec<i32>,
    },
    /// Bind (`B`): bind parameter values to a prepared statement → portal.
    Bind {
        portal: String,
        statement: String,
        /// Per-parameter format codes (0 = text, 1 = binary). Length is
        /// either 0 (all text), 1 (applies to all), or one per parameter.
        param_formats: Vec<i16>,
        /// Parameter values; `None` is SQL NULL.
        params: Vec<Option<Vec<u8>>>,
        /// Per-result-column format codes (0 = text, 1 = binary).
        result_formats: Vec<i16>,
    },
    /// Describe (`D`): describe a statement (`S`) or portal (`P`).
    Describe { kind: u8, name: String },
    /// Execute (`E`): run a portal, up to `max_rows` (0 = unlimited).
    Execute { portal: String, max_rows: i32 },
    /// Close (`C`): drop a statement (`S`) or portal (`P`).
    Close { kind: u8, name: String },
    /// Sync (`S`): end the current series, flush, send ReadyForQuery.
    Sync,
    /// Flush (`H`): flush pending output without ending the series.
    Flush,
    /// Terminate (`X`): client is closing the connection.
    Terminate,
}

fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

/// Read a null-terminated string starting at `*pos` in `buf`, advancing
/// `*pos` past the terminator.
fn read_cstr(buf: &[u8], pos: &mut usize) -> io::Result<String> {
    let start = *pos;
    while *pos < buf.len() && buf[*pos] != 0 {
        *pos += 1;
    }
    if *pos >= buf.len() {
        return Err(invalid("unterminated C string in message"));
    }
    let s = String::from_utf8_lossy(&buf[start..*pos]).to_string();
    *pos += 1; // skip the null
    Ok(s)
}

fn read_i16(buf: &[u8], pos: &mut usize) -> io::Result<i16> {
    if *pos + 2 > buf.len() {
        return Err(invalid("truncated i16 in message"));
    }
    let v = i16::from_be_bytes([buf[*pos], buf[*pos + 1]]);
    *pos += 2;
    Ok(v)
}

fn read_i32_at(buf: &[u8], pos: &mut usize) -> io::Result<i32> {
    if *pos + 4 > buf.len() {
        return Err(invalid("truncated i32 in message"));
    }
    let v = i32::from_be_bytes([buf[*pos], buf[*pos + 1], buf[*pos + 2], buf[*pos + 3]]);
    *pos += 4;
    Ok(v)
}

/// Read and decode one frontend message: a one-byte type tag, an `Int32`
/// length (inclusive of itself), and the body. Returns `UnexpectedEof`
/// when the client has closed the connection cleanly between messages.
pub async fn read_message<R: AsyncReadExt + Unpin>(
    reader: &mut R,
) -> io::Result<FrontendMessage> {
    let tag = reader.read_u8().await?;
    let len = reader.read_i32().await?;
    if !(4..=1_073_741_824).contains(&len) {
        return Err(invalid("invalid message length"));
    }
    let body_len = (len - 4) as usize;
    let mut buf = vec![0u8; body_len];
    reader.read_exact(&mut buf).await?;
    decode_frontend(tag, &buf)
}

/// Decode a frontend message body given its type tag. Split out from
/// [`read_message`] so it is unit-testable without a socket.
pub fn decode_frontend(tag: u8, buf: &[u8]) -> io::Result<FrontendMessage> {
    let mut pos = 0usize;
    match tag {
        b'Q' => {
            // Null-terminated query text.
            let mut end = buf.len();
            if end > 0 && buf[end - 1] == 0 {
                end -= 1;
            }
            Ok(FrontendMessage::Query(
                String::from_utf8_lossy(&buf[..end]).to_string(),
            ))
        }
        b'P' => {
            let statement = read_cstr(buf, &mut pos)?;
            let query = read_cstr(buf, &mut pos)?;
            let n = read_i16(buf, &mut pos)?;
            if n < 0 {
                return Err(invalid("negative parameter-type count in Parse"));
            }
            let mut param_types = Vec::with_capacity(n as usize);
            for _ in 0..n {
                param_types.push(read_i32_at(buf, &mut pos)?);
            }
            Ok(FrontendMessage::Parse {
                statement,
                query,
                param_types,
            })
        }
        b'B' => {
            let portal = read_cstr(buf, &mut pos)?;
            let statement = read_cstr(buf, &mut pos)?;
            let nf = read_i16(buf, &mut pos)?;
            if nf < 0 {
                return Err(invalid("negative format-code count in Bind"));
            }
            let mut param_formats = Vec::with_capacity(nf as usize);
            for _ in 0..nf {
                param_formats.push(read_i16(buf, &mut pos)?);
            }
            let np = read_i16(buf, &mut pos)?;
            if np < 0 {
                return Err(invalid("negative parameter count in Bind"));
            }
            let mut params = Vec::with_capacity(np as usize);
            for _ in 0..np {
                let plen = read_i32_at(buf, &mut pos)?;
                if plen < 0 {
                    params.push(None); // SQL NULL
                } else {
                    let n = plen as usize;
                    if pos + n > buf.len() {
                        return Err(invalid("Bind parameter value overruns message"));
                    }
                    params.push(Some(buf[pos..pos + n].to_vec()));
                    pos += n;
                }
            }
            let nr = read_i16(buf, &mut pos)?;
            if nr < 0 {
                return Err(invalid("negative result-format count in Bind"));
            }
            let mut result_formats = Vec::with_capacity(nr as usize);
            for _ in 0..nr {
                result_formats.push(read_i16(buf, &mut pos)?);
            }
            Ok(FrontendMessage::Bind {
                portal,
                statement,
                param_formats,
                params,
                result_formats,
            })
        }
        b'D' => {
            if buf.is_empty() {
                return Err(invalid("empty Describe message"));
            }
            let kind = buf[0];
            pos = 1;
            let name = read_cstr(buf, &mut pos)?;
            Ok(FrontendMessage::Describe { kind, name })
        }
        b'E' => {
            let portal = read_cstr(buf, &mut pos)?;
            let max_rows = read_i32_at(buf, &mut pos)?;
            Ok(FrontendMessage::Execute { portal, max_rows })
        }
        b'C' => {
            if buf.is_empty() {
                return Err(invalid("empty Close message"));
            }
            let kind = buf[0];
            pos = 1;
            let name = read_cstr(buf, &mut pos)?;
            Ok(FrontendMessage::Close { kind, name })
        }
        b'S' => Ok(FrontendMessage::Sync),
        b'H' => Ok(FrontendMessage::Flush),
        b'X' => Ok(FrontendMessage::Terminate),
        other => Err(invalid(&format!(
            "unsupported frontend message tag '{}' (0x{other:02x})",
            other as char
        ))),
    }
}

// ── Extended-protocol backend replies ───────────────────────────────

/// Write ParseComplete (`1`).
pub async fn write_parse_complete<W: AsyncWriteExt + Unpin>(writer: &mut W) -> io::Result<()> {
    writer.write_u8(b'1').await?;
    writer.write_i32(4).await?;
    Ok(())
}

/// Write BindComplete (`2`).
pub async fn write_bind_complete<W: AsyncWriteExt + Unpin>(writer: &mut W) -> io::Result<()> {
    writer.write_u8(b'2').await?;
    writer.write_i32(4).await?;
    Ok(())
}

/// Write CloseComplete (`3`).
pub async fn write_close_complete<W: AsyncWriteExt + Unpin>(writer: &mut W) -> io::Result<()> {
    writer.write_u8(b'3').await?;
    writer.write_i32(4).await?;
    Ok(())
}

/// Write NoData (`n`) — the response to Describe for a statement that
/// returns no rows (INSERT/UPDATE/DELETE/DDL).
pub async fn write_no_data<W: AsyncWriteExt + Unpin>(writer: &mut W) -> io::Result<()> {
    writer.write_u8(b'n').await?;
    writer.write_i32(4).await?;
    Ok(())
}

/// Write PortalSuspended (`s`) — sent when an Execute hit its row limit.
pub async fn write_portal_suspended<W: AsyncWriteExt + Unpin>(writer: &mut W) -> io::Result<()> {
    writer.write_u8(b's').await?;
    writer.write_i32(4).await?;
    Ok(())
}

/// Write EmptyQueryResponse (`I`) — the response to an empty query string.
pub async fn write_empty_query_response<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
) -> io::Result<()> {
    writer.write_u8(b'I').await?;
    writer.write_i32(4).await?;
    Ok(())
}

/// Write ParameterDescription (`t`): the type OID of each parameter.
pub async fn write_parameter_description<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
    param_type_oids: &[i32],
) -> io::Result<()> {
    let body_len = 2 + param_type_oids.len() * 4;
    writer.write_u8(b't').await?;
    writer.write_i32((body_len + 4) as i32).await?;
    writer.write_i16(param_type_oids.len() as i16).await?;
    for oid in param_type_oids {
        writer.write_i32(*oid).await?;
    }
    Ok(())
}

#[cfg(test)]
mod extended_tests {
    use super::*;

    #[test]
    fn decode_simple_query() {
        let body = b"SELECT 1\0";
        match decode_frontend(b'Q', body).unwrap() {
            FrontendMessage::Query(q) => assert_eq!(q, "SELECT 1"),
            other => panic!("expected Query, got {other:?}"),
        }
    }

    #[test]
    fn decode_parse_with_params() {
        // statement="s1", query="SELECT $1", 1 param type OID = 23 (int4)
        let mut body = Vec::new();
        body.extend_from_slice(b"s1\0");
        body.extend_from_slice(b"SELECT $1\0");
        body.extend_from_slice(&1i16.to_be_bytes());
        body.extend_from_slice(&23i32.to_be_bytes());
        match decode_frontend(b'P', &body).unwrap() {
            FrontendMessage::Parse {
                statement,
                query,
                param_types,
            } => {
                assert_eq!(statement, "s1");
                assert_eq!(query, "SELECT $1");
                assert_eq!(param_types, vec![23]);
            }
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    #[test]
    fn decode_bind_text_params_and_null() {
        // portal="", statement="s1", 0 format codes, 2 params ["7", NULL],
        // 0 result format codes.
        let mut body = Vec::new();
        body.extend_from_slice(b"\0"); // portal
        body.extend_from_slice(b"s1\0"); // statement
        body.extend_from_slice(&0i16.to_be_bytes()); // param format count
        body.extend_from_slice(&2i16.to_be_bytes()); // param count
        body.extend_from_slice(&1i32.to_be_bytes()); // len of "7"
        body.extend_from_slice(b"7");
        body.extend_from_slice(&(-1i32).to_be_bytes()); // NULL
        body.extend_from_slice(&0i16.to_be_bytes()); // result format count
        match decode_frontend(b'B', &body).unwrap() {
            FrontendMessage::Bind {
                portal,
                statement,
                param_formats,
                params,
                result_formats,
            } => {
                assert_eq!(portal, "");
                assert_eq!(statement, "s1");
                assert!(param_formats.is_empty());
                assert_eq!(params, vec![Some(b"7".to_vec()), None]);
                assert!(result_formats.is_empty());
            }
            other => panic!("expected Bind, got {other:?}"),
        }
    }

    #[test]
    fn decode_bind_binary_param() {
        // 1 format code = 1 (binary), 1 param = int4 BE 0x0000002A (42).
        let mut body = Vec::new();
        body.extend_from_slice(b"\0"); // portal
        body.extend_from_slice(b"\0"); // statement (unnamed)
        body.extend_from_slice(&1i16.to_be_bytes()); // one format code
        body.extend_from_slice(&1i16.to_be_bytes()); // binary
        body.extend_from_slice(&1i16.to_be_bytes()); // one param
        body.extend_from_slice(&4i32.to_be_bytes());
        body.extend_from_slice(&42i32.to_be_bytes());
        body.extend_from_slice(&0i16.to_be_bytes()); // result format count
        match decode_frontend(b'B', &body).unwrap() {
            FrontendMessage::Bind {
                param_formats,
                params,
                ..
            } => {
                assert_eq!(param_formats, vec![1]);
                assert_eq!(params, vec![Some(42i32.to_be_bytes().to_vec())]);
            }
            other => panic!("expected Bind, got {other:?}"),
        }
    }

    #[test]
    fn decode_describe_execute_close_sync() {
        // Describe statement "stmt": kind byte 'S' then the C-string.
        match decode_frontend(b'D', b"Sstmt\0").unwrap() {
            FrontendMessage::Describe { kind, name } => {
                assert_eq!(kind, b'S');
                assert_eq!(name, "stmt");
            }
            other => panic!("expected Describe, got {other:?}"),
        }
        let mut e = Vec::new();
        e.extend_from_slice(b"\0"); // portal
        e.extend_from_slice(&0i32.to_be_bytes()); // max rows
        assert_eq!(
            decode_frontend(b'E', &e).unwrap(),
            FrontendMessage::Execute { portal: String::new(), max_rows: 0 }
        );
        assert_eq!(
            decode_frontend(b'C', b"Pportal\0").unwrap(),
            FrontendMessage::Close { kind: b'P', name: "portal".to_string() }
        );
        assert_eq!(decode_frontend(b'S', &[]).unwrap(), FrontendMessage::Sync);
        assert_eq!(decode_frontend(b'H', &[]).unwrap(), FrontendMessage::Flush);
        assert_eq!(decode_frontend(b'X', &[]).unwrap(), FrontendMessage::Terminate);
    }

    #[test]
    fn decode_rejects_unterminated_cstring() {
        // Parse with a statement name that has no null terminator.
        assert!(decode_frontend(b'P', b"s1").is_err());
    }

    #[test]
    fn decode_rejects_unknown_tag() {
        assert!(decode_frontend(b'Z', &[]).is_err());
    }
}
