//! Sidecar communication protocol — shared between sidecar binary and engine client.
//!
//! Communication is via Unix socket with length-prefixed JSON messages.
//!
//! Wire format:
//! ```text
//! [u32 length (little-endian)][JSON payload (length bytes)]
//! ```
//!
//! Message types:
//! - EmbedRequest: engine → sidecar (generate embedding for text)
//! - EmbedResponse: sidecar → engine (embedding result)
//! - HeartbeatPing: sidecar → engine (alive signal)
//! - HeartbeatPong: engine → sidecar (acknowledgment)
//! - StatusRequest: engine → sidecar (query sidecar status)
//! - StatusResponse: sidecar → engine (model info, in-flight count)

use std::io::{self, Read, Write};

use serde::{Deserialize, Serialize};

/// Request from engine to sidecar: generate an embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedRequest {
    /// Row ID in the storage engine.
    pub row_id: u64,
    /// The text to embed.
    pub text: String,
    /// The embedding column name.
    pub column: String,
    /// Whether this text is a search **query** (vs a stored **document**).
    ///
    /// Asymmetric embedding models (Qwen3-Embedding, EmbeddingGemma,
    /// LFM2.5-Embedding) apply a different instruction/prefix to queries and
    /// documents. Symmetric models (all-MiniLM, BGE-M3) ignore this. Defaults
    /// to `false` (document) so older clients that omit the field keep the
    /// prior document-embedding behavior.
    #[serde(default)]
    pub is_query: bool,
}

impl EmbedRequest {
    /// Construct a request to embed a stored **document** row.
    pub fn document(row_id: u64, text: String, column: String) -> Self {
        Self {
            row_id,
            text,
            column,
            is_query: false,
        }
    }

    /// Construct a request to embed a search **query**.
    pub fn query(row_id: u64, text: String, column: String) -> Self {
        Self {
            row_id,
            text,
            column,
            is_query: true,
        }
    }
}

/// Response from sidecar to engine: embedding result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedResponse {
    /// Row ID (echoed from request).
    pub row_id: u64,
    /// The generated embedding vector.
    pub embedding: Vec<f32>,
    /// Model version that produced this embedding.
    pub model_version: String,
}

/// Heartbeat ping from sidecar to engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatPing {
    /// Sidecar's current in-flight count.
    pub in_flight: usize,
    /// Sidecar's model version.
    pub model_version: String,
}

/// Heartbeat pong from engine to sidecar.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatPong {
    /// Acknowledged.
    pub ok: bool,
}

/// Status request from engine to sidecar.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusRequest {}

/// Status response from sidecar to engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusResponse {
    pub model_id: String,
    pub model_version: String,
    pub dimensions: usize,
    pub in_flight: usize,
    pub max_in_flight: usize,
    /// Backbone architecture (e.g. `bert`, `qwen3`, `gemma3_bidirectional`). Additive:
    /// older sidecars omit it and it deserializes to an empty string.
    #[serde(default)]
    pub architecture: String,
    /// Pooling strategy (e.g. `mean`, `cls`, `last_token`). Additive (see `architecture`).
    #[serde(default)]
    pub pooling: String,
}

/// Envelope for all sidecar messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SidecarMessage {
    EmbedRequest(EmbedRequest),
    EmbedResponse(EmbedResponse),
    HeartbeatPing(HeartbeatPing),
    HeartbeatPong(HeartbeatPong),
    StatusRequest(StatusRequest),
    StatusResponse(StatusResponse),
    Error { message: String },
}

/// Write a length-prefixed JSON message to a writer.
pub fn write_message<W: Write>(writer: &mut W, msg: &SidecarMessage) -> io::Result<()> {
    let json = serde_json::to_vec(msg)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let len = json.len() as u32;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(&json)?;
    writer.flush()?;
    Ok(())
}

/// Read a length-prefixed JSON message from a reader.
pub fn read_message<R: Read>(reader: &mut R) -> io::Result<SidecarMessage> {
    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;

    if len > 10 * 1024 * 1024 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("message too large: {} bytes", len),
        ));
    }

    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;

    serde_json::from_slice(&buf)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Default Unix socket path for the sidecar.
pub fn default_socket_path(data_dir: &str) -> String {
    format!("{}/galaxdb_sidecar.sock", data_dir)
}

/// Default heartbeat interval in seconds.
pub const HEARTBEAT_INTERVAL_SECS: u64 = 5;

/// Heartbeat timeout in seconds (3 missed = degraded mode).
pub const HEARTBEAT_TIMEOUT_SECS: u64 = 2;

/// Maximum in-flight embedding requests before overflow to backlog.
pub const MAX_IN_FLIGHT: usize = 10_000;

/// Exponential backoff schedule for sidecar restart (seconds).
pub const RESTART_BACKOFF: &[u64] = &[1, 2, 4, 8, 16, 32, 60];

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn message_roundtrip_embed_request() {
        let msg = SidecarMessage::EmbedRequest(EmbedRequest {
            row_id: 42,
            text: "hello world".to_string(),
            column: "content_embedding".to_string(),
            is_query: true,
        });

        let mut buf = Vec::new();
        write_message(&mut buf, &msg).unwrap();

        let mut cursor = Cursor::new(&buf);
        let decoded = read_message(&mut cursor).unwrap();

        match decoded {
            SidecarMessage::EmbedRequest(req) => {
                assert_eq!(req.row_id, 42);
                assert_eq!(req.text, "hello world");
                assert_eq!(req.column, "content_embedding");
                assert!(req.is_query);
            }
            _ => panic!("expected EmbedRequest"),
        }
    }

    #[test]
    fn message_roundtrip_embed_response() {
        let msg = SidecarMessage::EmbedResponse(EmbedResponse {
            row_id: 42,
            embedding: vec![0.1, 0.2, 0.3],
            model_version: "v1.0".to_string(),
        });

        let mut buf = Vec::new();
        write_message(&mut buf, &msg).unwrap();

        let mut cursor = Cursor::new(&buf);
        let decoded = read_message(&mut cursor).unwrap();

        match decoded {
            SidecarMessage::EmbedResponse(resp) => {
                assert_eq!(resp.row_id, 42);
                assert_eq!(resp.embedding, vec![0.1, 0.2, 0.3]);
                assert_eq!(resp.model_version, "v1.0");
            }
            _ => panic!("expected EmbedResponse"),
        }
    }

    #[test]
    fn message_roundtrip_heartbeat() {
        let msg = SidecarMessage::HeartbeatPing(HeartbeatPing {
            in_flight: 42,
            model_version: "v1.0".to_string(),
        });

        let mut buf = Vec::new();
        write_message(&mut buf, &msg).unwrap();

        let mut cursor = Cursor::new(&buf);
        let decoded = read_message(&mut cursor).unwrap();

        match decoded {
            SidecarMessage::HeartbeatPing(ping) => {
                assert_eq!(ping.in_flight, 42);
            }
            _ => panic!("expected HeartbeatPing"),
        }
    }

    #[test]
    fn message_roundtrip_status() {
        let msg = SidecarMessage::StatusResponse(StatusResponse {
            model_id: "all-MiniLM-L6-v2".to_string(),
            model_version: "v1.0".to_string(),
            dimensions: 384,
            in_flight: 100,
            max_in_flight: 10_000,
            architecture: "bert".to_string(),
            pooling: "mean".to_string(),
        });

        let mut buf = Vec::new();
        write_message(&mut buf, &msg).unwrap();

        let mut cursor = Cursor::new(&buf);
        let decoded = read_message(&mut cursor).unwrap();

        match decoded {
            SidecarMessage::StatusResponse(status) => {
                assert_eq!(status.model_id, "all-MiniLM-L6-v2");
                assert_eq!(status.dimensions, 384);
                assert_eq!(status.max_in_flight, 10_000);
            }
            _ => panic!("expected StatusResponse"),
        }
    }

    #[test]
    fn message_too_large_rejected() {
        // Craft a message with a huge length prefix
        let mut buf = Vec::new();
        buf.extend_from_slice(&(20_000_000u32).to_le_bytes()); // 20MB
        buf.extend_from_slice(b"{}");

        let mut cursor = Cursor::new(&buf);
        let result = read_message(&mut cursor);
        assert!(result.is_err());
    }

    #[test]
    fn error_message_roundtrip() {
        let msg = SidecarMessage::Error {
            message: "model not found".to_string(),
        };

        let mut buf = Vec::new();
        write_message(&mut buf, &msg).unwrap();

        let mut cursor = Cursor::new(&buf);
        let decoded = read_message(&mut cursor).unwrap();

        match decoded {
            SidecarMessage::Error { message } => {
                assert_eq!(message, "model not found");
            }
            _ => panic!("expected Error"),
        }
    }
}
