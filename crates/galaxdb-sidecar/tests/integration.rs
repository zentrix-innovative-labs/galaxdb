//! Integration tests for the sidecar binary.
//!
//! These tests start the actual sidecar binary as a child process,
//! connect via Unix socket, and verify the protocol works end-to-end.

use std::io::{BufReader, BufWriter};
use std::os::unix::net::UnixStream;
use std::process::{Child, Command};
use std::time::Duration;

use galaxdb_sidecar::protocol::*;

/// Start the sidecar binary in mock mode and return the child process.
fn start_sidecar(socket_path: &str, dim: usize) -> Child {
    let binary = env!("CARGO_BIN_EXE_galaxdb-sidecar");
    Command::new(binary)
        .args([
            "--socket", socket_path,
            "--mock-dim", &dim.to_string(),
        ])
        .spawn()
        .expect("failed to start sidecar binary")
}

/// Wait for the sidecar to be ready (socket file exists).
fn wait_for_socket(path: &str, timeout: Duration) -> bool {
    let start = std::time::Instant::now();
    while start.elapsed() < timeout {
        if std::path::Path::new(path).exists() {
            // Give it a moment to start accepting
            std::thread::sleep(Duration::from_millis(50));
            return true;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    false
}

#[test]
fn sidecar_embed_request_response() {
    let dir = tempfile::tempdir().unwrap();
    let socket_path = dir.path().join("test.sock");
    let socket_str = socket_path.to_str().unwrap();

    let mut child = start_sidecar(socket_str, 128);
    assert!(wait_for_socket(socket_str, Duration::from_secs(5)), "sidecar did not start");

    // Connect
    let stream = UnixStream::connect(&socket_path).expect("connect to sidecar");
    stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut writer = BufWriter::new(stream);

    // Send embed request
    let req = SidecarMessage::EmbedRequest(EmbedRequest {
        row_id: 42,
        text: "hello world".to_string(),
        column: "content_embedding".to_string(),
    });
    write_message(&mut writer, &req).unwrap();

    // Read response
    let resp = read_message(&mut reader).unwrap();
    match resp {
        SidecarMessage::EmbedResponse(r) => {
            assert_eq!(r.row_id, 42);
            assert_eq!(r.embedding.len(), 128);
            assert!(!r.model_version.is_empty());

            // Verify the embedding is normalized (unit length)
            let norm: f32 = r.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 0.01,
                "embedding should be normalized, got norm={}",
                norm
            );
        }
        other => panic!("expected EmbedResponse, got {:?}", other),
    }

    // Send status request
    let status_req = SidecarMessage::StatusRequest(StatusRequest {});
    write_message(&mut writer, &status_req).unwrap();

    let status_resp = read_message(&mut reader).unwrap();
    match status_resp {
        SidecarMessage::StatusResponse(s) => {
            assert_eq!(s.dimensions, 128);
            assert_eq!(s.max_in_flight, MAX_IN_FLIGHT);
        }
        other => panic!("expected StatusResponse, got {:?}", other),
    }

    // Cleanup
    child.kill().ok();
    child.wait().ok();
}

#[test]
fn sidecar_deterministic_embeddings() {
    let dir = tempfile::tempdir().unwrap();
    let socket_path = dir.path().join("test_det.sock");
    let socket_str = socket_path.to_str().unwrap();

    let mut child = start_sidecar(socket_str, 64);
    assert!(wait_for_socket(socket_str, Duration::from_secs(5)));

    let stream = UnixStream::connect(&socket_path).expect("connect");
    stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut writer = BufWriter::new(stream);

    // Same text should produce same embedding (deterministic mock)
    let req1 = SidecarMessage::EmbedRequest(EmbedRequest {
        row_id: 1,
        text: "test input".to_string(),
        column: "emb".to_string(),
    });
    write_message(&mut writer, &req1).unwrap();
    let resp1 = read_message(&mut reader).unwrap();

    let req2 = SidecarMessage::EmbedRequest(EmbedRequest {
        row_id: 2,
        text: "test input".to_string(),
        column: "emb".to_string(),
    });
    write_message(&mut writer, &req2).unwrap();
    let resp2 = read_message(&mut reader).unwrap();

    let emb1 = match resp1 {
        SidecarMessage::EmbedResponse(r) => r.embedding,
        _ => panic!("expected EmbedResponse"),
    };
    let emb2 = match resp2 {
        SidecarMessage::EmbedResponse(r) => r.embedding,
        _ => panic!("expected EmbedResponse"),
    };

    assert_eq!(emb1, emb2, "same text should produce same embedding");

    // Different text should produce different embedding
    let req3 = SidecarMessage::EmbedRequest(EmbedRequest {
        row_id: 3,
        text: "different input".to_string(),
        column: "emb".to_string(),
    });
    write_message(&mut writer, &req3).unwrap();
    let resp3 = read_message(&mut reader).unwrap();
    let emb3 = match resp3 {
        SidecarMessage::EmbedResponse(r) => r.embedding,
        _ => panic!("expected EmbedResponse"),
    };

    assert_ne!(emb1, emb3, "different text should produce different embedding");

    child.kill().ok();
    child.wait().ok();
}
