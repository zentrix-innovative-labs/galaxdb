//! Integration tests for the sidecar binary.
//!
//! These tests spawn the real `galaxdb-sidecar` binary, which downloads
//! and loads a sentence-transformer model from HuggingFace Hub. They are
//! gated behind the `online-tests` feature flag so `cargo test
//! -p galaxdb-sidecar` without flags stays hermetic:
//!
//! ```text
//! cargo test -p galaxdb-sidecar --features online-tests
//! ```
//!
//! There is no mock mode — every embedding in these tests is computed by
//! the real model.

#![cfg(feature = "online-tests")]

use std::io::{BufReader, BufWriter};
use std::os::unix::net::UnixStream;
use std::process::{Child, Command};
use std::time::Duration;

use galaxdb_sidecar::protocol::*;

/// Default model id used by the online integration tests. Matches the
/// production default exposed at `galaxdb_sidecar::manager::DEFAULT_MODEL_ID`.
const TEST_MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";

/// Dimension emitted by `TEST_MODEL_ID`. Pinned so an accidental model
/// upgrade is caught by the test suite rather than silently changing
/// embedding sizes.
const TEST_MODEL_DIM: usize = 384;

/// Spawn the sidecar binary with a real model. The binary downloads the
/// model from HF Hub on first run and caches it locally; subsequent runs
/// load from cache.
fn start_sidecar(socket_path: &str, model_id: &str) -> Child {
    let binary = env!("CARGO_BIN_EXE_galaxdb-sidecar");
    Command::new(binary)
        .args(["--socket", socket_path, "--model", model_id])
        .spawn()
        .expect("failed to start sidecar binary")
}

/// Wait for the sidecar to be ready (socket file exists). First run
/// includes the ~90 MB model download; we allow up to 2 minutes.
fn wait_for_socket(path: &str, timeout: Duration) -> bool {
    let start = std::time::Instant::now();
    while start.elapsed() < timeout {
        if std::path::Path::new(path).exists() {
            // Give it a moment to start accepting.
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

    let mut child = start_sidecar(socket_str, TEST_MODEL_ID);
    assert!(
        wait_for_socket(socket_str, Duration::from_secs(120)),
        "sidecar did not start within 120s — check network / HF Hub"
    );

    // Connect.
    let stream = UnixStream::connect(&socket_path).expect("connect to sidecar");
    stream.set_read_timeout(Some(Duration::from_secs(30))).ok();
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut writer = BufWriter::new(stream);

    // Send embed request.
    let req = SidecarMessage::EmbedRequest(EmbedRequest {
        row_id: 42,
        text: "hello world".to_string(),
        column: "content_embedding".to_string(),
        is_query: false,
    });
    write_message(&mut writer, &req).unwrap();

    // Read response.
    let resp = read_message(&mut reader).unwrap();
    match resp {
        SidecarMessage::EmbedResponse(r) => {
            assert_eq!(r.row_id, 42);
            assert_eq!(
                r.embedding.len(),
                TEST_MODEL_DIM,
                "embedding dimension must match {TEST_MODEL_ID}"
            );
            assert_eq!(r.model_version, TEST_MODEL_ID);

            // Real sentence-transformer output is L2-normalized.
            let norm: f32 = r.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 0.01,
                "embedding should be L2-normalized, got norm={}",
                norm
            );
        }
        other => panic!("expected EmbedResponse, got {:?}", other),
    }

    // Send status request.
    let status_req = SidecarMessage::StatusRequest(StatusRequest {});
    write_message(&mut writer, &status_req).unwrap();

    let status_resp = read_message(&mut reader).unwrap();
    match status_resp {
        SidecarMessage::StatusResponse(s) => {
            assert_eq!(s.dimensions, TEST_MODEL_DIM);
            assert_eq!(s.max_in_flight, MAX_IN_FLIGHT);
            assert_eq!(s.model_id, TEST_MODEL_ID);
            assert_eq!(s.model_version, TEST_MODEL_ID);
        }
        other => panic!("expected StatusResponse, got {:?}", other),
    }

    // Cleanup.
    child.kill().ok();
    child.wait().ok();
}

#[test]
fn sidecar_deterministic_embeddings() {
    let dir = tempfile::tempdir().unwrap();
    let socket_path = dir.path().join("test_det.sock");
    let socket_str = socket_path.to_str().unwrap();

    let mut child = start_sidecar(socket_str, TEST_MODEL_ID);
    assert!(wait_for_socket(socket_str, Duration::from_secs(120)));

    let stream = UnixStream::connect(&socket_path).expect("connect");
    stream.set_read_timeout(Some(Duration::from_secs(30))).ok();
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut writer = BufWriter::new(stream);

    // The real model is deterministic for the same input — a real
    // regression test, not a property of a hash-based mock.
    let req1 = SidecarMessage::EmbedRequest(EmbedRequest {
        row_id: 1,
        text: "the quick brown fox".to_string(),
        column: "emb".to_string(),
        is_query: false,
    });
    write_message(&mut writer, &req1).unwrap();
    let resp1 = read_message(&mut reader).unwrap();

    let req2 = SidecarMessage::EmbedRequest(EmbedRequest {
        row_id: 2,
        text: "the quick brown fox".to_string(),
        column: "emb".to_string(),
        is_query: false,
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

    assert_eq!(emb1.len(), TEST_MODEL_DIM);
    assert_eq!(emb2.len(), TEST_MODEL_DIM);
    assert_eq!(
        emb1, emb2,
        "same text should produce byte-identical embeddings on the same model"
    );

    // Different text must produce a different embedding (this is the
    // whole point of the model — distinguishing similar strings).
    let req3 = SidecarMessage::EmbedRequest(EmbedRequest {
        row_id: 3,
        text: "a completely unrelated sentence about quantum physics".to_string(),
        column: "emb".to_string(),
        is_query: false,
    });
    write_message(&mut writer, &req3).unwrap();
    let resp3 = read_message(&mut reader).unwrap();
    let emb3 = match resp3 {
        SidecarMessage::EmbedResponse(r) => r.embedding,
        _ => panic!("expected EmbedResponse"),
    };

    assert_ne!(
        emb1, emb3,
        "different text must produce a different embedding"
    );

    // Semantic sanity: cosine similarity between near-duplicates should
    // be much higher than between unrelated sentences. This guards
    // against model-selection regressions (e.g. accidentally shipping a
    // model that outputs constant vectors).
    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
    }
    let req4 = SidecarMessage::EmbedRequest(EmbedRequest {
        row_id: 4,
        text: "the quick brown fox jumps".to_string(),
        column: "emb".to_string(),
        is_query: false,
    });
    write_message(&mut writer, &req4).unwrap();
    let emb4 = match read_message(&mut reader).unwrap() {
        SidecarMessage::EmbedResponse(r) => r.embedding,
        _ => panic!("expected EmbedResponse"),
    };

    let sim_near = cosine(&emb1, &emb4);
    let sim_far = cosine(&emb1, &emb3);
    assert!(
        sim_near > sim_far,
        "near-duplicate cosine ({sim_near}) must exceed unrelated cosine ({sim_far})"
    );

    child.kill().ok();
    child.wait().ok();
}
