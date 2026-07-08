//! Per-model fidelity tests for the multi-architecture registry (task A.6).
//!
//! Each test loads a **real** model through `galaxdb_sidecar::models::load`, embeds a fixed
//! probe set, and asserts: correct effective dimension, unit L2-norm, and semantic ordering
//! (a query is closer to its relevant document than to an unrelated one). There is no mock —
//! these exercise the exact code path the sidecar binary uses.
//!
//! Gated behind `online-tests` (they download weights from HF Hub):
//! ```text
//! cargo test -p galaxdb-sidecar --features online-tests --test models_online -- --nocapture
//! ```
//! Downloads are large (Qwen3 ~1.2 GB, BGE-M3 ~2.2 GB). Run individually as needed, e.g.
//! `... --test models_online lfm2`.
//!
//! The EmbeddingGemma test requires an accepted Google license + an HF token
//! (`HF_TOKEN` or `HUGGINGFACE_TOKEN`, or `~/.cache/huggingface/token`); it is skipped with a
//! clear message when no token is configured, since gated weights cannot be downloaded
//! without credentials. Every other model is public.

#![cfg(all(unix, feature = "online-tests"))]

use candle_core::Device;
use galaxdb_sidecar::models;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
}

/// Load `hf_id`, embed the probe set, and assert dim / norm / semantic ordering.
fn check_model(hf_id: &str, expected_dim: usize) {
    let device = Device::Cpu;
    let loaded = models::load(hf_id, &device)
        .unwrap_or_else(|e| panic!("load {hf_id} failed: {e}"));
    assert_eq!(
        loaded.embedder.dim(),
        expected_dim,
        "{hf_id}: effective dim mismatch"
    );

    let query = loaded
        .embedder
        .embed("What is the capital of France?", true)
        .unwrap();
    let relevant = loaded
        .embedder
        .embed("Paris is the capital and most populous city of France.", false)
        .unwrap();
    let unrelated = loaded
        .embedder
        .embed("The mitochondria is the powerhouse of the cell.", false)
        .unwrap();

    assert_eq!(query.len(), expected_dim, "{hf_id}: query dim");
    assert_eq!(relevant.len(), expected_dim, "{hf_id}: doc dim");

    let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-3,
        "{hf_id}: query not L2-normalized (norm={norm})"
    );

    let sim_rel = cosine(&query, &relevant);
    let sim_unrel = cosine(&query, &unrelated);
    println!("[{hf_id}] dim={expected_dim} norm={norm:.4} rel={sim_rel:.4} unrel={sim_unrel:.4}");
    assert!(
        sim_rel > sim_unrel,
        "{hf_id}: semantic ordering failed (rel={sim_rel} <= unrel={sim_unrel})"
    );
}

fn have_hf_token() -> bool {
    std::env::var("HF_TOKEN").is_ok()
        || std::env::var("HUGGINGFACE_TOKEN").is_ok()
        || dirs_home_token()
}

fn dirs_home_token() -> bool {
    if let Some(home) = std::env::var_os("HOME") {
        let p = std::path::Path::new(&home).join(".cache/huggingface/token");
        return p.exists();
    }
    false
}

#[test]
fn all_minilm_default() {
    check_model("sentence-transformers/all-MiniLM-L6-v2", 384);
}

#[test]
fn bge_m3_xlm_roberta_cls() {
    check_model("BAAI/bge-m3", 1024);
}

#[test]
fn qwen3_embedding_0_6b_last_token() {
    check_model("Qwen/Qwen3-Embedding-0.6B", 1024);
}

#[test]
fn lfm2_embedding_bidirectional_cls() {
    check_model("LiquidAI/LFM2.5-Embedding-350M", 1024);
}

#[test]
fn embeddinggemma_bidirectional_mean() {
    if !have_hf_token() {
        eprintln!(
            "[skip] embeddinggemma: no HF token configured (HF_TOKEN / HUGGINGFACE_TOKEN / \
             ~/.cache/huggingface/token). Gated weights require an accepted Google license."
        );
        return;
    }
    check_model("google/embeddinggemma-300m", 768);
}
