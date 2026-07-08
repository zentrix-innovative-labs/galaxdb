//! A.1c spike — prove Qwen3-Embedding-0.6B loads and embeds correctly in candle 0.11.
//!
//! Decoder architecture, **last-token** pooling, query **instruction prefix**. This is the
//! trickiest of the four launch architectures, so it goes first.
//!
//! Run (downloads ~1.2–2.4 GB on first use):
//!   cargo run -p galaxdb-sidecar --example probe_qwen3 --release
//!
//! Success bar for the spike (Req 2.1/2.4/2.6, sanity form of 2.2):
//!   - loads config + weights via candle_transformers::models::qwen3
//!   - output dimension == config.hidden_size (1024 for 0.6B)
//!   - each embedding is L2-normalized (norm ≈ 1.0)
//!   - semantic ordering holds: a query is closest to its relevant document and
//!     farther from an unrelated one.
//!
//! This is a spike, not the shipped path — it rebuilds the model per text because
//! `qwen3::Model::clear_kv_cache` is private in candle 0.11. The real loader (A.3)
//! will handle batching/caching properly.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

type E = Box<dyn std::error::Error>;

const MODEL_ID: &str = "Qwen/Qwen3-Embedding-0.6B";

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
}

/// Qwen3-Embedding query instruction format (documents get no prefix).
fn query_prompt(task: &str, query: &str) -> String {
    format!("Instruct: {task}\nQuery:{query}")
}

fn embed(
    weights: &std::collections::HashMap<String, Tensor>,
    cfg: &qwen3::Config,
    tok: &Tokenizer,
    device: &Device,
    text: &str,
) -> Result<Vec<f32>, E> {
    // Fresh model per call (spike-only) so the KV cache never carries across texts.
    // The Qwen3-Embedding checkpoint stores tensors at the root, but candle's
    // qwen3::Model expects the `model.` prefix — the weights map is already
    // re-keyed in main(). Tensor clones are Arc-cheap.
    let vb = VarBuilder::from_tensors(weights.clone(), DType::F32, device);
    let mut model = qwen3::Model::new(cfg, vb)?;

    let enc = tok.encode(text, true).map_err(|e| format!("tokenize: {e}"))?;
    let ids: Vec<u32> = enc.get_ids().to_vec();
    let seq_len = ids.len();
    let input = Tensor::new(ids.as_slice(), device)?.reshape((1, seq_len))?;

    // Base model forward returns final hidden states (1, seq_len, hidden).
    let hidden = model.forward(&input, 0)?;
    // Last-token pooling: take the final position.
    let last = hidden.narrow(1, seq_len - 1, 1)?.squeeze(1)?.squeeze(0)?;
    let mut v: Vec<f32> = last.to_vec1()?;
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    Ok(v)
}

fn main() -> Result<(), E> {
    let device = Device::Cpu;
    println!("[probe] model = {MODEL_ID} on {:?}", device);

    let api = Api::new()?;
    let repo = api.repo(Repo::new(MODEL_ID.to_string(), RepoType::Model));
    let config_path = repo.get("config.json")?;
    let tokenizer_path = repo.get("tokenizer.json")?;
    let weights_path = repo.get("model.safetensors")?;
    println!("[probe] downloaded config/tokenizer/weights");

    // Deserialize config; fill head_dim if the config omits it (some Qwen3 configs do).
    let mut cfg_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&config_path)?)?;
    if cfg_json.get("head_dim").is_none() {
        let hidden = cfg_json["hidden_size"].as_u64().ok_or("no hidden_size")?;
        let heads = cfg_json["num_attention_heads"].as_u64().ok_or("no num_attention_heads")?;
        cfg_json["head_dim"] = serde_json::json!(hidden / heads);
        println!("[probe] head_dim absent — computed {}", hidden / heads);
    }
    let cfg: qwen3::Config = serde_json::from_value(cfg_json)?;
    println!(
        "[probe] config: hidden_size={} layers={} heads={} kv_heads={}",
        cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
    );

    let tok = Tokenizer::from_file(&tokenizer_path).map_err(|e| format!("tokenizer: {e}"))?;

    // Load weights and re-key with the `model.` prefix that candle's qwen3::Model
    // expects; cast every tensor to F32 (CPU path). The checkpoint is rooted
    // (embed_tokens.weight, layers.*, norm.weight) with no lm_head.
    let raw = candle_core::safetensors::load(&weights_path, &device)?;
    let mut weights = std::collections::HashMap::with_capacity(raw.len());
    for (k, v) in raw {
        weights.insert(format!("model.{k}"), v.to_dtype(DType::F32)?);
    }
    println!("[probe] loaded + re-keyed {} tensors", weights.len());

    let task = "Given a web search query, retrieve relevant passages that answer the query";
    let query = embed(&weights, &cfg, &tok, &device,
        &query_prompt(task, "What is the capital of France?"))?;
    let doc_relevant = embed(&weights, &cfg, &tok, &device,
        "Paris is the capital and most populous city of France.")?;
    let doc_unrelated = embed(&weights, &cfg, &tok, &device,
        "The mitochondria is the powerhouse of the cell.")?;

    let dim = query.len();
    let norm = query.iter().map(|x| x * x).sum::<f32>().sqrt();
    let sim_rel = cosine(&query, &doc_relevant);
    let sim_unrel = cosine(&query, &doc_unrelated);

    println!("\n[probe] RESULTS");
    println!("  dim              = {dim}  (expected {})", cfg.hidden_size);
    println!("  query L2 norm    = {norm:.4}  (expected ~1.0)");
    println!("  cos(query, relevant)   = {sim_rel:.4}");
    println!("  cos(query, unrelated)  = {sim_unrel:.4}");

    let dim_ok = dim == cfg.hidden_size;
    let norm_ok = (norm - 1.0).abs() < 1e-3;
    let order_ok = sim_rel > sim_unrel;
    println!("\n  dim_ok={dim_ok}  norm_ok={norm_ok}  semantic_order_ok={order_ok}");

    if dim_ok && norm_ok && order_ok {
        println!("\n[probe] PASS — Qwen3-Embedding-0.6B loads and embeds correctly.");
        Ok(())
    } else {
        Err("probe FAILED one or more checks".into())
    }
}
