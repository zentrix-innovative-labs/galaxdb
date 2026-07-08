//! A.1e spike — prove BGE-M3 (BAAI/bge-m3) loads and embeds correctly in candle 0.11.
//!
//! XLM-RoBERTa encoder, **CLS** pooling (first token), dense embedding L2-normalized.
//!
//! Run (downloads ~2.2 GB on first use):
//!   cargo run -p galaxdb-sidecar --example probe_bge_m3 --release
//!
//! Success bar (same as the Qwen3 spike): dim == hidden_size (1024), L2-norm ≈ 1.0,
//! and semantic ordering (query closest to its relevant document).

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::xlm_roberta::{Config, XLMRobertaModel};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

type E = Box<dyn std::error::Error>;

const MODEL_ID: &str = "BAAI/bge-m3";

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
}

fn embed(
    weights: &std::collections::HashMap<String, Tensor>,
    cfg: &Config,
    tok: &Tokenizer,
    device: &Device,
    text: &str,
) -> Result<Vec<f32>, E> {
    let vb = VarBuilder::from_tensors(weights.clone(), DType::F32, device);
    let model = XLMRobertaModel::new(cfg, vb)?;

    let enc = tok.encode(text, true).map_err(|e| format!("tokenize: {e}"))?;
    let ids: Vec<u32> = enc.get_ids().to_vec();
    let seq_len = ids.len();
    let input = Tensor::new(ids.as_slice(), device)?.reshape((1, seq_len))?;
    // Full attention (no padding, single sequence); token types all zero.
    let attn = Tensor::ones((1, seq_len), DType::F32, device)?;
    let tt = Tensor::zeros((1, seq_len), DType::U32, device)?;

    let hidden = model.forward(&input, &attn, &tt, None, None, None)?; // (1, l, hidden)
    // CLS pooling: first token.
    let cls = hidden.narrow(1, 0, 1)?.squeeze(1)?.squeeze(0)?;
    let mut v: Vec<f32> = cls.to_vec1()?;
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
    // BGE-M3 ships only a PyTorch pickle (no safetensors) — read it via candle's pickle.
    let weights_path = repo.get("pytorch_model.bin")?;
    println!("[probe] downloaded config/tokenizer/weights (.bin)");

    let cfg: Config = serde_json::from_slice(&std::fs::read(&config_path)?)?;
    println!("[probe] config: hidden_size={}", cfg.hidden_size);

    let tok = Tokenizer::from_file(&tokenizer_path).map_err(|e| format!("tokenizer: {e}"))?;

    // Normalize tensor keys: strip a leading `roberta.` if present so the bare
    // XLMRobertaModel (which expects embeddings.* / encoder.* at root) resolves.
    let raw = candle_core::pickle::read_all(&weights_path)?;
    let mut weights = std::collections::HashMap::with_capacity(raw.len());
    for (k, v) in raw {
        let key = k.strip_prefix("roberta.").unwrap_or(&k).to_string();
        weights.insert(key, v.to_device(&device)?.to_dtype(DType::F32)?);
    }
    println!("[probe] loaded + normalized {} tensors", weights.len());

    let query = embed(&weights, &cfg, &tok, &device, "What is the capital of France?")?;
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
        println!("\n[probe] PASS — BGE-M3 loads and embeds correctly.");
        Ok(())
    } else {
        Err("probe FAILED one or more checks".into())
    }
}
