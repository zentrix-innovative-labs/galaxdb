//! Qwen3-Embedding embedder (0.6B/4B/8B), decoder / last-token pooling / instruction prefix.
//!
//! The checkpoint is root-keyed (no `model.` prefix, no `lm_head`); candle's `qwen3::Model`
//! expects the `model.` prefix, so tensors are re-keyed at load. The base `Model`'s KV-cache
//! reset is private in candle 0.11, so a fresh `Model` is built per call from the cached
//! (Arc-backed) weights map — correct, and fine for the CPU launch model. Verified in
//! `examples/probe_qwen3.rs`.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3;
use hf_hub::api::sync::ApiRepo;
use std::collections::HashMap;
use tokenizers::Tokenizer;

use super::{apply_prefix, l2_normalize, matryoshka_truncate, BoxError, ModelSpec, TextEmbedder};

pub struct Qwen3Embedder {
    weights: HashMap<String, Tensor>,
    config: qwen3::Config,
    tokenizer: Tokenizer,
    device: Device,
    spec: ModelSpec,
}

impl Qwen3Embedder {
    pub fn load(repo: &ApiRepo, spec: &ModelSpec, device: &Device) -> Result<Self, BoxError> {
        let config_path = repo.get("config.json")?;
        let tokenizer_path = repo.get("tokenizer.json")?;
        let weights_path = repo.get("model.safetensors")?;

        // Fill head_dim if the config omits it (some Qwen3 configs do).
        let mut cfg_json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&config_path)?)?;
        if cfg_json.get("head_dim").is_none() {
            let hidden = cfg_json["hidden_size"].as_u64().ok_or("no hidden_size")?;
            let heads = cfg_json["num_attention_heads"]
                .as_u64()
                .ok_or("no num_attention_heads")?;
            cfg_json["head_dim"] = serde_json::json!(hidden / heads);
        }
        let config: qwen3::Config = serde_json::from_value(cfg_json)?;
        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| format!("tokenizer: {e}"))?;

        // Re-key with the `model.` prefix candle expects.
        let raw = candle_core::safetensors::load(&weights_path, device)?;
        let mut weights = HashMap::with_capacity(raw.len());
        for (k, v) in raw {
            weights.insert(format!("model.{k}"), v.to_dtype(DType::F32)?);
        }

        Ok(Self {
            weights,
            config,
            tokenizer,
            device: device.clone(),
            spec: spec.clone(),
        })
    }
}

impl TextEmbedder for Qwen3Embedder {
    fn embed(&self, text: &str, is_query: bool) -> Result<Vec<f32>, BoxError> {
        let text = apply_prefix(&self.spec, text, is_query);
        let enc = self
            .tokenizer
            .encode(text.as_str(), true)
            .map_err(|e| format!("tokenize: {e}"))?;
        let ids: Vec<u32> = enc.get_ids().to_vec();
        let seq_len = ids.len();
        let input = Tensor::new(ids.as_slice(), &self.device)?.reshape((1, seq_len))?;

        // Fresh model per call so the KV cache never carries across texts.
        let vb = VarBuilder::from_tensors(self.weights.clone(), DType::F32, &self.device);
        let mut model = qwen3::Model::new(&self.config, vb)?;
        let hidden = model.forward(&input, 0)?;

        // Last-token pooling.
        let last = hidden.narrow(1, seq_len - 1, 1)?.squeeze(1)?.squeeze(0)?;
        let mut v: Vec<f32> = last.to_vec1()?;
        l2_normalize(&mut v);
        if let Some(dim) = self.spec.output_dim {
            matryoshka_truncate(&mut v, dim);
        }
        Ok(v)
    }

    fn dim(&self) -> usize {
        self.spec.effective_dim()
    }
}
