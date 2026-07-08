//! XLM-RoBERTa embedder (BGE-M3), CLS pooling.
//!
//! BGE-M3 ships only a PyTorch pickle (`pytorch_model.bin`) — loaded via candle's pickle
//! reader; a leading `roberta.` key prefix is stripped so the bare model resolves.
//! Verified in `examples/probe_bge_m3.rs`.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::xlm_roberta::{Config, XLMRobertaModel};
use hf_hub::api::sync::ApiRepo;
use std::collections::HashMap;
use tokenizers::Tokenizer;

use super::{apply_prefix, l2_normalize, matryoshka_truncate, BoxError, ModelSpec, TextEmbedder};

pub struct XlmRobertaEmbedder {
    weights: HashMap<String, Tensor>,
    config: Config,
    tokenizer: Tokenizer,
    device: Device,
    spec: ModelSpec,
}

impl XlmRobertaEmbedder {
    pub fn load(repo: &ApiRepo, spec: &ModelSpec, device: &Device) -> Result<Self, BoxError> {
        let config_path = repo.get("config.json")?;
        let tokenizer_path = repo.get("tokenizer.json")?;
        let config: Config = serde_json::from_slice(&std::fs::read(&config_path)?)?;
        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| format!("tokenizer: {e}"))?;

        // Prefer safetensors if present; fall back to the pickle BGE-M3 actually ships.
        let raw = if let Ok(st) = repo.get("model.safetensors") {
            candle_core::safetensors::load(&st, device)?
        } else {
            let bin = repo.get("pytorch_model.bin")?;
            candle_core::pickle::read_all(&bin)?.into_iter().collect()
        };
        let mut weights = HashMap::with_capacity(raw.len());
        for (k, v) in raw {
            let key = k.strip_prefix("roberta.").unwrap_or(&k).to_string();
            weights.insert(key, v.to_device(device)?.to_dtype(DType::F32)?);
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

impl TextEmbedder for XlmRobertaEmbedder {
    fn embed(&self, text: &str, is_query: bool) -> Result<Vec<f32>, BoxError> {
        let text = apply_prefix(&self.spec, text, is_query);
        let enc = self
            .tokenizer
            .encode(text.as_str(), true)
            .map_err(|e| format!("tokenize: {e}"))?;
        let ids: Vec<u32> = enc.get_ids().to_vec();
        let seq_len = ids.len();
        let input = Tensor::new(ids.as_slice(), &self.device)?.reshape((1, seq_len))?;
        let attn = Tensor::ones((1, seq_len), DType::F32, &self.device)?;
        let tt = Tensor::zeros((1, seq_len), DType::U32, &self.device)?;

        // XLMRobertaModel holds no owned mutable state; build per call (Arc-cheap tensors).
        let vb = VarBuilder::from_tensors(self.weights.clone(), DType::F32, &self.device);
        let model = XLMRobertaModel::new(&self.config, vb)?;
        let hidden = model.forward(&input, &attn, &tt, None, None, None)?;

        // CLS pooling: first token.
        let cls = hidden.narrow(1, 0, 1)?.squeeze(1)?.squeeze(0)?;
        let mut v: Vec<f32> = cls.to_vec1()?;
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
