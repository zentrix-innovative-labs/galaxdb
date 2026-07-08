//! BERT sentence-transformer embedder (all-MiniLM and compatible), mean pooling.
//!
//! This is the sidecar's original default path, moved behind the [`TextEmbedder`] trait
//! with no behavior change for `sentence-transformers/all-MiniLM-L6-v2`.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use hf_hub::api::sync::ApiRepo;
use tokenizers::Tokenizer;

use super::{apply_prefix, l2_normalize, matryoshka_truncate, BoxError, ModelSpec, TextEmbedder};

pub struct BertEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    spec: ModelSpec,
}

impl BertEmbedder {
    pub fn load(repo: &ApiRepo, spec: &ModelSpec, device: &Device) -> Result<Self, BoxError> {
        let config_path = repo.get("config.json")?;
        let tokenizer_path = repo.get("tokenizer.json")?;
        let weights_path = repo.get("model.safetensors")?;

        let config: BertConfig = serde_json::from_slice(&std::fs::read(&config_path)?)?;
        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| format!("tokenizer: {e}"))?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)? };
        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            spec: spec.clone(),
        })
    }
}

impl TextEmbedder for BertEmbedder {
    fn embed(&self, text: &str, is_query: bool) -> Result<Vec<f32>, BoxError> {
        let text = apply_prefix(&self.spec, text, is_query);
        let enc = self
            .tokenizer
            .encode(text.as_str(), true)
            .map_err(|e| format!("tokenize: {e}"))?;
        let ids: Vec<u32> = enc.get_ids().to_vec();
        let mask: Vec<u32> = enc.get_attention_mask().to_vec();
        let tt: Vec<u32> = enc.get_type_ids().to_vec();
        let seq_len = ids.len();

        let ids_t = Tensor::new(ids.as_slice(), &self.device)?.reshape((1, seq_len))?;
        let mask_t = Tensor::new(mask.as_slice(), &self.device)?.reshape((1, seq_len))?;
        let tt_t = Tensor::new(tt.as_slice(), &self.device)?.reshape((1, seq_len))?;

        let output = self.model.forward(&ids_t, &tt_t, Some(&mask_t))?;

        // Attention-masked mean pooling.
        let mask_f32 = mask_t
            .to_dtype(DType::F32)?
            .unsqueeze(2)?
            .broadcast_as(output.shape())?;
        let summed = (output * mask_f32.clone())?.sum(1)?;
        let counts = mask_f32.sum(1)?;
        let pooled = (summed / counts)?;

        let mut v: Vec<f32> = pooled.squeeze(0)?.to_vec1()?;
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
