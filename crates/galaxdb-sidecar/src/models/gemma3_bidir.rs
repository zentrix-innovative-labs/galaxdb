//! EmbeddingGemma-300M — custom **bidirectional** Gemma 3 encoder + sentence-transformers
//! Dense heads + mean pooling + L2 normalize (Matryoshka-capable).
//!
//! candle's `models::gemma3` is a causal LM. This forks its blocks (Gemma RmsNorm `weight+1`,
//! per-layer RoPE base local/global, 4-layernorm decoder, QK-norm, embed scale sqrt(hidden))
//! but runs **bidirectional** (no causal/sliding mask), applies the final norm to all tokens,
//! mean-pools, then applies two Dense heads (768→3072→768, no bias, Identity) and normalizes.
//! Verified in `examples/probe_embeddinggemma.rs`.
//!
//! For inputs longer than `sliding_window` (512) a bidirectional sliding mask would be needed
//! for exact parity; the sidecar caps encoded length at the model's window for now and logs
//! when truncation happens (documents beyond 512 tokens are rare for embedding use).

use std::sync::Arc;

use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{Embedding, Linear, VarBuilder};
use candle_transformers::utils::repeat_kv;
use hf_hub::api::sync::ApiRepo;
use tokenizers::Tokenizer;

use super::{apply_prefix, l2_normalize, matryoshka_truncate, BoxError, ModelSpec, TextEmbedder};

#[derive(Clone)]
struct Cfg {
    hidden_size: usize,
    head_dim: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    num_hidden_layers: usize,
    intermediate_size: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    rope_local_base_freq: f64,
    sliding_window_pattern: usize,
    query_pre_attn_scalar: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
}

impl Cfg {
    fn from_json(v: &serde_json::Value) -> Result<Self, BoxError> {
        let g = |k: &str| -> Result<u64, BoxError> {
            v.get(k)
                .and_then(|x| x.as_u64())
                .ok_or_else(|| format!("config missing u64 {k}").into())
        };
        let gf = |k: &str| -> Result<f64, BoxError> {
            v.get(k)
                .and_then(|x| x.as_f64())
                .ok_or_else(|| format!("config missing f64 {k}").into())
        };
        let swp = v
            .get("sliding_window_pattern")
            .or_else(|| v.get("_sliding_window_pattern"))
            .and_then(|x| x.as_u64())
            .ok_or("config missing sliding_window_pattern")?;
        Ok(Self {
            hidden_size: g("hidden_size")? as usize,
            head_dim: g("head_dim")? as usize,
            num_attention_heads: g("num_attention_heads")? as usize,
            num_key_value_heads: g("num_key_value_heads")? as usize,
            num_hidden_layers: g("num_hidden_layers")? as usize,
            intermediate_size: g("intermediate_size")? as usize,
            rms_norm_eps: gf("rms_norm_eps")?,
            rope_theta: gf("rope_theta")?,
            rope_local_base_freq: gf("rope_local_base_freq")?,
            sliding_window_pattern: swp as usize,
            query_pre_attn_scalar: g("query_pre_attn_scalar")? as usize,
            vocab_size: g("vocab_size")? as usize,
            max_position_embeddings: g("max_position_embeddings").unwrap_or(2048) as usize,
        })
    }
}

/// Gemma RmsNorm: normalize, then scale by (weight + 1).
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}
impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self, BoxError> {
        Ok(Self {
            weight: vb.get(dim, "weight")?,
            eps,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor, BoxError> {
        let hidden = x.dim(D::Minus1)?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        Ok(x_normed.broadcast_mul(&(&self.weight + 1.0)?)?)
    }
}

struct Rotary {
    sin: Tensor,
    cos: Tensor,
}
impl Rotary {
    fn new(head_dim: usize, base: f64, max_seq: usize, dev: &Device) -> Result<Self, BoxError> {
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / base.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        let n = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, n), dev)?;
        let t = Tensor::arange(0u32, max_seq as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }
    fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor), BoxError> {
        let (_b, _h, seq, _d) = q.dims4()?;
        let cos = self.cos.narrow(0, 0, seq)?;
        let sin = self.sin.narrow(0, 0, seq)?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q, k))
    }
}

fn linear_no_bias(inp: usize, out: usize, vb: VarBuilder) -> Result<Linear, BoxError> {
    Ok(Linear::new(vb.get((out, inp), "weight")?, None))
}

struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}
impl Mlp {
    fn new(cfg: &Cfg, vb: VarBuilder) -> Result<Self, BoxError> {
        Ok(Self {
            gate: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor, BoxError> {
        // gelu_pytorch_tanh == candle Tensor::gelu (tanh approximation).
        let lhs = self.gate.forward(x)?.gelu()?;
        let rhs = self.up.forward(x)?;
        Ok(self.down.forward(&(lhs * rhs)?)?)
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
    rotary: Arc<Rotary>,
}
impl Attention {
    fn new(cfg: &Cfg, rotary: Arc<Rotary>, vb: VarBuilder) -> Result<Self, BoxError> {
        let h = cfg.hidden_size;
        let hd = cfg.head_dim;
        Ok(Self {
            q_proj: linear_no_bias(h, cfg.num_attention_heads * hd, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(h, cfg.num_key_value_heads * hd, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(h, cfg.num_key_value_heads * hd, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(cfg.num_attention_heads * hd, h, vb.pp("o_proj"))?,
            q_norm: RmsNorm::new(hd, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: RmsNorm::new(hd, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: hd,
            scale: 1f64 / (cfg.query_pre_attn_scalar as f64).sqrt(),
            rotary,
        })
    }
    fn forward(&self, xs: &Tensor) -> Result<Tensor, BoxError> {
        let (b, seq, _) = xs.dims3()?;
        let q = self
            .q_proj
            .forward(xs)?
            .reshape((b, seq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(xs)?
            .reshape((b, seq, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape((b, seq, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        let (q, k) = self.rotary.apply(&q, &k)?;

        let k = repeat_kv(k, self.num_heads / self.num_kv_heads)?.contiguous()?;
        let v = repeat_kv(v, self.num_heads / self.num_kv_heads)?.contiguous()?;

        // Bidirectional full attention (probe/short inputs); no mask.
        let attn = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;

        Ok(out
            .transpose(1, 2)?
            .reshape((b, seq, ()))?
            .apply(&self.o_proj)?)
    }
}

struct Layer {
    attn: Attention,
    mlp: Mlp,
    input_ln: RmsNorm,
    post_attn_ln: RmsNorm,
    pre_ff_ln: RmsNorm,
    post_ff_ln: RmsNorm,
}
impl Layer {
    fn new(cfg: &Cfg, rotary: Arc<Rotary>, vb: VarBuilder) -> Result<Self, BoxError> {
        Ok(Self {
            attn: Attention::new(cfg, rotary, vb.pp("self_attn"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            input_ln: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_attn_ln: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            pre_ff_ln: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("pre_feedforward_layernorm"),
            )?,
            post_ff_ln: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_feedforward_layernorm"),
            )?,
        })
    }
    fn forward(&self, xs: &Tensor) -> Result<Tensor, BoxError> {
        let residual = xs;
        let xs = self.input_ln.forward(xs)?;
        let xs = self.attn.forward(&xs)?;
        let xs = self.post_attn_ln.forward(&xs)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let h = self.pre_ff_ln.forward(&xs)?;
        let h = self.mlp.forward(&h)?;
        let h = self.post_ff_ln.forward(&h)?;
        Ok((residual + h)?)
    }
}

struct Encoder {
    embed: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    hidden_size: usize,
}
impl Encoder {
    fn new(cfg: &Cfg, vb: VarBuilder, dev: &Device) -> Result<Self, BoxError> {
        let embed = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let max_seq = cfg.max_position_embeddings;
        let rope_local =
            Arc::new(Rotary::new(cfg.head_dim, cfg.rope_local_base_freq, max_seq, dev)?);
        let rope_global = Arc::new(Rotary::new(cfg.head_dim, cfg.rope_theta, max_seq, dev)?);
        let vb_l = vb.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let is_sliding = (i + 1) % cfg.sliding_window_pattern > 0;
            let rope = if is_sliding {
                rope_local.clone()
            } else {
                rope_global.clone()
            };
            layers.push(Layer::new(cfg, rope, vb_l.pp(i))?);
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        Ok(Self {
            embed,
            layers,
            norm,
            hidden_size: cfg.hidden_size,
        })
    }
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor, BoxError> {
        let xs = self.embed.forward(input_ids)?;
        let mut xs = (xs * (self.hidden_size as f64).sqrt())?;
        for layer in &self.layers {
            xs = layer.forward(&xs)?;
        }
        self.norm.forward(&xs)
    }
}

pub struct Gemma3Embedder {
    encoder: Encoder,
    dense2: Linear,
    dense3: Linear,
    tokenizer: Tokenizer,
    device: Device,
    spec: ModelSpec,
    max_tokens: usize,
}

impl Gemma3Embedder {
    pub fn load(repo: &ApiRepo, spec: &ModelSpec, device: &Device) -> Result<Self, BoxError> {
        let config_path = repo.get("config.json")?;
        let tokenizer_path = repo.get("tokenizer.json")?;
        let backbone_path = repo.get("model.safetensors")?;
        let dense2_path = repo.get("2_Dense/model.safetensors")?;
        let dense3_path = repo.get("3_Dense/model.safetensors")?;

        let cfg_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&config_path)?)?;
        let cfg = Cfg::from_json(&cfg_json)?;

        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| format!("tokenizer: {e}"))?;

        // Backbone weights are root-keyed (embed_tokens.weight, layers.*, norm.weight).
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[backbone_path], DType::F32, device)? };
        let encoder = Encoder::new(&cfg, vb, device)?;

        // Dense heads: single `linear.weight` each, no bias. Dims read from the checkpoint.
        let d2 = super::load_safetensors_f32(&dense2_path, device)?;
        let d2w = d2
            .get("linear.weight")
            .ok_or("2_Dense missing linear.weight")?;
        let (d2_out, d2_in) = d2w.dims2()?;
        let dense2 = Linear::new(d2w.clone(), None);

        let d3 = super::load_safetensors_f32(&dense3_path, device)?;
        let d3w = d3
            .get("linear.weight")
            .ok_or("3_Dense missing linear.weight")?;
        let dense3 = Linear::new(d3w.clone(), None);

        debug_assert_eq!(d2_in, cfg.hidden_size);
        debug_assert_eq!(d3w.dims2()?.0, cfg.hidden_size);
        let _ = d2_out;

        Ok(Self {
            encoder,
            dense2,
            dense3,
            tokenizer,
            device: device.clone(),
            spec: spec.clone(),
            max_tokens: cfg.max_position_embeddings.min(512),
        })
    }
}

impl TextEmbedder for Gemma3Embedder {
    fn embed(&self, text: &str, is_query: bool) -> Result<Vec<f32>, BoxError> {
        let text = apply_prefix(&self.spec, text, is_query);
        let enc = self
            .tokenizer
            .encode(text.as_str(), true)
            .map_err(|e| format!("tokenize: {e}"))?;
        let mut ids: Vec<u32> = enc.get_ids().to_vec();
        // Cap at the model window (bidirectional-full == bidirectional-sliding below 512).
        if ids.len() > self.max_tokens {
            ids.truncate(self.max_tokens);
        }
        let seq_len = ids.len();
        let input = Tensor::new(ids.as_slice(), &self.device)?.reshape((1, seq_len))?;

        let hidden = self.encoder.forward(&input)?; // (1, seq, hidden)
        // Mean pooling over all tokens.
        let pooled = hidden.squeeze(0)?.mean(0)?.reshape((1, ()))?;
        let x = self.dense2.forward(&pooled)?;
        let x = self.dense3.forward(&x)?;
        let mut v: Vec<f32> = x.squeeze(0)?.to_vec1()?;
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
