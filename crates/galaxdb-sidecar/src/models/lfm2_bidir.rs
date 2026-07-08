//! LFM2.5-Embedding-350M — custom **bidirectional** LFM2 hybrid encoder, CLS pooling.
//!
//! candle's `models::lfm2` is the causal LM. Per LiquidAI's `modeling_lfm2_bidirectional.py`
//! the embedding model makes exactly two changes: attention is non-causal (no mask), and the
//! short-conv is non-causal (symmetric `conv1d(padding=k//2)` instead of causal left-pad).
//! Final `embedding_norm` over all tokens → CLS (first token) → L2 normalize. No Dense heads.
//! Verified in `examples/probe_lfm2.rs`.

use std::sync::Arc;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Embedding, Linear, RmsNorm, VarBuilder};
use candle_transformers::utils::repeat_kv;
use hf_hub::api::sync::ApiRepo;
use tokenizers::Tokenizer;

use super::{apply_prefix, l2_normalize, matryoshka_truncate, BoxError, ModelSpec, TextEmbedder};

#[derive(Clone, Copy, PartialEq)]
enum LayerType {
    FullAttention,
    Conv,
}

#[derive(Clone)]
struct Cfg {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    norm_eps: f64,
    rope_theta: f32,
    conv_l_cache: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    layer_types: Vec<LayerType>,
}

impl Cfg {
    fn from_json(v: &serde_json::Value, intermediate_size: usize) -> Result<Self, BoxError> {
        let g = |k: &str| -> Result<u64, BoxError> {
            v.get(k)
                .and_then(|x| x.as_u64())
                .ok_or_else(|| format!("config missing u64 {k}").into())
        };
        let hidden = g("hidden_size")? as usize;
        let heads = g("num_attention_heads")? as usize;
        let layer_types = v
            .get("layer_types")
            .and_then(|x| x.as_array())
            .ok_or("config missing layer_types")?
            .iter()
            .map(|t| match t.as_str() {
                Some("conv") => Ok(LayerType::Conv),
                Some("full_attention") => Ok(LayerType::FullAttention),
                other => Err(format!("unknown layer_type {other:?}").into()),
            })
            .collect::<Result<Vec<_>, BoxError>>()?;
        Ok(Self {
            hidden_size: hidden,
            num_hidden_layers: g("num_hidden_layers")? as usize,
            num_attention_heads: heads,
            num_key_value_heads: g("num_key_value_heads")? as usize,
            head_dim: hidden / heads,
            intermediate_size,
            norm_eps: v.get("norm_eps").and_then(|x| x.as_f64()).unwrap_or(1e-5),
            rope_theta: v.get("rope_theta").and_then(|x| x.as_f64()).unwrap_or(1e6) as f32,
            conv_l_cache: v.get("conv_L_cache").and_then(|x| x.as_u64()).unwrap_or(3) as usize,
            vocab_size: g("vocab_size")? as usize,
            max_position_embeddings: g("max_position_embeddings").unwrap_or(512) as usize,
            layer_types,
        })
    }
}

fn rms_norm(dim: usize, eps: f64, vb: VarBuilder) -> Result<RmsNorm, BoxError> {
    Ok(candle_nn::rms_norm(dim, eps, vb)?)
}

fn linear_no_bias(inp: usize, out: usize, vb: VarBuilder) -> Result<Linear, BoxError> {
    Ok(Linear::new(vb.get((out, inp), "weight")?, None))
}

struct Rotary {
    sin: Tensor,
    cos: Tensor,
}
impl Rotary {
    fn new(head_dim: usize, theta: f32, max_seq: usize, dev: &Device) -> Result<Self, BoxError> {
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / theta.powf(i as f32 / head_dim as f32))
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
    fn apply(&self, x: &Tensor) -> Result<Tensor, BoxError> {
        let (_b, _h, seq, _d) = x.dims4()?;
        let cos = self.cos.narrow(0, 0, seq)?;
        let sin = self.sin.narrow(0, 0, seq)?;
        Ok(candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)?)
    }
}

struct Mlp {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}
impl Mlp {
    fn new(cfg: &Cfg, vb: VarBuilder) -> Result<Self, BoxError> {
        Ok(Self {
            w1: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("w1"))?,
            w3: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("w3"))?,
            w2: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("w2"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor, BoxError> {
        let gate = candle_nn::ops::silu(&self.w1.forward(x)?)?;
        let up = self.w3.forward(x)?;
        Ok(self.w2.forward(&(gate * up)?)?)
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
            o_proj: linear_no_bias(cfg.num_attention_heads * hd, h, vb.pp("out_proj"))?,
            q_norm: rms_norm(hd, cfg.norm_eps, vb.pp("q_layernorm"))?,
            k_norm: rms_norm(hd, cfg.norm_eps, vb.pp("k_layernorm"))?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: hd,
            rotary,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor, BoxError> {
        let (b, seq, _) = x.dims3()?;
        let q = self
            .q_proj
            .forward(x)?
            .reshape((b, seq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, seq, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, seq, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = self.q_norm.forward(&q.contiguous()?)?;
        let k = self.k_norm.forward(&k.contiguous()?)?;
        let q = self.rotary.apply(&q)?;
        let k = self.rotary.apply(&k)?;

        let k = repeat_kv(k, self.num_heads / self.num_kv_heads)?.contiguous()?;
        let v = repeat_kv(v, self.num_heads / self.num_kv_heads)?.contiguous()?;

        // Bidirectional: full softmax, no mask (single unpadded sequence).
        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att.matmul(&v)?;

        Ok(y
            .transpose(1, 2)?
            .reshape((b, seq, self.num_heads * self.head_dim))?
            .apply(&self.o_proj)?)
    }
}

/// Non-causal short convolution (symmetric padding = k//2).
struct ShortConv {
    in_proj: Linear,
    out_proj: Linear,
    conv: Conv1d,
    hidden_size: usize,
}
impl ShortConv {
    fn new(cfg: &Cfg, vb: VarBuilder) -> Result<Self, BoxError> {
        let h = cfg.hidden_size;
        let k = cfg.conv_l_cache;
        let conv_weight = vb.get((h, 1, k), "conv.weight")?;
        let conv = Conv1d::new(
            conv_weight,
            None,
            Conv1dConfig {
                padding: k / 2,
                groups: h,
                ..Default::default()
            },
        );
        Ok(Self {
            in_proj: linear_no_bias(h, 3 * h, vb.pp("in_proj"))?,
            out_proj: linear_no_bias(h, h, vb.pp("out_proj"))?,
            conv,
            hidden_size: h,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor, BoxError> {
        let (_b, seq, _) = x.dims3()?;
        let bcx = self.in_proj.forward(x)?.transpose(1, 2)?;
        let h = self.hidden_size;
        let b_gate = bcx.narrow(1, 0, h)?;
        let c_gate = bcx.narrow(1, h, h)?;
        let x_proj = bcx.narrow(1, 2 * h, h)?;
        let bx = (b_gate * &x_proj)?.contiguous()?;

        let mut conv_out = self.conv.forward(&bx)?;
        if conv_out.dim(2)? > seq {
            conv_out = conv_out.narrow(2, 0, seq)?;
        }
        let y = (c_gate * &conv_out)?;
        let y = y.transpose(1, 2)?.contiguous()?;
        Ok(self.out_proj.forward(&y)?)
    }
}

enum Kind {
    Attn(Attention),
    Conv(ShortConv),
}

struct Layer {
    operator_norm: RmsNorm,
    ffn_norm: RmsNorm,
    mlp: Mlp,
    kind: Kind,
}
impl Layer {
    fn new(cfg: &Cfg, ty: LayerType, rotary: Arc<Rotary>, vb: VarBuilder) -> Result<Self, BoxError> {
        let kind = match ty {
            LayerType::FullAttention => Kind::Attn(Attention::new(cfg, rotary, vb.pp("self_attn"))?),
            LayerType::Conv => Kind::Conv(ShortConv::new(cfg, vb.pp("conv"))?),
        };
        Ok(Self {
            operator_norm: rms_norm(cfg.hidden_size, cfg.norm_eps, vb.pp("operator_norm"))?,
            ffn_norm: rms_norm(cfg.hidden_size, cfg.norm_eps, vb.pp("ffn_norm"))?,
            mlp: Mlp::new(cfg, vb.pp("feed_forward"))?,
            kind,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor, BoxError> {
        let residual = x;
        let h = self.operator_norm.forward(x)?;
        let h = match &self.kind {
            Kind::Attn(a) => a.forward(&h)?,
            Kind::Conv(c) => c.forward(&h)?,
        };
        let x = (h + residual)?;
        let residual = &x;
        let h = self.ffn_norm.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        Ok((residual + h)?)
    }
}

struct Encoder {
    embed: Embedding,
    layers: Vec<Layer>,
    embedding_norm: RmsNorm,
}
impl Encoder {
    fn new(cfg: &Cfg, vb: VarBuilder, dev: &Device) -> Result<Self, BoxError> {
        let embed = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let rotary = Arc::new(Rotary::new(
            cfg.head_dim,
            cfg.rope_theta,
            cfg.max_position_embeddings.max(512),
            dev,
        )?);
        let vb_l = vb.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Layer::new(cfg, cfg.layer_types[i], rotary.clone(), vb_l.pp(i))?);
        }
        let embedding_norm = rms_norm(cfg.hidden_size, cfg.norm_eps, vb.pp("embedding_norm"))?;
        Ok(Self {
            embed,
            layers,
            embedding_norm,
        })
    }
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor, BoxError> {
        let mut xs = self.embed.forward(input_ids)?;
        for layer in &self.layers {
            xs = layer.forward(&xs)?;
        }
        Ok(self.embedding_norm.forward(&xs)?)
    }
}

pub struct Lfm2Embedder {
    encoder: Encoder,
    tokenizer: Tokenizer,
    device: Device,
    spec: ModelSpec,
    max_tokens: usize,
}

impl Lfm2Embedder {
    pub fn load(repo: &ApiRepo, spec: &ModelSpec, device: &Device) -> Result<Self, BoxError> {
        let config_path = repo.get("config.json")?;
        let tokenizer_path = repo.get("tokenizer.json")?;
        let weights_path = repo.get("model.safetensors")?;

        let cfg_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&config_path)?)?;

        // Real FFN dim varies with LFM2's auto-adjust — read it from the checkpoint.
        let weights = super::load_safetensors_f32(&weights_path, device)?;
        let inter = weights
            .get("layers.0.feed_forward.w1.weight")
            .ok_or("missing layers.0.feed_forward.w1.weight")?
            .dim(0)?;
        let cfg = Cfg::from_json(&cfg_json, inter)?;

        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| format!("tokenizer: {e}"))?;

        let vb = VarBuilder::from_tensors(weights, DType::F32, device);
        let encoder = Encoder::new(&cfg, vb, device)?;

        Ok(Self {
            encoder,
            tokenizer,
            device: device.clone(),
            spec: spec.clone(),
            max_tokens: cfg.max_position_embeddings,
        })
    }
}

impl TextEmbedder for Lfm2Embedder {
    fn embed(&self, text: &str, is_query: bool) -> Result<Vec<f32>, BoxError> {
        let text = apply_prefix(&self.spec, text, is_query);
        let enc = self
            .tokenizer
            .encode(text.as_str(), true)
            .map_err(|e| format!("tokenize: {e}"))?;
        let mut ids: Vec<u32> = enc.get_ids().to_vec();
        if ids.len() > self.max_tokens {
            ids.truncate(self.max_tokens);
        }
        let seq_len = ids.len();
        let input = Tensor::new(ids.as_slice(), &self.device)?.reshape((1, seq_len))?;
        let hidden = self.encoder.forward(&input)?; // (1, seq, hidden)

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
