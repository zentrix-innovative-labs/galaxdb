//! A.3e spike — prove LFM2.5-Embedding-350M (LiquidAI/LFM2.5-Embedding-350M) loads and
//! embeds correctly in candle 0.11 with a **custom bidirectional LFM2 encoder**.
//!
//! LFM2 is a hybrid backbone: 16 blocks in a `conv`/`full_attention` pattern (short-conv
//! blocks + GQA attention blocks), hidden 1024, SwiGLU MLP, per-head QK-norm, RoPE.
//! candle's `models::lfm2` is the **causal LM** (causal attention mask, left-padded causal
//! short-conv, last-token narrow + lm_head). The embedding model is LiquidAI's
//! `Lfm2BidirectionalModel`, which (per the repo's `modeling_lfm2_bidirectional.py`) makes
//! exactly two changes to the backbone:
//!   1. **Attention** `is_causal = False` + a pad-only mask (no causal mask);
//!   2. **ShortConv** becomes non-causal: symmetric `conv1d(padding=k//2)` instead of the
//!      causal left-pad-then-trim.
//! then **CLS pooling** (first token of the final-normed `last_hidden_state`) + **L2
//! normalize**. No Dense heads (modules.json = Transformer + Pooling only).
//!
//! This file forks candle's `lfm2.rs` with those two changes and drops the KV/conv caches
//! (single bidirectional forward). For a single unpadded sequence the pad-only mask is a
//! no-op, so no attention mask is needed here.
//!
//! Run (downloads ~0.7 GB on first use):
//!   cargo run -p galaxdb-sidecar --example probe_lfm2 --release
//!
//! Success bar: dim == 1024, L2-norm ≈ 1.0, semantic ordering with the official prompts
//! (`query: ` / `document: `). Public model — no token required.

use std::sync::Arc;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Embedding, Linear, RmsNorm, VarBuilder};
use candle_transformers::utils::repeat_kv;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

type E = Box<dyn std::error::Error>;

const MODEL_ID: &str = "LiquidAI/LFM2.5-Embedding-350M";

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
    layer_types: Vec<LayerType>,
}

impl Cfg {
    fn from_json(v: &serde_json::Value, intermediate_size: usize) -> Result<Self, E> {
        let g = |k: &str| -> Result<u64, E> {
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
            .collect::<Result<Vec<_>, E>>()?;
        Ok(Self {
            hidden_size: hidden,
            num_hidden_layers: g("num_hidden_layers")? as usize,
            num_attention_heads: heads,
            num_key_value_heads: g("num_key_value_heads")? as usize,
            head_dim: hidden / heads,
            intermediate_size,
            norm_eps: v.get("norm_eps").and_then(|x| x.as_f64()).unwrap_or(1e-5),
            rope_theta: v.get("rope_theta").and_then(|x| x.as_f64()).unwrap_or(1e6) as f32,
            conv_l_cache: v
                .get("conv_L_cache")
                .and_then(|x| x.as_u64())
                .unwrap_or(3) as usize,
            vocab_size: g("vocab_size")? as usize,
            layer_types,
        })
    }
}

fn rms_norm(dim: usize, eps: f64, vb: VarBuilder) -> Result<RmsNorm, E> {
    Ok(candle_nn::rms_norm(dim, eps, vb)?)
}

fn linear_no_bias(inp: usize, out: usize, vb: VarBuilder) -> Result<Linear, E> {
    let w = vb.get((out, inp), "weight")?;
    Ok(Linear::new(w, None))
}

struct Rotary {
    sin: Tensor,
    cos: Tensor,
}
impl Rotary {
    fn new(head_dim: usize, theta: f32, max_seq: usize, dev: &Device) -> Result<Self, E> {
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
    fn apply(&self, x: &Tensor) -> Result<Tensor, E> {
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
    fn new(cfg: &Cfg, vb: VarBuilder) -> Result<Self, E> {
        Ok(Self {
            w1: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("w1"))?,
            w3: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("w3"))?,
            w2: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("w2"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor, E> {
        let gate = candle_nn::ops::silu(&self.w1.forward(x)?)?;
        let up = self.w3.forward(x)?;
        Ok(self.w2.forward(&(gate * up)?)?)
    }
}

/// Bidirectional GQA attention — no causal mask, no KV cache.
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
    fn new(cfg: &Cfg, rotary: Arc<Rotary>, vb: VarBuilder) -> Result<Self, E> {
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
    fn forward(&self, x: &Tensor) -> Result<Tensor, E> {
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
    fn new(cfg: &Cfg, vb: VarBuilder) -> Result<Self, E> {
        let h = cfg.hidden_size;
        let k = cfg.conv_l_cache;
        let in_proj = linear_no_bias(h, 3 * h, vb.pp("in_proj"))?;
        let out_proj = linear_no_bias(h, h, vb.pp("out_proj"))?;
        let conv_weight = vb.get((h, 1, k), "conv.weight")?;
        // Symmetric padding k//2 → non-causal, output length == input length for odd k.
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
            in_proj,
            out_proj,
            conv,
            hidden_size: h,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor, E> {
        let (_b, seq, _) = x.dims3()?;
        // (b, seq, 3h) -> (b, 3h, seq)
        let bcx = self.in_proj.forward(x)?.transpose(1, 2)?;
        let h = self.hidden_size;
        let b_gate = bcx.narrow(1, 0, h)?;
        let c_gate = bcx.narrow(1, h, h)?;
        let x_proj = bcx.narrow(1, 2 * h, h)?;
        let bx = (b_gate * &x_proj)?.contiguous()?;

        let mut conv_out = self.conv.forward(&bx)?;
        // Guard length for even/odd kernel edge cases (k=3 → exact).
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
    fn new(cfg: &Cfg, ty: LayerType, rotary: Arc<Rotary>, vb: VarBuilder) -> Result<Self, E> {
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
    fn forward(&self, x: &Tensor) -> Result<Tensor, E> {
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
    fn new(cfg: &Cfg, vb: VarBuilder, dev: &Device) -> Result<Self, E> {
        let embed = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let rotary = Arc::new(Rotary::new(cfg.head_dim, cfg.rope_theta, 4096, dev)?);
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
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor, E> {
        let mut xs = self.embed.forward(input_ids)?;
        for layer in &self.layers {
            xs = layer.forward(&xs)?;
        }
        // Final norm over all tokens → last_hidden_state.
        Ok(self.embedding_norm.forward(&xs)?)
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
}

fn embed(enc: &Encoder, tok: &Tokenizer, dev: &Device, text: &str) -> Result<Vec<f32>, E> {
    let encoding = tok.encode(text, true).map_err(|e| format!("tokenize: {e}"))?;
    let ids: Vec<u32> = encoding.get_ids().to_vec();
    let seq_len = ids.len();
    let input = Tensor::new(ids.as_slice(), dev)?.reshape((1, seq_len))?;
    let hidden = enc.forward(&input)?; // (1, seq, 1024)
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

const QUERY_PREFIX: &str = "query: ";
const DOC_PREFIX: &str = "document: ";

fn main() -> Result<(), E> {
    let device = Device::Cpu;
    println!("[probe] model = {MODEL_ID} on {:?}", device);

    let api = Api::new()?;
    let repo = api.repo(Repo::new(MODEL_ID.to_string(), RepoType::Model));
    let config_path = repo.get("config.json")?;
    let tokenizer_path = repo.get("tokenizer.json")?;
    let weights_path = repo.get("model.safetensors")?;
    println!("[probe] downloaded config/tokenizer/weights");

    let cfg_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&config_path)?)?;

    // The FFN intermediate dim varies with LFM2's auto-adjust; read the real value from
    // the checkpoint (feed_forward.w1.weight is [intermediate, hidden]) rather than trust
    // the config's nominal block_ff_dim.
    let raw = candle_core::safetensors::load(&weights_path, &device)?;
    let inter = raw
        .get("layers.0.feed_forward.w1.weight")
        .ok_or("missing layers.0.feed_forward.w1.weight")?
        .dim(0)?;
    println!("[probe] intermediate_size (from weights) = {inter}");

    let cfg = Cfg::from_json(&cfg_json, inter)?;
    println!(
        "[probe] config: hidden={} layers={} heads={} kv_heads={} head_dim={} conv_k={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.conv_l_cache
    );

    let tok = Tokenizer::from_file(&tokenizer_path).map_err(|e| format!("tokenizer: {e}"))?;

    // Root-keyed checkpoint (embed_tokens.weight, layers.*, embedding_norm.weight) — no
    // `model.` prefix, no lm_head. Load at root.
    let mut weights = std::collections::HashMap::with_capacity(raw.len());
    for (k, t) in raw {
        weights.insert(k, t.to_dtype(DType::F32)?);
    }
    let vb = VarBuilder::from_tensors(weights, DType::F32, &device);
    let enc = Encoder::new(&cfg, vb, &device)?;
    println!("[probe] encoder loaded ({} layers)", cfg.num_hidden_layers);

    let query = embed(&enc, &tok, &device, &format!("{QUERY_PREFIX}What is the capital of France?"))?;
    let doc_relevant = embed(
        &enc,
        &tok,
        &device,
        &format!("{DOC_PREFIX}Paris is the capital and most populous city of France."),
    )?;
    let doc_unrelated = embed(
        &enc,
        &tok,
        &device,
        &format!("{DOC_PREFIX}The mitochondria is the powerhouse of the cell."),
    )?;

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
        println!("\n[probe] PASS — LFM2.5-Embedding-350M loads and embeds correctly.");
        Ok(())
    } else {
        Err("probe FAILED one or more checks".into())
    }
}
