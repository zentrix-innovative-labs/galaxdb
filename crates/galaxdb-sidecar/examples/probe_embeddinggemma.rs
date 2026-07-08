//! A.3d spike — prove EmbeddingGemma-300M (google/embeddinggemma-300m) loads and embeds
//! correctly in candle 0.11 with a **custom bidirectional Gemma 3 encoder**.
//!
//! candle's `models::gemma3` is a **causal LM** (last-token narrow + causal/sliding masks +
//! KV cache + lm_head). EmbeddingGemma is a sentence encoder built on the same backbone but:
//!   - **bidirectional** attention (`use_bidirectional_attention: true`) — no causal mask;
//!   - the backbone final `norm` is applied to **all** token positions;
//!   - **mean pooling** over all tokens (`pooling_mode_mean_tokens`, `include_prompt: true`);
//!   - two sentence-transformers **Dense** heads (768→3072 then 3072→768, no bias, Identity);
//!   - **L2 normalize** (Matryoshka full dim = 768).
//!
//! This file forks candle's `gemma3.rs` into an encoder: same RmsNorm (Gemma `weight + 1`
//! convention), same per-layer RoPE base (local 10000 for sliding layers, global 1e6 for full
//! layers), same 4-layernorm decoder block and embed scale by sqrt(hidden), but **no causal
//! mask, no sliding mask, and no KV cache**. Our probe texts are short (< sliding_window=512
//! tokens) so bidirectional-full and bidirectional-sliding are identical here; the shipped
//! loader (A.3d) will add a bidirectional sliding mask for long inputs.
//!
//! Gated model: needs an accepted Google license + an HF token in the environment
//! (`HF_TOKEN` / `HUGGINGFACE_TOKEN`) or `~/.cache/huggingface/token`.
//!
//! Run (downloads ~1.2 GB backbone + Dense heads on first use):
//!   cargo run -p galaxdb-sidecar --example probe_embeddinggemma --release
//!
//! Success bar: dim == 768, L2-norm ≈ 1.0, and semantic ordering (query closest to its
//! relevant document, far from an unrelated one), using the official query/document prompts.

use std::sync::Arc;

use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{Embedding, Linear, VarBuilder};
use candle_transformers::utils::repeat_kv;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

type E = Box<dyn std::error::Error>;

const MODEL_ID: &str = "google/embeddinggemma-300m";

/// Minimal config mirrored from the model's `config.json` (verified, not assumed).
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
}

impl Cfg {
    fn from_json(v: &serde_json::Value) -> Result<Self, E> {
        let g = |k: &str| -> Result<u64, E> {
            v.get(k)
                .and_then(|x| x.as_u64())
                .ok_or_else(|| format!("config missing u64 {k}").into())
        };
        let gf = |k: &str| -> Result<f64, E> {
            v.get(k)
                .and_then(|x| x.as_f64())
                .ok_or_else(|| format!("config missing f64 {k}").into())
        };
        // HF uses `_sliding_window_pattern`; accept either spelling.
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
        })
    }
}

/// Gemma RmsNorm: normalize, then scale by (weight + 1).
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}
impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self, E> {
        Ok(Self {
            weight: vb.get(dim, "weight")?,
            eps,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor, E> {
        let hidden = x.dim(D::Minus1)?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        Ok(x_normed.broadcast_mul(&(&self.weight + 1.0)?)?)
    }
}

/// Precomputed RoPE tables for one base frequency.
struct Rotary {
    sin: Tensor,
    cos: Tensor,
}
impl Rotary {
    fn new(head_dim: usize, base: f64, max_seq: usize, dev: &Device) -> Result<Self, E> {
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
    fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor), E> {
        let (_b, _h, seq, _d) = q.dims4()?;
        let cos = self.cos.narrow(0, 0, seq)?;
        let sin = self.sin.narrow(0, 0, seq)?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q, k))
    }
}

fn linear_no_bias(inp: usize, out: usize, vb: VarBuilder) -> Result<Linear, E> {
    let w = vb.get((out, inp), "weight")?;
    Ok(Linear::new(w, None))
}

struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}
impl Mlp {
    fn new(cfg: &Cfg, vb: VarBuilder) -> Result<Self, E> {
        Ok(Self {
            gate: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor, E> {
        // gelu_pytorch_tanh == candle Tensor::gelu (tanh approximation).
        let lhs = self.gate.forward(x)?.gelu()?;
        let rhs = self.up.forward(x)?;
        Ok(self.down.forward(&(lhs * rhs)?)?)
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
    num_kv_groups: usize,
    head_dim: usize,
    scale: f64,
    rotary: Arc<Rotary>,
}
impl Attention {
    fn new(cfg: &Cfg, rotary: Arc<Rotary>, vb: VarBuilder) -> Result<Self, E> {
        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        Ok(Self {
            q_proj: linear_no_bias(h, nh * hd, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(h, nkv * hd, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(h, nkv * hd, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(nh * hd, h, vb.pp("o_proj"))?,
            q_norm: RmsNorm::new(hd, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: RmsNorm::new(hd, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads: nh,
            num_kv_heads: nkv,
            num_kv_groups: nh / nkv,
            head_dim: hd,
            // Gemma3 uses query_pre_attn_scalar^-0.5 (== 1/sqrt(head_dim) here).
            scale: 1f64 / (cfg.query_pre_attn_scalar as f64).sqrt(),
            rotary,
        })
    }
    fn forward(&self, xs: &Tensor) -> Result<Tensor, E> {
        let (b, q_len, _) = xs.dims3()?;
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let q = q
            .reshape((b, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // q/k RMSNorm on head_dim (Gemma3 QK-norm), then RoPE.
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        let (q, k) = self.rotary.apply(&q, &k)?;

        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        // Bidirectional: full softmax, no mask (probe inputs << sliding_window).
        let attn = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;

        Ok(out
            .transpose(1, 2)?
            .reshape((b, q_len, ()))?
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
    fn new(cfg: &Cfg, rotary: Arc<Rotary>, vb: VarBuilder) -> Result<Self, E> {
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
    fn forward(&self, xs: &Tensor) -> Result<Tensor, E> {
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

/// Bidirectional Gemma 3 encoder → last_hidden_state (all tokens, final-normed).
struct Encoder {
    embed: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    hidden_size: usize,
}
impl Encoder {
    fn new(cfg: &Cfg, vb: VarBuilder, dev: &Device) -> Result<Self, E> {
        let embed = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        // Two RoPE tables shared across layers by type: local (sliding) vs global (full).
        let max_seq = 2048;
        let rope_local = Arc::new(Rotary::new(
            cfg.head_dim,
            cfg.rope_local_base_freq,
            max_seq,
            dev,
        )?);
        let rope_global = Arc::new(Rotary::new(cfg.head_dim, cfg.rope_theta, max_seq, dev)?);
        let vb_l = vb.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            // Same pattern as candle gemma3: (i+1) % pattern == 0 → full attention layer.
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
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor, E> {
        let xs = self.embed.forward(input_ids)?;
        let mut xs = (xs * (self.hidden_size as f64).sqrt())?;
        for layer in &self.layers {
            xs = layer.forward(&xs)?;
        }
        // Final norm applied to ALL token positions (encoder last_hidden_state).
        self.norm.forward(&xs)
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
}

fn embed(
    enc: &Encoder,
    dense2: &Linear,
    dense3: &Linear,
    tok: &Tokenizer,
    dev: &Device,
    text: &str,
) -> Result<Vec<f32>, E> {
    let encoding = tok.encode(text, true).map_err(|e| format!("tokenize: {e}"))?;
    let ids: Vec<u32> = encoding.get_ids().to_vec();
    let seq_len = ids.len();
    let input = Tensor::new(ids.as_slice(), dev)?.reshape((1, seq_len))?;

    let hidden = enc.forward(&input)?; // (1, seq, 768)
    // Mean pooling over all tokens (include_prompt=true, no padding here).
    let pooled = hidden.squeeze(0)?.mean(0)?; // (768)
    let pooled = pooled.reshape((1, ()))?;
    // Dense 768->3072 (Identity), Dense 3072->768 (Identity).
    let x = dense2.forward(&pooled)?;
    let x = dense3.forward(&x)?;
    let mut v: Vec<f32> = x.squeeze(0)?.to_vec1()?;
    // L2 normalize (4_Normalize).
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    Ok(v)
}

const QUERY_PREFIX: &str = "task: search result | query: ";
const DOC_PREFIX: &str = "title: none | text: ";

fn main() -> Result<(), E> {
    let device = Device::Cpu;
    println!("[probe] model = {MODEL_ID} on {:?}", device);

    let api = Api::new()?;
    let repo = api.repo(Repo::new(MODEL_ID.to_string(), RepoType::Model));
    let config_path = repo.get("config.json")?;
    let tokenizer_path = repo.get("tokenizer.json")?;
    let backbone_path = repo.get("model.safetensors")?;
    let dense2_path = repo.get("2_Dense/model.safetensors")?;
    let dense3_path = repo.get("3_Dense/model.safetensors")?;
    println!("[probe] downloaded config/tokenizer/backbone + 2 Dense heads");

    let cfg_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&config_path)?)?;
    let cfg = Cfg::from_json(&cfg_json)?;
    println!(
        "[probe] config: hidden={} head_dim={} heads={} kv_heads={} layers={} swp={}",
        cfg.hidden_size,
        cfg.head_dim,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.num_hidden_layers,
        cfg.sliding_window_pattern
    );

    let tok = Tokenizer::from_file(&tokenizer_path).map_err(|e| format!("tokenizer: {e}"))?;

    // Backbone weights are root-keyed (embed_tokens.weight, layers.*, norm.weight) — no
    // `model.` prefix, no lm_head. Load directly at root.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[backbone_path], DType::F32, &device)?
    };
    let enc = Encoder::new(&cfg, vb, &device)?;
    println!("[probe] backbone loaded ({} layers)", cfg.num_hidden_layers);

    // Dense heads: single `linear.weight` each, no bias.
    let vb2 = unsafe { VarBuilder::from_mmaped_safetensors(&[dense2_path], DType::F32, &device)? };
    let dense2 = linear_no_bias(768, 3072, vb2.pp("linear"))?;
    let vb3 = unsafe { VarBuilder::from_mmaped_safetensors(&[dense3_path], DType::F32, &device)? };
    let dense3 = linear_no_bias(3072, 768, vb3.pp("linear"))?;
    println!("[probe] Dense heads loaded (768->3072->768)");

    let query = embed(
        &enc,
        &dense2,
        &dense3,
        &tok,
        &device,
        &format!("{QUERY_PREFIX}What is the capital of France?"),
    )?;
    let doc_relevant = embed(
        &enc,
        &dense2,
        &dense3,
        &tok,
        &device,
        &format!("{DOC_PREFIX}Paris is the capital and most populous city of France."),
    )?;
    let doc_unrelated = embed(
        &enc,
        &dense2,
        &dense3,
        &tok,
        &device,
        &format!("{DOC_PREFIX}The mitochondria is the powerhouse of the cell."),
    )?;

    let dim = query.len();
    let norm = query.iter().map(|x| x * x).sum::<f32>().sqrt();
    let sim_rel = cosine(&query, &doc_relevant);
    let sim_unrel = cosine(&query, &doc_unrelated);

    println!("\n[probe] RESULTS");
    println!("  dim              = {dim}  (expected 768)");
    println!("  query L2 norm    = {norm:.4}  (expected ~1.0)");
    println!("  cos(query, relevant)   = {sim_rel:.4}");
    println!("  cos(query, unrelated)  = {sim_unrel:.4}");

    let dim_ok = dim == 768;
    let norm_ok = (norm - 1.0).abs() < 1e-3;
    let order_ok = sim_rel > sim_unrel;
    println!("\n  dim_ok={dim_ok}  norm_ok={norm_ok}  semantic_order_ok={order_ok}");

    if dim_ok && norm_ok && order_ok {
        println!("\n[probe] PASS — EmbeddingGemma-300M loads and embeds correctly.");
        Ok(())
    } else {
        Err("probe FAILED one or more checks".into())
    }
}
