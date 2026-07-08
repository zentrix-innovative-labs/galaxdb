//! Multi-architecture embedding model registry and loaders (v0.5, task A.2/A.3).
//!
//! This module turns the single hard-coded BERT path into a trait-based registry so the
//! sidecar can load any of the supported embedding architectures at runtime via its HF id.
//!
//! Design goals (see `.kiro/specs/galaxdb-v0.5`):
//! - **Real models only.** Every embedder loads real weights and computes real vectors. An
//!   unsupported/unknown model id yields a typed error — never a mock or a silent fallback.
//! - **Additively evolvable.** [`Architecture`] and [`Pooling`] are `#[non_exhaustive]`;
//!   adding a model is a new registry entry (+ a loader arm if it is a new architecture).
//! - **Asymmetric-model aware.** [`TextEmbedder::embed`] takes `is_query` so instruction /
//!   prefix models (Qwen3, EmbeddingGemma, LFM2.5) embed queries and documents correctly.
//!
//! The default model stays `sentence-transformers/all-MiniLM-L6-v2` (BERT / mean / 384-d).

use std::collections::HashMap;

use hf_hub::api::sync::ApiRepo;

mod bert;
mod gemma3_bidir;
mod lfm2_bidir;
mod qwen3;
mod xlm_roberta;

/// Boxed, thread-safe error used across the loaders.
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Pooling strategy that turns per-token hidden states into one sentence vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Pooling {
    /// Attention-masked mean over all tokens (BERT sentence-transformers default).
    Mean,
    /// First token ([CLS]) — XLM-RoBERTa / BGE-M3, LFM2.5-Embedding.
    Cls,
    /// Last token — decoder embedding models (Qwen3-Embedding).
    LastToken,
}

impl Pooling {
    /// Stable lower-snake string (matches the serde wire form).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Cls => "cls",
            Self::LastToken => "last_token",
        }
    }
}

/// Backbone family. Determines which loader builds the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Architecture {
    /// BERT encoder (all-MiniLM and most sentence-transformers).
    Bert,
    /// XLM-RoBERTa encoder (BGE-M3).
    XlmRoberta,
    /// Qwen3 decoder used as an embedding model (last-token + instruction prefix).
    Qwen3,
    /// Gemma 3 backbone made bidirectional + Dense heads (EmbeddingGemma).
    Gemma3Bidirectional,
    /// LFM2 hybrid (conv + attention) made bidirectional (LFM2.5-Embedding).
    Lfm2Bidirectional,
}

impl Architecture {
    /// Stable lower-snake string (matches the serde wire form).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Bert => "bert",
            Self::XlmRoberta => "xlm_roberta",
            Self::Qwen3 => "qwen3",
            Self::Gemma3Bidirectional => "gemma3_bidirectional",
            Self::Lfm2Bidirectional => "lfm2_bidirectional",
        }
    }

    /// Map a HuggingFace `config.json` `model_type` to an architecture.
    fn from_model_type(model_type: &str) -> Option<Self> {
        match model_type {
            "bert" => Some(Self::Bert),
            "xlm-roberta" | "xlm_roberta" | "roberta" => Some(Self::XlmRoberta),
            "qwen3" => Some(Self::Qwen3),
            "gemma3_text" | "gemma3" => Some(Self::Gemma3Bidirectional),
            "lfm2" => Some(Self::Lfm2Bidirectional),
            _ => None,
        }
    }
}

/// Rough resource requirement, surfaced to Cloud tiering (informational).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ResourceClass {
    /// Comfortably runs CPU-only.
    Cpu,
    /// Large model; prefers a GPU / high-memory box.
    Gpu,
}

/// Everything the sidecar needs to load and drive one model.
///
/// Curated for the launch set; auto-detected from the model repo for other ids.
/// Serializable so the GalaxDB Cloud control plane can mirror the engine's registry
/// (Cloud cross-repo item E-2/E-3) without duplicating these constants.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelSpec {
    pub hf_id: String,
    pub arch: Architecture,
    pub pooling: Pooling,
    /// Instruction/prefix prepended to a search query (asymmetric models). `None` = symmetric.
    pub query_prefix: Option<String>,
    /// Prefix prepended to a stored document (asymmetric models). `None` = symmetric.
    pub doc_prefix: Option<String>,
    /// The model's native embedding dimension (before any Matryoshka truncation).
    pub native_dim: usize,
    /// Matryoshka target dimension (truncate + renormalize). `None` = use `native_dim`.
    pub output_dim: Option<usize>,
    pub license: String,
    pub resource_class: ResourceClass,
    /// Whether the weights are gated (require an accepted license + HF token).
    pub gated: bool,
}

impl ModelSpec {
    /// Effective output dimension after optional Matryoshka truncation.
    pub fn effective_dim(&self) -> usize {
        self.output_dim.unwrap_or(self.native_dim)
    }
}

/// A loaded embedding model. Implementations own their weights + tokenizer.
pub trait TextEmbedder: Send + Sync {
    /// Embed one text. `is_query` selects the query vs document prefix for asymmetric models.
    fn embed(&self, text: &str, is_query: bool) -> Result<Vec<f32>, BoxError>;
    /// Effective output dimension (post-Matryoshka).
    fn dim(&self) -> usize;
}

/// A loaded model together with the spec that describes it.
pub struct Loaded {
    pub embedder: Box<dyn TextEmbedder>,
    pub spec: ModelSpec,
}

/// L2-normalize in place; returns the pre-normalization norm.
pub(crate) fn l2_normalize(v: &mut [f32]) -> f32 {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    norm
}

/// Truncate to `dim` and renormalize (Matryoshka). No-op if `dim >= len`.
pub(crate) fn matryoshka_truncate(v: &mut Vec<f32>, dim: usize) {
    if dim < v.len() {
        v.truncate(dim);
        l2_normalize(v);
    }
}

/// Curated specs for the launch set. Returns `None` for ids handled by auto-detection.
fn curated_spec(hf_id: &str) -> Option<ModelSpec> {
    let mit = |s: &str| s.to_string();
    match hf_id {
        "sentence-transformers/all-MiniLM-L6-v2" => Some(ModelSpec {
            hf_id: mit(hf_id),
            arch: Architecture::Bert,
            pooling: Pooling::Mean,
            query_prefix: None,
            doc_prefix: None,
            native_dim: 384,
            output_dim: None,
            license: mit("apache-2.0"),
            resource_class: ResourceClass::Cpu,
            gated: false,
        }),
        "BAAI/bge-m3" => Some(ModelSpec {
            hf_id: mit(hf_id),
            arch: Architecture::XlmRoberta,
            pooling: Pooling::Cls,
            query_prefix: None,
            doc_prefix: None,
            native_dim: 1024,
            output_dim: None,
            license: mit("mit"),
            resource_class: ResourceClass::Cpu,
            gated: false,
        }),
        "Qwen/Qwen3-Embedding-0.6B" | "Qwen/Qwen3-Embedding-4B" | "Qwen/Qwen3-Embedding-8B" => {
            let native_dim = match hf_id {
                "Qwen/Qwen3-Embedding-0.6B" => 1024,
                "Qwen/Qwen3-Embedding-4B" => 2560,
                _ => 4096,
            };
            let resource_class = if hf_id == "Qwen/Qwen3-Embedding-0.6B" {
                ResourceClass::Cpu
            } else {
                ResourceClass::Gpu
            };
            Some(ModelSpec {
                hf_id: mit(hf_id),
                arch: Architecture::Qwen3,
                pooling: Pooling::LastToken,
                query_prefix: Some(mit(
                    "Instruct: Given a web search query, retrieve relevant passages that \
                     answer the query\nQuery:",
                )),
                doc_prefix: None,
                native_dim,
                output_dim: None,
                license: mit("apache-2.0"),
                resource_class,
                gated: false,
            })
        }
        "google/embeddinggemma-300m" => Some(ModelSpec {
            hf_id: mit(hf_id),
            arch: Architecture::Gemma3Bidirectional,
            pooling: Pooling::Mean,
            query_prefix: Some(mit("task: search result | query: ")),
            doc_prefix: Some(mit("title: none | text: ")),
            native_dim: 768,
            output_dim: None,
            license: mit("gemma"),
            resource_class: ResourceClass::Cpu,
            gated: true,
        }),
        "LiquidAI/LFM2.5-Embedding-350M" => Some(ModelSpec {
            hf_id: mit(hf_id),
            arch: Architecture::Lfm2Bidirectional,
            pooling: Pooling::Cls,
            query_prefix: Some(mit("query: ")),
            doc_prefix: Some(mit("document: ")),
            native_dim: 1024,
            output_dim: None,
            license: mit("lfm-1.0"),
            resource_class: ResourceClass::Cpu,
            gated: false,
        }),
        _ => None,
    }
}

/// HuggingFace ids of the curated v0.5 launch set, in registry order.
pub const LAUNCH_MODEL_IDS: &[&str] = &[
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-m3",
    "Qwen/Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Embedding-4B",
    "Qwen/Qwen3-Embedding-8B",
    "google/embeddinggemma-300m",
    "LiquidAI/LFM2.5-Embedding-350M",
];

/// The curated launch-set specs. Used by GalaxDB Cloud to mirror the engine registry.
pub fn registry_specs() -> Vec<ModelSpec> {
    LAUNCH_MODEL_IDS
        .iter()
        .filter_map(|id| curated_spec(id))
        .collect()
}

/// Read pooling from a sentence-transformers `1_Pooling/config.json` if present.
fn detect_pooling(repo: &ApiRepo) -> Option<Pooling> {
    let path = repo.get("1_Pooling/config.json").ok()?;
    let json: serde_json::Value = serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
    let flag = |k: &str| json.get(k).and_then(|v| v.as_bool()).unwrap_or(false);
    if flag("pooling_mode_cls_token") {
        Some(Pooling::Cls)
    } else if flag("pooling_mode_lasttoken") {
        Some(Pooling::LastToken)
    } else if flag("pooling_mode_mean_tokens") {
        Some(Pooling::Mean)
    } else {
        None
    }
}

/// Read query/document prompts from `config_sentence_transformers.json` if present.
fn detect_prompts(repo: &ApiRepo) -> (Option<String>, Option<String>) {
    let Ok(path) = repo.get("config_sentence_transformers.json") else {
        return (None, None);
    };
    let Ok(bytes) = std::fs::read(path) else {
        return (None, None);
    };
    let Ok(json) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return (None, None);
    };
    let prompts = json.get("prompts");
    let get = |key: &str| {
        prompts
            .and_then(|p| p.get(key))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    };
    let query = get("query").or_else(|| get("Retrieval-query"));
    let doc = get("document").or_else(|| get("Retrieval-document"));
    (query, doc)
}

/// Auto-detect a spec for an id not in the curated set, from its repo config.
///
/// Reads `config.json` (`model_type` + `hidden_size`), sentence-transformers pooling and
/// prompt configs. Returns a typed error for architectures the sidecar cannot build — no
/// silent fallback to a wrong pooling or a mock.
fn autodetect_spec(hf_id: &str, repo: &ApiRepo) -> Result<ModelSpec, BoxError> {
    let config_path = repo.get("config.json")?;
    let config: serde_json::Value = serde_json::from_slice(&std::fs::read(config_path)?)?;

    let model_type = config
        .get("model_type")
        .and_then(|v| v.as_str())
        .ok_or("config.json missing model_type; cannot determine architecture")?;
    let arch = Architecture::from_model_type(model_type).ok_or_else(|| {
        format!(
            "unsupported model architecture '{model_type}' for '{hf_id}'. Supported: \
             bert, xlm-roberta, qwen3, gemma3_text, lfm2. Add a loader to support it."
        )
    })?;

    let native_dim = config
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .ok_or("config.json missing hidden_size")? as usize;

    // Default pooling per architecture when the repo ships no ST pooling config.
    let default_pooling = match arch {
        Architecture::Bert => Pooling::Mean,
        Architecture::XlmRoberta => Pooling::Cls,
        Architecture::Qwen3 => Pooling::LastToken,
        Architecture::Gemma3Bidirectional => Pooling::Mean,
        Architecture::Lfm2Bidirectional => Pooling::Cls,
    };
    let pooling = detect_pooling(repo).unwrap_or(default_pooling);
    let (query_prefix, doc_prefix) = detect_prompts(repo);

    Ok(ModelSpec {
        hf_id: hf_id.to_string(),
        arch,
        pooling,
        query_prefix,
        doc_prefix,
        native_dim,
        output_dim: None,
        license: config
            .get("license")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string(),
        resource_class: ResourceClass::Cpu,
        gated: false,
    })
}

/// Resolve the [`ModelSpec`] for an id: curated first, else auto-detected from the repo.
pub fn resolve_spec(hf_id: &str, repo: &ApiRepo) -> Result<ModelSpec, BoxError> {
    if let Some(spec) = curated_spec(hf_id) {
        return Ok(spec);
    }
    autodetect_spec(hf_id, repo)
}

/// Load a model by HuggingFace id on the given device. Downloads via the HF Hub API.
///
/// Returns a typed error (never a mock) if the id is unsupported or any load step fails.
pub fn load(hf_id: &str, device: &candle_core::Device) -> Result<Loaded, BoxError> {
    use hf_hub::{api::sync::Api, Repo, RepoType};
    let api = Api::new()?;
    let repo = api.repo(Repo::new(hf_id.to_string(), RepoType::Model));

    let spec = resolve_spec(hf_id, &repo)?;

    let embedder: Box<dyn TextEmbedder> = match spec.arch {
        Architecture::Bert => Box::new(bert::BertEmbedder::load(&repo, &spec, device)?),
        Architecture::XlmRoberta => {
            Box::new(xlm_roberta::XlmRobertaEmbedder::load(&repo, &spec, device)?)
        }
        Architecture::Qwen3 => Box::new(qwen3::Qwen3Embedder::load(&repo, &spec, device)?),
        Architecture::Gemma3Bidirectional => {
            Box::new(gemma3_bidir::Gemma3Embedder::load(&repo, &spec, device)?)
        }
        Architecture::Lfm2Bidirectional => {
            Box::new(lfm2_bidir::Lfm2Embedder::load(&repo, &spec, device)?)
        }
    };

    Ok(Loaded { embedder, spec })
}

/// Apply the appropriate prefix for a query/document, if the spec defines one.
pub(crate) fn apply_prefix(spec: &ModelSpec, text: &str, is_query: bool) -> String {
    let prefix = if is_query {
        spec.query_prefix.as_deref()
    } else {
        spec.doc_prefix.as_deref()
    };
    match prefix {
        Some(p) => format!("{p}{text}"),
        None => text.to_string(),
    }
}

/// Load tensors from a safetensors file into an F32 map on `device`.
pub(crate) fn load_safetensors_f32(
    path: &std::path::Path,
    device: &candle_core::Device,
) -> Result<HashMap<String, candle_core::Tensor>, BoxError> {
    let raw = candle_core::safetensors::load(path, device)?;
    let mut out = HashMap::with_capacity(raw.len());
    for (k, v) in raw {
        out.insert(k, v.to_dtype(candle_core::DType::F32)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_registry_is_complete_and_consistent() {
        let specs = registry_specs();
        // Every curated launch id resolves to a spec (no typos in the id table).
        assert_eq!(specs.len(), LAUNCH_MODEL_IDS.len());
        for spec in &specs {
            assert!(spec.native_dim > 0);
            assert_eq!(spec.effective_dim(), spec.native_dim); // no Matryoshka default
            // Asymmetric families must define at least a query prefix.
            match spec.arch {
                Architecture::Qwen3
                | Architecture::Gemma3Bidirectional
                | Architecture::Lfm2Bidirectional => {
                    assert!(
                        spec.query_prefix.is_some(),
                        "{} is asymmetric but has no query prefix",
                        spec.hf_id
                    );
                }
                Architecture::Bert | Architecture::XlmRoberta => {}
            }
        }
    }

    #[test]
    fn model_spec_round_trips_through_json() {
        // Cloud mirrors the registry over JSON — the contract must be stable.
        let spec = curated_spec("google/embeddinggemma-300m").unwrap();
        let json = serde_json::to_string(&spec).unwrap();
        let back: ModelSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.hf_id, spec.hf_id);
        assert_eq!(back.arch, Architecture::Gemma3Bidirectional);
        assert_eq!(back.pooling, Pooling::Mean);
        assert_eq!(back.native_dim, 768);
        assert!(back.gated);
    }

    #[test]
    fn architecture_maps_from_model_type() {
        assert_eq!(Architecture::from_model_type("bert"), Some(Architecture::Bert));
        assert_eq!(
            Architecture::from_model_type("gemma3_text"),
            Some(Architecture::Gemma3Bidirectional)
        );
        assert_eq!(Architecture::from_model_type("lfm2"), Some(Architecture::Lfm2Bidirectional));
        assert_eq!(Architecture::from_model_type("mamba"), None);
    }

    #[test]
    fn prefix_applied_only_when_defined() {
        let sym = curated_spec("sentence-transformers/all-MiniLM-L6-v2").unwrap();
        assert_eq!(apply_prefix(&sym, "hello", true), "hello");
        let asym = curated_spec("LiquidAI/LFM2.5-Embedding-350M").unwrap();
        assert_eq!(apply_prefix(&asym, "hello", true), "query: hello");
        assert_eq!(apply_prefix(&asym, "hello", false), "document: hello");
    }
}
