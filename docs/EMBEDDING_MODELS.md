# Embedding Models (GalaxDB v0.5)

GalaxDB's embedding sidecar is multi-architecture. It loads a real model from the
HuggingFace Hub by id at startup (`galaxdb-sidecar --model <hf-id>`), computes real vectors,
and exits with a typed error if the id is unknown or its architecture is unsupported. There is
no mock mode and no silent fallback — every embedding is produced by the loaded model.

The default model is unchanged: `sentence-transformers/all-MiniLM-L6-v2` (BERT, mean pooling,
384-d). Existing databases open exactly as before.

## Launch set (verified)

Each model below was verified end-to-end through the same `galaxdb_sidecar::models::load` path
the binary uses (see `crates/galaxdb-sidecar/tests/models_online.rs` and the spike probes under
`crates/galaxdb-sidecar/examples/`). "Verified cosine" is `cos(query, relevant)` vs
`cos(query, unrelated)` on a fixed probe pair (CPU, F32).

| Model | Arch | Dim | Pooling | Query prefix | Doc prefix | License | Class | Verified cosine (rel / unrel) |
|---|---|---|---|---|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | BERT | 384 | mean | — | — | apache-2.0 | CPU | 0.7546 / 0.1026 |
| `BAAI/bge-m3` | XLM-RoBERTa | 1024 | CLS | — | — | mit | CPU | 0.74 / 0.36 |
| `Qwen/Qwen3-Embedding-0.6B` | Qwen3 decoder | 1024 | last-token | `Instruct: …\nQuery:` | — | apache-2.0 | CPU | 0.76 / 0.16 |
| `Qwen/Qwen3-Embedding-4B` | Qwen3 decoder | 2560 | last-token | `Instruct: …\nQuery:` | — | apache-2.0 | GPU | same loader as 0.6B |
| `Qwen/Qwen3-Embedding-8B` | Qwen3 decoder | 4096 | last-token | `Instruct: …\nQuery:` | — | apache-2.0 | GPU | same loader as 0.6B |
| `google/embeddinggemma-300m` | Gemma 3 (bidirectional) + Dense | 768 | mean | `task: search result \| query: ` | `title: none \| text: ` | gemma (gated) | CPU | 0.5825 / 0.0790 |
| `LiquidAI/LFM2.5-Embedding-350M` | LFM2 hybrid (bidirectional) | 1024 | CLS | `query: ` | `document: ` | lfm-1.0 | CPU | 0.4248 / -0.0225 |

Notes:

- **Asymmetric models** (Qwen3, EmbeddingGemma, LFM2.5) apply a different prefix to search
  queries vs stored documents. GalaxDB sends the right one automatically: `SEMANTIC_MATCH`
  query text is embedded as a query, `INSERT`ed rows are embedded as documents (the sidecar
  protocol carries an `is_query` flag).
- **EmbeddingGemma** and **LFM2.5** are custom **bidirectional** encoders. candle's stock
  `gemma3`/`lfm2` modules are causal LMs; GalaxDB forks their blocks into encoders (no causal
  mask; LFM2's short-conv made symmetric). EmbeddingGemma also runs the two sentence-transformers
  Dense heads (768→3072→768) and supports Matryoshka truncation via `ModelSpec.output_dim`.
- **EmbeddingGemma is gated.** Downloading its weights needs an accepted Google license and an
  HF token (`HF_TOKEN` / `HUGGINGFACE_TOKEN`, or `~/.cache/huggingface/token`).
- **Qwen3 4B/8B** use the same loader as 0.6B and carry up to 4096-d vectors. They are in engine
  scope; run them on a workstation (Metal) or a GPU/CPU box. What GalaxDB Cloud offers per tier
  is a separate deployment decision.
- Exact numeric parity against the sentence-transformers reference for the two custom encoders
  is validated on the Linux CI/AWS box (PyTorch has no Intel-macOS wheels for the local dev
  machine). Structural checks (dim, unit norm) and semantic ordering pass on every platform.

## Selecting a model

```text
galaxdb-sidecar --socket /path/to.sock --model BAAI/bge-m3
```

Any id not in the curated set above is **auto-detected** from its repo: GalaxDB reads
`config.json` (`model_type`, `hidden_size`), the sentence-transformers `1_Pooling/config.json`
(pooling mode), and `config_sentence_transformers.json` (query/document prompts). If the
`model_type` is one GalaxDB has a loader for (`bert`, `xlm-roberta`, `qwen3`, `gemma3_text`,
`lfm2`) it loads; otherwise it exits with a typed "unsupported architecture" error listing what
is supported. This means most additional BERT / XLM-RoBERTa sentence-transformers work with no
code change.

The declared column `DIM` must equal the model's real output dimension. On mismatch, `INSERT`
fails with a clear error naming both dimensions (dimension-integrity check) rather than storing
a wrong-width vector.

## Adding a new architecture

The registry is designed so a **new model on an existing architecture** is just a curated
registry entry (or nothing — auto-detection may already cover it), and a **new architecture** is
a thin loader plus one enum variant. Checklist:

1. **Probe first.** Add `crates/galaxdb-sidecar/examples/probe_<model>.rs` that loads the real
   weights, embeds a fixed probe set, and checks dim / unit-norm / semantic ordering (and, on
   Linux, cosine parity vs the reference). Do not proceed until it passes. Read the model's
   `config.json` and tensor shapes — never assume architecture details.
2. **Add an `Architecture` variant** in `models/mod.rs` (the enum is `#[non_exhaustive]`), and
   map its HF `model_type` in `Architecture::from_model_type` + `Architecture::as_str`.
3. **Add a loader module** `models/<arch>.rs` implementing `TextEmbedder` (own the weights +
   tokenizer; apply `apply_prefix`, `l2_normalize`, and `matryoshka_truncate` from `mod.rs`).
4. **Dispatch it** in `models::load`'s `match spec.arch`.
5. **Register a curated `ModelSpec`** in `curated_spec` (arch, pooling, prefixes, native_dim,
   output_dim, license, resource_class, gated) and add the id to `LAUNCH_MODEL_IDS` if it ships.
6. **Pooling** lives in the `Pooling` enum (`#[non_exhaustive]`); add a variant only for a
   genuinely new pooling mode, and handle it in every loader that can use it.
7. **Add a fidelity test** arm in `tests/models_online.rs` (real load, dim, norm, ordering).
8. **Document it** in the table above with a reproducible verified-cosine number.

`ModelSpec`, `Architecture`, `Pooling`, and `ResourceClass` are serde-serializable so the
GalaxDB Cloud control plane can mirror this registry (`models::registry_specs()`) without
copying constants.
