//! GalaxDB Embedding Sidecar — standalone binary for sentence-transformer inference.
//!
//! Uses Candle (HuggingFace's pure Rust ML framework) for cross-platform inference.
//! No external runtime needed — works on macOS and Linux.
//!
//! This binary is Unix-only (uses Unix sockets for IPC). On Windows, users
//! run the server on Linux/macOS (Docker or WSL2) and connect via the wire protocol.

// The sidecar binary is Unix-only. On Windows this compiles to an empty
// binary that immediately exits — the real functionality requires Unix sockets.
#![cfg(unix)]
//! This binary:
//! 1. Downloads and loads a sentence-transformer model from HuggingFace Hub
//!    (`sentence-transformers/all-MiniLM-L6-v2` by default, any BERT-style
//!    sentence-transformer can be specified via `--model`)
//! 2. Listens on a Unix socket for embedding requests
//! 3. Responds with embeddings + model version
//! 4. Monitors parent PID and exits if parent dies
//!
//! If the model fails to load the process exits with status code 1 and a
//! typed error on stderr. There is **no mock fallback** — every embedding
//! this sidecar emits is computed by the real model.
//!
//! Usage:
//!
//! ```text
//! galaxdb-sidecar --socket /path/to/socket --model sentence-transformers/all-MiniLM-L6-v2
//! ```

use std::io::{BufReader, BufWriter};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

use galaxdb_sidecar::protocol::*;

/// Process exit code used when the model fails to load.
///
/// The parent `SidecarManager` observes this via `Child::try_wait` and then
/// enters degraded mode. Callers get a typed `GalaxError` — never fake
/// embeddings.
const EXIT_MODEL_LOAD_FAILED: i32 = 1;

/// Loaded sentence-transformer model (Candle + tokenizer).
struct EmbeddingModel {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    dim: usize,
    model_id: String,
}

impl EmbeddingModel {
    /// Load a sentence-transformer model from HuggingFace Hub.
    ///
    /// Downloads `config.json`, `tokenizer.json`, and `model.safetensors`
    /// from the HF Hub cache. Returns an error (never a mock) if any part
    /// of the download, parse, or weight-load step fails.
    fn load(model_id: &str) -> Result<Self, Box<dyn std::error::Error>> {
        eprintln!("[sidecar] downloading model: {}", model_id);
        let api = Api::new()?;
        let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

        let config_path = repo.get("config.json")?;
        let tokenizer_path = repo.get("tokenizer.json")?;
        let weights_path = repo.get("model.safetensors")?;

        eprintln!("[sidecar] loading config...");
        let config: BertConfig =
            serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
        let dim = config.hidden_size;

        eprintln!("[sidecar] loading tokenizer...");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("tokenizer load failed: {}", e))?;

        eprintln!("[sidecar] loading weights ({})...", weights_path.display());
        let device = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[weights_path],
                candle_core::DType::F32,
                &device,
            )?
        };

        let model = BertModel::load(vb, &config)?;
        eprintln!("[sidecar] model loaded: dim={}", dim);

        Ok(Self {
            model,
            tokenizer,
            device,
            dim,
            model_id: model_id.to_string(),
        })
    }

    /// Generate an L2-normalized embedding for `text`. Returns the
    /// pooled, normalized vector or an error describing which step
    /// (tokenize / forward / pool) failed.
    fn embed(&self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| format!("tokenize failed: {}", e))?;
        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        let attention_mask: Vec<u32> = encoding.get_attention_mask().to_vec();
        let token_type_ids: Vec<u32> = encoding.get_type_ids().to_vec();
        let seq_len = input_ids.len();

        let input_ids_t =
            Tensor::new(input_ids.as_slice(), &self.device)?.reshape((1, seq_len))?;
        let attention_mask_t =
            Tensor::new(attention_mask.as_slice(), &self.device)?.reshape((1, seq_len))?;
        let token_type_ids_t =
            Tensor::new(token_type_ids.as_slice(), &self.device)?.reshape((1, seq_len))?;

        let output =
            self.model
                .forward(&input_ids_t, &token_type_ids_t, Some(&attention_mask_t))?;

        // Mean pooling with attention mask.
        let mask_f32 = attention_mask_t
            .to_dtype(candle_core::DType::F32)?
            .unsqueeze(2)?
            .broadcast_as(output.shape())?;
        let masked = (output * mask_f32.clone())?;
        let summed = masked.sum(1)?;
        let mask_sum = mask_f32.sum(1)?;
        let pooled = (summed / mask_sum)?;

        // Extract and L2-normalize.
        let mut vec: Vec<f32> = pooled.squeeze(0)?.to_vec1()?;
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            for x in vec.iter_mut() {
                *x /= norm;
            }
        }
        Ok(vec)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let socket_path = get_arg(&args, "--socket")
        .unwrap_or_else(|| "/tmp/galaxdb_sidecar.sock".to_string());
    let model_id = get_arg(&args, "--model")
        .unwrap_or_else(|| "sentence-transformers/all-MiniLM-L6-v2".to_string());
    let parent_pid: Option<u32> = get_arg(&args, "--parent-pid")
        .and_then(|s| s.parse().ok());

    eprintln!("[sidecar] starting: socket={}", socket_path);

    // Load the real model. If the load fails for any reason, the process
    // exits with status 1 and a typed error on stderr. There is no mock
    // fallback — callers must get a real embedding or a typed error.
    let model = match EmbeddingModel::load(&model_id) {
        Ok(m) => Arc::new(m),
        Err(e) => {
            eprintln!(
                "[sidecar] FATAL: failed to load model '{}': {}",
                model_id, e
            );
            eprintln!(
                "[sidecar] verify (1) the model id is correct, \
                 (2) network access to huggingface.co, \
                 (3) disk space for the HF cache. Exiting."
            );
            std::process::exit(EXIT_MODEL_LOAD_FAILED);
        }
    };

    let dimensions = model.dim;
    let model_version = model.model_id.clone();

    // Remove stale socket file.
    let _ = std::fs::remove_file(&socket_path);

    // Bind Unix socket.
    let listener = UnixListener::bind(&socket_path).unwrap_or_else(|e| {
        eprintln!(
            "[sidecar] FATAL: failed to bind Unix socket at {}: {}",
            socket_path, e
        );
        std::process::exit(EXIT_MODEL_LOAD_FAILED);
    });
    eprintln!("[sidecar] listening on {}", socket_path);

    let running = Arc::new(AtomicBool::new(true));
    let in_flight = Arc::new(AtomicUsize::new(0));

    // Parent PID monitoring.
    #[cfg(target_os = "linux")]
    if let Some(ppid) = parent_pid {
        setup_parent_monitor_linux(ppid, running.clone());
    }
    #[cfg(target_os = "macos")]
    if let Some(ppid) = parent_pid {
        setup_parent_monitor_macos(ppid, running.clone());
    }

    listener.set_nonblocking(false).ok();

    while running.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((stream, _)) => {
                let dims = dimensions;
                let version = model_version.clone();
                let in_flight = in_flight.clone();
                let running = running.clone();
                let model = model.clone();

                std::thread::spawn(move || {
                    handle_connection(stream, dims, &version, &in_flight, &running, &model);
                });
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(e) => {
                eprintln!("[sidecar] accept error: {}", e);
                break;
            }
        }
    }

    let _ = std::fs::remove_file(&socket_path);
    eprintln!("[sidecar] shutdown");
}

fn handle_connection(
    stream: UnixStream,
    dimensions: usize,
    model_version: &str,
    in_flight: &AtomicUsize,
    running: &AtomicBool,
    model: &EmbeddingModel,
) {
    let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
    let mut writer = BufWriter::new(stream);

    while running.load(Ordering::Relaxed) {
        let msg = match read_message(&mut reader) {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => {
                eprintln!("[sidecar] read error: {}", e);
                break;
            }
        };

        let response = match msg {
            SidecarMessage::EmbedRequest(req) => {
                let current = in_flight.fetch_add(1, Ordering::Relaxed);
                if current >= MAX_IN_FLIGHT {
                    in_flight.fetch_sub(1, Ordering::Relaxed);
                    SidecarMessage::Error {
                        message: "max in-flight exceeded".to_string(),
                    }
                } else {
                    let result = model.embed(&req.text);
                    in_flight.fetch_sub(1, Ordering::Relaxed);
                    match result {
                        Ok(embedding) => SidecarMessage::EmbedResponse(EmbedResponse {
                            row_id: req.row_id,
                            embedding,
                            model_version: model_version.to_string(),
                        }),
                        Err(e) => SidecarMessage::Error {
                            message: format!("embed failed: {}", e),
                        },
                    }
                }
            }
            SidecarMessage::HeartbeatPong(_) => continue,
            SidecarMessage::StatusRequest(_) => SidecarMessage::StatusResponse(StatusResponse {
                model_id: model_version.to_string(),
                model_version: model_version.to_string(),
                dimensions,
                in_flight: in_flight.load(Ordering::Relaxed),
                max_in_flight: MAX_IN_FLIGHT,
            }),
            _ => SidecarMessage::Error {
                message: "unexpected message type".to_string(),
            },
        };

        if let Err(e) = write_message(&mut writer, &response) {
            eprintln!("[sidecar] write error: {}", e);
            break;
        }
    }
}

fn get_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

#[cfg(target_os = "linux")]
fn setup_parent_monitor_linux(parent_pid: u32, running: Arc<AtomicBool>) {
    use std::os::raw::c_int;
    unsafe extern "C" {
        fn prctl(option: c_int, arg2: c_int) -> c_int;
    }
    unsafe {
        let result = prctl(1, 15);
        if result != 0 {
            eprintln!("[sidecar] WARNING: prctl(PR_SET_PDEATHSIG) failed");
        } else {
            eprintln!(
                "[sidecar] parent PID monitoring active (prctl): ppid={}",
                parent_pid
            );
        }
    }
    let current_ppid = unsafe { libc::getppid() } as u32;
    if current_ppid != parent_pid {
        eprintln!("[sidecar] parent already exited");
        running.store(false, Ordering::SeqCst);
    }
}

#[cfg(target_os = "macos")]
fn setup_parent_monitor_macos(parent_pid: u32, running: Arc<AtomicBool>) {
    eprintln!(
        "[sidecar] parent PID monitoring active (poll): ppid={}",
        parent_pid
    );
    std::thread::spawn(move || loop {
        std::thread::sleep(Duration::from_secs(2));
        let ppid = unsafe { libc::getppid() } as u32;
        if ppid != parent_pid {
            eprintln!("[sidecar] parent exited (ppid changed), shutting down");
            running.store(false, Ordering::SeqCst);
            break;
        }
    });
}
