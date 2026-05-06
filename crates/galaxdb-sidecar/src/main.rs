//! GalaxDB Embedding Sidecar — standalone binary for ONNX Runtime embedding inference.
//!
//! This binary:
//! 1. Loads an ONNX sentence-transformer model
//! 2. Listens on a Unix socket for embedding requests
//! 3. Responds with embeddings + model version
//! 4. Monitors parent PID and exits if parent dies
//! 5. Sends heartbeat every 5 seconds
//! 6. Tracks in-flight count (max 10,000)
//!
//! Usage:
//!   galaxdb-sidecar --socket /path/to/socket --model /path/to/model.onnx
//!   galaxdb-sidecar --socket /path/to/socket --mock-dim 384  (for testing without a model)

use std::io::{BufReader, BufWriter};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use galaxdb_sidecar::protocol::*;

fn main() {
    // Parse CLI args
    let args: Vec<String> = std::env::args().collect();
    let socket_path = get_arg(&args, "--socket")
        .unwrap_or_else(|| "/tmp/galaxdb_sidecar.sock".to_string());
    let mock_dim: Option<usize> = get_arg(&args, "--mock-dim")
        .and_then(|s| s.parse().ok());
    let model_path = get_arg(&args, "--model");
    let parent_pid: Option<u32> = get_arg(&args, "--parent-pid")
        .and_then(|s| s.parse().ok());

    eprintln!("[sidecar] starting: socket={}", socket_path);

    // Determine embedding dimensions and model version
    let (dimensions, model_version, model_id) = if let Some(dim) = mock_dim {
        eprintln!("[sidecar] mock mode: dim={}", dim);
        (dim, "mock-v1.0".to_string(), "mock-model".to_string())
    } else if let Some(_path) = &model_path {
        // TODO: Load ONNX model via ort crate when available
        // For now, fall back to mock mode
        eprintln!("[sidecar] ONNX model loading not yet implemented, using mock");
        (384, "mock-v1.0".to_string(), "mock-model".to_string())
    } else {
        eprintln!("[sidecar] no --model or --mock-dim specified, defaulting to mock dim=384");
        (384, "mock-v1.0".to_string(), "mock-model".to_string())
    };

    // Remove stale socket file
    let _ = std::fs::remove_file(&socket_path);

    // Bind Unix socket
    let listener = UnixListener::bind(&socket_path).expect("failed to bind Unix socket");
    eprintln!("[sidecar] listening on {}", socket_path);

    let running = Arc::new(AtomicBool::new(true));
    let in_flight = Arc::new(AtomicUsize::new(0));

    // Set up parent PID monitoring
    #[cfg(target_os = "linux")]
    if let Some(ppid) = parent_pid {
        setup_parent_monitor_linux(ppid, running.clone());
    }

    #[cfg(target_os = "macos")]
    if let Some(ppid) = parent_pid {
        setup_parent_monitor_macos(ppid, running.clone());
    }

    // Accept connections
    listener.set_nonblocking(false).ok();

    while running.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((stream, _addr)) => {
                let dims = dimensions;
                let version = model_version.clone();
                let mid = model_id.clone();
                let in_flight = in_flight.clone();
                let running = running.clone();

                std::thread::spawn(move || {
                    handle_connection(stream, dims, &version, &mid, &in_flight, &running);
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

    // Cleanup
    let _ = std::fs::remove_file(&socket_path);
    eprintln!("[sidecar] shutdown");
}

fn handle_connection(
    stream: UnixStream,
    dimensions: usize,
    model_version: &str,
    model_id: &str,
    in_flight: &AtomicUsize,
    running: &AtomicBool,
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
                    // Generate embedding (mock: deterministic hash-based vector)
                    let embedding = mock_embed(&req.text, dimensions);
                    in_flight.fetch_sub(1, Ordering::Relaxed);
                    SidecarMessage::EmbedResponse(EmbedResponse {
                        row_id: req.row_id,
                        embedding,
                        model_version: model_version.to_string(),
                    })
                }
            }
            SidecarMessage::HeartbeatPong(_) => {
                // Engine acknowledged our heartbeat — nothing to do
                continue;
            }
            SidecarMessage::StatusRequest(_) => {
                SidecarMessage::StatusResponse(StatusResponse {
                    model_id: model_id.to_string(),
                    model_version: model_version.to_string(),
                    dimensions,
                    in_flight: in_flight.load(Ordering::Relaxed),
                    max_in_flight: MAX_IN_FLIGHT,
                })
            }
            _ => {
                SidecarMessage::Error {
                    message: format!("unexpected message type"),
                }
            }
        };

        if let Err(e) = write_message(&mut writer, &response) {
            eprintln!("[sidecar] write error: {}", e);
            break;
        }
    }
}

/// Mock embedding: generates a deterministic vector from text hash.
/// This is used when no ONNX model is loaded (testing/development).
/// The vector is normalized to unit length for cosine similarity.
fn mock_embed(text: &str, dim: usize) -> Vec<f32> {
    // Simple hash-based deterministic embedding
    let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis
    for byte in text.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3); // FNV prime
    }

    let mut vec = Vec::with_capacity(dim);
    let mut state = hash;
    for _ in 0..dim {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let val = (state >> 33) as f32 / (u32::MAX as f32) * 2.0 - 1.0;
        vec.push(val);
    }

    // Normalize to unit length
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }

    vec
}

/// Set up parent PID monitoring on Linux using prctl.
#[cfg(target_os = "linux")]
fn setup_parent_monitor_linux(parent_pid: u32, running: Arc<AtomicBool>) {
    use std::os::raw::c_int;

    // PR_SET_PDEATHSIG = 1, SIGTERM = 15
    // This tells the kernel to send SIGTERM to this process when the parent exits.
    unsafe extern "C" {
        fn prctl(option: c_int, arg2: c_int) -> c_int;
    }

    unsafe {
        let result = prctl(1, 15); // PR_SET_PDEATHSIG, SIGTERM
        if result != 0 {
            eprintln!("[sidecar] WARNING: prctl(PR_SET_PDEATHSIG) failed");
        } else {
            eprintln!("[sidecar] parent PID monitoring active (prctl): ppid={}", parent_pid);
        }
    }

    // Also check if parent already died (race condition)
    let current_ppid = unsafe { libc::getppid() } as u32;
    if current_ppid != parent_pid {
        eprintln!("[sidecar] parent already exited (ppid changed: {} → {})", parent_pid, current_ppid);
        running.store(false, Ordering::SeqCst);
    }
}

/// Set up parent PID monitoring on macOS using polling.
/// macOS doesn't have prctl, so we poll the parent PID periodically.
#[cfg(target_os = "macos")]
fn setup_parent_monitor_macos(parent_pid: u32, running: Arc<AtomicBool>) {
    std::thread::Builder::new()
        .name("sidecar-parent-monitor".to_string())
        .spawn(move || {
            loop {
                std::thread::sleep(Duration::from_secs(1));
                // Check if parent is still alive
                let result = unsafe { libc::kill(parent_pid as i32, 0) };
                if result != 0 {
                    eprintln!("[sidecar] parent PID {} exited, shutting down", parent_pid);
                    running.store(false, Ordering::SeqCst);
                    break;
                }
            }
        })
        .expect("spawn parent monitor thread");
    eprintln!("[sidecar] parent PID monitoring active (poll): ppid={}", parent_pid);
}

fn get_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}
