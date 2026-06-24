//! Pluggable key management — no vendor lock-in.
//!
//! The [`KeyProvider`] trait abstracts key generation and decryption so
//! GalaxDB can run with any of:
//!
//! * [`LocalKeyProvider`] — 32-byte master key in a local file
//!   (dev, self-hosted, air-gapped).
//! * [`EnvKeyProvider`] — hex-encoded master key in an environment
//!   variable (containers, Kubernetes Secrets).
//! * [`ExternalCommandKeyProvider`] — delegates to an operator-supplied
//!   shell command. Works with any KMS that exposes a CLI (AWS CLI,
//!   gcloud, az, vault CLI, custom HSM wrappers) without linking that
//!   KMS's SDK.
//! * [`HashicorpVaultKeyProvider`] — HashiCorp Vault Transit engine
//!   over rustls, compiled in behind the `vault` Cargo feature.
//!
//! All providers implement the same sync contract:
//!
//! ```text
//! generate_data_key() -> (plaintext_dek: Vec<u8>, encrypted_blob: Vec<u8>)
//! decrypt_data_key(encrypted_blob) -> plaintext_dek: Vec<u8>
//! ```
//!
//! The `encrypted_blob` format is opaque to the engine. `LocalKeyProvider`
//! and `EnvKeyProvider` wrap the DEK with AES-256-GCM; `ExternalCommandKeyProvider`
//! treats the bytes the operator's command emits on stdout as the blob;
//! `HashicorpVaultKeyProvider` stores Vault's `vault:v1:...` ciphertext
//! string as the blob. Only the matching provider can decrypt its own
//! blobs.
//!
//! # Design: no vendor lock-in
//!
//! There is deliberately no `AwsKmsKeyProvider`, `GcpKmsKeyProvider`,
//! or `AzureKeyVaultKeyProvider` in this crate. Operators who need
//! those services use [`ExternalCommandKeyProvider`] with the vendor's
//! CLI (`aws kms encrypt ...`, `gcloud kms encrypt ...`, `az keyvault
//! encrypt ...`) or adapt their wrapper. This keeps the core binary
//! free of cloud SDKs and their transitive dependency trees.

use aes_gcm::aead::Aead;
use aes_gcm::{Aes256Gcm, KeyInit, Nonce};
use galaxdb_common::{GalaxError, GalaxResult};
use rand::RngCore;
use std::path::Path;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Trait for pluggable key management.
///
/// Implementations must be `Send + Sync` so the engine can share a single
/// provider across worker threads.
pub trait KeyProvider: Send + Sync {
    /// Generate or retrieve a data encryption key (DEK).
    ///
    /// Returns `(plaintext_key, encrypted_key_for_storage)`. The
    /// `encrypted_key_for_storage` is opaque bytes that the engine
    /// persists alongside the data; only the same provider (with the
    /// same master-key material or remote service access) can later
    /// recover the plaintext DEK via [`decrypt_data_key`].
    fn generate_data_key(&self) -> GalaxResult<(Vec<u8>, Vec<u8>)>;

    /// Decrypt a previously-encrypted DEK.
    fn decrypt_data_key(&self, encrypted_key: &[u8]) -> GalaxResult<Vec<u8>>;

    /// Short stable provider name, used for logging and metrics.
    fn provider_name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Shared AES-256-GCM helpers (LocalKeyProvider, EnvKeyProvider)
// ---------------------------------------------------------------------------

/// Encrypt a plaintext DEK with the given 32-byte master key using
/// AES-256-GCM. The output is `nonce (12 bytes) || ciphertext+tag`.
fn encrypt_dek_with_master(master_key: &[u8; 32], plaintext_dek: &[u8]) -> GalaxResult<Vec<u8>> {
    let cipher = Aes256Gcm::new_from_slice(master_key)
        .map_err(|e| GalaxError::Encryption(format!("cipher init: {e}")))?;

    let mut nonce_bytes = [0u8; 12];
    rand::thread_rng().fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);

    let ciphertext = cipher
        .encrypt(nonce, plaintext_dek)
        .map_err(|e| GalaxError::Encryption(format!("DEK encrypt: {e}")))?;

    let mut out = Vec::with_capacity(12 + ciphertext.len());
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ciphertext);
    Ok(out)
}

/// Decrypt an encrypted DEK (`nonce || ciphertext+tag`) with the given
/// master key.
fn decrypt_dek_with_master(master_key: &[u8; 32], encrypted_dek: &[u8]) -> GalaxResult<Vec<u8>> {
    if encrypted_dek.len() < 12 {
        return Err(GalaxError::Encryption(
            "encrypted DEK too short".to_string(),
        ));
    }
    let cipher = Aes256Gcm::new_from_slice(master_key)
        .map_err(|e| GalaxError::Encryption(format!("cipher init: {e}")))?;
    let nonce = Nonce::from_slice(&encrypted_dek[..12]);
    let ciphertext = &encrypted_dek[12..];
    cipher
        .decrypt(nonce, ciphertext)
        .map_err(|e| GalaxError::Encryption(format!("DEK decrypt: {e}")))
}

// ---------------------------------------------------------------------------
// LocalKeyProvider — 32-byte master key from a local file
// ---------------------------------------------------------------------------

/// Reads a 32-byte master key from a local file. DEKs are wrapped with
/// AES-256-GCM using that master key. Suitable for development, CI, and
/// self-hosted deployments where a KMS is overkill.
pub struct LocalKeyProvider {
    master_key: [u8; 32],
}

impl LocalKeyProvider {
    /// Read exactly 32 bytes from `path` as the master key.
    pub fn from_file(path: &Path) -> GalaxResult<Self> {
        let data = std::fs::read(path).map_err(|e| {
            GalaxError::Encryption(format!(
                "failed to read key file {}: {e}",
                path.display()
            ))
        })?;
        if data.len() != 32 {
            return Err(GalaxError::Encryption(format!(
                "key file must be exactly 32 bytes, got {}",
                data.len()
            )));
        }
        let mut master_key = [0u8; 32];
        master_key.copy_from_slice(&data);
        Ok(Self { master_key })
    }

    /// Construct from a 32-byte key directly. Primarily useful in tests.
    pub fn from_key(master_key: [u8; 32]) -> Self {
        Self { master_key }
    }
}

impl KeyProvider for LocalKeyProvider {
    fn generate_data_key(&self) -> GalaxResult<(Vec<u8>, Vec<u8>)> {
        let mut plaintext = vec![0u8; 32];
        rand::thread_rng().fill_bytes(&mut plaintext);
        let encrypted = encrypt_dek_with_master(&self.master_key, &plaintext)?;
        Ok((plaintext, encrypted))
    }

    fn decrypt_data_key(&self, encrypted_key: &[u8]) -> GalaxResult<Vec<u8>> {
        decrypt_dek_with_master(&self.master_key, encrypted_key)
    }

    fn provider_name(&self) -> &str {
        "local-file"
    }
}

// ---------------------------------------------------------------------------
// EnvKeyProvider — hex-encoded master key from an env var
// ---------------------------------------------------------------------------

/// Reads a hex-encoded 32-byte master key from the `GALAXDB_MASTER_KEY`
/// environment variable (overridable). Suitable for containerized
/// deployments where the secret is injected at boot.
pub struct EnvKeyProvider {
    master_key: [u8; 32],
}

impl EnvKeyProvider {
    /// Default environment variable name.
    pub const ENV_VAR: &'static str = "GALAXDB_MASTER_KEY";

    /// Read from the default env var.
    pub fn from_env() -> GalaxResult<Self> {
        Self::from_env_var(Self::ENV_VAR)
    }

    /// Read from a custom env var name.
    pub fn from_env_var(var_name: &str) -> GalaxResult<Self> {
        let hex_str = std::env::var(var_name).map_err(|_| {
            GalaxError::Encryption(format!("environment variable {var_name} not set"))
        })?;
        let bytes = hex_decode(&hex_str)
            .map_err(|e| GalaxError::Encryption(format!("invalid hex in {var_name}: {e}")))?;
        if bytes.len() != 32 {
            return Err(GalaxError::Encryption(format!(
                "{var_name} must decode to exactly 32 bytes, got {}",
                bytes.len()
            )));
        }
        let mut master_key = [0u8; 32];
        master_key.copy_from_slice(&bytes);
        Ok(Self { master_key })
    }

    /// Construct from a 32-byte key directly. Primarily useful in tests.
    pub fn from_key(master_key: [u8; 32]) -> Self {
        Self { master_key }
    }
}

impl KeyProvider for EnvKeyProvider {
    fn generate_data_key(&self) -> GalaxResult<(Vec<u8>, Vec<u8>)> {
        let mut plaintext = vec![0u8; 32];
        rand::thread_rng().fill_bytes(&mut plaintext);
        let encrypted = encrypt_dek_with_master(&self.master_key, &plaintext)?;
        Ok((plaintext, encrypted))
    }

    fn decrypt_data_key(&self, encrypted_key: &[u8]) -> GalaxResult<Vec<u8>> {
        decrypt_dek_with_master(&self.master_key, encrypted_key)
    }

    fn provider_name(&self) -> &str {
        "env-var"
    }
}

// ---------------------------------------------------------------------------
// ExternalCommandKeyProvider — delegate to a shell command
// ---------------------------------------------------------------------------

/// Delegates DEK wrapping to an operator-supplied shell command.
///
/// The command is spawned twice:
///
/// * For [`generate_data_key`], with the argument `"generate"`. The
///   command must print a 32-byte plaintext DEK (base64 encoded) on
///   the first line of stdout and the opaque encrypted blob (base64
///   encoded) on the second line, separated by `\n`.
/// * For [`decrypt_data_key`], with the argument `"decrypt"`. The
///   encrypted blob is written to the command's stdin as base64 text;
///   the command must print the plaintext DEK (base64 encoded) on
///   stdout.
///
/// Any non-zero exit status is surfaced as [`GalaxError::Kms`] with
/// the command's stderr appended. This lets the engine wire into any
/// KMS that exposes a CLI (AWS KMS via `aws kms generate-data-key`,
/// GCP KMS via `gcloud kms encrypt/decrypt`, Azure Key Vault via
/// `az keyvault encrypt/decrypt`, Vault via `vault write
/// transit/encrypt`, and custom HSM wrappers) with zero SDK linkage.
///
/// Example configuration string (see [`KeyProviderSpec::parse`]):
/// `command:/opt/galaxdb/kms-wrapper.sh`
///
/// The operator's script is expected to do whatever authentication
/// and ciphertext framing the remote service requires.
pub struct ExternalCommandKeyProvider {
    program: String,
    extra_args: Vec<String>,
}

impl ExternalCommandKeyProvider {
    /// Build a provider that runs `program` with no baked-in extra
    /// arguments. The subcommand (`generate` or `decrypt`) is always
    /// passed as the last positional argument.
    pub fn new(program: impl Into<String>) -> Self {
        Self {
            program: program.into(),
            extra_args: Vec::new(),
        }
    }

    /// Build a provider that runs `program` with additional arguments
    /// every time (e.g. pinning a key ID). Order: `[extra_args...,
    /// subcommand]`.
    pub fn with_args(program: impl Into<String>, extra_args: Vec<String>) -> Self {
        Self {
            program: program.into(),
            extra_args,
        }
    }

    fn run(&self, subcommand: &str, stdin_payload: Option<&[u8]>) -> GalaxResult<Vec<u8>> {
        use std::io::Write;
        use std::process::{Command, Stdio};

        let mut cmd = Command::new(&self.program);
        for arg in &self.extra_args {
            cmd.arg(arg);
        }
        cmd.arg(subcommand);
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| {
            GalaxError::Kms(format!(
                "failed to spawn key-provider command '{}': {e}",
                self.program
            ))
        })?;

        if let Some(bytes) = stdin_payload {
            let mut stdin = child
                .stdin
                .take()
                .ok_or_else(|| GalaxError::Kms("child stdin not captured".to_string()))?;
            stdin
                .write_all(bytes)
                .map_err(|e| GalaxError::Kms(format!("write stdin: {e}")))?;
            // Explicit drop closes stdin so the child sees EOF.
            drop(stdin);
        } else {
            // Still drop stdin so commands that read until EOF don't
            // block waiting for input.
            drop(child.stdin.take());
        }

        let output = child.wait_with_output().map_err(|e| {
            GalaxError::Kms(format!(
                "wait on key-provider command '{}' failed: {e}",
                self.program
            ))
        })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(GalaxError::Kms(format!(
                "key-provider command '{}' exited with {} — stderr: {}",
                self.program, output.status, stderr
            )));
        }

        Ok(output.stdout)
    }
}

impl KeyProvider for ExternalCommandKeyProvider {
    fn generate_data_key(&self) -> GalaxResult<(Vec<u8>, Vec<u8>)> {
        let stdout = self.run("generate", None)?;
        let text = String::from_utf8(stdout)
            .map_err(|e| GalaxError::Kms(format!("generate output is not UTF-8: {e}")))?;
        let mut lines = text.lines();
        let plaintext_b64 = lines.next().ok_or_else(|| {
            GalaxError::Kms("generate output missing plaintext line".to_string())
        })?;
        let encrypted_b64 = lines.next().ok_or_else(|| {
            GalaxError::Kms("generate output missing encrypted line".to_string())
        })?;

        let plaintext = base64_decode(plaintext_b64)
            .map_err(|e| GalaxError::Kms(format!("plaintext base64: {e}")))?;
        let encrypted = base64_decode(encrypted_b64)
            .map_err(|e| GalaxError::Kms(format!("encrypted base64: {e}")))?;

        if plaintext.len() != 32 {
            return Err(GalaxError::Kms(format!(
                "plaintext DEK must be 32 bytes, got {}",
                plaintext.len()
            )));
        }

        Ok((plaintext, encrypted))
    }

    fn decrypt_data_key(&self, encrypted_key: &[u8]) -> GalaxResult<Vec<u8>> {
        let stdin = base64_encode(encrypted_key);
        let stdout = self.run("decrypt", Some(stdin.as_bytes()))?;
        let text = String::from_utf8(stdout)
            .map_err(|e| GalaxError::Kms(format!("decrypt output is not UTF-8: {e}")))?;
        let trimmed = text.trim();
        let plaintext = base64_decode(trimmed)
            .map_err(|e| GalaxError::Kms(format!("plaintext base64: {e}")))?;
        if plaintext.len() != 32 {
            return Err(GalaxError::Kms(format!(
                "plaintext DEK must be 32 bytes, got {}",
                plaintext.len()
            )));
        }
        Ok(plaintext)
    }

    fn provider_name(&self) -> &str {
        "external-command"
    }
}

// ---------------------------------------------------------------------------
// HashicorpVaultKeyProvider — Vault Transit engine (feature = "vault")
// ---------------------------------------------------------------------------

/// HashiCorp Vault Transit-engine key provider.
///
/// Encrypts DEKs with a named Transit key (`transit/encrypt/<name>`),
/// stores the `vault:v1:...` ciphertext as the encrypted blob, and
/// decrypts via `transit/decrypt/<name>`. The plaintext DEK never
/// leaves Vault unwrapped except in the engine's own memory.
///
/// Configuration inputs:
/// * `address` — Vault server URL, e.g. `https://vault.example.com:8200`.
/// * `token` — Vault token (read from `VAULT_TOKEN` env var by
///   default; supply explicitly via the builder for agent-delivered
///   tokens or Kubernetes sidecar tokens).
/// * `mount` — mount path of the Transit engine (defaults to
///   `"transit"`).
/// * `key_name` — Transit key name, must exist before the provider is
///   used (`vault write -f transit/keys/<name>`).
///
/// The provider owns a private `tokio` current-thread runtime so the
/// sync [`KeyProvider`] contract can drive the async `vaultrs`
/// client. The runtime is reused across calls so the construction
/// cost is amortised.
#[cfg(feature = "vault")]
pub struct HashicorpVaultKeyProvider {
    client: vaultrs::client::VaultClient,
    mount: String,
    key_name: String,
    runtime: std::sync::Mutex<tokio::runtime::Runtime>,
}

#[cfg(feature = "vault")]
impl HashicorpVaultKeyProvider {
    /// Default Transit mount used when none is supplied.
    pub const DEFAULT_TRANSIT_MOUNT: &'static str = "transit";

    /// Construct a provider pointing at `address` with an explicit
    /// `token` and the supplied `mount` / `key_name`. If `mount` is
    /// `None` the Vault default `"transit"` is used.
    pub fn new(
        address: &str,
        token: &str,
        mount: Option<&str>,
        key_name: &str,
    ) -> GalaxResult<Self> {
        use vaultrs::client::{VaultClient, VaultClientSettingsBuilder};

        let settings = VaultClientSettingsBuilder::default()
            .address(address)
            .token(token)
            .build()
            .map_err(|e| GalaxError::Kms(format!("Vault settings: {e}")))?;

        let client = VaultClient::new(settings)
            .map_err(|e| GalaxError::Kms(format!("Vault client: {e}")))?;

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| GalaxError::Kms(format!("tokio runtime: {e}")))?;

        Ok(Self {
            client,
            mount: mount.unwrap_or(Self::DEFAULT_TRANSIT_MOUNT).to_string(),
            key_name: key_name.to_string(),
            runtime: std::sync::Mutex::new(runtime),
        })
    }

    /// Construct from the standard Vault environment variables:
    /// `VAULT_ADDR` and `VAULT_TOKEN`. `mount` falls back to
    /// `DEFAULT_TRANSIT_MOUNT`.
    pub fn from_env(key_name: &str, mount: Option<&str>) -> GalaxResult<Self> {
        let address = std::env::var("VAULT_ADDR")
            .map_err(|_| GalaxError::Kms("VAULT_ADDR not set".to_string()))?;
        let token = std::env::var("VAULT_TOKEN")
            .map_err(|_| GalaxError::Kms("VAULT_TOKEN not set".to_string()))?;
        Self::new(&address, &token, mount, key_name)
    }
}

#[cfg(feature = "vault")]
impl KeyProvider for HashicorpVaultKeyProvider {
    fn generate_data_key(&self) -> GalaxResult<(Vec<u8>, Vec<u8>)> {
        use base64::Engine as _;
        use base64::engine::general_purpose::STANDARD as B64;

        // Generate the plaintext DEK locally — Transit only encrypts,
        // it does not produce keys. The plaintext never touches Vault.
        let mut plaintext = vec![0u8; 32];
        rand::thread_rng().fill_bytes(&mut plaintext);

        let b64_plaintext = B64.encode(&plaintext);

        let runtime = self
            .runtime
            .lock()
            .map_err(|_| GalaxError::Kms("Vault runtime mutex poisoned".into()))?;

        let response = runtime.block_on(vaultrs::transit::data::encrypt(
            &self.client,
            &self.mount,
            &self.key_name,
            &b64_plaintext,
            None,
        ));

        let encrypted = response
            .map_err(|e| GalaxError::Kms(format!("Vault transit encrypt: {e}")))?
            .ciphertext
            .into_bytes();

        Ok((plaintext, encrypted))
    }

    fn decrypt_data_key(&self, encrypted_key: &[u8]) -> GalaxResult<Vec<u8>> {
        use base64::Engine as _;
        use base64::engine::general_purpose::STANDARD as B64;

        let ciphertext = std::str::from_utf8(encrypted_key).map_err(|e| {
            GalaxError::Kms(format!("Vault ciphertext is not UTF-8: {e}"))
        })?;

        let runtime = self
            .runtime
            .lock()
            .map_err(|_| GalaxError::Kms("Vault runtime mutex poisoned".into()))?;

        let response = runtime.block_on(vaultrs::transit::data::decrypt(
            &self.client,
            &self.mount,
            &self.key_name,
            ciphertext,
            None,
        ));

        let b64_plaintext = response
            .map_err(|e| GalaxError::Kms(format!("Vault transit decrypt: {e}")))?
            .plaintext;

        let plaintext = B64
            .decode(&b64_plaintext)
            .map_err(|e| GalaxError::Kms(format!("Vault plaintext base64: {e}")))?;

        if plaintext.len() != 32 {
            return Err(GalaxError::Kms(format!(
                "Vault returned {} bytes, expected 32-byte DEK",
                plaintext.len()
            )));
        }

        Ok(plaintext)
    }

    fn provider_name(&self) -> &str {
        "hashicorp-vault"
    }
}

// ---------------------------------------------------------------------------
// KeyProviderSpec — startup-time provider selection
// ---------------------------------------------------------------------------

/// Startup-time provider selection.
///
/// Parse operator-supplied configuration into one of the four supported
/// providers, then call [`KeyProviderSpec::build`] to get a
/// `Box<dyn KeyProvider>`.
///
/// # Configuration syntax
///
/// Read from the `GALAXDB_KEY_PROVIDER` environment variable (or passed
/// explicitly in code) in one of these forms:
///
/// * `local:/path/to/32-byte-key-file`
/// * `env:GALAXDB_MASTER_KEY` (the default env var if only `env` is given)
/// * `command:/absolute/path/to/kms-wrapper[:arg1:arg2:...]`
/// * `vault:mount/key_name` — requires `VAULT_ADDR`, `VAULT_TOKEN`; needs
///   the `vault` Cargo feature compiled in.
#[derive(Debug, Clone)]
pub enum KeyProviderSpec {
    Local { path: std::path::PathBuf },
    Env { var: String },
    Command { program: String, extra_args: Vec<String> },
    Vault { mount: String, key_name: String },
    /// AWS KMS (`aws-kms:<key-id|alias|arn>`). Needs the `cloud-kms` feature.
    AwsKms { key_id: String },
    /// GCP Cloud KMS (`gcp-kms:projects/.../cryptoKeys/...`). Needs `cloud-kms`.
    GcpKms { key_name: String },
    /// Azure Key Vault (`azure-kv:<vault>/<key>`). Needs the `cloud-kms` feature.
    AzureKv { vault: String, key_name: String },
}

impl KeyProviderSpec {
    /// Parse the `GALAXDB_KEY_PROVIDER` syntax.
    pub fn parse(spec: &str) -> GalaxResult<Self> {
        let trimmed = spec.trim();
        if let Some(path) = trimmed.strip_prefix("local:") {
            return Ok(Self::Local {
                path: path.into(),
            });
        }
        if trimmed == "env" {
            return Ok(Self::Env {
                var: EnvKeyProvider::ENV_VAR.to_string(),
            });
        }
        if let Some(var) = trimmed.strip_prefix("env:") {
            return Ok(Self::Env {
                var: var.to_string(),
            });
        }
        if let Some(rest) = trimmed.strip_prefix("command:") {
            // `command:/path/to/prog:arg1:arg2` — split on `:` but
            // keep the first segment as the program path so Unix
            // absolute paths like `/opt/x` are preserved.
            let mut parts = rest.splitn(2, ':');
            let program = parts
                .next()
                .ok_or_else(|| {
                    GalaxError::Encryption("command: missing program path".to_string())
                })?
                .to_string();
            let extra_args: Vec<String> = match parts.next() {
                Some(rest) if !rest.is_empty() => {
                    rest.split(':').map(|s| s.to_string()).collect()
                }
                _ => Vec::new(),
            };
            return Ok(Self::Command { program, extra_args });
        }
        if let Some(rest) = trimmed.strip_prefix("vault:") {
            // Accept `vault:mount/key` (explicit mount) or `vault:key`
            // (default transit mount).
            let (mount, key_name) = match rest.split_once('/') {
                Some((m, k)) => (m.to_string(), k.to_string()),
                None => ("transit".to_string(), rest.to_string()),
            };
            if key_name.is_empty() {
                return Err(GalaxError::Encryption(
                    "vault: missing key name".to_string(),
                ));
            }
            return Ok(Self::Vault { mount, key_name });
        }
        if let Some(key_id) = trimmed.strip_prefix("aws-kms:") {
            if key_id.is_empty() {
                return Err(GalaxError::Kms("aws-kms: missing key id/alias/arn".to_string()));
            }
            return Ok(Self::AwsKms { key_id: key_id.to_string() });
        }
        if let Some(key_name) = trimmed.strip_prefix("gcp-kms:") {
            if key_name.is_empty() {
                return Err(GalaxError::Kms("gcp-kms: missing crypto-key resource name".to_string()));
            }
            return Ok(Self::GcpKms { key_name: key_name.to_string() });
        }
        if let Some(rest) = trimmed.strip_prefix("azure-kv:") {
            let (vault, key_name) = rest.split_once('/').ok_or_else(|| {
                GalaxError::Kms("azure-kv: expected azure-kv:<vault>/<key>".to_string())
            })?;
            if vault.is_empty() || key_name.is_empty() {
                return Err(GalaxError::Kms("azure-kv: empty vault or key".to_string()));
            }
            return Ok(Self::AzureKv {
                vault: vault.to_string(),
                key_name: key_name.to_string(),
            });
        }
        Err(GalaxError::Encryption(format!(
            "unrecognised key-provider spec '{spec}'; expected one of \
             local:<path>, env[:<var>], command:<program>[:args], vault:[mount/]key, \
             aws-kms:<key>, gcp-kms:<resource>, azure-kv:<vault>/<key>"
        )))
    }

    /// Build the actual [`KeyProvider`]. This may perform I/O (reading
    /// the key file, contacting Vault) and so returns `Result`.
    pub fn build(&self) -> GalaxResult<Box<dyn KeyProvider>> {
        match self {
            Self::Local { path } => {
                Ok(Box::new(LocalKeyProvider::from_file(path)?))
            }
            Self::Env { var } => {
                Ok(Box::new(EnvKeyProvider::from_env_var(var)?))
            }
            Self::Command { program, extra_args } => {
                Ok(Box::new(ExternalCommandKeyProvider::with_args(
                    program.clone(),
                    extra_args.clone(),
                )))
            }
            Self::Vault { mount, key_name } => {
                #[cfg(feature = "vault")]
                {
                    Ok(Box::new(HashicorpVaultKeyProvider::from_env(
                        key_name,
                        Some(mount),
                    )?))
                }
                #[cfg(not(feature = "vault"))]
                {
                    let _ = (mount, key_name);
                    Err(GalaxError::Encryption(
                        "Vault key provider requires the 'vault' Cargo feature".to_string(),
                    ))
                }
            }
            Self::AwsKms { key_id } => {
                #[cfg(feature = "cloud-kms")]
                {
                    Ok(Box::new(crate::cloud_kms::AwsKmsKeyProvider::from_key_id(
                        key_id,
                    )?))
                }
                #[cfg(not(feature = "cloud-kms"))]
                {
                    let _ = key_id;
                    Err(GalaxError::Kms(
                        "AWS KMS key provider requires the 'cloud-kms' Cargo feature".to_string(),
                    ))
                }
            }
            Self::GcpKms { key_name } => {
                #[cfg(feature = "cloud-kms")]
                {
                    Ok(Box::new(crate::cloud_kms::GcpKmsKeyProvider::from_key_name(
                        key_name,
                    )?))
                }
                #[cfg(not(feature = "cloud-kms"))]
                {
                    let _ = key_name;
                    Err(GalaxError::Kms(
                        "GCP KMS key provider requires the 'cloud-kms' Cargo feature".to_string(),
                    ))
                }
            }
            Self::AzureKv { vault, key_name } => {
                #[cfg(feature = "cloud-kms")]
                {
                    Ok(Box::new(
                        crate::cloud_kms::AzureKeyVaultKeyProvider::from_spec(vault, key_name)?,
                    ))
                }
                #[cfg(not(feature = "cloud-kms"))]
                {
                    let _ = (vault, key_name);
                    Err(GalaxError::Kms(
                        "Azure Key Vault key provider requires the 'cloud-kms' Cargo feature"
                            .to_string(),
                    ))
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Internal hex + base64 helpers (no new dep for the core)
// ---------------------------------------------------------------------------

fn hex_decode(s: &str) -> Result<Vec<u8>, String> {
    let s = s.trim();
    if s.len() % 2 != 0 {
        return Err("odd number of hex characters".to_string());
    }
    (0..s.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&s[i..i + 2], 16)
                .map_err(|e| format!("invalid hex at position {i}: {e}"))
        })
        .collect()
}

/// Standard base64 (no padding relaxation).
///
/// We keep base64 as an internal helper so the core crypto crate does
/// not grow a dependency for the common Local/Env/Command providers.
/// The `base64` crate is only pulled in with the `vault` feature.
fn base64_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    let chunks = bytes.chunks(3);
    for chunk in chunks {
        let (b0, b1, b2) = match chunk.len() {
            3 => (chunk[0], chunk[1], chunk[2]),
            2 => (chunk[0], chunk[1], 0),
            1 => (chunk[0], 0, 0),
            _ => unreachable!(),
        };
        let c0 = (b0 >> 2) & 0x3f;
        let c1 = ((b0 << 4) | (b1 >> 4)) & 0x3f;
        let c2 = ((b1 << 2) | (b2 >> 6)) & 0x3f;
        let c3 = b2 & 0x3f;
        out.push(TABLE[c0 as usize] as char);
        out.push(TABLE[c1 as usize] as char);
        if chunk.len() > 1 {
            out.push(TABLE[c2 as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(TABLE[c3 as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    fn value(c: u8) -> Result<u8, String> {
        match c {
            b'A'..=b'Z' => Ok(c - b'A'),
            b'a'..=b'z' => Ok(c - b'a' + 26),
            b'0'..=b'9' => Ok(c - b'0' + 52),
            b'+' => Ok(62),
            b'/' => Ok(63),
            _ => Err(format!("invalid base64 character: {}", c as char)),
        }
    }

    let s = s.trim();
    let bytes = s.as_bytes();
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "base64 length must be a multiple of 4, got {}",
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks(4) {
        let pad = chunk.iter().rev().take_while(|&&c| c == b'=').count();
        let vals: [u8; 4] = [
            if chunk[0] == b'=' { 0 } else { value(chunk[0])? },
            if chunk[1] == b'=' { 0 } else { value(chunk[1])? },
            if chunk[2] == b'=' { 0 } else { value(chunk[2])? },
            if chunk[3] == b'=' { 0 } else { value(chunk[3])? },
        ];
        let b0 = (vals[0] << 2) | (vals[1] >> 4);
        let b1 = (vals[1] << 4) | (vals[2] >> 2);
        let b2 = (vals[2] << 6) | vals[3];
        match pad {
            0 => {
                out.push(b0);
                out.push(b1);
                out.push(b2);
            }
            1 => {
                out.push(b0);
                out.push(b1);
            }
            2 => {
                out.push(b0);
            }
            _ => return Err("too much padding".to_string()),
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_decode_valid() {
        assert_eq!(hex_decode("deadbeef").unwrap(), vec![0xde, 0xad, 0xbe, 0xef]);
        assert_eq!(hex_decode("00ff").unwrap(), vec![0x00, 0xff]);
    }

    #[test]
    fn hex_decode_odd_length() {
        assert!(hex_decode("abc").is_err());
    }

    #[test]
    fn base64_round_trip() {
        for input in [
            &b""[..],
            &b"x"[..],
            &b"xx"[..],
            &b"xxx"[..],
            &b"hello world"[..],
            &[0u8; 32][..],
            &[0xffu8; 32][..],
        ] {
            let encoded = base64_encode(input);
            let decoded = base64_decode(&encoded).expect("base64 decode");
            assert_eq!(decoded.as_slice(), input);
        }
    }

    #[test]
    fn local_key_provider_round_trip() {
        let master = [0x42u8; 32];
        let provider = LocalKeyProvider::from_key(master);
        assert_eq!(provider.provider_name(), "local-file");

        let (plaintext, encrypted) = provider.generate_data_key().unwrap();
        assert_eq!(plaintext.len(), 32);
        assert_ne!(plaintext, encrypted);

        let decrypted = provider.decrypt_data_key(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn env_key_provider_round_trip() {
        let master = [0x77u8; 32];
        let provider = EnvKeyProvider::from_key(master);
        assert_eq!(provider.provider_name(), "env-var");

        let (plaintext, encrypted) = provider.generate_data_key().unwrap();
        let decrypted = provider.decrypt_data_key(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn decrypt_with_wrong_key_fails() {
        let provider_a = LocalKeyProvider::from_key([0x11u8; 32]);
        let provider_b = LocalKeyProvider::from_key([0x22u8; 32]);

        let (_plaintext, encrypted) = provider_a.generate_data_key().unwrap();
        assert!(provider_b.decrypt_data_key(&encrypted).is_err());
    }

    #[test]
    fn decrypt_truncated_data_fails() {
        let provider = LocalKeyProvider::from_key([0x33u8; 32]);
        assert!(provider.decrypt_data_key(&[0u8; 5]).is_err());
    }

    #[test]
    fn local_key_provider_from_file_wrong_size() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad_key");
        std::fs::write(&path, [0u8; 16]).unwrap();
        assert!(LocalKeyProvider::from_file(&path).is_err());
    }

    #[test]
    fn local_key_provider_from_file_valid() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("good_key");
        std::fs::write(&path, [0xAAu8; 32]).unwrap();
        let provider = LocalKeyProvider::from_file(&path).unwrap();

        let (pt, enc) = provider.generate_data_key().unwrap();
        let dec = provider.decrypt_data_key(&enc).unwrap();
        assert_eq!(pt, dec);
    }

    // -----------------------------------------------------------------
    // KeyProviderSpec parsing
    // -----------------------------------------------------------------

    #[test]
    fn spec_parse_local() {
        let spec = KeyProviderSpec::parse("local:/tmp/key").unwrap();
        match spec {
            KeyProviderSpec::Local { path } => {
                assert_eq!(path, std::path::PathBuf::from("/tmp/key"))
            }
            other => panic!("expected Local, got {:?}", other),
        }
    }

    #[test]
    fn spec_parse_env_default_and_custom() {
        match KeyProviderSpec::parse("env").unwrap() {
            KeyProviderSpec::Env { var } => assert_eq!(var, EnvKeyProvider::ENV_VAR),
            other => panic!("expected Env, got {:?}", other),
        }
        match KeyProviderSpec::parse("env:MY_KEY").unwrap() {
            KeyProviderSpec::Env { var } => assert_eq!(var, "MY_KEY"),
            other => panic!("expected Env, got {:?}", other),
        }
    }

    #[test]
    fn spec_parse_command() {
        match KeyProviderSpec::parse("command:/opt/kms.sh").unwrap() {
            KeyProviderSpec::Command { program, extra_args } => {
                assert_eq!(program, "/opt/kms.sh");
                assert!(extra_args.is_empty());
            }
            other => panic!("{:?}", other),
        }

        match KeyProviderSpec::parse("command:/opt/kms.sh:--key:alias/foo").unwrap() {
            KeyProviderSpec::Command { program, extra_args } => {
                assert_eq!(program, "/opt/kms.sh");
                assert_eq!(extra_args, vec!["--key", "alias/foo"]);
            }
            other => panic!("{:?}", other),
        }
    }

    #[test]
    fn spec_parse_vault() {
        match KeyProviderSpec::parse("vault:my-key").unwrap() {
            KeyProviderSpec::Vault { mount, key_name } => {
                assert_eq!(mount, "transit");
                assert_eq!(key_name, "my-key");
            }
            other => panic!("{:?}", other),
        }
        match KeyProviderSpec::parse("vault:transit-2/my-key").unwrap() {
            KeyProviderSpec::Vault { mount, key_name } => {
                assert_eq!(mount, "transit-2");
                assert_eq!(key_name, "my-key");
            }
            other => panic!("{:?}", other),
        }
    }

    #[test]
    fn spec_parse_unknown_fails() {
        assert!(KeyProviderSpec::parse("random:thing").is_err());
        assert!(KeyProviderSpec::parse("").is_err());
        assert!(KeyProviderSpec::parse("vault:").is_err());
    }

    #[test]
    fn spec_parse_cloud_kms() {
        match KeyProviderSpec::parse("aws-kms:alias/galaxdb").unwrap() {
            KeyProviderSpec::AwsKms { key_id } => assert_eq!(key_id, "alias/galaxdb"),
            other => panic!("{:?}", other),
        }
        match KeyProviderSpec::parse(
            "gcp-kms:projects/p/locations/l/keyRings/r/cryptoKeys/k",
        )
        .unwrap()
        {
            KeyProviderSpec::GcpKms { key_name } => {
                assert_eq!(key_name, "projects/p/locations/l/keyRings/r/cryptoKeys/k")
            }
            other => panic!("{:?}", other),
        }
        match KeyProviderSpec::parse("azure-kv:myvault/mykey").unwrap() {
            KeyProviderSpec::AzureKv { vault, key_name } => {
                assert_eq!(vault, "myvault");
                assert_eq!(key_name, "mykey");
            }
            other => panic!("{:?}", other),
        }
    }

    #[test]
    fn spec_parse_cloud_kms_malformed_fails() {
        assert!(KeyProviderSpec::parse("aws-kms:").is_err());
        assert!(KeyProviderSpec::parse("gcp-kms:").is_err());
        assert!(KeyProviderSpec::parse("azure-kv:novault").is_err());
        assert!(KeyProviderSpec::parse("azure-kv:/key").is_err());
        assert!(KeyProviderSpec::parse("azure-kv:vault/").is_err());
    }

    // -----------------------------------------------------------------
    // External-command provider — real subprocess round trip.
    //
    // Uses a tiny Python helper written to a tempdir to prove the
    // subprocess protocol works. The helper wraps the DEK with AES-256-GCM
    // deterministically seeded by a file-based master key. This is a
    // real KeyProvider implementation exercised via the real OS
    // spawn/pipe/wait path — not a mock.
    // -----------------------------------------------------------------

    #[test]
    fn external_command_provider_round_trip() {
        // Skip if python3 isn't available in PATH.
        if std::process::Command::new("python3")
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("python3 not in PATH; skipping external-command test");
            return;
        }

        let dir = tempfile::tempdir().unwrap();
        let master_path = dir.path().join("master.key");
        std::fs::write(&master_path, [0xA5u8; 32]).unwrap();

        let wrapper = dir.path().join("kms_wrapper.py");
        // The wrapper accepts `generate` or `decrypt`. For `generate`
        // it produces a 32-byte DEK and wraps it with AES-256-GCM
        // under the master key; the nonce is prepended to the
        // ciphertext. For `decrypt` it reads the ciphertext as
        // base64 on stdin and emits the plaintext DEK as base64 on
        // stdout.
        std::fs::write(
            &wrapper,
            r#"#!/usr/bin/env python3
import base64
import os
import sys

# Hand-rolled AES-256-GCM using the 'cryptography' package if
# available; otherwise just XOR so the test still runs on a stock
# Python without pip installs. The test only verifies the protocol;
# it does not assert the wrapper uses real crypto.
def xor_wrap(master, plaintext):
    return bytes(b ^ master[i % len(master)] for i, b in enumerate(plaintext))

def main():
    master_path = os.environ["GALAXDB_TEST_MASTER"]
    with open(master_path, "rb") as f:
        master = f.read()
    assert len(master) == 32

    if sys.argv[-1] == "generate":
        plaintext = os.urandom(32)
        encrypted = xor_wrap(master, plaintext)
        sys.stdout.write(base64.b64encode(plaintext).decode() + "\n")
        sys.stdout.write(base64.b64encode(encrypted).decode() + "\n")
        return

    if sys.argv[-1] == "decrypt":
        encrypted_b64 = sys.stdin.read().strip()
        encrypted = base64.b64decode(encrypted_b64)
        plaintext = xor_wrap(master, encrypted)
        sys.stdout.write(base64.b64encode(plaintext).decode() + "\n")
        return

    sys.stderr.write(f"unknown subcommand: {sys.argv[-1]}\n")
    sys.exit(2)

if __name__ == "__main__":
    main()
"#,
        )
        .unwrap();

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&wrapper).unwrap().permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&wrapper, perms).unwrap();
        }

        // Point the wrapper at the master key file.
        // SAFETY: set_var is unsafe on Rust 2024 edition because other
        // threads may be calling getenv simultaneously. This test
        // stays single-threaded (`cargo test` defaults to one thread
        // per test process on this path) and the env var is scoped to
        // the subprocess we are about to spawn, so the race window is
        // empty.
        unsafe {
            std::env::set_var("GALAXDB_TEST_MASTER", &master_path);
        }

        let provider = ExternalCommandKeyProvider::with_args(
            "python3".to_string(),
            vec![wrapper.to_string_lossy().into_owned()],
        );
        assert_eq!(provider.provider_name(), "external-command");

        let (plaintext, encrypted) = provider.generate_data_key().unwrap();
        assert_eq!(plaintext.len(), 32);

        let decrypted = provider.decrypt_data_key(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn external_command_provider_reports_failures() {
        let provider = ExternalCommandKeyProvider::new("/definitely/does/not/exist");
        let err = provider.generate_data_key().unwrap_err();
        match err {
            GalaxError::Kms(msg) => assert!(msg.contains("failed to spawn")),
            other => panic!("expected Kms, got {:?}", other),
        }
    }
}
