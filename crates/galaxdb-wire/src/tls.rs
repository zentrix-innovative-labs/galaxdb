//! TLS transport for the PostgreSQL wire protocol (Requirement 2).
//!
//! PostgreSQL negotiates TLS *before* the StartupMessage: the client
//! sends an `SSLRequest` packet (8 bytes: length `8`, request code
//! `80877103`), and the server replies with a single byte — `S` to
//! proceed with a TLS handshake on the same socket, or `N` to continue in
//! plaintext. This module owns:
//!
//! * loading the server certificate chain + private key from PEM
//!   ([`load_server_config`]),
//! * the [`TlsMode`] policy (`disable` / `allow` / `require`),
//! * reading the `SSLRequest` sentinel ([`peek_ssl_request`]).
//!
//! TLS is implemented with `rustls` (no OpenSSL / native-tls), so the
//! build stays cross-platform and free of system TLS libraries
//! (Requirement 2 AC7). rustls only ever offers TLS 1.2 and 1.3, which
//! satisfies the "no version below TLS 1.2" requirement (AC5).

use std::io;
use std::sync::Arc;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_rustls::rustls::pki_types::pem::PemObject;
use tokio_rustls::rustls::pki_types::{CertificateDer, PrivateKeyDer};
use tokio_rustls::rustls::ServerConfig;
use tokio_rustls::TlsAcceptor;

/// The PostgreSQL `SSLRequest` packet body code (the 4 bytes following the
/// length). `0x04D2162F` = `80877103`. Sent in place of a protocol
/// version in the pre-startup packet.
pub const SSL_REQUEST_CODE: i32 = 80_877_103;

/// The TLS negotiation policy for the wire server (Requirement 2 AC4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TlsMode {
    /// Never offer TLS; reply `N` to `SSLRequest` and serve plaintext.
    Disable,
    /// Offer TLS when the client asks, but also accept plaintext
    /// connections that skip `SSLRequest`. This is the default and
    /// matches PostgreSQL's permissive behavior.
    #[default]
    Allow,
    /// Require TLS: reply `S` to `SSLRequest`, and reject any client that
    /// sends a StartupMessage without first negotiating TLS (AC3).
    Require,
}

impl TlsMode {
    /// Parse a `tls_mode` config string. Unknown values are an error so a
    /// typo can't silently downgrade security.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "disable" | "disabled" | "off" => Ok(TlsMode::Disable),
            "allow" | "" => Ok(TlsMode::Allow),
            "require" | "required" | "on" => Ok(TlsMode::Require),
            other => Err(format!(
                "invalid tls_mode '{other}' (expected disable|allow|require)"
            )),
        }
    }

    /// A stable label for logging.
    pub fn label(self) -> &'static str {
        match self {
            TlsMode::Disable => "disable",
            TlsMode::Allow => "allow",
            TlsMode::Require => "require",
        }
    }
}

/// Load a rustls [`ServerConfig`] from a PEM certificate chain and
/// private key on disk (Requirement 2 AC2/AC4).
///
/// The cert file may contain a full chain (leaf first). The key file may
/// hold a PKCS#8, SEC1, or PKCS#1 private key — `rustls-pki-types`'
/// `PemObject` parser detects which. Returns a typed error (never a
/// fake/self-signed fallback) if a file is missing, malformed, or
/// contains no key, so a misconfigured TLS deployment fails loudly rather
/// than silently serving without the intended certificate.
pub fn load_server_config(
    cert_path: &str,
    key_path: &str,
) -> io::Result<Arc<ServerConfig>> {
    let cert_bytes = std::fs::read(cert_path).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("TLS: cannot read certificate '{cert_path}': {e}"),
        )
    })?;
    let key_bytes = std::fs::read(key_path).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("TLS: cannot read private key '{key_path}': {e}"),
        )
    })?;

    // Parse the (possibly multi-cert) chain via rustls-pki-types' PemObject.
    let certs: Vec<CertificateDer<'static>> = CertificateDer::pem_slice_iter(&cert_bytes)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("TLS: malformed certificate PEM in '{cert_path}': {e}"),
            )
        })?;
    if certs.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("TLS: no certificates found in '{cert_path}'"),
        ));
    }

    let key: PrivateKeyDer<'static> = PrivateKeyDer::from_pem_slice(&key_bytes).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("TLS: no usable private key in '{key_path}': {e}"),
        )
    })?;

    // rustls 0.23 needs a process-wide `CryptoProvider`. `ServerConfig::
    // builder()` only auto-installs one when exactly one crypto feature is
    // compiled in; the workspace's dependency graph enables both `aws_lc_rs`
    // and `ring` transitively, so auto-selection fails and the builder
    // *panics* ("Could not automatically determine the process-level
    // CryptoProvider"). The TLS integration tests sidestep this by calling
    // `install_default` in their setup — but the real `galaxdb-server`
    // binary did not, so starting it with `--tls-mode allow|require` would
    // crash on the first TLS config build. Install aws-lc-rs explicitly and
    // idempotently here (in the centralized TLS path) so the production
    // binary and every caller get a working provider. `install_default`
    // returns `Err` if a provider is already installed; that is benign and
    // intentionally ignored.
    let _ = tokio_rustls::rustls::crypto::aws_lc_rs::default_provider().install_default();

    // rustls' default builder offers only TLS 1.2 and 1.3 (AC5). No
    // client-cert auth in this phase (server auth + SCRAM inside the
    // channel).
    let config = ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("TLS: server config rejected the cert/key pair: {e}"),
            )
        })?;

    Ok(Arc::new(config))
}

/// Build a [`TlsAcceptor`] from a loaded server config.
pub fn acceptor(config: Arc<ServerConfig>) -> TlsAcceptor {
    TlsAcceptor::from(config)
}

/// Re-export so consumers (the server crate) can name the acceptor type
/// without taking a direct `tokio-rustls` dependency — TLS plumbing stays
/// centralized in this crate.
pub use tokio_rustls::TlsAcceptor as ReexportedTlsAcceptor;

/// The result of inspecting the first packet on a fresh connection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Prologue {
    /// The client sent an `SSLRequest`. The 8 bytes have been consumed;
    /// the caller replies `S`/`N` and (on `S`) performs the TLS
    /// handshake, then reads the StartupMessage from the (now encrypted)
    /// stream.
    SslRequest,
    /// The client sent a GSSAPI encryption request (code `80877104`).
    /// GalaxDB does not support GSSAPI encryption; the caller replies `N`
    /// and continues reading the next prologue packet.
    GssEncRequest,
    /// The client sent a plaintext StartupMessage (or CancelRequest). The
    /// 8 bytes already read — the 4-byte length and 4-byte protocol
    /// version/code — are returned so the caller can parse the startup
    /// without losing them.
    StartupHead {
        /// The message length field (includes itself).
        length: i32,
        /// The protocol version (or request code) field.
        code: i32,
    },
}

/// Read and classify the first 8 bytes of a connection: an `SSLRequest`,
/// a `GSSENCRequest`, or the head of a plaintext StartupMessage.
///
/// PostgreSQL frames both `SSLRequest` and the StartupMessage as
/// `Int32 length` then `Int32 code`. `SSLRequest` is exactly length `8`,
/// code `80877103`; `GSSENCRequest` is length `8`, code `80877104`.
/// Anything else is the start of a real StartupMessage and the 8 bytes
/// must be preserved for [`crate::messages::read_startup_after_head`].
pub async fn peek_ssl_request<R: AsyncReadExt + Unpin>(reader: &mut R) -> io::Result<Prologue> {
    let length = reader.read_i32().await?;
    let code = reader.read_i32().await?;
    if length == 8 && code == SSL_REQUEST_CODE {
        Ok(Prologue::SslRequest)
    } else if length == 8 && code == 80_877_104 {
        Ok(Prologue::GssEncRequest)
    } else {
        Ok(Prologue::StartupHead { length, code })
    }
}

/// Reply to an `SSLRequest`/`GSSENCRequest` with a single byte: `S` to
/// accept TLS, `N` to decline (the client then continues in plaintext or
/// disconnects).
pub async fn write_negotiation_reply<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
    accept: bool,
) -> io::Result<()> {
    writer.write_u8(if accept { b'S' } else { b'N' }).await?;
    writer.flush().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn tls_mode_parses_and_defaults() {
        assert_eq!(TlsMode::parse("disable").unwrap(), TlsMode::Disable);
        assert_eq!(TlsMode::parse("ALLOW").unwrap(), TlsMode::Allow);
        assert_eq!(TlsMode::parse("require").unwrap(), TlsMode::Require);
        assert_eq!(TlsMode::parse("").unwrap(), TlsMode::Allow);
        assert!(TlsMode::parse("bogus").is_err());
        assert_eq!(TlsMode::default(), TlsMode::Allow);
    }

    #[tokio::test]
    async fn peek_recognizes_ssl_request() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&8i32.to_be_bytes());
        buf.extend_from_slice(&SSL_REQUEST_CODE.to_be_bytes());
        let mut cur = Cursor::new(buf);
        assert_eq!(peek_ssl_request(&mut cur).await.unwrap(), Prologue::SslRequest);
    }

    #[tokio::test]
    async fn peek_recognizes_gss_request() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&8i32.to_be_bytes());
        buf.extend_from_slice(&80_877_104i32.to_be_bytes());
        let mut cur = Cursor::new(buf);
        assert_eq!(
            peek_ssl_request(&mut cur).await.unwrap(),
            Prologue::GssEncRequest
        );
    }

    #[tokio::test]
    async fn peek_preserves_startup_head() {
        // A real StartupMessage: length 0x00000054, protocol version 196608.
        let mut buf = Vec::new();
        buf.extend_from_slice(&84i32.to_be_bytes());
        buf.extend_from_slice(&196_608i32.to_be_bytes());
        let mut cur = Cursor::new(buf);
        match peek_ssl_request(&mut cur).await.unwrap() {
            Prologue::StartupHead { length, code } => {
                assert_eq!(length, 84);
                assert_eq!(code, 196_608);
            }
            other => panic!("expected StartupHead, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn negotiation_reply_writes_single_byte() {
        let mut s = Vec::new();
        write_negotiation_reply(&mut s, true).await.unwrap();
        assert_eq!(s, b"S");
        let mut n = Vec::new();
        write_negotiation_reply(&mut n, false).await.unwrap();
        assert_eq!(n, b"N");
    }

    /// Regression for the "Could not automatically determine the
    /// process-level CryptoProvider" panic: `load_server_config` must
    /// build a usable `ServerConfig` on its own, WITHOUT any caller having
    /// installed a rustls crypto provider first. The server-crate TLS
    /// integration tests call `install_default` in their setup, which
    /// masked the fact that the real `galaxdb-server` binary never did —
    /// so starting it with `--tls-mode allow|require` panicked on the
    /// first config build. No test in THIS crate installs a provider, so a
    /// green result here proves `load_server_config` self-installs one.
    #[test]
    fn load_server_config_installs_crypto_provider_and_builds() {
        let cert = rcgen::generate_simple_self_signed(vec!["127.0.0.1".to_string()])
            .expect("self-signed cert generation");
        let dir = tempfile::tempdir().unwrap();
        let cert_path = dir.path().join("c.pem");
        let key_path = dir.path().join("k.pem");
        std::fs::write(&cert_path, cert.cert.pem()).unwrap();
        std::fs::write(&key_path, cert.key_pair.serialize_pem()).unwrap();

        let cfg = load_server_config(
            cert_path.to_str().unwrap(),
            key_path.to_str().unwrap(),
        )
        .expect("load_server_config must build a ServerConfig without a pre-installed provider");
        // The acceptor must be constructible from the loaded config.
        let _acceptor = acceptor(cfg);
    }
}
