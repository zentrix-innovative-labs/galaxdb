//! SCRAM-SHA-256 server-side implementation (RFC 5802, RFC 7677).
//!
//! This is the credential mechanism PostgreSQL clients use by default.
//! The server stores a [`ScramVerifier`] (salt, iteration count, and two
//! derived keys) per role — never the plaintext password — and drives the
//! four-message exchange:
//!
//! ```text
//! client-first:  n,,n=<user>,r=<client-nonce>
//! server-first:  r=<client-nonce><server-nonce>,s=<salt-b64>,i=<iters>
//! client-final:  c=biws,r=<combined-nonce>,p=<client-proof-b64>
//! server-final:  v=<server-signature-b64>
//! ```
//!
//! The cryptographic core (RFC 5802 §3):
//!
//! ```text
//! SaltedPassword = PBKDF2(HMAC-SHA-256, password, salt, iters)
//! ClientKey      = HMAC(SaltedPassword, "Client Key")
//! StoredKey      = SHA-256(ClientKey)                 <- stored
//! ServerKey      = HMAC(SaltedPassword, "Server Key") <- stored
//! AuthMessage    = client-first-bare + "," + server-first + "," + client-final-without-proof
//! ClientSignature= HMAC(StoredKey, AuthMessage)
//! ClientProof    = ClientKey XOR ClientSignature      <- sent by client
//! ServerSignature= HMAC(ServerKey, AuthMessage)       <- sent by server
//! ```
//!
//! Verification recovers `ClientKey = ClientProof XOR ClientSignature`,
//! hashes it, and compares to `StoredKey` in constant time.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;
use hmac::{Hmac, Mac, KeyInit};
use rand::RngCore;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

type HmacSha256 = Hmac<Sha256>;

/// Default PBKDF2 iteration count (RFC 7677 recommends at least 4096).
pub const DEFAULT_ITERATIONS: u32 = 4096;

const SHA256_LEN: usize = 32;

/// The stored SCRAM credential for a role. Contains no plaintext password
/// and no value from which the password can be cheaply recovered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScramVerifier {
    /// Random per-role salt.
    pub salt: Vec<u8>,
    /// PBKDF2 iteration count.
    pub iterations: u32,
    /// `SHA-256(HMAC(SaltedPassword, "Client Key"))`.
    pub stored_key: [u8; SHA256_LEN],
    /// `HMAC(SaltedPassword, "Server Key")`.
    pub server_key: [u8; SHA256_LEN],
}

fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; SHA256_LEN] {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(msg);
    mac.finalize().into_bytes().into()
}

fn sha256(data: &[u8]) -> [u8; SHA256_LEN] {
    let mut h = Sha256::new();
    h.update(data);
    h.finalize().into()
}

fn salted_password(password: &[u8], salt: &[u8], iterations: u32) -> [u8; SHA256_LEN] {
    let mut out = [0u8; SHA256_LEN];
    pbkdf2::pbkdf2::<HmacSha256>(password, salt, iterations, &mut out)
        .expect("pbkdf2 output length is valid for SHA-256");
    out
}

impl ScramVerifier {
    /// Derive a verifier from a plaintext password using a fresh random
    /// salt and the default iteration count. The plaintext is consumed
    /// here and never stored.
    pub fn from_password(password: &str) -> Self {
        let mut salt = vec![0u8; 16];
        rand::thread_rng().fill_bytes(&mut salt);
        Self::from_password_with(password, salt, DEFAULT_ITERATIONS)
    }

    /// Derive a verifier with an explicit salt and iteration count
    /// (deterministic; used in tests and for reproducible provisioning).
    pub fn from_password_with(password: &str, salt: Vec<u8>, iterations: u32) -> Self {
        let salted = salted_password(password.as_bytes(), &salt, iterations);
        let client_key = hmac_sha256(&salted, b"Client Key");
        let stored_key = sha256(&client_key);
        let server_key = hmac_sha256(&salted, b"Server Key");
        ScramVerifier {
            salt,
            iterations,
            stored_key,
            server_key,
        }
    }

    /// Serialize to a compact, self-describing byte layout for storage:
    /// `[version:u8=1][iterations:u32 LE][salt_len:u16 LE][salt][stored_key:32][server_key:32]`.
    /// Contains no plaintext and is stable across restarts.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(1 + 4 + 2 + self.salt.len() + 64);
        out.push(1u8); // format version
        out.extend_from_slice(&self.iterations.to_le_bytes());
        out.extend_from_slice(&(self.salt.len() as u16).to_le_bytes());
        out.extend_from_slice(&self.salt);
        out.extend_from_slice(&self.stored_key);
        out.extend_from_slice(&self.server_key);
        out
    }

    /// Parse the layout produced by [`ScramVerifier::to_bytes`]. Returns
    /// `None` on any structural mismatch.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 1 + 4 + 2 {
            return None;
        }
        if bytes[0] != 1 {
            return None;
        }
        let iterations = u32::from_le_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]);
        let salt_len = u16::from_le_bytes([bytes[5], bytes[6]]) as usize;
        let salt_start = 7;
        let salt_end = salt_start + salt_len;
        let keys_end = salt_end + SHA256_LEN + SHA256_LEN;
        if bytes.len() != keys_end {
            return None;
        }
        let salt = bytes[salt_start..salt_end].to_vec();
        let mut stored_key = [0u8; SHA256_LEN];
        let mut server_key = [0u8; SHA256_LEN];
        stored_key.copy_from_slice(&bytes[salt_end..salt_end + SHA256_LEN]);
        server_key.copy_from_slice(&bytes[salt_end + SHA256_LEN..keys_end]);
        Some(ScramVerifier {
            salt,
            iterations,
            stored_key,
            server_key,
        })
    }
}

/// Errors during the SCRAM exchange. All map to authentication failure;
/// the variant is for logging, never sent verbatim to the client.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ScramError {
    /// A SCRAM message could not be parsed.
    #[error("malformed SCRAM message: {0}")]
    Malformed(String),
    /// The client proof did not verify against the stored key.
    #[error("client proof verification failed")]
    ProofMismatch,
    /// The server-nonce check failed (client did not echo our nonce).
    #[error("nonce mismatch in client-final")]
    NonceMismatch,
}

/// Parse `client-first-message`: `n,,n=user,r=nonce` (we ignore channel
/// binding and authzid). Returns `(username, client_nonce, client_first_bare)`.
pub fn parse_client_first(msg: &str) -> Result<(String, String, String), ScramError> {
    // GS2 header is the first two comma fields ("n," + authzid). The
    // client-first-bare is everything after the GS2 header.
    let bare_start = {
        // Find the end of the GS2 header: two commas.
        let mut commas = 0;
        let mut idx = None;
        for (i, b) in msg.bytes().enumerate() {
            if b == b',' {
                commas += 1;
                if commas == 2 {
                    idx = Some(i + 1);
                    break;
                }
            }
        }
        idx.ok_or_else(|| ScramError::Malformed("missing GS2 header".into()))?
    };
    let bare = &msg[bare_start..];

    let mut username = None;
    let mut nonce = None;
    for field in bare.split(',') {
        if let Some(v) = field.strip_prefix("n=") {
            username = Some(scram_unescape(v));
        } else if let Some(v) = field.strip_prefix("r=") {
            nonce = Some(v.to_string());
        }
    }
    let username = username.ok_or_else(|| ScramError::Malformed("missing n=".into()))?;
    let nonce = nonce.ok_or_else(|| ScramError::Malformed("missing r=".into()))?;
    if nonce.is_empty() {
        return Err(ScramError::Malformed("empty client nonce".into()));
    }
    Ok((username, nonce, bare.to_string()))
}

/// SASLprep-lite: SCRAM escapes `=` as `=3D` and `,` as `=2C` in the
/// username field. Reverse that.
fn scram_unescape(s: &str) -> String {
    s.replace("=2C", ",").replace("=3D", "=")
}

/// Generate a base64-safe random nonce of `n` bytes.
pub fn generate_nonce(n: usize) -> String {
    let mut buf = vec![0u8; n];
    rand::thread_rng().fill_bytes(&mut buf);
    // Use URL-safe-ish printable chars; SCRAM forbids comma in the nonce.
    B64.encode(&buf).replace(',', "_")
}

/// Build the `server-first-message`: `r=<combined>,s=<salt-b64>,i=<iters>`.
pub fn server_first_message(combined_nonce: &str, verifier: &ScramVerifier) -> String {
    format!(
        "r={},s={},i={}",
        combined_nonce,
        B64.encode(&verifier.salt),
        verifier.iterations
    )
}

/// Parse `client-final-message`: `c=<b64>,r=<combined-nonce>,p=<proof-b64>`.
/// Returns `(combined_nonce, client_proof, client_final_without_proof)`.
pub fn parse_client_final(msg: &str) -> Result<(String, Vec<u8>, String), ScramError> {
    let mut channel = None;
    let mut nonce = None;
    let mut proof_b64 = None;
    for field in msg.split(',') {
        if let Some(v) = field.strip_prefix("c=") {
            channel = Some(v.to_string());
        } else if let Some(v) = field.strip_prefix("r=") {
            nonce = Some(v.to_string());
        } else if let Some(v) = field.strip_prefix("p=") {
            proof_b64 = Some(v.to_string());
        }
    }
    let channel = channel.ok_or_else(|| ScramError::Malformed("missing c=".into()))?;
    let nonce = nonce.ok_or_else(|| ScramError::Malformed("missing r=".into()))?;
    let proof_b64 = proof_b64.ok_or_else(|| ScramError::Malformed("missing p=".into()))?;

    let proof = B64
        .decode(proof_b64.as_bytes())
        .map_err(|e| ScramError::Malformed(format!("bad proof base64: {e}")))?;
    if proof.len() != SHA256_LEN {
        return Err(ScramError::Malformed("proof wrong length".into()));
    }

    // client-final-without-proof is everything up to ",p=".
    let without_proof = format!("c={channel},r={nonce}");
    Ok((nonce, proof, without_proof))
}

/// Verify the client proof and produce the `server-final-message`
/// (`v=<server-signature-b64>`) on success.
///
/// `auth_message = client_first_bare + "," + server_first + "," + client_final_without_proof`.
pub fn verify_and_server_final(
    verifier: &ScramVerifier,
    auth_message: &str,
    client_proof: &[u8],
) -> Result<String, ScramError> {
    // ClientSignature = HMAC(StoredKey, AuthMessage)
    let client_signature = hmac_sha256(&verifier.stored_key, auth_message.as_bytes());

    // ClientKey = ClientProof XOR ClientSignature
    let mut client_key = [0u8; SHA256_LEN];
    for i in 0..SHA256_LEN {
        client_key[i] = client_proof[i] ^ client_signature[i];
    }

    // The recovered ClientKey must hash to the StoredKey.
    let recovered_stored = sha256(&client_key);
    if recovered_stored.ct_eq(&verifier.stored_key).unwrap_u8() != 1 {
        return Err(ScramError::ProofMismatch);
    }

    // ServerSignature = HMAC(ServerKey, AuthMessage)
    let server_signature = hmac_sha256(&verifier.server_key, auth_message.as_bytes());
    Ok(format!("v={}", B64.encode(server_signature)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A full client-side SCRAM run, used to drive the server side under
    /// test. This is the reference client computation from RFC 5802.
    fn client_proof(
        password: &str,
        verifier: &ScramVerifier,
        auth_message: &str,
    ) -> Vec<u8> {
        let salted = salted_password(password.as_bytes(), &verifier.salt, verifier.iterations);
        let client_key = hmac_sha256(&salted, b"Client Key");
        let stored_key = sha256(&client_key);
        let client_signature = hmac_sha256(&stored_key, auth_message.as_bytes());
        let mut proof = [0u8; SHA256_LEN];
        for i in 0..SHA256_LEN {
            proof[i] = client_key[i] ^ client_signature[i];
        }
        proof.to_vec()
    }

    #[test]
    fn verifier_derivation_is_deterministic() {
        let salt = vec![1u8; 16];
        let v1 = ScramVerifier::from_password_with("hunter2", salt.clone(), 4096);
        let v2 = ScramVerifier::from_password_with("hunter2", salt, 4096);
        assert_eq!(v1, v2);
        // Different password → different keys.
        let v3 = ScramVerifier::from_password_with("hunter3", vec![1u8; 16], 4096);
        assert_ne!(v1.stored_key, v3.stored_key);
    }

    #[test]
    fn verifier_stores_no_plaintext() {
        let v = ScramVerifier::from_password("super-secret");
        // The plaintext bytes must not appear in any stored field.
        let needle = b"super-secret";
        assert!(!v.salt.windows(needle.len()).any(|w| w == needle));
        assert!(!v.stored_key.windows(needle.len()).any(|w| w == needle));
        assert!(!v.server_key.windows(needle.len()).any(|w| w == needle));
    }

    #[test]
    fn full_exchange_succeeds_with_correct_password() {
        let password = "correct horse battery staple";
        // The verifier is derived from the PASSWORD (first arg), with a
        // fixed salt + iterations for determinism.
        let verifier = ScramVerifier::from_password_with(password, vec![7u8; 16], 4096);

        // client-first
        let client_nonce = "rOprNGfwEbeRWgbNEkqO";
        let client_first = format!("n,,n=alice,r={client_nonce}");
        let (user, c_nonce, client_first_bare) = parse_client_first(&client_first).unwrap();
        assert_eq!(user, "alice");
        assert_eq!(c_nonce, client_nonce);

        // server-first
        let server_nonce = "3rfcNHYJY1ZVvWVs7j";
        let combined = format!("{client_nonce}{server_nonce}");
        let server_first = server_first_message(&combined, &verifier);

        // client-final (without proof), then proof over the auth message
        let client_final_without_proof = format!("c=biws,r={combined}");
        let auth_message =
            format!("{client_first_bare},{server_first},{client_final_without_proof}");
        let proof = client_proof(password, &verifier, &auth_message);
        let client_final = format!("{client_final_without_proof},p={}", B64.encode(&proof));

        // server parses + verifies
        let (final_nonce, parsed_proof, without_proof) =
            parse_client_final(&client_final).unwrap();
        assert_eq!(final_nonce, combined);
        assert_eq!(without_proof, client_final_without_proof);

        let recomputed_auth =
            format!("{client_first_bare},{server_first},{without_proof}");
        let server_final =
            verify_and_server_final(&verifier, &recomputed_auth, &parsed_proof).unwrap();
        assert!(server_final.starts_with("v="));
    }

    #[test]
    fn wrong_password_fails_verification() {
        let verifier = ScramVerifier::from_password_with("alice", vec![9u8; 16], 4096);
        let client_first_bare = "n=alice,r=abc";
        let server_first = server_first_message("abcXYZ", &verifier);
        let without_proof = "c=biws,r=abcXYZ";
        let auth_message = format!("{client_first_bare},{server_first},{without_proof}");
        // Proof computed with the WRONG password.
        let proof = client_proof("wrong-password", &verifier, &auth_message);
        let err = verify_and_server_final(&verifier, &auth_message, &proof).unwrap_err();
        assert_eq!(err, ScramError::ProofMismatch);
    }

    #[test]
    fn malformed_messages_are_rejected() {
        assert!(matches!(
            parse_client_first("garbage"),
            Err(ScramError::Malformed(_))
        ));
        assert!(matches!(
            parse_client_final("no fields here"),
            Err(ScramError::Malformed(_))
        ));
    }

    #[test]
    fn username_escaping_roundtrips() {
        // A username containing '=' and ',' is escaped by the client.
        let client_first = "n,,n=od=3Dd=2Cd,r=nonce123";
        let (user, _, _) = parse_client_first(client_first).unwrap();
        assert_eq!(user, "od=d,d");
    }

    #[test]
    fn generated_nonce_has_no_comma() {
        for _ in 0..20 {
            let n = generate_nonce(18);
            assert!(!n.contains(','), "SCRAM nonce must not contain a comma");
            assert!(!n.is_empty());
        }
    }

    #[test]
    fn verifier_byte_roundtrip() {
        let v = ScramVerifier::from_password_with("pw", vec![3u8; 16], 8192);
        let bytes = v.to_bytes();
        let back = ScramVerifier::from_bytes(&bytes).expect("roundtrip");
        assert_eq!(v, back);
        // Wrong version / truncation rejected.
        assert!(ScramVerifier::from_bytes(&[]).is_none());
        assert!(ScramVerifier::from_bytes(&[2, 0, 0, 0, 0, 0, 0]).is_none());
        let mut truncated = bytes.clone();
        truncated.pop();
        assert!(ScramVerifier::from_bytes(&truncated).is_none());
    }
}
