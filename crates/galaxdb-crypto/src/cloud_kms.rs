//! Native cloud KMS key providers over REST (feature `cloud-kms`).
//!
//! No `aws-sdk-*` / `google-cloud-*` / `azure_*` dependency — every request is
//! built and signed in-crate and sent with `ureq` (rustls). Each provider
//! implements the [`crate::key_provider::KeyProvider`] DEK wrap/unwrap contract:
//!
//! - [`AwsKmsKeyProvider`]   — `kms:GenerateDataKey` / `kms:Decrypt`, SigV4.
//! - [`GcpKmsKeyProvider`]   — Cloud KMS `:encrypt` / `:decrypt`, OAuth2 bearer.
//! - [`AzureKeyVaultKeyProvider`] — Key Vault `wrapKey` / `unwrapKey`, Entra bearer.
//!
//! On any wrap/unwrap failure a typed [`GalaxError::Kms`] is returned — there is
//! never a fall back to a local or synthetic key (engineering-principles §2).
//! Credentials come from the environment and are never logged.

use base64::Engine as _;
use galaxdb_common::{GalaxError, GalaxResult};
use hmac::{Hmac, KeyInit, Mac};
use rand::RngCore;
use sha2::{Digest, Sha256};

use crate::key_provider::KeyProvider;

const B64: base64::engine::general_purpose::GeneralPurpose = base64::engine::general_purpose::STANDARD;

// ===========================================================================
// Shared signing / time helpers
// ===========================================================================

fn sha256_hex(data: &[u8]) -> String {
    let d = Sha256::digest(data);
    let mut s = String::with_capacity(64);
    for b in d {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn hmac_sha256(key: &[u8], msg: &[u8]) -> Vec<u8> {
    let mut mac = <Hmac<Sha256> as KeyInit>::new_from_slice(key).expect("any key len");
    mac.update(msg);
    mac.finalize().into_bytes().to_vec()
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

/// `(amz_date = YYYYMMDDTHHMMSSZ, date_stamp = YYYYMMDD)` for the current UTC
/// time (pure civil-date arithmetic; no chrono).
fn now_amz_time() -> (String, String) {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86_400) as i64;
    let sod = secs % 86_400;
    let (h, mi, s) = (sod / 3600, (sod % 3600) / 60, sod % 60);
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (
        format!("{:04}{:02}{:02}T{:02}{:02}{:02}Z", y, m, d, h, mi, s),
        format!("{:04}{:02}{:02}", y, m, d),
    )
}

// ===========================================================================
// AWS KMS (SigV4-signed JSON POST to kms.<region>.amazonaws.com)
// ===========================================================================

/// AWS KMS key provider. `key_id` is a key id, alias (`alias/...`), or ARN.
pub struct AwsKmsKeyProvider {
    key_id: String,
    region: String,
    endpoint_host: String,
    access_key: String,
    secret_key: String,
    session_token: Option<String>,
}

impl AwsKmsKeyProvider {
    /// Build from a key identifier, reading region + credentials from the
    /// environment (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
    /// `AWS_SESSION_TOKEN`, region via `GALAXDB_S3_REGION`/`AWS_REGION`/
    /// `AWS_DEFAULT_REGION`). `GALAXDB_KMS_ENDPOINT` overrides the host for
    /// testing against a KMS-compatible mock.
    pub fn from_key_id(key_id: &str) -> GalaxResult<Self> {
        let access_key = require_env("AWS_ACCESS_KEY_ID")?;
        let secret_key = require_env("AWS_SECRET_ACCESS_KEY")?;
        let session_token = std::env::var("AWS_SESSION_TOKEN").ok().filter(|s| !s.is_empty());
        let region = ["GALAXDB_KMS_REGION", "GALAXDB_S3_REGION", "AWS_REGION", "AWS_DEFAULT_REGION"]
            .iter()
            .find_map(|v| std::env::var(v).ok().filter(|s| !s.is_empty()))
            .unwrap_or_else(|| "us-east-1".to_string());
        let endpoint_host = std::env::var("GALAXDB_KMS_ENDPOINT")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| format!("kms.{}.amazonaws.com", region));
        Ok(Self {
            key_id: key_id.to_string(),
            region,
            endpoint_host,
            access_key,
            secret_key,
            session_token,
        })
    }

    /// Issue a signed KMS JSON POST for the given `X-Amz-Target` action.
    fn kms_call(&self, target: &str, body: &str) -> GalaxResult<String> {
        let (amz_date, date_stamp) = now_amz_time();
        let payload_hash = sha256_hex(body.as_bytes());
        let host = &self.endpoint_host;
        let content_type = "application/x-amz-json-1.1";

        // Canonical headers (sorted): content-type, host, x-amz-date,
        // x-amz-target [, x-amz-security-token].
        let mut canonical_headers = format!(
            "content-type:{content_type}\nhost:{host}\nx-amz-date:{amz_date}\nx-amz-target:{target}\n"
        );
        let mut signed_headers = "content-type;host;x-amz-date;x-amz-target".to_string();
        if let Some(tok) = &self.session_token {
            canonical_headers.push_str(&format!("x-amz-security-token:{tok}\n"));
            signed_headers.push_str(";x-amz-security-token");
        }

        let canonical_request = format!(
            "POST\n/\n\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
        );
        let scope = format!("{date_stamp}/{}/kms/aws4_request", self.region);
        let string_to_sign = format!(
            "AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}",
            sha256_hex(canonical_request.as_bytes())
        );
        let k_date = hmac_sha256(format!("AWS4{}", self.secret_key).as_bytes(), date_stamp.as_bytes());
        let k_region = hmac_sha256(&k_date, self.region.as_bytes());
        let k_service = hmac_sha256(&k_region, b"kms");
        let k_signing = hmac_sha256(&k_service, b"aws4_request");
        let signature = hex_lower(&hmac_sha256(&k_signing, string_to_sign.as_bytes()));
        let authorization = format!(
            "AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders={signed_headers}, Signature={signature}",
            self.access_key
        );

        let url = format!("https://{host}/");
        let mut req = ureq::post(&url)
            .set("Host", host)
            .set("Content-Type", content_type)
            .set("X-Amz-Date", &amz_date)
            .set("X-Amz-Target", target)
            .set("Authorization", &authorization);
        if let Some(tok) = &self.session_token {
            req = req.set("X-Amz-Security-Token", tok);
        }
        let resp = req.send_string(body).map_err(map_ureq("AWS KMS"))?;
        resp.into_string()
            .map_err(|e| GalaxError::Kms(format!("AWS KMS response read failed: {e}")))
    }
}

impl KeyProvider for AwsKmsKeyProvider {
    fn generate_data_key(&self) -> GalaxResult<(Vec<u8>, Vec<u8>)> {
        let body = format!(
            r#"{{"KeyId":"{}","KeySpec":"AES_256"}}"#,
            json_escape(&self.key_id)
        );
        let resp = self.kms_call("TrentService.GenerateDataKey", &body)?;
        let v: serde_json::Value = serde_json::from_str(&resp)
            .map_err(|e| GalaxError::Kms(format!("AWS KMS GenerateDataKey parse: {e}")))?;
        let plaintext_b64 = v.get("Plaintext").and_then(|x| x.as_str()).ok_or_else(|| {
            GalaxError::Kms("AWS KMS GenerateDataKey response missing Plaintext".into())
        })?;
        let blob_b64 = v.get("CiphertextBlob").and_then(|x| x.as_str()).ok_or_else(|| {
            GalaxError::Kms("AWS KMS GenerateDataKey response missing CiphertextBlob".into())
        })?;
        let plaintext = B64
            .decode(plaintext_b64)
            .map_err(|_| GalaxError::Kms("AWS KMS Plaintext is not base64".into()))?;
        let blob = B64
            .decode(blob_b64)
            .map_err(|_| GalaxError::Kms("AWS KMS CiphertextBlob is not base64".into()))?;
        Ok((plaintext, blob))
    }

    fn decrypt_data_key(&self, encrypted_key: &[u8]) -> GalaxResult<Vec<u8>> {
        let body = format!(r#"{{"CiphertextBlob":"{}"}}"#, B64.encode(encrypted_key));
        let resp = self.kms_call("TrentService.Decrypt", &body)?;
        let v: serde_json::Value = serde_json::from_str(&resp)
            .map_err(|e| GalaxError::Kms(format!("AWS KMS Decrypt parse: {e}")))?;
        let plaintext_b64 = v.get("Plaintext").and_then(|x| x.as_str()).ok_or_else(|| {
            GalaxError::Kms("AWS KMS Decrypt response missing Plaintext".into())
        })?;
        B64.decode(plaintext_b64)
            .map_err(|_| GalaxError::Kms("AWS KMS Plaintext is not base64".into()))
    }

    fn provider_name(&self) -> &str {
        "aws-kms"
    }
}

fn require_env(var: &str) -> GalaxResult<String> {
    std::env::var(var)
        .ok()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| GalaxError::Kms(format!("cloud KMS requires the {var} environment variable")))
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn map_ureq(op: &'static str) -> impl Fn(ureq::Error) -> GalaxError {
    move |e| match e {
        ureq::Error::Status(code, _) => GalaxError::Kms(format!("{op} failed with HTTP {code}")),
        ureq::Error::Transport(t) => GalaxError::Kms(format!("{op} transport error: {}", t.kind())),
    }
}

/// Generate a fresh random 32-byte data encryption key.
fn random_dek() -> Vec<u8> {
    let mut dek = vec![0u8; 32];
    rand::thread_rng().fill_bytes(&mut dek);
    dek
}

// ===========================================================================
// GCP Cloud KMS (REST, OAuth2 bearer)
// ===========================================================================

const GCP_METADATA_TOKEN_URL: &str = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token";

/// Google Cloud KMS provider. `key_name` is the full resource name
/// `projects/P/locations/L/keyRings/R/cryptoKeys/K`. The DEK is generated
/// locally and wrapped with the KMS key via `:encrypt`.
pub struct GcpKmsKeyProvider {
    key_name: String,
    access_token: String,
}

impl GcpKmsKeyProvider {
    /// Build from the crypto-key resource name. The OAuth2 access token comes
    /// from `GALAXDB_GCP_KMS_TOKEN` / `GALAXDB_GCS_ACCESS_TOKEN`, or the GCE
    /// metadata server (workload identity).
    pub fn from_key_name(key_name: &str) -> GalaxResult<Self> {
        Ok(Self {
            key_name: key_name.to_string(),
            access_token: resolve_gcp_token()?,
        })
    }

    fn call(&self, verb: &str, body: &str) -> GalaxResult<String> {
        let url = format!(
            "https://cloudkms.googleapis.com/v1/{}:{verb}",
            self.key_name
        );
        let resp = ureq::post(&url)
            .set("Authorization", &format!("Bearer {}", self.access_token))
            .set("Content-Type", "application/json")
            .send_string(body)
            .map_err(map_ureq("GCP KMS"))?;
        resp.into_string()
            .map_err(|e| GalaxError::Kms(format!("GCP KMS response read failed: {e}")))
    }
}

impl KeyProvider for GcpKmsKeyProvider {
    fn generate_data_key(&self) -> GalaxResult<(Vec<u8>, Vec<u8>)> {
        let dek = random_dek();
        let body = format!(r#"{{"plaintext":"{}"}}"#, B64.encode(&dek));
        let resp = self.call("encrypt", &body)?;
        let v: serde_json::Value = serde_json::from_str(&resp)
            .map_err(|e| GalaxError::Kms(format!("GCP KMS encrypt parse: {e}")))?;
        let ct_b64 = v.get("ciphertext").and_then(|x| x.as_str()).ok_or_else(|| {
            GalaxError::Kms("GCP KMS encrypt response missing ciphertext".into())
        })?;
        let blob = B64
            .decode(ct_b64)
            .map_err(|_| GalaxError::Kms("GCP KMS ciphertext is not base64".into()))?;
        Ok((dek, blob))
    }

    fn decrypt_data_key(&self, encrypted_key: &[u8]) -> GalaxResult<Vec<u8>> {
        let body = format!(r#"{{"ciphertext":"{}"}}"#, B64.encode(encrypted_key));
        let resp = self.call("decrypt", &body)?;
        let v: serde_json::Value = serde_json::from_str(&resp)
            .map_err(|e| GalaxError::Kms(format!("GCP KMS decrypt parse: {e}")))?;
        let pt_b64 = v.get("plaintext").and_then(|x| x.as_str()).ok_or_else(|| {
            GalaxError::Kms("GCP KMS decrypt response missing plaintext".into())
        })?;
        B64.decode(pt_b64)
            .map_err(|_| GalaxError::Kms("GCP KMS plaintext is not base64".into()))
    }

    fn provider_name(&self) -> &str {
        "gcp-kms"
    }
}

fn resolve_gcp_token() -> GalaxResult<String> {
    for var in ["GALAXDB_GCP_KMS_TOKEN", "GALAXDB_GCS_ACCESS_TOKEN"] {
        if let Ok(tok) = std::env::var(var) {
            if !tok.is_empty() {
                return Ok(tok);
            }
        }
    }
    match ureq::get(GCP_METADATA_TOKEN_URL)
        .set("Metadata-Flavor", "Google")
        .timeout(std::time::Duration::from_secs(2))
        .call()
    {
        Ok(resp) => {
            let body = resp
                .into_string()
                .map_err(|e| GalaxError::Kms(format!("GCP metadata token read: {e}")))?;
            let v: serde_json::Value = serde_json::from_str(&body)
                .map_err(|e| GalaxError::Kms(format!("GCP metadata token parse: {e}")))?;
            v.get("access_token")
                .and_then(|t| t.as_str())
                .map(|s| s.to_string())
                .ok_or_else(|| GalaxError::Kms("GCP metadata token missing access_token".into()))
        }
        Err(_) => Err(GalaxError::Kms(
            "GCP KMS requires GALAXDB_GCP_KMS_TOKEN (or a reachable GCE metadata server)".into(),
        )),
    }
}

// ===========================================================================
// Azure Key Vault (REST, Entra bearer, wrapKey/unwrapKey)
// ===========================================================================

/// Azure Key Vault provider. Wraps the locally-generated DEK with an RSA key in
/// the vault via `wrapKey` (RSA-OAEP-256). The DEK is generated locally; only
/// the wrapped blob is persisted.
pub struct AzureKeyVaultKeyProvider {
    /// Full key URL, e.g. `https://vault.vault.azure.net/keys/mykey` (an
    /// optional version segment may be appended).
    key_url: String,
    access_token: String,
}

impl AzureKeyVaultKeyProvider {
    /// Build from `vault/key` (or `vault/key/version`). The Entra bearer token
    /// comes from `GALAXDB_AZURE_KV_TOKEN`.
    pub fn from_spec(vault: &str, key: &str) -> GalaxResult<Self> {
        let key_url = format!("https://{vault}.vault.azure.net/keys/{key}");
        let access_token = require_env("GALAXDB_AZURE_KV_TOKEN")
            .map_err(|_| GalaxError::Kms(
                "Azure Key Vault requires GALAXDB_AZURE_KV_TOKEN (Entra bearer token)".into(),
            ))?;
        Ok(Self { key_url, access_token })
    }

    fn call(&self, op: &str, body: &str) -> GalaxResult<String> {
        let url = format!("{}/{op}?api-version=7.4", self.key_url);
        let resp = ureq::post(&url)
            .set("Authorization", &format!("Bearer {}", self.access_token))
            .set("Content-Type", "application/json")
            .send_string(body)
            .map_err(map_ureq("Azure Key Vault"))?;
        resp.into_string()
            .map_err(|e| GalaxError::Kms(format!("Azure KV response read failed: {e}")))
    }
}

/// URL-safe base64 without padding (the JWK `value` encoding used by Key Vault).
fn b64url_nopad_encode(data: &[u8]) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(data)
}
fn b64url_nopad_decode(s: &str) -> GalaxResult<Vec<u8>> {
    base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(s.trim_end_matches('='))
        .map_err(|_| GalaxError::Kms("Azure KV value is not base64url".into()))
}

impl KeyProvider for AzureKeyVaultKeyProvider {
    fn generate_data_key(&self) -> GalaxResult<(Vec<u8>, Vec<u8>)> {
        let dek = random_dek();
        let body = format!(
            r#"{{"alg":"RSA-OAEP-256","value":"{}"}}"#,
            b64url_nopad_encode(&dek)
        );
        let resp = self.call("wrapkey", &body)?;
        let v: serde_json::Value = serde_json::from_str(&resp)
            .map_err(|e| GalaxError::Kms(format!("Azure KV wrapKey parse: {e}")))?;
        let wrapped_b64 = v.get("value").and_then(|x| x.as_str()).ok_or_else(|| {
            GalaxError::Kms("Azure KV wrapKey response missing value".into())
        })?;
        Ok((dek, b64url_nopad_decode(wrapped_b64)?))
    }

    fn decrypt_data_key(&self, encrypted_key: &[u8]) -> GalaxResult<Vec<u8>> {
        let body = format!(
            r#"{{"alg":"RSA-OAEP-256","value":"{}"}}"#,
            b64url_nopad_encode(encrypted_key)
        );
        let resp = self.call("unwrapkey", &body)?;
        let v: serde_json::Value = serde_json::from_str(&resp)
            .map_err(|e| GalaxError::Kms(format!("Azure KV unwrapKey parse: {e}")))?;
        let dek_b64 = v.get("value").and_then(|x| x.as_str()).ok_or_else(|| {
            GalaxError::Kms("Azure KV unwrapKey response missing value".into())
        })?;
        b64url_nopad_decode(dek_b64)
    }

    fn provider_name(&self) -> &str {
        "azure-kv"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aws_sigv4_helpers_are_stable() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        // HMAC chain determinism.
        let a = hmac_sha256(b"AWS4secret", b"20260624");
        let b = hmac_sha256(b"AWS4secret", b"20260624");
        assert_eq!(a, b);
    }

    #[test]
    fn random_dek_is_32_bytes_and_unique() {
        let a = random_dek();
        let b = random_dek();
        assert_eq!(a.len(), 32);
        assert_ne!(a, b, "two DEKs must differ");
    }

    #[test]
    fn b64url_round_trips() {
        let data = b"0123456789abcdef0123456789abcdef";
        let enc = b64url_nopad_encode(data);
        assert!(!enc.contains('='), "no padding");
        assert_eq!(b64url_nopad_decode(&enc).unwrap(), data);
    }

    #[test]
    fn json_escape_handles_quotes_and_backslashes() {
        assert_eq!(json_escape(r#"a"b\c"#), r#"a\"b\\c"#);
    }

    #[test]
    fn aws_from_key_id_errors_without_credentials() {
        // Only assert the error message shape when creds are absent in env.
        if std::env::var("AWS_ACCESS_KEY_ID").is_err() {
            match AwsKmsKeyProvider::from_key_id("alias/test") {
                Ok(_) => panic!("expected error when AWS_ACCESS_KEY_ID is unset"),
                Err(err) => assert!(format!("{err}").contains("AWS_ACCESS_KEY_ID")),
            }
        }
    }
}
