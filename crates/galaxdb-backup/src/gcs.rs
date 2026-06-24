//! Google Cloud Storage object store over the JSON REST API (OAuth2 bearer).
//!
//! No `google-cloud-*` dependency. An OAuth2 access token is supplied by the
//! environment (`GALAXDB_GCS_ACCESS_TOKEN`) or fetched from the GCE metadata
//! server when running on Google infrastructure (workload identity). The token
//! is sent as a bearer credential and never logged.

use std::io::Read;

use galaxdb_common::{GalaxError, GalaxResult};

use crate::{join_key, split_bucket_prefix, ObjectStore};

const STORAGE_BASE: &str = "https://storage.googleapis.com/storage/v1";
const UPLOAD_BASE: &str = "https://storage.googleapis.com/upload/storage/v1";
const METADATA_TOKEN_URL: &str = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token";

/// A GCS bucket/prefix object store.
pub struct GcsObjectStore {
    bucket: String,
    prefix: String,
    access_token: String,
}

impl GcsObjectStore {
    /// Build from the body of a `gs://` URL (`bucket/prefix...`).
    pub fn from_url(rest: &str) -> GalaxResult<Self> {
        let (bucket, prefix) = split_bucket_prefix(rest);
        if bucket.is_empty() {
            return Err(GalaxError::Internal(
                "gs:// target must include a bucket: gs://bucket[/prefix]".into(),
            ));
        }
        let access_token = resolve_token()?;
        Ok(Self {
            bucket,
            prefix,
            access_token,
        })
    }

    fn object_name(&self, key: &str) -> String {
        join_key(&self.prefix, key)
    }

    fn bearer(&self) -> String {
        format!("Bearer {}", self.access_token)
    }
}

impl ObjectStore for GcsObjectStore {
    fn put(&self, key: &str, data: &[u8]) -> GalaxResult<()> {
        let name = url_encode(&self.object_name(key));
        let url = format!(
            "{UPLOAD_BASE}/b/{}/o?uploadType=media&name={}",
            self.bucket, name
        );
        ureq::post(&url)
            .set("Authorization", &self.bearer())
            .set("Content-Type", "application/octet-stream")
            .send_bytes(data)
            .map_err(map_ureq("GCS PUT"))?;
        Ok(())
    }

    fn get(&self, key: &str) -> GalaxResult<Vec<u8>> {
        let name = url_encode(&self.object_name(key));
        let url = format!("{STORAGE_BASE}/b/{}/o/{}?alt=media", self.bucket, name);
        let resp = ureq::get(&url)
            .set("Authorization", &self.bearer())
            .call()
            .map_err(map_ureq("GCS GET"))?;
        let mut buf = Vec::new();
        resp.into_reader().read_to_end(&mut buf).map_err(GalaxError::Io)?;
        Ok(buf)
    }

    fn list(&self) -> GalaxResult<Vec<String>> {
        let url = format!(
            "{STORAGE_BASE}/b/{}/o?prefix={}",
            self.bucket,
            url_encode(&self.prefix)
        );
        let body = ureq::get(&url)
            .set("Authorization", &self.bearer())
            .call()
            .map_err(map_ureq("GCS LIST"))?
            .into_string()
            .map_err(GalaxError::Io)?;
        Ok(parse_list_names(&body, &self.prefix))
    }

    fn delete(&self, key: &str) -> GalaxResult<()> {
        let name = url_encode(&self.object_name(key));
        let url = format!("{STORAGE_BASE}/b/{}/o/{}", self.bucket, name);
        match ureq::delete(&url).set("Authorization", &self.bearer()).call() {
            Ok(_) => Ok(()),
            Err(ureq::Error::Status(404, _)) => Ok(()),
            Err(e) => Err(map_ureq("GCS DELETE")(e)),
        }
    }

    fn scheme(&self) -> &'static str {
        "gs"
    }
}

/// Resolve an OAuth2 access token: explicit env first, then the GCE metadata
/// server (workload identity). Returns a typed error if neither is available —
/// never a silent fallback.
fn resolve_token() -> GalaxResult<String> {
    if let Ok(tok) = std::env::var("GALAXDB_GCS_ACCESS_TOKEN") {
        if !tok.is_empty() {
            return Ok(tok);
        }
    }
    // Metadata server (only reachable on GCP). Short timeout so a non-GCP host
    // fails fast rather than hanging.
    match ureq::get(METADATA_TOKEN_URL)
        .set("Metadata-Flavor", "Google")
        .timeout(std::time::Duration::from_secs(2))
        .call()
    {
        Ok(resp) => {
            let body = resp.into_string().map_err(GalaxError::Io)?;
            let v: serde_json::Value = serde_json::from_str(&body).map_err(|e| {
                GalaxError::Internal(format!("GCS metadata token parse failed: {e}"))
            })?;
            v.get("access_token")
                .and_then(|t| t.as_str())
                .map(|s| s.to_string())
                .ok_or_else(|| {
                    GalaxError::Internal("GCS metadata token response missing access_token".into())
                })
        }
        Err(_) => Err(GalaxError::Internal(
            "GCS backup requires GALAXDB_GCS_ACCESS_TOKEN (or a reachable GCE metadata server)"
                .into(),
        )),
    }
}

/// Parse the `items[].name` array of a GCS objects.list response and strip the
/// base prefix to return backup file names.
fn parse_list_names(json: &str, prefix: &str) -> Vec<String> {
    let mut out = Vec::new();
    let Ok(v) = serde_json::from_str::<serde_json::Value>(json) else {
        return out;
    };
    let Some(items) = v.get("items").and_then(|i| i.as_array()) else {
        return out;
    };
    for item in items {
        if let Some(name) = item.get("name").and_then(|n| n.as_str()) {
            let stripped = if prefix.is_empty() {
                name.to_string()
            } else {
                name.strip_prefix(&format!("{}/", prefix.trim_end_matches('/')))
                    .unwrap_or(name)
                    .to_string()
            };
            if !stripped.is_empty() {
                out.push(stripped);
            }
        }
    }
    out
}

/// Percent-encode an object name for a GCS URL path/query (slashes encoded).
fn url_encode(s: &str) -> String {
    crate::sign::uri_encode(s, true)
}

fn map_ureq(op: &'static str) -> impl Fn(ureq::Error) -> GalaxError {
    move |e| match e {
        ureq::Error::Status(code, _) => {
            GalaxError::Internal(format!("{op} failed with HTTP {code}"))
        }
        ureq::Error::Transport(t) => {
            GalaxError::Internal(format!("{op} transport error: {}", t.kind()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_names_strips_prefix() {
        let json = r#"{"items":[{"name":"bk/db/wal.log"},{"name":"bk/db/sst_2.pax"}]}"#;
        assert_eq!(
            parse_list_names(json, "bk/db"),
            vec!["wal.log".to_string(), "sst_2.pax".to_string()]
        );
    }

    #[test]
    fn list_names_empty_when_no_items() {
        assert!(parse_list_names("{}", "p").is_empty());
        assert!(parse_list_names("not json", "p").is_empty());
    }

    #[test]
    fn object_name_uses_prefix() {
        let s = GcsObjectStore {
            bucket: "b".into(),
            prefix: "bk/db".into(),
            access_token: "x".into(),
        };
        assert_eq!(s.object_name("wal.log"), "bk/db/wal.log");
    }
}
