//! AWS S3 / S3-compatible object store over REST with in-crate SigV4.
//!
//! No `aws-sdk-*` dependency — requests are built and signed here and sent with
//! `ureq` (rustls). Path-style addressing is used so the same code works for
//! AWS and S3-compatible stores (MinIO, Ceph, R2) via a custom endpoint.
//!
//! Configuration (all from the environment; never logged):
//! - `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (required), `AWS_SESSION_TOKEN` (optional).
//! - region: `GALAXDB_S3_REGION` → `AWS_REGION` → `AWS_DEFAULT_REGION` → `us-east-1`.
//! - `GALAXDB_S3_ENDPOINT` (optional): custom endpoint for S3-compatible stores,
//!   e.g. `https://minio.internal:9000`.

use galaxdb_common::{GalaxError, GalaxResult};

use crate::sign::{self, SigV4Request};
use crate::{join_key, split_bucket_prefix};

/// An S3 (or S3-compatible) object store rooted at `bucket`/`prefix`.
pub struct S3ObjectStore {
    bucket: String,
    prefix: String,
    region: String,
    /// URL scheme (`https` by default; `http` only for explicit insecure endpoints).
    url_scheme: String,
    /// Host authority used in both the URL and the SigV4 `Host` header.
    host: String,
    access_key: String,
    secret_key: String,
    session_token: Option<String>,
}

impl S3ObjectStore {
    /// Build from the body of an `s3://` URL (`bucket/prefix...`).
    pub fn from_url(rest: &str) -> GalaxResult<Self> {
        let (bucket, prefix) = split_bucket_prefix(rest);
        if bucket.is_empty() {
            return Err(GalaxError::Internal(
                "s3:// target must include a bucket: s3://bucket[/prefix]".into(),
            ));
        }

        let access_key = require_env("AWS_ACCESS_KEY_ID")?;
        let secret_key = require_env("AWS_SECRET_ACCESS_KEY")?;
        let session_token = std::env::var("AWS_SESSION_TOKEN").ok().filter(|s| !s.is_empty());

        let region = first_env(&["GALAXDB_S3_REGION", "AWS_REGION", "AWS_DEFAULT_REGION"])
            .unwrap_or_else(|| "us-east-1".to_string());

        let (url_scheme, host) = match std::env::var("GALAXDB_S3_ENDPOINT").ok() {
            Some(ep) if !ep.is_empty() => parse_endpoint(&ep),
            _ => ("https".to_string(), format!("s3.{}.amazonaws.com", region)),
        };

        Ok(Self {
            bucket,
            prefix,
            region,
            url_scheme,
            host,
            access_key,
            secret_key,
            session_token,
        })
    }

    /// Full canonical URI for path-style addressing: `/bucket/prefix/key`.
    fn canonical_uri(&self, key: &str) -> String {
        let object = join_key(&self.prefix, key);
        let path = format!("/{}/{}", self.bucket, object);
        sign::uri_encode(&path, false)
    }

    fn url(&self, canonical_uri: &str, query: &str) -> String {
        if query.is_empty() {
            format!("{}://{}{}", self.url_scheme, self.host, canonical_uri)
        } else {
            format!("{}://{}{}?{}", self.url_scheme, self.host, canonical_uri, query)
        }
    }

    fn sign_headers(
        &self,
        method: &str,
        canonical_uri: &str,
        canonical_query: &str,
        payload_hash: &str,
    ) -> sign::SigV4Headers {
        let (amz_date, date_stamp) = sign::now_amz_time();
        SigV4Request {
            access_key: &self.access_key,
            secret_key: &self.secret_key,
            region: &self.region,
            service: "s3",
            method,
            host: &self.host,
            canonical_uri,
            canonical_query,
            amz_date: &amz_date,
            date_stamp: &date_stamp,
            payload_sha256_hex: payload_hash,
            security_token: self.session_token.as_deref(),
        }
        .sign()
    }
}

impl crate::ObjectStore for S3ObjectStore {
    fn put(&self, key: &str, data: &[u8]) -> GalaxResult<()> {
        let canonical_uri = self.canonical_uri(key);
        let payload_hash = sign::sha256_hex(data);
        let h = self.sign_headers("PUT", &canonical_uri, "", &payload_hash);
        let mut req = ureq::put(&self.url(&canonical_uri, ""))
            .set("Host", &self.host)
            .set("x-amz-date", &h.amz_date)
            .set("x-amz-content-sha256", &h.content_sha256)
            .set("Authorization", &h.authorization);
        if let Some(token) = &self.session_token {
            req = req.set("x-amz-security-token", token);
        }
        req.send_bytes(data).map_err(map_ureq("S3 PUT"))?;
        Ok(())
    }

    fn get(&self, key: &str) -> GalaxResult<Vec<u8>> {
        let canonical_uri = self.canonical_uri(key);
        let payload_hash = sign::sha256_hex(b"");
        let h = self.sign_headers("GET", &canonical_uri, "", &payload_hash);
        let mut req = ureq::get(&self.url(&canonical_uri, ""))
            .set("Host", &self.host)
            .set("x-amz-date", &h.amz_date)
            .set("x-amz-content-sha256", &h.content_sha256)
            .set("Authorization", &h.authorization);
        if let Some(token) = &self.session_token {
            req = req.set("x-amz-security-token", token);
        }
        let resp = req.call().map_err(map_ureq("S3 GET"))?;
        let mut buf = Vec::new();
        resp.into_reader()
            .read_to_end(&mut buf)
            .map_err(GalaxError::Io)?;
        Ok(buf)
    }

    fn list(&self) -> GalaxResult<Vec<String>> {
        // ListObjectsV2 against the bucket, filtered by our prefix.
        let canonical_uri = sign::uri_encode(&format!("/{}", self.bucket), false);
        let enc_prefix = sign::uri_encode(&self.prefix, true);
        let canonical_query = format!("list-type=2&prefix={}", enc_prefix);
        let payload_hash = sign::sha256_hex(b"");
        let h = self.sign_headers("GET", &canonical_uri, &canonical_query, &payload_hash);
        let mut req = ureq::get(&self.url(&canonical_uri, &canonical_query))
            .set("Host", &self.host)
            .set("x-amz-date", &h.amz_date)
            .set("x-amz-content-sha256", &h.content_sha256)
            .set("Authorization", &h.authorization);
        if let Some(token) = &self.session_token {
            req = req.set("x-amz-security-token", token);
        }
        let body = req
            .call()
            .map_err(map_ureq("S3 LIST"))?
            .into_string()
            .map_err(GalaxError::Io)?;
        Ok(parse_list_keys(&body, &self.prefix))
    }

    fn delete(&self, key: &str) -> GalaxResult<()> {
        let canonical_uri = self.canonical_uri(key);
        let payload_hash = sign::sha256_hex(b"");
        let h = self.sign_headers("DELETE", &canonical_uri, "", &payload_hash);
        let mut req = ureq::delete(&self.url(&canonical_uri, ""))
            .set("Host", &self.host)
            .set("x-amz-date", &h.amz_date)
            .set("x-amz-content-sha256", &h.content_sha256)
            .set("Authorization", &h.authorization);
        if let Some(token) = &self.session_token {
            req = req.set("x-amz-security-token", token);
        }
        match req.call() {
            Ok(_) => Ok(()),
            // S3 returns 204 for delete; ureq treats 2xx as Ok. A 404 is fine.
            Err(ureq::Error::Status(404, _)) => Ok(()),
            Err(e) => Err(map_ureq("S3 DELETE")(e)),
        }
    }

    fn scheme(&self) -> &'static str {
        "s3"
    }
}

use std::io::Read;

/// Extract `<Key>` element bodies from a ListObjectsV2 XML response and return
/// them with the base `prefix` stripped (so callers see backup file names).
fn parse_list_keys(xml: &str, prefix: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = xml;
    while let Some(start) = rest.find("<Key>") {
        let after = &rest[start + 5..];
        let Some(end) = after.find("</Key>") else {
            break;
        };
        let full = &after[..end];
        let name = match prefix.is_empty() {
            true => full.to_string(),
            false => full
                .strip_prefix(&format!("{}/", prefix.trim_end_matches('/')))
                .unwrap_or(full)
                .to_string(),
        };
        if !name.is_empty() {
            out.push(name);
        }
        rest = &after[end + 6..];
    }
    out
}

fn require_env(var: &str) -> GalaxResult<String> {
    std::env::var(var)
        .ok()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            GalaxError::Internal(format!(
                "S3 backup requires the {var} environment variable"
            ))
        })
}

fn first_env(vars: &[&str]) -> Option<String> {
    vars.iter()
        .find_map(|v| std::env::var(v).ok().filter(|s| !s.is_empty()))
}

/// Parse `GALAXDB_S3_ENDPOINT` into `(scheme, host_authority)`.
fn parse_endpoint(ep: &str) -> (String, String) {
    if let Some(host) = ep.strip_prefix("https://") {
        ("https".to_string(), host.trim_end_matches('/').to_string())
    } else if let Some(host) = ep.strip_prefix("http://") {
        ("http".to_string(), host.trim_end_matches('/').to_string())
    } else {
        ("https".to_string(), ep.trim_end_matches('/').to_string())
    }
}

/// Map a `ureq::Error` into a `GalaxError` WITHOUT leaking credentials. Only
/// the operation name and HTTP status (not request headers) are surfaced.
fn map_ureq(op: &'static str) -> impl Fn(ureq::Error) -> GalaxError {
    move |e| match e {
        ureq::Error::Status(code, _resp) => {
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
    fn parse_endpoint_variants() {
        assert_eq!(
            parse_endpoint("https://minio:9000"),
            ("https".to_string(), "minio:9000".to_string())
        );
        assert_eq!(
            parse_endpoint("http://localhost:9000/"),
            ("http".to_string(), "localhost:9000".to_string())
        );
        assert_eq!(
            parse_endpoint("s3.example.com"),
            ("https".to_string(), "s3.example.com".to_string())
        );
    }

    #[test]
    fn list_keys_strips_prefix() {
        let xml = "<ListBucketResult><Contents><Key>backups/db1/wal.log</Key></Contents>\
                   <Contents><Key>backups/db1/sst_1.pax</Key></Contents></ListBucketResult>";
        let keys = parse_list_keys(xml, "backups/db1");
        assert_eq!(keys, vec!["wal.log".to_string(), "sst_1.pax".to_string()]);
    }

    #[test]
    fn list_keys_no_prefix() {
        let xml = "<Contents><Key>wal.log</Key></Contents>";
        assert_eq!(parse_list_keys(xml, ""), vec!["wal.log".to_string()]);
    }
}
