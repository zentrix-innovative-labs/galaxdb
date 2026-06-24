//! Azure Blob Storage object store over REST with SharedKey authorization.
//!
//! No `azure_*` dependency. Requests are signed in-crate with the storage
//! account key (HMAC-SHA256 over the canonicalized request). Account and key
//! come from the environment (`AZURE_STORAGE_ACCOUNT`, `AZURE_STORAGE_KEY`) and
//! are never logged.

use std::io::Read;

use base64::Engine as _;
use galaxdb_common::{GalaxError, GalaxResult};
use hmac::{Hmac, KeyInit, Mac};
use sha2::Sha256;

use crate::sign;
use crate::{join_key, split_bucket_prefix, ObjectStore};

const API_VERSION: &str = "2021-08-06";

/// An Azure Blob container/prefix object store.
pub struct AzureBlobObjectStore {
    account: String,
    container: String,
    prefix: String,
    /// Base64-encoded account key (as Azure presents it).
    account_key_b64: String,
}

impl AzureBlobObjectStore {
    /// Build from the body of an `az://` URL (`container/prefix...`). The
    /// account is taken from `AZURE_STORAGE_ACCOUNT`.
    pub fn from_url(rest: &str) -> GalaxResult<Self> {
        let (container, prefix) = split_bucket_prefix(rest);
        if container.is_empty() {
            return Err(GalaxError::Internal(
                "az:// target must include a container: az://container[/prefix]".into(),
            ));
        }
        let account = require_env("AZURE_STORAGE_ACCOUNT")?;
        let account_key_b64 = require_env("AZURE_STORAGE_KEY")?;
        Ok(Self {
            account,
            container,
            prefix,
            account_key_b64,
        })
    }

    fn blob_name(&self, key: &str) -> String {
        join_key(&self.prefix, key)
    }

    fn host(&self) -> String {
        format!("{}.blob.core.windows.net", self.account)
    }

    fn blob_url(&self, blob: &str) -> String {
        format!("https://{}/{}/{}", self.host(), self.container, blob)
    }

    /// Compute the `Authorization: SharedKey` header value for a request.
    ///
    /// `canonical_headers` must be the sorted `x-ms-*` header block (each line
    /// `name:value\n`). `canonical_resource` is `/account/container/blob` plus
    /// any sorted query params (`\nname:value`). `content_length` is the body
    /// length as a string, or empty for zero-length bodies.
    fn shared_key_auth(
        &self,
        verb: &str,
        content_length: &str,
        canonical_headers: &str,
        canonical_resource: &str,
    ) -> GalaxResult<String> {
        // VERB + 11 standard header fields (all empty except Content-Length) +
        // canonicalized headers + canonicalized resource.
        let string_to_sign = format!(
            "{verb}\n\n\n{content_length}\n\n\n\n\n\n\n\n\n{canonical_headers}{canonical_resource}"
        );

        let key = base64::engine::general_purpose::STANDARD
            .decode(self.account_key_b64.trim())
            .map_err(|_| {
                GalaxError::Internal("AZURE_STORAGE_KEY is not valid base64".into())
            })?;
        let mut mac = Hmac::<Sha256>::new_from_slice(&key)
            .map_err(|_| GalaxError::Internal("invalid Azure account key length".into()))?;
        mac.update(string_to_sign.as_bytes());
        let sig = base64::engine::general_purpose::STANDARD.encode(mac.finalize().into_bytes());
        Ok(format!("SharedKey {}:{}", self.account, sig))
    }
}

impl ObjectStore for AzureBlobObjectStore {
    fn put(&self, key: &str, data: &[u8]) -> GalaxResult<()> {
        let blob = self.blob_name(key);
        let date = sign::now_rfc1123();
        let clen = data.len().to_string();
        // Canonicalized headers (sorted): x-ms-blob-type, x-ms-date, x-ms-version.
        let canonical_headers = format!(
            "x-ms-blob-type:BlockBlob\nx-ms-date:{date}\nx-ms-version:{API_VERSION}\n"
        );
        let canonical_resource = format!("/{}/{}/{}", self.account, self.container, blob);
        let auth = self.shared_key_auth("PUT", &clen, &canonical_headers, &canonical_resource)?;

        ureq::put(&self.blob_url(&blob))
            .set("x-ms-date", &date)
            .set("x-ms-version", API_VERSION)
            .set("x-ms-blob-type", "BlockBlob")
            .set("Content-Length", &clen)
            .set("Authorization", &auth)
            .send_bytes(data)
            .map_err(map_ureq("Azure PUT"))?;
        Ok(())
    }

    fn get(&self, key: &str) -> GalaxResult<Vec<u8>> {
        let blob = self.blob_name(key);
        let date = sign::now_rfc1123();
        let canonical_headers = format!("x-ms-date:{date}\nx-ms-version:{API_VERSION}\n");
        let canonical_resource = format!("/{}/{}/{}", self.account, self.container, blob);
        let auth = self.shared_key_auth("GET", "", &canonical_headers, &canonical_resource)?;

        let resp = ureq::get(&self.blob_url(&blob))
            .set("x-ms-date", &date)
            .set("x-ms-version", API_VERSION)
            .set("Authorization", &auth)
            .call()
            .map_err(map_ureq("Azure GET"))?;
        let mut buf = Vec::new();
        resp.into_reader().read_to_end(&mut buf).map_err(GalaxError::Io)?;
        Ok(buf)
    }

    fn list(&self) -> GalaxResult<Vec<String>> {
        let date = sign::now_rfc1123();
        let canonical_headers = format!("x-ms-date:{date}\nx-ms-version:{API_VERSION}\n");
        // Query params must be sorted in the canonicalized resource:
        // comp, prefix, restype.
        let canonical_resource = format!(
            "/{}/{}\ncomp:list\nprefix:{}\nrestype:container",
            self.account, self.container, self.prefix
        );
        let auth = self.shared_key_auth("GET", "", &canonical_headers, &canonical_resource)?;

        let url = format!(
            "https://{}/{}?restype=container&comp=list&prefix={}",
            self.host(),
            self.container,
            sign::uri_encode(&self.prefix, true)
        );
        let body = ureq::get(&url)
            .set("x-ms-date", &date)
            .set("x-ms-version", API_VERSION)
            .set("Authorization", &auth)
            .call()
            .map_err(map_ureq("Azure LIST"))?
            .into_string()
            .map_err(GalaxError::Io)?;
        Ok(parse_blob_names(&body, &self.prefix))
    }

    fn delete(&self, key: &str) -> GalaxResult<()> {
        let blob = self.blob_name(key);
        let date = sign::now_rfc1123();
        let canonical_headers = format!("x-ms-date:{date}\nx-ms-version:{API_VERSION}\n");
        let canonical_resource = format!("/{}/{}/{}", self.account, self.container, blob);
        let auth = self.shared_key_auth("DELETE", "", &canonical_headers, &canonical_resource)?;

        match ureq::delete(&self.blob_url(&blob))
            .set("x-ms-date", &date)
            .set("x-ms-version", API_VERSION)
            .set("Authorization", &auth)
            .call()
        {
            Ok(_) => Ok(()),
            Err(ureq::Error::Status(404, _)) => Ok(()),
            Err(e) => Err(map_ureq("Azure DELETE")(e)),
        }
    }

    fn scheme(&self) -> &'static str {
        "az"
    }
}

/// Extract `<Name>` element bodies from a Blob list XML response, stripping the
/// base prefix to return backup file names.
fn parse_blob_names(xml: &str, prefix: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = xml;
    while let Some(start) = rest.find("<Name>") {
        let after = &rest[start + 6..];
        let Some(end) = after.find("</Name>") else {
            break;
        };
        let full = &after[..end];
        let name = if prefix.is_empty() {
            full.to_string()
        } else {
            full.strip_prefix(&format!("{}/", prefix.trim_end_matches('/')))
                .unwrap_or(full)
                .to_string()
        };
        if !name.is_empty() {
            out.push(name);
        }
        rest = &after[end + 7..];
    }
    out
}

fn require_env(var: &str) -> GalaxResult<String> {
    std::env::var(var)
        .ok()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| GalaxError::Internal(format!("Azure backup requires the {var} environment variable")))
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

    fn store() -> AzureBlobObjectStore {
        AzureBlobObjectStore {
            account: "acct".into(),
            container: "cont".into(),
            prefix: "bk/db".into(),
            // base64("0123456789abcdef0123456789abcdef") — a valid 32-byte key.
            account_key_b64: base64::engine::general_purpose::STANDARD
                .encode(b"0123456789abcdef0123456789abcdef"),
        }
    }

    #[test]
    fn shared_key_auth_is_deterministic_and_prefixed() {
        let s = store();
        let ch = format!("x-ms-date:{}\nx-ms-version:{}\n", "Sun, 06 Nov 1994 08:49:37 GMT", API_VERSION);
        let cr = "/acct/cont/bk/db/wal.log".to_string();
        let a = s.shared_key_auth("GET", "", &ch, &cr).unwrap();
        let b = s.shared_key_auth("GET", "", &ch, &cr).unwrap();
        assert_eq!(a, b, "signing must be deterministic");
        assert!(a.starts_with("SharedKey acct:"), "got {a}");
    }

    #[test]
    fn parse_blob_names_strips_prefix() {
        let xml = "<EnumerationResults><Blobs>\
                   <Blob><Name>bk/db/wal.log</Name></Blob>\
                   <Blob><Name>bk/db/sst_5.pax</Name></Blob>\
                   </Blobs></EnumerationResults>";
        assert_eq!(
            parse_blob_names(xml, "bk/db"),
            vec!["wal.log".to_string(), "sst_5.pax".to_string()]
        );
    }

    #[test]
    fn blob_name_and_url() {
        let s = store();
        assert_eq!(s.blob_name("wal.log"), "bk/db/wal.log");
        assert_eq!(
            s.blob_url("bk/db/wal.log"),
            "https://acct.blob.core.windows.net/cont/bk/db/wal.log"
        );
    }
}
