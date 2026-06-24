//! Object-store abstraction for GalaxDB backup/restore.
//!
//! A backup is the engine's `wal.log` + `sst_*.pax` files (already flushed and
//! checksummed by the storage engine). This crate is the *transport*: it moves
//! those files between a local directory and an object store. The engine's
//! existing checksum validation and write-quiesce behaviour are preserved — the
//! [`ObjectStore`] only moves bytes.
//!
//! Implementations:
//! - [`LocalFsObjectStore`] — the bundled default (equivalent to a directory copy).
//! - [`s3::S3ObjectStore`] — AWS S3 / S3-compatible over REST with in-crate SigV4.
//! - [`gcs::GcsObjectStore`] — Google Cloud Storage over REST (OAuth2 bearer).
//! - [`azure::AzureBlobObjectStore`] — Azure Blob over REST (SharedKey).
//!
//! No cloud vendor SDK is linked — every cloud call is a hand-built REST request
//! signed in-crate (see `deny.toml`, which bans `aws-sdk-*` / `google-cloud-*` /
//! `azure_*`). Credentials are read from the environment and never logged.

use std::path::Path;

use galaxdb_common::{GalaxError, GalaxResult};

pub mod azure;
pub mod gcs;
pub mod s3;
mod sign;

/// A flat key/value object store rooted at some base location (a local
/// directory, or a bucket+prefix in the cloud). Keys are backup file names
/// such as `wal.log` or `sst_7.pax` — the implementation prepends its own
/// base prefix.
pub trait ObjectStore: Send + Sync {
    /// Store `data` at `key`, overwriting any existing object.
    fn put(&self, key: &str, data: &[u8]) -> GalaxResult<()>;
    /// Fetch the object at `key`.
    fn get(&self, key: &str) -> GalaxResult<Vec<u8>>;
    /// List object keys (file-name granularity, base prefix stripped).
    fn list(&self) -> GalaxResult<Vec<String>>;
    /// Delete the object at `key`. Missing objects are not an error.
    fn delete(&self, key: &str) -> GalaxResult<()>;
    /// Stable scheme label for logging (`file`, `s3`, `gs`, `az`).
    fn scheme(&self) -> &'static str;
}

/// The backup file names the engine owns. Anything else in the data
/// directory (reserve file, lock files) is intentionally not backed up.
fn is_backup_file(name: &str) -> bool {
    name == "wal.log" || (name.starts_with("sst_") && name.ends_with(".pax"))
}

/// Upload every backup file in `dir` to `store`. Returns the uploaded keys.
///
/// The caller is responsible for having produced a consistent on-disk set
/// first (the engine flushes its memtable before calling this), so this is a
/// pure transfer.
pub fn upload_dir(store: &dyn ObjectStore, dir: &Path) -> GalaxResult<Vec<String>> {
    let mut uploaded = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(GalaxError::Io)? {
        let entry = entry.map_err(GalaxError::Io)?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !is_backup_file(name) {
            continue;
        }
        let bytes = std::fs::read(&path).map_err(GalaxError::Io)?;
        store.put(name, &bytes)?;
        uploaded.push(name.to_string());
    }
    tracing::info!(scheme = store.scheme(), files = uploaded.len(), "backup uploaded to object store");
    Ok(uploaded)
}

/// Download every backup object from `store` into `dir`. Returns the
/// downloaded file names. `dir` is created if missing.
pub fn download_dir(store: &dyn ObjectStore, dir: &Path) -> GalaxResult<Vec<String>> {
    std::fs::create_dir_all(dir).map_err(GalaxError::Io)?;
    let mut downloaded = Vec::new();
    for key in store.list()? {
        if !is_backup_file(&key) {
            continue;
        }
        let bytes = store.get(&key)?;
        std::fs::write(dir.join(&key), &bytes).map_err(GalaxError::Io)?;
        downloaded.push(key);
    }
    tracing::info!(scheme = store.scheme(), files = downloaded.len(), "backup downloaded from object store");
    Ok(downloaded)
}

/// Returns `true` if `target` is a cloud object-store URL (not a local path).
pub fn is_object_store_url(target: &str) -> bool {
    target.starts_with("s3://") || target.starts_with("gs://") || target.starts_with("az://")
}

/// Build an [`ObjectStore`] for a backup target.
///
/// * `s3://bucket/prefix` — AWS S3 / S3-compatible (region + optional custom
///   endpoint from env; see [`s3::S3ObjectStore::from_url`]).
/// * `gs://bucket/prefix` — Google Cloud Storage.
/// * `az://container/prefix` — Azure Blob (account from env).
/// * anything else — a local filesystem directory.
///
/// Credentials are sourced from the environment by each implementation and are
/// never echoed back into the returned value's `Debug` or any error message.
pub fn object_store_for_target(target: &str) -> GalaxResult<Box<dyn ObjectStore>> {
    if let Some(rest) = target.strip_prefix("s3://") {
        Ok(Box::new(s3::S3ObjectStore::from_url(rest)?))
    } else if let Some(rest) = target.strip_prefix("gs://") {
        Ok(Box::new(gcs::GcsObjectStore::from_url(rest)?))
    } else if let Some(rest) = target.strip_prefix("az://") {
        Ok(Box::new(azure::AzureBlobObjectStore::from_url(rest)?))
    } else {
        Ok(Box::new(LocalFsObjectStore::new(target)))
    }
}

/// Split a `bucket/prefix/...` URL body into `(bucket, key_prefix)`.
/// The key prefix has no leading or trailing slash.
fn split_bucket_prefix(body: &str) -> (String, String) {
    let body = body.trim_start_matches('/');
    match body.split_once('/') {
        Some((bucket, prefix)) => (bucket.to_string(), prefix.trim_matches('/').to_string()),
        None => (body.to_string(), String::new()),
    }
}

/// Join a base prefix and a key with a single `/`, skipping empties.
fn join_key(prefix: &str, key: &str) -> String {
    match (prefix.is_empty(), key.is_empty()) {
        (true, _) => key.to_string(),
        (false, true) => prefix.to_string(),
        (false, false) => format!("{}/{}", prefix.trim_end_matches('/'), key),
    }
}

/// The bundled default object store: a local directory. Equivalent to the
/// engine's pre-existing directory-copy backup behaviour.
pub struct LocalFsObjectStore {
    root: std::path::PathBuf,
}

impl LocalFsObjectStore {
    /// Create a store rooted at `root` (created on first `put`).
    pub fn new(root: impl Into<std::path::PathBuf>) -> Self {
        Self { root: root.into() }
    }
}

impl ObjectStore for LocalFsObjectStore {
    fn put(&self, key: &str, data: &[u8]) -> GalaxResult<()> {
        std::fs::create_dir_all(&self.root).map_err(GalaxError::Io)?;
        std::fs::write(self.root.join(key), data).map_err(GalaxError::Io)
    }

    fn get(&self, key: &str) -> GalaxResult<Vec<u8>> {
        std::fs::read(self.root.join(key)).map_err(GalaxError::Io)
    }

    fn list(&self) -> GalaxResult<Vec<String>> {
        let mut out = Vec::new();
        if !self.root.exists() {
            return Ok(out);
        }
        for entry in std::fs::read_dir(&self.root).map_err(GalaxError::Io)? {
            let entry = entry.map_err(GalaxError::Io)?;
            if entry.path().is_file() {
                if let Some(name) = entry.file_name().to_str() {
                    out.push(name.to_string());
                }
            }
        }
        Ok(out)
    }

    fn delete(&self, key: &str) -> GalaxResult<()> {
        let path = self.root.join(key);
        if path.exists() {
            std::fs::remove_file(path).map_err(GalaxError::Io)?;
        }
        Ok(())
    }

    fn scheme(&self) -> &'static str {
        "file"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_store_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalFsObjectStore::new(dir.path().join("bk"));
        store.put("wal.log", b"hello").unwrap();
        store.put("sst_1.pax", b"world").unwrap();
        let mut keys = store.list().unwrap();
        keys.sort();
        assert_eq!(keys, vec!["sst_1.pax".to_string(), "wal.log".to_string()]);
        assert_eq!(store.get("wal.log").unwrap(), b"hello");
        store.delete("wal.log").unwrap();
        assert_eq!(store.list().unwrap(), vec!["sst_1.pax".to_string()]);
        // delete of a missing key is not an error
        store.delete("nope").unwrap();
    }

    #[test]
    fn upload_then_download_round_trips_backup_files() {
        let src = tempfile::tempdir().unwrap();
        std::fs::write(src.path().join("wal.log"), b"WAL").unwrap();
        std::fs::write(src.path().join("sst_3.pax"), b"SST").unwrap();
        std::fs::write(src.path().join("ignore.txt"), b"nope").unwrap();

        let store_dir = tempfile::tempdir().unwrap();
        let store = LocalFsObjectStore::new(store_dir.path());
        let mut up = upload_dir(&store, src.path()).unwrap();
        up.sort();
        assert_eq!(up, vec!["sst_3.pax".to_string(), "wal.log".to_string()]);

        let dst = tempfile::tempdir().unwrap();
        let mut down = download_dir(&store, dst.path()).unwrap();
        down.sort();
        assert_eq!(down, vec!["sst_3.pax".to_string(), "wal.log".to_string()]);
        assert_eq!(std::fs::read(dst.path().join("wal.log")).unwrap(), b"WAL");
        assert!(!dst.path().join("ignore.txt").exists());
    }

    #[test]
    fn url_routing() {
        assert!(is_object_store_url("s3://b/p"));
        assert!(is_object_store_url("gs://b/p"));
        assert!(is_object_store_url("az://c/p"));
        assert!(!is_object_store_url("/local/path"));
        assert!(!is_object_store_url("./rel"));
    }

    #[test]
    fn bucket_prefix_split() {
        assert_eq!(split_bucket_prefix("b/p/q"), ("b".into(), "p/q".into()));
        assert_eq!(split_bucket_prefix("b"), ("b".into(), "".into()));
        assert_eq!(split_bucket_prefix("/b/p/"), ("b".into(), "p".into()));
    }

    #[test]
    fn key_join() {
        assert_eq!(join_key("p", "wal.log"), "p/wal.log");
        assert_eq!(join_key("", "wal.log"), "wal.log");
        assert_eq!(join_key("p/", "wal.log"), "p/wal.log");
    }
}
