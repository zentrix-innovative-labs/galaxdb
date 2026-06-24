//! Credential-gated integration tests for the cloud object stores.
//!
//! Each test runs a real put → list → get → delete round-trip against the live
//! service ONLY when the required environment variables are present; otherwise
//! it prints a skip line and returns (the same pattern as the Vault KMS test).
//! These exercise the real REST + signing paths — there are no mocks.
//!
//! To run S3 (or an S3-compatible store like MinIO):
//!   export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
//!   export GALAXDB_S3_TEST_BUCKET=my-bucket           # required to enable
//!   export GALAXDB_S3_REGION=us-east-1                # optional
//!   export GALAXDB_S3_ENDPOINT=https://minio:9000     # optional (S3-compatible)
//!   cargo test -p galaxdb-backup --test cloud_integration -- --nocapture
//!
//! GCS:   GALAXDB_GCS_TEST_BUCKET + GALAXDB_GCS_ACCESS_TOKEN
//! Azure: GALAXDB_AZURE_TEST_CONTAINER + AZURE_STORAGE_ACCOUNT + AZURE_STORAGE_KEY

use galaxdb_backup::{object_store_for_target, ObjectStore};

fn unique_prefix() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("galaxdb-backup-test/{nanos}")
}

/// Run a full put/list/get/delete round-trip against a configured store.
fn round_trip(store: &dyn ObjectStore) {
    store.put("wal.log", b"hello-wal").expect("put wal.log");
    store.put("sst_1.pax", b"sst-bytes").expect("put sst_1.pax");

    let mut keys = store.list().expect("list");
    keys.sort();
    assert!(
        keys.contains(&"wal.log".to_string()) && keys.contains(&"sst_1.pax".to_string()),
        "list must return both uploaded objects, got {keys:?}"
    );

    assert_eq!(store.get("wal.log").expect("get wal.log"), b"hello-wal");
    assert_eq!(store.get("sst_1.pax").expect("get sst_1.pax"), b"sst-bytes");

    store.delete("wal.log").expect("delete wal.log");
    store.delete("sst_1.pax").expect("delete sst_1.pax");
}

#[test]
fn s3_round_trip_when_configured() {
    let Ok(bucket) = std::env::var("GALAXDB_S3_TEST_BUCKET") else {
        eprintln!("SKIP s3_round_trip: set GALAXDB_S3_TEST_BUCKET + AWS creds to run");
        return;
    };
    if std::env::var("AWS_ACCESS_KEY_ID").is_err() {
        eprintln!("SKIP s3_round_trip: AWS_ACCESS_KEY_ID not set");
        return;
    }
    let target = format!("s3://{bucket}/{}", unique_prefix());
    let store = object_store_for_target(&target).expect("build S3 store");
    assert_eq!(store.scheme(), "s3");
    round_trip(store.as_ref());
}

#[test]
fn gcs_round_trip_when_configured() {
    let Ok(bucket) = std::env::var("GALAXDB_GCS_TEST_BUCKET") else {
        eprintln!("SKIP gcs_round_trip: set GALAXDB_GCS_TEST_BUCKET + GALAXDB_GCS_ACCESS_TOKEN to run");
        return;
    };
    if std::env::var("GALAXDB_GCS_ACCESS_TOKEN").is_err() {
        eprintln!("SKIP gcs_round_trip: GALAXDB_GCS_ACCESS_TOKEN not set");
        return;
    }
    let target = format!("gs://{bucket}/{}", unique_prefix());
    let store = object_store_for_target(&target).expect("build GCS store");
    assert_eq!(store.scheme(), "gs");
    round_trip(store.as_ref());
}

#[test]
fn azure_round_trip_when_configured() {
    let Ok(container) = std::env::var("GALAXDB_AZURE_TEST_CONTAINER") else {
        eprintln!("SKIP azure_round_trip: set GALAXDB_AZURE_TEST_CONTAINER + AZURE_STORAGE_ACCOUNT/KEY to run");
        return;
    };
    if std::env::var("AZURE_STORAGE_ACCOUNT").is_err() || std::env::var("AZURE_STORAGE_KEY").is_err() {
        eprintln!("SKIP azure_round_trip: AZURE_STORAGE_ACCOUNT / AZURE_STORAGE_KEY not set");
        return;
    }
    let target = format!("az://{container}/{}", unique_prefix());
    let store = object_store_for_target(&target).expect("build Azure store");
    assert_eq!(store.scheme(), "az");
    round_trip(store.as_ref());
}
