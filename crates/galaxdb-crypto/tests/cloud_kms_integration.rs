//! Integration tests for the native cloud KMS providers (feature `cloud-kms`).
//!
//! No mocks: each test only runs when the matching cloud's credentials and a
//! test key are present in the environment, and then performs a real
//! wrap/unwrap round trip against the live service. When the environment is not
//! configured the test prints a skip message and returns early.
//!
//! ## AWS KMS
//! ```text
//! export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_REGION=us-east-1
//! export GALAXDB_KMS_TEST_KEY_ID=alias/galaxdb-test   # a symmetric KMS key
//! cargo test -p galaxdb-crypto --features cloud-kms --test cloud_kms_integration
//! ```
//!
//! ## GCP Cloud KMS
//! ```text
//! export GALAXDB_GCP_KMS_TOKEN=$(gcloud auth print-access-token)
//! export GALAXDB_GCP_KMS_TEST_KEY=projects/P/locations/L/keyRings/R/cryptoKeys/K
//! cargo test -p galaxdb-crypto --features cloud-kms --test cloud_kms_integration
//! ```
//!
//! ## Azure Key Vault (RSA key, wrapKey/unwrapKey)
//! ```text
//! export GALAXDB_AZURE_KV_TOKEN=$(az account get-access-token \
//!   --resource https://vault.azure.net --query accessToken -o tsv)
//! export GALAXDB_AZURE_KV_TEST_VAULT=myvault
//! export GALAXDB_AZURE_KV_TEST_KEY=mykey
//! cargo test -p galaxdb-crypto --features cloud-kms --test cloud_kms_integration
//! ```

#![cfg(feature = "cloud-kms")]

use galaxdb_crypto::cloud_kms::{
    AwsKmsKeyProvider, AzureKeyVaultKeyProvider, GcpKmsKeyProvider,
};
use galaxdb_crypto::key_provider::KeyProvider;

fn round_trip(provider: &dyn KeyProvider) {
    let (plaintext, encrypted) = provider
        .generate_data_key()
        .expect("cloud KMS generate_data_key");
    assert_eq!(plaintext.len(), 32, "DEK must be 32 bytes");
    assert_ne!(
        plaintext.as_slice(),
        encrypted.as_slice(),
        "wrapped blob must differ from the plaintext DEK"
    );
    let decrypted = provider
        .decrypt_data_key(&encrypted)
        .expect("cloud KMS decrypt_data_key");
    assert_eq!(decrypted, plaintext, "decrypt must recover the exact DEK");
}

#[test]
fn aws_kms_round_trip() {
    let Ok(key_id) = std::env::var("GALAXDB_KMS_TEST_KEY_ID") else {
        eprintln!("GALAXDB_KMS_TEST_KEY_ID not set; skipping aws_kms_round_trip");
        return;
    };
    if std::env::var("AWS_ACCESS_KEY_ID").is_err() {
        eprintln!("AWS_ACCESS_KEY_ID not set; skipping aws_kms_round_trip");
        return;
    }
    let provider = AwsKmsKeyProvider::from_key_id(&key_id).expect("construct AWS KMS provider");
    assert_eq!(provider.provider_name(), "aws-kms");
    round_trip(&provider);
}

#[test]
fn gcp_kms_round_trip() {
    let Ok(key_name) = std::env::var("GALAXDB_GCP_KMS_TEST_KEY") else {
        eprintln!("GALAXDB_GCP_KMS_TEST_KEY not set; skipping gcp_kms_round_trip");
        return;
    };
    let provider =
        GcpKmsKeyProvider::from_key_name(&key_name).expect("construct GCP KMS provider");
    assert_eq!(provider.provider_name(), "gcp-kms");
    round_trip(&provider);
}

#[test]
fn azure_kv_round_trip() {
    let (Ok(vault), Ok(key)) = (
        std::env::var("GALAXDB_AZURE_KV_TEST_VAULT"),
        std::env::var("GALAXDB_AZURE_KV_TEST_KEY"),
    ) else {
        eprintln!(
            "GALAXDB_AZURE_KV_TEST_VAULT / GALAXDB_AZURE_KV_TEST_KEY not set; \
             skipping azure_kv_round_trip"
        );
        return;
    };
    let provider =
        AzureKeyVaultKeyProvider::from_spec(&vault, &key).expect("construct Azure KV provider");
    assert_eq!(provider.provider_name(), "azure-kv");
    round_trip(&provider);
}
