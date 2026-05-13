//! Integration tests for [`HashicorpVaultKeyProvider`].
//!
//! These tests are gated behind the `vault` feature and also require a
//! live Vault server in dev mode at `$VAULT_ADDR` with `$VAULT_TOKEN`
//! set. Spin one up with:
//!
//! ```text
//! docker run --rm -d --name galaxdb-test-vault \
//!   -p 8200:8200 \
//!   -e VAULT_DEV_ROOT_TOKEN_ID=galaxdb-test \
//!   hashicorp/vault:latest
//! export VAULT_ADDR=http://127.0.0.1:8200
//! export VAULT_TOKEN=galaxdb-test
//! vault secrets enable transit
//! vault write -f transit/keys/galaxdb-test-key
//! cargo test -p galaxdb-crypto --features vault --test vault_integration
//! ```
//!
//! When the environment variables are missing the test prints a skip
//! message and returns early. No mocks — every run hits a real Vault
//! server doing real Transit encrypt / decrypt.

#![cfg(feature = "vault")]

use galaxdb_crypto::key_provider::{HashicorpVaultKeyProvider, KeyProvider};

fn have_vault_env() -> Option<(String, String)> {
    let address = std::env::var("VAULT_ADDR").ok()?;
    let token = std::env::var("VAULT_TOKEN").ok()?;
    Some((address, token))
}

#[test]
fn vault_transit_round_trip() {
    let Some((address, token)) = have_vault_env() else {
        eprintln!(
            "VAULT_ADDR or VAULT_TOKEN not set; skipping vault_transit_round_trip — \
             see tests/vault_integration.rs header for docker run command"
        );
        return;
    };

    let key_name = std::env::var("GALAXDB_VAULT_KEY_NAME")
        .unwrap_or_else(|_| "galaxdb-test-key".to_string());
    let mount = std::env::var("GALAXDB_VAULT_MOUNT")
        .unwrap_or_else(|_| "transit".to_string());

    let provider = HashicorpVaultKeyProvider::new(
        &address,
        &token,
        Some(&mount),
        &key_name,
    )
    .expect("construct vault provider");
    assert_eq!(provider.provider_name(), "hashicorp-vault");

    // Encrypt a DEK via Vault.
    let (plaintext, encrypted) = provider
        .generate_data_key()
        .expect("vault transit encrypt");

    assert_eq!(plaintext.len(), 32, "DEK must be 32 bytes");
    assert_ne!(plaintext.as_slice(), encrypted.as_slice());

    // The ciphertext field from Vault transit is a UTF-8 string
    // starting with "vault:v1:".
    let ct_str = std::str::from_utf8(&encrypted).expect("vault ciphertext is UTF-8");
    assert!(
        ct_str.starts_with("vault:v"),
        "expected vault:v<version>: prefix, got {ct_str}"
    );

    // Round-trip through Vault.
    let decrypted = provider
        .decrypt_data_key(&encrypted)
        .expect("vault transit decrypt");

    assert_eq!(decrypted, plaintext, "decrypt must recover the exact DEK");
}

#[test]
fn vault_from_env_matches_explicit() {
    let Some((address, token)) = have_vault_env() else {
        eprintln!(
            "VAULT_ADDR or VAULT_TOKEN not set; skipping vault_from_env_matches_explicit"
        );
        return;
    };
    let _ = (address, token); // already in env

    let key_name = std::env::var("GALAXDB_VAULT_KEY_NAME")
        .unwrap_or_else(|_| "galaxdb-test-key".to_string());

    // from_env() reads VAULT_ADDR + VAULT_TOKEN.
    let p = HashicorpVaultKeyProvider::from_env(&key_name, None)
        .expect("construct vault provider from env");

    let (plaintext, encrypted) = p.generate_data_key().expect("encrypt");
    let decrypted = p.decrypt_data_key(&encrypted).expect("decrypt");
    assert_eq!(plaintext, decrypted);
}
