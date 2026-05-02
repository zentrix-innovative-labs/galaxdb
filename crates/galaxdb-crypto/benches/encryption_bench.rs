//! AES-256-GCM encryption benchmarks.
//!
//! Measures encrypt/decrypt throughput at various block sizes and nonce
//! generation throughput.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use galaxdb_crypto::{LocalKeyProvider, NonceGenerator, TdeModule};

fn make_tde() -> TdeModule {
    let provider = LocalKeyProvider::from_key([0xABu8; 32]);
    TdeModule::new(Box::new(provider)).unwrap()
}

/// Generate a deterministic payload of the given size.
fn make_payload(size: usize) -> Vec<u8> {
    (0..size).map(|i| (i % 256) as u8).collect()
}

fn encrypt_benchmarks(c: &mut Criterion) {
    let tde = make_tde();

    let sizes: &[(usize, &str)] = &[
        (1024, "1KB"),
        (65536, "64KB"),
        (1_048_576, "1MB"),
    ];

    let mut group = c.benchmark_group("aes256gcm_encrypt");
    for &(size, label) in sizes {
        let payload = make_payload(size);
        group.bench_with_input(BenchmarkId::new("encrypt", label), &payload, |b, data| {
            b.iter(|| {
                let encrypted = tde.encrypt(black_box(data)).unwrap();
                black_box(encrypted);
            });
        });
    }
    group.finish();
}

fn decrypt_1mb(c: &mut Criterion) {
    let tde = make_tde();
    let payload = make_payload(1_048_576);
    let encrypted = tde.encrypt(&payload).unwrap();

    c.bench_function("aes256gcm_decrypt_1mb", |b| {
        b.iter(|| {
            let decrypted = tde.decrypt(black_box(&encrypted)).unwrap();
            black_box(decrypted);
        });
    });
}

fn nonce_generation(c: &mut Criterion) {
    let nonce_gen = NonceGenerator::new();

    c.bench_function("nonce_generation_throughput", |b| {
        b.iter(|| {
            for _ in 0..1000 {
                let nonce = nonce_gen.next_nonce();
                black_box(nonce);
            }
        });
    });
}

criterion_group!(benches, encrypt_benchmarks, decrypt_1mb, nonce_generation);
criterion_main!(benches);
