//! Encryption benchmarks — AES-256-GCM and AEGIS-256.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use galaxdb_crypto::{AegisTdeModule, LocalKeyProvider, NonceGenerator, TdeModule};

fn make_tde() -> TdeModule {
    let provider = LocalKeyProvider::from_key([0xABu8; 32]);
    TdeModule::new(Box::new(provider)).unwrap()
}

fn make_aegis() -> AegisTdeModule {
    let provider = LocalKeyProvider::from_key([0xABu8; 32]);
    AegisTdeModule::new(&provider).unwrap()
}

fn make_payload(size: usize) -> Vec<u8> {
    (0..size).map(|i| (i % 256) as u8).collect()
}

fn aes_encrypt_benchmarks(c: &mut Criterion) {
    let tde = make_tde();
    let sizes: &[(usize, &str)] = &[(1024, "1KB"), (65536, "64KB"), (1_048_576, "1MB")];

    let mut group = c.benchmark_group("aes256gcm_encrypt");
    for &(size, label) in sizes {
        let payload = make_payload(size);
        group.bench_with_input(BenchmarkId::new("encrypt", label), &payload, |b, data| {
            b.iter(|| { black_box(tde.encrypt(black_box(data)).unwrap()); });
        });
    }
    group.finish();
}

fn aes_decrypt_1mb(c: &mut Criterion) {
    let tde = make_tde();
    let encrypted = tde.encrypt(&make_payload(1_048_576)).unwrap();
    c.bench_function("aes256gcm_decrypt_1mb", |b| {
        b.iter(|| { black_box(tde.decrypt(black_box(&encrypted)).unwrap()); });
    });
}

fn aegis_encrypt_benchmarks(c: &mut Criterion) {
    let aegis = make_aegis();
    let sizes: &[(usize, &str)] = &[(1024, "1KB"), (65536, "64KB"), (1_048_576, "1MB")];

    let mut group = c.benchmark_group("aegis256_encrypt");
    for &(size, label) in sizes {
        let payload = make_payload(size);
        group.bench_with_input(BenchmarkId::new("encrypt", label), &payload, |b, data| {
            b.iter(|| { black_box(aegis.encrypt(black_box(data)).unwrap()); });
        });
    }
    group.finish();
}

fn aegis_decrypt_1mb(c: &mut Criterion) {
    let aegis = make_aegis();
    let encrypted = aegis.encrypt(&make_payload(1_048_576)).unwrap();
    c.bench_function("aegis256_decrypt_1mb", |b| {
        b.iter(|| { black_box(aegis.decrypt(black_box(&encrypted)).unwrap()); });
    });
}

fn nonce_generation(c: &mut Criterion) {
    let nonce_gen = NonceGenerator::new();
    c.bench_function("nonce_generation_throughput", |b| {
        b.iter(|| {
            for _ in 0..1000 {
                black_box(nonce_gen.next_nonce());
            }
        });
    });
}

criterion_group!(
    benches,
    aes_encrypt_benchmarks,
    aes_decrypt_1mb,
    aegis_encrypt_benchmarks,
    aegis_decrypt_1mb,
    nonce_generation,
);
criterion_main!(benches);
