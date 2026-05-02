//! WAL record serialize/deserialize and write throughput benchmarks.
//!
//! Measures single-record serialization, deserialization, 10K-record write
//! throughput in STRICT and RELAXED modes, and WAL recovery replay.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use galaxdb_storage::wal::{
    recover_wal, DurabilityMode, WalRecord, WalRecordType, WalWriter, WalWriterConfig,
};
use std::io::Cursor;
use std::time::Duration;

/// Number of records for write throughput benchmarks.
/// Kept at 10K for relaxed/recovery, reduced for strict (fsync-per-record).
const STRICT_RECORD_COUNT: u64 = 1_000;
const RELAXED_RECORD_COUNT: u64 = 10_000;

/// Build a realistic 256-byte payload.
fn make_payload(seq: u64) -> Vec<u8> {
    let mut payload = vec![0u8; 256];
    let seed = seq.to_le_bytes();
    for (i, byte) in payload.iter_mut().enumerate() {
        *byte = seed[i % 8] ^ (i as u8);
    }
    payload
}

fn wal_record_serialize(c: &mut Criterion) {
    let record = WalRecord::new(WalRecordType::RowPut, 42, make_payload(42));

    c.bench_function("wal_record_serialize_256b", |b| {
        b.iter(|| {
            let bytes = black_box(&record).serialize();
            black_box(bytes);
        });
    });
}

fn wal_record_deserialize(c: &mut Criterion) {
    let record = WalRecord::new(WalRecordType::RowPut, 42, make_payload(42));
    let serialized = record.serialize();

    c.bench_function("wal_record_deserialize_256b", |b| {
        b.iter(|| {
            let mut cursor = Cursor::new(black_box(serialized.as_slice()));
            let rec = WalRecord::deserialize(&mut cursor).unwrap().unwrap();
            black_box(rec);
        });
    });
}

fn wal_write_strict(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    // Pre-generate payloads outside the benchmark loop
    let payloads: Vec<Vec<u8>> = (0..STRICT_RECORD_COUNT).map(make_payload).collect();

    let mut group = c.benchmark_group("wal_write_strict");
    group.sample_size(10);
    // Each iteration does STRICT_RECORD_COUNT fsyncs. On macOS, each fsync
    // takes ~1-6ms, so 1K records ≈ 1-6s per iteration. 60s measurement
    // time gives criterion enough room for 10 samples.
    group.measurement_time(Duration::from_secs(120));
    group.warm_up_time(Duration::from_secs(5));

    group.bench_function(
        format!("wal_write_{}_strict", STRICT_RECORD_COUNT),
        |b| {
            b.iter(|| {
                rt.block_on(async {
                    let dir = tempfile::tempdir().unwrap();
                    let config = WalWriterConfig {
                        wal_path: dir.path().join("bench_strict.wal"),
                        group_commit_interval: Duration::from_millis(10),
                        checkpoint_size_bytes: u64::MAX,
                        checkpoint_interval: Duration::from_secs(3600),
                    };
                    let writer = WalWriter::new(config).unwrap();

                    for payload in &payloads {
                        writer
                            .append(
                                WalRecordType::RowPut,
                                payload.clone(),
                                DurabilityMode::Strict,
                            )
                            .await
                            .unwrap();
                    }

                    writer.shutdown();
                    black_box(writer.current_size());
                });
            });
        },
    );

    group.finish();
}

fn wal_write_relaxed(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();

    // Pre-generate payloads
    let payloads: Vec<Vec<u8>> = (0..RELAXED_RECORD_COUNT).map(make_payload).collect();

    let mut group = c.benchmark_group("wal_write_relaxed");
    group.sample_size(10);
    // Relaxed mode batches writes. With a 2ms interval and 10K records,
    // each iteration should take a few seconds.
    group.measurement_time(Duration::from_secs(120));
    group.warm_up_time(Duration::from_secs(5));

    group.bench_function(
        format!("wal_write_{}_relaxed_group_commit", RELAXED_RECORD_COUNT),
        |b| {
            b.iter(|| {
                rt.block_on(async {
                    let dir = tempfile::tempdir().unwrap();
                    let config = WalWriterConfig {
                        wal_path: dir.path().join("bench_relaxed.wal"),
                        // Shorter interval for better batching throughput
                        group_commit_interval: Duration::from_millis(2),
                        checkpoint_size_bytes: u64::MAX,
                        checkpoint_interval: Duration::from_secs(3600),
                    };
                    let writer = WalWriter::new(config).unwrap();

                    // Fire all appends concurrently to maximize group commit batching
                    let mut handles = Vec::with_capacity(payloads.len());
                    for payload in &payloads {
                        let payload = payload.clone();
                        let w = &writer;
                        handles.push(async move {
                            w.append(WalRecordType::RowPut, payload, DurabilityMode::Relaxed)
                                .await
                                .unwrap()
                        });
                    }

                    // Await all in order (they're already submitted to the channel)
                    for handle in handles {
                        handle.await;
                    }

                    writer.shutdown();
                    black_box(writer.current_size());
                });
            });
        },
    );

    group.finish();
}

fn wal_recovery(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    // Setup: write 10K records to a WAL file once (using strict mode for simplicity)
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("bench_recovery.wal");

    rt.block_on(async {
        let config = WalWriterConfig {
            wal_path: wal_path.clone(),
            group_commit_interval: Duration::from_millis(5),
            checkpoint_size_bytes: u64::MAX,
            checkpoint_interval: Duration::from_secs(3600),
        };
        let writer = WalWriter::new(config).unwrap();

        for i in 0..10_000u64 {
            writer
                .append(
                    WalRecordType::RowPut,
                    make_payload(i),
                    DurabilityMode::Strict,
                )
                .await
                .unwrap();
        }

        writer.shutdown();
    });

    let mut group = c.benchmark_group("wal_recovery");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));

    // Benchmark: replay the WAL (pure CPU + sequential read, no fsync)
    group.bench_function("wal_recovery_10k_records", |b| {
        b.iter(|| {
            let (records, next_seq) = recover_wal(black_box(&wal_path)).unwrap();
            black_box(records.len());
            black_box(next_seq);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    wal_record_serialize,
    wal_record_deserialize,
    wal_write_strict,
    wal_write_relaxed,
    wal_recovery,
);
criterion_main!(benches);
