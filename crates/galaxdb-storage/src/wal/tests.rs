//! Tests for the WAL module.

use std::io::Cursor;
use std::sync::Arc;
use std::time::Duration;

use super::record::{WalRecord, WalRecordType, WAL_RECORD_HEADER_SIZE};
use super::writer::{recover_wal, DurabilityMode, WalWriter, WalWriterConfig};

// ---------------------------------------------------------------------------
// Record serialization / deserialization round-trip
// ---------------------------------------------------------------------------

#[test]
fn record_type_from_u8_roundtrip() {
    assert_eq!(WalRecordType::from_u8(0x01), Some(WalRecordType::RowPut));
    assert_eq!(WalRecordType::from_u8(0x02), Some(WalRecordType::RowDelete));
    assert_eq!(WalRecordType::from_u8(0x03), Some(WalRecordType::DeltaInsert));
    assert_eq!(WalRecordType::from_u8(0x04), Some(WalRecordType::DeltaTombstone));
    assert_eq!(WalRecordType::from_u8(0x05), Some(WalRecordType::Checkpoint));
    assert_eq!(WalRecordType::from_u8(0x06), Some(WalRecordType::BlobRef));
    assert_eq!(WalRecordType::from_u8(0x00), None);
    assert_eq!(WalRecordType::from_u8(0xFF), None);
}

#[test]
fn record_serialize_deserialize_roundtrip() {
    let payloads: Vec<Vec<u8>> = vec![
        b"hello world".to_vec(),
        vec![0u8; 0],           // empty payload
        vec![42u8; 4096],       // larger payload
        (0..=255).collect(),    // all byte values
    ];

    let types = [
        WalRecordType::RowPut,
        WalRecordType::RowDelete,
        WalRecordType::DeltaInsert,
        WalRecordType::Checkpoint,
    ];

    for (i, (payload, record_type)) in payloads.iter().zip(types.iter().cycle()).enumerate() {
        let record = WalRecord::new(*record_type, i as u64 + 1, payload.clone());
        let serialized = record.serialize();

        // Verify header size is correct
        assert!(serialized.len() >= WAL_RECORD_HEADER_SIZE);

        // Deserialize and verify
        let mut cursor = Cursor::new(&serialized);
        let deserialized = WalRecord::deserialize(&mut cursor)
            .expect("deserialize should succeed")
            .expect("should not be EOF");

        assert_eq!(deserialized.record_type, *record_type);
        assert_eq!(deserialized.seq_no, i as u64 + 1);
        assert_eq!(deserialized.payload, *payload);
    }
}

#[test]
fn multiple_records_serialize_deserialize() {
    let records: Vec<WalRecord> = (1..=10)
        .map(|i| {
            WalRecord::new(
                WalRecordType::RowPut,
                i,
                format!("record-{}", i).into_bytes(),
            )
        })
        .collect();

    let mut buf = Vec::new();
    for record in &records {
        record.write_to(&mut buf).unwrap();
    }

    let mut cursor = Cursor::new(&buf);
    for expected in &records {
        let actual = WalRecord::deserialize(&mut cursor)
            .expect("deserialize should succeed")
            .expect("should not be EOF");
        assert_eq!(actual, *expected);
    }

    // Should be EOF now
    let eof = WalRecord::deserialize(&mut cursor).unwrap();
    assert!(eof.is_none());
}

#[test]
fn corrupt_checksum_detected() {
    let record = WalRecord::new(WalRecordType::RowPut, 1, b"test data".to_vec());
    let mut serialized = record.serialize();

    // Corrupt a byte in the checksum field (bytes 13..21)
    serialized[15] ^= 0xFF;

    let mut cursor = Cursor::new(&serialized);
    let result = WalRecord::deserialize(&mut cursor);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("checksum mismatch"));
}

#[test]
fn corrupt_payload_detected() {
    let record = WalRecord::new(WalRecordType::RowPut, 1, b"test data".to_vec());
    let mut serialized = record.serialize();

    // Corrupt a byte in the payload area (after the header)
    if serialized.len() > WAL_RECORD_HEADER_SIZE + 1 {
        serialized[WAL_RECORD_HEADER_SIZE + 1] ^= 0xFF;
    }

    let mut cursor = Cursor::new(&serialized);
    let result = WalRecord::deserialize(&mut cursor);
    assert!(result.is_err());
}

#[test]
fn eof_on_empty_reader() {
    let mut cursor = Cursor::new(Vec::<u8>::new());
    let result = WalRecord::deserialize(&mut cursor).unwrap();
    assert!(result.is_none());
}

// ---------------------------------------------------------------------------
// WAL writer integration tests (require tokio runtime)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn wal_write_read_roundtrip_strict() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");

    let config = WalWriterConfig {
        wal_path: wal_path.clone(),
        group_commit_interval: Duration::from_millis(10),
        checkpoint_size_bytes: 512 * 1024 * 1024,
        checkpoint_interval: Duration::from_secs(60),
    };

    let writer = WalWriter::new(config).unwrap();

    // Write several records in STRICT mode
    let seq1 = writer
        .append(WalRecordType::RowPut, b"row-1".to_vec(), DurabilityMode::Strict)
        .await
        .unwrap();
    let seq2 = writer
        .append(WalRecordType::RowDelete, b"row-2".to_vec(), DurabilityMode::Strict)
        .await
        .unwrap();
    let seq3 = writer
        .append(WalRecordType::DeltaInsert, b"delta-1".to_vec(), DurabilityMode::Strict)
        .await
        .unwrap();

    assert_eq!(seq1, 1);
    assert_eq!(seq2, 2);
    assert_eq!(seq3, 3);
    assert!(writer.current_size() > 0);

    writer.shutdown();

    // Recover and verify
    let (records, next_seq) = recover_wal(&wal_path).unwrap();
    assert_eq!(records.len(), 3);
    assert_eq!(next_seq, 4);

    assert_eq!(records[0].record_type, WalRecordType::RowPut);
    assert_eq!(records[0].payload, b"row-1");
    assert_eq!(records[1].record_type, WalRecordType::RowDelete);
    assert_eq!(records[1].payload, b"row-2");
    assert_eq!(records[2].record_type, WalRecordType::DeltaInsert);
    assert_eq!(records[2].payload, b"delta-1");
}

#[tokio::test]
async fn wal_write_read_roundtrip_relaxed() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");

    let config = WalWriterConfig {
        wal_path: wal_path.clone(),
        group_commit_interval: Duration::from_millis(5),
        checkpoint_size_bytes: 512 * 1024 * 1024,
        checkpoint_interval: Duration::from_secs(60),
    };

    let writer = WalWriter::new(config).unwrap();

    // Write records in RELAXED mode (group commit)
    let seq1 = writer
        .append(WalRecordType::RowPut, b"relaxed-1".to_vec(), DurabilityMode::Relaxed)
        .await
        .unwrap();
    let seq2 = writer
        .append(WalRecordType::RowPut, b"relaxed-2".to_vec(), DurabilityMode::Relaxed)
        .await
        .unwrap();
    let seq3 = writer
        .append(WalRecordType::BlobRef, b"blob-ref-1".to_vec(), DurabilityMode::Relaxed)
        .await
        .unwrap();

    assert_eq!(seq1, 1);
    assert_eq!(seq2, 2);
    assert_eq!(seq3, 3);

    writer.shutdown();

    // Recover and verify
    let (records, next_seq) = recover_wal(&wal_path).unwrap();
    assert_eq!(records.len(), 3);
    assert_eq!(next_seq, 4);
    assert_eq!(records[0].payload, b"relaxed-1");
    assert_eq!(records[1].payload, b"relaxed-2");
    assert_eq!(records[2].payload, b"blob-ref-1");
}

#[tokio::test]
async fn group_commit_batches_writes() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");

    let config = WalWriterConfig {
        wal_path: wal_path.clone(),
        group_commit_interval: Duration::from_millis(50),
        checkpoint_size_bytes: 512 * 1024 * 1024,
        checkpoint_interval: Duration::from_secs(60),
    };

    let writer = Arc::new(WalWriter::new(config).unwrap());

    // Spawn multiple concurrent writes — they should be batched
    let mut handles = Vec::new();
    for i in 0..10 {
        let w = writer.clone();
        handles.push(tokio::spawn(async move {
            w.append(
                WalRecordType::RowPut,
                format!("batch-{}", i).into_bytes(),
                DurabilityMode::Relaxed,
            )
            .await
            .unwrap()
        }));
    }

    let mut seq_nos = Vec::new();
    for handle in handles {
        seq_nos.push(handle.await.unwrap());
    }

    // All sequence numbers should be unique
    let mut sorted = seq_nos.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), 10);

    writer.shutdown();

    // Verify all records are recoverable
    let (records, _) = recover_wal(&wal_path).unwrap();
    assert_eq!(records.len(), 10);
}

#[tokio::test]
async fn checkpoint_trigger_by_size() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");

    // Set a very small checkpoint size threshold
    let config = WalWriterConfig {
        wal_path: wal_path.clone(),
        group_commit_interval: Duration::from_millis(5),
        checkpoint_size_bytes: 100, // Very small — will trigger quickly
        checkpoint_interval: Duration::from_secs(3600), // Don't trigger by time
    };

    let writer = WalWriter::new(config).unwrap();

    // Write enough data to exceed the threshold
    writer
        .append(WalRecordType::RowPut, vec![0u8; 200], DurabilityMode::Strict)
        .await
        .unwrap();

    assert!(writer.should_checkpoint().await);

    // Write checkpoint
    let cp_seq = writer.write_checkpoint().await.unwrap();
    assert!(cp_seq > 0);

    // Write more records after checkpoint
    writer
        .append(WalRecordType::RowPut, b"after-cp".to_vec(), DurabilityMode::Strict)
        .await
        .unwrap();

    writer.shutdown();

    // Recovery should only return records after the checkpoint
    let (records, _) = recover_wal(&wal_path).unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].payload, b"after-cp");
}

#[tokio::test]
async fn checkpoint_trigger_by_time() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");

    let config = WalWriterConfig {
        wal_path: wal_path.clone(),
        group_commit_interval: Duration::from_millis(5),
        checkpoint_size_bytes: u64::MAX, // Don't trigger by size
        checkpoint_interval: Duration::from_millis(50), // Very short interval
    };

    let writer = WalWriter::new(config).unwrap();

    // Write a record so there's data
    writer
        .append(WalRecordType::RowPut, b"data".to_vec(), DurabilityMode::Strict)
        .await
        .unwrap();

    // Initially should_checkpoint is true because there's no checkpoint yet
    assert!(writer.should_checkpoint().await);

    // Write checkpoint
    writer.write_checkpoint().await.unwrap();

    // Right after checkpoint, time hasn't elapsed
    assert!(!writer.should_checkpoint().await);

    // Wait for the interval to pass
    tokio::time::sleep(Duration::from_millis(60)).await;

    // Now it should trigger
    assert!(writer.should_checkpoint().await);

    writer.shutdown();
}

#[tokio::test]
async fn truncate_to_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");

    let config = WalWriterConfig {
        wal_path: wal_path.clone(),
        group_commit_interval: Duration::from_millis(5),
        checkpoint_size_bytes: 512 * 1024 * 1024,
        checkpoint_interval: Duration::from_secs(60),
    };

    let writer = WalWriter::new(config).unwrap();

    // Write some records before checkpoint
    for i in 0..5 {
        writer
            .append(
                WalRecordType::RowPut,
                format!("before-{}", i).into_bytes(),
                DurabilityMode::Strict,
            )
            .await
            .unwrap();
    }

    let size_before = writer.current_size();

    // Write checkpoint
    writer.write_checkpoint().await.unwrap();

    // Write records after checkpoint
    for i in 0..3 {
        writer
            .append(
                WalRecordType::RowPut,
                format!("after-{}", i).into_bytes(),
                DurabilityMode::Strict,
            )
            .await
            .unwrap();
    }

    // Truncate
    writer.truncate_to_checkpoint().await.unwrap();

    let size_after = writer.current_size();
    assert!(size_after < size_before);

    writer.shutdown();

    // Recovery should find the checkpoint + 3 records after it
    let (records, _) = recover_wal(&wal_path).unwrap();
    assert_eq!(records.len(), 3);
    assert_eq!(records[0].payload, b"after-0");
    assert_eq!(records[1].payload, b"after-1");
    assert_eq!(records[2].payload, b"after-2");
}

#[tokio::test]
async fn recovery_with_corrupt_records() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");

    let config = WalWriterConfig {
        wal_path: wal_path.clone(),
        group_commit_interval: Duration::from_millis(5),
        checkpoint_size_bytes: 512 * 1024 * 1024,
        checkpoint_interval: Duration::from_secs(60),
    };

    let writer = WalWriter::new(config).unwrap();

    // Write 5 valid records
    for i in 0..5 {
        writer
            .append(
                WalRecordType::RowPut,
                format!("record-{}", i).into_bytes(),
                DurabilityMode::Strict,
            )
            .await
            .unwrap();
    }

    writer.shutdown();

    // Corrupt the file: flip a byte in the middle of the 3rd record
    // First, figure out where records are by reading them
    let file_data = std::fs::read(&wal_path).unwrap();

    // Find the approximate offset of the 3rd record by reading first two
    let mut cursor = Cursor::new(&file_data);
    let r1 = WalRecord::deserialize(&mut cursor).unwrap().unwrap();
    let r2 = WalRecord::deserialize(&mut cursor).unwrap().unwrap();
    let offset_of_third = cursor.position() as usize;

    // Corrupt the checksum of the 3rd record (bytes 13..21 relative to record start)
    let mut corrupted = file_data.clone();
    corrupted[offset_of_third + 15] ^= 0xFF;
    std::fs::write(&wal_path, &corrupted).unwrap();

    // Recovery should return only the first 2 records (stops at first checksum failure)
    let (records, next_seq) = recover_wal(&wal_path).unwrap();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].payload, b"record-0".to_vec());
    assert_eq!(records[1].payload, b"record-1".to_vec());
    // next_seq should be based on the max seq_no seen (which is 2)
    assert_eq!(next_seq, 3);

    // Suppress unused variable warnings
    let _ = r1;
    let _ = r2;
}

#[tokio::test]
async fn recovery_from_nonexistent_file() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("nonexistent.wal");

    let (records, next_seq) = recover_wal(&wal_path).unwrap();
    assert!(records.is_empty());
    assert_eq!(next_seq, 1);
}

#[tokio::test]
async fn recovery_from_empty_file() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("empty.wal");
    std::fs::write(&wal_path, []).unwrap();

    let (records, next_seq) = recover_wal(&wal_path).unwrap();
    assert!(records.is_empty());
    assert_eq!(next_seq, 1);
}

#[tokio::test]
async fn recovery_replays_from_last_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");

    let config = WalWriterConfig {
        wal_path: wal_path.clone(),
        group_commit_interval: Duration::from_millis(5),
        checkpoint_size_bytes: 512 * 1024 * 1024,
        checkpoint_interval: Duration::from_secs(60),
    };

    let writer = WalWriter::new(config).unwrap();

    // Write records, checkpoint, more records, checkpoint, more records
    for i in 0..3 {
        writer
            .append(
                WalRecordType::RowPut,
                format!("phase1-{}", i).into_bytes(),
                DurabilityMode::Strict,
            )
            .await
            .unwrap();
    }

    writer.write_checkpoint().await.unwrap();

    for i in 0..2 {
        writer
            .append(
                WalRecordType::RowPut,
                format!("phase2-{}", i).into_bytes(),
                DurabilityMode::Strict,
            )
            .await
            .unwrap();
    }

    writer.write_checkpoint().await.unwrap();

    // Records after the last checkpoint
    writer
        .append(
            WalRecordType::RowPut,
            b"final-record".to_vec(),
            DurabilityMode::Strict,
        )
        .await
        .unwrap();

    writer.shutdown();

    // Recovery should only return the record after the LAST checkpoint
    let (records, _) = recover_wal(&wal_path).unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].payload, b"final-record");
}

#[tokio::test]
async fn mixed_durability_modes() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");

    let config = WalWriterConfig {
        wal_path: wal_path.clone(),
        group_commit_interval: Duration::from_millis(5),
        checkpoint_size_bytes: 512 * 1024 * 1024,
        checkpoint_interval: Duration::from_secs(60),
    };

    let writer = WalWriter::new(config).unwrap();

    // Mix STRICT and RELAXED writes
    writer
        .append(WalRecordType::RowPut, b"strict-1".to_vec(), DurabilityMode::Strict)
        .await
        .unwrap();
    writer
        .append(WalRecordType::RowPut, b"relaxed-1".to_vec(), DurabilityMode::Relaxed)
        .await
        .unwrap();
    writer
        .append(WalRecordType::RowDelete, b"strict-2".to_vec(), DurabilityMode::Strict)
        .await
        .unwrap();
    writer
        .append(WalRecordType::DeltaInsert, b"relaxed-2".to_vec(), DurabilityMode::Relaxed)
        .await
        .unwrap();

    writer.shutdown();

    // All records should be recoverable regardless of durability mode
    let (records, _) = recover_wal(&wal_path).unwrap();
    assert_eq!(records.len(), 4);
}

#[tokio::test]
async fn all_record_types_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");

    let config = WalWriterConfig {
        wal_path: wal_path.clone(),
        group_commit_interval: Duration::from_millis(5),
        checkpoint_size_bytes: 512 * 1024 * 1024,
        checkpoint_interval: Duration::from_secs(60),
    };

    let writer = WalWriter::new(config).unwrap();

    let types_and_payloads = [
        (WalRecordType::RowPut, b"put-data".to_vec()),
        (WalRecordType::RowDelete, b"delete-key".to_vec()),
        (WalRecordType::DeltaInsert, b"vector-data".to_vec()),
        (WalRecordType::DeltaTombstone, b"vector-delete".to_vec()),
        (WalRecordType::BlobRef, b"blob-reference".to_vec()),
    ];

    for (rt, payload) in &types_and_payloads {
        writer
            .append(*rt, payload.clone(), DurabilityMode::Strict)
            .await
            .unwrap();
    }

    writer.shutdown();

    let (records, _) = recover_wal(&wal_path).unwrap();
    assert_eq!(records.len(), 5);

    for (i, (expected_type, expected_payload)) in types_and_payloads.iter().enumerate() {
        assert_eq!(records[i].record_type, *expected_type);
        assert_eq!(records[i].payload, *expected_payload);
    }
}

#[tokio::test]
async fn recovery_time_under_30_seconds() {
    use std::time::Instant;

    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");

    let config = WalWriterConfig {
        wal_path: wal_path.clone(),
        group_commit_interval: Duration::from_millis(5),
        checkpoint_size_bytes: 512 * 1024 * 1024,
        checkpoint_interval: Duration::from_secs(60),
    };

    let writer = WalWriter::new(config).unwrap();

    // Write a substantial number of records to test recovery performance
    let num_records = 10_000;
    let payload = vec![0u8; 256]; // 256-byte payloads

    for _ in 0..num_records {
        writer
            .append(WalRecordType::RowPut, payload.clone(), DurabilityMode::Strict)
            .await
            .unwrap();
    }

    writer.shutdown();

    // Time the recovery
    let start = Instant::now();
    let (records, _) = recover_wal(&wal_path).unwrap();
    let elapsed = start.elapsed();

    assert_eq!(records.len(), num_records);
    assert!(
        elapsed < Duration::from_secs(30),
        "Recovery took {:?}, which exceeds the 30-second target",
        elapsed
    );
}
