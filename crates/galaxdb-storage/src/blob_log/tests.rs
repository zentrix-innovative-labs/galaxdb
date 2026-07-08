//! Tests for the blob log module.

use super::*;
use tempfile::TempDir;

/// Helper: create a BlobLog in a temp directory with default settings.
fn create_test_blob_log() -> (BlobLog, TempDir) {
    let dir = TempDir::new().unwrap();
    let blob_dir = dir.path().join("blobs");
    let blob_log = BlobLog::with_defaults(blob_dir).unwrap();
    (blob_log, dir)
}

/// Helper: create a BlobLog with custom settings.
fn create_blob_log_with(num_queues: usize, threshold: usize) -> (BlobLog, TempDir) {
    let dir = TempDir::new().unwrap();
    let blob_dir = dir.path().join("blobs");
    let blob_log = BlobLog::new(blob_dir, num_queues, threshold).unwrap();
    (blob_log, dir)
}

/// Helper: generate a large value that exceeds the default threshold.
fn large_value(size: usize) -> Vec<u8> {
    (0..size).map(|i| (i % 256) as u8).collect()
}

// ── Sub-task 11.1: BlobLog with multi-queue parallel writers ──

#[test]
fn blob_log_creates_with_default_4_queues() {
    let (blob_log, _dir) = create_test_blob_log();
    assert_eq!(blob_log.num_queues(), DEFAULT_NUM_QUEUES);
    assert_eq!(blob_log.threshold(), DEFAULT_BLOB_THRESHOLD);
}

#[test]
fn blob_log_creates_with_custom_queue_count() {
    let (blob_log, _dir) = create_blob_log_with(8, 512);
    assert_eq!(blob_log.num_queues(), 8);
    assert_eq!(blob_log.threshold(), 512);
}

#[test]
fn blob_log_zero_queues_defaults_to_4() {
    let (blob_log, _dir) = create_blob_log_with(0, 1024);
    assert_eq!(blob_log.num_queues(), DEFAULT_NUM_QUEUES);
}

#[test]
fn blob_log_distributes_writes_across_queues() {
    let (blob_log, _dir) = create_test_blob_log();

    // Write multiple values — they should be distributed across files
    for i in 0..8 {
        let value = large_value(2048 + i);
        let (_hash, blob_ref) = blob_log.write(&value).unwrap();
        // With 4 queues and round-robin, file_ids should cycle 0,1,2,3,0,1,2,3
        assert_eq!(blob_ref.file_id, (i as u64) % 4);
    }
}

// ── Sub-task 11.2: WAL-time separation ──

#[test]
fn should_separate_respects_threshold() {
    assert!(!should_separate(&[0u8; 1024], DEFAULT_BLOB_THRESHOLD));
    assert!(should_separate(&[0u8; 1025], DEFAULT_BLOB_THRESHOLD));
    assert!(!should_separate(&[0u8; 100], DEFAULT_BLOB_THRESHOLD));
}

#[test]
fn blob_log_should_separate_method() {
    let (blob_log, _dir) = create_test_blob_log();
    assert!(!blob_log.should_separate(&[0u8; 1024]));
    assert!(blob_log.should_separate(&[0u8; 1025]));
}

#[test]
fn content_hash_is_deterministic() {
    let data = b"hello world blob data";
    let h1 = content_hash(data);
    let h2 = content_hash(data);
    assert_eq!(h1, h2);
}

#[test]
fn content_hash_differs_for_different_data() {
    let h1 = content_hash(b"data one");
    let h2 = content_hash(b"data two");
    assert_ne!(h1, h2);
}

#[test]
fn blob_ref_serialization_roundtrip() {
    let blob_ref = BlobRef {
        file_id: 42,
        offset: 12345,
        length: 9876,
    };
    let bytes = blob_ref.to_bytes();
    let decoded = BlobRef::from_bytes(&bytes).unwrap();
    assert_eq!(blob_ref, decoded);
}

#[test]
fn encode_decode_blob_ref_roundtrip() {
    let hash = content_hash(b"test data");
    let blob_ref = BlobRef {
        file_id: 1,
        offset: 100,
        length: 500,
    };
    let encoded = encode_blob_ref(&hash, &blob_ref);
    assert_eq!(encoded.len(), BLOB_MARKER_SIZE);

    let (decoded_hash, decoded_ref) = decode_blob_ref(&encoded).unwrap();
    assert_eq!(decoded_hash, hash);
    assert_eq!(decoded_ref, blob_ref);
}

#[test]
fn decode_blob_ref_returns_none_for_inline_values() {
    // Short value
    assert!(decode_blob_ref(b"short").is_none());
    // Wrong length
    assert!(decode_blob_ref(&[0u8; 100]).is_none());
    // Right length but wrong tag
    let mut fake = vec![0x00; BLOB_MARKER_SIZE];
    fake[0] = 0x00; // not BLOB_REF_TAG
    assert!(decode_blob_ref(&fake).is_none());
}

#[test]
fn write_and_read_large_value() {
    let (blob_log, _dir) = create_test_blob_log();
    let value = large_value(4096);

    let (hash, blob_ref) = blob_log.write(&value).unwrap();
    assert_eq!(blob_ref.length, 4096);

    // Read it back
    let read_value = blob_log.read(&blob_ref).unwrap();
    assert_eq!(read_value, value);

    // Verify content hash matches
    assert_eq!(content_hash(&value), hash);
}

#[test]
fn write_deduplicates_identical_values() {
    let (blob_log, _dir) = create_test_blob_log();
    let value = large_value(2048);

    let (hash1, ref1) = blob_log.write(&value).unwrap();
    let (hash2, ref2) = blob_log.write(&value).unwrap();

    // Same content hash
    assert_eq!(hash1, hash2);
    // Same blob ref (deduplicated)
    assert_eq!(ref1, ref2);
    // Only one entry in the index
    assert_eq!(blob_log.index_size(), 1);
}

// ── Sub-task 11.3: Transparent blob fetch ──

#[test]
fn read_transparent_fetches_blob_value() {
    let (blob_log, _dir) = create_test_blob_log();
    let value = large_value(2048);

    let (hash, blob_ref) = blob_log.write(&value).unwrap();
    let encoded = encode_blob_ref(&hash, &blob_ref);

    // Transparent read should fetch the actual value
    let result = blob_log.read_transparent(&encoded).unwrap();
    assert_eq!(result, value);
}

#[test]
fn read_transparent_returns_inline_value_as_is() {
    let (blob_log, _dir) = create_test_blob_log();
    let inline_value = b"small inline value";

    let result = blob_log.read_transparent(inline_value).unwrap();
    assert_eq!(result, inline_value);
}

// ── Sub-task 11.4: Blob GC ──

#[test]
fn gc_compacts_files_with_high_discard_ratio() {
    let (blob_log, _dir) = create_blob_log_with(1, 100); // 1 queue, low threshold

    // Write several values
    let mut refs = Vec::new();
    for i in 0..10 {
        let value = large_value(200 + i);
        let (_hash, blob_ref) = blob_log.write(&value).unwrap();
        refs.push(blob_ref);
    }

    // Mark most as dead (keep only 2 alive)
    for blob_ref in &refs[2..] {
        blob_log.mark_dead(blob_ref);
    }

    // Run GC — should compact since 80% is dead (> 50% threshold)
    let compacted = blob_log.run_gc().unwrap();
    assert!(compacted > 0, "GC should have compacted at least one file");

    // The 2 live values should still be readable
    for blob_ref in &refs[..2] {
        // After GC, the old refs point to deleted files, but the values
        // should be accessible via the index (which was updated)
        let hash = {
            let index = blob_log.index.read().unwrap();
            index.iter()
                .find(|(_, v)| v.length == blob_ref.length)
                .map(|(h, _)| *h)
        };
        if let Some(hash) = hash {
            let new_ref = blob_log.lookup_by_hash(&hash).unwrap();
            let value = blob_log.read(&new_ref).unwrap();
            assert_eq!(value.len(), blob_ref.length as usize);
        }
    }
}

#[test]
fn gc_deletes_file_with_no_live_values() {
    let (blob_log, _dir) = create_blob_log_with(1, 100);

    // Write a value
    let value = large_value(200);
    let (_hash, blob_ref) = blob_log.write(&value).unwrap();

    // Mark it dead
    blob_log.mark_dead(&blob_ref);

    // The file should exist before GC
    let path = blob_log.blob_dir().join(format!("blob_{}.dat", blob_ref.file_id));
    assert!(path.exists());

    // Run GC
    let compacted = blob_log.run_gc().unwrap();
    assert_eq!(compacted, 1);

    // The file should be deleted
    assert!(!path.exists());
}

#[test]
fn gc_skips_files_below_discard_threshold() {
    let (blob_log, _dir) = create_blob_log_with(1, 100);

    // Write values and keep all alive
    for i in 0..5 {
        let value = large_value(200 + i);
        let _ = blob_log.write(&value).unwrap();
    }

    // No dead values — GC should not compact anything
    let compacted = blob_log.run_gc().unwrap();
    assert_eq!(compacted, 0);
}

#[test]
fn blob_file_stats_discard_ratio() {
    let stats = BlobFileStats {
        live_bytes: 100,
        total_bytes: 200,
        live_count: 1,
        total_count: 2,
        path: PathBuf::from("test.dat"),
        file_id: 0,
    };
    assert!((stats.discard_ratio() - 0.5).abs() < f64::EPSILON);
    assert!(!stats.needs_gc()); // exactly 50% is not > 50%

    let stats2 = BlobFileStats {
        live_bytes: 90,
        total_bytes: 200,
        live_count: 1,
        total_count: 2,
        path: PathBuf::from("test.dat"),
        file_id: 0,
    };
    assert!(stats2.discard_ratio() > 0.5);
    assert!(stats2.needs_gc());
}

#[test]
fn blob_file_stats_empty_file() {
    let stats = BlobFileStats {
        live_bytes: 0,
        total_bytes: 0,
        live_count: 0,
        total_count: 0,
        path: PathBuf::from("test.dat"),
        file_id: 0,
    };
    assert_eq!(stats.discard_ratio(), 0.0);
    assert!(!stats.needs_gc());
}

// ── Sub-task 11.5: Integration tests ──

#[test]
fn large_value_separation_end_to_end() {
    let (blob_log, _dir) = create_test_blob_log();

    // Simulate the write path: value > 1 KB gets separated
    let large = large_value(4096);
    let small = b"small value".to_vec();

    assert!(blob_log.should_separate(&large));
    assert!(!blob_log.should_separate(&small));

    // Write the large value to blob log
    let (hash, blob_ref) = blob_log.write(&large).unwrap();
    let memtable_value = encode_blob_ref(&hash, &blob_ref);

    // The memtable value should be much smaller than the original
    assert!(memtable_value.len() < large.len());
    assert_eq!(memtable_value.len(), BLOB_MARKER_SIZE);

    // On read, transparently fetch the actual value
    let fetched = blob_log.read_transparent(&memtable_value).unwrap();
    assert_eq!(fetched, large);

    // Small values pass through unchanged
    let fetched_small = blob_log.read_transparent(&small).unwrap();
    assert_eq!(fetched_small, small);
}

#[test]
fn multiple_large_values_across_queues() {
    let (blob_log, _dir) = create_test_blob_log();

    let mut values_and_refs = Vec::new();
    for i in 0..20 {
        let value = large_value(2048 + i * 100);
        let (hash, blob_ref) = blob_log.write(&value).unwrap();
        values_and_refs.push((value, hash, blob_ref));
    }

    // All values should be readable
    for (original, _hash, blob_ref) in &values_and_refs {
        let read_back = blob_log.read(blob_ref).unwrap();
        assert_eq!(&read_back, original);
    }

    // Values should be distributed across all 4 queues
    let file_ids: std::collections::HashSet<u64> =
        values_and_refs.iter().map(|(_, _, br)| br.file_id).collect();
    assert_eq!(file_ids.len(), 4, "values should span all 4 writer queues");
}

#[test]
fn gc_reclaims_space_after_compaction() {
    let (blob_log, _dir) = create_blob_log_with(1, 100);

    // Write 10 values
    let mut all_refs = Vec::new();
    let mut all_values = Vec::new();
    for i in 0..10 {
        let value = large_value(500 + i * 10);
        let (hash, blob_ref) = blob_log.write(&value).unwrap();
        all_refs.push((hash, blob_ref));
        all_values.push(value);
    }

    // Mark 8 out of 10 as dead (simulating compaction removing those keys)
    for (_hash, blob_ref) in &all_refs[2..] {
        blob_log.mark_dead(blob_ref);
    }

    // Get file size before GC
    let stats_before = blob_log.collect_file_stats().unwrap();
    let total_before: u64 = stats_before.iter().map(|s| s.total_bytes).sum();

    // Run GC
    let compacted = blob_log.run_gc().unwrap();
    assert!(compacted > 0);

    // Get file size after GC
    let stats_after = blob_log.collect_file_stats().unwrap();
    let total_after: u64 = stats_after.iter().map(|s| s.total_bytes).sum();

    // Space should be reclaimed
    assert!(
        total_after < total_before,
        "GC should reclaim space: before={}, after={}",
        total_before,
        total_after
    );

    // The 2 surviving values should still be readable via the index
    for i in 0..2 {
        let hash = all_refs[i].0;
        let new_ref = blob_log.lookup_by_hash(&hash).unwrap();
        let value = blob_log.read(&new_ref).unwrap();
        assert_eq!(value, all_values[i]);
    }
}

// ── v0.5 format versioning: backward-compat read + newer-format refusal ──

/// A value written by the current engine reads back intact (versioned entry).
#[test]
fn versioned_blob_entry_roundtrips() {
    let (blob_log, _dir) = create_test_blob_log();
    let value = large_value(4096);
    let (_hash, blob_ref) = blob_log.write(&value).unwrap();
    assert_eq!(blob_log.read(&blob_ref).unwrap(), value);
}

/// A hand-written **legacy** (pre-v0.5, `BLOB` magic, no version field) entry
/// is still read by the current engine (backward-compat, Req 5.1).
#[test]
fn legacy_blob_entry_still_reads() {
    let (blob_log, dir) = create_test_blob_log();
    let blob_dir = dir.path().join("blobs");
    let value = large_value(200);
    let hash = content_hash(&value);

    let mut entry = Vec::new();
    entry.extend_from_slice(&BLOB_ENTRY_MAGIC.to_le_bytes()); // legacy magic
    entry.extend_from_slice(&(value.len() as u32).to_le_bytes());
    entry.extend_from_slice(&hash);
    entry.extend_from_slice(&value);
    let checksum = xxh3_64(&entry);
    entry.extend_from_slice(&checksum.to_le_bytes());
    std::fs::write(blob_dir.join("blob_99.dat"), &entry).unwrap();

    let br = BlobRef {
        file_id: 99,
        offset: 0,
        length: value.len() as u32,
    };
    assert_eq!(blob_log.read(&br).unwrap(), value);
}

/// A versioned entry whose format_version exceeds what this engine supports is
/// refused (rollback safety, Req 5.2) — surfaced as an InvalidData I/O error
/// carrying the typed FormatTooNew message.
#[test]
fn future_blob_entry_version_refused() {
    let (blob_log, dir) = create_test_blob_log();
    let blob_dir = dir.path().join("blobs");
    let value = large_value(120);
    let hash = content_hash(&value);
    let future = galaxdb_common::format::BLOB.current_write + 1;

    let mut entry = Vec::new();
    entry.extend_from_slice(&BLOB_ENTRY_MAGIC_V2.to_le_bytes());
    entry.extend_from_slice(&future.to_le_bytes());
    entry.extend_from_slice(&(value.len() as u32).to_le_bytes());
    entry.extend_from_slice(&hash);
    entry.extend_from_slice(&value);
    let checksum = xxh3_64(&entry);
    entry.extend_from_slice(&checksum.to_le_bytes());
    std::fs::write(blob_dir.join("blob_98.dat"), &entry).unwrap();

    let br = BlobRef {
        file_id: 98,
        offset: 0,
        length: value.len() as u32,
    };
    let err = blob_log.read(&br).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(
        err.to_string().contains("newer"),
        "expected a too-new format refusal, got: {err}"
    );
}
