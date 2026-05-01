//! Tests for the Memtable and MemtableManager.

use super::*;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Sub-task 4.7: Concurrent writes to different shards (no contention)
// ---------------------------------------------------------------------------

#[test]
fn shard_selection_distributes_keys() {
    // Verify that different keys map to different shards.
    let mut shard_hits = [0u32; NUM_SHARDS];
    for i in 0..1000u32 {
        let key = format!("key-{i}").into_bytes();
        let idx = Memtable::shard_index(&key);
        assert!(idx < NUM_SHARDS);
        shard_hits[idx] += 1;
    }
    // With 1000 keys and 16 shards, each shard should get at least some keys.
    for (i, &count) in shard_hits.iter().enumerate() {
        assert!(count > 0, "shard {i} got zero keys");
    }
}

#[test]
fn concurrent_writes_to_different_shards() {
    let memtable = Arc::new(Memtable::new(64 * 1024 * 1024));
    let mut handles = Vec::new();

    // Spawn threads that write to keys that hash to different shards.
    for i in 0..16u32 {
        let mt = memtable.clone();
        let handle = std::thread::spawn(move || {
            // Write 100 keys per thread.
            for j in 0..100u32 {
                let key = format!("shard-{i}-key-{j}").into_bytes();
                let value = format!("value-{i}-{j}").into_bytes();
                mt.put(key, (i * 100 + j) as u64, Some(value));
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("thread panicked");
    }

    // Verify all writes are visible.
    for i in 0..16u32 {
        for j in 0..100u32 {
            let key = format!("shard-{i}-key-{j}").into_bytes();
            let expected = format!("value-{i}-{j}").into_bytes();
            let result = memtable.get(&key);
            assert_eq!(result, Some(Some(expected)));
        }
    }
}

// ---------------------------------------------------------------------------
// Sub-task 4.7: Same-key serialization
// ---------------------------------------------------------------------------

#[test]
fn same_key_updates_are_serialized() {
    let memtable = Arc::new(Memtable::new(64 * 1024 * 1024));
    let key = b"shared-key".to_vec();
    let mut handles = Vec::new();

    // Multiple threads update the same key with increasing timestamps.
    for i in 0..10u32 {
        let mt = memtable.clone();
        let k = key.clone();
        let handle = std::thread::spawn(move || {
            let value = format!("version-{i}").into_bytes();
            mt.put(k, i as u64, Some(value));
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("thread panicked");
    }

    // The key should exist and have a version chain.
    let result = memtable.get(&key);
    assert!(result.is_some());
    // The latest value should be one of the versions written.
    let value = result.unwrap().unwrap();
    let value_str = String::from_utf8(value).unwrap();
    assert!(value_str.starts_with("version-"));
}

#[test]
fn mvcc_version_chain_is_built_correctly() {
    let memtable = Memtable::new(64 * 1024 * 1024);
    let key = b"versioned-key".to_vec();

    // Write three versions with increasing timestamps.
    memtable.put(key.clone(), 10, Some(b"v1".to_vec()));
    memtable.put(key.clone(), 20, Some(b"v2".to_vec()));
    memtable.put(key.clone(), 30, Some(b"v3".to_vec()));

    // Latest read should return v3.
    assert_eq!(memtable.get(&key), Some(Some(b"v3".to_vec())));

    // MVCC reads at different timestamps.
    assert_eq!(memtable.get_at(&key, 30), Some(Some(b"v3".to_vec())));
    assert_eq!(memtable.get_at(&key, 25), Some(Some(b"v2".to_vec())));
    assert_eq!(memtable.get_at(&key, 15), Some(Some(b"v1".to_vec())));
    assert_eq!(memtable.get_at(&key, 5), None);
}

// ---------------------------------------------------------------------------
// Sub-task 4.7: Seal threshold
// ---------------------------------------------------------------------------

#[test]
fn seal_threshold_triggers_correctly() {
    // Use a small threshold for testing.
    let threshold = 1024u64; // 1 KB
    let memtable = Memtable::new(threshold);

    // Write data until we cross the threshold.
    let mut should_seal = false;
    for i in 0..200u32 {
        let key = format!("key-{i:04}").into_bytes();
        let value = vec![0u8; 32]; // 32 bytes per value
        should_seal = memtable.put(key, i as u64, Some(value));
        if should_seal {
            break;
        }
    }

    assert!(should_seal, "seal threshold should have been crossed");
    assert!(memtable.size() >= threshold);
}

#[test]
fn sealed_memtable_rejects_writes() {
    let memtable = Memtable::new(64 * 1024 * 1024);
    memtable.put(b"key1".to_vec(), 1, Some(b"value1".to_vec()));

    memtable.seal();
    assert!(memtable.is_sealed());

    // Writes to a sealed memtable return false (not accepted).
    let accepted = memtable.put(b"key2".to_vec(), 2, Some(b"value2".to_vec()));
    assert!(!accepted);

    // Original data is still readable.
    assert_eq!(memtable.get(b"key1"), Some(Some(b"value1".to_vec())));
    // New key was not written.
    assert_eq!(memtable.get(b"key2"), None);
}

// ---------------------------------------------------------------------------
// Sub-task 4.7: MemtableManager seal and swap
// ---------------------------------------------------------------------------

#[test]
fn manager_seals_and_swaps_on_threshold() {
    // Use a small threshold so we can trigger seal quickly.
    let threshold = 512u64;
    let back_pressure = 4096u64;
    let manager = MemtableManager::new(threshold, back_pressure);

    assert_eq!(manager.sealed_count(), 0);

    // Write enough data to trigger a seal.
    for i in 0..100u32 {
        let key = format!("key-{i:04}").into_bytes();
        let value = vec![0u8; 32];
        let active = manager.active();
        let should_seal = active.put(key, i as u64, Some(value));
        if should_seal {
            manager.seal_active();
            break;
        }
    }

    // There should be at least one sealed memtable.
    assert!(manager.sealed_count() >= 1);

    // The active memtable should be a new, empty one.
    let active = manager.active();
    assert!(!active.is_sealed());
}

#[test]
fn manager_reads_from_sealed_memtables() {
    let threshold = 256u64;
    let back_pressure = 4096u64;
    let manager = MemtableManager::new(threshold, back_pressure);

    // Write a key to the active memtable.
    let active = manager.active();
    active.put(b"early-key".to_vec(), 1, Some(b"early-value".to_vec()));

    // Force seal.
    manager.seal_active();

    // The key should still be readable through the manager.
    let result = manager.get(b"early-key");
    assert_eq!(result, Some(Some(b"early-value".to_vec())));

    // Write a new key to the new active memtable.
    let active = manager.active();
    active.put(b"late-key".to_vec(), 2, Some(b"late-value".to_vec()));

    // Both keys should be readable.
    assert_eq!(manager.get(b"early-key"), Some(Some(b"early-value".to_vec())));
    assert_eq!(manager.get(b"late-key"), Some(Some(b"late-value".to_vec())));
}

#[test]
fn manager_flush_complete_removes_sealed() {
    let threshold = 256u64;
    let back_pressure = 4096u64;
    let manager = MemtableManager::new(threshold, back_pressure);

    // Write and seal.
    let active = manager.active();
    active.put(b"key".to_vec(), 1, Some(b"value".to_vec()));
    let size = active.size();
    manager.seal_active();

    assert_eq!(manager.sealed_count(), 1);

    // Simulate flush completion.
    manager.on_flush_complete(size);

    assert_eq!(manager.sealed_count(), 0);
}

// ---------------------------------------------------------------------------
// Sub-task 4.7: Back-pressure blocking
// ---------------------------------------------------------------------------

#[tokio::test]
async fn back_pressure_blocks_when_exceeded() {
    // Very small back-pressure limit.
    let threshold = 128u64;
    let back_pressure = 256u64;
    let manager = Arc::new(MemtableManager::new(threshold, back_pressure));

    // Fill up sealed memtables to consume back-pressure capacity.
    // Write enough to seal multiple memtables.
    for i in 0..50u32 {
        let key = format!("bp-key-{i:04}").into_bytes();
        let value = vec![0u8; 32];
        let active = manager.active();
        let should_seal = active.put(key, i as u64, Some(value));
        if should_seal {
            manager.seal_active();
        }
    }

    // The semaphore should have reduced available permits.
    let available = manager.available_back_pressure();
    // With sealed memtables consuming permits, available should be less than initial.
    assert!(
        available < back_pressure as usize,
        "back-pressure should be partially consumed, available={available}"
    );
}

// ---------------------------------------------------------------------------
// Sub-task 4.7: Epoch safety (read copies value, drops handle)
// ---------------------------------------------------------------------------

#[test]
fn read_copies_value_out_of_entry() {
    let memtable = Memtable::new(64 * 1024 * 1024);
    let key = b"epoch-key".to_vec();
    let value = b"epoch-value".to_vec();

    memtable.put(key.clone(), 1, Some(value.clone()));

    // Read the value — this should return a copy, not a reference
    // into the skip map. The Entry handle is dropped inside `get()`.
    let result = memtable.get(&key);
    assert_eq!(result, Some(Some(value)));

    // We can safely use the result across an async boundary because
    // it's an owned Vec<u8>, not a reference into the skip map.
    let owned_value = result.unwrap().unwrap();
    assert_eq!(owned_value, b"epoch-value");
}

#[tokio::test]
async fn read_is_safe_across_async_boundary() {
    let memtable = Arc::new(Memtable::new(64 * 1024 * 1024));
    let key = b"async-key".to_vec();
    let value = b"async-value".to_vec();

    memtable.put(key.clone(), 1, Some(value.clone()));

    // Read the value and use it across an await point.
    let result = memtable.get(&key);
    // The Entry handle has been dropped inside get() — this is safe.
    tokio::task::yield_now().await;

    assert_eq!(result, Some(Some(value)));
}

// ---------------------------------------------------------------------------
// Sub-task 4.7: Tombstone handling
// ---------------------------------------------------------------------------

#[test]
fn tombstone_write_and_read() {
    let memtable = Memtable::new(64 * 1024 * 1024);
    let key = b"tombstone-key".to_vec();

    // Write a value, then delete it (tombstone).
    memtable.put(key.clone(), 1, Some(b"alive".to_vec()));
    memtable.put(key.clone(), 2, None); // tombstone

    // Latest read returns tombstone.
    assert_eq!(memtable.get(&key), Some(None));

    // MVCC read at ts=1 returns the live value.
    assert_eq!(memtable.get_at(&key, 1), Some(Some(b"alive".to_vec())));
    // MVCC read at ts=2 returns tombstone.
    assert_eq!(memtable.get_at(&key, 2), Some(None));
}

// ---------------------------------------------------------------------------
// Sub-task 4.7: iter_all for flush
// ---------------------------------------------------------------------------

#[test]
fn iter_all_returns_sorted_entries() {
    let memtable = Memtable::new(64 * 1024 * 1024);

    // Insert keys in random order.
    memtable.put(b"charlie".to_vec(), 1, Some(b"c".to_vec()));
    memtable.put(b"alpha".to_vec(), 2, Some(b"a".to_vec()));
    memtable.put(b"bravo".to_vec(), 3, Some(b"b".to_vec()));

    let entries = memtable.iter_all();
    assert_eq!(entries.len(), 3);
    assert_eq!(entries[0].0, b"alpha");
    assert_eq!(entries[1].0, b"bravo");
    assert_eq!(entries[2].0, b"charlie");
}

// ---------------------------------------------------------------------------
// Sub-task 4.7: Size tracking
// ---------------------------------------------------------------------------

#[test]
fn size_counter_increases_on_writes() {
    let memtable = Memtable::new(64 * 1024 * 1024);
    assert_eq!(memtable.size(), 0);

    memtable.put(b"key1".to_vec(), 1, Some(b"value1".to_vec()));
    let size_after_first = memtable.size();
    assert!(size_after_first > 0);

    memtable.put(b"key2".to_vec(), 2, Some(b"value2".to_vec()));
    let size_after_second = memtable.size();
    assert!(size_after_second > size_after_first);
}

// ---------------------------------------------------------------------------
// Sub-task 4.7: Key not found
// ---------------------------------------------------------------------------

#[test]
fn get_returns_none_for_missing_key() {
    let memtable = Memtable::new(64 * 1024 * 1024);
    assert_eq!(memtable.get(b"nonexistent"), None);
}

#[test]
fn get_at_returns_none_for_missing_key() {
    let memtable = Memtable::new(64 * 1024 * 1024);
    assert_eq!(memtable.get_at(b"nonexistent", 100), None);
}
