//! Tests for the Adaptive Radix Tree (ART) primary key index.

use super::{ArtIndex, RowLocation};
use std::sync::Arc;
use std::thread;

fn memtable_loc(shard: u8, key: &[u8]) -> RowLocation {
    RowLocation::Memtable {
        shard,
        key: key.to_vec(),
    }
}

fn sst_loc(sst_id: u64, block_offset: u64, row_offset: u32) -> RowLocation {
    RowLocation::SST {
        sst_id,
        block_offset,
        row_offset,
    }
}

// ── Basic insert/lookup ────────────────────────────────────────────────

#[test]
fn insert_and_lookup_single_key() {
    let index = ArtIndex::new();
    index.insert(b"hello".to_vec(), memtable_loc(0, b"hello"));

    assert_eq!(index.lookup(b"hello"), Some(memtable_loc(0, b"hello")));
    assert_eq!(index.lookup(b"world"), None);
    assert_eq!(index.len(), 1);
}

#[test]
fn insert_and_lookup_multiple_keys() {
    let index = ArtIndex::new();
    index.insert(b"apple".to_vec(), sst_loc(1, 0, 0));
    index.insert(b"banana".to_vec(), sst_loc(1, 100, 1));
    index.insert(b"cherry".to_vec(), sst_loc(2, 0, 0));

    assert_eq!(index.lookup(b"apple"), Some(sst_loc(1, 0, 0)));
    assert_eq!(index.lookup(b"banana"), Some(sst_loc(1, 100, 1)));
    assert_eq!(index.lookup(b"cherry"), Some(sst_loc(2, 0, 0)));
    assert_eq!(index.lookup(b"date"), None);
    assert_eq!(index.len(), 3);
}

#[test]
fn insert_overwrites_existing_key() {
    let index = ArtIndex::new();
    index.insert(b"key1".to_vec(), memtable_loc(0, b"key1"));
    index.insert(b"key1".to_vec(), sst_loc(5, 200, 3));

    assert_eq!(index.lookup(b"key1"), Some(sst_loc(5, 200, 3)));
    assert_eq!(index.len(), 1);
}

// ── Delete ─────────────────────────────────────────────────────────────

#[test]
fn delete_existing_key() {
    let index = ArtIndex::new();
    index.insert(b"key1".to_vec(), memtable_loc(0, b"key1"));
    index.insert(b"key2".to_vec(), memtable_loc(1, b"key2"));

    let removed = index.delete(b"key1");
    assert_eq!(removed, Some(memtable_loc(0, b"key1")));
    assert_eq!(index.lookup(b"key1"), None);
    assert_eq!(index.lookup(b"key2"), Some(memtable_loc(1, b"key2")));
    assert_eq!(index.len(), 1);
}

#[test]
fn delete_nonexistent_key() {
    let index = ArtIndex::new();
    index.insert(b"key1".to_vec(), memtable_loc(0, b"key1"));

    let removed = index.delete(b"nonexistent");
    assert_eq!(removed, None);
    assert_eq!(index.len(), 1);
}

#[test]
fn delete_all_keys_leaves_empty_tree() {
    let index = ArtIndex::new();
    index.insert(b"a".to_vec(), memtable_loc(0, b"a"));
    index.insert(b"b".to_vec(), memtable_loc(1, b"b"));

    index.delete(b"a");
    index.delete(b"b");

    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
    assert_eq!(index.lookup(b"a"), None);
    assert_eq!(index.lookup(b"b"), None);
}

// ── Path compression ───────────────────────────────────────────────────

#[test]
fn keys_with_common_prefix() {
    let index = ArtIndex::new();
    index.insert(b"prefix_alpha".to_vec(), sst_loc(1, 0, 0));
    index.insert(b"prefix_beta".to_vec(), sst_loc(1, 0, 1));
    index.insert(b"prefix_gamma".to_vec(), sst_loc(1, 0, 2));

    assert_eq!(index.lookup(b"prefix_alpha"), Some(sst_loc(1, 0, 0)));
    assert_eq!(index.lookup(b"prefix_beta"), Some(sst_loc(1, 0, 1)));
    assert_eq!(index.lookup(b"prefix_gamma"), Some(sst_loc(1, 0, 2)));
    assert_eq!(index.lookup(b"prefix_"), None);
    assert_eq!(index.len(), 3);
}

#[test]
fn key_is_prefix_of_another() {
    let index = ArtIndex::new();
    index.insert(b"ab".to_vec(), sst_loc(1, 0, 0));
    index.insert(b"abc".to_vec(), sst_loc(1, 0, 1));
    index.insert(b"abcd".to_vec(), sst_loc(1, 0, 2));

    assert_eq!(index.lookup(b"ab"), Some(sst_loc(1, 0, 0)));
    assert_eq!(index.lookup(b"abc"), Some(sst_loc(1, 0, 1)));
    assert_eq!(index.lookup(b"abcd"), Some(sst_loc(1, 0, 2)));
    assert_eq!(index.lookup(b"a"), None);
    assert_eq!(index.len(), 3);
}

// ── Node growth (Node4 → Node16 → Node48 → Node256) ───────────────────

#[test]
fn node_growth_to_node16() {
    let index = ArtIndex::new();
    // Insert 5 keys that share a prefix but differ at the same byte position
    // This forces a Node4 → Node16 transition
    for i in 0..5u8 {
        let key = vec![b'k', i];
        index.insert(key.clone(), sst_loc(i as u64, 0, 0));
    }

    for i in 0..5u8 {
        let key = vec![b'k', i];
        assert_eq!(index.lookup(&key), Some(sst_loc(i as u64, 0, 0)));
    }
    assert_eq!(index.len(), 5);
}

#[test]
fn node_growth_to_node48() {
    let index = ArtIndex::new();
    // Insert 17 keys to force Node16 → Node48
    for i in 0..17u8 {
        let key = vec![b'x', i];
        index.insert(key, sst_loc(i as u64, 0, 0));
    }

    for i in 0..17u8 {
        let key = vec![b'x', i];
        assert_eq!(index.lookup(&key), Some(sst_loc(i as u64, 0, 0)));
    }
    assert_eq!(index.len(), 17);
}

#[test]
fn node_growth_to_node256() {
    let index = ArtIndex::new();
    // Insert 49 keys to force Node48 → Node256
    for i in 0..49u8 {
        let key = vec![b'y', i];
        index.insert(key, sst_loc(i as u64, 0, 0));
    }

    for i in 0..49u8 {
        let key = vec![b'y', i];
        assert_eq!(index.lookup(&key), Some(sst_loc(i as u64, 0, 0)));
    }
    assert_eq!(index.len(), 49);
}

#[test]
fn full_byte_range() {
    let index = ArtIndex::new();
    // Insert all 256 possible byte values as second byte
    for i in 0..=255u8 {
        let key = vec![b'z', i];
        index.insert(key, sst_loc(i as u64, 0, 0));
    }

    for i in 0..=255u8 {
        let key = vec![b'z', i];
        assert_eq!(index.lookup(&key), Some(sst_loc(i as u64, 0, 0)));
    }
    assert_eq!(index.len(), 256);
}

// ── Node shrinking on delete ───────────────────────────────────────────

#[test]
fn delete_causes_node_shrink() {
    let index = ArtIndex::new();
    // Insert enough to grow, then delete to shrink
    for i in 0..20u8 {
        let key = vec![b'n', i];
        index.insert(key, sst_loc(i as u64, 0, 0));
    }
    assert_eq!(index.len(), 20);

    // Delete most keys
    for i in 5..20u8 {
        let key = vec![b'n', i];
        index.delete(&key);
    }
    assert_eq!(index.len(), 5);

    // Remaining keys should still be accessible
    for i in 0..5u8 {
        let key = vec![b'n', i];
        assert_eq!(index.lookup(&key), Some(sst_loc(i as u64, 0, 0)));
    }
}

// ── Rebuild from entries ───────────────────────────────────────────────

#[test]
fn rebuild_from_sst_entries() {
    let index = ArtIndex::new();
    // Insert some initial data
    index.insert(b"old_key".to_vec(), memtable_loc(0, b"old_key"));

    // Simulate crash recovery: rebuild from SST entries
    let sst_entries: Vec<(Vec<u8>, RowLocation)> = vec![
        (b"user:1".to_vec(), sst_loc(1, 0, 0)),
        (b"user:2".to_vec(), sst_loc(1, 0, 1)),
        (b"user:3".to_vec(), sst_loc(1, 100, 0)),
    ];

    index.rebuild_from_entries(sst_entries);

    // Old data should be gone
    assert_eq!(index.lookup(b"old_key"), None);

    // New data should be present
    assert_eq!(index.lookup(b"user:1"), Some(sst_loc(1, 0, 0)));
    assert_eq!(index.lookup(b"user:2"), Some(sst_loc(1, 0, 1)));
    assert_eq!(index.lookup(b"user:3"), Some(sst_loc(1, 100, 0)));
    assert_eq!(index.len(), 3);
}

#[test]
fn rebuild_with_wal_replay_overwrites() {
    let index = ArtIndex::new();

    // Phase 1: rebuild from SST entries
    let sst_entries: Vec<(Vec<u8>, RowLocation)> = vec![
        (b"key_a".to_vec(), sst_loc(1, 0, 0)),
        (b"key_b".to_vec(), sst_loc(1, 0, 1)),
    ];
    index.rebuild_from_entries(sst_entries);

    // Phase 2: simulate WAL replay by inserting memtable locations
    // (WAL replay updates keys that were modified after the last SST flush)
    index.insert(b"key_a".to_vec(), memtable_loc(3, b"key_a"));
    index.insert(b"key_c".to_vec(), memtable_loc(5, b"key_c"));

    assert_eq!(index.lookup(b"key_a"), Some(memtable_loc(3, b"key_a")));
    assert_eq!(index.lookup(b"key_b"), Some(sst_loc(1, 0, 1)));
    assert_eq!(index.lookup(b"key_c"), Some(memtable_loc(5, b"key_c")));
    assert_eq!(index.len(), 3);
}

// ── Concurrent read/write safety ───────────────────────────────────────

#[test]
fn concurrent_reads_and_writes() {
    let index = Arc::new(ArtIndex::new());

    // Pre-populate some data
    for i in 0..100u32 {
        let key = format!("key_{:04}", i).into_bytes();
        index.insert(key, sst_loc(i as u64, 0, i));
    }

    let mut handles = Vec::new();

    // Spawn reader threads
    for _ in 0..4 {
        let idx = Arc::clone(&index);
        handles.push(thread::spawn(move || {
            for i in 0..100u32 {
                let key = format!("key_{:04}", i).into_bytes();
                let result = idx.lookup(&key);
                assert!(result.is_some(), "key_{:04} should exist", i);
            }
        }));
    }

    // Spawn writer threads
    for t in 0..2 {
        let idx = Arc::clone(&index);
        handles.push(thread::spawn(move || {
            for i in 100..200u32 {
                let key = format!("key_{:04}_{}", i, t).into_bytes();
                idx.insert(key, sst_loc(i as u64, t as u64, i));
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Verify original keys still accessible
    for i in 0..100u32 {
        let key = format!("key_{:04}", i).into_bytes();
        assert!(index.lookup(&key).is_some());
    }
}

#[test]
fn concurrent_insert_and_delete() {
    let index = Arc::new(ArtIndex::new());

    // Pre-populate
    for i in 0..50u32 {
        let key = format!("item_{:04}", i).into_bytes();
        index.insert(key, sst_loc(i as u64, 0, i));
    }

    let mut handles = Vec::new();

    // Writer: insert new keys
    let idx = Arc::clone(&index);
    handles.push(thread::spawn(move || {
        for i in 50..100u32 {
            let key = format!("item_{:04}", i).into_bytes();
            idx.insert(key, sst_loc(i as u64, 0, i));
        }
    }));

    // Deleter: delete some existing keys
    let idx = Arc::clone(&index);
    handles.push(thread::spawn(move || {
        for i in 0..25u32 {
            let key = format!("item_{:04}", i).into_bytes();
            idx.delete(&key);
        }
    }));

    // Reader: continuously read
    let idx = Arc::clone(&index);
    handles.push(thread::spawn(move || {
        for _ in 0..100 {
            for i in 25..50u32 {
                let key = format!("item_{:04}", i).into_bytes();
                // These keys should always exist (not being deleted)
                let _ = idx.lookup(&key);
            }
        }
    }));

    for handle in handles {
        handle.join().unwrap();
    }

    // Keys 25-49 should still exist
    for i in 25..50u32 {
        let key = format!("item_{:04}", i).into_bytes();
        assert!(
            index.lookup(&key).is_some(),
            "item_{:04} should still exist",
            i
        );
    }
}

// ── Edge cases ─────────────────────────────────────────────────────────

#[test]
fn empty_key() {
    let index = ArtIndex::new();
    index.insert(vec![], memtable_loc(0, &[]));

    assert_eq!(index.lookup(&[]), Some(memtable_loc(0, &[])));
    assert_eq!(index.len(), 1);

    let removed = index.delete(&[]);
    assert_eq!(removed, Some(memtable_loc(0, &[])));
    assert!(index.is_empty());
}

#[test]
fn single_byte_keys() {
    let index = ArtIndex::new();
    for b in 0..=255u8 {
        index.insert(vec![b], sst_loc(b as u64, 0, 0));
    }

    for b in 0..=255u8 {
        assert_eq!(index.lookup(&[b]), Some(sst_loc(b as u64, 0, 0)));
    }
    assert_eq!(index.len(), 256);
}

#[test]
fn long_keys() {
    let index = ArtIndex::new();
    let key1: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
    let key2: Vec<u8> = (0..1000).map(|i| ((i + 1) % 256) as u8).collect();

    index.insert(key1.clone(), sst_loc(1, 0, 0));
    index.insert(key2.clone(), sst_loc(2, 0, 0));

    assert_eq!(index.lookup(&key1), Some(sst_loc(1, 0, 0)));
    assert_eq!(index.lookup(&key2), Some(sst_loc(2, 0, 0)));
}

#[test]
fn binary_keys_with_zero_bytes() {
    let index = ArtIndex::new();
    index.insert(vec![0, 0, 0], sst_loc(1, 0, 0));
    index.insert(vec![0, 0, 1], sst_loc(2, 0, 0));
    index.insert(vec![0, 1, 0], sst_loc(3, 0, 0));

    assert_eq!(index.lookup(&[0, 0, 0]), Some(sst_loc(1, 0, 0)));
    assert_eq!(index.lookup(&[0, 0, 1]), Some(sst_loc(2, 0, 0)));
    assert_eq!(index.lookup(&[0, 1, 0]), Some(sst_loc(3, 0, 0)));
    assert_eq!(index.lookup(&[0, 0, 2]), None);
}

#[test]
fn default_creates_empty_index() {
    let index = ArtIndex::default();
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
}

#[test]
fn insert_delete_reinsert() {
    let index = ArtIndex::new();
    index.insert(b"key".to_vec(), sst_loc(1, 0, 0));
    index.delete(b"key");
    assert_eq!(index.lookup(b"key"), None);

    index.insert(b"key".to_vec(), sst_loc(2, 0, 0));
    assert_eq!(index.lookup(b"key"), Some(sst_loc(2, 0, 0)));
    assert_eq!(index.len(), 1);
}

// ── Stress test: many keys ─────────────────────────────────────────────

#[test]
fn stress_insert_lookup_delete() {
    let index = ArtIndex::new();
    let n = 10_000u32;

    // Insert
    for i in 0..n {
        let key = format!("stress_key_{:08}", i).into_bytes();
        index.insert(key, sst_loc(i as u64, (i / 100) as u64, i));
    }
    assert_eq!(index.len(), n as usize);

    // Lookup all
    for i in 0..n {
        let key = format!("stress_key_{:08}", i).into_bytes();
        assert_eq!(
            index.lookup(&key),
            Some(sst_loc(i as u64, (i / 100) as u64, i)),
            "failed lookup for key {}",
            i
        );
    }

    // Delete half
    for i in (0..n).step_by(2) {
        let key = format!("stress_key_{:08}", i).into_bytes();
        assert!(index.delete(&key).is_some());
    }
    assert_eq!(index.len(), (n / 2) as usize);

    // Verify remaining
    for i in 0..n {
        let key = format!("stress_key_{:08}", i).into_bytes();
        if i % 2 == 0 {
            assert_eq!(index.lookup(&key), None);
        } else {
            assert!(index.lookup(&key).is_some());
        }
    }
}
