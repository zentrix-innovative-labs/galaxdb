//! Snapshot Isolation transaction manager.
//!
//! Provides SI guarantees: no dirty reads, no non-repeatable reads, no phantoms.
//! Write-write conflicts are detected and the second writer is aborted.
//! Write-skew is possible (documented limitation, SSI deferred to v2).

use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

use galaxdb_common::{GalaxError, GalaxResult, Timestamp};

/// Manages transaction timestamps and active snapshots.
pub struct TransactionManager {
    /// Monotonically increasing timestamp counter.
    next_timestamp: AtomicU64,
    /// Set of timestamps for currently active read snapshots.
    active_snapshots: RwLock<BTreeSet<u64>>,
    /// Write locks: key → (writer_txn_id, write_timestamp).
    /// Used for write-write conflict detection.
    write_locks: RwLock<HashMap<Vec<u8>, u64>>,
    /// Recently-committed transactions' `(commit_ts, write_keys)` used by the
    /// v0.7 SSI certifier (inventory 8.14). Pruned to entries that could still
    /// conflict with an active transaction (commit_ts > oldest active begin),
    /// so it stays bounded and never yields a false negative. Only committed
    /// write-sets are recorded (reads don't go here).
    committed_writes: RwLock<Vec<(u64, Vec<Vec<u8>>)>>,
}

impl TransactionManager {
    /// Create a new transaction manager.
    pub fn new() -> Self {
        Self {
            next_timestamp: AtomicU64::new(1),
            active_snapshots: RwLock::new(BTreeSet::new()),
            write_locks: RwLock::new(HashMap::new()),
            committed_writes: RwLock::new(Vec::new()),
        }
    }

    /// Create a new transaction manager starting at a specific timestamp.
    pub fn with_start_timestamp(start: u64) -> Self {
        Self {
            next_timestamp: AtomicU64::new(start),
            active_snapshots: RwLock::new(BTreeSet::new()),
            write_locks: RwLock::new(HashMap::new()),
            committed_writes: RwLock::new(Vec::new()),
        }
    }

    /// Serializable Snapshot Isolation certification + commit (v0.7,
    /// inventory 8.14). Conservative certifier: if `serializable`, abort with
    /// [`GalaxError::WriteConflict`] (SQLSTATE 40001) when any key this
    /// transaction **read** was **written** by a transaction that committed
    /// after this transaction's snapshot (`begin_ts`) — the rw-antidependency
    /// that permits write-skew. Safe (false-positive aborts allowed, never a
    /// false negative). When `!serializable` this is a plain commit (SI).
    ///
    /// Atomic under the committed-ring lock so a concurrent certify/commit
    /// cannot slip a conflicting write past the check. Records this
    /// transaction's `write_keys` for future certifications, releases its
    /// write locks, drops its snapshot, and returns the commit timestamp.
    pub fn commit_serializable(
        &self,
        begin_ts: u64,
        read_keys: &std::collections::HashSet<Vec<u8>>,
        write_keys: Vec<Vec<u8>>,
        serializable: bool,
    ) -> GalaxResult<Timestamp> {
        let mut committed = self.committed_writes.write().unwrap();
        if serializable {
            for (cts, wkeys) in committed.iter() {
                if *cts > begin_ts && wkeys.iter().any(|k| read_keys.contains(k)) {
                    // rw-antidependency into a concurrent committer → abort.
                    return Err(GalaxError::WriteConflict);
                }
            }
        }
        let commit_ts = self.next_timestamp.fetch_add(1, Ordering::SeqCst);
        if !write_keys.is_empty() {
            committed.push((commit_ts, write_keys));
        }
        // Prune entries that can no longer conflict with any active txn: an
        // entry with commit_ts <= the oldest active begin can never satisfy
        // `commit_ts > begin_ts` for any active or future transaction. With no
        // active snapshots, everything is safe to drop.
        let oldest_active = {
            let snaps = self.active_snapshots.read().unwrap();
            snaps.iter().next().copied()
        };
        match oldest_active {
            Some(o) => committed.retain(|(cts, _)| *cts > o),
            None => committed.clear(),
        }
        drop(committed);

        self.release_write_locks(begin_ts);
        self.active_snapshots.write().unwrap().remove(&begin_ts);
        Ok(commit_ts)
    }

    /// Begin a new transaction, returning a Snapshot.
    pub fn begin(&self) -> Snapshot {
        let read_ts = self.next_timestamp.fetch_add(1, Ordering::SeqCst);
        {
            let mut snapshots = self.active_snapshots.write().unwrap();
            snapshots.insert(read_ts);
        }
        Snapshot {
            read_timestamp: read_ts,
            write_set: Vec::new(),
        }
    }

    /// Attempt to acquire a write lock on a key for a transaction.
    /// Returns Err if another transaction already holds the lock (write-write conflict).
    pub fn acquire_write_lock(&self, key: &[u8], txn_ts: u64) -> GalaxResult<()> {
        let mut locks = self.write_locks.write().unwrap();
        if let Some(&existing_ts) = locks.get(key) {
            if existing_ts != txn_ts {
                return Err(GalaxError::WriteConflict);
            }
        }
        locks.insert(key.to_vec(), txn_ts);
        Ok(())
    }

    /// Release all write locks held by a transaction.
    pub fn release_write_locks(&self, txn_ts: u64) {
        let mut locks = self.write_locks.write().unwrap();
        locks.retain(|_, &mut ts| ts != txn_ts);
    }

    /// Commit a transaction: assign a commit timestamp and release resources.
    pub fn commit(&self, snapshot: &Snapshot) -> GalaxResult<Timestamp> {
        let commit_ts = self.next_timestamp.fetch_add(1, Ordering::SeqCst);

        // Release write locks
        self.release_write_locks(snapshot.read_timestamp);

        // Remove from active snapshots
        {
            let mut snapshots = self.active_snapshots.write().unwrap();
            snapshots.remove(&snapshot.read_timestamp);
        }

        Ok(commit_ts)
    }

    /// Abort a transaction: release all resources without committing.
    pub fn abort(&self, snapshot: &Snapshot) {
        self.release_write_locks(snapshot.read_timestamp);
        let mut snapshots = self.active_snapshots.write().unwrap();
        snapshots.remove(&snapshot.read_timestamp);
    }

    /// Get the oldest active snapshot timestamp (used by MVCC GC).
    pub fn oldest_active_snapshot(&self) -> Option<u64> {
        let snapshots = self.active_snapshots.read().unwrap();
        snapshots.iter().next().copied()
    }

    /// Get the number of active snapshots.
    pub fn active_snapshot_count(&self) -> usize {
        self.active_snapshots.read().unwrap().len()
    }

    /// Get the current timestamp (next to be assigned).
    pub fn current_timestamp(&self) -> u64 {
        self.next_timestamp.load(Ordering::SeqCst)
    }
}

impl Default for TransactionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// A transaction snapshot with its read timestamp and write set.
#[derive(Debug, Clone)]
pub struct Snapshot {
    /// The timestamp at which this transaction reads data.
    /// Only MVCC versions with commit_ts <= read_timestamp are visible.
    pub read_timestamp: u64,
    /// Keys written by this transaction (for conflict detection).
    pub write_set: Vec<(Vec<u8>, u64)>,
}

impl Snapshot {
    /// Check if a version is visible to this snapshot.
    pub fn is_visible(&self, commit_ts: u64) -> bool {
        commit_ts <= self.read_timestamp
    }

    /// Record a write to the write set.
    pub fn record_write(&mut self, key: Vec<u8>, write_ts: u64) {
        self.write_set.push((key, write_ts));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn begin_assigns_monotonic_timestamps() {
        let tm = TransactionManager::new();
        let s1 = tm.begin();
        let s2 = tm.begin();
        let s3 = tm.begin();
        assert!(s1.read_timestamp < s2.read_timestamp);
        assert!(s2.read_timestamp < s3.read_timestamp);
    }

    #[test]
    fn snapshot_visibility() {
        let s = Snapshot {
            read_timestamp: 10,
            write_set: vec![],
        };
        assert!(s.is_visible(5));
        assert!(s.is_visible(10));
        assert!(!s.is_visible(11));
    }

    #[test]
    fn no_dirty_reads() {
        let tm = TransactionManager::new();
        let s1 = tm.begin(); // ts=1
        let s2 = tm.begin(); // ts=2

        // s2 writes at ts=2, but hasn't committed yet
        // s1 (ts=1) should NOT see s2's writes
        assert!(!s1.is_visible(s2.read_timestamp));
    }

    #[test]
    fn no_non_repeatable_reads() {
        let tm = TransactionManager::new();
        let s1 = tm.begin(); // ts=1

        // Another transaction commits at ts=3
        let s2 = tm.begin(); // ts=2
        let commit_ts = tm.commit(&s2).unwrap(); // commit_ts=3

        // s1 should NOT see the commit (commit_ts=3 > read_ts=1)
        assert!(!s1.is_visible(commit_ts));

        tm.abort(&s1);
    }

    #[test]
    fn write_write_conflict_detected() {
        let tm = TransactionManager::new();
        let s1 = tm.begin();
        let s2 = tm.begin();

        // s1 acquires write lock on key "x"
        tm.acquire_write_lock(b"x", s1.read_timestamp).unwrap();

        // s2 tries to write the same key — should fail
        let result = tm.acquire_write_lock(b"x", s2.read_timestamp);
        assert!(result.is_err());
        match result.unwrap_err() {
            GalaxError::WriteConflict => {} // expected
            other => panic!("expected WriteConflict, got {:?}", other),
        }

        tm.abort(&s1);
        tm.abort(&s2);
    }

    #[test]
    fn write_lock_released_on_commit() {
        let tm = TransactionManager::new();
        let s1 = tm.begin();

        tm.acquire_write_lock(b"x", s1.read_timestamp).unwrap();
        tm.commit(&s1).unwrap();

        // After s1 commits, another transaction can write to "x"
        let s2 = tm.begin();
        tm.acquire_write_lock(b"x", s2.read_timestamp).unwrap();
        tm.abort(&s2);
    }

    #[test]
    fn write_lock_released_on_abort() {
        let tm = TransactionManager::new();
        let s1 = tm.begin();

        tm.acquire_write_lock(b"x", s1.read_timestamp).unwrap();
        tm.abort(&s1);

        // After abort, lock is released
        let s2 = tm.begin();
        tm.acquire_write_lock(b"x", s2.read_timestamp).unwrap();
        tm.abort(&s2);
    }

    #[test]
    fn same_transaction_can_reacquire_own_lock() {
        let tm = TransactionManager::new();
        let s1 = tm.begin();

        tm.acquire_write_lock(b"x", s1.read_timestamp).unwrap();
        // Same transaction acquiring the same key again should succeed
        tm.acquire_write_lock(b"x", s1.read_timestamp).unwrap();

        tm.abort(&s1);
    }

    #[test]
    fn oldest_active_snapshot_tracks_correctly() {
        let tm = TransactionManager::new();
        assert!(tm.oldest_active_snapshot().is_none());

        let s1 = tm.begin(); // ts=1
        assert_eq!(tm.oldest_active_snapshot(), Some(s1.read_timestamp));

        let s2 = tm.begin(); // ts=2
        assert_eq!(tm.oldest_active_snapshot(), Some(s1.read_timestamp));

        tm.commit(&s1).unwrap();
        assert_eq!(tm.oldest_active_snapshot(), Some(s2.read_timestamp));

        tm.commit(&s2).unwrap();
        assert!(tm.oldest_active_snapshot().is_none());
    }

    #[test]
    fn active_snapshot_count() {
        let tm = TransactionManager::new();
        assert_eq!(tm.active_snapshot_count(), 0);

        let s1 = tm.begin();
        assert_eq!(tm.active_snapshot_count(), 1);

        let s2 = tm.begin();
        assert_eq!(tm.active_snapshot_count(), 2);

        tm.commit(&s1).unwrap();
        assert_eq!(tm.active_snapshot_count(), 1);

        tm.abort(&s2);
        assert_eq!(tm.active_snapshot_count(), 0);
    }

    #[test]
    fn concurrent_transactions_different_keys_no_conflict() {
        let tm = TransactionManager::new();
        let s1 = tm.begin();
        let s2 = tm.begin();

        // Different keys — no conflict
        tm.acquire_write_lock(b"x", s1.read_timestamp).unwrap();
        tm.acquire_write_lock(b"y", s2.read_timestamp).unwrap();

        tm.commit(&s1).unwrap();
        tm.commit(&s2).unwrap();
    }

    #[test]
    fn snapshot_record_write() {
        let mut s = Snapshot {
            read_timestamp: 1,
            write_set: vec![],
        };
        s.record_write(b"key1".to_vec(), 2);
        s.record_write(b"key2".to_vec(), 3);
        assert_eq!(s.write_set.len(), 2);
    }

    #[test]
    fn write_skew_is_possible() {
        // This test documents that write-skew is possible under SI.
        // Two transactions read overlapping data and write to different keys
        // based on what they read — both can commit.
        let tm = TransactionManager::new();
        let s1 = tm.begin();
        let s2 = tm.begin();

        // s1 reads key "a", writes key "b"
        tm.acquire_write_lock(b"b", s1.read_timestamp).unwrap();

        // s2 reads key "b", writes key "a"
        tm.acquire_write_lock(b"a", s2.read_timestamp).unwrap();

        // Both can commit — this is write-skew (SI limitation, SSI in v2)
        tm.commit(&s1).unwrap();
        tm.commit(&s2).unwrap();
    }

    // ── v0.7 SSI certifier (inventory 8.14) ──────────────────────────────

    fn keyset(keys: &[&[u8]]) -> std::collections::HashSet<Vec<u8>> {
        keys.iter().map(|k| k.to_vec()).collect()
    }

    #[test]
    fn ssi_prevents_write_skew() {
        // Classic write-skew: T1 reads X writes Y; T2 reads Y writes X.
        // Under SSI one must abort.
        let tm = TransactionManager::new();
        let t1 = tm.begin();
        let t2 = tm.begin();
        // T1 commits first (wrote Y).
        tm.commit_serializable(t1.read_timestamp, &keyset(&[b"X"]), vec![b"Y".to_vec()], true)
            .unwrap();
        // T2 read Y, which T1 committed after T2's snapshot → abort (40001).
        let res =
            tm.commit_serializable(t2.read_timestamp, &keyset(&[b"Y"]), vec![b"X".to_vec()], true);
        assert!(matches!(res, Err(GalaxError::WriteConflict)), "expected write-skew abort");
    }

    #[test]
    fn si_allows_write_skew_when_not_serializable() {
        // Same shape but serializable=false → SI, both commit (documented).
        let tm = TransactionManager::new();
        let t1 = tm.begin();
        let t2 = tm.begin();
        tm.commit_serializable(t1.read_timestamp, &keyset(&[b"X"]), vec![b"Y".to_vec()], false)
            .unwrap();
        assert!(tm
            .commit_serializable(t2.read_timestamp, &keyset(&[b"Y"]), vec![b"X".to_vec()], false)
            .is_ok());
    }

    #[test]
    fn ssi_no_false_positive_without_conflict() {
        // Disjoint read/write sets → no rw-antidependency → both commit.
        let tm = TransactionManager::new();
        let t1 = tm.begin();
        let t2 = tm.begin();
        tm.commit_serializable(t1.read_timestamp, &keyset(&[b"A"]), vec![b"C".to_vec()], true)
            .unwrap();
        assert!(tm
            .commit_serializable(t2.read_timestamp, &keyset(&[b"B"]), vec![b"D".to_vec()], true)
            .is_ok());
    }

    #[test]
    fn ssi_non_concurrent_read_after_commit_is_fine() {
        // T1 commits BEFORE T2 begins → not concurrent → T2 reading T1's write
        // key is serializable (T2's snapshot already includes T1). No abort.
        let tm = TransactionManager::new();
        let t1 = tm.begin();
        tm.commit_serializable(t1.read_timestamp, &keyset(&[]), vec![b"Y".to_vec()], true)
            .unwrap();
        let t2 = tm.begin(); // begins after T1 committed
        assert!(tm
            .commit_serializable(t2.read_timestamp, &keyset(&[b"Y"]), vec![b"Z".to_vec()], true)
            .is_ok());
    }
}
