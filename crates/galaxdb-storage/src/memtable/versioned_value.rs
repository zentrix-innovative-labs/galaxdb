//! MVCC versioned value with version chains.
//!
//! Each key in the memtable maps to a `VersionedValue` which forms a
//! singly-linked list of versions ordered by descending timestamp.
//! A `None` value represents a tombstone (deletion marker).

/// A versioned value in the MVCC version chain.
///
/// Each version carries a commit timestamp, an optional value (None = tombstone),
/// and a pointer to the previous version in the chain.
#[derive(Debug, Clone)]
pub struct VersionedValue {
    /// The MVCC commit timestamp for this version.
    pub timestamp: u64,
    /// The value bytes, or `None` for a tombstone (deletion marker).
    pub value: Option<Vec<u8>>,
    /// Pointer to the previous version in the chain (older timestamp).
    pub prev: Option<Box<VersionedValue>>,
}

impl VersionedValue {
    /// Creates a new versioned value with no previous version.
    pub fn new(timestamp: u64, value: Option<Vec<u8>>) -> Self {
        Self {
            timestamp,
            value,
            prev: None,
        }
    }

    /// Creates a new versioned value that chains onto an existing version.
    pub fn with_prev(timestamp: u64, value: Option<Vec<u8>>, prev: VersionedValue) -> Self {
        Self {
            timestamp,
            value,
            prev: Some(Box::new(prev)),
        }
    }

    /// Returns the value visible at the given read timestamp.
    ///
    /// Walks the version chain to find the latest version with
    /// `timestamp <= read_ts`. Returns `None` if no version is visible
    /// at that timestamp. Returns `Some(None)` if the visible version
    /// is a tombstone.
    pub fn get_at(&self, read_ts: u64) -> Option<Option<Vec<u8>>> {
        let mut current = self;
        loop {
            if current.timestamp <= read_ts {
                return Some(current.value.clone());
            }
            match &current.prev {
                Some(prev) => current = prev,
                None => return None,
            }
        }
    }

    /// Returns the latest timestamp in this version chain.
    pub fn latest_timestamp(&self) -> u64 {
        self.timestamp
    }

    /// Returns the number of versions in this chain.
    pub fn chain_length(&self) -> usize {
        let mut count = 1;
        let mut current = &self.prev;
        while let Some(prev) = current {
            count += 1;
            current = &prev.prev;
        }
        count
    }

    /// Returns `true` if the latest version is a tombstone.
    pub fn is_tombstone(&self) -> bool {
        self.value.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_versioned_value_has_no_prev() {
        let v = VersionedValue::new(1, Some(b"hello".to_vec()));
        assert_eq!(v.timestamp, 1);
        assert_eq!(v.value, Some(b"hello".to_vec()));
        assert!(v.prev.is_none());
        assert_eq!(v.chain_length(), 1);
    }

    #[test]
    fn with_prev_chains_versions() {
        let v1 = VersionedValue::new(1, Some(b"v1".to_vec()));
        let v2 = VersionedValue::with_prev(2, Some(b"v2".to_vec()), v1);
        assert_eq!(v2.timestamp, 2);
        assert_eq!(v2.chain_length(), 2);
        assert_eq!(v2.prev.as_ref().unwrap().timestamp, 1);
    }

    #[test]
    fn get_at_returns_correct_version() {
        let v1 = VersionedValue::new(10, Some(b"v1".to_vec()));
        let v2 = VersionedValue::with_prev(20, Some(b"v2".to_vec()), v1);
        let v3 = VersionedValue::with_prev(30, Some(b"v3".to_vec()), v2);

        // Read at ts=30 sees v3.
        assert_eq!(v3.get_at(30), Some(Some(b"v3".to_vec())));
        // Read at ts=25 sees v2.
        assert_eq!(v3.get_at(25), Some(Some(b"v2".to_vec())));
        // Read at ts=15 sees v1.
        assert_eq!(v3.get_at(15), Some(Some(b"v1".to_vec())));
        // Read at ts=5 sees nothing.
        assert_eq!(v3.get_at(5), None);
    }

    #[test]
    fn tombstone_is_detected() {
        let v = VersionedValue::new(1, None);
        assert!(v.is_tombstone());

        let v2 = VersionedValue::new(1, Some(b"data".to_vec()));
        assert!(!v2.is_tombstone());
    }

    #[test]
    fn get_at_returns_tombstone() {
        let v1 = VersionedValue::new(10, Some(b"alive".to_vec()));
        let v2 = VersionedValue::with_prev(20, None, v1); // tombstone

        assert_eq!(v2.get_at(20), Some(None)); // tombstone
        assert_eq!(v2.get_at(15), Some(Some(b"alive".to_vec())));
    }
}
