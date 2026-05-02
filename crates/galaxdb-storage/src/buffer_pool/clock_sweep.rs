//! Clock-sweep (second-chance) eviction cache.
//!
//! Each entry has a "referenced" bit. The clock hand sweeps entries:
//! if referenced, clear the bit and move on; if not referenced, evict.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// An entry in the clock-sweep buffer.
struct ClockEntry<K, V> {
    key: K,
    value: V,
    referenced: bool,
}

/// Slot in the circular buffer: either occupied or free.
enum ClockSlot<K, V> {
    Occupied(ClockEntry<K, V>),
    Free,
}

/// Clock-sweep eviction cache.
///
/// Entries are stored in a circular buffer. A clock hand sweeps through entries:
/// - If the entry is referenced, clear the bit and advance.
/// - If the entry is not referenced, evict it.
pub struct ClockSweep<K, V> {
    capacity: usize,
    slots: Vec<ClockSlot<K, V>>,
    map: HashMap<K, usize>, // key → slot index
    hand: usize,            // current clock hand position
    free_list: Vec<usize>,  // free slot indices
}

impl<K: Clone + Eq + Hash, V> ClockSweep<K, V> {
    /// Create a new clock-sweep cache with the given capacity.
    pub fn new(capacity: usize) -> Self {
        let mut slots = Vec::with_capacity(capacity);
        let mut free_list = Vec::with_capacity(capacity);
        for i in (0..capacity).rev() {
            slots.push(ClockSlot::Free);
            free_list.push(i);
        }

        ClockSweep {
            capacity,
            slots,
            map: HashMap::with_capacity(capacity),
            hand: 0,
            free_list,
        }
    }

    /// Returns the maximum number of entries.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the current number of entries.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns `true` if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Look up a key, setting its referenced bit. Returns a reference to the value.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if let Some(&idx) = self.map.get(key) {
            if let ClockSlot::Occupied(entry) = &mut self.slots[idx] {
                entry.referenced = true;
                return Some(&entry.value);
            }
        }
        None
    }

    /// Insert a key-value pair. If the cache is full, evict using clock-sweep.
    /// No constraint checking — use `put_with_constraint` for HotSet-aware eviction.
    pub fn put(&mut self, key: K, value: V) -> Option<(K, V)> {
        self.put_with_constraint(key, value, &HashSet::new())
    }

    /// Insert a key-value pair. If the cache is full, evict using clock-sweep,
    /// but never evict a block whose key is in `protected_keys` (the HotSet).
    ///
    /// If all candidates are protected, the oldest unprotected entry is evicted.
    /// If no unprotected entry exists, the insertion is a no-op (returns `None`).
    pub fn put_with_constraint(
        &mut self,
        key: K,
        value: V,
        protected_keys: &HashSet<K>,
    ) -> Option<(K, V)> {
        // If key already exists, update in place.
        if let Some(&idx) = self.map.get(&key) {
            if let ClockSlot::Occupied(entry) = &mut self.slots[idx] {
                entry.value = value;
                entry.referenced = true;
                return None;
            }
        }

        let mut evicted = None;

        // Need a free slot?
        if self.free_list.is_empty() {
            evicted = self.clock_sweep_evict(protected_keys);
            // If we still have no free slot (all protected), skip insertion.
            if self.free_list.is_empty() {
                return None;
            }
        }

        let idx = self.free_list.pop().expect("free list should have a slot");
        self.slots[idx] = ClockSlot::Occupied(ClockEntry {
            key: key.clone(),
            value,
            referenced: true,
        });
        self.map.insert(key, idx);

        evicted
    }

    /// Remove a key from the cache. Returns the value if present.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(idx) = self.map.remove(key) {
            let slot = std::mem::replace(&mut self.slots[idx], ClockSlot::Free);
            self.free_list.push(idx);
            match slot {
                ClockSlot::Occupied(entry) => Some(entry.value),
                ClockSlot::Free => None,
            }
        } else {
            None
        }
    }

    /// Check if a key is present.
    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Returns an iterator over all keys.
    pub fn keys(&self) -> impl Iterator<Item = K> + '_ {
        self.map.keys().cloned()
    }

    // ── Internal helpers ──────────────────────────────────────────────

    /// Run the clock-sweep algorithm to find one victim to evict.
    /// Skips entries whose keys are in `protected_keys`.
    /// Makes at most `2 * capacity` sweeps to avoid infinite loops.
    fn clock_sweep_evict(&mut self, protected_keys: &HashSet<K>) -> Option<(K, V)> {
        let max_sweeps = self.capacity * 2;
        for _ in 0..max_sweeps {
            let idx = self.hand;
            self.hand = (self.hand + 1) % self.capacity;

            if let ClockSlot::Occupied(entry) = &mut self.slots[idx] {
                // Skip protected entries.
                if protected_keys.contains(&entry.key) {
                    continue;
                }

                if entry.referenced {
                    // Second chance: clear the bit and move on.
                    entry.referenced = false;
                } else {
                    // Evict this entry.
                    let key = entry.key.clone();
                    self.map.remove(&key);
                    let slot = std::mem::replace(&mut self.slots[idx], ClockSlot::Free);
                    self.free_list.push(idx);
                    if let ClockSlot::Occupied(evicted) = slot {
                        return Some((evicted.key, evicted.value));
                    }
                }
            }
        }

        // All entries are protected or referenced — no eviction possible.
        None
    }
}
