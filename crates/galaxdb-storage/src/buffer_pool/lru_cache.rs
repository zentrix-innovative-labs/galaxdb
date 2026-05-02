//! LRU cache implementation using a `HashMap` + doubly-linked list pattern.
//!
//! Tracks capacity in number of entries. On eviction, the least-recently-used
//! entry is removed.

use std::collections::HashMap;
use std::hash::Hash;

/// A node in the doubly-linked list.
struct Node<K, V> {
    key: K,
    value: V,
    prev: Option<usize>,
    next: Option<usize>,
}

/// Slot in the arena: either occupied or free.
enum Slot<K, V> {
    Occupied(Node<K, V>),
    Free,
}

/// LRU cache backed by a HashMap and an intrusive doubly-linked list (arena-allocated).
///
/// The most-recently-used entry is at the head; the least-recently-used is at the tail.
pub struct LruCache<K, V> {
    capacity: usize,
    map: HashMap<K, usize>, // key → index in `slots`
    slots: Vec<Slot<K, V>>,
    /// Free-list of recycled slot indices.
    free: Vec<usize>,
    head: Option<usize>, // MRU end
    tail: Option<usize>, // LRU end
}

impl<K: Clone + Eq + Hash, V> LruCache<K, V> {
    /// Create a new LRU cache with the given capacity (number of entries).
    pub fn new(capacity: usize) -> Self {
        LruCache {
            capacity,
            map: HashMap::with_capacity(capacity),
            slots: Vec::with_capacity(capacity),
            free: Vec::new(),
            head: None,
            tail: None,
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

    /// Look up a key, promoting it to MRU position. Returns a reference to the value.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if let Some(&idx) = self.map.get(key) {
            self.move_to_head(idx);
            Some(&self.node(idx).value)
        } else {
            None
        }
    }

    /// Insert or update a key-value pair. Returns the evicted entry if the cache was full.
    pub fn put(&mut self, key: K, value: V) -> Option<(K, V)> {
        if let Some(&idx) = self.map.get(&key) {
            // Update existing entry.
            self.node_mut(idx).value = value;
            self.move_to_head(idx);
            return None;
        }

        let evicted = if self.map.len() >= self.capacity {
            self.evict_lru()
        } else {
            None
        };

        let idx = self.alloc_node(Node {
            key: key.clone(),
            value,
            prev: None,
            next: None,
        });

        self.push_head(idx);
        self.map.insert(key, idx);

        evicted
    }

    /// Remove a key from the cache. Returns the value if it was present.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(idx) = self.map.remove(key) {
            self.unlink(idx);
            let node = self.take_node(idx);
            Some(node.value)
        } else {
            None
        }
    }

    /// Returns an iterator over all keys currently in the cache.
    pub fn keys(&self) -> impl Iterator<Item = K> + '_ {
        self.map.keys().cloned()
    }

    /// Check if a key is present without promoting it.
    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    // ── Internal helpers ──────────────────────────────────────────────

    fn node(&self, idx: usize) -> &Node<K, V> {
        match &self.slots[idx] {
            Slot::Occupied(n) => n,
            Slot::Free => panic!("accessed free slot {idx}"),
        }
    }

    fn node_mut(&mut self, idx: usize) -> &mut Node<K, V> {
        match &mut self.slots[idx] {
            Slot::Occupied(n) => n,
            Slot::Free => panic!("accessed free slot {idx}"),
        }
    }

    fn alloc_node(&mut self, node: Node<K, V>) -> usize {
        if let Some(idx) = self.free.pop() {
            self.slots[idx] = Slot::Occupied(node);
            idx
        } else {
            let idx = self.slots.len();
            self.slots.push(Slot::Occupied(node));
            idx
        }
    }

    /// Take a node out of its slot, marking the slot as free.
    fn take_node(&mut self, idx: usize) -> Node<K, V> {
        let slot = std::mem::replace(&mut self.slots[idx], Slot::Free);
        self.free.push(idx);
        match slot {
            Slot::Occupied(n) => n,
            Slot::Free => panic!("take_node on free slot {idx}"),
        }
    }

    fn push_head(&mut self, idx: usize) {
        self.node_mut(idx).prev = None;
        self.node_mut(idx).next = self.head;
        if let Some(old_head) = self.head {
            self.node_mut(old_head).prev = Some(idx);
        }
        self.head = Some(idx);
        if self.tail.is_none() {
            self.tail = Some(idx);
        }
    }

    fn unlink(&mut self, idx: usize) {
        let prev = self.node(idx).prev;
        let next = self.node(idx).next;

        if let Some(p) = prev {
            self.node_mut(p).next = next;
        } else {
            self.head = next;
        }

        if let Some(n) = next {
            self.node_mut(n).prev = prev;
        } else {
            self.tail = prev;
        }

        self.node_mut(idx).prev = None;
        self.node_mut(idx).next = None;
    }

    fn move_to_head(&mut self, idx: usize) {
        if self.head == Some(idx) {
            return;
        }
        self.unlink(idx);
        self.push_head(idx);
    }

    fn evict_lru(&mut self) -> Option<(K, V)> {
        if let Some(tail_idx) = self.tail {
            let key = self.node(tail_idx).key.clone();
            self.map.remove(&key);
            self.unlink(tail_idx);
            let node = self.take_node(tail_idx);
            Some((node.key, node.value))
        } else {
            None
        }
    }
}
