//! Adaptive Radix Tree (ART) primary key index.
//!
//! Custom implementation following Leis et al. (ICDE 2013) with:
//! - Node4, Node16, Node48, Node256 node types
//! - Path compression (partial keys stored in nodes)
//! - Grow/shrink transitions between node types
//!
//! The ART maps primary keys (`Vec<u8>`) to `RowLocation` values,
//! enabling O(k) point lookups where k is the key length.

mod node;
#[cfg(test)]
mod tests;

use std::sync::RwLock;

use node::{ArtNode, InnerNode, LeafNode};

/// Location of a row in the storage engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RowLocation {
    /// Row lives in the active memtable.
    Memtable { shard: u8, key: Vec<u8> },
    /// Row lives in an SST file on disk.
    SST {
        sst_id: u64,
        block_offset: u64,
        row_offset: u32,
    },
}

/// Thread-safe Adaptive Radix Tree index.
///
/// Wraps the inner tree with an `RwLock` for concurrent read/write safety
/// (multiple readers, single writer).
pub struct ArtIndex {
    inner: RwLock<AdaptiveRadixTree>,
}

impl ArtIndex {
    /// Create a new empty ART index.
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(AdaptiveRadixTree::new()),
        }
    }

    /// Insert or update a key-value pair.
    pub fn insert(&self, key: Vec<u8>, location: RowLocation) {
        let mut tree = self.inner.write().unwrap();
        tree.insert(key, location);
    }

    /// Look up a key, returning its location if found.
    pub fn lookup(&self, key: &[u8]) -> Option<RowLocation> {
        let tree = self.inner.read().unwrap();
        tree.lookup(key)
    }

    /// Delete a key, returning its previous location if it existed.
    pub fn delete(&self, key: &[u8]) -> Option<RowLocation> {
        let mut tree = self.inner.write().unwrap();
        tree.delete(key)
    }

    /// Rebuild the index from an iterator of (key, location) pairs.
    ///
    /// This clears the existing tree and inserts all entries from the iterator.
    /// Used during crash recovery to rebuild from SST block headers + WAL replay.
    pub fn rebuild_from_entries<I>(&self, entries: I)
    where
        I: IntoIterator<Item = (Vec<u8>, RowLocation)>,
    {
        let mut tree = self.inner.write().unwrap();
        *tree = AdaptiveRadixTree::new();
        for (key, location) in entries {
            tree.insert(key, location);
        }
    }

    /// Return the number of entries in the tree.
    pub fn len(&self) -> usize {
        let tree = self.inner.read().unwrap();
        tree.len
    }

    /// Return true if the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for ArtIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// The inner (non-thread-safe) Adaptive Radix Tree.
struct AdaptiveRadixTree {
    root: Option<Box<ArtNode>>,
    len: usize,
}

impl AdaptiveRadixTree {
    fn new() -> Self {
        Self { root: None, len: 0 }
    }

    fn insert(&mut self, key: Vec<u8>, value: RowLocation) {
        if self.root.is_none() {
            self.root = Some(Box::new(ArtNode::Leaf(LeafNode {
                key,
                value,
            })));
            self.len += 1;
            return;
        }

        let replaced = Self::insert_recursive(&mut self.root, &key, value, 0);
        if !replaced {
            self.len += 1;
        }
    }

    fn lookup(&self, key: &[u8]) -> Option<RowLocation> {
        Self::lookup_recursive(self.root.as_deref(), key, 0)
    }

    fn delete(&mut self, key: &[u8]) -> Option<RowLocation> {
        let (removed, value) = Self::delete_recursive(&mut self.root, key, 0);
        if removed {
            self.len -= 1;
        }
        value
    }

    /// Recursively insert into the tree. Returns true if an existing key was replaced.
    fn insert_recursive(
        node_ref: &mut Option<Box<ArtNode>>,
        key: &[u8],
        value: RowLocation,
        depth: usize,
    ) -> bool {
        let node = node_ref.as_mut().unwrap();

        match node.as_mut() {
            ArtNode::Leaf(leaf) => {
                // If keys match, replace value
                if leaf.key == key {
                    leaf.value = value;
                    return true;
                }

                // Keys differ: create a new inner node with path compression
                let existing_leaf = node_ref.take().unwrap();
                let old_key = match existing_leaf.as_ref() {
                    ArtNode::Leaf(l) => &l.key,
                    _ => unreachable!(),
                };

                // Find the length of the common prefix starting at depth
                let prefix_len = common_prefix_len(
                    &old_key[depth..],
                    &key[depth..],
                );

                let mut inner = InnerNode::new_node4();
                inner.prefix = key[depth..depth + prefix_len].to_vec();

                let split_depth = depth + prefix_len;

                // Get the distinguishing bytes
                let old_byte = old_key.get(split_depth).copied().unwrap_or(0);
                let new_byte = key.get(split_depth).copied().unwrap_or(0);

                inner.add_child(old_byte, existing_leaf);
                inner.add_child(
                    new_byte,
                    Box::new(ArtNode::Leaf(LeafNode {
                        key: key.to_vec(),
                        value,
                    })),
                );

                *node_ref = Some(Box::new(ArtNode::Inner(Box::new(inner))));
                false
            }
            ArtNode::Inner(inner) => {
                // Check prefix match
                let key_remaining = if depth < key.len() {
                    &key[depth..]
                } else {
                    &[]
                };
                let prefix_match_len = common_prefix_len(&inner.prefix, key_remaining);

                if prefix_match_len < inner.prefix.len() {
                    // Prefix mismatch: split this node
                    let mut new_inner = InnerNode::new_node4();
                    new_inner.prefix = inner.prefix[..prefix_match_len].to_vec();

                    let old_prefix_byte = inner.prefix[prefix_match_len];
                    let remaining_prefix = inner.prefix[prefix_match_len + 1..].to_vec();

                    // Take the current node out and modify its prefix
                    let mut old_node = node_ref.take().unwrap();
                    if let ArtNode::Inner(old_inner) = old_node.as_mut() {
                        old_inner.prefix = remaining_prefix;
                    }

                    new_inner.add_child(old_prefix_byte, old_node);

                    let new_key_byte = key.get(depth + prefix_match_len).copied().unwrap_or(0);
                    new_inner.add_child(
                        new_key_byte,
                        Box::new(ArtNode::Leaf(LeafNode {
                            key: key.to_vec(),
                            value,
                        })),
                    );

                    *node_ref = Some(Box::new(ArtNode::Inner(Box::new(new_inner))));
                    return false;
                }

                // Full prefix match — descend
                let next_depth = depth + inner.prefix.len();
                let byte = key.get(next_depth).copied().unwrap_or(0);

                if let Some(child) = inner.find_child_mut(byte) {
                    if child.is_some() {
                        return Self::insert_recursive(child, key, value, next_depth + 1);
                    }
                }

                // No child for this byte — add a new leaf
                inner.add_child(
                    byte,
                    Box::new(ArtNode::Leaf(LeafNode {
                        key: key.to_vec(),
                        value,
                    })),
                );
                false
            }
        }
    }

    fn lookup_recursive(node: Option<&ArtNode>, key: &[u8], depth: usize) -> Option<RowLocation> {
        let node = node?;

        match node {
            ArtNode::Leaf(leaf) => {
                if leaf.key == key {
                    Some(leaf.value.clone())
                } else {
                    None
                }
            }
            ArtNode::Inner(inner) => {
                // Check prefix
                let key_remaining = if depth < key.len() {
                    &key[depth..]
                } else {
                    &[]
                };
                let prefix_match_len = common_prefix_len(&inner.prefix, key_remaining);
                if prefix_match_len < inner.prefix.len() {
                    return None; // prefix mismatch
                }

                let next_depth = depth + inner.prefix.len();
                let byte = key.get(next_depth).copied().unwrap_or(0);

                let child = inner.find_child(byte)?;
                Self::lookup_recursive(Some(child), key, next_depth + 1)
            }
        }
    }

    /// Recursively delete a key. Returns (was_removed, old_value).
    fn delete_recursive(
        node_ref: &mut Option<Box<ArtNode>>,
        key: &[u8],
        depth: usize,
    ) -> (bool, Option<RowLocation>) {
        if node_ref.is_none() {
            return (false, None);
        }

        // Check if this is a matching leaf
        let is_leaf_match = matches!(
            node_ref.as_ref().unwrap().as_ref(),
            ArtNode::Leaf(leaf) if leaf.key == key
        );

        if is_leaf_match {
            let old_node = node_ref.take().unwrap();
            let value = match *old_node {
                ArtNode::Leaf(leaf) => leaf.value,
                _ => unreachable!(),
            };
            return (true, Some(value));
        }

        // Check if it's a non-matching leaf
        if matches!(node_ref.as_ref().unwrap().as_ref(), ArtNode::Leaf(_)) {
            return (false, None);
        }

        // It's an inner node
        let node = node_ref.as_mut().unwrap();
        let inner = match node.as_mut() {
            ArtNode::Inner(inner) => inner.as_mut(),
            _ => unreachable!(),
        };

        // Check prefix
        let key_remaining = if depth < key.len() {
            &key[depth..]
        } else {
            &[]
        };
        let prefix_match_len = common_prefix_len(&inner.prefix, key_remaining);
        if prefix_match_len < inner.prefix.len() {
            return (false, None);
        }

        let next_depth = depth + inner.prefix.len();
        let byte = key.get(next_depth).copied().unwrap_or(0);

        let child_ref = match inner.find_child_mut(byte) {
            Some(child) => child,
            None => return (false, None),
        };

        let (removed, value) = Self::delete_recursive(child_ref, key, next_depth + 1);

        if removed && child_ref.is_none() {
            // Child was removed — clean up the inner node
            inner.remove_child(byte);

            // If only one child remains, collapse (path compression)
            if inner.num_children() == 1 {
                let (child_byte, child) = inner.take_single_child();
                match *child {
                    ArtNode::Inner(mut child_inner) => {
                        // Merge prefixes: current prefix + child_byte + child prefix
                        let mut merged_prefix = inner.prefix.clone();
                        merged_prefix.push(child_byte);
                        merged_prefix.extend_from_slice(&child_inner.prefix);
                        child_inner.prefix = merged_prefix;
                        *node_ref = Some(Box::new(ArtNode::Inner(child_inner)));
                    }
                    leaf @ ArtNode::Leaf(_) => {
                        *node_ref = Some(Box::new(leaf));
                    }
                }
            } else if inner.num_children() == 0 {
                *node_ref = None;
            }
        }

        (removed, value)
    }
}

/// Compute the length of the common prefix between two byte slices.
fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}
