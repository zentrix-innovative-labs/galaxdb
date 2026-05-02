//! ART node types: Node4, Node16, Node48, Node256 with path compression.
//!
//! Following Leis et al. (ICDE 2013):
//! - Node4:   up to 4 children, sorted key/child arrays
//! - Node16:  up to 16 children, sorted key/child arrays
//! - Node48:  up to 48 children, 256-byte index array → child positions
//! - Node256: up to 256 children, direct array indexed by key byte
//!
//! Nodes grow: Node4 → Node16 → Node48 → Node256
//! Nodes shrink: Node256 → Node48 → Node16 → Node4

use super::RowLocation;

/// A node in the Adaptive Radix Tree.
pub enum ArtNode {
    /// A leaf node storing the full key and its associated value.
    Leaf(LeafNode),
    /// An inner node with path compression and children.
    Inner(Box<InnerNode>),
}

/// Leaf node: stores the full key for verification and the value.
pub struct LeafNode {
    pub key: Vec<u8>,
    pub value: RowLocation,
}

/// Inner node with path compression prefix and one of four node types.
pub struct InnerNode {
    /// Compressed path prefix (path compression).
    pub prefix: Vec<u8>,
    /// The actual node storage variant.
    node_type: NodeType,
}

enum NodeType {
    Node4(Node4),
    Node16(Node16),
    Node48(Box<Node48>),
    Node256(Node256),
}

/// Up to 4 children, stored in sorted arrays.
struct Node4 {
    keys: [u8; 4],
    children: [Option<Box<ArtNode>>; 4],
    count: u8,
}

/// Up to 16 children, stored in sorted arrays.
struct Node16 {
    keys: [u8; 16],
    children: [Option<Box<ArtNode>>; 16],
    count: u8,
}

/// Up to 48 children. Uses a 256-byte index array mapping key bytes to
/// child positions in a compact child array.
struct Node48 {
    /// Maps key byte → index into `children` (255 = empty).
    child_index: [u8; 256],
    children: [Option<Box<ArtNode>>; 48],
    count: u8,
}

/// Up to 256 children, direct array indexed by key byte.
struct Node256 {
    children: Box<[Option<Box<ArtNode>>; 256]>,
    count: u16,
}

const EMPTY_SLOT: u8 = 255;

// ── Node4 ──────────────────────────────────────────────────────────────

impl Node4 {
    fn new() -> Self {
        Self {
            keys: [0; 4],
            children: [None, None, None, None],
            count: 0,
        }
    }

    fn is_full(&self) -> bool {
        self.count >= 4
    }

    fn find_child(&self, byte: u8) -> Option<&ArtNode> {
        for i in 0..self.count as usize {
            if self.keys[i] == byte {
                return self.children[i].as_deref();
            }
        }
        None
    }

    fn find_child_mut(&mut self, byte: u8) -> Option<&mut Option<Box<ArtNode>>> {
        for i in 0..self.count as usize {
            if self.keys[i] == byte {
                return Some(&mut self.children[i]);
            }
        }
        None
    }

    fn insert_child(&mut self, byte: u8, child: Box<ArtNode>) {
        debug_assert!(!self.is_full(), "Node4 overflow");

        // Insert in sorted order
        let pos = self.keys[..self.count as usize]
            .iter()
            .position(|&k| k > byte)
            .unwrap_or(self.count as usize);

        // Shift elements right
        for i in (pos..self.count as usize).rev() {
            self.keys[i + 1] = self.keys[i];
            self.children.swap(i, i + 1);
        }

        self.keys[pos] = byte;
        self.children[pos] = Some(child);
        self.count += 1;
    }

    fn remove_child(&mut self, byte: u8) {
        if let Some(pos) = self.keys[..self.count as usize]
            .iter()
            .position(|&k| k == byte)
        {
            self.children[pos] = None;
            for i in pos..self.count as usize - 1 {
                self.keys[i] = self.keys[i + 1];
                self.children.swap(i, i + 1);
            }
            self.keys[self.count as usize - 1] = 0;
            self.count -= 1;
        }
    }

    /// Grow into a Node16, transferring all children.
    fn into_node16(mut self) -> Node16 {
        let mut n16 = Node16::new();
        for i in 0..self.count as usize {
            n16.keys[i] = self.keys[i];
            n16.children[i] = self.children[i].take();
        }
        n16.count = self.count;
        n16
    }
}

// ── Node16 ─────────────────────────────────────────────────────────────

impl Node16 {
    fn new() -> Self {
        const NONE: Option<Box<ArtNode>> = None;
        Self {
            keys: [0; 16],
            children: [NONE; 16],
            count: 0,
        }
    }

    fn is_full(&self) -> bool {
        self.count >= 16
    }

    fn find_child(&self, byte: u8) -> Option<&ArtNode> {
        for i in 0..self.count as usize {
            if self.keys[i] == byte {
                return self.children[i].as_deref();
            }
        }
        None
    }

    fn find_child_mut(&mut self, byte: u8) -> Option<&mut Option<Box<ArtNode>>> {
        for i in 0..self.count as usize {
            if self.keys[i] == byte {
                return Some(&mut self.children[i]);
            }
        }
        None
    }

    fn insert_child(&mut self, byte: u8, child: Box<ArtNode>) {
        debug_assert!(!self.is_full(), "Node16 overflow");

        let pos = self.keys[..self.count as usize]
            .iter()
            .position(|&k| k > byte)
            .unwrap_or(self.count as usize);

        for i in (pos..self.count as usize).rev() {
            self.keys[i + 1] = self.keys[i];
            self.children.swap(i, i + 1);
        }

        self.keys[pos] = byte;
        self.children[pos] = Some(child);
        self.count += 1;
    }

    fn remove_child(&mut self, byte: u8) {
        if let Some(pos) = self.keys[..self.count as usize]
            .iter()
            .position(|&k| k == byte)
        {
            self.children[pos] = None;
            for i in pos..self.count as usize - 1 {
                self.keys[i] = self.keys[i + 1];
                self.children.swap(i, i + 1);
            }
            self.keys[self.count as usize - 1] = 0;
            self.count -= 1;
        }
    }

    /// Shrink into a Node4, transferring remaining children.
    fn into_node4(mut self) -> Node4 {
        debug_assert!(self.count <= 4);
        let mut n4 = Node4::new();
        for i in 0..self.count as usize {
            n4.keys[i] = self.keys[i];
            n4.children[i] = self.children[i].take();
        }
        n4.count = self.count;
        n4
    }

    /// Grow into a Node48, transferring all children.
    fn into_node48(mut self) -> Node48 {
        let mut n48 = Node48::new();
        for i in 0..self.count as usize {
            let byte = self.keys[i];
            n48.child_index[byte as usize] = i as u8;
            n48.children[i] = self.children[i].take();
        }
        n48.count = self.count;
        n48
    }
}

// ── Node48 ─────────────────────────────────────────────────────────────

impl Node48 {
    fn new() -> Self {
        const NONE: Option<Box<ArtNode>> = None;
        Self {
            child_index: [EMPTY_SLOT; 256],
            children: [
                NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE,
                NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE,
                NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE,
                NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE,
            ],
            count: 0,
        }
    }

    fn is_full(&self) -> bool {
        self.count >= 48
    }

    fn find_child(&self, byte: u8) -> Option<&ArtNode> {
        let idx = self.child_index[byte as usize];
        if idx == EMPTY_SLOT {
            return None;
        }
        self.children[idx as usize].as_deref()
    }

    fn find_child_mut(&mut self, byte: u8) -> Option<&mut Option<Box<ArtNode>>> {
        let idx = self.child_index[byte as usize];
        if idx == EMPTY_SLOT {
            return None;
        }
        Some(&mut self.children[idx as usize])
    }

    fn insert_child(&mut self, byte: u8, child: Box<ArtNode>) {
        debug_assert!(!self.is_full(), "Node48 overflow");

        let slot = self
            .children
            .iter()
            .position(|c| c.is_none())
            .expect("Node48 should have free slot");

        self.child_index[byte as usize] = slot as u8;
        self.children[slot] = Some(child);
        self.count += 1;
    }

    fn remove_child(&mut self, byte: u8) {
        let idx = self.child_index[byte as usize];
        if idx != EMPTY_SLOT {
            self.children[idx as usize] = None;
            self.child_index[byte as usize] = EMPTY_SLOT;
            self.count -= 1;
        }
    }

    /// Shrink into a Node16, transferring remaining children.
    fn into_node16(mut self) -> Node16 {
        debug_assert!(self.count <= 16);
        let mut n16 = Node16::new();
        let mut j = 0;
        for byte in 0..=255u8 {
            let idx = self.child_index[byte as usize];
            if idx != EMPTY_SLOT {
                n16.keys[j] = byte;
                n16.children[j] = self.children[idx as usize].take();
                j += 1;
            }
        }
        n16.count = j as u8;
        n16
    }

    /// Grow into a Node256, transferring all children.
    fn into_node256(mut self) -> Node256 {
        let mut n256 = Node256::new();
        let mut count = 0u16;
        for byte in 0..=255u8 {
            let idx = self.child_index[byte as usize];
            if idx != EMPTY_SLOT {
                n256.children[byte as usize] = self.children[idx as usize].take();
                count += 1;
            }
        }
        n256.count = count;
        n256
    }
}

// ── Node256 ────────────────────────────────────────────────────────────

impl Node256 {
    fn new() -> Self {
        let children: Vec<Option<Box<ArtNode>>> = (0..256).map(|_| None).collect();
        Self {
            children: children
                .try_into()
                .unwrap_or_else(|_| unreachable!()),
            count: 0,
        }
    }

    fn find_child(&self, byte: u8) -> Option<&ArtNode> {
        self.children[byte as usize].as_deref()
    }

    fn find_child_mut(&mut self, byte: u8) -> Option<&mut Option<Box<ArtNode>>> {
        Some(&mut self.children[byte as usize])
    }

    fn insert_child(&mut self, byte: u8, child: Box<ArtNode>) {
        debug_assert!(
            self.children[byte as usize].is_none(),
            "Node256 child already exists at byte {byte}"
        );
        self.children[byte as usize] = Some(child);
        self.count += 1;
    }

    fn remove_child(&mut self, byte: u8) {
        if self.children[byte as usize].is_some() {
            self.children[byte as usize] = None;
            self.count -= 1;
        }
    }

    /// Shrink into a Node48, transferring remaining children.
    fn into_node48(mut self) -> Node48 {
        debug_assert!(self.count <= 48);
        let mut n48 = Node48::new();
        let mut j = 0u8;
        for byte in 0..=255u8 {
            if self.children[byte as usize].is_some() {
                n48.child_index[byte as usize] = j;
                n48.children[j as usize] = self.children[byte as usize].take();
                j += 1;
            }
        }
        n48.count = j;
        n48
    }
}

// ── InnerNode public interface ─────────────────────────────────────────

impl InnerNode {
    /// Create a new inner node starting as Node4.
    pub fn new_node4() -> Self {
        Self {
            prefix: Vec::new(),
            node_type: NodeType::Node4(Node4::new()),
        }
    }

    /// Find a child by key byte (immutable).
    pub fn find_child(&self, byte: u8) -> Option<&ArtNode> {
        match &self.node_type {
            NodeType::Node4(n) => n.find_child(byte),
            NodeType::Node16(n) => n.find_child(byte),
            NodeType::Node48(n) => n.find_child(byte),
            NodeType::Node256(n) => n.find_child(byte),
        }
    }

    /// Find a child by key byte (mutable reference to the Option slot).
    pub fn find_child_mut(&mut self, byte: u8) -> Option<&mut Option<Box<ArtNode>>> {
        match &mut self.node_type {
            NodeType::Node4(n) => n.find_child_mut(byte),
            NodeType::Node16(n) => n.find_child_mut(byte),
            NodeType::Node48(n) => n.find_child_mut(byte),
            NodeType::Node256(n) => n.find_child_mut(byte),
        }
    }

    /// Add a child, growing the node type if necessary.
    pub fn add_child(&mut self, byte: u8, child: Box<ArtNode>) {
        // Grow if the current node type is full
        if self.is_full() {
            self.grow();
        }

        match &mut self.node_type {
            NodeType::Node4(n) => n.insert_child(byte, child),
            NodeType::Node16(n) => n.insert_child(byte, child),
            NodeType::Node48(n) => n.insert_child(byte, child),
            NodeType::Node256(n) => n.insert_child(byte, child),
        }
    }

    /// Remove a child by key byte.
    pub fn remove_child(&mut self, byte: u8) {
        match &mut self.node_type {
            NodeType::Node4(n) => n.remove_child(byte),
            NodeType::Node16(n) => n.remove_child(byte),
            NodeType::Node48(n) => n.remove_child(byte),
            NodeType::Node256(n) => n.remove_child(byte),
        }

        // Shrink if under-utilized
        if self.should_shrink() {
            self.shrink();
        }
    }

    /// Number of children in this node.
    pub fn num_children(&self) -> usize {
        match &self.node_type {
            NodeType::Node4(n) => n.count as usize,
            NodeType::Node16(n) => n.count as usize,
            NodeType::Node48(n) => n.count as usize,
            NodeType::Node256(n) => n.count as usize,
        }
    }

    /// Take the single remaining child from this node.
    /// Panics if the node doesn't have exactly one child.
    pub fn take_single_child(&mut self) -> (u8, Box<ArtNode>) {
        debug_assert_eq!(self.num_children(), 1, "expected exactly one child");

        match &mut self.node_type {
            NodeType::Node4(n) => {
                let byte = n.keys[0];
                let child = n.children[0].take().unwrap();
                n.count = 0;
                (byte, child)
            }
            NodeType::Node16(n) => {
                let byte = n.keys[0];
                let child = n.children[0].take().unwrap();
                n.count = 0;
                (byte, child)
            }
            NodeType::Node48(n) => {
                for byte in 0..=255u8 {
                    let idx = n.child_index[byte as usize];
                    if idx != EMPTY_SLOT {
                        let child = n.children[idx as usize].take().unwrap();
                        n.child_index[byte as usize] = EMPTY_SLOT;
                        n.count = 0;
                        return (byte, child);
                    }
                }
                unreachable!("Node48 with 1 child should have a valid entry");
            }
            NodeType::Node256(n) => {
                for byte in 0..=255u8 {
                    if n.children[byte as usize].is_some() {
                        let child = n.children[byte as usize].take().unwrap();
                        n.count = 0;
                        return (byte, child);
                    }
                }
                unreachable!("Node256 with 1 child should have a valid entry");
            }
        }
    }

    // ── Private helpers ────────────────────────────────────────────────

    fn is_full(&self) -> bool {
        match &self.node_type {
            NodeType::Node4(n) => n.is_full(),
            NodeType::Node16(n) => n.is_full(),
            NodeType::Node48(n) => n.is_full(),
            NodeType::Node256(_) => false,
        }
    }

    fn should_shrink(&self) -> bool {
        match &self.node_type {
            NodeType::Node4(_) => false,
            NodeType::Node16(n) => n.count <= 4,
            NodeType::Node48(n) => n.count <= 16,
            NodeType::Node256(n) => n.count <= 48,
        }
    }

    fn grow(&mut self) {
        let old_type = std::mem::replace(
            &mut self.node_type,
            NodeType::Node4(Node4::new()),
        );

        self.node_type = match old_type {
            NodeType::Node4(n4) => NodeType::Node16(n4.into_node16()),
            NodeType::Node16(n16) => NodeType::Node48(Box::new(n16.into_node48())),
            NodeType::Node48(n48) => NodeType::Node256((*n48).into_node256()),
            NodeType::Node256(_) => unreachable!("cannot grow Node256"),
        };
    }

    fn shrink(&mut self) {
        let old_type = std::mem::replace(
            &mut self.node_type,
            NodeType::Node4(Node4::new()),
        );

        self.node_type = match old_type {
            NodeType::Node4(_) => unreachable!("cannot shrink Node4"),
            NodeType::Node16(n16) => NodeType::Node4(n16.into_node4()),
            NodeType::Node48(n48) => NodeType::Node16((*n48).into_node16()),
            NodeType::Node256(n256) => NodeType::Node48(Box::new(n256.into_node48())),
        };
    }
}
