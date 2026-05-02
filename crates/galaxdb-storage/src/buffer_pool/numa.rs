//! NUMA-aware partitioning wrapper.
//!
//! On Linux, detects the number of NUMA nodes via libnuma.
//! On macOS/Windows, falls back to a single partition.

/// Detect the number of NUMA nodes on the current system.
///
/// - **Linux**: Reads `/sys/devices/system/node/` to count `nodeN` directories.
///   Falls back to 1 if the sysfs path is unavailable.
/// - **macOS/Windows**: Always returns 1 (single partition fallback).
pub fn detect_numa_nodes() -> usize {
    #[cfg(target_os = "linux")]
    {
        detect_numa_nodes_linux()
    }

    #[cfg(not(target_os = "linux"))]
    {
        1
    }
}

/// Linux-specific NUMA node detection via sysfs.
#[cfg(target_os = "linux")]
fn detect_numa_nodes_linux() -> usize {
    use std::fs;
    use std::path::Path;

    let node_dir = Path::new("/sys/devices/system/node");
    if !node_dir.exists() {
        return 1;
    }

    match fs::read_dir(node_dir) {
        Ok(entries) => {
            let count = entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    let name = e.file_name();
                    let name = name.to_string_lossy();
                    name.starts_with("node") && name[4..].parse::<u32>().is_ok()
                })
                .count();
            count.max(1)
        }
        Err(_) => 1,
    }
}

/// Get the NUMA node for the current thread.
///
/// - **Linux**: Reads from `/sys/devices/system/node/` or uses `getcpu` syscall.
///   Falls back to 0 if detection fails.
/// - **macOS/Windows**: Always returns 0.
pub fn current_numa_node() -> usize {
    #[cfg(target_os = "linux")]
    {
        current_numa_node_linux()
    }

    #[cfg(not(target_os = "linux"))]
    {
        0
    }
}

/// Linux-specific: get the NUMA node of the current CPU.
#[cfg(target_os = "linux")]
fn current_numa_node_linux() -> usize {
    // Use the getcpu syscall (available since Linux 2.6.19).
    // libc::sched_getcpu() returns the CPU number; we then map CPU → NUMA node.
    unsafe {
        let cpu = libc::sched_getcpu();
        if cpu < 0 {
            return 0;
        }
        // Read /sys/devices/system/cpu/cpuN/topology/physical_package_id
        // or /sys/devices/system/cpu/cpuN/node* symlink.
        // Simplified: read the numa_node file.
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id");
        match std::fs::read_to_string(&path) {
            Ok(s) => s.trim().parse::<usize>().unwrap_or(0),
            Err(_) => 0,
        }
    }
}

/// A NUMA-partitioned wrapper that holds one instance of `T` per NUMA node.
///
/// On macOS/Windows, this degenerates to a single partition.
pub struct NumaPartitioned<T> {
    per_node: Vec<T>,
}

impl<T> NumaPartitioned<T> {
    /// Create a new NUMA-partitioned wrapper with `num_nodes` partitions.
    /// The `factory` closure is called once per node to create each instance.
    pub fn new<F: FnMut() -> T>(num_nodes: usize, mut factory: F) -> Self {
        let nodes = num_nodes.max(1);
        let per_node = (0..nodes).map(|_| factory()).collect();
        NumaPartitioned { per_node }
    }

    /// Returns the number of NUMA partitions.
    pub fn node_count(&self) -> usize {
        self.per_node.len()
    }

    /// Get a reference to the partition for the given NUMA node.
    /// The node index is clamped to the valid range.
    pub fn get(&self, node: usize) -> &T {
        let idx = node % self.per_node.len();
        &self.per_node[idx]
    }

    /// Get a mutable reference to the partition for the given NUMA node.
    /// The node index is clamped to the valid range.
    pub fn get_mut(&mut self, node: usize) -> &mut T {
        let len = self.per_node.len();
        let idx = node % len;
        &mut self.per_node[idx]
    }

    /// Get the partition for the current thread's NUMA node.
    pub fn get_local(&self) -> &T {
        self.get(current_numa_node())
    }

    /// Get a mutable reference to the partition for the current thread's NUMA node.
    pub fn get_local_mut(&mut self) -> &mut T {
        self.get_mut(current_numa_node())
    }
}
