//! RGABH skewed-workload hit-rate benchmark (v0.7, inventory 8.1/8.3).
//!
//! Reproducible command (engineering principle §4):
//!
//! ```text
//! cargo run --release -p galaxdb-storage --example rgabh_hitrate
//! ```
//!
//! Drives an identical deterministic Zipfian-skewed access trace (YCSB-style:
//! ~80% of accesses hit a small hot set, the rest spread over a large cold tail)
//! through two buffer pools of the same capacity — one with the LRU/clock
//! baseline, one with RGABH adaptive admission/eviction — and reports the HotSet
//! hit rate of each. RGABH's durable `long_heat` term should retain the hot set
//! through cold-tail floods that evict it under plain LRU.
//!
//! The trace is generated in-process (an access pattern, not a stored dataset),
//! deterministic via a fixed xorshift seed so both pools see the exact same
//! sequence — a fair, repeatable comparison on any hardware.

use galaxdb_storage::buffer_pool::{AccessType, BufferPool, CachedBlock};

const POOL_CAPACITY: usize = 2_000;
const HOT_KEYS: u64 = 1_500;
const COLD_KEYS: u64 = 50_000;
const OPS: usize = 2_000_000;

struct SkewGen {
    state: u64,
}
impl SkewGen {
    fn new() -> Self {
        SkewGen {
            state: 0x1234_5678_9abc_def0,
        }
    }
    fn next(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }
    fn next_key(&mut self) -> u64 {
        let r = self.next();
        if r % 100 < 80 {
            r % HOT_KEYS
        } else {
            HOT_KEYS + (r % COLD_KEYS)
        }
    }
}

fn make_block(id: u64) -> CachedBlock {
    CachedBlock {
        block_id: id,
        data: vec![id as u8; 4096],
    }
}

fn run(pool: &mut BufferPool) -> f64 {
    let mut keygen = SkewGen::new();
    let mut hits = 0usize;
    for _ in 0..OPS {
        let key = keygen.next_key();
        if pool.get_for_point_lookup(key, 0).is_some() {
            hits += 1;
        } else {
            pool.insert(key, make_block(key), AccessType::PointLookup, 0);
        }
    }
    hits as f64 / OPS as f64
}

fn main() {
    println!("RGABH skewed-workload hit-rate benchmark");
    println!(
        "pool_capacity={POOL_CAPACITY} hot_keys={HOT_KEYS} cold_keys={COLD_KEYS} ops={OPS}"
    );

    let mut baseline = BufferPool::new(POOL_CAPACITY, 1);
    let t0 = std::time::Instant::now();
    let base_rate = run(&mut baseline);
    let base_dur = t0.elapsed();

    let mut adaptive = BufferPool::new_adaptive(POOL_CAPACITY, 1);
    let t1 = std::time::Instant::now();
    let adaptive_rate = run(&mut adaptive);
    let adaptive_dur = t1.elapsed();

    println!();
    println!("LRU/clock baseline : hit_rate={base_rate:.4}  ({:?})", base_dur);
    println!("RGABH adaptive     : hit_rate={adaptive_rate:.4}  ({:?})", adaptive_dur);
    let delta_pp = (adaptive_rate - base_rate) * 100.0;
    println!("delta              : {delta_pp:+.2} percentage points");
}
