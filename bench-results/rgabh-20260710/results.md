# RGABH single-node adaptive storage — skewed-workload hit-rate benchmark

**Feature:** v0.7 Task 5 (inventory 8.1/8.3) — Reinforcement-Gradient Adaptive Block Heat.

## Command (reproducible)

```bash
cargo run --release -p galaxdb-storage --example rgabh_hitrate
```

## Hardware / build

- CPU: Intel Core i7-7820HQ @ 2.90GHz (8 logical cores)
- Memory: 16 GB
- OS: macOS 13.7.8
- Toolchain: rustc 1.96.0, `--release` (optimized)
- Date: 2026-07-10

## Workload

Deterministic Zipfian-skewed access trace (YCSB-style): ~80% of accesses hit a
1,500-key hot set, ~20% spread over a 50,000-key cold tail. Identical seeded
trace driven through two buffer pools of the same capacity (2,000 slots →
HotSet = 1,400): one LRU/clock baseline, one RGABH-adaptive. 2,000,000 ops each.

## Result

| Policy | HotSet hit rate | Wall time |
|---|---|---|
| LRU/clock baseline | 0.6391 | 1.06 s |
| RGABH adaptive | 0.8030 | 2.14 s |
| **Delta** | **+16.39 percentage points** | ~2× (O(K) admission) |

RGABH's W-TinyLFU-style frequency admission keeps the durably-hot working set
resident instead of letting the one-shot cold-tail stream evict it (the failure
mode of plain LRU). Admission/eviction is O(K) (K=16 LRU-tail sample), so the
overhead is a small constant factor, not the O(capacity) scan of an exact
coldest policy.

## Off-switch

`BufferPool::new(..)` (non-adaptive) reproduces the LRU/clock baseline exactly;
verified by `rgabh_off_switch_reproduces_lru_baseline_exactly` and the existing
LRU/clock eviction tests, which pass unchanged.
