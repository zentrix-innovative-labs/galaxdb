# Write-path benchmarks — v2-phase1 wire work + write-path fixes

**Hardware:** AWS c6id.4xlarge (Intel Ice Lake, 16 vCPU, 32 GiB RAM, instance-store NVMe), Ubuntu, `--release`.
**Instance:** i-0b2dec9226f62db65 (started, run, stopped per engineering-principles §6).
**Date:** 2026-06-18.

All numbers are from real binaries against a real engine/server. Reproduce with the commands shown.

## Root-cause diagnosis (engine microbenchmark, facts not assumptions)

`cargo run --release -p galaxdb-benchmarks --bin engine-microbench -- --max 6000`

| Path | Before fixes | After fixes |
|------|-------------|-------------|
| `put_sync` (1 fsync/row, single client) | 72 rows/s (13.8 ms/row) | 368 rows/s (2.7 ms/row) |
| `put_batch_sync` cumulative (per-row, in-memory) | 6.8 µs/row (flat) | 2.1 µs/row (flat) |
| `put_batch_sync` single call, 5000 rows | — | 785k rows/s (1.3 µs/row) |

The in-memory write path (memtable + ART) is **flat** vs resident size (not O(n)). The
single-row cost was dominated by the WAL group-commit task **always waiting the full 10 ms
window** even for a lone writer, then capping at ~1000/(window). PostgreSQL achieves
~1000–2200 single-row fsync commits/s on NVMe because `commit_delay` defaults to 0.

## Fixes applied (all research-backed, verified here)

1. **TCP_NODELAY** on accepted sockets — the extended-query path sends several small backend
   messages per row; Nagle + delayed-ACK added ~40 ms/round-trip, capping prepared INSERT at 24 rows/s.
2. **Opportunistic ("flush-on-drain") group commit** — a lone writer pays only WAL write + one
   fsync; concurrent writers self-batch. (PostgreSQL `commit_delay=0` / RocksDB write-group model.)
3. **`put_batch_sync` for COPY/BULK INSERT**, committed in bounded chunks — one fsync per chunk
   instead of one per row.
4. **`fdatasync` (sync_data) on the WAL hot path** — PostgreSQL's default `wal_sync_method`.

All 22 `galaxdb-storage` WAL durability/recovery tests pass after the change.

## End-to-end wire results (tokio-postgres over TCP, real galaxdb-server)

`cargo run --release -p galaxdb-benchmarks --bin single-row-insert-bench -- --rows 3000`
`cargo run --release -p galaxdb-benchmarks --bin copy-bench -- --rows 5000`

| Method | Before fixes | After fixes |
|--------|-------------|-------------|
| single-row INSERT (simple) | 209 rows/s | 354 rows/s |
| single-row INSERT (prepared) | 24 rows/s | 354 rows/s |
| COPY FROM STDIN | 209 rows/s | **129,368 rows/s, 11.2 MiB/s** |

COPY is **368x** faster than row-by-row INSERT and the earlier super-linear scaling is gone
(5000-row COPY completes in 38.6 ms).

Single-row strict-durability INSERT (354/s) is bounded by the instance-store NVMe fsync
latency (~2.7 ms/commit); throughput scales with concurrency because writers now share fsyncs.
