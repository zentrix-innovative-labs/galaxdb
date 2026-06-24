# Inline pre-allocated WAL — GalaxDB vs PostgreSQL 16 (single NVMe)

**Hardware:** AWS c6id.4xlarge, instance-store NVMe (XFS, noatime), release builds.
**Both engines' data on the SAME NVMe** (`TMPDIR=/mnt/nvme/tmp` for GalaxDB; PostgreSQL
`PGDATA=/mnt/nvme/pgdata`). synchronous_commit=on, wal_sync_method=fdatasync. Date 2026-06-18.

## Concurrent single-row INSERT (rows/sec)

| Clients | GalaxDB | PostgreSQL 16 |
|---------|---------|---------------|
| 1  | 9,591  | 11,472 |
| 4  | 27,483 | 33,814 |
| 8  | 35,748 | 53,679 |
| 16 | 33,586 | 81,425 |

## Engine WAL microbenchmark

| Path | Before | After (inline pre-alloc WAL) |
|------|--------|------------------------------|
| `put_sync` (1 fsync/commit, serial) | 72 → 347/s | **24,517/s (40 µs/commit)** |
| `put_batch_sync` (per-row, in batch) | — | ~0.5 µs/row (≈1.9M/s) |

## What changed and why (research-backed, measured)

The single-client write path went **347 → 9,591 rows/s (28x)** via three fixes, each verified:

1. **Pre-allocated WAL file (zero-filled).** PostgreSQL pre-allocates 16 MB WAL segments;
   appending to a *growing* file makes every durable write flush extent-allocation + inode
   metadata (~2.7 ms on this NVMe). Writing into a pre-zeroed file in place makes `fdatasync`
   flush only dirty data pages. Measured raw inline write+fdatasync: **0.037 ms (≈25k/s)** on
   the pre-allocated file vs 2.7 ms appending.
2. **Inline commit, no thread hand-off.** The previous design handed each commit to a dedicated
   WAL thread over a channel (~0.8 ms of context-switch latency per commit). PostgreSQL backends
   write+fsync inline; we now do the same. This was the dominant remaining cost.
3. **Two-lock group commit** (PostgreSQL WALInsertLock + WALWriteLock): an *insert* lock guards
   appending bytes; a *flush* lock guards the single `fdatasync`. Concurrent committers coalesce
   into one fdatasync — a committer that finds `flush_offset` already past its bytes returns
   without syncing.

**Measurement integrity note:** an earlier run mistakenly placed GalaxDB's data dir on `/tmp`
(root EBS) while PostgreSQL was on the instance NVMe — a 10x-slower disk for GalaxDB. The table
above puts both on the same NVMe.

## Remaining gap (honest)

GalaxDB is now competitive at low concurrency (1c: 0.84x of PG) but PostgreSQL still scales
better past 8 cores. Two known causes, to be addressed next:
- Per-row `ExecutorContext` rebuild clones the catalog `HashMap` on every INSERT (PostgreSQL does
  not). Fix: make the catalog `Arc`-shared so DML builds the context with an O(1) refcount bump.
- Client-task + `spawn_blocking` thread oversubscription on a 16-vCPU box vs PostgreSQL's 16
  backend processes.

Security is preserved: the concurrent DML path still funnels through `execute_with_context`'s
authorization chokepoint (`enforce_authorization`) with the connection's session, so RBAC /
SQLSTATE 42501 enforcement is unchanged.

## Reproduce

```bash
TMPDIR=/mnt/nvme/tmp cargo run --release -p galaxdb-benchmarks --bin concurrent-insert-bench \
    -- --rows 2000 --clients 1,4,8,16 --pg-port 5432
```
