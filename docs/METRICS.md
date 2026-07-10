# GalaxDB Metrics

GalaxDB exposes Prometheus metrics on `/metrics` (default port `9090`, no auth) and a health
summary on `/health`. These are **neutral operational counters and gauges** — the engine has
no concept of tenants, tiers, prices, or billing. A downstream collector (e.g. a control
plane) may scrape and interpret them; the engine just reports what happened.

Scrape it:

```bash
curl -s localhost:9090/metrics | grep '^galaxdb_'
```

## Usage-metering metrics (v0.6, E-4)

These are the cumulative usage counters and capacity gauges. The six counters **persist across
restart and suspend/resume** (see [Persistence](#persistence-across-restart)); the gauges are
recomputed live from the engine.

| Metric | Type | Unit | Definition |
|---|---|---|---|
| `galaxdb_read_ops_total` | counter | operations | One per client **read statement** (point lookup, table scan, time-travel scan), regardless of rows returned. |
| `galaxdb_write_ops_total` | counter | operations | One per client **write statement** — `INSERT` (single-row, multi-row `VALUES`, or `BULK INSERT`), `UPDATE`, `DELETE`, or `COPY FROM` — regardless of the number of rows affected. |
| `galaxdb_vector_ops_total` | counter | operations | One per **semantic/vector search** statement (`SEMANTIC_MATCH`, hybrid, ANN, incl. `AT VERSION`). Disjoint from `read_ops` (a search never also counts as a read). |
| `galaxdb_embedding_ops_total` | counter | rows | One per **row embedded** by the sidecar (documents and queries both). A backlogged/failed embed counts only once it actually succeeds. |
| `galaxdb_near_dedup_rows_total` | counter | rows | Rows processed by a `WHERE NOT DUPLICATE` / near-dedup pass (the buffered candidate set the pass consumes). |
| `galaxdb_training_export_bytes_total` | counter | bytes | Bytes emitted by a successful training-dataset (Lance) export, measured from the on-disk dataset. |
| `galaxdb_storage_bytes` | gauge | bytes | **Physical** on-disk size of this database (post-compaction, compressed, encrypted), summed under the data directory. Refreshed on checkpoint/flush; accurate only while the process runs. |
| `galaxdb_rows_total` | gauge | rows | Total live row count (`Engine::row_count()`). |
| `galaxdb_process_start_time_seconds` | gauge | unix seconds | Process start time, set once at startup. Lets a collector detect a restart and reconcile any counter tail not yet persisted. |

### Definitions are stable

Once a counter counts a thing, later versions keep counting the same thing. A semantic change
is introduced under a **new metric name**, never by silently redefining an existing one. Op
counting happens once per client statement: reads and vector searches at the executor dispatch
(`execute_with_context`), writes at the statement ingress above the per-row `INSERT` fan-out —
so a 10k-row `INSERT` and a 10k-row `COPY` are each exactly one write op.

### Not counted

DDL (`CREATE`/`DROP`/`ALTER`, `CREATE INDEX`), role/grant administration, `BACKUP`/`RESTORE`,
`ANALYZE`, `SHOW`/`pg_catalog` introspection, transaction-control statements, and internal
engine work (catalog persistence, embedding-backlog drains, background near-dedup jobs) do
**not** move the op counters. They count client-visible data operations only.

## Persistence across restart

Scale-to-zero databases stop and start frequently, so the six cumulative counters must not
reset to 0 on start. GalaxDB implements **both** acceptable designs:

1. **Volume persistence (primary).** Cumulative totals are written to
   `<data_dir>/metering.gmet` — a versioned `GMET` header (the same `galaxdb-common::format`
   machinery used by every other on-disk artifact) followed by six little-endian `u64` totals.
   Written crash-safely (`atomic_replace`: temp → fsync → rename → fsync dir) on every
   checkpoint/flush and on graceful shutdown, and read back to seed the live counters on open.
   A crash mid-write leaves either the prior or the new totals, never a torn value. A metering
   file written by a **newer** engine is refused (typed `FormatTooNew`), never misread.
2. **Reset signal (safety net).** `galaxdb_process_start_time_seconds` changes on every
   restart, so a collector can detect the boundary and carry the last-persisted total forward
   across the small window between the last flush and an ungraceful crash.

## Operational metrics (pre-v0.6)

Gauges published since Phase E, unchanged: `galaxdb_buffer_pool_hot_set_usage`,
`galaxdb_buffer_pool_scan_buffer_usage`, `galaxdb_checkpoint_last_duration_ms`,
`galaxdb_compaction_pending_bytes`, `galaxdb_connections_active`,
`galaxdb_embedding_backlog_depth`, `galaxdb_embedding_queue_depth`,
`galaxdb_hnsw_recall_estimate_bp`, `galaxdb_sidecar_status`, `galaxdb_disk_full`,
`galaxdb_wal_write_latency_us`, and the counter `galaxdb_queries_total` (a coarse
all-statements counter, not restart-durable; superseded for usage accounting by the
read/write/vector split above but kept for backward compatibility).

## Semantic-cache-hit metric — deferred, not faked

The E-4 request included `galaxdb_semantic_cache_hits_total`. **GalaxDB has no semantic result
cache today** (the only caches are the DEK cache, the SQL statement/AST cache, and the
buffer-pool block cache — none caches semantic-search results). Semantic caching is a planned
open-source feature that does not exist yet, so per the engineering principles (no faked
metrics) this counter is **not exposed** rather than hardwired to zero. The metric name is
**reserved** and will keep a stable definition once the semantic-cache feature ships. A
collector should keep any cache-hit dimension gated until then.

## Answers to the control-plane metering questions

**Are read/write ops cleanly derivable, or does "operation" need a precise definition?**
One client statement = one op. Reads/searches funnel through a single dispatch and are counted
there (complete + exact); writes are counted once per statement at the ingress, above the
per-row `INSERT` fan-out, so bulk writes are one op regardless of row count. Reads, writes, and
vector ops are three disjoint dimensions. Row-level counters (`rows_read`/`rows_written`) are
not exposed in v0.6 but are cheap to add later as separate additive counters.

**Is per-restart counter persistence available, or must a collector rely on start-time reset
detection?** Persistence is implemented as the primary mechanism (volume-backed, crash-safe),
and `galaxdb_process_start_time_seconds` is also exposed as a reconciliation safety net. A
collector gets both.

**Are storage bytes logical or physical?** `galaxdb_storage_bytes` is **physical** —
post-compaction, compressed, encrypted bytes on disk, summed under the data directory,
accurate while the process runs. While a database is suspended (process stopped), read at-rest
size from the volume instead; the gauge is a running-process metric.
