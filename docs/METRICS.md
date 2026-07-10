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
| `galaxdb_semantic_cache_hits_total` | counter | operations | One per **semantic-cache hit** served (v0.7, E-4.1): a `SEMANTIC_MATCH` answered from the configured semantic cache instead of running HNSW. Zero per miss. See [below](#semantic-cache-hit-metric--live-in-v070-cloud-e-41). |
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

Scale-to-zero databases stop and start frequently, so the cumulative counters must not
reset to 0 on start. GalaxDB implements **both** acceptable designs:

1. **Volume persistence (primary).** Cumulative totals are written to
   `<data_dir>/metering.gmet` — a versioned `GMET` header (the same `galaxdb-common::format`
   machinery used by every other on-disk artifact) followed by the cumulative little-endian
   `u64` totals (six in v0.6, seven since v0.7 with `semantic_cache_hits`; the reader is
   length-tolerant so a v0.6 file loads with the new counter seeded to 0).
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

## Semantic-cache-hit metric — live in v0.7.0 (Cloud E-4.1)

`galaxdb_semantic_cache_hits_total` (counter) is **live as of v0.7.0**. The semantic cache
shipped this release (`CREATE SEMANTIC CACHE FOR TABLE <t> SIMILARITY <f> TTL <n>`): a query
whose embedding is within the configured cosine similarity of a cached query, inside its TTL
and with matching search params + embedding model, is served from cache and increments this
counter (the HNSW search is skipped). A miss does not move it. The counter:

- increments exactly once per served hit, zero per miss (monotonic, cumulative);
- is **restart-durable** as the 7th `u64` in `<data_dir>/metering.gmet` (a v0.6 six-counter
  file loads forward-compatibly with this counter seeded to 0 — length-tolerant read);
- is present-and-0 (never a stub) when no semantic cache is configured.

This closes the last E-4 billing dimension: Cloud already maps the name, so it flows end-to-end
with no Cloud change.

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
