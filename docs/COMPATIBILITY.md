# PostgreSQL Compatibility Matrix

GalaxDB is an **AI-native vector + data engine that speaks enough of the
PostgreSQL wire protocol and SQL to consolidate an AI application's data
stack behind one binary.** It is **not** a drop-in PostgreSQL replacement:
it deliberately implements the subset of PostgreSQL that matters for
operational + analytical + vector workloads, and is explicit about what it
does not do.

This document states, honestly, what is Supported, Partial, or Unsupported.
Every entry reflects the actual behavior of the code in this repository, not
an aspiration. When a feature is Partial, the limit is named.

> Legend: **✅ Supported** — works as described. **🟡 Partial** — works
> within a stated limit. **❌ Unsupported** — not implemented; a statement
> using it errors rather than silently doing the wrong thing.

## Connectivity & protocol

| Feature | Status | Notes |
|---|---|---|
| PostgreSQL wire protocol v3 | ✅ | Simple + extended query protocols. |
| Simple query (`Q`) | ✅ | |
| Extended query (Parse/Bind/Describe/Execute/Sync) | ✅ | Prepared statements, no re-parse per execute. |
| Bound parameters (text + binary) | ✅ | int2/4/8, float4/8, bool, text. |
| Result column type OIDs in RowDescription | 🟡 | Real OIDs for scalar + text types; numeric/uuid/date/timestamp reported as `text` (their binary encoding is deferred). |
| Binary result format | 🟡 | Honored for bool/int2/4/8/float4/8; other types served as text. |
| `COPY … FROM STDIN` / `TO STDOUT` | ✅ | Text format. |
| SCRAM-SHA-256 authentication | ✅ | Enable with `--auth`; wrong password/unknown role → `28P01`. |
| TLS (`sslmode=require`/`allow`) | ✅ | |
| SQL-commenter `traceparent` passthrough | ✅ | Accepted; wired to tracing. |

## Data definition (DDL)

| Feature | Status | Notes |
|---|---|---|
| `CREATE TABLE` | ✅ | Scalar + `EMBEDDING MODEL` columns. |
| `DROP TABLE [IF EXISTS]` | ✅ | |
| `ALTER TABLE … SET STORAGE {COLUMNAR\|LEGACY\|ROW}` | ✅ | GalaxDB extension; rewrites on-disk blocks. |
| `PRIMARY KEY` | 🟡 | Defines the row key; **uniqueness is not enforced** — a repeated PK upserts. |
| `NOT NULL`, `CHECK`, `UNIQUE`, `FOREIGN KEY` | ❌ | Accepted by the parser, **not enforced**. |
| `CREATE INDEX` / `DROP INDEX` | 🟡 | Single-column secondary indexes only; multi-column rejected. |
| `CREATE VIEW` / materialized views | ❌ | |
| `CREATE FUNCTION` / PL/pgSQL / stored procedures | ❌ | |
| Triggers | ❌ | |
| Sequences / `SERIAL` / identity | ❌ | Supply keys explicitly. |
| Schemas / namespaces | ❌ | Single implicit namespace. |

## Data manipulation (DML)

| Feature | Status | Notes |
|---|---|---|
| `INSERT … VALUES` | ✅ | Upsert semantics on primary key. |
| `BULK INSERT` | ✅ | GalaxDB extension; multi-row ingest. |
| `INSERT … SELECT` | ❌ | Not supported. |
| `UPDATE … WHERE` | ✅ | |
| `DELETE … WHERE` | ✅ | |
| `WHERE` (`=`,`<`,`>`,`<=`,`>=`,`!=`, `AND`/`OR`) | ✅ | Column-on-either-side. |

## Queries (DQL)

| Feature | Status | Notes |
|---|---|---|
| Single-table `SELECT` + projection + `WHERE` | ✅ | Native path (zone-map pruned, secondary-index aware). |
| Point lookup by primary key | ✅ | Native fast path. |
| `JOIN` (inner/outer) | ✅ | Analytical path (DataFusion). |
| Aggregates (`COUNT`/`SUM`/`AVG`/`MIN`/`MAX`/…) | ✅ | Analytical path. |
| `GROUP BY` / `HAVING` | ✅ | Analytical path. |
| `DISTINCT` | ✅ | Analytical path. |
| Subqueries (in `FROM`) | ✅ | Analytical path. |
| Set operations (`UNION`/`INTERSECT`/`EXCEPT`) | ✅ | Analytical path. |
| Window functions | ✅ | Analytical path. |
| `ORDER BY` / `LIMIT` / `OFFSET` | ✅ | Analytical path (multi-row); native for the trivial single-table case. |
| Common table expressions (`WITH`) | ❌ | Not routed to the analytical engine reliably; treat as unsupported. |

## Transactions

| Feature | Status | Notes |
|---|---|---|
| `BEGIN` / `COMMIT` / `ROLLBACK` | ✅ | Write-buffered, read-your-writes, snapshot isolation. |
| `START TRANSACTION` / `END` / `ABORT` | ✅ | Aliases. |
| `SAVEPOINT` / `RELEASE` / `ROLLBACK TO` | ✅ | |
| Write-write conflict detection | ✅ | Second writer → SQLSTATE `40001`. |
| Transaction status byte (`I`/`T`/`E`) | ✅ | Failed block rejects statements with `25P02` until end. |
| Analytical (`JOIN`/aggregate), `SEMANTIC_MATCH`, `AT VERSION`, DDL inside a txn | ❌ | Rejected with a typed error (the analytical/vector paths cannot see the uncommitted write buffer). |
| Serializable (SSI) isolation | ❌ | Snapshot isolation only; write skew is possible. |

## Authorization

| Feature | Status | Notes |
|---|---|---|
| `CREATE ROLE` / `DROP ROLE` / `ALTER ROLE … PASSWORD` | ✅ | Superuser-only. |
| `GRANT` / `REVOKE` (`SELECT`/`INSERT`/`UPDATE`/`DELETE`) | ✅ | Table-scoped; enforced at a single executor chokepoint; denied → `42501`. |
| Column-level / row-level security | ❌ | |

## AI-native (GalaxDB extensions, beyond PostgreSQL)

| Feature | Status | Notes |
|---|---|---|
| `EMBEDDING MODEL` columns | ✅ | Embeddings computed via the sidecar on INSERT. |
| `SEMANTIC_MATCH(col, query, threshold)` | ✅ | HNSW + delta buffer vector search. |
| `WHERE NOT DUPLICATE` | ✅ | Near-duplicate collapsing. |
| `AT VERSION <tag\|ts>` time-travel | 🟡 | Single-table native scans; not yet in analytical (JOIN) queries or transactions. |
| `CREATE VERSION TAG` | ✅ | Pins an MVCC snapshot; GC-exempt. |
| `SEMANTIC_MATCH` as a join operand | ❌ | Not yet a DataFusion physical operator. |

## Data types

`bool`, `int2/int4/int8`, `float4/float8`, `numeric`, `text`, `varchar`,
`bytea`, `date`, `timestamp`, `timestamptz`, `json`, `jsonb`, `uuid`, and 1-D
arrays of these are modeled by the type system and map to PostgreSQL OIDs.
Legacy tables continue to decode. See `crates/galaxdb-sql/src/types.rs` for
the authoritative `SqlType ↔ ColumnType ↔ Arrow ↔ OID` mapping.

## `pg_catalog` emulation

🟡 Partial. GalaxDB answers the `pg_catalog` / `pg_*` introspection queries
that common drivers (psql, psycopg, tokio-postgres) issue on connect and for
basic metadata. It is **not** a complete `pg_catalog`; queries outside the
emulated set may return empty or error. Coverage is expanded as real driver
sessions surface new queries (HTAP task 21).

## Migration guidance

GalaxDB is best adopted for **new** AI-native workloads or as a consolidation
target, not as an in-place swap under an application that depends on the full
PostgreSQL feature set.

- **Bulk data load:** use `COPY t FROM STDIN` (text format) or `BULK INSERT`.
  A `pg_dump --data-only` in the COPY text format ingests for tables whose
  schema you have recreated with `CREATE TABLE`.
- **Schema:** recreate tables with `CREATE TABLE`. Strip unsupported clauses
  (`FOREIGN KEY`, `CHECK`, `UNIQUE`, sequences/`SERIAL`); supply primary keys
  explicitly. `NOT NULL`/constraints are accepted but not enforced, so enforce
  invariants in the application if you rely on them.
- **What will not port:** stored procedures/PL/pgSQL, triggers, views,
  materialized views, foreign keys, sequences, and multi-column indexes.
- **Verify, don't assume:** run your driver against a GalaxDB instance and
  check the queries your ORM/driver issues; open an issue for any missing
  `pg_catalog` query so it can be added.

## Positioning (read this)

GalaxDB consolidates the vector DB + relational store + analytical layer for
AI applications into one engine with one wire protocol. Where it speaks
PostgreSQL, it does so faithfully; where it does not, it says so here rather
than failing quietly. If your workload needs full PostgreSQL semantics
(constraints, PL/pgSQL, triggers, FKs), keep PostgreSQL — GalaxDB is not a
replacement for it.
