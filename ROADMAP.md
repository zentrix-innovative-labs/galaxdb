# GalaxDB Roadmap

GalaxDB unifies transactional (OLTP), analytical (OLAP), and vector workloads in a single AI-native
database. This roadmap shows what ships today, what is in active development, and what is planned.

The open-source core is **Apache 2.0, free forever** — the entire single-node engine, including security
and encryption. A commercial enterprise edition adds distributed scale and governance for organizations
that outgrow a single node.

Track active work on the [issue tracker](https://github.com/zentrix-innovative-labs/galaxdb/issues). Items
labeled **`roadmap`** map to the entries below; **`good first issue`** marks places to start contributing.

---

## Available now

The single-node engine is feature-complete and benchmarked on SIFT-1M (see
[BENCHMARKS.md](docs/BENCHMARKS.md)).

**Storage** — LSM + PAX columnar storage, ART primary index, Monkey-optimal Bloom filters, NUMA-aware
buffer pool, Lazy Leveling compaction with MVCC GC, WAL with crash recovery, key-value separation,
write-stall mitigation, io_uring/tokio backends, automatic memory/configuration tuning (buffer pool,
memtable, compaction concurrency derived from host RAM/CPU), a **durable catalog** (table
definitions, storage modes, and constraints survive restart), and AES-256-GCM encryption at rest with
pluggable key management (local, environment, external command, HashiCorp Vault).

**SQL query engine** — relational and analytical SQL over an embedded DataFusion engine: joins,
aggregates (`COUNT`/`SUM`/`AVG`/`MIN`/`MAX`), `GROUP BY` / `HAVING`, `DISTINCT`, `ORDER BY` /
`LIMIT` / `OFFSET`, and FROM-less scalar queries (`SELECT 1+1`, `version()`, `current_database()`).
Single-table point reads, filtered scans, and vector search run on the native path; anything
analytical routes to DataFusion. `INSERT` / `UPDATE` value positions are real per-row expressions
(e.g. `SET bal = bal - 30`), `PRIMARY KEY` uniqueness is enforced (SQLSTATE `23505`, never a silent
overwrite), and arithmetic faults are typed (`22012` division-by-zero, `22003` overflow, `42804`
type mismatch).

**Transactions** — explicit `BEGIN` / `COMMIT` / `ROLLBACK` with snapshot isolation, read-your-writes
overlay, `SAVEPOINT` / `ROLLBACK TO`, and write-write conflict detection (SQLSTATE `40001`).

**Vector search** — mutable HNSW with crash-safe delta buffer, SQ8/FP16/RaBitQ quantization, parallel
construction.

**AI-native SQL** — `EMBEDDING MODEL` columns with a local embedding sidecar (any HuggingFace model),
`SEMANTIC_MATCH`, `AT VERSION` time-travel (including over flushed on-disk/SST data), version tags, MinHash
near-duplicate detection (`WHERE NOT DUPLICATE`), `FOR TRAINING` Lance export to PyTorch (including
embedding/vector columns as Arrow `FixedSizeList`, with float32/SQ8/RaBitQ training precision), and EU AI
Act Article 13 data lineage. Single-column secondary indexes (`CREATE INDEX` / `DROP INDEX`) accelerate
equality and range lookups on non-primary-key columns, with a transparent full-scan fallback.

**Security** — SCRAM-SHA-256 wire authentication, TLS 1.2/1.3 transport encryption (rustls, no OpenSSL),
role-based access control with table-level `GRANT` / `REVOKE` (unauthorized statements map to SQLSTATE
`42501`), and a JSONL security audit log recording authentication, authorization, and admin events.
AES-256-GCM encryption at rest with pluggable key management (local file, environment, external command,
HashiCorp Vault).

**Interfaces** — PostgreSQL wire protocol (simple **and** extended query protocol / prepared statements,
with a parsed-statement cache), `COPY FROM STDIN` / `COPY TO STDOUT` for bulk ingestion, Python client
(embedded and remote), `pg_catalog` compatibility, local backup/restore, and full observability
(`/health`, `/metrics`, tracing).

---

## In progress

Completing the open-source engine for production and managed-cloud use. Follow the linked issues for status.

### Security
- Native cloud KMS key providers (AWS KMS, GCP KMS, Azure Key Vault) over REST

### Durability and completeness
- Backup and restore to object storage (S3, GCS, Azure Blob, S3-compatible) over REST

---

## Planned

- Semantic query caching (`CREATE SEMANTIC CACHE`)
- Gradient-driven adaptive storage (single-node)
- Active-learning SQL: `FEEDBACK`, `ORDER BY ACTIVE_LEARNING()`, uncertainty and drift detection
- Versioned vector-index snapshots (`SEMANTIC_SNAPSHOT`)
- Serializable Snapshot Isolation
- Disk-resident ANN for larger-than-RAM vector sets

---

## Enterprise edition

Built on the open-source core through stable extension interfaces. The open core never depends on
enterprise code.

- Distributed clustering with Raft replication and consistent-hash sharding
- Distributed approximate nearest-neighbor search (correct global top-K)
- Cross-shard consistency and HTAP read replicas
- Automated storage tiering across NVMe, object storage, and cold archive
- SSO (OIDC / SAML), fine-grained RBAC, and audit logging
- Federated queries with differential privacy

For enterprise access, contact the team via the repository.

---

## Shaping the roadmap

Priorities are driven by what production users need most. Open an
[issue](https://github.com/zentrix-innovative-labs/galaxdb/issues/new/choose) or start a
[discussion](https://github.com/zentrix-innovative-labs/galaxdb/discussions) to request a feature or tell
us what matters for your workload.
