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
AES-256-GCM encryption at rest with pluggable key management: local file, environment, external command,
HashiCorp Vault, and native cloud KMS over REST — AWS KMS, GCP Cloud KMS, and Azure Key Vault
(credential-gated integration tests).

**Interfaces** — PostgreSQL wire protocol (simple **and** extended query protocol / prepared statements,
with a parsed-statement cache), `COPY FROM STDIN` / `COPY TO STDOUT` for bulk ingestion, Python client
(embedded and remote), `pg_catalog` compatibility, and full observability (`/health`, `/metrics`, tracing).

**Backup and restore** — `BACKUP TO` / `RESTORE FROM` to a local path or object storage over REST (no
vendor SDKs): `s3://` (plus S3-compatible endpoints — MinIO, Cloudflare R2), `gs://` (GCS), and `az://`
(Azure Blob), with checksum validation and restore-aborts-on-corruption.

The single-node open-source engine is **feature-complete** as of the v0.4.0 line: the entire relational +
analytical + vector + transactional surface above ships today. Remaining OSS work is verification and
published evidence, not missing features.

---

## In progress

Hardening and evidence for the shipped engine — not new capabilities.

- Broader PostgreSQL driver-compatibility matrix: live end-to-end tests against psql, psycopg,
  JDBC, and SQLAlchemy (tokio-postgres is covered today), and fuller `pg_catalog` table/column listing.
- Published ClickBench-style analytical + TPC-H-subset benchmarks on named hardware (SIFT-1M vector
  recall and the full test suite are already published in [BENCHMARKS.md](docs/BENCHMARKS.md)).
- Merkle-DAG-targeted block pruning for `AT VERSION` time-travel (a performance optimization; results
  are already correct via commit-timestamp filtering).

---

## Planned (v2 and research)

Future capabilities beyond the v0.4.0 single-node engine.

- Semantic query caching (`CREATE SEMANTIC CACHE`)
- Gradient-driven adaptive storage (single-node)
- Active-learning SQL: `FEEDBACK`, `ORDER BY ACTIVE_LEARNING()`, uncertainty and drift detection
- Versioned vector-index snapshots (`SEMANTIC_SNAPSHOT`)
- Serializable Snapshot Isolation (snapshot isolation ships today)
- Disk-resident ANN for larger-than-RAM vector sets

---

## Client SDKs and language support

GalaxDB speaks the PostgreSQL wire protocol, so **any language with a Postgres
driver already connects to the server today** — no GalaxDB-specific SDK is
required for remote use. The Python client (`galaxdb-client`) is special
because it also offers **embedded** mode (the engine in-process via PyO3),
which is the only capability that needs per-language FFI work.

Planned, ranked by value/cost:

1. **Publish the Rust crates to crates.io** (`galaxdb-embedded`, `galaxdb-sql`,
   and their dependencies). Near-zero cost — they already compile and version
   together. Rust is the native language, so embedded mode is free here, and it
   lets other Rust tools build directly on the engine. Do this first.
2. **"Connect from any language" docs page.** A five-line snippet per popular
   driver — Go (`pgx`), Node (`pg`), Java/Kotlin (JDBC), .NET (`Npgsql`),
   Ruby (`pg`) — plus the GalaxDB SQL extensions (`SEMANTIC_MATCH`,
   `EMBEDDING MODEL`). Highest ROI: pure docs, unlocks the whole ecosystem,
   and sets correct expectations that no SDK is needed for remote use.
3. **Thin TypeScript/Node SDK** (wrapper over `pg`, remote-only, no FFI). The
   AI/RAG/agent audience is overwhelmingly on TS/Node, matching GalaxDB's
   positioning. Typed helpers: `db.semanticMatch(col, query, threshold, {limit})`,
   `EMBEDDING MODEL` DDL builders, connection helpers.
4. **Thin Go module** (wrapper over `pgx`, remote-only). For the infra/platform
   teams who self-host the server. Build if backend/infra pull materializes.

Explicitly **not** planned near-term: embedded (in-process) bindings for
JS/Go/Java/.NET — cgo/JNI/napi plus per-platform binary distribution is a large,
permanent maintenance cost that the wire protocol already makes unnecessary for
remote use.

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

## Changelog

Now that the open-source engine is feature-complete, each release is tracked
here. Dates are release-tag dates. Versions follow semver; the PyPI client
(`galaxdb-client`), the Docker image (`harbi256/galaxdb`), and the server
binaries share the same version.

### v0.6.0

**Added — usage-metering metrics on `/metrics` (E-4).** The engine now exposes
neutral, billing-grade operational counters and capacity gauges so a control
plane can meter usage by scraping port 9090 (no auth change). New cumulative
counters: `galaxdb_read_ops_total`, `galaxdb_write_ops_total`,
`galaxdb_vector_ops_total` (disjoint from reads), `galaxdb_embedding_ops_total`
(per row), `galaxdb_near_dedup_rows_total`, and `galaxdb_training_export_bytes_total`.
New gauges: `galaxdb_storage_bytes` (physical on-disk size, post-compaction),
`galaxdb_rows_total`, and `galaxdb_process_start_time_seconds`. Op counting is
**one statement = one op** — reads/vector searches are counted at the executor
(the single dispatch every statement funnels through), writes once per statement
at the ingress above the per-row `INSERT` fan-out, so a 10k-row `INSERT` and a
10k-row `COPY` are each one write op. See `docs/METRICS.md` for exact definitions.

**Added — restart-durable counters.** The six cumulative counters persist to
`<data_dir>/metering.gmet` using the v0.5 versioned-header + crash-safe
`atomic_replace` machinery (a mid-write crash leaves the prior or the new totals,
never a torn value; a too-new file is refused with a typed error). They are
seeded back on open and flushed on every checkpoint and on graceful shutdown, so
scale-to-zero stop/start never resets usage to zero. `galaxdb_process_start_time_seconds`
lets a collector detect a restart and reconcile the unpersisted tail.

### v0.5.0

**Added — multi-architecture embedding models.** The sidecar is no longer a
single hardcoded BERT path. Models are selected at runtime by HuggingFace id
through a `TextEmbedder` trait + `ModelSpec` registry, each carrying its own
pooling, instruction/document prefixes, native dimension, and license. The
launch set spans five architectures, all verified against real weights on CPU:
`all-MiniLM-L6-v2` (BERT/mean, default), `BGE-M3` (XLM-RoBERTa/CLS, 1024-d,
multilingual), `Qwen3-Embedding` 0.6B/4B/8B (decoder/last-token, one loader,
up to 4096-d), and two custom bidirectional encoders — **EmbeddingGemma-300M**
(Gemma 3 made bidirectional + sentence-transformers Dense heads + mean pooling)
and **LFM2.5-Embedding-350M** (non-causal short-conv + bidirectional GQA, CLS).
Query vs document embedding is now distinguished over the sidecar protocol so
asymmetric models retrieve correctly. An unknown/unsupported model id is a
typed error + exit — never a silent substitution. See `docs/EMBEDDING_MODELS.md`.

**Added — upgrade-safe on-disk format versioning.** Every persistent artifact
(WAL, SST, PAX block, blob log, catalog, HNSW index) now carries an explicit,
range-checked format version. A newer engine reads data written by older
formats (backward-compatible), and any format **newer** than the running binary
supports is *refused* with a typed error rather than mis-read — the guarantee
that makes rolling a binary back on the same volume safe. Adds a crash-safe
upgrade-on-open primitive (write-new → fsync → atomic rename → fsync-dir) with
crash-injection tests, and a cross-version integration test proving forward
reads + newer-format refusal through the real engine open path. The PostgreSQL
wire framing is pinned by a contract test so patch releases stay
client-compatible.

**Fixed — semantic search survives a restart.** The per-table vector index
(HNSW + delta buffer) lived only in memory. After a server restart the row data
survived (WAL/SST) but the index did not, so `SEMANTIC_MATCH` on a recovered
table failed with "table not found" — semantic search silently broke on every
restart. On open, the engine now rebuilds each embedding table's index by
re-embedding its durable rows through the attached model (deterministic: the
same model reproduces the same vectors, so results are identical to before the
restart). Verified end-to-end on a real 600-row AG News dataset: precision holds
(0.90) across a restart.

**Fixed — sidecar restart race.** A stale `sidecar.sock` left on the data volume
by the previous run could make the engine try to embed before the freshly
spawned sidecar was accepting connections, causing a "connection refused" panic
on restart. Attaching the sidecar now waits for it to actually answer before
proceeding, so the socket file existing is no longer mistaken for readiness.

**Fixed — SST cross-version corruption path.** The SST registry silently
swallowed every footer-parse error into a single-block legacy fallback, so an
SST footer written by a *newer* engine would be mis-read as one giant legacy
block instead of refused. It now propagates the typed too-old / too-new format
errors and falls back only for a genuine legacy no-footer SST.

**Compatibility.** Existing v0.4 databases open unchanged; the default model
stays `all-MiniLM-L6-v2`. Legacy (pre-versioning) WAL/SST/blob files are read
as format v1 and migrated to the versioned layout on the next rewrite.

### v0.4.0

**Fixed — semantic search over the wire.** `SEMANTIC_MATCH` returned zero rows
for tables loaded through the server (PostgreSQL wire protocol). The concurrent
`INSERT` path and the `COPY`/bulk-load path computed embeddings via the sidecar
but never stored the resulting vectors in the index, so queries searched an
empty index. Embeddings are now populated through a single `on_row_inserted`
backend hook shared by every write path, verified end-to-end on a real
7,600-row dataset (semantic precision 0.70–1.00 across categories).

**Fixed — `SEMANTIC_MATCH ... LIMIT n`.** The result set was capped at 10 rows
regardless of `LIMIT`, and `LIMIT > 100` was silently truncated. A `LIMIT n`
now returns the *n* nearest matches; a bare `LIMIT` stays on the native
similarity-ranked path, while `ORDER BY` / `GROUP BY` / `JOIN` route to the
analytical engine. Without a `LIMIT`, the default page is the 10 nearest.

**Fixed — `SHOW EMBEDDING HEALTH FOR <table>`.** Returned a canned echo string
over the wire; now reports the real sidecar state and model version.

**Changed — release automation.** The Homebrew formula is regenerated from the
published release checksums (`scripts/gen-homebrew-formula.sh`) and committed by
the release workflow; `install.sh` resolves the latest release at runtime
instead of pinning a version.

### v0.3.0

Open-source single-node engine declared **feature-complete**: the full
relational + analytical + vector + transactional surface, security, and
encryption at rest. First distribution across all channels — PyPI
`galaxdb-client`, the `harbi256/galaxdb` Docker image, and GitHub release
binaries for Linux/macOS (x86_64 + aarch64) and Windows.

### v0.2.0

Server binary distribution: GitHub release binaries, the `install.sh`
one-liner, and the Homebrew formula.

### v0.1.x

Initial Python client releases on PyPI.

---

## Shaping the roadmap

Priorities are driven by what production users need most. Open an
[issue](https://github.com/zentrix-innovative-labs/galaxdb/issues/new/choose) or start a
[discussion](https://github.com/zentrix-innovative-labs/galaxdb/discussions) to request a feature or tell
us what matters for your workload.
