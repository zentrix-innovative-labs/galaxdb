<div align="center">
  <img src="assets/Icon.svg" alt="GalaxDB" width="140" />
  <h1>GalaxDB</h1>
  <p><strong>The AI-native database. SQL + vector search + training exports in one system.</strong></p>

  [![CI](https://github.com/zentrix-innovative-labs/galaxdb/actions/workflows/ci.yml/badge.svg)](https://github.com/zentrix-innovative-labs/galaxdb/actions/workflows/ci.yml)
  [![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
  [![Docker](https://img.shields.io/badge/docker-harbi256%2Fgalaxdb-blue)](https://hub.docker.com/r/harbi256/galaxdb)
  [![PyPI](https://img.shields.io/pypi/v/galaxdb-client.svg)](https://pypi.org/project/galaxdb-client/)
  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20355229.svg)](https://doi.org/10.5281/zenodo.20355229)
</div>

---

## What is GalaxDB?

Most AI applications bolt together 3–5 separate services: a relational database, a vector database, an embedding API, an object store, and a data pipeline. GalaxDB replaces all of them with a single binary that speaks PostgreSQL wire protocol. It's still one binary and one dependency to deploy — `galaxdb-server` is around 60 MB (it statically links the embedded analytical query engine, Apache Arrow, and the Lance training-export format) and `galaxdb-sidecar` is a separate, optional process for local embedding inference.

```
Before GalaxDB:
  PostgreSQL + pgvector + Pinecone + OpenAI API + S3 + Airflow

After GalaxDB:
  galaxdb-server
```

One connection string. One backup. One monitoring endpoint. Your existing `psycopg2`, `SQLAlchemy`, and `pg` code works unchanged.

---

## Quick Start

### Python — embedded mode (no server, like SQLite)

```bash
pip install galaxdb-client
```

```python
import galaxdb

db = galaxdb.Database("./mydata")

# Create a table with an embedding column
db.execute("""
    CREATE TABLE docs (
        id   INT PRIMARY KEY,
        text TEXT EMBEDDING MODEL 'sentence-transformers/all-MiniLM-L6-v2' DIM 384
    )
""")

# Insert — embeddings computed automatically by the local sidecar
db.execute("INSERT INTO docs (id, text) VALUES (1, 'machine learning is great')")
db.execute("INSERT INTO docs (id, text) VALUES (2, 'rust programming language')")
db.execute("INSERT INTO docs (id, text) VALUES (3, 'deep neural networks')")

# Semantic search — no external API, no separate vector DB
results = db.execute(
    "SELECT id, text FROM docs WHERE SEMANTIC_MATCH(text, 'AI and neural nets', 0.7)"
)

# Export a training dataset — one SQL command, Lance format, PyTorch-ready
db.execute("CREATE VERSION TAG 'v1' FOR TRAINING WITH TRAINING PRECISION 'float32'")
path = db.training_dataset("v1")

import lance
dataset = lance.dataset(path).to_pytorch()  # zero-copy, memory-mapped
```

### Server mode — multi-client, like PostgreSQL

```bash
# macOS
brew tap zentrix-innovative-labs/tap && brew install galaxdb

# Linux / macOS (direct install)
curl -fsSL https://raw.githubusercontent.com/zentrix-innovative-labs/galaxdb/main/install.sh | bash

# Docker
docker run -p 5433:5433 -p 9090:9090 -v /data:/data \
  harbi256/galaxdb:latest --data-dir /data
```

```python
import galaxdb

conn = galaxdb.connect("host=localhost port=5433 dbname=galaxdb sslmode=disable")
conn.execute("SELECT id, text FROM docs WHERE SEMANTIC_MATCH(text, 'AI', 0.8)")
```

Any PostgreSQL client works — `psycopg2`, `SQLAlchemy`, `tokio-postgres`, `pg` (Node.js), JDBC.

---

## AuroraSQL — SQL Extensions for AI

GalaxDB extends standard SQL with AI-native primitives:

```sql
-- Semantic search with similarity threshold
SELECT id, title
FROM articles
WHERE SEMANTIC_MATCH(title, 'climate change policy', 0.75)
  AND published_at > '2024-01-01';

-- Top-k nearest matches — add LIMIT to control how many come back
-- (without a LIMIT, the default is the 10 nearest)
SELECT id, title
FROM articles
WHERE SEMANTIC_MATCH(title, 'climate change policy', 0.5)
LIMIT 50;

-- Time-travel query — reproduce exactly what data existed at a point in time
SELECT * FROM docs AT VERSION 'training-v1';

-- Near-duplicate deduplication — cut training set size by 15–30%
SELECT * FROM docs WHERE NOT DUPLICATE;

-- Create a versioned training snapshot
CREATE VERSION TAG 'train-v2'
  FOR TRAINING
  WITH TRAINING PRECISION 'sq8'
  TRAINING SEED 42;

-- Bulk insert
BULK INSERT INTO docs (id, text) VALUES
  (1, 'first document'),
  (2, 'second document');

-- Backup and restore
BACKUP TO '/path/to/backup';
RESTORE FROM '/path/to/backup';
```

---

## It's a real SQL database

Alongside the AI primitives, GalaxDB is a transactional, analytical relational engine — not a
vector store with a SQL veneer.

```sql
-- Transactions with snapshot isolation, read-your-writes, and savepoints
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;   -- expressions are evaluated
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
SAVEPOINT before_fee;
UPDATE accounts SET balance = balance - 5 WHERE id = 1;
ROLLBACK TO before_fee;
COMMIT;

-- Analytical queries: joins, aggregates, GROUP BY / HAVING, DISTINCT, ORDER BY / LIMIT
SELECT d.name, COUNT(*), AVG(e.salary)
FROM employees e JOIN departments d ON e.dept = d.name
GROUP BY d.name
HAVING COUNT(*) > 1
ORDER BY AVG(e.salary) DESC
LIMIT 10;
```

Constraints are enforced (a duplicate `PRIMARY KEY` is rejected with SQLSTATE `23505`, never a
silent overwrite), arithmetic errors are typed (`22012` division-by-zero, `22003` overflow), and
the schema is **durable** — tables, storage modes, and constraints survive a restart. Analytical
queries run on an embedded DataFusion engine; single-table point reads, scans, and vector search
run on the native path.

---

## Performance

Measured on AWS c6id.4xlarge (Intel Xeon Platinum 8375C, 16 vCPU, 32 GiB RAM, 884 GB NVMe), release build.

### HNSW Vector Search — SIFT-1M

| ef_search | recall@10 | mean latency | p99 latency |
|-----------|-----------|--------------|-------------|
| 50        | 0.959     | 157 µs       | 229 µs      |
| 100       | 0.983     | 267 µs       | 364 µs      |
| **200**   | **0.990** | **459 µs**   | **612 µs**  |

For methodology and the full SIFT-1M run, see the [GalaxDB paper on Zenodo](https://doi.org/10.5281/zenodo.20355229).

### Storage Engine — durable write path

Measured with GalaxDB and PostgreSQL 16.14 on the **same instance-store NVMe** (PostgreSQL's data directory relocated to the NVMe), `fsync=on`, both using prepared statements — an apples-to-apples comparison.

| Workload | GalaxDB | PostgreSQL 16 |
|----------|---------|---------------|
| Concurrent INSERT, 1 client | 10,450 rows/s | 11,891 rows/s |
| Concurrent INSERT, 4 clients | 30,468 rows/s | 34,298 rows/s |
| Concurrent INSERT, 8 clients | 36,632 rows/s | 54,432 rows/s |
| Concurrent INSERT, 16 clients | 37,448 rows/s | 84,747 rows/s |
| `COPY` bulk load | 190,287 rows/s (17.1 MiB/s) | — |

On durable single-client and low-concurrency writes GalaxDB is competitive with PostgreSQL (within ~12%); PostgreSQL's mature process-per-connection model still scales better past 8 concurrent clients. Bulk ingestion via `COPY` reaches 190k rows/s (25.97× row-by-row INSERT). The in-memory storage path (memtable + ART) sustains ~1.9M rows/s when an fsync is amortized across a batch. See [BENCHMARKS.md](docs/BENCHMARKS.md) for reproduction commands.

**823 Rust tests passing (`--release`, AWS c6id.4xlarge, commit `6c7811f`).**


---

## How It Compares

| | GalaxDB | PostgreSQL + pgvector | Pinecone | Qdrant | Weaviate | LanceDB | ChromaDB | Milvus | DuckDB |
|---|---|---|---|---|---|---|---|---|---|
| SQL queries | ✅ Full | ✅ Full | ❌ | ❌ | Partial | Partial¹ | ❌ | ❌ | ✅ Full |
| Vector search | ✅ recall=0.990 | ⚠️ ~0.95 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Local embeddings | ✅ no API cost | ❌ | ❌ | ⚠️ FastEmbed | ✅ modules | ✅ | ✅ | ❌ | ❌ |
| Time-travel | ✅ `AT VERSION` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Training export | ✅ Lance format | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Near-dedup | ✅ MinHash LSH | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Embedded mode | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ |
| PostgreSQL wire | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Self-hosted | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Encryption at rest | ✅ AES-256-GCM | ✅ OS-level | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| MVCC / snapshots | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Single binary | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ |

¹ LanceDB OSS uses a Python/Arrow API; SQL is available via DuckDB bridge or Enterprise tier only.

→ [Full comparison with benchmarks, pricing, and use-case guidance](docs/COMPARISON.md)

---

## Architecture

```
Your application
      │
      │  PostgreSQL wire protocol (port 5433)
      │  or Python embedded API
      ▼
┌─────────────────────────────────────────────────────┐
│                   galaxdb-server                    │
│                                                     │
│  SQL Parser → Query Planner → Executor              │
│       │                           │                 │
│  ART index    HNSW graph    LSM storage engine      │
│  (point reads) (vector search) (WAL + PAX blocks)   │
│                                                     │
│  ┌──────────────────┐   HTTP :9090                  │
│  │ galaxdb-sidecar  │   /health  /metrics           │
│  │ (child process)  │                               │
│  │ ONNX/Candle model│                               │
│  └──────────────────┘                               │
└─────────────────────────────────────────────────────┘
```

The sidecar is spawned automatically — you don't manage it separately.

---

## Use Cases

**RAG applications** — store documents, compute embeddings locally, query with `SEMANTIC_MATCH` filtered by metadata. No Pinecone, no OpenAI embeddings API.

**ML training pipelines** — `CREATE VERSION TAG ... FOR TRAINING` snapshots your data and exports it as a Lance dataset. Load directly into PyTorch with zero-copy memory mapping.

**Hybrid search** — combine SQL filters with vector similarity in a single query. No application-side join between two systems.

**Audit-safe AI** — `AT VERSION` queries let you reproduce exactly what data a model was trained on. EU AI Act compliance built in.

**Time-series + semantic** — store sensor readings with text descriptions, query by time range AND semantic similarity in one SQL statement.

---

## Installation

### Python (embedded + remote)

```bash
pip install galaxdb-client
```

Requires Python 3.9+. Pre-built wheels for Linux x86-64, macOS Intel, macOS Apple Silicon, and Windows x86-64.

### macOS (Homebrew)

```bash
brew tap zentrix-innovative-labs/tap
brew install galaxdb
```

### Linux / macOS (direct install)

```bash
curl -fsSL https://raw.githubusercontent.com/zentrix-innovative-labs/galaxdb/main/install.sh | bash
```

### Docker

```bash
docker run -p 5433:5433 -p 9090:9090 -v /data:/data \
  harbi256/galaxdb:latest --data-dir /data
```

### Windows

- **Python (embedded + client):** `pip install galaxdb-client` — a native `win_amd64` wheel is
  published, so embedded mode and the remote client work on Windows out of the box.
- **Server:** run it under **Docker Desktop (WSL2 backend)** with the Docker command above, or use
  the native `galaxdb-server-windows-x86_64.exe` attached to each release. The
  server is cross-platform Rust (rustls, no OpenSSL; Linux-only io_uring falls back to tokio), and CI
  builds `galaxdb-server` on a native Windows runner on every change. The relational, analytical,
  transactional, storage, and vector-search engine is fully native on Windows.
- **Embeddings on Windows:** the embedding sidecar (`EMBEDDING MODEL` columns and live
  `SEMANTIC_MATCH` generation) is Unix-only — it uses Unix-domain sockets — so run it under WSL2 or
  Docker. Vector search over already-computed vectors works natively.

### GitHub Releases

Download pre-built `galaxdb-server` binaries for Linux (x86-64, aarch64), macOS (x86-64, arm64), and
Windows x86-64 from the [Releases page](https://github.com/zentrix-innovative-labs/galaxdb/releases).

### Rust (embed in your application)

```toml
[dependencies]
galaxdb-embedded = "0.4.0"
```

---

## Observability

Every server instance exposes:

```bash
# Health check — reflects real subsystem state
curl http://localhost:9090/health
# {"status":"ok","version":"0.4.0","subsystems":{"disk_full":false,"sidecar_healthy":true,"connections_active":3}}

# Prometheus metrics
curl http://localhost:9090/metrics
# galaxdb_connections_active 3
# galaxdb_wal_write_latency_us 42
# galaxdb_hnsw_recall_estimate_bp 9902
# galaxdb_embedding_queue_depth 0
# ...
```

---

## Key Management

GalaxDB supports pluggable encryption key management with no vendor lock-in:

```bash
# Local key file
GALAXDB_KEY_PROVIDER=local:/path/to/key.bin galaxdb-server ...

# Environment variable
GALAXDB_KEY_PROVIDER=env:GALAXDB_MASTER_KEY galaxdb-server ...

# Any KMS via shell command (AWS CLI, gcloud, az, vault, custom HSM)
GALAXDB_KEY_PROVIDER=command:aws kms decrypt ... galaxdb-server ...

# HashiCorp Vault Transit
GALAXDB_KEY_PROVIDER=vault:transit/galaxdb-prod galaxdb-server ...

# Native cloud KMS over REST (no vendor SDK; build with the `cloud-kms` feature)
GALAXDB_KEY_PROVIDER=aws-kms:alias/galaxdb galaxdb-server ...
GALAXDB_KEY_PROVIDER=gcp-kms:projects/p/locations/global/keyRings/r/cryptoKeys/k galaxdb-server ...
GALAXDB_KEY_PROVIDER=azure-kv:my-vault/my-key galaxdb-server ...
```

---

## Security status

GalaxDB encrypts data at rest (AES-256-GCM on every PAX block and WAL record, pluggable key
management above) and secures client connections with SCRAM-SHA-256 authentication, TLS 1.2/1.3
transport encryption (rustls, no OpenSSL), and role-based access control.

| Capability | Status |
|------------|--------|
| Encryption at rest (AES-256-GCM, pluggable KMS) | ✅ Available now |
| Wire authentication (SCRAM-SHA-256) | ✅ Available now |
| TLS transport encryption (TLS 1.2/1.3, rustls) | ✅ Available now |
| Roles, privileges, GRANT/REVOKE (SQLSTATE 42501) | ✅ Available now |
| JSONL security audit log (authN/authZ/DDL/admin) | ✅ Available now |
| Native cloud KMS (AWS/GCP/Azure over REST) | ✅ Available now |
| SSO / fine-grained RBAC | Enterprise edition |

Authentication is enabled with `--auth` (or `GALAXDB_AUTH=1`); the initial superuser is provisioned
from `GALAXDB_INITIAL_SUPERUSER[_PASSWORD]` on first start — no default password ships. With
`tls_mode=require`, plaintext connections are rejected and SCRAM runs inside the TLS channel. When
auth is disabled the server runs in trusted-local mode and logs a loud startup warning. See
[ROADMAP.md](ROADMAP.md) for what is shipping next.

---

## Documentation

- [Getting Started](docs/GETTING_STARTED.md) — installation, all features, Docker Compose, troubleshooting
- [Roadmap](ROADMAP.md) — shipped capabilities, in-progress hardening, and planned features (OSS vs Enterprise)
- [SQL Reference](docs/sql-reference.md) — full AuroraSQL syntax
- [Storage Engine](docs/STORAGE_ENGINE.md) — LSM tree, WAL, PAX blocks, HNSW
- [Benchmarks](docs/BENCHMARKS.md) — SIFT-1M recall, write throughput, latency
- [Database Comparison](docs/COMPARISON.md) — GalaxDB vs PostgreSQL, Pinecone, Qdrant, LanceDB, ChromaDB, Milvus, DuckDB, Weaviate
- [Research Paper](https://doi.org/10.5281/zenodo.20355229) — GalaxDB: A Unified AI-Native Storage Engine for Transactional, Analytical, and Vector Workloads

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Open an issue first for large changes. All PRs must pass the full test suite and three CI gates (no mocks, no vendor SDKs, task tracker).

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

<div align="center">
  <sub>Built by Zentrix Innovative Labs</sub>
</div>
