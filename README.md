<div align="center">
  <img src="assets/GalaxDB-avatar.svg" alt="GalaxDB" width="120" />
  <h1>GalaxDB</h1>
  <p><strong>The AI-native database. SQL + vector search + training exports in one system.</strong></p>

  [![CI](https://github.com/zentrix-innovative-labs/galaxdb/actions/workflows/ci.yml/badge.svg)](https://github.com/zentrix-innovative-labs/galaxdb/actions/workflows/ci.yml)
  [![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
  [![Docker](https://img.shields.io/badge/docker-harbi256%2Fgalaxdb-blue)](https://hub.docker.com/r/harbi256/galaxdb)
  [![PyPI](https://img.shields.io/pypi/v/galaxdb-client.svg)](https://pypi.org/project/galaxdb-client/)
</div>

---

## What is GalaxDB?

Most AI applications bolt together 3–5 separate services: a relational database, a vector database, an embedding API, an object store, and a data pipeline. GalaxDB replaces all of them with a single binary that speaks PostgreSQL wire protocol.

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

## Performance

Measured on AWS c6id.4xlarge (Intel Xeon Platinum 8375C, 16 vCPU, 32 GiB RAM, 884 GB NVMe), release build.

### HNSW Vector Search — SIFT-1M

| ef_search | recall@10 | mean latency | p99 latency |
|-----------|-----------|--------------|-------------|
| 50        | 0.959     | 158 µs       | 228 µs      |
| 100       | 0.983     | 267 µs       | 364 µs      |
| **200**   | **0.990** | **459 µs**   | **616 µs**  |

### Storage Engine

| Metric | GalaxDB | PostgreSQL 16 | RocksDB |
|--------|---------|---------------|---------|
| Write TPS | **258,555** | ~3,200 | ~80,000 |
| Read p50 | **3 µs** | ~95 µs | ~180 µs |
| Read p99 | **47 µs** | ~300 µs | ~500 µs |
| Scan throughput | **4.49 GB/s** | ~0.9 GB/s | — |

**740 Rust tests passing. 7 chaos scenarios in 10.9 s.** See [BENCHMARKS.md](docs/BENCHMARKS.md).

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

### GitHub Releases

Download pre-built binaries for Linux x86-64 and macOS x86-64 from the [Releases page](https://github.com/zentrix-innovative-labs/galaxdb/releases).

### Rust (embed in your application)

```toml
[dependencies]
galaxdb-embedded = "1.0.0-beta"
```

---

## Observability

Every server instance exposes:

```bash
# Health check — reflects real subsystem state
curl http://localhost:9090/health
# {"status":"ok","version":"1.0.0-beta.1","subsystems":{"disk_full":false,"sidecar_healthy":true,"connections_active":3}}

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
```

---

## Documentation

- [Getting Started](docs/GETTING_STARTED.md) — installation, all features, Docker Compose, troubleshooting
- [SQL Reference](docs/sql-reference.md) — full AuroraSQL syntax
- [Storage Engine](docs/STORAGE_ENGINE.md) — LSM tree, WAL, PAX blocks, HNSW
- [Benchmarks](docs/BENCHMARKS.md) — SIFT-1M recall, write throughput, latency
- [Database Comparison](docs/COMPARISON.md) — GalaxDB vs PostgreSQL, Pinecone, Qdrant, LanceDB, ChromaDB, Milvus, DuckDB, Weaviate

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Open an issue first for large changes. All PRs must pass the full test suite and three CI gates (no mocks, no vendor SDKs, task tracker).

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

<div align="center">
  <sub>Built by <a href="https://zentrix.ai">Zentrix Innovative Labs</a></sub>
</div>
