<p align="center">
	<img src="assets/GalaxDB-avatar.svg" alt="GalaxDB avatar" width="180" />
</p>

<h1 align="center">GalaxDB</h1>

<p align="center">
	<strong>The AI-native database.</strong><br />
	One engine. One binary. One SQL dialect.<br />
	From your laptop to a planet-scale cluster.
</p>

<p align="center">
	<img src="https://img.shields.io/badge/status-implementing_v1-1f6feb" alt="status" />
	<img src="https://img.shields.io/badge/license-Apache--2.0-333333" alt="license" />
</p>

---

## The Problem

Every AI team today stitches together five or more systems to serve a single model:

| System | Role |
|--------|------|
| PostgreSQL | Transactional user data |
| Pinecone / Qdrant / Weaviate | Vector similarity search |
| Redis | Caching and session state |
| S3 / Data Lake | Raw assets and logs |
| Feast / Tecton | Pre-computed model features |

This **five-database spaghetti** creates real pain:

- **Stale embeddings** — vectors lag behind transactional updates, silently poisoning model accuracy.
- **Operational hell** — five systems to deploy, monitor, back up, and scale. Five failure modes.
- **Slow iteration** — 30–50% of data science time goes to data plumbing, not modelling.
- **Irreproducible training** — no unified point-in-time snapshot means training sets become a forensic mystery.
- **No feedback loops** — model errors never flow back into the database for curation.

Existing databases were designed for a pre-AI world. Bolting vector search onto PostgreSQL or MongoDB doesn't fix the architectural mismatch.

---

## The Solution

GalaxDB is a **ground-up redesign of the database for the AI era**. It unifies transactional, analytical, and vector workloads into a single engine that actively improves the AI built on top of it.

```sql
-- One table holds relational data, embeddings, and lineage
CREATE TABLE products (
    id          BIGINT PRIMARY KEY,
    title       TEXT,
    description TEXT EMBEDDING MODEL 'all-MiniLM-L6-v2' DIM 384,
    price       DECIMAL,
    created_at  TIMESTAMP
);

-- Hybrid search: structured filters + semantic similarity in one query
SELECT title, price, similarity
FROM products
WHERE price < 100
  AND SEMANTIC_MATCH(description, 'lightweight camping tent', 0.7)
ORDER BY similarity DESC
LIMIT 20;

-- Time-travel: query data as it existed last month
SELECT * FROM products AT VERSION 'q4_snapshot' WHERE price < 50;

-- Training export: one command produces a PyTorch-ready dataset
CREATE VERSION TAG 'training_v3' FOR TRAINING
  WITH TRAINING PRECISION 'sq8' TRAINING SEED 42;
```

```python
import galaxdb

db = galaxdb.Database("./mydata")
dataset = db.training_dataset("training_v3")  # → PyTorch IterableDataset, zero-copy
```

**One binary. One SQL surface. One system to operate.**

---

## What Makes GalaxDB Different

### vs. PostgreSQL + pgvector

pgvector bolts vector search onto a row-store designed in the 1980s. GalaxDB was built from scratch with embeddings, versioning, and training export as first-class primitives — not extensions. The result: unified transactions across rows and vectors, built-in embedding generation, Merkle DAG versioning for reproducible training, and semantic guardrails that prevent silent data poisoning.

### vs. Pinecone / Qdrant / Weaviate

Dedicated vector databases are great at ANN search but force you to maintain a separate transactional database alongside them. GalaxDB eliminates that split entirely — OLTP, OLAP, and vector search live in one engine with one consistency model. No data synchronization pipelines, no stale embeddings, no operational overhead of running two systems.

### vs. MongoDB Atlas Vector Search

MongoDB retrofitted vector search onto a document store. GalaxDB's hybrid PAX storage is purpose-built for mixed workloads: columnar scans for analytics, row-level access for transactions, and mmap'd HNSW for vector search — all in one storage format. Add Merkle DAG versioning and training-aware export, and you have capabilities MongoDB simply doesn't offer.

### vs. LanceDB

LanceDB is an embedded vector database focused on the Lance format. GalaxDB is a full database engine with OLTP transactions, SQL, PostgreSQL wire compatibility, and a built-in embedding sidecar. Lance is one of our export formats — we use it for training data materialization.

### vs. SingleStore / TiDB

HTAP systems unify OLTP and OLAP but have no native vector search, no embedding generation, no versioned training export, and no semantic guardrails. GalaxDB adds the AI-native layer that HTAP systems are missing.

---

## Key Capabilities

| Capability | What It Does |
|-----------|-------------|
| **Hybrid SQL + Vector Search** | `SEMANTIC_MATCH` in standard SQL. Structured filters and similarity search in one query, one transaction. |
| **Built-in Embedding Generation** | Declare `EMBEDDING MODEL` in DDL. The engine generates embeddings automatically — no external microservices. |
| **Merkle DAG Versioning** | Every commit has a cryptographic hash. `AT VERSION` queries, named tags, and pinned training snapshots out of the box. |
| **Semantic Guardrails** | The engine rejects ambiguous time-travel + vector queries by default. No silent data poisoning in your training pipeline. |
| **Training-Optimized Export** | One SQL command produces a Lance-format dataset with quantized embeddings, near-duplicate exclusion, and a full audit trail (EU AI Act compliant). |
| **Zero-Copy PyTorch Integration** | `galaxdb.training_dataset(tag)` returns a PyTorch `IterableDataset` with zero deserialization overhead. |
| **Near-Duplicate Detection** | MinHash LSH signatures computed at write time. `WHERE NOT DUPLICATE` cuts training set size by 15–30% without quality loss. |
| **Model Version Tracking** | The engine tracks which model version produced each embedding. Model upgrades trigger automatic re-embedding — no stale vectors. |
| **PostgreSQL Compatible** | Connect with psycopg2, SQLAlchemy, and standard tooling. No new client libraries to learn. |
| **Encryption at Rest** | AES-256-GCM transparent data encryption with AWS KMS. TLS 1.3 for all connections. GDPR/HIPAA ready. |

---

## Benchmarks

GalaxDB v1 will be benchmarked against six systems under identical hardware and dataset conditions:

| System | Category |
|--------|----------|
| PostgreSQL 18 + pgvector | Relational + vector extension |
| Qdrant | Dedicated vector database |
| Weaviate | Dedicated vector database |
| Milvus | Dedicated vector database |
| SQLite | Embedded relational |
| DuckDB | Embedded analytical |

### Workloads

| Workload | What It Measures |
|----------|-----------------|
| **OLTP Row** | 70% point reads, 20% updates, 10% inserts — TPS, p50/p95/p99 latency |
| **ANN Retrieval** | k=10 and k=100 cosine similarity — QPS, latency, Recall@10, Recall@100 |
| **Hybrid Query** | Structured filter + semantic match — end-to-end latency and throughput |
| **Freshness** | Sustained inserts with embedding backlog — stale row %, merge duration, data loss check |
| **Versioned Queries** | AT VERSION snapshots, semantic guardrails, export reproducibility (byte-level hash) |
| **Durability** | Kill sidecar mid-request, kill DB mid-flush, WAL corruption, disk-full — recovery time and correctness |

### Success Gates

On a 16 vCPU / 64 GB / 2 TB NVMe production node:

- **Correctness:** Zero committed data loss in all durability tests. Semantic guardrails behave exactly as specified. Versioned exports are byte-identical across repeated runs.
- **Performance:** Row point-read p99 ≤ 15 ms. ANN Recall@10 ≥ 0.95 with p95 ≤ 20 ms on 10M vectors (384 dim). Hybrid query p95 ≤ 30 ms.
- **Recovery:** Crash recovery ≤ 30 seconds. Embedding backlog drains to steady-state after sidecar restart.

All benchmarks use identical datasets, query sets, and hardware. Every system is tuned using its documented best practices. Scripts and configs are version-controlled. See the full [Benchmark Plan](docs/architecture/BENCHMARK_PLAN.md) for details.

---

## How It Works

```
┌──────────────────────────────────────────────────────┐
│                  AuroraSQL Language                   │
│       (PostgreSQL wire protocol + AI extensions)      │
├──────────────────────────────────────────────────────┤
│           Query Optimizer, Planner & Executor         │
├──────────────┬───────────────┬───────────────────────┤
│  LSM + PAX   │  Mutable ANN  │ Embedding Sidecar     │
│  Store       │ (mmap + delta │ (ONNX Runtime,         │
│              │   + SQ8)      │  persistent backlog)   │
├──────────────┴───────────────┴───────────────────────┤
│       io_uring I/O Scheduler (HP/BK queues)           │
│         [Linux production; tokio on macOS/Windows]    │
├──────────────────────────────────────────────────────┤
│  Storage (NVMe, blob store)                           │
└──────────────────────────────────────────────────────┘
```

- **Storage:** LSM-tree with PAX columnar blocks. One row holds relational fields, embeddings, binaries, and lineage. Hybrid layout gives you row-level OLTP speed and columnar scan efficiency in the same engine.
- **Vector Index:** mmap'd HNSW base graph with a WAL-backed delta buffer. New vectors are searchable immediately. Crash-safe merges via atomic file rename. Platform-aware quantization: SQ8 on x86-64, FP16 on ARM64, RaBitQ (32× compression) opt-in.
- **Embedding Sidecar:** Standalone process running ONNX Runtime. Generates embeddings automatically on INSERT. Durable backlog ensures zero data loss even under overload. Model version tracking triggers automatic re-embedding on upgrades.
- **Versioning:** Merkle DAG over PAX block hashes. Named tags pin snapshots for training reproducibility. Compaction never garbage-collects pinned data.
- **Durability:** WAL with XXH3-64 checksums, group commit (10 ms default), AES-256-GCM encryption at rest. Recovery < 30 seconds.

---

## Deployment

GalaxDB runs as a < 70 MB embedded library or a standalone server. Same binary, same capabilities.

```bash
# Standalone server (PostgreSQL wire protocol on port 5432)
galaxdb --server --data-dir ./mydata

# Python embedded mode
python -c "import galaxdb; db = galaxdb.Database('./mydata')"
```

| Mode | Platforms | Production? |
|------|-----------|-------------|
| Embedded | Linux, macOS, Windows | Linux only |
| Standalone server | Linux, macOS | Linux only |

Production deployments target Linux 5.10+ with io_uring on NVMe. macOS and Windows are supported for development and testing.

---

## Roadmap

| Phase | Timeline | Focus |
|-------|----------|-------|
| **v1** | 4 months | Single-node engine: storage, SQL, vector search, embedding sidecar, versioning, training export, observability |
| **Cloud beta** | Month 6 | Managed free tier on AWS with schema-level multi-tenancy |
| **v2** | 12–18 months | Distributed clustering (Raft), adaptive storage tiering (RGABH), active learning, feedback loops, GPU-Direct, full PostgreSQL protocol |

---

## Research Basis

GalaxDB's design is grounded in peer-reviewed research, not marketing claims. Every architectural decision traces to a specific paper or production finding.

| Paper | Contribution to GalaxDB |
|-------|------------------------|
| Monkey (Dayan et al., TODS 2018) | Optimal Bloom filter allocation across LSM levels |
| ART (Leis et al., ICDE 2013) | Adaptive Radix Tree for primary key index |
| RaBitQ (Gao et al., SIGMOD 2024/2025) | 32× vector compression replacing Product Quantization |
| BVLSM (Li et al., arXiv 2025) | KV separation at WAL time, not flush time |
| vLSM (Xanthakis et al., arXiv 2024) | Compaction chain reduction for write stall mitigation |
| SILK (Balmau et al., ATC 2019) | Dynamic I/O bandwidth pre-emption for flush priority |
| HNSW (Malkov & Yashunin, TPAMI 2018) | Approximate nearest neighbor graph index |
| MinHash LSH (Broder, 1997) | Near-duplicate detection for training data quality |
| PostgreSQL SSI (Cahill et al., SIGMOD 2008) | Serializable Snapshot Isolation (v2) |

All 27 research-backed findings from the design review are resolved in the [architecture specification](docs/architecture/Final%20Version%204.2.md).

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture Specification](docs/architecture/Final%20Version%204.2.md) | The authoritative, locked v1+v2 spec. All 27 findings resolved. |
| [v2 Full System Design](docs/architecture/v2%20Full%20System.md) | Detailed v2: distributed clustering, RGABH, active learning, GPU-Direct |
| [Production Deployment](docs/architecture/Production%20Deployment%20Target.md) | AWS deployment, cloud tiers, multi-tenant isolation |
| [Benchmark Plan](docs/architecture/BENCHMARK_PLAN.md) | 6-week benchmark runbook against 6 competitor systems |
| [Business Model](docs/business/GalaxDB%20Business%20Model.md) | Open-core revenue model, unit economics, GTM strategy |

---

## Organization

Built by **Zentrix Innovative Labs Limited**.

## License

Apache License 2.0. See [LICENSE](LICENSE).
