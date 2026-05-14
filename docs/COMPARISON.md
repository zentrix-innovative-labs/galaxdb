# GalaxDB vs Other Databases

This document compares GalaxDB to the databases most commonly used in AI/ML applications. All GalaxDB performance numbers are measured on AWS c6id.4xlarge (Intel Xeon Platinum 8375C, 16 vCPU, 32 GiB RAM, 884 GB NVMe), release build. See [BENCHMARKS.md](BENCHMARKS.md) for reproduction commands.

---

## The Problem GalaxDB Solves

Most AI applications today need at least three separate systems:

1. A relational database (PostgreSQL, MySQL) for structured data and SQL queries
2. A vector database (Pinecone, Qdrant, Weaviate) for semantic search
3. A data pipeline (Airflow, dbt) to export training datasets to object storage

GalaxDB replaces all three with a single binary that speaks PostgreSQL wire protocol, runs embeddings locally, and exports training data with one SQL command.

```
Before GalaxDB:
  PostgreSQL + pgvector + Pinecone + OpenAI API + S3 + Airflow

After GalaxDB:
  galaxdb-server
```

---

## Feature Matrix

| Feature | GalaxDB | PostgreSQL + pgvector | Pinecone | Qdrant | Weaviate | LanceDB | ChromaDB | Milvus | DuckDB |
|---|---|---|---|---|---|---|---|---|---|
| **SQL queries** | ✅ Full | ✅ Full | ❌ | ❌ | Partial | Partial¹ | ❌ | ❌ | ✅ Full |
| **Vector search** | ✅ HNSW | ⚠️ pgvector | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **HNSW recall@10 (SIFT-1M)** | **0.990** | ~0.95 | N/A² | ~0.99³ | ~0.97 | ~0.97 | ~0.95 | ~0.98 | — |
| **Local embeddings** | ✅ Built-in | ❌ | ❌ | ⚠️ FastEmbed | ✅ modules | ✅ | ✅ | ❌ | ❌ |
| **Time-travel queries** | ✅ `AT VERSION` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Training export** | ✅ Lance format | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Near-dedup (MinHash LSH)** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Embedded mode** | ✅ (like SQLite) | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ |
| **Self-hosted** | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **PostgreSQL wire protocol** | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **MVCC / snapshots** | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Encryption at rest** | ✅ AES-256-GCM | ✅ OS-level | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| **Backup / restore** | ✅ SQL command | ✅ pg_dump | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| **Single binary** | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ |
| **Write TPS** | **258,555** | ~3,200 | N/A | N/A | N/A | N/A | N/A | N/A | ~50,000 |
| **Scan throughput** | **4.49 GB/s** | ~0.9 GB/s | N/A | N/A | N/A | N/A | N/A | N/A | ~5–10 GB/s |
| **Open source** | ✅ Apache 2.0 | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

¹ LanceDB OSS uses a Python/Arrow API. SQL is available via a DuckDB bridge or Enterprise tier only — not native.  
² Pinecone uses proprietary indexing; recall is not published against standard ANN benchmarks.  
³ Qdrant recall numbers from Qdrant's own published benchmarks on their hardware; not directly comparable.

---

## GalaxDB vs PostgreSQL + pgvector

**PostgreSQL** is the gold standard for relational data. **pgvector** is a popular extension that adds approximate nearest-neighbor search. Together they're the most common starting point for teams adding vector search to an existing Postgres stack.

### Performance comparison

| Metric | GalaxDB | PostgreSQL 16 + pgvector |
|--------|---------|--------------------------|
| Write TPS (16 threads, 1M rows) | **258,555** | ~3,200 |
| Read p50 | **3 µs** | ~95 µs |
| Read p99 | **47 µs** | ~300 µs |
| Scan throughput | **4.49 GB/s** | ~0.9 GB/s |
| HNSW recall@10 (SIFT-1M, ef=200) | **0.990** | ~0.95 |

### Where pgvector falls short

- Vector search recall is lower than a purpose-built HNSW implementation. pgvector's HNSW recall@10 on SIFT-1M is typically 0.92–0.95 at comparable ef settings; GalaxDB achieves 0.990.
- No local embedding generation — you still need an external API (OpenAI, Cohere) or a separate embedding service.
- No training export — getting data into PyTorch requires a custom pipeline.
- No time-travel queries.
- Write throughput is limited by PostgreSQL's MVCC heap — ~3,200 TPS vs GalaxDB's 258,555 TPS.

### When to stay with PostgreSQL + pgvector

- You have an existing Postgres deployment and just need basic vector search.
- Your team is deeply invested in the PostgreSQL ecosystem (extensions, managed services like RDS, Supabase).
- You don't need local embeddings or training exports.

**GalaxDB advantage:** Drop-in replacement for the wire protocol. Your existing psycopg2/SQLAlchemy code works unchanged. You gain local embeddings, training export, and time-travel without adding services.

---

## GalaxDB vs Pinecone

**Pinecone** is a fully managed, cloud-only vector database. It's the easiest way to get production vector search running quickly.

### Pricing reality

| Scale | Pinecone (Serverless) | GalaxDB (self-hosted) |
|-------|----------------------|----------------------|
| 1M vectors | ~$25–70/month | Server cost only |
| 10M vectors | ~$200–400/month | Server cost only |
| 100M vectors | $500+/month minimum | Server cost only |

Pinecone's Enterprise plan has a $500/month minimum commitment. At scale, self-hosting GalaxDB on a $100/month cloud instance is dramatically cheaper.

### Pinecone strengths

- Zero operational overhead — fully managed, auto-scaling.
- Excellent developer experience and documentation.
- High availability out of the box.

### Pinecone limitations

- No SQL. Filtering is metadata-based, not relational.
- No local embeddings — you pay for every embedding API call.
- No training export.
- No self-hosting — your data lives in Pinecone's cloud.
- No MVCC, no time-travel, no backup/restore SQL command.

### When to choose Pinecone

- You want zero ops and are comfortable with the cost.
- You don't need SQL or training exports.
- You're building a pure semantic search product with a small team.

**GalaxDB advantage:** Self-hosted, no per-query cost, SQL + vector in one query, training export built in.

---

## GalaxDB vs Qdrant

**Qdrant** is an open-source, high-performance vector search engine written in Rust. It's one of the fastest dedicated vector databases available and a strong competitor in the pure vector search space.

### Qdrant strengths

- Excellent vector search performance and recall (Qdrant publishes ~0.99 recall on their benchmarks).
- Rich payload filtering with complex conditions.
- Self-hosted or managed cloud (Qdrant Cloud).
- Active development, strong community, written in Rust.
- FastEmbed for lightweight local embeddings.

### Qdrant limitations

- No SQL — queries use a custom JSON API or gRPC.
- No relational joins or aggregations.
- No training export.
- No time-travel or MVCC.
- No WAL-based crash recovery with ACID guarantees.
- No PostgreSQL wire protocol — can't use psycopg2/SQLAlchemy directly.

### When to choose Qdrant

- You need pure vector search at scale with no SQL requirement.
- You want the best vector search performance in a dedicated system.
- You're comfortable managing embeddings externally.

**GalaxDB advantage:** SQL + vector in one system, local embeddings, training export, time-travel, PostgreSQL wire protocol. GalaxDB is the better choice when you need both relational and vector capabilities.

---

## GalaxDB vs Weaviate

**Weaviate** is an open-source vector database with a GraphQL API and built-in vectorization modules.

### Weaviate strengths

- Built-in vectorization via modules (OpenAI, Cohere, HuggingFace).
- GraphQL and REST APIs.
- Multi-tenancy support.
- Hybrid search (BM25 + vector).
- Active ecosystem.

### Weaviate limitations

- No SQL — GraphQL is powerful but not relational.
- High memory usage at scale — OOM issues are commonly reported in production deployments.
- No training export.
- No time-travel.
- Complex operational setup: requires multiple services for production.
- Qdrant achieves ~2.3× higher throughput per dollar on identical hardware (per published benchmarks).

### When to choose Weaviate

- You want a GraphQL API for vector search.
- You need multi-tenancy out of the box.
- You're building a knowledge graph or semantic search product.

**GalaxDB advantage:** SQL, lower operational complexity (single binary), training export, time-travel, MVCC, PostgreSQL wire protocol.

---

## GalaxDB vs LanceDB

**LanceDB** is an open-source embedded vector database built on the Lance columnar format. It's the closest conceptual relative to GalaxDB — both are embedded-first and both use the Lance format.

### LanceDB strengths

- Embedded mode (no server) — runs inside your Python process.
- Built on Lance format — excellent for multimodal data and ML workflows.
- Zero-copy PyTorch integration via `to_pytorch()`.
- Good vector search performance.
- Table versioning built into the Lance format.
- DuckDB bridge for SQL-like queries (OSS).

### LanceDB limitations

- No native SQL engine — OSS queries go through a Python API or DuckDB bridge. Full SQL is Enterprise-only.
- No PostgreSQL wire protocol — can't use psycopg2/SQLAlchemy directly.
- No local embedding generation in OSS (relies on external models or LanceDB's embedding registry).
- No `AT VERSION` SQL time-travel — Lance has versioning but LanceDB doesn't expose it as a SQL primitive.
- No WAL-based crash recovery — relies on Lance's copy-on-write semantics.
- No ACID write guarantees for concurrent writers.

### When to choose LanceDB

- You're building a pure ML/data pipeline and don't need SQL.
- You want the Lance format as your primary storage layer.
- You're working in a notebook or single-process environment.

**GalaxDB advantage:** Full SQL engine, PostgreSQL wire protocol (any client works), WAL-based crash recovery, time-travel with `AT VERSION`, MVCC, and the Lance export is a first-class feature (`CREATE VERSION TAG FOR TRAINING`). GalaxDB uses Lance as its training export format — so you get the best of both worlds.

---

## GalaxDB vs ChromaDB

**ChromaDB** is an open-source embedding database designed for LLM applications. It's the most popular choice for quick prototyping.

### ChromaDB strengths

- Extremely easy to get started — `pip install chromadb`, three lines of code.
- Good Python API.
- In-memory and persistent modes.
- Active community, many LangChain/LlamaIndex integrations.

### ChromaDB limitations

- No SQL.
- No PostgreSQL wire protocol.
- Limited production scalability — not designed for high-throughput workloads.
- No training export.
- No time-travel.
- No encryption at rest.
- No backup/restore.
- Batching constraints limit ingestion throughput at scale.

### When to choose ChromaDB

- Prototyping and development.
- Small-scale RAG applications.
- You want the fastest path from zero to working semantic search.

**GalaxDB advantage:** Production-grade storage engine (258K TPS, 4.49 GB/s scans), SQL, PostgreSQL wire protocol, training export, encryption, backup/restore. ChromaDB is great for prototyping; GalaxDB is for production.

---

## GalaxDB vs Milvus

**Milvus** is an open-source distributed vector database designed for billion-scale deployments.

### Milvus strengths

- Designed for massive scale (billions of vectors).
- Distributed architecture with horizontal scaling.
- Multiple index types (IVF, HNSW, DiskANN, SCANN).
- Strong enterprise features and Zilliz Cloud managed offering.
- Good recall at scale.

### Milvus limitations

- Complex deployment — requires etcd, MinIO, and multiple services. Not a single binary.
- No SQL — uses a custom query language (PyMilvus API).
- No local embeddings.
- No training export.
- No time-travel.
- High operational overhead for small-to-medium deployments.
- At 10M vectors, Milvus costs ~$500/month on Zilliz Cloud vs GalaxDB on a $100/month server.

### When to choose Milvus

- You need billion-scale vector search.
- You have a dedicated infrastructure team.
- You're building a large-scale recommendation or search system.

**GalaxDB advantage:** Single binary, SQL, training export, time-travel, far lower operational complexity. For workloads under 100M vectors, GalaxDB is simpler and more capable.

---

## GalaxDB vs DuckDB

**DuckDB** is an in-process analytical database. It's excellent for OLAP workloads and data analysis.

### Performance comparison

| Metric | GalaxDB | DuckDB |
|--------|---------|--------|
| OLAP scan throughput | 4.49 GB/s | ~5–10 GB/s |
| OLTP write TPS | **258,555** | ~50,000 |
| Vector search | ✅ HNSW | ❌ |
| Embeddings | ✅ Local | ❌ |
| Training export | ✅ Lance | ❌ |
| Embedded mode | ✅ | ✅ |

### DuckDB strengths

- Outstanding analytical performance — often faster than GalaxDB on pure OLAP.
- Excellent SQL support (window functions, CTEs, lateral joins).
- Embedded mode.
- Great ecosystem (Arrow, Parquet, CSV, JSON, Iceberg).
- Zero operational overhead.

### DuckDB limitations

- No vector search.
- No embeddings.
- No training export.
- Optimized for read-heavy analytics, not mixed OLTP+OLAP.
- No WAL-based crash recovery for concurrent writes.

### When to choose DuckDB

- Pure analytical workloads with no vector search requirement.
- Data analysis and exploration.
- ETL pipelines.

**GalaxDB advantage:** Vector search, embeddings, training export, OLTP write throughput. DuckDB wins on pure analytical throughput; GalaxDB wins on the full AI/ML workload.

---

## Summary: When to Choose GalaxDB

Choose GalaxDB when you need **more than one** of:

- SQL queries with relational semantics
- Vector similarity search (recall@10 = 0.990 on SIFT-1M)
- Local embedding generation (no API cost, no data leaving your infrastructure)
- Training dataset export to PyTorch (Lance format, zero-copy)
- Time-travel queries for reproducibility (`AT VERSION`)
- PostgreSQL wire protocol compatibility (psycopg2, SQLAlchemy, pg, JDBC — all work)
- Single-binary deployment with no external dependencies

If you only need one of these, a specialized tool may be simpler. If you need two or more, GalaxDB eliminates the integration overhead.

---

## Quick Decision Guide

| Your situation | Recommended |
|----------------|-------------|
| Existing Postgres, just need basic vector search | PostgreSQL + pgvector |
| Pure vector search, no SQL, managed cloud | Pinecone |
| Pure vector search, self-hosted, high performance | Qdrant |
| ML pipeline, notebook-first, Lance format | LanceDB |
| Prototyping a RAG app quickly | ChromaDB |
| Billion-scale vector search, dedicated infra team | Milvus |
| Pure analytics, no vector search | DuckDB |
| SQL + vector + embeddings + training export | **GalaxDB** |

---

*GalaxDB performance numbers measured on AWS c6id.4xlarge, release build. Competitor numbers sourced from official documentation and published benchmarks. See [BENCHMARKS.md](BENCHMARKS.md) for GalaxDB reproduction commands.*

*Content was rephrased for compliance with licensing restrictions.*
