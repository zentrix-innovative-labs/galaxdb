# GalaxDB vs Other Databases

This document compares GalaxDB to the databases most commonly used in AI/ML applications. The goal is to help you understand where GalaxDB fits and when to choose it.

---

## The Problem GalaxDB Solves

Most AI applications today need at least three separate systems:

1. A relational database (PostgreSQL, MySQL) for structured data and SQL queries
2. A vector database (Pinecone, Qdrant, Weaviate) for semantic search
3. A data pipeline (Airflow, dbt) to export training datasets to object storage

GalaxDB replaces all three with a single system that speaks PostgreSQL wire protocol, runs embeddings locally, and exports training data with one SQL command.

---

## Feature Matrix

| Feature | GalaxDB | PostgreSQL + pgvector | Pinecone | Qdrant | Weaviate | LanceDB | ChromaDB | Milvus |
|---|---|---|---|---|---|---|---|---|
| **SQL queries** | ✅ Full | ✅ Full | ❌ | ❌ | Partial | Partial | ❌ | ❌ |
| **Vector search** | ✅ HNSW | ⚠️ pgvector | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Local embeddings** | ✅ Built-in | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ |
| **Time-travel queries** | ✅ `AT VERSION` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Training export** | ✅ Lance format | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Near-dedup** | ✅ MinHash LSH | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Embedded mode** | ✅ (like SQLite) | ❌ | ❌ | ⚠️ in-memory | ❌ | ✅ | ✅ | ❌ |
| **Self-hosted** | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Wire protocol** | PostgreSQL | PostgreSQL | REST | REST/gRPC | REST/gRPC | Python API | Python API | gRPC |
| **MVCC / snapshots** | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Encryption at rest** | ✅ AES-256-GCM | ✅ (OS-level) | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| **Backup / restore** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |

---

## GalaxDB vs PostgreSQL + pgvector

**PostgreSQL** is the gold standard for relational data. **pgvector** is a popular extension that adds approximate nearest-neighbor search. Together they're the most common starting point for teams adding vector search to an existing Postgres stack.

**Where pgvector falls short:**
- Vector search recall is lower than dedicated HNSW implementations. pgvector's HNSW recall@10 on SIFT-1M is typically 0.92–0.95 at comparable ef settings; GalaxDB achieves 0.990.
- No local embedding generation — you still need an external API (OpenAI, Cohere) or a separate embedding service.
- No training export — getting data into PyTorch requires a custom pipeline.
- No time-travel queries.

**When to stay with PostgreSQL + pgvector:**
- You have an existing Postgres deployment and just need basic vector search.
- Your team is deeply invested in the PostgreSQL ecosystem (extensions, tooling, managed services).
- You don't need local embeddings or training exports.

**GalaxDB advantage:** Drop-in replacement for the wire protocol. Your existing psycopg2/SQLAlchemy code works unchanged. You gain local embeddings, training export, and time-travel without adding services.

---

## GalaxDB vs Pinecone

**Pinecone** is a fully managed, cloud-only vector database. It's the easiest way to get production vector search running quickly.

**Pinecone strengths:**
- Zero operational overhead — fully managed, auto-scaling.
- Excellent developer experience and documentation.
- High availability out of the box.

**Pinecone limitations:**
- No SQL. Filtering is metadata-based, not relational.
- No local embeddings — you pay for every embedding API call.
- No training export.
- No self-hosting — your data lives in Pinecone's cloud.
- Cost scales with vector count and query volume. At 10M vectors, costs can reach $700–$1,200/month.

**When to choose Pinecone:**
- You want zero ops and are comfortable with the cost.
- You don't need SQL or training exports.
- You're building a pure semantic search product.

**GalaxDB advantage:** Self-hosted, no per-query cost, SQL + vector in one query, training export built in.

---

## GalaxDB vs Qdrant

**Qdrant** is an open-source, high-performance vector search engine written in Rust. It's one of the fastest dedicated vector databases available.

**Qdrant strengths:**
- Excellent vector search performance and recall.
- Rich filtering with payload conditions.
- Self-hosted or managed cloud.
- Active development, strong community.

**Qdrant limitations:**
- No SQL — queries use a custom JSON API or gRPC.
- No local embedding generation (relies on external models).
- No training export.
- No time-travel or MVCC.
- No relational joins or aggregations.

**When to choose Qdrant:**
- You need pure vector search at scale with no SQL requirement.
- You want the best vector search performance in a dedicated system.
- You're comfortable managing embeddings externally.

**GalaxDB advantage:** SQL + vector in one system, local embeddings, training export, time-travel. GalaxDB is the better choice when you need both relational and vector capabilities.

---

## GalaxDB vs Weaviate

**Weaviate** is an open-source vector database with a GraphQL API and built-in vectorization modules.

**Weaviate strengths:**
- Built-in vectorization via modules (OpenAI, Cohere, HuggingFace).
- GraphQL and REST APIs.
- Multi-tenancy support.
- Active ecosystem.

**Weaviate limitations:**
- No SQL — GraphQL is powerful but not relational.
- High memory usage at scale (known OOM issues reported in production).
- No training export.
- No time-travel.
- Complex operational setup for production.

**When to choose Weaviate:**
- You want a GraphQL API for vector search.
- You need multi-tenancy out of the box.
- You're building a knowledge graph or semantic search product.

**GalaxDB advantage:** SQL, lower operational complexity, training export, time-travel, MVCC.

---

## GalaxDB vs LanceDB

**LanceDB** is an open-source embedded vector database built on the Lance columnar format. It's the closest conceptual relative to GalaxDB.

**LanceDB strengths:**
- Embedded mode (no server) — runs inside your Python process.
- Built on Lance format — excellent for multimodal data and ML workflows.
- Zero-copy PyTorch integration via `to_pytorch()`.
- Good vector search performance.
- Supports SQL-like queries via DuckDB integration.

**LanceDB limitations:**
- No full SQL engine — queries go through a Python API or DuckDB bridge.
- No PostgreSQL wire protocol — can't use psycopg2/SQLAlchemy directly.
- No local embedding generation (relies on external models or LanceDB's embedding registry).
- No time-travel queries (Lance format has versioning but LanceDB doesn't expose AT VERSION SQL).
- No WAL-based crash recovery — relies on Lance's copy-on-write.
- No write-ahead log for ACID guarantees.

**When to choose LanceDB:**
- You're building a pure ML/data pipeline and don't need SQL.
- You want the Lance format as your primary storage layer.
- You're working in a notebook or single-process environment.

**GalaxDB advantage:** Full SQL engine, PostgreSQL wire protocol (any client works), WAL-based crash recovery, time-travel with `AT VERSION`, MVCC, and the Lance export is a first-class feature (`CREATE VERSION TAG FOR TRAINING`). GalaxDB uses Lance as its training export format — so you get the best of both worlds.

---

## GalaxDB vs ChromaDB

**ChromaDB** is an open-source embedding database designed for LLM applications. It's the most popular choice for quick prototyping.

**ChromaDB strengths:**
- Extremely easy to get started — `pip install chromadb`, three lines of code.
- Good Python API.
- In-memory and persistent modes.
- Active community, many LangChain/LlamaIndex integrations.

**ChromaDB limitations:**
- No SQL.
- No PostgreSQL wire protocol.
- Limited production scalability — not designed for high-throughput workloads.
- No training export.
- No time-travel.
- No encryption at rest.
- No backup/restore.

**When to choose ChromaDB:**
- Prototyping and development.
- Small-scale RAG applications.
- You want the fastest path from zero to working semantic search.

**GalaxDB advantage:** Production-grade storage engine, SQL, PostgreSQL wire protocol, training export, encryption, backup/restore. ChromaDB is great for prototyping; GalaxDB is for production.

---

## GalaxDB vs Milvus

**Milvus** is an open-source distributed vector database designed for billion-scale deployments.

**Milvus strengths:**
- Designed for massive scale (billions of vectors).
- Distributed architecture with horizontal scaling.
- Multiple index types (IVF, HNSW, DiskANN).
- Strong enterprise features.

**Milvus limitations:**
- Complex deployment — requires etcd, MinIO, and multiple services.
- No SQL — uses a custom query language.
- No local embeddings.
- No training export.
- No time-travel.
- High operational overhead for small-to-medium deployments.

**When to choose Milvus:**
- You need billion-scale vector search.
- You have a dedicated infrastructure team.
- You're building a large-scale recommendation or search system.

**GalaxDB advantage:** Single binary, SQL, training export, time-travel, far lower operational complexity. For workloads under 100M vectors, GalaxDB is simpler and more capable.

---

## GalaxDB vs DuckDB

**DuckDB** is an in-process analytical database. It's excellent for OLAP workloads and data analysis.

**DuckDB strengths:**
- Outstanding analytical performance (often faster than GalaxDB on pure OLAP).
- Excellent SQL support.
- Embedded mode.
- Great ecosystem (Arrow, Parquet, CSV, JSON).

**DuckDB limitations:**
- No vector search.
- No embeddings.
- No training export.
- Optimized for read-heavy analytics, not mixed OLTP+OLAP.

**When to choose DuckDB:**
- Pure analytical workloads with no vector search requirement.
- Data analysis and exploration.
- ETL pipelines.

**GalaxDB advantage:** Vector search, embeddings, training export, OLTP write throughput. DuckDB wins on pure analytical throughput; GalaxDB wins on the full AI/ML workload.

---

## Summary: When to Choose GalaxDB

Choose GalaxDB when you need **more than one** of:

- SQL queries with relational semantics
- Vector similarity search
- Local embedding generation (no API cost)
- Training dataset export to PyTorch
- Time-travel queries for reproducibility
- PostgreSQL wire protocol compatibility
- Single-binary deployment

If you only need one of these, a specialized tool may be simpler. If you need two or more, GalaxDB eliminates the integration overhead.

---

*Comparison data sourced from official documentation and public benchmarks. GalaxDB performance numbers measured on AWS c6id.4xlarge, release build. See [BENCHMARKS.md](BENCHMARKS.md) for full details.*
