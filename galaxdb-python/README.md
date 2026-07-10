# galaxdb-client

The Python client for [GalaxDB](https://github.com/zentrix-innovative-labs/galaxdb) -- an AI-native database that combines SQL, vector search, and local embeddings in a single binary.

No external API keys. No separate vector database. No data pipeline. One connection string.

## Installation

```bash
pip install galaxdb-client
```

Requires Python 3.9+. Pre-built wheels for Linux x86_64, macOS Intel, macOS Apple Silicon, and Windows x86_64.

## What GalaxDB gives you

- **Full SQL** -- CREATE, INSERT, UPDATE, DELETE, SELECT with WHERE filters
- **Local embeddings** -- text to vector conversion runs inside the process, no API key needed
- **Semantic search** -- `SEMANTIC_MATCH(col, 'query', threshold)` in any WHERE clause; add `LIMIT n` for the *n* nearest matches
- **Semantic caching** -- `CREATE SEMANTIC CACHE FOR TABLE t SIMILARITY 0.98 TTL 300` serves near-identical queries from cache
- **HNSW vector index** -- recall@10 = 0.990 on SIFT-1M at ef=200, persisted across restart (no re-embed)
- **DiskANN** -- opt-in disk-resident (Vamana) index for larger-than-RAM vector sets
- **Time-travel queries** -- `SELECT ... AT VERSION 'tag'`, including exact historical vector search with `CONSISTENCY 'SEMANTIC_SNAPSHOT'`
- **Serializable isolation** -- opt-in Serializable Snapshot Isolation on top of the default snapshot isolation
- **Training export** -- `CREATE VERSION TAG ... FOR TRAINING` exports a Lance dataset, zero-copy PyTorch-ready
- **Near-dedup** -- `WHERE NOT DUPLICATE` removes near-duplicate rows using MinHash LSH
- **Crash safety** -- WAL + checksum, 7 chaos scenarios pass in under 11 seconds
- **Encryption at rest** -- AES-256-GCM on every block and WAL record

## Quick start -- embedded mode (no server)

```python
import galaxdb

# Open or create a database at a local path
db = galaxdb.Database("/tmp/mydb")

# Create a table
db.execute("CREATE TABLE products (id INT PRIMARY KEY, name TEXT, price INT)")

# Insert rows
db.execute("INSERT INTO products (id, name, price) VALUES (1, 'Laptop', 1200)")
db.execute("INSERT INTO products (id, name, price) VALUES (2, 'Headphones', 150)")
db.execute("INSERT INTO products (id, name, price) VALUES (3, 'Keyboard', 80)")

# Query with filter
rows = db.execute("SELECT * FROM products WHERE price > 100")
for row in rows:
    print(row)
# {'id': '1', 'name': 'Laptop', 'price': '1200'}
# {'id': '2', 'name': 'Headphones', 'price': '150'}

# Update
db.execute("UPDATE products SET price = 1100 WHERE id = 1")

# Delete
db.execute("DELETE FROM products WHERE id = 3")

# Table info
print(db.table_exists("products"))  # True
print(db.table_count)               # 1
```

## Semantic search with local embeddings

Start the server with the embedding sidecar to enable `SEMANTIC_MATCH`:

```bash
galaxdb-server \
  --data-dir ./data \
  --port 5433 \
  --sidecar /usr/local/bin/galaxdb-sidecar \
  --model sentence-transformers/all-MiniLM-L6-v2
```

Then connect from Python:

```python
import galaxdb

conn = galaxdb.connect("host=localhost port=5433 dbname=galaxdb sslmode=disable")

# Create a table with an embedding column
conn.execute("""
    CREATE TABLE docs (
        id   INT PRIMARY KEY,
        body TEXT EMBEDDING MODEL 'sentence-transformers/all-MiniLM-L6-v2' DIM 384
    )
""")

# Insert rows -- embeddings are computed automatically by the local sidecar
conn.execute("INSERT INTO docs (id, body) VALUES (1, 'machine learning and neural networks')")
conn.execute("INSERT INTO docs (id, body) VALUES (2, 'rust programming language systems')")
conn.execute("INSERT INTO docs (id, body) VALUES (3, 'cooking recipes italian pasta')")
conn.execute("INSERT INTO docs (id, body) VALUES (4, 'deep learning transformers attention')")

# Semantic search -- no external API, no separate vector DB
rows = conn.execute(
    "SELECT id, body FROM docs WHERE SEMANTIC_MATCH(body, 'artificial intelligence', 0.4)"
)
for row in rows:
    print(row)
# Returns rows 1 and 4 -- the AI/ML related documents

# SEMANTIC_MATCH is a top-k search. Without a LIMIT it returns the 10
# nearest matches; add LIMIT to control how many come back:
rows = conn.execute(
    "SELECT id, body FROM docs WHERE SEMANTIC_MATCH(body, 'artificial intelligence', 0.3) LIMIT 50"
)

conn.close()
```

## Time-travel queries

```python
# Create a named snapshot
conn.execute("CREATE VERSION TAG 'v1' FOR TRAINING WITH TRAINING PRECISION 'float32'")

# Insert more data after the snapshot
conn.execute("INSERT INTO docs (id, body) VALUES (5, 'new document added later')")

# Query the snapshot -- only sees data from before the tag
rows = conn.execute("SELECT * FROM docs AT VERSION 'v1'")
# Returns rows 1-4, not row 5
```

## Training export

```python
import galaxdb
import lance
import torch

db = galaxdb.Database("./data")

# Create a training snapshot
db.execute("CREATE VERSION TAG 'train-v1' FOR TRAINING WITH TRAINING PRECISION 'float32'")

# Export as a Lance dataset
path = db.training_dataset("train-v1")

# Load into PyTorch -- zero-copy, memory-mapped
dataset = lance.dataset(path).to_pytorch()
loader = torch.utils.data.DataLoader(dataset, batch_size=32)
```

## Bulk insert

```python
conn.execute("""
    BULK INSERT INTO products (id, name, price) VALUES
      (10, 'Monitor', 400),
      (11, 'Mouse', 30),
      (12, 'Webcam', 90)
""")
```

## Near-duplicate deduplication

```python
# Select only unique documents (one per near-duplicate cluster)
rows = conn.execute("SELECT * FROM docs WHERE NOT DUPLICATE")
```

## Backup and restore

```python
conn.execute("BACKUP TO '/path/to/backup'")
conn.execute("RESTORE FROM '/path/to/backup'")
```

## Server mode -- connect to a running GalaxDB server

```python
import galaxdb

# Connect using a PostgreSQL-style connection string
conn = galaxdb.connect("host=localhost port=5433 dbname=galaxdb sslmode=disable")

conn.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT, age INT)")
conn.execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")

rows = conn.execute("SELECT * FROM users WHERE age > 25")
for row in rows:
    print(row)

conn.close()
```

Any PostgreSQL client works -- psycopg2, SQLAlchemy, tokio-postgres, pg (Node.js), JDBC.

## Docker

```bash
docker run -d -p 5433:5433 -p 9090:9090 \
  -v /data:/data \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  harbi256/galaxdb:latest \
  --data-dir /data \
  --sidecar /usr/local/bin/galaxdb-sidecar \
  --model sentence-transformers/all-MiniLM-L6-v2
```

## Observability

```bash
# Health check
curl http://localhost:9090/health
# {"status":"ok","version":"0.7.0","subsystems":{"sidecar_healthy":true}}

# Prometheus metrics
curl http://localhost:9090/metrics
```

## Links

- [GitHub](https://github.com/zentrix-innovative-labs/galaxdb)
- [Getting Started](https://github.com/zentrix-innovative-labs/galaxdb/blob/main/docs/GETTING_STARTED.md)
- [Benchmarks](https://github.com/zentrix-innovative-labs/galaxdb/blob/main/docs/BENCHMARKS.md)
- [Docker Hub](https://hub.docker.com/r/harbi256/galaxdb)

## License

Apache 2.0
