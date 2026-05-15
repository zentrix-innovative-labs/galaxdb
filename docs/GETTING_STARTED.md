# GalaxDB Getting Started Guide

Everything you need to go from zero to a running GalaxDB instance with full AI features — SQL, vector search, local embeddings, time-travel, and training exports.

---

## What GalaxDB gives you

- **Full SQL** — CREATE, INSERT, UPDATE, DELETE, SELECT with WHERE, joins, aggregates
- **Local embeddings** — text → vector conversion runs inside the process, no API key, no data leaving your machine
- **Semantic search** — `SEMANTIC_MATCH(col, 'query', threshold)` in any WHERE clause
- **HNSW vector index** — recall@10 = 0.990 on SIFT-1M at ef=200
- **Time-travel** — `SELECT ... AT VERSION 'tag'` to query historical snapshots
- **Training export** — `CREATE VERSION TAG ... FOR TRAINING` exports a Lance dataset, zero-copy PyTorch-ready
- **Near-dedup** — `WHERE NOT DUPLICATE` removes near-duplicate rows using MinHash LSH
- **Crash safety** — WAL + checksum, 7 chaos scenarios pass in < 11 s
- **Encryption at rest** — AES-256-GCM on every block and WAL record
- **Observability** — `/health` and `/metrics` (Prometheus) on port 9090

---

## Installation

### Option 1 — curl installer (Linux / macOS)

```bash
curl -fsSL https://raw.githubusercontent.com/zentrix-innovative-labs/galaxdb/main/install.sh | bash
```

Installs `galaxdb-server` to `/usr/local/bin`. The sidecar binary (`galaxdb-sidecar`) is downloaded separately from the [Releases page](https://github.com/zentrix-innovative-labs/galaxdb/releases).

### Option 2 — Homebrew (macOS)

```bash
brew tap zentrix-innovative-labs/tap
brew install galaxdb
```

### Option 3 — Docker

```bash
# Without embeddings (SQL + vector search only)
docker run -d -p 5433:5433 -p 9090:9090 -v /data:/data \
  harbi256/galaxdb:latest

# With embeddings (full AI features)
docker run -d -p 5433:5433 -p 9090:9090 \
  -v /data:/data \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  harbi256/galaxdb:latest \
  --data-dir /data \
  --port 5433 \
  --observe-port 9090 \
  --sidecar /usr/local/bin/galaxdb-sidecar \
  --model sentence-transformers/all-MiniLM-L6-v2
```

The `-v ~/.cache/huggingface` mount caches the model (~90 MB) so it only downloads once.

### Option 4 — Python (embedded, no server)

```bash
pip install galaxdb
```

```python
import galaxdb
db = galaxdb.Database("./mydata")
```

---

## Starting the server

### Without embeddings (SQL + HNSW only)

```bash
galaxdb-server --data-dir ./data --port 5433 --observe-port 9090
```

### With embeddings (full AI features)

```bash
galaxdb-server \
  --data-dir ./data \
  --port 5433 \
  --observe-port 9090 \
  --sidecar /usr/local/bin/galaxdb-sidecar \
  --model sentence-transformers/all-MiniLM-L6-v2
```

On first run the sidecar downloads the model from HuggingFace Hub (~90 MB). Subsequent starts use the local cache and come up in seconds.

**Verify it's running:**

```bash
curl http://localhost:9090/health
# {"status":"ok","version":"1.0.0-beta.1","subsystems":{"disk_full":false,"sidecar_healthy":true,"connections_active":0}}
```

`sidecar_healthy: true` means embeddings are active.

---

## Connecting

Any PostgreSQL client works — GalaxDB speaks the PostgreSQL wire protocol.

```bash
# psql
psql "host=localhost port=5433 dbname=galaxdb sslmode=disable"

# Python
import psycopg2
conn = psycopg2.connect(host="localhost", port=5433, dbname="galaxdb", sslmode="disable")

# SQLAlchemy
from sqlalchemy import create_engine
engine = create_engine("postgresql://localhost:5433/galaxdb")

# Node.js (pg)
const { Pool } = require('pg')
const pool = new Pool({ host: 'localhost', port: 5433, database: 'galaxdb', ssl: false })
```

---

## Feature Reference

### Standard SQL

All standard SQL works as expected.

```sql
-- Create a table
CREATE TABLE users (
    id    INT PRIMARY KEY,
    name  TEXT,
    email TEXT,
    age   INT
);

-- Insert rows
INSERT INTO users (id, name, email, age) VALUES (1, 'Alice', 'alice@example.com', 30);
INSERT INTO users (id, name, email, age) VALUES (2, 'Bob', 'bob@example.com', 25);

-- Bulk insert (faster for many rows)
BULK INSERT INTO users (id, name, email, age) VALUES
  (3, 'Charlie', 'charlie@example.com', 35),
  (4, 'Diana', 'diana@example.com', 28);

-- Select with filter
SELECT id, name, age FROM users WHERE age > 28;

-- Update
UPDATE users SET age = 31 WHERE id = 1;

-- Delete
DELETE FROM users WHERE id = 4;

-- Drop table
DROP TABLE users;
```

---

### Embedding Columns

Declare a column as an embedding source with `EMBEDDING MODEL 'model-id' DIM n`. When you insert a row, the sidecar automatically computes the embedding — no extra code needed.

```sql
CREATE TABLE docs (
    id   INT PRIMARY KEY,
    body TEXT EMBEDDING MODEL 'sentence-transformers/all-MiniLM-L6-v2' DIM 384
);

-- Embeddings are computed automatically on insert
INSERT INTO docs (id, body) VALUES (1, 'machine learning and neural networks');
INSERT INTO docs (id, body) VALUES (2, 'rust programming language systems');
INSERT INTO docs (id, body) VALUES (3, 'cooking recipes italian pasta');
INSERT INTO docs (id, body) VALUES (4, 'deep learning transformers attention');
INSERT INTO docs (id, body) VALUES (5, 'database storage engine LSM tree');
```

**Supported models:** Any BERT-style sentence-transformer from HuggingFace Hub. Common choices:

| Model | Dim | Size | Use case |
|-------|-----|------|----------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 90 MB | General purpose, fast |
| `sentence-transformers/all-mpnet-base-v2` | 768 | 420 MB | Higher quality |
| `BAAI/bge-small-en-v1.5` | 384 | 130 MB | Retrieval-optimized |

---

### SEMANTIC_MATCH

Find rows semantically similar to a query string. The threshold is cosine similarity (0.0–1.0). Higher = stricter.

```sql
-- Find AI/ML related documents (threshold 0.4 = moderately similar)
SELECT id, body
FROM docs
WHERE SEMANTIC_MATCH(body, 'artificial intelligence deep learning', 0.4);

-- Find database-related documents
SELECT id, body
FROM docs
WHERE SEMANTIC_MATCH(body, 'database index storage', 0.4);

-- Combine with SQL filters (hybrid search)
SELECT id, body
FROM docs
WHERE SEMANTIC_MATCH(body, 'machine learning', 0.5)
  AND id > 2;
```

**Threshold guide:**
- `0.8+` — very close matches only (near-duplicates)
- `0.5–0.8` — clearly related content
- `0.3–0.5` — loosely related, broader results
- `0.0` — all rows ranked by similarity, no cutoff

**Requires:** server started with `--sidecar` and `--model` flags.

---

### Time-Travel Queries — AT VERSION

Query data as it existed at a named snapshot. Useful for reproducibility, auditing, and EU AI Act compliance.

```sql
-- Create a named snapshot
CREATE VERSION TAG 'training-v1';

-- Insert more data after the snapshot
INSERT INTO docs (id, body) VALUES (6, 'new document added later');

-- Query the snapshot — only sees data from before the tag
SELECT * FROM docs AT VERSION 'training-v1';
-- Returns rows 1-5, not row 6
```

---

### Training Export

Export a versioned dataset in Lance format, ready for PyTorch with zero-copy memory mapping.

```sql
-- Create a training snapshot with options
CREATE VERSION TAG 'train-v2'
  FOR TRAINING
  WITH TRAINING PRECISION 'float32'
  TRAINING SEED 42;
```

```python
import galaxdb
import lance
import torch

db = galaxdb.Database("./data")

# Export the snapshot as a Lance dataset
path = db.training_dataset("train-v2")

# Load into PyTorch — zero-copy, memory-mapped
dataset = lance.dataset(path).to_pytorch()
loader = torch.utils.data.DataLoader(dataset, batch_size=32)
```

**Precision options:**
- `'float32'` — full precision (default)
- `'sq8'` — 8-bit scalar quantization (4× smaller)
- `'rabitq'` — RaBitQ binary quantization (32× smaller)

---

### Near-Duplicate Deduplication

Remove near-duplicate rows from training data using MinHash LSH. Typically cuts dataset size by 15–30%.

```sql
-- Select only unique documents (one representative per near-duplicate cluster)
SELECT * FROM docs WHERE NOT DUPLICATE;

-- Use in training export
CREATE VERSION TAG 'deduped-v1'
  FOR TRAINING
  WITH TRAINING PRECISION 'float32';
```

---

### Backup and Restore

```sql
-- Backup to a directory
BACKUP TO '/path/to/backup';

-- Restore from a backup
RESTORE FROM '/path/to/backup';
```

---

### ANALYZE

Update table statistics used by the adaptive query planner (HNSW vs brute-force decision).

```sql
ANALYZE docs;
-- ANALYZE docs: 8 rows sampled
```

---

### SHOW EMBEDDING HEALTH

Check the status of the embedding sidecar.

```sql
SHOW EMBEDDING HEALTH;
SHOW EMBEDDING HEALTH FOR docs;
```

---

### Observability

```bash
# Health check
curl http://localhost:9090/health
# {"status":"ok","version":"1.0.0-beta.1","subsystems":{"disk_full":false,"sidecar_healthy":true,"connections_active":2}}

# Prometheus metrics
curl http://localhost:9090/metrics
# galaxdb_connections_active 2
# galaxdb_wal_write_latency_us 42
# galaxdb_hnsw_recall_estimate_bp 9902
# galaxdb_embedding_queue_depth 0
# galaxdb_checkpoint_last_duration_ms 8
# ...
```

---

### Encryption at Rest

```bash
# Local key file
GALAXDB_KEY_PROVIDER=local:/path/to/key.bin galaxdb-server ...

# Environment variable
GALAXDB_KEY_PROVIDER=env:GALAXDB_MASTER_KEY galaxdb-server ...

# AWS KMS (via CLI)
GALAXDB_KEY_PROVIDER=command:aws kms decrypt ... galaxdb-server ...

# HashiCorp Vault Transit
GALAXDB_KEY_PROVIDER=vault:transit/galaxdb-prod galaxdb-server ...
```

---

## Complete Example — RAG Application

```python
import psycopg2

conn = psycopg2.connect(host="localhost", port=5433, dbname="galaxdb", sslmode="disable")
cur = conn.cursor()

# Create a knowledge base with automatic embeddings
cur.execute("""
    CREATE TABLE knowledge (
        id      INT PRIMARY KEY,
        title   TEXT,
        content TEXT EMBEDDING MODEL 'sentence-transformers/all-MiniLM-L6-v2' DIM 384
    )
""")

# Insert documents — embeddings computed automatically
documents = [
    (1, "Python Tutorial", "Python is a high-level programming language known for readability"),
    (2, "Rust Guide", "Rust is a systems programming language focused on safety and performance"),
    (3, "ML Basics", "Machine learning is a subset of artificial intelligence using statistical methods"),
    (4, "Database Design", "Relational databases organize data into tables with rows and columns"),
    (5, "Vector Search", "Vector similarity search finds nearest neighbors in high-dimensional space"),
]
for doc in documents:
    cur.execute("INSERT INTO knowledge (id, title, content) VALUES (%s, %s, %s)", doc)
conn.commit()

# Semantic search — no external API needed
cur.execute("""
    SELECT id, title, content
    FROM knowledge
    WHERE SEMANTIC_MATCH(content, %s, 0.4)
""", ("AI and machine learning algorithms",))

results = cur.fetchall()
for row in results:
    print(f"[{row[0]}] {row[1]}: {row[2][:60]}...")

# Create a training snapshot
cur.execute("CREATE VERSION TAG 'v1' FOR TRAINING WITH TRAINING PRECISION 'float32'")
conn.commit()

cur.close()
conn.close()
```

---

## Docker Compose

```yaml
version: '3.8'
services:
  galaxdb:
    image: harbi256/galaxdb:latest
    ports:
      - "5433:5433"
      - "9090:9090"
    volumes:
      - galaxdb_data:/data
      - ~/.cache/huggingface:/root/.cache/huggingface
    command:
      - --data-dir
      - /data
      - --port
      - "5433"
      - --observe-port
      - "9090"
      - --sidecar
      - /usr/local/bin/galaxdb-sidecar
      - --model
      - sentence-transformers/all-MiniLM-L6-v2
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9090/health"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  galaxdb_data:
```

---

## Troubleshooting

**`sidecar_healthy: false` in /health**
The server was started without `--sidecar` and `--model` flags. SEMANTIC_MATCH and embedding columns will return `SidecarUnavailable`. Restart with the flags.

**First startup is slow**
The sidecar downloads the model from HuggingFace Hub on first run (~90 MB for `all-MiniLM-L6-v2`). Mount `-v ~/.cache/huggingface:/root/.cache/huggingface` in Docker to cache it across container restarts.

**SEMANTIC_MATCH returns 0 rows**
- Check `sidecar_healthy: true` in `/health`
- Lower the threshold (try `0.3` or `0.2`)
- Make sure rows were inserted after the sidecar was running (embeddings are computed at insert time)

**Port 9090 already in use**
Use `--observe-port 9091` (or any free port).

**io_uring warning in Docker**
Expected — Docker Desktop restricts io_uring. GalaxDB automatically falls back to the tokio I/O backend. No action needed.

---

*See [BENCHMARKS.md](BENCHMARKS.md) for performance numbers and [COMPARISON.md](COMPARISON.md) for how GalaxDB compares to other databases.*
