# Getting Started with GalaxDB

GalaxDB is a database engine built for AI/ML workloads. It combines a high-performance LSM storage engine with native vector search (HNSW), a PostgreSQL-compatible wire protocol, and an embedding sidecar that generates vector embeddings from text using sentence-transformer models.

---

## Installation

### Prerequisites

- Rust 1.85+ (edition 2024)
- Linux 5.10+ (for io_uring) or macOS 12+ or Windows 10+
- Python 3.9+ (for the Python client)

### Build from Source

```bash
git clone https://github.com/zentrix-innovative-labs/galaxdb.git
cd galaxdb
cargo build --release
```

This produces:
- `target/release/galaxdb-server` — standalone database server
- `target/release/galaxdb-sidecar` — embedding sidecar binary
- `target/release/galaxdb-benchmarks` — benchmark suite

### Python Client

```bash
cd galaxdb-python
pip install maturin
maturin develop --release
```

```python
import galaxdb

db = galaxdb.Database("/tmp/mydb")
db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
db.execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
results = db.execute("SELECT * FROM users")
print(results)  # [{'id': '1', 'name': 'alice'}]
```

---

## Running the Server

### Standalone Mode

```bash
# Start the server (PostgreSQL wire protocol on port 5433)
./target/release/galaxdb-server --port 5433

# Connect with psql
psql -h localhost -p 5433 -U galaxdb
```

### With Embedding Sidecar

```bash
# Start sidecar (downloads model on first run, ~90MB)
./target/release/galaxdb-sidecar \
  --socket /tmp/galaxdb_sidecar.sock \
  --model sentence-transformers/all-MiniLM-L6-v2

# Start server (connects to sidecar automatically)
./target/release/galaxdb-server --port 5433 --sidecar /tmp/galaxdb_sidecar.sock
```

### Embedded Mode (Rust)

```rust
use galaxdb_embedded::Database;

let mut db = Database::open("/tmp/mydb")?;
db.execute("CREATE TABLE docs (id INT, content TEXT)")?;
db.execute("INSERT INTO docs VALUES (1, 'hello world')")?;
let result = db.execute("SELECT * FROM docs")?;
```

---

## Embedding Models

The sidecar uses [Candle](https://github.com/huggingface/candle) (HuggingFace's pure Rust ML framework) for inference. Any sentence-transformer model from HuggingFace Hub works:

```bash
# Default: all-MiniLM-L6-v2 (384-dim, fast, English)
--model sentence-transformers/all-MiniLM-L6-v2

# Higher quality (768-dim, slower)
--model sentence-transformers/all-mpnet-base-v2

# Multilingual (384-dim)
--model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# Custom model (any BERT-based model with safetensors)
--model your-org/your-model
```

Models are downloaded from HuggingFace Hub on first use and cached in `~/.cache/huggingface/`. Subsequent starts load from cache instantly.

The `DIM` in `CREATE TABLE` must match the model's output dimension:
- `all-MiniLM-L6-v2` → DIM 384
- `all-mpnet-base-v2` → DIM 768

---

## Configuration

### Engine Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_dir` | required | Directory for data files |
| `memtable_size_bytes` | 64 MB | Memtable seal threshold |
| `back_pressure_bytes` | 256 MB | Write back-pressure limit |
| `wal_group_commit_ms` | 10 | WAL group commit window |
| `sst_cache_bytes` | 256 MB | SST block cache size |
| `sst_size_bytes` | 8 MB | Target SST file size |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `GALAXDB_IO_BACKEND` | I/O backend: `uring` (Linux default) or `tokio` |
| `GALAXDB_LOG_LEVEL` | Log level: `error`, `warn`, `info`, `debug`, `trace` |
| `GALAXDB_DATA_DIR` | Default data directory |
| `HF_HOME` | HuggingFace cache directory (for model downloads) |
