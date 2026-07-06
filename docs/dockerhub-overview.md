# GalaxDB

An AI-native database that unifies SQL, vector search, and local embeddings in a
single process. Speaks the **PostgreSQL wire protocol**, so any Postgres client
or driver connects with no changes. No external embedding API, no separate vector
database, no data pipeline — one connection string.

Apache-2.0. Source: https://github.com/zentrix-innovative-labs/galaxdb

## What's in this image

- `galaxdb-server` — the database (PostgreSQL wire protocol on `5433`, HTTP
  `/health` + `/metrics` on `9090`)
- `galaxdb-sidecar` — the local embedding model runner (opt-in; see below)

Ubuntu 24.04 base. On Linux the engine uses `io_uring` and transparently falls
back to tokio where it isn't available (e.g. inside Docker Desktop).

## Quick start

```bash
docker run -p 5433:5433 -p 9090:9090 -v /data:/data \
  harbi256/galaxdb:latest --data-dir /data
```

Then connect with any Postgres client:

```bash
psql "host=localhost port=5433 dbname=galaxdb user=galaxdb sslmode=disable"
```

Check health:

```bash
curl http://localhost:9090/health
# {"status":"ok","version":"0.4.0","subsystems":{"disk_full":false,"sidecar_healthy":false,"connections_active":0}}
```

## Semantic search with local embeddings

Embedding columns and `SEMANTIC_MATCH` need the sidecar attached. Pass
`--sidecar` and a HuggingFace model id; the model is downloaded on first run,
so mount the HF cache to persist it across restarts:

```bash
docker run -p 5433:5433 -p 9090:9090 \
  -v /data:/data \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  harbi256/galaxdb:latest \
  --data-dir /data \
  --sidecar /usr/local/bin/galaxdb-sidecar \
  --model sentence-transformers/all-MiniLM-L6-v2
```

```sql
CREATE TABLE docs (
  id   INT PRIMARY KEY,
  body TEXT EMBEDDING MODEL 'sentence-transformers/all-MiniLM-L6-v2' DIM 384
);
INSERT INTO docs (id, body) VALUES (1, 'machine learning and neural networks');

-- Top-k semantic search; add LIMIT to control how many matches come back
-- (without LIMIT, the default is the 10 nearest)
SELECT id, body FROM docs
WHERE SEMANTIC_MATCH(body, 'artificial intelligence', 0.4)
LIMIT 20;
```

## Features

- Full SQL — joins, aggregates, `GROUP BY`, transactions (snapshot isolation)
- Local embeddings + `SEMANTIC_MATCH` semantic search (HNSW vector index)
- Time-travel queries (`AT VERSION`), version tags, `FOR TRAINING` Lance export
- Near-duplicate detection (`WHERE NOT DUPLICATE`)
- SCRAM-SHA-256 auth, TLS, AES-256-GCM encryption at rest
- WAL crash recovery, PostgreSQL wire protocol, `/health` + `/metrics`

## Security — read before exposing this on a network

By default the server runs in **trusted-local mode**: any connection is
accepted as the superuser with no password. This is fine for local
development but must never be exposed to an untrusted network as-is.

For anything networked, enable authentication and set a real password:

```bash
docker run -p 5433:5433 -p 9090:9090 -v /data:/data \
  -e GALAXDB_AUTH=1 \
  -e GALAXDB_INITIAL_SUPERUSER=admin \
  -e GALAXDB_INITIAL_SUPERUSER_PASSWORD=<a-strong-password> \
  harbi256/galaxdb:latest --data-dir /data
```

Also consider `--tls-mode require` (with `--tls-cert` / `--tls-key`) to encrypt
the wire protocol in transit.

## Tags

- `latest` — most recent stable release
- `0.4.0` — pinned version

## Other install options

- Python client: `pip install galaxdb-client`
- Homebrew: `brew tap zentrix-innovative-labs/galaxdb https://github.com/zentrix-innovative-labs/galaxdb && brew install galaxdb`
- Binaries + checksums: [GitHub Releases](https://github.com/zentrix-innovative-labs/galaxdb/releases)
- One-liner installer: `curl -fsSL https://raw.githubusercontent.com/zentrix-innovative-labs/galaxdb/main/install.sh | bash`

## Ports & volumes

| Port | Purpose |
|------|---------|
| 5433 | PostgreSQL wire protocol |
| 9090 | HTTP `/health` and `/metrics` |

Mount `/data` for persistent storage. If using the sidecar, also mount
`~/.cache/huggingface` to avoid re-downloading the embedding model on restart.

## Documentation

- [Getting Started](https://github.com/zentrix-innovative-labs/galaxdb/blob/main/docs/GETTING_STARTED.md)
- [Benchmarks](https://github.com/zentrix-innovative-labs/galaxdb/blob/main/docs/BENCHMARKS.md)
- [Roadmap](https://github.com/zentrix-innovative-labs/galaxdb/blob/main/ROADMAP.md)

License: Apache-2.0
