# Vector Search Guide

GalaxDB provides native vector search using HNSW (Hierarchical Navigable Small World) graphs. This guide covers setup, usage, and tuning.

---

## Quick Start

```sql
-- 1. Create a table with an embedding column
CREATE TABLE articles (
    id INT PRIMARY KEY,
    title TEXT,
    body TEXT EMBEDDING MODEL 'all-MiniLM-L6-v2' DIM 384
);

-- 2. Insert data (embeddings generated automatically by the sidecar)
INSERT INTO articles (id, title, body) VALUES
    (1, 'Intro to ML', 'Machine learning uses algorithms to learn from data'),
    (2, 'Deep Learning', 'Neural networks with many layers can learn complex patterns'),
    (3, 'Cooking Tips', 'The best way to cook pasta is in salted boiling water');

-- 3. Search by meaning
SELECT * FROM articles
WHERE SEMANTIC_MATCH(body, 'how do neural networks learn', 0.3);
```

Results:
```
row_id | similarity
-------|----------
2      | 0.7234
1      | 0.5891
```

(Article 3 about cooking is excluded because its similarity is below the 0.3 threshold)

---

## How It Works

### On INSERT

1. Text value is sent to the embedding sidecar
2. Sidecar runs the sentence-transformer model (all-MiniLM-L6-v2)
3. Resulting 384-dim vector is normalized to unit length
4. Vector is inserted into the delta buffer (in-memory flat index)
5. Row data is written to storage (WAL + memtable + ART)

### On SEMANTIC_MATCH Query

1. Query text is embedded by the sidecar → query vector
2. HNSW graph is searched (approximate nearest neighbors, ef=200)
3. Delta buffer is searched (brute-force exact search)
4. Results are unioned and deduplicated
5. Re-ranked by exact cosine distance (fetching raw vectors from storage)
6. Tombstoned (deleted) rows are excluded
7. Similarity threshold applied
8. Top-k results returned sorted by similarity

### Background Merge

When the delta buffer exceeds `max(10,000, 1% of indexed vectors)`:
1. New HNSW graph is built incorporating delta buffer vectors
2. Written to `.hnsw.new` file, fsynced
3. Atomic rename to `.hnsw` (old graph released when in-flight queries complete)
4. Delta buffer cleared

---

## HNSW Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| M | 16 | Edges per node. Higher = better recall, more memory |
| ef_construction | 200 | Build quality. Higher = better graph, slower build |
| ef_search | 200 | Query quality. Higher = better recall, slower search |

### Tuning Guidelines

| Use Case | M | ef_construction | ef_search | Expected Recall |
|----------|---|-----------------|-----------|-----------------|
| Low latency (< 200µs) | 16 | 200 | 50 | 0.95 |
| Balanced (default) | 16 | 200 | 100 | 0.98 |
| High recall | 16 | 200 | 200 | 0.99 |
| Maximum recall | 32 | 400 | 500 | 0.999 |

---

## Supported Models

Any BERT-based sentence-transformer model from HuggingFace Hub:

| Model | Dimensions | Speed | Quality | Language |
|-------|-----------|-------|---------|----------|
| `all-MiniLM-L6-v2` | 384 | Fast | Good | English |
| `all-mpnet-base-v2` | 768 | Medium | Best | English |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Fast | Good | 50+ languages |
| `e5-small-v2` | 384 | Fast | Good | English |
| `bge-small-en-v1.5` | 384 | Fast | Good | English |

### Changing Models

When you change the model (different `--model` flag on sidecar restart):
1. Existing embeddings are marked as stale (`_embedding_stale = true`)
2. Background re-embedding is triggered automatically
3. During re-embedding, search uses old embeddings (degraded but functional)
4. `SHOW EMBEDDING HEALTH` shows progress

---

## Quantization

For training export and memory reduction:

| Method | Compression | Accuracy Loss | Use Case |
|--------|-------------|---------------|----------|
| float32 | 1× (baseline) | None | Default storage |
| SQ8 | 4× | < 1% recall loss | Training export |
| FP16 | 2× | Negligible | ARM64 deployment |
| RaBitQ | 32× | ~5% recall loss | Extreme scale |

Used via `CREATE VERSION TAG`:
```sql
CREATE VERSION TAG 'training-v1' FOR TRAINING WITH TRAINING PRECISION 'sq8';
```

---

## Performance (SIFT1M Benchmark)

Measured on AWS c6id.4xlarge (16 vCPU Ice Lake):

| Metric | GalaxDB | hnswlib (reference) |
|--------|---------|---------------------|
| Build speed | 14,728 vec/sec | 13,656 vec/sec |
| Recall@10 (ef=50) | 0.952 | 0.951 |
| Recall@10 (ef=100) | 0.980 | 0.980 |
| Recall@10 (ef=200) | 0.990 | 0.989 |
| Search QPS (ef=100) | 3,554 | — |
| Search latency (ef=100) | 281µs | — |

---

## Adaptive Query Planner

For queries combining a WHERE filter with SEMANTIC_MATCH:

```sql
SELECT * FROM articles
WHERE category = 'tech'
AND SEMANTIC_MATCH(body, 'machine learning', 0.5);
```

The planner automatically chooses the optimal strategy:

- **Filter cardinality < 1000 rows** → BruteForceFiltered (scan filtered set)
- **Filter cardinality > 1000 rows** → HnswWithPostFilter (HNSW search, then filter)

This is based on table statistics collected by `ANALYZE`.

---

## Tombstone Handling

When a row is deleted:
1. A tombstone is written to the delta buffer
2. The vector remains in the HNSW graph (not removed — expensive)
3. On query, tombstoned row_ids are excluded from results
4. On next merge, tombstoned vectors are permanently removed from the new graph

---

## Sidecar Unavailability

If the embedding sidecar crashes or is unreachable:
- `SEMANTIC_MATCH` queries return: `"semantic search temporarily unavailable — embedding sidecar is down"`
- INSERT operations continue (text is stored, embedding queued in backlog)
- When sidecar recovers, backlog is drained automatically (FIFO)
- Sidecar restarts with exponential backoff: 1s, 2s, 4s, 8s, 16s, 32s, 60s (capped)
