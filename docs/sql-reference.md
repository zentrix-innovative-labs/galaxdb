# SQL Reference

GalaxDB implements a PostgreSQL-compatible SQL dialect with extensions for vector search, versioning, and AI training workflows.

---

## Data Types

| Type | Description |
|------|-------------|
| `INT` / `INTEGER` | 64-bit signed integer |
| `FLOAT` / `REAL` / `DOUBLE` | 64-bit floating point |
| `TEXT` / `VARCHAR` | Variable-length UTF-8 string |
| `BOOL` / `BOOLEAN` | True/false |
| `BLOB` / `BYTEA` | Binary data |

---

## DDL (Data Definition)

### CREATE TABLE

```sql
CREATE TABLE users (
    id INT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT
);
```

#### With Embedding Column

```sql
CREATE TABLE documents (
    id INT PRIMARY KEY,
    title TEXT,
    content TEXT EMBEDDING MODEL 'all-MiniLM-L6-v2' DIM 384
);
```

The `EMBEDDING MODEL 'name' DIM n` annotation tells GalaxDB to:
1. Automatically generate vector embeddings for this column on INSERT
2. Create an HNSW index for fast similarity search
3. Enable `SEMANTIC_MATCH` queries on this column

### DROP TABLE

```sql
DROP TABLE users;
DROP TABLE IF EXISTS users;
```

---

## DML (Data Manipulation)

### INSERT

```sql
INSERT INTO users (id, name, email) VALUES (1, 'alice', 'alice@example.com');
INSERT INTO users (id, name) VALUES (2, 'bob');

-- Multi-row insert (batched for performance)
INSERT INTO documents (id, title, content) VALUES
    (1, 'Intro to ML', 'Machine learning is a subset of artificial intelligence'),
    (2, 'Deep Learning', 'Neural networks with multiple layers'),
    (3, 'Weather Report', 'The weather is sunny today');
```

When inserting into a table with an embedding column, the sidecar automatically generates embeddings. The embedding is stored in the delta buffer and becomes searchable immediately.

### SELECT

```sql
SELECT * FROM users;
SELECT id, name FROM users WHERE id = 1;
```

### UPDATE

```sql
UPDATE users SET email = 'newemail@example.com' WHERE id = 1;
```

Updating an embedding source column is not allowed (returns an error with guidance to use DELETE + INSERT instead):

```sql
-- This will error:
UPDATE documents SET content = 'new text' WHERE id = 1;
-- Error: cannot update embedding source column 'content'; use DELETE + INSERT instead
```

### DELETE

```sql
DELETE FROM documents WHERE id = 1;
```

Deleting a row with embeddings automatically tombstones the vector in the HNSW index.

---

## Vector Search

### SEMANTIC_MATCH

Search for semantically similar rows using vector embeddings:

```sql
SELECT * FROM documents
WHERE SEMANTIC_MATCH(content, 'how does machine learning work', 0.5);
```

Syntax: `SEMANTIC_MATCH(column, 'query_text', threshold)`

- `column` — the embedding column to search
- `query_text` — natural language query (embedded by the sidecar at query time)
- `threshold` — minimum cosine similarity (0.0 to 1.0). Use 0.0 to return all results ranked by similarity.

Results are returned sorted by similarity (highest first) with columns:
- `row_id` — the matching row's ID
- `similarity` — cosine similarity score (0.0 to 1.0)

### How It Works

1. Query text is sent to the embedding sidecar → produces a query vector
2. HNSW index is searched for approximate nearest neighbors
3. Delta buffer (recent inserts) is searched with brute-force
4. Results are unioned, deduplicated, and re-ranked by exact cosine distance
5. Tombstoned (deleted) rows are excluded
6. Similarity threshold is applied
7. Top-k results are returned

### Adaptive Query Planner

For hybrid queries (WHERE filter + SEMANTIC_MATCH), the planner automatically chooses:

- **HnswWithPostFilter** — when filter matches many rows (> 1000 or > 0.1% of table)
- **BruteForceFiltered** — when filter is very selective (< 1000 rows or < 0.1%)

---

## Extension Commands

### ANALYZE

Collect table statistics for the query planner:

```sql
ANALYZE documents;
```

### SHOW EMBEDDING HEALTH

Check embedding status (stale count, model versions):

```sql
SHOW EMBEDDING HEALTH;
SHOW EMBEDDING HEALTH FOR documents;
```

### CREATE VERSION TAG

Create a named snapshot for time-travel queries and training exports:

```sql
CREATE VERSION TAG 'v1.0';
CREATE VERSION TAG 'training-2024-01' FOR TRAINING;
CREATE VERSION TAG 'sq8-export' FOR TRAINING WITH TRAINING PRECISION 'sq8';
```

### BULK INSERT

High-throughput insert that bypasses the memtable:

```sql
BULK INSERT INTO documents FROM '/path/to/data.csv';
```

### BACKUP / RESTORE

```sql
BACKUP TO '/path/to/backup';
RESTORE FROM '/path/to/backup';
```

---

## Wire Protocol

GalaxDB implements the PostgreSQL wire protocol (simple query mode). Connect with any PostgreSQL client:

```bash
# psql
psql -h localhost -p 5433 -U galaxdb

# Python (psycopg2)
import psycopg2
conn = psycopg2.connect(host='localhost', port=5433, user='galaxdb')
cur = conn.cursor()
cur.execute("SELECT * FROM users")
rows = cur.fetchall()
```

### pg_catalog Support

GalaxDB implements the following pg_catalog tables for tool compatibility:

| Table | Description |
|-------|-------------|
| `pg_catalog.pg_type` | Data type metadata |
| `pg_catalog.pg_namespace` | Schema namespaces |
| `pg_catalog.pg_database` | Database list |
| `pg_catalog.pg_class` | Table/relation metadata |
| `pg_catalog.pg_attribute` | Column metadata |

Queries against unsupported pg_catalog tables return empty result sets (not errors).
