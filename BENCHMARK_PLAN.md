# Andromeda Benchmark Plan (v1)

This runbook defines how to benchmark Andromeda v1 against major existing databases for OLTP, ANN, hybrid search, versioned snapshots, and failure recovery.

## 1) Objectives

1. Validate correctness guarantees in the final architecture spec.
2. Quantify latency, throughput, recall, and durability behavior.
3. Compare against representative systems under identical hardware and dataset conditions.
4. Produce a repeatable benchmark package for internal and external validation.

## 2) Competitor Matrix

1. PostgreSQL 18 + pgvector
2. Qdrant (latest stable)
3. Weaviate (latest stable)
4. Milvus (latest stable)
5. SQLite (latest stable)
6. DuckDB (latest stable)

Notes:
1. Use single-node modes for all systems in v1 comparison.
2. Match vector dimension, metric, and top-k across all runs.
3. Disable distributed features in systems where possible to keep parity with Andromeda v1 scope.

## 3) Hardware Profiles

Run each workload on all three profiles.

1. Profile A (Developer)
- 8 vCPU
- 32 GB RAM
- 1 TB NVMe SSD
- Linux x86-64

2. Profile B (Target Production Single Node)
- 16 vCPU
- 64 GB RAM
- 2 TB NVMe SSD
- Linux x86-64

3. Profile C (Stress)
- 32 vCPU
- 128 GB RAM
- 4 TB NVMe SSD
- Linux x86-64

## 4) Datasets

Use public datasets where possible.

1. Row OLTP
- 100M rows synthetic orders/products/users schema
- Zipfian key distribution and mixed point read/write patterns

2. ANN
- SIFT1M (128 dim)
- GloVe-100 (100 dim)
- Cohere or MiniLM embeddings synthetic corpus (384 dim) at 10M vectors

3. Hybrid SQL + Vector
- Products table: 10M rows
- Columns: id, category, price, inventory, description, embedding
- Mixed predicate + semantic search queries

4. Versioning and Arrow Export
- Daily snapshots over 30 days
- Named tags at weekly boundaries

## 5) Workload Suite

1. W1: OLTP Row Workload
- 70% point reads, 20% updates, 10% inserts
- Measure TPS, p50, p95, p99 latency
- Isolation checks for consistency guarantees

2. W2: ANN Retrieval
- k = 10 and k = 100
- cosine similarity
- Measure QPS, p95/p99 latency, Recall@10, Recall@100
- Sweep search depth settings and report frontier

3. W3: Hybrid Query Workload
- Query pattern: structured filter + semantic match + order by similarity
- Measure end-to-end latency and throughput
- Measure planner path choices (graph traversal vs filtered brute force)

4. W4: Freshness and Index Merge
- Sustained inserts with embedding backlog pressure
- Measure backlog depth, stale row percentage, merge duration, query impact
- Validate no silent data loss and eventual indexing

5. W5: Versioned Queries and Reproducibility
- AT VERSION row snapshot queries
- Semantic guardrail behavior validation
- Arrow export reproducibility by byte-level hash of output batches

6. W6: Durability and Recovery
- Kill sidecar mid-request
- Kill DB mid-flush
- WAL corruption simulation
- Disk-full scenario
- Measure recovery time and data correctness

## 6) Measurement Standards

1. Always report:
- Throughput: TPS or QPS
- Latency: p50, p95, p99
- Recall metrics for ANN
- CPU utilization, RAM usage, disk read/write MB/s
- Startup and crash recovery time

2. Run policy:
- Warmup: 10 minutes
- Measurement window: 30 minutes
- Repetitions: 5 runs per test per profile
- Report median and variance

3. Correctness checks:
- Result set equivalence for deterministic row queries
- Recall computed against brute-force ground truth
- Snapshot hash consistency for repeated exports

## 7) Tooling

1. Load generation
- OLTP: custom Rust harness or pgbench-compatible driver
- ANN and hybrid: Python runner using identical query sets for all engines

2. Ground truth
- Exact brute-force cosine on sampled subsets

3. Observability
- Exported metrics endpoint for Andromeda
- System-level stats from Linux perf and iostat

4. Result packaging
- CSV per run
- JSON summary per workload
- Markdown report with plots

## 8) Success Gates (v1)

Andromeda v1 passes when all are true on Profile B:

1. Correctness
- Zero committed data loss in all W6 tests
- Semantic guardrails behave exactly as specified
- Versioned export reproducibility hash is stable across repeated runs

2. Performance
- W1: p99 row point-read latency <= 15 ms at target load
- W2: Recall@10 >= 0.95 with p95 <= 20 ms on 10M vectors (384 dim)
- W3: Hybrid query p95 <= 30 ms at agreed target QPS

3. Operability
- Recovery time <= 30 seconds after crash scenarios
- Backlog drains to steady-state after sidecar restoration

## 9) Benchmark Timeline (6 Weeks)

1. Week 1
- Harness implementation
- Dataset preparation
- Environment standardization

2. Week 2
- Baseline runs for PostgreSQL + pgvector, SQLite, DuckDB

3. Week 3
- Baseline runs for Qdrant, Weaviate, Milvus

4. Week 4
- Andromeda v1 runs on all workloads

5. Week 5
- Failure injection and recovery suite
- Repeatability and variance checks

6. Week 6
- Analysis, plots, pass/fail decision
- Optimization backlog and next milestone definition

## 10) Report Template

Each workload report should include:

1. Test configuration
2. Exact software versions
3. Hardware profile
4. Raw metrics table
5. Percentile latency plots
6. Recall-latency frontier plots (ANN/hybrid)
7. Key regressions and suspected causes
8. Pass/fail gate status

## 11) Fairness Rules

1. Same dataset, same query set, same top-k, same metric.
2. No vendor-specific extensions unless mirrored across systems.
3. Tune each system using documented best practices, and publish all settings.
4. Keep all scripts and configs version-controlled.
