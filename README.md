<p align="center">
	<img src="assets/GalaxDB-avatar.svg" alt="GalaxDB avatar" width="180" />
</p>

<h1 align="center">GalaxDB</h1>

<p align="center">
	<strong>AI-native database architecture</strong><br />
	Transactional + Analytical + Vector + Versioned data in one engine.
</p>

<p align="center">
	<img src="https://img.shields.io/badge/status-architecture_ready-1f6feb" alt="status" />
	<img src="https://img.shields.io/badge/v1-hardened-0b8f6a" alt="v1" />
	<img src="https://img.shields.io/badge/v2-designed-0c7db7" alt="v2" />
	<img src="https://img.shields.io/badge/license-Apache--2.0-333333" alt="license" />
</p>

## Why GalaxDB

GalaxDB is designed as a single-engine model where row storage, columnar scans, semantic retrieval, and versioned reproducibility are first-class capabilities instead of bolt-ons.

Core intent:
- OLTP row performance with durable write semantics
- OLAP-friendly scan behavior via hybrid storage strategy
- Mutable ANN retrieval with freshness controls
- Time-travel snapshots for reproducible analytics and model workflows

## Current Stage

- Architecture is finalized and documented.
- v1 scope is hardened and implementation-ready.
- v2 scope is fully designed.

## Architecture Highlights

- LSM + PAX storage layout for mixed workloads
- Mutable ANN path (base graph + delta buffer + merge policy)
- Embedding sidecar with durable backlog and degraded mode behavior
- PostgreSQL wire protocol Tier 1 (simple query protocol)
- Merkle DAG versioning with tag pinning and reproducibility controls

## Repository Guide

- [Final Version.md](Final%20Version.md): final architecture specification
- [Final Version 2.1 — v1 Hardened, v2 Fully Designed.md](Final%20Version%202.1%20%E2%80%94%20v1%20Hardened,%20v2%20Fully%20Designed.md): hardened revision with audit mapping
- [GalaxDB Architecture Specification.md](GalaxDB%20Architecture%20Specification.md): earlier consolidated version
- [GalaxDB v1 Architecture Specification.md](GalaxDB%20v1%20Architecture%20Specification.md): v1-focused baseline
- [BENCHMARK_PLAN.md](BENCHMARK_PLAN.md): concrete benchmark execution runbook

## Benchmark Scope

The benchmark suite validates claims against representative systems:

- PostgreSQL + pgvector
- Qdrant
- Weaviate
- Milvus
- SQLite
- DuckDB

Primary scorecard:
- Throughput (TPS/QPS)
- Latency (p50, p95, p99)
- Recall@10 and Recall@100 for ANN workloads
- CPU, memory, disk efficiency
- Crash recovery time and correctness after failure injection

See [BENCHMARK_PLAN.md](BENCHMARK_PLAN.md) for exact workload definitions, hardware profiles, success gates, and timeline.

## Planned Build Path

This repository currently contains design and planning artifacts.
Implementation should be introduced in staged modules:

1. Storage + WAL core
2. Query execution path and protocol layer
3. Vector indexing pipeline
4. Versioning and export stack
5. Benchmark harness and regression automation

## Organization

Developed by Zentrix Innovative Labs Limited.

## License

Apache License 2.0. See [LICENSE](LICENSE).
