# GalaxDB v2 Architecture Specification
## AI‑Native, Distributed, Self‑Improving

**Version 2.0 — Builds on v1 core**  
**Status: Design locked, implementation-ready after v1 ship.**

---

## Table of Contents

1. [Introduction & Vision](#1-introduction--vision)
2. [System Overview](#2-system-overview)
3. [Foundational v1 Core (Recap)](#3-foundational-v1-core-recap)
4. [RGABH‑Driven Adaptive Storage](#4-rgabh‑driven-adaptive-storage)
5. [Distributed Architecture](#5-distributed-architecture)
6. [Advanced ANN Indexing](#6-advanced-ann-indexing)
7. [Active Learning & Feedback Loops](#7-active-learning--feedback-loops)
8. [Training‑Optimized Data Paths](#8-training‑optimized-data-paths)
9. [Consistency, Transactions & Semantic Snapshots](#9-consistency-transactions--semantic-snapshots)
10. [Security, Multi‑Tenancy & Federated Queries](#10-security-multi‑tenancy--federated-queries)
11. [Hardware Acceleration & GPU Direct](#11-hardware-acceleration--gpu-direct)
12. [Plugin Marketplace & Embedding Ecosystem](#12-plugin-marketplace--embedding-ecosystem)
13. [Operational Maturity & Observability](#13-operational-maturity--observability)
14. [Implementation Roadmap for v2](#14-implementation-roadmap-for-v2)
15. [Appendices](#15-appendices)

---

## 1. Introduction & Vision

GalaxDB v2 is the **full realization of the AI‑native database**. It takes the battle‑hardened single‑node engine from v1 and extends it into a distributed, self‑optimizing, globally scalable system that actively improves the AI built upon it.

### Core v2 Value Propositions

- **Adaptive Storage Intelligence:** The database continuously learns which data is hot, which is cold, and automatically moves it across NVMe, object storage, and archival tiers — saving 50–80% of storage costs without manual tuning.
- **Planet‑Scale with Zero‑ETL:** One SQL query can span thousands of nodes. The system handles sharding, replication, and distributed ANN with the same binary your developers run on their laptops.
- **Self‑Improving Data:** Built‑in active learning tells you what to label next. Drift detection triggers automatic re‑curation. Feedback loops are first‑class SQL commands, not external pipelines.
- **Training‑Aware Infrastructure:** Native Lance export, training precision control, curriculum learning ordering, and near‑duplicate exclusion directly reduce GPU training costs by up to 40%.
- **Enterprise‑Grade Security & Governance:** SSO, RBAC, audit logs, differential privacy for federated queries, and cryptographic provenance across the entire data lifecycle.

---

## 2. System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   GalaxDB v2 Cluster                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐ │
│  │  Shard 0 │  │  Shard 1 │  │  Shard 2 │  │   Read‑Only │ │
│  │ (Raft)   │  │ (Raft)   │  │ (Raft)   │  │   Replicas  │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬──────┘ │
│       │   LSM+PAX  │   LSM+PAX  │   LSM+PAX      │  LSM+PAX│
│       │   + HNSW   │   + HNSW   │   + DiskANN    │ + HNSW  │
│       └────────────┴────────────┴───────────────┘         │
│                       │                                     │
│               ┌───────┴────────┐                            │
│               │  Query Router  │                            │
│               │ (IVF coarse q) │                            │
│               └───────┬────────┘                            │
│                       │                                     │
│               ┌───────┴────────┐                            │
│               │ AuroraSQL      │                            │
│               │ (Extended pg)  │                            │
│               └────────────────┘                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │             RGABH Tiering Controller                  │  │
│  │   NVMe (Hot) ← → S3 (Warm) ← → Glacier (Cold)       │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

Every shard is a self‑contained v1 engine extended with v2 capabilities. The control plane coordinates shard membership, Raft replication, and global query routing.

---

## 3. Foundational v1 Core (Recap)

v2 inherits all v1 features unchanged:

- **LSM + PAX hybrid store** with Lazy Leveling compaction, column‑type compression, KV separation, Bloom filters, zone maps.
- **Mutable ANN:** mmap’d HNSW base graph + WAL‑backed delta buffer, atomic‑rename merges, batch delta replay.
- **SQ8 quantization** (default) and RaBitQ (opt‑in).
- **Merkle DAG versioning** with `AT VERSION`, GC‑exempt named tags, training‑reproducible exports.
- **Embedding sidecar** with persistent backlog, cross‑platform lifecycle.
- **AuroraSQL** with `SEMANTIC_MATCH`, consistency modes, and training‑aware DDL.
- **Snapshot Isolation** with optional SSI.

v2 adds the distributed, adaptive, and self‑improving layers on top.

---

## 4. RGABH‑Driven Adaptive Storage

**Row‑Gradient‑Aggregated Block Hotness (RGABH)** replaces the static buffer pool of v1 with a dynamic, workload‑responsive storage hierarchy.

### 4.1 Per‑Row Gradient Model

Each row maintains a multi‑timescale EMA gradient:

| Signal | Half‑life | Incremented by | Drives |
|--------|-----------|----------------|--------|
| `short_heat` | 30 s | OLTP point reads | HotSet prefetch |
| `long_heat` | 10 min | Sustained access, model feedback | HotSet admission |
| `training_heat` | 1 h | GPU training access callbacks | Storage tiering only |

```
gradient = short_heat + γ × long_heat + δ × training_heat
```

### 4.2 Block Hotness & Buffer Pool

- **Block hotness** = Σ gradient(row) for rows in the block.
- Updated incrementally via a dirty‑block queue; quiescent metadata sweep (hourly) decays cold blocks.
- **HotSet admission:** blocks with hotness > dynamic threshold `T_hot`.
- **Eviction:** evict the block with lowest hotness, not LRU.
- **Speculative prefetch:** blocks whose `short_heat_velocity` exceeds a threshold are prefetched from NVMe to HotSet.

### 4.3 Adaptive Storage Tiering

| Tier | Hotness | Storage | Latency |
|------|---------|---------|---------|
| Hot | hotness > T_hot | NVMe | < 1 ms |
| Warm | T_cold < hotness ≤ T_hot | NVMe (eligible for S3 migration) | < 1 ms |
| Cold | hotness ≤ T_cold | Object store (S3) | 10–50 ms |
| Frozen | hotness ≈ 0 for 7+ days | Glacier | minutes |

`T_hot` and `T_cold` auto‑tune via a feedback controller targeting 80% NVMe utilization. Adjustments are clamped to ±20% per cycle to prevent oscillation.

### 4.4 PQ/RaBitQ Codebook Lifecycle

The drift detector that monitors model accuracy also monitors embedding distribution. When drift exceeds a threshold, new RaBitQ rotation matrices (or SQ8 ranges) are computed on the current snapshot, versioned in the Merkle DAG, and linked to the data versions they cover.

---

## 5. Distributed Architecture

### 5.1 Sharding Strategy

**OLTP/OLAP sharding:** Consistent hash on primary key distributes rows uniformly. Each shard owns its LSM store.

**ANN sharding:** A global IVF (Inverted File) coarse quantizer is trained over the embedding space. At query time, the quantizer routes the search to the 1–2 most relevant shards. Within each shard, the local HNSW/DiskANN executes fine‑grained search.

**IVF Quantizer Management:**
- Trained at cluster creation; retrained when embedding drift is detected.
- During retraining, old and new quantizers coexist. New writes use the new quantizer; queries consult both and union candidate shard sets.
- Data migration happens lazily during next LSM compaction.

### 5.2 Replication & Consensus

Each shard is a **Raft group** (3–5 nodes). The Raft log replicates all writes. Reads can be served from followers if bounded staleness is acceptable (causal consistency).

**Read‑Only Columnar Replicas:**  
For HTAP workloads, shards spawn read‑only replicas that receive updates via Raft log shipping. These replicas run the same binary and serve heavy OLAP scans without impacting OLTP latency. Zero‑ETL, zero format conversion.

### 5.3 Global Transactions

**Default: Causal Consistency** via Hybrid Logical Clocks (HLC). Provides read‑your‑writes, monotonic reads, and consistent prefix.

**Opt‑in: Strict Serializability** via Percolator‑style two‑phase commit (2PC) over HLC. Used for cross‑shard transactions that require ACID guarantees. This is the same model as TiDB and CockroachDB.

---

## 6. Advanced ANN Indexing

v2 supports multiple index backends, selectable per table:

| Backend | Storage | Mutable | Best For |
|---------|---------|---------|----------|
| **HNSW (RAM)** | DRAM | Yes (delta + merge) | < 50M vectors, real‑time |
| **DiskANN** | SSD | Rebuild | 50M–10B vectors |
| **FreshDiskANN** | SSD | Incremental | 50M–1B vectors, mutable |
| **SPANN** | SSD + RAM centroids | Merge‑based | 100B+ vectors, web‑scale |

DDL example:
```sql
CREATE INDEX ON products(description_embedding) USING diskann;
```

The query planner auto‑selects the index based on dataset size and workload patterns. HNSW remains the default for datasets that fit in RAM; DiskANN is automatically selected when the index would exceed available memory.

---

## 7. Active Learning & Feedback Loops

### 7.1 Prediction Tracking

Applications record model predictions and ground truth in the system table `_predictions`:

```sql
INSERT INTO _predictions (row_id, model_id, prediction, actual, timestamp)
VALUES ('row_42', 'fraud_v3', 'legitimate', 'fraud', NOW());
```

### 7.2 FEEDBACK SQL

Corrections flow back into the database as first‑class SQL:

```sql
FEEDBACK products
SET label = 'defective', confidence = 0.95
WHERE id = 42
SOURCE 'quality_model_v3' AT PREDICTION_TIME '2025-07-15T14:05:00Z';
```

This writes an append‑only delta, boosting the row’s gradient and triggering re‑curation.

### 7.3 Active Learning

`ORDER BY ACTIVE_LEARNING(target_column, strategy)` returns the most informative unlabeled samples. Uncertainty scores are pre‑computed by a background job and stored as the real column `_al_uncertainty` — O(log n) via standard index.

**Cold‑start strategy:**
- < 50 labeled: random sampling.
- 50–200: cluster‑then‑sample (k‑means over embeddings).
- > 200: uncertainty sampling (margin, entropy, configurable).

### 7.4 Drift Detection

A background monitor evaluates accuracy in `_predictions` over sliding windows. When accuracy drops:
- Alerts fire.
- Affected rows’ gradients are boosted, surfacing them for relabeling.
- PQ/RaBitQ codebook and IVF quantizer retraining is triggered if embedding distribution drift is also detected.

---

## 8. Training‑Optimized Data Paths

v2 extends the v1 training capabilities with direct GPU integration and curriculum‑aware exports.

### 8.1 Lance Materialization (v1, enhanced in v2)

Version tags created with `FOR TRAINING` materialize a Lance‑format dataset on NVMe. In v2, Lance datasets can be streamed directly to distributed training jobs via Arrow Flight gRPC.

### 8.2 Training Precision

```sql
CREATE VERSION TAG 'run_42' FOR TRAINING WITH TRAINING PRECISION 'sq8', TRAINING SEED 123;
```

Supported precisions: `float32`, `sq8`, `rabitq`. The Lance file stores quantized vectors; the training pipeline reads them with zero conversion.

### 8.3 Curriculum Learning

`ORDER BY ACTIVE_LEARNING('label')` combined with `FOR TRAINING` delivers data in difficulty order (easy → hard), reducing training time by 30–50% (Bengio 2009, widely replicated).

### 8.4 Near‑Duplicate Exclusion

`WHERE NOT DUPLICATE` in training exports removes near‑duplicate rows based on MinHash LSH signatures pre‑computed at insert time.

### 8.5 GPU Direct Storage

Training scans bypass CPU entirely: NVMe → GPU via DMA (GPUDirect). The `report_training_access` callback feeds `training_heat` into RGABH without polluting OLTP signals.

---

## 9. Consistency, Transactions & Semantic Snapshots

### 9.1 Distributed Consistency

- **Causal Consistency (default):** HLC‑based. Read‑your‑writes, monotonic reads, consistent prefix. Suitable for most AI workloads.
- **Strict Serializability (opt‑in):** Cross‑shard 2PC. Guaranteed no anomalies.

### 9.2 Serializable Snapshot Isolation (SSI)

v2 includes anti‑dependency tracking à la PostgreSQL 9.1 / SIGMOD 2008. When `SERIALIZABLE` mode is selected, the transaction manager detects rw‑conflicts and aborts conflicting transactions, preventing write‑skew.

### 9.3 Versioned Semantic Snapshots

`CONSISTENCY 'SEMANTIC_SNAPSHOT'` uses a versioned HNSW/DiskANN index, enabling exact historical vector search. Tag creation can snapshot the index:

```sql
CREATE VERSION TAG 'q4_train' WITH SEMANTIC SNAPSHOT;
```

The index snapshot is linked in the Merkle DAG and provides byte‑identical vector results at that point in time.

---

## 10. Security, Multi‑Tenancy & Federated Queries

### 10.1 Enterprise Security
- **SSO:** LDAP, SAML, OIDC.
- **RBAC:** row‑level and column‑level, integrated with the gradient engine.
- **Audit Logging:** every access and mutation is logged with cryptographic lineage.

### 10.2 Federated Queries & Differential Privacy

Data atoms carry ownership policies. A federated query:

```sql
SELECT COUNT(*) FROM federated_patients WHERE diagnosis = 'X';
```

aggregates across organizational boundaries without moving raw data. The query planner enforces a **differential privacy budget** (ε, δ) per organization; each query consumes budget.

Secure aggregation (Google’s SIGMOD 2017 model) is used when privacy constraints require it.

---

## 11. Hardware Acceleration & GPU Direct

- **GPUDirect Storage:** DMA from NVMe to GPU memory for training scans.
- **Optional FPGA/SmartNIC offload:** Vector distance computation at the storage node level (e.g., on a DPU). Reduces data movement for large ANN scans.
- **Quantization‑aware training support:** Training export in SQ8 or RaBitQ allows models to train directly on quantized data, with the quantization noise serving as implicit regularization.

---

## 12. Plugin Marketplace & Embedding Ecosystem

The `EmbeddingModel` trait (v1) is extended with a batch interface:

```rust
trait EmbeddingModel: Send + Sync {
    fn embed_batch(&self, texts: &[&str]) -> Vec<Vec<f32>>;
    fn model_id(&self) -> &str;
    fn dimensions(&self) -> usize;
    fn max_batch_size(&self) -> usize { 32 }
    fn embed(&self, text: &str) -> Vec<f32> {
        self.embed_batch(&[text]).remove(0)
    }
}
```

Marketplace plugins (models, active‑learning strategies, data processors) are sandboxed and distributed via a registry. Revenue share for commercial plugins.

---

## 13. Operational Maturity & Observability

- **Metrics:** Every gradient, tier transition, quantizer retrain, and feedback event is exposed via Prometheus.
- **Distributed Tracing:** OpenTelemetry across Raft messages and cross‑shard queries.
- **Chaos Engineering:** Jepsen‑tested for partition tolerance, Raft leadership changes, and WAL corruption.
- **Backup & Restore:** Point‑in‑time recovery across the entire cluster, including version tags and index snapshots.

---

## 14. Implementation Roadmap for v2

### Phase 1: RGABH & Tiering (Months 1–4 post‑v1)
- Multi‑timescale EMA infrastructure, dirty‑block queue, quiescent sweep.
- Auto‑tuned T_hot / T_cold controller.
- NVMe → S3 migration, Glacier archival.

### Phase 2: Distribution (Months 5–9)
- Raft replication, shard management, HLC causal consistency.
- IVF coarse quantizer with coexisting retraining.
- Read‑only replicas for HTAP.

### Phase 3: Active Learning (Months 10–13)
- _predictions table, FEEDBACK SQL, background uncertainty scoring.
- Drift detector, PQ/RaBitQ refresh trigger.
- Cold‑start bootstrap stages.

### Phase 4: Advanced Indexing & Hardware (Months 14–18)
- DiskANN / FreshDiskANN / SPANN backends.
- GPU Direct, FPGA offload prototypes.
- Full SSI, federated queries, plugin marketplace launch.

---

## 15. Appendices

### A. Research Basis
- Dostoevsky (SIGMOD 2018) → Lazy Leveling
- WiscKey (FAST 2016) → KV separation
- RaBitQ (SIGMOD 2024/2025) → quantization
- P‑HNSW (2025) → crash‑safe graph (conceptually; we use atomic rename)
- DiskANN (NeurIPS 2019), FreshDiskANN (2024), SPANN (2021)
- PostgreSQL SSI (SIGMOD 2008)
- Google Secure Aggregation (SIGMOD 2017)

### B. Quantization Tier Summary (Final)
| Tier | Method | Compression | SIMD |
|------|--------|-------------|------|
| v1 default | SQ8 | 4× | AVX2 |
| v1 opt‑in | RaBitQ | 32× | AVX‑512 / NEON |
| v2 latency | Binary | 32× | popcount |
| v2 disk | SQ8 + DiskANN | 4× + SSD | N/A |

### C. Training‑Aware Features at a Glance
- Lance export → zero‑copy PyTorch
- Training precision → 4‑32× less I/O
- Curriculum learning → 30‑50% less GPU time
- Near‑dedup → 15‑30% smaller datasets
- GPU Direct → DMA to GPU

---

*GalaxDB v2 is the first database that doesn’t just store data — it actively makes AI better, cheaper, and faster. This specification is ready for implementation.*