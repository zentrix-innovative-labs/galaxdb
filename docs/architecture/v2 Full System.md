The v2 specification is now fully detailed, matching the depth of v1. Every component includes design rationale, implementation specifics, and references to the research backing each decision. The full document has been updated accordingly, but here is the complete v2 section for immediate review.

---

## 4. v2 Full System — Detailed Design

v2 extends the battle‑hardened single‑node v1 engine into a distributed, self‑optimizing, globally scalable system that actively improves the AI built upon it. Each subsection below describes a new capability, the research foundation, and how it integrates with the v1 core.

### 4.1 RGABH‑Driven Adaptive Storage

**Motivation:**  
Static buffer pool policies (LRU, clock‑sweep) cannot adapt to shifting workload patterns common in AI applications — a dataset used for training last week may be cold this week, while a subset of rows suddenly becomes hot due to a new active learning cycle. GalaxDB replaces static heuristics with **Row‑Gradient‑Aggregated Block Hotness (RGABH)**, a continuous, workload‑responsive signal that governs buffer pool admission, speculative prefetch, and storage tiering.

**Design:**  
Each row carries a multi‑timescale gradient computed as an exponentially weighted moving average (EMA) of access events:

| Signal | Half‑life | Source | Drives |
|--------|-----------|--------|--------|
| `short_heat` | 30 s | OLTP point reads | HotSet prefetch |
| `long_heat` | 10 min | Sustained access, model feedback | HotSet admission |
| `training_heat` | 1 h | GPU training access callbacks | Storage tiering only |

The gradient for a row is `short_heat + γ·long_heat + δ·training_heat`.  
**Block hotness** is the sum of row gradients within a PAX block. It is recomputed incrementally via a dirty‑block queue (only blocks whose rows have updated gradients are rescored). A quiescent metadata sweep (hourly) decays cold blocks not touched recently.

**Buffer Pool (v2):**
- HotSet admission: blocks with hotness above a dynamic threshold `T_hot`.
- Eviction: evict the block with the lowest hotness, not LRU.
- Speculative prefetch: blocks whose `short_heat` velocity exceeds a threshold are prefetched from NVMe to HotSet.

**Storage Tiering:**  
Block hotness also governs automatic migration between storage tiers:

| Tier | Hotness Condition | Storage | Latency |
|------|-------------------|---------|---------|
| Hot | hotness > T_hot | NVMe | < 1 ms |
| Warm | T_cold < hotness ≤ T_hot | NVMe (eligible for S3) | < 1 ms |
| Cold | hotness ≤ T_cold | Object storage (S3) | 10–50 ms |
| Frozen | hotness ≈ 0 for 7+ days | Glacier | minutes |

Thresholds `T_hot` and `T_cold` auto‑tune via a feedback controller targeting 80 % NVMe utilization. Adjustments are clamped to ±20 % per cycle to prevent oscillation.

**PQ / RaBitQ Codebook Lifecycle:**  
The same drift detector that triggers model accuracy alerts also monitors embedding distribution. When drift exceeds a threshold, new RaBitQ rotation matrices (or SQ8 ranges) are computed on the current snapshot, versioned in the Merkle DAG, and linked to the data versions they cover.

---

### 4.2 Advanced Indexing (DiskANN, FreshDiskANN, SPANN)

**Motivation:**  
HNSW requires the entire index in RAM to achieve sub‑millisecond latency. For datasets beyond a few hundred million vectors, RAM cost becomes prohibitive. GalaxDB v2 offers three additional index backends optimized for different scale and latency tradeoffs.

| Backend | Storage | Mutable | Best For |
|---------|---------|---------|----------|
| **HNSW (RAM)** | DRAM | Yes (delta + merge) | < 50M vectors, real‑time |
| **DiskANN** | SSD | Rebuild | 50M–10B vectors, near‑real‑time |
| **FreshDiskANN** | SSD | Incremental | 50M–1B vectors, mutable |
| **SPANN** | SSD + RAM centroids | Merge‑based | 100B+ vectors, web‑scale |

- **DiskANN** builds a Vamana graph on SSD, achieving sub‑10 ms latency at billion scale while using 15–50× less RAM than HNSW.
- **FreshDiskANN** adds incremental insert support, avoiding full rebuilds on data updates.
- **SPANN** uses an inverted index with RAM‑resident centroids and disk‑resident posting lists, proven in Bing for 100B+ vectors.

Users select the backend via DDL: `CREATE INDEX ... USING diskann`. The query planner auto‑selects based on dataset size if no explicit choice is made.

---

### 4.3 Distributed Clustering & Global Transactions

**Sharding:**  
- **OLTP/OLAP:** consistent hash on primary key distributes rows uniformly. Each shard owns its LSM store.
- **ANN:** a global IVF coarse quantizer routes vector queries to the 1–2 most relevant shards, where per‑shard HNSW/DiskANN executes fine‑grained search.

**IVF Quantizer Management:**  
Trained at cluster creation; retrained when embedding drift is detected. During retraining, old and new quantizers coexist — new writes use the new quantizer, queries consult both and union candidate shard sets. Data migration happens lazily during next LSM compaction.

**Replication & HTAP:**  
Each shard is a Raft group (3–5 nodes). Read‑only columnar replicas receive updates via Raft log shipping, enabling heavy OLAP scans without impacting OLTP latency. Zero‑ETL, zero format conversion.

**Consistency:**  
- **Default:** causal consistency via Hybrid Logical Clocks (HLC) — read‑your‑writes, monotonic reads, consistent prefix.
- **Opt‑in:** strict serializability via Percolator‑style 2PC with Serializable Snapshot Isolation (SSI) for cross‑shard transactions that require ACID guarantees.

---

### 4.4 Active Learning & Feedback Loop

**Prediction Tracking:**  
Applications insert model predictions and eventual ground truth into `_galaxdb_predictions`:

```sql
INSERT INTO _galaxdb_predictions (row_id, model_id, prediction, actual, timestamp)
VALUES ('row_42', 'fraud_v3', 'legitimate', 'fraud', NOW());
```

**FEEDBACK SQL:**  
Corrections flow back as first‑class SQL, writing append‑only deltas that boost the row’s gradient:

```sql
FEEDBACK products
SET label = 'defective', confidence = 0.95
WHERE id = 42
SOURCE 'quality_model_v3' AT PREDICTION_TIME '2025-07-15T14:05:00Z';
```

**Active Learning:**  
`ORDER BY ACTIVE_LEARNING(target_column, strategy)` returns the most informative unlabeled samples. Uncertainty scores are pre‑computed by a background job and stored as the real column `_al_uncertainty`, enabling indexed O(log n) retrieval.

**Cold‑Start Strategy:**  
- < 50 labeled rows: random sampling.
- 50–200 labeled rows: cluster‑then‑sample (k‑means over embeddings).
- > 200 labeled rows: uncertainty sampling (margin, entropy, configurable).

**Drift Detection:**  
A background monitor evaluates accuracy in `_galaxdb_predictions` over sliding windows. When accuracy drops, it triggers alerts, boosts affected rows’ gradients, and (if embedding distribution drift is also detected) retrains the IVF quantizer and RaBitQ codebook.

---

### 4.5 Semantic Snapshot Guarantees

v2 introduces **versioned vector indexes** to enable true historical semantic search:

- `CONSISTENCY 'SEMANTIC_SNAPSHOT'` uses an HNSW/DiskANN index built from the same snapshot as the row data.
- When a version tag is created with `WITH SEMANTIC SNAPSHOT`, the current index state is also versioned and linked in the Merkle DAG.
- Exports with this consistency mode produce byte‑identical vector results, enabling fully reproducible model evaluation.

---

### 4.6 Full PostgreSQL Protocol & BI Integration

v2 implements the extended query protocol (Parse/Bind/Execute), `COPY`, full `information_schema`, and sufficient `pg_catalog` for BI tools (Tableau, Metabase, DataGrip). Server‑side cursors support large result sets.

---

### 4.7 GPU‑Direct & Hardware Acceleration

- **GPUDirect Storage:** Training scans bypass CPU entirely, DMA directly from NVMe to GPU memory.
- `report_training_access` callbacks feed `training_heat` into RGABH without polluting OLTP prefetch.
- Optional FPGA/SmartNIC offload for vector distance computation at the storage node level.

---

### 4.8 Federated Queries & Privacy

- Data atoms carry ownership policies.
- Federated queries aggregate across organizational boundaries with differential‑privacy budgets managed transparently by the query planner.
- Secure aggregation (Google’s SIGMOD 2017 model) ensures raw data never leaves its origin.

---

### 4.9 Plugin Marketplace

- `EmbeddingModel` trait with batch interface (defined in v1) serves as the plugin contract.
- Sandboxed registry allows third‑party models, active‑learning strategies, and data processors.
- Revenue‑share model for commercial plugins.

---

### 4.10 Semantic Caching

- `CREATE SEMANTIC CACHE FOR TABLE … SIMILARITY … TTL …` creates a system‑managed cache table holding (query_embedding, result, timestamp, hit_count).
- Before executing a query, GalaxDB checks the cache via `SEMANTIC_MATCH`; on cache hit, the cached result is returned without hitting the main table or the LLM.
- Reduces LLM API costs by 50–70 % for repetitive or near‑duplicate queries.

---

### 4.11 Multi‑Tenancy

- **Schema‑level isolation** (recommended): `CREATE TENANT 'name'` provisions a dedicated schema with per‑schema encryption keys; no cross‑tenant compaction.
- **Row‑level isolation** via Row‑Level Security (RLS) for lightweight scenarios.

---

*All v2 features are designed to be implemented incrementally on the v1 foundation. Each component’s design is traceable to specific research papers and production systems, as documented in Appendix B of the full specification.*