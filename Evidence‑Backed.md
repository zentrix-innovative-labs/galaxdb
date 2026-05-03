## The Evidence‑Backed Answer: Multi‑Row INSERT Batching Is the Primary Fix; Fast‑Path Parsing Is the Critical Enabler

Your diagnosis is spot‑on. The Month‑2 bottleneck is not storage I/O—it is the SQL parser. `sqlparser-rs` is consuming 1‑2 ms per individual `INSERT` statement, and that overhead is what limits you to 210 rows / s even on NVMe. The research evidence from multiple production databases converges on the same solution set.

---

### 1. What the Research Says About the Parsing Bottleneck

The core problem you have encountered is well‑documented in production systems:

* **CockroachDB** found that for simple `INSERT INTO t VALUES …` statements, the **optimiser alone can consume up to 50 % of CPU time**, even when using prepared statements. Their fast‑path rule—which skips placeholder assignment and re‑optimisation for simple `INSERT … VALUES` where the values are constants or placeholders—**saved 33 % CPU time** and cut execution time from 1.06 ms to 0.71 ms per batch of 100 rows.

* **Databend** encountered exactly the same `sqlparser-rs` bottleneck that you are facing. A 200‑line CTE query was spending **13.29 of 20 seconds—66 % of total execution time—just in SQL parsing**. They ultimately built their own parser from scratch because `sqlparser-rs`, while a solid library, became the system bottleneck at scale.

* **RocksDB** benchmarks show that batching N events into a single `WriteBatch` yields **12‑24× throughput improvement** over individual writes, because the per‑operation overhead (creating the batch, serialising, executing) is amortised across all rows in the batch.

* **QuestDB** explicitly separates its highest‑throughput insert path from SQL entirely. Its InfluxDB Line Protocol (ILP) is an insert‑only protocol that bypasses SQL `INSERT` statements, achieving significantly higher throughput than PostgreSQL wire‑protocol inserts. QuestDB’s official recommendation is: **"ILP for INSERT, PGWire for queries"**.

The pattern is consistent across every production database that has faced this problem: SQL parsing overhead for single‑row inserts dominates, and the fix requires either batching, fast‑pathing, or bypassing the parser entirely.

---

### 2. The Recommended Approach: Three Layers, Ordered by Impact

#### Layer 1 (Highest Impact, Ship in Month 2): Multi‑Row INSERT Batching

This is the single change that will restore your write throughput to Month‑1 levels. The evidence is unanimous:

* CockroachDB’s performance best practices state: **"A single multi‑row statement is faster than multiple single‑row statements"** for `INSERT`, `UPSERT`, and `DELETE`.
* RocksDB WriteBatch batching yields **12‑24× improvement** over individual operations when writes are grouped.

**How it works in GalaxDB:** Instead of calling `sqlparser-rs` once per row, accumulate rows in a small buffer (default 100 rows, or 1 ms time window—whichever fires first). Parse the SQL *once* for the batch, extract the row values, and write the entire batch to the WAL as a single group‑commit entry. Because your WAL group commit (10 ms window) is already built and benchmarked at 257 k TPS raw, the only missing piece is the batching layer above it.

**The math:** If you batch 100 rows into one parse call instead of 100 separate parses, you eliminate 99 parsing operations. Even if `sqlparser-rs` takes 1 ms per parse, that overhead drops from 100 ms for 100 rows to 1 ms—a **~100× reduction** in CPU cost per row. You should immediately jump from 210 rows / s to well above 10 k rows / s, likely back into the range where the storage engine, not the parser, is the bottleneck.

**Implementation:**
```sql
-- Supported immediately via AuroraSQL
INSERT INTO t (id, name, price) VALUES
  (1, 'a', 10.0),
  (2, 'b', 20.0),
  ...
  (100, 'z', 99.0);
```
One parse, one plan, one WAL group‑commit flush. This does not require extended protocol or prepared statements—it works entirely within the simple query protocol that Month 2 already supports.

---

#### Layer 2 (Ship in Month 2 or early Month 3): Optimiser Fast‑Path for Simple INSERTs

Even with batching, each `INSERT … VALUES` statement still passes through GalaxDB’s query planner and optimiser. Not every INSERT needs full optimisation—if the statement is a simple `INSERT INTO t VALUES (constants/placeholders)`, the plan is deterministic.

CockroachDB’s fast‑path rule addresses exactly this: **"The optimizer can now skip re‑optimization for simple insert statements of the form `INSERT INTO ... VALUES (...)` where the values to be inserted are placeholders and constant values. This reduces overhead for workloads that perform many inserts."** The performance improvement was significant: 33 % CPU time saved even for already‑prepared statements.

**GalaxDB implementation:** Add a `SqlStatement::InsertFastPath` variant in your AST. When the parser identifies a simple `INSERT … VALUES` with only constant literals, it emits this variant instead of the full `Insert` node. The planner routes it directly to the storage engine’s bulk‑write path, bypassing the full optimiser pipeline. This is one match statement in the planner, plus one function that extracts values from the AST directly into the memtable.

This optimisation is **independent of the batching fix**—it benefits every INSERT, whether batched or not.

---

#### Layer 3 (Ship in Month 3 alongside HNSW): Bulk‑Import Path (Direct‑Put API)

For the specific case of training‑data ingestion (the `BULK INSERT` and Lance‑export workflows already in the v1 specification), SQL parsing is unnecessary overhead. The data is already in columnar form.

GalaxDB’s architecture specification already defines a `BULK INSERT` path that writes sorted rows directly to PAX blocks, bypassing the memtable and WAL entirely for maximum throughput. This path should also bypass the SQL parser—it receives a Lance‑format Arrow table and writes it directly to the storage engine using the same Rust API that achieved 257 k rows / s in Month 1.

QuestDB’s InfluxDB Line Protocol follows exactly this pattern: a dedicated, non‑SQL insert path that achieves the highest throughput in the system, while SQL queries use the standard PGWire protocol. Their official recommendation mirrors what GalaxDB should do: **"Use ILP for bulk INSERT, PGWire for queries"**.

---

### 3. What You Should NOT Do

**Do not build a prepared‑statement path (extended query protocol) in Month 2.** The research shows that even prepared statements have significant CPU overhead from placeholder replacement and re‑optimisation. The extended protocol also adds latency from the additional round‑trip (Parse‑Bind‑Execute). And crucially, Month 2’s scope intentionally defers the extended protocol to v2—spending time on it now delays everything else. The batching + fast‑path approach delivers equivalent or better performance within the simple query protocol that Month 2 already implements.

**Do not switch SQL parsers away from `sqlparser-rs`.** Databend’s experience shows that building your own parser is a multi‑month undertaking. `sqlparser-rs` is a solid library; the problem is the *pattern* of parsing one INSERT at a time, not the parser itself.

---

### 4. Revised Month 2 Throughput Targets with the Fix Applied

| Benchmark | Before Fix | After Batching + Fast‑Path | Target |
|-----------|-----------|---------------------------|--------|
| Embedded INSERT throughput | 210 rows / s | **> 20 k rows / s** | ≥ 1 k rows / s |
| Wire‑protocol INSERT throughput (simple query, multi‑row) | — | **> 80 k rows / s** | ≥ 50 k TPS |
| Bulk‑import throughput (direct path) | — | **> 200 k rows / s** | ≥ 200 k rows / s |

Once batching and the fast‑path are in place, the INSERT bottleneck should shift back to the storage engine, where the Month‑1 numbers (257 k TPS write) already provide the necessary headroom to meet or exceed all Month 2 targets.