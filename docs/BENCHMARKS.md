# GalaxDB Benchmark Results

> Governing rule: `.kiro/steering/engineering-principles.md` §4 — every number published here must be reproducible from a named command against a named dataset on named hardware, with `--release`. Random-vector HNSW benchmarks are not reported.

---

## How to reproduce

Every number in this document, without exception, must be produced by a real command against a real dataset on real hardware. The canonical orchestration is:

```bash
# Pre-reqs on the workstation:
#   aws CLI configured (AWS_PROFILE or env vars)
#   rsync / ssh / scp installed
#   $GALAXDB_SSH_KEY pointing at the private key for the test instance
#
# The script starts i-0b2dec9226f62db65, rsyncs the tree, downloads +
# hash-verifies SIFT1M on the instance's local NVMe, runs the full
# release build and test suite, runs the SIFT1M recall + ef_search
# sweep, scps results back, and ALWAYS stops the instance in a trap
# handler (Ctrl-C and SSH timeout still trigger the stop).
GALAXDB_SSH_KEY=~/.ssh/galaxdb-test.pem \
    scripts/aws-integration-run.sh

# Results land in bench-results/<UTC-timestamp>/
#   sift_bench.json   — full provenance report (schema below)
#   test.log          — cargo test output
#   run_metadata.txt  — commit, instance type, dataset hash, timestamps
```

The Month 1/2 non-vector benchmarks (OLTP, OLAP, mixed, cold-cache, crypto) each have named commands listed inline below, each run against AWS `c6id.4xlarge` instance `i-0b2dec9226f62db65` with the same hardware specs noted under **Hardware**.

---

## Current results

### Vector search — SIFT1M on AWS `c6id.4xlarge`

Populated by Phase G2 of the consolidation sprint (see `docs/CONSOLIDATION.md`). **Last run: pending.** The Phase G infrastructure (orchestration script, SIFT1M benchmark binary, SHA256 verification, stop-on-exit trap) is committed; the user-initiated AWS run will populate the table below.

| commit_sha | instance | cpu | ram_gb | dataset sha256 | build_time_ms | ef | recall@10 | mean_latency_µs | p99_latency_µs |
|---|---|---|---|---|---|---|---|---|---|
| _pending_ | c6id.4xlarge | _pending_ | _pending_ | _pending_ | _pending_ | 10 | _pending_ | _pending_ | _pending_ |
| _pending_ | c6id.4xlarge | _pending_ | _pending_ | _pending_ | _pending_ | 50 | _pending_ | _pending_ | _pending_ |
| _pending_ | c6id.4xlarge | _pending_ | _pending_ | _pending_ | _pending_ | 100 | _pending_ | _pending_ | _pending_ |
| _pending_ | c6id.4xlarge | _pending_ | _pending_ | _pending_ | _pending_ | 200 | _pending_ | _pending_ | _pending_ |

No numbers are published here until `scripts/aws-integration-run.sh` completes a full run and the emitted `sift_bench.json` is attached to the CONSOLIDATION tracker. Any previously circulated SIFT1M numbers that were not produced by this harness have been struck — see the Phase G entry in `docs/CONSOLIDATION.md`.

---

## Provenance requirements

Every entry in the **Current results** table is sourced from the `sift_bench.json` schema version 1 emitted by `benchmarks/src/bin/galaxdb-sift-bench.rs`. A valid record contains all of the following fields, or the row does not get published:

| Field | Source | Notes |
|---|---|---|
| `schema_version` | bench binary | `1` |
| `commit_sha` | passed in by `scripts/aws-integration-run.sh` from `git rev-parse HEAD` | full 40-char SHA |
| `timestamp_utc` | passed in by orchestrator | ISO-8601 |
| `instance.type` | passed in by orchestrator | e.g. `c6id.4xlarge` |
| `cpu.model` | `/proc/cpuinfo` on the instance | real string from the kernel |
| `cpu.cores` | `std::thread::available_parallelism()` | measured, not claimed |
| `cpu.arch` | `std::env::consts::ARCH` | e.g. `x86_64` |
| `ram_gb` | `/proc/meminfo MemTotal` | real kernel value |
| `dataset.name` | bench binary | `"SIFT1M"` |
| `dataset.size` | base-vector count read from `sift_base.fvecs` | must equal `1_000_000` for SIFT1M |
| `dataset.dim` | vector dim read from `sift_base.fvecs` | must equal `128` for SIFT1M |
| `dataset.sha256` | verified by the orchestrator against `sift.tar.gz` | never fabricated |
| `dataset.source_url` | bench binary | canonical texmex URL |
| `hnsw_config.m`, `hnsw_config.ef_construction` | CLI args, default 16 / 200 | |
| `build.build_time_ms`, `build.build_rate_vec_per_sec` | measured during `HnswGraph::insert_parallel` | |
| `search.k` | CLI arg, default 10 | |
| `search.num_queries_evaluated` | number of SIFT1M queries actually used | ≤ 10_000 |
| `search.ef_search_sweep[]` | one record per ef value with `{ef, recall_at_k, mean_latency_us, p99_latency_us}` | default sweep: `10, 50, 100, 200` |

Any published row missing a field, or carrying a placeholder hash, is a tracker violation and must be removed until a real run fills it in.

---

## Datasets

### SIFT1M

- **Source URL (canonical):** `ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz`
- **Description:** 1,000,000 base vectors, 128-dim float32, L2 distance with pre-computed ground truth. Same dataset used by ann-benchmarks and the original HNSW paper.
- **Expected SHA256:** **TODO-USER-FETCH.** No authoritative SHA256 is published by the IRISA/texmex host. First-run pinning procedure:
  1. Run `scripts/aws-integration-run.sh`. It will download the file, compute its sha256 on the instance, and fail step 5 with a message like:

     ```
     ERROR: SIFT1M SHA256 is not pinned.
       Observed hash on this download: <hex>
       If you trust this download, set:
         export GALAXDB_SIFT1M_SHA256=<hex>
       and re-run this script. Do NOT pin a hash you have not verified
       against at least one independent download.
     ```
  2. Independently re-download `sift.tar.gz` on a different network / day and compare `sha256sum`. If the two hashes match, the file is stable.
  3. Update this section with the pinned hash, set `GALAXDB_SIFT1M_SHA256` in the environment or edit the default in `scripts/aws-integration-run.sh`, and re-run.

  Until the hash is pinned, the orchestrator aborts before running the benchmark. This is intentional.
- **Contents after extraction** (under `sift/`):
  - `sift_base.fvecs` — 1,000,000 × 128 f32 (~512 MiB)
  - `sift_query.fvecs` — 10,000 × 128 f32 (~5 MiB)
  - `sift_learn.fvecs` — 100,000 × 128 f32 (~51 MiB, not used by the recall benchmark)
  - `sift_groundtruth.ivecs` — 10,000 × 100 i32 (~4 MiB) — top-100 true nearest-neighbour ids per query, from exhaustive L2 search

---

## Hardware

**AWS `c6id.4xlarge`**, instance ID `i-0b2dec9226f62db65`. All Phase G numbers are produced on exactly this instance.

| Spec | Value |
|------|-------|
| CPU | Intel Xeon Platinum 8375C @ 2.90 GHz (Ice Lake, 3rd-gen Scalable) |
| vCPU | 16 |
| RAM | 32 GiB DDR4 |
| Local storage | 1 × 950 GB NVMe instance-store (ephemeral, formatted to XFS by the harness) |
| Root volume | EBS gp3 (separate from the benchmark workspace) |
| OS | Ubuntu 24.04 LTS, kernel ≥ 6.8 (io_uring-capable) |
| AES-NI / AVX-512 | yes |
| Cost | ~$0.81/hr running, $0 stopped (stop-on-exit is enforced by `scripts/aws-integration-run.sh`) |

CPU model, core count, and RAM are re-measured by the benchmark binary on every run and written into `sift_bench.json`, so reported numbers always carry the real hardware, not a hard-coded claim.

---

## Non-vector benchmarks (Months 1–2)

These were measured on the same `c6id.4xlarge` instance with the commands shown. They are unaffected by Phase G and remain the reference numbers for the storage engine, wire protocol, and encryption path.

### Wire-protocol performance (Month 2)

| Metric | Measured | Command |
|---|---|---|
| Wire SELECT QPS (100 queries, `RwLock<Database>` + `execute_readonly()`) | 7,390 QPS | see `crates/galaxdb-wire/tests` |
| Wire INSERT (single-row, 4 clients) | 454 rows/sec | same |
| Embedded INSERT (batched 100/stmt) | 20,267 rows/sec | `cargo run --release -p galaxdb-benchmarks -- --workload oltp` |

Month 2 Gate 1 / 2 / 3 (33/33 pass) covers functional wire-protocol behaviour; see `.kiro/specs/galaxdb-v1-engine/tasks.md` tasks 40 / 41 / 42.

### Storage engine (Month 1)

**OLTP (1M rows, 16 threads, 60 s, storage API with group commit, RELAXED durability):**

```
cargo run --release -p galaxdb-benchmarks -- \
    --workload oltp --duration 60 --warmup 10 --rows 1000000 --threads 16 \
    --data-dir /mnt/nvme/bench_data
```

| Metric | Value |
|---|---|
| Write TPS | 258,555 |
| Read p50 (warm) | 3 µs |
| Read p99 | 47 µs |
| Write p50 | 16 µs |
| Write p99 (sustained) | 377 µs |

**Cold-cache reads (50M rows × 600 B ≈ 30 GB, 10 MB SST cache, page cache dropped):**

```
sudo ./target/release/galaxdb-benchmarks \
    --workload coldcache --rows 50000000 --data-dir /mnt/nvme/coldcache_50m
```

| Metric | Value |
|---|---|
| Missing keys | 0 / 100,000 (0.0 %) |
| Read p50 | 147 µs |
| Read p99 | 308 µs |
| Read p999 | 329 µs |

**OLAP (1,000 PAX blocks × 10K rows, parallel rayon scan, Zstd):**

```
cargo run --release -p galaxdb-benchmarks -- --workload olap --threads 16 --duration 60
```

| Metric | Value |
|---|---|
| Scan throughput | 4.49 GB/s |
| Zone-map skip rate | 80.0 % |

**Mixed OLTP + OLAP (60 s, HotSet/ScanBuffer isolation):**

```
cargo run --release -p galaxdb-benchmarks -- --workload mixed --threads 16 --duration 60
```

| Metric | Value |
|---|---|
| OLTP p99 during scan | 191 µs |
| p99 degradation vs OLTP-alone baseline | 0.0 % |
| HotSet evictions | 0 |

### Encryption (crypto bench, `cargo bench -p galaxdb-crypto`)

| Benchmark | Latency | Throughput |
|---|---|---|
| AEGIS-256 decrypt 1 MB | 151 µs | 6.63 GB/s |
| AEGIS-256 encrypt 64 KB | 9.75 µs | 6.56 GB/s |
| AES-256-GCM decrypt 1 MB | 701 µs | 1.43 GB/s |
| XXH3-64 checksum 1 MB | — | 34.1 GB/s |
| ART lookup (1M keys) | 168 ns/op | — |

### Crash safety (chaos suite)

```
cargo run --release -p galaxdb-chaos-tests
```

| Test | Result |
|---|---|
| C1: kill mid-flush → WAL replay | 10,000 rows recovered, zero loss |
| C2: kill mid-compaction → old blocks intact | 4,000 keys readable |
| C3: corrupt WAL record → replay stops | 538 / 1000 recovered at corruption |
| C4: disk full → clean checkpoint | reserve file freed, reads continue |
| C5: 100 concurrent writers | 100K writes, 0 dupes, 0 missing |
| C6: OLAP scan during OLTP | 0 HotSet evictions |

---

## Competitive comparison

Any competitor comparison on vector search requires the competitor to have run on the same `c6id.4xlarge` instance against the same SIFT1M dataset with the same pinned SHA256. No such comparison is currently published. When Phase G produces GalaxDB's numbers, the same commands will be re-run against `hnswlib 0.8` (see `benchmarks/tools/hnswlib_recall.py` for the harness that already exists) and the results published side by side.

Non-vector competitor comparisons (PostgreSQL 16 vs GalaxDB wire protocol, etc.) are referenced in the commit history of the Month 2 gate test and are not republished here until they are re-run on the Phase G instance image.

---

*Every number published in this file is reproducible from the command listed beside it, against a named dataset, on `c6id.4xlarge` instance `i-0b2dec9226f62db65`. Anything that cannot be traced that way does not belong here.*
