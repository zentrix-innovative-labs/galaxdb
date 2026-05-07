# Storage Engine Reference

The GalaxDB storage engine is an LSM-tree implementation optimized for mixed OLTP/OLAP workloads with AI-native features.

---

## Write Path

### Durability Modes

| Mode | Behavior | Latency | Use Case |
|------|----------|---------|----------|
| STRICT | fsync per commit | ~1ms | Financial, critical data |
| RELAXED | Group commit (10ms window) | ~16µs p50 | Default, high throughput |

### Write Flow

```
Client Write
    → WAL append (LZ4 compressed, XXH3-64 checksum)
    → Group commit batch (RELAXED) or immediate fsync (STRICT)
    → Memtable insert (16-shard skiplist, MVCC versioned)
    → ART index update (primary key → row location)
    → If value > 1KB: blob log separation (BlobRef in memtable)
```

### Back-Pressure

- Memtable seals at 64MB → enqueued for flush
- Sealed-but-unflushed limit: 256MB (semaphore blocks writers)
- WriteController: proportional slowdown between 32GB-64GB pending compaction
- Hard stop at 64GB pending compaction (all writes blocked)

---

## Read Path

### Point Lookup

```
ART lookup (key → RowLocation)
    → If Memtable: direct read from skiplist
    → If SST: Bloom filter check → block index lookup → pread one PAX block
    → If BlobRef: fetch from blob log
```

**Cold-cache performance:** 147µs p50 (50M rows, 10MB cache, NVMe)

### Scan

```
Zone-map pruning (skip blocks where min > threshold)
    → Bloom filter (skip SSTs without target key)
    → PAX block decompression (column-at-a-time)
    → Buffer pool routing: scans → ScanBuffer, lookups → HotSet
```

**Scan throughput:** 4.49 GB/s (16 threads, PAX + Zstd decompression)

---

## PAX Block Format

```
┌─────────────────────────────────────────┐
│ Header                                  │
│   magic: 0x47414C41 ("GALA")            │
│   format_version: u32                   │
│   block_id: u64                         │
│   row_count: u32                        │
│   commit_timestamp: u64                 │
│   column_count: u32                     │
│   column_descriptors[]                  │
│     col_type, codec, offset,            │
│     compressed_len, zone_map_min/max    │
├─────────────────────────────────────────┤
│ Column chunks (compressed)              │
│   Fixed-width: delta + bit-packing      │
│   Variable-width: Zstandard L3          │
│   Embedding: uncompressed float32       │
├─────────────────────────────────────────┤
│ Row offset table                        │
├─────────────────────────────────────────┤
│ Footer: XXH3-64 checksum               │
└─────────────────────────────────────────┘
```

---

## SST File Format

Multi-block SSTs with block index (following RocksDB's BlockBasedTable pattern):

```
┌─────────────────────────────────────────┐
│ PAX Block 0 (~62KB, ~100 rows)          │
├─────────────────────────────────────────┤
│ PAX Block 1                             │
├─────────────────────────────────────────┤
│ ...                                     │
├─────────────────────────────────────────┤
│ PAX Block N                             │
├─────────────────────────────────────────┤
│ Block Index                             │
│   [min_key, file_offset, block_len]     │
│   per block                             │
├─────────────────────────────────────────┤
│ Bloom Filter (Monkey-optimal FPR)       │
├─────────────────────────────────────────┤
│ SST Footer (offsets, metadata)          │
└─────────────────────────────────────────┘
```

Default SST size: 8MB. Block index loaded into memory at SST registration.

---

## WAL (Write-Ahead Log)

### Record Format

```
[type: u8][seq_no: u64][length: u32][xxh3_checksum: u64][lz4_payload: bytes]
```

### Record Types

| Type | Code | Description |
|------|------|-------------|
| ROW_PUT | 0x01 | Insert/update a row |
| ROW_DELETE | 0x02 | Delete a row |
| DELTA_INSERT | 0x03 | Vector insert to delta buffer |
| DELTA_TOMBSTONE | 0x04 | Vector delete from delta buffer |
| CHECKPOINT | 0x05 | Flush completed, safe truncation point |
| BLOB_REF | 0x06 | Large value stored in blob log |

### Recovery

On startup:
1. Find last CHECKPOINT record
2. Replay all records after checkpoint
3. Verify XXH3-64 checksum per record
4. Stop at first checksum failure (skip corrupt tail)
5. Rebuild ART index from replayed data

Recovery time: < 30 seconds (verified by chaos tests)

---

## Compaction

### Lazy Leveling Strategy

| Level | Strategy | Max Files | Size Ratio |
|-------|----------|-----------|------------|
| L0 | Tiered | 4 | — |
| L1-L3 | Tiered | — | 10× |
| L4 | Leveled | — | 10× |

### MVCC Garbage Collection

During compaction:
- Discard versions not needed by any active snapshot
- Retain versions referenced by pinned version tags
- Merge tombstones with their target rows

---

## Buffer Pool

### Dual-Pool Architecture

| Pool | Capacity | Eviction | Purpose |
|------|----------|----------|---------|
| HotSet | 70% RAM | LRU | Point lookups, frequently accessed blocks |
| ScanBuffer | 30% RAM | Clock-sweep | Sequential scans, one-time reads |

**Isolation guarantee:** ScanBuffer never evicts a HotSet-resident block. Verified by chaos test (0 evictions under concurrent OLTP + OLAP).

### NUMA Awareness

On Linux with libnuma: partitions buffer pool per NUMA node, routes requests to local partition. On macOS/Windows: single partition fallback.

---

## Encryption (TDE)

### Configuration

```rust
let config = EngineConfig {
    tde_enabled: true,
    key_provider: Box::new(EnvKeyProvider::new("GALAXDB_MASTER_KEY")),
    ..Default::default()
};
```

### Performance Impact

| Operation | Without TDE | With TDE (AEGIS-256) | Overhead |
|-----------|-------------|---------------------|----------|
| Block read (64KB) | 9.5µs | 9.75µs | < 3% |
| Block write (64KB) | 10.1µs | 10.4µs | < 3% |
| WAL append (1KB) | 0.7µs | 1.4µs | ~100% (AES-GCM) |

AEGIS-256 decrypt at 6.63 GB/s means encryption is effectively free for reads.

---

## Disk Full Handling

1. Engine pre-allocates 32MB reserve file (`_galaxdb_reserve`) at startup
2. On disk-full detection (write fails with ENOSPC):
   - Delete reserve file (frees 32MB)
   - Perform clean checkpoint (flush memtable, write CHECKPOINT to WAL)
   - Block all writes
   - Emit `disk_full` metric
3. Reads continue normally
4. When space is freed, writes resume automatically

---

## Rate Limiting

### RateLimiter (I/O bandwidth)

Auto-tuned token bucket calibrated to 70% of NVMe write bandwidth at startup.

Adaptive: if HP-queue P99 exceeds 1.5× baseline for 3 consecutive 100ms windows, ceiling drops by 30%. Restores when latency normalizes.

### WriteController (compaction debt)

| Pending Compaction | Action |
|-------------------|--------|
| < 32GB (soft limit) | Full throughput |
| 32GB - 64GB | Proportional slowdown |
| ≥ 64GB (hard limit) | All writes blocked |
| Returns below 32GB | Full throughput restored |

---

## Performance Summary

Measured on AWS c6id.4xlarge (16 vCPU Ice Lake, NVMe):

| Metric | Value |
|--------|-------|
| Write TPS (RELAXED) | 268,505 |
| Read p50 (warm) | 3µs |
| Read p50 (cold, 50M rows) | 147µs |
| Column scan | 4.49 GB/s |
| Zone-map skip rate | 80% |
| OLTP p99 during OLAP scan | 109µs |
| Crash recovery | < 30s |
| Encryption overhead | < 3% |
