# GalaxDB Consolidation Sprint — Remove All Stubs, Mocks, and Fakes

> **Status doc for the stub-removal consolidation sprint. This file is updated after every phase completes. It is the source of truth for what has been fixed, what remains, and what the audit tripwires look like.**

## Background

During the Month 1–3 build-out, several task checkboxes in `tasks.md` were ticked while the underlying implementation was a stub (returning success without writing to storage) or a mock (returning synthetic data). An audit on 2026-05-10 identified these and the user mandated a full consolidation sprint before any new feature work resumes.

The governing rules live in `.kiro/steering/engineering-principles.md` and are always active.

## Full stub/mock inventory (as of 2026-05-10 audit)

### Production paths shipping mocks or stubs

| Location | Issue | Phase |
|---|---|---|
| `crates/galaxdb-sidecar/src/main.rs` | `--mock-dim` CLI flag, `mock_embed()` FNV-hash fake embedding, `"falling back to mock mode"` branch on model-load failure, `"mock-v1.0"` model-version tag | A |
| `crates/galaxdb-sidecar/src/manager.rs` | `SidecarConfig::mock_dim` field plumbs mock flag from engine to sidecar | A |
| `crates/galaxdb-embedded/src/lib.rs` | `Database::open_with_sidecar(..., mock_dim: Option<usize>)` exposes the mock flag in public API | A |
| `crates/galaxdb-sidecar/tests/integration.rs` | `start_sidecar(...)` launches `--mock-dim`; no real-model integration test | A |
| `crates/galaxdb-sql/src/executor.rs::execute_insert` | Validates inputs, returns `RowCount(1)` without writing to storage | B |
| `crates/galaxdb-sql/src/executor.rs::execute_update` | Returns `RowCount(0)` without writing new MVCC version | B |
| `crates/galaxdb-sql/src/executor.rs::execute_delete` | Returns `RowCount(0)` without writing tombstone | B |
| `crates/galaxdb-sql/src/executor.rs::execute_select` | Returns empty `Rows { }` — never reads | B |
| `crates/galaxdb-sql/src/executor.rs` `Analyze`/`Backup`/`Restore`/`BulkInsert`/`CreateVersionTag` arms | Return formatted strings without doing the work | B |
| `crates/galaxdb-sql/src/executor.rs::NoOpVectorBackend` | Publicly re-exported; used silently by `galaxdb-wire/src/server.rs` so wire-protocol SEMANTIC_MATCH returns empty | B |
| `crates/galaxdb-embedded/src/lib.rs::exec_standard_sync::Delete(_)` | `Ok(RowCount(0))` no-op | B |
| `crates/galaxdb-embedded/src/lib.rs::exec_update` | Returns `RowCount(0)` without touching storage | B |
| `crates/galaxdb-embedded/src/lib.rs::exec_extension` | `_ => Ok(QueryResult::Ok("OK".to_string()))` silently accepts unknown extensions; `BackupTo`, `RestoreFrom`, `BulkInsert` return success strings | B |
| `crates/galaxdb-wire/src/server.rs:277` | `Value::Integer(_) => None, // simplified` silently drops integer bind parameters | D |
| `crates/galaxdb-crypto/src/key_provider.rs::AwsKmsKeyProvider` | Stub returning `Err("...not yet implemented — this is a stub")`; AWS lock-in | C |
| `crates/galaxdb-storage/src/disk_full/mod.rs:117` | `"For now we rely on the tracing log line"` — `_disk_full` Prometheus metric not emitted | E |
| `crates/galaxdb-sql/src/planner.rs:180` | `plan_select` comment: "simplified — full SQL planning is complex" — no JOIN, no proper WHERE | B (folded into executor rewrite) |

### Tasks prematurely marked complete

Unticked as part of this consolidation — each has a real code prerequisite below.

| Task | Real state | Phase that closes it |
|---|---|---|
| 18.3 INSERT executor | Stub in `galaxdb-sql`, partial in `galaxdb-embedded` | B |
| 18.4 SELECT executor | Stub in `galaxdb-sql`, prefix string scan in `galaxdb-embedded` (no ART, zone-map, Bloom) | B |
| 18.5 UPDATE executor | No MVCC write in either path | B |
| 18.6 DELETE executor | No tombstone write in either path | B |
| 18.7 BULK INSERT executor | Returns success string | B |
| 32.3 `AT VERSION timestamp` | Helper exists in `MerkleDag`; executor never calls it | B |
| 32.4 `AT VERSION tag_name` | Same — guardrail only | B |
| 32.6 SEMANTIC_FRESH | Warning returned; never routed through executor | B |
| 33 Version tags (whole task) | `CREATE VERSION TAG` lives only in `galaxdb-embedded`; SQL executor returns a success string | B |
| 33.5 Compactor pinning | Compaction never consults `TagCatalog` | B |

### Acceptable `mock*` occurrences (stay)

These are test-only and correctly scoped. They stay because they are not in production code paths.

- `crates/galaxdb-sql/src/executor_tests.rs::MockVectorBackend` / `DownVectorBackend` — test file, `#[cfg(test)]` module.
- `crates/galaxdb-versioning/tests/lance_export_integration.rs::VecSource` — integration-test trait impl, not a mock.
- `rate_limiter/tests.rs::adjust_from_latency_noop_when_already_normal` — test function name.
- `disk_full/tests.rs::recover_is_noop_when_not_in_disk_full_mode` — test function name.
- `sidecar/src/tracking.rs::same_version_change_is_noop` — test function name.

## Phases

### Phase A — Remove sidecar mocks. Real model or hard fail.

- [x] A1: Delete `--mock-dim` CLI flag from `crates/galaxdb-sidecar/src/main.rs`
- [x] A2: Delete `mock_embed()` function
- [x] A3: Delete `"falling back to mock mode"` branch on model-load failure — replace with `exit(1)` and typed error log
- [x] A4: Delete `"mock-v1.0"` model version default — real model version always known
- [x] A5: Delete `SidecarConfig::mock_dim` field in `manager.rs`
- [x] A6: Delete `mock_dim` parameter from `Database::open_with_sidecar`
- [x] A7: Update `tests/integration.rs` to launch the sidecar with a real small model (`sentence-transformers/all-MiniLM-L6-v2`, ~90 MB) via `hf-hub`. Gated behind `cfg(feature = "online-tests")`.
- [x] A8: Update `semantic_match_end_to_end` test in `galaxdb-embedded` to use real model; placed behind `online-tests` feature.
- [x] A9: `cargo check --workspace` clean; `cargo test -p galaxdb-sidecar` clean

**Verification (all commands green on 2026-05-10)**:
- `cargo check --workspace --all-targets` — Exit 0
- `cargo test --workspace --exclude galaxdb-python --lib` — 648 tests pass across 11 crates
- `cargo test -p galaxdb-sidecar --test integration --features online-tests --release` — 2 online tests pass against real `sentence-transformers/all-MiniLM-L6-v2` (384-d, L2-normalized, deterministic, semantic sanity asserted)
- `cargo test -p galaxdb-sidecar --lib --features online-tests --release manager` — 5 manager lifecycle tests pass against real model
- `bash scripts/phase_a_smoke.sh` — real model loads in 3 s from HF cache, dim=384 confirmed
- `bash scripts/phase_a_hardfail.sh` — bogus model id → exit 1, no socket, no mock fallback, typed error
- `git grep -n -i 'mock' -- 'crates/galaxdb-sidecar/src/**/*.rs' 'crates/galaxdb-embedded/src/**/*.rs' 'crates/galaxdb-sidecar/tests/**/*.rs' | grep -v -E '^\s*//|test|Test|documentation|docs|tests'` — returns zero non-documentation matches

**Scripts added**: `scripts/phase_a_smoke.sh`, `scripts/phase_a_hardfail.sh`

### Phase B — Real SQL executor wired to storage. Delete `galaxdb-sql` stubs.

- [ ] B1: Move real execution code out of `galaxdb-embedded` into `galaxdb-sql::executor`. The executor owns `Arc<galaxdb_storage::Engine>`.
- [ ] B2: Add `galaxdb-storage.workspace = true` to `crates/galaxdb-sql/Cargo.toml`.
- [ ] B3: Implement `execute_insert` with real memtable write, WAL, ART update, Bloom update, MinHash compute, sidecar async embed trigger, delta-buffer insert.
- [ ] B4: Implement `execute_update` with real MVCC version write via `Engine::put_sync` at new timestamp.
- [ ] B5: Implement `execute_delete` with tombstone + `DELTA_TOMBSTONE` + ART removal.
- [ ] B6: Implement `execute_select` with ART lookup for point reads, `Engine::scan_all` for full scans with zone-map pruning + Bloom filter skip. Use the catalog for column projection.
- [ ] B7: Implement `execute_point_lookup` (same code path as point-read in select).
- [ ] B8: Implement `execute_analyze` wired to `galaxdb_storage::statistics` real ANALYZE task.
- [ ] B9: `execute_backup` and `execute_restore` return `GalaxError::NotYetAvailable { task_id: "37" }` — typed error, never a fake success. Task 37 will replace with real impl.
- [ ] B10: Implement `execute_bulk_insert` with real PAX block writer bypassing memtable (task 18.7 scope).
- [ ] B11: Implement `execute_create_version_tag` wired to `TagCatalog::create_tag` with `MerkleDag::latest_root` and pinned-block set. Move out of `galaxdb-embedded`.
- [ ] B12: Add `at_version: Option<VersionRef>` to `QueryPlan::FullScan` and `QueryPlan::SemanticSearch`. Executor filters blocks by pinned set before reading.
- [ ] B13: Enforce SEMANTIC_FRESH rule — error if `AT VERSION` + `SEMANTIC_MATCH` without explicit consistency mode.
- [ ] B14: Pinned-block compactor integration — `galaxdb-storage::compaction` accepts `Arc<dyn PinSet>`; `galaxdb-embedded` passes a `TagCatalogPinSet` adapter. Compactor calls `is_pinned(block_id)` before GCing versions.
- [ ] B15: Delete `NoOpVectorBackend` from `galaxdb-sql`.
- [ ] B16: `galaxdb-wire::server` — pass a real `VectorSearchBackend` from `main.rs`, or return `GalaxError::NoVectorBackendConfigured` explicitly. No silent empty.
- [ ] B17: `galaxdb-embedded::Database` becomes a thin wrapper over `Engine + Catalog + SidecarManager + galaxdb_sql::executor::execute`.
- [ ] B18: Delete every `// In the full implementation, this would ...` comment. Those paths are now real.
- [ ] B19: Full `cargo test --workspace` clean.

**Verification**: 
- `! git grep -n 'In the full implementation' -- 'crates/**/*.rs'` returns zero.
- `cargo test -p galaxdb-sql` passes with tests that insert → read → assert stored bytes round-trip.
- `cargo test -p galaxdb-embedded` passes with CRUD round-trip tests.

### Phase C — Pluggable key management. No AWS lock-in.

- [ ] C1: Delete `AwsKmsKeyProvider` stub.
- [ ] C2: Remove `aws-kms` Cargo feature from `crates/galaxdb-crypto/Cargo.toml`.
- [ ] C3: Add `ExternalCommandKeyProvider` — generic KMS via shell command. Engine calls `cmd generate` to create a DEK; `cmd decrypt` with ciphertext on stdin, plaintext on stdout. Works with AWS CLI, gcloud, az, vault CLI, or any custom provider.
- [ ] C4: Add `HashicorpVaultKeyProvider` using `vaultrs` crate (pure Rust). Auth via `VAULT_TOKEN` env var or Vault Agent sidecar. Supports Transit engine for encrypt/decrypt.
- [ ] C5: Keep `LocalKeyProvider` and `EnvKeyProvider` (already real).
- [ ] C6: Add `KeyProviderSpec` enum for startup selection. Expose via `GALAXDB_KEY_PROVIDER` env var with syntax: `local:/path` | `env:VARNAME` | `command:<shell>` | `vault:<secret-path>`.
- [ ] C7: Round-trip tests for every provider. Vault test uses `testcontainers` with Vault dev-mode; skipped if Docker unavailable. External-command test uses a small Python or Rust helper that does deterministic AES.
- [ ] C8: Update `docs/STORAGE_ENGINE.md` to document the provider matrix and syntax.

**Verification**: `! git grep -n 'AwsKmsKeyProvider' -- '**/*.rs' '**/*.toml'` returns zero. `cargo test -p galaxdb-crypto` covers all four real providers.

### Phase D — Wire-protocol bind parameter plumbing.

- [ ] D1: Replace `Value::Integer(_) => None, // simplified` in `crates/galaxdb-wire/src/server.rs` with full typed conversion for `Integer`, `Float`, `Boolean`, `Null`.
- [ ] D2: Write a wire test: `INSERT INTO t VALUES ($1)` with an integer parameter. Read back via `SELECT * FROM t` over the wire. Value must match.
- [ ] D3: PostgreSQL binary format vs text format: support both. Text format is enough for v1 but the test confirms we're routing through the executor correctly.

**Verification**: `! git grep -n '// simplified' -- 'crates/galaxdb-wire/**/*.rs'` returns zero.

### Phase E — `_disk_full` metric live.

- [ ] E1: Add a `prometheus::IntGauge` (0 or 1) to `galaxdb-storage::disk_full::DiskFullHandler`. Set to 1 on trip, 0 on recovery.
- [ ] E2: Gauge is registered with the default Prometheus registry exported by `galaxdb-observe` when that crate is present; fallback to a crate-local static registry otherwise.
- [ ] E3: Test: flip disk-full on, assert gauge reads 1; recover, assert gauge reads 0.

**Verification**: `! git grep -n "For now we rely" -- 'crates/**/*.rs'` returns zero. 

### Phase F — Reconcile `tasks.md`. Untick the fakes.

- [ ] F1: Untick 18.3, 18.4, 18.5, 18.6, 18.7 in `tasks.md`.
- [ ] F2: Untick 32.3, 32.4, 32.6.
- [ ] F3: Untick 33 and 33.5. (33.1, 33.2, 33.3, 33.4, 33.6 individually valid — verify before leaving ticked.)
- [ ] F4: Add a note at the top of `tasks.md`: "Tasks here MUST have real code verified by real tests on real infrastructure. See `.kiro/steering/engineering-principles.md`."
- [ ] F5: Add a "Consolidation Sprint" section tracking Phases A–H with the same checkbox discipline.

**Verification**: Nothing to verify automatically — human sign-off by user.

### Phase G — Real AWS benchmarking.

- [ ] G1: Write `scripts/aws-integration-run.sh` — start instance `i-0b2dec9226f62db65`, wait for SSH, rsync workspace, mount NVMe, `cargo build --release`, `cargo test --release --features online-tests --workspace`, collect logs + benchmark JSON, stop instance.
- [ ] G2: Run the script after Phases A–E complete. Do not re-tick tasks 18.3–18.7 / 32.3–32.6 / 33 until that run is green.
- [ ] G3: Download SIFT1M (1M × 128-d float32) fresh on the instance. Record dataset SHA256 in `docs/BENCHMARKS.md`.
- [ ] G4: Run HNSW build + recall@10 on SIFT1M in release mode on the AWS instance. Publish full provenance (commit SHA, instance type, CPU model, RAM, dataset hash, exact commands, date) alongside numbers.
- [ ] G5: Never publish random-vector HNSW numbers.
- [ ] G6: Stop the AWS instance at the end of every run. `aws ec2 describe-instances --instance-ids i-0b2dec9226f62db65` must report `stopped` before the script exits.

**Verification**: `docs/BENCHMARKS.md` updated, user reviews.

### Phase H — CI gates to prevent regression.

- [ ] H1: Add a CI step that runs `scripts/grep-for-mocks.sh` which fails the build if `mock` appears in any non-test source file.
- [ ] H2: Add `cargo deny check` to CI with `deny.toml` that blocks any new `aws-sdk-*`, `google-cloud-*`, `azure_*` dependency at the Cargo.lock level.
- [ ] H3: Add a CI step that fails if `tasks.md` has ticked boxes on tasks that match known stub patterns (the grep above + executor stub comments).
- [ ] H4: Document the CI gates in `.github/workflows/README.md`.

**Verification**: A PR that adds `fn mock_foo()` in a non-test file fails CI.

## Running log

### 2026-05-10 — Consolidation plan agreed

Audit complete. Plan signed off by user. Steering rules live in `.kiro/steering/engineering-principles.md`. This tracker created. Beginning Phase A.

### 2026-05-10 — Phase A complete

Sidecar mocks removed. Every embedding is now computed by `sentence-transformers/all-MiniLM-L6-v2` (384-d, L2-normalized) via Candle. Bogus model ids or network failures cause the sidecar to exit 1 with a typed error; the parent `SidecarManager` observes the dead child and enters degraded mode. No mock fallback anywhere in production code.

Files touched in Phase A:
- `crates/galaxdb-sidecar/src/main.rs` — fully rewritten. `--mock-dim` deleted, `mock_embed` deleted, fallback branch deleted, real-model-or-exit-1 path only. `EmbeddingModel::embed` now returns `Result<Vec<f32>, Box<dyn Error>>` so tokenize / forward / pool errors propagate as `SidecarMessage::Error` rather than panicking.
- `crates/galaxdb-sidecar/src/manager.rs` — `SidecarConfig::mock_dim` / `model_path` removed. Replaced with single `model_id: String`. `DEFAULT_MODEL_ID` constant added. Tests moved behind `#[cfg(all(test, feature = "online-tests"))]` and assert 384-d L2-normalized output.
- `crates/galaxdb-sidecar/Cargo.toml` — added `[features] online-tests`. Description updated from `(ort crate, ...)` to `(Candle, ...)`.
- `crates/galaxdb-sidecar/tests/integration.rs` — fully rewritten. Launches sidecar with real model id. Asserts 384-d output, L2 norm within 0.01 of 1.0, deterministic across runs, semantic cosine sanity (near-duplicates > unrelated). Gated behind `online-tests`.
- `crates/galaxdb-embedded/src/lib.rs` — `open_with_sidecar(path, binary, model_id: &str)`. `TableVectorIndex::dim` annotated `#[allow(dead_code)]` because only the `online-tests` test reads it. `semantic_match_end_to_end` rewritten to use real model, gated behind `online-tests`.
- `crates/galaxdb-embedded/Cargo.toml` — added `[features] online-tests`.
- `scripts/phase_a_smoke.sh` — direct binary smoke test: real model loads, socket appears.
- `scripts/phase_a_hardfail.sh` — hard-fail semantics: bogus model → exit 1, no socket, no mock.

**Next action**: Phase B — real SQL executor wired to storage. Delete `galaxdb-sql` executor stubs.
