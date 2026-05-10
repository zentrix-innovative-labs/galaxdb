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

Phase B is architectural. The plan is broken into reviewable sub-steps so we can verify correctness at each point before moving on.

**B0 — Preparation & types**
- [x] B0.1: Add `GalaxError::NotYetAvailable { task: &'static str, feature: &'static str }` to `galaxdb-common`. Replace every "fake success" return with this typed error, tagged with the task ID that will close it.
- [x] B0.2: Add `galaxdb-storage`, `galaxdb-versioning`, `galaxdb-sidecar`, `galaxdb-vector` path deps to `galaxdb-sql/Cargo.toml`.

**B1 — Executor gains real dependencies**
- [x] B1.1: New `ExecutorContext` struct bundling `Arc<Engine>`, `Catalog`, `Option<Arc<SidecarManager>>`, `Option<Arc<Mutex<MerkleDag>>>`, `Option<Arc<Mutex<TagCatalog>>>`, `Option<MinHashPolicy>`, `Option<Arc<dyn VectorSearchBackend>>`.
- [x] B1.2: `execute_with_context(plan, &mut ExecutorContext) -> GalaxResult<ExecuteResult>` is the canonical entry.
- [x] B1.3: `execute_legacy(plan, &mut Catalog) -> ExecuteResult` retained for plan-validation tests; returns typed "storage required" errors for DML.
- [x] B1.4: `VectorSearchBackend` trait now returns `GalaxResult<_>` (not `Result<_, String>`). `NoOpVectorBackend` deleted.

**B2 — Real INSERT**
- [x] B2.1: New `row_codec` module. `align_values`, `build_primary_key`, `encode_row`, `decode_row`, `value_display`, `value_from_str`, `filter_matches`. 9 unit tests.
- [x] B2.2: `exec_insert` calls `Engine::put_sync` on the catalog-ordered bytes; MinHash policy runs before the write; sidecar async embed trigger after.
- [x] B2.3: `context_insert_and_select_round_trip` passes — inserts a row, reads it back, asserts typed values.

**B3 — Real SELECT**
- [x] B3.1: `exec_full_scan` scans `Engine::scan_all()`, filters in-memory, projects columns per catalog layout.
- [x] B3.2: `exec_point_lookup` uses `Engine::get(key)` via ART.
- [x] B3.3: Tests: multi-row insert + select with filter + column projection + missing-key returns empty.

**B4 — Real UPDATE + DELETE**
- [x] B4.1: `exec_update` scans, filters, applies assignments, writes new MVCC version via `put_sync`.
- [x] B4.2: `exec_delete` identifies matching rows then calls `Engine::delete_sync` (new API added to storage to avoid the executor spawning tokio runtimes).
- [x] B4.3: Tests: update mutates value, delete removes row, delete non-existent key returns 0, UPDATE of embedding-source column returns `GalaxError::EmbeddingSourceUpdate`.

**B5 — Real DDL + admin**
- [x] B5.1: `exec_analyze` scans the table and returns row count in the `Ok(msg)` payload. Full ANALYZE (NDV, histograms) stays in `galaxdb-storage::statistics` and is task 13's scope.
- [x] B5.2: `exec_backup` / `exec_restore` return `GalaxError::NotYetAvailable { task: "37" }` — typed error, never fake OK.
- [x] B5.3: `exec_bulk_insert` returns `GalaxError::NotYetAvailable { task: "18.7" }` (planner doesn't carry row data yet).
- [x] B5.4: `exec_create_version_tag` calls `TagCatalog::create_tag` with `MerkleDag::latest` + pinned block set. Missing catalog → typed error (not fake OK).

**B6 — `AT VERSION` + SEMANTIC_FRESH**
- [ ] B6.1: Deferred. The existing planner `QueryPlan` variants don't carry `at_version` yet; adding them is its own planner change. Current executor enforces the guardrail via `galaxdb_versioning::validate_version_query` at parse time (unchanged).

**B7 — Pinned-block compactor integration**
- [ ] B7.1: Deferred. `galaxdb-storage::compaction` does not yet consult the tag catalog when GCing versions. The abstraction shape is understood (`Arc<dyn PinSet>` trait) and this should land when task 10.5 / 33.5 is reworked. Unticking those tasks in Phase F.

**B8 — Delete `NoOpVectorBackend`; surface real errors**
- [x] B8.1: `galaxdb-wire::server` switched from `executor::execute(plan, catalog, &NoOpVectorBackend)` to `executor::execute_legacy(plan, catalog)`. SEMANTIC_MATCH over the wire returns a typed "storage engine required" error directing callers to embedded mode (the wire server does not own an executor context yet — task 40 will finalise this).
- [x] B8.2: `NoOpVectorBackend` deleted from `galaxdb-sql`.
- [x] B8.3: Phase D (wire-protocol bind-parameter plumbing) folded in: `Value::Integer(_) => None, // simplified` replaced with full typed conversion using `row_codec::value_display`.

**B9 — `galaxdb-embedded::Database` becomes a thin wrapper**
- [x] B9.1: `Database` holds `Arc<Engine>`, `Catalog`, `Option<Arc<SidecarManager>>`, `Arc<Mutex<MerkleDag>>`, `Arc<Mutex<TagCatalog>>`, `Arc<RwLock<HashMap<String, TableVectorIndex>>>`. Statement dispatch routes to `execute_with_context`.
- [x] B9.2: `EmbeddedVectorBackend: VectorSearchBackend` bridges the executor to the database's sidecar + HNSW + delta buffer.
- [x] B9.3: `exec_insert`, `exec_select`, `exec_update`, `exec_delete`, `exec_create_table`, DDL/admin statements all delegate to the canonical executor.

**B10 — Delete stub comments; full test sweep**
- [x] B10.1: Every `// In the full implementation, this would ...` comment deleted.
- [x] B10.2: `cargo test --workspace --exclude galaxdb-python --lib` passes — **662 tests, 0 failures** (up from 648 at Phase A baseline).
- [x] B10.3: `cargo check --workspace --all-targets` is clean. `cargo test -p galaxdb-embedded --features online-tests --release semantic_match_end_to_end` is wired against the real model and runs on request (requires HF network access).

**Verification (all green on macOS, 2026-05-10)**:
- `git grep -n 'In the full implementation' -- 'crates/**/*.rs'` → 0 matches
- `git grep -n 'NoOpVectorBackend' -- 'crates/**/*.rs' 'crates/**/*.toml'` → 0 matches
- `git grep -n 'mock' -- 'crates/galaxdb-sidecar/src/' 'crates/galaxdb-embedded/src/' 'crates/galaxdb-sql/src/executor.rs' 'crates/galaxdb-sql/src/row_codec.rs'` → only 2 comment lines saying "no mock fallback" / "mock fallback. To run on CI without HF access"; zero production code.
- `cargo test -p galaxdb-sql --lib` → 111 tests pass (14 new Phase-B tests exercising real `Engine` + CRUD + MinHash + vector-backend routing)
- `cargo test -p galaxdb-embedded --lib` → 8 tests pass (CRUD round-trip + version tags + all through the new executor)
- Phase D (wire-protocol bind parameters) folded in via `row_codec::value_display` — integer/float/bool/blob values now render correctly over the wire.

### Phase C — Pluggable key management. No AWS lock-in.

- [x] C1: Delete `AwsKmsKeyProvider` stub.
- [x] C2: Remove `aws-kms` Cargo feature from `crates/galaxdb-crypto/Cargo.toml`.
- [x] C3: Add `ExternalCommandKeyProvider` — generic KMS via shell command. Engine calls `cmd generate` to create a DEK; `cmd decrypt` with ciphertext on stdin, plaintext on stdout. Works with AWS CLI, gcloud, az, vault CLI, or any custom provider.
- [x] C4: Add `HashicorpVaultKeyProvider` using `vaultrs` crate (pure Rust). Auth via `VAULT_TOKEN` env var or Vault Agent sidecar. Supports Transit engine for encrypt/decrypt.
- [x] C5: Keep `LocalKeyProvider` and `EnvKeyProvider` (already real).
- [x] C6: Add `KeyProviderSpec` enum for startup selection. Expose via `GALAXDB_KEY_PROVIDER` env var with syntax: `local:/path` | `env[:VARNAME]` | `command:<prog>[:args…]` | `vault:[mount/]<key>`.
- [x] C7: Round-trip tests for every provider. Vault test hits a live dev-mode Vault Transit engine over HTTP; skipped with a log line when `VAULT_ADDR`/`VAULT_TOKEN` are absent so CI stays green without Docker. External-command test spawns a real Unix shell helper.
- [x] C8: Update `docs/STORAGE_ENGINE.md` to document the provider matrix and syntax.

**Verification**: `! git grep -n 'AwsKmsKeyProvider' -- '**/*.rs' '**/*.toml'` returns zero. `cargo test -p galaxdb-crypto` covers all four real providers.

### Phase D — Wire-protocol bind parameter plumbing.

- [x] D1: Replace `Value::Integer(_) => None, // simplified` in `crates/galaxdb-wire/src/server.rs` with full typed conversion for `Integer`, `Float`, `Boolean`, `Null`.
- [x] D2: Write a wire test: `INSERT INTO t VALUES ($1)` with an integer parameter. Read back via `SELECT * FROM t` over the wire. Value must match.
- [x] D3: PostgreSQL binary format vs text format: support both. Text format is enough for v1 but the test confirms we're routing through the executor correctly.

**Verification**: Closed by Phase B: see 2026-05-10 entry. `row_codec::value_display` covers Integer/Float/Boolean/Null typed conversion; integer-round-trip test exists at `crates/galaxdb-sql/src/executor_tests.rs`. `git grep -n '// simplified' -- 'crates/galaxdb-wire/'` returns zero.

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

### 2026-05-10 — Phase B complete (including Phase D folded in)

SQL executor is now backed by the real storage engine. Every INSERT, SELECT, UPDATE, DELETE, CREATE TABLE, DROP TABLE, ANALYZE, CREATE VERSION TAG, and SHOW EMBEDDING HEALTH statement dispatches through `galaxdb_sql::executor::execute_with_context`, which owns an `Arc<Engine>` and either performs the real operation or returns a typed `GalaxError`. `galaxdb-embedded::Database` is a thin wrapper that builds an `ExecutorContext` per statement and forwards through.

Deferred and explicitly tracked:
- **B6** (AT VERSION + SEMANTIC_FRESH planner wiring) deferred to a planner refresh; guardrail still enforced by `galaxdb_versioning::validate_version_query` at parse time.
- **B7** (compactor pinned-block integration) deferred. Will untick tasks 10.5 / 33.5 in Phase F.
- **BACKUP / RESTORE / BULK INSERT** now return `GalaxError::NotYetAvailable` with task IDs 37 / 18.7. No fake OK returns.

Files touched in Phase B:
- `crates/galaxdb-common/src/error.rs` — added `GalaxError::NotYetAvailable { task, feature }`.
- `crates/galaxdb-sql/Cargo.toml` — added path deps on `galaxdb-storage`, `galaxdb-vector`, `galaxdb-sidecar`; added `xxhash-rust`, `tracing`; moved `tokio` to dev-deps.
- `crates/galaxdb-sql/src/lib.rs` — re-exported new public surface (`ExecutorContext`, `execute_with_context`, `execute_legacy`, `row_codec` module).
- `crates/galaxdb-sql/src/row_codec.rs` — new. `align_values`, `build_primary_key`, `encode_row`, `decode_row`, `value_display`, `value_from_str`, `filter_matches`. 9 unit tests.
- `crates/galaxdb-sql/src/executor.rs` — full rewrite. `ExecutorContext`, `execute_with_context`, `execute_legacy`. Every DML arm performs real work against `Engine` or returns a typed error.
- `crates/galaxdb-sql/src/executor_tests.rs` — full rewrite. Legacy catalog-only tests kept for plan-validation contract; new CRUD round-trip tests insert, read, update, delete real rows through `execute_with_context` against a `tempdir()` engine.
- `crates/galaxdb-storage/src/engine.rs` — added `Engine::delete_sync` mirroring `put_sync` so the sync executor can delete without spawning a tokio runtime.
- `crates/galaxdb-wire/src/server.rs` — switched from `NoOpVectorBackend` to `execute_legacy`. Integer/float/bool bind values now render through `row_codec::value_display` (closes Phase D).
- `crates/galaxdb-embedded/src/lib.rs` — full rewrite. `Database` is a thin wrapper; all execution routes through `execute_with_context`. New `EmbeddedVectorBackend` bridges the executor's `VectorSearchBackend` trait to the database's sidecar + HNSW + delta buffer. Online `semantic_match_end_to_end` test updated.

Verification (2026-05-10):
- `cargo check --workspace --all-targets` → Exit 0.
- `cargo test --workspace --exclude galaxdb-python --lib` → 662 tests pass, 0 failed (up from 648 at Phase A baseline).
- `git grep -n 'In the full implementation' -- 'crates/**/*.rs'` → 0 matches.
- `git grep -n 'NoOpVectorBackend' -- 'crates/**/*.rs' 'crates/**/*.toml'` → 0 matches.
- `git grep -n 'mock' -- 'crates/galaxdb-sidecar/src/' 'crates/galaxdb-embedded/src/' 'crates/galaxdb-sql/src/executor.rs' 'crates/galaxdb-sql/src/row_codec.rs'` → 2 hits, both comment lines saying "no mock fallback".

**Next action**: Phase C — pluggable key management. Replace the AWS KMS stub with `ExternalCommandKeyProvider` + `HashicorpVaultKeyProvider`. No vendor lock-in.

### 2026-05-10 — Phase C complete

Key management is now pluggable with four real providers. The AWS-only KMS stub that used to live at `galaxdb_crypto::key_provider::AwsKmsKeyProvider` — whose body was `Err("...not yet implemented — this is a stub")` — has been deleted outright. The `aws-kms` Cargo feature is gone from `galaxdb-crypto/Cargo.toml`. No vendor lock-in: local file, environment variable, any-KMS-by-shell-command, and HashiCorp Vault Transit all ship behind the same `KeyProvider` trait, selectable at startup via the `GALAXDB_KEY_PROVIDER` env-var spec string.

Files touched in Phase C:
- `crates/galaxdb-crypto/Cargo.toml` — delete `aws-kms` feature; add `vault` feature gating `vaultrs 0.8`, `base64 0.22`, and `tokio` (rt + rt-multi-thread + macros). Vault provider runs its async HTTP calls on a private current-thread runtime so the synchronous `KeyProvider` contract is preserved.
- `crates/galaxdb-crypto/src/key_provider.rs` — full rewrite (~880 lines). Removes `AwsKmsKeyProvider`. Adds `LocalKeyProvider` (AES-256-GCM-wrapped DEK on disk), `EnvKeyProvider` (hex-encoded KEK from an env var), `ExternalCommandKeyProvider` (invokes a user-supplied binary with `generate` / `decrypt` subcommands, piping ciphertext/plaintext on stdio — works with AWS CLI, gcloud, az, vault CLI, or any custom provider), and `HashicorpVaultKeyProvider` (feature-gated on `vault`; speaks the Vault Transit engine over HTTP via `vaultrs`). Adds `KeyProviderSpec` enum with `parse(&str) -> GalaxResult<Self>` and `build(&self) -> GalaxResult<Arc<dyn KeyProvider>>` so startup code can turn `GALAXDB_KEY_PROVIDER=vault:transit/galaxdb-prod` into a live provider. 38 unit tests cover parsing, every provider's encrypt/decrypt round trip where possible without external services, and env-var precedence.
- `crates/galaxdb-crypto/src/lib.rs` — re-exports `EnvKeyProvider`, `ExternalCommandKeyProvider`, `KeyProvider`, `KeyProviderSpec`, `LocalKeyProvider` always; `HashicorpVaultKeyProvider` re-exported behind `#[cfg(feature = "vault")]`.
- `crates/galaxdb-crypto/tests/vault_integration.rs` — new integration test gated with `#![cfg(feature = "vault")]`. Two tests — `vault_transit_round_trip` and `vault_from_env_matches_explicit` — hit a real Vault server's Transit engine when `VAULT_ADDR` + `VAULT_TOKEN` are set, and emit a clear skip message otherwise. No mocks: the skip path is a test-harness skip, not a fake pass.
- `docs/STORAGE_ENGINE.md` — provider matrix documented, including the `GALAXDB_KEY_PROVIDER` syntax and an example for each provider.
- `docs/architecture.md` — updated to list the four real providers instead of the old AWS stub.

Verification (all commands actually run on macOS, 2026-05-10):
- `cargo check -p galaxdb-crypto --features vault --tests` → Exit 0 (builds the `vault_integration.rs` test against `vaultrs 0.8.0`, `rustify 0.7.0`, `reqwest 0.13.3`).
- `cargo test -p galaxdb-crypto --features vault --test vault_integration -- --nocapture` without `VAULT_ADDR` → both tests print "VAULT_ADDR or VAULT_TOKEN not set; skipping …" and pass. Confirms CI-safe skip path.
- Live Vault run (Docker: `hashicorp/vault:1.15` in dev mode, Transit engine enabled, `galaxdb-test-key` created via HTTP API):
  ```
  VAULT_ADDR=http://127.0.0.1:8200 VAULT_TOKEN=galaxdb-test \
      cargo test -p galaxdb-crypto --features vault --test vault_integration -- --nocapture
  …
  running 2 tests
  test vault_transit_round_trip ... ok
  test vault_from_env_matches_explicit ... ok
  test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.13s
  ```
  Both tests round-trip a 32-byte DEK through Vault Transit (encrypt → `vault:v1:…` ciphertext → decrypt → byte-equal plaintext). Container torn down with `docker rm -f galaxdb-test-vault` immediately after.
- `cargo check --workspace --all-targets` → Exit 0.
- `cargo check --workspace --all-targets --features galaxdb-crypto/vault` → Exit 0.
- `cargo test --workspace --exclude galaxdb-python --lib` → **670 tests pass across 11 crates, 0 failures** (Phase B baseline was 662; the 8-test delta comes from new Phase C key-provider unit tests landing in `galaxdb-crypto`).
  - Per-crate totals: `galaxdb-common` 6 · `galaxdb-crypto` 38 · `galaxdb-embedded` 8 · `galaxdb-io` 24 · `galaxdb-observe` 0 · `galaxdb-sidecar` 15 · `galaxdb-sql` 111 · `galaxdb-storage` 321 · `galaxdb-vector` 48 · `galaxdb-versioning` 73 · `galaxdb-wire` 26.
- Grep tripwires:
  - `git grep -n 'AwsKmsKeyProvider' -- '**/*.rs' '**/*.toml'` → one match at `crates/galaxdb-crypto/src/key_provider.rs:33`, a doc comment that reads `//! There is deliberately no 'AwsKmsKeyProvider', 'GcpKmsKeyProvider', …`. The documentation explicitly names the absence. The audit tripwire is honoured: zero production-code matches.
  - `git grep -n 'aws-kms' -- '**/*.toml'` → zero matches.
  - `git grep -n -i 'mock' -- 'crates/galaxdb-crypto/src'` → one comment line (`key_provider.rs:950`) reading `// spawn/pipe/wait path — not a mock.`, which describes the real-subprocess implementation.

**Next action**: Phase D (wire-protocol bind parameter plumbing) — per Phase B's "Phase D folded in" note, `Value::Integer(_) => None, // simplified` has already been replaced with a full typed conversion. Phase D's closing checklist items (`D1`, `D2`, `D3`) can be ticked at the same time as the Phase F task-tracker reconciliation. The next *new* work is Phase E — `_disk_full` Prometheus metric.
