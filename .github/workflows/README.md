# CI workflows

## `ci.yml`

One workflow, four jobs that run in parallel on every push to `main` and every pull request targeting `main`.

### `build-and-test` — standard build + unit tests + clippy

Runs `cargo build --release`, the workspace unit test suite (excluding `galaxdb-python` which needs pyo3 headers in a dedicated image), and `cargo clippy -- -D warnings`.

### `no-mocks-gate` — Phase H gate H1

Runs [`scripts/grep-for-mocks.sh`](../../scripts/grep-for-mocks.sh). Fails if the word `mock` appears in any non-test Rust source file under `crates/`, `benchmarks/`, or `galaxdb-python/`.

Allowed locations (files matching any of these patterns are skipped):
- paths under `tests/`
- files ending in `_tests.rs` or `_test.rs`
- files named `tests.rs`
- paths under `benches/`
- the harness-style `benchmarks/src/integration_test.rs`

Allowed content inside production files: comments that explicitly document the *absence* of a mock, e.g. `// no mock fallback`, `// not a mock`, `//! There is deliberately no …`. The grep matches `\bmock` case-insensitively and then filters out these negation patterns.

If a PR adds `fn mock_foo()` to any production file, this gate fails. See `.kiro/steering/engineering-principles.md` §1.

### `no-vendor-sdk-gate` — Phase H gate H2

Runs `cargo deny check {bans,licenses,advisories}` against [`deny.toml`](../../deny.toml). The `[bans]` section denies every known cloud-provider SDK crate:
- `aws-sdk-*` (kms, s3, dynamodb, secretsmanager, sts, config)
- `google-cloud-*` / `gcloud-sdk`
- `azure_*` (core, identity, storage, security_keyvault)

If a PR adds a direct or transitive dependency on any of these, this gate fails. Cloud KMS users MUST use `ExternalCommandKeyProvider` with the vendor's CLI, or the pure-Rust `HashicorpVaultKeyProvider` (feature `vault`). See `.kiro/steering/engineering-principles.md` §5.

The same job also fails on any open RustSec advisory (`check advisories`) and any unrecognised license (`check licenses`).

### `task-tracker-gate` — Phase H gate H3

Runs [`scripts/check-tasks-no-stub-ticks.sh`](../../scripts/check-tasks-no-stub-ticks.sh). Fails if any of the following hold:
- `tasks.md` has `- [x]` for task 18.7 while `crates/galaxdb-sql/src/executor.rs` still contains `task: "18.7"` (the `GalaxError::NotYetAvailable` tag).
- `tasks.md` has `- [x]` for task 37 / 37.x while the executor still carries `task: "37"`.
- `tasks.md` has `- [x]` for tasks 32.3 / 32.4 / 32.6 while `CONSOLIDATION.md` still marks Phase B6 as deferred.
- `tasks.md` has `- [x]` for tasks 10.5 / 33.5 while `CONSOLIDATION.md` still marks Phase B7 as deferred.
- Any `crates/` file contains the stub comment `In the full implementation` (Phase B tripwire).
- Any `crates/` file contains `For now we rely` (Phase E tripwire).
- Any `crates/` file references `NoOpVectorBackend` (Phase B8 tripwire).
- Any `crates/*/src/` file references `AwsKmsKeyProvider` outside of doc comments that explicitly note its deliberate absence (Phase C tripwire).

See `.kiro/steering/engineering-principles.md` §7 ("Task tracker is the source of truth").

## Local preflight

Run the same gates locally before pushing:

```bash
bash scripts/grep-for-mocks.sh
bash scripts/check-tasks-no-stub-ticks.sh
cargo deny check bans licenses advisories
```

`cargo-deny` is installable once with `cargo install --locked cargo-deny`.
