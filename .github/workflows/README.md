# CI Workflows

## Jobs

### `build-and-test`
Runs on every push and PR to `main`.
- `cargo build --release` — full workspace release build
- `cargo test --workspace --exclude galaxdb-python --lib` — all Rust unit tests
- `cargo clippy -- -D warnings` — lint

### `no-mocks-gate`
Fails if any production Rust file contains a mock, stub, or fake on a non-test code path. Test files (`tests/`, `*_tests.rs`, `benches/`) are excluded.

Run locally: `bash scripts/grep-for-mocks.sh`

### `no-vendor-sdk-gate`
Fails if any AWS, Google Cloud, or Azure SDK appears in `Cargo.lock`. GalaxDB is vendor-neutral — cloud KMS is supported via `ExternalCommandKeyProvider` (shell command) or `HashicorpVaultKeyProvider`.

Run locally: `cargo deny check bans`

### `task-tracker-gate`
Fails if `tasks.md` ticks a task that still has a stub marker in the production code. Prevents premature task completion.

Run locally: `bash scripts/check-tasks-no-stub-ticks.sh`

### `python-integration-tests`
Builds the Python wheel via `maturin develop --release` and runs `pytest galaxdb-python/tests/python/ -v`. Requires Python 3.11+, lance, pyarrow, torch.

### `chaos-tests`
Runs the 7 chaos scenarios in release mode. All recovery scenarios must complete in < 30 s each.

Run locally: `cargo run --release -p galaxdb-chaos-tests`
