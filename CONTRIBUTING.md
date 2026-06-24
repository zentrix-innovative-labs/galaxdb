# Contributing to GalaxDB

Thank you for your interest in contributing. GalaxDB is an AI-native database written in Rust with a Python client. We welcome bug reports, feature requests, documentation improvements, and code contributions.

## Before you start

- Read the [Code of Conduct](CODE_OF_CONDUCT.md).
- For large changes, open an issue first to discuss the approach.
- All contributions must pass the full test suite and the three CI gates (no mocks, no vendor SDKs, task tracker).

## Development setup

**Requirements:** Rust stable (1.80+), Python 3.9+, maturin

```bash
git clone https://github.com/zentrix-innovative-labs/galaxdb
cd galaxdb

# Build and test the Rust workspace
cargo build --workspace
cargo test --workspace --exclude galaxdb-python --lib

# Build the Python wheel (for Python client tests)
pip install maturin
maturin develop -m galaxdb-python/Cargo.toml

# Run Python tests
pip install pytest pylance pyarrow psycopg2-binary sqlalchemy
pytest galaxdb-python/tests/python/ -v
```

## Two-repository model (OSS core + Enterprise edition)

GalaxDB ships as two repositories:

- **`galaxdb`** (this repo) — the Apache-2.0 open-source core: the entire single-node engine,
  including security, encryption, and the stable extension interfaces (`galaxdb-auth`,
  `galaxdb-cluster`, key-provider and object-store traits) the enterprise edition plugs into.
- **`galaxdb-enterprise`** (private) — distributed clustering, distributed ANN, SSO/fine-grained
  RBAC, and storage tiering, built **on top of** the OSS extension seams. The OSS core never
  depends on enterprise code (enforced by `deny.toml`, which bans any `galaxdb-enterprise*`
  dependency in the OSS graph).

### Bug-fix routing rule

- A bug in OSS behavior (anything under `crates/` in this repo) is fixed **here**, in the OSS repo.
  The enterprise edition then bumps its pinned OSS version tag to pick up the fix.
- A bug that exists only in enterprise code is fixed in the enterprise repo **only** — never work
  around an OSS-seam shortcoming by patching enterprise; if the seam is wrong, fix the seam here.
- Changing an extension seam (a public trait the enterprise edition implements) is a semver-relevant
  change: bump the crate version and call it out in the PR so the enterprise pin can be updated
  deliberately.

## Engineering rules (non-negotiable)

These rules are enforced by CI and will cause your PR to fail if violated:

1. **No mocks in production code paths.** Mocks belong in `#[cfg(test)]` blocks only.
2. **No silent fallbacks.** If a real implementation fails, surface a typed error.
3. **No task ticked without real implementation.** Tests must exercise real code.
4. **No vendor lock-in.** No `aws-sdk-*`, `google-cloud-*`, or `azure_*` dependencies.
5. **Cross-platform.** Code must compile and pass tests on macOS, Linux, and Windows.

## Running the CI gates locally

```bash
bash scripts/grep-for-mocks.sh
bash scripts/check-tasks-no-stub-ticks.sh
cargo deny check
```

## Pull request process

1. Fork the repo and create a branch from `main`.
2. Make your changes with tests.
3. Run `cargo test --workspace --exclude galaxdb-python --lib` — all tests must pass.
4. Run the three CI gates above — all must exit 0.
5. Open a PR against `main` with a clear description of what changed and why.
6. A maintainer will review within 5 business days.

## Reporting bugs

Open a GitHub issue with:
- GalaxDB version (`cargo pkgid galaxdb-server | cut -d# -f2`)
- OS and architecture
- Minimal reproduction case
- Expected vs actual behaviour

## Security vulnerabilities

Do not open a public issue for security vulnerabilities. Email security@zentrix.ai instead.
