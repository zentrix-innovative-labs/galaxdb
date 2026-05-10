#!/usr/bin/env bash
# Phase H gate H1: fail CI if the word "mock" appears in any non-test
# Rust source file.
#
# What counts as a test file (allowed to use "mock"):
#   * paths under a `tests/` directory
#   * files ending in `_tests.rs` or `_test.rs`
#   * files named exactly `tests.rs` (module-internal test files)
#   * files inside `benches/` (criterion harnesses)
#
# What counts as a production file (NOT allowed):
#   * anything else under `crates/*/src/`, `galaxdb-python/src/`,
#     `benchmarks/src/` (excluding `benchmarks/src/bin/*_bench*.rs`
#     and the `integration_test.rs` / `sift_bench.rs` harnesses which
#     are test-style even though they aren't under tests/)
#
# The test file `crates/galaxdb-sql/src/executor_tests.rs` contains a
# `MockVectorBackend` struct which is correct per the audit. That file
# matches the `_tests.rs` allowlist.
#
# Governing rule: .kiro/steering/engineering-principles.md §1.

set -euo pipefail

fail=0

while IFS= read -r -d '' file; do
  # Skip allowlisted test paths.
  case "$file" in
    */tests/*)       continue ;;
    *_tests.rs)      continue ;;
    *_test.rs)       continue ;;
    */tests.rs)      continue ;;
    */benches/*)     continue ;;
    # The benchmark crate has a few top-level harness files that are
    # effectively test drivers. The current tree:
    #   benchmarks/src/sift_bench.rs         - real-dataset bench (no mocks)
    #   benchmarks/src/vector_bench.rs       - internal diagnostic, may reference "mock" indirectly
    #   benchmarks/src/integration_test.rs   - harness-style, test-named
    # We only treat integration_test.rs as explicitly test-like.
    */benchmarks/src/integration_test.rs) continue ;;
  esac

  # Case-insensitive match on the literal word "mock", but ignore:
  #   * comments that explicitly document the *absence* of a mock
  #     (e.g. "no mock fallback", "not a mock")
  #   * doc-comment mentions of historical removals
  #
  # We use ripgrep when available for speed, else grep.
  if command -v rg >/dev/null 2>&1; then
    matches=$(rg -n -i '\bmock' "$file" || true)
  else
    matches=$(grep -n -i -E '\bmock' "$file" || true)
  fi

  # Filter out allow-listed comment patterns.
  filtered=$(echo "$matches" | grep -v -i -E '(no mock|not a mock|never a mock|not.{0,5}mock|without mocks|no mocks|there is deliberately no|mock fallback|skip a mock|allowed to use "mock")' || true)

  if [[ -n "$filtered" ]]; then
    echo "::error file=$file::forbidden 'mock' reference in production code"
    echo "$filtered"
    fail=1
  fi
done < <(find crates benchmarks galaxdb-python -name '*.rs' -print0 2>/dev/null)

if [[ $fail -ne 0 ]]; then
  echo
  echo "FAIL: Phase H gate H1 detected 'mock' in production code."
  echo "Rule: .kiro/steering/engineering-principles.md §1 — mocks are only"
  echo "allowed inside tests/, *_tests.rs, tests.rs, or benches/."
  exit 1
fi

echo "OK: no forbidden mock references in production code."
