#!/usr/bin/env bash
# Phase H gate H3: fail CI if tasks.md has a ticked checkbox on a line
# that references a known-stub file path AND the file still contains
# a stub marker.
#
# This is a narrow, conservative check. It catches regressions where
# someone re-ticks 18.7 BULK INSERT after Phase F unticked it, if the
# underlying executor still returns `GalaxError::NotYetAvailable`.
#
# It is NOT a general-purpose "is this task real" detector — that
# requires human review. See .kiro/steering/engineering-principles.md §7.

set -euo pipefail

TASKS=".kiro/specs/galaxdb-v1-engine/tasks.md"

# The spec tree under `.kiro/` is intentionally gitignored, so it is not
# present on a fresh CI checkout. When the tracker is absent we skip the
# tick-consistency checks (they have nothing to compare against) but still
# run the production-code stub tripwires below — those grep `crates/`,
# which IS tracked, and are the part of this gate that protects shipped
# code. Locally (where `.kiro/` exists) every check runs.
HAVE_TASKS=1
if [[ ! -f "$TASKS" ]]; then
  echo "NOTE: $TASKS not present (gitignored spec tree); running production-code tripwires only."
  HAVE_TASKS=0
fi

fail=0

# List of (task_id_regex, stub_grep_pattern, files) triples. If the
# task is ticked AND the stub pattern is present, fail.
check_stub_tick() {
  local task_pattern="$1"
  local stub_pattern="$2"
  shift 2
  local files=("$@")

  [[ "$HAVE_TASKS" -eq 1 ]] || return 0

  if grep -E "^\s*- \[x\] $task_pattern" "$TASKS" >/dev/null 2>&1; then
    for f in "${files[@]}"; do
      if [[ -f "$f" ]] && grep -q -F "$stub_pattern" "$f"; then
        echo "::error::Task '$task_pattern' is ticked but $f still contains stub marker: $stub_pattern"
        fail=1
      fi
    done
  fi
}

# 18.7 BULK INSERT — must not carry NotYetAvailable { task: "18.7" }
check_stub_tick '18\.7' 'task: "18.7"' crates/galaxdb-sql/src/executor.rs

# 37 BACKUP/RESTORE — must not carry NotYetAvailable { task: "37" }
check_stub_tick '37 ' 'task: "37"' crates/galaxdb-sql/src/executor.rs
check_stub_tick '37\.\d' 'task: "37"' crates/galaxdb-sql/src/executor.rs

# 32.3 / 32.4 AT VERSION — when these are ticked in tasks.md, the tracker
# must NOT still contain an explicit "Deferred" marker for Phase B6.
# We look for the exact phrase the Phase B entry used to carry
# (`[ ] B6.1: Deferred.`) rather than a loose substring, so "closed by
# Phase B6" references in later entries don't false-positive.
if [[ "$HAVE_TASKS" -eq 1 ]]; then
for tid in '32\.3' '32\.4' '32\.6'; do
  if grep -E "^\s*- \[x\] $tid" "$TASKS" >/dev/null 2>&1; then
    if grep -q '^\s*-\s*\[\s\]\s*B6\.1: Deferred\.' docs/CONSOLIDATION.md; then
      echo "::error::Task '$tid' is ticked but CONSOLIDATION.md still carries an open B6.1 deferral."
      fail=1
    fi
  fi
done

# 33.5 / 10.5 Pinned-block compactor — same rule for Phase B7.
for tid in '10\.5' '33\.5'; do
  if grep -E "^\s*- \[x\] $tid" "$TASKS" >/dev/null 2>&1; then
    if grep -q '^\s*-\s*\[\s\]\s*B7\.1: Deferred\.' docs/CONSOLIDATION.md; then
      echo "::error::Task '$tid' is ticked but CONSOLIDATION.md still carries an open B7.1 deferral."
      fail=1
    fi
  fi
done
fi

# Forbid 'In the full implementation' anywhere in crates/ (Phase B tripwire)
if grep -rn 'In the full implementation' crates/ 2>/dev/null | grep -v -E '^\s*#|//.*deleted'; then
  echo "::error::Stub comment 'In the full implementation' found in crates/ — Phase B tripwire."
  fail=1
fi

# Forbid 'For now we rely' in crates/ (Phase E tripwire)
if grep -rn 'For now we rely' crates/ 2>/dev/null; then
  echo "::error::Stub comment 'For now we rely' found in crates/ — Phase E tripwire."
  fail=1
fi

# Forbid NoOpVectorBackend anywhere (Phase B8 tripwire)
if grep -rn 'NoOpVectorBackend' crates/ 2>/dev/null; then
  echo "::error::'NoOpVectorBackend' still present — Phase B8 tripwire."
  fail=1
fi

# Forbid AwsKmsKeyProvider as a struct/type anywhere production
# (allow: doc comments that explicitly say "deliberately no AwsKmsKeyProvider")
if grep -rn 'AwsKmsKeyProvider' crates/*/src 2>/dev/null | grep -v -E '//!|// |deliberately no'; then
  echo "::error::'AwsKmsKeyProvider' present outside doc comments — Phase C tripwire."
  fail=1
fi

if [[ $fail -ne 0 ]]; then
  echo
  echo "FAIL: Phase H gate H3 detected task-tracker / code inconsistency."
  exit 1
fi

echo "OK: tasks.md and CONSOLIDATION.md are consistent with production code."
