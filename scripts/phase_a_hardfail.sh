#!/usr/bin/env bash
# Phase A hard-fail test: a bogus model id must make the sidecar exit
# non-zero. No mock fallback permitted.
set -uo pipefail

SOCK=/tmp/galaxdb_phaseA_hardfail.sock
rm -f "$SOCK"

BIN="target/release/galaxdb-sidecar"
if [ ! -x "$BIN" ]; then
  echo "FAIL: $BIN not found — build with 'cargo build -p galaxdb-sidecar --release'"
  exit 2
fi

set +e
"$BIN" --socket "$SOCK" --model "definitely-not-a-real-model-42abc/nope" 2>sidecar_err.log
status=$?
set -e

echo "Sidecar exit status: $status"
echo "--- stderr ---"
cat sidecar_err.log
echo "--------------"

if [ $status -eq 0 ]; then
  echo "FAIL: sidecar exited 0 on a nonexistent model — a mock fallback still exists"
  rm -f sidecar_err.log "$SOCK"
  exit 3
fi

if [ -S "$SOCK" ]; then
  echo "FAIL: sidecar created a socket before dying — it served requests without a real model"
  rm -f sidecar_err.log "$SOCK"
  exit 4
fi

if ! grep -q "failed to load model" sidecar_err.log; then
  echo "FAIL: sidecar died but did not print a typed model-load error"
  rm -f sidecar_err.log "$SOCK"
  exit 5
fi

echo "SUCCESS: bogus model → exit $status with typed error, no socket, no mock fallback"
rm -f sidecar_err.log "$SOCK"
exit 0
