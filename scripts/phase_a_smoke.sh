#!/usr/bin/env bash
# Phase A smoke test: launch the real sidecar with all-MiniLM-L6-v2 and
# verify it loads the model and accepts a Unix socket connection.
set -euo pipefail

SOCK=/tmp/galaxdb_phaseA_smoke.sock
rm -f "$SOCK"

BIN="target/release/galaxdb-sidecar"
if [ ! -x "$BIN" ]; then
  echo "FAIL: $BIN not found — build with 'cargo build -p galaxdb-sidecar --release'"
  exit 2
fi

"$BIN" --socket "$SOCK" --model sentence-transformers/all-MiniLM-L6-v2 &
SIDECAR_PID=$!
echo "Sidecar PID: $SIDECAR_PID"

# Cleanup on any exit path
trap 'kill "$SIDECAR_PID" 2>/dev/null || true; wait "$SIDECAR_PID" 2>/dev/null || true; rm -f "$SOCK"' EXIT INT TERM

# Wait up to 180 seconds for the socket to appear (first run includes
# the ~90 MB model download).
for i in $(seq 1 180); do
  if [ -S "$SOCK" ]; then
    echo "Socket appeared after ${i}s — real model loaded successfully"
    sleep 1
    echo "SUCCESS: sidecar started with real model"
    exit 0
  fi
  if ! kill -0 "$SIDECAR_PID" 2>/dev/null; then
    echo "FAIL: sidecar died before socket was ready"
    exit 3
  fi
  sleep 1
done

echo "FAIL: socket never appeared within 180s"
exit 4
