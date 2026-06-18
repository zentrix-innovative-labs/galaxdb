#!/usr/bin/env bash
# Real-hardware benchmark + integration run for the v2-phase1 WIRE work
# that has not yet been measured on the dedicated instance:
#
#   * task 9  — extended query protocol (Parse/Bind/Describe/Execute)
#   * task 10 — statement cache / single-row INSERT (parse-once)
#   * task 11 — COPY protocol (bulk load over the wire)
#
# Builds --release on the c6id.4xlarge, runs the networked
# wire_integration suite (real tokio-postgres over TCP), then runs the
# two throughput benchmarks (single-row-insert-bench, copy-bench) and
# collects their JSON. No mocks, no fabricated numbers.
#
# Governing rules: .kiro/steering/engineering-principles.md
#   §4 benchmarks: --release, named hardware, reproducible command.
#   §6 AWS discipline: always start, mount NVMe, run, then STOP.

set -euo pipefail

INSTANCE_ID="${GALAXDB_AWS_INSTANCE_ID:-i-0b2dec9226f62db65}"
AWS_REGION="${AWS_REGION:-us-east-1}"
SSH_KEY="${GALAXDB_SSH_KEY:-$HOME/.ssh/galaxdb-bench-key.pem}"
SSH_USER="${GALAXDB_SSH_USER:-ubuntu}"
REMOTE_WORKDIR="${GALAXDB_REMOTE_WORKDIR:-/mnt/nvme/galaxdb}"
ROWS="${GALAXDB_BENCH_ROWS:-200000}"

LOCAL_RESULTS_DIR="bench-results/wire-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$LOCAL_RESULTS_DIR"
COMMIT_SHA="$(git rev-parse HEAD)"
# We may run this BEFORE committing (rsync ships the working tree, not
# git). Mark the provenance honestly so a dirty run is never mistaken
# for a clean-commit benchmark.
if ! git diff --quiet || ! git diff --cached --quiet; then
  COMMIT_SHA="${COMMIT_SHA}-dirty"
fi
RUN_TS_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

SSH_OPTS=(
  -o StrictHostKeyChecking=accept-new
  -o UserKnownHostsFile=/dev/null
  -o LogLevel=ERROR
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=8
  -o ConnectTimeout=15
  -i "$SSH_KEY"
)

stop_instance() {
  local exit_code=$?
  echo
  echo "[teardown] stopping $INSTANCE_ID (script exit=$exit_code)"
  aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region "$AWS_REGION" \
      --no-cli-pager >/dev/null 2>&1 \
    || echo "[teardown] WARNING: stop-instances errored; verify manually"
  aws ec2 wait instance-stopped --instance-ids "$INSTANCE_ID" \
      --region "$AWS_REGION" --no-cli-pager 2>/dev/null || true
  local s
  s=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
      --region "$AWS_REGION" \
      --query 'Reservations[0].Instances[0].State.Name' \
      --output text --no-cli-pager 2>/dev/null || echo unknown)
  echo "[teardown] instance final state: $s"
  exit "$exit_code"
}
trap stop_instance EXIT INT TERM

[[ -f "$SSH_KEY" ]] || { echo "ERROR: SSH key not found at $SSH_KEY" >&2; exit 12; }

echo "[preflight] commit:   $COMMIT_SHA"
echo "[preflight] instance: $INSTANCE_ID ($AWS_REGION)"
echo "[preflight] rows:     $ROWS"
echo "[preflight] results:  $LOCAL_RESULTS_DIR"

echo "[1/6] starting instance"
aws ec2 start-instances --instance-ids "$INSTANCE_ID" --region "$AWS_REGION" \
    --no-cli-pager >/dev/null
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$AWS_REGION" \
    --no-cli-pager

echo "[2/6] resolving public IP"
PUBLIC_IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
    --region "$AWS_REGION" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text --no-cli-pager)
[[ -n "$PUBLIC_IP" && "$PUBLIC_IP" != "None" ]] || { echo "ERROR: no public IP" >&2; exit 20; }

echo "[3/6] waiting for SSH"
ready=0
for _ in $(seq 1 30); do
  if ssh "${SSH_OPTS[@]}" "$SSH_USER@$PUBLIC_IP" true 2>/dev/null; then ready=1; break; fi
  sleep 5
done
[[ $ready -eq 1 ]] || { echo "ERROR: ssh never ready" >&2; exit 30; }

echo "[4/6] mounting NVMe + ensuring toolchain"
ssh "${SSH_OPTS[@]}" "$SSH_USER@$PUBLIC_IP" bash -s <<'REMOTE_MOUNT'
set -euo pipefail
INSTANCE_NVME=""
for dev in /dev/nvme1n1 /dev/nvme2n1; do
  if [[ -b "$dev" ]] && sudo nvme id-ctrl -o json "$dev" 2>/dev/null | grep -q 'Instance Storage'; then
    INSTANCE_NVME="$dev"; break
  fi
done
[[ -n "$INSTANCE_NVME" ]] || { echo "ERROR: no instance-store NVMe" >&2; exit 1; }
if ! mountpoint -q /mnt/nvme; then
  sudo mkfs.xfs -f "$INSTANCE_NVME"
  sudo mkdir -p /mnt/nvme
  sudo mount -o noatime "$INSTANCE_NVME" /mnt/nvme
  sudo chown "$(id -u):$(id -g)" /mnt/nvme
fi
mkdir -p /mnt/nvme/galaxdb
if ! command -v protoc >/dev/null 2>&1 || ! command -v cargo >/dev/null 2>&1; then
  sudo apt-get update -q
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
      build-essential pkg-config libssl-dev xfsprogs nvme-cli \
      protobuf-compiler cmake ripgrep >/dev/null
fi
if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile minimal >/dev/null
fi
REMOTE_MOUNT

echo "[4b/6] syncing workspace"
rsync -az --delete \
    --exclude '.git' --exclude 'target' --exclude 'bench-results' \
    --exclude 'node_modules' --exclude '.venv' \
    --exclude 'galaxdb-docs' --exclude 'galaxdb-landing' \
    -e "ssh ${SSH_OPTS[*]}" \
    ./ "$SSH_USER@$PUBLIC_IP:$REMOTE_WORKDIR/"

echo "[5/6] release build + wire integration + throughput benchmarks"
ssh "${SSH_OPTS[@]}" "$SSH_USER@$PUBLIC_IP" \
    bash -s -- "$REMOTE_WORKDIR" "$ROWS" "$COMMIT_SHA" "$RUN_TS_UTC" \
    <<'REMOTE_RUN' | tee "$LOCAL_RESULTS_DIR/wire-bench.log"
set -euo pipefail
WORKDIR="$1"; ROWS="$2"; COMMIT="$3"; TS="$4"
source "$HOME/.cargo/env" 2>/dev/null || true
cd "$WORKDIR"

echo "=== rustc/cargo versions ==="
rustc --version; cargo --version
echo "=== uname / cpu / mem ==="
uname -a; nproc; grep MemTotal /proc/meminfo

echo "=== release build: server + benchmarks ==="
cargo build --release -p galaxdb-server -p galaxdb-benchmarks \
    --bin single-row-insert-bench --bin copy-bench

echo "=== networked wire_integration (--release): tasks 9/10/11 over real TCP ==="
cargo test --release -p galaxdb-server --test wire_integration

echo "=== single-row INSERT throughput (task 10): simple vs prepared ==="
./target/release/single-row-insert-bench --rows "$ROWS"

echo "=== COPY bulk-load throughput (task 11): insert vs copy (JSON) ==="
./target/release/copy-bench --rows "$ROWS" --json \
    --commit-sha "$COMMIT" --instance-type c6id.4xlarge --timestamp-utc "$TS" \
    | tee /mnt/nvme/galaxdb/copy_bench.json

echo "=== ALL WIRE BENCHMARKS COMPLETED ON REAL HARDWARE ==="
REMOTE_RUN

echo "[6/6] collecting JSON result"
scp "${SSH_OPTS[@]}" \
    "$SSH_USER@$PUBLIC_IP:/mnt/nvme/galaxdb/copy_bench.json" \
    "$LOCAL_RESULTS_DIR/copy_bench.json" 2>/dev/null \
  || echo "[6/6] WARNING: copy_bench.json not collected (see wire-bench.log)"

{
  echo "commit: $COMMIT_SHA"
  echo "instance_id: $INSTANCE_ID"
  echo "instance_type: c6id.4xlarge"
  echo "region: $AWS_REGION"
  echo "rows_per_method: $ROWS"
  echo "timestamp_utc: $RUN_TS_UTC"
  echo "run_kind: v2-phase1 wire benchmarks (tasks 9/10/11: extended protocol, stmt cache, COPY)"
} >"$LOCAL_RESULTS_DIR/run_metadata.txt"
echo "[done] results in $LOCAL_RESULTS_DIR; trap will stop the instance next"
