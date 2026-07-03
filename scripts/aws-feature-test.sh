#!/usr/bin/env bash
# Real-hardware integration run for the v2-phase1 feature work
# (auth/SCRAM, TLS, authorization + audit, secondary indexes, AT VERSION
# over SST). Builds --release on the dedicated c6id.4xlarge instance and
# runs the feature test suites — including the networked wire_integration
# tests that drive a real tokio-postgres client against a real server over
# TCP — so we verify the engine working in realtime, not just locally.
#
# No mocks, no fabricated results. Governing rules:
# .kiro/steering/engineering-principles.md (rule 6: always stop the
# instance, never leave it running unattended).

set -euo pipefail

INSTANCE_ID="${GALAXDB_AWS_INSTANCE_ID:?Set GALAXDB_AWS_INSTANCE_ID to your benchmark instance ID}"
AWS_REGION="${AWS_REGION:-us-east-1}"
SSH_KEY="${GALAXDB_SSH_KEY:-$HOME/.ssh/galaxdb-bench-key.pem}"
SSH_USER="${GALAXDB_SSH_USER:-ubuntu}"
REMOTE_WORKDIR="${GALAXDB_REMOTE_WORKDIR:-/mnt/nvme/galaxdb}"

LOCAL_RESULTS_DIR="bench-results/feature-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$LOCAL_RESULTS_DIR"
COMMIT_SHA="$(git rev-parse HEAD)"

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

echo "[5/6] release build + feature tests on the instance"
ssh "${SSH_OPTS[@]}" "$SSH_USER@$PUBLIC_IP" \
    bash -s -- "$REMOTE_WORKDIR" <<'REMOTE_TEST' | tee "$LOCAL_RESULTS_DIR/feature-test.log"
set -euo pipefail
WORKDIR="$1"
source "$HOME/.cargo/env" 2>/dev/null || true
cd "$WORKDIR"

echo "=== rustc/cargo versions ==="
rustc --version; cargo --version
echo "=== uname / cpu / mem ==="
uname -a; nproc; grep MemTotal /proc/meminfo

echo "=== release build (workspace, excluding galaxdb-python pyo3 linker) ==="
cargo build --release --workspace --exclude galaxdb-python

echo "=== feature lib tests (--release): auth, sql, embedded, wire ==="
cargo test --release --lib \
    -p galaxdb-auth -p galaxdb-sql -p galaxdb-embedded -p galaxdb-wire

echo "=== networked integration (--release): SCRAM + TLS + authz + audit over real TCP ==="
cargo test --release -p galaxdb-server --test wire_integration

echo "=== ALL FEATURE TESTS PASSED ON REAL HARDWARE ==="
REMOTE_TEST

echo "[6/6] results saved to $LOCAL_RESULTS_DIR/feature-test.log"
{
  echo "commit: $COMMIT_SHA"
  echo "instance_id: $INSTANCE_ID"
  echo "instance_type: c6id.4xlarge"
  echo "region: $AWS_REGION"
  echo "run_kind: v2-phase1 feature integration (auth/TLS/authz/audit/secondary-index/AT-VERSION)"
} >"$LOCAL_RESULTS_DIR/run_metadata.txt"
echo "[done] trap handler will stop the instance next"
