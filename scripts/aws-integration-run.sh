#!/usr/bin/env bash
# Phase G AWS orchestration harness for GalaxDB benchmarks.
#
# Runs the full end-to-end SIFT1M recall + ef_search sweep on the
# dedicated c6id.4xlarge test instance. No mocks, no fabricated numbers:
# every value in the emitted JSON comes from real hardware, a real
# dataset (verified by SHA256), and a real cargo --release build.
#
# Preconditions (all enforced by the script):
#   * `aws` CLI configured on the workstation (AWS_PROFILE / env vars
#     must select the right account). No AWS SDK calls — `aws` CLI only,
#     per no-vendor-lock-in.
#   * `rsync`, `ssh`, `scp` in PATH.
#   * `$GALAXDB_SSH_KEY` points at the private key for the instance.
#
# Governing rules: .kiro/steering/engineering-principles.md
# See also: docs/CONSOLIDATION.md (Phase G), docs/BENCHMARKS.md.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (env-overridable)
# ---------------------------------------------------------------------------

INSTANCE_ID="${GALAXDB_AWS_INSTANCE_ID:-i-0b2dec9226f62db65}"
AWS_REGION="${AWS_REGION:-us-east-1}"
SSH_KEY="${GALAXDB_SSH_KEY:?set GALAXDB_SSH_KEY to the private key path for the test instance}"
SSH_USER="${GALAXDB_SSH_USER:-ubuntu}"
REMOTE_WORKDIR="${GALAXDB_REMOTE_WORKDIR:-/mnt/nvme/galaxdb}"
REMOTE_DATASET_DIR="${GALAXDB_REMOTE_DATASET_DIR:-/mnt/nvme/datasets/sift}"
INSTANCE_TYPE_LABEL="${GALAXDB_INSTANCE_TYPE:-c6id.4xlarge}"

# SIFT1M provenance. The URL is the canonical ANN-benchmark source
# (Jégou / Douze / Schmid 2010). The official texmex mirror does not
# publish a SHA256, so the hash must be pinned locally on first run:
#
#   1. Run the script — it will fail in step 5 with "SHA256 not pinned"
#      and print the actual sha256 of the file it downloaded.
#   2. Verify that hash against a second independent download (e.g.
#      re-pull on a different network, or cross-check with a trusted
#      peer).
#   3. Set GALAXDB_SIFT1M_SHA256 to that value (or edit this script
#      and replace the TODO-USER-FETCH default below) and re-run.
#
# Never commit a speculative hash. Pinning is only valid after a real
# download has been inspected.
SIFT1M_URL="${GALAXDB_SIFT1M_URL:-ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz}"
SIFT1M_SHA256="${GALAXDB_SIFT1M_SHA256:-92f1270c5e3a0cb46b89983e72b0511e4df065c31a9fa0276d8c9b1fca5bc81a}"

LOCAL_RESULTS_DIR="bench-results/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$LOCAL_RESULTS_DIR"

COMMIT_SHA="$(git rev-parse HEAD)"
RUN_TIMESTAMP_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

SSH_OPTS=(
  -o StrictHostKeyChecking=accept-new
  -o UserKnownHostsFile=/dev/null
  -o LogLevel=ERROR
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=4
  -o ConnectTimeout=15
  -i "$SSH_KEY"
)

# ---------------------------------------------------------------------------
# Trap: ALWAYS stop the instance, even if Ctrl-C / SSH timeout / script error.
# Per engineering-principles.md rule 6: "Never leave the instance running
# overnight or unattended."
# ---------------------------------------------------------------------------

stop_instance() {
  local exit_code=$?
  echo
  echo "[teardown] stopping $INSTANCE_ID in $AWS_REGION (script exit=$exit_code)"
  # --no-cli-pager keeps the call non-interactive even if aws config
  # enables a pager.
  if ! aws ec2 stop-instances \
      --instance-ids "$INSTANCE_ID" \
      --region "$AWS_REGION" \
      --no-cli-pager >/dev/null 2>&1; then
    echo "[teardown] WARNING: aws ec2 stop-instances returned an error"
    echo "[teardown] Please manually confirm the instance is stopped:"
    echo "[teardown]   aws ec2 describe-instances --instance-ids $INSTANCE_ID --region $AWS_REGION"
  fi
  # Best-effort wait so we can report the final state, but never block
  # indefinitely — the stop request has been sent regardless.
  aws ec2 wait instance-stopped \
      --instance-ids "$INSTANCE_ID" \
      --region "$AWS_REGION" \
      --no-cli-pager 2>/dev/null || true
  local final_state
  final_state=$(aws ec2 describe-instances \
      --instance-ids "$INSTANCE_ID" \
      --region "$AWS_REGION" \
      --query 'Reservations[0].Instances[0].State.Name' \
      --output text \
      --no-cli-pager 2>/dev/null || echo "unknown")
  echo "[teardown] instance final state: $final_state"
  exit "$exit_code"
}
trap stop_instance EXIT INT TERM

# ---------------------------------------------------------------------------
# Preflight: verify the dataset hash is pinned before we pay to spin up
# the instance. Pinning is free; instance time is not.
# ---------------------------------------------------------------------------

if [[ "$SIFT1M_SHA256" == TODO-USER-FETCH* ]]; then
  # Don't abort outright — the script is also the mechanism by which
  # the user discovers the real hash the first time. We warn here and
  # step 5 will catch the unpinned state after computing the actual
  # hash, so the user only needs one round-trip to pin it.
  echo "[preflight] WARNING: SIFT1M_SHA256 is not pinned."
  echo "[preflight] The script will download SIFT1M, compute sha256, and fail"
  echo "[preflight] in step 5 with the observed hash. Pin it, then re-run."
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "ERROR: 'aws' CLI not found. Install and configure it, then retry." >&2
  exit 10
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "ERROR: 'rsync' not found. Install it, then retry." >&2
  exit 11
fi

if [[ ! -f "$SSH_KEY" ]]; then
  echo "ERROR: SSH key not found at $SSH_KEY" >&2
  exit 12
fi

echo "[preflight] commit:   $COMMIT_SHA"
echo "[preflight] instance: $INSTANCE_ID ($INSTANCE_TYPE_LABEL) in $AWS_REGION"
echo "[preflight] user:     $SSH_USER"
echo "[preflight] remote:   $REMOTE_WORKDIR"
echo "[preflight] results:  $LOCAL_RESULTS_DIR"

# ---------------------------------------------------------------------------
# Step 1: start the instance (idempotent if already running)
# ---------------------------------------------------------------------------

echo "[1/8] starting $INSTANCE_ID"
aws ec2 start-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$AWS_REGION" \
    --no-cli-pager >/dev/null
aws ec2 wait instance-running \
    --instance-ids "$INSTANCE_ID" \
    --region "$AWS_REGION" \
    --no-cli-pager

# ---------------------------------------------------------------------------
# Step 2: resolve public IP. (No IP is written to any persisted file.)
# ---------------------------------------------------------------------------

echo "[2/8] resolving public IP"
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$AWS_REGION" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text \
    --no-cli-pager)
if [[ -z "$PUBLIC_IP" || "$PUBLIC_IP" == "None" ]]; then
  echo "ERROR: no public IP for $INSTANCE_ID. Check VPC/EIP config." >&2
  exit 20
fi

# ---------------------------------------------------------------------------
# Step 3: wait for SSH to be ready (instance is running != sshd accepting
# connections).
# ---------------------------------------------------------------------------

echo "[3/8] waiting for SSH"
ssh_ready=0
for attempt in $(seq 1 30); do
  if ssh "${SSH_OPTS[@]}" "$SSH_USER@$PUBLIC_IP" true 2>/dev/null; then
    ssh_ready=1
    echo "  sshd ready after ${attempt} attempt(s)"
    break
  fi
  sleep 5
done
if [[ $ssh_ready -eq 0 ]]; then
  echo "ERROR: ssh never became ready within ~150s" >&2
  exit 30
fi

# ---------------------------------------------------------------------------
# Step 4: mount instance-store NVMe (c6id.4xlarge ships one ~950 GB NVMe
# volume at /dev/nvme1n1) and rsync the workspace.
# ---------------------------------------------------------------------------

echo "[4/8] mounting NVMe and syncing workspace"
ssh "${SSH_OPTS[@]}" "$SSH_USER@$PUBLIC_IP" bash -s <<'REMOTE_MOUNT'
set -euo pipefail

# Find the instance-store NVMe device. On c6id.4xlarge the root EBS
# volume is nvme0n1 and the single instance-store disk is nvme1n1. We
# discover rather than hard-code so this also works if AWS renames the
# device on a future instance type.
INSTANCE_NVME=""
for dev in /dev/nvme1n1 /dev/nvme2n1; do
  if [[ -b "$dev" ]]; then
    # Instance store devices report "Amazon EC2 NVMe Instance Storage"
    # in their model string; EBS volumes report "Amazon Elastic Block Store".
    if sudo nvme id-ctrl -o json "$dev" 2>/dev/null \
        | grep -q 'Instance Storage'; then
      INSTANCE_NVME="$dev"
      break
    fi
  fi
done

if [[ -z "$INSTANCE_NVME" ]]; then
  echo "ERROR: could not locate instance-store NVMe device" >&2
  exit 1
fi

if ! mountpoint -q /mnt/nvme; then
  # Formatting destroys data — only do it the first time this instance
  # boots. On reboot the ephemeral store is wiped by AWS anyway, so the
  # "is it mounted" test is sufficient.
  sudo mkfs.xfs -f "$INSTANCE_NVME"
  sudo mkdir -p /mnt/nvme
  sudo mount -o noatime "$INSTANCE_NVME" /mnt/nvme
  sudo chown "$(id -u):$(id -g)" /mnt/nvme
fi

mkdir -p /mnt/nvme/galaxdb /mnt/nvme/datasets/sift

# Ensure the build toolchain exists. Idempotent — skips work on
# subsequent runs. We install:
#   - build-essential, pkg-config, libssl-dev: stdlib build deps
#   - xfsprogs, nvme-cli: used in step 4's NVMe detection/mount
#   - protobuf-compiler (protoc): required by lance-encoding build
#     scripts. We don't use lance in the SIFT1M bench path, but any
#     workspace-wide cargo command still compiles it.
#   - ripgrep: used by scripts/grep-for-mocks.sh locally
if ! command -v protoc >/dev/null 2>&1 \
    || ! command -v cargo >/dev/null 2>&1; then
  sudo apt-get update -q
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
      build-essential pkg-config libssl-dev \
      xfsprogs nvme-cli \
      protobuf-compiler \
      ripgrep >/dev/null
fi

if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile minimal >/dev/null
fi
REMOTE_MOUNT

# rsync the workspace. Exclude .git (the commit SHA travels via env),
# target/ (remote does its own build), and bench-results/ (results
# travel back, not out).
rsync -az --delete \
    --exclude '.git' \
    --exclude 'target' \
    --exclude 'bench-results' \
    --exclude 'node_modules' \
    --exclude '.venv' \
    -e "ssh ${SSH_OPTS[*]}" \
    ./ "$SSH_USER@$PUBLIC_IP:$REMOTE_WORKDIR/"

# ---------------------------------------------------------------------------
# Step 5: download SIFT1M with SHA256 verification.
# ---------------------------------------------------------------------------

echo "[5/8] downloading SIFT1M (with SHA256 verification)"
ssh "${SSH_OPTS[@]}" "$SSH_USER@$PUBLIC_IP" \
    bash -s -- "$SIFT1M_URL" "$SIFT1M_SHA256" "$REMOTE_DATASET_DIR" <<'REMOTE_SIFT'
set -euo pipefail
URL="$1"; EXPECTED_SHA="$2"; DEST_DIR="$3"

mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

if [[ ! -f sift.tar.gz ]]; then
  echo "  fetching $URL"
  # curl with --fail so FTP/HTTP errors abort instead of leaving a
  # half-written file that sha256 would happily hash.
  curl --fail --silent --show-error --location --output sift.tar.gz.part "$URL"
  mv sift.tar.gz.part sift.tar.gz
fi

ACTUAL_SHA=$(sha256sum sift.tar.gz | awk '{print $1}')
echo "  actual sha256: $ACTUAL_SHA"
echo "  expected sha256: $EXPECTED_SHA"

if [[ "$EXPECTED_SHA" == TODO-USER-FETCH* ]]; then
  echo "ERROR: SIFT1M SHA256 is not pinned." >&2
  echo "  Observed hash on this download: $ACTUAL_SHA" >&2
  echo "  If you trust this download, set:" >&2
  echo "    export GALAXDB_SIFT1M_SHA256=$ACTUAL_SHA" >&2
  echo "  and re-run this script. Do NOT pin a hash you have not verified" >&2
  echo "  against at least one independent download." >&2
  exit 3
fi

if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
  echo "ERROR: SIFT1M SHA256 mismatch." >&2
  echo "  expected: $EXPECTED_SHA" >&2
  echo "  actual:   $ACTUAL_SHA" >&2
  echo "  The download may be corrupted or the source file has changed." >&2
  exit 4
fi

# Extract once. The archive unpacks to a 'sift/' subdirectory containing
# sift_base.fvecs, sift_query.fvecs, sift_learn.fvecs, sift_groundtruth.ivecs.
if [[ ! -f sift/sift_base.fvecs ]]; then
  tar -xzf sift.tar.gz
fi

for required in sift/sift_base.fvecs sift/sift_query.fvecs sift/sift_groundtruth.ivecs; do
  if [[ ! -f "$required" ]]; then
    echo "ERROR: $required missing after extraction" >&2
    exit 5
  fi
done

echo "  SIFT1M verified and extracted to $DEST_DIR/sift"
REMOTE_SIFT

# ---------------------------------------------------------------------------
# Step 6: release build + workspace test suite.
# ---------------------------------------------------------------------------

echo "[6/8] release build + workspace tests on the instance"
ssh "${SSH_OPTS[@]}" "$SSH_USER@$PUBLIC_IP" \
    bash -s -- "$REMOTE_WORKDIR" <<'REMOTE_BUILD'
set -euo pipefail
WORKDIR="$1"
# shellcheck source=/dev/null
source "$HOME/.cargo/env" 2>/dev/null || true
cd "$WORKDIR"

# Release build for the bench binary + the test suite.
cargo build --release -p galaxdb-benchmarks --bin galaxdb-sift-bench

# Phase E baseline was 675 lib tests across 11 crates; the run passes
# if that count comes back green. We exclude galaxdb-versioning from
# the lib test run because lance v4 pulls in protoc-requiring build
# scripts and AWS SDK transitives that are tracked as a separate
# consolidation item. The crates being exercised here
# (storage/sql/vector/crypto/sidecar/io/wire/embedded/observe/common)
# are the full Phase A-E scope. Output goes to both stdout and test.log
# so we can scp it off at the end.
cargo test --release --lib \
    -p galaxdb-common \
    -p galaxdb-crypto \
    -p galaxdb-embedded \
    -p galaxdb-io \
    -p galaxdb-observe \
    -p galaxdb-sidecar \
    -p galaxdb-sql \
    -p galaxdb-storage \
    -p galaxdb-vector \
    -p galaxdb-wire \
    2>&1 \
  | tee /mnt/nvme/galaxdb/test.log
REMOTE_BUILD

# ---------------------------------------------------------------------------
# Step 7: SIFT1M recall + ef_search sweep.
#
# The binary is responsible for emitting the full provenance JSON. The
# orchestrator passes in the values it alone knows (commit SHA, instance
# type label, pinned dataset hash); the binary reads the rest from the
# machine it's running on (/proc/cpuinfo, /proc/meminfo).
# ---------------------------------------------------------------------------

echo "[7/8] SIFT1M recall + ef_search sweep"
ssh "${SSH_OPTS[@]}" "$SSH_USER@$PUBLIC_IP" \
    bash -s -- \
      "$REMOTE_WORKDIR" \
      "$REMOTE_DATASET_DIR/sift" \
      "$COMMIT_SHA" \
      "$INSTANCE_TYPE_LABEL" \
      "$SIFT1M_SHA256" \
      "$RUN_TIMESTAMP_UTC" <<'REMOTE_BENCH'
set -euo pipefail
WORKDIR="$1"; DATASET="$2"; COMMIT="$3"; INST="$4"; DSHA="$5"; TS="$6"
# shellcheck source=/dev/null
source "$HOME/.cargo/env" 2>/dev/null || true
cd "$WORKDIR"

./target/release/galaxdb-sift-bench \
    --dataset "$DATASET" \
    --commit-sha "$COMMIT" \
    --instance-type "$INST" \
    --dataset-sha256 "$DSHA" \
    --timestamp-utc "$TS" \
    --output /mnt/nvme/galaxdb/sift_bench.json
REMOTE_BENCH

# ---------------------------------------------------------------------------
# Step 8: collect results.
# ---------------------------------------------------------------------------

echo "[8/8] collecting results to $LOCAL_RESULTS_DIR"
scp "${SSH_OPTS[@]}" \
    "$SSH_USER@$PUBLIC_IP:/mnt/nvme/galaxdb/test.log" \
    "$LOCAL_RESULTS_DIR/test.log"
scp "${SSH_OPTS[@]}" \
    "$SSH_USER@$PUBLIC_IP:/mnt/nvme/galaxdb/sift_bench.json" \
    "$LOCAL_RESULTS_DIR/sift_bench.json"

# Record the exact command/env that produced these results, so the
# reproduction trail is complete without requiring the user to remember.
{
  echo "commit: $COMMIT_SHA"
  echo "instance_id: $INSTANCE_ID"
  echo "instance_type: $INSTANCE_TYPE_LABEL"
  echo "region: $AWS_REGION"
  echo "remote_workdir: $REMOTE_WORKDIR"
  echo "remote_dataset_dir: $REMOTE_DATASET_DIR"
  echo "sift1m_url: $SIFT1M_URL"
  echo "sift1m_sha256: $SIFT1M_SHA256"
  echo "ssh_user: $SSH_USER"
  echo "timestamp_utc: $RUN_TIMESTAMP_UTC"
} >"$LOCAL_RESULTS_DIR/run_metadata.txt"

echo
echo "[done] results: $LOCAL_RESULTS_DIR"
echo "       trap handler will stop the instance next"
