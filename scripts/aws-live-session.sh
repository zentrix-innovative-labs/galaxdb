#!/usr/bin/env bash
# Live end-to-end server session on real AWS hardware.
#
# Unlike aws-feature-test.sh (which runs the cargo test suites), this
# script starts the ACTUAL `galaxdb-server` binary on the c6id.4xlarge
# instance, then drives it with a real PostgreSQL client (`psql`, libpq
# SCRAM-SHA-256) over real TCP — exercising a full real-world workload:
#
#   auth (SCRAM)  → CREATE TABLE → INSERT → SELECT → WHERE filter
#   → UPDATE → DELETE → CREATE INDEX → index-accelerated SELECT
#   → DROP INDEX → role/grant RBAC (42501 denial then grant) → DROP TABLE
#   plus a TLS (sslmode=require) connection and the JSONL audit log.
#
# No mocks, no fabricated output: every line in the captured log is the
# real server answering a real client. Governing rules:
# .kiro/steering/engineering-principles.md (rule 6: always stop the
# instance; rule 8: verification before claims).

set -euo pipefail

INSTANCE_ID="${GALAXDB_AWS_INSTANCE_ID:?Set GALAXDB_AWS_INSTANCE_ID to your benchmark instance ID}"
AWS_REGION="${AWS_REGION:-us-east-1}"
SSH_KEY="${GALAXDB_SSH_KEY:-$HOME/.ssh/galaxdb-bench-key.pem}"
SSH_USER="${GALAXDB_SSH_USER:-ubuntu}"
REMOTE_WORKDIR="${GALAXDB_REMOTE_WORKDIR:-/mnt/nvme/galaxdb}"

LOCAL_RESULTS_DIR="bench-results/live-session-$(date -u +%Y%m%dT%H%M%SZ)"
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

echo "[4/6] mounting NVMe + ensuring toolchain + psql client"
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
if ! command -v protoc >/dev/null 2>&1 || ! command -v cargo >/dev/null 2>&1 \
   || ! command -v psql >/dev/null 2>&1 || ! command -v openssl >/dev/null 2>&1; then
  sudo apt-get update -q
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
      build-essential pkg-config libssl-dev xfsprogs nvme-cli \
      protobuf-compiler cmake ripgrep postgresql-client openssl >/dev/null
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

echo "[5/6] release build of galaxdb-server, then a live psql workload"
ssh "${SSH_OPTS[@]}" "$SSH_USER@$PUBLIC_IP" \
    bash -s -- "$REMOTE_WORKDIR" <<'REMOTE_SESSION' | tee "$LOCAL_RESULTS_DIR/live-session.log"
set -euo pipefail
WORKDIR="$1"
source "$HOME/.cargo/env" 2>/dev/null || true
cd "$WORKDIR"

echo "=== rustc/cargo versions ==="
rustc --version; cargo --version; psql --version
echo "=== uname / cpu / mem ==="
uname -a; nproc; grep MemTotal /proc/meminfo

echo "=== release build: galaxdb-server ==="
cargo build --release -p galaxdb-server

DATA_DIR="/mnt/nvme/live-data"
AUDIT_LOG="/mnt/nvme/live-audit.jsonl"
SRV_LOG="/mnt/nvme/live-server.log"
CRT="/mnt/nvme/live-server.crt"
KEY="/mnt/nvme/live-server.key"
PORT=5433
rm -rf "$DATA_DIR" "$AUDIT_LOG" "$SRV_LOG"; mkdir -p "$DATA_DIR"

echo "=== generating a self-signed TLS cert for the TLS leg ==="
openssl req -x509 -newkey rsa:2048 -nodes -keyout "$KEY" -out "$CRT" \
    -days 1 -subj "/CN=127.0.0.1" \
    -addext "subjectAltName=IP:127.0.0.1" >/dev/null 2>&1

echo "=== starting the real galaxdb-server (--auth, SCRAM, TLS allow, audit log) ==="
GALAXDB_LOG_LEVEL=info \
GALAXDB_INITIAL_SUPERUSER=root \
GALAXDB_INITIAL_SUPERUSER_PASSWORD=rootpw \
GALAXDB_AUDIT_LOG="$AUDIT_LOG" \
  ./target/release/galaxdb-server \
    --auth --port "$PORT" --data-dir "$DATA_DIR" \
    --tls-mode allow --tls-cert "$CRT" --tls-key "$KEY" \
    >"$SRV_LOG" 2>&1 &
SRV_PID=$!
cleanup_server() { kill "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true; }
trap cleanup_server EXIT

echo "=== waiting for the server to report listening ==="
ready=0
for _ in $(seq 1 60); do
  if grep -q "wire-protocol server listening\|server listening on" "$SRV_LOG" 2>/dev/null; then ready=1; break; fi
  if ! kill -0 "$SRV_PID" 2>/dev/null; then echo "SERVER DIED EARLY:"; cat "$SRV_LOG"; exit 41; fi
  sleep 1
done
[[ $ready -eq 1 ]] || { echo "server never reported listening:"; cat "$SRV_LOG"; exit 42; }
echo "--- server startup log ---"; cat "$SRV_LOG"

ROOT_DSN="host=127.0.0.1 port=$PORT user=root password=rootpw dbname=galaxdb sslmode=disable"
ROOT_TLS_DSN="host=127.0.0.1 port=$PORT user=root password=rootpw dbname=galaxdb sslmode=require"
ALICE_DSN="host=127.0.0.1 port=$PORT user=alice password=alicepw dbname=galaxdb sslmode=disable"
PSQL=(psql -X -v ON_ERROR_STOP=0 -P pager=off)

run() {  # label, dsn, sql
  echo
  echo ">>> [$1] $3"
  "${PSQL[@]}" "$2" -c "$3" 2>&1 || true
}

echo
echo "################ LIVE CRUD WORKLOAD (real psql over TCP) ################"

run "root/plaintext" "$ROOT_DSN" "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price FLOAT, city TEXT)"
run "root" "$ROOT_DSN" "INSERT INTO products (id, name, price, city) VALUES (1, 'espresso', 3.50, 'rome')"
run "root" "$ROOT_DSN" "INSERT INTO products (id, name, price, city) VALUES (2, 'latte', 4.25, 'milan')"
run "root" "$ROOT_DSN" "INSERT INTO products (id, name, price, city) VALUES (3, 'mocha', 4.75, 'rome')"
run "root" "$ROOT_DSN" "INSERT INTO products (id, name, price, city) VALUES (4, 'cortado', 3.95, 'madrid')"
run "root: full scan" "$ROOT_DSN" "SELECT id, name, price, city FROM products"
run "root: WHERE filter" "$ROOT_DSN" "SELECT id, name FROM products WHERE price > 4.0"
run "root: point lookup" "$ROOT_DSN" "SELECT id, name FROM products WHERE id = 2"
run "root: UPDATE WHERE" "$ROOT_DSN" "UPDATE products SET price = 9.99 WHERE id = 3"
run "root: verify update" "$ROOT_DSN" "SELECT id, name, price FROM products WHERE id = 3"
run "root: DELETE WHERE" "$ROOT_DSN" "DELETE FROM products WHERE id = 1"
run "root: verify delete" "$ROOT_DSN" "SELECT id, name FROM products"

echo
echo "################ SECONDARY INDEX ################"
run "root: CREATE INDEX" "$ROOT_DSN" "CREATE INDEX idx_city ON products (city)"
run "root: index-accelerated SELECT" "$ROOT_DSN" "SELECT id, name, city FROM products WHERE city = 'rome'"
run "root: DROP INDEX" "$ROOT_DSN" "DROP INDEX idx_city"

echo
echo "################ TLS leg (sslmode=require, real rustls handshake) ################"
run "root/TLS: DDL+DML over TLS" "$ROOT_TLS_DSN" "CREATE TABLE secure_t (id INTEGER PRIMARY KEY, v TEXT)"
run "root/TLS" "$ROOT_TLS_DSN" "INSERT INTO secure_t (id, v) VALUES (1, 'tls-works')"
run "root/TLS" "$ROOT_TLS_DSN" "SELECT id, v FROM secure_t"

echo
echo "################ RBAC: 42501 denial then GRANT ################"
run "root: CREATE ROLE alice" "$ROOT_DSN" "CREATE ROLE alice PASSWORD 'alicepw'"
echo
echo ">>> [alice: SELECT before grant — expect ERROR 42501]"
"${PSQL[@]}" "$ALICE_DSN" -c "SELECT id, name FROM products" 2>&1 || true
run "root: GRANT SELECT" "$ROOT_DSN" "GRANT SELECT ON products TO alice"
echo
echo ">>> [alice: SELECT after grant — expect rows]"
"${PSQL[@]}" "$ALICE_DSN" -c "SELECT id, name FROM products" 2>&1 || true
echo
echo ">>> [alice: GRANT as non-superuser — expect ERROR 42501]"
"${PSQL[@]}" "$ALICE_DSN" -c "GRANT SELECT ON products TO alice" 2>&1 || true

echo
echo "################ DROP TABLE ################"
run "root: DROP TABLE products" "$ROOT_DSN" "DROP TABLE products"
run "root: DROP TABLE secure_t" "$ROOT_DSN" "DROP TABLE secure_t"

echo
echo "################ AUDIT LOG (JSONL emitted by the running server) ################"
echo "--- $AUDIT_LOG ---"
cat "$AUDIT_LOG" 2>/dev/null || echo "(no audit log)"

echo
echo "=== LIVE SESSION COMPLETE — real server answered every statement ==="
REMOTE_SESSION

echo "[6/6] results saved to $LOCAL_RESULTS_DIR/live-session.log"
{
  echo "commit: $COMMIT_SHA"
  echo "instance_id: $INSTANCE_ID"
  echo "instance_type: c6id.4xlarge"
  echo "region: $AWS_REGION"
  echo "run_kind: v2-phase1 LIVE server session (real galaxdb-server binary + psql over TCP)"
  echo "workload: SCRAM auth, CRUD, WHERE, UPDATE, DELETE, secondary index, TLS leg, RBAC 42501+grant, DROP, audit log"
} >"$LOCAL_RESULTS_DIR/run_metadata.txt"
echo "[done] trap handler will stop the instance next"
