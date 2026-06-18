#!/usr/bin/env bash
# PostgreSQL 16 vs GalaxDB head-to-head single-row INSERT benchmark
# Same hardware, same NVMe, same durability: synchronous_commit=on, fdatasync.
# Run on AWS c6id.4xlarge. Results written to /mnt/nvme/bench_comparison.json
set -euo pipefail

PGDATA=/mnt/nvme/pgdata
GALAXDB_DIR=/mnt/nvme/galaxdb
RESULTS=/mnt/nvme/bench_comparison.json

# ── 1. Mount NVMe (skip if already mounted) ───────────────────────────────────
if mountpoint -q /mnt/nvme; then
  echo "NVMe already mounted — skipping"
else
  NVME=""
  for dev in /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1; do
    if [ -b "$dev" ]; then
      MODEL=$(sudo nvme id-ctrl -o json "$dev" 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('mn',''))" 2>/dev/null || true)
      if echo "$MODEL" | grep -qi "Instance Storage"; then
        NVME="$dev"; break
      fi
    fi
  done
  if [ -z "$NVME" ]; then
    NVME=$(lsblk -dn -o NAME,SIZE,TYPE | grep disk | grep nvme | grep -v nvme0 | head -1 | awk '{print "/dev/"$1}')
  fi
  echo "Formatting + mounting $NVME"
  sudo mkfs.xfs -f "$NVME"
  sudo mkdir -p /mnt/nvme
  sudo mount -o noatime "$NVME" /mnt/nvme
  sudo chown ubuntu:ubuntu /mnt/nvme
fi
mkdir -p /mnt/nvme/galaxdb
echo "NVMe ready"

# ── 2. Install PostgreSQL 16 (skip if present) ───────────────────────────────
if ! command -v pgbench > /dev/null 2>&1; then
  echo "Installing PostgreSQL 16..."
  sudo apt-get update -q
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
      postgresql-16 postgresql-client-16
fi
echo "PostgreSQL $(psql --version 2>/dev/null | head -1) available"

# ── 3. Init cluster on NVMe (skip if exists) ──────────────────────────────────
sudo systemctl stop postgresql 2>/dev/null || true
sudo pkill -u postgres postgres 2>/dev/null || true
sleep 2

if [ ! -f "$PGDATA/PG_VERSION" ]; then
  echo "Initialising PG cluster on NVMe..."
  sudo mkdir -p "$PGDATA"
  sudo chown postgres:postgres "$PGDATA"
  sudo -u postgres /usr/lib/postgresql/16/bin/initdb -D "$PGDATA"
  printf '\nsynchronous_commit = on\nwal_sync_method = fdatasync\ncheckpoint_timeout = 1h\nmax_wal_size = 4GB\nshared_buffers = 1GB\n' \
      | sudo -u postgres tee -a "$PGDATA/postgresql.conf" > /dev/null
  echo "Cluster initialised"
else
  echo "Cluster exists — reusing"
fi

sudo -u postgres /usr/lib/postgresql/16/bin/pg_ctl -D "$PGDATA" -l /tmp/pg.log start
sleep 3
sudo -u postgres createdb bench 2>/dev/null || true
echo "PostgreSQL started"

# ── 4. pgbench TPC-B (mixed read/write, 1 client, 60s) ───────────────────────
echo "=== Initialising pgbench schema (scale=10) ==="
sudo -u postgres pgbench -i -s 10 -q bench

echo "=== pgbench TPC-B: 1 client, 60s, synchronous_commit=on ==="
PG_TPCB=$(sudo -u postgres pgbench -c 1 -j 1 -T 60 -M prepared bench 2>&1)
echo "$PG_TPCB"
PG_TPS=$(echo "$PG_TPCB" | grep "^tps" | tail -1 | awk '{print $3}' | tr -d '()')

# ── 5. pgbench INSERT-only (1 client, 2000 transactions) ─────────────────────
echo "=== pgbench INSERT-only: 1 client, 2000 rows, synchronous_commit=on ==="
sudo -u postgres psql bench -q -c "DROP TABLE IF EXISTS ins_bench; CREATE TABLE ins_bench (id BIGSERIAL PRIMARY KEY, val TEXT);"
cat > /tmp/pg_insert.sql << 'PGSCRIPT'
INSERT INTO ins_bench (val) VALUES ('x');
PGSCRIPT
PG_INS=$(sudo -u postgres pgbench -c 1 -j 1 -t 2000 -M prepared -f /tmp/pg_insert.sql bench 2>&1)
echo "$PG_INS"
PG_INS_TPS=$(echo "$PG_INS" | grep "^tps" | tail -1 | awk '{print $3}' | tr -d '()')

# ── 6. Build GalaxDB --release ────────────────────────────────────────────────
source "$HOME/.cargo/env" 2>/dev/null || true
echo "=== Building GalaxDB benchmarks --release ==="
cd "$GALAXDB_DIR"
cargo build --release -p galaxdb-benchmarks --bin single-row-insert-bench 2>&1 | tail -4
echo "BUILD_DONE"

# ── 7. GalaxDB single-row INSERT ─────────────────────────────────────────────
echo "=== GalaxDB single-row INSERT: 1 client, 1000 rows, synchronous ==="
GALAXDB_OUT=$(./target/release/single-row-insert-bench --rows 1000 2>&1)
echo "$GALAXDB_OUT"
GALAXDB_RPS=$(echo "$GALAXDB_OUT" | grep "simple" | grep "rows/sec" | awk '{print $(NF-1)}')

# ── 8. Write results JSON ─────────────────────────────────────────────────────
python3 - << PYEOF
import json, datetime
results = {
    "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "hardware": "AWS c6id.4xlarge (Intel Ice Lake 16 vCPU 32GiB instance-store NVMe XFS noatime)",
    "conditions": "synchronous_commit=on, wal_sync_method=fdatasync, 1 client, serial",
    "postgresql_16": {
        "pgbench_tpcb_tps_60s": "$PG_TPS",
        "insert_only_tps_2000rows": "$PG_INS_TPS"
    },
    "galaxdb_v02": {
        "single_row_insert_rps_1000rows": "$GALAXDB_RPS"
    }
}
print(json.dumps(results, indent=2))
with open("$RESULTS", "w") as f:
    json.dump(results, f, indent=2)
PYEOF

echo "=== RESULTS SAVED: $RESULTS ==="
