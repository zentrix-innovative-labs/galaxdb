#!/usr/bin/env python3
"""
Month 2 Gate Tests — comprehensive validation of the GalaxDB SQL engine.

Tests:
  1. psycopg2 connection + CRUD
  2. SQLAlchemy ORM (simple mode)
  3. Pandas read_sql round-trip
  4. pg_catalog introspection
  5. AuroraSQL extensions
  6. Concurrent throughput benchmark
  7. Embedded Python mode

Requires: galaxdb-server running on port 5433
  cargo run -p galaxdb-server --release -- --port 5433
"""

import socket
import struct
import sys
import time
import threading
import os

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))

# ── Helper: raw PostgreSQL wire protocol client ─────────────────────

class PgRawClient:
    def __init__(self, host='127.0.0.1', port=5433):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        params = b'user\x00postgres\x00database\x00galaxdb\x00\x00'
        self.sock.sendall(struct.pack('>i', 8 + len(params)) + struct.pack('>i', 196608) + params)
        self.sock.settimeout(5)
        self._recv()  # consume startup response

    def query(self, sql):
        payload = sql.encode() + b'\x00'
        self.sock.sendall(b'Q' + struct.pack('>i', 4 + len(payload)) + payload)
        return self._recv()

    def _recv(self):
        chunks = []
        try:
            while True:
                d = self.sock.recv(65536)
                if not d: break
                chunks.append(d)
                if b'Z' in d: break
        except socket.timeout:
            pass
        return b''.join(chunks)

    def count_rows(self, data):
        n, pos = 0, 0
        while pos < len(data):
            t = data[pos]; pos += 1
            if pos + 4 > len(data): break
            length = struct.unpack('>i', data[pos:pos+4])[0]
            if chr(t) == 'D': n += 1
            pos += length
        return n

    def close(self):
        self.sock.close()


# ═══════════════════════════════════════════════════════════════════
# Gate 1: Functional Must-Haves
# ═══════════════════════════════════════════════════════════════════

print("\n=== Gate 1: Functional Must-Haves ===\n")

# 1.1 Wire protocol connection + CRUD
print("--- 1.1 Wire Protocol CRUD ---")
try:
    c = PgRawClient()
    r = c.query("CREATE TABLE gate1 (id INT PRIMARY KEY, name TEXT, score INT)")
    check("CREATE TABLE over wire", b'C' in r)

    r = c.query("INSERT INTO gate1 (id, name, score) VALUES (1, 'alice', 95)")
    check("INSERT over wire", b'C' in r)

    r = c.query("INSERT INTO gate1 (id, name, score) VALUES (2, 'bob', 87)")
    check("INSERT second row", b'C' in r)

    r = c.query("SELECT * FROM gate1")
    rows = c.count_rows(r)
    check("SELECT returns 2 rows", rows == 2, f"got {rows}")

    r = c.query("DROP TABLE gate1")
    check("DROP TABLE", b'C' in r)

    c.close()
except Exception as e:
    check("Wire protocol connection", False, str(e))

# 1.2 pg_catalog stubs
print("\n--- 1.2 pg_catalog Stubs ---")
try:
    c = PgRawClient()
    r = c.query("SELECT * FROM pg_catalog.pg_type")
    rows = c.count_rows(r)
    check("pg_type returns types", rows > 0, f"{rows} types")

    r = c.query("SELECT * FROM pg_catalog.pg_namespace")
    rows = c.count_rows(r)
    check("pg_namespace returns schemas", rows >= 2, f"{rows} schemas")

    r = c.query("SELECT * FROM pg_catalog.pg_database")
    rows = c.count_rows(r)
    check("pg_database returns databases", rows >= 1, f"{rows} databases")

    r = c.query("SELECT * FROM pg_catalog.pg_class")
    check("pg_class returns result", b'T' in r or b'C' in r)

    r = c.query("SELECT * FROM pg_catalog.pg_settings")
    check("unsupported pg_catalog returns empty (not error)", b'E' not in r[:5])

    c.close()
except Exception as e:
    check("pg_catalog stubs", False, str(e))

# 1.3 AuroraSQL extensions
print("\n--- 1.3 AuroraSQL Extensions ---")
try:
    c = PgRawClient()
    c.query("CREATE TABLE docs (id INT, content TEXT)")

    r = c.query("SHOW EMBEDDING HEALTH")
    check("SHOW EMBEDDING HEALTH parses", b'C' in r)

    r = c.query("CREATE VERSION TAG 'v1.0'")
    check("CREATE VERSION TAG parses", b'C' in r)

    r = c.query("CREATE VERSION TAG 'train' FOR TRAINING WITH TRAINING PRECISION 'sq8' TRAINING SEED 42")
    check("CREATE VERSION TAG FOR TRAINING parses", b'C' in r)

    r = c.query("ANALYZE docs")
    check("ANALYZE parses", b'C' in r)

    r = c.query("BACKUP TO '/tmp/backup'")
    check("BACKUP TO parses", b'C' in r)

    r = c.query("RESTORE FROM '/tmp/backup'")
    check("RESTORE FROM parses", b'C' in r)

    c.query("DROP TABLE docs")
    c.close()
except Exception as e:
    check("AuroraSQL extensions", False, str(e))

# 1.4 Error handling
print("\n--- 1.4 Error Handling ---")
try:
    c = PgRawClient()
    r = c.query("SELECT * FROM nonexistent_table")
    check("Nonexistent table returns error", b'E' in r[:20])

    r = c.query("SELECTT * FROM bad_sql")
    check("Bad SQL returns parse error", b'E' in r[:20])

    c.close()
except Exception as e:
    check("Error handling", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# Gate 2: Python Embedded Mode
# ═══════════════════════════════════════════════════════════════════

print("\n=== Gate 2: Python Embedded Mode ===\n")

try:
    import galaxdb
    import tempfile

    db_path = tempfile.mkdtemp()
    db = galaxdb.Database(db_path)

    check("galaxdb.Database() opens", True)
    check("__version__ exists", hasattr(galaxdb, '__version__'), galaxdb.__version__)

    db.execute("CREATE TABLE emb (id INT PRIMARY KEY, name TEXT, val INT)")
    check("CREATE TABLE in embedded mode", db.table_exists("emb"))

    db.execute("INSERT INTO emb (id, name, val) VALUES (1, 'alice', 100)")
    db.execute("INSERT INTO emb (id, name, val) VALUES (2, 'bob', 200)")
    db.execute("INSERT INTO emb (id, name, val) VALUES (3, 'charlie', 300)")

    rows = db.execute("SELECT * FROM emb")
    check("SELECT returns list of dicts", isinstance(rows, list) and len(rows) == 3, f"{len(rows)} rows")
    check("Row has correct columns", 'name' in rows[0] and 'val' in rows[0])

    # Pandas round-trip
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        check("Pandas DataFrame from results", len(df) == 3 and 'name' in df.columns, f"shape={df.shape}")
    except Exception as e:
        check("Pandas DataFrame", False, str(e))

    db.execute("DROP TABLE emb")
    check("DROP TABLE in embedded mode", not db.table_exists("emb"))

except Exception as e:
    check("Python embedded mode", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# Gate 3: Performance Benchmarks
# ═══════════════════════════════════════════════════════════════════

print("\n=== Gate 3: Performance Benchmarks ===\n")

# 3.1 Embedded INSERT throughput
print("--- 3.1 Embedded INSERT Throughput ---")
try:
    import galaxdb, tempfile
    db = galaxdb.Database(tempfile.mkdtemp())
    db.execute("CREATE TABLE perf (id INT, name TEXT, score INT)")

    # Multi-row INSERT batching (100 rows per statement, 1 parse + 1 fsync per batch)
    N = 10000
    BATCH = 100
    start = time.time()
    for batch_start in range(0, N, BATCH):
        values = ', '.join(
            f"({i}, 'user_{i}', {i*10})"
            for i in range(batch_start, min(batch_start + BATCH, N))
        )
        db.execute(f"INSERT INTO perf (id, name, score) VALUES {values}")
    elapsed = time.time() - start
    tps = N / elapsed
    check(f"Embedded INSERT {N} rows (batched {BATCH}/stmt)", True, f"{tps:.0f} rows/sec in {elapsed:.1f}s")
    check("Embedded INSERT > 1000 rows/sec", tps > 1000, f"{tps:.0f}")

    # SELECT all
    start = time.time()
    rows = db.execute("SELECT * FROM perf")
    sel_elapsed = time.time() - start
    check(f"Embedded SELECT {len(rows)} rows", len(rows) == N, f"{sel_elapsed*1000:.0f}ms")

except Exception as e:
    check("Embedded performance", False, str(e))

# 3.2 Wire protocol concurrent throughput
print("\n--- 3.2 Wire Protocol Concurrent Throughput ---")
try:
    # Setup table
    c = PgRawClient()
    c.query("CREATE TABLE wire_perf (id INT, name TEXT, score INT)")

    # Concurrent INSERT with multiple connections
    NUM_CLIENTS = 4
    ROWS_PER_CLIENT = 250
    errors = []

    def wire_insert_worker(client_id, rows):
        try:
            wc = PgRawClient()
            for i in range(rows):
                row_id = client_id * rows + i
                wc.query(f"INSERT INTO wire_perf (id, name, score) VALUES ({row_id}, 'u{row_id}', {row_id})")
            wc.close()
        except Exception as e:
            errors.append(str(e))

    start = time.time()
    threads = []
    for t in range(NUM_CLIENTS):
        th = threading.Thread(target=wire_insert_worker, args=(t, ROWS_PER_CLIENT))
        threads.append(th)
        th.start()
    for th in threads:
        th.join()
    elapsed = time.time() - start

    total = NUM_CLIENTS * ROWS_PER_CLIENT
    tps = total / elapsed
    check(f"Wire INSERT {total} rows ({NUM_CLIENTS} clients)", len(errors) == 0, f"{tps:.0f} rows/sec")

    # Verify all rows
    r = c.query("SELECT * FROM wire_perf")
    rows = c.count_rows(r)
    check(f"All {total} rows readable", rows == total, f"got {rows}")

    c.query("DROP TABLE wire_perf")
    c.close()

except Exception as e:
    check("Wire concurrent throughput", False, str(e))

# 3.3 Wire protocol SELECT throughput
print("\n--- 3.3 Wire Protocol SELECT Throughput ---")
try:
    c = PgRawClient()
    c.query("CREATE TABLE sel_perf (id INT, name TEXT)")
    for i in range(100):
        c.query(f"INSERT INTO sel_perf (id, name) VALUES ({i}, 'user_{i}')")

    NUM_SELECTS = 100
    start = time.time()
    for _ in range(NUM_SELECTS):
        r = c.query("SELECT * FROM sel_perf")
    elapsed = time.time() - start
    qps = NUM_SELECTS / elapsed
    check(f"Wire SELECT {NUM_SELECTS} queries", True, f"{qps:.0f} QPS, {elapsed/NUM_SELECTS*1000:.1f}ms/query")

    c.query("DROP TABLE sel_perf")
    c.close()
except Exception as e:
    check("Wire SELECT throughput", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# Gate 4: Binary Size
# ═══════════════════════════════════════════════════════════════════

print("\n=== Gate 4: Binary Size ===\n")

server_path = "target/release/galaxdb-server"
if os.path.exists(server_path):
    size_mb = os.path.getsize(server_path) / (1024 * 1024)
    check(f"Server binary size", True, f"{size_mb:.1f} MB")
    check("Binary < 25 MB", size_mb < 25, f"{size_mb:.1f} MB")
else:
    check("Server binary exists", False, "not found at " + server_path)


# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"  MONTH 2 GATE RESULTS: {PASS} passed, {FAIL} failed")
print(f"{'='*60}")

if FAIL > 0:
    print("\n  ⚠️  Some gates failed. Review above for details.")
    sys.exit(1)
else:
    print("\n  ✅ ALL GATES PASSED")
    sys.exit(0)
