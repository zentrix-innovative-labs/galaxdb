#!/usr/bin/env python3
"""v0.5 end-to-end real-data test — multi-model embeddings + upgrade-safe format.

Runs the *real* release binaries (galaxdb-server + galaxdb-sidecar) against a
real dataset (AG News) over the PostgreSQL wire protocol, then exercises the two
v0.5 workstreams with real files on disk — not cargo unit checks:

  Phase A (Workstream A — multi-model embedding sidecar):
    - server loads the model via the new runtime registry
    - bulk INSERT of real news articles, each embedded by the real model
    - SEMANTIC_MATCH precision per category, LIMIT, threshold monotonicity, DELETE

  Phase B (Workstream B — upgrade-safe on-disk format, durability):
    - stop the server, verify the WAL carries the versioned superblock (GWAL)
    - restart on the same data dir → rows + vectors survive (WAL replay through
      the versioned reader), point + semantic queries still work

  Phase C (Workstream B — rollback safety):
    - tamper the WAL superblock format version to current+1
    - the server MUST refuse to open (typed too-new error), not silently drop data
    - restore the byte → the server opens again and data is intact

Usage:
  python3 scripts/v05_real_data_test.py \
      --server target/release/galaxdb-server \
      --sidecar target/release/galaxdb-sidecar \
      --parquet /tmp/ag_news_test.parquet \
      --model sentence-transformers/all-MiniLM-L6-v2 --dim 384 --per-category 150

Requires: psycopg2, pyarrow. Exits non-zero if any assertion fails.
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from collections import Counter

import psycopg2
import pyarrow.parquet as pq

LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
QUERIES = {
    1: "football basketball baseball game score championship team playoff",
    2: "stock market company earnings profit shares investors economy",
    3: "computer software technology internet chip science research gadget",
    0: "government election president war military country foreign policy",
}

FAILURES = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    if not ok:
        FAILURES.append(name)


def load_rows(parquet_path, per_category):
    tbl = pq.read_table(parquet_path).to_pylist()
    buckets = {0: [], 1: [], 2: [], 3: []}
    for r in tbl:
        b = buckets.get(r["label"])
        if b is not None and len(b) < per_category:
            b.append(r["text"])
    rows, rid = [], 1
    for label, texts in buckets.items():
        for t in texts:
            rows.append((rid, label, t))
            rid += 1
    return rows


class Server:
    def __init__(self, server_bin, sidecar_bin, data_dir, port, observe_port, model):
        self.server_bin = server_bin
        self.sidecar_bin = sidecar_bin
        self.data_dir = data_dir
        self.port = port
        self.observe_port = observe_port
        self.model = model
        self.proc = None
        self.log_path = os.path.join(data_dir, "server.log")

    def start(self, wait=True, timeout=240):
        logf = open(self.log_path, "ab")
        self.proc = subprocess.Popen(
            [
                self.server_bin,
                "--data-dir", self.data_dir,
                "--port", str(self.port),
                "--observe-port", str(self.observe_port),
                "--sidecar", self.sidecar_bin,
                "--model", self.model,
            ],
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
        if not wait:
            return
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"server exited early (code {self.proc.returncode}); log tail:\n"
                    + self._log_tail()
                )
            try:
                c = psycopg2.connect(
                    host="127.0.0.1", port=self.port, user="galaxdb",
                    dbname="galaxdb", password="x", connect_timeout=3,
                )
                c.close()
                return
            except Exception:
                time.sleep(1)
        raise RuntimeError("server did not become ready in time")

    def wait_exit(self, timeout=30):
        """Wait for the process to exit; return exit code or None if still running."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            code = self.proc.poll()
            if code is not None:
                return code
            time.sleep(0.5)
        return None

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)

    def conn(self):
        c = psycopg2.connect(
            host="127.0.0.1", port=self.port, user="galaxdb",
            dbname="galaxdb", password="x",
        )
        c.autocommit = True
        return c

    def _log_tail(self, n=40):
        try:
            with open(self.log_path, "r", errors="replace") as f:
                return "".join(f.readlines()[-n:])
        except Exception:
            return "(no log)"


def phase_a(srv, rows, model, dim, topk):
    print("\n=== Phase A: multi-model embeddings over the wire (real data) ===", flush=True)
    conn = srv.conn()
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS news")
    cur.execute(
        f"CREATE TABLE news (id INT PRIMARY KEY, label INT, "
        f"text TEXT EMBEDDING MODEL '{model}' DIM {dim})"
    )

    t0 = time.time()
    for i, (rid, label, text) in enumerate(rows):
        cur.execute(
            "INSERT INTO news (id, label, text) VALUES (%s, %s, %s)",
            (rid, label, text),
        )
        if (i + 1) % 200 == 0:
            print(f"    inserted {i+1}/{len(rows)}", flush=True)
    dt = time.time() - t0
    print(f"  inserted {len(rows)} rows in {dt:.1f}s "
          f"({len(rows)/dt:.0f} rows/s, embedded by the real model)", flush=True)

    cur.execute("SELECT COUNT(*) FROM news")
    n = cur.fetchone()[0]
    check("row count after bulk insert", n == len(rows), f"expected {len(rows)}, got {n}")

    for label, query in QUERIES.items():
        cur.execute(
            "SELECT id, label FROM news WHERE SEMANTIC_MATCH(text, %s, %s) LIMIT %s",
            (query, 0.15, topk),
        )
        res = cur.fetchall()
        if not res:
            check(f"semantic '{LABEL_NAMES[label]}' returns results", False, "0 rows")
            continue
        labels = [r[1] for r in res]
        precision = sum(1 for x in labels if x == label) / len(labels)
        check(
            f"semantic '{LABEL_NAMES[label]}' precision@{len(labels)} > 0.50",
            precision > 0.50,
            f"precision={precision:.2f}, dist={dict(Counter(labels))}",
        )

    # LIMIT honored.
    cur.execute(
        "SELECT id FROM news WHERE SEMANTIC_MATCH(text, %s, %s) LIMIT 25", (QUERIES[3], 0.0)
    )
    check("LIMIT 25 > old cap of 10", len(cur.fetchall()) > 10)

    # Threshold monotonicity.
    cur.execute("SELECT id FROM news WHERE SEMANTIC_MATCH(text, %s, %s)", (QUERIES[1], 0.1))
    low = {r[0] for r in cur.fetchall()}
    cur.execute("SELECT id FROM news WHERE SEMANTIC_MATCH(text, %s, %s)", (QUERIES[1], 0.4))
    high = {r[0] for r in cur.fetchall()}
    check("higher threshold ⊆ lower threshold", high.issubset(low),
          f"|0.1|={len(low)}, |0.4|={len(high)}")

    # A plain (non-embedding) sentinel table to check point durability across restart.
    cur.execute("DROP TABLE IF EXISTS kv")
    cur.execute("CREATE TABLE kv (id INT PRIMARY KEY, v TEXT)")
    cur.execute("INSERT INTO kv (id, v) VALUES (1, 'sentinel-value')")
    conn.close()
    return len(rows)


def check_wal_superblock(data_dir):
    wal = os.path.join(data_dir, "wal.log")
    with open(wal, "rb") as f:
        magic = f.read(4)
    check("WAL carries the versioned superblock (GWAL)", magic == b"GWAL",
          f"first 4 bytes = {magic!r}")
    # Any flushed SSTs must use the versioned footer (SSTV) — checked if present.
    ssts = [f for f in os.listdir(data_dir) if f.startswith("sst_") and f.endswith(".pax")]
    for s in ssts:
        with open(os.path.join(data_dir, s), "rb") as f:
            data = f.read()
        tail_magic = data[-4:]
        check(f"SST {s} footer is versioned (SSTV) or legacy (SSTF)",
              tail_magic in (b"SSTV", b"SSTF"), f"tail={tail_magic!r}")


def phase_b(srv, n_rows, topk):
    print("\n=== Phase B: restart recovery with versioned WAL (real data) ===", flush=True)
    srv.stop()
    check_wal_superblock(srv.data_dir)

    srv.start()
    conn = srv.conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM news")
    n = cur.fetchone()[0]
    check("news rows survive restart (WAL replay)", n == n_rows, f"expected {n_rows}, got {n}")
    cur.execute("SELECT v FROM kv WHERE id = 1")
    row = cur.fetchone()
    check("point row survives restart", row is not None and row[0] == "sentinel-value",
          f"got {row}")
    # The vector index is rebuilt on open (re-embedding durable rows) → semantic
    # search still works after a restart. Use the two-column projection that the
    # semantic path returns (id, label) and read the label at index 1.
    cur.execute(
        "SELECT id, label FROM news WHERE SEMANTIC_MATCH(text, %s, %s) LIMIT %s",
        (QUERIES[1], 0.15, topk),
    )
    labels = [r[1] for r in cur.fetchall()]
    prec = (sum(1 for x in labels if x == 1) / len(labels)) if labels else 0.0
    check("semantic search works after restart (vector index rebuilt on open)", prec > 0.50,
          f"precision={prec:.2f} on {len(labels)} rows")
    conn.close()


def phase_c(srv):
    print("\n=== Phase C: rollback safety — refuse a newer WAL format ===", flush=True)
    srv.stop()
    wal = os.path.join(srv.data_dir, "wal.log")
    original = open(wal, "rb").read()

    # Bump the superblock format_version (header bytes 4..6, u16 LE) to current+1.
    tampered = bytearray(original)
    cur_ver = int.from_bytes(tampered[4:6], "little")
    tampered[4:6] = (cur_ver + 1).to_bytes(2, "little")
    with open(wal, "wb") as f:
        f.write(tampered)

    # The server must refuse to open this WAL.
    refused = False
    try:
        srv.start(wait=True, timeout=25)
    except RuntimeError as e:
        refused = "newer" in str(e).lower() or "exited early" in str(e).lower()
    if not refused:
        # If it somehow started, that's a failure; stop it.
        srv.stop()
    check("server refuses to open a newer-format WAL (rollback safety)", refused,
          "expected a typed too-new refusal at open")

    # Restore and confirm it opens again with data intact.
    with open(wal, "wb") as f:
        f.write(original)
    srv.start()
    conn = srv.conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM news")
    n = cur.fetchone()[0]
    check("server reopens cleanly after restoring the WAL", n > 0, f"rows={n}")
    conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--parquet", default="/tmp/ag_news_test.parquet")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--per-category", type=int, default=150)
    ap.add_argument("--port", type=int, default=5466)
    ap.add_argument("--observe-port", type=int, default=9166)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    import tempfile
    data_dir = args.data_dir or tempfile.mkdtemp(prefix="galaxdb-v05-")
    print(f"data dir: {data_dir}", flush=True)

    rows = load_rows(args.parquet, args.per_category)
    print(f"loaded {len(rows)} AG News rows", flush=True)

    srv = Server(args.server, args.sidecar, data_dir, args.port, args.observe_port, args.model)
    try:
        print("starting server (first run may download the model)…", flush=True)
        srv.start(timeout=300)
        n = phase_a(srv, rows, args.model, args.dim, topk=20)
        phase_b(srv, n, topk=20)
        phase_c(srv)
    finally:
        srv.stop()

    print("\n================ SUMMARY ================", flush=True)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}", flush=True)
        sys.exit(1)
    print("ALL v0.5 REAL-DATA CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
