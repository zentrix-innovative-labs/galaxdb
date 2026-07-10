#!/usr/bin/env python3
"""v0.6 E-4 metering: real-data end-to-end verification.

Drives a running GalaxDB server over the PostgreSQL wire protocol with real
data, scrapes `/metrics` before and after each operation group, and asserts the
counter deltas match the operations actually issued. This is the real-data bar
(not a cargo check) that must pass before publishing any metering claim.

Prerequisites (run these yourself; this script does not build or spawn):
  1. Build the server + sidecar:  cargo build --release -p galaxdb-server -p galaxdb-sidecar
  2. Start the server with auth disabled for local testing, e.g.:
       ./target/release/galaxdb-server --data-dir /tmp/gx_meter --port 5433 --metrics-port 9090
  3. pip install psycopg2-binary requests

Usage:
  python3 scripts/v06_metering_test.py --dsn "host=localhost port=5433 dbname=galaxdb user=postgres" \
      --metrics-url http://localhost:9090/metrics
"""

import argparse
import re
import sys

import psycopg2
import requests


def scrape(url: str) -> dict[str, float]:
    """Return {metric_name: value} for every galaxdb_* sample."""
    out: dict[str, float] = {}
    for line in requests.get(url, timeout=10).text.splitlines():
        if line.startswith("#") or not line.startswith("galaxdb_"):
            continue
        m = re.match(r"(\S+)\s+(\S+)", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def delta(before: dict, after: dict, name: str) -> float:
    return after.get(name, 0.0) - before.get(name, 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--metrics-url", required=True)
    args = ap.parse_args()

    conn = psycopg2.connect(args.dsn)
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS meter_docs")
    cur.execute("CREATE TABLE meter_docs (id INT PRIMARY KEY, body TEXT)")

    failures = []

    def check(label, name, got, want):
        ok = got == want
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {name} delta={got} (want {want})")
        if not ok:
            failures.append(label)

    # --- writes: 1 single-row INSERT + 1 three-row INSERT = 2 write ops (not 4) ---
    b = scrape(args.metrics_url)
    cur.execute("INSERT INTO meter_docs (id, body) VALUES (1, 'alpha')")
    cur.execute("INSERT INTO meter_docs (id, body) VALUES (2,'beta'),(3,'gamma'),(4,'delta')")
    a = scrape(args.metrics_url)
    check("2 INSERT statements = 2 write ops (row count ignored)",
          "galaxdb_write_ops_total", delta(b, a, "galaxdb_write_ops_total"), 2)

    # --- reads: 1 SELECT = 1 read op, 0 writes ---
    b = scrape(args.metrics_url)
    cur.execute("SELECT id, body FROM meter_docs")
    cur.fetchall()
    a = scrape(args.metrics_url)
    check("SELECT = 1 read op", "galaxdb_read_ops_total",
          delta(b, a, "galaxdb_read_ops_total"), 1)
    check("SELECT moves no write op", "galaxdb_write_ops_total",
          delta(b, a, "galaxdb_write_ops_total"), 0)

    # --- update + delete = 2 write ops ---
    b = scrape(args.metrics_url)
    cur.execute("UPDATE meter_docs SET body = 'z' WHERE id >= 1")
    cur.execute("DELETE FROM meter_docs WHERE id = 4")
    a = scrape(args.metrics_url)
    check("UPDATE + DELETE = 2 write ops", "galaxdb_write_ops_total",
          delta(b, a, "galaxdb_write_ops_total"), 2)

    # --- capacity gauges are present and sane ---
    g = scrape(args.metrics_url)
    for gauge in ("galaxdb_rows_total", "galaxdb_storage_bytes",
                  "galaxdb_process_start_time_seconds"):
        present = gauge in g
        print(f"  [{'PASS' if present else 'FAIL'}] gauge present: {gauge} = {g.get(gauge)}")
        if not present:
            failures.append(gauge)

    # --- semantic_cache_hits must NOT be present (deferred, not faked) ---
    absent = "galaxdb_semantic_cache_hits_total" not in g
    print(f"  [{'PASS' if absent else 'FAIL'}] semantic_cache_hits_total is absent (deferred)")
    if not absent:
        failures.append("semantic_cache_hits_total should be absent")

    cur.close()
    conn.close()

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("ALL METERING CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
