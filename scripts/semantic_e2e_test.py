#!/usr/bin/env python3
"""Real-dataset end-to-end test for GalaxDB semantic search over the wire.

Loads a slice of the AG News dataset (real news articles in four categories:
0=World, 1=Sports, 2=Business, 3=Sci/Tech) into a running galaxdb-server via
the PostgreSQL wire protocol, then exercises SEMANTIC_MATCH with real relevance
assertions — not toy 4-row checks.

What it validates:
  1. Bulk INSERT of N rows through the wire path, each embedded by the real
     sidecar model (this is the path that silently stored nothing before the
     on_row_inserted fix).
  2. Semantic relevance: for a category-specific query, the top-k results are
     dominated by that category (precision@k >> random baseline of 0.25).
  3. Threshold monotonicity: a higher threshold returns a subset of a lower one.
  4. DELETE removes rows from future semantic results (delta tombstone path).
  5. SHOW EMBEDDING HEALTH FOR <table> reports real sidecar state.

Usage:
  python3 scripts/semantic_e2e_test.py \
      --parquet /tmp/ag_news_test.parquet \
      --host localhost --port 5455 \
      --per-category 200 --model sentence-transformers/all-MiniLM-L6-v2 --dim 384

Requires: psycopg2, pyarrow. Exits non-zero if any assertion fails.
"""
import argparse
import sys
import time
from collections import Counter

import psycopg2
import pyarrow.parquet as pq

LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

# Category-specific queries. Each should retrieve mostly its own category.
QUERIES = {
    1: "football basketball baseball game score championship team playoff",
    2: "stock market company earnings profit shares investors economy",
    3: "computer software technology internet chip science research gadget",
    0: "government election president war military country foreign policy",
}


def connect(host, port):
    conn = psycopg2.connect(
        host=host, port=port, user="galaxdb", dbname="galaxdb", password="x"
    )
    conn.autocommit = True
    return conn


def load_rows(parquet_path, per_category):
    tbl = pq.read_table(parquet_path).to_pylist()
    buckets = {0: [], 1: [], 2: [], 3: []}
    for r in tbl:
        b = buckets.get(r["label"])
        if b is not None and len(b) < per_category:
            b.append(r["text"])
    rows = []
    rid = 1
    for label, texts in buckets.items():
        for t in texts:
            rows.append((rid, label, t))
            rid += 1
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5455)
    ap.add_argument("--per-category", type=int, default=200)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    rows = load_rows(args.parquet, args.per_category)
    print(f"Loaded {len(rows)} rows from {args.parquet} "
          f"({args.per_category}/category)")

    conn = connect(args.host, args.port)
    cur = conn.cursor()

    failures = []

    def check(name, ok, detail=""):
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    # --- Schema ---
    cur.execute("DROP TABLE IF EXISTS news")
    cur.execute(
        f"CREATE TABLE news (id INT PRIMARY KEY, label INT, "
        f"text TEXT EMBEDDING MODEL '{args.model}' DIM {args.dim})"
    )
    print("Created table 'news' with embedding column.")

    # --- Bulk insert (real embedding per row) ---
    t0 = time.time()
    for i, (rid, label, text) in enumerate(rows):
        # Parameterized to avoid quoting issues in real article text.
        cur.execute(
            "INSERT INTO news (id, label, text) VALUES (%s, %s, %s)",
            (rid, label, text),
        )
        if (i + 1) % 200 == 0:
            print(f"    inserted {i + 1}/{len(rows)} "
                  f"({(i + 1) / (time.time() - t0):.0f} rows/s)")
    dt = time.time() - t0
    print(f"Inserted {len(rows)} rows in {dt:.1f}s "
          f"({len(rows) / dt:.0f} rows/s, each embedded by the sidecar).")

    # --- Sanity: all rows present ---
    cur.execute("SELECT COUNT(*) FROM news")
    n = cur.fetchone()[0]
    check("row count after bulk insert", n == len(rows),
          f"expected {len(rows)}, got {n}")

    # --- Semantic relevance per category ---
    for label, query in QUERIES.items():
        cur.execute(
            "SELECT id, label, text FROM news "
            "WHERE SEMANTIC_MATCH(text, %s, %s)",
            (query, 0.15),
        )
        res = cur.fetchall()
        topk = res[: args.topk]
        if not topk:
            check(f"semantic '{LABEL_NAMES[label]}' returns results", False,
                  "0 rows")
            continue
        labels = [r[1] for r in topk]
        hit = sum(1 for l in labels if l == label)
        precision = hit / len(topk)
        dist = Counter(labels)
        # Random baseline is 0.25; require a clear signal.
        check(
            f"semantic '{LABEL_NAMES[label]}' precision@{len(topk)} > 0.50",
            precision > 0.50,
            f"precision={precision:.2f}, dist={dict(dist)}",
        )

    # --- LIMIT is honored (0.4 fix: was silently capped at 10) ---
    q = QUERIES[3]
    cur.execute(
        "SELECT id FROM news WHERE SEMANTIC_MATCH(text, %s, %s) LIMIT 25",
        (q, 0.0),
    )
    n25 = len(cur.fetchall())
    check("SEMANTIC_MATCH LIMIT 25 returns more than the old cap of 10",
          n25 > 10,
          f"got {n25} rows (was capped at 10 before the fix)")
    cur.execute(
        "SELECT id FROM news WHERE SEMANTIC_MATCH(text, %s, %s) LIMIT 5",
        (q, 0.0),
    )
    n5 = len(cur.fetchall())
    check("SEMANTIC_MATCH LIMIT 5 returns at most 5", n5 <= 5, f"got {n5} rows")

    # --- Threshold monotonicity ---
    q = QUERIES[1]
    cur.execute("SELECT id FROM news WHERE SEMANTIC_MATCH(text, %s, %s)", (q, 0.1))
    low = {r[0] for r in cur.fetchall()}
    cur.execute("SELECT id FROM news WHERE SEMANTIC_MATCH(text, %s, %s)", (q, 0.4))
    high = {r[0] for r in cur.fetchall()}
    check("higher threshold is a subset of lower threshold",
          high.issubset(low),
          f"|low@0.1|={len(low)}, |high@0.4|={len(high)}")

    # --- DELETE removes from semantic results (tombstone path) ---
    cur.execute(
        "SELECT id FROM news WHERE SEMANTIC_MATCH(text, %s, %s)",
        (QUERIES[2], 0.2),
    )
    before = [r[0] for r in cur.fetchall()]
    if before:
        victim = before[0]
        cur.execute("DELETE FROM news WHERE id = %s", (victim,))
        cur.execute(
            "SELECT id FROM news WHERE SEMANTIC_MATCH(text, %s, %s)",
            (QUERIES[2], 0.2),
        )
        after = {r[0] for r in cur.fetchall()}
        check("DELETE removes row from semantic results",
              victim not in after,
              f"deleted id={victim}")
    else:
        check("DELETE test had a candidate row", False, "no rows to delete")

    # --- BULK COPY path (exec_bulk_insert embedding population) ---
    # COPY routes through bulk_insert_with_session -> exec_bulk_insert, a
    # DIFFERENT code path than single-row INSERT. It had the identical
    # discard-the-embedding bug. Load a second table via COPY and confirm
    # semantic search over the bulk-loaded rows works.
    import io
    cur.execute("DROP TABLE IF EXISTS news_copy")
    cur.execute(
        f"CREATE TABLE news_copy (id INT PRIMARY KEY, label INT, "
        f"text TEXT EMBEDDING MODEL '{args.model}' DIM {args.dim})"
    )
    copy_rows = rows[: min(120, len(rows))]
    buf = io.StringIO()
    for rid, label, text in copy_rows:
        safe = text.replace("\\", "\\\\").replace("\t", " ").replace("\n", " ")
        buf.write(f"{rid}\t{label}\t{safe}\n")
    buf.seek(0)
    t_copy = time.time()
    cur.copy_expert("COPY news_copy (id, label, text) FROM STDIN", buf)
    print(f"COPY-loaded {len(copy_rows)} rows in {time.time() - t_copy:.1f}s.")
    cur.execute("SELECT COUNT(*) FROM news_copy")
    nc = cur.fetchone()[0]
    check("COPY bulk insert row count", nc == len(copy_rows),
          f"expected {len(copy_rows)}, got {nc}")
    cur.execute(
        "SELECT id, label FROM news_copy WHERE SEMANTIC_MATCH(text, %s, %s)",
        (QUERIES[1], 0.15),
    )
    copy_res = cur.fetchall()
    check("semantic search over COPY-loaded rows returns results",
          len(copy_res) > 0,
          f"got {len(copy_res)} rows (was 0 before the bulk-insert fix)")

    # --- SHOW EMBEDDING HEALTH reports real state ---
    try:
        cur.execute("SHOW EMBEDDING HEALTH FOR news")
        health = cur.fetchall()
        # Real impl returns (table, sidecar_state, model_version); the old
        # stub returned a single canned "status" string.
        ok = bool(health) and len(health[0]) >= 3
        check("SHOW EMBEDDING HEALTH returns real columns", ok,
              f"row={health[0] if health else None}")
    except Exception as e:  # noqa: BLE001
        check("SHOW EMBEDDING HEALTH executes", False, str(e))

    cur.close()
    conn.close()

    print()
    if failures:
        print(f"FAILED ({len(failures)}): {', '.join(failures)}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
