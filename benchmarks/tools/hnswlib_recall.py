#!/usr/bin/env python3
"""HNSWLib baseline recall/build-speed check.

Usage example:
  python benchmarks/tools/hnswlib_recall.py --num-elements 100000 --num-queries 100

Notes:
- Uses cosine space with normalized vectors to match GalaxDB HNSW.
- For large scales, ground-truth brute force can be slow.
"""

from __future__ import annotations

import argparse
import time
from typing import Tuple

import numpy as np

try:
    import hnswlib
except ImportError as exc:
    raise SystemExit(
        "hnswlib is required. Install with: pip install hnswlib"
    ) from exc


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def build_index(
    data: np.ndarray,
    m: int,
    ef_construction: int,
    num_threads: int,
) -> Tuple["hnswlib.Index", float]:
    index = hnswlib.Index(space="cosine", dim=data.shape[1])
    index.init_index(max_elements=data.shape[0], ef_construction=ef_construction, M=m)
    index.set_num_threads(num_threads)

    start = time.time()
    index.add_items(data)
    build_secs = time.time() - start
    return index, build_secs


def brute_force_topk(data: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    # Cosine similarity is dot product for normalized vectors.
    sims = data @ queries.T
    # argsort descending over axis 0 (queries), take top-k
    topk = np.argpartition(-sims, kth=k - 1, axis=0)[:k, :]
    # Sort top-k for stable comparison
    topk_sorted = np.take_along_axis(
        topk,
        np.argsort(-np.take_along_axis(sims, topk, axis=0), axis=0),
        axis=0,
    )
    return topk_sorted.T  # shape: (num_queries, k)


def recall_at_k(hnsw_labels: np.ndarray, true_labels: np.ndarray) -> float:
    correct = 0
    k = hnsw_labels.shape[1]
    for i in range(hnsw_labels.shape[0]):
        correct += len(set(hnsw_labels[i]).intersection(set(true_labels[i])))
    return correct / float(hnsw_labels.shape[0] * k)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-elements", type=int, default=100_000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    data = rng.uniform(-1.0, 1.0, size=(args.num_elements, args.dim)).astype(np.float32)
    data = normalize_rows(data)

    queries = rng.uniform(-1.0, 1.0, size=(args.num_queries, args.dim)).astype(np.float32)
    queries = normalize_rows(queries)

    num_threads = args.num_threads if args.num_threads > 0 else 0

    index, build_secs = build_index(data, args.m, args.ef_construction, num_threads)
    index.set_ef(args.ef_search)

    labels, _distances = index.knn_query(queries, k=args.k)
    true_labels = brute_force_topk(data, queries, args.k)

    recall = recall_at_k(labels, true_labels)
    build_rate = args.num_elements / build_secs

    print("{\n  \"hnswlib_baseline\": {")
    print(f"    \"num_elements\": {args.num_elements},")
    print(f"    \"dim\": {args.dim},")
    print(f"    \"k\": {args.k},")
    print(f"    \"m\": {args.m},")
    print(f"    \"ef_construction\": {args.ef_construction},")
    print(f"    \"ef_search\": {args.ef_search},")
    print(f"    \"build_time_secs\": {build_secs:.2f},")
    print(f"    \"build_rate_vec_per_sec\": {build_rate:.0f},")
    print(f"    \"recall_at_{args.k}\": {recall:.4f}")
    print("  }\n}")


if __name__ == "__main__":
    main()
