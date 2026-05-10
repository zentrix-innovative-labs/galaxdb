#!/usr/bin/env python3
"""Minimal client that speaks the GalaxDB sidecar protocol.

Sends real EmbedRequests over a Unix socket, prints the returned 384-dim
vectors. No mocks — every number printed comes from the Candle-loaded
sentence-transformers/all-MiniLM-L6-v2 model running in the sidecar.
"""
import json
import socket
import struct
import sys

SOCK = sys.argv[1] if len(sys.argv) > 1 else "/tmp/galaxdb-sidecar.sock"


def send(s, msg):
    data = json.dumps(msg).encode()
    s.sendall(struct.pack("<I", len(data)) + data)


def recv(s):
    header = s.recv(4)
    if len(header) < 4:
        raise RuntimeError("short header")
    n = struct.unpack("<I", header)[0]
    buf = b""
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise RuntimeError("short body")
        buf += chunk
    return json.loads(buf)


def embed(s, row_id, text, column="content"):
    send(s, {
        "type": "EmbedRequest",
        "row_id": row_id,
        "text": text,
        "column": column,
    })
    return recv(s)


def main():
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "A fast auburn fox leaps across a sleepy canine",  # near-duplicate of the first
        "The stock market closed 2 percent higher today",   # unrelated
    ]

    # Open a fresh connection per request (the sidecar handles one request
    # per connection per the manager-side wiring we've seen in the code).
    embeddings = []
    for i, text in enumerate(texts):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(SOCK)
            resp = embed(s, row_id=i, text=text)
        assert resp["type"] == "EmbedResponse", resp
        emb = resp["embedding"]
        print(f"text[{i}]: {text!r}")
        print(f"  dim={len(emb)}  model_version={resp['model_version']}")
        print(f"  first 8 dims: {[round(x, 4) for x in emb[:8]]}")
        norm = sum(x * x for x in emb) ** 0.5
        print(f"  L2 norm = {norm:.4f}")
        embeddings.append(emb)
        print()

    # Cosine similarity between each pair (all vectors should be L2-normalized
    # so dot product == cosine).
    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    print("=== cosine similarity ===")
    print(f"  text[0] vs text[1] (near-duplicate): {dot(embeddings[0], embeddings[1]):.4f}")
    print(f"  text[0] vs text[2] (unrelated)    : {dot(embeddings[0], embeddings[2]):.4f}")
    print(f"  text[1] vs text[2] (unrelated)    : {dot(embeddings[1], embeddings[2]):.4f}")
    print()
    print("If near-duplicate > unrelated, the real model semantics are working.")


if __name__ == "__main__":
    main()
