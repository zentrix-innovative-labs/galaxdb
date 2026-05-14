FROM ubuntu:24.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates build-essential pkg-config libssl-dev \
    protobuf-compiler libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build
COPY . .

RUN RUSTFLAGS="-C target-cpu=x86-64" cargo build --release -p galaxdb-server -p galaxdb-sidecar

# ── Runtime image ──────────────────────────────────────────────────
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/galaxdb-server /usr/local/bin/galaxdb-server
COPY --from=builder /build/target/release/galaxdb-sidecar /usr/local/bin/galaxdb-sidecar
RUN chmod +x /usr/local/bin/galaxdb-server /usr/local/bin/galaxdb-sidecar

# Data directory
VOLUME ["/data"]

# Wire protocol
EXPOSE 5433
# HTTP observability (/health, /metrics)
EXPOSE 9090

ENTRYPOINT ["galaxdb-server"]
CMD ["--data-dir", "/data", "--port", "5433", "--observe-port", "9090"]
