FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built Linux binary
COPY release-binaries/linux-x86_64/galaxdb-server /usr/local/bin/galaxdb-server
COPY release-binaries/linux-x86_64/galaxdb-sidecar /usr/local/bin/galaxdb-sidecar
RUN chmod +x /usr/local/bin/galaxdb-server /usr/local/bin/galaxdb-sidecar

# Data directory
VOLUME ["/data"]

# Wire protocol
EXPOSE 5433
# HTTP observability (/health, /metrics)
EXPOSE 9090

ENTRYPOINT ["galaxdb-server"]
CMD ["--data-dir", "/data", "--port", "5433", "--observe-port", "9090"]
