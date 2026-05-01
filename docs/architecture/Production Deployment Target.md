Our database engine is Rust—that agreement is absolute. Go enters only when we need to orchestrate the platform, which in practice means the v2 distributed control plane, the Kubernetes operator, and cloud‑management services. This document explains why those two languages are complementary, how the complete GalaxDB Cloud and Enterprise stack fits together, and how a shared multi‑tenant deployment can be implemented securely on Amazon Web Services (AWS).

---

## 1. Production Deployment Target

GalaxDB Cloud runs exclusively on **Linux 5.10+ LTS** instances on AWS, using `io_uring` for storage I/O. macOS and Windows are supported as development and testing platforms only and do not receive the same performance guarantees.

---

## 2. Language Strategy: Rust + Go

### 2.1 Rust — The Core Engine

The entire database engine — LSM storage, HNSW index, SQL planner, embedding sidecar, encryption, durability — is implemented in **Rust**. It ships as a single binary that can be embedded in a Python process, run as a standalone server, or deployed as a distributed node. All performance‑sensitive, correctness‑critical components are in Rust.

### 2.2 Go — Orchestration and Cloud‑Native Integration

Go powers the **v2 distributed control plane** and the **GalaxDB Cloud management API**. Its role is limited to:

- **Cluster orchestration** — Raft leader election, shard rebalancing, node health monitoring.
- **Kubernetes operator** — managing GalaxDB cluster lifecycle inside Kubernetes via CRDs, leveraging the Operator SDK and controller‑runtime.
- **Cloud API server** — REST/gRPC endpoints for provisioning, billing, metering, and the management dashboard.
- **Sidecar‑spawning logic** — the Go manager starts the Rust engine and the embedding inference sidecar on each node, watches their health, and restarts them when necessary.

A **gRPC link** connects the Go control plane to the Rust engine. The Rust engine exposes an internal gRPC server (using `tonic`) that receives configuration changes and reports health metrics; the Go controller consumes that interface. This separation means the Go components never touch the hot data path.

### 2.3 Key Rust Crates and Go Modules

| Component | Language | Key Frameworks/Libraries |
|-----------|----------|---------------------------|
| Storage engine, SQL, ANN | Rust | `crossbeam‑skiplist‑mvcc`, `sqlparser‑rs`, `tonic`, `opentelemetry‑rust`, `aws‑sdk‑rust` |
| Embedding sidecar | Rust | `ort` (ONNX Runtime), `tonic` (gRPC), `tokio` (async runtime) |
| Distributed control plane | Go | `client‑go`, `controller‑runtime`, `grpc‑go`, `opentelemetry‑go` |
| Cloud API & dashboards | Go | `gin` or `echo`, `grpc‑go`, AWS SDK for Go |

---

## 3. GalaxDB Cloud on AWS — Free Tier

We need the free tier to serve legitimate developers while costing us almost nothing. Existing cloud databases (Neon, CockroachDB Serverless, PlanetScale) have proven that feasible multi‑tenant density makes this economically viable .

### 3.1 Free Tier Isolation Strategy

The free tier uses a **pooled** model: many tenants share one GalaxDB single‑node instance. Within that instance, each tenant is mapped to a **separate schema (namespace)**. GalaxDB’s SQL parser already supports PostgreSQL‑compatible schemas, and the storage engine never compacts or merges data across schemas. Schema‑level isolation provides substantially better blast radius containment than row‑level isolation while keeping per‑tenant infrastructure costs near zero .

This tier will be marketed to startups building their MVPs. It is not designed for business‑critical workloads, and its limitations—shared compute, memory caps, and a low connection ceiling—will be transparently documented. Enterprise customers and high‑growth teams can upgrade to Pro or Enterprise plans to escape the blast radius.

### 3.2 Instance Selection

| Component | Instance | vCPU | RAM | Notes |
|-----------|---------|------|-----|-------|
| GalaxDB shard (free tier) | **t4g.medium** (ARM Graviton) | 2 | 4 GB | Burstable, scales to zero. With group commit, handles ~85k TPS. ARM‑optimised with FP16 quantisation  |
| Sidecar embedding inference | Co‑located on same instance | — | — | Model runs on CPU via ONNX Runtime; negligible incremental cost. |

### 3.3 Cost Per Free‑Tier User

| Item | Monthly Estimate |
|------|-----------------|
| Compute (100 hours active) | $3.36 |
| EBS gp3 (5 GB) | $0.40 |
| Total AWS Cost | **$3.76 per active user** |
| Idle user (scaled to zero) | $0.40 (storage only) |

Group commit (10 ms window) yields approximately 95k TPS per node, supporting roughly 500 tenants at 190 TPS each without memory pressure. Scaling to 50,000 free users costs about $25k–$50k/month, a reasonable customer‑acquisition cost for a high‑conversion product.

### 3.4 Free‑Tier Architecture Diagram

```
User (Free Tier) → Amazon API Gateway → GalaxDB Cloud API (Go)
                                              │
                           ┌──────────────────┴──────────────────┐
                           │  Shared Cluster  (t4g.medium)       │
                           │  ┌──────────────────────────────┐   │
                           │  │  GalaxDB Engine (Rust)       │   │
                           │  │  Schema A │ Schema B │ ...   │   │
                           │  └──────────────────────────────┘   │
                           └─────────────────────────────────────┘
```

---

## 4. GalaxDB Cloud on AWS — Pro Tier

### 4.1 Instance Selection

| Component | Instance | vCPU | RAM | Notes |
|-----------|---------|------|-----|-------|
| GalaxDB (Pro) | **i4i.large** (Intel Xeon) | 2 | 16 GB | 468 GB local NVMe for < 1 ms vector search  |
| Embedding sidecar | Co‑located | — | — | CPU inference; model loaded in shared RAM |
| S3 backup (durable) | S3 Standard | — | — | Nightly snapshots; incremental Merkle‑root manifests |

### 4.2 Per‑Customer Unit Economics (Monthly)

| Meter | Revenue | AWS Cost | Margin |
|-------|---------|----------|--------|
| Compute (2 vCPUs × 730 hours) | $365.00 | $175.20 | 52 % |
| Storage (100 GB NVMe) | $10.00 | $8.00 | 20 % |
| Embedding (50M tokens) | $2.50 | $0 | 100 % |
| **Total** | **$377.50** | **$183.20** | **51 %** |

---

## 5. GalaxDB Enterprise — On‑Prem or VPC

Enterprise customers receive the full distributed v2 cluster directly on their infrastructure. The control plane (Go) manages the cluster; the engine (Rust) runs on customer hardware. No cloud compute margin applies, so license revenue is nearly all gross profit.

### 5.1 Enterprise Deployment Stack (Customer’s Environment)

| Tier | Component | Deployment |
|------|-----------|------------|
| **Control Plane** | GalaxDB Control Plane (Go) | Kubernetes cluster or dedicated VMs |
| | Kubernetes Operator (Go) | Helm chart with CRDs |
| **Data Plane** | GalaxDB Engine (Rust) | Dedicated EC2 instances (i4i, c7g) or bare‑metal |
| | Embedding Sidecar (Rust) | Co‑located or GPU‑dedicated (p4d/p5) |
| **Observability** | OTEL Collector + Prometheus | Deployed alongside the cluster |

### 5.2 Compliance & Security

- **Network isolation:** All intra‑cluster traffic (Raft, gRPC) uses TLS 1.3 and remains within the customer’s VPC.
- **Key sovereignty:** Encryption keys live in the customer’s AWS KMS or on‑prem HSM; GalaxDB never sees them.
- **Audit trails:** The `_galaxdb_training_exports` and `_galaxdb_predictions` tables provide the data lineage required by the EU AI Act.

---

## 6. Multi‑Tenant Isolation in GalaxDB Engine

The engine enforces isolation at the **schema** level:

- **Schema‑per‑tenant** maps each tenant to a PostgreSQL‑compatible schema. Data blocks belonging to different schemas are never compacted together, providing physical separation within a shared LSM store .
- **Credentials** are restricted via AWS IAM task roles so that each tenant can connect only to its own schema; a misconfigured query cannot select from another schema because the IAM policy denies the cross‑schema reference before the SQL planner executes it .
- **Cross‑tenant compaction** is disabled. A maintenance background job runs per schema, preventing any single heavy compaction from starving neighbouring tenants.
- **Row‑Level Security (RLS)** supplements schema‑level isolation for use cases where a tenant may sub‑let a database to multiple internal teams while keeping all data within a single schema .
- **Noisy‑neighbour protection** is provided by per‑schema connection pools and a token‑bucket rate limiter that caps queries per second per tenant, preventing a free‑tier user from consuming all instance resources .

---

## 7. Technology Stack Summary

### 7.1 Infrastructure Platform

- **Compute:** Amazon ECS on AWS Fargate (serverless containers) for the Cloud API. EC2 bare-metal instances for the database engine (io_uring requires direct NVMe access).
- **Orchestration:** Kubernetes (EKS) for v2 distributed clusters. The Go Kubernetes operator manages cluster lifecycle. ECS Service Connect handles service discovery within the cloud control plane .
- **Observability:** Prometheus metrics scraped from every component. OpenTelemetry trace context propagated across gRPC calls (Go ↔ Rust) using the W3C Trace Context format . Traces exported to AWS X‑Ray or an OTLP collector.
- **Key Management:** AWS KMS for TDE key generation and rotation. AES‑256‑GCM encryption with AES‑NI hardware acceleration.
- **CI/CD:** GitHub Actions with `cargo build --release` for Rust binaries and `go build` for Go components. Container images pushed to Amazon ECR. Infrastructure provisioned via CDK (TypeScript or Go).

### 7.2 GalaxDB-Specific Services

| Service | Language | Framework | Purpose |
|---------|----------|-----------|---------|
| **Engine binary** | Rust | tokio, tonic, sqlparser-rs | v1 single‑node; v2 distributed node |
| **Embedding sidecar** | Rust | ort (ONNX Runtime), tonic | Embedding inference on CPU/GPU |
| **Control plane** | Go | grpc-go, client-go | Cluster orchestration, shard management |
| **Kubernetes operator** | Go | controller-runtime, operator-sdk | CRD‑based cluster lifecycle |
| **Cloud API** | Go | gin (HTTP) + grpc-go | Provisioning, billing, metering |
| **Management dashboard** | Go (backend), React (frontend) | gin, Next.js | User‑facing console |

---

## 8. Deployment Automation

### 8.1 GalaxDB Cloud Provisioning Flow

1. **User signs up** via the dashboard (or API). The GalaxDB Cloud API (Go) receives the request.
2. **API provisions a schema** on a shared cluster by calling a gRPC endpoint on the Go control plane.
3. **Control plane** generates per‑schema encryption keys via AWS KMS and injects them into the Rust engine.
4. **Engine** (Rust) creates the schema and returns connection credentials.
5. **Credentials** are delivered to the user — a PostgreSQL‑compatible connection string with schema pre‑selected.

### 8.2 Enterprise Provisioning Flow

1. **Customer deploys** the GalaxDB Helm chart into their Kubernetes cluster.
2. **Go operator** creates the CRD and spins up the specified number of Rust engine pods.
3. **Operator** handles Raft bootstrap, shard assignment, and health monitoring.
4. **Customer connects** directly to the cluster endpoint (a Network Load Balancer in their VPC).

---

## 9. Security Architecture

- **Data at rest:** AES‑256‑GCM block‑level encryption with AES‑NI. Keys stored in AWS KMS, never in the database process .
- **Data in transit:** TLS 1.3 for all PostgreSQL wire‑protocol connections. Mutual TLS for intra‑cluster Raft and gRPC traffic.
- **IAM integration:** ECS tasks assume fine‑grained IAM roles that grant access only to their own schema and their own KMS key .
- **io_uring security:** io_uring is disabled by default in Fargate and Docker seccomp profiles. The Cloud API runs in Fargate with `GALAXDB_IO_BACKEND=tokio`. The database engine runs on dedicated EC2 with `seccomp=unconfined` and `io_uring` enabled. This separation ensures the attack surface is exposed only where the performance justifies it.
- **Audit:** The `_galaxdb_training_exports` table provides complete data lineage for AI training workloads, satisfying EU AI Act Article 13.

---

## 10. Monitoring and Observability

Every component emits three standard signals:

| Signal | Technology | Consumer |
|--------|-----------|----------|
| **Metrics** | Prometheus (Rust: `prometheus` crate; Go: `promhttp`) | CloudWatch, Grafana |
| **Logging** | Structured JSON (Rust: `tracing-subscriber`; Go: `zerolog`) | CloudWatch Logs |
| **Tracing** | OpenTelemetry (W3C Trace Context over gRPC) | AWS X‑Ray or OTLP collector |

All metrics defined in the architecture specification (buffer pressure, embedding queue depth, checkpoint status, compaction debt) are automatically exported. The Go control plane emits cluster‑level metrics (shard health, Raft leader election count, gRPC latency), while the Rust engine emits storage‑level metrics (LSM compaction stalls, HNSW recall, WAL write latency).

---

## 11. Cost Summary

### 11.1 Cloud Service Operating Costs (Monthly Estimate)

| Tier | Users (Est.) | AWS infra cost | Staff cost (amortised) | Total burn |
|------|-------------|---------------|----------------------|------------|
| Free tier | 10,000 | $25k | $15k | $40k |
| Pro tier | 200 | $35k | $25k | $60k |
| Enterprise | 10 | $0 (customer hosts) | $20k | $20k |

### 11.2 Break‑Even Timeline

With 190 Pro customers at $377.50/month each, GalaxDB Cloud generates approximately $72k in monthly recurring revenue. After deducting $50k in infrastructure costs, the service reaches gross profit at roughly 200 paying customers — achievable within 12‑18 months of cloud launch.

---

## 12. Summary

- **The engine is Rust** — storage, SQL, HNSW, encryption, sidecar.
- **Orchestration is Go** — control plane, Kubernetes operator, cloud API.
- **Free tier** uses schema‑level isolation on shared t4g.medium instances, costing less than $4 per active user per month.
- **Pro tier** runs on dedicated i4i.large instances with local NVMe, generating approximately 51 % gross margin.
- **Enterprise** deploys the full v2 distributed cluster on customer infrastructure, generating 85 %+ gross margin on license fees.
- **AWS is the deployment target** — ECS Fargate for control plane, EC2 with io_uring for the engine, KMS for encryption, S3 for backup and cold‑tier storage.

The infrastructure is designed to scale from a single shared EC2 instance serving thousands of free‑tier developers to hundreds of dedicated enterprise clusters deployed globally, all running the same Rust engine binary and the same Go orchestration layer.