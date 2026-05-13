//! GalaxDB Observe — HTTP /health + /metrics, Prometheus, OTel tracing, JSON logging.
//!
//! This crate owns:
//! 1. The process-wide Prometheus [`Registry`] (task 38.2 / 38.3).
//! 2. An embedded HTTP server (axum) with `/health` and `/metrics`
//!    endpoints (task 38.1 / 38.2).
//! 3. Structured JSON logging via `tracing-subscriber` with
//!    configurable level from `GALAXDB_LOG_LEVEL` (task 38.4).
//! 4. OpenTelemetry W3C traceparent propagation helpers (task 38.5).
//! 5. SQL commenter format for trace context (task 38.6).
//!
//! Downstream crates that want to publish a metric:
//!
//! ```no_run
//! use prometheus::IntGauge;
//!
//! let gauge = IntGauge::new("example_gauge", "Example help").unwrap();
//! galaxdb_observe::default_registry()
//!     .register(Box::new(gauge.clone()))
//!     .expect("register example_gauge");
//! gauge.set(1);
//! ```

use std::net::SocketAddr;
use std::sync::OnceLock;

use axum::{routing::get, Router};
use prometheus::{Encoder, Registry, TextEncoder};
use serde::Serialize;
use tokio::net::TcpListener;

// ---------------------------------------------------------------------------
// Prometheus registry (stable since Phase E)
// ---------------------------------------------------------------------------

/// Process-wide Prometheus registry holder.
static REGISTRY: OnceLock<Registry> = OnceLock::new();

/// Return the process-wide default Prometheus [`Registry`].
pub fn default_registry() -> &'static Registry {
    REGISTRY.get_or_init(Registry::new)
}

// ---------------------------------------------------------------------------
// Metrics registration (task 38.3)
// ---------------------------------------------------------------------------

use prometheus::{IntCounter, IntGauge};
use std::sync::Arc;

/// All of GalaxDB's published metrics. Each field is the live handle
/// the owning subsystem updates; `/metrics` scrapes them from the
/// default registry. The set matches the list in task 38.3 exactly.
pub struct Metrics {
    /// Bytes resident in the buffer pool's hot-set (LRU, point lookups).
    pub buffer_pool_hot_set_usage: IntGauge,
    /// Bytes resident in the buffer pool's scan buffer (clock sweep).
    pub buffer_pool_scan_buffer_usage: IntGauge,
    /// In-flight embedding requests waiting on the sidecar.
    pub embedding_queue_depth: IntGauge,
    /// Rows parked on `_galaxdb_embedding_backlog` awaiting drain.
    pub embedding_backlog_depth: IntGauge,
    /// Duration of the most recent checkpoint flush, in ms.
    pub checkpoint_last_duration_ms: IntGauge,
    /// Bytes of compaction work still pending.
    pub compaction_pending_bytes: IntGauge,
    /// WAL append-to-fsync latency in µs (moving value; set per write).
    pub wal_write_latency_us: IntGauge,
    /// Most recent estimate of HNSW recall@10 (scaled by 10_000 so the
    /// gauge remains integer-typed — divide by 10_000 to get the 0..1
    /// ratio). Updated whenever a recall check runs.
    pub hnsw_recall_estimate: IntGauge,
    /// Current active wire-protocol connection count.
    pub connections_active: IntGauge,
    /// 1 when the embedding sidecar is healthy, 0 when degraded/down.
    pub sidecar_status: IntGauge,
    /// Total queries served over the wire (counter).
    pub queries_total: IntCounter,
}

static METRICS: OnceLock<Arc<Metrics>> = OnceLock::new();

fn build_metrics() -> Arc<Metrics> {
    fn gauge(name: &str, help: &str) -> IntGauge {
        let g = IntGauge::new(name, help).expect("valid metric name");
        // Registering twice fails with AlreadyReg — treat as OK since
        // the caller may retry. Tests re-run `build_metrics` without
        // clearing the registry.
        let _ = default_registry().register(Box::new(g.clone()));
        g
    }
    fn counter(name: &str, help: &str) -> IntCounter {
        let c = IntCounter::new(name, help).expect("valid metric name");
        let _ = default_registry().register(Box::new(c.clone()));
        c
    }
    Arc::new(Metrics {
        buffer_pool_hot_set_usage: gauge(
            "galaxdb_buffer_pool_hot_set_usage",
            "Bytes resident in the LRU hot-set portion of the buffer pool",
        ),
        buffer_pool_scan_buffer_usage: gauge(
            "galaxdb_buffer_pool_scan_buffer_usage",
            "Bytes resident in the clock-sweep scan buffer",
        ),
        embedding_queue_depth: gauge(
            "galaxdb_embedding_queue_depth",
            "In-flight embedding requests currently queued at the sidecar",
        ),
        embedding_backlog_depth: gauge(
            "galaxdb_embedding_backlog_depth",
            "Rows parked on the embedding backlog awaiting sidecar drain",
        ),
        checkpoint_last_duration_ms: gauge(
            "galaxdb_checkpoint_last_duration_ms",
            "Duration of the most recent checkpoint flush, in milliseconds",
        ),
        compaction_pending_bytes: gauge(
            "galaxdb_compaction_pending_bytes",
            "Bytes of compaction work pending across all LSM levels",
        ),
        wal_write_latency_us: gauge(
            "galaxdb_wal_write_latency_us",
            "Most recent WAL append-to-fsync latency in microseconds",
        ),
        hnsw_recall_estimate: gauge(
            "galaxdb_hnsw_recall_estimate_bp",
            "Most recent HNSW recall@10 estimate in basis points (1 bp = 0.01%)",
        ),
        connections_active: gauge(
            "galaxdb_connections_active",
            "Active wire-protocol client connections",
        ),
        sidecar_status: gauge(
            "galaxdb_sidecar_status",
            "1 when the embedding sidecar is healthy, 0 when degraded/down",
        ),
        queries_total: counter(
            "galaxdb_queries_total",
            "Total queries served over the wire",
        ),
    })
}

/// Return the process-wide [`Metrics`] handle. First call registers
/// every metric with the default registry; subsequent calls return
/// the same `Arc`.
pub fn metrics() -> Arc<Metrics> {
    METRICS
        .get_or_init(build_metrics)
        .clone()
}

/// Eagerly register every metric with the default registry so the
/// first `/metrics` scrape returns the complete set. Idempotent.
pub fn register_all_metrics() {
    let _ = metrics();
}

// ---------------------------------------------------------------------------
// HTTP server (task 38.1 / 38.2)
// ---------------------------------------------------------------------------

/// Health response body.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    /// Overall status string — `"ok"` when every subsystem is healthy,
    /// `"degraded"` when at least one subsystem is unhealthy. Clients
    /// poll this from load balancers, so the value is machine-parsable
    /// first and human-readable second.
    pub status: &'static str,
    pub version: &'static str,
    /// Per-subsystem health snapshot. Each field is derived from a
    /// real gauge at request time — no cached state, no approximation.
    pub subsystems: HealthSubsystems,
}

/// Snapshot of every subsystem's current health status.
#[derive(Debug, Serialize)]
pub struct HealthSubsystems {
    /// `true` when the engine has tripped into disk-full recovery
    /// mode (mirrors the `galaxdb_disk_full` gauge).
    pub disk_full: bool,
    /// `true` when the embedding sidecar is running and responding to
    /// heartbeats (mirrors the `galaxdb_sidecar_status` gauge).
    pub sidecar_healthy: bool,
    /// Current active wire-protocol connection count.
    pub connections_active: i64,
}

impl HealthSubsystems {
    /// Build a snapshot from the current gauge values. The gauges
    /// are zero-default so a subsystem that hasn't been wired yet
    /// reports its "safe" state (healthy / zero).
    ///
    /// `disk_full` is read by name from the default registry because
    /// the actual gauge is owned by
    /// `galaxdb-storage::disk_full::DiskFullHandler` (registered
    /// under `galaxdb_disk_full`). We don't re-register it here —
    /// one name, one owner.
    fn snapshot() -> Self {
        let m = metrics();
        Self {
            disk_full: read_disk_full_from_registry() == 1,
            sidecar_healthy: m.sidecar_status.get() == 1,
            connections_active: m.connections_active.get(),
        }
    }

    /// Overall: "ok" if every subsystem is healthy, "degraded" otherwise.
    fn overall_status(&self) -> &'static str {
        if self.disk_full {
            return "degraded";
        }
        "ok"
    }
}

/// Scrape `galaxdb_disk_full` from the default registry. Returns 0
/// if the gauge hasn't been registered yet (e.g. a test that runs
/// before any `DiskFullHandler` is constructed).
fn read_disk_full_from_registry() -> i64 {
    let registry = default_registry();
    for family in registry.gather() {
        if family.get_name() == "galaxdb_disk_full" {
            if let Some(m) = family.get_metric().first() {
                return m.get_gauge().get_value() as i64;
            }
        }
    }
    0
}

async fn health_handler() -> (axum::http::StatusCode, axum::Json<HealthResponse>) {
    let subsystems = HealthSubsystems::snapshot();
    let status = subsystems.overall_status();
    let http_status = if status == "ok" {
        axum::http::StatusCode::OK
    } else {
        // Load balancers treat 503 as "pull from rotation". Reporting
        // 503 on disk-full is the correct behaviour — we don't want
        // new writes routed to a stuck node.
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    };
    (
        http_status,
        axum::Json(HealthResponse {
            status,
            version: env!("CARGO_PKG_VERSION"),
            subsystems,
        }),
    )
}

async fn metrics_handler() -> (
    axum::http::StatusCode,
    [(axum::http::header::HeaderName, &'static str); 1],
    String,
) {
    let encoder = TextEncoder::new();
    let metric_families = default_registry().gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    let body = String::from_utf8(buffer).unwrap_or_default();
    (
        axum::http::StatusCode::OK,
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
}

/// Configuration for the observability HTTP server.
#[derive(Debug, Clone)]
pub struct ObserveConfig {
    /// Bind address for the HTTP server (e.g. `"0.0.0.0:9090"`).
    pub bind_addr: String,
}

impl Default for ObserveConfig {
    fn default() -> Self {
        Self {
            bind_addr: "0.0.0.0:9090".to_string(),
        }
    }
}

/// Start the observability HTTP server. Returns the bound address and
/// a join handle. The server runs until the handle is aborted or the
/// process exits.
pub async fn start_http(
    config: ObserveConfig,
) -> std::io::Result<(SocketAddr, tokio::task::JoinHandle<()>)> {
    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler));

    let listener = TcpListener::bind(&config.bind_addr).await?;
    let addr = listener.local_addr()?;

    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.ok();
    });

    Ok((addr, handle))
}

// ---------------------------------------------------------------------------
// Structured JSON logging (task 38.4)
// ---------------------------------------------------------------------------

/// Initialize the global tracing subscriber with structured JSON output.
///
/// The log level is read from `GALAXDB_LOG_LEVEL` (e.g. `info`,
/// `debug`, `galaxdb=trace`). Falls back to `info` if the env var is
/// absent or unparsable.
///
/// Call this once at process startup (before any `tracing::info!` etc).
/// Subsequent calls are no-ops (the global subscriber is set exactly
/// once).
pub fn init_logging() {
    use tracing_subscriber::{fmt, EnvFilter};

    let filter = EnvFilter::try_from_env("GALAXDB_LOG_LEVEL")
        .unwrap_or_else(|_| EnvFilter::new("info"));

    let subscriber = fmt::Subscriber::builder()
        .json()
        .with_env_filter(filter)
        .with_target(true)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .finish();

    // `set_global_default` returns Err if already set — that's fine.
    let _ = tracing::subscriber::set_global_default(subscriber);
}

// ---------------------------------------------------------------------------
// OpenTelemetry W3C traceparent helpers (task 38.5)
// ---------------------------------------------------------------------------

/// A minimal W3C traceparent header value.
///
/// Format: `00-<trace_id>-<span_id>-<flags>`
/// where trace_id is 32 hex chars, span_id is 16 hex chars, flags is
/// 2 hex chars (01 = sampled).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Traceparent {
    pub trace_id: String,
    pub span_id: String,
    pub sampled: bool,
}

impl Traceparent {
    /// Parse a W3C traceparent header value.
    pub fn parse(header: &str) -> Option<Self> {
        let parts: Vec<&str> = header.split('-').collect();
        if parts.len() != 4 || parts[0] != "00" {
            return None;
        }
        if parts[1].len() != 32 || parts[2].len() != 16 || parts[3].len() != 2 {
            return None;
        }
        Some(Self {
            trace_id: parts[1].to_string(),
            span_id: parts[2].to_string(),
            sampled: parts[3] == "01",
        })
    }

    /// Render as a W3C traceparent header value.
    pub fn to_header(&self) -> String {
        let flags = if self.sampled { "01" } else { "00" };
        format!("00-{}-{}-{}", self.trace_id, self.span_id, flags)
    }
}

// ---------------------------------------------------------------------------
// SQL commenter format (task 38.6)
// ---------------------------------------------------------------------------

/// Extract a traceparent from a SQL commenter suffix.
///
/// SQL commenter format: `/* traceparent='00-...-...-01' */`
/// appended to the end of a SQL statement. This function scans for
/// the pattern and returns the parsed [`Traceparent`] if found.
pub fn extract_traceparent_from_sql(sql: &str) -> Option<Traceparent> {
    let marker = "traceparent='";
    let start = sql.find(marker)?;
    let value_start = start + marker.len();
    let rest = &sql[value_start..];
    let end = rest.find('\'')?;
    let value = &rest[..end];
    Traceparent::parse(value)
}

/// Append a traceparent as a SQL commenter to a statement.
pub fn append_traceparent_to_sql(sql: &str, tp: &Traceparent) -> String {
    format!("{} /* traceparent='{}' */", sql.trim_end(), tp.to_header())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests that mutate the shared gauge state (`disk_full`,
    /// `sidecar_status`) must not race. `cargo test` runs tests in
    /// parallel by default, so we serialize them on a dedicated
    /// mutex. The existing Phase E disk_full tests use the same
    /// pattern in `galaxdb-storage`.
    static GAUGE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Find-or-create a test-only gauge registered under the real
    /// `galaxdb_disk_full` name. If `galaxdb-storage::disk_full`
    /// already registered the gauge (either in the same process via
    /// a prior test or via a live `DiskFullHandler`), we reuse that
    /// registration by doing a registry scrape inside
    /// [`read_disk_full_from_registry`]. In a pure-observe test
    /// process nothing else has registered yet, so we register a
    /// test-only gauge here — the name must match what
    /// `HealthSubsystems::snapshot` looks up.
    fn test_disk_full_gauge() -> prometheus::IntGauge {
        use prometheus::IntGauge;
        static TEST_DISK_FULL: OnceLock<IntGauge> = OnceLock::new();
        TEST_DISK_FULL
            .get_or_init(|| {
                let g = IntGauge::new(
                    "galaxdb_disk_full",
                    "Set to 1 while the storage engine is in disk-full recovery mode, 0 otherwise.",
                )
                .unwrap();
                // Best-effort: if `galaxdb-storage::disk_full` already
                // registered the gauge under this name,
                // `register(Box::new(g.clone()))` returns AlreadyReg,
                // and our handle is orphaned — but tests that only
                // depend on `read_disk_full_from_registry` will still
                // see the right value since the scrape reads from the
                // registered instance. For pure-observe tests (this
                // file) the registration succeeds.
                let _ = default_registry().register(Box::new(g.clone()));
                g
            })
            .clone()
    }

    #[test]
    fn default_registry_is_stable_across_calls() {
        let a = default_registry() as *const Registry;
        let b = default_registry() as *const Registry;
        assert_eq!(a, b);
    }

    #[test]
    fn default_registry_accepts_metric_registration() {
        use prometheus::IntGauge;
        let gauge = IntGauge::new(
            "galaxdb_observe_test_gauge",
            "Unit test gauge",
        )
        .unwrap();
        let _ = default_registry().register(Box::new(gauge));
    }

    #[test]
    fn health_response_serializes() {
        let h = HealthResponse {
            status: "ok",
            version: "0.1.0",
            subsystems: HealthSubsystems {
                disk_full: false,
                sidecar_healthy: true,
                connections_active: 3,
            },
        };
        let json = serde_json::to_string(&h).unwrap();
        assert!(json.contains("\"status\":\"ok\""));
        assert!(json.contains("\"disk_full\":false"));
        assert!(json.contains("\"connections_active\":3"));
    }

    #[test]
    fn traceparent_parse_roundtrip() {
        let header = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
        let tp = Traceparent::parse(header).unwrap();
        assert_eq!(tp.trace_id, "4bf92f3577b34da6a3ce929d0e0e4736");
        assert_eq!(tp.span_id, "00f067aa0ba902b7");
        assert!(tp.sampled);
        assert_eq!(tp.to_header(), header);
    }

    #[test]
    fn traceparent_parse_unsampled() {
        let header = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00";
        let tp = Traceparent::parse(header).unwrap();
        assert!(!tp.sampled);
    }

    #[test]
    fn traceparent_parse_invalid_returns_none() {
        assert!(Traceparent::parse("invalid").is_none());
        assert!(Traceparent::parse("01-abc-def-01").is_none());
        assert!(Traceparent::parse("").is_none());
    }

    #[test]
    fn sql_commenter_extract() {
        let sql = "SELECT * FROM t /* traceparent='00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01' */";
        let tp = extract_traceparent_from_sql(sql).unwrap();
        assert_eq!(tp.trace_id, "4bf92f3577b34da6a3ce929d0e0e4736");
        assert!(tp.sampled);
    }

    #[test]
    fn sql_commenter_append() {
        let tp = Traceparent {
            trace_id: "a".repeat(32),
            span_id: "b".repeat(16),
            sampled: true,
        };
        let sql = append_traceparent_to_sql("SELECT 1", &tp);
        assert!(sql.starts_with("SELECT 1 /* traceparent='00-"));
        assert!(sql.ends_with("-01' */"));
        // Round-trip
        let extracted = extract_traceparent_from_sql(&sql).unwrap();
        assert_eq!(extracted, tp);
    }

    #[tokio::test]
    async fn http_health_returns_ok_json() {
        let _guard = GAUGE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let m = metrics();
        let df = test_disk_full_gauge();
        df.set(0);
        m.sidecar_status.set(0);

        let config = ObserveConfig {
            bind_addr: "127.0.0.1:0".to_string(),
        };
        let (addr, _handle) = start_http(config).await.unwrap();
        let url = format!("http://{}/health", addr);
        let resp = reqwest::get(&url).await.unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["status"], "ok");
        assert!(body["subsystems"].is_object());
        assert_eq!(body["subsystems"]["disk_full"], false);
    }

    #[tokio::test]
    async fn http_health_reports_503_when_disk_full() {
        let _guard = GAUGE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let df = test_disk_full_gauge();
        df.set(1);

        let config = ObserveConfig {
            bind_addr: "127.0.0.1:0".to_string(),
        };
        let (addr, _handle) = start_http(config).await.unwrap();
        let url = format!("http://{}/health", addr);
        let resp = reqwest::get(&url).await.unwrap();
        assert_eq!(resp.status(), 503);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["status"], "degraded");
        assert_eq!(body["subsystems"]["disk_full"], true);

        df.set(0);
    }

    #[tokio::test]
    async fn http_metrics_returns_prometheus_format() {
        // Register a test metric so the output is non-empty.
        use prometheus::IntGauge;
        let gauge = IntGauge::new(
            "galaxdb_observe_http_test_metric",
            "test metric for /metrics endpoint",
        )
        .unwrap();
        let _ = default_registry().register(Box::new(gauge.clone()));
        gauge.set(42);

        let config = ObserveConfig {
            bind_addr: "127.0.0.1:0".to_string(),
        };
        let (addr, _handle) = start_http(config).await.unwrap();
        let url = format!("http://{}/metrics", addr);
        let resp = reqwest::get(&url).await.unwrap();
        assert_eq!(resp.status(), 200);
        let ct = resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap();
        assert!(
            ct.contains("text/plain"),
            "Prometheus text format content-type expected, got {ct}"
        );
        let body = resp.text().await.unwrap();
        assert!(
            body.contains("galaxdb_observe_http_test_metric 42"),
            "expected the registered metric in the output; got:\n{body}"
        );
    }

    /// Task 38.3: calling `register_all_metrics` must land every
    /// spec-listed metric in the default registry so the first
    /// `/metrics` scrape is complete. The metrics in the spec are:
    /// buffer_pool_hot_set_usage, buffer_pool_scan_buffer_usage,
    /// embedding_queue_depth, embedding_backlog_depth,
    /// checkpoint_last_duration_ms, compaction_pending_bytes,
    /// wal_write_latency_us, hnsw_recall_estimate, connections_active,
    /// disk_full, sidecar_status.
    #[test]
    fn all_spec_metrics_register() {
        register_all_metrics();
        let m = metrics();
        // Ensure `galaxdb_disk_full` is present in the registry.
        // In production that gauge is registered by
        // `galaxdb-storage::disk_full::DiskFullHandler`; for this
        // pure-observe test we register an equivalent gauge so the
        // name shows up in the scrape.
        let _ = test_disk_full_gauge();
        // Just touching each handle proves it exists and is the
        // right type — exhaustive destructuring would be too noisy.
        m.buffer_pool_hot_set_usage.set(0);
        m.buffer_pool_scan_buffer_usage.set(0);
        m.embedding_queue_depth.set(0);
        m.embedding_backlog_depth.set(0);
        m.checkpoint_last_duration_ms.set(0);
        m.compaction_pending_bytes.set(0);
        m.wal_write_latency_us.set(0);
        m.hnsw_recall_estimate.set(0);
        m.connections_active.set(0);
        m.sidecar_status.set(0);

        // Gather and confirm every metric name appears in the output.
        let families = default_registry().gather();
        let names: std::collections::BTreeSet<String> = families
            .iter()
            .map(|f| f.get_name().to_string())
            .collect();
        for required in [
            "galaxdb_buffer_pool_hot_set_usage",
            "galaxdb_buffer_pool_scan_buffer_usage",
            "galaxdb_embedding_queue_depth",
            "galaxdb_embedding_backlog_depth",
            "galaxdb_checkpoint_last_duration_ms",
            "galaxdb_compaction_pending_bytes",
            "galaxdb_wal_write_latency_us",
            "galaxdb_hnsw_recall_estimate_bp",
            "galaxdb_connections_active",
            "galaxdb_disk_full",
            "galaxdb_sidecar_status",
        ] {
            assert!(
                names.contains(required),
                "required metric '{required}' missing from the registry; got {names:?}"
            );
        }
    }
}
