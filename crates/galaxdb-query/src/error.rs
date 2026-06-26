//! DataFusion → GalaxDB error mapping (HTAP task 12, Req 7.3).
//!
//! DataFusion errors must never reach the wire verbatim: users see a
//! GalaxDB-phrased message and a stable PostgreSQL SQLSTATE, while the raw
//! DataFusion text is kept for server-side diagnostics (logs only). This is
//! the single place that classifies a `DataFusionError`.

use datafusion::error::DataFusionError;
use galaxdb_common::GalaxError;

/// Map a [`DataFusionError`] to a GalaxDB-owned [`GalaxError::Query`] with a
/// PostgreSQL SQLSTATE and a sanitized, DataFusion-free message. The raw
/// error is emitted at `debug` for operators.
pub fn map_datafusion_error(e: DataFusionError) -> GalaxError {
    let raw = e.to_string();
    tracing::debug!(datafusion_error = %raw, "query engine error");

    let (sqlstate, kind): (&'static str, &str) = match &e {
        // Planning / binding / schema / SQL-parse problems are the user's
        // query being invalid: syntax_error_or_access_rule_violation class.
        DataFusionError::Plan(_) | DataFusionError::SchemaError(..) => {
            ("42601", "invalid query")
        }
        DataFusionError::SQL(..) => ("42601", "SQL syntax error"),
        DataFusionError::NotImplemented(_) => ("0A000", "unsupported query feature"),
        DataFusionError::ResourcesExhausted(_) => {
            ("53200", "query exceeded the memory budget")
        }
        _ => ("XX000", "query execution failed"),
    };

    GalaxError::Query {
        sqlstate,
        message: format!("{kind}: {}", sanitize(&raw)),
    }
}

/// Strip DataFusion-branded prefixes and references from a message so the
/// wire never exposes the engine behind the query layer (Req 7.3).
fn sanitize(raw: &str) -> String {
    let mut s = raw;
    // Drop common DataFusion error prefixes, keeping the useful detail.
    for prefix in [
        "Error during planning: ",
        "This feature is not implemented: ",
        "Schema error: ",
        "Execution error: ",
        "SQL error: ",
        "Internal error: ",
        "External error: ",
        "Resources exhausted: ",
        "Arrow error: ",
    ] {
        if let Some(rest) = s.strip_prefix(prefix) {
            s = rest;
            break;
        }
    }
    // Never expose the "DataFusion" brand.
    s.replace("DataFusion", "the query engine")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn not_implemented_maps_to_0a000_without_brand() {
        let e = DataFusionError::NotImplemented("FROBNICATE via DataFusion".into());
        let g = map_datafusion_error(e);
        assert_eq!(g.sqlstate(), "0A000");
        let msg = g.to_string();
        assert!(!msg.contains("DataFusion"), "msg leaked brand: {msg}");
        assert!(msg.contains("unsupported query feature"));
    }

    #[test]
    fn plan_error_maps_to_42601() {
        let e = DataFusionError::Plan("column \"x\" not found".into());
        let g = map_datafusion_error(e);
        assert_eq!(g.sqlstate(), "42601");
        let msg = g.to_string();
        assert!(msg.contains("column \"x\" not found"));
        assert!(!msg.contains("Error during planning"));
    }

    #[test]
    fn generic_error_is_xx000() {
        let e = DataFusionError::Execution("boom".into());
        assert_eq!(map_datafusion_error(e).sqlstate(), "XX000");
    }
}
