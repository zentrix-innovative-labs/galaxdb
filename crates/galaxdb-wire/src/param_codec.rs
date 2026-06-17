//! Bound-parameter decoding for the extended query protocol (Req 6 AC5).
//!
//! A `Bind` message carries each parameter as either text or binary
//! (per-parameter format code) plus the parameter type OID declared in the
//! preceding `Parse`. To execute a prepared statement, GalaxDB substitutes
//! each bound value into the statement text as a **SQL literal** and runs
//! the resulting concrete statement through the normal
//! `execute_with_context` path — so authentication and authorization apply
//! identically to the simple-query path (Req 6 AC7), and the value is
//! always rendered as a properly quoted/escaped literal (no injection: the
//! bytes come pre-typed over the wire, never as raw SQL).
//!
//! Supported types match what the executor already understands (Req 6 AC5):
//! integer, bigint, float4, float8, boolean, text. Unknown OIDs are treated
//! as text. Binary encodings follow the PostgreSQL on-wire representation
//! (network byte order for numerics, 1 byte for bool).

/// PostgreSQL type OIDs we decode. Anything else is treated as text.
pub mod oid {
    pub const BOOL: i32 = 16;
    pub const INT8: i32 = 20;
    pub const INT2: i32 = 21;
    pub const INT4: i32 = 23;
    pub const TEXT: i32 = 25;
    pub const FLOAT4: i32 = 700;
    pub const FLOAT8: i32 = 701;
    pub const VARCHAR: i32 = 1043;
}

/// Format code for a bound parameter: 0 = text, 1 = binary.
const FORMAT_TEXT: i16 = 0;
const FORMAT_BINARY: i16 = 1;

/// Render a single bound parameter as a SQL literal suitable for textual
/// substitution into the prepared statement.
///
/// * `value` — the raw parameter bytes, or `None` for SQL NULL.
/// * `format` — 0 (text) or 1 (binary).
/// * `type_oid` — the parameter's type OID from `Parse` (0 = unspecified).
pub fn param_to_sql_literal(
    value: Option<&[u8]>,
    format: i16,
    type_oid: i32,
) -> Result<String, String> {
    let bytes = match value {
        None => return Ok("NULL".to_string()),
        Some(b) => b,
    };

    match format {
        FORMAT_TEXT => render_text(bytes, type_oid),
        FORMAT_BINARY => render_binary(bytes, type_oid),
        other => Err(format!("unsupported parameter format code {other}")),
    }
}

/// Render a text-format parameter. Numeric/boolean OIDs are emitted
/// unquoted (validated as parseable); everything else is a quoted,
/// escaped string literal.
fn render_text(bytes: &[u8], type_oid: i32) -> Result<String, String> {
    let s = std::str::from_utf8(bytes)
        .map_err(|_| "text parameter is not valid UTF-8".to_string())?;
    match type_oid {
        oid::BOOL => {
            // PostgreSQL text bool: t/f/true/false/1/0/...
            let v = parse_bool(s)?;
            Ok(if v { "TRUE".into() } else { "FALSE".into() })
        }
        oid::INT2 | oid::INT4 | oid::INT8 => {
            s.trim()
                .parse::<i64>()
                .map_err(|_| format!("invalid integer parameter '{s}'"))?;
            Ok(s.trim().to_string())
        }
        oid::FLOAT4 | oid::FLOAT8 => {
            s.trim()
                .parse::<f64>()
                .map_err(|_| format!("invalid float parameter '{s}'"))?;
            Ok(s.trim().to_string())
        }
        _ => Ok(quote_string(s)),
    }
}

/// Render a binary-format parameter from the PostgreSQL on-wire encoding.
fn render_binary(bytes: &[u8], type_oid: i32) -> Result<String, String> {
    match type_oid {
        oid::BOOL => {
            if bytes.len() != 1 {
                return Err("binary bool must be exactly 1 byte".into());
            }
            Ok(if bytes[0] != 0 { "TRUE".into() } else { "FALSE".into() })
        }
        oid::INT2 => {
            let arr: [u8; 2] = bytes
                .try_into()
                .map_err(|_| "binary int2 must be 2 bytes".to_string())?;
            Ok(i16::from_be_bytes(arr).to_string())
        }
        oid::INT4 => {
            let arr: [u8; 4] = bytes
                .try_into()
                .map_err(|_| "binary int4 must be 4 bytes".to_string())?;
            Ok(i32::from_be_bytes(arr).to_string())
        }
        oid::INT8 => {
            let arr: [u8; 8] = bytes
                .try_into()
                .map_err(|_| "binary int8 must be 8 bytes".to_string())?;
            Ok(i64::from_be_bytes(arr).to_string())
        }
        oid::FLOAT4 => {
            let arr: [u8; 4] = bytes
                .try_into()
                .map_err(|_| "binary float4 must be 4 bytes".to_string())?;
            Ok(format_float(f32::from_be_bytes(arr) as f64))
        }
        oid::FLOAT8 => {
            let arr: [u8; 8] = bytes
                .try_into()
                .map_err(|_| "binary float8 must be 8 bytes".to_string())?;
            Ok(format_float(f64::from_be_bytes(arr)))
        }
        // TEXT/VARCHAR/unspecified: raw UTF-8 bytes.
        _ => {
            let s = std::str::from_utf8(bytes)
                .map_err(|_| "binary text parameter is not valid UTF-8".to_string())?;
            Ok(quote_string(s))
        }
    }
}

fn parse_bool(s: &str) -> Result<bool, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "t" | "true" | "1" | "yes" | "on" => Ok(true),
        "f" | "false" | "0" | "no" | "off" => Ok(false),
        other => Err(format!("invalid boolean parameter '{other}'")),
    }
}

/// Format a float as a SQL numeric literal, preserving non-finite values
/// as the PostgreSQL spellings.
fn format_float(v: f64) -> String {
    if v.is_nan() {
        "'NaN'".to_string()
    } else if v.is_infinite() {
        if v > 0.0 { "'Infinity'".into() } else { "'-Infinity'".into() }
    } else {
        // Always include a decimal point so the literal is float-typed.
        let s = v.to_string();
        if s.contains('.') || s.contains('e') || s.contains('E') {
            s
        } else {
            format!("{s}.0")
        }
    }
}

/// Quote and escape a string as a SQL single-quoted literal (doubling any
/// embedded single quotes).
fn quote_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for ch in s.chars() {
        if ch == '\'' {
            out.push('\'');
        }
        out.push(ch);
    }
    out.push('\'');
    out
}

/// Substitute bound parameters (`$1`, `$2`, …) in `query` with their
/// rendered SQL literals. Replaces higher indices first so `$1` does not
/// match the prefix of `$10`.
pub fn substitute_parameters(query: &str, literals: &[String]) -> String {
    let mut out = query.to_string();
    for i in (1..=literals.len()).rev() {
        let placeholder = format!("${i}");
        out = out.replace(&placeholder, &literals[i - 1]);
    }
    out
}

/// Decode a bound parameter into a typed [`galaxdb_sql::BoundValue`] for
/// AST-level binding into a prepared statement template (the parse-once
/// path, Req 7). Mirrors [`param_to_sql_literal`] but yields a typed value
/// instead of a literal string, so the executor reuses the cached parse.
pub fn param_to_bound_value(
    value: Option<&[u8]>,
    format: i16,
    type_oid: i32,
) -> Result<galaxdb_sql::BoundValue, String> {
    use galaxdb_sql::BoundValue;
    let bytes = match value {
        None => return Ok(BoundValue::Null),
        Some(b) => b,
    };
    match format {
        FORMAT_TEXT => {
            let s = std::str::from_utf8(bytes)
                .map_err(|_| "text parameter is not valid UTF-8".to_string())?;
            match type_oid {
                oid::BOOL => Ok(BoundValue::Bool(parse_bool(s)?)),
                oid::INT2 | oid::INT4 | oid::INT8 => s
                    .trim()
                    .parse::<i64>()
                    .map(BoundValue::Int)
                    .map_err(|_| format!("invalid integer parameter '{s}'")),
                oid::FLOAT4 | oid::FLOAT8 => s
                    .trim()
                    .parse::<f64>()
                    .map(BoundValue::Float)
                    .map_err(|_| format!("invalid float parameter '{s}'")),
                _ => Ok(BoundValue::Text(s.to_string())),
            }
        }
        FORMAT_BINARY => match type_oid {
            oid::BOOL => {
                if bytes.len() != 1 {
                    return Err("binary bool must be exactly 1 byte".into());
                }
                Ok(BoundValue::Bool(bytes[0] != 0))
            }
            oid::INT2 => {
                let a: [u8; 2] = bytes.try_into().map_err(|_| "binary int2 must be 2 bytes".to_string())?;
                Ok(BoundValue::Int(i16::from_be_bytes(a) as i64))
            }
            oid::INT4 => {
                let a: [u8; 4] = bytes.try_into().map_err(|_| "binary int4 must be 4 bytes".to_string())?;
                Ok(BoundValue::Int(i32::from_be_bytes(a) as i64))
            }
            oid::INT8 => {
                let a: [u8; 8] = bytes.try_into().map_err(|_| "binary int8 must be 8 bytes".to_string())?;
                Ok(BoundValue::Int(i64::from_be_bytes(a)))
            }
            oid::FLOAT4 => {
                let a: [u8; 4] = bytes.try_into().map_err(|_| "binary float4 must be 4 bytes".to_string())?;
                Ok(BoundValue::Float(f32::from_be_bytes(a) as f64))
            }
            oid::FLOAT8 => {
                let a: [u8; 8] = bytes.try_into().map_err(|_| "binary float8 must be 8 bytes".to_string())?;
                Ok(BoundValue::Float(f64::from_be_bytes(a)))
            }
            _ => {
                let s = std::str::from_utf8(bytes)
                    .map_err(|_| "binary text parameter is not valid UTF-8".to_string())?;
                Ok(BoundValue::Text(s.to_string()))
            }
        },
        other => Err(format!("unsupported parameter format code {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_is_sql_null() {
        assert_eq!(param_to_sql_literal(None, 0, oid::INT4).unwrap(), "NULL");
    }

    #[test]
    fn text_int_unquoted() {
        assert_eq!(
            param_to_sql_literal(Some(b"42"), 0, oid::INT4).unwrap(),
            "42"
        );
    }

    #[test]
    fn text_string_quoted_and_escaped() {
        assert_eq!(
            param_to_sql_literal(Some(b"O'Brien"), 0, oid::TEXT).unwrap(),
            "'O''Brien'"
        );
    }

    #[test]
    fn binary_int4() {
        let b = 42i32.to_be_bytes();
        assert_eq!(
            param_to_sql_literal(Some(&b), 1, oid::INT4).unwrap(),
            "42"
        );
    }

    #[test]
    fn binary_int8_and_float8() {
        assert_eq!(
            param_to_sql_literal(Some(&1234567890123i64.to_be_bytes()), 1, oid::INT8).unwrap(),
            "1234567890123"
        );
        assert_eq!(
            param_to_sql_literal(Some(&3.5f64.to_be_bytes()), 1, oid::FLOAT8).unwrap(),
            "3.5"
        );
    }

    #[test]
    fn binary_bool() {
        assert_eq!(param_to_sql_literal(Some(&[1]), 1, oid::BOOL).unwrap(), "TRUE");
        assert_eq!(param_to_sql_literal(Some(&[0]), 1, oid::BOOL).unwrap(), "FALSE");
    }

    #[test]
    fn binary_wrong_length_errors() {
        assert!(param_to_sql_literal(Some(&[1, 2]), 1, oid::INT4).is_err());
    }

    #[test]
    fn float_without_point_gets_one() {
        // 4.0 round-trips to "4" via to_string; we re-add the point.
        assert_eq!(param_to_sql_literal(Some(&4.0f64.to_be_bytes()), 1, oid::FLOAT8).unwrap(), "4.0");
    }

    #[test]
    fn substitute_handles_double_digit_indices() {
        let lits: Vec<String> = (1..=11).map(|i| i.to_string()).collect();
        let out = substitute_parameters("VALUES ($1, $10, $11, $2)", &lits);
        assert_eq!(out, "VALUES (1, 10, 11, 2)");
    }

    #[test]
    fn substitute_string_literal() {
        let out = substitute_parameters(
            "INSERT INTO t (id, name) VALUES ($1, $2)",
            &["1".to_string(), "'alice'".to_string()],
        );
        assert_eq!(out, "INSERT INTO t (id, name) VALUES (1, 'alice')");
    }

    #[test]
    fn bound_value_text_and_binary() {
        use galaxdb_sql::BoundValue;
        assert_eq!(param_to_bound_value(None, 0, oid::INT4).unwrap(), BoundValue::Null);
        assert_eq!(
            param_to_bound_value(Some(b"42"), 0, oid::INT4).unwrap(),
            BoundValue::Int(42)
        );
        assert_eq!(
            param_to_bound_value(Some(&7i32.to_be_bytes()), 1, oid::INT4).unwrap(),
            BoundValue::Int(7)
        );
        assert_eq!(
            param_to_bound_value(Some(&3.5f64.to_be_bytes()), 1, oid::FLOAT8).unwrap(),
            BoundValue::Float(3.5)
        );
        assert_eq!(
            param_to_bound_value(Some(&[1]), 1, oid::BOOL).unwrap(),
            BoundValue::Bool(true)
        );
        assert_eq!(
            param_to_bound_value(Some(b"hi"), 0, oid::TEXT).unwrap(),
            BoundValue::Text("hi".to_string())
        );
    }
}
