//! Logical SQL type system — the single source of truth that ties the
//! catalog's column types, the PostgreSQL wire type OIDs, the physical
//! storage [`ColumnType`], and value parse/format together (HTAP query
//! engine, Req 5.3).
//!
//! # Two type levels
//!
//! GalaxDB distinguishes a **logical** SQL type (what the user declares and
//! what drivers see over the wire, with a PostgreSQL type OID) from the
//! **physical** [`ColumnType`] the storage engine persists. The logical
//! type is the contract; the physical type is an implementation detail.
//!
//! Several logical types are physically encoded over an existing
//! [`ColumnType`] **losslessly** (not a silent fallback — the logical type
//! and OID are preserved in the catalog):
//!
//! | Logical `SqlType` | Physical `ColumnType` | Encoding |
//! |-------------------|-----------------------|----------|
//! | `Int2/4/8`        | `Int16/Int32/Int64`   | native |
//! | `Float4/8`        | `Float32/Float64`     | native |
//! | `Bool`            | `Boolean`             | native |
//! | `Text`/`Varchar`  | `Text`                | native |
//! | `Bytea`           | `Blob`                | native |
//! | `Json`/`Jsonb`    | `Json`                | native |
//! | `Timestamp[Tz]`   | `Int64`               | microseconds since Unix epoch |
//! | `Date`            | `Int32`               | days since Unix epoch |
//! | `Uuid`            | `Blob`                | 16 raw bytes |
//! | `Numeric`         | `Text`                | canonical decimal string |
//! | `Array(T)`        | `Blob`                | text array literal bytes |
//!
//! The `Numeric→Text` and `Array(T)→Blob` rows are **bridge** encodings:
//! correct and lossless, but a native columnar `Decimal128`/`List` physical
//! type is the end state (HTAP tasks 5/6). Upgrading them changes only
//! [`SqlType::to_column_type`] — the logical contract above is unaffected.

use galaxdb_common::{ColumnType, GalaxError, GalaxResult};

use crate::planner::Value;

/// PostgreSQL type OIDs (from `pg_type`). Single source of truth for the
/// wire layer; `galaxdb-wire`'s decoder constants should defer to these.
pub mod oid {
    // -- scalar OIDs --
    pub const BOOL: u32 = 16;
    pub const BYTEA: u32 = 17;
    pub const INT8: u32 = 20;
    pub const INT2: u32 = 21;
    pub const INT4: u32 = 23;
    pub const TEXT: u32 = 25;
    pub const JSON: u32 = 114;
    pub const FLOAT4: u32 = 700;
    pub const FLOAT8: u32 = 701;
    pub const VARCHAR: u32 = 1043;
    pub const DATE: u32 = 1082;
    pub const TIMESTAMP: u32 = 1114;
    pub const TIMESTAMPTZ: u32 = 1184;
    pub const NUMERIC: u32 = 1700;
    pub const UUID: u32 = 2950;
    pub const JSONB: u32 = 3802;

    // -- array OIDs (element type's array form) --
    pub const BOOL_ARRAY: u32 = 1000;
    pub const BYTEA_ARRAY: u32 = 1001;
    pub const INT2_ARRAY: u32 = 1005;
    pub const INT4_ARRAY: u32 = 1007;
    pub const TEXT_ARRAY: u32 = 1009;
    pub const INT8_ARRAY: u32 = 1016;
    pub const FLOAT4_ARRAY: u32 = 1021;
    pub const FLOAT8_ARRAY: u32 = 1022;
    pub const VARCHAR_ARRAY: u32 = 1015;
    pub const DATE_ARRAY: u32 = 1182;
    pub const TIMESTAMP_ARRAY: u32 = 1115;
    pub const TIMESTAMPTZ_ARRAY: u32 = 1185;
    pub const NUMERIC_ARRAY: u32 = 1231;
    pub const UUID_ARRAY: u32 = 2951;
    pub const JSON_ARRAY: u32 = 199;
    pub const JSONB_ARRAY: u32 = 3807;
}

/// A logical SQL type in the GalaxDB dialect, aligned to the core
/// PostgreSQL type set (Req 5.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SqlType {
    /// `SMALLINT` / `INT2`.
    Int2,
    /// `INTEGER` / `INT` / `INT4`.
    Int4,
    /// `BIGINT` / `INT8`.
    Int8,
    /// `REAL` / `FLOAT4`.
    Float4,
    /// `DOUBLE PRECISION` / `FLOAT8`.
    Float8,
    /// `NUMERIC` / `DECIMAL`, optional precision and scale.
    Numeric {
        /// Total significant digits, if specified.
        precision: Option<u8>,
        /// Digits after the decimal point, if specified.
        scale: Option<u8>,
    },
    /// `BOOLEAN`.
    Bool,
    /// `TEXT`.
    Text,
    /// `VARCHAR(n)` / `CHARACTER VARYING`, optional length cap.
    Varchar(Option<u32>),
    /// `BYTEA`.
    Bytea,
    /// `TIMESTAMP` (without time zone).
    Timestamp,
    /// `TIMESTAMPTZ` / `TIMESTAMP WITH TIME ZONE`.
    TimestampTz,
    /// `DATE`.
    Date,
    /// `JSON`.
    Json,
    /// `JSONB`.
    Jsonb,
    /// `UUID`.
    Uuid,
    /// 1-dimensional array of a scalar element type.
    Array(Box<SqlType>),
}

impl SqlType {
    /// Parse a catalog/DDL type name (case-insensitive) into a `SqlType`.
    ///
    /// Accepts the PostgreSQL spellings and common aliases, optional
    /// `(precision[, scale])` / `(length)` modifiers, and a trailing `[]`
    /// for a 1-D array. Returns [`GalaxError::FeatureNotSupported`]
    /// (SQLSTATE `0A000`) for an unrecognized type rather than guessing.
    pub fn from_sql_name(raw: &str) -> GalaxResult<SqlType> {
        let trimmed = raw.trim();
        // 1-D array suffix `[]` (also tolerate `ARRAY`).
        if let Some(inner) = trimmed.strip_suffix("[]") {
            let elem = SqlType::from_sql_name(inner)?;
            return Ok(SqlType::Array(Box::new(elem)));
        }
        if let Some(inner) = trimmed
            .to_ascii_uppercase()
            .strip_suffix(" ARRAY")
            .map(|_| &trimmed[..trimmed.len() - 6])
        {
            let elem = SqlType::from_sql_name(inner)?;
            return Ok(SqlType::Array(Box::new(elem)));
        }

        // Split an optional `(...)` modifier off the base name.
        let (base, modifier) = match trimmed.split_once('(') {
            Some((b, rest)) => (b.trim(), Some(rest.trim_end_matches(')'))),
            None => (trimmed, None),
        };
        let upper = base.trim().to_ascii_uppercase();

        let ty = match upper.as_str() {
            "SMALLINT" | "INT2" => SqlType::Int2,
            "INT" | "INTEGER" | "INT4" => SqlType::Int4,
            "BIGINT" | "INT8" => SqlType::Int8,
            "REAL" | "FLOAT4" => SqlType::Float4,
            "DOUBLE PRECISION" | "FLOAT8" | "DOUBLE" => SqlType::Float8,
            "FLOAT" => {
                // PG: FLOAT(p) p<=24 → float4, else float8; bare FLOAT → float8.
                match modifier.and_then(|m| m.trim().parse::<u8>().ok()) {
                    Some(p) if p <= 24 => SqlType::Float4,
                    _ => SqlType::Float8,
                }
            }
            "NUMERIC" | "DECIMAL" | "DEC" => {
                let (precision, scale) = parse_numeric_modifier(modifier);
                SqlType::Numeric { precision, scale }
            }
            "BOOL" | "BOOLEAN" => SqlType::Bool,
            "TEXT" => SqlType::Text,
            "VARCHAR" | "CHARACTER VARYING" => {
                SqlType::Varchar(modifier.and_then(|m| m.trim().parse::<u32>().ok()))
            }
            "BYTEA" => SqlType::Bytea,
            "TIMESTAMP" | "TIMESTAMP WITHOUT TIME ZONE" => SqlType::Timestamp,
            "TIMESTAMPTZ" | "TIMESTAMP WITH TIME ZONE" => SqlType::TimestampTz,
            "DATE" => SqlType::Date,
            "JSON" => SqlType::Json,
            "JSONB" => SqlType::Jsonb,
            "UUID" => SqlType::Uuid,
            other => {
                return Err(GalaxError::FeatureNotSupported(format!(
                    "column type '{other}' is not supported"
                )));
            }
        };
        Ok(ty)
    }

    /// The PostgreSQL type OID reported to wire clients.
    pub fn pg_oid(&self) -> u32 {
        match self {
            SqlType::Int2 => oid::INT2,
            SqlType::Int4 => oid::INT4,
            SqlType::Int8 => oid::INT8,
            SqlType::Float4 => oid::FLOAT4,
            SqlType::Float8 => oid::FLOAT8,
            SqlType::Numeric { .. } => oid::NUMERIC,
            SqlType::Bool => oid::BOOL,
            SqlType::Text => oid::TEXT,
            SqlType::Varchar(_) => oid::VARCHAR,
            SqlType::Bytea => oid::BYTEA,
            SqlType::Timestamp => oid::TIMESTAMP,
            SqlType::TimestampTz => oid::TIMESTAMPTZ,
            SqlType::Date => oid::DATE,
            SqlType::Json => oid::JSON,
            SqlType::Jsonb => oid::JSONB,
            SqlType::Uuid => oid::UUID,
            SqlType::Array(elem) => elem.array_oid(),
        }
    }

    /// The array OID whose element is `self` (used for `Array` reporting).
    fn array_oid(&self) -> u32 {
        match self {
            SqlType::Int2 => oid::INT2_ARRAY,
            SqlType::Int4 => oid::INT4_ARRAY,
            SqlType::Int8 => oid::INT8_ARRAY,
            SqlType::Float4 => oid::FLOAT4_ARRAY,
            SqlType::Float8 => oid::FLOAT8_ARRAY,
            SqlType::Numeric { .. } => oid::NUMERIC_ARRAY,
            SqlType::Bool => oid::BOOL_ARRAY,
            SqlType::Text => oid::TEXT_ARRAY,
            SqlType::Varchar(_) => oid::VARCHAR_ARRAY,
            SqlType::Bytea => oid::BYTEA_ARRAY,
            SqlType::Timestamp => oid::TIMESTAMP_ARRAY,
            SqlType::TimestampTz => oid::TIMESTAMPTZ_ARRAY,
            SqlType::Date => oid::DATE_ARRAY,
            SqlType::Json => oid::JSON_ARRAY,
            SqlType::Jsonb => oid::JSONB_ARRAY,
            SqlType::Uuid => oid::UUID_ARRAY,
            // Nested arrays are not represented; report the element form.
            SqlType::Array(elem) => elem.array_oid(),
        }
    }

    /// The physical [`ColumnType`] used to persist this logical type.
    /// See the module-level table; encodings are lossless.
    pub fn to_column_type(&self) -> ColumnType {
        match self {
            SqlType::Int2 => ColumnType::Int16,
            SqlType::Int4 => ColumnType::Int32,
            SqlType::Int8 => ColumnType::Int64,
            SqlType::Float4 => ColumnType::Float32,
            SqlType::Float8 => ColumnType::Float64,
            SqlType::Bool => ColumnType::Boolean,
            SqlType::Text | SqlType::Varchar(_) => ColumnType::Text,
            SqlType::Bytea | SqlType::Uuid => ColumnType::Blob,
            SqlType::Json | SqlType::Jsonb => ColumnType::Json,
            // microseconds since Unix epoch / days since Unix epoch.
            SqlType::Timestamp | SqlType::TimestampTz => ColumnType::Int64,
            SqlType::Date => ColumnType::Int32,
            // Bridge encodings (HTAP tasks 5/6 may make these native).
            SqlType::Numeric { .. } => ColumnType::Text,
            SqlType::Array(_) => ColumnType::Blob,
        }
    }

    /// Canonical lowercase type name (for `pg_catalog`/`\d` style output).
    pub fn name(&self) -> String {
        match self {
            SqlType::Int2 => "smallint".into(),
            SqlType::Int4 => "integer".into(),
            SqlType::Int8 => "bigint".into(),
            SqlType::Float4 => "real".into(),
            SqlType::Float8 => "double precision".into(),
            SqlType::Numeric { precision, scale } => match (precision, scale) {
                (Some(p), Some(s)) => format!("numeric({p},{s})"),
                (Some(p), None) => format!("numeric({p})"),
                _ => "numeric".into(),
            },
            SqlType::Bool => "boolean".into(),
            SqlType::Text => "text".into(),
            SqlType::Varchar(Some(n)) => format!("varchar({n})"),
            SqlType::Varchar(None) => "varchar".into(),
            SqlType::Bytea => "bytea".into(),
            SqlType::Timestamp => "timestamp".into(),
            SqlType::TimestampTz => "timestamptz".into(),
            SqlType::Date => "date".into(),
            SqlType::Json => "json".into(),
            SqlType::Jsonb => "jsonb".into(),
            SqlType::Uuid => "uuid".into(),
            SqlType::Array(elem) => format!("{}[]", elem.name()),
        }
    }
}

/// Parse a `NUMERIC(p[,s])` modifier string into `(precision, scale)`.
fn parse_numeric_modifier(modifier: Option<&str>) -> (Option<u8>, Option<u8>) {
    let Some(m) = modifier else {
        return (None, None);
    };
    let mut parts = m.split(',');
    let precision = parts.next().and_then(|p| p.trim().parse::<u8>().ok());
    let scale = parts.next().and_then(|s| s.trim().parse::<u8>().ok());
    (precision, scale)
}

// ---------------------------------------------------------------------------
// Value parse / format under a logical SqlType (Req 5.3 "mapped on read and
// write"). Temporal types use microseconds / days since the Unix epoch.
// ---------------------------------------------------------------------------

/// Microseconds per day.
const MICROS_PER_DAY: i64 = 86_400_000_000;

/// Parse a textual literal into a [`Value`] under the given logical type.
///
/// The empty/`NULL` sentinel is the caller's responsibility; this function
/// always produces a non-NULL value or a typed error. Returns
/// [`GalaxError::SqlParse`] when the literal does not match the type.
pub fn parse_value(s: &str, ty: &SqlType) -> GalaxResult<Value> {
    let parse_err = |msg: String| GalaxError::SqlParse { position: 0, message: msg };
    match ty {
        SqlType::Int2 | SqlType::Int4 | SqlType::Int8 => s
            .trim()
            .parse::<i64>()
            .map(Value::Integer)
            .map_err(|_| parse_err(format!("invalid integer literal '{s}'"))),
        SqlType::Float4 | SqlType::Float8 => s
            .trim()
            .parse::<f64>()
            .map(Value::Float)
            .map_err(|_| parse_err(format!("invalid float literal '{s}'"))),
        SqlType::Numeric { .. } => {
            // Validate it is a well-formed decimal, store canonical text.
            let t = s.trim();
            if t.parse::<f64>().is_err() {
                return Err(parse_err(format!("invalid numeric literal '{s}'")));
            }
            Ok(Value::Text(t.to_string()))
        }
        SqlType::Bool => match s.trim().to_ascii_lowercase().as_str() {
            "t" | "true" | "1" | "yes" | "on" => Ok(Value::Bool(true)),
            "f" | "false" | "0" | "no" | "off" => Ok(Value::Bool(false)),
            _ => Err(parse_err(format!("invalid boolean literal '{s}'"))),
        },
        SqlType::Text | SqlType::Varchar(_) | SqlType::Json | SqlType::Jsonb => {
            Ok(Value::Text(s.to_string()))
        }
        SqlType::Bytea => parse_bytea(s).map(Value::Blob).map_err(parse_err),
        SqlType::Uuid => parse_uuid(s).map(|b| Value::Blob(b.to_vec())).map_err(parse_err),
        SqlType::Date => parse_date_to_days(s).map(Value::Integer).map_err(parse_err),
        SqlType::Timestamp | SqlType::TimestampTz => {
            parse_timestamp_to_micros(s).map(Value::Integer).map_err(parse_err)
        }
        SqlType::Array(elem) => {
            let items = parse_array_elements(s)
                .map_err(&parse_err)?
                .into_iter()
                .map(|e| {
                    if e.eq_ignore_ascii_case("NULL") {
                        Ok(Value::Null)
                    } else {
                        parse_value(&e, elem)
                    }
                })
                .collect::<GalaxResult<Vec<_>>>()?;
            Ok(Value::Array(items))
        }
    }
}

/// Render a [`Value`] as the PostgreSQL text representation for `ty`.
pub fn format_value(v: &Value, ty: &SqlType) -> String {
    if matches!(v, Value::Null) {
        return "NULL".to_string();
    }
    match (ty, v) {
        (SqlType::Date, Value::Integer(days)) => format_days_as_date(*days),
        (SqlType::Timestamp | SqlType::TimestampTz, Value::Integer(micros)) => {
            format_micros_as_timestamp(*micros)
        }
        (SqlType::Uuid, Value::Blob(b)) => format_uuid(b),
        (SqlType::Bytea, Value::Blob(b)) => format_bytea(b),
        (SqlType::Array(elem), Value::Array(items)) => {
            let mut s = String::from("{");
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push_str(&format_value(item, elem));
            }
            s.push('}');
            s
        }
        // Fall back to the generic display for scalar/native encodings.
        _ => crate::row_codec::value_display(v),
    }
}

// --- bytea (PostgreSQL hex format `\xDEADBEEF`, or bare hex) ---

fn parse_bytea(s: &str) -> Result<Vec<u8>, String> {
    let hex = s.trim().strip_prefix("\\x").unwrap_or(s.trim());
    if hex.len() % 2 != 0 {
        return Err(format!("invalid bytea literal '{s}': odd hex length"));
    }
    (0..hex.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&hex[i..i + 2], 16)
                .map_err(|_| format!("invalid bytea literal '{s}'"))
        })
        .collect()
}

fn format_bytea(b: &[u8]) -> String {
    let mut s = String::with_capacity(2 + b.len() * 2);
    s.push_str("\\x");
    for byte in b {
        use std::fmt::Write;
        let _ = write!(&mut s, "{byte:02x}");
    }
    s
}

// --- uuid (canonical 8-4-4-4-12 hyphenated form) ---

fn parse_uuid(s: &str) -> Result<[u8; 16], String> {
    let clean: String = s.trim().chars().filter(|c| *c != '-').collect();
    if clean.len() != 32 {
        return Err(format!("invalid uuid literal '{s}'"));
    }
    let mut out = [0u8; 16];
    for i in 0..16 {
        out[i] = u8::from_str_radix(&clean[i * 2..i * 2 + 2], 16)
            .map_err(|_| format!("invalid uuid literal '{s}'"))?;
    }
    Ok(out)
}

fn format_uuid(b: &[u8]) -> String {
    if b.len() != 16 {
        // Not a valid UUID payload; render as hex so no data is hidden.
        return format_bytea(b);
    }
    let h: Vec<String> = b.iter().map(|x| format!("{x:02x}")).collect();
    let j = h.concat();
    format!("{}-{}-{}-{}-{}", &j[0..8], &j[8..12], &j[12..16], &j[16..20], &j[20..32])
}

// --- date / timestamp epoch math (Howard Hinnant's civil algorithms) ---

/// Days since 1970-01-01 for the given proleptic-Gregorian Y/M/D.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400; // [0, 399]
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146097 + doe - 719468
}

/// Inverse of [`days_from_civil`]: returns `(year, month, day)`.
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = z - era * 146097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn parse_date_to_days(s: &str) -> Result<i64, String> {
    let t = s.trim();
    let parts: Vec<&str> = t.split('-').collect();
    if parts.len() != 3 {
        return Err(format!("invalid date literal '{s}' (expected YYYY-MM-DD)"));
    }
    let y = parts[0].parse::<i64>().map_err(|_| format!("invalid date '{s}'"))?;
    let m = parts[1].parse::<i64>().map_err(|_| format!("invalid date '{s}'"))?;
    let d = parts[2].parse::<i64>().map_err(|_| format!("invalid date '{s}'"))?;
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return Err(format!("invalid date '{s}': month/day out of range"));
    }
    Ok(days_from_civil(y, m, d))
}

fn format_days_as_date(days: i64) -> String {
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}")
}

fn parse_timestamp_to_micros(s: &str) -> Result<i64, String> {
    let t = s.trim();
    // Accept `YYYY-MM-DD[ T]HH:MM:SS[.ffffff]`. A trailing `Z` is tolerated.
    let t = t.strip_suffix('Z').unwrap_or(t);
    let (date_part, time_part) = match t.find([' ', 'T']) {
        Some(i) => (&t[..i], &t[i + 1..]),
        None => (t, "00:00:00"),
    };
    let days = parse_date_to_days(date_part)?;
    let micros_in_day = parse_time_to_micros(time_part)?;
    Ok(days * MICROS_PER_DAY + micros_in_day)
}

fn parse_time_to_micros(s: &str) -> Result<i64, String> {
    if s.is_empty() {
        return Ok(0);
    }
    let (hms, frac) = match s.split_once('.') {
        Some((a, b)) => (a, b),
        None => (s, ""),
    };
    let parts: Vec<&str> = hms.split(':').collect();
    if parts.is_empty() || parts.len() > 3 {
        return Err(format!("invalid time literal '{s}'"));
    }
    let h = parts[0].parse::<i64>().map_err(|_| format!("invalid time '{s}'"))?;
    let mi = parts.get(1).map_or(Ok(0), |v| v.parse::<i64>()).map_err(|_| format!("invalid time '{s}'"))?;
    let se = parts.get(2).map_or(Ok(0), |v| v.parse::<i64>()).map_err(|_| format!("invalid time '{s}'"))?;
    // Fractional seconds → microseconds (pad/truncate to 6 digits).
    let mut frac6 = String::from(frac);
    frac6.truncate(6);
    while frac6.len() < 6 {
        frac6.push('0');
    }
    let micros_frac = if frac.is_empty() {
        0
    } else {
        frac6.parse::<i64>().map_err(|_| format!("invalid time fraction '{s}'"))?
    };
    Ok(((h * 60 + mi) * 60 + se) * 1_000_000 + micros_frac)
}

fn format_micros_as_timestamp(micros: i64) -> String {
    let mut days = micros.div_euclid(MICROS_PER_DAY);
    let mut rem = micros.rem_euclid(MICROS_PER_DAY);
    if rem < 0 {
        rem += MICROS_PER_DAY;
        days -= 1;
    }
    let (y, mo, d) = civil_from_days(days);
    let secs = rem / 1_000_000;
    let frac = rem % 1_000_000;
    let (h, mi, se) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    if frac == 0 {
        format!("{y:04}-{mo:02}-{d:02} {h:02}:{mi:02}:{se:02}")
    } else {
        format!("{y:04}-{mo:02}-{d:02} {h:02}:{mi:02}:{se:02}.{frac:06}")
    }
}

// --- array literal splitting (`{a,b,"c,d"}`) ---

/// Split a PostgreSQL array literal body into element strings, honoring
/// double-quoted elements (which may contain commas and escaped quotes).
fn parse_array_elements(s: &str) -> Result<Vec<String>, String> {
    let t = s.trim();
    let body = t
        .strip_prefix('{')
        .and_then(|x| x.strip_suffix('}'))
        .ok_or_else(|| format!("invalid array literal '{s}' (expected {{...}})"))?;
    let mut out = Vec::new();
    if body.trim().is_empty() {
        return Ok(out);
    }
    let mut cur = String::new();
    let mut in_quotes = false;
    let mut chars = body.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' => in_quotes = !in_quotes,
            '\\' if in_quotes => {
                if let Some(next) = chars.next() {
                    cur.push(next);
                }
            }
            ',' if !in_quotes => out.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    out.push(cur);
    Ok(out.into_iter().map(|e| e.trim().to_string()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_scalar_type_names() {
        assert_eq!(SqlType::from_sql_name("SMALLINT").unwrap(), SqlType::Int2);
        assert_eq!(SqlType::from_sql_name("integer").unwrap(), SqlType::Int4);
        assert_eq!(SqlType::from_sql_name("INT").unwrap(), SqlType::Int4);
        assert_eq!(SqlType::from_sql_name("BIGINT").unwrap(), SqlType::Int8);
        assert_eq!(SqlType::from_sql_name("real").unwrap(), SqlType::Float4);
        assert_eq!(
            SqlType::from_sql_name("DOUBLE PRECISION").unwrap(),
            SqlType::Float8
        );
        assert_eq!(SqlType::from_sql_name("BOOLEAN").unwrap(), SqlType::Bool);
        assert_eq!(SqlType::from_sql_name("TEXT").unwrap(), SqlType::Text);
        assert_eq!(SqlType::from_sql_name("BYTEA").unwrap(), SqlType::Bytea);
        assert_eq!(SqlType::from_sql_name("DATE").unwrap(), SqlType::Date);
        assert_eq!(SqlType::from_sql_name("UUID").unwrap(), SqlType::Uuid);
        assert_eq!(SqlType::from_sql_name("JSONB").unwrap(), SqlType::Jsonb);
        assert_eq!(
            SqlType::from_sql_name("TIMESTAMPTZ").unwrap(),
            SqlType::TimestampTz
        );
    }

    #[test]
    fn parse_type_modifiers() {
        assert_eq!(
            SqlType::from_sql_name("VARCHAR(255)").unwrap(),
            SqlType::Varchar(Some(255))
        );
        assert_eq!(
            SqlType::from_sql_name("NUMERIC(10,2)").unwrap(),
            SqlType::Numeric { precision: Some(10), scale: Some(2) }
        );
        assert_eq!(
            SqlType::from_sql_name("decimal").unwrap(),
            SqlType::Numeric { precision: None, scale: None }
        );
    }

    #[test]
    fn parse_array_type() {
        assert_eq!(
            SqlType::from_sql_name("int[]").unwrap(),
            SqlType::Array(Box::new(SqlType::Int4))
        );
        assert_eq!(
            SqlType::from_sql_name("TEXT[]").unwrap(),
            SqlType::Array(Box::new(SqlType::Text))
        );
    }

    #[test]
    fn unknown_type_is_typed_error() {
        let err = SqlType::from_sql_name("MONEYBAGS").unwrap_err();
        assert_eq!(err.sqlstate(), "0A000");
    }

    #[test]
    fn oid_and_physical_mappings() {
        assert_eq!(SqlType::Int4.pg_oid(), oid::INT4);
        assert_eq!(SqlType::Int4.to_column_type(), ColumnType::Int32);
        assert_eq!(SqlType::Int8.to_column_type(), ColumnType::Int64);
        assert_eq!(SqlType::Float4.to_column_type(), ColumnType::Float32);
        assert_eq!(SqlType::Timestamp.to_column_type(), ColumnType::Int64);
        assert_eq!(SqlType::Date.to_column_type(), ColumnType::Int32);
        assert_eq!(SqlType::Uuid.to_column_type(), ColumnType::Blob);
        assert_eq!(SqlType::Json.to_column_type(), ColumnType::Json);
        assert_eq!(
            SqlType::Array(Box::new(SqlType::Int4)).pg_oid(),
            oid::INT4_ARRAY
        );
    }

    #[test]
    fn int_float_bool_roundtrip() {
        assert_eq!(parse_value("42", &SqlType::Int4).unwrap(), Value::Integer(42));
        assert_eq!(parse_value("3.5", &SqlType::Float8).unwrap(), Value::Float(3.5));
        assert_eq!(parse_value("true", &SqlType::Bool).unwrap(), Value::Bool(true));
        assert_eq!(parse_value("f", &SqlType::Bool).unwrap(), Value::Bool(false));
        assert!(parse_value("notanint", &SqlType::Int4).is_err());
    }

    #[test]
    fn date_roundtrip() {
        // 1970-01-01 is day 0; 2000-01-01 is day 10957.
        assert_eq!(parse_value("1970-01-01", &SqlType::Date).unwrap(), Value::Integer(0));
        assert_eq!(
            parse_value("2000-01-01", &SqlType::Date).unwrap(),
            Value::Integer(10957)
        );
        let v = parse_value("2026-06-25", &SqlType::Date).unwrap();
        assert_eq!(format_value(&v, &SqlType::Date), "2026-06-25");
    }

    #[test]
    fn timestamp_roundtrip() {
        let v = parse_value("2026-06-25 12:30:45", &SqlType::Timestamp).unwrap();
        assert_eq!(format_value(&v, &SqlType::Timestamp), "2026-06-25 12:30:45");
        // Epoch.
        assert_eq!(
            parse_value("1970-01-01 00:00:00", &SqlType::Timestamp).unwrap(),
            Value::Integer(0)
        );
        // Sub-second precision preserved.
        let v2 = parse_value("2026-06-25 12:30:45.123456", &SqlType::Timestamp).unwrap();
        assert_eq!(
            format_value(&v2, &SqlType::Timestamp),
            "2026-06-25 12:30:45.123456"
        );
    }

    #[test]
    fn uuid_roundtrip() {
        let s = "550e8400-e29b-41d4-a716-446655440000";
        let v = parse_value(s, &SqlType::Uuid).unwrap();
        match &v {
            Value::Blob(b) => assert_eq!(b.len(), 16),
            _ => panic!("expected blob"),
        }
        assert_eq!(format_value(&v, &SqlType::Uuid), s);
    }

    #[test]
    fn bytea_roundtrip() {
        let v = parse_value("\\xdeadbeef", &SqlType::Bytea).unwrap();
        assert_eq!(v, Value::Blob(vec![0xde, 0xad, 0xbe, 0xef]));
        assert_eq!(format_value(&v, &SqlType::Bytea), "\\xdeadbeef");
    }

    #[test]
    fn numeric_lossless_text() {
        let v = parse_value("12345.6789", &SqlType::Numeric { precision: None, scale: None })
            .unwrap();
        assert_eq!(v, Value::Text("12345.6789".to_string()));
        assert!(parse_value("abc", &SqlType::Numeric { precision: None, scale: None }).is_err());
    }

    #[test]
    fn array_roundtrip() {
        let ty = SqlType::Array(Box::new(SqlType::Int4));
        let v = parse_value("{1,2,3}", &ty).unwrap();
        assert_eq!(
            v,
            Value::Array(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)])
        );
        assert_eq!(format_value(&v, &ty), "{1,2,3}");

        let tty = SqlType::Array(Box::new(SqlType::Text));
        let tv = parse_value("{alice,bob}", &tty).unwrap();
        assert_eq!(format_value(&tv, &tty), "{alice,bob}");
    }

    #[test]
    fn array_with_null_element() {
        let ty = SqlType::Array(Box::new(SqlType::Int4));
        let v = parse_value("{1,NULL,3}", &ty).unwrap();
        assert_eq!(
            v,
            Value::Array(vec![Value::Integer(1), Value::Null, Value::Integer(3)])
        );
    }

    #[test]
    fn type_display_names() {
        assert_eq!(SqlType::Int8.name(), "bigint");
        assert_eq!(SqlType::Varchar(Some(10)).name(), "varchar(10)");
        assert_eq!(
            SqlType::Numeric { precision: Some(10), scale: Some(2) }.name(),
            "numeric(10,2)"
        );
        assert_eq!(SqlType::Array(Box::new(SqlType::Int4)).name(), "integer[]");
    }
}
