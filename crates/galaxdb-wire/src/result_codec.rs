//! Result-column encoding for the extended query protocol (HTAP task 22).
//!
//! GalaxDB computes result values as text (the executor renders every
//! `Value` to a string). A `Bind` message, however, lets the client request
//! each result column in **text (0)** or **binary (1)** format. When binary
//! is requested we must send the PostgreSQL on-wire binary encoding, or a
//! binary-mode driver (e.g. tokio-postgres, which requests binary for every
//! type it knows) will mis-decode the bytes.
//!
//! This module owns two things so the RowDescription type OID and the
//! DataRow bytes always agree:
//!
//! * [`reportable_oid`] — the OID the wire is allowed to advertise for a
//!   column, given what [`encode_field`] can actually serve. Types we can
//!   encode in binary (the fixed-width scalars) and types whose binary form
//!   is byte-identical to text (text/varchar/bytea/json) keep their real
//!   OID; anything else is reported as `TEXT` so a client never requests a
//!   binary form we cannot produce.
//! * [`encode_field`] — encode one already-rendered text value to the bytes
//!   to put on the wire for a `(type_oid, format_code)` pair.

use galaxdb_sql::types::oid;

/// Result format codes (from `Bind`).
pub const FORMAT_TEXT: i16 = 0;
pub const FORMAT_BINARY: i16 = 1;

/// The type OID the wire may advertise for `oid`. Fixed-width scalars and
/// text-identical types keep their OID; everything else (numeric, uuid,
/// date/timestamp, arrays, …) is reported as `TEXT` because GalaxDB stores
/// those values as text and cannot yet emit their PostgreSQL binary form —
/// reporting the real OID would let a binary-mode client request a binary
/// encoding we cannot honor (an honest downgrade, not a silent wrong type).
pub fn reportable_oid(oid: u32) -> u32 {
    match oid {
        oid::BOOL
        | oid::INT2
        | oid::INT4
        | oid::INT8
        | oid::FLOAT4
        | oid::FLOAT8
        | oid::TEXT
        | oid::VARCHAR
        | oid::BYTEA
        | oid::JSON => oid,
        _ => oid::TEXT,
    }
}

/// Encode one result field. `text` is the executor-rendered value; `type_oid`
/// is the (already [`reportable_oid`]-downgraded) column OID; `format` is the
/// requested format code. Text format returns the UTF-8 bytes unchanged.
/// Binary format returns the PostgreSQL binary encoding for the fixed-width
/// scalars, and the raw bytes for text-identical types. A value that fails to
/// parse for its declared numeric type falls back to its text bytes rather
/// than panicking (the RowDescription still named the type; a client reading
/// binary sees the raw text — visibly wrong beats a crash, and this only
/// arises if the executor rendered a non-conforming value).
pub fn encode_field(text: &str, type_oid: u32, format: i16) -> Vec<u8> {
    if format != FORMAT_BINARY {
        return text.as_bytes().to_vec();
    }
    match type_oid {
        oid::BOOL => {
            let b = matches!(text, "t" | "true" | "TRUE" | "1");
            vec![b as u8]
        }
        oid::INT2 => match text.trim().parse::<i16>() {
            Ok(v) => v.to_be_bytes().to_vec(),
            Err(_) => text.as_bytes().to_vec(),
        },
        oid::INT4 => match text.trim().parse::<i32>() {
            Ok(v) => v.to_be_bytes().to_vec(),
            Err(_) => text.as_bytes().to_vec(),
        },
        oid::INT8 => match text.trim().parse::<i64>() {
            Ok(v) => v.to_be_bytes().to_vec(),
            Err(_) => text.as_bytes().to_vec(),
        },
        oid::FLOAT4 => match text.trim().parse::<f32>() {
            Ok(v) => v.to_be_bytes().to_vec(),
            Err(_) => text.as_bytes().to_vec(),
        },
        oid::FLOAT8 => match text.trim().parse::<f64>() {
            Ok(v) => v.to_be_bytes().to_vec(),
            Err(_) => text.as_bytes().to_vec(),
        },
        // text / varchar / bytea / json: binary form is the raw bytes.
        _ => text.as_bytes().to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_format_is_raw_bytes() {
        assert_eq!(encode_field("42", oid::INT4, FORMAT_TEXT), b"42".to_vec());
        assert_eq!(encode_field("hi", oid::TEXT, FORMAT_TEXT), b"hi".to_vec());
    }

    #[test]
    fn binary_scalars_are_network_order() {
        assert_eq!(encode_field("42", oid::INT4, FORMAT_BINARY), 42i32.to_be_bytes());
        assert_eq!(
            encode_field("42", oid::INT8, FORMAT_BINARY),
            42i64.to_be_bytes()
        );
        assert_eq!(
            encode_field("1.5", oid::FLOAT8, FORMAT_BINARY),
            1.5f64.to_be_bytes()
        );
        assert_eq!(encode_field("true", oid::BOOL, FORMAT_BINARY), vec![1u8]);
        assert_eq!(encode_field("f", oid::BOOL, FORMAT_BINARY), vec![0u8]);
    }

    #[test]
    fn binary_text_is_raw_bytes() {
        assert_eq!(encode_field("hi", oid::TEXT, FORMAT_BINARY), b"hi".to_vec());
    }

    #[test]
    fn reportable_oid_downgrades_unsupported() {
        assert_eq!(reportable_oid(oid::INT4), oid::INT4);
        assert_eq!(reportable_oid(oid::TEXT), oid::TEXT);
        // uuid/date/timestamp/numeric → downgraded to text.
        assert_eq!(reportable_oid(oid::UUID), oid::TEXT);
        assert_eq!(reportable_oid(oid::DATE), oid::TEXT);
        assert_eq!(reportable_oid(oid::NUMERIC), oid::TEXT);
    }
}
