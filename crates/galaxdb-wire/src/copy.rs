//! COPY command parsing and PostgreSQL text-format row codec (Req 8).
//!
//! The wire server detects a `COPY ... FROM STDIN` / `COPY ... TO STDOUT`
//! statement on the simple-query path and drives the copy sub-protocol.
//! This module owns the lightweight command parse and the text-format
//! encode/decode so the server stays focused on message flow.
//!
//! Text format (the PostgreSQL default, Req 8 AC4): rows are newline
//! separated, columns are tab separated, `\N` is SQL NULL, and the escape
//! sequences `\\ \t \n \r` are recognised. Binary format is not
//! implemented (the spec marks it optional).

/// Direction of a COPY statement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopyDirection {
    /// `COPY t FROM STDIN` — client streams rows to the server.
    In,
    /// `COPY t TO STDOUT` — server streams rows to the client.
    Out,
}

/// A parsed COPY command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CopyCommand {
    /// Target table.
    pub table: String,
    /// Explicit column list, or empty for all columns in catalog order.
    pub columns: Vec<String>,
    /// Copy direction.
    pub direction: CopyDirection,
}

/// Detect and parse a `COPY ... FROM STDIN` / `COPY ... TO STDOUT`
/// statement. Returns `None` if `sql` is not a COPY-with-STDIN/STDOUT
/// statement (e.g. `COPY ... FROM '/file'`, which is not the wire
/// sub-protocol). Case-insensitive on keywords.
pub fn parse_copy(sql: &str) -> Option<CopyCommand> {
    let trimmed = sql.trim().trim_end_matches(';').trim();
    let upper = trimmed.to_uppercase();
    if !upper.starts_with("COPY ") {
        return None;
    }

    let direction = if upper.contains(" FROM STDIN") {
        CopyDirection::In
    } else if upper.contains(" TO STDOUT") {
        CopyDirection::Out
    } else {
        // COPY to/from a file path is not the wire sub-protocol.
        return None;
    };

    // Slice between `COPY` and the FROM/TO keyword is "table [(cols)]".
    let after_copy = trimmed[4..].trim_start();
    let kw_pos = match direction {
        CopyDirection::In => upper.find(" FROM STDIN")?,
        CopyDirection::Out => upper.find(" TO STDOUT")?,
    };
    // Recompute the target slice in the original (non-uppercased) text.
    let target = trimmed[4..kw_pos].trim();
    let _ = after_copy;

    let (table, columns) = if let Some(paren) = target.find('(') {
        let table = target[..paren].trim().to_string();
        let close = target.rfind(')')?;
        if close < paren {
            return None;
        }
        let cols = target[paren + 1..close]
            .split(',')
            .map(|c| c.trim().to_string())
            .filter(|c| !c.is_empty())
            .collect();
        (table, cols)
    } else {
        (target.trim().to_string(), Vec::new())
    };

    if table.is_empty() {
        return None;
    }
    Some(CopyCommand {
        table,
        columns,
        direction,
    })
}

/// Decode a single text-format COPY data line into its column cell
/// tokens, ready for `value_from_str`. `\N` becomes the `NULL` token; the
/// escape sequences `\\ \t \n \r` are unescaped. A trailing `\r` (CRLF) is
/// stripped by the caller before this is invoked.
pub fn decode_text_row(line: &str) -> Vec<String> {
    line.split('\t').map(decode_text_cell).collect()
}

fn decode_text_cell(cell: &str) -> String {
    if cell == "\\N" {
        return "NULL".to_string();
    }
    let mut out = String::with_capacity(cell.len());
    let mut chars = cell.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('t') => out.push('\t'),
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some('\\') => out.push('\\'),
                Some(other) => {
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Encode one row's display cells into a text-format COPY line (no
/// trailing newline). A cell equal to the `NULL` sentinel is emitted as
/// `\N`; tab/newline/carriage-return/backslash are escaped.
pub fn encode_text_row(cells: &[(bool, &str)]) -> String {
    // `cells` is (is_null, display_text). The server marks NULLs.
    let mut parts = Vec::with_capacity(cells.len());
    for (is_null, text) in cells {
        if *is_null {
            parts.push("\\N".to_string());
        } else {
            parts.push(encode_text_cell(text));
        }
    }
    parts.join("\t")
}

fn encode_text_cell(cell: &str) -> String {
    let mut out = String::with_capacity(cell.len());
    for c in cell.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_copy_from_stdin_with_columns() {
        let c = parse_copy("COPY products (id, name, price) FROM STDIN").unwrap();
        assert_eq!(c.table, "products");
        assert_eq!(c.columns, vec!["id", "name", "price"]);
        assert_eq!(c.direction, CopyDirection::In);
    }

    #[test]
    fn parse_copy_to_stdout_all_columns() {
        let c = parse_copy("COPY products TO STDOUT").unwrap();
        assert_eq!(c.table, "products");
        assert!(c.columns.is_empty());
        assert_eq!(c.direction, CopyDirection::Out);
    }

    #[test]
    fn parse_copy_case_insensitive_and_semicolon() {
        let c = parse_copy("copy t from stdin;").unwrap();
        assert_eq!(c.table, "t");
        assert_eq!(c.direction, CopyDirection::In);
    }

    #[test]
    fn non_stdin_copy_is_ignored() {
        assert!(parse_copy("COPY t FROM '/tmp/data.csv'").is_none());
        assert!(parse_copy("SELECT 1").is_none());
    }

    #[test]
    fn decode_row_tabs_null_and_escapes() {
        let row = decode_text_row("1\thello\t\\N\twith\\ttab");
        assert_eq!(row, vec!["1", "hello", "NULL", "with\ttab"]);
    }

    #[test]
    fn encode_row_round_trips_through_decode() {
        let line = encode_text_row(&[(false, "1"), (false, "a\tb"), (true, ""), (false, "x")]);
        // NULL → \N, embedded tab escaped.
        assert_eq!(line, "1\ta\\tb\t\\N\tx");
        let decoded = decode_text_row(&line);
        assert_eq!(decoded, vec!["1", "a\tb", "NULL", "x"]);
    }
}
