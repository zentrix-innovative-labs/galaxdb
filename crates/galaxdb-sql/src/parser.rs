//! AuroraSQL parser — extends sqlparser-rs with GalaxDB extensions.
//!
//! Parses standard SQL via sqlparser, then detects and parses AuroraSQL
//! extensions: EMBEDDING MODEL, SEMANTIC_MATCH, AT VERSION, CREATE VERSION TAG,
//! BULK INSERT, SHOW EMBEDDING HEALTH, BACKUP TO, RESTORE FROM, ANALYZE.

use galaxdb_common::{GalaxError, GalaxResult};
use sqlparser::dialect::PostgreSqlDialect;
use sqlparser::parser::Parser;

use crate::ast::*;

/// Parse a SQL string into an AuroraSQL statement.
///
/// First checks for AuroraSQL-specific extensions. If none match,
/// falls back to standard sqlparser parsing.
pub fn parse(sql: &str) -> GalaxResult<Vec<AuroraStatement>> {
    let trimmed = sql.trim();
    if trimmed.is_empty() {
        return Err(GalaxError::SqlParse {
            position: 0,
            message: "empty SQL statement".to_string(),
        });
    }

    let upper = trimmed.to_uppercase();

    // Try AuroraSQL extensions first
    if upper.starts_with("SHOW EMBEDDING HEALTH") {
        return Ok(vec![parse_show_embedding_health(trimmed)?]);
    }
    if upper.starts_with("BACKUP TO") {
        return Ok(vec![parse_backup_to(trimmed)?]);
    }
    if upper.starts_with("RESTORE FROM") {
        return Ok(vec![parse_restore_from(trimmed)?]);
    }
    if upper.starts_with("CREATE VERSION TAG") {
        return Ok(vec![parse_create_version_tag(trimmed)?]);
    }
    if upper.starts_with("BULK INSERT") {
        return Ok(vec![parse_bulk_insert(trimmed)?]);
    }
    if upper.starts_with("ANALYZE") && !upper.starts_with("ANALYZE SELECT") {
        return Ok(vec![parse_analyze(trimmed)?]);
    }

    // Try standard sqlparser, then post-process for extensions
    let dialect = PostgreSqlDialect {};
    let statements = Parser::parse_sql(&dialect, trimmed).map_err(|e| {
        let pos = extract_error_position(&e);
        GalaxError::SqlParse {
            position: pos,
            message: format!("{}", e),
        }
    })?;

    let mut results = Vec::with_capacity(statements.len());
    for stmt in statements {
        // Check if CREATE TABLE has embedding columns
        if let sqlparser::ast::Statement::CreateTable(ref ct) = stmt {
            if let Some(aurora_ct) = try_parse_create_table_with_embedding(ct) {
                results.push(AuroraStatement::CreateTable(aurora_ct));
                continue;
            }
        }
        results.push(AuroraStatement::Standard(Box::new(stmt)));
    }

    Ok(results)
}

/// Parse SHOW EMBEDDING HEALTH [FOR table].
fn parse_show_embedding_health(sql: &str) -> GalaxResult<AuroraStatement> {
    let upper = sql.to_uppercase();
    let table = if let Some(pos) = upper.find("FOR ") {
        let rest = sql[pos + 4..].trim().trim_end_matches(';').trim();
        if rest.is_empty() {
            None
        } else {
            Some(rest.to_string())
        }
    } else {
        None
    };
    Ok(AuroraStatement::ShowEmbeddingHealth { table })
}

/// Parse BACKUP TO '/path'.
fn parse_backup_to(sql: &str) -> GalaxResult<AuroraStatement> {
    let rest = sql.trim_start_matches(|c: char| c.is_alphabetic() || c == ' ');
    // rest should be like "TO '/path'" or just "'/path'"
    let path = extract_quoted_string(rest.trim_start_matches("TO").trim_start_matches("to").trim())
        .ok_or_else(|| GalaxError::SqlParse {
            position: 10,
            message: "BACKUP TO requires a quoted path, e.g. BACKUP TO '/path'".to_string(),
        })?;
    Ok(AuroraStatement::BackupTo { path })
}

/// Parse RESTORE FROM '/path'.
fn parse_restore_from(sql: &str) -> GalaxResult<AuroraStatement> {
    let upper = sql.to_uppercase();
    let from_pos = upper.find("FROM").unwrap_or(0);
    let rest = sql[from_pos + 4..].trim();
    let path = extract_quoted_string(rest).ok_or_else(|| GalaxError::SqlParse {
        position: from_pos + 4,
        message: "RESTORE FROM requires a quoted path, e.g. RESTORE FROM '/path'".to_string(),
    })?;
    Ok(AuroraStatement::RestoreFrom { path })
}

/// Parse ANALYZE table_name.
fn parse_analyze(sql: &str) -> GalaxResult<AuroraStatement> {
    let rest = sql.trim()[7..].trim().trim_end_matches(';').trim();
    if rest.is_empty() {
        return Err(GalaxError::SqlParse {
            position: 7,
            message: "ANALYZE requires a table name".to_string(),
        });
    }
    Ok(AuroraStatement::Analyze {
        table: rest.to_string(),
    })
}

/// Parse CREATE VERSION TAG 'name' [FOR TRAINING [WITH TRAINING PRECISION ...] [TRAINING SEED n]].
fn parse_create_version_tag(sql: &str) -> GalaxResult<AuroraStatement> {
    let upper = sql.to_uppercase();
    // Skip "CREATE VERSION TAG"
    let rest = sql[18..].trim().trim_end_matches(';').trim();

    let name = extract_quoted_string(rest).ok_or_else(|| GalaxError::SqlParse {
        position: 18,
        message: "CREATE VERSION TAG requires a quoted name".to_string(),
    })?;

    let for_training = upper.contains("FOR TRAINING");

    let training_opts = if for_training {
        let precision = if upper.contains("TRAINING PRECISION") {
            if upper.contains("'SQ8'") {
                Some(TrainingPrecision::Sq8)
            } else if upper.contains("'RABITQ'") {
                Some(TrainingPrecision::Rabitq)
            } else if upper.contains("'FLOAT32'") {
                Some(TrainingPrecision::Float32)
            } else {
                None
            }
        } else {
            None
        };

        let seed = if let Some(pos) = upper.find("TRAINING SEED") {
            let seed_rest = sql[pos + 13..].trim().trim_end_matches(';').trim();
            seed_rest
                .split_whitespace()
                .next()
                .and_then(|s| s.parse::<u64>().ok())
        } else {
            None
        };

        Some(TrainingOpts { precision, seed })
    } else {
        None
    };

    Ok(AuroraStatement::CreateVersionTag(CreateVersionTagStmt {
        name,
        for_training,
        training_opts,
    }))
}

/// Parse BULK INSERT INTO table (cols) VALUES (...), (...).
fn parse_bulk_insert(sql: &str) -> GalaxResult<AuroraStatement> {
    // Simple parsing: BULK INSERT INTO table (col1, col2) VALUES (v1, v2), (v3, v4)
    let upper = sql.to_uppercase();
    let into_pos = upper.find("INTO").ok_or_else(|| GalaxError::SqlParse {
        position: 0,
        message: "BULK INSERT requires INTO clause".to_string(),
    })?;

    let rest = sql[into_pos + 4..].trim();

    // Extract table name (up to '(' or whitespace)
    let table_end = rest.find(|c: char| c == '(' || c.is_whitespace()).unwrap_or(rest.len());
    let table = rest[..table_end].trim().trim_end_matches(';').to_string();

    Ok(AuroraStatement::BulkInsert(BulkInsertStmt {
        table,
        columns: Vec::new(),
        values: Vec::new(),
    }))
}

/// Try to detect EMBEDDING MODEL in a CREATE TABLE parsed by sqlparser.
fn try_parse_create_table_with_embedding(
    ct: &sqlparser::ast::CreateTable,
) -> Option<CreateTableStmt> {
    let sql_str = format!("{}", ct);
    let upper = sql_str.to_uppercase();

    // Only intercept if there's an EMBEDDING keyword
    if !upper.contains("EMBEDDING") {
        return None;
    }

    let table_name = ct.name.to_string();
    let mut columns = Vec::new();

    for col in &ct.columns {
        let col_name = col.name.to_string();
        let data_type = format!("{}", col.data_type);

        let mut primary_key = false;
        let mut nullable = true;

        for opt in &col.options {
            match &opt.option {
                sqlparser::ast::ColumnOption::Unique { is_primary, .. } => {
                    if *is_primary {
                        primary_key = true;
                    }
                }
                sqlparser::ast::ColumnOption::NotNull => {
                    nullable = false;
                }
                _ => {}
            }
        }

        columns.push(ColumnDef {
            name: col_name,
            data_type,
            nullable,
            primary_key,
            embedding: None, // Embedding detection would need custom parsing
        });
    }

    Some(CreateTableStmt {
        table_name,
        columns,
        if_not_exists: ct.if_not_exists,
    })
}

/// Parse SEMANTIC_MATCH(col, 'query', threshold) from a string fragment.
pub fn parse_semantic_match(expr: &str) -> GalaxResult<SemanticMatchExpr> {
    let trimmed = expr.trim();
    if !trimmed.to_uppercase().starts_with("SEMANTIC_MATCH(") {
        return Err(GalaxError::SqlParse {
            position: 0,
            message: "expected SEMANTIC_MATCH(column, 'query', threshold)".to_string(),
        });
    }

    let inner = trimmed[15..].trim_end_matches(')').trim();
    let parts: Vec<&str> = split_respecting_quotes(inner);

    if parts.len() != 3 {
        return Err(GalaxError::SqlParse {
            position: 15,
            message: format!(
                "SEMANTIC_MATCH requires 3 arguments (column, 'query', threshold), got {}",
                parts.len()
            ),
        });
    }

    let column = parts[0].trim().to_string();
    let query = extract_quoted_string(parts[1].trim()).ok_or_else(|| GalaxError::SqlParse {
        position: 15,
        message: "SEMANTIC_MATCH query must be a quoted string".to_string(),
    })?;
    let threshold: f64 = parts[2].trim().parse().map_err(|_| GalaxError::SqlParse {
        position: 15,
        message: format!("invalid threshold: {}", parts[2].trim()),
    })?;

    Ok(SemanticMatchExpr {
        column,
        query,
        threshold,
    })
}

/// Parse AT VERSION timestamp_or_tag [CONSISTENCY 'mode'].
pub fn parse_at_version(expr: &str) -> GalaxResult<AtVersionExpr> {
    let upper = expr.trim().to_uppercase();
    if !upper.starts_with("AT VERSION") {
        return Err(GalaxError::SqlParse {
            position: 0,
            message: "expected AT VERSION".to_string(),
        });
    }

    let rest = expr.trim()[10..].trim();

    // Extract version ref
    let (version_str, remainder) = if let Some(stripped) = rest.strip_prefix('\'') {
        let end = stripped.find('\'').map(|p| p + 2).unwrap_or(rest.len());
        (&rest[..end], rest[end..].trim())
    } else {
        let end = rest
            .find(|c: char| c.is_whitespace())
            .unwrap_or(rest.len());
        (&rest[..end], rest[end..].trim())
    };

    let version = if let Ok(ts) = version_str.trim_matches('\'').parse::<u64>() {
        VersionRef::Timestamp(ts)
    } else {
        VersionRef::Tag(
            version_str
                .trim_matches('\'')
                .trim_matches('"')
                .to_string(),
        )
    };

    // Check for CONSISTENCY mode
    let consistency = if remainder.to_uppercase().contains("CONSISTENCY") {
        let upper_rem = remainder.to_uppercase();
        if upper_rem.contains("ROW_SNAPSHOT") || upper_rem.contains("'ROW_SNAPSHOT'") {
            Some(ConsistencyMode::RowSnapshot)
        } else if upper_rem.contains("SEMANTIC_FRESH") || upper_rem.contains("'SEMANTIC_FRESH'") {
            Some(ConsistencyMode::SemanticFresh)
        } else {
            None
        }
    } else {
        None
    };

    Ok(AtVersionExpr {
        version,
        consistency,
    })
}

// ── Helpers ────────────────────────────────────────────────────────

/// Extract a single-quoted string: 'value' → value.
fn extract_quoted_string(s: &str) -> Option<String> {
    let trimmed = s.trim().trim_end_matches(';');
    if trimmed.starts_with('\'') && trimmed.len() >= 2 {
        let end = trimmed[1..].find('\'')?;
        Some(trimmed[1..1 + end].to_string())
    } else if trimmed.starts_with('"') && trimmed.len() >= 2 {
        let end = trimmed[1..].find('"')?;
        Some(trimmed[1..1 + end].to_string())
    } else {
        None
    }
}

/// Split a string by commas, respecting quoted strings.
fn split_respecting_quotes(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut in_quote = false;
    let mut quote_char = ' ';

    for (i, c) in s.char_indices() {
        if !in_quote && (c == '\'' || c == '"') {
            in_quote = true;
            quote_char = c;
        } else if in_quote && c == quote_char {
            in_quote = false;
        } else if !in_quote && c == ',' {
            parts.push(&s[start..i]);
            start = i + 1;
        }
    }
    if start < s.len() {
        parts.push(&s[start..]);
    }
    parts
}

/// Extract error position from sqlparser error.
fn extract_error_position(e: &sqlparser::parser::ParserError) -> usize {
    let msg = format!("{}", e);
    // sqlparser errors often contain "at Line: X, Column: Y"
    if let Some(col_pos) = msg.find("Column: ") {
        let rest = &msg[col_pos + 8..];
        if let Some(end) = rest.find(|c: char| !c.is_ascii_digit()) {
            rest[..end].parse().unwrap_or(0)
        } else {
            rest.parse().unwrap_or(0)
        }
    } else {
        0
    }
}
