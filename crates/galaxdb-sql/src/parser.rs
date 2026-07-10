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
    // Task 38.5: `sql.parse` span over the parser entry point so
    // OTel backends see per-query parse duration as a child of the
    // `query.execute` root span.
    let _span = tracing::info_span!("sql.parse", bytes = sql.len()).entered();

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
    if upper.starts_with("CREATE ROLE") {
        return Ok(vec![parse_create_role(trimmed)?]);
    }
    if upper.starts_with("DROP ROLE") {
        return Ok(vec![parse_drop_role(trimmed)?]);
    }
    if upper.starts_with("ALTER TABLE") {
        return Ok(vec![parse_alter_table_set_storage(trimmed)?]);
    }
    if upper.starts_with("ALTER ROLE") {
        return Ok(vec![parse_alter_role(trimmed)?]);
    }
    if upper.starts_with("GRANT ") {
        return Ok(vec![parse_grant(trimmed, false)?]);
    }
    if upper.starts_with("REVOKE ") {
        return Ok(vec![parse_grant(trimmed, true)?]);
    }
    if upper.starts_with("CREATE SEMANTIC CACHE") {
        return Ok(vec![parse_create_semantic_cache(trimmed)?]);
    }
    if upper.starts_with("DROP SEMANTIC CACHE") {
        return Ok(vec![parse_drop_semantic_cache(trimmed)?]);
    }
    if upper.starts_with("CREATE INDEX") || upper.starts_with("CREATE UNIQUE INDEX") {
        return Ok(vec![parse_create_index(trimmed)?]);
    }
    if upper.starts_with("DROP INDEX") {
        return Ok(vec![parse_drop_index(trimmed)?]);
    }

    // Try standard sqlparser, then post-process for extensions
    // If CREATE TABLE contains EMBEDDING, pre-process to strip it before sqlparser
    if upper.starts_with("CREATE TABLE") && upper.contains("EMBEDDING") {
        return Ok(vec![parse_create_table_with_embedding(trimmed)?]);
    }

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

/// Strip an optional trailing `;` and surrounding whitespace.
fn trim_stmt(s: &str) -> &str {
    s.trim().trim_end_matches(';').trim()
}

/// Parse a bare or quoted SQL identifier token (role/table name) from the
/// front of `s`, returning `(identifier, rest)`. Quoted identifiers use
/// double quotes; bare identifiers run until whitespace.
fn take_ident(s: &str) -> Option<(String, &str)> {
    let s = s.trim_start();
    if let Some(stripped) = s.strip_prefix('"') {
        let end = stripped.find('"')?;
        let ident = stripped[..end].to_string();
        Some((ident, &stripped[end + 1..]))
    } else {
        let end = s.find(char::is_whitespace).unwrap_or(s.len());
        if end == 0 {
            return None;
        }
        Some((s[..end].to_string(), &s[end..]))
    }
}

/// Parse `CREATE ROLE name [WITH] [PASSWORD '...'] [SUPERUSER]`.
fn parse_create_role(sql: &str) -> GalaxResult<AuroraStatement> {
    // Skip "CREATE ROLE".
    let rest = trim_stmt(&sql[11..]);
    let upper = rest.to_uppercase();

    let (name, _after_name) = take_ident(rest).ok_or_else(|| GalaxError::SqlParse {
        position: 11,
        message: "CREATE ROLE requires a role name".to_string(),
    })?;

    // PASSWORD '...' is optional; extract the quoted literal that follows
    // the PASSWORD keyword if present.
    let password = if let Some(pos) = upper.find("PASSWORD") {
        let after = rest[pos + "PASSWORD".len()..].trim_start();
        Some(
            extract_quoted_string(after).ok_or_else(|| GalaxError::SqlParse {
                position: pos,
                message: "PASSWORD requires a quoted string, e.g. PASSWORD 'secret'"
                    .to_string(),
            })?,
        )
    } else {
        None
    };

    let is_superuser = upper.contains("SUPERUSER");

    Ok(AuroraStatement::CreateRole(CreateRoleStmt {
        name,
        password,
        is_superuser,
    }))
}

/// Parse `DROP ROLE [IF EXISTS] name`.
fn parse_drop_role(sql: &str) -> GalaxResult<AuroraStatement> {
    let rest = trim_stmt(&sql[9..]); // skip "DROP ROLE"
    let upper = rest.to_uppercase();
    let (if_exists, name_part) = if upper.starts_with("IF EXISTS") {
        (true, rest["IF EXISTS".len()..].trim_start())
    } else {
        (false, rest)
    };
    let (name, _) = take_ident(name_part).ok_or_else(|| GalaxError::SqlParse {
        position: 9,
        message: "DROP ROLE requires a role name".to_string(),
    })?;
    Ok(AuroraStatement::DropRole { name, if_exists })
}

/// Parse `ALTER ROLE name [WITH] PASSWORD '...'`.
fn parse_alter_role(sql: &str) -> GalaxResult<AuroraStatement> {
    let rest = trim_stmt(&sql[10..]); // skip "ALTER ROLE"
    let upper = rest.to_uppercase();
    let (name, _) = take_ident(rest).ok_or_else(|| GalaxError::SqlParse {
        position: 10,
        message: "ALTER ROLE requires a role name".to_string(),
    })?;
    let pos = upper.find("PASSWORD").ok_or_else(|| GalaxError::SqlParse {
        position: 10,
        message: "ALTER ROLE currently supports only PASSWORD '...'".to_string(),
    })?;
    let after = rest[pos + "PASSWORD".len()..].trim_start();
    let password = extract_quoted_string(after).ok_or_else(|| GalaxError::SqlParse {
        position: pos,
        message: "ALTER ROLE ... PASSWORD requires a quoted string".to_string(),
    })?;
    Ok(AuroraStatement::AlterRolePassword { name, password })
}

/// Parse `ALTER TABLE <name> SET STORAGE {COLUMNAR|LEGACY|ROW}` (a GalaxDB
/// extension — `SET STORAGE` at table scope, distinct from PostgreSQL's
/// per-column `ALTER COLUMN ... SET STORAGE`). `ROW` is an alias for
/// `LEGACY`. HTAP task 9.
fn parse_alter_table_set_storage(sql: &str) -> GalaxResult<AuroraStatement> {
    let rest = trim_stmt(&sql["ALTER TABLE".len()..]);
    let (table, after_name) = take_ident(rest).ok_or_else(|| GalaxError::SqlParse {
        position: 0,
        message: "ALTER TABLE requires a table name".to_string(),
    })?;
    let tail = after_name.trim();
    let upper = tail.to_uppercase();
    let kw = upper
        .strip_prefix("SET STORAGE")
        .map(|s| s.trim())
        .ok_or_else(|| GalaxError::SqlParse {
            position: 0,
            message: "ALTER TABLE currently supports only SET STORAGE \
                      {COLUMNAR|LEGACY|ROW}"
                .to_string(),
        })?;
    let mode = match kw {
        "COLUMNAR" => galaxdb_common::StorageMode::Columnar,
        "LEGACY" | "ROW" => galaxdb_common::StorageMode::Legacy,
        other => {
            return Err(GalaxError::SqlParse {
                position: 0,
                message: format!(
                    "unknown storage mode {other:?}; expected COLUMNAR, LEGACY, or ROW"
                ),
            })
        }
    };
    Ok(AuroraStatement::AlterTableSetStorage { table, mode })
}

/// Parse `GRANT priv ON table TO role` / `REVOKE priv ON table FROM role`.
fn parse_grant(sql: &str, revoke: bool) -> GalaxResult<AuroraStatement> {
    // Skip leading GRANT / REVOKE.
    let keyword_len = if revoke { "REVOKE".len() } else { "GRANT".len() };
    let rest = trim_stmt(&sql[keyword_len..]);
    let upper = rest.to_uppercase();

    // <priv> ON <table> (TO|FROM) <role>
    let on_pos = upper.find(" ON ").ok_or_else(|| GalaxError::SqlParse {
        position: 0,
        message: "expected ON in GRANT/REVOKE".to_string(),
    })?;
    let priv_str = rest[..on_pos].trim();
    let privilege = match priv_str.to_uppercase().as_str() {
        "SELECT" => Privilege::Select,
        "INSERT" => Privilege::Insert,
        "UPDATE" => Privilege::Update,
        "DELETE" => Privilege::Delete,
        other => {
            return Err(GalaxError::SqlParse {
                position: 0,
                message: format!(
                    "unsupported privilege '{}'; expected SELECT/INSERT/UPDATE/DELETE",
                    other
                ),
            });
        }
    };

    let after_on = &rest[on_pos + 4..];
    // table (TO|FROM) role
    let conn_upper = if revoke { "FROM" } else { "TO" };
    let (table_part, role_part) =
        split_on_keyword(after_on, conn_upper).ok_or_else(|| GalaxError::SqlParse {
            position: on_pos,
            message: format!("expected {} in GRANT/REVOKE", conn_upper),
        })?;

    let (table, _) = take_ident(table_part.trim()).ok_or_else(|| GalaxError::SqlParse {
        position: on_pos,
        message: "GRANT/REVOKE requires a table name after ON".to_string(),
    })?;
    let (role, _) = take_ident(role_part.trim()).ok_or_else(|| GalaxError::SqlParse {
        position: on_pos,
        message: "GRANT/REVOKE requires a role name".to_string(),
    })?;

    let stmt = GrantStmt {
        privilege,
        table,
        role,
    };
    if revoke {
        Ok(AuroraStatement::Revoke(stmt))
    } else {
        Ok(AuroraStatement::Grant(stmt))
    }
}

/// Split `s` on the first whitespace-delimited occurrence of `keyword`
/// (case-insensitive), returning `(before, after)`.
fn split_on_keyword<'a>(s: &'a str, keyword: &str) -> Option<(&'a str, &'a str)> {
    let upper = s.to_uppercase();
    let ku = keyword.to_uppercase();
    let mut search_from = 0;
    while let Some(rel) = upper[search_from..].find(&ku) {
        let pos = search_from + rel;
        let before_ok = pos == 0
            || upper.as_bytes()[pos - 1].is_ascii_whitespace();
        let after_idx = pos + ku.len();
        let after_ok = after_idx == upper.len()
            || upper.as_bytes()[after_idx].is_ascii_whitespace();
        if before_ok && after_ok {
            return Some((&s[..pos], &s[after_idx..]));
        }
        search_from = pos + ku.len();
    }
    None
}

/// Parse `CREATE [UNIQUE] INDEX [IF NOT EXISTS] name ON table (column)`.
///
/// The `UNIQUE` keyword is accepted for PostgreSQL syntax compatibility
/// but uniqueness enforcement is not part of this phase; the index is
/// built as a non-unique secondary index regardless (a uniqueness
/// constraint check is a separate, larger feature). Only a single column
/// is supported.
fn parse_create_index(sql: &str) -> GalaxResult<AuroraStatement> {
    let upper = sql.to_uppercase();
    // Locate the INDEX keyword (after optional UNIQUE).
    let after_index = {
        let pos = upper.find("INDEX").ok_or_else(|| GalaxError::SqlParse {
            position: 0,
            message: "expected INDEX in CREATE INDEX".to_string(),
        })?;
        trim_stmt(&sql[pos + "INDEX".len()..])
    };

    // Optional IF NOT EXISTS.
    let au = after_index.to_uppercase();
    let (if_not_exists, after_ine) = if au.starts_with("IF NOT EXISTS") {
        (true, after_index["IF NOT EXISTS".len()..].trim_start())
    } else {
        (false, after_index)
    };

    // name ON table (column)
    let (name, after_name) =
        take_ident(after_ine).ok_or_else(|| GalaxError::SqlParse {
            position: 0,
            message: "CREATE INDEX requires an index name".to_string(),
        })?;

    let (_before_on, after_on) =
        split_on_keyword(after_name, "ON").ok_or_else(|| GalaxError::SqlParse {
            position: 0,
            message: "CREATE INDEX requires ON <table> (<column>)".to_string(),
        })?;

    // table (column)
    let paren = after_on.find('(').ok_or_else(|| GalaxError::SqlParse {
        position: 0,
        message: "CREATE INDEX requires a parenthesised column, e.g. ON t (col)".to_string(),
    })?;
    let (table, _) = take_ident(after_on[..paren].trim()).ok_or_else(|| GalaxError::SqlParse {
        position: 0,
        message: "CREATE INDEX requires a table name".to_string(),
    })?;
    let close = after_on.find(')').ok_or_else(|| GalaxError::SqlParse {
        position: 0,
        message: "CREATE INDEX column list is missing a closing ')'".to_string(),
    })?;
    if close < paren {
        return Err(GalaxError::SqlParse {
            position: 0,
            message: "CREATE INDEX has malformed parentheses".to_string(),
        });
    }
    let inner = after_on[paren + 1..close].trim();
    if inner.contains(',') {
        return Err(GalaxError::SqlParse {
            position: 0,
            message: "multi-column indexes are not supported in this phase; index one column"
                .to_string(),
        });
    }
    let (column, _) = take_ident(inner).ok_or_else(|| GalaxError::SqlParse {
        position: 0,
        message: "CREATE INDEX requires a column name in parentheses".to_string(),
    })?;

    Ok(AuroraStatement::CreateIndex(CreateIndexStmt {
        name,
        table,
        column,
        if_not_exists,
    }))
}

/// Parse `DROP INDEX [IF EXISTS] name`.
fn parse_drop_index(sql: &str) -> GalaxResult<AuroraStatement> {
    let rest = trim_stmt(&sql[ "DROP INDEX".len()..]); // skip "DROP INDEX"
    let upper = rest.to_uppercase();
    let (if_exists, name_part) = if upper.starts_with("IF EXISTS") {
        (true, rest["IF EXISTS".len()..].trim_start())
    } else {
        (false, rest)
    };
    let (name, _) = take_ident(name_part).ok_or_else(|| GalaxError::SqlParse {
        position: 0,
        message: "DROP INDEX requires an index name".to_string(),
    })?;
    Ok(AuroraStatement::DropIndex { name, if_exists })
}

/// Parse `CREATE SEMANTIC CACHE FOR TABLE <t> SIMILARITY <f> TTL <n>` (v0.7).
fn parse_create_semantic_cache(sql: &str) -> GalaxResult<AuroraStatement> {
    // Normalize: strip trailing ';' and collapse whitespace for keyword scans.
    let s = sql.trim().trim_end_matches(';').trim();
    let upper = s.to_uppercase();

    let for_table = "CREATE SEMANTIC CACHE FOR TABLE";
    if !upper.starts_with(for_table) {
        return Err(GalaxError::SqlParse {
            position: 0,
            message: "expected CREATE SEMANTIC CACHE FOR TABLE <table> SIMILARITY <f> TTL <n>"
                .to_string(),
        });
    }
    let rest = s[for_table.len()..].trim();
    // table name is the first identifier.
    let (table, after_table) = take_ident(rest).ok_or_else(|| GalaxError::SqlParse {
        position: 0,
        message: "CREATE SEMANTIC CACHE requires a table name".to_string(),
    })?;

    let after_upper = after_table.to_uppercase();
    let sim_pos = after_upper.find("SIMILARITY").ok_or_else(|| GalaxError::SqlParse {
        position: 0,
        message: "CREATE SEMANTIC CACHE requires SIMILARITY <float>".to_string(),
    })?;
    let ttl_pos = after_upper.find("TTL").ok_or_else(|| GalaxError::SqlParse {
        position: 0,
        message: "CREATE SEMANTIC CACHE requires TTL <seconds>".to_string(),
    })?;
    if ttl_pos < sim_pos {
        return Err(GalaxError::SqlParse {
            position: 0,
            message: "CREATE SEMANTIC CACHE expects SIMILARITY before TTL".to_string(),
        });
    }
    let sim_str = after_table[sim_pos + "SIMILARITY".len()..ttl_pos].trim();
    let ttl_str = after_table[ttl_pos + "TTL".len()..].trim();

    let similarity: f32 = sim_str.split_whitespace().next().unwrap_or("").parse().map_err(|_| {
        GalaxError::SqlParse {
            position: 0,
            message: format!("SIMILARITY must be a float in (0.0, 1.0]; got '{sim_str}'"),
        }
    })?;
    if !(similarity > 0.0 && similarity <= 1.0) {
        return Err(GalaxError::SqlParse {
            position: 0,
            message: format!("SIMILARITY must be in (0.0, 1.0]; got {similarity}"),
        });
    }
    let ttl_secs: u32 = ttl_str.split_whitespace().next().unwrap_or("").parse().map_err(|_| {
        GalaxError::SqlParse {
            position: 0,
            message: format!("TTL must be a positive integer number of seconds; got '{ttl_str}'"),
        }
    })?;
    if ttl_secs == 0 {
        return Err(GalaxError::SqlParse {
            position: 0,
            message: "TTL must be a positive number of seconds".to_string(),
        });
    }

    Ok(AuroraStatement::CreateSemanticCache(
        crate::ast::CreateSemanticCacheStmt {
            table,
            similarity,
            ttl_secs,
        },
    ))
}

/// Parse `DROP SEMANTIC CACHE FOR TABLE <t>` (v0.7).
fn parse_drop_semantic_cache(sql: &str) -> GalaxResult<AuroraStatement> {
    let s = sql.trim().trim_end_matches(';').trim();
    let upper = s.to_uppercase();
    let prefix = "DROP SEMANTIC CACHE FOR TABLE";
    if !upper.starts_with(prefix) {
        return Err(GalaxError::SqlParse {
            position: 0,
            message: "expected DROP SEMANTIC CACHE FOR TABLE <table>".to_string(),
        });
    }
    let rest = s[prefix.len()..].trim();
    let (table, _) = take_ident(rest).ok_or_else(|| GalaxError::SqlParse {
        position: 0,
        message: "DROP SEMANTIC CACHE requires a table name".to_string(),
    })?;
    Ok(AuroraStatement::DropSemanticCache { table })
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

/// Parse `BULK INSERT INTO table (col1, col2) VALUES (v1, v2), (v3, v4)`.
///
/// Real implementation — not a stub. Extracts the table name, the
/// column list (optional), and every row in the VALUES clause. Values
/// are stored as raw string tokens so the executor can parse them
/// per-column against the catalog's typed schema. This mirrors the
/// payload format `sqlparser` would emit for a regular INSERT and
/// keeps the BULK INSERT path alignable with the standard INSERT path
/// at execution time (task 18.7 closed in Phase L).
fn parse_bulk_insert(sql: &str) -> GalaxResult<AuroraStatement> {
    let upper = sql.to_uppercase();
    let into_pos = upper.find("INTO").ok_or_else(|| GalaxError::SqlParse {
        position: 0,
        message: "BULK INSERT requires INTO clause".to_string(),
    })?;

    let rest = sql[into_pos + 4..].trim();

    // Table name up to '(' or whitespace.
    let table_end = rest.find(|c: char| c == '(' || c.is_whitespace()).unwrap_or(rest.len());
    let table = rest[..table_end].trim().trim_end_matches(';').to_string();
    if table.is_empty() {
        return Err(GalaxError::SqlParse {
            position: into_pos + 4,
            message: "BULK INSERT requires a table name after INTO".to_string(),
        });
    }

    // Optional column list.
    let mut after_table = rest[table_end..].trim_start();
    let columns = if after_table.starts_with('(') {
        let close = after_table
            .find(')')
            .ok_or_else(|| GalaxError::SqlParse {
                position: into_pos + 4,
                message: "unterminated column list in BULK INSERT".to_string(),
            })?;
        let inside = &after_table[1..close];
        let cols: Vec<String> = inside
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        after_table = after_table[close + 1..].trim_start();
        cols
    } else {
        Vec::new()
    };

    // VALUES keyword.
    let upper_rest = after_table.to_uppercase();
    if !upper_rest.starts_with("VALUES") {
        return Err(GalaxError::SqlParse {
            position: 0,
            message: "BULK INSERT requires a VALUES clause".to_string(),
        });
    }
    let mut rows_text = after_table[6..].trim().trim_end_matches(';').to_string();
    if rows_text.is_empty() {
        return Err(GalaxError::SqlParse {
            position: 0,
            message: "BULK INSERT VALUES clause is empty".to_string(),
        });
    }

    // Split into value tuples, respecting quotes and parentheses.
    let mut values: Vec<Vec<String>> = Vec::new();
    while !rows_text.is_empty() {
        let trimmed = rows_text.trim_start_matches(|c: char| c == ',' || c.is_whitespace());
        if trimmed.is_empty() {
            break;
        }
        if !trimmed.starts_with('(') {
            return Err(GalaxError::SqlParse {
                position: 0,
                message: format!(
                    "expected `(` at start of VALUES tuple, got: {:.40}",
                    trimmed
                ),
            });
        }
        let (tuple, remainder) = slice_balanced_paren(trimmed).ok_or_else(|| {
            GalaxError::SqlParse {
                position: 0,
                message: "unterminated VALUES tuple in BULK INSERT".to_string(),
            }
        })?;
        let row: Vec<String> = split_respecting_quotes(&tuple[1..tuple.len() - 1])
            .into_iter()
            .map(|s| s.trim().to_string())
            .collect();
        if !columns.is_empty() && row.len() != columns.len() {
            return Err(GalaxError::SqlParse {
                position: 0,
                message: format!(
                    "BULK INSERT row has {} values but column list has {}",
                    row.len(),
                    columns.len()
                ),
            });
        }
        values.push(row);
        rows_text = remainder.trim().to_string();
    }
    if values.is_empty() {
        return Err(GalaxError::SqlParse {
            position: 0,
            message: "BULK INSERT VALUES clause produced zero rows".to_string(),
        });
    }

    Ok(AuroraStatement::BulkInsert(BulkInsertStmt {
        table,
        columns,
        values,
    }))
}

/// Slice off the next balanced `(..)` group from the start of `s`.
/// Returns `(tuple_including_parens, remainder)`. Returns `None` if
/// the parens are unbalanced.
fn slice_balanced_paren(s: &str) -> Option<(String, String)> {
    let bytes = s.as_bytes();
    if bytes.is_empty() || bytes[0] != b'(' {
        return None;
    }
    let mut depth = 0i32;
    let mut in_quote = false;
    let mut quote_char = 0u8;
    for (i, &c) in bytes.iter().enumerate() {
        if in_quote {
            if c == quote_char {
                in_quote = false;
            }
            continue;
        }
        if c == b'\'' || c == b'"' {
            in_quote = true;
            quote_char = c;
            continue;
        }
        if c == b'(' {
            depth += 1;
        } else if c == b')' {
            depth -= 1;
            if depth == 0 {
                return Some((s[..=i].to_string(), s[i + 1..].to_string()));
            }
        }
    }
    None
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

    // Anything after the version specifier must be a `CONSISTENCY '<mode>'`
    // clause and nothing else. `AT VERSION` is the final clause of a query, so
    // a trailing `WHERE`/`ORDER BY`/`LIMIT` here means the user wrote them in
    // the wrong order — reject with a clear error rather than silently
    // dropping the predicate (which would return an unfiltered result).
    let upper_rem = remainder.trim().to_uppercase();
    let consistency = if upper_rem.is_empty() {
        None
    } else if upper_rem.starts_with("CONSISTENCY") {
        // Check SEMANTIC_SNAPSHOT before SEMANTIC_FRESH is irrelevant (distinct
        // substrings); order chosen for readability.
        if upper_rem.contains("SEMANTIC_SNAPSHOT") {
            Some(ConsistencyMode::SemanticSnapshot)
        } else if upper_rem.contains("ROW_SNAPSHOT") {
            Some(ConsistencyMode::RowSnapshot)
        } else if upper_rem.contains("SEMANTIC_FRESH") {
            Some(ConsistencyMode::SemanticFresh)
        } else {
            return Err(GalaxError::SqlParse {
                position: 0,
                message: format!(
                    "unrecognized CONSISTENCY mode in AT VERSION clause: '{}' \
                     (expected 'ROW_SNAPSHOT', 'SEMANTIC_FRESH', or 'SEMANTIC_SNAPSHOT')",
                    remainder.trim()
                ),
            });
        }
    } else {
        return Err(GalaxError::SqlParse {
            position: 0,
            message: format!(
                "unexpected tokens after the AT VERSION specifier: '{}'. \
                 AT VERSION must be the final clause of the query — put \
                 WHERE / ORDER BY / LIMIT before it, e.g. \
                 SELECT ... FROM t WHERE ... AT VERSION '<tag|timestamp>'",
                remainder.trim()
            ),
        });
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

/// Parse CREATE TABLE with EMBEDDING MODEL annotations.
/// Custom parser that handles: CREATE TABLE name (col TYPE EMBEDDING MODEL 'name' DIM n, ...)
fn parse_create_table_with_embedding(sql: &str) -> GalaxResult<AuroraStatement> {
    let upper = sql.to_uppercase();
    let if_not_exists = upper.contains("IF NOT EXISTS");

    // Extract table name
    let after_table = if if_not_exists {
        let pos = upper.find("IF NOT EXISTS").unwrap() + "IF NOT EXISTS".len();
        sql[pos..].trim()
    } else {
        let pos = upper.find("TABLE").unwrap() + "TABLE".len();
        sql[pos..].trim()
    };

    let paren_pos = after_table.find('(')
        .ok_or_else(|| GalaxError::SqlParse { position: 0, message: "expected '(' in CREATE TABLE".into() })?;
    let table_name = after_table[..paren_pos].trim().to_string();

    // Extract column definitions between ( and )
    let cols_str = &after_table[paren_pos + 1..];
    let close_paren = cols_str.rfind(')')
        .ok_or_else(|| GalaxError::SqlParse { position: 0, message: "expected ')' in CREATE TABLE".into() })?;
    let cols_str = &cols_str[..close_paren];

    // Split by comma (respecting parentheses for types like DECIMAL(10,2))
    let col_defs = split_column_defs(cols_str);

    let mut columns = Vec::new();
    for col_def in &col_defs {
        let trimmed = col_def.trim();
        if trimmed.is_empty() { continue; }

        let col_upper = trimmed.to_uppercase();

        // Parse: name TYPE [PRIMARY KEY] [NOT NULL] [EMBEDDING MODEL 'name' DIM n]
        let parts: Vec<&str> = trimmed.splitn(2, char::is_whitespace).collect();
        if parts.len() < 2 {
            columns.push(ColumnDef {
                name: parts[0].to_string(),
                data_type: "TEXT".to_string(),
                nullable: true,
                primary_key: false,
                embedding: None,
            });
            continue;
        }

        let col_name = parts[0].to_string();
        let rest = parts[1].trim();

        // Extract data type (first word or until EMBEDDING/PRIMARY/NOT)
        let type_end = rest.to_uppercase()
            .find("EMBEDDING")
            .or_else(|| rest.to_uppercase().find("PRIMARY"))
            .or_else(|| rest.to_uppercase().find("NOT NULL"))
            .unwrap_or(rest.len());
        let data_type = rest[..type_end].trim().to_string();

        let primary_key = col_upper.contains("PRIMARY KEY");
        let nullable = !col_upper.contains("NOT NULL");

        // Parse EMBEDDING MODEL 'name' DIM n
        let embedding = if col_upper.contains("EMBEDDING") {
            let emb_pos = col_upper.find("EMBEDDING MODEL").unwrap_or(col_upper.find("EMBEDDING").unwrap());
            let emb_str = &trimmed[emb_pos..];

            // Extract model name (between quotes)
            let model_name = if let Some(q1) = emb_str.find('\'') {
                let after_q1 = &emb_str[q1 + 1..];
                if let Some(q2) = after_q1.find('\'') {
                    after_q1[..q2].to_string()
                } else {
                    "default".to_string()
                }
            } else {
                "default".to_string()
            };

            // Extract DIM n
            let dimensions = if let Some(dim_pos) = emb_str.to_uppercase().find("DIM") {
                let after_dim = emb_str[dim_pos + 3..].trim();
                after_dim.split_whitespace().next()
                    .and_then(|s| s.parse::<u32>().ok())
            } else {
                None
            };

            Some(EmbeddingDef { model_name, dimensions })
        } else {
            None
        };

        columns.push(ColumnDef {
            name: col_name,
            data_type,
            nullable,
            primary_key,
            embedding,
        });
    }

    Ok(AuroraStatement::CreateTable(CreateTableStmt {
        table_name,
        columns,
        if_not_exists,
    }))
}

/// Split column definitions by comma, respecting parentheses.
fn split_column_defs(s: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut depth = 0;

    for ch in s.chars() {
        match ch {
            '(' => { depth += 1; current.push(ch); }
            ')' => { depth -= 1; current.push(ch); }
            ',' if depth == 0 => {
                result.push(current.clone());
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    if !current.trim().is_empty() {
        result.push(current);
    }
    result
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
