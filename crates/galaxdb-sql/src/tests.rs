//! Tests for the AuroraSQL parser.

use super::ast::*;
use super::parser::*;
use galaxdb_common::GalaxError;

// ── Standard SQL parsing ───────────────────────────────────────────

#[test]
fn parse_standard_select() {
    let stmts = parse("SELECT * FROM users WHERE id = 1").unwrap();
    assert_eq!(stmts.len(), 1);
    assert!(matches!(stmts[0], AuroraStatement::Standard(_)));
}

#[test]
fn parse_standard_insert() {
    let stmts = parse("INSERT INTO users (id, name) VALUES (1, 'alice')").unwrap();
    assert_eq!(stmts.len(), 1);
    assert!(matches!(stmts[0], AuroraStatement::Standard(_)));
}

#[test]
fn parse_standard_update() {
    let stmts = parse("UPDATE users SET name = 'bob' WHERE id = 1").unwrap();
    assert_eq!(stmts.len(), 1);
    assert!(matches!(stmts[0], AuroraStatement::Standard(_)));
}

#[test]
fn parse_standard_delete() {
    let stmts = parse("DELETE FROM users WHERE id = 1").unwrap();
    assert_eq!(stmts.len(), 1);
    assert!(matches!(stmts[0], AuroraStatement::Standard(_)));
}

#[test]
fn parse_standard_create_table() {
    let stmts = parse("CREATE TABLE users (id INT PRIMARY KEY, name TEXT NOT NULL)").unwrap();
    assert_eq!(stmts.len(), 1);
    // Standard CREATE TABLE without EMBEDDING → Standard variant
    assert!(matches!(stmts[0], AuroraStatement::Standard(_)));
}

#[test]
fn parse_standard_drop_table() {
    let stmts = parse("DROP TABLE users").unwrap();
    assert_eq!(stmts.len(), 1);
    assert!(matches!(stmts[0], AuroraStatement::Standard(_)));
}

// ── SEMANTIC_MATCH ─────────────────────────────────────────────────

#[test]
fn parse_semantic_match_expr() {
    let sm = parse_semantic_match("SEMANTIC_MATCH(content, 'machine learning', 0.85)").unwrap();
    assert_eq!(sm.column, "content");
    assert_eq!(sm.query, "machine learning");
    assert!((sm.threshold - 0.85).abs() < f64::EPSILON);
}

#[test]
fn parse_semantic_match_with_spaces() {
    let sm = parse_semantic_match("SEMANTIC_MATCH( title , 'rust database' , 0.7 )").unwrap();
    assert_eq!(sm.column, "title");
    assert_eq!(sm.query, "rust database");
    assert!((sm.threshold - 0.7).abs() < f64::EPSILON);
}

#[test]
fn parse_semantic_match_invalid_args() {
    let result = parse_semantic_match("SEMANTIC_MATCH(col, 'query')");
    assert!(result.is_err());
}

#[test]
fn parse_semantic_match_not_semantic_match() {
    let result = parse_semantic_match("NOT_SEMANTIC(col, 'q', 0.5)");
    assert!(result.is_err());
}

// ── AT VERSION ─────────────────────────────────────────────────────

#[test]
fn parse_at_version_timestamp() {
    let av = parse_at_version("AT VERSION 1234567890").unwrap();
    assert_eq!(av.version, VersionRef::Timestamp(1234567890));
    assert!(av.consistency.is_none());
}

#[test]
fn parse_at_version_tag() {
    let av = parse_at_version("AT VERSION 'v1.0-release'").unwrap();
    assert_eq!(av.version, VersionRef::Tag("v1.0-release".to_string()));
    assert!(av.consistency.is_none());
}

#[test]
fn parse_at_version_with_row_snapshot() {
    let av =
        parse_at_version("AT VERSION 'my_tag' CONSISTENCY 'ROW_SNAPSHOT'").unwrap();
    assert_eq!(av.version, VersionRef::Tag("my_tag".to_string()));
    assert_eq!(av.consistency, Some(ConsistencyMode::RowSnapshot));
}

#[test]
fn parse_at_version_with_semantic_fresh() {
    let av =
        parse_at_version("AT VERSION 100 CONSISTENCY 'SEMANTIC_FRESH'").unwrap();
    assert_eq!(av.version, VersionRef::Timestamp(100));
    assert_eq!(av.consistency, Some(ConsistencyMode::SemanticFresh));
}

// ── CREATE VERSION TAG ─────────────────────────────────────────────

#[test]
fn parse_create_version_tag_simple() {
    let stmts = parse("CREATE VERSION TAG 'v1.0'").unwrap();
    assert_eq!(stmts.len(), 1);
    match &stmts[0] {
        AuroraStatement::CreateVersionTag(tag) => {
            assert_eq!(tag.name, "v1.0");
            assert!(!tag.for_training);
            assert!(tag.training_opts.is_none());
        }
        other => panic!("expected CreateVersionTag, got {:?}", other),
    }
}

#[test]
fn parse_create_version_tag_for_training() {
    let stmts = parse("CREATE VERSION TAG 'train-v2' FOR TRAINING").unwrap();
    match &stmts[0] {
        AuroraStatement::CreateVersionTag(tag) => {
            assert_eq!(tag.name, "train-v2");
            assert!(tag.for_training);
        }
        other => panic!("expected CreateVersionTag, got {:?}", other),
    }
}

#[test]
fn parse_create_version_tag_with_precision_and_seed() {
    let stmts = parse(
        "CREATE VERSION TAG 'train-v3' FOR TRAINING WITH TRAINING PRECISION 'sq8' TRAINING SEED 42",
    )
    .unwrap();
    match &stmts[0] {
        AuroraStatement::CreateVersionTag(tag) => {
            assert_eq!(tag.name, "train-v3");
            assert!(tag.for_training);
            let opts = tag.training_opts.as_ref().unwrap();
            assert_eq!(opts.precision, Some(TrainingPrecision::Sq8));
            assert_eq!(opts.seed, Some(42));
        }
        other => panic!("expected CreateVersionTag, got {:?}", other),
    }
}

#[test]
fn parse_create_version_tag_rabitq() {
    let stmts = parse(
        "CREATE VERSION TAG 'rq' FOR TRAINING WITH TRAINING PRECISION 'rabitq'",
    )
    .unwrap();
    match &stmts[0] {
        AuroraStatement::CreateVersionTag(tag) => {
            let opts = tag.training_opts.as_ref().unwrap();
            assert_eq!(opts.precision, Some(TrainingPrecision::Rabitq));
        }
        other => panic!("expected CreateVersionTag, got {:?}", other),
    }
}

#[test]
fn parse_create_version_tag_float32() {
    let stmts = parse(
        "CREATE VERSION TAG 'f32' FOR TRAINING WITH TRAINING PRECISION 'float32'",
    )
    .unwrap();
    match &stmts[0] {
        AuroraStatement::CreateVersionTag(tag) => {
            let opts = tag.training_opts.as_ref().unwrap();
            assert_eq!(opts.precision, Some(TrainingPrecision::Float32));
        }
        other => panic!("expected CreateVersionTag, got {:?}", other),
    }
}

// ── SHOW EMBEDDING HEALTH ──────────────────────────────────────────

#[test]
fn parse_show_embedding_health_no_table() {
    let stmts = parse("SHOW EMBEDDING HEALTH").unwrap();
    match &stmts[0] {
        AuroraStatement::ShowEmbeddingHealth { table } => {
            assert!(table.is_none());
        }
        other => panic!("expected ShowEmbeddingHealth, got {:?}", other),
    }
}

#[test]
fn parse_show_embedding_health_with_table() {
    let stmts = parse("SHOW EMBEDDING HEALTH FOR documents").unwrap();
    match &stmts[0] {
        AuroraStatement::ShowEmbeddingHealth { table } => {
            assert_eq!(table.as_deref(), Some("documents"));
        }
        other => panic!("expected ShowEmbeddingHealth, got {:?}", other),
    }
}

// ── BACKUP TO / RESTORE FROM ───────────────────────────────────────

#[test]
fn parse_backup_to() {
    let stmts = parse("BACKUP TO '/tmp/backup'").unwrap();
    match &stmts[0] {
        AuroraStatement::BackupTo { path } => {
            assert_eq!(path, "/tmp/backup");
        }
        other => panic!("expected BackupTo, got {:?}", other),
    }
}

#[test]
fn parse_restore_from() {
    let stmts = parse("RESTORE FROM '/tmp/backup'").unwrap();
    match &stmts[0] {
        AuroraStatement::RestoreFrom { path } => {
            assert_eq!(path, "/tmp/backup");
        }
        other => panic!("expected RestoreFrom, got {:?}", other),
    }
}

// ── ANALYZE ────────────────────────────────────────────────────────

#[test]
fn parse_analyze_table() {
    let stmts = parse("ANALYZE users").unwrap();
    match &stmts[0] {
        AuroraStatement::Analyze { table } => {
            assert_eq!(table, "users");
        }
        other => panic!("expected Analyze, got {:?}", other),
    }
}

#[test]
fn parse_analyze_with_semicolon() {
    let stmts = parse("ANALYZE my_table;").unwrap();
    match &stmts[0] {
        AuroraStatement::Analyze { table } => {
            assert_eq!(table, "my_table");
        }
        other => panic!("expected Analyze, got {:?}", other),
    }
}

// ── BULK INSERT ────────────────────────────────────────────────────

#[test]
fn parse_bulk_insert_basic() {
    let stmts = parse("BULK INSERT INTO documents (id, content) VALUES (1, 'hello')").unwrap();
    match &stmts[0] {
        AuroraStatement::BulkInsert(bi) => {
            assert_eq!(bi.table, "documents");
            assert_eq!(bi.columns, vec!["id", "content"]);
            assert_eq!(bi.values.len(), 1);
            assert_eq!(bi.values[0], vec!["1", "'hello'"]);
        }
        other => panic!("expected BulkInsert, got {:?}", other),
    }
}

#[test]
fn parse_bulk_insert_multirow() {
    let stmts = parse(
        "BULK INSERT INTO t (id, name) VALUES (1, 'a'), (2, 'b'), (3, 'c')",
    )
    .unwrap();
    match &stmts[0] {
        AuroraStatement::BulkInsert(bi) => {
            assert_eq!(bi.table, "t");
            assert_eq!(bi.columns, vec!["id", "name"]);
            assert_eq!(bi.values.len(), 3);
            assert_eq!(bi.values[0], vec!["1", "'a'"]);
            assert_eq!(bi.values[2], vec!["3", "'c'"]);
        }
        other => panic!("expected BulkInsert, got {:?}", other),
    }
}

#[test]
fn parse_bulk_insert_mismatched_cols_errors() {
    let err = parse("BULK INSERT INTO t (id, name) VALUES (1, 'a', 'extra')").unwrap_err();
    match err {
        GalaxError::SqlParse { message, .. } => {
            assert!(
                message.contains("values")
                    && (message.contains("column list") || message.contains("3")),
                "expected row/column count mismatch, got: {message}",
            );
        }
        other => panic!("expected SqlParse, got {:?}", other),
    }
}

// ── Error handling ─────────────────────────────────────────────────

#[test]
fn parse_empty_sql_returns_error() {
    let result = parse("");
    assert!(result.is_err());
    match result.unwrap_err() {
        GalaxError::SqlParse { position, message } => {
            assert_eq!(position, 0);
            assert!(message.contains("empty"));
        }
        other => panic!("expected SqlParse error, got {:?}", other),
    }
}

#[test]
fn parse_invalid_sql_returns_error_with_position() {
    let result = parse("SELECTT * FROM users");
    assert!(result.is_err());
    match result.unwrap_err() {
        GalaxError::SqlParse { message, .. } => {
            assert!(!message.is_empty());
        }
        other => panic!("expected SqlParse error, got {:?}", other),
    }
}

#[test]
fn parse_backup_without_path_returns_error() {
    let result = parse("BACKUP TO");
    assert!(result.is_err());
}

#[test]
fn parse_analyze_without_table_returns_error() {
    let result = parse("ANALYZE");
    assert!(result.is_err());
}

// ── Round-trip: standard SQL survives parsing ──────────────────────

#[test]
fn standard_sql_roundtrip_select() {
    let stmts = parse("SELECT id, name FROM users WHERE age > 21 ORDER BY name").unwrap();
    assert_eq!(stmts.len(), 1);
    assert!(matches!(stmts[0], AuroraStatement::Standard(_)));
}

#[test]
fn standard_sql_roundtrip_join() {
    let stmts = parse(
        "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id",
    )
    .unwrap();
    assert_eq!(stmts.len(), 1);
    assert!(matches!(stmts[0], AuroraStatement::Standard(_)));
}

#[test]
fn standard_sql_roundtrip_subquery() {
    let stmts = parse("SELECT * FROM users WHERE id IN (SELECT user_id FROM orders)").unwrap();
    assert_eq!(stmts.len(), 1);
    assert!(matches!(stmts[0], AuroraStatement::Standard(_)));
}

#[test]
fn multiple_statements() {
    let stmts = parse("SELECT 1; SELECT 2").unwrap();
    assert_eq!(stmts.len(), 2);
}

// ── WHERE NOT DUPLICATE (AuroraSQL extension — task 35.5) ──────────

/// `WHERE NOT DUPLICATE` parses to a `UnaryOp { Not, Identifier("DUPLICATE") }`
/// with sqlparser — that's the AST shape the executor's filter
/// conversion anchors on to produce `FilterExpr::NotDuplicate`. This
/// test pins the shape so a sqlparser upgrade that changes it fails
/// loudly instead of silently breaking the group-level dedup pass.
#[test]
fn parse_where_not_duplicate_bare() {
    use sqlparser::ast::{Expr, SetExpr, Statement, UnaryOperator};

    let stmts = parse("SELECT * FROM docs WHERE NOT DUPLICATE").unwrap();
    assert_eq!(stmts.len(), 1);
    let AuroraStatement::Standard(boxed) = &stmts[0] else {
        panic!("expected Standard(Query)");
    };
    let Statement::Query(q) = boxed.as_ref() else {
        panic!("expected Query");
    };
    let SetExpr::Select(s) = q.body.as_ref() else {
        panic!("expected Select body");
    };
    let selection = s.selection.as_ref().expect("WHERE clause present");
    let Expr::UnaryOp { op, expr } = selection else {
        panic!("expected UnaryOp, got {:?}", selection);
    };
    assert!(matches!(op, UnaryOperator::Not));
    let Expr::Identifier(id) = expr.as_ref() else {
        panic!("expected Identifier under Not, got {:?}", expr);
    };
    assert!(id.value.eq_ignore_ascii_case("DUPLICATE"));
}

/// `WHERE price > 4 AND NOT DUPLICATE` — composition with an ordinary
/// comparison predicate must survive, because the executor's dedup
/// pass walks the filter tree for the NOT DUPLICATE marker.
#[test]
fn parse_where_not_duplicate_composed_with_and() {
    use sqlparser::ast::{BinaryOperator, Expr, SetExpr, Statement, UnaryOperator};

    let stmts = parse("SELECT * FROM docs WHERE price > 4 AND NOT DUPLICATE").unwrap();
    let AuroraStatement::Standard(boxed) = &stmts[0] else {
        panic!("expected Standard(Query)");
    };
    let Statement::Query(q) = boxed.as_ref() else {
        panic!("expected Query");
    };
    let SetExpr::Select(s) = q.body.as_ref() else {
        panic!("expected Select body");
    };
    let selection = s.selection.as_ref().expect("WHERE clause present");
    let Expr::BinaryOp { op, right, .. } = selection else {
        panic!("expected BinaryOp at top, got {:?}", selection);
    };
    assert!(matches!(op, BinaryOperator::And));
    let Expr::UnaryOp {
        op: uop,
        expr: inner,
    } = right.as_ref()
    else {
        panic!("expected UnaryOp on RHS, got {:?}", right);
    };
    assert!(matches!(uop, UnaryOperator::Not));
    let Expr::Identifier(id) = inner.as_ref() else {
        panic!("expected Identifier under Not, got {:?}", inner);
    };
    assert!(id.value.eq_ignore_ascii_case("DUPLICATE"));
}
