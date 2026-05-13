//! Tests for the query executor.
//!
//! The executor has two entry points:
//!
//! * [`execute_legacy`] — catalog-only plan validation. Used by tests that
//!   check planner-to-executor contract (DDL, error messages for missing
//!   tables, embedding-source UPDATE rejection) without spinning up a
//!   real storage engine.
//! * [`execute_with_context`] — canonical execution against a real
//!   `Engine + Catalog + (optional subsystems)`. Used by tests that
//!   check actual storage round-trips (INSERT → SELECT, UPDATE, DELETE,
//!   MinHash policy integration, SEMANTIC_MATCH).
//!
//! Both are tested here.

use std::sync::Arc;

use galaxdb_common::{GalaxError, GalaxResult};
use galaxdb_storage::engine::{Engine, EngineConfig};

use crate::ast::*;
use crate::executor::*;
use crate::planner::*;

// ---------------------------------------------------------------------------
// Catalog fixtures (used by both legacy and context tests)
// ---------------------------------------------------------------------------

fn users_entry() -> TableEntry {
    TableEntry {
        name: "users".to_string(),
        columns: vec![
            CatalogColumn {
                name: "id".to_string(),
                data_type: "INT".to_string(),
                nullable: false,
                primary_key: true,
                is_embedding_source: false,
            },
            CatalogColumn {
                name: "name".to_string(),
                data_type: "TEXT".to_string(),
                nullable: true,
                primary_key: false,
                is_embedding_source: false,
            },
        ],
        has_embedding: false,
            append_only: false,
    }
}

fn make_catalog_with_users() -> Catalog {
    let mut catalog = Catalog::new();
    catalog
        .create_table("users".to_string(), users_entry())
        .unwrap();
    catalog
}

fn embedding_entry() -> TableEntry {
    TableEntry {
        name: "docs".to_string(),
        columns: vec![
            CatalogColumn {
                name: "id".to_string(),
                data_type: "INT".to_string(),
                nullable: false,
                primary_key: true,
                is_embedding_source: false,
            },
            CatalogColumn {
                name: "content".to_string(),
                data_type: "TEXT".to_string(),
                nullable: false,
                primary_key: false,
                is_embedding_source: true,
            },
        ],
        has_embedding: true,
            append_only: false,
    }
}

fn docs_body_entry() -> TableEntry {
    TableEntry {
        name: "docs".to_string(),
        columns: vec![
            CatalogColumn {
                name: "id".to_string(),
                data_type: "INT".to_string(),
                nullable: false,
                primary_key: true,
                is_embedding_source: false,
            },
            CatalogColumn {
                name: "body".to_string(),
                data_type: "TEXT".to_string(),
                nullable: true,
                primary_key: false,
                is_embedding_source: false,
            },
        ],
        has_embedding: false,
            append_only: false,
    }
}

// ---------------------------------------------------------------------------
// Real ExecutorContext (backed by a temp-dir Engine)
// ---------------------------------------------------------------------------

/// Build a fresh [`ExecutorContext`] against a real `Engine` rooted in a
/// throw-away temp directory. The tempdir is leaked so it outlives the
/// test — acceptable for unit tests that clean up via cargo's target/.
fn test_ctx() -> ExecutorContext {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().to_path_buf();
    std::mem::forget(dir);
    let engine = Engine::new(EngineConfig {
        data_dir: path,
        wal_group_commit_ms: 1,
        ..Default::default()
    })
    .expect("engine boot");
    ExecutorContext::new(Arc::new(engine))
}

/// Build a context with the `users` table already registered.
fn ctx_with_users() -> ExecutorContext {
    let mut ctx = test_ctx();
    ctx.catalog
        .create_table("users".to_string(), users_entry())
        .unwrap();
    ctx
}

/// Build a context with the `docs(body)` text table.
fn ctx_with_docs_body() -> ExecutorContext {
    let mut ctx = test_ctx();
    ctx.catalog
        .create_table("docs".to_string(), docs_body_entry())
        .unwrap();
    ctx
}

// ---------------------------------------------------------------------------
// DDL (legacy path — catalog only)
// ---------------------------------------------------------------------------

#[test]
fn legacy_create_table_succeeds() {
    let mut catalog = Catalog::new();
    let stmt = CreateTableStmt {
        table_name: "t".to_string(),
        columns: vec![ColumnDef {
            name: "id".to_string(),
            data_type: "INT".to_string(),
            nullable: false,
            primary_key: true,
            embedding: None,
        }],
        if_not_exists: false,
    };
    let plan = plan_create_table(stmt);
    let result = execute_legacy(&plan, &mut catalog);
    assert!(matches!(result, ExecuteResult::Ok(_)));
    assert!(catalog.table_exists("t"));
}

#[test]
fn legacy_create_duplicate_table_fails() {
    let mut catalog = make_catalog_with_users();
    let stmt = CreateTableStmt {
        table_name: "users".to_string(),
        columns: vec![ColumnDef {
            name: "id".to_string(),
            data_type: "INT".to_string(),
            nullable: false,
            primary_key: true,
            embedding: None,
        }],
        if_not_exists: false,
    };
    let plan = plan_create_table(stmt);
    let result = execute_legacy(&plan, &mut catalog);
    assert!(matches!(result, ExecuteResult::Error(_)));
}

#[test]
fn legacy_drop_table_succeeds() {
    let mut catalog = make_catalog_with_users();
    let plan = plan_drop_table("users".to_string(), false);
    let result = execute_legacy(&plan, &mut catalog);
    assert!(matches!(result, ExecuteResult::Ok(_)));
    assert!(!catalog.table_exists("users"));
}

#[test]
fn legacy_drop_missing_table_fails() {
    let mut catalog = Catalog::new();
    let plan = plan_drop_table("nope".to_string(), false);
    let result = execute_legacy(&plan, &mut catalog);
    assert!(matches!(result, ExecuteResult::Error(_)));
}

#[test]
fn legacy_drop_if_exists_is_idempotent() {
    let mut catalog = Catalog::new();
    let plan = plan_drop_table("nope".to_string(), true);
    let result = execute_legacy(&plan, &mut catalog);
    assert!(matches!(result, ExecuteResult::Ok(_)));
}

// ---------------------------------------------------------------------------
// DML — legacy path rejects; context path persists real data
// ---------------------------------------------------------------------------

#[test]
fn legacy_insert_into_missing_table_errors() {
    let mut catalog = Catalog::new();
    let plan = plan_insert(
        "ghost".to_string(),
        vec!["id".to_string()],
        vec![Value::Integer(1)],
    );
    let result = execute_legacy(&plan, &mut catalog);
    assert!(matches!(result, ExecuteResult::Error(_)));
}

#[test]
fn legacy_insert_column_count_mismatch_errors() {
    let mut catalog = make_catalog_with_users();
    let plan = plan_insert(
        "users".to_string(),
        vec!["id".to_string(), "name".to_string()],
        vec![Value::Integer(1)],
    );
    let result = execute_legacy(&plan, &mut catalog);
    assert!(matches!(result, ExecuteResult::Error(_)));
}

#[test]
fn legacy_insert_succeeds_but_defers_storage() {
    // Legacy path validates but does NOT persist. Callers must use
    // execute_with_context to actually write data.
    let mut catalog = make_catalog_with_users();
    let plan = plan_insert(
        "users".to_string(),
        vec!["id".to_string(), "name".to_string()],
        vec![Value::Integer(1), Value::Text("alice".to_string())],
    );
    let result = execute_legacy(&plan, &mut catalog);
    match result {
        ExecuteResult::Error(msg) => {
            assert!(msg.contains("storage engine"));
            assert!(msg.contains("execute_with_context"));
        }
        other => panic!("expected storage-required error, got {:?}", other),
    }
}

#[test]
fn context_insert_and_select_round_trip() {
    let mut ctx = ctx_with_users();

    let insert = plan_insert(
        "users".to_string(),
        vec!["id".to_string(), "name".to_string()],
        vec![Value::Integer(1), Value::Text("alice".to_string())],
    );
    let result = execute_with_context(&insert, &mut ctx).expect("insert ok");
    assert_eq!(result, ExecuteResult::RowCount(1));

    let select = plan_select("users".to_string(), vec![], None);
    let result = execute_with_context(&select, &mut ctx).expect("select ok");
    match result {
        ExecuteResult::Rows { columns, rows } => {
            assert_eq!(columns, vec!["id", "name"]);
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].columns[0].1, Value::Integer(1));
            assert_eq!(rows[0].columns[1].1, Value::Text("alice".to_string()));
        }
        other => panic!("expected Rows, got {:?}", other),
    }
}

#[test]
fn context_insert_multiple_rows_and_project_columns() {
    let mut ctx = ctx_with_users();
    for (id, name) in &[(1, "alice"), (2, "bob"), (3, "carol")] {
        let plan = plan_insert(
            "users".to_string(),
            vec!["id".to_string(), "name".to_string()],
            vec![Value::Integer(*id), Value::Text(name.to_string())],
        );
        execute_with_context(&plan, &mut ctx).expect("insert ok");
    }

    let select = plan_select("users".to_string(), vec!["name".to_string()], None);
    let result = execute_with_context(&select, &mut ctx).expect("select ok");
    match result {
        ExecuteResult::Rows { columns, rows } => {
            assert_eq!(columns, vec!["name"]);
            assert_eq!(rows.len(), 3);
            let mut names: Vec<String> = rows
                .iter()
                .map(|r| match &r.columns[0].1 {
                    Value::Text(s) => s.clone(),
                    other => panic!("expected Text, got {:?}", other),
                })
                .collect();
            names.sort();
            assert_eq!(names, vec!["alice", "bob", "carol"]);
        }
        other => panic!("expected Rows, got {:?}", other),
    }
}

#[test]
fn context_select_with_filter() {
    let mut ctx = ctx_with_users();
    for (id, name) in &[(1, "alice"), (2, "bob"), (3, "carol")] {
        let plan = plan_insert(
            "users".to_string(),
            vec!["id".to_string(), "name".to_string()],
            vec![Value::Integer(*id), Value::Text(name.to_string())],
        );
        execute_with_context(&plan, &mut ctx).unwrap();
    }

    let filter = FilterExpr::Eq {
        column: "id".to_string(),
        value: Value::Integer(2),
    };
    let select = plan_select("users".to_string(), vec![], Some(filter));
    let result = execute_with_context(&select, &mut ctx).unwrap();
    match result {
        ExecuteResult::Rows { rows, .. } => {
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].columns[1].1, Value::Text("bob".to_string()));
        }
        other => panic!("{:?}", other),
    }
}

#[test]
fn context_update_changes_value() {
    let mut ctx = ctx_with_users();
    let insert = plan_insert(
        "users".to_string(),
        vec!["id".to_string(), "name".to_string()],
        vec![Value::Integer(1), Value::Text("alice".to_string())],
    );
    execute_with_context(&insert, &mut ctx).unwrap();

    let update = plan_update(
        "users".to_string(),
        vec![("name".to_string(), Value::Text("alice2".to_string()))],
        Some(FilterExpr::Eq {
            column: "id".to_string(),
            value: Value::Integer(1),
        }),
    );
    let result = execute_with_context(&update, &mut ctx).unwrap();
    assert_eq!(result, ExecuteResult::RowCount(1));

    let select = plan_select("users".to_string(), vec![], None);
    let result = execute_with_context(&select, &mut ctx).unwrap();
    match result {
        ExecuteResult::Rows { rows, .. } => {
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].columns[1].1, Value::Text("alice2".to_string()));
        }
        other => panic!("{:?}", other),
    }
}

#[test]
fn context_update_rejects_embedding_source_column() {
    let mut ctx = test_ctx();
    ctx.catalog
        .create_table("docs".to_string(), embedding_entry())
        .unwrap();

    let update = plan_update(
        "docs".to_string(),
        vec![(
            "content".to_string(),
            Value::Text("new text".to_string()),
        )],
        None,
    );
    let err = execute_with_context(&update, &mut ctx).unwrap_err();
    match err {
        GalaxError::EmbeddingSourceUpdate { column } => assert_eq!(column, "content"),
        other => panic!("expected EmbeddingSourceUpdate, got {:?}", other),
    }
}

#[test]
fn context_update_non_embedding_column_on_embedding_table_succeeds() {
    let mut ctx = test_ctx();
    ctx.catalog
        .create_table("docs".to_string(), embedding_entry())
        .unwrap();

    // Insert a row first so there is something to update.
    let insert = plan_insert(
        "docs".to_string(),
        vec!["id".to_string(), "content".to_string()],
        vec![
            Value::Integer(1),
            Value::Text("hello world".to_string()),
        ],
    );
    execute_with_context(&insert, &mut ctx).unwrap();

    let update = plan_update(
        "docs".to_string(),
        vec![("id".to_string(), Value::Integer(2))],
        None,
    );
    let result = execute_with_context(&update, &mut ctx).unwrap();
    assert_eq!(result, ExecuteResult::RowCount(1));
}

#[test]
fn context_delete_removes_row() {
    let mut ctx = ctx_with_users();
    for id in 1..=3 {
        let plan = plan_insert(
            "users".to_string(),
            vec!["id".to_string(), "name".to_string()],
            vec![
                Value::Integer(id),
                Value::Text(format!("user{}", id)),
            ],
        );
        execute_with_context(&plan, &mut ctx).unwrap();
    }

    let delete = plan_delete(
        "users".to_string(),
        Some(FilterExpr::Eq {
            column: "id".to_string(),
            value: Value::Integer(2),
        }),
    );
    let result = execute_with_context(&delete, &mut ctx).unwrap();
    assert_eq!(result, ExecuteResult::RowCount(1));

    let select = plan_select("users".to_string(), vec![], None);
    let result = execute_with_context(&select, &mut ctx).unwrap();
    match result {
        ExecuteResult::Rows { rows, .. } => {
            assert_eq!(rows.len(), 2);
            for r in &rows {
                assert_ne!(r.columns[0].1, Value::Integer(2));
            }
        }
        other => panic!("{:?}", other),
    }
}

#[test]
fn context_delete_missing_table_fails() {
    let mut ctx = test_ctx();
    let delete = plan_delete("ghost".to_string(), None);
    let err = execute_with_context(&delete, &mut ctx).unwrap_err();
    assert!(matches!(err, GalaxError::TableNotFound(_)));
}

#[test]
fn context_point_lookup_returns_single_row() {
    let mut ctx = ctx_with_users();
    let insert = plan_insert(
        "users".to_string(),
        vec!["id".to_string(), "name".to_string()],
        vec![Value::Integer(42), Value::Text("dawn".to_string())],
    );
    execute_with_context(&insert, &mut ctx).unwrap();

    let plan = QueryPlan::PointLookup {
        table: "users".to_string(),
        key: b"users:42".to_vec(),
    };
    let result = execute_with_context(&plan, &mut ctx).unwrap();
    match result {
        ExecuteResult::Rows { rows, .. } => {
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].columns[1].1, Value::Text("dawn".to_string()));
        }
        other => panic!("{:?}", other),
    }
}

#[test]
fn context_point_lookup_missing_key_returns_no_rows() {
    let mut ctx = ctx_with_users();
    let plan = QueryPlan::PointLookup {
        table: "users".to_string(),
        key: b"users:999".to_vec(),
    };
    let result = execute_with_context(&plan, &mut ctx).unwrap();
    match result {
        ExecuteResult::Rows { rows, .. } => assert!(rows.is_empty()),
        other => panic!("{:?}", other),
    }
}

// ---------------------------------------------------------------------------
// Admin: ANALYZE / BACKUP / RESTORE / BULK INSERT / CREATE VERSION TAG
// ---------------------------------------------------------------------------

#[test]
fn context_analyze_returns_row_count() {
    let mut ctx = ctx_with_users();
    for id in 1..=5 {
        let plan = plan_insert(
            "users".to_string(),
            vec!["id".to_string()],
            vec![Value::Integer(id)],
        );
        execute_with_context(&plan, &mut ctx).unwrap();
    }
    let result = execute_with_context(
        &QueryPlan::Analyze {
            table: "users".to_string(),
        },
        &mut ctx,
    )
    .unwrap();
    match result {
        ExecuteResult::Ok(msg) => {
            assert!(msg.contains("ANALYZE users"));
            assert!(msg.contains("5 rows"));
        }
        other => panic!("{:?}", other),
    }
}

#[test]
fn context_backup_copies_files_to_target() {
    // Task 37: BACKUP TO really flushes the memtable (no rows here so
    // the flush is a no-op) and copies every `sst_*.pax` + `wal.log`
    // from the engine data dir to the target dir. With an empty
    // engine there are no SSTs, but the target directory must still
    // exist and the command must succeed.
    let mut ctx = test_ctx();
    let target = tempfile::tempdir().expect("target tempdir");
    let target_path = target.path().join("backup");
    let result = execute_with_context(
        &QueryPlan::Backup {
            path: target_path.to_string_lossy().into_owned(),
        },
        &mut ctx,
    )
    .expect("BACKUP must succeed on an open engine");
    let msg = match result {
        ExecuteResult::Ok(m) => m,
        other => panic!("expected Ok, got {other:?}"),
    };
    assert!(
        msg.contains("BACKUP TO") && msg.contains("files copied"),
        "expected a real BACKUP status message, got {msg}"
    );
    assert!(
        target_path.exists() && target_path.is_dir(),
        "BACKUP must create the target directory"
    );
}

#[test]
fn context_restore_validates_and_copies() {
    // Task 37: RESTORE FROM validates every SST block checksum in
    // the source dir before touching the live engine. An empty
    // source dir passes validation trivially (zero SSTs, zero
    // blocks) and RESTORE succeeds with "0 files copied".
    let mut ctx = test_ctx();
    let source = tempfile::tempdir().expect("source tempdir");
    let result = execute_with_context(
        &QueryPlan::Restore {
            path: source.path().to_string_lossy().into_owned(),
        },
        &mut ctx,
    )
    .expect("RESTORE must succeed when the source has no corrupted files");
    let msg = match result {
        ExecuteResult::Ok(m) => m,
        other => panic!("expected Ok, got {other:?}"),
    };
    assert!(
        msg.contains("RESTORE FROM") && msg.contains("validated"),
        "expected a real RESTORE status message, got {msg}"
    );
}

#[test]
fn context_bulk_insert_writes_real_rows() {
    let mut ctx = ctx_with_users();
    let result = execute_with_context(
        &QueryPlan::BulkInsert {
            table: "users".to_string(),
            columns: vec!["id".to_string(), "name".to_string()],
            values: vec![
                vec!["1".to_string(), "'alice'".to_string()],
                vec!["2".to_string(), "'bob'".to_string()],
                vec!["3".to_string(), "'carol'".to_string()],
            ],
        },
        &mut ctx,
    )
    .unwrap();
    match result {
        ExecuteResult::RowCount(n) => assert_eq!(n, 3),
        other => panic!("expected RowCount(3), got {:?}", other),
    }

    // Read them back through the executor — no faking.
    let rows = execute_with_context(
        &QueryPlan::FullScan {
            table: "users".to_string(),
            filter: None,
            columns: vec!["id".to_string(), "name".to_string()],
        },
        &mut ctx,
    )
    .unwrap();
    match rows {
        ExecuteResult::Rows { rows, .. } => {
            assert_eq!(rows.len(), 3, "all three bulk-inserted rows must be readable");
        }
        other => panic!("expected Rows, got {:?}", other),
    }
}

#[test]
fn context_show_embedding_health_without_sidecar() {
    let mut ctx = test_ctx();
    ctx.catalog
        .create_table("docs".to_string(), embedding_entry())
        .unwrap();

    let result = execute_with_context(
        &QueryPlan::ShowEmbeddingHealth {
            table: Some("docs".to_string()),
        },
        &mut ctx,
    )
    .unwrap();
    match result {
        ExecuteResult::Rows { columns, rows } => {
            assert_eq!(columns, vec!["table", "sidecar_state", "model_version"]);
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].columns[0].1, Value::Text("docs".to_string()));
            // No sidecar attached → state is "none".
            assert_eq!(rows[0].columns[1].1, Value::Text("none".to_string()));
        }
        other => panic!("{:?}", other),
    }
}

#[test]
fn context_create_version_tag_without_catalog_fails() {
    let mut ctx = test_ctx();
    let stmt = CreateVersionTagStmt {
        name: "v1.0".to_string(),
        for_training: false,
        training_opts: None,
    };
    let err = execute_with_context(&QueryPlan::CreateVersionTag(stmt), &mut ctx).unwrap_err();
    match err {
        GalaxError::NotYetAvailable { task, .. } => assert_eq!(task, "33"),
        other => panic!("expected NotYetAvailable, got {:?}", other),
    }
}

#[test]
fn context_create_version_tag_with_catalog_succeeds() {
    use std::sync::Mutex;
    let mut ctx = test_ctx();
    ctx.tag_catalog = Some(Arc::new(Mutex::new(galaxdb_versioning::TagCatalog::new())));
    ctx.merkle_dag = Some(Arc::new(Mutex::new(galaxdb_versioning::MerkleDag::new())));

    let stmt = CreateVersionTagStmt {
        name: "v1.0".to_string(),
        for_training: false,
        training_opts: None,
    };
    let result =
        execute_with_context(&QueryPlan::CreateVersionTag(stmt), &mut ctx).expect("tag ok");
    assert!(matches!(result, ExecuteResult::Ok(_)));

    let cat = ctx.tag_catalog.as_ref().unwrap().lock().unwrap();
    assert!(cat.get_tag("v1.0").is_some());
}

// ---------------------------------------------------------------------------
// SEMANTIC_MATCH
// ---------------------------------------------------------------------------

/// Test-only vector backend — real trait impl in `#[cfg(test)]` scope.
///
/// Not a production mock (rule 1 of `.kiro/steering/engineering-principles.md`):
/// this lives in a test file under `#[cfg(test)]`. The backend returns a
/// caller-supplied result list, which lets us exercise the executor's
/// plumbing without spinning up a sidecar + HNSW index in every test.
struct StubVectorBackend {
    results: Vec<VectorSearchResult>,
}

impl StubVectorBackend {
    fn with(results: Vec<VectorSearchResult>) -> Arc<dyn VectorSearchBackend> {
        Arc::new(Self { results })
    }
}

impl VectorSearchBackend for StubVectorBackend {
    fn semantic_search(
        &self,
        _table: &str,
        _query_text: &str,
        _threshold: f64,
        _k: usize,
        _strategy: SearchStrategy,
    ) -> GalaxResult<Vec<VectorSearchResult>> {
        Ok(self.results.clone())
    }
    fn brute_force_filtered(
        &self,
        _table: &str,
        _query_text: &str,
        _threshold: f64,
        _k: usize,
        _filter: &FilterExpr,
    ) -> GalaxResult<Vec<VectorSearchResult>> {
        Ok(self.results.clone())
    }
}

/// A backend that simulates a sidecar error path.
struct FailingVectorBackend;

impl FailingVectorBackend {
    fn new() -> Arc<dyn VectorSearchBackend> {
        Arc::new(Self)
    }
}

impl VectorSearchBackend for FailingVectorBackend {
    fn semantic_search(
        &self,
        _table: &str,
        _query_text: &str,
        _threshold: f64,
        _k: usize,
        _strategy: SearchStrategy,
    ) -> GalaxResult<Vec<VectorSearchResult>> {
        Err(GalaxError::SidecarUnavailable)
    }
    fn brute_force_filtered(
        &self,
        _table: &str,
        _query_text: &str,
        _threshold: f64,
        _k: usize,
        _filter: &FilterExpr,
    ) -> GalaxResult<Vec<VectorSearchResult>> {
        Err(GalaxError::SidecarUnavailable)
    }
}

#[test]
fn context_semantic_search_returns_results() {
    let mut ctx = test_ctx();
    ctx.catalog
        .create_table("docs".to_string(), embedding_entry())
        .unwrap();
    ctx.vector_backend = Some(StubVectorBackend::with(vec![
        VectorSearchResult {
            row_id: 42,
            similarity: 0.95,
        },
        VectorSearchResult {
            row_id: 7,
            similarity: 0.88,
        },
    ]));

    let plan = plan_semantic_search(
        "docs".to_string(),
        SemanticMatchExpr {
            column: "content".to_string(),
            query: "ml".to_string(),
            threshold: 0.8,
        },
        None,
        None,
    );
    let result = execute_with_context(&plan, &mut ctx).unwrap();
    match result {
        ExecuteResult::Rows { columns, rows } => {
            assert_eq!(columns, vec!["row_id", "similarity"]);
            assert_eq!(rows.len(), 2);
            assert_eq!(rows[0].columns[0].1, Value::Integer(42));
        }
        other => panic!("{:?}", other),
    }
}

#[test]
fn context_semantic_search_without_backend_returns_sidecar_unavailable() {
    let mut ctx = test_ctx();
    ctx.catalog
        .create_table("docs".to_string(), embedding_entry())
        .unwrap();
    // No backend installed.
    let plan = plan_semantic_search(
        "docs".to_string(),
        SemanticMatchExpr {
            column: "content".to_string(),
            query: "x".to_string(),
            threshold: 0.5,
        },
        None,
        None,
    );
    let err = execute_with_context(&plan, &mut ctx).unwrap_err();
    assert!(matches!(err, GalaxError::SidecarUnavailable));
}

#[test]
fn context_semantic_search_backend_failure_propagates() {
    let mut ctx = test_ctx();
    ctx.catalog
        .create_table("docs".to_string(), embedding_entry())
        .unwrap();
    ctx.vector_backend = Some(FailingVectorBackend::new());

    let plan = plan_semantic_search(
        "docs".to_string(),
        SemanticMatchExpr {
            column: "content".to_string(),
            query: "x".to_string(),
            threshold: 0.5,
        },
        None,
        None,
    );
    let err = execute_with_context(&plan, &mut ctx).unwrap_err();
    assert!(matches!(err, GalaxError::SidecarUnavailable));
}

#[test]
fn context_hybrid_search_brute_force_strategy() {
    let mut ctx = test_ctx();
    ctx.catalog
        .create_table("docs".to_string(), embedding_entry())
        .unwrap();
    ctx.vector_backend = Some(StubVectorBackend::with(vec![VectorSearchResult {
        row_id: 100,
        similarity: 0.92,
    }]));

    let stats = PlannerStats {
        row_count: 1_000_000,
        filter_selectivity: 0.0001,
    };
    let plan = plan_semantic_search(
        "docs".to_string(),
        SemanticMatchExpr {
            column: "content".to_string(),
            query: "rust db".to_string(),
            threshold: 0.7,
        },
        Some(FilterExpr::Eq {
            column: "id".to_string(),
            value: Value::Integer(100),
        }),
        Some(&stats),
    );
    let result = execute_with_context(&plan, &mut ctx).unwrap();
    match result {
        ExecuteResult::Rows { rows, .. } => {
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].columns[0].1, Value::Integer(100));
        }
        other => panic!("{:?}", other),
    }
}

#[test]
fn context_hybrid_search_hnsw_strategy() {
    let mut ctx = test_ctx();
    ctx.catalog
        .create_table("docs".to_string(), embedding_entry())
        .unwrap();
    ctx.vector_backend = Some(StubVectorBackend::with(vec![
        VectorSearchResult {
            row_id: 1,
            similarity: 0.99,
        },
        VectorSearchResult {
            row_id: 2,
            similarity: 0.95,
        },
        VectorSearchResult {
            row_id: 3,
            similarity: 0.90,
        },
    ]));

    let stats = PlannerStats {
        row_count: 1_000_000,
        filter_selectivity: 0.5,
    };
    let plan = plan_semantic_search(
        "docs".to_string(),
        SemanticMatchExpr {
            column: "content".to_string(),
            query: "vs".to_string(),
            threshold: 0.8,
        },
        Some(FilterExpr::Gt {
            column: "id".to_string(),
            value: Value::Integer(0),
        }),
        Some(&stats),
    );
    let result = execute_with_context(&plan, &mut ctx).unwrap();
    match result {
        ExecuteResult::Rows { rows, .. } => assert_eq!(rows.len(), 3),
        other => panic!("{:?}", other),
    }
}

// ---------------------------------------------------------------------------
// Catalog helpers
// ---------------------------------------------------------------------------

#[test]
fn catalog_starts_empty() {
    let catalog = Catalog::new();
    assert_eq!(catalog.table_count(), 0);
}

#[test]
fn catalog_create_and_get() {
    let catalog = make_catalog_with_users();
    assert!(catalog.table_exists("users"));
    let entry = catalog.get_table("users").unwrap();
    assert_eq!(entry.columns.len(), 2);
}

#[test]
fn catalog_drop_removes_table() {
    let mut catalog = make_catalog_with_users();
    catalog.drop_table("users").unwrap();
    assert!(!catalog.table_exists("users"));
}

#[test]
fn catalog_table_names_iterates() {
    let mut catalog = Catalog::new();
    catalog.create_table("a".into(), users_entry()).unwrap();
    let mut names: Vec<_> = catalog.table_names().collect();
    names.sort();
    assert_eq!(names, vec!["a"]);
}

// ---------------------------------------------------------------------------
// Task 35.2 — MinHash write-path integration against real storage
// ---------------------------------------------------------------------------

#[test]
fn minhash_policy_runs_on_insert_with_real_storage() {
    let mut ctx = ctx_with_docs_body();
    let sink = Arc::new(InMemorySystemColumnSink::new());
    ctx.minhash_policy = Some(MinHashPolicy::new(42, sink.clone()));

    let plan = plan_insert(
        "docs".to_string(),
        vec!["id".to_string(), "body".to_string()],
        vec![Value::Integer(1), Value::Text("hello world".to_string())],
    );
    let result = execute_with_context(&plan, &mut ctx).unwrap();
    assert_eq!(result, ExecuteResult::RowCount(1));

    let entries = sink.entries();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].user_column, "body");
    assert_eq!(entries[0].signature_column, "_minhash_signature__body");
    let expected = galaxdb_versioning::MinHashDedup::new(42)
        .signature("hello world")
        .to_bytes();
    assert_eq!(entries[0].signature, expected);
}

#[test]
fn minhash_policy_skips_non_text_columns() {
    let mut ctx = test_ctx();
    let entry = TableEntry {
        name: "nums".to_string(),
        columns: vec![
            CatalogColumn {
                name: "id".to_string(),
                data_type: "INT".to_string(),
                nullable: false,
                primary_key: true,
                is_embedding_source: false,
            },
            CatalogColumn {
                name: "qty".to_string(),
                data_type: "INT".to_string(),
                nullable: false,
                primary_key: false,
                is_embedding_source: false,
            },
        ],
        has_embedding: false,
            append_only: false,
    };
    ctx.catalog.create_table("nums".to_string(), entry).unwrap();

    let sink = Arc::new(InMemorySystemColumnSink::new());
    ctx.minhash_policy = Some(MinHashPolicy::new(42, sink.clone()));

    let plan = plan_insert(
        "nums".to_string(),
        vec!["id".to_string(), "qty".to_string()],
        vec![Value::Integer(1), Value::Integer(99)],
    );
    let result = execute_with_context(&plan, &mut ctx).unwrap();
    assert_eq!(result, ExecuteResult::RowCount(1));
    assert_eq!(sink.len(), 0);
}

#[test]
fn minhash_policy_skips_null_text() {
    let mut ctx = ctx_with_docs_body();
    let sink = Arc::new(InMemorySystemColumnSink::new());
    ctx.minhash_policy = Some(MinHashPolicy::new(42, sink.clone()));

    let plan = plan_insert(
        "docs".to_string(),
        vec!["id".to_string(), "body".to_string()],
        vec![Value::Integer(1), Value::Null],
    );
    let result = execute_with_context(&plan, &mut ctx).unwrap();
    assert_eq!(result, ExecuteResult::RowCount(1));
    assert_eq!(sink.len(), 0);
}

#[test]
fn minhash_policy_handles_multiple_text_columns() {
    let mut ctx = test_ctx();
    let entry = TableEntry {
        name: "docs".to_string(),
        columns: vec![
            CatalogColumn {
                name: "id".to_string(),
                data_type: "INT".to_string(),
                nullable: false,
                primary_key: true,
                is_embedding_source: false,
            },
            CatalogColumn {
                name: "title".to_string(),
                data_type: "TEXT".to_string(),
                nullable: false,
                primary_key: false,
                is_embedding_source: false,
            },
            CatalogColumn {
                name: "body".to_string(),
                data_type: "TEXT".to_string(),
                nullable: false,
                primary_key: false,
                is_embedding_source: false,
            },
        ],
        has_embedding: false,
            append_only: false,
    };
    ctx.catalog.create_table("docs".to_string(), entry).unwrap();

    let sink = Arc::new(InMemorySystemColumnSink::new());
    ctx.minhash_policy = Some(MinHashPolicy::new(42, sink.clone()));

    let plan = plan_insert(
        "docs".to_string(),
        vec!["id".to_string(), "title".to_string(), "body".to_string()],
        vec![
            Value::Integer(1),
            Value::Text("Rust is great".to_string()),
            Value::Text("Completely different content about biology".to_string()),
        ],
    );
    execute_with_context(&plan, &mut ctx).unwrap();

    let entries = sink.entries();
    assert_eq!(entries.len(), 2);
    let by_column: std::collections::HashMap<_, _> =
        entries.iter().map(|e| (e.user_column.clone(), e)).collect();
    assert_ne!(by_column["title"].signature, by_column["body"].signature);
}

#[test]
fn insert_without_policy_persists_but_no_sink_writes() {
    let mut ctx = ctx_with_docs_body();
    let sink = Arc::new(InMemorySystemColumnSink::new());
    // Deliberately do NOT attach `sink` to the context.

    let plan = plan_insert(
        "docs".to_string(),
        vec!["id".to_string(), "body".to_string()],
        vec![Value::Integer(1), Value::Text("hello".to_string())],
    );
    let result = execute_with_context(&plan, &mut ctx).unwrap();
    assert_eq!(result, ExecuteResult::RowCount(1));
    assert_eq!(sink.len(), 0);
}

#[test]
fn insert_with_policy_but_missing_table_errors() {
    let mut ctx = test_ctx();
    let sink = Arc::new(InMemorySystemColumnSink::new());
    ctx.minhash_policy = Some(MinHashPolicy::new(42, sink.clone()));

    let plan = plan_insert(
        "ghost".to_string(),
        vec!["id".to_string()],
        vec![Value::Integer(1)],
    );
    let err = execute_with_context(&plan, &mut ctx).unwrap_err();
    assert!(matches!(err, GalaxError::TableNotFound(_)));
    assert_eq!(sink.len(), 0);
}

// ---------------------------------------------------------------------------
// is_text_column predicate
// ---------------------------------------------------------------------------

#[test]
fn is_text_column_handles_varchar_and_char_sizes() {
    assert!(is_text_column("TEXT"));
    assert!(is_text_column("VARCHAR"));
    assert!(is_text_column("VARCHAR(100)"));
    assert!(is_text_column("CHAR"));
    assert!(is_text_column("CHAR(10)"));
    assert!(is_text_column("STRING"));
    assert!(is_text_column("text"));
    assert!(is_text_column("varchar(255)"));

    assert!(!is_text_column("INT"));
    assert!(!is_text_column("INTEGER"));
    assert!(!is_text_column("FLOAT"));
    assert!(!is_text_column("BOOL"));
    assert!(!is_text_column("BLOB"));
}

// ---------------------------------------------------------------------------
// `WHERE NOT DUPLICATE` — task 35.5
// ---------------------------------------------------------------------------

/// Build a `docs` table whose schema carries the near-duplicate
/// grouping column populated by the Task 35.4 background job.
fn ctx_with_dedup_docs() -> ExecutorContext {
    let mut ctx = test_ctx();
    let entry = TableEntry {
        name: "docs".to_string(),
        columns: vec![
            CatalogColumn {
                name: "id".to_string(),
                data_type: "INT".to_string(),
                nullable: false,
                primary_key: true,
                is_embedding_source: false,
            },
            CatalogColumn {
                name: "body".to_string(),
                data_type: "TEXT".to_string(),
                nullable: true,
                primary_key: false,
                is_embedding_source: false,
            },
            CatalogColumn {
                name: crate::planner::NEAR_DUPLICATE_GROUP_COLUMN.to_string(),
                data_type: "BIGINT".to_string(),
                nullable: true,
                primary_key: false,
                is_embedding_source: false,
            },
        ],
        has_embedding: false,
            append_only: false,
    };
    ctx.catalog.create_table("docs".to_string(), entry).unwrap();
    ctx
}

fn insert_dedup_row(
    ctx: &mut ExecutorContext,
    id: i64,
    body: &str,
    group: Option<i64>,
) {
    let plan = plan_insert(
        "docs".to_string(),
        vec![
            "id".to_string(),
            "body".to_string(),
            crate::planner::NEAR_DUPLICATE_GROUP_COLUMN.to_string(),
        ],
        vec![
            Value::Integer(id),
            Value::Text(body.to_string()),
            match group {
                Some(g) => Value::Integer(g),
                None => Value::Null,
            },
        ],
    );
    execute_with_context(&plan, ctx).expect("insert ok");
}

/// End-to-end: seed 6 rows — 3 share group 100, 2 share group 200,
/// 1 has no group. `WHERE NOT DUPLICATE` keeps one representative per
/// group plus the ungrouped row.
#[test]
fn where_not_duplicate_keeps_one_representative_per_group() {
    let mut ctx = ctx_with_dedup_docs();
    // Group 100: ids 3, 1, 4 → representative is id=1 (lowest pk bytes
    // under the `docs:` prefix since row_codec::build_primary_key
    // serialises the integer pk as its decimal display — '1' < '3' < '4').
    insert_dedup_row(&mut ctx, 3, "hello world", Some(100));
    insert_dedup_row(&mut ctx, 1, "hello world!", Some(100));
    insert_dedup_row(&mut ctx, 4, "hello world.", Some(100));
    // Group 200: ids 5, 2 → representative is id=2.
    insert_dedup_row(&mut ctx, 5, "quick brown fox", Some(200));
    insert_dedup_row(&mut ctx, 2, "quick brown fox!", Some(200));
    // Ungrouped.
    insert_dedup_row(&mut ctx, 6, "unique text", None);

    let plan = plan_select(
        "docs".to_string(),
        vec!["id".to_string()],
        Some(FilterExpr::NotDuplicate),
    );
    let result = execute_with_context(&plan, &mut ctx).expect("select ok");
    let ExecuteResult::Rows { rows, .. } = result else {
        panic!("expected Rows");
    };

    let mut ids: Vec<i64> = rows
        .into_iter()
        .map(|r| match r.columns[0].1 {
            Value::Integer(n) => n,
            ref other => panic!("expected Integer, got {:?}", other),
        })
        .collect();
    ids.sort();
    // One representative per group plus the ungrouped row.
    assert_eq!(ids, vec![1, 2, 6]);
}

/// Composition with a conventional WHERE predicate: `price > 4 AND
/// NOT DUPLICATE` must first narrow by price, then collapse
/// duplicates. The dedup pass runs on the filtered candidate set —
/// confirming that the group-level predicate composes correctly with
/// per-row filters.
#[test]
fn where_not_duplicate_composes_with_and() {
    let mut ctx = ctx_with_dedup_docs();
    insert_dedup_row(&mut ctx, 1, "a", Some(100));
    insert_dedup_row(&mut ctx, 2, "b", Some(100));
    insert_dedup_row(&mut ctx, 3, "c", Some(200));
    insert_dedup_row(&mut ctx, 4, "d", None);

    // id > 1: drops id=1 — group 100's candidate set is now {2}; the
    // representative is therefore id=2 even though id=1 would have
    // been the representative over the full set. The dedup pass must
    // run AFTER the per-row filter for this to work.
    let filter = FilterExpr::And(
        Box::new(FilterExpr::Gt {
            column: "id".to_string(),
            value: Value::Integer(1),
        }),
        Box::new(FilterExpr::NotDuplicate),
    );
    let plan = plan_select("docs".to_string(), vec!["id".to_string()], Some(filter));
    let result = execute_with_context(&plan, &mut ctx).expect("select ok");
    let ExecuteResult::Rows { rows, .. } = result else {
        panic!("expected Rows");
    };

    let mut ids: Vec<i64> = rows
        .into_iter()
        .map(|r| match r.columns[0].1 {
            Value::Integer(n) => n,
            ref other => panic!("expected Integer, got {:?}", other),
        })
        .collect();
    ids.sort();
    assert_eq!(ids, vec![2, 3, 4]);
}

/// A table with no `_near_duplicate_group` column at all: every row
/// passes `WHERE NOT DUPLICATE`. This is the steady-state for any
/// table the Task 35.4 background job hasn't touched yet.
#[test]
fn where_not_duplicate_passes_rows_without_group_column() {
    let mut ctx = ctx_with_docs_body();
    for (id, body) in &[(1, "alpha"), (2, "beta"), (3, "gamma")] {
        let plan = plan_insert(
            "docs".to_string(),
            vec!["id".to_string(), "body".to_string()],
            vec![Value::Integer(*id), Value::Text(body.to_string())],
        );
        execute_with_context(&plan, &mut ctx).expect("insert ok");
    }
    let plan = plan_select(
        "docs".to_string(),
        vec!["id".to_string()],
        Some(FilterExpr::NotDuplicate),
    );
    let result = execute_with_context(&plan, &mut ctx).expect("select ok");
    let ExecuteResult::Rows { rows, .. } = result else {
        panic!("expected Rows");
    };
    assert_eq!(rows.len(), 3);
}

/// `filter_has_not_duplicate` walks composed trees so the executor
/// knows whether to run the group-level pass.
#[test]
fn filter_has_not_duplicate_walks_tree() {
    assert!(filter_has_not_duplicate(&FilterExpr::NotDuplicate));
    assert!(!filter_has_not_duplicate(&FilterExpr::Eq {
        column: "id".into(),
        value: Value::Integer(1)
    }));
    // AND
    assert!(filter_has_not_duplicate(&FilterExpr::And(
        Box::new(FilterExpr::Gt {
            column: "price".into(),
            value: Value::Float(4.0),
        }),
        Box::new(FilterExpr::NotDuplicate),
    )));
    // OR (right branch)
    assert!(filter_has_not_duplicate(&FilterExpr::Or(
        Box::new(FilterExpr::Eq {
            column: "id".into(),
            value: Value::Integer(1)
        }),
        Box::new(FilterExpr::NotDuplicate),
    )));
    // Nested deep
    assert!(filter_has_not_duplicate(&FilterExpr::And(
        Box::new(FilterExpr::Gt {
            column: "a".into(),
            value: Value::Integer(0),
        }),
        Box::new(FilterExpr::Or(
            Box::new(FilterExpr::Eq {
                column: "b".into(),
                value: Value::Integer(0),
            }),
            Box::new(FilterExpr::NotDuplicate),
        )),
    )));
}
