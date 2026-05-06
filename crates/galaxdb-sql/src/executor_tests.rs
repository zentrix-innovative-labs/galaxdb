//! Tests for the query executor.

use crate::ast::*;
use crate::executor::*;
use crate::planner::*;

fn make_catalog_with_users() -> Catalog {
    let mut catalog = Catalog::new();
    let entry = TableEntry {
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
    };
    catalog.create_table("users".to_string(), entry).unwrap();
    catalog
}

fn make_catalog_with_embedding_table() -> Catalog {
    let mut catalog = Catalog::new();
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
                name: "content".to_string(),
                data_type: "TEXT".to_string(),
                nullable: false,
                primary_key: false,
                is_embedding_source: true,
            },
        ],
        has_embedding: true,
    };
    catalog.create_table("docs".to_string(), entry).unwrap();
    catalog
}

// ── DDL tests ──────────────────────────────────────────────────────

#[test]
fn execute_create_table_succeeds() {
    let mut catalog = Catalog::new();
    let stmt = CreateTableStmt {
        table_name: "test".to_string(),
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
    let result = execute(&plan, &mut catalog, &NoOpVectorBackend);

    assert!(matches!(result, ExecuteResult::Ok(_)));
    assert!(catalog.table_exists("test"));
}

#[test]
fn execute_create_duplicate_table_fails() {
    let mut catalog = make_catalog_with_users();
    let stmt = CreateTableStmt {
        table_name: "users".to_string(),
        columns: vec![],
        if_not_exists: false,
    };
    let plan = plan_create_table(stmt);
    let result = execute(&plan, &mut catalog, &NoOpVectorBackend);

    assert!(matches!(result, ExecuteResult::Error(_)));
}

#[test]
fn execute_drop_table_succeeds() {
    let mut catalog = make_catalog_with_users();
    let plan = plan_drop_table("users".to_string(), false);
    let result = execute(&plan, &mut catalog, &NoOpVectorBackend);

    assert!(matches!(result, ExecuteResult::Ok(_)));
    assert!(!catalog.table_exists("users"));
}

#[test]
fn execute_drop_nonexistent_table_fails() {
    let mut catalog = Catalog::new();
    let plan = plan_drop_table("nope".to_string(), false);
    let result = execute(&plan, &mut catalog, &NoOpVectorBackend);

    assert!(matches!(result, ExecuteResult::Error(_)));
}

#[test]
fn execute_drop_if_exists_nonexistent_succeeds() {
    let mut catalog = Catalog::new();
    let plan = plan_drop_table("nope".to_string(), true);
    let result = execute(&plan, &mut catalog, &NoOpVectorBackend);

    assert!(matches!(result, ExecuteResult::Ok(_)));
}

// ── DML tests ──────────────────────────────────────────────────────

#[test]
fn execute_insert_succeeds() {
    let mut catalog = make_catalog_with_users();
    let plan = plan_insert(
        "users".to_string(),
        vec!["id".to_string(), "name".to_string()],
        vec![Value::Integer(1), Value::Text("alice".to_string())],
    );
    let result = execute(&plan, &mut catalog, &NoOpVectorBackend);

    assert_eq!(result, ExecuteResult::RowCount(1));
}

#[test]
fn execute_insert_nonexistent_table_fails() {
    let mut catalog = Catalog::new();
    let plan = plan_insert(
        "nope".to_string(),
        vec!["id".to_string()],
        vec![Value::Integer(1)],
    );
    let result = execute(&plan, &mut catalog, &NoOpVectorBackend);

    assert!(matches!(result, ExecuteResult::Error(_)));
}

#[test]
fn execute_insert_column_count_mismatch_fails() {
    let mut catalog = make_catalog_with_users();
    let plan = plan_insert(
        "users".to_string(),
        vec!["id".to_string(), "name".to_string()],
        vec![Value::Integer(1)], // only 1 value for 2 columns
    );
    let result = execute(&plan, &mut catalog, &NoOpVectorBackend);

    assert!(matches!(result, ExecuteResult::Error(_)));
}

#[test]
fn execute_select_succeeds() {
    let mut catalog = make_catalog_with_users();
    let plan = plan_select("users".to_string(), vec!["id".to_string()], None);
    let result = execute(&plan, &mut catalog, &NoOpVectorBackend);

    assert!(matches!(result, ExecuteResult::Rows { .. }));
}

#[test]
fn execute_select_nonexistent_table_fails() {
    let mut catalog = Catalog::new();
    let plan = plan_select("nope".to_string(), vec![], None);
    let result = execute(&plan, &mut catalog, &NoOpVectorBackend);

    assert!(matches!(result, ExecuteResult::Error(_)));
}

#[test]
fn execute_update_succeeds() {
    let mut catalog = make_catalog_with_users();
    let plan = plan_update(
        "users".to_string(),
        vec![("name".to_string(), Value::Text("bob".to_string()))],
        None,
    );
    let result = execute(&plan, &mut catalog, &NoOpVectorBackend);

    assert_eq!(result, ExecuteResult::RowCount(0));
}

#[test]
fn execute_delete_succeeds() {
    let mut catalog = make_catalog_with_users();
    let plan = plan_delete("users".to_string(), None);
    let result = execute(&plan, &mut catalog, &NoOpVectorBackend);

    assert_eq!(result, ExecuteResult::RowCount(0));
}

// ── Embedding source update rejection (Req 15.5) ──────────────────

#[test]
fn execute_update_embedding_source_rejected() {
    let mut catalog = make_catalog_with_embedding_table();
    let plan = plan_update(
        "docs".to_string(),
        vec![("content".to_string(), Value::Text("new text".to_string()))],
        None,
    );
    let result = execute(&plan, &mut catalog, &NoOpVectorBackend);

    match result {
        ExecuteResult::Error(msg) => {
            assert!(msg.contains("embedding source"));
            assert!(msg.contains("DELETE + INSERT"));
        }
        other => panic!("expected Error, got {:?}", other),
    }
}

#[test]
fn execute_update_non_embedding_column_succeeds() {
    let mut catalog = make_catalog_with_embedding_table();
    let plan = plan_update(
        "docs".to_string(),
        vec![("id".to_string(), Value::Integer(2))],
        None,
    );
    let result = execute(&plan, &mut catalog, &NoOpVectorBackend);

    // id is not an embedding source, so update should succeed
    assert_eq!(result, ExecuteResult::RowCount(0));
}

// ── Extension commands ─────────────────────────────────────────────

#[test]
fn execute_analyze_succeeds() {
    let mut catalog = make_catalog_with_users();
    let plan = QueryPlan::Analyze {
        table: "users".to_string(),
    };
    let result = execute(&plan, &mut catalog, &NoOpVectorBackend);

    assert!(matches!(result, ExecuteResult::Ok(_)));
}

#[test]
fn execute_analyze_nonexistent_table_fails() {
    let mut catalog = Catalog::new();
    let plan = QueryPlan::Analyze {
        table: "nope".to_string(),
    };
    let result = execute(&plan, &mut catalog, &NoOpVectorBackend);

    assert!(matches!(result, ExecuteResult::Error(_)));
}

#[test]
fn execute_backup_succeeds() {
    let mut catalog = Catalog::new();
    let plan = QueryPlan::Backup {
        path: "/tmp/backup".to_string(),
    };
    let result = execute(&plan, &mut catalog, &NoOpVectorBackend);

    assert!(matches!(result, ExecuteResult::Ok(_)));
}

#[test]
fn execute_restore_succeeds() {
    let mut catalog = Catalog::new();
    let plan = QueryPlan::Restore {
        path: "/tmp/backup".to_string(),
    };
    let result = execute(&plan, &mut catalog, &NoOpVectorBackend);

    assert!(matches!(result, ExecuteResult::Ok(_)));
}

#[test]
fn execute_show_embedding_health() {
    let mut catalog = Catalog::new();
    let plan = QueryPlan::ShowEmbeddingHealth { table: None };
    let result = execute(&plan, &mut catalog, &NoOpVectorBackend);

    assert!(matches!(result, ExecuteResult::Rows { .. }));
}

#[test]
fn execute_create_version_tag() {
    let mut catalog = Catalog::new();
    let plan = QueryPlan::CreateVersionTag(CreateVersionTagStmt {
        name: "v1.0".to_string(),
        for_training: false,
        training_opts: None,
    });
    let result = execute(&plan, &mut catalog, &NoOpVectorBackend);

    assert!(matches!(result, ExecuteResult::Ok(_)));
}

// ── Catalog tests ──────────────────────────────────────────────────

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

// ── SEMANTIC_MATCH execution tests ─────────────────────────────────

/// Mock vector backend that returns pre-configured results.
struct MockVectorBackend {
    results: Vec<VectorSearchResult>,
}

impl MockVectorBackend {
    fn with_results(results: Vec<VectorSearchResult>) -> Self {
        Self { results }
    }
}

impl VectorSearchBackend for MockVectorBackend {
    fn semantic_search(
        &self, _table: &str, _query_text: &str, _threshold: f64, _k: usize, _strategy: SearchStrategy,
    ) -> Result<Vec<VectorSearchResult>, String> {
        Ok(self.results.clone())
    }

    fn brute_force_filtered(
        &self, _table: &str, _query_text: &str, _threshold: f64, _k: usize, _filter: &FilterExpr,
    ) -> Result<Vec<VectorSearchResult>, String> {
        Ok(self.results.clone())
    }
}

/// Mock backend that simulates sidecar being down.
struct DownVectorBackend;

impl VectorSearchBackend for DownVectorBackend {
    fn semantic_search(
        &self, _table: &str, _query_text: &str, _threshold: f64, _k: usize, _strategy: SearchStrategy,
    ) -> Result<Vec<VectorSearchResult>, String> {
        Err("semantic search temporarily unavailable — embedding sidecar is down".to_string())
    }

    fn brute_force_filtered(
        &self, _table: &str, _query_text: &str, _threshold: f64, _k: usize, _filter: &FilterExpr,
    ) -> Result<Vec<VectorSearchResult>, String> {
        Err("semantic search temporarily unavailable — embedding sidecar is down".to_string())
    }
}

#[test]
fn execute_semantic_search_returns_results() {
    let mut catalog = make_catalog_with_embedding_table();
    let backend = MockVectorBackend::with_results(vec![
        VectorSearchResult { row_id: 42, similarity: 0.95 },
        VectorSearchResult { row_id: 7, similarity: 0.88 },
    ]);

    let plan = plan_semantic_search(
        "docs".to_string(),
        SemanticMatchExpr {
            column: "content".to_string(),
            query: "machine learning".to_string(),
            threshold: 0.8,
        },
        None,
        None,
    );

    let result = execute(&plan, &mut catalog, &backend);

    match result {
        ExecuteResult::Rows { columns, rows } => {
            assert_eq!(columns, vec!["id", "score"]);
            assert_eq!(rows.len(), 2);
            assert_eq!(rows[0].columns[0], ("id".to_string(), Value::Integer(42)));
            assert_eq!(rows[1].columns[0], ("id".to_string(), Value::Integer(7)));
        }
        other => panic!("expected Rows, got {:?}", other),
    }
}

#[test]
fn execute_semantic_search_sidecar_down_returns_error() {
    let mut catalog = make_catalog_with_embedding_table();
    let backend = DownVectorBackend;

    let plan = plan_semantic_search(
        "docs".to_string(),
        SemanticMatchExpr {
            column: "content".to_string(),
            query: "test query".to_string(),
            threshold: 0.5,
        },
        None,
        None,
    );

    let result = execute(&plan, &mut catalog, &backend);

    match result {
        ExecuteResult::Error(msg) => {
            assert!(msg.contains("semantic search temporarily unavailable"));
            assert!(msg.contains("sidecar is down"));
        }
        other => panic!("expected Error, got {:?}", other),
    }
}

#[test]
fn execute_semantic_search_nonexistent_table_fails() {
    let mut catalog = Catalog::new();
    let backend = MockVectorBackend::with_results(vec![]);

    let plan = plan_semantic_search(
        "nonexistent".to_string(),
        SemanticMatchExpr {
            column: "content".to_string(),
            query: "test".to_string(),
            threshold: 0.5,
        },
        None,
        None,
    );

    let result = execute(&plan, &mut catalog, &backend);
    assert!(matches!(result, ExecuteResult::Error(_)));
}

#[test]
fn execute_hybrid_search_brute_force_strategy() {
    let mut catalog = make_catalog_with_embedding_table();
    let backend = MockVectorBackend::with_results(vec![
        VectorSearchResult { row_id: 100, similarity: 0.92 },
    ]);

    let stats = PlannerStats {
        row_count: 1_000_000,
        filter_selectivity: 0.0001, // very selective → brute force
    };

    let plan = plan_semantic_search(
        "docs".to_string(),
        SemanticMatchExpr {
            column: "content".to_string(),
            query: "rust database".to_string(),
            threshold: 0.7,
        },
        Some(FilterExpr::Eq {
            column: "id".to_string(),
            value: Value::Integer(100),
        }),
        Some(&stats),
    );

    let result = execute(&plan, &mut catalog, &backend);

    match result {
        ExecuteResult::Rows { rows, .. } => {
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].columns[0], ("id".to_string(), Value::Integer(100)));
        }
        other => panic!("expected Rows, got {:?}", other),
    }
}

#[test]
fn execute_hybrid_search_hnsw_strategy() {
    let mut catalog = make_catalog_with_embedding_table();
    let backend = MockVectorBackend::with_results(vec![
        VectorSearchResult { row_id: 1, similarity: 0.99 },
        VectorSearchResult { row_id: 2, similarity: 0.95 },
        VectorSearchResult { row_id: 3, similarity: 0.90 },
    ]);

    let stats = PlannerStats {
        row_count: 1_000_000,
        filter_selectivity: 0.5, // high cardinality → HNSW
    };

    let plan = plan_semantic_search(
        "docs".to_string(),
        SemanticMatchExpr {
            column: "content".to_string(),
            query: "vector search".to_string(),
            threshold: 0.8,
        },
        Some(FilterExpr::Gt {
            column: "id".to_string(),
            value: Value::Integer(0),
        }),
        Some(&stats),
    );

    let result = execute(&plan, &mut catalog, &backend);

    match result {
        ExecuteResult::Rows { rows, .. } => {
            assert_eq!(rows.len(), 3);
        }
        other => panic!("expected Rows, got {:?}", other),
    }
}
