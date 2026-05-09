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

// ── Task 35.2: MinHash write-path integration ──────────────────────

use std::sync::Arc;

/// Build a catalog with a `docs(id INT PK, body TEXT)` table for MinHash
/// integration tests.
fn make_catalog_with_docs_body() -> Catalog {
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
                name: "body".to_string(),
                data_type: "TEXT".to_string(),
                nullable: true,
                primary_key: false,
                is_embedding_source: false,
            },
        ],
        has_embedding: false,
    };
    catalog.create_table("docs".to_string(), entry).unwrap();
    catalog
}

#[test]
fn insert_computes_minhash_for_text_column() {
    let mut catalog = make_catalog_with_docs_body();
    let sink = Arc::new(InMemorySystemColumnSink::new());
    let policy = MinHashPolicy::new(42, sink.clone());

    let plan = plan_insert(
        "docs".to_string(),
        vec!["id".to_string(), "body".to_string()],
        vec![Value::Integer(1), Value::Text("hello world".to_string())],
    );

    let result = execute_with_policies(&plan, &mut catalog, &NoOpVectorBackend, Some(&policy));
    assert_eq!(result, ExecuteResult::RowCount(1));

    let entries = sink.entries();
    assert_eq!(entries.len(), 1, "expected exactly one MinHash write");
    assert_eq!(entries[0].table, "docs");
    assert_eq!(entries[0].user_column, "body");
    assert_eq!(entries[0].signature_column, "_minhash_signature__body");

    // The signature must match an independent computation from the same seed.
    let expected = galaxdb_versioning::MinHashDedup::new(42)
        .signature("hello world")
        .to_bytes();
    assert_eq!(entries[0].signature, expected);
}

#[test]
fn insert_skips_non_text_columns() {
    let mut catalog = Catalog::new();
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
    };
    catalog.create_table("nums".to_string(), entry).unwrap();

    let sink = Arc::new(InMemorySystemColumnSink::new());
    let policy = MinHashPolicy::new(42, sink.clone());

    let plan = plan_insert(
        "nums".to_string(),
        vec!["id".to_string(), "qty".to_string()],
        vec![Value::Integer(1), Value::Integer(99)],
    );

    let result = execute_with_policies(&plan, &mut catalog, &NoOpVectorBackend, Some(&policy));
    assert_eq!(result, ExecuteResult::RowCount(1));
    assert_eq!(sink.len(), 0, "non-TEXT table should not produce MinHash writes");
}

#[test]
fn insert_skips_null_text_values() {
    let mut catalog = make_catalog_with_docs_body();
    let sink = Arc::new(InMemorySystemColumnSink::new());
    let policy = MinHashPolicy::new(42, sink.clone());

    let plan = plan_insert(
        "docs".to_string(),
        vec!["id".to_string(), "body".to_string()],
        vec![Value::Integer(1), Value::Null],
    );

    let result = execute_with_policies(&plan, &mut catalog, &NoOpVectorBackend, Some(&policy));
    assert_eq!(result, ExecuteResult::RowCount(1));
    assert_eq!(sink.len(), 0, "NULL text value should not yield a MinHash signature");
}

#[test]
fn insert_handles_multiple_text_columns() {
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
    };
    catalog.create_table("docs".to_string(), entry).unwrap();

    let sink = Arc::new(InMemorySystemColumnSink::new());
    let policy = MinHashPolicy::new(42, sink.clone());

    let plan = plan_insert(
        "docs".to_string(),
        vec!["id".to_string(), "title".to_string(), "body".to_string()],
        vec![
            Value::Integer(1),
            Value::Text("Rust is great".to_string()),
            Value::Text("Completely different content about biology".to_string()),
        ],
    );

    let result = execute_with_policies(&plan, &mut catalog, &NoOpVectorBackend, Some(&policy));
    assert_eq!(result, ExecuteResult::RowCount(1));

    let entries = sink.entries();
    assert_eq!(entries.len(), 2, "expected one MinHash write per TEXT column");

    let by_column: std::collections::HashMap<_, _> =
        entries.iter().map(|e| (e.user_column.clone(), e)).collect();
    assert!(by_column.contains_key("title"));
    assert!(by_column.contains_key("body"));
    assert_eq!(
        by_column["title"].signature_column,
        "_minhash_signature__title"
    );
    assert_eq!(
        by_column["body"].signature_column,
        "_minhash_signature__body"
    );

    // Distinct texts → distinct signatures.
    assert_ne!(
        by_column["title"].signature, by_column["body"].signature,
        "unrelated texts should produce distinct signatures"
    );
}

#[test]
fn insert_determinism() {
    let sink_a = Arc::new(InMemorySystemColumnSink::new());
    let sink_b = Arc::new(InMemorySystemColumnSink::new());
    let policy_a = MinHashPolicy::new(42, sink_a.clone());
    let policy_b = MinHashPolicy::new(42, sink_b.clone());

    let mut catalog_a = make_catalog_with_docs_body();
    let mut catalog_b = make_catalog_with_docs_body();

    let make_plan = || {
        plan_insert(
            "docs".to_string(),
            vec!["id".to_string(), "body".to_string()],
            vec![
                Value::Integer(7),
                Value::Text("deterministic minhash run".to_string()),
            ],
        )
    };

    let _ = execute_with_policies(&make_plan(), &mut catalog_a, &NoOpVectorBackend, Some(&policy_a));
    let _ = execute_with_policies(&make_plan(), &mut catalog_b, &NoOpVectorBackend, Some(&policy_b));

    let a = sink_a.entries();
    let b = sink_b.entries();
    assert_eq!(a.len(), 1);
    assert_eq!(b.len(), 1);
    assert_eq!(
        a[0].signature, b[0].signature,
        "same seed + same text must yield byte-identical signatures"
    );
}

#[test]
fn insert_uses_column_names_when_provided() {
    let mut catalog = make_catalog_with_docs_body();
    let sink = Arc::new(InMemorySystemColumnSink::new());
    let policy = MinHashPolicy::new(42, sink.clone());

    // Explicit column list in reverse order.
    let plan = plan_insert(
        "docs".to_string(),
        vec!["body".to_string(), "id".to_string()],
        vec![Value::Text("hello".to_string()), Value::Integer(1)],
    );

    let result = execute_with_policies(&plan, &mut catalog, &NoOpVectorBackend, Some(&policy));
    assert_eq!(result, ExecuteResult::RowCount(1));

    let entries = sink.entries();
    assert_eq!(entries.len(), 1, "exactly one TEXT column → one signature");
    assert_eq!(entries[0].user_column, "body");
    assert_eq!(entries[0].signature_column, "_minhash_signature__body");

    // Signature is of "hello", not Integer(1).
    let expected = galaxdb_versioning::MinHashDedup::new(42)
        .signature("hello")
        .to_bytes();
    assert_eq!(entries[0].signature, expected);
}

#[test]
fn is_text_column_handles_varchar_and_char_sizes() {
    assert!(is_text_column("TEXT"));
    assert!(is_text_column("VARCHAR"));
    assert!(is_text_column("VARCHAR(100)"));
    assert!(is_text_column("CHAR"));
    assert!(is_text_column("CHAR(10)"));
    assert!(is_text_column("STRING"));
    assert!(is_text_column("text")); // case-insensitive
    assert!(is_text_column("varchar(255)"));

    assert!(!is_text_column("INT"));
    assert!(!is_text_column("INTEGER"));
    assert!(!is_text_column("FLOAT"));
    assert!(!is_text_column("BOOL"));
    assert!(!is_text_column("BLOB"));
}

#[test]
fn execute_without_policy_does_not_call_sink() {
    let mut catalog = make_catalog_with_docs_body();
    // The sink is created but never handed to `execute`.
    let sink = Arc::new(InMemorySystemColumnSink::new());

    let plan = plan_insert(
        "docs".to_string(),
        vec!["id".to_string(), "body".to_string()],
        vec![Value::Integer(1), Value::Text("hello".to_string())],
    );

    // Legacy `execute` path.
    let result = execute(&plan, &mut catalog, &NoOpVectorBackend);
    assert_eq!(result, ExecuteResult::RowCount(1));
    assert_eq!(sink.len(), 0, "legacy execute must not invoke any sink");
}

#[test]
fn execute_with_policy_but_missing_table_does_not_panic() {
    let mut catalog = Catalog::new(); // no tables registered
    let sink = Arc::new(InMemorySystemColumnSink::new());
    let policy = MinHashPolicy::new(42, sink.clone());

    let plan = plan_insert(
        "ghost".to_string(),
        vec!["id".to_string()],
        vec![Value::Integer(1)],
    );

    let result = execute_with_policies(&plan, &mut catalog, &NoOpVectorBackend, Some(&policy));
    match result {
        ExecuteResult::Error(msg) => assert!(msg.contains("table not found")),
        other => panic!("expected Error for missing table, got {:?}", other),
    }
    assert_eq!(sink.len(), 0, "missing table must not produce sink writes");
}
