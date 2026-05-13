//! Tests for the query planner.

use crate::ast::*;
use crate::planner::*;

#[test]
fn plan_create_table_produces_correct_plan() {
    let stmt = CreateTableStmt {
        table_name: "users".to_string(),
        columns: vec![
            ColumnDef {
                name: "id".to_string(),
                data_type: "INT".to_string(),
                nullable: false,
                primary_key: true,
                embedding: None,
            },
            ColumnDef {
                name: "name".to_string(),
                data_type: "TEXT".to_string(),
                nullable: true,
                primary_key: false,
                embedding: None,
            },
        ],
        if_not_exists: false,
    };

    let plan = plan_create_table(stmt.clone());
    assert_eq!(plan, QueryPlan::CreateTable(stmt));
}

#[test]
fn plan_drop_table_with_if_exists() {
    let plan = plan_drop_table("users".to_string(), true);
    assert_eq!(
        plan,
        QueryPlan::DropTable {
            name: "users".to_string(),
            if_exists: true,
        }
    );
}

#[test]
fn plan_insert_produces_correct_plan() {
    let plan = plan_insert(
        "users".to_string(),
        vec!["id".to_string(), "name".to_string()],
        vec![Value::Integer(1), Value::Text("alice".to_string())],
    );

    match plan {
        QueryPlan::Insert {
            table,
            columns,
            values,
        } => {
            assert_eq!(table, "users");
            assert_eq!(columns.len(), 2);
            assert_eq!(values.len(), 2);
        }
        other => panic!("expected Insert, got {:?}", other),
    }
}

#[test]
fn plan_delete_with_filter() {
    let filter = FilterExpr::Eq {
        column: "id".to_string(),
        value: Value::Integer(1),
    };
    let plan = plan_delete("users".to_string(), Some(filter.clone()));

    match plan {
        QueryPlan::Delete { table, filter: f } => {
            assert_eq!(table, "users");
            assert_eq!(f, Some(filter));
        }
        other => panic!("expected Delete, got {:?}", other),
    }
}

#[test]
fn plan_update_with_assignments() {
    let plan = plan_update(
        "users".to_string(),
        vec![("name".to_string(), Value::Text("bob".to_string()))],
        None,
    );

    match plan {
        QueryPlan::Update {
            table,
            assignments,
            filter,
        } => {
            assert_eq!(table, "users");
            assert_eq!(assignments.len(), 1);
            assert!(filter.is_none());
        }
        other => panic!("expected Update, got {:?}", other),
    }
}

#[test]
fn plan_select_full_scan() {
    let plan = plan_select(
        "users".to_string(),
        vec!["id".to_string(), "name".to_string()],
        None,
    );

    match plan {
        QueryPlan::FullScan {
            table,
            filter,
            columns,
        } => {
            assert_eq!(table, "users");
            assert!(filter.is_none());
            assert_eq!(columns.len(), 2);
        }
        other => panic!("expected FullScan, got {:?}", other),
    }
}

// ── Adaptive planner (Req 22) ──────────────────────────────────────

#[test]
fn choose_brute_force_for_low_cardinality() {
    let stats = PlannerStats {
        row_count: 1_000_000,
        filter_selectivity: 0.0001, // 100 rows out of 1M
    };
    assert_eq!(
        choose_search_strategy(&stats),
        SearchStrategy::BruteForceFiltered
    );
}

#[test]
fn choose_hnsw_for_high_cardinality() {
    let stats = PlannerStats {
        row_count: 1_000_000,
        filter_selectivity: 0.5, // 500K rows
    };
    assert_eq!(
        choose_search_strategy(&stats),
        SearchStrategy::HnswWithPostFilter
    );
}

#[test]
fn choose_brute_force_under_1000_rows() {
    let stats = PlannerStats {
        row_count: 10_000,
        filter_selectivity: 0.05, // 500 rows
    };
    assert_eq!(
        choose_search_strategy(&stats),
        SearchStrategy::BruteForceFiltered
    );
}

#[test]
fn choose_hnsw_at_1000_rows() {
    let stats = PlannerStats {
        row_count: 100_000,
        filter_selectivity: 0.01, // 1000 rows
    };
    assert_eq!(
        choose_search_strategy(&stats),
        SearchStrategy::HnswWithPostFilter
    );
}

#[test]
fn plan_semantic_search_without_filter() {
    let semantic = SemanticMatchExpr {
        column: "content".to_string(),
        query: "machine learning".to_string(),
        threshold: 0.8,
    };

    let plan = plan_semantic_search("docs".to_string(), semantic, None, None);

    match plan {
        QueryPlan::SemanticSearch {
            table,
            strategy,
            threshold,
            ..
        } => {
            assert_eq!(table, "docs");
            assert_eq!(strategy, SearchStrategy::HnswWithPostFilter);
            assert!((threshold - 0.8).abs() < f64::EPSILON);
        }
        other => panic!("expected SemanticSearch, got {:?}", other),
    }
}

#[test]
fn plan_hybrid_search_with_stats() {
    let semantic = SemanticMatchExpr {
        column: "content".to_string(),
        query: "rust database".to_string(),
        threshold: 0.7,
    };
    let filter = FilterExpr::Eq {
        column: "category".to_string(),
        value: Value::Text("tech".to_string()),
    };
    let stats = PlannerStats {
        row_count: 1_000_000,
        filter_selectivity: 0.0001,
    };

    let plan = plan_semantic_search(
        "docs".to_string(),
        semantic,
        Some(filter),
        Some(&stats),
    );

    match plan {
        QueryPlan::HybridSearch { strategy, .. } => {
            assert_eq!(strategy, SearchStrategy::BruteForceFiltered);
        }
        other => panic!("expected HybridSearch, got {:?}", other),
    }
}

#[test]
fn planner_stats_cardinality_estimation() {
    let stats = PlannerStats {
        row_count: 10_000,
        filter_selectivity: 0.01,
    };
    assert_eq!(stats.estimated_cardinality(), 100);
}

// ---------------------------------------------------------------------------
// `WHERE NOT DUPLICATE` plan carrying — task 35.5
// ---------------------------------------------------------------------------

#[test]
fn plan_select_carries_not_duplicate_predicate() {
    let plan = plan_select(
        "docs".to_string(),
        vec!["id".to_string()],
        Some(FilterExpr::NotDuplicate),
    );
    match plan {
        QueryPlan::FullScan {
            table,
            columns,
            filter,
        } => {
            assert_eq!(table, "docs");
            assert_eq!(columns, vec!["id"]);
            assert_eq!(filter, Some(FilterExpr::NotDuplicate));
        }
        other => panic!("expected FullScan, got {:?}", other),
    }
}

#[test]
fn plan_select_carries_composed_not_duplicate_predicate() {
    let composed = FilterExpr::And(
        Box::new(FilterExpr::Gt {
            column: "price".to_string(),
            value: Value::Float(4.0),
        }),
        Box::new(FilterExpr::NotDuplicate),
    );
    let plan = plan_select("docs".to_string(), Vec::new(), Some(composed.clone()));
    let QueryPlan::FullScan { filter, .. } = plan else {
        panic!("expected FullScan");
    };
    assert_eq!(filter, Some(composed));
}

#[test]
fn near_duplicate_group_column_name_is_stable() {
    // The canonical column name is part of the public contract —
    // any change must be deliberate and coordinated with the Task 35.4
    // background job, the executor, and the Lance exporter.
    assert_eq!(NEAR_DUPLICATE_GROUP_COLUMN, "_near_duplicate_group");
}
