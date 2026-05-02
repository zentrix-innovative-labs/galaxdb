//! Tests for the statistics collection module.

use super::*;
use crate::pax::{ColumnData, PaxBlock};
use galaxdb_common::ColumnType;

// ---------------------------------------------------------------------------
// HyperLogLog NDV accuracy tests
// ---------------------------------------------------------------------------

#[test]
fn hll_empty_returns_zero() {
    let hll = HyperLogLog::new();
    assert_eq!(hll.estimate(), 0);
}

#[test]
fn hll_single_value() {
    let mut hll = HyperLogLog::new();
    hll.add_hash(xxhash_rust::xxh3::xxh3_64(b"hello"));
    let est = hll.estimate();
    assert!(est >= 1 && est <= 3, "expected ~1, got {}", est);
}

#[test]
fn hll_ndv_accuracy_1000_distinct() {
    let mut hll = HyperLogLog::new();
    let actual_ndv = 1_000u64;
    for i in 0..actual_ndv {
        let hash = xxhash_rust::xxh3::xxh3_64(&i.to_le_bytes());
        hll.add_hash(hash);
    }
    let estimated = hll.estimate();
    let error = (estimated as f64 - actual_ndv as f64).abs() / actual_ndv as f64;
    // HyperLogLog with 16384 registers should have ~1% error,
    // allow up to 5% for small cardinalities
    assert!(
        error < 0.05,
        "NDV estimate {} for actual {} has error {:.2}%",
        estimated,
        actual_ndv,
        error * 100.0
    );
}

#[test]
fn hll_ndv_accuracy_100000_distinct() {
    let mut hll = HyperLogLog::new();
    let actual_ndv = 100_000u64;
    for i in 0..actual_ndv {
        let hash = xxhash_rust::xxh3::xxh3_64(&i.to_le_bytes());
        hll.add_hash(hash);
    }
    let estimated = hll.estimate();
    let error = (estimated as f64 - actual_ndv as f64).abs() / actual_ndv as f64;
    // Should be within ~2% for large cardinalities
    assert!(
        error < 0.03,
        "NDV estimate {} for actual {} has error {:.2}%",
        estimated,
        actual_ndv,
        error * 100.0
    );
}

#[test]
fn hll_duplicate_values_dont_inflate() {
    let mut hll = HyperLogLog::new();
    // Add 100 distinct values, each repeated 100 times
    for i in 0..100u64 {
        let hash = xxhash_rust::xxh3::xxh3_64(&i.to_le_bytes());
        for _ in 0..100 {
            hll.add_hash(hash);
        }
    }
    let estimated = hll.estimate();
    let error = (estimated as f64 - 100.0).abs() / 100.0;
    assert!(
        error < 0.10,
        "NDV estimate {} for actual 100 has error {:.2}%",
        estimated,
        error * 100.0
    );
}

#[test]
fn hll_merge_combines_sketches() {
    let mut hll1 = HyperLogLog::new();
    let mut hll2 = HyperLogLog::new();

    // Add 0..500 to hll1, 500..1000 to hll2
    for i in 0..500u64 {
        hll1.add_hash(xxhash_rust::xxh3::xxh3_64(&i.to_le_bytes()));
    }
    for i in 500..1000u64 {
        hll2.add_hash(xxhash_rust::xxh3::xxh3_64(&i.to_le_bytes()));
    }

    hll1.merge(&hll2);
    let estimated = hll1.estimate();
    let error = (estimated as f64 - 1000.0).abs() / 1000.0;
    assert!(
        error < 0.05,
        "Merged NDV estimate {} for actual 1000 has error {:.2}%",
        estimated,
        error * 100.0
    );
}

// ---------------------------------------------------------------------------
// Equi-height histogram tests
// ---------------------------------------------------------------------------

#[test]
fn histogram_empty_input() {
    let hist = EquiHeightHistogram::build(&[], 10);
    assert!(hist.buckets.is_empty());
    assert_eq!(hist.total_count, 0);
}

#[test]
fn histogram_single_value() {
    let values = vec![vec![42u8]];
    let hist = EquiHeightHistogram::build(&values, 10);
    assert_eq!(hist.buckets.len(), 1);
    assert_eq!(hist.total_count, 1);
    assert_eq!(hist.buckets[0].lower, vec![42u8]);
    assert_eq!(hist.buckets[0].upper, vec![42u8]);
}

#[test]
fn histogram_bucket_distribution() {
    // 1000 values, 10 buckets → ~100 values per bucket
    let mut values: Vec<Vec<u8>> = (0..=255u8)
        .flat_map(|v| std::iter::repeat(vec![v]).take(4))
        .collect();
    // 1024 values total
    values.sort();

    let hist = EquiHeightHistogram::build(&values, 10);
    assert_eq!(hist.buckets.len(), 10);
    assert_eq!(hist.total_count, 1024);

    // Each bucket should have approximately 102-103 values
    for bucket in &hist.buckets {
        assert!(
            bucket.count >= 100 && bucket.count <= 106,
            "bucket count {} not in expected range",
            bucket.count
        );
    }
}

#[test]
fn histogram_buckets_cover_full_range() {
    let values: Vec<Vec<u8>> = (0..100u32)
        .map(|v| v.to_le_bytes().to_vec())
        .collect();
    let mut sorted = values;
    sorted.sort();

    let hist = EquiHeightHistogram::build(&sorted, 10);
    assert_eq!(hist.buckets.len(), 10);

    // First bucket should start at min, last bucket should end at max
    assert_eq!(hist.buckets[0].lower, sorted[0]);
    assert_eq!(hist.buckets.last().unwrap().upper, *sorted.last().unwrap());
}

#[test]
fn histogram_equality_estimate() {
    // Build histogram from 100 distinct values
    let values: Vec<Vec<u8>> = (0..100u8).map(|v| vec![v]).collect();
    let hist = EquiHeightHistogram::build(&values, 10);

    // Equality estimate for a value in the middle
    let sel = hist.estimate_equality(&[50]);
    // With 10 buckets of 10 values each, NDV per bucket = 10,
    // so selectivity ≈ (10/10) / 100 = 0.01
    assert!(
        sel > 0.0 && sel < 0.1,
        "equality selectivity {} out of expected range",
        sel
    );
}

#[test]
fn histogram_range_estimate() {
    let values: Vec<Vec<u8>> = (0..100u8).map(|v| vec![v]).collect();
    let hist = EquiHeightHistogram::build(&values, 10);

    // Range covering roughly half the values
    let sel = hist.estimate_range(&[25], &[75]);
    assert!(
        sel > 0.3 && sel < 0.8,
        "range selectivity {} out of expected range",
        sel
    );
}

#[test]
fn histogram_range_outside_returns_zero() {
    let values: Vec<Vec<u8>> = (10..20u8).map(|v| vec![v]).collect();
    let hist = EquiHeightHistogram::build(&values, 5);

    let sel = hist.estimate_range(&[100], &[200]);
    assert_eq!(sel, 0.0);
}

// ---------------------------------------------------------------------------
// Correlation statistics tests
// ---------------------------------------------------------------------------

#[test]
fn correlation_perfect_positive() {
    let names = vec!["a".to_string(), "b".to_string()];
    let values = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![2.0, 4.0, 6.0, 8.0, 10.0], // b = 2*a
    ];
    let stats = CorrelationStats::compute(names, &values);

    let corr = stats.get_correlation(0, 1).unwrap();
    assert!(
        (corr - 1.0).abs() < 1e-10,
        "expected perfect positive correlation, got {}",
        corr
    );
}

#[test]
fn correlation_perfect_negative() {
    let names = vec!["a".to_string(), "b".to_string()];
    let values = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![10.0, 8.0, 6.0, 4.0, 2.0], // b = -2*a + 12
    ];
    let stats = CorrelationStats::compute(names, &values);

    let corr = stats.get_correlation(0, 1).unwrap();
    assert!(
        (corr - (-1.0)).abs() < 1e-10,
        "expected perfect negative correlation, got {}",
        corr
    );
}

#[test]
fn correlation_diagonal_is_one() {
    let names = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let values = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];
    let stats = CorrelationStats::compute(names, &values);

    for i in 0..3 {
        let corr = stats.get_correlation(i, i).unwrap();
        assert!(
            (corr - 1.0).abs() < 1e-10,
            "diagonal correlation[{},{}] = {}, expected 1.0",
            i,
            i,
            corr
        );
    }
}

#[test]
fn correlation_symmetric() {
    let names = vec!["a".to_string(), "b".to_string()];
    let values = vec![
        vec![1.0, 3.0, 2.0, 5.0, 4.0],
        vec![2.0, 1.0, 4.0, 3.0, 5.0],
    ];
    let stats = CorrelationStats::compute(names, &values);

    let corr_ab = stats.get_correlation(0, 1).unwrap();
    let corr_ba = stats.get_correlation(1, 0).unwrap();
    assert!(
        (corr_ab - corr_ba).abs() < 1e-10,
        "correlation not symmetric: corr(a,b)={}, corr(b,a)={}",
        corr_ab,
        corr_ba
    );
}

#[test]
fn correlation_zero_stddev_returns_zero() {
    let names = vec!["a".to_string(), "b".to_string()];
    let values = vec![
        vec![5.0, 5.0, 5.0, 5.0], // constant column
        vec![1.0, 2.0, 3.0, 4.0],
    ];
    let stats = CorrelationStats::compute(names, &values);

    let corr = stats.get_correlation(0, 1).unwrap();
    assert_eq!(corr, 0.0, "constant column should have 0 correlation");
}

// ---------------------------------------------------------------------------
// Reservoir sampler tests
// ---------------------------------------------------------------------------

#[test]
fn reservoir_sampler_small_stream() {
    let mut sampler = ReservoirSampler::new(100, 42);
    for i in 0..50u64 {
        sampler.add(vec![i.to_le_bytes().to_vec()]);
    }
    let samples = sampler.into_samples();
    assert_eq!(samples.len(), 50);
}

#[test]
fn reservoir_sampler_large_stream() {
    let mut sampler = ReservoirSampler::new(100, 42);
    for i in 0..10_000u64 {
        sampler.add(vec![i.to_le_bytes().to_vec()]);
    }
    let samples = sampler.into_samples();
    assert_eq!(samples.len(), 100);
}

#[test]
fn reservoir_sampler_total_seen() {
    let mut sampler = ReservoirSampler::new(10, 42);
    for i in 0..500u64 {
        sampler.add(vec![i.to_le_bytes().to_vec()]);
    }
    assert_eq!(sampler.total_seen(), 500);
}

// ---------------------------------------------------------------------------
// Full ANALYZE pipeline tests
// ---------------------------------------------------------------------------

fn make_int32_column(values: &[i32]) -> ColumnData {
    ColumnData {
        col_type: ColumnType::Int32,
        values: values.iter().map(|v| v.to_le_bytes().to_vec()).collect(),
    }
}

fn make_test_blocks(col1: &[i32], col2: &[i32]) -> Vec<PaxBlock> {
    assert_eq!(col1.len(), col2.len());
    let columns = vec![make_int32_column(col1), make_int32_column(col2)];
    let block = PaxBlock::write(1, 1000, &columns).expect("failed to write PAX block");
    vec![block]
}

#[test]
fn analyze_empty_blocks() {
    let stats = analyze_blocks(&[], &[], &AnalyzeConfig::default(), 1000);
    assert_eq!(stats.row_count, 0);
    assert!(stats.columns.is_empty());
    assert!(stats.multi_column.is_empty());
    assert_eq!(stats.last_analyzed, Some(1000));
}

#[test]
fn analyze_computes_ndv() {
    // 100 distinct values
    let col1: Vec<i32> = (0..100).collect();
    let col2: Vec<i32> = (0..100).map(|i| i * 2).collect();
    let blocks = make_test_blocks(&col1, &col2);
    let column_names = vec!["a".to_string(), "b".to_string()];

    let stats = analyze_blocks(&blocks, &column_names, &AnalyzeConfig::default(), 2000);

    assert_eq!(stats.row_count, 100);

    let a_stats = stats.columns.get("a").expect("missing column a stats");
    let b_stats = stats.columns.get("b").expect("missing column b stats");

    // NDV should be close to 100
    let a_error = (a_stats.ndv as f64 - 100.0).abs() / 100.0;
    assert!(
        a_error < 0.10,
        "column a NDV {} has error {:.2}%",
        a_stats.ndv,
        a_error * 100.0
    );

    let b_error = (b_stats.ndv as f64 - 100.0).abs() / 100.0;
    assert!(
        b_error < 0.10,
        "column b NDV {} has error {:.2}%",
        b_stats.ndv,
        b_error * 100.0
    );
}

#[test]
fn analyze_builds_histogram() {
    let col1: Vec<i32> = (0..200).collect();
    let col2: Vec<i32> = (0..200).map(|i| i * 3).collect();
    let blocks = make_test_blocks(&col1, &col2);
    let column_names = vec!["x".to_string(), "y".to_string()];

    let config = AnalyzeConfig {
        histogram_buckets: 20,
        ..AnalyzeConfig::default()
    };
    let stats = analyze_blocks(&blocks, &column_names, &config, 3000);

    let x_stats = stats.columns.get("x").expect("missing column x stats");
    assert_eq!(x_stats.histogram.buckets.len(), 20);
    assert_eq!(x_stats.histogram.total_count, 200);

    // Each bucket should have ~10 values
    for bucket in &x_stats.histogram.buckets {
        assert_eq!(bucket.count, 10);
    }
}

#[test]
fn analyze_computes_correlations() {
    // Perfectly correlated columns: b = 2*a
    let col1: Vec<i32> = (1..=100).collect();
    let col2: Vec<i32> = (1..=100).map(|i| i * 2).collect();
    let blocks = make_test_blocks(&col1, &col2);
    let column_names = vec!["a".to_string(), "b".to_string()];

    let stats = analyze_blocks(&blocks, &column_names, &AnalyzeConfig::default(), 4000);

    assert!(!stats.multi_column.is_empty(), "should have correlation stats");
    let corr = &stats.multi_column[0];
    assert_eq!(corr.columns.len(), 2);

    let r = corr.get_correlation(0, 1).unwrap();
    assert!(
        (r - 1.0).abs() < 0.01,
        "expected strong positive correlation, got {}",
        r
    );
}

#[test]
fn analyze_null_fraction() {
    // Create a block where some values are "null" (empty)
    // We'll use a custom approach: create a column with some empty values
    let values: Vec<Vec<u8>> = (0..80i32).map(|v| v.to_le_bytes().to_vec()).collect();
    // Add 20 "null" rows (empty bytes) — but PAX blocks require fixed-width,
    // so we test null_fraction = 0 for non-null data
    let col1 = ColumnData {
        col_type: ColumnType::Int32,
        values: values.clone(),
    };
    let col2 = ColumnData {
        col_type: ColumnType::Int32,
        values: (0..80i32).map(|v| (v * 2).to_le_bytes().to_vec()).collect(),
    };

    let block = PaxBlock::write(1, 1000, &[col1, col2]).expect("write block");
    let column_names = vec!["a".to_string(), "b".to_string()];

    let stats = analyze_blocks(&[block], &column_names, &AnalyzeConfig::default(), 5000);

    let a_stats = stats.columns.get("a").unwrap();
    // All values are non-null, so null_fraction should be 0
    assert_eq!(a_stats.null_fraction, 0.0);
}

// ---------------------------------------------------------------------------
// TableStatistics selectivity estimation tests
// ---------------------------------------------------------------------------

#[test]
fn selectivity_equality_with_stats() {
    let col1: Vec<i32> = (0..100).collect();
    let col2: Vec<i32> = (0..100).collect();
    let blocks = make_test_blocks(&col1, &col2);
    let column_names = vec!["id".to_string(), "val".to_string()];

    let stats = analyze_blocks(&blocks, &column_names, &AnalyzeConfig::default(), 6000);

    let sel = stats.estimate_equality_selectivity("id", &50i32.to_le_bytes());
    assert!(
        sel > 0.0 && sel < 0.1,
        "equality selectivity {} out of range",
        sel
    );
}

#[test]
fn selectivity_range_with_stats() {
    let col1: Vec<i32> = (0..100).collect();
    let col2: Vec<i32> = (0..100).collect();
    let blocks = make_test_blocks(&col1, &col2);
    let column_names = vec!["id".to_string(), "val".to_string()];

    let stats = analyze_blocks(&blocks, &column_names, &AnalyzeConfig::default(), 7000);

    let sel = stats.estimate_range_selectivity(
        "id",
        &25i32.to_le_bytes(),
        &75i32.to_le_bytes(),
    );
    assert!(
        sel > 0.2 && sel < 0.9,
        "range selectivity {} out of range",
        sel
    );
}

#[test]
fn selectivity_unknown_column_returns_default() {
    let stats = TableStatistics {
        row_count: 1000,
        ..Default::default()
    };

    let sel = stats.estimate_equality_selectivity("unknown", &[1, 2, 3]);
    assert_eq!(sel, 0.01); // default for unknown column

    let sel = stats.estimate_range_selectivity("unknown", &[0], &[100]);
    assert_eq!(sel, 0.33); // default for unknown column
}

#[test]
fn cardinality_estimation() {
    let stats = TableStatistics {
        row_count: 10_000,
        ..Default::default()
    };

    assert_eq!(stats.estimate_cardinality(0.01), 100);
    assert_eq!(stats.estimate_cardinality(0.5), 5000);
    assert_eq!(stats.estimate_cardinality(1.0), 10000);
    assert_eq!(stats.estimate_cardinality(0.0), 0);
}

// ---------------------------------------------------------------------------
// StatisticsCatalog tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn catalog_store_and_get() {
    let catalog = StatisticsCatalog::new();

    let stats = TableStatistics {
        row_count: 42,
        ..Default::default()
    };

    catalog.store("test_table".to_string(), stats).await;

    let retrieved = catalog.get("test_table").await;
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().row_count, 42);
}

#[tokio::test]
async fn catalog_get_missing_returns_none() {
    let catalog = StatisticsCatalog::new();
    assert!(catalog.get("nonexistent").await.is_none());
}

#[tokio::test]
async fn catalog_remove() {
    let catalog = StatisticsCatalog::new();

    catalog
        .store(
            "t".to_string(),
            TableStatistics {
                row_count: 1,
                ..Default::default()
            },
        )
        .await;

    catalog.remove("t").await;
    assert!(catalog.get("t").await.is_none());
}

// ---------------------------------------------------------------------------
// Background ANALYZE spawn test
// ---------------------------------------------------------------------------

#[tokio::test]
async fn spawn_analyze_stores_in_catalog() {
    let col1: Vec<i32> = (0..50).collect();
    let col2: Vec<i32> = (0..50).collect();
    let blocks = make_test_blocks(&col1, &col2);
    let column_names = vec!["a".to_string(), "b".to_string()];
    let catalog = StatisticsCatalog::new();

    let handle = spawn_analyze(
        blocks,
        column_names,
        "my_table".to_string(),
        catalog.clone(),
        AnalyzeConfig::default(),
        8000,
    );

    handle.await.expect("analyze task panicked");

    let stats = catalog.get("my_table").await.expect("stats not found");
    assert_eq!(stats.row_count, 50);
    assert!(stats.columns.contains_key("a"));
    assert!(stats.columns.contains_key("b"));
    assert_eq!(stats.last_analyzed, Some(8000));
}
