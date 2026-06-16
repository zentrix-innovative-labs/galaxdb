//! Statistics Collection for GalaxDB.
//!
//! Implements the `ANALYZE` command subsystem: per-column NDV estimation via
//! HyperLogLog, equi-height histograms, null fraction tracking, and
//! multi-column correlation statistics following the PostgreSQL extended
//! statistics model.
//!
//! Statistics are collected as a background tokio task using reservoir sampling
//! of PAX blocks, and stored in the catalog for use by the query planner.

#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::sync::Arc;

use galaxdb_common::Timestamp;
use tokio::sync::RwLock;

use crate::pax::PaxBlock;

// ---------------------------------------------------------------------------
// HyperLogLog NDV estimator
// ---------------------------------------------------------------------------

/// Number of HyperLogLog registers (2^14 = 16384 for ~1% standard error).
const HLL_NUM_REGISTERS: usize = 1 << 14; // 16384
/// Number of bits used for the register index (14).
const HLL_INDEX_BITS: u32 = 14;

/// A HyperLogLog sketch for estimating the number of distinct values (NDV).
///
/// Uses 2^14 = 16384 registers, giving approximately 1% standard error.
#[derive(Debug, Clone)]
pub struct HyperLogLog {
    registers: Vec<u8>,
}

impl HyperLogLog {
    /// Create a new empty HyperLogLog sketch.
    pub fn new() -> Self {
        Self {
            registers: vec![0u8; HLL_NUM_REGISTERS],
        }
    }

    /// Add a pre-hashed 64-bit value to the sketch.
    pub fn add_hash(&mut self, hash: u64) {
        let index = (hash >> (64 - HLL_INDEX_BITS)) as usize;
        let remaining = (hash << HLL_INDEX_BITS) | (1 << (HLL_INDEX_BITS - 1));
        let rank = remaining.leading_zeros() as u8 + 1;
        if rank > self.registers[index] {
            self.registers[index] = rank;
        }
    }

    /// Estimate the number of distinct values seen so far.
    pub fn estimate(&self) -> u64 {
        let m = HLL_NUM_REGISTERS as f64;
        // alpha_m constant for m = 16384
        let alpha_m = 0.7213 / (1.0 + 1.079 / m);

        let raw_estimate: f64 = {
            let sum: f64 = self
                .registers
                .iter()
                .map(|&r| 2.0_f64.powi(-(r as i32)))
                .sum();
            alpha_m * m * m / sum
        };

        // Small range correction
        if raw_estimate <= 2.5 * m {
            let zeros = self.registers.iter().filter(|&&r| r == 0).count() as f64;
            if zeros > 0.0 {
                (m * (m / zeros).ln()) as u64
            } else {
                raw_estimate as u64
            }
        } else if raw_estimate <= (1u64 << 32) as f64 / 30.0 {
            // Intermediate range — no correction needed
            raw_estimate as u64
        } else {
            // Large range correction
            let two_32 = (1u64 << 32) as f64;
            (-two_32 * (1.0 - raw_estimate / two_32).ln()) as u64
        }
    }

    /// Merge another HyperLogLog sketch into this one (element-wise max).
    pub fn merge(&mut self, other: &HyperLogLog) {
        for (a, &b) in self.registers.iter_mut().zip(other.registers.iter()) {
            if b > *a {
                *a = b;
            }
        }
    }
}

impl Default for HyperLogLog {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Equi-height histogram
// ---------------------------------------------------------------------------

/// Default number of histogram buckets.
pub const DEFAULT_HISTOGRAM_BUCKETS: usize = 100;

/// A single bucket in an equi-height histogram.
#[derive(Debug, Clone, PartialEq)]
pub struct HistogramBucket {
    /// Lower bound of this bucket (inclusive), as raw bytes.
    pub lower: Vec<u8>,
    /// Upper bound of this bucket (inclusive), as raw bytes.
    pub upper: Vec<u8>,
    /// Number of values in this bucket.
    pub count: u64,
    /// Number of distinct values in this bucket.
    pub ndv: u64,
}

/// An equi-height (equi-depth) histogram where each bucket contains
/// approximately the same number of values.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct EquiHeightHistogram {
    /// The histogram buckets, ordered by value range.
    pub buckets: Vec<HistogramBucket>,
    /// Total number of values used to build this histogram.
    pub total_count: u64,
}

impl EquiHeightHistogram {
    /// Build an equi-height histogram from sorted sample values.
    ///
    /// `sorted_values` must be pre-sorted in ascending order.
    /// `num_buckets` is the target number of buckets (default 100).
    pub fn build(sorted_values: &[Vec<u8>], num_buckets: usize) -> Self {
        if sorted_values.is_empty() {
            return Self {
                buckets: Vec::new(),
                total_count: 0,
            };
        }

        let total = sorted_values.len();
        let actual_buckets = num_buckets.min(total);
        let values_per_bucket = total / actual_buckets;
        let remainder = total % actual_buckets;

        let mut buckets = Vec::with_capacity(actual_buckets);
        let mut offset = 0;

        for i in 0..actual_buckets {
            // Distribute remainder across first `remainder` buckets
            let bucket_size = values_per_bucket + if i < remainder { 1 } else { 0 };
            let end = offset + bucket_size;

            let bucket_values = &sorted_values[offset..end];
            let ndv = {
                let mut distinct = 1u64;
                for j in 1..bucket_values.len() {
                    if bucket_values[j] != bucket_values[j - 1] {
                        distinct += 1;
                    }
                }
                distinct
            };

            buckets.push(HistogramBucket {
                lower: bucket_values[0].clone(),
                upper: bucket_values[bucket_values.len() - 1].clone(),
                count: bucket_size as u64,
                ndv,
            });

            offset = end;
        }

        Self {
            buckets,
            total_count: total as u64,
        }
    }

    /// Estimate the fraction of rows matching an equality predicate.
    pub fn estimate_equality(&self, value: &[u8]) -> f64 {
        if self.buckets.is_empty() || self.total_count == 0 {
            return 0.0;
        }

        for bucket in &self.buckets {
            if value >= bucket.lower.as_slice() && value <= bucket.upper.as_slice() {
                // Assume uniform distribution within bucket
                if bucket.ndv == 0 {
                    return 0.0;
                }
                return (bucket.count as f64 / bucket.ndv as f64) / self.total_count as f64;
            }
        }

        0.0
    }

    /// Estimate the fraction of rows in a range [low, high] (inclusive).
    pub fn estimate_range(&self, low: &[u8], high: &[u8]) -> f64 {
        if self.buckets.is_empty() || self.total_count == 0 {
            return 0.0;
        }

        let mut matching = 0.0_f64;

        for bucket in &self.buckets {
            if bucket.upper.as_slice() < low || bucket.lower.as_slice() > high {
                // Bucket entirely outside range
                continue;
            }

            if bucket.lower.as_slice() >= low && bucket.upper.as_slice() <= high {
                // Bucket entirely inside range
                matching += bucket.count as f64;
            } else {
                // Partial overlap — assume uniform distribution within bucket
                matching += bucket.count as f64 * 0.5;
            }
        }

        matching / self.total_count as f64
    }
}



// ---------------------------------------------------------------------------
// Per-column statistics
// ---------------------------------------------------------------------------

/// Per-column statistics collected by the ANALYZE command.
#[derive(Debug, Clone)]
pub struct ColumnStats {
    /// Estimated number of distinct values (via HyperLogLog).
    pub ndv: u64,
    /// Fraction of NULL values in the column (0.0 to 1.0).
    pub null_fraction: f64,
    /// Equi-height histogram with configurable bucket count (default 100).
    pub histogram: EquiHeightHistogram,
}

impl Default for ColumnStats {
    fn default() -> Self {
        Self {
            ndv: 0,
            null_fraction: 0.0,
            histogram: EquiHeightHistogram::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-column correlation statistics
// ---------------------------------------------------------------------------

/// Multi-column correlation statistics following the PostgreSQL extended
/// statistics model. Stores Pearson correlation coefficients between pairs
/// of numeric columns.
#[derive(Debug, Clone)]
pub struct CorrelationStats {
    /// Column names involved in this correlation group.
    pub columns: Vec<String>,
    /// Flattened correlation matrix (row-major, N×N where N = columns.len()).
    /// `correlation_matrix[i * N + j]` is the Pearson correlation between
    /// column `i` and column `j`.
    pub correlation_matrix: Vec<f64>,
}

impl CorrelationStats {
    /// Compute Pearson correlation statistics for a set of numeric columns.
    ///
    /// `column_names` — names of the columns.
    /// `column_values` — parallel vectors of f64 values for each column.
    ///   All vectors must have the same length.
    pub fn compute(column_names: Vec<String>, column_values: &[Vec<f64>]) -> Self {
        let n_cols = column_names.len();
        let mut matrix = vec![0.0_f64; n_cols * n_cols];

        if column_values.is_empty() || column_values[0].is_empty() {
            return Self {
                columns: column_names,
                correlation_matrix: matrix,
            };
        }

        let n_rows = column_values[0].len();

        // Compute means
        let means: Vec<f64> = column_values
            .iter()
            .map(|vals| vals.iter().sum::<f64>() / n_rows as f64)
            .collect();

        // Compute standard deviations
        let stddevs: Vec<f64> = column_values
            .iter()
            .zip(means.iter())
            .map(|(vals, &mean)| {
                let variance =
                    vals.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n_rows as f64;
                variance.sqrt()
            })
            .collect();

        // Compute pairwise Pearson correlations
        for i in 0..n_cols {
            for j in 0..n_cols {
                if i == j {
                    matrix[i * n_cols + j] = 1.0;
                } else if stddevs[i] == 0.0 || stddevs[j] == 0.0 {
                    matrix[i * n_cols + j] = 0.0;
                } else {
                    let covariance: f64 = column_values[i]
                        .iter()
                        .zip(column_values[j].iter())
                        .map(|(&a, &b)| (a - means[i]) * (b - means[j]))
                        .sum::<f64>()
                        / n_rows as f64;
                    matrix[i * n_cols + j] = covariance / (stddevs[i] * stddevs[j]);
                }
            }
        }

        Self {
            columns: column_names,
            correlation_matrix: matrix,
        }
    }

    /// Get the correlation between two columns by index.
    pub fn get_correlation(&self, col_i: usize, col_j: usize) -> Option<f64> {
        let n = self.columns.len();
        if col_i < n && col_j < n {
            Some(self.correlation_matrix[col_i * n + col_j])
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Table-level statistics
// ---------------------------------------------------------------------------

/// Aggregate statistics for a table, collected by the ANALYZE command.
#[derive(Debug, Clone, Default)]
pub struct TableStatistics {
    /// Total number of rows in the table.
    pub row_count: u64,
    /// Per-column statistics keyed by column name.
    pub columns: HashMap<String, ColumnStats>,
    /// Multi-column correlation statistics groups.
    pub multi_column: Vec<CorrelationStats>,
    /// Timestamp of the last ANALYZE run, or `None` if never analyzed.
    pub last_analyzed: Option<Timestamp>,
}



impl TableStatistics {
    /// Estimate the selectivity (fraction of rows) for an equality predicate
    /// on a single column.
    ///
    /// Returns a value between 0.0 and 1.0.
    pub fn estimate_equality_selectivity(&self, column: &str, value: &[u8]) -> f64 {
        if self.row_count == 0 {
            return 0.0;
        }
        match self.columns.get(column) {
            Some(stats) => {
                // Use histogram if available, otherwise use 1/NDV
                if !stats.histogram.buckets.is_empty() {
                    stats.histogram.estimate_equality(value)
                } else if stats.ndv > 0 {
                    1.0 / stats.ndv as f64
                } else {
                    0.0
                }
            }
            None => {
                // No statistics — assume uniform distribution, default 1%
                0.01
            }
        }
    }

    /// Estimate the selectivity (fraction of rows) for a range predicate
    /// on a single column.
    ///
    /// Returns a value between 0.0 and 1.0.
    pub fn estimate_range_selectivity(&self, column: &str, low: &[u8], high: &[u8]) -> f64 {
        if self.row_count == 0 {
            return 0.0;
        }
        match self.columns.get(column) {
            Some(stats) if !stats.histogram.buckets.is_empty() => {
                stats.histogram.estimate_range(low, high)
            }
            // Default: 33% for range queries without statistics
            _ => 0.33,
        }
    }

    /// Estimate the number of rows matching a filter for use by the adaptive
    /// query planner (Req 22). This is a simplified cardinality estimator.
    pub fn estimate_cardinality(&self, selectivity: f64) -> u64 {
        (self.row_count as f64 * selectivity.clamp(0.0, 1.0)) as u64
    }
}

// ---------------------------------------------------------------------------
// Statistics catalog (thread-safe storage)
// ---------------------------------------------------------------------------

/// Thread-safe catalog storing statistics for all tables.
/// The query planner reads from this; the ANALYZE background task writes to it.
#[derive(Debug, Clone)]
pub struct StatisticsCatalog {
    inner: Arc<RwLock<HashMap<String, TableStatistics>>>,
}

impl StatisticsCatalog {
    /// Create a new empty statistics catalog.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Store statistics for a table, replacing any previous statistics.
    pub async fn store(&self, table_name: String, stats: TableStatistics) {
        let mut catalog = self.inner.write().await;
        catalog.insert(table_name, stats);
    }

    /// Retrieve statistics for a table, if available.
    pub async fn get(&self, table_name: &str) -> Option<TableStatistics> {
        let catalog = self.inner.read().await;
        catalog.get(table_name).cloned()
    }

    /// Remove statistics for a table (e.g., on DROP TABLE).
    pub async fn remove(&self, table_name: &str) {
        let mut catalog = self.inner.write().await;
        catalog.remove(table_name);
    }
}

impl Default for StatisticsCatalog {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Reservoir sampling
// ---------------------------------------------------------------------------

/// Reservoir sampler for selecting a uniform random sample from a stream
/// of PAX blocks without knowing the total count in advance.
pub struct ReservoirSampler {
    /// Maximum number of items to keep in the reservoir.
    capacity: usize,
    /// The reservoir of sampled items (column values as raw bytes).
    /// Each entry is a vector of column values for one row.
    reservoir: Vec<Vec<Vec<u8>>>,
    /// Total number of items seen so far.
    seen: usize,
    /// Simple RNG state (xorshift64).
    rng_state: u64,
}

impl ReservoirSampler {
    /// Create a new reservoir sampler with the given capacity.
    pub fn new(capacity: usize, seed: u64) -> Self {
        Self {
            capacity,
            reservoir: Vec::with_capacity(capacity),
            seen: 0,
            rng_state: seed.max(1), // Ensure non-zero seed
        }
    }

    /// Add a row (vector of column values) to the reservoir.
    pub fn add(&mut self, row: Vec<Vec<u8>>) {
        self.seen += 1;
        if self.reservoir.len() < self.capacity {
            self.reservoir.push(row);
        } else {
            let j = self.next_random() % self.seen;
            if j < self.capacity {
                self.reservoir[j] = row;
            }
        }
    }

    /// Return the sampled rows.
    pub fn into_samples(self) -> Vec<Vec<Vec<u8>>> {
        self.reservoir
    }

    /// Return the total number of items seen.
    pub fn total_seen(&self) -> usize {
        self.seen
    }

    /// Simple xorshift64 PRNG.
    fn next_random(&mut self) -> usize {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x as usize
    }
}

// ---------------------------------------------------------------------------
// ANALYZE execution
// ---------------------------------------------------------------------------

/// Configuration for the ANALYZE command.
#[derive(Debug, Clone)]
pub struct AnalyzeConfig {
    /// Maximum number of rows to sample (reservoir size).
    pub sample_size: usize,
    /// Number of histogram buckets.
    pub histogram_buckets: usize,
    /// Random seed for reservoir sampling.
    pub seed: u64,
}

impl Default for AnalyzeConfig {
    fn default() -> Self {
        Self {
            sample_size: 30_000,
            histogram_buckets: DEFAULT_HISTOGRAM_BUCKETS,
            seed: 42,
        }
    }
}

/// Analyze a set of PAX blocks and produce table statistics.
///
/// This is the core ANALYZE logic. It:
/// 1. Reservoir-samples rows from the provided PAX blocks.
/// 2. Computes per-column NDV via HyperLogLog.
/// 3. Builds equi-height histograms from sorted samples.
/// 4. Computes null fractions.
/// 5. Computes multi-column correlations for numeric column pairs.
///
/// The `column_names` must correspond to the columns in the PAX blocks.
pub fn analyze_blocks(
    blocks: &[PaxBlock],
    column_names: &[String],
    config: &AnalyzeConfig,
    current_timestamp: Timestamp,
) -> TableStatistics {
    if blocks.is_empty() || column_names.is_empty() {
        return TableStatistics {
            row_count: 0,
            columns: HashMap::new(),
            multi_column: Vec::new(),
            last_analyzed: Some(current_timestamp),
        };
    }

    let num_columns = column_names.len();

    // Phase 1: Reservoir sampling + HyperLogLog
    let mut sampler = ReservoirSampler::new(config.sample_size, config.seed);
    let mut hlls: Vec<HyperLogLog> = (0..num_columns).map(|_| HyperLogLog::new()).collect();
    let mut null_counts: Vec<u64> = vec![0; num_columns];
    let mut total_rows: u64 = 0;

    for block in blocks {
        let row_count = block.header.row_count as usize;
        total_rows += row_count as u64;

        // Read all columns from this block
        let mut block_columns: Vec<Vec<Vec<u8>>> = Vec::with_capacity(num_columns);
        for col_idx in 0..num_columns.min(block.header.column_descriptors.len()) {
            match block.read_column(col_idx) {
                Ok(values) => block_columns.push(values),
                Err(_) => block_columns.push(vec![Vec::new(); row_count]),
            }
        }

        // Pad if block has fewer columns than expected
        while block_columns.len() < num_columns {
            block_columns.push(vec![Vec::new(); row_count]);
        }

        // Process each row
        for row_idx in 0..row_count {
            let mut row_values: Vec<Vec<u8>> = Vec::with_capacity(num_columns);
            for (col_idx, col_data) in block_columns.iter().enumerate() {
                let value = if row_idx < col_data.len() {
                    col_data[row_idx].clone()
                } else {
                    Vec::new()
                };

                // Track nulls (empty value = null for our purposes)
                if value.is_empty() {
                    null_counts[col_idx] += 1;
                } else {
                    // Add to HyperLogLog
                    let hash = xxhash_rust::xxh3::xxh3_64(&value);
                    hlls[col_idx].add_hash(hash);
                }

                row_values.push(value);
            }
            sampler.add(row_values);
        }
    }

    // Phase 2: Build per-column statistics from samples
    let samples = sampler.into_samples();
    let mut column_stats: HashMap<String, ColumnStats> = HashMap::new();

    for (col_idx, col_name) in column_names.iter().enumerate() {
        let ndv = hlls[col_idx].estimate();
        let null_fraction = if total_rows > 0 {
            null_counts[col_idx] as f64 / total_rows as f64
        } else {
            0.0
        };

        // Extract non-null sample values for this column and sort them
        let mut sample_values: Vec<Vec<u8>> = samples
            .iter()
            .filter_map(|row| {
                let val = row.get(col_idx)?;
                if val.is_empty() {
                    None
                } else {
                    Some(val.clone())
                }
            })
            .collect();
        sample_values.sort();

        let histogram =
            EquiHeightHistogram::build(&sample_values, config.histogram_buckets);

        column_stats.insert(
            col_name.clone(),
            ColumnStats {
                ndv,
                null_fraction,
                histogram,
            },
        );
    }

    // Phase 3: Compute multi-column correlations for numeric columns
    let numeric_col_indices: Vec<usize> = column_names
        .iter()
        .enumerate()
        .filter(|(col_idx, _)| {
            blocks
                .first()
                .and_then(|b| b.header.column_descriptors.get(*col_idx))
                .map(|desc| is_numeric_type(&desc.col_type))
                .unwrap_or(false)
        })
        .map(|(idx, _)| idx)
        .collect();

    let multi_column = if numeric_col_indices.len() >= 2 {
        let numeric_names: Vec<String> = numeric_col_indices
            .iter()
            .map(|&idx| column_names[idx].clone())
            .collect();

        let numeric_values: Vec<Vec<f64>> = numeric_col_indices
            .iter()
            .map(|&col_idx| {
                samples
                    .iter()
                    .filter_map(|row| {
                        let val = row.get(col_idx)?;
                        bytes_to_f64(val)
                    })
                    .collect()
            })
            .collect();

        // Ensure all columns have the same number of values
        let min_len = numeric_values.iter().map(|v| v.len()).min().unwrap_or(0);
        if min_len > 0 {
            let trimmed: Vec<Vec<f64>> = numeric_values
                .into_iter()
                .map(|v| v[..min_len].to_vec())
                .collect();
            vec![CorrelationStats::compute(numeric_names, &trimmed)]
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    TableStatistics {
        row_count: total_rows,
        columns: column_stats,
        multi_column,
        last_analyzed: Some(current_timestamp),
    }
}

/// Spawn the ANALYZE command as a background tokio task that does not block
/// user queries.
///
/// Returns a `JoinHandle` that resolves when analysis is complete.
pub fn spawn_analyze(
    blocks: Vec<PaxBlock>,
    column_names: Vec<String>,
    table_name: String,
    catalog: StatisticsCatalog,
    config: AnalyzeConfig,
    current_timestamp: Timestamp,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let stats = analyze_blocks(&blocks, &column_names, &config, current_timestamp);
        catalog.store(table_name, stats).await;
    })
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Check if a column type is numeric (suitable for correlation computation).
fn is_numeric_type(col_type: &galaxdb_common::ColumnType) -> bool {
    matches!(
        col_type,
        galaxdb_common::ColumnType::Int8
            | galaxdb_common::ColumnType::Int16
            | galaxdb_common::ColumnType::Int32
            | galaxdb_common::ColumnType::Int64
            | galaxdb_common::ColumnType::UInt8
            | galaxdb_common::ColumnType::UInt16
            | galaxdb_common::ColumnType::UInt32
            | galaxdb_common::ColumnType::UInt64
            | galaxdb_common::ColumnType::Float32
            | galaxdb_common::ColumnType::Float64
    )
}

/// Convert raw bytes to f64 for correlation computation.
/// Supports all numeric column types.
fn bytes_to_f64(bytes: &[u8]) -> Option<f64> {
    match bytes.len() {
        1 => Some(bytes[0] as f64),
        2 => {
            let arr: [u8; 2] = bytes.try_into().ok()?;
            // Try as i16 first (most common 2-byte type)
            Some(i16::from_le_bytes(arr) as f64)
        }
        4 => {
            let arr: [u8; 4] = bytes.try_into().ok()?;
            // Try as f32 first, fall back to i32
            let f = f32::from_le_bytes(arr);
            if f.is_finite() {
                Some(f as f64)
            } else {
                Some(i32::from_le_bytes(arr) as f64)
            }
        }
        8 => {
            let arr: [u8; 8] = bytes.try_into().ok()?;
            let f = f64::from_le_bytes(arr);
            if f.is_finite() {
                Some(f)
            } else {
                Some(i64::from_le_bytes(arr) as f64)
            }
        }
        _ => None,
    }
}
