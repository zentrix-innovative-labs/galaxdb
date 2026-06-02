//! Platform-aware vector quantization for memory-efficient HNSW storage.
//!
//! Three quantization methods, selected based on platform and user config:
//!
//! - **SQ8** (Scalar Quantization, int8): 4× compression. Maps each float32
//!   dimension to uint8 using per-dimension min/max calibration. Distance
//!   computed on quantized values with correction factor. Default on x86-64
//!   with AVX2.
//!
//! - **FP16** (Half-precision float): 2× compression. Uses IEEE 754 half-
//!   precision. Nearly lossless for normalized embeddings. Default on ARM64
//!   with NEON.
//!
//! - **RaBitQ** (Random Binary Quantization): 32× compression. Applies random
//!   orthogonal rotation then takes sign bits. Uses Hamming distance. Opt-in
//!   only. Based on Gao et al. 2024.
//!
//! References:
//! - SQ8: Milvus IVF_SQ8, Elasticsearch scalar quantization, Qdrant
//! - FP16: IEEE 754-2008, ARM NEON native support
//! - RaBitQ: "Quantizing High-Dimensional Vectors with a Theoretical Error
//!   Bound for Approximate Nearest Neighbor Search" (arXiv:2405.12497)



/// Quantizer trait — all quantization methods implement this.
pub trait Quantizer: Send + Sync {
    /// Quantize a float32 vector to compressed bytes.
    fn quantize(&self, vector: &[f32]) -> Vec<u8>;

    /// Dequantize compressed bytes back to float32 vector.
    fn dequantize(&self, quantized: &[u8]) -> Vec<f32>;

    /// Compute approximate distance between two quantized vectors.
    /// Returns cosine distance estimate (smaller = more similar).
    fn distance(&self, a: &[u8], b: &[u8]) -> f32;

    /// Compression ratio (e.g., 4.0 for SQ8, 2.0 for FP16, 32.0 for RaBitQ).
    fn compression_ratio(&self) -> f32;

    /// Name of this quantizer.
    fn name(&self) -> &str;

    /// Dimensionality this quantizer was configured for.
    fn dim(&self) -> usize;
}

// ---------------------------------------------------------------------------
// SQ8: Scalar Quantization to uint8
// ---------------------------------------------------------------------------

/// Per-dimension min/max calibration data for SQ8.
#[derive(Debug, Clone)]
pub struct Sq8Calibration {
    /// Minimum value per dimension.
    pub mins: Vec<f32>,
    /// Range (max - min) per dimension.
    pub ranges: Vec<f32>,
}

/// SQ8 scalar quantizer: maps each float32 dimension to uint8 [0, 255].
///
/// Calibration: compute min/max per dimension from a representative dataset.
/// Quantize: `q[i] = round((v[i] - min[i]) / range[i] * 255)`
/// Dequantize: `v[i] = q[i] / 255.0 * range[i] + min[i]`
///
/// Distance is computed on quantized values using L2 on uint8, then
/// scaled back to approximate the original float32 distance.
pub struct Sq8Quantizer {
    dim: usize,
    calibration: Sq8Calibration,
}

impl Sq8Quantizer {
    /// Create an SQ8 quantizer from calibration data.
    pub fn new(calibration: Sq8Calibration) -> Self {
        let dim = calibration.mins.len();
        assert_eq!(dim, calibration.ranges.len());
        Self { dim, calibration }
    }

    /// Calibrate from a set of vectors by computing per-dimension min/max.
    ///
    /// This should be called with a representative sample of vectors
    /// (e.g., 10K-100K vectors from the dataset).
    pub fn calibrate(vectors: &[&[f32]], dim: usize) -> Self {
        assert!(!vectors.is_empty(), "need at least one vector for calibration");
        assert!(vectors[0].len() == dim);

        let mut mins = vec![f32::MAX; dim];
        let mut maxs = vec![f32::MIN; dim];

        for v in vectors {
            for (i, &val) in v.iter().enumerate() {
                if val < mins[i] { mins[i] = val; }
                if val > maxs[i] { maxs[i] = val; }
            }
        }

        let ranges: Vec<f32> = mins.iter().zip(maxs.iter())
            .map(|(&min, &max)| {
                let r = max - min;
                if r < f32::EPSILON { 1.0 } else { r } // avoid division by zero
            })
            .collect();

        Self {
            dim,
            calibration: Sq8Calibration { mins, ranges },
        }
    }

    /// Get the calibration data (for serialization).
    pub fn calibration(&self) -> &Sq8Calibration {
        &self.calibration
    }
}

impl Quantizer for Sq8Quantizer {
    fn quantize(&self, vector: &[f32]) -> Vec<u8> {
        assert_eq!(vector.len(), self.dim);
        vector.iter().enumerate().map(|(i, &val)| {
            let normalized = (val - self.calibration.mins[i]) / self.calibration.ranges[i];
            let clamped = normalized.clamp(0.0, 1.0);
            (clamped * 255.0).round() as u8
        }).collect()
    }

    fn dequantize(&self, quantized: &[u8]) -> Vec<f32> {
        assert_eq!(quantized.len(), self.dim);
        quantized.iter().enumerate().map(|(i, &q)| {
            (q as f32 / 255.0) * self.calibration.ranges[i] + self.calibration.mins[i]
        }).collect()
    }

    fn distance(&self, a: &[u8], b: &[u8]) -> f32 {
        assert_eq!(a.len(), self.dim);
        assert_eq!(b.len(), self.dim);

        // Compute L2 distance on uint8 values, then scale to approximate
        // the original float32 cosine distance.
        // For normalized vectors, L2² ≈ 2 * (1 - cosine_similarity)
        let mut sum_sq: u64 = 0;
        for i in 0..self.dim {
            let diff = a[i] as i32 - b[i] as i32;
            sum_sq += (diff * diff) as u64;
        }

        // Scale back: each dimension was mapped to [0, 255] from [min, min+range]
        // The average range² factor converts uint8 L2² to float32 L2²
        let avg_range_sq: f32 = self.calibration.ranges.iter()
            .map(|r| r * r)
            .sum::<f32>() / self.dim as f32;
        let scale = avg_range_sq / (255.0 * 255.0);

        let l2_sq = sum_sq as f32 * scale;
        // Convert L2² to approximate cosine distance: cos_dist ≈ L2²/2 for normalized vectors
        l2_sq / 2.0
    }

    fn compression_ratio(&self) -> f32 { 4.0 }
    fn name(&self) -> &str { "SQ8" }
    fn dim(&self) -> usize { self.dim }
}

// ---------------------------------------------------------------------------
// FP16: Half-precision float quantization
// ---------------------------------------------------------------------------

/// FP16 quantizer: maps float32 to IEEE 754 half-precision (16-bit).
///
/// 2× compression with nearly lossless precision for values in [-1, 1].
/// Uses the `half` crate for conversion.
pub struct Fp16Quantizer {
    dim: usize,
}

impl Fp16Quantizer {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Quantizer for Fp16Quantizer {
    fn quantize(&self, vector: &[f32]) -> Vec<u8> {
        assert_eq!(vector.len(), self.dim);
        let mut bytes = Vec::with_capacity(self.dim * 2);
        for &val in vector {
            let h = half::f16::from_f32(val);
            bytes.extend_from_slice(&h.to_le_bytes());
        }
        bytes
    }

    fn dequantize(&self, quantized: &[u8]) -> Vec<f32> {
        assert_eq!(quantized.len(), self.dim * 2);
        (0..self.dim).map(|i| {
            let offset = i * 2;
            let h = half::f16::from_le_bytes([quantized[offset], quantized[offset + 1]]);
            h.to_f32()
        }).collect()
    }

    fn distance(&self, a: &[u8], b: &[u8]) -> f32 {
        // Dequantize both and compute exact cosine distance
        let va = self.dequantize(a);
        let vb = self.dequantize(b);
        crate::distance::cosine_distance(&va, &vb)
    }

    fn compression_ratio(&self) -> f32 { 2.0 }
    fn name(&self) -> &str { "FP16" }
    fn dim(&self) -> usize { self.dim }
}

// ---------------------------------------------------------------------------
// RaBitQ: Random Binary Quantization (1-bit per dimension)
// ---------------------------------------------------------------------------

/// RaBitQ quantizer: random rotation + binary quantization.
///
/// 32× compression. Each float32 dimension is reduced to 1 bit.
///
/// Process:
/// 1. Normalize the input vector to unit length
/// 2. Apply a random orthogonal rotation matrix (fixed at construction)
/// 3. Take the sign bit of each rotated dimension
/// 4. Pack bits into bytes
///
/// Distance is approximated using Hamming distance on the binary codes.
///
/// Based on: "Quantizing High-Dimensional Vectors with a Theoretical Error
/// Bound for Approximate Nearest Neighbor Search" (Gao et al., 2024)
pub struct RabitqQuantizer {
    dim: usize,
    /// Random rotation matrix (dim × dim), stored row-major.
    /// Generated from a random seed for reproducibility.
    rotation: Vec<f32>,
}

impl RabitqQuantizer {
    /// Create a RaBitQ quantizer with a random rotation matrix.
    ///
    /// The rotation matrix is generated deterministically from the seed
    /// for reproducibility across restarts.
    pub fn new(dim: usize, seed: u64) -> Self {
        let rotation = generate_random_rotation(dim, seed);
        Self { dim, rotation }
    }

    /// Apply the rotation matrix to a vector.
    //
    // Index-based loops are intentional here: this is a dense
    // matrix-vector multiply where both indices participate in the
    // flattened `row_offset + j` / `i * dim` address arithmetic.
    // Rewriting as iterator chains obscures the linear algebra without
    // changing the generated code.
    #[allow(clippy::needless_range_loop)]
    fn rotate(&self, vector: &[f32]) -> Vec<f32> {
        let mut rotated = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            let mut sum = 0.0f32;
            let row_offset = i * self.dim;
            for j in 0..self.dim {
                sum += self.rotation[row_offset + j] * vector[j];
            }
            rotated[i] = sum;
        }
        rotated
    }
}

impl Quantizer for RabitqQuantizer {
    fn quantize(&self, vector: &[f32]) -> Vec<u8> {
        assert_eq!(vector.len(), self.dim);

        // Normalize to unit length
        let mut normalized = vector.to_vec();
        crate::distance::normalize(&mut normalized);

        // Apply random rotation
        let rotated = self.rotate(&normalized);

        // Binary quantization: take sign bit, pack into bytes
        let num_bytes = self.dim.div_ceil(8);
        let mut bytes = vec![0u8; num_bytes];
        for (i, &val) in rotated.iter().enumerate() {
            if val > 0.0 {
                bytes[i / 8] |= 1 << (i % 8);
            }
        }
        bytes
    }

    #[allow(clippy::needless_range_loop)]
    fn dequantize(&self, quantized: &[u8]) -> Vec<f32> {
        // Binary quantization is lossy — dequantize produces ±1/√dim values
        // which approximate the original direction after inverse rotation.
        //
        // Index-based loops are intentional: the first unpacks bit `i`
        // from byte `i/8`; the second is the transposed matrix-vector
        // multiply (`rotation[j * dim + i]`). Both rely on the index in
        // address arithmetic.
        let scale = 1.0 / (self.dim as f32).sqrt();
        let mut binary = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            let bit = (quantized[i / 8] >> (i % 8)) & 1;
            binary[i] = if bit == 1 { scale } else { -scale };
        }

        // Apply inverse rotation (transpose for orthogonal matrix)
        let mut result = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            let mut sum = 0.0f32;
            for j in 0..self.dim {
                // Transpose: rotation[j][i] = rotation[j * dim + i]
                sum += self.rotation[j * self.dim + i] * binary[j];
            }
            result[i] = sum;
        }
        result
    }

    fn distance(&self, a: &[u8], b: &[u8]) -> f32 {
        // Hamming distance: count differing bits
        let hamming: u32 = a.iter().zip(b.iter())
            .map(|(&x, &y)| (x ^ y).count_ones())
            .sum();

        // Convert Hamming distance to approximate cosine distance.
        // For random binary codes of dimension D:
        // E[hamming] = D/2 * (1 - cos_sim)
        // So: cos_dist = 1 - cos_sim ≈ 2 * hamming / D
        2.0 * hamming as f32 / self.dim as f32
    }

    fn compression_ratio(&self) -> f32 { 32.0 }
    fn name(&self) -> &str { "RaBitQ" }
    fn dim(&self) -> usize { self.dim }
}

/// Generate a pseudo-random orthogonal rotation matrix using Gram-Schmidt.
///
/// This is a simplified approach. For production, a proper random orthogonal
/// matrix from the Haar measure (e.g., via QR decomposition of a random
/// Gaussian matrix) would be more rigorous. This implementation uses
/// Gram-Schmidt orthogonalization on random vectors for correctness.
fn generate_random_rotation(dim: usize, seed: u64) -> Vec<f32> {
    // Simple LCG PRNG for deterministic generation
    let mut state = seed;
    let mut next_f32 = || -> f32 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // Map to [-1, 1]
        (state >> 33) as f32 / (u32::MAX as f32 / 2.0) - 1.0
    };

    // Generate random vectors and orthogonalize via Gram-Schmidt
    let mut matrix = vec![0.0f32; dim * dim];

    for i in 0..dim {
        // Generate random vector
        let row_start = i * dim;
        for j in 0..dim {
            matrix[row_start + j] = next_f32();
        }

        // Orthogonalize against all previous rows
        for k in 0..i {
            let prev_start = k * dim;
            // dot product with previous row
            let mut dot = 0.0f32;
            for j in 0..dim {
                dot += matrix[row_start + j] * matrix[prev_start + j];
            }
            // subtract projection
            for j in 0..dim {
                matrix[row_start + j] -= dot * matrix[prev_start + j];
            }
        }

        // Normalize
        let mut norm = 0.0f32;
        for j in 0..dim {
            norm += matrix[row_start + j] * matrix[row_start + j];
        }
        let norm = norm.sqrt();
        if norm > f32::EPSILON {
            for j in 0..dim {
                matrix[row_start + j] /= norm;
            }
        }
    }

    matrix
}

// ---------------------------------------------------------------------------
// Platform detection and default quantizer selection
// ---------------------------------------------------------------------------

/// Select the default quantizer based on the current platform.
///
/// - x86-64 with AVX2: SQ8 (4× compression, SIMD-friendly uint8 ops)
/// - ARM64: FP16 (2× compression, NEON native FP16 support)
/// - Other: SQ8 (fallback)
///
/// The caller must provide calibration data for SQ8 or a seed for RaBitQ.
pub fn select_default_quantizer(dim: usize, calibration_vectors: &[&[f32]]) -> Box<dyn Quantizer> {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return Box::new(Sq8Quantizer::calibrate(calibration_vectors, dim));
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return Box::new(Fp16Quantizer::new(dim));
    }

    // Fallback
    Box::new(Sq8Quantizer::calibrate(calibration_vectors, dim))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::{cosine_distance, cosine_similarity};

    // --- SQ8 tests ---

    #[test]
    fn sq8_quantize_dequantize_roundtrip() {
        let vectors: Vec<Vec<f32>> = vec![
            vec![0.1, 0.5, -0.3, 0.8],
            vec![-0.2, 0.9, 0.1, -0.7],
            vec![0.4, -0.1, 0.6, 0.3],
        ];
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let q = Sq8Quantizer::calibrate(&refs, 4);

        for v in &vectors {
            let quantized = q.quantize(v);
            assert_eq!(quantized.len(), 4); // 4 dimensions → 4 bytes
            let dequantized = q.dequantize(&quantized);
            assert_eq!(dequantized.len(), 4);

            // Check roundtrip accuracy (should be within ~1/255 of range per dim)
            for (i, (&orig, &deq)) in v.iter().zip(dequantized.iter()).enumerate() {
                let max_error = q.calibration.ranges[i] / 255.0 + 0.01;
                assert!(
                    (orig - deq).abs() < max_error,
                    "dim {}: orig={}, deq={}, error={}, max_error={}",
                    i, orig, deq, (orig - deq).abs(), max_error
                );
            }
        }
    }

    #[test]
    fn sq8_compression_ratio() {
        let vectors = [vec![0.0f32; 768]];
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let q = Sq8Quantizer::calibrate(&refs, 768);
        assert_eq!(q.compression_ratio(), 4.0);
        assert_eq!(q.name(), "SQ8");

        let quantized = q.quantize(&vectors[0]);
        assert_eq!(quantized.len(), 768); // 768 bytes vs 768*4=3072 bytes original
    }

    #[test]
    fn sq8_distance_preserves_ordering() {
        // Three vectors: a is close to b, far from c
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.9, 0.1, 0.0, 0.0];
        let c = vec![0.0, 0.0, 0.0, 1.0];

        let vectors = [a.clone(), b.clone(), c.clone()];
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let q = Sq8Quantizer::calibrate(&refs, 4);

        let qa = q.quantize(&a);
        let qb = q.quantize(&b);
        let qc = q.quantize(&c);

        let dist_ab = q.distance(&qa, &qb);
        let dist_ac = q.distance(&qa, &qc);

        assert!(
            dist_ab < dist_ac,
            "SQ8 distance should preserve ordering: d(a,b)={} < d(a,c)={}",
            dist_ab, dist_ac
        );
    }

    #[test]
    fn sq8_high_dimensional_recall() {
        // Test that SQ8 preserves similarity ordering on 128-dim vectors
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};

        let dim = 128;
        let n = 200;
        let mut rng = SmallRng::seed_from_u64(42);

        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let q = Sq8Quantizer::calibrate(&refs, dim);

        // Pick a query and find top-10 by exact cosine distance
        let query = &vectors[0];
        let mut exact_dists: Vec<(usize, f32)> = vectors.iter().enumerate()
            .map(|(i, v)| (i, cosine_distance(query, v)))
            .collect();
        exact_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let exact_top10: std::collections::HashSet<usize> = exact_dists.iter().take(10).map(|d| d.0).collect();

        // Find top-10 by SQ8 quantized distance
        let qq = q.quantize(query);
        let mut sq8_dists: Vec<(usize, f32)> = vectors.iter().enumerate()
            .map(|(i, v)| (i, q.distance(&qq, &q.quantize(v))))
            .collect();
        sq8_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let sq8_top10: std::collections::HashSet<usize> = sq8_dists.iter().take(10).map(|d| d.0).collect();

        let recall = exact_top10.intersection(&sq8_top10).count() as f64 / 10.0;
        assert!(
            recall >= 0.7,
            "SQ8 recall@10 should be >= 0.7, got {:.2}",
            recall
        );
    }

    // --- FP16 tests ---

    #[test]
    fn fp16_quantize_dequantize_roundtrip() {
        let q = Fp16Quantizer::new(4);
        let v = vec![0.1, 0.5, -0.3, 0.8];
        let quantized = q.quantize(&v);
        assert_eq!(quantized.len(), 8); // 4 dims × 2 bytes
        let dequantized = q.dequantize(&quantized);

        for (i, (&orig, &deq)) in v.iter().zip(dequantized.iter()).enumerate() {
            assert!(
                (orig - deq).abs() < 0.001,
                "FP16 dim {}: orig={}, deq={}", i, orig, deq
            );
        }
    }

    #[test]
    fn fp16_compression_ratio() {
        let q = Fp16Quantizer::new(768);
        assert_eq!(q.compression_ratio(), 2.0);
        assert_eq!(q.name(), "FP16");
    }

    #[test]
    fn fp16_distance_matches_exact() {
        let q = Fp16Quantizer::new(4);
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0];

        let qa = q.quantize(&a);
        let qb = q.quantize(&b);
        let dist = q.distance(&qa, &qb);
        let exact = cosine_distance(&a, &b);

        assert!(
            (dist - exact).abs() < 0.01,
            "FP16 distance should match exact: {} vs {}", dist, exact
        );
    }

    // --- RaBitQ tests ---

    #[test]
    fn rabitq_quantize_dequantize_direction() {
        let q = RabitqQuantizer::new(32, 42);
        let v: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1).sin()).collect();

        let quantized = q.quantize(&v);
        assert_eq!(quantized.len(), 4); // 32 bits = 4 bytes (32× compression)

        let dequantized = q.dequantize(&quantized);
        assert_eq!(dequantized.len(), 32);

        // RaBitQ is very lossy — check that the general direction is preserved
        let sim = cosine_similarity(&v, &dequantized);
        assert!(
            sim > 0.3,
            "RaBitQ should preserve general direction, got similarity {}",
            sim
        );
    }

    #[test]
    fn rabitq_compression_ratio() {
        let q = RabitqQuantizer::new(768, 42);
        assert_eq!(q.compression_ratio(), 32.0);
        assert_eq!(q.name(), "RaBitQ");

        let v = vec![0.1f32; 768];
        let quantized = q.quantize(&v);
        assert_eq!(quantized.len(), 96); // 768 bits = 96 bytes
    }

    #[test]
    fn rabitq_distance_preserves_ordering() {
        let q = RabitqQuantizer::new(64, 42);

        // a is close to b (similar direction), far from c (orthogonal)
        let mut a = vec![0.0f32; 64];
        let mut b = vec![0.0f32; 64];
        let mut c = vec![0.0f32; 64];
        for i in 0..32 { a[i] = 1.0; b[i] = 0.9; }
        for slot in c.iter_mut().take(64).skip(32) { *slot = 1.0; }

        let qa = q.quantize(&a);
        let qb = q.quantize(&b);
        let qc = q.quantize(&c);

        let dist_ab = q.distance(&qa, &qb);
        let dist_ac = q.distance(&qa, &qc);

        assert!(
            dist_ab < dist_ac,
            "RaBitQ distance should preserve ordering: d(a,b)={} < d(a,c)={}",
            dist_ab, dist_ac
        );
    }

    #[test]
    fn rabitq_hamming_distance_correct() {
        let q = RabitqQuantizer::new(8, 42);
        // Two identical vectors should have 0 Hamming distance
        let v = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let qa = q.quantize(&v);
        let qb = q.quantize(&v);
        let dist = q.distance(&qa, &qb);
        assert_eq!(dist, 0.0, "identical vectors should have 0 distance");
    }

    // --- Platform detection test ---

    #[test]
    fn platform_detection_returns_quantizer() {
        let vectors = vec![vec![0.0f32; 128]; 10];
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let q = select_default_quantizer(128, &refs);

        // Should return a valid quantizer regardless of platform
        assert_eq!(q.dim(), 128);
        assert!(q.compression_ratio() >= 2.0);

        let v = vec![0.5f32; 128];
        let quantized = q.quantize(&v);
        let dequantized = q.dequantize(&quantized);
        assert_eq!(dequantized.len(), 128);
    }
}
