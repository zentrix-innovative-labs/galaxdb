//! Distance computation for vector similarity search.
//!
//! Supports cosine similarity (primary metric for embedding search) and
//! Euclidean distance. Uses SIMD acceleration on x86-64 (AVX2) when available,
//! with a portable scalar fallback.
//!
//! Cosine similarity = dot(a, b) / (||a|| × ||b||)
//! For normalized vectors: cosine_similarity = dot(a, b)
//!
//! We store cosine *distance* = 1.0 - cosine_similarity so that smaller = closer,
//! which is consistent with the HNSW algorithm's min-heap ordering.

/// Compute cosine distance between two f32 vectors.
/// Returns 1.0 - cosine_similarity. Range: [0.0, 2.0].
/// 0.0 = identical direction, 1.0 = orthogonal, 2.0 = opposite.
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must have same dimension");
    let sim = cosine_similarity(a, b);
    1.0 - sim
}

/// Fast cosine distance for PRE-NORMALIZED vectors (||a|| = ||b|| = 1).
/// Returns 1.0 - dot(a, b). This is 3× faster than full cosine_distance
/// because it skips the norm computations.
///
/// IMPORTANT: Only correct if both vectors are unit-length. Use `normalize()`
/// before inserting into the HNSW graph.
#[inline]
pub fn cosine_distance_normalized(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    1.0 - dot_product_fast(a, b)
}

/// Fast dot product with SIMD acceleration.
#[inline]
pub fn dot_product_fast(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
            return unsafe { dot_product_avx2(a, b) };
        }
    }

    // Scalar fallback
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// AVX2 + FMA accelerated dot product.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        use std::arch::x86_64::*;

        let n = a.len();
        let chunks = n / 8;
        let remainder = n % 8;

        let mut acc = _mm256_setzero_ps();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a_ptr.add(offset));
            let vb = _mm256_loadu_ps(b_ptr.add(offset));
            acc = _mm256_fmadd_ps(va, vb, acc);
        }

        let mut sum = hsum_avx2(acc);
        let start = chunks * 8;
        for i in start..start + remainder {
            sum += a[i] * b[i];
        }
        sum
    }
}

/// Compute cosine similarity between two f32 vectors.
/// Returns dot(a,b) / (||a|| × ||b||). Range: [-1.0, 1.0].
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must have same dimension");

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
            // Safety: we checked for AVX2+FMA support
            return unsafe { cosine_similarity_avx2(a, b) };
        }
    }

    cosine_similarity_scalar(a, b)
}

/// Scalar (portable) cosine similarity.
#[inline]
fn cosine_similarity_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom < f32::EPSILON {
        return 0.0;
    }
    dot / denom
}

/// AVX2 + FMA accelerated cosine similarity.
///
/// Processes 8 floats per iteration using 256-bit SIMD registers.
/// Uses fused multiply-add (FMA) for dot product and norm accumulation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn cosine_similarity_avx2(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut dot_acc = _mm256_setzero_ps();
    let mut norm_a_acc = _mm256_setzero_ps();
    let mut norm_b_acc = _mm256_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a_ptr.add(offset));
        let vb = _mm256_loadu_ps(b_ptr.add(offset));

        // dot += a[i] * b[i]  (fused multiply-add)
        dot_acc = _mm256_fmadd_ps(va, vb, dot_acc);
        // norm_a += a[i] * a[i]
        norm_a_acc = _mm256_fmadd_ps(va, va, norm_a_acc);
        // norm_b += b[i] * b[i]
        norm_b_acc = _mm256_fmadd_ps(vb, vb, norm_b_acc);
    }

    // Horizontal sum of 8-wide accumulators
    let dot = hsum_avx2(dot_acc);
    let norm_a = hsum_avx2(norm_a_acc);
    let norm_b = hsum_avx2(norm_b_acc);

    // Handle remainder elements with scalar
    let mut dot_r = dot;
    let mut norm_a_r = norm_a;
    let mut norm_b_r = norm_b;
    let start = chunks * 8;
    for i in start..start + remainder {
        dot_r += a[i] * b[i];
        norm_a_r += a[i] * a[i];
        norm_b_r += b[i] * b[i];
    }

    let denom = (norm_a_r * norm_b_r).sqrt();
    if denom < f32::EPSILON {
        return 0.0;
    }
    dot_r / denom
    }
}

/// Horizontal sum of 8 f32 values in a __m256 register.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_avx2(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;

    // v = [a0, a1, a2, a3, a4, a5, a6, a7]
    let hi128 = _mm256_extractf128_ps(v, 1); // [a4, a5, a6, a7]
    let lo128 = _mm256_castps256_ps128(v);    // [a0, a1, a2, a3]
    let sum128 = _mm_add_ps(lo128, hi128);    // [a0+a4, a1+a5, a2+a6, a3+a7]

    let hi64 = _mm_movehl_ps(sum128, sum128); // [a2+a6, a3+a7, ...]
    let sum64 = _mm_add_ps(sum128, hi64);     // [a0+a2+a4+a6, a1+a3+a5+a7, ...]

    let hi32 = _mm_shuffle_ps(sum64, sum64, 0x01); // [a1+a3+a5+a7, ...]
    let sum32 = _mm_add_ss(sum64, hi32);

    _mm_cvtss_f32(sum32)
}

/// Compute squared Euclidean distance (L2²) between two f32 vectors.
/// Useful for distance comparisons without the sqrt.
#[inline]
pub fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Normalize a vector to unit length (L2 norm = 1.0).
/// For normalized vectors, cosine_similarity = dot product.
#[inline]
pub fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Compute dot product of two f32 vectors.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6, "identical vectors should have similarity 1.0, got {}", sim);
        assert!(cosine_distance(&a, &a) < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6, "orthogonal vectors should have similarity 0.0, got {}", sim);
        assert!((cosine_distance(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_opposite_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-6, "opposite vectors should have similarity -1.0, got {}", sim);
        assert!((cosine_distance(&a, &b) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_known_value() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        // dot = 4+10+18 = 32
        // ||a|| = sqrt(14), ||b|| = sqrt(77)
        // sim = 32 / sqrt(14*77) = 32 / sqrt(1078) ≈ 0.9746
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 0.9746).abs() < 0.001, "got {}", sim);
    }

    #[test]
    fn cosine_high_dimensional() {
        // 768-dimensional vectors (typical embedding size)
        let mut a = vec![0.0f32; 768];
        let mut b = vec![0.0f32; 768];
        for i in 0..768 {
            a[i] = (i as f32 * 0.01).sin();
            b[i] = (i as f32 * 0.01 + 0.1).sin();
        }
        let sim = cosine_similarity(&a, &b);
        // Similar vectors should have high similarity
        assert!(sim > 0.9, "similar 768-dim vectors should have high similarity, got {}", sim);
    }

    #[test]
    fn cosine_scalar_matches_simd() {
        let mut a = vec![0.0f32; 100];
        let mut b = vec![0.0f32; 100];
        let mut rng = 42u64;
        for i in 0..100 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            a[i] = (rng as f32) / (u64::MAX as f32) * 2.0 - 1.0;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            b[i] = (rng as f32) / (u64::MAX as f32) * 2.0 - 1.0;
        }

        let scalar = cosine_similarity_scalar(&a, &b);
        let auto = cosine_similarity(&a, &b);
        assert!(
            (scalar - auto).abs() < 1e-5,
            "scalar ({}) and auto ({}) should match",
            scalar, auto
        );
    }

    #[test]
    fn normalize_produces_unit_vector() {
        let mut v = vec![3.0, 4.0];
        normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn l2_distance_squared_known() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        // (3² + 3² + 3²) = 27
        assert!((l2_distance_squared(&a, &b) - 27.0).abs() < 1e-6);
    }

    #[test]
    fn zero_vector_cosine() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0, "zero vector should have 0 similarity");
    }
}
