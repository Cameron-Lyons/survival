//! Vectorisation-friendly reductions over `f64` slices.
//!
//! Each reduction keeps four independent accumulators so the compiler can
//! emit packed instructions and the summation order is deterministic. This
//! is the single implementation; `crate::simd_ops` re-exports the public
//! subset under its historical `*_simd` names.

const LANES: usize = 4;

#[inline]
fn combine(accumulators: [f64; LANES]) -> f64 {
    (accumulators[0] + accumulators[1]) + (accumulators[2] + accumulators[3])
}

/// Sum of `data`; `0.0` for an empty slice.
pub fn sum_f64(data: &[f64]) -> f64 {
    let mut accumulators = [0.0; LANES];
    let chunks = data.chunks_exact(LANES);
    let remainder = chunks.remainder();
    for chunk in chunks {
        for (accumulator, &value) in accumulators.iter_mut().zip(chunk) {
            *accumulator += value;
        }
    }
    let mut total = combine(accumulators);
    for &value in remainder {
        total += value;
    }
    total
}

/// Dot product over the common prefix of `a` and `b`; `0.0` when either is
/// empty.
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let mut accumulators = [0.0; LANES];
    let a_chunks = a[..n].chunks_exact(LANES);
    let b_chunks = b[..n].chunks_exact(LANES);
    let a_rest = a_chunks.remainder();
    let b_rest = b_chunks.remainder();
    for (a_chunk, b_chunk) in a_chunks.zip(b_chunks) {
        for lane in 0..LANES {
            accumulators[lane] = a_chunk[lane].mul_add(b_chunk[lane], accumulators[lane]);
        }
    }
    let mut total = combine(accumulators);
    for (&left, &right) in a_rest.iter().zip(b_rest) {
        total = left.mul_add(right, total);
    }
    total
}

pub fn sum_of_squares(values: &[f64]) -> f64 {
    dot_product(values, values)
}

/// Arithmetic mean; `0.0` for an empty slice.
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    sum_f64(values) / values.len() as f64
}

pub fn subtract_scalar(values: &[f64], scalar: f64) -> Vec<f64> {
    values.iter().map(|&value| value - scalar).collect()
}

/// Unbiased (`n - 1`) sample variance; `0.0` when fewer than two values.
pub fn variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let centered = subtract_scalar(values, mean(values));
    sum_of_squares(&centered) / (values.len() - 1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert!((sum_f64(&data) - 55.0).abs() < 1e-10);
        assert_eq!(sum_f64(&[]), 0.0);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        assert!((dot_product(&a, &b) - 70.0).abs() < 1e-10);
        assert_eq!(dot_product(&[], &[]), 0.0);
        // Common-prefix semantics.
        assert!((dot_product(&a[..2], &b) - 8.0).abs() < 1e-10);
        assert!((sum_of_squares(&a) - 55.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean_and_variance() {
        let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((mean(&values) - 5.0).abs() < 1e-12);
        assert!((variance(&values) - 32.0 / 7.0).abs() < 1e-12);
        assert_eq!(mean(&[]), 0.0);
        assert_eq!(variance(&[1.0]), 0.0);
        assert_eq!(subtract_scalar(&[5.0, 10.0], 5.0), vec![0.0, 5.0]);
    }

    #[test]
    fn test_large_array_matches_naive_sum() {
        let n = 10_000;
        let a: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..n).map(|i| (i * 2) as f64).collect();
        let expected: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((dot_product(&a, &b) - expected).abs() < 1e-6);
        let expected: f64 = a.iter().sum();
        assert!((sum_f64(&a) - expected).abs() < 1e-6);
    }
}
