use pulp::{Arch, Simd, WithSimd};

pub fn dot_product_simd(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());

    struct DotProduct<'a>(&'a [f64], &'a [f64]);

    impl WithSimd for DotProduct<'_> {
        type Output = f64;

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let (a_head, a_tail) = S::as_simd_f64s(self.0);
            let (b_head, b_tail) = S::as_simd_f64s(self.1);

            let mut acc = simd.splat_f64s(0.0);
            for (&a_chunk, &b_chunk) in a_head.iter().zip(b_head.iter()) {
                acc = simd.mul_add_f64s(a_chunk, b_chunk, acc);
            }

            simd.reduce_sum_f64s(acc)
                + a_tail
                    .iter()
                    .zip(b_tail.iter())
                    .map(|(&a_val, &b_val)| a_val * b_val)
                    .sum::<f64>()
        }
    }

    Arch::new().dispatch(DotProduct(&a[..n], &b[..n]))
}

pub fn weighted_sum_simd(values: &[f64], weights: &[f64]) -> f64 {
    dot_product_simd(values, weights)
}

pub fn sum_simd(values: &[f64]) -> f64 {
    crate::internal::simd::sum_f64(values)
}

pub fn sum_of_squares_simd(values: &[f64]) -> f64 {
    dot_product_simd(values, values)
}

pub fn mean_simd(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    sum_simd(values) / values.len() as f64
}

pub fn subtract_scalar_simd(values: &[f64], scalar: f64) -> Vec<f64> {
    values.iter().map(|&value| value - scalar).collect()
}

pub fn multiply_elementwise_simd(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len().min(b.len());
    a[..n]
        .iter()
        .zip(b[..n].iter())
        .map(|(&a_val, &b_val)| a_val * b_val)
        .collect()
}

pub fn exp_approx_simd(values: &[f64]) -> Vec<f64> {
    values.iter().map(|&value| value.exp()).collect()
}

pub fn logistic_simd(values: &[f64]) -> Vec<f64> {
    values.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
}

pub fn variance_simd(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let mean = mean_simd(values);
    let centered = subtract_scalar_simd(values, mean);
    sum_of_squares_simd(&centered) / (values.len() - 1) as f64
}

pub fn covariance_simd(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }

    let mean_x = mean_simd(&x[..n]);
    let mean_y = mean_simd(&y[..n]);

    let centered_x = subtract_scalar_simd(&x[..n], mean_x);
    let centered_y = subtract_scalar_simd(&y[..n], mean_y);

    dot_product_simd(&centered_x, &centered_y) / (n - 1) as f64
}

pub struct PairwiseCounts {
    pub concordant: usize,
    pub discordant: usize,
    pub tied: usize,
    pub valid_pairs: usize,
}

pub fn count_concordant_pairs_simd(
    risk_i: f64,
    time_i: f64,
    risk_scores: &[f64],
    times: &[f64],
    skip_idx: usize,
) -> PairwiseCounts {
    let n = risk_scores.len().min(times.len());
    let mut concordant = 0usize;
    let mut discordant = 0usize;
    let mut tied = 0usize;
    let mut valid_pairs = 0usize;

    for j in 0..n {
        if j == skip_idx || times[j] <= time_i {
            continue;
        }
        valid_pairs += 1;
        if risk_i > risk_scores[j] {
            concordant += 1;
        } else if risk_i < risk_scores[j] {
            discordant += 1;
        } else {
            tied += 1;
        }
    }

    PairwiseCounts {
        concordant,
        discordant,
        tied,
        valid_pairs,
    }
}

pub fn weighted_concordance_simd(
    risk_i: f64,
    time_i: f64,
    weight: f64,
    risk_scores: &[f64],
    times: &[f64],
    skip_idx: usize,
) -> (f64, f64, f64, f64) {
    let counts = count_concordant_pairs_simd(risk_i, time_i, risk_scores, times, skip_idx);

    (
        counts.concordant as f64 * weight,
        counts.discordant as f64 * weight,
        counts.tied as f64 * weight,
        counts.valid_pairs as f64 * weight,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0];

        let result = dot_product_simd(&a, &b);
        let expected: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_sum() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let result = sum_simd(&values);
        let expected: f64 = values.iter().sum();

        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_sum_of_squares() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = sum_of_squares_simd(&values);
        let expected: f64 = values.iter().map(|x| x * x).sum();

        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_mean() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = mean_simd(&values);
        let expected = 3.0;

        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_subtract_scalar() {
        let values = vec![5.0, 10.0, 15.0, 20.0, 25.0];
        let scalar = 5.0;

        let result = subtract_scalar_simd(&values, scalar);
        let expected: Vec<f64> = values.iter().map(|x| x - scalar).collect();

        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-10);
        }
    }

    #[test]
    fn test_multiply_elementwise() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0];

        let result = multiply_elementwise_simd(&a, &b);
        let expected: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();

        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-10);
        }
    }

    #[test]
    fn test_variance() {
        let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];

        let result = variance_simd(&values);
        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        let expected: f64 =
            values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;

        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_empty_inputs() {
        assert!((sum_simd(&[]) - 0.0).abs() < 1e-10);
        assert!((mean_simd(&[]) - 0.0).abs() < 1e-10);
        assert!((dot_product_simd(&[], &[]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_large_array() {
        let n = 10000;
        let a: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..n).map(|i| (i * 2) as f64).collect();

        let result = dot_product_simd(&a, &b);
        let expected: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        assert!((result - expected).abs() < 1e-6);
    }
}
