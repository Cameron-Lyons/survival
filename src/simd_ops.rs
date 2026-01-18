use wide::{CmpGt, CmpLt, f64x4};

pub fn dot_product_simd(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum = f64x4::ZERO;

    for i in 0..chunks {
        let idx = i * 4;
        let va = f64x4::new([a[idx], a[idx + 1], a[idx + 2], a[idx + 3]]);
        let vb = f64x4::new([b[idx], b[idx + 1], b[idx + 2], b[idx + 3]]);
        sum += va * vb;
    }

    let arr = sum.to_array();
    let mut result = arr[0] + arr[1] + arr[2] + arr[3];

    let base = chunks * 4;
    for i in 0..remainder {
        result += a[base + i] * b[base + i];
    }

    result
}

pub fn weighted_sum_simd(values: &[f64], weights: &[f64]) -> f64 {
    dot_product_simd(values, weights)
}

pub fn sum_simd(values: &[f64]) -> f64 {
    let n = values.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum = f64x4::ZERO;

    for i in 0..chunks {
        let idx = i * 4;
        let v = f64x4::new([
            values[idx],
            values[idx + 1],
            values[idx + 2],
            values[idx + 3],
        ]);
        sum += v;
    }

    let arr = sum.to_array();
    let mut result = arr[0] + arr[1] + arr[2] + arr[3];

    let base = chunks * 4;
    for i in 0..remainder {
        result += values[base + i];
    }

    result
}

pub fn sum_of_squares_simd(values: &[f64]) -> f64 {
    let n = values.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum = f64x4::ZERO;

    for i in 0..chunks {
        let idx = i * 4;
        let v = f64x4::new([
            values[idx],
            values[idx + 1],
            values[idx + 2],
            values[idx + 3],
        ]);
        sum += v * v;
    }

    let arr = sum.to_array();
    let mut result = arr[0] + arr[1] + arr[2] + arr[3];

    let base = chunks * 4;
    for i in 0..remainder {
        result += values[base + i] * values[base + i];
    }

    result
}

pub fn mean_simd(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    sum_simd(values) / values.len() as f64
}

pub fn subtract_scalar_simd(values: &[f64], scalar: f64) -> Vec<f64> {
    let n = values.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut result = Vec::with_capacity(n);
    let scalar_vec = f64x4::splat(scalar);

    for i in 0..chunks {
        let idx = i * 4;
        let v = f64x4::new([
            values[idx],
            values[idx + 1],
            values[idx + 2],
            values[idx + 3],
        ]);
        let diff = v - scalar_vec;
        let arr = diff.to_array();
        result.extend_from_slice(&arr);
    }

    let base = chunks * 4;
    for i in 0..remainder {
        result.push(values[base + i] - scalar);
    }

    result
}

pub fn multiply_elementwise_simd(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len().min(b.len());
    let chunks = n / 4;
    let remainder = n % 4;

    let mut result = Vec::with_capacity(n);

    for i in 0..chunks {
        let idx = i * 4;
        let va = f64x4::new([a[idx], a[idx + 1], a[idx + 2], a[idx + 3]]);
        let vb = f64x4::new([b[idx], b[idx + 1], b[idx + 2], b[idx + 3]]);
        let prod = va * vb;
        let arr = prod.to_array();
        result.extend_from_slice(&arr);
    }

    let base = chunks * 4;
    for i in 0..remainder {
        result.push(a[base + i] * b[base + i]);
    }

    result
}

pub fn exp_approx_simd(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut result = Vec::with_capacity(n);

    for i in 0..chunks {
        let idx = i * 4;
        result.push(values[idx].exp());
        result.push(values[idx + 1].exp());
        result.push(values[idx + 2].exp());
        result.push(values[idx + 3].exp());
    }

    let base = chunks * 4;
    for i in 0..remainder {
        result.push(values[base + i].exp());
    }

    result
}

pub fn logistic_simd(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut result = Vec::with_capacity(n);

    for &x in values {
        result.push(1.0 / (1.0 + (-x).exp()));
    }

    result
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
    let chunks = n / 4;
    let remainder = n % 4;

    let risk_i_vec = f64x4::splat(risk_i);
    let time_i_vec = f64x4::splat(time_i);

    let mut concordant = 0usize;
    let mut discordant = 0usize;
    let mut tied = 0usize;
    let mut valid_pairs = 0usize;

    for chunk in 0..chunks {
        let idx = chunk * 4;

        if idx <= skip_idx && skip_idx < idx + 4 {
            for j in idx..idx + 4 {
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
            continue;
        }

        let times_j = f64x4::new([times[idx], times[idx + 1], times[idx + 2], times[idx + 3]]);
        let risks_j = f64x4::new([
            risk_scores[idx],
            risk_scores[idx + 1],
            risk_scores[idx + 2],
            risk_scores[idx + 3],
        ]);

        let valid_mask = times_j.simd_gt(time_i_vec);
        let concordant_mask = risk_i_vec.simd_gt(risks_j);
        let discordant_mask = risk_i_vec.simd_lt(risks_j);

        let valid_arr = valid_mask.to_array();
        let conc_arr = concordant_mask.to_array();
        let disc_arr = discordant_mask.to_array();

        for k in 0..4 {
            if valid_arr[k].to_bits() != 0 {
                valid_pairs += 1;
                if conc_arr[k].to_bits() != 0 {
                    concordant += 1;
                } else if disc_arr[k].to_bits() != 0 {
                    discordant += 1;
                } else {
                    tied += 1;
                }
            }
        }
    }

    let base = chunks * 4;
    for j in base..base + remainder {
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
