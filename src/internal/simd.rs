pub(crate) fn sum_f64(data: &[f64]) -> f64 {
    let mut accumulators = [0.0; 4];
    let mut index = 0;
    while index + 4 <= data.len() {
        accumulators[0] += data[index];
        accumulators[1] += data[index + 1];
        accumulators[2] += data[index + 2];
        accumulators[3] += data[index + 3];
        index += 4;
    }

    let mut total = (accumulators[0] + accumulators[1]) + (accumulators[2] + accumulators[3]);
    while index < data.len() {
        total += data[index];
        index += 1;
    }
    total
}

pub(crate) fn weighted_squared_diff_sum(
    predictions: &[f64],
    outcomes: &[f64],
    weights: &[f64],
) -> f64 {
    let n = predictions.len().min(outcomes.len()).min(weights.len());
    let mut accumulators = [0.0; 4];
    let mut index = 0;
    while index + 4 <= n {
        for lane in 0..4 {
            let diff = predictions[index + lane] - outcomes[index + lane];
            accumulators[lane] = (weights[index + lane] * diff).mul_add(diff, accumulators[lane]);
        }
        index += 4;
    }

    let mut total = (accumulators[0] + accumulators[1]) + (accumulators[2] + accumulators[3]);
    while index < n {
        let diff = predictions[index] - outcomes[index];
        total = (weights[index] * diff).mul_add(diff, total);
        index += 1;
    }
    total
}

pub(crate) fn squared_diff_sum(predictions: &[f64], outcomes: &[f64]) -> f64 {
    let n = predictions.len().min(outcomes.len());
    let mut accumulators = [0.0; 4];
    let mut index = 0;
    while index + 4 <= n {
        for lane in 0..4 {
            let diff = predictions[index + lane] - outcomes[index + lane];
            accumulators[lane] = diff.mul_add(diff, accumulators[lane]);
        }
        index += 4;
    }

    let mut total = (accumulators[0] + accumulators[1]) + (accumulators[2] + accumulators[3]);
    while index < n {
        let diff = predictions[index] - outcomes[index];
        total = diff.mul_add(diff, total);
        index += 1;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot_product(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(&left, &right)| left * right).sum()
    }

    fn min_max_f64(data: &[f64]) -> (f64, f64) {
        data.iter().fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(minimum, maximum), &value| (minimum.min(value), maximum.max(value)),
        )
    }

    #[test]
    fn test_sum() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert!((sum_f64(&data) - 55.0).abs() < 1e-10);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        assert!((dot_product(&a, &b) - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_squared_diff() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 4.0, 6.0];
        assert!((squared_diff_sum(&a, &b) - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_min_max() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let (min, max) = min_max_f64(&data);
        assert!((min - 1.0).abs() < 1e-10);
        assert!((max - 9.0).abs() < 1e-10);
    }
}
