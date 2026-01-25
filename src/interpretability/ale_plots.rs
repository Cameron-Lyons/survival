#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct ALEResult {
    #[pyo3(get)]
    pub feature_values: Vec<f64>,
    #[pyo3(get)]
    pub ale_values: Vec<f64>,
    #[pyo3(get)]
    pub feature_index: usize,
    #[pyo3(get)]
    pub num_intervals: usize,
}

#[pymethods]
impl ALEResult {
    fn __repr__(&self) -> String {
        format!(
            "ALEResult(feature_index={}, num_intervals={})",
            self.feature_index, self.num_intervals
        )
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct ALE2DResult {
    #[pyo3(get)]
    pub feature1_values: Vec<f64>,
    #[pyo3(get)]
    pub feature2_values: Vec<f64>,
    #[pyo3(get)]
    pub ale_values: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub feature1_index: usize,
    #[pyo3(get)]
    pub feature2_index: usize,
}

#[pymethods]
impl ALE2DResult {
    fn __repr__(&self) -> String {
        format!(
            "ALE2DResult(feature1={}, feature2={})",
            self.feature1_index, self.feature2_index
        )
    }
}

fn compute_quantile_bins(values: &[f64], num_intervals: usize) -> Vec<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    let mut bins = Vec::with_capacity(num_intervals + 1);

    for i in 0..=num_intervals {
        let idx = (i * (n - 1)) / num_intervals;
        bins.push(sorted[idx]);
    }

    bins.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
    bins
}

fn find_interval(value: f64, bins: &[f64]) -> usize {
    for (i, window) in bins.windows(2).enumerate() {
        if value >= window[0] && value < window[1] {
            return i;
        }
    }
    bins.len().saturating_sub(2)
}

#[pyfunction]
#[pyo3(signature = (
    covariates,
    predictions,
    feature_index,
    num_intervals=20
))]
pub fn compute_ale(
    covariates: Vec<Vec<f64>>,
    predictions: Vec<f64>,
    feature_index: usize,
    num_intervals: usize,
) -> PyResult<ALEResult> {
    if covariates.is_empty() || predictions.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "covariates and predictions must not be empty",
        ));
    }

    if covariates.len() != predictions.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "covariates and predictions must have the same length",
        ));
    }

    if feature_index >= covariates[0].len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "feature_index out of bounds",
        ));
    }

    let feature_values: Vec<f64> = covariates.iter().map(|c| c[feature_index]).collect();
    let bins = compute_quantile_bins(&feature_values, num_intervals);
    let actual_intervals = bins.len() - 1;

    let interval_diffs: Vec<(usize, f64)> = covariates
        .par_iter()
        .zip(predictions.par_iter())
        .filter_map(|(cov, &pred)| {
            let interval = find_interval(cov[feature_index], &bins);
            if interval < actual_intervals {
                let lower = bins[interval];
                let upper = bins[interval + 1];
                let range = upper - lower;
                if range > 1e-10 {
                    let diff = pred * (upper - lower) / range;
                    Some((interval, diff))
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    let mut interval_sums = vec![0.0; actual_intervals];
    let mut interval_counts = vec![0usize; actual_intervals];

    for (interval, diff) in interval_diffs {
        interval_sums[interval] += diff;
        interval_counts[interval] += 1;
    }

    let mut ale_values = vec![0.0; actual_intervals + 1];
    let mut cumsum = 0.0;

    for i in 0..actual_intervals {
        let avg_diff = if interval_counts[i] > 0 {
            interval_sums[i] / interval_counts[i] as f64
        } else {
            0.0
        };
        cumsum += avg_diff;
        ale_values[i + 1] = cumsum;
    }

    let mean_ale: f64 = ale_values.iter().sum::<f64>() / ale_values.len() as f64;
    for v in &mut ale_values {
        *v -= mean_ale;
    }

    Ok(ALEResult {
        feature_values: bins,
        ale_values,
        feature_index,
        num_intervals: actual_intervals,
    })
}

#[pyfunction]
#[pyo3(signature = (
    covariates,
    predictions,
    feature1_index,
    feature2_index,
    num_intervals=10
))]
pub fn compute_ale_2d(
    covariates: Vec<Vec<f64>>,
    predictions: Vec<f64>,
    feature1_index: usize,
    feature2_index: usize,
    num_intervals: usize,
) -> PyResult<ALE2DResult> {
    if covariates.is_empty() || predictions.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "covariates and predictions must not be empty",
        ));
    }

    let n_features = covariates[0].len();
    if feature1_index >= n_features || feature2_index >= n_features {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "feature indices out of bounds",
        ));
    }

    let feature1_values: Vec<f64> = covariates.iter().map(|c| c[feature1_index]).collect();
    let feature2_values: Vec<f64> = covariates.iter().map(|c| c[feature2_index]).collect();

    let bins1 = compute_quantile_bins(&feature1_values, num_intervals);
    let bins2 = compute_quantile_bins(&feature2_values, num_intervals);

    let n1 = bins1.len() - 1;
    let n2 = bins2.len() - 1;

    let mut ale_grid = vec![vec![0.0; n2 + 1]; n1 + 1];

    let cell_diffs: Vec<(usize, usize, f64)> = covariates
        .par_iter()
        .zip(predictions.par_iter())
        .filter_map(|(cov, &pred)| {
            let i1 = find_interval(cov[feature1_index], &bins1);
            let i2 = find_interval(cov[feature2_index], &bins2);
            if i1 < n1 && i2 < n2 {
                Some((i1, i2, pred))
            } else {
                None
            }
        })
        .collect();

    let mut cell_sums = vec![vec![0.0; n2]; n1];
    let mut cell_counts = vec![vec![0usize; n2]; n1];

    for (i1, i2, diff) in cell_diffs {
        cell_sums[i1][i2] += diff;
        cell_counts[i1][i2] += 1;
    }

    for i in 0..n1 {
        for j in 0..n2 {
            let avg = if cell_counts[i][j] > 0 {
                cell_sums[i][j] / cell_counts[i][j] as f64
            } else {
                0.0
            };
            ale_grid[i + 1][j + 1] = avg;
        }
    }

    for i in 1..=n1 {
        for j in 1..=n2 {
            ale_grid[i][j] += ale_grid[i - 1][j] + ale_grid[i][j - 1] - ale_grid[i - 1][j - 1];
        }
    }

    let total: f64 = ale_grid.iter().flat_map(|row| row.iter()).sum();
    let count = ((n1 + 1) * (n2 + 1)) as f64;
    let mean = total / count;

    for row in &mut ale_grid {
        for v in row {
            *v -= mean;
        }
    }

    Ok(ALE2DResult {
        feature1_values: bins1,
        feature2_values: bins2,
        ale_values: ale_grid,
        feature1_index,
        feature2_index,
    })
}

#[pyfunction]
#[pyo3(signature = (
    covariates,
    predictions,
    time_points,
    feature_index,
    num_intervals=20
))]
pub fn compute_time_varying_ale(
    covariates: Vec<Vec<f64>>,
    predictions: Vec<Vec<f64>>,
    time_points: Vec<f64>,
    feature_index: usize,
    num_intervals: usize,
) -> PyResult<Vec<ALEResult>> {
    if covariates.is_empty() || predictions.is_empty() || time_points.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "inputs must not be empty",
        ));
    }

    let n_times = time_points.len();
    if predictions[0].len() != n_times {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "predictions must have same number of time points",
        ));
    }

    let results: Vec<ALEResult> = (0..n_times)
        .into_par_iter()
        .map(|t| {
            let preds_at_t: Vec<f64> = predictions.iter().map(|p| p[t]).collect();
            compute_ale(covariates.clone(), preds_at_t, feature_index, num_intervals).unwrap()
        })
        .collect();

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantile_bins() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let bins = compute_quantile_bins(&values, 4);
        assert!(bins.len() >= 2);
        assert!((bins[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_find_interval() {
        let bins = vec![0.0, 1.0, 2.0, 3.0];
        assert_eq!(find_interval(0.5, &bins), 0);
        assert_eq!(find_interval(1.5, &bins), 1);
        assert_eq!(find_interval(2.5, &bins), 2);
    }

    #[test]
    fn test_ale_computation() {
        let covariates = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let predictions = vec![0.0, 1.0, 2.0, 3.0];
        let result = compute_ale(covariates, predictions, 0, 2).unwrap();
        assert!(!result.ale_values.is_empty());
    }
}
