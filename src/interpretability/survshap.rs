#![allow(clippy::too_many_arguments)]

use pyo3::Py;
use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass]
pub enum AggregationMethod {
    Mean,
    Integral,
    MaxAbsolute,
    TimeWeighted,
}

#[pymethods]
impl AggregationMethod {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "mean" => Ok(AggregationMethod::Mean),
            "integral" => Ok(AggregationMethod::Integral),
            "max_absolute" | "maxabsolute" => Ok(AggregationMethod::MaxAbsolute),
            "time_weighted" | "timeweighted" => Ok(AggregationMethod::TimeWeighted),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown aggregation method. Use 'mean', 'integral', 'max_absolute', or 'time_weighted'",
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct SurvShapConfig {
    #[pyo3(get, set)]
    pub n_coalitions: usize,
    #[pyo3(get, set)]
    pub n_background: usize,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
    #[pyo3(get, set)]
    pub parallel: bool,
}

#[pymethods]
impl SurvShapConfig {
    #[new]
    #[pyo3(signature = (n_coalitions=2048, n_background=100, seed=None, parallel=true))]
    pub fn new(
        n_coalitions: usize,
        n_background: usize,
        seed: Option<u64>,
        parallel: bool,
    ) -> PyResult<Self> {
        if n_coalitions < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "n_coalitions must be at least 2",
            ));
        }
        if n_background == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "n_background must be positive",
            ));
        }

        Ok(SurvShapConfig {
            n_coalitions,
            n_background,
            seed,
            parallel,
        })
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct SurvShapResult {
    #[pyo3(get)]
    pub shap_values: Vec<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub base_value: Vec<f64>,
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub aggregated_importance: Option<Vec<f64>>,
}

#[pymethods]
impl SurvShapResult {
    fn __repr__(&self) -> String {
        let n_samples = self.shap_values.len();
        let n_features = if n_samples > 0 {
            self.shap_values[0].len()
        } else {
            0
        };
        let n_times = self.time_points.len();
        format!(
            "SurvShapResult(samples={}, features={}, time_points={})",
            n_samples, n_features, n_times
        )
    }

    fn get_sample_shap(&self, sample_idx: usize) -> PyResult<Vec<Vec<f64>>> {
        if sample_idx >= self.shap_values.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "sample_idx out of bounds",
            ));
        }
        Ok(self.shap_values[sample_idx].clone())
    }

    fn get_feature_shap(&self, feature_idx: usize) -> PyResult<Vec<Vec<f64>>> {
        let n_samples = self.shap_values.len();
        if n_samples == 0 {
            return Ok(Vec::new());
        }
        let n_features = self.shap_values[0].len();
        if feature_idx >= n_features {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "feature_idx out of bounds",
            ));
        }
        let result: Vec<Vec<f64>> = self
            .shap_values
            .iter()
            .map(|sample| sample[feature_idx].clone())
            .collect();
        Ok(result)
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct SurvShapExplanation {
    #[pyo3(get)]
    pub shap_values: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub base_value: Vec<f64>,
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub feature_values: Vec<f64>,
    #[pyo3(get)]
    pub aggregated_importance: Option<Vec<f64>>,
}

#[pymethods]
impl SurvShapExplanation {
    fn __repr__(&self) -> String {
        let n_features = self.shap_values.len();
        let n_times = self.time_points.len();
        format!(
            "SurvShapExplanation(features={}, time_points={})",
            n_features, n_times
        )
    }
}

fn compute_shapley_kernel_weights(n_features: usize, coalition_sizes: &[usize]) -> Vec<f64> {
    let n = n_features as f64;
    coalition_sizes
        .iter()
        .map(|&k| {
            if k == 0 || k == n_features {
                f64::INFINITY
            } else {
                let k_f = k as f64;
                let binom = binomial(n_features, k) as f64;
                (n - 1.0) / (binom * k_f * (n - k_f))
            }
        })
        .collect()
}

fn binomial(n: usize, k: usize) -> u64 {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut result: u64 = 1;
    for i in 0..k {
        result = result.saturating_mul((n - i) as u64) / ((i + 1) as u64);
    }
    result
}

fn sample_coalitions(
    n_features: usize,
    n_coalitions: usize,
    seed: u64,
) -> (Vec<Vec<bool>>, Vec<usize>) {
    let mut rng = fastrand::Rng::with_seed(seed);

    let mut coalitions = Vec::with_capacity(n_coalitions);
    let mut coalition_sizes = Vec::with_capacity(n_coalitions);

    coalitions.push(vec![false; n_features]);
    coalition_sizes.push(0);

    coalitions.push(vec![true; n_features]);
    coalition_sizes.push(n_features);

    let n_remaining = n_coalitions.saturating_sub(2);

    for _ in 0..n_remaining {
        let target_size = if rng.bool() {
            let half = n_features / 2;
            let offset = rng.usize(0..=(n_features / 4).max(1));
            if rng.bool() {
                (half + offset).min(n_features - 1)
            } else {
                half.saturating_sub(offset).max(1)
            }
        } else {
            rng.usize(1..n_features)
        };

        let mut coalition = vec![false; n_features];
        let mut indices: Vec<usize> = (0..n_features).collect();
        for i in (1..n_features).rev() {
            let j = rng.usize(0..=i);
            indices.swap(i, j);
        }
        for &idx in indices.iter().take(target_size) {
            coalition[idx] = true;
        }

        coalitions.push(coalition);
        coalition_sizes.push(target_size);
    }

    (coalitions, coalition_sizes)
}

fn weighted_least_squares(
    x_matrix: &[f64],
    y: &[f64],
    weights: &[f64],
    n_rows: usize,
    n_cols: usize,
) -> Vec<f64> {
    let mut xtwx = vec![0.0; n_cols * n_cols];
    let mut xtwy = vec![0.0; n_cols];

    for i in 0..n_rows {
        let w = weights[i];
        if !w.is_finite() || w <= 0.0 {
            continue;
        }

        for j in 0..n_cols {
            let xij = x_matrix[i * n_cols + j];
            xtwy[j] += w * xij * y[i];
            for k in 0..n_cols {
                let xik = x_matrix[i * n_cols + k];
                xtwx[j * n_cols + k] += w * xij * xik;
            }
        }
    }

    let reg = 1e-8;
    for j in 0..n_cols {
        xtwx[j * n_cols + j] += reg;
    }

    solve_positive_definite(&mut xtwx, &xtwy, n_cols)
}

fn solve_positive_definite(a: &mut [f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum <= 0.0 {
                    l[i * n + j] = 1e-10;
                } else {
                    l[i * n + j] = sum.sqrt();
                }
            } else {
                l[i * n + j] = sum / l[j * n + j].max(1e-10);
            }
        }
    }

    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i * n + j] * y[j];
        }
        y[i] = sum / l[i * n + i].max(1e-10);
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[j * n + i] * x[j];
        }
        x[i] = sum / l[i * n + i].max(1e-10);
    }

    x
}

fn evaluate_coalition_predictions(
    _x_explain: &[f64],
    _x_background: &[f64],
    predictions_explain: &[f64],
    predictions_background: &[f64],
    coalitions: &[Vec<bool>],
    n_explain: usize,
    n_background: usize,
    n_features: usize,
    n_times: usize,
    parallel: bool,
) -> Vec<Vec<f64>> {
    let process_coalition = |coalition: &Vec<bool>| -> Vec<f64> {
        let mut coalition_preds = vec![0.0; n_explain * n_times];

        for i in 0..n_explain {
            for bg_idx in 0..n_background {
                for t in 0..n_times {
                    let mut uses_explain_fully = true;
                    let mut uses_background_fully = true;

                    for &included in coalition.iter().take(n_features) {
                        if included {
                            uses_background_fully = false;
                        } else {
                            uses_explain_fully = false;
                        }
                    }

                    let pred = if uses_explain_fully {
                        predictions_explain[i * n_times + t]
                    } else if uses_background_fully {
                        predictions_background[bg_idx * n_times + t]
                    } else {
                        let weight_explain: f64 =
                            coalition.iter().map(|&c| if c { 1.0 } else { 0.0 }).sum();
                        let weight_bg = n_features as f64 - weight_explain;
                        let total = n_features as f64;

                        (weight_explain / total) * predictions_explain[i * n_times + t]
                            + (weight_bg / total) * predictions_background[bg_idx * n_times + t]
                    };

                    coalition_preds[i * n_times + t] += pred / n_background as f64;
                }
            }
        }

        coalition_preds
    };

    if parallel {
        coalitions.par_iter().map(process_coalition).collect()
    } else {
        coalitions.iter().map(process_coalition).collect()
    }
}

#[pyfunction]
#[pyo3(signature = (
    x_explain,
    x_background,
    predictions_explain,
    predictions_background,
    time_points,
    n_explain,
    n_background,
    n_features,
    config=None,
    aggregation_method=None
))]
pub fn survshap(
    x_explain: Vec<f64>,
    x_background: Vec<f64>,
    predictions_explain: Vec<f64>,
    predictions_background: Vec<f64>,
    time_points: Vec<f64>,
    n_explain: usize,
    n_background: usize,
    n_features: usize,
    config: Option<&SurvShapConfig>,
    aggregation_method: Option<AggregationMethod>,
) -> PyResult<SurvShapResult> {
    let n_times = time_points.len();

    if x_explain.len() != n_explain * n_features {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x_explain length must equal n_explain * n_features",
        ));
    }
    if x_background.len() != n_background * n_features {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x_background length must equal n_background * n_features",
        ));
    }
    if predictions_explain.len() != n_explain * n_times {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "predictions_explain length must equal n_explain * n_times",
        ));
    }
    if predictions_background.len() != n_background * n_times {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "predictions_background length must equal n_background * n_times",
        ));
    }

    let default_config = SurvShapConfig::new(2048, 100, None, true)?;
    let cfg = config.unwrap_or(&default_config);

    let seed = cfg.seed.unwrap_or(42);
    let (coalitions, coalition_sizes) = sample_coalitions(n_features, cfg.n_coalitions, seed);
    let kernel_weights = compute_shapley_kernel_weights(n_features, &coalition_sizes);

    let coalition_preds = evaluate_coalition_predictions(
        &x_explain,
        &x_background,
        &predictions_explain,
        &predictions_background,
        &coalitions,
        n_explain,
        n_background,
        n_features,
        n_times,
        cfg.parallel,
    );

    let base_value: Vec<f64> = (0..n_times)
        .map(|t| {
            predictions_background
                .iter()
                .skip(t)
                .step_by(n_times)
                .sum::<f64>()
                / n_background as f64
        })
        .collect();

    let mut shap_values = vec![vec![vec![0.0; n_times]; n_features]; n_explain];

    for sample_idx in 0..n_explain {
        for t in 0..n_times {
            let n_coalitions = coalitions.len();
            let mut x_matrix = vec![0.0; n_coalitions * n_features];
            let mut y = vec![0.0; n_coalitions];

            for (c_idx, coalition) in coalitions.iter().enumerate() {
                for (f_idx, &included) in coalition.iter().enumerate() {
                    x_matrix[c_idx * n_features + f_idx] = if included { 1.0 } else { 0.0 };
                }
                y[c_idx] = coalition_preds[c_idx][sample_idx * n_times + t] - base_value[t];
            }

            let shap_t =
                weighted_least_squares(&x_matrix, &y, &kernel_weights, n_coalitions, n_features);

            for (f_idx, &val) in shap_t.iter().enumerate() {
                shap_values[sample_idx][f_idx][t] = val;
            }
        }
    }

    let aggregated_importance = aggregation_method.map(|method| {
        aggregate_shap_values(&shap_values, &time_points, method, n_features, n_times)
    });

    Ok(SurvShapResult {
        shap_values,
        base_value,
        time_points,
        aggregated_importance,
    })
}

fn aggregate_shap_values(
    shap_values: &[Vec<Vec<f64>>],
    time_points: &[f64],
    method: AggregationMethod,
    n_features: usize,
    n_times: usize,
) -> Vec<f64> {
    let n_samples = shap_values.len();
    if n_samples == 0 || n_times == 0 {
        return vec![0.0; n_features];
    }

    let mut importance = vec![0.0; n_features];

    for f in 0..n_features {
        let mut feature_agg = 0.0;

        for sample in shap_values.iter() {
            let shap_t = &sample[f];

            let sample_agg = match method {
                AggregationMethod::Mean => {
                    shap_t.iter().map(|v| v.abs()).sum::<f64>() / n_times as f64
                }
                AggregationMethod::MaxAbsolute => {
                    shap_t.iter().map(|v| v.abs()).fold(0.0, f64::max)
                }
                AggregationMethod::Integral => {
                    if n_times < 2 {
                        shap_t.first().copied().unwrap_or(0.0).abs()
                    } else {
                        let mut integral = 0.0;
                        for i in 1..n_times {
                            let dt = time_points[i] - time_points[i - 1];
                            let avg = (shap_t[i].abs() + shap_t[i - 1].abs()) / 2.0;
                            integral += avg * dt;
                        }
                        integral
                    }
                }
                AggregationMethod::TimeWeighted => {
                    let total_time =
                        time_points.last().unwrap_or(&1.0) - time_points.first().unwrap_or(&0.0);
                    if total_time <= 0.0 {
                        shap_t.iter().map(|v| v.abs()).sum::<f64>() / n_times as f64
                    } else {
                        let mut weighted_sum = 0.0;
                        for (i, &t) in time_points.iter().enumerate() {
                            let weight = 1.0 - (t - time_points[0]) / total_time;
                            weighted_sum += shap_t[i].abs() * weight;
                        }
                        weighted_sum / n_times as f64
                    }
                }
            };

            feature_agg += sample_agg;
        }

        importance[f] = feature_agg / n_samples as f64;
    }

    importance
}

#[pyfunction]
#[pyo3(signature = (shap_values, time_points, method))]
pub fn aggregate_survshap(
    shap_values: Vec<Vec<Vec<f64>>>,
    time_points: Vec<f64>,
    method: AggregationMethod,
) -> PyResult<Vec<f64>> {
    let n_samples = shap_values.len();
    if n_samples == 0 {
        return Ok(Vec::new());
    }

    let n_features = shap_values[0].len();
    let n_times = time_points.len();

    if n_features == 0 {
        return Ok(Vec::new());
    }

    for sample in &shap_values {
        if sample.len() != n_features {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "All samples must have the same number of features",
            ));
        }
        for feature_shap in sample {
            if feature_shap.len() != n_times {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "SHAP values time dimension must match time_points length",
                ));
            }
        }
    }

    Ok(aggregate_shap_values(
        &shap_values,
        &time_points,
        method,
        n_features,
        n_times,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    x_explain,
    x_background,
    time_points,
    n_explain,
    n_background,
    n_features,
    predict_fn,
    config=None,
    aggregation_method=None
))]
pub fn survshap_from_model(
    py: Python<'_>,
    x_explain: Vec<f64>,
    x_background: Vec<f64>,
    time_points: Vec<f64>,
    n_explain: usize,
    n_background: usize,
    n_features: usize,
    predict_fn: Py<PyAny>,
    config: Option<&SurvShapConfig>,
    aggregation_method: Option<AggregationMethod>,
) -> PyResult<SurvShapResult> {
    if x_explain.len() != n_explain * n_features {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x_explain length must equal n_explain * n_features",
        ));
    }
    if x_background.len() != n_background * n_features {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x_background length must equal n_background * n_features",
        ));
    }

    let predictions_explain: Vec<f64> = predict_fn
        .call(py, (x_explain.clone(), n_explain), None)?
        .extract(py)?;

    let predictions_background: Vec<f64> = predict_fn
        .call(py, (x_background.clone(), n_background), None)?
        .extract(py)?;

    survshap(
        x_explain,
        x_background,
        predictions_explain,
        predictions_background,
        time_points,
        n_explain,
        n_background,
        n_features,
        config,
        aggregation_method,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binomial() {
        assert_eq!(binomial(5, 0), 1);
        assert_eq!(binomial(5, 1), 5);
        assert_eq!(binomial(5, 2), 10);
        assert_eq!(binomial(5, 3), 10);
        assert_eq!(binomial(5, 4), 5);
        assert_eq!(binomial(5, 5), 1);
        assert_eq!(binomial(10, 5), 252);
    }

    #[test]
    fn test_sample_coalitions() {
        let (coalitions, sizes) = sample_coalitions(5, 100, 42);
        assert_eq!(coalitions.len(), 100);
        assert_eq!(sizes.len(), 100);

        assert!(coalitions[0].iter().all(|&c| !c));
        assert_eq!(sizes[0], 0);

        assert!(coalitions[1].iter().all(|&c| c));
        assert_eq!(sizes[1], 5);

        for (i, (coalition, &size)) in coalitions.iter().zip(sizes.iter()).enumerate() {
            let actual_size = coalition.iter().filter(|&&c| c).count();
            assert_eq!(actual_size, size, "Coalition {} size mismatch", i);
        }
    }

    #[test]
    fn test_kernel_weights() {
        let weights = compute_shapley_kernel_weights(4, &[0, 1, 2, 3, 4]);

        assert!(weights[0].is_infinite());
        assert!(weights[4].is_infinite());

        assert!(weights[1] > 0.0);
        assert!(weights[2] > 0.0);
        assert!(weights[3] > 0.0);
    }

    #[test]
    fn test_weighted_least_squares() {
        let x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let y = vec![1.0, 2.0, 3.0];
        let weights = vec![1.0, 1.0, 1.0];

        let result = weighted_least_squares(&x, &y, &weights, 3, 2);
        assert_eq!(result.len(), 2);
        assert!(result.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_survshap_basic() {
        let n_explain = 2;
        let n_background = 3;
        let n_features = 3;
        let n_times = 4;

        let x_explain: Vec<f64> = (0..(n_explain * n_features))
            .map(|i| (i as f64) * 0.1)
            .collect();
        let x_background: Vec<f64> = (0..(n_background * n_features))
            .map(|i| (i as f64) * 0.05)
            .collect();

        let predictions_explain: Vec<f64> = (0..(n_explain * n_times))
            .map(|i| 1.0 - (i as f64) * 0.05)
            .collect();
        let predictions_background: Vec<f64> = (0..(n_background * n_times))
            .map(|i| 0.9 - (i as f64) * 0.02)
            .collect();

        let time_points: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];

        let config = SurvShapConfig::new(100, n_background, Some(42), false).unwrap();

        let result = survshap(
            x_explain,
            x_background,
            predictions_explain.clone(),
            predictions_background,
            time_points.clone(),
            n_explain,
            n_background,
            n_features,
            Some(&config),
            Some(AggregationMethod::Mean),
        )
        .unwrap();

        assert_eq!(result.shap_values.len(), n_explain);
        assert_eq!(result.shap_values[0].len(), n_features);
        assert_eq!(result.shap_values[0][0].len(), n_times);
        assert_eq!(result.base_value.len(), n_times);
        assert_eq!(result.time_points.len(), n_times);
        assert!(result.aggregated_importance.is_some());
        assert_eq!(result.aggregated_importance.unwrap().len(), n_features);
    }

    #[test]
    fn test_aggregation_methods() {
        let shap_values = vec![vec![vec![0.1, 0.2, 0.3, 0.4]; 2]; 3];
        let time_points = vec![1.0, 2.0, 3.0, 4.0];

        for method in [
            AggregationMethod::Mean,
            AggregationMethod::Integral,
            AggregationMethod::MaxAbsolute,
            AggregationMethod::TimeWeighted,
        ] {
            let result =
                aggregate_survshap(shap_values.clone(), time_points.clone(), method).unwrap();
            assert_eq!(result.len(), 2);
            assert!(result.iter().all(|&v| v.is_finite() && v >= 0.0));
        }
    }

    #[test]
    fn test_shap_additivity() {
        let n_explain = 1;
        let n_background = 10;
        let n_features = 2;
        let n_times = 3;

        let x_explain: Vec<f64> = vec![0.5, 0.5];
        let x_background: Vec<f64> = (0..(n_background * n_features))
            .map(|i| (i as f64) * 0.1 % 1.0)
            .collect();

        let predictions_explain: Vec<f64> = vec![0.9, 0.8, 0.7];
        let predictions_background: Vec<f64> = (0..(n_background * n_times))
            .map(|i| 0.95 - (i as f64) * 0.01)
            .collect();

        let time_points: Vec<f64> = vec![1.0, 2.0, 3.0];

        let config = SurvShapConfig::new(500, n_background, Some(42), false).unwrap();

        let result = survshap(
            x_explain,
            x_background,
            predictions_explain.clone(),
            predictions_background,
            time_points,
            n_explain,
            n_background,
            n_features,
            Some(&config),
            None,
        )
        .unwrap();

        for t in 0..n_times {
            let shap_sum: f64 = (0..n_features).map(|f| result.shap_values[0][f][t]).sum();
            let reconstructed = result.base_value[t] + shap_sum;
            let error = (reconstructed - predictions_explain[t]).abs();
            assert!(
                error < 0.5,
                "Additivity check failed at t={}: reconstructed={}, actual={}, error={}",
                t,
                reconstructed,
                predictions_explain[t],
                error
            );
        }
    }

    #[test]
    fn test_config_validation() {
        assert!(SurvShapConfig::new(1, 100, None, true).is_err());
        assert!(SurvShapConfig::new(100, 0, None, true).is_err());
        assert!(SurvShapConfig::new(100, 50, Some(42), false).is_ok());
    }
}
