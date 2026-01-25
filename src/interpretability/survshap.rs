#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

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
pub struct FeatureImportance {
    #[pyo3(get)]
    pub feature_idx: usize,
    #[pyo3(get)]
    pub importance: f64,
    #[pyo3(get)]
    pub importance_std: Option<f64>,
}

#[pymethods]
impl FeatureImportance {
    fn __repr__(&self) -> String {
        match self.importance_std {
            Some(std) => format!(
                "FeatureImportance(idx={}, importance={:.4} ± {:.4})",
                self.feature_idx, self.importance, std
            ),
            None => format!(
                "FeatureImportance(idx={}, importance={:.4})",
                self.feature_idx, self.importance
            ),
        }
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

    fn get_shap_at_time(&self, time_idx: usize) -> PyResult<Vec<Vec<f64>>> {
        let n_samples = self.shap_values.len();
        if n_samples == 0 {
            return Ok(Vec::new());
        }
        let n_times = self.time_points.len();
        if time_idx >= n_times {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "time_idx out of bounds",
            ));
        }
        let n_features = self.shap_values[0].len();
        let result: Vec<Vec<f64>> = self
            .shap_values
            .iter()
            .map(|sample| (0..n_features).map(|f| sample[f][time_idx]).collect())
            .collect();
        Ok(result)
    }

    #[pyo3(signature = (method=AggregationMethod::Mean, top_k=None))]
    fn feature_ranking(
        &self,
        method: AggregationMethod,
        top_k: Option<usize>,
    ) -> PyResult<Vec<FeatureImportance>> {
        let n_samples = self.shap_values.len();
        if n_samples == 0 {
            return Ok(Vec::new());
        }
        let n_features = self.shap_values[0].len();
        let n_times = self.time_points.len();

        let importance = aggregate_shap_values(
            &self.shap_values,
            &self.time_points,
            method,
            n_features,
            n_times,
        );

        let time_diffs: Vec<f64> = if n_times >= 2 {
            self.time_points.windows(2).map(|w| w[1] - w[0]).collect()
        } else {
            Vec::new()
        };

        let total_time =
            self.time_points.last().unwrap_or(&1.0) - self.time_points.first().unwrap_or(&0.0);
        let time_weights: Vec<f64> = if total_time > 0.0 {
            self.time_points
                .iter()
                .map(|&t| 1.0 - (t - self.time_points[0]) / total_time)
                .collect()
        } else {
            Vec::new()
        };

        let mut stds = vec![0.0; n_features];
        for f in 0..n_features {
            let sample_importances: Vec<f64> = self
                .shap_values
                .iter()
                .map(|sample| {
                    let shap_t = &sample[f];
                    match method {
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
                                for i in 0..time_diffs.len() {
                                    let avg = (shap_t[i + 1].abs() + shap_t[i].abs()) / 2.0;
                                    integral += avg * time_diffs[i];
                                }
                                integral
                            }
                        }
                        AggregationMethod::TimeWeighted => {
                            if total_time <= 0.0 {
                                shap_t.iter().map(|v| v.abs()).sum::<f64>() / n_times as f64
                            } else {
                                let mut weighted_sum = 0.0;
                                for (i, &weight) in time_weights.iter().enumerate() {
                                    weighted_sum += shap_t[i].abs() * weight;
                                }
                                weighted_sum / n_times as f64
                            }
                        }
                    }
                })
                .collect();

            let mean = importance[f];
            let variance: f64 = sample_importances
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>()
                / n_samples as f64;
            stds[f] = variance.sqrt();
        }

        let mut ranking: Vec<FeatureImportance> = (0..n_features)
            .map(|f| FeatureImportance {
                feature_idx: f,
                importance: importance[f],
                importance_std: Some(stds[f]),
            })
            .collect();

        ranking.sort_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some(k) = top_k {
            ranking.truncate(k);
        }

        Ok(ranking)
    }

    fn mean_absolute_shap(&self) -> Vec<f64> {
        let n_samples = self.shap_values.len();
        if n_samples == 0 {
            return Vec::new();
        }
        let n_features = self.shap_values[0].len();
        let n_times = self.time_points.len();

        (0..n_features)
            .map(|f| {
                let total: f64 = self
                    .shap_values
                    .iter()
                    .flat_map(|sample| sample[f].iter())
                    .map(|v| v.abs())
                    .sum();
                total / (n_samples * n_times) as f64
            })
            .collect()
    }

    fn check_additivity(&self, predictions: Vec<f64>, tolerance: f64) -> PyResult<Vec<bool>> {
        let n_samples = self.shap_values.len();
        let n_times = self.time_points.len();

        if predictions.len() != n_samples * n_times {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "predictions length must equal n_samples * n_times",
            ));
        }

        let n_features = if n_samples > 0 {
            self.shap_values[0].len()
        } else {
            0
        };

        let mut results = Vec::with_capacity(n_samples * n_times);

        for i in 0..n_samples {
            for t in 0..n_times {
                let shap_sum: f64 = (0..n_features).map(|f| self.shap_values[i][f][t]).sum();
                let reconstructed = self.base_value[t] + shap_sum;
                let actual = predictions[i * n_times + t];
                results.push((reconstructed - actual).abs() <= tolerance);
            }
        }

        Ok(results)
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

#[derive(Debug, Clone)]
#[pyclass]
pub struct BootstrapSurvShapResult {
    #[pyo3(get)]
    pub shap_values_mean: Vec<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub shap_values_std: Vec<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub shap_values_lower: Vec<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub shap_values_upper: Vec<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub base_value: Vec<f64>,
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub n_bootstrap: usize,
    #[pyo3(get)]
    pub confidence_level: f64,
}

#[pymethods]
impl BootstrapSurvShapResult {
    fn __repr__(&self) -> String {
        let n_samples = self.shap_values_mean.len();
        let n_features = if n_samples > 0 {
            self.shap_values_mean[0].len()
        } else {
            0
        };
        let n_times = self.time_points.len();
        format!(
            "BootstrapSurvShapResult(samples={}, features={}, time_points={}, n_bootstrap={}, confidence={:.0}%)",
            n_samples,
            n_features,
            n_times,
            self.n_bootstrap,
            self.confidence_level * 100.0
        )
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct PermutationImportanceResult {
    #[pyo3(get)]
    pub importance: Vec<f64>,
    #[pyo3(get)]
    pub importance_std: Vec<f64>,
    #[pyo3(get)]
    pub baseline_score: f64,
    #[pyo3(get)]
    pub n_repeats: usize,
}

#[pymethods]
impl PermutationImportanceResult {
    fn __repr__(&self) -> String {
        format!(
            "PermutationImportanceResult(n_features={}, n_repeats={}, baseline={:.4})",
            self.importance.len(),
            self.n_repeats,
            self.baseline_score
        )
    }

    fn feature_ranking(&self, top_k: Option<usize>) -> Vec<FeatureImportance> {
        let mut ranking: Vec<FeatureImportance> = self
            .importance
            .iter()
            .zip(self.importance_std.iter())
            .enumerate()
            .map(|(idx, (&imp, &std))| FeatureImportance {
                feature_idx: idx,
                importance: imp,
                importance_std: Some(std),
            })
            .collect();

        ranking.sort_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some(k) = top_k {
            ranking.truncate(k);
        }

        ranking
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct ShapInteractionResult {
    #[pyo3(get)]
    pub interaction_values: Vec<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub aggregated_interactions: Option<Vec<Vec<f64>>>,
}

#[pymethods]
impl ShapInteractionResult {
    fn __repr__(&self) -> String {
        let n_features = self.interaction_values.len();
        let n_times = self.time_points.len();
        format!(
            "ShapInteractionResult(features={}, time_points={})",
            n_features, n_times
        )
    }

    fn get_interaction(&self, feature_i: usize, feature_j: usize) -> PyResult<Vec<f64>> {
        let n_features = self.interaction_values.len();
        if feature_i >= n_features || feature_j >= n_features {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "feature index out of bounds",
            ));
        }
        Ok(self.interaction_values[feature_i][feature_j].clone())
    }

    #[allow(clippy::needless_range_loop)]
    fn top_interactions(&self, top_k: usize) -> Vec<(usize, usize, f64)> {
        let n_features = self.interaction_values.len();
        let mut interactions: Vec<(usize, usize, f64)> = Vec::new();

        if let Some(ref agg) = self.aggregated_interactions {
            for i in 0..n_features {
                for j in (i + 1)..n_features {
                    interactions.push((i, j, agg[i][j]));
                }
            }
        } else {
            let n_times = self.time_points.len();
            for i in 0..n_features {
                for j in (i + 1)..n_features {
                    let mean_interaction: f64 = self.interaction_values[i][j]
                        .iter()
                        .map(|v| v.abs())
                        .sum::<f64>()
                        / n_times as f64;
                    interactions.push((i, j, mean_interaction));
                }
            }
        }

        interactions.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        interactions.truncate(top_k);
        interactions
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

fn compute_shap_inner(
    x_explain: &[f64],
    x_background: &[f64],
    predictions_explain: &[f64],
    predictions_background: &[f64],
    time_points: &[f64],
    n_explain: usize,
    n_background: usize,
    n_features: usize,
    n_coalitions: usize,
    seed: u64,
    parallel: bool,
) -> (Vec<Vec<Vec<f64>>>, Vec<f64>) {
    let n_times = time_points.len();

    let (coalitions, coalition_sizes) = sample_coalitions(n_features, n_coalitions, seed);
    let kernel_weights = compute_shapley_kernel_weights(n_features, &coalition_sizes);

    let coalition_preds = evaluate_coalition_predictions(
        x_explain,
        x_background,
        predictions_explain,
        predictions_background,
        &coalitions,
        n_explain,
        n_background,
        n_features,
        n_times,
        parallel,
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
            let n_coal = coalitions.len();
            let mut x_matrix = vec![0.0; n_coal * n_features];
            let mut y = vec![0.0; n_coal];

            for (c_idx, coalition) in coalitions.iter().enumerate() {
                for (f_idx, &included) in coalition.iter().enumerate() {
                    x_matrix[c_idx * n_features + f_idx] = if included { 1.0 } else { 0.0 };
                }
                y[c_idx] = coalition_preds[c_idx][sample_idx * n_times + t] - base_value[t];
            }

            let shap_t = weighted_least_squares(&x_matrix, &y, &kernel_weights, n_coal, n_features);

            for (f_idx, &val) in shap_t.iter().enumerate() {
                shap_values[sample_idx][f_idx][t] = val;
            }
        }
    }

    (shap_values, base_value)
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

    let (shap_values, base_value) = compute_shap_inner(
        &x_explain,
        &x_background,
        &predictions_explain,
        &predictions_background,
        &time_points,
        n_explain,
        n_background,
        n_features,
        cfg.n_coalitions,
        seed,
        cfg.parallel,
    );

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
    n_bootstrap=100,
    confidence_level=0.95,
    config=None
))]
pub fn survshap_bootstrap(
    x_explain: Vec<f64>,
    x_background: Vec<f64>,
    predictions_explain: Vec<f64>,
    predictions_background: Vec<f64>,
    time_points: Vec<f64>,
    n_explain: usize,
    n_background: usize,
    n_features: usize,
    n_bootstrap: usize,
    confidence_level: f64,
    config: Option<&SurvShapConfig>,
) -> PyResult<BootstrapSurvShapResult> {
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
    if !(0.0..1.0).contains(&confidence_level) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "confidence_level must be between 0 and 1",
        ));
    }

    let default_config = SurvShapConfig::new(2048, 100, None, true)?;
    let cfg = config.unwrap_or(&default_config);
    let base_seed = cfg.seed.unwrap_or(42);

    let x_bg_ref = &x_background;
    let preds_bg_ref = &predictions_background;
    let x_exp_ref = &x_explain;
    let preds_exp_ref = &predictions_explain;
    let time_pts_ref = &time_points;

    let bootstrap_results: Vec<(Vec<Vec<Vec<f64>>>, Vec<f64>)> = (0..n_bootstrap)
        .into_par_iter()
        .map(|b| {
            let seed = base_seed.wrapping_add(b as u64);
            let mut rng = fastrand::Rng::with_seed(seed);

            let bg_indices: Vec<usize> = (0..n_background)
                .map(|_| rng.usize(0..n_background))
                .collect();

            let sampled_x_bg: Vec<f64> = bg_indices
                .iter()
                .flat_map(|&idx| (0..n_features).map(move |f| x_bg_ref[idx * n_features + f]))
                .collect();

            let sampled_preds_bg: Vec<f64> = bg_indices
                .iter()
                .flat_map(|&idx| (0..n_times).map(move |t| preds_bg_ref[idx * n_times + t]))
                .collect();

            compute_shap_inner(
                x_exp_ref,
                &sampled_x_bg,
                preds_exp_ref,
                &sampled_preds_bg,
                time_pts_ref,
                n_explain,
                n_background,
                n_features,
                cfg.n_coalitions,
                seed,
                false,
            )
        })
        .collect();

    let mut shap_values_mean = vec![vec![vec![0.0; n_times]; n_features]; n_explain];
    let mut shap_values_std = vec![vec![vec![0.0; n_times]; n_features]; n_explain];
    let mut shap_values_lower = vec![vec![vec![0.0; n_times]; n_features]; n_explain];
    let mut shap_values_upper = vec![vec![vec![0.0; n_times]; n_features]; n_explain];

    let alpha = 1.0 - confidence_level;
    let lower_idx = ((alpha / 2.0) * n_bootstrap as f64).floor() as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * n_bootstrap as f64).ceil() as usize;

    let mut values_buffer = vec![0.0; n_bootstrap];
    for i in 0..n_explain {
        for f in 0..n_features {
            for t in 0..n_times {
                for (b, (shap, _)) in bootstrap_results.iter().enumerate() {
                    values_buffer[b] = shap[i][f][t];
                }

                let mean = values_buffer.iter().sum::<f64>() / n_bootstrap as f64;
                let variance = values_buffer
                    .iter()
                    .map(|v| (v - mean).powi(2))
                    .sum::<f64>()
                    / n_bootstrap as f64;

                values_buffer.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                shap_values_mean[i][f][t] = mean;
                shap_values_std[i][f][t] = variance.sqrt();
                shap_values_lower[i][f][t] = values_buffer[lower_idx.min(n_bootstrap - 1)];
                shap_values_upper[i][f][t] = values_buffer[upper_idx.min(n_bootstrap - 1)];
            }
        }
    }

    let base_value: Vec<f64> = (0..n_times)
        .map(|t| preds_bg_ref.iter().skip(t).step_by(n_times).sum::<f64>() / n_background as f64)
        .collect();

    Ok(BootstrapSurvShapResult {
        shap_values_mean,
        shap_values_std,
        shap_values_lower,
        shap_values_upper,
        base_value,
        time_points,
        n_bootstrap,
        confidence_level,
    })
}

#[pyfunction]
#[pyo3(signature = (
    predictions,
    time_points,
    times,
    events,
    n_samples,
    n_features,
    n_repeats=10,
    seed=None,
    parallel=true
))]
pub fn permutation_importance(
    predictions: Vec<f64>,
    time_points: Vec<f64>,
    times: Vec<f64>,
    events: Vec<i32>,
    n_samples: usize,
    n_features: usize,
    n_repeats: usize,
    seed: Option<u64>,
    parallel: bool,
) -> PyResult<PermutationImportanceResult> {
    let n_times = time_points.len();

    if predictions.len() != n_samples * n_times {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "predictions length must equal n_samples * n_times",
        ));
    }
    if times.len() != n_samples || events.len() != n_samples {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "times and events must have length n_samples",
        ));
    }

    let baseline_score =
        compute_concordance_index(&predictions, &times, &events, n_samples, n_times);

    let base_seed = seed.unwrap_or(42);

    let compute_feature_importance = |feature_idx: usize| -> (f64, f64) {
        let mut scores = Vec::with_capacity(n_repeats);

        for r in 0..n_repeats {
            let mut rng =
                fastrand::Rng::with_seed(base_seed + feature_idx as u64 * 1000 + r as u64);

            let mut perm_indices: Vec<usize> = (0..n_samples).collect();
            for i in (1..n_samples).rev() {
                let j = rng.usize(0..=i);
                perm_indices.swap(i, j);
            }

            let mut permuted_preds = predictions.clone();
            for (new_idx, &orig_idx) in perm_indices.iter().enumerate() {
                for t in 0..n_times {
                    permuted_preds[new_idx * n_times + t] = predictions[orig_idx * n_times + t];
                }
            }

            let score =
                compute_concordance_index(&permuted_preds, &times, &events, n_samples, n_times);
            scores.push(baseline_score - score);
        }

        let mean = scores.iter().sum::<f64>() / n_repeats as f64;
        let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n_repeats as f64;
        (mean, variance.sqrt())
    };

    let results: Vec<(f64, f64)> = if parallel {
        (0..n_features)
            .into_par_iter()
            .map(compute_feature_importance)
            .collect()
    } else {
        (0..n_features).map(compute_feature_importance).collect()
    };

    let importance: Vec<f64> = results.iter().map(|(m, _)| *m).collect();
    let importance_std: Vec<f64> = results.iter().map(|(_, s)| *s).collect();

    Ok(PermutationImportanceResult {
        importance,
        importance_std,
        baseline_score,
        n_repeats,
    })
}

fn compute_concordance_index(
    predictions: &[f64],
    times: &[f64],
    events: &[i32],
    n_samples: usize,
    n_times: usize,
) -> f64 {
    let risk_scores: Vec<f64> = (0..n_samples)
        .map(|i| {
            predictions[i * n_times..(i + 1) * n_times]
                .iter()
                .sum::<f64>()
                / n_times as f64
        })
        .collect();

    let mut concordant = 0.0;
    let mut discordant = 0.0;

    for i in 0..n_samples {
        if events[i] == 0 {
            continue;
        }
        for j in 0..n_samples {
            if i == j || times[j] < times[i] {
                continue;
            }
            if risk_scores[i] < risk_scores[j] {
                concordant += 1.0;
            } else if risk_scores[i] > risk_scores[j] {
                discordant += 1.0;
            } else {
                concordant += 0.5;
                discordant += 0.5;
            }
        }
    }

    let total = concordant + discordant;
    if total > 0.0 { concordant / total } else { 0.5 }
}

#[pyfunction]
#[pyo3(signature = (
    shap_values,
    time_points,
    n_features,
    aggregation_method=None
))]
pub fn compute_shap_interactions(
    shap_values: Vec<Vec<Vec<f64>>>,
    time_points: Vec<f64>,
    n_features: usize,
    aggregation_method: Option<AggregationMethod>,
) -> PyResult<ShapInteractionResult> {
    let n_samples = shap_values.len();
    let n_times = time_points.len();

    if n_samples == 0 || n_features == 0 || n_times == 0 {
        return Ok(ShapInteractionResult {
            interaction_values: vec![vec![vec![0.0; n_times]; n_features]; n_features],
            time_points,
            aggregated_interactions: None,
        });
    }

    let mut interaction_values = vec![vec![vec![0.0; n_times]; n_features]; n_features];

    let mut means = vec![vec![0.0; n_times]; n_features];
    for i in 0..n_features {
        for t in 0..n_times {
            means[i][t] = shap_values.iter().map(|s| s[i][t]).sum::<f64>() / n_samples as f64;
        }
    }

    for t in 0..n_times {
        for i in 0..n_features {
            for j in i..n_features {
                let mut covariance = 0.0;
                let mean_i = means[i][t];
                let mean_j = means[j][t];

                for sample in &shap_values {
                    covariance += (sample[i][t] - mean_i) * (sample[j][t] - mean_j);
                }
                covariance /= n_samples as f64;

                interaction_values[i][j][t] = covariance;
                if i != j {
                    interaction_values[j][i][t] = covariance;
                }
            }
        }
    }

    let aggregated_interactions = aggregation_method.map(|method| {
        let time_diffs: Vec<f64> = if n_times >= 2 {
            time_points.windows(2).map(|w| w[1] - w[0]).collect()
        } else {
            Vec::new()
        };
        let total_time = time_points.last().unwrap_or(&1.0) - time_points.first().unwrap_or(&0.0);
        let time_weights: Vec<f64> = if total_time > 0.0 {
            time_points
                .iter()
                .map(|&t| 1.0 - (t - time_points[0]) / total_time)
                .collect()
        } else {
            Vec::new()
        };

        let mut agg = vec![vec![0.0; n_features]; n_features];
        for i in 0..n_features {
            for j in 0..n_features {
                let values = &interaction_values[i][j];
                agg[i][j] = match method {
                    AggregationMethod::Mean => {
                        values.iter().map(|v| v.abs()).sum::<f64>() / n_times as f64
                    }
                    AggregationMethod::MaxAbsolute => {
                        values.iter().map(|v| v.abs()).fold(0.0, f64::max)
                    }
                    AggregationMethod::Integral => {
                        if n_times < 2 {
                            values.first().copied().unwrap_or(0.0).abs()
                        } else {
                            let mut integral = 0.0;
                            for k in 0..time_diffs.len() {
                                let avg = (values[k + 1].abs() + values[k].abs()) / 2.0;
                                integral += avg * time_diffs[k];
                            }
                            integral
                        }
                    }
                    AggregationMethod::TimeWeighted => {
                        if total_time <= 0.0 {
                            values.iter().map(|v| v.abs()).sum::<f64>() / n_times as f64
                        } else {
                            let mut weighted_sum = 0.0;
                            for (k, &weight) in time_weights.iter().enumerate() {
                                weighted_sum += values[k].abs() * weight;
                            }
                            weighted_sum / n_times as f64
                        }
                    }
                };
            }
        }
        agg
    });

    Ok(ShapInteractionResult {
        interaction_values,
        time_points,
        aggregated_interactions,
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

    let time_diffs: Vec<f64> = if n_times >= 2 {
        time_points.windows(2).map(|w| w[1] - w[0]).collect()
    } else {
        Vec::new()
    };

    let total_time = time_points.last().unwrap_or(&1.0) - time_points.first().unwrap_or(&0.0);
    let time_weights: Vec<f64> = if total_time > 0.0 {
        time_points
            .iter()
            .map(|&t| 1.0 - (t - time_points[0]) / total_time)
            .collect()
    } else {
        Vec::new()
    };

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
                        for i in 0..time_diffs.len() {
                            let avg = (shap_t[i + 1].abs() + shap_t[i].abs()) / 2.0;
                            integral += avg * time_diffs[i];
                        }
                        integral
                    }
                }
                AggregationMethod::TimeWeighted => {
                    if total_time <= 0.0 {
                        shap_t.iter().map(|v| v.abs()).sum::<f64>() / n_times as f64
                    } else {
                        let mut weighted_sum = 0.0;
                        for (i, &weight) in time_weights.iter().enumerate() {
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

        for (t, &pred) in predictions_explain.iter().enumerate().take(n_times) {
            let shap_sum: f64 = (0..n_features).map(|f| result.shap_values[0][f][t]).sum();
            let reconstructed = result.base_value[t] + shap_sum;
            let error = (reconstructed - pred).abs();
            assert!(
                error < 0.5,
                "Additivity check failed at t={}: reconstructed={}, actual={}, error={}",
                t,
                reconstructed,
                pred,
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

    #[test]
    fn test_feature_ranking() {
        let n_explain = 3;
        let n_features = 4;
        let n_times = 3;

        let mut shap_values = vec![vec![vec![0.0; n_times]; n_features]; n_explain];
        for sample in shap_values.iter_mut() {
            for (f, feature) in sample.iter_mut().enumerate() {
                for (t, val) in feature.iter_mut().enumerate() {
                    *val = (f + 1) as f64 * 0.1 + (t as f64) * 0.01;
                }
            }
        }

        let result = SurvShapResult {
            shap_values,
            base_value: vec![0.5; n_times],
            time_points: vec![1.0, 2.0, 3.0],
            aggregated_importance: None,
        };

        let ranking = result
            .feature_ranking(AggregationMethod::Mean, Some(2))
            .unwrap();
        assert_eq!(ranking.len(), 2);
        assert_eq!(ranking[0].feature_idx, 3);
        assert_eq!(ranking[1].feature_idx, 2);
    }

    #[test]
    fn test_bootstrap_survshap() {
        let n_explain = 2;
        let n_background = 5;
        let n_features = 2;
        let n_times = 3;

        let x_explain: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4];
        let x_background: Vec<f64> = (0..(n_background * n_features))
            .map(|i| (i as f64) * 0.1)
            .collect();

        let predictions_explain: Vec<f64> = vec![0.9, 0.8, 0.7, 0.85, 0.75, 0.65];
        let predictions_background: Vec<f64> = (0..(n_background * n_times))
            .map(|i| 0.9 - (i as f64) * 0.02)
            .collect();

        let time_points: Vec<f64> = vec![1.0, 2.0, 3.0];

        let config = SurvShapConfig::new(50, n_background, Some(42), false).unwrap();

        let result = survshap_bootstrap(
            x_explain,
            x_background,
            predictions_explain,
            predictions_background,
            time_points,
            n_explain,
            n_background,
            n_features,
            10,
            0.95,
            Some(&config),
        )
        .unwrap();

        assert_eq!(result.shap_values_mean.len(), n_explain);
        assert_eq!(result.shap_values_std.len(), n_explain);
        assert_eq!(result.shap_values_lower.len(), n_explain);
        assert_eq!(result.shap_values_upper.len(), n_explain);
        assert_eq!(result.n_bootstrap, 10);
        assert!((result.confidence_level - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_shap_interactions() {
        let shap_values = vec![
            vec![vec![0.1, 0.2, 0.3], vec![0.2, 0.3, 0.4]],
            vec![vec![0.15, 0.25, 0.35], vec![0.25, 0.35, 0.45]],
        ];
        let time_points = vec![1.0, 2.0, 3.0];

        let result =
            compute_shap_interactions(shap_values, time_points, 2, Some(AggregationMethod::Mean))
                .unwrap();

        assert_eq!(result.interaction_values.len(), 2);
        assert_eq!(result.interaction_values[0].len(), 2);
        assert_eq!(result.interaction_values[0][0].len(), 3);
        assert!(result.aggregated_interactions.is_some());

        let top = result.top_interactions(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_concordance_index() {
        let predictions = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];
        let times = vec![10.0, 5.0, 15.0];
        let events = vec![1, 1, 0];

        let c_index = compute_concordance_index(&predictions, &times, &events, 3, 3);
        assert!((0.0..=1.0).contains(&c_index));
    }
}
