#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Debug, Clone)]
#[pyclass]
pub struct MCDropoutConfig {
    #[pyo3(get, set)]
    pub n_samples: usize,
    #[pyo3(get, set)]
    pub dropout_rate: f64,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl MCDropoutConfig {
    #[new]
    #[pyo3(signature = (n_samples=100, dropout_rate=0.1, seed=None))]
    pub fn new(n_samples: usize, dropout_rate: f64, seed: Option<u64>) -> PyResult<Self> {
        if n_samples == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "n_samples must be positive",
            ));
        }
        if !(0.0..=1.0).contains(&dropout_rate) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "dropout_rate must be between 0 and 1",
            ));
        }
        Ok(Self {
            n_samples,
            dropout_rate,
            seed,
        })
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct UncertaintyResult {
    #[pyo3(get)]
    pub mean_prediction: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub std_prediction: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub lower_ci: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub upper_ci: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub epistemic_uncertainty: Vec<f64>,
    #[pyo3(get)]
    pub aleatoric_uncertainty: Vec<f64>,
}

#[pymethods]
impl UncertaintyResult {
    fn __repr__(&self) -> String {
        format!(
            "UncertaintyResult(n_samples={}, n_times={})",
            self.mean_prediction.len(),
            self.mean_prediction.first().map(|p| p.len()).unwrap_or(0)
        )
    }

    fn total_uncertainty(&self) -> Vec<f64> {
        self.epistemic_uncertainty
            .iter()
            .zip(self.aleatoric_uncertainty.iter())
            .map(|(&e, &a)| (e.powi(2) + a.powi(2)).sqrt())
            .collect()
    }
}

#[allow(dead_code)]
fn apply_dropout(values: &[f64], dropout_rate: f64, rng: &mut fastrand::Rng) -> Vec<f64> {
    let scale = 1.0 / (1.0 - dropout_rate);
    values
        .iter()
        .map(|&v| {
            if rng.f64() < dropout_rate {
                0.0
            } else {
                v * scale
            }
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature = (
    predictions,
    config=None
))]
pub fn mc_dropout_uncertainty(
    predictions: Vec<Vec<Vec<f64>>>,
    config: Option<MCDropoutConfig>,
) -> PyResult<UncertaintyResult> {
    let _config = config.unwrap_or_else(|| MCDropoutConfig::new(100, 0.1, None).unwrap());

    if predictions.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "predictions must not be empty",
        ));
    }

    let n_samples = predictions.len();
    let n_times = predictions[0].first().map(|p| p.len()).unwrap_or(0);
    let n_obs = predictions[0].len();

    let mean_prediction: Vec<Vec<f64>> = (0..n_obs)
        .into_par_iter()
        .map(|i| {
            (0..n_times)
                .map(|t| predictions.iter().map(|p| p[i][t]).sum::<f64>() / n_samples as f64)
                .collect()
        })
        .collect();

    let std_prediction: Vec<Vec<f64>> = (0..n_obs)
        .into_par_iter()
        .map(|i| {
            (0..n_times)
                .map(|t| {
                    let mean = mean_prediction[i][t];
                    let var: f64 = predictions
                        .iter()
                        .map(|p| (p[i][t] - mean).powi(2))
                        .sum::<f64>()
                        / n_samples as f64;
                    var.sqrt()
                })
                .collect()
        })
        .collect();

    let lower_ci: Vec<Vec<f64>> = mean_prediction
        .iter()
        .zip(std_prediction.iter())
        .map(|(m, s)| {
            m.iter()
                .zip(s.iter())
                .map(|(&mi, &si)| (mi - 1.96 * si).clamp(0.0, 1.0))
                .collect()
        })
        .collect();

    let upper_ci: Vec<Vec<f64>> = mean_prediction
        .iter()
        .zip(std_prediction.iter())
        .map(|(m, s)| {
            m.iter()
                .zip(s.iter())
                .map(|(&mi, &si)| (mi + 1.96 * si).clamp(0.0, 1.0))
                .collect()
        })
        .collect();

    let epistemic_uncertainty: Vec<f64> = std_prediction
        .iter()
        .map(|s| s.iter().sum::<f64>() / s.len() as f64)
        .collect();

    let aleatoric_uncertainty: Vec<f64> = mean_prediction
        .iter()
        .map(|m| {
            let var: f64 = m.iter().map(|&mi| mi * (1.0 - mi)).sum::<f64>() / m.len() as f64;
            var.sqrt()
        })
        .collect();

    Ok(UncertaintyResult {
        mean_prediction,
        std_prediction,
        lower_ci,
        upper_ci,
        epistemic_uncertainty,
        aleatoric_uncertainty,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct EnsembleUncertaintyResult {
    #[pyo3(get)]
    pub mean_prediction: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub std_prediction: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub model_disagreement: Vec<f64>,
    #[pyo3(get)]
    pub prediction_intervals: Vec<Vec<(f64, f64)>>,
}

#[pymethods]
impl EnsembleUncertaintyResult {
    fn __repr__(&self) -> String {
        format!(
            "EnsembleUncertaintyResult(n_samples={})",
            self.mean_prediction.len()
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    model_predictions,
    confidence_level=0.95
))]
pub fn ensemble_uncertainty(
    model_predictions: Vec<Vec<Vec<f64>>>,
    confidence_level: f64,
) -> PyResult<EnsembleUncertaintyResult> {
    if model_predictions.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "model_predictions must not be empty",
        ));
    }

    let n_models = model_predictions.len();
    let n_obs = model_predictions[0].len();
    let n_times = model_predictions[0].first().map(|p| p.len()).unwrap_or(0);

    let mean_prediction: Vec<Vec<f64>> = (0..n_obs)
        .into_par_iter()
        .map(|i| {
            (0..n_times)
                .map(|t| model_predictions.iter().map(|p| p[i][t]).sum::<f64>() / n_models as f64)
                .collect()
        })
        .collect();

    let std_prediction: Vec<Vec<f64>> = (0..n_obs)
        .into_par_iter()
        .map(|i| {
            (0..n_times)
                .map(|t| {
                    let mean = mean_prediction[i][t];
                    let var: f64 = model_predictions
                        .iter()
                        .map(|p| (p[i][t] - mean).powi(2))
                        .sum::<f64>()
                        / n_models as f64;
                    var.sqrt()
                })
                .collect()
        })
        .collect();

    #[allow(clippy::needless_range_loop)]
    let model_disagreement: Vec<f64> = (0..n_obs)
        .into_par_iter()
        .map(|i| {
            let mut total_disagreement = 0.0;
            for t_idx in 0..n_times {
                for m1 in 0..n_models {
                    for m2 in (m1 + 1)..n_models {
                        total_disagreement += (model_predictions[m1][i][t_idx]
                            - model_predictions[m2][i][t_idx])
                            .abs();
                    }
                }
            }
            let n_pairs = (n_models * (n_models - 1) / 2) as f64;
            total_disagreement / (n_pairs * n_times as f64)
        })
        .collect();

    let z = 1.96 * (1.0 + (1.0 - confidence_level).ln().abs()).sqrt();

    let prediction_intervals: Vec<Vec<(f64, f64)>> = mean_prediction
        .iter()
        .zip(std_prediction.iter())
        .map(|(m, s)| {
            m.iter()
                .zip(s.iter())
                .map(|(&mi, &si)| ((mi - z * si).clamp(0.0, 1.0), (mi + z * si).clamp(0.0, 1.0)))
                .collect()
        })
        .collect();

    Ok(EnsembleUncertaintyResult {
        mean_prediction,
        std_prediction,
        model_disagreement,
        prediction_intervals,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct QuantileRegressionResult {
    #[pyo3(get)]
    pub median: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub lower_quantile: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub upper_quantile: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub quantiles: Vec<f64>,
}

#[pymethods]
impl QuantileRegressionResult {
    fn __repr__(&self) -> String {
        format!("QuantileRegressionResult(quantiles={:?})", self.quantiles)
    }

    fn prediction_interval_width(&self) -> Vec<Vec<f64>> {
        self.upper_quantile
            .iter()
            .zip(self.lower_quantile.iter())
            .map(|(u, l)| u.iter().zip(l.iter()).map(|(&ui, &li)| ui - li).collect())
            .collect()
    }
}

fn compute_quantile(values: &mut [f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = (q * (values.len() - 1) as f64).round() as usize;
    values[idx.min(values.len() - 1)]
}

#[pyfunction]
#[pyo3(signature = (
    bootstrap_predictions,
    quantiles=None
))]
pub fn quantile_regression_intervals(
    bootstrap_predictions: Vec<Vec<Vec<f64>>>,
    quantiles: Option<Vec<f64>>,
) -> PyResult<QuantileRegressionResult> {
    let quantiles = quantiles.unwrap_or_else(|| vec![0.025, 0.5, 0.975]);

    if bootstrap_predictions.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "bootstrap_predictions must not be empty",
        ));
    }

    let _n_bootstrap = bootstrap_predictions.len();
    let n_obs = bootstrap_predictions[0].len();
    let n_times = bootstrap_predictions[0]
        .first()
        .map(|p| p.len())
        .unwrap_or(0);

    let lower_q = quantiles.first().copied().unwrap_or(0.025);
    let median_q = quantiles.get(1).copied().unwrap_or(0.5);
    let upper_q = quantiles.last().copied().unwrap_or(0.975);

    #[allow(clippy::type_complexity)]
    let (lower_quantile, median, upper_quantile): (
        Vec<Vec<f64>>,
        Vec<Vec<f64>>,
        Vec<Vec<f64>>,
    ) = (0..n_obs)
        .into_par_iter()
        .map(|i| {
            let mut lower = vec![0.0; n_times];
            let mut med = vec![0.0; n_times];
            let mut upper = vec![0.0; n_times];

            for t in 0..n_times {
                let mut values: Vec<f64> = bootstrap_predictions.iter().map(|p| p[i][t]).collect();
                lower[t] = compute_quantile(&mut values.clone(), lower_q);
                med[t] = compute_quantile(&mut values.clone(), median_q);
                upper[t] = compute_quantile(&mut values, upper_q);
            }

            (lower, med, upper)
        })
        .fold(
            || (Vec::new(), Vec::new(), Vec::new()),
            |mut acc, (l, m, u)| {
                acc.0.push(l);
                acc.1.push(m);
                acc.2.push(u);
                acc
            },
        )
        .reduce(
            || (Vec::new(), Vec::new(), Vec::new()),
            |mut a, b| {
                a.0.extend(b.0);
                a.1.extend(b.1);
                a.2.extend(b.2);
                a
            },
        );

    Ok(QuantileRegressionResult {
        median,
        lower_quantile,
        upper_quantile,
        quantiles,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct CalibrationUncertaintyResult {
    #[pyo3(get)]
    pub expected_coverage: f64,
    #[pyo3(get)]
    pub observed_coverage: f64,
    #[pyo3(get)]
    pub calibration_error: f64,
    #[pyo3(get)]
    pub sharpness: f64,
}

#[pymethods]
impl CalibrationUncertaintyResult {
    fn __repr__(&self) -> String {
        format!(
            "CalibrationUncertaintyResult(expected={:.3}, observed={:.3}, error={:.3})",
            self.expected_coverage, self.observed_coverage, self.calibration_error
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    true_times,
    true_events,
    lower_bounds,
    upper_bounds,
    expected_coverage=0.95
))]
pub fn calibrate_prediction_intervals(
    true_times: Vec<f64>,
    true_events: Vec<i32>,
    lower_bounds: Vec<f64>,
    upper_bounds: Vec<f64>,
    expected_coverage: f64,
) -> PyResult<CalibrationUncertaintyResult> {
    let n = true_times.len();
    if n == 0 || true_events.len() != n || lower_bounds.len() != n || upper_bounds.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All inputs must have the same non-zero length",
        ));
    }

    let mut covered = 0;
    let mut total_width = 0.0;

    for i in 0..n {
        if true_events[i] == 1
            && true_times[i] >= lower_bounds[i]
            && true_times[i] <= upper_bounds[i]
        {
            covered += 1;
        }
        total_width += upper_bounds[i] - lower_bounds[i];
    }

    let n_events = true_events.iter().filter(|&&e| e == 1).count();
    let observed_coverage = if n_events > 0 {
        covered as f64 / n_events as f64
    } else {
        0.0
    };

    let calibration_error = (observed_coverage - expected_coverage).abs();
    let sharpness = total_width / n as f64;

    Ok(CalibrationUncertaintyResult {
        expected_coverage,
        observed_coverage,
        calibration_error,
        sharpness,
    })
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct ConformalSurvivalConfig {
    #[pyo3(get, set)]
    pub alpha: f64,
    #[pyo3(get, set)]
    pub method: String,
    #[pyo3(get, set)]
    pub n_bootstrap: usize,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl ConformalSurvivalConfig {
    #[new]
    #[pyo3(signature = (alpha=0.1, method="cqr".to_string(), n_bootstrap=100, seed=None))]
    pub fn new(alpha: f64, method: String, n_bootstrap: usize, seed: Option<u64>) -> Self {
        Self {
            alpha,
            method,
            n_bootstrap,
            seed,
        }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct ConformalSurvivalResult {
    #[pyo3(get)]
    pub lower_bounds: Vec<f64>,
    #[pyo3(get)]
    pub upper_bounds: Vec<f64>,
    #[pyo3(get)]
    pub point_predictions: Vec<f64>,
    #[pyo3(get)]
    pub coverage: f64,
    #[pyo3(get)]
    pub interval_widths: Vec<f64>,
    #[pyo3(get)]
    pub calibration_scores: Vec<f64>,
}

#[pymethods]
impl ConformalSurvivalResult {
    #[new]
    pub fn new(
        lower_bounds: Vec<f64>,
        upper_bounds: Vec<f64>,
        point_predictions: Vec<f64>,
        coverage: f64,
        interval_widths: Vec<f64>,
        calibration_scores: Vec<f64>,
    ) -> Self {
        Self {
            lower_bounds,
            upper_bounds,
            point_predictions,
            coverage,
            interval_widths,
            calibration_scores,
        }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct BayesianBootstrapConfig {
    #[pyo3(get, set)]
    pub n_bootstrap: usize,
    #[pyo3(get, set)]
    pub confidence_level: f64,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl BayesianBootstrapConfig {
    #[new]
    #[pyo3(signature = (n_bootstrap=1000, confidence_level=0.95, seed=None))]
    pub fn new(n_bootstrap: usize, confidence_level: f64, seed: Option<u64>) -> Self {
        Self {
            n_bootstrap,
            confidence_level,
            seed,
        }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct BayesianBootstrapResult {
    #[pyo3(get)]
    pub mean_survival: Vec<f64>,
    #[pyo3(get)]
    pub lower_ci: Vec<f64>,
    #[pyo3(get)]
    pub upper_ci: Vec<f64>,
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub posterior_samples: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub credible_bands: HashMap<String, Vec<f64>>,
}

#[pymethods]
impl BayesianBootstrapResult {
    #[new]
    pub fn new(
        mean_survival: Vec<f64>,
        lower_ci: Vec<f64>,
        upper_ci: Vec<f64>,
        time_points: Vec<f64>,
        posterior_samples: Vec<Vec<f64>>,
        credible_bands: HashMap<String, Vec<f64>>,
    ) -> Self {
        Self {
            mean_survival,
            lower_ci,
            upper_ci,
            time_points,
            posterior_samples,
            credible_bands,
        }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct JackknifePlusConfig {
    #[pyo3(get, set)]
    pub alpha: f64,
    #[pyo3(get, set)]
    pub plus_variant: bool,
    #[pyo3(get, set)]
    pub cv_folds: usize,
}

#[pymethods]
impl JackknifePlusConfig {
    #[new]
    #[pyo3(signature = (alpha=0.1, plus_variant=true, cv_folds=5))]
    pub fn new(alpha: f64, plus_variant: bool, cv_folds: usize) -> Self {
        Self {
            alpha,
            plus_variant,
            cv_folds,
        }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct JackknifePlusResult {
    #[pyo3(get)]
    pub lower_bounds: Vec<f64>,
    #[pyo3(get)]
    pub upper_bounds: Vec<f64>,
    #[pyo3(get)]
    pub point_predictions: Vec<f64>,
    #[pyo3(get)]
    pub coverage: f64,
    #[pyo3(get)]
    pub residuals: Vec<f64>,
}

#[pymethods]
impl JackknifePlusResult {
    #[new]
    pub fn new(
        lower_bounds: Vec<f64>,
        upper_bounds: Vec<f64>,
        point_predictions: Vec<f64>,
        coverage: f64,
        residuals: Vec<f64>,
    ) -> Self {
        Self {
            lower_bounds,
            upper_bounds,
            point_predictions,
            coverage,
            residuals,
        }
    }
}

fn compute_conformity_scores(
    time: &[f64],
    event: &[i32],
    predictions: &[f64],
    method: &str,
) -> Vec<f64> {
    match method {
        "cqr" => time
            .iter()
            .zip(predictions.iter())
            .map(|(t, p)| (t - p).abs())
            .collect(),
        "weighted" => time
            .iter()
            .zip(event.iter())
            .zip(predictions.iter())
            .map(|((t, e), p)| {
                let weight = if *e == 1 { 1.0 } else { 0.5 };
                weight * (t - p).abs()
            })
            .collect(),
        "censoring_adjusted" => {
            let max_time = time.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            time.iter()
                .zip(event.iter())
                .zip(predictions.iter())
                .map(|((t, e), p)| {
                    if *e == 1 {
                        (t - p).abs()
                    } else {
                        ((max_time - t) / max_time) * (t - p).abs()
                    }
                })
                .collect()
        }
        _ => time
            .iter()
            .zip(predictions.iter())
            .map(|(t, p)| (t - p).abs())
            .collect(),
    }
}

fn compute_coverage(lower: &[f64], upper: &[f64], actual: &[f64]) -> f64 {
    if actual.is_empty() || lower.len() != upper.len() {
        return 0.0;
    }

    let n = lower.len().min(actual.len());
    let covered = (0..n)
        .filter(|&i| actual[i] >= lower[i] && actual[i] <= upper[i])
        .count();

    covered as f64 / n as f64
}

#[pyfunction]
#[pyo3(signature = (
    cal_time,
    cal_event,
    cal_predictions,
    test_predictions,
    config
))]
pub fn conformal_survival(
    cal_time: Vec<f64>,
    cal_event: Vec<i32>,
    cal_predictions: Vec<f64>,
    test_predictions: Vec<f64>,
    config: ConformalSurvivalConfig,
) -> PyResult<ConformalSurvivalResult> {
    let n_cal = cal_time.len();
    let n_test = test_predictions.len();

    if n_cal == 0 || n_test == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Calibration and test sets must be non-empty",
        ));
    }

    let scores = compute_conformity_scores(&cal_time, &cal_event, &cal_predictions, &config.method);

    let quantile_idx = ((n_cal as f64 + 1.0) * (1.0 - config.alpha)).ceil() as usize;
    let quantile_idx = quantile_idx.min(scores.len()).saturating_sub(1);

    let mut sorted_scores = scores.clone();
    sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = sorted_scores[quantile_idx];

    let mut lower_bounds = Vec::with_capacity(n_test);
    let mut upper_bounds = Vec::with_capacity(n_test);
    let mut interval_widths = Vec::with_capacity(n_test);

    for pred in &test_predictions {
        let lower = (pred - q).max(0.0);
        let upper = pred + q;

        interval_widths.push(upper - lower);
        lower_bounds.push(lower);
        upper_bounds.push(upper);
    }

    let coverage = compute_coverage(&lower_bounds, &upper_bounds, &cal_time);

    Ok(ConformalSurvivalResult {
        lower_bounds,
        upper_bounds,
        point_predictions: test_predictions,
        coverage,
        interval_widths,
        calibration_scores: scores,
    })
}

fn generate_dirichlet_weights(n: usize, rng: &mut fastrand::Rng) -> Vec<f64> {
    let mut weights: Vec<f64> = (0..n)
        .map(|_| {
            let u: f64 = rng.f64().max(1e-10);
            -u.ln()
        })
        .collect();

    let sum: f64 = weights.iter().sum();
    for w in &mut weights {
        *w /= sum;
    }

    weights
}

fn weighted_kaplan_meier(
    time: &[f64],
    event: &[i32],
    weights: &[f64],
    eval_times: &[f64],
) -> Vec<f64> {
    let mut indices: Vec<usize> = (0..time.len()).collect();
    indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut survival = vec![1.0; eval_times.len()];
    let mut surv_prob = 1.0;
    let mut at_risk_weight: f64 = weights.iter().sum();
    let mut last_time = 0.0;

    for &idx in &indices {
        let t = time[idx];
        let e = event[idx];
        let w = weights[idx];

        if t > last_time && at_risk_weight > 0.0 && e == 1 {
            surv_prob *= 1.0 - w / at_risk_weight;
        }

        for (i, &eval_t) in eval_times.iter().enumerate() {
            if t > eval_t && last_time <= eval_t {
                survival[i] = surv_prob;
            }
        }

        at_risk_weight -= w;
        last_time = t;
    }

    for (i, &eval_t) in eval_times.iter().enumerate() {
        if eval_t >= last_time {
            survival[i] = surv_prob;
        }
    }

    survival
}

fn compute_quantile_posterior(samples: &[Vec<f64>], q: f64) -> Vec<f64> {
    if samples.is_empty() || samples[0].is_empty() {
        return vec![];
    }

    let n_times = samples[0].len();
    let mut result = vec![0.0; n_times];

    for t in 0..n_times {
        let mut values: Vec<f64> = samples.iter().map(|s| s[t]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = (values.len() as f64 * q).floor() as usize;
        result[t] = values[idx.min(values.len() - 1)];
    }

    result
}

#[pyfunction]
#[pyo3(signature = (
    time,
    event,
    eval_times,
    config
))]
pub fn bayesian_bootstrap_survival(
    time: Vec<f64>,
    event: Vec<i32>,
    eval_times: Vec<f64>,
    config: BayesianBootstrapConfig,
) -> PyResult<BayesianBootstrapResult> {
    let n = time.len();

    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Time vector must be non-empty",
        ));
    }

    let mut rng = if let Some(seed) = config.seed {
        fastrand::Rng::with_seed(seed)
    } else {
        fastrand::Rng::new()
    };

    let n_times = eval_times.len();
    let mut posterior_samples: Vec<Vec<f64>> = vec![vec![0.0; n_times]; config.n_bootstrap];

    for sample in posterior_samples.iter_mut() {
        let weights = generate_dirichlet_weights(n, &mut rng);
        let survival = weighted_kaplan_meier(&time, &event, &weights, &eval_times);
        *sample = survival;
    }

    let mean_survival = (0..n_times)
        .map(|t| posterior_samples.iter().map(|s| s[t]).sum::<f64>() / config.n_bootstrap as f64)
        .collect::<Vec<_>>();

    let alpha = 1.0 - config.confidence_level;
    let lower_idx = (config.n_bootstrap as f64 * (alpha / 2.0)).floor() as usize;
    let upper_idx = (config.n_bootstrap as f64 * (1.0 - alpha / 2.0)).ceil() as usize;

    let mut lower_ci = vec![0.0; n_times];
    let mut upper_ci = vec![0.0; n_times];

    for t in 0..n_times {
        let mut values: Vec<f64> = posterior_samples.iter().map(|s| s[t]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        lower_ci[t] = values[lower_idx.min(values.len() - 1)];
        upper_ci[t] = values[upper_idx.min(values.len() - 1)];
    }

    let mut credible_bands = HashMap::new();
    credible_bands.insert(
        "50%_lower".to_string(),
        compute_quantile_posterior(&posterior_samples, 0.25),
    );
    credible_bands.insert(
        "50%_upper".to_string(),
        compute_quantile_posterior(&posterior_samples, 0.75),
    );
    credible_bands.insert(
        "90%_lower".to_string(),
        compute_quantile_posterior(&posterior_samples, 0.05),
    );
    credible_bands.insert(
        "90%_upper".to_string(),
        compute_quantile_posterior(&posterior_samples, 0.95),
    );

    Ok(BayesianBootstrapResult {
        mean_survival,
        lower_ci,
        upper_ci,
        time_points: eval_times,
        posterior_samples,
        credible_bands,
    })
}

fn simple_cox_predictions(time: &[f64], event: &[i32], covariates: &[Vec<f64>]) -> Vec<f64> {
    if time.is_empty() {
        return vec![];
    }

    let n = time.len();
    let p = if !covariates.is_empty() && !covariates[0].is_empty() {
        covariates[0].len()
    } else {
        0
    };

    if p == 0 {
        let mean_time: f64 = time.iter().sum::<f64>() / n as f64;
        return vec![mean_time; n];
    }

    let beta = estimate_cox_coefficients(time, event, covariates);

    let linear_pred: Vec<f64> = covariates
        .iter()
        .map(|cov| cov.iter().zip(beta.iter()).map(|(x, b)| x * b).sum::<f64>())
        .collect();

    let mean_lp = linear_pred.iter().sum::<f64>() / n as f64;
    let baseline_mean = time
        .iter()
        .zip(event.iter())
        .filter(|(_, e)| **e == 1)
        .map(|(t, _)| t)
        .sum::<f64>()
        / event.iter().filter(|&&e| e == 1).count().max(1) as f64;

    linear_pred
        .iter()
        .map(|lp| baseline_mean * (-lp + mean_lp).exp())
        .collect()
}

fn estimate_cox_coefficients(time: &[f64], event: &[i32], covariates: &[Vec<f64>]) -> Vec<f64> {
    let n = time.len();
    let p = covariates[0].len();
    let mut beta: Vec<f64> = vec![0.0; p];

    let learning_rate = 0.01;
    let max_iter = 100;

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[b]
            .partial_cmp(&time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for _ in 0..max_iter {
        let mut gradient: Vec<f64> = vec![0.0; p];

        let linear_pred: Vec<f64> = covariates
            .iter()
            .map(|cov| cov.iter().zip(beta.iter()).map(|(x, b)| x * b).sum::<f64>())
            .collect();

        let exp_lp: Vec<f64> = linear_pred.iter().map(|lp| lp.exp()).collect();

        let mut risk_sum = 0.0;
        let mut weighted_cov_sum: Vec<f64> = vec![0.0; p];

        for &i in indices.iter() {
            risk_sum += exp_lp[i];
            for k in 0..p {
                weighted_cov_sum[k] += covariates[i][k] * exp_lp[i];
            }

            if event[i] == 1 {
                for k in 0..p {
                    gradient[k] += covariates[i][k] - weighted_cov_sum[k] / risk_sum;
                }
            }
        }

        for k in 0..p {
            beta[k] += learning_rate * gradient[k];
        }
    }

    beta
}

fn compute_kernel_weights(target: &[f64], reference: &[Vec<f64>]) -> Vec<f64> {
    let bandwidth = 1.0;

    reference
        .iter()
        .map(|ref_cov| {
            let dist_sq: f64 = target
                .iter()
                .zip(ref_cov.iter())
                .map(|(t, r)| (t - r).powi(2))
                .sum();
            (-dist_sq / (2.0 * bandwidth * bandwidth)).exp()
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature = (
    time,
    event,
    covariates,
    config
))]
pub fn jackknife_plus_survival(
    time: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<Vec<f64>>,
    config: JackknifePlusConfig,
) -> PyResult<JackknifePlusResult> {
    let n = time.len();

    if n < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Need at least 2 observations for jackknife",
        ));
    }

    let full_predictions = simple_cox_predictions(&time, &event, &covariates);

    let mut loo_residuals = vec![0.0; n];

    for i in 0..n {
        let loo_time: Vec<f64> = time
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, t)| *t)
            .collect();
        let loo_event: Vec<i32> = event
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, e)| *e)
            .collect();
        let loo_cov: Vec<Vec<f64>> = covariates
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, c)| c.clone())
            .collect();

        let loo_predictions = simple_cox_predictions(&loo_time, &loo_event, &loo_cov);

        let pred_i = if !loo_predictions.is_empty() {
            let weights = compute_kernel_weights(&covariates[i], &loo_cov);
            let weighted_sum: f64 = loo_predictions
                .iter()
                .zip(weights.iter())
                .map(|(p, w)| p * w)
                .sum();
            let weight_sum: f64 = weights.iter().sum();
            if weight_sum > 0.0 {
                weighted_sum / weight_sum
            } else {
                loo_predictions[0]
            }
        } else {
            full_predictions[i]
        };

        loo_residuals[i] = (time[i] - pred_i).abs();
    }

    let quantile_idx = if config.plus_variant {
        ((n as f64 + 1.0) * (1.0 - config.alpha)).ceil() as usize
    } else {
        ((n as f64) * (1.0 - config.alpha)).ceil() as usize
    };

    let mut sorted_residuals = loo_residuals.clone();
    sorted_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = sorted_residuals[quantile_idx.min(n - 1)];

    let lower_bounds: Vec<f64> = full_predictions.iter().map(|p| (p - q).max(0.0)).collect();
    let upper_bounds: Vec<f64> = full_predictions.iter().map(|p| p + q).collect();

    let coverage = compute_coverage(&lower_bounds, &upper_bounds, &time);

    Ok(JackknifePlusResult {
        lower_bounds,
        upper_bounds,
        point_predictions: full_predictions,
        coverage,
        residuals: loo_residuals,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mc_dropout_config() {
        let config = MCDropoutConfig::new(100, 0.1, None).unwrap();
        assert_eq!(config.n_samples, 100);
    }

    #[test]
    fn test_mc_dropout_config_validation() {
        let result = MCDropoutConfig::new(0, 0.1, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_quantile() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(compute_quantile(&mut values, 0.5), 3.0);
    }

    #[test]
    fn test_apply_dropout() {
        let mut rng = fastrand::Rng::new();
        rng.seed(42);
        let values = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let dropped = apply_dropout(&values, 0.5, &mut rng);
        assert_eq!(dropped.len(), 5);
    }

    #[test]
    fn test_conformal_survival_basic() {
        let cal_time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let cal_event = vec![1, 1, 1, 0, 1, 1, 0, 1, 1, 0];
        let cal_predictions = vec![1.5, 2.2, 2.8, 4.5, 4.8, 5.5, 7.5, 7.8, 8.5, 10.5];
        let test_predictions = vec![3.0, 6.0, 9.0];

        let config = ConformalSurvivalConfig::new(0.1, "cqr".to_string(), 100, Some(42));
        let result = conformal_survival(
            cal_time,
            cal_event,
            cal_predictions,
            test_predictions,
            config,
        )
        .unwrap();

        assert_eq!(result.lower_bounds.len(), 3);
        assert_eq!(result.upper_bounds.len(), 3);

        for i in 0..3 {
            assert!(result.lower_bounds[i] <= result.point_predictions[i]);
            assert!(result.upper_bounds[i] >= result.point_predictions[i]);
        }
    }

    #[test]
    fn test_bayesian_bootstrap_survival() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let event = vec![1, 1, 0, 1, 0];
        let eval_times = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let config = BayesianBootstrapConfig::new(100, 0.95, Some(42));
        let result = bayesian_bootstrap_survival(time, event, eval_times, config).unwrap();

        assert_eq!(result.mean_survival.len(), 5);
        assert_eq!(result.lower_ci.len(), 5);
        assert_eq!(result.upper_ci.len(), 5);

        for i in 0..5 {
            assert!(result.lower_ci[i] <= result.mean_survival[i]);
            assert!(result.upper_ci[i] >= result.mean_survival[i]);
            assert!(result.mean_survival[i] >= 0.0 && result.mean_survival[i] <= 1.0);
        }
    }

    #[test]
    fn test_jackknife_plus_survival() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let event = vec![1, 1, 1, 0, 1, 1, 0, 1, 1, 0];
        let covariates = vec![
            vec![0.5, 1.0],
            vec![1.0, 1.2],
            vec![1.5, 1.4],
            vec![2.0, 1.6],
            vec![2.5, 1.8],
            vec![3.0, 2.0],
            vec![3.5, 2.2],
            vec![4.0, 2.4],
            vec![4.5, 2.6],
            vec![5.0, 2.8],
        ];

        let config = JackknifePlusConfig::new(0.1, true, 5);
        let result = jackknife_plus_survival(time, event, covariates, config).unwrap();

        assert_eq!(result.lower_bounds.len(), 10);
        assert_eq!(result.upper_bounds.len(), 10);
        assert_eq!(result.residuals.len(), 10);

        for i in 0..10 {
            assert!(result.lower_bounds[i] <= result.upper_bounds[i]);
        }
    }

    #[test]
    fn test_conformity_score_methods() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let event = vec![1, 1, 0, 1, 0];
        let predictions = vec![1.5, 2.2, 2.8, 4.5, 4.8];

        let cqr_scores = compute_conformity_scores(&time, &event, &predictions, "cqr");
        let weighted_scores = compute_conformity_scores(&time, &event, &predictions, "weighted");
        let censoring_scores =
            compute_conformity_scores(&time, &event, &predictions, "censoring_adjusted");

        assert_eq!(cqr_scores.len(), 5);
        assert_eq!(weighted_scores.len(), 5);
        assert_eq!(censoring_scores.len(), 5);

        for i in 0..5 {
            if event[i] == 0 {
                assert!(
                    weighted_scores[i] < cqr_scores[i] || weighted_scores[i] == cqr_scores[i] * 0.5
                );
            }
        }
    }
}
