use pyo3::prelude::*;
use rayon::prelude::*;

use crate::constants::{
    DEFAULT_CONFORMAL_COVERAGE, DEFAULT_IPCW_TRIM, DEFAULT_MIN_GROUP_SIZE, DEFAULT_WEIGHT_TRIM,
    MAX_WEIGHT_RATIO,
};

#[derive(Debug, Clone)]
#[pyclass]
pub struct ConformalCalibrationResult {
    #[pyo3(get)]
    pub conformity_scores: Vec<f64>,
    #[pyo3(get)]
    pub ipcw_weights: Option<Vec<f64>>,
    #[pyo3(get)]
    pub quantile_threshold: f64,
    #[pyo3(get)]
    pub coverage_level: f64,
    #[pyo3(get)]
    pub n_calibration: usize,
    #[pyo3(get)]
    pub n_effective: f64,
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct ConformalPredictionResult {
    #[pyo3(get)]
    pub lower_predictive_bound: Vec<f64>,
    #[pyo3(get)]
    pub predicted_time: Vec<f64>,
    #[pyo3(get)]
    pub coverage_level: f64,
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct ConformalDiagnostics {
    #[pyo3(get)]
    pub empirical_coverage: f64,
    #[pyo3(get)]
    pub expected_coverage: f64,
    #[pyo3(get)]
    pub coverage_ci_lower: f64,
    #[pyo3(get)]
    pub coverage_ci_upper: f64,
    #[pyo3(get)]
    pub mean_lpb: f64,
    #[pyo3(get)]
    pub median_lpb: f64,
}

fn weighted_quantile(values: &[f64], weights: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }

    let n = values.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let total_weight: f64 = weights.iter().sum();
    if total_weight <= 0.0 {
        return f64::NAN;
    }

    let target = q * total_weight;
    let mut cumulative = 0.0;

    for i in 0..n {
        let idx = indices[i];
        let prev_cumulative = cumulative;
        cumulative += weights[idx];

        if cumulative >= target {
            if i == 0 || (cumulative - target).abs() < 1e-10 {
                return values[idx];
            }
            let prev_idx = indices[i - 1];
            let fraction = (target - prev_cumulative) / weights[idx];
            return values[prev_idx] + fraction * (values[idx] - values[prev_idx]);
        }
    }

    values[indices[n - 1]]
}

fn compute_km_censoring_survival(time: &[f64], status: &[i32]) -> Vec<f64> {
    let n = time.len();
    if n == 0 {
        return vec![];
    }

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut km_surv = vec![1.0; n];
    let mut cum_surv = 1.0;
    let mut at_risk = n;

    let mut i = 0;
    while i < n {
        let current_time = time[indices[i]];
        let mut censored_count = 0;

        let start_i = i;
        while i < n && (time[indices[i]] - current_time).abs() < 1e-10 {
            if status[indices[i]] == 0 {
                censored_count += 1;
            }
            i += 1;
        }

        if censored_count > 0 && at_risk > 0 {
            cum_surv *= 1.0 - censored_count as f64 / at_risk as f64;
        }

        for j in start_i..i {
            km_surv[indices[j]] = cum_surv;
        }

        at_risk -= i - start_i;
    }

    km_surv
}

fn compute_conformity_scores(
    time: &[f64],
    status: &[i32],
    predicted: &[f64],
    use_ipcw: bool,
    trim: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = time.len();
    let mut scores = Vec::with_capacity(n);
    let mut weights = Vec::with_capacity(n);

    let censoring_surv = if use_ipcw {
        compute_km_censoring_survival(time, status)
    } else {
        vec![1.0; n]
    };

    for i in 0..n {
        if status[i] == 1 {
            let score = time[i] - predicted[i];
            scores.push(score);

            let w = if use_ipcw {
                1.0 / censoring_surv[i].max(trim)
            } else {
                1.0
            };
            weights.push(w);
        }
    }

    (scores, weights)
}

#[pyfunction]
#[pyo3(signature = (time, status, predicted, coverage_level=None, use_ipcw=None))]
pub fn conformal_calibrate(
    time: Vec<f64>,
    status: Vec<i32>,
    predicted: Vec<f64>,
    coverage_level: Option<f64>,
    use_ipcw: Option<bool>,
) -> PyResult<ConformalCalibrationResult> {
    let n = time.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays cannot be empty",
        ));
    }
    if status.len() != n || predicted.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time, status, and predicted must have the same length",
        ));
    }

    let coverage = coverage_level.unwrap_or(DEFAULT_CONFORMAL_COVERAGE);
    if !(0.0..1.0).contains(&coverage) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "coverage_level must be between 0 and 1",
        ));
    }

    let use_ipcw_flag = use_ipcw.unwrap_or(true);
    let (scores, weights) =
        compute_conformity_scores(&time, &status, &predicted, use_ipcw_flag, DEFAULT_IPCW_TRIM);

    let n_uncensored = scores.len();
    if n_uncensored == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "No uncensored observations in calibration set",
        ));
    }

    let quantile_level = (1.0 - coverage) * (1.0 + 1.0 / n_uncensored as f64);
    let quantile_level = quantile_level.min(1.0);

    let quantile_threshold = weighted_quantile(&scores, &weights, quantile_level);

    let n_effective = if use_ipcw_flag {
        let sum_weights: f64 = weights.iter().sum();
        let sum_sq_weights: f64 = weights.iter().map(|w| w * w).sum();
        if sum_sq_weights > 0.0 {
            sum_weights * sum_weights / sum_sq_weights
        } else {
            n_uncensored as f64
        }
    } else {
        n_uncensored as f64
    };

    Ok(ConformalCalibrationResult {
        conformity_scores: scores,
        ipcw_weights: if use_ipcw_flag { Some(weights) } else { None },
        quantile_threshold,
        coverage_level: coverage,
        n_calibration: n_uncensored,
        n_effective,
    })
}

#[pyfunction]
#[pyo3(signature = (quantile_threshold, predicted_new, coverage_level=None))]
pub fn conformal_predict(
    quantile_threshold: f64,
    predicted_new: Vec<f64>,
    coverage_level: Option<f64>,
) -> PyResult<ConformalPredictionResult> {
    if predicted_new.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "predicted_new cannot be empty",
        ));
    }

    let coverage = coverage_level.unwrap_or(DEFAULT_CONFORMAL_COVERAGE);

    let lower_predictive_bound: Vec<f64> = predicted_new
        .iter()
        .map(|&p| (p - quantile_threshold).max(0.0))
        .collect();

    Ok(ConformalPredictionResult {
        lower_predictive_bound,
        predicted_time: predicted_new,
        coverage_level: coverage,
    })
}

#[pyfunction]
#[pyo3(signature = (time_calib, status_calib, predicted_calib, predicted_new, coverage_level=None, use_ipcw=None))]
pub fn conformal_survival_from_predictions(
    time_calib: Vec<f64>,
    status_calib: Vec<i32>,
    predicted_calib: Vec<f64>,
    predicted_new: Vec<f64>,
    coverage_level: Option<f64>,
    use_ipcw: Option<bool>,
) -> PyResult<ConformalPredictionResult> {
    let calib_result = conformal_calibrate(
        time_calib,
        status_calib,
        predicted_calib,
        coverage_level,
        use_ipcw,
    )?;

    conformal_predict(
        calib_result.quantile_threshold,
        predicted_new,
        Some(calib_result.coverage_level),
    )
}

#[pyfunction]
#[pyo3(signature = (time_test, status_test, lpb, coverage_level=None))]
pub fn conformal_coverage_test(
    time_test: Vec<f64>,
    status_test: Vec<i32>,
    lpb: Vec<f64>,
    coverage_level: Option<f64>,
) -> PyResult<ConformalDiagnostics> {
    let n = time_test.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays cannot be empty",
        ));
    }
    if status_test.len() != n || lpb.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time_test, status_test, and lpb must have the same length",
        ));
    }

    let coverage = coverage_level.unwrap_or(DEFAULT_CONFORMAL_COVERAGE);

    let mut covered_count = 0usize;
    let mut total_count = 0usize;

    for i in 0..n {
        if status_test[i] == 1 {
            total_count += 1;
            if time_test[i] >= lpb[i] {
                covered_count += 1;
            }
        }
    }

    if total_count == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "No uncensored observations in test set",
        ));
    }

    let empirical_coverage = covered_count as f64 / total_count as f64;
    let expected_coverage = coverage;

    let se = (empirical_coverage * (1.0 - empirical_coverage) / total_count as f64).sqrt();
    let z = 1.96;
    let coverage_ci_lower = (empirical_coverage - z * se).max(0.0);
    let coverage_ci_upper = (empirical_coverage + z * se).min(1.0);

    let mut sorted_lpb: Vec<f64> = lpb.clone();
    sorted_lpb.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mean_lpb = lpb.iter().sum::<f64>() / n as f64;
    let median_lpb = if n.is_multiple_of(2) {
        (sorted_lpb[n / 2 - 1] + sorted_lpb[n / 2]) / 2.0
    } else {
        sorted_lpb[n / 2]
    };

    Ok(ConformalDiagnostics {
        empirical_coverage,
        expected_coverage,
        coverage_ci_lower,
        coverage_ci_upper,
        mean_lpb,
        median_lpb,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct DoublyRobustConformalResult {
    #[pyo3(get)]
    pub lower_predictive_bound: Vec<f64>,
    #[pyo3(get)]
    pub predicted_time: Vec<f64>,
    #[pyo3(get)]
    pub coverage_level: f64,
    #[pyo3(get)]
    pub quantile_threshold: f64,
    #[pyo3(get)]
    pub imputed_censoring_times: Vec<f64>,
    #[pyo3(get)]
    pub censoring_probs: Vec<f64>,
    #[pyo3(get)]
    pub n_imputed: usize,
    #[pyo3(get)]
    pub n_effective: f64,
}

struct CensoringModel {
    unique_times: Vec<f64>,
    survival_probs: Vec<f64>,
}

impl CensoringModel {
    fn fit(time: &[f64], status: &[i32]) -> Self {
        let n = time.len();
        if n == 0 {
            return Self {
                unique_times: vec![],
                survival_probs: vec![],
            };
        }

        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            time[a]
                .partial_cmp(&time[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut unique_times = Vec::new();
        let mut survival_probs = Vec::new();
        let mut cum_surv = 1.0;
        let mut at_risk = n;

        let mut i = 0;
        while i < n {
            let current_time = time[indices[i]];
            let mut censored_count = 0;
            let mut event_count = 0;

            while i < n && (time[indices[i]] - current_time).abs() < 1e-10 {
                if status[indices[i]] == 0 {
                    censored_count += 1;
                } else {
                    event_count += 1;
                }
                i += 1;
            }

            if censored_count > 0 && at_risk > 0 {
                cum_surv *= 1.0 - censored_count as f64 / at_risk as f64;
            }

            unique_times.push(current_time);
            survival_probs.push(cum_surv);

            at_risk -= event_count + censored_count;
        }

        Self {
            unique_times,
            survival_probs,
        }
    }

    fn survival_at(&self, t: f64) -> f64 {
        if self.unique_times.is_empty() {
            return 1.0;
        }

        let mut surv = 1.0;
        for (i, &time) in self.unique_times.iter().enumerate() {
            if time > t {
                break;
            }
            surv = self.survival_probs[i];
        }
        surv
    }

    fn sample_truncated(&self, lower_bound: f64, rng_seed: u64) -> f64 {
        let surv_lower = self.survival_at(lower_bound);
        if surv_lower <= 0.0 || self.unique_times.is_empty() {
            return lower_bound * 1.5 + 1.0;
        }

        let mut rng_state = rng_seed;
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let u = (rng_state as f64) / (u64::MAX as f64);

        let target_surv = surv_lower * u;

        for (i, &surv) in self.survival_probs.iter().enumerate() {
            if surv <= target_surv && self.unique_times[i] > lower_bound {
                return self.unique_times[i];
            }
        }

        self.unique_times
            .last()
            .copied()
            .unwrap_or(lower_bound)
            .max(lower_bound)
            * 1.5
            + 1.0
    }
}

fn impute_censoring_times(
    time: &[f64],
    status: &[i32],
    censoring_model: &CensoringModel,
    seed: u64,
) -> Vec<f64> {
    let n = time.len();
    let mut imputed = Vec::with_capacity(n);

    for i in 0..n {
        if status[i] == 1 {
            imputed.push(time[i] * 2.0 + 1.0);
        } else {
            let sample_seed = seed.wrapping_add(i as u64).wrapping_mul(0x517cc1b727220a95);
            imputed.push(censoring_model.sample_truncated(time[i], sample_seed));
        }
    }

    imputed
}

fn compute_censoring_probs(
    imputed_censoring: &[f64],
    cutoff: f64,
    censoring_model: &CensoringModel,
    trim: f64,
) -> Vec<f64> {
    imputed_censoring
        .iter()
        .map(|&c| {
            if c >= cutoff {
                censoring_model.survival_at(cutoff).max(trim)
            } else {
                0.0
            }
        })
        .collect()
}

fn compute_dr_conformity_scores(
    time: &[f64],
    predicted: &[f64],
    imputed_censoring: &[f64],
    cutoff: f64,
    censoring_probs: &[f64],
    trim: f64,
) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    let n = time.len();
    let mut scores = Vec::new();
    let mut weights = Vec::new();
    let mut indices = Vec::new();

    for i in 0..n {
        if imputed_censoring[i] >= cutoff {
            let score = time[i] - predicted[i];
            scores.push(score);

            let w = 1.0 / censoring_probs[i].max(trim);
            weights.push(w);
            indices.push(i);
        }
    }

    (scores, weights, indices)
}

#[pyfunction]
#[pyo3(signature = (time, status, predicted, coverage_level=None, cutoff=None, seed=None, trim=None))]
#[allow(clippy::too_many_arguments)]
pub fn doubly_robust_conformal_calibrate(
    time: Vec<f64>,
    status: Vec<i32>,
    predicted: Vec<f64>,
    coverage_level: Option<f64>,
    cutoff: Option<f64>,
    seed: Option<u64>,
    trim: Option<f64>,
) -> PyResult<DoublyRobustConformalResult> {
    let n = time.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays cannot be empty",
        ));
    }
    if status.len() != n || predicted.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time, status, and predicted must have the same length",
        ));
    }

    let coverage = coverage_level.unwrap_or(DEFAULT_CONFORMAL_COVERAGE);
    if !(0.0..1.0).contains(&coverage) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "coverage_level must be between 0 and 1",
        ));
    }

    let trim_val = trim.unwrap_or(DEFAULT_IPCW_TRIM);
    let rng_seed = seed.unwrap_or(42);

    let censoring_model = CensoringModel::fit(&time, &status);

    let imputed_censoring = impute_censoring_times(&time, &status, &censoring_model, rng_seed);

    let cutoff_val = cutoff.unwrap_or_else(|| {
        let mut sorted_imputed: Vec<f64> = imputed_censoring.clone();
        sorted_imputed.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = (0.9 * n as f64) as usize;
        sorted_imputed[idx.min(n - 1)]
    });

    let censoring_probs =
        compute_censoring_probs(&imputed_censoring, cutoff_val, &censoring_model, trim_val);

    let (scores, weights, _filtered_indices) = compute_dr_conformity_scores(
        &time,
        &predicted,
        &imputed_censoring,
        cutoff_val,
        &censoring_probs,
        trim_val,
    );

    let n_filtered = scores.len();
    if n_filtered == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "No observations remaining after filtering by cutoff",
        ));
    }

    let quantile_level = (1.0 - coverage) * (1.0 + 1.0 / n_filtered as f64);
    let quantile_level = quantile_level.min(1.0);

    let quantile_threshold = weighted_quantile(&scores, &weights, quantile_level);

    let sum_weights: f64 = weights.iter().sum();
    let sum_sq_weights: f64 = weights.iter().map(|w| w * w).sum();
    let n_effective = if sum_sq_weights > 0.0 {
        sum_weights * sum_weights / sum_sq_weights
    } else {
        n_filtered as f64
    };

    let n_imputed = status.iter().filter(|&&s| s == 0).count();

    Ok(DoublyRobustConformalResult {
        lower_predictive_bound: vec![],
        predicted_time: predicted,
        coverage_level: coverage,
        quantile_threshold,
        imputed_censoring_times: imputed_censoring,
        censoring_probs,
        n_imputed,
        n_effective,
    })
}

#[pyfunction]
#[pyo3(signature = (time_calib, status_calib, predicted_calib, predicted_new, coverage_level=None, cutoff=None, seed=None, trim=None))]
#[allow(clippy::too_many_arguments)]
pub fn doubly_robust_conformal_survival(
    time_calib: Vec<f64>,
    status_calib: Vec<i32>,
    predicted_calib: Vec<f64>,
    predicted_new: Vec<f64>,
    coverage_level: Option<f64>,
    cutoff: Option<f64>,
    seed: Option<u64>,
    trim: Option<f64>,
) -> PyResult<DoublyRobustConformalResult> {
    let calib_result = doubly_robust_conformal_calibrate(
        time_calib,
        status_calib,
        predicted_calib,
        coverage_level,
        cutoff,
        seed,
        trim,
    )?;

    let lower_predictive_bound: Vec<f64> = predicted_new
        .iter()
        .map(|&p| (p - calib_result.quantile_threshold).max(0.0))
        .collect();

    Ok(DoublyRobustConformalResult {
        lower_predictive_bound,
        predicted_time: predicted_new,
        coverage_level: calib_result.coverage_level,
        quantile_threshold: calib_result.quantile_threshold,
        imputed_censoring_times: calib_result.imputed_censoring_times,
        censoring_probs: calib_result.censoring_probs,
        n_imputed: calib_result.n_imputed,
        n_effective: calib_result.n_effective,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct TwoSidedConformalResult {
    #[pyo3(get)]
    pub lower_bound: Vec<f64>,
    #[pyo3(get)]
    pub upper_bound: Vec<f64>,
    #[pyo3(get)]
    pub predicted_time: Vec<f64>,
    #[pyo3(get)]
    pub is_two_sided: Vec<bool>,
    #[pyo3(get)]
    pub coverage_level: f64,
    #[pyo3(get)]
    pub n_two_sided: usize,
    #[pyo3(get)]
    pub n_one_sided: usize,
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct TwoSidedCalibrationResult {
    #[pyo3(get)]
    pub lower_quantile: f64,
    #[pyo3(get)]
    pub upper_quantile: f64,
    #[pyo3(get)]
    pub censoring_score_threshold: f64,
    #[pyo3(get)]
    pub coverage_level: f64,
    #[pyo3(get)]
    pub n_uncensored: usize,
    #[pyo3(get)]
    pub n_censored: usize,
}

fn compute_censoring_scores(status: &[i32], predicted: &[f64], time: &[f64]) -> Vec<f64> {
    let n = time.len();
    let mut scores = Vec::with_capacity(n);

    let mean_time: f64 = time.iter().sum::<f64>() / n as f64;
    let mean_pred: f64 = predicted.iter().sum::<f64>() / n as f64;

    for i in 0..n {
        let time_ratio = time[i] / mean_time;
        let pred_ratio = predicted[i] / mean_pred;
        let score = if status[i] == 0 {
            (time_ratio - pred_ratio).abs() + 0.5
        } else {
            (time_ratio - pred_ratio).abs()
        };
        scores.push(score);
    }

    scores
}

fn classify_uncensored_like(
    censoring_score: f64,
    threshold: f64,
    alpha_half: f64,
    n_censored: usize,
) -> bool {
    let p_value = (1.0 + (n_censored as f64 * (1.0 - censoring_score / threshold).max(0.0)))
        / (1.0 + n_censored as f64);
    p_value >= alpha_half
}

fn compute_two_sided_scores(
    time: &[f64],
    status: &[i32],
    predicted: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let mut lower_scores = Vec::new();
    let mut upper_scores = Vec::new();

    for i in 0..time.len() {
        if status[i] == 1 {
            lower_scores.push(predicted[i] - time[i]);
            upper_scores.push(time[i] - predicted[i]);
        }
    }

    (lower_scores, upper_scores)
}

#[pyfunction]
#[pyo3(signature = (time, status, predicted, coverage_level=None))]
pub fn two_sided_conformal_calibrate(
    time: Vec<f64>,
    status: Vec<i32>,
    predicted: Vec<f64>,
    coverage_level: Option<f64>,
) -> PyResult<TwoSidedCalibrationResult> {
    let n = time.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays cannot be empty",
        ));
    }
    if status.len() != n || predicted.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time, status, and predicted must have the same length",
        ));
    }

    let coverage = coverage_level.unwrap_or(DEFAULT_CONFORMAL_COVERAGE);
    if !(0.0..1.0).contains(&coverage) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "coverage_level must be between 0 and 1",
        ));
    }

    let alpha = 1.0 - coverage;
    let alpha_half = alpha / 2.0;

    let n_uncensored = status.iter().filter(|&&s| s == 1).count();
    let n_censored = n - n_uncensored;

    if n_uncensored == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "No uncensored observations in calibration set",
        ));
    }

    let (lower_scores, upper_scores) = compute_two_sided_scores(&time, &status, &predicted);

    let uniform_weights: Vec<f64> = vec![1.0; n_uncensored];

    let lower_quantile_level =
        (1.0 - alpha_half) * (n_uncensored as f64 + 1.0) / n_uncensored as f64;
    let lower_quantile_level = lower_quantile_level.min(1.0);
    let lower_quantile = weighted_quantile(&lower_scores, &uniform_weights, lower_quantile_level);

    let upper_quantile_level =
        (1.0 - alpha_half) * (n_uncensored as f64 + 1.0) / n_uncensored as f64;
    let upper_quantile_level = upper_quantile_level.min(1.0);
    let upper_quantile = weighted_quantile(&upper_scores, &uniform_weights, upper_quantile_level);

    let censoring_scores = compute_censoring_scores(&status, &predicted, &time);
    let censored_scores: Vec<f64> = censoring_scores
        .iter()
        .zip(status.iter())
        .filter(|(_, s)| **s == 0)
        .map(|(score, _)| *score)
        .collect();

    let censoring_score_threshold = if censored_scores.is_empty() {
        f64::INFINITY
    } else {
        let mut sorted_scores = censored_scores.clone();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((1.0 - alpha_half) * sorted_scores.len() as f64) as usize;
        sorted_scores[idx.min(sorted_scores.len() - 1)]
    };

    Ok(TwoSidedCalibrationResult {
        lower_quantile,
        upper_quantile,
        censoring_score_threshold,
        coverage_level: coverage,
        n_uncensored,
        n_censored,
    })
}

#[pyfunction]
#[pyo3(signature = (calibration, predicted_new, censoring_scores_new=None))]
pub fn two_sided_conformal_predict(
    calibration: &TwoSidedCalibrationResult,
    predicted_new: Vec<f64>,
    censoring_scores_new: Option<Vec<f64>>,
) -> PyResult<TwoSidedConformalResult> {
    let n_new = predicted_new.len();
    if n_new == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "predicted_new cannot be empty",
        ));
    }

    let alpha = 1.0 - calibration.coverage_level;
    let alpha_half = alpha / 2.0;

    let censor_scores = censoring_scores_new.unwrap_or_else(|| vec![0.0; n_new]);

    if censor_scores.len() != n_new {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "censoring_scores_new must have the same length as predicted_new",
        ));
    }

    let mut lower_bound = Vec::with_capacity(n_new);
    let mut upper_bound = Vec::with_capacity(n_new);
    let mut is_two_sided = Vec::with_capacity(n_new);

    for i in 0..n_new {
        let is_uncensored_like = classify_uncensored_like(
            censor_scores[i],
            calibration.censoring_score_threshold,
            alpha_half,
            calibration.n_censored,
        );

        let lb = (predicted_new[i] - calibration.lower_quantile).max(0.0);
        lower_bound.push(lb);

        if is_uncensored_like {
            let ub = predicted_new[i] + calibration.upper_quantile;
            upper_bound.push(ub);
            is_two_sided.push(true);
        } else {
            upper_bound.push(f64::INFINITY);
            is_two_sided.push(false);
        }
    }

    let n_two_sided = is_two_sided.iter().filter(|&&x| x).count();
    let n_one_sided = n_new - n_two_sided;

    Ok(TwoSidedConformalResult {
        lower_bound,
        upper_bound,
        predicted_time: predicted_new,
        is_two_sided,
        coverage_level: calibration.coverage_level,
        n_two_sided,
        n_one_sided,
    })
}

#[pyfunction]
#[pyo3(signature = (time_calib, status_calib, predicted_calib, predicted_new, coverage_level=None))]
pub fn two_sided_conformal_survival(
    time_calib: Vec<f64>,
    status_calib: Vec<i32>,
    predicted_calib: Vec<f64>,
    predicted_new: Vec<f64>,
    coverage_level: Option<f64>,
) -> PyResult<TwoSidedConformalResult> {
    let calibration = two_sided_conformal_calibrate(
        time_calib.clone(),
        status_calib.clone(),
        predicted_calib.clone(),
        coverage_level,
    )?;

    let mean_pred: f64 = predicted_calib.iter().sum::<f64>() / predicted_calib.len() as f64;
    let censoring_scores_new: Vec<f64> = predicted_new
        .iter()
        .map(|&p| (p / mean_pred - 1.0).abs())
        .collect();

    two_sided_conformal_predict(&calibration, predicted_new, Some(censoring_scores_new))
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct ConformalSurvivalDistribution {
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub survival_lower: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub survival_upper: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub survival_median: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub coverage_level: f64,
    #[pyo3(get)]
    pub n_subjects: usize,
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct BootstrapConformalResult {
    #[pyo3(get)]
    pub lower_bound: Vec<f64>,
    #[pyo3(get)]
    pub upper_bound: Vec<f64>,
    #[pyo3(get)]
    pub predicted_time: Vec<f64>,
    #[pyo3(get)]
    pub coverage_level: f64,
    #[pyo3(get)]
    pub n_bootstrap: usize,
    #[pyo3(get)]
    pub bootstrap_quantile_lower: f64,
    #[pyo3(get)]
    pub bootstrap_quantile_upper: f64,
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct CQRConformalResult {
    #[pyo3(get)]
    pub lower_bound: Vec<f64>,
    #[pyo3(get)]
    pub upper_bound: Vec<f64>,
    #[pyo3(get)]
    pub predicted_quantile_lower: Vec<f64>,
    #[pyo3(get)]
    pub predicted_quantile_upper: Vec<f64>,
    #[pyo3(get)]
    pub coverage_level: f64,
    #[pyo3(get)]
    pub quantile_lower: f64,
    #[pyo3(get)]
    pub quantile_upper: f64,
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct ConformalCalibrationPlot {
    #[pyo3(get)]
    pub coverage_levels: Vec<f64>,
    #[pyo3(get)]
    pub empirical_coverages: Vec<f64>,
    #[pyo3(get)]
    pub ci_lower: Vec<f64>,
    #[pyo3(get)]
    pub ci_upper: Vec<f64>,
    #[pyo3(get)]
    pub n_test: usize,
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct ConformalWidthAnalysis {
    #[pyo3(get)]
    pub mean_width: f64,
    #[pyo3(get)]
    pub median_width: f64,
    #[pyo3(get)]
    pub std_width: f64,
    #[pyo3(get)]
    pub min_width: f64,
    #[pyo3(get)]
    pub max_width: f64,
    #[pyo3(get)]
    pub quantile_25: f64,
    #[pyo3(get)]
    pub quantile_75: f64,
    #[pyo3(get)]
    pub width_by_predicted: Vec<(f64, f64)>,
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct CoverageSelectionResult {
    #[pyo3(get)]
    pub optimal_coverage: f64,
    #[pyo3(get)]
    pub coverage_candidates: Vec<f64>,
    #[pyo3(get)]
    pub mean_widths: Vec<f64>,
    #[pyo3(get)]
    pub empirical_coverages: Vec<f64>,
    #[pyo3(get)]
    pub efficiency_scores: Vec<f64>,
}

#[pyfunction]
#[pyo3(signature = (time_points, survival_probs_calib, time_calib, status_calib, survival_probs_new, coverage_level=None))]
#[allow(clippy::too_many_arguments)]
pub fn conformalized_survival_distribution(
    time_points: Vec<f64>,
    survival_probs_calib: Vec<Vec<f64>>,
    time_calib: Vec<f64>,
    status_calib: Vec<i32>,
    survival_probs_new: Vec<Vec<f64>>,
    coverage_level: Option<f64>,
) -> PyResult<ConformalSurvivalDistribution> {
    let n_calib = time_calib.len();
    let n_new = survival_probs_new.len();
    let n_times = time_points.len();

    if n_calib == 0 || n_new == 0 || n_times == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays cannot be empty",
        ));
    }

    if survival_probs_calib.len() != n_calib {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "survival_probs_calib length must match time_calib",
        ));
    }

    let coverage = coverage_level.unwrap_or(DEFAULT_CONFORMAL_COVERAGE);
    let alpha = 1.0 - coverage;

    let mut conformity_scores: Vec<Vec<f64>> = vec![Vec::new(); n_times];

    for i in 0..n_calib {
        if status_calib[i] == 1 {
            let event_time = time_calib[i];
            for (t_idx, &t) in time_points.iter().enumerate() {
                if t <= event_time {
                    let predicted_surv = survival_probs_calib[i].get(t_idx).copied().unwrap_or(1.0);
                    let actual_surv = if event_time > t { 1.0 } else { 0.0 };
                    let score = (predicted_surv - actual_surv).abs();
                    conformity_scores[t_idx].push(score);
                }
            }
        }
    }

    let mut survival_lower = vec![vec![0.0; n_times]; n_new];
    let mut survival_upper = vec![vec![1.0; n_times]; n_new];
    let mut survival_median = vec![vec![0.5; n_times]; n_new];

    for t_idx in 0..n_times {
        if conformity_scores[t_idx].is_empty() {
            continue;
        }

        let scores = &conformity_scores[t_idx];
        let n_scores = scores.len();
        let quantile_level = (1.0 - alpha) * (n_scores as f64 + 1.0) / n_scores as f64;
        let quantile_level = quantile_level.min(1.0);

        let weights: Vec<f64> = vec![1.0; n_scores];
        let threshold = weighted_quantile(scores, &weights, quantile_level);

        for i in 0..n_new {
            let pred_surv = survival_probs_new[i].get(t_idx).copied().unwrap_or(0.5);
            survival_lower[i][t_idx] = (pred_surv - threshold).max(0.0);
            survival_upper[i][t_idx] = (pred_surv + threshold).min(1.0);
            survival_median[i][t_idx] = pred_surv;
        }
    }

    Ok(ConformalSurvivalDistribution {
        time_points,
        survival_lower,
        survival_upper,
        survival_median,
        coverage_level: coverage,
        n_subjects: n_new,
    })
}

fn bootstrap_sample_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut indices = Vec::with_capacity(n);
    let mut rng_state = seed;

    for _ in 0..n {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let idx = (rng_state as usize) % n;
        indices.push(idx);
    }

    indices
}

#[pyfunction]
#[pyo3(signature = (time, status, predicted, predicted_new, coverage_level=None, n_bootstrap=None, seed=None))]
#[allow(clippy::too_many_arguments)]
pub fn bootstrap_conformal_survival(
    time: Vec<f64>,
    status: Vec<i32>,
    predicted: Vec<f64>,
    predicted_new: Vec<f64>,
    coverage_level: Option<f64>,
    n_bootstrap: Option<usize>,
    seed: Option<u64>,
) -> PyResult<BootstrapConformalResult> {
    let n = time.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays cannot be empty",
        ));
    }

    let coverage = coverage_level.unwrap_or(DEFAULT_CONFORMAL_COVERAGE);
    let n_boot = n_bootstrap.unwrap_or(200);
    let base_seed = seed.unwrap_or(42);

    let alpha = 1.0 - coverage;

    let bootstrap_thresholds: Vec<(f64, f64)> = (0..n_boot)
        .into_par_iter()
        .map(|b| {
            let boot_seed = base_seed.wrapping_add(b as u64);
            let indices = bootstrap_sample_indices(n, boot_seed);

            let mut lower_scores = Vec::new();
            let mut upper_scores = Vec::new();

            for &i in &indices {
                if status[i] == 1 {
                    lower_scores.push(predicted[i] - time[i]);
                    upper_scores.push(time[i] - predicted[i]);
                }
            }

            if lower_scores.is_empty() {
                return (0.0, 0.0);
            }

            let weights: Vec<f64> = vec![1.0; lower_scores.len()];
            let q_level =
                (1.0 - alpha / 2.0) * (lower_scores.len() as f64 + 1.0) / lower_scores.len() as f64;
            let q_level = q_level.min(1.0);

            let lower_q = weighted_quantile(&lower_scores, &weights, q_level);
            let upper_q = weighted_quantile(&upper_scores, &weights, q_level);

            (lower_q, upper_q)
        })
        .collect();

    let mut all_lower: Vec<f64> = bootstrap_thresholds.iter().map(|(l, _)| *l).collect();
    let mut all_upper: Vec<f64> = bootstrap_thresholds.iter().map(|(_, u)| *u).collect();

    all_lower.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    all_upper.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let upper_idx = ((1.0 - alpha / 2.0) * n_boot as f64) as usize;
    let upper_idx = upper_idx.min(n_boot - 1);

    let final_lower_q = all_lower[upper_idx];
    let final_upper_q = all_upper[upper_idx];

    let lower_bound: Vec<f64> = predicted_new
        .iter()
        .map(|&p| (p - final_lower_q).max(0.0))
        .collect();

    let upper_bound: Vec<f64> = predicted_new.iter().map(|&p| p + final_upper_q).collect();

    Ok(BootstrapConformalResult {
        lower_bound,
        upper_bound,
        predicted_time: predicted_new,
        coverage_level: coverage,
        n_bootstrap: n_boot,
        bootstrap_quantile_lower: final_lower_q,
        bootstrap_quantile_upper: final_upper_q,
    })
}

fn estimate_conditional_quantile(
    time: &[f64],
    status: &[i32],
    predicted: &[f64],
    target_pred: f64,
    quantile: f64,
    bandwidth: f64,
) -> f64 {
    let mut weighted_times = Vec::new();
    let mut weights = Vec::new();

    for i in 0..time.len() {
        if status[i] == 1 {
            let dist = ((predicted[i] - target_pred) / bandwidth).abs();
            let weight = (-0.5 * dist * dist).exp();
            if weight > 1e-10 {
                weighted_times.push(time[i]);
                weights.push(weight);
            }
        }
    }

    if weighted_times.is_empty() {
        return target_pred;
    }

    weighted_quantile(&weighted_times, &weights, quantile)
}

#[pyfunction]
#[pyo3(signature = (time, status, predicted, predicted_new, coverage_level=None, bandwidth=None))]
#[allow(clippy::too_many_arguments)]
pub fn cqr_conformal_survival(
    time: Vec<f64>,
    status: Vec<i32>,
    predicted: Vec<f64>,
    predicted_new: Vec<f64>,
    coverage_level: Option<f64>,
    bandwidth: Option<f64>,
) -> PyResult<CQRConformalResult> {
    let n = time.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays cannot be empty",
        ));
    }

    let coverage = coverage_level.unwrap_or(DEFAULT_CONFORMAL_COVERAGE);
    let alpha = 1.0 - coverage;

    let pred_std = {
        let mean: f64 = predicted.iter().sum::<f64>() / n as f64;
        let variance: f64 = predicted.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        variance.sqrt()
    };
    let bw = bandwidth.unwrap_or(pred_std * 0.5);

    let quantile_lower = alpha / 2.0;
    let quantile_upper = 1.0 - alpha / 2.0;

    let results: Vec<(f64, f64)> = predicted_new
        .par_iter()
        .map(|&p| {
            let q_low =
                estimate_conditional_quantile(&time, &status, &predicted, p, quantile_lower, bw);
            let q_high =
                estimate_conditional_quantile(&time, &status, &predicted, p, quantile_upper, bw);
            (q_low, q_high)
        })
        .collect();

    let predicted_quantile_lower: Vec<f64> = results.iter().map(|(l, _)| *l).collect();
    let predicted_quantile_upper: Vec<f64> = results.iter().map(|(_, u)| *u).collect();

    let mut conformity_scores = Vec::new();
    for i in 0..n {
        if status[i] == 1 {
            let q_low = estimate_conditional_quantile(
                &time,
                &status,
                &predicted,
                predicted[i],
                quantile_lower,
                bw,
            );
            let q_high = estimate_conditional_quantile(
                &time,
                &status,
                &predicted,
                predicted[i],
                quantile_upper,
                bw,
            );
            let score = (q_low - time[i]).max(time[i] - q_high).max(0.0);
            conformity_scores.push(score);
        }
    }

    let n_scores = conformity_scores.len();
    if n_scores == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "No uncensored observations",
        ));
    }

    let weights: Vec<f64> = vec![1.0; n_scores];
    let q_level = (1.0 - alpha) * (n_scores as f64 + 1.0) / n_scores as f64;
    let q_level = q_level.min(1.0);
    let threshold = weighted_quantile(&conformity_scores, &weights, q_level);

    let lower_bound: Vec<f64> = predicted_quantile_lower
        .iter()
        .map(|&q| (q - threshold).max(0.0))
        .collect();

    let upper_bound: Vec<f64> = predicted_quantile_upper
        .iter()
        .map(|&q| q + threshold)
        .collect();

    Ok(CQRConformalResult {
        lower_bound,
        upper_bound,
        predicted_quantile_lower,
        predicted_quantile_upper,
        coverage_level: coverage,
        quantile_lower,
        quantile_upper,
    })
}

#[pyfunction]
#[pyo3(signature = (time_test, status_test, lower_bounds, upper_bounds=None, n_levels=None))]
pub fn conformal_calibration_plot(
    time_test: Vec<f64>,
    status_test: Vec<i32>,
    lower_bounds: Vec<Vec<f64>>,
    upper_bounds: Option<Vec<Vec<f64>>>,
    n_levels: Option<usize>,
) -> PyResult<ConformalCalibrationPlot> {
    let n_test = time_test.len();
    if n_test == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Test data cannot be empty",
        ));
    }

    let n_levels_val = n_levels.unwrap_or(10);
    let coverage_levels: Vec<f64> = (1..=n_levels_val)
        .map(|i| i as f64 / n_levels_val as f64)
        .collect();

    let n_bounds = lower_bounds.len();
    if n_bounds != n_levels_val {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "lower_bounds length must match n_levels",
        ));
    }

    let has_upper = upper_bounds.is_some();
    let upper = upper_bounds.unwrap_or_else(|| vec![vec![f64::INFINITY; n_test]; n_levels_val]);

    let mut empirical_coverages = Vec::with_capacity(n_levels_val);
    let mut ci_lower = Vec::with_capacity(n_levels_val);
    let mut ci_upper = Vec::with_capacity(n_levels_val);

    for level_idx in 0..n_levels_val {
        let lb = &lower_bounds[level_idx];
        let ub = &upper[level_idx];

        let mut covered = 0usize;
        let mut total = 0usize;

        for i in 0..n_test {
            if status_test[i] == 1 {
                total += 1;
                let above_lower = time_test[i] >= lb[i];
                let below_upper = !has_upper || time_test[i] <= ub[i];
                if above_lower && below_upper {
                    covered += 1;
                }
            }
        }

        let emp_cov = if total > 0 {
            covered as f64 / total as f64
        } else {
            0.0
        };
        let se = (emp_cov * (1.0 - emp_cov) / total.max(1) as f64).sqrt();
        let z = 1.96;

        empirical_coverages.push(emp_cov);
        ci_lower.push((emp_cov - z * se).max(0.0));
        ci_upper.push((emp_cov + z * se).min(1.0));
    }

    Ok(ConformalCalibrationPlot {
        coverage_levels,
        empirical_coverages,
        ci_lower,
        ci_upper,
        n_test,
    })
}

#[pyfunction]
#[pyo3(signature = (lower_bounds, upper_bounds, predicted))]
pub fn conformal_width_analysis(
    lower_bounds: Vec<f64>,
    upper_bounds: Vec<f64>,
    predicted: Vec<f64>,
) -> PyResult<ConformalWidthAnalysis> {
    let n = lower_bounds.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays cannot be empty",
        ));
    }
    if upper_bounds.len() != n || predicted.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All arrays must have the same length",
        ));
    }

    let widths: Vec<f64> = lower_bounds
        .iter()
        .zip(upper_bounds.iter())
        .map(|(&l, &u)| if u.is_finite() { u - l } else { f64::INFINITY })
        .filter(|&w| w.is_finite())
        .collect();

    if widths.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "No finite interval widths",
        ));
    }

    let n_finite = widths.len();
    let mean_width = widths.iter().sum::<f64>() / n_finite as f64;

    let mut sorted_widths = widths.clone();
    sorted_widths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median_width = if n_finite.is_multiple_of(2) {
        (sorted_widths[n_finite / 2 - 1] + sorted_widths[n_finite / 2]) / 2.0
    } else {
        sorted_widths[n_finite / 2]
    };

    let variance = widths
        .iter()
        .map(|&w| (w - mean_width).powi(2))
        .sum::<f64>()
        / n_finite as f64;
    let std_width = variance.sqrt();

    let min_width = sorted_widths[0];
    let max_width = sorted_widths[n_finite - 1];
    let quantile_25 = sorted_widths[(0.25 * n_finite as f64) as usize];
    let quantile_75 = sorted_widths[((0.75 * n_finite as f64) as usize).min(n_finite - 1)];

    let mut width_by_predicted: Vec<(f64, f64)> = predicted
        .iter()
        .zip(lower_bounds.iter().zip(upper_bounds.iter()))
        .filter(|(_, (_, u))| u.is_finite())
        .map(|(p, (l, u))| (*p, *u - *l))
        .collect();
    width_by_predicted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    Ok(ConformalWidthAnalysis {
        mean_width,
        median_width,
        std_width,
        min_width,
        max_width,
        quantile_25,
        quantile_75,
        width_by_predicted,
    })
}

#[pyfunction]
#[pyo3(signature = (time, status, predicted, n_folds=None, coverage_candidates=None, seed=None))]
#[allow(clippy::too_many_arguments)]
pub fn conformal_coverage_cv(
    time: Vec<f64>,
    status: Vec<i32>,
    predicted: Vec<f64>,
    n_folds: Option<usize>,
    coverage_candidates: Option<Vec<f64>>,
    seed: Option<u64>,
) -> PyResult<CoverageSelectionResult> {
    let n = time.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays cannot be empty",
        ));
    }

    let k = n_folds.unwrap_or(5);
    let candidates = coverage_candidates.unwrap_or_else(|| vec![0.80, 0.85, 0.90, 0.95, 0.99]);
    let base_seed = seed.unwrap_or(42);

    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng_state = base_seed;
    for i in (1..n).rev() {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let j = (rng_state as usize) % (i + 1);
        indices.swap(i, j);
    }

    let fold_size = n / k;
    let folds: Vec<Vec<usize>> = (0..k)
        .map(|i| {
            let start = i * fold_size;
            let end = if i == k - 1 { n } else { (i + 1) * fold_size };
            indices[start..end].to_vec()
        })
        .collect();

    let results: Vec<(f64, f64, f64)> = candidates
        .par_iter()
        .map(|&coverage| {
            let mut total_width = 0.0;
            let mut total_covered = 0usize;
            let mut total_events = 0usize;

            for fold_idx in 0..k {
                let test_indices = &folds[fold_idx];
                let train_indices: Vec<usize> = (0..k)
                    .filter(|&i| i != fold_idx)
                    .flat_map(|i| folds[i].iter().copied())
                    .collect();

                let train_time: Vec<f64> = train_indices.iter().map(|&i| time[i]).collect();
                let train_status: Vec<i32> = train_indices.iter().map(|&i| status[i]).collect();
                let train_pred: Vec<f64> = train_indices.iter().map(|&i| predicted[i]).collect();

                let mut scores = Vec::new();
                for i in 0..train_time.len() {
                    if train_status[i] == 1 {
                        scores.push(train_time[i] - train_pred[i]);
                    }
                }

                if scores.is_empty() {
                    continue;
                }

                let n_scores = scores.len();
                let q_level = (1.0 - coverage) * (n_scores as f64 + 1.0) / n_scores as f64;
                let q_level = q_level.min(1.0);
                let weights: Vec<f64> = vec![1.0; n_scores];
                let threshold = weighted_quantile(&scores, &weights, q_level);

                for &i in test_indices {
                    if status[i] == 1 {
                        total_events += 1;
                        let lb = (predicted[i] - threshold).max(0.0);
                        total_width += predicted[i] - lb;
                        if time[i] >= lb {
                            total_covered += 1;
                        }
                    }
                }
            }

            let emp_coverage = if total_events > 0 {
                total_covered as f64 / total_events as f64
            } else {
                0.0
            };
            let mean_width = if total_events > 0 {
                total_width / total_events as f64
            } else {
                f64::INFINITY
            };

            (coverage, emp_coverage, mean_width)
        })
        .collect();

    let coverage_candidates: Vec<f64> = results.iter().map(|(c, _, _)| *c).collect();
    let empirical_coverages: Vec<f64> = results.iter().map(|(_, e, _)| *e).collect();
    let mean_widths: Vec<f64> = results.iter().map(|(_, _, w)| *w).collect();

    let efficiency_scores: Vec<f64> = results
        .iter()
        .map(|(target, emp, width)| {
            let coverage_gap = (emp - target).abs();
            if *width > 0.0 && width.is_finite() {
                (1.0 - coverage_gap) / width
            } else {
                0.0
            }
        })
        .collect();

    let optimal_idx = efficiency_scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let optimal_coverage = coverage_candidates[optimal_idx];

    Ok(CoverageSelectionResult {
        optimal_coverage,
        coverage_candidates,
        mean_widths,
        empirical_coverages,
        efficiency_scores,
    })
}

#[pyfunction]
#[pyo3(signature = (time, status, predicted, predicted_new, coverage_level=None))]
pub fn conformal_survival_parallel(
    time: Vec<f64>,
    status: Vec<i32>,
    predicted: Vec<f64>,
    predicted_new: Vec<f64>,
    coverage_level: Option<f64>,
) -> PyResult<ConformalPredictionResult> {
    let n = time.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays cannot be empty",
        ));
    }

    let coverage = coverage_level.unwrap_or(DEFAULT_CONFORMAL_COVERAGE);

    let scores: Vec<f64> = (0..n)
        .into_par_iter()
        .filter_map(|i| {
            if status[i] == 1 {
                Some(time[i] - predicted[i])
            } else {
                None
            }
        })
        .collect();

    let n_scores = scores.len();
    if n_scores == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "No uncensored observations",
        ));
    }

    let weights: Vec<f64> = vec![1.0; n_scores];
    let q_level = (1.0 - coverage) * (n_scores as f64 + 1.0) / n_scores as f64;
    let q_level = q_level.min(1.0);
    let threshold = weighted_quantile(&scores, &weights, q_level);

    let lower_predictive_bound: Vec<f64> = predicted_new
        .par_iter()
        .map(|&p| (p - threshold).max(0.0))
        .collect();

    Ok(ConformalPredictionResult {
        lower_predictive_bound,
        predicted_time: predicted_new,
        coverage_level: coverage,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct WeightDiagnostics {
    #[pyo3(get)]
    pub effective_sample_size: f64,
    #[pyo3(get)]
    pub min_weight: f64,
    #[pyo3(get)]
    pub max_weight: f64,
    #[pyo3(get)]
    pub weight_variance: f64,
    #[pyo3(get)]
    pub n_trimmed: usize,
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct CovariateShiftConformalResult {
    #[pyo3(get)]
    pub lower_predictive_bound: Vec<f64>,
    #[pyo3(get)]
    pub predicted_time: Vec<f64>,
    #[pyo3(get)]
    pub coverage_level: f64,
    #[pyo3(get)]
    pub quantile_threshold: f64,
    #[pyo3(get)]
    pub combined_weights: Vec<f64>,
    #[pyo3(get)]
    pub weight_diagnostics: WeightDiagnostics,
    #[pyo3(get)]
    pub n_calibration: usize,
    #[pyo3(get)]
    pub n_effective: f64,
}

fn compute_weight_diagnostics(weights: &[f64], n_trimmed: usize) -> WeightDiagnostics {
    if weights.is_empty() {
        return WeightDiagnostics {
            effective_sample_size: 0.0,
            min_weight: 0.0,
            max_weight: 0.0,
            weight_variance: 0.0,
            n_trimmed,
        };
    }

    let sum_weights: f64 = weights.iter().sum();
    let sum_sq_weights: f64 = weights.iter().map(|w| w * w).sum();
    let effective_sample_size = if sum_sq_weights > 0.0 {
        sum_weights * sum_weights / sum_sq_weights
    } else {
        weights.len() as f64
    };

    let min_weight = weights.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_weight = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mean_weight = sum_weights / weights.len() as f64;
    let weight_variance = weights
        .iter()
        .map(|&w| (w - mean_weight).powi(2))
        .sum::<f64>()
        / weights.len() as f64;

    WeightDiagnostics {
        effective_sample_size,
        min_weight,
        max_weight,
        weight_variance,
        n_trimmed,
    }
}

#[pyfunction]
#[pyo3(signature = (time, status, predicted, importance_weights, predicted_new, coverage_level=None, use_ipcw=None, weight_trim=None))]
#[allow(clippy::too_many_arguments)]
pub fn covariate_shift_conformal_survival(
    time: Vec<f64>,
    status: Vec<i32>,
    predicted: Vec<f64>,
    importance_weights: Vec<f64>,
    predicted_new: Vec<f64>,
    coverage_level: Option<f64>,
    use_ipcw: Option<bool>,
    weight_trim: Option<f64>,
) -> PyResult<CovariateShiftConformalResult> {
    let n = time.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays cannot be empty",
        ));
    }
    if status.len() != n || predicted.len() != n || importance_weights.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time, status, predicted, and importance_weights must have the same length",
        ));
    }

    let coverage = coverage_level.unwrap_or(DEFAULT_CONFORMAL_COVERAGE);
    if !(0.0..1.0).contains(&coverage) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "coverage_level must be between 0 and 1",
        ));
    }

    let use_ipcw_flag = use_ipcw.unwrap_or(true);
    let trim = weight_trim.unwrap_or(DEFAULT_WEIGHT_TRIM);

    let censoring_surv = if use_ipcw_flag {
        compute_km_censoring_survival(&time, &status)
    } else {
        vec![1.0; n]
    };

    let mut scores = Vec::with_capacity(n);
    let mut combined_weights = Vec::with_capacity(n);
    let mut n_trimmed = 0usize;

    for i in 0..n {
        if status[i] == 1 {
            let score = time[i] - predicted[i];
            scores.push(score);

            let ipcw_weight = if use_ipcw_flag {
                1.0 / censoring_surv[i].max(trim)
            } else {
                1.0
            };

            let mut combined = importance_weights[i] * ipcw_weight;

            let max_combined =
                importance_weights.iter().cloned().fold(0.0_f64, f64::max) * MAX_WEIGHT_RATIO;
            if combined > max_combined && max_combined > 0.0 {
                combined = max_combined;
                n_trimmed += 1;
            }
            if combined < trim {
                combined = trim;
                n_trimmed += 1;
            }

            combined_weights.push(combined);
        }
    }

    let n_uncensored = scores.len();
    if n_uncensored == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "No uncensored observations in calibration set",
        ));
    }

    let quantile_level = (1.0 - coverage) * (1.0 + 1.0 / n_uncensored as f64);
    let quantile_level = quantile_level.min(1.0);

    let quantile_threshold = weighted_quantile(&scores, &combined_weights, quantile_level);

    let weight_diagnostics = compute_weight_diagnostics(&combined_weights, n_trimmed);

    let lower_predictive_bound: Vec<f64> = predicted_new
        .iter()
        .map(|&p| (p - quantile_threshold).max(0.0))
        .collect();

    Ok(CovariateShiftConformalResult {
        lower_predictive_bound,
        predicted_time: predicted_new,
        coverage_level: coverage,
        quantile_threshold,
        combined_weights,
        weight_diagnostics: weight_diagnostics.clone(),
        n_calibration: n_uncensored,
        n_effective: weight_diagnostics.effective_sample_size,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct CVPlusCalibrationResult {
    #[pyo3(get)]
    pub conformity_scores: Vec<f64>,
    #[pyo3(get)]
    pub quantile_threshold: f64,
    #[pyo3(get)]
    pub coverage_level: f64,
    #[pyo3(get)]
    pub n_calibration: usize,
    #[pyo3(get)]
    pub adjustment_factor: f64,
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct CVPlusConformalResult {
    #[pyo3(get)]
    pub lower_predictive_bound: Vec<f64>,
    #[pyo3(get)]
    pub predicted_time: Vec<f64>,
    #[pyo3(get)]
    pub coverage_level: f64,
    #[pyo3(get)]
    pub quantile_threshold: f64,
    #[pyo3(get)]
    pub loo_scores: Vec<f64>,
    #[pyo3(get)]
    pub n_calibration: usize,
}

#[pyfunction]
#[pyo3(signature = (time, status, predicted_loo, coverage_level=None))]
pub fn cvplus_conformal_calibrate(
    time: Vec<f64>,
    status: Vec<i32>,
    predicted_loo: Vec<f64>,
    coverage_level: Option<f64>,
) -> PyResult<CVPlusCalibrationResult> {
    let n = time.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays cannot be empty",
        ));
    }
    if status.len() != n || predicted_loo.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time, status, and predicted_loo must have the same length",
        ));
    }

    let coverage = coverage_level.unwrap_or(DEFAULT_CONFORMAL_COVERAGE);
    if !(0.0..1.0).contains(&coverage) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "coverage_level must be between 0 and 1",
        ));
    }

    let mut scores = Vec::with_capacity(n);
    for i in 0..n {
        if status[i] == 1 {
            let score = time[i] - predicted_loo[i];
            scores.push(score);
        }
    }

    let n_uncensored = scores.len();
    if n_uncensored == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "No uncensored observations in calibration set",
        ));
    }

    let adjustment_factor = (n_uncensored as f64 + 1.0) / n_uncensored as f64;
    let quantile_level = (1.0 - coverage) * adjustment_factor;
    let quantile_level = quantile_level.min(1.0);

    let weights: Vec<f64> = vec![1.0; n_uncensored];
    let quantile_threshold = weighted_quantile(&scores, &weights, quantile_level);

    Ok(CVPlusCalibrationResult {
        conformity_scores: scores,
        quantile_threshold,
        coverage_level: coverage,
        n_calibration: n_uncensored,
        adjustment_factor,
    })
}

#[pyfunction]
#[pyo3(signature = (time, status, predicted_loo, predicted_new, coverage_level=None))]
pub fn cvplus_conformal_survival(
    time: Vec<f64>,
    status: Vec<i32>,
    predicted_loo: Vec<f64>,
    predicted_new: Vec<f64>,
    coverage_level: Option<f64>,
) -> PyResult<CVPlusConformalResult> {
    let calib_result = cvplus_conformal_calibrate(time, status, predicted_loo, coverage_level)?;

    let lower_predictive_bound: Vec<f64> = predicted_new
        .iter()
        .map(|&p| (p - calib_result.quantile_threshold).max(0.0))
        .collect();

    Ok(CVPlusConformalResult {
        lower_predictive_bound,
        predicted_time: predicted_new,
        coverage_level: calib_result.coverage_level,
        quantile_threshold: calib_result.quantile_threshold,
        loo_scores: calib_result.conformity_scores,
        n_calibration: calib_result.n_calibration,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct MondrianDiagnostics {
    #[pyo3(get)]
    pub group_labels: Vec<i32>,
    #[pyo3(get)]
    pub group_sizes: Vec<usize>,
    #[pyo3(get)]
    pub group_thresholds: Vec<f64>,
    #[pyo3(get)]
    pub n_small_groups: usize,
    #[pyo3(get)]
    pub global_threshold: f64,
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct MondrianCalibrationResult {
    #[pyo3(get)]
    pub group_thresholds: std::collections::HashMap<i32, f64>,
    #[pyo3(get)]
    pub global_threshold: f64,
    #[pyo3(get)]
    pub coverage_level: f64,
    #[pyo3(get)]
    pub group_sizes: std::collections::HashMap<i32, usize>,
    #[pyo3(get)]
    pub min_group_size: usize,
    #[pyo3(get)]
    pub diagnostics: MondrianDiagnostics,
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct MondrianConformalResult {
    #[pyo3(get)]
    pub lower_predictive_bound: Vec<f64>,
    #[pyo3(get)]
    pub predicted_time: Vec<f64>,
    #[pyo3(get)]
    pub coverage_level: f64,
    #[pyo3(get)]
    pub applied_thresholds: Vec<f64>,
    #[pyo3(get)]
    pub group_labels_used: Vec<i32>,
    #[pyo3(get)]
    pub used_global_fallback: Vec<bool>,
    #[pyo3(get)]
    pub diagnostics: MondrianDiagnostics,
}

#[pyfunction]
#[pyo3(signature = (time, status, predicted, group_labels, coverage_level=None, min_group_size=None))]
#[allow(clippy::too_many_arguments)]
pub fn mondrian_conformal_calibrate(
    time: Vec<f64>,
    status: Vec<i32>,
    predicted: Vec<f64>,
    group_labels: Vec<i32>,
    coverage_level: Option<f64>,
    min_group_size: Option<usize>,
) -> PyResult<MondrianCalibrationResult> {
    let n = time.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays cannot be empty",
        ));
    }
    if status.len() != n || predicted.len() != n || group_labels.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time, status, predicted, and group_labels must have the same length",
        ));
    }

    let coverage = coverage_level.unwrap_or(DEFAULT_CONFORMAL_COVERAGE);
    if !(0.0..1.0).contains(&coverage) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "coverage_level must be between 0 and 1",
        ));
    }

    let min_size = min_group_size.unwrap_or(DEFAULT_MIN_GROUP_SIZE);

    let mut group_scores: std::collections::HashMap<i32, Vec<f64>> =
        std::collections::HashMap::new();
    let mut all_scores = Vec::new();

    for i in 0..n {
        if status[i] == 1 {
            let score = time[i] - predicted[i];
            all_scores.push(score);
            group_scores.entry(group_labels[i]).or_default().push(score);
        }
    }

    if all_scores.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "No uncensored observations in calibration set",
        ));
    }

    let n_all = all_scores.len();
    let global_q_level = (1.0 - coverage) * (n_all as f64 + 1.0) / n_all as f64;
    let global_q_level = global_q_level.min(1.0);
    let global_weights: Vec<f64> = vec![1.0; n_all];
    let global_threshold = weighted_quantile(&all_scores, &global_weights, global_q_level);

    let mut group_thresholds: std::collections::HashMap<i32, f64> =
        std::collections::HashMap::new();
    let mut group_sizes: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
    let mut n_small_groups = 0usize;

    let mut diag_group_labels = Vec::new();
    let mut diag_group_sizes = Vec::new();
    let mut diag_group_thresholds = Vec::new();

    for (&group, scores) in &group_scores {
        let size = scores.len();
        group_sizes.insert(group, size);
        diag_group_labels.push(group);
        diag_group_sizes.push(size);

        if size >= min_size {
            let q_level = (1.0 - coverage) * (size as f64 + 1.0) / size as f64;
            let q_level = q_level.min(1.0);
            let weights: Vec<f64> = vec![1.0; size];
            let threshold = weighted_quantile(scores, &weights, q_level);
            group_thresholds.insert(group, threshold);
            diag_group_thresholds.push(threshold);
        } else {
            group_thresholds.insert(group, global_threshold);
            diag_group_thresholds.push(global_threshold);
            n_small_groups += 1;
        }
    }

    let diagnostics = MondrianDiagnostics {
        group_labels: diag_group_labels,
        group_sizes: diag_group_sizes,
        group_thresholds: diag_group_thresholds,
        n_small_groups,
        global_threshold,
    };

    Ok(MondrianCalibrationResult {
        group_thresholds,
        global_threshold,
        coverage_level: coverage,
        group_sizes,
        min_group_size: min_size,
        diagnostics,
    })
}

#[pyfunction]
#[pyo3(signature = (calibration, predicted_new, group_labels_new))]
pub fn mondrian_conformal_predict(
    calibration: &MondrianCalibrationResult,
    predicted_new: Vec<f64>,
    group_labels_new: Vec<i32>,
) -> PyResult<MondrianConformalResult> {
    let n_new = predicted_new.len();
    if n_new == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "predicted_new cannot be empty",
        ));
    }
    if group_labels_new.len() != n_new {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "predicted_new and group_labels_new must have the same length",
        ));
    }

    let mut lower_predictive_bound = Vec::with_capacity(n_new);
    let mut applied_thresholds = Vec::with_capacity(n_new);
    let mut used_global_fallback = Vec::with_capacity(n_new);

    for i in 0..n_new {
        let group = group_labels_new[i];
        let (threshold, used_fallback) = if let Some(&t) = calibration.group_thresholds.get(&group)
        {
            let size = calibration.group_sizes.get(&group).copied().unwrap_or(0);
            if size >= calibration.min_group_size {
                (t, false)
            } else {
                (calibration.global_threshold, true)
            }
        } else {
            (calibration.global_threshold, true)
        };

        lower_predictive_bound.push((predicted_new[i] - threshold).max(0.0));
        applied_thresholds.push(threshold);
        used_global_fallback.push(used_fallback);
    }

    Ok(MondrianConformalResult {
        lower_predictive_bound,
        predicted_time: predicted_new,
        coverage_level: calibration.coverage_level,
        applied_thresholds,
        group_labels_used: group_labels_new,
        used_global_fallback,
        diagnostics: calibration.diagnostics.clone(),
    })
}

#[pyfunction]
#[pyo3(signature = (time, status, predicted, group_labels, predicted_new, group_labels_new, coverage_level=None, min_group_size=None))]
#[allow(clippy::too_many_arguments)]
pub fn mondrian_conformal_survival(
    time: Vec<f64>,
    status: Vec<i32>,
    predicted: Vec<f64>,
    group_labels: Vec<i32>,
    predicted_new: Vec<f64>,
    group_labels_new: Vec<i32>,
    coverage_level: Option<f64>,
    min_group_size: Option<usize>,
) -> PyResult<MondrianConformalResult> {
    let calibration = mondrian_conformal_calibrate(
        time,
        status,
        predicted,
        group_labels,
        coverage_level,
        min_group_size,
    )?;

    mondrian_conformal_predict(&calibration, predicted_new, group_labels_new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weighted_quantile_basic() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0];

        let q50 = weighted_quantile(&values, &weights, 0.5);
        assert!((2.0..=3.5).contains(&q50));

        let q90 = weighted_quantile(&values, &weights, 0.9);
        assert!((4.0..=5.0).contains(&q90));
    }

    #[test]
    fn test_weighted_quantile_unequal_weights() {
        let values = vec![1.0, 2.0, 3.0];
        let weights = vec![1.0, 2.0, 1.0];

        let q50 = weighted_quantile(&values, &weights, 0.5);
        assert!((1.5..=2.5).contains(&q50));
    }

    #[test]
    fn test_conformity_scores_uncensored() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 1, 1, 1];
        let predicted = vec![1.1, 1.9, 3.2, 3.8, 5.1];

        let (scores, weights) = compute_conformity_scores(&time, &status, &predicted, false, 0.01);

        assert_eq!(scores.len(), 5);
        assert_eq!(weights.len(), 5);
        assert!((scores[0] - (-0.1)).abs() < 1e-10);
        assert!((scores[1] - 0.1).abs() < 1e-10);
        assert!(weights.iter().all(|&w| (w - 1.0).abs() < 1e-10));
    }

    #[test]
    fn test_conformal_calibrate_no_censoring() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 1, 1, 1];
        let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = conformal_calibrate(time, status, predicted, Some(0.9), Some(false)).unwrap();

        assert_eq!(result.n_calibration, 5);
        assert!((result.coverage_level - 0.9).abs() < 1e-10);
        assert!(result.ipcw_weights.is_none());
    }

    #[test]
    fn test_conformal_with_ipcw() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 0, 1, 1];
        let predicted = vec![1.1, 1.9, 2.8, 4.2, 4.8];

        let result = conformal_calibrate(time, status, predicted, Some(0.9), Some(true)).unwrap();

        assert_eq!(result.n_calibration, 4);
        assert!(result.ipcw_weights.is_some());
        let weights = result.ipcw_weights.unwrap();
        assert_eq!(weights.len(), 4);
        assert!(weights.iter().all(|&w| w >= 1.0));
    }

    #[test]
    fn test_conformal_coverage_guarantee() {
        let time_calib = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let status_calib = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let predicted_calib = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let predicted_new = vec![3.0, 5.0, 7.0];

        let result = conformal_survival_from_predictions(
            time_calib,
            status_calib,
            predicted_calib,
            predicted_new.clone(),
            Some(0.9),
            Some(false),
        )
        .unwrap();

        assert_eq!(result.lower_predictive_bound.len(), 3);
        assert_eq!(result.predicted_time.len(), 3);
        for (lower, pred) in result
            .lower_predictive_bound
            .iter()
            .zip(predicted_new.iter())
        {
            assert!(lower <= pred);
        }
    }

    #[test]
    fn test_conformal_empty_input() {
        let result = conformal_calibrate(vec![], vec![], vec![], None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_conformal_all_censored() {
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![0, 0, 0];
        let predicted = vec![1.0, 2.0, 3.0];

        let result = conformal_calibrate(time, status, predicted, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_conformal_coverage_test() {
        let time_test = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status_test = vec![1, 1, 1, 1, 1];
        let lpb = vec![0.5, 1.5, 2.5, 3.5, 4.5];

        let result = conformal_coverage_test(time_test, status_test, lpb, Some(0.9)).unwrap();

        assert!((result.empirical_coverage - 1.0).abs() < 1e-10);
        assert!((result.expected_coverage - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_conformal_predict_basic() {
        let predicted = vec![5.0, 10.0, 15.0];
        let quantile_threshold = 2.0;

        let result = conformal_predict(quantile_threshold, predicted, Some(0.9)).unwrap();

        assert_eq!(result.lower_predictive_bound.len(), 3);
        assert!((result.lower_predictive_bound[0] - 3.0).abs() < 1e-10);
        assert!((result.lower_predictive_bound[1] - 8.0).abs() < 1e-10);
        assert!((result.lower_predictive_bound[2] - 13.0).abs() < 1e-10);
    }

    #[test]
    fn test_conformal_predict_clamps_to_zero() {
        let predicted = vec![1.0];
        let quantile_threshold = 5.0;

        let result = conformal_predict(quantile_threshold, predicted, None).unwrap();

        assert!((result.lower_predictive_bound[0] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_censoring_model_fit() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 0, 1];

        let model = CensoringModel::fit(&time, &status);

        assert!(!model.unique_times.is_empty());
        assert_eq!(model.unique_times.len(), model.survival_probs.len());
        assert!(
            model
                .survival_probs
                .iter()
                .all(|&s| (0.0..=1.0).contains(&s))
        );
    }

    #[test]
    fn test_censoring_model_survival_at() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 0, 1];

        let model = CensoringModel::fit(&time, &status);

        let surv_0 = model.survival_at(0.0);
        assert!((surv_0 - 1.0).abs() < 1e-10);

        let surv_10 = model.survival_at(10.0);
        assert!(surv_10 <= 1.0);
        assert!(surv_10 >= 0.0);
    }

    #[test]
    fn test_impute_censoring_times() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 0, 1];

        let model = CensoringModel::fit(&time, &status);
        let imputed = impute_censoring_times(&time, &status, &model, 42);

        assert_eq!(imputed.len(), 5);
        assert!(imputed[1] > time[1]);
        assert!(imputed[3] > time[3]);
    }

    #[test]
    fn test_doubly_robust_conformal_calibrate() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let status = vec![1, 1, 0, 1, 0, 1, 1, 0, 1, 1];
        let predicted = vec![1.1, 1.9, 2.8, 4.2, 4.8, 6.1, 6.9, 7.8, 9.2, 9.8];

        let result = doubly_robust_conformal_calibrate(
            time,
            status,
            predicted,
            Some(0.9),
            None,
            Some(42),
            None,
        )
        .unwrap();

        assert!((result.coverage_level - 0.9).abs() < 1e-10);
        assert_eq!(result.imputed_censoring_times.len(), 10);
        assert_eq!(result.censoring_probs.len(), 10);
        assert!(result.n_imputed > 0);
        assert!(result.n_effective > 0.0);
    }

    #[test]
    fn test_doubly_robust_conformal_survival() {
        let time_calib = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let status_calib = vec![1, 1, 0, 1, 0, 1, 1, 0, 1, 1];
        let predicted_calib = vec![1.1, 1.9, 2.8, 4.2, 4.8, 6.1, 6.9, 7.8, 9.2, 9.8];
        let predicted_new = vec![3.0, 5.0, 7.0];

        let result = doubly_robust_conformal_survival(
            time_calib,
            status_calib,
            predicted_calib,
            predicted_new.clone(),
            Some(0.9),
            None,
            Some(42),
            None,
        )
        .unwrap();

        assert_eq!(result.lower_predictive_bound.len(), 3);
        assert_eq!(result.predicted_time.len(), 3);
        for (lower, pred) in result
            .lower_predictive_bound
            .iter()
            .zip(predicted_new.iter())
        {
            assert!(lower <= pred);
            assert!(*lower >= 0.0);
        }
    }

    #[test]
    fn test_doubly_robust_empty_input() {
        let result =
            doubly_robust_conformal_calibrate(vec![], vec![], vec![], None, None, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_doubly_robust_all_uncensored() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 1, 1, 1];
        let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = doubly_robust_conformal_calibrate(
            time,
            status,
            predicted,
            Some(0.9),
            None,
            Some(42),
            None,
        )
        .unwrap();

        assert_eq!(result.n_imputed, 0);
    }

    #[test]
    fn test_doubly_robust_deterministic_with_seed() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 0, 1];
        let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result1 = doubly_robust_conformal_calibrate(
            time.clone(),
            status.clone(),
            predicted.clone(),
            Some(0.9),
            None,
            Some(123),
            None,
        )
        .unwrap();

        let result2 = doubly_robust_conformal_calibrate(
            time,
            status,
            predicted,
            Some(0.9),
            None,
            Some(123),
            None,
        )
        .unwrap();

        assert!((result1.quantile_threshold - result2.quantile_threshold).abs() < 1e-10);
    }

    #[test]
    fn test_two_sided_conformal_calibrate() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let status = vec![1, 1, 0, 1, 0, 1, 1, 0, 1, 1];
        let predicted = vec![1.1, 1.9, 2.8, 4.2, 4.8, 6.1, 6.9, 7.8, 9.2, 9.8];

        let result = two_sided_conformal_calibrate(time, status, predicted, Some(0.9)).unwrap();

        assert!((result.coverage_level - 0.9).abs() < 1e-10);
        assert_eq!(result.n_uncensored, 7);
        assert_eq!(result.n_censored, 3);
        assert!(result.lower_quantile.is_finite());
        assert!(result.upper_quantile.is_finite());
    }

    #[test]
    fn test_two_sided_conformal_calibrate_all_uncensored() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 1, 1, 1];
        let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = two_sided_conformal_calibrate(time, status, predicted, Some(0.9)).unwrap();

        assert_eq!(result.n_uncensored, 5);
        assert_eq!(result.n_censored, 0);
        assert!(result.censoring_score_threshold.is_infinite());
    }

    #[test]
    fn test_two_sided_conformal_predict() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let status = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let calibration =
            two_sided_conformal_calibrate(time, status, predicted, Some(0.9)).unwrap();

        let predicted_new = vec![3.0, 5.0, 7.0];
        let result =
            two_sided_conformal_predict(&calibration, predicted_new.clone(), None).unwrap();

        assert_eq!(result.lower_bound.len(), 3);
        assert_eq!(result.upper_bound.len(), 3);
        assert_eq!(result.is_two_sided.len(), 3);

        for (i, pred) in predicted_new.iter().enumerate() {
            assert!(result.lower_bound[i] <= *pred);
            assert!(result.lower_bound[i] >= 0.0);
            if result.is_two_sided[i] {
                assert!(result.upper_bound[i] >= *pred);
                assert!(result.upper_bound[i].is_finite());
            }
        }
    }

    #[test]
    fn test_two_sided_conformal_survival() {
        let time_calib = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let status_calib = vec![1, 1, 0, 1, 0, 1, 1, 0, 1, 1];
        let predicted_calib = vec![1.1, 1.9, 2.8, 4.2, 4.8, 6.1, 6.9, 7.8, 9.2, 9.8];
        let predicted_new = vec![3.0, 5.0, 7.0];

        let result = two_sided_conformal_survival(
            time_calib,
            status_calib,
            predicted_calib,
            predicted_new.clone(),
            Some(0.9),
        )
        .unwrap();

        assert_eq!(result.lower_bound.len(), 3);
        assert_eq!(result.upper_bound.len(), 3);
        assert_eq!(result.n_two_sided + result.n_one_sided, 3);

        for (lower, pred) in result.lower_bound.iter().zip(predicted_new.iter()) {
            assert!(lower <= pred);
            assert!(*lower >= 0.0);
        }
    }

    #[test]
    fn test_two_sided_conformal_empty_input() {
        let result = two_sided_conformal_calibrate(vec![], vec![], vec![], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_two_sided_conformal_all_censored() {
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![0, 0, 0];
        let predicted = vec![1.0, 2.0, 3.0];

        let result = two_sided_conformal_calibrate(time, status, predicted, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_two_sided_bounds_ordering() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 1, 1, 1];
        let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let calibration =
            two_sided_conformal_calibrate(time, status, predicted, Some(0.9)).unwrap();

        let predicted_new = vec![5.0];
        let result = two_sided_conformal_predict(&calibration, predicted_new, None).unwrap();

        assert!(result.lower_bound[0] <= result.upper_bound[0]);
    }

    #[test]
    fn test_compute_two_sided_scores() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 0, 1, 1];
        let predicted = vec![1.1, 1.9, 2.8, 4.2, 4.8];

        let (lower_scores, upper_scores) = compute_two_sided_scores(&time, &status, &predicted);

        assert_eq!(lower_scores.len(), 4);
        assert_eq!(upper_scores.len(), 4);
    }

    #[test]
    fn test_conformalized_survival_distribution() {
        let time_points = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let survival_probs_calib = vec![
            vec![0.9, 0.8, 0.7, 0.6, 0.5],
            vec![0.95, 0.85, 0.75, 0.65, 0.55],
            vec![0.85, 0.75, 0.65, 0.55, 0.45],
        ];
        let time_calib = vec![3.0, 4.0, 2.0];
        let status_calib = vec![1, 1, 1];
        let survival_probs_new = vec![vec![0.9, 0.8, 0.7, 0.6, 0.5]];

        let result = conformalized_survival_distribution(
            time_points.clone(),
            survival_probs_calib,
            time_calib,
            status_calib,
            survival_probs_new,
            Some(0.9),
        )
        .unwrap();

        assert_eq!(result.time_points.len(), 5);
        assert_eq!(result.n_subjects, 1);
        assert_eq!(result.survival_lower.len(), 1);
        assert_eq!(result.survival_upper.len(), 1);
    }

    #[test]
    fn test_bootstrap_conformal_survival() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let status = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let predicted_new = vec![3.0, 5.0, 7.0];

        let result = bootstrap_conformal_survival(
            time,
            status,
            predicted,
            predicted_new.clone(),
            Some(0.9),
            Some(50),
            Some(42),
        )
        .unwrap();

        assert_eq!(result.lower_bound.len(), 3);
        assert_eq!(result.upper_bound.len(), 3);
        assert_eq!(result.n_bootstrap, 50);

        for i in 0..3 {
            assert!(result.lower_bound[i] <= result.upper_bound[i]);
        }
    }

    #[test]
    fn test_cqr_conformal_survival() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let status = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let predicted_new = vec![3.0, 5.0, 7.0];

        let result =
            cqr_conformal_survival(time, status, predicted, predicted_new, Some(0.9), None)
                .unwrap();

        assert_eq!(result.lower_bound.len(), 3);
        assert_eq!(result.upper_bound.len(), 3);

        for i in 0..3 {
            assert!(result.lower_bound[i] <= result.upper_bound[i]);
        }
    }

    #[test]
    fn test_conformal_width_analysis() {
        let lower_bounds = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let upper_bounds = vec![3.0, 4.0, 5.0, 6.0, 7.0];
        let predicted = vec![2.0, 3.0, 4.0, 5.0, 6.0];

        let result = conformal_width_analysis(lower_bounds, upper_bounds, predicted).unwrap();

        assert!((result.mean_width - 2.0).abs() < 1e-10);
        assert!((result.median_width - 2.0).abs() < 1e-10);
        assert!((result.std_width - 0.0).abs() < 1e-10);
        assert!((result.min_width - 2.0).abs() < 1e-10);
        assert!((result.max_width - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_conformal_coverage_cv() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let status = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let result = conformal_coverage_cv(
            time,
            status,
            predicted,
            Some(5),
            Some(vec![0.8, 0.9, 0.95]),
            Some(42),
        )
        .unwrap();

        assert_eq!(result.coverage_candidates.len(), 3);
        assert_eq!(result.mean_widths.len(), 3);
        assert_eq!(result.empirical_coverages.len(), 3);
        assert!(result.optimal_coverage >= 0.8 && result.optimal_coverage <= 0.95);
    }

    #[test]
    fn test_conformal_survival_parallel() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let status = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let predicted_new = vec![3.0, 5.0, 7.0];

        let result =
            conformal_survival_parallel(time, status, predicted, predicted_new.clone(), Some(0.9))
                .unwrap();

        assert_eq!(result.lower_predictive_bound.len(), 3);
        for (lower, pred) in result
            .lower_predictive_bound
            .iter()
            .zip(predicted_new.iter())
        {
            assert!(lower <= pred);
            assert!(*lower >= 0.0);
        }
    }

    #[test]
    fn test_bootstrap_sample_indices() {
        let indices = bootstrap_sample_indices(10, 42);
        assert_eq!(indices.len(), 10);
        assert!(indices.iter().all(|&i| i < 10));
    }

    #[test]
    fn test_covariate_shift_conformal_survival() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 1, 1, 1];
        let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let importance_weights = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let predicted_new = vec![2.5, 4.0];

        let result = covariate_shift_conformal_survival(
            time,
            status,
            predicted,
            importance_weights,
            predicted_new.clone(),
            Some(0.9),
            Some(false),
            None,
        )
        .unwrap();

        assert_eq!(result.lower_predictive_bound.len(), 2);
        assert_eq!(result.predicted_time.len(), 2);
        assert_eq!(result.n_calibration, 5);
        for lower in result.lower_predictive_bound.iter() {
            assert!(*lower >= 0.0);
            assert!(lower.is_finite());
        }
    }

    #[test]
    fn test_covariate_shift_with_different_weights() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 1, 1, 1];
        let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let importance_weights = vec![0.5, 1.0, 1.5, 1.0, 0.5];
        let predicted_new = vec![3.0];

        let result = covariate_shift_conformal_survival(
            time,
            status,
            predicted,
            importance_weights,
            predicted_new,
            Some(0.9),
            Some(true),
            None,
        )
        .unwrap();

        assert_eq!(result.combined_weights.len(), 5);
        assert!(result.weight_diagnostics.effective_sample_size > 0.0);
        assert!(result.weight_diagnostics.min_weight > 0.0);
    }

    #[test]
    fn test_covariate_shift_empty_input() {
        let result = covariate_shift_conformal_survival(
            vec![],
            vec![],
            vec![],
            vec![],
            vec![1.0],
            None,
            None,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_cvplus_conformal_calibrate() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 1, 1, 1];
        let predicted_loo = vec![1.2, 1.8, 3.1, 4.1, 4.9];

        let result = cvplus_conformal_calibrate(time, status, predicted_loo, Some(0.9)).unwrap();

        assert_eq!(result.n_calibration, 5);
        assert!((result.coverage_level - 0.9).abs() < 1e-10);
        assert!(result.quantile_threshold.is_finite());
        assert!(result.adjustment_factor > 1.0);
    }

    #[test]
    fn test_cvplus_conformal_survival() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 1, 1, 1];
        let predicted_loo = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let predicted_new = vec![2.5, 4.0];

        let result = cvplus_conformal_survival(
            time,
            status,
            predicted_loo,
            predicted_new.clone(),
            Some(0.9),
        )
        .unwrap();

        assert_eq!(result.lower_predictive_bound.len(), 2);
        assert_eq!(result.n_calibration, 5);
        for lower in result.lower_predictive_bound.iter() {
            assert!(*lower >= 0.0);
            assert!(lower.is_finite());
        }
    }

    #[test]
    fn test_cvplus_empty_input() {
        let result = cvplus_conformal_calibrate(vec![], vec![], vec![], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_cvplus_all_censored() {
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![0, 0, 0];
        let predicted_loo = vec![1.0, 2.0, 3.0];

        let result = cvplus_conformal_calibrate(time, status, predicted_loo, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_mondrian_conformal_calibrate() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let status = vec![1, 1, 1, 1, 1, 1];
        let predicted = vec![1.1, 1.9, 3.2, 3.8, 5.1, 6.2];
        let group_labels = vec![0, 0, 0, 1, 1, 1];

        let result =
            mondrian_conformal_calibrate(time, status, predicted, group_labels, Some(0.9), Some(2))
                .unwrap();

        assert_eq!(result.group_thresholds.len(), 2);
        assert_eq!(result.group_sizes.len(), 2);
        assert!(result.global_threshold.is_finite());
    }

    #[test]
    fn test_mondrian_conformal_survival() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let status = vec![1, 1, 1, 1, 1, 1];
        let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let group_labels = vec![0, 0, 0, 1, 1, 1];
        let predicted_new = vec![2.5, 5.0];
        let group_labels_new = vec![0, 1];

        let result = mondrian_conformal_survival(
            time,
            status,
            predicted,
            group_labels,
            predicted_new.clone(),
            group_labels_new,
            Some(0.9),
            Some(2),
        )
        .unwrap();

        assert_eq!(result.lower_predictive_bound.len(), 2);
        assert_eq!(result.applied_thresholds.len(), 2);
        for lower in result.lower_predictive_bound.iter() {
            assert!(*lower >= 0.0);
            assert!(lower.is_finite());
        }
    }

    #[test]
    fn test_mondrian_conformal_predict() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let status = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let group_labels = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

        let calibration =
            mondrian_conformal_calibrate(time, status, predicted, group_labels, Some(0.9), Some(3))
                .unwrap();

        let predicted_new = vec![3.0, 7.0];
        let group_labels_new = vec![0, 1];

        let result =
            mondrian_conformal_predict(&calibration, predicted_new, group_labels_new).unwrap();

        assert_eq!(result.lower_predictive_bound.len(), 2);
        assert_eq!(result.used_global_fallback.len(), 2);
    }

    #[test]
    fn test_mondrian_small_group_fallback() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let status = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let group_labels = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1];

        let calibration =
            mondrian_conformal_calibrate(time, status, predicted, group_labels, Some(0.9), Some(5))
                .unwrap();

        assert!(calibration.diagnostics.n_small_groups > 0);

        let result = mondrian_conformal_predict(&calibration, vec![9.5], vec![1]).unwrap();
        assert!(result.used_global_fallback[0]);
    }

    #[test]
    fn test_mondrian_empty_input() {
        let result = mondrian_conformal_calibrate(vec![], vec![], vec![], vec![], None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_mondrian_new_group_fallback() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 1, 1, 1];
        let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let group_labels = vec![0, 0, 0, 0, 0];

        let calibration =
            mondrian_conformal_calibrate(time, status, predicted, group_labels, Some(0.9), Some(3))
                .unwrap();

        let result = mondrian_conformal_predict(&calibration, vec![3.0], vec![99]).unwrap();
        assert!(result.used_global_fallback[0]);
    }
}
