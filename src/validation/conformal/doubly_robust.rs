use super::*;

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
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

pub(super) struct CensoringModel {
    pub(super) unique_times: Vec<f64>,
    pub(super) survival_probs: Vec<f64>,
}

impl CensoringModel {
    pub(super) fn fit(time: &[f64], status: &[i32]) -> Self {
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

            while i < n && (time[indices[i]] - current_time).abs() < crate::constants::TIME_EPSILON
            {
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

    pub(super) fn survival_at(&self, t: f64) -> f64 {
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

    pub(super) fn sample_truncated(&self, lower_bound: f64, rng_seed: u64) -> f64 {
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

pub(super) fn impute_censoring_times(
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
    let rng_seed = seed.unwrap_or(crate::constants::DEFAULT_RANDOM_SEED);

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
