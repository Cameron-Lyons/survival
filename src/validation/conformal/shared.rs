use super::*;

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
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
#[pyclass(from_py_object)]
pub struct ConformalPredictionResult {
    #[pyo3(get)]
    pub lower_predictive_bound: Vec<f64>,
    #[pyo3(get)]
    pub predicted_time: Vec<f64>,
    #[pyo3(get)]
    pub coverage_level: f64,
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
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

pub(super) fn weighted_quantile(values: &[f64], weights: &[f64], q: f64) -> f64 {
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

pub(super) fn compute_km_censoring_survival(time: &[f64], status: &[i32]) -> Vec<f64> {
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
        while i < n && (time[indices[i]] - current_time).abs() < crate::constants::TIME_EPSILON {
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

pub(super) fn compute_conformity_scores(
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
