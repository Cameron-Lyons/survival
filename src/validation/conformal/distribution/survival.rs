use super::super::*;

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
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
