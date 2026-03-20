

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]

pub struct DynamicPredictionResult {
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub survival_mean: Vec<f64>,
    #[pyo3(get)]
    pub survival_lower: Vec<f64>,
    #[pyo3(get)]
    pub survival_upper: Vec<f64>,
    #[pyo3(get)]
    pub cumulative_risk: Vec<f64>,
    #[pyo3(get)]
    pub conditional_survival: Vec<f64>,
    #[pyo3(get)]
    pub auc: f64,
    #[pyo3(get)]
    pub brier_score: f64,
}

#[pyfunction]
#[pyo3(signature = (
    beta_long,
    gamma_surv,
    alpha,
    random_effects,
    baseline_hazard,
    baseline_times,
    y_history,
    times_history,
    x_long_fixed,
    n_history,
    n_long_vars,
    x_surv,
    n_surv_vars,
    landmark_time,
    prediction_times,
    n_monte_carlo=500
))]
#[allow(clippy::too_many_arguments)]
pub fn dynamic_prediction(
    beta_long: Vec<f64>,
    gamma_surv: Vec<f64>,
    alpha: f64,
    random_effects: Vec<f64>,
    baseline_hazard: Vec<f64>,
    baseline_times: Vec<f64>,
    y_history: Vec<f64>,
    times_history: Vec<f64>,
    x_long_fixed: Vec<f64>,
    n_history: usize,
    n_long_vars: usize,
    x_surv: Vec<f64>,
    n_surv_vars: usize,
    landmark_time: f64,
    prediction_times: Vec<f64>,
    n_monte_carlo: usize,
) -> PyResult<DynamicPredictionResult> {
    if y_history.len() != times_history.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "y_history and times_history must have the same length",
        ));
    }
    if n_history != y_history.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n_history must match y_history length",
        ));
    }
    if n_surv_vars != x_surv.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n_surv_vars must match x_surv length",
        ));
    }
    if baseline_hazard.len() != baseline_times.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "baseline_hazard and baseline_times must have the same length",
        ));
    }
    if n_monte_carlo == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n_monte_carlo must be > 0",
        ));
    }
    if n_long_vars > x_long_fixed.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n_long_vars cannot exceed x_long_fixed length",
        ));
    }
    let b0 = random_effects.first().copied().unwrap_or(0.0);
    let b1 = random_effects.get(1).copied().unwrap_or(0.0);

    let prediction_times_filtered: Vec<f64> = prediction_times
        .into_iter()
        .filter(|&t| t > landmark_time)
        .collect();

    let n_times = prediction_times_filtered.len();

    let survival_samples: Vec<Vec<f64>> = (0..n_monte_carlo)
        .into_par_iter()
        .map(|mc_idx| {
            let mut rng = fastrand::Rng::with_seed(mc_idx as u64);

            let b0_sample = b0 + 0.1 * sample_normal(&mut rng);
            let b1_sample = b1 + 0.05 * sample_normal(&mut rng);

            prediction_times_filtered
                .iter()
                .map(|&t| {
                    let mut eta = 0.0;
                    for (k, &xk) in x_surv.iter().enumerate() {
                        if k < gamma_surv.len() {
                            eta += gamma_surv[k] * xk;
                        }
                    }

                    let mut m_t = b0_sample + b1_sample * t;
                    let x_avg: Vec<f64> = (0..n_long_vars)
                        .map(|j| {
                            (0..n_history)
                                .map(|i| x_long_fixed[i * n_long_vars + j])
                                .sum::<f64>()
                                / n_history.max(1) as f64
                        })
                        .collect();

                    for (j, &xj) in x_avg.iter().enumerate() {
                        if j < beta_long.len() {
                            m_t += beta_long[j] * xj;
                        }
                    }

                    eta += alpha * m_t;

                    let mut cum_hazard = 0.0;
                    for (t_idx, &bt) in baseline_times.iter().enumerate() {
                        if bt > landmark_time && bt <= t && t_idx < baseline_hazard.len() {
                            cum_hazard += baseline_hazard[t_idx] * eta.exp();
                        }
                    }

                    (-cum_hazard).exp()
                })
                .collect()
        })
        .collect();

    let survival_mean: Vec<f64> = (0..n_times)
        .map(|t| survival_samples.iter().map(|s| s[t]).sum::<f64>() / n_monte_carlo as f64)
        .collect();

    let survival_lower: Vec<f64> = (0..n_times)
        .map(|t| {
            let mut vals: Vec<f64> = survival_samples.iter().map(|s| s[t]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            vals[(n_monte_carlo as f64 * 0.025) as usize]
        })
        .collect();

    let survival_upper: Vec<f64> = (0..n_times)
        .map(|t| {
            let mut vals: Vec<f64> = survival_samples.iter().map(|s| s[t]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            vals[(n_monte_carlo as f64 * 0.975) as usize]
        })
        .collect();

    let cumulative_risk: Vec<f64> = survival_mean.iter().map(|&s| 1.0 - s).collect();

    let s_landmark = if !survival_mean.is_empty() {
        survival_mean[0]
    } else {
        1.0
    };
    let conditional_survival: Vec<f64> = survival_mean
        .iter()
        .map(|&s| if s_landmark > 0.0 { s / s_landmark } else { s })
        .collect();

    Ok(DynamicPredictionResult {
        time_points: prediction_times_filtered,
        survival_mean,
        survival_lower,
        survival_upper,
        cumulative_risk,
        conditional_survival,
        auc: 0.0,
        brier_score: 0.0,
    })
}

#[pyfunction]
#[pyo3(signature = (
    beta_long,
    gamma_surv,
    alpha,
    baseline_hazard,
    baseline_times,
    y_observed,
    times_observed,
    x_long_fixed,
    n_obs,
    n_long_vars,
    x_surv,
    n_surv_vars,
    event_time,
    event_status,
    horizon
))]
#[allow(clippy::too_many_arguments)]
pub fn dynamic_auc(
    beta_long: Vec<f64>,
    gamma_surv: Vec<f64>,
    alpha: f64,
    baseline_hazard: Vec<f64>,
    baseline_times: Vec<f64>,
    y_observed: Vec<f64>,
    times_observed: Vec<f64>,
    x_long_fixed: Vec<f64>,
    n_obs: usize,
    n_long_vars: usize,
    x_surv: Vec<f64>,
    n_surv_vars: usize,
    event_time: Vec<f64>,
    event_status: Vec<i32>,
    horizon: f64,
) -> PyResult<f64> {
    if baseline_hazard.len() != baseline_times.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "baseline_hazard and baseline_times must have the same length",
        ));
    }
    if !y_observed.is_empty() && y_observed.len() != times_observed.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "y_observed and times_observed must have the same length when y_observed is provided",
        ));
    }
    let n_subjects = event_time.len();
    if event_status.len() != n_subjects {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "event_time and event_status must have the same length",
        ));
    }
    if n_obs != n_subjects {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n_obs must match number of subjects in event_time",
        ));
    }
    if n_surv_vars == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n_surv_vars must be > 0",
        ));
    }
    if x_surv.len() != n_subjects * n_surv_vars {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x_surv length must be n_obs * n_surv_vars",
        ));
    }
    if n_long_vars > 0 && x_long_fixed.len() < n_subjects * n_long_vars {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x_long_fixed length must be at least n_obs * n_long_vars",
        ));
    }

    let risk_scores: Vec<f64> = (0..n_subjects)
        .map(|i| {
            let mut eta = 0.0;
            for (k, &xk) in x_surv[i * n_surv_vars..(i + 1) * n_surv_vars]
                .iter()
                .enumerate()
            {
                if k < gamma_surv.len() {
                    eta += gamma_surv[k] * xk;
                }
            }

            let mut m_t = 0.0;

            for (j, &bj) in beta_long.iter().enumerate() {
                if j < n_long_vars && i * n_long_vars + j < x_long_fixed.len() {
                    m_t += bj * x_long_fixed[i * n_long_vars + j];
                }
            }

            eta += alpha * m_t;
            eta
        })
        .collect();

    let auc =
        concordance_index_with_horizon(&risk_scores, &event_time, &event_status, Some(horizon));

    Ok(auc)
}

#[pyfunction]
#[pyo3(signature = (
    survival_predictions,
    event_time,
    event_status,
    prediction_times
))]
pub fn dynamic_brier_score(
    survival_predictions: Vec<Vec<f64>>,
    event_time: Vec<f64>,
    event_status: Vec<i32>,
    prediction_times: Vec<f64>,
) -> PyResult<Vec<f64>> {
    let n_subjects = event_time.len();
    let n_times = prediction_times.len();

    let brier_scores: Vec<f64> = (0..n_times)
        .map(|t_idx| {
            let t = prediction_times[t_idx];
            let mut score_sum = 0.0;
            let mut weight_sum = 0.0;

            for i in 0..n_subjects {
                let pred = if t_idx < survival_predictions[i].len() {
                    survival_predictions[i][t_idx]
                } else {
                    0.5
                };

                let outcome = if event_time[i] <= t && event_status[i] == 1 {
                    0.0
                } else if event_time[i] > t {
                    1.0
                } else {
                    continue;
                };

                score_sum += (pred - outcome).powi(2);
                weight_sum += 1.0;
            }

            if weight_sum > 0.0 {
                score_sum / weight_sum
            } else {
                0.0
            }
        })
        .collect();

    Ok(brier_scores)
}
