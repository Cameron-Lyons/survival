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
    crate::internal::validation::validate_length(n_cal, cal_event.len(), "cal_event")?;
    crate::internal::validation::validate_length(n_cal, cal_predictions.len(), "cal_predictions")?;
    crate::internal::validation::validate_finite(&cal_time, "cal_time")?;
    crate::internal::validation::validate_non_negative(&cal_time, "cal_time")?;
    crate::internal::validation::validate_binary_i32(&cal_event, "cal_event")?;
    validate_finite_vector(&cal_predictions, "cal_predictions")?;
    validate_finite_vector(&test_predictions, "test_predictions")?;
    validate_probability_open(config.alpha, "alpha")?;
    validate_conformal_method(&config.method)?;

    if config.n_bootstrap == 0 {
        return Err(uncertainty_value_error("n_bootstrap must be positive"));
    }
    if config.method == "censoring_adjusted" && !cal_time.iter().any(|&time| time > 0.0) {
        return Err(uncertainty_value_error(
            "cal_time must contain a positive value for censoring_adjusted method",
        ));
    }

    let scores = compute_conformity_scores(&cal_time, &cal_event, &cal_predictions, &config.method);

    let quantile_idx = ((n_cal as f64 + 1.0) * (1.0 - config.alpha)).ceil() as usize;
    let quantile_idx = quantile_idx.min(scores.len()).saturating_sub(1);

    let mut sorted_scores = scores.clone();
    sorted_scores.sort_by(f64::total_cmp);
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
    indices.sort_by(|&a, &b| time[a].total_cmp(&time[b]));

    let mut eval_order: Vec<usize> = (0..eval_times.len()).collect();
    eval_order.sort_by(|&a, &b| eval_times[a].total_cmp(&eval_times[b]));

    let mut survival = vec![1.0; eval_times.len()];
    let mut surv_prob = 1.0;
    let mut at_risk_weight: f64 = weights.iter().sum();
    let mut eval_cursor = 0usize;
    let mut group_start = 0usize;

    while group_start < indices.len() {
        let current_time = time[indices[group_start]];

        while eval_cursor < eval_order.len() {
            let eval_idx = eval_order[eval_cursor];
            let eval_t = eval_times[eval_idx];
            if eval_t < current_time && !crate::constants::same_time(eval_t, current_time) {
                survival[eval_idx] = surv_prob;
                eval_cursor += 1;
            } else {
                break;
            }
        }

        let mut group_end = group_start + 1;
        while group_end < indices.len()
            && crate::constants::same_time(time[indices[group_end]], current_time)
        {
            group_end += 1;
        }

        let mut event_weight = 0.0;
        let mut group_weight = 0.0;
        for &idx in &indices[group_start..group_end] {
            let w = weights[idx];
            group_weight += w;
            if event[idx] == 1 {
                event_weight += w;
            }
        }

        if at_risk_weight > 0.0 && event_weight > 0.0 {
            surv_prob *= 1.0 - (event_weight / at_risk_weight).min(1.0);
        }

        while eval_cursor < eval_order.len() {
            let eval_idx = eval_order[eval_cursor];
            if crate::constants::same_time(eval_times[eval_idx], current_time) {
                survival[eval_idx] = surv_prob;
                eval_cursor += 1;
            } else {
                break;
            }
        }

        at_risk_weight = (at_risk_weight - group_weight).max(0.0);
        group_start = group_end;
    }

    for &eval_idx in &eval_order[eval_cursor..] {
        survival[eval_idx] = surv_prob;
    }

    survival
}

fn validate_bayesian_bootstrap_inputs(
    time: &[f64],
    event: &[i32],
    eval_times: &[f64],
    config: &BayesianBootstrapConfig,
) -> PyResult<()> {
    crate::internal::validation::validate_non_empty(time, "time")?;
    crate::internal::validation::validate_length(time.len(), event.len(), "event")?;
    crate::internal::validation::validate_non_empty(eval_times, "eval_times")?;
    crate::internal::validation::validate_finite(time, "time")?;
    crate::internal::validation::validate_non_negative(time, "time")?;
    crate::internal::validation::validate_finite(eval_times, "eval_times")?;
    crate::internal::validation::validate_non_negative(eval_times, "eval_times")?;
    crate::internal::validation::validate_binary_i32(event, "event")?;
    crate::internal::validation::validate_confidence_level(config.confidence_level)?;

    if config.n_bootstrap == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_bootstrap must be positive",
        ));
    }

    Ok(())
}

fn compute_quantile_posterior(samples: &[Vec<f64>], q: f64) -> Vec<f64> {
    if samples.is_empty() || samples[0].is_empty() {
        return vec![];
    }

    let n_times = samples[0].len();
    let mut result = vec![0.0; n_times];

    for t in 0..n_times {
        let mut values: Vec<f64> = samples.iter().map(|s| s[t]).collect();
        values.sort_by(f64::total_cmp);

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
    validate_bayesian_bootstrap_inputs(&time, &event, &eval_times, &config)?;

    let mut rng = if let Some(seed) = config.seed {
        fastrand::Rng::with_seed(seed)
    } else {
        fastrand::Rng::new()
    };

    let n_times = eval_times.len();
    let mut posterior_samples: Vec<Vec<f64>> = vec![vec![0.0; n_times]; config.n_bootstrap];

    for sample in posterior_samples.iter_mut() {
        let weights = generate_dirichlet_weights(time.len(), &mut rng);
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
        values.sort_by(f64::total_cmp);

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
    indices.sort_by(|&a, &b| time[b].total_cmp(&time[a]));

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

#[inline]
fn compute_kernel_weight(target: &[f64], reference: &[f64]) -> f64 {
    let bandwidth = 1.0;
    let dist_sq: f64 = target
        .iter()
        .zip(reference.iter())
        .map(|(t, r)| (t - r).powi(2))
        .sum();
    (-dist_sq / (2.0 * bandwidth * bandwidth)).exp()
}

fn kernel_weighted_prediction(target: &[f64], reference: &[Vec<f64>], predictions: &[f64]) -> f64 {
    let mut weighted_sum = 0.0;
    let mut weight_sum = 0.0;

    for (pred, ref_cov) in predictions.iter().zip(reference.iter()) {
        let weight = compute_kernel_weight(target, ref_cov);
        weighted_sum += pred * weight;
        weight_sum += weight;
    }

    if weight_sum > 0.0 {
        weighted_sum / weight_sum
    } else {
        predictions.first().copied().unwrap_or(0.0)
    }
}
