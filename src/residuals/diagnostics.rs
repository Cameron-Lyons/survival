#![allow(
    unused_variables,
    unused_imports,
    unused_mut,
    unused_assignments,
    dead_code,
    clippy::too_many_arguments,
    clippy::needless_range_loop
)]

use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utilities::statistical::{lower_incomplete_gamma, normal_cdf};

#[derive(Debug, Clone)]
#[pyclass]
pub struct DfbetaResult {
    #[pyo3(get)]
    pub dfbeta: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub dfbetas: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub max_dfbeta: Vec<f64>,
    #[pyo3(get)]
    pub influential_obs: Vec<usize>,
    #[pyo3(get)]
    pub n_obs: usize,
    #[pyo3(get)]
    pub n_vars: usize,
}

#[pyfunction]
#[pyo3(signature = (time, event, covariates, n_covariates, coefficients, threshold=None))]
pub fn dfbeta_cox(
    time: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<f64>,
    n_covariates: usize,
    coefficients: Vec<f64>,
    threshold: Option<f64>,
) -> PyResult<DfbetaResult> {
    let n = time.len();
    if event.len() != n || covariates.len() != n * n_covariates {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input dimensions mismatch",
        ));
    }
    if coefficients.len() != n_covariates {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "coefficients must have length n_covariates",
        ));
    }

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| time[b].partial_cmp(&time[a]).unwrap());

    let eta: Vec<f64> = (0..n)
        .map(|i| {
            let mut e = 0.0;
            for j in 0..n_covariates {
                e += covariates[i * n_covariates + j] * coefficients[j];
            }
            e.clamp(-500.0, 500.0)
        })
        .collect();

    let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

    let info_inv = compute_information_inverse(
        &time,
        &event,
        &covariates,
        n_covariates,
        &exp_eta,
        &sorted_indices,
    );

    let score_residuals = compute_score_residuals(
        &time,
        &event,
        &covariates,
        n_covariates,
        &exp_eta,
        &sorted_indices,
    );

    let mut dfbeta: Vec<Vec<f64>> = vec![vec![0.0; n_covariates]; n];
    for i in 0..n {
        for j in 0..n_covariates {
            for k in 0..n_covariates {
                dfbeta[i][j] += info_inv[j * n_covariates + k] * score_residuals[i][k];
            }
        }
    }

    let coef_se: Vec<f64> = (0..n_covariates)
        .map(|j| info_inv[j * n_covariates + j].sqrt().max(1e-10))
        .collect();

    let mut dfbetas: Vec<Vec<f64>> = vec![vec![0.0; n_covariates]; n];
    for i in 0..n {
        for j in 0..n_covariates {
            dfbetas[i][j] = dfbeta[i][j] / coef_se[j];
        }
    }

    let max_dfbeta: Vec<f64> = (0..n_covariates)
        .map(|j| {
            dfbeta
                .iter()
                .map(|row| row[j].abs())
                .fold(0.0_f64, f64::max)
        })
        .collect();

    let thresh = threshold.unwrap_or(2.0 / (n as f64).sqrt());
    let influential_obs: Vec<usize> = (0..n)
        .filter(|&i| dfbetas[i].iter().any(|&d| d.abs() > thresh))
        .collect();

    Ok(DfbetaResult {
        dfbeta,
        dfbetas,
        max_dfbeta,
        influential_obs,
        n_obs: n,
        n_vars: n_covariates,
    })
}

fn compute_information_inverse(
    time: &[f64],
    event: &[i32],
    covariates: &[f64],
    n_covariates: usize,
    exp_eta: &[f64],
    sorted_indices: &[usize],
) -> Vec<f64> {
    let n = time.len();
    let mut info = vec![0.0; n_covariates * n_covariates];

    let mut risk_sum = 0.0;
    let mut weighted_x = vec![0.0; n_covariates];
    let mut weighted_xx = vec![0.0; n_covariates * n_covariates];

    for &i in sorted_indices {
        risk_sum += exp_eta[i];
        for j in 0..n_covariates {
            weighted_x[j] += exp_eta[i] * covariates[i * n_covariates + j];
            for k in 0..n_covariates {
                weighted_xx[j * n_covariates + k] += exp_eta[i]
                    * covariates[i * n_covariates + j]
                    * covariates[i * n_covariates + k];
            }
        }

        if event[i] == 1 && risk_sum > 0.0 {
            for j in 0..n_covariates {
                let x_bar_j = weighted_x[j] / risk_sum;
                for k in 0..n_covariates {
                    let x_bar_k = weighted_x[k] / risk_sum;
                    let xx_bar = weighted_xx[j * n_covariates + k] / risk_sum;
                    info[j * n_covariates + k] += xx_bar - x_bar_j * x_bar_k;
                }
            }
        }
    }

    invert_matrix(&info, n_covariates)
}

fn compute_score_residuals(
    time: &[f64],
    event: &[i32],
    covariates: &[f64],
    n_covariates: usize,
    exp_eta: &[f64],
    sorted_indices: &[usize],
) -> Vec<Vec<f64>> {
    let n = time.len();
    let mut score_resid: Vec<Vec<f64>> = vec![vec![0.0; n_covariates]; n];

    let mut risk_sum = 0.0;
    let mut weighted_x = vec![0.0; n_covariates];

    let mut cumulative_term = vec![vec![0.0; n_covariates]; n];

    for (idx, &i) in sorted_indices.iter().enumerate() {
        risk_sum += exp_eta[i];
        for j in 0..n_covariates {
            weighted_x[j] += exp_eta[i] * covariates[i * n_covariates + j];
        }

        if event[i] == 1 && risk_sum > 0.0 {
            for k in 0..n {
                if time[k] >= time[i] {
                    let x_bar = weighted_x.iter().map(|&w| w / risk_sum).collect::<Vec<_>>();
                    for j in 0..n_covariates {
                        cumulative_term[k][j] +=
                            exp_eta[k] * (covariates[k * n_covariates + j] - x_bar[j]) / risk_sum;
                    }
                }
            }
        }
    }

    for i in 0..n {
        for j in 0..n_covariates {
            if event[i] == 1 {
                score_resid[i][j] = covariates[i * n_covariates + j];
            }
            score_resid[i][j] -= cumulative_term[i][j];
        }
    }

    score_resid
}

fn invert_matrix(a: &[f64], n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![if a[0].abs() > 1e-10 { 1.0 / a[0] } else { 0.0 }];
    }

    let mut aug = vec![0.0; n * 2 * n];

    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = a[i * n + j];
        }
        aug[i * 2 * n + n + i] = 1.0;
    }

    for i in 0..n {
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[k * 2 * n + i].abs() > aug[max_row * 2 * n + i].abs() {
                max_row = k;
            }
        }

        for j in 0..(2 * n) {
            aug.swap(i * 2 * n + j, max_row * 2 * n + j);
        }

        let pivot = aug[i * 2 * n + i];
        if pivot.abs() < 1e-10 {
            continue;
        }

        for j in 0..(2 * n) {
            aug[i * 2 * n + j] /= pivot;
        }

        for k in 0..n {
            if k != i {
                let factor = aug[k * 2 * n + i];
                for j in 0..(2 * n) {
                    aug[k * 2 * n + j] -= factor * aug[i * 2 * n + j];
                }
            }
        }
    }

    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * 2 * n + n + j];
        }
    }

    inv
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct LeverageResult {
    #[pyo3(get)]
    pub leverage: Vec<f64>,
    #[pyo3(get)]
    pub lmax: Vec<f64>,
    #[pyo3(get)]
    pub mean_leverage: f64,
    #[pyo3(get)]
    pub high_leverage_obs: Vec<usize>,
    #[pyo3(get)]
    pub n_obs: usize,
}

#[pyfunction]
#[pyo3(signature = (time, event, covariates, n_covariates, coefficients, threshold_multiplier=2.0))]
pub fn leverage_cox(
    time: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<f64>,
    n_covariates: usize,
    coefficients: Vec<f64>,
    threshold_multiplier: f64,
) -> PyResult<LeverageResult> {
    let n = time.len();
    if event.len() != n || covariates.len() != n * n_covariates {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input dimensions mismatch",
        ));
    }

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| time[b].partial_cmp(&time[a]).unwrap());

    let eta: Vec<f64> = (0..n)
        .map(|i| {
            let mut e = 0.0;
            for j in 0..n_covariates {
                e += covariates[i * n_covariates + j] * coefficients[j];
            }
            e.clamp(-500.0, 500.0)
        })
        .collect();

    let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

    let info_inv = compute_information_inverse(
        &time,
        &event,
        &covariates,
        n_covariates,
        &exp_eta,
        &sorted_indices,
    );

    let mut leverage = vec![0.0; n];
    let mut lmax = vec![0.0; n];

    let mut risk_sum = 0.0;
    let mut weighted_x = vec![0.0; n_covariates];

    for &i in &sorted_indices {
        risk_sum += exp_eta[i];
        for j in 0..n_covariates {
            weighted_x[j] += exp_eta[i] * covariates[i * n_covariates + j];
        }

        if risk_sum > 0.0 {
            let x_bar: Vec<f64> = weighted_x.iter().map(|&w| w / risk_sum).collect();

            for k in 0..n {
                if time[k] >= time[i] {
                    let mut h_ik = 0.0;
                    for j1 in 0..n_covariates {
                        let x_diff1 = covariates[k * n_covariates + j1] - x_bar[j1];
                        for j2 in 0..n_covariates {
                            let x_diff2 = covariates[k * n_covariates + j2] - x_bar[j2];
                            h_ik += x_diff1 * info_inv[j1 * n_covariates + j2] * x_diff2;
                        }
                    }
                    h_ik *= exp_eta[k] / risk_sum;

                    if event[i] == 1 {
                        leverage[k] += h_ik;
                    }
                }
            }
        }
    }

    for i in 0..n {
        let mut max_contrib: f64 = 0.0;
        for j in 0..n_covariates {
            let x_j = covariates[i * n_covariates + j];
            let contrib = x_j.abs() * info_inv[j * n_covariates + j].sqrt();
            max_contrib = max_contrib.max(contrib);
        }
        lmax[i] = max_contrib;
    }

    let mean_leverage = leverage.iter().sum::<f64>() / n as f64;
    let threshold = threshold_multiplier * (n_covariates as f64) / (n as f64);

    let high_leverage_obs: Vec<usize> = (0..n).filter(|&i| leverage[i] > threshold).collect();

    Ok(LeverageResult {
        leverage,
        lmax,
        mean_leverage,
        high_leverage_obs,
        n_obs: n,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct SchoenfeldSmoothResult {
    #[pyo3(get)]
    pub times: Vec<f64>,
    #[pyo3(get)]
    pub smoothed_residuals: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub coefficient_path: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub slope_test_stats: Vec<f64>,
    #[pyo3(get)]
    pub slope_p_values: Vec<f64>,
    #[pyo3(get)]
    pub non_proportional_vars: Vec<usize>,
    #[pyo3(get)]
    pub n_events: usize,
    #[pyo3(get)]
    pub n_vars: usize,
}

#[pyfunction]
#[pyo3(signature = (event_times, schoenfeld_residuals, n_covariates, coefficients, bandwidth=None, transform="identity"))]
pub fn smooth_schoenfeld(
    event_times: Vec<f64>,
    schoenfeld_residuals: Vec<f64>,
    n_covariates: usize,
    coefficients: Vec<f64>,
    bandwidth: Option<f64>,
    transform: &str,
) -> PyResult<SchoenfeldSmoothResult> {
    let n_events = event_times.len();
    if schoenfeld_residuals.len() != n_events * n_covariates {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "schoenfeld_residuals must have length n_events * n_covariates",
        ));
    }

    let transformed_times: Vec<f64> = match transform.to_lowercase().as_str() {
        "identity" => event_times.clone(),
        "log" => event_times.iter().map(|&t| (t.max(1e-10)).ln()).collect(),
        "km" => {
            let mut km = vec![0.0; n_events];
            let mut n_risk = n_events as f64;
            for i in 0..n_events {
                km[i] = 1.0 - 1.0 / n_risk;
                n_risk -= 1.0;
            }
            km
        }
        "rank" => {
            let mut ranks: Vec<f64> = (1..=n_events)
                .map(|i| i as f64 / (n_events as f64 + 1.0))
                .collect();
            ranks
        }
        _ => event_times.clone(),
    };

    let h = bandwidth.unwrap_or_else(|| {
        let time_range = transformed_times
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
            - transformed_times
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
        0.2 * time_range
    });

    let mut smoothed: Vec<Vec<f64>> = vec![vec![0.0; n_covariates]; n_events];
    let mut coefficient_path: Vec<Vec<f64>> = vec![vec![0.0; n_covariates]; n_events];

    for i in 0..n_events {
        let t_i = transformed_times[i];

        for j in 0..n_covariates {
            let mut sum_w = 0.0;
            let mut sum_wy = 0.0;
            let mut sum_wty = 0.0;
            let mut sum_wt = 0.0;
            let mut sum_wtt = 0.0;

            for k in 0..n_events {
                let t_k = transformed_times[k];
                let diff = (t_i - t_k) / h;
                let w = (-0.5 * diff * diff).exp();

                let y = schoenfeld_residuals[k * n_covariates + j];

                sum_w += w;
                sum_wy += w * y;
                sum_wty += w * t_k * y;
                sum_wt += w * t_k;
                sum_wtt += w * t_k * t_k;
            }

            if sum_w > 1e-10 {
                let y_mean = sum_wy / sum_w;
                smoothed[i][j] = y_mean;

                coefficient_path[i][j] = coefficients[j] + y_mean;
            }
        }
    }

    let mut slope_test_stats = vec![0.0; n_covariates];
    let mut slope_p_values = vec![0.0; n_covariates];

    for j in 0..n_covariates {
        let (slope, se_slope) =
            compute_slope_test(&transformed_times, &schoenfeld_residuals, j, n_covariates);

        if se_slope > 1e-10 {
            let z = slope / se_slope;
            slope_test_stats[j] = z;
            slope_p_values[j] = 2.0 * (1.0 - normal_cdf(z.abs()));
        }
    }

    let non_proportional_vars: Vec<usize> = (0..n_covariates)
        .filter(|&j| slope_p_values[j] < 0.05)
        .collect();

    Ok(SchoenfeldSmoothResult {
        times: transformed_times,
        smoothed_residuals: smoothed,
        coefficient_path,
        slope_test_stats,
        slope_p_values,
        non_proportional_vars,
        n_events,
        n_vars: n_covariates,
    })
}

fn compute_slope_test(
    times: &[f64],
    residuals: &[f64],
    var_idx: usize,
    n_covariates: usize,
) -> (f64, f64) {
    let n = times.len();

    let mean_t: f64 = times.iter().sum::<f64>() / n as f64;
    let mean_r: f64 = (0..n)
        .map(|i| residuals[i * n_covariates + var_idx])
        .sum::<f64>()
        / n as f64;

    let mut sum_tt = 0.0;
    let mut sum_tr = 0.0;
    let mut sum_rr = 0.0;

    for i in 0..n {
        let t_diff = times[i] - mean_t;
        let r_diff = residuals[i * n_covariates + var_idx] - mean_r;
        sum_tt += t_diff * t_diff;
        sum_tr += t_diff * r_diff;
        sum_rr += r_diff * r_diff;
    }

    let slope = if sum_tt > 1e-10 { sum_tr / sum_tt } else { 0.0 };

    let residual_var = if n > 2 && sum_tt > 1e-10 {
        (sum_rr - slope * slope * sum_tt) / (n - 2) as f64
    } else {
        0.0
    };

    let se_slope = if sum_tt > 1e-10 {
        (residual_var / sum_tt).sqrt()
    } else {
        0.0
    };

    (slope, se_slope)
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct OutlierDetectionResult {
    #[pyo3(get)]
    pub martingale_residuals: Vec<f64>,
    #[pyo3(get)]
    pub deviance_residuals: Vec<f64>,
    #[pyo3(get)]
    pub standardized_deviance: Vec<f64>,
    #[pyo3(get)]
    pub outlier_indices: Vec<usize>,
    #[pyo3(get)]
    pub extreme_survivor_indices: Vec<usize>,
    #[pyo3(get)]
    pub outlier_scores: Vec<f64>,
    #[pyo3(get)]
    pub threshold: f64,
    #[pyo3(get)]
    pub n_outliers: usize,
}

#[pyfunction]
#[pyo3(signature = (time, event, covariates, n_covariates, coefficients, outlier_threshold=3.0))]
pub fn outlier_detection_cox(
    time: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<f64>,
    n_covariates: usize,
    coefficients: Vec<f64>,
    outlier_threshold: f64,
) -> PyResult<OutlierDetectionResult> {
    let n = time.len();
    if event.len() != n || covariates.len() != n * n_covariates {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input dimensions mismatch",
        ));
    }

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| time[b].partial_cmp(&time[a]).unwrap());

    let eta: Vec<f64> = (0..n)
        .map(|i| {
            let mut e = 0.0;
            for j in 0..n_covariates {
                e += covariates[i * n_covariates + j] * coefficients[j];
            }
            e.clamp(-500.0, 500.0)
        })
        .collect();

    let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

    let mut cumulative_hazard = vec![0.0; n];
    let mut risk_sum = 0.0;

    for &i in &sorted_indices {
        risk_sum += exp_eta[i];

        if event[i] == 1 && risk_sum > 0.0 {
            let h0_increment = 1.0 / risk_sum;

            for k in 0..n {
                if time[k] >= time[i] {
                    cumulative_hazard[k] += exp_eta[k] * h0_increment;
                }
            }
        }
    }

    let martingale_residuals: Vec<f64> = (0..n)
        .map(|i| event[i] as f64 - cumulative_hazard[i])
        .collect();

    let deviance_residuals: Vec<f64> = (0..n)
        .map(|i| {
            let m = martingale_residuals[i];
            let d = event[i] as f64;
            let sign = if m >= 0.0 { 1.0 } else { -1.0 };

            let dev_sq = -2.0 * (m + d * (d - m).max(1e-10).ln());
            sign * dev_sq.max(0.0).sqrt()
        })
        .collect();

    let mean_dev: f64 = deviance_residuals.iter().sum::<f64>() / n as f64;
    let std_dev: f64 = (deviance_residuals
        .iter()
        .map(|&d| (d - mean_dev).powi(2))
        .sum::<f64>()
        / (n - 1) as f64)
        .sqrt()
        .max(1e-10);

    let standardized_deviance: Vec<f64> = deviance_residuals
        .iter()
        .map(|&d| (d - mean_dev) / std_dev)
        .collect();

    let outlier_indices: Vec<usize> = (0..n)
        .filter(|&i| standardized_deviance[i].abs() > outlier_threshold)
        .collect();

    let extreme_survivor_indices: Vec<usize> = (0..n)
        .filter(|&i| event[i] == 0 && martingale_residuals[i] < -2.0)
        .collect();

    let outlier_scores: Vec<f64> = standardized_deviance.iter().map(|&d| d.abs()).collect();

    Ok(OutlierDetectionResult {
        martingale_residuals,
        deviance_residuals,
        standardized_deviance,
        outlier_indices: outlier_indices.clone(),
        extreme_survivor_indices,
        outlier_scores,
        threshold: outlier_threshold,
        n_outliers: outlier_indices.len(),
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct ModelInfluenceResult {
    #[pyo3(get)]
    pub cooks_distance: Vec<f64>,
    #[pyo3(get)]
    pub covratio: Vec<f64>,
    #[pyo3(get)]
    pub dffits: Vec<f64>,
    #[pyo3(get)]
    pub likelihood_displacement: Vec<f64>,
    #[pyo3(get)]
    pub influential_by_cooks: Vec<usize>,
    #[pyo3(get)]
    pub influential_by_covratio: Vec<usize>,
    #[pyo3(get)]
    pub influential_by_dffits: Vec<usize>,
    #[pyo3(get)]
    pub overall_influential: Vec<usize>,
    #[pyo3(get)]
    pub n_obs: usize,
}

#[pyfunction]
#[pyo3(signature = (time, event, covariates, n_covariates, coefficients))]
pub fn model_influence_cox(
    time: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<f64>,
    n_covariates: usize,
    coefficients: Vec<f64>,
) -> PyResult<ModelInfluenceResult> {
    let n = time.len();
    if event.len() != n || covariates.len() != n * n_covariates {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input dimensions mismatch",
        ));
    }

    let dfbeta_result = dfbeta_cox(
        time.clone(),
        event.clone(),
        covariates.clone(),
        n_covariates,
        coefficients.clone(),
        None,
    )?;

    let leverage_result = leverage_cox(
        time.clone(),
        event.clone(),
        covariates.clone(),
        n_covariates,
        coefficients.clone(),
        2.0,
    )?;

    let coef_var = estimate_coefficient_variance(&dfbeta_result.dfbeta, n_covariates);

    let mut cooks_distance = vec![0.0; n];
    for i in 0..n {
        let mut cook = 0.0;
        for j in 0..n_covariates {
            if coef_var[j] > 1e-10 {
                cook += dfbeta_result.dfbeta[i][j].powi(2) / coef_var[j];
            }
        }
        cooks_distance[i] = cook / n_covariates as f64;
    }

    let mut covratio = vec![1.0; n];
    for i in 0..n {
        let h_i = leverage_result.leverage[i];
        if h_i < 1.0 {
            covratio[i] = 1.0 / (1.0 - h_i).powf(n_covariates as f64);
        }
    }

    let mut dffits = vec![0.0; n];
    for i in 0..n {
        let h_i = leverage_result.leverage[i];
        if h_i > 0.0 && h_i < 1.0 {
            let sum_dfbeta_sq: f64 = dfbeta_result.dfbeta[i].iter().map(|&d| d * d).sum();
            dffits[i] = sum_dfbeta_sq.sqrt() * (h_i / (1.0 - h_i)).sqrt();
        }
    }

    let mut likelihood_displacement = vec![0.0; n];
    for i in 0..n {
        likelihood_displacement[i] = cooks_distance[i] * n_covariates as f64;
    }

    let cooks_threshold = 4.0 / n as f64;
    let covratio_threshold = 1.0 + 3.0 * n_covariates as f64 / n as f64;
    let dffits_threshold = 2.0 * ((n_covariates as f64 + 1.0) / n as f64).sqrt();

    let influential_by_cooks: Vec<usize> = (0..n)
        .filter(|&i| cooks_distance[i] > cooks_threshold)
        .collect();

    let influential_by_covratio: Vec<usize> = (0..n)
        .filter(|&i| covratio[i] > covratio_threshold || covratio[i] < 1.0 / covratio_threshold)
        .collect();

    let influential_by_dffits: Vec<usize> = (0..n)
        .filter(|&i| dffits[i].abs() > dffits_threshold)
        .collect();

    let mut overall_influential: Vec<usize> = influential_by_cooks
        .iter()
        .chain(influential_by_covratio.iter())
        .chain(influential_by_dffits.iter())
        .cloned()
        .collect();
    overall_influential.sort_unstable();
    overall_influential.dedup();

    Ok(ModelInfluenceResult {
        cooks_distance,
        covratio,
        dffits,
        likelihood_displacement,
        influential_by_cooks,
        influential_by_covratio,
        influential_by_dffits,
        overall_influential,
        n_obs: n,
    })
}

fn estimate_coefficient_variance(dfbeta: &[Vec<f64>], n_covariates: usize) -> Vec<f64> {
    let n = dfbeta.len();
    let mut var = vec![0.0; n_covariates];

    for j in 0..n_covariates {
        let mean: f64 = dfbeta.iter().map(|row| row[j]).sum::<f64>() / n as f64;
        var[j] = dfbeta
            .iter()
            .map(|row| (row[j] - mean).powi(2))
            .sum::<f64>()
            / (n - 1) as f64;
    }

    var
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct GofTestResult {
    #[pyo3(get)]
    pub global_test_stat: f64,
    #[pyo3(get)]
    pub global_p_value: f64,
    #[pyo3(get)]
    pub variable_test_stats: Vec<f64>,
    #[pyo3(get)]
    pub variable_p_values: Vec<f64>,
    #[pyo3(get)]
    pub linear_test_stat: f64,
    #[pyo3(get)]
    pub linear_p_value: f64,
    #[pyo3(get)]
    pub df: usize,
    #[pyo3(get)]
    pub n_obs: usize,
}

#[pyfunction]
#[pyo3(signature = (time, event, covariates, n_covariates, coefficients))]
pub fn goodness_of_fit_cox(
    time: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<f64>,
    n_covariates: usize,
    coefficients: Vec<f64>,
) -> PyResult<GofTestResult> {
    let n = time.len();
    if event.len() != n || covariates.len() != n * n_covariates {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input dimensions mismatch",
        ));
    }

    let outlier_result = outlier_detection_cox(
        time.clone(),
        event.clone(),
        covariates.clone(),
        n_covariates,
        coefficients.clone(),
        3.0,
    )?;

    let n_events = event.iter().filter(|&&e| e == 1).count();

    let chi_sq: f64 = outlier_result
        .deviance_residuals
        .iter()
        .map(|&d| d * d)
        .sum();

    let df = n_events - n_covariates;
    let global_p_value = chi_sq_p_value(chi_sq, df);

    let mut variable_test_stats = vec![0.0; n_covariates];
    let mut variable_p_values = vec![0.0; n_covariates];

    for j in 0..n_covariates {
        let mut corr_sum = 0.0;
        for i in 0..n {
            corr_sum += covariates[i * n_covariates + j] * outlier_result.martingale_residuals[i];
        }
        let test_stat = corr_sum.powi(2);
        variable_test_stats[j] = test_stat;
        variable_p_values[j] = chi_sq_p_value(test_stat, 1);
    }

    let mut linear_corr = 0.0;
    for i in 0..n {
        let eta: f64 = (0..n_covariates)
            .map(|j| covariates[i * n_covariates + j] * coefficients[j])
            .sum();
        linear_corr += eta * outlier_result.martingale_residuals[i];
    }
    let linear_test_stat = linear_corr.powi(2);
    let linear_p_value = chi_sq_p_value(linear_test_stat, 1);

    Ok(GofTestResult {
        global_test_stat: chi_sq,
        global_p_value,
        variable_test_stats,
        variable_p_values,
        linear_test_stat,
        linear_p_value,
        df,
        n_obs: n,
    })
}

fn chi_sq_p_value(chi_sq: f64, df: usize) -> f64 {
    if df == 0 {
        return 0.0;
    }
    1.0 - lower_incomplete_gamma(df as f64 / 2.0, chi_sq / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dfbeta() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let covariates = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let coefficients = vec![0.5];

        let result = dfbeta_cox(time, event, covariates, 1, coefficients, None).unwrap();

        assert_eq!(result.n_obs, 8);
        assert_eq!(result.n_vars, 1);
        assert_eq!(result.dfbeta.len(), 8);
    }

    #[test]
    fn test_leverage() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let covariates = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let coefficients = vec![0.5];

        let result = leverage_cox(time, event, covariates, 1, coefficients, 2.0).unwrap();

        assert_eq!(result.n_obs, 8);
        assert_eq!(result.leverage.len(), 8);
        assert!(result.mean_leverage >= 0.0);
    }

    #[test]
    fn test_schoenfeld_smooth() {
        let event_times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let schoenfeld = vec![0.1, -0.1, 0.2, -0.2, 0.1];
        let coefficients = vec![0.5];

        let result =
            smooth_schoenfeld(event_times, schoenfeld, 1, coefficients, None, "identity").unwrap();

        assert_eq!(result.n_events, 5);
        assert_eq!(result.n_vars, 1);
        assert_eq!(result.smoothed_residuals.len(), 5);
    }

    #[test]
    fn test_outlier_detection() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let covariates = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let coefficients = vec![0.5];

        let result = outlier_detection_cox(time, event, covariates, 1, coefficients, 3.0).unwrap();

        assert_eq!(result.martingale_residuals.len(), 8);
        assert_eq!(result.deviance_residuals.len(), 8);
    }

    #[test]
    fn test_model_influence() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let covariates = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let coefficients = vec![0.5];

        let result = model_influence_cox(time, event, covariates, 1, coefficients).unwrap();

        assert_eq!(result.n_obs, 8);
        assert_eq!(result.cooks_distance.len(), 8);
        assert_eq!(result.covratio.len(), 8);
    }

    #[test]
    fn test_goodness_of_fit() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let covariates = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let coefficients = vec![0.5];

        let result = goodness_of_fit_cox(time, event, covariates, 1, coefficients).unwrap();

        assert_eq!(result.n_obs, 8);
        assert!(result.global_p_value >= 0.0 && result.global_p_value <= 1.0);
    }
}
