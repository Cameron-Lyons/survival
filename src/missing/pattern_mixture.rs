#![allow(
    unused_variables,
    unused_imports,
    unused_mut,
    unused_assignments,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::type_complexity,
    clippy::collapsible_if
)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct PatternMixtureResult {
    #[pyo3(get)]
    pub pattern_coefficients: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub pattern_se: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub pattern_weights: Vec<f64>,
    #[pyo3(get)]
    pub averaged_coefficients: Vec<f64>,
    #[pyo3(get)]
    pub averaged_se: Vec<f64>,
    #[pyo3(get)]
    pub averaged_ci_lower: Vec<f64>,
    #[pyo3(get)]
    pub averaged_ci_upper: Vec<f64>,
    #[pyo3(get)]
    pub n_patterns: usize,
    #[pyo3(get)]
    pub pattern_sizes: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass]
pub enum SensitivityAnalysisType {
    TiltingModel,
    SelectionModel,
    DeltaAdjustment,
}

#[pymethods]
impl SensitivityAnalysisType {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "tilting" | "tiltingmodel" => Ok(SensitivityAnalysisType::TiltingModel),
            "selection" | "selectionmodel" => Ok(SensitivityAnalysisType::SelectionModel),
            "delta" | "deltaadjustment" => Ok(SensitivityAnalysisType::DeltaAdjustment),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown sensitivity analysis type",
            )),
        }
    }
}

#[pyfunction]
#[pyo3(signature = (
    time,
    status,
    x,
    n_obs,
    n_vars,
    dropout_pattern,
    dropout_time=None
))]
pub fn pattern_mixture_model(
    time: Vec<f64>,
    status: Vec<i32>,
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    dropout_pattern: Vec<i32>,
    dropout_time: Option<Vec<f64>>,
) -> PyResult<PatternMixtureResult> {
    if time.len() != n_obs || status.len() != n_obs || dropout_pattern.len() != n_obs {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All arrays must have length n_obs",
        ));
    }
    if x.len() != n_obs * n_vars {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x must have length n_obs * n_vars",
        ));
    }

    let unique_patterns: Vec<i32> = {
        let mut p: Vec<i32> = dropout_pattern.clone();
        p.sort();
        p.dedup();
        p
    };

    let n_patterns = unique_patterns.len();
    let mut pattern_coefficients = Vec::with_capacity(n_patterns);
    let mut pattern_se = Vec::with_capacity(n_patterns);
    let mut pattern_weights = Vec::with_capacity(n_patterns);
    let mut pattern_sizes = Vec::with_capacity(n_patterns);

    for &pattern in &unique_patterns {
        let pattern_idx: Vec<usize> = (0..n_obs)
            .filter(|&i| dropout_pattern[i] == pattern)
            .collect();

        let n_pattern = pattern_idx.len();
        pattern_sizes.push(n_pattern);
        pattern_weights.push(n_pattern as f64 / n_obs as f64);

        if n_pattern < n_vars + 1 {
            pattern_coefficients.push(vec![0.0; n_vars]);
            pattern_se.push(vec![f64::INFINITY; n_vars]);
            continue;
        }

        let pattern_time: Vec<f64> = pattern_idx.iter().map(|&i| time[i]).collect();
        let pattern_status: Vec<i32> = pattern_idx.iter().map(|&i| status[i]).collect();
        let pattern_x: Vec<f64> = {
            let mut result = Vec::with_capacity(pattern_idx.len() * n_vars);
            for &i in &pattern_idx {
                for j in 0..n_vars {
                    result.push(x[i * n_vars + j]);
                }
            }
            result
        };

        let (coef, se) = fit_cox_pattern(
            &pattern_time,
            &pattern_status,
            &pattern_x,
            n_pattern,
            n_vars,
        );
        pattern_coefficients.push(coef);
        pattern_se.push(se);
    }

    let mut averaged_coef = vec![0.0; n_vars];
    for (coef, &weight) in pattern_coefficients.iter().zip(pattern_weights.iter()) {
        for j in 0..n_vars {
            averaged_coef[j] += coef[j] * weight;
        }
    }

    let mut averaged_var = vec![0.0; n_vars];
    for (i, (coef, se)) in pattern_coefficients
        .iter()
        .zip(pattern_se.iter())
        .enumerate()
    {
        let weight = pattern_weights[i];
        for j in 0..n_vars {
            let within_var = se[j].powi(2);
            let between_var = (coef[j] - averaged_coef[j]).powi(2);
            averaged_var[j] += weight * (within_var + between_var);
        }
    }

    let averaged_se: Vec<f64> = averaged_var.iter().map(|&v| v.sqrt()).collect();

    let z = 1.96;
    let averaged_ci_lower: Vec<f64> = averaged_coef
        .iter()
        .zip(averaged_se.iter())
        .map(|(&c, &se)| c - z * se)
        .collect();

    let averaged_ci_upper: Vec<f64> = averaged_coef
        .iter()
        .zip(averaged_se.iter())
        .map(|(&c, &se)| c + z * se)
        .collect();

    Ok(PatternMixtureResult {
        pattern_coefficients,
        pattern_se,
        pattern_weights,
        averaged_coefficients: averaged_coef,
        averaged_se,
        averaged_ci_lower,
        averaged_ci_upper,
        n_patterns,
        pattern_sizes,
    })
}

fn fit_cox_pattern(
    time: &[f64],
    status: &[i32],
    x: &[f64],
    n_obs: usize,
    n_vars: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut beta = vec![0.0; n_vars];
    let max_iter = 25;
    let tol = 1e-6;

    let mut indices: Vec<usize> = (0..n_obs).collect();
    indices.sort_by(|&a, &b| {
        time[b]
            .partial_cmp(&time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let n_events: usize = status.iter().filter(|&&s| s == 1).count();
    if n_events == 0 {
        return (vec![0.0; n_vars], vec![f64::INFINITY; n_vars]);
    }

    for _ in 0..max_iter {
        let eta: Vec<f64> = (0..n_obs)
            .map(|i| {
                let mut e = 0.0;
                for j in 0..n_vars {
                    e += x[i * n_vars + j] * beta[j];
                }
                e.clamp(-500.0, 500.0)
            })
            .collect();

        let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

        let mut gradient = vec![0.0; n_vars];
        let mut hessian_diag = vec![0.0; n_vars];

        let mut risk_sum = 0.0;
        let mut weighted_x = vec![0.0; n_vars];
        let mut weighted_x_sq = vec![0.0; n_vars];

        for &i in &indices {
            risk_sum += exp_eta[i];
            for j in 0..n_vars {
                weighted_x[j] += exp_eta[i] * x[i * n_vars + j];
                weighted_x_sq[j] += exp_eta[i] * x[i * n_vars + j].powi(2);
            }

            if status[i] == 1 && risk_sum > 0.0 {
                for j in 0..n_vars {
                    let x_bar = weighted_x[j] / risk_sum;
                    let x_sq_bar = weighted_x_sq[j] / risk_sum;
                    gradient[j] += x[i * n_vars + j] - x_bar;
                    hessian_diag[j] += x_sq_bar - x_bar.powi(2);
                }
            }
        }

        let mut max_change: f64 = 0.0;
        for j in 0..n_vars {
            if hessian_diag[j].abs() > 1e-10 {
                let update = gradient[j] / hessian_diag[j];
                beta[j] += update;
                max_change = max_change.max(update.abs());
            }
        }

        if max_change < tol {
            break;
        }
    }

    let se = compute_se(time, status, x, &beta, n_obs, n_vars);

    (beta, se)
}

fn compute_se(
    time: &[f64],
    status: &[i32],
    x: &[f64],
    beta: &[f64],
    n_obs: usize,
    n_vars: usize,
) -> Vec<f64> {
    let mut indices: Vec<usize> = (0..n_obs).collect();
    indices.sort_by(|&a, &b| {
        time[b]
            .partial_cmp(&time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let eta: Vec<f64> = (0..n_obs)
        .map(|i| {
            let mut e = 0.0;
            for j in 0..n_vars {
                e += x[i * n_vars + j] * beta[j];
            }
            e.clamp(-500.0, 500.0)
        })
        .collect();

    let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

    let mut info_diag = vec![0.0; n_vars];
    let mut risk_sum = 0.0;
    let mut weighted_x = vec![0.0; n_vars];
    let mut weighted_x_sq = vec![0.0; n_vars];

    for &i in &indices {
        risk_sum += exp_eta[i];
        for j in 0..n_vars {
            weighted_x[j] += exp_eta[i] * x[i * n_vars + j];
            weighted_x_sq[j] += exp_eta[i] * x[i * n_vars + j].powi(2);
        }

        if status[i] == 1 && risk_sum > 0.0 {
            for j in 0..n_vars {
                let x_bar = weighted_x[j] / risk_sum;
                let x_sq_bar = weighted_x_sq[j] / risk_sum;
                info_diag[j] += x_sq_bar - x_bar.powi(2);
            }
        }
    }

    info_diag
        .iter()
        .map(|&info| {
            if info > 1e-10 {
                (1.0 / info).sqrt()
            } else {
                f64::INFINITY
            }
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature = (
    time,
    status,
    x,
    n_obs,
    n_vars,
    dropout_pattern,
    delta_values,
    analysis_type=SensitivityAnalysisType::DeltaAdjustment
))]
pub fn sensitivity_analysis(
    time: Vec<f64>,
    status: Vec<i32>,
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    dropout_pattern: Vec<i32>,
    delta_values: Vec<f64>,
    analysis_type: SensitivityAnalysisType,
) -> PyResult<Vec<(f64, Vec<f64>, Vec<f64>)>> {
    let mut results = Vec::with_capacity(delta_values.len());

    for &delta in &delta_values {
        let adjusted_result = match analysis_type {
            SensitivityAnalysisType::DeltaAdjustment => {
                delta_adjustment_model(&time, &status, &x, n_obs, n_vars, &dropout_pattern, delta)
            }
            SensitivityAnalysisType::TiltingModel => {
                tilting_model(&time, &status, &x, n_obs, n_vars, &dropout_pattern, delta)
            }
            SensitivityAnalysisType::SelectionModel => {
                selection_model(&time, &status, &x, n_obs, n_vars, &dropout_pattern, delta)
            }
        };

        results.push((delta, adjusted_result.0, adjusted_result.1));
    }

    Ok(results)
}

fn delta_adjustment_model(
    time: &[f64],
    status: &[i32],
    x: &[f64],
    n_obs: usize,
    n_vars: usize,
    dropout_pattern: &[i32],
    delta: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut adjusted_x = x.to_vec();

    for i in 0..n_obs {
        if dropout_pattern[i] > 0 {
            for j in 0..n_vars {
                adjusted_x[i * n_vars + j] += delta * dropout_pattern[i] as f64;
            }
        }
    }

    fit_cox_pattern(time, status, &adjusted_x, n_obs, n_vars)
}

fn tilting_model(
    time: &[f64],
    status: &[i32],
    x: &[f64],
    n_obs: usize,
    n_vars: usize,
    dropout_pattern: &[i32],
    tilt_param: f64,
) -> (Vec<f64>, Vec<f64>) {
    let weights: Vec<f64> = dropout_pattern
        .iter()
        .map(|&p| (tilt_param * p as f64).exp())
        .collect();

    let total_weight: f64 = weights.iter().sum();
    let normalized_weights: Vec<f64> = weights
        .iter()
        .map(|&w| w / total_weight * n_obs as f64)
        .collect();

    fit_weighted_cox(time, status, x, &normalized_weights, n_obs, n_vars)
}

fn selection_model(
    time: &[f64],
    status: &[i32],
    x: &[f64],
    n_obs: usize,
    n_vars: usize,
    dropout_pattern: &[i32],
    selection_param: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut ipw_weights = vec![1.0; n_obs];

    for i in 0..n_obs {
        if dropout_pattern[i] > 0 {
            let linear_pred: f64 = (0..n_vars).map(|j| x[i * n_vars + j]).sum();
            let dropout_prob = 1.0 / (1.0 + (-selection_param * linear_pred).exp());
            ipw_weights[i] = 1.0 / dropout_prob.max(0.01);
        }
    }

    fit_weighted_cox(time, status, x, &ipw_weights, n_obs, n_vars)
}

fn fit_weighted_cox(
    time: &[f64],
    status: &[i32],
    x: &[f64],
    weights: &[f64],
    n_obs: usize,
    n_vars: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut beta = vec![0.0; n_vars];
    let max_iter = 25;
    let tol = 1e-6;
    let mut hessian_diag = vec![0.0; n_vars];

    let mut indices: Vec<usize> = (0..n_obs).collect();
    indices.sort_by(|&a, &b| {
        time[b]
            .partial_cmp(&time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for _ in 0..max_iter {
        let eta: Vec<f64> = (0..n_obs)
            .map(|i| {
                let mut e = 0.0;
                for j in 0..n_vars {
                    e += x[i * n_vars + j] * beta[j];
                }
                e.clamp(-500.0, 500.0)
            })
            .collect();

        let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

        let mut gradient = vec![0.0; n_vars];
        hessian_diag = vec![0.0; n_vars];

        let mut risk_sum = 0.0;
        let mut weighted_x = vec![0.0; n_vars];
        let mut weighted_x_sq = vec![0.0; n_vars];

        for &i in &indices {
            risk_sum += weights[i] * exp_eta[i];
            for j in 0..n_vars {
                weighted_x[j] += weights[i] * exp_eta[i] * x[i * n_vars + j];
                weighted_x_sq[j] += weights[i] * exp_eta[i] * x[i * n_vars + j].powi(2);
            }

            if status[i] == 1 && risk_sum > 0.0 {
                for j in 0..n_vars {
                    let x_bar = weighted_x[j] / risk_sum;
                    let x_sq_bar = weighted_x_sq[j] / risk_sum;
                    gradient[j] += weights[i] * (x[i * n_vars + j] - x_bar);
                    hessian_diag[j] += weights[i] * (x_sq_bar - x_bar.powi(2));
                }
            }
        }

        let mut max_change: f64 = 0.0;
        for j in 0..n_vars {
            if hessian_diag[j].abs() > 1e-10 {
                let update = gradient[j] / hessian_diag[j];
                beta[j] += update;
                max_change = max_change.max(update.abs());
            }
        }

        if max_change < tol {
            break;
        }
    }

    let se = hessian_diag
        .iter()
        .map(|&h| {
            if h.abs() > 1e-10 {
                (1.0 / h).sqrt()
            } else {
                f64::INFINITY
            }
        })
        .collect();

    (beta, se)
}

#[pyfunction]
pub fn tipping_point_analysis(
    time: Vec<f64>,
    status: Vec<i32>,
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    dropout_pattern: Vec<i32>,
    coef_index: usize,
    target_value: f64,
    delta_range: (f64, f64),
    n_steps: usize,
) -> PyResult<Option<f64>> {
    if coef_index >= n_vars {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "coef_index must be less than n_vars",
        ));
    }

    let delta_step = (delta_range.1 - delta_range.0) / n_steps as f64;

    let mut prev_coef = None;
    let mut prev_delta = None;

    for i in 0..=n_steps {
        let delta = delta_range.0 + i as f64 * delta_step;

        let (coef, _) =
            delta_adjustment_model(&time, &status, &x, n_obs, n_vars, &dropout_pattern, delta);

        let current_coef = coef[coef_index];

        if let Some(prev) = prev_coef {
            if (prev < target_value && current_coef >= target_value)
                || (prev > target_value && current_coef <= target_value)
            {
                let frac = (target_value - prev) / (current_coef - prev);
                let tipping_delta = prev_delta.unwrap() + frac * delta_step;
                return Ok(Some(tipping_delta));
            }
        }

        prev_coef = Some(current_coef);
        prev_delta = Some(delta);
    }

    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_mixture_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let status = vec![1, 0, 1, 0, 1, 0, 1, 0];
        let x = vec![
            1.0, 0.5, 0.0, 1.2, 1.0, 0.0, 0.0, 0.8, 1.0, 1.5, 0.0, 0.3, 1.0, 0.9, 0.0, 1.1,
        ];
        let dropout = vec![0, 0, 0, 1, 1, 1, 2, 2];

        let result = pattern_mixture_model(time, status, x, 8, 2, dropout, None).unwrap();

        assert!(result.n_patterns >= 1);
        assert_eq!(result.averaged_coefficients.len(), 2);
    }

    #[test]
    fn test_sensitivity_analysis() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 0, 1];
        let x = vec![1.0, 0.0, 1.0, 0.0, 1.0];
        let dropout = vec![0, 0, 1, 1, 0];
        let deltas = vec![-0.5, 0.0, 0.5];

        let results = sensitivity_analysis(
            time,
            status,
            x,
            5,
            1,
            dropout,
            deltas,
            SensitivityAnalysisType::DeltaAdjustment,
        )
        .unwrap();

        assert_eq!(results.len(), 3);
    }
}
