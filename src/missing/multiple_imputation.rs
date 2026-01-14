#![allow(
    unused_variables,
    unused_imports,
    unused_mut,
    unused_assignments,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::manual_range_contains,
    clippy::manual_swap
)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct MultipleImputationResult {
    #[pyo3(get)]
    pub pooled_coefficients: Vec<f64>,
    #[pyo3(get)]
    pub pooled_se: Vec<f64>,
    #[pyo3(get)]
    pub pooled_ci_lower: Vec<f64>,
    #[pyo3(get)]
    pub pooled_ci_upper: Vec<f64>,
    #[pyo3(get)]
    pub within_variance: Vec<f64>,
    #[pyo3(get)]
    pub between_variance: Vec<f64>,
    #[pyo3(get)]
    pub total_variance: Vec<f64>,
    #[pyo3(get)]
    pub fraction_missing_info: Vec<f64>,
    #[pyo3(get)]
    pub relative_efficiency: Vec<f64>,
    #[pyo3(get)]
    pub n_imputations: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass]
pub enum ImputationMethod {
    PMM,
    Regression,
    MICE,
    KNN,
}

#[pymethods]
impl ImputationMethod {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "pmm" | "predictive_mean_matching" => Ok(ImputationMethod::PMM),
            "regression" | "reg" => Ok(ImputationMethod::Regression),
            "mice" | "chained_equations" => Ok(ImputationMethod::MICE),
            "knn" | "k_nearest" => Ok(ImputationMethod::KNN),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown imputation method",
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
    missing_indicators,
    n_imputations=5,
    method=ImputationMethod::PMM,
    max_iter=20,
    seed=None
))]
pub fn multiple_imputation_survival(
    time: Vec<f64>,
    status: Vec<i32>,
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    missing_indicators: Vec<bool>,
    n_imputations: usize,
    method: ImputationMethod,
    max_iter: usize,
    seed: Option<u64>,
) -> PyResult<MultipleImputationResult> {
    if time.len() != n_obs || status.len() != n_obs {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and status must have length n_obs",
        ));
    }
    if x.len() != n_obs * n_vars {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x must have length n_obs * n_vars",
        ));
    }
    if missing_indicators.len() != n_obs * n_vars {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "missing_indicators must have length n_obs * n_vars",
        ));
    }

    let base_seed = seed.unwrap_or(12345);

    let imputation_results: Vec<Vec<f64>> = (0..n_imputations)
        .into_par_iter()
        .map(|m| {
            let mut rng = fastrand::Rng::with_seed(base_seed + m as u64);
            let imputed_x = impute_dataset(
                &x,
                &missing_indicators,
                n_obs,
                n_vars,
                method,
                max_iter,
                &mut rng,
            );
            fit_cox_model(&time, &status, &imputed_x, n_obs, n_vars)
        })
        .collect();

    let pooled = rubin_rules(&imputation_results, n_vars);

    Ok(pooled)
}

fn impute_dataset(
    x: &[f64],
    missing: &[bool],
    n_obs: usize,
    n_vars: usize,
    method: ImputationMethod,
    max_iter: usize,
    rng: &mut fastrand::Rng,
) -> Vec<f64> {
    let mut imputed = x.to_vec();

    for j in 0..n_vars {
        let col_missing: Vec<usize> = (0..n_obs).filter(|&i| missing[i * n_vars + j]).collect();

        if col_missing.is_empty() {
            continue;
        }

        let col_observed: Vec<usize> = (0..n_obs).filter(|&i| !missing[i * n_vars + j]).collect();

        if col_observed.is_empty() {
            continue;
        }

        match method {
            ImputationMethod::PMM => {
                impute_pmm(
                    &mut imputed,
                    &col_missing,
                    &col_observed,
                    j,
                    n_obs,
                    n_vars,
                    rng,
                );
            }
            ImputationMethod::Regression => {
                impute_regression(
                    &mut imputed,
                    &col_missing,
                    &col_observed,
                    j,
                    n_obs,
                    n_vars,
                    rng,
                );
            }
            ImputationMethod::MICE => {
                for _ in 0..max_iter {
                    impute_regression(
                        &mut imputed,
                        &col_missing,
                        &col_observed,
                        j,
                        n_obs,
                        n_vars,
                        rng,
                    );
                }
            }
            ImputationMethod::KNN => {
                impute_knn(
                    &mut imputed,
                    &col_missing,
                    &col_observed,
                    j,
                    n_obs,
                    n_vars,
                    5,
                );
            }
        }
    }

    imputed
}

fn impute_pmm(
    x: &mut [f64],
    missing_idx: &[usize],
    observed_idx: &[usize],
    var_j: usize,
    _n_obs: usize,
    n_vars: usize,
    rng: &mut fastrand::Rng,
) {
    let observed_values: Vec<f64> = observed_idx
        .iter()
        .map(|&i| x[i * n_vars + var_j])
        .collect();

    let obs_preds: Vec<f64> = observed_idx
        .iter()
        .map(|&obs_i| {
            let mut pred = 0.0;
            for k in 0..n_vars {
                if k != var_j {
                    pred += x[obs_i * n_vars + k];
                }
            }
            pred
        })
        .collect();

    for &i in missing_idx {
        let mut other_pred = 0.0;
        for k in 0..n_vars {
            if k != var_j {
                other_pred += x[i * n_vars + k];
            }
        }

        let k_donors = 5.min(observed_idx.len());

        if k_donors >= observed_idx.len() {
            let donor_idx = rng.usize(0..observed_values.len());
            x[i * n_vars + var_j] = observed_values[donor_idx];
        } else {
            let mut distances: Vec<(f64, f64)> = obs_preds
                .iter()
                .zip(observed_values.iter())
                .map(|(&obs_pred, &val)| ((other_pred - obs_pred).abs(), val))
                .collect();

            let _ = distances.select_nth_unstable_by(k_donors - 1, |a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            let donor_idx = rng.usize(0..k_donors);
            x[i * n_vars + var_j] = distances[donor_idx].1;
        }
    }
}

fn impute_regression(
    x: &mut [f64],
    missing_idx: &[usize],
    observed_idx: &[usize],
    var_j: usize,
    n_obs: usize,
    n_vars: usize,
    rng: &mut fastrand::Rng,
) {
    if n_vars < 2 {
        let mean_obs: f64 = observed_idx
            .iter()
            .map(|&i| x[i * n_vars + var_j])
            .sum::<f64>()
            / observed_idx.len() as f64;
        for &i in missing_idx {
            x[i * n_vars + var_j] = mean_obs;
        }
        return;
    }

    let y_obs: Vec<f64> = observed_idx
        .iter()
        .map(|&i| x[i * n_vars + var_j])
        .collect();

    let mut x_obs: Vec<Vec<f64>> = Vec::with_capacity(observed_idx.len());
    for &i in observed_idx {
        let mut row = Vec::with_capacity(n_vars);
        row.push(1.0);
        for k in 0..n_vars {
            if k != var_j {
                row.push(x[i * n_vars + k]);
            }
        }
        x_obs.push(row);
    }

    let p = x_obs[0].len();
    let mut xtx = vec![0.0; p * p];
    let mut xty = vec![0.0; p];

    for (i, row) in x_obs.iter().enumerate() {
        for j in 0..p {
            xty[j] += row[j] * y_obs[i];
            for k in 0..p {
                xtx[j * p + k] += row[j] * row[k];
            }
        }
    }

    for j in 0..p {
        xtx[j * p + j] += 0.001;
    }

    let beta = solve_linear_system(&xtx, &xty, p);

    let mut residual_var = 0.0;
    for (i, row) in x_obs.iter().enumerate() {
        let pred: f64 = row.iter().zip(beta.iter()).map(|(x, b)| x * b).sum();
        residual_var += (y_obs[i] - pred).powi(2);
    }
    residual_var /= (observed_idx.len() - p).max(1) as f64;
    let residual_sd = residual_var.sqrt();

    for &i in missing_idx {
        let mut row = Vec::with_capacity(p);
        row.push(1.0);
        for k in 0..n_vars {
            if k != var_j {
                row.push(x[i * n_vars + k]);
            }
        }

        let pred: f64 = row.iter().zip(beta.iter()).map(|(x, b)| x * b).sum();
        let noise = standard_normal(rng) * residual_sd;
        x[i * n_vars + var_j] = pred + noise;
    }
}

fn impute_knn(
    x: &mut [f64],
    missing_idx: &[usize],
    observed_idx: &[usize],
    var_j: usize,
    n_obs: usize,
    n_vars: usize,
    k: usize,
) {
    for &i in missing_idx {
        let mut distances: Vec<(f64, f64)> = observed_idx
            .iter()
            .map(|&obs_i| {
                let mut dist = 0.0;
                for v in 0..n_vars {
                    if v != var_j {
                        dist += (x[i * n_vars + v] - x[obs_i * n_vars + v]).powi(2);
                    }
                }
                (dist.sqrt(), x[obs_i * n_vars + var_j])
            })
            .collect();

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let k_actual = k.min(distances.len());
        let imputed_val: f64 =
            distances.iter().take(k_actual).map(|(_, v)| v).sum::<f64>() / k_actual as f64;
        x[i * n_vars + var_j] = imputed_val;
    }
}

fn fit_cox_model(time: &[f64], status: &[i32], x: &[f64], n_obs: usize, n_vars: usize) -> Vec<f64> {
    let mut beta = vec![0.0; n_vars];
    let max_iter = 25;
    let tol = 1e-6;

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

    let mut result = beta.clone();

    let se = compute_standard_errors(time, status, x, &beta, n_obs, n_vars);
    result.extend(se);

    result
}

fn compute_standard_errors(
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
                0.0
            }
        })
        .collect()
}

fn rubin_rules(results: &[Vec<f64>], n_vars: usize) -> MultipleImputationResult {
    let m = results.len();

    let mut pooled_coef = vec![0.0; n_vars];
    for result in results {
        for j in 0..n_vars {
            pooled_coef[j] += result[j];
        }
    }
    for j in 0..n_vars {
        pooled_coef[j] /= m as f64;
    }

    let mut within_var = vec![0.0; n_vars];
    for result in results {
        for j in 0..n_vars {
            let se = if j + n_vars < result.len() {
                result[j + n_vars]
            } else {
                0.1
            };
            within_var[j] += se.powi(2);
        }
    }
    for j in 0..n_vars {
        within_var[j] /= m as f64;
    }

    let mut between_var = vec![0.0; n_vars];
    for result in results {
        for j in 0..n_vars {
            between_var[j] += (result[j] - pooled_coef[j]).powi(2);
        }
    }
    for j in 0..n_vars {
        between_var[j] /= (m - 1).max(1) as f64;
    }

    let total_var: Vec<f64> = (0..n_vars)
        .map(|j| within_var[j] + (1.0 + 1.0 / m as f64) * between_var[j])
        .collect();

    let pooled_se: Vec<f64> = total_var.iter().map(|&v| v.sqrt()).collect();

    let z = 1.96;
    let pooled_ci_lower: Vec<f64> = pooled_coef
        .iter()
        .zip(pooled_se.iter())
        .map(|(&c, &se)| c - z * se)
        .collect();

    let pooled_ci_upper: Vec<f64> = pooled_coef
        .iter()
        .zip(pooled_se.iter())
        .map(|(&c, &se)| c + z * se)
        .collect();

    let fmi: Vec<f64> = (0..n_vars)
        .map(|j| {
            if total_var[j] > 0.0 {
                ((1.0 + 1.0 / m as f64) * between_var[j]) / total_var[j]
            } else {
                0.0
            }
        })
        .collect();

    let rel_eff: Vec<f64> = fmi.iter().map(|&f| 1.0 / (1.0 + f / m as f64)).collect();

    MultipleImputationResult {
        pooled_coefficients: pooled_coef,
        pooled_se,
        pooled_ci_lower,
        pooled_ci_upper,
        within_variance: within_var,
        between_variance: between_var,
        total_variance: total_var,
        fraction_missing_info: fmi,
        relative_efficiency: rel_eff,
        n_imputations: m,
    }
}

fn solve_linear_system(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut aug = vec![0.0; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    for i in 0..n {
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[k * (n + 1) + i].abs() > aug[max_row * (n + 1) + i].abs() {
                max_row = k;
            }
        }

        for j in 0..(n + 1) {
            let temp = aug[i * (n + 1) + j];
            aug[i * (n + 1) + j] = aug[max_row * (n + 1) + j];
            aug[max_row * (n + 1) + j] = temp;
        }

        let pivot = aug[i * (n + 1) + i];
        if pivot.abs() < 1e-10 {
            continue;
        }

        for j in i..(n + 1) {
            aug[i * (n + 1) + j] /= pivot;
        }

        for k in 0..n {
            if k != i {
                let factor = aug[k * (n + 1) + i];
                for j in i..(n + 1) {
                    aug[k * (n + 1) + j] -= factor * aug[i * (n + 1) + j];
                }
            }
        }
    }

    (0..n).map(|i| aug[i * (n + 1) + n]).collect()
}

fn standard_normal(rng: &mut fastrand::Rng) -> f64 {
    let u1: f64 = rng.f64().max(1e-10);
    let u2: f64 = rng.f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[pyfunction]
#[pyo3(signature = (
    time,
    status,
    x,
    n_obs,
    n_vars,
    missing_indicators,
    n_imputations=5
))]
pub fn analyze_missing_pattern(
    time: Vec<f64>,
    status: Vec<i32>,
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    missing_indicators: Vec<bool>,
    n_imputations: usize,
) -> PyResult<(Vec<f64>, Vec<String>, bool)> {
    let mut missing_per_var = vec![0usize; n_vars];
    for i in 0..n_obs {
        for j in 0..n_vars {
            if missing_indicators[i * n_vars + j] {
                missing_per_var[j] += 1;
            }
        }
    }

    let missing_pct: Vec<f64> = missing_per_var
        .iter()
        .map(|&m| m as f64 / n_obs as f64 * 100.0)
        .collect();

    let mut patterns: Vec<String> = Vec::new();
    for i in 0..n_obs {
        let pattern: String = (0..n_vars)
            .map(|j| {
                if missing_indicators[i * n_vars + j] {
                    '1'
                } else {
                    '0'
                }
            })
            .collect();
        if !patterns.contains(&pattern) {
            patterns.push(pattern);
        }
    }

    let is_monotone = check_monotone_pattern(&missing_indicators, n_obs, n_vars);

    Ok((missing_pct, patterns, is_monotone))
}

fn check_monotone_pattern(missing: &[bool], n_obs: usize, n_vars: usize) -> bool {
    for i in 0..n_obs {
        let mut found_missing = false;
        for j in 0..n_vars {
            if missing[i * n_vars + j] {
                found_missing = true;
            } else if found_missing {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiple_imputation_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 0, 1];
        let x = vec![1.0, 0.5, 0.0, 1.2, 1.0, 0.0, 0.0, 0.8, 1.0, 1.5];
        let missing = vec![
            false, false, false, true, false, false, false, false, false, true,
        ];

        let result = multiple_imputation_survival(
            time,
            status,
            x,
            5,
            2,
            missing,
            3,
            ImputationMethod::PMM,
            10,
            Some(42),
        )
        .unwrap();

        assert_eq!(result.pooled_coefficients.len(), 2);
        assert_eq!(result.n_imputations, 3);
    }

    #[test]
    fn test_rubin_rules() {
        let results = vec![
            vec![0.5, 0.3, 0.1, 0.1],
            vec![0.6, 0.2, 0.1, 0.1],
            vec![0.4, 0.4, 0.1, 0.1],
        ];

        let pooled = rubin_rules(&results, 2);

        assert!((pooled.pooled_coefficients[0] - 0.5).abs() < 0.01);
        assert!(
            pooled
                .fraction_missing_info
                .iter()
                .all(|&f| f >= 0.0 && f <= 1.0)
        );
    }
}
