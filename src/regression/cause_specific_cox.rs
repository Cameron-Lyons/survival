#![allow(
    unused_variables,
    unused_imports,
    clippy::too_many_arguments,
    clippy::needless_range_loop
)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass]
pub enum CensoringType {
    Censored,
    Competing,
}

#[pymethods]
impl CensoringType {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "censored" => Ok(CensoringType::Censored),
            "competing" => Ok(CensoringType::Competing),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown censoring type. Use 'censored' or 'competing'",
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct CauseSpecificCoxConfig {
    #[pyo3(get, set)]
    pub cause_of_interest: i32,
    #[pyo3(get, set)]
    pub treat_other_causes_as: CensoringType,
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
    #[pyo3(get, set)]
    pub ties: String,
}

#[pymethods]
impl CauseSpecificCoxConfig {
    #[new]
    #[pyo3(signature = (
        cause_of_interest=1,
        treat_other_causes_as=CensoringType::Censored,
        max_iter=100,
        tol=1e-9,
        ties="breslow"
    ))]
    pub fn new(
        cause_of_interest: i32,
        treat_other_causes_as: CensoringType,
        max_iter: usize,
        tol: f64,
        ties: &str,
    ) -> PyResult<Self> {
        if cause_of_interest < 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "cause_of_interest must be >= 1",
            ));
        }
        if max_iter == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "max_iter must be positive",
            ));
        }
        let ties_lower = ties.to_lowercase();
        if ties_lower != "breslow" && ties_lower != "efron" {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "ties must be 'breslow' or 'efron'",
            ));
        }

        Ok(CauseSpecificCoxConfig {
            cause_of_interest,
            treat_other_causes_as,
            max_iter,
            tol,
            ties: ties_lower,
        })
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct CauseSpecificCoxResult {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    #[pyo3(get)]
    pub std_errors: Vec<f64>,
    #[pyo3(get)]
    pub hazard_ratios: Vec<f64>,
    #[pyo3(get)]
    pub hr_ci_lower: Vec<f64>,
    #[pyo3(get)]
    pub hr_ci_upper: Vec<f64>,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub n_events: usize,
    #[pyo3(get)]
    pub n_at_risk: usize,
    #[pyo3(get)]
    pub n_competing: usize,
    #[pyo3(get)]
    pub n_censored: usize,
    #[pyo3(get)]
    pub n_iter: usize,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub cause_of_interest: i32,
    #[pyo3(get)]
    pub baseline_hazard_times: Vec<f64>,
    #[pyo3(get)]
    pub baseline_hazard: Vec<f64>,
    #[pyo3(get)]
    pub cumulative_baseline_hazard: Vec<f64>,
}

#[pymethods]
impl CauseSpecificCoxResult {
    fn __repr__(&self) -> String {
        format!(
            "CauseSpecificCoxResult(cause={}, n_events={}, converged={})",
            self.cause_of_interest, self.n_events, self.converged
        )
    }

    fn predict_cumulative_hazard(&self, x: Vec<f64>, n_obs: usize) -> Vec<Vec<f64>> {
        let n_vars = self.coefficients.len();
        let n_times = self.cumulative_baseline_hazard.len();

        (0..n_obs)
            .into_par_iter()
            .map(|i| {
                let mut linear_pred = 0.0;
                for j in 0..n_vars {
                    linear_pred += x[i * n_vars + j] * self.coefficients[j];
                }
                let exp_lp = linear_pred.exp();

                self.cumulative_baseline_hazard
                    .iter()
                    .map(|&h0| h0 * exp_lp)
                    .collect()
            })
            .collect()
    }

    fn predict_survival(&self, x: Vec<f64>, n_obs: usize) -> Vec<Vec<f64>> {
        let cum_haz = self.predict_cumulative_hazard(x, n_obs);
        cum_haz
            .into_par_iter()
            .map(|h| h.iter().map(|&ch| (-ch).exp()).collect())
            .collect()
    }

    fn predict_cif(&self, x: Vec<f64>, n_obs: usize) -> Vec<Vec<f64>> {
        let n_vars = self.coefficients.len();
        let n_times = self.baseline_hazard_times.len();

        (0..n_obs)
            .into_par_iter()
            .map(|i| {
                let mut linear_pred = 0.0;
                for j in 0..n_vars {
                    linear_pred += x[i * n_vars + j] * self.coefficients[j];
                }
                let exp_lp = linear_pred.exp();

                let mut cif = Vec::with_capacity(n_times);
                let mut cum_inc = 0.0;
                let mut prev_surv = 1.0;

                for t in 0..n_times {
                    let h0_t = self.baseline_hazard[t];
                    let h_t = h0_t * exp_lp;
                    cum_inc += prev_surv * h_t;
                    prev_surv *= (1.0 - h_t).max(0.0);
                    cif.push(cum_inc);
                }

                cif
            })
            .collect()
    }
}

fn compute_cause_specific_gradient_hessian(
    x: &[f64],
    n: usize,
    p: usize,
    time: &[f64],
    cause: &[i32],
    weights: &[f64],
    beta: &[f64],
    cause_of_interest: i32,
    treat_other_as: CensoringType,
    ties: &str,
) -> (Vec<f64>, Vec<Vec<f64>>, f64) {
    let eta: Vec<f64> = (0..n)
        .map(|i| {
            let mut e = 0.0;
            for j in 0..p {
                e += x[i * p + j] * beta[j];
            }
            e.clamp(-700.0, 700.0)
        })
        .collect();

    let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| {
        time[b]
            .partial_cmp(&time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let is_at_risk = |idx: usize| -> bool {
        match treat_other_as {
            CensoringType::Censored => true,
            CensoringType::Competing => cause[idx] == 0 || cause[idx] == cause_of_interest,
        }
    };

    let mut gradient = vec![0.0; p];
    let mut hessian = vec![vec![0.0; p]; p];
    let mut loglik = 0.0;

    let mut risk_sum = 0.0;
    let mut weighted_x = vec![0.0; p];
    let mut weighted_x_outer = vec![vec![0.0; p]; p];

    for &idx in &sorted_indices {
        if is_at_risk(idx) {
            let w = weights[idx] * exp_eta[idx];
            risk_sum += w;

            for j in 0..p {
                let xij = x[idx * p + j];
                weighted_x[j] += w * xij;

                for k in 0..p {
                    let xik = x[idx * p + k];
                    weighted_x_outer[j][k] += w * xij * xik;
                }
            }
        }

        if cause[idx] == cause_of_interest && risk_sum > 0.0 {
            loglik += weights[idx] * (eta[idx] - risk_sum.ln());

            for j in 0..p {
                let xij = x[idx * p + j];
                let x_bar = weighted_x[j] / risk_sum;
                gradient[j] += weights[idx] * (xij - x_bar);

                for k in 0..p {
                    let xik = x[idx * p + k];
                    let x_bar_k = weighted_x[k] / risk_sum;
                    let x_outer_bar = weighted_x_outer[j][k] / risk_sum;
                    hessian[j][k] -= weights[idx] * (x_outer_bar - x_bar * x_bar_k);
                }
            }
        }
    }

    (gradient, hessian, loglik)
}

fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut aug: Vec<Vec<f64>> = a.to_vec();
    let mut rhs = b.to_vec();

    for i in 0..n {
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[k][i].abs() > aug[max_row][i].abs() {
                max_row = k;
            }
        }
        aug.swap(i, max_row);
        rhs.swap(i, max_row);

        if aug[i][i].abs() < 1e-12 {
            return None;
        }

        for k in (i + 1)..n {
            let factor = aug[k][i] / aug[i][i];
            rhs[k] -= factor * rhs[i];
            for j in i..n {
                aug[k][j] -= factor * aug[i][j];
            }
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = rhs[i];
        for j in (i + 1)..n {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }

    Some(x)
}

fn invert_matrix(mat: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = mat.len();
    if n == 0 {
        return None;
    }

    let mut aug: Vec<Vec<f64>> = mat
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut new_row = row.clone();
            new_row.extend(vec![0.0; n]);
            new_row[n + i] = 1.0;
            new_row
        })
        .collect();

    for i in 0..n {
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[k][i].abs() > aug[max_row][i].abs() {
                max_row = k;
            }
        }
        aug.swap(i, max_row);

        if aug[i][i].abs() < 1e-12 {
            return None;
        }

        let pivot = aug[i][i];
        for val in aug[i].iter_mut() {
            *val /= pivot;
        }

        for k in 0..n {
            if k != i {
                let factor = aug[k][i];
                for j in 0..(2 * n) {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }
    }

    Some(aug.into_iter().map(|row| row[n..].to_vec()).collect())
}

fn compute_baseline_hazard(
    n: usize,
    time: &[f64],
    cause: &[i32],
    weights: &[f64],
    exp_eta: &[f64],
    cause_of_interest: i32,
    treat_other_as: CensoringType,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let is_at_risk = |idx: usize| -> bool {
        match treat_other_as {
            CensoringType::Censored => true,
            CensoringType::Competing => cause[idx] == 0 || cause[idx] == cause_of_interest,
        }
    };

    let mut unique_event_times = Vec::new();
    let mut baseline_hazard = Vec::new();
    let mut cumulative_hazard = Vec::new();

    let mut cum_h0 = 0.0;
    let mut i = 0;

    while i < n {
        let idx = sorted_indices[i];
        if cause[idx] != cause_of_interest {
            i += 1;
            continue;
        }

        let current_time = time[idx];

        let mut n_events = 0.0;
        let start_i = i;
        while i < n && (time[sorted_indices[i]] - current_time).abs() < 1e-9 {
            if cause[sorted_indices[i]] == cause_of_interest {
                n_events += weights[sorted_indices[i]];
            }
            i += 1;
        }

        let mut risk_sum = 0.0;
        for &j in &sorted_indices {
            if time[j] >= current_time && is_at_risk(j) {
                risk_sum += weights[j] * exp_eta[j];
            }
        }

        if risk_sum > 0.0 && n_events > 0.0 {
            let h0 = n_events / risk_sum;
            cum_h0 += h0;

            unique_event_times.push(current_time);
            baseline_hazard.push(h0);
            cumulative_hazard.push(cum_h0);
        }
    }

    (unique_event_times, baseline_hazard, cumulative_hazard)
}

#[pyfunction]
#[pyo3(signature = (x, n_obs, n_vars, time, cause, config, weights=None))]
pub fn cause_specific_cox(
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    time: Vec<f64>,
    cause: Vec<i32>,
    config: &CauseSpecificCoxConfig,
    weights: Option<Vec<f64>>,
) -> PyResult<CauseSpecificCoxResult> {
    if x.len() != n_obs * n_vars {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x length must equal n_obs * n_vars",
        ));
    }
    if time.len() != n_obs || cause.len() != n_obs {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and cause must have length n_obs",
        ));
    }

    let wt = weights.unwrap_or_else(|| vec![1.0; n_obs]);

    let n_events = cause
        .iter()
        .filter(|&&c| c == config.cause_of_interest)
        .count();
    let n_competing = cause
        .iter()
        .filter(|&&c| c > 0 && c != config.cause_of_interest)
        .count();
    let n_censored = cause.iter().filter(|&&c| c == 0).count();

    let mut beta = vec![0.0; n_vars];
    let mut converged = false;
    let mut n_iter = 0;
    let mut loglik = 0.0;

    for iter in 0..config.max_iter {
        n_iter = iter + 1;

        let (gradient, hessian, ll) = compute_cause_specific_gradient_hessian(
            &x,
            n_obs,
            n_vars,
            &time,
            &cause,
            &wt,
            &beta,
            config.cause_of_interest,
            config.treat_other_causes_as,
            &config.ties,
        );
        loglik = ll;

        let neg_hessian: Vec<Vec<f64>> = hessian.to_vec();

        let delta = match solve_linear_system(&neg_hessian, &gradient) {
            Some(d) => d,
            None => break,
        };

        let max_change: f64 = delta.iter().map(|d| d.abs()).fold(0.0, f64::max);

        for j in 0..n_vars {
            beta[j] += delta[j];
        }

        if max_change < config.tol {
            converged = true;
            break;
        }
    }

    let (_, final_hessian, _) = compute_cause_specific_gradient_hessian(
        &x,
        n_obs,
        n_vars,
        &time,
        &cause,
        &wt,
        &beta,
        config.cause_of_interest,
        config.treat_other_causes_as,
        &config.ties,
    );

    let neg_hessian: Vec<Vec<f64>> = final_hessian.to_vec();

    let var_cov = invert_matrix(&neg_hessian).unwrap_or_else(|| vec![vec![0.0; n_vars]; n_vars]);

    let std_errors: Vec<f64> = (0..n_vars)
        .map(|j| var_cov[j][j].abs().sqrt().max(1e-10))
        .collect();

    let hazard_ratios: Vec<f64> = beta.iter().map(|&b| b.exp()).collect();

    let z = 1.96;
    let hr_ci_lower: Vec<f64> = beta
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| (b - z * se).exp())
        .collect();
    let hr_ci_upper: Vec<f64> = beta
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| (b + z * se).exp())
        .collect();

    let exp_eta: Vec<f64> = (0..n_obs)
        .map(|i| {
            let mut e = 0.0;
            for j in 0..n_vars {
                e += x[i * n_vars + j] * beta[j];
            }
            e.clamp(-700.0, 700.0).exp()
        })
        .collect();

    let (baseline_times, baseline_hazard, cum_baseline_hazard) = compute_baseline_hazard(
        n_obs,
        &time,
        &cause,
        &wt,
        &exp_eta,
        config.cause_of_interest,
        config.treat_other_causes_as,
    );

    Ok(CauseSpecificCoxResult {
        coefficients: beta,
        std_errors,
        hazard_ratios,
        hr_ci_lower,
        hr_ci_upper,
        log_likelihood: loglik,
        n_events,
        n_at_risk: n_obs,
        n_competing,
        n_censored,
        n_iter,
        converged,
        cause_of_interest: config.cause_of_interest,
        baseline_hazard_times: baseline_times,
        baseline_hazard,
        cumulative_baseline_hazard: cum_baseline_hazard,
    })
}

#[pyfunction]
#[pyo3(signature = (x, n_obs, n_vars, time, cause, max_cause, weights=None, max_iter=100, tol=1e-9))]
pub fn cause_specific_cox_all(
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    time: Vec<f64>,
    cause: Vec<i32>,
    max_cause: i32,
    weights: Option<Vec<f64>>,
    max_iter: usize,
    tol: f64,
) -> PyResult<Vec<CauseSpecificCoxResult>> {
    let mut results = Vec::with_capacity(max_cause as usize);

    for c in 1..=max_cause {
        let config =
            CauseSpecificCoxConfig::new(c, CensoringType::Censored, max_iter, tol, "breslow")?;

        let result = cause_specific_cox(
            x.clone(),
            n_obs,
            n_vars,
            time.clone(),
            cause.clone(),
            &config,
            weights.clone(),
        )?;

        results.push(result);
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config() {
        let config =
            CauseSpecificCoxConfig::new(1, CensoringType::Censored, 100, 1e-9, "breslow").unwrap();
        assert_eq!(config.cause_of_interest, 1);
    }

    #[test]
    fn test_config_validation() {
        assert!(
            CauseSpecificCoxConfig::new(0, CensoringType::Censored, 100, 1e-9, "breslow").is_err()
        );
        assert!(
            CauseSpecificCoxConfig::new(1, CensoringType::Censored, 0, 1e-9, "breslow").is_err()
        );
        assert!(
            CauseSpecificCoxConfig::new(1, CensoringType::Censored, 100, 1e-9, "invalid").is_err()
        );
    }

    #[test]
    fn test_cause_specific_cox_basic() {
        let x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.5, 0.5];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cause = vec![1, 2, 0, 1, 2];
        let config =
            CauseSpecificCoxConfig::new(1, CensoringType::Censored, 100, 1e-5, "breslow").unwrap();

        let result = cause_specific_cox(x, 5, 2, time, cause, &config, None).unwrap();
        assert_eq!(result.coefficients.len(), 2);
        assert_eq!(result.n_events, 2);
        assert_eq!(result.n_competing, 2);
        assert_eq!(result.n_censored, 1);
    }

    #[test]
    fn test_competing_censoring_type() {
        let x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.5, 0.5];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cause = vec![1, 2, 0, 1, 2];

        let config_censored =
            CauseSpecificCoxConfig::new(1, CensoringType::Censored, 100, 1e-5, "breslow").unwrap();
        let config_competing =
            CauseSpecificCoxConfig::new(1, CensoringType::Competing, 100, 1e-5, "breslow").unwrap();

        let result_censored = cause_specific_cox(
            x.clone(),
            5,
            2,
            time.clone(),
            cause.clone(),
            &config_censored,
            None,
        )
        .unwrap();
        let result_competing =
            cause_specific_cox(x, 5, 2, time, cause, &config_competing, None).unwrap();

        assert_eq!(result_censored.n_events, result_competing.n_events);
    }
}
