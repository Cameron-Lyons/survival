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
pub enum CorrelationType {
    Independent,
    SharedFrailty,
    CopulaBased,
}

#[pymethods]
impl CorrelationType {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "independent" => Ok(CorrelationType::Independent),
            "shared_frailty" | "sharedfrailty" | "frailty" => Ok(CorrelationType::SharedFrailty),
            "copula_based" | "copulabased" | "copula" => Ok(CorrelationType::CopulaBased),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown correlation type. Use 'independent', 'shared_frailty', or 'copula_based'",
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct JointCompetingRisksConfig {
    #[pyo3(get, set)]
    pub num_causes: usize,
    #[pyo3(get, set)]
    pub correlation_structure: CorrelationType,
    #[pyo3(get, set)]
    pub frailty_variance: f64,
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
    #[pyo3(get, set)]
    pub estimate_correlation: bool,
}

#[pymethods]
impl JointCompetingRisksConfig {
    #[new]
    #[pyo3(signature = (
        num_causes=2,
        correlation_structure=CorrelationType::Independent,
        frailty_variance=1.0,
        max_iter=100,
        tol=1e-6,
        estimate_correlation=true
    ))]
    pub fn new(
        num_causes: usize,
        correlation_structure: CorrelationType,
        frailty_variance: f64,
        max_iter: usize,
        tol: f64,
        estimate_correlation: bool,
    ) -> PyResult<Self> {
        if num_causes < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "num_causes must be at least 2",
            ));
        }
        if frailty_variance <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "frailty_variance must be positive",
            ));
        }
        if max_iter == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "max_iter must be positive",
            ));
        }

        Ok(JointCompetingRisksConfig {
            num_causes,
            correlation_structure,
            frailty_variance,
            max_iter,
            tol,
            estimate_correlation,
        })
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct CauseResult {
    #[pyo3(get)]
    pub cause: usize,
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    #[pyo3(get)]
    pub std_errors: Vec<f64>,
    #[pyo3(get)]
    pub hazard_ratios: Vec<f64>,
    #[pyo3(get)]
    pub baseline_hazard_times: Vec<f64>,
    #[pyo3(get)]
    pub baseline_hazard: Vec<f64>,
    #[pyo3(get)]
    pub cumulative_baseline_hazard: Vec<f64>,
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct JointCompetingRisksResult {
    #[pyo3(get)]
    pub cause_specific_results: Vec<CauseResult>,
    #[pyo3(get)]
    pub subdistribution_results: Vec<CauseResult>,
    #[pyo3(get)]
    pub correlation_matrix: Option<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub frailty_variance: Option<f64>,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub aic: f64,
    #[pyo3(get)]
    pub bic: f64,
    #[pyo3(get)]
    pub n_events_by_cause: Vec<usize>,
    #[pyo3(get)]
    pub n_obs: usize,
    #[pyo3(get)]
    pub n_iter: usize,
    #[pyo3(get)]
    pub converged: bool,
}

#[pymethods]
impl JointCompetingRisksResult {
    fn __repr__(&self) -> String {
        format!(
            "JointCompetingRisksResult(n_causes={}, n_obs={}, converged={})",
            self.cause_specific_results.len(),
            self.n_obs,
            self.converged
        )
    }

    fn predict_cif(&self, x: Vec<f64>, n_obs: usize, cause_idx: usize) -> PyResult<Vec<Vec<f64>>> {
        if cause_idx >= self.cause_specific_results.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "cause_idx out of range",
            ));
        }

        let cs = &self.cause_specific_results[cause_idx];
        let n_vars = cs.coefficients.len();
        let n_times = cs.baseline_hazard_times.len();

        let all_cum_hazards: Vec<Vec<Vec<f64>>> = self
            .cause_specific_results
            .iter()
            .map(|cr| {
                (0..n_obs)
                    .map(|i| {
                        let mut lp = 0.0;
                        for j in 0..cr.coefficients.len().min(n_vars) {
                            lp += x[i * n_vars + j] * cr.coefficients[j];
                        }
                        let exp_lp = lp.exp();
                        cr.cumulative_baseline_hazard
                            .iter()
                            .map(|&h0| h0 * exp_lp)
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let cif: Vec<Vec<f64>> = (0..n_obs)
            .into_par_iter()
            .map(|i| {
                let cs = &self.cause_specific_results[cause_idx];
                let mut cif_vec = Vec::with_capacity(n_times);
                let mut cum_inc = 0.0;
                let mut prev_surv = 1.0;

                for t in 0..n_times {
                    let mut total_hazard = 0.0;
                    for k in 0..self.cause_specific_results.len() {
                        if t < all_cum_hazards[k][i].len() {
                            let h_t = if t == 0 {
                                all_cum_hazards[k][i][t]
                            } else {
                                all_cum_hazards[k][i][t] - all_cum_hazards[k][i][t - 1]
                            };
                            total_hazard += h_t.max(0.0);
                        }
                    }

                    let h_cause_t = if t == 0 {
                        all_cum_hazards[cause_idx][i][t]
                    } else {
                        all_cum_hazards[cause_idx][i][t] - all_cum_hazards[cause_idx][i][t - 1]
                    };

                    cum_inc += prev_surv * h_cause_t.max(0.0);
                    prev_surv *= (-total_hazard).exp();
                    cif_vec.push(cum_inc.min(1.0));
                }

                cif_vec
            })
            .collect();

        Ok(cif)
    }

    fn predict_overall_survival(&self, x: Vec<f64>, n_obs: usize) -> Vec<Vec<f64>> {
        let n_times = self.cause_specific_results[0].baseline_hazard_times.len();
        let n_vars = self.cause_specific_results[0].coefficients.len();

        (0..n_obs)
            .into_par_iter()
            .map(|i| {
                let mut surv_vec = Vec::with_capacity(n_times);
                let mut cum_surv = 1.0;

                for t in 0..n_times {
                    let mut total_hazard = 0.0;

                    for cs in &self.cause_specific_results {
                        let mut lp = 0.0;
                        for j in 0..cs.coefficients.len().min(n_vars) {
                            lp += x[i * n_vars + j] * cs.coefficients[j];
                        }
                        let exp_lp = lp.exp();

                        let h_t = if t == 0 {
                            cs.cumulative_baseline_hazard
                                .first()
                                .copied()
                                .unwrap_or(0.0)
                        } else {
                            cs.cumulative_baseline_hazard.get(t).copied().unwrap_or(0.0)
                                - cs.cumulative_baseline_hazard
                                    .get(t - 1)
                                    .copied()
                                    .unwrap_or(0.0)
                        };

                        total_hazard += h_t * exp_lp;
                    }

                    cum_surv *= (-total_hazard).exp();
                    surv_vec.push(cum_surv.clamp(0.0, 1.0));
                }

                surv_vec
            })
            .collect()
    }
}

fn fit_cause_specific_cox(
    x: &[f64],
    n: usize,
    p: usize,
    time: &[f64],
    cause: &[i32],
    weights: &[f64],
    cause_of_interest: i32,
    max_iter: usize,
    tol: f64,
) -> (Vec<f64>, Vec<f64>, f64, bool, usize) {
    let mut beta = vec![0.0; p];
    let mut converged = false;
    let mut n_iter = 0;
    let mut loglik = 0.0;

    for iter in 0..max_iter {
        n_iter = iter + 1;

        let (gradient, hessian, ll) =
            compute_gradient_hessian(x, n, p, time, cause, weights, &beta, cause_of_interest);
        loglik = ll;

        let delta = match solve_system(&hessian, &gradient) {
            Some(d) => d,
            None => break,
        };

        let max_change: f64 = delta.iter().map(|d| d.abs()).fold(0.0, f64::max);

        for j in 0..p {
            beta[j] += delta[j];
        }

        if max_change < tol {
            converged = true;
            break;
        }
    }

    let (_, final_hessian, _) =
        compute_gradient_hessian(x, n, p, time, cause, weights, &beta, cause_of_interest);

    let var_cov = invert_matrix(&final_hessian).unwrap_or_else(|| vec![vec![0.0; p]; p]);
    let std_errors: Vec<f64> = (0..p)
        .map(|j| var_cov[j][j].abs().sqrt().max(1e-10))
        .collect();

    (beta, std_errors, loglik, converged, n_iter)
}

fn compute_gradient_hessian(
    x: &[f64],
    n: usize,
    p: usize,
    time: &[f64],
    cause: &[i32],
    weights: &[f64],
    beta: &[f64],
    cause_of_interest: i32,
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

    let mut gradient = vec![0.0; p];
    let mut hessian = vec![vec![0.0; p]; p];
    let mut loglik = 0.0;

    let mut risk_sum = 0.0;
    let mut weighted_x = vec![0.0; p];
    let mut weighted_x_outer = vec![vec![0.0; p]; p];

    for &idx in &sorted_indices {
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

        if cause[idx] == cause_of_interest && risk_sum > 0.0 {
            loglik += weights[idx] * (eta[idx] - risk_sum.ln());

            for j in 0..p {
                let xij = x[idx * p + j];
                let x_bar = weighted_x[j] / risk_sum;
                gradient[j] += weights[idx] * (xij - x_bar);

                for k in 0..p {
                    let x_bar_k = weighted_x[k] / risk_sum;
                    let x_outer_bar = weighted_x_outer[j][k] / risk_sum;
                    hessian[j][k] -= weights[idx] * (x_outer_bar - x_bar * x_bar_k);
                }
            }
        }
    }

    (gradient, hessian, loglik)
}

fn solve_system(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
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
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut unique_times = Vec::new();
    let mut baseline = Vec::new();
    let mut cumulative = Vec::new();
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

        while i < n && (time[sorted_indices[i]] - current_time).abs() < 1e-9 {
            if cause[sorted_indices[i]] == cause_of_interest {
                n_events += weights[sorted_indices[i]];
            }
            i += 1;
        }

        let mut risk_sum = 0.0;
        for &j in &sorted_indices {
            if time[j] >= current_time {
                risk_sum += weights[j] * exp_eta[j];
            }
        }

        if risk_sum > 0.0 && n_events > 0.0 {
            let h0 = n_events / risk_sum;
            cum_h0 += h0;

            unique_times.push(current_time);
            baseline.push(h0);
            cumulative.push(cum_h0);
        }
    }

    (unique_times, baseline, cumulative)
}

#[pyfunction]
#[pyo3(signature = (x, n_obs, n_vars, time, cause, config, weights=None))]
pub fn joint_competing_risks(
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    time: Vec<f64>,
    cause: Vec<i32>,
    config: &JointCompetingRisksConfig,
    weights: Option<Vec<f64>>,
) -> PyResult<JointCompetingRisksResult> {
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

    let n_events_by_cause: Vec<usize> = (1..=config.num_causes as i32)
        .map(|c| cause.iter().filter(|&&cc| cc == c).count())
        .collect();

    let mut total_loglik = 0.0;
    let mut total_n_iter = 0;
    let mut all_converged = true;

    let mut cause_specific_results = Vec::with_capacity(config.num_causes);

    for c in 1..=config.num_causes as i32 {
        let (beta, std_errors, loglik, converged, n_iter) = fit_cause_specific_cox(
            &x,
            n_obs,
            n_vars,
            &time,
            &cause,
            &wt,
            c,
            config.max_iter,
            config.tol,
        );

        total_loglik += loglik;
        total_n_iter = total_n_iter.max(n_iter);
        all_converged = all_converged && converged;

        let exp_eta: Vec<f64> = (0..n_obs)
            .map(|i| {
                let mut e = 0.0;
                for j in 0..n_vars {
                    e += x[i * n_vars + j] * beta[j];
                }
                e.clamp(-700.0, 700.0).exp()
            })
            .collect();

        let (times, baseline, cumulative) =
            compute_baseline_hazard(n_obs, &time, &cause, &wt, &exp_eta, c);

        let hazard_ratios: Vec<f64> = beta.iter().map(|&b| b.exp()).collect();

        cause_specific_results.push(CauseResult {
            cause: c as usize,
            coefficients: beta,
            std_errors,
            hazard_ratios,
            baseline_hazard_times: times,
            baseline_hazard: baseline,
            cumulative_baseline_hazard: cumulative,
        });
    }

    let subdistribution_results = cause_specific_results.clone();

    let correlation_matrix = match config.correlation_structure {
        CorrelationType::Independent => None,
        CorrelationType::SharedFrailty | CorrelationType::CopulaBased => {
            let mut corr = vec![vec![0.0; config.num_causes]; config.num_causes];
            for i in 0..config.num_causes {
                corr[i][i] = 1.0;
            }
            Some(corr)
        }
    };

    let frailty_variance = match config.correlation_structure {
        CorrelationType::SharedFrailty => Some(config.frailty_variance),
        _ => None,
    };

    let n_params = n_vars * config.num_causes;
    let aic = -2.0 * total_loglik + 2.0 * n_params as f64;
    let bic = -2.0 * total_loglik + (n_params as f64) * (n_obs as f64).ln();

    Ok(JointCompetingRisksResult {
        cause_specific_results,
        subdistribution_results,
        correlation_matrix,
        frailty_variance,
        log_likelihood: total_loglik,
        aic,
        bic,
        n_events_by_cause,
        n_obs,
        n_iter: total_n_iter,
        converged: all_converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config() {
        let config =
            JointCompetingRisksConfig::new(2, CorrelationType::Independent, 1.0, 100, 1e-6, true)
                .unwrap();
        assert_eq!(config.num_causes, 2);
    }

    #[test]
    fn test_config_validation() {
        assert!(
            JointCompetingRisksConfig::new(1, CorrelationType::Independent, 1.0, 100, 1e-6, true)
                .is_err()
        );
        assert!(
            JointCompetingRisksConfig::new(2, CorrelationType::Independent, -1.0, 100, 1e-6, true)
                .is_err()
        );
    }

    #[test]
    fn test_joint_competing_risks_basic() {
        let x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.5, 0.5];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cause = vec![1, 2, 0, 1, 2];

        let config =
            JointCompetingRisksConfig::new(2, CorrelationType::Independent, 1.0, 100, 1e-5, true)
                .unwrap();

        let result = joint_competing_risks(x, 5, 2, time, cause, &config, None).unwrap();

        assert_eq!(result.cause_specific_results.len(), 2);
        assert_eq!(result.n_events_by_cause.len(), 2);
        assert_eq!(result.n_obs, 5);
    }
}
