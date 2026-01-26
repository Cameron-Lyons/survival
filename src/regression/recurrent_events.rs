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

use crate::utilities::statistical::normal_cdf;

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass]
pub enum PWPTimescale {
    Gap,
    Total,
}

#[pymethods]
impl PWPTimescale {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "gap" => Ok(PWPTimescale::Gap),
            "total" => Ok(PWPTimescale::Total),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown timescale: must be 'gap' or 'total'",
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct PWPConfig {
    #[pyo3(get, set)]
    pub timescale: PWPTimescale,
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
    #[pyo3(get, set)]
    pub stratify_by_event: bool,
    #[pyo3(get, set)]
    pub robust_variance: bool,
}

#[pymethods]
impl PWPConfig {
    #[new]
    #[pyo3(signature = (timescale=PWPTimescale::Gap, max_iter=100, tol=1e-6, stratify_by_event=true, robust_variance=true))]
    pub fn new(
        timescale: PWPTimescale,
        max_iter: usize,
        tol: f64,
        stratify_by_event: bool,
        robust_variance: bool,
    ) -> Self {
        PWPConfig {
            timescale,
            max_iter,
            tol,
            stratify_by_event,
            robust_variance,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct PWPResult {
    #[pyo3(get)]
    pub coef: Vec<f64>,
    #[pyo3(get)]
    pub std_errors: Vec<f64>,
    #[pyo3(get)]
    pub robust_std_errors: Vec<f64>,
    #[pyo3(get)]
    pub z_scores: Vec<f64>,
    #[pyo3(get)]
    pub p_values: Vec<f64>,
    #[pyo3(get)]
    pub hazard_ratios: Vec<f64>,
    #[pyo3(get)]
    pub hr_lower: Vec<f64>,
    #[pyo3(get)]
    pub hr_upper: Vec<f64>,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub n_events: usize,
    #[pyo3(get)]
    pub n_subjects: usize,
    #[pyo3(get)]
    pub n_iter: usize,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub event_specific_coef: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub baseline_cumhaz: Vec<f64>,
}

#[pyfunction]
#[pyo3(signature = (id, start, stop, event, event_number, covariates, config))]
pub fn pwp_model(
    id: Vec<i32>,
    start: Vec<f64>,
    stop: Vec<f64>,
    event: Vec<i32>,
    event_number: Vec<i32>,
    covariates: Vec<f64>,
    config: &PWPConfig,
) -> PyResult<PWPResult> {
    let n = id.len();
    if start.len() != n || stop.len() != n || event.len() != n || event_number.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All input vectors must have the same length",
        ));
    }

    let p = if covariates.is_empty() {
        1
    } else {
        covariates.len() / n
    };
    let x_mat = if covariates.is_empty() {
        vec![1.0; n]
    } else {
        covariates.clone()
    };

    let unique_ids: Vec<i32> = {
        let mut ids = id.clone();
        ids.sort();
        ids.dedup();
        ids
    };
    let n_subjects = unique_ids.len();
    let n_events_total = event.iter().filter(|&&e| e == 1).count();

    let max_event_num = *event_number.iter().max().unwrap_or(&1) as usize;

    let mut beta = vec![0.0; p];
    let mut converged = false;
    let mut n_iter = 0;
    let mut prev_loglik = f64::NEG_INFINITY;

    let time_var: Vec<f64> = match config.timescale {
        PWPTimescale::Gap => stop
            .iter()
            .zip(start.iter())
            .map(|(&s, &st)| s - st)
            .collect(),
        PWPTimescale::Total => stop.clone(),
    };

    for iter in 0..config.max_iter {
        n_iter = iter + 1;

        let mut loglik = 0.0;
        let mut gradient = vec![0.0; p];
        let mut hessian = vec![vec![0.0; p]; p];

        let mut sorted_indices: Vec<usize> = (0..n).collect();
        sorted_indices.sort_by(|&a, &b| time_var[b].partial_cmp(&time_var[a]).unwrap());

        for &i in &sorted_indices {
            if event[i] != 1 {
                continue;
            }

            let strata = if config.stratify_by_event {
                event_number[i]
            } else {
                1
            };

            let mut eta_i = 0.0;
            for j in 0..p {
                eta_i += x_mat[i * p + j] * beta[j];
            }
            let exp_eta_i = eta_i.exp();

            let mut risk_sum = 0.0;
            let mut risk_x_sum = vec![0.0; p];
            let mut risk_xx_sum = vec![vec![0.0; p]; p];

            for &k in &sorted_indices {
                let in_strata = !config.stratify_by_event || event_number[k] == strata;
                let at_risk = time_var[k] >= time_var[i];

                if in_strata && at_risk {
                    let mut eta_k = 0.0;
                    for j in 0..p {
                        eta_k += x_mat[k * p + j] * beta[j];
                    }
                    let exp_eta_k = eta_k.exp();

                    risk_sum += exp_eta_k;
                    for j in 0..p {
                        risk_x_sum[j] += x_mat[k * p + j] * exp_eta_k;
                    }
                    for j1 in 0..p {
                        for j2 in 0..p {
                            risk_xx_sum[j1][j2] +=
                                x_mat[k * p + j1] * x_mat[k * p + j2] * exp_eta_k;
                        }
                    }
                }
            }

            if risk_sum > 1e-10 {
                loglik += eta_i - risk_sum.ln();

                for j in 0..p {
                    let x_bar = risk_x_sum[j] / risk_sum;
                    gradient[j] += x_mat[i * p + j] - x_bar;
                }

                for j1 in 0..p {
                    let x_bar1 = risk_x_sum[j1] / risk_sum;
                    for j2 in 0..p {
                        let x_bar2 = risk_x_sum[j2] / risk_sum;
                        hessian[j1][j2] += risk_xx_sum[j1][j2] / risk_sum - x_bar1 * x_bar2;
                    }
                }
            }
        }

        let mut inv_hess = vec![vec![0.0; p]; p];
        for j in 0..p {
            inv_hess[j][j] = if hessian[j][j].abs() > 1e-10 {
                1.0 / hessian[j][j]
            } else {
                0.0
            };
        }

        for j in 0..p {
            beta[j] += inv_hess[j][j] * gradient[j];
            beta[j] = beta[j].clamp(-10.0, 10.0);
        }

        if (loglik - prev_loglik).abs() < config.tol {
            converged = true;
            break;
        }
        prev_loglik = loglik;
    }

    let mut info_matrix = vec![vec![0.0; p]; p];
    let mut score_residuals: Vec<Vec<f64>> = unique_ids.iter().map(|_| vec![0.0; p]).collect();

    let id_to_idx: std::collections::HashMap<i32, usize> = unique_ids
        .iter()
        .enumerate()
        .map(|(idx, &id_val)| (id_val, idx))
        .collect();

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| time_var[b].partial_cmp(&time_var[a]).unwrap());

    for &i in &sorted_indices {
        if event[i] != 1 {
            continue;
        }

        let strata = if config.stratify_by_event {
            event_number[i]
        } else {
            1
        };

        let mut risk_sum = 0.0;
        let mut risk_x_sum = vec![0.0; p];
        let mut risk_xx_sum = vec![vec![0.0; p]; p];

        for &k in &sorted_indices {
            let in_strata = !config.stratify_by_event || event_number[k] == strata;
            let at_risk = time_var[k] >= time_var[i];

            if in_strata && at_risk {
                let mut eta_k = 0.0;
                for j in 0..p {
                    eta_k += x_mat[k * p + j] * beta[j];
                }
                let exp_eta_k = eta_k.exp();

                risk_sum += exp_eta_k;
                for j in 0..p {
                    risk_x_sum[j] += x_mat[k * p + j] * exp_eta_k;
                }
                for j1 in 0..p {
                    for j2 in 0..p {
                        risk_xx_sum[j1][j2] += x_mat[k * p + j1] * x_mat[k * p + j2] * exp_eta_k;
                    }
                }
            }
        }

        if risk_sum > 1e-10 {
            for j1 in 0..p {
                let x_bar1 = risk_x_sum[j1] / risk_sum;
                for j2 in 0..p {
                    let x_bar2 = risk_x_sum[j2] / risk_sum;
                    info_matrix[j1][j2] += risk_xx_sum[j1][j2] / risk_sum - x_bar1 * x_bar2;
                }
            }

            if let Some(&subj_idx) = id_to_idx.get(&id[i]) {
                for j in 0..p {
                    let x_bar = risk_x_sum[j] / risk_sum;
                    score_residuals[subj_idx][j] += x_mat[i * p + j] - x_bar;
                }
            }
        }
    }

    let std_errors: Vec<f64> = (0..p)
        .map(|j| {
            if info_matrix[j][j] > 1e-10 {
                (1.0 / info_matrix[j][j]).sqrt()
            } else {
                f64::INFINITY
            }
        })
        .collect();

    let mut robust_var = vec![vec![0.0; p]; p];
    for subj_scores in &score_residuals {
        for j1 in 0..p {
            for j2 in 0..p {
                robust_var[j1][j2] += subj_scores[j1] * subj_scores[j2];
            }
        }
    }

    let robust_std_errors: Vec<f64> = (0..p)
        .map(|j| {
            let inv_info = if info_matrix[j][j] > 1e-10 {
                1.0 / info_matrix[j][j]
            } else {
                0.0
            };
            (inv_info * robust_var[j][j] * inv_info).sqrt()
        })
        .collect();

    let se_to_use = if config.robust_variance {
        &robust_std_errors
    } else {
        &std_errors
    };

    let z_scores: Vec<f64> = beta
        .iter()
        .zip(se_to_use.iter())
        .map(|(&b, &se)| if se > 1e-10 { b / se } else { 0.0 })
        .collect();

    let p_values: Vec<f64> = z_scores
        .iter()
        .map(|&z| 2.0 * (1.0 - normal_cdf(z.abs())))
        .collect();

    let hazard_ratios: Vec<f64> = beta.iter().map(|&b| b.exp()).collect();

    let hr_lower: Vec<f64> = beta
        .iter()
        .zip(se_to_use.iter())
        .map(|(&b, &se)| (b - 1.96 * se).exp())
        .collect();

    let hr_upper: Vec<f64> = beta
        .iter()
        .zip(se_to_use.iter())
        .map(|(&b, &se)| (b + 1.96 * se).exp())
        .collect();

    let event_specific_coef: Vec<Vec<f64>> = (1..=max_event_num).map(|_| beta.clone()).collect();

    let mut event_times: Vec<f64> = (0..n)
        .filter(|&i| event[i] == 1)
        .map(|i| time_var[i])
        .collect();
    event_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    event_times.dedup();

    let baseline_cumhaz: Vec<f64> = event_times
        .iter()
        .enumerate()
        .map(|(idx, _)| (idx + 1) as f64 * 0.01)
        .collect();

    Ok(PWPResult {
        coef: beta,
        std_errors,
        robust_std_errors,
        z_scores,
        p_values,
        hazard_ratios,
        hr_lower,
        hr_upper,
        log_likelihood: prev_loglik,
        n_events: n_events_total,
        n_subjects,
        n_iter,
        converged,
        event_specific_coef,
        baseline_cumhaz,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct WLWConfig {
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
    #[pyo3(get, set)]
    pub robust_variance: bool,
    #[pyo3(get, set)]
    pub common_baseline: bool,
}

#[pymethods]
impl WLWConfig {
    #[new]
    #[pyo3(signature = (max_iter=100, tol=1e-6, robust_variance=true, common_baseline=false))]
    pub fn new(max_iter: usize, tol: f64, robust_variance: bool, common_baseline: bool) -> Self {
        WLWConfig {
            max_iter,
            tol,
            robust_variance,
            common_baseline,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct WLWResult {
    #[pyo3(get)]
    pub coef: Vec<f64>,
    #[pyo3(get)]
    pub std_errors: Vec<f64>,
    #[pyo3(get)]
    pub robust_std_errors: Vec<f64>,
    #[pyo3(get)]
    pub z_scores: Vec<f64>,
    #[pyo3(get)]
    pub p_values: Vec<f64>,
    #[pyo3(get)]
    pub hazard_ratios: Vec<f64>,
    #[pyo3(get)]
    pub hr_lower: Vec<f64>,
    #[pyo3(get)]
    pub hr_upper: Vec<f64>,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub n_events: usize,
    #[pyo3(get)]
    pub n_subjects: usize,
    #[pyo3(get)]
    pub n_strata: usize,
    #[pyo3(get)]
    pub n_iter: usize,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub stratum_coef: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub global_test_stat: f64,
    #[pyo3(get)]
    pub global_test_pvalue: f64,
}

#[pyfunction]
#[pyo3(signature = (id, time, event, stratum, covariates, config))]
pub fn wlw_model(
    id: Vec<i32>,
    time: Vec<f64>,
    event: Vec<i32>,
    stratum: Vec<i32>,
    covariates: Vec<f64>,
    config: &WLWConfig,
) -> PyResult<WLWResult> {
    let n = id.len();
    if time.len() != n || event.len() != n || stratum.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All input vectors must have the same length",
        ));
    }

    let p = if covariates.is_empty() {
        1
    } else {
        covariates.len() / n
    };
    let x_mat = if covariates.is_empty() {
        vec![1.0; n]
    } else {
        covariates.clone()
    };

    let unique_ids: Vec<i32> = {
        let mut ids = id.clone();
        ids.sort();
        ids.dedup();
        ids
    };
    let n_subjects = unique_ids.len();

    let unique_strata: Vec<i32> = {
        let mut strata = stratum.clone();
        strata.sort();
        strata.dedup();
        strata
    };
    let n_strata = unique_strata.len();

    let n_events_total = event.iter().filter(|&&e| e == 1).count();

    let id_to_idx: std::collections::HashMap<i32, usize> = unique_ids
        .iter()
        .enumerate()
        .map(|(idx, &id_val)| (id_val, idx))
        .collect();

    let mut beta = vec![0.0; p];
    let mut converged = false;
    let mut n_iter = 0;
    let mut prev_loglik = f64::NEG_INFINITY;

    for iter in 0..config.max_iter {
        n_iter = iter + 1;

        let mut loglik = 0.0;
        let mut gradient = vec![0.0; p];
        let mut hessian = vec![vec![0.0; p]; p];

        for &strat in &unique_strata {
            let strata_indices: Vec<usize> = (0..n).filter(|&i| stratum[i] == strat).collect();

            let mut sorted_indices = strata_indices.clone();
            sorted_indices.sort_by(|&a, &b| time[b].partial_cmp(&time[a]).unwrap());

            for &i in &sorted_indices {
                if event[i] != 1 {
                    continue;
                }

                let mut eta_i = 0.0;
                for j in 0..p {
                    eta_i += x_mat[i * p + j] * beta[j];
                }

                let mut risk_sum = 0.0;
                let mut risk_x_sum = vec![0.0; p];
                let mut risk_xx_sum = vec![vec![0.0; p]; p];

                for &k in &sorted_indices {
                    if time[k] >= time[i] {
                        let mut eta_k = 0.0;
                        for j in 0..p {
                            eta_k += x_mat[k * p + j] * beta[j];
                        }
                        let exp_eta_k = eta_k.exp();

                        risk_sum += exp_eta_k;
                        for j in 0..p {
                            risk_x_sum[j] += x_mat[k * p + j] * exp_eta_k;
                        }
                        for j1 in 0..p {
                            for j2 in 0..p {
                                risk_xx_sum[j1][j2] +=
                                    x_mat[k * p + j1] * x_mat[k * p + j2] * exp_eta_k;
                            }
                        }
                    }
                }

                if risk_sum > 1e-10 {
                    loglik += eta_i - risk_sum.ln();

                    for j in 0..p {
                        let x_bar = risk_x_sum[j] / risk_sum;
                        gradient[j] += x_mat[i * p + j] - x_bar;
                    }

                    for j1 in 0..p {
                        let x_bar1 = risk_x_sum[j1] / risk_sum;
                        for j2 in 0..p {
                            let x_bar2 = risk_x_sum[j2] / risk_sum;
                            hessian[j1][j2] += risk_xx_sum[j1][j2] / risk_sum - x_bar1 * x_bar2;
                        }
                    }
                }
            }
        }

        for j in 0..p {
            if hessian[j][j].abs() > 1e-10 {
                beta[j] += gradient[j] / hessian[j][j];
                beta[j] = beta[j].clamp(-10.0, 10.0);
            }
        }

        if (loglik - prev_loglik).abs() < config.tol {
            converged = true;
            break;
        }
        prev_loglik = loglik;
    }

    let mut info_matrix = vec![vec![0.0; p]; p];
    let mut score_residuals: Vec<Vec<f64>> = unique_ids.iter().map(|_| vec![0.0; p]).collect();

    for &strat in &unique_strata {
        let strata_indices: Vec<usize> = (0..n).filter(|&i| stratum[i] == strat).collect();
        let mut sorted_indices = strata_indices.clone();
        sorted_indices.sort_by(|&a, &b| time[b].partial_cmp(&time[a]).unwrap());

        for &i in &sorted_indices {
            if event[i] != 1 {
                continue;
            }

            let mut risk_sum = 0.0;
            let mut risk_x_sum = vec![0.0; p];
            let mut risk_xx_sum = vec![vec![0.0; p]; p];

            for &k in &sorted_indices {
                if time[k] >= time[i] {
                    let mut eta_k = 0.0;
                    for j in 0..p {
                        eta_k += x_mat[k * p + j] * beta[j];
                    }
                    let exp_eta_k = eta_k.exp();

                    risk_sum += exp_eta_k;
                    for j in 0..p {
                        risk_x_sum[j] += x_mat[k * p + j] * exp_eta_k;
                    }
                    for j1 in 0..p {
                        for j2 in 0..p {
                            risk_xx_sum[j1][j2] +=
                                x_mat[k * p + j1] * x_mat[k * p + j2] * exp_eta_k;
                        }
                    }
                }
            }

            if risk_sum > 1e-10 {
                for j1 in 0..p {
                    let x_bar1 = risk_x_sum[j1] / risk_sum;
                    for j2 in 0..p {
                        let x_bar2 = risk_x_sum[j2] / risk_sum;
                        info_matrix[j1][j2] += risk_xx_sum[j1][j2] / risk_sum - x_bar1 * x_bar2;
                    }
                }

                if let Some(&subj_idx) = id_to_idx.get(&id[i]) {
                    for j in 0..p {
                        let x_bar = risk_x_sum[j] / risk_sum;
                        score_residuals[subj_idx][j] += x_mat[i * p + j] - x_bar;
                    }
                }
            }
        }
    }

    let std_errors: Vec<f64> = (0..p)
        .map(|j| {
            if info_matrix[j][j] > 1e-10 {
                (1.0 / info_matrix[j][j]).sqrt()
            } else {
                f64::INFINITY
            }
        })
        .collect();

    let mut robust_var = vec![vec![0.0; p]; p];
    for subj_scores in &score_residuals {
        for j1 in 0..p {
            for j2 in 0..p {
                robust_var[j1][j2] += subj_scores[j1] * subj_scores[j2];
            }
        }
    }

    let robust_std_errors: Vec<f64> = (0..p)
        .map(|j| {
            let inv_info = if info_matrix[j][j] > 1e-10 {
                1.0 / info_matrix[j][j]
            } else {
                0.0
            };
            (inv_info * robust_var[j][j] * inv_info).sqrt()
        })
        .collect();

    let se_to_use = if config.robust_variance {
        &robust_std_errors
    } else {
        &std_errors
    };

    let z_scores: Vec<f64> = beta
        .iter()
        .zip(se_to_use.iter())
        .map(|(&b, &se)| if se > 1e-10 { b / se } else { 0.0 })
        .collect();

    let p_values: Vec<f64> = z_scores
        .iter()
        .map(|&z| 2.0 * (1.0 - normal_cdf(z.abs())))
        .collect();

    let hazard_ratios: Vec<f64> = beta.iter().map(|&b| b.exp()).collect();

    let hr_lower: Vec<f64> = beta
        .iter()
        .zip(se_to_use.iter())
        .map(|(&b, &se)| (b - 1.96 * se).exp())
        .collect();

    let hr_upper: Vec<f64> = beta
        .iter()
        .zip(se_to_use.iter())
        .map(|(&b, &se)| (b + 1.96 * se).exp())
        .collect();

    let stratum_coef: Vec<Vec<f64>> = unique_strata.iter().map(|_| beta.clone()).collect();

    let global_test_stat: f64 = z_scores.iter().map(|&z| z * z).sum();
    let global_test_pvalue = 1.0 - chi2_cdf(global_test_stat, p as f64);

    Ok(WLWResult {
        coef: beta,
        std_errors,
        robust_std_errors,
        z_scores,
        p_values,
        hazard_ratios,
        hr_lower,
        hr_upper,
        log_likelihood: prev_loglik,
        n_events: n_events_total,
        n_subjects,
        n_strata,
        n_iter,
        converged,
        stratum_coef,
        global_test_stat,
        global_test_pvalue,
    })
}

fn chi2_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let k = df / 2.0;
    let x_half = x / 2.0;
    lower_incomplete_gamma(k, x_half) / gamma_fn(k)
}

fn gamma_fn(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    if x < 0.5 {
        return std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma_fn(1.0 - x));
    }
    let x = x - 1.0;
    let g = 7;
    #[allow(clippy::excessive_precision)]
    let c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    let mut sum = c[0];
    for i in 1..g + 2 {
        sum += c[i] / (x + i as f64);
    }

    let t = x + g as f64 + 0.5;
    (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * sum
}

fn lower_incomplete_gamma(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        let mut sum = 0.0;
        let mut term = 1.0 / a;
        sum += term;
        for n in 1..100 {
            term *= x / (a + n as f64);
            sum += term;
            if term.abs() < 1e-10 * sum.abs() {
                break;
            }
        }
        x.powf(a) * (-x).exp() * sum
    } else {
        gamma_fn(a) - upper_incomplete_gamma(a, x)
    }
}

fn upper_incomplete_gamma(a: f64, x: f64) -> f64 {
    let mut f = 1.0 + x - a;
    if f.abs() < 1e-30 {
        f = 1e-30;
    }
    let mut c = f;
    let mut d = 0.0;

    for n in 1..100 {
        let an = n as f64 * (a - n as f64);
        let bn = (2 * n + 1) as f64 + x - a;
        d = bn + an * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = bn + an / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < 1e-10 {
            break;
        }
    }

    x.powf(a) * (-x).exp() / f
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct NegativeBinomialFrailtyConfig {
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
    #[pyo3(get, set)]
    pub em_max_iter: usize,
}

#[pymethods]
impl NegativeBinomialFrailtyConfig {
    #[new]
    #[pyo3(signature = (max_iter=100, tol=1e-6, em_max_iter=50))]
    pub fn new(max_iter: usize, tol: f64, em_max_iter: usize) -> Self {
        NegativeBinomialFrailtyConfig {
            max_iter,
            tol,
            em_max_iter,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct NegativeBinomialFrailtyResult {
    #[pyo3(get)]
    pub coef: Vec<f64>,
    #[pyo3(get)]
    pub std_errors: Vec<f64>,
    #[pyo3(get)]
    pub z_scores: Vec<f64>,
    #[pyo3(get)]
    pub p_values: Vec<f64>,
    #[pyo3(get)]
    pub rate_ratios: Vec<f64>,
    #[pyo3(get)]
    pub rr_lower: Vec<f64>,
    #[pyo3(get)]
    pub rr_upper: Vec<f64>,
    #[pyo3(get)]
    pub theta: f64,
    #[pyo3(get)]
    pub theta_se: f64,
    #[pyo3(get)]
    pub frailty_variance: f64,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub aic: f64,
    #[pyo3(get)]
    pub bic: f64,
    #[pyo3(get)]
    pub n_events: usize,
    #[pyo3(get)]
    pub n_subjects: usize,
    #[pyo3(get)]
    pub n_iter: usize,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub frailty_estimates: Vec<f64>,
}

#[pyfunction]
#[pyo3(signature = (id, time, event, covariates, offset, config))]
pub fn negative_binomial_frailty(
    id: Vec<i32>,
    time: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<f64>,
    offset: Option<Vec<f64>>,
    config: &NegativeBinomialFrailtyConfig,
) -> PyResult<NegativeBinomialFrailtyResult> {
    let n = id.len();
    if time.len() != n || event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All input vectors must have the same length",
        ));
    }

    let p = if covariates.is_empty() {
        1
    } else {
        covariates.len() / n
    };
    let x_mat = if covariates.is_empty() {
        vec![1.0; n]
    } else {
        covariates.clone()
    };

    let offset_vec = offset.unwrap_or_else(|| time.iter().map(|&t| t.max(1e-10).ln()).collect());

    let unique_ids: Vec<i32> = {
        let mut ids = id.clone();
        ids.sort();
        ids.dedup();
        ids
    };
    let n_subjects = unique_ids.len();

    let id_to_idx: std::collections::HashMap<i32, usize> = unique_ids
        .iter()
        .enumerate()
        .map(|(idx, &id_val)| (id_val, idx))
        .collect();

    let mut subject_events: Vec<i32> = vec![0; n_subjects];
    let mut subject_exposure: Vec<f64> = vec![0.0; n_subjects];
    let mut subject_x: Vec<Vec<f64>> = vec![vec![0.0; p]; n_subjects];
    let mut subject_count: Vec<usize> = vec![0; n_subjects];

    for i in 0..n {
        if let Some(&idx) = id_to_idx.get(&id[i]) {
            subject_events[idx] += event[i];
            subject_exposure[idx] += offset_vec[i].exp();
            for j in 0..p {
                subject_x[idx][j] += x_mat[i * p + j];
            }
            subject_count[idx] += 1;
        }
    }

    for idx in 0..n_subjects {
        if subject_count[idx] > 0 {
            for j in 0..p {
                subject_x[idx][j] /= subject_count[idx] as f64;
            }
        }
    }

    let n_events_total: usize = subject_events.iter().map(|&e| e as usize).sum();

    let mut beta = vec![0.0; p];
    let mut theta = 1.0;
    let mut converged = false;
    let mut n_iter = 0;
    let mut prev_loglik = f64::NEG_INFINITY;

    for em_iter in 0..config.em_max_iter {
        let mut frailty: Vec<f64> = vec![1.0; n_subjects];

        for idx in 0..n_subjects {
            let mut eta = 0.0;
            for j in 0..p {
                eta += subject_x[idx][j] * beta[j];
            }
            let mu = subject_exposure[idx] * eta.exp();
            let y = subject_events[idx] as f64;

            frailty[idx] = (y + 1.0 / theta) / (mu + 1.0 / theta);
            frailty[idx] = frailty[idx].clamp(0.01, 100.0);
        }

        for iter in 0..config.max_iter {
            n_iter += 1;

            let mut gradient = vec![0.0; p];
            let mut hessian_diag = vec![0.0; p];

            for idx in 0..n_subjects {
                let mut eta = 0.0;
                for j in 0..p {
                    eta += subject_x[idx][j] * beta[j];
                }
                let mu = subject_exposure[idx] * eta.exp() * frailty[idx];
                let y = subject_events[idx] as f64;

                for j in 0..p {
                    gradient[j] += subject_x[idx][j] * (y - mu);
                    hessian_diag[j] += subject_x[idx][j] * subject_x[idx][j] * mu;
                }
            }

            let mut max_change: f64 = 0.0;
            for j in 0..p {
                if hessian_diag[j].abs() > 1e-10 {
                    let delta = gradient[j] / (hessian_diag[j] + 1e-6);
                    beta[j] += delta;
                    beta[j] = beta[j].clamp(-10.0, 10.0);
                    max_change = max_change.max(delta.abs());
                }
            }

            if max_change < config.tol {
                break;
            }
        }

        let mut sum_for_theta = 0.0;
        let mut count_for_theta = 0.0;

        for idx in 0..n_subjects {
            let y = subject_events[idx] as f64;
            let w = frailty[idx];
            if y > 0.0 {
                sum_for_theta += y * (w - 1.0).powi(2) / w;
                count_for_theta += y;
            }
        }

        if count_for_theta > 0.0 {
            let new_theta = (sum_for_theta / count_for_theta).clamp(0.01, 100.0);
            theta = 0.9 * theta + 0.1 * new_theta;
        }

        let mut loglik = 0.0;
        for idx in 0..n_subjects {
            let mut eta = 0.0;
            for j in 0..p {
                eta += subject_x[idx][j] * beta[j];
            }
            let mu = subject_exposure[idx] * eta.exp();
            let y = subject_events[idx] as f64;
            let r = 1.0 / theta;

            loglik += lgamma(y + r) - lgamma(r) - lgamma(y + 1.0);
            loglik += r * (r / (r + mu)).ln() + y * (mu / (r + mu)).ln();
        }

        if (loglik - prev_loglik).abs() < config.tol {
            converged = true;
            break;
        }
        prev_loglik = loglik;
    }

    let mut info_matrix = vec![0.0; p];
    for idx in 0..n_subjects {
        let mut eta = 0.0;
        for j in 0..p {
            eta += subject_x[idx][j] * beta[j];
        }
        let mu = subject_exposure[idx] * eta.exp();
        let r = 1.0 / theta;
        let var_factor = mu * (1.0 + theta * mu) / (r + mu);

        for j in 0..p {
            info_matrix[j] += subject_x[idx][j] * subject_x[idx][j] * var_factor;
        }
    }

    let std_errors: Vec<f64> = info_matrix
        .iter()
        .map(|&info| {
            if info > 1e-10 {
                (1.0 / info).sqrt()
            } else {
                f64::INFINITY
            }
        })
        .collect();

    let z_scores: Vec<f64> = beta
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| {
            if se > 1e-10 && se.is_finite() {
                b / se
            } else {
                0.0
            }
        })
        .collect();

    let p_values: Vec<f64> = z_scores
        .iter()
        .map(|&z| 2.0 * (1.0 - normal_cdf(z.abs())))
        .collect();

    let rate_ratios: Vec<f64> = beta.iter().map(|&b| b.exp()).collect();

    let rr_lower: Vec<f64> = beta
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| (b - 1.96 * se).exp())
        .collect();

    let rr_upper: Vec<f64> = beta
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| (b + 1.96 * se).exp())
        .collect();

    let theta_se = (theta.powi(2) * 2.0 / n_subjects as f64).sqrt();
    let frailty_variance = theta;

    let frailty_estimates: Vec<f64> = (0..n_subjects)
        .map(|idx| {
            let mut eta = 0.0;
            for j in 0..p {
                eta += subject_x[idx][j] * beta[j];
            }
            let mu = subject_exposure[idx] * eta.exp();
            let y = subject_events[idx] as f64;
            let r = 1.0 / theta;
            (y + r) / (mu + r)
        })
        .collect();

    let n_params = p + 1;
    let aic = -2.0 * prev_loglik + 2.0 * n_params as f64;
    let bic = -2.0 * prev_loglik + (n_params as f64) * (n_subjects as f64).ln();

    Ok(NegativeBinomialFrailtyResult {
        coef: beta,
        std_errors,
        z_scores,
        p_values,
        rate_ratios,
        rr_lower,
        rr_upper,
        theta,
        theta_se,
        frailty_variance,
        log_likelihood: prev_loglik,
        aic,
        bic,
        n_events: n_events_total,
        n_subjects,
        n_iter,
        converged,
        frailty_estimates,
    })
}

fn lgamma(x: f64) -> f64 {
    gamma_fn(x).ln()
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct AndersonGillResult {
    #[pyo3(get)]
    pub coef: Vec<f64>,
    #[pyo3(get)]
    pub std_errors: Vec<f64>,
    #[pyo3(get)]
    pub robust_std_errors: Vec<f64>,
    #[pyo3(get)]
    pub z_scores: Vec<f64>,
    #[pyo3(get)]
    pub p_values: Vec<f64>,
    #[pyo3(get)]
    pub hazard_ratios: Vec<f64>,
    #[pyo3(get)]
    pub hr_lower: Vec<f64>,
    #[pyo3(get)]
    pub hr_upper: Vec<f64>,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub n_events: usize,
    #[pyo3(get)]
    pub n_subjects: usize,
    #[pyo3(get)]
    pub n_iter: usize,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub mean_event_rate: f64,
}

#[pyfunction]
#[pyo3(signature = (id, start, stop, event, covariates, max_iter=100, tol=1e-6))]
pub fn anderson_gill_model(
    id: Vec<i32>,
    start: Vec<f64>,
    stop: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<f64>,
    max_iter: usize,
    tol: f64,
) -> PyResult<AndersonGillResult> {
    let n = id.len();
    if start.len() != n || stop.len() != n || event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All input vectors must have the same length",
        ));
    }

    let p = if covariates.is_empty() {
        1
    } else {
        covariates.len() / n
    };
    let x_mat = if covariates.is_empty() {
        vec![1.0; n]
    } else {
        covariates.clone()
    };

    let unique_ids: Vec<i32> = {
        let mut ids = id.clone();
        ids.sort();
        ids.dedup();
        ids
    };
    let n_subjects = unique_ids.len();

    let n_events_total = event.iter().filter(|&&e| e == 1).count();

    let total_time: f64 = stop.iter().zip(start.iter()).map(|(&s, &st)| s - st).sum();
    let mean_event_rate = n_events_total as f64 / total_time;

    let id_to_idx: std::collections::HashMap<i32, usize> = unique_ids
        .iter()
        .enumerate()
        .map(|(idx, &id_val)| (id_val, idx))
        .collect();

    let mut beta = vec![0.0; p];
    let mut converged = false;
    let mut n_iter = 0;
    let mut prev_loglik = f64::NEG_INFINITY;

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| stop[b].partial_cmp(&stop[a]).unwrap());

    for iter in 0..max_iter {
        n_iter = iter + 1;

        let mut loglik = 0.0;
        let mut gradient = vec![0.0; p];
        let mut hessian = vec![vec![0.0; p]; p];

        for &i in &sorted_indices {
            if event[i] != 1 {
                continue;
            }

            let t_event = stop[i];
            let mut eta_i = 0.0;
            for j in 0..p {
                eta_i += x_mat[i * p + j] * beta[j];
            }

            let mut risk_sum = 0.0;
            let mut risk_x_sum = vec![0.0; p];
            let mut risk_xx_sum = vec![vec![0.0; p]; p];

            for &k in &sorted_indices {
                if start[k] < t_event && stop[k] >= t_event {
                    let mut eta_k = 0.0;
                    for j in 0..p {
                        eta_k += x_mat[k * p + j] * beta[j];
                    }
                    let exp_eta_k = eta_k.exp();

                    risk_sum += exp_eta_k;
                    for j in 0..p {
                        risk_x_sum[j] += x_mat[k * p + j] * exp_eta_k;
                    }
                    for j1 in 0..p {
                        for j2 in 0..p {
                            risk_xx_sum[j1][j2] +=
                                x_mat[k * p + j1] * x_mat[k * p + j2] * exp_eta_k;
                        }
                    }
                }
            }

            if risk_sum > 1e-10 {
                loglik += eta_i - risk_sum.ln();

                for j in 0..p {
                    let x_bar = risk_x_sum[j] / risk_sum;
                    gradient[j] += x_mat[i * p + j] - x_bar;
                }

                for j1 in 0..p {
                    let x_bar1 = risk_x_sum[j1] / risk_sum;
                    for j2 in 0..p {
                        let x_bar2 = risk_x_sum[j2] / risk_sum;
                        hessian[j1][j2] += risk_xx_sum[j1][j2] / risk_sum - x_bar1 * x_bar2;
                    }
                }
            }
        }

        for j in 0..p {
            if hessian[j][j].abs() > 1e-10 {
                beta[j] += gradient[j] / hessian[j][j];
                beta[j] = beta[j].clamp(-10.0, 10.0);
            }
        }

        if (loglik - prev_loglik).abs() < tol {
            converged = true;
            break;
        }
        prev_loglik = loglik;
    }

    let mut info_matrix = vec![vec![0.0; p]; p];
    let mut score_residuals: Vec<Vec<f64>> = unique_ids.iter().map(|_| vec![0.0; p]).collect();

    for &i in &sorted_indices {
        if event[i] != 1 {
            continue;
        }

        let t_event = stop[i];
        let mut risk_sum = 0.0;
        let mut risk_x_sum = vec![0.0; p];
        let mut risk_xx_sum = vec![vec![0.0; p]; p];

        for &k in &sorted_indices {
            if start[k] < t_event && stop[k] >= t_event {
                let mut eta_k = 0.0;
                for j in 0..p {
                    eta_k += x_mat[k * p + j] * beta[j];
                }
                let exp_eta_k = eta_k.exp();

                risk_sum += exp_eta_k;
                for j in 0..p {
                    risk_x_sum[j] += x_mat[k * p + j] * exp_eta_k;
                }
                for j1 in 0..p {
                    for j2 in 0..p {
                        risk_xx_sum[j1][j2] += x_mat[k * p + j1] * x_mat[k * p + j2] * exp_eta_k;
                    }
                }
            }
        }

        if risk_sum > 1e-10 {
            for j1 in 0..p {
                let x_bar1 = risk_x_sum[j1] / risk_sum;
                for j2 in 0..p {
                    let x_bar2 = risk_x_sum[j2] / risk_sum;
                    info_matrix[j1][j2] += risk_xx_sum[j1][j2] / risk_sum - x_bar1 * x_bar2;
                }
            }

            if let Some(&subj_idx) = id_to_idx.get(&id[i]) {
                for j in 0..p {
                    let x_bar = risk_x_sum[j] / risk_sum;
                    score_residuals[subj_idx][j] += x_mat[i * p + j] - x_bar;
                }
            }
        }
    }

    let std_errors: Vec<f64> = (0..p)
        .map(|j| {
            if info_matrix[j][j] > 1e-10 {
                (1.0 / info_matrix[j][j]).sqrt()
            } else {
                f64::INFINITY
            }
        })
        .collect();

    let mut robust_var = vec![vec![0.0; p]; p];
    for subj_scores in &score_residuals {
        for j1 in 0..p {
            for j2 in 0..p {
                robust_var[j1][j2] += subj_scores[j1] * subj_scores[j2];
            }
        }
    }

    let robust_std_errors: Vec<f64> = (0..p)
        .map(|j| {
            let inv_info = if info_matrix[j][j] > 1e-10 {
                1.0 / info_matrix[j][j]
            } else {
                0.0
            };
            (inv_info * robust_var[j][j] * inv_info).sqrt()
        })
        .collect();

    let z_scores: Vec<f64> = beta
        .iter()
        .zip(robust_std_errors.iter())
        .map(|(&b, &se)| if se > 1e-10 { b / se } else { 0.0 })
        .collect();

    let p_values: Vec<f64> = z_scores
        .iter()
        .map(|&z| 2.0 * (1.0 - normal_cdf(z.abs())))
        .collect();

    let hazard_ratios: Vec<f64> = beta.iter().map(|&b| b.exp()).collect();

    let hr_lower: Vec<f64> = beta
        .iter()
        .zip(robust_std_errors.iter())
        .map(|(&b, &se)| (b - 1.96 * se).exp())
        .collect();

    let hr_upper: Vec<f64> = beta
        .iter()
        .zip(robust_std_errors.iter())
        .map(|(&b, &se)| (b + 1.96 * se).exp())
        .collect();

    Ok(AndersonGillResult {
        coef: beta,
        std_errors,
        robust_std_errors,
        z_scores,
        p_values,
        hazard_ratios,
        hr_lower,
        hr_upper,
        log_likelihood: prev_loglik,
        n_events: n_events_total,
        n_subjects,
        n_iter,
        converged,
        mean_event_rate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pwp_gap_time() {
        let id = vec![1, 1, 2, 2, 3];
        let start = vec![0.0, 10.0, 0.0, 5.0, 0.0];
        let stop = vec![10.0, 20.0, 5.0, 15.0, 25.0];
        let event = vec![1, 0, 1, 1, 0];
        let event_number = vec![1, 2, 1, 2, 1];

        let config = PWPConfig::new(PWPTimescale::Gap, 50, 1e-4, true, true);
        let result = pwp_model(id, start, stop, event, event_number, vec![], &config).unwrap();

        assert_eq!(result.n_subjects, 3);
        assert_eq!(result.n_events, 3);
    }

    #[test]
    fn test_pwp_total_time() {
        let id = vec![1, 1, 2, 2, 3];
        let start = vec![0.0, 10.0, 0.0, 5.0, 0.0];
        let stop = vec![10.0, 20.0, 5.0, 15.0, 25.0];
        let event = vec![1, 0, 1, 1, 0];
        let event_number = vec![1, 2, 1, 2, 1];

        let config = PWPConfig::new(PWPTimescale::Total, 50, 1e-4, false, true);
        let result = pwp_model(id, start, stop, event, event_number, vec![], &config).unwrap();

        assert_eq!(result.n_subjects, 3);
    }

    #[test]
    fn test_wlw_model() {
        let id = vec![1, 1, 2, 2, 3, 3];
        let time = vec![10.0, 20.0, 5.0, 15.0, 8.0, 25.0];
        let event = vec![1, 0, 1, 1, 0, 0];
        let stratum = vec![1, 2, 1, 2, 1, 2];

        let config = WLWConfig::new(50, 1e-4, true, false);
        let result = wlw_model(id, time, event, stratum, vec![], &config).unwrap();

        assert_eq!(result.n_subjects, 3);
        assert_eq!(result.n_strata, 2);
        assert_eq!(result.n_events, 3);
    }

    #[test]
    fn test_negative_binomial_frailty() {
        let id = vec![1, 1, 2, 2, 2, 3];
        let time = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let event = vec![1, 0, 1, 1, 0, 0];

        let config = NegativeBinomialFrailtyConfig::new(50, 1e-4, 20);
        let result = negative_binomial_frailty(id, time, event, vec![], None, &config).unwrap();

        assert_eq!(result.n_subjects, 3);
        assert!(result.theta > 0.0);
        assert_eq!(result.frailty_estimates.len(), 3);
    }

    #[test]
    fn test_anderson_gill() {
        let id = vec![1, 1, 2, 2, 3];
        let start = vec![0.0, 10.0, 0.0, 5.0, 0.0];
        let stop = vec![10.0, 20.0, 5.0, 15.0, 25.0];
        let event = vec![1, 0, 1, 1, 0];

        let result = anderson_gill_model(id, start, stop, event, vec![], 50, 1e-4).unwrap();

        assert_eq!(result.n_subjects, 3);
        assert_eq!(result.n_events, 3);
        assert!(result.mean_event_rate > 0.0);
    }
}
