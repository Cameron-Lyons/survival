use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utilities::statistical::{chi2_cdf, ln_gamma, normal_cdf};

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(from_py_object)]
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
#[pyclass(from_py_object)]
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
#[pyclass(from_py_object)]
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
        sorted_indices.sort_by(|&a, &b| {
            time_var[b]
                .partial_cmp(&time_var[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

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

            if risk_sum > crate::constants::DIVISION_FLOOR {
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
            inv_hess[j][j] = if hessian[j][j].abs() > crate::constants::DIVISION_FLOOR {
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
    sorted_indices.sort_by(|&a, &b| {
        time_var[b]
            .partial_cmp(&time_var[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

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

        if risk_sum > crate::constants::DIVISION_FLOOR {
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
            if info_matrix[j][j] > crate::constants::DIVISION_FLOOR {
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
            let inv_info = if info_matrix[j][j] > crate::constants::DIVISION_FLOOR {
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
        .map(|(&b, &se)| {
            if se > crate::constants::DIVISION_FLOOR {
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
    event_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
#[pyclass(from_py_object)]
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
#[pyclass(from_py_object)]
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
            sorted_indices.sort_by(|&a, &b| {
                time[b]
                    .partial_cmp(&time[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

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

                if risk_sum > crate::constants::DIVISION_FLOOR {
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
            if hessian[j][j].abs() > crate::constants::DIVISION_FLOOR {
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
        sorted_indices.sort_by(|&a, &b| {
            time[b]
                .partial_cmp(&time[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

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

            if risk_sum > crate::constants::DIVISION_FLOOR {
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
            if info_matrix[j][j] > crate::constants::DIVISION_FLOOR {
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
            let inv_info = if info_matrix[j][j] > crate::constants::DIVISION_FLOOR {
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
        .map(|(&b, &se)| {
            if se > crate::constants::DIVISION_FLOOR {
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

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
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
#[pyclass(from_py_object)]
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

    let offset_vec = offset.unwrap_or_else(|| {
        time.iter()
            .map(|&t| t.max(crate::constants::DIVISION_FLOOR).ln())
            .collect()
    });

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

    subject_x
        .par_iter_mut()
        .zip(subject_count.par_iter())
        .for_each(|(x_row, &count)| {
            if count > 0 {
                for x_j in x_row.iter_mut().take(p) {
                    *x_j /= count as f64;
                }
            }
        });

    let n_events_total: usize = subject_events.iter().map(|&e| e as usize).sum();

    let mut beta = vec![0.0; p];
    let mut theta = 1.0;
    let mut converged = false;
    let mut n_iter = 0;
    let mut prev_loglik = f64::NEG_INFINITY;

    for _em_iter in 0..config.em_max_iter {
        let frailty: Vec<f64> = (0..n_subjects)
            .into_par_iter()
            .map(|idx| {
                let eta: f64 = subject_x[idx]
                    .iter()
                    .zip(beta.iter())
                    .map(|(&x, &b)| x * b)
                    .sum();
                let mu = subject_exposure[idx] * eta.exp();
                let y = subject_events[idx] as f64;
                ((y + 1.0 / theta) / (mu + 1.0 / theta)).clamp(0.01, 100.0)
            })
            .collect();
        for _iter in 0..config.max_iter {
            n_iter += 1;

            let (gradient, hessian_diag) = (0..n_subjects)
                .into_par_iter()
                .fold(
                    || (vec![0.0; p], vec![0.0; p]),
                    |(mut gradient, mut hessian_diag), idx| {
                        let eta: f64 = subject_x[idx]
                            .iter()
                            .zip(beta.iter())
                            .map(|(&x, &b)| x * b)
                            .sum();
                        let mu = subject_exposure[idx] * eta.exp() * frailty[idx];
                        let y = subject_events[idx] as f64;

                        for j in 0..p {
                            gradient[j] += subject_x[idx][j] * (y - mu);
                            hessian_diag[j] += subject_x[idx][j] * subject_x[idx][j] * mu;
                        }
                        (gradient, hessian_diag)
                    },
                )
                .reduce(
                    || (vec![0.0; p], vec![0.0; p]),
                    |(mut g1, mut h1), (g2, h2)| {
                        for j in 0..p {
                            g1[j] += g2[j];
                            h1[j] += h2[j];
                        }
                        (g1, h1)
                    },
                );

            let mut max_change: f64 = 0.0;
            for j in 0..p {
                if hessian_diag[j].abs() > crate::constants::DIVISION_FLOOR {
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

        let (sum_for_theta, count_for_theta) = (0..n_subjects)
            .into_par_iter()
            .map(|idx| {
                let y = subject_events[idx] as f64;
                let w = frailty[idx];
                if y > 0.0 {
                    (y * (w - 1.0).powi(2) / w, y)
                } else {
                    (0.0, 0.0)
                }
            })
            .reduce(|| (0.0, 0.0), |(s1, c1), (s2, c2)| (s1 + s2, c1 + c2));

        if count_for_theta > 0.0 {
            let new_theta = (sum_for_theta / count_for_theta).clamp(0.01, 100.0);
            theta = 0.9 * theta + 0.1 * new_theta;
        }

        let loglik: f64 = (0..n_subjects)
            .into_par_iter()
            .map(|idx| {
                let eta: f64 = subject_x[idx]
                    .iter()
                    .zip(beta.iter())
                    .map(|(&x, &b)| x * b)
                    .sum();
                let mu = subject_exposure[idx] * eta.exp();
                let y = subject_events[idx] as f64;
                let r = 1.0 / theta;

                let mut ll = lgamma(y + r) - lgamma(r) - lgamma(y + 1.0);
                ll += r * (r / (r + mu)).ln() + y * (mu / (r + mu)).ln();
                ll
            })
            .sum();

        if (loglik - prev_loglik).abs() < config.tol {
            converged = true;
            break;
        }
        prev_loglik = loglik;
    }

    let info_matrix: Vec<f64> = (0..n_subjects)
        .into_par_iter()
        .fold(
            || vec![0.0; p],
            |mut info, idx| {
                let eta: f64 = subject_x[idx]
                    .iter()
                    .zip(beta.iter())
                    .map(|(&x, &b)| x * b)
                    .sum();
                let mu = subject_exposure[idx] * eta.exp();
                let r = 1.0 / theta;
                let var_factor = mu * (1.0 + theta * mu) / (r + mu);

                for j in 0..p {
                    info[j] += subject_x[idx][j] * subject_x[idx][j] * var_factor;
                }
                info
            },
        )
        .reduce(
            || vec![0.0; p],
            |mut a, b| {
                for j in 0..p {
                    a[j] += b[j];
                }
                a
            },
        );

    let std_errors: Vec<f64> = info_matrix
        .par_iter()
        .map(|&info| {
            if info > crate::constants::DIVISION_FLOOR {
                (1.0 / info).sqrt()
            } else {
                f64::INFINITY
            }
        })
        .collect();

    let z_scores: Vec<f64> = beta
        .par_iter()
        .zip(std_errors.par_iter())
        .map(|(&b, &se)| {
            if se > crate::constants::DIVISION_FLOOR && se.is_finite() {
                b / se
            } else {
                0.0
            }
        })
        .collect();

    let p_values: Vec<f64> = z_scores
        .par_iter()
        .map(|&z| 2.0 * (1.0 - normal_cdf(z.abs())))
        .collect();

    let rate_ratios: Vec<f64> = beta.par_iter().map(|&b| b.exp()).collect();

    let rr_lower: Vec<f64> = beta
        .par_iter()
        .zip(std_errors.par_iter())
        .map(|(&b, &se)| (b - 1.96 * se).exp())
        .collect();

    let rr_upper: Vec<f64> = beta
        .par_iter()
        .zip(std_errors.par_iter())
        .map(|(&b, &se)| (b + 1.96 * se).exp())
        .collect();

    let theta_se = (theta.powi(2) * 2.0 / n_subjects as f64).sqrt();
    let frailty_variance = theta;

    let frailty_estimates: Vec<f64> = (0..n_subjects)
        .into_par_iter()
        .map(|idx| {
            let eta: f64 = subject_x[idx]
                .iter()
                .zip(beta.iter())
                .map(|(&x, &b)| x * b)
                .sum();
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
    ln_gamma(x)
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
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
    sorted_indices.sort_by(|&a, &b| {
        stop[b]
            .partial_cmp(&stop[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

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

            if risk_sum > crate::constants::DIVISION_FLOOR {
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
            if hessian[j][j].abs() > crate::constants::DIVISION_FLOOR {
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

        if risk_sum > crate::constants::DIVISION_FLOOR {
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
            if info_matrix[j][j] > crate::constants::DIVISION_FLOOR {
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
            let inv_info = if info_matrix[j][j] > crate::constants::DIVISION_FLOOR {
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
        .map(|(&b, &se)| {
            if se > crate::constants::DIVISION_FLOOR {
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

    struct LegacyBladderData {
        id: Vec<i32>,
        start: Vec<f64>,
        stop: Vec<f64>,
        event: Vec<i32>,
        event_number: Vec<i32>,
        covariates: Vec<f64>,
        wlw_id: Vec<i32>,
        wlw_time: Vec<f64>,
        wlw_event: Vec<i32>,
        wlw_stratum: Vec<i32>,
        wlw_covariates: Vec<f64>,
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    fn load_legacy_bladder_data() -> LegacyBladderData {
        let csv = include_str!("../datasets/data/bladder.csv");
        let mut rows: Vec<(i32, i32, i32, i32, i32, i32, i32)> = csv
            .lines()
            .skip(1)
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                let values: Vec<i32> = line
                    .split(',')
                    .map(|value| value.trim().parse::<i32>().expect("valid bladder integer"))
                    .collect();
                assert_eq!(values.len(), 8, "unexpected bladder row width");
                (
                    values[1], values[2], values[3], values[4], values[5], values[6], values[7],
                )
            })
            .collect();

        rows.sort_by(|a, b| a.0.cmp(&b.0).then(a.6.cmp(&b.6)));

        let mut id = Vec::new();
        let mut start = Vec::new();
        let mut stop = Vec::new();
        let mut event = Vec::new();
        let mut event_number = Vec::new();
        let mut covariates = Vec::new();

        let mut wlw_id = Vec::with_capacity(rows.len());
        let mut wlw_time = Vec::with_capacity(rows.len());
        let mut wlw_event = Vec::with_capacity(rows.len());
        let mut wlw_stratum = Vec::with_capacity(rows.len());
        let mut wlw_covariates = Vec::with_capacity(rows.len() * 3);

        let mut idx = 0;
        while idx < rows.len() {
            let current_id = rows[idx].0;
            let mut subject_rows = Vec::new();
            while idx < rows.len() && rows[idx].0 == current_id {
                let row = rows[idx];
                subject_rows.push(row);
                wlw_id.push(row.0);
                wlw_time.push(row.4 as f64);
                wlw_event.push(row.5);
                wlw_stratum.push(row.6);
                wlw_covariates.extend([row.1 as f64, row.3 as f64, row.2 as f64]);
                idx += 1;
            }

            let mut previous_stop = 0.0;
            for &(subject_id, rx, number, size, subject_stop, subject_event, subject_enum) in
                &subject_rows
            {
                let subject_stop = subject_stop as f64;
                if subject_event == 1 || subject_stop > previous_stop {
                    id.push(subject_id);
                    start.push(previous_stop);
                    stop.push(subject_stop);
                    event.push(subject_event);
                    event_number.push(subject_enum);
                    covariates.extend([rx as f64, size as f64, number as f64]);
                    previous_stop = subject_stop;
                }
            }
        }

        assert_eq!(wlw_id.len(), rows.len());

        LegacyBladderData {
            id,
            start,
            stop,
            event,
            event_number,
            covariates,
            wlw_id,
            wlw_time,
            wlw_event,
            wlw_stratum,
            wlw_covariates,
        }
    }

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
    fn test_bladder_recurrent_event_models() {
        let bladder = load_legacy_bladder_data();

        assert_eq!(bladder.id.len(), 178);
        assert_eq!(bladder.event.iter().filter(|&&e| e == 1).count(), 112);

        let gap_config = PWPConfig::new(PWPTimescale::Gap, 100, 1e-6, true, true);
        let gap = pwp_model(
            bladder.id.clone(),
            bladder.start.clone(),
            bladder.stop.clone(),
            bladder.event.clone(),
            bladder.event_number.clone(),
            bladder.covariates.clone(),
            &gap_config,
        );
        assert!(gap.is_ok());
        let gap = gap.expect("gap-time PWP result should be present");

        let total_config = PWPConfig::new(PWPTimescale::Total, 100, 1e-6, true, true);
        let total = pwp_model(
            bladder.id.clone(),
            bladder.start.clone(),
            bladder.stop.clone(),
            bladder.event.clone(),
            bladder.event_number.clone(),
            bladder.covariates.clone(),
            &total_config,
        );
        assert!(total.is_ok());
        let total = total.expect("total-time PWP result should be present");

        let ag = anderson_gill_model(
            bladder.id.clone(),
            bladder.start.clone(),
            bladder.stop.clone(),
            bladder.event.clone(),
            bladder.covariates.clone(),
            100,
            1e-6,
        );
        assert!(ag.is_ok());
        let ag = ag.expect("Anderson-Gill result should be present");

        let wlw_config = WLWConfig::new(100, 1e-6, true, false);
        let wlw = wlw_model(
            bladder.wlw_id,
            bladder.wlw_time,
            bladder.wlw_event,
            bladder.wlw_stratum,
            bladder.wlw_covariates,
            &wlw_config,
        );
        assert!(wlw.is_ok());
        let wlw = wlw.expect("WLW result should be present");

        assert_eq!(gap.n_subjects, 85);
        assert_eq!(gap.n_events, 112);
        assert!(gap.converged);
        assert_eq!(gap.event_specific_coef.len(), 4);
        assert_eq!(gap.baseline_cumhaz.len(), 28);
        assert_close(gap.coef[0], -0.2695101867681618, 1e-9);
        assert_close(gap.coef[1], 0.0068363097719597865, 1e-9);
        assert_close(gap.coef[2], 0.15353662917417513, 1e-9);

        assert_eq!(total.n_subjects, 85);
        assert_eq!(total.n_events, 112);
        assert!(total.converged);
        assert_close(total.coef[0], -0.5167094835791394, 1e-9);
        assert_close(total.coef[1], -0.007743184659185533, 1e-9);
        assert_close(total.coef[2], 0.10287711173954855, 1e-9);

        assert_eq!(ag.n_subjects, 85);
        assert_eq!(ag.n_events, 112);
        assert!(ag.converged);
        assert_close(ag.coef[0], -0.45978826074398993, 1e-9);
        assert_close(ag.coef[1], -0.04256340004595282, 1e-9);
        assert_close(ag.coef[2], 0.17164542460626836, 1e-9);

        assert_eq!(wlw.n_subjects, 85);
        assert_eq!(wlw.n_events, 112);
        assert_eq!(wlw.n_strata, 4);
        assert!(wlw.converged);
        assert_close(wlw.coef[0], -0.5798694870405632, 1e-9);
        assert_close(wlw.coef[1], -0.050935433404071695, 1e-9);
        assert_close(wlw.coef[2], 0.20849094150265948, 1e-9);
        assert_close(wlw.global_test_stat, 12.37081136240878, 1e-9);
        assert_close(wlw.global_test_pvalue, 0.006215083463136151, 1e-12);

        assert!(gap.hazard_ratios[0] < 1.0);
        assert!(total.hazard_ratios[0] < gap.hazard_ratios[0]);
        assert!(wlw.hazard_ratios[2] > 1.0);
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
