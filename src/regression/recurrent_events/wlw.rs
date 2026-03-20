
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
