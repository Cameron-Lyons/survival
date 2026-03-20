
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

