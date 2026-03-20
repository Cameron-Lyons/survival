
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SuperLandmarkResult {
    #[pyo3(get)]
    pub landmark_times: Vec<f64>,
    #[pyo3(get)]
    pub coefficients: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub std_errors: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub c_indices: Vec<f64>,
    #[pyo3(get)]
    pub brier_scores: Vec<f64>,
    #[pyo3(get)]
    pub n_at_risk: Vec<usize>,
    #[pyo3(get)]
    pub n_events: Vec<usize>,
    #[pyo3(get)]
    pub pooled_coef: Vec<f64>,
    #[pyo3(get)]
    pub pooled_se: Vec<f64>,
}

#[pyfunction]
#[pyo3(signature = (event_time, event_status, covariates, n_vars, landmark_times, horizon, max_iter=50))]
pub fn super_landmark_model(
    event_time: Vec<f64>,
    event_status: Vec<i32>,
    covariates: Vec<f64>,
    n_vars: usize,
    landmark_times: Vec<f64>,
    horizon: f64,
    max_iter: usize,
) -> PyResult<SuperLandmarkResult> {
    let n_subjects = event_time.len();
    let n_landmarks = landmark_times.len();

    let mut all_coefs = Vec::new();
    let mut all_ses = Vec::new();
    let mut c_indices = Vec::new();
    let mut brier_scores = Vec::new();
    let mut n_at_risk_vec = Vec::new();
    let mut n_events_vec = Vec::new();

    for &lm in &landmark_times {
        let eligible: Vec<usize> = (0..n_subjects).filter(|&i| event_time[i] > lm).collect();
        let n_eligible = eligible.len();
        n_at_risk_vec.push(n_eligible);

        if n_eligible < 20 {
            all_coefs.push(vec![0.0; n_vars]);
            all_ses.push(vec![f64::INFINITY; n_vars]);
            c_indices.push(0.5);
            brier_scores.push(0.25);
            n_events_vec.push(0);
            continue;
        }

        let lm_time: Vec<f64> = eligible
            .iter()
            .map(|&i| (event_time[i] - lm).min(horizon - lm))
            .collect();

        let lm_status: Vec<i32> = eligible
            .iter()
            .map(|&i| {
                if event_time[i] <= horizon && event_status[i] == 1 {
                    1
                } else {
                    0
                }
            })
            .collect();

        let n_events_lm = lm_status.iter().filter(|&&s| s == 1).count();
        n_events_vec.push(n_events_lm);

        let lm_x: Vec<f64> = {
            let mut result = Vec::with_capacity(n_eligible * n_vars);
            for &i in &eligible {
                for j in 0..n_vars {
                    result.push(covariates[i * n_vars + j]);
                }
            }
            result
        };

        let mut beta = vec![0.0; n_vars];
        let mut info_diag = vec![0.0; n_vars];

        for _ in 0..max_iter {
            let mut gradient = vec![0.0; n_vars];
            let mut hessian_diag = vec![0.0; n_vars];

            let mut indices: Vec<usize> = (0..n_eligible).collect();
            indices.sort_by(|&a, &b| {
                lm_time[b]
                    .partial_cmp(&lm_time[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let eta: Vec<f64> = (0..n_eligible)
                .map(|i| {
                    let mut e = 0.0;
                    for j in 0..n_vars {
                        e += lm_x[i * n_vars + j] * beta[j];
                    }
                    e.clamp(-700.0, 700.0)
                })
                .collect();

            let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

            let mut risk_sum = 0.0;
            let mut weighted_x = vec![0.0; n_vars];
            let mut weighted_x_sq = vec![0.0; n_vars];

            for &i in &indices {
                risk_sum += exp_eta[i];
                for j in 0..n_vars {
                    weighted_x[j] += exp_eta[i] * lm_x[i * n_vars + j];
                    weighted_x_sq[j] += exp_eta[i] * lm_x[i * n_vars + j] * lm_x[i * n_vars + j];
                }

                if lm_status[i] == 1 && risk_sum > 0.0 {
                    for j in 0..n_vars {
                        let x_bar = weighted_x[j] / risk_sum;
                        let x_sq_bar = weighted_x_sq[j] / risk_sum;
                        gradient[j] += lm_x[i * n_vars + j] - x_bar;
                        hessian_diag[j] += x_sq_bar - x_bar * x_bar;
                    }
                }
            }

            for j in 0..n_vars {
                if hessian_diag[j].abs() > 1e-10 {
                    beta[j] += gradient[j] / hessian_diag[j];
                    beta[j] = beta[j].clamp(-10.0, 10.0);
                }
                info_diag[j] = hessian_diag[j];
            }
        }

        let std_errs: Vec<f64> = info_diag
            .iter()
            .map(|&info| {
                if info > 1e-10 {
                    (1.0 / info).sqrt()
                } else {
                    f64::INFINITY
                }
            })
            .collect();

        let c_idx = compute_concordance(&lm_time, &lm_status, &lm_x, n_eligible, n_vars, &beta);

        let risk_scores: Vec<f64> = (0..n_eligible)
            .map(|i| {
                let mut eta = 0.0;
                for j in 0..n_vars {
                    eta += lm_x[i * n_vars + j] * beta[j];
                }
                eta
            })
            .collect();

        let pred_surv: Vec<f64> = risk_scores
            .iter()
            .map(|&r| (-r.exp() * 0.1).exp())
            .collect();

        let mut brier = 0.0;
        let mut brier_n = 0.0;
        for i in 0..n_eligible {
            let outcome = if lm_status[i] == 1 { 0.0 } else { 1.0 };
            brier += (pred_surv[i] - outcome).powi(2);
            brier_n += 1.0;
        }
        let brier_score = if brier_n > 0.0 { brier / brier_n } else { 0.25 };

        all_coefs.push(beta);
        all_ses.push(std_errs);
        c_indices.push(c_idx);
        brier_scores.push(brier_score);
    }

    let mut pooled_coef = vec![0.0; n_vars];
    let mut pooled_weights = vec![0.0; n_vars];

    for lm_idx in 0..n_landmarks {
        for j in 0..n_vars {
            let se = all_ses[lm_idx][j];
            if se.is_finite() && se > 1e-10 {
                let weight = 1.0 / (se * se);
                pooled_coef[j] += weight * all_coefs[lm_idx][j];
                pooled_weights[j] += weight;
            }
        }
    }

    for j in 0..n_vars {
        if pooled_weights[j] > 0.0 {
            pooled_coef[j] /= pooled_weights[j];
        }
    }

    let pooled_se: Vec<f64> = pooled_weights
        .iter()
        .map(|&w| {
            if w > 0.0 {
                (1.0 / w).sqrt()
            } else {
                f64::INFINITY
            }
        })
        .collect();

    Ok(SuperLandmarkResult {
        landmark_times,
        coefficients: all_coefs,
        std_errors: all_ses,
        c_indices,
        brier_scores,
        n_at_risk: n_at_risk_vec,
        n_events: n_events_vec,
        pooled_coef,
        pooled_se,
    })
}
