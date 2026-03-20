
#[pyfunction]
pub fn landmarking_analysis(
    event_time: Vec<f64>,
    event_status: Vec<i32>,
    covariates: Vec<f64>,
    n_subjects: usize,
    n_vars: usize,
    landmark_times: Vec<f64>,
    horizon: f64,
) -> PyResult<Vec<(f64, Vec<f64>, f64)>> {
    let mut results = Vec::new();

    for &lm in &landmark_times {
        let eligible: Vec<usize> = (0..n_subjects).filter(|&i| event_time[i] > lm).collect();

        if eligible.len() < 10 {
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

        let lm_x: Vec<f64> = {
            let mut result = Vec::with_capacity(eligible.len() * n_vars);
            for &i in &eligible {
                for j in 0..n_vars {
                    result.push(covariates[i * n_vars + j]);
                }
            }
            result
        };

        let n_lm = eligible.len();

        let mut beta = vec![0.0; n_vars];

        for _ in 0..50 {
            let mut gradient = vec![0.0; n_vars];
            let mut hessian_diag = vec![0.0; n_vars];

            let mut indices: Vec<usize> = (0..n_lm).collect();
            indices.sort_by(|&a, &b| {
                lm_time[b]
                    .partial_cmp(&lm_time[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let eta: Vec<f64> = (0..n_lm)
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
                }
            }
        }

        let concordance = compute_concordance(&lm_time, &lm_status, &lm_x, n_lm, n_vars, &beta);

        results.push((lm, beta, concordance));
    }

    Ok(results)
}

fn compute_concordance(
    time: &[f64],
    status: &[i32],
    x: &[f64],
    n: usize,
    p: usize,
    beta: &[f64],
) -> f64 {
    let risk_scores: Vec<f64> = (0..n)
        .map(|i| {
            let mut eta = 0.0;
            for j in 0..p {
                eta += x[i * p + j] * beta[j];
            }
            eta
        })
        .collect();

    concordance_index_with_horizon(&risk_scores, time, status, None)
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct TimeVaryingAUCResult {
    #[pyo3(get)]
    pub times: Vec<f64>,
    #[pyo3(get)]
    pub auc_values: Vec<f64>,
    #[pyo3(get)]
    pub auc_lower: Vec<f64>,
    #[pyo3(get)]
    pub auc_upper: Vec<f64>,
    #[pyo3(get)]
    pub integrated_auc: f64,
    #[pyo3(get)]
    pub n_cases: Vec<usize>,
    #[pyo3(get)]
    pub n_controls: Vec<usize>,
}

#[pyfunction]
#[pyo3(signature = (risk_scores, event_time, event_status, eval_times, prediction_window, method="cumulative/dynamic"))]
pub fn time_varying_auc(
    risk_scores: Vec<f64>,
    event_time: Vec<f64>,
    event_status: Vec<i32>,
    eval_times: Vec<f64>,
    prediction_window: f64,
    method: &str,
) -> PyResult<TimeVaryingAUCResult> {
    let n = risk_scores.len();
    if event_time.len() != n || event_status.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All input vectors must have the same length",
        ));
    }

    let mut auc_values = Vec::new();
    let mut auc_lower = Vec::new();
    let mut auc_upper = Vec::new();
    let mut n_cases_vec = Vec::new();
    let mut n_controls_vec = Vec::new();

    for &t in &eval_times {
        let (cases, controls): (Vec<usize>, Vec<usize>) = match method {
            "incident/dynamic" => {
                let cases: Vec<usize> = (0..n)
                    .filter(|&i| {
                        event_time[i] > t
                            && event_time[i] <= t + prediction_window
                            && event_status[i] == 1
                    })
                    .collect();
                let controls: Vec<usize> = (0..n)
                    .filter(|&i| event_time[i] > t + prediction_window)
                    .collect();
                (cases, controls)
            }
            _ => {
                let cases: Vec<usize> = (0..n)
                    .filter(|&i| event_time[i] <= t + prediction_window && event_status[i] == 1)
                    .collect();
                let controls: Vec<usize> = (0..n)
                    .filter(|&i| event_time[i] > t + prediction_window)
                    .collect();
                (cases, controls)
            }
        };

        n_cases_vec.push(cases.len());
        n_controls_vec.push(controls.len());

        if cases.is_empty() || controls.is_empty() {
            auc_values.push(0.5);
            auc_lower.push(0.0);
            auc_upper.push(1.0);
            continue;
        }

        let mut concordant = 0.0;
        let total = cases.len() as f64 * controls.len() as f64;

        for &case_idx in &cases {
            for &control_idx in &controls {
                if risk_scores[case_idx] > risk_scores[control_idx] {
                    concordant += 1.0;
                } else if (risk_scores[case_idx] - risk_scores[control_idx]).abs() < 1e-10 {
                    concordant += 0.5;
                }
            }
        }

        let auc = concordant / total;
        auc_values.push(auc);

        let se = (auc * (1.0 - auc) / cases.len().min(controls.len()) as f64).sqrt();
        auc_lower.push((auc - 1.96 * se).max(0.0));
        auc_upper.push((auc + 1.96 * se).min(1.0));
    }

    let integrated_auc = if eval_times.len() > 1 {
        let mut integral = 0.0;
        let mut total_weight = 0.0;
        for i in 1..eval_times.len() {
            let dt = eval_times[i] - eval_times[i - 1];
            let weight = (n_cases_vec[i] + n_cases_vec[i - 1]) as f64 / 2.0;
            integral += (auc_values[i] + auc_values[i - 1]) / 2.0 * dt * weight;
            total_weight += dt * weight;
        }
        if total_weight > 0.0 {
            integral / total_weight
        } else {
            auc_values.iter().sum::<f64>() / auc_values.len() as f64
        }
    } else if !auc_values.is_empty() {
        auc_values[0]
    } else {
        0.5
    };

    Ok(TimeVaryingAUCResult {
        times: eval_times,
        auc_values,
        auc_lower,
        auc_upper,
        integrated_auc,
        n_cases: n_cases_vec,
        n_controls: n_controls_vec,
    })
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct DynamicCIndexResult {
    #[pyo3(get)]
    pub c_index: f64,
    #[pyo3(get)]
    pub se: f64,
    #[pyo3(get)]
    pub lower: f64,
    #[pyo3(get)]
    pub upper: f64,
    #[pyo3(get)]
    pub n_concordant: usize,
    #[pyo3(get)]
    pub n_discordant: usize,
    #[pyo3(get)]
    pub n_tied: usize,
    #[pyo3(get)]
    pub n_pairs: usize,
    #[pyo3(get)]
    pub time_dependent_c: Vec<f64>,
    #[pyo3(get)]
    pub eval_times: Vec<f64>,
}

#[pyfunction]
#[pyo3(signature = (risk_scores, event_time, event_status, landmark_time, horizon, eval_times=None))]
pub fn dynamic_c_index(
    risk_scores: Vec<f64>,
    event_time: Vec<f64>,
    event_status: Vec<i32>,
    landmark_time: f64,
    horizon: f64,
    eval_times: Option<Vec<f64>>,
) -> PyResult<DynamicCIndexResult> {
    let n = risk_scores.len();
    if event_time.len() != n || event_status.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All input vectors must have the same length",
        ));
    }

    let eligible: Vec<usize> = (0..n).filter(|&i| event_time[i] > landmark_time).collect();

    let mut concordant = 0usize;
    let mut discordant = 0usize;
    let mut tied = 0usize;
    let mut total_pairs = 0usize;

    for (idx_i, &i) in eligible.iter().enumerate() {
        for &j in eligible.iter().skip(idx_i + 1) {
            let ti = event_time[i].min(horizon);
            let tj = event_time[j].min(horizon);
            let si = event_status[i];
            let sj = event_status[j];

            let i_event = si == 1 && event_time[i] <= horizon;
            let j_event = sj == 1 && event_time[j] <= horizon;

            if i_event && ti < tj {
                total_pairs += 1;
                if risk_scores[i] > risk_scores[j] {
                    concordant += 1;
                } else if risk_scores[i] < risk_scores[j] {
                    discordant += 1;
                } else {
                    tied += 1;
                }
            } else if j_event && tj < ti {
                total_pairs += 1;
                if risk_scores[j] > risk_scores[i] {
                    concordant += 1;
                } else if risk_scores[j] < risk_scores[i] {
                    discordant += 1;
                } else {
                    tied += 1;
                }
            }
        }
    }

    let c_index = if total_pairs > 0 {
        (concordant as f64 + 0.5 * tied as f64) / total_pairs as f64
    } else {
        0.5
    };

    let se = if total_pairs > 10 {
        (c_index * (1.0 - c_index) / total_pairs as f64).sqrt()
    } else {
        0.0
    };

    let lower = (c_index - 1.96 * se).max(0.0);
    let upper = (c_index + 1.96 * se).min(1.0);

    let default_times: Vec<f64> = {
        let min_t = event_time
            .iter()
            .filter(|&&t| t > landmark_time)
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max_t = horizon;
        (0..10)
            .map(|i| min_t + (max_t - min_t) * i as f64 / 9.0)
            .collect()
    };
    let times = eval_times.unwrap_or(default_times);

    let time_dependent_c: Vec<f64> = times
        .iter()
        .map(|&t| {
            let mut conc = 0.0;
            let mut pairs = 0.0;

            for (idx_i, &i) in eligible.iter().enumerate() {
                for &j in eligible.iter().skip(idx_i + 1) {
                    let ti = event_time[i];
                    let tj = event_time[j];
                    let si = event_status[i];

                    if si == 1 && ti <= t && tj > ti {
                        pairs += 1.0;
                        if risk_scores[i] > risk_scores[j] {
                            conc += 1.0;
                        } else if (risk_scores[i] - risk_scores[j]).abs() < 1e-10 {
                            conc += 0.5;
                        }
                    }

                    let sj = event_status[j];
                    if sj == 1 && tj <= t && ti > tj {
                        pairs += 1.0;
                        if risk_scores[j] > risk_scores[i] {
                            conc += 1.0;
                        } else if (risk_scores[i] - risk_scores[j]).abs() < 1e-10 {
                            conc += 0.5;
                        }
                    }
                }
            }

            if pairs > 0.0 { conc / pairs } else { 0.5 }
        })
        .collect();

    Ok(DynamicCIndexResult {
        c_index,
        se,
        lower,
        upper,
        n_concordant: concordant,
        n_discordant: discordant,
        n_tied: tied,
        n_pairs: total_pairs,
        time_dependent_c,
        eval_times: times,
    })
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct IPCWAUCResult {
    #[pyo3(get)]
    pub times: Vec<f64>,
    #[pyo3(get)]
    pub auc_values: Vec<f64>,
    #[pyo3(get)]
    pub auc_se: Vec<f64>,
    #[pyo3(get)]
    pub integrated_auc: f64,
    #[pyo3(get)]
    pub ipcw_weights: Vec<f64>,
}

#[pyfunction]
#[pyo3(signature = (risk_scores, event_time, event_status, eval_times))]
pub fn ipcw_auc(
    risk_scores: Vec<f64>,
    event_time: Vec<f64>,
    event_status: Vec<i32>,
    eval_times: Vec<f64>,
) -> PyResult<IPCWAUCResult> {
    let n = risk_scores.len();
    if event_time.len() != n || event_status.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All input vectors must have the same length",
        ));
    }

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| {
        event_time[a]
            .partial_cmp(&event_time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut km_surv = vec![1.0; n];
    let mut at_risk = n as f64;
    let mut prev_surv = 1.0;

    for &i in sorted_indices.iter() {
        if event_status[i] == 0 {
            let d = 1.0;
            prev_surv *= 1.0 - d / at_risk;
        }
        km_surv[i] = prev_surv.max(0.01);
        at_risk -= 1.0;
    }

    let ipcw_weights: Vec<f64> = (0..n).map(|i| 1.0 / km_surv[i]).collect();

    let max_weight = ipcw_weights
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let ipcw_weights: Vec<f64> = ipcw_weights.iter().map(|&w| w / max_weight).collect();

    let mut auc_values = Vec::new();
    let mut auc_se = Vec::new();

    for &t in &eval_times {
        let cases: Vec<usize> = (0..n)
            .filter(|&i| event_time[i] <= t && event_status[i] == 1)
            .collect();

        let controls: Vec<usize> = (0..n).filter(|&i| event_time[i] > t).collect();

        if cases.is_empty() || controls.is_empty() {
            auc_values.push(0.5);
            auc_se.push(0.0);
            continue;
        }

        let mut weighted_concordant = 0.0;
        let mut total_weight = 0.0;

        for &case_idx in &cases {
            for &control_idx in &controls {
                let weight = ipcw_weights[case_idx] * ipcw_weights[control_idx];
                total_weight += weight;

                if risk_scores[case_idx] > risk_scores[control_idx] {
                    weighted_concordant += weight;
                } else if (risk_scores[case_idx] - risk_scores[control_idx]).abs() < 1e-10 {
                    weighted_concordant += 0.5 * weight;
                }
            }
        }

        let auc = if total_weight > 0.0 {
            weighted_concordant / total_weight
        } else {
            0.5
        };

        auc_values.push(auc);

        let se = (auc * (1.0 - auc) / (cases.len().min(controls.len()) as f64).max(1.0)).sqrt();
        auc_se.push(se);
    }

    let integrated_auc = if !auc_values.is_empty() {
        auc_values.iter().sum::<f64>() / auc_values.len() as f64
    } else {
        0.5
    };

    Ok(IPCWAUCResult {
        times: eval_times,
        auc_values,
        auc_se,
        integrated_auc,
        ipcw_weights,
    })
}
