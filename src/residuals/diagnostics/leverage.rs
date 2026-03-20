
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct LeverageResult {
    #[pyo3(get)]
    pub leverage: Vec<f64>,
    #[pyo3(get)]
    pub lmax: Vec<f64>,
    #[pyo3(get)]
    pub mean_leverage: f64,
    #[pyo3(get)]
    pub high_leverage_obs: Vec<usize>,
    #[pyo3(get)]
    pub n_obs: usize,
}

#[pyfunction]
#[pyo3(signature = (time, event, covariates, n_covariates, coefficients, threshold_multiplier=2.0))]
pub fn leverage_cox(
    time: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<f64>,
    n_covariates: usize,
    coefficients: Vec<f64>,
    threshold_multiplier: f64,
) -> PyResult<LeverageResult> {
    let n = time.len();
    if event.len() != n || covariates.len() != n * n_covariates {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input dimensions mismatch",
        ));
    }

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    if n > PARALLEL_THRESHOLD_MEDIUM {
        sorted_indices.par_sort_by(|&a, &b| {
            time[b]
                .partial_cmp(&time[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        sorted_indices.sort_by(|&a, &b| {
            time[b]
                .partial_cmp(&time[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    let eta: Vec<f64> = (0..n)
        .map(|i| {
            let mut e = 0.0;
            for j in 0..n_covariates {
                e += covariates[i * n_covariates + j] * coefficients[j];
            }
            e.clamp(-500.0, 500.0)
        })
        .collect();

    let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

    let info_inv =
        compute_information_inverse(&event, &covariates, n_covariates, &exp_eta, &sorted_indices);

    let mut leverage = vec![0.0; n];
    let mut lmax = vec![0.0; n];

    let mut risk_sum = 0.0;
    let mut weighted_x = vec![0.0; n_covariates];

    for &i in &sorted_indices {
        risk_sum += exp_eta[i];
        for j in 0..n_covariates {
            weighted_x[j] += exp_eta[i] * covariates[i * n_covariates + j];
        }

        if risk_sum > 0.0 {
            let x_bar: Vec<f64> = weighted_x.iter().map(|&w| w / risk_sum).collect();
            let time_i = time[i];
            let event_i = event[i];
            if n > PARALLEL_THRESHOLD_MEDIUM {
                leverage.par_iter_mut().enumerate().for_each(|(k, lev)| {
                    if time[k] >= time_i {
                        let mut h_ik = 0.0;
                        let base_k = k * n_covariates;
                        for j1 in 0..n_covariates {
                            let x_diff1 = covariates[base_k + j1] - x_bar[j1];
                            for j2 in 0..n_covariates {
                                let x_diff2 = covariates[base_k + j2] - x_bar[j2];
                                h_ik += x_diff1 * info_inv[j1 * n_covariates + j2] * x_diff2;
                            }
                        }
                        h_ik *= exp_eta[k] / risk_sum;
                        if event_i == 1 {
                            *lev += h_ik;
                        }
                    }
                });
            } else {
                for k in 0..n {
                    if time[k] >= time_i {
                        let mut h_ik = 0.0;
                        let base_k = k * n_covariates;
                        for j1 in 0..n_covariates {
                            let x_diff1 = covariates[base_k + j1] - x_bar[j1];
                            for j2 in 0..n_covariates {
                                let x_diff2 = covariates[base_k + j2] - x_bar[j2];
                                h_ik += x_diff1 * info_inv[j1 * n_covariates + j2] * x_diff2;
                            }
                        }
                        h_ik *= exp_eta[k] / risk_sum;
                        if event_i == 1 {
                            leverage[k] += h_ik;
                        }
                    }
                }
            }
        }
    }

    lmax = if n > PARALLEL_THRESHOLD_MEDIUM {
        (0..n)
            .into_par_iter()
            .map(|i| {
                let mut max_contrib: f64 = 0.0;
                for j in 0..n_covariates {
                    let x_j = covariates[i * n_covariates + j];
                    let contrib = x_j.abs() * info_inv[j * n_covariates + j].sqrt();
                    max_contrib = max_contrib.max(contrib);
                }
                max_contrib
            })
            .collect()
    } else {
        for i in 0..n {
            let mut max_contrib: f64 = 0.0;
            for j in 0..n_covariates {
                let x_j = covariates[i * n_covariates + j];
                let contrib = x_j.abs() * info_inv[j * n_covariates + j].sqrt();
                max_contrib = max_contrib.max(contrib);
            }
            lmax[i] = max_contrib;
        }
        lmax
    };

    let mean_leverage = if n > PARALLEL_THRESHOLD_MEDIUM {
        leverage.par_iter().sum::<f64>() / n as f64
    } else {
        leverage.iter().sum::<f64>() / n as f64
    };
    let threshold = threshold_multiplier * (n_covariates as f64) / (n as f64);

    let high_leverage_obs: Vec<usize> = if n > PARALLEL_THRESHOLD_MEDIUM {
        (0..n)
            .into_par_iter()
            .filter(|&i| leverage[i] > threshold)
            .collect()
    } else {
        (0..n).filter(|&i| leverage[i] > threshold).collect()
    };

    Ok(LeverageResult {
        leverage,
        lmax,
        mean_leverage,
        high_leverage_obs,
        n_obs: n,
    })
}
