

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]

pub struct DfbetaResult {
    #[pyo3(get)]
    pub dfbeta: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub dfbetas: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub max_dfbeta: Vec<f64>,
    #[pyo3(get)]
    pub influential_obs: Vec<usize>,
    #[pyo3(get)]
    pub n_obs: usize,
    #[pyo3(get)]
    pub n_vars: usize,
}

#[pyfunction]
#[pyo3(signature = (time, event, covariates, n_covariates, coefficients, threshold=None))]
pub fn dfbeta_cox(
    time: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<f64>,
    n_covariates: usize,
    coefficients: Vec<f64>,
    threshold: Option<f64>,
) -> PyResult<DfbetaResult> {
    let n = time.len();
    if event.len() != n || covariates.len() != n * n_covariates {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input dimensions mismatch",
        ));
    }
    if coefficients.len() != n_covariates {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "coefficients must have length n_covariates",
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

    let score_residuals = compute_score_residuals(
        &time,
        &event,
        &covariates,
        n_covariates,
        &exp_eta,
        &sorted_indices,
    );

    let dfbeta: Vec<Vec<f64>> = if n > PARALLEL_THRESHOLD_MEDIUM {
        (0..n)
            .into_par_iter()
            .map(|i| {
                let mut row = vec![0.0; n_covariates];
                for j in 0..n_covariates {
                    let mut acc = 0.0;
                    for k in 0..n_covariates {
                        acc += info_inv[j * n_covariates + k] * score_residuals[i][k];
                    }
                    row[j] = acc;
                }
                row
            })
            .collect()
    } else {
        let mut rows = vec![vec![0.0; n_covariates]; n];
        for i in 0..n {
            for j in 0..n_covariates {
                for k in 0..n_covariates {
                    rows[i][j] += info_inv[j * n_covariates + k] * score_residuals[i][k];
                }
            }
        }
        rows
    };

    let coef_se: Vec<f64> = (0..n_covariates)
        .map(|j| info_inv[j * n_covariates + j].sqrt().max(1e-10))
        .collect();

    let dfbetas: Vec<Vec<f64>> = if n > PARALLEL_THRESHOLD_MEDIUM {
        dfbeta
            .par_iter()
            .map(|row| {
                (0..n_covariates)
                    .map(|j| row[j] / coef_se[j])
                    .collect::<Vec<f64>>()
            })
            .collect()
    } else {
        let mut rows = vec![vec![0.0; n_covariates]; n];
        for i in 0..n {
            for j in 0..n_covariates {
                rows[i][j] = dfbeta[i][j] / coef_se[j];
            }
        }
        rows
    };

    let max_dfbeta: Vec<f64> = if n_covariates > PARALLEL_THRESHOLD_MEDIUM {
        (0..n_covariates)
            .into_par_iter()
            .map(|j| {
                dfbeta
                    .iter()
                    .map(|row| row[j].abs())
                    .fold(0.0_f64, f64::max)
            })
            .collect()
    } else {
        (0..n_covariates)
            .map(|j| {
                dfbeta
                    .iter()
                    .map(|row| row[j].abs())
                    .fold(0.0_f64, f64::max)
            })
            .collect()
    };

    let thresh = threshold.unwrap_or(2.0 / (n as f64).sqrt());
    let influential_obs: Vec<usize> = if n > PARALLEL_THRESHOLD_MEDIUM {
        (0..n)
            .into_par_iter()
            .filter(|&i| dfbetas[i].iter().any(|&d| d.abs() > thresh))
            .collect()
    } else {
        (0..n)
            .filter(|&i| dfbetas[i].iter().any(|&d| d.abs() > thresh))
            .collect()
    };

    Ok(DfbetaResult {
        dfbeta,
        dfbetas,
        max_dfbeta,
        influential_obs,
        n_obs: n,
        n_vars: n_covariates,
    })
}

fn compute_information_inverse(
    event: &[i32],
    covariates: &[f64],
    n_covariates: usize,
    exp_eta: &[f64],
    sorted_indices: &[usize],
) -> Vec<f64> {
    let mut info = vec![0.0; n_covariates * n_covariates];

    let mut risk_sum = 0.0;
    let mut weighted_x = vec![0.0; n_covariates];
    let mut weighted_xx = vec![0.0; n_covariates * n_covariates];

    for &i in sorted_indices {
        risk_sum += exp_eta[i];
        for j in 0..n_covariates {
            weighted_x[j] += exp_eta[i] * covariates[i * n_covariates + j];
            for k in 0..n_covariates {
                weighted_xx[j * n_covariates + k] += exp_eta[i]
                    * covariates[i * n_covariates + j]
                    * covariates[i * n_covariates + k];
            }
        }

        if event[i] == 1 && risk_sum > 0.0 {
            for j in 0..n_covariates {
                let x_bar_j = weighted_x[j] / risk_sum;
                for k in 0..n_covariates {
                    let x_bar_k = weighted_x[k] / risk_sum;
                    let xx_bar = weighted_xx[j * n_covariates + k] / risk_sum;
                    info[j * n_covariates + k] += xx_bar - x_bar_j * x_bar_k;
                }
            }
        }
    }

    invert_matrix(&info, n_covariates)
}

fn compute_score_residuals(
    time: &[f64],
    event: &[i32],
    covariates: &[f64],
    n_covariates: usize,
    exp_eta: &[f64],
    sorted_indices: &[usize],
) -> Vec<Vec<f64>> {
    let n = time.len();
    let mut risk_sum = 0.0;
    let mut weighted_x = vec![0.0; n_covariates];
    let mut cumulative_term = vec![vec![0.0; n_covariates]; n];

    for &i in sorted_indices.iter() {
        risk_sum += exp_eta[i];
        for j in 0..n_covariates {
            weighted_x[j] += exp_eta[i] * covariates[i * n_covariates + j];
        }

        if event[i] == 1 && risk_sum > 0.0 {
            let x_bar: Vec<f64> = weighted_x.iter().map(|&w| w / risk_sum).collect();
            let time_i = time[i];
            if n > PARALLEL_THRESHOLD_MEDIUM {
                cumulative_term
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(k, row)| {
                        if time[k] >= time_i {
                            let scale = exp_eta[k] / risk_sum;
                            let base = k * n_covariates;
                            for j in 0..n_covariates {
                                row[j] += scale * (covariates[base + j] - x_bar[j]);
                            }
                        }
                    });
            } else {
                for k in 0..n {
                    if time[k] >= time_i {
                        let scale = exp_eta[k] / risk_sum;
                        let base = k * n_covariates;
                        for j in 0..n_covariates {
                            cumulative_term[k][j] += scale * (covariates[base + j] - x_bar[j]);
                        }
                    }
                }
            }
        }
    }

    if n > PARALLEL_THRESHOLD_MEDIUM {
        (0..n)
            .into_par_iter()
            .map(|i| {
                let mut row = vec![0.0; n_covariates];
                let base = i * n_covariates;
                for j in 0..n_covariates {
                    let event_term = if event[i] == 1 {
                        covariates[base + j]
                    } else {
                        0.0
                    };
                    row[j] = event_term - cumulative_term[i][j];
                }
                row
            })
            .collect()
    } else {
        let mut score_resid: Vec<Vec<f64>> = vec![vec![0.0; n_covariates]; n];
        for i in 0..n {
            for j in 0..n_covariates {
                if event[i] == 1 {
                    score_resid[i][j] = covariates[i * n_covariates + j];
                }
                score_resid[i][j] -= cumulative_term[i][j];
            }
        }
        score_resid
    }
}

fn invert_matrix(a: &[f64], n: usize) -> Vec<f64> {
    invert_flat_square_matrix_with_fallback(a, n)
}
