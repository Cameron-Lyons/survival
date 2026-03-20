
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
