

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]

pub struct GroupLassoConfig {
    #[pyo3(get, set)]
    pub lambda: f64,
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
    #[pyo3(get, set)]
    pub standardize: bool,
    #[pyo3(get, set)]
    pub group_weights: Option<Vec<f64>>,
}

#[pymethods]
impl GroupLassoConfig {
    #[new]
    #[pyo3(signature = (lambda=1.0, max_iter=1000, tol=1e-6, standardize=true, group_weights=None))]
    pub fn new(
        lambda: f64,
        max_iter: usize,
        tol: f64,
        standardize: bool,
        group_weights: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        if lambda < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "lambda must be non-negative",
            ));
        }
        Ok(GroupLassoConfig {
            lambda,
            max_iter,
            tol,
            standardize,
            group_weights,
        })
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct GroupLassoResult {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    #[pyo3(get)]
    pub selected_groups: Vec<usize>,
    #[pyo3(get)]
    pub group_norms: Vec<f64>,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub n_iter: usize,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub lambda: f64,
    #[pyo3(get)]
    pub n_groups: usize,
    #[pyo3(get)]
    pub df: usize,
}

fn soft_threshold(x: f64, lambda: f64) -> f64 {
    if x > lambda {
        x - lambda
    } else if x < -lambda {
        x + lambda
    } else {
        0.0
    }
}

fn group_soft_threshold(beta_group: &[f64], lambda: f64) -> Vec<f64> {
    let norm: f64 = beta_group.iter().map(|&b| b * b).sum::<f64>().sqrt();
    if norm <= lambda {
        vec![0.0; beta_group.len()]
    } else {
        let scale = 1.0 - lambda / norm;
        beta_group.iter().map(|&b| b * scale).collect()
    }
}

#[pyfunction]
#[pyo3(signature = (time, event, covariates, groups, config))]
pub fn group_lasso_cox(
    time: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<f64>,
    groups: Vec<usize>,
    config: &GroupLassoConfig,
) -> PyResult<GroupLassoResult> {
    let n = time.len();
    if event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and event must have same length",
        ));
    }

    let p = if covariates.is_empty() {
        0
    } else {
        covariates.len() / n
    };

    if p == 0 || groups.len() != p {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "groups must have length equal to number of covariates",
        ));
    }

    let max_group = *groups.iter().max().unwrap_or(&0);
    let n_groups = max_group + 1;

    let group_indices: Vec<Vec<usize>> = (0..n_groups)
        .map(|g| {
            groups
                .iter()
                .enumerate()
                .filter(|(_, gr)| **gr == g)
                .map(|(i, _)| i)
                .collect()
        })
        .collect();

    let group_weights: Vec<f64> = config.group_weights.clone().unwrap_or_else(|| {
        group_indices
            .iter()
            .map(|g| (g.len() as f64).sqrt())
            .collect()
    });

    let x_means: Vec<f64> = (0..p)
        .map(|j| (0..n).map(|i| covariates[i * p + j]).sum::<f64>() / n as f64)
        .collect();

    let x_stds: Vec<f64> = (0..p)
        .map(|j| {
            let mean = x_means[j];
            let var: f64 = (0..n)
                .map(|i| (covariates[i * p + j] - mean).powi(2))
                .sum::<f64>()
                / n as f64;
            var.sqrt().max(crate::constants::DIVISION_FLOOR)
        })
        .collect();

    let x_std: Vec<f64> = if config.standardize {
        (0..n * p)
            .map(|idx| {
                let j = idx % p;
                (covariates[idx] - x_means[j]) / x_stds[j]
            })
            .collect()
    } else {
        covariates.clone()
    };

    let mut beta = vec![0.0; p];
    let mut converged = false;
    let mut n_iter = 0;

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| {
        time[b]
            .partial_cmp(&time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for iter in 0..config.max_iter {
        n_iter = iter + 1;
        let beta_old = beta.clone();

        for g in 0..n_groups {
            let indices = &group_indices[g];
            if indices.is_empty() {
                continue;
            }

            let mut gradient = vec![0.0; indices.len()];
            let mut hessian_diag = vec![0.0; indices.len()];

            let eta: Vec<f64> = (0..n)
                .map(|i| {
                    let mut e = 0.0;
                    for j in 0..p {
                        e += x_std[i * p + j] * beta[j];
                    }
                    e.clamp(-700.0, 700.0)
                })
                .collect();

            let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

            let mut risk_sum = 0.0;
            let mut weighted_x: Vec<f64> = vec![0.0; indices.len()];
            let mut weighted_xx: Vec<f64> = vec![0.0; indices.len()];

            for &i in &sorted_indices {
                risk_sum += exp_eta[i];
                for (k, &j) in indices.iter().enumerate() {
                    weighted_x[k] += exp_eta[i] * x_std[i * p + j];
                    weighted_xx[k] += exp_eta[i] * x_std[i * p + j] * x_std[i * p + j];
                }

                if event[i] == 1 && risk_sum > 0.0 {
                    for (k, &j) in indices.iter().enumerate() {
                        let x_bar = weighted_x[k] / risk_sum;
                        let x_sq_bar = weighted_xx[k] / risk_sum;
                        gradient[k] += x_std[i * p + j] - x_bar;
                        hessian_diag[k] += x_sq_bar - x_bar * x_bar;
                    }
                }
            }

            let z: Vec<f64> = indices
                .iter()
                .enumerate()
                .map(|(k, &j)| {
                    let h = hessian_diag[k].max(crate::constants::DIVISION_FLOOR);
                    beta[j] + gradient[k] / h
                })
                .collect();

            let avg_hessian: f64 =
                hessian_diag.iter().sum::<f64>() / hessian_diag.len().max(1) as f64;
            let group_lambda = config.lambda * group_weights[g]
                / avg_hessian.max(crate::constants::DIVISION_FLOOR);

            let new_beta_group = group_soft_threshold(&z, group_lambda);

            for (k, &j) in indices.iter().enumerate() {
                beta[j] = new_beta_group[k].clamp(-10.0, 10.0);
            }
        }

        let max_change: f64 = beta
            .iter()
            .zip(beta_old.iter())
            .map(|(&b, &bo)| (b - bo).abs())
            .fold(0.0, f64::max);

        if max_change < config.tol {
            converged = true;
            break;
        }
    }

    if config.standardize {
        for j in 0..p {
            beta[j] /= x_stds[j];
        }
    }

    let group_norms: Vec<f64> = (0..n_groups)
        .map(|g| {
            group_indices[g]
                .iter()
                .map(|&j| beta[j] * beta[j])
                .sum::<f64>()
                .sqrt()
        })
        .collect();

    let selected_groups: Vec<usize> = (0..n_groups)
        .filter(|&g| group_norms[g] > crate::constants::DIVISION_FLOOR)
        .collect();

    let df: usize = beta
        .iter()
        .filter(|&&b| b.abs() > crate::constants::DIVISION_FLOOR)
        .count();

    let mut log_likelihood = 0.0;
    let eta: Vec<f64> = (0..n)
        .map(|i| {
            let mut e = 0.0;
            for j in 0..p {
                e += covariates[i * p + j] * beta[j];
            }
            e
        })
        .collect();

    let mut risk_sum = 0.0;
    for &i in &sorted_indices {
        risk_sum += eta[i].exp();
        if event[i] == 1 && risk_sum > 0.0 {
            log_likelihood += eta[i] - risk_sum.ln();
        }
    }

    Ok(GroupLassoResult {
        coefficients: beta,
        selected_groups,
        group_norms,
        log_likelihood,
        n_iter,
        converged,
        lambda: config.lambda,
        n_groups,
        df,
    })
}
