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

#[derive(Debug, Clone)]
#[pyclass]
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
#[pyclass]
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
            var.sqrt().max(1e-10)
        })
        .collect();

    let x_std: Vec<f64> = if config.standardize {
        (0..n * p)
            .map(|idx| {
                let j = idx % p;
                let i = idx / p;
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
    sorted_indices.sort_by(|&a, &b| time[b].partial_cmp(&time[a]).unwrap());

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

            let mut z: Vec<f64> = indices
                .iter()
                .enumerate()
                .map(|(k, &j)| {
                    let h = hessian_diag[k].max(1e-10);
                    beta[j] + gradient[k] / h
                })
                .collect();

            let avg_hessian: f64 =
                hessian_diag.iter().sum::<f64>() / hessian_diag.len().max(1) as f64;
            let group_lambda = config.lambda * group_weights[g] / avg_hessian.max(1e-10);

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

    let selected_groups: Vec<usize> = (0..n_groups).filter(|&g| group_norms[g] > 1e-10).collect();

    let df: usize = beta.iter().filter(|&&b| b.abs() > 1e-10).count();

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

#[derive(Debug, Clone)]
#[pyclass]
pub struct SparseBoostingConfig {
    #[pyo3(get, set)]
    pub n_iterations: usize,
    #[pyo3(get, set)]
    pub learning_rate: f64,
    #[pyo3(get, set)]
    pub subsample_ratio: f64,
    #[pyo3(get, set)]
    pub early_stopping_rounds: usize,
    #[pyo3(get, set)]
    pub l1_penalty: f64,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl SparseBoostingConfig {
    #[new]
    #[pyo3(signature = (n_iterations=100, learning_rate=0.1, subsample_ratio=0.8, early_stopping_rounds=10, l1_penalty=0.0, seed=None))]
    pub fn new(
        n_iterations: usize,
        learning_rate: f64,
        subsample_ratio: f64,
        early_stopping_rounds: usize,
        l1_penalty: f64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if learning_rate <= 0.0 || learning_rate > 1.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "learning_rate must be in (0, 1]",
            ));
        }
        if subsample_ratio <= 0.0 || subsample_ratio > 1.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "subsample_ratio must be in (0, 1]",
            ));
        }
        Ok(SparseBoostingConfig {
            n_iterations,
            learning_rate,
            subsample_ratio,
            early_stopping_rounds,
            l1_penalty,
            seed,
        })
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct SparseBoostingResult {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    #[pyo3(get)]
    pub selected_features: Vec<usize>,
    #[pyo3(get)]
    pub feature_importance: Vec<f64>,
    #[pyo3(get)]
    pub iteration_scores: Vec<f64>,
    #[pyo3(get)]
    pub best_iteration: usize,
    #[pyo3(get)]
    pub n_selected: usize,
}

#[pyfunction]
#[pyo3(signature = (time, event, covariates, config))]
pub fn sparse_boosting_cox(
    time: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<f64>,
    config: &SparseBoostingConfig,
) -> PyResult<SparseBoostingResult> {
    let n = time.len();
    if event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and event must have same length",
        ));
    }

    let p = if covariates.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "covariates cannot be empty",
        ));
    } else {
        covariates.len() / n
    };

    let mut rng = fastrand::Rng::new();
    if let Some(seed) = config.seed {
        rng.seed(seed);
    }

    let mut beta = vec![0.0; p];
    let mut feature_selection_count = vec![0usize; p];
    let mut iteration_scores = Vec::new();
    let mut best_score = f64::NEG_INFINITY;
    let mut best_iteration = 0;
    let mut no_improvement_count = 0;

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| time[b].partial_cmp(&time[a]).unwrap());

    for iter in 0..config.n_iterations {
        let sample_size = (n as f64 * config.subsample_ratio) as usize;
        let mut sample_indices: Vec<usize> = (0..n).collect();
        for i in 0..sample_size.min(n) {
            let j = i + rng.usize(0..(n - i));
            sample_indices.swap(i, j);
        }
        sample_indices.truncate(sample_size);

        let eta: Vec<f64> = (0..n)
            .map(|i| {
                let mut e = 0.0;
                for j in 0..p {
                    e += covariates[i * p + j] * beta[j];
                }
                e.clamp(-700.0, 700.0)
            })
            .collect();

        let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

        let gradients: Vec<f64> = (0..p)
            .map(|j| {
                let mut gradient = 0.0;
                let mut risk_sum = 0.0;
                let mut weighted_x = 0.0;

                for &i in &sorted_indices {
                    if !sample_indices.contains(&i) {
                        continue;
                    }
                    risk_sum += exp_eta[i];
                    weighted_x += exp_eta[i] * covariates[i * p + j];

                    if event[i] == 1 && risk_sum > 0.0 {
                        let x_bar = weighted_x / risk_sum;
                        gradient += covariates[i * p + j] - x_bar;
                    }
                }
                gradient
            })
            .collect();

        let (best_j, best_grad) = gradients
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(j, &g)| (j, g))
            .unwrap();

        let update = config.learning_rate
            * best_grad.signum()
            * (best_grad.abs() - config.l1_penalty).max(0.0);

        if update.abs() > 1e-10 {
            beta[best_j] += update;
            beta[best_j] = beta[best_j].clamp(-10.0, 10.0);
            feature_selection_count[best_j] += 1;
        }

        let score = compute_partial_likelihood(&time, &event, &covariates, &beta, n, p);
        iteration_scores.push(score);

        if score > best_score {
            best_score = score;
            best_iteration = iter;
            no_improvement_count = 0;
        } else {
            no_improvement_count += 1;
        }

        if no_improvement_count >= config.early_stopping_rounds {
            break;
        }
    }

    let selected_features: Vec<usize> = (0..p).filter(|&j| beta[j].abs() > 1e-10).collect();

    let total_selections: usize = feature_selection_count.iter().sum();
    let feature_importance: Vec<f64> = feature_selection_count
        .iter()
        .map(|&c| c as f64 / total_selections.max(1) as f64)
        .collect();

    Ok(SparseBoostingResult {
        coefficients: beta,
        selected_features: selected_features.clone(),
        feature_importance,
        iteration_scores,
        best_iteration,
        n_selected: selected_features.len(),
    })
}

fn compute_partial_likelihood(
    time: &[f64],
    event: &[i32],
    covariates: &[f64],
    beta: &[f64],
    n: usize,
    p: usize,
) -> f64 {
    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| time[b].partial_cmp(&time[a]).unwrap());

    let eta: Vec<f64> = (0..n)
        .map(|i| {
            let mut e = 0.0;
            for j in 0..p {
                e += covariates[i * p + j] * beta[j];
            }
            e.clamp(-700.0, 700.0)
        })
        .collect();

    let mut log_lik = 0.0;
    let mut risk_sum = 0.0;

    for &i in &sorted_indices {
        risk_sum += eta[i].exp();
        if event[i] == 1 && risk_sum > 0.0 {
            log_lik += eta[i] - risk_sum.ln();
        }
    }

    log_lik
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct SISConfig {
    #[pyo3(get, set)]
    pub n_select: usize,
    #[pyo3(get, set)]
    pub iterative: bool,
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub threshold: f64,
}

#[pymethods]
impl SISConfig {
    #[new]
    #[pyo3(signature = (n_select=None, iterative=false, max_iter=5, threshold=0.0))]
    pub fn new(
        n_select: Option<usize>,
        iterative: bool,
        max_iter: usize,
        threshold: f64,
    ) -> PyResult<Self> {
        Ok(SISConfig {
            n_select: n_select.unwrap_or(0),
            iterative,
            max_iter,
            threshold,
        })
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct SISResult {
    #[pyo3(get)]
    pub selected_features: Vec<usize>,
    #[pyo3(get)]
    pub marginal_scores: Vec<f64>,
    #[pyo3(get)]
    pub ranking: Vec<usize>,
    #[pyo3(get)]
    pub n_selected: usize,
    #[pyo3(get)]
    pub iteration_selections: Vec<Vec<usize>>,
}

#[pyfunction]
#[pyo3(signature = (time, event, covariates, config))]
pub fn sis_cox(
    time: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<f64>,
    config: &SISConfig,
) -> PyResult<SISResult> {
    let n = time.len();
    if event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and event must have same length",
        ));
    }

    let p = if covariates.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "covariates cannot be empty",
        ));
    } else {
        covariates.len() / n
    };

    let n_select = if config.n_select > 0 {
        config.n_select.min(p)
    } else {
        (n as f64 / (n as f64).ln()).floor() as usize
    };

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| time[b].partial_cmp(&time[a]).unwrap());

    let marginal_scores: Vec<f64> = (0..p)
        .into_par_iter()
        .map(|j| {
            let mut beta = 0.0;

            for _ in 0..20 {
                let mut gradient = 0.0;
                let mut hessian = 0.0;
                let mut risk_sum = 0.0;
                let mut weighted_x = 0.0;
                let mut weighted_xx = 0.0;

                for &i in &sorted_indices {
                    let x_ij = covariates[i * p + j];
                    let exp_bx = (beta * x_ij).clamp(-700.0, 700.0).exp();

                    risk_sum += exp_bx;
                    weighted_x += exp_bx * x_ij;
                    weighted_xx += exp_bx * x_ij * x_ij;

                    if event[i] == 1 && risk_sum > 0.0 {
                        let x_bar = weighted_x / risk_sum;
                        let x_sq_bar = weighted_xx / risk_sum;
                        gradient += x_ij - x_bar;
                        hessian += x_sq_bar - x_bar * x_bar;
                    }
                }

                if hessian.abs() > 1e-10 {
                    beta += gradient / hessian;
                    beta = beta.clamp(-10.0, 10.0);
                }
            }

            beta.abs()
        })
        .collect();

    let mut ranking: Vec<usize> = (0..p).collect();
    ranking.sort_by(|&a, &b| {
        marginal_scores[b]
            .partial_cmp(&marginal_scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut iteration_selections = Vec::new();

    let selected_features = if config.iterative {
        let mut selected: Vec<usize> = ranking[..n_select.min(p)].to_vec();
        iteration_selections.push(selected.clone());

        for iter in 0..config.max_iter {
            let residual_scores: Vec<f64> = (0..p)
                .map(|j| {
                    if selected.contains(&j) {
                        0.0
                    } else {
                        let corr_with_selected: f64 = selected
                            .iter()
                            .map(|&k| {
                                let mut sum_jk = 0.0;
                                let mut sum_j = 0.0;
                                let mut sum_k = 0.0;
                                let mut sum_jj = 0.0;
                                let mut sum_kk = 0.0;

                                for i in 0..n {
                                    let x_j = covariates[i * p + j];
                                    let x_k = covariates[i * p + k];
                                    sum_jk += x_j * x_k;
                                    sum_j += x_j;
                                    sum_k += x_k;
                                    sum_jj += x_j * x_j;
                                    sum_kk += x_k * x_k;
                                }

                                let cov =
                                    sum_jk / n as f64 - (sum_j / n as f64) * (sum_k / n as f64);
                                let var_j = sum_jj / n as f64 - (sum_j / n as f64).powi(2);
                                let var_k = sum_kk / n as f64 - (sum_k / n as f64).powi(2);

                                if var_j > 1e-10 && var_k > 1e-10 {
                                    (cov / (var_j.sqrt() * var_k.sqrt())).abs()
                                } else {
                                    0.0
                                }
                            })
                            .fold(0.0f64, f64::max);

                        marginal_scores[j] * (1.0 - corr_with_selected)
                    }
                })
                .collect();

            let mut new_ranking: Vec<usize> = (0..p).collect();
            new_ranking.sort_by(|&a, &b| {
                residual_scores[b]
                    .partial_cmp(&residual_scores[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let new_selected: Vec<usize> = new_ranking
                .iter()
                .filter(|&&j| !selected.contains(&j) && residual_scores[j] > config.threshold)
                .take(n_select / 2)
                .copied()
                .collect();

            if new_selected.is_empty() {
                break;
            }

            selected.extend(new_selected);
            iteration_selections.push(selected.clone());
        }

        selected
    } else {
        ranking[..n_select.min(p)].to_vec()
    };

    Ok(SISResult {
        selected_features: selected_features.clone(),
        marginal_scores,
        ranking,
        n_selected: selected_features.len(),
        iteration_selections,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct StabilitySelectionConfig {
    #[pyo3(get, set)]
    pub n_bootstrap: usize,
    #[pyo3(get, set)]
    pub subsample_ratio: f64,
    #[pyo3(get, set)]
    pub lambda_range: Vec<f64>,
    #[pyo3(get, set)]
    pub threshold: f64,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl StabilitySelectionConfig {
    #[new]
    #[pyo3(signature = (n_bootstrap=100, subsample_ratio=0.5, lambda_range=None, threshold=0.6, seed=None))]
    pub fn new(
        n_bootstrap: usize,
        subsample_ratio: f64,
        lambda_range: Option<Vec<f64>>,
        threshold: f64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if subsample_ratio <= 0.0 || subsample_ratio > 1.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "subsample_ratio must be in (0, 1]",
            ));
        }
        if threshold <= 0.0 || threshold > 1.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "threshold must be in (0, 1]",
            ));
        }
        Ok(StabilitySelectionConfig {
            n_bootstrap,
            subsample_ratio,
            lambda_range: lambda_range.unwrap_or_else(|| vec![0.01, 0.1, 1.0]),
            threshold,
            seed,
        })
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct StabilitySelectionResult {
    #[pyo3(get)]
    pub selected_features: Vec<usize>,
    #[pyo3(get)]
    pub selection_probabilities: Vec<f64>,
    #[pyo3(get)]
    pub stable_features: Vec<usize>,
    #[pyo3(get)]
    pub per_lambda_selections: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub n_selected: usize,
}

#[pyfunction]
#[pyo3(signature = (time, event, covariates, config))]
pub fn stability_selection_cox(
    time: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<f64>,
    config: &StabilitySelectionConfig,
) -> PyResult<StabilitySelectionResult> {
    let n = time.len();
    if event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and event must have same length",
        ));
    }

    let p = if covariates.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "covariates cannot be empty",
        ));
    } else {
        covariates.len() / n
    };

    let mut rng = fastrand::Rng::new();
    if let Some(seed) = config.seed {
        rng.seed(seed);
    }

    let n_lambdas = config.lambda_range.len();
    let mut selection_counts: Vec<Vec<usize>> = vec![vec![0; p]; n_lambdas];

    for _bootstrap in 0..config.n_bootstrap {
        let sample_size = (n as f64 * config.subsample_ratio) as usize;
        let mut sample_indices: Vec<usize> = (0..n).collect();
        for i in 0..sample_size.min(n) {
            let j = i + rng.usize(0..(n - i));
            sample_indices.swap(i, j);
        }
        sample_indices.truncate(sample_size);

        let sample_time: Vec<f64> = sample_indices.iter().map(|&i| time[i]).collect();
        let sample_event: Vec<i32> = sample_indices.iter().map(|&i| event[i]).collect();
        let cov_ref = &covariates;
        let sample_cov: Vec<f64> = sample_indices
            .iter()
            .flat_map(|&i| (0..p).map(move |j| cov_ref[i * p + j]))
            .collect();

        for (lambda_idx, &lambda) in config.lambda_range.iter().enumerate() {
            let selected = fit_lasso_simple(&sample_time, &sample_event, &sample_cov, lambda);

            for j in selected {
                selection_counts[lambda_idx][j] += 1;
            }
        }
    }

    let per_lambda_selections: Vec<Vec<f64>> = selection_counts
        .iter()
        .map(|counts| {
            counts
                .iter()
                .map(|&c| c as f64 / config.n_bootstrap as f64)
                .collect()
        })
        .collect();

    let selection_probabilities: Vec<f64> = (0..p)
        .map(|j| {
            per_lambda_selections
                .iter()
                .map(|probs| probs[j])
                .fold(0.0f64, f64::max)
        })
        .collect();

    let selected_features: Vec<usize> = (0..p)
        .filter(|&j| selection_probabilities[j] >= config.threshold)
        .collect();

    let stable_features: Vec<usize> = (0..p)
        .filter(|&j| selection_probabilities[j] >= 0.9)
        .collect();

    Ok(StabilitySelectionResult {
        selected_features: selected_features.clone(),
        selection_probabilities,
        stable_features,
        per_lambda_selections,
        n_selected: selected_features.len(),
    })
}

fn fit_lasso_simple(time: &[f64], event: &[i32], covariates: &[f64], lambda: f64) -> Vec<usize> {
    let n = time.len();
    let p = covariates.len() / n;

    let mut beta = vec![0.0; p];

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| time[b].partial_cmp(&time[a]).unwrap());

    for _ in 0..50 {
        for j in 0..p {
            let eta: Vec<f64> = (0..n)
                .map(|i| {
                    let mut e = 0.0;
                    for k in 0..p {
                        if k != j {
                            e += covariates[i * p + k] * beta[k];
                        }
                    }
                    e.clamp(-700.0, 700.0)
                })
                .collect();

            let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

            let mut gradient = 0.0;
            let mut hessian = 0.0;
            let mut risk_sum = 0.0;
            let mut weighted_x = 0.0;
            let mut weighted_xx = 0.0;

            for &i in &sorted_indices {
                let x_ij = covariates[i * p + j];
                let w = exp_eta[i] * (beta[j] * x_ij).clamp(-700.0, 700.0).exp();

                risk_sum += w;
                weighted_x += w * x_ij;
                weighted_xx += w * x_ij * x_ij;

                if event[i] == 1 && risk_sum > 0.0 {
                    let x_bar = weighted_x / risk_sum;
                    let x_sq_bar = weighted_xx / risk_sum;
                    gradient += x_ij - x_bar;
                    hessian += x_sq_bar - x_bar * x_bar;
                }
            }

            if hessian.abs() > 1e-10 {
                let z = beta[j] + gradient / hessian;
                beta[j] = soft_threshold(z, lambda / hessian);
                beta[j] = beta[j].clamp(-10.0, 10.0);
            }
        }
    }

    (0..p).filter(|&j| beta[j].abs() > 1e-10).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_lasso() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let covariates = vec![
            0.5, 0.3, 0.2, 0.7, 0.4, 0.1, 0.8, 0.2, 0.3, 0.6, 0.5, 0.4, 0.3, 0.8, 0.6, 0.2, 0.4,
            0.5, 0.7, 0.3, 0.2, 0.4, 0.8, 0.5,
        ];
        let groups = vec![0, 0, 1];

        let config = GroupLassoConfig::new(0.1, 100, 1e-4, true, None).unwrap();
        let result = group_lasso_cox(time, event, covariates, groups, &config).unwrap();

        assert_eq!(result.coefficients.len(), 3);
        assert!(result.n_groups == 2);
    }

    #[test]
    fn test_sparse_boosting() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let covariates = vec![
            0.5, 0.3, 0.7, 0.4, 0.8, 0.2, 0.6, 0.5, 0.3, 0.8, 0.4, 0.5, 0.7, 0.3, 0.4, 0.8,
        ];

        let config = SparseBoostingConfig::new(50, 0.1, 0.8, 10, 0.01, Some(42)).unwrap();
        let result = sparse_boosting_cox(time, event, covariates, &config).unwrap();

        assert_eq!(result.coefficients.len(), 2);
        assert!(!result.iteration_scores.is_empty());
    }

    #[test]
    fn test_sis() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let covariates = vec![
            0.5, 0.3, 0.2, 0.7, 0.4, 0.1, 0.8, 0.2, 0.3, 0.6, 0.5, 0.4, 0.3, 0.8, 0.6, 0.2, 0.4,
            0.5, 0.7, 0.3, 0.2, 0.4, 0.8, 0.5,
        ];

        let config = SISConfig::new(Some(2), false, 5, 0.0).unwrap();
        let result = sis_cox(time, event, covariates, &config).unwrap();

        assert_eq!(result.n_selected, 2);
        assert_eq!(result.ranking.len(), 3);
    }

    #[test]
    fn test_stability_selection() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let covariates = vec![
            0.5, 0.3, 0.7, 0.4, 0.8, 0.2, 0.6, 0.5, 0.3, 0.8, 0.4, 0.5, 0.7, 0.3, 0.4, 0.8,
        ];

        let config =
            StabilitySelectionConfig::new(20, 0.5, Some(vec![0.1, 0.5]), 0.3, Some(42)).unwrap();
        let result = stability_selection_cox(time, event, covariates, &config).unwrap();

        assert_eq!(result.selection_probabilities.len(), 2);
    }
}
