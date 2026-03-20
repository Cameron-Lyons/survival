
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
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
#[pyclass(from_py_object)]
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
    sorted_indices.sort_by(|&a, &b| {
        time[b]
            .partial_cmp(&time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

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

                if hessian.abs() > crate::constants::DIVISION_FLOOR {
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

        for _iter in 0..config.max_iter {
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

                                if var_j > crate::constants::DIVISION_FLOOR
                                    && var_k > crate::constants::DIVISION_FLOOR
                                {
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
#[pyclass(from_py_object)]
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
#[pyclass(from_py_object)]
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
    sorted_indices.sort_by(|&a, &b| {
        time[b]
            .partial_cmp(&time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

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

            if hessian.abs() > crate::constants::DIVISION_FLOOR {
                let z = beta[j] + gradient / hessian;
                beta[j] = soft_threshold(z, lambda / hessian);
                beta[j] = beta[j].clamp(-10.0, 10.0);
            }
        }
    }

    (0..p)
        .filter(|&j| beta[j].abs() > crate::constants::DIVISION_FLOOR)
        .collect()
}

