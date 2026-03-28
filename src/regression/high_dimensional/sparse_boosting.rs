
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
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
#[pyclass(from_py_object)]
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
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and event must not be empty",
        ));
    }
    if event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and event must have same length",
        ));
    }

    let p = if covariates.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "covariates cannot be empty",
        ));
    } else if !covariates.len().is_multiple_of(n) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "covariates length must be divisible by number of observations",
        ));
    } else {
        covariates.len() / n
    };
    if p == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "number of covariates must be positive",
        ));
    }

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
    sorted_indices.sort_by(|&a, &b| {
        time[b]
            .partial_cmp(&time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

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

        let Some((best_j, best_grad)) = gradients
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.abs()
                    .partial_cmp(&b.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(j, &g)| (j, g))
        else {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "internal error: unable to select gradient",
            ));
        };

        let update = config.learning_rate
            * best_grad.signum()
            * (best_grad.abs() - config.l1_penalty).max(0.0);

        if update.abs() > crate::constants::DIVISION_FLOOR {
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

    let selected_features: Vec<usize> = (0..p)
        .filter(|&j| beta[j].abs() > crate::constants::DIVISION_FLOOR)
        .collect();

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
    sorted_indices.sort_by(|&a, &b| {
        time[b]
            .partial_cmp(&time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

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
