
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct ComponentwiseBoostingConfig {
    #[pyo3(get, set)]
    pub n_iterations: usize,
    #[pyo3(get, set)]
    pub learning_rate: f64,
    #[pyo3(get, set)]
    pub early_stopping_rounds: Option<usize>,
    #[pyo3(get, set)]
    pub subsample_ratio: f64,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl ComponentwiseBoostingConfig {
    #[new]
    #[pyo3(signature = (
        n_iterations=100,
        learning_rate=0.1,
        early_stopping_rounds=None,
        subsample_ratio=1.0,
        seed=None
    ))]
    pub fn new(
        n_iterations: usize,
        learning_rate: f64,
        early_stopping_rounds: Option<usize>,
        subsample_ratio: f64,
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
        Ok(Self {
            n_iterations,
            learning_rate,
            early_stopping_rounds,
            subsample_ratio,
            seed,
        })
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct ComponentwiseBoostingResult {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    #[pyo3(get)]
    pub selected_features: Vec<usize>,
    #[pyo3(get)]
    pub iteration_log_likelihood: Vec<f64>,
    #[pyo3(get)]
    pub feature_importance: Vec<f64>,
    #[pyo3(get)]
    pub optimal_iterations: usize,
}

#[pymethods]
impl ComponentwiseBoostingResult {
    fn __repr__(&self) -> String {
        format!(
            "ComponentwiseBoostingResult(n_selected={}, iterations={})",
            self.selected_features
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            self.optimal_iterations
        )
    }

    fn predict_risk(&self, covariates: Vec<Vec<f64>>) -> Vec<f64> {
        covariates
            .par_iter()
            .map(|x| {
                x.iter()
                    .zip(self.coefficients.iter())
                    .map(|(&xi, &bi)| xi * bi)
                    .sum::<f64>()
                    .exp()
            })
            .collect()
    }
}

fn compute_partial_log_likelihood(time: &[f64], event: &[i32], linear_pred: &[f64]) -> f64 {
    let n = time.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[b]
            .partial_cmp(&time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let exp_lp: Vec<f64> = linear_pred.iter().map(|&lp| lp.exp()).collect();

    let mut ll = 0.0;
    let mut risk_sum = 0.0;

    for &i in &indices {
        risk_sum += exp_lp[i];
        if event[i] == 1 {
            ll += linear_pred[i] - risk_sum.ln();
        }
    }

    ll
}

#[pyfunction]
#[pyo3(signature = (
    time,
    event,
    covariates,
    config
))]
pub fn componentwise_boosting(
    time: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<Vec<f64>>,
    config: ComponentwiseBoostingConfig,
) -> PyResult<ComponentwiseBoostingResult> {
    let n = time.len();
    if n == 0 || event.len() != n || covariates.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays must have the same non-zero length",
        ));
    }

    let n_features = if covariates.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Covariates cannot be empty",
        ));
    } else {
        covariates[0].len()
    };

    let seed = config.seed.unwrap_or(crate::constants::DEFAULT_RANDOM_SEED);
    let mut rng_state = seed;

    let mut coefficients: Vec<f64> = vec![0.0; n_features];
    let mut linear_pred: Vec<f64> = vec![0.0; n];
    let mut selected_features = Vec::new();
    let mut iteration_log_likelihood = Vec::new();
    let mut feature_selection_count = vec![0usize; n_features];

    let mut best_ll = f64::NEG_INFINITY;
    let mut rounds_without_improvement = 0;
    let mut optimal_iterations = 0;

    for iter in 0..config.n_iterations {
        let sample_indices: Vec<usize> = if config.subsample_ratio < 1.0 {
            let sample_size = (n as f64 * config.subsample_ratio).ceil() as usize;
            let mut indices: Vec<usize> = (0..n).collect();
            lcg64_shuffle_with_state(&mut indices, &mut rng_state);
            indices.truncate(sample_size);
            indices
        } else {
            (0..n).collect()
        };

        let exp_lp: Vec<f64> = linear_pred.iter().map(|&lp| lp.exp()).collect();

        let mut sorted_indices: Vec<usize> = sample_indices.clone();
        sorted_indices.sort_by(|&a, &b| {
            time[b]
                .partial_cmp(&time[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut best_feature = 0;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_update = 0.0;
        for (j, _) in coefficients.iter().enumerate().take(n_features) {
            let mut gradient = 0.0;
            let mut hessian = 0.0;
            let mut risk_sum = 0.0;
            let mut weighted_sum = 0.0;
            let mut weighted_sq_sum = 0.0;

            for &i in &sorted_indices {
                risk_sum += exp_lp[i];
                weighted_sum += covariates[i][j] * exp_lp[i];
                weighted_sq_sum += covariates[i][j].powi(2) * exp_lp[i];

                if event[i] == 1 {
                    let mean = weighted_sum / risk_sum;
                    gradient += covariates[i][j] - mean;
                    hessian += weighted_sq_sum / risk_sum - mean.powi(2);
                }
            }

            if hessian.abs() > 1e-10 {
                let update = gradient / hessian;
                let score = gradient.abs();

                if score > best_score {
                    best_score = score;
                    best_feature = j;
                    best_update = update;
                }
            }
        }

        coefficients[best_feature] += config.learning_rate * best_update;
        selected_features.push(best_feature);
        feature_selection_count[best_feature] += 1;

        for i in 0..n {
            linear_pred[i] = coefficients
                .iter()
                .zip(covariates[i].iter())
                .map(|(&b, &x)| b * x)
                .sum();
        }

        let ll = compute_partial_log_likelihood(&time, &event, &linear_pred);
        iteration_log_likelihood.push(ll);

        if ll > best_ll {
            best_ll = ll;
            optimal_iterations = iter + 1;
            rounds_without_improvement = 0;
        } else {
            rounds_without_improvement += 1;
        }

        if let Some(patience) = config.early_stopping_rounds
            && rounds_without_improvement >= patience
        {
            break;
        }
    }

    let total_selections: f64 = feature_selection_count.iter().sum::<usize>() as f64;
    let feature_importance: Vec<f64> = if total_selections > 0.0 {
        feature_selection_count
            .iter()
            .map(|&c| c as f64 / total_selections)
            .collect()
    } else {
        vec![0.0; n_features]
    };

    Ok(ComponentwiseBoostingResult {
        coefficients,
        selected_features,
        iteration_log_likelihood,
        feature_importance,
        optimal_iterations,
    })
}
