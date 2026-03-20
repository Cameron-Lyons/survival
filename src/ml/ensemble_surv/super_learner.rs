
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]

pub struct SuperLearnerConfig {
    #[pyo3(get, set)]
    pub n_folds: usize,
    #[pyo3(get, set)]
    pub meta_learner: String,
    #[pyo3(get, set)]
    pub include_original_features: bool,
    #[pyo3(get, set)]
    pub optimize_weights: bool,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl SuperLearnerConfig {
    #[new]
    #[pyo3(signature = (
        n_folds=5,
        meta_learner="nnls",
        include_original_features=false,
        optimize_weights=true,
        seed=None
    ))]
    pub fn new(
        n_folds: usize,
        meta_learner: &str,
        include_original_features: bool,
        optimize_weights: bool,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if n_folds < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "n_folds must be at least 2",
            ));
        }
        Ok(Self {
            n_folds,
            meta_learner: meta_learner.to_string(),
            include_original_features,
            optimize_weights,
            seed,
        })
    }
}

fn create_cv_folds(n: usize, n_folds: usize, seed: u64) -> Vec<Vec<usize>> {
    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng_state = seed;
    lcg64_shuffle_with_state(&mut indices, &mut rng_state);

    let fold_size = n / n_folds;
    let mut folds = Vec::with_capacity(n_folds);

    for i in 0..n_folds {
        let start = i * fold_size;
        let end = if i == n_folds - 1 {
            n
        } else {
            (i + 1) * fold_size
        };
        folds.push(indices[start..end].to_vec());
    }

    folds
}

fn fit_base_cox(
    time: &[f64],
    event: &[i32],
    covariates: &[Vec<f64>],
    train_indices: &[usize],
    learning_rate: f64,
    n_iter: usize,
) -> Vec<f64> {
    let n_features = if covariates.is_empty() {
        0
    } else {
        covariates[0].len()
    };

    let mut coefficients = vec![0.0; n_features];

    let mut sorted_indices: Vec<usize> = train_indices.to_vec();
    sorted_indices.sort_by(|&a, &b| {
        time[b]
            .partial_cmp(&time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for _ in 0..n_iter {
        let linear_pred: Vec<f64> = sorted_indices
            .iter()
            .map(|&i| {
                covariates[i]
                    .iter()
                    .zip(coefficients.iter())
                    .map(|(&x, &b)| x * b)
                    .sum()
            })
            .collect();

        let exp_lp: Vec<f64> = linear_pred.iter().map(|&lp| lp.exp()).collect();

        let mut gradient = vec![0.0; n_features];
        let mut risk_sum = 0.0;
        let mut weighted_sum = vec![0.0; n_features];

        for (idx, &i) in sorted_indices.iter().enumerate() {
            risk_sum += exp_lp[idx];
            for (j, &xij) in covariates[i].iter().enumerate() {
                weighted_sum[j] += xij * exp_lp[idx];
            }

            if event[i] == 1 {
                for (j, g) in gradient.iter_mut().enumerate() {
                    *g += covariates[i][j] - weighted_sum[j] / risk_sum;
                }
            }
        }

        for (b, g) in coefficients.iter_mut().zip(gradient.iter()) {
            *b += learning_rate * g / train_indices.len() as f64;
        }
    }

    coefficients
}

fn nnls_weights(predictions: &[Vec<f64>], outcomes: &[f64], n_models: usize) -> Vec<f64> {
    let n = outcomes.len();
    let mut weights = vec![1.0 / n_models as f64; n_models];

    for _ in 0..100 {
        let mut gradient = vec![0.0; n_models];

        for i in 0..n {
            let pred: f64 = (0..n_models).map(|m| weights[m] * predictions[m][i]).sum();
            let error = pred - outcomes[i];

            for m in 0..n_models {
                gradient[m] += 2.0 * error * predictions[m][i] / n as f64;
            }
        }

        for (w, g) in weights.iter_mut().zip(gradient.iter()) {
            *w = (*w - 0.01 * g).max(0.0);
        }

        let sum: f64 = weights.iter().sum();
        if sum > 0.0 {
            for w in &mut weights {
                *w /= sum;
            }
        }
    }

    weights
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SuperLearnerResult {
    #[pyo3(get)]
    pub weights: Vec<f64>,
    #[pyo3(get)]
    pub cv_risks: Vec<f64>,
    #[pyo3(get)]
    pub model_names: Vec<String>,
    #[pyo3(get)]
    pub ensemble_c_index: f64,
    #[pyo3(get)]
    pub individual_c_indices: Vec<f64>,
}

#[pymethods]
impl SuperLearnerResult {
    fn __repr__(&self) -> String {
        format!(
            "SuperLearnerResult(n_models={}, C-index={:.4})",
            self.weights.len(),
            self.ensemble_c_index
        )
    }

    fn best_model(&self) -> (String, f64) {
        let (idx, &max_c) = self
            .individual_c_indices
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));
        (self.model_names[idx].clone(), max_c)
    }
}

fn compute_c_index(time: &[f64], event: &[i32], risk: &[f64]) -> f64 {
    let n = time.len();
    let (concordant, discordant) = (0..n)
        .into_par_iter()
        .map(|i| {
            if event[i] != 1 {
                return (0.0, 0.0);
            }

            let mut concordant = 0.0;
            let mut discordant = 0.0;
            for j in 0..n {
                if time[j] > time[i] {
                    if risk[i] > risk[j] {
                        concordant += 1.0;
                    } else if risk[i] < risk[j] {
                        discordant += 1.0;
                    } else {
                        concordant += 0.5;
                        discordant += 0.5;
                    }
                }
            }
            (concordant, discordant)
        })
        .reduce(|| (0.0, 0.0), |(c1, d1), (c2, d2)| (c1 + c2, d1 + d2));

    if concordant + discordant > 0.0 {
        concordant / (concordant + discordant)
    } else {
        0.5
    }
}

#[pyfunction]
#[pyo3(signature = (
    time,
    event,
    covariates,
    base_learner_predictions,
    model_names,
    config
))]
pub fn super_learner_survival(
    time: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<Vec<f64>>,
    base_learner_predictions: Vec<Vec<f64>>,
    model_names: Vec<String>,
    config: SuperLearnerConfig,
) -> PyResult<SuperLearnerResult> {
    let n = time.len();
    let n_models = base_learner_predictions.len();

    if n == 0 || event.len() != n || covariates.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays must have the same non-zero length",
        ));
    }
    if n_models == 0 || model_names.len() != n_models {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Must provide predictions from at least one model",
        ));
    }

    let seed = config.seed.unwrap_or(crate::constants::DEFAULT_RANDOM_SEED);
    let folds = create_cv_folds(n, config.n_folds, seed);

    let mut cv_predictions: Vec<Vec<f64>> = vec![vec![0.0; n]; n_models];

    for test_indices in folds.iter() {
        let train_indices: Vec<usize> = (0..n).filter(|i| !test_indices.contains(i)).collect();
        cv_predictions
            .par_iter_mut()
            .enumerate()
            .for_each(|(m, model_cv_predictions)| {
                let train_sum: f64 = train_indices
                    .iter()
                    .map(|&i| base_learner_predictions[m][i])
                    .sum();
                let scale = if train_indices.is_empty() {
                    1.0
                } else {
                    train_sum / train_indices.len() as f64
                };

                for &test_i in test_indices {
                    model_cv_predictions[test_i] =
                        base_learner_predictions[m][test_i] / scale.max(1e-10);
                }
            });
    }

    let outcomes: Vec<f64> = event.iter().map(|&e| e as f64).collect();
    let weights = if config.optimize_weights {
        nnls_weights(&cv_predictions, &outcomes, n_models)
    } else {
        vec![1.0 / n_models as f64; n_models]
    };

    let ensemble_risk: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            (0..n_models)
                .map(|m| weights[m] * base_learner_predictions[m][i])
                .sum()
        })
        .collect();

    let ensemble_c_index = compute_c_index(&time, &event, &ensemble_risk);

    let individual_c_indices: Vec<f64> = base_learner_predictions
        .par_iter()
        .map(|preds| compute_c_index(&time, &event, preds))
        .collect();

    let cv_risks: Vec<f64> = (0..n_models)
        .into_par_iter()
        .map(|m| {
            cv_predictions[m]
                .iter()
                .zip(outcomes.iter())
                .map(|(&p, &o)| (p - o).powi(2))
                .sum::<f64>()
                / n as f64
        })
        .collect();

    Ok(SuperLearnerResult {
        weights,
        cv_risks,
        model_names,
        ensemble_c_index,
        individual_c_indices,
    })
}
