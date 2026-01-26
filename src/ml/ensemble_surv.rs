#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
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
    for i in (1..n).rev() {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (rng_state as usize) % (i + 1);
        indices.swap(i, j);
    }

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
    sorted_indices.sort_by(|&a, &b| time[b].partial_cmp(&time[a]).unwrap());

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
#[pyclass]
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
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((0, &0.0));
        (self.model_names[idx].clone(), max_c)
    }
}

fn compute_c_index(time: &[f64], event: &[i32], risk: &[f64]) -> f64 {
    let n = time.len();
    let mut concordant = 0.0;
    let mut discordant = 0.0;

    for i in 0..n {
        if event[i] == 1 {
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
        }
    }

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

    let seed = config.seed.unwrap_or(42);
    let folds = create_cv_folds(n, config.n_folds, seed);

    let mut cv_predictions: Vec<Vec<f64>> = vec![vec![0.0; n]; n_models];

    for test_indices in folds.iter() {
        let train_indices: Vec<usize> = (0..n).filter(|i| !test_indices.contains(i)).collect();

        for m in 0..n_models {
            let train_preds: Vec<f64> = train_indices
                .iter()
                .map(|&i| base_learner_predictions[m][i])
                .collect();
            let test_preds: Vec<f64> = test_indices
                .iter()
                .map(|&i| base_learner_predictions[m][i])
                .collect();

            let scale = if !train_preds.is_empty() {
                train_preds.iter().sum::<f64>() / train_preds.len() as f64
            } else {
                1.0
            };

            for (idx, &test_i) in test_indices.iter().enumerate() {
                cv_predictions[m][test_i] = test_preds[idx] / scale.max(1e-10);
            }
        }
    }

    let outcomes: Vec<f64> = event.iter().map(|&e| e as f64).collect();
    let weights = if config.optimize_weights {
        nnls_weights(&cv_predictions, &outcomes, n_models)
    } else {
        vec![1.0 / n_models as f64; n_models]
    };

    let ensemble_risk: Vec<f64> = (0..n)
        .map(|i| {
            (0..n_models)
                .map(|m| weights[m] * base_learner_predictions[m][i])
                .sum()
        })
        .collect();

    let ensemble_c_index = compute_c_index(&time, &event, &ensemble_risk);

    let individual_c_indices: Vec<f64> = base_learner_predictions
        .iter()
        .map(|preds| compute_c_index(&time, &event, preds))
        .collect();

    let cv_risks: Vec<f64> = (0..n_models)
        .map(|m| {
            let mse: f64 = cv_predictions[m]
                .iter()
                .zip(outcomes.iter())
                .map(|(&p, &o)| (p - o).powi(2))
                .sum::<f64>()
                / n as f64;
            mse
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

#[derive(Debug, Clone)]
#[pyclass]
pub struct StackingConfig {
    #[pyo3(get, set)]
    pub n_folds: usize,
    #[pyo3(get, set)]
    pub meta_model: String,
    #[pyo3(get, set)]
    pub use_probabilities: bool,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl StackingConfig {
    #[new]
    #[pyo3(signature = (
        n_folds=5,
        meta_model="cox",
        use_probabilities=true,
        seed=None
    ))]
    pub fn new(
        n_folds: usize,
        meta_model: &str,
        use_probabilities: bool,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if n_folds < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "n_folds must be at least 2",
            ));
        }
        Ok(Self {
            n_folds,
            meta_model: meta_model.to_string(),
            use_probabilities,
            seed,
        })
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct StackingResult {
    #[pyo3(get)]
    pub meta_coefficients: Vec<f64>,
    #[pyo3(get)]
    pub stacked_predictions: Vec<f64>,
    #[pyo3(get)]
    pub c_index: f64,
    #[pyo3(get)]
    pub base_model_importance: Vec<f64>,
}

#[pymethods]
impl StackingResult {
    fn __repr__(&self) -> String {
        format!(
            "StackingResult(n_base_models={}, C-index={:.4})",
            self.meta_coefficients.len(),
            self.c_index
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    time,
    event,
    base_predictions,
    config
))]
pub fn stacking_survival(
    time: Vec<f64>,
    event: Vec<i32>,
    base_predictions: Vec<Vec<f64>>,
    config: StackingConfig,
) -> PyResult<StackingResult> {
    let n = time.len();
    let n_models = base_predictions.len();

    if n == 0 || event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and event must have the same non-zero length",
        ));
    }
    if n_models == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Must provide at least one base model",
        ));
    }

    let seed = config.seed.unwrap_or(42);
    let folds = create_cv_folds(n, config.n_folds, seed);

    let mut oof_predictions: Vec<Vec<f64>> = vec![vec![0.0; n]; n_models];

    for test_indices in &folds {
        let train_indices: Vec<usize> = (0..n).filter(|i| !test_indices.contains(i)).collect();

        for m in 0..n_models {
            let train_mean: f64 = train_indices
                .iter()
                .map(|&i| base_predictions[m][i])
                .sum::<f64>()
                / train_indices.len() as f64;

            for &test_i in test_indices {
                oof_predictions[m][test_i] = base_predictions[m][test_i] / train_mean.max(1e-10);
            }
        }
    }

    let meta_features: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n_models).map(|m| oof_predictions[m][i]).collect())
        .collect();

    let train_indices: Vec<usize> = (0..n).collect();
    let meta_coefficients = fit_base_cox(&time, &event, &meta_features, &train_indices, 0.01, 100);

    let stacked_predictions: Vec<f64> = meta_features
        .iter()
        .map(|x| {
            x.iter()
                .zip(meta_coefficients.iter())
                .map(|(&xi, &bi)| xi * bi)
                .sum::<f64>()
                .exp()
        })
        .collect();

    let c_index = compute_c_index(&time, &event, &stacked_predictions);

    let total_abs: f64 = meta_coefficients.iter().map(|&c| c.abs()).sum();
    let base_model_importance: Vec<f64> = if total_abs > 0.0 {
        meta_coefficients
            .iter()
            .map(|&c| c.abs() / total_abs)
            .collect()
    } else {
        vec![1.0 / n_models as f64; n_models]
    };

    Ok(StackingResult {
        meta_coefficients,
        stacked_predictions,
        c_index,
        base_model_importance,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
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
#[pyclass]
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
    indices.sort_by(|&a, &b| time[b].partial_cmp(&time[a]).unwrap());

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

    let seed = config.seed.unwrap_or(42);
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
            for i in (1..n).rev() {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let j = (rng_state as usize) % (i + 1);
                indices.swap(i, j);
            }
            indices.truncate(sample_size);
            indices
        } else {
            (0..n).collect()
        };

        let exp_lp: Vec<f64> = linear_pred.iter().map(|&lp| lp.exp()).collect();

        let mut sorted_indices: Vec<usize> = sample_indices.clone();
        sorted_indices.sort_by(|&a, &b| time[b].partial_cmp(&time[a]).unwrap());

        let mut best_feature = 0;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_update = 0.0;

        #[allow(clippy::needless_range_loop)]
        for j in 0..n_features {
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

#[derive(Debug, Clone)]
#[pyclass]
pub struct BlendingResult {
    #[pyo3(get)]
    pub blend_weights: Vec<f64>,
    #[pyo3(get)]
    pub blended_predictions: Vec<f64>,
    #[pyo3(get)]
    pub validation_c_index: f64,
}

#[pymethods]
impl BlendingResult {
    fn __repr__(&self) -> String {
        format!(
            "BlendingResult(n_models={}, val_C={:.4})",
            self.blend_weights.len(),
            self.validation_c_index
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    val_time,
    val_event,
    val_predictions,
    test_predictions
))]
pub fn blending_survival(
    val_time: Vec<f64>,
    val_event: Vec<i32>,
    val_predictions: Vec<Vec<f64>>,
    test_predictions: Vec<Vec<f64>>,
) -> PyResult<BlendingResult> {
    let n_val = val_time.len();
    let n_models = val_predictions.len();

    if n_val == 0 || val_event.len() != n_val {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Validation arrays must have the same non-zero length",
        ));
    }
    if n_models == 0 || test_predictions.len() != n_models {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Must have same number of models for validation and test",
        ));
    }

    let outcomes: Vec<f64> = val_event.iter().map(|&e| e as f64).collect();
    let blend_weights = nnls_weights(&val_predictions, &outcomes, n_models);

    let n_test = test_predictions[0].len();
    let blended_predictions: Vec<f64> = (0..n_test)
        .map(|i| {
            (0..n_models)
                .map(|m| blend_weights[m] * test_predictions[m][i])
                .sum()
        })
        .collect();

    let val_blended: Vec<f64> = (0..n_val)
        .map(|i| {
            (0..n_models)
                .map(|m| blend_weights[m] * val_predictions[m][i])
                .sum()
        })
        .collect();

    let validation_c_index = compute_c_index(&val_time, &val_event, &val_blended);

    Ok(BlendingResult {
        blend_weights,
        blended_predictions,
        validation_c_index,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_super_learner() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let event = vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let covariates: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.1]).collect();
        let pred1: Vec<f64> = (0..10).map(|i| 0.1 + i as f64 * 0.05).collect();
        let pred2: Vec<f64> = (0..10).map(|i| 0.2 + i as f64 * 0.03).collect();

        let config = SuperLearnerConfig::new(3, "nnls", false, true, Some(42)).unwrap();
        let result = super_learner_survival(
            time,
            event,
            covariates,
            vec![pred1, pred2],
            vec!["model1".to_string(), "model2".to_string()],
            config,
        )
        .unwrap();

        assert_eq!(result.weights.len(), 2);
        assert!((result.weights.iter().sum::<f64>() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_stacking() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let event = vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let pred1: Vec<f64> = (0..10).map(|i| 0.1 + i as f64 * 0.05).collect();
        let pred2: Vec<f64> = (0..10).map(|i| 0.2 + i as f64 * 0.03).collect();

        let config = StackingConfig::new(3, "cox", true, Some(42)).unwrap();
        let result = stacking_survival(time, event, vec![pred1, pred2], config).unwrap();

        assert_eq!(result.meta_coefficients.len(), 2);
    }

    #[test]
    fn test_componentwise_boosting() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let event = vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let covariates: Vec<Vec<f64>> = (0..10)
            .map(|i| vec![i as f64 * 0.1, (10 - i) as f64 * 0.1])
            .collect();

        let config = ComponentwiseBoostingConfig::new(50, 0.1, Some(10), 1.0, Some(42)).unwrap();
        let result = componentwise_boosting(time, event, covariates, config).unwrap();

        assert_eq!(result.coefficients.len(), 2);
        assert!(!result.selected_features.is_empty());
    }

    #[test]
    fn test_blending() {
        let val_time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let val_event = vec![1, 0, 1, 0, 1];
        let val_pred1 = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let val_pred2 = vec![0.15, 0.25, 0.35, 0.45, 0.55];
        let test_pred1 = vec![0.2, 0.3, 0.4];
        let test_pred2 = vec![0.25, 0.35, 0.45];

        let result = blending_survival(
            val_time,
            val_event,
            vec![val_pred1, val_pred2],
            vec![test_pred1, test_pred2],
        )
        .unwrap();

        assert_eq!(result.blend_weights.len(), 2);
        assert_eq!(result.blended_predictions.len(), 3);
    }

    #[test]
    fn test_super_learner_config_validation() {
        let result = SuperLearnerConfig::new(1, "nnls", false, true, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_stacking_config_validation() {
        let result = StackingConfig::new(1, "cox", true, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_componentwise_boosting_config_validation() {
        let result = ComponentwiseBoostingConfig::new(100, 0.0, None, 1.0, None);
        assert!(result.is_err());

        let result = ComponentwiseBoostingConfig::new(100, 1.5, None, 1.0, None);
        assert!(result.is_err());

        let result = ComponentwiseBoostingConfig::new(100, 0.1, None, 0.0, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_super_learner_empty_input() {
        let config = SuperLearnerConfig::new(3, "nnls", false, true, Some(42)).unwrap();
        let result = super_learner_survival(vec![], vec![], vec![], vec![], vec![], config);
        assert!(result.is_err());
    }

    #[test]
    fn test_super_learner_uniform_weights() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let event = vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let covariates: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.1]).collect();
        let pred1: Vec<f64> = (0..10).map(|i| 0.1 + i as f64 * 0.05).collect();
        let pred2: Vec<f64> = (0..10).map(|i| 0.2 + i as f64 * 0.03).collect();

        let config = SuperLearnerConfig::new(3, "nnls", false, false, Some(42)).unwrap();
        let result = super_learner_survival(
            time,
            event,
            covariates,
            vec![pred1, pred2],
            vec!["m1".to_string(), "m2".to_string()],
            config,
        )
        .unwrap();

        assert!((result.weights[0] - 0.5).abs() < 1e-6);
        assert!((result.weights[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_componentwise_boosting_predict_risk() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let event = vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let covariates: Vec<Vec<f64>> = (0..10)
            .map(|i| vec![i as f64 * 0.1, (10 - i) as f64 * 0.1])
            .collect();

        let config = ComponentwiseBoostingConfig::new(50, 0.1, Some(10), 1.0, Some(42)).unwrap();
        let result = componentwise_boosting(time, event, covariates.clone(), config).unwrap();

        let risks = result.predict_risk(covariates);
        assert_eq!(risks.len(), 10);
        assert!(risks.iter().all(|&r| r > 0.0));
    }

    #[test]
    fn test_stacking_empty_input() {
        let config = StackingConfig::new(3, "cox", true, Some(42)).unwrap();
        let result = stacking_survival(vec![], vec![], vec![], config);
        assert!(result.is_err());
    }

    #[test]
    fn test_blending_empty_input() {
        let result = blending_survival(vec![], vec![], vec![], vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_blending_mismatched_models() {
        let val_time = vec![1.0, 2.0, 3.0];
        let val_event = vec![1, 0, 1];
        let val_preds = vec![vec![0.1, 0.2, 0.3]];
        let test_preds = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
        let result = blending_survival(val_time, val_event, val_preds, test_preds);
        assert!(result.is_err());
    }

    #[test]
    fn test_componentwise_boosting_feature_importance() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let event = vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let covariates: Vec<Vec<f64>> = (0..10)
            .map(|i| vec![i as f64 * 0.1, (10 - i) as f64 * 0.1, 0.5])
            .collect();

        let config = ComponentwiseBoostingConfig::new(50, 0.1, None, 1.0, Some(42)).unwrap();
        let result = componentwise_boosting(time, event, covariates, config).unwrap();

        assert_eq!(result.feature_importance.len(), 3);
        let total: f64 = result.feature_importance.iter().sum();
        assert!((total - 1.0).abs() < 1e-6);
    }
}
