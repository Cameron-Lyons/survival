#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(eq, eq_int)]
pub enum SearchStrategy {
    Grid,
    Random,
    Bayesian,
}

#[pymethods]
impl SearchStrategy {
    fn __repr__(&self) -> String {
        match self {
            SearchStrategy::Grid => "SearchStrategy.Grid".to_string(),
            SearchStrategy::Random => "SearchStrategy.Random".to_string(),
            SearchStrategy::Bayesian => "SearchStrategy.Bayesian".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct HyperparameterSearchConfig {
    #[pyo3(get, set)]
    pub strategy: SearchStrategy,
    #[pyo3(get, set)]
    pub n_iter: usize,
    #[pyo3(get, set)]
    pub cv_folds: usize,
    #[pyo3(get, set)]
    pub scoring: String,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl HyperparameterSearchConfig {
    #[new]
    #[pyo3(signature = (
        strategy=SearchStrategy::Random,
        n_iter=20,
        cv_folds=5,
        scoring=None,
        seed=None
    ))]
    pub fn new(
        strategy: SearchStrategy,
        n_iter: usize,
        cv_folds: usize,
        scoring: Option<String>,
        seed: Option<u64>,
    ) -> Self {
        Self {
            strategy,
            n_iter,
            cv_folds,
            scoring: scoring.unwrap_or_else(|| "c_index".to_string()),
            seed,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct HyperparameterResult {
    #[pyo3(get)]
    pub best_params: Vec<(String, f64)>,
    #[pyo3(get)]
    pub best_score: f64,
    #[pyo3(get)]
    pub all_scores: Vec<f64>,
    #[pyo3(get)]
    pub all_params: Vec<Vec<(String, f64)>>,
    #[pyo3(get)]
    pub cv_scores: Vec<Vec<f64>>,
}

#[pymethods]
impl HyperparameterResult {
    fn __repr__(&self) -> String {
        format!(
            "HyperparameterResult(best_score={:.4}, n_evaluated={})",
            self.best_score,
            self.all_scores.len()
        )
    }

    fn get_param(&self, name: &str) -> Option<f64> {
        self.best_params
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| *v)
    }
}

fn compute_cv_c_index(
    risk_scores: &[f64],
    time: &[f64],
    event: &[i32],
    train_idx: &[usize],
    test_idx: &[usize],
) -> f64 {
    let train_risk: Vec<f64> = train_idx.iter().map(|&i| risk_scores[i]).collect();
    let _train_time: Vec<f64> = train_idx.iter().map(|&i| time[i]).collect();
    let _train_event: Vec<i32> = train_idx.iter().map(|&i| event[i]).collect();

    let mean_risk: f64 = train_risk.iter().sum::<f64>() / train_risk.len() as f64;
    let std_risk: f64 = {
        let var: f64 = train_risk
            .iter()
            .map(|&r| (r - mean_risk).powi(2))
            .sum::<f64>()
            / train_risk.len() as f64;
        var.sqrt().max(1e-10)
    };

    let test_risk: Vec<f64> = test_idx
        .iter()
        .map(|&i| (risk_scores[i] - mean_risk) / std_risk)
        .collect();
    let test_time: Vec<f64> = test_idx.iter().map(|&i| time[i]).collect();
    let test_event: Vec<i32> = test_idx.iter().map(|&i| event[i]).collect();

    let n = test_risk.len();
    if n < 2 {
        return 0.5;
    }

    let mut concordant = 0.0;
    let mut comparable = 0.0;

    for i in 0..n {
        for j in (i + 1)..n {
            if test_event[i] == 1 && test_time[i] < test_time[j] {
                comparable += 1.0;
                if test_risk[i] > test_risk[j] {
                    concordant += 1.0;
                } else if (test_risk[i] - test_risk[j]).abs() < 1e-10 {
                    concordant += 0.5;
                }
            } else if test_event[j] == 1 && test_time[j] < test_time[i] {
                comparable += 1.0;
                if test_risk[j] > test_risk[i] {
                    concordant += 1.0;
                } else if (test_risk[i] - test_risk[j]).abs() < 1e-10 {
                    concordant += 0.5;
                }
            }
        }
    }

    if comparable > 0.0 {
        concordant / comparable
    } else {
        0.5
    }
}

fn create_cv_folds(n: usize, n_folds: usize, rng: &mut fastrand::Rng) -> Vec<Vec<usize>> {
    let mut indices: Vec<usize> = (0..n).collect();
    rng.shuffle(&mut indices);

    let fold_size = n / n_folds;
    let remainder = n % n_folds;

    let mut folds = Vec::with_capacity(n_folds);
    let mut start = 0;

    for i in 0..n_folds {
        let extra = if i < remainder { 1 } else { 0 };
        let end = start + fold_size + extra;
        folds.push(indices[start..end].to_vec());
        start = end;
    }

    folds
}

#[pyfunction]
#[pyo3(signature = (
    risk_scores,
    time,
    event,
    param_grid,
    config=None
))]
pub fn hyperparameter_search(
    risk_scores: Vec<f64>,
    time: Vec<f64>,
    event: Vec<i32>,
    param_grid: Vec<(String, Vec<f64>)>,
    config: Option<HyperparameterSearchConfig>,
) -> PyResult<HyperparameterResult> {
    let config = config.unwrap_or_else(|| {
        HyperparameterSearchConfig::new(SearchStrategy::Random, 20, 5, None, None)
    });

    let n = risk_scores.len();
    if n == 0 || time.len() != n || event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All inputs must have the same non-zero length",
        ));
    }

    if param_grid.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "param_grid must not be empty",
        ));
    }

    let mut rng = fastrand::Rng::new();
    if let Some(s) = config.seed {
        rng.seed(s);
    }

    let folds = create_cv_folds(n, config.cv_folds, &mut rng);

    let param_combinations: Vec<Vec<(String, f64)>> = match config.strategy {
        SearchStrategy::Grid => {
            let mut combos = vec![Vec::new()];
            for (name, values) in &param_grid {
                let mut new_combos = Vec::new();
                for combo in &combos {
                    for &val in values {
                        let mut new_combo = combo.clone();
                        new_combo.push((name.clone(), val));
                        new_combos.push(new_combo);
                    }
                }
                combos = new_combos;
            }
            combos
        }
        SearchStrategy::Random | SearchStrategy::Bayesian => (0..config.n_iter)
            .map(|_| {
                param_grid
                    .iter()
                    .map(|(name, values)| {
                        let idx = rng.usize(0..values.len());
                        (name.clone(), values[idx])
                    })
                    .collect()
            })
            .collect(),
    };

    #[allow(clippy::type_complexity)]
    let results: Vec<(Vec<(String, f64)>, Vec<f64>)> = param_combinations
        .par_iter()
        .map(|params| {
            let param_scale: f64 = params.iter().map(|(_, v)| v).sum::<f64>() / params.len() as f64;

            let cv_scores: Vec<f64> = folds
                .iter()
                .enumerate()
                .map(|(fold_idx, test_fold)| {
                    let train_idx: Vec<usize> = folds
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| *i != fold_idx)
                        .flat_map(|(_, f)| f.iter().copied())
                        .collect();

                    let scaled_risks: Vec<f64> =
                        risk_scores.iter().map(|&r| r * param_scale).collect();

                    compute_cv_c_index(&scaled_risks, &time, &event, &train_idx, test_fold)
                })
                .collect();

            (params.clone(), cv_scores)
        })
        .collect();

    let all_scores: Vec<f64> = results
        .iter()
        .map(|(_, scores)| scores.iter().sum::<f64>() / scores.len() as f64)
        .collect();

    let all_params: Vec<Vec<(String, f64)>> = results.iter().map(|(p, _)| p.clone()).collect();

    let cv_scores: Vec<Vec<f64>> = results.iter().map(|(_, s)| s.clone()).collect();

    let best_idx = all_scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let best_params = all_params[best_idx].clone();
    let best_score = all_scores[best_idx];

    Ok(HyperparameterResult {
        best_params,
        best_score,
        all_scores,
        all_params,
        cv_scores,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct BenchmarkResult {
    #[pyo3(get)]
    pub model_names: Vec<String>,
    #[pyo3(get)]
    pub c_indices: Vec<f64>,
    #[pyo3(get)]
    pub brier_scores: Vec<f64>,
    #[pyo3(get)]
    pub training_times_ms: Vec<f64>,
    #[pyo3(get)]
    pub best_model: String,
}

#[pymethods]
impl BenchmarkResult {
    fn __repr__(&self) -> String {
        format!(
            "BenchmarkResult(best_model={}, n_models={})",
            self.best_model,
            self.model_names.len()
        )
    }

    fn ranking(&self) -> Vec<(String, usize)> {
        let mut indexed: Vec<(usize, f64)> = self.c_indices.iter().copied().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
        indexed
            .iter()
            .enumerate()
            .map(|(rank, (idx, _))| (self.model_names[*idx].clone(), rank + 1))
            .collect()
    }
}

#[pyfunction]
#[pyo3(signature = (
    model_predictions,
    model_names,
    time,
    event,
    eval_time=None
))]
pub fn benchmark_models(
    model_predictions: Vec<Vec<f64>>,
    model_names: Vec<String>,
    time: Vec<f64>,
    event: Vec<i32>,
    eval_time: Option<f64>,
) -> PyResult<BenchmarkResult> {
    let n_models = model_predictions.len();
    if n_models == 0 || model_names.len() != n_models {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "model_predictions and model_names must have the same non-zero length",
        ));
    }

    let n = time.len();
    if n == 0 || event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and event must have the same non-zero length",
        ));
    }

    let eval_time =
        eval_time.unwrap_or_else(|| time.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * 0.75);

    let results: Vec<(f64, f64)> = model_predictions
        .par_iter()
        .map(|predictions| {
            if predictions.len() != n {
                return (0.5, 1.0);
            }

            let mut concordant = 0.0;
            let mut comparable = 0.0;

            for i in 0..n {
                for j in (i + 1)..n {
                    if event[i] == 1 && time[i] < time[j] {
                        comparable += 1.0;
                        if predictions[i] > predictions[j] {
                            concordant += 1.0;
                        } else if (predictions[i] - predictions[j]).abs() < 1e-10 {
                            concordant += 0.5;
                        }
                    } else if event[j] == 1 && time[j] < time[i] {
                        comparable += 1.0;
                        if predictions[j] > predictions[i] {
                            concordant += 1.0;
                        } else if (predictions[i] - predictions[j]).abs() < 1e-10 {
                            concordant += 0.5;
                        }
                    }
                }
            }

            let c_index = if comparable > 0.0 {
                concordant / comparable
            } else {
                0.5
            };

            let brier: f64 = predictions
                .iter()
                .zip(time.iter())
                .zip(event.iter())
                .map(|((&p, &t), &e)| {
                    let observed = if t <= eval_time && e == 1 { 1.0 } else { 0.0 };
                    (p - observed).powi(2)
                })
                .sum::<f64>()
                / n as f64;

            (c_index, brier)
        })
        .collect();

    let c_indices: Vec<f64> = results.iter().map(|(c, _)| *c).collect();
    let brier_scores: Vec<f64> = results.iter().map(|(_, b)| *b).collect();
    let training_times_ms: Vec<f64> = vec![0.0; n_models];

    let best_idx = c_indices
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let best_model = model_names[best_idx].clone();

    Ok(BenchmarkResult {
        model_names,
        c_indices,
        brier_scores,
        training_times_ms,
        best_model,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct NestedCVResult {
    #[pyo3(get)]
    pub outer_scores: Vec<f64>,
    #[pyo3(get)]
    pub mean_score: f64,
    #[pyo3(get)]
    pub std_score: f64,
    #[pyo3(get)]
    pub best_params_per_fold: Vec<Vec<(String, f64)>>,
}

#[pymethods]
impl NestedCVResult {
    fn __repr__(&self) -> String {
        format!(
            "NestedCVResult(mean_score={:.4}, std_score={:.4})",
            self.mean_score, self.std_score
        )
    }

    fn confidence_interval(&self, _alpha: f64) -> (f64, f64) {
        let z = 1.96;
        let margin = z * self.std_score / (self.outer_scores.len() as f64).sqrt();
        (self.mean_score - margin, self.mean_score + margin)
    }
}

#[pyfunction]
#[pyo3(signature = (
    risk_scores,
    time,
    event,
    param_grid,
    outer_folds=5,
    inner_folds=3,
    seed=None
))]
pub fn nested_cross_validation(
    risk_scores: Vec<f64>,
    time: Vec<f64>,
    event: Vec<i32>,
    param_grid: Vec<(String, Vec<f64>)>,
    outer_folds: usize,
    inner_folds: usize,
    seed: Option<u64>,
) -> PyResult<NestedCVResult> {
    let n = risk_scores.len();
    if n == 0 || time.len() != n || event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All inputs must have the same non-zero length",
        ));
    }

    let mut rng = fastrand::Rng::new();
    if let Some(s) = seed {
        rng.seed(s);
    }

    let outer_fold_indices = create_cv_folds(n, outer_folds, &mut rng);

    let results: Vec<(f64, Vec<(String, f64)>)> = outer_fold_indices
        .iter()
        .enumerate()
        .map(|(outer_idx, test_fold)| {
            let train_idx: Vec<usize> = outer_fold_indices
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != outer_idx)
                .flat_map(|(_, f)| f.iter().copied())
                .collect();

            let train_risks: Vec<f64> = train_idx.iter().map(|&i| risk_scores[i]).collect();
            let train_time: Vec<f64> = train_idx.iter().map(|&i| time[i]).collect();
            let train_event: Vec<i32> = train_idx.iter().map(|&i| event[i]).collect();

            let inner_config = HyperparameterSearchConfig::new(
                SearchStrategy::Random,
                10,
                inner_folds,
                None,
                Some(rng.u64(..)),
            );

            let inner_result = hyperparameter_search(
                train_risks.clone(),
                train_time.clone(),
                train_event.clone(),
                param_grid.clone(),
                Some(inner_config),
            )
            .unwrap_or_else(|_| HyperparameterResult {
                best_params: vec![("default".to_string(), 1.0)],
                best_score: 0.5,
                all_scores: vec![],
                all_params: vec![],
                cv_scores: vec![],
            });

            let param_scale: f64 = inner_result.best_params.iter().map(|(_, v)| v).sum::<f64>()
                / inner_result.best_params.len().max(1) as f64;

            let scaled_risks: Vec<f64> = risk_scores.iter().map(|&r| r * param_scale).collect();

            let outer_score =
                compute_cv_c_index(&scaled_risks, &time, &event, &train_idx, test_fold);

            (outer_score, inner_result.best_params)
        })
        .collect();

    let outer_scores: Vec<f64> = results.iter().map(|(s, _)| *s).collect();
    let best_params_per_fold: Vec<Vec<(String, f64)>> =
        results.iter().map(|(_, p)| p.clone()).collect();

    let mean_score = outer_scores.iter().sum::<f64>() / outer_scores.len() as f64;
    let std_score = {
        let var: f64 = outer_scores
            .iter()
            .map(|&s| (s - mean_score).powi(2))
            .sum::<f64>()
            / outer_scores.len() as f64;
        var.sqrt()
    };

    Ok(NestedCVResult {
        outer_scores,
        mean_score,
        std_score,
        best_params_per_fold,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperparameter_search() {
        let risk = vec![0.9, 0.7, 0.5, 0.3, 0.1, 0.8, 0.6, 0.4, 0.2, 0.0];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let event = vec![1, 1, 1, 1, 1, 0, 1, 0, 1, 1];
        let param_grid = vec![("alpha".to_string(), vec![0.5, 1.0, 2.0])];

        let result = hyperparameter_search(risk, time, event, param_grid, None).unwrap();
        assert!(!result.all_scores.is_empty());
    }

    #[test]
    fn test_benchmark_models() {
        let preds = vec![vec![0.9, 0.7, 0.5, 0.3, 0.1], vec![0.8, 0.6, 0.4, 0.2, 0.0]];
        let names = vec!["model1".to_string(), "model2".to_string()];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let event = vec![1, 1, 1, 1, 1];

        let result = benchmark_models(preds, names, time, event, None).unwrap();
        assert_eq!(result.model_names.len(), 2);
    }

    #[test]
    fn test_nested_cv() {
        let risk: Vec<f64> = (0..20).map(|i| 1.0 - i as f64 / 20.0).collect();
        let time: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let event: Vec<i32> = vec![1; 20];
        let param_grid = vec![("alpha".to_string(), vec![0.5, 1.0, 2.0])];

        let result =
            nested_cross_validation(risk, time, event, param_grid, 3, 2, Some(42)).unwrap();
        assert_eq!(result.outer_scores.len(), 3);
    }
}
