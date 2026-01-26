#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct CausalForestConfig {
    #[pyo3(get, set)]
    pub n_trees: usize,
    #[pyo3(get, set)]
    pub max_depth: usize,
    #[pyo3(get, set)]
    pub min_samples_leaf: usize,
    #[pyo3(get, set)]
    pub min_samples_split: usize,
    #[pyo3(get, set)]
    pub max_features: Option<usize>,
    #[pyo3(get, set)]
    pub honesty: bool,
    #[pyo3(get, set)]
    pub honesty_fraction: f64,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl CausalForestConfig {
    #[new]
    #[pyo3(signature = (
        n_trees=100,
        max_depth=10,
        min_samples_leaf=5,
        min_samples_split=10,
        max_features=None,
        honesty=true,
        honesty_fraction=0.5,
        seed=None
    ))]
    pub fn new(
        n_trees: usize,
        max_depth: usize,
        min_samples_leaf: usize,
        min_samples_split: usize,
        max_features: Option<usize>,
        honesty: bool,
        honesty_fraction: f64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if n_trees == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "n_trees must be greater than 0",
            ));
        }
        if honesty_fraction <= 0.0 || honesty_fraction >= 1.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "honesty_fraction must be between 0 and 1",
            ));
        }
        Ok(Self {
            n_trees,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            max_features,
            honesty,
            honesty_fraction,
            seed,
        })
    }
}

#[derive(Debug, Clone)]
struct CausalTreeNode {
    split_feature: Option<usize>,
    split_value: Option<f64>,
    treatment_effect: f64,
    n_samples: usize,
    left: Option<Box<CausalTreeNode>>,
    right: Option<Box<CausalTreeNode>>,
}

impl CausalTreeNode {
    fn new_leaf(effect: f64, n_samples: usize) -> Self {
        Self {
            split_feature: None,
            split_value: None,
            treatment_effect: effect,
            n_samples,
            left: None,
            right: None,
        }
    }
}

fn compute_treatment_effect(outcomes: &[f64], treatment: &[i32], indices: &[usize]) -> f64 {
    let mut sum_treated = 0.0;
    let mut n_treated = 0;
    let mut sum_control = 0.0;
    let mut n_control = 0;

    for &i in indices {
        if treatment[i] == 1 {
            sum_treated += outcomes[i];
            n_treated += 1;
        } else {
            sum_control += outcomes[i];
            n_control += 1;
        }
    }

    let mean_treated = if n_treated > 0 {
        sum_treated / n_treated as f64
    } else {
        0.0
    };
    let mean_control = if n_control > 0 {
        sum_control / n_control as f64
    } else {
        0.0
    };

    mean_treated - mean_control
}

fn compute_causal_split_criterion(
    covariates: &[Vec<f64>],
    outcomes: &[f64],
    treatment: &[i32],
    indices: &[usize],
    feature: usize,
    split_value: f64,
) -> f64 {
    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();

    for &i in indices {
        if covariates[i][feature] <= split_value {
            left_indices.push(i);
        } else {
            right_indices.push(i);
        }
    }

    if left_indices.is_empty() || right_indices.is_empty() {
        return f64::NEG_INFINITY;
    }

    let left_effect = compute_treatment_effect(outcomes, treatment, &left_indices);
    let right_effect = compute_treatment_effect(outcomes, treatment, &right_indices);

    let n_left = left_indices.len() as f64;
    let n_right = right_indices.len() as f64;
    let n_total = indices.len() as f64;

    (n_left * n_right / (n_total * n_total)) * (left_effect - right_effect).powi(2)
}

fn find_best_split(
    covariates: &[Vec<f64>],
    outcomes: &[f64],
    treatment: &[i32],
    indices: &[usize],
    feature_indices: &[usize],
    min_samples_leaf: usize,
) -> Option<(usize, f64, f64)> {
    let mut best_gain = f64::NEG_INFINITY;
    let mut best_feature = 0;
    let mut best_value = 0.0;

    for &feature in feature_indices {
        let mut values: Vec<f64> = indices.iter().map(|&i| covariates[i][feature]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        values.dedup();

        for i in 0..values.len().saturating_sub(1) {
            let split_value = (values[i] + values[i + 1]) / 2.0;

            let n_left = indices
                .iter()
                .filter(|&&idx| covariates[idx][feature] <= split_value)
                .count();
            let n_right = indices.len() - n_left;

            if n_left < min_samples_leaf || n_right < min_samples_leaf {
                continue;
            }

            let gain = compute_causal_split_criterion(
                covariates,
                outcomes,
                treatment,
                indices,
                feature,
                split_value,
            );

            if gain > best_gain {
                best_gain = gain;
                best_feature = feature;
                best_value = split_value;
            }
        }
    }

    if best_gain > f64::NEG_INFINITY {
        Some((best_feature, best_value, best_gain))
    } else {
        None
    }
}

fn build_causal_tree(
    covariates: &[Vec<f64>],
    outcomes: &[f64],
    treatment: &[i32],
    indices: &[usize],
    feature_indices: &[usize],
    depth: usize,
    config: &CausalForestConfig,
) -> CausalTreeNode {
    let effect = compute_treatment_effect(outcomes, treatment, indices);

    if depth >= config.max_depth
        || indices.len() < config.min_samples_split
        || indices.len() < 2 * config.min_samples_leaf
    {
        return CausalTreeNode::new_leaf(effect, indices.len());
    }

    let split = find_best_split(
        covariates,
        outcomes,
        treatment,
        indices,
        feature_indices,
        config.min_samples_leaf,
    );

    match split {
        Some((feature, value, _)) => {
            let (left_indices, right_indices): (Vec<_>, Vec<_>) = indices
                .iter()
                .partition(|&&i| covariates[i][feature] <= value);

            if left_indices.is_empty() || right_indices.is_empty() {
                return CausalTreeNode::new_leaf(effect, indices.len());
            }

            let left = build_causal_tree(
                covariates,
                outcomes,
                treatment,
                &left_indices,
                feature_indices,
                depth + 1,
                config,
            );
            let right = build_causal_tree(
                covariates,
                outcomes,
                treatment,
                &right_indices,
                feature_indices,
                depth + 1,
                config,
            );

            CausalTreeNode {
                split_feature: Some(feature),
                split_value: Some(value),
                treatment_effect: effect,
                n_samples: indices.len(),
                left: Some(Box::new(left)),
                right: Some(Box::new(right)),
            }
        }
        None => CausalTreeNode::new_leaf(effect, indices.len()),
    }
}

fn predict_tree(node: &CausalTreeNode, x: &[f64]) -> f64 {
    match (
        &node.left,
        &node.right,
        node.split_feature,
        node.split_value,
    ) {
        (Some(left), Some(right), Some(feature), Some(value)) => {
            if x[feature] <= value {
                predict_tree(left, x)
            } else {
                predict_tree(right, x)
            }
        }
        _ => node.treatment_effect,
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct CausalForestSurvival {
    trees: Vec<CausalTreeNode>,
    #[allow(dead_code)]
    config: CausalForestConfig,
    n_features: usize,
}

#[pymethods]
impl CausalForestSurvival {
    fn predict_cate(&self, covariates: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let predictions: Vec<f64> = covariates
            .par_iter()
            .map(|x| {
                let sum: f64 = self.trees.iter().map(|tree| predict_tree(tree, x)).sum();
                sum / self.trees.len() as f64
            })
            .collect();

        Ok(predictions)
    }

    fn predict_variance(&self, covariates: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let variances: Vec<f64> = covariates
            .par_iter()
            .map(|x| {
                let predictions: Vec<f64> = self
                    .trees
                    .iter()
                    .map(|tree| predict_tree(tree, x))
                    .collect();
                let mean = predictions.iter().sum::<f64>() / predictions.len() as f64;
                predictions.iter().map(|&p| (p - mean).powi(2)).sum::<f64>()
                    / predictions.len() as f64
            })
            .collect();

        Ok(variances)
    }

    fn feature_importance(&self) -> Vec<f64> {
        let mut importance = vec![0.0; self.n_features];
        let mut counts = vec![0usize; self.n_features];

        fn traverse(node: &CausalTreeNode, importance: &mut [f64], counts: &mut [usize]) {
            if let Some(feature) = node.split_feature {
                importance[feature] += node.n_samples as f64;
                counts[feature] += 1;

                if let Some(ref left) = node.left {
                    traverse(left, importance, counts);
                }
                if let Some(ref right) = node.right {
                    traverse(right, importance, counts);
                }
            }
        }

        for tree in &self.trees {
            traverse(tree, &mut importance, &mut counts);
        }

        let total: f64 = importance.iter().sum();
        if total > 0.0 {
            for imp in &mut importance {
                *imp /= total;
            }
        }

        importance
    }

    fn __repr__(&self) -> String {
        format!(
            "CausalForestSurvival(n_trees={}, n_features={})",
            self.trees.len(),
            self.n_features
        )
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct CausalForestResult {
    #[pyo3(get)]
    pub cate_estimates: Vec<f64>,
    #[pyo3(get)]
    pub cate_se: Vec<f64>,
    #[pyo3(get)]
    pub feature_importance: Vec<f64>,
    #[pyo3(get)]
    pub ate: f64,
    #[pyo3(get)]
    pub ate_se: f64,
}

#[pymethods]
impl CausalForestResult {
    fn __repr__(&self) -> String {
        format!(
            "CausalForestResult(ATE={:.4}, n_obs={})",
            self.ate,
            self.cate_estimates.len()
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    covariates,
    treatment,
    time,
    event,
    time_horizon,
    config=None
))]
pub fn causal_forest_survival(
    covariates: Vec<Vec<f64>>,
    treatment: Vec<i32>,
    time: Vec<f64>,
    event: Vec<i32>,
    time_horizon: f64,
    config: Option<CausalForestConfig>,
) -> PyResult<(CausalForestSurvival, CausalForestResult)> {
    let config = config
        .unwrap_or_else(|| CausalForestConfig::new(100, 10, 5, 10, None, true, 0.5, None).unwrap());

    let n = covariates.len();
    if n == 0 || treatment.len() != n || time.len() != n || event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All inputs must have the same non-zero length",
        ));
    }

    let n_features = covariates[0].len();
    let max_features = config
        .max_features
        .unwrap_or((n_features as f64).sqrt() as usize);

    let outcomes: Vec<f64> = time
        .iter()
        .zip(event.iter())
        .map(|(&t, &e)| {
            if t <= time_horizon && e == 1 {
                0.0
            } else {
                1.0
            }
        })
        .collect();

    let mut rng = fastrand::Rng::new();
    if let Some(seed) = config.seed {
        rng.seed(seed);
    }

    let trees: Vec<CausalTreeNode> = (0..config.n_trees)
        .map(|_| {
            let mut sample_indices: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = rng.usize(0..=i);
                sample_indices.swap(i, j);
            }
            let sample_size = (n as f64 * 0.632) as usize;
            let indices: Vec<usize> = sample_indices.into_iter().take(sample_size).collect();

            let mut feature_indices: Vec<usize> = (0..n_features).collect();
            for i in (1..n_features).rev() {
                let j = rng.usize(0..=i);
                feature_indices.swap(i, j);
            }
            let feature_indices: Vec<usize> =
                feature_indices.into_iter().take(max_features).collect();

            build_causal_tree(
                &covariates,
                &outcomes,
                &treatment,
                &indices,
                &feature_indices,
                0,
                &config,
            )
        })
        .collect();

    let forest = CausalForestSurvival {
        trees,
        config: config.clone(),
        n_features,
    };

    let cate_estimates = forest.predict_cate(covariates.clone())?;
    let variances = forest.predict_variance(covariates)?;
    let cate_se: Vec<f64> = variances.iter().map(|&v| v.sqrt()).collect();

    let feature_importance = forest.feature_importance();

    let ate = cate_estimates.iter().sum::<f64>() / cate_estimates.len() as f64;
    let ate_var = cate_estimates
        .iter()
        .map(|&c| (c - ate).powi(2))
        .sum::<f64>()
        / (cate_estimates.len() * (cate_estimates.len() - 1)) as f64;
    let ate_se = ate_var.sqrt();

    let result = CausalForestResult {
        cate_estimates,
        cate_se,
        feature_importance,
        ate,
        ate_se,
    };

    Ok((forest, result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let result = CausalForestConfig::new(0, 10, 5, 10, None, true, 0.5, None);
        assert!(result.is_err());

        let result = CausalForestConfig::new(100, 10, 5, 10, None, true, 1.5, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_treatment_effect() {
        let outcomes = vec![1.0, 2.0, 3.0, 4.0];
        let treatment = vec![0, 0, 1, 1];
        let indices = vec![0, 1, 2, 3];

        let effect = compute_treatment_effect(&outcomes, &treatment, &indices);
        assert!((effect - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_config_honesty_fraction_boundary() {
        let result = CausalForestConfig::new(10, 5, 2, 4, None, true, 0.0, None);
        assert!(result.is_err());

        let result = CausalForestConfig::new(10, 5, 2, 4, None, true, 1.0, None);
        assert!(result.is_err());

        let result = CausalForestConfig::new(10, 5, 2, 4, None, true, 0.5, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_treatment_effect_all_treated() {
        let outcomes = vec![1.0, 2.0, 3.0];
        let treatment = vec![1, 1, 1];
        let indices = vec![0, 1, 2];

        let effect = compute_treatment_effect(&outcomes, &treatment, &indices);
        assert!((effect - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_treatment_effect_all_control() {
        let outcomes = vec![1.0, 2.0, 3.0];
        let treatment = vec![0, 0, 0];
        let indices = vec![0, 1, 2];

        let effect = compute_treatment_effect(&outcomes, &treatment, &indices);
        assert!((effect - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_causal_forest_survival_basic() {
        let covariates = vec![
            vec![1.0, 0.5],
            vec![0.0, 1.0],
            vec![1.5, 0.3],
            vec![0.2, 0.8],
            vec![1.1, 0.6],
            vec![0.3, 1.2],
            vec![0.9, 0.4],
            vec![0.5, 0.9],
            vec![1.3, 0.7],
            vec![0.1, 1.1],
            vec![0.7, 0.2],
            vec![0.4, 1.3],
            vec![1.2, 0.8],
            vec![0.6, 0.5],
            vec![0.8, 1.0],
            vec![1.4, 0.1],
            vec![0.3, 0.6],
            vec![1.0, 1.2],
            vec![0.5, 0.3],
            vec![0.2, 0.7],
        ];
        let treatment = vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let time = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5,
            8.5, 9.5, 10.5,
        ];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0];

        let config = CausalForestConfig::new(5, 3, 2, 4, None, false, 0.5, Some(42)).unwrap();
        let (_forest, result) =
            causal_forest_survival(covariates, treatment, time, event, 5.0, Some(config)).unwrap();

        assert_eq!(result.cate_estimates.len(), 20);
        assert_eq!(result.cate_se.len(), 20);
        assert_eq!(result.feature_importance.len(), 2);
        assert!(result.ate.is_finite());
        assert!(result.ate_se >= 0.0);

        let importance_sum: f64 = result.feature_importance.iter().sum();
        if importance_sum > 0.0 {
            assert!((importance_sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_causal_forest_empty_input() {
        let result = causal_forest_survival(vec![], vec![], vec![], vec![], 5.0, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_causal_forest_dimension_mismatch() {
        let covariates = vec![vec![1.0], vec![2.0]];
        let treatment = vec![1, 0, 1];
        let time = vec![1.0, 2.0];
        let event = vec![1, 0];
        let result = causal_forest_survival(covariates, treatment, time, event, 5.0, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_causal_forest_predict_cate_and_variance() {
        let covariates = vec![
            vec![1.0, 0.5],
            vec![0.0, 1.0],
            vec![1.5, 0.3],
            vec![0.2, 0.8],
            vec![1.1, 0.6],
            vec![0.3, 1.2],
            vec![0.9, 0.4],
            vec![0.5, 0.9],
            vec![1.3, 0.7],
            vec![0.1, 1.1],
            vec![0.7, 0.2],
            vec![0.4, 1.3],
            vec![1.2, 0.8],
            vec![0.6, 0.5],
            vec![0.8, 1.0],
            vec![1.4, 0.1],
            vec![0.3, 0.6],
            vec![1.0, 1.2],
            vec![0.5, 0.3],
            vec![0.2, 0.7],
        ];
        let treatment = vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let time = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5,
            8.5, 9.5, 10.5,
        ];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0];

        let config = CausalForestConfig::new(5, 3, 2, 4, None, false, 0.5, Some(42)).unwrap();
        let (forest, _) = causal_forest_survival(
            covariates.clone(),
            treatment,
            time,
            event,
            5.0,
            Some(config),
        )
        .unwrap();

        let new_data = vec![vec![0.5, 0.5], vec![1.0, 1.0]];
        let preds = forest.predict_cate(new_data.clone()).unwrap();
        assert_eq!(preds.len(), 2);
        assert!(preds.iter().all(|p| p.is_finite()));

        let vars = forest.predict_variance(new_data).unwrap();
        assert_eq!(vars.len(), 2);
        assert!(vars.iter().all(|&v| v >= 0.0));
    }
}
