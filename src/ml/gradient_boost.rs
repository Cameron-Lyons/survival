#![allow(
    unused_variables,
    unused_imports,
    unused_mut,
    unused_assignments,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::upper_case_acronyms,
    clippy::collapsible_else_if,
    clippy::collapsible_if
)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass]
pub enum GBSurvLoss {
    CoxPH,
    AFT,
    SquaredError,
    Huber,
}

#[pymethods]
impl GBSurvLoss {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "coxph" | "cox" | "cox_ph" => Ok(GBSurvLoss::CoxPH),
            "aft" => Ok(GBSurvLoss::AFT),
            "squared" | "squared_error" | "mse" => Ok(GBSurvLoss::SquaredError),
            "huber" => Ok(GBSurvLoss::Huber),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown loss function",
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct GradientBoostSurvivalConfig {
    #[pyo3(get, set)]
    pub n_estimators: usize,
    #[pyo3(get, set)]
    pub learning_rate: f64,
    #[pyo3(get, set)]
    pub max_depth: usize,
    #[pyo3(get, set)]
    pub min_samples_split: usize,
    #[pyo3(get, set)]
    pub min_samples_leaf: usize,
    #[pyo3(get, set)]
    pub subsample: f64,
    #[pyo3(get, set)]
    pub max_features: Option<usize>,
    #[pyo3(get, set)]
    pub loss: GBSurvLoss,
    #[pyo3(get, set)]
    pub dropout_rate: f64,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl GradientBoostSurvivalConfig {
    #[new]
    #[pyo3(signature = (
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=1.0,
        max_features=None,
        loss=GBSurvLoss::CoxPH,
        dropout_rate=0.0,
        seed=None
    ))]
    pub fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        min_samples_split: usize,
        min_samples_leaf: usize,
        subsample: f64,
        max_features: Option<usize>,
        loss: GBSurvLoss,
        dropout_rate: f64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if n_estimators == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "n_estimators must be positive",
            ));
        }
        if learning_rate <= 0.0 || learning_rate > 1.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "learning_rate must be in (0, 1]",
            ));
        }
        if subsample <= 0.0 || subsample > 1.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "subsample must be in (0, 1]",
            ));
        }

        Ok(GradientBoostSurvivalConfig {
            n_estimators,
            learning_rate,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            subsample,
            max_features,
            loss,
            dropout_rate,
            seed,
        })
    }
}

#[derive(Debug, Clone)]
struct RegressionTreeNode {
    split_var: Option<usize>,
    split_value: Option<f64>,
    left: Option<Box<RegressionTreeNode>>,
    right: Option<Box<RegressionTreeNode>>,
    prediction: f64,
    n_samples: usize,
}

impl RegressionTreeNode {
    fn new_leaf(prediction: f64, n_samples: usize) -> Self {
        RegressionTreeNode {
            split_var: None,
            split_value: None,
            left: None,
            right: None,
            prediction,
            n_samples,
        }
    }
}

fn compute_cox_gradient_hessian(
    time: &[f64],
    status: &[i32],
    predictions: &[f64],
    indices: &[usize],
) -> (Vec<f64>, Vec<f64>) {
    let n = indices.len();
    let mut gradient = vec![0.0; n];
    let mut hessian = vec![0.0; n];

    if n == 0 {
        return (gradient, hessian);
    }

    let mut sorted_indices = indices.to_vec();
    sorted_indices.sort_by(|&a, &b| {
        time[b]
            .partial_cmp(&time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut risk_sum = 0.0;
    let mut risk_sum_sq = 0.0;

    let max_idx = indices.iter().copied().max().unwrap_or(0) + 1;
    let mut idx_to_pos = vec![usize::MAX; max_idx];
    for (pos, &idx) in indices.iter().enumerate() {
        idx_to_pos[idx] = pos;
    }

    let mut cumulative_risk = vec![0.0; n];
    let mut cumulative_risk_sq = vec![0.0; n];

    let mut sorted_pos_map = vec![usize::MAX; max_idx];
    for (sorted_pos, &idx) in sorted_indices.iter().enumerate() {
        let exp_pred = predictions[idx].clamp(-700.0, 700.0).exp();
        risk_sum += exp_pred;
        risk_sum_sq += exp_pred * exp_pred;
        cumulative_risk[sorted_pos] = risk_sum;
        cumulative_risk_sq[sorted_pos] = risk_sum_sq;
        sorted_pos_map[idx] = sorted_pos;
    }

    for &idx in indices {
        let pos = idx_to_pos[idx];
        let sorted_pos = sorted_pos_map[idx];

        let exp_pred = predictions[idx].clamp(-700.0, 700.0).exp();
        let risk_at_time = cumulative_risk[sorted_pos];

        if status[idx] == 1 {
            gradient[pos] = 1.0 - exp_pred / risk_at_time.max(1e-10);
            hessian[pos] =
                exp_pred / risk_at_time.max(1e-10) * (1.0 - exp_pred / risk_at_time.max(1e-10));
        } else {
            gradient[pos] = -exp_pred / risk_at_time.max(1e-10);
            hessian[pos] =
                exp_pred / risk_at_time.max(1e-10) * (1.0 - exp_pred / risk_at_time.max(1e-10));
        }
    }

    (gradient, hessian)
}

fn find_best_split_regression(
    x: &[f64],
    n: usize,
    p: usize,
    gradient: &[f64],
    hessian: &[f64],
    indices: &[usize],
    max_features: usize,
    min_samples_leaf: usize,
    rng: &mut fastrand::Rng,
) -> Option<(usize, f64, Vec<usize>, Vec<usize>)> {
    if indices.len() < 2 * min_samples_leaf {
        return None;
    }

    let max_idx = indices.iter().copied().max().unwrap_or(0) + 1;
    let mut idx_to_pos = vec![usize::MAX; max_idx];
    for (pos, &idx) in indices.iter().enumerate() {
        idx_to_pos[idx] = pos;
    }

    let mut candidate_vars: Vec<usize> = (0..p).collect();
    for i in (1..candidate_vars.len()).rev() {
        let j = rng.usize(0..=i);
        candidate_vars.swap(i, j);
    }
    candidate_vars.truncate(max_features);

    let mut best_gain = f64::NEG_INFINITY;
    let mut best_split: Option<(usize, f64, Vec<usize>, Vec<usize>)> = None;

    let total_grad: f64 = indices.iter().map(|&i| gradient[idx_to_pos[i]]).sum();
    let total_hess: f64 = indices.iter().map(|&i| hessian[idx_to_pos[i]]).sum();

    for &var in &candidate_vars {
        let mut var_indices: Vec<(f64, usize)> =
            indices.iter().map(|&i| (x[i * p + var], i)).collect();
        var_indices.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut left_grad = 0.0;
        let mut left_hess = 0.0;

        for i in 0..(var_indices.len() - min_samples_leaf) {
            let (_, idx) = var_indices[i];
            let pos = idx_to_pos[idx];
            left_grad += gradient[pos];
            left_hess += hessian[pos];

            if i + 1 < min_samples_leaf {
                continue;
            }

            let right_grad = total_grad - left_grad;
            let right_hess = total_hess - left_hess;

            if left_hess < 1e-10 || right_hess < 1e-10 {
                continue;
            }

            let gain = 0.5
                * (left_grad * left_grad / left_hess + right_grad * right_grad / right_hess
                    - total_grad * total_grad / total_hess);

            if gain > best_gain {
                let split_value = (var_indices[i].0 + var_indices[i + 1].0) / 2.0;
                let left_idx: Vec<usize> = var_indices[..=i].iter().map(|(_, idx)| *idx).collect();
                let right_idx: Vec<usize> =
                    var_indices[i + 1..].iter().map(|(_, idx)| *idx).collect();

                best_gain = gain;
                best_split = Some((var, split_value, left_idx, right_idx));
            }
        }
    }

    best_split
}

fn build_regression_tree(
    x: &[f64],
    n: usize,
    p: usize,
    gradient: &[f64],
    hessian: &[f64],
    indices: &[usize],
    config: &GradientBoostSurvivalConfig,
    depth: usize,
    rng: &mut fastrand::Rng,
) -> RegressionTreeNode {
    let max_idx = indices.iter().copied().max().unwrap_or(0) + 1;
    let mut idx_to_pos = vec![usize::MAX; max_idx];
    for (pos, &idx) in indices.iter().enumerate() {
        idx_to_pos[idx] = pos;
    }

    let sum_grad: f64 = indices.iter().map(|&i| gradient[idx_to_pos[i]]).sum();
    let sum_hess: f64 = indices.iter().map(|&i| hessian[idx_to_pos[i]]).sum();
    let leaf_value = if sum_hess.abs() > 1e-10 {
        -sum_grad / sum_hess
    } else {
        0.0
    };

    if depth >= config.max_depth || indices.len() < config.min_samples_split {
        return RegressionTreeNode::new_leaf(leaf_value, indices.len());
    }

    let max_features = config.max_features.unwrap_or(p);

    let best_split = find_best_split_regression(
        x,
        n,
        p,
        gradient,
        hessian,
        indices,
        max_features,
        config.min_samples_leaf,
        rng,
    );

    match best_split {
        Some((split_var, split_value, left_idx, right_idx)) => {
            let left_child = build_regression_tree(
                x,
                n,
                p,
                gradient,
                hessian,
                &left_idx,
                config,
                depth + 1,
                rng,
            );
            let right_child = build_regression_tree(
                x,
                n,
                p,
                gradient,
                hessian,
                &right_idx,
                config,
                depth + 1,
                rng,
            );

            RegressionTreeNode {
                split_var: Some(split_var),
                split_value: Some(split_value),
                left: Some(Box::new(left_child)),
                right: Some(Box::new(right_child)),
                prediction: leaf_value,
                n_samples: indices.len(),
            }
        }
        None => RegressionTreeNode::new_leaf(leaf_value, indices.len()),
    }
}

fn predict_regression_tree(node: &RegressionTreeNode, x_row: &[f64]) -> f64 {
    match (&node.split_var, &node.split_value) {
        (Some(var), Some(val)) => {
            if x_row[*var] <= *val {
                if let Some(ref left) = node.left {
                    return predict_regression_tree(left, x_row);
                }
            } else {
                if let Some(ref right) = node.right {
                    return predict_regression_tree(right, x_row);
                }
            }
            node.prediction
        }
        _ => node.prediction,
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct GradientBoostSurvival {
    trees: Vec<RegressionTreeNode>,
    #[pyo3(get)]
    pub learning_rate: f64,
    #[pyo3(get)]
    pub baseline_hazard: Vec<f64>,
    #[pyo3(get)]
    pub unique_times: Vec<f64>,
    #[pyo3(get)]
    pub feature_importance: Vec<f64>,
    #[pyo3(get)]
    pub train_loss: Vec<f64>,
    n_vars: usize,
}

#[pymethods]
impl GradientBoostSurvival {
    #[staticmethod]
    #[pyo3(signature = (x, n_obs, n_vars, time, status, config))]
    pub fn fit(
        x: Vec<f64>,
        n_obs: usize,
        n_vars: usize,
        time: Vec<f64>,
        status: Vec<i32>,
        config: &GradientBoostSurvivalConfig,
    ) -> PyResult<Self> {
        if x.len() != n_obs * n_vars {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "x length must equal n_obs * n_vars",
            ));
        }
        if time.len() != n_obs || status.len() != n_obs {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "time and status must have length n_obs",
            ));
        }

        let mut unique_times: Vec<f64> = time.clone();
        unique_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        unique_times.dedup();

        let mut predictions = vec![0.0; n_obs];
        let mut trees = Vec::with_capacity(config.n_estimators);
        let mut train_loss = Vec::with_capacity(config.n_estimators);
        let mut feature_importance = vec![0.0; n_vars];

        let base_seed = config.seed.unwrap_or(42);

        for iter in 0..config.n_estimators {
            let mut rng = fastrand::Rng::with_seed(base_seed.wrapping_add(iter as u64));

            let sample_size = (n_obs as f64 * config.subsample).ceil() as usize;
            let indices: Vec<usize> = if config.subsample < 1.0 {
                (0..sample_size).map(|_| rng.usize(0..n_obs)).collect()
            } else {
                (0..n_obs).collect()
            };

            let (gradient, hessian) = match config.loss {
                GBSurvLoss::CoxPH => {
                    compute_cox_gradient_hessian(&time, &status, &predictions, &indices)
                }
                _ => compute_cox_gradient_hessian(&time, &status, &predictions, &indices),
            };

            let tree = build_regression_tree(
                &x, n_obs, n_vars, &gradient, &hessian, &indices, config, 0, &mut rng,
            );

            update_feature_importance(&tree, &mut feature_importance);

            for i in 0..n_obs {
                let x_row: Vec<f64> = (0..n_vars).map(|j| x[i * n_vars + j]).collect();
                predictions[i] += config.learning_rate * predict_regression_tree(&tree, &x_row);
            }

            let loss = compute_cox_loss(&time, &status, &predictions);
            train_loss.push(loss);

            trees.push(tree);
        }

        let total_importance: f64 = feature_importance.iter().sum();
        if total_importance > 0.0 {
            for imp in &mut feature_importance {
                *imp /= total_importance;
            }
        }

        let baseline_hazard = compute_baseline_hazard(&time, &status, &predictions, &unique_times);

        Ok(GradientBoostSurvival {
            trees,
            learning_rate: config.learning_rate,
            baseline_hazard,
            unique_times,
            feature_importance,
            train_loss,
            n_vars,
        })
    }

    #[pyo3(signature = (x_new, n_new))]
    pub fn predict_risk(&self, x_new: Vec<f64>, n_new: usize) -> PyResult<Vec<f64>> {
        if x_new.len() != n_new * self.n_vars {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "x_new dimensions don't match",
            ));
        }

        let predictions: Vec<f64> = (0..n_new)
            .into_par_iter()
            .map(|i| {
                let x_row: Vec<f64> = (0..self.n_vars)
                    .map(|j| x_new[i * self.n_vars + j])
                    .collect();
                let mut pred = 0.0;
                for tree in &self.trees {
                    pred += self.learning_rate * predict_regression_tree(tree, &x_row);
                }
                pred
            })
            .collect();

        Ok(predictions)
    }

    #[pyo3(signature = (x_new, n_new))]
    pub fn predict_survival(&self, x_new: Vec<f64>, n_new: usize) -> PyResult<Vec<Vec<f64>>> {
        let risk_scores = self.predict_risk(x_new, n_new)?;

        let survival: Vec<Vec<f64>> = risk_scores
            .par_iter()
            .map(|&risk| {
                self.baseline_hazard
                    .iter()
                    .map(|&h| (-h * risk.exp()).exp())
                    .collect()
            })
            .collect();

        Ok(survival)
    }

    #[pyo3(signature = (x_new, n_new))]
    pub fn predict_cumulative_hazard(
        &self,
        x_new: Vec<f64>,
        n_new: usize,
    ) -> PyResult<Vec<Vec<f64>>> {
        let risk_scores = self.predict_risk(x_new, n_new)?;

        let cumhaz: Vec<Vec<f64>> = risk_scores
            .par_iter()
            .map(|&risk| {
                self.baseline_hazard
                    .iter()
                    .map(|&h| h * risk.exp())
                    .collect()
            })
            .collect();

        Ok(cumhaz)
    }

    #[getter]
    pub fn get_n_estimators(&self) -> usize {
        self.trees.len()
    }
}

fn update_feature_importance(node: &RegressionTreeNode, importance: &mut [f64]) {
    if let Some(var) = node.split_var {
        if var < importance.len() {
            importance[var] += node.n_samples as f64;
        }
    }

    if let Some(ref left) = node.left {
        update_feature_importance(left, importance);
    }
    if let Some(ref right) = node.right {
        update_feature_importance(right, importance);
    }
}

fn compute_cox_loss(time: &[f64], status: &[i32], predictions: &[f64]) -> f64 {
    let n = time.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[b]
            .partial_cmp(&time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut loglik = 0.0;
    let mut risk_sum = 0.0;

    for &i in &indices {
        let exp_pred = predictions[i].clamp(-700.0, 700.0).exp();
        risk_sum += exp_pred;

        if status[i] == 1 {
            loglik += predictions[i] - risk_sum.ln();
        }
    }

    -loglik
}

fn compute_baseline_hazard(
    time: &[f64],
    status: &[i32],
    predictions: &[f64],
    unique_times: &[f64],
) -> Vec<f64> {
    let n = time.len();

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let exp_preds: Vec<f64> = predictions
        .iter()
        .map(|&p| p.clamp(-700.0, 700.0).exp())
        .collect();

    let mut risk_sum = exp_preds.iter().sum::<f64>();
    let mut baseline_hazard = Vec::with_capacity(unique_times.len());
    let mut cum_haz = 0.0;

    let mut time_idx = 0;

    for &ut in unique_times {
        while time_idx < n && time[indices[time_idx]] <= ut {
            let idx = indices[time_idx];
            if status[idx] == 1 && risk_sum > 0.0 {
                cum_haz += 1.0 / risk_sum;
            }
            risk_sum -= exp_preds[idx];
            time_idx += 1;
        }
        baseline_hazard.push(cum_haz);
    }

    baseline_hazard
}

#[pyfunction]
#[pyo3(signature = (x, n_obs, n_vars, time, status, config=None))]
pub fn gradient_boost_survival(
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    time: Vec<f64>,
    status: Vec<i32>,
    config: Option<&GradientBoostSurvivalConfig>,
) -> PyResult<GradientBoostSurvival> {
    let cfg = config.cloned().unwrap_or_else(|| {
        GradientBoostSurvivalConfig::new(
            100,
            0.1,
            3,
            10,
            5,
            1.0,
            None,
            GBSurvLoss::CoxPH,
            0.0,
            None,
        )
        .unwrap()
    });

    GradientBoostSurvival::fit(x, n_obs, n_vars, time, status, &cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = GradientBoostSurvivalConfig::new(
            50,
            0.1,
            3,
            5,
            2,
            0.8,
            None,
            GBSurvLoss::CoxPH,
            0.0,
            None,
        )
        .unwrap();
        assert_eq!(config.n_estimators, 50);
        assert_eq!(config.max_depth, 3);
    }

    #[test]
    fn test_gradient_boost_basic() {
        let x = vec![1.0, 0.5, 0.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 1.5, 0.5];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let status = vec![1, 1, 0, 1, 0, 1];

        let config = GradientBoostSurvivalConfig::new(
            5,
            0.1,
            2,
            2,
            1,
            1.0,
            None,
            GBSurvLoss::CoxPH,
            0.0,
            Some(42),
        )
        .unwrap();

        let model = GradientBoostSurvival::fit(x, 6, 2, time, status, &config).unwrap();
        assert_eq!(model.get_n_estimators(), 5);
    }
}
