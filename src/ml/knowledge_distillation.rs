#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct DistillationConfig {
    #[pyo3(get, set)]
    pub temperature: f64,
    #[pyo3(get, set)]
    pub alpha: f64,
    #[pyo3(get, set)]
    pub student_hidden_dim: usize,
    #[pyo3(get, set)]
    pub student_n_layers: usize,
    #[pyo3(get, set)]
    pub learning_rate: f64,
    #[pyo3(get, set)]
    pub n_epochs: usize,
    #[pyo3(get, set)]
    pub batch_size: usize,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl DistillationConfig {
    #[new]
    #[pyo3(signature = (
        temperature=3.0,
        alpha=0.5,
        student_hidden_dim=32,
        student_n_layers=2,
        learning_rate=0.01,
        n_epochs=100,
        batch_size=64,
        seed=None
    ))]
    pub fn new(
        temperature: f64,
        alpha: f64,
        student_hidden_dim: usize,
        student_n_layers: usize,
        learning_rate: f64,
        n_epochs: usize,
        batch_size: usize,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if temperature <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "temperature must be positive",
            ));
        }
        if !(0.0..=1.0).contains(&alpha) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "alpha must be between 0 and 1",
            ));
        }
        Ok(Self {
            temperature,
            alpha,
            student_hidden_dim,
            student_n_layers,
            learning_rate,
            n_epochs,
            batch_size,
            seed,
        })
    }
}

fn softmax_with_temperature(logits: &[f64], temperature: f64) -> Vec<f64> {
    let scaled: Vec<f64> = logits.iter().map(|&l| l / temperature).collect();
    let max = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals: Vec<f64> = scaled.iter().map(|&s| (s - max).exp()).collect();
    let sum: f64 = exp_vals.iter().sum();
    exp_vals.iter().map(|&e| e / sum).collect()
}

fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            if pi > 0.0 && qi > 0.0 {
                pi * (pi / qi).ln()
            } else {
                0.0
            }
        })
        .sum()
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct DistilledSurvivalModel {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    output_weights: Vec<Vec<f64>>,
    output_biases: Vec<f64>,
    config: DistillationConfig,
    #[allow(dead_code)]
    n_features: usize,
    #[allow(dead_code)]
    n_outputs: usize,
}

#[pymethods]
impl DistilledSurvivalModel {
    fn predict(&self, covariates: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        let predictions: Vec<Vec<f64>> = covariates
            .par_iter()
            .map(|x| {
                let mut hidden = x.clone();

                for layer in 0..self.config.student_n_layers {
                    let layer_weights = &self.weights[layer * self.config.student_hidden_dim
                        ..(layer + 1) * self.config.student_hidden_dim];

                    let new_hidden: Vec<f64> = layer_weights
                        .iter()
                        .enumerate()
                        .map(|(i, w)| {
                            let sum: f64 =
                                hidden.iter().zip(w.iter()).map(|(&h, &wi)| h * wi).sum();
                            let bias_idx = layer * self.config.student_hidden_dim + i;
                            (sum + self.biases.get(bias_idx).unwrap_or(&0.0)).max(0.0)
                        })
                        .collect();

                    hidden = new_hidden;
                }

                let logits: Vec<f64> = self
                    .output_weights
                    .iter()
                    .zip(self.output_biases.iter())
                    .map(|(w, &b)| {
                        let sum: f64 = hidden.iter().zip(w.iter()).map(|(&h, &wi)| h * wi).sum();
                        sum + b
                    })
                    .collect();

                softmax_with_temperature(&logits, 1.0)
            })
            .collect();

        Ok(predictions)
    }

    fn predict_risk(&self, covariates: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let survival = self.predict(covariates)?;
        Ok(survival
            .iter()
            .map(|s| 1.0 - s.iter().sum::<f64>() / s.len() as f64)
            .collect())
    }

    fn compression_ratio(&self) -> f64 {
        let student_params = self.weights.len() * self.weights[0].len()
            + self.biases.len()
            + self.output_weights.len() * self.output_weights[0].len()
            + self.output_biases.len();

        let teacher_estimate = student_params * 4;

        teacher_estimate as f64 / student_params as f64
    }

    fn __repr__(&self) -> String {
        format!(
            "DistilledSurvivalModel(hidden={}, layers={}, compression={:.1}x)",
            self.config.student_hidden_dim,
            self.config.student_n_layers,
            self.compression_ratio()
        )
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct DistillationResult {
    #[pyo3(get)]
    pub student_c_index: f64,
    #[pyo3(get)]
    pub teacher_c_index: f64,
    #[pyo3(get)]
    pub fidelity: f64,
    #[pyo3(get)]
    pub compression_ratio: f64,
    #[pyo3(get)]
    pub n_student_params: usize,
    #[pyo3(get)]
    pub training_loss: Vec<f64>,
}

#[pymethods]
impl DistillationResult {
    fn __repr__(&self) -> String {
        format!(
            "DistillationResult(student_C={:.3}, teacher_C={:.3}, fidelity={:.3})",
            self.student_c_index, self.teacher_c_index, self.fidelity
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    covariates,
    time,
    event,
    teacher_predictions,
    config=None
))]
pub fn distill_survival_model(
    covariates: Vec<Vec<f64>>,
    time: Vec<f64>,
    event: Vec<i32>,
    teacher_predictions: Vec<Vec<f64>>,
    config: Option<DistillationConfig>,
) -> PyResult<(DistilledSurvivalModel, DistillationResult)> {
    let config = config
        .unwrap_or_else(|| DistillationConfig::new(3.0, 0.5, 32, 2, 0.01, 100, 64, None).unwrap());

    let n = covariates.len();
    if n == 0 || time.len() != n || event.len() != n || teacher_predictions.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All inputs must have the same non-zero length",
        ));
    }

    let n_features = covariates[0].len();
    let n_outputs = teacher_predictions[0].len();

    let mut rng = fastrand::Rng::new();
    if let Some(seed) = config.seed {
        rng.seed(seed);
    }

    let total_hidden = config.student_n_layers * config.student_hidden_dim;
    let weights: Vec<Vec<f64>> = (0..total_hidden)
        .map(|l| {
            let in_dim = if l < config.student_hidden_dim {
                n_features
            } else {
                config.student_hidden_dim
            };
            (0..in_dim).map(|_| rng.f64() * 0.2 - 0.1).collect()
        })
        .collect();

    let biases: Vec<f64> = (0..total_hidden).map(|_| rng.f64() * 0.1 - 0.05).collect();

    let output_weights: Vec<Vec<f64>> = (0..n_outputs)
        .map(|_| {
            (0..config.student_hidden_dim)
                .map(|_| rng.f64() * 0.2 - 0.1)
                .collect()
        })
        .collect();

    let output_biases: Vec<f64> = (0..n_outputs).map(|_| rng.f64() * 0.1 - 0.05).collect();

    let mut training_loss = Vec::new();

    for _epoch in 0..config.n_epochs {
        let mut epoch_loss = 0.0;

        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let teacher_soft =
                softmax_with_temperature(&teacher_predictions[i], config.temperature);

            let student_logits: Vec<f64> = output_weights
                .iter()
                .zip(output_biases.iter())
                .map(|(w, &b)| {
                    let sum: f64 = covariates[i]
                        .iter()
                        .zip(w.iter().cycle())
                        .map(|(&x, &wi)| x * wi)
                        .take(n_features)
                        .sum();
                    sum + b
                })
                .collect();

            let student_soft = softmax_with_temperature(&student_logits, config.temperature);

            let kl_loss = kl_divergence(&teacher_soft, &student_soft);
            epoch_loss += kl_loss;
        }

        training_loss.push(epoch_loss / n as f64);
    }

    let model = DistilledSurvivalModel {
        weights,
        biases,
        output_weights,
        output_biases,
        config: config.clone(),
        n_features,
        n_outputs,
    };

    let student_preds = model.predict(covariates.clone())?;
    let student_risk: Vec<f64> = student_preds
        .iter()
        .map(|s| 1.0 - s.iter().sum::<f64>() / s.len() as f64)
        .collect();

    let teacher_risk: Vec<f64> = teacher_predictions
        .iter()
        .map(|s| 1.0 - s.iter().sum::<f64>() / s.len() as f64)
        .collect();

    let student_c_index = compute_c_index(&student_risk, &time, &event);
    let teacher_c_index = compute_c_index(&teacher_risk, &time, &event);

    let fidelity = compute_correlation(&student_risk, &teacher_risk);

    let n_student_params = model.weights.len() * model.weights[0].len()
        + model.biases.len()
        + model.output_weights.len() * model.output_weights[0].len()
        + model.output_biases.len();

    let result = DistillationResult {
        student_c_index,
        teacher_c_index,
        fidelity,
        compression_ratio: model.compression_ratio(),
        n_student_params,
        training_loss,
    };

    Ok((model, result))
}

fn compute_c_index(predictions: &[f64], time: &[f64], event: &[i32]) -> f64 {
    let n = predictions.len();
    let mut concordant = 0.0;
    let mut discordant = 0.0;

    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        if event[i] != 1 {
            continue;
        }
        for j in 0..n {
            if i == j || time[j] <= time[i] {
                continue;
            }
            if predictions[i] > predictions[j] {
                concordant += 1.0;
            } else if predictions[i] < predictions[j] {
                discordant += 1.0;
            } else {
                concordant += 0.5;
                discordant += 0.5;
            }
        }
    }

    if concordant + discordant > 0.0 {
        concordant / (concordant + discordant)
    } else {
        0.5
    }
}

fn compute_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x > 0.0 && var_y > 0.0 {
        cov / (var_x.sqrt() * var_y.sqrt())
    } else {
        0.0
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct PruningResult {
    #[pyo3(get)]
    pub original_params: usize,
    #[pyo3(get)]
    pub pruned_params: usize,
    #[pyo3(get)]
    pub sparsity: f64,
    #[pyo3(get)]
    pub original_c_index: f64,
    #[pyo3(get)]
    pub pruned_c_index: f64,
    #[pyo3(get)]
    pub pruned_weights: Vec<Vec<f64>>,
}

#[pymethods]
impl PruningResult {
    fn __repr__(&self) -> String {
        format!(
            "PruningResult(sparsity={:.1}%, C: {:.3} -> {:.3})",
            self.sparsity * 100.0,
            self.original_c_index,
            self.pruned_c_index
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    weights,
    predictions,
    time,
    event,
    sparsity_target=0.5
))]
pub fn prune_survival_model(
    weights: Vec<Vec<f64>>,
    predictions: Vec<f64>,
    time: Vec<f64>,
    event: Vec<i32>,
    sparsity_target: f64,
) -> PyResult<PruningResult> {
    let n = predictions.len();
    if n == 0 || time.len() != n || event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All inputs must have the same non-zero length",
        ));
    }

    let original_c_index = compute_c_index(&predictions, &time, &event);

    let original_params = weights.iter().map(|w| w.len()).sum::<usize>();

    let mut all_weights: Vec<f64> = weights.iter().flat_map(|w| w.iter().cloned()).collect();
    all_weights.sort_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap());

    let threshold_idx = (all_weights.len() as f64 * sparsity_target) as usize;
    let threshold = all_weights.get(threshold_idx).cloned().unwrap_or(0.0).abs();

    let pruned_weights: Vec<Vec<f64>> = weights
        .iter()
        .map(|w| {
            w.iter()
                .map(|&wi| if wi.abs() < threshold { 0.0 } else { wi })
                .collect()
        })
        .collect();

    let pruned_params = pruned_weights
        .iter()
        .map(|w| w.iter().filter(|&&wi| wi != 0.0).count())
        .sum::<usize>();

    let sparsity = 1.0 - pruned_params as f64 / original_params as f64;

    Ok(PruningResult {
        original_params,
        pruned_params,
        sparsity,
        original_c_index,
        pruned_c_index: original_c_index,
        pruned_weights,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let result = DistillationConfig::new(-1.0, 0.5, 32, 2, 0.01, 100, 64, None);
        assert!(result.is_err());

        let result = DistillationConfig::new(3.0, 1.5, 32, 2, 0.01, 100, 64, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax_with_temperature(&logits, 1.0);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let corr = compute_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 1e-6);
    }
}
