#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(eq, eq_int)]
pub enum TransferStrategy {
    FineTune,
    FeatureExtraction,
    DomainAdaptation,
}

#[pymethods]
impl TransferStrategy {
    fn __repr__(&self) -> String {
        match self {
            TransferStrategy::FineTune => "TransferStrategy.FineTune".to_string(),
            TransferStrategy::FeatureExtraction => "TransferStrategy.FeatureExtraction".to_string(),
            TransferStrategy::DomainAdaptation => "TransferStrategy.DomainAdaptation".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct TransferLearningConfig {
    #[pyo3(get, set)]
    pub strategy: TransferStrategy,
    #[pyo3(get, set)]
    pub freeze_layers: usize,
    #[pyo3(get, set)]
    pub learning_rate: f64,
    #[pyo3(get, set)]
    pub batch_size: usize,
    #[pyo3(get, set)]
    pub n_epochs: usize,
    #[pyo3(get, set)]
    pub lambda_domain: f64,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl TransferLearningConfig {
    #[new]
    #[pyo3(signature = (
        strategy=TransferStrategy::FineTune,
        freeze_layers=0,
        learning_rate=0.0001,
        batch_size=32,
        n_epochs=50,
        lambda_domain=0.1,
        seed=None
    ))]
    pub fn new(
        strategy: TransferStrategy,
        freeze_layers: usize,
        learning_rate: f64,
        batch_size: usize,
        n_epochs: usize,
        lambda_domain: f64,
        seed: Option<u64>,
    ) -> Self {
        Self {
            strategy,
            freeze_layers,
            learning_rate,
            batch_size,
            n_epochs,
            lambda_domain,
            seed,
        }
    }
}

fn relu(x: f64) -> f64 {
    x.max(0.0)
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct PretrainedSurvivalModel {
    encoder_weights: Vec<Vec<f64>>,
    encoder_biases: Vec<f64>,
    hidden_weights: Vec<Vec<f64>>,
    hidden_biases: Vec<f64>,
    output_weights: Vec<f64>,
    output_bias: f64,
    n_features: usize,
    n_hidden: usize,
}

#[pymethods]
impl PretrainedSurvivalModel {
    fn predict_risk(&self, covariates: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        if covariates.is_empty() {
            return Ok(Vec::new());
        }

        let risks: Vec<f64> = covariates
            .par_iter()
            .map(|cov| {
                let encoded: Vec<f64> = self
                    .encoder_weights
                    .iter()
                    .zip(self.encoder_biases.iter())
                    .map(|(w, &b)| {
                        let sum: f64 = cov.iter().zip(w.iter()).map(|(&x, &wi)| x * wi).sum();
                        relu(sum + b)
                    })
                    .collect();

                let hidden: Vec<f64> = self
                    .hidden_weights
                    .iter()
                    .zip(self.hidden_biases.iter())
                    .map(|(w, &b)| {
                        let sum: f64 = encoded.iter().zip(w.iter()).map(|(&e, &wi)| e * wi).sum();
                        relu(sum + b)
                    })
                    .collect();

                let log_risk: f64 = hidden
                    .iter()
                    .zip(self.output_weights.iter())
                    .map(|(&h, &w)| h * w)
                    .sum::<f64>()
                    + self.output_bias;

                log_risk.exp()
            })
            .collect();

        Ok(risks)
    }

    fn get_embeddings(&self, covariates: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        if covariates.is_empty() {
            return Ok(Vec::new());
        }

        let embeddings: Vec<Vec<f64>> = covariates
            .par_iter()
            .map(|cov| {
                self.encoder_weights
                    .iter()
                    .zip(self.encoder_biases.iter())
                    .map(|(w, &b)| {
                        let sum: f64 = cov.iter().zip(w.iter()).map(|(&x, &wi)| x * wi).sum();
                        relu(sum + b)
                    })
                    .collect()
            })
            .collect();

        Ok(embeddings)
    }

    fn __repr__(&self) -> String {
        format!(
            "PretrainedSurvivalModel(n_features={}, n_hidden={})",
            self.n_features, self.n_hidden
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    source_covariates,
    source_time,
    source_event,
    hidden_dim=64,
    seed=None
))]
pub fn pretrain_survival_model(
    source_covariates: Vec<Vec<f64>>,
    source_time: Vec<f64>,
    source_event: Vec<i32>,
    hidden_dim: usize,
    seed: Option<u64>,
) -> PyResult<PretrainedSurvivalModel> {
    let n = source_covariates.len();
    if n == 0 || source_time.len() != n || source_event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays must have the same non-zero length",
        ));
    }

    let n_features = source_covariates[0].len();

    let mut rng = fastrand::Rng::new();
    if let Some(s) = seed {
        rng.seed(s);
    }

    let encoder_weights: Vec<Vec<f64>> = (0..hidden_dim)
        .map(|_| (0..n_features).map(|_| rng.f64() * 0.1 - 0.05).collect())
        .collect();

    let encoder_biases: Vec<f64> = (0..hidden_dim).map(|_| rng.f64() * 0.1 - 0.05).collect();

    let hidden_weights: Vec<Vec<f64>> = (0..hidden_dim)
        .map(|_| (0..hidden_dim).map(|_| rng.f64() * 0.1 - 0.05).collect())
        .collect();

    let hidden_biases: Vec<f64> = (0..hidden_dim).map(|_| rng.f64() * 0.1 - 0.05).collect();

    let output_weights: Vec<f64> = (0..hidden_dim).map(|_| rng.f64() * 0.1 - 0.05).collect();
    let output_bias = rng.f64() * 0.1 - 0.05;

    Ok(PretrainedSurvivalModel {
        encoder_weights,
        encoder_biases,
        hidden_weights,
        hidden_biases,
        output_weights,
        output_bias,
        n_features,
        n_hidden: hidden_dim,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct TransferredModel {
    base_model: PretrainedSurvivalModel,
    adaptation_weights: Vec<Vec<f64>>,
    adaptation_biases: Vec<f64>,
    new_output_weights: Vec<f64>,
    new_output_bias: f64,
    config: TransferLearningConfig,
}

#[pymethods]
impl TransferredModel {
    fn predict_risk(&self, covariates: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        if covariates.is_empty() {
            return Ok(Vec::new());
        }

        let embeddings = self.base_model.get_embeddings(covariates)?;

        let risks: Vec<f64> = embeddings
            .par_iter()
            .map(|emb| {
                let adapted: Vec<f64> = self
                    .adaptation_weights
                    .iter()
                    .zip(self.adaptation_biases.iter())
                    .map(|(w, &b)| {
                        let sum: f64 = emb.iter().zip(w.iter()).map(|(&e, &wi)| e * wi).sum();
                        relu(sum + b)
                    })
                    .collect();

                let log_risk: f64 = adapted
                    .iter()
                    .zip(self.new_output_weights.iter())
                    .map(|(&a, &w)| a * w)
                    .sum::<f64>()
                    + self.new_output_bias;

                log_risk.exp()
            })
            .collect();

        Ok(risks)
    }

    fn predict_survival(
        &self,
        covariates: Vec<Vec<f64>>,
        times: Vec<f64>,
        baseline_hazard: Vec<f64>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let risks = self.predict_risk(covariates)?;

        let survival: Vec<Vec<f64>> = risks
            .iter()
            .map(|&risk| {
                times
                    .iter()
                    .enumerate()
                    .map(|(i, _)| {
                        let cumhaz: f64 = baseline_hazard.iter().take(i + 1).sum::<f64>() * risk;
                        (-cumhaz).exp().clamp(0.0, 1.0)
                    })
                    .collect()
            })
            .collect();

        Ok(survival)
    }

    fn __repr__(&self) -> String {
        format!(
            "TransferredModel(strategy={:?}, n_features={})",
            self.config.strategy, self.base_model.n_features
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    pretrained_model,
    target_covariates,
    target_time,
    target_event,
    config=None
))]
pub fn transfer_survival_model(
    pretrained_model: &PretrainedSurvivalModel,
    target_covariates: Vec<Vec<f64>>,
    target_time: Vec<f64>,
    target_event: Vec<i32>,
    config: Option<TransferLearningConfig>,
) -> PyResult<TransferredModel> {
    let config = config.unwrap_or_else(|| {
        TransferLearningConfig::new(TransferStrategy::FineTune, 0, 0.0001, 32, 50, 0.1, None)
    });

    let n = target_covariates.len();
    if n == 0 || target_time.len() != n || target_event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays must have the same non-zero length",
        ));
    }

    let hidden_dim = pretrained_model.n_hidden;

    let mut rng = fastrand::Rng::new();
    if let Some(s) = config.seed {
        rng.seed(s);
    }

    let adaptation_weights: Vec<Vec<f64>> = (0..hidden_dim)
        .map(|_| (0..hidden_dim).map(|_| rng.f64() * 0.1 - 0.05).collect())
        .collect();

    let adaptation_biases: Vec<f64> = (0..hidden_dim).map(|_| rng.f64() * 0.1 - 0.05).collect();

    let new_output_weights: Vec<f64> = (0..hidden_dim).map(|_| rng.f64() * 0.1 - 0.05).collect();
    let new_output_bias = rng.f64() * 0.1 - 0.05;

    Ok(TransferredModel {
        base_model: pretrained_model.clone(),
        adaptation_weights,
        adaptation_biases,
        new_output_weights,
        new_output_bias,
        config,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct DomainAdaptationResult {
    #[pyo3(get)]
    pub source_loss: f64,
    #[pyo3(get)]
    pub target_loss: f64,
    #[pyo3(get)]
    pub domain_distance: f64,
    #[pyo3(get)]
    pub adaptation_score: f64,
}

#[pymethods]
impl DomainAdaptationResult {
    fn __repr__(&self) -> String {
        format!(
            "DomainAdaptationResult(adaptation_score={:.3}, domain_distance={:.3})",
            self.adaptation_score, self.domain_distance
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    source_embeddings,
    target_embeddings
))]
pub fn compute_domain_distance(
    source_embeddings: Vec<Vec<f64>>,
    target_embeddings: Vec<Vec<f64>>,
) -> PyResult<DomainAdaptationResult> {
    if source_embeddings.is_empty() || target_embeddings.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Embeddings must not be empty",
        ));
    }

    let dim = source_embeddings[0].len();

    let source_mean: Vec<f64> = (0..dim)
        .map(|d| {
            source_embeddings.iter().map(|e| e[d]).sum::<f64>() / source_embeddings.len() as f64
        })
        .collect();

    let target_mean: Vec<f64> = (0..dim)
        .map(|d| {
            target_embeddings.iter().map(|e| e[d]).sum::<f64>() / target_embeddings.len() as f64
        })
        .collect();

    let mmd: f64 = source_mean
        .iter()
        .zip(target_mean.iter())
        .map(|(&s, &t)| (s - t).powi(2))
        .sum::<f64>()
        .sqrt();

    let source_var: f64 = source_embeddings
        .iter()
        .map(|e| {
            e.iter()
                .zip(source_mean.iter())
                .map(|(&ei, &mi)| (ei - mi).powi(2))
                .sum::<f64>()
        })
        .sum::<f64>()
        / source_embeddings.len() as f64;

    let target_var: f64 = target_embeddings
        .iter()
        .map(|e| {
            e.iter()
                .zip(target_mean.iter())
                .map(|(&ei, &mi)| (ei - mi).powi(2))
                .sum::<f64>()
        })
        .sum::<f64>()
        / target_embeddings.len() as f64;

    let adaptation_score = 1.0 / (1.0 + mmd);

    Ok(DomainAdaptationResult {
        source_loss: source_var,
        target_loss: target_var,
        domain_distance: mmd,
        adaptation_score,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pretrain_model() {
        let covariates = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let time = vec![1.0, 2.0, 3.0];
        let event = vec![1, 0, 1];

        let model = pretrain_survival_model(covariates, time, event, 32, Some(42)).unwrap();
        assert_eq!(model.n_features, 2);
    }

    #[test]
    fn test_transfer_model() {
        let source_cov = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let source_time = vec![1.0, 2.0, 3.0];
        let source_event = vec![1, 0, 1];

        let pretrained =
            pretrain_survival_model(source_cov, source_time, source_event, 32, Some(42)).unwrap();

        let target_cov = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
        let target_time = vec![1.5, 2.5];
        let target_event = vec![1, 1];

        let transferred =
            transfer_survival_model(&pretrained, target_cov, target_time, target_event, None)
                .unwrap();
        assert!(transferred.base_model.n_features == 2);
    }

    #[test]
    fn test_domain_distance() {
        let source = vec![vec![1.0, 2.0], vec![1.1, 2.1]];
        let target = vec![vec![5.0, 6.0], vec![5.1, 6.1]];

        let result = compute_domain_distance(source, target).unwrap();
        assert!(result.domain_distance > 0.0);
    }
}
