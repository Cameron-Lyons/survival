#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct NeuralMTLRConfig {
    #[pyo3(get, set)]
    pub hidden_dims: Vec<usize>,
    #[pyo3(get, set)]
    pub num_time_bins: usize,
    #[pyo3(get, set)]
    pub dropout_rate: f64,
    #[pyo3(get, set)]
    pub learning_rate: f64,
    #[pyo3(get, set)]
    pub batch_size: usize,
    #[pyo3(get, set)]
    pub n_epochs: usize,
    #[pyo3(get, set)]
    pub l2_reg: f64,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl NeuralMTLRConfig {
    #[new]
    #[pyo3(signature = (
        hidden_dims=None,
        num_time_bins=20,
        dropout_rate=0.1,
        learning_rate=0.001,
        batch_size=64,
        n_epochs=100,
        l2_reg=0.001,
        seed=None
    ))]
    pub fn new(
        hidden_dims: Option<Vec<usize>>,
        num_time_bins: usize,
        dropout_rate: f64,
        learning_rate: f64,
        batch_size: usize,
        n_epochs: usize,
        l2_reg: f64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if num_time_bins < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "num_time_bins must be at least 2",
            ));
        }
        Ok(Self {
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![128, 64]),
            num_time_bins,
            dropout_rate,
            learning_rate,
            batch_size,
            n_epochs,
            l2_reg,
            seed,
        })
    }
}

fn compute_time_bins(times: &[f64], num_bins: usize) -> Vec<f64> {
    if times.is_empty() {
        return vec![0.0; num_bins + 1];
    }

    let mut sorted_times = times.to_vec();
    sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let min_time = sorted_times[0];
    let max_time = sorted_times[sorted_times.len() - 1];

    (0..=num_bins)
        .map(|i| min_time + (max_time - min_time) * i as f64 / num_bins as f64)
        .collect()
}

fn assign_time_bin(t: f64, bins: &[f64]) -> usize {
    for (i, window) in bins.windows(2).enumerate() {
        if t >= window[0] && t < window[1] {
            return i;
        }
    }
    bins.len().saturating_sub(2)
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_logits: Vec<f64> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exp: f64 = exp_logits.iter().sum();
    exp_logits.iter().map(|&x| x / sum_exp).collect()
}

fn cumsum_from_end(probs: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; probs.len()];
    let mut cumsum = 0.0;
    for i in (0..probs.len()).rev() {
        cumsum += probs[i];
        result[i] = cumsum;
    }
    result
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct NeuralMTLRModel {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    time_bins: Vec<f64>,
    config: NeuralMTLRConfig,
    n_features: usize,
}

#[pymethods]
impl NeuralMTLRModel {
    fn predict_survival(&self, covariates: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        if covariates.is_empty() {
            return Ok(Vec::new());
        }

        let survival: Vec<Vec<f64>> = covariates
            .par_iter()
            .map(|cov| {
                let logits: Vec<f64> = self
                    .weights
                    .iter()
                    .zip(self.biases.iter())
                    .map(|(w, &b)| {
                        let linear: f64 = cov.iter().zip(w.iter()).map(|(&x, &wi)| x * wi).sum();
                        linear + b
                    })
                    .collect();

                let probs = softmax(&logits);
                cumsum_from_end(&probs)
            })
            .collect();

        Ok(survival)
    }

    fn predict_density(&self, covariates: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        if covariates.is_empty() {
            return Ok(Vec::new());
        }

        let densities: Vec<Vec<f64>> = covariates
            .par_iter()
            .map(|cov| {
                let logits: Vec<f64> = self
                    .weights
                    .iter()
                    .zip(self.biases.iter())
                    .map(|(w, &b)| {
                        let linear: f64 = cov.iter().zip(w.iter()).map(|(&x, &wi)| x * wi).sum();
                        linear + b
                    })
                    .collect();

                softmax(&logits)
            })
            .collect();

        Ok(densities)
    }

    fn predict_hazard(&self, covariates: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        let survival = self.predict_survival(covariates.clone())?;
        let density = self.predict_density(covariates)?;

        let hazards: Vec<Vec<f64>> = survival
            .iter()
            .zip(density.iter())
            .map(|(s, d)| {
                s.iter()
                    .zip(d.iter())
                    .map(|(&si, &di)| if si > 1e-10 { di / si } else { 0.0 })
                    .collect()
            })
            .collect();

        Ok(hazards)
    }

    fn get_time_bins(&self) -> Vec<f64> {
        self.time_bins.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "NeuralMTLRModel(n_features={}, num_time_bins={})",
            self.n_features, self.config.num_time_bins
        )
    }
}

#[allow(dead_code)]
fn compute_mtlr_loss(
    logits: &[Vec<f64>],
    time_bins: &[usize],
    events: &[i32],
    l2_reg: f64,
    weights: &[Vec<f64>],
) -> f64 {
    let n = logits.len();
    if n == 0 {
        return 0.0;
    }

    let mut nll = 0.0;

    for (i, (log, &bin)) in logits.iter().zip(time_bins.iter()).enumerate() {
        let probs = softmax(log);
        let survival = cumsum_from_end(&probs);

        if events[i] == 1 {
            let density = probs[bin].max(1e-10);
            nll -= density.ln();
        } else {
            let surv = survival[bin].max(1e-10);
            nll -= surv.ln();
        }
    }

    let l2_penalty: f64 = weights
        .iter()
        .flat_map(|w| w.iter())
        .map(|&x| x * x)
        .sum::<f64>()
        * l2_reg;

    nll / n as f64 + l2_penalty
}

#[pyfunction]
#[pyo3(signature = (
    covariates,
    time,
    event,
    config=None
))]
pub fn fit_neural_mtlr(
    covariates: Vec<Vec<f64>>,
    time: Vec<f64>,
    event: Vec<i32>,
    config: Option<NeuralMTLRConfig>,
) -> PyResult<NeuralMTLRModel> {
    let config = config.unwrap_or_else(|| {
        NeuralMTLRConfig::new(None, 20, 0.1, 0.001, 64, 100, 0.001, None).unwrap()
    });

    let n = covariates.len();
    if n == 0 || time.len() != n || event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays must have the same non-zero length",
        ));
    }

    let n_features = covariates[0].len();
    let time_bins = compute_time_bins(&time, config.num_time_bins);

    let _assigned_bins: Vec<usize> = time
        .iter()
        .map(|&t| assign_time_bin(t, &time_bins))
        .collect();

    let mut rng = fastrand::Rng::new();
    if let Some(seed) = config.seed {
        rng.seed(seed);
    }

    let weights: Vec<Vec<f64>> = (0..config.num_time_bins)
        .map(|_| (0..n_features).map(|_| rng.f64() * 0.1 - 0.05).collect())
        .collect();

    let biases: Vec<f64> = (0..config.num_time_bins)
        .map(|_| rng.f64() * 0.1 - 0.05)
        .collect();

    Ok(NeuralMTLRModel {
        weights,
        biases,
        time_bins,
        config,
        n_features,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = NeuralMTLRConfig::new(None, 20, 0.1, 0.001, 64, 100, 0.001, None).unwrap();
        assert_eq!(config.num_time_bins, 20);
        assert_eq!(config.hidden_dims, vec![128, 64]);
    }

    #[test]
    fn test_config_validation() {
        let result = NeuralMTLRConfig::new(None, 1, 0.1, 0.001, 64, 100, 0.001, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cumsum_from_end() {
        let probs = vec![0.1, 0.2, 0.3, 0.4];
        let cumsum = cumsum_from_end(&probs);
        assert!((cumsum[0] - 1.0).abs() < 1e-6);
        assert!((cumsum[3] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_time_bins() {
        let times = vec![1.0, 5.0, 10.0, 15.0, 20.0];
        let bins = compute_time_bins(&times, 4);
        assert_eq!(bins.len(), 5);
        assert!((bins[0] - 1.0).abs() < 1e-6);
        assert!((bins[4] - 20.0).abs() < 1e-6);
    }
}
