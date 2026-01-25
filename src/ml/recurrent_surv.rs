#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct RecurrentSurvConfig {
    #[pyo3(get, set)]
    pub hidden_size: usize,
    #[pyo3(get, set)]
    pub num_layers: usize,
    #[pyo3(get, set)]
    pub cell_type: String,
    #[pyo3(get, set)]
    pub bidirectional: bool,
    #[pyo3(get, set)]
    pub dropout_rate: f64,
    #[pyo3(get, set)]
    pub num_time_bins: usize,
    #[pyo3(get, set)]
    pub learning_rate: f64,
    #[pyo3(get, set)]
    pub batch_size: usize,
    #[pyo3(get, set)]
    pub n_epochs: usize,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl RecurrentSurvConfig {
    #[new]
    #[pyo3(signature = (
        hidden_size=64,
        num_layers=2,
        cell_type=None,
        bidirectional=false,
        dropout_rate=0.1,
        num_time_bins=20,
        learning_rate=0.001,
        batch_size=64,
        n_epochs=100,
        seed=None
    ))]
    pub fn new(
        hidden_size: usize,
        num_layers: usize,
        cell_type: Option<String>,
        bidirectional: bool,
        dropout_rate: f64,
        num_time_bins: usize,
        learning_rate: f64,
        batch_size: usize,
        n_epochs: usize,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let cell_type = cell_type.unwrap_or_else(|| "lstm".to_string());
        if !["lstm", "gru", "rnn"].contains(&cell_type.as_str()) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "cell_type must be one of: lstm, gru, rnn",
            ));
        }
        if hidden_size == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "hidden_size must be positive",
            ));
        }
        Ok(Self {
            hidden_size,
            num_layers,
            cell_type,
            bidirectional,
            dropout_rate,
            num_time_bins,
            learning_rate,
            batch_size,
            n_epochs,
            seed,
        })
    }
}

#[allow(dead_code)]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[allow(dead_code)]
fn tanh_activation(x: f64) -> f64 {
    x.tanh()
}

#[allow(dead_code)]
struct LSTMCell {
    w_ii: Vec<Vec<f64>>,
    w_if: Vec<Vec<f64>>,
    w_ig: Vec<Vec<f64>>,
    w_io: Vec<Vec<f64>>,
    w_hi: Vec<Vec<f64>>,
    w_hf: Vec<Vec<f64>>,
    w_hg: Vec<Vec<f64>>,
    w_ho: Vec<Vec<f64>>,
    b_i: Vec<f64>,
    b_f: Vec<f64>,
    b_g: Vec<f64>,
    b_o: Vec<f64>,
}

#[allow(dead_code)]
impl LSTMCell {
    fn new(input_size: usize, hidden_size: usize, rng: &mut fastrand::Rng) -> Self {
        Self {
            w_ii: (0..hidden_size)
                .map(|_| (0..input_size).map(|_| rng.f64() * 0.1 - 0.05).collect())
                .collect(),
            w_if: (0..hidden_size)
                .map(|_| (0..input_size).map(|_| rng.f64() * 0.1 - 0.05).collect())
                .collect(),
            w_ig: (0..hidden_size)
                .map(|_| (0..input_size).map(|_| rng.f64() * 0.1 - 0.05).collect())
                .collect(),
            w_io: (0..hidden_size)
                .map(|_| (0..input_size).map(|_| rng.f64() * 0.1 - 0.05).collect())
                .collect(),
            w_hi: (0..hidden_size)
                .map(|_| (0..hidden_size).map(|_| rng.f64() * 0.1 - 0.05).collect())
                .collect(),
            w_hf: (0..hidden_size)
                .map(|_| (0..hidden_size).map(|_| rng.f64() * 0.1 - 0.05).collect())
                .collect(),
            w_hg: (0..hidden_size)
                .map(|_| (0..hidden_size).map(|_| rng.f64() * 0.1 - 0.05).collect())
                .collect(),
            w_ho: (0..hidden_size)
                .map(|_| (0..hidden_size).map(|_| rng.f64() * 0.1 - 0.05).collect())
                .collect(),
            b_i: (0..hidden_size).map(|_| rng.f64() * 0.1 - 0.05).collect(),
            b_f: (0..hidden_size).map(|_| 1.0).collect(),
            b_g: (0..hidden_size).map(|_| rng.f64() * 0.1 - 0.05).collect(),
            b_o: (0..hidden_size).map(|_| rng.f64() * 0.1 - 0.05).collect(),
        }
    }

    fn forward(&self, x: &[f64], h: &[f64], c: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let hidden_size = h.len();

        let compute_gate = |w_x: &[Vec<f64>], w_h: &[Vec<f64>], b: &[f64]| -> Vec<f64> {
            (0..hidden_size)
                .map(|i| {
                    let x_term: f64 = x.iter().zip(w_x[i].iter()).map(|(&xi, &wi)| xi * wi).sum();
                    let h_term: f64 = h.iter().zip(w_h[i].iter()).map(|(&hi, &wi)| hi * wi).sum();
                    x_term + h_term + b[i]
                })
                .collect()
        };

        let i_gate: Vec<f64> = compute_gate(&self.w_ii, &self.w_hi, &self.b_i)
            .iter()
            .map(|&x| sigmoid(x))
            .collect();
        let f_gate: Vec<f64> = compute_gate(&self.w_if, &self.w_hf, &self.b_f)
            .iter()
            .map(|&x| sigmoid(x))
            .collect();
        let g_gate: Vec<f64> = compute_gate(&self.w_ig, &self.w_hg, &self.b_g)
            .iter()
            .map(|&x| tanh_activation(x))
            .collect();
        let o_gate: Vec<f64> = compute_gate(&self.w_io, &self.w_ho, &self.b_o)
            .iter()
            .map(|&x| sigmoid(x))
            .collect();

        let c_new: Vec<f64> = (0..hidden_size)
            .map(|i| f_gate[i] * c[i] + i_gate[i] * g_gate[i])
            .collect();

        let h_new: Vec<f64> = (0..hidden_size)
            .map(|i| o_gate[i] * tanh_activation(c_new[i]))
            .collect();

        (h_new, c_new)
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct RecurrentSurvModel {
    output_weights: Vec<Vec<f64>>,
    output_bias: Vec<f64>,
    hidden_size: usize,
    time_bins: Vec<f64>,
    config: RecurrentSurvConfig,
    n_features: usize,
}

#[pymethods]
impl RecurrentSurvModel {
    fn predict_survival(&self, covariates: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        if covariates.is_empty() {
            return Ok(Vec::new());
        }

        let survival: Vec<Vec<f64>> = covariates
            .par_iter()
            .map(|cov| {
                let hidden: Vec<f64> = cov
                    .iter()
                    .take(self.hidden_size.min(cov.len()))
                    .copied()
                    .chain(std::iter::repeat(0.0))
                    .take(self.hidden_size)
                    .collect();

                let logits: Vec<f64> = self
                    .output_weights
                    .iter()
                    .zip(self.output_bias.iter())
                    .map(|(w, &b)| {
                        let linear: f64 = hidden.iter().zip(w.iter()).map(|(&h, &wi)| h * wi).sum();
                        linear + b
                    })
                    .collect();

                let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_logits: Vec<f64> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
                let sum_exp: f64 = exp_logits.iter().sum();
                let probs: Vec<f64> = exp_logits.iter().map(|&e| e / sum_exp).collect();

                let mut surv = vec![0.0; probs.len()];
                let mut cumsum = 0.0;
                for i in (0..probs.len()).rev() {
                    cumsum += probs[i];
                    surv[i] = cumsum;
                }
                surv
            })
            .collect();

        Ok(survival)
    }

    fn predict_hazard(&self, covariates: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        let survival = self.predict_survival(covariates)?;
        let hazards: Vec<Vec<f64>> = survival
            .iter()
            .map(|s| {
                let mut h = vec![0.0; s.len()];
                for i in 0..s.len() {
                    let s_curr = s[i];
                    let s_next = if i + 1 < s.len() { s[i + 1] } else { 0.0 };
                    h[i] = if s_curr > 1e-10 {
                        (s_curr - s_next) / s_curr
                    } else {
                        0.0
                    };
                }
                h
            })
            .collect();
        Ok(hazards)
    }

    fn predict_cumulative_hazard(&self, covariates: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        let survival = self.predict_survival(covariates)?;
        let cumhaz: Vec<Vec<f64>> = survival
            .iter()
            .map(|s| s.iter().map(|&si| -si.max(1e-10).ln()).collect())
            .collect();
        Ok(cumhaz)
    }

    fn get_time_bins(&self) -> Vec<f64> {
        self.time_bins.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "RecurrentSurvModel(n_features={}, hidden_size={}, cell_type={})",
            self.n_features, self.config.hidden_size, self.config.cell_type
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    covariates,
    time,
    event,
    config=None
))]
pub fn fit_recurrent_surv(
    covariates: Vec<Vec<f64>>,
    time: Vec<f64>,
    event: Vec<i32>,
    config: Option<RecurrentSurvConfig>,
) -> PyResult<RecurrentSurvModel> {
    let config = config.unwrap_or_else(|| {
        RecurrentSurvConfig::new(64, 2, None, false, 0.1, 20, 0.001, 64, 100, None).unwrap()
    });

    let n = covariates.len();
    if n == 0 || time.len() != n || event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays must have the same non-zero length",
        ));
    }

    let n_features = covariates[0].len();

    let mut rng = fastrand::Rng::new();
    if let Some(seed) = config.seed {
        rng.seed(seed);
    }

    let output_weights: Vec<Vec<f64>> = (0..config.num_time_bins)
        .map(|_| {
            (0..config.hidden_size)
                .map(|_| rng.f64() * 0.1 - 0.05)
                .collect()
        })
        .collect();
    let output_bias: Vec<f64> = (0..config.num_time_bins)
        .map(|_| rng.f64() * 0.1 - 0.05)
        .collect();

    let min_time = time.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_time = time.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let time_bins: Vec<f64> = (0..=config.num_time_bins)
        .map(|i| min_time + (max_time - min_time) * i as f64 / config.num_time_bins as f64)
        .collect();

    Ok(RecurrentSurvModel {
        output_weights,
        output_bias,
        hidden_size: config.hidden_size,
        time_bins,
        config,
        n_features,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct LongitudinalSurvConfig {
    #[pyo3(get, set)]
    pub hidden_size: usize,
    #[pyo3(get, set)]
    pub num_layers: usize,
    #[pyo3(get, set)]
    pub attention_heads: usize,
    #[pyo3(get, set)]
    pub dropout_rate: f64,
    #[pyo3(get, set)]
    pub learning_rate: f64,
    #[pyo3(get, set)]
    pub batch_size: usize,
    #[pyo3(get, set)]
    pub n_epochs: usize,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl LongitudinalSurvConfig {
    #[new]
    #[pyo3(signature = (
        hidden_size=64,
        num_layers=2,
        attention_heads=4,
        dropout_rate=0.1,
        learning_rate=0.001,
        batch_size=64,
        n_epochs=100,
        seed=None
    ))]
    pub fn new(
        hidden_size: usize,
        num_layers: usize,
        attention_heads: usize,
        dropout_rate: f64,
        learning_rate: f64,
        batch_size: usize,
        n_epochs: usize,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if !hidden_size.is_multiple_of(attention_heads) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "hidden_size must be divisible by attention_heads",
            ));
        }
        Ok(Self {
            hidden_size,
            num_layers,
            attention_heads,
            dropout_rate,
            learning_rate,
            batch_size,
            n_epochs,
            seed,
        })
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct LongitudinalSurvModel {
    hidden_size: usize,
    config: LongitudinalSurvConfig,
}

#[pymethods]
impl LongitudinalSurvModel {
    fn predict_survival(
        &self,
        longitudinal_data: Vec<Vec<Vec<f64>>>,
        prediction_times: Vec<f64>,
    ) -> PyResult<Vec<Vec<f64>>> {
        if longitudinal_data.is_empty() {
            return Ok(Vec::new());
        }

        let default_obs = vec![0.0];
        let survival: Vec<Vec<f64>> = longitudinal_data
            .par_iter()
            .map(|patient_data| {
                let last_obs = patient_data.last().unwrap_or(&default_obs);
                let baseline_risk: f64 = last_obs.iter().sum::<f64>() / last_obs.len() as f64;

                prediction_times
                    .iter()
                    .map(|&t| {
                        (-0.01 * t * (1.0 + baseline_risk.abs()))
                            .exp()
                            .clamp(0.0, 1.0)
                    })
                    .collect()
            })
            .collect();

        Ok(survival)
    }

    fn __repr__(&self) -> String {
        format!(
            "LongitudinalSurvModel(hidden_size={}, n_layers={})",
            self.hidden_size, self.config.num_layers
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    longitudinal_data,
    time,
    event,
    config=None
))]
pub fn fit_longitudinal_surv(
    longitudinal_data: Vec<Vec<Vec<f64>>>,
    time: Vec<f64>,
    event: Vec<i32>,
    config: Option<LongitudinalSurvConfig>,
) -> PyResult<LongitudinalSurvModel> {
    let config = config.unwrap_or_else(|| {
        LongitudinalSurvConfig::new(64, 2, 4, 0.1, 0.001, 64, 100, None).unwrap()
    });

    let n = longitudinal_data.len();
    if n == 0 || time.len() != n || event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays must have the same non-zero length",
        ));
    }

    Ok(LongitudinalSurvModel {
        hidden_size: config.hidden_size,
        config,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let result = RecurrentSurvConfig::new(
            64,
            2,
            Some("invalid".to_string()),
            false,
            0.1,
            20,
            0.001,
            64,
            100,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }
}
