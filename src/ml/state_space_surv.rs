#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct MambaSurvConfig {
    #[pyo3(get, set)]
    pub d_model: usize,
    #[pyo3(get, set)]
    pub d_state: usize,
    #[pyo3(get, set)]
    pub d_conv: usize,
    #[pyo3(get, set)]
    pub expand_factor: usize,
    #[pyo3(get, set)]
    pub num_layers: usize,
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
    pub seed: Option<u64>,
}

#[pymethods]
impl MambaSurvConfig {
    #[new]
    #[pyo3(signature = (
        d_model=64,
        d_state=16,
        d_conv=4,
        expand_factor=2,
        num_layers=4,
        num_time_bins=20,
        dropout_rate=0.1,
        learning_rate=0.001,
        batch_size=64,
        n_epochs=100,
        seed=None
    ))]
    pub fn new(
        d_model: usize,
        d_state: usize,
        d_conv: usize,
        expand_factor: usize,
        num_layers: usize,
        num_time_bins: usize,
        dropout_rate: f64,
        learning_rate: f64,
        batch_size: usize,
        n_epochs: usize,
        seed: Option<u64>,
    ) -> Self {
        Self {
            d_model,
            d_state,
            d_conv,
            expand_factor,
            num_layers,
            num_time_bins,
            dropout_rate,
            learning_rate,
            batch_size,
            n_epochs,
            seed,
        }
    }
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

fn selective_scan(
    x: &[f64],
    delta: &[f64],
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    c: &[Vec<f64>],
    d: f64,
) -> Vec<f64> {
    let seq_len = x.len();
    let d_state = a[0].len();

    let mut h = vec![0.0; d_state];
    let mut outputs = vec![0.0; seq_len];

    for t in 0..seq_len {
        let dt = delta[t].exp();

        for s in 0..d_state {
            let a_bar = (-dt * a[t][s].abs()).exp();
            let b_bar = (1.0 - a_bar) * b[t][s];
            h[s] = a_bar * h[s] + b_bar * x[t];
        }

        let y: f64 = h.iter().zip(c[t].iter()).map(|(&hi, &ci)| hi * ci).sum();
        outputs[t] = y + d * x[t];
    }

    outputs
}

fn conv1d(input: &[f64], kernel: &[f64], padding: usize) -> Vec<f64> {
    let input_len = input.len();
    let kernel_len = kernel.len();

    let mut padded = vec![0.0; padding];
    padded.extend_from_slice(input);
    padded.extend(vec![0.0; padding + kernel_len - 1]);

    (0..input_len)
        .map(|i| {
            kernel
                .iter()
                .enumerate()
                .map(|(k, &w)| padded[i + k] * w)
                .sum()
        })
        .collect()
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct MambaSurvModel {
    input_proj_weights: Vec<Vec<f64>>,
    input_proj_biases: Vec<f64>,
    conv_weights: Vec<Vec<f64>>,
    delta_proj_weights: Vec<Vec<f64>>,
    a_log: Vec<Vec<f64>>,
    d_weights: Vec<f64>,
    out_proj_weights: Vec<Vec<f64>>,
    output_weights: Vec<Vec<f64>>,
    output_biases: Vec<f64>,
    time_bins: Vec<f64>,
    config: MambaSurvConfig,
    #[allow(dead_code)]
    n_features: usize,
}

#[pymethods]
impl MambaSurvModel {
    fn predict_survival(&self, sequences: Vec<Vec<Vec<f64>>>) -> PyResult<Vec<Vec<f64>>> {
        if sequences.is_empty() {
            return Ok(Vec::new());
        }

        let survival: Vec<Vec<f64>> = sequences
            .par_iter()
            .map(|seq| {
                let seq_len = seq.len();
                let d_inner = self.config.d_model * self.config.expand_factor;

                let mut x_proj: Vec<Vec<f64>> = seq
                    .iter()
                    .map(|s| {
                        self.input_proj_weights
                            .iter()
                            .zip(self.input_proj_biases.iter())
                            .map(|(w, &b)| {
                                let sum: f64 = s.iter().zip(w.iter()).map(|(&x, &wi)| x * wi).sum();
                                silu(sum + b)
                            })
                            .collect()
                    })
                    .collect();

                for d in 0..d_inner.min(x_proj[0].len()) {
                    let channel: Vec<f64> = x_proj.iter().map(|x| x[d]).collect();
                    let conv_kernel = &self.conv_weights[d % self.conv_weights.len()];
                    let convolved = conv1d(&channel, conv_kernel, self.config.d_conv / 2);
                    for (t, &v) in convolved.iter().enumerate() {
                        if t < x_proj.len() {
                            x_proj[t][d] = silu(v);
                        }
                    }
                }

                let delta: Vec<f64> = x_proj
                    .iter()
                    .map(|x| {
                        let proj: f64 = x
                            .iter()
                            .zip(self.delta_proj_weights[0].iter())
                            .map(|(&xi, &wi)| xi * wi)
                            .sum();
                        proj.clamp(-10.0, 10.0)
                    })
                    .collect();

                let a: Vec<Vec<f64>> = (0..seq_len)
                    .map(|_| self.a_log[0].iter().map(|&a| -a.exp()).collect())
                    .collect();

                let b: Vec<Vec<f64>> = x_proj
                    .iter()
                    .map(|x| x.iter().take(self.config.d_state).cloned().collect())
                    .collect();

                let c: Vec<Vec<f64>> = x_proj
                    .iter()
                    .map(|x| {
                        x.iter()
                            .skip(self.config.d_state)
                            .take(self.config.d_state)
                            .cloned()
                            .collect()
                    })
                    .collect();

                let x_flat: Vec<f64> = x_proj.iter().map(|x| x.iter().sum::<f64>()).collect();
                let d_val = self.d_weights[0];

                let ssm_out = selective_scan(&x_flat, &delta, &a, &b, &c, d_val);

                let final_hidden: Vec<f64> = self
                    .out_proj_weights
                    .iter()
                    .map(|w| {
                        ssm_out
                            .iter()
                            .zip(w.iter().cycle())
                            .map(|(&s, &wi)| s * wi)
                            .sum::<f64>()
                            / seq_len as f64
                    })
                    .collect();

                let logits: Vec<f64> = self
                    .output_weights
                    .iter()
                    .zip(self.output_biases.iter())
                    .map(|(w, &b)| {
                        let sum: f64 = final_hidden
                            .iter()
                            .zip(w.iter())
                            .map(|(&h, &wi)| h * wi)
                            .sum();
                        sum + b
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
                    surv[i] = cumsum.min(1.0);
                }
                surv
            })
            .collect();

        Ok(survival)
    }

    fn get_hidden_states(&self, sequence: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        let _seq_len = sequence.len();
        let _d_inner = self.config.d_model * self.config.expand_factor;

        let hidden_states: Vec<Vec<f64>> = sequence
            .iter()
            .map(|s| {
                self.input_proj_weights
                    .iter()
                    .zip(self.input_proj_biases.iter())
                    .map(|(w, &b)| {
                        let sum: f64 = s.iter().zip(w.iter()).map(|(&x, &wi)| x * wi).sum();
                        silu(sum + b)
                    })
                    .collect()
            })
            .collect();

        Ok(hidden_states)
    }

    fn get_time_bins(&self) -> Vec<f64> {
        self.time_bins.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "MambaSurvModel(d_model={}, d_state={}, layers={})",
            self.config.d_model, self.config.d_state, self.config.num_layers
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    sequences,
    time,
    event,
    config=None
))]
pub fn fit_mamba_surv(
    sequences: Vec<Vec<Vec<f64>>>,
    time: Vec<f64>,
    event: Vec<i32>,
    config: Option<MambaSurvConfig>,
) -> PyResult<MambaSurvModel> {
    let config = config
        .unwrap_or_else(|| MambaSurvConfig::new(64, 16, 4, 2, 4, 20, 0.1, 0.001, 64, 100, None));

    let n = sequences.len();
    if n == 0 || time.len() != n || event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays must have the same non-zero length",
        ));
    }

    let n_features = sequences[0].first().map(|s| s.len()).unwrap_or(1);
    let d_inner = config.d_model * config.expand_factor;

    let mut rng = fastrand::Rng::new();
    if let Some(seed) = config.seed {
        rng.seed(seed);
    }

    let input_proj_weights: Vec<Vec<f64>> = (0..d_inner)
        .map(|_| (0..n_features).map(|_| rng.f64() * 0.1 - 0.05).collect())
        .collect();
    let input_proj_biases: Vec<f64> = (0..d_inner).map(|_| rng.f64() * 0.1 - 0.05).collect();

    let conv_weights: Vec<Vec<f64>> = (0..d_inner)
        .map(|_| (0..config.d_conv).map(|_| rng.f64() * 0.1 - 0.05).collect())
        .collect();

    let delta_proj_weights: Vec<Vec<f64>> = (0..1)
        .map(|_| (0..d_inner).map(|_| rng.f64() * 0.1 - 0.05).collect())
        .collect();

    let a_log: Vec<Vec<f64>> = (0..config.num_layers)
        .map(|_| (0..config.d_state).map(|_| rng.f64() + 0.5).collect())
        .collect();

    let d_weights: Vec<f64> = (0..config.num_layers).map(|_| rng.f64() * 0.1).collect();

    let out_proj_weights: Vec<Vec<f64>> = (0..config.d_model)
        .map(|_| (0..d_inner).map(|_| rng.f64() * 0.1 - 0.05).collect())
        .collect();

    let output_weights: Vec<Vec<f64>> = (0..config.num_time_bins)
        .map(|_| {
            (0..config.d_model)
                .map(|_| rng.f64() * 0.1 - 0.05)
                .collect()
        })
        .collect();
    let output_biases: Vec<f64> = (0..config.num_time_bins)
        .map(|_| rng.f64() * 0.1 - 0.05)
        .collect();

    let min_time = time.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_time = time.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let time_bins: Vec<f64> = (0..=config.num_time_bins)
        .map(|i| min_time + (max_time - min_time) * i as f64 / config.num_time_bins as f64)
        .collect();

    Ok(MambaSurvModel {
        input_proj_weights,
        input_proj_biases,
        conv_weights,
        delta_proj_weights,
        a_log,
        d_weights,
        out_proj_weights,
        output_weights,
        output_biases,
        time_bins,
        config,
        n_features,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        assert!(silu(1.0) > 0.5);
    }

    #[test]
    fn test_conv1d() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![0.5, 0.5];
        let output = conv1d(&input, &kernel, 0);
        assert_eq!(output.len(), 5);
    }

    #[test]
    fn test_selective_scan() {
        let x = vec![1.0, 2.0, 3.0];
        let delta = vec![0.1, 0.1, 0.1];
        let a = vec![vec![-0.5, -0.5], vec![-0.5, -0.5], vec![-0.5, -0.5]];
        let b = vec![vec![0.1, 0.1], vec![0.1, 0.1], vec![0.1, 0.1]];
        let c = vec![vec![0.1, 0.1], vec![0.1, 0.1], vec![0.1, 0.1]];
        let output = selective_scan(&x, &delta, &a, &b, &c, 0.1);
        assert_eq!(output.len(), 3);
    }
}
