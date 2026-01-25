#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct NeuralODESurvConfig {
    #[pyo3(get, set)]
    pub hidden_dims: Vec<usize>,
    #[pyo3(get, set)]
    pub ode_hidden_dim: usize,
    #[pyo3(get, set)]
    pub num_integration_steps: usize,
    #[pyo3(get, set)]
    pub solver: String,
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
impl NeuralODESurvConfig {
    #[new]
    #[pyo3(signature = (
        hidden_dims=None,
        ode_hidden_dim=32,
        num_integration_steps=100,
        solver=None,
        dropout_rate=0.1,
        learning_rate=0.001,
        batch_size=64,
        n_epochs=100,
        seed=None
    ))]
    pub fn new(
        hidden_dims: Option<Vec<usize>>,
        ode_hidden_dim: usize,
        num_integration_steps: usize,
        solver: Option<String>,
        dropout_rate: f64,
        learning_rate: f64,
        batch_size: usize,
        n_epochs: usize,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let solver = solver.unwrap_or_else(|| "rk4".to_string());
        if !["euler", "rk4", "dopri5"].contains(&solver.as_str()) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "solver must be one of: euler, rk4, dopri5",
            ));
        }
        Ok(Self {
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![64, 32]),
            ode_hidden_dim,
            num_integration_steps,
            solver,
            dropout_rate,
            learning_rate,
            batch_size,
            n_epochs,
            seed,
        })
    }
}

fn tanh(x: f64) -> f64 {
    x.tanh()
}

fn ode_func(state: &[f64], weights: &[Vec<f64>], biases: &[f64]) -> Vec<f64> {
    weights
        .iter()
        .zip(biases.iter())
        .map(|(w, &b)| {
            let sum: f64 = state.iter().zip(w.iter()).map(|(&s, &wi)| s * wi).sum();
            tanh(sum + b)
        })
        .collect()
}

fn euler_step(state: &[f64], weights: &[Vec<f64>], biases: &[f64], dt: f64) -> Vec<f64> {
    let derivative = ode_func(state, weights, biases);
    state
        .iter()
        .zip(derivative.iter())
        .map(|(&s, &d)| s + dt * d)
        .collect()
}

fn rk4_step(state: &[f64], weights: &[Vec<f64>], biases: &[f64], dt: f64) -> Vec<f64> {
    let k1 = ode_func(state, weights, biases);

    let state2: Vec<f64> = state
        .iter()
        .zip(k1.iter())
        .map(|(&s, &k)| s + 0.5 * dt * k)
        .collect();
    let k2 = ode_func(&state2, weights, biases);

    let state3: Vec<f64> = state
        .iter()
        .zip(k2.iter())
        .map(|(&s, &k)| s + 0.5 * dt * k)
        .collect();
    let k3 = ode_func(&state3, weights, biases);

    let state4: Vec<f64> = state
        .iter()
        .zip(k3.iter())
        .map(|(&s, &k)| s + dt * k)
        .collect();
    let k4 = ode_func(&state4, weights, biases);

    state
        .iter()
        .enumerate()
        .map(|(i, &s)| s + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
        .collect()
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct NeuralODESurvModel {
    encoder_weights: Vec<Vec<f64>>,
    encoder_biases: Vec<f64>,
    ode_weights: Vec<Vec<f64>>,
    ode_biases: Vec<f64>,
    output_weights: Vec<f64>,
    output_bias: f64,
    #[allow(dead_code)]
    max_time: f64,
    config: NeuralODESurvConfig,
    n_features: usize,
}

#[pymethods]
impl NeuralODESurvModel {
    fn predict_survival(
        &self,
        covariates: Vec<Vec<f64>>,
        times: Vec<f64>,
    ) -> PyResult<Vec<Vec<f64>>> {
        if covariates.is_empty() {
            return Ok(Vec::new());
        }

        let survival: Vec<Vec<f64>> = covariates
            .par_iter()
            .map(|cov| {
                let initial_state: Vec<f64> = self
                    .encoder_weights
                    .iter()
                    .zip(self.encoder_biases.iter())
                    .map(|(w, &b)| {
                        let sum: f64 = cov.iter().zip(w.iter()).map(|(&x, &wi)| x * wi).sum();
                        tanh(sum + b)
                    })
                    .collect();

                times
                    .iter()
                    .map(|&t| {
                        let n_steps = self.config.num_integration_steps;
                        let dt = t / n_steps as f64;

                        let mut state = initial_state.clone();
                        for _ in 0..n_steps {
                            state = match self.config.solver.as_str() {
                                "euler" => {
                                    euler_step(&state, &self.ode_weights, &self.ode_biases, dt)
                                }
                                _ => rk4_step(&state, &self.ode_weights, &self.ode_biases, dt),
                            };
                        }

                        let log_cumhaz: f64 = state
                            .iter()
                            .zip(self.output_weights.iter())
                            .map(|(&s, &w)| s * w)
                            .sum::<f64>()
                            + self.output_bias;

                        (-log_cumhaz.exp()).exp().clamp(0.0, 1.0)
                    })
                    .collect()
            })
            .collect();

        Ok(survival)
    }

    fn predict_hazard(
        &self,
        covariates: Vec<Vec<f64>>,
        times: Vec<f64>,
    ) -> PyResult<Vec<Vec<f64>>> {
        if covariates.is_empty() || times.is_empty() {
            return Ok(Vec::new());
        }

        let survival = self.predict_survival(covariates.clone(), times.clone())?;

        let hazards: Vec<Vec<f64>> = survival
            .iter()
            .map(|s| {
                let mut h = vec![0.0; s.len()];
                for i in 0..s.len() {
                    if i == 0 {
                        h[i] = if s[i] > 1e-10 {
                            -s[i].ln() / times[i].max(1e-10)
                        } else {
                            0.0
                        };
                    } else {
                        let dt = times[i] - times[i - 1];
                        h[i] = if s[i] > 1e-10 && dt > 1e-10 {
                            (s[i - 1].ln() - s[i].ln()) / dt
                        } else {
                            0.0
                        };
                    }
                }
                h
            })
            .collect();

        Ok(hazards)
    }

    fn get_trajectory(&self, covariates: Vec<f64>, times: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
        let initial_state: Vec<f64> = self
            .encoder_weights
            .iter()
            .zip(self.encoder_biases.iter())
            .map(|(w, &b)| {
                let sum: f64 = covariates
                    .iter()
                    .zip(w.iter())
                    .map(|(&x, &wi)| x * wi)
                    .sum();
                tanh(sum + b)
            })
            .collect();

        let trajectories: Vec<Vec<f64>> = times
            .iter()
            .map(|&t| {
                let n_steps = self.config.num_integration_steps;
                let dt = t / n_steps as f64;

                let mut state = initial_state.clone();
                for _ in 0..n_steps {
                    state = match self.config.solver.as_str() {
                        "euler" => euler_step(&state, &self.ode_weights, &self.ode_biases, dt),
                        _ => rk4_step(&state, &self.ode_weights, &self.ode_biases, dt),
                    };
                }
                state
            })
            .collect();

        Ok(trajectories)
    }

    fn __repr__(&self) -> String {
        format!(
            "NeuralODESurvModel(n_features={}, ode_dim={}, solver={})",
            self.n_features, self.config.ode_hidden_dim, self.config.solver
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
pub fn fit_neural_ode_surv(
    covariates: Vec<Vec<f64>>,
    time: Vec<f64>,
    event: Vec<i32>,
    config: Option<NeuralODESurvConfig>,
) -> PyResult<NeuralODESurvModel> {
    let config = config.unwrap_or_else(|| {
        NeuralODESurvConfig::new(None, 32, 100, None, 0.1, 0.001, 64, 100, None).unwrap()
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

    let encoder_weights: Vec<Vec<f64>> = (0..config.ode_hidden_dim)
        .map(|_| (0..n_features).map(|_| rng.f64() * 0.1 - 0.05).collect())
        .collect();

    let encoder_biases: Vec<f64> = (0..config.ode_hidden_dim)
        .map(|_| rng.f64() * 0.1 - 0.05)
        .collect();

    let ode_weights: Vec<Vec<f64>> = (0..config.ode_hidden_dim)
        .map(|_| {
            (0..config.ode_hidden_dim)
                .map(|_| rng.f64() * 0.1 - 0.05)
                .collect()
        })
        .collect();

    let ode_biases: Vec<f64> = (0..config.ode_hidden_dim)
        .map(|_| rng.f64() * 0.1 - 0.05)
        .collect();

    let output_weights: Vec<f64> = (0..config.ode_hidden_dim)
        .map(|_| rng.f64() * 0.1 - 0.05)
        .collect();

    let output_bias = rng.f64() * 0.1 - 0.05;

    let max_time = time.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    Ok(NeuralODESurvModel {
        encoder_weights,
        encoder_biases,
        ode_weights,
        ode_biases,
        output_weights,
        output_bias,
        max_time,
        config,
        n_features,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let result = NeuralODESurvConfig::new(
            None,
            32,
            100,
            Some("invalid".to_string()),
            0.1,
            0.001,
            64,
            100,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_euler_step() {
        let state = vec![1.0, 0.0];
        let weights = vec![vec![0.1, 0.0], vec![0.0, 0.1]];
        let biases = vec![0.0, 0.0];
        let new_state = euler_step(&state, &weights, &biases, 0.1);
        assert_eq!(new_state.len(), 2);
    }

    #[test]
    fn test_rk4_step() {
        let state = vec![1.0, 0.0];
        let weights = vec![vec![0.1, 0.0], vec![0.0, 0.1]];
        let biases = vec![0.0, 0.0];
        let new_state = rk4_step(&state, &weights, &biases, 0.1);
        assert_eq!(new_state.len(), 2);
    }
}
