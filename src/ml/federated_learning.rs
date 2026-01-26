#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct FederatedConfig {
    #[pyo3(get, set)]
    pub n_rounds: usize,
    #[pyo3(get, set)]
    pub n_clients: usize,
    #[pyo3(get, set)]
    pub client_fraction: f64,
    #[pyo3(get, set)]
    pub local_epochs: usize,
    #[pyo3(get, set)]
    pub learning_rate: f64,
    #[pyo3(get, set)]
    pub aggregation_strategy: String,
    #[pyo3(get, set)]
    pub differential_privacy: bool,
    #[pyo3(get, set)]
    pub noise_multiplier: f64,
    #[pyo3(get, set)]
    pub clip_norm: f64,
}

#[pymethods]
impl FederatedConfig {
    #[new]
    #[pyo3(signature = (
        n_rounds=10,
        n_clients=5,
        client_fraction=1.0,
        local_epochs=5,
        learning_rate=0.01,
        aggregation_strategy="fedavg",
        differential_privacy=false,
        noise_multiplier=1.0,
        clip_norm=1.0
    ))]
    pub fn new(
        n_rounds: usize,
        n_clients: usize,
        client_fraction: f64,
        local_epochs: usize,
        learning_rate: f64,
        aggregation_strategy: &str,
        differential_privacy: bool,
        noise_multiplier: f64,
        clip_norm: f64,
    ) -> PyResult<Self> {
        if n_clients == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "n_clients must be positive",
            ));
        }
        if client_fraction <= 0.0 || client_fraction > 1.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "client_fraction must be in (0, 1]",
            ));
        }
        Ok(Self {
            n_rounds,
            n_clients,
            client_fraction,
            local_epochs,
            learning_rate,
            aggregation_strategy: aggregation_strategy.to_string(),
            differential_privacy,
            noise_multiplier,
            clip_norm,
        })
    }
}

#[derive(Debug, Clone)]
struct ClientModel {
    weights: Vec<f64>,
    n_samples: usize,
}

fn clip_gradient(gradient: &mut [f64], clip_norm: f64) {
    let norm: f64 = gradient.iter().map(|&g| g * g).sum::<f64>().sqrt();
    if norm > clip_norm {
        let scale = clip_norm / norm;
        for g in gradient.iter_mut() {
            *g *= scale;
        }
    }
}

fn add_gaussian_noise(values: &mut [f64], noise_multiplier: f64, clip_norm: f64, seed: u64) {
    let mut rng_state = seed;
    for v in values.iter_mut() {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u1 = (rng_state as f64) / (u64::MAX as f64);
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u2 = (rng_state as f64) / (u64::MAX as f64);
        let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        *v += noise * noise_multiplier * clip_norm;
    }
}

fn train_local_cox_model(
    time: &[f64],
    event: &[i32],
    covariates: &[Vec<f64>],
    initial_weights: &[f64],
    learning_rate: f64,
    n_epochs: usize,
    differential_privacy: bool,
    noise_multiplier: f64,
    clip_norm: f64,
    seed: u64,
) -> Vec<f64> {
    let n = time.len();
    let n_features = if covariates.is_empty() {
        0
    } else {
        covariates[0].len()
    };

    let mut weights = initial_weights.to_vec();

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| time[b].partial_cmp(&time[a]).unwrap());

    for _epoch in 0..n_epochs {
        let mut gradient = vec![0.0; n_features];

        let linear_pred: Vec<f64> = covariates
            .iter()
            .map(|x| x.iter().zip(weights.iter()).map(|(&xi, &wi)| xi * wi).sum())
            .collect();

        let exp_lp: Vec<f64> = linear_pred.iter().map(|&lp| lp.exp()).collect();

        let mut risk_sum = 0.0;
        let mut weighted_sum = vec![0.0; n_features];

        for &i in &indices {
            risk_sum += exp_lp[i];
            for (j, &xij) in covariates[i].iter().enumerate() {
                weighted_sum[j] += xij * exp_lp[i];
            }

            if event[i] == 1 {
                for (j, g) in gradient.iter_mut().enumerate() {
                    *g += covariates[i][j] - weighted_sum[j] / risk_sum;
                }
            }
        }

        if differential_privacy {
            clip_gradient(&mut gradient, clip_norm);
            add_gaussian_noise(&mut gradient, noise_multiplier, clip_norm, seed);
        }

        for (w, g) in weights.iter_mut().zip(gradient.iter()) {
            *w += learning_rate * g / n as f64;
        }
    }

    weights
}

fn fedavg_aggregate(client_models: &[ClientModel]) -> Vec<f64> {
    if client_models.is_empty() {
        return vec![];
    }

    let total_samples: usize = client_models.iter().map(|c| c.n_samples).sum();
    let n_weights = client_models[0].weights.len();
    let mut aggregated = vec![0.0; n_weights];

    for client in client_models {
        let weight = client.n_samples as f64 / total_samples as f64;
        for (i, &w) in client.weights.iter().enumerate() {
            aggregated[i] += weight * w;
        }
    }

    aggregated
}

fn fedprox_aggregate(client_models: &[ClientModel], global_weights: &[f64], mu: f64) -> Vec<f64> {
    let mut aggregated = fedavg_aggregate(client_models);

    for (i, w) in aggregated.iter_mut().enumerate() {
        *w = (1.0 - mu) * *w + mu * global_weights[i];
    }

    aggregated
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct FederatedSurvivalResult {
    #[pyo3(get)]
    pub global_weights: Vec<f64>,
    #[pyo3(get)]
    pub round_metrics: Vec<f64>,
    #[pyo3(get)]
    pub client_sample_sizes: Vec<usize>,
    #[pyo3(get)]
    pub convergence_round: Option<usize>,
    #[pyo3(get)]
    pub privacy_budget: Option<f64>,
}

#[pymethods]
impl FederatedSurvivalResult {
    fn __repr__(&self) -> String {
        format!(
            "FederatedSurvivalResult(n_weights={}, converged_at={:?})",
            self.global_weights.len(),
            self.convergence_round
        )
    }

    fn predict_risk(&self, covariates: Vec<Vec<f64>>) -> Vec<f64> {
        covariates
            .par_iter()
            .map(|x| {
                x.iter()
                    .zip(self.global_weights.iter())
                    .map(|(&xi, &wi)| xi * wi)
                    .sum::<f64>()
                    .exp()
            })
            .collect()
    }

    fn predict_survival(
        &self,
        covariates: Vec<Vec<f64>>,
        baseline_hazard: Vec<f64>,
    ) -> Vec<Vec<f64>> {
        let risks = self.predict_risk(covariates);
        risks
            .into_iter()
            .map(|risk| baseline_hazard.iter().map(|&h| (-h * risk).exp()).collect())
            .collect()
    }
}

#[pyfunction]
#[pyo3(signature = (
    client_times,
    client_events,
    client_covariates,
    config,
    seed=None
))]
pub fn federated_cox(
    client_times: Vec<Vec<f64>>,
    client_events: Vec<Vec<i32>>,
    client_covariates: Vec<Vec<Vec<f64>>>,
    config: FederatedConfig,
    seed: Option<u64>,
) -> PyResult<FederatedSurvivalResult> {
    let n_clients = client_times.len();
    if n_clients == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Must have at least one client",
        ));
    }
    if client_events.len() != n_clients || client_covariates.len() != n_clients {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All client data must have the same length",
        ));
    }

    let n_features = if client_covariates[0].is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Covariates cannot be empty",
        ));
    } else {
        client_covariates[0][0].len()
    };

    let seed = seed.unwrap_or(42);
    let mut global_weights = vec![0.0; n_features];
    let mut round_metrics = Vec::new();
    let mut convergence_round = None;

    let client_sample_sizes: Vec<usize> = client_times.iter().map(|t| t.len()).collect();

    let n_selected = ((n_clients as f64 * config.client_fraction).ceil() as usize).max(1);

    for round in 0..config.n_rounds {
        let mut selected_clients: Vec<usize> = (0..n_clients).collect();
        let round_seed = seed.wrapping_add(round as u64);
        let mut rng_state = round_seed;
        for i in (1..selected_clients.len()).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng_state as usize) % (i + 1);
            selected_clients.swap(i, j);
        }
        selected_clients.truncate(n_selected);

        let client_models: Vec<ClientModel> = selected_clients
            .par_iter()
            .map(|&client_idx| {
                let client_seed = seed.wrapping_add((round * n_clients + client_idx) as u64);
                let weights = train_local_cox_model(
                    &client_times[client_idx],
                    &client_events[client_idx],
                    &client_covariates[client_idx],
                    &global_weights,
                    config.learning_rate,
                    config.local_epochs,
                    config.differential_privacy,
                    config.noise_multiplier,
                    config.clip_norm,
                    client_seed,
                );
                ClientModel {
                    weights,
                    n_samples: client_times[client_idx].len(),
                }
            })
            .collect();

        let new_weights = match config.aggregation_strategy.as_str() {
            "fedprox" => fedprox_aggregate(&client_models, &global_weights, 0.01),
            _ => fedavg_aggregate(&client_models),
        };

        let weight_change: f64 = global_weights
            .iter()
            .zip(new_weights.iter())
            .map(|(&old, &new)| (new - old).powi(2))
            .sum::<f64>()
            .sqrt();

        round_metrics.push(weight_change);

        if weight_change < 1e-6 && convergence_round.is_none() {
            convergence_round = Some(round);
        }

        global_weights = new_weights;
    }

    let privacy_budget = if config.differential_privacy {
        let epsilon =
            (2.0 * (1.25_f64 / config.n_rounds as f64).ln()).sqrt() / config.noise_multiplier;
        Some(epsilon * config.n_rounds as f64)
    } else {
        None
    };

    Ok(FederatedSurvivalResult {
        global_weights,
        round_metrics,
        client_sample_sizes,
        convergence_round,
        privacy_budget,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct SecureAggregationResult {
    #[pyo3(get)]
    pub aggregated_gradient: Vec<f64>,
    #[pyo3(get)]
    pub n_participants: usize,
    #[pyo3(get)]
    pub dropout_rate: f64,
}

#[pymethods]
impl SecureAggregationResult {
    fn __repr__(&self) -> String {
        format!(
            "SecureAggregationResult(n_participants={}, dropout={:.1}%)",
            self.n_participants,
            self.dropout_rate * 100.0
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    client_gradients,
    threshold=0.5,
    seed=None
))]
pub fn secure_aggregate(
    client_gradients: Vec<Vec<f64>>,
    threshold: f64,
    seed: Option<u64>,
) -> PyResult<SecureAggregationResult> {
    let n_clients = client_gradients.len();
    if n_clients == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Must have at least one client",
        ));
    }

    let seed = seed.unwrap_or(42);
    let mut rng_state = seed;

    let mut active_clients = Vec::new();
    for (i, gradient) in client_gradients.iter().enumerate() {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let drop_prob = (rng_state as f64) / (u64::MAX as f64);
        if drop_prob > 0.1 {
            active_clients.push((i, gradient));
        }
    }

    let required = (n_clients as f64 * threshold).ceil() as usize;
    if active_clients.len() < required {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Too many clients dropped out",
        ));
    }

    let n_features = client_gradients[0].len();
    let mut aggregated = vec![0.0; n_features];

    for (_, gradient) in &active_clients {
        for (i, &g) in gradient.iter().enumerate() {
            aggregated[i] += g;
        }
    }

    let n_active = active_clients.len();
    for g in &mut aggregated {
        *g /= n_active as f64;
    }

    let dropout_rate = 1.0 - (n_active as f64 / n_clients as f64);

    Ok(SecureAggregationResult {
        aggregated_gradient: aggregated,
        n_participants: n_active,
        dropout_rate,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct PrivacyAccountant {
    epsilon_spent: f64,
    delta: f64,
    #[pyo3(get)]
    pub queries: Vec<f64>,
}

#[pymethods]
impl PrivacyAccountant {
    #[new]
    #[pyo3(signature = (delta=1e-5))]
    pub fn new(delta: f64) -> Self {
        Self {
            epsilon_spent: 0.0,
            delta,
            queries: Vec::new(),
        }
    }

    fn add_query(&mut self, epsilon: f64) {
        self.epsilon_spent += epsilon;
        self.queries.push(epsilon);
    }

    fn get_epsilon(&self) -> f64 {
        self.epsilon_spent
    }

    fn get_delta(&self) -> f64 {
        self.delta
    }

    fn remaining_budget(&self, total_budget: f64) -> f64 {
        (total_budget - self.epsilon_spent).max(0.0)
    }

    fn __repr__(&self) -> String {
        format!(
            "PrivacyAccountant(epsilon={:.4}, delta={:.2e}, queries={})",
            self.epsilon_spent,
            self.delta,
            self.queries.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_federated_cox() {
        let client_times = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![2.0, 3.0, 4.0, 5.0, 6.0]];
        let client_events = vec![vec![1, 0, 1, 0, 1], vec![1, 1, 0, 1, 0]];
        let client_covariates = vec![
            vec![
                vec![0.5, 0.2],
                vec![0.3, 0.8],
                vec![0.7, 0.1],
                vec![0.2, 0.9],
                vec![0.8, 0.4],
            ],
            vec![
                vec![0.1, 0.6],
                vec![0.9, 0.3],
                vec![0.4, 0.7],
                vec![0.6, 0.2],
                vec![0.2, 0.8],
            ],
        ];

        let config = FederatedConfig::new(5, 2, 1.0, 3, 0.01, "fedavg", false, 1.0, 1.0).unwrap();

        let result = federated_cox(
            client_times,
            client_events,
            client_covariates,
            config,
            Some(42),
        )
        .unwrap();

        assert_eq!(result.global_weights.len(), 2);
        assert_eq!(result.client_sample_sizes.len(), 2);
    }

    #[test]
    fn test_federated_cox_with_dp() {
        let client_times = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![2.0, 3.0, 4.0, 5.0, 6.0]];
        let client_events = vec![vec![1, 0, 1, 0, 1], vec![1, 1, 0, 1, 0]];
        let client_covariates = vec![
            vec![
                vec![0.5, 0.2],
                vec![0.3, 0.8],
                vec![0.7, 0.1],
                vec![0.2, 0.9],
                vec![0.8, 0.4],
            ],
            vec![
                vec![0.1, 0.6],
                vec![0.9, 0.3],
                vec![0.4, 0.7],
                vec![0.6, 0.2],
                vec![0.2, 0.8],
            ],
        ];

        let config = FederatedConfig::new(5, 2, 1.0, 3, 0.01, "fedavg", true, 1.0, 1.0).unwrap();

        let result = federated_cox(
            client_times,
            client_events,
            client_covariates,
            config,
            Some(42),
        )
        .unwrap();

        assert!(result.privacy_budget.is_some());
    }

    #[test]
    fn test_secure_aggregate() {
        let gradients = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.2, 0.3, 0.4],
            vec![0.3, 0.4, 0.5],
        ];

        let result = secure_aggregate(gradients, 0.5, Some(42)).unwrap();
        assert_eq!(result.aggregated_gradient.len(), 3);
        assert!(result.n_participants > 0);
    }

    #[test]
    fn test_privacy_accountant() {
        let mut accountant = PrivacyAccountant::new(1e-5);
        accountant.add_query(0.1);
        accountant.add_query(0.2);

        assert!((accountant.get_epsilon() - 0.3).abs() < 1e-10);
        assert!((accountant.remaining_budget(1.0) - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_federated_config_invalid_clients() {
        let result = FederatedConfig::new(10, 0, 1.0, 5, 0.01, "fedavg", false, 1.0, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_federated_config_invalid_fraction() {
        let result = FederatedConfig::new(10, 5, 0.0, 5, 0.01, "fedavg", false, 1.0, 1.0);
        assert!(result.is_err());

        let result = FederatedConfig::new(10, 5, 1.5, 5, 0.01, "fedavg", false, 1.0, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_federated_cox_empty_clients() {
        let config = FederatedConfig::new(5, 2, 1.0, 3, 0.01, "fedavg", false, 1.0, 1.0).unwrap();
        let result = federated_cox(vec![], vec![], vec![], config, Some(42));
        assert!(result.is_err());
    }

    #[test]
    fn test_federated_cox_fedprox() {
        let client_times = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![2.0, 3.0, 4.0, 5.0, 6.0]];
        let client_events = vec![vec![1, 0, 1, 0, 1], vec![1, 1, 0, 1, 0]];
        let client_covariates = vec![
            vec![
                vec![0.5, 0.2],
                vec![0.3, 0.8],
                vec![0.7, 0.1],
                vec![0.2, 0.9],
                vec![0.8, 0.4],
            ],
            vec![
                vec![0.1, 0.6],
                vec![0.9, 0.3],
                vec![0.4, 0.7],
                vec![0.6, 0.2],
                vec![0.2, 0.8],
            ],
        ];

        let config = FederatedConfig::new(5, 2, 1.0, 3, 0.01, "fedprox", false, 1.0, 1.0).unwrap();
        let result = federated_cox(
            client_times,
            client_events,
            client_covariates,
            config,
            Some(42),
        )
        .unwrap();

        assert_eq!(result.global_weights.len(), 2);
        assert_eq!(result.round_metrics.len(), 5);
    }

    #[test]
    fn test_federated_cox_predict_risk() {
        let client_times = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let client_events = vec![vec![1, 0, 1, 0, 1]];
        let client_covariates = vec![vec![
            vec![0.5, 0.2],
            vec![0.3, 0.8],
            vec![0.7, 0.1],
            vec![0.2, 0.9],
            vec![0.8, 0.4],
        ]];

        let config = FederatedConfig::new(10, 1, 1.0, 5, 0.01, "fedavg", false, 1.0, 1.0).unwrap();
        let result = federated_cox(
            client_times,
            client_events,
            client_covariates,
            config,
            Some(42),
        )
        .unwrap();

        let test_covariates = vec![vec![0.5, 0.5], vec![0.1, 0.9]];
        let risks = result.predict_risk(test_covariates);
        assert_eq!(risks.len(), 2);
        assert!(risks.iter().all(|&r| r > 0.0));
    }

    #[test]
    fn test_federated_cox_predict_survival() {
        let client_times = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let client_events = vec![vec![1, 0, 1, 0, 1]];
        let client_covariates = vec![vec![
            vec![0.5, 0.2],
            vec![0.3, 0.8],
            vec![0.7, 0.1],
            vec![0.2, 0.9],
            vec![0.8, 0.4],
        ]];

        let config = FederatedConfig::new(5, 1, 1.0, 3, 0.01, "fedavg", false, 1.0, 1.0).unwrap();
        let result = federated_cox(
            client_times,
            client_events,
            client_covariates,
            config,
            Some(42),
        )
        .unwrap();

        let test_covariates = vec![vec![0.5, 0.5]];
        let baseline = vec![0.01, 0.02, 0.03];
        let survival = result.predict_survival(test_covariates, baseline);
        assert_eq!(survival.len(), 1);
        assert_eq!(survival[0].len(), 3);
        assert!(survival[0].iter().all(|&s| (0.0..=1.0).contains(&s)));
    }

    #[test]
    fn test_secure_aggregate_empty() {
        let result = secure_aggregate(vec![], 0.5, Some(42));
        assert!(result.is_err());
    }

    #[test]
    fn test_privacy_accountant_budget_exhausted() {
        let mut accountant = PrivacyAccountant::new(1e-5);
        accountant.add_query(0.5);
        accountant.add_query(0.5);
        accountant.add_query(0.5);

        assert!((accountant.get_epsilon() - 1.5).abs() < 1e-10);
        assert!((accountant.remaining_budget(1.0) - 0.0).abs() < 1e-10);
        assert_eq!(accountant.queries.len(), 3);
    }
}
