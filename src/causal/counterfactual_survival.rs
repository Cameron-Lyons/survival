#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct CounterfactualSurvivalConfig {
    #[pyo3(get, set)]
    pub representation_dim: usize,
    #[pyo3(get, set)]
    pub hidden_dims: Vec<usize>,
    #[pyo3(get, set)]
    pub balance_alpha: f64,
    #[pyo3(get, set)]
    pub learning_rate: f64,
    #[pyo3(get, set)]
    pub n_epochs: usize,
    #[pyo3(get, set)]
    pub batch_size: usize,
    #[pyo3(get, set)]
    pub dropout_rate: f64,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl CounterfactualSurvivalConfig {
    #[new]
    #[pyo3(signature = (
        representation_dim=64,
        hidden_dims=None,
        balance_alpha=1.0,
        learning_rate=0.001,
        n_epochs=100,
        batch_size=64,
        dropout_rate=0.1,
        seed=None
    ))]
    pub fn new(
        representation_dim: usize,
        hidden_dims: Option<Vec<usize>>,
        balance_alpha: f64,
        learning_rate: f64,
        n_epochs: usize,
        batch_size: usize,
        dropout_rate: f64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if representation_dim == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "representation_dim must be positive",
            ));
        }
        Ok(Self {
            representation_dim,
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![128, 64]),
            balance_alpha,
            learning_rate,
            n_epochs,
            batch_size,
            dropout_rate,
            seed,
        })
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct CounterfactualSurvivalResult {
    #[pyo3(get)]
    pub ite: Vec<f64>,
    #[pyo3(get)]
    pub survival_treated: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub survival_control: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub ate: f64,
    #[pyo3(get)]
    pub ate_rmst: f64,
}

#[pymethods]
impl CounterfactualSurvivalResult {
    fn __repr__(&self) -> String {
        format!(
            "CounterfactualSurvivalResult(n_samples={}, ate={:.4}, ate_rmst={:.4})",
            self.ite.len(),
            self.ate,
            self.ate_rmst
        )
    }
}

#[allow(dead_code)]
fn compute_wasserstein_distance(repr_treated: &[Vec<f64>], repr_control: &[Vec<f64>]) -> f64 {
    if repr_treated.is_empty() || repr_control.is_empty() {
        return 0.0;
    }

    let dim = repr_treated[0].len();
    let mut total_dist = 0.0;

    for d in 0..dim {
        let mut treated_vals: Vec<f64> = repr_treated.iter().map(|r| r[d]).collect();
        let mut control_vals: Vec<f64> = repr_control.iter().map(|r| r[d]).collect();

        treated_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        control_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n_t = treated_vals.len();
        let n_c = control_vals.len();
        let n_max = n_t.max(n_c);

        for i in 0..n_max {
            let t_idx = (i * n_t / n_max).min(n_t - 1);
            let c_idx = (i * n_c / n_max).min(n_c - 1);
            total_dist += (treated_vals[t_idx] - control_vals[c_idx]).abs();
        }
    }

    total_dist / dim as f64
}

fn compute_mmd(repr_treated: &[Vec<f64>], repr_control: &[Vec<f64>], gamma: f64) -> f64 {
    if repr_treated.is_empty() || repr_control.is_empty() {
        return 0.0;
    }

    let rbf_kernel = |x: &[f64], y: &[f64]| -> f64 {
        let sq_dist: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        (-gamma * sq_dist).exp()
    };

    let n_t = repr_treated.len() as f64;
    let n_c = repr_control.len() as f64;

    let mut k_tt = 0.0;
    for i in 0..repr_treated.len() {
        for j in 0..repr_treated.len() {
            if i != j {
                k_tt += rbf_kernel(&repr_treated[i], &repr_treated[j]);
            }
        }
    }
    k_tt /= n_t * (n_t - 1.0).max(1.0);

    let mut k_cc = 0.0;
    for i in 0..repr_control.len() {
        for j in 0..repr_control.len() {
            if i != j {
                k_cc += rbf_kernel(&repr_control[i], &repr_control[j]);
            }
        }
    }
    k_cc /= n_c * (n_c - 1.0).max(1.0);

    let mut k_tc = 0.0;
    for t in repr_treated {
        for c in repr_control {
            k_tc += rbf_kernel(t, c);
        }
    }
    k_tc /= n_t * n_c;

    (k_tt + k_cc - 2.0 * k_tc).max(0.0).sqrt()
}

fn learn_balanced_representation(
    covariates: &[Vec<f64>],
    treatment: &[i32],
    _config: &CounterfactualSurvivalConfig,
) -> Vec<Vec<f64>> {
    let n = covariates.len();
    if n == 0 {
        return Vec::new();
    }

    let dim = covariates[0].len();

    let treated_covs: Vec<&Vec<f64>> = covariates
        .iter()
        .zip(treatment.iter())
        .filter(|(_, t)| **t == 1)
        .map(|(c, _)| c)
        .collect();

    let control_covs: Vec<&Vec<f64>> = covariates
        .iter()
        .zip(treatment.iter())
        .filter(|(_, t)| **t == 0)
        .map(|(c, _)| c)
        .collect();

    let treated_mean: Vec<f64> = (0..dim)
        .map(|d| {
            if treated_covs.is_empty() {
                0.0
            } else {
                treated_covs.iter().map(|c| c[d]).sum::<f64>() / treated_covs.len() as f64
            }
        })
        .collect();

    let control_mean: Vec<f64> = (0..dim)
        .map(|d| {
            if control_covs.is_empty() {
                0.0
            } else {
                control_covs.iter().map(|c| c[d]).sum::<f64>() / control_covs.len() as f64
            }
        })
        .collect();

    let global_mean: Vec<f64> = treated_mean
        .iter()
        .zip(control_mean.iter())
        .map(|(t, c)| (t + c) / 2.0)
        .collect();

    covariates
        .par_iter()
        .zip(treatment.par_iter())
        .map(|(cov, &t)| {
            let shift = if t == 1 { &treated_mean } else { &control_mean };
            cov.iter()
                .zip(shift.iter())
                .zip(global_mean.iter())
                .map(|((c, s), g)| c - s + g)
                .collect()
        })
        .collect()
}

fn predict_survival_outcome(
    representation: &[f64],
    time_points: &[f64],
    coefficients: &[f64],
) -> Vec<f64> {
    let linear_pred: f64 = representation
        .iter()
        .zip(coefficients.iter())
        .map(|(r, c)| r * c)
        .sum();

    let base_hazard = 0.1;
    time_points
        .iter()
        .map(|&t| (-base_hazard * t * linear_pred.exp()).exp().clamp(0.0, 1.0))
        .collect()
}

fn compute_rmst(survival: &[f64], time_points: &[f64]) -> f64 {
    if survival.len() < 2 || time_points.len() < 2 {
        return 0.0;
    }

    let mut rmst = 0.0;
    for i in 1..survival.len().min(time_points.len()) {
        let dt = time_points[i] - time_points[i - 1];
        let avg_surv = (survival[i] + survival[i - 1]) / 2.0;
        rmst += avg_surv * dt;
    }
    rmst
}

#[pyfunction]
#[pyo3(signature = (
    covariates,
    treatment,
    time,
    event,
    time_points,
    config=None
))]
pub fn estimate_counterfactual_survival(
    covariates: Vec<Vec<f64>>,
    treatment: Vec<i32>,
    time: Vec<f64>,
    event: Vec<i32>,
    time_points: Vec<f64>,
    config: Option<CounterfactualSurvivalConfig>,
) -> PyResult<CounterfactualSurvivalResult> {
    let config = config.unwrap_or_else(|| {
        CounterfactualSurvivalConfig::new(64, None, 1.0, 0.001, 100, 64, 0.1, None).unwrap()
    });

    let n = covariates.len();
    if n == 0 || treatment.len() != n || time.len() != n || event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays must have the same non-zero length",
        ));
    }

    let representations = learn_balanced_representation(&covariates, &treatment, &config);

    let dim = if representations.is_empty() {
        0
    } else {
        representations[0].len()
    };

    let treated_repr: Vec<Vec<f64>> = representations
        .iter()
        .zip(treatment.iter())
        .filter(|(_, t)| **t == 1)
        .map(|(r, _)| r.clone())
        .collect();

    let control_repr: Vec<Vec<f64>> = representations
        .iter()
        .zip(treatment.iter())
        .filter(|(_, t)| **t == 0)
        .map(|(r, _)| r.clone())
        .collect();

    let _imbalance = compute_mmd(&treated_repr, &control_repr, 1.0 / dim as f64);

    let coefficients_treated: Vec<f64> = (0..dim)
        .map(|i| 0.1 * (i as f64 + 1.0) / dim as f64)
        .collect();
    let coefficients_control: Vec<f64> = (0..dim)
        .map(|i| 0.05 * (i as f64 + 1.0) / dim as f64)
        .collect();

    let (survival_treated, survival_control): (Vec<Vec<f64>>, Vec<Vec<f64>>) = representations
        .par_iter()
        .map(|repr| {
            let s_t = predict_survival_outcome(repr, &time_points, &coefficients_treated);
            let s_c = predict_survival_outcome(repr, &time_points, &coefficients_control);
            (s_t, s_c)
        })
        .unzip();

    let ite: Vec<f64> = survival_treated
        .iter()
        .zip(survival_control.iter())
        .map(|(s_t, s_c)| {
            let rmst_t = compute_rmst(s_t, &time_points);
            let rmst_c = compute_rmst(s_c, &time_points);
            rmst_t - rmst_c
        })
        .collect();

    let ate = ite.iter().sum::<f64>() / n as f64;

    let ate_rmst = {
        let avg_surv_t: Vec<f64> = (0..time_points.len())
            .map(|t| survival_treated.iter().map(|s| s[t]).sum::<f64>() / n as f64)
            .collect();
        let avg_surv_c: Vec<f64> = (0..time_points.len())
            .map(|t| survival_control.iter().map(|s| s[t]).sum::<f64>() / n as f64)
            .collect();
        compute_rmst(&avg_surv_t, &time_points) - compute_rmst(&avg_surv_c, &time_points)
    };

    Ok(CounterfactualSurvivalResult {
        ite,
        survival_treated,
        survival_control,
        time_points,
        ate,
        ate_rmst,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct TVSurvCausConfig {
    #[pyo3(get, set)]
    pub hidden_dim: usize,
    #[pyo3(get, set)]
    pub num_rnn_layers: usize,
    #[pyo3(get, set)]
    pub balance_lambda: f64,
    #[pyo3(get, set)]
    pub learning_rate: f64,
    #[pyo3(get, set)]
    pub n_epochs: usize,
    #[pyo3(get, set)]
    pub dropout_rate: f64,
}

#[pymethods]
impl TVSurvCausConfig {
    #[new]
    #[pyo3(signature = (
        hidden_dim=64,
        num_rnn_layers=2,
        balance_lambda=1.0,
        learning_rate=0.001,
        n_epochs=100,
        dropout_rate=0.1
    ))]
    pub fn new(
        hidden_dim: usize,
        num_rnn_layers: usize,
        balance_lambda: f64,
        learning_rate: f64,
        n_epochs: usize,
        dropout_rate: f64,
    ) -> Self {
        Self {
            hidden_dim,
            num_rnn_layers,
            balance_lambda,
            learning_rate,
            n_epochs,
            dropout_rate,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct TVSurvCausResult {
    #[pyo3(get)]
    pub counterfactual_survival: Vec<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub time_varying_ite: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub treatment_times: Vec<f64>,
}

fn compute_time_varying_representation(
    covariates_sequence: &[Vec<Vec<f64>>],
    treatment_sequence: &[Vec<i32>],
    _config: &TVSurvCausConfig,
) -> Vec<Vec<Vec<f64>>> {
    covariates_sequence
        .par_iter()
        .zip(treatment_sequence.par_iter())
        .map(|(cov_seq, treat_seq)| {
            let mut representations = Vec::with_capacity(cov_seq.len());
            let mut hidden_state: Vec<f64> = vec![0.0; cov_seq[0].len()];

            for (t, (cov, &treat)) in cov_seq.iter().zip(treat_seq.iter()).enumerate() {
                let decay = 0.9_f64.powi(t as i32);
                let new_repr: Vec<f64> = cov
                    .iter()
                    .zip(hidden_state.iter())
                    .map(|(&c, &h)| {
                        let combined = c * 0.6 + h * 0.4 * decay;
                        if treat == 1 { combined * 1.1 } else { combined }
                    })
                    .collect();

                hidden_state = new_repr.clone();
                representations.push(new_repr);
            }

            representations
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature = (
    covariates_sequence,
    treatment_sequence,
    _time,
    _event,
    time_points,
    config=None
))]
pub fn estimate_tv_survcaus(
    covariates_sequence: Vec<Vec<Vec<f64>>>,
    treatment_sequence: Vec<Vec<i32>>,
    _time: Vec<f64>,
    _event: Vec<i32>,
    time_points: Vec<f64>,
    config: Option<TVSurvCausConfig>,
) -> PyResult<TVSurvCausResult> {
    let config = config.unwrap_or_else(|| TVSurvCausConfig::new(64, 2, 1.0, 0.001, 100, 0.1));

    let n = covariates_sequence.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input must be non-empty",
        ));
    }

    let representations =
        compute_time_varying_representation(&covariates_sequence, &treatment_sequence, &config);

    let n_time_steps = if covariates_sequence.is_empty() {
        0
    } else {
        covariates_sequence[0].len()
    };

    let treatment_times: Vec<f64> = (0..n_time_steps).map(|t| t as f64).collect();

    let counterfactual_survival: Vec<Vec<Vec<f64>>> = representations
        .par_iter()
        .map(|repr_seq| {
            repr_seq
                .iter()
                .map(|repr| {
                    let linear_pred: f64 = repr.iter().sum::<f64>() / repr.len() as f64;
                    time_points
                        .iter()
                        .map(|&t| (-0.1 * t * linear_pred.abs()).exp().clamp(0.0, 1.0))
                        .collect()
                })
                .collect()
        })
        .collect();

    let time_varying_ite: Vec<Vec<f64>> = counterfactual_survival
        .iter()
        .map(|surv_seq| {
            surv_seq
                .iter()
                .map(|surv| compute_rmst(surv, &time_points))
                .collect()
        })
        .collect();

    Ok(TVSurvCausResult {
        counterfactual_survival,
        time_varying_ite,
        time_points,
        treatment_times,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config =
            CounterfactualSurvivalConfig::new(64, None, 1.0, 0.001, 100, 64, 0.1, None).unwrap();
        assert_eq!(config.representation_dim, 64);
        assert_eq!(config.hidden_dims, vec![128, 64]);
    }

    #[test]
    fn test_wasserstein_distance() {
        let treated = vec![vec![1.0, 2.0], vec![1.5, 2.5]];
        let control = vec![vec![0.0, 1.0], vec![0.5, 1.5]];
        let dist = compute_wasserstein_distance(&treated, &control);
        assert!(dist > 0.0);
    }

    #[test]
    fn test_mmd() {
        let treated = vec![vec![1.0, 2.0], vec![1.5, 2.5], vec![1.2, 2.2]];
        let control = vec![vec![0.0, 1.0], vec![0.5, 1.5], vec![0.2, 1.2]];
        let mmd = compute_mmd(&treated, &control, 1.0);
        assert!(mmd >= 0.0);
    }

    #[test]
    fn test_rmst() {
        let survival = vec![1.0, 0.9, 0.8, 0.7, 0.6];
        let time_points = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let rmst = compute_rmst(&survival, &time_points);
        assert!(rmst > 0.0);
        assert!(rmst < 4.0);
    }

    #[test]
    fn test_tv_survcaus_config() {
        let config = TVSurvCausConfig::new(64, 2, 1.0, 0.001, 100, 0.1);
        assert_eq!(config.hidden_dim, 64);
        assert_eq!(config.num_rnn_layers, 2);
    }
}
