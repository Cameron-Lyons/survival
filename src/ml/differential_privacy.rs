#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct DPConfig {
    #[pyo3(get, set)]
    pub epsilon: f64,
    #[pyo3(get, set)]
    pub delta: f64,
    #[pyo3(get, set)]
    pub sensitivity: f64,
    #[pyo3(get, set)]
    pub mechanism: String,
    #[pyo3(get, set)]
    pub clip_bounds: Option<(f64, f64)>,
}

#[pymethods]
impl DPConfig {
    #[new]
    #[pyo3(signature = (
        epsilon=1.0,
        delta=1e-5,
        sensitivity=1.0,
        mechanism="gaussian",
        clip_bounds=None
    ))]
    pub fn new(
        epsilon: f64,
        delta: f64,
        sensitivity: f64,
        mechanism: &str,
        clip_bounds: Option<(f64, f64)>,
    ) -> PyResult<Self> {
        if epsilon <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "epsilon must be positive",
            ));
        }
        if !(0.0..1.0).contains(&delta) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "delta must be in [0, 1)",
            ));
        }
        Ok(Self {
            epsilon,
            delta,
            sensitivity,
            mechanism: mechanism.to_string(),
            clip_bounds,
        })
    }
}

fn generate_laplace_noise(scale: f64, seed: u64) -> f64 {
    let mut rng_state = seed;
    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
    let u = (rng_state as f64) / (u64::MAX as f64) - 0.5;
    -scale * u.signum() * (1.0 - 2.0 * u.abs()).ln()
}

fn generate_gaussian_noise(sigma: f64, seed: u64) -> f64 {
    let mut rng_state = seed;
    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
    let u1 = (rng_state as f64) / (u64::MAX as f64);
    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
    let u2 = (rng_state as f64) / (u64::MAX as f64);
    let z = (-2.0 * u1.max(1e-10).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    z * sigma
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct DPSurvivalResult {
    #[pyo3(get)]
    pub survival_curve: Vec<f64>,
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub noise_added: Vec<f64>,
    #[pyo3(get)]
    pub epsilon_used: f64,
}

#[pymethods]
impl DPSurvivalResult {
    fn __repr__(&self) -> String {
        format!(
            "DPSurvivalResult(n_points={}, epsilon={:.4})",
            self.survival_curve.len(),
            self.epsilon_used
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    time,
    event,
    config,
    seed=None
))]
pub fn dp_kaplan_meier(
    time: Vec<f64>,
    event: Vec<i32>,
    config: DPConfig,
    seed: Option<u64>,
) -> PyResult<DPSurvivalResult> {
    let n = time.len();
    if n == 0 || event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and event must have the same non-zero length",
        ));
    }

    let seed = seed.unwrap_or(42);

    let mut unique_times: Vec<f64> = time.to_vec();
    unique_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    unique_times.dedup();

    let n_times = unique_times.len();
    let per_query_epsilon = config.epsilon / n_times as f64;
    let sigma = config.sensitivity * (2.0 * (1.25 / config.delta).ln()).sqrt() / per_query_epsilon;

    let mut survival_curve = Vec::with_capacity(n_times);
    let mut noise_added = Vec::with_capacity(n_times);
    let mut surv = 1.0;

    for (i, &t) in unique_times.iter().enumerate() {
        let at_risk = time.iter().filter(|&&ti| ti >= t).count() as f64;
        let events = time
            .iter()
            .zip(event.iter())
            .filter(|&(&ti, &ei)| (ti - t).abs() < 1e-10 && ei == 1)
            .count() as f64;

        let noise = match config.mechanism.as_str() {
            "laplace" => {
                generate_laplace_noise(config.sensitivity / per_query_epsilon, seed + i as u64)
            }
            _ => generate_gaussian_noise(sigma, seed + i as u64),
        };

        let noisy_events = (events + noise).clamp(0.0, at_risk);
        noise_added.push(noise);

        if at_risk > 0.0 {
            surv *= 1.0 - noisy_events / at_risk;
        }
        surv = surv.clamp(0.0, 1.0);
        survival_curve.push(surv);
    }

    Ok(DPSurvivalResult {
        survival_curve,
        time_points: unique_times,
        noise_added,
        epsilon_used: config.epsilon,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct DPCoxResult {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    #[pyo3(get)]
    pub standard_errors: Vec<f64>,
    #[pyo3(get)]
    pub noise_scale: f64,
    #[pyo3(get)]
    pub epsilon_used: f64,
    #[pyo3(get)]
    pub convergence_iterations: usize,
}

#[pymethods]
impl DPCoxResult {
    fn __repr__(&self) -> String {
        format!(
            "DPCoxResult(n_features={}, epsilon={:.4})",
            self.coefficients.len(),
            self.epsilon_used
        )
    }

    fn predict_risk(&self, covariates: Vec<Vec<f64>>) -> Vec<f64> {
        covariates
            .par_iter()
            .map(|x| {
                x.iter()
                    .zip(self.coefficients.iter())
                    .map(|(&xi, &bi)| xi * bi)
                    .sum::<f64>()
                    .exp()
            })
            .collect()
    }
}

#[pyfunction]
#[pyo3(signature = (
    time,
    event,
    covariates,
    config,
    max_iter=100,
    seed=None
))]
pub fn dp_cox_regression(
    time: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<Vec<f64>>,
    config: DPConfig,
    max_iter: usize,
    seed: Option<u64>,
) -> PyResult<DPCoxResult> {
    let n = time.len();
    if n == 0 || event.len() != n || covariates.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays must have the same non-zero length",
        ));
    }

    let n_features = if covariates.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Covariates cannot be empty",
        ));
    } else {
        covariates[0].len()
    };

    let seed = seed.unwrap_or(42);

    let per_iter_epsilon = config.epsilon / max_iter as f64;
    let sigma = config.sensitivity * (2.0 * (1.25 / config.delta).ln()).sqrt() / per_iter_epsilon;

    let mut coefficients = vec![0.0; n_features];
    let learning_rate = 0.01;

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| time[b].partial_cmp(&time[a]).unwrap());

    let mut convergence_iterations = max_iter;

    for iter in 0..max_iter {
        let linear_pred: Vec<f64> = covariates
            .iter()
            .map(|x| {
                x.iter()
                    .zip(coefficients.iter())
                    .map(|(&xi, &bi)| xi * bi)
                    .sum()
            })
            .collect();

        let exp_lp: Vec<f64> = linear_pred.iter().map(|&lp| lp.exp()).collect();

        let mut gradient = vec![0.0; n_features];
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

        let grad_norm: f64 = gradient.iter().map(|&g| g * g).sum::<f64>().sqrt();
        if grad_norm > config.sensitivity {
            let scale = config.sensitivity / grad_norm;
            for g in &mut gradient {
                *g *= scale;
            }
        }

        for (j, g) in gradient.iter_mut().enumerate() {
            let noise = generate_gaussian_noise(sigma, seed + (iter * n_features + j) as u64);
            *g += noise;
        }

        let mut max_change = 0.0f64;
        for (b, g) in coefficients.iter_mut().zip(gradient.iter()) {
            let change = learning_rate * g / n as f64;
            *b += change;
            max_change = max_change.max(change.abs());
        }

        if max_change < 1e-6 {
            convergence_iterations = iter + 1;
            break;
        }
    }

    let information_matrix = compute_fisher_information(&time, &event, &covariates, &coefficients);
    let standard_errors: Vec<f64> = information_matrix
        .iter()
        .map(|row| {
            let diag = row.iter().sum::<f64>().abs();
            if diag > 1e-10 {
                (1.0 / diag).sqrt()
            } else {
                f64::INFINITY
            }
        })
        .collect();

    Ok(DPCoxResult {
        coefficients,
        standard_errors,
        noise_scale: sigma,
        epsilon_used: config.epsilon,
        convergence_iterations,
    })
}

fn compute_fisher_information(
    time: &[f64],
    event: &[i32],
    covariates: &[Vec<f64>],
    coefficients: &[f64],
) -> Vec<Vec<f64>> {
    let n = time.len();
    let n_features = coefficients.len();

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| time[b].partial_cmp(&time[a]).unwrap());

    let linear_pred: Vec<f64> = covariates
        .iter()
        .map(|x| {
            x.iter()
                .zip(coefficients.iter())
                .map(|(&xi, &bi)| xi * bi)
                .sum()
        })
        .collect();

    let exp_lp: Vec<f64> = linear_pred.iter().map(|&lp| lp.exp()).collect();

    let mut information = vec![vec![0.0; n_features]; n_features];

    let mut risk_sum = 0.0;
    let mut weighted_sum = vec![0.0; n_features];
    let mut weighted_outer = vec![vec![0.0; n_features]; n_features];

    for &i in &indices {
        risk_sum += exp_lp[i];
        for (j, &xij) in covariates[i].iter().enumerate() {
            weighted_sum[j] += xij * exp_lp[i];
            for (k, &xik) in covariates[i].iter().enumerate() {
                weighted_outer[j][k] += xij * xik * exp_lp[i];
            }
        }

        if event[i] == 1 && risk_sum > 0.0 {
            for j in 0..n_features {
                for k in 0..n_features {
                    information[j][k] += weighted_outer[j][k] / risk_sum
                        - (weighted_sum[j] * weighted_sum[k]) / (risk_sum * risk_sum);
                }
            }
        }
    }

    information
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct DPHistogramResult {
    #[pyo3(get)]
    pub counts: Vec<f64>,
    #[pyo3(get)]
    pub bin_edges: Vec<f64>,
    #[pyo3(get)]
    pub noise_added: Vec<f64>,
    #[pyo3(get)]
    pub epsilon_used: f64,
}

#[pymethods]
impl DPHistogramResult {
    fn __repr__(&self) -> String {
        format!(
            "DPHistogramResult(n_bins={}, epsilon={:.4})",
            self.counts.len(),
            self.epsilon_used
        )
    }

    fn normalized(&self) -> Vec<f64> {
        let total: f64 = self.counts.iter().sum();
        if total > 0.0 {
            self.counts.iter().map(|&c| c / total).collect()
        } else {
            vec![0.0; self.counts.len()]
        }
    }
}

#[pyfunction]
#[pyo3(signature = (
    values,
    n_bins,
    config,
    seed=None
))]
pub fn dp_histogram(
    values: Vec<f64>,
    n_bins: usize,
    config: DPConfig,
    seed: Option<u64>,
) -> PyResult<DPHistogramResult> {
    if values.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "values cannot be empty",
        ));
    }
    if n_bins == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n_bins must be positive",
        ));
    }

    let seed = seed.unwrap_or(42);

    let (min_val, max_val) = if let Some((lo, hi)) = config.clip_bounds {
        (lo, hi)
    } else {
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (min, max)
    };

    let bin_width = (max_val - min_val) / n_bins as f64;
    let bin_edges: Vec<f64> = (0..=n_bins)
        .map(|i| min_val + i as f64 * bin_width)
        .collect();

    let mut counts = vec![0.0; n_bins];
    for &v in &values {
        let v_clipped = v.clamp(min_val, max_val);
        let bin = ((v_clipped - min_val) / bin_width).floor() as usize;
        let bin = bin.min(n_bins - 1);
        counts[bin] += 1.0;
    }

    let per_bin_epsilon = config.epsilon / n_bins as f64;
    let scale = config.sensitivity / per_bin_epsilon;

    let mut noise_added = Vec::with_capacity(n_bins);
    for (i, count) in counts.iter_mut().enumerate() {
        let noise = match config.mechanism.as_str() {
            "gaussian" => {
                let sigma = config.sensitivity * (2.0 * (1.25 / config.delta).ln()).sqrt()
                    / per_bin_epsilon;
                generate_gaussian_noise(sigma, seed + i as u64)
            }
            _ => generate_laplace_noise(scale, seed + i as u64),
        };
        noise_added.push(noise);
        *count = (*count + noise).max(0.0);
    }

    Ok(DPHistogramResult {
        counts,
        bin_edges,
        noise_added,
        epsilon_used: config.epsilon,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct LocalDPResult {
    #[pyo3(get)]
    pub perturbed_values: Vec<f64>,
    #[pyo3(get)]
    pub estimated_mean: f64,
    #[pyo3(get)]
    pub estimated_variance: f64,
    #[pyo3(get)]
    pub epsilon_per_user: f64,
}

#[pymethods]
impl LocalDPResult {
    fn __repr__(&self) -> String {
        format!(
            "LocalDPResult(n={}, mean={:.4}, epsilon={:.4})",
            self.perturbed_values.len(),
            self.estimated_mean,
            self.epsilon_per_user
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    values,
    epsilon,
    bounds,
    seed=None
))]
pub fn local_dp_mean(
    values: Vec<f64>,
    epsilon: f64,
    bounds: (f64, f64),
    seed: Option<u64>,
) -> PyResult<LocalDPResult> {
    if values.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "values cannot be empty",
        ));
    }
    if epsilon <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "epsilon must be positive",
        ));
    }

    let (lo, hi) = bounds;
    let seed = seed.unwrap_or(42);

    let perturbed_values: Vec<f64> = values
        .par_iter()
        .enumerate()
        .map(|(i, &v)| {
            let v_clipped = v.clamp(lo, hi);
            let normalized = (v_clipped - lo) / (hi - lo);

            let mut rng_state = seed.wrapping_add(i as u64);
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (rng_state as f64) / (u64::MAX as f64);

            let p = (epsilon.exp() / (epsilon.exp() + 1.0)) * normalized
                + (1.0 / (epsilon.exp() + 1.0)) * (1.0 - normalized);

            let perturbed_bit = if u < p { 1.0 } else { 0.0 };

            lo + (hi - lo) * perturbed_bit
        })
        .collect();

    let raw_mean: f64 = perturbed_values.iter().sum::<f64>() / perturbed_values.len() as f64;
    let p_high = (raw_mean - lo) / (hi - lo);
    let corrected = (p_high * (epsilon.exp() + 1.0) - 1.0) / (epsilon.exp() - 1.0);
    let estimated_mean = lo + (hi - lo) * corrected.clamp(0.0, 1.0);

    let estimated_variance = perturbed_values
        .iter()
        .map(|&v| (v - estimated_mean).powi(2))
        .sum::<f64>()
        / perturbed_values.len() as f64;

    Ok(LocalDPResult {
        perturbed_values,
        estimated_mean,
        estimated_variance,
        epsilon_per_user: epsilon,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dp_kaplan_meier() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let event = vec![1, 0, 1, 0, 1];
        let config = DPConfig::new(1.0, 1e-5, 1.0, "gaussian", None).unwrap();

        let result = dp_kaplan_meier(time, event, config, Some(42)).unwrap();
        assert!(!result.survival_curve.is_empty());
        assert!(
            result
                .survival_curve
                .iter()
                .all(|&s| (0.0..=1.0).contains(&s))
        );
    }

    #[test]
    fn test_dp_cox_regression() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let event = vec![1, 0, 1, 0, 1];
        let covariates = vec![
            vec![0.5, 0.2],
            vec![0.3, 0.8],
            vec![0.7, 0.1],
            vec![0.2, 0.9],
            vec![0.8, 0.4],
        ];
        let config = DPConfig::new(1.0, 1e-5, 1.0, "gaussian", None).unwrap();

        let result = dp_cox_regression(time, event, covariates, config, 50, Some(42)).unwrap();
        assert_eq!(result.coefficients.len(), 2);
    }

    #[test]
    fn test_dp_histogram() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let config = DPConfig::new(1.0, 1e-5, 1.0, "laplace", Some((0.0, 100.0))).unwrap();

        let result = dp_histogram(values, 10, config, Some(42)).unwrap();
        assert_eq!(result.counts.len(), 10);
        assert_eq!(result.bin_edges.len(), 11);
    }

    #[test]
    fn test_local_dp_mean() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();

        let result = local_dp_mean(values, 1.0, (0.0, 100.0), Some(42)).unwrap();
        assert_eq!(result.perturbed_values.len(), 100);
        assert!((result.estimated_mean - 50.0).abs() < 30.0);
    }
}
