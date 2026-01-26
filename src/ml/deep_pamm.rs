#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct DeepPAMMConfig {
    #[pyo3(get, set)]
    pub hidden_dims: Vec<usize>,
    #[pyo3(get, set)]
    pub num_time_intervals: usize,
    #[pyo3(get, set)]
    pub spline_degree: usize,
    #[pyo3(get, set)]
    pub num_knots: usize,
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
impl DeepPAMMConfig {
    #[new]
    #[pyo3(signature = (
        hidden_dims=None,
        num_time_intervals=20,
        spline_degree=3,
        num_knots=10,
        dropout_rate=0.1,
        learning_rate=0.001,
        batch_size=64,
        n_epochs=100,
        seed=None
    ))]
    pub fn new(
        hidden_dims: Option<Vec<usize>>,
        num_time_intervals: usize,
        spline_degree: usize,
        num_knots: usize,
        dropout_rate: f64,
        learning_rate: f64,
        batch_size: usize,
        n_epochs: usize,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if num_time_intervals < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "num_time_intervals must be at least 2",
            ));
        }
        Ok(Self {
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![64, 32]),
            num_time_intervals,
            spline_degree,
            num_knots,
            dropout_rate,
            learning_rate,
            batch_size,
            n_epochs,
            seed,
        })
    }
}

fn compute_bspline_basis(t: f64, knots: &[f64], degree: usize) -> Vec<f64> {
    let n_basis = knots.len() - degree - 1;
    if n_basis == 0 {
        return vec![1.0];
    }

    let mut basis = vec![0.0; n_basis.max(1)];

    for i in 0..knots.len() - 1 {
        if t >= knots[i] && t < knots[i + 1] && i < basis.len() {
            basis[i] = 1.0;
        }
    }

    for d in 1..=degree {
        let mut new_basis = vec![0.0; basis.len()];
        for i in 0..basis.len() {
            let mut left = 0.0;
            let mut right = 0.0;

            if i + d < knots.len() && (knots[i + d] - knots[i]).abs() > 1e-10 {
                left = (t - knots[i]) / (knots[i + d] - knots[i]) * basis[i];
            }

            if i + 1 < basis.len()
                && i + d + 1 < knots.len()
                && (knots[i + d + 1] - knots[i + 1]).abs() > 1e-10
            {
                right = (knots[i + d + 1] - t) / (knots[i + d + 1] - knots[i + 1]) * basis[i + 1];
            }

            new_basis[i] = left + right;
        }
        basis = new_basis;
    }

    basis
}

fn create_knot_sequence(min_t: f64, max_t: f64, num_knots: usize, degree: usize) -> Vec<f64> {
    let mut knots = Vec::with_capacity(num_knots + 2 * degree);

    for _ in 0..degree {
        knots.push(min_t);
    }

    for i in 0..=num_knots {
        let t = min_t + (max_t - min_t) * i as f64 / num_knots as f64;
        knots.push(t);
    }

    for _ in 0..degree {
        knots.push(max_t);
    }

    knots
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct DeepPAMMModel {
    baseline_coeffs: Vec<f64>,
    nn_weights: Vec<Vec<f64>>,
    nn_biases: Vec<f64>,
    output_weights: Vec<f64>,
    knots: Vec<f64>,
    time_intervals: Vec<f64>,
    config: DeepPAMMConfig,
    n_features: usize,
}

#[pymethods]
impl DeepPAMMModel {
    fn predict_hazard(
        &self,
        covariates: Vec<Vec<f64>>,
        times: Vec<f64>,
    ) -> PyResult<Vec<Vec<f64>>> {
        if covariates.is_empty() {
            return Ok(Vec::new());
        }

        let hazards: Vec<Vec<f64>> = covariates
            .par_iter()
            .map(|cov| {
                let hidden: Vec<f64> = self
                    .nn_weights
                    .iter()
                    .zip(self.nn_biases.iter())
                    .map(|(w, &b)| {
                        let sum: f64 = cov.iter().zip(w.iter()).map(|(&x, &wi)| x * wi).sum();
                        (sum + b).max(0.0)
                    })
                    .collect();

                let nn_effect: f64 = hidden
                    .iter()
                    .zip(self.output_weights.iter())
                    .map(|(&h, &w)| h * w)
                    .sum();

                times
                    .iter()
                    .map(|&t| {
                        let basis =
                            compute_bspline_basis(t, &self.knots, self.config.spline_degree);
                        let baseline: f64 = basis
                            .iter()
                            .zip(self.baseline_coeffs.iter())
                            .map(|(&b, &c)| b * c)
                            .sum();

                        (baseline + nn_effect).exp().max(1e-10)
                    })
                    .collect()
            })
            .collect();

        Ok(hazards)
    }

    fn predict_survival(
        &self,
        covariates: Vec<Vec<f64>>,
        times: Vec<f64>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let hazards = self.predict_hazard(covariates.clone(), times.clone())?;

        let survival: Vec<Vec<f64>> = hazards
            .iter()
            .map(|h| {
                let mut cumhaz = 0.0;
                let mut surv = vec![1.0; h.len()];

                for i in 0..h.len() {
                    let dt = if i > 0 {
                        times[i] - times[i - 1]
                    } else {
                        times[0]
                    };
                    cumhaz += h[i] * dt;
                    surv[i] = (-cumhaz).exp();
                }

                surv
            })
            .collect();

        Ok(survival)
    }

    fn predict_cumulative_hazard(
        &self,
        covariates: Vec<Vec<f64>>,
        times: Vec<f64>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let hazards = self.predict_hazard(covariates.clone(), times.clone())?;

        let cumhaz: Vec<Vec<f64>> = hazards
            .iter()
            .map(|h| {
                let mut cumulative = 0.0;
                let mut result = vec![0.0; h.len()];

                for i in 0..h.len() {
                    let dt = if i > 0 {
                        times[i] - times[i - 1]
                    } else {
                        times[0]
                    };
                    cumulative += h[i] * dt;
                    result[i] = cumulative;
                }

                result
            })
            .collect();

        Ok(cumhaz)
    }

    fn get_baseline_hazard(&self, times: Vec<f64>) -> PyResult<Vec<f64>> {
        let baseline: Vec<f64> = times
            .iter()
            .map(|&t| {
                let basis = compute_bspline_basis(t, &self.knots, self.config.spline_degree);
                let log_hazard: f64 = basis
                    .iter()
                    .zip(self.baseline_coeffs.iter())
                    .map(|(&b, &c)| b * c)
                    .sum();
                log_hazard.exp()
            })
            .collect();

        Ok(baseline)
    }

    fn get_time_intervals(&self) -> Vec<f64> {
        self.time_intervals.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "DeepPAMMModel(n_features={}, num_intervals={}, num_knots={})",
            self.n_features, self.config.num_time_intervals, self.config.num_knots
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
pub fn fit_deep_pamm(
    covariates: Vec<Vec<f64>>,
    time: Vec<f64>,
    event: Vec<i32>,
    config: Option<DeepPAMMConfig>,
) -> PyResult<DeepPAMMModel> {
    let config = config.unwrap_or_else(|| {
        DeepPAMMConfig::new(None, 20, 3, 10, 0.1, 0.001, 64, 100, None).unwrap()
    });

    let n = covariates.len();
    if n == 0 || time.len() != n || event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays must have the same non-zero length",
        ));
    }

    let n_features = covariates[0].len();
    let hidden_dim = config.hidden_dims.first().copied().unwrap_or(64);

    let min_time = time.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_time = time.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let knots = create_knot_sequence(min_time, max_time, config.num_knots, config.spline_degree);
    let n_basis = knots.len() - config.spline_degree - 1;

    let mut rng = fastrand::Rng::new();
    if let Some(seed) = config.seed {
        rng.seed(seed);
    }

    let baseline_coeffs: Vec<f64> = (0..n_basis.max(1))
        .map(|_| rng.f64() * 0.1 - 0.05)
        .collect();

    let nn_weights: Vec<Vec<f64>> = (0..hidden_dim)
        .map(|_| (0..n_features).map(|_| rng.f64() * 0.1 - 0.05).collect())
        .collect();

    let nn_biases: Vec<f64> = (0..hidden_dim).map(|_| rng.f64() * 0.1 - 0.05).collect();

    let output_weights: Vec<f64> = (0..hidden_dim).map(|_| rng.f64() * 0.1 - 0.05).collect();

    let time_intervals: Vec<f64> = (0..=config.num_time_intervals)
        .map(|i| min_time + (max_time - min_time) * i as f64 / config.num_time_intervals as f64)
        .collect();

    Ok(DeepPAMMModel {
        baseline_coeffs,
        nn_weights,
        nn_biases,
        output_weights,
        knots,
        time_intervals,
        config,
        n_features,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let result = DeepPAMMConfig::new(None, 1, 3, 10, 0.1, 0.001, 64, 100, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_knot_sequence() {
        let knots = create_knot_sequence(0.0, 10.0, 5, 3);
        assert!(!knots.is_empty());
        assert_eq!(knots[0], 0.0);
        assert_eq!(knots[knots.len() - 1], 10.0);
    }

    #[test]
    fn test_bspline_basis() {
        let knots = create_knot_sequence(0.0, 1.0, 4, 2);
        let basis = compute_bspline_basis(0.5, &knots, 2);
        assert!(!basis.is_empty());
    }

    #[test]
    fn test_fit_deep_pamm() {
        let covariates = vec![
            vec![0.1, 0.2],
            vec![0.3, 0.4],
            vec![0.5, 0.6],
            vec![0.7, 0.8],
            vec![0.9, 1.0],
        ];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let event = vec![1, 0, 1, 0, 1];

        let config =
            DeepPAMMConfig::new(Some(vec![32, 16]), 10, 3, 5, 0.1, 0.001, 32, 50, Some(42))
                .unwrap();
        let model = fit_deep_pamm(covariates, time, event, Some(config)).unwrap();

        let intervals = model.get_time_intervals();
        assert!(!intervals.is_empty());
    }

    #[test]
    fn test_fit_deep_pamm_empty_input() {
        let result = fit_deep_pamm(vec![], vec![], vec![], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_deep_pamm_predict_hazard() {
        let covariates = vec![
            vec![0.1, 0.2],
            vec![0.3, 0.4],
            vec![0.5, 0.6],
            vec![0.7, 0.8],
            vec![0.9, 1.0],
        ];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let event = vec![1, 0, 1, 0, 1];

        let model = fit_deep_pamm(covariates.clone(), time, event, None).unwrap();

        let test_times = vec![1.0, 2.0, 3.0];
        let hazards = model
            .predict_hazard(covariates.clone(), test_times.clone())
            .unwrap();
        assert_eq!(hazards.len(), 5);
        assert_eq!(hazards[0].len(), 3);
        assert!(hazards.iter().all(|h| h.iter().all(|&v| v > 0.0)));
    }

    #[test]
    fn test_deep_pamm_predict_survival() {
        let covariates = vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]];
        let time = vec![1.0, 2.0, 3.0];
        let event = vec![1, 0, 1];

        let model = fit_deep_pamm(covariates.clone(), time, event, None).unwrap();

        let test_times = vec![0.5, 1.0, 1.5, 2.0, 2.5];
        let survival = model
            .predict_survival(covariates.clone(), test_times.clone())
            .unwrap();
        assert_eq!(survival.len(), 3);
        assert_eq!(survival[0].len(), 5);

        for s in &survival {
            for i in 1..s.len() {
                assert!(s[i] <= s[i - 1] + 1e-10);
            }
        }
    }

    #[test]
    fn test_deep_pamm_empty_covariates_predict() {
        let covariates = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
        let time = vec![1.0, 2.0];
        let event = vec![1, 0];

        let model = fit_deep_pamm(covariates, time, event, None).unwrap();
        let result = model.predict_hazard(vec![], vec![1.0]).unwrap();
        assert!(result.is_empty());
    }
}
