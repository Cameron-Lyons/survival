use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

use crate::constants::TIME_EPSILON;
use crate::internal::statistical::erf;
use crate::internal::validation::{validate_finite, validate_no_nan, validate_non_negative};

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(eq, eq_int, from_py_object)]
pub enum SojournDistribution {
    Exponential,
    Weibull,
    LogNormal,
    Gamma,
    GeneralizedGamma,
}

#[pymethods]
impl SojournDistribution {
    fn __repr__(&self) -> String {
        match self {
            SojournDistribution::Exponential => "SojournDistribution.Exponential".to_string(),
            SojournDistribution::Weibull => "SojournDistribution.Weibull".to_string(),
            SojournDistribution::LogNormal => "SojournDistribution.LogNormal".to_string(),
            SojournDistribution::Gamma => "SojournDistribution.Gamma".to_string(),
            SojournDistribution::GeneralizedGamma => {
                "SojournDistribution.GeneralizedGamma".to_string()
            }
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SemiMarkovConfig {
    #[pyo3(get, set)]
    pub n_states: usize,
    #[pyo3(get, set)]
    pub state_names: Vec<String>,
    #[pyo3(get, set)]
    pub sojourn_distributions: Vec<SojournDistribution>,
    #[pyo3(get, set)]
    pub absorbing_states: Vec<usize>,
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
}

#[pymethods]
impl SemiMarkovConfig {
    #[new]
    #[pyo3(signature = (n_states, state_names=None, sojourn_distributions=None, absorbing_states=None, max_iter=100, tol=1e-6))]
    pub fn new(
        n_states: usize,
        state_names: Option<Vec<String>>,
        sojourn_distributions: Option<Vec<SojournDistribution>>,
        absorbing_states: Option<Vec<usize>>,
        max_iter: usize,
        tol: f64,
    ) -> PyResult<Self> {
        let state_names =
            state_names.unwrap_or_else(|| (0..n_states).map(|i| format!("State_{}", i)).collect());

        let sojourn_distributions =
            sojourn_distributions.unwrap_or_else(|| vec![SojournDistribution::Weibull; n_states]);

        let absorbing_states = absorbing_states.unwrap_or_else(|| {
            if n_states == 0 {
                Vec::new()
            } else {
                vec![n_states - 1]
            }
        });

        validate_semi_markov_config_values(
            n_states,
            &state_names,
            &sojourn_distributions,
            &absorbing_states,
            max_iter,
            tol,
        )?;

        Ok(Self {
            n_states,
            state_names,
            sojourn_distributions,
            absorbing_states,
            max_iter,
            tol,
        })
    }
}

fn validate_semi_markov_config_values(
    n_states: usize,
    state_names: &[String],
    sojourn_distributions: &[SojournDistribution],
    absorbing_states: &[usize],
    max_iter: usize,
    tol: f64,
) -> PyResult<()> {
    if n_states == 0 {
        return Err(PyValueError::new_err("n_states must be positive"));
    }
    if state_names.len() != n_states {
        return Err(PyValueError::new_err(format!(
            "state_names length must equal n_states: expected {}, got {}",
            n_states,
            state_names.len()
        )));
    }
    if let Some((index, _)) = state_names
        .iter()
        .enumerate()
        .find(|(_, name)| name.trim().is_empty())
    {
        return Err(PyValueError::new_err(format!(
            "state_names entries must be non-empty; got empty name at index {}",
            index
        )));
    }
    if sojourn_distributions.len() != n_states {
        return Err(PyValueError::new_err(format!(
            "sojourn_distributions length must equal n_states: expected {}, got {}",
            n_states,
            sojourn_distributions.len()
        )));
    }
    let mut seen_absorbing_states = HashSet::with_capacity(absorbing_states.len());
    for &state in absorbing_states {
        if state >= n_states {
            return Err(PyValueError::new_err(format!(
                "absorbing_states must contain values in 0..n_states; got {}",
                state
            )));
        }
        if !seen_absorbing_states.insert(state) {
            return Err(PyValueError::new_err(format!(
                "absorbing_states must not contain duplicates; got {} more than once",
                state
            )));
        }
    }
    if max_iter == 0 {
        return Err(PyValueError::new_err("max_iter must be positive"));
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err(PyValueError::new_err(
            "tol must be finite and strictly positive",
        ));
    }
    Ok(())
}

fn validate_semi_markov_config(config: &SemiMarkovConfig) -> PyResult<()> {
    validate_semi_markov_config_values(
        config.n_states,
        &config.state_names,
        &config.sojourn_distributions,
        &config.absorbing_states,
        config.max_iter,
        config.tol,
    )
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SojournTimeParams {
    #[pyo3(get)]
    pub distribution: SojournDistribution,
    #[pyo3(get)]
    pub shape: f64,
    #[pyo3(get)]
    pub scale: f64,
    #[pyo3(get)]
    pub location: f64,
    #[pyo3(get)]
    pub mean: f64,
    #[pyo3(get)]
    pub variance: f64,
    #[pyo3(get)]
    pub median: f64,
}

#[pymethods]
impl SojournTimeParams {
    fn __repr__(&self) -> String {
        format!(
            "SojournTimeParams(dist={:?}, mean={:.3}, var={:.3})",
            self.distribution, self.mean, self.variance
        )
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SemiMarkovResult {
    #[pyo3(get)]
    pub transition_probs: HashMap<String, f64>,
    #[pyo3(get)]
    pub sojourn_params: Vec<SojournTimeParams>,
    #[pyo3(get)]
    pub state_occupation_probs: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub mean_sojourn_times: Vec<f64>,
    #[pyo3(get)]
    pub n_transitions: HashMap<String, usize>,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub aic: f64,
    #[pyo3(get)]
    pub bic: f64,
}

#[pymethods]
impl SemiMarkovResult {
    fn __repr__(&self) -> String {
        format!(
            "SemiMarkovResult(n_states={}, ll={:.2}, aic={:.2})",
            self.sojourn_params.len(),
            self.log_likelihood,
            self.aic
        )
    }

    fn get_transition_prob(&self, from_state: usize, to_state: usize) -> f64 {
        let key = format!("{}_{}", from_state, to_state);
        *self.transition_probs.get(&key).unwrap_or(&0.0)
    }

    fn predict_state_at_time(&self, time: f64) -> Vec<f64> {
        if self.time_points.is_empty() {
            return vec![0.0; self.sojourn_params.len()];
        }

        let idx = self
            .time_points
            .iter()
            .position(|&t| t >= time)
            .unwrap_or(self.time_points.len() - 1);

        self.state_occupation_probs[idx].clone()
    }
}

fn weibull_pdf(t: f64, shape: f64, scale: f64) -> f64 {
    if t <= 0.0 || shape <= 0.0 || scale <= 0.0 {
        return 0.0;
    }
    (shape / scale) * (t / scale).powf(shape - 1.0) * (-(t / scale).powf(shape)).exp()
}

fn weibull_cdf(t: f64, shape: f64, scale: f64) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }
    1.0 - (-(t / scale).powf(shape)).exp()
}

fn weibull_survival(t: f64, shape: f64, scale: f64) -> f64 {
    if t <= 0.0 {
        return 1.0;
    }
    (-(t / scale).powf(shape)).exp()
}

fn lognormal_pdf(t: f64, mu: f64, sigma: f64) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        return 0.0;
    }
    let log_t = t.ln();
    (1.0 / (t * sigma * (2.0 * std::f64::consts::PI).sqrt()))
        * (-0.5 * ((log_t - mu) / sigma).powi(2)).exp()
}

fn lognormal_cdf(t: f64, mu: f64, sigma: f64) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }
    0.5 * (1.0 + erf((t.ln() - mu) / (sigma * std::f64::consts::SQRT_2)))
}

fn gamma_pdf(t: f64, shape: f64, rate: f64) -> f64 {
    if t <= 0.0 || shape <= 0.0 || rate <= 0.0 {
        return 0.0;
    }
    let ln_gamma = ln_gamma_fn(shape);
    (shape * rate.ln() + (shape - 1.0) * t.ln() - rate * t - ln_gamma).exp()
}

fn ln_gamma_fn(x: f64) -> f64 {
    let coeffs = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];

    let tmp = x + 5.5;
    let tmp = tmp - (x + 0.5) * tmp.ln();
    let mut ser = 1.000000000190015;
    for (i, &coeff) in coeffs.iter().enumerate() {
        ser += coeff / (x + 1.0 + i as f64);
    }
    -tmp + (2.5066282746310005 * ser / x).ln()
}

fn fit_weibull_mle(times: &[f64]) -> (f64, f64) {
    if times.is_empty() {
        return (1.0, 1.0);
    }

    let n = times.len() as f64;

    let mut shape = 1.0;
    for _ in 0..50 {
        let sum_t_k: f64 = times.iter().map(|&t| t.powf(shape)).sum();
        let sum_t_k_ln_t: f64 = times.iter().map(|&t| t.powf(shape) * t.ln()).sum();
        let sum_ln_t: f64 = times.iter().map(|&t| t.ln()).sum();

        if sum_t_k.abs() < 1e-10 {
            break;
        }

        let f = sum_t_k_ln_t / sum_t_k - 1.0 / shape - sum_ln_t / n;
        let df = -1.0 / shape.powi(2);

        let new_shape = shape - f / df;
        if (new_shape - shape).abs() < 1e-6 {
            shape = new_shape.max(0.1);
            break;
        }
        shape = new_shape.max(0.1);
    }

    let scale = (times.iter().map(|&t| t.powf(shape)).sum::<f64>() / n).powf(1.0 / shape);

    (shape, scale.max(1e-10))
}

fn fit_lognormal_mle(times: &[f64]) -> (f64, f64) {
    if times.is_empty() {
        return (0.0, 1.0);
    }

    let log_times: Vec<f64> = times.iter().map(|&t| t.max(1e-10).ln()).collect();
    let n = log_times.len() as f64;
    let mu = log_times.iter().sum::<f64>() / n;
    let sigma = (log_times.iter().map(|&lt| (lt - mu).powi(2)).sum::<f64>() / n)
        .sqrt()
        .max(0.01);

    (mu, sigma)
}

fn fit_gamma_mle(times: &[f64]) -> (f64, f64) {
    if times.is_empty() {
        return (1.0, 1.0);
    }

    let n = times.len() as f64;
    let mean = times.iter().sum::<f64>() / n;
    let log_mean = times.iter().map(|&t| t.max(1e-10).ln()).sum::<f64>() / n;

    let s = mean.ln() - log_mean;
    let shape = if s > 0.0 {
        (3.0 - s + ((s - 3.0).powi(2) + 24.0 * s).sqrt()) / (12.0 * s)
    } else {
        1.0
    };

    let rate = shape / mean;

    (shape.max(0.1), rate.max(0.01))
}

fn validate_state_sequence(field: &'static str, states: &[i32], n_states: usize) -> PyResult<()> {
    for (index, &state) in states.iter().enumerate() {
        if state < 0 || state as usize >= n_states {
            return Err(PyValueError::new_err(format!(
                "{} must contain values in 0..n_states; got {} at index {}",
                field, state, index
            )));
        }
    }
    Ok(())
}

fn validate_semi_markov_fit_inputs(
    entry_times: &[f64],
    exit_times: &[f64],
    from_states: &[i32],
    to_states: &[i32],
    config: &SemiMarkovConfig,
) -> PyResult<()> {
    validate_semi_markov_config(config)?;

    let n = entry_times.len();
    if exit_times.len() != n || from_states.len() != n || to_states.len() != n {
        return Err(PyValueError::new_err(
            "All input vectors must have the same length",
        ));
    }
    if n == 0 {
        return Err(PyValueError::new_err("input vectors must be non-empty"));
    }

    validate_no_nan(entry_times, "entry_times")?;
    validate_finite(entry_times, "entry_times")?;
    validate_non_negative(entry_times, "entry_times")?;
    validate_no_nan(exit_times, "exit_times")?;
    validate_finite(exit_times, "exit_times")?;
    validate_non_negative(exit_times, "exit_times")?;

    for (index, (&entry_time, &exit_time)) in entry_times.iter().zip(exit_times.iter()).enumerate()
    {
        if entry_time > exit_time + TIME_EPSILON {
            return Err(PyValueError::new_err(format!(
                "entry_times must be <= exit_times; got entry_time {} and exit_time {} at index {}",
                entry_time, exit_time, index
            )));
        }
    }

    validate_state_sequence("from_states", from_states, config.n_states)?;
    validate_state_sequence("to_states", to_states, config.n_states)?;

    Ok(())
}

#[pyfunction]
#[pyo3(signature = (entry_times, exit_times, from_states, to_states, config))]
pub fn fit_semi_markov(
    entry_times: Vec<f64>,
    exit_times: Vec<f64>,
    from_states: Vec<i32>,
    to_states: Vec<i32>,
    config: &SemiMarkovConfig,
) -> PyResult<SemiMarkovResult> {
    validate_semi_markov_fit_inputs(&entry_times, &exit_times, &from_states, &to_states, config)?;

    let n = entry_times.len();

    let sojourn_times: Vec<f64> = entry_times
        .iter()
        .zip(exit_times.iter())
        .map(|(&entry, &exit)| (exit - entry).max(1e-10))
        .collect();

    let mut transition_counts: HashMap<String, usize> = HashMap::new();
    let mut state_counts: Vec<usize> = vec![0; config.n_states];

    for i in 0..n {
        let from = from_states[i] as usize;
        let to = to_states[i] as usize;
        if from < config.n_states && to < config.n_states {
            let key = format!("{}_{}", from, to);
            *transition_counts.entry(key).or_insert(0) += 1;
            state_counts[from] += 1;
        }
    }

    let mut transition_probs: HashMap<String, f64> = HashMap::new();
    for (from, &state_count) in state_counts.iter().enumerate().take(config.n_states) {
        for to in 0..config.n_states {
            let key = format!("{}_{}", from, to);
            let count = *transition_counts.get(&key).unwrap_or(&0);
            let prob = if state_count > 0 {
                count as f64 / state_count as f64
            } else {
                0.0
            };
            transition_probs.insert(key, prob);
        }
    }

    let mut sojourn_params: Vec<SojournTimeParams> = Vec::new();
    let mut log_likelihood = 0.0;
    let mut n_params = 0;

    for state in 0..config.n_states {
        let state_sojourn: Vec<f64> = (0..n)
            .filter(|&i| from_states[i] as usize == state)
            .map(|i| sojourn_times[i])
            .collect();

        let dist = config.sojourn_distributions[state];
        let (shape, scale, location, mean, variance, median) = if state_sojourn.is_empty() {
            (1.0, 1.0, 0.0, 1.0, 1.0, 1.0)
        } else {
            match dist {
                SojournDistribution::Exponential => {
                    let mean = state_sojourn.iter().sum::<f64>() / state_sojourn.len() as f64;
                    let rate = 1.0 / mean;
                    for &t in &state_sojourn {
                        log_likelihood += rate.ln() - rate * t;
                    }
                    n_params += 1;
                    (1.0, mean, 0.0, mean, mean.powi(2), mean * 2.0_f64.ln())
                }
                SojournDistribution::Weibull => {
                    let (shape, scale) = fit_weibull_mle(&state_sojourn);
                    for &t in &state_sojourn {
                        let pdf = weibull_pdf(t, shape, scale);
                        if pdf > 1e-300 {
                            log_likelihood += pdf.ln();
                        }
                    }
                    n_params += 2;
                    let mean = scale * ln_gamma_fn(1.0 + 1.0 / shape).exp();
                    let var = scale.powi(2)
                        * (ln_gamma_fn(1.0 + 2.0 / shape).exp()
                            - ln_gamma_fn(1.0 + 1.0 / shape).exp().powi(2));
                    let median = scale * 2.0_f64.ln().powf(1.0 / shape);
                    (shape, scale, 0.0, mean, var, median)
                }
                SojournDistribution::LogNormal => {
                    let (mu, sigma) = fit_lognormal_mle(&state_sojourn);
                    for &t in &state_sojourn {
                        let pdf = lognormal_pdf(t, mu, sigma);
                        if pdf > 1e-300 {
                            log_likelihood += pdf.ln();
                        }
                    }
                    n_params += 2;
                    let mean = (mu + sigma.powi(2) / 2.0).exp();
                    let var = (sigma.powi(2).exp() - 1.0) * (2.0 * mu + sigma.powi(2)).exp();
                    let median = mu.exp();
                    (sigma, mu.exp(), mu, mean, var, median)
                }
                SojournDistribution::Gamma => {
                    let (shape, rate) = fit_gamma_mle(&state_sojourn);
                    for &t in &state_sojourn {
                        let pdf = gamma_pdf(t, shape, rate);
                        if pdf > 1e-300 {
                            log_likelihood += pdf.ln();
                        }
                    }
                    n_params += 2;
                    let mean = shape / rate;
                    let var = shape / rate.powi(2);
                    let median = mean * (1.0 - 1.0 / (9.0 * shape)).powi(3);
                    (shape, 1.0 / rate, 0.0, mean, var, median)
                }
                SojournDistribution::GeneralizedGamma => {
                    let (shape, scale) = fit_weibull_mle(&state_sojourn);
                    for &t in &state_sojourn {
                        let pdf = weibull_pdf(t, shape, scale);
                        if pdf > 1e-300 {
                            log_likelihood += pdf.ln();
                        }
                    }
                    n_params += 3;
                    let mean = scale * ln_gamma_fn(1.0 + 1.0 / shape).exp();
                    let var = scale.powi(2)
                        * (ln_gamma_fn(1.0 + 2.0 / shape).exp()
                            - ln_gamma_fn(1.0 + 1.0 / shape).exp().powi(2));
                    let median = scale * 2.0_f64.ln().powf(1.0 / shape);
                    (shape, scale, 0.0, mean, var, median)
                }
            }
        };

        sojourn_params.push(SojournTimeParams {
            distribution: dist,
            shape,
            scale,
            location,
            mean,
            variance,
            median,
        });
    }

    let max_time = exit_times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let n_time_points = 100;
    let time_points: Vec<f64> = (0..=n_time_points)
        .map(|i| i as f64 * max_time / n_time_points as f64)
        .collect();

    let mut state_occupation_probs: Vec<Vec<f64>> = Vec::new();
    for &t in &time_points {
        let mut probs = vec![0.0; config.n_states];
        if t == 0.0 {
            probs[0] = 1.0;
        } else {
            for state in 0..config.n_states {
                if config.absorbing_states.contains(&state) {
                    let mut absorb_prob = 0.0;
                    for (from, params) in sojourn_params.iter().enumerate().take(config.n_states) {
                        if !config.absorbing_states.contains(&from) {
                            let key = format!("{}_{}", from, state);
                            let trans_prob = *transition_probs.get(&key).unwrap_or(&0.0);
                            let cdf = match params.distribution {
                                SojournDistribution::Weibull
                                | SojournDistribution::GeneralizedGamma => {
                                    weibull_cdf(t, params.shape, params.scale)
                                }
                                SojournDistribution::Exponential => 1.0 - (-t / params.scale).exp(),
                                SojournDistribution::LogNormal => {
                                    lognormal_cdf(t, params.location, params.shape)
                                }
                                SojournDistribution::Gamma => {
                                    1.0 - (-t * params.shape / params.scale).exp()
                                }
                            };
                            absorb_prob += trans_prob * cdf;
                        }
                    }
                    probs[state] = absorb_prob.min(1.0);
                } else {
                    let params = &sojourn_params[state];
                    let surv = match params.distribution {
                        SojournDistribution::Weibull | SojournDistribution::GeneralizedGamma => {
                            weibull_survival(t, params.shape, params.scale)
                        }
                        SojournDistribution::Exponential => (-t / params.scale).exp(),
                        SojournDistribution::LogNormal => {
                            1.0 - lognormal_cdf(t, params.location, params.shape)
                        }
                        SojournDistribution::Gamma => (-t * params.shape / params.scale).exp(),
                    };
                    probs[state] = surv * (1.0 - probs.iter().sum::<f64>()).max(0.0);
                }
            }
        }

        let sum: f64 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
        }
        state_occupation_probs.push(probs);
    }

    let mean_sojourn_times: Vec<f64> = sojourn_params.iter().map(|p| p.mean).collect();

    let n_obs = n as f64;
    let aic = -2.0 * log_likelihood + 2.0 * n_params as f64;
    let bic = -2.0 * log_likelihood + (n_params as f64) * n_obs.ln();

    Ok(SemiMarkovResult {
        transition_probs,
        sojourn_params,
        state_occupation_probs,
        time_points,
        mean_sojourn_times,
        n_transitions: transition_counts,
        log_likelihood,
        aic,
        bic,
    })
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SemiMarkovPrediction {
    #[pyo3(get)]
    pub state_probs: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub expected_sojourn: Vec<f64>,
    #[pyo3(get)]
    pub transition_hazards: HashMap<String, Vec<f64>>,
}

fn validate_semi_markov_prediction_inputs(
    model: &SemiMarkovResult,
    current_state: usize,
    time_in_state: f64,
    prediction_times: &[f64],
) -> PyResult<()> {
    let n_states = model.sojourn_params.len();
    if n_states == 0 {
        return Err(PyValueError::new_err(
            "model must contain at least one state",
        ));
    }
    if current_state >= n_states {
        return Err(PyValueError::new_err(
            "current_state must be less than number of states",
        ));
    }
    if !time_in_state.is_finite() {
        return Err(PyValueError::new_err("time_in_state must be finite"));
    }
    if time_in_state < 0.0 {
        return Err(PyValueError::new_err("time_in_state must be non-negative"));
    }

    validate_no_nan(prediction_times, "prediction_times")?;
    validate_finite(prediction_times, "prediction_times")?;
    validate_non_negative(prediction_times, "prediction_times")?;

    Ok(())
}

#[pyfunction]
#[pyo3(signature = (model, current_state, time_in_state, prediction_times))]
pub fn predict_semi_markov(
    model: &SemiMarkovResult,
    current_state: usize,
    time_in_state: f64,
    prediction_times: Vec<f64>,
) -> PyResult<SemiMarkovPrediction> {
    validate_semi_markov_prediction_inputs(model, current_state, time_in_state, &prediction_times)?;

    let n_states = model.sojourn_params.len();

    let params = &model.sojourn_params[current_state];
    let current_survival = match params.distribution {
        SojournDistribution::Weibull | SojournDistribution::GeneralizedGamma => {
            weibull_survival(time_in_state, params.shape, params.scale)
        }
        SojournDistribution::Exponential => (-time_in_state / params.scale).exp(),
        _ => 1.0,
    };

    let mut state_probs: Vec<Vec<f64>> = Vec::new();
    let mut transition_hazards: HashMap<String, Vec<f64>> = HashMap::new();

    for to_state in 0..n_states {
        let key = format!("{}_{}", current_state, to_state);
        transition_hazards.insert(key.clone(), Vec::new());
    }

    for &t in &prediction_times {
        let total_time = time_in_state + t;
        let mut probs = vec![0.0; n_states];

        let future_survival = match params.distribution {
            SojournDistribution::Weibull | SojournDistribution::GeneralizedGamma => {
                weibull_survival(total_time, params.shape, params.scale)
            }
            SojournDistribution::Exponential => (-total_time / params.scale).exp(),
            _ => 1.0,
        };

        let conditional_survival = if current_survival > 1e-10 {
            future_survival / current_survival
        } else {
            0.0
        };

        probs[current_state] = conditional_survival;

        let exit_prob = 1.0 - conditional_survival;
        for (to_state, prob) in probs.iter_mut().enumerate().take(n_states) {
            if to_state != current_state {
                let key = format!("{}_{}", current_state, to_state);
                let trans_prob = *model.transition_probs.get(&key).unwrap_or(&0.0);
                *prob = exit_prob * trans_prob;
            }
        }

        state_probs.push(probs);

        for to_state in 0..n_states {
            let key = format!("{}_{}", current_state, to_state);
            let trans_prob = *model.transition_probs.get(&key).unwrap_or(&0.0);

            let hazard = if conditional_survival > 1e-10 {
                let pdf = match params.distribution {
                    SojournDistribution::Weibull | SojournDistribution::GeneralizedGamma => {
                        weibull_pdf(total_time, params.shape, params.scale)
                    }
                    SojournDistribution::Exponential => {
                        (1.0 / params.scale) * (-total_time / params.scale).exp()
                    }
                    _ => 0.0,
                };
                trans_prob * pdf / future_survival
            } else {
                0.0
            };

            if let Some(hazards) = transition_hazards.get_mut(&key) {
                hazards.push(hazard);
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "internal error: missing transition hazard bucket",
                ));
            }
        }
    }

    let expected_sojourn: Vec<f64> = model.sojourn_params.iter().map(|p| p.mean).collect();

    Ok(SemiMarkovPrediction {
        state_probs,
        time_points: prediction_times,
        expected_sojourn,
        transition_hazards,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semi_markov_config() {
        let config = SemiMarkovConfig::new(3, None, None, None, 100, 1e-6).unwrap();
        assert_eq!(config.n_states, 3);
        assert_eq!(config.state_names.len(), 3);
        assert_eq!(config.sojourn_distributions.len(), 3);

        assert!(SemiMarkovConfig::new(0, None, None, None, 100, 1e-6).is_err());
        assert!(
            SemiMarkovConfig::new(3, Some(vec!["Only".to_string()]), None, None, 100, 1e-6)
                .is_err()
        );
        assert!(
            SemiMarkovConfig::new(
                3,
                None,
                Some(vec![SojournDistribution::Weibull]),
                None,
                100,
                1e-6,
            )
            .is_err()
        );
        assert!(SemiMarkovConfig::new(3, None, None, Some(vec![3]), 100, 1e-6).is_err());
        assert!(SemiMarkovConfig::new(3, None, None, None, 0, 1e-6).is_err());
        assert!(SemiMarkovConfig::new(3, None, None, None, 100, f64::NAN).is_err());
    }

    #[test]
    fn test_weibull_functions() {
        let pdf = weibull_pdf(1.0, 2.0, 1.0);
        assert!(pdf > 0.0 && pdf < 1.0);

        let cdf = weibull_cdf(1.0, 2.0, 1.0);
        assert!(cdf > 0.0 && cdf < 1.0);

        let surv = weibull_survival(1.0, 2.0, 1.0);
        assert!((surv + cdf - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fit_semi_markov() {
        let entry_times = vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.5, 2.5, 3.5];
        let exit_times = vec![1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 5.0];
        let from_states = vec![0, 0, 1, 1, 0, 0, 1, 1];
        let to_states = vec![1, 1, 2, 2, 1, 1, 2, 2];

        let config = SemiMarkovConfig::new(3, None, None, Some(vec![2]), 100, 1e-6).unwrap();
        let result =
            fit_semi_markov(entry_times, exit_times, from_states, to_states, &config).unwrap();

        assert_eq!(result.sojourn_params.len(), 3);
        assert!(!result.transition_probs.is_empty());
        assert!(result.log_likelihood.is_finite());
    }

    #[test]
    fn test_fit_semi_markov_rejects_malformed_inputs() {
        let config = SemiMarkovConfig::new(3, None, None, Some(vec![2]), 100, 1e-6).unwrap();

        let err = fit_semi_markov(vec![], vec![], vec![], vec![], &config).unwrap_err();
        assert!(err.to_string().contains("input vectors must be non-empty"));

        let err =
            fit_semi_markov(vec![0.0], vec![f64::INFINITY], vec![0], vec![1], &config).unwrap_err();
        assert!(err.to_string().contains("exit_times contains non-finite"));

        let err = fit_semi_markov(vec![2.0], vec![1.0], vec![0], vec![1], &config).unwrap_err();
        assert!(
            err.to_string()
                .contains("entry_times must be <= exit_times")
        );

        let err = fit_semi_markov(vec![0.0], vec![1.0], vec![-1], vec![1], &config).unwrap_err();
        assert!(
            err.to_string()
                .contains("from_states must contain values in 0..n_states")
        );

        let err = fit_semi_markov(vec![0.0], vec![1.0], vec![0], vec![3], &config).unwrap_err();
        assert!(
            err.to_string()
                .contains("to_states must contain values in 0..n_states")
        );

        let mutated_config = SemiMarkovConfig {
            n_states: 3,
            state_names: vec![
                "State_0".to_string(),
                "State_1".to_string(),
                "State_2".to_string(),
            ],
            sojourn_distributions: vec![SojournDistribution::Weibull],
            absorbing_states: vec![2],
            max_iter: 100,
            tol: 1e-6,
        };
        let err =
            fit_semi_markov(vec![0.0], vec![1.0], vec![0], vec![1], &mutated_config).unwrap_err();
        assert!(err.to_string().contains("sojourn_distributions length"));
    }

    #[test]
    fn test_predict_semi_markov() {
        let entry_times = vec![0.0, 1.0, 2.0, 3.0];
        let exit_times = vec![1.0, 2.0, 3.0, 4.0];
        let from_states = vec![0, 0, 1, 1];
        let to_states = vec![1, 1, 2, 2];

        let config = SemiMarkovConfig::new(3, None, None, Some(vec![2]), 100, 1e-6).unwrap();
        let model =
            fit_semi_markov(entry_times, exit_times, from_states, to_states, &config).unwrap();

        let prediction = predict_semi_markov(&model, 0, 0.5, vec![0.5, 1.0, 1.5, 2.0]).unwrap();

        assert_eq!(prediction.state_probs.len(), 4);
        assert_eq!(prediction.time_points.len(), 4);
    }

    #[test]
    fn test_predict_semi_markov_rejects_malformed_inputs() {
        let entry_times = vec![0.0, 1.0, 2.0, 3.0];
        let exit_times = vec![1.0, 2.0, 3.0, 4.0];
        let from_states = vec![0, 0, 1, 1];
        let to_states = vec![1, 1, 2, 2];

        let config = SemiMarkovConfig::new(3, None, None, Some(vec![2]), 100, 1e-6).unwrap();
        let model =
            fit_semi_markov(entry_times, exit_times, from_states, to_states, &config).unwrap();

        let err = predict_semi_markov(&model, 3, 0.5, vec![1.0]).unwrap_err();
        assert!(
            err.to_string()
                .contains("current_state must be less than number of states")
        );

        let err = predict_semi_markov(&model, 0, f64::INFINITY, vec![1.0]).unwrap_err();
        assert!(err.to_string().contains("time_in_state must be finite"));

        let err = predict_semi_markov(&model, 0, -0.5, vec![1.0]).unwrap_err();
        assert!(
            err.to_string()
                .contains("time_in_state must be non-negative")
        );

        let err = predict_semi_markov(&model, 0, 0.5, vec![-1.0]).unwrap_err();
        assert!(
            err.to_string()
                .contains("prediction_times contains negative value")
        );

        let err = predict_semi_markov(&model, 0, 0.5, vec![f64::INFINITY]).unwrap_err();
        assert!(
            err.to_string()
                .contains("prediction_times contains non-finite")
        );
    }
}
