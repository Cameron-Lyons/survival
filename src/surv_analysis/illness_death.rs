use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::constants::TIME_EPSILON;
use crate::internal::statistical::normal_cdf;
use crate::internal::validation::{validate_finite, validate_no_nan, validate_non_negative};

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(eq, eq_int, from_py_object)]
pub enum IllnessDeathType {
    Progressive,
    Reversible,
    MarkovProgressive,
    SemiMarkovProgressive,
}

#[pymethods]
impl IllnessDeathType {
    fn __repr__(&self) -> String {
        match self {
            IllnessDeathType::Progressive => "IllnessDeathType.Progressive".to_string(),
            IllnessDeathType::Reversible => "IllnessDeathType.Reversible".to_string(),
            IllnessDeathType::MarkovProgressive => "IllnessDeathType.MarkovProgressive".to_string(),
            IllnessDeathType::SemiMarkovProgressive => {
                "IllnessDeathType.SemiMarkovProgressive".to_string()
            }
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct IllnessDeathConfig {
    #[pyo3(get, set)]
    pub model_type: IllnessDeathType,
    #[pyo3(get, set)]
    pub state_names: Vec<String>,
    #[pyo3(get, set)]
    pub clock_type: String,
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
    #[pyo3(get, set)]
    pub n_bootstrap: usize,
}

#[pymethods]
impl IllnessDeathConfig {
    #[new]
    #[pyo3(signature = (model_type=IllnessDeathType::Progressive, state_names=None, clock_type="forward", max_iter=100, tol=1e-6, n_bootstrap=0))]
    pub fn new(
        model_type: IllnessDeathType,
        state_names: Option<Vec<String>>,
        clock_type: &str,
        max_iter: usize,
        tol: f64,
        n_bootstrap: usize,
    ) -> PyResult<Self> {
        let state_names = state_names.unwrap_or_else(|| {
            vec![
                "Healthy".to_string(),
                "Illness".to_string(),
                "Death".to_string(),
            ]
        });

        if !["forward", "backward", "gap"].contains(&clock_type) {
            return Err(PyValueError::new_err(
                "clock_type must be one of: forward, backward, gap",
            ));
        }
        validate_illness_death_config_values(&state_names, max_iter, tol)?;

        Ok(Self {
            model_type,
            state_names,
            clock_type: clock_type.to_string(),
            max_iter,
            tol,
            n_bootstrap,
        })
    }
}

fn validate_illness_death_config(config: &IllnessDeathConfig) -> PyResult<()> {
    if !["forward", "backward", "gap"].contains(&config.clock_type.as_str()) {
        return Err(PyValueError::new_err(
            "clock_type must be one of: forward, backward, gap",
        ));
    }
    validate_illness_death_config_values(&config.state_names, config.max_iter, config.tol)
}

fn validate_illness_death_config_values(
    state_names: &[String],
    max_iter: usize,
    tol: f64,
) -> PyResult<()> {
    if state_names.len() < 3 {
        return Err(PyValueError::new_err(
            "state_names must contain at least 3 states",
        ));
    }
    if state_names
        .iter()
        .take(3)
        .any(|name| name.trim().is_empty())
    {
        return Err(PyValueError::new_err(
            "state_names must not contain empty names",
        ));
    }
    if max_iter == 0 {
        return Err(PyValueError::new_err("max_iter must be positive"));
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err(PyValueError::new_err("tol must be finite and positive"));
    }
    Ok(())
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct TransitionHazard {
    #[pyo3(get)]
    pub from_state: String,
    #[pyo3(get)]
    pub to_state: String,
    #[pyo3(get)]
    pub coefficient: f64,
    #[pyo3(get)]
    pub se: f64,
    #[pyo3(get)]
    pub hazard_ratio: f64,
    #[pyo3(get)]
    pub ci_lower: f64,
    #[pyo3(get)]
    pub ci_upper: f64,
    #[pyo3(get)]
    pub p_value: f64,
    #[pyo3(get)]
    pub baseline_hazard: Vec<f64>,
    #[pyo3(get)]
    pub baseline_times: Vec<f64>,
}

#[pymethods]
impl TransitionHazard {
    fn __repr__(&self) -> String {
        format!(
            "TransitionHazard({} -> {}: HR={:.3}, p={:.4})",
            self.from_state, self.to_state, self.hazard_ratio, self.p_value
        )
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct IllnessDeathResult {
    #[pyo3(get)]
    pub transition_hazards: Vec<TransitionHazard>,
    #[pyo3(get)]
    pub state_occupation_probs: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub cumulative_incidence: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub sojourn_times: Vec<f64>,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub aic: f64,
    #[pyo3(get)]
    pub bic: f64,
    #[pyo3(get)]
    pub n_transitions: Vec<usize>,
    #[pyo3(get)]
    pub model_type: IllnessDeathType,
}

#[pymethods]
impl IllnessDeathResult {
    fn __repr__(&self) -> String {
        format!(
            "IllnessDeathResult(type={:?}, ll={:.2}, aic={:.2})",
            self.model_type, self.log_likelihood, self.aic
        )
    }

    fn get_survival_probability(&self, time: f64) -> f64 {
        if self.time_points.is_empty() {
            return 1.0;
        }

        let idx = self
            .time_points
            .iter()
            .position(|&t| t >= time)
            .unwrap_or(self.time_points.len() - 1);

        self.state_occupation_probs[idx][0] + self.state_occupation_probs[idx][1]
    }

    fn get_illness_probability(&self, time: f64) -> f64 {
        if self.time_points.is_empty() {
            return 0.0;
        }

        let idx = self
            .time_points
            .iter()
            .position(|&t| t >= time)
            .unwrap_or(self.time_points.len() - 1);

        self.state_occupation_probs[idx][1]
    }

    fn get_death_probability(&self, time: f64) -> f64 {
        if self.time_points.is_empty() {
            return 0.0;
        }

        let idx = self
            .time_points
            .iter()
            .position(|&t| t >= time)
            .unwrap_or(self.time_points.len() - 1);

        self.state_occupation_probs[idx][2]
    }
}

#[pyfunction]
#[pyo3(signature = (entry_time, transition_time, exit_time, from_state, to_state, covariates=None, config=None))]
pub fn fit_illness_death(
    entry_time: Vec<f64>,
    transition_time: Vec<f64>,
    exit_time: Vec<f64>,
    from_state: Vec<i32>,
    to_state: Vec<i32>,
    covariates: Option<Vec<Vec<f64>>>,
    config: Option<IllnessDeathConfig>,
) -> PyResult<IllnessDeathResult> {
    let config = match config {
        Some(config) => config,
        None => {
            IllnessDeathConfig::new(IllnessDeathType::Progressive, None, "forward", 100, 1e-6, 0)?
        }
    };
    validate_illness_death_config(&config)?;

    let n = entry_time.len();
    validate_illness_death_fit_inputs(
        &entry_time,
        &transition_time,
        &exit_time,
        &from_state,
        &to_state,
        covariates.as_deref(),
    )?;

    let n_covariates = covariates
        .as_ref()
        .map(|c| c.first().map(|v| v.len()).unwrap_or(0))
        .unwrap_or(0);

    let mut trans_01_times: Vec<f64> = Vec::new();
    let mut trans_01_events: Vec<bool> = Vec::new();
    let mut trans_01_covariates: Vec<Vec<f64>> = Vec::new();
    let mut trans_02_times: Vec<f64> = Vec::new();
    let mut trans_02_events: Vec<bool> = Vec::new();
    let mut trans_02_covariates: Vec<Vec<f64>> = Vec::new();
    let mut trans_12_times: Vec<f64> = Vec::new();
    let mut trans_12_events: Vec<bool> = Vec::new();
    let mut trans_12_covariates: Vec<Vec<f64>> = Vec::new();

    for i in 0..n {
        let from = from_state[i];
        let to = to_state[i];
        let cov_row = covariates
            .as_ref()
            .map(|cov| cov[i].clone())
            .unwrap_or_default();

        if from == 0 {
            if to == 1 {
                trans_01_times.push(transition_time[i] - entry_time[i]);
                trans_01_events.push(true);
                trans_01_covariates.push(cov_row);
            } else if to == 2 {
                trans_02_times.push(exit_time[i] - entry_time[i]);
                trans_02_events.push(true);
                trans_02_covariates.push(cov_row);
            } else {
                trans_01_times.push(exit_time[i] - entry_time[i]);
                trans_01_events.push(false);
                trans_01_covariates.push(cov_row.clone());
                trans_02_times.push(exit_time[i] - entry_time[i]);
                trans_02_events.push(false);
                trans_02_covariates.push(cov_row);
            }
        } else if from == 1 {
            trans_12_times.push(exit_time[i] - transition_time[i]);
            trans_12_events.push(to == 2);
            trans_12_covariates.push(cov_row);
        }
    }

    let fit_cox =
        |times: &[f64], events: &[bool], cov: &[Vec<f64>]| -> (f64, f64, f64, Vec<f64>, Vec<f64>) {
            let n_obs = times.len();
            if n_obs == 0 {
                return (0.0, 1.0, 0.0, Vec::new(), Vec::new());
            }

            let n_events: usize = events.iter().filter(|&&e| e).count();
            if n_events == 0 {
                return (0.0, 1.0, 0.0, Vec::new(), Vec::new());
            }

            let mut sorted_indices: Vec<usize> = (0..n_obs).collect();
            sorted_indices.sort_by(|&a, &b| times[a].total_cmp(&times[b]).then_with(|| a.cmp(&b)));

            let coefficient = if n_covariates > 0 {
                if cov.len() != n_obs {
                    return (0.0, 1.0, 0.0, Vec::new(), Vec::new());
                }
                let mut sum_cov = 0.0;
                let mut sum_event = 0;
                for &idx in &sorted_indices {
                    if events[idx] && !cov[idx].is_empty() {
                        sum_cov += cov[idx][0];
                        sum_event += 1;
                    }
                }
                if sum_event > 0 {
                    sum_cov / sum_event as f64
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let hessian = (n_events as f64).max(1.0);
            let se = (1.0 / hessian).sqrt();

            let unique_times = unique_illness_times(times);

            let baseline_hazard: Vec<f64> = unique_times
                .iter()
                .map(|&t| {
                    let at_risk = times.iter().filter(|&&ti| ti + TIME_EPSILON >= t).count() as f64;
                    let events_at_t = times
                        .iter()
                        .zip(events.iter())
                        .filter(|&(ti, e)| same_illness_time(*ti, t) && *e)
                        .count() as f64;
                    if at_risk > 0.0 {
                        events_at_t / at_risk
                    } else {
                        0.0
                    }
                })
                .collect();

            let log_lik = -(n_events as f64) * (n_events as f64 / n_obs as f64).ln().max(-100.0);

            (coefficient, se, log_lik, baseline_hazard, unique_times)
        };

    let (coef_01, se_01, ll_01, bh_01, bt_01) =
        fit_cox(&trans_01_times, &trans_01_events, &trans_01_covariates);
    let (coef_02, se_02, ll_02, bh_02, bt_02) =
        fit_cox(&trans_02_times, &trans_02_events, &trans_02_covariates);
    let (coef_12, se_12, ll_12, bh_12, bt_12) =
        fit_cox(&trans_12_times, &trans_12_events, &trans_12_covariates);

    let make_transition_hazard = |from: &str,
                                  to: &str,
                                  coef: f64,
                                  se: f64,
                                  bh: Vec<f64>,
                                  bt: Vec<f64>|
     -> TransitionHazard {
        let hr = coef.exp();
        let z = if se > 1e-10 { coef / se } else { 0.0 };
        let p_value = 2.0 * (1.0 - normal_cdf(z.abs()));
        TransitionHazard {
            from_state: from.to_string(),
            to_state: to.to_string(),
            coefficient: coef,
            se,
            hazard_ratio: hr,
            ci_lower: (coef - 1.96 * se).exp(),
            ci_upper: (coef + 1.96 * se).exp(),
            p_value,
            baseline_hazard: bh,
            baseline_times: bt,
        }
    };

    let transition_hazards = vec![
        make_transition_hazard(
            &config.state_names[0],
            &config.state_names[1],
            coef_01,
            se_01,
            bh_01,
            bt_01,
        ),
        make_transition_hazard(
            &config.state_names[0],
            &config.state_names[2],
            coef_02,
            se_02,
            bh_02,
            bt_02,
        ),
        make_transition_hazard(
            &config.state_names[1],
            &config.state_names[2],
            coef_12,
            se_12,
            bh_12,
            bt_12,
        ),
    ];

    let max_time = exit_time.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let n_time_points = 100;
    let time_points: Vec<f64> = (0..=n_time_points)
        .map(|i| i as f64 * max_time / n_time_points as f64)
        .collect();

    let n_trans_01 = trans_01_events.iter().filter(|&&e| e).count();
    let n_trans_02 = trans_02_events.iter().filter(|&&e| e).count();
    let n_trans_12 = trans_12_events.iter().filter(|&&e| e).count();

    let hazard_01 = if !trans_01_times.is_empty() {
        n_trans_01 as f64 / trans_01_times.iter().sum::<f64>().max(1.0)
    } else {
        0.01
    };
    let hazard_02 = if !trans_02_times.is_empty() {
        n_trans_02 as f64 / trans_02_times.iter().sum::<f64>().max(1.0)
    } else {
        0.01
    };
    let hazard_12 = if !trans_12_times.is_empty() {
        n_trans_12 as f64 / trans_12_times.iter().sum::<f64>().max(1.0)
    } else {
        0.01
    };

    let mut state_occupation_probs: Vec<Vec<f64>> = Vec::new();
    let mut cumulative_incidence: Vec<Vec<f64>> = Vec::new();

    for &t in &time_points {
        let p0 = (-(hazard_01 + hazard_02) * t).exp();
        let p2_direct = if (hazard_01 + hazard_02).abs() > 1e-10 {
            hazard_02 / (hazard_01 + hazard_02) * (1.0 - p0)
        } else {
            0.0
        };

        let p2_via_illness = if (hazard_01 + hazard_02 - hazard_12).abs() > 1e-10 {
            hazard_01 * hazard_12 / ((hazard_01 + hazard_02) * (hazard_01 + hazard_02 - hazard_12))
                * (1.0 - p0 - (hazard_01 + hazard_02) / hazard_12 * ((-hazard_12 * t).exp() - p0))
        } else {
            hazard_01 * t * (-hazard_12 * t).exp() * (1.0 - p0) / 2.0
        };

        let p2 = (p2_direct + p2_via_illness).clamp(0.0, 1.0);
        let p1 = (1.0 - p0 - p2).max(0.0);

        state_occupation_probs.push(vec![p0, p1, p2]);
        cumulative_incidence.push(vec![1.0 - p0, p2_direct, p2_via_illness.max(0.0)]);
    }

    let sojourn_0 = if (hazard_01 + hazard_02) > 1e-10 {
        1.0 / (hazard_01 + hazard_02)
    } else {
        f64::INFINITY
    };
    let sojourn_1 = if hazard_12 > 1e-10 {
        1.0 / hazard_12
    } else {
        f64::INFINITY
    };
    let sojourn_times = vec![sojourn_0, sojourn_1, 0.0];

    let log_likelihood = ll_01 + ll_02 + ll_12;
    let n_params = if n_covariates > 0 {
        3 + 3 * n_covariates
    } else {
        3
    };
    let n_obs = n as f64;
    let aic = -2.0 * log_likelihood + 2.0 * n_params as f64;
    let bic = -2.0 * log_likelihood + (n_params as f64) * n_obs.ln();

    Ok(IllnessDeathResult {
        transition_hazards,
        state_occupation_probs,
        time_points,
        cumulative_incidence,
        sojourn_times,
        log_likelihood,
        aic,
        bic,
        n_transitions: vec![n_trans_01, n_trans_02, n_trans_12],
        model_type: config.model_type,
    })
}

fn same_illness_time(left: f64, right: f64) -> bool {
    (left - right).abs() < TIME_EPSILON
}

fn unique_illness_times(times: &[f64]) -> Vec<f64> {
    let mut unique_times = times.to_vec();
    unique_times.sort_by(|a, b| a.total_cmp(b));
    unique_times.dedup_by(|a, b| same_illness_time(*a, *b));
    unique_times
}

fn validate_illness_death_fit_inputs(
    entry_time: &[f64],
    transition_time: &[f64],
    exit_time: &[f64],
    from_state: &[i32],
    to_state: &[i32],
    covariates: Option<&[Vec<f64>]>,
) -> PyResult<()> {
    let n = entry_time.len();
    if transition_time.len() != n
        || exit_time.len() != n
        || from_state.len() != n
        || to_state.len() != n
    {
        return Err(PyValueError::new_err(
            "All input vectors must have the same length",
        ));
    }
    if n == 0 {
        return Err(PyValueError::new_err("input vectors must be non-empty"));
    }

    validate_no_nan(entry_time, "entry_time")?;
    validate_finite(entry_time, "entry_time")?;
    validate_non_negative(entry_time, "entry_time")?;
    validate_no_nan(transition_time, "transition_time")?;
    validate_finite(transition_time, "transition_time")?;
    validate_non_negative(transition_time, "transition_time")?;
    validate_no_nan(exit_time, "exit_time")?;
    validate_finite(exit_time, "exit_time")?;
    validate_non_negative(exit_time, "exit_time")?;

    for i in 0..n {
        let from = from_state[i];
        let to = to_state[i];
        if !(0..=1).contains(&from) {
            return Err(PyValueError::new_err(format!(
                "from_state must contain only 0/1 values; got {from} at index {i}"
            )));
        }
        if !(0..=2).contains(&to) {
            return Err(PyValueError::new_err(format!(
                "to_state must contain only 0/1/2 values; got {to} at index {i}"
            )));
        }
        if entry_time[i] > exit_time[i] + TIME_EPSILON {
            return Err(PyValueError::new_err(format!(
                "entry_time must be <= exit_time at index {i}"
            )));
        }
        if (from == 1 || (from == 0 && to == 1))
            && (transition_time[i] + TIME_EPSILON < entry_time[i]
                || transition_time[i] > exit_time[i] + TIME_EPSILON)
        {
            return Err(PyValueError::new_err(format!(
                "transition_time must be between entry_time and exit_time at index {i}"
            )));
        }
        if from == 1 && to == 0 {
            return Err(PyValueError::new_err(
                "to_state cannot return to 0 when from_state is 1",
            ));
        }
    }

    if let Some(covariates) = covariates {
        if covariates.len() != n {
            return Err(PyValueError::new_err(
                "covariates must have the same number of rows as input vectors",
            ));
        }
        let n_cols = covariates.first().map_or(0, Vec::len);
        for (row_idx, row) in covariates.iter().enumerate() {
            if row.len() != n_cols {
                return Err(PyValueError::new_err(format!(
                    "covariates row {row_idx} has {} columns, expected {n_cols}",
                    row.len()
                )));
            }
            validate_no_nan(row, "covariates")?;
            validate_finite(row, "covariates")?;
        }
    }

    Ok(())
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct IllnessDeathPrediction {
    #[pyo3(get)]
    pub state_probs: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub survival_prob: Vec<f64>,
    #[pyo3(get)]
    pub illness_free_survival: Vec<f64>,
    #[pyo3(get)]
    pub death_prob: Vec<f64>,
}

#[pyfunction]
#[pyo3(signature = (model, current_state, time_in_state, prediction_times, covariates=None))]
pub fn predict_illness_death(
    model: &IllnessDeathResult,
    current_state: usize,
    time_in_state: f64,
    prediction_times: Vec<f64>,
    covariates: Option<Vec<f64>>,
) -> PyResult<IllnessDeathPrediction> {
    validate_illness_death_prediction_inputs(
        model,
        current_state,
        time_in_state,
        &prediction_times,
        covariates.as_deref(),
    )?;

    if current_state == 2 {
        let n_times = prediction_times.len();
        return Ok(IllnessDeathPrediction {
            state_probs: vec![vec![0.0, 0.0, 1.0]; n_times],
            time_points: prediction_times,
            survival_prob: vec![0.0; n_times],
            illness_free_survival: vec![0.0; n_times],
            death_prob: vec![1.0; n_times],
        });
    }

    let covariate_effect = if let Some(cov) = &covariates {
        if !cov.is_empty() {
            model
                .transition_hazards
                .iter()
                .map(|h| (h.coefficient * cov[0]).exp())
                .collect::<Vec<f64>>()
        } else {
            vec![1.0; 3]
        }
    } else {
        vec![1.0; 3]
    };

    let hazard_01 = if !model.transition_hazards[0].baseline_hazard.is_empty() {
        model.transition_hazards[0]
            .baseline_hazard
            .iter()
            .sum::<f64>()
            / model.transition_hazards[0].baseline_hazard.len() as f64
            * covariate_effect[0]
    } else {
        0.01 * covariate_effect[0]
    };

    let hazard_02 = if !model.transition_hazards[1].baseline_hazard.is_empty() {
        model.transition_hazards[1]
            .baseline_hazard
            .iter()
            .sum::<f64>()
            / model.transition_hazards[1].baseline_hazard.len() as f64
            * covariate_effect[1]
    } else {
        0.01 * covariate_effect[1]
    };

    let hazard_12 = if !model.transition_hazards[2].baseline_hazard.is_empty() {
        model.transition_hazards[2]
            .baseline_hazard
            .iter()
            .sum::<f64>()
            / model.transition_hazards[2].baseline_hazard.len() as f64
            * covariate_effect[2]
    } else {
        0.01 * covariate_effect[2]
    };

    let mut state_probs: Vec<Vec<f64>> = Vec::new();
    let mut survival_prob: Vec<f64> = Vec::new();
    let mut illness_free_survival: Vec<f64> = Vec::new();
    let mut death_prob: Vec<f64> = Vec::new();

    for &t in &prediction_times {
        let total_time = time_in_state + t;

        let (mut p0, mut p1, mut p2) = if current_state == 0 {
            let p0 = (-(hazard_01 + hazard_02) * total_time).exp();
            let denom = hazard_01 + hazard_02 - hazard_12;
            let p2 = if denom.abs() > 1e-10 {
                let term = hazard_01 / denom
                    * ((-(hazard_01 + hazard_02) * total_time).exp()
                        - (-hazard_12 * total_time).exp());
                (1.0 - p0 - term).max(0.0)
            } else {
                hazard_02 / (hazard_01 + hazard_02 + 1e-10) * (1.0 - p0)
            };
            let p1 = (1.0 - p0 - p2).max(0.0);
            (p0, p1, p2)
        } else {
            let p1 = (-hazard_12 * total_time).exp();
            let p2 = 1.0 - p1;
            (0.0, p1, p2)
        };

        let sum = p0 + p1 + p2;
        if sum > 1e-10 {
            p0 /= sum;
            p1 /= sum;
            p2 /= sum;
        }

        state_probs.push(vec![p0, p1, p2]);
        survival_prob.push(p0 + p1);
        illness_free_survival.push(p0);
        death_prob.push(p2);
    }

    Ok(IllnessDeathPrediction {
        state_probs,
        time_points: prediction_times,
        survival_prob,
        illness_free_survival,
        death_prob,
    })
}

fn validate_illness_death_prediction_inputs(
    model: &IllnessDeathResult,
    current_state: usize,
    time_in_state: f64,
    prediction_times: &[f64],
    covariates: Option<&[f64]>,
) -> PyResult<()> {
    if model.transition_hazards.len() < 3 {
        return Err(PyValueError::new_err(
            "model must contain at least 3 transition hazards",
        ));
    }
    if current_state > 2 {
        return Err(PyValueError::new_err(
            "current_state must be 0 (Healthy), 1 (Illness), or 2 (Death)",
        ));
    }
    if !time_in_state.is_finite() || time_in_state < 0.0 {
        return Err(PyValueError::new_err(
            "time_in_state must be finite and non-negative",
        ));
    }
    validate_no_nan(prediction_times, "prediction_times")?;
    validate_finite(prediction_times, "prediction_times")?;
    validate_non_negative(prediction_times, "prediction_times")?;

    if let Some(covariates) = covariates {
        validate_no_nan(covariates, "covariates")?;
        validate_finite(covariates, "covariates")?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_illness_death_config() {
        let config =
            IllnessDeathConfig::new(IllnessDeathType::Progressive, None, "forward", 100, 1e-6, 0)
                .unwrap();
        assert_eq!(config.state_names.len(), 3);
        assert_eq!(config.model_type, IllnessDeathType::Progressive);

        assert!(
            IllnessDeathConfig::new(
                IllnessDeathType::Progressive,
                None,
                "sideways",
                100,
                1e-6,
                0
            )
            .is_err()
        );
        assert!(
            IllnessDeathConfig::new(IllnessDeathType::Progressive, None, "forward", 0, 1e-6, 0)
                .is_err()
        );
        assert!(
            IllnessDeathConfig::new(
                IllnessDeathType::Progressive,
                Some(vec!["Healthy".to_string(), "Illness".to_string()]),
                "forward",
                100,
                1e-6,
                0,
            )
            .is_err()
        );
    }

    #[test]
    fn test_fit_illness_death() {
        let entry_time = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let transition_time = vec![1.0, 0.0, 1.5, 0.0, 2.0, 0.0, 1.2, 0.0];
        let exit_time = vec![2.0, 3.0, 2.5, 4.0, 3.0, 2.0, 2.2, 5.0];
        let from_state = vec![0, 0, 0, 0, 0, 0, 0, 0];
        let to_state = vec![1, 2, 1, 0, 1, 2, 1, 0];

        let config =
            IllnessDeathConfig::new(IllnessDeathType::Progressive, None, "forward", 100, 1e-6, 0)
                .unwrap();

        let result = fit_illness_death(
            entry_time,
            transition_time,
            exit_time,
            from_state,
            to_state,
            None,
            Some(config),
        )
        .unwrap();

        assert_eq!(result.transition_hazards.len(), 3);
        assert!(!result.state_occupation_probs.is_empty());
        assert!(result.log_likelihood.is_finite());
    }

    #[test]
    fn test_fit_illness_death_groups_near_tied_times_and_aligns_covariates() {
        let entry_time = vec![0.0, 0.0, 0.0];
        let transition_time = vec![1.0, 0.0, 1.0 + TIME_EPSILON / 2.0];
        let exit_time = vec![2.0, 2.0, 2.0];
        let from_state = vec![0, 0, 0];
        let to_state = vec![1, 2, 1];
        let covariates = vec![vec![10.0], vec![20.0], vec![30.0]];

        let result = fit_illness_death(
            entry_time,
            transition_time,
            exit_time,
            from_state,
            to_state,
            Some(covariates),
            None,
        )
        .unwrap();

        assert_eq!(result.transition_hazards[0].baseline_times.len(), 1);
        assert!((result.transition_hazards[0].baseline_times[0] - 1.0).abs() < TIME_EPSILON);
        assert!((result.transition_hazards[0].baseline_hazard[0] - 1.0).abs() < 1e-12);
        assert!((result.transition_hazards[0].coefficient - 20.0).abs() < 1e-12);
    }

    #[test]
    fn test_fit_illness_death_rejects_malformed_inputs() {
        let err =
            fit_illness_death(vec![], vec![], vec![], vec![], vec![], None, None).unwrap_err();
        assert!(err.to_string().contains("input vectors must be non-empty"));

        let err = fit_illness_death(
            vec![0.0],
            vec![0.0],
            vec![f64::INFINITY],
            vec![0],
            vec![0],
            None,
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("exit_time contains non-finite"));

        let err = fit_illness_death(
            vec![0.0],
            vec![0.0],
            vec![1.0],
            vec![3],
            vec![0],
            None,
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("from_state must contain only 0/1"));

        let err = fit_illness_death(
            vec![0.0],
            vec![3.0],
            vec![2.0],
            vec![0],
            vec![1],
            None,
            None,
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("transition_time must be between entry_time and exit_time")
        );

        let err = fit_illness_death(
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![2.0, 2.0],
            vec![0, 0],
            vec![1, 2],
            Some(vec![vec![1.0], vec![2.0, 3.0]]),
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("covariates row 1 has 2 columns"));
    }

    #[test]
    fn test_predict_illness_death() {
        let entry_time = vec![0.0, 0.0, 0.0, 0.0];
        let transition_time = vec![1.0, 0.0, 1.5, 0.0];
        let exit_time = vec![2.0, 3.0, 2.5, 4.0];
        let from_state = vec![0, 0, 0, 0];
        let to_state = vec![1, 2, 1, 0];

        let model = fit_illness_death(
            entry_time,
            transition_time,
            exit_time,
            from_state,
            to_state,
            None,
            None,
        )
        .unwrap();

        let prediction =
            predict_illness_death(&model, 0, 0.0, vec![0.5, 1.0, 1.5, 2.0], None).unwrap();

        assert_eq!(prediction.state_probs.len(), 4);
        for probs in &prediction.state_probs {
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 0.1);
        }

        let err = predict_illness_death(&model, 3, 0.0, vec![1.0], None).unwrap_err();
        assert!(
            err.to_string()
                .contains("current_state must be 0 (Healthy), 1 (Illness), or 2 (Death)")
        );

        let err = predict_illness_death(&model, 0, f64::INFINITY, vec![1.0], None).unwrap_err();
        assert!(
            err.to_string()
                .contains("time_in_state must be finite and non-negative")
        );

        let err = predict_illness_death(&model, 0, 0.0, vec![-1.0], None).unwrap_err();
        assert!(
            err.to_string()
                .contains("prediction_times contains negative value")
        );
    }

    #[test]
    fn test_illness_death_result_methods() {
        let entry_time = vec![0.0, 0.0, 0.0, 0.0];
        let transition_time = vec![1.0, 0.0, 1.5, 0.0];
        let exit_time = vec![2.0, 3.0, 2.5, 4.0];
        let from_state = vec![0, 0, 0, 0];
        let to_state = vec![1, 2, 1, 0];

        let result = fit_illness_death(
            entry_time,
            transition_time,
            exit_time,
            from_state,
            to_state,
            None,
            None,
        )
        .unwrap();

        let surv = result.get_survival_probability(1.0);
        assert!((0.0..=1.0).contains(&surv));

        let illness = result.get_illness_probability(1.0);
        assert!((0.0..=1.0).contains(&illness));

        let death = result.get_death_probability(1.0);
        assert!((0.0..=1.0).contains(&death));
    }
}
