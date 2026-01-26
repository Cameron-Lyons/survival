#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

use crate::utilities::statistical::normal_cdf;

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(eq, eq_int)]
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
#[pyclass]
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
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "clock_type must be one of: forward, backward, gap",
            ));
        }

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

#[derive(Debug, Clone)]
#[pyclass]
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
#[pyclass]
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
    let config = config.unwrap_or_else(|| {
        IllnessDeathConfig::new(IllnessDeathType::Progressive, None, "forward", 100, 1e-6, 0)
            .unwrap()
    });

    let n = entry_time.len();
    if transition_time.len() != n
        || exit_time.len() != n
        || from_state.len() != n
        || to_state.len() != n
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All input vectors must have the same length",
        ));
    }

    let n_covariates = covariates
        .as_ref()
        .map(|c| c.first().map(|v| v.len()).unwrap_or(0))
        .unwrap_or(0);

    let mut trans_01_times: Vec<f64> = Vec::new();
    let mut trans_01_events: Vec<bool> = Vec::new();
    let mut trans_02_times: Vec<f64> = Vec::new();
    let mut trans_02_events: Vec<bool> = Vec::new();
    let mut trans_12_times: Vec<f64> = Vec::new();
    let mut trans_12_events: Vec<bool> = Vec::new();

    for i in 0..n {
        let from = from_state[i];
        let to = to_state[i];

        if from == 0 {
            if to == 1 {
                trans_01_times.push(transition_time[i] - entry_time[i]);
                trans_01_events.push(true);
            } else if to == 2 {
                trans_02_times.push(exit_time[i] - entry_time[i]);
                trans_02_events.push(true);
            } else {
                trans_01_times.push(exit_time[i] - entry_time[i]);
                trans_01_events.push(false);
                trans_02_times.push(exit_time[i] - entry_time[i]);
                trans_02_events.push(false);
            }
        } else if from == 1 {
            trans_12_times.push(exit_time[i] - transition_time[i]);
            trans_12_events.push(to == 2);
        }
    }

    let fit_cox = |times: &[f64],
                   events: &[bool],
                   cov: &Option<Vec<Vec<f64>>>|
     -> (f64, f64, f64, Vec<f64>, Vec<f64>) {
        let n_obs = times.len();
        if n_obs == 0 {
            return (0.0, 1.0, 0.0, Vec::new(), Vec::new());
        }

        let n_events: usize = events.iter().filter(|&&e| e).count();
        if n_events == 0 {
            return (0.0, 1.0, 0.0, Vec::new(), Vec::new());
        }

        let mut sorted_indices: Vec<usize> = (0..n_obs).collect();
        sorted_indices.sort_by(|&a, &b| times[a].partial_cmp(&times[b]).unwrap());

        let coefficient = if n_covariates > 0 && cov.is_some() {
            let cov_vec = cov.as_ref().unwrap();
            let mut sum_cov = 0.0;
            let mut sum_event = 0;
            for &idx in &sorted_indices {
                if events[idx] && idx < cov_vec.len() && !cov_vec[idx].is_empty() {
                    sum_cov += cov_vec[idx][0];
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

        let mut unique_times: Vec<f64> = times.to_vec();
        unique_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique_times.dedup();

        let baseline_hazard: Vec<f64> = unique_times
            .iter()
            .map(|&t| {
                let at_risk = times.iter().filter(|&&ti| ti >= t).count() as f64;
                let events_at_t = times
                    .iter()
                    .zip(events.iter())
                    .filter(|&(&ti, &e)| (ti - t).abs() < 1e-10 && e)
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

    let cov_subset: Option<Vec<Vec<f64>>> = covariates.clone();

    let (coef_01, se_01, ll_01, bh_01, bt_01) =
        fit_cox(&trans_01_times, &trans_01_events, &cov_subset);
    let (coef_02, se_02, ll_02, bh_02, bt_02) =
        fit_cox(&trans_02_times, &trans_02_events, &cov_subset);
    let (coef_12, se_12, ll_12, bh_12, bt_12) =
        fit_cox(&trans_12_times, &trans_12_events, &cov_subset);

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

#[derive(Debug, Clone)]
#[pyclass]
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
    if current_state > 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "current_state must be 0 (Healthy), 1 (Illness), or 2 (Death)",
        ));
    }

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
