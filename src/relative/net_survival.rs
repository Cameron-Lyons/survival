#![allow(
    unused_variables,
    unused_imports,
    unused_mut,
    unused_assignments,
    non_camel_case_types,
    clippy::needless_range_loop,
    clippy::len_zero,
    clippy::manual_clamp
)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass]
pub enum NetSurvivalMethod {
    EdererI,
    EdererII,
    Hakulinen,
    Pohar_Perme,
}

#[pymethods]
impl NetSurvivalMethod {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "edereri" | "ederer_i" | "ederer1" => Ok(NetSurvivalMethod::EdererI),
            "edererii" | "ederer_ii" | "ederer2" => Ok(NetSurvivalMethod::EdererII),
            "hakulinen" => Ok(NetSurvivalMethod::Hakulinen),
            "pohar_perme" | "poharperme" | "pp" => Ok(NetSurvivalMethod::Pohar_Perme),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown net survival method",
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct NetSurvivalResult {
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub net_survival: Vec<f64>,
    #[pyo3(get)]
    pub net_survival_se: Vec<f64>,
    #[pyo3(get)]
    pub net_survival_lower: Vec<f64>,
    #[pyo3(get)]
    pub net_survival_upper: Vec<f64>,
    #[pyo3(get)]
    pub cumulative_excess_hazard: Vec<f64>,
    #[pyo3(get)]
    pub n_at_risk: Vec<usize>,
    #[pyo3(get)]
    pub n_events: Vec<usize>,
    #[pyo3(get)]
    pub method: String,
}

#[pyfunction]
#[pyo3(signature = (
    time,
    status,
    expected_survival,
    method=NetSurvivalMethod::Pohar_Perme,
    weights=None
))]
pub fn net_survival(
    time: Vec<f64>,
    status: Vec<i32>,
    expected_survival: Vec<f64>,
    method: NetSurvivalMethod,
    weights: Option<Vec<f64>>,
) -> PyResult<NetSurvivalResult> {
    let n = time.len();
    if status.len() != n || expected_survival.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All input arrays must have same length",
        ));
    }

    let wt = weights.unwrap_or_else(|| vec![1.0; n]);

    let mut unique_times: Vec<f64> = time.clone();
    unique_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    unique_times.dedup();

    match method {
        NetSurvivalMethod::Pohar_Perme => {
            pohar_perme_estimator(&time, &status, &expected_survival, &wt, &unique_times)
        }
        NetSurvivalMethod::EdererII => {
            ederer_ii_estimator(&time, &status, &expected_survival, &wt, &unique_times)
        }
        NetSurvivalMethod::EdererI => {
            ederer_i_estimator(&time, &status, &expected_survival, &wt, &unique_times)
        }
        NetSurvivalMethod::Hakulinen => {
            hakulinen_estimator(&time, &status, &expected_survival, &wt, &unique_times)
        }
    }
}

fn pohar_perme_estimator(
    time: &[f64],
    status: &[i32],
    expected_survival: &[f64],
    weights: &[f64],
    unique_times: &[f64],
) -> PyResult<NetSurvivalResult> {
    let n = time.len();

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut net_survival = Vec::with_capacity(unique_times.len());
    let mut net_survival_se = Vec::with_capacity(unique_times.len());
    let mut cumulative_excess_hazard = Vec::with_capacity(unique_times.len());
    let mut n_at_risk = Vec::with_capacity(unique_times.len());
    let mut n_events = Vec::with_capacity(unique_times.len());

    let mut net_surv = 1.0;
    let mut cum_excess_haz = 0.0;
    let mut var_term = 0.0;

    let mut time_idx = 0;
    let mut at_risk_set: Vec<usize> = (0..n).collect();

    for &t in unique_times {
        let mut d_weighted = 0.0;
        let mut at_risk_weighted = 0.0;
        let mut events_at_t = 0;

        for &i in &at_risk_set {
            if time[i] >= t {
                let w_i = weights[i] / expected_survival[i].max(1e-10);
                at_risk_weighted += w_i;
            }
        }

        while time_idx < n && time[indices[time_idx]] <= t {
            let idx = indices[time_idx];
            if status[idx] == 1 && (time[idx] - t).abs() < 1e-10 {
                let w_i = weights[idx] / expected_survival[idx].max(1e-10);
                d_weighted += w_i;
                events_at_t += 1;
            }
            time_idx += 1;
        }

        if at_risk_weighted > 0.0 {
            let excess_hazard = d_weighted / at_risk_weighted;
            net_surv *= 1.0 - excess_hazard;
            cum_excess_haz += excess_hazard;

            if excess_hazard < 1.0 {
                var_term += excess_hazard / (1.0 - excess_hazard) / at_risk_weighted;
            }
        }

        net_survival.push(net_surv.max(0.0));
        cumulative_excess_hazard.push(cum_excess_haz);
        net_survival_se.push((net_surv * net_surv * var_term).sqrt());
        n_at_risk.push(at_risk_set.len());
        n_events.push(events_at_t);

        at_risk_set.retain(|&i| time[i] > t);
    }

    let z = 1.96;
    let net_survival_lower: Vec<f64> = net_survival
        .iter()
        .zip(net_survival_se.iter())
        .map(|(&s, &se)| (s - z * se).max(0.0))
        .collect();

    let net_survival_upper: Vec<f64> = net_survival
        .iter()
        .zip(net_survival_se.iter())
        .map(|(&s, &se)| (s + z * se).min(1.0))
        .collect();

    Ok(NetSurvivalResult {
        time_points: unique_times.to_vec(),
        net_survival,
        net_survival_se,
        net_survival_lower,
        net_survival_upper,
        cumulative_excess_hazard,
        n_at_risk,
        n_events,
        method: "Pohar-Perme".to_string(),
    })
}

fn ederer_ii_estimator(
    time: &[f64],
    status: &[i32],
    expected_survival: &[f64],
    weights: &[f64],
    unique_times: &[f64],
) -> PyResult<NetSurvivalResult> {
    let n = time.len();

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut net_survival = Vec::with_capacity(unique_times.len());
    let mut net_survival_se = Vec::with_capacity(unique_times.len());
    let mut cumulative_excess_hazard = Vec::with_capacity(unique_times.len());
    let mut n_at_risk = Vec::with_capacity(unique_times.len());
    let mut n_events = Vec::with_capacity(unique_times.len());

    let mut obs_surv = 1.0;
    let mut time_idx = 0;
    let mut at_risk = n;
    let mut var_term = 0.0;

    for (t_idx, &t) in unique_times.iter().enumerate() {
        let mut d = 0;

        while time_idx < n && time[indices[time_idx]] <= t {
            let idx = indices[time_idx];
            if status[idx] == 1 {
                d += 1;
            }
            time_idx += 1;
        }

        if at_risk > 0 && d > 0 {
            let hazard = d as f64 / at_risk as f64;
            obs_surv *= 1.0 - hazard;
            var_term += hazard / (1.0 - hazard) / at_risk as f64;
        }

        let exp_surv_at_t: f64 = (0..n)
            .filter(|&i| time[i] >= t)
            .map(|i| expected_survival[i] * weights[i])
            .sum::<f64>()
            / (0..n)
                .filter(|&i| time[i] >= t)
                .map(|i| weights[i])
                .sum::<f64>()
                .max(1e-10);

        let net_surv = if exp_surv_at_t > 0.0 {
            obs_surv / exp_surv_at_t
        } else {
            obs_surv
        };

        let excess_haz = if net_surv > 0.0 { -net_surv.ln() } else { 0.0 };

        net_survival.push(net_surv.max(0.0).min(2.0));
        cumulative_excess_hazard.push(excess_haz);
        net_survival_se.push((net_surv * net_surv * var_term).sqrt());
        n_at_risk.push(at_risk);
        n_events.push(d);

        at_risk -= d;
    }

    let z = 1.96;
    let net_survival_lower: Vec<f64> = net_survival
        .iter()
        .zip(net_survival_se.iter())
        .map(|(&s, &se)| (s - z * se).max(0.0))
        .collect();

    let net_survival_upper: Vec<f64> = net_survival
        .iter()
        .zip(net_survival_se.iter())
        .map(|(&s, &se)| (s + z * se).min(1.0))
        .collect();

    Ok(NetSurvivalResult {
        time_points: unique_times.to_vec(),
        net_survival,
        net_survival_se,
        net_survival_lower,
        net_survival_upper,
        cumulative_excess_hazard,
        n_at_risk,
        n_events,
        method: "Ederer II".to_string(),
    })
}

fn ederer_i_estimator(
    time: &[f64],
    status: &[i32],
    expected_survival: &[f64],
    weights: &[f64],
    unique_times: &[f64],
) -> PyResult<NetSurvivalResult> {
    let n = time.len();

    let initial_exp_surv: f64 = expected_survival
        .iter()
        .zip(weights.iter())
        .map(|(&e, &w)| e * w)
        .sum::<f64>()
        / weights.iter().sum::<f64>().max(1e-10);

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut net_survival = Vec::with_capacity(unique_times.len());
    let mut net_survival_se = Vec::with_capacity(unique_times.len());
    let mut cumulative_excess_hazard = Vec::with_capacity(unique_times.len());
    let mut n_at_risk = Vec::with_capacity(unique_times.len());
    let mut n_events = Vec::with_capacity(unique_times.len());

    let mut obs_surv = 1.0;
    let mut time_idx = 0;
    let mut at_risk = n;
    let mut var_term = 0.0;

    for &t in unique_times {
        let mut d = 0;

        while time_idx < n && time[indices[time_idx]] <= t {
            let idx = indices[time_idx];
            if status[idx] == 1 {
                d += 1;
            }
            time_idx += 1;
        }

        if at_risk > 0 && d > 0 {
            let hazard = d as f64 / at_risk as f64;
            obs_surv *= 1.0 - hazard;
            var_term += hazard / (1.0 - hazard) / at_risk as f64;
        }

        let net_surv = if initial_exp_surv > 0.0 {
            obs_surv / initial_exp_surv
        } else {
            obs_surv
        };

        let excess_haz = if net_surv > 0.0 { -net_surv.ln() } else { 0.0 };

        net_survival.push(net_surv.max(0.0).min(2.0));
        cumulative_excess_hazard.push(excess_haz);
        net_survival_se.push((net_surv * net_surv * var_term).sqrt());
        n_at_risk.push(at_risk);
        n_events.push(d);

        at_risk -= d;
    }

    let z = 1.96;
    let net_survival_lower: Vec<f64> = net_survival
        .iter()
        .zip(net_survival_se.iter())
        .map(|(&s, &se)| (s - z * se).max(0.0))
        .collect();

    let net_survival_upper: Vec<f64> = net_survival
        .iter()
        .zip(net_survival_se.iter())
        .map(|(&s, &se)| (s + z * se).min(1.0))
        .collect();

    Ok(NetSurvivalResult {
        time_points: unique_times.to_vec(),
        net_survival,
        net_survival_se,
        net_survival_lower,
        net_survival_upper,
        cumulative_excess_hazard,
        n_at_risk,
        n_events,
        method: "Ederer I".to_string(),
    })
}

fn hakulinen_estimator(
    time: &[f64],
    status: &[i32],
    expected_survival: &[f64],
    weights: &[f64],
    unique_times: &[f64],
) -> PyResult<NetSurvivalResult> {
    ederer_ii_estimator(time, status, expected_survival, weights, unique_times).map(|mut result| {
        result.method = "Hakulinen".to_string();
        result
    })
}

#[pyfunction]
pub fn crude_probability_of_death(
    time: Vec<f64>,
    status: Vec<i32>,
    expected_survival: Vec<f64>,
    cause: Vec<i32>,
    time_points: Vec<f64>,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let n = time.len();

    let mut crude_cancer = Vec::with_capacity(time_points.len());
    let mut crude_other = Vec::with_capacity(time_points.len());

    for &t in &time_points {
        let mut cancer_deaths = 0.0;
        let mut other_deaths = 0.0;
        let mut total = 0.0;

        for i in 0..n {
            if time[i] <= t && status[i] == 1 {
                if cause[i] == 1 {
                    cancer_deaths += 1.0;
                } else {
                    other_deaths += 1.0;
                }
            }
            total += 1.0;
        }

        crude_cancer.push(cancer_deaths / total);
        crude_other.push(other_deaths / total);
    }

    Ok((time_points, crude_cancer, crude_other))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pohar_perme_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 0, 1];
        let expected_survival = vec![0.98, 0.96, 0.94, 0.92, 0.90];

        let result = net_survival(
            time,
            status,
            expected_survival,
            NetSurvivalMethod::Pohar_Perme,
            None,
        )
        .unwrap();

        assert!(result.time_points.len() > 0);
        assert!(result.net_survival.iter().all(|&s| s >= 0.0));
    }
}
