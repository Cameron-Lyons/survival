use crate::constants::PARALLEL_THRESHOLD_XLARGE;
use pyo3::prelude::*;
use rayon::prelude::*;
#[derive(Debug, Clone)]
#[pyclass]
pub struct NelsonAalenResult {
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub cumulative_hazard: Vec<f64>,
    #[pyo3(get)]
    pub variance: Vec<f64>,
    #[pyo3(get)]
    pub ci_lower: Vec<f64>,
    #[pyo3(get)]
    pub ci_upper: Vec<f64>,
    #[pyo3(get)]
    pub n_risk: Vec<usize>,
    #[pyo3(get)]
    pub n_events: Vec<usize>,
}
#[pymethods]
impl NelsonAalenResult {
    #[new]
    fn new(
        time: Vec<f64>,
        cumulative_hazard: Vec<f64>,
        variance: Vec<f64>,
        ci_lower: Vec<f64>,
        ci_upper: Vec<f64>,
        n_risk: Vec<usize>,
        n_events: Vec<usize>,
    ) -> Self {
        Self {
            time,
            cumulative_hazard,
            variance,
            ci_lower,
            ci_upper,
            n_risk,
            n_events,
        }
    }
    fn survival(&self) -> Vec<f64> {
        self.cumulative_hazard.iter().map(|&h| (-h).exp()).collect()
    }
}
pub fn nelson_aalen(
    time: &[f64],
    status: &[i32],
    weights: Option<&[f64]>,
    confidence_level: f64,
) -> NelsonAalenResult {
    let n = time.len();
    if n == 0 {
        return NelsonAalenResult {
            time: vec![],
            cumulative_hazard: vec![],
            variance: vec![],
            ci_lower: vec![],
            ci_upper: vec![],
            n_risk: vec![],
            n_events: vec![],
        };
    }
    let default_weights: Vec<f64> = vec![1.0; n];
    let weights = weights.unwrap_or(&default_weights);
    let mut indices: Vec<usize> = (0..n).collect();
    if n > PARALLEL_THRESHOLD_XLARGE {
        indices.par_sort_by(|&a, &b| {
            time[a]
                .partial_cmp(&time[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        indices.sort_by(|&a, &b| {
            time[a]
                .partial_cmp(&time[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    let mut unique_times: Vec<f64> = Vec::new();
    let mut events_at_time: Vec<f64> = Vec::new();
    let mut at_risk: Vec<f64> = Vec::new();
    let mut n_events_vec: Vec<usize> = Vec::new();
    let mut n_risk_vec: Vec<usize> = Vec::new();
    let mut total_weight: f64 = weights.iter().sum();
    let mut total_count = n;
    let mut i = 0;
    while i < n {
        let current_time = time[indices[i]];
        let mut events = 0.0;
        let mut event_count = 0usize;
        let mut censored_weight = 0.0;
        let mut censored_count = 0usize;
        while i < n && time[indices[i]] == current_time {
            let idx = indices[i];
            if status[idx] == 1 {
                events += weights[idx];
                event_count += 1;
            } else {
                censored_weight += weights[idx];
                censored_count += 1;
            }
            i += 1;
        }
        if event_count > 0 {
            unique_times.push(current_time);
            events_at_time.push(events);
            at_risk.push(total_weight);
            n_events_vec.push(event_count);
            n_risk_vec.push(total_count);
        }
        total_weight -= events + censored_weight;
        total_count -= event_count + censored_count;
    }
    let m = unique_times.len();
    let mut cumulative_hazard = Vec::with_capacity(m);
    let mut variance = Vec::with_capacity(m);
    let mut cum_h = 0.0;
    let mut cum_var = 0.0;
    for j in 0..m {
        let d = events_at_time[j];
        let y = at_risk[j];
        if y > 0.0 {
            cum_h += d / y;
            cum_var += d / (y * y);
        }
        cumulative_hazard.push(cum_h);
        variance.push(cum_var);
    }
    let z = if confidence_level >= 0.99 {
        2.576
    } else if confidence_level >= 0.95 {
        1.96
    } else if confidence_level >= 0.90 {
        1.645
    } else {
        1.28
    };
    let mut ci_lower = Vec::with_capacity(m);
    let mut ci_upper = Vec::with_capacity(m);
    for j in 0..m {
        let se = variance[j].sqrt();
        ci_lower.push((cumulative_hazard[j] - z * se).max(0.0));
        ci_upper.push(cumulative_hazard[j] + z * se);
    }
    NelsonAalenResult {
        time: unique_times,
        cumulative_hazard,
        variance,
        ci_lower,
        ci_upper,
        n_risk: n_risk_vec,
        n_events: n_events_vec,
    }
}
/// Compute Nelson-Aalen cumulative hazard estimate.
///
/// Parameters
/// ----------
/// time : array-like
///     Survival/censoring times.
/// status : array-like
///     Event indicator (1=event, 0=censored).
/// weights : array-like, optional
///     Case weights.
/// confidence_level : float, optional
///     Confidence level (default 0.95).
///
/// Returns
/// -------
/// NelsonAalenResult
///     Object with: time, cumulative_hazard, std_err, conf_lower, conf_upper, n_risk, n_events.
#[pyfunction]
#[pyo3(signature = (time, status, weights=None, confidence_level=None))]
pub fn nelson_aalen_estimator(
    time: Vec<f64>,
    status: Vec<i32>,
    weights: Option<Vec<f64>>,
    confidence_level: Option<f64>,
) -> PyResult<NelsonAalenResult> {
    let conf = confidence_level.unwrap_or(0.95);
    let weights_ref = weights.as_deref();
    Ok(nelson_aalen(&time, &status, weights_ref, conf))
}
#[derive(Debug, Clone)]
#[pyclass]
pub struct StratifiedKMResult {
    #[pyo3(get)]
    pub strata: Vec<i32>,
    #[pyo3(get)]
    pub times: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub survival: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub ci_lower: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub ci_upper: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub n_risk: Vec<Vec<usize>>,
    #[pyo3(get)]
    pub n_events: Vec<Vec<usize>>,
}
#[pymethods]
impl StratifiedKMResult {
    #[new]
    fn new(
        strata: Vec<i32>,
        times: Vec<Vec<f64>>,
        survival: Vec<Vec<f64>>,
        ci_lower: Vec<Vec<f64>>,
        ci_upper: Vec<Vec<f64>>,
        n_risk: Vec<Vec<usize>>,
        n_events: Vec<Vec<usize>>,
    ) -> Self {
        Self {
            strata,
            times,
            survival,
            ci_lower,
            ci_upper,
            n_risk,
            n_events,
        }
    }
}
pub fn stratified_km(
    time: &[f64],
    status: &[i32],
    strata: &[i32],
    confidence_level: f64,
) -> StratifiedKMResult {
    let n = time.len();
    if n == 0 {
        return StratifiedKMResult {
            strata: vec![],
            times: vec![],
            survival: vec![],
            ci_lower: vec![],
            ci_upper: vec![],
            n_risk: vec![],
            n_events: vec![],
        };
    }
    let mut unique_strata: Vec<i32> = strata.to_vec();
    unique_strata.sort();
    unique_strata.dedup();
    let mut all_times = Vec::new();
    let mut all_survival = Vec::new();
    let mut all_ci_lower = Vec::new();
    let mut all_ci_upper = Vec::new();
    let mut all_n_risk = Vec::new();
    let mut all_n_events = Vec::new();
    let stratum_results: Vec<_> = unique_strata
        .par_iter()
        .map(|&stratum| {
            let mask: Vec<bool> = strata.iter().map(|&s| s == stratum).collect();
            let stratum_time: Vec<f64> = time
                .iter()
                .zip(&mask)
                .filter(|&(_, m)| *m)
                .map(|(&t, _)| t)
                .collect();
            let stratum_status: Vec<i32> = status
                .iter()
                .zip(&mask)
                .filter(|&(_, m)| *m)
                .map(|(&s, _)| s)
                .collect();
            kaplan_meier(&stratum_time, &stratum_status, None, confidence_level)
        })
        .collect();
    for result in stratum_results {
        all_times.push(result.time);
        all_survival.push(result.survival);
        all_ci_lower.push(result.ci_lower);
        all_ci_upper.push(result.ci_upper);
        all_n_risk.push(result.n_risk);
        all_n_events.push(result.n_events);
    }
    StratifiedKMResult {
        strata: unique_strata,
        times: all_times,
        survival: all_survival,
        ci_lower: all_ci_lower,
        ci_upper: all_ci_upper,
        n_risk: all_n_risk,
        n_events: all_n_events,
    }
}
#[derive(Debug, Clone)]
struct KMResult {
    time: Vec<f64>,
    survival: Vec<f64>,
    ci_lower: Vec<f64>,
    ci_upper: Vec<f64>,
    n_risk: Vec<usize>,
    n_events: Vec<usize>,
}
fn kaplan_meier(
    time: &[f64],
    status: &[i32],
    weights: Option<&[f64]>,
    confidence_level: f64,
) -> KMResult {
    let n = time.len();
    if n == 0 {
        return KMResult {
            time: vec![],
            survival: vec![],
            ci_lower: vec![],
            ci_upper: vec![],
            n_risk: vec![],
            n_events: vec![],
        };
    }
    let default_weights: Vec<f64> = vec![1.0; n];
    let weights = weights.unwrap_or(&default_weights);
    let mut indices: Vec<usize> = (0..n).collect();
    if n > PARALLEL_THRESHOLD_XLARGE {
        indices.par_sort_by(|&a, &b| {
            time[a]
                .partial_cmp(&time[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        indices.sort_by(|&a, &b| {
            time[a]
                .partial_cmp(&time[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    let mut unique_times: Vec<f64> = Vec::new();
    let mut events_at_time: Vec<f64> = Vec::new();
    let mut at_risk: Vec<f64> = Vec::new();
    let mut n_events_vec: Vec<usize> = Vec::new();
    let mut n_risk_vec: Vec<usize> = Vec::new();
    let mut total_weight: f64 = weights.iter().sum();
    let mut total_count = n;
    let mut i = 0;
    while i < n {
        let current_time = time[indices[i]];
        let mut events = 0.0;
        let mut event_count = 0usize;
        let mut removed_weight = 0.0;
        let mut removed_count = 0usize;
        while i < n && time[indices[i]] == current_time {
            let idx = indices[i];
            removed_weight += weights[idx];
            removed_count += 1;
            if status[idx] == 1 {
                events += weights[idx];
                event_count += 1;
            }
            i += 1;
        }
        if event_count > 0 {
            unique_times.push(current_time);
            events_at_time.push(events);
            at_risk.push(total_weight);
            n_events_vec.push(event_count);
            n_risk_vec.push(total_count);
        }
        total_weight -= removed_weight;
        total_count -= removed_count;
    }
    let m = unique_times.len();
    let mut survival = Vec::with_capacity(m);
    let mut greenwood_var = Vec::with_capacity(m);
    let mut surv = 1.0;
    let mut var_sum = 0.0;
    for j in 0..m {
        let d = events_at_time[j];
        let y = at_risk[j];
        if y > 0.0 {
            surv *= 1.0 - d / y;
            if y > d {
                var_sum += d / (y * (y - d));
            }
        }
        survival.push(surv);
        greenwood_var.push(surv * surv * var_sum);
    }
    let z = if confidence_level >= 0.99 {
        2.576
    } else if confidence_level >= 0.95 {
        1.96
    } else if confidence_level >= 0.90 {
        1.645
    } else {
        1.28
    };
    let mut ci_lower = Vec::with_capacity(m);
    let mut ci_upper = Vec::with_capacity(m);
    for j in 0..m {
        let se = greenwood_var[j].sqrt();
        ci_lower.push((survival[j] - z * se).clamp(0.0, 1.0));
        ci_upper.push((survival[j] + z * se).clamp(0.0, 1.0));
    }
    KMResult {
        time: unique_times,
        survival,
        ci_lower,
        ci_upper,
        n_risk: n_risk_vec,
        n_events: n_events_vec,
    }
}
#[pyfunction]
#[pyo3(signature = (time, status, strata, confidence_level=None))]
pub fn stratified_kaplan_meier(
    time: Vec<f64>,
    status: Vec<i32>,
    strata: Vec<i32>,
    confidence_level: Option<f64>,
) -> PyResult<StratifiedKMResult> {
    let conf = confidence_level.unwrap_or(0.95);
    Ok(stratified_km(&time, &status, &strata, conf))
}
