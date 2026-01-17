use crate::constants::PARALLEL_THRESHOLD_LARGE;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fmt;

#[derive(Debug, Clone)]
#[pyclass(str, get_all)]
pub struct TimeDepAUCResult {
    pub auc: f64,
    pub time: f64,
    pub n_cases: usize,
    pub n_controls: usize,
    pub std_error: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
}

impl fmt::Display for TimeDepAUCResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TimeDepAUCResult(auc={:.4}, time={:.2}, n_cases={}, n_controls={})",
            self.auc, self.time, self.n_cases, self.n_controls
        )
    }
}

#[pymethods]
impl TimeDepAUCResult {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        auc: f64,
        time: f64,
        n_cases: usize,
        n_controls: usize,
        std_error: f64,
        ci_lower: f64,
        ci_upper: f64,
    ) -> Self {
        Self {
            auc,
            time,
            n_cases,
            n_controls,
            std_error,
            ci_lower,
            ci_upper,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(str, get_all)]
pub struct CumulativeDynamicAUCResult {
    pub times: Vec<f64>,
    pub auc: Vec<f64>,
    pub mean_auc: f64,
    pub integrated_auc: f64,
    pub n_cases: Vec<usize>,
    pub n_controls: Vec<usize>,
}

impl fmt::Display for CumulativeDynamicAUCResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CumulativeDynamicAUCResult(n_times={}, mean_auc={:.4}, integrated_auc={:.4})",
            self.times.len(),
            self.mean_auc,
            self.integrated_auc
        )
    }
}

#[pymethods]
impl CumulativeDynamicAUCResult {
    #[new]
    fn new(
        times: Vec<f64>,
        auc: Vec<f64>,
        mean_auc: f64,
        integrated_auc: f64,
        n_cases: Vec<usize>,
        n_controls: Vec<usize>,
    ) -> Self {
        Self {
            times,
            auc,
            mean_auc,
            integrated_auc,
            n_cases,
            n_controls,
        }
    }
}

fn compute_survival_km(time: &[f64], status: &[i32]) -> (Vec<f64>, Vec<f64>) {
    let n = time.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut unique_times = Vec::new();
    let mut km_values = Vec::new();
    let mut cum_surv = 1.0;
    let mut at_risk = n;

    let mut i = 0;
    while i < n {
        let current_time = time[indices[i]];
        let mut event_count = 0;
        let mut total_at_time = 0;

        while i < n && (time[indices[i]] - current_time).abs() < 1e-10 {
            if status[indices[i]] == 1 {
                event_count += 1;
            }
            total_at_time += 1;
            i += 1;
        }

        if event_count > 0 && at_risk > 0 {
            cum_surv *= 1.0 - event_count as f64 / at_risk as f64;
        }

        unique_times.push(current_time);
        km_values.push(cum_surv);

        at_risk -= total_at_time;
    }

    (unique_times, km_values)
}

fn compute_censoring_km(time: &[f64], status: &[i32]) -> (Vec<f64>, Vec<f64>) {
    let n = time.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut unique_times = Vec::new();
    let mut km_values = Vec::new();
    let mut cum_surv = 1.0;
    let mut at_risk = n;

    let mut i = 0;
    while i < n {
        let current_time = time[indices[i]];
        let mut censored_count = 0;
        let mut total_at_time = 0;

        while i < n && (time[indices[i]] - current_time).abs() < 1e-10 {
            if status[indices[i]] == 0 {
                censored_count += 1;
            }
            total_at_time += 1;
            i += 1;
        }

        if censored_count > 0 && at_risk > 0 {
            cum_surv *= 1.0 - censored_count as f64 / at_risk as f64;
        }

        unique_times.push(current_time);
        km_values.push(cum_surv);

        at_risk -= total_at_time;
    }

    (unique_times, km_values)
}

fn get_km_prob(t: f64, unique_times: &[f64], km_values: &[f64]) -> f64 {
    if unique_times.is_empty() {
        return 1.0;
    }

    if t < unique_times[0] {
        return 1.0;
    }

    let mut left = 0;
    let mut right = unique_times.len();

    while left < right {
        let mid = (left + right) / 2;
        if unique_times[mid] <= t {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    if left == 0 { 1.0 } else { km_values[left - 1] }
}

pub fn time_dependent_auc_core(
    time: &[f64],
    status: &[i32],
    marker: &[f64],
    t: f64,
) -> TimeDepAUCResult {
    let n = time.len();

    if n == 0 {
        return TimeDepAUCResult {
            auc: 0.5,
            time: t,
            n_cases: 0,
            n_controls: 0,
            std_error: 0.0,
            ci_lower: 0.5,
            ci_upper: 0.5,
        };
    }

    let (cens_times, cens_km) = compute_censoring_km(time, status);
    let (surv_times, surv_km) = compute_survival_km(time, status);

    let s_t = get_km_prob(t, &surv_times, &surv_km);
    let min_g = 0.01;

    let mut cases: Vec<(usize, f64)> = Vec::new();
    let mut controls: Vec<(usize, f64)> = Vec::new();

    for i in 0..n {
        if time[i] <= t && status[i] == 1 {
            let g_ti = get_km_prob(time[i], &cens_times, &cens_km).max(min_g);
            let weight = 1.0 / g_ti;
            cases.push((i, weight));
        } else if time[i] > t {
            controls.push((i, 1.0));
        }
    }

    let n_cases = cases.len();
    let n_controls = controls.len();

    if n_cases == 0 || n_controls == 0 {
        return TimeDepAUCResult {
            auc: 0.5,
            time: t,
            n_cases,
            n_controls,
            std_error: 0.0,
            ci_lower: 0.5,
            ci_upper: 0.5,
        };
    }

    let compute_pair_contribution =
        |case_idx: usize, case_weight: f64, ctrl_idx: usize| -> (f64, f64) {
            let m_case = marker[case_idx];
            let m_ctrl = marker[ctrl_idx];

            let indicator = if m_case > m_ctrl {
                1.0
            } else if (m_case - m_ctrl).abs() < 1e-10 {
                0.5
            } else {
                0.0
            };

            (indicator * case_weight, case_weight)
        };

    let (numerator, denominator) = if n_cases * n_controls > PARALLEL_THRESHOLD_LARGE {
        cases
            .par_iter()
            .map(|&(case_idx, case_weight)| {
                let mut local_num = 0.0;
                let mut local_den = 0.0;
                for &(ctrl_idx, _) in &controls {
                    let (num, den) = compute_pair_contribution(case_idx, case_weight, ctrl_idx);
                    local_num += num;
                    local_den += den;
                }
                (local_num, local_den)
            })
            .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1))
    } else {
        let mut num = 0.0;
        let mut den = 0.0;
        for &(case_idx, case_weight) in &cases {
            for &(ctrl_idx, _) in &controls {
                let (n, d) = compute_pair_contribution(case_idx, case_weight, ctrl_idx);
                num += n;
                den += d;
            }
        }
        (num, den)
    };

    let auc = if denominator > 0.0 {
        numerator / denominator
    } else {
        0.5
    };

    let effective_n = (n_cases as f64 * n_controls as f64).sqrt();
    let var_auc = if effective_n > 1.0 {
        auc * (1.0 - auc) / effective_n
    } else {
        0.0
    };
    let std_error = var_auc.sqrt();
    let z = 1.96;
    let ci_lower = (auc - z * std_error).clamp(0.0, 1.0);
    let ci_upper = (auc + z * std_error).clamp(0.0, 1.0);

    let _ = s_t;

    TimeDepAUCResult {
        auc,
        time: t,
        n_cases,
        n_controls,
        std_error,
        ci_lower,
        ci_upper,
    }
}

pub fn cumulative_dynamic_auc_core(
    time: &[f64],
    status: &[i32],
    marker: &[f64],
    times: &[f64],
) -> CumulativeDynamicAUCResult {
    let n = time.len();

    if n == 0 || times.is_empty() {
        return CumulativeDynamicAUCResult {
            times: times.to_vec(),
            auc: vec![0.5; times.len()],
            mean_auc: 0.5,
            integrated_auc: 0.5,
            n_cases: vec![0; times.len()],
            n_controls: vec![0; times.len()],
        };
    }

    let results: Vec<TimeDepAUCResult> = if times.len() > 4 {
        times
            .par_iter()
            .map(|&t| time_dependent_auc_core(time, status, marker, t))
            .collect()
    } else {
        times
            .iter()
            .map(|&t| time_dependent_auc_core(time, status, marker, t))
            .collect()
    };

    let auc_values: Vec<f64> = results.iter().map(|r| r.auc).collect();
    let n_cases: Vec<usize> = results.iter().map(|r| r.n_cases).collect();
    let n_controls: Vec<usize> = results.iter().map(|r| r.n_controls).collect();

    let valid_aucs: Vec<f64> = auc_values
        .iter()
        .zip(n_cases.iter())
        .zip(n_controls.iter())
        .filter(|((_, nc), nctrl)| **nc > 0 && **nctrl > 0)
        .map(|((auc, _), _)| *auc)
        .collect();

    let mean_auc = if valid_aucs.is_empty() {
        0.5
    } else {
        valid_aucs.iter().sum::<f64>() / valid_aucs.len() as f64
    };

    let integrated_auc = if times.len() < 2 {
        mean_auc
    } else {
        let mut integrated = 0.0;
        let mut total_weight = 0.0;

        for i in 0..times.len() - 1 {
            if n_cases[i] > 0 && n_controls[i] > 0 && n_cases[i + 1] > 0 && n_controls[i + 1] > 0 {
                let dt = times[i + 1] - times[i];
                let avg_auc = (auc_values[i] + auc_values[i + 1]) / 2.0;
                integrated += avg_auc * dt;
                total_weight += dt;
            }
        }

        if total_weight > 0.0 {
            integrated / total_weight
        } else {
            mean_auc
        }
    };

    CumulativeDynamicAUCResult {
        times: times.to_vec(),
        auc: auc_values,
        mean_auc,
        integrated_auc,
        n_cases,
        n_controls,
    }
}

#[pyfunction]
#[pyo3(signature = (time, status, marker, t))]
pub fn time_dependent_auc(
    time: Vec<f64>,
    status: Vec<i32>,
    marker: Vec<f64>,
    t: f64,
) -> PyResult<TimeDepAUCResult> {
    let n = time.len();
    if n != status.len() || n != marker.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "time, status, and marker must have the same length",
        ));
    }
    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "input arrays must not be empty",
        ));
    }
    if t <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "time point t must be positive",
        ));
    }

    Ok(time_dependent_auc_core(&time, &status, &marker, t))
}

#[pyfunction]
#[pyo3(signature = (time, status, marker, times))]
pub fn cumulative_dynamic_auc(
    time: Vec<f64>,
    status: Vec<i32>,
    marker: Vec<f64>,
    times: Vec<f64>,
) -> PyResult<CumulativeDynamicAUCResult> {
    let n = time.len();
    if n != status.len() || n != marker.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "time, status, and marker must have the same length",
        ));
    }
    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "input arrays must not be empty",
        ));
    }
    if times.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "times array must not be empty",
        ));
    }

    for &t in &times {
        if t <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "all time points must be positive",
            ));
        }
    }

    Ok(cumulative_dynamic_auc_core(&time, &status, &marker, &times))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_dependent_auc_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let status = vec![1, 1, 0, 1, 0, 1, 1, 0, 1, 0];
        let marker = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05];

        let result = time_dependent_auc_core(&time, &status, &marker, 5.0);

        assert!(result.auc >= 0.0 && result.auc <= 1.0);
        assert!(result.n_cases > 0);
        assert!(result.n_controls > 0);
    }

    #[test]
    fn test_time_dependent_auc_perfect_discrimination() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0];
        let status = vec![1, 1, 1, 1, 0, 0, 0, 0];
        let marker = vec![0.9, 0.85, 0.8, 0.75, 0.2, 0.15, 0.1, 0.05];

        let result = time_dependent_auc_core(&time, &status, &marker, 5.0);

        assert!(result.auc > 0.9);
        assert_eq!(result.n_cases, 4);
        assert_eq!(result.n_controls, 4);
    }

    #[test]
    fn test_time_dependent_auc_random() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let status = vec![1, 0, 1, 0, 1, 0, 1, 0];
        let marker = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];

        let result = time_dependent_auc_core(&time, &status, &marker, 4.5);

        assert!((result.auc - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_cumulative_dynamic_auc() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let status = vec![1, 1, 0, 1, 0, 1, 1, 0, 1, 0];
        let marker = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05];
        let times = vec![3.0, 5.0, 7.0];

        let result = cumulative_dynamic_auc_core(&time, &status, &marker, &times);

        assert_eq!(result.times.len(), 3);
        assert_eq!(result.auc.len(), 3);
        assert!(result.mean_auc >= 0.0 && result.mean_auc <= 1.0);
        assert!(result.integrated_auc >= 0.0 && result.integrated_auc <= 1.0);
    }

    #[test]
    fn test_empty_input() {
        let time: Vec<f64> = vec![];
        let status: Vec<i32> = vec![];
        let marker: Vec<f64> = vec![];

        let result = time_dependent_auc_core(&time, &status, &marker, 5.0);

        assert_eq!(result.auc, 0.5);
        assert_eq!(result.n_cases, 0);
        assert_eq!(result.n_controls, 0);
    }

    #[test]
    fn test_no_cases() {
        let time = vec![10.0, 11.0, 12.0];
        let status = vec![0, 0, 0];
        let marker = vec![0.5, 0.6, 0.7];

        let result = time_dependent_auc_core(&time, &status, &marker, 5.0);

        assert_eq!(result.auc, 0.5);
        assert_eq!(result.n_cases, 0);
    }

    #[test]
    fn test_no_controls() {
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![1, 1, 1];
        let marker = vec![0.5, 0.6, 0.7];

        let result = time_dependent_auc_core(&time, &status, &marker, 5.0);

        assert_eq!(result.auc, 0.5);
        assert_eq!(result.n_controls, 0);
    }
}
