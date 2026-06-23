use crate::constants::{
    DIVISION_FLOOR, IPCW_SURVIVAL_FLOOR, PARALLEL_THRESHOLD_LARGE, clamped_normal_ci_95, same_time,
};
use crate::internal::statistical::{compute_censoring_km, km_step_prob_at};
use crate::internal::validation::{validate_binary_i32, validate_finite, validate_non_negative};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fmt;

const TIED_MARKER_TOLERANCE: f64 = DIVISION_FLOOR;

fn control_marker_counts(control_markers: &[f64], marker_value: f64) -> (usize, usize) {
    let lower_count = control_markers.partition_point(|&value| value < marker_value);
    let tied_upper =
        control_markers.partition_point(|&value| value - marker_value < TIED_MARKER_TOLERANCE);
    (lower_count, tied_upper.saturating_sub(lower_count))
}

#[derive(Debug, Clone)]
#[pyclass(str, get_all, from_py_object)]
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
#[pyclass(str, get_all, from_py_object)]
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

pub fn time_dependent_auc_core(
    time: &[f64],
    status: &[i32],
    marker: &[f64],
    t: f64,
) -> TimeDepAUCResult {
    let (cens_times, cens_km) = compute_censoring_km(time, status);
    time_dependent_auc_with_censoring(time, status, marker, t, &cens_times, &cens_km)
}

fn time_dependent_auc_with_censoring(
    time: &[f64],
    status: &[i32],
    marker: &[f64],
    t: f64,
    cens_times: &[f64],
    cens_km: &[f64],
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

    let min_g = IPCW_SURVIVAL_FLOOR;

    let mut cases: Vec<(usize, f64)> = Vec::new();
    let mut controls: Vec<usize> = Vec::new();

    for i in 0..n {
        let at_or_before_t = time[i] <= t || same_time(time[i], t);
        if at_or_before_t && status[i] == 1 {
            let g_ti = km_step_prob_at(time[i], cens_times, cens_km).max(min_g);
            let weight = 1.0 / g_ti;
            cases.push((i, weight));
        } else if time[i] > t && !same_time(time[i], t) {
            controls.push(i);
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

    let mut control_markers: Vec<f64> = controls
        .iter()
        .map(|&control_idx| marker[control_idx])
        .collect();
    control_markers.sort_by(f64::total_cmp);

    let compute_case_contribution = |case_idx: usize, case_weight: f64| -> (f64, f64) {
        let (lower_count, tied_count) = control_marker_counts(&control_markers, marker[case_idx]);
        let favorable = lower_count as f64 + 0.5 * tied_count as f64;
        (favorable * case_weight, n_controls as f64 * case_weight)
    };

    let (numerator, denominator) = if n_cases > PARALLEL_THRESHOLD_LARGE {
        cases
            .par_iter()
            .map(|&(case_idx, case_weight)| compute_case_contribution(case_idx, case_weight))
            .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1))
    } else {
        let mut num = 0.0;
        let mut den = 0.0;
        for &(case_idx, case_weight) in &cases {
            let (n, d) = compute_case_contribution(case_idx, case_weight);
            num += n;
            den += d;
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
    let (ci_lower, ci_upper) = clamped_normal_ci_95(auc, std_error, 0.0, 1.0);

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

    let (cens_times, cens_km) = compute_censoring_km(time, status);
    let compute_at_time =
        |t| time_dependent_auc_with_censoring(time, status, marker, t, &cens_times, &cens_km);

    let results: Vec<TimeDepAUCResult> = if times.len() > 4 {
        times.par_iter().map(|&t| compute_at_time(t)).collect()
    } else {
        times.iter().map(|&t| compute_at_time(t)).collect()
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

fn validate_auc_input_values(time: &[f64], status: &[i32], marker: &[f64]) -> PyResult<()> {
    validate_finite(time, "time")?;
    validate_non_negative(time, "time")?;
    validate_binary_i32(status, "status")?;
    validate_finite(marker, "marker")?;
    Ok(())
}

fn validate_positive_time_point(value: f64, field: &'static str) -> PyResult<()> {
    validate_finite(&[value], field)?;
    if value <= 0.0 {
        return Err(PyValueError::new_err(format!("{field} must be positive")));
    }
    Ok(())
}

fn validate_cumulative_auc_times(times: &[f64]) -> PyResult<()> {
    validate_finite(times, "times")?;
    for (index, &value) in times.iter().enumerate() {
        if value <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "times must contain only positive values; got {value} at index {index}"
            )));
        }
    }
    for (index, pair) in times.windows(2).enumerate() {
        if pair[1] < pair[0] {
            return Err(PyValueError::new_err(format!(
                "times must be sorted in nondecreasing order; index {} has {} before {}",
                index + 1,
                pair[1],
                pair[0]
            )));
        }
    }
    Ok(())
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
        return Err(PyValueError::new_err(
            "time, status, and marker must have the same length",
        ));
    }
    if n == 0 {
        return Err(PyValueError::new_err("input arrays must not be empty"));
    }
    validate_auc_input_values(&time, &status, &marker)?;
    validate_positive_time_point(t, "t")?;

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
        return Err(PyValueError::new_err(
            "time, status, and marker must have the same length",
        ));
    }
    if n == 0 {
        return Err(PyValueError::new_err("input arrays must not be empty"));
    }
    if times.is_empty() {
        return Err(PyValueError::new_err("times array must not be empty"));
    }

    validate_auc_input_values(&time, &status, &marker)?;
    validate_cumulative_auc_times(&times)?;

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
    fn test_cumulative_dynamic_auc_matches_direct_auc_calls_with_shared_censoring_state() {
        let time = vec![0.8, 1.0, 1.4, 2.0, 2.5, 3.0, 3.8, 4.2, 5.0, 5.5, 6.2, 7.0];
        let status = vec![1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0];
        let marker = vec![
            0.9, 0.2, 0.85, 0.7, 0.4, 0.65, 0.3, 0.6, 0.5, 0.1, 0.45, 0.05,
        ];
        let times = vec![1.0, 1.5, 2.5, 3.5, 4.5, 5.5];

        let cumulative = cumulative_dynamic_auc_core(&time, &status, &marker, &times);
        let direct: Vec<TimeDepAUCResult> = times
            .iter()
            .map(|&t| time_dependent_auc_core(&time, &status, &marker, t))
            .collect();

        assert_eq!(
            cumulative.n_cases,
            direct.iter().map(|r| r.n_cases).collect::<Vec<_>>()
        );
        assert_eq!(
            cumulative.n_controls,
            direct.iter().map(|r| r.n_controls).collect::<Vec<_>>()
        );
        for (actual_auc, expected) in cumulative.auc.iter().zip(direct.iter()) {
            assert!((actual_auc - expected.auc).abs() < 1e-12);
        }
    }

    #[test]
    fn test_control_marker_counts_preserves_tie_boundary() {
        let controls = vec![
            0.4,
            0.5,
            0.5 + TIED_MARKER_TOLERANCE / 2.0,
            0.5 + TIED_MARKER_TOLERANCE * 2.0,
            0.7,
        ];

        let (lower_count, tied_count) = control_marker_counts(&controls, 0.5);

        assert_eq!(lower_count, 1);
        assert_eq!(tied_count, 2);
    }

    #[test]
    fn test_time_dependent_auc_large_case_set_uses_sorted_counting() {
        let n_cases = PARALLEL_THRESHOLD_LARGE + 5;
        let n_controls = 10;
        let mut time = Vec::with_capacity(n_cases + n_controls);
        let mut status = Vec::with_capacity(n_cases + n_controls);
        let mut marker = Vec::with_capacity(n_cases + n_controls);

        for i in 0..n_cases {
            time.push(1.0);
            status.push(1);
            marker.push(1.0 + i as f64 * 1e-6);
        }
        for i in 0..n_controls {
            time.push(2.0 + i as f64 * 1e-6);
            status.push(0);
            marker.push(i as f64 * 1e-6);
        }

        let result = time_dependent_auc_core(&time, &status, &marker, 1.0);

        assert_eq!(result.n_cases, n_cases);
        assert_eq!(result.n_controls, n_controls);
        assert!((result.auc - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_time_dependent_auc_counts_near_tied_events_at_eval_time() {
        let exact_time = vec![1.0, 1.0, 2.0, 3.0];
        let near_time = vec![1.0, 1.0 + crate::constants::TIME_EPSILON / 2.0, 2.0, 3.0];
        let status = vec![1, 1, 0, 0];
        let marker = vec![0.9, 0.8, 0.2, 0.1];

        let expected = time_dependent_auc_core(&exact_time, &status, &marker, 1.0);
        let actual = time_dependent_auc_core(&near_time, &status, &marker, 1.0);

        assert_eq!(actual.n_cases, expected.n_cases);
        assert_eq!(actual.n_controls, expected.n_controls);
        assert!((actual.auc - expected.auc).abs() < 1e-12);
        assert!((actual.std_error - expected.std_error).abs() < 1e-12);
        assert!((actual.ci_lower - expected.ci_lower).abs() < 1e-12);
        assert!((actual.ci_upper - expected.ci_upper).abs() < 1e-12);
    }

    #[test]
    fn test_cumulative_dynamic_auc_counts_near_tied_events_at_eval_time() {
        let exact_time = vec![1.0, 1.0, 2.0, 3.0];
        let near_time = vec![1.0, 1.0 + crate::constants::TIME_EPSILON / 2.0, 2.0, 3.0];
        let status = vec![1, 1, 0, 0];
        let marker = vec![0.9, 0.8, 0.2, 0.1];
        let times = vec![1.0, 2.0];

        let expected = cumulative_dynamic_auc_core(&exact_time, &status, &marker, &times);
        let actual = cumulative_dynamic_auc_core(&near_time, &status, &marker, &times);

        assert_eq!(actual.n_cases, expected.n_cases);
        assert_eq!(actual.n_controls, expected.n_controls);
        for (actual_auc, expected_auc) in actual.auc.iter().zip(expected.auc.iter()) {
            assert!((actual_auc - expected_auc).abs() < 1e-12);
        }
        assert!((actual.mean_auc - expected.mean_auc).abs() < 1e-12);
        assert!((actual.integrated_auc - expected.integrated_auc).abs() < 1e-12);
    }

    #[test]
    fn public_time_dependent_auc_validates_numeric_inputs() {
        let err = time_dependent_auc(vec![f64::NAN], vec![1], vec![0.5], 1.0).unwrap_err();
        assert!(err.to_string().contains("time contains non-finite"));

        let err = time_dependent_auc(vec![-1.0], vec![1], vec![0.5], 1.0).unwrap_err();
        assert!(err.to_string().contains("time contains negative value"));

        let err = time_dependent_auc(vec![1.0], vec![2], vec![0.5], 1.0).unwrap_err();
        assert!(err.to_string().contains("status must contain only 0/1"));

        let err = time_dependent_auc(vec![1.0], vec![1], vec![f64::INFINITY], 1.0).unwrap_err();
        assert!(err.to_string().contains("marker contains non-finite"));

        let err = time_dependent_auc(vec![1.0], vec![1], vec![0.5], f64::NAN).unwrap_err();
        assert!(err.to_string().contains("t contains non-finite"));
    }

    #[test]
    fn public_cumulative_dynamic_auc_validates_eval_times() {
        let err =
            cumulative_dynamic_auc(vec![1.0, 2.0], vec![1, 0], vec![0.8, 0.2], vec![2.0, 1.0])
                .unwrap_err();
        assert!(err.to_string().contains("times must be sorted"));

        let err =
            cumulative_dynamic_auc(vec![1.0], vec![1], vec![0.5], vec![f64::INFINITY]).unwrap_err();
        assert!(err.to_string().contains("times contains non-finite"));

        let err = cumulative_dynamic_auc(vec![1.0], vec![1], vec![0.5], vec![0.0]).unwrap_err();
        assert!(err.to_string().contains("times must contain only positive"));
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
