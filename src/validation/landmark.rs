//! Landmark, conditional-survival and life-table summaries of a
//! Kaplan-Meier curve (no direct R `survival` counterpart).  Every curve
//! is taken from `surv_analysis::survfitkm`.

use crate::constants::{
    PARALLEL_THRESHOLD_SMALL, clamped_normal_ci, exp_ci, same_time, z_score_for_confidence,
};
use crate::internal::dist::pnorm;
use crate::internal::validation::{
    validate_binary_i32, validate_confidence_level, validate_finite, validate_no_nan,
};
use crate::surv_analysis::{SurvfitKMData, SurvfitKMOptions, survfitkm};
use crate::validation::logrank::logrank_test;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// One event time of a Kaplan-Meier curve: the survival, the Greenwood
/// sum `sum d / (n (n - d))`, the number at risk and the cumulative
/// number of events.
struct KmStep {
    time: f64,
    survival: f64,
    greenwood: f64,
    n_risk: usize,
    cumulative_events: usize,
}

/// The Kaplan-Meier event times of right-censored data.
fn kaplan_meier_steps(time: &[f64], status: &[i32]) -> Vec<KmStep> {
    let Ok(data) = SurvfitKMData::right_censored(time.to_vec(), status.to_vec()) else {
        return Vec::new();
    };
    let options = SurvfitKMOptions {
        se_fit: false,
        ..SurvfitKMOptions::default()
    };
    let Ok(km) = survfitkm(&data, &options) else {
        return Vec::new();
    };
    let mut greenwood = 0.0;
    let mut cumulative_events = 0usize;
    let mut steps = Vec::new();
    for i in 0..km.time.len() {
        let (n_risk, d) = (km.n_risk[i], km.n_event[i]);
        if d <= 0.0 {
            continue;
        }
        cumulative_events += d as usize;
        if n_risk > d {
            greenwood += d / (n_risk * (n_risk - d));
        }
        steps.push(KmStep {
            time: km.time[i],
            survival: km.surv[i],
            greenwood,
            n_risk: n_risk as usize,
            cumulative_events,
        });
    }
    steps
}

/// The last step at or before `at` (with the near-tie tolerance).
fn step_at(steps: &[KmStep], at: f64) -> Option<&KmStep> {
    let idx = steps.partition_point(|step| step.time <= at || same_time(step.time, at));
    idx.checked_sub(1).map(|i| &steps[i])
}
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct LandmarkResult {
    #[pyo3(get)]
    pub landmark_time: f64,
    #[pyo3(get)]
    pub n_at_risk: usize,
    #[pyo3(get)]
    pub n_excluded: usize,
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub status: Vec<i32>,
    #[pyo3(get)]
    pub original_indices: Vec<usize>,
}
#[pymethods]
impl LandmarkResult {
    #[new]
    fn new(
        landmark_time: f64,
        n_at_risk: usize,
        n_excluded: usize,
        time: Vec<f64>,
        status: Vec<i32>,
        original_indices: Vec<usize>,
    ) -> Self {
        Self {
            landmark_time,
            n_at_risk,
            n_excluded,
            time,
            status,
            original_indices,
        }
    }
}

fn validate_landmark_survival_inputs(time: &[f64], status: &[i32]) -> PyResult<()> {
    if status.len() != time.len() {
        return Err(PyValueError::new_err(format!(
            "time and status must have same length, got {} and {}",
            time.len(),
            status.len()
        )));
    }
    validate_no_nan(time, "time")?;
    validate_finite(time, "time")?;
    validate_binary_i32(status, "status")?;
    Ok(())
}

fn validate_finite_scalar(value: f64, name: &str) -> PyResult<()> {
    if !value.is_finite() {
        return Err(PyValueError::new_err(format!("{name} must be finite")));
    }
    Ok(())
}

fn validate_confidence_option(confidence_level: Option<f64>) -> PyResult<f64> {
    let confidence = confidence_level.unwrap_or(0.95);
    validate_confidence_level(confidence)?;
    Ok(confidence)
}

pub(crate) fn compute_landmark(time: &[f64], status: &[i32], landmark_time: f64) -> LandmarkResult {
    let n = time.len();
    let mut new_time = Vec::new();
    let mut new_status = Vec::new();
    let mut original_indices = Vec::new();
    let mut n_excluded = 0usize;
    for i in 0..n {
        if time[i] > landmark_time {
            new_time.push(time[i] - landmark_time);
            new_status.push(status[i]);
            original_indices.push(i);
        } else {
            n_excluded += 1;
        }
    }
    let n_at_risk = new_time.len();
    LandmarkResult {
        landmark_time,
        n_at_risk,
        n_excluded,
        time: new_time,
        status: new_status,
        original_indices,
    }
}
#[pyfunction]
pub fn landmark_analysis(
    time: Vec<f64>,
    status: Vec<i32>,
    landmark_time: f64,
) -> PyResult<LandmarkResult> {
    validate_landmark_survival_inputs(&time, &status)?;
    validate_finite_scalar(landmark_time, "landmark_time")?;
    Ok(compute_landmark(&time, &status, landmark_time))
}
pub(crate) fn compute_landmarks_parallel(
    time: &[f64],
    status: &[i32],
    landmark_times: &[f64],
) -> Vec<LandmarkResult> {
    landmark_times
        .par_iter()
        .map(|&lt| compute_landmark(time, status, lt))
        .collect()
}
#[pyfunction]
pub fn landmark_analysis_batch(
    time: Vec<f64>,
    status: Vec<i32>,
    landmark_times: Vec<f64>,
) -> PyResult<Vec<LandmarkResult>> {
    validate_landmark_survival_inputs(&time, &status)?;
    validate_no_nan(&landmark_times, "landmark_times")?;
    validate_finite(&landmark_times, "landmark_times")?;
    Ok(compute_landmarks_parallel(&time, &status, &landmark_times))
}
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct ConditionalSurvivalResult {
    #[pyo3(get)]
    pub given_time: f64,
    #[pyo3(get)]
    pub target_time: f64,
    #[pyo3(get)]
    pub conditional_survival: f64,
    #[pyo3(get)]
    pub ci_lower: f64,
    #[pyo3(get)]
    pub ci_upper: f64,
    #[pyo3(get)]
    pub n_at_risk: usize,
}
#[pymethods]
impl ConditionalSurvivalResult {
    #[new]
    fn new(
        given_time: f64,
        target_time: f64,
        conditional_survival: f64,
        ci_lower: f64,
        ci_upper: f64,
        n_at_risk: usize,
    ) -> Self {
        Self {
            given_time,
            target_time,
            conditional_survival,
            ci_lower,
            ci_upper,
            n_at_risk,
        }
    }
}
pub(crate) fn compute_conditional_survival(
    time: &[f64],
    status: &[i32],
    given_time: f64,
    target_time: f64,
    confidence_level: f64,
) -> ConditionalSurvivalResult {
    let n = time.len();
    if n == 0 || target_time <= given_time {
        return ConditionalSurvivalResult {
            given_time,
            target_time,
            conditional_survival: 1.0,
            ci_lower: 1.0,
            ci_upper: 1.0,
            n_at_risk: 0,
        };
    }
    let steps = kaplan_meier_steps(time, status);
    let (surv_given, var_given) =
        step_at(&steps, given_time).map_or((1.0, 0.0), |step| (step.survival, step.greenwood));
    let (surv_target, var_target) =
        step_at(&steps, target_time).map_or((1.0, 0.0), |step| (step.survival, step.greenwood));
    // number still at risk just after the given time
    let n_at_given = time
        .iter()
        .filter(|&&t| t > given_time && !same_time(t, given_time))
        .count();
    let conditional = if surv_given > 0.0 {
        surv_target / surv_given
    } else {
        0.0
    };
    let z = z_score_for_confidence(confidence_level);
    let var_conditional = if surv_given > 0.0 {
        conditional * conditional * (var_target - var_given).abs()
    } else {
        0.0
    };
    let se = var_conditional.sqrt();
    let (ci_lower, ci_upper) = clamped_normal_ci(conditional, se, z, 0.0, 1.0);
    ConditionalSurvivalResult {
        given_time,
        target_time,
        conditional_survival: conditional,
        ci_lower,
        ci_upper,
        n_at_risk: n_at_given,
    }
}
#[pyfunction]
#[pyo3(signature = (time, status, given_time, target_time, confidence_level=None))]
pub fn conditional_survival(
    time: Vec<f64>,
    status: Vec<i32>,
    given_time: f64,
    target_time: f64,
    confidence_level: Option<f64>,
) -> PyResult<ConditionalSurvivalResult> {
    validate_landmark_survival_inputs(&time, &status)?;
    validate_finite_scalar(given_time, "given_time")?;
    validate_finite_scalar(target_time, "target_time")?;
    let conf = validate_confidence_option(confidence_level)?;
    Ok(compute_conditional_survival(
        &time,
        &status,
        given_time,
        target_time,
        conf,
    ))
}
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct HazardRatioResult {
    #[pyo3(get)]
    pub hazard_ratio: f64,
    #[pyo3(get)]
    pub ci_lower: f64,
    #[pyo3(get)]
    pub ci_upper: f64,
    #[pyo3(get)]
    pub se_log_hr: f64,
    #[pyo3(get)]
    pub z_statistic: f64,
    #[pyo3(get)]
    pub p_value: f64,
}
#[pymethods]
impl HazardRatioResult {
    #[new]
    fn new(
        hazard_ratio: f64,
        ci_lower: f64,
        ci_upper: f64,
        se_log_hr: f64,
        z_statistic: f64,
        p_value: f64,
    ) -> Self {
        Self {
            hazard_ratio,
            ci_lower,
            ci_upper,
            se_log_hr,
            z_statistic,
            p_value,
        }
    }
}
/// Log-rank (Peto) estimate of the hazard ratio of the second group
/// against the first: `exp((O - E) / V)` with the log-rank variance `V`.
pub(crate) fn compute_hazard_ratio(
    time: &[f64],
    status: &[i32],
    group: &[i32],
    confidence_level: f64,
) -> HazardRatioResult {
    let neutral = HazardRatioResult {
        hazard_ratio: 1.0,
        ci_lower: 1.0,
        ci_upper: 1.0,
        se_log_hr: 0.0,
        z_statistic: 0.0,
        p_value: 1.0,
    };
    let mut unique_groups: Vec<i32> = group.to_vec();
    unique_groups.sort_unstable();
    unique_groups.dedup();
    if unique_groups.len() < 2 {
        return neutral;
    }
    // Only the first two groups take part in the comparison.
    let rows: Vec<usize> = (0..time.len())
        .filter(|&i| group[i] == unique_groups[0] || group[i] == unique_groups[1])
        .collect();
    let time: Vec<f64> = rows.iter().map(|&i| time[i]).collect();
    let status: Vec<i32> = rows.iter().map(|&i| status[i]).collect();
    let group: Vec<i32> = rows.iter().map(|&i| group[i]).collect();
    let Ok(test) = logrank_test(&time, &status, &group, None, None, 0.0, true) else {
        return neutral;
    };
    let sum_o_e = test.observed[0] - test.expected[0];
    let sum_var = test.variance[0][0];
    let log_hr: f64 = if sum_var > 0.0 {
        sum_o_e / sum_var
    } else {
        0.0
    };
    let hazard_ratio = log_hr.exp();
    let se_log_hr: f64 = if sum_var > 0.0 {
        1.0 / sum_var.sqrt()
    } else {
        0.0
    };
    let z = z_score_for_confidence(confidence_level);
    let (ci_lower, ci_upper) = exp_ci(log_hr, se_log_hr, z);
    let z_statistic: f64 = if se_log_hr > 0.0 {
        log_hr / se_log_hr
    } else {
        0.0
    };
    let p_value = 2.0 * pnorm(z_statistic.abs(), false, false);
    HazardRatioResult {
        hazard_ratio,
        ci_lower,
        ci_upper,
        se_log_hr,
        z_statistic,
        p_value,
    }
}
#[pyfunction]
#[pyo3(signature = (time, status, group, confidence_level=None))]
pub fn hazard_ratio(
    time: Vec<f64>,
    status: Vec<i32>,
    group: Vec<i32>,
    confidence_level: Option<f64>,
) -> PyResult<HazardRatioResult> {
    validate_landmark_survival_inputs(&time, &status)?;
    if group.len() != time.len() {
        return Err(PyValueError::new_err(format!(
            "group must have same length as time, got {} and {}",
            group.len(),
            time.len()
        )));
    }
    let conf = validate_confidence_option(confidence_level)?;
    Ok(compute_hazard_ratio(&time, &status, &group, conf))
}
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SurvivalAtTimeResult {
    #[pyo3(get)]
    pub time: f64,
    #[pyo3(get)]
    pub survival: f64,
    #[pyo3(get)]
    pub ci_lower: f64,
    #[pyo3(get)]
    pub ci_upper: f64,
    #[pyo3(get)]
    pub n_at_risk: usize,
    #[pyo3(get)]
    pub n_events: usize,
}
#[pymethods]
impl SurvivalAtTimeResult {
    #[new]
    fn new(
        time: f64,
        survival: f64,
        ci_lower: f64,
        ci_upper: f64,
        n_at_risk: usize,
        n_events: usize,
    ) -> Self {
        Self {
            time,
            survival,
            ci_lower,
            ci_upper,
            n_at_risk,
            n_events,
        }
    }
}
pub(crate) fn compute_survival_at_times(
    time: &[f64],
    status: &[i32],
    eval_times: &[f64],
    confidence_level: f64,
) -> Vec<SurvivalAtTimeResult> {
    let n = time.len();
    let steps = if n == 0 {
        Vec::new()
    } else {
        kaplan_meier_steps(time, status)
    };
    let z = z_score_for_confidence(confidence_level);
    let evaluate = |t: f64| {
        let (survival, var, n_risk, n_events) = match step_at(&steps, t) {
            Some(step) => (
                step.survival,
                step.survival * step.survival * step.greenwood,
                step.n_risk,
                step.cumulative_events,
            ),
            None => (1.0, 0.0, n, 0),
        };
        let (ci_lower, ci_upper) = clamped_normal_ci(survival, var.sqrt(), z, 0.0, 1.0);
        SurvivalAtTimeResult {
            time: t,
            survival,
            ci_lower,
            ci_upper,
            n_at_risk: n_risk,
            n_events,
        }
    };
    if eval_times.len() > PARALLEL_THRESHOLD_SMALL {
        eval_times.par_iter().map(|&t| evaluate(t)).collect()
    } else {
        eval_times.iter().map(|&t| evaluate(t)).collect()
    }
}
#[pyfunction]
#[pyo3(signature = (time, status, eval_times, confidence_level=None))]
pub fn survival_at_times(
    time: Vec<f64>,
    status: Vec<i32>,
    eval_times: Vec<f64>,
    confidence_level: Option<f64>,
) -> PyResult<Vec<SurvivalAtTimeResult>> {
    validate_landmark_survival_inputs(&time, &status)?;
    validate_no_nan(&eval_times, "eval_times")?;
    validate_finite(&eval_times, "eval_times")?;
    let conf = validate_confidence_option(confidence_level)?;
    Ok(compute_survival_at_times(&time, &status, &eval_times, conf))
}
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct LifeTableResult {
    #[pyo3(get)]
    pub interval_start: Vec<f64>,
    #[pyo3(get)]
    pub interval_end: Vec<f64>,
    #[pyo3(get)]
    pub n_at_risk: Vec<f64>,
    #[pyo3(get)]
    pub n_deaths: Vec<f64>,
    #[pyo3(get)]
    pub n_censored: Vec<f64>,
    #[pyo3(get)]
    pub n_effective: Vec<f64>,
    #[pyo3(get)]
    pub hazard: Vec<f64>,
    #[pyo3(get)]
    pub survival: Vec<f64>,
    #[pyo3(get)]
    pub se_survival: Vec<f64>,
}
#[pymethods]
impl LifeTableResult {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        interval_start: Vec<f64>,
        interval_end: Vec<f64>,
        n_at_risk: Vec<f64>,
        n_deaths: Vec<f64>,
        n_censored: Vec<f64>,
        n_effective: Vec<f64>,
        hazard: Vec<f64>,
        survival: Vec<f64>,
        se_survival: Vec<f64>,
    ) -> Self {
        Self {
            interval_start,
            interval_end,
            n_at_risk,
            n_deaths,
            n_censored,
            n_effective,
            hazard,
            survival,
            se_survival,
        }
    }
}
pub(crate) fn compute_life_table(time: &[f64], status: &[i32], breaks: &[f64]) -> LifeTableResult {
    let n = time.len();
    let n_intervals = breaks.len().saturating_sub(1);
    if n == 0 || n_intervals == 0 {
        return LifeTableResult {
            interval_start: vec![],
            interval_end: vec![],
            n_at_risk: vec![],
            n_deaths: vec![],
            n_censored: vec![],
            n_effective: vec![],
            hazard: vec![],
            survival: vec![],
            se_survival: vec![],
        };
    }
    let mut interval_start = Vec::with_capacity(n_intervals);
    let mut interval_end = Vec::with_capacity(n_intervals);
    let mut n_deaths = vec![0.0; n_intervals];
    let mut n_censored = vec![0.0; n_intervals];
    for i in 0..n_intervals {
        interval_start.push(breaks[i]);
        interval_end.push(breaks[i + 1]);
    }
    for i in 0..n {
        let t = time[i];
        for j in 0..n_intervals {
            let is_final_interval = j + 1 == n_intervals;
            if t >= breaks[j]
                && (t < breaks[j + 1] || (is_final_interval && same_time(t, breaks[j + 1])))
            {
                if status[i] == 1 {
                    n_deaths[j] += 1.0;
                } else {
                    n_censored[j] += 1.0;
                }
                break;
            }
        }
    }
    let mut n_at_risk = Vec::with_capacity(n_intervals);
    let mut remaining = n as f64;
    for j in 0..n_intervals {
        n_at_risk.push(remaining);
        remaining -= n_deaths[j] + n_censored[j];
    }
    let n_effective: Vec<f64> = (0..n_intervals)
        .map(|j| n_at_risk[j] - n_censored[j] / 2.0)
        .collect();
    let hazard: Vec<f64> = (0..n_intervals)
        .map(|j| {
            if n_effective[j] > 0.0 {
                n_deaths[j] / n_effective[j]
            } else {
                0.0
            }
        })
        .collect();
    let mut survival = Vec::with_capacity(n_intervals);
    let mut se_survival = Vec::with_capacity(n_intervals);
    let mut surv = 1.0;
    let mut var_sum = 0.0;
    for j in 0..n_intervals {
        surv *= 1.0 - hazard[j];
        survival.push(surv);
        if n_effective[j] > 0.0 && n_effective[j] > n_deaths[j] {
            var_sum += n_deaths[j] / (n_effective[j] * (n_effective[j] - n_deaths[j]));
        }
        se_survival.push(surv * var_sum.sqrt());
    }
    LifeTableResult {
        interval_start,
        interval_end,
        n_at_risk,
        n_deaths,
        n_censored,
        n_effective,
        hazard,
        survival,
        se_survival,
    }
}

fn validate_life_table_inputs(time: &[f64], status: &[i32], breaks: &[f64]) -> PyResult<()> {
    if status.len() != time.len() {
        return Err(PyValueError::new_err(format!(
            "time and status must have same length, got {} and {}",
            time.len(),
            status.len()
        )));
    }
    if breaks.len() < 2 {
        return Err(PyValueError::new_err(
            "breaks must define at least one interval",
        ));
    }

    validate_no_nan(time, "time")?;
    validate_finite(time, "time")?;
    validate_no_nan(breaks, "breaks")?;
    validate_finite(breaks, "breaks")?;
    validate_binary_i32(status, "status")?;

    for (index, window) in breaks.windows(2).enumerate() {
        if window[1] <= window[0] || same_time(window[0], window[1]) {
            return Err(PyValueError::new_err(format!(
                "breaks must be strictly increasing; got {} then {} at positions {} and {}",
                window[0],
                window[1],
                index,
                index + 1
            )));
        }
    }

    let first = breaks[0];
    let last = breaks[breaks.len() - 1];
    for (index, &value) in time.iter().enumerate() {
        if (value < first && !same_time(value, first)) || (value > last && !same_time(value, last))
        {
            return Err(PyValueError::new_err(format!(
                "time values must fall within the break range; got {} at index {} outside [{}, {}]",
                value, index, first, last
            )));
        }
    }

    Ok(())
}

#[pyfunction]
pub fn life_table(time: Vec<f64>, status: Vec<i32>, breaks: Vec<f64>) -> PyResult<LifeTableResult> {
    validate_life_table_inputs(&time, &status, &breaks)?;
    Ok(compute_life_table(&time, &status, &breaks))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_landmark_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 0, 1];
        let landmark_time = 2.0;

        let result = compute_landmark(&time, &status, landmark_time);

        assert_eq!(result.landmark_time, 2.0);
        assert_eq!(result.n_at_risk, 3);
        assert_eq!(result.n_excluded, 2);
        assert_eq!(result.time.len(), 3);
    }

    #[test]
    fn test_compute_landmark_all_excluded() {
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![1, 1, 1];
        let landmark_time = 5.0;

        let result = compute_landmark(&time, &status, landmark_time);

        assert_eq!(result.n_at_risk, 0);
        assert_eq!(result.n_excluded, 3);
    }

    #[test]
    fn test_compute_landmark_none_excluded() {
        let time = vec![5.0, 6.0, 7.0];
        let status = vec![1, 0, 1];
        let landmark_time = 1.0;

        let result = compute_landmark(&time, &status, landmark_time);

        assert_eq!(result.n_at_risk, 3);
        assert_eq!(result.n_excluded, 0);
        assert_eq!(result.time[0], 4.0);
    }

    #[test]
    fn test_compute_landmarks_parallel() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 0, 1];
        let landmarks = vec![1.0, 2.0, 3.0];

        let results = compute_landmarks_parallel(&time, &status, &landmarks);

        assert_eq!(results.len(), 3);
        assert!(results[0].n_at_risk >= results[1].n_at_risk);
        assert!(results[1].n_at_risk >= results[2].n_at_risk);
    }

    #[test]
    fn test_landmark_public_wrappers_reject_malformed_inputs() {
        let err = landmark_analysis(vec![1.0], vec![], 0.5).unwrap_err();
        assert!(
            err.to_string()
                .contains("time and status must have same length")
        );

        let err = landmark_analysis(vec![f64::NAN], vec![1], 0.5).unwrap_err();
        assert!(err.to_string().contains("time contains NaN"));

        let err = landmark_analysis(vec![1.0], vec![2], 0.5).unwrap_err();
        assert!(
            err.to_string()
                .contains("status must contain only 0/1 values")
        );

        let err = landmark_analysis_batch(vec![1.0], vec![1], vec![f64::INFINITY]).unwrap_err();
        assert!(
            err.to_string()
                .contains("landmark_times contains non-finite")
        );

        let err = conditional_survival(vec![1.0], vec![1], f64::NAN, 2.0, None).unwrap_err();
        assert!(err.to_string().contains("given_time must be finite"));

        let err = conditional_survival(vec![1.0], vec![1], 0.5, 2.0, Some(1.0)).unwrap_err();
        assert!(err.to_string().contains("confidence_level"));

        let err = hazard_ratio(vec![1.0], vec![1], vec![], None).unwrap_err();
        assert!(err.to_string().contains("group must have same length"));

        let err = survival_at_times(vec![1.0], vec![1], vec![f64::NAN], None).unwrap_err();
        assert!(err.to_string().contains("eval_times contains NaN"));
    }

    #[test]
    fn test_conditional_survival_groups_near_tied_event_times() {
        let exact_time = vec![1.0, 1.0, 2.0, 3.0];
        let near_time = vec![1.0, 1.0 + crate::constants::TIME_EPSILON / 2.0, 2.0, 3.0];
        let status = vec![1, 1, 0, 0];

        let expected = compute_conditional_survival(&exact_time, &status, 1.0, 2.0, 0.95);
        let actual = compute_conditional_survival(&near_time, &status, 1.0, 2.0, 0.95);

        assert!((actual.conditional_survival - expected.conditional_survival).abs() < 1e-12);
        assert!((actual.ci_lower - expected.ci_lower).abs() < 1e-12);
        assert!((actual.ci_upper - expected.ci_upper).abs() < 1e-12);
        assert_eq!(actual.n_at_risk, expected.n_at_risk);
    }

    #[test]
    fn test_hazard_ratio_groups_near_tied_event_times() {
        let exact_time = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        let near_time = vec![
            1.0,
            1.0 + crate::constants::TIME_EPSILON / 2.0,
            2.0,
            2.0 + crate::constants::TIME_EPSILON / 2.0,
            3.0,
            3.0,
        ];
        let status = vec![1, 1, 1, 0, 0, 0];
        let group = vec![0, 1, 0, 1, 0, 1];

        let expected = compute_hazard_ratio(&exact_time, &status, &group, 0.95);
        let actual = compute_hazard_ratio(&near_time, &status, &group, 0.95);

        assert!((actual.hazard_ratio - expected.hazard_ratio).abs() < 1e-12);
        assert!((actual.se_log_hr - expected.se_log_hr).abs() < 1e-12);
        assert!((actual.z_statistic - expected.z_statistic).abs() < 1e-12);
        assert!((actual.p_value - expected.p_value).abs() < 1e-12);
    }

    #[test]
    fn test_survival_at_times_groups_near_tied_event_times() {
        let exact_time = vec![1.0, 1.0, 2.0, 3.0];
        let near_time = vec![1.0, 1.0 + crate::constants::TIME_EPSILON / 2.0, 2.0, 3.0];
        let status = vec![1, 1, 0, 0];

        let expected = compute_survival_at_times(&exact_time, &status, &[1.0, 2.0], 0.95);
        let actual = compute_survival_at_times(&near_time, &status, &[1.0, 2.0], 0.95);

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual.survival - expected.survival).abs() < 1e-12);
            assert!((actual.ci_lower - expected.ci_lower).abs() < 1e-12);
            assert!((actual.ci_upper - expected.ci_upper).abs() < 1e-12);
            assert_eq!(actual.n_at_risk, expected.n_at_risk);
            assert_eq!(actual.n_events, expected.n_events);
        }
    }

    #[test]
    fn test_compute_life_table_basic() {
        let time = vec![1.5, 2.5, 3.5, 4.5, 5.5];
        let status = vec![1, 1, 0, 1, 0];
        let breaks = vec![0.0, 2.0, 4.0, 6.0];

        let result = compute_life_table(&time, &status, &breaks);

        assert_eq!(result.interval_start.len(), 3);
        assert_eq!(result.survival.len(), 3);
        assert!(result.survival.iter().all(|&s| (0.0..=1.0).contains(&s)));
    }

    #[test]
    fn test_compute_life_table_no_events() {
        let time = vec![1.5, 3.5, 5.5];
        let status = vec![0, 0, 0];
        let breaks = vec![0.0, 2.0, 4.0, 6.0];

        let result = compute_life_table(&time, &status, &breaks);

        assert_eq!(result.interval_start.len(), 3);
        assert!(result.n_deaths.iter().all(|&d| d == 0.0));
        assert!(result.survival.iter().all(|&s| s == 1.0));
    }

    #[test]
    fn test_life_table_includes_final_break() {
        let result = life_table(vec![2.0], vec![1], vec![0.0, 1.0, 2.0]).unwrap();

        assert_eq!(result.n_deaths, vec![0.0, 1.0]);
        assert_eq!(result.n_censored, vec![0.0, 0.0]);
    }

    #[test]
    fn test_life_table_rejects_malformed_public_inputs() {
        let err = life_table(vec![1.0], vec![], vec![0.0, 2.0]).unwrap_err();
        assert!(
            err.to_string()
                .contains("time and status must have same length")
        );

        let err = life_table(vec![1.0], vec![1], vec![0.0]).unwrap_err();
        assert!(
            err.to_string()
                .contains("breaks must define at least one interval")
        );

        let err = life_table(vec![f64::NAN], vec![1], vec![0.0, 2.0]).unwrap_err();
        assert!(err.to_string().contains("time contains NaN"));

        let err = life_table(vec![1.0], vec![2], vec![0.0, 2.0]).unwrap_err();
        assert!(
            err.to_string()
                .contains("status must contain only 0/1 values")
        );

        let err = life_table(vec![1.0], vec![1], vec![0.0, 0.0]).unwrap_err();
        assert!(
            err.to_string()
                .contains("breaks must be strictly increasing")
        );

        let err = life_table(vec![3.0], vec![1], vec![0.0, 2.0]).unwrap_err();
        assert!(
            err.to_string()
                .contains("time values must fall within the break range")
        );
    }

    #[test]
    fn test_landmark_result_new() {
        let result = LandmarkResult::new(2.0, 5, 3, vec![1.0, 2.0], vec![1, 0], vec![3, 4]);

        assert_eq!(result.landmark_time, 2.0);
        assert_eq!(result.n_at_risk, 5);
        assert_eq!(result.n_excluded, 3);
        assert_eq!(result.time.len(), 2);
    }
}
