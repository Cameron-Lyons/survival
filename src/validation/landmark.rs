//! Landmark, conditional-survival and life-table summaries of a
//! Kaplan-Meier curve (no direct R `survival` counterpart).  Every curve
//! is taken from `surv_analysis::survfitkm`, and the observation times are
//! binned together with the query times (landmark, evaluation and break
//! points) by `aeqSurv` (`data_prep::aeq_times`), so near ties are ties
//! exactly as `survfit`'s `timefix` makes them.
//!
//! The `compute_*` functions are the kernels; the `*_py` functions are
//! the Python entry points and validate the raw arguments.

use crate::constants::{
    PARALLEL_THRESHOLD_SMALL, clamped_normal_ci, exp_ci, z_score_for_confidence,
};
use crate::data_prep::aeq_times;
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::dist::pnorm;
use crate::internal::validation::{validate_binary_i32, validate_finite, validate_no_nan};
use crate::surv_analysis::{SurvfitKMData, SurvfitKMOptions, survfitkm};
use crate::validation::logrank::logrank_test;
use pyo3::prelude::*;
use rayon::prelude::*;

/// The observation times and the `query` times after one joint `aeqSurv`:
/// a query within tolerance of an observation time compares equal to it.
fn bin_times(time: &[f64], query: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut all = Vec::with_capacity(time.len() + query.len());
    all.extend_from_slice(time);
    all.extend_from_slice(query);
    let mut binned = aeq_times(&all);
    let query = binned.split_off(time.len());
    (binned, query)
}

/// One event time of a Kaplan-Meier curve: the survival, the Greenwood
/// sum `sum d / (n (n - d))` (`std.err^2` of `log S`), the number at risk
/// and the cumulative number of events.
struct KmStep {
    time: f64,
    survival: f64,
    greenwood: f64,
    n_risk: usize,
    cumulative_events: usize,
}

/// The Kaplan-Meier event times of right-censored data whose times are
/// already binned.
fn kaplan_meier_steps(time: &[f64], status: &[i32]) -> SurvivalResult<Vec<KmStep>> {
    let data = SurvfitKMData::right_censored(time.to_vec(), status.to_vec())?;
    let options = SurvfitKMOptions {
        conf_type: crate::surv_analysis::ConfType::None,
        timefix: false,
        ..SurvfitKMOptions::default()
    };
    let km = survfitkm(&data, &options)?;
    let std_err = km.std_err.as_deref().expect("se.fit is on");
    let mut cumulative_events = 0usize;
    let steps = (0..km.time.len())
        .filter(|&i| km.n_event[i] > 0.0)
        .map(|i| {
            cumulative_events += km.n_event[i] as usize;
            KmStep {
                time: km.time[i],
                survival: km.surv[i],
                greenwood: std_err[i] * std_err[i],
                n_risk: km.n_risk[i] as usize,
                cumulative_events,
            }
        })
        .collect();
    Ok(steps)
}

/// The last step at or before `at`.
fn step_at(steps: &[KmStep], at: f64) -> Option<&KmStep> {
    let idx = steps.partition_point(|step| step.time <= at);
    idx.checked_sub(1).map(|i| &steps[i])
}

/// Greenwood variance of `S(t)` from a step: `S^2 * var(log S)`, 0 once
/// the curve has reached 0.
fn survival_variance(step: &KmStep) -> f64 {
    if step.survival > 0.0 && step.greenwood.is_finite() {
        step.survival * step.survival * step.greenwood
    } else {
        0.0
    }
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

fn validate_survival_inputs(time: &[f64], status: &[i32]) -> SurvivalResult<()> {
    if status.len() != time.len() {
        return Err(SurvivalError::invalid_input(format!(
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

fn validate_finite_scalar(value: f64, name: &str) -> SurvivalResult<()> {
    if !value.is_finite() {
        return Err(SurvivalError::invalid_input(format!(
            "{name} must be finite"
        )));
    }
    Ok(())
}

/// The confidence level, 0.95 by default.
fn confidence_or_default(confidence_level: Option<f64>) -> SurvivalResult<f64> {
    let confidence = confidence_level.unwrap_or(0.95);
    if !confidence.is_finite() || confidence <= 0.0 || confidence >= 1.0 {
        return Err(SurvivalError::invalid_input(
            "confidence_level must be a finite value between 0 and 1",
        ));
    }
    Ok(confidence)
}

/// The observations still at risk after `landmark_time`, with the clock
/// reset to it.  A time within `aeqSurv`'s tolerance of the landmark
/// counts as the landmark itself and is excluded.
pub(crate) fn compute_landmark(time: &[f64], status: &[i32], landmark_time: f64) -> LandmarkResult {
    let (time, landmark) = bin_times(time, &[landmark_time]);
    let landmark = landmark[0];
    let mut new_time = Vec::new();
    let mut new_status = Vec::new();
    let mut original_indices = Vec::new();
    let mut n_excluded = 0usize;
    for (i, &t) in time.iter().enumerate() {
        if t > landmark {
            new_time.push(t - landmark);
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

/// Python entry point of `compute_landmark`.
#[pyfunction(name = "landmark_analysis")]
pub fn landmark_analysis_py(
    time: Vec<f64>,
    status: Vec<i32>,
    landmark_time: f64,
) -> PyResult<LandmarkResult> {
    validate_survival_inputs(&time, &status)?;
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

/// Python entry point of `compute_landmarks_parallel`.
#[pyfunction(name = "landmark_analysis_batch")]
pub fn landmark_analysis_batch_py(
    time: Vec<f64>,
    status: Vec<i32>,
    landmark_times: Vec<f64>,
) -> PyResult<Vec<LandmarkResult>> {
    validate_survival_inputs(&time, &status)?;
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

/// `S(target) / S(given)` from the Kaplan-Meier curve, with a normal
/// interval on the ratio (its variance from the difference of the
/// Greenwood sums) clamped to `[0, 1]`.
pub(crate) fn compute_conditional_survival(
    time: &[f64],
    status: &[i32],
    given_time: f64,
    target_time: f64,
    confidence_level: f64,
) -> SurvivalResult<ConditionalSurvivalResult> {
    let n = time.len();
    if n == 0 || target_time <= given_time {
        return Ok(ConditionalSurvivalResult {
            given_time,
            target_time,
            conditional_survival: 1.0,
            ci_lower: 1.0,
            ci_upper: 1.0,
            n_at_risk: 0,
        });
    }
    let (time, query) = bin_times(time, &[given_time, target_time]);
    let (given, target) = (query[0], query[1]);
    let steps = kaplan_meier_steps(&time, status)?;
    let (surv_given, var_given) =
        step_at(&steps, given).map_or((1.0, 0.0), |step| (step.survival, step.greenwood));
    let (surv_target, var_target) =
        step_at(&steps, target).map_or((1.0, 0.0), |step| (step.survival, step.greenwood));
    // number still at risk just after the given time
    let n_at_given = time.iter().filter(|&&t| t > given).count();
    let conditional = if surv_given > 0.0 {
        surv_target / surv_given
    } else {
        0.0
    };
    let z = z_score_for_confidence(confidence_level);
    let var_conditional = if surv_given > 0.0 && conditional > 0.0 {
        conditional * conditional * (var_target - var_given).abs()
    } else {
        0.0
    };
    let se = var_conditional.sqrt();
    let (ci_lower, ci_upper) = clamped_normal_ci(conditional, se, z, 0.0, 1.0);
    Ok(ConditionalSurvivalResult {
        given_time,
        target_time,
        conditional_survival: conditional,
        ci_lower,
        ci_upper,
        n_at_risk: n_at_given,
    })
}

/// Python entry point of `compute_conditional_survival`.
#[pyfunction(name = "conditional_survival")]
#[pyo3(signature = (time, status, given_time, target_time, confidence_level=None))]
pub fn conditional_survival_py(
    time: Vec<f64>,
    status: Vec<i32>,
    given_time: f64,
    target_time: f64,
    confidence_level: Option<f64>,
) -> PyResult<ConditionalSurvivalResult> {
    validate_survival_inputs(&time, &status)?;
    validate_finite_scalar(given_time, "given_time")?;
    validate_finite_scalar(target_time, "target_time")?;
    let conf = confidence_or_default(confidence_level)?;
    Ok(compute_conditional_survival(
        &time,
        &status,
        given_time,
        target_time,
        conf,
    )?)
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
/// against the first: `exp((O - E) / V)` with the log-rank variance `V`
/// (`survdiff`, which bins the times itself).  Fewer than two groups give
/// the neutral ratio 1.
pub(crate) fn compute_hazard_ratio(
    time: &[f64],
    status: &[i32],
    group: &[i32],
    confidence_level: f64,
) -> SurvivalResult<HazardRatioResult> {
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
        return Ok(neutral);
    }
    // Only the first two groups take part in the comparison.
    let rows: Vec<usize> = (0..time.len())
        .filter(|&i| group[i] == unique_groups[0] || group[i] == unique_groups[1])
        .collect();
    let time: Vec<f64> = rows.iter().map(|&i| time[i]).collect();
    let status: Vec<i32> = rows.iter().map(|&i| status[i]).collect();
    let group: Vec<i32> = rows.iter().map(|&i| group[i]).collect();
    let test = logrank_test(&time, &status, &group, None, None, 0.0, true)?;
    let sum_o_e = test.observed[0] - test.expected[0];
    let sum_var = test.variance[0][0];
    if sum_var <= 0.0 {
        return Ok(neutral); // no events: nothing to compare
    }
    let log_hr = sum_o_e / sum_var;
    let hazard_ratio = log_hr.exp();
    let se_log_hr = 1.0 / sum_var.sqrt();
    let z = z_score_for_confidence(confidence_level);
    let (ci_lower, ci_upper) = exp_ci(log_hr, se_log_hr, z);
    let z_statistic = log_hr / se_log_hr;
    let p_value = 2.0 * pnorm(z_statistic.abs(), false, false);
    Ok(HazardRatioResult {
        hazard_ratio,
        ci_lower,
        ci_upper,
        se_log_hr,
        z_statistic,
        p_value,
    })
}

/// Python entry point of `compute_hazard_ratio`.
#[pyfunction(name = "hazard_ratio")]
#[pyo3(signature = (time, status, group, confidence_level=None))]
pub fn hazard_ratio_py(
    time: Vec<f64>,
    status: Vec<i32>,
    group: Vec<i32>,
    confidence_level: Option<f64>,
) -> PyResult<HazardRatioResult> {
    validate_survival_inputs(&time, &status)?;
    if group.len() != time.len() {
        return Err(SurvivalError::invalid_input(format!(
            "group must have same length as time, got {} and {}",
            group.len(),
            time.len()
        ))
        .into());
    }
    let conf = confidence_or_default(confidence_level)?;
    Ok(compute_hazard_ratio(&time, &status, &group, conf)?)
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

/// The Kaplan-Meier curve at `eval_times`: the survival with a plain
/// Greenwood interval clamped to `[0, 1]`, the number at risk at the last
/// event time reached and the events so far.
pub(crate) fn compute_survival_at_times(
    time: &[f64],
    status: &[i32],
    eval_times: &[f64],
    confidence_level: f64,
) -> SurvivalResult<Vec<SurvivalAtTimeResult>> {
    let n = time.len();
    let (time, query) = bin_times(time, eval_times);
    let steps = if n == 0 {
        Vec::new()
    } else {
        kaplan_meier_steps(&time, status)?
    };
    let z = z_score_for_confidence(confidence_level);
    let evaluate = |(&t, &at): (&f64, &f64)| {
        let (survival, var, n_risk, n_events) = match step_at(&steps, at) {
            Some(step) => (
                step.survival,
                survival_variance(step),
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
    let results = if eval_times.len() > PARALLEL_THRESHOLD_SMALL {
        eval_times.par_iter().zip(&query).map(evaluate).collect()
    } else {
        eval_times.iter().zip(&query).map(evaluate).collect()
    };
    Ok(results)
}

/// Python entry point of `compute_survival_at_times`.
#[pyfunction(name = "survival_at_times")]
#[pyo3(signature = (time, status, eval_times, confidence_level=None))]
pub fn survival_at_times_py(
    time: Vec<f64>,
    status: Vec<i32>,
    eval_times: Vec<f64>,
    confidence_level: Option<f64>,
) -> PyResult<Vec<SurvivalAtTimeResult>> {
    validate_survival_inputs(&time, &status)?;
    validate_no_nan(&eval_times, "eval_times")?;
    validate_finite(&eval_times, "eval_times")?;
    let conf = confidence_or_default(confidence_level)?;
    Ok(compute_survival_at_times(
        &time,
        &status,
        &eval_times,
        conf,
    )?)
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

/// Actuarial life table on the intervals `[breaks[j], breaks[j + 1])`
/// (the last one closed): censored observations count for half an
/// interval at risk.  Times are binned with the breaks.
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
    let (time, binned_breaks) = bin_times(time, breaks);
    let mut n_deaths = vec![0.0; n_intervals];
    let mut n_censored = vec![0.0; n_intervals];
    for (i, &t) in time.iter().enumerate() {
        // the last interval also holds its upper break
        let j = binned_breaks[1..n_intervals].partition_point(|&b| b <= t);
        if t < binned_breaks[0] || t > binned_breaks[n_intervals] {
            continue;
        }
        if status[i] == 1 {
            n_deaths[j] += 1.0;
        } else {
            n_censored[j] += 1.0;
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
        interval_start: breaks[..n_intervals].to_vec(),
        interval_end: breaks[1..].to_vec(),
        n_at_risk,
        n_deaths,
        n_censored,
        n_effective,
        hazard,
        survival,
        se_survival,
    }
}

fn validate_life_table_inputs(time: &[f64], status: &[i32], breaks: &[f64]) -> SurvivalResult<()> {
    validate_survival_inputs(time, status)?;
    if breaks.len() < 2 {
        return Err(SurvivalError::invalid_input(
            "breaks must define at least one interval",
        ));
    }
    validate_no_nan(breaks, "breaks")?;
    validate_finite(breaks, "breaks")?;
    let (time, binned_breaks) = bin_times(time, breaks);
    for (index, window) in binned_breaks.windows(2).enumerate() {
        if window[1] <= window[0] {
            return Err(SurvivalError::invalid_input(format!(
                "breaks must be strictly increasing; got {} then {} at positions {} and {}",
                breaks[index],
                breaks[index + 1],
                index,
                index + 1
            )));
        }
    }
    let first = binned_breaks[0];
    let last = binned_breaks[binned_breaks.len() - 1];
    if let Some(index) = time.iter().position(|&value| value < first || value > last) {
        return Err(SurvivalError::invalid_input(format!(
            "time values must fall within the break range; got {} at index {} outside [{}, {}]",
            time[index],
            index,
            breaks[0],
            breaks[breaks.len() - 1]
        )));
    }
    Ok(())
}

/// Python entry point of `compute_life_table`.
#[pyfunction(name = "life_table")]
pub fn life_table_py(
    time: Vec<f64>,
    status: Vec<i32>,
    breaks: Vec<f64>,
) -> PyResult<LifeTableResult> {
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
        let err = landmark_analysis_py(vec![1.0], vec![], 0.5).unwrap_err();
        assert!(
            err.to_string()
                .contains("time and status must have same length")
        );

        let err = landmark_analysis_py(vec![f64::NAN], vec![1], 0.5).unwrap_err();
        assert!(err.to_string().contains("time contains NaN"));

        let err = landmark_analysis_py(vec![1.0], vec![2], 0.5).unwrap_err();
        assert!(
            err.to_string()
                .contains("status must contain only 0/1 values")
        );

        let err = landmark_analysis_batch_py(vec![1.0], vec![1], vec![f64::INFINITY]).unwrap_err();
        assert!(
            err.to_string()
                .contains("landmark_times contains non-finite")
        );

        let err = conditional_survival_py(vec![1.0], vec![1], f64::NAN, 2.0, None).unwrap_err();
        assert!(err.to_string().contains("given_time must be finite"));

        let err = conditional_survival_py(vec![1.0], vec![1], 0.5, 2.0, Some(1.0)).unwrap_err();
        assert!(err.to_string().contains("confidence_level"));

        let err = hazard_ratio_py(vec![1.0], vec![1], vec![], None).unwrap_err();
        assert!(err.to_string().contains("group must have same length"));

        let err = survival_at_times_py(vec![1.0], vec![1], vec![f64::NAN], None).unwrap_err();
        assert!(err.to_string().contains("eval_times contains NaN"));
    }

    #[test]
    fn test_conditional_survival_groups_near_tied_event_times() {
        let exact_time = vec![1.0, 1.0, 2.0, 3.0];
        let near_time = vec![1.0, 1.0 + 5e-10, 2.0, 3.0];
        let status = vec![1, 1, 0, 0];

        let expected = compute_conditional_survival(&exact_time, &status, 1.0, 2.0, 0.95).unwrap();
        let actual = compute_conditional_survival(&near_time, &status, 1.0, 2.0, 0.95).unwrap();

        assert!((actual.conditional_survival - expected.conditional_survival).abs() < 1e-12);
        assert!((actual.ci_lower - expected.ci_lower).abs() < 1e-12);
        assert!((actual.ci_upper - expected.ci_upper).abs() < 1e-12);
        assert_eq!(actual.n_at_risk, expected.n_at_risk);
        // a query time within tolerance of an event time is that event time
        let shifted =
            compute_conditional_survival(&exact_time, &status, 1.0 + 5e-10, 2.0, 0.95).unwrap();
        assert!((shifted.conditional_survival - expected.conditional_survival).abs() < 1e-12);
        assert_eq!(shifted.n_at_risk, 2);
    }

    #[test]
    fn test_conditional_survival_matches_the_kaplan_meier_ratio() {
        // survfit(Surv(c(1,2,3,4,5), c(1,1,0,1,1)) ~ 1): S(1) = .8, S(4) = .3
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 0, 1, 1];
        let result = compute_conditional_survival(&time, &status, 1.0, 4.0, 0.95).unwrap();
        assert!((result.conditional_survival - 0.3 / 0.8).abs() < 1e-12);
        assert_eq!(result.n_at_risk, 4);
        let trivial = compute_conditional_survival(&time, &status, 4.0, 2.0, 0.95).unwrap();
        assert_eq!(trivial.conditional_survival, 1.0);
    }

    #[test]
    fn test_hazard_ratio_groups_near_tied_event_times() {
        let exact_time = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        let near_time = vec![1.0, 1.0 + 5e-10, 2.0, 2.0 + 5e-10, 3.0, 3.0];
        let status = vec![1, 1, 1, 0, 0, 0];
        let group = vec![0, 1, 0, 1, 0, 1];

        let expected = compute_hazard_ratio(&exact_time, &status, &group, 0.95).unwrap();
        let actual = compute_hazard_ratio(&near_time, &status, &group, 0.95).unwrap();

        assert!((actual.hazard_ratio - expected.hazard_ratio).abs() < 1e-12);
        assert!((actual.se_log_hr - expected.se_log_hr).abs() < 1e-12);
        assert!((actual.z_statistic - expected.z_statistic).abs() < 1e-12);
        assert!((actual.p_value - expected.p_value).abs() < 1e-12);
        // one group, or no events, gives the neutral ratio
        let single = compute_hazard_ratio(&exact_time, &status, &[1; 6], 0.95).unwrap();
        assert_eq!(single.hazard_ratio, 1.0);
        let censored = compute_hazard_ratio(&exact_time, &[0; 6], &group, 0.95).unwrap();
        assert_eq!(censored.hazard_ratio, 1.0);
    }

    #[test]
    fn test_survival_at_times_groups_near_tied_event_times() {
        let exact_time = vec![1.0, 1.0, 2.0, 3.0];
        let near_time = vec![1.0, 1.0 + 5e-10, 2.0, 3.0];
        let status = vec![1, 1, 0, 0];

        let expected = compute_survival_at_times(&exact_time, &status, &[1.0, 2.0], 0.95).unwrap();
        let actual = compute_survival_at_times(&near_time, &status, &[1.0, 2.0], 0.95).unwrap();

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual.survival - expected.survival).abs() < 1e-12);
            assert!((actual.ci_lower - expected.ci_lower).abs() < 1e-12);
            assert!((actual.ci_upper - expected.ci_upper).abs() < 1e-12);
            assert_eq!(actual.n_at_risk, expected.n_at_risk);
            assert_eq!(actual.n_events, expected.n_events);
        }
        // the reported time is the one asked for
        let shifted =
            compute_survival_at_times(&exact_time, &status, &[1.0 + 5e-10], 0.95).unwrap();
        assert_eq!(shifted[0].time, 1.0 + 5e-10);
        assert!((shifted[0].survival - 0.5).abs() < 1e-12);
        assert_eq!(shifted[0].n_events, 2);
        let before = compute_survival_at_times(&exact_time, &status, &[0.5], 0.95).unwrap();
        assert_eq!(before[0].survival, 1.0);
        assert_eq!(before[0].n_at_risk, 4);
        assert!(compute_survival_at_times(&[], &[], &[0.5], 0.95).unwrap()[0].n_at_risk == 0);
    }

    #[test]
    fn test_survival_at_times_uses_the_greenwood_interval() {
        // S(2) = .5 with var(log S) = 1/(4*3) + 1/(3*2): se(S) = .5 * sqrt(.25)
        let time = vec![1.0, 2.0, 3.0, 4.0];
        let status = vec![1, 1, 0, 1];
        let at2 = compute_survival_at_times(&time, &status, &[2.0, 4.0], 0.95).unwrap();
        let se = 0.5 * (1.0_f64 / 12.0 + 1.0 / 6.0).sqrt();
        let z = z_score_for_confidence(0.95);
        assert!((at2[0].ci_lower - (0.5 - z * se)).abs() < 1e-12);
        assert!((at2[0].ci_upper - (0.5 + z * se)).abs() < 1e-12);
        // once the curve reaches 0 the interval collapses
        assert_eq!(at2[1].survival, 0.0);
        assert_eq!((at2[1].ci_lower, at2[1].ci_upper), (0.0, 0.0));
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
        assert_eq!(result.n_deaths, vec![1.0, 1.0, 1.0]);
        assert_eq!(result.n_censored, vec![0.0, 1.0, 1.0]);
        assert_eq!(result.n_at_risk, vec![5.0, 4.0, 2.0]);
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
        let result = life_table_py(vec![2.0], vec![1], vec![0.0, 1.0, 2.0]).unwrap();

        assert_eq!(result.n_deaths, vec![0.0, 1.0]);
        assert_eq!(result.n_censored, vec![0.0, 0.0]);
        // a time within tolerance of a break falls on the break's side
        let near = life_table_py(vec![1.0 + 5e-10], vec![1], vec![0.0, 1.0, 2.0]).unwrap();
        assert_eq!(near.n_deaths, vec![0.0, 1.0]);
        let near_end = life_table_py(vec![2.0 + 5e-10], vec![1], vec![0.0, 1.0, 2.0]).unwrap();
        assert_eq!(near_end.n_deaths, vec![0.0, 1.0]);
    }

    #[test]
    fn test_life_table_rejects_malformed_public_inputs() {
        let err = life_table_py(vec![1.0], vec![], vec![0.0, 2.0]).unwrap_err();
        assert!(
            err.to_string()
                .contains("time and status must have same length")
        );

        let err = life_table_py(vec![1.0], vec![1], vec![0.0]).unwrap_err();
        assert!(
            err.to_string()
                .contains("breaks must define at least one interval")
        );

        let err = life_table_py(vec![f64::NAN], vec![1], vec![0.0, 2.0]).unwrap_err();
        assert!(err.to_string().contains("time contains NaN"));

        let err = life_table_py(vec![1.0], vec![2], vec![0.0, 2.0]).unwrap_err();
        assert!(
            err.to_string()
                .contains("status must contain only 0/1 values")
        );

        let err = life_table_py(vec![1.0], vec![1], vec![0.0, 0.0]).unwrap_err();
        assert!(
            err.to_string()
                .contains("breaks must be strictly increasing")
        );
        let err = life_table_py(vec![1.0], vec![1], vec![0.0, 2.0, 2.0 + 5e-10]).unwrap_err();
        assert!(
            err.to_string()
                .contains("breaks must be strictly increasing")
        );

        let err = life_table_py(vec![3.0], vec![1], vec![0.0, 2.0]).unwrap_err();
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
