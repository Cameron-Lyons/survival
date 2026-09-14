//! Number needed to treat at a time horizon from two Kaplan-Meier curves
//! (no R counterpart): the absolute risk reduction `S_1(t) - S_0(t)`
//! with Greenwood variances, and its reciprocal.

use super::kaplan_meier;
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::dist::qnorm;
use crate::internal::validation::validate_length;
use pyo3::prelude::*;

/// Absolute risk reduction and number needed to treat at `time_horizon`.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object, get_all)]
pub struct NNTResult {
    pub nnt: f64,
    pub nnt_ci_lower: f64,
    pub nnt_ci_upper: f64,
    pub absolute_risk_reduction: f64,
    pub arr_ci_lower: f64,
    pub arr_ci_upper: f64,
    pub time_horizon: f64,
}

/// Kaplan-Meier survival at `at` and its Greenwood variance.
fn survival_at(time: &[f64], status: &[i32], at: f64) -> SurvivalResult<(f64, f64)> {
    let km = kaplan_meier(time, status, None, 0.95)?;
    let mut survival = 1.0;
    let mut greenwood = 0.0;
    for i in 0..km.time.len() {
        if km.time[i] > at {
            break;
        }
        let (n, d) = (km.n_risk[i], km.n_event[i]);
        if d > 0.0 && n > 0.0 {
            survival *= 1.0 - d / n;
            if n > d {
                greenwood += d / (n * (n - d));
            }
        }
    }
    Ok((survival, survival * survival * greenwood))
}

/// Number needed to treat comparing the second group (treated) with the
/// first (control) of `group`'s sorted labels.
pub fn number_needed_to_treat(
    time: &[f64],
    status: &[i32],
    group: &[i32],
    time_horizon: f64,
    conf_level: f64,
) -> SurvivalResult<NNTResult> {
    validate_length(time.len(), group.len(), "group")?;
    if !time_horizon.is_finite() || time_horizon < 0.0 {
        return Err(SurvivalError::invalid_input(
            "time_horizon must be non-negative",
        ));
    }
    if !(conf_level > 0.0 && conf_level < 1.0) {
        return Err(SurvivalError::invalid_input(
            "conf_level must be between 0 and 1",
        ));
    }
    let mut labels = group.to_vec();
    labels.sort_unstable();
    labels.dedup();
    if labels.len() != 2 {
        return Err(SurvivalError::invalid_input(
            "number_needed_to_treat needs exactly two groups",
        ));
    }
    let mut arms = Vec::with_capacity(2);
    for &label in &labels {
        let rows: Vec<usize> = (0..time.len()).filter(|&i| group[i] == label).collect();
        let time: Vec<f64> = rows.iter().map(|&i| time[i]).collect();
        let status: Vec<i32> = rows.iter().map(|&i| status[i]).collect();
        arms.push(survival_at(&time, &status, time_horizon)?);
    }
    let (control, treated) = (arms[0], arms[1]);
    let arr = (1.0 - control.0) - (1.0 - treated.0);
    let arr_se = (control.1 + treated.1).sqrt();
    let z = qnorm((1.0 + conf_level) / 2.0, true, false);
    let (arr_ci_lower, arr_ci_upper) = (arr - z * arr_se, arr + z * arr_se);
    let nnt = if arr.abs() > 1e-10 {
        1.0 / arr
    } else {
        f64::INFINITY
    };
    let (nnt_ci_lower, nnt_ci_upper) = if arr_ci_lower > 0.0 && arr_ci_upper > 0.0 {
        (1.0 / arr_ci_upper, 1.0 / arr_ci_lower)
    } else if arr_ci_lower < 0.0 && arr_ci_upper < 0.0 {
        (1.0 / arr_ci_lower, 1.0 / arr_ci_upper)
    } else {
        (f64::NEG_INFINITY, f64::INFINITY)
    };
    Ok(NNTResult {
        nnt,
        nnt_ci_lower,
        nnt_ci_upper,
        absolute_risk_reduction: arr,
        arr_ci_lower,
        arr_ci_upper,
        time_horizon,
    })
}

#[pyfunction(name = "number_needed_to_treat")]
#[pyo3(signature = (time, status, group, time_horizon, conf_level=0.95))]
pub fn number_needed_to_treat_py(
    time: Vec<f64>,
    status: Vec<i32>,
    group: Vec<i32>,
    time_horizon: f64,
    conf_level: f64,
) -> PyResult<NNTResult> {
    Ok(number_needed_to_treat(
        &time,
        &status,
        &group,
        time_horizon,
        conf_level,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nnt_is_the_reciprocal_of_the_risk_difference() {
        let time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let status = [1, 1, 1, 0, 1, 0, 0, 0];
        let group = [0, 0, 0, 0, 1, 1, 1, 1];
        let result = number_needed_to_treat(&time, &status, &group, 4.5, 0.95).unwrap();
        // control: S(4.5) = 0.25; treated: S(4.5) = 1 -> ARR = 0.75
        assert!((result.absolute_risk_reduction - 0.75).abs() < 1e-12);
        assert!((result.nnt - 4.0 / 3.0).abs() < 1e-12);
        assert!(result.arr_ci_lower < result.absolute_risk_reduction);
        assert!(number_needed_to_treat(&time, &status, &[0; 8], 4.5, 0.95).is_err());
    }
}
