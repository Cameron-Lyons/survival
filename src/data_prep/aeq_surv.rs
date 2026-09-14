//! R's `aeqSurv` (`R/aeqSurv.R`): treat time values that differ by less
//! than a tolerance as tied, using the same decision as `all.equal`.
//!
//! This is the one definition of "timefix" in the crate: routines with a
//! `timefix` argument call [`aeq_times`] (one time column) or [`aeq_surv`]
//! (a `Surv` object) rather than comparing times with an epsilon.

use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::validate_length;
use pyo3::prelude::*;

/// R's default `tolerance = sqrt(.Machine$double.eps)`.
pub const DEFAULT_TOLERANCE: f64 = 1.4901161193847656e-8;

/// The time columns of a `Surv` object after `aeqSurv`.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct AeqSurvResult {
    /// The (first) time column with near ties collapsed.
    #[pyo3(get)]
    pub time: Vec<f64>,
    /// The second time column of counting-process data.
    #[pyo3(get)]
    pub time2: Option<Vec<f64>>,
    /// Zero-based rows in which at least one time value changed.
    #[pyo3(get)]
    pub changed: Vec<usize>,
}

/// The unique finite times with near ties removed (R's `cuts`), or `None`
/// when no two values are within the tolerance and nothing needs to change.
fn tie_cuts(columns: &[&[f64]], tolerance: f64) -> Option<Vec<f64>> {
    let mut y: Vec<f64> = columns
        .iter()
        .flat_map(|c| c.iter().copied())
        .filter(|v| v.is_finite())
        .collect();
    y.sort_by(|a, b| a.total_cmp(b));
    y.dedup();
    if y.len() < 2 {
        return None;
    }
    let mean_abs = y.iter().map(|v| v.abs()).sum::<f64>() / y.len() as f64;
    let mut cuts = Vec::with_capacity(y.len());
    cuts.push(y[0]);
    let mut any_tied = false;
    for pair in y.windows(2) {
        let dy = pair[1] - pair[0];
        if dy <= tolerance || dy / mean_abs <= tolerance {
            any_tied = true;
        } else {
            cuts.push(pair[1]);
        }
    }
    any_tied.then_some(cuts)
}

/// R's `cuts[findInterval(x, cuts)]`: the largest cut not above `x`, or
/// `NaN` (R's `NA`) below the first cut.
fn snap(x: f64, cuts: &[f64]) -> f64 {
    let index = cuts.partition_point(|&cut| cut <= x);
    if index == 0 {
        f64::NAN
    } else {
        cuts[index - 1]
    }
}

/// `aeqSurv` for a right-censored (`time`) or counting-process (`time`,
/// `time2`) response.  With `tolerance <= 0` nothing is changed.
pub fn aeq_surv(
    time: &[f64],
    time2: Option<&[f64]>,
    tolerance: Option<f64>,
) -> SurvivalResult<AeqSurvResult> {
    if let Some(time2) = time2 {
        validate_length(time.len(), time2.len(), "time2")?;
    }
    let tolerance = match tolerance {
        Some(value) if !value.is_finite() => {
            return Err(SurvivalError::invalid_input("invalid value for tolerance"));
        }
        Some(value) => value,
        None => DEFAULT_TOLERANCE,
    };
    let unchanged = || AeqSurvResult {
        time: time.to_vec(),
        time2: time2.map(<[f64]>::to_vec),
        changed: Vec::new(),
    };
    if tolerance <= 0.0 {
        return Ok(unchanged());
    }
    let columns: Vec<&[f64]> = std::iter::once(time).chain(time2).collect();
    let Some(cuts) = tie_cuts(&columns, tolerance) else {
        return Ok(unchanged());
    };
    let new_time: Vec<f64> = time.iter().map(|&t| snap(t, &cuts)).collect();
    let new_time2: Option<Vec<f64>> = time2.map(|t2| t2.iter().map(|&t| snap(t, &cuts)).collect());
    if let (Some(t2), Some(new_t2)) = (time2, &new_time2) {
        // We may have created zero length intervals.
        for i in 0..time.len() {
            if new_time[i] == new_t2[i] && time[i] != t2[i] {
                return Err(SurvivalError::invalid_input(
                    "aeqSurv exception, an interval has effective length 0",
                ));
            }
        }
    }
    let same = |a: f64, b: f64| a == b || (a.is_nan() && b.is_nan());
    let changed = (0..time.len())
        .filter(|&i| {
            !same(new_time[i], time[i])
                || new_time2
                    .as_ref()
                    .zip(time2)
                    .is_some_and(|(n, o)| !same(n[i], o[i]))
        })
        .collect();
    Ok(AeqSurvResult {
        time: new_time,
        time2: new_time2,
        changed,
    })
}

/// `aeqSurv` on a single time column with R's default tolerance: the
/// crate-wide "timefix" step for routines that snap near-tied times.
pub fn aeq_times(time: &[f64]) -> Vec<f64> {
    aeq_surv(time, None, None)
        .map(|result| result.time)
        .unwrap_or_else(|_| time.to_vec())
}

/// Python entry point of [`aeq_surv`].
#[pyfunction(name = "aeq_surv")]
#[pyo3(signature = (time, time2=None, tolerance=None))]
pub fn aeq_surv_py(
    time: Vec<f64>,
    time2: Option<Vec<f64>>,
    tolerance: Option<f64>,
) -> PyResult<AeqSurvResult> {
    Ok(aeq_surv(&time, time2.as_deref(), tolerance)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distinct_times_are_left_alone() {
        let time = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = aeq_surv(&time, None, None).unwrap();
        assert_eq!(result.time, time);
        assert!(result.changed.is_empty());
        assert!(aeq_surv(&[], None, None).unwrap().time.is_empty());
        assert_eq!(aeq_times(&[1.0, 1.0, 1.0]), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn near_ties_collapse_onto_the_earliest_value() {
        let result = aeq_surv(&[1.0, 1.0 + 1e-10, 2.0, 3.0], None, Some(1e-8)).unwrap();
        assert_eq!(result.time, vec![1.0, 1.0, 2.0, 3.0]);
        assert_eq!(result.changed, vec![1]);

        // Adjacent cutpoints within tolerance chain onto the first.
        let result = aeq_surv(&[1.0, 1.0 + 9e-9, 1.0 + 18e-9], None, Some(1e-8)).unwrap();
        assert_eq!(result.time, vec![1.0, 1.0, 1.0]);
        assert_eq!(result.changed, vec![1, 2]);
    }

    #[test]
    fn relative_tolerance_uses_the_mean_absolute_time() {
        let result = aeq_surv(&[1e9, 1e9 + 1.0, 1e9 + 20.0], None, Some(1e-8)).unwrap();
        assert_eq!(result.time, vec![1e9, 1e9, 1e9 + 20.0]);
        assert_eq!(result.changed, vec![1]);
        // The fixture case `right_1e9`: 2.000000001 is tied with 2.
        let result = aeq_surv(&[1.0, 1.00000000000001, 2.0, 2.000000001, 3.0], None, None).unwrap();
        assert_eq!(result.time, vec![1.0, 1.0, 2.0, 2.0, 3.0]);
    }

    #[test]
    fn counting_process_columns_are_snapped_together() {
        let start = [0.0, 1e-14, 1.0, 1.5];
        let stop = [1.0, 1.00000000000001, 2.0, 2.5];
        let result = aeq_surv(&start, Some(&stop), None).unwrap();
        assert_eq!(result.time, vec![0.0, 0.0, 1.0, 1.5]);
        assert_eq!(result.time2, Some(vec![1.0, 1.0, 2.0, 2.5]));
        assert_eq!(result.changed, vec![1]);

        let zero_length = aeq_surv(&[0.0, 1.0], Some(&[1.0, 1.0 + 1e-12]), None);
        assert!(
            zero_length
                .unwrap_err()
                .to_string()
                .contains("effective length 0")
        );
    }

    #[test]
    fn infinite_times_map_onto_the_last_finite_cut_as_in_r() {
        let result = aeq_surv(&[1.0, 1.0 + 1e-12, f64::INFINITY], None, None).unwrap();
        assert_eq!(result.time, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn nonpositive_or_invalid_tolerances() {
        let time = [1.0, 1.0 + 1e-10];
        assert_eq!(aeq_surv(&time, None, Some(0.0)).unwrap().time, time);
        assert_eq!(aeq_surv(&time, None, Some(-1.0)).unwrap().time, time);
        assert!(aeq_surv(&time, None, Some(f64::INFINITY)).is_err());
        assert!(aeq_surv(&time, Some(&[1.0]), None).is_err());
    }
}
