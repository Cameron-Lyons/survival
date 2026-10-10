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
}

/// The unique finite times with near ties removed (R's `cuts`), or `None`
/// when no two values are within the tolerance and nothing needs to change.
fn tie_cuts(columns: &[&[f64]], tolerance: f64) -> Option<Vec<f64>> {
    let mut y: Vec<f64> = columns
        .iter()
        .flat_map(|c| c.iter().copied())
        .filter(|v| v.is_finite())
        .collect();
    y.sort_unstable_by(f64::total_cmp);
    y.dedup();
    if y.len() < 2 {
        return None;
    }
    let total_abs = y.iter().map(|v| v.abs()).sum::<f64>();
    let mean_abs = if total_abs.is_finite() {
        total_abs / y.len() as f64
    } else {
        // The mean of finite magnitudes is finite even if their sum overflows.
        // An infinite denominator would turn every finite gap into a near tie.
        // Scale only on overflow to preserve the ordinary arithmetic exactly.
        let largest = y[0].abs().max(y[y.len() - 1].abs());
        let scaled_sum = y.iter().map(|v| v.abs() / largest).sum::<f64>();
        (scaled_sum / y.len() as f64) * largest
    };
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

/// The largest finite cut not above `x`. Nonfinite endpoints remain unchanged:
/// applying R's finite-cut indexing to them truncates positive infinity and
/// drops negative-infinity rows when the resulting zero index is subscripted.
fn snap(x: f64, cuts: &[f64]) -> f64 {
    if !x.is_finite() {
        return x;
    }
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
    Ok(AeqSurvResult {
        time: new_time,
        time2: new_time2,
    })
}

/// `aeqSurv` with R's default tolerance on the time columns of a
/// right-censored (`start` absent) or counting-process response, returned
/// as `(start, stop)`.
pub fn aeq_counting(
    start: Option<&[f64]>,
    stop: &[f64],
) -> SurvivalResult<(Option<Vec<f64>>, Vec<f64>)> {
    match start {
        Some(start) => {
            let fixed = aeq_surv(start, Some(stop), None)?;
            let stop = fixed.time2.expect("aeq_surv keeps the second column");
            Ok((Some(fixed.time), stop))
        }
        None => Ok((None, aeq_surv(stop, None, None)?.time)),
    }
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
        assert!(aeq_surv(&[], None, None).unwrap().time.is_empty());
        assert_eq!(aeq_times(&[1.0, 1.0, 1.0]), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn near_ties_collapse_onto_the_earliest_value() {
        let result = aeq_surv(&[1.0, 1.0 + 1e-10, 2.0, 3.0], None, Some(1e-8)).unwrap();
        assert_eq!(result.time, vec![1.0, 1.0, 2.0, 3.0]);

        // Adjacent cutpoints within tolerance chain onto the first.
        let result = aeq_surv(&[1.0, 1.0 + 9e-9, 1.0 + 18e-9], None, Some(1e-8)).unwrap();
        assert_eq!(result.time, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn relative_tolerance_uses_the_mean_absolute_time() {
        let result = aeq_surv(&[1e9, 1e9 + 1.0, 1e9 + 20.0], None, Some(1e-8)).unwrap();
        assert_eq!(result.time, vec![1e9, 1e9, 1e9 + 20.0]);
        // The fixture case `right_1e9`: 2.000000001 is tied with 2.
        let result = aeq_surv(&[1.0, 1.00000000000001, 2.0, 2.000000001, 3.0], None, None).unwrap();
        assert_eq!(result.time, vec![1.0, 1.0, 2.0, 2.0, 3.0]);
    }

    #[test]
    fn large_finite_times_do_not_create_spurious_near_ties() {
        // Independent stock-R aeqSurv results: mean(abs(time)) remains finite.
        for time in [
            vec![1e308, 1.1e308, 1.2e308],
            vec![-1e308, 0.0, 1e308],
            vec![f64::MAX / 2.0, f64::MAX * 0.7, f64::MAX],
        ] {
            assert_eq!(aeq_surv(&time, None, None).unwrap().time, time);
        }
        let time = [1e308, 1e308 * (1.0 + 1e-12), 1.2e308];
        assert_eq!(
            aeq_surv(&time, None, None).unwrap().time,
            vec![1e308, 1e308, 1.2e308]
        );
        assert_eq!(
            aeq_surv(&[1e308, 1.1e308, 1.2e308], None, Some(0.1))
                .unwrap()
                .time,
            vec![1e308; 3]
        );
    }

    #[test]
    fn large_counting_times_and_nonfinite_endpoints_stay_aligned() {
        let start = [0.0, 1e308, 1.1e308];
        let stop = [1e308, 1.1e308, 1.2e308];
        let result = aeq_surv(&start, Some(&stop), None).unwrap();
        assert_eq!(result.time, start);
        assert_eq!(result.time2.unwrap(), stop);

        let time = [
            f64::NEG_INFINITY,
            1e308,
            1e308 * (1.0 + 1e-12),
            1.2e308,
            f64::INFINITY,
            f64::NAN,
        ];
        let result = aeq_surv(&time, None, None).unwrap();
        assert_eq!(
            &result.time[..5],
            &[f64::NEG_INFINITY, 1e308, 1e308, 1.2e308, f64::INFINITY]
        );
        assert!(result.time[5].is_nan());
    }

    #[test]
    fn large_time_normalization_and_curves_match_independent_stock_r() {
        use crate::surv_analysis::{SurvfitKMData, SurvfitKMOptions, survfitkm};
        use serde_json::Value;

        fn numbers(value: &Value) -> Vec<f64> {
            value
                .as_array()
                .unwrap()
                .iter()
                .map(|number| {
                    if number.is_null() {
                        f64::NAN
                    } else if let Some(text) = number.as_str() {
                        text.parse().unwrap()
                    } else {
                        number.as_f64().unwrap()
                    }
                })
                .collect()
        }

        let reference: Value = serde_json::from_str(include_str!(
            "../../python/tests/fixtures/aeq_large_time_reference.json"
        ))
        .unwrap();
        for case in reference["cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let time = numbers(&case["time"]);
            let start = (!case["start"].is_null()).then(|| numbers(&case["start"]));
            let tolerance = case["tolerance"].as_f64();
            let normalized = match start.as_deref() {
                Some(start) => aeq_surv(start, Some(&time), tolerance),
                None => aeq_surv(&time, None, tolerance),
            };
            if let Some(error) = case["expected"]["error"].as_str() {
                assert_eq!(normalized.unwrap_err().to_string(), error, "{name}");
                continue;
            }
            let normalized = normalized.unwrap();
            let normalized_time = normalized.time2.as_ref().unwrap_or(&normalized.time);
            assert_eq!(
                normalized_time,
                &numbers(&case["expected"]["time"]),
                "{name} time"
            );
            if start.is_some() {
                assert_eq!(
                    normalized.time,
                    numbers(&case["expected"]["start"]),
                    "{name} start"
                );
            }
            if case["fit"].is_null() {
                continue;
            }
            let status = numbers(&case["status"])
                .into_iter()
                .map(|status| status as i32)
                .collect();
            let data = SurvfitKMData::try_new(start, time, status, None, None, None, None).unwrap();
            let fit = survfitkm(&data, &SurvfitKMOptions::default()).unwrap();
            for (field, actual) in [
                ("time", &fit.time),
                ("n_risk", &fit.n_risk),
                ("n_event", &fit.n_event),
                ("n_censor", &fit.n_censor),
                ("surv", &fit.surv),
                ("cumhaz", &fit.cumhaz),
                ("std_err", fit.std_err.as_ref().unwrap()),
                ("std_chaz", fit.std_chaz.as_ref().unwrap()),
                ("lower", fit.lower.as_ref().unwrap()),
                ("upper", fit.upper.as_ref().unwrap()),
            ] {
                let expected = numbers(&case["fit"][field]);
                assert_eq!(actual.len(), expected.len(), "{name} {field}");
                if field == "time" || field.starts_with("n_") {
                    assert_eq!(actual, &expected, "{name} {field}");
                    continue;
                }
                for (&actual, expected) in actual.iter().zip(expected) {
                    assert!(
                        actual == expected
                            || (actual.is_nan() && expected.is_nan())
                            || (actual.is_finite()
                                && expected.is_finite()
                                && (actual - expected).abs() <= 1e-14 + expected.abs() * 1e-12),
                        "{name} {field}: {actual:?} != {expected:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn counting_process_columns_are_snapped_together() {
        let start = [0.0, 1e-14, 1.0, 1.5];
        let stop = [1.0, 1.00000000000001, 2.0, 2.5];
        let result = aeq_surv(&start, Some(&stop), None).unwrap();
        assert_eq!(result.time, vec![0.0, 0.0, 1.0, 1.5]);
        assert_eq!(result.time2, Some(vec![1.0, 1.0, 2.0, 2.5]));
        let (fixed_start, fixed_stop) = aeq_counting(Some(&start), &stop).unwrap();
        assert_eq!(fixed_start, Some(result.time));
        assert_eq!(fixed_stop, vec![1.0, 1.0, 2.0, 2.5]);
        assert_eq!(
            aeq_counting(None, &stop).unwrap(),
            (None, vec![1.0, 1.0, 2.0, 2.5])
        );

        let zero_length = aeq_surv(&[0.0, 1.0], Some(&[1.0, 1.0 + 1e-12]), None);
        assert!(
            zero_length
                .unwrap_err()
                .to_string()
                .contains("effective length 0")
        );
    }

    #[test]
    fn near_ties_preserve_nonfinite_endpoints_and_row_alignment() {
        let time = [f64::NEG_INFINITY, 1.0, 1.0 + 1e-12, f64::INFINITY, f64::NAN];
        let result = aeq_surv(&time, None, None).unwrap();
        assert_eq!(
            &result.time[..4],
            &[f64::NEG_INFINITY, 1.0, 1.0, f64::INFINITY]
        );
        assert!(result.time[4].is_nan());
        let result = aeq_surv(
            &[f64::NEG_INFINITY, 0.0, 1.0],
            Some(&[1.0, 1.0 + 1e-12, f64::INFINITY]),
            None,
        )
        .unwrap();
        assert_eq!(result.time, vec![f64::NEG_INFINITY, 0.0, 1.0]);
        assert_eq!(result.time2, Some(vec![1.0, 1.0, f64::INFINITY]));
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
