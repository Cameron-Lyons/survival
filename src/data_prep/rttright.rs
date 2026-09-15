//! R's `rttright` (`R/rttright.R`): redistribute-to-the-right weights.
//! Each censored observation hands its case weight on to the observations
//! still at risk, so that the weighted sum of events reproduces the
//! Kaplan-Meier (or Aalen-Johansen) estimate; the weights are the case
//! weights divided by the left-continuous censoring distribution `G`.
//!
//! Right-censored, multi-state (`mright`) and counting-process data with
//! an `id` are supported as far as R's compact algorithm reaches: every
//! subject enters at the same time in the same state.  One deviation from
//! R 3.8: for counting-process data at reporting times R tests `Y[, 2] > 0`
//! (the stop time) where it means the status, so every subject's last row
//! keeps a weight there even when it is censored; this port uses the
//! status, as R's single-time-point branch does.

use super::aeq_surv::aeq_surv;
use super::id_value::{IdValue, SubjectId, first_appearance_codes};
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::{validate_finite, validate_length, validate_non_negative};
use pyo3::prelude::*;

/// The response of an `rttright` call.
pub struct RttrightInput<'a, I> {
    /// Entry times of counting-process data (`Surv(tstart, tstop, status)`).
    pub start: Option<&'a [f64]>,
    pub time: &'a [f64],
    /// `0` for censored; any positive code is an event (multi-state).
    pub status: &'a [i32],
    /// Zero-based stratum of each observation (the formula's right-hand
    /// side); `None` for a single stratum.
    pub strata: Option<&'a [usize]>,
    pub weights: Option<&'a [f64]>,
    /// Subject identifier, required for counting-process data.
    pub id: Option<&'a [I]>,
    /// Reporting times; `None` gives the final weights.
    pub times: Option<&'a [f64]>,
    pub timefix: bool,
    pub renorm: bool,
}

/// The weights: one row per observation and one column per reporting
/// time (a single column when no times were requested).
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct RttrightResult {
    #[pyo3(get)]
    pub weights: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub times: Vec<f64>,
}

/// The censoring distribution `G` of one stratum as a step function:
/// `G(t-)` is the product over censoring times strictly before `t`.
struct CensoringCurve {
    times: Vec<f64>,
    surv: Vec<f64>,
}

impl CensoringCurve {
    /// `G` just before `t` (left continuous).
    fn before(&self, t: f64) -> f64 {
        let k = self.times.partition_point(|&c| c < t);
        if k == 0 { 1.0 } else { self.surv[k - 1] }
    }
}

/// Kaplan-Meier estimate of the censoring distribution from rows whose
/// censoring is shifted just after the events at the same time, as R does
/// by adding a small `delta` to the censored times.
fn censoring_curve(
    rows: &[usize],
    start: Option<&[f64]>,
    stop: &[f64],
    censor: &[bool],
    weight: &[f64],
) -> CensoringCurve {
    let mut censor_times: Vec<f64> = rows
        .iter()
        .filter(|&&i| censor[i])
        .map(|&i| stop[i])
        .collect();
    censor_times.sort_by(|a, b| a.total_cmp(b));
    censor_times.dedup();
    let mut surv = Vec::with_capacity(censor_times.len());
    let mut g = 1.0;
    for &c in &censor_times {
        let mut at_risk = 0.0;
        let mut censored = 0.0;
        for &i in rows {
            // At risk just after the events at c: rows still open then.
            let entered = start.is_none_or(|s| s[i] <= c);
            let open = stop[i] > c || (stop[i] == c && censor[i]);
            if entered && open {
                at_risk += weight[i];
                if stop[i] == c && censor[i] {
                    censored += weight[i];
                }
            }
        }
        if at_risk > 0.0 {
            g *= 1.0 - censored / at_risk;
        }
        surv.push(g);
    }
    CensoringCurve {
        times: censor_times,
        surv,
    }
}

/// Redistribute-to-the-right weights.
pub fn rttright<I: SubjectId>(input: RttrightInput<'_, I>) -> SurvivalResult<RttrightResult> {
    let n = input.time.len();
    validate_length(n, input.status.len(), "status")?;
    validate_finite(input.time, "time")?;
    if input.status.iter().any(|&s| s < 0) {
        return Err(SurvivalError::invalid_input(
            "status must be 0 (censored) or a positive event code",
        ));
    }
    let mut casewt = match input.weights {
        Some(w) => {
            validate_length(n, w.len(), "weights")?;
            validate_finite(w, "weights")?;
            validate_non_negative(w, "weights")?;
            w.to_vec()
        }
        None => vec![1.0; n],
    };
    if let Some(strata) = input.strata {
        validate_length(n, strata.len(), "strata")?;
    }
    if let Some(times) = input.times {
        validate_finite(times, "times")?;
    }
    if input.start.is_some() && input.id.is_none() {
        return Err(SurvivalError::invalid_input(
            "id is required for start-stop data",
        ));
    }
    let start = match input.start {
        Some(s) => {
            validate_length(n, s.len(), "start")?;
            validate_finite(s, "start")?;
            Some(s.to_vec())
        }
        None => None,
    };

    // Near-tie fix, on both time columns together.
    let (start, stop) = if input.timefix {
        match &start {
            Some(s) => {
                let fixed = aeq_surv(s, Some(input.time), None)?;
                (Some(fixed.time), fixed.time2.unwrap_or_default())
            }
            None => (None, aeq_surv(input.time, None, None)?.time),
        }
    } else {
        (start, input.time.to_vec())
    };

    // Subject bookkeeping (counting-process data): the last row of each
    // subject, one entry time for everybody and one weight per subject.
    // With (time, status) data every row is its own subject, as in R.
    let id_codes = match input.id {
        Some(id) => {
            validate_length(n, id.len(), "id")?;
            Some(first_appearance_codes(id).0)
        }
        None => None,
    };
    let mut last = vec![true; n];
    if let Some(codes) = &id_codes {
        let n_subjects = codes.iter().max().map_or(0, |c| c + 1);
        let mut weight_range = vec![(f64::INFINITY, f64::NEG_INFINITY); n_subjects];
        for i in 0..n {
            let (lo, hi) = weight_range[codes[i]];
            weight_range[codes[i]] = (lo.min(casewt[i]), hi.max(casewt[i]));
        }
        if weight_range.iter().any(|(lo, hi)| hi - lo > 0.0) {
            return Err(SurvivalError::invalid_input(
                "there are subjects with multiple weights",
            ));
        }
    }
    if let (Some(codes), Some(s)) = (&id_codes, &start) {
        let n_subjects = codes.iter().max().map_or(0, |c| c + 1);
        let mut last_row = vec![usize::MAX; n_subjects];
        let mut last_time = vec![f64::NEG_INFINITY; n_subjects];
        let mut first_time = vec![f64::INFINITY; n_subjects];
        for i in 0..n {
            let code = codes[i];
            if stop[i] > last_time[code] {
                last_time[code] = stop[i];
                last_row[code] = i;
            }
            first_time[code] = first_time[code].min(s[i]);
        }
        if first_time.windows(2).any(|w| w[0] != w[1]) {
            return Err(SurvivalError::invalid_input(
                "function not defined for delayed entry or multistate data",
            ));
        }
        last = vec![false; n];
        for &row in &last_row {
            last[row] = true;
        }
    }
    let censor: Vec<bool> = (0..n).map(|i| last[i] && input.status[i] == 0).collect();
    let has_weight: Vec<bool> = (0..n).map(|i| last[i] && input.status[i] > 0).collect();

    let strata: Vec<usize> = input.strata.map_or_else(|| vec![0; n], <[usize]>::to_vec);
    let n_strata = strata.iter().max().map_or(0, |s| s + 1);
    if input.renorm {
        for s in 0..n_strata {
            let mut total = 0.0;
            let mut seen = vec![false; id_codes.as_ref().map_or(0, |c| c.len())];
            for i in 0..n {
                if strata[i] != s {
                    continue;
                }
                match &id_codes {
                    Some(codes) => {
                        if !seen[codes[i]] {
                            seen[codes[i]] = true;
                            total += casewt[i];
                        }
                    }
                    None => total += casewt[i],
                }
            }
            if total <= 0.0 {
                return Err(SurvivalError::invalid_input(
                    "weights must have a positive sum in every stratum when renorm is true",
                ));
            }
            for i in 0..n {
                if strata[i] == s {
                    casewt[i] /= total;
                }
            }
        }
    }

    let query_times: Vec<f64> = input.times.map_or_else(Vec::new, <[f64]>::to_vec);
    let n_columns = query_times.len().max(1);
    let mut weights = vec![vec![0.0; n_columns]; n];
    for s in 0..n_strata {
        let rows: Vec<usize> = (0..n).filter(|&i| strata[i] == s).collect();
        let curve = censoring_curve(&rows, start.as_deref(), &stop, &censor, &casewt);
        if input.times.is_none() {
            // A single column: the final weight of every event.
            for &i in &rows {
                if has_weight[i] {
                    weights[i][0] = casewt[i] / curve.before(stop[i]);
                }
            }
            continue;
        }
        let gwt: Vec<f64> = query_times.iter().map(|&t| curve.before(t)).collect();
        for &i in &rows {
            let g_row = curve.before(stop[i]);
            for (col, (&t, &g_col)) in query_times.iter().zip(&gwt).enumerate() {
                weights[i][col] = if has_weight[i] {
                    // The last, uncensored row of a subject keeps its weight.
                    casewt[i] / g_row.max(g_col)
                } else {
                    let inside = match &start {
                        Some(s) => s[i] < t && stop[i] >= t,
                        None => stop[i] >= t,
                    };
                    if inside { casewt[i] / g_col } else { 0.0 }
                };
            }
        }
    }
    Ok(RttrightResult {
        weights,
        times: query_times,
    })
}

/// Python entry point of [`rttright`].
#[pyfunction(name = "rttright")]
#[pyo3(signature = (time, status, start=None, strata=None, weights=None, id=None, times=None, timefix=true, renorm=true))]
#[allow(clippy::too_many_arguments)]
pub fn rttright_py(
    time: Vec<f64>,
    status: Vec<i32>,
    start: Option<Vec<f64>>,
    strata: Option<Vec<usize>>,
    weights: Option<Vec<f64>>,
    id: Option<Vec<IdValue>>,
    times: Option<Vec<f64>>,
    timefix: bool,
    renorm: bool,
) -> PyResult<RttrightResult> {
    if id.iter().flatten().any(IdValue::is_missing) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "id must not contain missing values",
        ));
    }
    Ok(rttright(RttrightInput {
        start: start.as_deref(),
        time: &time,
        status: &status,
        strata: strata.as_deref(),
        weights: weights.as_deref(),
        id: id.as_deref(),
        times: times.as_deref(),
        timefix,
        renorm,
    })?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple(time: &[f64], status: &[i32], times: Option<&[f64]>) -> RttrightResult {
        rttright::<i64>(RttrightInput {
            start: None,
            time,
            status,
            strata: None,
            weights: None,
            id: None,
            times,
            timefix: true,
            renorm: true,
        })
        .unwrap()
    }

    fn column(result: &RttrightResult, col: usize) -> Vec<f64> {
        result.weights.iter().map(|row| row[col]).collect()
    }

    #[test]
    fn final_weights_reproduce_the_kaplan_meier_estimate() {
        // aml maintained: events at 9, 13, 18, 23, 31, 34, 48; censored at 13, 28, 45, 161.
        let time = [
            9.0, 13.0, 13.0, 18.0, 23.0, 28.0, 31.0, 34.0, 45.0, 48.0, 161.0,
        ];
        let status = [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0];
        let result = simple(&time, &status, None);
        let w = column(&result, 0);
        // Kaplan-Meier at 48 from the redistributed weights: survfit gives
        // S(48) = 0.1840909 for this group, and the weight of the trailing
        // censor at 161 is not redistributed to anyone (R's sum is 0.8159091).
        let km_48: f64 = 1.0
            - w.iter()
                .zip(&time)
                .filter(|(_, t)| **t <= 48.0)
                .map(|(w, _)| w)
                .sum::<f64>();
        assert!((km_48 - 0.1840909).abs() < 1e-6);
        assert!((w.iter().sum::<f64>() - 0.8159091).abs() < 1e-6);
        assert!((w[0] - 0.09090909).abs() < 1e-7);
        assert!((w[9] - 0.18409091).abs() < 1e-7);
        assert_eq!(w[2], 0.0);
        assert_eq!(w[10], 0.0);
        assert!(result.times.is_empty());
    }

    #[test]
    fn reporting_times_hand_the_weight_forward() {
        let time = [1.0, 2.0, 3.0, 4.0];
        let status = [0, 1, 0, 1];
        let result = simple(&time, &status, Some(&[1.5, 3.5]));
        assert_eq!(result.times, vec![1.5, 3.5]);
        // At 1.5: the censor at 1 has given 1/4 to the three others.
        assert_eq!(
            column(&result, 0),
            vec![0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        );
        // At 3.5: the event at 2 keeps 1/3; the censor at 3 gives its 1/3 to row 4.
        assert_eq!(column(&result, 1), vec![0.0, 1.0 / 3.0, 0.0, 2.0 / 3.0]);
    }

    #[test]
    fn strata_and_weights_are_handled_per_stratum() {
        let time = [1.0, 2.0, 3.0, 1.0, 2.0];
        let status = [0, 1, 1, 1, 0];
        let strata = [0, 0, 0, 1, 1];
        let weights = [1.0, 2.0, 1.0, 3.0, 1.0];
        let result = rttright::<i64>(RttrightInput {
            start: None,
            time: &time,
            status: &status,
            strata: Some(&strata),
            weights: Some(&weights),
            id: None,
            times: None,
            timefix: true,
            renorm: true,
        })
        .unwrap();
        let w = column(&result, 0);
        assert!((w[0..3].iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((w[1] - 2.0 / 3.0).abs() < 1e-12);
        assert!((w[2] - 1.0 / 3.0).abs() < 1e-12);
        assert_eq!(w[3], 0.75);
        assert_eq!(w[4], 0.0);

        let unnormalised = rttright::<i64>(RttrightInput {
            start: None,
            time: &time,
            status: &status,
            strata: Some(&strata),
            weights: Some(&weights),
            id: None,
            times: None,
            timefix: true,
            renorm: false,
        })
        .unwrap();
        assert_eq!(column(&unnormalised, 0)[3], 3.0);
    }

    #[test]
    fn multistate_events_all_receive_weight() {
        let time = [1.0, 2.0, 3.0];
        let status = [2, 0, 1];
        let result = simple(&time, &status, None);
        assert_eq!(column(&result, 0), vec![1.0 / 3.0, 0.0, 2.0 / 3.0]);
    }

    #[test]
    fn counting_process_rows_pass_the_baton_within_subject() {
        let id = [1i64, 1, 2, 3];
        let start = [0.0, 2.0, 0.0, 0.0];
        let stop = [2.0, 5.0, 3.0, 6.0];
        let status = [0, 1, 0, 1];
        let result = rttright(RttrightInput {
            start: Some(&start),
            time: &stop,
            status: &status,
            strata: None,
            weights: None,
            id: Some(&id),
            times: Some(&[1.0, 4.0, 7.0]),
            timefix: true,
            renorm: true,
        })
        .unwrap();
        // Three subjects, so each starts with weight 1/3; only subject 2's
        // last row at 3 is a censoring.  As in R, a subject's last event row
        // carries its weight at every reporting time, even before the row's
        // interval has started.
        let close = |actual: Vec<f64>, expected: &[f64]| {
            assert!(
                actual
                    .iter()
                    .zip(expected)
                    .all(|(a, e)| (a - e).abs() < 1e-12),
                "{actual:?} != {expected:?}"
            );
        };
        close(
            column(&result, 0),
            &[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        );
        close(column(&result, 1), &[0.0, 0.5, 0.0, 0.5]);
        close(column(&result, 2), &[0.0, 0.5, 0.0, 0.5]);

        let no_id = rttright::<i64>(RttrightInput {
            start: Some(&start),
            time: &stop,
            status: &status,
            strata: None,
            weights: None,
            id: None,
            times: None,
            timefix: true,
            renorm: true,
        });
        assert!(no_id.is_err());
    }

    #[test]
    fn rejects_bad_inputs() {
        assert!(
            rttright::<i64>(RttrightInput {
                start: None,
                time: &[1.0],
                status: &[-1],
                strata: None,
                weights: None,
                id: None,
                times: None,
                timefix: true,
                renorm: true,
            })
            .is_err()
        );
        assert!(
            rttright::<i64>(RttrightInput {
                start: None,
                time: &[1.0, 2.0],
                status: &[1, 1],
                strata: None,
                weights: Some(&[0.0, 0.0]),
                id: None,
                times: None,
                timefix: true,
                renorm: true,
            })
            .is_err()
        );
        assert!(
            rttright(RttrightInput {
                start: None,
                time: &[1.0, 2.0],
                status: &[1, 1],
                strata: None,
                weights: Some(&[1.0, 2.0]),
                id: Some(&[1i64, 1]),
                times: None,
                timefix: true,
                renorm: true,
            })
            .is_err()
        );
    }
}
