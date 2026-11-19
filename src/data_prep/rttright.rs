use super::aeq_surv::aeq_surv;
use super::id_value::{IdValue, SubjectId, first_appearance_codes};
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::risk_weight::{RiskWeight, needs_exact_updates};
use crate::internal::validation::{validate_finite, validate_length, validate_non_negative};
use pyo3::prelude::*;
use std::collections::BTreeMap;

pub struct RttrightInput<'a, I> {
    pub start: Option<&'a [f64]>,
    pub time: &'a [f64],
    pub status: &'a [i32],
    pub strata: Option<&'a [usize]>,
    pub weights: Option<&'a [f64]>,
    pub id: Option<&'a [I]>,
    pub times: Option<&'a [f64]>,
    pub timefix: bool,
    pub renorm: bool,
}

#[doc = " The weights: one row per observation and one column per reporting"]
#[doc = " time (a single column when no times were requested)."]
#[derive(Debug, Clone, PartialEq)]
#[pyclass(module = "survival._survival", from_py_object)]
pub struct RttrightResult {
    #[pyo3(get)]
    pub weights: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub times: Vec<f64>,
}

struct CensoringCurve {
    times: Vec<f64>,
    surv: Vec<f64>,
    log_scale: bool,
}

impl CensoringCurve {
    fn before(&self, t: f64) -> f64 {
        let k = self.times.partition_point(|&c| c < t);
        self.surv[k]
    }
}

/// Include this tied censoring group in the reverse risk sweep and return
/// log2 of the fraction that survives censoring at its time.
fn add_censorings(
    active: &mut RiskWeight,
    group: &[usize],
    weight: &[f64],
    censored_weight: f64,
) -> f64 {
    // Read the remaining risk before inserting the tied censorings.
    // Subtracting their weight from a rounded total can erase it.
    let remaining = active.total();
    let log_remaining = if remaining.is_finite() {
        remaining.log2()
    } else {
        active.log2_total()
    };
    for &row in group {
        active.add(weight[row]);
    }
    let odds = if remaining.is_finite() {
        censored_weight / remaining
    } else {
        (censored_weight.log2() - log_remaining).exp2()
    };
    if odds.is_finite() {
        -odds.ln_1p() * std::f64::consts::LOG2_E
    } else {
        log_remaining - active.log2_total()
    }
}

fn censoring_curve<const LOG_SCALE: bool>(
    rows: &[usize],
    start: Option<&[f64]>,
    stop: &[f64],
    censor: &[bool],
    weight: &[f64],
) -> CensoringCurve {
    let initial = if LOG_SCALE { 0.0 } else { 1.0 };
    let mut times = Vec::new();
    let mut surv = vec![initial];
    if !rows.iter().any(|&row| censor[row]) {
        return CensoringCurve {
            times,
            surv,
            log_scale: LOG_SCALE,
        };
    }
    let mut by_stop = rows.to_vec();
    by_stop.sort_by(|&a, &b| {
        stop[a]
            .partial_cmp(&stop[b])
            .expect("validated finite stop times")
            .then(censor[a].cmp(&censor[b]))
    });
    let entry = start.map(|s| {
        let mut by_start = rows.to_vec();
        by_start.sort_by(|&a, &b| s[a].total_cmp(&s[b]));
        (s, by_start)
    });
    let mut next_entry = entry.as_ref().map_or(0, |(_, order)| order.len());
    let mut at_risk = 0.0;
    let mut exact = RiskWeight::default();
    let y2_times = by_stop.chunk_by(|&a, &b| stop[a] == stop[b] && censor[a] == censor[b]);
    for group in y2_times.rev() {
        let (time, censored) = (stop[group[0]], censor[group[0]]);
        let mut censored_weight = 0.0;
        for &row in group.iter().rev() {
            let w = weight[row];
            if !LOG_SCALE {
                at_risk += w;
            } else if !censored {
                exact.add(w);
            }
            if censored {
                censored_weight += w;
            }
        }
        if let Some((s, by_start)) = &entry {
            while next_entry > 0 {
                let entered = s[by_start[next_entry - 1]];
                if entered < time || (censored && entered == time) {
                    break;
                }
                let leaving = weight[by_start[next_entry - 1]];
                if LOG_SCALE {
                    exact.add(-leaving);
                } else {
                    at_risk -= leaving;
                }
                next_entry -= 1;
            }
        }
        if censored {
            let step = if censored_weight == 0.0 {
                initial
            } else if LOG_SCALE {
                add_censorings(&mut exact, group, weight, censored_weight)
            } else {
                (at_risk - censored_weight) / at_risk
            };
            times.push(time);
            surv.push(step);
        }
    }
    times.reverse();
    surv[1..].reverse();
    let mut g = initial;
    let mut correction = 0.0;
    for step in &mut surv[1..] {
        let next = if LOG_SCALE {
            let increment = *step - correction;
            let next = g + increment;
            correction = if next.is_finite() {
                (next - g) - increment
            } else {
                0.0
            };
            next
        } else {
            g * *step
        };
        // Repeated censoring in successive intervals can exhaust the exponent
        // range even when every subject has the same small case weight.
        if !LOG_SCALE && g > 0.0 && *step > 0.0 && next < f64::MIN_POSITIVE {
            return censoring_curve::<true>(rows, start, stop, censor, weight);
        }
        g = next;
        *step = g;
    }
    CensoringCurve {
        times,
        surv,
        log_scale: LOG_SCALE,
    }
}

fn divide_by_log2(weight: f64, denominator: f64) -> f64 {
    if !denominator.is_finite() || denominator.abs() <= 1022.0 {
        return weight / denominator.exp2();
    }
    if weight == 0.0 {
        return 0.0;
    }
    // A finite quotient of binary64 weights spans at most 2098 binary
    // exponents. Split the scale into three factors, retaining the original
    // significand even for a weight of f64::MAX.
    let shift = (-denominator / 3.0).trunc();
    let factor = shift.exp2();
    weight * factor * factor * (-denominator - 2.0 * shift).exp2()
}

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
    if let Some(s) = input.start {
        validate_length(n, s.len(), "start")?;
        validate_finite(s, "start")?;
        if let Some(index) = s.iter().zip(input.time).position(|(s, t)| s >= t) {
            return Err(SurvivalError::invalid_input(format!(
                "Stop time must be > start time (row {index})"
            )));
        }
    }

    let (start, stop) = match (input.start, input.timefix) {
        (Some(s), true) => {
            let fixed = aeq_surv(s, Some(input.time), None)?;
            (Some(fixed.time), fixed.time2.unwrap_or_default())
        }
        (None, true) => (None, aeq_surv(input.time, None, None)?.time),
        (s, false) => (s.map(<[f64]>::to_vec), input.time.to_vec()),
    };

    let subjects = match input.id {
        Some(id) => {
            validate_length(n, id.len(), "id")?;
            let (codes, representatives) = first_appearance_codes(id);
            Some((codes, representatives.len()))
        }
        None => None,
    };
    let mut last = vec![true; n];
    if let Some((codes, n_subjects)) = &subjects {
        let mut weight_range = vec![(f64::INFINITY, f64::NEG_INFINITY); *n_subjects];
        for i in 0..n {
            let (lo, hi) = weight_range[codes[i]];
            weight_range[codes[i]] = (lo.min(casewt[i]), hi.max(casewt[i]));
        }
        if weight_range.iter().any(|(lo, hi)| hi - lo > 0.0) {
            return Err(SurvivalError::invalid_input(
                "there are subjects with multiple weights",
            ));
        }
        if let Some(s) = &start {
            let mut last_row = vec![usize::MAX; *n_subjects];
            let mut last_time = vec![f64::NEG_INFINITY; *n_subjects];
            let mut first_time = vec![f64::INFINITY; *n_subjects];
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
    }
    let censor: Vec<bool> = (0..n).map(|i| last[i] && input.status[i] == 0).collect();
    let has_weight: Vec<bool> = (0..n).map(|i| last[i] && input.status[i] > 0).collect();

    let stratum_rows: Vec<Vec<usize>> = match input.strata {
        Some(strata) => {
            let mut groups = BTreeMap::<usize, Vec<usize>>::new();
            for (row, &label) in strata.iter().enumerate() {
                groups.entry(label).or_default().push(row);
            }
            groups.into_values().collect()
        }
        None => vec![(0..n).collect()],
    };
    let query_times = input.times.unwrap_or_default();
    let n_columns = input.times.map_or(1, <[f64]>::len);
    let mut weights = vec![vec![0.0; n_columns]; n];
    let mut counted_in = vec![
        usize::MAX;
        subjects
            .as_ref()
            .filter(|_| input.renorm)
            .map_or(0, |(_, k)| *k)
    ];
    for (s, rows) in stratum_rows.iter().enumerate() {
        if rows.is_empty() {
            continue;
        }
        // A log ratio can remain representable even when either the stratum
        // total overflows or the normalized case weight underflows.
        let log_scale =
            input.weights.is_some() && needs_exact_updates(rows, &casewt, start.is_some());
        let mut total = 0.0;
        let mut exact = RiskWeight::default();
        if input.renorm {
            for &i in rows {
                if let Some((codes, _)) = &subjects {
                    if counted_in[codes[i]] == s {
                        continue;
                    }
                    counted_in[codes[i]] = s;
                }
                if log_scale {
                    exact.add(casewt[i]);
                } else {
                    total += casewt[i];
                }
            }
            total = if log_scale { exact.log2_total() } else { total };
            if (log_scale && total == f64::NEG_INFINITY) || (!log_scale && total == 0.0) {
                return Err(SurvivalError::invalid_input(
                    "weights must have a positive sum in every stratum when renorm is true",
                ));
            }
        }
        if n_columns == 0 {
            continue;
        }
        if !log_scale && input.renorm {
            for &i in rows {
                casewt[i] /= total;
            }
        }
        let curve = if log_scale {
            censoring_curve::<true>(rows, start.as_deref(), &stop, &censor, &casewt)
        } else {
            censoring_curve::<false>(rows, start.as_deref(), &stop, &censor, &casewt)
        };
        // The ordinary path has already normalized its weights if a small
        // censoring product triggered a retry on the log scale.
        let log_total = if log_scale { total } else { 0.0 };
        let adjust = |w: f64, g: f64| {
            if curve.log_scale {
                divide_by_log2(w, log_total + g)
            } else {
                w / g
            }
        };
        if input.times.is_none() {
            for &i in rows {
                if has_weight[i] {
                    weights[i][0] = adjust(casewt[i], curve.before(stop[i]));
                }
            }
            continue;
        }
        let gwt: Vec<f64> = query_times.iter().map(|&t| curve.before(t)).collect();
        for &i in rows {
            let row = &mut weights[i];
            let w = casewt[i];
            if has_weight[i] {
                let g_row = curve.before(stop[i]);
                for (value, &g_col) in row.iter_mut().zip(&gwt) {
                    *value = adjust(w, g_row.max(g_col));
                }
            } else {
                for ((value, &t), &g_col) in row.iter_mut().zip(query_times).zip(&gwt) {
                    if stop[i] >= t && start.as_ref().is_none_or(|s| s[i] < t) {
                        *value = adjust(w, g_col);
                    }
                }
            }
        }
    }
    Ok(RttrightResult {
        weights,
        times: query_times.to_vec(),
    })
}

#[doc = " Python entry point of [`rttright`]."]
#[pyfunction(name = "rttright")]
#[pyo3(signature = (time, status, start=None, strata=None, weights=None, id=None, times=None, timefix=true, renorm=true))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn rttright_py(
    py: Python<'_>,
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
    Ok(py.detach(|| {
        rttright(RttrightInput {
            start: start.as_deref(),
            time: &time,
            status: &status,
            strata: strata.as_deref(),
            weights: weights.as_deref(),
            id: id.as_deref(),
            times: times.as_deref(),
            timefix,
            renorm,
        })
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
        let time = [
            9.0, 13.0, 13.0, 18.0, 23.0, 28.0, 31.0, 34.0, 45.0, 48.0, 161.0,
        ];
        let status = [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0];
        let result = simple(&time, &status, None);
        let w = column(&result, 0);
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
        assert_eq!(
            column(&result, 0),
            vec![0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        );
        assert_eq!(column(&result, 1), vec![0.0, 1.0 / 3.0, 0.0, 2.0 / 3.0]);

        let empty = simple(&time, &status, Some(&[]));
        assert_eq!(empty.weights, vec![Vec::<f64>::new(); time.len()]);
        assert!(empty.times.is_empty());
    }

    #[test]
    fn censoring_redistributes_mass_across_the_full_weight_range() {
        // The first subject is censored. The two later events divide its
        // weight in proportion to their own weights. Their initial shares
        // can underflow even though their redistributed shares are 1/4, 3/4.
        for (large, small) in [(1e17, 1.0), (1e300, 1e-300), (f64::MAX, f64::from_bits(1))] {
            let result = rttright::<i64>(RttrightInput {
                start: None,
                time: &[1., 2., 3.],
                status: &[0, 1, 1],
                strata: None,
                weights: Some(&[large, small, small * 3.]),
                id: None,
                times: Some(&[2., 1., 4., 2.]),
                timefix: false,
                renorm: true,
            })
            .unwrap();
            for t in [0, 2, 3] {
                assert_eq!(result.weights[0][t], 0.0);
                assert!((result.weights[1][t] - 0.25).abs() < 2e-13);
                assert!((result.weights[2][t] - 0.75).abs() < 2e-13);
            }
            assert!((result.weights[0][1] - 1.).abs() < 2e-13);
        }
    }

    #[test]
    fn counting_rows_normalize_once_per_subject_when_the_total_overflows() {
        let result = rttright(RttrightInput {
            start: Some(&[0., 0., 1.]),
            time: &[1., 2., 3.],
            status: &[0, 0, 1],
            strata: None,
            weights: Some(&[1e308; 3]),
            id: Some(&[1_i64, 2, 1]),
            times: Some(&[1., 2., 3.]),
            timefix: false,
            renorm: true,
        })
        .unwrap();
        for (column_index, expected) in [[0.5, 0.5, 0.5], [0., 0.5, 0.5], [0., 0., 1.]]
            .iter()
            .enumerate()
        {
            for (row, &w) in expected.iter().enumerate() {
                assert!((result.weights[row][column_index] - w).abs() < 2e-13);
            }
        }
    }

    #[test]
    fn a_rounded_weight_bound_still_selects_overflow_safe_normalization() {
        let fit = rttright::<i64>(RttrightInput {
            start: None,
            time: &[1., 2., 3.],
            status: &[1; 3],
            strata: None,
            weights: Some(&[f64::MAX / 3.; 3]),
            id: None,
            times: None,
            timefix: false,
            renorm: true,
        })
        .unwrap();
        for row in fit.weights {
            assert!((row[0] - 1. / 3.).abs() < 1e-13);
        }
    }

    #[test]
    fn censor_and_event_at_signed_zero_share_the_same_time() {
        for weights in [None, Some([1e20, 1e20, 2.].as_slice())] {
            let fit = |first: f64| {
                rttright::<i64>(RttrightInput {
                    start: None,
                    time: &[first, 0., 1.],
                    status: &[0, 1, 1],
                    strata: None,
                    weights,
                    id: None,
                    times: None,
                    timefix: true,
                    renorm: true,
                })
                .unwrap()
            };
            assert_eq!(fit(-0.), fit(0.));
        }
    }

    #[test]
    fn successive_stratified_censorings_preserve_finite_redistributed_weights() {
        // Each pair of subjects moves to stratum 1 together. One is censored
        // there and the other has the event, before the next pair arrives.
        for (pairs, c, e) in [
            (1080, 2.0_f64.powi(-1000), 2.0_f64.powi(-1000)),
            (1000, 1e286, 1e300),
        ] {
            let (mut start, mut stop, mut status, mut id, mut strata, mut weights) =
                (vec![], vec![], vec![], vec![], vec![], vec![]);
            for j in 0..pairs {
                let entry = (3 * j + 2) as f64;
                start.extend([0., 0., entry, entry]);
                stop.extend([entry, entry, entry + 1., entry + 2.]);
                status.extend([0, 0, 0, 1]);
                let subject = 2 * i64::from(j);
                id.extend([subject, subject + 1, subject, subject + 1]);
                strata.extend([0, 0, 1, 1]);
                weights.extend([c, e, c, e]);
            }
            let fit = rttright(RttrightInput {
                start: Some(&start),
                time: &stop,
                status: &status,
                strata: Some(&strata),
                weights: Some(&weights),
                id: Some(&id),
                times: None,
                timefix: false,
                renorm: false,
            })
            .unwrap();
            let expected = if c == e {
                2.0_f64.powi(pairs - 1000)
            } else {
                e * (f64::from(pairs) * (c / e).ln_1p()).exp()
            };
            assert!((fit.weights.last().unwrap()[0] / expected - 1.0).abs() < 1e-13);
        }
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

    fn censoring_curve_by_definition(
        start: Option<&[f64]>,
        stop: &[f64],
        censor: &[bool],
        weight: &[f64],
    ) -> CensoringCurve {
        let rows = 0..stop.len();
        let mut times: Vec<f64> = rows
            .clone()
            .filter(|&i| censor[i])
            .map(|i| stop[i])
            .collect();
        times.sort_by(f64::total_cmp);
        times.dedup();
        let mut g = 1.0;
        let surv = std::iter::once(1.0)
            .chain(times.iter().map(|&c| {
                let (mut at_risk, mut censored) = (0.0, 0.0);
                for i in rows.clone() {
                    let entered = start.is_none_or(|s| s[i] <= c);
                    if entered && (stop[i] > c || (stop[i] == c && censor[i])) {
                        at_risk += weight[i];
                        if stop[i] == c && censor[i] {
                            censored += weight[i];
                        }
                    }
                }
                if censored > 0.0 {
                    g *= (at_risk - censored) / at_risk;
                }
                g
            }))
            .collect();
        CensoringCurve {
            times,
            surv,
            log_scale: false,
        }
    }

    #[test]
    fn censoring_sweep_matches_the_definition() {
        for seed in 0..256 {
            let mut rng = crate::internal::rng::Rng::with_seed(seed);
            let n = rng.usize(1..40);
            let stop: Vec<f64> = (0..n).map(|_| rng.usize(1..12) as f64).collect();
            let start: Vec<f64> = stop
                .iter()
                .map(|&t| rng.usize(0..t as usize) as f64)
                .collect();
            let censor: Vec<bool> = (0..n).map(|_| rng.bool()).collect();
            let weight: Vec<f64> = (0..n).map(|_| rng.usize(0..4) as f64 / 2.0).collect();
            let rows: Vec<usize> = (0..n).collect();
            for start in [None, Some(start.as_slice())] {
                let sweep = censoring_curve::<false>(&rows, start, &stop, &censor, &weight);
                let expected = censoring_curve_by_definition(start, &stop, &censor, &weight);
                assert_eq!(sweep.times, expected.times, "seed {seed}");
                for (a, b) in sweep.surv.iter().zip(&expected.surv) {
                    assert!((a - b).abs() < 1e-14, "seed {seed}: {a} != {b}");
                }
            }
        }
    }

    #[test]
    fn rejects_bad_inputs() {
        let reversed = rttright(RttrightInput {
            start: Some(&[0.0, 3.0]),
            time: &[2.0, 3.0],
            status: &[1, 1],
            strata: None,
            weights: None,
            id: Some(&[1i64, 2]),
            times: None,
            timefix: false,
            renorm: true,
        });
        assert!(
            reversed
                .unwrap_err()
                .to_string()
                .contains("Stop time must be > start time (row 1)")
        );
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
