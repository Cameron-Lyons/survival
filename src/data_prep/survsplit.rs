//! R's `survSplit` (`R/survSplit.R` and `src/survsplit.c`): split
//! follow-up intervals at a set of cutpoints.
//!
//! The C kernel ([`survsplit_intervals`]) splits `(tstart, tstop]`
//! intervals; [`survsplit`] adds the data side of the R function: the
//! near-tie fix that includes the cutpoints, the `zero` start of
//! right-censored data, and the censoring of every row that ends at a cut
//! (for timeline data, of every row that starts at one).

use super::aeq_surv::aeq_surv;
use super::id_value::{IdValue, SubjectId};
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::validate_length;
use pyo3::prelude::*;

/// The response of a `survSplit` call.
pub enum SurvSplitResponse<'a, I> {
    /// `Surv(time, status)`: follow-up starts at `zero`.
    Right { time: &'a [f64], status: &'a [f64] },
    /// `Surv(tstart, tstop, status)`.
    Counting {
        start: &'a [f64],
        stop: &'a [f64],
        status: &'a [f64],
    },
    /// `Surv2(time, state)` timeline data, one row per visit of `id`.
    Timeline {
        id: &'a [I],
        time: &'a [f64],
        status: &'a [f64],
    },
}

/// The split data set; every vector has one entry per output row.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct SurvSplitResult {
    /// Zero-based row of the input each output row came from.
    #[pyo3(get)]
    pub row: Vec<usize>,
    /// Zero-based episode number (R's `episode` column minus one).
    #[pyo3(get)]
    pub interval: Vec<usize>,
    #[pyo3(get)]
    pub start: Vec<f64>,
    #[pyo3(get)]
    pub end: Vec<f64>,
    /// The status of each output row, censored (0) at every cut.
    #[pyo3(get)]
    pub status: Vec<f64>,
    /// Whether the row was created by a cut (R's `added` column).
    #[pyo3(get)]
    pub censor: Vec<bool>,
    /// The cutpoints after the near-tie fix.
    #[pyo3(get)]
    pub cut: Vec<f64>,
}

/// The kernel of `src/survsplit.c`: split every `(tstart, tstop]` at the
/// cutpoints strictly inside it.  Rows with a missing endpoint pass
/// through unchanged.  `cut` must be sorted and unique.
pub fn survsplit_intervals(
    tstart: &[f64],
    tstop: &[f64],
    cut: &[f64],
) -> SurvivalResult<SurvSplitResult> {
    validate_length(tstart.len(), tstop.len(), "tstop")?;
    let n = tstart.len();
    let mut extra = 0;
    for i in 0..n {
        if !tstart[i].is_nan() && !tstop[i].is_nan() {
            extra += cut
                .iter()
                .filter(|&&c| c > tstart[i] && c < tstop[i])
                .count();
        }
    }
    let n2 = n + extra;
    let mut out = SurvSplitResult {
        row: Vec::with_capacity(n2),
        interval: Vec::with_capacity(n2),
        start: Vec::with_capacity(n2),
        end: Vec::with_capacity(n2),
        status: Vec::new(),
        censor: Vec::with_capacity(n2),
        cut: cut.to_vec(),
    };
    for i in 0..n {
        if tstart[i].is_nan() || tstop[i].is_nan() {
            out.row.push(i);
            out.interval.push(0);
            out.start.push(tstart[i]);
            out.end.push(tstop[i]);
            out.censor.push(false);
            continue;
        }
        let mut j = cut.partition_point(|&c| c <= tstart[i]);
        out.row.push(i);
        out.interval.push(j);
        out.start.push(tstart[i]);
        while j < cut.len() && cut[j] < tstop[i] {
            // cut[j] > tstart[i] holds by construction of j.
            out.end.push(cut[j]);
            out.censor.push(true);
            out.row.push(i);
            out.interval.push(j + 1);
            out.start.push(cut[j]);
            j += 1;
        }
        out.end.push(tstop[i]);
        out.censor.push(false);
    }
    Ok(out)
}

/// The data side of `survSplit`.
pub fn survsplit<I: SubjectId>(
    response: SurvSplitResponse<'_, I>,
    cut: &[f64],
    zero: f64,
    timefix: bool,
) -> SurvivalResult<SurvSplitResult> {
    if cut.iter().any(|c| !c.is_finite()) {
        return Err(SurvivalError::invalid_input(
            "cut must be a vector of finite numbers",
        ));
    }
    let mut cut = cut.to_vec();
    cut.sort_by(|a, b| a.total_cmp(b));
    cut.dedup();
    let ntimes = cut.len();

    match response {
        SurvSplitResponse::Timeline { id, time, status } => {
            validate_length(id.len(), time.len(), "time")?;
            validate_length(id.len(), status.len(), "status")?;
            let n = id.len();
            let mut time = time.to_vec();
            if timefix {
                // Include the cutpoints in the aeqSurv process.
                let mut joint = cut.clone();
                joint.extend_from_slice(&time);
                let fixed = aeq_surv(&joint, None, None)?.time;
                cut = fixed[..ntimes].to_vec();
                time = fixed[ntimes..].to_vec();
            }
            // A fake stop for each row: the subject's next time, or the
            // row's own time for its last row.
            let keys: Vec<I::Key> = id.iter().map(SubjectId::key).collect();
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| {
                keys[a]
                    .cmp(&keys[b])
                    .then_with(|| time[a].total_cmp(&time[b]))
            });
            let mut fake_stop = time.clone();
            for pair in order.windows(2) {
                if keys[pair[0]] == keys[pair[1]] {
                    fake_stop[pair[0]] = time[pair[1]];
                }
            }
            let mut out = survsplit_intervals(&time, &fake_stop, &cut)?;
            out.status = out.row.iter().map(|&r| status[r]).collect();
            // The C routine marks the parent of each insertion as censored;
            // for timeline data it is the inserted row that is censored.
            for k in 0..out.censor.len() {
                if out.censor[k] && k + 1 < out.status.len() {
                    out.status[k + 1] = 0.0;
                }
            }
            Ok(out)
        }
        SurvSplitResponse::Right { time, status } => {
            validate_length(time.len(), status.len(), "status")?;
            let mut time = time.to_vec();
            if timefix {
                let mut joint = cut.clone();
                joint.extend_from_slice(&time);
                let fixed = aeq_surv(&joint, None, None)?.time;
                cut = fixed[..ntimes].to_vec();
                time = fixed[ntimes..].to_vec();
            }
            if time.iter().any(|&t| !t.is_nan() && t <= zero) {
                return Err(SurvivalError::invalid_input(
                    "'zero' parameter must be less than any observed times",
                ));
            }
            let start = vec![zero; time.len()];
            finish_surv(&start, &time, status, &cut)
        }
        SurvSplitResponse::Counting {
            start,
            stop,
            status,
        } => {
            validate_length(start.len(), stop.len(), "stop")?;
            validate_length(start.len(), status.len(), "status")?;
            let (mut start, mut stop) = (start.to_vec(), stop.to_vec());
            if timefix {
                // The cutpoints ride along as stop times of fake rows.
                let min_start = start
                    .iter()
                    .copied()
                    .filter(|s| !s.is_nan())
                    .fold(f64::INFINITY, f64::min);
                let mut joint_start = vec![min_start; ntimes];
                joint_start.extend_from_slice(&start);
                let mut joint_stop = cut.clone();
                joint_stop.extend_from_slice(&stop);
                let fixed = aeq_surv(&joint_start, Some(&joint_stop), None)?;
                let fixed_stop = fixed.time2.unwrap_or_default();
                cut = fixed_stop[..ntimes].to_vec();
                start = fixed.time[ntimes..].to_vec();
                stop = fixed_stop[ntimes..].to_vec();
            }
            finish_surv(&start, &stop, status, &cut)
        }
    }
}

fn finish_surv(
    start: &[f64],
    stop: &[f64],
    status: &[f64],
    cut: &[f64],
) -> SurvivalResult<SurvSplitResult> {
    if start
        .iter()
        .zip(stop)
        .any(|(&s, &e)| !s.is_nan() && !e.is_nan() && s >= e)
    {
        return Err(SurvivalError::invalid_input(
            "start time must be < stop time",
        ));
    }
    let mut out = survsplit_intervals(start, stop, cut)?;
    out.status = out
        .row
        .iter()
        .zip(&out.censor)
        .map(|(&r, &censored)| if censored { 0.0 } else { status[r] })
        .collect();
    Ok(out)
}

/// Python entry point of [`survsplit`]: a right-censored response is
/// `(time, status)`, a counting-process one adds `start`, and timeline
/// data adds `id` instead.
#[pyfunction(name = "survsplit")]
#[pyo3(signature = (time, status, cut, start=None, id=None, zero=0.0, timefix=true))]
#[allow(clippy::too_many_arguments)]
pub fn survsplit_py(
    time: Vec<f64>,
    status: Vec<f64>,
    cut: Vec<f64>,
    start: Option<Vec<f64>>,
    id: Option<Vec<IdValue>>,
    zero: f64,
    timefix: bool,
) -> PyResult<SurvSplitResult> {
    let response = match (&start, &id) {
        (Some(start), None) => SurvSplitResponse::Counting {
            start,
            stop: &time,
            status: &status,
        },
        (None, Some(id)) => SurvSplitResponse::Timeline {
            id,
            time: &time,
            status: &status,
        },
        (None, None) => SurvSplitResponse::Right {
            time: &time,
            status: &status,
        },
        (Some(_), Some(_)) => {
            return Err(SurvivalError::invalid_input(
                "timeline data has no start times; give either start or id",
            )
            .into());
        }
    };
    Ok(survsplit(response, &cut, zero, timefix)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn right<'a>(time: &'a [f64], status: &'a [f64]) -> SurvSplitResponse<'a, i64> {
        SurvSplitResponse::Right { time, status }
    }

    #[test]
    fn kernel_splits_at_interior_cuts_only() {
        let out = survsplit_intervals(&[0.0, 5.0], &[10.0, 15.0], &[]).unwrap();
        assert_eq!(out.start, vec![0.0, 5.0]);
        assert_eq!(out.end, vec![10.0, 15.0]);

        let out = survsplit_intervals(&[0.0], &[10.0], &[0.0, 3.0, 7.0, 10.0]).unwrap();
        assert_eq!(out.row, vec![0, 0, 0]);
        assert_eq!(out.interval, vec![1, 2, 3]);
        assert_eq!(out.start, vec![0.0, 3.0, 7.0]);
        assert_eq!(out.end, vec![3.0, 7.0, 10.0]);
        assert_eq!(out.censor, vec![true, true, false]);

        let nan = survsplit_intervals(&[f64::NAN], &[f64::NAN], &[5.0]).unwrap();
        assert_eq!(nan.row, vec![0]);
        assert!(nan.start[0].is_nan());
    }

    #[test]
    fn right_censored_rows_start_at_zero_and_cuts_censor() {
        let out = survsplit(
            right(&[8.0, 25.0], &[1.0, 1.0]),
            &[20.0, 10.0, 10.0],
            0.0,
            true,
        )
        .unwrap();
        assert_eq!(out.cut, vec![10.0, 20.0]);
        assert_eq!(out.row, vec![0, 1, 1, 1]);
        assert_eq!(out.interval, vec![0, 0, 1, 2]);
        assert_eq!(out.start, vec![0.0, 0.0, 10.0, 20.0]);
        assert_eq!(out.end, vec![8.0, 10.0, 20.0, 25.0]);
        assert_eq!(out.status, vec![1.0, 0.0, 0.0, 1.0]);
        assert_eq!(out.censor, vec![false, true, true, false]);

        let err = survsplit(right(&[0.0], &[1.0]), &[1.0], 0.0, true).unwrap_err();
        assert!(err.to_string().contains("'zero' parameter"));
        let err = survsplit(right(&[1.0], &[1.0]), &[f64::NAN], 0.0, true).unwrap_err();
        assert!(err.to_string().contains("finite"));
    }

    #[test]
    fn counting_process_rows_keep_their_start_and_reject_reversed_intervals() {
        let response: SurvSplitResponse<'_, i64> = SurvSplitResponse::Counting {
            start: &[2.0, 0.0],
            stop: &[9.0, 4.0],
            status: &[1.0, 0.0],
        };
        let out = survsplit(response, &[3.0, 6.0], 0.0, true).unwrap();
        assert_eq!(out.start, vec![2.0, 3.0, 6.0, 0.0, 3.0]);
        assert_eq!(out.end, vec![3.0, 6.0, 9.0, 3.0, 4.0]);
        assert_eq!(out.status, vec![0.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(out.interval, vec![0, 1, 2, 0, 1]);

        let reversed: SurvSplitResponse<'_, i64> = SurvSplitResponse::Counting {
            start: &[5.0],
            stop: &[5.0],
            status: &[1.0],
        };
        assert!(survsplit(reversed, &[3.0], 0.0, false).is_err());
    }

    #[test]
    fn near_tied_cutpoints_are_snapped_with_the_times() {
        let out = survsplit(
            right(&[10.0 + 1e-12, 20.0], &[1.0, 1.0]),
            &[10.0],
            0.0,
            true,
        )
        .unwrap();
        // 10 + 1e-12 becomes 10, so no split happens for the first row.
        assert_eq!(out.row, vec![0, 1, 1]);
        assert_eq!(out.end[0], 10.0);
        let unfixed = survsplit(
            right(&[10.0 + 1e-12, 20.0], &[1.0, 1.0]),
            &[10.0],
            0.0,
            false,
        )
        .unwrap();
        assert_eq!(unfixed.row, vec![0, 0, 1, 1]);
    }

    #[test]
    fn timeline_data_censors_the_inserted_rows() {
        let response = SurvSplitResponse::Timeline {
            id: &[1i64, 1, 2],
            time: &[0.0, 10.0, 4.0],
            status: &[1.0, 2.0, 1.0],
        };
        let out = survsplit(response, &[5.0], 0.0, true).unwrap();
        assert_eq!(out.row, vec![0, 0, 1, 2]);
        assert_eq!(out.start, vec![0.0, 5.0, 10.0, 4.0]);
        assert_eq!(out.status, vec![1.0, 0.0, 2.0, 1.0]);
        assert_eq!(out.censor, vec![true, false, false, false]);
    }
}
