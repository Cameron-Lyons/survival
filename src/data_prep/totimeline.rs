//! R's `totimeline` (`R/fromtimeline.R`), the reverse of `surv2counting`:
//! turn counting-process rows `(time1, time2, state)` into timeline rows.
//! A subject with `k` intervals yields `k + 1` rows: its entry time with
//! the initial state, then the end of every interval with its outcome.
//! R ships this function as an untested draft; the row bookkeeping here
//! follows it, with the initial state supplied by the caller.

use super::id_value::{IdValue, SubjectId};
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::{validate_finite, validate_length};
use pyo3::prelude::*;

/// Timeline rows.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct TotimelineResult {
    /// Zero-based input row that supplies the time and state.
    #[pyo3(get)]
    pub time_row: Vec<usize>,
    /// Zero-based input row that supplies the covariates (a subject's last
    /// row is repeated for its final time point).
    #[pyo3(get)]
    pub covariate_row: Vec<usize>,
    #[pyo3(get)]
    pub time: Vec<f64>,
    /// State at `time`: `istate` on a subject's first row, the interval's
    /// outcome otherwise (0 for censored).
    #[pyo3(get)]
    pub state: Vec<i32>,
}

/// Convert counting-process rows to timeline rows.  Rows of a subject
/// must be consecutive and in time order; `istate` is the state each row
/// starts in (only the first row of a subject is used).
pub fn totimeline<I: SubjectId>(
    id: &[I],
    time1: &[f64],
    time2: &[f64],
    status: &[i32],
    istate: &[i32],
) -> SurvivalResult<TotimelineResult> {
    let n = id.len();
    validate_length(n, time1.len(), "time1")?;
    validate_length(n, time2.len(), "time2")?;
    validate_length(n, status.len(), "status")?;
    validate_length(n, istate.len(), "istate")?;
    validate_finite(time1, "time1")?;
    validate_finite(time2, "time2")?;
    let keys: Vec<I::Key> = id.iter().map(SubjectId::key).collect();
    let mut out = TotimelineResult {
        time_row: Vec::with_capacity(2 * n),
        covariate_row: Vec::with_capacity(2 * n),
        time: Vec::with_capacity(2 * n),
        state: Vec::with_capacity(2 * n),
    };
    for i in 0..n {
        let first = i == 0 || keys[i - 1] != keys[i];
        let last = i + 1 == n || keys[i + 1] != keys[i];
        if !first && time1[i] < time2[i - 1] {
            return Err(SurvivalError::invalid_input(
                "rows of a subject must be consecutive and in time order",
            ));
        }
        if first {
            out.time_row.push(i);
            out.covariate_row.push(i);
            out.time.push(time1[i]);
            out.state.push(istate[i]);
        }
        out.time_row.push(i);
        out.covariate_row.push(if last { i } else { i + 1 });
        out.time.push(time2[i]);
        out.state.push(status[i]);
    }
    Ok(out)
}

/// Python entry point of [`totimeline`].
#[pyfunction(name = "totimeline")]
pub fn totimeline_py(
    id: Vec<IdValue>,
    time1: Vec<f64>,
    time2: Vec<f64>,
    status: Vec<i32>,
    istate: Vec<i32>,
) -> PyResult<TotimelineResult> {
    if id.iter().any(IdValue::is_missing) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "id must not contain missing values",
        ));
    }
    Ok(totimeline(&id, &time1, &time2, &status, &istate)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_prep::surv2counting::{Repeated, surv2counting};

    #[test]
    fn each_subject_gains_one_row() {
        let out = totimeline(
            &[1i64, 1, 2],
            &[0.0, 2.0, 0.0],
            &[2.0, 5.0, 3.0],
            &[2, 3, 0],
            &[1, 2, 1],
        )
        .unwrap();
        assert_eq!(out.time, vec![0.0, 2.0, 5.0, 0.0, 3.0]);
        assert_eq!(out.state, vec![1, 2, 3, 1, 0]);
        assert_eq!(out.time_row, vec![0, 0, 1, 2, 2]);
        assert_eq!(out.covariate_row, vec![0, 1, 1, 2, 2]);
    }

    #[test]
    fn round_trips_through_surv2counting() {
        let out = totimeline(
            &["a", "a", "b"],
            &[0.0, 2.0, 0.0],
            &[2.0, 5.0, 3.0],
            &[2, 3, 0],
            &[1, 2, 1],
        )
        .unwrap();
        let ids: Vec<&str> = out.time_row.iter().map(|&r| ["a", "a", "b"][r]).collect();
        let status: Vec<Option<i32>> = out.state.iter().map(|&s| Some(s)).collect();
        let back = surv2counting(&ids, &out.time, &status, true, Repeated::No, &[]).unwrap();
        assert_eq!(back.tstart, vec![0.0, 2.0, 0.0]);
        assert_eq!(back.tstop, vec![2.0, 5.0, 3.0]);
        assert_eq!(back.status, vec![Some(2), Some(3), Some(0)]);
        assert_eq!(back.istate, Some(vec![1, 2, 1]));
    }

    #[test]
    fn rejects_unordered_subjects_and_bad_shapes() {
        assert!(totimeline(&[1i64, 1], &[2.0, 0.0], &[3.0, 1.0], &[0, 1], &[1, 1]).is_err());
        assert!(totimeline(&[1i64], &[], &[1.0], &[0], &[1]).is_err());
        assert!(
            totimeline::<i64>(&[], &[], &[], &[], &[])
                .unwrap()
                .time
                .is_empty()
        );
    }
}
