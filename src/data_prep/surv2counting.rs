//! R's `surv2counting` (`R/fromtimeline.R`), the engine of `fromtimeline`
//! and of `coxph`/`survfit` on `Surv2` responses: turn a timeline data set
//! (one row per subject per time, giving the state or event at that time)
//! into counting-process rows `(tstart, tstop, status)` with the subject's
//! current state.
//!
//! A subject with `k` rows yields `k - 1` intervals; row `j + 1` supplies
//! the end time and outcome of interval `j`, row `j` everything else.
//! Covariate values are carried forward over missing entries; the caller
//! copies values (of any type) along [`Surv2CountingResult::carry_from`].

use super::id_value::{IdValue, SubjectId};
use super::tmerge::tmerge_carry_forward;
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::{validate_finite, validate_length};
use pyo3::prelude::*;

/// R's `repeated` argument of `Surv2`/`fromtimeline`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Repeated {
    /// A transition into the subject's current state is censored.
    No,
    /// Every recorded outcome counts, even a repeat of the current state.
    Yes,
    /// Only the first instance of each outcome counts.
    First,
}

impl Repeated {
    /// Parse `FALSE`/`TRUE`/`"first"`.
    pub fn parse(value: &str) -> SurvivalResult<Self> {
        match value.to_lowercase().as_str() {
            "false" | "no" => Ok(Self::No),
            "true" | "yes" => Ok(Self::Yes),
            "first" => Ok(Self::First),
            _ => Err(SurvivalError::invalid_input(
                "invalid value for repeated option",
            )),
        }
    }
}

/// Counting-process rows, in the order of the input rows they start from.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct Surv2CountingResult {
    /// Zero-based input row each interval starts from (all but the last
    /// row of every subject).
    #[pyo3(get)]
    pub row: Vec<usize>,
    #[pyo3(get)]
    pub tstart: Vec<f64>,
    #[pyo3(get)]
    pub tstop: Vec<f64>,
    /// Outcome at `tstop`: 0 censored, otherwise the state code; `None`
    /// keeps a missing status of a plain 0/1 response.
    #[pyo3(get)]
    pub status: Vec<Option<i32>>,
    /// The state the subject is in during the interval; absent when the
    /// first row of every subject is censored (ordinary or competing-risks
    /// data).
    #[pyo3(get)]
    pub istate: Option<Vec<i32>>,
    /// For each requested covariate (outer) and output row (inner), the
    /// input row whose value applies: the interval's own start row, or an
    /// earlier row of the subject when that value is missing.
    #[pyo3(get)]
    pub carry_from: Vec<Vec<usize>>,
    /// Whether any subject has more than one interval, i.e. the response
    /// is of counting-process rather than right-censored type.
    #[pyo3(get)]
    pub counting: bool,
}

/// Convert timeline rows to counting-process rows.
///
/// `status[i]` is the recorded outcome at `time[i]` (`None` for missing),
/// `has_states` says whether the outcomes are factor levels (in which
/// case a missing status means censored), and `missing[c][i]` flags the
/// missing entries of covariate `c` for last-value-carried-forward.
pub fn surv2counting<I: SubjectId>(
    id: &[I],
    time: &[f64],
    status: &[Option<i32>],
    has_states: bool,
    repeated: Repeated,
    missing: &[Vec<bool>],
) -> SurvivalResult<Surv2CountingResult> {
    let n = id.len();
    validate_length(n, time.len(), "time")?;
    validate_length(n, status.len(), "status")?;
    validate_finite(time, "time")?;
    for (c, column) in missing.iter().enumerate() {
        if column.len() != n {
            return Err(SurvivalError::invalid_input(format!(
                "missing[{c}] must have one entry per row"
            )));
        }
    }
    if status.iter().flatten().any(|&s| s < 0) {
        return Err(SurvivalError::invalid_input(
            "status codes must be non-negative",
        ));
    }

    // isort <- order(id, y[, 1]); the data is not reordered on output.
    let keys: Vec<I::Key> = id.iter().map(SubjectId::key).collect();
    let mut isort: Vec<usize> = (0..n).collect();
    isort.sort_by(|&a, &b| {
        keys[a]
            .cmp(&keys[b])
            .then_with(|| time[a].total_cmp(&time[b]))
    });
    let same_subject = |p: usize, q: usize| keys[isort[p]] == keys[isort[q]];
    for p in 1..n {
        if same_subject(p - 1, p) && time[isort[p - 1]] == time[isort[p]] {
            return Err(SurvivalError::invalid_input("duplicated time for an id"));
        }
    }
    let first: Vec<bool> = (0..n).map(|p| p == 0 || !same_subject(p - 1, p)).collect();
    let last: Vec<bool> = (0..n)
        .map(|p| p + 1 == n || !same_subject(p, p + 1))
        .collect();

    // Sorted status with the substitutions R applies before pairing rows.
    let mut y_status: Vec<Option<i32>> = isort.iter().map(|&i| status[i]).collect();
    if has_states {
        for s in y_status.iter_mut() {
            s.get_or_insert(0);
        }
    }
    if repeated == Repeated::First {
        let mut seen: Vec<i32> = Vec::new();
        for p in 0..n {
            if first[p] {
                seen.clear();
            }
            if let Some(s) = y_status[p]
                && s != 0
            {
                if seen.contains(&s) {
                    y_status[p] = Some(0);
                } else {
                    seen.push(s);
                }
            }
        }
    }

    // The positions (in sorted order) that start an interval, and integer
    // ids for the carry-forward kernel.
    let starts: Vec<usize> = (0..n).filter(|&p| !last[p]).collect();
    let mut idi = Vec::with_capacity(starts.len());
    let mut code = 0;
    for (k, &p) in starts.iter().enumerate() {
        if k > 0 && !same_subject(starts[k - 1], p) {
            code += 1;
        }
        idi.push(code);
    }

    // Initial state: censored (or missing) first rows mean no initial state.
    let censored: Vec<bool> = y_status.iter().map(|s| s.is_none_or(|s| s == 0)).collect();
    let first_censored = (0..n).filter(|&p| first[p]).map(|p| censored[p]);
    let istate = if first_censored.clone().all(|c| c) {
        None
    } else if first_censored.clone().any(|c| c) {
        return Err(SurvivalError::invalid_input(
            "everyone or no one should have an initial state",
        ));
    } else {
        let mut istate: Vec<i32> = starts.iter().map(|&p| y_status[p].unwrap_or(0)).collect();
        let censored_starts: Vec<bool> = starts.iter().map(|&p| censored[p]).collect();
        if censored_starts.iter().any(|&c| c) {
            let carry = tmerge_carry_forward(&idi, &censored_starts)?;
            for k in 0..istate.len() {
                if censored_starts[k]
                    && let Some(from) = carry[k]
                {
                    istate[k] = istate[from];
                }
            }
        }
        Some(istate)
    };

    // status <- y2[!first, 2], censoring repeats of the current state.
    let mut out_status: Vec<Option<i32>> =
        (0..n).filter(|&p| !first[p]).map(|p| y_status[p]).collect();
    if repeated == Repeated::No
        && let Some(istate) = &istate
    {
        for (s, &current) in out_status.iter_mut().zip(istate) {
            if *s == Some(current) {
                *s = Some(0);
            }
        }
    }
    let tstart: Vec<f64> = starts.iter().map(|&p| time[isort[p]]).collect();
    let tstop: Vec<f64> = (0..n)
        .filter(|&p| !first[p])
        .map(|p| time[isort[p]])
        .collect();
    let counting = idi.windows(2).any(|w| w[0] == w[1]);

    // Last value carried forward for each covariate, in sorted order.
    let carry_from: Vec<Vec<usize>> = missing
        .iter()
        .map(|column| {
            let miss: Vec<bool> = starts.iter().map(|&p| column[isort[p]]).collect();
            tmerge_carry_forward(&idi, &miss).map(|carry| {
                carry
                    .iter()
                    .enumerate()
                    .map(|(k, from)| isort[starts[from.unwrap_or(k)]])
                    .collect()
            })
        })
        .collect::<SurvivalResult<_>>()?;

    // Put the rows back into the original order: order(isort[!last]).
    let rows: Vec<usize> = starts.iter().map(|&p| isort[p]).collect();
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by_key(|&k| rows[k]);
    let reorder = |values: &[usize]| order.iter().map(|&k| values[k]).collect::<Vec<_>>();
    Ok(Surv2CountingResult {
        row: reorder(&rows),
        tstart: order.iter().map(|&k| tstart[k]).collect(),
        tstop: order.iter().map(|&k| tstop[k]).collect(),
        status: order.iter().map(|&k| out_status[k]).collect(),
        istate: istate.map(|v| order.iter().map(|&k| v[k]).collect()),
        carry_from: carry_from.iter().map(|c| reorder(c)).collect(),
        counting,
    })
}

/// Python entry point of [`surv2counting`]; `repeated` is `"false"`,
/// `"true"` or `"first"`.
#[pyfunction(name = "surv2counting")]
#[pyo3(signature = (id, time, status, has_states=false, repeated="false", missing=Vec::new()))]
pub fn surv2counting_py(
    id: Vec<IdValue>,
    time: Vec<f64>,
    status: Vec<Option<i32>>,
    has_states: bool,
    repeated: &str,
    missing: Vec<Vec<bool>>,
) -> PyResult<Surv2CountingResult> {
    if id.iter().any(IdValue::is_missing) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "id and time cannot be missing",
        ));
    }
    Ok(surv2counting(
        &id,
        &time,
        &status,
        has_states,
        Repeated::parse(repeated)?,
        &missing,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subjects_become_intervals_in_original_row_order() {
        // Rows are deliberately shuffled across subjects.
        let id = [1i64, 2, 1, 1, 2];
        let time = [0.0, 0.0, 5.0, 2.0, 3.0];
        let status = [Some(1), Some(1), Some(3), Some(2), None];
        let out = surv2counting(&id, &time, &status, true, Repeated::No, &[]).unwrap();
        assert_eq!(out.row, vec![0, 1, 3]);
        assert_eq!(out.tstart, vec![0.0, 0.0, 2.0]);
        assert_eq!(out.tstop, vec![2.0, 3.0, 5.0]);
        assert_eq!(out.status, vec![Some(2), Some(0), Some(3)]);
        assert_eq!(out.istate, Some(vec![1, 1, 2]));
        assert!(out.counting);
    }

    #[test]
    fn repeated_states_are_censored_unless_allowed() {
        let id = [1i64, 1, 1];
        let time = [0.0, 1.0, 2.0];
        let status = [Some(1), Some(1), Some(2)];
        let stutter = surv2counting(&id, &time, &status, true, Repeated::No, &[]).unwrap();
        assert_eq!(stutter.status, vec![Some(0), Some(2)]);
        let repeated = surv2counting(&id, &time, &status, true, Repeated::Yes, &[]).unwrap();
        assert_eq!(repeated.status, vec![Some(1), Some(2)]);
        // "first": the second 1 is dropped before the current state is set.
        let first = surv2counting(
            &[1i64, 1, 1, 1],
            &[0.0, 1.0, 2.0, 3.0],
            &[Some(1), Some(2), Some(1), Some(2)],
            true,
            Repeated::First,
            &[],
        )
        .unwrap();
        assert_eq!(first.status, vec![Some(2), Some(0), Some(0)]);
        assert_eq!(first.istate, Some(vec![1, 2, 2]));
    }

    #[test]
    fn censored_first_rows_give_no_initial_state_and_right_censored_type() {
        let id = ["a", "a", "b", "b"];
        let time = [0.0, 4.0, 0.0, 6.0];
        let status = [Some(0), Some(1), Some(0), None];
        let out = surv2counting(&id, &time, &status, false, Repeated::No, &[]).unwrap();
        assert_eq!(out.istate, None);
        assert_eq!(out.status, vec![Some(1), None]);
        assert!(!out.counting);
        let mixed = surv2counting(
            &id,
            &time,
            &[Some(1), Some(1), Some(0), Some(1)],
            true,
            Repeated::No,
            &[],
        );
        assert!(mixed.is_err());
    }

    #[test]
    fn missing_covariates_are_carried_forward_within_subject() {
        let id = [1i64, 1, 1, 2, 2];
        let time = [0.0, 1.0, 2.0, 0.0, 1.0];
        let status = [Some(1), Some(0), Some(2), Some(1), Some(2)];
        let missing = vec![vec![false, true, true, true, false]];
        let out = surv2counting(&id, &time, &status, true, Repeated::No, &missing).unwrap();
        assert_eq!(out.row, vec![0, 1, 3]);
        assert_eq!(out.carry_from, vec![vec![0, 0, 3]]);
        assert_eq!(out.istate, Some(vec![1, 1, 1]));
    }

    #[test]
    fn rejects_duplicate_times_and_bad_shapes() {
        assert!(
            surv2counting(
                &[1i64, 1],
                &[0.0, 0.0],
                &[Some(1), Some(1)],
                true,
                Repeated::No,
                &[]
            )
            .is_err()
        );
        assert!(surv2counting(&[1i64], &[], &[Some(1)], true, Repeated::No, &[]).is_err());
        assert!(surv2counting(&[1i64], &[0.0], &[Some(1)], true, Repeated::No, &[vec![]]).is_err());
        assert!(surv2counting(&[1i64], &[0.0], &[Some(-1)], true, Repeated::No, &[]).is_err());
        assert!(Repeated::parse("maybe").is_err());
        assert_eq!(Repeated::parse("FIRST").unwrap(), Repeated::First);
    }
}
