//! O'Brien's logit-rank transformation of a survival data set.
//!
//! Port of R survival `R/survobrien.R`: the data set is expanded into one
//! block per event time containing everybody at risk, and within each block
//! every continuous covariate is replaced by the logit of its (mid-)rank
//! percentile.  A Cox model on the expanded data, stratified on the block,
//! gives O'Brien's test.  Formula handling (which terms are continuous,
//! keeper columns, cluster terms) belongs to the caller: it passes the
//! continuous columns and copies its keeper columns with [`SurvObrienExpansion::row`].

use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::{validate_binary_i32, validate_finite, validate_length};
use pyo3::prelude::*;

/// Inputs of [`survobrien`].
#[derive(Debug, Clone)]
pub struct SurvObrienInput<'a> {
    /// Start times of (start, stop] data; `None` for right-censored data.
    pub start: Option<&'a [f64]>,
    pub time: &'a [f64],
    pub status: &'a [i32],
    /// Strata codes; risk sets are formed within a stratum.
    pub strata: Option<&'a [i32]>,
    /// The continuous covariates to transform, one column each.
    pub continuous: &'a [Vec<f64>],
}

/// The expanded data set (R's returned data frame).
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object, get_all)]
pub struct SurvObrienExpansion {
    /// Original (0-based) row of each expanded row; R's `.id.` is `row + 1`.
    pub row: Vec<usize>,
    pub start: Option<Vec<f64>>,
    pub time: Vec<f64>,
    /// 1 for the event(s) defining the block, 0 for the rest of the risk set.
    pub status: Vec<i32>,
    /// R's `.strata.`: 1-based index of the risk set.
    pub strata: Vec<usize>,
    /// The transformed continuous columns, in input order.
    pub transformed: Vec<Vec<f64>>,
    /// The event time defining each risk set, in block order.
    pub event_times: Vec<f64>,
}

/// O'Brien's default transform: logits of the mid-rank percentiles.
fn logit_rank_transform(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut ranks = vec![0.0; n];
    let mut start = 0;
    while start < n {
        let mut end = start + 1;
        while end < n && values[order[end]] == values[order[start]] {
            end += 1;
        }
        let average_rank = ((start + 1) + end) as f64 / 2.0;
        for &idx in &order[start..end] {
            ranks[idx] = average_rank;
        }
        start = end;
    }
    ranks
        .iter()
        .map(|&rank| {
            let percentile = (rank - 0.5) / n as f64;
            (percentile / (1.0 - percentile)).ln()
        })
        .collect()
}

fn validate(input: &SurvObrienInput<'_>) -> SurvivalResult<()> {
    let n = input.time.len();
    if n == 0 {
        return Err(SurvivalError::invalid_input(
            "No (non-missing) observations",
        ));
    }
    validate_length(n, input.status.len(), "status")?;
    validate_finite(input.time, "time")?;
    validate_binary_i32(input.status, "status")?;
    if let Some(start) = input.start {
        validate_length(n, start.len(), "start")?;
        validate_finite(start, "start")?;
        if let Some(index) = (0..n).find(|&i| start[i] >= input.time[i]) {
            return Err(SurvivalError::invalid_input(format!(
                "Stop time must be > start time (row {index})"
            )));
        }
    }
    if let Some(strata) = input.strata {
        validate_length(n, strata.len(), "strata")?;
    }
    if input.continuous.is_empty() {
        return Err(SurvivalError::invalid_input(
            "No continuous variables to modify",
        ));
    }
    for (column, values) in input.continuous.iter().enumerate() {
        validate_length(n, values.len(), &format!("continuous[{column}]"))?;
        validate_finite(values, "continuous")?;
    }
    Ok(())
}

/// The risk sets: one `(event time, stratum, rows at risk)` per distinct
/// event time (per stratum), in R's order.
fn risk_sets(input: &SurvObrienInput<'_>) -> Vec<(f64, Vec<usize>)> {
    let n = input.time.len();
    let at_risk = |i: usize, at: f64| match input.start {
        Some(start) => start[i] < at && input.time[i] >= at,
        None => input.time[i] >= at,
    };
    match input.strata {
        None => {
            // etime <- sort(unique(y[event, time]))
            let mut event_times: Vec<f64> = (0..n)
                .filter(|&i| input.status[i] == 1)
                .map(|i| input.time[i])
                .collect();
            event_times.sort_by(f64::total_cmp);
            event_times.dedup();
            event_times
                .into_iter()
                .map(|at| (at, (0..n).filter(|&i| at_risk(i, at)).collect()))
                .collect()
        }
        Some(strata) => {
            // unique(data.frame(time, strata)[event, ]): first-appearance
            // order of the (time, stratum) pairs among the events.  R's own
            // stratified branches compare the status column with the time
            // (right-censored data) and select the *other* strata for
            // (start, stop] data; both are typos, the intent — everybody
            // at risk in the same stratum — is implemented here.
            let mut pairs: Vec<(f64, i32)> = Vec::new();
            for i in (0..n).filter(|&i| input.status[i] == 1) {
                let pair = (input.time[i], strata[i]);
                if !pairs.contains(&pair) {
                    pairs.push(pair);
                }
            }
            pairs
                .into_iter()
                .map(|(at, stratum)| {
                    (
                        at,
                        (0..n)
                            .filter(|&i| at_risk(i, at) && strata[i] == stratum)
                            .collect(),
                    )
                })
                .collect()
        }
    }
}

/// Expand the data into risk sets with the logit-rank transform applied
/// to every continuous covariate within each risk set.
pub fn survobrien(input: &SurvObrienInput<'_>) -> SurvivalResult<SurvObrienExpansion> {
    validate(input)?;
    let sets = risk_sets(input);
    let total: usize = sets.iter().map(|(_, rows)| rows.len()).sum();
    let mut row = Vec::with_capacity(total);
    let mut time = Vec::with_capacity(total);
    let mut status = Vec::with_capacity(total);
    let mut strata = Vec::with_capacity(total);
    let mut start = input.start.map(|_| Vec::with_capacity(total));
    let mut transformed: Vec<Vec<f64>> = input
        .continuous
        .iter()
        .map(|_| Vec::with_capacity(total))
        .collect();
    let mut event_times = Vec::with_capacity(sets.len());
    for (set_index, (at, rows)) in sets.iter().enumerate() {
        event_times.push(*at);
        for &i in rows {
            row.push(i);
            time.push(input.time[i]);
            status.push(i32::from(input.time[i] == *at && input.status[i] == 1));
            strata.push(set_index + 1);
            if let (Some(start_values), Some(out)) = (input.start, start.as_mut()) {
                out.push(start_values[i]);
            }
        }
        for (column, values) in input.continuous.iter().enumerate() {
            let block: Vec<f64> = rows.iter().map(|&i| values[i]).collect();
            transformed[column].extend(logit_rank_transform(&block));
        }
    }
    Ok(SurvObrienExpansion {
        row,
        start,
        time,
        status,
        strata,
        transformed,
        event_times,
    })
}

/// Python entry point: `survobrien(time, status, continuous, start=None,
/// strata=None)`; `continuous` is a list of columns.
#[pyfunction(name = "survobrien")]
#[pyo3(signature = (time, status, continuous, start=None, strata=None))]
pub fn survobrien_py(
    time: Vec<f64>,
    status: Vec<i32>,
    continuous: Vec<Vec<f64>>,
    start: Option<Vec<f64>>,
    strata: Option<Vec<i32>>,
) -> PyResult<SurvObrienExpansion> {
    Ok(survobrien(&SurvObrienInput {
        start: start.as_deref(),
        time: &time,
        status: &status,
        strata: strata.as_deref(),
        continuous: &continuous,
    })?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_risk_sets_and_transforms_within_each() {
        let result = survobrien(&SurvObrienInput {
            start: None,
            time: &[1.0, 2.0, 2.0, 3.0],
            status: &[1, 1, 0, 1],
            strata: None,
            continuous: &[vec![4.0, 1.0, 1.0, 3.0]],
        })
        .unwrap();
        assert_eq!(result.event_times, vec![1.0, 2.0, 3.0]);
        assert_eq!(result.row, vec![0, 1, 2, 3, 1, 2, 3, 3]);
        assert_eq!(result.status, vec![1, 0, 0, 0, 1, 0, 0, 1]);
        assert_eq!(result.strata, vec![1, 1, 1, 1, 2, 2, 2, 3]);
        // first block: values 4, 1, 1, 3 -> ranks 4, 1.5, 1.5, 3 over n = 4
        let logit = |rank: f64| {
            let p = (rank - 0.5) / 4.0;
            (p / (1.0 - p)).ln()
        };
        assert!((result.transformed[0][0] - logit(4.0)).abs() < 1e-12);
        assert!((result.transformed[0][1] - logit(1.5)).abs() < 1e-12);
        assert!((result.transformed[0][2] - logit(1.5)).abs() < 1e-12);
        // a block of one has percentile 0.5 -> logit 0
        assert!(result.transformed[0][7].abs() < 1e-12);
    }

    #[test]
    fn counting_process_risk_sets_use_the_open_interval() {
        let result = survobrien(&SurvObrienInput {
            start: Some(&[0.0, 1.0, 0.0]),
            time: &[2.0, 3.0, 1.0],
            status: &[1, 1, 1],
            strata: None,
            continuous: &[vec![1.0, 2.0, 3.0]],
        })
        .unwrap();
        // event at 1: rows with start < 1 <= stop -> rows 0 and 2
        assert_eq!(result.event_times, vec![1.0, 2.0, 3.0]);
        assert_eq!(result.row, vec![0, 2, 0, 1, 1]);
        assert_eq!(
            result.start.as_deref(),
            Some(&[0.0, 0.0, 0.0, 1.0, 1.0][..])
        );
    }

    #[test]
    fn strata_keep_risk_sets_within_a_stratum() {
        let result = survobrien(&SurvObrienInput {
            start: None,
            time: &[1.0, 2.0, 1.0, 2.0],
            status: &[1, 0, 1, 1],
            strata: Some(&[1, 1, 2, 2]),
            continuous: &[vec![1.0, 2.0, 3.0, 4.0]],
        })
        .unwrap();
        assert_eq!(result.event_times, vec![1.0, 1.0, 2.0]);
        assert_eq!(result.row, vec![0, 1, 2, 3, 3]);
    }

    #[test]
    fn inputs_are_validated() {
        assert!(
            survobrien(&SurvObrienInput {
                start: None,
                time: &[1.0],
                status: &[1],
                strata: None,
                continuous: &[],
            })
            .is_err()
        );
        assert!(
            survobrien(&SurvObrienInput {
                start: Some(&[2.0]),
                time: &[1.0],
                status: &[1],
                strata: None,
                continuous: &[vec![1.0]],
            })
            .is_err()
        );
    }
}
