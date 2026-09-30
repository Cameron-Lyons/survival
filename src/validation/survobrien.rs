//! O'Brien's logit-rank expansion, with risk sets restricted to each stratum.
//!
//! Formula evaluation and keeper columns belong to the caller. Events retain
//! R's ordering: chronological without strata, first appearance with strata.
//! A sweep maintains active rows and ordered covariates within each stratum;
//! output is written directly into precomputed block slices.

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::core::strata_order::validate_intervals;
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::numpy_utils::{FloatVec, IntVec};
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
    /// Continuous columns. NaN stays missing; infinities participate in ranks.
    /// May be empty for [`survobrien_expand`], which computes only risk sets.
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
    /// Transformed columns; empty for [`survobrien_expand`].
    pub transformed: Vec<Vec<f64>>,
    /// The event time defining each risk set, in block order.
    pub event_times: Vec<f64>,
    /// Half-open output slices: block i occupies `offsets[i]..offsets[i + 1]`.
    pub block_offsets: Vec<usize>,
}

#[cfg(feature = "python")]
#[pymethods]
impl SurvObrienExpansion {
    /// Owned NumPy snapshots, avoiding per-element conversion in an R bridge.
    fn to_arrays<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        use numpy::IntoPyArray;
        let result = pyo3::types::PyDict::new(py);
        let indices = |values: &[usize]| {
            values
                .iter()
                .map(|&v| v as i64)
                .collect::<Vec<_>>()
                .into_pyarray(py)
        };
        result.set_item("row", indices(&self.row))?;
        result.set_item(
            "start",
            self.start.as_ref().map(|v| v.clone().into_pyarray(py)),
        )?;
        result.set_item("time", self.time.clone().into_pyarray(py))?;
        result.set_item("status", self.status.clone().into_pyarray(py))?;
        result.set_item("strata", indices(&self.strata))?;
        result.set_item("event_times", self.event_times.clone().into_pyarray(py))?;
        result.set_item("block_offsets", indices(&self.block_offsets))?;
        result.set_item(
            "transformed",
            self.transformed
                .iter()
                .map(|v| v.clone().into_pyarray(py))
                .collect::<Vec<_>>(),
        )?;
        Ok(result)
    }
}

fn validate(input: &SurvObrienInput<'_>, transform: bool) -> SurvivalResult<()> {
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
        validate_intervals(start, input.time)?;
    }
    if let Some(strata) = input.strata {
        validate_length(n, strata.len(), "strata")?;
    }
    if transform && input.continuous.is_empty() {
        return Err(SurvivalError::invalid_input(
            "No continuous variables to modify",
        ));
    }
    for (column, values) in input.continuous.iter().enumerate() {
        validate_length(n, values.len(), &format!("continuous[{column}]"))?;
    }
    Ok(())
}

struct Group {
    /// Global row numbers, in original order.
    rows: Vec<usize>,
    /// Local row indices, ordered by stop and start respectively.
    stops: Vec<usize>,
    starts: Vec<usize>,
    /// Output block numbers, ordered by event time for the sweep.
    events: Vec<usize>,
}

fn groups(input: &SurvObrienInput<'_>) -> (Vec<Group>, Vec<f64>) {
    let mut groups: Vec<Group> = Vec::new();
    let mut codes = HashMap::new();
    let mut seen = HashSet::new();
    let mut events = Vec::new();
    for row in 0..input.time.len() {
        let code = input.strata.map_or(0, |s| s[row]);
        let next = groups.len();
        let group = *codes.entry(code).or_insert_with(|| {
            groups.push(Group {
                rows: Vec::new(),
                stops: Vec::new(),
                starts: Vec::new(),
                events: Vec::new(),
            });
            next
        });
        groups[group].rows.push(row);
        let at = input.time[row];
        // IEEE -0 and +0 denote the same event time, as in R's unique().
        let bits = if at == 0.0 { 0 } else { at.to_bits() };
        if input.status[row] == 1 && seen.insert((code, bits)) {
            groups[group].events.push(events.len());
            events.push(at);
        }
    }
    if input.strata.is_none() {
        events.sort_by(f64::total_cmp);
        groups[0].events = (0..events.len()).collect();
    }
    for group in &mut groups {
        group
            .events
            .sort_by(|&i, &j| events[i].total_cmp(&events[j]));
        group.stops = (0..group.rows.len()).collect();
        group
            .stops
            .sort_by(|&i, &j| input.time[group.rows[i]].total_cmp(&input.time[group.rows[j]]));
        if let Some(start) = input.start {
            group.starts = (0..group.rows.len()).collect();
            group
                .starts
                .sort_by(|&i, &j| start[group.rows[i]].total_cmp(&start[group.rows[j]]));
        }
    }
    (groups, events)
}

struct OrderedColumn {
    /// Local rows sorted by value; missing values have no position.
    rows: Vec<usize>,
    position: Vec<usize>,
    active: BTreeSet<usize>,
}

impl OrderedColumn {
    fn new(values: &[f64], rows: &[usize]) -> Self {
        let mut order: Vec<usize> = (0..rows.len())
            .filter(|&i| !values[rows[i]].is_nan())
            .collect();
        order.sort_by(|&i, &j| values[rows[i]].total_cmp(&values[rows[j]]));
        let mut position = vec![usize::MAX; rows.len()];
        for (rank, &row) in order.iter().enumerate() {
            position[row] = rank;
        }
        Self {
            rows: order,
            position,
            active: BTreeSet::new(),
        }
    }

    fn enter(&mut self, row: usize) {
        let rank = self.position[row];
        if rank != usize::MAX {
            self.active.insert(rank);
        }
    }

    fn leave(&mut self, row: usize) {
        self.active.remove(&self.position[row]);
    }

    fn transform(
        &self,
        values: &[f64],
        rows: &[usize],
        output_rows: &[usize],
        output: &mut [f64],
        tied: &mut Vec<usize>,
    ) {
        let mut ordered = self.active.iter().peekable();
        let mut rank = 1;
        let n = self.active.len() as f64;
        while let Some(&position) = ordered.next() {
            let first = rank;
            let local = self.rows[position];
            let value = values[rows[local]];
            tied.clear();
            tied.push(output_rows[local]);
            rank += 1;
            while let Some(&&next) = ordered.peek() {
                let local = self.rows[next];
                if values[rows[local]] != value {
                    break;
                }
                tied.push(output_rows[local]);
                ordered.next();
                rank += 1;
            }
            let percentile = ((first + rank - 1) as f64 / 2.0 - 0.5) / n;
            let logit = (percentile / (1.0 - percentile)).ln();
            for &row in tied.iter() {
                output[row] = logit;
            }
        }
    }
}

/// Expand risk sets and apply the logit of each continuous column's mid-rank
/// percentile. Missing values are retained and excluded from the denominator.
pub fn survobrien(input: &SurvObrienInput<'_>) -> SurvivalResult<SurvObrienExpansion> {
    expand(input, true)
}

/// Expand only the risk sets, for callers supplying their own transformation.
/// No covariate sorting or rank work is performed; `continuous` may be empty.
pub fn survobrien_expand(input: &SurvObrienInput<'_>) -> SurvivalResult<SurvObrienExpansion> {
    expand(input, false)
}

fn expand(input: &SurvObrienInput<'_>, transform: bool) -> SurvivalResult<SurvObrienExpansion> {
    validate(input, transform)?;
    let (groups, event_times) = groups(input);
    let mut counts = vec![0; event_times.len()];
    // Determine output slices without materializing any risk-set row lists.
    for group in &groups {
        let mut entered = if input.start.is_none() {
            group.rows.len()
        } else {
            0
        };
        let mut exited = 0;
        for &block in &group.events {
            let at = event_times[block];
            if let Some(start) = input.start {
                while entered < group.starts.len() && start[group.rows[group.starts[entered]]] < at
                {
                    entered += 1;
                }
            }
            while exited < group.stops.len() && input.time[group.rows[group.stops[exited]]] < at {
                exited += 1;
            }
            counts[block] = entered - exited;
        }
    }
    let mut offsets = Vec::with_capacity(counts.len() + 1);
    offsets.push(0usize);
    for count in counts {
        let next = offsets.last().unwrap().checked_add(count).ok_or_else(|| {
            SurvivalError::invalid_input("expanded row count exceeds addressable memory")
        })?;
        offsets.push(next);
    }
    let total = *offsets.last().unwrap();
    let mut output = SurvObrienExpansion {
        row: vec![0; total],
        start: input.start.map(|_| vec![0.0; total]),
        time: vec![0.0; total],
        status: vec![0; total],
        strata: vec![0; total],
        transformed: if transform {
            vec![vec![f64::NAN; total]; input.continuous.len()]
        } else {
            Vec::new()
        },
        event_times,
        block_offsets: offsets,
    };
    for group in groups {
        if group.events.is_empty() {
            continue;
        }
        let mut columns: Vec<OrderedColumn> = if transform {
            input
                .continuous
                .iter()
                .map(|values| OrderedColumn::new(values, &group.rows))
                .collect()
        } else {
            Vec::new()
        };
        let mut active = BTreeSet::new();
        let mut entered = 0;
        let mut exited = 0;
        let mut output_rows = vec![0; group.rows.len()];
        let mut tied = Vec::new();
        for block in group.events {
            let at = output.event_times[block];
            while entered < group.rows.len() {
                let local = if input.start.is_some() {
                    group.starts[entered]
                } else {
                    entered
                };
                if input
                    .start
                    .is_some_and(|start| start[group.rows[local]] >= at)
                {
                    break;
                }
                active.insert(local);
                for column in &mut columns {
                    column.enter(local);
                }
                entered += 1;
            }
            while exited < group.stops.len() && input.time[group.rows[group.stops[exited]]] < at {
                let local = group.stops[exited];
                active.remove(&local);
                for column in &mut columns {
                    column.leave(local);
                }
                exited += 1;
            }
            for (index, &local) in active.iter().enumerate() {
                let position = output.block_offsets[block] + index;
                let row = group.rows[local];
                output_rows[local] = position;
                output.row[position] = row;
                output.time[position] = input.time[row];
                output.status[position] =
                    i32::from(input.time[row] == at && input.status[row] == 1);
                output.strata[position] = block + 1;
                if let (Some(values), Some(start)) = (input.start, &mut output.start) {
                    start[position] = values[row];
                }
            }
            for (index, column) in columns.iter().enumerate() {
                column.transform(
                    &input.continuous[index],
                    &group.rows,
                    &output_rows,
                    &mut output.transformed[index],
                    &mut tied,
                );
            }
        }
    }
    Ok(output)
}

/// Python entry point. `continuous` is a list of columns; `transform=False`
/// skips ranking for custom transforms. Input conversion occurs once and the
/// expansion runs without the GIL.
#[pyfunction(name = "survobrien")]
#[pyo3(signature = (time, status, continuous, start=None, strata=None, transform=true))]
pub fn survobrien_py(
    py: Python<'_>,
    time: FloatVec,
    status: IntVec,
    continuous: Vec<FloatVec>,
    start: Option<FloatVec>,
    strata: Option<IntVec>,
    transform: bool,
) -> PyResult<SurvObrienExpansion> {
    let continuous: Vec<_> = continuous.into_iter().map(FloatVec::into_inner).collect();
    let input = SurvObrienInput {
        start: start.as_deref(),
        time: &time,
        status: &status,
        strata: strata.as_deref(),
        continuous: &continuous,
    };
    Ok(py.detach(|| expand(&input, transform))?)
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

    #[test]
    fn missing_values_and_infinities_keep_ranks_within_each_event_block() {
        let input = SurvObrienInput {
            start: None,
            time: &[3.0, 1.0, 2.0, 3.0],
            status: &[1, 1, 0, 1],
            strata: Some(&[1, 2, 1, 2]),
            continuous: &[vec![
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::INFINITY,
            ]],
        };
        let result = survobrien(&input).unwrap();
        assert_eq!(result.event_times, vec![3.0, 1.0, 3.0]);
        assert_eq!(result.row, vec![0, 1, 3, 3]);
        assert_eq!(result.block_offsets, vec![0, 1, 3, 4]);
        assert!(result.transformed[0][0].is_nan());
        assert_eq!(&result.transformed[0][1..], &[0.0, 0.0, 0.0]);
        let mut geometry = input;
        geometry.continuous = &[];
        let expanded = survobrien_expand(&geometry).unwrap();
        assert_eq!(expanded.row, result.row);
        assert_eq!(expanded.block_offsets, result.block_offsets);
        assert!(expanded.transformed.is_empty());
    }

    #[test]
    fn no_events_return_empty_columns_and_one_offset() {
        let result = survobrien(&SurvObrienInput {
            start: Some(&[0.0, 1.0]),
            time: &[1.0, 2.0],
            status: &[0, 0],
            strata: None,
            continuous: &[vec![1.0, 2.0]],
        })
        .unwrap();
        assert_eq!(result.block_offsets, vec![0]);
        assert!(result.row.is_empty());
        assert_eq!(result.start, Some(vec![]));
        assert_eq!(result.transformed, vec![Vec::<f64>::new()]);
    }
}
