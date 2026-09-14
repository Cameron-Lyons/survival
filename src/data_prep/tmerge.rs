//! R's `tmerge` (`R/tmerge.R` and `src/tmerge.c`): add a time-dependent
//! covariate (`tdc`, `cumtdc`) or an event indicator (`event`, `cumevent`)
//! to a `(tstart, tstop]` data set, splitting intervals at the new time
//! points.
//!
//! One call of [`tmerge_step`] processes one `name = kind(time, value)`
//! argument of an R `tmerge` call against the current data set, the way
//! R's loop over its `...` arguments does; the caller applies the returned
//! row map and value sources to columns of any type.  The C kernels are
//! [`tmerge_cumulative`] (`tmerge`), [`tmerge_lookup`] (`tmerge2`) and
//! [`tmerge_carry_forward`] (`tmerge3`).

use super::id_value::{IdValue, SubjectId, first_appearance_codes};
use super::neardate::{NeardateBest, neardate};
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::{validate_finite, validate_length};
use pyo3::prelude::*;
use std::collections::BTreeMap;

/// The kind of a `tmerge` argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TmergeKind {
    /// Time-dependent covariate: the most recent value carried forward.
    Tdc,
    /// Cumulative time-dependent covariate: a running sum.
    Cumtdc,
    /// An event at the end of an interval.
    Event,
    /// A cumulative count of events.
    Cumevent,
}

impl TmergeKind {
    /// Parse the R keyword.
    pub fn parse(value: &str) -> SurvivalResult<Self> {
        match value {
            "tdc" => Ok(Self::Tdc),
            "cumtdc" => Ok(Self::Cumtdc),
            "event" => Ok(Self::Event),
            "cumevent" => Ok(Self::Cumevent),
            other => Err(SurvivalError::invalid_input(format!(
                "argument(s) {other} not a recognized type"
            ))),
        }
    }
}

/// The current `(id, tstart, tstop)` data set (R's `newdata`), with the
/// rows of each subject contiguous and ordered by time.
pub struct TmergeBase<'a, I> {
    pub id: &'a [I],
    pub start: &'a [f64],
    pub stop: &'a [f64],
}

/// The rows of `data2` behind one argument: `time` (`NaN` for `NA`) and,
/// for `cumtdc`/`cumevent` or numeric `tdc`/`event`, the numeric `value`
/// (`NaN` for `NA`).  Values of another type are described only by their
/// missingness in `missing`.
pub struct TmergeUpdate<'a, I> {
    pub id: &'a [I],
    pub time: &'a [f64],
    pub value: Option<&'a [f64]>,
    pub missing: Option<&'a [bool]>,
}

/// R's `tmerge.control` options that affect one step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TmergeOptions {
    /// Delay added to covariate change times after a subject's entry.
    pub delay: f64,
    /// Drop update rows with a missing value.
    pub na_rm: bool,
    /// Whether this is not the first call, so update rows whose id is
    /// absent from the data set are dropped and counted as `missid`.
    pub check_ids: bool,
}

impl Default for TmergeOptions {
    fn default() -> Self {
        Self {
            delay: 0.0,
            na_rm: true,
            check_ids: true,
        }
    }
}

/// R's `tcount` columns.
pub const TCOUNT_NAMES: [&str; 9] = [
    "early", "late", "gap", "within", "boundary", "leading", "trailing", "tied", "missid",
];

/// The outcome of one `tmerge` argument.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct TmergeStep {
    /// Zero-based row of the input data set behind each output row; rows
    /// split by an event time appear several times.
    #[pyo3(get)]
    pub row: Vec<usize>,
    #[pyo3(get)]
    pub start: Vec<f64>,
    #[pyo3(get)]
    pub stop: Vec<f64>,
    /// Output rows created as the earlier part of a split, whose existing
    /// event variables must be reset to their censoring value.
    #[pyo3(get)]
    pub censor_rows: Vec<usize>,
    /// R's `tcount` row for this argument, in [`TCOUNT_NAMES`] order.
    #[pyo3(get)]
    pub tcount: Vec<usize>,
    /// `tdc`: for each output row, the zero-based update row whose value
    /// applies (`None` when no value precedes the interval).
    #[pyo3(get)]
    pub source: Vec<Option<usize>>,
    /// `cumtdc`: the new variable.
    #[pyo3(get)]
    pub cumulative: Vec<f64>,
    /// `event`/`cumevent`: output rows receiving an event, the update row
    /// each came from, and its numeric value (`1` without values, the
    /// running count for `cumevent`).
    #[pyo3(get)]
    pub event_row: Vec<usize>,
    #[pyo3(get)]
    pub event_source: Vec<usize>,
    #[pyo3(get)]
    pub event_value: Vec<f64>,
}

/// `tmerge.c`: cumulative sum of `x` over the update rows of a subject
/// whose time is at or before each base interval's start.  Both data sets
/// are ordered by time within id, with ids as comparable codes; a base
/// row with no preceding update keeps `newx` (and `NaN` entries that do
/// get one are replaced rather than incremented).
pub fn tmerge_cumulative(
    id: &[usize],
    time1: &[f64],
    newx: &[f64],
    nid: &[usize],
    ntime: &[f64],
    x: &[f64],
) -> SurvivalResult<Vec<f64>> {
    validate_length(id.len(), time1.len(), "time1")?;
    validate_length(id.len(), newx.len(), "newx")?;
    validate_length(nid.len(), ntime.len(), "ntime")?;
    validate_length(nid.len(), x.len(), "x")?;
    let mut result = newx.to_vec();
    let mut k = 0;
    let mut previous: Option<usize> = None;
    let mut csum = 0.0;
    let mut has_one = false;
    for i in 0..id.len() {
        if previous != Some(id[i]) {
            csum = 0.0;
            previous = Some(id[i]);
            has_one = false;
        }
        while k < nid.len() && nid[k] < id[i] {
            k += 1;
        }
        while k < nid.len() && nid[k] == id[i] && ntime[k] <= time1[i] {
            csum += x[k];
            has_one = true;
            k += 1;
        }
        if has_one {
            result[i] = if result[i].is_nan() {
                csum
            } else {
                result[i] + csum
            };
        }
    }
    Ok(result)
}

/// `tmerge2`: for each base row, the last update row of the same subject
/// whose time is at or before the interval start (last value carried
/// forward), or `None`.  Both inputs are ordered by time within subject
/// code, as R's `tmerge` arranges them.
pub fn tmerge_lookup(
    id: &[usize],
    time1: &[f64],
    nid: &[usize],
    ntime: &[f64],
) -> SurvivalResult<Vec<Option<usize>>> {
    validate_length(id.len(), time1.len(), "time1")?;
    validate_length(nid.len(), ntime.len(), "ntime")?;
    let mut index = vec![None; id.len()];
    let mut k = 0;
    for i in 0..id.len() {
        while k < nid.len() && nid[k] < id[i] {
            k += 1;
        }
        while k < nid.len() && nid[k] == id[i] && ntime[k] <= time1[i] {
            index[i] = Some(k);
            k += 1;
        }
        // The C code ends every row with `k--`: the row that supplied this
        // interval's value is carried forward to the subject's later
        // intervals when no newer update precedes them.  Without a match
        // the scan stopped on a row that is still ahead of the next base
        // row, so it stays where it is.
        if let Some(last) = index[i] {
            k = last;
        }
    }
    Ok(index)
}

/// `tmerge3`: last value carried forward within subject; for each row,
/// the most recent non-missing row of the same subject (itself when not
/// missing), or `None` when the subject has had no value yet.  `id` must
/// be sorted.
pub fn tmerge_carry_forward(id: &[usize], missing: &[bool]) -> SurvivalResult<Vec<Option<usize>>> {
    validate_length(id.len(), missing.len(), "missing")?;
    let mut index = vec![None; id.len()];
    let mut last_good = None;
    let mut previous: Option<usize> = None;
    for i in 0..id.len() {
        if previous != Some(id[i]) {
            last_good = None;
            previous = Some(id[i]);
        }
        if missing[i] {
            index[i] = last_good;
        } else {
            index[i] = Some(i);
            last_good = Some(i);
        }
    }
    Ok(index)
}

/// R's `itype` categories of an update time relative to a subject's
/// intervals, with the boundary subtypes spelled out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Placement {
    Early,
    Late,
    Gap,
    Within,
    /// Touches the shared end of `(a, b]` and start of `(b, d]`.
    Boundary,
    /// Equals the start of an interval that follows a gap.
    Leading,
    /// Equals the end of an interval that precedes a gap.
    Trailing,
}

impl Placement {
    fn tcount_column(self) -> usize {
        match self {
            Self::Early => 0,
            Self::Late => 1,
            Self::Gap => 2,
            Self::Within => 3,
            Self::Boundary => 4,
            Self::Leading => 5,
            Self::Trailing => 6,
        }
    }

    /// R's `subtype == 1 | subtype == 3`: the rows that receive an event.
    fn receives_event(self) -> bool {
        matches!(self, Self::Boundary | Self::Trailing)
    }
}

/// Where every event time falls: `indx1 <- neardate(..., dstart, "prior")`,
/// `indx2 <- neardate(..., dstop, "after")` and the classification.
struct Classification {
    indx1: Vec<Option<usize>>,
    indx2: Vec<Option<usize>>,
    placement: Vec<Placement>,
}

fn classify(
    base_keys: &[usize],
    start: &[f64],
    stop: &[f64],
    event_keys: &[usize],
    etime: &[f64],
) -> SurvivalResult<Classification> {
    if etime.is_empty() {
        return Ok(Classification {
            indx1: Vec::new(),
            indx2: Vec::new(),
            placement: Vec::new(),
        });
    }
    let indx1 = neardate(event_keys, etime, base_keys, start, NeardateBest::Prior)?;
    let indx2 = neardate(event_keys, etime, base_keys, stop, NeardateBest::After)?;
    let placement = indx1
        .iter()
        .zip(&indx2)
        .zip(etime)
        .map(|((&i1, &i2), &t)| match (i1, i2) {
            (None, _) => Placement::Early,
            (_, None) => Placement::Late,
            (Some(i1), Some(i2)) if i2 > i1 => Placement::Gap,
            (Some(i1), Some(i2)) if t == start[i1] || t == stop[i2] => {
                if i1 == i2 + 1 {
                    Placement::Boundary
                } else if t == start[i1] {
                    Placement::Leading
                } else {
                    Placement::Trailing
                }
            }
            _ => Placement::Within,
        })
        .collect();
    Ok(Classification {
        indx1,
        indx2,
        placement,
    })
}

/// Process one `tmerge` argument.
pub fn tmerge_step<I: SubjectId>(
    base: &TmergeBase<'_, I>,
    update: &TmergeUpdate<'_, I>,
    kind: TmergeKind,
    options: &TmergeOptions,
    prior: Option<&[f64]>,
    default: f64,
) -> SurvivalResult<TmergeStep> {
    let n = base.id.len();
    validate_length(n, base.start.len(), "tstart")?;
    validate_length(n, base.stop.len(), "tstop")?;
    validate_finite(base.start, "tstart")?;
    validate_finite(base.stop, "tstop")?;
    let n2 = update.id.len();
    validate_length(n2, update.time.len(), "time")?;
    if let Some(value) = update.value {
        validate_length(n2, value.len(), "value")?;
    }
    if let Some(missing) = update.missing {
        validate_length(n2, missing.len(), "missing")?;
    }
    if let Some(prior) = prior {
        validate_length(n, prior.len(), "prior")?;
    }
    if options.delay.is_nan() || options.delay < 0.0 {
        return Err(SurvivalError::invalid_input(
            "delay option must be a number >= 0",
        ));
    }
    if matches!(kind, TmergeKind::Cumtdc | TmergeKind::Cumevent)
        && update
            .value
            .is_some_and(|v| v.iter().any(|x| x.is_infinite()))
    {
        return Err(SurvivalError::invalid_input(
            "invalid increment for cumtdc or cumevent",
        ));
    }

    // Subjects as codes in order of first appearance in the data set.
    let (base_codes, representatives) = first_appearance_codes(base.id);
    let base_position: std::collections::HashMap<I::Key, usize> = representatives
        .iter()
        .enumerate()
        .map(|(code, id)| (id.key(), code))
        .collect();

    // Which update rows are usable, and how many were dropped for an
    // unknown id.
    let mut missid = 0;
    let mut kept: Vec<(usize, usize)> = Vec::with_capacity(n2); // (subject code, update row)
    for row in 0..n2 {
        let Some(&code) = base_position.get(&update.id[row].key()) else {
            if options.check_ids {
                missid += 1;
                continue;
            }
            return Err(SurvivalError::invalid_input(
                "setting the range, and data2 has id values not in data1",
            ));
        };
        if update.time[row].is_nan() {
            continue;
        }
        let value_missing =
            update.value.is_some_and(|v| v[row].is_nan()) || update.missing.is_some_and(|m| m[row]);
        if options.na_rm && value_missing {
            continue;
        }
        kept.push((code, row));
    }
    // order(match(id, baseid), etime), a stable sort.
    kept.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| update.time[a.1].total_cmp(&update.time[b.1]))
    });
    let event_codes: Vec<usize> = kept.iter().map(|(code, _)| *code).collect();
    let event_rows: Vec<usize> = kept.iter().map(|(_, row)| *row).collect();
    let mut etime: Vec<f64> = event_rows.iter().map(|&row| update.time[row]).collect();

    if options.delay > 0.0 && matches!(kind, TmergeKind::Tdc | TmergeKind::Cumtdc) {
        let mut min_start = vec![f64::INFINITY; representatives.len()];
        for (i, &code) in base_codes.iter().enumerate() {
            min_start[code] = min_start[code].min(base.start[i]);
        }
        for (t, &code) in etime.iter_mut().zip(&event_codes) {
            if *t > min_start[code] {
                *t += options.delay;
            }
        }
    }

    let Classification {
        indx1,
        mut indx2,
        mut placement,
    } = classify(&base_codes, base.start, base.stop, &event_codes, &etime)?;

    let mut tcount = vec![0; 9];
    for p in &placement {
        tcount[p.tcount_column()] += 1;
    }
    // Ties: duplicated times within subject.
    tcount[7] = event_codes
        .windows(2)
        .zip(etime.windows(2))
        .filter(|(codes, times)| codes[0] == codes[1] && times[0] == times[1])
        .count();
    tcount[8] = missid;

    // Split the intervals that contain an event time strictly inside.
    let mut cuts: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    for (k, p) in placement.iter().enumerate() {
        if *p == Placement::Within {
            let row = indx1[k].expect("within events have a prior interval");
            let times = cuts.entry(row).or_default();
            if times.last() != Some(&etime[k]) {
                times.push(etime[k]);
            }
        }
    }
    let mut row = Vec::with_capacity(n);
    let mut start = Vec::with_capacity(n);
    let mut stop = Vec::with_capacity(n);
    let mut censor_rows = Vec::new();
    for i in 0..n {
        let mut left = base.start[i];
        if let Some(times) = cuts.get(&i) {
            for &t in times {
                censor_rows.push(row.len());
                row.push(i);
                start.push(left);
                stop.push(t);
                left = t;
            }
        }
        row.push(i);
        start.push(left);
        stop.push(base.stop[i]);
    }
    let out_codes: Vec<usize> = row.iter().map(|&i| base_codes[i]).collect();
    if !cuts.is_empty() {
        let refreshed = classify(&out_codes, &start, &stop, &event_codes, &etime)?;
        indx2 = refreshed.indx2;
        placement = refreshed.placement;
        for p in placement.iter_mut() {
            if *p == Placement::Within {
                *p = Placement::Boundary;
            }
        }
    }
    let n_out = row.len();

    let mut step = TmergeStep {
        row,
        start,
        stop,
        censor_rows,
        tcount,
        source: Vec::new(),
        cumulative: Vec::new(),
        event_row: Vec::new(),
        event_source: Vec::new(),
        event_value: Vec::new(),
    };
    match kind {
        TmergeKind::Tdc => {
            let index = tmerge_lookup(&out_codes, &step.start, &event_codes, &etime)?;
            step.source = index.iter().map(|k| k.map(|k| event_rows[k])).collect();
        }
        TmergeKind::Cumtdc => {
            // Changes after the last interval are ignored.
            let keep: Vec<usize> = (0..etime.len())
                .filter(|&k| placement[k] != Placement::Late)
                .collect();
            let nid: Vec<usize> = keep.iter().map(|&k| event_codes[k]).collect();
            let ntime: Vec<f64> = keep.iter().map(|&k| etime[k]).collect();
            let increments: Vec<f64> = keep
                .iter()
                .map(|&k| update.value.map_or(1.0, |v| v[event_rows[k]]))
                .collect();
            let newvar: Vec<f64> = match prior {
                Some(prior) => step.row.iter().map(|&i| prior[i]).collect(),
                None if update.value.is_none() => vec![0.0; n_out],
                None => vec![default; n_out],
            };
            step.cumulative =
                tmerge_cumulative(&out_codes, &step.start, &newvar, &nid, &ntime, &increments)?;
        }
        TmergeKind::Event | TmergeKind::Cumevent => {
            let mut values: Vec<f64> = event_rows
                .iter()
                .map(|&row| update.value.map_or(1.0, |v| v[row]))
                .collect();
            let mut ykeep = vec![true; values.len()];
            if kind == TmergeKind::Cumevent {
                let mut running = vec![0.0; representatives.len()];
                for (k, value) in values.iter_mut().enumerate() {
                    ykeep[k] = *value != 0.0;
                    running[event_codes[k]] += *value;
                    *value = running[event_codes[k]];
                }
            }
            for k in 0..values.len() {
                if placement[k].receives_event() && ykeep[k] {
                    step.event_row
                        .push(indx2[k].expect("boundary events have an interval"));
                    step.event_source.push(event_rows[k]);
                    step.event_value.push(values[k]);
                }
            }
        }
    }
    Ok(step)
}

/// Python entry point of [`tmerge_step`].  `prior` is the existing numeric
/// variable for `cumtdc` and `default` its value for rows without an
/// update (`NaN` for `NA`).
#[pyfunction(name = "tmerge_step")]
#[pyo3(signature = (id, start, stop, update_id, update_time, kind, value=None, missing=None, prior=None, default=f64::NAN, delay=0.0, na_rm=true, check_ids=true))]
#[allow(clippy::too_many_arguments)]
pub fn tmerge_step_py(
    id: Vec<IdValue>,
    start: Vec<f64>,
    stop: Vec<f64>,
    update_id: Vec<IdValue>,
    update_time: Vec<f64>,
    kind: &str,
    value: Option<Vec<f64>>,
    missing: Option<Vec<bool>>,
    prior: Option<Vec<f64>>,
    default: f64,
    delay: f64,
    na_rm: bool,
    check_ids: bool,
) -> PyResult<TmergeStep> {
    if id.iter().chain(&update_id).any(IdValue::is_missing) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "id variable cannot have missing values",
        ));
    }
    let base = TmergeBase {
        id: &id,
        start: &start,
        stop: &stop,
    };
    let update = TmergeUpdate {
        id: &update_id,
        time: &update_time,
        value: value.as_deref(),
        missing: missing.as_deref(),
    };
    let options = TmergeOptions {
        delay,
        na_rm,
        check_ids,
    };
    Ok(tmerge_step(
        &base,
        &update,
        TmergeKind::parse(kind)?,
        &options,
        prior.as_deref(),
        default,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(
        base: (&[i64], &[f64], &[f64]),
        update: (&[i64], &[f64], Option<&[f64]>),
        kind: TmergeKind,
    ) -> TmergeStep {
        tmerge_step(
            &TmergeBase {
                id: base.0,
                start: base.1,
                stop: base.2,
            },
            &TmergeUpdate {
                id: update.0,
                time: update.1,
                value: update.2,
                missing: None,
            },
            kind,
            &TmergeOptions::default(),
            None,
            f64::NAN,
        )
        .unwrap()
    }

    #[test]
    fn kernels_follow_tmerge_c() {
        let sums = tmerge_cumulative(
            &[0, 0, 1, 1, 1],
            &[1.0, 3.0, 0.5, 1.5, 3.0],
            &[f64::NAN, 10.0, f64::NAN, 1.0, f64::NAN],
            &[0, 0, 1, 1, 1],
            &[0.5, 2.5, 0.25, 1.0, 2.0],
            &[2.0, 3.0, 5.0, 7.0, 11.0],
        )
        .unwrap();
        assert_eq!(sums, vec![2.0, 15.0, 5.0, 13.0, 23.0]);

        let index = tmerge_lookup(
            &[0, 0, 1, 1],
            &[1.0, 3.0, 0.5, 3.0],
            &[0, 0, 1, 1, 1],
            &[0.5, 2.5, 0.25, 1.0, 2.0],
        )
        .unwrap();
        assert_eq!(index, vec![Some(0), Some(1), Some(2), Some(4)]);
        assert_eq!(
            tmerge_lookup(&[0], &[0.0], &[1], &[1.0]).unwrap(),
            vec![None]
        );

        let carried =
            tmerge_carry_forward(&[0, 0, 1, 1, 1], &[false, true, true, false, true]).unwrap();
        assert_eq!(carried, vec![Some(0), Some(0), None, Some(3), Some(3)]);
        assert!(tmerge_carry_forward(&[0], &[]).is_err());
    }

    #[test]
    fn lookup_carries_the_last_value_forward() {
        // (0,5], (5,10], (10,15] with updates at -1 and 7: the first update
        // serves the first two intervals, the second the last (the tmerge2
        // comment's example, without the gap).
        assert_eq!(
            tmerge_lookup(&[0, 0, 0], &[0.0, 5.0, 10.0], &[0, 0], &[-1.0, 7.0]).unwrap(),
            vec![Some(0), Some(0), Some(1)]
        );
        // The tmerge2 comment: (0,5), (5,10), (15,20) and updates at
        // -1, 5, 11, 12 give rows 1, 2 and 4; the third is never used.
        assert_eq!(
            tmerge_lookup(
                &[0, 0, 0],
                &[0.0, 5.0, 15.0],
                &[0, 0, 0, 0],
                &[-1.0, 5.0, 11.0, 12.0]
            )
            .unwrap(),
            vec![Some(0), Some(1), Some(3)]
        );
        // A subject without updates, one whose only update is carried over
        // every interval, and one whose first interval precedes its update.
        assert_eq!(
            tmerge_lookup(
                &[0, 0, 1, 1, 1, 2, 2],
                &[0.0, 5.0, 0.0, 5.0, 10.0, 0.0, 5.0],
                &[1, 2],
                &[-1.0, 5.0]
            )
            .unwrap(),
            vec![None, None, Some(0), Some(0), Some(0), None, Some(1)]
        );

        // Through tmerge_step: an update at 7 splits (5,10] and the value
        // from time -1 covers every interval before it.
        let out = step(
            (
                &[1i64, 1, 1][..],
                &[0.0, 5.0, 10.0][..],
                &[5.0, 10.0, 15.0][..],
            ),
            (&[1i64, 1], &[-1.0, 7.0], Some(&[7.0, 8.0])),
            TmergeKind::Tdc,
        );
        assert_eq!(out.start, vec![0.0, 5.0, 7.0, 10.0]);
        assert_eq!(out.stop, vec![5.0, 7.0, 10.0, 15.0]);
        assert_eq!(out.source, vec![Some(0), Some(0), Some(1), Some(1)]);
        assert_eq!(out.tcount, vec![1, 0, 0, 1, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn events_split_intervals_and_land_on_their_end() {
        // The synthetic fixture: four subjects with one interval each.
        let base = (
            &[1i64, 2, 3, 4][..],
            &[0.0, 0.0, 0.0, 0.0][..],
            &[10.0, 20.0, 15.0, 30.0][..],
        );
        let long_id = [1i64, 1, 2, 2, 2, 3, 4, 4, 4, 4];
        let long_time = [2.0, 5.0, 3.0, 8.0, 21.0, 4.0, 5.0, 10.0, 15.0, 25.0];
        let lab = [1.1, 1.5, 0.9, 1.2, 1.8, 2.0, 0.5, 0.7, 1.4, 1.9];
        let out = step(base, (&long_id, &long_time, Some(&lab)), TmergeKind::Tdc);
        assert_eq!(out.row, vec![0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3]);
        assert_eq!(
            out.start,
            vec![
                0.0, 2.0, 5.0, 0.0, 3.0, 8.0, 0.0, 4.0, 0.0, 5.0, 10.0, 15.0, 25.0
            ]
        );
        assert_eq!(
            out.stop,
            vec![
                2.0, 5.0, 10.0, 3.0, 8.0, 20.0, 4.0, 15.0, 5.0, 10.0, 15.0, 25.0, 30.0
            ]
        );
        assert_eq!(out.censor_rows, vec![0, 1, 3, 4, 6, 8, 9, 10, 11]);
        assert_eq!(out.tcount, vec![0, 1, 0, 9, 0, 0, 0, 0, 0]);
        assert_eq!(
            out.source,
            vec![
                None,
                Some(0),
                Some(1),
                None,
                Some(2),
                Some(3),
                None,
                Some(5),
                None,
                Some(6),
                Some(7),
                Some(8),
                Some(9)
            ]
        );

        // cumtdc on the split data counts the changes.
        let split = (
            &[1i64, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4][..],
            &out.start[..],
            &out.stop[..],
        );
        let counts = step(split, (&long_id, &long_time, None), TmergeKind::Cumtdc);
        assert_eq!(counts.row.len(), 13);
        assert_eq!(
            counts.cumulative,
            vec![
                0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0
            ]
        );
        assert_eq!(counts.tcount, vec![0, 1, 0, 0, 9, 0, 0, 0, 0]);

        let infection = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0];
        let events = step(
            split,
            (&long_id, &long_time, Some(&infection)),
            TmergeKind::Event,
        );
        assert_eq!(events.event_row, vec![0, 1, 3, 4, 6, 8, 9, 10, 11]);
        assert_eq!(
            events.event_value,
            vec![1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
        );
        let cumulative = step(
            split,
            (&long_id, &long_time, Some(&infection)),
            TmergeKind::Cumevent,
        );
        assert_eq!(cumulative.event_row, vec![0, 3, 4, 6, 9, 10, 11]);
        assert_eq!(
            cumulative.event_value,
            vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn placements_are_counted_like_r() {
        let base = (&[1i64, 1][..], &[0.0, 7.0][..], &[5.0, 10.0][..]);
        let times = [-1.0, 0.0, 3.0, 5.0, 6.0, 7.0, 8.0, 8.0, 10.0, 11.0];
        let out = step(base, (&[1i64; 10], &times, None), TmergeKind::Event);
        // early, leading, within, trailing, gap, leading, within x2, trailing, late
        assert_eq!(out.tcount, vec![1, 1, 1, 3, 0, 2, 2, 1, 0]);
        assert_eq!(out.row, vec![0, 0, 1, 1]);
        assert_eq!(out.stop, vec![3.0, 5.0, 8.0, 10.0]);
        // Events land on trailing edges and (after splitting) boundaries.
        assert_eq!(out.event_row, vec![0, 1, 2, 2, 3]);

        let shared = (&[1i64, 1][..], &[0.0, 5.0][..], &[5.0, 10.0][..]);
        let out = step(shared, (&[1i64], &[5.0], None), TmergeKind::Event);
        assert_eq!(out.tcount, vec![0, 0, 0, 0, 1, 0, 0, 0, 0]);
        assert_eq!(out.event_row, vec![0]);
    }

    #[test]
    fn missing_ids_times_and_values_are_dropped_or_rejected() {
        let base = (&[1i64][..], &[0.0][..], &[10.0][..]);
        let out = tmerge_step(
            &TmergeBase {
                id: base.0,
                start: base.1,
                stop: base.2,
            },
            &TmergeUpdate {
                id: &[1, 2, 1, 1],
                time: &[3.0, 4.0, f64::NAN, 6.0],
                value: Some(&[1.0, 1.0, 1.0, f64::NAN]),
                missing: None,
            },
            TmergeKind::Tdc,
            &TmergeOptions::default(),
            None,
            f64::NAN,
        )
        .unwrap();
        assert_eq!(out.tcount[8], 1);
        assert_eq!(out.row, vec![0, 0]);
        assert_eq!(out.source, vec![None, Some(0)]);

        let first_call = tmerge_step(
            &TmergeBase {
                id: base.0,
                start: base.1,
                stop: base.2,
            },
            &TmergeUpdate {
                id: &[2],
                time: &[1.0],
                value: None,
                missing: None,
            },
            TmergeKind::Event,
            &TmergeOptions {
                check_ids: false,
                ..TmergeOptions::default()
            },
            None,
            f64::NAN,
        );
        assert!(first_call.is_err());
        assert!(TmergeKind::parse("cumtdc").is_ok());
        assert!(TmergeKind::parse("other").is_err());
    }

    #[test]
    fn delay_moves_covariate_changes_after_entry() {
        let base = (&[1i64][..], &[0.0][..], &[10.0][..]);
        let out = tmerge_step(
            &TmergeBase {
                id: base.0,
                start: base.1,
                stop: base.2,
            },
            &TmergeUpdate {
                id: &[1, 1],
                time: &[0.0, 4.0],
                value: Some(&[1.0, 2.0]),
                missing: None,
            },
            TmergeKind::Tdc,
            &TmergeOptions {
                delay: 3.0,
                ..TmergeOptions::default()
            },
            None,
            f64::NAN,
        )
        .unwrap();
        assert_eq!(out.stop, vec![7.0, 10.0]);
        assert_eq!(out.source, vec![Some(0), Some(1)]);
    }

    #[test]
    fn cumtdc_uses_prior_values_and_defaults() {
        let base = (&[1i64, 1][..], &[0.0, 5.0][..], &[5.0, 10.0][..]);
        let update = TmergeUpdate {
            id: &[1, 1],
            time: &[5.0, 12.0],
            value: Some(&[2.0, 100.0]),
            missing: None,
        };
        let with_prior = tmerge_step(
            &TmergeBase {
                id: base.0,
                start: base.1,
                stop: base.2,
            },
            &update,
            TmergeKind::Cumtdc,
            &TmergeOptions::default(),
            Some(&[1.0, f64::NAN]),
            f64::NAN,
        )
        .unwrap();
        // The late change is ignored; the NaN prior is replaced, not added.
        assert_eq!(with_prior.cumulative, vec![1.0, 2.0]);
        let with_default = tmerge_step(
            &TmergeBase {
                id: base.0,
                start: base.1,
                stop: base.2,
            },
            &update,
            TmergeKind::Cumtdc,
            &TmergeOptions::default(),
            None,
            -1.0,
        )
        .unwrap();
        // A non-missing default is incremented, as in tmerge.c.
        assert_eq!(with_default.cumulative, vec![-1.0, 1.0]);
    }
}
