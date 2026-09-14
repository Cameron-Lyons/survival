//! Population-averaged curves: the port of R's `aggregate.survfit`
//! (`R/aggregate.survfit.R`).  A `survfit(coxfit, newdata)` object holds one
//! curve per row of `newdata`, its `data` margin; the method summarises the
//! `surv` (time x data) and `pstate` (time x data x state) columns within
//! groups of that margin with `FUN` and drops the components that do not
//! collapse (`std.err`, `std.cumhaz`, `lower`, `upper`, `conf.int`,
//! `conf.type`, `logse`, `cumhaz`).  The remaining components of the object
//! (`time`, `n.risk`, `strata`, ...) are unchanged and stay with the caller.

use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::validate_length;
use ndarray::{Array2, Array3};
use pyo3::prelude::*;

/// The `FUN` argument: the summary of one group's curve values at a time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AggregateFun {
    /// `mean`, the default: the population average.
    #[default]
    Mean,
    /// `median`; an even count averages the two middle values, as R.
    Median,
    Min,
    Max,
}

impl AggregateFun {
    pub fn parse(name: &str) -> SurvivalResult<Self> {
        match name {
            "mean" => Ok(Self::Mean),
            "median" => Ok(Self::Median),
            "min" => Ok(Self::Min),
            "max" => Ok(Self::Max),
            other => Err(SurvivalError::invalid_input(format!(
                "FUN must be one of mean, median, min or max; got {other:?}"
            ))),
        }
    }

    /// The summary of `values`, which are reordered in place; an `NA`
    /// (`NaN`) makes the summary `NA`, as R's `mean`, `median`, `min` and
    /// `max` do.
    fn apply(self, values: &mut [f64]) -> f64 {
        if values.iter().any(|value| value.is_nan()) {
            return f64::NAN;
        }
        let n = values.len() as f64;
        match self {
            Self::Mean => {
                // R's mean: the sum divided by n, then refined by the mean
                // of the residuals (`src/main/summary.c`)
                let first: f64 = values.iter().sum::<f64>() / n;
                first + values.iter().map(|value| value - first).sum::<f64>() / n
            }
            Self::Median => {
                values.sort_by(f64::total_cmp);
                let half = values.len() / 2;
                if values.len() % 2 == 1 {
                    values[half]
                } else {
                    (values[half - 1] + values[half]) / 2.0
                }
            }
            Self::Min => values.iter().copied().fold(f64::INFINITY, f64::min),
            Self::Max => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        }
    }
}

/// One element of R's `by` list after `as.factor`: a level code per data
/// column (`codes[j]` indexes `levels`) and, for a named list, the element's
/// name.
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct GroupingFactor {
    #[pyo3(get)]
    pub name: Option<String>,
    #[pyo3(get)]
    pub codes: Vec<usize>,
    #[pyo3(get)]
    pub levels: Vec<String>,
}

impl GroupingFactor {
    pub fn try_new(
        codes: Vec<usize>,
        levels: Vec<String>,
        name: Option<String>,
    ) -> SurvivalResult<Self> {
        if let Some((index, &code)) = codes
            .iter()
            .enumerate()
            .find(|&(_, &code)| code >= levels.len())
        {
            return Err(SurvivalError::invalid_input(format!(
                "by: level code {code} at index {index} is not below the number of levels {}",
                levels.len()
            )));
        }
        Ok(Self {
            name,
            codes,
            levels,
        })
    }
}

#[pymethods]
impl GroupingFactor {
    #[new]
    #[pyo3(signature = (codes, levels, name=None))]
    pub fn new(codes: Vec<usize>, levels: Vec<String>, name: Option<String>) -> PyResult<Self> {
        Ok(Self::try_new(codes, levels, name)?)
    }
}

/// The `newdata` of the aggregated object: one row per group with the level
/// label of every grouping variable.
#[derive(Debug, Clone, PartialEq, Eq)]
#[pyclass(from_py_object)]
pub struct AggregateGroups {
    /// The column names: the names of `by`, `Group.k` for an unnamed element
    /// of a list, or the single column `aggregate` for a bare vector.
    #[pyo3(get)]
    pub names: Vec<String>,
    /// `labels[group][column]`.
    #[pyo3(get)]
    pub labels: Vec<Vec<String>>,
}

/// The collapsed columns of the aggregated survfit object.
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct AggregateSurvfitResult {
    /// `surv[time][group]`; absent when the object had no `surv`.
    #[pyo3(get)]
    pub surv: Option<Vec<Vec<f64>>>,
    /// `pstate[time][group][state]`; absent when the object had no `pstate`.
    #[pyo3(get)]
    pub pstate: Option<Vec<Vec<Vec<f64>>>>,
    /// The group labels; `None` without `by` (or when every column falls
    /// in one group), where R drops the group dimension and `newdata`.
    #[pyo3(get)]
    pub newdata: Option<AggregateGroups>,
}

/// The `data` margin of the object and the group of every column.
struct Grouping {
    n_data: usize,
    /// `index` of the R code, 0-based and without holes.
    index: Vec<usize>,
    n_groups: usize,
    newdata: Option<AggregateGroups>,
}

/// The size of the data margin the two arrays share.
fn data_margin(surv: Option<&Array2<f64>>, pstate: Option<&Array3<f64>>) -> SurvivalResult<usize> {
    match (surv, pstate) {
        (None, None) => Err(SurvivalError::invalid_input(
            "aggregate.survfit needs a surv or a pstate matrix",
        )),
        (Some(surv), None) => Ok(surv.ncols()),
        (None, Some(pstate)) => Ok(pstate.dim().1),
        (Some(surv), Some(pstate)) => {
            let (n_time, n_data, _) = pstate.dim();
            validate_length(surv.nrows(), n_time, "pstate times")?;
            validate_length(surv.ncols(), n_data, "pstate data margin")?;
            Ok(n_data)
        }
    }
}

/// `index <- match(tapply(by[[1]], by), sort(unique(...)))`: the group of
/// every column, the first grouping variable varying fastest, renumbered
/// without holes; plus the labels `aggregate.survfit` stores as `newdata`.
fn grouping(by: &[GroupingFactor], n_data: usize) -> SurvivalResult<Grouping> {
    if by.is_empty() {
        return Ok(Grouping {
            n_data,
            index: vec![0; n_data],
            n_groups: 1,
            newdata: None,
        });
    }
    for factor in by {
        validate_length(n_data, factor.codes.len(), "by")?;
    }
    // the tapply group: 1 + sum_k code_k * prod_{j < k} nlevels_j
    let mut keys = vec![0usize; n_data];
    let mut stride = 1usize;
    for factor in by {
        for (key, &code) in keys.iter_mut().zip(&factor.codes) {
            *key += code * stride;
        }
        stride *= factor.levels.len();
    }
    let mut present = keys.clone();
    present.sort_unstable();
    present.dedup();
    let index: Vec<usize> = keys
        .iter()
        .map(|key| present.binary_search(key).expect("every key is present"))
        .collect();
    if present.len() == 1 {
        // all in one group: R drops back to the no-`by` case
        return Ok(Grouping {
            n_data,
            index,
            n_groups: 1,
            newdata: None,
        });
    }

    // The labels: `levels(as.factor(by[[1]]))` for a bare vector (its
    // levels are exactly the groups unless the factor carried unused
    // levels, where R's newdata would have more rows than the curves have
    // columns; the groups are listed here), the `aggregate` data frame of
    // the combinations present otherwise.
    let names: Vec<String> = if by.len() == 1 && by[0].name.is_none() {
        vec!["aggregate".to_string()]
    } else {
        by.iter()
            .enumerate()
            .map(|(k, factor)| {
                factor
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("Group.{}", k + 1))
            })
            .collect()
    };
    let labels = present
        .iter()
        .map(|&key| {
            let mut rest = key;
            by.iter()
                .map(|factor| {
                    let code = rest % factor.levels.len();
                    rest /= factor.levels.len();
                    factor.levels[code].clone()
                })
                .collect()
        })
        .collect();
    Ok(Grouping {
        n_data,
        index,
        n_groups: present.len(),
        newdata: Some(AggregateGroups { names, labels }),
    })
}

/// Apply `fun` within each group to the `n_data` values `column(j)`.
fn collapse(
    grouping: &Grouping,
    fun: AggregateFun,
    column: impl Fn(usize) -> f64,
    scratch: &mut [Vec<f64>],
) -> Vec<f64> {
    for group in scratch.iter_mut() {
        group.clear();
    }
    for j in 0..grouping.n_data {
        scratch[grouping.index[j]].push(column(j));
    }
    scratch.iter_mut().map(|group| fun.apply(group)).collect()
}

/// Port of `aggregate.survfit`: `surv` is the time x data matrix and
/// `pstate` the time x data x state array of the object (the rows of every
/// stratum stacked, as R stores them); `by` is empty for the plain average
/// over all columns.  The group order is R's: the first grouping variable
/// varies fastest and only the combinations present get a column.
pub fn aggregate_survfit(
    surv: Option<&Array2<f64>>,
    pstate: Option<&Array3<f64>>,
    by: &[GroupingFactor],
    fun: AggregateFun,
) -> SurvivalResult<AggregateSurvfitResult> {
    let n_data = data_margin(surv, pstate)?;
    if n_data == 0 {
        return Err(SurvivalError::invalid_input(
            "survfit object does not have a 'data' margin",
        ));
    }
    let grouping = grouping(by, n_data)?;
    let mut scratch = vec![Vec::with_capacity(n_data); grouping.n_groups];

    // apply(x$surv, 1, function(z) tapply(z, index, FUN))
    let surv = surv.map(|surv| {
        (0..surv.nrows())
            .map(|t| collapse(&grouping, fun, |j| surv[[t, j]], &mut scratch))
            .collect()
    });
    // apply(x$pstate, c(1, 3), function(z) tapply(z, index, FUN)), permuted
    // back to time x group x state
    let pstate = pstate.map(|pstate| {
        let (n_time, _, n_state) = pstate.dim();
        (0..n_time)
            .map(|t| {
                let mut by_group = vec![Vec::with_capacity(n_state); grouping.n_groups];
                for s in 0..n_state {
                    let values = collapse(&grouping, fun, |j| pstate[[t, j, s]], &mut scratch);
                    for (row, value) in by_group.iter_mut().zip(values) {
                        row.push(value);
                    }
                }
                by_group
            })
            .collect()
    });
    Ok(AggregateSurvfitResult {
        surv,
        pstate,
        newdata: grouping.newdata,
    })
}

/// The `[time][data]` rows of `surv` as a matrix.
fn surv_matrix(rows: &[Vec<f64>]) -> SurvivalResult<Array2<f64>> {
    let n_data = rows.first().map_or(0, Vec::len);
    let mut flat = Vec::with_capacity(rows.len() * n_data);
    for (t, row) in rows.iter().enumerate() {
        validate_length(n_data, row.len(), &format!("surv row {t}"))?;
        flat.extend_from_slice(row);
    }
    Array2::from_shape_vec((rows.len(), n_data), flat)
        .map_err(|err| SurvivalError::invalid_input(format!("surv: {err}")))
}

/// The `[time][data][state]` entries of `pstate` as an array.
fn pstate_array(entries: &[Vec<Vec<f64>>]) -> SurvivalResult<Array3<f64>> {
    let n_data = entries.first().map_or(0, Vec::len);
    let n_state = entries
        .first()
        .and_then(|row| row.first())
        .map_or(0, Vec::len);
    let mut flat = Vec::with_capacity(entries.len() * n_data * n_state);
    for (t, row) in entries.iter().enumerate() {
        validate_length(n_data, row.len(), &format!("pstate time {t}"))?;
        for (j, states) in row.iter().enumerate() {
            validate_length(n_state, states.len(), &format!("pstate time {t} data {j}"))?;
            flat.extend_from_slice(states);
        }
    }
    Array3::from_shape_vec((entries.len(), n_data, n_state), flat)
        .map_err(|err| SurvivalError::invalid_input(format!("pstate: {err}")))
}

/// Python binding of [`aggregate_survfit`]; `fun` is `"mean"`, `"median"`,
/// `"min"` or `"max"`.
#[pyfunction(name = "aggregate_survfit")]
#[pyo3(signature = (surv=None, pstate=None, by=None, fun="mean"))]
pub fn aggregate_survfit_py(
    surv: Option<Vec<Vec<f64>>>,
    pstate: Option<Vec<Vec<Vec<f64>>>>,
    by: Option<Vec<GroupingFactor>>,
    fun: &str,
) -> PyResult<AggregateSurvfitResult> {
    let surv = surv.as_deref().map(surv_matrix).transpose()?;
    let pstate = pstate.as_deref().map(pstate_array).transpose()?;
    Ok(aggregate_survfit(
        surv.as_ref(),
        pstate.as_ref(),
        by.as_deref().unwrap_or_default(),
        AggregateFun::parse(fun)?,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    // three times x four newdata rows, column by column
    fn surv() -> Array2<f64> {
        Array2::from_shape_vec(
            (4, 3),
            vec![
                0.9, 0.8, 0.7, 0.95, 0.85, 0.6, 0.8, 0.5, 0.4, 0.7, 0.65, 0.3,
            ],
        )
        .unwrap()
        .reversed_axes()
    }

    fn by_ab() -> GroupingFactor {
        // as.factor(c("b", "a", "b", "a"))
        GroupingFactor::try_new(
            vec![1, 0, 1, 0],
            vec!["a".to_string(), "b".to_string()],
            None,
        )
        .unwrap()
    }

    fn assert_rows_close(actual: &[Vec<f64>], expected: &[&[f64]]) {
        assert_eq!(actual.len(), expected.len());
        for (row, (left, right)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(left.len(), right.len(), "row {row}");
            for (col, (a, e)) in left.iter().zip(right.iter()).enumerate() {
                assert!(
                    (a - e).abs() < 1e-12 || (a.is_nan() && e.is_nan()),
                    "row {row} col {col}: {a} != {e}"
                );
            }
        }
    }

    #[test]
    fn plain_average_matches_r() {
        // aggregate(x)$surv
        let result = aggregate_survfit(Some(&surv()), None, &[], AggregateFun::Mean).unwrap();
        assert_rows_close(result.surv.as_ref().unwrap(), &[&[0.8375], &[0.7], &[0.5]]);
        assert!(result.pstate.is_none());
        assert!(result.newdata.is_none());
    }

    #[test]
    fn grouped_summaries_match_r() {
        let surv = surv();
        let by = [by_ab()];
        // aggregate(x, by = c("b", "a", "b", "a"))
        let mean = aggregate_survfit(Some(&surv), None, &by, AggregateFun::Mean).unwrap();
        assert_rows_close(
            mean.surv.as_ref().unwrap(),
            &[&[0.825, 0.85], &[0.75, 0.65], &[0.45, 0.55]],
        );
        assert_eq!(
            mean.newdata.unwrap(),
            AggregateGroups {
                names: vec!["aggregate".to_string()],
                labels: vec![vec!["a".to_string()], vec!["b".to_string()]],
            }
        );
        // FUN = median (two values per group: their average), max, min
        let median = aggregate_survfit(Some(&surv), None, &by, AggregateFun::Median).unwrap();
        assert_rows_close(
            median.surv.as_ref().unwrap(),
            &[&[0.825, 0.85], &[0.75, 0.65], &[0.45, 0.55]],
        );
        let max = aggregate_survfit(Some(&surv), None, &by, AggregateFun::Max).unwrap();
        assert_rows_close(
            max.surv.as_ref().unwrap(),
            &[&[0.95, 0.9], &[0.85, 0.8], &[0.6, 0.7]],
        );
        let min = aggregate_survfit(Some(&surv), None, &by, AggregateFun::Min).unwrap();
        assert_rows_close(
            min.surv.as_ref().unwrap(),
            &[&[0.7, 0.8], &[0.65, 0.5], &[0.3, 0.4]],
        );
    }

    #[test]
    fn two_grouping_variables_order_the_first_fastest() {
        // aggregate(x, by = list(g = c("b","a","b","a"), h = c(1, 1, 2, 1)))
        let h = GroupingFactor::try_new(
            vec![0, 0, 1, 0],
            vec!["1".to_string(), "2".to_string()],
            Some("h".to_string()),
        )
        .unwrap();
        let mut g = by_ab();
        g.name = Some("g".to_string());
        let result =
            aggregate_survfit(Some(&surv()), None, &[g, h.clone()], AggregateFun::Mean).unwrap();
        assert_rows_close(
            result.surv.as_ref().unwrap(),
            &[&[0.825, 0.9, 0.8], &[0.75, 0.8, 0.5], &[0.45, 0.7, 0.4]],
        );
        let newdata = result.newdata.unwrap();
        assert_eq!(newdata.names, vec!["g", "h"]);
        assert_eq!(
            newdata.labels,
            vec![vec!["a", "1"], vec!["b", "1"], vec!["b", "2"]]
        );
        // an unnamed element of a list is `Group.k`
        let unnamed = aggregate_survfit(Some(&surv()), None, &[by_ab(), h], AggregateFun::Mean)
            .unwrap()
            .newdata
            .unwrap();
        assert_eq!(unnamed.names, vec!["Group.1", "h"]);
    }

    #[test]
    fn a_single_group_drops_back_to_the_plain_average() {
        // aggregate(x, by = c(2, 2, 2, 2))
        let by = GroupingFactor::try_new(vec![0; 4], vec!["2".to_string()], None).unwrap();
        let result = aggregate_survfit(Some(&surv()), None, &[by], AggregateFun::Mean).unwrap();
        assert_rows_close(result.surv.as_ref().unwrap(), &[&[0.8375], &[0.7], &[0.5]]);
        assert!(result.newdata.is_none());
    }

    #[test]
    fn pstate_is_collapsed_per_state() {
        // array(seq(0.01, by = 0.01, length.out = 36), c(3, 4, 3)): R fills
        // the first index fastest
        let mut pstate = Array3::zeros((3, 4, 3));
        for s in 0..3 {
            for j in 0..4 {
                for t in 0..3 {
                    pstate[[t, j, s]] = 0.01 * (1 + t + 3 * j + 12 * s) as f64;
                }
            }
        }
        let plain = aggregate_survfit(None, Some(&pstate), &[], AggregateFun::Mean).unwrap();
        let plain = plain.pstate.unwrap();
        assert_eq!(plain.len(), 3);
        assert_rows_close(&plain[0], &[&[0.055, 0.175, 0.295]]);
        assert_rows_close(&plain[2], &[&[0.075, 0.195, 0.315]]);
        let grouped =
            aggregate_survfit(None, Some(&pstate), &[by_ab()], AggregateFun::Mean).unwrap();
        let grouped = grouped.pstate.unwrap();
        assert_rows_close(&grouped[0], &[&[0.07, 0.19, 0.31], &[0.04, 0.16, 0.28]]);
        assert_rows_close(&grouped[1], &[&[0.08, 0.20, 0.32], &[0.05, 0.17, 0.29]]);
        assert_rows_close(&grouped[2], &[&[0.09, 0.21, 0.33], &[0.06, 0.18, 0.30]]);
    }

    #[test]
    fn na_propagates_like_r() {
        let mut surv = surv();
        surv[[1, 0]] = f64::NAN;
        let max = aggregate_survfit(Some(&surv), None, &[by_ab()], AggregateFun::Max).unwrap();
        assert_rows_close(
            max.surv.as_ref().unwrap(),
            &[&[0.95, 0.9], &[0.85, f64::NAN], &[0.6, 0.7]],
        );
        let mean = aggregate_survfit(Some(&surv), None, &[], AggregateFun::Mean).unwrap();
        assert_rows_close(
            mean.surv.as_ref().unwrap(),
            &[&[0.8375], &[f64::NAN], &[0.5]],
        );
    }

    #[test]
    fn rejects_bad_inputs() {
        let err = aggregate_survfit(None, None, &[], AggregateFun::Mean).unwrap_err();
        assert!(err.to_string().contains("surv or a pstate"));
        let short =
            GroupingFactor::try_new(vec![0, 1], vec!["a".into(), "b".into()], None).unwrap();
        let err = aggregate_survfit(Some(&surv()), None, &[short], AggregateFun::Mean).unwrap_err();
        assert!(err.to_string().contains("by length mismatch"));
        assert!(GroupingFactor::try_new(vec![0, 2], vec!["a".into(), "b".into()], None).is_err());
        assert!(AggregateFun::parse("sd").is_err());
        let pstate = Array3::zeros((3, 2, 2));
        let err =
            aggregate_survfit(Some(&surv()), Some(&pstate), &[], AggregateFun::Mean).unwrap_err();
        assert!(err.to_string().contains("pstate data margin"));
        let ragged =
            aggregate_survfit_py(Some(vec![vec![0.9, 0.8], vec![0.7]]), None, None, "mean");
        assert!(ragged.is_err());
    }
}
