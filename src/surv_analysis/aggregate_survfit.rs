//! Population-averaged curves: the port of R's `aggregate.survfit`
//! (`R/aggregate.survfit.R`).  A `survfit(coxfit, newdata)` object holds one
//! curve per row of `newdata`, its `data` margin; the method summarises the
//! `surv` (time x data) and `pstate` (time x data x state) columns within
//! groups of that margin with `FUN` and drops the components that do not
//! collapse (`std.err`, `std.cumhaz`, `lower`, `upper`, `conf.int`,
//! `conf.type`, `logse`, `cumhaz`).  The remaining components of the object
//! (`time`, `n.risk`, `strata`, ...) are unchanged and stay with the caller.

use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::numpy_utils::{FloatArray3, FloatMatrix};
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
                let half = values.len() / 2;
                let even = values.len().is_multiple_of(2);
                // Select the middle value in linear time. For an even group,
                // the maximum of the lower partition is the other middle.
                let (lower, middle, _) = values.select_nth_unstable_by(half, f64::total_cmp);
                if even {
                    let lower = lower
                        .iter()
                        .copied()
                        .max_by(f64::total_cmp)
                        .expect("an even group has a lower middle value");
                    (lower + *middle) / 2.0
                } else {
                    *middle
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
        let factor = Self {
            name,
            codes,
            levels,
        };
        factor.validate_codes()?;
        Ok(factor)
    }

    fn validate_codes(&self) -> SurvivalResult<()> {
        if let Some((index, &code)) = self
            .codes
            .iter()
            .enumerate()
            .find(|&(_, &code)| code >= self.levels.len())
        {
            return Err(SurvivalError::invalid_input(format!(
                "by: level code {code} at index {index} is not below the number of levels {}",
                self.levels.len()
            )));
        }
        Ok(())
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
    /// The location of each input column in one contiguous group buffer.
    positions: Vec<usize>,
    /// Group boundaries in that buffer, including the final end.
    bounds: Vec<usize>,
    n_groups: usize,
    newdata: Option<AggregateGroups>,
}

impl Grouping {
    fn new(index: Vec<usize>, n_groups: usize, newdata: Option<AggregateGroups>) -> Self {
        let mut bounds = vec![0; n_groups + 1];
        for &group in &index {
            bounds[group + 1] += 1;
        }
        for group in 0..n_groups {
            bounds[group + 1] += bounds[group];
        }
        let mut next = bounds[..n_groups].to_vec();
        let positions = index
            .into_iter()
            .map(|group| {
                let position = next[group];
                next[group] += 1;
                position
            })
            .collect();
        Self {
            positions,
            bounds,
            n_groups,
            newdata,
        }
    }
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
        return Ok(Grouping::new(vec![0; n_data], 1, None));
    }
    for factor in by {
        validate_length(n_data, factor.codes.len(), "by")?;
        // The public fields can be constructed directly or changed after new().
        factor.validate_codes()?;
    }
    let (index, labels) = group_combinations(by, n_data);
    if labels.len() == 1 {
        // all in one group: R drops back to the no-`by` case
        return Ok(Grouping::new(index, 1, None));
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
    Ok(Grouping::new(
        index,
        labels.len(),
        Some(AggregateGroups { names, labels }),
    ))
}

/// Rank the observed factor combinations with the first variable fastest.
fn group_combinations(by: &[GroupingFactor], n_data: usize) -> (Vec<usize>, Vec<Vec<String>>) {
    // The tapply group: sum_k code_k * prod_{j < k} nlevels_j. Declared
    // levels can make this product overflow even for only a few data columns;
    // in that case order observed code tuples without materialising the product.
    let mut keys = vec![0usize; n_data];
    let mut stride = 1usize;
    for factor in by {
        let Some(next_stride) = stride.checked_mul(factor.levels.len()) else {
            return group_combinations_by_tuple(by, n_data);
        };
        for (key, &code) in keys.iter_mut().zip(&factor.codes) {
            *key += code * stride;
        }
        stride = next_stride;
    }
    let mut present = keys.clone();
    present.sort_unstable();
    present.dedup();
    let index = keys
        .iter()
        .map(|key| present.binary_search(key).expect("every key is present"))
        .collect();
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
    (index, labels)
}

fn group_combinations_by_tuple(
    by: &[GroupingFactor],
    n_data: usize,
) -> (Vec<usize>, Vec<Vec<String>>) {
    let mut order: Vec<usize> = (0..n_data).collect();
    order.sort_unstable_by(|&a, &b| {
        by.iter()
            .rev()
            .map(|factor| factor.codes[a])
            .cmp(by.iter().rev().map(|factor| factor.codes[b]))
    });
    let mut index = vec![0; n_data];
    let mut labels = Vec::new();
    for (position, &row) in order.iter().enumerate() {
        if position == 0
            || by
                .iter()
                .any(|factor| factor.codes[row] != factor.codes[order[position - 1]])
        {
            labels.push(
                by.iter()
                    .map(|factor| factor.levels[factor.codes[row]].clone())
                    .collect(),
            );
        }
        index[row] = labels.len() - 1;
    }
    (index, labels)
}

/// Apply `fun` within each group to the `n_data` values `column(j)`.
fn collapse(
    grouping: &Grouping,
    fun: AggregateFun,
    column: impl Fn(usize) -> f64,
    scratch: &mut [f64],
) -> Vec<f64> {
    for (j, &position) in grouping.positions.iter().enumerate() {
        scratch[position] = column(j);
    }
    grouping
        .bounds
        .windows(2)
        .map(|range| fun.apply(&mut scratch[range[0]..range[1]]))
        .collect()
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
    let mut scratch = vec![0.0; n_data];

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

/// Python binding of [`aggregate_survfit`]; `fun` is `"mean"`, `"median"`,
/// `"min"` or `"max"`.
#[pyfunction(name = "aggregate_survfit")]
#[pyo3(signature = (surv=None, pstate=None, by=None, fun="mean"))]
pub fn aggregate_survfit_py(
    py: Python<'_>,
    surv: Option<FloatMatrix>,
    pstate: Option<FloatArray3>,
    by: Option<Vec<GroupingFactor>>,
    fun: &str,
) -> PyResult<AggregateSurvfitResult> {
    let surv = surv.map(FloatMatrix::into_inner);
    let pstate = pstate.map(FloatArray3::into_inner);
    let fun = AggregateFun::parse(fun)?;
    let by = by.unwrap_or_default();
    Ok(py.detach(|| aggregate_survfit(surv.as_ref(), pstate.as_ref(), &by, fun))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted_median(mut values: Vec<f64>) -> f64 {
        if values.iter().any(|value| value.is_nan()) {
            return f64::NAN;
        }
        values.sort_by(f64::total_cmp);
        let half = values.len() / 2;
        if values.len().is_multiple_of(2) {
            (values[half - 1] + values[half]) / 2.0
        } else {
            values[half]
        }
    }

    fn assert_same_float(actual: f64, expected: f64) {
        assert!(
            actual.to_bits() == expected.to_bits() || (actual.is_nan() && expected.is_nan()),
            "{actual:?} != {expected:?}"
        );
    }

    #[test]
    fn selected_median_matches_sorting_for_odd_even_ties_and_nonfinite_values() {
        for n in 1..=257 {
            let ordered: Vec<f64> = (0..n).map(|j| j as f64 / n as f64).collect();
            let scrambled: Vec<f64> = (0..n)
                .map(|j| ((j * 7919 + n * 101) % 100003) as f64 / 100003.0)
                .collect();
            let mut with_nonfinite = scrambled.clone();
            with_nonfinite[0] = f64::INFINITY;
            if n > 1 {
                with_nonfinite[1] = f64::NEG_INFINITY;
            }
            let mut with_nan = scrambled.clone();
            with_nan[n / 2] = f64::NAN;
            for mut values in [
                ordered.clone(),
                ordered.into_iter().rev().collect(),
                scrambled,
                (0..n).map(|j| (j % 3) as f64).collect(),
                (0..n)
                    .map(|j| if j % 2 == 0 { -0.0 } else { 0.0 })
                    .collect(),
                with_nonfinite,
                with_nan,
            ] {
                let expected = sorted_median(values.clone());
                let actual = AggregateFun::Median.apply(&mut values);
                assert_same_float(actual, expected);
            }
        }
    }

    #[test]
    fn grouped_medians_preserve_members_on_strided_curves_and_multistate_arrays() {
        let n = 257;
        let value = |t: usize, j: usize, s: usize| {
            ((j * 7919 + t * 101 + s * 37) % 100003) as f64 / 100003.0
        };
        let surv = Array2::from_shape_fn((n, 5), |(j, t)| value(t, j, 0)).reversed_axes();
        let pstate = Array3::from_shape_fn((5, n, 3), |(t, j, s)| value(t, j, s));
        // Uneven groups plus an unused declared level; member order is original
        // data order even when the median routine permutes the scratch buffer.
        let codes: Vec<usize> = (0..n).map(|j| if j % 3 == 0 { 2 } else { j % 2 }).collect();
        let by = [GroupingFactor::try_new(
            codes.clone(),
            vec!["a".into(), "b".into(), "c".into(), "unused".into()],
            None,
        )
        .unwrap()];
        let result =
            aggregate_survfit(Some(&surv), Some(&pstate), &by, AggregateFun::Median).unwrap();
        for t in 0..5 {
            for group in 0..3 {
                let members: Vec<usize> = (0..n).filter(|&j| codes[j] == group).collect();
                assert_same_float(
                    result.surv.as_ref().unwrap()[t][group],
                    sorted_median(members.iter().map(|&j| surv[[t, j]]).collect()),
                );
                for s in 0..3 {
                    assert_same_float(
                        result.pstate.as_ref().unwrap()[t][group][s],
                        sorted_median(members.iter().map(|&j| pstate[[t, j, s]]).collect()),
                    );
                }
            }
        }
        assert_eq!(result.newdata.unwrap().labels.len(), 3);
    }

    #[test]
    fn many_declared_factor_levels_preserve_first_variable_fastest_order() {
        let by: Vec<GroupingFactor> = (0..usize::BITS)
            .map(|factor| GroupingFactor {
                name: None,
                codes: if factor == 0 {
                    vec![0, 1, 0, 0]
                } else if factor + 1 == usize::BITS {
                    vec![1, 0, 0, 1]
                } else {
                    vec![0; 4]
                },
                levels: vec!["a".into(), "b".into()],
            })
            .collect();
        let surv = Array2::from_shape_vec((2, 4), vec![9., 3., 1., 7., 8., 4., 2., 6.]).unwrap();
        let result = aggregate_survfit(Some(&surv), None, &by, AggregateFun::Median).unwrap();
        assert_eq!(
            result.surv.unwrap(),
            vec![vec![1., 3., 8.], vec![2., 4., 7.]]
        );
        let labels = result.newdata.unwrap().labels;
        assert!(labels[0].iter().all(|label| label == "a"));
        assert_eq!(labels[1][0], "b");
        assert!(labels[1][1..].iter().all(|label| label == "a"));
        assert_eq!(labels[2].last().unwrap(), "b");
        assert!(
            labels[2][..labels[2].len() - 1]
                .iter()
                .all(|label| label == "a")
        );
    }

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
        for invalid in [
            GroupingFactor {
                name: None,
                codes: vec![0, 1, 0, 2],
                levels: vec!["a".into(), "b".into()],
            },
            GroupingFactor {
                name: None,
                codes: vec![0; 4],
                levels: vec![],
            },
        ] {
            let err = aggregate_survfit(Some(&surv()), None, &[invalid], AggregateFun::Median)
                .unwrap_err();
            assert!(err.to_string().contains("by: level code"));
        }
        let empty_surv = Array2::zeros((2, 0));
        let empty_pstate = Array3::zeros((2, 0, 3));
        for (surv, pstate) in [(Some(&empty_surv), None), (None, Some(&empty_pstate))] {
            let err = aggregate_survfit(surv, pstate, &[], AggregateFun::Median).unwrap_err();
            assert!(err.to_string().contains("does not have a 'data' margin"));
        }
    }
}
