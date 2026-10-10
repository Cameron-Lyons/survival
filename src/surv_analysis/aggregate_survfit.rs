//! Population-averaged curves: the port of R's `aggregate.survfit`
//! (`R/aggregate.survfit.R`).  A `survfit(coxfit, newdata)` object holds one
//! curve per row of `newdata`, its `data` margin; the method summarises the
//! `surv` (time x data) and `pstate` (time x data x state) columns within
//! groups of that margin with `FUN` and drops the components that do not
//! collapse (`std.err`, `std.cumhaz`, `lower`, `upper`, `conf.int`,
//! `conf.type`, `logse`, `cumhaz`).  The remaining components of the object
//! (`time`, `n.risk`, `strata`, ...) are unchanged and stay with the caller.

use super::aggregate_arithmetic::{r_mean, r_row_mean, r_sum};
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::numpy_utils::{FloatArray3, FloatMatrix};
use crate::internal::validation::validate_length;
use ndarray::{Array2, Array3};
use pyo3::prelude::*;

/// The `FUN` argument: the summary of one group's curve values at a time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AggregateFun {
    /// R's omitted `FUN`: `rowMeans` for ungrouped survival, otherwise `mean`.
    #[default]
    DefaultMean,
    /// Explicit `mean`: the population average, including residual refinement.
    Mean,
    /// `median`; an even count averages the two middle values, as R.
    Median,
    Min,
    Max,
    /// `sum`, the total of the group's values.
    Sum,
}

impl AggregateFun {
    pub fn parse(name: &str) -> SurvivalResult<Self> {
        match name {
            "mean" => Ok(Self::Mean),
            "median" => Ok(Self::Median),
            "min" => Ok(Self::Min),
            "max" => Ok(Self::Max),
            "sum" => Ok(Self::Sum),
            other => Err(SurvivalError::invalid_input(format!(
                "FUN must be one of mean, median, min, max or sum; got {other:?}"
            ))),
        }
    }

    /// The summary of `values`, which are reordered in place; an `NA`
    /// (`NaN`) makes the summary `NA`, as R's built-in summaries do.
    fn apply(self, values: &mut [f64]) -> f64 {
        if matches!(self, Self::DefaultMean | Self::Mean) {
            return r_mean(values);
        }
        if self == Self::Sum {
            return r_sum(values);
        }
        if values.iter().any(|value| value.is_nan()) {
            return f64::NAN;
        }
        match self {
            Self::DefaultMean | Self::Mean | Self::Sum => {
                unreachable!("handled before the NaN scan")
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

/// Gather once, retaining original data order within each group.
fn gather(grouping: &Grouping, column: impl Fn(usize) -> f64, scratch: &mut [f64]) {
    for (j, &position) in grouping.positions.iter().enumerate() {
        scratch[position] = column(j);
    }
}

/// Apply `fun` within each group to the `n_data` values `column(j)`.
fn collapse<E>(
    grouping: &Grouping,
    fun: &mut impl FnMut(&mut [f64]) -> Result<f64, E>,
    column: impl Fn(usize) -> f64,
    scratch: &mut [f64],
) -> Result<Vec<f64>, E> {
    gather(grouping, column, scratch);
    grouping
        .bounds
        .windows(2)
        .map(|range| fun(&mut scratch[range[0]..range[1]]))
        .collect()
}

fn prepare_grouping(
    surv: Option<&Array2<f64>>,
    pstate: Option<&Array3<f64>>,
    by: &[GroupingFactor],
) -> SurvivalResult<Grouping> {
    let n_data = data_margin(surv, pstate)?;
    if n_data == 0 {
        return Err(SurvivalError::invalid_input(
            "survfit object does not have a 'data' margin",
        ));
    }
    grouping(by, n_data)
}

/// Run reductions over the shared group layout. Python uses `PyErr` here so
/// an exception from a callback reaches its caller without conversion.
#[derive(Clone, Copy)]
enum CurveComponent {
    Survival,
    StateProbabilities,
}

fn aggregate_curves<E>(
    surv: Option<&Array2<f64>>,
    pstate: Option<&Array3<f64>>,
    grouping: Grouping,
    scratch: &mut [f64],
    fun: &mut impl FnMut(CurveComponent, &mut [f64]) -> Result<f64, E>,
    state_first: bool,
) -> Result<AggregateSurvfitResult, E> {
    // apply(x$surv, 1, function(z) tapply(z, index, FUN))
    let surv = surv
        .map(|surv| {
            (0..surv.nrows())
                .map(|t| {
                    collapse(
                        &grouping,
                        &mut |values| fun(CurveComponent::Survival, values),
                        |j| surv[[t, j]],
                        scratch,
                    )
                })
                .collect::<Result<Vec<_>, E>>()
        })
        .transpose()?;
    let pstate = pstate
        .map(|pstate| {
            let (n_time, _, n_state) = pstate.dim();
            if n_time == 0 {
                return Ok(Vec::new());
            }
            let mut result: Vec<_> = (0..n_time)
                .map(|_| vec![vec![0.0; n_state]; grouping.n_groups])
                .collect();
            if state_first {
                // R's apply(..., c(1, 3), ...) visits time before state changes.
                for s in 0..n_state {
                    for (t, row) in result.iter_mut().enumerate() {
                        let values = collapse(
                            &grouping,
                            &mut |values| fun(CurveComponent::StateProbabilities, values),
                            |j| pstate[[t, j, s]],
                            scratch,
                        )?;
                        for (group, value) in row.iter_mut().zip(values) {
                            group[s] = value;
                        }
                    }
                }
            } else {
                // Stateless built-ins retain their more local traversal.
                for (t, row) in result.iter_mut().enumerate() {
                    for s in 0..n_state {
                        let values = collapse(
                            &grouping,
                            &mut |values| fun(CurveComponent::StateProbabilities, values),
                            |j| pstate[[t, j, s]],
                            scratch,
                        )?;
                        for (group, value) in row.iter_mut().zip(values) {
                            group[s] = value;
                        }
                    }
                }
            }
            Ok(result)
        })
        .transpose()?;
    Ok(AggregateSurvfitResult {
        surv,
        pstate,
        newdata: grouping.newdata,
    })
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
    let grouping = prepare_grouping(surv, pstate, by)?;
    let use_row_mean = fun == AggregateFun::DefaultMean && grouping.n_groups == 1;
    let mut scratch = vec![0.0; grouping.positions.len()];
    aggregate_curves(
        surv,
        pstate,
        grouping,
        &mut scratch,
        &mut |component, values| {
            Ok(
                if use_row_mean && matches!(component, CurveComponent::Survival) {
                    r_row_mean(values)
                } else {
                    fun.apply(values)
                },
            )
        },
        false,
    )
}

/// Aggregate with a custom scalar summary, using the same group layout and
/// scratch buffer as the built-in reducers. The callback first receives each
/// group's 1-based data indices, as R's validation call does. It then receives
/// survival values in time/group order, followed by state probabilities in
/// state/time/group order. Each slice retains original data-column order.
/// A callback error stops evaluation immediately; `NaN` and infinities are
/// valid summaries. Input arrays are never changed.
pub fn aggregate_survfit_with(
    surv: Option<&Array2<f64>>,
    pstate: Option<&Array3<f64>>,
    by: &[GroupingFactor],
    mut fun: impl FnMut(&[f64]) -> SurvivalResult<f64>,
) -> SurvivalResult<AggregateSurvfitResult> {
    let grouping = prepare_grouping(surv, pstate, by)?;
    let mut scratch = vec![0.0; grouping.positions.len()];
    collapse(
        &grouping,
        &mut |values| fun(values),
        |j| (j + 1) as f64,
        &mut scratch,
    )?;
    aggregate_curves(
        surv,
        pstate,
        grouping,
        &mut scratch,
        &mut |_, values| fun(values),
        true,
    )
}

/// Python binding of [`aggregate_survfit`], or [`aggregate_survfit_with`] for
/// a callable receiving a fresh one-dimensional NumPy array per invocation.
/// Named built-ins run detached; Python callbacks retain the GIL and their
/// original exceptions. A callback must return a real numeric scalar or a
/// numeric NumPy array containing one element.
#[cfg(feature = "python")]
#[pyfunction(name = "aggregate_survfit")]
#[pyo3(signature = (surv=None, pstate=None, by=None, fun=None, **_kwargs))]
pub fn aggregate_survfit_py(
    py: Python<'_>,
    surv: Option<FloatMatrix>,
    pstate: Option<FloatArray3>,
    by: Option<Vec<GroupingFactor>>,
    fun: Option<Py<PyAny>>,
    _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<AggregateSurvfitResult> {
    use numpy::PyArray1;
    use pyo3::types::PyString;

    let surv = surv.map(FloatMatrix::into_inner);
    let pstate = pstate.map(FloatArray3::into_inner);
    let by = by.unwrap_or_default();
    let Some(fun) = fun else {
        return Ok(py.detach(|| {
            aggregate_survfit(
                surv.as_ref(),
                pstate.as_ref(),
                &by,
                AggregateFun::DefaultMean,
            )
        })?);
    };
    let fun = fun.bind(py);
    if let Ok(name) = fun.cast::<PyString>() {
        let fun = AggregateFun::parse(name.to_str()?)?;
        return Ok(py.detach(|| aggregate_survfit(surv.as_ref(), pstate.as_ref(), &by, fun))?);
    }
    if !fun.is_callable() {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "FUN must be a supported name or a callable",
        ));
    }
    let grouping = prepare_grouping(surv.as_ref(), pstate.as_ref(), &by)?;
    let mut scratch = vec![0.0; grouping.positions.len()];
    gather(&grouping, |j| (j + 1) as f64, &mut scratch);
    // Stock tapply evaluates every validation group before aggregate checks
    // whether all its returns are numeric scalar summaries.
    let checked = grouping
        .bounds
        .windows(2)
        .map(|range| fun.call1((PyArray1::from_slice(py, &scratch[range[0]..range[1]]),)))
        .collect::<PyResult<Vec<_>>>()?;
    let numpy_number = py.import("numpy")?.getattr("number")?;
    for value in checked {
        python_scalar(&value, &numpy_number)?;
    }
    aggregate_curves(
        surv.as_ref(),
        pstate.as_ref(),
        grouping,
        &mut scratch,
        &mut |_, values| {
            let value = fun.call1((PyArray1::from_slice(py, values),))?;
            python_scalar(&value, &numpy_number)
        },
        true,
    )
}

#[cfg(feature = "python")]
fn python_scalar(value: &Bound<'_, PyAny>, numpy_number: &Bound<'_, PyAny>) -> PyResult<f64> {
    use numpy::{PyUntypedArray, PyUntypedArrayMethods};
    use pyo3::types::{PyBool, PyFloat, PyInt};

    let invalid =
        || pyo3::exceptions::PyValueError::new_err("FUN must return a single value summary");
    if value.is_instance_of::<PyBool>() {
        return Err(invalid());
    }
    if value.is_instance_of::<PyFloat>() || value.is_instance_of::<PyInt>() {
        return value.extract::<f64>();
    }
    if let Ok(array) = value.cast::<PyUntypedArray>() {
        if array.len() != 1 {
            return Err(invalid());
        }
    } else if !value.is_instance(numpy_number)? {
        return Err(invalid());
    }
    let kind = value
        .getattr("dtype")?
        .getattr("kind")?
        .extract::<String>()?;
    if !matches!(kind.as_str(), "i" | "u" | "f") {
        return Err(invalid());
    }
    value.call_method0("item")?.extract::<f64>()
}

// The Rust-only shim has no Python objects to inspect. Keep its existing
// named reducer wrapper so the module exports match both feature builds.
#[cfg(not(feature = "python"))]
pub fn aggregate_survfit_py(
    py: Python<'_>,
    surv: Option<FloatMatrix>,
    pstate: Option<FloatArray3>,
    by: Option<Vec<GroupingFactor>>,
    fun: Option<&str>,
) -> PyResult<AggregateSurvfitResult> {
    let surv = surv.map(FloatMatrix::into_inner);
    let pstate = pstate.map(FloatArray3::into_inner);
    let by = by.unwrap_or_default();
    let fun = fun
        .map(AggregateFun::parse)
        .transpose()?
        .unwrap_or_default();
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
    fn callback_validation_order_and_scratch_match_r() {
        let surv = Array2::from_shape_fn((2, 4), |(t, j)| (10 * (t + 1) + j) as f64);
        let pstate =
            Array3::from_shape_fn((2, 4, 2), |(t, j, s)| (100 * (s + 1) + 10 * t + j) as f64);
        let before_surv = surv.clone();
        let before_pstate = pstate.clone();
        let mut calls = Vec::new();
        let mut buffers = Vec::new();
        let result = aggregate_survfit_with(Some(&surv), Some(&pstate), &[by_ab()], |values| {
            calls.push(values.to_vec());
            buffers.push(values.as_ptr());
            Ok(calls.len() as f64)
        })
        .unwrap();
        assert_eq!(
            calls,
            vec![
                vec![2., 4.],
                vec![1., 3.],
                vec![11., 13.],
                vec![10., 12.],
                vec![21., 23.],
                vec![20., 22.],
                vec![101., 103.],
                vec![100., 102.],
                vec![111., 113.],
                vec![110., 112.],
                vec![201., 203.],
                vec![200., 202.],
                vec![211., 213.],
                vec![210., 212.],
            ]
        );
        // The two groups borrow fixed disjoint portions of one buffer; it is
        // reused for validation, survival rows and every state probability.
        for (call, &pointer) in buffers.iter().enumerate() {
            assert_eq!(pointer, buffers[call % 2]);
        }
        assert_eq!(result.surv.unwrap(), vec![vec![3., 4.], vec![5., 6.]]);
        assert_eq!(
            result.pstate.unwrap(),
            vec![
                vec![vec![7., 11.], vec![8., 12.]],
                vec![vec![9., 13.], vec![10., 14.]]
            ]
        );
        assert_eq!(surv, before_surv);
        assert_eq!(pstate, before_pstate);
    }

    #[test]
    fn callback_errors_stop_validation_or_curve_evaluation() {
        let curves = surv();
        let error = SurvivalError::computation("custom summary failed");
        for fail_at in [1, 2, 3, 5] {
            let mut calls = 0;
            let actual = aggregate_survfit_with(Some(&curves), None, &[by_ab()], |_| {
                calls += 1;
                if calls == fail_at {
                    Err(error.clone())
                } else {
                    Ok(0.0)
                }
            })
            .unwrap_err();
            assert_eq!(actual, error);
            assert_eq!(calls, fail_at);
        }
        let mut calls = 0;
        let short = GroupingFactor::try_new(vec![0], vec!["a".into()], None).unwrap();
        assert!(
            aggregate_survfit_with(Some(&curves), None, &[short], |_| {
                calls += 1;
                Ok(0.0)
            })
            .is_err()
        );
        assert_eq!(calls, 0);
    }

    #[test]
    fn callback_accepts_nonfinite_scalar_summaries() {
        for expected in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let result =
                aggregate_survfit_with(Some(&surv()), None, &[], |_| Ok(expected)).unwrap();
            for row in result.surv.unwrap() {
                assert_same_float(row[0], expected);
            }
        }
    }

    #[test]
    fn empty_time_and_state_axes_preserve_shapes_without_state_allocation() {
        let by =
            [GroupingFactor::try_new(vec![1, 0, 1], vec!["a".into(), "b".into()], None).unwrap()];
        // The input contains no data. Its large state margin must never
        // allocate a state-vector template when the time margin is empty.
        let pstate = Array3::from_shape_vec((0, 3, 1usize << 28), Vec::new()).unwrap();
        let named = aggregate_survfit(None, Some(&pstate), &by, AggregateFun::Sum).unwrap();
        assert!(named.pstate.unwrap().is_empty());
        let mut calls = 0;
        let custom = aggregate_survfit_with(None, Some(&pstate), &by, |values| {
            calls += 1;
            assert_eq!(values, if calls == 1 { &[2.][..] } else { &[1., 3.][..] });
            Ok(0.0)
        })
        .unwrap();
        assert_eq!(calls, 2);
        assert!(custom.pstate.unwrap().is_empty());
        let no_states = Array3::zeros((2, 3, 0));
        calls = 0;
        let result = aggregate_survfit_with(None, Some(&no_states), &by, |_| {
            calls += 1;
            Ok(0.0)
        })
        .unwrap();
        assert_eq!(calls, 2);
        assert_eq!(result.pstate.unwrap(), vec![vec![Vec::<f64>::new(); 2]; 2]);
    }

    #[test]
    fn sum_and_mean_match_r_on_exceptional_values() {
        let curves = Array2::from_shape_vec(
            (6, 4),
            vec![
                1e16,
                1.,
                -1e16,
                0.,
                1e300,
                1.,
                -1e300,
                0.,
                1e308,
                1e308,
                -1e308,
                0.,
                f64::INFINITY,
                1.,
                2.,
                3.,
                f64::NEG_INFINITY,
                1.,
                2.,
                3.,
                f64::INFINITY,
                f64::NEG_INFINITY,
                1.,
                2.,
            ],
        )
        .unwrap();
        let totals = aggregate_survfit(Some(&curves), None, &[], AggregateFun::Sum).unwrap();
        let means = aggregate_survfit(Some(&curves), None, &[], AggregateFun::Mean).unwrap();
        for (row, expected) in totals.surv.unwrap().iter().zip([
            1.,
            0.,
            1e308,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ]) {
            assert_same_float(row[0], expected);
        }
        for (row, expected) in means.surv.unwrap().iter().zip([
            0.25,
            0.,
            2.5e307,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ]) {
            assert_same_float(row[0], expected);
        }
        assert_eq!(AggregateFun::parse("sum").unwrap(), AggregateFun::Sum);
    }

    #[test]
    fn default_uses_row_means_only_for_ungrouped_survival() {
        let values = vec![f64::MAX, f64::MAX, -f64::MAX, -f64::MAX, 1.];
        let surv = Array2::from_shape_vec((1, 5), values.clone()).unwrap();
        let pstate = Array3::from_shape_vec((1, 5, 1), values.clone()).unwrap();
        let constant = [GroupingFactor::try_new(vec![0; 5], vec!["same".into()], None).unwrap()];
        for by in [&[][..], &constant[..]] {
            let result =
                aggregate_survfit(Some(&surv), Some(&pstate), by, AggregateFun::default()).unwrap();
            assert_same_float(result.surv.unwrap()[0][0], 0.2);
            assert_same_float(result.pstate.unwrap()[0][0][0], 0.36);
        }
        let explicit = aggregate_survfit(Some(&surv), None, &[], AggregateFun::Mean).unwrap();
        assert_same_float(explicit.surv.unwrap()[0][0], 0.36);
        let surv =
            Array2::from_shape_vec((1, 10), [values.clone(), values.clone()].concat()).unwrap();
        let by = [GroupingFactor::try_new(
            vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            vec!["a".into(), "b".into()],
            None,
        )
        .unwrap()];
        let grouped = aggregate_survfit(Some(&surv), None, &by, AggregateFun::DefaultMean).unwrap();
        assert_eq!(grouped.surv.unwrap(), vec![vec![0.36, 0.36]]);
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
