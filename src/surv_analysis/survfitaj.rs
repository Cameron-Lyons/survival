//! Aalen-Johansen estimates of the probability in state for multi-state
//! survival data: the port of R's `survfitAJ` (`R/survfitAJ.R`), the data
//! checks of `survcheck2` (`R/survcheck.R`, `src/multicheck.c`) it relies
//! on, and its C kernel `survfitaj` (`src/survfitaj.c`).

use super::survfit_aj_summary::{
    AJMeanTable, summary_rows, summary_survfit_aj_with_counts, survmean_aj, survmean_coxms,
};
use super::survfit_confint::{ConfType, survfit_confint, validate_conf_int};
use super::survfit_summary::{RmeanOption, survfit0_aj_rows};
use super::survfitkm::{
    check_curve_indices, curve_ranges, rows_by_curve, select_items, strata_index, survflag,
};
use crate::core::strata_order::validate_intervals;
use crate::data_prep::{aeq_counting, first_appearance_codes};
use crate::error::{SurvivalError, SurvivalResult};
#[cfg(feature = "python")]
use crate::internal::numpy_utils::readonly_view;
use crate::internal::numpy_utils::{FloatVec, IntVec};
use crate::internal::sorting::ordered_subset;
use crate::internal::validation::{
    validate_finite, validate_length, validate_non_empty, validate_non_negative,
};
use ndarray::{Array2, Array3, Axis, ShapeBuilder, s};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;

mod independent;

/// The data of a `survfit(Surv(...) ~ strata, id, istate, weights, cluster)`
/// call with a multi-state outcome.
///
/// `state[i]` is the state entered at the end of interval `i`: 0 for
/// censored, `k` for `states[k - 1]` (R's `Surv(time, factor)` coding).
/// `istate` names the state occupied at the start of each interval; names
/// not among `states` (such as R's default `"(s0)"`) become extra initial
/// states, ordered by `istate_levels` when given and alphabetically
/// otherwise.
#[derive(Debug, Clone)]
pub struct SurvfitAJData {
    pub start: Option<Vec<f64>>,
    pub time: Vec<f64>,
    pub state: Vec<i32>,
    pub states: Vec<String>,
    pub weights: Option<Vec<f64>>,
    pub strata: Option<Vec<i32>>,
    pub id: Option<Vec<i64>>,
    pub istate: Option<Vec<String>>,
    pub istate_levels: Option<Vec<String>>,
    pub cluster: Option<Vec<i64>>,
}

impl SurvfitAJData {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        start: Option<Vec<f64>>,
        time: Vec<f64>,
        state: Vec<i32>,
        states: Vec<String>,
        weights: Option<Vec<f64>>,
        strata: Option<Vec<i32>>,
        id: Option<Vec<i64>>,
        istate: Option<Vec<String>>,
        istate_levels: Option<Vec<String>>,
        cluster: Option<Vec<i64>>,
    ) -> SurvivalResult<Self> {
        let data = Self {
            start,
            time,
            state,
            states,
            weights,
            strata,
            id,
            istate,
            istate_levels,
            cluster,
        };
        data.validate()?;
        Ok(data)
    }

    /// Check row alignment and state codes, including after public fields have
    /// been modified. [`survfitaj`] validates these inputs before fitting.
    pub fn validate(&self) -> SurvivalResult<()> {
        validate_non_empty(&self.time, "time")?;
        validate_finite(&self.time, "time")?;
        validate_length(self.time.len(), self.state.len(), "state")?;
        if self.states.is_empty() {
            return Err(SurvivalError::invalid_input(
                "a multi-state outcome needs at least one state",
            ));
        }
        if let Some((index, &code)) = self
            .state
            .iter()
            .enumerate()
            .find(|&(_, &code)| code < 0 || code as usize > self.states.len())
        {
            return Err(SurvivalError::invalid_input(format!(
                "state code {code} at index {index} is not 0 (censored) or 1..{}",
                self.states.len()
            )));
        }
        if let Some(start) = &self.start {
            validate_length(self.time.len(), start.len(), "start")?;
            validate_finite(start, "start")?;
            validate_intervals(start, &self.time)?;
        }
        if let Some(weights) = &self.weights {
            validate_length(self.time.len(), weights.len(), "weights")?;
            validate_finite(weights, "weights")?;
            validate_non_negative(weights, "weights")?;
        }
        for (name, values) in [
            ("strata", self.strata.as_ref().map(Vec::len)),
            ("id", self.id.as_ref().map(Vec::len)),
            ("cluster", self.cluster.as_ref().map(Vec::len)),
            ("istate", self.istate.as_ref().map(Vec::len)),
        ] {
            if let Some(len) = values {
                validate_length(self.time.len(), len, name)?;
            }
        }
        if let (Some(istate), Some(levels)) = (&self.istate, &self.istate_levels) {
            let levels: HashSet<&str> = levels.iter().map(String::as_str).collect();
            if let Some(bad) = istate.iter().find(|value| !levels.contains(value.as_str())) {
                return Err(SurvivalError::invalid_input(format!(
                    "istate value {bad:?} is not one of istate_levels"
                )));
            }
        }
        Ok(())
    }
}

/// The arguments of `survfitAJ` beyond the data.  Only `stype = 1`,
/// `ctype = 1` and the robust variance exist for multi-state curves, so
/// those are not options.
#[derive(Debug, Clone)]
pub struct SurvfitAJOptions {
    pub se_fit: bool,
    pub conf_int: f64,
    pub conf_type: ConfType,
    /// Return the per-cluster influence on `pstate`.
    pub influence: bool,
    pub start_time: Option<f64>,
    /// The initial state distribution; estimated from the data at `t0`
    /// when absent.
    pub p0: Option<Vec<f64>>,
    /// Report the number entering the risk set (counting-process data
    /// with an id only).
    pub entry: bool,
    /// Report a row at `t0` (and skip `se0`/`i0`).
    pub time0: bool,
    pub timefix: bool,
}

impl Default for SurvfitAJOptions {
    fn default() -> Self {
        Self {
            se_fit: true,
            conf_int: 0.95,
            conf_type: ConfType::Log,
            influence: false,
            start_time: None,
            p0: None,
            entry: false,
            time0: false,
            timefix: true,
        }
    }
}

/// Unweighted counts of a multi-state fit, reported when case weights are
/// present.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[pyclass(module = "survival._survival", from_py_object)]
pub struct SurvfitAJCounts {
    #[pyo3(get)]
    pub n_risk: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub n_transition: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub n_censor: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub n_enter: Option<Vec<Vec<f64>>>,
}

crate::internal::pickle::picklable!(SurvfitAJCounts);

/// One curve's influence on `pstate`: `values[[cluster, time, state]]`,
/// column-major like R's `influence.pstate` array.  The array is shared:
/// clones of the fit and the NumPy array Python reads (a read-only view) do
/// not copy it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[pyclass(module = "survival._survival", frozen, from_py_object)]
pub struct SurvfitAJInfluence {
    /// R's row names `uclust`: the number (1, 2, ...) of each cluster in
    /// order of first appearance in the rows the fit uses.
    #[pyo3(get)]
    pub cluster: Vec<i64>,
    #[serde(with = "crate::internal::pickle::memory_order")]
    pub values: Arc<Array3<f64>>,
    /// The influence on the estimated `p0`, `[[cluster, state]]`, when it
    /// was estimated and not every subject started in the same state.
    #[serde(with = "crate::internal::pickle::memory_order::option")]
    pub i0: Option<Array2<f64>>,
}

#[cfg(feature = "python")]
#[pymethods]
impl SurvfitAJInfluence {
    /// Pickle and copy support (see `internal::pickle`).
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<crate::internal::pickle::Reduced<'py>> {
        crate::internal::pickle::reduce(py, self)
    }

    /// The `clusters x times x states` array as a read-only NumPy array.
    #[getter(values)]
    fn values_array<'py>(this: &Bound<'py, Self>) -> Bound<'py, numpy::PyArray3<f64>> {
        // SAFETY: the frozen object owns the array through its `Arc` and
        // never changes it.
        unsafe { readonly_view(&this.get().values, this.as_any()) }
    }

    /// The `clusters x states` matrix as a read-only NumPy array.
    #[getter(i0)]
    fn i0_array<'py>(this: &Bound<'py, Self>) -> Option<Bound<'py, numpy::PyArray2<f64>>> {
        // SAFETY: the frozen object owns the matrix and never changes it.
        this.get()
            .i0
            .as_ref()
            .map(|i0| unsafe { readonly_view(i0, this.as_any()) })
    }
}

/// A `survfitms` object.  Row-major matrices have one row per time; the
/// curves are stacked as in R.  Columns of `n_risk`, `n_event`, `n_censor`,
/// `pstate`, `std_err`, `lower`, `upper` are `states`; columns of
/// `n_transition`, `cumhaz`, `std_chaz` are the observed transitions
/// `hazard_from[k] -> hazard_to[k]` (0-based state indices, R's
/// `"from:to"` column names use 1-based ones).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[pyclass(module = "survival._survival", from_py_object)]
pub struct SurvfitAJResult {
    #[pyo3(get)]
    pub n: Vec<usize>,
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub n_risk: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub n_event: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub n_censor: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub n_enter: Option<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub n_transition: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub counts: Option<SurvfitAJCounts>,
    #[pyo3(get)]
    pub pstate: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub cumhaz: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub std_err: Option<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub std_chaz: Option<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub std_auc: Option<Vec<Vec<f64>>>,
    /// Standard error of the estimated `p0`, one row per curve, when `p0`
    /// was estimated and `time0` is off.
    #[pyo3(get)]
    pub se0: Option<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub lower: Option<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub upper: Option<Vec<Vec<f64>>>,
    /// `p0` of each curve, one row per curve.
    #[pyo3(get)]
    pub p0: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub strata: Option<Vec<usize>>,
    #[pyo3(get)]
    pub strata_codes: Option<Vec<i32>>,
    #[pyo3(get)]
    pub n_id: Vec<usize>,
    #[pyo3(get)]
    pub states: Vec<String>,
    /// Counts of observed transitions, `states x (states + censored)`;
    /// empty once [`SurvfitAJResult::select_states`] dropped it.
    #[pyo3(get)]
    pub transitions: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub hazard_from: Vec<usize>,
    #[pyo3(get)]
    pub hazard_to: Vec<usize>,
    #[pyo3(get)]
    pub logse: bool,
    #[pyo3(get)]
    pub conf_int: f64,
    #[pyo3(get)]
    pub conf_type: String,
    /// `"mright"` or `"mcounting"`.
    #[pyo3(get, name = "type")]
    pub type_: String,
    #[pyo3(get)]
    pub t0: f64,
    #[pyo3(get)]
    pub start_time: Option<f64>,
    #[pyo3(get)]
    pub influence_pstate: Option<Vec<SurvfitAJInfluence>>,
}

impl SurvfitAJResult {
    pub fn n_curves(&self) -> usize {
        self.strata.as_ref().map_or(1, Vec::len)
    }

    /// Row range of each curve in the stacked matrices.
    pub fn curve_ranges(&self) -> Vec<std::ops::Range<usize>> {
        curve_ranges(self.strata.as_deref(), self.time.len())
    }

    /// `fit[curves, ]` (`[.survfitms`): the curves at the given positions of
    /// `strata` (0-based, in the order given) as a result of their own.  A
    /// single curve has no `strata`, like a fit without strata.
    pub fn select_curves(&self, curves: &[usize]) -> SurvivalResult<Self> {
        let ranges = self.curve_ranges();
        check_curve_indices(curves, ranges.len())?;
        let rows: Vec<usize> = curves
            .iter()
            .flat_map(|&curve| ranges[curve].clone())
            .collect();
        let pick = |values: &[Vec<f64>]| select_items(values, &rows);
        let several = curves.len() > 1;
        Ok(Self {
            n: select_items(&self.n, curves),
            time: select_items(&self.time, &rows),
            n_risk: pick(&self.n_risk),
            n_event: pick(&self.n_event),
            n_censor: pick(&self.n_censor),
            n_enter: self.n_enter.as_deref().map(pick),
            n_transition: pick(&self.n_transition),
            counts: self.counts.as_ref().map(|counts| SurvfitAJCounts {
                n_risk: pick(&counts.n_risk),
                n_transition: pick(&counts.n_transition),
                n_censor: pick(&counts.n_censor),
                n_enter: counts.n_enter.as_deref().map(pick),
            }),
            pstate: pick(&self.pstate),
            cumhaz: pick(&self.cumhaz),
            std_err: self.std_err.as_deref().map(pick),
            std_chaz: self.std_chaz.as_deref().map(pick),
            std_auc: self.std_auc.as_deref().map(pick),
            se0: self.se0.as_deref().map(|se0| select_items(se0, curves)),
            lower: self.lower.as_deref().map(pick),
            upper: self.upper.as_deref().map(pick),
            p0: select_items(&self.p0, curves),
            strata: several.then(|| curves.iter().map(|&curve| ranges[curve].len()).collect()),
            strata_codes: self
                .strata_codes
                .as_deref()
                .filter(|_| several)
                .map(|codes| select_items(codes, curves)),
            n_id: select_items(&self.n_id, curves),
            states: self.states.clone(),
            transitions: self.transitions.clone(),
            hazard_from: self.hazard_from.clone(),
            hazard_to: self.hazard_to.clone(),
            logse: self.logse,
            conf_int: self.conf_int,
            conf_type: self.conf_type.clone(),
            type_: self.type_.clone(),
            t0: self.t0,
            start_time: self.start_time,
            influence_pstate: self
                .influence_pstate
                .as_deref()
                .map(|list| select_items(list, curves)),
        })
    }

    /// `fit[, states]` (`[.survfitms`): the columns of the given states
    /// (0-based, in the order given) of every per-state matrix, `p0`, `se0`
    /// and the influence.  Unless every state is kept in its original order
    /// the per-transition parts (`n_transition`, `cumhaz`, `std_chaz`) go, as
    /// R has no consistent way to subscript them; the `transitions` table and
    /// the unweighted `counts` always go.
    pub fn select_states(&self, states: &[usize]) -> SurvivalResult<Self> {
        let nstate = self.states.len();
        if states.is_empty() {
            return Err(SurvivalError::invalid_input("select at least one state"));
        }
        if let Some(state) = states.iter().find(|&&state| state >= nstate) {
            return Err(SurvivalError::invalid_input(format!(
                "state {state} is out of bounds for a fit with {nstate} states"
            )));
        }
        let columns = |matrix: &[Vec<f64>]| -> Vec<Vec<f64>> {
            matrix.iter().map(|row| select_items(row, states)).collect()
        };
        let every_state = states.iter().copied().eq(0..nstate);
        let per_transition = |matrix: &[Vec<f64>]| -> Vec<Vec<f64>> {
            if every_state {
                matrix.to_vec()
            } else {
                vec![Vec::new(); matrix.len()]
            }
        };
        let (hazard_from, hazard_to) = if every_state {
            (self.hazard_from.clone(), self.hazard_to.clone())
        } else {
            (Vec::new(), Vec::new())
        };
        Ok(Self {
            n: self.n.clone(),
            time: self.time.clone(),
            n_risk: columns(&self.n_risk),
            n_event: columns(&self.n_event),
            n_censor: columns(&self.n_censor),
            n_enter: self.n_enter.as_deref().map(columns),
            n_transition: per_transition(&self.n_transition),
            counts: None,
            pstate: columns(&self.pstate),
            cumhaz: per_transition(&self.cumhaz),
            std_err: self.std_err.as_deref().map(columns),
            std_chaz: self.std_chaz.clone().filter(|_| every_state),
            std_auc: self.std_auc.as_deref().map(columns),
            se0: self.se0.as_deref().map(columns),
            lower: self.lower.as_deref().map(columns),
            upper: self.upper.as_deref().map(columns),
            p0: columns(&self.p0),
            strata: self.strata.clone(),
            strata_codes: self.strata_codes.clone(),
            n_id: self.n_id.clone(),
            states: select_items(&self.states, states),
            transitions: Vec::new(),
            hazard_from,
            hazard_to,
            logse: self.logse,
            conf_int: self.conf_int,
            conf_type: self.conf_type.clone(),
            type_: self.type_.clone(),
            t0: self.t0,
            start_time: self.start_time,
            influence_pstate: self.influence_pstate.as_ref().map(|list| {
                list.iter()
                    .map(|influence| {
                        let (nid, ntime, _) = influence.values.dim();
                        SurvfitAJInfluence {
                            cluster: influence.cluster.clone(),
                            values: Arc::new(Array3::from_shape_fn(
                                (nid, ntime, states.len()).f(),
                                |(g, t, k)| influence.values[[g, t, states[k]]],
                            )),
                            i0: influence.i0.as_ref().map(|i0| i0.select(Axis(1), states)),
                        }
                    })
                    .collect()
            }),
        })
    }
}

#[pymethods]
impl SurvfitAJResult {
    /// Pickle and copy support (see `internal::pickle`).
    #[cfg(feature = "python")]
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<crate::internal::pickle::Reduced<'py>> {
        crate::internal::pickle::reduce(py, self)
    }

    #[pyo3(signature=(times=None, censored=false, extend=false, dosum=true))]
    fn summary(
        &self,
        py: Python<'_>,
        times: Option<FloatVec>,
        censored: bool,
        extend: bool,
        dosum: bool,
    ) -> PyResult<Self> {
        let times = times.map(FloatVec::into_inner);
        py.detach(|| {
            summary_survfit_aj_with_counts(self, times.as_deref(), censored, extend, dosum)
        })
        .map_err(Into::into)
    }

    #[pyo3(signature=(scale=1.0, rmean="common"))]
    fn mean_table(&self, py: Python<'_>, scale: f64, rmean: &str) -> PyResult<AJMeanTable> {
        let option = RmeanOption::parse(rmean)?;
        py.detach(|| survmean_aj(self, scale, option))
            .map_err(Into::into)
    }

    /// For each row of `survfit0_aj(self)`, the 0-based row of `self` it
    /// copies, or `-1 - s` for the row at `t0` inserted into curve `s`.
    fn survfit0_rows(&self) -> IntVec {
        IntVec(
            survfit0_aj_rows(self)
                .into_iter()
                .map(|row| row as i32)
                .collect(),
        )
    }

    /// The rows whose `pstate` and `cumhaz` `summary(self, times, censored,
    /// extend)` reports, encoded as in `survfit0_rows`.
    #[pyo3(signature=(times=None, censored=false, extend=false))]
    fn summary_rows(
        &self,
        times: Option<Vec<f64>>,
        censored: bool,
        extend: bool,
    ) -> PyResult<IntVec> {
        let rows = summary_rows(self, times.as_deref(), censored, extend)?;
        Ok(IntVec(rows.into_iter().map(|row| row as i32).collect()))
    }

    /// `survmean2`'s table for curves on this fit's time grid and counts
    /// with a newdata dimension (`survfit.coxphms`): `pstate` is `(ntime,
    /// ndata, nstate)` and `p0` has a row per curve.  The rows run over the
    /// curves fastest, then the newdata rows, then the states.  The table is
    /// one pass over `pstate`, read in place.
    #[cfg(feature = "python")]
    #[pyo3(signature=(pstate, p0, scale=1.0, rmean="common"))]
    fn mean_table_data(
        &self,
        pstate: numpy::PyReadonlyArray3<'_, f64>,
        p0: Vec<Vec<f64>>,
        scale: f64,
        rmean: &str,
    ) -> PyResult<AJMeanTable> {
        let option = RmeanOption::parse(rmean)?;
        let pstate = pstate.as_array();
        let (ntime, ndata, nstate) = pstate.dim();
        if ntime != self.time.len() || nstate != self.states.len() {
            return Err(SurvivalError::invalid_input(
                "pstate must be (ntime, ndata, nstate) on this fit's times and states",
            )
            .into());
        }
        if p0.len() != self.n_curves() || p0.iter().any(|row| row.len() != nstate) {
            return Err(SurvivalError::invalid_input("p0 needs a row per curve").into());
        }
        survmean_coxms(
            self,
            ndata,
            |i, j, state| pstate[(i, j, state)],
            &p0,
            scale,
            option,
        )
        .map_err(Into::into)
    }

    /// `fit[curves, ]`: see [`SurvfitAJResult::select_curves`].
    #[pyo3(name = "select_curves")]
    fn py_select_curves(&self, curves: Vec<usize>) -> PyResult<Self> {
        Ok(self.select_curves(&curves)?)
    }

    /// `fit[, states]`: see [`SurvfitAJResult::select_states`].
    #[pyo3(name = "select_states")]
    fn py_select_states(&self, states: Vec<usize>) -> PyResult<Self> {
        Ok(self.select_states(&states)?)
    }
}

// ---------------------------------------------------------------------------
// survcheck2 / multicheck: states, current states, transitions, flags
// ---------------------------------------------------------------------------

/// What `survcheck2` derives from the data.
pub(crate) struct SurvCheck {
    /// All state names: initial-only states first, then the event states.
    pub states: Vec<String>,
    /// State at the start of each interval (0-based index into `states`),
    /// chained forward through each subject's transitions.
    pub istate: Vec<usize>,
    /// Event state of each interval re-indexed into `states` (1-based, 0 =
    /// censored).
    pub stat2: Vec<usize>,
    /// Transition counts `states x (states + 1)`, the last column being
    /// censored intervals.
    pub transitions: Vec<Vec<f64>>,
}

/// Port of `survcheck2` (`R/survcheck.R`) and `multicheck` (`src/multicheck.c`):
/// the normalised states and current states, the transition table, and the
/// overlap/gap/jump/teleport checks, which are errors here because
/// `survfitAJ` refuses data with any flag set.
pub(crate) fn survcheck2(
    start: Option<&[f64]>,
    time: &[f64],
    state: &[i32],
    ystate: &[String],
    id: &[usize],
    istate: Option<&[usize]>,
    istate_levels: &[String],
) -> SurvivalResult<SurvCheck> {
    let n = time.len();
    // states: initial-state levels that are not destinations, then ystate
    let mut states: Vec<String> = istate_levels
        .iter()
        .filter(|level| !ystate.contains(level))
        .cloned()
        .collect();
    states.extend(ystate.iter().cloned());
    let state_index = |name: &str| {
        states
            .iter()
            .position(|s| s == name)
            .expect("state is listed")
    };
    let cstate2: Vec<usize> = match istate {
        Some(istate) => istate
            .iter()
            .map(|&level| state_index(&istate_levels[level]))
            .collect(),
        None => vec![state_index(&istate_levels[0]); n],
    };
    let stat2: Vec<usize> = state
        .iter()
        .map(|&code| {
            if code == 0 {
                0
            } else {
                state_index(&ystate[code as usize - 1]) + 1
            }
        })
        .collect();
    // multicheck: dupid (last observation), gap, chained current state
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        id[a]
            .cmp(&id[b])
            .then_with(|| time[a].total_cmp(&time[b]))
            .then_with(|| match start {
                Some(start) => start[a].total_cmp(&start[b]),
                None => std::cmp::Ordering::Equal,
            })
            .then_with(|| a.cmp(&b))
    });
    let mut last = vec![false; n];
    let mut gap = vec![0i8; n];
    let mut cstate = vec![0usize; n];
    let mut previous: Option<usize> = None;
    for &ii in &order {
        match previous {
            Some(oldii) if id[oldii] == id[ii] => {
                let entry = start.map_or(0.0, |start| start[ii]);
                gap[ii] = if entry == time[oldii] {
                    0
                } else if entry > time[oldii] {
                    1
                } else {
                    -1
                };
                cstate[ii] = if stat2[oldii] > 0 {
                    stat2[oldii] - 1
                } else {
                    cstate[oldii]
                };
            }
            _ => {
                gap[ii] = 0;
                cstate[ii] = cstate2[ii];
                if let Some(oldii) = previous {
                    last[oldii] = true;
                }
            }
        }
        previous = Some(ii);
    }
    if let Some(oldii) = previous {
        last[oldii] = true;
    }
    // without an istate, counting-process data use the chained state
    let cstate2 = if istate.is_none() && start.is_some() {
        cstate.clone()
    } else {
        cstate2
    };
    let nstate = states.len();
    let mut transitions = vec![vec![0.0; nstate + 1]; nstate];
    for i in 0..n {
        if stat2[i] != 0 || last[i] {
            let to = if stat2[i] == 0 { nstate } else { stat2[i] - 1 };
            transitions[cstate2[i]][to] += 1.0;
        }
    }
    let mismatch = |i: usize| cstate2[i] != cstate[i];
    let overlap = (0..n).filter(|&i| gap[i] < 0).count();
    let gaps = (0..n).filter(|&i| gap[i] > 0 && !mismatch(i)).count();
    let jump = (0..n).filter(|&i| gap[i] > 0 && mismatch(i)).count();
    let teleport = (0..n).filter(|&i| gap[i] == 0 && mismatch(i)).count();
    if overlap > 0 {
        return Err(SurvivalError::invalid_input(
            "a subject has overlapping time intervals",
        ));
    }
    if gaps > 0 || jump > 0 || teleport > 0 {
        return Err(SurvivalError::invalid_input(format!(
            "one or more flags are >0 in survcheck (gap = {gaps}, jump = {jump}, teleport = {teleport})"
        )));
    }
    Ok(SurvCheck {
        states,
        istate: cstate,
        stat2,
        transitions,
    })
}

// ---------------------------------------------------------------------------
// The C kernel
// ---------------------------------------------------------------------------

/// Everything `survfitaj.c` reads.
struct AJKernelData<'a> {
    right: bool,
    time1: &'a [f64],
    time2: &'a [f64],
    /// 1-based state entered, 0 = censored.
    state: &'a [usize],
    sort1: &'a [usize],
    sort2: &'a [usize],
    utime: &'a [f64],
    cstate: &'a [usize],
    wt: &'a [f64],
    /// Cluster code of each row, `0..ngrp`.
    grp: &'a [usize],
    ngrp: usize,
    p0: &'a [f64],
    /// Initial influence, `ngrp x nstate` (zero when `p0` was given),
    /// or an empty matrix when standard errors are disabled.
    i0: &'a Array2<f64>,
    /// 0 = no standard errors, 1 = standard errors, 3 = both.
    sefit: u8,
    entry: bool,
    position: &'a [u8],
    /// `hindx[[from, to]]` = transition column, `None` when it never occurs.
    hindx: &'a Array2<Option<usize>>,
    /// `(from, to)` of each transition column.
    trmat: &'a [(usize, usize)],
    t0: f64,
}

/// Output of the kernel for one curve.
struct AJCurveFit {
    n_risk: Array2<f64>,
    n_event: Array2<f64>,
    n_censor: Array2<f64>,
    n_enter: Option<Array2<f64>>,
    n_transition: Array2<f64>,
    pstate: Array2<f64>,
    cumhaz: Array2<f64>,
    std_err: Option<Array2<f64>>,
    std_chaz: Option<Array2<f64>>,
    std_auc: Option<Array2<f64>>,
    /// `[cluster, time, state]`.
    influence: Option<Array3<f64>>,
}

/// Robust-variance workspaces, allocated only when standard errors are needed.
struct AJInfluence {
    u: Array2<f64>,
    ua: Array2<f64>,
    c: Array2<f64>,
    wg: Array2<f64>,
    uold: Array2<f64>,
    h: Array2<f64>,
    se1: Vec<f64>,
    se2: Vec<f64>,
    se3: Vec<f64>,
}

impl AJInfluence {
    fn new(d: &AJKernelData<'_>) -> Self {
        let nstate = d.p0.len();
        let nhaz = d.trmat.len();
        let mut u = Array2::<f64>::zeros((nstate, d.ngrp));
        let mut se1 = vec![0.0; nstate];
        for j in 0..nstate {
            for g in 0..d.ngrp {
                u[[j, g]] = d.i0[[g, j]];
            }
            se1[j] = u.row(j).iter().map(|v| v * v).sum::<f64>().sqrt();
        }
        Self {
            u,
            ua: Array2::zeros((nstate, d.ngrp)),
            c: Array2::zeros((nhaz, d.ngrp)),
            wg: Array2::zeros((nstate, d.ngrp)),
            uold: Array2::zeros((nstate, d.ngrp)),
            h: Array2::zeros((nstate, nstate)),
            se1,
            se2: vec![0.0; nhaz],
            se3: vec![0.0; nstate],
        }
    }
}

/// Port of `survfitaj` (`src/survfitaj.c`) for one curve.  Count matrices
/// carry the weighted columns first and the unweighted ones after them,
/// as in the C output.
fn aj_kernel(d: &AJKernelData<'_>) -> AJCurveFit {
    let ntime = d.utime.len();
    let nused = d.sort2.len();
    let nstate = d.p0.len();
    let nhaz = d.trmat.len();
    let ngrp = d.ngrp;

    let mut nrisk = Array2::<f64>::zeros((ntime, 2 * nstate));
    let mut nevent = Array2::<f64>::zeros((ntime, nstate));
    let mut ntrans = Array2::<f64>::zeros((ntime, 2 * nhaz));
    let mut ncensor = Array2::<f64>::zeros((ntime, 2 * nstate));
    let mut nenter = d.entry.then(|| Array2::<f64>::zeros((ntime, 2 * nstate)));

    // First pass, in reverse time order, to create all the counts: ntemp
    // is the weighted and unweighted number at risk per state, driven by
    // the current state of each interval.
    {
        let mut ntemp = vec![0.0; 2 * nstate];
        let mut person1 = nused;
        let mut person2 = nused;
        for i in (0..ntime).rev() {
            let ctime = d.utime[i];
            // remove intervals that no longer overlap ctime (time1 >= ctime)
            while person1 > 0 && d.time1[d.sort1[person1 - 1]] >= ctime {
                let i1 = d.sort1[person1 - 1];
                let j = d.cstate[i1];
                ntemp[j] -= d.wt[i1];
                ntemp[j + nstate] -= 1.0;
                if let Some(nenter) = &mut nenter
                    && d.position[i1] & 1 == 1
                {
                    nenter[[i, j]] += d.wt[i1];
                    nenter[[i, j + nstate]] += 1.0;
                }
                person1 -= 1;
            }
            // add intervals that overlap ctime but did not at the prior step
            while person2 > 0 && d.time2[d.sort2[person2 - 1]] >= ctime {
                let i2 = d.sort2[person2 - 1];
                let j = d.cstate[i2];
                ntemp[j] += d.wt[i2];
                ntemp[j + nstate] += 1.0;
                if d.state[i2] > 0 {
                    let k = d.state[i2] - 1;
                    let jk = d.hindx[[j, k]].expect("observed transition is indexed");
                    ntrans[[i, jk]] += d.wt[i2];
                    ntrans[[i, jk + nhaz]] += 1.0;
                    nevent[[i, k]] += d.wt[i2];
                } else if d.position[i2] > 1 {
                    ncensor[[i, j]] += d.wt[i2];
                    ncensor[[i, j + nstate]] += 1.0;
                }
                person2 -= 1;
            }
            for j in 0..2 * nstate {
                nrisk[[i, j]] = ntemp[j];
            }
        }
    }

    if let Some(estimates) = independent::estimates(d, &nrisk, &ntrans) {
        return AJCurveFit {
            n_risk: nrisk,
            n_event: nevent,
            n_censor: ncensor,
            n_enter: nenter,
            n_transition: ntrans,
            pstate: estimates.pstate,
            cumhaz: estimates.cumhaz,
            std_err: Some(estimates.std_err),
            std_chaz: Some(estimates.std_chaz),
            std_auc: Some(estimates.std_auc),
            influence: None,
        };
    }

    let mut pstate = Array2::<f64>::zeros((ntime, nstate));
    let mut cumhaz = Array2::<f64>::zeros((ntime, nhaz));
    let mut phat: Vec<f64> = d.p0.to_vec();
    let mut before = vec![0.0; nstate];
    let mut chaz = vec![0.0; nhaz];
    let se = d.sefit > 0;
    let mut stdp = se.then(|| Array2::<f64>::zeros((ntime, nstate)));
    let mut stdc = se.then(|| Array2::<f64>::zeros((ntime, nhaz)));
    let mut stda = se.then(|| Array2::<f64>::zeros((ntime, nstate)));
    // column-major like R's influence.pstate: one contiguous run of ngrp
    // values per time and state
    let mut usave = (d.sefit > 1).then(|| Array3::<f64>::zeros((ngrp, ntime, nstate).f()));
    // Influence of pstate (U), AUC (UA), cumhaz (C), and risk-set weights
    // by cluster and state are unused for point estimates alone.
    let mut influence = se.then(|| AJInfluence::new(d));

    // Walk forward in time and compute the AJ; see the methods document,
    // Andersen-Gill: influence.
    let mut person1 = 0;
    let mut person2 = 0;
    for i in 0..ntime {
        let ctime = d.utime[i];
        let mut skip_update = false;
        if let Some(AJInfluence {
            u,
            ua,
            c,
            wg,
            uold,
            h,
            se1,
            se2,
            se3,
        }) = &mut influence
        {
            let delta = if i > 0 {
                ctime - d.utime[i - 1]
            } else {
                ctime - d.t0
            };
            for j in 0..nstate {
                let mut temp = 0.0;
                for g in 0..ngrp {
                    ua[[j, g]] += delta * u[[j, g]];
                    temp += ua[[j, g]] * ua[[j, g]];
                }
                se3[j] = temp.sqrt();
            }
            while person1 < nused {
                let i1 = d.sort1[person1];
                if d.time1[i1] < ctime {
                    wg[[d.cstate[i1], d.grp[i1]]] += d.wt[i1];
                    person1 += 1;
                } else {
                    break;
                }
            }
            while person2 < nused {
                let i2 = d.sort2[person2];
                if d.time2[i2] < ctime {
                    wg[[d.cstate[i2], d.grp[i2]]] -= d.wt[i2];
                    person2 += 1;
                } else {
                    break;
                }
            }
            // the dN part of C, and H - I
            let mut tdeath = 0;
            h.fill(0.0);
            for &i2 in &d.sort2[person2..] {
                if d.time2[i2] > ctime {
                    break;
                }
                if d.time2[i2] == ctime && d.state[i2] > 0 {
                    tdeath += 1;
                    let j = d.cstate[i2];
                    let k = d.state[i2] - 1;
                    let jk = d.hindx[[j, k]].expect("observed transition is indexed");
                    let g = d.grp[i2];
                    c[[jk, g]] += d.wt[i2] / nrisk[[i, j]];
                    if j != k {
                        h[[j, j]] -= d.wt[i2] / nrisk[[i, j]];
                        h[[j, k]] += d.wt[i2] / nrisk[[i, j]];
                    }
                }
            }
            if tdeath == 0 {
                skip_update = true; // no events: C and U do not change
            } else {
                // U = U + U H, using the pre-update rows as multipliers
                uold.assign(u);
                for j in 0..nstate {
                    if h[[j, j]] == 0.0 {
                        continue;
                    }
                    for k in 0..nstate {
                        if k != j && h[[j, k]] != 0.0 {
                            for g in 0..ngrp {
                                u[[k, g]] += uold[[j, g]] * h[[j, k]];
                            }
                        }
                    }
                    for g in 0..ngrp {
                        u[[j, g]] += uold[[j, g]] * h[[j, j]];
                    }
                }
                // the dN part of U
                for &i2 in &d.sort2[person2..] {
                    if d.time2[i2] > ctime {
                        break;
                    }
                    if d.time2[i2] == ctime && d.state[i2] > 0 {
                        let j = d.cstate[i2];
                        let k = d.state[i2] - 1;
                        let g = d.grp[i2];
                        if j != k {
                            let term = d.wt[i2] * phat[j] / nrisk[[i, j]];
                            u[[j, g]] -= term;
                            u[[k, g]] += term;
                        }
                    }
                }
                // the phat part of the influence
                for jk in 0..nhaz {
                    if ntrans[[i, jk]] > 0.0 {
                        let (j, k) = d.trmat[jk];
                        let haz = ntrans[[i, jk]] / nrisk[[i, j]];
                        let htemp = haz / nrisk[[i, j]]; // scaled hazard
                        for g in 0..ngrp {
                            if wg[[j, g]] > 0.0 {
                                c[[jk, g]] -= wg[[j, g]] * htemp;
                            }
                        }
                        if j != k {
                            for g in 0..ngrp {
                                if wg[[j, g]] > 0.0 {
                                    let term = wg[[j, g]] * phat[j] * htemp;
                                    u[[j, g]] += term;
                                    u[[k, g]] -= term;
                                }
                            }
                        }
                    }
                }
                for (j, se) in se1.iter_mut().enumerate() {
                    *se = u.row(j).iter().map(|v| v * v).sum::<f64>().sqrt();
                }
                for (jk, se) in se2.iter_mut().enumerate() {
                    *se = c.row(jk).iter().map(|v| v * v).sum::<f64>().sqrt();
                }
            }
        }
        if !skip_update {
            // update phat; p(t-) was needed for the IJ above
            before.copy_from_slice(&phat);
            for jk in 0..nhaz {
                if ntrans[[i, jk]] > 0.0 {
                    let (j, k) = d.trmat[jk];
                    let haz = ntrans[[i, jk]] / nrisk[[i, j]];
                    chaz[jk] += haz;
                    phat[j] -= before[j] * haz;
                    phat[k] += before[j] * haz;
                }
            }
        }
        // save out the results
        for j in 0..nstate {
            pstate[[i, j]] = phat[j];
            if let (Some(stdp), Some(stda), Some(influence)) = (&mut stdp, &mut stda, &influence) {
                stdp[[i, j]] = influence.se1[j];
                stda[[i, j]] = influence.se3[j];
            }
        }
        for jk in 0..nhaz {
            cumhaz[[i, jk]] = chaz[jk];
            if let (Some(stdc), Some(influence)) = (&mut stdc, &influence) {
                stdc[[i, jk]] = influence.se2[jk];
            }
        }
        if let (Some(usave), Some(influence)) = (&mut usave, &influence) {
            for j in 0..nstate {
                usave.slice_mut(s![.., i, j]).assign(&influence.u.row(j));
            }
        }
    }

    AJCurveFit {
        n_risk: nrisk,
        n_event: nevent,
        n_censor: ncensor,
        n_enter: nenter,
        n_transition: ntrans,
        pstate,
        cumhaz,
        std_err: stdp,
        std_chaz: stdc,
        std_auc: stda,
        influence: usave,
    }
}

// ---------------------------------------------------------------------------
// The R-level driver
// ---------------------------------------------------------------------------

fn rows_to_vec(matrix: &Array2<f64>, columns: std::ops::Range<usize>) -> Vec<Vec<f64>> {
    matrix
        .outer_iter()
        .map(|row| row.slice(ndarray::s![columns.clone()]).to_vec())
        .collect()
}

/// The data after the steps `survfitAJ` and `residuals.survfit` share:
/// `timefix`, the dummy id, the normalised states of `survcheck2`.
pub(crate) struct AJPrepared {
    pub start: Option<Vec<f64>>,
    pub time: Vec<f64>,
    pub weights: Vec<f64>,
    /// Subject codes in order of first appearance.
    pub id: Vec<usize>,
    pub check: SurvCheck,
}

pub(crate) fn aj_prepare(data: &SurvfitAJData, timefix: bool) -> SurvivalResult<AJPrepared> {
    let n_all = data.time.len();
    let counting = data.start.is_some();
    let (start, time) = if timefix {
        let (start, time) = aeq_counting(data.start.as_deref(), &data.time)?;
        if let Some(start) = &start
            && start.iter().zip(&time).any(|(s, t)| s == t)
        {
            return Err(SurvivalError::invalid_input(
                "aeqSurv exception, an interval has effective length 0",
            ));
        }
        (start, time)
    } else {
        (data.start.clone(), data.time.clone())
    };
    if counting && data.id.is_none() {
        return Err(SurvivalError::invalid_input(
            "an id statement is required for start,stop data",
        ));
    }
    let weights: Vec<f64> = data.weights.clone().unwrap_or_else(|| vec![1.0; n_all]);
    // id: a dummy value when absent
    let (id, _) = match &data.id {
        Some(id) => first_appearance_codes(id),
        None => ((0..n_all).collect(), (0..n_all as i64).collect()),
    };
    // istate levels: as given, else alphabetical, else "(s0)"
    let istate_levels: Vec<String> = match (&data.istate, &data.istate_levels) {
        (Some(_), Some(levels)) => levels.clone(),
        (Some(istate), None) => {
            let mut levels = istate.clone();
            levels.sort();
            levels.dedup();
            levels
        }
        (None, _) => vec!["(s0)".to_string()],
    };
    let istate_codes: Option<Vec<usize>> = data.istate.as_ref().map(|istate| {
        istate
            .iter()
            .map(|value| {
                istate_levels
                    .iter()
                    .position(|level| level == value)
                    .expect("istate values are among the levels")
            })
            .collect()
    });
    let check = survcheck2(
        start.as_deref(),
        &time,
        &data.state,
        &data.states,
        &id,
        istate_codes.as_deref(),
        &istate_levels,
    )?;
    Ok(AJPrepared {
        start,
        time,
        weights,
        id,
        check,
    })
}

/// Port of `survfitAJ` (`R/survfitAJ.R`).
pub fn survfitaj(
    data: &SurvfitAJData,
    options: &SurvfitAJOptions,
) -> SurvivalResult<SurvfitAJResult> {
    data.validate()?;
    validate_conf_int(options.conf_int)?;
    let n_all = data.time.len();
    let counting = data.start.is_some();
    let has_id = data.id.is_some();
    let AJPrepared {
        start,
        time,
        weights,
        id: id_codes,
        check,
    } = aj_prepare(data, options.timefix)?;
    let states = check.states;
    let nstate = states.len();
    if let Some(p0) = &options.p0 {
        if p0.len() != nstate {
            return Err(SurvivalError::invalid_input("wrong length for p0"));
        }
        let total: f64 = p0.iter().sum();
        if p0.iter().any(|v| !v.is_finite()) || (total - 1.0).abs() > 1.5e-8 {
            return Err(SurvivalError::invalid_input(
                "p0 must be a numeric vector that adds to 1",
            ));
        }
    }
    // curves: the strata levels come from the full data
    let (strata_levels, x_all) = strata_index(data.strata.as_deref(), n_all);
    // start.time: remove rows that end before it
    let rows: Vec<usize> = match options.start_time {
        Some(start_time) => {
            if !start_time.is_finite() {
                return Err(SurvivalError::invalid_input(
                    "start.time must be a single numeric value",
                ));
            }
            for keep in rows_by_curve(&x_all, strata_levels.len()) {
                let cmax = keep.iter().map(|&i| time[i]).fold(f64::NAN, f64::max);
                if cmax <= start_time {
                    return Err(SurvivalError::invalid_input(
                        "start.time has removed all the observations from at least one curve",
                    ));
                }
            }
            (0..n_all).filter(|&i| time[i] >= start_time).collect()
        }
        None => (0..n_all).collect(),
    };
    let n = rows.len();
    let pick = |values: &[f64]| -> Vec<f64> { rows.iter().map(|&i| values[i]).collect() };
    let time = pick(&time);
    let stop_start = start.as_deref().map(pick);
    let weights = pick(&weights);
    let stat2: Vec<usize> = rows.iter().map(|&i| check.stat2[i]).collect();
    let istate: Vec<usize> = rows.iter().map(|&i| check.istate[i]).collect();
    let x: Vec<usize> = rows.iter().map(|&i| x_all[i]).collect();
    let id: Vec<usize> = rows.iter().map(|&i| id_codes[i]).collect();

    // cluster: the explicit cluster, else the id, else each observation,
    // numbered in order of first appearance
    let influence = options.influence && options.se_fit;
    let cluster: Vec<usize> = match data.cluster.as_ref().or(data.id.as_ref()) {
        Some(labels) => {
            let subset: Vec<i64> = rows.iter().map(|&i| labels[i]).collect();
            first_appearance_codes(&subset).0
        }
        None => (0..n).collect(),
    };

    // does everyone start in the same state?
    let (samestate, stemp) = if !counting {
        (istate.iter().all(|&s| s == istate[0]), istate[0])
    } else {
        let start = stop_start.as_deref().expect("counting data");
        let mut indx: Vec<usize> = (0..n).collect();
        indx.sort_by(|&a, &b| {
            x[a].cmp(&x[b])
                .then_with(|| id[a].cmp(&id[b]))
                .then_with(|| start[a].total_cmp(&start[b]))
                .then_with(|| a.cmp(&b))
        });
        let mut first_states = Vec::new();
        let mut seen: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
        for &i in &indx {
            if seen.insert((x[i], id[i])) {
                first_states.push(istate[i]);
            }
        }
        (
            first_states.iter().all(|&s| s == first_states[0]),
            first_states[0],
        )
    };
    let mut p0_common: Option<Vec<f64>> = options.p0.clone();
    if p0_common.is_none() && samestate {
        let mut p0 = vec![0.0; nstate];
        p0[stemp] = 1.0;
        p0_common = Some(p0);
    }
    // time 0 for all the curves
    let t0 = if let Some(start_time) = options.start_time {
        start_time
    } else if !counting {
        // min(c(0, Y[, 2])) with Y[, 2] the status column: always 0
        0.0
    } else {
        let start = stop_start.as_deref().expect("counting data");
        if samestate {
            start.iter().copied().fold(f64::INFINITY, f64::min)
        } else {
            // R first tests whether every subject enters at the same time,
            // but compares the named tapply() result with an unnamed
            // vector, so that branch never applies: the default is the
            // first event time, the hardest case (curves on an age scale)
            let t0 = (0..n)
                .filter(|&i| stat2[i] != 0)
                .map(|i| time[i])
                .fold(f64::INFINITY, f64::min);
            for keep in rows_by_curve(&x, strata_levels.len()) {
                let cmax = keep.iter().map(|&i| time[i]).fold(f64::NAN, f64::max);
                let cmin = keep.iter().map(|&i| start[i]).fold(f64::NAN, f64::min);
                if cmax < t0 || cmin >= t0 {
                    return Err(SurvivalError::invalid_input(format!(
                        "no obs overlap the default start time of {t0} in at least one curve; specify a start.time"
                    )));
                }
            }
            t0
        }
    };
    // a dummy start column for competing risks data
    let (time1, position, entry): (Vec<f64>, Vec<u8>, bool) = match &stop_start {
        Some(start) => (
            start.clone(),
            survflag(start, &time, &id, &x),
            options.entry && has_id,
        ),
        None => {
            let min_y = time.iter().copied().fold(f64::INFINITY, f64::min);
            let dummy = if min_y > 0.0 { 0.0 } else { 2.0 * min_y - 1.0 };
            (vec![dummy; n], vec![3; n], false)
        }
    };
    // hazard indexing: the transitions that occur, in column-major order
    let mut hindx = Array2::<Option<usize>>::from_elem((nstate, nstate), None);
    let mut trmat = Vec::new();
    for to in 0..nstate {
        for from in 0..nstate {
            if check.transitions[from][to] > 0.0 {
                hindx[[from, to]] = Some(trmat.len());
                trmat.push((from, to));
            }
        }
    }
    let nhaz = trmat.len();
    let sefit: u8 = 2 * u8::from(influence) + u8::from(options.se_fit);

    // one kernel call per curve
    let n_curves = strata_levels.len();
    struct Curve {
        code: i32,
        n: usize,
        n_id: usize,
        utime: Vec<f64>,
        p0: Vec<f64>,
        sd0: Option<Vec<f64>>,
        clusters: Vec<i64>,
        i0: Option<Array2<f64>>,
        fit: AJCurveFit,
    }
    let mut curves: Vec<Curve> = Vec::with_capacity(n_curves);
    let mut c2 = vec![0usize; n];
    // R's logical atrisk, set for one curve at a time, and its row offset
    // n2 <- cumsum(c(0, n.per.curve))
    let mut is_atrisk = vec![false; n];
    let mut row_offset = 0;
    let single = n_curves == 1;
    // Initial-state estimation uses the full data's earliest entry time.
    // Right-censored curves do not need this scan, and counting curves share it.
    let min_start = (counting && p0_common.is_none())
        .then(|| time1.iter().copied().fold(f64::INFINITY, f64::min));
    for (&code, keep) in strata_levels.iter().zip(rows_by_curve(&x, n_curves)) {
        let curve_offset = row_offset;
        row_offset += keep.len();
        if keep.is_empty() {
            continue;
        }
        let sort1 = ordered_subset(&keep, &time1, single);
        let sort2 = ordered_subset(&keep, &time, single);
        // R survival 3.8-12 reports transitions and final stops. With entry
        // counts, include subject starts within each curve as well.
        // Censored continuation boundaries do not add rows.
        let mut utime: Vec<f64> = keep
            .iter()
            .filter(|&&i| stat2[i] != 0 || position[i] > 1)
            .map(|&i| time[i])
            .collect();
        if entry {
            utime.extend(
                keep.iter()
                    .filter(|&&i| position[i] & 1 == 1)
                    .map(|&i| time1[i]),
            );
        }
        utime.sort_by(f64::total_cmp);
        utime.dedup();
        let utime: Vec<f64> = if options.time0 {
            std::iter::once(t0)
                .chain(utime.into_iter().filter(|&t| t > t0))
                .collect()
        } else {
            utime.into_iter().filter(|&t| t >= t0).collect()
        };
        // clusters renumbered per curve, in order of appearance
        let subset: Vec<usize> = keep.iter().map(|&i| cluster[i]).collect();
        let (renumbered, unique) = first_appearance_codes(&subset);
        for (&i, code) in keep.iter().zip(renumbered) {
            c2[i] = code;
        }
        let nclust = unique.len();
        // R names the influence rows `uclust`: the clusters' numbers
        let clusters: Vec<i64> = unique.iter().map(|&code| code as i64 + 1).collect();
        let n_id = {
            let mut ids: Vec<usize> = keep.iter().map(|&i| id[i]).collect();
            ids.sort_unstable();
            ids.dedup();
            ids.len()
        };
        // p0 per curve, from the distribution of states at t0, with its
        // (clustered, weighted) influence U0
        let mut u0 = Array2::<f64>::zeros((if sefit > 0 { nclust } else { 0 }, nstate).f());
        let mut sd0 = None;
        let mut has_i0 = false;
        let p00: Vec<f64> = match &p0_common {
            Some(p0) => p0.clone(),
            None => {
                let start = &time1;
                let atrisk: Vec<usize> = keep
                    .iter()
                    .copied()
                    .filter(|&i| {
                        if !counting {
                            true
                        } else if Some(t0) == min_start {
                            start[i] <= t0 && time[i] >= t0
                        } else {
                            start[i] < t0 && time[i] >= t0
                        }
                    })
                    .collect();
                if atrisk.is_empty() {
                    return Err(SurvivalError::invalid_input(
                        "no one at risk for one of the curves, at the default time 0; specify a start.time",
                    ));
                }
                let wtsum: f64 = atrisk.iter().map(|&i| weights[i]).sum();
                let mut p00 = vec![0.0; nstate];
                for &i in &atrisk {
                    p00[istate[i]] += weights[i];
                }
                for value in &mut p00 {
                    *value /= wtsum;
                }
                if p00.iter().all(|&p| p < 1.0) && sefit > 0 {
                    // survfitAJ.R computes the per-observation influence over
                    // rows `indx`, the curve's positions in the sort order,
                    // used as row numbers of the data; with several curves
                    // those are not the curve's own rows.  Kept as R does it:
                    // row offset + k contributes to the cluster of the k-th
                    // row of the curve.
                    for &i in &atrisk {
                        is_atrisk[i] = true;
                    }
                    for (k, &curve_row) in keep.iter().enumerate() {
                        let row = curve_offset + k;
                        if !is_atrisk[row] {
                            continue;
                        }
                        for j in 0..nstate {
                            let indicator = if istate[row] == j { 1.0 } else { 0.0 };
                            u0[[c2[curve_row], j]] += weights[row] * (indicator - p00[j]) / wtsum;
                        }
                    }
                    for &i in &atrisk {
                        is_atrisk[i] = false;
                    }
                    sd0 = Some(
                        (0..nstate)
                            .map(|j| u0.column(j).iter().map(|v| v * v).sum::<f64>().sqrt())
                            .collect(),
                    );
                    has_i0 = u0.iter().any(|&v| v != 0.0);
                }
                p00
            }
        };
        let kernel_data = AJKernelData {
            right: !counting,
            time1: &time1,
            time2: &time,
            state: &stat2,
            sort1: &sort1,
            sort2: &sort2,
            utime: &utime,
            cstate: &istate,
            wt: &weights,
            grp: &c2,
            ngrp: nclust,
            p0: &p00,
            i0: &u0,
            sefit,
            entry,
            position: &position,
            hindx: &hindx,
            trmat: &trmat,
            t0,
        };
        let fit = aj_kernel(&kernel_data);
        curves.push(Curve {
            code,
            n: keep.len(),
            n_id,
            utime,
            p0: p00,
            sd0,
            clusters,
            i0: has_i0.then_some(u0),
            fit,
        });
    }

    // stack the curves into a survfitms object
    let total: usize = curves.iter().map(|c| c.utime.len()).sum();
    let addcounts = weights.iter().any(|&w| w != 1.0);
    let mut result = SurvfitAJResult {
        n: curves.iter().map(|c| c.n).collect(),
        time: Vec::with_capacity(total),
        n_risk: Vec::with_capacity(total),
        n_event: Vec::with_capacity(total),
        n_censor: Vec::with_capacity(total),
        n_enter: entry.then(|| Vec::with_capacity(total)),
        n_transition: Vec::with_capacity(total),
        counts: addcounts.then(|| SurvfitAJCounts {
            n_risk: Vec::with_capacity(total),
            n_transition: Vec::with_capacity(total),
            n_censor: Vec::with_capacity(total),
            n_enter: entry.then(|| Vec::with_capacity(total)),
        }),
        pstate: Vec::with_capacity(total),
        cumhaz: Vec::with_capacity(total),
        std_err: options.se_fit.then(|| Vec::with_capacity(total)),
        std_chaz: options.se_fit.then(|| Vec::with_capacity(total)),
        std_auc: options.se_fit.then(|| Vec::with_capacity(total)),
        se0: None,
        lower: None,
        upper: None,
        p0: curves.iter().map(|c| c.p0.clone()).collect(),
        strata: (n_curves > 1).then(|| curves.iter().map(|c| c.utime.len()).collect()),
        strata_codes: (n_curves > 1).then(|| curves.iter().map(|c| c.code).collect()),
        n_id: curves.iter().map(|c| c.n_id).collect(),
        states: states.clone(),
        transitions: check.transitions.clone(),
        hazard_from: trmat.iter().map(|&(from, _)| from).collect(),
        hazard_to: trmat.iter().map(|&(_, to)| to).collect(),
        logse: false,
        conf_int: options.conf_int,
        conf_type: options.conf_type.as_str().to_string(),
        type_: if counting { "mcounting" } else { "mright" }.to_string(),
        t0,
        start_time: options.start_time,
        influence_pstate: influence.then(Vec::new),
    };
    if options.se_fit && !options.time0 && curves.iter().all(|c| c.sd0.is_some()) {
        result.se0 = Some(
            curves
                .iter()
                .map(|c| c.sd0.clone().expect("checked"))
                .collect(),
        );
    }
    for curve in curves {
        let fit = curve.fit;
        result.time.extend_from_slice(&curve.utime);
        result.n_risk.extend(rows_to_vec(&fit.n_risk, 0..nstate));
        result.n_event.extend(rows_to_vec(&fit.n_event, 0..nstate));
        result
            .n_censor
            .extend(rows_to_vec(&fit.n_censor, 0..nstate));
        result
            .n_transition
            .extend(rows_to_vec(&fit.n_transition, 0..nhaz));
        if let (Some(target), Some(source)) = (&mut result.n_enter, &fit.n_enter) {
            target.extend(rows_to_vec(source, 0..nstate));
        }
        if let Some(counts) = &mut result.counts {
            counts
                .n_risk
                .extend(rows_to_vec(&fit.n_risk, nstate..2 * nstate));
            counts
                .n_transition
                .extend(rows_to_vec(&fit.n_transition, nhaz..2 * nhaz));
            counts
                .n_censor
                .extend(rows_to_vec(&fit.n_censor, nstate..2 * nstate));
            if let (Some(target), Some(source)) = (&mut counts.n_enter, &fit.n_enter) {
                target.extend(rows_to_vec(source, nstate..2 * nstate));
            }
        }
        result.pstate.extend(rows_to_vec(&fit.pstate, 0..nstate));
        result.cumhaz.extend(rows_to_vec(&fit.cumhaz, 0..nhaz));
        for (target, source) in [
            (&mut result.std_err, &fit.std_err),
            (&mut result.std_auc, &fit.std_auc),
        ] {
            if let (Some(target), Some(source)) = (target, source) {
                target.extend(rows_to_vec(source, 0..nstate));
            }
        }
        if let (Some(target), Some(source)) = (&mut result.std_chaz, &fit.std_chaz) {
            target.extend(rows_to_vec(source, 0..nhaz));
        }
        if let (Some(list), Some(values)) = (&mut result.influence_pstate, fit.influence) {
            list.push(SurvfitAJInfluence {
                cluster: curve.clusters,
                values: Arc::new(values),
                i0: if options.time0 { None } else { curve.i0 },
            });
        }
    }
    // confidence bands, element by element on the pstate matrix
    if options.se_fit && options.conf_type != ConfType::None {
        let std_err = result.std_err.as_ref().expect("se.fit keeps std.err");
        let p: Vec<f64> = result.pstate.iter().flatten().copied().collect();
        let se: Vec<f64> = std_err.iter().flatten().copied().collect();
        let bands = survfit_confint(
            &p,
            &se,
            false,
            options.conf_type,
            options.conf_int,
            None,
            true,
        )?;
        let reshape = |flat: Vec<f64>| -> Vec<Vec<f64>> {
            flat.chunks(nstate.max(1)).map(<[f64]>::to_vec).collect()
        };
        result.lower = Some(reshape(bands.lower));
        result.upper = Some(reshape(bands.upper));
    }
    Ok(result)
}

/// Python binding of [`survfitaj`].
#[pyfunction(name = "survfitaj")]
#[pyo3(signature = (time, state, states, start=None, weights=None, strata=None, id=None, istate=None, istate_levels=None, cluster=None, se_fit=true, conf_int=0.95, conf_type="log", influence=false, start_time=None, p0=None, entry=false, time0=false, timefix=true))]
#[allow(clippy::too_many_arguments)]
pub fn survfitaj_py(
    py: Python<'_>,
    time: FloatVec,
    state: IntVec,
    states: Vec<String>,
    start: Option<FloatVec>,
    weights: Option<FloatVec>,
    strata: Option<IntVec>,
    id: Option<Vec<i64>>,
    istate: Option<Vec<String>>,
    istate_levels: Option<Vec<String>>,
    cluster: Option<Vec<i64>>,
    se_fit: bool,
    conf_int: f64,
    conf_type: &str,
    influence: bool,
    start_time: Option<f64>,
    p0: Option<Vec<f64>>,
    entry: bool,
    time0: bool,
    timefix: bool,
) -> PyResult<SurvfitAJResult> {
    let data = SurvfitAJData::try_new(
        start.map(FloatVec::into_inner),
        time.into_inner(),
        state.into_inner(),
        states,
        weights.map(FloatVec::into_inner),
        strata.map(IntVec::into_inner),
        id,
        istate,
        istate_levels,
        cluster,
    )?;
    let options = SurvfitAJOptions {
        se_fit,
        conf_int,
        conf_type: ConfType::parse(conf_type)?,
        influence,
        start_time,
        p0,
        entry,
        time0,
        timefix,
    };
    Ok(py.detach(|| survfitaj(&data, &options))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs().max(1.0)
    }

    fn names(values: &[&str]) -> Vec<String> {
        values.iter().map(|s| s.to_string()).collect()
    }

    /// The synthetic_ties_mstate fixture frame.
    fn ties_data() -> SurvfitAJData {
        let time = vec![
            1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 6.0, 7.0, 8.0, 9.0, 9.0,
        ];
        let event = [
            "a", "b", "censor", "b", "a", "b", "censor", "b", "a", "b", "a", "censor", "a",
            "censor", "a", "b",
        ];
        let state: Vec<i32> = event
            .iter()
            .map(|e| match *e {
                "a" => 1,
                "b" => 2,
                _ => 0,
            })
            .collect();
        SurvfitAJData::try_new(
            None,
            time,
            state,
            names(&["a", "b"]),
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap()
    }

    /// The synthetic_mstate fixture frame (istate, counting process).
    fn istate_data() -> SurvfitAJData {
        let tstart = vec![0.0, 3.0, 0.0, 2.0, 0.0, 0.0, 5.0, 0.0, 0.0, 4.0, 0.0, 0.0];
        let tstop = vec![3.0, 8.0, 2.0, 9.0, 6.0, 5.0, 7.0, 4.0, 4.0, 10.0, 7.0, 9.0];
        let event = [
            "ill", "dead", "ill", "well", "dead", "ill", "dead", "censor", "ill", "censor", "dead",
            "censor",
        ];
        let istate = [
            "well", "ill", "well", "ill", "well", "ill", "ill", "well", "well", "ill", "well",
            "ill",
        ];
        let id = vec![1, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8];
        let state: Vec<i32> = event
            .iter()
            .map(|e| match *e {
                "well" => 1,
                "ill" => 2,
                "dead" => 3,
                _ => 0,
            })
            .collect();
        SurvfitAJData::try_new(
            Some(tstart),
            tstop,
            state,
            names(&["well", "ill", "dead"]),
            None,
            None,
            Some(id),
            Some(names(&istate)),
            Some(names(&["well", "ill", "dead"])),
            None,
        )
        .unwrap()
    }

    #[test]
    fn competing_risks_match_r() {
        // survfit(Surv(time, event) ~ 1, synthetic_ties_mstate)
        let fit = survfitaj(&ties_data(), &SurvfitAJOptions::default()).unwrap();
        assert_eq!(fit.states, names(&["(s0)", "a", "b"]));
        assert_eq!(fit.n, vec![16]);
        assert_eq!(fit.p0, vec![vec![1.0, 0.0, 0.0]]);
        assert_eq!(fit.t0, 0.0);
        assert_eq!(fit.type_, "mright");
        assert_eq!(fit.time[..2], [1.0, 2.0]);
        assert_eq!(fit.n_risk[0], vec![16.0, 0.0, 0.0]);
        assert_eq!(fit.n_risk[1], vec![13.0, 0.0, 0.0]);
        assert_eq!(fit.n_event[0], vec![0.0, 1.0, 1.0]);
        assert_eq!(fit.n_censor[0], vec![1.0, 0.0, 0.0]);
        assert_eq!(fit.hazard_from, vec![0, 0]);
        assert_eq!(fit.hazard_to, vec![1, 2]);
        assert_eq!(fit.n_transition[0], vec![1.0, 1.0]);
        assert!(close(fit.pstate[0][0], 0.875, 1e-12));
        assert!(close(fit.pstate[1][1], 0.129807692307692, 1e-12));
        assert!(close(fit.cumhaz[1][0], 0.139423076923077, 1e-12));
        let se = fit.std_err.as_ref().unwrap();
        assert!(close(se[0][0], 0.0826797284707685, 1e-10));
        assert!(close(se[1][1], 0.0857820274553689, 1e-10));
        let sc = fit.std_chaz.as_ref().unwrap();
        assert!(close(sc[1][0], 0.0955201706587362, 1e-10));
        let lower = fit.lower.as_ref().unwrap();
        assert!(close(lower[0][0], 0.727071409585483, 1e-10));
        assert!(close(lower[1][1], 0.0355461834550143, 1e-10));
        assert_eq!(fit.upper.as_ref().unwrap()[0][0], 1.0);
        assert!(!fit.logse);
        assert_eq!(fit.transitions[0], vec![0.0, 6.0, 6.0, 4.0]);
    }

    #[test]
    fn istate_counting_data_matches_r() {
        // survfit(Surv(tstart, tstop, event) ~ 1, synthetic_mstate, id = id, istate = istate)
        let fit = survfitaj(&istate_data(), &SurvfitAJOptions::default()).unwrap();
        assert_eq!(fit.states, names(&["well", "ill", "dead"]));
        assert_eq!(fit.t0, 2.0);
        assert_eq!(fit.p0, vec![vec![0.75, 0.25, 0.0]]);
        assert_eq!(fit.type_, "mcounting");
        assert_eq!(fit.time[..2], [2.0, 3.0]);
        assert_eq!(fit.n_risk[0], vec![6.0, 2.0, 0.0]);
        assert_eq!(fit.n_risk[1], vec![5.0, 3.0, 0.0]);
        assert_eq!(fit.hazard_from, vec![1, 0, 1, 0, 1]);
        assert_eq!(fit.hazard_to, vec![0, 1, 1, 2, 2]);
        assert!(close(fit.pstate[0][0], 0.625, 1e-12));
        assert!(close(fit.pstate[1][1], 0.5, 1e-12));
        assert!(close(fit.cumhaz[1][1], 0.366666666666667, 1e-12));
        let se = fit.std_err.as_ref().unwrap();
        assert!(close(se[0][0], 0.171163299220364, 1e-10));
        assert!(close(se[1][0], 0.176776695296637, 1e-10));
        assert!(close(
            fit.std_chaz.as_ref().unwrap()[1][1],
            0.234836428494704,
            1e-10
        ));
        assert!(close(
            fit.lower.as_ref().unwrap()[0][1],
            0.153289601742952,
            1e-10
        ));
        assert!(fit.se0.is_some());
        assert_eq!(fit.transitions[0], vec![0.0, 3.0, 2.0, 1.0]);
        assert_eq!(fit.transitions[1], vec![1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn influence_reproduces_the_standard_errors() {
        for data in [ties_data(), istate_data()] {
            let fit = survfitaj(
                &data,
                &SurvfitAJOptions {
                    influence: true,
                    ..Default::default()
                },
            )
            .unwrap();
            let influence = &fit.influence_pstate.as_ref().unwrap()[0];
            assert!(influence.values.t().is_standard_layout(), "column-major");
            let se = fit.std_err.as_ref().unwrap();
            for (t, row) in se.iter().enumerate() {
                for (j, &expected) in row.iter().enumerate() {
                    let norm = influence
                        .values
                        .slice(s![.., t, j])
                        .iter()
                        .map(|v| v * v)
                        .sum::<f64>()
                        .sqrt();
                    assert!(close(norm, expected, 1e-10), "{norm} != {expected}");
                }
            }
        }
    }

    #[test]
    fn point_estimates_preserve_initial_states_clusters_and_options() {
        for mut data in [ties_data(), istate_data()] {
            let n = data.time.len();
            data.weights = Some((0..n).map(|row| (row % 5 + 1) as f64 / 3.0).collect());
            data.cluster = Some((0..n).map(|row| (row % 3) as i64).collect());
            data.strata = Some(match &data.id {
                Some(id) => id.iter().map(|id| (id % 2) as i32).collect(),
                None => (0..n).map(|row| (row % 2) as i32).collect(),
            });
            for start_time in [None, Some(4.0)] {
                for time0 in [false, true] {
                    for p0 in [None, Some(vec![0.5, 0.25, 0.25])] {
                        let options = SurvfitAJOptions {
                            start_time,
                            time0,
                            p0,
                            entry: true,
                            influence: true,
                            ..Default::default()
                        };
                        let reference = survfitaj(&data, &options).unwrap();
                        let point_estimates = survfitaj(
                            &data,
                            &SurvfitAJOptions {
                                se_fit: false,
                                ..options
                            },
                        )
                        .unwrap();
                        let mut expected = serde_json::to_value(reference).unwrap();
                        let actual = serde_json::to_value(point_estimates).unwrap();
                        for field in [
                            "std_err",
                            "std_chaz",
                            "std_auc",
                            "se0",
                            "lower",
                            "upper",
                            "influence_pstate",
                        ] {
                            assert!(actual[field].is_null(), "{field}");
                            expected[field] = serde_json::Value::Null;
                        }
                        assert_eq!(actual, expected);
                    }
                }
            }
        }
    }

    fn assert_independent_ij_matches_general(data: &SurvfitAJData, options: &SurvfitAJOptions) {
        let fast = survfitaj(data, options).unwrap();
        let reference = survfitaj(
            data,
            &SurvfitAJOptions {
                influence: true,
                ..options.clone()
            },
        )
        .unwrap();
        assert_eq!(fast.time, reference.time);
        assert_eq!(fast.pstate, reference.pstate);
        assert_eq!(fast.cumhaz, reference.cumhaz);
        assert_eq!(fast.n_risk, reference.n_risk);
        assert_eq!(fast.n_transition, reference.n_transition);
        for (name, actual, expected) in [
            ("probability", &fast.std_err, &reference.std_err),
            ("hazard", &fast.std_chaz, &reference.std_chaz),
            ("area", &fast.std_auc, &reference.std_auc),
            ("lower", &fast.lower, &reference.lower),
            ("upper", &fast.upper, &reference.upper),
        ] {
            for (time, (actual, expected)) in actual
                .as_ref()
                .unwrap()
                .iter()
                .zip(expected.as_ref().unwrap())
                .enumerate()
            {
                for (state, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
                    assert!(
                        actual == expected
                            || (actual.is_nan() && expected.is_nan())
                            || (actual.is_finite()
                                && expected.is_finite()
                                && close(actual, expected, 2e-11)),
                        "{name} at time {time}, column {state}: {actual} != {expected}; \
                         p={}, fast se={}, reference se={}; weights={:?}",
                        fast.pstate[time][state.min(fast.states.len() - 1)],
                        fast.std_err.as_ref().unwrap()[time][state.min(fast.states.len() - 1)],
                        reference.std_err.as_ref().unwrap()[time][state.min(fast.states.len() - 1)],
                        data.weights
                    );
                }
            }
        }
    }

    #[test]
    fn independent_moments_match_general_weighted_tied_and_stratified_ij() {
        for seed in 0..48_u64 {
            let mut rng = seed + 1;
            let mut next = || {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                rng >> 32
            };
            let rows = 48 + seed as usize;
            let time = (0..rows)
                .map(|row| {
                    if seed % 2 == 0 {
                        (next() % 13 + 1) as f64
                    } else {
                        row as f64 / 4.0 + 1.0
                    }
                })
                .collect();
            let state = (0..rows).map(|_| (next() % 4) as i32).collect();
            let weights = (0..rows).map(|_| (next() % 17) as f64 / 4.0).collect();
            let grouped = seed % 3 == 0;
            let data = SurvfitAJData::try_new(
                None,
                time,
                state,
                names(&["a", "b", "c"]),
                Some(weights),
                grouped.then(|| (0..rows).map(|row| (row % 3) as i32).collect()),
                None,
                Some(
                    (0..rows)
                        .map(|row| {
                            ["a", "b", "c"][(seed as usize + if grouped { row % 3 } else { 0 }) % 3]
                                .to_string()
                        })
                        .collect(),
                ),
                Some(names(&["a", "b", "c"])),
                None,
            )
            .unwrap();
            let options = SurvfitAJOptions {
                p0: (seed % 4 != 0).then(|| vec![0.15, 0.5, 0.35]),
                start_time: (seed % 5 == 0).then_some(2.0),
                time0: seed % 2 == 0,
                timefix: false,
                ..Default::default()
            };
            assert_independent_ij_matches_general(&data, &options);
        }
    }

    #[test]
    fn independent_moments_preserve_single_destination_absorption() {
        let data = SurvfitAJData::try_new(
            None,
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1, 1, 1, 1],
            names(&["dead"]),
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        for p0 in [None, Some(vec![0.6, 0.4])] {
            let options = SurvfitAJOptions {
                p0,
                ..Default::default()
            };
            assert_independent_ij_matches_general(&data, &options);
            let fit = survfitaj(&data, &options).unwrap();
            assert_eq!(fit.std_err.as_ref().unwrap().last().unwrap(), &[0.0, 0.0]);
            for matrix in [fit.std_err.unwrap(), fit.std_auc.unwrap()] {
                assert!(matrix.iter().all(|row| row[0] == row[1]));
            }
        }
    }

    #[test]
    fn independent_moments_preserve_extreme_weights_and_zero_risk_events() {
        for weights in [
            vec![1e150, 2e150, 3e150, 4e150],
            vec![1e-150, 2e-150, 3e-150, 4e-150],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0; 4],
            vec![1e300, 1e-300, 1.0, 0.0],
            vec![1e-320, 2e-320, 3e-320, 4e-320],
        ] {
            let data = SurvfitAJData::try_new(
                None,
                vec![1.0, 2.0, 2.0, 3.0],
                vec![1, 2, 0, 1],
                names(&["a", "b"]),
                Some(weights),
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();
            assert_independent_ij_matches_general(&data, &SurvfitAJOptions::default());
        }
    }

    #[test]
    fn independent_single_destination_is_local_to_each_stratum() {
        let data = SurvfitAJData::try_new(
            None,
            vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
            vec![1, 1, 1, 1, 2, 2, 2, 2],
            names(&["a", "b"]),
            Some(vec![0.375, 3.0, 0.25, 1.875, 0.375, 3.0, 0.25, 1.875]),
            Some(vec![0, 0, 0, 0, 1, 1, 1, 1]),
            None,
            None,
            None,
            None,
        )
        .unwrap();
        for conf_type in [
            ConfType::Plain,
            ConfType::Log,
            ConfType::LogLog,
            ConfType::Logit,
            ConfType::Arcsin,
        ] {
            assert_independent_ij_matches_general(
                &data,
                &SurvfitAJOptions {
                    conf_type,
                    timefix: false,
                    ..Default::default()
                },
            );
        }
    }

    #[test]
    fn independent_single_destination_ignores_zero_weight_and_hidden_events() {
        for competitor in [(1.0, 0.0), (-1.0, 2.0)] {
            let data = SurvfitAJData::try_new(
                None,
                vec![1.0, 2.0, 3.0, 4.0, competitor.0],
                vec![1, 1, 1, 1, 2],
                names(&["a", "b"]),
                Some(vec![0.375, 3.0, 0.25, 1.875, competitor.1]),
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();
            assert_independent_ij_matches_general(
                &data,
                &SurvfitAJOptions {
                    conf_type: ConfType::LogLog,
                    timefix: false,
                    ..Default::default()
                },
            );
        }
    }

    #[test]
    fn independent_moments_retain_general_variance_square_range() {
        for p0 in [vec![-1e200, 1e200, 1.0], vec![1e-200, 1.0, 0.0]] {
            let options = SurvfitAJOptions {
                p0: Some(p0),
                conf_type: ConfType::LogLog,
                ..Default::default()
            };
            assert_independent_ij_matches_general(&ties_data(), &options);
        }
        let mut data = ties_data();
        data.time.iter_mut().for_each(|time| *time *= 1e200);
        assert_independent_ij_matches_general(
            &data,
            &SurvfitAJOptions {
                timefix: false,
                ..Default::default()
            },
        );
    }

    #[test]
    fn independent_moments_preserve_absorption_below_zero_from_rounding() {
        let data = SurvfitAJData::try_new(
            None,
            vec![1.0, 2.0, 2.0, 3.0, 4.0, 4.0],
            vec![1, 2, 1, 2, 1, 2],
            names(&["a", "b"]),
            Some(vec![0.5, 1.0, 2.0, 1.5, 1.25, 2.75]),
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        assert_independent_ij_matches_general(&data, &SurvfitAJOptions::default());
    }

    #[test]
    fn estimated_initial_states_match_separately_fitted_curves() {
        let make_data = |rows: &[usize], counting: bool, grouped: bool| {
            let time = [1.0, 3.0, 2.0, 4.0];
            let weights = [0.5, 1.5, 2.0, 1.0];
            let istate = ["a", "b", "a", "b"];
            let strata = [9, 9, -2, -2];
            SurvfitAJData::try_new(
                counting.then(|| vec![0.0; rows.len()]),
                rows.iter().map(|&i| time[i]).collect(),
                vec![1; rows.len()],
                names(&["dead"]),
                Some(rows.iter().map(|&i| weights[i]).collect()),
                grouped.then(|| rows.iter().map(|&i| strata[i]).collect()),
                Some(rows.iter().map(|&i| i as i64).collect()),
                Some(rows.iter().map(|&i| istate[i].to_string()).collect()),
                Some(names(&["a", "b"])),
                None,
            )
            .unwrap()
        };
        for counting in [false, true] {
            for start_time in [0.0, 1.0] {
                // At the earliest entry time, subjects starting there are at
                // risk; later estimates use intervals starting strictly before it.
                let options = SurvfitAJOptions {
                    se_fit: false,
                    start_time: Some(start_time),
                    time0: true,
                    timefix: false,
                    ..Default::default()
                };
                let grouped =
                    survfitaj(&make_data(&[0, 1, 2, 3], counting, true), &options).unwrap();
                for (curve, rows) in [[2, 3], [0, 1]].iter().enumerate() {
                    let alone = survfitaj(&make_data(rows, counting, false), &options).unwrap();
                    let range = grouped.curve_ranges()[curve].clone();
                    assert_eq!(grouped.states, alone.states);
                    assert_eq!(grouped.p0[curve], alone.p0[0]);
                    assert_eq!(grouped.time[range.clone()], alone.time);
                    for (combined, separate) in [
                        (&grouped.n_risk, &alone.n_risk),
                        (&grouped.n_event, &alone.n_event),
                        (&grouped.n_censor, &alone.n_censor),
                        (&grouped.n_transition, &alone.n_transition),
                        (&grouped.pstate, &alone.pstate),
                        (&grouped.cumhaz, &alone.cumhaz),
                    ] {
                        assert_eq!(combined[range.clone()], *separate);
                    }
                }
            }
        }
    }

    #[test]
    fn initial_state_estimation_uses_entry_minimum_across_curves() {
        let mut data = SurvfitAJData::try_new(
            Some(vec![0.0, 0.0, 1.0, 1.0]),
            vec![2.0, 3.0, 2.0, 4.0],
            vec![1; 4],
            names(&["dead"]),
            None,
            Some(vec![9, 9, -2, -2]),
            Some(vec![0, 1, 2, 3]),
            Some(names(&["a", "b", "a", "b"])),
            None,
            None,
        )
        .unwrap();
        let options = SurvfitAJOptions {
            se_fit: false,
            start_time: Some(1.0),
            timefix: false,
            ..Default::default()
        };
        // Entry at a later curve's own minimum is excluded when other
        // curves enter earlier. Taking the minimum per curve would admit it.
        assert!(
            survfitaj(&data, &options)
                .unwrap_err()
                .to_string()
                .contains("no one at risk for one of the curves")
        );
        data.start = Some(vec![1.0; 4]);
        let fit = survfitaj(&data, &options).unwrap();
        assert_eq!(fit.p0, vec![vec![0.5, 0.5, 0.0]; 2]);
    }

    #[test]
    fn estimated_p0_influence_reads_rows_at_r_curve_offsets() {
        // survfit(Surv(start, stop, ev) ~ g, id = id, istate = istate,
        // influence = TRUE) with the two curves' rows interleaved: p0 is
        // estimated at t0 = 2 and survfitAJ.R reads curve k's rows at its
        // offset in the data, R's n2 <- cumsum(c(0, n.per.curve))
        let istate = ["a", "b", "a", "a", "b", "b", "a", "b", "a", "b", "a", "a"];
        let data = SurvfitAJData::try_new(
            Some(vec![0.0; 12]),
            vec![5.0, 3.0, 7.0, 2.0, 4.0, 6.0, 8.0, 1.0, 9.0, 3.0, 6.0, 5.0],
            vec![1, 0, 2, 1, 0, 2, 2, 0, 1, 2, 0, 1],
            names(&["b", "c"]),
            None,
            Some((0..12).map(|i| i % 2).collect()),
            Some((1..=12).collect()),
            Some(names(&istate)),
            Some(names(&["a", "b", "c"])),
            None,
        )
        .unwrap();
        let options = SurvfitAJOptions {
            influence: true,
            ..Default::default()
        };
        let fit = survfitaj(&data, &options).unwrap();
        assert_eq!(fit.t0, 2.0);
        assert!(close(fit.p0[0][0], 5.0 / 6.0, 1e-12) && close(fit.p0[1][0], 0.4, 1e-12));
        // fit$i0[[k]][, "a"] (column "b" is its negative) and fit$se0
        let influence = fit.influence_pstate.as_ref().unwrap();
        let expected = [
            [1.0 / 36.0, 0.0, 1.0 / 36.0, 0.0, -5.0 / 36.0, 0.0],
            [0.0, 0.0, 0.0, -0.08, 0.0, 0.12],
        ];
        for (curve, expected) in influence.iter().zip(expected) {
            let i0 = curve.i0.as_ref().unwrap();
            for (k, &value) in expected.iter().enumerate() {
                assert!(close(i0[[k, 0]], value, 1e-12) && close(i0[[k, 1]], -value, 1e-12));
                assert_eq!(i0[[k, 2]], 0.0);
            }
        }
        let se0 = fit.se0.as_ref().unwrap();
        for (row, expected) in se0.iter().zip([(1.0f64 / 48.0).sqrt(), 0.0208f64.sqrt()]) {
            assert!(close(row[0], expected, 1e-12) && close(row[1], expected, 1e-12));
        }
    }

    #[test]
    fn user_p0_and_strata_and_options() {
        let mut data = ties_data();
        data.strata = Some((0..16).map(|i| i % 2).collect());
        let fit = survfitaj(
            &data,
            &SurvfitAJOptions {
                p0: Some(vec![0.8, 0.1, 0.1]),
                conf_type: ConfType::LogLog,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(fit.strata.as_ref().map(Vec::len), Some(2));
        assert_eq!(fit.strata_codes, Some(vec![0, 1]));
        assert_eq!(fit.p0, vec![vec![0.8, 0.1, 0.1]; 2]);
        assert!(fit.se0.is_none());
        let ranges = fit.curve_ranges();
        assert_eq!(ranges[0].start, 0);
        // pstate stays a distribution
        for row in &fit.pstate {
            assert!(close(row.iter().sum::<f64>(), 1.0, 1e-12));
        }
        let no_se = survfitaj(
            &data,
            &SurvfitAJOptions {
                se_fit: false,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(no_se.std_err.is_none() && no_se.lower.is_none());
        assert!(
            survfitaj(
                &data,
                &SurvfitAJOptions {
                    p0: Some(vec![0.5, 0.5]),
                    ..Default::default()
                }
            )
            .is_err()
        );
    }

    #[test]
    fn select_curves_slices_each_curve() {
        let mut data = ties_data();
        data.strata = Some((0..16).map(|i| i % 2).collect());
        let fit = survfitaj(
            &data,
            &SurvfitAJOptions {
                influence: true,
                ..Default::default()
            },
        )
        .unwrap();
        let rows = fit.curve_ranges()[1].clone();
        let second = fit.select_curves(&[1]).unwrap();
        assert_eq!(second.time, fit.time[rows.clone()]);
        assert_eq!(second.pstate, fit.pstate[rows.clone()]);
        assert_eq!(
            second.std_err.as_deref(),
            Some(&fit.std_err.as_ref().unwrap()[rows])
        );
        assert_eq!((second.n, second.n_id), (vec![fit.n[1]], vec![fit.n_id[1]]));
        assert_eq!(second.p0, vec![fit.p0[1].clone()]);
        assert_eq!((second.strata, second.strata_codes), (None, None));
        assert_eq!(
            second.influence_pstate.unwrap(),
            vec![fit.influence_pstate.as_ref().unwrap()[1].clone()]
        );
        // R numbers the clusters (here the rows) by first appearance in the data
        let clusters = &fit.influence_pstate.as_ref().unwrap()[1].cluster;
        assert_eq!(clusters, &(1..=8).map(|k| 2 * k).collect::<Vec<i64>>());
        let swapped = fit.select_curves(&[1, 0]).unwrap();
        assert_eq!(swapped.strata_codes, Some(vec![1, 0]));
        assert!(fit.select_curves(&[2]).is_err());
    }

    #[test]
    fn select_states_matches_r() {
        // e <- factor(c("censor","a","b","a","b","censor","a","b"), c("censor","a","b"))
        // f <- survfit(Surv(1:8, e) ~ 1); f[, "a"]$pstate, $std.err
        let data = SurvfitAJData::try_new(
            None,
            (1..=8).map(f64::from).collect(),
            vec![0, 1, 2, 1, 2, 0, 1, 2],
            names(&["a", "b"]),
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let fit = survfitaj(&data, &SurvfitAJOptions::default()).unwrap();
        assert_eq!(fit.states, names(&["(s0)", "a", "b"]));
        let a = fit.select_states(&[1]).unwrap();
        assert_eq!(a.states, names(&["a"]));
        let pstate = [0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.5, 3.5].map(|p| p / 7.0);
        let se = [
            0.0,
            0.132260014253222,
            0.132260014253222,
            0.170746944190628,
            0.170746944190628,
            0.170746944190628,
            0.208248281958761,
            0.208248281958761,
        ];
        for (t, row) in a.pstate.iter().enumerate() {
            assert!(close(row[0], pstate[t], 1e-12));
            assert!(close(a.std_err.as_ref().unwrap()[t][0], se[t], 1e-12));
        }
        assert_eq!(a.p0, vec![vec![0.0]]);
        assert!(a.n_censor.iter().all(|row| row.len() == 1));
        // the transitions go: R has no consistent way to subscript them
        assert!(a.hazard_from.is_empty() && a.transitions.is_empty());
        assert!(a.cumhaz.iter().all(Vec::is_empty) && a.std_chaz.is_none());
        let all = fit.select_states(&[0, 1, 2]).unwrap();
        assert_eq!(
            (&all.cumhaz, &all.hazard_from),
            (&fit.cumhaz, &fit.hazard_from)
        );
        assert!(
            fit.select_states(&[2, 1, 0])
                .unwrap()
                .hazard_from
                .is_empty()
        );
        assert!(fit.select_states(&[3]).is_err());
    }

    #[test]
    fn start_time_and_time0() {
        let fit = survfitaj(
            &ties_data(),
            &SurvfitAJOptions {
                start_time: Some(2.0),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(fit.t0, 2.0);
        assert_eq!(fit.n, vec![13]);
        assert_eq!(fit.start_time, Some(2.0));
        let with_zero = survfitaj(
            &ties_data(),
            &SurvfitAJOptions {
                time0: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(with_zero.time[0], 0.0);
        assert_eq!(with_zero.pstate[0], vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn survcheck_rejects_bad_subjects() {
        let mut data = istate_data();
        // an overlapping interval for subject 1
        data.start.as_mut().unwrap()[1] = 2.0;
        assert!(survfitaj(&data, &SurvfitAJOptions::default()).is_err());
        let mut data = istate_data();
        data.id = None;
        assert!(survfitaj(&data, &SurvfitAJOptions::default()).is_err());
        assert!(
            SurvfitAJData::try_new(
                None,
                vec![1.0],
                vec![3],
                names(&["a", "b"]),
                None,
                None,
                None,
                None,
                None,
                None
            )
            .is_err()
        );
    }
}
