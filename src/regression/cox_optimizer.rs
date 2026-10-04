//! Newton-Raphson engine for the Cox partial likelihood.
//!
//! Ports of R survival's C fitters, selected by the data and tie method:
//!
//! * `src/coxfit6.c` — right-censored data, Breslow and Efron ties;
//! * `src/agfit4.c` — (start, stop] counting-process data, Breslow and Efron;
//! * `src/coxexact.c` — right-censored data, exact partial likelihood;
//! * `src/agexact.c` — counting-process data, exact partial likelihood.
//!
//! `CoxFit::fit` follows each C source's convergence and step-halving
//! rules.  `coxfit6` halves ever more aggressively (`(newbeta + halving *
//! beta) / (halving + 1)`) and accepts convergence during halving with flag
//! `-2`.  `agfit4` halves the same way but converges only after a full
//! Newton step, and also halves when the rank of the information matrix
//! changes or its diagonal is not finite.  The exact fitters halve by
//! `1/2`, never declare convergence mid-halving, and stop at `iter ==
//! maxiter` without a final step.
//!
//! Covariates are centred and scaled as the C code does (`doscale`), so the
//! Newton steps are well conditioned; coefficients, the score vector and the
//! variance are returned on the original scale.  Data may arrive in any row
//! order: the builder sorts by (stratum, time) once and every result is
//! order independent.

use crate::constants::{COX_CONVERGENCE_TOLERANCE, COX_MAX_ITER, COX_RANK_TOLERANCE};
use crate::core::risk_sweep::{RecenteredRiskSet, RiskSetSums, spans_a_death};
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::matrix::{chinv2, cholesky2, chsolve2};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use super::exact_ties::{ExactRiskAccumulator, ExactRiskTree, exact_tied_moments};

/// Tie handling of the partial likelihood (R's `coxph(ties = )`), shared by
/// the fitters and by every residual kernel of the package.  The C kernels
/// receive it as `method = as.integer(method == "efron")`, so for them
/// `Exact` behaves like `Breslow` (R's `coxmart2.c` for the exact fitters);
/// the routines R refuses for an exact fit (score, Schoenfeld and detail
/// output) reject it explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[pyclass(module = "survival._survival", eq, eq_int, from_py_object)]
pub enum TieMethod {
    Breslow,
    Efron,
    Exact,
}

crate::internal::pickle::picklable!(TieMethod);

impl TieMethod {
    /// Parses R's `ties` argument (case-insensitive); `None` is R's default,
    /// Efron.
    pub fn parse(name: Option<&str>) -> SurvivalResult<Self> {
        match name.map(str::to_ascii_lowercase).as_deref() {
            None | Some("efron") => Ok(Self::Efron),
            Some("breslow") => Ok(Self::Breslow),
            Some("exact") => Ok(Self::Exact),
            Some(other) => Err(SurvivalError::invalid_input(format!(
                "ties must be 'efron', 'breslow' or 'exact', got '{other}'"
            ))),
        }
    }

    /// R's name for the method (`fit$method`).
    pub fn r_name(self) -> &'static str {
        match self {
            Self::Breslow => "breslow",
            Self::Efron => "efron",
            Self::Exact => "exact",
        }
    }

    /// The `method == "efron"` flag of the C residual kernels.
    pub(crate) fn is_efron(self) -> bool {
        self == Self::Efron
    }

    /// `residuals.coxph`'s refusal for an exact fit: `<what> residuals are
    /// not available for the exact method`.
    pub(crate) fn reject_exact(self, what: &str) -> SurvivalResult<()> {
        if self == Self::Exact {
            return Err(SurvivalError::invalid_input(format!(
                "{what} residuals are not available for the exact method"
            )));
        }
        Ok(())
    }
}

/// Fit result on the original covariate scale (R's `coxfit$...` list).
#[derive(Debug, Clone)]
pub(crate) struct CoxFitResults {
    pub coefficients: Vec<f64>,
    /// Column centres used for the linear predictor (`coxfit6` `means`;
    /// zero for `nocenter` columns).
    pub means: Vec<f64>,
    /// Score vector (first derivative) at the final coefficients.
    pub score: Vec<f64>,
    /// Inverse information matrix (`fit$var`); rows and columns of
    /// redundant covariates are zero.
    pub var: Array2<f64>,
    /// Log partial likelihood at the initial and final coefficients.
    pub loglik: [f64; 2],
    /// Score test at the initial coefficients.
    pub sctest: f64,
    /// Rank of the information matrix, `-2` when convergence was reached
    /// while step halving, `1000` when the iteration budget ran out.
    pub flag: i32,
    pub iter: usize,
    /// `agreg.fit`'s `info` for an `agfit4` fit: the rank of the information
    /// matrix at the initial coefficients, the number of recentrings of the
    /// risk scores, the number of step halvings, and 1 when the iterations
    /// ran out.
    pub info: Option<[i32; 4]>,
    /// Rows sorted by (stratum, time, original index); the order the
    /// residual kernels use.
    pub order: Vec<usize>,
}

/// Inputs of a Cox fit.  Rows may be in any order; `strata` are stratum
/// codes (any integers), `entry_times` switches to the counting-process
/// likelihood.
pub(crate) struct CoxFitBuilder {
    time: Array1<f64>,
    status: Array1<i32>,
    covar: Array2<f64>,
    entry_times: Option<Array1<f64>>,
    strata: Option<Array1<i32>>,
    offset: Option<Array1<f64>>,
    weights: Option<Array1<f64>>,
    method: TieMethod,
    max_iter: usize,
    iterate_empty: bool,
    eps: f64,
    toler: f64,
    doscale: Option<Vec<bool>>,
    initial_beta: Option<Vec<f64>>,
}

impl CoxFitBuilder {
    pub(crate) fn new(time: Array1<f64>, status: Array1<i32>, covar: Array2<f64>) -> Self {
        Self {
            time,
            status,
            covar,
            entry_times: None,
            strata: None,
            offset: None,
            weights: None,
            method: TieMethod::Breslow,
            max_iter: COX_MAX_ITER,
            iterate_empty: false,
            eps: COX_CONVERGENCE_TOLERANCE,
            toler: COX_RANK_TOLERANCE,
            doscale: None,
            initial_beta: None,
        }
    }

    pub(crate) fn entry_times(mut self, entry_times: Array1<f64>) -> Self {
        self.entry_times = Some(entry_times);
        self
    }

    pub(crate) fn strata(mut self, strata: Array1<i32>) -> Self {
        self.strata = Some(strata);
        self
    }

    pub(crate) fn offset(mut self, offset: Array1<f64>) -> Self {
        self.offset = Some(offset);
        self
    }

    pub(crate) fn weights(mut self, weights: Array1<f64>) -> Self {
        self.weights = Some(weights);
        self
    }

    pub(crate) fn method(mut self, method: TieMethod) -> Self {
        self.method = method;
        self
    }

    pub(crate) fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// The bare agexact fitter still iterates when the matrix has no columns.
    pub(crate) fn iterate_empty(mut self, value: bool) -> Self {
        self.iterate_empty = value;
        self
    }

    pub(crate) fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    pub(crate) fn toler(mut self, toler: f64) -> Self {
        self.toler = toler;
        self
    }

    /// Per column: centre and scale (`true`) or leave alone (`false`, R's
    /// `nocenter` columns).
    pub(crate) fn doscale(mut self, doscale: Vec<bool>) -> Self {
        self.doscale = Some(doscale);
        self
    }

    pub(crate) fn initial_beta(mut self, initial_beta: Vec<f64>) -> Self {
        self.initial_beta = Some(initial_beta);
        self
    }

    pub(crate) fn build(self) -> SurvivalResult<CoxFit> {
        let n = self.covar.nrows();
        let nvar = self.covar.ncols();
        let check = |name: &str, len: usize| -> SurvivalResult<()> {
            if len != n {
                return Err(SurvivalError::invalid_input(format!(
                    "{name} has {len} rows but covariates has {n}"
                )));
            }
            Ok(())
        };
        check("time", self.time.len())?;
        check("status", self.status.len())?;
        if let Some(entry) = &self.entry_times {
            check("entry_times", entry.len())?;
        }
        if let Some(strata) = &self.strata {
            check("strata", strata.len())?;
        }
        if let Some(offset) = &self.offset {
            check("offset", offset.len())?;
        }
        if let Some(weights) = &self.weights {
            check("weights", weights.len())?;
            if self.method == TieMethod::Exact && weights.iter().any(|&w| w != 1.0) {
                return Err(SurvivalError::invalid_input(
                    "Case weights are not supported for the exact method",
                ));
            }
        }
        let doscale = self.doscale.unwrap_or_else(|| vec![true; nvar]);
        if doscale.len() != nvar {
            return Err(SurvivalError::invalid_input(format!(
                "doscale has {} entries but covariates has {nvar} columns",
                doscale.len()
            )));
        }
        let initial_beta = self.initial_beta.unwrap_or_else(|| vec![0.0; nvar]);
        if initial_beta.len() != nvar {
            return Err(SurvivalError::invalid_input(
                "Wrong length for inital values",
            ));
        }
        if !(self.eps.is_finite() && self.eps > 0.0) {
            return Err(SurvivalError::invalid_input(
                "eps must be a finite positive value",
            ));
        }
        if !(self.toler.is_finite() && self.toler > 0.0) {
            return Err(SurvivalError::invalid_input(
                "toler.chol must be a finite positive value",
            ));
        }

        // R sorts by (strata, time) with a stable order; the per-stratum
        // end markers are what the C sweeps read.
        let codes = self.strata.as_ref();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&lhs, &rhs| {
            codes
                .map_or(std::cmp::Ordering::Equal, |c| c[lhs].cmp(&c[rhs]))
                .then_with(|| self.time[lhs].total_cmp(&self.time[rhs]))
                .then_with(|| lhs.cmp(&rhs))
        });
        let mut stratum_end = vec![0i32; n];
        for (position, &row) in order.iter().enumerate() {
            let last = position + 1 == n || codes.is_some_and(|c| c[row] != c[order[position + 1]]);
            if last {
                stratum_end[position] = 1;
            }
        }
        let gather = |values: &Array1<f64>| Array1::from_iter(order.iter().map(|&i| values[i]));
        let time = gather(&self.time);
        let status = Array1::from_iter(order.iter().map(|&i| self.status[i]));
        let offset = self
            .offset
            .as_ref()
            .map_or_else(|| Array1::zeros(n), gather);
        let weights = self
            .weights
            .as_ref()
            .map_or_else(|| Array1::ones(n), gather);
        let mut covar = Array2::zeros((n, nvar));
        for (position, &row) in order.iter().enumerate() {
            covar.row_mut(position).assign(&self.covar.row(row));
        }
        let fitter = match (self.method, self.entry_times.as_ref().map(gather)) {
            (TieMethod::Exact, None) => Fitter::Coxexact,
            (TieMethod::Exact, Some(entry)) => Fitter::Agexact {
                walk: AgWalk::new(
                    entry.as_slice().expect("contiguous"),
                    time.as_slice().expect("contiguous"),
                    status.as_slice().expect("contiguous"),
                    &stratum_end,
                ),
            },
            (_, None) => Fitter::Coxfit6,
            (_, Some(entry)) => Fitter::Agfit4 {
                walk: AgWalk::new(
                    entry.as_slice().expect("contiguous"),
                    time.as_slice().expect("contiguous"),
                    status.as_slice().expect("contiguous"),
                    &stratum_end,
                ),
                risk_set: RecenteredRiskSet::new(nvar),
            },
        };

        let mut fit = CoxFit {
            data: CoxData {
                time,
                status,
                covar,
                stratum_end: Array1::from_vec(stratum_end),
                offset,
                weights,
                efron: self.method == TieMethod::Efron,
            },
            fitter,
            max_iter: self.max_iter,
            iterate_empty: self.iterate_empty,
            eps: self.eps,
            toler: self.toler,
            scale: vec![1.0; nvar],
            means: vec![0.0; nvar],
            beta: initial_beta,
            u: vec![0.0; nvar],
            imat: Array2::zeros((nvar, nvar)),
            loglik: [0.0; 2],
            sctest: 0.0,
            flag: 0,
            iter: 0,
            info: None,
            order,
        };
        fit.scale_center(&doscale);
        Ok(fit)
    }
}

/// `agfit4.c`'s walk over (start, stop] data: per stratum, the rows at risk
/// at one or more of its death times (`agreg.fit`'s `!ignore`, see
/// [`spans_a_death`]) by decreasing stop time, ties in data order
/// (`agreg.fit`'s `sort.end`), and by decreasing entry time, ties in the
/// stop-time order.
struct AgWalk {
    by_stop: Vec<Joining>,
    /// The same rows with their entry times.
    by_entry: Vec<(f64, usize)>,
    /// `[start, end)` of each stratum in both orders.
    bounds: Vec<(usize, usize)>,
    /// Original sorted row bounds before rows that span no death were omitted.
    row_bounds: Vec<(usize, usize)>,
    /// All retained entries precede the first death: the risk set only grows.
    growing: Vec<bool>,
}

impl AgWalk {
    /// `entry`, `time`, `status` and `stratum_end` are in sorted row order.
    fn new(entry: &[f64], time: &[f64], status: &[i32], stratum_end: &[i32]) -> Self {
        let n = time.len();
        let positions: Vec<usize> = (0..n).collect();
        let mut by_stop = Vec::with_capacity(n);
        let mut by_entry = Vec::with_capacity(n);
        let mut bounds = Vec::new();
        let mut row_bounds = Vec::new();
        let mut growing = Vec::new();
        let mut start = 0;
        for end in 0..n {
            if stratum_end[end] != 1 {
                continue;
            }
            let spans = spans_a_death(&positions[start..=end], time, entry, status);
            let first = by_stop.len();
            // Positions are sorted by (time, data row): walk the tied blocks
            // from the last, each in position order.
            let mut block_end = end + 1;
            while block_end > start {
                let mut block_start = block_end - 1;
                while block_start > start && time[block_start - 1] == time[block_end - 1] {
                    block_start -= 1;
                }
                by_stop.extend(
                    (block_start..block_end)
                        .filter(|&p| spans[p - start])
                        .map(|row| Joining {
                            row,
                            time: time[row],
                            death: status[row] == 1,
                        }),
                );
                block_end = block_start;
            }
            let leaving = by_entry.len();
            by_entry.extend(by_stop[first..].iter().map(|j| (entry[j.row], j.row)));
            by_entry[leaving..].sort_by(|l, r| r.0.total_cmp(&l.0));
            bounds.push((first, by_stop.len()));
            row_bounds.push((start, end + 1));
            let first_death = by_stop[first..].iter().rev().find(|row| row.death);
            growing.push(first_death.is_none_or(|death| by_entry[leaving].0 < death.time));
            start = end + 1;
        }
        Self {
            by_stop,
            by_entry,
            bounds,
            row_bounds,
            growing,
        }
    }
}

/// A row of [`AgWalk::by_stop`]: its sorted position, stop time and
/// whether it is a death.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Joining {
    row: usize,
    time: f64,
    death: bool,
}

/// The C fitter the data and tie method select.
enum Fitter {
    Coxfit6,
    /// With its walk over the rows and its running risk set.
    Agfit4 {
        walk: AgWalk,
        risk_set: RecenteredRiskSet,
    },
    Coxexact,
    /// With the prepared entry/stop walk and `agexact` iteration policy.
    Agexact {
        walk: AgWalk,
    },
}

impl Fitter {
    /// `agfit4`'s recentrings of the risk scores so far.
    fn rescales(&self) -> Option<i32> {
        match self {
            Self::Agfit4 { risk_set, .. } => Some(risk_set.rescales),
            _ => None,
        }
    }
}

/// The sorted, centred and scaled data of a Cox model: what one evaluation
/// of the partial likelihood reads.
struct CoxData {
    time: Array1<f64>,
    status: Array1<i32>,
    covar: Array2<f64>,
    /// `1` on the last row of each stratum (the C `strata` convention).
    stratum_end: Array1<i32>,
    offset: Array1<f64>,
    weights: Array1<f64>,
    efron: bool,
}

/// A Cox model ready to iterate: the data plus the current state of the
/// Newton iteration.
pub(crate) struct CoxFit {
    data: CoxData,
    fitter: Fitter,
    max_iter: usize,
    iterate_empty: bool,
    eps: f64,
    toler: f64,
    scale: Vec<f64>,
    means: Vec<f64>,
    beta: Vec<f64>,
    /// Score left by the last evaluation.
    u: Vec<f64>,
    /// Information matrix (upper triangle) left by the last evaluation.
    imat: Array2<f64>,
    loglik: [f64; 2],
    sctest: f64,
    flag: i32,
    iter: usize,
    info: Option<[i32; 4]>,
    order: Vec<usize>,
}

/// The running risk set `agfit4`'s walk drives; a row is named by its
/// sorted position.
trait AgRiskSet {
    fn clear(&mut self);
    fn add(&mut self, person: usize, x: &[f64]) -> SurvivalResult<()>;
    fn remove(&mut self, person: usize, x: &[f64]);
    /// The weighted risk score of a row in the set.
    fn risk(&self, person: usize) -> f64;
    /// The constant the risk scores' linear predictors are taken from.
    fn recenter(&self) -> f64;
    fn sums(&self) -> &RiskSetSums;
}

/// The risk set while its centre stays at zero: `w exp(eta)` per row.
struct CentredAtZero<'a> {
    sums: RiskSetSums,
    risk: Vec<f64>,
    weights: &'a Array1<f64>,
}

impl AgRiskSet for CentredAtZero<'_> {
    fn clear(&mut self) {
        self.sums.clear();
    }

    fn add(&mut self, person: usize, x: &[f64]) -> SurvivalResult<()> {
        self.sums.add(self.weights[person], self.risk[person], x);
        Ok(())
    }

    fn remove(&mut self, person: usize, x: &[f64]) {
        self.sums.remove(self.weights[person], self.risk[person], x);
    }

    fn risk(&self, person: usize) -> f64 {
        self.risk[person]
    }

    fn recenter(&self) -> f64 {
        0.0
    }

    fn sums(&self) -> &RiskSetSums {
        &self.sums
    }
}

/// The general risk set, recentred as agfit4.c does.
struct Recentred<'a> {
    set: &'a mut RecenteredRiskSet,
    eta: &'a [f64],
    weights: &'a Array1<f64>,
}

impl AgRiskSet for Recentred<'_> {
    fn clear(&mut self) {
        self.set.clear();
    }

    fn add(&mut self, person: usize, x: &[f64]) -> SurvivalResult<()> {
        self.set.add(self.eta[person], self.weights[person], x)
    }

    fn remove(&mut self, person: usize, x: &[f64]) {
        self.set.remove(self.eta[person], self.weights[person], x);
    }

    fn risk(&self, person: usize) -> f64 {
        self.set.risk(self.eta[person], self.weights[person])
    }

    fn recenter(&self) -> f64 {
        self.set.recenter
    }

    fn sums(&self) -> &RiskSetSums {
        &self.set.sums
    }
}

/// `u += w x`.
fn add_scaled(u: &mut [f64], w: f64, x: &[f64]) {
    for (u, &x) in u.iter_mut().zip(x) {
        *u += w * x;
    }
}

/// `sum w / sum w |x|` over `rows` of a centred column, 1 for a constant
/// column: the `coxfit6.c` and `agfit4.c` scale.
fn mean_abs_scale(
    weights: &Array1<f64>,
    column: ArrayView1<'_, f64>,
    rows: impl Iterator<Item = usize>,
) -> f64 {
    let (weight, abs_sum) = rows.fold((0.0, 0.0), |(weight, abs_sum), person| {
        (
            weight + weights[person],
            abs_sum + weights[person] * column[person].abs(),
        )
    });
    if abs_sum > 0.0 { weight / abs_sum } else { 1.0 }
}

/// One death time's Breslow or Efron contribution (`coxfit6.c`,
/// `agfit4.c`), given the sums over its risk set, deaths included, and over
/// its `d` tied deaths.  Efron's `j`-th term (`j = 0..d`) sees the deaths
/// down-weighted by `j/d`.  Subtracts the risk-set means from the score,
/// adds the information to the upper triangle of `imat` (all `cholesky2`
/// reads) and returns the `-log(denominator)` part of the log likelihood;
/// the deaths' own `w eta` and `w x` terms are the caller's.
fn death_time_update(
    efron: bool,
    risk_set: &RiskSetSums,
    tied: &RiskSetSums,
    u: &mut [f64],
    imat: &mut Array2<f64>,
) -> f64 {
    let nvar = u.len();
    let ndead = tied.count;
    let steps = if efron { ndead } else { 1 };
    let wtave = tied.weight / steps as f64;
    let (cmat, cmat2) = (&risk_set.cmat, &tied.cmat);
    let imat = imat.as_slice_mut().expect("standard layout");
    let mut loglik = 0.0;
    for j in 0..steps {
        let fraction = j as f64 / ndead as f64;
        let sum = |all: f64, deaths: f64| if j == 0 { all } else { all - fraction * deaths };
        let denom = sum(risk_set.denom, tied.denom);
        loglik -= wtave * denom.ln();
        for i in 0..nvar {
            let mean = sum(risk_set.a[i], tied.a[i]) / denom;
            u[i] -= wtave * mean;
            for k in 0..=i {
                let second = sum(cmat[i * nvar + k], cmat2[i * nvar + k]);
                let first = sum(risk_set.a[k], tied.a[k]);
                imat[k * nvar + i] += wtave * ((second - mean * first) / denom);
            }
        }
    }
    loglik
}

/// Adds one death time's exact-likelihood contribution given the conditional
/// moments of the tied-death subsets (`coxexact.c`: `newlk -= log(d0)`,
/// `u -= d1/d0`, `imat += d2/d0 - d1 d1'`).
#[allow(clippy::too_many_arguments)]
fn apply_exact_event_moments(
    covar: &Array2<f64>,
    weights: &Array1<f64>,
    u: &mut [f64],
    imat: &mut Array2<f64>,
    death_indices: &[usize],
    linear_predictors: &[f64],
    log_denom: f64,
    mean: &[f64],
    covariance: ArrayView2<'_, f64>,
) -> f64 {
    let mut contribution = -log_denom;
    for &person in death_indices {
        contribution += weights[person] * linear_predictors[person];
        for (variable, value) in u.iter_mut().enumerate() {
            *value += weights[person] * covar[(person, variable)];
        }
    }
    for (variable, value) in u.iter_mut().enumerate() {
        *value -= mean[variable];
        for other in 0..covar.ncols() {
            imat[(variable, other)] += covariance[(variable, other)];
        }
    }
    contribution
}

#[allow(clippy::too_many_arguments)]
fn add_exact_event_contribution(
    covar: &Array2<f64>,
    weights: &Array1<f64>,
    u: &mut [f64],
    imat: &mut Array2<f64>,
    death_indices: &[usize],
    risk_indices: &[usize],
    linear_predictors: &[f64],
    log_risk: &[f64],
) -> f64 {
    if death_indices.len() == risk_indices.len() {
        return 0.0;
    }
    let moments = exact_tied_moments(risk_indices, death_indices.len(), log_risk, covar);
    apply_exact_event_moments(
        covar,
        weights,
        u,
        imat,
        death_indices,
        linear_predictors,
        moments.log_denom,
        &moments.mean,
        moments.covariance.view(),
    )
}

impl CoxData {
    /// The covariate row of each sorted position.
    fn rows<'a>(&'a self) -> impl Fn(usize) -> &'a [f64] + 'a {
        let nvar = self.covar.ncols();
        let covar = self.covar.as_slice().expect("row-major covariates");
        move |person| &covar[person * nvar..(person + 1) * nvar]
    }

    fn linear_predictors(&self, beta: &[f64]) -> Vec<f64> {
        let row = self.rows();
        (0..self.covar.nrows())
            .map(|person| {
                self.offset[person]
                    + beta
                        .iter()
                        .zip(row(person))
                        .fold(0.0, |sum, (&b, &x)| sum + b * x)
            })
            .collect()
    }

    /// Linear predictors and their log weighted risk `eta + log(w)`.
    fn exact_predictors(&self, beta: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let eta = self.linear_predictors(beta);
        let log_risk = eta
            .iter()
            .zip(self.weights.iter())
            .map(|(&e, &w)| e + w.ln())
            .collect();
        (eta, log_risk)
    }

    /// `coxfit6_iter`: the log likelihood, score and information for
    /// right-censored data with Breslow or Efron ties.  The risk set grows
    /// from the largest time downwards.
    fn coxfit6(&self, beta: &[f64], u: &mut [f64], imat: &mut Array2<f64>) -> f64 {
        let row = self.rows();
        let eta = self.linear_predictors(beta);
        let mut risk_set = RiskSetSums::zeros(self.covar.ncols(), true);
        let mut tied = risk_set.clone();
        let mut loglik = 0.0;
        let mut person = self.time.len();
        while person > 0 {
            if self.stratum_end[person - 1] == 1 {
                risk_set.clear();
            }
            let time = self.time[person - 1];
            // Tied times do not cross strata.
            loop {
                person -= 1;
                let (weight, x) = (self.weights[person], row(person));
                let risk = eta[person].exp() * weight;
                risk_set.add(weight, risk, x);
                if self.status[person] != 0 {
                    tied.add(weight, risk, x);
                    loglik += weight * eta[person];
                    add_scaled(u, weight, x);
                }
                if person == 0 || self.stratum_end[person - 1] == 1 || self.time[person - 1] != time
                {
                    break;
                }
            }
            if tied.count > 0 {
                loglik += death_time_update(self.efron, &risk_set, &tied, u, imat);
                tied.clear();
            }
        }
        loglik
    }

    /// `agfit4.c`'s accumulation loop for (start, stop] data.  Per stratum,
    /// one running risk set walks the death times from the largest down:
    /// first the rows that entered at or after the death time leave, then
    /// the rows whose stop time is at or after it join (see
    /// [`RecenteredRiskSet`]).  agfit4.c moves the centre of the risk scores
    /// only when the mean linear predictor of a risk set lies more than 200
    /// from it, so while every linear predictor lies within 199 of zero the
    /// centre stays at zero and the risk scores are computed up front.
    fn agfit4(
        &self,
        beta: &[f64],
        walk: &AgWalk,
        risk_set: &mut RecenteredRiskSet,
        u: &mut [f64],
        imat: &mut Array2<f64>,
    ) -> SurvivalResult<f64> {
        let eta = self.linear_predictors(beta);
        if eta.iter().all(|eta| eta.abs() < 199.0) {
            let centred = CentredAtZero {
                sums: RiskSetSums::zeros(self.covar.ncols(), true),
                risk: eta
                    .iter()
                    .zip(self.weights.iter())
                    .map(|(eta, weight)| eta.exp() * weight)
                    .collect(),
                weights: &self.weights,
            };
            self.agfit4_walk(walk, &eta, centred, u, imat)
        } else {
            risk_set.restart();
            let recentred = Recentred {
                set: risk_set,
                eta: &eta,
                weights: &self.weights,
            };
            self.agfit4_walk(walk, &eta, recentred, u, imat)
        }
    }

    /// The walk of [`CoxData::agfit4`] with the running risk set `risk_set`.
    /// The deaths' `eta` terms are taken relative to the set's centre, which
    /// therefore cancels.
    fn agfit4_walk(
        &self,
        walk: &AgWalk,
        eta: &[f64],
        mut risk_set: impl AgRiskSet,
        u: &mut [f64],
        imat: &mut Array2<f64>,
    ) -> SurvivalResult<f64> {
        let row = self.rows();
        let mut tied = RiskSetSums::zeros(self.covar.ncols(), true);
        let mut loglik = 0.0;
        for &(start, end) in &walk.bounds {
            risk_set.clear();
            let (by_stop, by_entry) = (&walk.by_stop[start..end], &walk.by_entry[start..end]);
            let (mut joined, mut left) = (0, 0);
            // The next death time is the stop time of the first death among
            // the rows still to join; its deaths are among the last rows to
            // join at it.
            while let Some(next) = by_stop[joined..].iter().position(|j| j.death) {
                let first_death = joined + next;
                let time = by_stop[first_death].time;
                while left < by_entry.len() && by_entry[left].0 >= time {
                    let person = by_entry[left].1;
                    risk_set.remove(person, row(person));
                    left += 1;
                }
                while joined < by_stop.len() && by_stop[joined].time >= time {
                    let person = by_stop[joined].row;
                    risk_set.add(person, row(person))?;
                    joined += 1;
                }
                tied.clear();
                for joining in by_stop[first_death..joined].iter().filter(|j| j.death) {
                    let person = joining.row;
                    let (weight, x) = (self.weights[person], row(person));
                    tied.add(weight, risk_set.risk(person), x);
                    loglik += weight * (eta[person] - risk_set.recenter());
                    add_scaled(u, weight, x);
                }
                loglik += death_time_update(self.efron, risk_set.sums(), &tied, u, imat);
            }
        }
        Ok(loglik)
    }

    /// `coxexact.c`: exact partial likelihood for right-censored data.  The
    /// risk set grows from the largest time downwards; a single death uses
    /// the running moments, tied deaths the subset dynamic programme.
    fn coxexact(&self, beta: &[f64], u: &mut [f64], imat: &mut Array2<f64>) -> f64 {
        let (linear_predictors, log_risk) = self.exact_predictors(beta);
        let mut loglik = 0.0;
        let mut stratum_start = 0usize;

        for stratum_end in 0..self.covar.nrows() {
            if self.stratum_end[stratum_end] != 1 {
                continue;
            }
            let mut risk_indices = Vec::with_capacity(stratum_end - stratum_start + 1);
            let mut death_indices = Vec::new();
            let mut singleton_moments = ExactRiskAccumulator::new(self.covar.ncols());
            let mut time_end = stratum_end;
            loop {
                let event_time = self.time[time_end];
                let mut time_start = time_end;
                while time_start > stratum_start && self.time[time_start - 1] == event_time {
                    time_start -= 1;
                }
                risk_indices.extend(time_start..=time_end);
                for (person, &log_weight) in log_risk
                    .iter()
                    .enumerate()
                    .take(time_end + 1)
                    .skip(time_start)
                {
                    singleton_moments.add(person, log_weight, &self.covar);
                }
                death_indices.clear();
                death_indices
                    .extend((time_start..=time_end).filter(|&person| self.status[person] != 0));
                if !death_indices.is_empty() {
                    loglik += if death_indices.len() == 1 && risk_indices.len() > 1 {
                        apply_exact_event_moments(
                            &self.covar,
                            &self.weights,
                            u,
                            imat,
                            &death_indices,
                            &linear_predictors,
                            singleton_moments.log_denom,
                            &singleton_moments.mean,
                            singleton_moments.covariance.view(),
                        )
                    } else {
                        add_exact_event_contribution(
                            &self.covar,
                            &self.weights,
                            u,
                            imat,
                            &death_indices,
                            &risk_indices,
                            &linear_predictors,
                            &log_risk,
                        )
                    };
                }
                if time_start == stratum_start {
                    break;
                }
                time_end = time_start - 1;
            }
            stratum_start = stratum_end + 1;
        }
        loglik
    }

    /// Exact partial likelihood for (start, stop] data. Each row joins and
    /// leaves once in the prepared walk. Growing risk sets use one running
    /// accumulator; general singleton moments use a blocked tree whose
    /// ancestors are recomputed after removal, preserving small risk scores.
    /// Tied deaths still use the exact subset dynamic programme.
    fn agexact(&self, beta: &[f64], walk: &AgWalk, u: &mut [f64], imat: &mut Array2<f64>) -> f64 {
        let (linear_predictors, log_risk) = self.exact_predictors(beta);
        let mut loglik = 0.0;
        let mut death_indices = Vec::new();
        let mut risk_indices = Vec::new();

        for (stratum, &(start, end)) in walk.bounds.iter().enumerate() {
            if start == end {
                continue;
            }
            let growing = walk.growing[stratum];
            let mut singleton = growing.then(|| ExactRiskAccumulator::new(self.covar.ncols()));
            let mut singleton_rows = 0;
            let mut tree = (!growing).then(|| {
                let (first, last) = walk.row_bounds[stratum];
                ExactRiskTree::new(first, last - first, self.covar.ncols())
            });
            risk_indices.clear();
            let (mut joined, mut left) = (start, start);
            while joined < end {
                let event_time = walk.by_stop[joined].time;
                if let Some(tree) = tree.as_mut() {
                    while left < end && walk.by_entry[left].0 >= event_time {
                        tree.set_active(walk.by_entry[left].1, false);
                        left += 1;
                    }
                }
                death_indices.clear();
                while joined < end && walk.by_stop[joined].time == event_time {
                    let row = walk.by_stop[joined].row;
                    if let Some(tree) = tree.as_mut() {
                        tree.set_active(row, true);
                    } else {
                        risk_indices.push(row);
                    }
                    if walk.by_stop[joined].death {
                        death_indices.push(row);
                    }
                    joined += 1;
                }
                if death_indices.is_empty() {
                    continue;
                }
                let nrisk = tree.as_ref().map_or(risk_indices.len(), ExactRiskTree::len);
                if death_indices.len() == nrisk {
                    // Selecting the entire risk set has likelihood one and
                    // zero score/information, regardless of its risk spread.
                    continue;
                }
                if death_indices.len() == 1 {
                    loglik += if let Some(tree) = tree.as_mut() {
                        tree.refresh(&log_risk, &self.covar);
                        apply_exact_event_moments(
                            &self.covar,
                            &self.weights,
                            u,
                            imat,
                            &death_indices,
                            &linear_predictors,
                            tree.log_denom(),
                            tree.mean(),
                            tree.covariance(),
                        )
                    } else {
                        // Tied and complete-risk-set events do not need the
                        // singleton moments. Grow them lazily, adding each
                        // row at most once before an untied evaluation.
                        let moments = singleton.as_mut().expect("growing risk set");
                        for &row in &risk_indices[singleton_rows..] {
                            moments.add(row, log_risk[row], &self.covar);
                        }
                        singleton_rows = risk_indices.len();
                        apply_exact_event_moments(
                            &self.covar,
                            &self.weights,
                            u,
                            imat,
                            &death_indices,
                            &linear_predictors,
                            moments.log_denom,
                            &moments.mean,
                            moments.covariance.view(),
                        )
                    };
                } else {
                    if let Some(tree) = tree.as_ref() {
                        risk_indices.clear();
                        risk_indices.extend(tree.active_rows());
                    }
                    loglik += add_exact_event_contribution(
                        &self.covar,
                        &self.weights,
                        u,
                        imat,
                        &death_indices,
                        &risk_indices,
                        &linear_predictors,
                        &log_risk,
                    );
                }
            }
        }
        loglik
    }
}

impl CoxFit {
    /// Centres and scales the covariate columns in place the way each C
    /// fitter does, so that the Newton iterates (and hence iteration counts
    /// and step halving) follow R:
    ///
    /// * `coxfit6.c`: weighted mean, scale `sum w / sum w |x - mean|`;
    /// * `agfit4.c`: the value of the first row of its walk (its per-stratum
    ///   mean loop subtracts inside the loop body, so only the first value
    ///   is ever used) and the same kind of scale over the rows it walks;
    ///   `agreg.fit` then reports the unweighted column means;
    /// * `coxexact.fit`: R's `scale()`, mean and standard deviation;
    /// * `agexact.c`: mean only.
    fn scale_center(&mut self, doscale: &[bool]) {
        let CoxData {
            ref mut covar,
            ref weights,
            ..
        } = self.data;
        let fitter = &self.fitter;
        let nused = covar.nrows();
        let total_weight: f64 = weights.sum();
        for (i, &scale_column) in doscale.iter().enumerate() {
            if !scale_column {
                self.means[i] = 0.0;
                self.scale[i] = 1.0;
                continue;
            }
            let plain_mean = covar.column(i).sum() / nused as f64;
            let (center, mean) = match fitter {
                Fitter::Coxfit6 => {
                    let weighted_mean = (0..nused)
                        .map(|person| weights[person] * covar[(person, i)])
                        .sum::<f64>()
                        / total_weight;
                    (weighted_mean, weighted_mean)
                }
                Fitter::Agfit4 { walk, .. } => (
                    walk.by_stop
                        .first()
                        .map_or(0.0, |first| covar[(first.row, i)]),
                    plain_mean,
                ),
                Fitter::Coxexact | Fitter::Agexact { .. } => (plain_mean, plain_mean),
            };
            covar.column_mut(i).mapv_inplace(|value| value - center);
            self.means[i] = mean;
            let column = covar.column(i);
            let scale = match fitter {
                Fitter::Coxfit6 => mean_abs_scale(weights, column, 0..nused),
                Fitter::Agfit4 { walk, .. } => {
                    mean_abs_scale(weights, column, walk.by_stop.iter().map(|j| j.row))
                }
                Fitter::Coxexact => {
                    let sd = (column.mapv(|v| v * v).sum() / (nused as f64 - 1.0)).sqrt();
                    if sd > 0.0 { 1.0 / sd } else { 1.0 }
                }
                Fitter::Agexact { .. } => 1.0,
            };
            covar.column_mut(i).mapv_inplace(|value| value * scale);
            self.scale[i] = scale;
        }
        for (beta, &scale) in self.beta.iter_mut().zip(&self.scale) {
            *beta /= scale;
        }
    }

    /// Evaluates the log likelihood at `beta`, leaving the score in `u` and
    /// the information matrix (upper triangle) in `imat`.
    fn evaluate(&mut self, beta: &[f64]) -> SurvivalResult<f64> {
        self.u.fill(0.0);
        self.imat.fill(0.0);
        let (data, u, imat) = (&self.data, &mut self.u, &mut self.imat);
        Ok(match &mut self.fitter {
            Fitter::Coxfit6 => data.coxfit6(beta, u, imat),
            Fitter::Agfit4 { walk, risk_set } => data.agfit4(beta, walk, risk_set, u, imat)?,
            Fitter::Coxexact => data.coxexact(beta, u, imat),
            Fitter::Agexact { walk } => data.agexact(beta, walk, u, imat),
        })
    }

    /// Inverts the factored information matrix and returns coefficients,
    /// score and variance to the original covariate scale.
    fn finish_inverse(&mut self) {
        chinv2(&mut self.imat);
        let nvar = self.beta.len();
        for i in 0..nvar {
            self.beta[i] *= self.scale[i];
            self.u[i] /= self.scale[i];
            self.imat[(i, i)] *= self.scale[i] * self.scale[i];
            for j in 0..i {
                self.imat[(j, i)] *= self.scale[i] * self.scale[j];
                self.imat[(i, j)] = self.imat[(j, i)];
            }
        }
    }

    fn state_is_finite(&self, loglik: f64) -> bool {
        loglik.is_finite()
            && self.u.iter().all(|value| value.is_finite())
            && self.imat.iter().all(|value| value.is_finite())
    }

    /// Runs the Newton-Raphson iteration (`coxfit6`, `agfit4`, `coxexact`,
    /// `agexact`).  A singular information matrix zeroes the redundant
    /// directions (`cholesky2`/`chsolve2`) and an exhausted iteration budget
    /// is reported through `flag == 1000`, exactly as R does; the one error
    /// is `agfit4`'s overflow of the risk scores.
    pub(crate) fn fit(&mut self) -> SurvivalResult<()> {
        let nvar = self.beta.len();
        let beta0 = self.beta.clone();
        self.iter = 0;
        self.loglik[0] = self.evaluate(&beta0)?;
        self.loglik[1] = self.loglik[0];
        if nvar == 0 && !self.iterate_empty {
            self.flag = 0;
            return Ok(());
        }

        let mut a = self.u.clone();
        self.flag = cholesky2(&mut self.imat, self.toler);
        chsolve2(&self.imat, &mut a);
        self.sctest = a.iter().zip(&self.u).map(|(ai, ui)| ai * ui).sum();

        if self.max_iter == 0 || !self.loglik[0].is_finite() {
            self.finish_inverse();
            // agexact.c returns flag 0 when no iterations were requested.
            if matches!(self.fitter, Fitter::Agexact { .. }) {
                self.flag = 0;
            }
            self.info = self
                .fitter
                .rescales()
                .map(|rescales| [self.flag, rescales, 0, 0]);
            return Ok(());
        }
        let newbeta = self.beta.iter().zip(&a).map(|(b, a)| b + a).collect();
        if matches!(self.fitter, Fitter::Agfit4 { .. }) {
            self.iterate_agfit4(newbeta, a)
        } else {
            self.iterate(newbeta, a)
        }
    }

    /// The Newton iteration of `coxfit6.c`, `coxexact.c` and `agexact.c`
    /// from the first trial `newbeta`; `a` is scratch for the steps.
    fn iterate(&mut self, mut newbeta: Vec<f64>, mut a: Vec<f64>) -> SurvivalResult<()> {
        let (exact, counting) = match self.fitter {
            Fitter::Coxexact => (true, false),
            Fitter::Agexact { .. } => (true, true),
            _ => (false, false),
        };
        let mut halving = 0usize;
        let mut newlk = self.loglik[1];
        for iter in 1..=self.max_iter {
            self.iter = iter;
            newlk = self.evaluate(&newbeta)?;
            self.flag = cholesky2(&mut self.imat, self.toler);
            let finite = self.state_is_finite(newlk);
            if finite
                && (1.0 - self.loglik[1] / newlk).abs() <= self.eps
                && (!exact || halving == 0)
            {
                self.loglik[1] = newlk;
                self.beta.copy_from_slice(&newbeta);
                self.finish_inverse();
                if !exact && halving > 0 {
                    self.flag = -2;
                }
                return Ok(());
            }
            if exact && iter == self.max_iter {
                break;
            }
            if !finite || newlk < self.loglik[1] {
                halving += 1;
                for (new, old) in newbeta.iter_mut().zip(&self.beta) {
                    *new = if exact {
                        (*new + old) / 2.0
                    } else {
                        (*new + halving as f64 * old) / (halving as f64 + 1.0)
                    };
                }
            } else {
                halving = 0;
                self.loglik[1] = newlk;
                self.beta.copy_from_slice(&newbeta);
                a.copy_from_slice(&self.u);
                chsolve2(&self.imat, &mut a);
                for (new, (old, step)) in newbeta.iter_mut().zip(self.beta.iter().zip(&a)) {
                    *new = old + step;
                }
            }
        }

        // Out of iterations.  agexact.c keeps the last trial, and so does
        // coxexact.c when only one iteration was allowed ("if maxiter = 0 or
        // 1, leave well enough alone"); otherwise coxfit6.c and coxexact.c go
        // back to the last accepted coefficients and refit the information
        // there (both omit the Cholesky factorisation before inverting, a bug
        // this port does not copy).  coxfit6.c's loop counter ends one past
        // `maxiter`.
        if exact && (counting || self.max_iter <= 1) {
            self.loglik[1] = newlk;
            self.beta.copy_from_slice(&newbeta);
        } else if self.max_iter > 1 {
            let beta = self.beta.clone();
            self.loglik[1] = self.evaluate(&beta)?;
            self.flag = cholesky2(&mut self.imat, self.toler);
        }
        if !exact {
            self.iter = self.max_iter + 1;
        }
        self.finish_inverse();
        self.flag = 1000;
        Ok(())
    }

    /// `agfit4.c`'s Newton iteration from the first trial `trial`; `step` is
    /// scratch for the steps.  A step fails when the log likelihood or a
    /// diagonal element of the information is not finite or the rank of the
    /// information differs from its rank at the initial coefficients; the
    /// fit converges only after a full step, and when the iterations run
    /// out it keeps the last trial unless that is worse than the best by
    /// more than `eps`.
    fn iterate_agfit4(&mut self, mut trial: Vec<f64>, mut step: Vec<f64>) -> SurvivalResult<()> {
        let rank = self.flag;
        let (mut halving, mut halvings) = (0, 0);
        let mut ran_out = false;
        let mut newlk = self.loglik[1];
        for iter in 1..=self.max_iter {
            self.iter = iter;
            newlk = self.evaluate(&trial)?;
            let infinite = self.imat.diag().iter().filter(|v| !v.is_finite()).count();
            let rank2 = cholesky2(&mut self.imat, self.toler);
            let fail = infinite + usize::from(!newlk.is_finite()) + rank.abs_diff(rank2) as usize;
            if fail == 0 && halving == 0 && (1.0 - self.loglik[1] / newlk).abs() <= self.eps {
                break;
            }
            if iter == self.max_iter {
                ran_out = true;
                if self.max_iter > 1 && (newlk - self.loglik[1]) / self.loglik[1].abs() < -self.eps
                {
                    trial.copy_from_slice(&self.beta);
                    newlk = self.evaluate(&trial)?;
                    cholesky2(&mut self.imat, self.toler);
                }
                break;
            }
            if fail > 0 || newlk < self.loglik[1] {
                halving += 1;
                halvings += 1;
                let h = f64::from(halving);
                for (new, old) in trial.iter_mut().zip(&self.beta) {
                    *new = (old * h + *new) / (h + 1.0);
                }
            } else {
                halving = 0;
                self.loglik[1] = newlk;
                step.copy_from_slice(&self.u);
                chsolve2(&self.imat, &mut step);
                self.beta.copy_from_slice(&trial);
                for (new, s) in trial.iter_mut().zip(&step) {
                    *new += s;
                }
            }
        }
        self.beta = trial;
        self.loglik[1] = newlk;
        self.finish_inverse();
        self.flag = if ran_out { 1000 } else { rank };
        self.info = self
            .fitter
            .rescales()
            .map(|rescales| [rank, rescales, halvings, i32::from(ran_out)]);
        Ok(())
    }

    pub(crate) fn results(self) -> CoxFitResults {
        CoxFitResults {
            coefficients: self.beta,
            means: self.means,
            score: self.u,
            var: self.imat,
            loglik: self.loglik,
            sctest: self.sctest,
            flag: self.flag,
            iter: self.iter,
            info: self.info,
            order: self.order,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counting_process_order_fixture() -> CoxFit {
        CoxFitBuilder::new(
            Array1::from_vec(vec![2.0, 3.0, 4.0, 2.5, 4.0, 5.0]),
            Array1::from_vec(vec![1, 1, 0, 1, 0, 1]),
            Array2::from_shape_vec(
                (6, 2),
                vec![0.2, 1.0, 0.8, 0.4, 0.5, 1.2, 1.1, 0.3, 0.7, 0.9, 1.3, 0.6],
            )
            .expect("counting-process fixture covariates should have a valid shape"),
        )
        .entry_times(Array1::from_vec(vec![0.5, 1.5, 1.5, 2.0, 0.25, 2.0]))
        .strata(Array1::from_vec(vec![0, 0, 0, 1, 1, 1]))
        .method(TieMethod::Efron)
        .max_iter(10)
        .eps(1e-9)
        .toler(1e-9)
        .build()
        .expect("counting-process fixture should initialize")
    }

    #[test]
    fn tie_method_parses_r_names() {
        assert_eq!(TieMethod::parse(None).unwrap(), TieMethod::Efron);
        assert_eq!(
            TieMethod::parse(Some("Breslow")).unwrap(),
            TieMethod::Breslow
        );
        assert_eq!(TieMethod::parse(Some("exact")).unwrap(), TieMethod::Exact);
        assert!(TieMethod::parse(Some("discrete")).is_err());
        assert_eq!(TieMethod::Efron.r_name(), "efron");
    }

    #[test]
    fn builder_rejects_length_mismatches_and_weighted_exact_fits() {
        let time = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let status = Array1::from_vec(vec![1, 0, 1]);
        let covar = Array2::from_shape_vec((3, 1), vec![0.5, 1.0, 0.3]).unwrap();
        assert!(
            CoxFitBuilder::new(time.clone(), status.clone(), covar.clone())
                .weights(Array1::from_vec(vec![1.0, 2.0]))
                .build()
                .is_err()
        );
        assert!(
            CoxFitBuilder::new(time.clone(), status.clone(), covar.clone())
                .weights(Array1::from_vec(vec![1.0, 2.0, 1.0]))
                .method(TieMethod::Exact)
                .build()
                .is_err()
        );
        assert!(
            CoxFitBuilder::new(time, status, covar)
                .initial_beta(vec![0.0, 0.0])
                .build()
                .is_err()
        );
    }

    #[test]
    fn unsorted_input_matches_sorted_input() {
        let time: Vec<f64> = vec![5.0, 1.0, 4.0, 2.0, 3.0, 6.0, 8.0, 7.0];
        let status: Vec<i32> = vec![1, 1, 0, 0, 1, 0, 0, 1];
        let x: Vec<f64> = vec![0.6, 0.5, 0.8, 1.0, 0.3, 0.4, 0.2, 0.9];
        let strata: Vec<i32> = vec![1, 0, 1, 0, 1, 0, 1, 0];
        let mut order: Vec<usize> = (0..8).collect();
        order.sort_by(|&a, &b| strata[a].cmp(&strata[b]).then(time[a].total_cmp(&time[b])));
        let fit = |idx: &[usize]| {
            let mut engine = CoxFitBuilder::new(
                Array1::from_iter(idx.iter().map(|&i| time[i])),
                Array1::from_iter(idx.iter().map(|&i| status[i])),
                Array2::from_shape_vec((8, 1), idx.iter().map(|&i| x[i]).collect()).unwrap(),
            )
            .strata(Array1::from_iter(idx.iter().map(|&i| strata[i])))
            .method(TieMethod::Efron)
            .build()
            .unwrap();
            engine.fit().unwrap();
            engine.results()
        };
        let unsorted = fit(&(0..8).collect::<Vec<_>>());
        let sorted = fit(&order);
        assert!((unsorted.coefficients[0] - sorted.coefficients[0]).abs() < 1e-12);
        assert!((unsorted.var[(0, 0)] - sorted.var[(0, 0)]).abs() < 1e-12);
        assert_eq!(unsorted.loglik, sorted.loglik);
        assert_eq!(unsorted.order, order);
    }

    #[test]
    fn exact_builder_treats_default_strata_as_one_complete_stratum() {
        let mut fit = CoxFitBuilder::new(
            Array1::from_vec(vec![1.0, 2.0, 3.0]),
            Array1::from_vec(vec![1, 1, 0]),
            Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap(),
        )
        .method(TieMethod::Exact)
        .max_iter(0)
        .build()
        .expect("default-stratum exact fit should initialize");

        fit.fit().unwrap();
        let results = fit.results();
        assert!(results.loglik[0] < 0.0);
        assert!(results.score[0].is_finite());
        assert!(results.var[(0, 0)].is_finite());
    }

    #[test]
    fn converged_coefficients_match_the_reported_log_likelihood() {
        let time = Array1::from_vec((1..=16).map(f64::from).collect());
        let status = Array1::from_vec(vec![1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0]);
        let covariates = vec![
            0.5, 1.2, 1.8, 0.3, 0.2, 2.1, 2.5, 0.8, 0.8, 1.5, 1.5, 0.5, 0.3, 1.8, 2.2, 1.1, 1.0,
            0.9, 0.7, 1.7, 2.0, 0.4, 1.2, 1.3, 0.9, 2.0, 1.6, 0.7, 0.4, 1.4, 2.1, 1.0,
        ];
        let covar = Array2::from_shape_vec((16, 2), covariates).unwrap();

        let mut fit = CoxFitBuilder::new(time.clone(), status.clone(), covar.clone())
            .max_iter(20)
            .eps(1e-5)
            .build()
            .unwrap();
        fit.fit().unwrap();
        let results = fit.results();

        let mut evaluation = CoxFitBuilder::new(time, status, covar)
            .max_iter(0)
            .initial_beta(results.coefficients)
            .build()
            .unwrap();
        evaluation.fit().unwrap();
        assert!((evaluation.results().loglik[0] - results.loglik[1]).abs() < 1e-12);
    }

    #[test]
    fn nonconverged_fit_refactors_information_at_the_last_accepted_beta() {
        let time = Array1::from_vec((1..=8).map(f64::from).collect());
        let status = Array1::from_vec(vec![1, 0, 1, 1, 0, 1, 0, 1]);
        let covar = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.2, 1.0, 0.7, 0.4, 1.2, 1.5, 0.1, 0.8, 1.0, 1.1, 0.3, 1.8, 0.9, 2.0, 0.5,
            ],
        )
        .unwrap();
        let mut fit = CoxFitBuilder::new(time.clone(), status.clone(), covar.clone())
            .max_iter(2)
            .eps(1e-12)
            .build()
            .unwrap();
        fit.fit().unwrap();
        let results = fit.results();
        assert_eq!(results.flag, 1000);
        // coxfit6.c's loop counter ends one past iter.max.
        assert_eq!(results.iter, 3);

        let mut evaluation = CoxFitBuilder::new(time, status, covar)
            .max_iter(0)
            .initial_beta(results.coefficients)
            .build()
            .unwrap();
        evaluation.fit().unwrap();
        let evaluated = evaluation.results();
        assert!((evaluated.loglik[0] - results.loglik[1]).abs() < 1e-12);
        for i in 0..2 {
            assert!((evaluated.score[i] - results.score[i]).abs() < 1e-12);
            for j in 0..2 {
                assert!((evaluated.var[(i, j)] - results.var[(i, j)]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn exact_single_iteration_keeps_the_trial_step_like_coxexact() {
        // R survival 3.8.11: coxph(Surv(time, status) ~ x1 + x2, ties = "exact",
        // iter.max = 1).  coxexact.c breaks out at iter == maxiter and, with
        // maxiter <= 1, reports the trial coefficients beta0 + u and their
        // log-likelihood instead of refitting at the last accepted step.
        let time = Array1::from_vec(vec![
            1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0,
        ]);
        let status = Array1::from_vec(vec![1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0]);
        let covar = Array2::from_shape_vec(
            (12, 2),
            vec![
                0.5, 1.0, 1.2, 0.9, 1.8, 0.7, 0.3, 1.7, 0.2, 2.0, 2.1, 0.4, 2.5, 1.2, 0.8, 1.3,
                0.8, 0.9, 1.5, 2.0, 1.5, 1.6, 0.5, 0.7,
            ],
        )
        .unwrap();
        let mut fit = CoxFitBuilder::new(time, status, covar)
            .method(TieMethod::Exact)
            .max_iter(1)
            .build()
            .unwrap();
        fit.fit().unwrap();
        let results = fit.results();

        assert_eq!(results.iter, 1);
        assert_eq!(results.flag, 1000);
        let expected_coef = [0.349960180979692, -0.21357557725758675];
        let expected_loglik = [-13.748889870622378, -13.508141594122979];
        let expected_var = [
            [0.2913411531607994, 0.0038229400661355514],
            [0.0038229400661355514, 0.5565122484267278],
        ];
        for (actual, expected) in results.coefficients.iter().zip(&expected_coef) {
            assert!((actual - expected).abs() < 1e-8);
        }
        for (actual, expected) in results.loglik.iter().zip(&expected_loglik) {
            assert!((actual - expected).abs() < 1e-8);
        }
        for (i, row) in expected_var.iter().enumerate() {
            for (j, expected) in row.iter().enumerate() {
                assert!((results.var[(i, j)] - expected).abs() < 1e-6);
            }
        }
        assert!((results.sctest - 0.49084981706979436).abs() < 1e-8);
    }

    #[test]
    fn counting_process_tied_methods_fit_with_entry_times() {
        for method in [TieMethod::Efron, TieMethod::Exact] {
            let mut cox = CoxFitBuilder::new(
                Array1::from_vec(vec![2.0, 2.0, 3.0, 4.0, 4.0, 5.0]),
                Array1::from_vec(vec![1, 1, 0, 1, 1, 0]),
                Array2::from_shape_vec((6, 1), vec![0.0, 0.4, 0.2, 1.0, 1.4, 0.8]).unwrap(),
            )
            .entry_times(Array1::from_vec(vec![0.0, 0.5, 0.0, 1.0, 2.0, 0.0]))
            .method(method)
            .max_iter(5)
            .eps(1e-8)
            .toler(1e-8)
            .build()
            .unwrap();
            cox.fit().unwrap();
            let results = cox.results();
            assert!(results.coefficients[0].is_finite());
            assert!(results.var[(0, 0)].is_finite());
            assert!(results.loglik[1].is_finite());
        }
    }

    #[test]
    fn agfit4_walks_rows_by_decreasing_stop_and_entry_time() {
        let fit = counting_process_order_fixture();
        assert_eq!(fit.order, vec![0, 1, 2, 3, 4, 5]);
        let Fitter::Agfit4 { walk, .. } = &fit.fitter else {
            panic!("a counting-process Efron fit is agfit4's");
        };
        assert_eq!(walk.bounds, vec![(0, 3), (3, 6)]);
        let rows = |walk: &AgWalk| walk.by_stop.iter().map(|j| j.row).collect::<Vec<_>>();
        assert_eq!(rows(walk), vec![2, 1, 0, 5, 4, 3]);
        // Entry ties (1.5 twice, 2.0 twice) keep the stop-time order.
        assert_eq!(
            walk.by_entry,
            vec![(1.5, 2), (1.5, 1), (0.5, 0), (2.0, 5), (2.0, 3), (0.25, 4)]
        );

        // A row spanning no death time is left out of both walks.
        let fit = CoxFitBuilder::new(
            Array1::from_vec(vec![2.0, 3.0, 2.5, 4.0]),
            Array1::from_vec(vec![1, 0, 0, 1]),
            Array2::from_shape_vec((4, 1), vec![0.1, 0.2, 0.3, 0.4]).unwrap(),
        )
        .entry_times(Array1::from_vec(vec![0.0, 2.2, 2.0, 1.0]))
        .method(TieMethod::Breslow)
        .build()
        .unwrap();
        let Fitter::Agfit4 { walk, .. } = &fit.fitter else {
            panic!("a counting-process Breslow fit is agfit4's");
        };
        // Sorted positions: 0 = row 0 (2.0], 1 = row 2 (2.0, 2.5],
        // 2 = row 1 (2.2, 3.0], 3 = row 3 (1.0, 4.0].
        assert_eq!(rows(walk), vec![3, 0]);
        assert_eq!(walk.by_entry, vec![(1.0, 3), (0.0, 0)]);
    }

    #[test]
    fn counting_process_evaluation_is_deterministic() {
        let mut fit = counting_process_order_fixture();
        let beta = [0.2, -0.15];
        let first_loglik = fit.evaluate(&beta).unwrap();
        let first_score = fit.u.clone();
        let first_information = fit.imat.clone();
        let second_loglik = fit.evaluate(&beta).unwrap();
        assert_eq!(second_loglik, first_loglik);
        assert_eq!(fit.u, first_score);
        assert_eq!(fit.imat, first_information);
    }

    #[test]
    fn collinear_tied_efron_fit_reports_rank_and_zero_alias_variance() {
        let time = Array1::from_vec(vec![1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let status = Array1::from_vec(vec![1, 1, 1, 0, 1, 1, 0, 1, 0, 1]);
        let x1 = [0.0, 0.4, 0.8, 0.2, 1.0, 1.4, 0.6, 1.2, 1.6, 1.8];
        let x2 = [0.2, 0.16, 0.62, -0.07, 0.95, 0.61, 0.49, 0.68, 1.24, 0.97];
        let mut covariates = Vec::with_capacity(30);
        for (&first, &second) in x1.iter().zip(&x2) {
            covariates.extend_from_slice(&[first, second, first + second]);
        }
        let covar = Array2::from_shape_vec((10, 3), covariates).unwrap();
        let mut fit = CoxFitBuilder::new(time, status, covar)
            .method(TieMethod::Efron)
            .max_iter(50)
            .eps(1e-9)
            .toler(1e-12)
            .build()
            .unwrap();
        fit.fit().unwrap();
        let results = fit.results();

        assert_eq!(results.flag, 2);
        assert_eq!(results.iter, 4);
        let expected_beta = [-2.3468678070137803, 0.5775928193386433, 0.0];
        let expected_variance = [
            [3.9806704210981683, -4.116538359266848, 0.0],
            [-4.116538359266848, 6.056737323572425, 0.0],
            [0.0, 0.0, 0.0],
        ];
        for (i, expected_row) in expected_variance.iter().enumerate() {
            assert!((results.coefficients[i] - expected_beta[i]).abs() < 1e-10);
            for (j, expected) in expected_row.iter().enumerate() {
                assert!((results.var[(i, j)] - expected).abs() < 1e-10);
            }
        }
        assert!((results.loglik[0] - -11.079060882340368).abs() < 1e-12);
        assert!((results.loglik[1] - -9.002136268091796).abs() < 1e-12);
    }

    #[test]
    fn null_model_evaluates_the_log_likelihood_only() {
        let mut fit = CoxFitBuilder::new(
            Array1::from_vec(vec![1.0, 2.0, 3.0]),
            Array1::from_vec(vec![1, 1, 0]),
            Array2::zeros((3, 0)),
        )
        .build()
        .unwrap();
        fit.fit().unwrap();
        let results = fit.results();
        assert!(results.coefficients.is_empty());
        assert!((results.loglik[0] - (-(3.0_f64.ln()) - 2.0_f64.ln())).abs() < 1e-12);
        assert_eq!(results.iter, 0);
    }

    /// Counting-process data of the form `(entry, time]` with two
    /// covariates given row by row.
    fn counting_builder(time: &[f64], entry: &[f64], status: &[i32], x: &[f64]) -> CoxFitBuilder {
        CoxFitBuilder::new(
            Array1::from_vec(time.to_vec()),
            Array1::from_vec(status.to_vec()),
            Array2::from_shape_vec((time.len(), 2), x.to_vec()).unwrap(),
        )
        .entry_times(Array1::from_vec(entry.to_vec()))
    }

    const CASE_1520_TIME: [f64; 11] =
        [15.0, 4.0, 25.0, 8.0, 18.0, 6.0, 9.0, 14.0, 11.0, 64.0, 19.0];
    const CASE_1520_ENTRY: [f64; 11] = [9.0, 2.0, 5.0, 1.0, 14.0, 3.0, 7.0, 6.0, 3.0, 39.0, 2.0];
    const CASE_1520_STATUS: [i32; 11] = [1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1];
    const CASE_1520_X: [f64; 22] = [
        -3.397587783734524,
        -1.2399609396099633,
        -4.001456275604981,
        -0.9612195837228918,
        1.2485501179741572,
        -0.36802848351666784,
        -1.922243349085923,
        -0.29505277170923994,
        -0.6191833991904704,
        -0.432759419947449,
        0.9541844726560894,
        3.636570321763567,
        0.9722226187833938,
        -1.5241526403027048,
        -2.083451024212451,
        1.3906982107759198,
        -0.557594058080056,
        1.1836246992905952,
        -6.582305367737362,
        2.3943458650099303,
        0.9175209607015593,
        -1.2770581196683075,
    ];

    fn case_1520() -> CoxFitBuilder {
        counting_builder(
            &CASE_1520_TIME,
            &CASE_1520_ENTRY,
            &CASE_1520_STATUS,
            &CASE_1520_X,
        )
    }

    #[test]
    fn agfit4_keeps_full_precision_when_risk_scores_span_many_magnitudes() {
        // R 3.8-12, agreg.fit: coxph(Surv(entry, time, status) ~ x1 + x2,
        // init = init, timefix = FALSE).
        let init = vec![-0.10734736591853931, 0.5817618903234776];
        for method in [TieMethod::Efron, TieMethod::Breslow] {
            let mut fit = case_1520()
                .method(method)
                .initial_beta(init.clone())
                .build()
                .unwrap();
            fit.fit().unwrap();
            let results = fit.results();
            assert_eq!(results.iter, 7);
            assert_eq!(results.flag, 2);
            assert_eq!(results.info, Some([2, 0, 0, 0]));
            for (actual, expected) in results
                .coefficients
                .iter()
                .zip([-2.7295106722, 2.39104660699])
            {
                assert!((actual - expected).abs() < 1e-9, "{actual} vs {expected}");
            }
            assert!((results.loglik[0] - -6.73608164457).abs() < 1e-10);
            assert!((results.loglik[1] - -2.02753429303).abs() < 1e-10);
        }
        let mut at_r = case_1520()
            .method(TieMethod::Efron)
            .initial_beta(vec![-2.729510672, 2.391046607])
            .max_iter(0)
            .build()
            .unwrap();
        at_r.fit().unwrap();
        assert!((at_r.results().loglik[0] - -2.02753429303384).abs() < 1e-12);
    }

    #[test]
    fn agfit4_recentres_the_risk_scores_so_offsets_cancel() {
        // R 3.8-12, agreg.fit with offset 0 and -750 (init 0): the centre of
        // the risk scores moves 9 times and the fit is the offset-free one,
        // (-2.72951067339449, 2.39104660779831) after 8 iterations.
        let mut fit = case_1520()
            .method(TieMethod::Efron)
            .offset(Array1::from_elem(11, -750.0))
            .build()
            .unwrap();
        fit.fit().unwrap();
        let results = fit.results();
        assert_eq!(results.iter, 8);
        assert_eq!(results.info, Some([2, 9, 0, 0]));
        for (actual, expected) in results
            .coefficients
            .iter()
            .zip([-2.72951067339457, 2.39104660779839])
        {
            assert!((actual - expected).abs() < 1e-11, "{actual} vs {expected}");
        }
        assert!((results.loglik[0] - -7.78322401633604).abs() < 1e-11);
        assert!((results.loglik[1] - -2.02753429303382).abs() < 1e-11);
    }

    #[test]
    fn agfit4_never_converges_while_step_halving() {
        // R 3.8-12: the coefficients run off to infinity, iter 20,
        // info = (2, 26, 2, 1), loglik -7.44594093298 -> -5.09e-07.
        let time = [43.0, 7.0, 2.0, 1.0, 13.0, 3.0, 41.0, 4.0, 10.0];
        let entry = [28.0, 6.0, 1.0, 0.0, 4.0, 0.0, 31.0, 1.0, 5.0];
        let x = [
            0.055792698117579004,
            -1.7304770998879826,
            -1.3617180976588907,
            -0.9673886866286651,
            0.8665565328654526,
            0.9626229915832482,
            0.25602914821193906,
            -0.9861502050896039,
            -0.3219648826811489,
            0.2535879462914172,
            3.191390489555803,
            2.1186597213685254,
            -0.6752767149783483,
            -3.530622409131793,
            1.538739604857286,
            1.4570505303916683,
            0.8826048048813887,
            0.8028742413755462,
        ];
        let mut fit = counting_builder(&time, &entry, &[1; 9], &x)
            .method(TieMethod::Efron)
            .initial_beta(vec![-1.7291049186754255, -1.9022957381326357])
            .build()
            .unwrap();
        fit.fit().unwrap();
        let results = fit.results();
        assert_eq!(results.iter, 20);
        assert_eq!(results.flag, 1000);
        assert_eq!(results.info, Some([2, 26, 2, 1]));
        assert!((results.loglik[0] - -7.44594093298).abs() < 1e-10);
        assert!(results.loglik[1] > -1e-5 && results.loglik[1] < 0.0);
        assert!(results.coefficients[0] > 100.0 && results.coefficients[1] < -200.0);
    }

    #[test]
    fn agfit4_stops_on_an_overflow_of_the_risk_scores_like_r() {
        // R 3.8-12 stops with "exp overflow due to covariates".
        let time = [
            4.0, 9.0, 3.0, 18.0, 1.0, 13.0, 17.0, 3.0, 19.0, 7.0, 19.0, 16.0, 1.0, 4.0, 28.0, 2.0,
            9.0, 9.0, 10.0, 4.0, 30.0, 2.0, 11.0, 6.0, 18.0, 2.0, 1.0, 9.0, 1.0, 7.0, 3.0, 7.0,
            29.0, 30.0, 6.0,
        ];
        let entry = [
            0.0, 3.0, 0.0, 0.0, 0.0, 6.0, 14.0, 2.0, 7.0, 3.0, 12.0, 8.0, 0.0, 2.0, 19.0, 1.0, 1.0,
            7.0, 1.0, 0.0, 24.0, 0.0, 1.0, 3.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 5.0, 9.0, 14.0,
            2.0,
        ];
        let status = [
            0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1,
            1, 1, 0, 0, 1, 1,
        ];
        let x = [
            3.6165533269227357,
            -0.4879961376492757,
            -0.07374622784398623,
            -3.4507779043965923,
            1.6358656834944607,
            -0.8598919210443141,
            -1.628034953476259,
            3.044605832130217,
            -0.1381907769240786,
            -2.290022336297825,
            -5.169547312611223,
            -1.2426684167216626,
            3.3797074865680248,
            0.6998353670379976,
            -0.9738348201398073,
            -1.4788787564880854,
            -1.1408907245660664,
            -2.449214977809486,
            2.977876199920989,
            -0.5818949605748973,
            0.3239725741127991,
            4.027702794420345,
            4.526175051724141,
            0.4359046589960573,
            -0.843288251132816,
            4.678555461786062,
            0.2663470996748524,
            1.292766192793381,
            2.5523899200933173,
            -0.7906935015141873,
            -3.443563059344763,
            1.9884475813699718,
            2.9034185242739685,
            -1.4551660130081876,
            -3.6622036738115136,
            -2.4905286981434718,
            -3.01730333128785,
            1.214847903113503,
            -3.2548423663121304,
            1.2155411367063176,
            3.999558195625989,
            1.0942043157770942,
            -3.4069257920971605,
            -4.734540598859402,
            -6.436541308269048,
            0.19070655020480456,
            0.6682351070805835,
            3.5491378111610317,
            0.9147200886960943,
            -6.385859270136145,
            -1.1551436516541982,
            1.6647006754981148,
            -0.22948718455017655,
            -1.0099375772708195,
            -0.15953077748546501,
            -0.42787448803560046,
            1.2554503602389138,
            -0.04548069174472863,
            0.7710867253472405,
            -2.8163031265829344,
            -2.178341021716092,
            2.5274301278230458,
            -0.015919602663027263,
            -4.6923141276901745,
            -0.018512404298677402,
            4.250801466919925,
            -0.8491604697861556,
            -1.1548651559308794,
            2.5756076923351796,
            2.5733307079845056,
        ];
        for method in [TieMethod::Efron, TieMethod::Breslow] {
            let mut fit = counting_builder(&time, &entry, &status, &x)
                .method(method)
                .initial_beta(vec![-7.291479784304963, 0.7249413356562676])
                .build()
                .unwrap();
            let error = fit.fit().unwrap_err();
            assert_eq!(error.to_string(), "exp overflow due to covariates");
        }
    }

    /// survSplit-style data: subject `i` is followed in unit intervals up
    /// to `1 + 53 i mod 97`, dies there unless `i` is a multiple of 3, and
    /// has the time-varying covariate `z = x_i (t - 1) / 5` with
    /// `x_i = (37 i mod 101) / 50`.
    fn split_data() -> CoxFitBuilder {
        let (mut time, mut entry, mut status, mut z) = (vec![], vec![], vec![], vec![]);
        for i in 0..120 {
            let x = ((i * 37) % 101) as f64 / 50.0;
            let last = 1 + (i * 53) % 97;
            for t in 1..=last {
                entry.push(f64::from(t - 1));
                time.push(f64::from(t));
                status.push(i32::from(i % 3 != 0 && t == last));
                z.push(x * f64::from(t - 1) / 5.0);
            }
        }
        let n = time.len();
        CoxFitBuilder::new(
            Array1::from_vec(time),
            Array1::from_vec(status),
            Array2::from_shape_vec((n, 1), z).unwrap(),
        )
        .entry_times(Array1::from_vec(entry))
    }

    #[test]
    fn agfit4_log_likelihood_of_survsplit_data_matches_r() {
        // R 3.8-12: coxph(Surv(start, stop, status) ~ z, init = beta,
        // iter.max = 0, timefix = FALSE)$loglik[1] on the 5769 rows.
        let expected = [
            (0.5, -478.751116295908, -478.888803621125),
            (1.0, -743.829243873272, -743.94983459056),
            (2.0, -1321.85189636427, -1321.94656960294),
        ];
        // An offset of -750 cancels from the likelihood but takes the walk
        // that recentres the risk scores.
        for (beta, efron, breslow) in expected {
            for (method, value) in [(TieMethod::Efron, efron), (TieMethod::Breslow, breslow)] {
                for offset in [0.0, -750.0] {
                    let builder = split_data();
                    let n = builder.time.len();
                    let mut fit = builder
                        .method(method)
                        .initial_beta(vec![beta])
                        .offset(Array1::from_elem(n, offset))
                        .max_iter(0)
                        .build()
                        .unwrap();
                    fit.fit().unwrap();
                    let loglik = fit.results().loglik[0];
                    assert!(
                        (loglik - value).abs() < 1e-9,
                        "{method:?} beta {beta} offset {offset}: {loglik}"
                    );
                }
            }
        }
    }
}
