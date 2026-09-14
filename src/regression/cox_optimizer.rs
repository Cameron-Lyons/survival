//! Newton-Raphson engine for the Cox partial likelihood.
//!
//! Ports of R survival's C fitters, selected by the data and tie method:
//!
//! * `src/coxfit6.c` — right-censored data, Breslow and Efron ties;
//! * `src/agfit4.c` — (start, stop] counting-process data, Breslow and Efron;
//! * `src/coxexact.c` — right-censored data, exact partial likelihood;
//! * `src/agexact.c` — counting-process data, exact partial likelihood.
//!
//! The four share one iteration loop (`CoxFit::fit`) that follows the C
//! sources' convergence and step-halving rules: `coxfit6`/`agfit4` halve
//! ever more aggressively (`(newbeta + halving * beta) / (halving + 1)`) and
//! accept convergence during halving with flag `-2`, while the exact fitters
//! halve by `1/2`, never declare convergence mid-halving, and stop at
//! `iter == maxiter` without a final step.
//!
//! Covariates are centred and scaled as the C code does (`doscale`), so the
//! Newton steps are well conditioned; coefficients, the score vector and the
//! variance are returned on the original scale.  Data may arrive in any row
//! order: the builder sorts by (stratum, time) once and every result is
//! order independent.

use crate::constants::{COX_CONVERGENCE_TOLERANCE, COX_MAX_ITER, COX_RANK_TOLERANCE};
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::matrix::{chinv2, cholesky2, chsolve2};
use ndarray::{Array1, Array2};
use pyo3::prelude::*;

use super::exact_ties::{ExactRiskAccumulator, exact_tied_moments};

/// Tie handling of the partial likelihood (R's `coxph(ties = )`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[pyclass(eq, eq_int, from_py_object)]
pub enum TieMethod {
    Breslow,
    Efron,
    Exact,
}

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
        let entry_times = self.entry_times.as_ref().map(gather);
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
        let entry_order = entry_times
            .as_ref()
            .map(|entry| entry_order_by_stratum(entry, &stratum_end));

        let mut fit = CoxFit {
            time,
            status,
            entry_times,
            entry_order,
            covar,
            stratum_end: Array1::from_vec(stratum_end),
            offset,
            weights,
            method: self.method,
            max_iter: self.max_iter,
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
            order,
        };
        fit.scale_center(&doscale);
        Ok(fit)
    }
}

/// Per stratum, rows ordered by decreasing entry time (ties by decreasing
/// position), the `sort1` order of `agfit4.c`.
fn entry_order_by_stratum(entry_times: &Array1<f64>, stratum_end: &[i32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..entry_times.len()).collect();
    let mut start = 0;
    for end in 0..stratum_end.len() {
        if stratum_end[end] != 1 {
            continue;
        }
        order[start..=end].sort_by(|&lhs, &rhs| {
            entry_times[rhs]
                .total_cmp(&entry_times[lhs])
                .then_with(|| rhs.cmp(&lhs))
        });
        start = end + 1;
    }
    order
}

/// A Cox model ready to iterate: sorted, centred and scaled data plus the
/// current state of the Newton iteration.
pub(crate) struct CoxFit {
    time: Array1<f64>,
    status: Array1<i32>,
    entry_times: Option<Array1<f64>>,
    entry_order: Option<Vec<usize>>,
    covar: Array2<f64>,
    /// `1` on the last row of each stratum (the C `strata` convention).
    stratum_end: Array1<i32>,
    offset: Array1<f64>,
    weights: Array1<f64>,
    method: TieMethod,
    max_iter: usize,
    eps: f64,
    toler: f64,
    scale: Vec<f64>,
    means: Vec<f64>,
    beta: Vec<f64>,
    u: Vec<f64>,
    imat: Array2<f64>,
    loglik: [f64; 2],
    sctest: f64,
    flag: i32,
    iter: usize,
    order: Vec<usize>,
}

fn add_risk_sums(
    covar: &Array2<f64>,
    person: usize,
    risk: f64,
    denom: &mut f64,
    a: &mut [f64],
    cmat: &mut Array2<f64>,
) {
    *denom += risk;
    for i in 0..a.len() {
        let risk_covar_i = risk * covar[(person, i)];
        a[i] += risk_covar_i;
        for j in 0..=i {
            cmat[(i, j)] += risk_covar_i * covar[(person, j)];
        }
    }
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
    covariance: &Array2<f64>,
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
        &moments.covariance,
    )
}

impl CoxFit {
    /// Centres and scales the covariate columns in place the way each C
    /// fitter does, so that the Newton iterates (and hence iteration counts
    /// and step halving) follow R:
    ///
    /// * `coxfit6.c`: weighted mean, scale `sum w / sum w |x - mean|`;
    /// * `agfit4.c`: per-stratum weighted mean, the same overall scale;
    ///   `agreg.fit` then reports the unweighted column means;
    /// * `coxexact.fit`: R's `scale()`, mean and standard deviation;
    /// * `agexact.c`: mean only.
    fn scale_center(&mut self, doscale: &[bool]) {
        let nvar = self.covar.ncols();
        let nused = self.covar.nrows();
        let counting = self.entry_times.is_some();
        let exact = self.method == TieMethod::Exact;
        let total_weight: f64 = self.weights.sum();
        for (i, &scale_column) in doscale.iter().enumerate().take(nvar) {
            if !scale_column {
                self.means[i] = 0.0;
                self.scale[i] = 1.0;
                continue;
            }
            let plain_mean = self.covar.column(i).sum() / nused as f64;
            let weighted_mean = (0..nused)
                .map(|person| self.weights[person] * self.covar[(person, i)])
                .sum::<f64>()
                / total_weight;
            if counting && !exact {
                // agfit4.c centres within each stratum.
                let mut start = 0;
                for end in 0..nused {
                    if self.stratum_end[end] != 1 {
                        continue;
                    }
                    let weight: f64 = (start..=end).map(|p| self.weights[p]).sum();
                    let mean = (start..=end)
                        .map(|p| self.weights[p] * self.covar[(p, i)])
                        .sum::<f64>()
                        / weight;
                    for p in start..=end {
                        self.covar[(p, i)] -= mean;
                    }
                    start = end + 1;
                }
                self.means[i] = plain_mean;
            } else {
                let mean = if exact { plain_mean } else { weighted_mean };
                for person in 0..nused {
                    self.covar[(person, i)] -= mean;
                }
                self.means[i] = mean;
            }
            let scale = match (exact, counting) {
                (true, true) => 1.0,
                (true, false) => {
                    let sd =
                        (self.covar.column(i).mapv(|v| v * v).sum() / (nused as f64 - 1.0)).sqrt();
                    if sd > 0.0 { 1.0 / sd } else { 1.0 }
                }
                (false, _) => {
                    let abs_sum: f64 = (0..nused)
                        .map(|person| self.weights[person] * self.covar[(person, i)].abs())
                        .sum();
                    if abs_sum > 0.0 {
                        total_weight / abs_sum
                    } else {
                        1.0
                    }
                }
            };
            for person in 0..nused {
                self.covar[(person, i)] *= scale;
            }
            self.scale[i] = scale;
        }
        for (beta, &scale) in self.beta.iter_mut().zip(&self.scale) {
            *beta /= scale;
        }
    }

    fn linear_predictors(&self, beta: &[f64]) -> Vec<f64> {
        (0..self.covar.nrows())
            .map(|person| {
                self.offset[person]
                    + beta
                        .iter()
                        .enumerate()
                        .fold(0.0, |sum, (i, &b)| sum + b * self.covar[(person, i)])
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

    /// `coxexact.c`: exact partial likelihood for right-censored data.  The
    /// risk set grows from the largest time downwards; a single death uses
    /// the running moments, tied deaths the subset dynamic programme.
    fn iterate_right_censored_exact(&mut self, beta: &[f64]) -> f64 {
        self.u.fill(0.0);
        self.imat.fill(0.0);
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
                            &mut self.u,
                            &mut self.imat,
                            &death_indices,
                            &linear_predictors,
                            singleton_moments.log_denom,
                            &singleton_moments.mean,
                            &singleton_moments.covariance,
                        )
                    } else {
                        add_exact_event_contribution(
                            &self.covar,
                            &self.weights,
                            &mut self.u,
                            &mut self.imat,
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

    /// `agexact.c`: exact partial likelihood for (start, stop] data.  As in
    /// the C code the risk set of each death time (`start < t <= stop`) is
    /// gathered afresh, which keeps the conditional moments exact even when
    /// risk scores span many orders of magnitude; the cost is
    /// `O(deaths * n)` per evaluation, as in R.
    fn iterate_counting_process_exact(&mut self, beta: &[f64]) -> f64 {
        let Some(entry_times) = self.entry_times.as_ref() else {
            return self.iterate_right_censored_exact(beta);
        };
        self.u.fill(0.0);
        self.imat.fill(0.0);
        let (linear_predictors, log_risk) = self.exact_predictors(beta);
        let mut loglik = 0.0;
        let mut stratum_start = 0usize;
        let mut death_indices = Vec::new();
        let mut risk_indices = Vec::new();

        for stratum_end in 0..self.covar.nrows() {
            if self.stratum_end[stratum_end] != 1 {
                continue;
            }
            let mut time_end = stratum_end;
            loop {
                let event_time = self.time[time_end];
                let mut time_start = time_end;
                while time_start > stratum_start && self.time[time_start - 1] == event_time {
                    time_start -= 1;
                }
                death_indices.clear();
                death_indices
                    .extend((time_start..=time_end).filter(|&person| self.status[person] != 0));
                if !death_indices.is_empty() {
                    risk_indices.clear();
                    risk_indices.extend((stratum_start..=stratum_end).filter(|&person| {
                        entry_times[person] < event_time && self.time[person] >= event_time
                    }));
                    loglik += if death_indices.len() == 1 && risk_indices.len() > 1 {
                        let mut moments = ExactRiskAccumulator::new(self.covar.ncols());
                        for &person in &risk_indices {
                            moments.add(person, log_risk[person], &self.covar);
                        }
                        apply_exact_event_moments(
                            &self.covar,
                            &self.weights,
                            &mut self.u,
                            &mut self.imat,
                            &death_indices,
                            &linear_predictors,
                            moments.log_denom,
                            &moments.mean,
                            &moments.covariance,
                        )
                    } else {
                        add_exact_event_contribution(
                            &self.covar,
                            &self.weights,
                            &mut self.u,
                            &mut self.imat,
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

    /// `coxfit6_iter`: one evaluation of the log likelihood, score and
    /// information for right-censored data with Breslow or Efron ties.
    fn iterate_right_censored(&mut self, beta: &[f64]) -> f64 {
        let nvar = self.covar.ncols();
        let nused = self.covar.nrows();
        let method = self.method;
        self.u.fill(0.0);
        self.imat.fill(0.0);
        let mut a = vec![0.0; nvar];
        let mut a2 = vec![0.0; nvar];
        let mut cmat = Array2::zeros((nvar, nvar));
        let mut cmat2 = Array2::zeros((nvar, nvar));
        let mut loglik = 0.0;
        let mut denom = 0.0;
        let zbeta = self.linear_predictors(beta);
        let risk: Vec<f64> = zbeta
            .iter()
            .zip(self.weights.iter())
            .map(|(&zb, &w)| zb.exp() * w)
            .collect();

        let mut person = nused as isize - 1;
        while person >= 0 {
            let person_idx = person as usize;
            if self.stratum_end[person_idx] == 1 {
                a.fill(0.0);
                cmat.fill(0.0);
                denom = 0.0;
            }
            let dtime = self.time[person_idx];
            let mut ndead = 0;
            let mut deadwt = 0.0;
            let mut denom2 = 0.0;
            while person >= 0 && self.time[person as usize] == dtime {
                let p = person as usize;
                if self.status[p] == 0 {
                    add_risk_sums(&self.covar, p, risk[p], &mut denom, &mut a, &mut cmat);
                } else {
                    ndead += 1;
                    deadwt += self.weights[p];
                    loglik += self.weights[p] * zbeta[p];
                    for i in 0..nvar {
                        self.u[i] += self.weights[p] * self.covar[(p, i)];
                    }
                    add_risk_sums(&self.covar, p, risk[p], &mut denom2, &mut a2, &mut cmat2);
                }
                person -= 1;
                if person >= 0 && self.stratum_end[person as usize] == 1 {
                    break;
                }
            }
            if ndead > 0 {
                if method == TieMethod::Breslow || ndead == 1 {
                    denom += denom2;
                    loglik -= deadwt * denom.ln();
                    for i in 0..nvar {
                        a[i] += a2[i];
                        let temp = a[i] / denom;
                        self.u[i] -= deadwt * temp;
                        for j in 0..=i {
                            cmat[(i, j)] += cmat2[(i, j)];
                            let val = deadwt * (cmat[(i, j)] - temp * a[j]) / denom;
                            self.imat[(j, i)] += val;
                            if i != j {
                                self.imat[(i, j)] += val;
                            }
                        }
                    }
                } else {
                    let death_count = ndead as f64;
                    let risk_fraction = denom2 / death_count;
                    let weight_average = deadwt / death_count;
                    for _ in 0..ndead {
                        denom += risk_fraction;
                        loglik -= weight_average * denom.ln();
                        for i in 0..nvar {
                            a[i] += a2[i] / death_count;
                            let temp = a[i] / denom;
                            self.u[i] -= weight_average * temp;
                            for j in 0..=i {
                                cmat[(i, j)] += cmat2[(i, j)] / death_count;
                                let val = weight_average * (cmat[(i, j)] - temp * a[j]) / denom;
                                self.imat[(j, i)] += val;
                                if i != j {
                                    self.imat[(i, j)] += val;
                                }
                            }
                        }
                    }
                }
                a2.fill(0.0);
                cmat2.fill(0.0);
            }
        }
        loglik
    }

    /// `agfit4.c`'s accumulation loop for (start, stop] data: rows enter the
    /// risk-set sums from the largest stop time downwards and the rows whose
    /// entry time is at or after the current death time are subtracted.
    fn iterate_counting_process(&mut self, beta: &[f64]) -> f64 {
        let Some(entry_times) = self.entry_times.as_ref() else {
            return self.iterate_right_censored(beta);
        };
        let entry_order = self
            .entry_order
            .as_deref()
            .expect("entry order accompanies counting-process entry times");
        let nvar = self.covar.ncols();
        let nused = self.covar.nrows();
        let method = self.method;
        self.u.fill(0.0);
        self.imat.fill(0.0);
        let zbeta = self.linear_predictors(beta);
        let risk: Vec<f64> = zbeta
            .iter()
            .zip(self.weights.iter())
            .map(|(&zb, &w)| zb.exp() * w)
            .collect();

        let mut loglik = 0.0;
        let mut stratum_start = 0usize;
        let mut death_a = vec![0.0; nvar];
        let mut death_cmat: Array2<f64> = Array2::zeros((nvar, nvar));
        let mut event_a = vec![0.0; nvar];
        let mut event_cmat: Array2<f64> = Array2::zeros((nvar, nvar));
        for stratum_end in 0..nused {
            if self.stratum_end[stratum_end] != 1 {
                continue;
            }
            let start_order = &entry_order[stratum_start..=stratum_end];
            let mut stop_denom = 0.0;
            let mut stop_a = vec![0.0; nvar];
            let mut stop_cmat: Array2<f64> = Array2::zeros((nvar, nvar));
            let mut unentered_denom = 0.0;
            let mut unentered_a = vec![0.0; nvar];
            let mut unentered_cmat: Array2<f64> = Array2::zeros((nvar, nvar));
            let mut stop_ptr = stratum_end as isize;
            let mut start_ptr = 0usize;
            let mut time_end = stratum_end;

            loop {
                let event_time = self.time[time_end];
                while stop_ptr >= stratum_start as isize
                    && self.time[stop_ptr as usize] >= event_time
                {
                    let person = stop_ptr as usize;
                    add_risk_sums(
                        &self.covar,
                        person,
                        risk[person],
                        &mut stop_denom,
                        &mut stop_a,
                        &mut stop_cmat,
                    );
                    stop_ptr -= 1;
                }
                while start_ptr < start_order.len()
                    && entry_times[start_order[start_ptr]] >= event_time
                {
                    let person = start_order[start_ptr];
                    add_risk_sums(
                        &self.covar,
                        person,
                        risk[person],
                        &mut unentered_denom,
                        &mut unentered_a,
                        &mut unentered_cmat,
                    );
                    start_ptr += 1;
                }

                let mut time_start = time_end;
                while time_start > stratum_start && self.time[time_start - 1] == event_time {
                    time_start -= 1;
                }

                let mut ndead = 0usize;
                let mut deadwt = 0.0;
                let mut denom2 = 0.0;
                death_a.fill(0.0);
                death_cmat.fill(0.0);
                for person in time_start..=time_end {
                    if self.status[person] == 0 {
                        continue;
                    }
                    ndead += 1;
                    deadwt += self.weights[person];
                    loglik += self.weights[person] * zbeta[person];
                    add_risk_sums(
                        &self.covar,
                        person,
                        risk[person],
                        &mut denom2,
                        &mut death_a,
                        &mut death_cmat,
                    );
                    for i in 0..nvar {
                        self.u[i] += self.weights[person] * self.covar[(person, i)];
                    }
                }

                if ndead > 0 {
                    let denom = stop_denom - unentered_denom;
                    for i in 0..nvar {
                        event_a[i] = stop_a[i] - unentered_a[i];
                        for j in 0..=i {
                            event_cmat[(i, j)] = stop_cmat[(i, j)] - unentered_cmat[(i, j)];
                        }
                    }
                    if method == TieMethod::Breslow || ndead == 1 {
                        loglik -= deadwt * denom.ln();
                        for i in 0..nvar {
                            let temp = event_a[i] / denom;
                            self.u[i] -= deadwt * temp;
                            for j in 0..=i {
                                let val = deadwt * (event_cmat[(i, j)] - temp * event_a[j]) / denom;
                                self.imat[(j, i)] += val;
                                if i != j {
                                    self.imat[(i, j)] += val;
                                }
                            }
                        }
                    } else {
                        let death_count = ndead as f64;
                        let risk_fraction = denom2 / death_count;
                        let weight_average = deadwt / death_count;
                        let mut efron_denom = denom - denom2;
                        for i in 0..nvar {
                            event_a[i] -= death_a[i];
                            for j in 0..=i {
                                event_cmat[(i, j)] -= death_cmat[(i, j)];
                            }
                        }
                        for _ in 0..ndead {
                            efron_denom += risk_fraction;
                            loglik -= weight_average * efron_denom.ln();
                            for i in 0..nvar {
                                event_a[i] += death_a[i] / death_count;
                                let temp = event_a[i] / efron_denom;
                                self.u[i] -= weight_average * temp;
                                for j in 0..=i {
                                    event_cmat[(i, j)] += death_cmat[(i, j)] / death_count;
                                    let val = weight_average
                                        * (event_cmat[(i, j)] - temp * event_a[j])
                                        / efron_denom;
                                    self.imat[(j, i)] += val;
                                    if i != j {
                                        self.imat[(i, j)] += val;
                                    }
                                }
                            }
                        }
                    }
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

    /// Evaluates the log likelihood at `beta`, leaving the score in `u` and
    /// the information matrix in `imat`.
    fn iterate(&mut self, beta: &[f64]) -> f64 {
        match (self.method, self.entry_times.is_some()) {
            (TieMethod::Exact, false) => self.iterate_right_censored_exact(beta),
            (TieMethod::Exact, true) => self.iterate_counting_process_exact(beta),
            (_, false) => self.iterate_right_censored(beta),
            (_, true) => self.iterate_counting_process(beta),
        }
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
    /// `agexact`).  Never fails: a singular information matrix zeroes the
    /// redundant directions (`cholesky2`/`chsolve2`) and an exhausted
    /// iteration budget is reported through `flag == 1000`, exactly as R
    /// does.
    pub(crate) fn fit(&mut self) {
        let nvar = self.beta.len();
        let exact = self.method == TieMethod::Exact;
        let counting = self.entry_times.is_some();
        let beta0 = self.beta.clone();
        self.iter = 0;
        self.loglik[0] = self.iterate(&beta0);
        self.loglik[1] = self.loglik[0];
        if nvar == 0 {
            self.flag = 0;
            return;
        }

        let mut a = self.u.clone();
        self.flag = cholesky2(&mut self.imat, self.toler);
        chsolve2(&self.imat, &mut a);
        self.sctest = a.iter().zip(&self.u).map(|(ai, ui)| ai * ui).sum();

        if self.max_iter == 0 || !self.loglik[0].is_finite() {
            self.finish_inverse();
            if exact && counting {
                // agexact.c returns flag 0 when no iterations were requested.
                self.flag = 0;
            }
            return;
        }

        let mut newbeta: Vec<f64> = self.beta.iter().zip(&a).map(|(b, a)| b + a).collect();
        let mut halving = 0usize;
        let mut newlk = self.loglik[1];
        for iter in 1..=self.max_iter {
            self.iter = iter;
            newlk = self.iterate(&newbeta);
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
                return;
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
        // there (coxexact.c omits the Cholesky factorisation before
        // inverting, which is a bug this port does not copy).
        if exact && (counting || self.max_iter <= 1) {
            self.loglik[1] = newlk;
            self.beta.copy_from_slice(&newbeta);
        } else if self.max_iter > 1 {
            let beta = self.beta.clone();
            self.loglik[1] = self.iterate(&beta);
            self.flag = cholesky2(&mut self.imat, self.toler);
        }
        self.finish_inverse();
        self.flag = 1000;
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
            engine.fit();
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

        fit.fit();
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
        fit.fit();
        let results = fit.results();

        let mut evaluation = CoxFitBuilder::new(time, status, covar)
            .max_iter(0)
            .initial_beta(results.coefficients)
            .build()
            .unwrap();
        evaluation.fit();
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
        fit.fit();
        let results = fit.results();
        assert_eq!(results.flag, 1000);

        let mut evaluation = CoxFitBuilder::new(time, status, covar)
            .max_iter(0)
            .initial_beta(results.coefficients)
            .build()
            .unwrap();
        evaluation.fit();
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
        fit.fit();
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
            cox.fit();
            let results = cox.results();
            assert!(results.coefficients[0].is_finite());
            assert!(results.var[(0, 0)].is_finite());
            assert!(results.loglik[1].is_finite());
        }
    }

    #[test]
    fn counting_process_entry_order_is_precomputed_per_stratum_with_index_ties() {
        let fit = counting_process_order_fixture();
        assert_eq!(fit.order, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(
            fit.entry_order.as_deref(),
            Some([2, 1, 0, 5, 3, 4].as_slice())
        );
    }

    #[test]
    fn counting_process_evaluation_is_deterministic() {
        let mut fit = counting_process_order_fixture();
        let beta = [0.2, -0.15];
        let first_loglik = fit.iterate(&beta);
        let first_score = fit.u.clone();
        let first_information = fit.imat.clone();
        let second_loglik = fit.iterate(&beta);
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
        fit.fit();
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
        fit.fit();
        let results = fit.results();
        assert!(results.coefficients.is_empty());
        assert!((results.loglik[0] - (-(3.0_f64.ln()) - 2.0_f64.ln())).abs() < 1e-12);
        assert_eq!(results.iter, 0);
    }
}
