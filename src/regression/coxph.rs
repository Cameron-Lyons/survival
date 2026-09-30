//! The Cox proportional hazards model: R survival's `coxph` object.
//!
//! [`CoxPHFit::fit`] is the driver behind R's `coxph()`: it dispatches to the
//! Newton-Raphson engine (`cox_optimizer`, ports of `coxfit6.c`, `agfit4.c`,
//! `coxexact.c` and `agexact.c`), then performs the post-processing of
//! `R/coxph.fit.R`, `R/agreg.fit.R`, `R/coxexact.fit.R`, `R/agexact.fit.R`
//! and `R/coxph.R`: centred linear predictors, martingale residuals, the
//! robust (cluster sandwich) variance, the Wald test and the concordance
//! of the linear predictors.
//!
//! The fitted object keeps the data it was fitted to, so the methods R
//! reconstructs from the model frame are plain method calls here:
//! `basehaz()`, `survfit()` (`R/survfit.coxph.R` on top of
//! `surv_analysis::agsurv`), `predict()` (`R/predict.coxph.R`) and the
//! residual types of `R/residuals.coxph.R` (`coxph_diagnostics`).  The
//! per-stratum baseline curves are computed once and cached.

use crate::concordance::{ConcordanceCounts, ConcordanceFit, ConcordanceOptions, concordancefit};
use crate::constants::{COX_CONVERGENCE_TOLERANCE, COX_MAX_ITER, COX_RANK_TOLERANCE};
use crate::core::SurvResponse;
use crate::core::strata_order::{order_within_strata, validate_intervals};
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::matrix::matrix_rows;
use crate::internal::numpy_utils::{FloatMatrix, FloatVec, IntVec};
use crate::internal::step::{find_interval, step_at};
use crate::internal::typed_inputs::{CountingProcessData, SurvivalData};
use crate::internal::validation::{validate_binary_i32, validate_finite, validate_length};
use crate::regression::cox_optimizer::{CoxFitBuilder, CoxFitResults, TieMethod};
use crate::regression::coxph_diagnostics::{
    SchoenfeldResiduals, collapse_rows, martingale_residuals_at, score_residuals_at,
};
use crate::regression::coxph_wtest::{wald_statistic, wald_tests};
use crate::surv_analysis::agsurv::{
    AgsurvCurve, AgsurvData, CoxSurvType, IndividualInterval, IntegratedCurve, agsurv_rows,
    cum_xbar_at, cumhaz_at, expand_curve, individual_curve, integrate_curve,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::OnceLock;

/// Validated inputs of a Cox fit, in the caller's row order (`Surv(time,
/// status) ~ x` or `Surv(entry, time, status) ~ x`); the fit checks the
/// predictor values and weights ([`CoxphData::check_fit_input`]).
#[derive(Debug, Clone)]
pub struct CoxphData {
    pub time: Vec<f64>,
    pub entry: Option<Vec<f64>>,
    pub status: Vec<i32>,
    /// `n x nvar` design matrix (no intercept column).
    pub x: Array2<f64>,
    pub weights: Option<Vec<f64>>,
    pub strata: Option<Vec<i32>>,
    pub offset: Option<Vec<f64>>,
}

impl CoxphData {
    pub fn try_new(
        time: Vec<f64>,
        entry: Option<Vec<f64>>,
        status: Vec<i32>,
        x: Array2<f64>,
        weights: Option<Vec<f64>>,
        strata: Option<Vec<i32>>,
        offset: Option<Vec<f64>>,
    ) -> SurvivalResult<Self> {
        let n = time.len();
        if n == 0 {
            return Err(SurvivalError::invalid_input(
                "No (non-missing) observations",
            ));
        }
        validate_finite(&time, "time")?;
        validate_length(n, status.len(), "status")?;
        validate_binary_i32(&status, "status")?;
        validate_length(n, x.nrows(), "x")?;
        if let Some(entry) = &entry {
            validate_length(n, entry.len(), "entry")?;
            validate_finite(entry, "entry")?;
            validate_intervals(entry, &time)?;
        }
        if let Some(weights) = &weights {
            validate_length(n, weights.len(), "weights")?;
            validate_finite(weights, "weights")?;
        }
        if let Some(strata) = &strata {
            validate_length(n, strata.len(), "strata")?;
        }
        if let Some(offset) = &offset {
            validate_length(n, offset.len(), "offset")?;
            validate_finite(offset, "offset")?;
        }
        Ok(Self {
            time,
            entry,
            status,
            x,
            weights,
            strata,
            offset,
        })
    }

    pub fn n(&self) -> usize {
        self.time.len()
    }

    /// The checks `coxph()` makes only once the data have events (data
    /// without events get their fit first): finite predictors, and the
    /// fitters' positive case weights (`coxph.fit`, `agreg.fit`,
    /// `coxpenal.fit`).
    pub fn check_fit_input(&self) -> SurvivalResult<()> {
        if let Some(value) = self.x.iter().find(|value| !value.is_finite()) {
            return Err(SurvivalError::invalid_input(format!(
                "x contains non-finite value {value}"
            )));
        }
        if self
            .weights
            .as_ref()
            .is_some_and(|weights| weights.iter().any(|&w| w <= 0.0))
        {
            return Err(SurvivalError::invalid_input("Invalid weights, must be >0"));
        }
        Ok(())
    }
}

/// Fitting options: the `coxph()` arguments beyond the data.
#[derive(Debug, Clone)]
pub struct CoxphOptions {
    pub method: TieMethod,
    /// Initial coefficients (`init`); zero when absent.
    pub init: Option<Vec<f64>>,
    /// `coxph.control(iter.max, eps, toler.chol)`.
    pub iter_max: usize,
    pub eps: f64,
    pub toler_chol: f64,
    /// R's `nocenter`: a column whose values all belong to this set is
    /// neither centred nor scaled inside the fitter (its mean is reported
    /// as 0).  `None` is R's `nocenter = NULL`: centre every column.
    pub nocenter: Option<Vec<f64>>,
    /// Cluster codes for the robust sandwich variance (`cluster()`).
    pub cluster: Option<Vec<i32>>,
    /// Robust variance; defaults to `cluster.is_some()`.  `Some(true)`
    /// without a cluster clusters on the observations.
    pub robust: Option<bool>,
}

impl Default for CoxphOptions {
    fn default() -> Self {
        Self {
            method: TieMethod::Efron,
            init: None,
            iter_max: COX_MAX_ITER,
            eps: COX_CONVERGENCE_TOLERANCE,
            toler_chol: COX_RANK_TOLERANCE,
            nocenter: Some(vec![-1.0, 0.0, 1.0]),
            cluster: None,
            robust: None,
        }
    }
}

/// Rows grouped by stratum and sorted by (stratum, time, original index),
/// the order every C kernel of the package expects.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct SortedRows {
    /// Original row indices in sorted order.
    pub order: Vec<usize>,
    /// `[start, end)` positions in `order` of each stratum.
    pub bounds: Vec<(usize, usize)>,
    /// Stratum code of each stratum, ascending.
    pub codes: Vec<i32>,
    /// Stratum position (index into `codes`) of each original row.
    pub stratum_index: Vec<usize>,
}

impl SortedRows {
    fn new(order: Vec<usize>, strata: Option<&[i32]>) -> Self {
        let n = order.len();
        let mut bounds = Vec::new();
        let mut codes = Vec::new();
        let mut stratum_index = vec![0; n];
        let mut start = 0;
        for position in 0..n {
            let code = strata.map_or(0, |s| s[order[position]]);
            let last = position + 1 == n || strata.is_some_and(|s| s[order[position + 1]] != code);
            if last {
                bounds.push((start, position + 1));
                codes.push(code);
                for &row in &order[start..=position] {
                    stratum_index[row] = codes.len() - 1;
                }
                start = position + 1;
            }
        }
        Self {
            order,
            bounds,
            codes,
            stratum_index,
        }
    }

    /// The rows in (stratum, time, original index) order, the order the
    /// engine sorts them in.
    pub(crate) fn by_time(time: &[f64], strata: Option<&[i32]>) -> Self {
        let order = match strata {
            Some(strata) => order_within_strata(strata, |a, b| time[a].total_cmp(&time[b])),
            None => {
                let mut order: Vec<usize> = (0..time.len()).collect();
                order.sort_by(|&a, &b| time[a].total_cmp(&time[b]));
                order
            }
        };
        Self::new(order, strata)
    }

    pub(crate) fn nstrata(&self) -> usize {
        self.codes.len()
    }

    /// Position of a stratum code, if the fit has it.
    pub(crate) fn position_of(&self, code: i32) -> Option<usize> {
        self.codes.binary_search(&code).ok()
    }
}

/// A fitted Cox model (R's `coxph` object).  As in R, the coefficient of a
/// redundant (aliased) covariate is `NaN` (R's `NA`) with a zero row and
/// column in `var`; every computation on the fit treats it as 0.
#[pyclass(module = "survival._survival", skip_from_py_object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoxPHFit {
    /// Coefficients; `NaN` marks an aliased covariate.
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    /// Variance of the coefficients; the robust sandwich when `naive_var`
    /// is present.
    pub var: Array2<f64>,
    /// Model-based variance, kept when `var` is the robust one.
    pub naive_var: Option<Array2<f64>>,
    /// Log partial likelihood at the initial and the final coefficients.
    #[pyo3(get)]
    pub loglik: [f64; 2],
    /// Score test at the initial coefficients.
    #[pyo3(get)]
    pub score: f64,
    /// Robust score test (present with the robust variance).
    #[pyo3(get)]
    pub rscore: Option<f64>,
    #[pyo3(get)]
    pub wald_test: f64,
    #[pyo3(get)]
    pub iter: usize,
    /// Rank of the information matrix (`< nvar` marks aliased columns),
    /// `-2` converged while step halving, `1000` did not converge.
    #[pyo3(get)]
    pub flag: i32,
    /// `agreg.fit`'s `info` for a (start, stop] Breslow or Efron fit: the
    /// rank of the information matrix at the initial coefficients, the
    /// number of recentrings of the risk scores, the number of step
    /// halvings, and 1 when the iterations ran out.  `None` for the other
    /// fitters.
    #[pyo3(get)]
    pub info: Option<[i32; 4]>,
    /// `x %*% coef + offset - sum(coef * means)`.
    #[pyo3(get)]
    pub linear_predictors: Vec<f64>,
    /// Martingale residuals.
    #[pyo3(get)]
    pub residuals: Vec<f64>,
    /// Column centres (weighted means for right-censored data, plain means
    /// for counting-process data; 0 for `nocenter` columns).
    #[pyo3(get)]
    pub means: Vec<f64>,
    /// Score vector at the final coefficients (`agreg.fit`'s `first`).
    #[pyo3(get)]
    pub first: Vec<f64>,
    #[pyo3(get)]
    pub n: usize,
    #[pyo3(get)]
    pub nevent: usize,
    #[pyo3(get)]
    pub method: TieMethod,
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub entry: Option<Vec<f64>>,
    #[pyo3(get)]
    pub status: Vec<i32>,
    /// Design matrix in the caller's row order.
    pub x: Array2<f64>,
    #[pyo3(get)]
    pub weights: Vec<f64>,
    #[pyo3(get)]
    pub strata: Option<Vec<i32>>,
    #[pyo3(get)]
    pub offset: Vec<f64>,
    /// Columns that were neither centred nor scaled (`nocenter`).
    #[pyo3(get)]
    pub nocenter: Vec<bool>,
    #[pyo3(get)]
    pub cluster: Option<Vec<i32>>,
    /// The concordance of the linear predictors with the outcome, as
    /// `coxph()` computes it: `concordancefit(Y, lp, strata, weights,
    /// cluster, reverse = TRUE, timefix = FALSE)`.  R's `fit$concordance`
    /// vector is `(colSums(count), concordance, sqrt(var))` of this object,
    /// and `summary(fit)$concordance` its last two entries.
    #[pyo3(get)]
    pub concordance: ConcordanceFit,
    pub(crate) sorted: SortedRows,
    /// Per-stratum baseline curves at `x - means`, `risk = exp(lp)`, for the
    /// hazard type matching the tie method.
    #[serde(skip)]
    curves: OnceLock<Vec<AgsurvCurve>>,
}

/// `predict.coxph`'s `reference` argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictReference {
    Strata,
    Sample,
    Zero,
}

impl PredictReference {
    pub fn parse(name: &str) -> SurvivalResult<Self> {
        match name {
            "strata" => Ok(Self::Strata),
            "sample" => Ok(Self::Sample),
            "zero" => Ok(Self::Zero),
            other => Err(SurvivalError::invalid_input(format!(
                "reference must be 'strata', 'sample' or 'zero', got '{other}'"
            ))),
        }
    }
}

/// New observations for `predict()` / `survfit()`: covariate rows plus the
/// optional stratum, offset and (for expected counts) follow-up.
#[derive(Debug, Clone)]
pub struct CoxNewData {
    pub x: Array2<f64>,
    pub strata: Option<Vec<i32>>,
    pub offset: Option<Vec<f64>>,
    pub time: Option<Vec<f64>>,
    pub entry: Option<Vec<f64>>,
}

impl CoxNewData {
    pub fn try_new(
        x: Array2<f64>,
        strata: Option<Vec<i32>>,
        offset: Option<Vec<f64>>,
        time: Option<Vec<f64>>,
        entry: Option<Vec<f64>>,
    ) -> SurvivalResult<Self> {
        Self::validated(x, strata, offset, time, entry, false)
    }

    /// Prediction rows may contain NaN covariates or offsets. Missing values
    /// propagate only into predictions that use them; times remain finite.
    pub fn try_new_prediction(
        x: Array2<f64>,
        strata: Option<Vec<i32>>,
        offset: Option<Vec<f64>>,
        time: Option<Vec<f64>>,
        entry: Option<Vec<f64>>,
    ) -> SurvivalResult<Self> {
        Self::validated(x, strata, offset, time, entry, true)
    }

    fn validated(
        x: Array2<f64>,
        strata: Option<Vec<i32>>,
        offset: Option<Vec<f64>>,
        time: Option<Vec<f64>>,
        entry: Option<Vec<f64>>,
        allow_missing: bool,
    ) -> SurvivalResult<Self> {
        let m = x.nrows();
        if m == 0 {
            return Err(SurvivalError::invalid_input(
                "newdata must have at least one row",
            ));
        }
        if let Some(value) = x
            .iter()
            .find(|value| value.is_infinite() || (!allow_missing && value.is_nan()))
        {
            return Err(SurvivalError::invalid_input(format!(
                "newdata contains non-finite value {value}"
            )));
        }
        if let Some(strata) = &strata {
            validate_length(m, strata.len(), "newdata strata")?;
        }
        if let Some(offset) = &offset {
            validate_length(m, offset.len(), "newdata offset")?;
            if let Some(value) = offset
                .iter()
                .find(|value| value.is_infinite() || (!allow_missing && value.is_nan()))
            {
                return Err(SurvivalError::invalid_input(format!(
                    "newdata offset contains non-finite value {value}"
                )));
            }
        }
        if let Some(time) = &time {
            validate_length(m, time.len(), "newdata time")?;
            validate_finite(time, "newdata time")?;
        }
        if let Some(entry) = &entry {
            validate_length(m, entry.len(), "newdata entry")?;
            validate_finite(entry, "newdata entry")?;
        }
        Ok(Self {
            x,
            strata,
            offset,
            time,
            entry,
        })
    }

    fn nrows(&self) -> usize {
        self.x.nrows()
    }
}

/// `predict(type = "lp" | "risk" | "expected" | "survival")` output.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct CoxPrediction {
    #[pyo3(get)]
    pub fit: Vec<f64>,
    #[pyo3(get)]
    pub se_fit: Option<Vec<f64>>,
}

/// `predict(type = "terms")` output: one column per term.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct CoxTermsPrediction {
    #[pyo3(get)]
    pub fit: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub se_fit: Option<Vec<Vec<f64>>>,
    /// `sum(coefficients * means)`, R's `attr(pred, "constant")`.
    #[pyo3(get)]
    pub constant: f64,
}

/// `basehaz(fit)`: the cumulative hazard at each event or censoring time
/// per stratum, rows concatenated in stratum order.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct Basehaz {
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub hazard: Vec<f64>,
    /// Stratum code of each row (absent for an unstratified fit).
    #[pyo3(get)]
    pub strata: Option<Vec<i32>>,
}

/// One curve of `survfit(fit, newdata)`: `surv`, `cumhaz` and `std_err`
/// have one row per time and one column per new observation.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct CoxSurvfitCurve {
    /// Stratum code the curve belongs to.
    #[pyo3(get)]
    pub stratum: i32,
    /// Number of observations in the stratum.
    #[pyo3(get)]
    pub n: usize,
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub n_risk: Vec<f64>,
    #[pyo3(get)]
    pub n_event: Vec<f64>,
    #[pyo3(get)]
    pub n_censor: Vec<f64>,
    #[pyo3(get)]
    pub surv: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub cumhaz: Vec<Vec<f64>>,
    /// Standard error of the cumulative hazard (`std.err` with `logse`).
    #[pyo3(get)]
    pub std_err: Option<Vec<Vec<f64>>>,
}

/// `survfit.coxph` arguments.
#[derive(Debug, Clone, Copy)]
pub struct SurvfitOptions {
    /// 1 = product limit (Kalbfleisch-Prentice), 2 = `exp(-cumhaz)`.
    pub stype: u8,
    /// 1 = Breslow, 2 = Efron; defaults to the fit's tie method.
    pub ctype: Option<u8>,
    pub se_fit: bool,
    /// `censor = FALSE` drops the rows without an event.
    pub censor: bool,
    /// `start.time`: the curves are built from the rows whose stop time is
    /// at or after it; the risk scores stay those of the whole fit.
    pub start_time: Option<f64>,
}

impl Default for SurvfitOptions {
    fn default() -> Self {
        Self {
            stype: 2,
            ctype: None,
            se_fit: true,
            censor: true,
            start_time: None,
        }
    }
}

fn crossprod(rows: &Array2<f64>) -> Array2<f64> {
    rows.t().dot(rows)
}

/// `coxph()`'s offset centring: the offsets minus their (unweighted) mean,
/// which keeps `exp()` of the risk scores in range, and that mean.  An
/// absent or all-zero offset is zero with mean 0.
pub(crate) fn centre_offset(offset: Option<&[f64]>, n: usize) -> (Vec<f64>, f64) {
    match offset {
        Some(offset) if offset.iter().any(|&value| value != 0.0) => {
            let mean = offset.iter().sum::<f64>() / n as f64;
            (offset.iter().map(|value| value - mean).collect(), mean)
        }
        _ => (vec![0.0; n], 0.0),
    }
}

/// R's `nocenter` rule: a column whose values all belong to `values` is
/// neither centred nor scaled by the fitter.
pub(crate) fn nocenter_columns(x: &Array2<f64>, values: Option<&[f64]>) -> Vec<bool> {
    (0..x.ncols())
        .map(|col| {
            values.is_some_and(|values| x.column(col).iter().all(|value| values.contains(value)))
        })
        .collect()
}

/// `coxph()` adds the mean offset back to the linear predictors the fitter
/// computed at the centred offset.
pub(crate) fn add_offset_mean(linear_predictors: &mut [f64], offset_mean: f64) {
    if offset_mean != 0.0 {
        for value in linear_predictors {
            *value += offset_mean;
        }
    }
}

/// A Cox model whose parameters another fitter estimated (`coxpenal.fit`),
/// in the data's row order: what [`CoxPHFit::from_fitted`] assembles.
pub(crate) struct FittedCox {
    pub method: TieMethod,
    pub coefficients: Vec<f64>,
    pub var: Array2<f64>,
    pub loglik: [f64; 2],
    pub iter: usize,
    pub flag: i32,
    pub means: Vec<f64>,
    pub nocenter: Vec<bool>,
    /// Score vector at the final coefficients.
    pub first: Vec<f64>,
    /// The linear predictors at the centred offset ([`centre_offset`]).
    pub linear_predictors: Vec<f64>,
    /// The offsets' mean, added back to the stored linear predictors.
    pub offset_mean: f64,
    /// Martingale residuals; `None` computes them from the linear predictors.
    pub residuals: Option<Vec<f64>>,
    pub wald_test: f64,
}

/// Shared optimizer setup; callers choose whether offsets have already been centered.
pub(crate) fn fit_cox_engine(
    data: &CoxphData,
    options: &CoxphOptions,
    offset: Option<&[f64]>,
    iterate_empty: bool,
) -> SurvivalResult<(CoxFitResults, Vec<bool>)> {
    let nvar = data.x.ncols();
    data.check_fit_input()?;
    if let Some(init) = &options.init {
        if init.len() != nvar {
            return Err(SurvivalError::invalid_input(
                "Wrong length for inital values",
            ));
        }
        validate_finite(init, "init")?;
    }
    let nocenter = nocenter_columns(&data.x, options.nocenter.as_deref());
    let doscale = nocenter.iter().map(|&skip| !skip).collect();

    let mut engine = CoxFitBuilder::new(
        Array1::from_vec(data.time.clone()),
        Array1::from_vec(data.status.clone()),
        data.x.clone(),
    )
    .method(options.method)
    .max_iter(options.iter_max)
    .iterate_empty(iterate_empty)
    .eps(options.eps)
    .toler(options.toler_chol)
    .doscale(doscale)
    .initial_beta(options.init.clone().unwrap_or_else(|| vec![0.0; nvar]));
    if let Some(entry) = &data.entry {
        engine = engine.entry_times(Array1::from_vec(entry.clone()));
    }
    if let Some(strata) = &data.strata {
        engine = engine.strata(Array1::from_vec(strata.clone()));
    }
    if let Some(offset) = offset {
        engine = engine.offset(Array1::from_vec(offset.to_vec()));
    }
    if let Some(weights) = &data.weights {
        engine = engine.weights(Array1::from_vec(weights.clone()));
    }
    let mut engine = engine.build()?;
    engine.fit()?;
    Ok((engine.results(), nocenter))
}

impl CoxPHFit {
    /// Fits the model: `coxph.fit` / `agreg.fit` / `coxexact.fit` /
    /// `agexact.fit` at the centred offset, followed by the post-processing
    /// of `coxph()`.
    pub fn fit(data: CoxphData, options: CoxphOptions) -> SurvivalResult<Self> {
        let n = data.n();
        let nvar = data.x.ncols();
        let nevent = data.status.iter().filter(|&&s| s == 1).count();
        let (centred_offset, offset_mean) = centre_offset(data.offset.as_deref(), n);
        if nevent == 0 {
            return Ok(Self::without_events(data, centred_offset, &options));
        }
        let (results, nocenter) = fit_cox_engine(&data, &options, Some(&centred_offset), false)?;

        let offset = data.offset.unwrap_or_else(|| vec![0.0; n]);
        let weights = data.weights.unwrap_or_else(|| vec![1.0; n]);
        // The linear predictor uses the fitted values; only afterwards are
        // the aliased coefficients marked NA (`coef[which.sing] <- NA`).
        // Residuals, the robust variance and the concordance use it at the
        // centred offset; the mean offset is added back at the end.
        let mut coefficients = results.coefficients;
        let center: f64 = coefficients
            .iter()
            .zip(&results.means)
            .map(|(b, m)| b * m)
            .sum();
        let linear_predictors: Vec<f64> = (0..n)
            .map(|i| {
                data.x
                    .row(i)
                    .iter()
                    .zip(&coefficients)
                    .map(|(x, b)| x * b)
                    .sum::<f64>()
                    + centred_offset[i]
                    - center
            })
            .collect();
        // `which.sing <- diag(var) == 0` when the rank is below nvar
        // (`coxph.fit`'s flag, negative ranks and -2 included; `agreg.fit`'s
        // rank at the initial coefficients).  `coxph.fit` and `agreg.fit`
        // leave the coefficients alone when no iterations were requested,
        // the exact fitters do not.
        let rank = results.info.map_or(results.flag, |info| info[0]);
        let exact = options.method == TieMethod::Exact;
        if (exact || options.iter_max > 0) && i64::from(rank) < nvar as i64 {
            for (j, coefficient) in coefficients.iter_mut().enumerate() {
                if results.var[(j, j)] == 0.0 {
                    *coefficient = f64::NAN;
                }
            }
        }
        let sorted = SortedRows::new(results.order, data.strata.as_deref());
        // As in `coxph()`, the cluster enters the concordance whenever it was
        // given, even when the variance is not robust.
        let concordance = fit_concordance(
            &data.time,
            data.entry.as_deref(),
            &data.status,
            &linear_predictors,
            &weights,
            data.strata.as_deref(),
            options.cluster.as_deref(),
        )?;

        let mut fit = Self {
            coefficients,
            var: results.var,
            naive_var: None,
            loglik: results.loglik,
            score: results.sctest,
            rscore: None,
            wald_test: 0.0,
            iter: results.iter,
            flag: results.flag,
            info: results.info,
            linear_predictors,
            residuals: Vec::new(),
            means: results.means,
            first: results.score,
            n,
            nevent,
            method: options.method,
            time: data.time,
            entry: data.entry,
            status: data.status,
            x: data.x,
            weights,
            strata: data.strata,
            offset,
            nocenter,
            cluster: options.cluster,
            concordance,
            sorted,
            curves: OnceLock::new(),
        };
        fit.residuals = martingale_residuals_at(&fit, &fit.linear_predictors);

        let robust = options.robust.unwrap_or(fit.cluster.is_some());
        if !robust {
            // "cluster specified with robust=FALSE, cluster ignored"
            fit.cluster = None;
        }
        if robust && fit.coefficients.iter().any(|b| !b.is_nan()) {
            if fit.method == TieMethod::Exact {
                return Err(SurvivalError::invalid_input(
                    "dfbeta residuals are not available for the exact method",
                ));
            }
            let cluster = fit
                .cluster
                .clone()
                .unwrap_or_else(|| (0..n as i32).collect());
            fit.naive_var = Some(fit.var.clone());
            let dfbeta = fit.dfbeta_matrix(&fit.linear_predictors, true, Some(&cluster))?;
            fit.var = crossprod(&dfbeta);
            // Robust score test at the initial coefficients (lp = X init).
            let lp0: Vec<f64> = match &options.init {
                Some(init) => (0..n)
                    .map(|i| fit.x.row(i).iter().zip(init).map(|(x, b)| x * b).sum())
                    .collect(),
                None => vec![0.0; n],
            };
            let scores = score_residuals_at(&fit, &lp0)?;
            let scores = collapse_rows(&scores, Some(&fit.weights), Some(&cluster));
            let u: Vec<f64> = (0..nvar).map(|j| scores.column(j).sum()).collect();
            let u_matrix = Array2::from_shape_vec((nvar, 1), u)
                .map_err(|err| SurvivalError::computation(err.to_string()))?;
            fit.rscore =
                Some(wald_tests(&crossprod(&scores), &u_matrix, options.toler_chol)?.test[0]);
        }

        let shift: Vec<f64> = fit
            .coefficients_or_zero()
            .iter()
            .enumerate()
            .map(|(i, b)| b - options.init.as_ref().map_or(0.0, |init| init[i]))
            .collect();
        fit.wald_test = wald_statistic(&fit.var, &shift, options.toler_chol)?;
        add_offset_mean(&mut fit.linear_predictors, offset_mean);
        Ok(fit)
    }

    /// `coxph()`'s fit of data without events, made before any fitter runs:
    /// `NA` coefficients, a zero variance, `loglik = c(0, 0)`, zero
    /// residuals, the unweighted column means and a concordance without
    /// pairs.  As in R the linear predictors are the centred offset.
    fn without_events(data: CoxphData, centred_offset: Vec<f64>, options: &CoxphOptions) -> Self {
        let n = data.n();
        let nvar = data.x.ncols();
        let means = (0..nvar)
            .map(|col| data.x.column(col).sum() / n as f64)
            .collect();
        let concordance = ConcordanceFit {
            concordance: vec![f64::NAN],
            n,
            count: vec![ConcordanceCounts {
                concordant: 0.0,
                discordant: 0.0,
                tied_x: 0.0,
                tied_y: 0.0,
                tied_xy: 0.0,
            }],
            count_strata: None,
            var: None,
            cvar: None,
            dfbeta: None,
            influence: None,
            ranks: None,
        };
        Self {
            coefficients: vec![f64::NAN; nvar],
            var: Array2::zeros((nvar, nvar)),
            naive_var: None,
            loglik: [0.0; 2],
            score: 0.0,
            rscore: None,
            wald_test: 0.0,
            iter: 0,
            flag: 0,
            info: None,
            linear_predictors: centred_offset,
            residuals: vec![0.0; n],
            means,
            first: vec![0.0; nvar],
            n,
            nevent: 0,
            method: options.method,
            nocenter: nocenter_columns(&data.x, options.nocenter.as_deref()),
            sorted: SortedRows::by_time(&data.time, data.strata.as_deref()),
            time: data.time,
            entry: data.entry,
            status: data.status,
            x: data.x,
            weights: data.weights.unwrap_or_else(|| vec![1.0; n]),
            strata: data.strata,
            offset: data.offset.unwrap_or_else(|| vec![0.0; n]),
            cluster: None,
            concordance,
            curves: OnceLock::new(),
        }
    }

    /// A fit whose parameters another fitter estimated (`coxpenal.fit`),
    /// with `coxph()`'s post-processing: the martingale residuals when not
    /// given and the concordance of the final linear predictors.  `cluster`
    /// only enters the concordance: `coxph()` passes it to `concordancefit`
    /// although a penalized fit has no robust variance.  There is no score
    /// test.
    pub(crate) fn from_fitted(
        data: CoxphData,
        fitted: FittedCox,
        cluster: Option<&[i32]>,
    ) -> SurvivalResult<Self> {
        let n = data.n();
        let weights = data.weights.unwrap_or_else(|| vec![1.0; n]);
        let concordance = fit_concordance(
            &data.time,
            data.entry.as_deref(),
            &data.status,
            &fitted.linear_predictors,
            &weights,
            data.strata.as_deref(),
            cluster,
        )?;
        let mut fit = Self {
            coefficients: fitted.coefficients,
            var: fitted.var,
            naive_var: None,
            loglik: fitted.loglik,
            score: f64::NAN,
            rscore: None,
            wald_test: fitted.wald_test,
            iter: fitted.iter,
            flag: fitted.flag,
            info: None,
            linear_predictors: fitted.linear_predictors,
            residuals: Vec::new(),
            means: fitted.means,
            first: fitted.first,
            n,
            nevent: data.status.iter().filter(|&&s| s == 1).count(),
            method: fitted.method,
            sorted: SortedRows::by_time(&data.time, data.strata.as_deref()),
            time: data.time,
            entry: data.entry,
            status: data.status,
            x: data.x,
            weights,
            strata: data.strata,
            offset: data.offset.unwrap_or_else(|| vec![0.0; n]),
            nocenter: fitted.nocenter,
            cluster: None,
            concordance,
            curves: OnceLock::new(),
        };
        fit.residuals = match fitted.residuals {
            Some(residuals) => residuals,
            None => martingale_residuals_at(&fit, &fit.linear_predictors),
        };
        add_offset_mean(&mut fit.linear_predictors, fitted.offset_mean);
        Ok(fit)
    }

    pub fn nvar(&self) -> usize {
        self.coefficients.len()
    }

    /// Coefficients with aliased (`NaN`) entries replaced by 0, R's
    /// `ifelse(is.na(coef), 0, coef)`.
    pub fn coefficients_or_zero(&self) -> Vec<f64> {
        self.coefficients
            .iter()
            .map(|b| if b.is_nan() { 0.0 } else { *b })
            .collect()
    }

    /// Survival-curve types matching the tie method (`survfit.coxph`:
    /// `ctype` 2 for Efron, 1 otherwise).
    fn default_survtype(&self) -> CoxSurvType {
        if self.method == TieMethod::Efron {
            CoxSurvType::Efron
        } else {
            CoxSurvType::Breslow
        }
    }

    /// Weighted mean of the offsets (`survfit.coxph`'s `offset.mean`).
    fn offset_mean(&self) -> f64 {
        let total: f64 = self.weights.iter().sum();
        self.offset
            .iter()
            .zip(&self.weights)
            .map(|(o, w)| o * w)
            .sum::<f64>()
            / total
    }

    /// Per-stratum `agsurv` pieces at `x - means`, in the fit's stratum
    /// order, with `survfit.coxph`'s `risk = exp(X %*% beta + offset -
    /// xcenter)`: the risks relative to a subject at the means and the mean
    /// offset, so a large offset neither overflows nor rounds the baseline
    /// survival to 1.  `coxsurv.fit` uses `survtype` for the variance too.
    ///
    /// `start_time` keeps only the rows whose stop time is at or after it
    /// (`survfit.coxph`'s `keep <- Y[, ncol(Y) - 1] >= start.time`); a
    /// stratum it empties keeps an empty curve.
    fn compute_curves(
        &self,
        survtype: CoxSurvType,
        start_time: Option<f64>,
    ) -> SurvivalResult<Vec<AgsurvCurve>> {
        if let Some(t0) = start_time
            && !(0..self.n).any(|i| self.status[i] == 1 && self.time[i] >= t0)
        {
            return Err(SurvivalError::invalid_input(
                "start.time argument has removed all endpoints",
            ));
        }
        let offset_mean = self.offset_mean();
        let risk: Vec<f64> = self
            .linear_predictors
            .iter()
            .map(|lp| (lp - offset_mean).exp())
            .collect();
        let data = AgsurvData {
            start: self.entry.as_deref(),
            stop: &self.time,
            status: &self.status,
            x: self.x.view(),
            means: Some(&self.means),
            weights: &self.weights,
            risk: &risk,
        };
        self.sorted
            .bounds
            .iter()
            .map(|&(start, end)| {
                let rows = &self.sorted.order[start..end];
                match start_time {
                    None => agsurv_rows(&data, rows, survtype, survtype),
                    Some(t0) => {
                        let kept: Vec<usize> = rows
                            .iter()
                            .copied()
                            .filter(|&i| self.time[i] >= t0)
                            .collect();
                        agsurv_rows(&data, &kept, survtype, survtype)
                    }
                }
            })
            .collect()
    }

    /// The cached baseline curves for the fit's own hazard type.
    pub(crate) fn baseline_curves(&self) -> SurvivalResult<&[AgsurvCurve]> {
        if let Some(curves) = self.curves.get() {
            return Ok(curves);
        }
        let curves = self.compute_curves(self.default_survtype(), None)?;
        Ok(self.curves.get_or_init(|| curves))
    }

    /// The curves of one hazard type from the rows kept by `start_time`:
    /// the cached ones for the fit's own type and every row.
    fn curves_for(
        &self,
        survtype: CoxSurvType,
        start_time: Option<f64>,
    ) -> SurvivalResult<Cow<'_, [AgsurvCurve]>> {
        if survtype == self.default_survtype() && start_time.is_none() {
            Ok(Cow::Borrowed(self.baseline_curves()?))
        } else {
            Ok(Cow::Owned(self.compute_curves(survtype, start_time)?))
        }
    }

    fn check_newdata(&self, newdata: &CoxNewData) -> SurvivalResult<()> {
        if newdata.x.ncols() != self.nvar() {
            return Err(SurvivalError::invalid_input(format!(
                "newdata has {} columns but the model has {}",
                newdata.x.ncols(),
                self.nvar()
            )));
        }
        if let Some(strata) = &newdata.strata
            && let Some(code) = strata
                .iter()
                .find(|code| self.sorted.position_of(**code).is_none())
        {
            return Err(SurvivalError::invalid_input(format!(
                "New data has a strata not found in the original model: {code}"
            )));
        }
        Ok(())
    }

    /// Centred new covariate rows (`newx - means`) and their relative risks
    /// on the baseline curves' scale, `survfit.coxph`'s `risk2 = exp(x2 %*%
    /// beta + offset2 - xcenter)`.
    fn centered_newdata(&self, newdata: &CoxNewData) -> (Array2<f64>, Vec<f64>) {
        self.centered_rows(newdata.x.view(), newdata.offset.as_deref())
    }

    fn centered_rows(
        &self,
        x: ArrayView2<'_, f64>,
        offset: Option<&[f64]>,
    ) -> (Array2<f64>, Vec<f64>) {
        let mut x2c = x.to_owned();
        for (col, &mean) in self.means.iter().enumerate() {
            x2c.column_mut(col).mapv_inplace(|value| value - mean);
        }
        let coef = self.coefficients_or_zero();
        let offset_mean = self.offset_mean();
        let risk2: Vec<f64> = x2c
            .outer_iter()
            .enumerate()
            .map(|(i, row)| {
                let offset2 = offset.map_or(0.0, |o| o[i]);
                (row.dot(&ArrayView1::from(&coef)) + offset2 - offset_mean).exp()
            })
            .collect();
        (x2c, risk2)
    }

    /// `basehaz(fit, centered)`.
    pub fn basehaz(&self, centered: bool) -> SurvivalResult<Basehaz> {
        let curves = self.baseline_curves()?;
        // the curves are survfit(fit)'s, at x = means and the mean offset;
        // uncentred divides the offset sum(means * coef) back out.
        let scale = if centered {
            1.0
        } else {
            let center: f64 = self
                .means
                .iter()
                .zip(self.coefficients_or_zero())
                .map(|(m, b)| m * b)
                .sum();
            (-center).exp()
        };
        let mut time = Vec::new();
        let mut hazard = Vec::new();
        let mut strata = Vec::new();
        for (position, curve) in curves.iter().enumerate() {
            time.extend_from_slice(&curve.time);
            hazard.extend(curve.cumhaz.iter().map(|h| h * scale));
            strata.extend(std::iter::repeat_n(
                self.sorted.codes[position],
                curve.time.len(),
            ));
        }
        Ok(Basehaz {
            time,
            hazard,
            strata: self.strata.as_ref().map(|_| strata),
        })
    }

    /// `survfit(fit, newdata)`: one curve per stratum (all new rows as
    /// columns), or one curve per new row when `newdata` carries strata.
    /// Without `newdata` the curve is for a covariate row at the means.
    pub fn survfit(
        &self,
        newdata: Option<&CoxNewData>,
        options: SurvfitOptions,
    ) -> SurvivalResult<Vec<CoxSurvfitCurve>> {
        let ctype = options.ctype.unwrap_or(if self.method == TieMethod::Efron {
            2
        } else {
            1
        });
        let survtype = CoxSurvType::from_stype_ctype(options.stype, ctype)?;
        if let Some(newdata) = newdata {
            self.check_newdata(newdata)?;
        }
        let curves = self.curves_for(survtype, options.start_time)?;
        let (x2c, risk2) = match newdata {
            Some(newdata) => self.centered_newdata(newdata),
            // the curve at the means and the mean offset
            None => (Array2::zeros((1, self.nvar())), vec![1.0]),
        };
        let varmat = options.se_fit.then_some(&self.var);
        let mut result = Vec::new();
        let new_strata = newdata.and_then(|newdata| newdata.strata.as_deref());
        if let Some(new_strata) = new_strata {
            for (i, &code) in new_strata.iter().enumerate() {
                let position = self
                    .sorted
                    .position_of(code)
                    .expect("strata were checked against the fit");
                let expanded = expand_curve(
                    &curves[position],
                    survtype,
                    x2c.row(i).insert_axis(ndarray::Axis(0)),
                    &risk2[i..=i],
                    varmat,
                )?;
                result.push(finish_curve(code, expanded, options.censor));
            }
        } else {
            for (position, curve) in curves.iter().enumerate() {
                let expanded = expand_curve(curve, survtype, x2c.view(), &risk2, varmat)?;
                result.push(finish_curve(
                    self.sorted.codes[position],
                    expanded,
                    options.censor,
                ));
            }
        }
        Ok(result)
    }

    /// Survival probabilities at `times`, with one column per observation.
    /// Without `newdata`, predict for the training rows in their original order.
    /// Fits with multiple strata require a stratum for each new observation.
    ///
    /// Uses the same default estimate as [`Self::survfit`] (`stype = 2`,
    /// `ctype` matching the fitted tie method). Times may be unsorted or
    /// repeated; survival is 1 before the first time and stays at the last
    /// value beyond follow-up. Only the requested `times.len() * nrows`
    /// probabilities are allocated, rather than the full curves and hazards.
    pub fn predict_survival_at(
        &self,
        times: &[f64],
        newdata: Option<&CoxNewData>,
    ) -> SurvivalResult<Array2<f64>> {
        validate_finite(times, "times")?;
        let (x, strata, offset) = match newdata {
            Some(newdata) => {
                self.check_newdata(newdata)?;
                if newdata.strata.is_none() && self.sorted.nstrata() > 1 {
                    return Err(SurvivalError::invalid_input(
                        "newdata must carry the strata for survival predictions",
                    ));
                }
                (
                    newdata.x.view(),
                    newdata.strata.as_deref(),
                    newdata.offset.as_deref(),
                )
            }
            None => (
                self.x.view(),
                self.strata.as_deref(),
                Some(self.offset.as_slice()),
            ),
        };
        let mut result = Array2::ones((times.len(), x.nrows()));
        if times.is_empty() {
            return Ok(result);
        }
        let (_, risk) = self.centered_rows(x, offset);
        let curves = self.baseline_curves()?;
        let mut rows_by_stratum = vec![Vec::new(); curves.len()];
        for row in 0..x.nrows() {
            let position = strata.map_or(0, |codes| {
                self.sorted
                    .position_of(codes[row])
                    .expect("strata were checked against the fit")
            });
            rows_by_stratum[position].push(row);
        }
        for (curve, rows) in curves.iter().zip(rows_by_stratum) {
            if rows.is_empty() {
                continue;
            }
            for (i, &at) in times.iter().enumerate() {
                let index = find_interval(&curve.time, at, false);
                if index == 0 {
                    continue;
                }
                // Match expand_curve's exp(-H).powf(risk), including its
                // underflow behavior, rather than reassociating the exponent.
                let baseline = (-curve.cumhaz[index - 1]).exp();
                let mut output = result.row_mut(i);
                for &row in &rows {
                    output[row] = baseline.powf(risk[row]);
                }
            }
        }
        Ok(result)
    }

    /// Cohort expected survival, aggregating baseline hazards directly instead
    /// of materializing every subject's survival and cumulative-hazard curve.
    pub fn expected_survival(
        &self,
        newdata: &CoxNewData,
        group: &[usize],
        weights: &[f64],
        y: Option<&[f64]>,
        times: Option<&[f64]>,
        method: &str,
    ) -> SurvivalResult<crate::population::SurvExpResult> {
        use crate::population::{CoxExpectedBaseline, survexp_cox_prepared};
        self.check_newdata(newdata)?;
        if newdata.strata.is_none() && self.sorted.nstrata() > 1 {
            return Err(SurvivalError::invalid_input(
                "newdata must carry the strata for expected survival",
            ));
        }
        let (_, risk) = self.centered_newdata(newdata);
        let strata: Vec<usize> = (0..newdata.nrows())
            .map(|i| {
                newdata.strata.as_ref().map_or(0, |codes| {
                    self.sorted
                        .position_of(codes[i])
                        .expect("strata were checked")
                })
            })
            .collect();
        let baselines = self
            .baseline_curves()?
            .iter()
            .map(|curve| {
                let rows = || {
                    curve
                        .n_event
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &n)| (n > 0.0).then_some(i))
                };
                CoxExpectedBaseline {
                    time: rows().map(|i| curve.time[i]).collect(),
                    cumhaz: rows().map(|i| curve.cumhaz[i]).collect(),
                }
            })
            .collect::<Vec<_>>();
        survexp_cox_prepared(&baselines, &risk, &strata, group, weights, y, times, method)
    }

    /// `survfit(fit, newdata, id)`: one curve per subject whose covariates
    /// change over the (entry, time] intervals of `newdata`.
    pub fn survfit_individual(
        &self,
        newdata: &CoxNewData,
        id: &[i32],
        options: SurvfitOptions,
    ) -> SurvivalResult<Vec<CoxSurvfitCurve>> {
        self.check_newdata(newdata)?;
        let (Some(entry), Some(time)) = (&newdata.entry, &newdata.time) else {
            return Err(SurvivalError::invalid_input(
                "Individual=TRUE is only valid for counting process data",
            ));
        };
        if id.len() != newdata.nrows() {
            return Err(SurvivalError::invalid_input(
                "id must have one value per newdata row",
            ));
        }
        let ctype = options.ctype.unwrap_or(if self.method == TieMethod::Efron {
            2
        } else {
            1
        });
        let survtype = CoxSurvType::from_stype_ctype(options.stype, ctype)?;
        let curves = self.curves_for(survtype, options.start_time)?;
        let (x2c, risk2) = self.centered_newdata(newdata);
        let varmat = options.se_fit.then_some(&self.var);
        let mut positions = HashMap::new();
        let mut rows: Vec<Vec<usize>> = Vec::new();
        for (i, &subject) in id.iter().enumerate() {
            let position = *positions.entry(subject).or_insert_with(|| {
                rows.push(Vec::new());
                rows.len() - 1
            });
            rows[position].push(i);
        }
        let mut result = Vec::with_capacity(rows.len());
        for subject_rows in rows {
            let intervals: Vec<IndividualInterval<'_>> = subject_rows
                .into_iter()
                .map(|i| IndividualInterval {
                    start: entry[i],
                    stop: time[i],
                    stratum: newdata.strata.as_ref().map_or(0, |s| {
                        self.sorted
                            .position_of(s[i])
                            .expect("strata were checked against the fit")
                    }),
                    x2: x2c.row(i).to_slice().expect("row is contiguous"),
                    risk2: risk2[i],
                })
                .collect();
            let curve = individual_curve(&curves, survtype, &intervals, varmat)?;
            let stratum = intervals
                .first()
                .map_or(0, |interval| self.sorted.codes[interval.stratum]);
            result.push(finish_curve(stratum, curve, options.censor));
        }
        Ok(result)
    }

    /// Weighted per-stratum column means (`predict.coxph`'s `xmeans`).
    fn stratum_means(&self) -> Vec<Vec<f64>> {
        let nvar = self.nvar();
        let mut sums = vec![vec![0.0; nvar]; self.sorted.nstrata()];
        let mut totals = vec![0.0; self.sorted.nstrata()];
        for row in 0..self.n {
            let s = self.sorted.stratum_index[row];
            totals[s] += self.weights[row];
            for (col, sum) in sums[s].iter_mut().enumerate().take(nvar) {
                *sum += self.weights[row] * self.x[(row, col)];
            }
        }
        for (sum, total) in sums.iter_mut().zip(&totals) {
            for value in sum.iter_mut() {
                *value /= total;
            }
        }
        sums
    }

    /// Whether `predict.coxph` reads back the training design and offset
    /// (its `use.x` branch): for `se.fit`, for a stratified fit centred
    /// within strata, or for `reference = "zero"` with non-zero means.
    fn uses_training_x(&self, se_fit: bool, reference: PredictReference) -> bool {
        se_fit
            || (self.strata.is_some() && reference == PredictReference::Strata)
            || (reference == PredictReference::Zero && self.means.iter().any(|&m| m != 0.0))
    }

    /// The design rows `predict.coxph` uses for `lp`, `risk` and `terms`:
    /// centred per the reference, plus the offset.  With `training_offset`
    /// the offset is centred at the mean training offset, as R's `offset -
    /// mean(offset)` does in its `use.x` branch; otherwise the training
    /// offset is 0 and a new offset is used as it is.
    fn prediction_rows(
        &self,
        newdata: Option<&CoxNewData>,
        reference: PredictReference,
        training_offset: bool,
    ) -> SurvivalResult<(Array2<f64>, Vec<f64>)> {
        let offset_mean = if training_offset {
            self.offset.iter().sum::<f64>() / self.n as f64
        } else {
            0.0
        };
        let has_strata = self.strata.is_some();
        let (mut newx, offset, stratum_index): (Array2<f64>, Vec<f64>, Vec<usize>) = match newdata {
            None => (
                self.x.clone(),
                self.offset.iter().map(|o| o - offset_mean).collect(),
                self.sorted.stratum_index.clone(),
            ),
            Some(newdata) => {
                self.check_newdata(newdata)?;
                let m = newdata.nrows();
                let offset = newdata.offset.as_ref().map_or_else(
                    || vec![-offset_mean; m],
                    |o| o.iter().map(|v| v - offset_mean).collect(),
                );
                let stratum_index = match &newdata.strata {
                    Some(strata) => strata
                        .iter()
                        .map(|&code| self.sorted.position_of(code).expect("checked"))
                        .collect(),
                    None => {
                        if has_strata && reference == PredictReference::Strata {
                            return Err(SurvivalError::invalid_input(
                                "newdata must carry the strata for reference = 'strata'",
                            ));
                        }
                        vec![0; m]
                    }
                };
                (newdata.x.clone(), offset, stratum_index)
            }
        };
        if has_strata && reference == PredictReference::Strata {
            let xmeans = self.stratum_means();
            for (i, &s) in stratum_index.iter().enumerate() {
                for col in 0..self.nvar() {
                    newx[(i, col)] -= xmeans[s][col];
                }
            }
        } else if reference != PredictReference::Zero {
            for (col, &mean) in self.means.iter().enumerate() {
                newx.column_mut(col).mapv_inplace(|value| value - mean);
            }
        }
        Ok((newx, offset))
    }

    /// `predict(type = "lp")` (and `"risk"` via [`Self::predict_risk`]).
    pub fn predict_lp(
        &self,
        newdata: Option<&CoxNewData>,
        se_fit: bool,
        reference: PredictReference,
    ) -> SurvivalResult<CoxPrediction> {
        let training_x = self.uses_training_x(se_fit, reference);
        if newdata.is_none() && !training_x {
            return Ok(CoxPrediction {
                fit: self.linear_predictors.clone(),
                se_fit: None,
            });
        }
        let (newx, offset) = self.prediction_rows(newdata, reference, training_x)?;
        let coef = self.coefficients_or_zero();
        let fit: Vec<f64> = newx
            .outer_iter()
            .zip(&offset)
            .map(|(row, o)| row.iter().zip(&coef).map(|(x, b)| x * b).sum::<f64>() + o)
            .collect();
        let se_fit = se_fit.then(|| {
            newx.outer_iter()
                .map(|row| row.dot(&self.var.dot(&row)).sqrt())
                .collect()
        });
        Ok(CoxPrediction { fit, se_fit })
    }

    /// `predict(type = "risk")`: `exp(lp)` with R's Taylor-series standard error.
    pub fn predict_risk(
        &self,
        newdata: Option<&CoxNewData>,
        se_fit: bool,
        reference: PredictReference,
    ) -> SurvivalResult<CoxPrediction> {
        let lp = self.predict_lp(newdata, se_fit, reference)?;
        let fit: Vec<f64> = lp.fit.iter().map(|v| v.exp()).collect();
        let se_fit = lp
            .se_fit
            .map(|se| se.iter().zip(&fit).map(|(s, p)| s * p.sqrt()).collect());
        Ok(CoxPrediction { fit, se_fit })
    }

    /// `predict(type = "terms")`: `assign` lists the columns of each term.
    pub fn predict_terms(
        &self,
        newdata: Option<&CoxNewData>,
        se_fit: bool,
        reference: PredictReference,
        assign: &[Vec<usize>],
    ) -> SurvivalResult<CoxTermsPrediction> {
        validate_assign(assign, self.nvar())?;
        let (newx, _) = self.prediction_rows(newdata, reference, true)?;
        let coef = self.coefficients_or_zero();
        let nterms = assign.len();
        let mut fit = vec![vec![0.0; nterms]; newx.nrows()];
        let mut se = se_fit.then(|| vec![vec![0.0; nterms]; newx.nrows()]);
        for (t, columns) in assign.iter().enumerate() {
            for (i, row) in newx.outer_iter().enumerate() {
                fit[i][t] = columns.iter().map(|&c| row[c] * coef[c]).sum();
                if let Some(se) = se.as_mut() {
                    let mut total = 0.0;
                    for &c1 in columns {
                        for &c2 in columns {
                            total += row[c1] * self.var[(c1, c2)] * row[c2];
                        }
                    }
                    se[i][t] = total.sqrt();
                }
            }
        }
        Ok(CoxTermsPrediction {
            fit,
            se_fit: se,
            constant: coef.iter().zip(&self.means).map(|(b, m)| b * m).sum(),
        })
    }

    /// `predict(type = "expected")`: the expected number of events over each
    /// observation's follow-up.  `newdata` needs `time` (and `entry` for a
    /// counting-process fit).
    pub fn predict_expected(
        &self,
        newdata: Option<&CoxNewData>,
        se_fit: bool,
    ) -> SurvivalResult<CoxPrediction> {
        let counting = self.entry.is_some();
        let Some(newdata) = newdata else {
            let fit: Vec<f64> = self
                .status
                .iter()
                .zip(&self.residuals)
                .map(|(&s, r)| f64::from(s) - r)
                .collect();
            if !se_fit {
                return Ok(CoxPrediction { fit, se_fit: None });
            }
            // predict.coxph's exp(linear.predictors), relative to the mean
            // offset as the baseline curves are (the standard error does not
            // depend on that scale)
            let offset_mean = self.offset_mean();
            let risk: Vec<f64> = self
                .linear_predictors
                .iter()
                .map(|lp| (lp - offset_mean).exp())
                .collect();
            let se = self.expected_se(
                self.x.view(),
                Some(&self.means),
                &risk,
                &self.sorted.stratum_index,
                self.entry.as_deref(),
                &self.time,
            )?;
            return Ok(CoxPrediction {
                fit,
                se_fit: Some(se),
            });
        };
        self.check_newdata(newdata)?;
        let Some(new_time) = newdata.time.as_deref() else {
            return Err(SurvivalError::invalid_input(
                "newdata must contain the follow-up time for type = 'expected'",
            ));
        };
        if counting && newdata.entry.is_none() {
            return Err(SurvivalError::invalid_input(
                "New data has a different survival type than the model",
            ));
        }
        let (x2c, risk2) = self.centered_newdata(newdata);
        let curves = self.baseline_curves()?;
        let stratum_index: Vec<usize> = match &newdata.strata {
            Some(strata) => strata
                .iter()
                .map(|&code| self.sorted.position_of(code).expect("checked"))
                .collect(),
            None => vec![0; newdata.nrows()],
        };
        let fit: Vec<f64> = (0..newdata.nrows())
            .map(|i| {
                let curve = &curves[stratum_index[i]];
                let stop = cumhaz_at(curve, new_time[i]);
                let start = newdata
                    .entry
                    .as_ref()
                    .map_or(0.0, |entry| cumhaz_at(curve, entry[i]));
                (stop - start) * risk2[i]
            })
            .collect();
        let se = if se_fit {
            Some(self.expected_se(
                x2c.view(),
                None,
                &risk2,
                &stratum_index,
                newdata.entry.as_deref(),
                new_time,
            )?)
        } else {
            None
        };
        Ok(CoxPrediction { fit, se_fit: se })
    }

    /// Standard error of an expected count (`predict.coxph`, `type =
    /// "expected"`): `sqrt(varh + dt' V dt) * risk`, differenced over
    /// (entry, time] for counting-process data.  The covariate rows are
    /// `x - means` when `means` is given, `x` itself otherwise.
    #[allow(clippy::too_many_arguments)]
    fn expected_se(
        &self,
        x: ArrayView2<'_, f64>,
        means: Option<&[f64]>,
        risk: &[f64],
        stratum_index: &[usize],
        entry: Option<&[f64]>,
        time: &[f64],
    ) -> SurvivalResult<Vec<f64>> {
        let curves = self.baseline_curves()?;
        let integrated: Vec<IntegratedCurve> = curves.iter().map(integrate_curve).collect();
        let variance_at =
            |curve: &AgsurvCurve, integrated: &IntegratedCurve, t: f64, row: usize| {
                let chaz = cumhaz_at(curve, t);
                let varh = step_at(&curve.time, &integrated.cum_varhaz, t, 0.0);
                let xbar = cum_xbar_at(curve, integrated, t);
                let dt: Vec<f64> = (0..self.nvar())
                    .map(|k| chaz * (x[(row, k)] - means.map_or(0.0, |m| m[k])) - xbar[k])
                    .collect();
                let mut quad = 0.0;
                for (i, &left) in dt.iter().enumerate() {
                    for (j, &right) in dt.iter().enumerate() {
                        quad += left * self.var[(i, j)] * right;
                    }
                }
                varh + quad
            };
        Ok((0..time.len())
            .map(|i| {
                let s = stratum_index[i];
                let v2 = variance_at(&curves[s], &integrated[s], time[i], i);
                let v1 = entry.map_or(0.0, |entry| {
                    variance_at(&curves[s], &integrated[s], entry[i], i)
                });
                (v2 - v1).sqrt() * risk[i]
            })
            .collect())
    }

    /// `predict(type = "survival")`: `exp(-expected)`.
    pub fn predict_survival(
        &self,
        newdata: Option<&CoxNewData>,
        se_fit: bool,
    ) -> SurvivalResult<CoxPrediction> {
        let expected = self.predict_expected(newdata, se_fit)?;
        let fit: Vec<f64> = expected.fit.iter().map(|e| (-e).exp()).collect();
        let se_fit = expected
            .se_fit
            .map(|se| se.iter().zip(&fit).map(|(s, p)| s * p).collect());
        Ok(CoxPrediction { fit, se_fit })
    }

    pub fn hazard_ratios(&self) -> Vec<f64> {
        self.coefficients.iter().map(|b| b.exp()).collect()
    }
}

/// `coxph()`'s concordance step: `concordancefit(Y, lp, strata, weights,
/// cluster, reverse = TRUE, timefix = FALSE)` on the fitted linear
/// predictors.
fn fit_concordance(
    time: &[f64],
    entry: Option<&[f64]>,
    status: &[i32],
    linear_predictors: &[f64],
    weights: &[f64],
    strata: Option<&[i32]>,
    cluster: Option<&[i32]>,
) -> SurvivalResult<ConcordanceFit> {
    let x = ArrayView2::from_shape((linear_predictors.len(), 1), linear_predictors)
        .map_err(|err| SurvivalError::computation(err.to_string()))?;
    let options = ConcordanceOptions {
        reverse: true,
        timefix: false,
        ..ConcordanceOptions::default()
    };
    linear_predictor_concordance(time, entry, status, x, weights, strata, cluster, &options)
}

/// `concordancefit(Y, x, strata, weights, cluster)` on the response of a
/// Cox model's data, `x` holding the linear predictors of one or more fits
/// to those data: the concordance step of `coxph()` and `concordance.coxph`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn linear_predictor_concordance(
    time: &[f64],
    entry: Option<&[f64]>,
    status: &[i32],
    x: ArrayView2<'_, f64>,
    weights: &[f64],
    strata: Option<&[i32]>,
    cluster: Option<&[i32]>,
    options: &ConcordanceOptions,
) -> SurvivalResult<ConcordanceFit> {
    match entry {
        Some(entry) => {
            let data = CountingProcessData {
                start: entry.to_vec(),
                stop: time.to_vec(),
                event: status.to_vec(),
            };
            concordancefit(
                SurvResponse::Counting(&data),
                x,
                Some(weights),
                strata,
                cluster,
                options,
            )
        }
        None => {
            let data = SurvivalData {
                time: time.to_vec(),
                status: status.to_vec(),
            };
            concordancefit(
                SurvResponse::Right(&data),
                x,
                Some(weights),
                strata,
                cluster,
                options,
            )
        }
    }
}

/// Checks that `assign` groups existing columns.  A term may have no
/// columns left (all of them aliased); its prediction is then 0, as in
/// `predict.coxph`.
pub(crate) fn validate_assign(assign: &[Vec<usize>], nvar: usize) -> SurvivalResult<()> {
    for (term, columns) in assign.iter().enumerate() {
        if let Some(column) = columns.iter().find(|&&c| c >= nvar) {
            return Err(SurvivalError::invalid_input(format!(
                "assign[{term}] refers to column {column}, but the model has {nvar}"
            )));
        }
    }
    Ok(())
}

/// One column per coefficient, the default term structure.
pub(crate) fn default_assign(nvar: usize) -> Vec<Vec<usize>> {
    (0..nvar).map(|c| vec![c]).collect()
}

fn finish_curve(
    stratum: i32,
    curve: crate::surv_analysis::agsurv::CoxSurvCurve,
    censor: bool,
) -> CoxSurvfitCurve {
    let keep: Vec<usize> = (0..curve.time.len())
        .filter(|&g| censor || curve.n_event[g] > 0.0)
        .collect();
    let pick = |values: &[f64]| keep.iter().map(|&g| values[g]).collect::<Vec<_>>();
    let pick_rows = |matrix: &Array2<f64>| {
        keep.iter()
            .map(|&g| matrix.row(g).to_vec())
            .collect::<Vec<_>>()
    };
    CoxSurvfitCurve {
        stratum,
        n: curve.n,
        time: pick(&curve.time),
        n_risk: pick(&curve.n_risk),
        n_event: pick(&curve.n_event),
        n_censor: pick(&curve.n_censor),
        surv: pick_rows(&curve.surv),
        cumhaz: pick_rows(&curve.cumhaz),
        std_err: curve.std_err.as_ref().map(pick_rows),
    }
}

/// A `CoxNewData` from the binding arguments; `None` without `x`.
pub(crate) fn newdata_from_python(
    x: Option<FloatMatrix>,
    strata: Option<IntVec>,
    offset: Option<FloatVec>,
    time: Option<FloatVec>,
    entry: Option<FloatVec>,
) -> SurvivalResult<Option<CoxNewData>> {
    newdata_from_python_impl(x, strata, offset, time, entry, false)
}

pub(crate) fn prediction_from_python(
    x: Option<FloatMatrix>,
    strata: Option<IntVec>,
    offset: Option<FloatVec>,
    time: Option<FloatVec>,
    entry: Option<FloatVec>,
) -> SurvivalResult<Option<CoxNewData>> {
    newdata_from_python_impl(x, strata, offset, time, entry, true)
}

fn newdata_from_python_impl(
    x: Option<FloatMatrix>,
    strata: Option<IntVec>,
    offset: Option<FloatVec>,
    time: Option<FloatVec>,
    entry: Option<FloatVec>,
    allow_missing: bool,
) -> SurvivalResult<Option<CoxNewData>> {
    let Some(x) = x else {
        if strata.is_some() || offset.is_some() || time.is_some() || entry.is_some() {
            return Err(SurvivalError::invalid_input(
                "new_strata, new_offset, new_time and new_entry require newdata",
            ));
        }
        return Ok(None);
    };
    Ok(Some(CoxNewData::validated(
        x.into_inner(),
        strata.map(IntVec::into_inner),
        offset.map(FloatVec::into_inner),
        time.map(FloatVec::into_inner),
        entry.map(FloatVec::into_inner),
        allow_missing,
    )?))
}

#[pymethods]
impl CoxPHFit {
    /// Pickle and copy support (see `internal::pickle`).
    #[cfg(feature = "python")]
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<crate::internal::pickle::Reduced<'py>> {
        crate::internal::pickle::reduce(py, self)
    }

    #[getter]
    fn var(&self) -> Vec<Vec<f64>> {
        matrix_rows(&self.var)
    }

    #[getter]
    fn naive_var(&self) -> Option<Vec<Vec<f64>>> {
        self.naive_var.as_ref().map(matrix_rows)
    }

    #[getter]
    fn x(&self) -> Vec<Vec<f64>> {
        matrix_rows(&self.x)
    }

    #[getter(nvar)]
    fn nvar_getter(&self) -> usize {
        self.nvar()
    }

    #[pyo3(name = "hazard_ratios")]
    fn hazard_ratios_py(&self) -> Vec<f64> {
        self.hazard_ratios()
    }

    /// `basehaz(fit, centered)`.
    #[pyo3(name = "basehaz", signature = (centered = true))]
    fn basehaz_py(&self, centered: bool) -> PyResult<Basehaz> {
        Ok(self.basehaz(centered)?)
    }

    /// `predict(fit, newdata, type, se.fit, reference)` for the vector-valued
    /// types `lp`, `risk`, `expected` and `survival`.
    #[pyo3(signature = (r#type = "lp", newdata = None, new_strata = None, new_offset = None, new_time = None, new_entry = None, se_fit = false, reference = "strata"))]
    #[allow(clippy::too_many_arguments)]
    fn predict(
        &self,
        py: Python<'_>,
        r#type: &str,
        newdata: Option<FloatMatrix>,
        new_strata: Option<IntVec>,
        new_offset: Option<FloatVec>,
        new_time: Option<FloatVec>,
        new_entry: Option<FloatVec>,
        se_fit: bool,
        reference: &str,
    ) -> PyResult<CoxPrediction> {
        let newdata = prediction_from_python(newdata, new_strata, new_offset, new_time, new_entry)?;
        let reference = PredictReference::parse(reference)?;
        let newdata = newdata.as_ref();
        Ok(match r#type {
            "lp" => py.detach(|| self.predict_lp(newdata, se_fit, reference))?,
            "risk" => py.detach(|| self.predict_risk(newdata, se_fit, reference))?,
            "expected" => py.detach(|| self.predict_expected(newdata, se_fit))?,
            "survival" => py.detach(|| self.predict_survival(newdata, se_fit))?,
            other => {
                return Err(SurvivalError::invalid_input(format!(
                    "type must be 'lp', 'risk', 'expected' or 'survival', got '{other}'; use predict_terms for 'terms'"
                ))
                .into());
            }
        })
    }

    /// `predict(fit, type = "terms")`; `assign` lists the columns of each
    /// term (default: one term per column).
    #[pyo3(name = "predict_terms", signature = (newdata = None, new_strata = None, new_offset = None, se_fit = false, reference = "sample", assign = None))]
    #[allow(clippy::too_many_arguments)]
    fn predict_terms_py(
        &self,
        py: Python<'_>,
        newdata: Option<FloatMatrix>,
        new_strata: Option<IntVec>,
        new_offset: Option<FloatVec>,
        se_fit: bool,
        reference: &str,
        assign: Option<Vec<Vec<usize>>>,
    ) -> PyResult<CoxTermsPrediction> {
        let newdata = prediction_from_python(newdata, new_strata, new_offset, None, None)?;
        let reference = PredictReference::parse(reference)?;
        let assign = assign.unwrap_or_else(|| default_assign(self.nvar()));
        Ok(py.detach(|| self.predict_terms(newdata.as_ref(), se_fit, reference, &assign))?)
    }

    /// `survfit(fit, newdata, stype, ctype, se.fit, censor, start.time)`.
    #[pyo3(name = "survfit", signature = (newdata = None, new_strata = None, new_offset = None, stype = 2, ctype = None, se_fit = true, censor = true, start_time = None))]
    #[allow(clippy::too_many_arguments)]
    fn survfit_py(
        &self,
        py: Python<'_>,
        newdata: Option<FloatMatrix>,
        new_strata: Option<IntVec>,
        new_offset: Option<FloatVec>,
        stype: u8,
        ctype: Option<u8>,
        se_fit: bool,
        censor: bool,
        start_time: Option<f64>,
    ) -> PyResult<Vec<CoxSurvfitCurve>> {
        let newdata = newdata_from_python(newdata, new_strata, new_offset, None, None)?;
        let options = SurvfitOptions {
            stype,
            ctype,
            se_fit,
            censor,
            start_time,
        };
        Ok(py.detach(|| self.survfit(newdata.as_ref(), options))?)
    }

    /// Survival at requested times: a NumPy matrix of shape (n_times, n_rows).
    /// Omit newdata to predict for the original training rows.
    #[pyo3(name = "predict_survival_at", signature = (times, newdata = None, new_strata = None, new_offset = None))]
    fn predict_survival_at_py(
        &self,
        py: Python<'_>,
        times: FloatVec,
        newdata: Option<FloatMatrix>,
        new_strata: Option<IntVec>,
        new_offset: Option<FloatVec>,
    ) -> PyResult<FloatMatrix> {
        let newdata = newdata_from_python(newdata, new_strata, new_offset, None, None)?;
        Ok(FloatMatrix::new(py.detach(|| {
            self.predict_survival_at(&times, newdata.as_ref())
        })?))
    }

    #[pyo3(name = "expected_survival", signature = (newdata, group, weights, new_strata=None, new_offset=None, y=None, times=None, method="ederer"))]
    #[allow(clippy::too_many_arguments)]
    fn expected_survival_py(
        &self,
        py: Python<'_>,
        newdata: FloatMatrix,
        group: IntVec,
        weights: FloatVec,
        new_strata: Option<IntVec>,
        new_offset: Option<FloatVec>,
        y: Option<FloatVec>,
        times: Option<FloatVec>,
        method: &str,
    ) -> PyResult<crate::population::SurvExpResult> {
        let new = newdata_from_python(Some(newdata), new_strata, new_offset, None, None)?
            .expect("newdata supplied");
        let group = group
            .iter()
            .map(|&v| {
                usize::try_from(v)
                    .map_err(|_| SurvivalError::invalid_input("group codes must be nonnegative"))
            })
            .collect::<SurvivalResult<Vec<_>>>()?;
        Ok(py.detach(|| {
            self.expected_survival(
                &new,
                &group,
                &weights,
                y.as_deref(),
                times.as_deref(),
                method,
            )
        })?)
    }

    /// `residuals(fit, type = "martingale", weighted, collapse)`.
    #[pyo3(name = "martingale_residuals", signature = (weighted = false, collapse = None))]
    fn martingale_residuals_py(
        &self,
        weighted: bool,
        collapse: Option<IntVec>,
    ) -> PyResult<Vec<f64>> {
        Ok(self.martingale_residuals(weighted, collapse.as_deref())?)
    }

    /// `residuals(fit, type = "deviance", weighted, collapse)`.
    #[pyo3(name = "deviance_residuals", signature = (weighted = false, collapse = None))]
    fn deviance_residuals_py(
        &self,
        weighted: bool,
        collapse: Option<IntVec>,
    ) -> PyResult<Vec<f64>> {
        Ok(self.deviance_residuals(weighted, collapse.as_deref())?)
    }

    /// `residuals(fit, type = "score", weighted, collapse)`.
    #[pyo3(name = "score_residuals", signature = (weighted = false, collapse = None))]
    fn score_residuals_py(
        &self,
        py: Python<'_>,
        weighted: bool,
        collapse: Option<IntVec>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let residuals = py.detach(|| self.score_residuals(weighted, collapse.as_deref()))?;
        Ok(matrix_rows(&residuals))
    }

    /// `residuals(fit, type = "dfbeta", weighted, collapse)`.
    #[pyo3(name = "dfbeta", signature = (weighted = true, collapse = None))]
    fn dfbeta_py(
        &self,
        py: Python<'_>,
        weighted: bool,
        collapse: Option<IntVec>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let dfbeta = py.detach(|| self.dfbeta(weighted, collapse.as_deref()))?;
        Ok(matrix_rows(&dfbeta))
    }

    /// `residuals(fit, type = "dfbetas", weighted, collapse)`.
    #[pyo3(name = "dfbetas", signature = (weighted = true, collapse = None))]
    fn dfbetas_py(
        &self,
        py: Python<'_>,
        weighted: bool,
        collapse: Option<IntVec>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let dfbetas = py.detach(|| self.dfbetas(weighted, collapse.as_deref()))?;
        Ok(matrix_rows(&dfbetas))
    }

    /// `residuals(fit, type = "schoenfeld", weighted)`.
    #[pyo3(name = "schoenfeld_residuals", signature = (weighted = false))]
    fn schoenfeld_residuals_py(
        &self,
        py: Python<'_>,
        weighted: bool,
    ) -> PyResult<SchoenfeldResiduals> {
        Ok(py.detach(|| self.schoenfeld_residuals(weighted))?)
    }

    /// `residuals(fit, type = "scaledsch", weighted)`.
    #[pyo3(name = "scaled_schoenfeld_residuals", signature = (weighted = false))]
    fn scaled_schoenfeld_residuals_py(
        &self,
        py: Python<'_>,
        weighted: bool,
    ) -> PyResult<SchoenfeldResiduals> {
        Ok(py.detach(|| self.scaled_schoenfeld_residuals(weighted))?)
    }

    /// `residuals(fit, type = "partial", weighted, collapse)`; `assign`
    /// lists the columns of each term (default: one term per column).
    #[pyo3(name = "partial_residuals", signature = (assign = None, weighted = false, collapse = None))]
    fn partial_residuals_py(
        &self,
        py: Python<'_>,
        assign: Option<Vec<Vec<usize>>>,
        weighted: bool,
        collapse: Option<IntVec>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let assign = assign.unwrap_or_else(|| default_assign(self.nvar()));
        let residuals =
            py.detach(|| self.partial_residuals(&assign, weighted, collapse.as_deref()))?;
        Ok(matrix_rows(&residuals))
    }

    /// `survfit(fit, newdata, id)` for time-dependent new data.
    #[pyo3(name = "survfit_individual", signature = (newdata, new_entry, new_time, id, new_strata = None, new_offset = None, stype = 2, ctype = None, se_fit = true, censor = true, start_time = None))]
    #[allow(clippy::too_many_arguments)]
    fn survfit_individual_py(
        &self,
        py: Python<'_>,
        newdata: FloatMatrix,
        new_entry: FloatVec,
        new_time: FloatVec,
        id: IntVec,
        new_strata: Option<IntVec>,
        new_offset: Option<FloatVec>,
        stype: u8,
        ctype: Option<u8>,
        se_fit: bool,
        censor: bool,
        start_time: Option<f64>,
    ) -> PyResult<Vec<CoxSurvfitCurve>> {
        let newdata = newdata_from_python(
            Some(newdata),
            new_strata,
            new_offset,
            Some(new_time),
            Some(new_entry),
        )?
        .expect("newdata was supplied");
        let options = SurvfitOptions {
            stype,
            ctype,
            se_fit,
            censor,
            start_time,
        };
        Ok(py.detach(|| self.survfit_individual(&newdata, &id, options))?)
    }
}

/// `coxph()` on explicit data: fits `Surv(time, status) ~ x` (or
/// `Surv(entry, time, status) ~ x` when `entry` is given).
///
/// `nocenter` lists the values of a column that exempt it from centring
/// (R's default `c(-1, 0, 1)` when omitted; an empty list centres every
/// column, R's `nocenter = NULL`).  `cluster` requests the robust variance.
#[pyfunction]
#[pyo3(signature = (time, status, x, entry=None, strata=None, weights=None, offset=None, method="efron", init=None, iter_max=None, eps=None, toler_chol=None, nocenter=None, cluster=None, robust=None))]
#[allow(clippy::too_many_arguments)]
pub fn coxph_fit(
    py: Python<'_>,
    time: FloatVec,
    status: IntVec,
    x: FloatMatrix,
    entry: Option<FloatVec>,
    strata: Option<IntVec>,
    weights: Option<FloatVec>,
    offset: Option<FloatVec>,
    method: &str,
    init: Option<Vec<f64>>,
    iter_max: Option<usize>,
    eps: Option<f64>,
    toler_chol: Option<f64>,
    nocenter: Option<Vec<f64>>,
    cluster: Option<IntVec>,
    robust: Option<bool>,
) -> PyResult<CoxPHFit> {
    if x.nrow() != time.len() {
        return Err(SurvivalError::invalid_input(format!(
            "x has {} rows but time has {}",
            x.nrow(),
            time.len()
        ))
        .into());
    }
    let data = CoxphData::try_new(
        time.into_inner(),
        entry.map(FloatVec::into_inner),
        status.into_inner(),
        x.into_inner(),
        weights.map(FloatVec::into_inner),
        strata.map(IntVec::into_inner),
        offset.map(FloatVec::into_inner),
    )?;
    let defaults = CoxphOptions::default();
    let options = CoxphOptions {
        method: TieMethod::parse(Some(method))?,
        init,
        iter_max: iter_max.unwrap_or(defaults.iter_max),
        eps: eps.unwrap_or(defaults.eps),
        toler_chol: toler_chol.unwrap_or(defaults.toler_chol),
        nocenter: nocenter.or(defaults.nocenter),
        cluster: cluster.map(IntVec::into_inner),
        robust,
    };
    Ok(py.detach(move || CoxPHFit::fit(data, options))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lung_like_data() -> CoxphData {
        let time = vec![1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0];
        let status = vec![1, 1, 0, 1, 1, 0, 1, 0];
        let x1 = [0.2, 0.8, 0.4, 1.1, 0.7, 0.3, 1.3, 0.5];
        let x2 = [1.0, 0.2, 0.7, 1.3, 0.4, 1.1, 0.5, 0.9];
        let x = Array2::from_shape_vec(
            (8, 2),
            x1.iter().zip(&x2).flat_map(|(&a, &b)| [a, b]).collect(),
        )
        .unwrap();
        CoxphData::try_new(time, None, status, x, None, None, None).unwrap()
    }

    #[test]
    fn missing_prediction_values_preserve_independent_terms_and_errors() {
        let fit = CoxPHFit::fit(lung_like_data(), CoxphOptions::default()).unwrap();
        let x = ndarray::array![[f64::NAN, 2.0], [1.0, 3.0]];
        assert!(CoxNewData::try_new(x.clone(), None, None, None, None).is_err());
        let newdata =
            CoxNewData::try_new_prediction(x, None, Some(vec![0.0, f64::NAN]), None, None).unwrap();
        let lp = fit
            .predict_lp(Some(&newdata), true, PredictReference::Zero)
            .unwrap();
        assert!(lp.fit.iter().all(|v| v.is_nan()));
        let se = lp.se_fit.unwrap();
        assert!(se[0].is_nan());
        assert!(se[1].is_finite());
        let terms = fit
            .predict_terms(
                Some(&newdata),
                true,
                PredictReference::Zero,
                &[vec![0], vec![1]],
            )
            .unwrap();
        assert!(terms.fit[0][0].is_nan());
        assert_eq!(terms.fit[0][1], 2.0 * fit.coefficients[1]);
        assert!(terms.fit[1].iter().all(|v| v.is_finite()));
        assert!(terms.se_fit.unwrap()[0][1].is_finite());
        assert!(
            CoxNewData::try_new_prediction(
                ndarray::array![[f64::INFINITY, 1.0]],
                None,
                None,
                None,
                None
            )
            .is_err()
        );
        assert!(
            CoxNewData::try_new_prediction(
                ndarray::array![[1.0, 1.0]],
                None,
                None,
                Some(vec![f64::NAN]),
                None
            )
            .is_err()
        );
    }

    #[test]
    fn default_controls_match_reference_efron_fit() {
        let fit = CoxPHFit::fit(lung_like_data(), CoxphOptions::default()).unwrap();
        let expected = [0.103_056_235_224_469_12, -1.021_973_929_290_916];
        for (actual, expected) in fit.coefficients.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert!((fit.loglik[0] - -7.714_231_144_849_085_5).abs() < 1e-12);
        assert!((fit.loglik[1] - -7.430_873_243_936_032).abs() < 1e-12);
        assert_eq!(fit.flag, 2);
        assert_eq!(fit.iter, 4);
        assert_eq!(fit.n, 8);
        assert_eq!(fit.nevent, 5);
        assert_eq!(fit.method, TieMethod::Efron);
        // Linear predictors are centred at the means.
        let mean_lp: f64 = fit.linear_predictors.iter().sum::<f64>() / 8.0;
        assert!(mean_lp.abs() < 1e-12);
        assert_eq!(fit.residuals.len(), 8);
        let total: f64 = fit.residuals.iter().sum();
        assert!(total.abs() < 1e-10, "martingale residuals sum to zero");
    }

    #[test]
    fn default_rank_tolerance_preserves_near_collinear_columns() {
        let n = 20;
        let time: Vec<f64> = (1..=n).map(|value| value as f64).collect();
        let status: Vec<i32> = (0..n).map(|idx| i32::from(idx % 3 != 0)).collect();
        let rows: Vec<f64> = (0..n)
            .flat_map(|idx| {
                let first = (idx % 7) as f64 * 0.3 + (idx / 7) as f64 * 0.11;
                let direction = if idx % 2 == 0 { 1.0 } else { -1.0 };
                let perturbation = direction * (0.2 + (idx % 5) as f64 * 0.13);
                [first, first + 1e-5 * perturbation]
            })
            .collect();
        let x = Array2::from_shape_vec((n, 2), rows).unwrap();
        let fit_with = |toler: Option<f64>| {
            let data = CoxphData::try_new(
                time.clone(),
                None,
                status.clone(),
                x.clone(),
                None,
                None,
                None,
            )
            .unwrap();
            let options = CoxphOptions {
                method: TieMethod::Breslow,
                iter_max: 0,
                toler_chol: toler.unwrap_or(COX_RANK_TOLERANCE),
                ..CoxphOptions::default()
            };
            CoxPHFit::fit(data, options).unwrap()
        };
        assert_eq!(fit_with(None).flag, 2);
        assert_eq!(fit_with(Some(1e-9)).flag, 1);
    }

    #[test]
    fn robust_variance_is_the_dfbeta_crossproduct() {
        let data = lung_like_data();
        let options = CoxphOptions {
            cluster: Some(vec![0, 0, 1, 1, 2, 2, 3, 3]),
            ..CoxphOptions::default()
        };
        let fit = CoxPHFit::fit(data, options).unwrap();
        let naive = fit.naive_var.as_ref().expect("naive variance is kept");
        let dfbeta = fit
            .dfbeta_matrix(&fit.linear_predictors, true, fit.cluster.as_deref())
            .unwrap();
        let expected = crossprod(&dfbeta);
        for i in 0..2 {
            for j in 0..2 {
                assert!((fit.var[(i, j)] - expected[(i, j)]).abs() < 1e-12);
                assert!(fit.var[(i, j)] != naive[(i, j)] || fit.var[(i, j)] == 0.0);
            }
        }
        assert!(fit.rscore.is_some());
    }

    #[test]
    fn basehaz_uncentred_removes_the_mean_offset() {
        let fit = CoxPHFit::fit(lung_like_data(), CoxphOptions::default()).unwrap();
        let centred = fit.basehaz(true).unwrap();
        let uncentred = fit.basehaz(false).unwrap();
        let center: f64 = fit
            .means
            .iter()
            .zip(&fit.coefficients)
            .map(|(m, b)| m * b)
            .sum();
        assert_eq!(centred.time, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        for (c, u) in centred.hazard.iter().zip(&uncentred.hazard) {
            assert!((u - c * (-center).exp()).abs() < 1e-12);
        }
        assert!(centred.strata.is_none());
    }

    #[test]
    fn survfit_at_the_means_matches_the_baseline_hazard() {
        let fit = CoxPHFit::fit(lung_like_data(), CoxphOptions::default()).unwrap();
        let curves = fit.survfit(None, SurvfitOptions::default()).unwrap();
        assert_eq!(curves.len(), 1);
        let basehaz = fit.basehaz(true).unwrap();
        for (g, row) in curves[0].cumhaz.iter().enumerate() {
            assert!((row[0] - basehaz.hazard[g]).abs() < 1e-12);
            assert!((curves[0].surv[g][0] - (-row[0]).exp()).abs() < 1e-12);
        }
        assert!(curves[0].std_err.is_some());
        let expected = fit.predict_expected(None, true).unwrap();
        assert_eq!(expected.fit.len(), 8);
        for (e, (&s, r)) in expected
            .fit
            .iter()
            .zip(fit.status.iter().zip(&fit.residuals))
        {
            assert!((e - (f64::from(s) - r)).abs() < 1e-12);
        }
    }

    #[test]
    fn survival_at_requested_times_matches_full_curves() {
        let times = [8.0, 2.0, -1.0, 2.0, 3.5];
        for method in [TieMethod::Breslow, TieMethod::Efron, TieMethod::Exact] {
            for stratified in [false, true] {
                let mut data = lung_like_data();
                data.offset = Some(vec![0.1, 0.3, -0.2, 0.0, 0.4, 0.2, -0.1, 0.5]);
                if stratified {
                    data.strata = Some(vec![17, -3, 17, -3, 17, -3, 17, -3]);
                }
                let fit = CoxPHFit::fit(
                    data,
                    CoxphOptions {
                        method,
                        ..Default::default()
                    },
                )
                .unwrap();
                let newdata = CoxNewData::try_new(
                    fit.x.clone(),
                    fit.strata.clone(),
                    Some(fit.offset.clone()),
                    None,
                    None,
                )
                .unwrap();
                let full = fit
                    .survfit(
                        Some(&newdata),
                        SurvfitOptions {
                            se_fit: false,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                let actual = fit.predict_survival_at(&times, None).unwrap();
                assert_eq!(
                    actual,
                    fit.predict_survival_at(&times, Some(&newdata)).unwrap()
                );
                assert_eq!(actual.dim(), (times.len(), fit.n));
                for (i, &time) in times.iter().enumerate() {
                    for row in 0..fit.n {
                        let (curve, column) = if stratified {
                            (&full[row], 0)
                        } else {
                            (&full[0], row)
                        };
                        let index = find_interval(&curve.time, time, false);
                        let expected = if index == 0 {
                            1.0
                        } else {
                            curve.surv[index - 1][column]
                        };
                        assert_eq!(actual[(i, row)], expected);
                    }
                }
                assert_eq!(
                    fit.predict_survival_at(&[], None).unwrap().dim(),
                    (0, fit.n)
                );
                assert!(fit.predict_survival_at(&[f64::NAN], None).is_err());
                if stratified {
                    let missing_strata = CoxNewData {
                        strata: None,
                        ..newdata
                    };
                    assert!(
                        fit.predict_survival_at(&times, Some(&missing_strata))
                            .is_err()
                    );
                }
            }
        }
    }

    #[test]
    fn predictions_follow_the_reference_argument() {
        let fit = CoxPHFit::fit(lung_like_data(), CoxphOptions::default()).unwrap();
        let sample = fit
            .predict_lp(None, false, PredictReference::Sample)
            .unwrap();
        assert_eq!(sample.fit, fit.linear_predictors);
        let zero = fit.predict_lp(None, true, PredictReference::Zero).unwrap();
        let center: f64 = fit
            .means
            .iter()
            .zip(&fit.coefficients)
            .map(|(m, b)| m * b)
            .sum();
        for (z, lp) in zero.fit.iter().zip(&fit.linear_predictors) {
            assert!((z - (lp + center)).abs() < 1e-12);
        }
        assert!(zero.se_fit.is_some());
        let newdata = CoxNewData::try_new(
            Array2::from_shape_vec((1, 2), fit.means.clone()).unwrap(),
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let at_means = fit
            .predict_risk(Some(&newdata), true, PredictReference::Strata)
            .unwrap();
        assert!((at_means.fit[0] - 1.0).abs() < 1e-12);
        let terms = fit
            .predict_terms(None, true, PredictReference::Sample, &default_assign(2))
            .unwrap();
        assert_eq!(terms.fit.len(), 8);
        assert!((terms.constant - center).abs() < 1e-12);
    }

    #[test]
    fn newdata_offset_is_centred_only_in_the_use_x_branch() {
        let offset = vec![0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0];
        let data = CoxphData {
            offset: Some(offset.clone()),
            ..lung_like_data()
        };
        let fit = CoxPHFit::fit(data, CoxphOptions::default()).unwrap();
        let newdata =
            CoxNewData::try_new(fit.x.clone(), None, Some(offset.clone()), None, None).unwrap();
        // predict.coxph without se.fit: newx %*% beta + newoffset, so the
        // training rows give back the linear predictors
        let plain = fit
            .predict_lp(Some(&newdata), false, PredictReference::Sample)
            .unwrap();
        for (p, lp) in plain.fit.iter().zip(&fit.linear_predictors) {
            assert!((p - lp).abs() < 1e-12);
        }
        // with se.fit the offset is centred at mean(offset)
        let with_se = fit
            .predict_lp(Some(&newdata), true, PredictReference::Sample)
            .unwrap();
        let offset_mean = offset.iter().sum::<f64>() / offset.len() as f64;
        for (p, lp) in with_se.fit.iter().zip(&fit.linear_predictors) {
            assert!((p - (lp - offset_mean)).abs() < 1e-12);
        }
    }

    #[test]
    fn strata_and_counting_process_fits_keep_row_order() {
        let time = vec![5.0, 1.0, 4.0, 2.0, 3.0, 6.0, 8.0, 7.0];
        let entry = vec![0.0, 0.0, 1.0, 0.0, 0.5, 2.0, 0.0, 1.0];
        let status = vec![1, 1, 0, 0, 1, 0, 0, 1];
        let x =
            Array2::from_shape_vec((8, 1), vec![0.6, 0.5, 0.8, 1.0, 0.3, 0.4, 0.2, 0.9]).unwrap();
        let strata = vec![1, 0, 1, 0, 1, 0, 1, 0];
        let data = CoxphData::try_new(
            time.clone(),
            Some(entry.clone()),
            status.clone(),
            x.clone(),
            None,
            Some(strata.clone()),
            None,
        )
        .unwrap();
        let fit = CoxPHFit::fit(data, CoxphOptions::default()).unwrap();
        assert_eq!(fit.sorted.codes, vec![0, 1]);
        assert_eq!(fit.time, time);
        assert_eq!(fit.strata.as_deref(), Some(strata.as_slice()));
        let curves = fit.survfit(None, SurvfitOptions::default()).unwrap();
        assert_eq!(curves.len(), 2);
        assert_eq!(curves[0].stratum, 0);
        let basehaz = fit.basehaz(true).unwrap();
        assert_eq!(basehaz.strata.as_ref().unwrap().len(), basehaz.time.len());
        let total: f64 = fit.residuals.iter().sum();
        assert!(total.abs() < 1e-10);
    }

    /// `survfit(coxph(Surv(time, status) ~ x + strata(g), d), start.time =
    /// 15)` in R 3.8-12 (`d` from `set.seed(1)`, `x = rnorm(20)`): the rows
    /// before 15 leave, stratum 1 keeps an empty curve with `n = 0`, and the
    /// risk scores stay those of the whole fit.
    #[test]
    fn start_time_drops_the_earlier_rows_and_keeps_an_emptied_stratum() {
        let time: Vec<f64> = (1..=10).chain(21..=30).map(f64::from).collect();
        let status = [1, 0, 1, 1, 0].repeat(4);
        let strata: Vec<i32> = [1; 10].into_iter().chain([2; 10]).collect();
        let x = [
            -0.626_453_810_742_332_4,
            0.183_643_324_222_082_24,
            -0.835_628_612_410_047_2,
            1.595_280_802_137_791_6,
            0.329_507_771_815_360_5,
            -0.820_468_384_118_015_3,
            0.487_429_052_428_485_3,
            0.738_324_705_129_217_3,
            0.575_781_351_653_492_3,
            -0.305_388_387_156_356,
            1.511_781_168_450_848,
            0.389_843_236_411_431_1,
            -0.621_240_580_541_803_8,
            -2.214_699_887_177_5,
            1.124_930_918_143_108_2,
            -0.044_933_609_015_230_85,
            -0.016_190_263_098_946_087,
            0.943_836_210_685_299_2,
            0.821_221_195_098_088_6,
            0.593_901_321_217_508_8,
        ];
        let data = CoxphData::try_new(
            time,
            None,
            status,
            Array2::from_shape_vec((20, 1), x.to_vec()).unwrap(),
            None,
            Some(strata),
            None,
        )
        .unwrap();
        let fit = CoxPHFit::fit(data, CoxphOptions::default()).unwrap();
        let after = |start_time| SurvfitOptions {
            start_time: Some(start_time),
            ..SurvfitOptions::default()
        };
        let curves = fit.survfit(None, after(15.0)).unwrap();
        assert_eq!(curves.len(), 2);
        assert_eq!((curves[0].stratum, curves[0].n), (1, 0));
        assert!(curves[0].time.is_empty() && curves[0].surv.is_empty());
        assert_eq!(curves[1].n, 10);
        let times: Vec<f64> = (21..=30).map(f64::from).collect();
        assert_eq!(curves[1].time, times);
        let surv = [
            0.911_703_185_270_467,
            0.911_703_185_270_467,
            0.818_916_920_480_482,
            0.721_769_214_450_701,
            0.721_769_214_450_701,
            0.579_122_152_428_905,
            0.579_122_152_428_905,
            0.378_441_860_225_577,
            0.203_946_105_189_285,
            0.203_946_105_189_285,
        ];
        let std_err = [
            0.093_842_646_361_357,
            0.093_842_646_361_357,
            0.147_738_856_126_131,
            0.202_627_188_554_77,
            0.202_627_188_554_77,
            0.295_853_546_336_666,
            0.295_853_546_336_666,
            0.516_581_455_410_216,
            0.819_221_646_496_032,
            0.819_221_646_496_032,
        ];
        let se = curves[1].std_err.as_ref().unwrap();
        for g in 0..10 {
            assert!((curves[1].surv[g][0] - surv[g]).abs() < 1e-12 * surv[g]);
            assert!((se[g][0] - std_err[g]).abs() < 1e-12 * std_err[g]);
        }
        // the cached curves of the whole fit are left alone
        assert_eq!(
            fit.survfit(None, SurvfitOptions::default()).unwrap()[0].n,
            10
        );

        let kp = fit
            .survfit(
                None,
                SurvfitOptions {
                    stype: 1,
                    ..after(15.0)
                },
            )
            .unwrap();
        assert!((kp[1].surv[9][0] - 0.139_802_086_569_338).abs() < 1e-12);

        // the last row, at 30, is censored
        let error = fit.survfit(None, after(30.0)).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("start.time argument has removed all endpoints"),
            "{error}"
        );
    }

    /// `coxph()` fits at the centred offset and adds the mean back, so
    /// offsets near the limits of `exp()` give R's fit
    /// (`coxph(Surv(time, status) ~ x1 + offset(708 + x2))`, and `-740`).
    #[test]
    fn offsets_are_centred_before_fitting() {
        let base = lung_like_data();
        let lp_without_shift = [
            0.689_312_979_161_027_5,
            0.292_366_411_600_775_64,
            0.523_664_123_307_610_3,
            1.593_893_127_820_649_6,
            0.425_190_839_527_484_2,
            0.856_488_551_234_319,
            0.928_244_271_967_232_4,
            0.790_839_695_380_901_6,
        ];
        let residuals = [
            0.833_716_415_337_278_3,
            0.888_195_914_321_604_3,
            -0.190_854_674_879_980_42,
            -0.158_633_253_137_838_05,
            0.639_931_579_226_752_9,
            -0.668_367_507_721_713_4,
            -0.252_386_478_362_365_3,
            -1.091_601_994_783_737_2,
        ];
        for shift in [708.0, -740.0] {
            let offset = base.x.column(1).iter().map(|x2| shift + x2).collect();
            let data = CoxphData::try_new(
                base.time.clone(),
                None,
                base.status.clone(),
                base.x.slice(ndarray::s![.., ..1]).to_owned(),
                None,
                None,
                Some(offset),
            )
            .unwrap();
            let fit = CoxPHFit::fit(data, CoxphOptions::default()).unwrap();
            assert!((fit.coefficients[0] - 0.671_755_720_732_935_5).abs() < 1e-9);
            assert!((fit.loglik[0] - -8.483_319_079_073_608).abs() < 1e-9);
            assert!((fit.loglik[1] - -8.313_974_981_255_075).abs() < 1e-9);
            for (lp, expected) in fit.linear_predictors.iter().zip(lp_without_shift) {
                assert!((lp - (shift + expected)).abs() < 1e-9);
            }
            for (actual, expected) in fit.residuals.iter().zip(residuals) {
                assert!((actual - expected).abs() < 1e-9);
            }
            assert!((fit.concordance.concordance[0] - 6.0 / 19.0).abs() < 1e-12);
            // The offset is kept as given.
            assert!((fit.offset[0] - (shift + 1.0)).abs() < 1e-12);
        }
    }

    /// Without events `coxph()` returns before any fitter runs: R's
    /// `coxph(Surv(start, stop, status) ~ x + offset(off), weights = w)` on
    /// four censored rows.
    #[test]
    fn data_without_events_gets_coxph_skeleton_fit() {
        let data = CoxphData::try_new(
            vec![2.0, 3.0, 4.0, 5.0],
            Some(vec![0.0, 0.0, 1.0, 1.0]),
            vec![0; 4],
            Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            Some(vec![1.0, 2.0, 3.0, 4.0]),
            None,
            Some(vec![0.1, 0.2, 0.3, 0.4]),
        )
        .unwrap();
        let fit = CoxPHFit::fit(data, CoxphOptions::default()).unwrap();
        assert!(fit.coefficients[0].is_nan());
        assert_eq!(fit.var[(0, 0)], 0.0);
        assert_eq!(fit.loglik, [0.0, 0.0]);
        assert_eq!((fit.score, fit.wald_test, fit.iter), (0.0, 0.0, 0));
        // Unweighted column means; the linear predictors are the centred offset.
        assert_eq!(fit.means, vec![2.5]);
        for (lp, expected) in fit.linear_predictors.iter().zip([-0.15, -0.05, 0.05, 0.15]) {
            assert!((lp - expected).abs() < 1e-12);
        }
        assert_eq!(fit.residuals, vec![0.0; 4]);
        assert_eq!(fit.nevent, 0);
        assert!(fit.concordance.concordance[0].is_nan());
        assert_eq!(fit.concordance.count[0].concordant, 0.0);
        assert!(fit.concordance.var.is_none());
    }

    #[test]
    fn predictors_and_weights_are_checked_only_for_data_with_events() {
        let data = |status: Vec<i32>| {
            CoxphData::try_new(
                vec![2.0, 3.0, 4.0, 5.0],
                None,
                status,
                Array2::from_shape_vec((4, 1), vec![1.0, f64::INFINITY, 3.0, 4.0]).unwrap(),
                Some(vec![0.0, 2.0, 3.0, 4.0]),
                None,
                None,
            )
            .unwrap()
        };
        let fit = CoxPHFit::fit(data(vec![0; 4]), CoxphOptions::default()).unwrap();
        assert!(fit.coefficients[0].is_nan());
        assert_eq!(fit.means, vec![f64::INFINITY]);
        let err = CoxPHFit::fit(data(vec![1, 0, 0, 0]), CoxphOptions::default()).unwrap_err();
        assert!(err.to_string().contains("x contains non-finite value inf"));
        let mut finite = data(vec![1, 0, 0, 0]);
        finite.x[(1, 0)] = 2.0;
        let err = CoxPHFit::fit(finite, CoxphOptions::default()).unwrap_err();
        assert!(err.to_string().contains("Invalid weights, must be >0"));
    }

    #[test]
    fn null_model_reports_the_log_likelihood_and_residuals() {
        let data = CoxphData::try_new(
            vec![1.0, 2.0, 3.0],
            None,
            vec![1, 1, 0],
            Array2::zeros((3, 0)),
            None,
            None,
            None,
        )
        .unwrap();
        let fit = CoxPHFit::fit(data, CoxphOptions::default()).unwrap();
        assert!(fit.coefficients.is_empty());
        assert_eq!(fit.linear_predictors, vec![0.0; 3]);
        assert!((fit.residuals[0] - (1.0 - 1.0 / 3.0)).abs() < 1e-12);
        assert_eq!(fit.wald_test, 0.0);
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        assert!(
            CoxphData::try_new(
                vec![],
                None,
                vec![],
                Array2::zeros((0, 1)),
                None,
                None,
                None
            )
            .is_err()
        );
        assert!(
            CoxphData::try_new(
                vec![1.0, 2.0],
                None,
                vec![1, 2],
                Array2::zeros((2, 1)),
                None,
                None,
                None
            )
            .is_err()
        );
        assert!(
            CoxphData::try_new(
                vec![1.0, 2.0],
                Some(vec![0.0, 2.0]),
                vec![1, 0],
                Array2::zeros((2, 1)),
                None,
                None,
                None
            )
            .is_err()
        );
        let data = lung_like_data();
        let options = CoxphOptions {
            method: TieMethod::Exact,
            robust: Some(true),
            ..CoxphOptions::default()
        };
        assert!(CoxPHFit::fit(data, options).is_err());
    }
}
