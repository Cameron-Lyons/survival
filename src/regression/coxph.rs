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

use crate::concordance::{ConcordanceFit, ConcordanceOptions, concordancefit};
use crate::constants::{COX_CONVERGENCE_TOLERANCE, COX_MAX_ITER, COX_RANK_TOLERANCE};
use crate::core::SurvResponse;
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::typed_inputs::{CountingProcessData, SurvivalData};
use crate::internal::validation::{validate_binary_i32, validate_finite, validate_length};
use crate::regression::cox_optimizer::{CoxFitBuilder, TieMethod};
use crate::regression::coxph_diagnostics::{
    ResidualType, Residuals, SchoenfeldResiduals, collapse_rows, martingale_residuals,
    schoenfeld_residuals, score_residuals,
};
use crate::regression::coxph_wtest::{wald_statistic, wald_tests};
use crate::surv_analysis::agsurv::{
    AgsurvCurve, AgsurvData, CoxSurvType, IndividualInterval, IntegratedCurve, agsurv_rows,
    cum_xbar_at, cumhaz_at, expand_curve, individual_curve, integrate_curve, step_at,
};
use ndarray::{Array1, Array2, ArrayView2};
use pyo3::prelude::*;
use std::sync::OnceLock;

/// Validated inputs of a Cox fit, in the caller's row order (`Surv(time,
/// status) ~ x` or `Surv(entry, time, status) ~ x`).
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
        if let Some(value) = x.iter().find(|value| !value.is_finite()) {
            return Err(SurvivalError::invalid_input(format!(
                "x contains non-finite value {value}"
            )));
        }
        if let Some(entry) = &entry {
            validate_length(n, entry.len(), "entry")?;
            validate_finite(entry, "entry")?;
            if let Some(index) = (0..n).find(|&i| entry[i] >= time[i]) {
                return Err(SurvivalError::invalid_input(format!(
                    "Stop time must be > start time (row {index})"
                )));
            }
        }
        if let Some(weights) = &weights {
            validate_length(n, weights.len(), "weights")?;
            validate_finite(weights, "weights")?;
            if weights.iter().any(|&w| w <= 0.0) {
                return Err(SurvivalError::invalid_input("Invalid weights, must be >0"));
            }
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
#[derive(Debug, Clone, PartialEq)]
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
#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone)]
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
        let m = x.nrows();
        if m == 0 {
            return Err(SurvivalError::invalid_input(
                "newdata must have at least one row",
            ));
        }
        if let Some(value) = x.iter().find(|value| !value.is_finite()) {
            return Err(SurvivalError::invalid_input(format!(
                "newdata contains non-finite value {value}"
            )));
        }
        if let Some(strata) = &strata {
            validate_length(m, strata.len(), "newdata strata")?;
        }
        if let Some(offset) = &offset {
            validate_length(m, offset.len(), "newdata offset")?;
            validate_finite(offset, "newdata offset")?;
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
}

impl Default for SurvfitOptions {
    fn default() -> Self {
        Self {
            stype: 2,
            ctype: None,
            se_fit: true,
            censor: true,
        }
    }
}

fn matrix_rows(matrix: &Array2<f64>) -> Vec<Vec<f64>> {
    matrix.outer_iter().map(|row| row.to_vec()).collect()
}

fn matrix_from_rows(rows: &[Vec<f64>], name: &str) -> SurvivalResult<Array2<f64>> {
    let ncols = rows.first().map_or(0, Vec::len);
    if rows.iter().any(|row| row.len() != ncols) {
        return Err(SurvivalError::invalid_input(format!(
            "{name} must be rectangular"
        )));
    }
    Array2::from_shape_vec(
        (rows.len(), ncols),
        rows.iter().flatten().copied().collect(),
    )
    .map_err(|err| SurvivalError::invalid_input(err.to_string()))
}

fn crossprod(rows: &Array2<f64>) -> Array2<f64> {
    rows.t().dot(rows)
}

impl CoxPHFit {
    /// Fits the model: `coxph.fit` / `agreg.fit` / `coxexact.fit` /
    /// `agexact.fit` followed by the post-processing of `coxph()`.
    pub fn fit(data: CoxphData, options: CoxphOptions) -> SurvivalResult<Self> {
        let n = data.n();
        let nvar = data.x.ncols();
        let nevent = data.status.iter().filter(|&&s| s == 1).count();
        if data.entry.is_some() && nevent == 0 {
            return Err(SurvivalError::invalid_input(
                "Can't fit a Cox model with 0 failures",
            ));
        }
        if let Some(init) = &options.init {
            if init.len() != nvar {
                return Err(SurvivalError::invalid_input(
                    "Wrong length for inital values",
                ));
            }
            validate_finite(init, "init")?;
        }
        let nocenter: Vec<bool> = (0..nvar)
            .map(|col| {
                options.nocenter.as_ref().is_some_and(|values| {
                    data.x
                        .column(col)
                        .iter()
                        .all(|value| values.contains(value))
                })
            })
            .collect();
        let doscale = nocenter.iter().map(|&skip| !skip).collect();

        let mut engine = CoxFitBuilder::new(
            Array1::from_vec(data.time.clone()),
            Array1::from_vec(data.status.clone()),
            data.x.clone(),
        )
        .method(options.method)
        .max_iter(options.iter_max)
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
        if let Some(offset) = &data.offset {
            engine = engine.offset(Array1::from_vec(offset.clone()));
        }
        if let Some(weights) = &data.weights {
            engine = engine.weights(Array1::from_vec(weights.clone()));
        }
        let mut engine = engine.build()?;
        engine.fit();
        let results = engine.results();

        let offset = data.offset.unwrap_or_else(|| vec![0.0; n]);
        let weights = data.weights.unwrap_or_else(|| vec![1.0; n]);
        // The linear predictor uses the fitted values; only afterwards are
        // the aliased coefficients marked NA (`coxph.fit`: `coef[which.sing] <- NA`
        // unless iter.max = 0).
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
                    + offset[i]
                    - center
            })
            .collect();
        if options.iter_max > 0 && results.flag >= 0 && (results.flag as usize) < nvar {
            for (j, coefficient) in coefficients.iter_mut().enumerate() {
                if results.var[(j, j)] == 0.0 {
                    *coefficient = f64::NAN;
                }
            }
        }
        let sorted = SortedRows::new(results.order, data.strata.as_deref());
        // As in `coxph()`, the cluster enters the concordance whenever it was
        // given, even when the variance is not robust.
        let concordance = linear_predictor_concordance(
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
        fit.residuals = martingale_residuals(&fit, &fit.linear_predictors);

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
            let scores = score_residuals(&fit, &lp0)?;
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

    /// Per-stratum `agsurv` pieces at `x - means` and `risk =
    /// exp(linear_predictors - log_risk_shift)`, in the fit's stratum order.
    fn compute_curves(
        &self,
        survtype: CoxSurvType,
        vartype: CoxSurvType,
        log_risk_shift: f64,
    ) -> SurvivalResult<Vec<AgsurvCurve>> {
        let risk: Vec<f64> = self
            .linear_predictors
            .iter()
            .map(|lp| (lp - log_risk_shift).exp())
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
                agsurv_rows(&data, &self.sorted.order[start..end], survtype, vartype)
            })
            .collect()
    }

    /// The cached baseline curves for the fit's own hazard type.
    pub(crate) fn baseline_curves(&self) -> SurvivalResult<&[AgsurvCurve]> {
        if let Some(curves) = self.curves.get() {
            return Ok(curves);
        }
        let survtype = self.default_survtype();
        let curves = self.compute_curves(survtype, survtype, 0.0)?;
        Ok(self.curves.get_or_init(|| curves))
    }

    /// Relative risk of a centred covariate row: `exp(x2c %*% coef + offset2)`.
    fn relative_risk(&self, x2c: &[f64], offset2: f64) -> f64 {
        (x2c.iter()
            .zip(self.coefficients_or_zero())
            .map(|(x, b)| x * b)
            .sum::<f64>()
            + offset2)
            .exp()
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
    /// on the fit's scale.
    fn centered_newdata(&self, newdata: &CoxNewData) -> (Array2<f64>, Vec<f64>) {
        let mut x2c = newdata.x.clone();
        for (col, &mean) in self.means.iter().enumerate() {
            x2c.column_mut(col).mapv_inplace(|value| value - mean);
        }
        let risk2: Vec<f64> = (0..newdata.nrows())
            .map(|i| {
                let offset2 = newdata.offset.as_ref().map_or(0.0, |o| o[i]);
                self.relative_risk(&x2c.row(i).to_vec(), offset2)
            })
            .collect();
        (x2c, risk2)
    }

    /// `basehaz(fit, centered)`.
    pub fn basehaz(&self, centered: bool) -> SurvivalResult<Basehaz> {
        let curves = self.baseline_curves()?;
        // survfit(fit) evaluates the curve at x = means and the mean
        // offset; uncentred divides the offset sum(means * coef) back out.
        let mut scale = self.offset_mean().exp();
        if !centered {
            let center: f64 = self
                .means
                .iter()
                .zip(self.coefficients_or_zero())
                .map(|(m, b)| m * b)
                .sum();
            scale *= (-center).exp();
        }
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
        let offset_mean = self.offset_mean();
        // Kalbfleisch-Prentice needs the risks on survfit's scale
        // (relative to the mean offset); the other types only depend on
        // risk2 * baseline, so the cached predict-scale curves serve.
        let kp = survtype == CoxSurvType::KalbfleischPrentice;
        let shift = if kp { offset_mean } else { 0.0 };
        let computed;
        let curves: &[AgsurvCurve] = if !kp && survtype == self.default_survtype() {
            self.baseline_curves()?
        } else {
            computed = self.compute_curves(survtype, survtype, shift)?;
            &computed
        };
        let (x2c, mut risk2) = match newdata {
            Some(newdata) => self.centered_newdata(newdata),
            None => (Array2::zeros((1, self.nvar())), vec![offset_mean.exp()]),
        };
        for value in risk2.iter_mut() {
            *value *= (-shift).exp();
        }
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
        let kp = survtype == CoxSurvType::KalbfleischPrentice;
        let shift = if kp { self.offset_mean() } else { 0.0 };
        let computed;
        let curves: &[AgsurvCurve] = if !kp && survtype == self.default_survtype() {
            self.baseline_curves()?
        } else {
            computed = self.compute_curves(survtype, survtype, shift)?;
            &computed
        };
        let (x2c, mut risk2) = self.centered_newdata(newdata);
        for value in risk2.iter_mut() {
            *value *= (-shift).exp();
        }
        let varmat = options.se_fit.then_some(&self.var);
        let mut ids: Vec<i32> = Vec::new();
        for &value in id {
            if !ids.contains(&value) {
                ids.push(value);
            }
        }
        let mut result = Vec::new();
        for subject in ids {
            let intervals: Vec<IndividualInterval<'_>> = (0..newdata.nrows())
                .filter(|&i| id[i] == subject)
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
            let curve = individual_curve(curves, survtype, &intervals, varmat)?;
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

    /// The design rows `predict.coxph` uses for `lp`, `risk` and `terms`:
    /// centred per the reference, plus the centred offset.
    fn prediction_rows(
        &self,
        newdata: Option<&CoxNewData>,
        reference: PredictReference,
    ) -> SurvivalResult<(Array2<f64>, Vec<f64>)> {
        let offset_mean = self.offset.iter().sum::<f64>() / self.n as f64;
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
        let has_strata = self.strata.is_some();
        let use_x = newdata.is_some()
            || se_fit
            || (has_strata && reference == PredictReference::Strata)
            || (reference == PredictReference::Zero && self.means.iter().any(|&m| m != 0.0));
        if !use_x {
            return Ok(CoxPrediction {
                fit: self.linear_predictors.clone(),
                se_fit: None,
            });
        }
        let (newx, offset) = self.prediction_rows(newdata, reference)?;
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
        let (newx, _) = self.prediction_rows(newdata, reference)?;
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
            let risk: Vec<f64> = self.linear_predictors.iter().map(|lp| lp.exp()).collect();
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
                let varh = step_at(&curve.time, &integrated.cum_varhaz, t);
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
fn linear_predictor_concordance(
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
                &options,
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
                &options,
            )
        }
    }
}

/// Checks that `assign` groups existing columns.
pub(crate) fn validate_assign(assign: &[Vec<usize>], nvar: usize) -> SurvivalResult<()> {
    for (term, columns) in assign.iter().enumerate() {
        if columns.is_empty() {
            return Err(SurvivalError::invalid_input(format!(
                "assign[{term}] cannot be empty"
            )));
        }
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

impl CoxPHFit {
    fn residual_vector(
        &self,
        kind: ResidualType,
        weighted: bool,
        collapse: Option<&[i32]>,
    ) -> PyResult<Vec<f64>> {
        match self.residuals(kind, Some(weighted), collapse, None)? {
            Residuals::Vector(values) => Ok(values),
            Residuals::Matrix(_) => unreachable!("vector residual types"),
        }
    }

    fn residual_matrix(
        &self,
        kind: ResidualType,
        weighted: bool,
        collapse: Option<&[i32]>,
        assign: Option<&[Vec<usize>]>,
    ) -> PyResult<Vec<Vec<f64>>> {
        match self.residuals(kind, Some(weighted), collapse, assign)? {
            Residuals::Matrix(values) => Ok(matrix_rows(&values)),
            Residuals::Vector(_) => unreachable!("matrix residual types"),
        }
    }
}

fn newdata_from_python(
    fit: &CoxPHFit,
    x: Option<Vec<Vec<f64>>>,
    strata: Option<Vec<i32>>,
    offset: Option<Vec<f64>>,
    time: Option<Vec<f64>>,
    entry: Option<Vec<f64>>,
) -> SurvivalResult<Option<CoxNewData>> {
    let Some(x) = x else {
        if strata.is_some() || offset.is_some() || time.is_some() || entry.is_some() {
            return Err(SurvivalError::invalid_input(
                "new_strata, new_offset, new_time and new_entry require newdata",
            ));
        }
        return Ok(None);
    };
    let x = if x.is_empty() {
        Array2::zeros((0, fit.nvar()))
    } else {
        matrix_from_rows(&x, "newdata")?
    };
    Ok(Some(CoxNewData::try_new(x, strata, offset, time, entry)?))
}

#[pymethods]
impl CoxPHFit {
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
        r#type: &str,
        newdata: Option<Vec<Vec<f64>>>,
        new_strata: Option<Vec<i32>>,
        new_offset: Option<Vec<f64>>,
        new_time: Option<Vec<f64>>,
        new_entry: Option<Vec<f64>>,
        se_fit: bool,
        reference: &str,
    ) -> PyResult<CoxPrediction> {
        let newdata =
            newdata_from_python(self, newdata, new_strata, new_offset, new_time, new_entry)?;
        let reference = PredictReference::parse(reference)?;
        Ok(match r#type {
            "lp" => self.predict_lp(newdata.as_ref(), se_fit, reference)?,
            "risk" => self.predict_risk(newdata.as_ref(), se_fit, reference)?,
            "expected" => self.predict_expected(newdata.as_ref(), se_fit)?,
            "survival" => self.predict_survival(newdata.as_ref(), se_fit)?,
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
        newdata: Option<Vec<Vec<f64>>>,
        new_strata: Option<Vec<i32>>,
        new_offset: Option<Vec<f64>>,
        se_fit: bool,
        reference: &str,
        assign: Option<Vec<Vec<usize>>>,
    ) -> PyResult<CoxTermsPrediction> {
        let newdata = newdata_from_python(self, newdata, new_strata, new_offset, None, None)?;
        let reference = PredictReference::parse(reference)?;
        let assign = assign.unwrap_or_else(|| default_assign(self.nvar()));
        Ok(self.predict_terms(newdata.as_ref(), se_fit, reference, &assign)?)
    }

    /// `survfit(fit, newdata, stype, ctype, se.fit, censor)`.
    #[pyo3(name = "survfit", signature = (newdata = None, new_strata = None, new_offset = None, stype = 2, ctype = None, se_fit = true, censor = true))]
    #[allow(clippy::too_many_arguments)]
    fn survfit_py(
        &self,
        newdata: Option<Vec<Vec<f64>>>,
        new_strata: Option<Vec<i32>>,
        new_offset: Option<Vec<f64>>,
        stype: u8,
        ctype: Option<u8>,
        se_fit: bool,
        censor: bool,
    ) -> PyResult<Vec<CoxSurvfitCurve>> {
        let newdata = newdata_from_python(self, newdata, new_strata, new_offset, None, None)?;
        Ok(self.survfit(
            newdata.as_ref(),
            SurvfitOptions {
                stype,
                ctype,
                se_fit,
                censor,
            },
        )?)
    }

    /// `residuals(fit, type = "martingale", weighted, collapse)`.
    #[pyo3(name = "martingale_residuals", signature = (weighted = false, collapse = None))]
    fn martingale_residuals_py(
        &self,
        weighted: bool,
        collapse: Option<Vec<i32>>,
    ) -> PyResult<Vec<f64>> {
        self.residual_vector(ResidualType::Martingale, weighted, collapse.as_deref())
    }

    /// `residuals(fit, type = "deviance", weighted, collapse)`.
    #[pyo3(name = "deviance_residuals", signature = (weighted = false, collapse = None))]
    fn deviance_residuals_py(
        &self,
        weighted: bool,
        collapse: Option<Vec<i32>>,
    ) -> PyResult<Vec<f64>> {
        self.residual_vector(ResidualType::Deviance, weighted, collapse.as_deref())
    }

    /// `residuals(fit, type = "score", weighted, collapse)`.
    #[pyo3(name = "score_residuals", signature = (weighted = false, collapse = None))]
    fn score_residuals_py(
        &self,
        weighted: bool,
        collapse: Option<Vec<i32>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        self.residual_matrix(ResidualType::Score, weighted, collapse.as_deref(), None)
    }

    /// `residuals(fit, type = "dfbeta", weighted, collapse)`.
    #[pyo3(name = "dfbeta", signature = (weighted = true, collapse = None))]
    fn dfbeta_py(&self, weighted: bool, collapse: Option<Vec<i32>>) -> PyResult<Vec<Vec<f64>>> {
        self.residual_matrix(ResidualType::Dfbeta, weighted, collapse.as_deref(), None)
    }

    /// `residuals(fit, type = "dfbetas", weighted, collapse)`.
    #[pyo3(name = "dfbetas", signature = (weighted = true, collapse = None))]
    fn dfbetas_py(&self, weighted: bool, collapse: Option<Vec<i32>>) -> PyResult<Vec<Vec<f64>>> {
        self.residual_matrix(ResidualType::Dfbetas, weighted, collapse.as_deref(), None)
    }

    /// `residuals(fit, type = "schoenfeld", weighted)`.
    #[pyo3(name = "schoenfeld_residuals", signature = (weighted = false))]
    fn schoenfeld_residuals_py(&self, weighted: bool) -> PyResult<SchoenfeldResiduals> {
        Ok(schoenfeld_residuals(self, weighted)?)
    }

    /// `residuals(fit, type = "scaledsch", weighted)`.
    #[pyo3(name = "scaled_schoenfeld_residuals", signature = (weighted = false))]
    fn scaled_schoenfeld_residuals_py(&self, weighted: bool) -> PyResult<SchoenfeldResiduals> {
        Ok(self.scaled_schoenfeld_residuals(weighted)?)
    }

    /// `residuals(fit, type = "partial", weighted, collapse)`; `assign`
    /// lists the columns of each term (default: one term per column).
    #[pyo3(name = "partial_residuals", signature = (assign = None, weighted = false, collapse = None))]
    fn partial_residuals_py(
        &self,
        assign: Option<Vec<Vec<usize>>>,
        weighted: bool,
        collapse: Option<Vec<i32>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        self.residual_matrix(
            ResidualType::Partial,
            weighted,
            collapse.as_deref(),
            assign.as_deref(),
        )
    }

    /// `survfit(fit, newdata, id)` for time-dependent new data.
    #[pyo3(name = "survfit_individual", signature = (newdata, new_entry, new_time, id, new_strata = None, new_offset = None, stype = 2, ctype = None, se_fit = true, censor = true))]
    #[allow(clippy::too_many_arguments)]
    fn survfit_individual_py(
        &self,
        newdata: Vec<Vec<f64>>,
        new_entry: Vec<f64>,
        new_time: Vec<f64>,
        id: Vec<i32>,
        new_strata: Option<Vec<i32>>,
        new_offset: Option<Vec<f64>>,
        stype: u8,
        ctype: Option<u8>,
        se_fit: bool,
        censor: bool,
    ) -> PyResult<Vec<CoxSurvfitCurve>> {
        let newdata = newdata_from_python(
            self,
            Some(newdata),
            new_strata,
            new_offset,
            Some(new_time),
            Some(new_entry),
        )?
        .expect("newdata was supplied");
        Ok(self.survfit_individual(
            &newdata,
            &id,
            SurvfitOptions {
                stype,
                ctype,
                se_fit,
                censor,
            },
        )?)
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
    time: Vec<f64>,
    status: Vec<i32>,
    x: Vec<Vec<f64>>,
    entry: Option<Vec<f64>>,
    strata: Option<Vec<i32>>,
    weights: Option<Vec<f64>>,
    offset: Option<Vec<f64>>,
    method: &str,
    init: Option<Vec<f64>>,
    iter_max: Option<usize>,
    eps: Option<f64>,
    toler_chol: Option<f64>,
    nocenter: Option<Vec<f64>>,
    cluster: Option<Vec<i32>>,
    robust: Option<bool>,
) -> PyResult<CoxPHFit> {
    if x.len() != time.len() {
        return Err(SurvivalError::invalid_input(format!(
            "x has {} rows but time has {}",
            x.len(),
            time.len()
        ))
        .into());
    }
    let x = matrix_from_rows(&x, "x")?;
    let data = CoxphData::try_new(time, entry, status, x, weights, strata, offset)?;
    let defaults = CoxphOptions::default();
    let options = CoxphOptions {
        method: TieMethod::parse(Some(method))?,
        init,
        iter_max: iter_max.unwrap_or(defaults.iter_max),
        eps: eps.unwrap_or(defaults.eps),
        toler_chol: toler_chol.unwrap_or(defaults.toler_chol),
        nocenter: nocenter.or(defaults.nocenter),
        cluster,
        robust,
    };
    Ok(CoxPHFit::fit(data, options)?)
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
        assert!(
            CoxphData::try_new(
                vec![1.0, 2.0],
                None,
                vec![1, 0],
                Array2::zeros((2, 1)),
                Some(vec![1.0, 0.0]),
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
