//! Optimizer setup and post-processing of fitted Cox parameters.

use super::{CoxPHFit, CoxphData, CoxphOptions, SortedRows};
use crate::concordance::{ConcordanceCounts, ConcordanceFit, ConcordanceOptions, concordancefit};
use crate::core::SurvResponse;
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::typed_inputs::{CountingProcessData, SurvivalData};
use crate::internal::validation::validate_finite;
use crate::regression::cox_optimizer::{CoxFitBuilder, CoxFitResults, TieMethod};
use crate::regression::coxph_diagnostics::{
    collapse_rows, martingale_residuals_at, score_residuals_at,
};
use crate::regression::coxph_wtest::{wald_statistic, wald_tests};
use ndarray::{Array2, ArrayView2};
use std::sync::OnceLock;

pub(super) fn crossprod(rows: &Array2<f64>) -> Array2<f64> {
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

    let mut engine =
        CoxFitBuilder::new(data.time.as_slice(), data.status.as_slice(), data.x.view())
            .method(options.method)
            .max_iter(options.iter_max)
            .iterate_empty(iterate_empty)
            .eps(options.eps)
            .toler(options.toler_chol)
            .doscale(doscale)
            .initial_beta(options.init.clone().unwrap_or_else(|| vec![0.0; nvar]));
    if let Some(entry) = &data.entry {
        engine = engine.entry_times(entry.as_slice());
    }
    if let Some(strata) = &data.strata {
        engine = engine.strata(strata.as_slice());
    }
    if let Some(offset) = offset {
        engine = engine.offset(offset);
    }
    if let Some(weights) = &data.weights {
        engine = engine.weights(weights.as_slice());
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
        data.validate()?;
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
