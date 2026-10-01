//! Penalised parametric models: R survival's `survpenal.fit`
//! (`R/survpenal.fit.R`), the fitter behind `survreg()` when the formula has
//! `ridge()` or `pspline()` terms, with the `survreg()` post-processing of
//! `R/survreg.R` (the scale, `logcorrect`, `df.residual`, the robust
//! variance), producing the `survreg.penal` object.
//!
//! The penalty terms, their `cfun`s, `coxpenal.df` and the outer loop are
//! those of `coxpenal.fit` ([`crate::regression::penalized`]); [`kernel`] is
//! the inner Newton iteration, `survreg7.c`.  The outer loop fits the
//! intercept-only model for the starting scale and `n.eff`, removes a sparse
//! frailty column from the design, iterates `survreg7` and the `cfun`s over
//! `theta`, restarting each inner fit from the solution of the closest
//! earlier `theta`, and computes the degrees of freedom and variances.
//!
//! The fitted model is also kept as a [`SurvregFit`], so that `predict()`
//! and `residuals()` of a `survreg.penal` object come from the same code as
//! for an unpenalised one (R's `NextMethod()`).

#[cfg(test)]
mod density_tests;
mod kernel;
mod lowlevel;
pub use lowlevel::{SurvpenalFitResult, survpenal_fit_raw};
#[cfg(test)]
mod tests;

use self::kernel::{Survreg7Fit, survreg7};
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::matrix::{lu_inverse, matrix_rows};
use crate::internal::numpy_utils::{FloatMatrix, FloatVec};
use crate::internal::validation::validate_finite;
use crate::regression::coxpenal::CoxPenalty;
use crate::regression::parametric_survival::{
    FittingResponse, SurvregControl, SurvregData, SurvregFit, fitting_response, intercept_only_fit,
    mark_singular, robust_variance,
};
use crate::regression::penalized::df::{DfInput, TermDf, coxpenal_df};
use crate::regression::penalized::terms::{
    Composer, ModelTerm, PenaltyHistory, PenaltyShape, build_term_states, closest_saved,
    drop_sparse_column, histories, model_terms, penalty_shape, update_controls, validate_terms,
};
use crate::regression::survreg_distributions::{SurvregDistribution, SurvregFamily};
use crate::regression::survreg_predict::SurvregPrediction;
use crate::regression::survregc1::{SparseFrailty, SurvregKernel};
use crate::residuals::survreg_resid::SurvregResiduals;
use ndarray::Array2;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Validated inputs of a penalised parametric fit.  `survreg.covariates`
/// holds every design column, including the single column of group codes
/// of a sparse frailty term.
#[derive(Debug, Clone)]
pub struct SurvpenalData {
    pub survreg: SurvregData,
    /// The model terms in formula order; every column belongs to exactly
    /// one term and at least one term is penalised.
    pub terms: Vec<ModelTerm>,
}

impl SurvpenalData {
    pub fn try_new(survreg: SurvregData, terms: Vec<ModelTerm>) -> SurvivalResult<Self> {
        survreg.validate()?;
        validate_terms(survreg.nvar(), &terms)?;
        Ok(Self { survreg, terms })
    }
}

/// `survreg()`'s `init`, `scale`, `control` and `robust`.
#[derive(Debug, Clone, Default)]
pub struct SurvpenalOptions {
    /// Starting values: the `nvar` dense coefficients (completed with the
    /// intercept-only fit's `log(scale)`s), those plus the `log(scale)`s,
    /// or all of them after the frailties.
    pub init: Option<Vec<f64>>,
    /// A fixed scale (`0` estimates it).
    pub scale: f64,
    pub control: SurvregControl,
    /// The robust (dfbeta sandwich) variance; implied by a cluster.
    pub robust: bool,
}

/// A fitted penalised parametric model (R's `survreg.penal` object).
#[pyclass(module = "survival._survival", skip_from_py_object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurvpenalFit {
    /// The fit as a `survreg` object: coefficients (`NaN` where aliased,
    /// then the `log(scale)`s), `var`, the log likelihoods, the linear
    /// predictors, `sum(df)` and `n - sum(df)`.
    pub survreg: SurvregFit,
    /// `H^{-1} I H^{-1}` for the dense coefficients and scales.
    pub var2: Array2<f64>,
    /// Outer iterations and total inner iterations.
    #[pyo3(get)]
    pub iter: [usize; 2],
    /// Outer iterations whose inner loop did not converge (with
    /// `iter.max > 1`); R means to warn about them but never does.
    #[pyo3(get)]
    pub inner_failures: Vec<usize>,
    /// Effective degrees of freedom of each term of `assign2`.
    #[pyo3(get)]
    pub df: Vec<f64>,
    /// `c(0, P)`: the penalty of the final fit.
    #[pyo3(get)]
    pub penalty: [f64; 2],
    /// Per model term: 0 ordinary, 1 penalised, 2 sparse.
    pub pterms: Vec<u8>,
    /// R's `assign2`: the columns of each term in the dense design (a
    /// sparse term keeps its original column), then the `log(scale)`s when
    /// they are estimated (R's `sigma` term).
    #[pyo3(get)]
    pub assign2: Vec<Vec<usize>>,
    #[pyo3(get)]
    pub history: Vec<PenaltyHistory>,
    /// The fitted frailties of a sparse term (`frail`), the diagonal of
    /// their variance (`fvar`) and the 0-based group of each observation.
    #[pyo3(get)]
    pub frail: Option<Vec<f64>>,
    #[pyo3(get)]
    pub fvar: Option<Vec<f64>>,
    #[pyo3(get)]
    pub frail_index: Option<Vec<usize>>,
    /// `n.eff`, the effective sample size the `aic` searches use.
    #[pyo3(get)]
    pub n_eff: f64,
    /// `fit$u`: the penalised score, frailties first.
    #[pyo3(get)]
    pub score: Vec<f64>,
}

/// R's `sd$variance(temp^2)` for `n.eff`: the variance of the standard
/// distribution, where the `t` family's `variance(df)` receives the squared
/// mean scale as its degrees of freedom (survpenal.fit.R:402, kept).
fn neff_variance(distribution: &SurvregDistribution, scale2: f64) -> SurvivalResult<f64> {
    match distribution.family {
        SurvregFamily::T => Ok(scale2 / (scale2 - 2.0)),
        SurvregFamily::Custom => {
            let value = distribution
                .custom_callbacks()?
                .fitting_variance(scale2, &distribution.parms)?;
            if !value.is_finite() {
                return Err(SurvivalError::invalid_input(
                    "fitting variance must be finite",
                ));
            }
            Ok(value)
        }
        _ => distribution.variance(),
    }
}

/// The inputs of `coxpenal.df` for an inner fit: the penalty of the dense
/// terms padded with zero rows and columns for the `log(scale)`s
/// (survpenal.fit.R:480-494).
fn term_df(
    fit: &Survreg7Fit,
    hdiag: &[f64],
    assign2: &[Vec<usize>],
    shape: PenaltyShape,
    composer: &Composer<'_>,
    sparse_term: Option<usize>,
) -> SurvivalResult<TermDf> {
    let nvar2 = fit.hmat.nrows();
    let second = &composer.coxlist2.second;
    let pen2 = if shape.full_imat {
        let nvar = composer.coxlist2.coef.len();
        let mut pen2 = vec![0.0; nvar2 * nvar2];
        for col in 0..nvar {
            pen2[col * nvar2..col * nvar2 + nvar].copy_from_slice(&second[col * nvar..][..nvar]);
        }
        pen2
    } else {
        let mut pen2 = second.clone();
        pen2.resize(nvar2, 0.0);
        pen2
    };
    coxpenal_df(DfInput {
        hmat: &fit.hmat,
        hinv: &fit.hinv,
        fdiag: hdiag,
        assign: assign2,
        shape,
        pen1: &composer.coxlist1.second,
        pen2: &pen2,
        sparse_term,
    })
}

struct SurvpenalEngineResult {
    fit: SurvpenalFitResult,
    response: FittingResponse,
    covariates: Array2<f64>,
    strata: Vec<usize>,
    offset: Vec<f64>,
    frail_index: Option<Vec<usize>>,
}

impl SurvpenalFit {
    /// Full model construction after the shared penalized fitting engine.
    pub fn fit(
        data: &SurvpenalData,
        distribution: &SurvregDistribution,
        options: &SurvpenalOptions,
    ) -> SurvivalResult<Self> {
        Self::fit_inputs(&data.survreg, &data.terms, distribution, options)
    }

    fn fit_inputs(
        data: &SurvregData,
        terms: &[ModelTerm],
        distribution: &SurvregDistribution,
        options: &SurvpenalOptions,
    ) -> SurvivalResult<Self> {
        let result = Self::fit_engine(data, terms, distribution, options, None)?;
        let raw = result.fit;
        let nfrail = raw.frail.as_ref().map_or(0, Vec::len);
        let df_total: f64 = raw.df.iter().sum();
        let means = result
            .covariates
            .columns()
            .into_iter()
            .map(|column| column.sum() / raw.n as f64)
            .collect();
        let mut survreg = SurvregFit {
            coefficients: raw.coefficients,
            icoef: raw.icoef,
            variance_matrix: matrix_rows(&raw.var),
            naive_variance_matrix: None,
            log_likelihood: raw.loglik[1],
            intercept_only_log_likelihood: raw.loglik[0],
            iterations: raw.iter[1],
            converged: raw.converged,
            linear_predictors: raw.linear_predictors,
            scale: raw.scale,
            df: df_total,
            df_residual: raw.n as f64 - df_total,
            means,
            n: raw.n,
            distribution: distribution.clone(),
            time: result.response.time,
            time2: result.response.time2,
            status: result.response.status,
            covariates: result.covariates,
            strata: result.strata,
            weights: data.weights.clone(),
            offset: result.offset,
            cluster: data.cluster.clone(),
            score: raw.score[nfrail..].to_vec(),
        };
        if options.robust || data.cluster.is_some() {
            if raw.frail.is_some() {
                return Err(SurvivalError::invalid_input(
                    "robust variance is not available with a sparse frailty term",
                ));
            }
            let sandwich = robust_variance(&survreg, data.cluster.as_deref())?;
            survreg.naive_variance_matrix =
                Some(std::mem::replace(&mut survreg.variance_matrix, sandwich));
        }
        mark_singular(&mut survreg);
        Ok(Self {
            survreg,
            var2: raw.var2,
            iter: raw.iter,
            inner_failures: raw.inner_failures,
            df: raw.df,
            penalty: raw.penalty,
            pterms: raw.pterms,
            assign2: raw.assign2,
            history: raw.history,
            frail: raw.frail,
            fvar: raw.fvar,
            frail_index: result.frail_index,
            n_eff: raw.n_eff,
            score: raw.score,
        })
    }

    fn fit_engine(
        survreg_data: &SurvregData,
        model_terms: &[ModelTerm],
        distribution: &SurvregDistribution,
        options: &SurvpenalOptions,
        nstrata: Option<usize>,
    ) -> SurvivalResult<SurvpenalEngineResult> {
        survreg_data.validate()?;
        validate_terms(survreg_data.nvar(), model_terms)?;
        let control = &options.control;
        distribution.validate()?;
        control.validate()?;
        if control.outer_max == 0 {
            return Err(SurvivalError::invalid_input("invalid value for outer.max"));
        }
        let n = survreg_data.n();
        let nstrata = survreg_data.fitting_strata(nstrata)?;
        let eps = control.rel_tolerance;
        let tol_chol = control.toler_chol;
        let scale = distribution.scale.unwrap_or(options.scale);
        if !scale.is_finite() || scale < 0.0 {
            return Err(SurvivalError::invalid_input("Invalid scale value"));
        }
        if scale > 0.0 && nstrata > 1 {
            return Err(SurvivalError::invalid_input(
                "The scale argument is not valid with multiple strata",
            ));
        }
        let weights = survreg_data.weights.clone().unwrap_or_else(|| vec![1.0; n]);
        let offset = survreg_data.offset.clone().unwrap_or_else(|| vec![0.0; n]);
        let strata = survreg_data.strata.clone().unwrap_or_else(|| vec![0; n]);
        let response = fitting_response(survreg_data, distribution, &weights)?;
        // The number of scales to estimate.
        let nstrat2 = if scale > 0.0 { 0 } else { nstrata };

        let (pterms, sparse_term, shape) = penalty_shape(model_terms);
        // Remove the sparse term's column from the design.
        let (xx, mut assign2, frailx, nfrail) =
            drop_sparse_column(survreg_data.design().view(), model_terms);
        let nvar = xx.ncols();
        let nvar2 = nvar + nstrat2;
        let nvar3 = nvar2 + nfrail;
        if nvar2 == 0 {
            return Err(SurvivalError::invalid_input(
                "Cannot fit a model with no coefficients other than sparse ones",
            ));
        }
        let eps2 = eps.sqrt();
        let terms = build_term_states(
            model_terms,
            &assign2,
            &xx,
            frailx.as_deref(),
            nfrail,
            n,
            &response.status,
            eps2,
        )?;
        let need_df = terms.iter().any(|term| term.needs_df());
        let mut composer = Composer::new(terms, nfrail, nvar, shape.full_imat, true);

        // The intercept-only fit gives the starting scale, the first
        // log likelihood and the "effective n".
        let fit0 = intercept_only_fit(
            distribution,
            &response,
            &weights,
            &offset,
            &strata,
            nstrata,
            nstrat2,
            scale,
            eps,
            tol_chol,
        )?;
        let mean_scale =
            fit0.beta[1..].iter().map(|v| v.exp()).sum::<f64>() / (fit0.beta.len() - 1) as f64;
        let n_eff =
            neff_variance(distribution, mean_scale * mean_scale)? * lu_inverse(&fit0.var)?[(0, 0)];
        composer.neff = n_eff;

        // Starting values: frailties, dense coefficients, log(scale)s.
        let mut init = match &options.init {
            Some(init) => {
                validate_finite(init, "init")?;
                let mut start = vec![0.0; nfrail];
                if init.len() == nvar && nstrat2 > 0 {
                    start.extend_from_slice(init);
                    start.extend_from_slice(&fit0.beta[1..]);
                } else if init.len() == nvar2 {
                    start.extend_from_slice(init);
                } else if init.len() == nvar3 {
                    start = init.clone();
                } else {
                    return Err(SurvivalError::invalid_input(
                        "Wrong length for inital values",
                    ));
                }
                if scale > 0.0 {
                    start.push(scale.ln());
                }
                start
            }
            None => {
                // fit0's intercept goes on the first dense column.
                let mut start = vec![0.0; nfrail + nvar];
                if nvar > 0 {
                    start[nfrail] = fit0.beta[0];
                }
                start.extend_from_slice(&fit0.beta[1..]);
                start
            }
        };
        // The scales form the last term of assign2, so that df covers them.
        if nstrat2 > 0 {
            assign2.push((nvar..nvar2).collect());
        }

        let kernel = SurvregKernel {
            y1: &response.y1,
            y2: &response.y2,
            status: &response.status,
            covariates: xx.view(),
            weights: &weights,
            offset: &offset,
            strata: &strata,
            nstrat: nstrat2,
            distribution,
        };
        let frailty = frailx
            .as_deref()
            .map(|group| SparseFrailty { group, nf: nfrail });

        // The outer loop over theta.
        let mut iter = 0;
        let mut iter2 = 0;
        let mut inner_failures = Vec::new();
        let mut theta_save: Vec<Vec<f64>> = Vec::new();
        let mut coef_save: Vec<Vec<f64>> = Vec::new();
        let mut last: Option<(Survreg7Fit, Vec<f64>)> = None;
        let mut dftemp: Option<TermDf> = None;
        for outer in 1..=control.outer_max {
            let thetas: Vec<f64> = composer.terms.iter().map(|term| term.state.theta).collect();
            let fit = survreg7(
                &kernel,
                frailty.as_ref(),
                control.iter_max,
                init.clone(),
                eps,
                tol_chol,
                shape,
                &mut composer,
            )?;
            iter = outer;
            iter2 += fit.iter;
            // survreg7.c:453 means to flag an inner fit that ran out of
            // iterations (or failed) when iter.max > 1.
            if !fit.converged && control.iter_max > 1 {
                inner_failures.push(outer);
            }
            theta_save.push(thetas);
            coef_save.push(fit.beta.clone());
            // An infinite penalty made the C code set hdiag = 1 in
            // self-defence; those coefficients are zero and so is their
            // variance.
            let mut hdiag = fit.hdiag.clone();
            if nfrail > 0 && composer.coxlist1.flag[0] {
                hdiag[..nfrail].fill(0.0);
            }
            if shape.dense {
                for i in 0..nvar {
                    if composer.coxlist2.flag[i] {
                        hdiag[nfrail + i] = 0.0;
                    }
                }
            }
            if need_df {
                dftemp = Some(term_df(
                    &fit,
                    &hdiag,
                    &assign2,
                    shape,
                    &composer,
                    sparse_term,
                )?);
            }
            let done = update_controls(
                &mut composer.terms,
                outer,
                fit.loglik - fit.penalty,
                fit.loglik,
                n_eff,
                dftemp.as_ref(),
                &fit.beta[nfrail..],
                &fit.beta[..nfrail],
            )?;
            last = Some((fit, hdiag));
            if done {
                break;
            }
            // Starting values for the next iteration: the solution of the
            // closest earlier theta.
            let next: Vec<f64> = composer.terms.iter().map(|term| term.state.theta).collect();
            init.clone_from(&coef_save[closest_saved(&theta_save, &next)]);
        }
        let (fit, hdiag) = last.expect("outer.max >= 1");
        let dftemp = match dftemp {
            Some(df) => df,
            None => term_df(&fit, &hdiag, &assign2, shape, &composer, sparse_term)?,
        };

        // The coefficients (NaN where aliased), the linear predictors
        // (aliased columns count 0) and the scales.
        let mut coefficients = fit.beta[nfrail..nfrail + nvar2].to_vec();
        for i in 0..nvar {
            if hdiag[nfrail + i] == 0.0 {
                coefficients[i] = f64::NAN;
            }
        }
        let frail = frailx.is_some().then(|| fit.beta[..nfrail].to_vec());
        let linear_predictors: Vec<f64> = (0..n)
            .map(|i| {
                let mut lp: f64 = xx
                    .row(i)
                    .iter()
                    .zip(&coefficients[..nvar])
                    .filter(|(_, b)| !b.is_nan())
                    .map(|(x, b)| x * b)
                    .sum();
                lp += offset[i];
                if let (Some(frail), Some(frailx)) = (&frail, &frailx) {
                    lp += frail[frailx[i]];
                }
                lp
            })
            .collect();
        let scales: Vec<f64> = if scale > 0.0 {
            vec![scale]
        } else {
            coefficients[nvar..].iter().map(|v| v.exp()).collect()
        };
        Ok(SurvpenalEngineResult {
            fit: SurvpenalFitResult {
                coefficients,
                icoef: fit0.beta,
                var: dftemp.var,
                var2: dftemp.var2,
                loglik: [
                    fit0.loglik + response.logcorrect,
                    fit.loglik - fit.penalty + response.logcorrect,
                ],
                iter: [iter, iter2],
                inner_failures,
                converged: fit.converged,
                linear_predictors,
                df: dftemp.df,
                penalty: [0.0, -fit.penalty],
                pterms,
                assign2,
                history: histories(&composer.terms),
                frail,
                fvar: dftemp.fvar,
                n_eff,
                score: fit.u,
                n,
                nvar,
                scale: scales,
            },
            response,
            covariates: xx,
            strata,
            offset,
            frail_index: frailx,
        })
    }

    /// `df.residual`: `n - sum(df)`.
    pub fn df_residual(&self) -> f64 {
        self.survreg.df_residual
    }

    /// `predict.survreg.penal` and `residuals.survreg.penal` refuse a sparse
    /// frailty model with `message`.
    fn check_not_sparse(&self, message: &str) -> SurvivalResult<()> {
        if self.frail.is_some() {
            return Err(SurvivalError::invalid_input(message.to_string()));
        }
        Ok(())
    }
}

#[pymethods]
impl SurvpenalFit {
    /// Pickle metadata and, for custom distributions, their callable state.
    #[cfg(feature = "python")]
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        let distribution = &self.survreg.distribution;
        let custom = distribution.callbacks.is_some() || distribution.transform_callbacks.is_some();
        let mut metadata;
        let state_value = if custom {
            metadata = self.clone();
            metadata.survreg.distribution = distribution.detached_callbacks();
            &metadata
        } else {
            self
        };
        let state = bincode::serde::encode_to_vec(state_value, bincode::config::standard())
            .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
        let rebuild = py
            .import("survival._survival")?
            .getattr("_survpenal_fit_from_state")?;
        (
            rebuild,
            (
                pyo3::types::PyBytes::new(py, &state),
                custom.then(|| distribution.clone()),
            ),
        )
            .into_pyobject(py)
    }

    /// The fit as a `SurvregFit` (a copy: a formula-level fit holds this
    /// and its own `SurvregModelResult.fit`).
    #[getter(survreg)]
    fn survreg_getter(&self) -> SurvregFit {
        self.survreg.clone()
    }

    #[getter]
    fn coefficients(&self) -> Vec<f64> {
        self.survreg.coefficients.clone()
    }

    #[getter]
    fn var(&self) -> Vec<Vec<f64>> {
        self.survreg.variance_matrix.clone()
    }

    #[getter(var2)]
    fn var2_getter(&self) -> Vec<Vec<f64>> {
        matrix_rows(&self.var2)
    }

    #[getter]
    fn loglik(&self) -> [f64; 2] {
        [
            self.survreg.intercept_only_log_likelihood,
            self.survreg.log_likelihood,
        ]
    }

    #[getter(df_residual)]
    fn df_residual_getter(&self) -> f64 {
        self.df_residual()
    }

    #[getter(pterms)]
    fn pterms_getter(&self) -> Vec<usize> {
        self.pterms.iter().map(|&p| usize::from(p)).collect()
    }

    #[getter]
    fn scale(&self) -> Vec<f64> {
        self.survreg.scale.clone()
    }

    /// `1 + nstrata` when the scales were estimated, else 1.
    #[getter]
    fn idf(&self) -> usize {
        1 + self.survreg.coefficients.len() - self.survreg.nvar()
    }

    #[getter]
    fn icoef(&self) -> Vec<f64> {
        self.survreg.icoef.clone()
    }

    #[getter]
    fn linear_predictors(&self) -> Vec<f64> {
        self.survreg.linear_predictors.clone()
    }

    /// `predict(object, ...)` as for a `SurvregFit`; R refuses a sparse
    /// frailty model.
    #[pyo3(name = "predict")]
    #[pyo3(signature = (newdata=None, predict_type="response", se_fit=false, p=None, offset=None, strata=None, assign=None, terms=None))]
    #[allow(clippy::too_many_arguments)]
    fn predict_py(
        &self,
        py: Python<'_>,
        newdata: Option<FloatMatrix>,
        predict_type: &str,
        se_fit: bool,
        p: Option<FloatVec>,
        offset: Option<FloatVec>,
        strata: Option<Vec<usize>>,
        assign: Option<Vec<usize>>,
        terms: Option<Vec<usize>>,
    ) -> PyResult<SurvregPrediction> {
        self.check_not_sparse("Predictions not available for sparse models")?;
        self.survreg.predict_py(
            py,
            newdata,
            predict_type,
            se_fit,
            p,
            offset,
            strata,
            assign,
            terms,
        )
    }

    /// `residuals(object, ...)` as for a `SurvregFit`; R refuses a sparse
    /// frailty model (its message, typo included).
    #[pyo3(name = "residuals")]
    #[pyo3(signature = (residual_type="response", rsigma=true, collapse=None, weighted=false))]
    fn residuals_py(
        &self,
        py: Python<'_>,
        residual_type: &str,
        rsigma: bool,
        collapse: Option<Vec<usize>>,
        weighted: bool,
    ) -> PyResult<SurvregResiduals> {
        self.check_not_sparse("Residualss not available for sparse models")?;
        self.survreg
            .residuals_py(py, residual_type, rsigma, collapse, weighted)
    }

    fn __repr__(&self) -> String {
        format!(
            "SurvpenalFit(distribution='{}', n={}, iter={:?}, df={:?}, loglik={:.4})",
            self.survreg.distribution.name,
            self.survreg.n,
            self.iter,
            self.df,
            self.survreg.log_likelihood
        )
    }
}

/// `survreg()`'s penalty branch on explicit data (survreg.R:203-293 with
/// `survpenal.fit`): `penalties[i]` applies to the columns `pcols[i]` of
/// `data.covariates`; `assign` lists the columns of every model term (each
/// `pcols` entry must be one of them) and defaults to the penalised groups
/// plus one term per remaining column, in column order.  A sparse frailty
/// term is a single column of group codes.  The response is untransformed,
/// as for `survreg_fit`.
#[pyfunction]
#[pyo3(signature = (data, distribution, penalties, pcols, assign=None, init=None, scale=0.0, control=None, robust=None))]
#[allow(clippy::too_many_arguments)]
pub fn survpenal_fit(
    py: Python<'_>,
    data: &SurvregData,
    distribution: &SurvregDistribution,
    penalties: Vec<CoxPenalty>,
    pcols: Vec<Vec<usize>>,
    assign: Option<Vec<Vec<usize>>>,
    init: Option<Vec<f64>>,
    scale: f64,
    control: Option<SurvregControl>,
    robust: Option<bool>,
) -> PyResult<SurvpenalFit> {
    let options = SurvpenalOptions {
        init,
        scale,
        control: control.unwrap_or_default(),
        robust: robust.unwrap_or(false),
    };
    Ok(py.detach(|| {
        let terms = model_terms(
            data.nvar(),
            penalties.into_iter().map(|penalty| penalty.term).collect(),
            pcols,
            assign,
        )?;
        SurvpenalFit::fit_inputs(data, &terms, distribution, &options)
    })?)
}

/// Rebuilds a pickled [`SurvpenalFit`] from its `__reduce__` state.
#[cfg(feature = "python")]
#[pyfunction(name = "_survpenal_fit_from_state")]
#[pyo3(signature = (state, distribution=None))]
pub fn survpenal_fit_from_state(
    py: Python<'_>,
    state: &[u8],
    distribution: Option<&SurvregDistribution>,
) -> PyResult<SurvpenalFit> {
    let mut result: SurvpenalFit = crate::internal::pickle::decode(py, state)?;
    if let Some(distribution) = distribution {
        distribution.validate()?;
        result.survreg.distribution = distribution.clone();
    }
    Ok(result)
}
