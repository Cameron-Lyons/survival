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

#[derive(Debug, Clone)]
pub struct SurvpenalData {
    pub survreg: SurvregData,
    pub terms: Vec<ModelTerm>,
}

impl SurvpenalData {
    pub fn try_new(survreg: SurvregData, terms: Vec<ModelTerm>) -> SurvivalResult<Self> {
        survreg.validate()?;
        validate_terms(survreg.nvar(), &terms)?;
        Ok(Self { survreg, terms })
    }
}

#[derive(Debug, Clone, Default)]
pub struct SurvpenalOptions {
    pub init: Option<Vec<f64>>,
    pub scale: f64,
    pub control: SurvregControl,
    pub robust: bool,
}

#[doc = " A fitted penalised parametric model (R's `survreg.penal` object)."]
#[pyclass(module = "survival._survival", skip_from_py_object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurvpenalFit {
    pub survreg: SurvregFit,
    pub var2: Array2<f64>,
    #[doc = " Outer iterations and total inner iterations."]
    #[pyo3(get)]
    pub iter: [usize; 2],
    #[doc = " Outer iterations whose inner loop did not converge (with"]
    #[doc = " `iter.max > 1`); R means to warn about them but never does."]
    #[pyo3(get)]
    pub inner_failures: Vec<usize>,
    #[doc = " Effective degrees of freedom of each term of `assign2`."]
    #[pyo3(get)]
    pub df: Vec<f64>,
    #[doc = " `c(0, P)`: the penalty of the final fit."]
    #[pyo3(get)]
    pub penalty: [f64; 2],
    pub pterms: Vec<u8>,
    #[doc = " R's `assign2`: the columns of each term in the dense design (a"]
    #[doc = " sparse term keeps its original column), then the `log(scale)`s when"]
    #[doc = " they are estimated (R's `sigma` term)."]
    #[pyo3(get)]
    pub assign2: Vec<Vec<usize>>,
    #[pyo3(get)]
    pub history: Vec<PenaltyHistory>,
    #[doc = " The fitted frailties of a sparse term (`frail`), the diagonal of"]
    #[doc = " their variance (`fvar`) and the 0-based group of each observation."]
    #[pyo3(get)]
    pub frail: Option<Vec<f64>>,
    #[pyo3(get)]
    pub fvar: Option<Vec<f64>>,
    #[pyo3(get)]
    pub frail_index: Option<Vec<usize>>,
    #[doc = " `n.eff`, the effective sample size the `aic` searches use."]
    #[pyo3(get)]
    pub n_eff: f64,
    #[doc = " `fit$u`: the penalised score, frailties first."]
    #[pyo3(get)]
    pub score: Vec<f64>,
}

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
    pub fn fit(
        data: &SurvpenalData,
        distribution: &SurvregDistribution,
        options: &SurvpenalOptions,
    ) -> SurvivalResult<Self> {
        Self::fit_with_nstrata(data, distribution, options, None)
    }

    pub fn fit_with_nstrata(
        data: &SurvpenalData,
        distribution: &SurvregDistribution,
        options: &SurvpenalOptions,
        nstrata: Option<usize>,
    ) -> SurvivalResult<Self> {
        Self::fit_inputs(&data.survreg, &data.terms, distribution, options, nstrata)
    }

    fn fit_inputs(
        data: &SurvregData,
        terms: &[ModelTerm],
        distribution: &SurvregDistribution,
        options: &SurvpenalOptions,
        nstrata: Option<usize>,
    ) -> SurvivalResult<Self> {
        let result = Self::fit_engine(data, terms, distribution, options, nstrata)?;
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
        let nstrat2 = if scale > 0.0 { 0 } else { nstrata };

        let (pterms, sparse_term, shape) = penalty_shape(model_terms);
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
        let mut observed = vec![false; nstrata];
        for &stratum in &strata {
            observed[stratum] = true;
        }
        let scales: Vec<usize> = if nstrat2 == 0 {
            vec![1]
        } else {
            (0..nstrata)
                .filter(|&s| observed[s])
                .map(|s| s + 1)
                .collect()
        };
        let mean_scale =
            scales.iter().map(|&s| fit0.beta[s].exp()).sum::<f64>() / scales.len() as f64;
        let information = if nstrat2 > 0 && scales.len() < nstrata {
            let keep: Vec<usize> = std::iter::once(0).chain(scales).collect();
            let covariance = Array2::from_shape_fn((keep.len(), keep.len()), |(i, j)| {
                fit0.var[(keep[i], keep[j])]
            });
            lu_inverse(&covariance)?[(0, 0)]
        } else {
            lu_inverse(&fit0.var)?[(0, 0)]
        };
        let n_eff = neff_variance(distribution, mean_scale * mean_scale)? * information;
        composer.neff = n_eff;

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
                let mut start = vec![0.0; nfrail + nvar];
                if nvar > 0 {
                    start[nfrail] = fit0.beta[0];
                }
                start.extend_from_slice(&fit0.beta[1..]);
                start
            }
        };
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
            if !fit.converged && control.iter_max > 1 {
                inner_failures.push(outer);
            }
            theta_save.push(thetas);
            coef_save.push(fit.beta.clone());
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
            let next: Vec<f64> = composer.terms.iter().map(|term| term.state.theta).collect();
            init.clone_from(&coef_save[closest_saved(&theta_save, &next)]);
        }
        let (fit, hdiag) = last.expect("outer.max >= 1");
        let dftemp = match dftemp {
            Some(df) => df,
            None => term_df(&fit, &hdiag, &assign2, shape, &composer, sparse_term)?,
        };

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

    pub fn df_residual(&self) -> f64 {
        self.survreg.df_residual
    }

    fn check_not_sparse(&self, message: &str) -> SurvivalResult<()> {
        if self.frail.is_some() {
            return Err(SurvivalError::invalid_input(message.to_string()));
        }
        Ok(())
    }
}

#[pymethods]
impl SurvpenalFit {
    #[doc = " Pickle metadata and, for custom distributions, their callable state."]
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

    #[doc = " The fit as a `SurvregFit` (a copy: a formula-level fit holds this"]
    #[doc = " and its own `SurvregModelResult.fit`)."]
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

    #[doc = " `1 + nstrata` when the scales were estimated, else 1."]
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

    #[doc = " `predict(object, ...)` as for a `SurvregFit`; R refuses a sparse"]
    #[doc = " frailty model."]
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

    #[doc = " `residuals(object, ...)` as for a `SurvregFit`; R refuses a sparse"]
    #[doc = " frailty model (its message, typo included)."]
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

#[doc = " `survreg()`'s penalty branch on explicit data (survreg.R:203-293 with"]
#[doc = " `survpenal.fit`): `penalties[i]` applies to the columns `pcols[i]` of"]
#[doc = " `data.covariates`; `assign` lists the columns of every model term (each"]
#[doc = " `pcols` entry must be one of them) and defaults to the penalised groups"]
#[doc = " plus one term per remaining column, in column order.  A sparse frailty"]
#[doc = " term is a single column of group codes.  The response is untransformed,"]
#[doc = " as for `survreg_fit`."]
#[pyfunction]
#[pyo3(signature = (data, distribution, penalties, pcols, assign=None, init=None, scale=0.0, control=None, robust=None, nstrat=None))]
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
    nstrat: Option<usize>,
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
        SurvpenalFit::fit_inputs(data, &terms, distribution, &options, nstrat)
    })?)
}

#[doc = " Rebuilds a pickled [`SurvpenalFit`] from its `__reduce__` state."]
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
