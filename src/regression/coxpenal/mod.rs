//! Penalised Cox models: R survival's `coxpenal.fit` (`R/coxpenal.fit.R`),
//! the fitter behind `coxph()` when the formula has `ridge()`, `pspline()`
//! or `frailty()` terms, producing the `coxph.penal` object.
//!
//! The penalty terms, their `cfun`s, `coxpenal.df` and the pieces of the
//! outer loop are shared with `survpenal.fit` in
//! [`crate::regression::penalized`]; [`kernel`] holds the C Newton-Raphson
//! kernels (`coxfit5.c`, `agfit5.c`).  This module is the outer loop: it
//! removes a sparse frailty column from the design, composes the penalty
//! callbacks (`f.expr1`/`f.expr2`, the persistent `coxlist1`/`coxlist2`),
//! iterates the inner fit and the `cfun`s over `theta`, restarts each inner
//! fit from the solution of the closest earlier `theta`, and assembles the
//! fit (`coxph.penal` plus the `coxph()` post-processing of
//! `R/coxph.R`: the offset centring, `n`, `nevent`, the Wald test and the
//! concordance).
//!
//! The fitted dense part is kept as a [`CoxPHFit`] so that `predict()`,
//! the residual types, the concordance and `survfit()` of a `coxph.penal`
//! object come from the same code as for a plain Cox model;
//! `survfit.coxph` drops the sparse frailty from the risk scores of a
//! frailty model, which [`CoxpenalFit::survfit`] reproduces.

mod kernel;

#[cfg(feature = "python")]
pub use crate::regression::penalized::CallbackPenalty;
pub use crate::regression::penalized::{
    CoxPenaltyTerms, FrailtyFamily, FrailtyMethod, FrailtyPenalty, ModelTerm, PenaltyHistory,
    PenaltyTerm, PsplineMethod, PsplinePenalty, RidgePenalty,
};

use self::kernel::{InnerFit, Kernel, KernelData};
use crate::constants::{COX_CONVERGENCE_TOLERANCE, COX_MAX_ITER, COX_RANK_TOLERANCE};
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::matrix::{matrix_from_rows, matrix_rows};
use crate::internal::validation::validate_finite;
use crate::regression::cox_optimizer::TieMethod;
use crate::regression::coxph::{
    Basehaz, CoxNewData, CoxPHFit, CoxSurvfitCurve, CoxphData, FittedCox, SurvfitOptions,
    add_offset_mean, centre_offset, newdata_from_python, nocenter_columns,
};
use crate::regression::coxph_wtest::wald_statistic;
use crate::regression::penalized::df::{DfInput, TermDf, coxpenal_df};
use crate::regression::penalized::terms::{
    Composer, PenaltyCallback, build_term_states, closest_saved, drop_sparse_column, histories,
    model_terms, penalty_shape, update_controls, validate_terms,
};
use ndarray::Array2;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// `coxph.control()$outer.max`.
pub const COXPENAL_OUTER_MAX: usize = 10;

/// Validated inputs of a penalised Cox fit.  `cox.x` holds every design
/// column, including the single column of group codes of a sparse frailty
/// term.
#[derive(Debug, Clone)]
pub struct CoxpenalData {
    pub cox: CoxphData,
    /// The model terms in formula order; every column belongs to exactly
    /// one term and at least one term is penalised.
    pub terms: Vec<ModelTerm>,
}

impl CoxpenalData {
    pub fn try_new(cox: CoxphData, terms: Vec<ModelTerm>) -> SurvivalResult<Self> {
        // coxpenal.fit is reached only for data with events
        cox.check_fit_input()?;
        validate_terms(cox.x.ncols(), &terms)?;
        Ok(Self { cox, terms })
    }
}

/// Fitting options: `coxph.control()` (`iter.max`, `outer.max`, `eps`,
/// `toler.chol`), `init`, `nocenter` and the cluster.
#[derive(Debug, Clone)]
pub struct CoxpenalOptions {
    /// Breslow or Efron; the exact method has no penalised kernel.
    pub method: TieMethod,
    /// Initial coefficients: one per dense column, optionally followed by
    /// one per frailty group.
    pub init: Option<Vec<f64>>,
    /// Inner (Newton) iterations per outer step.
    pub iter_max: usize,
    /// Outer iterations over the smoothing parameters.
    pub outer_max: usize,
    pub eps: f64,
    pub toler_chol: f64,
    /// As in [`CoxphOptions`]: columns whose values all lie in this set are
    /// not centred.  (R evaluates the rule on the full design and reads the
    /// flags of the dense columns by position, which misaligns them when a
    /// sparse frailty column is not the last one; this port evaluates the
    /// rule on the dense columns themselves.)
    pub nocenter: Option<Vec<f64>>,
    /// Cluster codes (`cluster()`, or the id of a robust request).  A
    /// penalised fit has no robust variance, but `coxph()` still passes the
    /// cluster to the concordance.
    pub cluster: Option<Vec<i32>>,
}

impl Default for CoxpenalOptions {
    fn default() -> Self {
        Self {
            method: TieMethod::Efron,
            init: None,
            iter_max: COX_MAX_ITER,
            outer_max: COXPENAL_OUTER_MAX,
            eps: COX_CONVERGENCE_TOLERANCE,
            toler_chol: COX_RANK_TOLERANCE,
            nocenter: Some(vec![-1.0, 0.0, 1.0]),
            cluster: None,
        }
    }
}

/// A fitted penalised Cox model (R's `coxph.penal` object).
///
/// The dense coefficients live in `coxph`, a [`CoxPHFit`] whose
/// coefficients, variance (`H^{-1}`), means, linear predictors (including a
/// sparse frailty) and martingale residuals are those of the penalised fit;
/// its `predict()` and residual methods are R's for a `coxph.penal` object.
#[pyclass(module = "survival._survival", skip_from_py_object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoxpenalFit {
    pub coxph: CoxPHFit,
    /// The same fit with the frailty removed from the linear predictors,
    /// present for a sparse frailty model: the risk scores `survfit.coxph`
    /// uses.
    curve_fit: Option<CoxPHFit>,
    /// `H^{-1} I H^{-1}` for the dense coefficients (`var2`).
    pub var2: Array2<f64>,
    /// Outer iterations and total inner iterations.
    #[pyo3(get)]
    pub iter: [usize; 2],
    /// Outer iterations whose inner loop used up `iter.max` (R warns about
    /// them).
    #[pyo3(get)]
    pub inner_failures: Vec<usize>,
    /// Effective degrees of freedom of each model term.
    #[pyo3(get)]
    pub df: Vec<f64>,
    /// The penalty at the first and the final outer iteration.
    #[pyo3(get)]
    pub penalty: [f64; 2],
    /// Per model term: 0 ordinary, 1 penalised, 2 sparse.
    pub pterms: Vec<u8>,
    /// R's `assign2`: the columns of each term in the dense design (a
    /// sparse term keeps its original column).
    #[pyo3(get)]
    pub assign2: Vec<Vec<usize>>,
    #[pyo3(get)]
    pub history: Vec<PenaltyHistory>,
    /// The fitted frailties of a sparse term (`frail`) and the diagonal
    /// of their variance (`fvar`).
    #[pyo3(get)]
    pub frail: Option<Vec<f64>>,
    #[pyo3(get)]
    pub fvar: Option<Vec<f64>>,
    /// 0-based frailty group of each observation.
    #[pyo3(get)]
    pub frail_index: Option<Vec<usize>>,
    /// The last penalty evaluations (`coxlist1` for the sparse term,
    /// `coxlist2` for the dense ones); `cox.zph` adds the dense second
    /// derivative to the information of a fit without a sparse term.
    #[pyo3(get)]
    pub coxlist1: Option<CoxPenaltyTerms>,
    #[pyo3(get)]
    pub coxlist2: Option<CoxPenaltyTerms>,
}

fn dot_row(x: &Array2<f64>, row: usize, coef: &[f64]) -> f64 {
    x.row(row).iter().zip(coef).map(|(v, b)| v * b).sum()
}

impl CoxpenalFit {
    /// `coxpenal.fit` at the centred offset, followed by the `coxph()`
    /// post-processing.
    pub fn fit(data: CoxpenalData, options: CoxpenalOptions) -> SurvivalResult<Self> {
        let CoxpenalData {
            cox: data,
            terms: model_terms,
        } = data;
        if options.method == TieMethod::Exact {
            return Err(SurvivalError::invalid_input(
                "penalised Cox models support ties = 'breslow' or 'efron' only",
            ));
        }
        if options.outer_max == 0 {
            return Err(SurvivalError::invalid_input("invalid value for outer.max"));
        }
        let n = data.n();
        let nevent = data.status.iter().filter(|&&s| s == 1).count();
        let n_eff = nevent as f64;
        let weights = data.weights.clone().unwrap_or_else(|| vec![1.0; n]);
        let (offset, offset_mean) = centre_offset(data.offset.as_deref(), n);
        let eps2 = options.eps.sqrt();

        let (pterms, sparse_term, shape) = penalty_shape(&model_terms);
        // Remove the sparse term's column from the design.
        let (xx, assign2, frailx, nfrail) = drop_sparse_column(data.x.view(), &model_terms);
        let nvar = xx.ncols();

        // Initial values.
        let (init, finit) = match &options.init {
            None => (vec![0.0; nvar], vec![0.0; nfrail]),
            Some(init) if init.len() == nvar => (init.clone(), vec![0.0; nfrail]),
            Some(init) if init.len() == nvar + nfrail => {
                (init[..nvar].to_vec(), init[nvar..].to_vec())
            }
            Some(_) => {
                return Err(SurvivalError::invalid_input(
                    "Wrong length for inital values",
                ));
            }
        };
        validate_finite(&init, "init")?;
        validate_finite(&finit, "init")?;

        // The penalised terms: pparm, cfun and the first theta.
        let terms = build_term_states(
            &model_terms,
            &assign2,
            &xx,
            frailx.as_deref(),
            nfrail,
            n,
            &data.status,
            eps2,
        )?;
        let need_df = terms.iter().any(|term| term.control.needs_df());
        let mut composer = Composer::new(terms, nfrail, nvar, shape.full_imat, false);

        let nocenter = nocenter_columns(&xx, options.nocenter.as_deref());
        let docenter: Vec<bool> = nocenter.iter().map(|&skip| !skip).collect();
        let mut kernel = Kernel::new(KernelData {
            stop: &data.time,
            start: data.entry.as_deref(),
            status: &data.status,
            x: &xx,
            weights: &weights,
            offset: &offset,
            strata: data.strata.as_deref(),
            frail: frailx.as_deref(),
            nfrail,
            efron: options.method == TieMethod::Efron,
            docenter: &docenter,
        });
        let means = kernel.means().to_vec();

        // coxfit5_a / agfit5a: the log likelihood at the initial values
        // without the frailty, plus the dense penalty at the first thetas.
        let mut loglik0 = kernel.initial_loglik(&init);
        if shape.dense {
            let mut beta = init.clone();
            loglik0 += composer.call(2, &mut beta)?.penalty;
        }

        // The outer loop over theta.
        let mut iter2 = 0;
        let mut iter = 0;
        let mut inner_failures = Vec::new();
        let mut theta_save: Vec<Vec<f64>> = Vec::new();
        let mut coef_save: Vec<Vec<f64>> = Vec::new();
        let mut fcoef_save: Vec<Vec<f64>> = Vec::new();
        let mut init = init;
        let mut finit = finit;
        let mut penalty0 = 0.0;
        let mut penalty = 0.0;
        let mut coxfit: Option<InnerFit> = None;
        let mut beta = Vec::new();
        let mut fbeta = Vec::new();
        let mut fdiag = Vec::new();
        let mut dftemp: Option<TermDf> = None;
        for outer in 1..=options.outer_max {
            let thetas: Vec<f64> = composer.terms.iter().map(|term| term.state.theta).collect();
            beta = init.clone();
            fbeta = finit.clone();
            let inner = kernel.newton(
                options.iter_max,
                &mut beta,
                &mut fbeta,
                options.eps,
                options.toler_chol,
                shape,
                &mut composer,
            )?;
            iter = outer;
            iter2 += inner.iter;
            if inner.iter >= options.iter_max {
                inner_failures.push(outer);
            }
            theta_save.push(thetas);
            coef_save.push(beta.clone());
            fcoef_save.push(fbeta.clone());
            // An infinite penalty made the C code set fdiag = 1 in
            // self-defence; those coefficients are zero and so is their
            // variance.
            fdiag = inner.fdiag.clone();
            if nfrail > 0 && composer.coxlist1.flag[0] {
                fdiag[..nfrail].fill(0.0);
            }
            if shape.dense {
                for i in 0..nvar {
                    if composer.coxlist2.flag[i] {
                        fdiag[nfrail + i] = 0.0;
                    }
                }
            }
            if need_df {
                dftemp = Some(coxpenal_df(DfInput {
                    hmat: &inner.hmat,
                    hinv: &inner.hinv,
                    fdiag: &fdiag,
                    assign: &assign2,
                    shape,
                    pen1: &composer.coxlist1.second,
                    pen2: &composer.coxlist2.second,
                    sparse_term,
                })?);
            }
            penalty = 0.0;
            if nfrail > 0 {
                penalty -= composer.coxlist1.penalty;
            }
            if shape.dense {
                penalty -= composer.coxlist2.penalty;
            }
            // The C code returns PL - penalty.
            let loglik1 = inner.loglik + penalty;
            if outer == 1 {
                penalty0 = penalty;
            }

            // The control functions.
            let done = update_controls(
                &mut composer.terms,
                outer,
                loglik1,
                inner.loglik,
                n_eff,
                dftemp.as_ref(),
                &beta,
                &fbeta,
            )?;
            coxfit = Some(inner);
            if done {
                break;
            }

            // Starting values for the next iteration: the solution of the
            // closest earlier theta (the first of equally close ones).
            let next: Vec<f64> = composer.terms.iter().map(|term| term.state.theta).collect();
            let which = closest_saved(&theta_save, &next);
            init = coef_save[which].clone();
            finit = fcoef_save[which].clone();
        }
        let coxfit = coxfit.expect("outer.max >= 1");

        // Linear predictors (with the frailty) at the centred offset.
        let center: f64 = means.iter().zip(&beta).map(|(m, b)| m * b).sum();
        let lp_no_frailty: Vec<f64> = (0..n)
            .map(|i| offset[i] + dot_row(&xx, i, &beta) - center)
            .collect();
        let (lp, lp_no_frailty) = match &frailx {
            Some(frailx) => (
                lp_no_frailty
                    .iter()
                    .zip(frailx)
                    .map(|(lp, &g)| lp + fbeta[g])
                    .collect(),
                Some(lp_no_frailty),
            ),
            None => (lp_no_frailty, None),
        };

        let dftemp = match dftemp {
            Some(df) => df,
            None => coxpenal_df(DfInput {
                hmat: &coxfit.hmat,
                hinv: &coxfit.hinv,
                fdiag: &fdiag,
                assign: &assign2,
                shape,
                pen1: &composer.coxlist1.second,
                pen2: &composer.coxlist2.second,
                sparse_term,
            })?,
        };
        let coefficients: Vec<f64> = (0..nvar)
            .map(|i| {
                if fdiag[nfrail + i] == 0.0 {
                    f64::NAN
                } else {
                    beta[i]
                }
            })
            .collect();
        let coef_or_zero: Vec<f64> = coefficients
            .iter()
            .map(|b| if b.is_nan() { 0.0 } else { *b })
            .collect();

        // Martingale residuals: coxfit5_c's expected events for
        // right-censored data, agmart3 (from the linear predictors) for
        // (start, stop] data.
        let residuals = data.entry.is_none().then(|| {
            let expected = kernel.expected_events();
            data.status
                .iter()
                .zip(&expected)
                .map(|(&s, e)| f64::from(s) - e)
                .collect()
        });
        let shift: Vec<f64> = coef_or_zero
            .iter()
            .enumerate()
            .map(|(i, b)| b - options.init.as_ref().map_or(0.0, |init| init[i]))
            .collect();
        let wald_test = wald_statistic(&dftemp.var, &shift, options.toler_chol)?;

        // The dense part as a Cox model, with coxph()'s concordance of the
        // final linear predictors.
        let coxph = CoxPHFit::from_fitted(
            CoxphData { x: xx, ..data },
            FittedCox {
                method: options.method,
                coefficients,
                var: dftemp.var,
                loglik: [loglik0, coxfit.loglik + penalty],
                iter: iter2,
                flag: coxfit.flag,
                means,
                nocenter,
                first: coxfit.u[nfrail..].to_vec(),
                linear_predictors: lp,
                offset_mean,
                residuals,
                wald_test,
            },
            options.cluster.as_deref(),
        )?;
        let curve_fit = lp_no_frailty.map(|mut lp| {
            add_offset_mean(&mut lp, offset_mean);
            let mut curve = coxph.clone();
            curve.linear_predictors = lp;
            curve
        });

        let history = histories(&composer.terms);
        Ok(Self {
            coxph,
            curve_fit,
            var2: dftemp.var2,
            iter: [iter, iter2],
            inner_failures,
            df: dftemp.df,
            penalty: [penalty0, penalty],
            pterms,
            assign2,
            history,
            frail: frailx.is_some().then_some(fbeta),
            fvar: dftemp.fvar,
            frail_index: frailx,
            coxlist1: sparse_term.map(|_| composer.coxlist1),
            coxlist2: shape.dense.then_some(composer.coxlist2),
        })
    }

    /// The fit `survfit.coxph` works from: for a sparse frailty model the
    /// risk scores omit the frailty.
    fn curve_source(&self) -> &CoxPHFit {
        self.curve_fit.as_ref().unwrap_or(&self.coxph)
    }

    /// `basehaz(fit, centered)`.
    pub fn basehaz(&self, centered: bool) -> SurvivalResult<Basehaz> {
        self.curve_source().basehaz(centered)
    }

    /// New data cannot be used with a frailty term, as in R.
    fn check_newdata_allowed(&self) -> SurvivalResult<()> {
        if self.frail.is_some() {
            return Err(SurvivalError::invalid_input(
                "Newdata cannot be used when a model has frailty terms",
            ));
        }
        Ok(())
    }

    /// `survfit(fit, newdata, ...)`.
    pub fn survfit(
        &self,
        newdata: Option<&CoxNewData>,
        options: SurvfitOptions,
    ) -> SurvivalResult<Vec<CoxSurvfitCurve>> {
        if newdata.is_some() {
            self.check_newdata_allowed()?;
        }
        self.curve_source().survfit(newdata, options)
    }

    /// `survfit(fit, newdata, id)` for time-dependent new data.
    pub fn survfit_individual(
        &self,
        newdata: &CoxNewData,
        id: &[i32],
        options: SurvfitOptions,
    ) -> SurvivalResult<Vec<CoxSurvfitCurve>> {
        self.check_newdata_allowed()?;
        self.curve_source().survfit_individual(newdata, id, options)
    }
}

/// A penalty term for [`coxpenal_fit`]: `ridge()`, `pspline()`, `frailty()`
/// or a Python callback.
#[pyclass(module = "survival._survival", from_py_object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoxPenalty {
    pub term: PenaltyTerm,
}

#[pymethods]
impl CoxPenalty {
    /// Pickle and copy support (see `internal::pickle`); a callback term
    /// pickles its callable, which pickle must be able to reach by name.
    #[cfg(feature = "python")]
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        match &self.term {
            PenaltyTerm::Callback(callback) => (
                py.get_type::<Self>().getattr("callback")?,
                (callback.fexpr.clone_ref(py), callback.diag, callback.sparse),
            )
                .into_pyobject(py),
            _ => crate::internal::pickle::reduce(py, self)?.into_pyobject(py),
        }
    }

    /// `ridge(..., theta, df, eps, scale)`; `scale_values` are the column
    /// variances R's `ridge()` takes in the model frame, before `subset` and
    /// `na.action` (by default those of the rows fitted).
    #[staticmethod]
    #[pyo3(signature = (theta=None, df=None, eps=0.1, scale=true, scale_values=None))]
    fn ridge(
        theta: Option<f64>,
        df: Option<f64>,
        eps: f64,
        scale: bool,
        scale_values: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        Ok(Self {
            term: PenaltyTerm::ridge(theta, df, eps, scale, scale_values)?,
        })
    }

    /// `pspline(x, df, theta, nterm, eps, method, intercept)` without the
    /// basis: build the columns with `pspline_basis(x, nterm, degree,
    /// boundary_knots)` (dropping the first column unless `intercept`).
    #[staticmethod]
    #[pyo3(signature = (df=4.0, theta=None, nterm=None, eps=None, method=None, intercept=false))]
    fn pspline(
        df: f64,
        theta: Option<f64>,
        nterm: Option<f64>,
        eps: Option<f64>,
        method: Option<&str>,
        intercept: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            term: PenaltyTerm::pspline(df, theta, nterm, eps, method, intercept)?,
        })
    }

    /// `frailty(x, distribution, sparse, theta, df, eps, method, tdf, caic,
    /// init)`: a sparse term is one column of group codes, a dense one the
    /// indicator matrix of the groups.  `n` is R's `length(x)` in the model
    /// frame, before `subset` and `na.action` (by default the rows fitted).
    #[staticmethod]
    #[pyo3(signature = (distribution="gamma", sparse=true, theta=None, df=None, eps=None, method=None, tdf=5.0, caic=false, init=None, n=None))]
    #[allow(clippy::too_many_arguments)]
    fn frailty(
        distribution: &str,
        sparse: bool,
        theta: Option<f64>,
        df: Option<f64>,
        eps: Option<f64>,
        method: Option<&str>,
        tdf: f64,
        caic: bool,
        init: Option<Vec<f64>>,
        n: Option<usize>,
    ) -> PyResult<Self> {
        let distribution = FrailtyFamily::parse(distribution, tdf)?;
        Ok(Self {
            term: PenaltyTerm::frailty(
                distribution,
                sparse,
                theta,
                df,
                eps,
                method,
                caic,
                init,
                n,
            )?,
        })
    }

    /// A user-defined penalty: `fexpr(coef, which=which)` returns the
    /// `coxlist` of `cox_callback` for the term's coefficients.
    #[cfg(feature = "python")]
    #[staticmethod]
    #[pyo3(signature = (fexpr, diag=true, sparse=false))]
    fn callback(fexpr: Py<PyAny>, diag: bool, sparse: bool) -> Self {
        Self {
            term: PenaltyTerm::Callback(CallbackPenalty {
                fexpr: std::sync::Arc::new(fexpr),
                diag,
                sparse,
            }),
        }
    }

    /// `"ridge"`, `"pspline"`, `"frailty"` or `"callback"`.
    #[getter]
    fn kind(&self) -> &'static str {
        self.term.r_name()
    }

    #[getter]
    fn sparse(&self) -> bool {
        self.term.is_sparse()
    }

    #[getter]
    fn diag(&self) -> bool {
        self.term.is_diagonal()
    }

    /// The `nterm` of a `pspline` term (its basis size is `nterm + degree`).
    #[getter]
    fn nterm(&self) -> Option<usize> {
        match &self.term {
            PenaltyTerm::Pspline(spline) => Some(spline.nterm),
            _ => None,
        }
    }

    /// The distribution of a `frailty` term: `"gamma"`, `"gaussian"` or `"t"`.
    #[getter]
    fn distribution(&self) -> Option<&'static str> {
        match &self.term {
            PenaltyTerm::Frailty(frailty) => Some(frailty.distribution.r_name()),
            _ => None,
        }
    }
}

#[pymethods]
impl CoxpenalFit {
    /// Pickle and copy support (see `internal::pickle`).
    #[cfg(feature = "python")]
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<crate::internal::pickle::Reduced<'py>> {
        crate::internal::pickle::reduce(py, self)
    }

    /// The dense part of the fit as a Cox model: `predict()`, the residual
    /// types and `survfit()` with new data come from here.
    #[getter(coxph)]
    fn coxph_getter(&self) -> CoxPHFit {
        self.coxph.clone()
    }

    #[getter(pterms)]
    fn pterms_getter(&self) -> Vec<usize> {
        self.pterms
            .iter()
            .map(|&value| usize::from(value))
            .collect()
    }

    #[getter]
    fn coefficients(&self) -> Vec<f64> {
        self.coxph.coefficients.clone()
    }

    #[getter]
    fn var(&self) -> Vec<Vec<f64>> {
        matrix_rows(&self.coxph.var)
    }

    #[getter(var2)]
    fn var2_getter(&self) -> Vec<Vec<f64>> {
        matrix_rows(&self.var2)
    }

    #[getter]
    fn loglik(&self) -> [f64; 2] {
        self.coxph.loglik
    }

    #[getter]
    fn linear_predictors(&self) -> Vec<f64> {
        self.coxph.linear_predictors.clone()
    }

    #[getter]
    fn residuals(&self) -> Vec<f64> {
        self.coxph.residuals.clone()
    }

    #[getter]
    fn means(&self) -> Vec<f64> {
        self.coxph.means.clone()
    }

    #[getter]
    fn method(&self) -> TieMethod {
        self.coxph.method
    }

    #[getter]
    fn n(&self) -> usize {
        self.coxph.n
    }

    #[getter]
    fn nevent(&self) -> usize {
        self.coxph.nevent
    }

    #[getter]
    fn wald_test(&self) -> f64 {
        self.coxph.wald_test
    }

    /// Rank flag of the last inner fit (1000: it did not converge).
    #[getter]
    fn flag(&self) -> i32 {
        self.coxph.flag
    }

    /// `basehaz(fit, centered)`.
    #[pyo3(name = "basehaz", signature = (centered = true))]
    fn basehaz_py(&self, centered: bool) -> PyResult<Basehaz> {
        Ok(self.basehaz(centered)?)
    }

    /// `survfit(fit, newdata, stype, ctype, se.fit, censor, start.time)`.
    #[pyo3(name = "survfit", signature = (newdata = None, new_strata = None, new_offset = None, stype = 2, ctype = None, se_fit = true, censor = true, start_time = None))]
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
        start_time: Option<f64>,
    ) -> PyResult<Vec<CoxSurvfitCurve>> {
        let newdata =
            newdata_from_python(&self.coxph, newdata, new_strata, new_offset, None, None)?;
        Ok(self.survfit(
            newdata.as_ref(),
            SurvfitOptions {
                stype,
                ctype,
                se_fit,
                censor,
                start_time,
            },
        )?)
    }

    /// `survfit(fit, newdata, id)` for time-dependent new data.
    #[pyo3(name = "survfit_individual", signature = (newdata, new_entry, new_time, id, new_strata = None, new_offset = None, stype = 2, ctype = None, se_fit = true, censor = true, start_time = None))]
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
        start_time: Option<f64>,
    ) -> PyResult<Vec<CoxSurvfitCurve>> {
        let newdata = newdata_from_python(
            &self.coxph,
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
                start_time,
            },
        )?)
    }
}

/// `coxph()` with penalised terms on explicit data: `penalties[i]` applies
/// to the columns `pcols[i]` of `x`; `assign` lists the columns of every
/// model term (each `pcols` entry must be one of them) and defaults to the
/// penalised groups plus one term per remaining column, in column order.
/// A sparse frailty term is a single column of group codes.  `cluster`
/// only enters the concordance (a penalised fit has no robust variance).
#[pyfunction]
#[pyo3(signature = (time, status, x, penalties, pcols, assign=None, entry=None, strata=None, weights=None, offset=None, method="efron", init=None, iter_max=None, outer_max=None, eps=None, toler_chol=None, nocenter=None, cluster=None))]
#[allow(clippy::too_many_arguments)]
pub fn coxpenal_fit(
    time: Vec<f64>,
    status: Vec<i32>,
    x: Vec<Vec<f64>>,
    penalties: Vec<CoxPenalty>,
    pcols: Vec<Vec<usize>>,
    assign: Option<Vec<Vec<usize>>>,
    entry: Option<Vec<f64>>,
    strata: Option<Vec<i32>>,
    weights: Option<Vec<f64>>,
    offset: Option<Vec<f64>>,
    method: &str,
    init: Option<Vec<f64>>,
    iter_max: Option<usize>,
    outer_max: Option<usize>,
    eps: Option<f64>,
    toler_chol: Option<f64>,
    nocenter: Option<Vec<f64>>,
    cluster: Option<Vec<i32>>,
) -> PyResult<CoxpenalFit> {
    if x.len() != time.len() {
        return Err(SurvivalError::invalid_input(format!(
            "x has {} rows but time has {}",
            x.len(),
            time.len()
        ))
        .into());
    }
    let x = matrix_from_rows(&x, "x")?;
    let terms = model_terms(
        x.ncols(),
        penalties.into_iter().map(|penalty| penalty.term).collect(),
        pcols,
        assign,
    )?;
    let data = CoxpenalData::try_new(
        CoxphData::try_new(time, entry, status, x, weights, strata, offset)?,
        terms,
    )?;
    let defaults = CoxpenalOptions::default();
    let options = CoxpenalOptions {
        method: TieMethod::parse(Some(method))?,
        init,
        iter_max: iter_max.unwrap_or(defaults.iter_max),
        outer_max: outer_max.unwrap_or(defaults.outer_max),
        eps: eps.unwrap_or(defaults.eps),
        toler_chol: toler_chol.unwrap_or(defaults.toler_chol),
        nocenter: nocenter.or(defaults.nocenter),
        cluster,
    };
    Ok(CoxpenalFit::fit(data, options)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regression::coxph::CoxphOptions;
    use crate::regression::penalized::terms::control_of;

    fn kidney_like() -> (Vec<f64>, Vec<i32>, Array2<f64>) {
        // Four groups of three, one covariate.
        let time = vec![
            8.0, 16.0, 23.0, 13.0, 22.0, 28.0, 30.0, 12.0, 24.0, 15.0, 7.0, 9.0,
        ];
        let status = vec![1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0];
        let rows: Vec<f64> = (0..12)
            .flat_map(|i| [(i % 5) as f64 * 0.7 - 1.0, (i / 3 + 1) as f64])
            .collect();
        (time, status, Array2::from_shape_vec((12, 2), rows).unwrap())
    }

    fn fit_terms(terms: Vec<ModelTerm>, options: CoxpenalOptions) -> CoxpenalFit {
        let (time, status, x) = kidney_like();
        let data = CoxpenalData::try_new(
            CoxphData::try_new(time, None, status, x, None, None, None).unwrap(),
            terms,
        )
        .unwrap();
        CoxpenalFit::fit(data, options).unwrap()
    }

    #[test]
    fn ridge_with_a_tiny_theta_matches_the_unpenalised_fit() {
        let (time, status, x) = kidney_like();
        let plain = CoxPHFit::fit(
            CoxphData::try_new(
                time.clone(),
                None,
                status.clone(),
                x.slice(ndarray::s![.., ..1]).to_owned(),
                None,
                None,
                None,
            )
            .unwrap(),
            CoxphOptions::default(),
        )
        .unwrap();
        let data = CoxpenalData::try_new(
            CoxphData::try_new(
                time,
                None,
                status,
                x.slice(ndarray::s![.., ..1]).to_owned(),
                None,
                None,
                None,
            )
            .unwrap(),
            vec![ModelTerm {
                columns: vec![0],
                penalty: Some(PenaltyTerm::ridge(Some(1e-10), None, 0.1, false, None).unwrap()),
            }],
        )
        .unwrap();
        let fit = CoxpenalFit::fit(data, CoxpenalOptions::default()).unwrap();
        assert!((fit.coxph.coefficients[0] - plain.coefficients[0]).abs() < 1e-6);
        assert!((fit.coxph.loglik[1] - plain.loglik[1]).abs() < 1e-8);
        assert!((fit.coxph.loglik[0] - plain.loglik[0]).abs() < 1e-12);
        assert_eq!(fit.iter[0], 1);
        assert_eq!(fit.pterms, vec![1]);
        assert!((fit.df[0] - 1.0).abs() < 1e-6);
        assert!(fit.frail.is_none());
        let total: f64 = fit.coxph.residuals.iter().sum();
        assert!(total.abs() < 1e-10);
        for (a, b) in fit.coxph.residuals.iter().zip(&plain.residuals) {
            assert!((a - b).abs() < 1e-6);
        }
        assert_eq!(fit.history.len(), 1);
        assert!(fit.history[0].done);
        let curves = fit.survfit(None, SurvfitOptions::default()).unwrap();
        assert_eq!(curves.len(), 1);
    }

    #[test]
    fn sparse_gamma_frailty_fits_and_reports_the_frailty_pieces() {
        let fit = fit_terms(
            vec![
                ModelTerm {
                    columns: vec![0],
                    penalty: None,
                },
                ModelTerm {
                    columns: vec![1],
                    penalty: Some(
                        PenaltyTerm::frailty(
                            FrailtyFamily::Gamma,
                            true,
                            Some(0.5),
                            None,
                            None,
                            None,
                            false,
                            None,
                            None,
                        )
                        .unwrap(),
                    ),
                },
            ],
            CoxpenalOptions::default(),
        );
        assert_eq!(fit.pterms, vec![0, 2]);
        assert_eq!(fit.coxph.nvar(), 1);
        let frail = fit.frail.as_ref().unwrap();
        assert_eq!(frail.len(), 4);
        // The gamma penalty recentres so that mean(exp(frail)) is one.
        let mean_exp: f64 = frail.iter().map(|f| f.exp()).sum::<f64>() / 4.0;
        assert!((mean_exp - 1.0).abs() < 1e-8);
        assert_eq!(fit.fvar.as_ref().unwrap().len(), 4);
        assert_eq!(fit.df.len(), 2);
        assert!(fit.df[1] > 0.0 && fit.df[1] < 4.0);
        assert_eq!(fit.assign2, vec![vec![0], vec![1]]);
        assert!(fit.history[0].c_loglik.is_some());
        assert!(fit.history[0].done);
        // The linear predictor carries the frailty; survfit does not.
        let lp = &fit.coxph.linear_predictors;
        let index = fit.frail_index.as_ref().unwrap();
        for i in 0..12 {
            let without = lp[i] - frail[index[i]];
            assert!((without - fit.curve_fit.as_ref().unwrap().linear_predictors[i]).abs() < 1e-12);
        }
        let total: f64 = fit.coxph.residuals.iter().sum();
        assert!(total.abs() < 1e-10);
        assert!(fit.coxlist1.is_some() && fit.coxlist2.is_none());
        assert!(fit.survfit(None, SurvfitOptions::default()).is_ok());
    }

    /// A ridge penalty with a negligible theta reproduces `coxph` on the
    /// same data; the residuals exercise the stratum reset of `coxfit5_c`
    /// (right-censored) and `agmart3` ((start, stop]), the weighted Efron
    /// fit the `efron_wt` of `agfit5b`.
    #[test]
    fn negligible_penalty_matches_coxph_with_strata_weights_and_entry() {
        let (time, status, x) = kidney_like();
        let x = x.slice(ndarray::s![.., ..1]).to_owned();
        let entry: Vec<f64> = time.iter().map(|t| t * 0.25).collect();
        let strata = vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
        let weights = vec![1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0];
        for (entry, strata, weights) in [
            (None, Some(strata.clone()), None),
            (Some(entry.clone()), None, Some(weights.clone())),
            (Some(entry), Some(strata), Some(weights)),
        ] {
            let plain = CoxPHFit::fit(
                CoxphData::try_new(
                    time.clone(),
                    entry.clone(),
                    status.clone(),
                    x.clone(),
                    weights.clone(),
                    strata.clone(),
                    None,
                )
                .unwrap(),
                CoxphOptions::default(),
            )
            .unwrap();
            let data = CoxpenalData::try_new(
                CoxphData::try_new(
                    time.clone(),
                    entry,
                    status.clone(),
                    x.clone(),
                    weights,
                    strata,
                    None,
                )
                .unwrap(),
                vec![ModelTerm {
                    columns: vec![0],
                    penalty: Some(PenaltyTerm::ridge(Some(1e-12), None, 0.1, false, None).unwrap()),
                }],
            )
            .unwrap();
            let fit = CoxpenalFit::fit(data, CoxpenalOptions::default()).unwrap();
            assert!((fit.coxph.coefficients[0] - plain.coefficients[0]).abs() < 1e-6);
            assert!((fit.coxph.loglik[1] - plain.loglik[1]).abs() < 1e-8);
            for (a, b) in fit.coxph.residuals.iter().zip(&plain.residuals) {
                assert!((a - b).abs() < 1e-6, "{a} != {b}");
            }
        }
    }

    #[test]
    fn a_frailty_alone_is_a_null_model_with_random_effects() {
        let (time, status, x) = kidney_like();
        let data = CoxpenalData::try_new(
            CoxphData::try_new(
                time,
                None,
                status,
                x.slice(ndarray::s![.., 1..]).to_owned(),
                None,
                None,
                None,
            )
            .unwrap(),
            vec![ModelTerm {
                columns: vec![0],
                penalty: Some(
                    PenaltyTerm::frailty(
                        FrailtyFamily::Gaussian,
                        true,
                        Some(0.4),
                        None,
                        None,
                        None,
                        false,
                        None,
                        None,
                    )
                    .unwrap(),
                ),
            }],
        )
        .unwrap();
        let fit = CoxpenalFit::fit(data, CoxpenalOptions::default()).unwrap();
        assert!(fit.coxph.coefficients.is_empty());
        assert_eq!(fit.coxph.var.shape(), &[0, 0]);
        assert_eq!(fit.pterms, vec![2]);
        assert_eq!(fit.df.len(), 1);
        let frail = fit.frail.as_ref().unwrap();
        assert_eq!(frail.len(), 4);
        // The Gaussian penalty recentres the frailties at zero.
        assert!(frail.iter().sum::<f64>().abs() < 1e-8);
        assert_eq!(fit.coxph.wald_test, 0.0);
        assert!(fit.history[0].done);
        assert!(fit.history[0].c_loglik.is_none());
    }

    #[test]
    fn the_frailty_df_search_starts_from_the_model_frame_length() {
        // frailty(x, df = 4) starts from theta = 3 * df / length(x), x being
        // the model-frame column before subset and na.action.
        let frailty = |n| {
            PenaltyTerm::frailty(
                FrailtyFamily::Gamma,
                true,
                None,
                Some(4.0),
                None,
                None,
                false,
                None,
                n,
            )
            .unwrap()
        };
        let guess = |term: &PenaltyTerm| control_of(term, 5, 200, 1e-4).unwrap().initial().theta;
        assert_eq!(guess(&frailty(Some(228))), 3.0 * 4.0 / 228.0);
        assert_eq!(guess(&frailty(None)), 3.0 * 4.0 / 200.0);
    }

    #[test]
    fn invalid_terms_are_rejected() {
        let (time, status, x) = kidney_like();
        let ridge = PenaltyTerm::ridge(Some(1.0), None, 0.1, true, None).unwrap();
        let frailty = |sparse: bool| {
            PenaltyTerm::frailty(
                FrailtyFamily::Gamma,
                sparse,
                Some(1.0),
                None,
                None,
                None,
                false,
                None,
                None,
            )
            .unwrap()
        };
        let make = |terms: Vec<ModelTerm>| {
            CoxpenalData::try_new(
                CoxphData::try_new(
                    time.clone(),
                    None,
                    status.clone(),
                    x.clone(),
                    None,
                    None,
                    None,
                )
                .unwrap(),
                terms,
            )
        };
        // No penalty at all.
        assert!(
            make(vec![ModelTerm {
                columns: vec![0, 1],
                penalty: None
            }])
            .is_err()
        );
        // A column in two terms.
        assert!(
            make(vec![
                ModelTerm {
                    columns: vec![0, 1],
                    penalty: Some(ridge.clone())
                },
                ModelTerm {
                    columns: vec![1],
                    penalty: None
                }
            ])
            .is_err()
        );
        // Two sparse terms, and a multi-column sparse term.
        assert!(
            make(vec![
                ModelTerm {
                    columns: vec![0],
                    penalty: Some(frailty(true))
                },
                ModelTerm {
                    columns: vec![1],
                    penalty: Some(frailty(true))
                }
            ])
            .is_err()
        );
        assert!(
            make(vec![ModelTerm {
                columns: vec![0, 1],
                penalty: Some(frailty(true))
            }])
            .is_err()
        );
        let data = make(vec![ModelTerm {
            columns: vec![0, 1],
            penalty: Some(ridge),
        }])
        .unwrap();
        let exact = CoxpenalOptions {
            method: TieMethod::Exact,
            ..CoxpenalOptions::default()
        };
        assert!(CoxpenalFit::fit(data.clone(), exact).is_err());
        let bad_init = CoxpenalOptions {
            init: Some(vec![0.0]),
            ..CoxpenalOptions::default()
        };
        assert!(CoxpenalFit::fit(data, bad_init).is_err());
    }
}
