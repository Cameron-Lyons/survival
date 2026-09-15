//! Parametric (accelerated failure time) regression: R's `survreg`.
//!
//! Ports `R/survreg.R` (response transform, likelihood correction, robust
//! variance), `R/survreg.fit.R` (rescaling, starting values, the
//! intercept-only fit) and `src/survreg6.c` (the Newton-Raphson iteration
//! with step halving and the Fisher-scoring fall back) from the CRAN
//! `survival` package.  The log-likelihood kernel lives in
//! [`crate::regression::survregc1`], the distributions in
//! [`crate::regression::survreg_distributions`], predictions in
//! [`crate::regression::survreg_predict`] and residuals in
//! [`crate::residuals::survreg_resid`].

use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::matrix::{cholesky2, chsolve2, symmetric_inverse_via_cholesky};
use crate::internal::validation::{
    validate_finite, validate_length, validate_non_empty, validate_positive,
};
use crate::regression::survreg_distributions::{SurvregDistribution, SurvregTransform};
use crate::regression::survreg_predict::{
    SurvregNewdata, SurvregPredictType, SurvregPrediction, predict_survreg,
};
use crate::regression::survregc1::{SurvregKernel, SurvregLikelihood};
use crate::residuals::survreg_resid::{
    SurvregResidType, SurvregResiduals, residuals_survreg, survreg_deriv,
};
use ndarray::{Array2, ArrayView2};
use pyo3::prelude::*;

/// `survreg.control()`: the iteration settings of a fit.
#[pyclass(from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurvregControl {
    /// `iter.max` (R default 30).
    #[pyo3(get, set)]
    pub iter_max: usize,
    /// `rel.tolerance`: iteration stops when the relative change in the
    /// log-likelihood drops below it (R default 1e-9).
    #[pyo3(get, set)]
    pub rel_tolerance: f64,
    /// `toler.chol`: the Cholesky pivot tolerance (R default 1e-10).
    #[pyo3(get, set)]
    pub toler_chol: f64,
}

impl Default for SurvregControl {
    fn default() -> Self {
        Self {
            iter_max: 30,
            rel_tolerance: 1e-9,
            toler_chol: 1e-10,
        }
    }
}

impl SurvregControl {
    fn validate(&self) -> SurvivalResult<()> {
        if !self.rel_tolerance.is_finite() || self.rel_tolerance <= 0.0 {
            return Err(SurvivalError::invalid_input(
                "rel_tolerance must be a finite positive value",
            ));
        }
        if !self.toler_chol.is_finite() || self.toler_chol <= 0.0 {
            return Err(SurvivalError::invalid_input(
                "toler_chol must be a finite positive value",
            ));
        }
        Ok(())
    }
}

#[pymethods]
impl SurvregControl {
    #[new]
    #[pyo3(signature = (iter_max=30, rel_tolerance=1e-9, toler_chol=1e-10))]
    fn new(iter_max: usize, rel_tolerance: f64, toler_chol: f64) -> PyResult<Self> {
        let control = Self {
            iter_max,
            rel_tolerance,
            toler_chol,
        };
        control.validate()?;
        Ok(control)
    }
}

/// The data of a `survreg` fit: the response with R's `Surv` censoring
/// codes, the design matrix (which must contain the intercept column when
/// one is wanted, as `model.matrix` would) and the optional case weights,
/// offset, strata and cluster.
///
/// `status` uses the codes of an `interval`-type `Surv`: 0 right censored,
/// 1 exact, 2 left censored, 3 interval censored between `time` and
/// `time2`; `time2` is only read for code-3 rows.  Weights must be positive
/// (`survreg.fit`: "Invalid weights, must be >0").  Strata and cluster are
/// zero-based codes.
#[pyclass(from_py_object)]
#[derive(Debug, Clone, PartialEq)]
pub struct SurvregData {
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub time2: Option<Vec<f64>>,
    #[pyo3(get)]
    pub status: Vec<i32>,
    #[pyo3(get)]
    pub covariates: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub weights: Option<Vec<f64>>,
    #[pyo3(get)]
    pub offset: Option<Vec<f64>>,
    #[pyo3(get)]
    pub strata: Option<Vec<usize>>,
    #[pyo3(get)]
    pub cluster: Option<Vec<usize>>,
}

impl SurvregData {
    /// Validates every component once; the fit relies on it.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        time: Vec<f64>,
        status: Vec<i32>,
        covariates: Vec<Vec<f64>>,
        time2: Option<Vec<f64>>,
        weights: Option<Vec<f64>>,
        offset: Option<Vec<f64>>,
        strata: Option<Vec<usize>>,
        cluster: Option<Vec<usize>>,
    ) -> SurvivalResult<Self> {
        validate_non_empty(&time, "time")?;
        validate_finite(&time, "time")?;
        let n = time.len();
        validate_length(n, status.len(), "status")?;
        validate_length(n, covariates.len(), "covariates")?;
        let nvar = covariates[0].len();
        if nvar == 0 {
            return Err(SurvivalError::invalid_input(
                "covariates must have at least one column (the intercept)",
            ));
        }
        for (index, row) in covariates.iter().enumerate() {
            validate_length(nvar, row.len(), &format!("covariates row {index}"))?;
            validate_finite(row, &format!("covariates row {index}"))?;
        }
        for (index, &code) in status.iter().enumerate() {
            if !(0..=3).contains(&code) {
                return Err(SurvivalError::invalid_input(format!(
                    "status must contain the codes 0 (right censored), 1 (exact), 2 (left censored) or 3 (interval censored); got {code} at index {index}"
                )));
            }
        }
        let interval_rows: Vec<usize> = (0..n).filter(|&i| status[i] == 3).collect();
        match &time2 {
            Some(time2) => {
                validate_length(n, time2.len(), "time2")?;
                for &i in &interval_rows {
                    if !time2[i].is_finite() {
                        return Err(SurvivalError::invalid_input(format!(
                            "time2 contains non-finite value {} at interval-censored index {i}",
                            time2[i]
                        )));
                    }
                    if time2[i] < time[i] {
                        return Err(SurvivalError::invalid_input(format!(
                            "Invalid interval: start > stop at index {i}"
                        )));
                    }
                }
            }
            None if !interval_rows.is_empty() => {
                return Err(SurvivalError::invalid_input(
                    "time2 is required for interval-censored (status 3) rows",
                ));
            }
            None => {}
        }
        if let Some(weights) = &weights {
            validate_length(n, weights.len(), "weights")?;
            validate_finite(weights, "weights")?;
            validate_positive(weights, "weights")?;
        }
        if let Some(offset) = &offset {
            validate_length(n, offset.len(), "offset")?;
            validate_finite(offset, "offset")?;
        }
        if let Some(strata) = &strata {
            validate_length(n, strata.len(), "strata")?;
        }
        if let Some(cluster) = &cluster {
            validate_length(n, cluster.len(), "cluster")?;
        }
        Ok(Self {
            time,
            time2,
            status,
            covariates,
            weights,
            offset,
            strata,
            cluster,
        })
    }

    pub fn n(&self) -> usize {
        self.time.len()
    }

    pub fn nvar(&self) -> usize {
        self.covariates[0].len()
    }

    /// `max(strata)`: 1 without strata.
    pub fn nstrata(&self) -> usize {
        self.strata
            .as_ref()
            .and_then(|s| s.iter().max())
            .map_or(1, |&max| max + 1)
    }
}

#[pymethods]
impl SurvregData {
    #[new]
    #[pyo3(signature = (time, status, covariates, time2=None, weights=None, offset=None, strata=None, cluster=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        time: Vec<f64>,
        status: Vec<i32>,
        covariates: Vec<Vec<f64>>,
        time2: Option<Vec<f64>>,
        weights: Option<Vec<f64>>,
        offset: Option<Vec<f64>>,
        strata: Option<Vec<usize>>,
        cluster: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        Ok(Self::try_new(
            time, status, covariates, time2, weights, offset, strata, cluster,
        )?)
    }

    fn __len__(&self) -> usize {
        self.n()
    }
}

/// A fitted `survreg` model: the components of R's `survreg` object.
#[pyclass(from_py_object)]
#[derive(Debug, Clone, PartialEq)]
pub struct SurvregFit {
    /// The location coefficients followed by one `Log(scale)` per estimated
    /// stratum, as `survreg.fit` returns them (R's `survreg` moves the
    /// `Log(scale)` entries into `scale`).  A location coefficient whose
    /// column was redundant is `NaN`, as R sets it to `NA`.
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    /// `icoef`: the coefficients of the intercept-only fit (intercept and
    /// `Log(scale)` per stratum) used for the starting values.
    #[pyo3(get)]
    pub icoef: Vec<f64>,
    /// `var`: the variance of `coefficients` (the robust variance when
    /// `robust`).
    #[pyo3(get)]
    pub variance_matrix: Vec<Vec<f64>>,
    /// `naive.var`: the model-based variance of a robust fit.
    #[pyo3(get)]
    pub naive_variance_matrix: Option<Vec<Vec<f64>>>,
    /// `loglik[2]`: the log-likelihood of the fitted model, on the scale of
    /// the original response (the transform's Jacobian is included).
    #[pyo3(get)]
    pub log_likelihood: f64,
    /// `loglik[1]`: the log-likelihood of the intercept-only model.
    #[pyo3(get)]
    pub intercept_only_log_likelihood: f64,
    /// `iter`: Newton-Raphson iterations used.
    #[pyo3(get)]
    pub iterations: usize,
    /// `false` when the fit "Ran out of iterations and did not converge"
    /// (R warns instead of failing).
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub linear_predictors: Vec<f64>,
    /// `scale`: one value per stratum (the fixed value when the scale was
    /// fixed).
    #[pyo3(get)]
    pub scale: Vec<f64>,
    /// `df`: number of parameters (`length(coefficients)`).
    #[pyo3(get)]
    pub df: usize,
    /// `df.residual`: `n - df`, negative when the design has more
    /// parameters than observations (R does not guard this either).
    #[pyo3(get)]
    pub df_residual: i64,
    /// `means`: column means of the design matrix.
    #[pyo3(get)]
    pub means: Vec<f64>,
    #[pyo3(get)]
    pub n: usize,
    /// `dist`/`parms`: the fitted distribution.
    #[pyo3(get)]
    pub distribution: SurvregDistribution,
    /// `y`: the response as fitted (interval-censored rows with a lower
    /// bound of zero are recoded as left censored for log-transformed
    /// distributions, as `survreg` does).
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub time2: Option<Vec<f64>>,
    #[pyo3(get)]
    pub status: Vec<i32>,
    /// `x`: the design matrix.
    #[pyo3(get)]
    pub covariates: Vec<Vec<f64>>,
    /// Zero-based stratum of every observation.
    #[pyo3(get)]
    pub strata: Vec<usize>,
    /// `weights`: the case weights when they were supplied.
    #[pyo3(get)]
    pub weights: Option<Vec<f64>>,
    #[pyo3(get)]
    pub offset: Vec<f64>,
    /// The cluster used for the robust variance, when one was supplied.
    #[pyo3(get)]
    pub cluster: Option<Vec<usize>>,
    /// `score`: the score vector at the solution.
    #[pyo3(get)]
    pub score: Vec<f64>,
}

impl SurvregFit {
    /// Number of location coefficients (columns of the design matrix).
    pub fn nvar(&self) -> usize {
        self.covariates[0].len()
    }

    /// Number of strata (scales).
    pub fn nstrata(&self) -> usize {
        self.scale.len()
    }

    /// Whether the design starts with R's `(Intercept)` column of ones.
    pub fn has_intercept(&self) -> bool {
        self.covariates.iter().all(|row| row[0] == 1.0)
    }

    /// `predict.survreg`; see [`predict_survreg`].
    pub fn predict(
        &self,
        newdata: Option<&SurvregNewdata<'_>>,
        predict_type: SurvregPredictType,
        se_fit: bool,
        p: &[f64],
        assign: Option<&[usize]>,
        terms: Option<&[usize]>,
    ) -> SurvivalResult<SurvregPrediction> {
        predict_survreg(self, newdata, predict_type, se_fit, p, assign, terms)
    }

    /// `residuals.survreg`; see [`residuals_survreg`].
    pub fn residuals(
        &self,
        residual_type: SurvregResidType,
        rsigma: bool,
        collapse: Option<&[usize]>,
        weighted: bool,
    ) -> SurvivalResult<SurvregResiduals> {
        residuals_survreg(self, residual_type, rsigma, collapse, weighted)
    }
}

#[pymethods]
impl SurvregFit {
    /// `predict(object, newdata, type, se.fit, p, terms)`.  `offset` and
    /// `strata` describe the rows of `newdata`; `assign` gives the term
    /// number of every design column for `type = "terms"`.
    #[pyo3(name = "predict")]
    #[pyo3(signature = (newdata=None, predict_type="response", se_fit=false, p=None, offset=None, strata=None, assign=None, terms=None))]
    #[allow(clippy::too_many_arguments)]
    fn predict_py(
        &self,
        newdata: Option<Vec<Vec<f64>>>,
        predict_type: &str,
        se_fit: bool,
        p: Option<Vec<f64>>,
        offset: Option<Vec<f64>>,
        strata: Option<Vec<usize>>,
        assign: Option<Vec<usize>>,
        terms: Option<Vec<usize>>,
    ) -> PyResult<SurvregPrediction> {
        let predict_type = SurvregPredictType::parse(predict_type)?;
        let p = p.unwrap_or_else(|| vec![0.1, 0.9]);
        let newdata = newdata.as_ref().map(|covariates| SurvregNewdata {
            covariates,
            offset: offset.as_deref(),
            strata: strata.as_deref(),
        });
        if newdata.is_none() && (offset.is_some() || strata.is_some()) {
            return Err(SurvivalError::invalid_input(
                "offset and strata describe newdata; supply newdata as well",
            )
            .into());
        }
        Ok(self.predict(
            newdata.as_ref(),
            predict_type,
            se_fit,
            &p,
            assign.as_deref(),
            terms.as_deref(),
        )?)
    }

    /// `residuals(object, type, rsigma, collapse, weighted)`.
    #[pyo3(name = "residuals")]
    #[pyo3(signature = (residual_type="response", rsigma=true, collapse=None, weighted=false))]
    fn residuals_py(
        &self,
        residual_type: &str,
        rsigma: bool,
        collapse: Option<Vec<usize>>,
        weighted: bool,
    ) -> PyResult<SurvregResiduals> {
        let residual_type = SurvregResidType::parse(residual_type)?;
        Ok(self.residuals(residual_type, rsigma, collapse.as_deref(), weighted)?)
    }

    fn __repr__(&self) -> String {
        format!(
            "SurvregFit(distribution='{}', n={}, coefficients={}, scale={:?}, loglik={:.4}, iterations={})",
            self.distribution.name,
            self.n,
            self.nvar(),
            self.scale,
            self.log_likelihood,
            self.iterations
        )
    }
}

/// The response on the fitting scale plus the likelihood correction for the
/// transform, as prepared by `survreg()` before calling `survreg.fit`.
struct FittingResponse {
    time: Vec<f64>,
    time2: Option<Vec<f64>>,
    status: Vec<i32>,
    y1: Vec<f64>,
    y2: Vec<f64>,
    /// `logcorrect`: `sum(weights * log(dtrans(y)))` over exact rows.
    logcorrect: f64,
}

fn fitting_response(
    data: &SurvregData,
    distribution: &SurvregDistribution,
    weights: &[f64],
) -> SurvivalResult<FittingResponse> {
    let n = data.n();
    let transform = distribution.transform;
    let transforms = transform != SurvregTransform::Identity;
    let mut time = data.time.clone();
    let mut time2 = data.time2.clone();
    let mut status = data.status.clone();
    // Interval censored with a lower bound of zero: convert to left censored
    // (R does this for the log-transformed distributions; its name list
    // misspells "Log logistic", which is corrected here).
    if transforms && let Some(time2) = time2.as_mut() {
        for i in 0..n {
            if status[i] == 3 && time[i] == 0.0 {
                time[i] = time2[i];
                time2[i] = 1.0;
                status[i] = 2;
            }
        }
    }
    let mut logcorrect = 0.0;
    if transforms {
        for i in (0..n).filter(|&i| status[i] == 1) {
            logcorrect += weights[i] * transform.derivative(time[i]).ln();
        }
    }
    let y1: Vec<f64> = time.iter().map(|&t| transform.apply(t)).collect();
    let y2: Vec<f64> = (0..n)
        .map(|i| match &time2 {
            Some(time2) if status[i] == 3 => transform.apply(time2[i]),
            _ => y1[i],
        })
        .collect();
    if y1.iter().chain(&y2).any(|v| !v.is_finite()) || !logcorrect.is_finite() {
        return Err(SurvivalError::invalid_input(
            "Invalid survival times for this distribution",
        ));
    }
    Ok(FittingResponse {
        time,
        time2,
        status,
        y1,
        y2,
        logcorrect,
    })
}

/// The `derfun` of `survreg.fit`: `dg` and `ddg` of every observation at
/// `eta` (per observation) and `sigma` (per observation).
fn derfun(
    distribution: &SurvregDistribution,
    response: &FittingResponse,
    eta: &[f64],
    sigma: impl Fn(usize) -> f64,
) -> (Vec<f64>, Vec<f64>) {
    (0..response.y1.len())
        .map(|i| {
            let d = survreg_deriv(
                distribution,
                response.y1[i],
                response.y2[i],
                response.status[i],
                eta[i],
                sigma(i),
            );
            (d[1], d[2])
        })
        .unzip()
}

/// The value of `survreg6`.
struct Survreg6Fit {
    /// The parameter vector as passed in, with the first `nvar2` entries
    /// updated (a trailing fixed `log(scale)` is left untouched).
    beta: Vec<f64>,
    iter: usize,
    var: Array2<f64>,
    loglik: f64,
    converged: bool,
    u: Vec<f64>,
}

/// `chsolve2` of the score against `imat`, or against `JJ` when `imat` is
/// not positive definite: the Newton (or Fisher) step.
fn newton_step(lik: &SurvregLikelihood, tol_chol: f64) -> Vec<f64> {
    let mut chol = lik.imat.clone();
    if cholesky2(&mut chol, tol_chol) < 0 {
        chol = lik.jj.clone();
        cholesky2(&mut chol, tol_chol);
    }
    let mut step = lik.u.clone();
    chsolve2(&chol, &mut step);
    step
}

/// `cholesky2` + `chinv2` of the information matrix, symmetrised, as the C
/// code does before returning `imat` as the variance.
fn invert_information(imat: &Array2<f64>, tol_chol: f64) -> SurvivalResult<Array2<f64>> {
    symmetric_inverse_via_cholesky(imat, tol_chol)
        .map(|inverse| inverse.inverse)
        .map_err(|_| {
            SurvivalError::computation(
                "the information matrix is not finite; the fit diverged (use starting estimates?)",
            )
        })
}

/// `src/survreg6.c`: Newton-Raphson with step halving.  `beta` holds the
/// starting values (`nvar` coefficients, the `log(scale)` per estimated
/// stratum, or the fixed `log(scale)` when none is estimated).
fn survreg6(
    kernel: &SurvregKernel<'_>,
    maxiter: usize,
    mut beta: Vec<f64>,
    eps: f64,
    tol_chol: f64,
) -> SurvivalResult<Survreg6Fit> {
    let nvar = kernel.nvar();
    let nstrat = kernel.nstrat;
    let nvar2 = kernel.nvar2();
    let mut newbeta = beta.clone();

    // The initial iteration step.
    let lik = kernel.evaluate(&beta);
    let mut loglik = lik.loglik;
    let mut usave = lik.u.clone();
    let step = newton_step(&lik, tol_chol);
    for i in 0..nvar2 {
        newbeta[i] = beta[i] + step[i];
    }
    // Never complain about convergence on the first step: with maxiter == 0
    // the caller gets the log-likelihood and information at `beta`.
    if maxiter == 0 {
        return Ok(Survreg6Fit {
            var: invert_information(&lik.imat, tol_chol)?,
            beta,
            iter: 0,
            loglik,
            converged: true,
            u: usave,
        });
    }

    let mut halving = 0;
    let mut newlik = kernel.evaluate(&newbeta);
    usave.clone_from(&newlik.u);
    for iter in 1..=maxiter {
        // A Newton-Raphson step gone seriously awry leaves an infinite or
        // NaN log-likelihood, score or information.
        let newlk = newlik.loglik;
        let bad = !newlk.is_finite()
            || (0..nvar2).any(|j| !newlik.imat[[j, j]].is_finite() || !newlik.u[j].is_finite());
        // Convergence: the relative change in the log-likelihood, or the
        // absolute change when the log-likelihood settles near zero.
        if !bad
            && halving == 0
            && ((1.0 - loglik / newlk).abs() <= eps || (loglik - newlk).abs() <= eps)
        {
            beta[..nvar2].copy_from_slice(&newbeta[..nvar2]);
            return Ok(Survreg6Fit {
                var: invert_information(&newlik.imat, tol_chol)?,
                beta,
                iter,
                loglik: newlk,
                converged: true,
                u: usave,
            });
        }

        if bad || newlk < loglik {
            // Not converging: back off two thirds of the way to the last
            // good point.
            halving += 1;
            for i in 0..nvar2 {
                newbeta[i] = (newbeta[i] + 2.0 * beta[i]) / 3.0;
            }
            // The sigmas are often the trouble: the prior step may have
            // shrunk one by a factor of more than 10, and halving is not
            // enough of a back away.  Make sure the new try differs from
            // the last good one by no more than log(3) ~ 1.1, first time
            // only.
            if halving == 1 {
                for i in 0..nstrat {
                    if beta[nvar + i] - newbeta[nvar + i] > 1.1 {
                        newbeta[nvar + i] = beta[nvar + i] - 1.1;
                    }
                }
            }
        } else {
            // A standard Newton-Raphson step.
            halving = 0;
            loglik = newlk;
            let step = newton_step(&newlik, tol_chol);
            beta[..nvar2].copy_from_slice(&newbeta[..nvar2]);
            for (value, delta) in newbeta.iter_mut().zip(&step) {
                *value += delta;
            }
        }
        newlik = kernel.evaluate(&newbeta);
        usave.clone_from(&newlik.u);
    }

    // Ran out of iterations.  The C code accepts the final step when it
    // improved the fit but reports the log-likelihood of the step before it;
    // otherwise it reports the information at the last good point.
    let information = if halving == 0 && newlik.loglik.is_finite() && newlik.loglik >= loglik {
        beta[..nvar2].copy_from_slice(&newbeta[..nvar2]);
        newlik.imat
    } else {
        kernel.evaluate(&beta).imat
    };
    Ok(Survreg6Fit {
        var: invert_information(&information, tol_chol)?,
        beta,
        iter: maxiter,
        loglik,
        converged: false,
        u: usave,
    })
}

/// Whether every entry of the column is 0 or 1 (`survreg.fit` leaves such
/// columns out of the rescaling).
fn is_binary_column(column: ndarray::ArrayView1<'_, f64>) -> bool {
    column.iter().all(|&v| v == 0.0 || v == 1.0)
}

/// R's `sd`: the sample standard deviation with an `n - 1` denominator.
fn sample_sd(column: ndarray::ArrayView1<'_, f64>) -> f64 {
    let n = column.len() as f64;
    let mean = column.sum() / n;
    (column.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
}

/// `coxph.wtest(t(x) %*% (wt * x), c((wt * eta + weights * dg) %*% x))$solve`:
/// the "glim" starting coefficients of `survreg.fit`.
fn glim_start(x: ArrayView2<'_, f64>, wt: &[f64], rhs_weight: &[f64], tol_chol: f64) -> Vec<f64> {
    let nvar = x.ncols();
    let mut matrix = Array2::<f64>::zeros((nvar, nvar));
    let mut rhs = vec![0.0; nvar];
    for (i, row) in x.outer_iter().enumerate() {
        for j in 0..nvar {
            rhs[j] += rhs_weight[i] * row[j];
            for k in 0..=j {
                matrix[[j, k]] += wt[i] * row[j] * row[k];
            }
        }
    }
    for j in 0..nvar {
        for k in 0..j {
            matrix[[k, j]] = matrix[[j, k]];
        }
    }
    cholesky2(&mut matrix, tol_chol);
    chsolve2(&matrix, &mut rhs);
    rhs
}

fn all_finite<'a>(values: impl IntoIterator<Item = &'a f64>) -> bool {
    values.into_iter().all(|v| v.is_finite())
}

/// `survreg(...)` for prepared inputs: fits `distribution` to `data`.
///
/// `init` are starting values (`nvar` coefficients, optionally followed by
/// the `log(scale)` per stratum); `scale > 0` fixes the scale (`0`
/// estimates it; a distribution with its own fixed scale, such as the
/// exponential, overrides it); `robust` requests the sandwich variance from
/// the dfbeta residuals, grouped by `data.cluster` when present.
pub fn survreg_fit(
    data: &SurvregData,
    distribution: &SurvregDistribution,
    init: Option<&[f64]>,
    scale: f64,
    control: &SurvregControl,
    robust: bool,
) -> SurvivalResult<SurvregFit> {
    distribution.validate()?;
    control.validate()?;
    let n = data.n();
    let nvar = data.nvar();
    let nstrata = data.nstrata();
    let eps = control.rel_tolerance;
    let tol_chol = control.toler_chol;

    let scale = distribution.scale.unwrap_or(scale);
    if !scale.is_finite() || scale < 0.0 {
        return Err(SurvivalError::invalid_input("Invalid scale value"));
    }
    if scale > 0.0 && nstrata > 1 {
        return Err(SurvivalError::invalid_input(
            "The scale argument is not valid with multiple strata",
        ));
    }
    let weights: Vec<f64> = data.weights.clone().unwrap_or_else(|| vec![1.0; n]);
    let offset: Vec<f64> = data.offset.clone().unwrap_or_else(|| vec![0.0; n]);
    let strata: Vec<usize> = data.strata.clone().unwrap_or_else(|| vec![0; n]);
    let response = fitting_response(data, distribution, &weights)?;
    // Number of variances to estimate.
    let nstrat2 = if scale > 0.0 { 0 } else { nstrata };
    let nvar2 = nvar + nstrat2;

    let x_original = Array2::from_shape_fn((n, nvar), |(i, j)| data.covariates[i][j]);
    let means: Vec<f64> = x_original
        .columns()
        .into_iter()
        .map(|column| column.sum() / n as f64)
        .collect();

    // Rescale the X matrix (more stable), but only if the first column is an
    // intercept and no starting values were given.
    let mut x = x_original.clone();
    let mut rescaled: Option<(Vec<f64>, Vec<f64>)> = None;
    if init.is_none() && x.column(0).iter().all(|&v| v == 1.0) && nvar > 1 {
        let okay: Vec<bool> = x.columns().into_iter().map(is_binary_column).collect();
        if !okay.iter().all(|&ok| ok) {
            let center: Vec<f64> = (0..nvar)
                .map(|j| if okay[j] { 0.0 } else { means[j] })
                .collect();
            let stdev: Vec<f64> = (0..nvar)
                .map(|j| if okay[j] { 1.0 } else { sample_sd(x.column(j)) })
                .collect();
            for j in 0..nvar {
                x.column_mut(j)
                    .iter_mut()
                    .for_each(|v| *v = (*v - center[j]) / stdev[j]);
            }
            rescaled = Some((center, stdev));
        }
    }

    // A good initial value of the scale is critical: fit a model with only
    // the mean and the scale first, unless the main fit is that model.
    let meanonly = nvar == 1 && x.column(0).iter().all(|&v| v == 1.0);
    let yy: Vec<f64> = (0..n)
        .map(|i| {
            if response.status[i] != 3 {
                response.y1[i]
            } else {
                (response.y1[i] + response.y2[i]) / 2.0
            }
        })
        .collect();
    let ones = Array2::<f64>::ones((n, 1));
    let fit0 = if meanonly {
        None
    } else {
        let coef = distribution.init(&yy, &weights)?;
        // init returns sigma^2; we need log(sigma), doubled for safety.
        let vars = if scale > 0.0 {
            scale.ln()
        } else {
            (4.0 * coef[1]).ln() / 2.0
        };
        let mut coef0 = vec![coef[0]];
        coef0.extend(std::iter::repeat_n(vars, nstrata));
        // A better initial value for the mean using the "glim" trick.
        let (dg, ddg) = derfun(distribution, &response, &yy, |_| vars.exp());
        let wt: Vec<f64> = ddg.iter().zip(&weights).map(|(d, w)| -d * w).collect();
        coef0[0] = (0..n)
            .map(|i| weights[i] * dg[i] + wt[i] * (yy[i] - offset[i]))
            .sum::<f64>()
            / wt.iter().sum::<f64>();
        let kernel0 = SurvregKernel {
            y1: &response.y1,
            y2: &response.y2,
            status: &response.status,
            covariates: ones.view(),
            weights: &weights,
            offset: &offset,
            strata: &strata,
            nstrat: nstrat2,
            distribution,
        };
        let fit0 = survreg6(&kernel0, 20, coef0, eps, tol_chol)?;
        if !all_finite(&fit0.beta) || !fit0.loglik.is_finite() || !all_finite(fit0.var.iter()) {
            return Err(SurvivalError::computation(
                "initial iteration failed (use starting estimates?)",
            ));
        }
        Some(fit0)
    };

    let start: Vec<f64> = match init {
        Some(init) => {
            let mut start = init.to_vec();
            validate_finite(&start, "init")?;
            if start.len() == nvar && nvar2 > nvar {
                let fit0 = fit0.as_ref().ok_or_else(|| {
                    SurvivalError::invalid_input("Wrong length for initial parameters")
                })?;
                start.extend_from_slice(&fit0.beta[1..]);
            }
            if start.len() != nvar2 {
                return Err(SurvivalError::invalid_input(
                    "Wrong length for initial parameters",
                ));
            }
            if scale > 0.0 {
                start.push(scale.ln());
            }
            start
        }
        None => {
            let vars: Vec<f64> = match &fit0 {
                Some(fit0) => fit0.beta[1..].to_vec(),
                None => {
                    let coef = distribution.init(&yy, &weights)?;
                    let value = if scale > 0.0 {
                        scale.ln()
                    } else {
                        (4.0 * coef[1]).ln() / 2.0
                    };
                    vec![value; nstrata]
                }
            };
            let eta: Vec<f64> = yy.iter().zip(&offset).map(|(y, o)| y - o).collect();
            let (dg, ddg) = derfun(distribution, &response, &yy, |i| vars[strata[i]].exp());
            let wt: Vec<f64> = ddg.iter().zip(&weights).map(|(d, w)| -d * w).collect();
            let rhs_weight: Vec<f64> = (0..n)
                .map(|i| wt[i] * eta[i] + weights[i] * dg[i])
                .collect();
            let mut start = glim_start(x.view(), &wt, &rhs_weight, tol_chol);
            start.extend(vars);
            start
        }
    };

    // The fit in earnest.
    let kernel = SurvregKernel {
        y1: &response.y1,
        y2: &response.y2,
        status: &response.status,
        covariates: x.view(),
        weights: &weights,
        offset: &offset,
        strata: &strata,
        nstrat: nstrat2,
        distribution,
    };
    let fit = survreg6(&kernel, control.iter_max, start, eps, tol_chol)?;
    let mut coefficients = fit.beta;
    coefficients.truncate(nvar2);
    let (icoef, loglik0) = match &fit0 {
        Some(fit0) => (fit0.beta.clone(), fit0.loglik),
        None => (coefficients.clone(), fit.loglik),
    };
    let linear_predictors: Vec<f64> = x
        .outer_iter()
        .zip(&offset)
        .map(|(row, o)| {
            row.iter()
                .zip(&coefficients[..nvar])
                .map(|(x, b)| x * b)
                .sum::<f64>()
                + o
        })
        .collect();

    let mut var = fit.var;
    if let Some((center, stdev)) = rescaled {
        // Undo the rescaling: coef <- vtemp %*% coef, var <- vtemp var vtemp'.
        let mut vtemp = Array2::<f64>::eye(nvar2);
        for j in 0..nvar {
            vtemp[[j, j]] = 1.0 / stdev[j];
        }
        for j in 1..nvar {
            vtemp[[0, j]] = -center[j] / stdev[j];
        }
        let coef =
            Array2::from_shape_vec((nvar2, 1), coefficients.clone()).expect("nvar2 coefficients");
        coefficients = vtemp.dot(&coef).column(0).to_vec();
        var = vtemp.dot(&var).dot(&vtemp.t());
    }
    let variance_matrix: Vec<Vec<f64>> = var.outer_iter().map(|row| row.to_vec()).collect();

    let scales: Vec<f64> = if scale > 0.0 {
        vec![scale]
    } else {
        coefficients[nvar..].iter().map(|v| v.exp()).collect()
    };
    let robust = robust || data.cluster.is_some();
    let mut fit = SurvregFit {
        coefficients,
        icoef,
        variance_matrix,
        naive_variance_matrix: None,
        log_likelihood: fit.loglik + response.logcorrect,
        intercept_only_log_likelihood: loglik0 + response.logcorrect,
        iterations: fit.iter,
        converged: fit.converged,
        linear_predictors,
        scale: scales,
        df: nvar2,
        df_residual: n as i64 - nvar2 as i64,
        means,
        n,
        distribution: distribution.clone(),
        time: response.time,
        time2: response.time2,
        status: response.status,
        covariates: data.covariates.clone(),
        strata,
        weights: data.weights.clone(),
        offset,
        cluster: data.cluster.clone(),
        score: fit.u,
    };

    if robust {
        // var <- crossprod(residuals(fit, "dfbeta", weighted = TRUE, collapse = cluster))
        let dfbeta = fit.residuals(
            SurvregResidType::Dfbeta,
            true,
            data.cluster.as_deref(),
            true,
        )?;
        let width = nvar2;
        let mut sandwich = vec![vec![0.0; width]; width];
        for row in &dfbeta.values {
            for j in 0..width {
                for k in 0..width {
                    sandwich[j][k] += row[j] * row[k];
                }
            }
        }
        fit.naive_variance_matrix = Some(std::mem::replace(&mut fit.variance_matrix, sandwich));
    }

    // Set singular coefficients to NA; purposely not done until the
    // residuals have been computed.
    let singular_variance = fit
        .naive_variance_matrix
        .as_deref()
        .unwrap_or(&fit.variance_matrix);
    let singular: Vec<bool> = (0..nvar).map(|j| singular_variance[j][j] == 0.0).collect();
    for (coefficient, &is_singular) in fit.coefficients.iter_mut().zip(&singular) {
        if is_singular {
            *coefficient = f64::NAN;
        }
    }
    Ok(fit)
}

/// `survreg(Surv(time, time2, status) ~ x, weights, offset, strata, dist,
/// init, scale, parms, control)` on prepared inputs.
///
/// `covariates` is the full design matrix (include a column of ones for the
/// intercept).  `distribution` is an R distribution name (`weibull`,
/// `exponential`, `rayleigh`, `extreme`, `gaussian`, `logistic`,
/// `lognormal`/`loggaussian`, `loglogistic`, `t`), `distribution_parameter`
/// the degrees of freedom of the `t` family, `fixed_scale` R's `scale`
/// argument (`None` estimates it).  A user-defined [`SurvregDistribution`]
/// goes through [`survreg_fit`] instead.
#[pyfunction]
#[pyo3(signature = (time, status, covariates, weights=None, offsets=None, initial_beta=None, strata=None, distribution=None, max_iter=None, eps=None, tol_chol=None, time2=None, fixed_scale=None, distribution_parameter=None))]
#[allow(clippy::too_many_arguments)]
pub fn survreg(
    time: Vec<f64>,
    status: Vec<f64>,
    covariates: Vec<Vec<f64>>,
    weights: Option<Vec<f64>>,
    offsets: Option<Vec<f64>>,
    initial_beta: Option<Vec<f64>>,
    strata: Option<Vec<usize>>,
    distribution: Option<&str>,
    max_iter: Option<usize>,
    eps: Option<f64>,
    tol_chol: Option<f64>,
    time2: Option<Vec<f64>>,
    fixed_scale: Option<f64>,
    distribution_parameter: Option<f64>,
) -> PyResult<SurvregFit> {
    let parms: Option<Vec<f64>> = distribution_parameter.map(|df| vec![df]);
    let distribution =
        SurvregDistribution::from_name(distribution.unwrap_or("weibull"), parms.as_deref())?;
    let status: Vec<i32> = status
        .iter()
        .enumerate()
        .map(|(index, &code)| {
            if code.fract() == 0.0 && (0.0..=3.0).contains(&code) {
                Ok(code as i32)
            } else {
                Err(SurvivalError::invalid_input(format!(
                    "status must contain the codes 0, 1, 2 or 3; got {code} at index {index}"
                )))
            }
        })
        .collect::<SurvivalResult<_>>()?;
    let data = SurvregData::try_new(
        time, status, covariates, time2, weights, offsets, strata, None,
    )?;
    let defaults = SurvregControl::default();
    let control = SurvregControl {
        iter_max: max_iter.unwrap_or(defaults.iter_max),
        rel_tolerance: eps.unwrap_or(defaults.rel_tolerance),
        toler_chol: tol_chol.unwrap_or(defaults.toler_chol),
    };
    Ok(survreg_fit(
        &data,
        &distribution,
        initial_beta.as_deref(),
        fixed_scale.unwrap_or(0.0),
        &control,
        false,
    )?)
}

/// `survreg.fit` with typed inputs: [`survreg_fit`] for Python, accepting a
/// user-defined [`SurvregDistribution`].
#[pyfunction(name = "survreg_fit")]
#[pyo3(signature = (data, distribution, init=None, scale=0.0, control=None, robust=None))]
pub fn survreg_fit_py(
    data: &SurvregData,
    distribution: &SurvregDistribution,
    init: Option<Vec<f64>>,
    scale: f64,
    control: Option<SurvregControl>,
    robust: Option<bool>,
) -> PyResult<SurvregFit> {
    Ok(survreg_fit(
        data,
        distribution,
        init.as_deref(),
        scale,
        &control.unwrap_or_default(),
        robust.unwrap_or(false),
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regression::survreg_distributions::{SurvregFamily, SurvregTransform};

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
            "expected {expected}, got {actual}"
        );
    }

    /// The first 12 rows of the `ovarian` data: futime, fustat, age.
    fn ovarian() -> SurvregData {
        let time = vec![
            59.0, 115.0, 156.0, 421.0, 431.0, 448.0, 464.0, 475.0, 477.0, 563.0, 638.0, 744.0,
        ];
        let status = vec![1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0];
        let age = [
            72.3315, 74.4932, 66.4658, 53.3644, 50.3397, 56.4301, 56.9370, 59.8548, 64.1753,
            55.1781, 56.7562, 50.1096,
        ];
        let covariates = age.iter().map(|&a| vec![1.0, a]).collect();
        SurvregData::try_new(time, status, covariates, None, None, None, None, None).unwrap()
    }

    #[test]
    fn control_defaults_are_r_defaults() {
        let control = SurvregControl::default();
        assert_eq!(control.iter_max, 30);
        assert_eq!(control.rel_tolerance, 1e-9);
        assert_eq!(control.toler_chol, 1e-10);
        assert!(SurvregControl::new(30, 0.0, 1e-10).is_err());
    }

    #[test]
    fn data_validation_rejects_bad_inputs() {
        let base = ovarian();
        let n = base.n();
        assert!(
            SurvregData::try_new(
                base.time.clone(),
                vec![4; n],
                base.covariates.clone(),
                None,
                None,
                None,
                None,
                None
            )
            .is_err()
        );
        assert!(
            SurvregData::try_new(
                base.time.clone(),
                vec![3; n],
                base.covariates.clone(),
                None,
                None,
                None,
                None,
                None
            )
            .is_err(),
            "interval rows need time2"
        );
        assert!(
            SurvregData::try_new(
                base.time.clone(),
                base.status.clone(),
                base.covariates.clone(),
                None,
                Some(vec![0.0; n]),
                None,
                None,
                None
            )
            .is_err(),
            "weights must be positive"
        );
        assert!(
            SurvregData::try_new(
                base.time.clone(),
                base.status.clone(),
                vec![vec![]; n],
                None,
                None,
                None,
                None,
                None
            )
            .is_err()
        );
        assert_eq!(base.nstrata(), 1);
    }

    #[test]
    fn weibull_fit_matches_r() {
        // survreg(Surv(futime, fustat) ~ age, ovarian[1:12, ])
        let fit = survreg_fit(
            &ovarian(),
            &SurvregDistribution::from_name("weibull", None).unwrap(),
            None,
            0.0,
            &SurvregControl::default(),
            false,
        )
        .unwrap();
        assert!(fit.converged);
        assert_eq!(fit.coefficients.len(), 3);
        assert_eq!(fit.df, 3);
        assert_eq!(fit.df_residual, 9);
        assert_eq!(fit.scale.len(), 1);
        assert_eq!(fit.icoef.len(), 2);
        assert_eq!(fit.variance_matrix.len(), 3);
        assert!(fit.log_likelihood > fit.intercept_only_log_likelihood);
        // Score is zero at the solution.
        assert!(fit.score.iter().all(|s| s.abs() < 1e-6));
        // The linear predictor reproduces x %*% coef on the original scale.
        for (i, lp) in fit.linear_predictors.iter().enumerate() {
            let expected = fit.coefficients[0] + fit.coefficients[1] * fit.covariates[i][1];
            assert_close(*lp, expected, 1e-10);
        }
        let mean_age = fit.covariates.iter().map(|row| row[1]).sum::<f64>() / 12.0;
        assert_close(fit.means[1], mean_age, 1e-12);
    }

    #[test]
    fn exponential_fixes_the_scale_at_one() {
        let fit = survreg_fit(
            &ovarian(),
            &SurvregDistribution::from_name("exponential", None).unwrap(),
            None,
            0.0,
            &SurvregControl::default(),
            false,
        )
        .unwrap();
        assert_eq!(fit.scale, vec![1.0]);
        assert_eq!(fit.coefficients.len(), 2);
        assert_eq!(fit.variance_matrix.len(), 2);
        assert_eq!(fit.icoef, vec![fit.icoef[0], 0.0]);
        assert_eq!(fit.df, 2);
    }

    #[test]
    fn fixed_scale_agrees_with_the_fixed_distribution() {
        let data = ovarian();
        let weibull = SurvregDistribution::from_name("weibull", None).unwrap();
        let fixed = survreg_fit(
            &data,
            &weibull,
            None,
            1.0,
            &SurvregControl::default(),
            false,
        )
        .unwrap();
        let exponential = survreg_fit(
            &data,
            &SurvregDistribution::from_name("exponential", None).unwrap(),
            None,
            0.0,
            &SurvregControl::default(),
            false,
        )
        .unwrap();
        for (a, b) in fixed.coefficients.iter().zip(&exponential.coefficients) {
            assert_close(*a, *b, 1e-12);
        }
        assert_close(fixed.log_likelihood, exponential.log_likelihood, 1e-12);
    }

    #[test]
    fn user_starting_values_are_honoured() {
        let data = ovarian();
        let weibull = SurvregDistribution::from_name("weibull", None).unwrap();
        let reference = survreg_fit(
            &data,
            &weibull,
            None,
            0.0,
            &SurvregControl::default(),
            false,
        )
        .unwrap();
        let from_init = survreg_fit(
            &data,
            &weibull,
            Some(&[7.0, 0.0]),
            0.0,
            &SurvregControl::default(),
            false,
        )
        .unwrap();
        for (a, b) in reference.coefficients.iter().zip(&from_init.coefficients) {
            assert_close(*a, *b, 1e-6);
        }
        // nvar2 starting values are taken as they are; other lengths are
        // "Wrong length for initial parameters".
        assert!(
            survreg_fit(
                &data,
                &weibull,
                Some(&[7.0, 0.0, -0.3]),
                0.0,
                &SurvregControl::default(),
                false
            )
            .is_ok()
        );
        assert!(
            survreg_fit(
                &data,
                &weibull,
                Some(&[7.0]),
                0.0,
                &SurvregControl::default(),
                false
            )
            .is_err()
        );
        assert!(
            survreg_fit(
                &data,
                &weibull,
                Some(&[7.0, 0.0, 1.0, 2.0]),
                0.0,
                &SurvregControl::default(),
                false
            )
            .is_err()
        );
    }

    #[test]
    fn zero_iterations_return_the_starting_point() {
        let data = ovarian();
        let weibull = SurvregDistribution::from_name("weibull", None).unwrap();
        let control = SurvregControl {
            iter_max: 0,
            ..SurvregControl::default()
        };
        let fit = survreg_fit(
            &data,
            &weibull,
            Some(&[6.0, 0.1, -0.2]),
            0.0,
            &control,
            false,
        )
        .unwrap();
        assert_eq!(fit.iterations, 0);
        assert_eq!(fit.coefficients, vec![6.0, 0.1, -0.2]);
    }

    #[test]
    fn non_positive_times_are_accepted_by_untransformed_distributions() {
        let time = vec![0.0, 0.7, 1.4, 0.0, 2.0, 0.0, 3.3, 1.0];
        let status = vec![2, 1, 1, 2, 1, 2, 1, 1];
        let covariates: Vec<Vec<f64>> = (0..8).map(|i| vec![1.0, i as f64]).collect();
        let data =
            SurvregData::try_new(time, status, covariates, None, None, None, None, None).unwrap();
        let gaussian = SurvregDistribution::from_name("gaussian", None).unwrap();
        let fit = survreg_fit(
            &data,
            &gaussian,
            None,
            0.0,
            &SurvregControl::default(),
            false,
        )
        .unwrap();
        assert!(fit.converged);
        let weibull = SurvregDistribution::from_name("weibull", None).unwrap();
        let err = survreg_fit(
            &data,
            &weibull,
            None,
            0.0,
            &SurvregControl::default(),
            false,
        )
        .unwrap_err();
        assert!(err.to_string().contains("Invalid survival times"));
    }

    #[test]
    fn interval_lower_bound_of_zero_becomes_left_censored() {
        let time = vec![0.0, 1.0, 2.0, 1.5, 3.0, 2.5];
        let time2 = vec![2.0, 3.0, 2.0, 1.5, 4.0, 2.5];
        let status = vec![3, 3, 1, 1, 3, 0];
        let covariates = vec![vec![1.0]; 6];
        let data = SurvregData::try_new(
            time,
            status,
            covariates,
            Some(time2),
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let weibull = SurvregDistribution::from_name("weibull", None).unwrap();
        let fit = survreg_fit(
            &data,
            &weibull,
            None,
            0.0,
            &SurvregControl::default(),
            false,
        )
        .unwrap();
        assert_eq!(fit.status[0], 2);
        assert_eq!(fit.time[0], 2.0);
        assert_eq!(fit.time2.as_ref().unwrap()[0], 1.0);
        assert!(fit.converged);
    }

    #[test]
    fn strata_estimate_one_scale_per_stratum() {
        let mut data = ovarian();
        data.strata = Some((0..data.n()).map(|i| i % 2).collect());
        let weibull = SurvregDistribution::from_name("weibull", None).unwrap();
        let fit = survreg_fit(
            &data,
            &weibull,
            None,
            0.0,
            &SurvregControl::default(),
            false,
        )
        .unwrap();
        assert_eq!(fit.scale.len(), 2);
        assert_eq!(fit.coefficients.len(), 4);
        assert_eq!(fit.icoef.len(), 3);
        assert!(
            survreg_fit(
                &data,
                &weibull,
                None,
                1.0,
                &SurvregControl::default(),
                false
            )
            .is_err()
        );
    }

    #[test]
    fn robust_variance_is_the_dfbeta_crossproduct() {
        let mut data = ovarian();
        data.cluster = Some((0..data.n()).map(|i| i / 3).collect());
        let weibull = SurvregDistribution::from_name("weibull", None).unwrap();
        let fit =
            survreg_fit(&data, &weibull, None, 0.0, &SurvregControl::default(), true).unwrap();
        let naive = fit.naive_variance_matrix.as_ref().unwrap();
        assert_eq!(naive.len(), 3);
        let dfbeta = fit
            .residuals(
                SurvregResidType::Dfbeta,
                true,
                data.cluster.as_deref(),
                true,
            )
            .unwrap();
        assert_eq!(dfbeta.values.len(), 4);
        for (j, actual_row) in fit.variance_matrix.iter().enumerate() {
            for (k, actual) in actual_row.iter().enumerate() {
                let expected: f64 = dfbeta.values.iter().map(|row| row[j] * row[k]).sum();
                assert_close(*actual, expected, 1e-12);
            }
        }
    }

    #[test]
    fn redundant_columns_get_nan_coefficients() {
        let mut data = ovarian();
        for row in data.covariates.iter_mut() {
            let age = row[1];
            row.push(2.0 * age);
        }
        let weibull = SurvregDistribution::from_name("weibull", None).unwrap();
        let fit = survreg_fit(
            &data,
            &weibull,
            None,
            0.0,
            &SurvregControl::default(),
            false,
        )
        .unwrap();
        assert!(fit.coefficients[2].is_nan());
        assert!(fit.coefficients[1].is_finite());
        assert_eq!(fit.variance_matrix[2][2], 0.0);
    }

    #[test]
    fn saturated_designs_report_a_negative_df_residual() {
        // survreg(Surv(t, s) ~ x1 + x2, data.frame(t = 1:3, s = 1, x1 = ..., x2 = ...))
        // has 4 parameters for 3 observations; R returns df.residual = -1
        // rather than failing.
        let data = SurvregData::try_new(
            vec![1.0, 2.0, 3.0],
            vec![1, 1, 1],
            vec![
                vec![1.0, 0.5, 2.0],
                vec![1.0, 1.5, 1.0],
                vec![1.0, 2.5, 4.0],
            ],
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let fit = survreg_fit(
            &data,
            &SurvregDistribution::from_name("weibull", None).unwrap(),
            None,
            0.0,
            &SurvregControl::default(),
            false,
        )
        .unwrap();
        assert_eq!(fit.n, 3);
        assert_eq!(fit.df, 4);
        assert_eq!(fit.df_residual, -1);
    }

    #[test]
    fn custom_distribution_matches_its_builtin_twin() {
        let data = ovarian();
        let builtin = SurvregDistribution::from_name("lognormal", None).unwrap();
        let custom = SurvregDistribution::custom(
            "mine",
            SurvregFamily::Gaussian,
            SurvregTransform::Log,
            None,
            None,
        );
        let a = survreg_fit(
            &data,
            &builtin,
            None,
            0.0,
            &SurvregControl::default(),
            false,
        )
        .unwrap();
        let b = survreg_fit(&data, &custom, None, 0.0, &SurvregControl::default(), false).unwrap();
        assert_eq!(a.coefficients, b.coefficients);
        assert_eq!(a.log_likelihood, b.log_likelihood);
        assert_eq!(b.distribution.name, "mine");
    }

    #[test]
    fn python_entry_point_parses_names_and_codes() {
        let data = ovarian();
        let fit = survreg(
            data.time.clone(),
            data.status.iter().map(|&s| f64::from(s)).collect(),
            data.covariates.clone(),
            None,
            None,
            None,
            None,
            Some("t"),
            None,
            None,
            None,
            None,
            None,
            Some(6.0),
        )
        .unwrap();
        assert_eq!(fit.distribution.parms, vec![6.0]);
        assert!(
            survreg(
                data.time.clone(),
                vec![0.5; data.n()],
                data.covariates.clone(),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .is_err()
        );
    }
}
