#![allow(clippy::redundant_closure)]
use crate::constants::{CHOLESKY_TOL, CONVERGENCE_EPSILON, DEFAULT_MAX_ITER, NEAR_ZERO_MATRIX};
use crate::regression::survregc1::{SurvivalDist, survregc1};
use crate::utilities::matrix::cholesky_solve;
use ndarray::{Array1, Array2, ArrayView1};
use pyo3::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct SurvregConfig {
    #[pyo3(get, set)]
    pub max_iter: usize,

    #[pyo3(get, set)]
    pub eps: f64,

    #[pyo3(get, set)]
    pub tol_chol: f64,

    #[pyo3(get, set)]
    pub distribution: DistributionType,
}

#[pymethods]
impl SurvregConfig {
    #[new]
    #[pyo3(signature = (distribution=None, max_iter=None, eps=None, tol_chol=None))]
    fn new(
        distribution: Option<DistributionType>,
        max_iter: Option<usize>,
        eps: Option<f64>,
        tol_chol: Option<f64>,
    ) -> Self {
        Self {
            distribution: distribution.unwrap_or(DistributionType::ExtremeValue),
            max_iter: max_iter.unwrap_or(DEFAULT_MAX_ITER),
            eps: eps.unwrap_or(CONVERGENCE_EPSILON),
            tol_chol: tol_chol.unwrap_or(CHOLESKY_TOL),
        }
    }
}

impl Default for SurvregConfig {
    fn default() -> Self {
        Self {
            max_iter: DEFAULT_MAX_ITER,
            eps: CONVERGENCE_EPSILON,
            tol_chol: CHOLESKY_TOL,
            distribution: DistributionType::ExtremeValue,
        }
    }
}

impl SurvregConfig {
    pub fn create(
        distribution: Option<DistributionType>,
        max_iter: Option<usize>,
        eps: Option<f64>,
        tol_chol: Option<f64>,
    ) -> Self {
        Self {
            distribution: distribution.unwrap_or(DistributionType::ExtremeValue),
            max_iter: max_iter.unwrap_or(DEFAULT_MAX_ITER),
            eps: eps.unwrap_or(CONVERGENCE_EPSILON),
            tol_chol: tol_chol.unwrap_or(CHOLESKY_TOL),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct SurvivalFit {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    #[pyo3(get)]
    pub iterations: usize,
    #[pyo3(get)]
    pub variance_matrix: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub convergence_flag: i32,
    #[pyo3(get)]
    pub score_vector: Vec<f64>,
}
struct LikelihoodInput<'a> {
    n: usize,
    nvar: usize,
    nstrat: usize,
    beta: &'a [f64],
    distribution: &'a DistributionType,
    strata: &'a [usize],
    offsets: &'a Array1<f64>,
    time1: &'a ArrayView1<'a, f64>,
    time2: Option<&'a ArrayView1<'a, f64>>,
    status: &'a ArrayView1<'a, f64>,
    weights: &'a Array1<f64>,
    covariates: &'a Array2<f64>,
}

struct LikelihoodOutput<'a> {
    imat: &'a mut Array2<f64>,
    jj: &'a mut Array2<f64>,
    u: &'a mut Array1<f64>,
}

fn calculate_likelihood(
    input: &LikelihoodInput<'_>,
    output: &mut LikelihoodOutput<'_>,
) -> Result<f64, Box<dyn std::error::Error>> {
    let n = input.n;
    let nvar = input.nvar;
    let nstrat = input.nstrat;
    let beta = input.beta;
    let distribution = input.distribution;
    let strata = input.strata;
    let offsets = input.offsets;
    let time1 = input.time1;
    let time2 = input.time2;
    let status = input.status;
    let weights = input.weights;
    let covariates = input.covariates;
    let imat = &mut *output.imat;
    let jj = &mut *output.jj;
    let u = &mut *output.u;
    let dist = match distribution {
        DistributionType::ExtremeValue => SurvivalDist::ExtremeValue,
        DistributionType::Logistic => SurvivalDist::Logistic,
        DistributionType::Gaussian => SurvivalDist::Gaussian,
        DistributionType::Weibull => SurvivalDist::Weibull,
        DistributionType::LogNormal => SurvivalDist::LogNormal,
    };
    let strat_vec: Vec<i32> = strata.iter().map(|&s| (s + 1) as i32).collect();
    let strat_arr = Array1::from_vec(strat_vec);
    let status_vec: Vec<i32> = status.iter().map(|&s| s as i32).collect();
    let status_arr = Array1::from_vec(status_vec);
    let beta_arr = Array1::from_vec(beta.to_vec());
    let frail_arr = Array1::from_vec(vec![0i32; n]);
    let nvar2 = nvar + nstrat;
    let result = survregc1(
        n,
        nvar,
        nstrat,
        false,
        &beta_arr.view(),
        dist,
        &strat_arr.view(),
        &offsets.view(),
        time1,
        time2,
        &status_arr.view(),
        &weights.view(),
        &covariates.view(),
        0,
        &frail_arr.view(),
    )?;
    for i in 0..nvar2.min(u.len()) {
        if i < result.u.len() {
            u[i] = result.u[i];
        }
    }
    for i in 0..nvar2.min(imat.nrows()) {
        for j in 0..nvar2.min(imat.ncols()) {
            if i < result.imat.nrows() && j < result.imat.ncols() {
                imat[[i, j]] = -result.imat[[i, j]];
            }
        }
    }
    for i in 0..nvar2.min(jj.nrows()) {
        for j in 0..nvar2.min(jj.ncols()) {
            if i < result.jj.nrows() && j < result.jj.ncols() {
                jj[[i, j]] = result.jj[[i, j]];
            }
        }
    }
    Ok(result.loglik)
}
fn check_convergence(old: f64, new: f64, eps: f64) -> bool {
    (1.0 - new / old).abs() <= eps || (old - new).abs() <= eps
}
fn adjust_strata(newbeta: &mut [f64], beta: &[f64], nvar: usize, nstrat: usize) {
    for i in 0..nstrat {
        let idx = nvar + i;
        if beta[idx] - newbeta[idx] > 1.1 {
            newbeta[idx] = beta[idx] - 1.1;
        }
    }
}
fn calculate_variance_matrix(
    imat: Array2<f64>,
    _nvar2: usize,
    _tol_chol: f64,
) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    use crate::utilities::matrix::matrix_inverse;
    if imat.nrows() == 0 || imat.ncols() == 0 {
        return Ok(imat);
    }
    let max_val = imat.iter().map(|&x| x.abs()).fold(0.0f64, f64::max);
    if max_val < NEAR_ZERO_MATRIX {
        return Ok(imat);
    }
    match matrix_inverse(&imat) {
        Some(inv) => Ok(inv),
        None => Ok(imat),
    }
}
#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass]
pub enum DistributionType {
    #[pyo3(name = "extreme_value")]
    ExtremeValue,
    #[pyo3(name = "logistic")]
    Logistic,
    #[pyo3(name = "gaussian")]
    Gaussian,
    #[pyo3(name = "weibull")]
    Weibull,
    #[pyo3(name = "lognormal")]
    LogNormal,
}

#[derive(Debug, Clone, Default)]
#[pyclass]
pub struct SurvregOptions {
    #[pyo3(get, set)]
    pub weights: Option<Vec<f64>>,
    #[pyo3(get, set)]
    pub offsets: Option<Vec<f64>>,
    #[pyo3(get, set)]
    pub initial_beta: Option<Vec<f64>>,
    #[pyo3(get, set)]
    pub strata: Option<Vec<usize>>,
    #[pyo3(get, set)]
    pub distribution: Option<String>,
    #[pyo3(get, set)]
    pub max_iter: Option<usize>,
    #[pyo3(get, set)]
    pub eps: Option<f64>,
    #[pyo3(get, set)]
    pub tol_chol: Option<f64>,
}

#[pymethods]
impl SurvregOptions {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_weights(mut self_: PyRefMut<'_, Self>, weights: Vec<f64>) -> PyRefMut<'_, Self> {
        self_.weights = Some(weights);
        self_
    }

    pub fn with_offsets(mut self_: PyRefMut<'_, Self>, offsets: Vec<f64>) -> PyRefMut<'_, Self> {
        self_.offsets = Some(offsets);
        self_
    }

    pub fn with_initial_beta(
        mut self_: PyRefMut<'_, Self>,
        initial_beta: Vec<f64>,
    ) -> PyRefMut<'_, Self> {
        self_.initial_beta = Some(initial_beta);
        self_
    }

    pub fn with_strata(mut self_: PyRefMut<'_, Self>, strata: Vec<usize>) -> PyRefMut<'_, Self> {
        self_.strata = Some(strata);
        self_
    }

    pub fn with_distribution(
        mut self_: PyRefMut<'_, Self>,
        distribution: String,
    ) -> PyRefMut<'_, Self> {
        self_.distribution = Some(distribution);
        self_
    }

    pub fn with_max_iter(mut self_: PyRefMut<'_, Self>, max_iter: usize) -> PyRefMut<'_, Self> {
        self_.max_iter = Some(max_iter);
        self_
    }

    pub fn with_eps(mut self_: PyRefMut<'_, Self>, eps: f64) -> PyRefMut<'_, Self> {
        self_.eps = Some(eps);
        self_
    }

    pub fn with_tol_chol(mut self_: PyRefMut<'_, Self>, tol_chol: f64) -> PyRefMut<'_, Self> {
        self_.tol_chol = Some(tol_chol);
        self_
    }
}

/// Fit a parametric survival regression model.
///
/// Parameters
/// ----------
/// time : array-like
///     Survival/censoring times.
/// status : array-like
///     Event indicator (1=event, 0=censored).
/// covariates : list of lists
///     Covariate matrix (n_obs x n_vars).
/// weights : array-like, optional
///     Case weights.
/// offsets : array-like, optional
///     Offset terms for the linear predictor.
/// initial_beta : array-like, optional
///     Starting values for coefficients.
/// strata : array-like, optional
///     Stratum indicators for stratified analysis.
/// distribution : str, optional
///     Error distribution: "weibull" (default), "lognormal", "loglogistic", "gaussian", "exponential".
/// max_iter : int, optional
///     Maximum iterations (default 30).
/// eps : float, optional
///     Convergence tolerance (default 1e-6).
/// tol_chol : float, optional
///     Cholesky tolerance (default 1e-10).
///
/// Returns
/// -------
/// SurvivalFit
///     Object with: coefficients, std_errors, variance_matrix, log_likelihood, convergence info.
#[pyfunction]
#[pyo3(signature = (time, status, covariates, weights=None, offsets=None, initial_beta=None, strata=None, distribution=None, max_iter=None, eps=None, tol_chol=None))]
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
) -> PyResult<SurvivalFit> {
    let dist_type = distribution.map(|s| match s {
        "weibull" => DistributionType::Weibull,
        "exponential" | "extreme_value" => DistributionType::ExtremeValue,
        "gaussian" | "normal" => DistributionType::Gaussian,
        "logistic" => DistributionType::Logistic,
        "lognormal" => DistributionType::LogNormal,
        _ => DistributionType::ExtremeValue,
    });
    let config = SurvregConfig::create(dist_type, max_iter, eps, tol_chol);
    let n = time.len();
    if status.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Length mismatch: time has {} elements but status has {}. Both must have the same length.",
            n,
            status.len()
        )));
    }
    let nvar = if !covariates.is_empty() {
        covariates[0].len()
    } else {
        0
    };
    if !covariates.is_empty() && covariates.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Length mismatch: time has {} observations but covariates has {} rows. \
             Covariates should be a list of {} rows, each with {} covariate values.",
            n,
            covariates.len(),
            n,
            nvar
        )));
    }
    let weights = weights.unwrap_or_else(|| vec![1.0; n]);
    let offsets = offsets.unwrap_or_else(|| vec![0.0; n]);
    let strata = strata.unwrap_or_else(|| vec![0; n]);
    let nstrat = if strata.is_empty() {
        1
    } else {
        strata.iter().max().copied().unwrap_or(0) + 1
    };
    let initial_beta = initial_beta.unwrap_or_else(|| vec![0.0; nvar + nstrat]);
    let y = {
        let mut y_data = Vec::new();
        for i in 0..n {
            y_data.push(vec![time[i], status[i]]);
        }
        Array2::from_shape_vec((n, 2), y_data.into_iter().flatten().collect())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?
    };
    let cov_array = if nvar > 0 {
        let flat: Vec<f64> = covariates.into_iter().flatten().collect();
        let temp = Array2::from_shape_vec((n, nvar), flat)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        temp.t().to_owned()
    } else {
        Array2::zeros((0, n))
    };
    let weights_arr = Array1::from_vec(weights);
    let offsets_arr = Array1::from_vec(offsets);
    let result = compute_survreg(
        config.max_iter,
        nvar,
        &y,
        &cov_array,
        &weights_arr,
        &offsets_arr,
        initial_beta,
        nstrat,
        &strata,
        config.eps,
        config.tol_chol,
        config.distribution,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
    let variance_matrix = result
        .variance_matrix
        .outer_iter()
        .map(|row| row.iter().cloned().collect())
        .collect();
    Ok(SurvivalFit {
        coefficients: result.coefficients,
        iterations: result.iterations,
        variance_matrix,
        log_likelihood: result.log_likelihood,
        convergence_flag: result.convergence_flag,
        score_vector: result.score_vector,
    })
}
#[allow(clippy::too_many_arguments)]
fn compute_survreg(
    max_iter: usize,
    nvar: usize,
    y: &Array2<f64>,
    covariates: &Array2<f64>,
    weights: &Array1<f64>,
    offsets: &Array1<f64>,
    mut beta: Vec<f64>,
    nstrat: usize,
    strata: &[usize],
    eps: f64,
    tol_chol: f64,
    distribution: DistributionType,
) -> Result<SurvivalFitComputed, Box<dyn std::error::Error>> {
    let n = y.nrows();
    let ny = y.ncols();
    let nvar2 = nvar + nstrat;
    let mut imat = Array2::zeros((nvar2, nvar2));
    let mut jj = Array2::zeros((nvar2, nvar2));
    let mut u = Array1::zeros(nvar2);
    let mut newbeta = beta.clone();
    let mut usave = Array1::zeros(nvar2);
    let time1_vec: Vec<f64> = y.column(0).iter().cloned().collect();
    let status_vec: Vec<f64> = if ny == 2 {
        y.column(1).iter().cloned().collect()
    } else {
        y.column(2).iter().cloned().collect()
    };
    let time2_vec: Option<Vec<f64>> = if ny == 3 {
        Some(y.column(1).iter().cloned().collect())
    } else {
        None
    };
    let time1_arr = Array1::from_vec(time1_vec);
    let status_arr = Array1::from_vec(status_vec);
    let time2_arr = time2_vec.map(|v| Array1::from_vec(v));
    let time1 = time1_arr.view();
    let status = status_arr.view();
    let time2_view: Option<ArrayView1<f64>> = time2_arr.as_ref().map(|v| v.view());
    let input = LikelihoodInput {
        n,
        nvar,
        nstrat,
        beta: &beta,
        distribution: &distribution,
        strata,
        offsets,
        time1: &time1,
        time2: time2_view.as_ref(),
        status: &status,
        weights,
        covariates,
    };
    let mut output = LikelihoodOutput {
        imat: &mut imat,
        jj: &mut jj,
        u: &mut u,
    };
    let mut loglik = calculate_likelihood(&input, &mut output)?;
    usave.assign(&u);
    let mut iter = 0;
    let mut halving = 0;
    while iter < max_iter {
        let chol_result = cholesky_solve(&imat, &u, tol_chol);
        let delta = match chol_result {
            Ok(d) => d,
            Err(_) => cholesky_solve(&jj, &u, tol_chol)?,
        };
        newbeta
            .iter_mut()
            .zip(beta.iter().zip(delta.iter()))
            .for_each(|(nb, (b, d))| *nb = b + d);
        let new_input = LikelihoodInput {
            n,
            nvar,
            nstrat,
            beta: &newbeta,
            distribution: &distribution,
            strata,
            offsets,
            time1: &time1,
            time2: time2_view.as_ref(),
            status: &status,
            weights,
            covariates,
        };
        let mut new_output = LikelihoodOutput {
            imat: &mut imat,
            jj: &mut jj,
            u: &mut u,
        };
        let newlik = calculate_likelihood(&new_input, &mut new_output)?;
        if check_convergence(loglik, newlik, eps) && halving == 0 {
            loglik = newlik;
            std::mem::swap(&mut beta, &mut newbeta);
            iter += 1;
            break;
        }
        if newlik.is_nan() || newlik < loglik {
            halving += 1;
            newbeta
                .iter_mut()
                .zip(&beta)
                .for_each(|(nb, b)| *nb = (*nb + 2.0 * b) / 3.0);
            if halving == 1 {
                adjust_strata(&mut newbeta, &beta, nvar, nstrat);
            }
        } else {
            halving = 0;
            loglik = newlik;
            std::mem::swap(&mut beta, &mut newbeta);
        }
        iter += 1;
    }
    let converged = iter < max_iter;
    let convergence_flag = if converged { 0 } else { -1 };
    let variance = calculate_variance_matrix(imat, nvar2, tol_chol)?;
    Ok(SurvivalFitComputed {
        coefficients: beta,
        iterations: iter,
        variance_matrix: variance,
        log_likelihood: loglik,
        convergence_flag,
        score_vector: usave.to_vec(),
    })
}
pub(crate) struct SurvivalFitComputed {
    coefficients: Vec<f64>,
    iterations: usize,
    variance_matrix: Array2<f64>,
    log_likelihood: f64,
    convergence_flag: i32,
    score_vector: Vec<f64>,
}

#[pyfunction]
pub fn survreg_with_options(
    time: Vec<f64>,
    status: Vec<f64>,
    covariates: Vec<Vec<f64>>,
    options: Option<&SurvregOptions>,
) -> PyResult<SurvivalFit> {
    let opts = options.cloned().unwrap_or_default();
    survreg(
        time,
        status,
        covariates,
        opts.weights,
        opts.offsets,
        opts.initial_beta,
        opts.strata,
        opts.distribution.as_deref(),
        opts.max_iter,
        opts.eps,
        opts.tol_chol,
    )
}

#[pymodule]
#[pyo3(name = "survreg")]
fn survreg_module(_py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(survreg, &m)?)?;
    m.add_function(wrap_pyfunction!(survreg_with_options, &m)?)?;
    m.add_class::<SurvivalFit>()?;
    m.add_class::<DistributionType>()?;
    m.add_class::<SurvregOptions>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_survreg_config_default() {
        let config = SurvregConfig::default();
        assert_eq!(config.max_iter, 30);
        assert!((config.eps - 1e-6).abs() < 1e-10);
        assert!((config.tol_chol - 1e-10).abs() < 1e-15);
        assert_eq!(config.distribution, DistributionType::ExtremeValue);
    }

    #[test]
    fn test_survreg_config_create() {
        let config = SurvregConfig::create(
            Some(DistributionType::Gaussian),
            Some(50),
            Some(1e-8),
            Some(1e-12),
        );
        assert_eq!(config.max_iter, 50);
        assert!((config.eps - 1e-8).abs() < 1e-15);
        assert_eq!(config.distribution, DistributionType::Gaussian);
    }

    #[test]
    fn test_distribution_type_variants() {
        let _ = DistributionType::ExtremeValue;
        let _ = DistributionType::Weibull;
        let _ = DistributionType::Gaussian;
        let _ = DistributionType::Logistic;
        let _ = DistributionType::LogNormal;
    }
}
