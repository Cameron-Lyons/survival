#![allow(clippy::redundant_closure)]
use ndarray::{Array1, Array2, ArrayView1};
use pyo3::prelude::*;

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

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn survreg_original(
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
) -> Result<SurvivalFitInternal, Box<dyn std::error::Error>> {
    let n = y.nrows();
    let ny = y.ncols();
    let nvar2 = nvar + nstrat;
    let flag = 0;

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

    let mut loglik = calculate_likelihood(
        n,
        nvar,
        nstrat,
        &beta,
        &distribution,
        strata,
        offsets,
        &time1,
        time2_view.as_ref(),
        &status,
        weights,
        covariates,
        &mut imat,
        &mut jj,
        &mut u,
    )?;
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

        let newlik = calculate_likelihood(
            n,
            nvar,
            nstrat,
            &newbeta,
            &distribution,
            strata,
            offsets,
            &time1,
            time2_view.as_ref(),
            &status,
            weights,
            covariates,
            &mut imat,
            &mut jj,
            &mut u,
        )?;

        if check_convergence(loglik, newlik, eps) && halving == 0 {
            loglik = newlik;
            beta = newbeta.clone();
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
            beta = newbeta.clone();
        }

        iter += 1;
    }

    let variance = calculate_variance_matrix(imat, nvar2, tol_chol)?;

    Ok(SurvivalFitInternal {
        coefficients: beta,
        iterations: iter,
        variance_matrix: variance,
        log_likelihood: loglik,
        convergence_flag: flag,
        score_vector: usave.to_vec(),
    })
}

#[allow(clippy::too_many_arguments)]
fn calculate_likelihood(
    _n: usize,
    _nvar: usize,
    _nstrat: usize,
    _beta: &[f64],
    _distribution: &DistributionType,
    _strata: &[usize],
    _offsets: &Array1<f64>,
    _time1: &ArrayView1<f64>,
    _time2: Option<&ArrayView1<f64>>,
    _status: &ArrayView1<f64>,
    _weights: &Array1<f64>,
    _covariates: &Array2<f64>,
    _imat: &mut Array2<f64>,
    _jj: &mut Array2<f64>,
    _u: &mut Array1<f64>,
) -> Result<f64, Box<dyn std::error::Error>> {
    Ok(0.0)
}

fn cholesky_solve(
    _matrix: &Array2<f64>,
    vector: &Array1<f64>,
    _tol: f64,
) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
    Ok(Array1::zeros(vector.len()))
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
    nvar2: usize,
    tol_chol: f64,
) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    cholesky_solve(&imat, &Array1::zeros(nvar2), tol_chol)?;
    Ok(imat)
}

#[derive(Debug, Clone, Copy)]
#[pyclass]
pub enum DistributionType {
    #[pyo3(name = "extreme_value")]
    ExtremeValue,
    #[pyo3(name = "logistic")]
    Logistic,
    #[pyo3(name = "gaussian")]
    Gaussian,
}

#[pyfunction]
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
    let n = time.len();
    if status.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and status must have the same length",
        ));
    }

    let nvar = if !covariates.is_empty() {
        covariates[0].len()
    } else {
        0
    };
    if !covariates.is_empty() && covariates.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "covariates must have the same number of rows as time",
        ));
    }

    let weights = weights.unwrap_or_else(|| vec![1.0; n]);
    let offsets = offsets.unwrap_or_else(|| vec![0.0; n]);
    let initial_beta = initial_beta.unwrap_or_else(|| vec![0.0; nvar]);
    let strata = strata.unwrap_or_else(|| vec![0; n]);
    let max_iter = max_iter.unwrap_or(20);
    let eps = eps.unwrap_or(1e-5);
    let tol_chol = tol_chol.unwrap_or(1e-9);

    let dist_type = match distribution {
        Some("logistic") | Some("Logistic") => DistributionType::Logistic,
        Some("gaussian") | Some("Gaussian") | Some("normal") | Some("Normal") => {
            DistributionType::Gaussian
        }
        _ => DistributionType::ExtremeValue,
    };

    let nstrat = if strata.is_empty() {
        1
    } else {
        strata.iter().max().copied().unwrap_or(0) + 1
    };

    let y = {
        let mut y_data = Vec::new();
        for i in 0..n {
            y_data.push(vec![time[i], status[i]]);
        }
        Array2::from_shape_vec((n, 2), y_data.into_iter().flatten().collect())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?
    };

    let cov_array = if nvar > 0 {
        Array2::from_shape_vec((n, nvar), covariates.into_iter().flatten().collect())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?
    } else {
        Array2::zeros((n, 0))
    };

    let weights_arr = Array1::from_vec(weights);
    let offsets_arr = Array1::from_vec(offsets);

    let result = survreg_internal(
        max_iter,
        nvar,
        &y,
        &cov_array,
        &weights_arr,
        &offsets_arr,
        initial_beta,
        nstrat,
        &strata,
        eps,
        tol_chol,
        dist_type,
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
fn survreg_internal(
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
) -> Result<SurvivalFitInternal, Box<dyn std::error::Error>> {
    let n = y.nrows();
    let ny = y.ncols();
    let nvar2 = nvar + nstrat;
    let flag = 0;

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

    let mut loglik = calculate_likelihood(
        n,
        nvar,
        nstrat,
        &beta,
        &distribution,
        strata,
        offsets,
        &time1,
        time2_view.as_ref(),
        &status,
        weights,
        covariates,
        &mut imat,
        &mut jj,
        &mut u,
    )?;
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

        let newlik = calculate_likelihood(
            n,
            nvar,
            nstrat,
            &newbeta,
            &distribution,
            strata,
            offsets,
            &time1,
            time2_view.as_ref(),
            &status,
            weights,
            covariates,
            &mut imat,
            &mut jj,
            &mut u,
        )?;

        if check_convergence(loglik, newlik, eps) && halving == 0 {
            loglik = newlik;
            beta = newbeta.clone();
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
            beta = newbeta.clone();
        }

        iter += 1;
    }

    let variance = calculate_variance_matrix(imat, nvar2, tol_chol)?;

    Ok(SurvivalFitInternal {
        coefficients: beta,
        iterations: iter,
        variance_matrix: variance,
        log_likelihood: loglik,
        convergence_flag: flag,
        score_vector: usave.to_vec(),
    })
}

pub(crate) struct SurvivalFitInternal {
    coefficients: Vec<f64>,
    iterations: usize,
    variance_matrix: Array2<f64>,
    log_likelihood: f64,
    convergence_flag: i32,
    score_vector: Vec<f64>,
}

#[pymodule]
#[pyo3(name = "survreg")]
fn survreg_module(_py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(survreg, &m)?)?;
    m.add_class::<SurvivalFit>()?;
    m.add_class::<DistributionType>()?;
    Ok(())
}
