#![allow(dead_code)]
#![allow(clippy::redundant_closure)]
use ndarray::{Array1, Array2, ArrayView1};

#[derive(Debug)]
pub struct SurvivalFit {
    pub coefficients: Vec<f64>,
    pub iterations: usize,
    pub variance_matrix: Array2<f64>,
    pub log_likelihood: f64,
    pub convergence_flag: i32,
    pub score_vector: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
pub fn survreg(
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
) -> Result<SurvivalFit, Box<dyn std::error::Error>> {
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

    Ok(SurvivalFit {
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

pub enum DistributionType {
    ExtremeValue,
    Logistic,
    Gaussian,
    Custom(Box<dyn Fn(f64) -> f64 + Send + Sync>),
}
