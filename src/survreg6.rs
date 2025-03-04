use ndarray::{arr2, Array1, Array2, Axis};
use std::f64::EPSILON;

#[derive(Debug)]
pub struct SurvivalFit {
    pub coefficients: Vec<f64>,
    pub iterations: usize,
    pub variance_matrix: Array2<f64>,
    pub log_likelihood: f64,
    pub convergence_flag: i32,
    pub score_vector: Vec<f64>,
}

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
) -> Result<SurvivalFit, &'static str> {
    let n = y.nrows();
    let ny = y.ncols();
    let nvar2 = nvar + nstrat;
    let mut loglik = 0.0;
    let mut flag = 0;

    let mut imat = Array2::zeros((nvar2, nvar2));
    let mut jj = Array2::zeros((nvar2, nvar2));
    let mut u = Array1::zeros(nvar2);
    let mut newbeta = beta.clone();
    let mut usave = Array1::zeros(nvar2);

    let (time1, time2, status) = match ny {
        2 => (y.column(0).to_owned(), None, y.column(1).to_owned()),
        3 => (
            y.column(0).to_owned(),
            Some(y.column(1).to_owned()),
            y.column(2).to_owned(),
        ),
        _ => return Err("Invalid y matrix columns"),
    };

    loglik = calculate_likelihood(
        n,
        nvar,
        nstrat,
        &beta,
        &distribution,
        strata,
        offsets,
        &time1,
        time2.as_ref(),
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
            time2.as_ref(),
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

fn calculate_likelihood(
    n: usize,
    nvar: usize,
    nstrat: usize,
    beta: &[f64],
    distribution: &DistributionType,
    strata: &[usize],
    offsets: &Array1<f64>,
    time1: &Array1<f64>,
    time2: Option<&Array1<f64>>,
    status: &Array1<f64>,
    weights: &Array1<f64>,
    covariates: &Array2<f64>,
    imat: &mut Array2<f64>,
    jj: &mut Array2<f64>,
    u: &mut Array1<f64>,
) -> Result<f64, &'static str> {
    Ok(0.0)
}

fn cholesky_solve(
    matrix: &Array2<f64>,
    vector: &Array1<f64>,
    tol: f64,
) -> Result<Array1<f64>, &'static str> {
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
    mut imat: Array2<f64>,
    nvar2: usize,
    tol_chol: f64,
) -> Result<Array2<f64>, &'static str> {
    cholesky_solve(&imat, &Array1::zeros(nvar2), tol_chol)?;
    Ok(imat)
}

#[derive(Debug, Clone)]
pub enum DistributionType {
    ExtremeValue,
    Logistic,
    Gaussian,
    Custom(Box<dyn Fn(f64) -> f64>),
}
