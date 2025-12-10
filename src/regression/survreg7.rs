#![allow(dead_code)]
#![allow(clippy::redundant_closure)]
use crate::core::survpenal::{self, MatrixBuffers, PenaltyParams, PenaltyResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use ndarray_linalg::{Cholesky, Inverse, Solve, UPLO};

#[derive(Debug)]
pub struct SurvivalResult {
    pub coefficients: Vec<f64>,
    pub iterations: usize,
    pub h_matrix: Array2<f64>,
    pub h_inv: Array2<f64>,
    pub h_diag: Vec<f64>,
    pub log_likelihood: f64,
    pub score: Vec<f64>,
    pub penalty: f64,
    pub convergence_flag: i32,
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
    distribution: Distribution,
    ptype: PenaltyType,
    pdiag: bool,
    nfrail: usize,
    _fgrp: &[usize],
) -> Result<SurvivalResult, Box<dyn std::error::Error>> {
    let n = y.nrows();
    let ny = y.ncols();
    let nvar2 = nvar + nstrat;
    let nvar3 = nvar2 + nfrail;

    let mut hmat = Array2::zeros((nvar3, nvar2));
    let mut jj = Array2::zeros((nvar3, nvar2));
    let mut hdiag = Array1::zeros(nvar3);
    let mut jdiag = Array1::zeros(nfrail);
    let mut u = Array1::zeros(nvar3);
    let mut newbeta = beta.clone();
    let flag = 0;

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

    let time1_view = time1_arr.view();
    let status_view = status_arr.view();
    let time2_view = time2_arr.as_ref().map(|v| v.view());

    let mut loglik = calculate_likelihood(
        n,
        nvar,
        nstrat,
        &beta,
        &distribution,
        strata,
        &offsets.view(),
        &time1_view,
        time2_view.as_ref(),
        &status_view,
        &weights.view(),
        &covariates.view(),
        &mut hmat,
        &mut jj,
        &mut u,
        &mut hdiag,
        &mut jdiag,
    )?;

    let mut penalty_val = apply_penalties(
        &mut hmat, &mut jj, &mut hdiag, &mut jdiag, &mut u, &mut beta, nvar, nfrail, ptype, pdiag,
    )?;
    loglik += penalty_val;

    let mut iter = 0;
    while iter < max_iter {
        let delta = match hmat.cholesky(UPLO::Lower) {
            Ok(chol) => chol
                .solve(&u)
                .map_err(|_| "Cholesky solve failed".to_string())?,
            Err(_) => {
                let jj_chol = jj
                    .cholesky(UPLO::Lower)
                    .map_err(|_| "Cholesky decomposition failed".to_string())?;
                jj_chol
                    .solve(&u)
                    .map_err(|_| "Cholesky solve failed".to_string())?
            }
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
            &offsets.view(),
            &time1_view,
            time2_view.as_ref(),
            &status_view,
            &weights.view(),
            &covariates.view(),
            &mut hmat,
            &mut jj,
            &mut u,
            &mut hdiag,
            &mut jdiag,
        )?;

        let new_penalty = apply_penalties(
            &mut hmat,
            &mut jj,
            &mut hdiag,
            &mut jdiag,
            &mut u,
            &mut newbeta,
            nvar,
            nfrail,
            ptype,
            pdiag,
        )?;
        let newlik = newlik + new_penalty;

        if (1.0 - (loglik / newlik)).abs() <= eps {
            loglik = newlik;
            penalty_val = new_penalty;
            beta.copy_from_slice(&newbeta);
            break;
        }

        if newlik < loglik {
            let alpha = golden_section_search(
                &beta,
                &newbeta,
                loglik,
                n,
                nvar,
                nstrat,
                strata,
                &offsets.view(),
                &time1_view,
                time2_view.as_ref(),
                &status_view,
                &weights.view(),
                &covariates.view(),
                &distribution,
            )?;
            newbeta
                .iter_mut()
                .zip(beta.iter())
                .for_each(|(nb, b)| *nb = b + alpha * (*nb - b));
        }

        beta.copy_from_slice(&newbeta);
        loglik = newlik;
        penalty_val = new_penalty;
        iter += 1;
    }

    let h_inv = calculate_inverse(&hmat, nvar3, nfrail, &hdiag, tol_chol)?;

    Ok(SurvivalResult {
        coefficients: beta,
        iterations: iter,
        h_matrix: hmat,
        h_inv,
        h_diag: hdiag.to_vec(),
        log_likelihood: loglik,
        score: u.to_vec(),
        penalty: penalty_val,
        convergence_flag: flag,
    })
}

#[allow(clippy::too_many_arguments)]
fn calculate_likelihood(
    _n: usize,
    _nvar: usize,
    _nstrat: usize,
    _beta: &[f64],
    _distribution: &Distribution,
    _strata: &[usize],
    _offsets: &ArrayView1<f64>,
    _time1: &ArrayView1<f64>,
    _time2: Option<&ArrayView1<f64>>,
    _status: &ArrayView1<f64>,
    _weights: &ArrayView1<f64>,
    _covariates: &ArrayView2<f64>,
    _hmat: &mut Array2<f64>,
    _jj: &mut Array2<f64>,
    _u: &mut Array1<f64>,
    _hdiag: &mut Array1<f64>,
    _jdiag: &mut Array1<f64>,
) -> Result<f64, Box<dyn std::error::Error>> {
    Ok(0.0)
}

#[allow(clippy::too_many_arguments)]
fn apply_penalties(
    hmat: &mut Array2<f64>,
    jj: &mut Array2<f64>,
    hdiag: &mut Array1<f64>,
    jdiag: &mut Array1<f64>,
    u: &mut Array1<f64>,
    beta: &mut [f64],
    nvar: usize,
    nfrail: usize,
    ptype: PenaltyType,
    pdiag: bool,
) -> Result<f64, Box<dyn std::error::Error>> {
    // Map PenaltyType enum to ptype integer
    // 0 = None, 1 = Sparse only, 2 = Dense only, 3 = Both
    let ptype_int = match ptype {
        PenaltyType::None => 0,
        PenaltyType::Sparse => 1,
        PenaltyType::Dense => 2,
        PenaltyType::Both => 3,
    };

    // If no penalty, return early
    if ptype_int == 0 {
        return Ok(0.0);
    }

    let pdiag_int = if pdiag { 1 } else { 0 };
    let whichcase = 0; // 0 means update matrices, 1 means just compute penalty

    // Get mutable slices from ndarray
    let hmat_slice = hmat.as_slice_mut().ok_or("Failed to get hmat slice")?;
    let jj_slice = jj.as_slice_mut().ok_or("Failed to get jj slice")?;
    let hdiag_slice = hdiag.as_slice_mut().ok_or("Failed to get hdiag slice")?;
    let jdiag_slice = jdiag.as_slice_mut().ok_or("Failed to get jdiag slice")?;
    let u_slice = u.as_slice_mut().ok_or("Failed to get u slice")?;

    // Default penalty functions that return zero penalties
    // These can be replaced with actual penalty implementations later
    let sparse_penalty = |coef: &[f64]| -> PenaltyResult {
        PenaltyResult {
            new_coef: coef.to_vec(), // Preserve coefficients
            first_deriv: vec![0.0; coef.len()],
            second_deriv: vec![0.0; coef.len()],
            loglik_penalty: 0.0,
            flags: vec![0; coef.len()],
        }
    };

    let dense_penalty = |coef: &[f64]| -> PenaltyResult {
        PenaltyResult {
            new_coef: coef.to_vec(), // Preserve coefficients
            first_deriv: vec![0.0; coef.len()],
            second_deriv: vec![0.0; coef.len()],
            loglik_penalty: 0.0,
            flags: vec![0; coef.len()],
        }
    };

    let params = PenaltyParams {
        whichcase,
        nfrail,
        nvar,
        ptype: ptype_int,
        pdiag: pdiag_int,
    };

    let matrices = MatrixBuffers {
        hmat: hmat_slice,
        JJ: jj_slice,
        hdiag: hdiag_slice,
        jdiag: jdiag_slice,
        u: u_slice,
        beta,
    };

    let mut penalty = 0.0;
    survpenal::survpenal(
        params,
        matrices,
        &mut penalty,
        sparse_penalty,
        dense_penalty,
    );

    Ok(penalty)
}

#[allow(clippy::too_many_arguments)]
fn golden_section_search(
    _beta: &[f64],
    _newbeta: &[f64],
    _current_loglik: f64,
    _n: usize,
    _nvar: usize,
    _nstrat: usize,
    _strata: &[usize],
    _offsets: &ArrayView1<f64>,
    _time1: &ArrayView1<f64>,
    _time2: Option<&ArrayView1<f64>>,
    _status: &ArrayView1<f64>,
    _weights: &ArrayView1<f64>,
    _covariates: &ArrayView2<f64>,
    _distribution: &Distribution,
) -> Result<f64, Box<dyn std::error::Error>> {
    Ok(0.5)
}

fn calculate_inverse(
    hmat: &Array2<f64>,
    _nvar3: usize,
    _nfrail: usize,
    _hdiag: &Array1<f64>,
    _tol_chol: f64,
) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    Ok(hmat
        .inv()
        .map_err(|_| "Matrix inversion failed".to_string())?)
}

pub enum Distribution {
    ExtremeValue,
    Logistic,
    Gaussian,
    Custom(Box<dyn Fn(f64) -> f64 + Send + Sync>),
}

#[derive(Debug, Clone, Copy)]
pub enum PenaltyType {
    None,
    Sparse,
    Dense,
    Both,
}
