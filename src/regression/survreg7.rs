#![allow(dead_code)]
#![allow(clippy::redundant_closure)]
use crate::core::survpenal::{self, MatrixBuffers, PenaltyParams, PenaltyResult};
use crate::regression::survregc1::SurvivalDist;
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
    fgrp: &[usize],
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
    let mut flag = 0;

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
        nfrail,
        fgrp,
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
            nfrail,
            fgrp,
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
            flag = 0;
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
                nfrail,
                fgrp,
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

    if iter >= max_iter {
        flag = 1;
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
    n: usize,
    nvar: usize,
    nstrat: usize,
    beta: &[f64],
    distribution: &Distribution,
    strata: &[usize],
    offsets: &ArrayView1<f64>,
    time1: &ArrayView1<f64>,
    time2: Option<&ArrayView1<f64>>,
    status: &ArrayView1<f64>,
    weights: &ArrayView1<f64>,
    covariates: &ArrayView2<f64>,
    hmat: &mut Array2<f64>,
    jj: &mut Array2<f64>,
    u: &mut Array1<f64>,
    hdiag: &mut Array1<f64>,
    _jdiag: &mut Array1<f64>,
    nfrail: usize,
    fgrp: &[usize],
) -> Result<f64, Box<dyn std::error::Error>> {
    use crate::regression::survregc1::survregc1;

    let dist = match distribution {
        Distribution::ExtremeValue => SurvivalDist::ExtremeValue,
        Distribution::Logistic => SurvivalDist::Logistic,
        Distribution::Gaussian => SurvivalDist::Gaussian,
        Distribution::Custom(_) => {
            return Err("Custom distributions not yet supported in calculate_likelihood".into());
        }
    };

    let strat_vec: Vec<i32> = strata.iter().map(|&s| (s + 1) as i32).collect();
    let strat_arr = Array1::from_vec(strat_vec);

    let status_vec: Vec<i32> = status.iter().map(|&s| s as i32).collect();
    let status_arr = Array1::from_vec(status_vec);

    let beta_arr = Array1::from_vec(beta.to_vec());
    let beta_view = beta_arr.view();

    let frail_vec: Vec<i32> = if nfrail > 0 && !fgrp.is_empty() {
        fgrp.iter().map(|&g| (g + 1) as i32).collect()
    } else {
        vec![0; n]
    };
    let frail_arr = Array1::from_vec(frail_vec);
    let frail_view = frail_arr.view();

    let result = survregc1(
        n,
        nvar,
        nstrat,
        false,
        &beta_view,
        dist,
        &strat_arr.view(),
        offsets,
        time1,
        time2,
        &status_arr.view(),
        weights,
        covariates,
        nfrail,
        &frail_view,
    )?;

    let nvar2 = nvar + nstrat;
    let nvar3 = nvar2 + nfrail;

    for i in 0..nvar3.min(result.u.len()) {
        u[i] = result.u[i];
    }

    for i in 0..nvar3.min(hmat.nrows()) {
        for j in 0..nvar2.min(hmat.ncols()) {
            if i < result.imat.ncols() && j < result.imat.nrows() {
                hmat[[i, j]] = -result.imat[[j, i]];
            }
        }
    }

    for i in 0..nvar3.min(jj.nrows()) {
        for j in 0..nvar2.min(jj.ncols()) {
            if i < result.jj.ncols() && j < result.jj.nrows() {
                jj[[i, j]] = result.jj[[j, i]];
            }
        }
    }

    for i in 0..nvar3.min(hdiag.len()) {
        if i < result.imat.nrows() && i < result.imat.ncols() {
            hdiag[i] = -result.imat[[i, i]];
        }
    }

    if nfrail > 0 {
        for i in 0..nfrail.min(result.fdiag.len()) {
            hdiag[i] = -result.fdiag[i];
        }
    }

    Ok(result.loglik)
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
    let ptype_int = match ptype {
        PenaltyType::None => 0,
        PenaltyType::Sparse => 1,
        PenaltyType::Dense => 2,
        PenaltyType::Both => 3,
    };

    if ptype_int == 0 {
        return Ok(0.0);
    }

    let pdiag_int = if pdiag { 1 } else { 0 };
    let whichcase = 0;

    let hmat_slice = hmat.as_slice_mut().ok_or("Failed to get hmat slice")?;
    let jj_slice = jj.as_slice_mut().ok_or("Failed to get jj slice")?;
    let hdiag_slice = hdiag.as_slice_mut().ok_or("Failed to get hdiag slice")?;
    let jdiag_slice = jdiag.as_slice_mut().ok_or("Failed to get jdiag slice")?;
    let u_slice = u.as_slice_mut().ok_or("Failed to get u slice")?;

    const LAMBDA: f64 = 0.1;

    let sparse_penalty = |coef: &[f64]| -> PenaltyResult {
        let n = coef.len();
        let mut first_deriv = vec![0.0; n];
        let mut second_deriv = vec![0.0; n];
        let mut loglik_penalty = 0.0;

        for i in 0..n {
            loglik_penalty += LAMBDA * 0.5 * coef[i].powi(2);
            first_deriv[i] = LAMBDA * coef[i];
            second_deriv[i] = LAMBDA;
        }

        PenaltyResult {
            new_coef: coef.to_vec(),
            first_deriv,
            second_deriv,
            loglik_penalty,
            flags: vec![0; n],
        }
    };

    let dense_penalty = |coef: &[f64]| -> PenaltyResult {
        let n = coef.len();
        let mut first_deriv = vec![0.0; n];
        let mut second_deriv = vec![0.0; n];
        let mut loglik_penalty = 0.0;

        for i in 0..n {
            loglik_penalty += LAMBDA * 0.5 * coef[i].powi(2);
            first_deriv[i] = LAMBDA * coef[i];
            second_deriv[i] = LAMBDA;
        }

        PenaltyResult {
            new_coef: coef.to_vec(),
            first_deriv,
            second_deriv,
            loglik_penalty,
            flags: vec![0; n],
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
    beta: &[f64],
    newbeta: &[f64],
    _current_loglik: f64,
    n: usize,
    nvar: usize,
    nstrat: usize,
    strata: &[usize],
    offsets: &ArrayView1<f64>,
    time1: &ArrayView1<f64>,
    time2: Option<&ArrayView1<f64>>,
    status: &ArrayView1<f64>,
    weights: &ArrayView1<f64>,
    covariates: &ArrayView2<f64>,
    distribution: &Distribution,
    nfrail: usize,
    fgrp: &[usize],
) -> Result<f64, Box<dyn std::error::Error>> {
    const GOLDEN_RATIO: f64 = 0.6180339887498949;
    const TOL: f64 = 1e-6;
    const MAX_ITER: usize = 50;

    let mut a = 0.0;
    let mut b = 1.0;
    let mut c = b - GOLDEN_RATIO * (b - a);
    let mut d = a + GOLDEN_RATIO * (b - a);

    let mut fc = evaluate_likelihood_at_alpha(
        beta,
        newbeta,
        c,
        n,
        nvar,
        nstrat,
        strata,
        offsets,
        time1,
        time2,
        status,
        weights,
        covariates,
        distribution,
        nfrail,
        fgrp,
    )?;
    let mut fd = evaluate_likelihood_at_alpha(
        beta,
        newbeta,
        d,
        n,
        nvar,
        nstrat,
        strata,
        offsets,
        time1,
        time2,
        status,
        weights,
        covariates,
        distribution,
        nfrail,
        fgrp,
    )?;

    let mut iter = 0;
    while (b - a).abs() > TOL && iter < MAX_ITER {
        if fc > fd {
            b = d;
            d = c;
            fd = fc;
            c = b - GOLDEN_RATIO * (b - a);
            fc = evaluate_likelihood_at_alpha(
                beta,
                newbeta,
                c,
                n,
                nvar,
                nstrat,
                strata,
                offsets,
                time1,
                time2,
                status,
                weights,
                covariates,
                distribution,
                nfrail,
                fgrp,
            )?;
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + GOLDEN_RATIO * (b - a);
            fd = evaluate_likelihood_at_alpha(
                beta,
                newbeta,
                d,
                n,
                nvar,
                nstrat,
                strata,
                offsets,
                time1,
                time2,
                status,
                weights,
                covariates,
                distribution,
                nfrail,
                fgrp,
            )?;
        }
        iter += 1;
    }

    Ok((a + b) / 2.0)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_likelihood_at_alpha(
    beta: &[f64],
    newbeta: &[f64],
    alpha: f64,
    n: usize,
    nvar: usize,
    nstrat: usize,
    strata: &[usize],
    offsets: &ArrayView1<f64>,
    time1: &ArrayView1<f64>,
    time2: Option<&ArrayView1<f64>>,
    status: &ArrayView1<f64>,
    weights: &ArrayView1<f64>,
    covariates: &ArrayView2<f64>,
    distribution: &Distribution,
    nfrail: usize,
    fgrp: &[usize],
) -> Result<f64, Box<dyn std::error::Error>> {
    let beta_alpha: Vec<f64> = beta
        .iter()
        .zip(newbeta.iter())
        .map(|(b, nb)| b + alpha * (nb - b))
        .collect();

    let nvar2 = nvar + nstrat;
    let nvar3 = nvar2 + nfrail;
    let mut hmat_temp = Array2::zeros((nvar3, nvar2));
    let mut jj_temp = Array2::zeros((nvar3, nvar2));
    let mut u_temp = Array1::zeros(nvar3);
    let mut hdiag_temp = Array1::zeros(nvar3);
    let mut jdiag_temp = Array1::zeros(nfrail);

    calculate_likelihood(
        n,
        nvar,
        nstrat,
        &beta_alpha,
        distribution,
        strata,
        offsets,
        time1,
        time2,
        status,
        weights,
        covariates,
        &mut hmat_temp,
        &mut jj_temp,
        &mut u_temp,
        &mut hdiag_temp,
        &mut jdiag_temp,
        nfrail,
        fgrp,
    )
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
