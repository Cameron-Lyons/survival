use crate::error::SurvivalResult;
use crate::regression::penalized::cholesky3::{chinv3, cholesky3, chsolve3, finish_factors};
use crate::regression::penalized::terms::{PenaltyCallback, PenaltyShape};
use crate::regression::survregc1::{BlockLikelihood, SparseFrailty, SurvregKernel};
use ndarray::Array2;

pub(super) struct Survreg7Fit {
    pub beta: Vec<f64>,
    pub iter: usize,
    pub converged: bool,
    pub loglik: f64,
    pub penalty: f64,
    pub u: Vec<f64>,
    pub hmat: Array2<f64>,
    pub hinv: Array2<f64>,
    pub hdiag: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Case {
    Full,
    LoglikOnly,
}

#[derive(Default)]
pub(super) struct JjPenalty {
    sparse_flag: bool,
    second1: Vec<f64>,
    second2: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn add_penalty(
    case: Case,
    nf: usize,
    nvar: usize,
    lik: &mut BlockLikelihood,
    beta: &mut [f64],
    shape: PenaltyShape,
    callback: &mut impl PenaltyCallback,
    jj_penalty: &mut JjPenalty,
) -> SurvivalResult<f64> {
    let full = case == Case::Full;
    let mut penalty = 0.0;
    if shape.sparse {
        let mut scratch;
        let coef1 = if full {
            &mut beta[..nf]
        } else {
            scratch = beta[..nf].to_vec();
            &mut scratch[..]
        };
        let list = callback.call(1, coef1)?;
        penalty += list.penalty;
        if full {
            jj_penalty.sparse_flag = list.flag[0];
            if list.flag[0] {
                for i in 0..nf {
                    lik.fdiag[i] = 1.0;
                    lik.u[i] = 0.0;
                    for j in 0..nvar {
                        lik.hmat[(j, i)] = 0.0;
                    }
                }
                if lik.has_jj {
                    lik.jdiag.fill(1.0);
                }
            } else {
                for i in 0..nf {
                    lik.u[i] += list.first[i];
                    lik.fdiag[i] += list.second[i];
                }
                if lik.has_jj {
                    for (j, s) in lik.jdiag.iter_mut().zip(&list.second) {
                        *j += s;
                    }
                }
                jj_penalty.second1.clone_from(&list.second);
            }
        }
    }
    if shape.dense {
        let mut scratch;
        let coef2 = if full {
            &mut beta[nf..nf + nvar]
        } else {
            scratch = beta[nf..nf + nvar].to_vec();
            &mut scratch[..]
        };
        let list = callback.call(2, coef2)?;
        penalty += list.penalty;
        if full {
            for i in 0..nvar {
                lik.u[nf + i] += list.first[i];
            }
            add_dense_second(&mut lik.hmat, nf, nvar, &list.second, shape.full_imat);
            if lik.has_jj {
                add_dense_second(&mut lik.jj, nf, nvar, &list.second, shape.full_imat);
            }
            for i in 0..nvar {
                if list.flag[i] {
                    lik.u[nf + i] = 0.0;
                    lik.hmat[(i, nf + i)] = 1.0;
                    for j in 0..i {
                        lik.hmat[(i, nf + j)] = 0.0;
                    }
                }
            }
            jj_penalty.second2.clone_from(&list.second);
        }
    }
    Ok(penalty)
}

fn add_dense_second(
    matrix: &mut Array2<f64>,
    nf: usize,
    nvar: usize,
    second: &[f64],
    full_imat: bool,
) {
    if full_imat {
        let mut k = 0;
        for i in 0..nvar {
            for j in nf..nvar + nf {
                matrix[(i, j)] += second[k];
                k += 1;
            }
        }
    } else {
        for i in 0..nvar {
            matrix[(i, i + nf)] += second[i];
        }
    }
}

pub(super) fn ensure_jj(
    kernel: &SurvregKernel<'_>,
    frailty: Option<&SparseFrailty<'_>>,
    lik: &mut BlockLikelihood,
    scratch: &mut Option<BlockLikelihood>,
    jj_penalty: &JjPenalty,
    shape: PenaltyShape,
) -> SurvivalResult<()> {
    if lik.has_jj {
        return Ok(());
    }
    let nf = lik.fdiag.len();
    let nvar = kernel.nvar();
    let scratch = scratch.get_or_insert_with(|| BlockLikelihood::new(nf, lik.hmat.nrows()));
    kernel.evaluate_blocks(&lik.at, frailty, true, scratch)?;
    lik.jj.assign(&scratch.jj);
    lik.jdiag.copy_from_slice(&scratch.jdiag);
    if shape.sparse {
        if jj_penalty.sparse_flag {
            lik.jdiag.fill(1.0);
        } else {
            for (j, s) in lik.jdiag.iter_mut().zip(&jj_penalty.second1) {
                *j += s;
            }
        }
    }
    if shape.dense {
        add_dense_second(&mut lik.jj, nf, nvar, &jj_penalty.second2, shape.full_imat);
    }
    lik.has_jj = true;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn loglik_along(
    kernel: &SurvregKernel<'_>,
    frailty: Option<&SparseFrailty<'_>>,
    beta: &[f64],
    u: &[f64],
    x: f64,
    trial: &mut [f64],
    lik: &mut BlockLikelihood,
    shape: PenaltyShape,
    callback: &mut impl PenaltyCallback,
    jj_penalty: &mut JjPenalty,
) -> SurvivalResult<f64> {
    for (i, (b, step)) in beta.iter().zip(u).enumerate() {
        trial[i] = b + step * x;
    }
    let nf = frailty.map_or(0, |f| f.nf);
    let loglik = kernel.loglik_at(trial, frailty)?;
    let penalty = add_penalty(
        Case::LoglikOnly,
        nf,
        kernel.nvar(),
        lik,
        trial,
        shape,
        callback,
        jj_penalty,
    )?;
    Ok(loglik + penalty)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn survreg7(
    kernel: &SurvregKernel<'_>,
    frailty: Option<&SparseFrailty<'_>>,
    maxiter: usize,
    mut beta: Vec<f64>,
    eps: f64,
    tol_chol: f64,
    shape: PenaltyShape,
    callback: &mut impl PenaltyCallback,
) -> SurvivalResult<Survreg7Fit> {
    let nf = frailty.map_or(0, |f| f.nf);
    let nvar = kernel.nvar();
    let nstrat = kernel.nstrat;
    let nvar3 = nf + nvar + nstrat;
    let mut lik = BlockLikelihood::new(nf, nvar + nstrat);
    let mut scratch = None;
    let mut jj_penalty = JjPenalty::default();
    let mut newbeta = beta.clone();
    let mut step = vec![0.0; nvar3];

    kernel.evaluate_blocks(&beta, frailty, false, &mut lik)?;
    let mut penalty = add_penalty(
        Case::Full,
        nf,
        nvar,
        &mut lik,
        &mut beta,
        shape,
        callback,
        &mut jj_penalty,
    )?;
    let mut loglik = lik.loglik + penalty;
    let mut usave = lik.u.clone();

    let mut iter = 1;
    let mut converged = false;
    while iter <= maxiter {
        let flag = cholesky3(&mut lik.hmat, nf, &mut lik.fdiag, tol_chol);
        step.copy_from_slice(&lik.u);
        if flag < 0 {
            ensure_jj(kernel, frailty, &mut lik, &mut scratch, &jj_penalty, shape)?;
            cholesky3(&mut lik.jj, nf, &mut lik.jdiag, tol_chol);
            chsolve3(&lik.jj, nf, &lik.jdiag, &mut step);
        } else {
            chsolve3(&lik.hmat, nf, &lik.fdiag, &mut step);
        }
        for i in 0..nvar3 {
            newbeta[i] = beta[i] + step[i];
        }
        kernel.evaluate_blocks(&newbeta, frailty, false, &mut lik)?;
        let mut newpen = add_penalty(
            Case::Full,
            nf,
            nvar,
            &mut lik,
            &mut newbeta,
            shape,
            callback,
            &mut jj_penalty,
        )?;
        let mut newlk = lik.loglik + newpen;

        if (1.0 - loglik / newlk).abs() <= eps {
            loglik = newlk;
            penalty = newpen;
            beta[..nvar3].copy_from_slice(&newbeta[..nvar3]);
            usave.copy_from_slice(&lik.u);
            converged = true;
            break;
        }

        if newlk < loglik {
            for i in 0..nvar3 {
                step[i] = newbeta[i] - beta[i];
            }
            for i in 0..nstrat {
                if step[i + nvar + nf] < -0.7 {
                    step[i + nvar + nf] = -1.1;
                }
            }
            let along = |x: f64,
                         newbeta: &mut [f64],
                         lik: &mut BlockLikelihood,
                         callback: &mut _,
                         jj_penalty: &mut JjPenalty| {
                loglik_along(
                    kernel,
                    frailty,
                    &beta[..nvar3],
                    &step,
                    x,
                    newbeta,
                    lik,
                    shape,
                    callback,
                    jj_penalty,
                )
            };
            let mut x4 = 1.0;
            let mut x1 = 0.0;
            let mut x3 = 1.0;
            let mut y1 = loglik;
            let mut y3 = newlk;
            while y1 > y3 {
                x4 = x3;
                x3 = x1;
                x1 = x3 - (x4 - x3) / 0.618;
                y3 = y1;
                y1 = along(x1, &mut newbeta, &mut lik, callback, &mut jj_penalty)?;
            }
            let mut x2 = 0.618 * x3 + 0.382 * x1;
            let mut y2 = along(x2, &mut newbeta, &mut lik, callback, &mut jj_penalty)?;
            for _ in 0..8 {
                if y3 > y2 {
                    x1 = x2;
                    x2 = x3;
                    x3 = 0.618 * x4 + 0.382 * x1;
                    y2 = y3;
                    y3 = along(x3, &mut newbeta, &mut lik, callback, &mut jj_penalty)?;
                } else {
                    x4 = x3;
                    x3 = x2;
                    x2 = 0.618 * x1 + 0.382 * x4;
                    y3 = y2;
                    y2 = along(x2, &mut newbeta, &mut lik, callback, &mut jj_penalty)?;
                }
            }
            if !(y2 > loglik || y3 > loglik) {
                break;
            }
            let x = if y2 > y3 { x2 } else { x3 };
            for i in 0..nvar3 {
                newbeta[i] = beta[i] + step[i] * x;
            }
            kernel.evaluate_blocks(&newbeta, frailty, false, &mut lik)?;
            newpen = add_penalty(
                Case::Full,
                nf,
                nvar,
                &mut lik,
                &mut newbeta,
                shape,
                callback,
                &mut jj_penalty,
            )?;
            newlk = lik.loglik + newpen;
        }

        beta[..nvar3].copy_from_slice(&newbeta[..nvar3]);
        usave.copy_from_slice(&lik.u);
        loglik = newlk;
        penalty = newpen;
        iter += 1;
    }

    cholesky3(&mut lik.hmat, nf, &mut lik.fdiag, tol_chol);
    let mut hmat = lik.hmat;
    let mut hinv = hmat.clone();
    let mut hdiag = lik.fdiag;
    hdiag.resize(nvar3, 0.0);
    chinv3(&mut hinv, nf, &mut hdiag);
    finish_factors(&mut hmat, &mut hinv, nf, &mut hdiag);
    Ok(Survreg7Fit {
        beta,
        iter,
        converged,
        loglik,
        penalty,
        u: usave,
        hmat,
        hinv,
        hdiag,
    })
}
