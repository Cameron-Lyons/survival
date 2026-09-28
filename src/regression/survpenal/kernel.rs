//! `src/survreg7.c`: the penalised Newton-Raphson iteration of a
//! parametric model, with `src/survpenal.c`, which adds the penalty to the
//! log likelihood, the score and the information.
//!
//! Each step is a Newton step on the penalised information, or a Fisher
//! step on `JJ` when that is not positive definite; a step that lowers the
//! penalised log likelihood is replaced by eight rounds of a golden-section
//! search along it (after capping any drop in a `log(scale)` at -1.1), and
//! the iteration stops when the search finds no improvement.  The
//! information keeps the sparse-plus-dense layout of
//! [`crate::regression::penalized::cholesky3`], so a frailty with `nf`
//! groups costs `O(nvar2 (nf + nvar2))` memory.  `JJ` is only needed by a
//! Fisher step, so it is computed only then, at the point the failed
//! information was evaluated.

use crate::error::SurvivalResult;
use crate::regression::penalized::cholesky3::{chinv3, cholesky3, chsolve3, finish_factors};
use crate::regression::penalized::terms::{PenaltyCallback, PenaltyShape};
use crate::regression::survregc1::{BlockLikelihood, SparseFrailty, SurvregKernel};
use ndarray::Array2;

/// What `survreg7` returns (R's `fit`).
pub(super) struct Survreg7Fit {
    /// `fit$coef`: frailties, `x` coefficients, `log(scale)`s (and the
    /// trailing fixed `log(scale)` when the scale is fixed).
    pub beta: Vec<f64>,
    /// `fit$iter`: the converging iteration, `maxiter + 1` when the budget
    /// ran out, or the iteration of an abject failure of the search.
    pub iter: usize,
    /// The relative-change test passed (R's intended `flag != 1000`, which
    /// survreg7.c:460 overwrites with the rank).
    pub converged: bool,
    /// `fit$loglik`: the penalised log likelihood on the fitting scale.
    pub loglik: f64,
    /// `fit$penalty`: the sum of the `coxlist` penalties (C's sign, `-P`).
    pub penalty: f64,
    /// `fit$u`: the penalised score.
    pub u: Vec<f64>,
    /// `fit$hmat`, `fit$hinv` and `fit$hdiag` after [`finish_factors`].
    pub hmat: Array2<f64>,
    pub hinv: Array2<f64>,
    pub hdiag: Vec<f64>,
}

/// survpenal.c's `whichcase`: a full evaluation, or the log likelihood
/// alone during the golden-section search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Case {
    Full,
    LoglikOnly,
}

/// What `add_penalty(Case::Full)` added to `JJ` and its sparse diagonal,
/// kept so that a Fisher step can build `JJ` after the fact.
#[derive(Default)]
pub(super) struct JjPenalty {
    sparse_flag: bool,
    second1: Vec<f64>,
    second2: Vec<f64>,
}

/// `survpenal.c`: evaluates the sparse (`f.expr1`) and dense (`f.expr2`)
/// penalties at `beta` and returns their sum.  A full evaluation writes the
/// recentred coefficients back into `beta` and adds the derivatives to
/// `lik`; a flagged (infinite) penalty sets its score to zero and its
/// information to the identity instead.  The composer's `coxlist`s are
/// updated either way, as R's `f.expr`s reassign them.
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

/// The dense penalty's second derivative added to `matrix` as survpenal.c
/// does: the diagonal, or the full `nvar x nvar` block read row by row
/// (survpenal.c:109-116; the matrix is symmetric).
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

/// `JJ` and its sparse diagonal at the point `lik` was evaluated (`lik.at`,
/// before the penalty recentred the coefficients), with the penalty terms
/// of that evaluation replayed: bit for bit what an evaluation with `JJ`
/// would have given.
pub(super) fn ensure_jj(
    kernel: &SurvregKernel<'_>,
    frailty: Option<&SparseFrailty<'_>>,
    lik: &mut BlockLikelihood,
    scratch: &mut Option<BlockLikelihood>,
    jj_penalty: &JjPenalty,
    shape: PenaltyShape,
) {
    if lik.has_jj {
        return;
    }
    let nf = lik.fdiag.len();
    let nvar = kernel.nvar();
    let scratch = scratch.get_or_insert_with(|| BlockLikelihood::new(nf, lik.hmat.nrows()));
    kernel.evaluate_blocks(&lik.at, frailty, true, scratch);
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
}

/// The penalised log likelihood at `beta + u * x`, written into `trial`
/// (the whichcase = 1 calls of survreg7.c's golden-section search).
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
    let loglik = kernel.loglik_at(trial, frailty);
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

/// `survreg7`: iterates from `beta` (see [`Survreg7Fit::beta`] for the
/// layout) for at most `maxiter` steps.  R's quirks are kept: the relative
/// change of the penalised log likelihood is the only convergence test,
/// after an abject failure of the search the returned information is that
/// of the failed step, and `maxiter = 0` returns the start with `iter = 1`.
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
    // A fixed log(scale) tacked onto the end of beta is copied to newbeta
    // as well (survreg7.c:196).
    let mut newbeta = beta.clone();
    let mut step = vec![0.0; nvar3];

    kernel.evaluate_blocks(&beta, frailty, false, &mut lik);
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
        // A Newton-Raphson step, or a Fisher step when the information is
        // not positive definite.
        let flag = cholesky3(&mut lik.hmat, nf, &mut lik.fdiag, tol_chol);
        step.copy_from_slice(&lik.u);
        if flag < 0 {
            ensure_jj(kernel, frailty, &mut lik, &mut scratch, &jj_penalty, shape);
            cholesky3(&mut lik.jj, nf, &mut lik.jdiag, tol_chol);
            chsolve3(&lik.jj, nf, &lik.jdiag, &mut step);
        } else {
            chsolve3(&lik.hmat, nf, &lik.fdiag, &mut step);
        }
        for i in 0..nvar3 {
            newbeta[i] = beta[i] + step[i];
        }
        kernel.evaluate_blocks(&newbeta, frailty, false, &mut lik);
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
            // Eight steps of a golden-section search over beta + x u, after
            // limiting any shrinkage of a sigma to a -1.1 change in
            // log(sigma).  x1 is the left end of the search, x2 and x3 the
            // middle, x4 the right end; first bracket the maximum.
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
                    // Toss away the interval from x1 to x2.
                    x1 = x2;
                    x2 = x3;
                    x3 = 0.618 * x4 + 0.382 * x1;
                    y2 = y3;
                    y3 = along(x3, &mut newbeta, &mut lik, callback, &mut jj_penalty)?;
                } else {
                    // Toss away the interval from x3 to x4.
                    x4 = x3;
                    x3 = x2;
                    x2 = 0.618 * x1 + 0.382 * x4;
                    y3 = y2;
                    y2 = along(x2, &mut newbeta, &mut lik, callback, &mut jj_penalty)?;
                }
            }
            if !(y2 > loglik || y3 > loglik) {
                // Abject failure.
                break;
            }
            // Success: keep the better point and its derivatives.
            let x = if y2 > y3 { x2 } else { x3 };
            for i in 0..nvar3 {
                newbeta[i] = beta[i] + step[i] * x;
            }
            kernel.evaluate_blocks(&newbeta, frailty, false, &mut lik);
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

        // newbeta is an improvement: keep it.
        beta[..nvar3].copy_from_slice(&newbeta[..nvar3]);
        usave.copy_from_slice(&lik.u);
        loglik = newlk;
        penalty = newpen;
        iter += 1;
    }

    // The rank this returns is R's fit$flag, which survpenal.fit never reads.
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
