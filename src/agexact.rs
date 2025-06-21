use itertools::Itertools;
use std::f64::EPSILON;

/// Helper function to compute combinations and handle risk sets
fn init_doloop(start: usize, end: usize, k: usize) -> Vec<Vec<usize>> {
    (start..end).combinations(k).collect()
}

/// Anderson-Gill exact Cox model implementation
pub fn agexact(
    maxiter: &mut i32,
    nused: &i32,
    nvar: &i32,
    start: &[f64],
    stop: &[f64],
    event: &[i32],
    covar: &mut [f64],
    offset: &[f64],
    strata: &[i32],
    means: &mut [f64],
    beta: &mut [f64],
    u: &mut [f64],
    imat: &mut [f64],
    loglik: &mut [f64; 2],
    flag: &mut i32,
    work: &mut [f64],
    work2: &mut [i32],
    eps: &f64,
    tol_chol: &f64,
    sctest: &mut f64,
    nocenter: &[i32],
) {
    let n = *nused as usize;
    let nvar = *nvar as usize;
    let p = nvar;

    // Split work array into components
    let (cmat, rest) = work.split_at_mut(p * p);
    let (a, rest) = rest.split_at_mut(p);
    let (newbeta, rest) = rest.split_at_mut(p);
    let (score, newvar) = rest.split_at_mut(n);

    let index = &mut work2[0..n];
    let atrisk = &mut work2[n..2 * n];

    // Initialize covariate means
    for i in 0..nvar {
        if nocenter[i] == 0 {
            means[i] = 0.0;
        } else {
            let mut sum = 0.0;
            for j in 0..n {
                sum += covar[i * n + j];
            }
            means[i] = sum / n as f64;
            for j in 0..n {
                covar[i * n + j] -= means[i];
            }
        }
    }

    // Initial score calculation
    for person in 0..n {
        let mut zbeta = 0.0;
        for i in 0..nvar {
            zbeta += beta[i] * covar[i * n + person];
        }
        score[person] = (zbeta + offset[person]).exp();
    }

    // Initial accumulation
    loglik[1] = 0.0;
    u.fill(0.0);
    imat.fill(0.0);

    let mut person = 0;
    while person < n {
        if event[person] == 0 {
            person += 1;
        } else {
            let time = stop[person];
            let mut deaths = 0;
            let mut nrisk = 0;
            let mut k = person;

            // Determine risk set and deaths
            while k < n {
                if stop[k] == time {
                    deaths += event[k];
                }
                if start[k] < time {
                    atrisk[nrisk] = k as i32;
                    nrisk += 1;
                }
                if strata[k] == 1 {
                    k += 1;
                    break;
                }
                k += 1;
            }

            let mut denom = 0.0;
            a.fill(0.0);
            cmat.fill(0.0);

            if deaths == 1 {
                for l in 0..nrisk {
                    let k = atrisk[l] as usize;
                    let weight = score[k];
                    denom += weight;
                    for i in 0..nvar {
                        let covar_ik = covar[i * n + k];
                        a[i] += weight * covar_ik;
                        for j in 0..=i {
                            let covar_jk = covar[j * n + k];
                            cmat[i * p + j] += weight * covar_ik * covar_jk;
                        }
                    }
                }
            } else {
                let combinations = init_doloop(0, nrisk, deaths.try_into().unwrap());
                for indices in combinations {
                    newvar.fill(0.0);
                    let mut weight = 1.0;
                    for &idx in &indices {
                        let k = atrisk[idx] as usize;
                        weight *= score[k];
                        for i in 0..nvar {
                            newvar[i] += covar[i * n + k];
                        }
                    }
                    denom += weight;
                    for i in 0..nvar {
                        a[i] += weight * newvar[i];
                        for j in 0..=i {
                            cmat[i * p + j] += weight * newvar[i] * newvar[j];
                        }
                    }
                }
            }

            // Update log-likelihood and information matrix
            loglik[1] -= denom.ln();
            for i in 0..nvar {
                u[i] -= a[i] / denom;
                for j in 0..=i {
                    let cmat_ij = cmat[i * p + j];
                    let term = (cmat_ij - a[i] * a[j] / denom) / denom;
                    imat[j * p + i] += term;
                }
            }

            // Process events at this time point
            let mut k = person;
            while k < n && stop[k] == time {
                if event[k] == 1 {
                    loglik[1] += score[k].ln();
                    for i in 0..nvar {
                        u[i] += covar[i * n + k];
                    }
                }
                person += 1;
                if strata[k] == 1 {
                    break;
                }
                k += 1;
            }
        }
    }

    loglik[0] = loglik[1];
    let mut a_copy = a.to_vec();
    *flag = cholesky2(imat, p, *tol_chol);
    chsolve2(imat, p, &mut a_copy);
    *sctest = a_copy.iter().zip(u.iter()).map(|(a, u)| a * u).sum();

    if *maxiter == 0 {
        chinv2(imat, p);
        for i in 0..p {
            for j in 0..i {
                imat[i * p + j] = imat[j * p + i];
            }
        }
        *flag = 0;
        return;
    }

    // Main iteration loop
    let mut iter = 0;
    let mut halving = false;
    let mut newbeta_vec = newbeta.to_vec();
    let mut newlk = 0.0;

    while iter < *maxiter {
        iter += 1;
        newlk = 0.0;
        u.fill(0.0);
        imat.fill(0.0);

        // Calculate new scores
        for person in 0..n {
            let mut zbeta = 0.0;
            for i in 0..nvar {
                zbeta += newbeta_vec[i] * covar[i * n + person];
            }
            score[person] = (zbeta + offset[person]).exp();
        }

        let mut person = 0;
        while person < n {
            if event[person] == 0 {
                person += 1;
            } else {
                let time = stop[person];
                let mut deaths = 0;
                let mut nrisk = 0;
                let mut k = person;

                while k < n {
                    if stop[k] == time {
                        deaths += event[k];
                    }
                    if start[k] < time {
                        atrisk[nrisk] = k as i32;
                        nrisk += 1;
                    }
                    if strata[k] == 1 {
                        k += 1;
                        break;
                    }
                    k += 1;
                }

                let mut denom = 0.0;
                a.fill(0.0);
                cmat.fill(0.0);

                if deaths == 1 {
                    for l in 0..nrisk {
                        let k = atrisk[l] as usize;
                        let weight = score[k];
                        denom += weight;
                        for i in 0..nvar {
                            let covar_ik = covar[i * n + k];
                            a[i] += weight * covar_ik;
                            for j in 0..=i {
                                cmat[i * p + j] += weight * covar_ik * covar[j * n + k];
                            }
                        }
                    }
                } else {
                    let combinations = init_doloop(0, nrisk, deaths.try_into().unwrap());
                    for indices in combinations {
                        newvar.fill(0.0);
                        let mut weight = 1.0;
                        for &idx in &indices {
                            let k = atrisk[idx] as usize;
                            weight *= score[k];
                            for i in 0..nvar {
                                newvar[i] += covar[i * n + k];
                            }
                        }
                        denom += weight;
                        for i in 0..nvar {
                            a[i] += weight * newvar[i];
                            for j in 0..=i {
                                cmat[i * p + j] += weight * newvar[i] * newvar[j];
                            }
                        }
                    }
                }

                newlk -= denom.ln();
                for i in 0..nvar {
                    u[i] -= a[i] / denom;
                    for j in 0..=i {
                        let cmat_ij = cmat[i * p + j];
                        let term = (cmat_ij - a[i] * a[j] / denom) / denom;
                        imat[j * p + i] += term;
                    }
                }

                let mut k = person;
                while k < n && stop[k] == time {
                    if event[k] == 1 {
                        newlk += score[k].ln();
                        for i in 0..nvar {
                            u[i] += covar[i * n + k];
                        }
                    }
                    person += 1;
                    if strata[k] == 1 {
                        break;
                    }
                    k += 1;
                }
            }
        }

        // Check convergence
        if (1.0 - (loglik[1] / newlk)).abs() <= *eps && !halving {
            loglik[1] = newlk;
            chinv2(imat, p);
            for i in 0..p {
                for j in 0..i {
                    imat[i * p + j] = imat[j * p + i];
                }
            }
            beta.copy_from_slice(&newbeta_vec);
            *maxiter = iter;
            return;
        }

        if iter == *maxiter {
            break;
        }

        if newlk < loglik[1] {
            // Step halving
            halving = true;
            for i in 0..nvar {
                newbeta_vec[i] = (newbeta_vec[i] + beta[i]) / 2.0;
            }
        } else {
            halving = false;
            loglik[1] = newlk;
            *flag = cholesky2(imat, p, *tol_chol);
            let mut u_copy = u.to_vec();
            chsolve2(imat, p, &mut u_copy);

            for i in 0..nvar {
                beta[i] = newbeta_vec[i];
                newbeta_vec[i] += u_copy[i];
            }
        }
    }

    // Final updates if max iterations reached
    loglik[1] = newlk;
    chinv2(imat, p);
    for i in 0..p {
        for j in 0..i {
            imat[i * p + j] = imat[j * p + i];
        }
    }
    beta.copy_from_slice(&newbeta_vec);
    *flag = 1000;
}

/// Cholesky decomposition (lower triangular)
fn cholesky2(matrix: &mut [f64], n: usize, tol: f64) -> i32 {
    for i in 0..n {
        for j in i..n {
            let mut temp = matrix[i * n + j];
            for k in 0..i {
                temp -= matrix[i * n + k] * matrix[j * n + k];
            }
            if j == i {
                if temp <= 0.0 {
                    matrix[i * n + i] = 0.0;
                    return (i + 1) as i32;
                }
                if temp < tol * matrix[i * n + i].abs() {
                    temp = 0.0;
                }
                matrix[i * n + i] = temp.sqrt();
            } else {
                matrix[j * n + i] = temp / matrix[i * n + i];
            }
        }
    }
    0
}

/// Solve Ax = b using Cholesky decomposition
fn chsolve2(chol: &mut [f64], n: usize, b: &mut [f64]) {
    // Forward substitution (L y = b)
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= chol[i * n + j] * b[j];
        }
        b[i] = sum / chol[i * n + i];
    }

    // Backward substitution (L^T x = y)
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= chol[j * n + i] * b[j];
        }
        b[i] = sum / chol[i * n + i];
    }
}

/// Invert Cholesky decomposed matrix
fn chinv2(chol: &mut [f64], n: usize) {
    // Invert the Cholesky factor
    for i in 0..n {
        chol[i * n + i] = 1.0 / chol[i * n + i];
        for j in (i + 1)..n {
            let mut sum = 0.0;
            for k in i..j {
                sum += chol[j * n + k] * chol[k * n + i];
            }
            chol[j * n + i] = -sum * chol[j * n + j];
        }
    }

    // Compute the full inverse
    for i in 0..n {
        for j in i..n {
            let mut sum = 0.0;
            for k in j..n {
                sum += chol[k * n + i] * chol[k * n + j];
            }
            chol[i * n + j] = sum;
        }
    }
}
