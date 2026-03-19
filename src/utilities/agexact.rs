use crate::constants::{CONVERGENCE_FLAG, PARALLEL_THRESHOLD_LARGE};
use itertools::Itertools;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

pub struct AgexactData {
    pub start: Vec<f64>,
    pub stop: Vec<f64>,
    pub event: Vec<i32>,
    pub covar: Vec<f64>,
    pub offset: Vec<f64>,
    pub strata: Vec<i32>,
    pub nocenter: Vec<i32>,
}

pub struct AgexactState {
    pub means: Vec<f64>,
    pub beta: Vec<f64>,
    pub u: Vec<f64>,
    pub imat: Vec<f64>,
    pub loglik: Vec<f64>,
    pub work: Vec<f64>,
    pub work2: Vec<i32>,
}

pub struct AgexactParams {
    pub maxiter: i32,
    pub nused: i32,
    pub nvar: i32,
    pub eps: f64,
    pub tol_chol: f64,
}

fn validate_agexact_inputs(
    maxiter: i32,
    nused: i32,
    nvar: i32,
    start: &[f64],
    stop: &[f64],
    event: &[i32],
    covar: &[f64],
    offset: &[f64],
    strata: &[i32],
    means: &[f64],
    beta: &[f64],
    u: &[f64],
    imat: &[f64],
    work: &[f64],
    work2: &[i32],
    nocenter: &[i32],
) -> PyResult<(usize, usize)> {
    if maxiter < 0 || nused < 0 || nvar < 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "maxiter, nused, and nvar must be non-negative",
        ));
    }

    let n = nused as usize;
    let p = nvar as usize;

    if start.len() != n
        || stop.len() != n
        || event.len() != n
        || offset.len() != n
        || strata.len() != n
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "start, stop, event, offset, and strata must all have length nused",
        ));
    }

    let covar_len = p.checked_mul(n).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("nused * nvar is too large")
    })?;
    if covar.len() != covar_len {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "covar must have length nused * nvar",
        ));
    }

    if means.len() != p || beta.len() != p || u.len() != p || nocenter.len() != p {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "means, beta, u, and nocenter must all have length nvar",
        ));
    }

    let imat_len = p.checked_mul(p).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("nvar * nvar is too large")
    })?;
    if imat.len() != imat_len {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "imat must have length nvar * nvar",
        ));
    }

    let extra_work = p.checked_mul(3).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("required work length overflows usize")
    })?;
    let min_work_len = imat_len
        .checked_add(extra_work)
        .and_then(|len| len.checked_add(n))
        .ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("required work length overflows usize")
        })?;
    if work.len() < min_work_len {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "work must have length at least {}",
            min_work_len
        )));
    }

    let min_work2_len = n.checked_mul(2).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("required work2 length overflows usize")
    })?;
    if work2.len() < min_work2_len {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "work2 must have length at least {}",
            min_work2_len
        )));
    }

    Ok((n, p))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn agexact(
    maxiter: i32,
    nused: i32,
    nvar: i32,
    start: Vec<f64>,
    stop: Vec<f64>,
    event: Vec<i32>,
    covar: Vec<f64>,
    offset: Vec<f64>,
    strata: Vec<i32>,
    means: Vec<f64>,
    beta: Vec<f64>,
    u: Vec<f64>,
    imat: Vec<f64>,
    loglik: Vec<f64>,
    work: Vec<f64>,
    work2: Vec<i32>,
    eps: f64,
    tol_chol: f64,
    nocenter: Vec<i32>,
) -> PyResult<Py<PyDict>> {
    validate_agexact_inputs(
        maxiter, nused, nvar, &start, &stop, &event, &covar, &offset, &strata, &means, &beta, &u,
        &imat, &work, &work2, &nocenter,
    )?;

    let data = AgexactData {
        start,
        stop,
        event,
        covar,
        offset,
        strata,
        nocenter,
    };
    let state = AgexactState {
        means,
        beta,
        u,
        imat,
        loglik,
        work,
        work2,
    };
    let params = AgexactParams {
        maxiter,
        nused,
        nvar,
        eps,
        tol_chol,
    };
    agexact_impl(data, state, params)
}

fn agexact_impl(
    data: AgexactData,
    state: AgexactState,
    params: AgexactParams,
) -> PyResult<Py<PyDict>> {
    let AgexactData {
        start,
        stop,
        event,
        mut covar,
        offset,
        strata,
        nocenter,
    } = data;
    let AgexactState {
        mut means,
        mut beta,
        mut u,
        mut imat,
        mut loglik,
        mut work,
        mut work2,
    } = state;
    let AgexactParams {
        mut maxiter,
        nused,
        nvar,
        eps,
        tol_chol,
    } = params;
    let n = nused as usize;
    let nvar_usize = nvar as usize;
    let p = nvar_usize;
    let (cmat, rest) = work.split_at_mut(p * p);
    let (a, rest) = rest.split_at_mut(p);
    let (newbeta, rest) = rest.split_at_mut(p);
    let (score, newvar) = rest.split_at_mut(n);
    let atrisk = &mut work2[n..2 * n];
    if nvar_usize > 4 {
        let mean_updates: Vec<(usize, f64)> = (0..nvar_usize)
            .into_par_iter()
            .filter_map(|i| {
                if nocenter[i] == 0 {
                    Some((i, 0.0))
                } else {
                    let sum: f64 = (0..n).map(|j| covar[i * n + j]).sum();
                    Some((i, sum / n as f64))
                }
            })
            .collect();
        for (i, mean_val) in mean_updates {
            means[i] = mean_val;
            if nocenter[i] != 0 {
                for j in 0..n {
                    covar[i * n + j] -= mean_val;
                }
            }
        }
    } else {
        for i in 0..nvar_usize {
            if nocenter[i] == 0 {
                means[i] = 0.0;
            } else {
                let mut sum = 0.0;
                for j in 0..n {
                    sum += covar[i * n + j];
                }
                means[i] = sum / n as f64;
                let mean_val = means[i];
                for j in 0..n {
                    covar[i * n + j] -= mean_val;
                }
            }
        }
    }
    if n > PARALLEL_THRESHOLD_LARGE {
        let scores: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|person| {
                let mut zbeta = 0.0;
                for i in 0..nvar_usize {
                    zbeta += beta[i] * covar[i * n + person];
                }
                (zbeta + offset[person]).exp()
            })
            .collect();
        score.copy_from_slice(&scores);
    } else {
        for person in 0..n {
            let mut zbeta = 0.0;
            for i in 0..nvar_usize {
                zbeta += beta[i] * covar[i * n + person];
            }
            score[person] = (zbeta + offset[person]).exp();
        }
    }
    if loglik.len() < 2 {
        loglik.resize(2, 0.0);
    }
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
            while k < n {
                if stop[k] == time {
                    deaths += event[k];
                }
                if start[k] < time {
                    atrisk[nrisk] = k as i32;
                    nrisk += 1;
                }
                if strata[k] == 1 {
                    break;
                }
                k += 1;
            }
            let mut denom = 0.0;
            a.fill(0.0);
            cmat.fill(0.0);
            if deaths == 1 {
                for &at_risk_idx in atrisk.iter().take(nrisk) {
                    let k = at_risk_idx as usize;
                    let weight = score[k];
                    denom += weight;
                    for i in 0..nvar_usize {
                        let covar_ik = covar[i * n + k];
                        a[i] += weight * covar_ik;
                        for j in 0..=i {
                            let covar_jk = covar[j * n + k];
                            cmat[i * p + j] += weight * covar_ik * covar_jk;
                        }
                    }
                }
            } else {
                for indices in iter_combinations(0, nrisk, deaths as usize) {
                    newvar.fill(0.0);
                    let mut weight = 1.0;
                    for &idx in &indices {
                        let k = atrisk[idx] as usize;
                        weight *= score[k];
                        for i in 0..nvar_usize {
                            newvar[i] += covar[i * n + k];
                        }
                    }
                    denom += weight;
                    for i in 0..nvar_usize {
                        a[i] += weight * newvar[i];
                        for j in 0..=i {
                            cmat[i * p + j] += weight * newvar[i] * newvar[j];
                        }
                    }
                }
            }
            loglik[1] -= denom.ln();
            for i in 0..nvar_usize {
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
                    loglik[1] += score[k].ln();
                    for i in 0..nvar_usize {
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
    cholesky2(&mut imat[..p * p], p, tol_chol);
    chsolve2(&mut imat[..p * p], p, &mut a_copy);
    let sctest = a_copy.iter().zip(u.iter()).map(|(a, u)| a * u).sum::<f64>();
    if maxiter == 0 {
        chinv2(&mut imat[..p * p], p);
        for i in 0..p {
            for j in 0..i {
                imat[i * p + j] = imat[j * p + i];
            }
        }
        let final_flag = 0;
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("maxiter", maxiter)?;
            dict.set_item("covar", covar.to_vec())?;
            dict.set_item("means", means.to_vec())?;
            dict.set_item("beta", beta.to_vec())?;
            dict.set_item("u", u.to_vec())?;
            dict.set_item("imat", imat.to_vec())?;
            dict.set_item("loglik", loglik.to_vec())?;
            dict.set_item("flag", final_flag)?;
            dict.set_item("sctest", sctest)?;
            Ok(dict.into())
        })
    } else {
        let mut iter = 0;
        let mut halving = false;
        let mut newbeta_vec = newbeta.to_vec();
        let mut newlk = 0.0;
        while iter < maxiter {
            iter += 1;
            newlk = 0.0;
            u.fill(0.0);
            imat.fill(0.0);
            for person in 0..n {
                let mut zbeta = 0.0;
                for i in 0..nvar_usize {
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
                            break;
                        }
                        k += 1;
                    }
                    let mut denom = 0.0;
                    a.fill(0.0);
                    cmat.fill(0.0);
                    if deaths == 1 {
                        for &at_risk_idx in atrisk.iter().take(nrisk) {
                            let k = at_risk_idx as usize;
                            let weight = score[k];
                            denom += weight;
                            for i in 0..nvar_usize {
                                let covar_ik = covar[i * n + k];
                                a[i] += weight * covar_ik;
                                for j in 0..=i {
                                    cmat[i * p + j] += weight * covar_ik * covar[j * n + k];
                                }
                            }
                        }
                    } else {
                        for indices in iter_combinations(0, nrisk, deaths as usize) {
                            newvar.fill(0.0);
                            let mut weight = 1.0;
                            for &idx in &indices {
                                let k = atrisk[idx] as usize;
                                weight *= score[k];
                                for i in 0..nvar_usize {
                                    newvar[i] += covar[i * n + k];
                                }
                            }
                            denom += weight;
                            for i in 0..nvar_usize {
                                a[i] += weight * newvar[i];
                                for j in 0..=i {
                                    cmat[i * p + j] += weight * newvar[i] * newvar[j];
                                }
                            }
                        }
                    }
                    newlk -= denom.ln();
                    for i in 0..nvar_usize {
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
                            for i in 0..nvar_usize {
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
            if (1.0 - (loglik[1] / newlk)).abs() <= eps && !halving {
                loglik[1] = newlk;
                chinv2(&mut imat[..p * p], p);
                for i in 0..p {
                    for j in 0..i {
                        imat[i * p + j] = imat[j * p + i];
                    }
                }
                beta.copy_from_slice(&newbeta_vec);
                maxiter = iter;
                return Python::attach(|py| {
                    let dict = PyDict::new(py);
                    dict.set_item("maxiter", maxiter)?;
                    dict.set_item("covar", covar.to_vec())?;
                    dict.set_item("means", means.to_vec())?;
                    dict.set_item("beta", beta.to_vec())?;
                    dict.set_item("u", u.to_vec())?;
                    dict.set_item("imat", imat.to_vec())?;
                    dict.set_item("loglik", loglik.to_vec())?;
                    dict.set_item("flag", 0)?;
                    dict.set_item("sctest", sctest)?;
                    Ok(dict.into())
                });
            } else {
                if iter == maxiter {
                    break;
                }
                if newlk < loglik[1] {
                    halving = true;
                    for i in 0..nvar_usize {
                        newbeta_vec[i] = (newbeta_vec[i] + beta[i]) / 2.0;
                    }
                } else {
                    halving = false;
                    loglik[1] = newlk;
                    let mut u_copy = u.to_vec();
                    chsolve2(&mut imat[..p * p], p, &mut u_copy);
                    beta[..nvar_usize].copy_from_slice(&newbeta_vec[..nvar_usize]);
                    for i in 0..nvar_usize {
                        newbeta_vec[i] += u_copy[i];
                    }
                }
            }
        }
        loglik[1] = newlk;
        chinv2(&mut imat[..p * p], p);
        for i in 0..p {
            for j in 0..i {
                imat[i * p + j] = imat[j * p + i];
            }
        }
        beta.copy_from_slice(&newbeta_vec);
        let final_flag = CONVERGENCE_FLAG;
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("maxiter", maxiter)?;
            dict.set_item("covar", covar.to_vec())?;
            dict.set_item("means", means.to_vec())?;
            dict.set_item("beta", beta.to_vec())?;
            dict.set_item("u", u.to_vec())?;
            dict.set_item("imat", imat.to_vec())?;
            dict.set_item("loglik", loglik.to_vec())?;
            dict.set_item("flag", final_flag)?;
            dict.set_item("sctest", sctest)?;
            Ok(dict.into())
        })
    }
}
#[inline]
fn iter_combinations(
    start: usize,
    end: usize,
    k: usize,
) -> itertools::Combinations<std::ops::Range<usize>> {
    (start..end).combinations(k)
}
#[inline]
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
#[inline]
fn chsolve2(chol: &mut [f64], n: usize, b: &mut [f64]) {
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= chol[i * n + j] * b[j];
        }
        b[i] = sum / chol[i * n + i];
    }
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= chol[j * n + i] * b[j];
        }
        b[i] = sum / chol[i * n + i];
    }
}
#[inline]
fn chinv2(chol: &mut [f64], n: usize) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;

    #[test]
    fn validate_agexact_rejects_short_workspace() {
        Python::initialize();

        let err = validate_agexact_inputs(
            1,
            2,
            1,
            &[0.0, 0.0],
            &[1.0, 2.0],
            &[1, 0],
            &[1.0, 2.0],
            &[0.0, 0.0],
            &[0, 1],
            &[0.0],
            &[0.0],
            &[0.0],
            &[1.0],
            &[0.0, 0.0, 0.0],
            &[0, 0, 0, 0],
            &[1],
        )
        .unwrap_err();

        assert!(err.to_string().contains("work must have length at least"));
    }

    #[test]
    fn validate_agexact_rejects_length_mismatches() {
        Python::initialize();

        let err = validate_agexact_inputs(
            1,
            2,
            1,
            &[0.0],
            &[1.0, 2.0],
            &[1, 0],
            &[1.0, 2.0],
            &[0.0, 0.0],
            &[0, 1],
            &[0.0],
            &[0.0],
            &[0.0],
            &[1.0],
            &[0.0, 0.0, 0.0, 0.0, 0.0],
            &[0, 0, 0, 0],
            &[1],
        )
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("start, stop, event, offset, and strata must all have length nused")
        );
    }
}
