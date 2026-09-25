//! Test of the proportional-hazards assumption: a port of R survival's
//! `cox.zph()` (`R/cox.zph.R`) with its C kernels `src/zph1.c` (right-
//! censored data) and `src/zph2.c` ((start, stop] data).
//!
//! The test is the score test for the time-varying coefficient model
//! `beta(t) = beta + theta g(t)` at `theta = 0`: the kernels return the
//! `2p` score vector `(0, sum_i g(t_i) (x_i - xbar(t_i)))`, its `2p x 2p`
//! information, the Schoenfeld residuals and a per-stratum table of which
//! covariates vary.  For a penalized fit the penalty's second derivative is
//! added to both diagonal blocks of the information.  `cox.zph` then forms
//! one test per term (`terms`), optionally a single-df test along the fitted
//! linear predictor (`singledf`), and the global test, and returns the
//! scaled Schoenfeld residuals `y` against the transformed times `x` for
//! plotting.

use crate::core::risk_sweep::StratumSweep;
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::dist::pchisq;
use crate::internal::matrix::LuDecomposition;
use crate::regression::cox_optimizer::TieMethod;
use crate::regression::coxpenal::CoxpenalFit;
use crate::regression::coxph::{CoxPHFit, default_assign, validate_assign};
use ndarray::{Array2, s};
use pyo3::prelude::*;

/// The time transform of `cox.zph`.
#[derive(Debug, Clone, PartialEq)]
pub enum ZphTransform {
    Km,
    Rank,
    Identity,
    Log,
    /// A user function's values at each observation's (stop) time.
    Values(Vec<f64>),
}

impl ZphTransform {
    pub fn parse(name: &str) -> SurvivalResult<Self> {
        match name {
            "km" => Ok(Self::Km),
            "rank" => Ok(Self::Rank),
            "identity" => Ok(Self::Identity),
            "log" => Ok(Self::Log),
            other => Err(SurvivalError::invalid_input(format!(
                "Unrecognized transform '{other}'"
            ))),
        }
    }

    /// The result's `transform` label; R labels a function by its deparsed
    /// expression, or `"user"` when that spans several lines.
    pub fn r_name(&self) -> &'static str {
        match self {
            Self::Km => "km",
            Self::Rank => "rank",
            Self::Identity => "identity",
            Self::Log => "log",
            Self::Values(_) => "user",
        }
    }
}

/// `cox.zph`'s `transform` as it arrives from Python: a name, or the values
/// of a user function at the stop times.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "python", derive(pyo3::FromPyObject))]
pub enum ZphTransformArg {
    #[cfg_attr(feature = "python", pyo3(transparent))]
    Name(String),
    #[cfg_attr(feature = "python", pyo3(transparent))]
    Values(Vec<f64>),
}

/// What `cox.zph` reads from a penalized (`coxpenal.fit`) fit.
#[derive(Debug, Clone, PartialEq)]
pub struct ZphPenalty<'a> {
    /// `coxlist2$second`, the dense penalties' second derivative (see
    /// [`penalty_matrix`]).
    second: Option<&'a [f64]>,
    /// `fit$df`, which the tested terms read by position.
    df: Vec<f64>,
}

impl<'a> From<&'a CoxpenalFit> for ZphPenalty<'a> {
    fn from(fit: &'a CoxpenalFit) -> Self {
        let sparse = fit.pterms.iter().position(|&kind| kind == 2);
        // `fit$df` has an entry for every model term, but a sparse frailty
        // is not tested.  R pairs the tested terms with `fit$df[ii]`, which
        // is right when the frailty is the last term, the only order R
        // accepts ("subscript out of bounds" otherwise); moving the
        // frailty's entry last does the same for any order.
        let mut df = fit.df.clone();
        if let Some(term) = sparse.filter(|&term| term < df.len()) {
            df[term..].rotate_left(1);
        }
        Self {
            // coxpenal.fit returns coxlist2 only when there is no sparse term.
            second: fit
                .coxlist2
                .as_ref()
                .filter(|_| sparse.is_none())
                .map(|list| list.second.as_slice()),
            df,
        }
    }
}

/// One row of the `cox.zph` table.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct CoxZphTest {
    #[pyo3(get)]
    pub chisq: f64,
    /// Degrees of freedom; `fit$df` for a penalized fit, so fractional.
    #[pyo3(get)]
    pub df: f64,
    #[pyo3(get)]
    pub p: f64,
}

/// `cox.zph(fit)` result.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct CoxZph {
    /// One test per term, in `assign` order.
    #[pyo3(get)]
    pub table: Vec<CoxZphTest>,
    /// The GLOBAL test (`global = TRUE`); named `global_test` because
    /// `global` is a reserved word in Python.
    #[pyo3(get)]
    pub global_test: Option<CoxZphTest>,
    /// Transformed death times (`x`).
    #[pyo3(get)]
    pub x: Vec<f64>,
    /// Death times.
    #[pyo3(get)]
    pub time: Vec<f64>,
    /// Stratum code of each death (absent for an unstratified fit).
    #[pyo3(get)]
    pub strata: Option<Vec<i32>>,
    /// Scaled Schoenfeld residuals plus the coefficient, one column per
    /// term (`y`); `NaN` where a term plays no role in a stratum.
    #[pyo3(get)]
    pub y: Vec<Vec<f64>>,
    /// Variance of a row of `y` (`var`).
    #[pyo3(get)]
    pub var: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub transform: String,
}

/// What `zph1.c` / `zph2.c` return.
struct ZphKernel {
    u: Vec<f64>,
    imat: Array2<f64>,
    /// Schoenfeld residuals of the deaths in (stratum, time) order.
    schoen: Array2<f64>,
    /// Original row of each death, same order as `schoen`.
    death_rows: Vec<usize>,
    /// `nstrata x nvar`: the number of deaths in the stratum, or 0 when the
    /// covariate is constant there.
    used: Array2<f64>,
}

/// Port of `zph1.c` / `zph2.c`: one backward sweep per stratum
/// accumulating the score and information of the time-weighted model.
/// `keep` lists the design columns in use (aliased ones are dropped).
fn zph_kernel(fit: &CoxPHFit, gtime: &[f64], eta: &[f64], keep: &[usize]) -> ZphKernel {
    let nvar = keep.len();
    let efron = fit.method == TieMethod::Efron;
    let risk: Vec<f64> = eta.iter().map(|value| value.exp()).collect();
    let nstrata = fit.sorted.nstrata();
    let mut x = Array2::zeros((fit.n, nvar));
    for (k, &column) in keep.iter().enumerate() {
        x.column_mut(k).assign(&fit.x.column(column));
    }
    let mut used = Array2::zeros((nstrata, nvar));
    for stratum in 0..nstrata {
        let (start, end) = fit.sorted.bounds[stratum];
        let rows = &fit.sorted.order[start..end];
        let ndead_stratum = rows.iter().filter(|&&row| fit.status[row] == 1).count() as f64;
        for j in 0..nvar {
            let first = x[(rows[0], j)];
            if rows.iter().any(|&row| x[(row, j)] != first) {
                used[(stratum, j)] = ndead_stratum;
            }
        }
    }
    // "Recenter the X matrix to make the variance computation more stable":
    // subtract each column's plain mean (after `used` has been filled).
    for mut column in x.columns_mut() {
        let mean = column.iter().sum::<f64>() / fit.n as f64;
        column.mapv_inplace(|value| value - mean);
    }
    let mut u = vec![0.0; 2 * nvar];
    let mut imat = Array2::zeros((2 * nvar, 2 * nvar));
    let mut schoen_rows: Vec<(usize, Vec<f64>)> = Vec::new();
    for stratum in 0..nstrata {
        let (start, end) = fit.sorted.bounds[stratum];
        let rows = &fit.sorted.order[start..end];
        let sweep = StratumSweep {
            stop: &fit.time,
            entry: fit.entry.as_deref(),
            status: &fit.status,
            x: x.view(),
            weights: &fit.weights,
            risk: &risk,
            rows,
            second_moments: true,
        };
        let mut per_stratum: Vec<(usize, Vec<f64>)> = Vec::new();
        sweep.for_each_death_time(|death| {
            let d = death.ndead();
            let timewt = gtime[death.deaths[0]];
            let deadwt = death.tied.weight;
            for &row in death.deaths {
                for i in 0..nvar {
                    let value = fit.weights[row] * x[(row, i)];
                    u[i] += value;
                    u[i + nvar] += timewt * value;
                }
            }
            let steps = if efron && d > 1 { d } else { 1 };
            let wtave = deadwt / steps as f64;
            // Efron's mean of the step means for the Schoenfeld residuals.
            let mut mean = vec![0.0; nvar];
            for k in 0..steps {
                let step = if steps == 1 { 0 } else { k };
                let denom = death.efron_denom(step);
                for i in 0..nvar {
                    let temp2 = death.efron_a(step, i) / denom;
                    mean[i] += temp2 / steps as f64;
                    u[i] -= wtave * temp2;
                    u[i + nvar] -= timewt * wtave * temp2;
                    for j in 0..=i {
                        let temp = wtave
                            * (death.efron_cmat(step, i, j) - temp2 * death.efron_a(step, j))
                            / denom;
                        imat[(j, i)] += temp;
                        imat[(j, i + nvar)] += temp * timewt;
                        imat[(j + nvar, i + nvar)] += temp * timewt * timewt;
                    }
                }
            }
            // zph1.c lists tied deaths in data order, zph2.c (which walks
            // `order(-strata, -stop)`) in reverse data order.
            let residual = |row: usize| (row, (0..nvar).map(|i| x[(row, i)] - mean[i]).collect());
            if fit.entry.is_some() {
                per_stratum.extend(death.deaths.iter().map(|&row| residual(row)));
            } else {
                per_stratum.extend(death.deaths.iter().rev().map(|&row| residual(row)));
            }
        });
        per_stratum.reverse();
        schoen_rows.extend(per_stratum);
    }
    // Fill in the symmetric parts of the information matrix.
    for i in 0..nvar {
        for j in 0..i {
            imat[(i, j)] = imat[(j, i)];
            imat[(i, j + nvar)] = imat[(j, i + nvar)];
            imat[(i + nvar, j + nvar)] = imat[(j + nvar, i + nvar)];
        }
    }
    for i in 0..nvar {
        for j in 0..nvar {
            imat[(i + nvar, j)] = imat[(j, i + nvar)];
        }
    }
    let death_rows: Vec<usize> = schoen_rows.iter().map(|(row, _)| *row).collect();
    let schoen = Array2::from_shape_vec(
        (schoen_rows.len(), nvar),
        schoen_rows
            .into_iter()
            .flat_map(|(_, values)| values)
            .collect(),
    )
    .expect("one row per death");
    ZphKernel {
        u,
        imat,
        schoen,
        death_rows,
        used,
    }
}

/// Left-continuous Kaplan-Meier of the whole sample (`survfitKM` on
/// `factor(rep(1, n))` without weights, so the strata are pooled),
/// evaluated at each row's time: `1 - S(t-)`.  One pass over the rows in
/// time order; with (start, stop] data the risk set at `t` is
/// `#{stop >= t} - #{entry >= t}`, as every entry precedes its stop.
fn km_transform(time: &[f64], entry: Option<&[f64]>, status: &[i32]) -> Vec<f64> {
    let n = time.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| time[a].total_cmp(&time[b]));
    let sorted_entry = entry.map(|entry| {
        let mut sorted = entry.to_vec();
        sorted.sort_unstable_by(f64::total_cmp);
        sorted
    });
    let mut km = vec![0.0; n];
    let mut surv = 1.0;
    let mut position = 0;
    while position < n {
        let t = time[order[position]];
        let mut end = position;
        let mut deaths = 0usize;
        while end < n && time[order[end]] == t {
            km[order[end]] = 1.0 - surv;
            deaths += usize::from(status[order[end]] == 1);
            end += 1;
        }
        if deaths > 0 {
            let not_entered = sorted_entry
                .as_ref()
                .map_or(0, |entry| n - entry.partition_point(|&e| e < t));
            let nrisk = (n - position - not_entered) as f64;
            surv *= (nrisk - deaths as f64) / nrisk;
        }
        position = end;
    }
    km
}

/// R's `rank()`: average ranks for ties.
fn average_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut ranks = vec![0.0; n];
    let mut position = 0;
    while position < n {
        let mut end = position;
        while end < n && values[order[end]] == values[order[position]] {
            end += 1;
        }
        let average = (position + 1 + end) as f64 / 2.0;
        for &row in &order[position..end] {
            ranks[row] = average;
        }
        position = end;
    }
    ranks
}

fn solve(matrix: &Array2<f64>, rhs: &[f64]) -> SurvivalResult<Vec<f64>> {
    LuDecomposition::decompose(matrix)?.solve(rhs)
}

/// `coxlist2$second` as the penalty matrix `tmat` over the non-aliased
/// coefficients `keep` of the `nfull` in the fit.  It holds either all
/// `nfull x nfull` entries (column-major) or, for diagonal penalties such as
/// `ridge()`, the diagonal.  R reads the latter with `matrix(second, nfull)`,
/// which recycles it into `tmat[i, j] = second[i]`; here it is the diagonal
/// matrix it stands for.
fn penalty_matrix(second: &[f64], keep: &[usize], nfull: usize) -> SurvivalResult<Array2<f64>> {
    let dense = second.len() == nfull * nfull;
    if !dense && second.len() != nfull {
        return Err(SurvivalError::invalid_input(format!(
            "the penalty's second derivative has {} values for {nfull} coefficients",
            second.len()
        )));
    }
    Ok(Array2::from_shape_fn(
        (keep.len(), keep.len()),
        |(a, b)| match (keep[a], keep[b]) {
            (i, j) if dense => second[j * nfull + i],
            (i, j) if i == j => second[i],
            _ => 0.0,
        },
    ))
}

fn zph_test(chisq: f64, df: f64) -> CoxZphTest {
    CoxZphTest {
        chisq,
        df,
        p: pchisq(chisq, df, false, false),
    }
}

fn submatrix(matrix: &Array2<f64>, rows: &[usize], cols: &[usize]) -> Array2<f64> {
    let mut out = Array2::zeros((rows.len(), cols.len()));
    for (i, &r) in rows.iter().enumerate() {
        for (j, &c) in cols.iter().enumerate() {
            out[(i, j)] = matrix[(r, c)];
        }
    }
    out
}

/// `cox.zph(fit, transform, terms, singledf, global)` (`global_test` is
/// R's `global`).  `assign` lists the columns of each term (default: one
/// term per coefficient); it is ignored with `terms = FALSE`.  A penalized
/// fit passes its [`ZphPenalty`], and `fit` is then its `coxph` part.
pub fn cox_zph(
    fit: &CoxPHFit,
    transform: &ZphTransform,
    terms: bool,
    singledf: bool,
    global_test: bool,
    assign: Option<&[Vec<usize>]>,
    penalty: Option<&ZphPenalty>,
) -> SurvivalResult<CoxZph> {
    if fit.nvar() == 0 {
        return Err(SurvivalError::invalid_input(
            "there are no score residuals for a Null model",
        ));
    }
    let singledf = singledf && terms;
    let default = default_assign(fit.nvar());
    let assign: Vec<Vec<usize>> = if terms {
        let assign = assign.unwrap_or(&default);
        validate_assign(assign, fit.nvar())?;
        assign.to_vec()
    } else {
        default
    };
    // Aliased coefficients (NA) drop out of X and of the terms.
    let keep: Vec<usize> = (0..fit.nvar())
        .filter(|&j| !fit.coefficients[j].is_nan())
        .collect();
    let nvar = keep.len();
    if nvar == 0 {
        return Err(SurvivalError::invalid_input("every coefficient is aliased"));
    }
    let coef: Vec<f64> = keep.iter().map(|&j| fit.coefficients[j]).collect();
    let tmat = penalty
        .and_then(|penalty| penalty.second)
        .map(|second| penalty_matrix(second, &keep, fit.nvar()))
        .transpose()?;
    let assign: Vec<Vec<usize>> = assign
        .iter()
        .map(|columns| {
            columns
                .iter()
                .filter_map(|column| keep.iter().position(|&k| k == *column))
                .collect::<Vec<_>>()
        })
        .filter(|columns: &Vec<usize>| !columns.is_empty())
        .collect();
    let nterm = assign.len();

    // Centring the linear predictors only reduces round-off.
    let mean_lp = fit.linear_predictors.iter().sum::<f64>() / fit.n as f64;
    let eta: Vec<f64> = fit
        .linear_predictors
        .iter()
        .map(|lp| lp - mean_lp)
        .collect();
    let ttimes: Vec<f64> = match transform {
        ZphTransform::Identity => fit.time.clone(),
        ZphTransform::Rank => average_ranks(&fit.time),
        ZphTransform::Log => fit.time.iter().map(|t| t.ln()).collect(),
        ZphTransform::Km => km_transform(&fit.time, fit.entry.as_deref(), &fit.status),
        ZphTransform::Values(values) => {
            if values.len() != fit.n || values.iter().any(|value| !value.is_finite()) {
                return Err(SurvivalError::invalid_input(
                    "the transform must give one finite value per observation",
                ));
            }
            values.clone()
        }
    };
    let event: Vec<bool> = fit.status.iter().map(|&s| s == 1).collect();
    let nevent = event.iter().filter(|&&e| e).count();
    let event_mean = ttimes
        .iter()
        .zip(&event)
        .filter(|(_, e)| **e)
        .map(|(t, _)| t)
        .sum::<f64>()
        / nevent as f64;
    let gtime: Vec<f64> = ttimes.iter().map(|t| t - event_mean).collect();

    let kernel = zph_kernel(fit, &gtime, &eta, &keep);
    // R's `imatr`: the information plus the penalty in both diagonal blocks.
    let mut imatr = kernel.imat;
    if let Some(tmat) = &tmat {
        for block in [s![..nvar, ..nvar], s![nvar.., nvar..]] {
            let mut view = imatr.slice_mut(block);
            view += tmat;
        }
    }
    let all: Vec<usize> = (0..nvar).collect();
    let mut table = Vec::with_capacity(nterm);
    for (ii, columns) in assign.iter().enumerate() {
        let kk: Vec<usize> = all
            .iter()
            .copied()
            .chain(columns.iter().map(|&j| j + nvar))
            .collect();
        let imat = submatrix(&imatr, &kk, &kk);
        let test = if singledf && columns.len() > 1 {
            let inverse = LuDecomposition::decompose(&imat)?.inverse()?;
            let offset = nvar;
            let t1: f64 = columns.iter().map(|&j| coef[j] * kernel.u[j + nvar]).sum();
            let mut quad = 0.0;
            for (a, &ja) in columns.iter().enumerate() {
                for (b, &jb) in columns.iter().enumerate() {
                    quad += coef[ja] * inverse[(offset + a, offset + b)] * coef[jb];
                }
            }
            zph_test(t1 * t1 * quad, 1.0)
        } else {
            let mut u = vec![0.0; kk.len()];
            for (position, &j) in columns.iter().enumerate() {
                u[nvar + position] = kernel.u[j + nvar];
            }
            let solved = solve(&imat, &u)?;
            // R takes `fit$df[ii]` by position among the tested terms,
            // aliased ones dropped (NA past its end).
            let term_df = match penalty {
                Some(penalty) => penalty.df.get(ii).copied().unwrap_or(f64::NAN),
                None => columns.len() as f64,
            };
            zph_test(solved.iter().zip(&u).map(|(s, u)| s * u).sum(), term_df)
        };
        table.push(test);
    }
    let global_test = if global_test {
        let mut u = vec![0.0; 2 * nvar];
        u[nvar..].copy_from_slice(&kernel.u[nvar..]);
        let solved = solve(&imatr, &u)?;
        let chisq: f64 = solved.iter().zip(&u).map(|(s, u)| s * u).sum();
        Some(zph_test(
            chisq,
            penalty.map_or(nvar as f64, |penalty| penalty.df.iter().sum()),
        ))
    } else {
        None
    };

    // A factor level unused in a stratum still counts as used when any
    // column of its term varies there.
    let mut used = kernel.used.clone();
    for columns in &assign {
        if columns.len() > 1 {
            for s in 0..used.nrows() {
                let max = columns.iter().map(|&j| used[(s, j)]).fold(0.0, f64::max);
                if columns.iter().any(|&j| used[(s, j)] == 0.0) {
                    for &j in columns {
                        used[(s, j)] = max;
                    }
                }
            }
        }
    }
    let mut wtmat = Array2::zeros((nvar, nvar));
    for s in 0..used.nrows() {
        for i in 0..nvar {
            for j in 0..nvar {
                wtmat[(i, j)] += used[(s, i)].min(used[(s, j)]);
            }
        }
    }
    let mut vmean = Array2::zeros((nvar, nvar));
    for i in 0..nvar {
        for j in 0..nvar {
            let weight = if wtmat[(i, j)] == 0.0 {
                1.0
            } else {
                wtmat[(i, j)]
            };
            vmean[(i, j)] = imatr[(i, j)] / weight;
        }
    }

    // Collapse multi-column terms onto their linear predictor.
    let mut sresid = kernel.schoen.clone();
    let mut used_terms = used.clone();
    if terms && assign.iter().any(|columns| columns.len() > 1) {
        let mut temp = Array2::zeros((nvar, nterm));
        for (t, columns) in assign.iter().enumerate() {
            for &j in columns {
                temp[(j, t)] = if columns.len() == 1 { 1.0 } else { coef[j] };
            }
        }
        sresid = sresid.dot(&temp);
        vmean = temp.t().dot(&vmean).dot(&temp);
        let firsts: Vec<usize> = assign.iter().map(|columns| columns[0]).collect();
        used_terms = submatrix(&used, &(0..used.nrows()).collect::<Vec<_>>(), &firsts);
    }
    let ncol = sresid.ncols();

    // Rescale the residuals within each stratum by the inverse of vmean
    // over the covariates used there.
    let death_strata: Vec<usize> = kernel
        .death_rows
        .iter()
        .map(|&row| fit.sorted.stratum_index[row])
        .collect();
    let mut y = sresid.clone();
    for s in 0..used_terms.nrows() {
        let k: Vec<usize> = (0..ncol).filter(|&j| used_terms[(s, j)] > 0.0).collect();
        let rows: Vec<usize> = (0..y.nrows()).filter(|&g| death_strata[g] == s).collect();
        if k.is_empty() || rows.is_empty() {
            continue;
        }
        let vk = submatrix(&vmean, &k, &k);
        if k.len() == 1 {
            for &g in &rows {
                y[(g, k[0])] = sresid[(g, k[0])] / vk[(0, 0)];
            }
        } else {
            let lu = LuDecomposition::decompose(&vk)?;
            for &g in &rows {
                let rhs: Vec<f64> = k.iter().map(|&j| sresid[(g, j)]).collect();
                let solved = lu.solve(&rhs)?;
                for (position, &j) in k.iter().enumerate() {
                    y[(g, j)] = solved[position];
                }
            }
        }
        for &g in &rows {
            for j in 0..ncol {
                if !k.contains(&j) {
                    y[(g, j)] = f64::NAN;
                }
            }
        }
    }
    // Add the coefficient (1 for a multi-column term's linear predictor).
    for (t, columns) in assign.iter().enumerate() {
        let shift = if columns.len() == 1 {
            coef[columns[0]]
        } else {
            1.0
        };
        for g in 0..y.nrows() {
            y[(g, t)] += shift;
        }
    }
    let var = LuDecomposition::decompose(&vmean)?.inverse()?;

    Ok(CoxZph {
        table,
        global_test,
        x: kernel.death_rows.iter().map(|&row| ttimes[row]).collect(),
        time: kernel.death_rows.iter().map(|&row| fit.time[row]).collect(),
        strata: fit
            .strata
            .as_ref()
            .map(|strata| kernel.death_rows.iter().map(|&row| strata[row]).collect()),
        y: y.outer_iter().map(|row| row.to_vec()).collect(),
        var: var.outer_iter().map(|row| row.to_vec()).collect(),
        transform: transform.r_name().to_string(),
    })
}

/// `cox.zph(fit, transform, terms, singledf, global)`; `global_test` is
/// R's `global` (a reserved word in Python), `assign` lists the columns
/// of each term, `transform` is a name or a user function's values at the
/// stop times, and a penalized fit passes its `coxpenal` fit as `penalized`
/// (with its `coxph` part as `fit`).
#[pyfunction(name = "cox_zph")]
#[pyo3(
    signature = (fit, transform = ZphTransformArg::Name("km".to_string()), terms = true, singledf = false, global_test = true, assign = None, penalized = None),
    text_signature = "(fit, transform='km', terms=True, singledf=False, global_test=True, assign=None, penalized=None)"
)]
pub fn cox_zph_py(
    fit: &CoxPHFit,
    transform: ZphTransformArg,
    terms: bool,
    singledf: bool,
    global_test: bool,
    assign: Option<Vec<Vec<usize>>>,
    penalized: Option<&CoxpenalFit>,
) -> PyResult<CoxZph> {
    let transform = match transform {
        ZphTransformArg::Name(name) => ZphTransform::parse(&name)?,
        ZphTransformArg::Values(values) => ZphTransform::Values(values),
    };
    Ok(cox_zph(
        fit,
        &transform,
        terms,
        singledf,
        global_test,
        assign.as_deref(),
        penalized.map(ZphPenalty::from).as_ref(),
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regression::coxpenal::{
        CoxpenalData, CoxpenalOptions, FrailtyFamily, ModelTerm, PenaltyTerm,
    };
    use crate::regression::coxph::{CoxphData, CoxphOptions};

    const TIME: [f64; 10] = [1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    const START: [f64; 10] = [0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 3.0, 0.0];
    const STATUS: [i32; 10] = [1, 1, 1, 0, 1, 1, 0, 1, 1, 1];
    const X1: [f64; 10] = [0.0, 0.4, 0.8, 0.2, 1.0, 1.4, 0.6, 1.2, 1.6, 1.8];
    const X2: [f64; 10] = [0.2, 0.16, 0.62, -0.07, 0.95, 0.61, 0.49, 0.68, 1.24, 0.97];

    /// The toy data with `shift` added to the first covariate.
    fn fitted_shifted(entry: bool, strata: bool, shift: f64) -> CoxPHFit {
        let x = Array2::from_shape_fn(
            (10, 2),
            |(row, col)| {
                if col == 0 { X1[row] + shift } else { X2[row] }
            },
        );
        let data = CoxphData::try_new(
            TIME.to_vec(),
            entry.then_some(START.to_vec()),
            STATUS.to_vec(),
            x,
            None,
            strata.then_some(vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
            None,
        )
        .unwrap();
        CoxPHFit::fit(data, CoxphOptions::default()).unwrap()
    }

    fn fitted(entry: bool, strata: bool) -> CoxPHFit {
        fitted_shifted(entry, strata, 0.0)
    }

    fn zph(fit: &CoxPHFit, transform: &ZphTransform) -> CoxZph {
        cox_zph(fit, transform, true, false, true, None, None).unwrap()
    }

    fn penalized_zph(fit: &CoxpenalFit) -> CoxZph {
        let penalty = ZphPenalty::from(fit);
        cox_zph(
            &fit.coxph,
            &ZphTransform::Km,
            true,
            false,
            true,
            None,
            Some(&penalty),
        )
        .unwrap()
    }

    fn chisqs(zph: &CoxZph) -> Vec<f64> {
        zph.table
            .iter()
            .chain(&zph.global_test)
            .map(|test| test.chisq)
            .collect()
    }

    fn assert_rel(actual: &[f64], expected: &[f64], rtol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!(
                (a - e).abs() <= rtol * e.abs(),
                "{actual:?} != {expected:?}"
            );
        }
    }

    #[test]
    fn zph_tables_are_positive_and_consistent_across_transforms() {
        for (entry, strata) in [(false, false), (true, false), (false, true), (true, true)] {
            let fit = fitted(entry, strata);
            for transform in [
                ZphTransform::Km,
                ZphTransform::Rank,
                ZphTransform::Identity,
                ZphTransform::Log,
            ] {
                let zph = zph(&fit, &transform);
                assert_eq!(zph.table.len(), 2);
                for test in &zph.table {
                    assert!(test.chisq >= 0.0, "{transform:?}: {}", test.chisq);
                    assert_eq!(test.df, 1.0);
                    assert!((0.0..=1.0).contains(&test.p));
                }
                let global = zph.global_test.as_ref().unwrap();
                assert_eq!(global.df, 2.0);
                assert!(global.chisq >= 0.0);
                assert_eq!(zph.x.len(), fit.nevent);
                assert_eq!(zph.y.len(), fit.nevent);
                assert_eq!(zph.var.len(), 2);
                assert_eq!(zph.transform, transform.r_name());
                assert!(zph.y.iter().flatten().all(|v| v.is_finite()));
            }
        }
    }

    #[test]
    fn km_transform_is_left_continuous() {
        let km = km_transform(&TIME, None, &STATUS);
        // The first death time sees S(t-) = 1.
        assert_eq!(km[0], 0.0);
        assert!(km[1] > 0.0 && km[1] < 1.0);
        assert_eq!(km[1], km[2]);
        assert_eq!(
            average_ranks(&[3.0, 1.0, 3.0, 2.0]),
            vec![3.5, 1.0, 3.5, 2.0]
        );
    }

    /// `1 - S(t-)` with the risk set counted from its definition,
    /// `#{entry < t <= stop}`, at every distinct time.
    fn km_transform_by_definition(time: &[f64], entry: Option<&[f64]>, status: &[i32]) -> Vec<f64> {
        let mut times = time.to_vec();
        times.sort_by(f64::total_cmp);
        times.dedup();
        let mut surv_before = Vec::with_capacity(times.len());
        let mut surv = 1.0;
        for &t in &times {
            surv_before.push(surv);
            let rows = 0..time.len();
            let deaths = rows
                .clone()
                .filter(|&row| time[row] == t && status[row] == 1)
                .count();
            let nrisk = rows
                .filter(|&row| time[row] >= t && entry.is_none_or(|entry| entry[row] < t))
                .count() as f64;
            if deaths > 0 {
                surv *= (nrisk - deaths as f64) / nrisk;
            }
        }
        time.iter()
            .map(|t| 1.0 - surv_before[times.partition_point(|u| u < t)])
            .collect()
    }

    #[test]
    fn km_transform_counts_the_risk_set_like_its_definition() {
        assert_eq!(
            km_transform(&TIME, Some(&START), &STATUS),
            km_transform_by_definition(&TIME, Some(&START), &STATUS)
        );
        // Heavily tied times, and entries that coincide with other rows'
        // stops (not yet at risk there).
        let n = 300;
        let time: Vec<f64> = (0..n).map(|i| (1 + (i * 37) % 23) as f64).collect();
        let entry: Vec<f64> = (0..n)
            .map(|i| {
                if i % 3 == 0 {
                    0.0
                } else {
                    ((i * 11) % 29) as f64 % time[i]
                }
            })
            .collect();
        let status: Vec<i32> = (0..n).map(|i| i32::from(i % 4 != 1)).collect();
        assert_eq!(
            km_transform(&time, None, &status),
            km_transform_by_definition(&time, None, &status)
        );
        assert_eq!(
            km_transform(&time, Some(&entry), &status),
            km_transform_by_definition(&time, Some(&entry), &status)
        );
    }

    // R: d <- data.frame(time, start, status, x1, x2) as in `fitted_shifted`,
    //    d$big <- d$x1 + 1e6
    //    cox.zph(coxph(Surv(time, status) ~ big + x2, d))$table[, "chisq"]
    //    cox.zph(coxph(Surv(start, time, status) ~ big + x2, d))$table[, "chisq"]
    #[test]
    fn a_large_covariate_mean_is_centred_away() {
        let right = zph(&fitted_shifted(false, false, 1e6), &ZphTransform::Km);
        assert_rel(
            &chisqs(&right),
            &[
                0.0384689811357126,
                0.003106708078431738,
                0.08717811043355465,
            ],
            1e-8,
        );
        let counting = zph(&fitted_shifted(true, false, 1e6), &ZphTransform::Km);
        assert_rel(
            &chisqs(&counting),
            &[0.005064211207604128, 0.3229318187886463, 0.5629667393851748],
            1e-8,
        );
    }

    #[test]
    fn supplied_transform_values_feed_the_same_test() {
        let fit = fitted(true, true);
        let identity = zph(&fit, &ZphTransform::Identity);
        let supplied = zph(&fit, &ZphTransform::Values(fit.time.clone()));
        assert_eq!(chisqs(&supplied), chisqs(&identity));
        assert_eq!(supplied.x, identity.x);
        assert_eq!(supplied.transform, "user");
        for bad in [vec![1.0; 3], {
            let mut values = fit.time.clone();
            values[4] = f64::NAN;
            values
        }] {
            assert!(matches!(
                cox_zph(
                    &fit,
                    &ZphTransform::Values(bad),
                    true,
                    false,
                    true,
                    None,
                    None
                ),
                Err(SurvivalError::InvalidInput(_))
            ));
        }
    }

    #[test]
    fn penalty_matrix_reads_full_and_diagonal_second_derivatives() {
        // column-major 3 x 3, restricted to coefficients 0 and 2
        let full = [1.0, 2.0, 3.0, 2.0, 5.0, 6.0, 3.0, 6.0, 9.0];
        assert_eq!(
            penalty_matrix(&full, &[0, 2], 3).unwrap(),
            ndarray::arr2(&[[1.0, 3.0], [3.0, 9.0]])
        );
        assert_eq!(
            penalty_matrix(&[4.0, 5.0, 0.0], &[0, 1, 2], 3).unwrap(),
            ndarray::arr2(&[[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.0]])
        );
        assert!(penalty_matrix(&[1.0, 2.0], &[0, 1, 2], 3).is_err());
    }

    #[test]
    fn penalized_df_follow_r_positional_rule() {
        let fit = fitted(false, false);
        let penalty = ZphPenalty {
            second: Some(&[0.5, 0.0]),
            df: vec![0.7, 0.9, 3.0],
        };
        let penalized = cox_zph(
            &fit,
            &ZphTransform::Km,
            true,
            false,
            true,
            None,
            Some(&penalty),
        )
        .unwrap();
        let df: Vec<f64> = penalized.table.iter().map(|test| test.df).collect();
        assert_eq!(df, vec![0.7, 0.9]);
        let global = penalized.global_test.as_ref().unwrap();
        assert!((global.df - 4.6).abs() < 1e-12);
        assert_eq!(global.p, pchisq(global.chisq, global.df, false, false));
        // The penalty enters the information, so the tests change.
        assert_ne!(
            chisqs(&penalized)[0],
            chisqs(&zph(&fit, &ZphTransform::Km))[0]
        );
        // Past the end of fit$df R reads NA.
        let one_df = ZphPenalty {
            second: None,
            df: vec![0.7],
        };
        let short = cox_zph(
            &fit,
            &ZphTransform::Km,
            true,
            false,
            false,
            None,
            Some(&one_df),
        )
        .unwrap();
        assert_eq!(short.table[0].df, 0.7);
        assert!(short.table[1].df.is_nan() && short.table[1].p.is_nan());
    }

    /// `ridge(x1, theta = 1) + x2` with a sparse gamma frailty (theta 0.5)
    /// over five pairs of the toy data as term `position`.
    fn frailty_fit(position: usize) -> CoxpenalFit {
        let mut columns = vec![X1, X2];
        columns.insert(position, [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0]);
        let x = Array2::from_shape_fn((10, 3), |(row, col)| columns[col][row]);
        let frailty = PenaltyTerm::frailty(
            FrailtyFamily::Gamma,
            true,
            Some(0.5),
            None,
            None,
            None,
            false,
            None,
        )
        .unwrap();
        let ridge = PenaltyTerm::ridge(Some(1.0), None, 1e-5, false).unwrap();
        let x1 = usize::from(position == 0);
        let terms = (0..3)
            .map(|col| ModelTerm {
                columns: vec![col],
                penalty: if col == position {
                    Some(frailty.clone())
                } else if col == x1 {
                    Some(ridge.clone())
                } else {
                    None
                },
            })
            .collect();
        let data = CoxpenalData::try_new(
            TIME.to_vec(),
            None,
            STATUS.to_vec(),
            x,
            None,
            None,
            None,
            terms,
        )
        .unwrap();
        CoxpenalFit::fit(data, CoxpenalOptions::default()).unwrap()
    }

    #[test]
    fn a_sparse_frailty_keeps_its_df_away_from_the_tested_terms() {
        let last = frailty_fit(2);
        // The fit keeps the ridge's coxlist2, which R's coxpenal.fit drops
        // beside a sparse term, so cox.zph adds no penalty.
        assert!(last.coxlist2.is_some());
        assert_eq!(ZphPenalty::from(&last).second, None);
        let expected = penalized_zph(&last);
        let df: Vec<f64> = expected.table.iter().map(|test| test.df).collect();
        assert_eq!(df, last.df[..2]);
        let global = expected.global_test.as_ref().unwrap();
        assert_eq!(global.df, last.df.iter().sum::<f64>());
        for position in [0, 1] {
            let fit = frailty_fit(position);
            let zph = penalized_zph(&fit);
            let rows = zph.table.iter().chain(&zph.global_test);
            for (row, want) in rows.zip(expected.table.iter().chain(&expected.global_test)) {
                assert!((row.chisq - want.chisq).abs() < 1e-8, "{position}");
                assert!((row.df - want.df).abs() < 1e-8, "{position}");
            }
        }
    }

    #[test]
    fn terms_collapse_multi_column_terms_onto_the_linear_predictor() {
        let fit = fitted(false, false);
        let joint_terms = [vec![0, 1]];
        let joint = cox_zph(
            &fit,
            &ZphTransform::Km,
            true,
            false,
            true,
            Some(&joint_terms),
            None,
        )
        .unwrap();
        assert_eq!(joint.table.len(), 1);
        assert_eq!(joint.table[0].df, 2.0);
        assert_eq!(joint.y[0].len(), 1);
        let single = cox_zph(
            &fit,
            &ZphTransform::Km,
            true,
            true,
            false,
            Some(&joint_terms),
            None,
        )
        .unwrap();
        assert_eq!(single.table[0].df, 1.0);
        assert!(single.global_test.is_none());
        let global = zph(&fit, &ZphTransform::Km);
        assert!((joint.table[0].chisq - global.global_test.unwrap().chisq).abs() < 1e-10);
    }
}
