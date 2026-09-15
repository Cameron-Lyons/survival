//! Test of the proportional-hazards assumption: a port of R survival's
//! `cox.zph()` (`R/cox.zph.R`) with its C kernels `src/zph1.c` (right-
//! censored data) and `src/zph2.c` ((start, stop] data).
//!
//! The test is the score test for the time-varying coefficient model
//! `beta(t) = beta + theta g(t)` at `theta = 0`: the kernels return the
//! `2p` score vector `(0, sum_i g(t_i) (x_i - xbar(t_i)))`, its `2p x 2p`
//! information, the Schoenfeld residuals and a per-stratum table of which
//! covariates vary.  `cox.zph` then forms one test per term (`terms`),
//! optionally a single-df test along the fitted linear predictor
//! (`singledf`), and the global test, and returns the scaled Schoenfeld
//! residuals `y` against the transformed times `x` for plotting.

use crate::core::risk_sweep::StratumSweep;
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::matrix::LuDecomposition;
use crate::internal::statistical::chi2_sf;
use crate::regression::cox_optimizer::TieMethod;
use crate::regression::coxph::{CoxPHFit, default_assign, validate_assign};
use ndarray::Array2;
use pyo3::prelude::*;

/// The time transform of `cox.zph`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZphTransform {
    Km,
    Rank,
    Identity,
    Log,
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

    pub fn r_name(self) -> &'static str {
        match self {
            Self::Km => "km",
            Self::Rank => "rank",
            Self::Identity => "identity",
            Self::Log => "log",
        }
    }
}

/// One row of the `cox.zph` table.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct CoxZphTest {
    #[pyo3(get)]
    pub chisq: f64,
    #[pyo3(get)]
    pub df: usize,
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
    let mut u = vec![0.0; 2 * nvar];
    let mut imat = Array2::zeros((2 * nvar, 2 * nvar));
    let mut used = Array2::zeros((nstrata, nvar));
    let mut schoen_rows: Vec<(usize, Vec<f64>)> = Vec::new();
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
/// `factor(rep(1, n))` without weights), evaluated at each row's time:
/// `1 - S(t-)`.
fn km_transform(fit: &CoxPHFit) -> Vec<f64> {
    let n = fit.n;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| fit.time[a].total_cmp(&fit.time[b]).then(a.cmp(&b)));
    let mut times = Vec::new();
    let mut surv = Vec::new();
    let mut running = 1.0;
    let mut position = 0;
    while position < n {
        let t = fit.time[order[position]];
        let mut end = position;
        let mut deaths = 0usize;
        while end < n && fit.time[order[end]] == t {
            deaths += usize::from(fit.status[order[end]] == 1);
            end += 1;
        }
        let at_risk = (0..n)
            .filter(|&row| {
                fit.time[row] >= t && fit.entry.as_ref().is_none_or(|entry| entry[row] < t)
            })
            .count();
        if deaths > 0 {
            running *= 1.0 - deaths as f64 / at_risk as f64;
        }
        times.push(t);
        surv.push(running);
        position = end;
    }
    fit.time
        .iter()
        .map(|&t| {
            let index = times.partition_point(|&time| time < t);
            let before = if index == 0 { 1.0 } else { surv[index - 1] };
            1.0 - before
        })
        .collect()
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
/// term per coefficient); it is ignored with `terms = FALSE`.
pub fn cox_zph(
    fit: &CoxPHFit,
    transform: ZphTransform,
    terms: bool,
    singledf: bool,
    global_test: bool,
    assign: Option<&[Vec<usize>]>,
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
        ZphTransform::Km => km_transform(fit),
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
    let all: Vec<usize> = (0..nvar).collect();
    let mut table = Vec::with_capacity(nterm);
    for columns in &assign {
        let kk: Vec<usize> = all
            .iter()
            .copied()
            .chain(columns.iter().map(|&j| j + nvar))
            .collect();
        let imat = submatrix(&kernel.imat, &kk, &kk);
        let (chisq, df) = if singledf && columns.len() > 1 {
            let inverse = LuDecomposition::decompose(&imat)?.inverse()?;
            let offset = nvar;
            let t1: f64 = columns.iter().map(|&j| coef[j] * kernel.u[j + nvar]).sum();
            let mut quad = 0.0;
            for (a, &ja) in columns.iter().enumerate() {
                for (b, &jb) in columns.iter().enumerate() {
                    quad += coef[ja] * inverse[(offset + a, offset + b)] * coef[jb];
                }
            }
            (t1 * t1 * quad, 1)
        } else {
            let mut u = vec![0.0; kk.len()];
            for (position, &j) in columns.iter().enumerate() {
                u[nvar + position] = kernel.u[j + nvar];
            }
            let solved = solve(&imat, &u)?;
            (
                solved.iter().zip(&u).map(|(s, u)| s * u).sum(),
                columns.len(),
            )
        };
        table.push(CoxZphTest {
            chisq,
            df,
            p: chi2_sf(chisq, df),
        });
    }
    let global_test = if global_test {
        let mut u = vec![0.0; 2 * nvar];
        u[nvar..].copy_from_slice(&kernel.u[nvar..]);
        let solved = solve(&kernel.imat, &u)?;
        let chisq: f64 = solved.iter().zip(&u).map(|(s, u)| s * u).sum();
        Some(CoxZphTest {
            chisq,
            df: nvar,
            p: chi2_sf(chisq, nvar),
        })
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
            vmean[(i, j)] = kernel.imat[(i, j)] / weight;
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
/// R's `global` (a reserved word in Python) and `assign` lists the columns
/// of each term.
#[pyfunction(name = "cox_zph")]
#[pyo3(signature = (fit, transform = "km", terms = true, singledf = false, global_test = true, assign = None))]
pub fn cox_zph_py(
    fit: &CoxPHFit,
    transform: &str,
    terms: bool,
    singledf: bool,
    global_test: bool,
    assign: Option<Vec<Vec<usize>>>,
) -> PyResult<CoxZph> {
    Ok(cox_zph(
        fit,
        ZphTransform::parse(transform)?,
        terms,
        singledf,
        global_test,
        assign.as_deref(),
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regression::coxph::{CoxphData, CoxphOptions};

    fn fitted(entry: bool, strata: bool) -> CoxPHFit {
        let time = vec![1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let start = vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 3.0, 0.0];
        let status = vec![1, 1, 1, 0, 1, 1, 0, 1, 1, 1];
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                0.0, 0.2, 0.4, 0.16, 0.8, 0.62, 0.2, -0.07, 1.0, 0.95, 1.4, 0.61, 0.6, 0.49, 1.2,
                0.68, 1.6, 1.24, 1.8, 0.97,
            ],
        )
        .unwrap();
        let data = CoxphData::try_new(
            time,
            entry.then_some(start),
            status,
            x,
            None,
            strata.then_some(vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
            None,
        )
        .unwrap();
        CoxPHFit::fit(data, CoxphOptions::default()).unwrap()
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
                let zph = cox_zph(&fit, transform, true, false, true, None).unwrap();
                assert_eq!(zph.table.len(), 2);
                for test in &zph.table {
                    assert!(test.chisq >= 0.0, "{transform:?}: {}", test.chisq);
                    assert_eq!(test.df, 1);
                    assert!((0.0..=1.0).contains(&test.p));
                }
                let global = zph.global_test.as_ref().unwrap();
                assert_eq!(global.df, 2);
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
        let fit = fitted(false, false);
        let km = km_transform(&fit);
        // The first death time sees S(t-) = 1.
        assert_eq!(km[0], 0.0);
        assert!(km[1] > 0.0 && km[1] < 1.0);
        assert_eq!(km[1], km[2]);
        assert_eq!(
            average_ranks(&[3.0, 1.0, 3.0, 2.0]),
            vec![3.5, 1.0, 3.5, 2.0]
        );
    }

    #[test]
    fn terms_collapse_multi_column_terms_onto_the_linear_predictor() {
        let fit = fitted(false, false);
        let joint = cox_zph(
            &fit,
            ZphTransform::Km,
            true,
            false,
            true,
            Some(&[vec![0, 1]]),
        )
        .unwrap();
        assert_eq!(joint.table.len(), 1);
        assert_eq!(joint.table[0].df, 2);
        assert_eq!(joint.y[0].len(), 1);
        let single = cox_zph(
            &fit,
            ZphTransform::Km,
            true,
            true,
            false,
            Some(&[vec![0, 1]]),
        )
        .unwrap();
        assert_eq!(single.table[0].df, 1);
        assert!(single.global_test.is_none());
        let global = cox_zph(&fit, ZphTransform::Km, true, false, true, None).unwrap();
        assert!((joint.table[0].chisq - global.global_test.unwrap().chisq).abs() < 1e-10);
    }
}
