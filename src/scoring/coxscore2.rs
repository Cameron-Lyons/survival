//! Score residuals of a right-censored Cox model.
//!
//! `coxscore2_sorted` ports `coxscore2.c` (survival 3.8-12, the O(np)
//! version of April 2023): the residual of observation `i` is
//! `sum_t (x_i - xbar(t)) dM_i(t)`, accumulated with running totals of the
//! cumulative hazard and of `sum xbar(t) dLambda(t)`.  [`coxscore2`] does
//! what `residuals.coxph` does around it: sort with
//! `order(strata, time, -status)`, run the kernel, restore input order.

use crate::core::strata_order::order_within_strata;
use crate::error::SurvivalResult;
use crate::internal::typed_inputs::SurvivalData;
use crate::internal::validation::validate_binary_i32;
use crate::regression::TieMethod;
use crate::scoring::validate_score_inputs;
use ndarray::{Array2, ArrayView2};

/// Score residuals (`n x p`, input order) for right-censored data.
///
/// `covariates` is the `n x p` design matrix, `score` is `exp(eta)` and the
/// residual is unweighted (R multiplies by the case weights afterwards when
/// `weighted = TRUE`).  As in R, an exact fit has no score residuals.
pub fn coxscore2(
    survival: &SurvivalData,
    covariates: ArrayView2<'_, f64>,
    score: &[f64],
    weights: Option<&[f64]>,
    strata: Option<&[i32]>,
    method: TieMethod,
) -> SurvivalResult<Array2<f64>> {
    let n = survival.len();
    SurvivalData::validate_parts(&survival.time, &survival.status)?;
    validate_binary_i32(&survival.status, "status")?;
    validate_score_inputs(n, covariates, score, weights, strata)?;
    method.reject_exact("score")?;
    let unit = vec![1.0; n];
    let zero = vec![0; n];
    Ok(coxscore2_rows(
        &survival.time,
        &survival.status,
        covariates,
        score,
        weights.unwrap_or(&unit),
        strata.unwrap_or(&zero),
        method,
    ))
}

/// `residuals.coxph`'s score step on validated, unsorted rows: sort with
/// `order(strata, time, -status)`, run [`coxscore2_sorted`] and restore the
/// input order.
pub(crate) fn coxscore2_rows(
    time: &[f64],
    status: &[i32],
    covariates: ArrayView2<'_, f64>,
    score: &[f64],
    weights: &[f64],
    strata: &[i32],
    method: TieMethod,
) -> Array2<f64> {
    let n = time.len();
    let nvar = covariates.ncols();
    let order = order_within_strata(strata, |a, b| {
        time[a]
            .total_cmp(&time[b])
            .then_with(|| status[b].cmp(&status[a]))
    });
    let sorted_time: Vec<f64> = order.iter().map(|&i| time[i]).collect();
    let sorted_status: Vec<i32> = order.iter().map(|&i| status[i]).collect();
    let sorted_strata: Vec<i32> = order.iter().map(|&i| strata[i]).collect();
    let sorted_score: Vec<f64> = order.iter().map(|&i| score[i]).collect();
    let sorted_weights: Vec<f64> = order.iter().map(|&i| weights[i]).collect();
    let mut sorted_covar = Array2::zeros((n, nvar));
    for (row, &i) in order.iter().enumerate() {
        sorted_covar.row_mut(row).assign(&covariates.row(i));
    }

    let sorted = coxscore2_sorted(
        &sorted_time,
        &sorted_status,
        sorted_covar.view(),
        &sorted_strata,
        &sorted_score,
        &sorted_weights,
        method,
    );
    let mut resid = Array2::zeros((n, nvar));
    for (row, &i) in order.iter().enumerate() {
        resid.row_mut(i).assign(&sorted.row(row));
    }
    resid
}

/// `coxscore2.c` on data sorted by stratum and ascending time within
/// stratum.  `strata` are labels (the C code compares them directly).
pub(crate) fn coxscore2_sorted(
    time: &[f64],
    status: &[i32],
    covar: ArrayView2<'_, f64>,
    strata: &[i32],
    score: &[f64],
    weights: &[f64],
    method: TieMethod,
) -> Array2<f64> {
    let n = time.len();
    let nvar = covar.ncols();
    let mut resid = Array2::zeros((n, nvar));
    if n == 0 {
        return resid;
    }
    let mut a = vec![0.0; nvar];
    let mut a2 = vec![0.0; nvar];
    let mut xhaz = vec![0.0; nvar];
    let mut mh2 = vec![0.0; nvar];
    let mut mh3 = vec![0.0; nvar];
    let mut denom = 0.0;
    let mut cumhaz = 0.0;
    let mut stratastart = n - 1;
    let mut currentstrata = strata[n - 1];

    // `i` walks from the last row down to -1, in spurts of tied times.
    let mut i = n as isize - 1;
    while i >= 0 {
        let newtime = time[i as usize];
        let mut deaths = 0.0;
        let mut e_denom = 0.0;
        let mut meanwt = 0.0;
        let mut group_size = 0;
        a2.fill(0.0);
        while i >= 0 {
            let row = i as usize;
            if time[row] != newtime || strata[row] != currentstrata {
                break;
            }
            group_size += 1;
            let risk = score[row] * weights[row];
            denom += risk;
            for j in 0..nvar {
                // Future accumulated risk that new entries must not get.
                resid[[row, j]] = score[row] * (covar[[row, j]] * cumhaz - xhaz[j]);
                a[j] += risk * covar[[row, j]];
            }
            if status[row] == 1 {
                deaths += 1.0;
                e_denom += risk;
                meanwt += weights[row];
                for j in 0..nvar {
                    a2[j] += risk * covar[[row, j]];
                }
            }
            i -= 1;
        }
        // The C code relies on the deaths being sorted first within a tied
        // time (rows `group_start..group_start + deaths`); scanning the
        // status of the whole tie group instead gives the same result for
        // any order of the ties.
        let group_start = (i + 1) as usize;
        let group_end = group_start + group_size;
        let death_rows = || (group_start..group_end).filter(|&k| status[k] == 1);

        if deaths > 0.0 {
            if deaths < 2.0 || !method.is_efron() {
                let hazard = meanwt / denom;
                cumhaz += hazard;
                for j in 0..nvar {
                    let xbar = a[j] / denom;
                    xhaz[j] += xbar * hazard;
                    for k in death_rows() {
                        resid[[k, j]] += covar[[k, j]] - xbar;
                    }
                }
            } else {
                // Efron: the deaths are charged ahead for the part of the
                // hazard increment they should not receive at the end of
                // the stratum.  coxscore2.c adds, for each death k,
                // sum_dd (x_k - xbar_dd) (1/deaths + score_k hazard_dd
                // downwt_dd); the sums over dd are accumulated first
                // (mh1, mh2, mh3, as agscore3.c does) so the deaths are
                // visited once.
                meanwt /= deaths;
                let mut mh1 = 0.0;
                mh2.fill(0.0);
                mh3.fill(0.0);
                for dd in 0..deaths as usize {
                    let downwt = dd as f64 / deaths;
                    let temp = denom - downwt * e_denom;
                    let hazard = meanwt / temp;
                    cumhaz += hazard;
                    mh1 += hazard * downwt;
                    for j in 0..nvar {
                        let xbar = (a[j] - downwt * a2[j]) / temp;
                        xhaz[j] += xbar * hazard;
                        mh2[j] += xbar * hazard * downwt;
                        mh3[j] += xbar / deaths;
                    }
                }
                for k in death_rows() {
                    for j in 0..nvar {
                        resid[[k, j]] +=
                            (covar[[k, j]] - mh3[j]) + score[k] * (covar[[k, j]] * mh1 - mh2[j]);
                    }
                }
            }
        }

        if i < 0 || strata[i as usize] != currentstrata {
            // End of a stratum: final term for every observation in it.
            for k in group_start..=stratastart {
                for j in 0..nvar {
                    resid[[k, j]] += score[k] * (xhaz[j] - covar[[k, j]] * cumhaz);
                }
            }
            denom = 0.0;
            cumhaz = 0.0;
            a.fill(0.0);
            xhaz.fill(0.0);
            if i >= 0 {
                stratastart = i as usize;
                currentstrata = strata[i as usize];
            }
        }
    }
    resid
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn survival(time: Vec<f64>, status: Vec<i32>) -> SurvivalData {
        SurvivalData::try_new(time, status).unwrap()
    }

    /// Direct O(n^2) evaluation of `sum_t (x_i - xbar(t)) dM_i(t)` for the
    /// Breslow approximation, one covariate, no ties.
    fn naive_breslow(time: &[f64], status: &[i32], x: &[f64], score: &[f64]) -> Vec<f64> {
        let n = time.len();
        let mut resid = vec![0.0; n];
        for t in 0..n {
            if status[t] != 1 {
                continue;
            }
            let at_risk: Vec<usize> = (0..n).filter(|&k| time[k] >= time[t]).collect();
            let denom: f64 = at_risk.iter().map(|&k| score[k]).sum();
            let xbar: f64 = at_risk.iter().map(|&k| score[k] * x[k]).sum::<f64>() / denom;
            let hazard = 1.0 / denom;
            for &k in &at_risk {
                let dn = if k == t { 1.0 } else { 0.0 };
                resid[k] += (x[k] - xbar) * (dn - score[k] * hazard);
            }
        }
        resid
    }

    #[test]
    fn matches_direct_definition_without_ties() {
        let time = vec![4.0, 1.0, 6.0, 3.0, 5.0, 2.0];
        let status = vec![1, 1, 0, 1, 1, 0];
        let x = vec![0.3, -1.2, 2.0, 0.7, -0.4, 1.5];
        let score: Vec<f64> = x.iter().map(|v| f64::exp(0.4 * v)).collect();
        let covar = Array2::from_shape_vec((6, 1), x.clone()).unwrap();
        let resid = coxscore2(
            &survival(time.clone(), status.clone()),
            covar.view(),
            &score,
            None,
            None,
            TieMethod::Efron,
        )
        .unwrap();
        let expected = naive_breslow(&time, &status, &x, &score);
        for (row, expected) in expected.iter().enumerate() {
            assert!(
                (resid[[row, 0]] - expected).abs() < 1e-12,
                "row {row}: {} != {expected}",
                resid[[row, 0]]
            );
        }
    }

    #[test]
    fn score_residuals_sum_to_the_score_vector() {
        // At the true maximum the score residuals sum to the score vector,
        // which is zero; away from it they sum to U(beta).  With beta = 0
        // (unit scores) U = sum over deaths of (x - xbar).
        let time = vec![1.0, 1.0, 2.0, 3.0, 3.0, 4.0];
        let status = vec![1, 1, 0, 1, 1, 1];
        let covar = array![
            [1.0, 0.5],
            [0.0, 2.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [1.0, 3.0],
            [0.0, 1.5]
        ];
        let resid = coxscore2(
            &survival(time.clone(), status.clone()),
            covar.view(),
            &[1.0; 6],
            None,
            None,
            TieMethod::Breslow,
        )
        .unwrap();
        for j in 0..2 {
            let mut u = 0.0;
            for t in 0..6 {
                if status[t] != 1 {
                    continue;
                }
                let at_risk: Vec<usize> = (0..6).filter(|&k| time[k] >= time[t]).collect();
                let xbar =
                    at_risk.iter().map(|&k| covar[[k, j]]).sum::<f64>() / at_risk.len() as f64;
                u += covar[[t, j]] - xbar;
            }
            let total: f64 = resid.column(j).sum();
            assert!((total - u).abs() < 1e-12, "column {j}: {total} != {u}");
        }
    }

    #[test]
    fn efron_ties_match_r_with_weights() {
        // R: coxph(Surv(time, status) ~ x1 + x2, weights = w, ties = "efron",
        //    init = c(0.3, -0.2), iter.max = 0); residuals(fit, type = "score").
        // Tie groups of 3 and 2 deaths, each with a censored row.
        let time = vec![2.0, 2.0, 2.0, 2.0, 3.0, 5.0, 5.0, 5.0, 7.0, 8.0];
        let status = vec![1, 1, 0, 1, 1, 1, 0, 1, 1, 0];
        let x1: [f64; 10] = [0.5, -1.2, 0.3, 1.1, -0.4, 0.8, 1.5, -0.7, 0.2, -1.0];
        let x2 = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0];
        let weights = [1.0, 2.0, 1.5, 0.5, 1.0, 1.0, 2.0, 1.0, 0.8, 1.2];
        let covar = Array2::from_shape_fn((10, 2), |(i, j)| if j == 0 { x1[i] } else { x2[i] });
        let score: Vec<f64> = (0..10).map(|i| (0.3 * x1[i] - 0.2 * x2[i]).exp()).collect();
        let resid = coxscore2(
            &survival(time, status),
            covar.view(),
            &score,
            Some(&weights),
            None,
            TieMethod::Efron,
        )
        .unwrap();
        let expected = [
            [0.12880443708929956, 0.2991707894699495],
            [-1.297937692648438, -0.5245704049725539],
            [0.011135712838710593, -0.11674529985902629],
            [0.5697089245772788, 0.2832545990082292],
            [-0.5456835256343971, -0.3112239236232024],
            [-0.16583014045179723, -0.11155647160376048],
            [-1.0949377353459728, -0.3712761645748478],
            [-0.6387207314797603, -0.3339168631913795],
            [0.5338335395247131, -0.25137507956271266],
            [0.9725493803037063, -0.1753784419751265],
        ];
        for (row, expected) in expected.iter().enumerate() {
            for (j, &value) in expected.iter().enumerate() {
                assert!(
                    (resid[[row, j]] - value).abs() < 1e-13,
                    "[{row}, {j}]: {} != {value}",
                    resid[[row, j]]
                );
            }
        }
    }

    #[test]
    fn strata_and_order_are_respected() {
        let time = vec![3.0, 1.0, 2.0, 2.0];
        let status = vec![1, 1, 1, 0];
        let covar = array![[1.0], [2.0], [3.0], [4.0]];
        let score = vec![1.0, 2.0, 0.5, 1.5];
        let pooled_a = coxscore2(
            &survival(time[..2].to_vec(), status[..2].to_vec()),
            covar.slice(ndarray::s![..2, ..]),
            &score[..2],
            None,
            None,
            TieMethod::Efron,
        )
        .unwrap();
        let pooled_b = coxscore2(
            &survival(time[2..].to_vec(), status[2..].to_vec()),
            covar.slice(ndarray::s![2.., ..]),
            &score[2..],
            None,
            None,
            TieMethod::Efron,
        )
        .unwrap();
        let stratified = coxscore2(
            &survival(time.clone(), status.clone()),
            covar.view(),
            &score,
            None,
            Some(&[2, 2, 1, 1]),
            TieMethod::Efron,
        )
        .unwrap();
        assert!((stratified[[0, 0]] - pooled_a[[0, 0]]).abs() < 1e-12);
        assert!((stratified[[1, 0]] - pooled_a[[1, 0]]).abs() < 1e-12);
        assert!((stratified[[2, 0]] - pooled_b[[0, 0]]).abs() < 1e-12);
        assert!((stratified[[3, 0]] - pooled_b[[1, 0]]).abs() < 1e-12);
    }

    #[test]
    fn rejects_mismatched_inputs() {
        let covar = array![[1.0], [2.0]];
        assert!(
            coxscore2(
                &survival(vec![1.0, 2.0, 3.0], vec![1, 0, 1]),
                covar.view(),
                &[1.0; 3],
                None,
                None,
                TieMethod::Breslow,
            )
            .is_err()
        );
        assert!(
            coxscore2(
                &survival(vec![1.0, 2.0], vec![1, 0]),
                covar.view(),
                &[1.0, -1.0],
                None,
                None,
                TieMethod::Breslow,
            )
            .is_err()
        );
    }
}
