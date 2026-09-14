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
use crate::residuals::TieMethod;
use crate::scoring::validate_score_inputs;
use ndarray::{Array2, ArrayView2};

/// Score residuals (`n x p`, input order) for right-censored data.
///
/// `covariates` is the `n x p` design matrix, `score` is `exp(eta)` and the
/// residual is unweighted (R multiplies by the case weights afterwards when
/// `weighted = TRUE`).
pub fn coxscore2(
    survival: &SurvivalData,
    covariates: ArrayView2<'_, f64>,
    score: &[f64],
    weights: Option<&[f64]>,
    strata: Option<&[i32]>,
    method: TieMethod,
) -> SurvivalResult<Array2<f64>> {
    let n = survival.len();
    validate_binary_i32(&survival.status, "status")?;
    validate_score_inputs(n, covariates, score, weights, strata)?;
    let nvar = covariates.ncols();
    let time = &survival.time;
    let status = &survival.status;
    let unit = vec![1.0; n];
    let weights = weights.unwrap_or(&unit);
    let zero = vec![0; n];
    let strata = strata.unwrap_or(&zero);

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
    Ok(resid)
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
            if deaths < 2.0 || method == TieMethod::Breslow {
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
                // the stratum.
                meanwt /= deaths;
                for dd in 0..deaths as usize {
                    let downwt = dd as f64 / deaths;
                    let temp = denom - downwt * e_denom;
                    let hazard = meanwt / temp;
                    cumhaz += hazard;
                    for j in 0..nvar {
                        let xbar = (a[j] - downwt * a2[j]) / temp;
                        xhaz[j] += xbar * hazard;
                        for k in death_rows() {
                            let temp2 = covar[[k, j]] - xbar;
                            resid[[k, j]] += temp2 / deaths;
                            resid[[k, j]] += temp2 * score[k] * hazard * downwt;
                        }
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
