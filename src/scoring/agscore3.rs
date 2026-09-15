//! Score residuals of a Cox model on (start, stop] data.
//!
//! `agscore3_sorted` ports `agscore3.c` (survival 3.8-12): each
//! observation's residual is updated only when it enters and when it leaves
//! the risk set, from running totals of the cumulative hazard and of
//! `sum xbar(t) dLambda(t)`.  [`agscore3`] reproduces the surrounding
//! `residuals.coxph` code: sort with `order(strata, stop, -status)`, build
//! the start-time order `sort1 = order(strata, start)` on the sorted rows,
//! run the kernel and restore input order.

use crate::core::strata_order::{order_within_strata, validate_intervals};
use crate::error::SurvivalResult;
use crate::internal::typed_inputs::CountingProcessData;
use crate::internal::validation::validate_binary_i32;
use crate::regression::TieMethod;
use crate::scoring::validate_score_inputs;
use ndarray::{Array2, ArrayView2};

/// Score residuals (`n x p`, input order) for (start, stop] data; see
/// [`crate::scoring::coxscore2()`] for the argument conventions.
pub fn agscore3(
    counting: &CountingProcessData,
    covariates: ArrayView2<'_, f64>,
    score: &[f64],
    weights: Option<&[f64]>,
    strata: Option<&[i32]>,
    method: TieMethod,
) -> SurvivalResult<Array2<f64>> {
    let n = counting.len();
    validate_binary_i32(&counting.event, "event")?;
    validate_intervals(&counting.start, &counting.stop)?;
    validate_score_inputs(n, covariates, score, weights, strata)?;
    method.reject_exact("score")?;
    let unit = vec![1.0; n];
    let zero = vec![0; n];
    Ok(agscore3_rows(
        &counting.start,
        &counting.stop,
        &counting.event,
        covariates,
        score,
        weights.unwrap_or(&unit),
        strata.unwrap_or(&zero),
        method,
    ))
}

/// `residuals.coxph`'s score step on validated, unsorted (start, stop]
/// rows: the two sort orders, [`agscore3_sorted`], input order restored.
#[allow(clippy::too_many_arguments)]
pub(crate) fn agscore3_rows(
    start: &[f64],
    stop: &[f64],
    event: &[i32],
    covariates: ArrayView2<'_, f64>,
    score: &[f64],
    weights: &[f64],
    strata: &[i32],
    method: TieMethod,
) -> Array2<f64> {
    let n = start.len();
    let nvar = covariates.ncols();
    let order = order_within_strata(strata, |a, b| {
        stop[a]
            .total_cmp(&stop[b])
            .then_with(|| event[b].cmp(&event[a]))
    });
    let sorted_start: Vec<f64> = order.iter().map(|&i| start[i]).collect();
    let sorted_stop: Vec<f64> = order.iter().map(|&i| stop[i]).collect();
    let sorted_event: Vec<i32> = order.iter().map(|&i| event[i]).collect();
    let sorted_strata: Vec<i32> = order.iter().map(|&i| strata[i]).collect();
    let sorted_score: Vec<f64> = order.iter().map(|&i| score[i]).collect();
    let sorted_weights: Vec<f64> = order.iter().map(|&i| weights[i]).collect();
    let mut sorted_covar = Array2::zeros((n, nvar));
    for (row, &i) in order.iter().enumerate() {
        sorted_covar.row_mut(row).assign(&covariates.row(i));
    }
    let sort1 = order_within_strata(&sorted_strata, |a, b| {
        sorted_start[a].total_cmp(&sorted_start[b])
    });

    let sorted = agscore3_sorted(
        &sorted_start,
        &sorted_stop,
        &sorted_event,
        sorted_covar.view(),
        &sorted_strata,
        &sorted_score,
        &sorted_weights,
        method,
        &sort1,
    );
    let mut resid = Array2::zeros((n, nvar));
    for (row, &i) in order.iter().enumerate() {
        resid.row_mut(i).assign(&sorted.row(row));
    }
    resid
}

/// `agscore3.c` on data sorted by stratum and ascending stop time within
/// stratum; `sort1` orders the same rows by stratum and ascending start
/// time.  `strata` are labels, compared directly as the C code does.
#[allow(clippy::too_many_arguments)]
pub(crate) fn agscore3_sorted(
    tstart: &[f64],
    tstop: &[f64],
    event: &[i32],
    covar: ArrayView2<'_, f64>,
    strata: &[i32],
    score: &[f64],
    weights: &[f64],
    method: TieMethod,
    sort1: &[usize],
) -> Array2<f64> {
    let n = tstart.len();
    let nvar = covar.ncols();
    let mut resid = Array2::zeros((n, nvar));
    if n == 0 {
        return resid;
    }
    let mut a = vec![0.0; nvar];
    let mut a2 = vec![0.0; nvar];
    let mut mean = vec![0.0; nvar];
    let mut mh1 = vec![0.0; nvar];
    let mut mh2 = vec![0.0; nvar];
    let mut mh3 = vec![0.0; nvar];
    let mut xhaz = vec![0.0; nvar];
    let mut cumhaz = 0.0;
    let mut denom = 0.0;
    let mut i1 = n as isize - 1;
    let mut currentstrata = strata[n - 1];

    // `person` walks from the last row down to -1 (the C loop is
    // `for (person = n-1; person >= 0; )`); every row is visited, including
    // row 0.
    let mut person = n as isize - 1;
    while person >= 0 {
        let dtime = tstop[person as usize];
        if strata[person as usize] != currentstrata {
            // First observation of a new stratum: finish off the prior one.
            while i1 >= 0 && sort1[i1 as usize] as isize > person {
                let k = sort1[i1 as usize];
                for j in 0..nvar {
                    resid[[k, j]] -= score[k] * (cumhaz * covar[[k, j]] - xhaz[j]);
                }
                i1 -= 1;
            }
            cumhaz = 0.0;
            denom = 0.0;
            a.fill(0.0);
            xhaz.fill(0.0);
            currentstrata = strata[person as usize];
        } else {
            // Remove those whose start time is at or beyond `dtime`.
            while i1 >= 0 && tstart[sort1[i1 as usize]] >= dtime {
                let k = sort1[i1 as usize];
                if strata[k] != currentstrata {
                    break;
                }
                let risk = score[k] * weights[k];
                denom -= risk;
                for j in 0..nvar {
                    resid[[k, j]] -= score[k] * (cumhaz * covar[[k, j]] - xhaz[j]);
                    a[j] -= risk * covar[[k, j]];
                }
                i1 -= 1;
            }
        }

        // Count up over this time point.
        let mut e_denom = 0.0;
        let mut meanwt = 0.0;
        let mut deaths = 0.0;
        let mut group_size = 0;
        a2.fill(0.0);
        while person >= 0 && tstop[person as usize] == dtime {
            let row = person as usize;
            // Rare: the first observation of the next stratum has exactly
            // the same stop time as the last one of the current stratum.
            if strata[row] != currentstrata {
                break;
            }
            group_size += 1;
            for j in 0..nvar {
                resid[[row, j]] = (covar[[row, j]] * cumhaz - xhaz[j]) * score[row];
            }
            let risk = score[row] * weights[row];
            denom += risk;
            for j in 0..nvar {
                a[j] += risk * covar[[row, j]];
            }
            if event[row] == 1 {
                deaths += 1.0;
                e_denom += risk;
                meanwt += weights[row];
                for j in 0..nvar {
                    a2[j] += risk * covar[[row, j]];
                }
            }
            person -= 1;
        }
        // As in `coxscore2`, the deaths of the tie group are found by
        // status rather than assumed to be sorted first.
        let group_start = (person + 1) as usize;
        let group_end = group_start + group_size;
        let death_rows = || (group_start..group_end).filter(|&k| event[k] == 1);

        if deaths > 0.0 {
            if deaths < 2.0 || !method.is_efron() {
                let hazard = meanwt / denom;
                cumhaz += hazard;
                for j in 0..nvar {
                    mean[j] = a[j] / denom;
                    xhaz[j] += mean[j] * hazard;
                    for k in death_rows() {
                        resid[[k, j]] += covar[[k, j]] - mean[j];
                    }
                }
            } else {
                // Efron: k tied deaths are k pseudo death times, each death
                // present with probability (k - d)/k at the d-th; the deaths
                // also get a look-ahead correction for the hazard they do
                // not deserve.
                mh1.fill(0.0);
                mh2.fill(0.0);
                mh3.fill(0.0);
                meanwt /= deaths;
                for dd in 0..deaths as usize {
                    let downwt = dd as f64 / deaths;
                    let d2 = denom - downwt * e_denom;
                    let hazard = meanwt / d2;
                    cumhaz += hazard;
                    for j in 0..nvar {
                        mean[j] = (a[j] - downwt * a2[j]) / d2;
                        xhaz[j] += mean[j] * hazard;
                        mh1[j] += hazard * downwt;
                        mh2[j] += mean[j] * hazard * downwt;
                        mh3[j] += mean[j] / deaths;
                    }
                }
                for k in death_rows() {
                    for j in 0..nvar {
                        resid[[k, j]] +=
                            (covar[[k, j]] - mh3[j]) + score[k] * (covar[[k, j]] * mh1[j] - mh2[j]);
                    }
                }
            }
        }
    }

    // Finish those in the final stratum.
    while i1 >= 0 {
        let k = sort1[i1 as usize];
        for j in 0..nvar {
            resid[[k, j]] -= score[k] * (covar[[k, j]] * cumhaz - xhaz[j]);
        }
        i1 -= 1;
    }
    resid
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internal::typed_inputs::SurvivalData;
    use crate::scoring::coxscore2;
    use ndarray::array;

    fn counting(start: Vec<f64>, stop: Vec<f64>, event: Vec<i32>) -> CountingProcessData {
        CountingProcessData::try_new(start, stop, event).unwrap()
    }

    #[test]
    fn zero_start_times_reduce_to_coxscore2() {
        let time = vec![5.0, 3.0, 3.0, 8.0, 1.0, 3.0, 8.0, 2.0];
        let status = vec![1, 1, 1, 1, 0, 0, 0, 1];
        let covar = array![
            [0.3, 1.0],
            [-1.2, 0.0],
            [2.0, 1.0],
            [0.7, 0.0],
            [-0.4, 1.0],
            [1.5, 0.0],
            [0.1, 1.0],
            [0.9, 0.0]
        ];
        let score: Vec<f64> = covar
            .rows()
            .into_iter()
            .map(|r| f64::exp(0.3 * r[0]))
            .collect();
        let weights = vec![1.0, 2.0, 1.0, 0.5, 1.0, 1.0, 3.0, 1.0];
        let strata = vec![1, 1, 1, 1, 2, 2, 2, 2];
        for method in [TieMethod::Breslow, TieMethod::Efron] {
            let right = coxscore2(
                &SurvivalData::try_new(time.clone(), status.clone()).unwrap(),
                covar.view(),
                &score,
                Some(&weights),
                Some(&strata),
                method,
            )
            .unwrap();
            let start_stop = agscore3(
                &counting(vec![0.0; 8], time.clone(), status.clone()),
                covar.view(),
                &score,
                Some(&weights),
                Some(&strata),
                method,
            )
            .unwrap();
            for (a, b) in right.iter().zip(start_stop.iter()) {
                assert!((a - b).abs() < 1e-12, "{a} != {b} ({method:?})");
            }
        }
    }

    #[test]
    fn row_zero_is_processed() {
        // A single death in row 0 of the sorted data: its residual is
        // x - xbar over the risk set, which the pre-fix port left at zero.
        let resid = agscore3(
            &counting(vec![0.0, 0.0], vec![1.0, 2.0], vec![1, 0]),
            array![[1.0], [3.0]].view(),
            &[1.0, 1.0],
            None,
            None,
            TieMethod::Breslow,
        )
        .unwrap();
        assert!((resid[[0, 0]] - (1.0 - 2.0) * 0.5).abs() < 1e-12);
        assert!((resid[[1, 0]] - (3.0 - 2.0) * -0.5).abs() < 1e-12);
    }

    #[test]
    fn delayed_entry_excludes_earlier_risk_sets() {
        // Subject 1 enters after the first event, so its residual only sees
        // the second risk set, where it is the sole member besides subject 2.
        let resid = agscore3(
            &counting(vec![0.0, 2.0, 0.0], vec![1.0, 4.0, 4.0], vec![1, 1, 0]),
            array![[1.0], [2.0], [0.0]].view(),
            &[1.0; 3],
            None,
            None,
            TieMethod::Breslow,
        )
        .unwrap();
        // Event 1: risk set {0, 2}, xbar 0.5. Event 4: risk set {1, 2}, xbar 1.
        assert!((resid[[0, 0]] - (1.0 - 0.5) * (1.0 - 0.5)).abs() < 1e-12);
        assert!((resid[[1, 0]] - (2.0 - 1.0) * (1.0 - 0.5)).abs() < 1e-12);
        let expected2 = (0.0 - 0.5) * (-0.5) + (0.0 - 1.0) * (-0.5);
        assert!((resid[[2, 0]] - expected2).abs() < 1e-12);
    }
}
