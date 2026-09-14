//! Schoenfeld residuals of a Cox model.
//!
//! `coxscho` ports `coxscho.c` (survival 3.8-12): for each death the
//! residual is `x - xbar(t)`, the covariate minus its risk-weighted mean
//! over the risk set (Efron: averaged over the tied-death pseudo risk sets).
//! [`schoenfeld_residuals`] adds the preparation `residuals.coxph` does:
//! sorting with `order(strata, time, -status)` (deaths first within tied
//! times, which the kernel requires) and the R convention that the risk
//! score passed down is `exp(eta) * weight`.

use crate::core::strata_order::{
    SurvResponse, last_of_run, order_within_strata, validate_intervals,
};
use crate::error::SurvivalResult;
use crate::internal::validation::validate_binary_i32;
use crate::residuals::TieMethod;
use crate::scoring::validate_score_inputs;
use ndarray::{Array2, ArrayView2};
use pyo3::prelude::*;

/// Schoenfeld residuals, one row per event in order of (stratum, time).
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct CoxschoResiduals {
    /// Event time of each row.
    #[pyo3(get)]
    pub time: Vec<f64>,
    /// Row index into the input data of each event.
    #[pyo3(get)]
    pub index: Vec<usize>,
    /// Stratum label of each event.
    #[pyo3(get)]
    pub strata: Vec<i32>,
    /// `n_events x p` residual matrix `x - xbar(t)` (unweighted).
    #[pyo3(get)]
    pub residuals: Vec<Vec<f64>>,
}

/// Schoenfeld residuals for right-censored or (start, stop] data.
/// `covariates` is `n x p`, `score` is `exp(eta)`, and the weights, when
/// given, enter the risk-set means as `exp(eta) * weight`.
pub fn schoenfeld_residuals(
    response: SurvResponse<'_>,
    covariates: ArrayView2<'_, f64>,
    score: &[f64],
    weights: Option<&[f64]>,
    strata: Option<&[i32]>,
    method: TieMethod,
) -> SurvivalResult<CoxschoResiduals> {
    let start = response.start();
    let stop = response.stop();
    let event = response.status();
    let n = stop.len();
    if let Some(start) = start {
        validate_intervals(start, stop)?;
    }
    validate_binary_i32(event, "event")?;
    validate_score_inputs(n, covariates, score, weights, strata)?;
    let nvar = covariates.ncols();
    let zero = vec![0; n];
    let strata = strata.unwrap_or(&zero);

    let order = order_within_strata(strata, |a, b| {
        stop[a]
            .total_cmp(&stop[b])
            .then_with(|| event[b].cmp(&event[a]))
    });
    // Right-censored data get a start time below every stop time, the
    // `cbind(-1, y)` / `2 * mintime - 1` trick of `residuals.coxph`.
    let sorted_start: Vec<f64> = match start {
        Some(start) => order.iter().map(|&i| start[i]).collect(),
        None => {
            let min = stop.iter().copied().fold(f64::INFINITY, f64::min);
            let before = if min < 0.0 { 2.0 * min - 1.0 } else { -1.0 };
            vec![before; n]
        }
    };
    let sorted_stop: Vec<f64> = order.iter().map(|&i| stop[i]).collect();
    let sorted_event: Vec<i32> = order.iter().map(|&i| event[i]).collect();
    let sorted_strata: Vec<i32> = order.iter().map(|&i| strata[i]).collect();
    let sorted_score: Vec<f64> = order
        .iter()
        .map(|&i| score[i] * weights.map_or(1.0, |w| w[i]))
        .collect();
    let mut sorted_covar = Array2::zeros((n, nvar));
    for (row, &i) in order.iter().enumerate() {
        sorted_covar.row_mut(row).assign(&covariates.row(i));
    }

    coxscho(
        &sorted_start,
        &sorted_stop,
        &sorted_event,
        &mut sorted_covar,
        &sorted_score,
        &sorted_strata,
        method,
    );
    let deaths: Vec<usize> = (0..n).filter(|&row| sorted_event[row] == 1).collect();
    Ok(CoxschoResiduals {
        time: deaths.iter().map(|&row| sorted_stop[row]).collect(),
        index: deaths.iter().map(|&row| order[row]).collect(),
        strata: deaths.iter().map(|&row| sorted_strata[row]).collect(),
        residuals: deaths
            .iter()
            .map(|&row| sorted_covar.row(row).to_vec())
            .collect(),
    })
}

/// `coxscho.c`: replaces the covariate row of every death by its residual.
/// Data must be sorted by stratum and ascending stop time with deaths first
/// within tied times; `score` already includes the case weight.
pub(crate) fn coxscho(
    start: &[f64],
    stop: &[f64],
    event: &[i32],
    covar: &mut Array2<f64>,
    score: &[f64],
    strata: &[i32],
    method: TieMethod,
) {
    let nused = stop.len();
    let nvar = covar.ncols();
    let last = last_of_run(strata);
    let efron = method.efron_flag();
    let mut a = vec![0.0; nvar];
    let mut a2 = vec![0.0; nvar];
    let mut mean = vec![0.0; nvar];

    let mut person = 0;
    while person < nused {
        if event[person] == 0 {
            person += 1;
            continue;
        }
        // Means over the risk set (a) and over the deaths (a2).
        let mut denom = 0.0;
        let mut efron_wt = 0.0;
        a.fill(0.0);
        a2.fill(0.0);
        let time = stop[person];
        let mut deaths = 0.0;
        for k in person..nused {
            if start[k] < time {
                let weight = score[k];
                denom += weight;
                for i in 0..nvar {
                    a[i] += weight * covar[[k, i]];
                }
                if stop[k] == time && event[k] == 1 {
                    deaths += 1.0;
                    efron_wt += weight;
                    for i in 0..nvar {
                        a2[i] += weight * covar[[k, i]];
                    }
                }
            }
            if last[k] {
                break;
            }
        }
        mean.fill(0.0);
        for k in 0..deaths as usize {
            let temp = efron * k as f64 / deaths;
            for i in 0..nvar {
                mean[i] += (a[i] - temp * a2[i]) / (deaths * (denom - temp * efron_wt));
            }
        }
        // The residuals for this time point.
        let mut k = person;
        while k < nused && stop[k] == time {
            if event[k] == 1 {
                for i in 0..nvar {
                    covar[[k, i]] -= mean[i];
                }
            }
            person += 1;
            if last[k] {
                break;
            }
            k += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internal::typed_inputs::{CountingProcessData, SurvivalData};
    use ndarray::array;

    #[test]
    fn residual_is_covariate_minus_risk_weighted_mean() {
        let stop = [3.0, 1.0, 2.0, 2.0];
        let event = [1, 1, 0, 1];
        let covar = array![[1.0], [2.0], [3.0], [4.0]];
        let score = [1.0, 2.0, 0.5, 1.5];
        let data = SurvivalData::try_new(stop.to_vec(), event.to_vec()).unwrap();
        let out = schoenfeld_residuals(
            SurvResponse::Right(&data),
            covar.view(),
            &score,
            None,
            None,
            TieMethod::Breslow,
        )
        .unwrap();
        assert_eq!(out.time, vec![1.0, 2.0, 3.0]);
        assert_eq!(out.index, vec![1, 3, 0]);
        // t = 1: everyone at risk.
        let xbar1 = (1.0 + 4.0 + 1.5 + 6.0) / 5.0;
        // t = 2: rows with stop >= 2.
        let xbar2 = (1.0 + 1.5 + 6.0) / 3.0;
        let expected = [2.0 - xbar1, 4.0 - xbar2, 1.0 - 1.0];
        for (row, expected) in expected.iter().enumerate() {
            assert!((out.residuals[row][0] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn efron_averages_the_tied_pseudo_risk_sets() {
        let stop = [1.0, 1.0, 2.0];
        let event = [1, 1, 0];
        let covar = array![[0.0], [1.0], [2.0]];
        let score = [1.0, 1.0, 1.0];
        let data = SurvivalData::try_new(stop.to_vec(), event.to_vec()).unwrap();
        let out = schoenfeld_residuals(
            SurvResponse::Right(&data),
            covar.view(),
            &score,
            None,
            None,
            TieMethod::Efron,
        )
        .unwrap();
        // First pseudo set: all three (mean 1); second: deaths down-weighted
        // by 1/2 -> (0.5*0 + 0.5*1 + 2) / 2 = 1.25; average 1.125.
        let mean = (1.0 + 1.25) / 2.0;
        assert!((out.residuals[0][0] - (0.0 - mean)).abs() < 1e-12);
        assert!((out.residuals[1][0] - (1.0 - mean)).abs() < 1e-12);
    }

    #[test]
    fn strata_and_weights_enter_the_means() {
        let stop = [1.0, 2.0, 1.0, 2.0];
        let event = [1, 0, 1, 0];
        let covar = array![[1.0], [3.0], [10.0], [20.0]];
        let score = [1.0; 4];
        let weights = [1.0, 3.0, 1.0, 1.0];
        let data = SurvivalData::try_new(stop.to_vec(), event.to_vec()).unwrap();
        let out = schoenfeld_residuals(
            SurvResponse::Right(&data),
            covar.view(),
            &score,
            Some(&weights),
            Some(&[1, 1, 2, 2]),
            TieMethod::Breslow,
        )
        .unwrap();
        assert_eq!(out.strata, vec![1, 2]);
        assert!((out.residuals[0][0] - (1.0 - (1.0 + 9.0) / 4.0)).abs() < 1e-12);
        assert!((out.residuals[1][0] - (10.0 - 15.0)).abs() < 1e-12);
    }

    #[test]
    fn counting_process_start_times_limit_the_risk_set() {
        let start = [0.0, 1.5, 0.0];
        let stop = [1.0, 3.0, 3.0];
        let event = [1, 1, 0];
        let covar = array![[0.0], [2.0], [4.0]];
        let data =
            CountingProcessData::try_new(start.to_vec(), stop.to_vec(), event.to_vec()).unwrap();
        let out = schoenfeld_residuals(
            SurvResponse::Counting(&data),
            covar.view(),
            &[1.0; 3],
            None,
            None,
            TieMethod::Breslow,
        )
        .unwrap();
        // Subject 1 enters at 1.5, so the risk set at time 1 is {0, 2}.
        assert!((out.residuals[0][0] - (0.0 - 2.0)).abs() < 1e-12);
        assert!((out.residuals[1][0] - (2.0 - 3.0)).abs() < 1e-12);
    }
}
