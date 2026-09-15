//! Royston and Sauerbrei's D and the related explained-variation measures
//! for a Cox model.
//!
//! Port of R survival `R/royston.R` for the `newdata`-free call: the linear
//! predictor of the fit is replaced by normal scores and a Cox model is
//! refitted on that predictor (`coxph(y ~ qhat)`, Efron ties), from which
//! D, its standard error and R²_D follow; Kent-O'Quigley, Nagelkerke and
//! Gönen-Heller measures are computed alongside.

use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::dist::qnorm;
use crate::internal::validation::{validate_binary_i32, validate_finite, validate_length};
use crate::regression::coxph::{CoxPHFit, CoxphData, CoxphOptions};
use ndarray::Array2;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::f64::consts::PI;

const PARALLEL_THRESHOLD: usize = 2_000;

/// Inputs of [`royston`]: the fitted model's linear predictors and response.
#[derive(Debug, Clone)]
pub struct RoystonInput<'a> {
    /// `predict(fit)`: the centred linear predictor.
    pub eta: &'a [f64],
    pub time: &'a [f64],
    pub status: &'a [i32],
    /// Start times for (start, stop] responses.
    pub entry_times: Option<&'a [f64]>,
    /// `fit$loglik`: null and final partial log-likelihood.
    pub loglik: [f64; 2],
    /// `fit$nevent` and the number of coefficients, for `adjust`.
    pub n_event: usize,
    pub n_coef: usize,
    /// Average the normal scores over tied `eta` values (R `ties`).
    pub ties: bool,
    /// Apply the overfitting adjustment (R `adjust`).
    pub adjust: bool,
}

/// R's named result vector.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object, get_all)]
pub struct RoystonResult {
    pub d: f64,
    pub se_d: f64,
    pub r_d: f64,
    pub r_ko: f64,
    pub r_n: f64,
    pub c_gh: f64,
}

/// Blom normal scores of `eta`, averaged over ties when `ties` is set
/// (R's `qhat`).
fn normal_scores(eta: &[f64], ties: bool) -> Vec<f64> {
    let n = eta.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| eta[a].total_cmp(&eta[b]).then_with(|| a.cmp(&b)));
    let blom = |rank: f64| qnorm((rank - 3.0 / 8.0) / (n as f64 + 0.25), true, false);
    let mut scores = vec![0.0; n];
    let mut start = 0;
    while start < n {
        let mut end = start + 1;
        while end < n && eta[order[end]] == eta[order[start]] {
            end += 1;
        }
        if ties {
            // mean of the scores at ranks start+1..=end (ties.method = "first")
            let mean =
                (start..end).map(|k| blom((k + 1) as f64)).sum::<f64>() / (end - start) as f64;
            for &idx in &order[start..end] {
                scores[idx] = mean;
            }
        } else {
            // rank(eta) with the default average rank for ties
            let average_rank = ((start + 1) + end) as f64 / 2.0;
            for &idx in &order[start..end] {
                scores[idx] = blom(average_rank);
            }
        }
        start = end;
    }
    scores
}

/// Gönen and Heller's concordance `2/(n(n-1)) * sum_{i<j} 1/(1+exp(eta_i - eta_j))`
/// over the sorted predictor.
fn gonen_heller(eta: &[f64]) -> f64 {
    let n = eta.len();
    let mut sorted = eta.to_vec();
    sorted.sort_by(f64::total_cmp);
    let row_sum = |i: usize| -> f64 {
        sorted[i + 1..]
            .iter()
            .map(|&later| 1.0 / (1.0 + (sorted[i] - later).exp()))
            .sum()
    };
    let total: f64 = if n >= PARALLEL_THRESHOLD {
        (0..n - 1).into_par_iter().map(row_sum).sum()
    } else {
        (0..n - 1).map(row_sum).sum()
    };
    total * 2.0 / (n as f64 * (n - 1) as f64)
}

fn validate(input: &RoystonInput<'_>) -> SurvivalResult<()> {
    let n = input.eta.len();
    if n < 2 {
        return Err(SurvivalError::invalid_input(
            "royston needs at least two observations",
        ));
    }
    validate_length(n, input.time.len(), "time")?;
    validate_length(n, input.status.len(), "status")?;
    validate_finite(input.eta, "eta")?;
    validate_finite(input.time, "time")?;
    validate_binary_i32(input.status, "status")?;
    if let Some(entry) = input.entry_times {
        validate_length(n, entry.len(), "entry_times")?;
        validate_finite(entry, "entry_times")?;
    }
    if input.loglik.iter().any(|v| !v.is_finite()) {
        return Err(SurvivalError::invalid_input("loglik must be finite"));
    }
    if input.adjust && input.n_event <= input.n_coef {
        return Err(SurvivalError::invalid_input(
            "adjust requires more events than coefficients",
        ));
    }
    Ok(())
}

/// Royston's D and the R²_D, Kent-O'Quigley, Nagelkerke and Gönen-Heller
/// measures of a Cox fit.
pub fn royston(input: &RoystonInput<'_>) -> SurvivalResult<RoystonResult> {
    validate(input)?;
    let eta = input.eta;
    let n = eta.len() as f64;
    let mean = eta.iter().sum::<f64>() / n;
    let var_eta = eta.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let r_ko = var_eta / (PI * PI / 6.0 + var_eta);

    let qhat = normal_scores(eta, input.ties);
    let n_obs = qhat.len();
    let data = CoxphData::try_new(
        input.time.to_vec(),
        input.entry_times.map(<[f64]>::to_vec),
        input.status.to_vec(),
        Array2::from_shape_vec((n_obs, 1), qhat)
            .map_err(|err| SurvivalError::computation(err.to_string()))?,
        None,
        None,
        None,
    )?;
    let refit = CoxPHFit::fit(data, CoxphOptions::default())
        .map_err(|err| SurvivalError::computation(format!("coxph(y ~ qhat): {err}")))?;
    let beta = refit.coefficients[0];
    let var = refit.var[[0, 0]];
    let mut d = beta * (8.0 / PI).sqrt();
    let mut se_d = (var * 8.0 / PI).sqrt();
    let mut r_d = beta * beta / (PI * PI / 6.0 + beta * beta);
    let r_i = beta * beta / (1.0 + beta * beta);
    if input.adjust {
        let r = input.n_event as f64 / (input.n_event - input.n_coef) as f64;
        let temp = (1.0 + beta * beta - r) / r;
        d = beta.signum() * temp.signum() * (temp.abs() * 8.0 / PI).sqrt();
        se_d = se_d * beta.abs() / (r * temp.abs().sqrt());
        r_d = 1.0 - r * (1.0 - r_i);
    }
    let c_gh = gonen_heller(eta);
    let logtest = -2.0 * (input.loglik[0] - input.loglik[1]);
    let r_n = (1.0 - (-logtest / n).exp()) / (1.0 - (2.0 * input.loglik[0] / n).exp());
    Ok(RoystonResult {
        d,
        se_d,
        r_d,
        r_ko,
        r_n,
        c_gh,
    })
}

/// Python entry point: `royston(eta, time, status, loglik, n_event,
/// n_coef, entry_times=None, ties=True, adjust=False)`.
#[pyfunction(name = "royston")]
#[pyo3(signature = (eta, time, status, loglik, n_event, n_coef, entry_times=None, ties=true, adjust=false))]
#[allow(clippy::too_many_arguments)]
pub fn royston_py(
    eta: Vec<f64>,
    time: Vec<f64>,
    status: Vec<i32>,
    loglik: [f64; 2],
    n_event: usize,
    n_coef: usize,
    entry_times: Option<Vec<f64>>,
    ties: bool,
    adjust: bool,
) -> PyResult<RoystonResult> {
    Ok(royston(&RoystonInput {
        eta: &eta,
        time: &time,
        status: &status,
        entry_times: entry_times.as_deref(),
        loglik,
        n_event,
        n_coef,
        ties,
        adjust,
    })?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_scores_average_over_ties_when_requested() {
        let eta = [0.3, 0.1, 0.3, 0.7];
        let tied = normal_scores(&eta, true);
        assert_eq!(tied[0], tied[2]);
        let z2 = qnorm((2.0 - 0.375) / 4.25, true, false);
        let z3 = qnorm((3.0 - 0.375) / 4.25, true, false);
        assert!((tied[0] - (z2 + z3) / 2.0).abs() < 1e-12);
        let untied = normal_scores(&eta, false);
        assert!((untied[0] - qnorm((2.5 - 0.375) / 4.25, true, false)).abs() < 1e-12);
        assert!(tied[1] < tied[0] && tied[0] < tied[3]);
    }

    #[test]
    fn gonen_heller_is_half_for_a_constant_predictor() {
        assert!((gonen_heller(&[0.2; 5]) - 0.5).abs() < 1e-12);
        assert!(gonen_heller(&[-1.0, 0.0, 1.0, 2.0]) > 0.5);
    }

    #[test]
    fn a_stronger_predictor_gives_a_larger_d() {
        let time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let status = [1, 1, 1, 1, 1, 1, 1, 0];
        let strong = [1.0, 0.9, 0.8, 0.7, -0.7, -0.8, -0.9, -1.0];
        let weak = [0.1, -0.2, 0.3, 0.0, 0.2, -0.1, 0.1, -0.3];
        let run = |eta: &[f64]| {
            royston(&RoystonInput {
                eta,
                time: &time,
                status: &status,
                entry_times: None,
                loglik: [-10.0, -8.0],
                n_event: 7,
                n_coef: 1,
                ties: true,
                adjust: false,
            })
            .unwrap()
        };
        let strong = run(&strong);
        let weak = run(&weak);
        assert!(strong.d > weak.d.abs());
        assert!(strong.r_d > weak.r_d);
        assert!(strong.se_d > 0.0);
        assert!((0.0..=1.0).contains(&strong.c_gh));
    }

    #[test]
    fn inputs_are_validated() {
        let bad = royston(&RoystonInput {
            eta: &[0.1, f64::NAN],
            time: &[1.0, 2.0],
            status: &[1, 1],
            entry_times: None,
            loglik: [-1.0, -1.0],
            n_event: 2,
            n_coef: 1,
            ties: true,
            adjust: false,
        });
        assert!(bad.is_err());
        let adjust = royston(&RoystonInput {
            eta: &[0.1, 0.2],
            time: &[1.0, 2.0],
            status: &[1, 1],
            entry_times: None,
            loglik: [-1.0, -1.0],
            n_event: 1,
            n_coef: 1,
            ties: true,
            adjust: true,
        });
        assert!(adjust.is_err());
    }
}
