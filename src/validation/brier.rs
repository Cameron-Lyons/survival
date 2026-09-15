//! Brier score of a Cox model with inverse-probability-of-censoring weights.
//!
//! Port of R survival `R/brier.R` for right-censored data and for (start,
//! stop] data whose subjects all enter at the same time (R stops with
//! "delayed entry is not yet implemented" otherwise).  The model's predicted
//! event probabilities at the evaluation times come from the caller (R
//! evaluates `survfit(fit, newdata)` there); the null model, the censoring
//! distribution and the weighting are computed here.  The Python entry
//! point lives in `pybridge::brier`.

use crate::data_prep::aeq_surv;
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::{validate_binary_i32, validate_finite, validate_length};
use crate::surv_analysis::{HazardType, SurvType, SurvfitKMData, SurvfitKMOptions, survfitkm};
use pyo3::prelude::*;
use rayon::prelude::*;

const PARALLEL_WORK_THRESHOLD: usize = 8_192;

/// Inputs of [`brier`].
#[derive(Debug, Clone)]
pub struct BrierInput<'a> {
    /// Entry times of (start, stop] rows, which only the null model's
    /// risk sets see (R's `survfit(Y ~ 1)`); `None` for right-censored data.
    pub start: Option<&'a [f64]>,
    pub time: &'a [f64],
    pub status: &'a [i32],
    /// Case weights (R `weights`); `None` for unit weights.
    pub weights: Option<&'a [f64]>,
    /// Evaluation times (R `times`; the default, the event times of the
    /// null curve, is chosen by the caller since `phat` must match).
    pub times: &'a [f64],
    /// Model predicted probability of an event by each evaluation time:
    /// `phat[i][j]` for time `i`, subject `j` (R's `p1`).
    pub phat: &'a [Vec<f64>],
    /// Move censorings just after tied events before estimating the
    /// censoring distribution (R `ties`).
    pub ties: bool,
    /// Use the Efron-type baseline for the null model (`survfit(..., ctype
    /// = 2, stype = 2)`); R does so when `efron = TRUE` and the fit used the
    /// Efron approximation.
    pub efron: bool,
    /// Apply R's `aeqSurv` near-tie rounding to `time` first.
    pub timefix: bool,
}

/// Output of [`brier`]: R's `brier` object with `detail = TRUE`.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object, get_all)]
pub struct BrierResult {
    pub times: Vec<f64>,
    pub brier: Vec<f64>,
    pub rsquared: Vec<f64>,
    /// Null-model event probabilities at `times`.
    pub p0: Vec<f64>,
    /// Model event probabilities, one row per time (the `phat` input).
    pub phat: Vec<Vec<f64>>,
    /// Effective sample size `1 / sum(w^2)` at each time.
    pub eff_n: Vec<f64>,
}

/// Value of a right-continuous step function at `at`: `1` before the
/// first step (`summary.survfit(extend = TRUE)` before any event).
fn step_value_at(times: &[f64], values: &[f64], at: f64) -> f64 {
    let index = times.partition_point(|&time| time <= at);
    if index == 0 { 1.0 } else { values[index - 1] }
}

/// `survfit(Surv(time, status) ~ 1, weights, se.fit = FALSE)`: the
/// Kaplan-Meier curve, or `exp(-H)` with the Efron-corrected hazard
/// (`ctype = 2`, `stype = 2`) that matches an Efron Cox fit's baseline.
fn survfit_curve(
    start: Option<&[f64]>,
    time: &[f64],
    status: &[i32],
    weights: &[f64],
    efron: bool,
) -> SurvivalResult<(Vec<f64>, Vec<f64>)> {
    let data = SurvfitKMData::try_new(
        start.map(<[f64]>::to_vec),
        time.to_vec(),
        status.to_vec(),
        Some(weights.to_vec()),
        None,
        None,
        None,
    )?;
    let (stype, ctype) = if efron {
        (SurvType::ExpCumhaz, HazardType::FlemingHarrington)
    } else {
        (SurvType::KaplanMeier, HazardType::NelsonAalen)
    };
    let options = SurvfitKMOptions {
        stype,
        ctype,
        se_fit: false,
        ..SurvfitKMOptions::default()
    };
    let fit = survfitkm(&data, &options)?;
    Ok((fit.time, fit.surv))
}

fn validate(input: &BrierInput<'_>) -> SurvivalResult<()> {
    let n = input.time.len();
    if n == 0 {
        return Err(SurvivalError::invalid_input("time must not be empty"));
    }
    validate_length(n, input.status.len(), "status")?;
    validate_finite(input.time, "time")?;
    if let Some(start) = input.start {
        validate_length(n, start.len(), "start")?;
        validate_finite(start, "start")?;
    }
    validate_binary_i32(input.status, "status")?;
    if let Some(weights) = input.weights {
        validate_length(n, weights.len(), "weights")?;
        validate_finite(weights, "weights")?;
        if weights.iter().any(|&w| w < 0.0) {
            return Err(SurvivalError::invalid_input("weights must be non-negative"));
        }
    }
    validate_finite(input.times, "times")?;
    validate_length(input.times.len(), input.phat.len(), "phat rows")?;
    for (row, values) in input.phat.iter().enumerate() {
        validate_length(n, values.len(), &format!("phat[{row}]"))?;
        validate_finite(values, "phat")?;
    }
    Ok(())
}

/// Brier score, its null-model counterpart and the resulting R-squared at
/// each evaluation time.
pub fn brier(input: &BrierInput<'_>) -> SurvivalResult<BrierResult> {
    validate(input)?;
    let n = input.time.len();
    let (start, time): (Option<Vec<f64>>, Vec<f64>) = match (input.timefix, input.start) {
        (false, start) => (start.map(<[f64]>::to_vec), input.time.to_vec()),
        (true, None) => (None, aeq_surv(input.time, None, None)?.time),
        (true, Some(start)) => {
            let fixed = aeq_surv(start, Some(input.time), None)?;
            let stop = fixed
                .time2
                .ok_or_else(|| SurvivalError::computation("aeqSurv dropped the stop times"))?;
            (Some(fixed.time), stop)
        }
    };
    let status: Vec<f64> = input.status.iter().map(|&s| f64::from(s)).collect();
    let weights: Vec<f64> = input.weights.map_or_else(|| vec![1.0; n], <[f64]>::to_vec);
    let total_weight: f64 = weights.iter().sum();
    if total_weight <= 0.0 {
        return Err(SurvivalError::invalid_input(
            "weights must have a positive sum",
        ));
    }
    // Null model: survfit(Y ~ 1, weights = casewt), evaluated with extend = TRUE.
    let (null_time, null_surv) =
        survfit_curve(start.as_deref(), &time, input.status, &weights, input.efron)?;
    let p0: Vec<f64> = input
        .times
        .iter()
        .map(|&at| 1.0 - step_value_at(&null_time, &null_surv, at))
        .collect();

    // Censoring distribution, with censorings nudged past tied events.
    let shifted: Vec<f64> = if input.ties {
        let mut unique = time.clone();
        unique.sort_by(f64::total_cmp);
        unique.dedup();
        let mindiff = unique
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .fold(f64::INFINITY, f64::min);
        time.iter()
            .zip(&status)
            .map(|(&t, &s)| if s == 0.0 { t + mindiff / 2.0 } else { t })
            .collect()
    } else {
        time.clone()
    };
    let censor_status: Vec<i32> = input.status.iter().map(|&s| 1 - s).collect();
    let (censor_time, censor_surv) =
        survfit_curve(None, &shifted, &censor_status, &weights, false)?;

    let case_weight: Vec<f64> = weights.iter().map(|w| w / total_weight).collect();
    let score_at = |i: usize| -> (f64, f64, f64) {
        let at = input.times[i];
        let mut weight_sum = 0.0;
        let mut weight_square_sum = 0.0;
        let mut null_sum = 0.0;
        let mut model_sum = 0.0;
        for j in 0..n {
            let dtime = shifted[j];
            let weight = if dtime < at && status[j] == 0.0 {
                0.0
            } else {
                case_weight[j] / step_value_at(&censor_time, &censor_surv, dtime.min(at))
            };
            let (b0, b1) = if dtime > at {
                (p0[i] * p0[i], input.phat[i][j] * input.phat[i][j])
            } else {
                (
                    (status[j] - p0[i]).powi(2),
                    (status[j] - input.phat[i][j]).powi(2),
                )
            };
            weight_sum += weight;
            weight_square_sum += weight * weight;
            null_sum += weight * b0;
            model_sum += weight * b1;
        }
        (
            null_sum / weight_sum,
            model_sum / weight_sum,
            1.0 / weight_square_sum,
        )
    };
    let n_times = input.times.len();
    let rows: Vec<(f64, f64, f64)> = if n.saturating_mul(n_times) >= PARALLEL_WORK_THRESHOLD {
        (0..n_times).into_par_iter().map(score_at).collect()
    } else {
        (0..n_times).map(score_at).collect()
    };
    let mut brier = Vec::with_capacity(n_times);
    let mut rsquared = Vec::with_capacity(n_times);
    let mut eff_n = Vec::with_capacity(n_times);
    for (null_brier, model_brier, effective) in rows {
        brier.push(model_brier);
        rsquared.push(1.0 - model_brier / null_brier);
        eff_n.push(effective);
    }
    Ok(BrierResult {
        times: input.times.to_vec(),
        brier,
        rsquared,
        p0,
        phat: input.phat.to_vec(),
        eff_n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_a_hand_calculation() {
        // Two subjects, no censoring: the censoring curve stays at 1 and the
        // weights are the normalised case weights.
        let phat = vec![vec![0.2, 0.6]];
        let result = brier(&BrierInput {
            start: None,
            time: &[1.0, 3.0],
            status: &[1, 1],
            weights: None,
            times: &[2.0],
            phat: &phat,
            ties: true,
            efron: false,
            timefix: true,
        })
        .unwrap();
        // KM at t=2: S = 0.5, so p0 = 0.5.  Subject 1 (event before 2):
        // (1 - p)^2; subject 2 (after): p^2.
        assert!((result.p0[0] - 0.5).abs() < 1e-12);
        let expected_null = (0.25 + 0.25) / 2.0;
        let expected_model = ((1.0f64 - 0.2).powi(2) + 0.6f64.powi(2)) / 2.0;
        assert!((result.brier[0] - expected_model).abs() < 1e-12);
        assert!((result.rsquared[0] - (1.0 - expected_model / expected_null)).abs() < 1e-12);
        assert!((result.eff_n[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn censoring_weights_follow_the_ipcw_rule() {
        // Subject 2 censored at 2 (shifted to 2.5) gets weight zero at time
        // 3; the others are reweighted by the censoring survival.
        let phat = vec![vec![0.3, 0.3, 0.3]];
        let result = brier(&BrierInput {
            start: None,
            time: &[1.0, 2.0, 4.0],
            status: &[1, 0, 1],
            weights: None,
            times: &[3.0],
            phat: &phat,
            ties: true,
            efron: false,
            timefix: true,
        })
        .unwrap();
        // G(t): drops to 0.5 at 2.5.  Weights: subject 1 -> (1/3)/1,
        // subject 2 -> 0, subject 3 -> (1/3)/0.5.
        let w1 = 1.0 / 3.0;
        let w3 = 2.0 / 3.0;
        assert!((result.eff_n[0] - 1.0 / (w1 * w1 + w3 * w3)).abs() < 1e-12);
        assert!(result.brier[0].is_finite());
    }

    #[test]
    fn efron_null_model_uses_the_corrected_hazard() {
        let phat = vec![vec![0.5; 3]];
        let km = brier(&BrierInput {
            start: None,
            time: &[1.0, 1.0, 2.0],
            status: &[1, 1, 0],
            weights: None,
            times: &[1.5],
            phat: &phat,
            ties: true,
            efron: false,
            timefix: true,
        })
        .unwrap();
        let efron = brier(&BrierInput {
            start: None,
            time: &[1.0, 1.0, 2.0],
            status: &[1, 1, 0],
            weights: None,
            times: &[1.5],
            phat: &phat,
            ties: true,
            efron: true,
            timefix: true,
        })
        .unwrap();
        assert!((km.p0[0] - (1.0 - 1.0 / 3.0)).abs() < 1e-12);
        let hazard: f64 = 1.0 / 3.0 + 1.0 / 2.0;
        assert!((efron.p0[0] - (1.0 - (-hazard).exp())).abs() < 1e-12);
    }

    #[test]
    fn shapes_are_validated() {
        let bad = brier(&BrierInput {
            start: None,
            time: &[1.0, 2.0],
            status: &[1, 0],
            weights: None,
            times: &[1.0, 2.0],
            phat: &[vec![0.1, 0.2], vec![0.3]],
            ties: true,
            efron: false,
            timefix: true,
        });
        assert!(bad.is_err());
    }
}
