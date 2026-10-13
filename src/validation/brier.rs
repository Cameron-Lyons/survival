use crate::data_prep::aeq_counting;
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::step::step_at;
use crate::internal::validation::{validate_binary_i32, validate_finite, validate_length};
use crate::surv_analysis::{HazardType, SurvType, SurvfitKMData, SurvfitKMOptions, survfitkm};
use pyo3::prelude::*;
use rayon::prelude::*;

const PARALLEL_WORK_THRESHOLD: usize = 8_192;

#[derive(Debug, Clone)]
pub struct BrierInput<'a> {
    pub start: Option<&'a [f64]>,
    pub time: &'a [f64],
    pub status: &'a [i32],
    pub weights: Option<&'a [f64]>,
    pub times: &'a [f64],
    pub phat: Vec<Vec<f64>>,
    pub ties: bool,
    pub efron: bool,
    pub timefix: bool,
}

#[doc = " Output of [`brier`]: R's `brier` object with `detail = TRUE`."]
#[derive(Debug, Clone, PartialEq)]
#[pyclass(module = "survival._survival", from_py_object, get_all)]
pub struct BrierResult {
    pub times: Vec<f64>,
    pub brier: Vec<f64>,
    pub rsquared: Vec<f64>,
    #[doc = " Null-model event probabilities at `times`."]
    pub p0: Vec<f64>,
    #[doc = " Model event probabilities, one row per time (the `phat` input)."]
    pub phat: Vec<Vec<f64>>,
    #[doc = " Effective sample size `1 / sum(w^2)` at each time."]
    pub eff_n: Vec<f64>,
}

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

pub fn brier(input: BrierInput<'_>) -> SurvivalResult<BrierResult> {
    validate(&input)?;
    let n = input.time.len();
    let (start, time) = if input.timefix {
        aeq_counting(input.start, input.time)?
    } else {
        (input.start.map(<[f64]>::to_vec), input.time.to_vec())
    };
    let status: Vec<f64> = input.status.iter().map(|&s| f64::from(s)).collect();
    let weights: Vec<f64> = input.weights.map_or_else(|| vec![1.0; n], <[f64]>::to_vec);
    let total_weight: f64 = weights.iter().sum();
    if total_weight <= 0.0 {
        return Err(SurvivalError::invalid_input(
            "weights must have a positive sum",
        ));
    }
    let (null_time, null_surv) =
        survfit_curve(start.as_deref(), &time, input.status, &weights, input.efron)?;
    let p0: Vec<f64> = input
        .times
        .iter()
        .map(|&at| 1.0 - step_at(&null_time, &null_surv, at, 1.0))
        .collect();

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
    let censor_at_subject: Vec<f64> = shifted
        .iter()
        .map(|&at| step_at(&censor_time, &censor_surv, at, 1.0))
        .collect();
    let score_at = |i: usize| -> (f64, f64, f64) {
        let at = input.times[i];
        let censor_at_time = step_at(&censor_time, &censor_surv, at, 1.0);
        let mut weight_sum = 0.0;
        let mut weight_square_sum = 0.0;
        let mut null_sum = 0.0;
        let mut model_sum = 0.0;
        for j in 0..n {
            let dtime = shifted[j];
            let weight = if dtime < at && status[j] == 0.0 {
                0.0
            } else {
                let censoring = if dtime <= at {
                    censor_at_subject[j]
                } else {
                    censor_at_time
                };
                case_weight[j] / censoring
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
        phat: input.phat,
        eff_n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_censoring_values_match_direct_ipcw() {
        for copies in [1, 256] {
            let time = [4.0, 1.0, 2.0, 2.0, 6.0, 5.0, 6.0, 3.0].repeat(copies);
            let status = [1, 1, 0, 1, 0, 0, 1, 1].repeat(copies);
            let weights = [0.5, 1.0, 0.0, 1.5, 0.5, 2.0, 1.0, 1.5].repeat(copies);
            let times = [-1.0, 2.0, 2.5, 7.0, 4.0, 2.0, 6.0];
            let phat: Vec<Vec<f64>> = times.iter().map(|_| vec![0.3; time.len()]).collect();
            for ties in [false, true] {
                let shifted: Vec<f64> = time
                    .iter()
                    .zip(&status)
                    .map(|(&t, &s)| if ties && s == 0 { t + 0.5 } else { t })
                    .collect();
                let mut unique = shifted.clone();
                unique.sort_by(f64::total_cmp);
                unique.dedup();
                let censoring: Vec<f64> = unique
                    .iter()
                    .scan(1.0, |survival, &t| {
                        let risk: f64 = (0..time.len())
                            .filter(|&j| shifted[j] >= t)
                            .map(|j| weights[j])
                            .sum();
                        let censored: f64 = (0..time.len())
                            .filter(|&j| shifted[j] == t && status[j] == 0)
                            .map(|j| weights[j])
                            .sum();
                        *survival *= 1.0 - censored / risk;
                        Some(*survival)
                    })
                    .collect();
                let total: f64 = weights.iter().sum();
                for efron in [false, true] {
                    let result = brier(BrierInput {
                        start: None,
                        time: &time,
                        status: &status,
                        weights: Some(&weights),
                        times: &times,
                        phat: phat.clone(),
                        ties,
                        efron,
                        timefix: false,
                    })
                    .unwrap();
                    for (i, &at) in times.iter().enumerate() {
                        let mut weight_sum = 0.0;
                        let mut weight_square_sum = 0.0;
                        let mut null_sum = 0.0;
                        let mut model_sum = 0.0;
                        for j in 0..time.len() {
                            let weight = if shifted[j] < at && status[j] == 0 {
                                0.0
                            } else {
                                weights[j]
                                    / total
                                    / step_at(&unique, &censoring, shifted[j].min(at), 1.0)
                            };
                            let outcome = if shifted[j] > at {
                                0.0
                            } else {
                                f64::from(status[j])
                            };
                            weight_sum += weight;
                            weight_square_sum += weight * weight;
                            null_sum += weight * (outcome - result.p0[i]).powi(2);
                            model_sum += weight * (outcome - phat[i][j]).powi(2);
                        }
                        let expected = [
                            model_sum / weight_sum,
                            1.0 - model_sum / null_sum,
                            1.0 / weight_square_sum,
                        ];
                        let actual = [result.brier[i], result.rsquared[i], result.eff_n[i]];
                        for (actual, expected) in actual.into_iter().zip(expected) {
                            assert!(
                                (actual.is_nan() && expected.is_nan())
                                    || actual == expected
                                    || (actual - expected).abs() <= 1e-11 * expected.abs().max(1.0),
                                "{actual} != {expected}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn matches_a_hand_calculation() {
        let result = brier(BrierInput {
            start: None,
            time: &[1.0, 3.0],
            status: &[1, 1],
            weights: None,
            times: &[2.0],
            phat: vec![vec![0.2, 0.6]],
            ties: true,
            efron: false,
            timefix: true,
        })
        .unwrap();
        assert!((result.p0[0] - 0.5).abs() < 1e-12);
        let expected_null = (0.25 + 0.25) / 2.0;
        let expected_model = ((1.0f64 - 0.2).powi(2) + 0.6f64.powi(2)) / 2.0;
        assert!((result.brier[0] - expected_model).abs() < 1e-12);
        assert!((result.rsquared[0] - (1.0 - expected_model / expected_null)).abs() < 1e-12);
        assert!((result.eff_n[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn censoring_weights_follow_the_ipcw_rule() {
        let result = brier(BrierInput {
            start: None,
            time: &[1.0, 2.0, 4.0],
            status: &[1, 0, 1],
            weights: None,
            times: &[3.0],
            phat: vec![vec![0.3; 3]],
            ties: true,
            efron: false,
            timefix: true,
        })
        .unwrap();
        let w1 = 1.0 / 3.0;
        let w3 = 2.0 / 3.0;
        assert!((result.eff_n[0] - 1.0 / (w1 * w1 + w3 * w3)).abs() < 1e-12);
        assert!(result.brier[0].is_finite());
    }

    #[test]
    fn efron_null_model_uses_the_corrected_hazard() {
        let km = brier(BrierInput {
            start: None,
            time: &[1.0, 1.0, 2.0],
            status: &[1, 1, 0],
            weights: None,
            times: &[1.5],
            phat: vec![vec![0.5; 3]],
            ties: true,
            efron: false,
            timefix: true,
        })
        .unwrap();
        let efron = brier(BrierInput {
            start: None,
            time: &[1.0, 1.0, 2.0],
            status: &[1, 1, 0],
            weights: None,
            times: &[1.5],
            phat: vec![vec![0.5; 3]],
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
        let bad = brier(BrierInput {
            start: None,
            time: &[1.0, 2.0],
            status: &[1, 0],
            weights: None,
            times: &[1.0, 2.0],
            phat: vec![vec![0.1, 0.2], vec![0.3]],
            ties: true,
            efron: false,
            timefix: true,
        });
        assert!(bad.is_err());
    }
}
