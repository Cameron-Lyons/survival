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

/// The outcome data and options of a [`BrierInput`], without its predictions.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BrierOutcomes<'a> {
    pub start: Option<&'a [f64]>,
    pub time: &'a [f64],
    pub status: &'a [i32],
    pub weights: Option<&'a [f64]>,
    pub times: &'a [f64],
    pub ties: bool,
    pub efron: bool,
    pub timefix: bool,
}

impl BrierInput<'_> {
    fn outcomes(&self) -> BrierOutcomes<'_> {
        BrierOutcomes {
            start: self.start,
            time: self.time,
            status: self.status,
            weights: self.weights,
            times: self.times,
            ties: self.ties,
            efron: self.efron,
            timefix: self.timefix,
        }
    }
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

fn validate_outcomes(outcomes: &BrierOutcomes<'_>) -> SurvivalResult<()> {
    let n = outcomes.time.len();
    if n == 0 {
        return Err(SurvivalError::invalid_input("time must not be empty"));
    }
    validate_length(n, outcomes.status.len(), "status")?;
    validate_finite(outcomes.time, "time")?;
    if let Some(start) = outcomes.start {
        validate_length(n, start.len(), "start")?;
        validate_finite(start, "start")?;
    }
    validate_binary_i32(outcomes.status, "status")?;
    if let Some(weights) = outcomes.weights {
        validate_length(n, weights.len(), "weights")?;
        validate_finite(weights, "weights")?;
        if weights.iter().any(|&w| w < 0.0) {
            return Err(SurvivalError::invalid_input("weights must be non-negative"));
        }
    }
    validate_finite(outcomes.times, "times")?;
    Ok(())
}

fn validate(input: &BrierInput<'_>) -> SurvivalResult<()> {
    validate_outcomes(&input.outcomes())?;
    let n = input.time.len();
    validate_length(input.times.len(), input.phat.len(), "phat rows")?;
    for (row, values) in input.phat.iter().enumerate() {
        validate_length(n, values.len(), &format!("phat[{row}]"))?;
        validate_finite(values, "phat")?;
    }
    Ok(())
}

/// Everything a Brier score reads besides the predictions: the null model at the
/// requested times and the inverse-probability-of-censoring weights.
struct BrierScoring {
    p0: Vec<f64>,
    shifted: Vec<f64>,
    status: Vec<f64>,
    case_weight: Vec<f64>,
    censor_time: Vec<f64>,
    censor_surv: Vec<f64>,
    censor_at_subject: Vec<f64>,
}

impl BrierScoring {
    fn new(outcomes: &BrierOutcomes<'_>) -> SurvivalResult<Self> {
        let n = outcomes.time.len();
        let (start, time) = if outcomes.timefix {
            aeq_counting(outcomes.start, outcomes.time)?
        } else {
            (outcomes.start.map(<[f64]>::to_vec), outcomes.time.to_vec())
        };
        let status: Vec<f64> = outcomes.status.iter().map(|&s| f64::from(s)).collect();
        let weights: Vec<f64> = outcomes
            .weights
            .map_or_else(|| vec![1.0; n], <[f64]>::to_vec);
        let total_weight: f64 = weights.iter().sum();
        if total_weight <= 0.0 {
            return Err(SurvivalError::invalid_input(
                "weights must have a positive sum",
            ));
        }
        let (null_time, null_surv) = survfit_curve(
            start.as_deref(),
            &time,
            outcomes.status,
            &weights,
            outcomes.efron,
        )?;
        let p0: Vec<f64> = outcomes
            .times
            .iter()
            .map(|&at| 1.0 - step_at(&null_time, &null_surv, at, 1.0))
            .collect();

        let shifted: Vec<f64> = if outcomes.ties {
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
        let censor_status: Vec<i32> = outcomes.status.iter().map(|&s| 1 - s).collect();
        let (censor_time, censor_surv) =
            survfit_curve(None, &shifted, &censor_status, &weights, false)?;

        let case_weight: Vec<f64> = weights.iter().map(|w| w / total_weight).collect();
        let censor_at_subject: Vec<f64> = shifted
            .iter()
            .map(|&at| step_at(&censor_time, &censor_surv, at, 1.0))
            .collect();
        Ok(Self {
            p0,
            shifted,
            status,
            case_weight,
            censor_time,
            censor_surv,
            censor_at_subject,
        })
    }

    /// The null and model Brier scores and the effective sample size at the `i`th
    /// time `at`, for event probabilities `phat` there.
    fn score_at(&self, i: usize, at: f64, phat: &[f64]) -> (f64, f64, f64) {
        let p0 = self.p0[i];
        let status = &self.status;
        let censor_at_time = step_at(&self.censor_time, &self.censor_surv, at, 1.0);
        let mut weight_sum = 0.0;
        let mut weight_square_sum = 0.0;
        let mut null_sum = 0.0;
        let mut model_sum = 0.0;
        for (j, &dtime) in self.shifted.iter().enumerate() {
            let weight = if dtime < at && status[j] == 0.0 {
                0.0
            } else {
                let censoring = if dtime <= at {
                    self.censor_at_subject[j]
                } else {
                    censor_at_time
                };
                self.case_weight[j] / censoring
            };
            let (b0, b1) = if dtime > at {
                (p0 * p0, phat[j] * phat[j])
            } else {
                ((status[j] - p0).powi(2), (status[j] - phat[j]).powi(2))
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
    }

    fn into_result(
        self,
        times: &[f64],
        rows: Vec<(f64, f64, f64)>,
        phat: Vec<Vec<f64>>,
    ) -> BrierResult {
        let mut brier = Vec::with_capacity(rows.len());
        let mut rsquared = Vec::with_capacity(rows.len());
        let mut eff_n = Vec::with_capacity(rows.len());
        for (null_brier, model_brier, effective) in rows {
            brier.push(model_brier);
            rsquared.push(1.0 - model_brier / null_brier);
            eff_n.push(effective);
        }
        BrierResult {
            times: times.to_vec(),
            brier,
            rsquared,
            p0: self.p0,
            phat,
            eff_n,
        }
    }
}

pub fn brier(input: BrierInput<'_>) -> SurvivalResult<BrierResult> {
    validate(&input)?;
    let scoring = BrierScoring::new(&input.outcomes())?;
    let score = |i: usize| scoring.score_at(i, input.times[i], &input.phat[i]);
    let n_times = input.times.len();
    let rows: Vec<(f64, f64, f64)> =
        if input.time.len().saturating_mul(n_times) >= PARALLEL_WORK_THRESHOLD {
            (0..n_times).into_par_iter().map(score).collect()
        } else {
            (0..n_times).map(score).collect()
        };
    Ok(scoring.into_result(input.times, rows, input.phat))
}

/// [`brier`] of predictions that are evaluated rather than stored: `fill(i, row)`
/// writes the `columns` event probabilities at `outcomes.times[i]`. Each row is
/// scored as it is filled, so memory stays one row per thread and the result's
/// `phat` is empty. Errors are those of [`brier`] on the stored rows, in its order.
pub(crate) fn brier_of_rows<F>(
    outcomes: BrierOutcomes<'_>,
    columns: usize,
    fill: F,
) -> SurvivalResult<BrierResult>
where
    F: Fn(usize, &mut [f64]) + Sync,
{
    validate_outcomes(&outcomes)?;
    let n = outcomes.time.len();
    let n_times = outcomes.times.len();
    if n_times > 0 {
        validate_length(n, columns, "phat[0]")?;
    }
    let evaluate = |i: usize, row: &mut [f64]| -> SurvivalResult<()> {
        fill(i, row);
        validate_finite(row, "phat")?;
        Ok(())
    };
    let scoring = match BrierScoring::new(&outcomes) {
        Ok(scoring) => scoring,
        Err(error) => {
            // `brier` checks every prediction row before this setup can fail.
            let mut row = vec![0.0; n];
            for i in 0..n_times {
                evaluate(i, &mut row)?;
            }
            return Err(error);
        }
    };
    let score = |row: &mut Vec<f64>, i: usize| -> SurvivalResult<(f64, f64, f64)> {
        evaluate(i, row)?;
        Ok(scoring.score_at(i, outcomes.times[i], row))
    };
    let rows: Vec<SurvivalResult<(f64, f64, f64)>> =
        if n.saturating_mul(n_times) >= PARALLEL_WORK_THRESHOLD {
            (0..n_times)
                .into_par_iter()
                .map_init(|| vec![0.0; n], score)
                .collect()
        } else {
            let mut row = vec![0.0; n];
            (0..n_times).map(|i| score(&mut row, i)).collect()
        };
    // The first failing row in time order, as `brier`'s up-front check reports it.
    let rows = rows.into_iter().collect::<SurvivalResult<Vec<_>>>()?;
    Ok(scoring.into_result(outcomes.times, rows, Vec::new()))
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

    /// `brier` of the stored rows and `brier_of_rows` filling the same rows.
    fn both_paths(
        input: BrierInput<'_>,
        columns: usize,
    ) -> (SurvivalResult<BrierResult>, SurvivalResult<BrierResult>) {
        let outcomes = input.outcomes();
        let rows = input.phat.clone();
        let evaluated = brier_of_rows(outcomes, columns, |i, row| {
            row.copy_from_slice(&rows[i]);
        });
        (brier(input), evaluated)
    }

    fn bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    #[test]
    fn evaluated_rows_score_as_the_stored_matrix_bitwise() {
        let mut rng = crate::internal::rng::Rng::with_seed(77);
        for n in [5, 400] {
            let time: Vec<f64> = (0..n).map(|_| (rng.f64() * 40.0).round() / 4.0).collect();
            let status: Vec<i32> = (0..n).map(|_| i32::from(rng.f64() < 0.6)).collect();
            let start: Vec<f64> = time.iter().map(|t| t - 1.0 - rng.f64()).collect();
            let weights: Vec<f64> = (0..n)
                .map(|i| if i % 7 == 3 { 0.0 } else { rng.f64() * 2.0 })
                .collect();
            let times: Vec<f64> = [-1.0, 0.0, 2.5, 2.5, 1e9]
                .into_iter()
                .chain((0..60).map(|_| rng.f64() * 10.0))
                .collect();
            let phat: Vec<Vec<f64>> = times
                .iter()
                .map(|_| (0..n).map(|_| rng.f64()).collect())
                .collect();
            for (ties, efron, timefix) in [
                (true, false, true),
                (false, true, false),
                (true, true, false),
            ] {
                for start in [None, Some(start.as_slice())] {
                    for weights in [None, Some(weights.as_slice())] {
                        let (stored, evaluated) = both_paths(
                            BrierInput {
                                start,
                                time: &time,
                                status: &status,
                                weights,
                                times: &times,
                                phat: phat.clone(),
                                ties,
                                efron,
                                timefix,
                            },
                            n,
                        );
                        let (stored, evaluated) = (stored.unwrap(), evaluated.unwrap());
                        assert_eq!(bits(&evaluated.brier), bits(&stored.brier));
                        assert_eq!(bits(&evaluated.rsquared), bits(&stored.rsquared));
                        assert_eq!(bits(&evaluated.eff_n), bits(&stored.eff_n));
                        assert_eq!(bits(&evaluated.p0), bits(&stored.p0));
                        assert_eq!(evaluated.times, stored.times);
                        assert!(evaluated.phat.is_empty());
                    }
                }
            }
        }
    }

    #[test]
    fn evaluated_rows_report_the_stored_matrix_errors() {
        const TIME: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
        const TIMES: [f64; 4] = [0.5, 1.5, 2.5, 3.5];
        fn input<'a>(
            weights: Option<&'a [f64]>,
            phat: &[Vec<f64>],
            times: &'a [f64],
        ) -> BrierInput<'a> {
            BrierInput {
                start: None,
                time: &TIME,
                status: &[1, 0, 1, 1],
                weights,
                times,
                phat: phat.to_vec(),
                ties: true,
                efron: false,
                timefix: true,
            }
        }
        let mut phat = vec![vec![0.25; 4]; 4];
        phat[1][3] = f64::NAN;
        phat[3][0] = f64::INFINITY;
        let zero = [0.0; 4];
        let finite = vec![vec![0.25; 4]; 4];
        let cases = [
            // The first non-finite row in time order, also before a zero weight sum.
            input(None, &phat, &TIMES),
            input(Some(&zero), &phat, &TIMES),
            input(Some(&zero), &finite, &TIMES),
            input(None, &[], &TIMES[..0]),
        ];
        for input in cases {
            match both_paths(input, 4) {
                (Err(stored), Err(evaluated)) => {
                    assert_eq!(evaluated.to_string(), stored.to_string());
                }
                (Ok(stored), Ok(evaluated)) => assert_eq!(evaluated.brier, stored.brier),
                (stored, evaluated) => panic!("{stored:?} != {evaluated:?}"),
            }
        }
        let wide = brier_of_rows(input(None, &finite, &TIMES).outcomes(), 5, |_, _| {})
            .unwrap_err()
            .to_string();
        let mut long_rows = finite.clone();
        long_rows[0].push(0.25);
        let stored = brier(input(None, &long_rows, &TIMES)).unwrap_err();
        assert_eq!(wide, stored.to_string());
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
