//! Survival predictions filled one time row at a time, checked bit for bit against
//! the per-stratum loop they replace, and the Brier scores computed from them.

use super::*;
use crate::error::SurvivalResult;
use crate::internal::rng::Rng;
use crate::internal::step::find_interval;
use crate::internal::validation::validate_finite;
use crate::regression::cox_optimizer::TieMethod;
use crate::validation::brier::BrierOutcomes;
use crate::validation::{BrierInput, BrierResult, brier};
use ndarray::Array2;

/// `predict_survival_at` as one loop per stratum over every time, writing into a
/// matrix of ones: the bitwise reference for the row-by-row fill.
fn predict_survival_at_reference(
    fit: &CoxPHFit,
    times: &[f64],
    newdata: Option<&CoxNewData>,
) -> SurvivalResult<Array2<f64>> {
    fit.check_prediction_dimensions()?;
    validate_finite(times, "times")?;
    let (x, strata, offset) = match newdata {
        Some(newdata) => {
            fit.check_newdata(newdata, false)?;
            (
                newdata.x.view(),
                newdata.strata.as_deref(),
                newdata.offset.as_deref(),
            )
        }
        None => (
            fit.x.view(),
            fit.strata.as_deref(),
            Some(fit.offset.as_slice()),
        ),
    };
    let mut result = Array2::ones((times.len(), x.nrows()));
    if times.is_empty() {
        return Ok(result);
    }
    let curves = fit.baseline_curves()?;
    let mut rows_by_stratum = vec![Vec::new(); curves.len()];
    for row in 0..x.nrows() {
        let position = strata.map_or(0, |codes| fit.sorted.position_of(codes[row]).unwrap());
        rows_by_stratum[position].push(row);
    }
    let log_risk = if newdata.is_none() && fit.has_curve_origins() {
        fit.training_prediction_risks(&curves.origins)
    } else {
        fit.prediction_log_risks_at(x, offset, &curves.origins, strata)
    };
    let risk = log_risk.risks();
    let stable = log_risk.needs_logs(curves.stable);
    for (position, (curve, rows)) in curves.iter().zip(rows_by_stratum).enumerate() {
        if rows.is_empty() {
            continue;
        }
        for (i, &at) in times.iter().enumerate() {
            let index = find_interval(&curve.time, at, false);
            if index == 0 {
                continue;
            }
            let baseline = (-curve.cumhaz[index - 1]).exp();
            let mut output = result.row_mut(i);
            for &row in &rows {
                output[row] = if stable {
                    (-curves.logs()[position].log_cumulative[index - 1]
                        .exp_at_scale(log_risk.scale(row)))
                    .exp()
                } else if baseline == 0.0 && curve.cumhaz[index - 1].is_finite() {
                    (-curve.cumhaz[index - 1] * risk[row]).exp()
                } else {
                    baseline.powf(risk[row])
                };
            }
        }
    }
    Ok(result)
}

const STRATA: [i32; 3] = [4, -2, 9];

struct Case {
    method: TieMethod,
    stratified: bool,
    weighted: bool,
    counting: bool,
}

/// A fit to `n` random rows with tied times, of the shape `case` asks for.
fn random_fit(rng: &mut Rng, n: usize, case: &Case) -> CoxPHFit {
    let x = Array2::from_shape_fn((n, 2), |_| rng.f64() * 2.0 - 1.0);
    // Times shrink with the first covariate, so its coefficient is clearly nonzero.
    let time: Vec<f64> = (0..n)
        .map(|row| 0.1 + (rng.f64() * 30.0 * (-x[(row, 0)]).exp()).round() / 10.0)
        .collect();
    let status: Vec<i32> = (0..n).map(|_| i32::from(rng.f64() < 0.7)).collect();
    let entry = case
        .counting
        .then(|| time.iter().map(|t| t * rng.f64() * 0.6).collect());
    let weights = case
        .weighted
        .then(|| (0..n).map(|_| 0.2 + rng.f64() * 2.0).collect());
    let strata = case
        .stratified
        .then(|| (0..n).map(|_| STRATA[rng.usize(0..3)]).collect());
    let data = CoxphData::try_new(time, entry, status, x, weights, strata, None).unwrap();
    CoxPHFit::fit(
        data,
        CoxphOptions {
            method: case.method,
            ..Default::default()
        },
    )
    .unwrap()
}

/// New rows for `fit`, some far enough out that the log-scale curves are needed.
fn random_newdata(rng: &mut Rng, fit: &CoxPHFit, rows: usize, extreme: bool) -> CoxNewData {
    let x = Array2::from_shape_fn((rows, 2), |(row, _)| {
        if extreme && row % 9 == 0 {
            if row % 2 == 0 { 300.0 } else { -300.0 }
        } else {
            rng.f64() * 3.0 - 1.5
        }
    });
    let strata = fit
        .strata
        .is_some()
        .then(|| (0..rows).map(|_| STRATA[rng.usize(0..3)]).collect());
    let offset = Some((0..rows).map(|_| rng.f64() * 0.4 - 0.2).collect());
    CoxNewData::try_new(x, strata, offset, None, None).unwrap()
}

fn bits(values: impl IntoIterator<Item = f64>) -> Vec<u64> {
    values.into_iter().map(f64::to_bits).collect()
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for method in [TieMethod::Breslow, TieMethod::Efron] {
        for (stratified, weighted, counting) in [
            (false, false, false),
            (true, false, false),
            (false, true, false),
            (true, true, true),
            (false, false, true),
        ] {
            cases.push(Case {
                method,
                stratified,
                weighted,
                counting,
            });
        }
    }
    cases
}

fn requested_times(rng: &mut Rng) -> Vec<f64> {
    [-1.0, 0.0, 0.1, 2.5, 2.5, 1e9]
        .into_iter()
        .chain((0..80).map(|_| rng.f64() * 6.0))
        .collect()
}

#[test]
fn row_by_row_survival_matches_the_per_stratum_loop_bitwise() {
    let mut rng = Rng::with_seed(4_242);
    for case in cases() {
        // 250 rows by 86 times is enough work for the parallel fill.
        let fit = random_fit(&mut rng, 250, &case);
        let times = requested_times(&mut rng);
        let newdata = [
            random_newdata(&mut rng, &fit, 230, false),
            random_newdata(&mut rng, &fit, 7, true),
            random_newdata(&mut rng, &fit, 240, true),
        ];
        let targets = [
            None,
            Some(&newdata[0]),
            Some(&newdata[1]),
            Some(&newdata[2]),
        ];
        for newdata in targets {
            let actual = fit.predict_survival_at(&times, newdata).unwrap();
            let expected = predict_survival_at_reference(&fit, &times, newdata).unwrap();
            assert_eq!(actual.dim(), expected.dim());
            assert_eq!(bits(actual), bits(expected));
        }
        // The extreme rows take the log-scale curves.
        let curves = fit.baseline_curves().unwrap();
        let extreme = &newdata[2];
        let log_risk = fit.prediction_log_risks_at(
            extreme.x.view(),
            extreme.offset.as_deref(),
            &curves.origins,
            extreme.strata.as_deref(),
        );
        assert!(log_risk.needs_logs(false));
        assert_eq!(fit.predict_survival_at(&[], None).unwrap().dim(), (0, 250));
    }
}

#[test]
fn cox_brier_scores_match_brier_of_the_prediction_matrix_bitwise() {
    let mut rng = Rng::with_seed(9_001);
    for case in cases() {
        let fit = random_fit(&mut rng, 220, &case);
        let times = requested_times(&mut rng);
        let newdata = random_newdata(&mut rng, &fit, 180, true);
        let new_time: Vec<f64> = (0..180).map(|_| 0.1 + rng.f64() * 5.0).collect();
        let new_status: Vec<i32> = (0..180).map(|_| i32::from(rng.f64() < 0.6)).collect();
        // Brier weights may be zero, unlike the fit's.
        let weights: Vec<f64> = (0..220)
            .map(|i| if i % 5 == 0 { 0.0 } else { rng.f64() })
            .collect();
        let targets = [
            (
                None,
                fit.time.as_slice(),
                fit.status.as_slice(),
                fit.entry.as_deref(),
            ),
            (Some(&newdata), &new_time[..], &new_status[..], None),
        ];
        for (newdata, time, status, start) in targets {
            let weights = (newdata.is_none() && case.weighted).then_some(&weights[..]);
            for (ties, efron, timefix) in [(true, false, true), (false, true, false)] {
                let outcomes = BrierOutcomes {
                    start,
                    time,
                    status,
                    weights,
                    times: &times,
                    ties,
                    efron,
                    timefix,
                };
                let survival = fit.predict_survival_at(&times, newdata).unwrap();
                let phat = survival
                    .rows()
                    .into_iter()
                    .map(|row| row.iter().map(|s| 1.0 - s).collect())
                    .collect();
                let expected = brier(BrierInput {
                    start,
                    time,
                    status,
                    weights,
                    times: &times,
                    phat,
                    ties,
                    efron,
                    timefix,
                })
                .unwrap();
                let actual = fit.brier_scores(outcomes, newdata).unwrap();
                assert_eq!(bits(actual.brier), bits(expected.brier));
                assert_eq!(bits(actual.rsquared), bits(expected.rsquared));
                assert_eq!(bits(actual.eff_n), bits(expected.eff_n));
                assert_eq!(bits(actual.p0), bits(expected.p0));
                assert_eq!(actual.times, expected.times);
                assert!(actual.phat.is_empty());
            }
        }
    }
}

#[test]
fn cox_brier_scores_report_prediction_errors_first() {
    let mut rng = Rng::with_seed(5);
    let fit = random_fit(
        &mut rng,
        40,
        &Case {
            method: TieMethod::Efron,
            stratified: true,
            weighted: false,
            counting: false,
        },
    );
    const NAN_TIMES: [f64; 2] = [1.0, f64::NAN];
    let outcomes = |times: &'static [f64]| BrierOutcomes {
        start: None,
        time: &fit.time,
        status: &fit.status,
        weights: None,
        times,
        ties: true,
        efron: false,
        timefix: true,
    };
    let message = |result: SurvivalResult<BrierResult>| result.unwrap_err().to_string();
    assert_eq!(
        message(fit.brier_scores(outcomes(&NAN_TIMES), None)),
        fit.predict_survival_at(&NAN_TIMES, None)
            .unwrap_err()
            .to_string()
    );
    let mut newdata = random_newdata(&mut rng, &fit, 12, false);
    newdata.strata = None;
    assert_eq!(
        message(fit.brier_scores(outcomes(&[1.0]), Some(&newdata))),
        fit.predict_survival_at(&[1.0], Some(&newdata))
            .unwrap_err()
            .to_string()
    );
    newdata.strata = Some(vec![17; 12]);
    assert_eq!(
        message(fit.brier_scores(outcomes(&[1.0]), Some(&newdata))),
        fit.predict_survival_at(&[1.0], Some(&newdata))
            .unwrap_err()
            .to_string()
    );
    newdata.strata = Some(vec![4; 12]);
    // No requested times: nothing is predicted, as for an empty prediction matrix.
    let empty = fit.brier_scores(outcomes(&[]), Some(&newdata)).unwrap();
    assert!(empty.brier.is_empty() && empty.times.is_empty());
    // Training outcomes against new rows: `brier`'s row length error.
    let stored = brier(BrierInput {
        start: None,
        time: &fit.time,
        status: &fit.status,
        weights: None,
        times: &[1.0],
        phat: vec![vec![0.5; 12]],
        ties: true,
        efron: false,
        timefix: true,
    });
    assert_eq!(
        message(fit.brier_scores(outcomes(&[1.0]), Some(&newdata))),
        message(stored)
    );
}
