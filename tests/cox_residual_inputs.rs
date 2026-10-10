//! Standalone public Cox residual and score kernels reject mutated inputs.
//! Numerical controls compute each event's risk set directly, independently
//! of the optimized forward/backward sweeps used by the kernels.

use ndarray::{Array2, ArrayView2, array};
use std::collections::BTreeSet;
use survival::core::{SurvResponse, schoenfeld_residuals};
use survival::prelude::{AndersenGillInput, CountingProcessData, SurvivalData, Weights};
use survival::regression::TieMethod;
use survival::residuals::agmart;
use survival::scoring::{agscore3, coxscore2};

fn right() -> SurvivalData {
    SurvivalData::try_new(vec![1.0, 2.0, 3.0], vec![1, 0, 1]).unwrap()
}

fn counting() -> CountingProcessData {
    CountingProcessData::try_new(vec![0.0, 0.0, 1.0], vec![1.0, 2.0, 3.0], vec![1, 0, 1]).unwrap()
}

fn mart() -> AndersenGillInput {
    AndersenGillInput::try_new(
        counting(),
        vec![1.0, 2.0, 0.5],
        Some(Weights::try_new(vec![0.5, 2.0, 1.5]).unwrap()),
        Some(vec![-3, -3, 7]),
    )
    .unwrap()
}

#[test]
fn right_score_and_schoenfeld_reject_mutated_response() {
    let x = array![[1.0], [2.0], [3.0]];
    let check = |data: &SurvivalData| {
        for method in [TieMethod::Breslow, TieMethod::Efron] {
            assert!(coxscore2(data, x.view(), &[1.0; 3], None, None, method).is_err());
            assert!(
                schoenfeld_residuals(
                    SurvResponse::Right(data),
                    x.view(),
                    &[1.0; 3],
                    None,
                    None,
                    method,
                )
                .is_err()
            );
        }
    };
    for length in [0, 1, 2, 4] {
        let mut data = right();
        data.status = vec![1; length];
        check(&data);
        let mut data = right();
        data.time = vec![1.0; length];
        check(&data);
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut data = right();
        data.time[1] = value;
        check(&data);
    }
    for status in [-1, 2] {
        let mut data = right();
        data.status[1] = status;
        check(&data);
    }
}

#[test]
fn counting_score_and_schoenfeld_reject_mutated_response() {
    let x = array![[1.0], [2.0], [3.0]];
    let check = |data: &CountingProcessData| {
        for method in [TieMethod::Breslow, TieMethod::Efron] {
            assert!(agscore3(data, x.view(), &[1.0; 3], None, None, method).is_err());
            assert!(
                schoenfeld_residuals(
                    SurvResponse::Counting(data),
                    x.view(),
                    &[1.0; 3],
                    None,
                    None,
                    method,
                )
                .is_err()
            );
        }
    };
    for length in [0, 1, 2, 4] {
        for field in ["start", "stop", "event"] {
            let mut data = counting();
            match field {
                "start" => data.start = vec![0.0; length],
                "stop" => data.stop = vec![4.0; length],
                "event" => data.event = vec![1; length],
                _ => unreachable!(),
            }
            check(&data);
        }
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut data = counting();
        data.start[1] = value;
        check(&data);
        let mut data = counting();
        data.stop[1] = value;
        check(&data);
    }
    for event in [-1, 2] {
        let mut data = counting();
        data.event[1] = event;
        check(&data);
    }
    for start in [2.0, 3.0] {
        let mut data = counting();
        data.start[1] = start;
        check(&data);
    }
}

#[test]
fn counting_martingale_rejects_mutated_response_and_predictors() {
    for method in [TieMethod::Breslow, TieMethod::Efron, TieMethod::Exact] {
        for length in [0, 1, 2, 4] {
            for field in ["start", "stop", "event", "score", "weights", "strata"] {
                let mut input = mart();
                match field {
                    "start" => input.counting.start = vec![0.0; length],
                    "stop" => input.counting.stop = vec![4.0; length],
                    "event" => input.counting.event = vec![1; length],
                    "score" => input.score = vec![1.0; length],
                    "weights" => input.weights.as_mut().unwrap().values = vec![1.0; length],
                    "strata" => input.strata = Some(vec![0; length]),
                    _ => unreachable!(),
                }
                assert!(agmart(&input, method).is_err(), "{field} length {length}");
            }
        }
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for field in ["start", "stop", "score", "weights"] {
                let mut input = mart();
                match field {
                    "start" => input.counting.start[1] = value,
                    "stop" => input.counting.stop[1] = value,
                    "score" => input.score[1] = value,
                    "weights" => input.weights.as_mut().unwrap().values[1] = value,
                    _ => unreachable!(),
                }
                assert!(agmart(&input, method).is_err(), "{field} {value}");
            }
        }
        for event in [-1, 2] {
            let mut input = mart();
            input.counting.event[1] = event;
            assert!(agmart(&input, method).is_err());
        }
        for start in [2.0, 3.0] {
            let mut input = mart();
            input.counting.start[1] = start;
            assert!(agmart(&input, method).is_err());
        }
        let mut input = mart();
        input.weights.as_mut().unwrap().values[1] = -1.0;
        assert!(agmart(&input, method).is_err());
    }
}

#[derive(Debug)]
struct DirectResiduals {
    martingale: Vec<f64>,
    score: Array2<f64>,
    event_rows: Vec<usize>,
    schoenfeld: Vec<Vec<f64>>,
}

#[allow(clippy::too_many_arguments)]
fn direct_residuals(
    start: Option<&[f64]>,
    stop: &[f64],
    event: &[i32],
    x: ArrayView2<'_, f64>,
    risk: &[f64],
    weight: &[f64],
    strata: &[i32],
    method: TieMethod,
) -> DirectResiduals {
    let mut result = DirectResiduals {
        martingale: event.iter().map(|&status| f64::from(status)).collect(),
        score: Array2::zeros(x.dim()),
        event_rows: Vec::new(),
        schoenfeld: Vec::new(),
    };
    for label in strata.iter().copied().collect::<BTreeSet<_>>() {
        let mut times: Vec<f64> = (0..stop.len())
            .filter(|&row| strata[row] == label && event[row] == 1)
            .map(|row| stop[row])
            .collect();
        times.sort_by(f64::total_cmp);
        times.dedup();
        for time in times {
            let deaths: Vec<usize> = (0..stop.len())
                .filter(|&row| strata[row] == label && event[row] == 1 && stop[row] == time)
                .collect();
            let at_risk: Vec<usize> = (0..stop.len())
                .filter(|&row| {
                    strata[row] == label
                        && stop[row] >= time
                        && start.is_none_or(|start| start[row] < time)
                })
                .collect();
            let denom: f64 = at_risk.iter().map(|&row| risk[row] * weight[row]).sum();
            let death_risk: f64 = deaths.iter().map(|&row| risk[row] * weight[row]).sum();
            let death_weight: f64 = deaths.iter().map(|&row| weight[row]).sum();
            let steps = if method == TieMethod::Efron {
                deaths.len()
            } else {
                1
            };
            let mut mean = vec![0.0; x.ncols()];
            for step in 0..steps {
                let fraction = step as f64 / steps as f64;
                let adjusted_denom = denom - fraction * death_risk;
                let hazard = (death_weight / steps as f64) / adjusted_denom;
                for column in 0..x.ncols() {
                    let sum: f64 = at_risk
                        .iter()
                        .map(|&row| risk[row] * weight[row] * x[[row, column]])
                        .sum();
                    let death_sum: f64 = deaths
                        .iter()
                        .map(|&row| risk[row] * weight[row] * x[[row, column]])
                        .sum();
                    let risk_mean = (sum - fraction * death_sum) / adjusted_denom;
                    mean[column] += risk_mean / steps as f64;
                    for &row in &at_risk {
                        let is_death = deaths.contains(&row);
                        let exposure = if is_death { 1.0 - fraction } else { 1.0 };
                        let increment =
                            f64::from(is_death) / steps as f64 - risk[row] * exposure * hazard;
                        result.score[[row, column]] += (x[[row, column]] - risk_mean) * increment;
                    }
                }
                for &row in &at_risk {
                    let exposure = if deaths.contains(&row) {
                        1.0 - fraction
                    } else {
                        1.0
                    };
                    result.martingale[row] -= risk[row] * exposure * hazard;
                }
            }
            for row in deaths {
                result.event_rows.push(row);
                result.schoenfeld.push(
                    (0..x.ncols())
                        .map(|column| x[[row, column]] - mean[column])
                        .collect(),
                );
            }
        }
    }
    result
}

#[test]
fn weighted_tied_and_delayed_entry_residuals_match_direct_risk_sets() {
    let start = [0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
    let stop = [4.0, 4.0, 5.0, 5.0, 2.0, 6.0, 3.0];
    let event = [1, 1, 0, 1, 1, 0, 0];
    let strata = [9, 9, 9, 9, -2, -2, -2];
    let risk = [1.1, 0.7, 1.4, 1.3, 0.8, 2.0, 0.9];
    let weight = [0.5, 1.25, 2.0, 0.75, 1.5, 0.4, 2.0];
    let x = array![
        [1.0, 5.0],
        [1.0, -2.0],
        [1.0, 1.0],
        [1.0, 3.0],
        [1.0, 2.0],
        [1.0, 4.0],
        [1.0, 6.0]
    ];
    for shift in [0.0, -10.0] {
        let start: Vec<f64> = start.iter().map(|time| time + shift).collect();
        let stop: Vec<f64> = stop.iter().map(|time| time + shift).collect();
        let counting =
            CountingProcessData::try_new(start.clone(), stop.clone(), event.to_vec()).unwrap();
        let input = AndersenGillInput::try_new(
            counting.clone(),
            risk.to_vec(),
            Some(Weights::try_new(weight.to_vec()).unwrap()),
            Some(strata.to_vec()),
        )
        .unwrap();
        let right = SurvivalData::try_new(stop.clone(), event.to_vec()).unwrap();
        for method in [TieMethod::Breslow, TieMethod::Efron] {
            for entry in [None, Some(start.as_slice())] {
                let expected = direct_residuals(
                    entry,
                    &stop,
                    &event,
                    x.view(),
                    &risk,
                    &weight,
                    &strata,
                    method,
                );
                let response = if entry.is_some() {
                    SurvResponse::Counting(&counting)
                } else {
                    SurvResponse::Right(&right)
                };
                let actual = schoenfeld_residuals(
                    response,
                    x.view(),
                    &risk,
                    Some(&weight),
                    Some(&strata),
                    method,
                )
                .unwrap();
                assert_eq!(actual.index, expected.event_rows);
                for (actual, expected) in actual
                    .residuals
                    .iter()
                    .flatten()
                    .zip(expected.schoenfeld.iter().flatten())
                {
                    assert!(
                        (actual - expected).abs() < 1e-12,
                        "Schoenfeld {method:?}: {actual} != {expected}"
                    );
                }
                let scores = if entry.is_some() {
                    agscore3(
                        &counting,
                        x.view(),
                        &risk,
                        Some(&weight),
                        Some(&strata),
                        method,
                    )
                } else {
                    coxscore2(
                        &right,
                        x.view(),
                        &risk,
                        Some(&weight),
                        Some(&strata),
                        method,
                    )
                }
                .unwrap();
                for (actual, expected) in scores.iter().zip(&expected.score) {
                    assert!(
                        (actual - expected).abs() < 1e-12,
                        "score {method:?}: {actual} != {expected}"
                    );
                }
                if entry.is_some() {
                    for (actual, expected) in agmart(&input, method)
                        .unwrap()
                        .iter()
                        .zip(&expected.martingale)
                    {
                        assert!(
                            (actual - expected).abs() < 1e-12,
                            "martingale {method:?}: {actual} != {expected}"
                        );
                    }
                }
            }
        }
        let expected = direct_residuals(
            Some(&start),
            &stop,
            &event,
            x.view(),
            &risk,
            &weight,
            &strata,
            TieMethod::Breslow,
        );
        for (actual, expected) in agmart(&input, TieMethod::Exact)
            .unwrap()
            .iter()
            .zip(&expected.martingale)
        {
            assert!((actual - expected).abs() < 1e-12);
        }
    }
}
