//! Public mutable typed inputs are revalidated before Cox kernel indexing.

use survival::core::{coxcount1, coxcount2};
use survival::prelude::{CountingProcessData, CoxMartInput, SurvivalData, Weights};
use survival::regression::TieMethod;
use survival::residuals::coxmart;

fn right() -> SurvivalData {
    SurvivalData::try_new(vec![1.0, 2.0, 3.0], vec![1, 0, 1]).unwrap()
}

fn counting() -> CountingProcessData {
    CountingProcessData::try_new(vec![0.0, 0.0, 1.0], vec![1.0, 2.0, 3.0], vec![1, 0, 1]).unwrap()
}

fn mart() -> CoxMartInput {
    CoxMartInput::try_new(
        right(),
        vec![1.0, 2.0, 0.5],
        Some(Weights::try_new(vec![0.5, 2.0, 1.5]).unwrap()),
        Some(vec![-3, -3, 7]),
    )
    .unwrap()
}

#[test]
fn right_risk_sets_reject_mutated_response_vectors() {
    for length in [0, 1, 2, 4] {
        let mut data = right();
        data.status = vec![1; length];
        assert!(coxcount1(&data, None).is_err(), "status length {length}");
        let mut data = right();
        data.time = vec![1.0; length];
        assert!(coxcount1(&data, None).is_err(), "time length {length}");
        assert!(coxcount1(&right(), Some(&vec![0; length])).is_err());
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut data = right();
        data.time[1] = value;
        assert!(coxcount1(&data, None).is_err(), "time {value}");
    }
    for status in [-1, 2] {
        let mut data = right();
        data.status[1] = status;
        assert!(coxcount1(&data, None).is_err(), "status {status}");
    }
}

#[test]
fn counting_risk_sets_reject_mutated_response_vectors() {
    for length in [0, 1, 2, 4] {
        for field in ["start", "stop", "event"] {
            let mut data = counting();
            match field {
                "start" => data.start = vec![0.0; length],
                "stop" => data.stop = vec![4.0; length],
                "event" => data.event = vec![1; length],
                _ => unreachable!(),
            }
            assert!(coxcount2(&data, None).is_err(), "{field} length {length}");
        }
        assert!(coxcount2(&counting(), Some(&vec![0; length])).is_err());
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for field in ["start", "stop"] {
            let mut data = counting();
            match field {
                "start" => data.start[1] = value,
                "stop" => data.stop[1] = value,
                _ => unreachable!(),
            }
            assert!(coxcount2(&data, None).is_err(), "{field} {value}");
        }
    }
    for event in [-1, 2] {
        let mut data = counting();
        data.event[1] = event;
        assert!(coxcount2(&data, None).is_err(), "event {event}");
    }
    for start in [2.0, 3.0] {
        let mut data = counting();
        data.start[1] = start;
        assert!(coxcount2(&data, None).is_err(), "start {start}");
    }
}

#[test]
fn martingale_residuals_reject_mutated_response_and_predictor_vectors() {
    for method in [TieMethod::Breslow, TieMethod::Efron, TieMethod::Exact] {
        for length in [0, 1, 2, 4] {
            for field in ["time", "status", "score", "weights", "strata"] {
                let mut input = mart();
                match field {
                    "time" => input.survival.time = vec![1.0; length],
                    "status" => input.survival.status = vec![1; length],
                    "score" => input.score = vec![1.0; length],
                    "weights" => input.weights.as_mut().unwrap().values = vec![1.0; length],
                    "strata" => input.strata = Some(vec![0; length]),
                    _ => unreachable!(),
                }
                assert!(coxmart(&input, method).is_err(), "{field} length {length}");
            }
        }
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for field in ["time", "score", "weights"] {
                let mut input = mart();
                match field {
                    "time" => input.survival.time[1] = value,
                    "score" => input.score[1] = value,
                    "weights" => input.weights.as_mut().unwrap().values[1] = value,
                    _ => unreachable!(),
                }
                assert!(coxmart(&input, method).is_err(), "{field} {value}");
            }
        }
        for status in [-1, 2] {
            let mut input = mart();
            input.survival.status[1] = status;
            assert!(coxmart(&input, method).is_err(), "status {status}");
        }
        let mut input = mart();
        input.weights.as_mut().unwrap().values[1] = -1.0;
        assert!(coxmart(&input, method).is_err());
    }
}

#[test]
fn valid_risk_sets_and_weighted_martingale_residuals_keep_input_order() {
    let expanded = coxcount1(&right(), None).unwrap();
    assert_eq!(expanded.time, [3.0, 1.0]);
    assert_eq!(expanded.nrisk, [1, 3]);
    assert_eq!(expanded.index, [2, 2, 1, 0]);
    assert_eq!(expanded.status, [1, 0, 0, 1]);
    let expanded = coxcount2(&counting(), None).unwrap();
    assert_eq!(expanded.time, [3.0, 1.0]);
    assert_eq!(expanded.nrisk, [1, 2]);
    assert_eq!(expanded.index, [2, 1, 0]);
    assert_eq!(expanded.status, [1, 0, 1]);

    // Independent two-stratum hand calculation: the first event's weighted
    // hazard is .5 / (.5 * 1 + 2 * 2) = 1/9; the second stratum has one death.
    for method in [TieMethod::Breslow, TieMethod::Efron, TieMethod::Exact] {
        let residuals = coxmart(&mart(), method).unwrap();
        for (actual, expected) in residuals.into_iter().zip([8.0 / 9.0, -2.0 / 9.0, 0.0]) {
            assert!((actual - expected).abs() < 1e-12);
        }
        let mut shuffled = mart();
        shuffled.survival.time.swap(0, 2);
        shuffled.survival.status.swap(0, 2);
        shuffled.score.swap(0, 2);
        shuffled.weights.as_mut().unwrap().values.swap(0, 2);
        shuffled.strata.as_mut().unwrap().swap(0, 2);
        let residuals = coxmart(&shuffled, method).unwrap();
        for (actual, expected) in residuals.into_iter().zip([0.0, -2.0 / 9.0, 8.0 / 9.0]) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }
}
