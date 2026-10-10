//! Public Cox curve components remain checked after Rust callers mutate them.
//! Numerical controls rebuild weighted risk sets directly at each stop time.

use ndarray::{Array2, array};
use survival::surv_analysis::{
    AgsurvCurve, AgsurvData, CoxSurvType, IndividualInterval, agsurv, expand_curve,
    individual_curve,
};

const START: [f64; 6] = [0.0, 1.0, 0.0, 2.0, 0.0, 3.0];
const STOP: [f64; 6] = [4.0, 4.0, 5.0, 5.0, 2.0, 6.0];
const STATUS: [i32; 6] = [1, 1, 0, 1, 1, 0];
const WEIGHT: [f64; 6] = [0.5, 1.25, 2.0, 0.75, 1.5, 0.4];
const RISK: [f64; 6] = [1.1, 0.7, 1.4, 1.3, 0.8, 2.0];
const MEANS: [f64; 2] = [0.25, -0.5];

fn design() -> Array2<f64> {
    array![
        [1.0, 5.0],
        [1.0, -2.0],
        [1.0, 1.0],
        [1.0, 3.0],
        [1.0, 2.0],
        [1.0, 4.0]
    ]
}

fn baseline(kind: CoxSurvType, risk_scale: f64) -> AgsurvCurve {
    let x = design();
    let risk: Vec<f64> = RISK.iter().map(|value| value * risk_scale).collect();
    agsurv(
        &AgsurvData {
            start: Some(&START),
            stop: &STOP,
            status: &STATUS,
            x: x.view(),
            means: Some(&MEANS),
            weights: &WEIGHT,
            risk: &risk,
        },
        kind,
        kind,
    )
    .unwrap()
}

#[test]
fn expansions_reject_mutated_baseline_shapes_and_time_grids() {
    let x2 = array![[0.7, -0.25]];
    let v = array![[0.2, 0.03], [0.03, 0.1]];
    let intervals = [IndividualInterval {
        start: 0.0,
        stop: 10.0,
        stratum: 0,
        x2: &[0.7, -0.25],
        risk2: 1.0,
    }];
    for kind in [
        CoxSurvType::Breslow,
        CoxSurvType::Efron,
        CoxSurvType::KalbfleischPrentice,
    ] {
        let original = baseline(kind, 1.0);
        let check = |curve: AgsurvCurve| {
            for varmat in [None, Some(&v)] {
                assert!(expand_curve(&curve, kind, x2.view(), &[1.0], varmat).is_err());
                assert!(
                    individual_curve(std::slice::from_ref(&curve), kind, &intervals, varmat)
                        .is_err()
                );
            }
        };
        for length in [0, 1, original.time.len() - 1, original.time.len() + 1] {
            for field in [
                "time", "n_event", "n_risk", "n_censor", "hazard", "cumhaz", "varhaz", "ndeath",
                "xbar", "surv",
            ] {
                let mut curve = original.clone();
                match field {
                    "time" => curve.time = vec![1.0; length],
                    "n_event" => curve.n_event = vec![1.0; length],
                    "n_risk" => curve.n_risk = vec![1.0; length],
                    "n_censor" => curve.n_censor = vec![0.0; length],
                    "hazard" => curve.hazard = vec![0.1; length],
                    "cumhaz" => curve.cumhaz = vec![0.1; length],
                    "varhaz" => curve.varhaz = vec![0.01; length],
                    "ndeath" => curve.ndeath = vec![1; length],
                    "xbar" => curve.xbar = Array2::zeros((length, 2)),
                    "surv" => curve.surv = Some(vec![0.9; length]),
                    _ => unreachable!(),
                }
                check(curve);
            }
        }
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut curve = original.clone();
            curve.time[0] = value;
            check(curve);
        }
        let mut curve = original.clone();
        curve.time.swap(0, 3);
        check(curve);
    }
}

#[test]
fn standalone_baselines_and_predictions_reject_invalid_values() {
    let x = design();
    let data = AgsurvData {
        start: Some(&START),
        stop: &STOP,
        status: &STATUS,
        x: x.view(),
        means: Some(&MEANS),
        weights: &WEIGHT,
        risk: &RISK,
    };
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut stop = STOP;
        stop[1] = value;
        let malformed = AgsurvData {
            stop: &stop,
            ..data
        };
        assert!(agsurv(&malformed, CoxSurvType::Breslow, CoxSurvType::Breslow).is_err());
        let mut start = START;
        start[1] = value;
        let malformed = AgsurvData {
            start: Some(&start),
            ..data
        };
        assert!(agsurv(&malformed, CoxSurvType::Breslow, CoxSurvType::Breslow).is_err());
        let mut weight = WEIGHT;
        weight[1] = value;
        let malformed = AgsurvData {
            weights: &weight,
            ..data
        };
        assert!(agsurv(&malformed, CoxSurvType::Breslow, CoxSurvType::Breslow).is_err());
        let mut risk = RISK;
        risk[1] = value;
        let malformed = AgsurvData {
            risk: &risk,
            ..data
        };
        assert!(agsurv(&malformed, CoxSurvType::Breslow, CoxSurvType::Breslow).is_err());
        let means = [0.0, value];
        let malformed = AgsurvData {
            means: Some(&means),
            ..data
        };
        assert!(agsurv(&malformed, CoxSurvType::Breslow, CoxSurvType::Breslow).is_err());
        let mut bad_x = x.clone();
        bad_x[[1, 1]] = value;
        let malformed = AgsurvData {
            x: bad_x.view(),
            ..data
        };
        assert!(agsurv(&malformed, CoxSurvType::Breslow, CoxSurvType::Breslow).is_err());
    }
    for value in [-1, 2] {
        let mut status = STATUS;
        status[1] = value;
        let malformed = AgsurvData {
            status: &status,
            ..data
        };
        assert!(agsurv(&malformed, CoxSurvType::Breslow, CoxSurvType::Breslow).is_err());
    }
    for value in [STOP[1], STOP[1] + 1.0] {
        let mut start = START;
        start[1] = value;
        let malformed = AgsurvData {
            start: Some(&start),
            ..data
        };
        assert!(agsurv(&malformed, CoxSurvType::Breslow, CoxSurvType::Breslow).is_err());
    }
    let negative = [-1.0; 6];
    let malformed = AgsurvData {
        weights: &negative,
        ..data
    };
    assert!(agsurv(&malformed, CoxSurvType::Breslow, CoxSurvType::Breslow).is_err());
    for value in [0.0, -1.0] {
        let risk = [value; 6];
        let malformed = AgsurvData {
            risk: &risk,
            ..data
        };
        assert!(agsurv(&malformed, CoxSurvType::Breslow, CoxSurvType::Breslow).is_err());
    }
    let curve = baseline(CoxSurvType::Breslow, 1.0);
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
        assert!(
            expand_curve(
                &curve,
                CoxSurvType::Breslow,
                array![[0.0, 0.0]].view(),
                &[value],
                None
            )
            .is_err()
        );
        let intervals = [IndividualInterval {
            start: 0.0,
            stop: 6.0,
            stratum: 0,
            x2: &[0.0, 0.0],
            risk2: value,
        }];
        assert!(
            individual_curve(
                std::slice::from_ref(&curve),
                CoxSurvType::Breslow,
                &intervals,
                None
            )
            .is_err()
        );
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            expand_curve(
                &curve,
                CoxSurvType::Breslow,
                array![[0.0, value]].view(),
                &[1.0],
                None
            )
            .is_err()
        );
        let v = array![[0.2, value], [0.0, 0.1]];
        assert!(
            expand_curve(
                &curve,
                CoxSurvType::Breslow,
                array![[0.0, 0.0]].view(),
                &[1.0],
                Some(&v)
            )
            .is_err()
        );
        for field in ["start", "stop", "x2"] {
            let covariates = [0.0, value];
            let mut interval = IndividualInterval {
                start: 0.0,
                stop: 6.0,
                stratum: 0,
                x2: &[0.0, 0.0],
                risk2: 1.0,
            };
            match field {
                "start" => interval.start = value,
                "stop" => interval.stop = value,
                "x2" => interval.x2 = &covariates,
                _ => unreachable!(),
            }
            assert!(
                individual_curve(
                    std::slice::from_ref(&curve),
                    CoxSurvType::Breslow,
                    &[interval],
                    None
                )
                .is_err()
            );
        }
    }
    for stop in [-1.0, -2.0] {
        let interval = IndividualInterval {
            start: 0.0,
            stop,
            stratum: 0,
            x2: &[0.0, 0.0],
            risk2: 1.0,
        };
        assert!(
            individual_curve(
                std::slice::from_ref(&curve),
                CoxSurvType::Breslow,
                &[interval],
                None
            )
            .is_err()
        );
    }
}

struct DirectBaseline {
    time: Vec<f64>,
    risk_counts: Vec<f64>,
    events: Vec<f64>,
    censor: Vec<f64>,
    hazard: Vec<f64>,
    variance: Vec<f64>,
    xbar: Vec<[f64; 2]>,
}

fn direct_baseline(kind: CoxSurvType, risk_scale: f64) -> DirectBaseline {
    let x = design();
    let mut times = STOP.to_vec();
    times.sort_by(f64::total_cmp);
    times.dedup();
    let mut output = DirectBaseline {
        time: times,
        risk_counts: vec![],
        events: vec![],
        censor: vec![],
        hazard: vec![],
        variance: vec![],
        xbar: vec![],
    };
    for &time in &output.time {
        let at_risk: Vec<usize> = (0..STOP.len())
            .filter(|&row| START[row] < time && STOP[row] >= time)
            .collect();
        let deaths: Vec<usize> = (0..STOP.len())
            .filter(|&row| STOP[row] == time && STATUS[row] == 1)
            .collect();
        let denom: f64 = at_risk
            .iter()
            .map(|&row| WEIGHT[row] * RISK[row] * risk_scale)
            .sum();
        let death_risk: f64 = deaths
            .iter()
            .map(|&row| WEIGHT[row] * RISK[row] * risk_scale)
            .sum();
        let event_weight: f64 = deaths.iter().map(|&row| WEIGHT[row]).sum();
        let steps = if kind == CoxSurvType::Efron {
            deaths.len().max(1)
        } else {
            1
        };
        let mut hazard = 0.0;
        let mut variance = 0.0;
        let mut xbar = [0.0; 2];
        for step in 0..steps {
            let fraction = step as f64 / steps as f64;
            let adjusted = denom - fraction * death_risk;
            let increment = event_weight / steps as f64 / adjusted;
            hazard += increment;
            variance += event_weight / steps as f64 / (adjusted * adjusted);
            for (column, value) in xbar.iter_mut().enumerate() {
                let sum: f64 = at_risk
                    .iter()
                    .map(|&row| {
                        WEIGHT[row] * RISK[row] * risk_scale * (x[[row, column]] - MEANS[column])
                    })
                    .sum();
                let death_sum: f64 = deaths
                    .iter()
                    .map(|&row| {
                        WEIGHT[row] * RISK[row] * risk_scale * (x[[row, column]] - MEANS[column])
                    })
                    .sum();
                *value += increment * (sum - fraction * death_sum) / adjusted;
            }
        }
        output
            .risk_counts
            .push(at_risk.iter().map(|&row| WEIGHT[row]).sum());
        output.events.push(event_weight);
        output.censor.push(
            (0..STOP.len())
                .filter(|&row| STOP[row] == time && STATUS[row] == 0)
                .map(|row| WEIGHT[row])
                .sum(),
        );
        output.hazard.push(hazard);
        output.variance.push(variance);
        output.xbar.push(xbar);
    }
    output
}

fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
}

#[test]
fn weighted_tied_counting_baselines_and_scaled_curves_match_direct_sums() {
    let x2 = array![[0.7, -0.25], [-0.1, 1.0], [0.0, 0.0]];
    let prediction_risk = [0.8, 1.3, 0.0];
    let v = array![[0.2, 0.03], [0.03, 0.1]];
    for kind in [CoxSurvType::Breslow, CoxSurvType::Efron] {
        let curve = baseline(kind, 1.0);
        let expected = direct_baseline(kind, 1.0);
        assert_eq!(curve.time, expected.time);
        for row in 0..curve.time.len() {
            close(curve.n_risk[row], expected.risk_counts[row]);
            close(curve.n_event[row], expected.events[row]);
            close(curve.n_censor[row], expected.censor[row]);
            close(curve.hazard[row], expected.hazard[row]);
            close(curve.varhaz[row], expected.variance[row]);
            for column in 0..2 {
                close(curve.xbar[[row, column]], expected.xbar[row][column]);
            }
        }
        let expanded = expand_curve(&curve, kind, x2.view(), &prediction_risk, Some(&v)).unwrap();
        for prediction in 0..x2.nrows() {
            let mut hazard = 0.0;
            let mut variance = 0.0;
            let mut dt = [0.0; 2];
            for row in 0..curve.time.len() {
                hazard += expected.hazard[row];
                variance += expected.variance[row];
                for (column, value) in dt.iter_mut().enumerate() {
                    *value += expected.hazard[row] * x2[[prediction, column]]
                        - expected.xbar[row][column];
                }
                let coefficient_variance =
                    0.2 * dt[0] * dt[0] + 0.06 * dt[0] * dt[1] + 0.1 * dt[1] * dt[1];
                close(
                    expanded.cumhaz[[row, prediction]],
                    hazard * prediction_risk[prediction],
                );
                close(
                    expanded.surv[[row, prediction]],
                    (-hazard).exp().powf(prediction_risk[prediction]),
                );
                close(
                    expanded.std_err.as_ref().unwrap()[[row, prediction]],
                    (variance + coefficient_variance).sqrt() * prediction_risk[prediction],
                );
            }
        }
        let curves = [curve, baseline(kind, 2.0)];
        let intervals = [
            IndividualInterval {
                start: 0.0,
                stop: 4.0,
                stratum: 0,
                x2: &[0.7, -0.25],
                risk2: 0.8,
            },
            IndividualInterval {
                start: 4.0,
                stop: 6.0,
                stratum: 1,
                x2: &[-0.1, 1.0],
                risk2: 1.3,
            },
        ];
        let path = individual_curve(&curves, kind, &intervals, Some(&v)).unwrap();
        let mut hazard = 0.0;
        let mut variance = 0.0;
        let mut dt = [0.0; 2];
        let mut row = 0;
        for interval in &intervals {
            let expected = direct_baseline(kind, if interval.stratum == 0 { 1.0 } else { 2.0 });
            for index in 0..expected.time.len() {
                if expected.time[index] <= interval.start || expected.time[index] > interval.stop {
                    continue;
                }
                hazard += expected.hazard[index] * interval.risk2;
                variance += expected.variance[index] * interval.risk2 * interval.risk2;
                for (column, value) in dt.iter_mut().enumerate() {
                    *value += (expected.hazard[index] * interval.x2[column]
                        - expected.xbar[index][column])
                        * interval.risk2;
                }
                let coefficient_variance =
                    0.2 * dt[0] * dt[0] + 0.06 * dt[0] * dt[1] + 0.1 * dt[1] * dt[1];
                close(path.time[row], expected.time[index]);
                close(path.cumhaz[[row, 0]], hazard);
                close(path.surv[[row, 0]], (-hazard).exp());
                close(
                    path.std_err.as_ref().unwrap()[[row, 0]],
                    (variance + coefficient_variance).sqrt(),
                );
                row += 1;
            }
        }
        assert_eq!(row, path.time.len());
    }
}

#[test]
fn unit_risk_kp_curve_has_hand_calculated_increments() {
    let x = array![[0.0], [0.0], [0.0], [0.0]];
    let curve = agsurv(
        &AgsurvData {
            start: None,
            stop: &[2.0, 2.0, 4.0, 3.0],
            status: &[1, 1, 1, 0],
            x: x.view(),
            means: None,
            weights: &[0.5, 1.5, 2.0, 1.0],
            risk: &[1.0; 4],
        },
        CoxSurvType::KalbfleischPrentice,
        CoxSurvType::KalbfleischPrentice,
    )
    .unwrap();
    // All risks equal one, so KP's estimating equation gives 1 - d/Y,
    // including weighted tied deaths: .6, 1, 0 at times 2, 3, 4.
    let expanded = expand_curve(
        &curve,
        CoxSurvType::KalbfleischPrentice,
        array![[0.0], [0.0]].view(),
        &[1.0, 2.0],
        None,
    )
    .unwrap();
    // The tied KP estimate is found by 35 bisection steps. Compare the exact
    // equation's solution at a tolerance above that solver's final bracket.
    for (row, column, expected) in [(0, 0, 0.6), (1, 0, 0.6), (0, 1, 0.36), (1, 1, 0.36)] {
        assert!((expanded.surv[[row, column]] - expected).abs() < 1e-10);
    }
    close(expanded.surv[[2, 0]], 0.0);
    close(expanded.surv[[2, 1]], 0.0);
    let mut empty = curve;
    empty.n = 0;
    empty.time.clear();
    empty.n_event.clear();
    empty.n_risk.clear();
    empty.n_censor.clear();
    empty.hazard.clear();
    empty.cumhaz.clear();
    empty.varhaz.clear();
    empty.ndeath.clear();
    empty.xbar = Array2::zeros((0, 1));
    empty.surv = Some(vec![]);
    let expanded = expand_curve(
        &empty,
        CoxSurvType::KalbfleischPrentice,
        array![[0.0]].view(),
        &[1.0],
        Some(&array![[0.2]]),
    )
    .unwrap();
    assert_eq!(expanded.surv.dim(), (0, 1));
}
