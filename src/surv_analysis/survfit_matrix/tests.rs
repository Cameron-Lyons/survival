use super::*;
use crate::surv_analysis::StackedCurves;
use serde_json::Value;

fn numbers(value: &Value) -> Vec<f64> {
    serde_json::from_value(value.clone()).unwrap()
}

fn close_rows(actual: &[Vec<f64>], expected: &Value) {
    let expected: Vec<Vec<f64>> = serde_json::from_value(expected.clone()).unwrap();
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(&expected) {
        assert_eq!(actual.len(), expected.len());
        for (&a, &b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 2e-12, "{a} != {b}");
        }
    }
}

#[test]
fn matches_r_matrix_products_and_counts() {
    let reference: Value = serde_json::from_str(include_str!(
        "../../../python/tests/fixtures/survfit_matrix_reference.json"
    ))
    .unwrap();
    for case in reference["cases"].as_array().unwrap() {
        let kind = case["kind"].as_str().unwrap();
        let curves: Vec<Vec<_>> = reference["curves"][kind]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                let columns = if kind == "cox" { 2 } else { 1 };
                (0..columns)
                    .map(|column| {
                        let extract = |name: &str| -> Vec<f64> {
                            if kind == "km" {
                                numbers(&c[name])
                            } else {
                                c[name]
                                    .as_array()
                                    .unwrap()
                                    .iter()
                                    .map(|row| row[column].as_f64().unwrap())
                                    .collect()
                            }
                        };
                        let mut data = StackedCurves::new(
                            numbers(&c["time"]),
                            numbers(&c["n_risk"]),
                            numbers(&c["n_event"]),
                            extract("surv"),
                            Some(
                                c["strata"]
                                    .as_object()
                                    .unwrap()
                                    .values()
                                    .map(|n| n.as_u64().unwrap() as usize)
                                    .collect(),
                            ),
                            serde_json::from_value(c["n"].clone()).unwrap(),
                        );
                        data.cumhaz = Some(extract("cumhaz"));
                        SurvfitKMResult::from_stacked(data).unwrap()
                    })
                    .collect()
            })
            .collect();
        let transitions: Vec<_> = [(0, 1), (0, 2), (1, 2)]
            .into_iter()
            .enumerate()
            .map(|(k, (from, to))| SurvfitMatrixTransition {
                from,
                to,
                curves: &curves[k],
            })
            .collect();
        let expected = &case["expected"];
        let p0: Vec<Vec<f64>> = serde_json::from_value(expected["p0"].clone()).unwrap();
        let p0 = Array2::from_shape_vec((p0.len(), 3), p0.into_iter().flatten().collect()).unwrap();
        let fit = survfit_matrix(
            &transitions,
            &["1".into(), "2".into(), "3".into()],
            Some(p0.view()),
            SurvfitMatrixMethod::parse(case["method"].as_str().unwrap()).unwrap(),
            Some(case["start"].as_f64().unwrap()),
        )
        .unwrap();
        assert_eq!(fit.time, numbers(&expected["time"]), "{}", case["name"]);
        close_rows(&fit.pstate, &expected["pstate"]);
        close_rows(&fit.n_risk, &expected["n_risk"]);
        close_rows(&fit.n_event, &expected["n_event"]);
        assert_eq!(fit.n.len(), fit.n_curves());
    }
}

fn curve(hazard: f64) -> Vec<SurvfitKMResult> {
    let mut data = StackedCurves::new(
        vec![1.0, 2.0],
        vec![10.0, 8.0],
        vec![1.0, 1.0],
        vec![(-hazard).exp(), (-2.0 * hazard).exp()],
        None,
        vec![10],
    );
    data.cumhaz = Some(vec![hazard, 2.0 * hazard]);
    vec![SurvfitKMResult::from_stacked(data).unwrap()]
}

#[test]
fn reversible_chain_matches_closed_form_and_preserves_probability() {
    let forward = curve(0.2);
    let backward = curve(0.3);
    let transitions = [
        SurvfitMatrixTransition {
            from: 0,
            to: 1,
            curves: &forward,
        },
        SurvfitMatrixTransition {
            from: 1,
            to: 0,
            curves: &backward,
        },
    ];
    let fit = survfit_matrix(
        &transitions,
        &["A".into(), "B".into()],
        None,
        SurvfitMatrixMethod::MatrixExponential,
        None,
    )
    .unwrap();
    for (t, p) in fit.time.iter().zip(&fit.pstate) {
        let a = 0.6 + 0.4 * (-0.5 * t).exp();
        assert!((p[0] - a).abs() < 1e-14);
        assert!((p[1] - (1.0 - a)).abs() < 1e-14);
    }
}

#[test]
fn malformed_public_curves_are_errors_not_panics() {
    let valid = curve(0.2);
    for failure in 0..5 {
        let mut bad = valid.clone();
        match failure {
            0 => {
                bad[0].n_risk.clear();
            }
            1 => {
                bad[0].time[1] = 0.0;
            }
            2 => {
                bad[0].cumhaz[1] = 0.0;
            }
            3 => {
                bad[0].cumhaz[0] = f64::NAN;
            }
            _ => {
                bad[0].strata = Some(vec![usize::MAX, 3]);
                bad[0].n = vec![10, 10];
            }
        }
        let transitions = [
            SurvfitMatrixTransition {
                from: 0,
                to: 1,
                curves: &bad,
            },
            SurvfitMatrixTransition {
                from: 1,
                to: 0,
                curves: &bad,
            },
        ];
        assert!(
            survfit_matrix(
                &transitions,
                &["A".into(), "B".into()],
                None,
                SurvfitMatrixMethod::Discrete,
                None
            )
            .is_err()
        );
    }
    for (from, to) in [(0, 1), (2, 0), (usize::MAX, 0)] {
        let transitions = [
            SurvfitMatrixTransition {
                from: 0,
                to: 1,
                curves: &valid,
            },
            SurvfitMatrixTransition {
                from,
                to,
                curves: &valid,
            },
        ];
        assert!(
            survfit_matrix(
                &transitions,
                &["A".into(), "B".into()],
                None,
                SurvfitMatrixMethod::Discrete,
                None
            )
            .is_err()
        );
    }
}
