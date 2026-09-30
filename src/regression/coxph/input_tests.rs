use super::*;

fn training_data() -> CoxphData {
    CoxphData::try_new(
        vec![1.0, 2.0, 3.0, 4.0],
        None,
        vec![1, 0, 1, 0],
        ndarray::array![[0.2], [0.4], [0.1], [0.7]],
        None,
        None,
        None,
    )
    .unwrap()
}

#[test]
fn public_newdata_offset_is_checked_by_predictions() {
    let fit = CoxPHFit::fit(training_data(), CoxphOptions::default()).unwrap();
    let new = CoxNewData {
        x: ndarray::array![[0.3], [0.5]],
        strata: None,
        offset: Some(vec![0.0]),
        time: Some(vec![2.0, 3.0]),
        entry: None,
    };
    assert!(fit.predict_expected(Some(&new), false).is_err());
}

#[test]
fn public_no_event_fit_data_is_checked_before_sorting() {
    let mut data = training_data();
    data.status.fill(0);
    data.strata = Some(vec![0]);
    assert!(CoxPHFit::fit(data, CoxphOptions::default()).is_err());
}

fn new_data() -> CoxNewData {
    CoxNewData {
        x: ndarray::array![[0.3], [0.5]],
        strata: Some(vec![0, 0]),
        offset: Some(vec![0.1, -0.2]),
        time: Some(vec![2.0, 3.0]),
        entry: Some(vec![0.0, 1.0]),
    }
}

fn prediction_results(fit: &CoxPHFit, new: &CoxNewData) -> Vec<(&'static str, SurvivalResult<()>)> {
    vec![
        (
            "lp",
            fit.predict_lp(Some(new), true, PredictReference::Sample)
                .map(|_| ()),
        ),
        (
            "risk",
            fit.predict_risk(Some(new), true, PredictReference::Zero)
                .map(|_| ()),
        ),
        (
            "terms",
            fit.predict_terms(Some(new), true, PredictReference::Sample, &[vec![0]])
                .map(|_| ()),
        ),
        (
            "expected",
            fit.predict_expected(Some(new), true).map(|_| ()),
        ),
        (
            "survival",
            fit.predict_survival(Some(new), true).map(|_| ()),
        ),
        (
            "curves",
            fit.survfit(Some(new), SurvfitOptions::default())
                .map(|_| ()),
        ),
        (
            "requested times",
            fit.predict_survival_at(&[1.0, 2.0], Some(new)).map(|_| ()),
        ),
        (
            "empty requested times",
            fit.predict_survival_at(&[], Some(new)).map(|_| ()),
        ),
        (
            "individual",
            fit.survfit_individual(new, &[1, 2], SurvfitOptions::default())
                .map(|_| ()),
        ),
        (
            "cohort",
            fit.expected_survival(new, &[0, 0], &[1.0, 1.0], None, Some(&[1.0, 2.0]), "ederer")
                .map(|_| ()),
        ),
    ]
}

fn malformed_new_data() -> Vec<(&'static str, CoxNewData)> {
    let mut cases = vec![
        (
            "empty rows",
            CoxNewData {
                x: Array2::zeros((0, 1)),
                ..new_data()
            },
        ),
        (
            "wrong columns",
            CoxNewData {
                x: Array2::zeros((2, 2)),
                ..new_data()
            },
        ),
        (
            "unknown stratum",
            CoxNewData {
                strata: Some(vec![0, 19]),
                ..new_data()
            },
        ),
    ];
    for length in [0, 1, 3] {
        cases.extend([
            (
                "strata length",
                CoxNewData {
                    strata: Some(vec![0; length]),
                    ..new_data()
                },
            ),
            (
                "offset length",
                CoxNewData {
                    offset: Some(vec![0.0; length]),
                    ..new_data()
                },
            ),
            (
                "time length",
                CoxNewData {
                    time: Some(vec![3.0; length]),
                    ..new_data()
                },
            ),
            (
                "entry length",
                CoxNewData {
                    entry: Some(vec![0.0; length]),
                    ..new_data()
                },
            ),
        ]);
    }
    for value in [f64::NEG_INFINITY, f64::INFINITY] {
        let mut new = new_data();
        new.x[(1, 0)] = value;
        cases.push(("infinite covariate", new));
        cases.push((
            "infinite offset",
            CoxNewData {
                offset: Some(vec![0.0, value]),
                ..new_data()
            },
        ));
    }
    for value in [f64::NEG_INFINITY, f64::INFINITY, f64::NAN] {
        cases.push((
            "nonfinite time",
            CoxNewData {
                time: Some(vec![2.0, value]),
                ..new_data()
            },
        ));
        cases.push((
            "nonfinite entry",
            CoxNewData {
                entry: Some(vec![0.0, value]),
                ..new_data()
            },
        ));
    }
    for start in [3.0, 4.0] {
        cases.push((
            "invalid interval",
            CoxNewData {
                entry: Some(vec![0.0, start]),
                ..new_data()
            },
        ));
    }
    cases
}

#[test]
fn every_prediction_consumer_revalidates_public_fields() {
    let fit = CoxPHFit::fit(training_data(), CoxphOptions::default()).unwrap();
    for (case, new) in malformed_new_data() {
        for (method, result) in prediction_results(&fit, &new) {
            assert!(result.is_err(), "{case} was accepted by {method}");
        }
    }
}

#[test]
fn permissive_prediction_inputs_cannot_bypass_strict_curve_checks() {
    let fit = CoxPHFit::fit(training_data(), CoxphOptions::default()).unwrap();
    for offset_missing in [false, true] {
        let mut new = new_data();
        if offset_missing {
            new.offset.as_mut().unwrap()[1] = f64::NAN;
        } else {
            new.x[(1, 0)] = f64::NAN;
        }
        // Partial predictions keep a valid row and propagate only the missing row.
        let expected = fit.predict_expected(Some(&new), true).unwrap();
        assert!(expected.fit[0].is_finite());
        assert!(expected.fit[1].is_nan());
        assert!(
            fit.predict_lp(Some(&new), true, PredictReference::Zero)
                .is_ok()
        );
        assert!(
            fit.predict_risk(Some(&new), false, PredictReference::Zero)
                .is_ok()
        );
        assert!(
            fit.predict_terms(Some(&new), true, PredictReference::Zero, &[vec![0]])
                .is_ok()
        );
        assert!(fit.predict_survival(Some(&new), true).is_ok());
        assert!(fit.survfit(Some(&new), SurvfitOptions::default()).is_err());
        assert!(
            fit.survfit_individual(&new, &[1, 2], SurvfitOptions::default())
                .is_err()
        );
        assert!(fit.predict_survival_at(&[], Some(&new)).is_err());
        assert!(
            fit.expected_survival(&new, &[0, 0], &[1.0, 1.0], None, None, "ederer")
                .is_err()
        );
    }
}

#[test]
fn newdata_constructors_check_lengths_and_intervals() {
    for (case, new) in malformed_new_data() {
        // Column counts and stratum membership belong to the model.
        if ["wrong columns", "unknown stratum"].contains(&case) {
            continue;
        }
        for missing in [false, true] {
            let result = CoxNewData::validated(
                new.x.clone(),
                new.strata.clone(),
                new.offset.clone(),
                new.time.clone(),
                new.entry.clone(),
                missing,
            );
            assert!(result.is_err(), "constructor accepted {case}");
        }
    }
}

fn penalized_data(cox: CoxphData) -> crate::regression::CoxpenalData {
    use crate::regression::{CoxpenalData, ModelTerm, PenaltyTerm};
    CoxpenalData {
        cox,
        terms: vec![ModelTerm {
            columns: vec![0],
            penalty: Some(PenaltyTerm::ridge(Some(1.0), None, 0.1, false, None).unwrap()),
        }],
    }
}

#[test]
fn penalized_curves_revalidate_newdata() {
    use crate::regression::{CoxpenalFit, CoxpenalOptions};
    let fit =
        CoxpenalFit::fit(penalized_data(training_data()), CoxpenalOptions::default()).unwrap();
    for (case, new) in malformed_new_data() {
        assert!(
            fit.survfit(Some(&new), SurvfitOptions::default()).is_err(),
            "{case}"
        );
        assert!(
            fit.survfit_individual(&new, &[1, 2], SurvfitOptions::default())
                .is_err(),
            "{case}"
        );
        assert!(
            fit.expected_survival(&new, &[0, 0], &[1.0, 1.0], None, None, "ederer")
                .is_err(),
            "{case}"
        );
    }
}

#[test]
fn fitting_revalidates_public_data_with_and_without_events() {
    use crate::regression::{CoxpenalFit, CoxpenalOptions, CoxphFitResult};
    for no_events in [false, true] {
        let mut valid = training_data();
        if no_events {
            valid.status.fill(0);
        }
        let mut cases = vec![
            CoxphData {
                time: vec![],
                ..valid.clone()
            },
            CoxphData {
                time: vec![1.0, f64::NAN, 3.0, 4.0],
                ..valid.clone()
            },
            CoxphData {
                status: vec![0, 0, 2, 0],
                ..valid.clone()
            },
            CoxphData {
                x: Array2::zeros((3, 1)),
                ..valid.clone()
            },
            CoxphData {
                offset: Some(vec![f64::INFINITY; 4]),
                ..valid.clone()
            },
            CoxphData {
                weights: Some(vec![f64::NAN; 4]),
                ..valid.clone()
            },
            CoxphData {
                entry: Some(vec![1.0; 4]),
                ..valid.clone()
            },
        ];
        for length in [0, 3, 5] {
            cases.extend([
                CoxphData {
                    status: vec![0; length],
                    ..valid.clone()
                },
                CoxphData {
                    strata: Some(vec![0; length]),
                    ..valid.clone()
                },
                CoxphData {
                    offset: Some(vec![0.0; length]),
                    ..valid.clone()
                },
                CoxphData {
                    weights: Some(vec![1.0; length]),
                    ..valid.clone()
                },
                CoxphData {
                    entry: Some(vec![0.0; length]),
                    ..valid.clone()
                },
            ]);
        }
        for data in cases {
            assert!(CoxPHFit::fit(data.clone(), CoxphOptions::default()).is_err());
            assert!(CoxphFitResult::fit(data.clone(), CoxphOptions::default(), false).is_err());
            assert!(CoxpenalFit::fit(penalized_data(data), CoxpenalOptions::default()).is_err());
        }
    }
}

#[test]
fn no_event_fits_keep_r_predictor_and_weight_semantics() {
    let mut data = training_data();
    data.status.fill(0);
    data.x[(1, 0)] = f64::NAN;
    data.weights = Some(vec![0.0; 4]);
    let fit = CoxPHFit::fit(data, CoxphOptions::default()).unwrap();
    assert_eq!(fit.nevent, 0);
    assert_eq!(fit.residuals, vec![0.0; 4]);
}

#[test]
fn penalized_fitting_checks_mutated_term_assignments() {
    use crate::regression::{CoxpenalFit, CoxpenalOptions};
    for columns in [vec![], vec![1], vec![0, 0]] {
        let mut data = penalized_data(training_data());
        data.terms[0].columns = columns;
        assert!(CoxpenalFit::fit(data, CoxpenalOptions::default()).is_err());
    }
}

#[test]
fn ambiguous_strata_are_rejected_for_single_baseline_predictions() {
    let mut data = training_data();
    data.strata = Some(vec![17, -3, 17, -3]);
    let fit = CoxPHFit::fit(data, CoxphOptions::default()).unwrap();
    let new = CoxNewData {
        strata: None,
        ..new_data()
    };
    assert!(fit.predict_expected(Some(&new), false).is_err());
    assert!(fit.predict_survival(Some(&new), false).is_err());
    let first_stratum = CoxNewData {
        strata: Some(vec![-3, -3]),
        ..new.clone()
    };
    assert_eq!(
        fit.survfit_individual(&new, &[1, 2], SurvfitOptions::default())
            .unwrap(),
        fit.survfit_individual(&first_stratum, &[1, 2], SurvfitOptions::default())
            .unwrap(),
    );
    // A full curve request explicitly returns every stratum, so it is unambiguous.
    assert_eq!(
        fit.survfit(Some(&new), SurvfitOptions::default())
            .unwrap()
            .len(),
        2
    );
    assert!(
        fit.predict_lp(Some(&new), false, PredictReference::Sample)
            .is_ok()
    );
}

#[test]
fn streamed_risks_match_full_centering_for_matrix_layouts() {
    for columns in [0, 1, 3, 32] {
        let data = CoxphData::try_new(
            (1..=48).map(f64::from).collect(),
            None,
            (0..48).map(|i| i32::from(i % 3 != 0)).collect(),
            Array2::from_shape_fn((48, columns), |(i, j)| ((i * 17 + j * 29 + 3) as f64).sin()),
            Some(vec![1.0; 48]),
            None,
            Some(vec![0.1; 48]),
        )
        .unwrap();
        let fit = CoxPHFit::fit(data, CoxphOptions::default()).unwrap();
        let x = Array2::from_shape_fn((7, columns), |(i, j)| ((i * 5 + j * 11 + 1) as f64).cos());
        let offsets = [0.0, 0.1, -0.3, 0.7, 0.0, 0.2, -0.8];
        let mut column_major = Array2::zeros((columns, 7));
        column_major.assign(&x.t());
        let column_major = column_major.reversed_axes();
        for matrix in [&x, &column_major] {
            for missing in [false, true] {
                let mut new = matrix.clone();
                if missing && columns > 0 {
                    new[(3, columns - 1)] = f64::NAN;
                }
                let (_, expected) = fit.centered_rows(new.view(), Some(&offsets));
                let actual = fit.prediction_risks(new.view(), Some(&offsets));
                for (a, b) in actual.iter().zip(&expected) {
                    assert!(
                        a == b || (a.is_nan() && b.is_nan()),
                        "{columns}: {a} != {b}"
                    );
                }
            }
        }
    }
}
