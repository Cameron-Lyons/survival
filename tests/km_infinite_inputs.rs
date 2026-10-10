//! Infinite KM endpoints, uncertainty and queries retain independent stock rows.

use serde_json::Value;
use survival::surv_analysis::{
    HazardType, InfluenceRequest, StackedCurves, SurvType, SurvfitKMData, SurvfitKMOptions,
    SurvfitKMResult, quantile_survfit, summary_survfit_times_with_counts, survfit0, survfitkm,
};

fn reference() -> Value {
    serde_json::from_str(include_str!(
        "../python/tests/fixtures/km_infinite_reference.json"
    ))
    .unwrap()
}

fn number(value: &Value) -> f64 {
    match value {
        Value::Null => f64::NAN,
        Value::String(value) => value.parse().unwrap(),
        _ => value.as_f64().unwrap(),
    }
}

fn numbers(value: &Value) -> Vec<f64> {
    value.as_array().unwrap().iter().map(number).collect()
}

fn assert_numbers(actual: &[f64], expected: &Value, name: &str) {
    let expected = numbers(expected);
    assert_eq!(actual.len(), expected.len(), "{name}: length");
    for (i, (&actual, expected)) in actual.iter().zip(expected).enumerate() {
        if expected.is_nan() {
            assert!(actual.is_nan(), "{name}[{i}]: {actual} != NaN");
        } else if expected.is_infinite() {
            assert_eq!(actual, expected, "{name}[{i}]");
        } else {
            assert!(
                (actual - expected).abs() <= 2e-13 + 2e-11 * expected.abs(),
                "{name}[{i}]: {actual} != {expected}"
            );
        }
    }
}

fn fit_case(reference: &Value, case: &Value) -> SurvfitKMResult {
    let input = &reference["inputs"][case["input"].as_str().unwrap()];
    let time = numbers(&input["time"]);
    let n = time.len();
    let data = SurvfitKMData::try_new(
        input.get("start").map(numbers),
        time,
        numbers(&input["status"])
            .into_iter()
            .map(|value| value as i32)
            .collect(),
        (!case["weights"].is_null()).then(|| numbers(&case["weights"])),
        (!case["group"].is_null()).then(|| {
            case["group"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| i32::from(value == "b"))
                .collect()
        }),
        Some((1..=n as i64).collect()),
        None,
    )
    .unwrap();
    let options = SurvfitKMOptions {
        stype: SurvType::from_code(case["stype"].as_i64().unwrap() as i32).unwrap(),
        ctype: HazardType::from_code(case["ctype"].as_i64().unwrap() as i32).unwrap(),
        influence: if case["weights"].is_null() {
            InfluenceRequest::None
        } else {
            InfluenceRequest::Both
        },
        entry: data.start.is_some(),
        timefix: case["timefix"].as_bool().unwrap(),
        ..Default::default()
    };
    survfitkm(&data, &options).unwrap()
}

fn assert_fit(fit: &SurvfitKMResult, expected: &Value, name: &str) {
    for (field, values) in [
        ("time", &fit.time),
        ("n_risk", &fit.n_risk),
        ("n_event", &fit.n_event),
        ("n_censor", &fit.n_censor),
        ("surv", &fit.surv),
        ("cumhaz", &fit.cumhaz),
    ] {
        assert_numbers(values, &expected[field], &format!("{name}.{field}"));
    }
    for (field, values) in [
        ("n_enter", &fit.n_enter),
        ("std_err", &fit.std_err),
        ("std_chaz", &fit.std_chaz),
        ("lower", &fit.lower),
        ("upper", &fit.upper),
    ] {
        if let Some(values) = values {
            assert_numbers(values, &expected[field], &format!("{name}.{field}"));
        } else {
            assert!(expected[field].is_null(), "{name}.{field}");
        }
    }
    assert_numbers(
        &fit.n.iter().map(|&value| value as f64).collect::<Vec<_>>(),
        &expected["n"],
        name,
    );
    assert_numbers(&[fit.t0], &expected["t0"], name);
    if let Some(values) = &fit.strata {
        assert_numbers(
            &values.iter().map(|&value| value as f64).collect::<Vec<_>>(),
            &expected["strata"],
            name,
        );
    } else {
        assert!(expected["strata"].is_null(), "{name}.strata");
    }
    if let Some(counts) = &fit.counts {
        for (field, values) in [
            ("n_risk", &counts.n_risk),
            ("n_event", &counts.n_event),
            ("n_censor", &counts.n_censor),
        ] {
            assert_numbers(
                values,
                &expected["counts"][field],
                &format!("{name}.counts.{field}"),
            );
        }
        if let Some(values) = &counts.n_enter {
            assert_numbers(values, &expected["counts"]["n_enter"], name);
        }
    } else {
        assert!(expected["counts"].is_null(), "{name}.counts");
    }
    for (field, curves) in [
        ("influence_surv", &fit.influence_surv),
        ("influence_chaz", &fit.influence_chaz),
    ] {
        if let Some(curves) = curves {
            let values: Vec<&Value> = if fit.strata.is_none() {
                vec![&expected[field]]
            } else {
                expected[field].as_array().unwrap().iter().collect()
            };
            assert_eq!(curves.len(), values.len(), "{name}.{field}");
            for (curve, value) in curves.iter().zip(values) {
                let rows = value.as_array().unwrap();
                assert_eq!(curve.values.nrows(), rows.len(), "{name}.{field}");
                for (row, expected) in curve.values.rows().into_iter().zip(rows) {
                    assert_numbers(&row.to_vec(), expected, &format!("{name}.{field}"));
                }
            }
        } else {
            assert!(expected[field].is_null(), "{name}.{field}");
        }
    }
}

#[test]
fn infinite_fits_zero_rows_and_uncertainty_match_independent_stock_r() {
    let reference = reference();
    assert_eq!(reference["metadata"]["survival_version"], "3.8.12");
    for case in reference["cases"].as_array().unwrap() {
        let fit = fit_case(&reference, case);
        let name = case["name"].as_str().unwrap();
        assert_fit(&fit, &case["fit"], name);
        assert_fit(&survfit0(&fit), &case["zero"], name);
    }
}

#[test]
fn infinite_summary_queries_and_curve_quantiles_match_independent_stock_r() {
    let reference = reference();
    for case in reference["cases"].as_array().unwrap() {
        let fit = fit_case(&reference, case);
        let name = case["name"].as_str().unwrap();
        for query in case["summaries"].as_array().unwrap() {
            let actual = summary_survfit_times_with_counts(
                &fit,
                &numbers(&query["times"]),
                query["extend"].as_bool().unwrap(),
                query["dosum"].as_bool(),
            );
            if query["expected"].get("error").is_some() {
                assert!(actual.is_err(), "{name}: expected summary error");
            } else {
                let actual = actual.unwrap();
                let expected = &query["expected"]["value"];
                for (field, values) in [
                    ("time", &actual.time),
                    ("n_risk", &actual.n_risk),
                    ("n_event", &actual.n_event),
                    ("n_censor", &actual.n_censor),
                    ("surv", &actual.surv),
                    ("cumhaz", &actual.cumhaz),
                ] {
                    assert_numbers(values, &expected[field], &format!("{name}.{field}"));
                }
            }
        }
        for query in case["quantiles"].as_array().unwrap() {
            let actual = quantile_survfit(
                &fit,
                &[0.0, 0.25, 0.5, 0.75, 1.0],
                query["confidence"].as_bool().unwrap(),
                number(&query["scale"][0]),
                None,
            )
            .unwrap();
            for (field, values) in [
                ("quantile", Some(&actual.quantile)),
                ("lower", actual.lower.as_ref()),
                ("upper", actual.upper.as_ref()),
            ] {
                if let Some(values) = values {
                    let expected = &query["expected"]["value"][field];
                    let rows: Vec<&Value> = if fit.strata.is_none() {
                        vec![expected]
                    } else {
                        expected.as_array().unwrap().iter().collect()
                    };
                    assert_eq!(values.len(), rows.len(), "{name}.{field}");
                    for (row, expected) in values.iter().zip(rows) {
                        assert_numbers(row, expected, &format!("{name}.{field}"));
                        for (&actual, expected) in row.iter().zip(numbers(expected)) {
                            if expected == 0.0 {
                                assert_eq!(
                                    actual.is_sign_negative(),
                                    expected.is_sign_negative(),
                                    "{name}.{field}: zero sign"
                                );
                            }
                        }
                    }
                } else {
                    assert!(
                        query["expected"]["value"].get(field).is_none(),
                        "{name}.{field}"
                    );
                }
            }
        }
    }
}

#[test]
fn stacked_infinite_curves_keep_their_origin_and_reject_missing_rows() {
    let reference = reference();
    for case in reference["cases"].as_array().unwrap() {
        let fit = fit_case(&reference, case);
        let curves = StackedCurves {
            n: fit.n.clone(),
            time: fit.time.clone(),
            n_risk: fit.n_risk.clone(),
            n_event: fit.n_event.clone(),
            n_censor: Some(fit.n_censor.clone()),
            surv: fit.surv.clone(),
            cumhaz: Some(fit.cumhaz.clone()),
            std_err: fit.std_err.clone(),
            std_chaz: fit.std_chaz.clone(),
            lower: fit.lower.clone(),
            upper: fit.upper.clone(),
            strata: fit.strata.clone(),
            n_id: fit.n_id.clone(),
            logse: fit.logse,
            conf_int: fit.conf_int,
            conf_type: fit.conf_type.clone(),
            type_: fit.type_.clone(),
            t0: fit.t0,
        };
        let stacked = SurvfitKMResult::from_stacked(curves.clone()).unwrap();
        assert_eq!(stacked.time, fit.time);
        assert_eq!(stacked.t0, fit.t0);
        let mut missing = curves.clone();
        missing.time[0] = f64::NAN;
        assert!(SurvfitKMResult::from_stacked(missing).is_err());
        let mut missing = curves;
        missing.t0 = f64::NAN;
        assert!(SurvfitKMResult::from_stacked(missing).is_err());
    }
}

#[test]
fn infinite_inputs_remain_checked_after_public_mutation() {
    let data = SurvfitKMData::try_new(
        Some(vec![f64::NEG_INFINITY, 0.0, 1.0]),
        vec![1.0, 2.0, f64::INFINITY],
        vec![1, 0, 1],
        None,
        None,
        None,
        None,
    )
    .unwrap();
    assert!(survfitkm(&data, &SurvfitKMOptions::default()).is_ok());
    for mutate in [
        (|d: &mut SurvfitKMData| d.time[0] = f64::NAN) as fn(&mut SurvfitKMData),
        |d| d.start.as_mut().unwrap()[0] = f64::NAN,
        |d| d.start.as_mut().unwrap()[0] = f64::INFINITY,
        |d| d.time.pop().map(|_| ()).unwrap(),
        |d| d.status.clear(),
        |d| d.weights = Some(vec![1.0, 1.0, f64::INFINITY]),
    ] {
        let mut mutated = data.clone();
        mutate(&mut mutated);
        assert!(survfitkm(&mutated, &SurvfitKMOptions::default()).is_err());
    }
    let fit = survfitkm(&data, &SurvfitKMOptions::default()).unwrap();
    assert!(
        summary_survfit_times_with_counts(&fit, &[f64::NEG_INFINITY, f64::INFINITY], true, None)
            .is_ok()
    );
    assert!(summary_survfit_times_with_counts(&fit, &[f64::NAN], true, None).is_err());
}

#[test]
fn queried_summaries_reject_origins_after_counting_entry_rows() {
    let data = SurvfitKMData::try_new(
        Some(vec![f64::NEG_INFINITY, 0.0, 1.0, 2.0]),
        vec![1.0, 2.0, 3.0, f64::INFINITY],
        vec![1, 0, 1, 1],
        None,
        None,
        Some(vec![1, 2, 3, 4]),
        None,
    )
    .unwrap();
    for (origin, expected_times) in [
        (2.0, vec![0.0, 1.0, 2.0, 3.0, f64::INFINITY]),
        (3.0, vec![1.0, 2.0, 3.0, f64::INFINITY]),
        (f64::INFINITY, vec![2.0, f64::INFINITY]),
    ] {
        let options = SurvfitKMOptions {
            entry: true,
            start_time: Some(origin),
            ..Default::default()
        };
        let fit = survfitkm(&data, &options).unwrap();
        assert_eq!(fit.time, expected_times);
        assert_eq!(fit.t0, origin);
        // Preserve the public initial-row operation, including R's order.
        let zero = survfit0(&fit);
        assert_eq!(zero.time[0], origin);
        assert_eq!(&zero.time[1..], fit.time.as_slice());
        assert!(zero.time[1] < zero.time[0]);
        let valid = survfitkm(
            &data,
            &SurvfitKMOptions {
                entry: false,
                ..options
            },
        )
        .unwrap();
        assert!(survfit0(&valid).time.is_sorted());
        // Both sparse interval searches and dense sweeps must fail before
        // reading the unsorted curve; the valid no-entry curve still works.
        for queries in [
            vec![0.0, 2.0, f64::INFINITY],
            vec![f64::INFINITY],
            vec![
                f64::NEG_INFINITY,
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                f64::INFINITY,
            ],
        ] {
            for extend in [false, true] {
                for dosum in [None, Some(false), Some(true)] {
                    let error = summary_survfit_times_with_counts(&fit, &queries, extend, dosum)
                        .unwrap_err();
                    assert!(
                        error
                            .to_string()
                            .contains("'vec' must be sorted non-decreasingly and not contain NAs")
                    );
                    assert!(
                        summary_survfit_times_with_counts(&valid, &queries, extend, dosum).is_ok()
                    );
                }
            }
        }
    }
}
