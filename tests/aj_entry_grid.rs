//! Entry reporting grids, counts and estimates match stock R survival 3.8-12.

use serde_json::Value;
use survival::surv_analysis::{SurvfitAJData, SurvfitAJOptions, SurvfitAJResult, survfitaj};

fn assert_point_estimates_equal(actual: &SurvfitAJResult, reference: &SurvfitAJResult) {
    let actual = serde_json::to_value(actual).unwrap();
    let mut expected = serde_json::to_value(reference).unwrap();
    for field in [
        "std_err",
        "std_chaz",
        "std_auc",
        "se0",
        "lower",
        "upper",
        "influence_pstate",
    ] {
        assert_eq!(actual[field], Value::Null, "{field}");
        expected[field] = Value::Null;
    }
    assert_eq!(actual, expected);
}

fn values(values: &[Vec<f64>]) -> Value {
    serde_json::to_value(values).unwrap()
}

fn assert_values(actual: &Value, expected: &Value, path: &str) {
    assert_values_with_tolerance(actual, expected, path, 2e-12, 1e-14);
}

fn assert_values_with_tolerance(
    actual: &Value,
    expected: &Value,
    path: &str,
    rtol: f64,
    atol: f64,
) {
    if let Some(expected) = expected.as_array() {
        let actual = actual.as_array().expect(path);
        assert_eq!(actual.len(), expected.len(), "{path}: shape");
        for (i, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_values_with_tolerance(actual, expected, &format!("{path}[{i}]"), rtol, atol);
        }
    } else if let Some(expected) = expected.as_f64() {
        let actual = actual.as_f64().expect(path);
        assert!(
            (actual - expected).abs() <= atol + rtol * expected.abs(),
            "{path}: {actual} != {expected}"
        );
    } else {
        assert_eq!(actual, expected, "{path}");
    }
}

#[test]
fn entry_grids_match_current_stock_r_for_subject_continuations() {
    let reference: Value = serde_json::from_str(include_str!(
        "../python/tests/fixtures/aj_entry_reference.json"
    ))
    .unwrap();
    assert_eq!(reference["survival_version"], "3.8.12");
    let source = &reference["data"];
    let states: Vec<String> = reference["event_levels"].as_array().unwrap()[1..]
        .iter()
        .map(|level| level.as_str().unwrap().to_string())
        .collect();
    for case in reference["cases"].as_array().unwrap() {
        let rows: Vec<usize> = case["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row.as_u64().unwrap() as usize - 1)
            .collect();
        let numeric = |name: &str| {
            rows.iter()
                .map(|&row| source[name][row].as_f64().unwrap())
                .collect()
        };
        let data = SurvfitAJData::try_new(
            Some(numeric("start")),
            numeric("stop"),
            rows.iter()
                .map(|&row| match source["event"][row].as_str().unwrap() {
                    "censor" => 0,
                    "a" => 1,
                    "b" => 2,
                    _ => unreachable!("fixture event level"),
                })
                .collect(),
            states.clone(),
            case["weighted"]
                .as_bool()
                .unwrap()
                .then(|| numeric("weight")),
            case["grouped"].as_bool().unwrap().then(|| {
                rows.iter()
                    .map(|&row| i32::from(source["group"][row] == "two"))
                    .collect()
            }),
            Some(
                rows.iter()
                    .map(|&row| source["id"][row].as_i64().unwrap())
                    .collect(),
            ),
            None,
            None,
            None,
        )
        .unwrap();
        let options = SurvfitAJOptions {
            entry: case["entry"].as_bool().unwrap(),
            time0: case["time0"].as_bool().unwrap(),
            start_time: case["start_time"].as_f64(),
            influence: true,
            ..Default::default()
        };
        let fit = survfitaj(&data, &options).unwrap();
        let point_estimates = survfitaj(
            &data,
            &SurvfitAJOptions {
                se_fit: false,
                ..options
            },
        )
        .unwrap();
        // Every result field other than uncertainty matches the complete
        // general-IJ fit checked against stock R below.
        assert_point_estimates_equal(&point_estimates, &fit);
        let expected = &case["expected"];
        let name = case["name"].as_str().unwrap();
        assert_eq!(
            serde_json::to_value(&fit.n).unwrap(),
            expected["n"],
            "{name}: n"
        );
        assert_eq!(
            serde_json::to_value(&fit.states).unwrap(),
            expected["states"],
            "{name}: states"
        );
        assert_eq!(
            serde_json::to_value(&fit.strata).unwrap(),
            expected["strata"],
            "{name}: strata"
        );
        assert_eq!(
            serde_json::to_value(&fit.n_id).unwrap(),
            expected["n_id"],
            "{name}: n_id"
        );
        assert_values(&values(&fit.p0), &expected["p0"], &format!("{name}: p0"));
        assert_values(
            &serde_json::to_value(fit.t0).unwrap(),
            &expected["t0"],
            &format!("{name}: t0"),
        );
        assert_values(
            &serde_json::to_value(&fit.time).unwrap(),
            &expected["time"],
            &format!("{name}: time"),
        );
        for (field, actual) in [
            ("n_risk", values(&fit.n_risk)),
            ("n_event", values(&fit.n_event)),
            ("n_censor", values(&fit.n_censor)),
            ("n_transition", values(&fit.n_transition)),
            ("pstate", values(&fit.pstate)),
            ("cumhaz", values(&fit.cumhaz)),
            ("n_enter", serde_json::to_value(&fit.n_enter).unwrap()),
            ("std_err", serde_json::to_value(&fit.std_err).unwrap()),
            ("std_chaz", serde_json::to_value(&fit.std_chaz).unwrap()),
            ("std_auc", serde_json::to_value(&fit.std_auc).unwrap()),
            ("lower", serde_json::to_value(&fit.lower).unwrap()),
            ("upper", serde_json::to_value(&fit.upper).unwrap()),
        ] {
            assert_values(&actual, &expected[field], &format!("{name}: {field}"));
        }
        let influence: Vec<Vec<Vec<Vec<f64>>>> = fit
            .influence_pstate
            .as_ref()
            .unwrap()
            .iter()
            .map(|curve| {
                let (ncluster, ntime, nstate) = curve.values.dim();
                (0..ncluster)
                    .map(|cluster| {
                        (0..ntime)
                            .map(|time| {
                                (0..nstate)
                                    .map(|state| curve.values[[cluster, time, state]])
                                    .collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        assert_values(
            &serde_json::to_value(influence).unwrap(),
            &expected["influence_pstate"],
            &format!("{name}: influence_pstate"),
        );
        if let Some(counts) = &fit.counts {
            for (field, actual) in [
                ("n_risk", values(&counts.n_risk)),
                ("n_transition", values(&counts.n_transition)),
                ("n_censor", values(&counts.n_censor)),
                ("n_enter", serde_json::to_value(&counts.n_enter).unwrap()),
            ] {
                assert_values_with_tolerance(
                    &actual,
                    &expected["counts"][field],
                    &format!("{name}: unweighted {field}"),
                    0.0,
                    0.0,
                );
            }
        } else {
            assert!(expected["counts"].is_null(), "{name}: unweighted counts");
        }
    }
}

#[test]
fn subject_gaps_are_rejected_before_constructing_an_entry_grid() {
    for entry in [false, true] {
        for timefix in [false, true] {
            let mut data = SurvfitAJData::try_new(
                Some(vec![0.0, 2.75, 0.0]),
                vec![2.0, 5.0, 4.0],
                vec![0, 0, 1],
                vec!["a".into()],
                None,
                None,
                Some(vec![2, 2, 3]),
                None,
                None,
                None,
            )
            .unwrap();
            let options = SurvfitAJOptions {
                entry,
                timefix,
                ..Default::default()
            };
            let error = survfitaj(&data, &options).unwrap_err().to_string();
            assert!(error.contains("gap = 1"), "{error}");
            // The same censor-only trajectory is valid when contiguous.
            data.start.as_mut().unwrap()[1] = 2.0;
            assert!(survfitaj(&data, &options).is_ok());
        }
    }
}
