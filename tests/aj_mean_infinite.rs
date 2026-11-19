use serde_json::Value;
use survival::surv_analysis::{
    ConfType, RmeanOption, SurvfitAJData, SurvfitAJOptions, SurvfitAJResult, summary_survfit_aj,
    survfit0_aj, survfitaj, survmean_aj,
};

fn reference() -> Value {
    serde_json::from_str(include_str!(
        "../python/tests/fixtures/aj_mean_infinite_reference.json"
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

fn strings(value: &Value) -> Vec<String> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_str().unwrap().to_owned())
        .collect()
}

fn data(case: &Value) -> SurvfitAJData {
    let source = &case["input"];
    let optional = |name: &str| (!source[name].is_null()).then(|| numbers(&source[name]));
    let ids = |name: &str| {
        optional(name).map(|values| values.into_iter().map(|value| value as i64).collect())
    };
    SurvfitAJData::try_new(
        optional("start"),
        numbers(&source["time"]),
        numbers(&source["state"])
            .into_iter()
            .map(|v| v as i32)
            .collect(),
        strings(&source["states"]),
        optional("weights"),
        optional("strata").map(|values| values.into_iter().map(|v| v as i32).collect()),
        ids("id"),
        (!source["istate"].is_null()).then(|| strings(&source["istate"])),
        (!source["istate_levels"].is_null()).then(|| strings(&source["istate_levels"])),
        ids("cluster"),
    )
    .unwrap()
}

fn options(case: &Value) -> SurvfitAJOptions {
    let options = &case["options"];
    SurvfitAJOptions {
        influence: options["influence"].as_bool().unwrap(),
        time0: options["time0"].as_bool().unwrap(),
        entry: options["entry"].as_bool().unwrap(),
        timefix: options["timefix"].as_bool().unwrap(),
        conf_type: ConfType::Plain,
        ..Default::default()
    }
}

fn assert_vector(actual: &[f64], expected: &Value, label: &str) {
    let expected = numbers(expected);
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (&actual, expected) in actual.iter().zip(expected) {
        if expected.is_nan() {
            assert!(actual.is_nan(), "{label}: {actual} != NaN");
        } else if expected.is_infinite() {
            assert_eq!(actual, expected, "{label}");
        } else {
            assert!(
                (actual - expected).abs() <= 2e-12 * (1. + expected.abs()),
                "{label}: {actual} != {expected}"
            );
        }
    }
}

fn assert_matrix(actual: &[Vec<f64>], expected: &Value, label: &str) {
    let expected = expected.as_array().unwrap();
    assert_eq!(actual.len(), expected.len(), "{label}: rows");
    for (actual, expected) in actual.iter().zip(expected) {
        assert_vector(actual, expected, label);
    }
}

fn assert_fit(fit: &SurvfitAJResult, case: &Value) {
    let expected = &case["fit"];
    let name = case["name"].as_str().unwrap();
    assert_vector(&fit.time, &expected["time"], name);
    assert_vector(&[fit.t0], &expected["t0"], name);
    assert_eq!(fit.states, strings(&expected["states"]), "{name}");
    for (field, values) in [
        ("p0", &fit.p0),
        ("pstate", &fit.pstate),
        ("cumhaz", &fit.cumhaz),
        ("n_risk", &fit.n_risk),
        ("n_event", &fit.n_event),
        ("n_censor", &fit.n_censor),
        ("n_transition", &fit.n_transition),
    ] {
        assert_matrix(values, &expected[field], &format!("{name}.{field}"));
    }
    for (field, values) in [
        ("n_enter", &fit.n_enter),
        ("std_err", &fit.std_err),
        ("std_chaz", &fit.std_chaz),
        ("std_auc", &fit.std_auc),
    ] {
        if let Some(values) = values {
            assert_matrix(values, &expected[field], &format!("{name}.{field}"));
        } else {
            assert!(expected[field].is_null(), "{name}.{field}");
        }
    }
}

#[test]
fn aj_endpoints_and_influence_means_match_stock_r() {
    for case in reference()["cases"].as_array().unwrap() {
        let fit = survfitaj(&data(case), &options(case)).unwrap();
        assert_fit(&fit, case);
        let normalized = survfitaj(
            &data(case),
            &SurvfitAJOptions {
                timefix: true,
                ..options(case)
            },
        )
        .unwrap();
        assert_fit(&normalized, case);
        let name = case["name"].as_str().unwrap();
        for query in case["tables"].as_array().unwrap() {
            let expected = &query["expected"];
            if expected.get("error").is_some() {
                // summary.survfitms has stricter truncation checks than the
                // native mean-table helper; the Python fixture test covers them.
                continue;
            }
            let option = query["rmean"].as_str().map_or_else(
                || RmeanOption::At(number(&query["rmean"][0])),
                |value| RmeanOption::parse(value).unwrap(),
            );
            for source in [&fit, &survfit0_aj(&fit)] {
                let (table, ends, columns) = survmean_aj(source, 1.0, option).unwrap();
                assert_matrix(&table, &expected["values"], name);
                assert_eq!(columns, strings(&expected["columns"]), "{name}");
                if expected["ends"].is_null() {
                    assert!(ends.is_empty());
                } else {
                    assert_vector(&ends, &expected["ends"], name);
                }
            }
        }
    }
}

#[test]
fn infinite_aj_inputs_keep_missing_values_weights_and_options_checked() {
    let reference = reference();
    let case = reference["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "counting_negative_entry_TRUE_TRUE")
        .unwrap();
    let data = data(case);
    for mutate in [
        (|data: &mut SurvfitAJData| data.time[0] = f64::NAN) as fn(&mut SurvfitAJData),
        |data| data.start.as_mut().unwrap()[0] = f64::NAN,
        |data| data.start.as_mut().unwrap()[0] = f64::INFINITY,
        |data| data.weights = Some(vec![1., 1., f64::INFINITY]),
        |data| data.state.pop().map(|_| ()).unwrap(),
    ] {
        let mut invalid = data.clone();
        mutate(&mut invalid);
        assert!(survfitaj(&invalid, &options(case)).is_err());
    }
    let fit = survfitaj(&data, &options(case)).unwrap();
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            survfitaj(
                &data,
                &SurvfitAJOptions {
                    start_time: Some(value),
                    ..options(case)
                }
            )
            .is_err()
        );
        assert!(summary_survfit_aj(&fit, Some(&[value]), false, true).is_err());
    }
}
