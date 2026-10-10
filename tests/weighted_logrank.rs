use serde_json::Value;
use survival::surv_analysis::{SurvdiffData, survdiff};

fn floats(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

fn integers(value: &Value) -> Vec<i32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect()
}

fn assert_close(actual: f64, expected: f64, name: &str) {
    assert!(
        (actual - expected).abs() <= 2e-13 + 2e-12 * expected.abs(),
        "{name}: {actual} != {expected}"
    );
}

fn assert_matrix(actual: &[Vec<f64>], expected: &Value, name: &str) {
    let expected = expected.as_array().unwrap();
    assert_eq!(actual.len(), expected.len(), "{name}");
    for (actual, expected) in actual.iter().zip(expected) {
        let expected = floats(expected);
        assert_eq!(actual.len(), expected.len(), "{name}");
        for (&actual, &expected) in actual.iter().zip(&expected) {
            assert_close(actual, expected, name);
        }
    }
}

#[test]
fn complete_g_rho_results_match_independent_r_references() {
    let reference: Value = serde_json::from_str(include_str!(
        "../python/tests/fixtures/weighted_logrank_reference.json"
    ))
    .unwrap();
    for case in reference["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let source = &reference["data"];
        let mut time = floats(&source["time"]);
        if case["near"].as_bool().unwrap() {
            time[1] += 5e-10;
        }
        let counting = case["counting"].as_bool().unwrap();
        let grouped = case["grouped"].as_bool().unwrap();
        let data = SurvdiffData::try_new(
            counting.then(|| floats(&source["start"])),
            time,
            integers(&source["status"]),
            integers(&source["group"]),
            grouped.then(|| integers(&source["stratum"])),
        )
        .unwrap();
        let actual = survdiff(
            &data,
            case["rho"].as_f64().unwrap(),
            case["timefix"].as_bool().unwrap(),
        )
        .unwrap();
        let expected = &case["expected"];
        assert_eq!(
            actual.n,
            integers(&expected["n"])
                .iter()
                .map(|&n| n as usize)
                .collect::<Vec<_>>(),
            "{name}"
        );
        assert_eq!(
            actual.group_codes,
            integers(&expected["group_codes"]),
            "{name}"
        );
        assert_eq!(
            actual.df,
            expected["df"].as_u64().unwrap() as usize,
            "{name}"
        );
        assert_eq!(
            actual.strata,
            grouped.then(|| integers(&expected["strata"])
                .iter()
                .map(|&n| n as usize)
                .collect()),
            "{name}"
        );
        assert_matrix(&actual.obs, &expected["obs"], name);
        assert_matrix(&actual.exp, &expected["exp"], name);
        assert_matrix(&actual.var, &expected["var"], name);
        assert_close(actual.chisq, expected["chisq"].as_f64().unwrap(), name);
        assert_close(actual.pvalue, expected["pvalue"].as_f64().unwrap(), name);
    }
}
