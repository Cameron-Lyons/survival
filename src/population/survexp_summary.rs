//! Time selection for R's `summary.survexp`.

use super::SurvExpResult;
use crate::error::{SurvivalError, SurvivalResult};
use pyo3::prelude::*;

/// Summarize expected survival curves at the requested times.
///
/// Omitted times retain the original rows. Requested times are sorted, with
/// duplicates retained and missing or out-of-range values removed. Survival
/// uses the preceding observation (1 before the first); risk counts use the
/// following observation. Repeated source times use R's averaged, truncated
/// interpolation indices. A sorted sweep takes O(n + q) after sorting queries.
/// `scale` divides the selected times, including IEEE zero/nonfinite behavior
/// as in R's scalar arithmetic.
pub fn summary_survexp(
    fit: &SurvExpResult,
    times: Option<&[f64]>,
    scale: f64,
) -> SurvivalResult<SurvExpResult> {
    let n = fit.time.len();
    if fit.surv.len() != n || fit.n_risk.len() != n {
        return Err(SurvivalError::invalid_input(
            "survival and risk matrices must have one row per time",
        ));
    }
    if fit.time.iter().any(|t| !t.is_finite()) || fit.time.windows(2).any(|w| w[0] > w[1]) {
        return Err(SurvivalError::invalid_input(
            "source times must be finite and nondecreasing",
        ));
    }
    let ncols = fit.surv.first().map_or(0, Vec::len);
    if n > 0
        && (ncols == 0
            || fit.surv.iter().any(|row| row.len() != ncols)
            || fit.n_risk.iter().any(|row| row.len() != ncols))
    {
        return Err(SurvivalError::invalid_input(
            "survival and risk matrices must have the same positive column count",
        ));
    }
    let Some(times) = times else {
        return Ok(SurvExpResult {
            time: fit.time.iter().map(|t| t / scale).collect(),
            surv: fit.surv.clone(),
            n_risk: fit.n_risk.clone(),
            method: fit.method.clone(),
        });
    };
    let mut result = SurvExpResult {
        time: Vec::new(),
        surv: Vec::new(),
        n_risk: Vec::new(),
        method: fit.method.clone(),
    };
    if n == 0 {
        return Ok(result);
    }
    let mintime = fit.time[0].min(0.0);
    let maxtime = fit.time[n - 1];
    let mut requested: Vec<f64> = times
        .iter()
        .copied()
        .filter(|&t| t >= mintime && t <= maxtime)
        .collect();
    if !requested.is_sorted() {
        requested.sort_unstable_by(f64::total_cmp);
    }
    result.time.reserve(requested.len());
    result.surv.reserve(requested.len());
    result.n_risk.reserve(requested.len());

    // Average the 1-based indices of every equal-time block, then truncate
    // for R's matrix subscripts. In zero-based form this is its midpoint.
    let mut levels = Vec::with_capacity(n);
    let mut start = 0;
    while start < n {
        let mut end = start + 1;
        while end < n && fit.time[end] == fit.time[start] {
            end += 1;
        }
        levels.push((fit.time[start], start + (end - start - 1) / 2));
        start = end;
    }
    let mut next = 0;
    for t in requested {
        while levels[next].0 < t {
            next += 1;
        }
        let previous = if levels[next].0 == t {
            Some(levels[next].1)
        } else {
            next.checked_sub(1).map(|idx| levels[idx].1)
        };
        result.time.push(t / scale);
        result
            .surv
            .push(previous.map_or_else(|| vec![1.0; ncols], |idx| fit.surv[idx].clone()));
        result.n_risk.push(fit.n_risk[levels[next].1].clone());
    }
    Ok(result)
}

/// Python entry point with row-major `time x curve` matrices.
#[pyfunction(name = "summary_survexp")]
#[pyo3(signature = (time, surv, n_risk, times=None, scale=1.0, method=String::new()))]
pub fn summary_survexp_py(
    time: Vec<f64>,
    surv: Vec<Vec<f64>>,
    n_risk: Vec<Vec<f64>>,
    times: Option<Vec<f64>>,
    scale: f64,
    method: String,
) -> PyResult<SurvExpResult> {
    Ok(summary_survexp(
        &SurvExpResult {
            time,
            surv,
            n_risk,
            method,
        },
        times.as_deref(),
        scale,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_r_summary_time_selection() {
        let reference: serde_json::Value = serde_json::from_str(include_str!(
            "../../python/tests/fixtures/population_summary_reference.json"
        ))
        .unwrap();
        let number = |value: &serde_json::Value| match value.as_str() {
            Some("Inf") => f64::INFINITY,
            Some("-Inf") => f64::NEG_INFINITY,
            _ => value.as_f64().unwrap_or(f64::NAN),
        };
        let vector = |value: &serde_json::Value| {
            value
                .as_array()
                .unwrap()
                .iter()
                .map(number)
                .collect::<Vec<_>>()
        };
        let matrix = |value: &serde_json::Value| {
            value
                .as_array()
                .unwrap()
                .iter()
                .map(vector)
                .collect::<Vec<_>>()
        };
        for case in reference["native_cases"].as_array().unwrap() {
            let fit = SurvExpResult {
                time: vector(&case["time"]),
                surv: matrix(&case["surv"]),
                n_risk: matrix(&case["n_risk"]),
                method: "cohort".into(),
            };
            let times = vector(&case["times"]);
            let times = (!case["omitted"].as_bool().unwrap()).then_some(times.as_slice());
            let actual = summary_survexp(&fit, times, number(&case["scale"])).unwrap();
            let expected = &case["expected"];
            assert_eq!(actual.method, "cohort");
            let equal = |actual: &[f64], expected: &[f64]| {
                assert_eq!(actual.len(), expected.len(), "{}", case["name"]);
                for (&a, &b) in actual.iter().zip(expected) {
                    assert!(
                        a == b || (a.is_nan() && b.is_nan()),
                        "{}: {a} != {b}",
                        case["name"]
                    );
                }
            };
            equal(&actual.time, &vector(&expected["time"]));
            for (actual, expected) in [
                (&actual.surv, matrix(&expected["surv"])),
                (&actual.n_risk, matrix(&expected["n_risk"])),
            ] {
                assert_eq!(actual.len(), expected.len(), "{}", case["name"]);
                for (a, b) in actual.iter().zip(expected) {
                    equal(a, &b);
                }
            }
        }
    }

    #[test]
    fn rejects_malformed_shapes_and_source_times() {
        let fit = SurvExpResult {
            time: vec![1.0, 3.0],
            surv: vec![vec![0.9], vec![0.8]],
            n_risk: vec![vec![3.0], vec![2.0]],
            method: "cohort".into(),
        };
        let mut invalid = fit.clone();
        invalid.surv.pop();
        assert!(summary_survexp(&invalid, None, 1.0).is_err());
        invalid = fit.clone();
        invalid.n_risk[0].push(1.0);
        assert!(summary_survexp(&invalid, None, 1.0).is_err());
        for times in [
            vec![2.0, 1.0],
            vec![f64::NAN, 3.0],
            vec![1.0, f64::INFINITY],
        ] {
            invalid = fit.clone();
            invalid.time = times;
            assert!(summary_survexp(&invalid, None, 1.0).is_err());
        }
    }
}
