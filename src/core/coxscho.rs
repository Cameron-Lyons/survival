use pyo3::prelude::*;

use crate::internal::validation::{
    ValidationError, validate_binary_f64, validate_binary_i32, validate_finite,
    validate_non_negative,
};

pub(crate) struct CoxSchoInput<'a> {
    pub y: &'a [f64],
    pub score: &'a [f64],
    pub strata: &'a [i32],
}
pub(crate) struct CoxSchoParams {
    pub nused: usize,
    pub nvar: usize,
    pub method: i32,
}

fn validation_err_to_py(err: ValidationError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(err.to_string())
}

fn validate_schoenfeld_inputs(
    y: &[f64],
    score: &[f64],
    strata: &[i32],
    covar: &[f64],
    nvar: usize,
    method: i32,
) -> PyResult<()> {
    let nused = score.len();
    let expected_y_len = 3usize.checked_mul(nused).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("3 * n exceeds supported array size")
    })?;
    let expected_covar_len = nvar.checked_mul(nused).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("nvar * n exceeds supported array size")
    })?;
    if y.len() < expected_y_len {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "y array must have length >= 3 * n (start, stop, event)",
        ));
    }
    if strata.len() < nused {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "strata array length must match score length",
        ));
    }
    if covar.len() < expected_covar_len {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "covar array must have length >= nvar * n",
        ));
    }
    if method != 0 && method != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "method must be 0 (Breslow) or 1 (Efron)",
        ));
    }

    validate_finite(&y[..expected_y_len], "y").map_err(validation_err_to_py)?;
    validate_finite(score, "score").map_err(validation_err_to_py)?;
    validate_non_negative(score, "score").map_err(validation_err_to_py)?;
    validate_binary_i32(&strata[..nused], "strata").map_err(validation_err_to_py)?;
    validate_finite(&covar[..expected_covar_len], "covar").map_err(validation_err_to_py)?;
    validate_binary_f64(&y[2 * nused..expected_y_len], "event").map_err(validation_err_to_py)?;

    Ok(())
}

pub(crate) fn coxscho(
    params: CoxSchoParams,
    input: CoxSchoInput,
    covar: &mut [f64],
    work: &mut [f64],
) {
    assert!(input.y.len() >= 3 * params.nused, "y array too short");
    assert!(
        covar.len() >= params.nvar * params.nused,
        "covar array too short for nvar and nused"
    );
    assert!(input.score.len() >= params.nused, "score array too short");
    assert!(input.strata.len() >= params.nused, "strata array too short");
    assert!(
        work.len() >= 3 * params.nvar,
        "work array must be at least 3 * nvar in length"
    );
    let start = &input.y[0..params.nused];
    let stop = &input.y[params.nused..2 * params.nused];
    let event = &input.y[2 * params.nused..3 * params.nused];
    let (a, rest) = work.split_at_mut(params.nvar);
    let (a2, mean) = rest.split_at_mut(params.nvar);
    let mut entry_order = Vec::new();
    let mut active = vec![false; params.nused];
    let mut event_ranges = Vec::new();
    let mut event_means = Vec::new();
    let mut stratum_start = 0;

    while stratum_start < params.nused {
        let mut stratum_end = stratum_start;
        while stratum_end + 1 < params.nused && input.strata[stratum_end] != 1 {
            stratum_end += 1;
        }

        entry_order.clear();
        entry_order.extend(stratum_start..=stratum_end);
        entry_order.sort_by(|&left, &right| {
            start[left]
                .total_cmp(&start[right])
                .then_with(|| left.cmp(&right))
        });
        active[stratum_start..=stratum_end].fill(false);
        event_ranges.clear();
        event_means.clear();

        let mut denom = 0.0;
        a.fill(0.0);
        let mut entry_pos = 0;
        let mut stop_pos = stratum_start;
        let mut time_start = stratum_start;

        while time_start <= stratum_end {
            let time = stop[time_start];
            let mut time_end = time_start;
            while time_end < stratum_end && stop[time_end + 1] == time {
                time_end += 1;
            }

            let death_count = (time_start..=time_end)
                .filter(|&row| event[row] == 1.0)
                .count();
            if death_count > 0 {
                while entry_pos < entry_order.len() && start[entry_order[entry_pos]] < time {
                    let row = entry_order[entry_pos];
                    entry_pos += 1;
                    if active[row] {
                        continue;
                    }
                    active[row] = true;
                    let risk = input.score[row];
                    denom += risk;
                    for var in 0..params.nvar {
                        a[var] += risk * covar[var * params.nused + row];
                    }
                }

                while stop_pos <= stratum_end && stop[stop_pos] < time {
                    let row = stop_pos;
                    stop_pos += 1;
                    if !active[row] {
                        continue;
                    }
                    active[row] = false;
                    let risk = input.score[row];
                    denom -= risk;
                    for var in 0..params.nvar {
                        a[var] -= risk * covar[var * params.nused + row];
                    }
                }

                let deaths = death_count as f64;
                let mut efron_wt = 0.0;
                a2.fill(0.0);
                for row in (time_start..=time_end).filter(|&row| event[row] == 1.0) {
                    let risk = input.score[row];
                    efron_wt += risk;
                    for var in 0..params.nvar {
                        a2[var] += risk * covar[var * params.nused + row];
                    }
                }

                mean.fill(0.0);
                for step in 0..death_count {
                    let fraction = if params.method == 1 {
                        step as f64 / deaths
                    } else {
                        0.0
                    };
                    let step_denom = deaths * (denom - fraction * efron_wt);
                    if step_denom == 0.0 {
                        continue;
                    }
                    for var in 0..params.nvar {
                        mean[var] += (a[var] - fraction * a2[var]) / step_denom;
                    }
                }

                let mean_offset = event_means.len();
                event_means.extend_from_slice(mean);
                event_ranges.push((time_start, time_end, mean_offset));
            }

            if time_end == stratum_end {
                break;
            }
            time_start = time_end + 1;
        }

        for &(time_start, time_end, mean_offset) in &event_ranges {
            let event_mean = &event_means[mean_offset..mean_offset + params.nvar];
            for row in (time_start..=time_end).filter(|&row| event[row] == 1.0) {
                for (var, &mean_value) in event_mean.iter().enumerate() {
                    covar[var * params.nused + row] -= mean_value;
                }
            }
        }

        stratum_start = stratum_end + 1;
    }
}
#[pyfunction]
#[pyo3(signature = (y, score, strata, covar, nvar, method=0))]
pub fn schoenfeld_residuals(
    y: Vec<f64>,
    score: Vec<f64>,
    strata: Vec<i32>,
    covar: Vec<f64>,
    nvar: usize,
    method: i32,
) -> PyResult<Vec<f64>> {
    let nused = score.len();
    validate_schoenfeld_inputs(&y, &score, &strata, &covar, nvar, method)?;
    let work_len = 3usize.checked_mul(nvar).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("3 * nvar exceeds supported array size")
    })?;
    let mut covar_copy = covar.clone();
    let mut work = vec![0.0; work_len];
    let params = CoxSchoParams {
        nused,
        nvar,
        method,
    };
    let input = CoxSchoInput {
        y: &y,
        score: &score,
        strata: &strata,
    };
    coxscho(params, input, &mut covar_copy, &mut work);
    Ok(covar_copy)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-12,
                "value {idx} differed: {actual} != {expected}"
            );
        }
    }

    fn valid_inputs() -> (Vec<f64>, Vec<f64>, Vec<i32>, Vec<f64>) {
        (
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 0.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0],
            vec![0, 0, 0, 0],
            vec![1.0, 2.0, 3.0, 4.0],
        )
    }

    #[test]
    fn schoenfeld_wrapper_rejects_invalid_method() {
        let (y, score, strata, covar) = valid_inputs();

        let err = schoenfeld_residuals(y, score, strata, covar, 1, 2)
            .expect_err("unsupported method should fail");

        assert!(
            err.to_string()
                .contains("method must be 0 (Breslow) or 1 (Efron)")
        );
    }

    #[test]
    fn schoenfeld_wrapper_rejects_non_finite_inputs() {
        let (mut y, score, strata, covar) = valid_inputs();
        y[1] = f64::NAN;

        let err = schoenfeld_residuals(y, score, strata, covar, 1, 0)
            .expect_err("NaN y value should fail");

        assert!(err.to_string().contains("y contains non-finite"));
    }

    #[test]
    fn schoenfeld_wrapper_rejects_negative_score() {
        let (y, mut score, strata, covar) = valid_inputs();
        score[2] = -1.0;

        let err = schoenfeld_residuals(y, score, strata, covar, 1, 0)
            .expect_err("negative score should fail");

        assert!(err.to_string().contains("score contains negative value"));
    }

    #[test]
    fn schoenfeld_wrapper_rejects_non_binary_event() {
        let (mut y, score, strata, covar) = valid_inputs();
        y[9] = 0.5;

        let err = schoenfeld_residuals(y, score, strata, covar, 1, 0)
            .expect_err("non-binary event should fail");

        assert!(err.to_string().contains("event values must be 0 or 1"));
    }

    #[test]
    fn weighted_risk_scores_match_stratified_tied_event_reference() {
        let y = vec![
            0.0, 0.0, 0.0, 1.0, 1.5, 0.0, 0.0, 1.0, 0.0, // start
            1.0, 2.0, 2.0, 2.0, 3.0, 1.0, 2.0, 2.0, 3.0, // stop
            1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, // event
        ];
        let covar = vec![
            -1.0, 0.2, 1.1, 0.5, -0.4, 0.8, -0.7, 0.3, 1.4, // x1
            0.5, -1.0, 0.7, 1.2, -0.2, -0.6, 0.9, -1.3, 0.1, // x2
        ];
        let strata = vec![0, 0, 0, 0, 1, 0, 0, 0, 1];
        let score = vec![1.0, 1.2, 0.8, 1.5, 0.7, 1.1, 0.9, 1.4, 0.6];
        let cases = [
            (
                0,
                vec![
                    -1.04,
                    -0.178571428571429,
                    0.721428571428572,
                    0.5,
                    -0.4,
                    0.8,
                    -0.917241379310345,
                    0.0827586206896552,
                    1.4,
                    0.546666666666667,
                    -1.24285714285714,
                    0.457142857142857,
                    1.2,
                    -0.2,
                    -0.6,
                    1.22758620689655,
                    -0.972413793103448,
                    0.1,
                ],
            ),
            (
                1,
                vec![
                    -1.04,
                    -0.150223214285714,
                    0.749776785714286,
                    0.5,
                    -0.4,
                    0.8,
                    -1.01862068965517,
                    -0.0186206896551724,
                    1.4,
                    0.546666666666667,
                    -1.33080357142857,
                    0.369196428571429,
                    1.2,
                    -0.2,
                    -0.6,
                    1.19093596059113,
                    -1.00906403940887,
                    0.1,
                ],
            ),
        ];

        for (method, expected) in cases {
            let actual = schoenfeld_residuals(
                y.clone(),
                score.clone(),
                strata.clone(),
                covar.clone(),
                2,
                method,
            )
            .unwrap();
            assert_close(&actual, &expected);
        }
    }
}
