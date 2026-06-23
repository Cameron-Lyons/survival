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
    let mut covar_cols = Vec::with_capacity(params.nvar);
    let mut remaining = covar;
    for _ in 0..params.nvar {
        let (col, rest) = remaining.split_at_mut(params.nused);
        covar_cols.push(col);
        remaining = rest;
    }
    let (a, rest) = work.split_at_mut(params.nvar);
    let (a2, mean) = rest.split_at_mut(params.nvar);
    let mut person = 0;
    while person < params.nused {
        if event[person] != 1.0 {
            person += 1;
            continue;
        }
        let time = stop[person];
        let mut deaths = 0.0;
        let mut denom = 0.0;
        let mut efron_wt = 0.0;
        for i in 0..params.nvar {
            a[i] = 0.0;
            a2[i] = 0.0;
        }
        let mut k = person;
        while k < params.nused {
            if start[k] < time {
                let weight = input.score[k];
                denom += weight;
                for i in 0..params.nvar {
                    a[i] += weight * covar_cols[i][k];
                }
                if stop[k] == time && event[k] == 1.0 {
                    deaths += 1.0;
                    efron_wt += weight;
                    for i in 0..params.nvar {
                        a2[i] += weight * covar_cols[i][k];
                    }
                }
            }
            if input.strata[k] == 1 {
                break;
            }
            k += 1;
        }
        for mean_i in mean.iter_mut().take(params.nvar) {
            *mean_i = 0.0;
        }
        if deaths > 0.0 {
            for k_death in 0..(deaths as usize) {
                let temp = if params.method == 1 {
                    (k_death as f64) / deaths
                } else {
                    0.0
                };
                for i in 0..params.nvar {
                    let denominator = deaths * (denom - temp * efron_wt);
                    if denominator != 0.0 {
                        mean[i] += (a[i] - temp * a2[i]) / denominator;
                    }
                }
            }
        }
        let mut k = person;
        while k < params.nused && stop[k] == time {
            if event[k] == 1.0 {
                for i in 0..params.nvar {
                    covar_cols[i][k] -= mean[i];
                }
            }
            person += 1;
            if input.strata[k] == 1 {
                break;
            }
            k += 1;
        }
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

    fn valid_inputs() -> (Vec<f64>, Vec<f64>, Vec<i32>, Vec<f64>) {
        (
            vec![
                0.0, 0.0, 0.0, 0.0, // start
                1.0, 2.0, 3.0, 4.0, // stop
                1.0, 1.0, 0.0, 1.0, // event
            ],
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
}
