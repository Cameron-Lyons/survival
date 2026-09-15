//! Legacy Cox baseline bindings kept for the Python facade, all thin views
//! of `agsurv` (`R/agsurv.R`, `src/agsurv4.c`, `src/agsurv5.c`):
//!
//! * `cox_expected_baseline_by_stratum` — the cumulative hazard, its
//!   variance and the cumulative `xbar` per stratum at the event times, the
//!   pieces `predict.coxph(type = "expected")` integrates;
//! * `compute_baseline_survival_steps` / `agsurv4` — the Kalbfleisch-
//!   Prentice increments;
//! * `compute_tied_baseline_summaries` / `agsurv5` — the Efron sums.
//!
//! The argument checks are those of the original bindings; the arithmetic
//! is the one implementation in `crate::surv_analysis::agsurv`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::core::strata_order::stratum_groups;
use crate::surv_analysis::agsurv::{AgsurvData, CoxSurvType, agsurv_rows, agsurv4, agsurv5};
use ndarray::{Array2, ShapeBuilder};

const PY_EXP_CLAMP_MIN: f64 = -745.0;
const PY_EXP_CLAMP_MAX: f64 = 709.0;

type CoxExpectedBaselineOutput = (
    Vec<i32>,
    Vec<Vec<f64>>,
    Vec<Vec<f64>>,
    Vec<Vec<f64>>,
    Vec<Vec<Vec<f64>>>,
);

fn value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn validate_exact_len(name: &str, got: usize, expected: usize) -> PyResult<()> {
    if got != expected {
        return Err(value_error(format!(
            "{name} length must be {expected}; got {got}"
        )));
    }
    Ok(())
}

fn validate_min_len(name: &str, got: usize, minimum: usize) -> PyResult<()> {
    if got < minimum {
        return Err(value_error(format!(
            "{name} length must be at least {minimum}; got {got}"
        )));
    }
    Ok(())
}

fn validate_finite_values(name: &str, values: &[f64]) -> PyResult<()> {
    for (idx, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(value_error(format!(
                "{name} contains non-finite value {value} at index {idx}"
            )));
        }
    }
    Ok(())
}

fn validate_positive_finite(name: &str, values: &[f64]) -> PyResult<()> {
    validate_finite_values(name, values)?;
    for (idx, &value) in values.iter().enumerate() {
        if value <= 0.0 {
            return Err(value_error(format!(
                "{name} must be positive; got {value} at index {idx}"
            )));
        }
    }
    Ok(())
}

fn validate_nonnegative_finite(name: &str, values: &[f64]) -> PyResult<()> {
    validate_finite_values(name, values)?;
    for (idx, &value) in values.iter().enumerate() {
        if value < 0.0 {
            return Err(value_error(format!(
                "{name} must be non-negative; got {value} at index {idx}"
            )));
        }
    }
    Ok(())
}

fn validate_baseline_survival_steps_inputs(
    ndeath: &[i32],
    risk: &[f64],
    wt: &[f64],
    sn: usize,
    denom: &[f64],
) -> PyResult<()> {
    validate_exact_len("ndeath", ndeath.len(), sn)?;
    validate_exact_len("denom", denom.len(), sn)?;

    let mut total_deaths = 0usize;
    for (idx, &value) in ndeath.iter().enumerate() {
        if value < 0 {
            return Err(value_error(format!(
                "ndeath must be non-negative; got {value} at index {idx}"
            )));
        }
        total_deaths = total_deaths
            .checked_add(value as usize)
            .ok_or_else(|| value_error("total number of deaths is too large"))?;
    }

    validate_min_len("risk", risk.len(), total_deaths)?;
    validate_min_len("wt", wt.len(), total_deaths)?;
    validate_positive_finite("risk", &risk[..total_deaths])?;
    validate_nonnegative_finite("wt", &wt[..total_deaths])?;
    validate_positive_finite("denom", denom)?;

    let mut death_index = 0usize;
    for (time_index, &deaths) in ndeath.iter().enumerate() {
        if deaths == 1 {
            let contribution = wt[death_index] * risk[death_index];
            let tolerance = f64::EPSILON * contribution.abs().max(denom[time_index].abs()) * 16.0;
            if contribution - denom[time_index] > tolerance {
                return Err(value_error(format!(
                    "death contribution must not exceed denom at index {time_index}"
                )));
            }
        }
        death_index += deaths as usize;
    }

    Ok(())
}

fn validate_tied_baseline_summaries_inputs(
    n: usize,
    nvar: usize,
    dd: &[i32],
    x1: &[f64],
    x2: &[f64],
    xsum: &[f64],
    xsum2: &[f64],
) -> PyResult<()> {
    validate_exact_len("dd", dd.len(), n)?;
    validate_exact_len("x1", x1.len(), n)?;
    validate_exact_len("x2", x2.len(), n)?;
    let matrix_len = n
        .checked_mul(nvar)
        .ok_or_else(|| value_error("n * nvar is too large"))?;
    validate_exact_len("xsum", xsum.len(), matrix_len)?;
    validate_exact_len("xsum2", xsum2.len(), matrix_len)?;

    for (idx, &value) in dd.iter().enumerate() {
        if value <= 0 {
            return Err(value_error(format!(
                "dd must contain positive event counts; got {value} at index {idx}"
            )));
        }
    }

    validate_positive_finite("x1", x1)?;
    validate_finite_values("x2", x2)?;
    validate_finite_values("xsum", xsum)?;
    validate_finite_values("xsum2", xsum2)?;

    for (idx, &deaths) in dd.iter().enumerate() {
        let d = deaths as f64;
        for tied_index in 0..deaths {
            let denominator = x1[idx] - x2[idx] * tied_index as f64 / d;
            if denominator <= 0.0 || !denominator.is_finite() {
                return Err(value_error(format!(
                    "tied denominator must be positive at row {idx}, tied death {tied_index}"
                )));
            }
        }
    }

    Ok(())
}

fn validate_rectangular_matrix(
    name: &str,
    matrix: &[Vec<f64>],
    n_rows: usize,
    n_cols: usize,
) -> PyResult<()> {
    validate_exact_len(name, matrix.len(), n_rows)?;
    for (row_idx, row) in matrix.iter().enumerate() {
        validate_exact_len(&format!("{name} row {row_idx}"), row.len(), n_cols)?;
        validate_finite_values(&format!("{name} row {row_idx}"), row)?;
    }
    Ok(())
}

fn validate_binary_status(status: &[i32]) -> PyResult<()> {
    for (idx, &value) in status.iter().enumerate() {
        if value != 0 && value != 1 {
            return Err(value_error(format!(
                "status must contain only 0/1 values; got {value} at index {idx}"
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_expected_baseline_inputs(
    time: &[f64],
    status: &[i32],
    covariates: &[Vec<f64>],
    beta: &[f64],
    weights: &[f64],
    strata: &[i32],
    offset: &[f64],
    means: &[f64],
    entry_times: Option<&[f64]>,
) -> PyResult<usize> {
    let n = time.len();
    let nvar = beta.len();
    validate_exact_len("status", status.len(), n)?;
    validate_exact_len("weights", weights.len(), n)?;
    validate_exact_len("strata", strata.len(), n)?;
    validate_exact_len("offset", offset.len(), n)?;
    validate_exact_len("means", means.len(), nvar)?;
    if let Some(entry_times) = entry_times {
        validate_exact_len("entry_times", entry_times.len(), n)?;
        validate_finite_values("entry_times", entry_times)?;
        for (idx, (&entry, &exit)) in entry_times.iter().zip(time.iter()).enumerate() {
            if entry >= exit {
                return Err(value_error(format!(
                    "entry_times[{idx}] must be less than time[{idx}]"
                )));
            }
        }
    }
    validate_finite_values("time", time)?;
    validate_binary_status(status)?;
    validate_rectangular_matrix("covariates", covariates, n, nvar)?;
    validate_finite_values("beta", beta)?;
    validate_nonnegative_finite("weights", weights)?;
    validate_finite_values("offset", offset)?;
    validate_finite_values("means", means)?;
    Ok(nvar)
}

fn safe_exp(value: f64) -> f64 {
    value.clamp(PY_EXP_CLAMP_MIN, PY_EXP_CLAMP_MAX).exp()
}

/// A risk set of zero total weight (`0/0` in R's `agsurv`, which never sees
/// zero weights because `coxph` rejects them) contributes no hazard.
pub(crate) fn zero_if_nan(value: f64) -> f64 {
    if value.is_nan() { 0.0 } else { value }
}

/// Row-major `Vec<Vec<f64>>` (validated rectangular) as an `n x nvar` array.
pub(crate) fn rows_to_array(rows: &[Vec<f64>], nvar: usize) -> Array2<f64> {
    Array2::from_shape_vec((rows.len(), nvar), rows.iter().flatten().copied().collect())
        .expect("rows were validated to be rectangular")
}

/// Per stratum (ascending code), the event times with the cumulative hazard,
/// its variance and the cumulative `xbar` at each: the Breslow (`method =
/// "breslow"` / `"exact"`) or Efron pieces of `agsurv` at the covariate
/// means, with risk `weights * exp(offset + x beta)`.
#[pyfunction]
#[pyo3(signature = (time, status, covariates, beta, weights, strata, offset, means, entry_times=None, method=None))]
#[allow(clippy::too_many_arguments)]
pub fn cox_expected_baseline_by_stratum(
    time: Vec<f64>,
    status: Vec<i32>,
    covariates: Vec<Vec<f64>>,
    beta: Vec<f64>,
    weights: Vec<f64>,
    strata: Vec<i32>,
    offset: Vec<f64>,
    means: Vec<f64>,
    entry_times: Option<Vec<f64>>,
    method: Option<String>,
) -> PyResult<CoxExpectedBaselineOutput> {
    let nvar = validate_expected_baseline_inputs(
        &time,
        &status,
        &covariates,
        &beta,
        &weights,
        &strata,
        &offset,
        &means,
        entry_times.as_deref(),
    )?;
    let survtype = match method.as_deref().unwrap_or("breslow") {
        "efron" => CoxSurvType::Efron,
        "breslow" | "exact" => CoxSurvType::Breslow,
        _ => return Err(value_error("method must be 'breslow', 'efron', or 'exact'")),
    };

    let x = rows_to_array(&covariates, nvar);
    let risk: Vec<f64> = (0..time.len())
        .map(|i| safe_exp(offset[i] + x.row(i).iter().zip(&beta).map(|(x, b)| x * b).sum::<f64>()))
        .collect();
    let data = AgsurvData {
        start: entry_times.as_deref(),
        stop: &time,
        status: &status,
        x: x.view(),
        means: Some(&means),
        weights: &weights,
        risk: &risk,
    };

    let mut out_strata = Vec::new();
    let mut out_times = Vec::new();
    let mut out_hazard = Vec::new();
    let mut out_varhaz = Vec::new();
    let mut out_xbar = Vec::new();
    for (stratum, rows) in stratum_groups(&strata) {
        let curve = agsurv_rows(&data, &rows, survtype, survtype)?;
        let mut cumhaz = 0.0;
        let mut cumvar = 0.0;
        let mut cumxbar = vec![0.0; nvar];
        let mut times = Vec::new();
        let mut hazard = Vec::new();
        let mut varhaz = Vec::new();
        let mut xbar = Vec::new();
        for g in 0..curve.time.len() {
            cumhaz += zero_if_nan(curve.hazard[g]);
            cumvar += zero_if_nan(curve.varhaz[g]);
            for (k, value) in cumxbar.iter_mut().enumerate() {
                *value += zero_if_nan(curve.xbar[(g, k)]);
            }
            if curve.ndeath[g] > 0 {
                times.push(curve.time[g]);
                hazard.push(cumhaz);
                varhaz.push(cumvar);
                xbar.push(cumxbar.clone());
            }
        }
        out_strata.push(stratum);
        out_times.push(times);
        out_hazard.push(hazard);
        out_varhaz.push(varhaz);
        out_xbar.push(xbar);
    }

    Ok((out_strata, out_times, out_hazard, out_varhaz, out_xbar))
}

/// `agsurv4.c`: the Kalbfleisch-Prentice survival increment at each of the
/// `sn` times; `risk` and `wt` are those of the deaths in time order.
#[pyfunction]
pub fn compute_baseline_survival_steps(
    ndeath: Vec<i32>,
    risk: Vec<f64>,
    wt: Vec<f64>,
    sn: usize,
    denom: Vec<f64>,
) -> PyResult<Vec<f64>> {
    validate_baseline_survival_steps_inputs(&ndeath, &risk, &wt, sn, &denom)?;
    let ndeath: Vec<usize> = ndeath.iter().map(|&d| d as usize).collect();
    Ok(agsurv4(&ndeath, &risk, &wt, &denom))
}

/// `agsurv5.c` on R's column-major layout: `(sum1, sum2, xbar)` with `xbar`
/// flattened as `xbar[i + n * k]`.
fn tied_baseline_summaries(
    n: usize,
    nvar: usize,
    dd: &[i32],
    x1: &[f64],
    x2: &[f64],
    xsum: Vec<f64>,
    xsum2: Vec<f64>,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    validate_tied_baseline_summaries_inputs(n, nvar, dd, x1, x2, &xsum, &xsum2)?;
    let ndeath: Vec<usize> = dd.iter().map(|&d| d as usize).collect();
    let column_major = |values: Vec<f64>| {
        Array2::from_shape_vec((n, nvar).f(), values).expect("length was validated")
    };
    let sums = agsurv5(&ndeath, x1, x2, &column_major(xsum), &column_major(xsum2));
    let mut xbar = vec![0.0; n * nvar];
    for k in 0..nvar {
        for i in 0..n {
            xbar[i + n * k] = sums.xbar[(i, k)];
        }
    }
    Ok((sums.sum1, sums.sum2, xbar))
}

/// `agsurv5.c`: the Efron sums `sum1`, `sum2` and `xbar` for `n` death
/// times with `dd` tied deaths each; `xsum` / `xsum2` and the returned
/// `xbar` are `n x nvar` matrices stored column-major, as R passes them.
#[pyfunction]
pub fn compute_tied_baseline_summaries(
    n: usize,
    nvar: usize,
    dd: Vec<i32>,
    x1: Vec<f64>,
    x2: Vec<f64>,
    xsum: Vec<f64>,
    xsum2: Vec<f64>,
) -> PyResult<Py<PyDict>> {
    let (sum1, sum2, xbar) = tied_baseline_summaries(n, nvar, &dd, &x1, &x2, xsum, xsum2)?;
    Python::attach(|py| {
        let dict = PyDict::new(py);
        dict.set_item("sum1", sum1)?;
        dict.set_item("sum2", sum2)?;
        dict.set_item("xbar", xbar)?;
        Ok(dict.into())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    type BaselineArgs = (
        Vec<f64>,
        Vec<i32>,
        Vec<Vec<f64>>,
        Vec<f64>,
        Vec<f64>,
        Vec<i32>,
        Vec<f64>,
        Vec<f64>,
    );

    fn baseline_args() -> BaselineArgs {
        (
            vec![1.0, 2.0, 2.0, 3.0],
            vec![1, 1, 1, 0],
            vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]],
            vec![0.0],
            vec![1.0; 4],
            vec![0; 4],
            vec![0.0; 4],
            vec![1.5],
        )
    }

    #[test]
    fn expected_baseline_handles_breslow_and_efron_ties() {
        let (time, status, covariates, beta, weights, strata, offset, means) = baseline_args();
        let breslow = cox_expected_baseline_by_stratum(
            time.clone(),
            status.clone(),
            covariates.clone(),
            beta.clone(),
            weights.clone(),
            strata.clone(),
            offset.clone(),
            means.clone(),
            None,
            Some("breslow".to_string()),
        )
        .expect("breslow expected baseline should compute");
        let efron = cox_expected_baseline_by_stratum(
            time,
            status,
            covariates,
            beta,
            weights,
            strata,
            offset,
            means,
            None,
            Some("efron".to_string()),
        )
        .expect("efron expected baseline should compute");

        assert_eq!(breslow.0, vec![0]);
        assert_eq!(breslow.1[0], vec![1.0, 2.0]);
        assert!((breslow.2[0][0] - 0.25).abs() < 1e-12);
        assert!((breslow.2[0][1] - 11.0 / 12.0).abs() < 1e-12);
        assert!((breslow.3[0][1] - 41.0 / 144.0).abs() < 1e-12);
        assert!((breslow.4[0][1][0] - 1.0 / 3.0).abs() < 1e-12);
        assert!((efron.2[0][1] - 13.0 / 12.0).abs() < 1e-12);
        assert!((efron.3[0][1] - 61.0 / 144.0).abs() < 1e-12);
        assert!((efron.4[0][1][0] - 13.0 / 24.0).abs() < 1e-12);
    }

    #[test]
    fn expected_baseline_keeps_distinct_times_distinct() {
        // Times are compared exactly, as R's agsurv does (aeqSurv is the
        // caller's job): 1 and 1 + 5e-10 are two event times.
        let time = vec![1.0, 1.0 + 5e-10, 2.0];
        let status = vec![1, 1, 0];
        let covariates = vec![vec![0.0], vec![1.0], vec![2.0]];
        let result = cox_expected_baseline_by_stratum(
            time,
            status,
            covariates,
            vec![0.0],
            vec![1.0; 3],
            vec![0; 3],
            vec![0.0; 3],
            vec![1.0],
            None,
            Some("breslow".to_string()),
        )
        .expect("expected baseline should compute");

        assert_eq!(result.1[0], vec![1.0, 1.0 + 5e-10]);
        assert!((result.2[0][0] - 1.0 / 3.0).abs() < 1e-12);
        assert!((result.2[0][1] - (1.0 / 3.0 + 1.0 / 2.0)).abs() < 1e-12);
        assert!((result.3[0][1] - (1.0 / 9.0 + 1.0 / 4.0)).abs() < 1e-12);
    }

    #[test]
    fn expected_baseline_rejects_invalid_entry_intervals() {
        let (time, status, covariates, beta, weights, strata, offset, means) = baseline_args();
        let err = cox_expected_baseline_by_stratum(
            time,
            status,
            covariates,
            beta,
            weights,
            strata,
            offset,
            means,
            Some(vec![0.0, 2.0, 0.0, 0.0]),
            Some("breslow".to_string()),
        )
        .expect_err("entry time equal to exit time should fail");

        assert!(
            err.to_string()
                .contains("entry_times[1] must be less than time[1]")
        );
    }

    #[test]
    fn tied_summaries_are_returned_column_major() {
        let (sum1, sum2, xbar) = tied_baseline_summaries(
            2,
            2,
            &[1, 2],
            &[10.0, 9.0],
            &[0.0, 1.0],
            vec![10.0, 9.0, 4.0, 3.0],
            vec![0.0, 0.5, 0.0, 0.25],
        )
        .unwrap();
        assert!((sum1[0] - 0.1).abs() < 1e-12);
        assert!((sum2[0] - 0.01).abs() < 1e-12);
        assert!((sum1[1] - (1.0 / 9.0 + 1.0 / 8.5) / 2.0).abs() < 1e-12);
        assert_eq!(xbar.len(), 4);
        assert!((xbar[0] - 10.0 / 100.0).abs() < 1e-12);
        assert!((xbar[2] - 4.0 / 100.0).abs() < 1e-12);
        let expected = (9.0 / 81.0 + (9.0 - 0.25) / (8.5 * 8.5)) / 2.0;
        assert!((xbar[1] - expected).abs() < 1e-12);
        assert!(
            tied_baseline_summaries(1, 1, &[0], &[1.0], &[0.0], vec![1.0], vec![0.0])
                .unwrap_err()
                .to_string()
                .contains("positive event counts")
        );
    }
}
