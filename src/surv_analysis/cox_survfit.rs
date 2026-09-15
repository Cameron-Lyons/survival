//! `cox_survfit_baseline`: the legacy one-stratum `agsurv` binding of the
//! Python facade (`R/agsurv.R` on a `y` matrix with 2 or 3 columns).  The
//! argument checks are the original binding's; the curve is
//! [`crate::surv_analysis::agsurv`].

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::internal::numpy_utils::{extract_matrix_f64, extract_vec_f64};
use crate::surv_analysis::agsurv::{AgsurvCurve, AgsurvData, CoxSurvType, agsurv};

use super::cox_baseline::rows_to_array;

fn value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn validate_inputs(
    y: &[Vec<f64>],
    x: &[Vec<f64>],
    weights: &[f64],
    risk: &[f64],
) -> PyResult<(usize, usize)> {
    let n = y.len();
    if n == 0 {
        return Err(value_error("y must contain at least one row"));
    }
    let ycols = y[0].len();
    if ycols != 2 && ycols != 3 {
        return Err(value_error("y must have 2 or 3 columns"));
    }
    for (idx, row) in y.iter().enumerate() {
        if row.len() != ycols {
            return Err(value_error(format!("y row {idx} has inconsistent width")));
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err(value_error(format!(
                "y row {idx} contains a non-finite value"
            )));
        }
        let status = row[ycols - 1];
        if status != 0.0 && status != 1.0 {
            return Err(value_error(format!(
                "y status must contain only 0/1 values; got {status} at row {idx}"
            )));
        }
        if ycols == 3 && row[0] >= row[1] {
            return Err(value_error(format!(
                "y start must be less than stop at row {idx}"
            )));
        }
    }

    if x.len() != n {
        return Err(value_error(format!(
            "x must contain {n} rows; got {}",
            x.len()
        )));
    }
    let nvar = x.first().map_or(0, Vec::len);
    for (idx, row) in x.iter().enumerate() {
        if row.len() != nvar {
            return Err(value_error(format!("x row {idx} has inconsistent width")));
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err(value_error(format!(
                "x row {idx} contains a non-finite value"
            )));
        }
    }

    if weights.len() != n {
        return Err(value_error(format!(
            "weights length must be {n}; got {}",
            weights.len()
        )));
    }
    if risk.len() != n {
        return Err(value_error(format!(
            "risk length must be {n}; got {}",
            risk.len()
        )));
    }
    for (idx, &weight) in weights.iter().enumerate() {
        if !weight.is_finite() || weight < 0.0 {
            return Err(value_error(format!(
                "weights must be finite and non-negative; got {weight} at row {idx}"
            )));
        }
    }
    for (idx, &value) in risk.iter().enumerate() {
        if !value.is_finite() || value <= 0.0 {
            return Err(value_error(format!(
                "risk must be finite and positive; got {value} at row {idx}"
            )));
        }
    }
    Ok((ycols, nvar))
}

fn survtype_code(name: &str, code: i32) -> PyResult<CoxSurvType> {
    CoxSurvType::from_code(code).ok_or_else(|| value_error(format!("{name} must be 1, 2, or 3")))
}

/// The curve of the whole data as one stratum.
fn baseline_curve(
    y: &[Vec<f64>],
    x: &[Vec<f64>],
    weights: &[f64],
    risk: &[f64],
    survtype: i32,
    vartype: i32,
) -> PyResult<AgsurvCurve> {
    let (ycols, nvar) = validate_inputs(y, x, weights, risk)?;
    let survtype = survtype_code("survtype", survtype)?;
    let vartype = survtype_code("vartype", vartype)?;
    let start: Option<Vec<f64>> = (ycols == 3).then(|| y.iter().map(|row| row[0]).collect());
    let stop: Vec<f64> = y.iter().map(|row| row[ycols - 2]).collect();
    let status: Vec<i32> = y.iter().map(|row| row[ycols - 1] as i32).collect();
    let x = rows_to_array(x, nvar);
    let curve = agsurv(
        &AgsurvData {
            start: start.as_deref(),
            stop: &stop,
            status: &status,
            x: x.view(),
            means: None,
            weights,
            risk,
        },
        survtype,
        vartype,
    )?;
    // Zero weights are allowed here (R's coxph rejects them); a risk set of
    // zero total weight has no hazard, which the original binding refused.
    if let Some(g) = curve.hazard.iter().position(|h| h.is_nan()) {
        return Err(value_error(format!(
            "risk-set denominator must be positive at time {}",
            curve.time[g]
        )));
    }
    Ok(curve)
}

/// `agsurv(y, x, weights, risk, survtype, vartype)` for one stratum: the
/// unique times with their counts, hazard, cumulative hazard, hazard
/// variance, number of deaths, `xbar` rows and (survtype 1) the
/// Kalbfleisch-Prentice increments.
#[pyfunction]
pub fn cox_survfit_baseline(
    y: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    weights: &Bound<'_, PyAny>,
    risk: &Bound<'_, PyAny>,
    survtype: i32,
    vartype: i32,
) -> PyResult<Py<PyDict>> {
    let y = extract_matrix_f64(y)?;
    let x = extract_matrix_f64(x)?;
    let weights = extract_vec_f64(weights)?;
    let risk = extract_vec_f64(risk)?;
    let curve = baseline_curve(&y, &x, &weights, &risk, survtype, vartype)?;
    Python::attach(|py| {
        let dict = PyDict::new(py);
        dict.set_item("n", curve.n)?;
        dict.set_item("time", curve.time)?;
        dict.set_item("n_event", curve.n_event)?;
        dict.set_item("n_risk", curve.n_risk)?;
        dict.set_item("n_censor", curve.n_censor)?;
        dict.set_item("hazard", curve.hazard)?;
        dict.set_item("cumhaz", curve.cumhaz)?;
        dict.set_item("varhaz", curve.varhaz)?;
        let ndeath: Vec<i32> = curve.ndeath.iter().map(|&d| d as i32).collect();
        dict.set_item("ndeath", ndeath)?;
        let xbar: Vec<Vec<f64>> = curve.xbar.outer_iter().map(|row| row.to_vec()).collect();
        dict.set_item("xbar", xbar)?;
        if let Some(surv) = curve.surv {
            dict.set_item("surv", surv)?;
        }
        Ok(dict.into())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn right_censored_baseline_matches_weighted_risk_sets() {
        let result = baseline_curve(
            &[
                vec![1.0, 1.0],
                vec![2.0, 1.0],
                vec![2.0, 0.0],
                vec![3.0, 1.0],
            ],
            &[vec![0.0], vec![1.0], vec![2.0], vec![3.0]],
            &[1.0, 2.0, 1.0, 1.0],
            &[1.0, 2.0, 1.0, 0.5],
            2,
            2,
        )
        .unwrap();

        assert_eq!(result.time, vec![1.0, 2.0, 3.0]);
        assert_eq!(result.n_event, vec![1.0, 2.0, 1.0]);
        assert_eq!(result.n_censor, vec![0.0, 1.0, 0.0]);
        assert_eq!(result.n_risk, vec![5.0, 4.0, 1.0]);
        assert_eq!(result.ndeath, vec![1, 1, 1]);
        assert!((result.hazard[0] - 1.0 / 6.5).abs() < 1e-12);
        assert!((result.hazard[1] - 2.0 / 5.5).abs() < 1e-12);
        assert!((result.hazard[2] - 2.0).abs() < 1e-12);
        assert!((result.xbar[(0, 0)] - 7.5 / 6.5_f64.powi(2)).abs() < 1e-12);
    }

    #[test]
    fn counting_baseline_excludes_rows_before_entry() {
        let result = baseline_curve(
            &[
                vec![0.0, 2.0, 1.0],
                vec![1.0, 3.0, 1.0],
                vec![2.0, 4.0, 0.0],
            ],
            &[vec![0.0], vec![1.0], vec![2.0]],
            &[1.0, 1.0, 1.0],
            &[1.0, 2.0, 4.0],
            3,
            3,
        )
        .unwrap();

        assert_eq!(result.n_risk, vec![2.0, 2.0, 1.0]);
        assert!((result.hazard[0] - 1.0 / 3.0).abs() < 1e-12);
        assert!((result.hazard[1] - 1.0 / 6.0).abs() < 1e-12);
        assert_eq!(result.hazard[2], 0.0);
    }

    #[test]
    fn kalbfleisch_prentice_increments_are_reported() {
        let result = baseline_curve(
            &[vec![1.0, 1.0], vec![2.0, 0.0]],
            &[vec![0.0], vec![1.0]],
            &[1.0, 1.0],
            &[2.0, 1.0],
            1,
            1,
        )
        .unwrap();
        let surv = result.surv.unwrap();
        assert!((surv[0] - (1.0 - 2.0 / 3.0_f64).powf(0.5)).abs() < 1e-12);
        assert_eq!(surv[1], 1.0);
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        let error =
            baseline_curve(&[vec![1.0, 1.0, 0.0]], &[vec![0.0]], &[1.0], &[1.0], 2, 2).unwrap_err();
        assert!(error.to_string().contains("start must be less than stop"));
        let error =
            baseline_curve(&[vec![1.0, 1.0]], &[vec![0.0]], &[1.0], &[1.0], 4, 2).unwrap_err();
        assert!(error.to_string().contains("survtype must be 1, 2, or 3"));
        let error =
            baseline_curve(&[vec![1.0, 1.0]], &[vec![0.0]], &[0.0], &[1.0], 2, 2).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("risk-set denominator must be positive")
        );
    }
}
