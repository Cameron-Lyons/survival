use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

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
            if contribution > denom[time_index] {
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

#[pyfunction]
pub fn compute_baseline_survival_steps(
    ndeath: Vec<i32>,
    risk: Vec<f64>,
    wt: Vec<f64>,
    sn: usize,
    denom: Vec<f64>,
) -> PyResult<Vec<f64>> {
    validate_baseline_survival_steps_inputs(&ndeath, &risk, &wt, sn, &denom)?;
    let ndeath_slice = &ndeath;
    let risk_slice = &risk;
    let wt_slice = &wt;
    let denom_slice = &denom;
    let mut km = vec![0.0; sn];
    let n = sn;
    let mut j = 0;
    for i in 0..n {
        match ndeath_slice[i] {
            0 => km[i] = 1.0,
            1 => {
                let numerator = wt_slice[j] * risk_slice[j];
                km[i] = (1.0 - numerator / denom_slice[i]).powf(1.0 / risk_slice[j]);
                j += 1;
            }
            _ => {
                let mut guess: f64 = 0.5;
                let mut inc = 0.25;
                let death_count = ndeath_slice[i] as usize;
                for _ in 0..35 {
                    let mut sumt = 0.0;
                    for k in j..(j + death_count) {
                        let term = wt_slice[k] * risk_slice[k] / (1.0 - guess.powf(risk_slice[k]));
                        sumt += term;
                    }
                    if sumt < denom_slice[i] {
                        guess += inc;
                    } else {
                        guess -= inc;
                    }
                    inc /= 2.0;
                }
                km[i] = guess;
                j += death_count;
            }
        }
    }
    Ok(km)
}

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
    validate_tied_baseline_summaries_inputs(n, nvar, &dd, &x1, &x2, &xsum, &xsum2)?;
    let dd_slice = &dd;
    let x1_slice = &x1;
    let x2_slice = &x2;
    let xsum_slice = &xsum;
    let xsum2_slice = &xsum2;
    let mut sum1 = vec![0.0; n];
    let mut sum2 = vec![0.0; n];
    let mut xbar = vec![0.0; n * nvar];
    for i in 0..n {
        let d = dd_slice[i] as f64;
        if d == 1.0 {
            let temp = 1.0 / x1_slice[i];
            sum1[i] = temp;
            sum2[i] = temp.powi(2);
            for k in 0..nvar {
                let idx = i + n * k;
                xbar[idx] = xsum_slice[idx] * temp.powi(2);
            }
        } else {
            let d_int = dd_slice[i];
            let mut temp;
            for j in 0..d_int {
                let j_f64 = j as f64;
                temp = 1.0 / (x1_slice[i] - x2_slice[i] * j_f64 / d);
                sum1[i] += temp / d;
                sum2[i] += temp.powi(2) / d;
                for k in 0..nvar {
                    let idx = i + n * k;
                    let weighted_x = xsum_slice[idx] - xsum2_slice[idx] * j_f64 / d;
                    xbar[idx] += (weighted_x * temp.powi(2)) / d;
                }
            }
        }
    }
    Python::attach(|py| {
        let dict = PyDict::new(py);
        dict.set_item("sum1", sum1)?;
        dict.set_item("sum2", sum2)?;
        dict.set_item("xbar", xbar)?;
        Ok(dict.into())
    })
}
