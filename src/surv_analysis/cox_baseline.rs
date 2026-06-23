use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::BTreeMap;

use crate::constants::TIME_EPSILON;

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

fn sorted_unique_event_times(indices: &[usize], time: &[f64], status: &[i32]) -> Vec<f64> {
    let mut event_times: Vec<f64> = indices
        .iter()
        .filter_map(|&idx| (status[idx] == 1).then_some(time[idx]))
        .collect();
    event_times.sort_by(f64::total_cmp);
    event_times.dedup_by(|left, right| *left == *right);
    event_times
}

#[allow(clippy::too_many_arguments)]
fn accumulate_expected_baseline_stratum(
    indices: &[usize],
    event_times: &[f64],
    time: &[f64],
    status: &[i32],
    centered_rows: &[Vec<f64>],
    risk_weights: &[f64],
    weights: &[f64],
    entry_times: Option<&[f64]>,
    method: &str,
    nvar: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let mut out_times = Vec::with_capacity(event_times.len());
    let mut out_hazard = Vec::with_capacity(event_times.len());
    let mut out_varhaz = Vec::with_capacity(event_times.len());
    let mut out_xbar = Vec::with_capacity(event_times.len());
    let mut cumulative_hazard = 0.0;
    let mut cumulative_varhaz = 0.0;
    let mut cumulative_xbar = vec![0.0; nvar];

    for &event_time in event_times {
        let at_risk: Vec<usize> = indices
            .iter()
            .copied()
            .filter(|&idx| {
                time[idx] >= event_time && entry_times.is_none_or(|entry| entry[idx] < event_time)
            })
            .collect();
        let deaths: Vec<usize> = at_risk
            .iter()
            .copied()
            .filter(|&idx| status[idx] == 1 && (time[idx] - event_time).abs() < TIME_EPSILON)
            .collect();
        if deaths.is_empty() {
            continue;
        }

        let event_weight: f64 = deaths.iter().map(|&idx| weights[idx]).sum();
        let denom: f64 = at_risk.iter().map(|&idx| risk_weights[idx]).sum();
        let (hazard, varhaz, xbar_increment) = if denom <= 0.0 {
            (0.0, 0.0, vec![0.0; nvar])
        } else {
            let mut risk_xsum = vec![0.0; nvar];
            for &idx in &at_risk {
                for (col_idx, value) in risk_xsum.iter_mut().enumerate() {
                    *value += risk_weights[idx] * centered_rows[idx][col_idx];
                }
            }

            if method == "efron" && deaths.len() > 1 {
                let death_risk: f64 = deaths.iter().map(|&idx| risk_weights[idx]).sum();
                let mut death_xsum = vec![0.0; nvar];
                for &idx in &deaths {
                    for (col_idx, value) in death_xsum.iter_mut().enumerate() {
                        *value += risk_weights[idx] * centered_rows[idx][col_idx];
                    }
                }

                let step_weight = event_weight / deaths.len() as f64;
                let mut hazard = 0.0;
                let mut varhaz = 0.0;
                let mut xbar_increment = vec![0.0; nvar];
                for step in 0..deaths.len() {
                    let fraction = step as f64 / deaths.len() as f64;
                    let step_denom = denom - fraction * death_risk;
                    if step_denom <= 0.0 {
                        continue;
                    }
                    hazard += step_weight / step_denom;
                    varhaz += step_weight / (step_denom * step_denom);
                    for col_idx in 0..nvar {
                        let step_xsum = risk_xsum[col_idx] - fraction * death_xsum[col_idx];
                        xbar_increment[col_idx] +=
                            step_weight * step_xsum / (step_denom * step_denom);
                    }
                }
                (hazard, varhaz, xbar_increment)
            } else {
                let hazard = event_weight / denom;
                let varhaz = event_weight / (denom * denom);
                let xbar_increment = risk_xsum
                    .iter()
                    .map(|&value| hazard * value / denom)
                    .collect();
                (hazard, varhaz, xbar_increment)
            }
        };

        cumulative_hazard += hazard;
        cumulative_varhaz += varhaz;
        for col_idx in 0..nvar {
            cumulative_xbar[col_idx] += xbar_increment[col_idx];
        }
        out_times.push(event_time);
        out_hazard.push(cumulative_hazard);
        out_varhaz.push(cumulative_varhaz);
        out_xbar.push(cumulative_xbar.clone());
    }

    (out_times, out_hazard, out_varhaz, out_xbar)
}

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
    let method = method.unwrap_or_else(|| "breslow".to_string());
    if method != "breslow" && method != "efron" && method != "exact" {
        return Err(value_error("method must be 'breslow', 'efron', or 'exact'"));
    }

    let centered_rows: Vec<Vec<f64>> = covariates
        .iter()
        .map(|row| {
            row.iter()
                .zip(means.iter())
                .map(|(&value, &mean)| value - mean)
                .collect()
        })
        .collect();
    let risk_weights: Vec<f64> = covariates
        .iter()
        .enumerate()
        .map(|(idx, row)| {
            let linear_predictor = offset[idx]
                + row
                    .iter()
                    .zip(beta.iter())
                    .map(|(&value, &coefficient)| value * coefficient)
                    .sum::<f64>();
            weights[idx] * safe_exp(linear_predictor)
        })
        .collect();

    let mut indices_by_stratum: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
    for (idx, &stratum) in strata.iter().enumerate() {
        indices_by_stratum.entry(stratum).or_default().push(idx);
    }

    let mut out_strata = Vec::with_capacity(indices_by_stratum.len());
    let mut out_times = Vec::with_capacity(indices_by_stratum.len());
    let mut out_hazard = Vec::with_capacity(indices_by_stratum.len());
    let mut out_varhaz = Vec::with_capacity(indices_by_stratum.len());
    let mut out_xbar = Vec::with_capacity(indices_by_stratum.len());

    for (stratum, indices) in indices_by_stratum {
        let event_times = sorted_unique_event_times(&indices, &time, &status);
        let (times, cumhaz, varhaz, xbar) = accumulate_expected_baseline_stratum(
            &indices,
            &event_times,
            &time,
            &status,
            &centered_rows,
            &risk_weights,
            &weights,
            entry_times.as_deref(),
            &method,
            nvar,
        );
        out_strata.push(stratum);
        out_times.push(times);
        out_hazard.push(cumhaz);
        out_varhaz.push(varhaz);
        out_xbar.push(xbar);
    }

    Ok((out_strata, out_times, out_hazard, out_varhaz, out_xbar))
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
}
