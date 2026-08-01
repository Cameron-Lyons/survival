use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::internal::numpy_utils::{extract_matrix_f64, extract_vec_f64};

use super::cox_baseline::compute_baseline_survival_steps;

#[derive(Debug)]
struct CoxSurvfitBaseline {
    n: usize,
    time: Vec<f64>,
    n_event: Vec<f64>,
    n_risk: Vec<f64>,
    n_censor: Vec<f64>,
    hazard: Vec<f64>,
    cumhaz: Vec<f64>,
    varhaz: Vec<f64>,
    ndeath: Vec<i32>,
    xbar: Vec<Vec<f64>>,
    surv: Option<Vec<f64>>,
}

fn value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn validate_inputs(
    y: &[Vec<f64>],
    x: &[Vec<f64>],
    weights: &[f64],
    risk: &[f64],
    survtype: i32,
    vartype: i32,
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
    if !(1..=3).contains(&survtype) {
        return Err(value_error("survtype must be 1, 2, or 3"));
    }
    if !(1..=3).contains(&vartype) {
        return Err(value_error("vartype must be 1, 2, or 3"));
    }
    Ok((ycols, nvar))
}

fn add_row(
    idx: usize,
    x: &[Vec<f64>],
    weights: &[f64],
    weighted_risk: &[f64],
    denominator: &mut f64,
    number_at_risk: &mut f64,
    xsum: &mut [f64],
) {
    *denominator += weighted_risk[idx];
    *number_at_risk += weights[idx];
    for (sum, &value) in xsum.iter_mut().zip(&x[idx]) {
        *sum += weighted_risk[idx] * value;
    }
}

fn remove_row(
    idx: usize,
    x: &[Vec<f64>],
    weights: &[f64],
    weighted_risk: &[f64],
    denominator: &mut f64,
    number_at_risk: &mut f64,
    xsum: &mut [f64],
) {
    *denominator -= weighted_risk[idx];
    *number_at_risk -= weights[idx];
    for (sum, &value) in xsum.iter_mut().zip(&x[idx]) {
        *sum -= weighted_risk[idx] * value;
    }
}

fn compute_baseline(
    y: &[Vec<f64>],
    x: &[Vec<f64>],
    weights: &[f64],
    risk: &[f64],
    survtype: i32,
    vartype: i32,
) -> PyResult<CoxSurvfitBaseline> {
    let (ycols, nvar) = validate_inputs(y, x, weights, risk, survtype, vartype)?;
    let n = y.len();
    let stop_col = ycols - 2;
    let status_col = ycols - 1;
    let stop: Vec<f64> = y.iter().map(|row| row[stop_col]).collect();
    let status: Vec<i32> = y.iter().map(|row| row[status_col] as i32).collect();
    let weighted_risk: Vec<f64> = weights
        .iter()
        .zip(risk)
        .map(|(&weight, &risk_value)| weight * risk_value)
        .collect();

    let mut stop_order: Vec<usize> = (0..n).collect();
    stop_order.sort_by(|&left, &right| {
        stop[left]
            .total_cmp(&stop[right])
            .then_with(|| left.cmp(&right))
    });
    let mut output_times = Vec::new();
    for &idx in &stop_order {
        if output_times.last().copied() != Some(stop[idx]) {
            output_times.push(stop[idx]);
        }
    }

    let entry_order = if ycols == 3 {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&left, &right| {
            y[left][0]
                .total_cmp(&y[right][0])
                .then_with(|| left.cmp(&right))
        });
        Some(order)
    } else {
        None
    };
    let mut active = vec![ycols == 2; n];
    let mut denominator = 0.0;
    let mut number_at_risk = 0.0;
    let mut xsum = vec![0.0; nvar];
    if ycols == 2 {
        for idx in 0..n {
            add_row(
                idx,
                x,
                weights,
                &weighted_risk,
                &mut denominator,
                &mut number_at_risk,
                &mut xsum,
            );
        }
    }

    let ntime = output_times.len();
    let mut n_event = Vec::with_capacity(ntime);
    let mut n_risk = Vec::with_capacity(ntime);
    let mut n_censor = Vec::with_capacity(ntime);
    let mut hazard = Vec::with_capacity(ntime);
    let mut cumhaz = Vec::with_capacity(ntime);
    let mut varhaz = Vec::with_capacity(ntime);
    let mut ndeath = Vec::with_capacity(ntime);
    let mut xbar = Vec::with_capacity(ntime);
    let mut denominators = Vec::with_capacity(ntime);
    let mut death_risk_values = Vec::new();
    let mut death_weights = Vec::new();
    let mut entry_pos = 0;
    let mut expired_pos = 0;
    let mut group_pos = 0;
    let mut cumulative_hazard = 0.0;
    let mut deaths = Vec::new();

    for &time in &output_times {
        if let Some(order) = entry_order.as_ref() {
            while entry_pos < order.len() && y[order[entry_pos]][0] < time {
                let idx = order[entry_pos];
                if !active[idx] {
                    active[idx] = true;
                    add_row(
                        idx,
                        x,
                        weights,
                        &weighted_risk,
                        &mut denominator,
                        &mut number_at_risk,
                        &mut xsum,
                    );
                }
                entry_pos += 1;
            }
        }
        while expired_pos < stop_order.len() && stop[stop_order[expired_pos]] < time {
            let idx = stop_order[expired_pos];
            if active[idx] {
                active[idx] = false;
                remove_row(
                    idx,
                    x,
                    weights,
                    &weighted_risk,
                    &mut denominator,
                    &mut number_at_risk,
                    &mut xsum,
                );
            }
            expired_pos += 1;
        }
        if denominator <= 0.0 || !denominator.is_finite() {
            return Err(value_error(format!(
                "risk-set denominator must be positive at time {time}"
            )));
        }

        let group_start = group_pos;
        while group_pos < stop_order.len() && stop[stop_order[group_pos]] == time {
            group_pos += 1;
        }
        let group = &stop_order[group_start..group_pos];
        deaths.clear();
        deaths.extend(group.iter().copied().filter(|&idx| status[idx] == 1));
        let death_count = i32::try_from(deaths.len())
            .map_err(|_| value_error("number of tied deaths is too large"))?;
        let event_weight: f64 = deaths.iter().map(|&idx| weights[idx]).sum();
        let censor_weight: f64 = group
            .iter()
            .filter(|&&idx| status[idx] == 0)
            .map(|&idx| weights[idx])
            .sum();

        let mut tied_sum1 = 0.0;
        let mut tied_sum2 = 0.0;
        let mut tied_xbar = vec![0.0; nvar];
        if death_count > 0 && (survtype == 3 || vartype == 3) {
            let death_risk: f64 = deaths.iter().map(|&idx| weighted_risk[idx]).sum();
            let mut death_xsum = vec![0.0; nvar];
            for &idx in &deaths {
                for (sum, &value) in death_xsum.iter_mut().zip(&x[idx]) {
                    *sum += weighted_risk[idx] * value;
                }
            }
            let d = f64::from(death_count);
            for step in 0..death_count {
                let fraction = f64::from(step) / d;
                let tied_denom = denominator - fraction * death_risk;
                if tied_denom <= 0.0 || !tied_denom.is_finite() {
                    return Err(value_error(format!(
                        "tied risk-set denominator must be positive at time {time}"
                    )));
                }
                let inverse = 1.0 / tied_denom;
                tied_sum1 += inverse / d;
                tied_sum2 += inverse * inverse / d;
                for col in 0..nvar {
                    tied_xbar[col] +=
                        (xsum[col] - fraction * death_xsum[col]) * inverse * inverse / d;
                }
            }
        }

        let step_hazard = if survtype == 3 {
            event_weight * tied_sum1
        } else {
            event_weight / denominator
        };
        let step_variance = match vartype {
            1 => {
                let remaining = if event_weight >= denominator {
                    denominator
                } else {
                    denominator - event_weight
                };
                event_weight / (denominator * remaining)
            }
            2 => event_weight / (denominator * denominator),
            3 => event_weight * tied_sum2,
            _ => unreachable!(),
        };
        let step_xbar = if vartype == 3 {
            tied_xbar
                .into_iter()
                .map(|value| event_weight * value)
                .collect()
        } else {
            xsum.iter()
                .map(|&value| value / denominator * step_hazard)
                .collect()
        };
        cumulative_hazard += step_hazard;

        n_event.push(event_weight);
        n_risk.push(number_at_risk);
        n_censor.push(censor_weight);
        hazard.push(step_hazard);
        cumhaz.push(cumulative_hazard);
        varhaz.push(step_variance);
        ndeath.push(death_count);
        xbar.push(step_xbar);
        denominators.push(denominator);
        for &idx in &deaths {
            death_risk_values.push(risk[idx]);
            death_weights.push(weights[idx]);
        }
    }

    let surv = if survtype == 1 {
        Some(compute_baseline_survival_steps(
            ndeath.clone(),
            death_risk_values,
            death_weights,
            ntime,
            denominators,
        )?)
    } else {
        None
    };

    Ok(CoxSurvfitBaseline {
        n,
        time: output_times,
        n_event,
        n_risk,
        n_censor,
        hazard,
        cumhaz,
        varhaz,
        ndeath,
        xbar,
        surv,
    })
}

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
    let result = compute_baseline(&y, &x, &weights, &risk, survtype, vartype)?;
    Python::attach(|py| {
        let dict = PyDict::new(py);
        dict.set_item("n", result.n)?;
        dict.set_item("time", result.time)?;
        dict.set_item("n_event", result.n_event)?;
        dict.set_item("n_risk", result.n_risk)?;
        dict.set_item("n_censor", result.n_censor)?;
        dict.set_item("hazard", result.hazard)?;
        dict.set_item("cumhaz", result.cumhaz)?;
        dict.set_item("varhaz", result.varhaz)?;
        dict.set_item("ndeath", result.ndeath)?;
        dict.set_item("xbar", result.xbar)?;
        if let Some(surv) = result.surv {
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
        let result = compute_baseline(
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
        assert!((result.hazard[0] - 1.0 / 6.5).abs() < 1e-12);
        assert!((result.hazard[1] - 2.0 / 5.5).abs() < 1e-12);
        assert!((result.hazard[2] - 2.0).abs() < 1e-12);
        assert!((result.xbar[0][0] - 7.5 / 6.5_f64.powi(2)).abs() < 1e-12);
    }

    #[test]
    fn counting_baseline_excludes_rows_before_entry() {
        let result = compute_baseline(
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
    fn invalid_intervals_are_rejected() {
        Python::initialize();
        let error = compute_baseline(&[vec![1.0, 1.0, 0.0]], &[vec![0.0]], &[1.0], &[1.0], 2, 2)
            .unwrap_err();
        assert!(error.to_string().contains("start must be less than stop"));
    }
}
