use super::pystep::pystep;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

struct RateTable<'a> {
    factors: &'a [i32],
    dims: &'a [usize],
    cuts: Vec<&'a [f64]>,
    rates: &'a [f64],
}

fn checked_product(dims: &[usize]) -> PyResult<usize> {
    if dims.is_empty() || dims.contains(&0) {
        return Err(PyValueError::new_err(
            "expected_dims must contain positive dimensions",
        ));
    }
    dims.iter().try_fold(1usize, |product, &dim| {
        product
            .checked_mul(dim)
            .ok_or_else(|| PyValueError::new_err("expected table is too large"))
    })
}

fn validate_finite(name: &str, values: &[f64]) -> PyResult<()> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "{name} contains a non-finite value at index {index}"
        )));
    }
    Ok(())
}

fn prepare_table<'a>(
    factors: &'a [i32],
    dims: &'a [usize],
    cuts: &'a [f64],
    rates: &'a [f64],
) -> PyResult<RateTable<'a>> {
    if factors.len() != dims.len() {
        return Err(PyValueError::new_err(
            "expected_factors and expected_dims must have equal length",
        ));
    }
    if factors
        .iter()
        .take(factors.len().saturating_sub(1))
        .any(|&factor| factor > 1)
    {
        return Err(PyValueError::new_err(
            "only the final expected dimension may be interpolated",
        ));
    }
    let table_size = checked_product(dims)?;
    if rates.len() != table_size {
        return Err(PyValueError::new_err(format!(
            "expected_rates must have length {table_size}"
        )));
    }
    validate_finite("expected_rates", rates)?;

    let mut cut_slices = Vec::with_capacity(dims.len());
    let mut offset = 0usize;
    for (dimension, (&factor, &dim)) in factors.iter().zip(dims).enumerate() {
        if factor < 0 {
            return Err(PyValueError::new_err(format!(
                "invalid expected factor type at dimension {dimension}"
            )));
        }
        let count = match factor {
            1 => 0,
            0 => dim,
            _ => (factor as usize - 1)
                .checked_mul(dim)
                .and_then(|count| count.checked_add(1))
                .ok_or_else(|| PyValueError::new_err("expected cutpoint count is too large"))?,
        };
        let end = offset
            .checked_add(count)
            .ok_or_else(|| PyValueError::new_err("expected cutpoint count is too large"))?;
        if end > cuts.len() {
            return Err(PyValueError::new_err(
                "expected_cuts does not contain enough cutpoints",
            ));
        }
        let dimension_cuts = &cuts[offset..end];
        validate_finite("expected_cuts", dimension_cuts)?;
        if dimension_cuts.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(PyValueError::new_err(
                "expected cutpoints must be strictly increasing",
            ));
        }
        cut_slices.push(dimension_cuts);
        offset = end;
    }
    if offset != cuts.len() {
        return Err(PyValueError::new_err(
            "expected_cuts contains unused cutpoints",
        ));
    }

    Ok(RateTable {
        factors,
        dims,
        cuts: cut_slices,
        rates,
    })
}

#[allow(clippy::too_many_arguments)]
fn survexp_fit(
    conditional: bool,
    table: &RateTable<'_>,
    groups: &[usize],
    expected_data: &[f64],
    followup: &[f64],
    times: &[f64],
    n_groups: usize,
) -> (Vec<f64>, Vec<usize>) {
    let n = followup.len();
    let n_times = times.len();
    let output_len = n_times * n_groups;
    let mut interval_survival = vec![0.0; output_len];
    let mut denominators = vec![0.0; output_len];
    let mut n_risk = vec![0usize; output_len];
    let mut position = vec![0.0; table.dims.len()];

    for subject in 0..n {
        for (dimension, value) in position.iter_mut().enumerate() {
            *value = expected_data[dimension * n + subject];
        }
        let mut cumulative_hazard: f64 = 0.0;
        let mut time_left = followup[subject];
        let mut time = 0.0;
        let group_offset = (groups[subject] - 1) * n_times;

        for (time_index, &output_time) in times.iter().enumerate() {
            if time_left <= 0.0 {
                break;
            }
            let interval = (output_time - time).min(time_left);
            let output_index = group_offset + time_index;
            let mut remaining = interval;
            let mut interval_hazard = 0.0;

            while remaining > 0.0 {
                let (step, index, next_index, weight) = pystep(
                    table.dims.len(),
                    &position,
                    table.factors,
                    table.dims,
                    &table.cuts,
                    remaining,
                    true,
                );
                debug_assert!(step > 0.0 && index >= 0 && next_index >= 0);
                let rate = if weight < 1.0 {
                    weight * table.rates[index as usize]
                        + (1.0 - weight) * table.rates[next_index as usize]
                } else {
                    table.rates[index as usize]
                };
                interval_hazard += step * rate;
                for (dimension, value) in position.iter_mut().enumerate() {
                    if table.factors[dimension] != 1 {
                        *value += step;
                    }
                }
                remaining -= step;
            }

            if output_time == 0.0 {
                denominators[output_index] = 1.0;
                interval_survival[output_index] = if conditional { 0.0 } else { 1.0 };
            } else if conditional {
                interval_survival[output_index] += interval_hazard * interval;
                denominators[output_index] += interval;
            } else {
                interval_survival[output_index] +=
                    (-(cumulative_hazard + interval_hazard)).exp() * interval;
                denominators[output_index] += (-cumulative_hazard).exp() * interval;
            }
            n_risk[output_index] += 1;
            cumulative_hazard += interval_hazard;
            time += interval;
            time_left -= interval;
        }
    }

    for (survival, denominator) in interval_survival.iter_mut().zip(denominators) {
        if denominator > 0.0 {
            if conditional {
                *survival = (-*survival / denominator).exp();
            } else {
                *survival /= denominator;
            }
        } else if conditional {
            *survival = (-*survival).exp();
        }
    }
    (interval_survival, n_risk)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn perform_survexp_fit(
    conditional: bool,
    expected_factors: Vec<i32>,
    expected_dims: Vec<usize>,
    expected_cuts: Vec<f64>,
    expected_rates: Vec<f64>,
    groups: Vec<usize>,
    expected_data: Vec<f64>,
    followup: Vec<f64>,
    times: Vec<f64>,
    n_groups: usize,
) -> PyResult<Py<PyAny>> {
    let n = followup.len();
    if n == 0 {
        return Err(PyValueError::new_err("followup must not be empty"));
    }
    if n_groups == 0 {
        return Err(PyValueError::new_err("n_groups must be positive"));
    }
    if groups.len() != n {
        return Err(PyValueError::new_err(
            "groups and followup must have equal length",
        ));
    }
    if let Some(index) = groups
        .iter()
        .position(|&group| group == 0 || group > n_groups)
    {
        return Err(PyValueError::new_err(format!(
            "groups contains an out-of-range value at index {index}"
        )));
    }
    validate_finite("followup", &followup)?;
    if followup.iter().any(|&value| value < 0.0) {
        return Err(PyValueError::new_err("followup must be non-negative"));
    }
    if times.is_empty() {
        return Err(PyValueError::new_err("times must not be empty"));
    }
    validate_finite("times", &times)?;
    if times[0] < 0.0 || times.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(PyValueError::new_err(
            "times must be non-negative and strictly increasing",
        ));
    }
    let table = prepare_table(
        &expected_factors,
        &expected_dims,
        &expected_cuts,
        &expected_rates,
    )?;
    let expected_len = n
        .checked_mul(table.dims.len())
        .ok_or_else(|| PyValueError::new_err("expected_data is too large"))?;
    if expected_data.len() != expected_len {
        return Err(PyValueError::new_err(format!(
            "expected_data must have length {expected_len}"
        )));
    }
    validate_finite("expected_data", &expected_data)?;
    for (dimension, &factor) in table.factors.iter().enumerate() {
        if factor == 1 {
            for subject in 0..n {
                let value = expected_data[dimension * n + subject];
                if value.fract() != 0.0 || value < 1.0 || value > table.dims[dimension] as f64 {
                    return Err(PyValueError::new_err(format!(
                        "expected factor codes must be integers between 1 and {}",
                        table.dims[dimension]
                    )));
                }
            }
        }
    }
    let output_len = times
        .len()
        .checked_mul(n_groups)
        .ok_or_else(|| PyValueError::new_err("survexp output is too large"))?;
    if output_len == 0 {
        return Err(PyValueError::new_err("survexp output is empty"));
    }

    let (survival, n_risk) = survexp_fit(
        conditional,
        &table,
        &groups,
        &expected_data,
        &followup,
        &times,
        n_groups,
    );
    Python::attach(|py| {
        let result = PyDict::new(py);
        result.set_item("surv", survival)?;
        result.set_item("n", n_risk)?;
        Ok(result.into())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_table<'a>(
        rates: &'a [f64],
        dims: &'a [usize],
        cuts: Vec<&'a [f64]>,
    ) -> RateTable<'a> {
        RateTable {
            factors: &[0],
            dims,
            cuts,
            rates,
        }
    }

    #[test]
    fn constant_rate_returns_interval_ratios_and_censoring_counts() {
        let table = simple_table(&[0.1], &[1], vec![&[0.0]]);
        let (survival, n_risk) = survexp_fit(
            false,
            &table,
            &[1, 1],
            &[0.0, 0.0],
            &[3.0, 2.0],
            &[1.0, 2.0, 3.0],
            1,
        );
        for value in survival {
            assert!((value - (-0.1f64).exp()).abs() < 1e-14);
        }
        assert_eq!(n_risk, vec![2, 2, 1]);
    }

    #[test]
    fn cohort_and_conditional_reductions_are_distinct() {
        let table = simple_table(&[0.1, 0.3], &[2], vec![&[0.0, 1.0]]);
        let data = [0.0, 1.0];
        let cohort = survexp_fit(false, &table, &[1, 1], &data, &[1.0, 1.0], &[1.0], 1).0;
        let conditional = survexp_fit(true, &table, &[1, 1], &data, &[1.0, 1.0], &[1.0], 1).0;
        let expected_cohort = ((-0.1f64).exp() + (-0.3f64).exp()) / 2.0;
        assert!((cohort[0] - expected_cohort).abs() < 1e-14);
        assert!((conditional[0] - (-0.2f64).exp()).abs() < 1e-14);
    }

    #[test]
    fn groups_are_stored_in_column_major_curve_order() {
        let table = simple_table(&[0.1, 0.3], &[2], vec![&[0.0, 1.0]]);
        let (survival, n_risk) =
            survexp_fit(false, &table, &[1, 2], &[0.0, 1.0], &[1.0, 1.0], &[1.0], 2);
        assert!((survival[0] - (-0.1f64).exp()).abs() < 1e-14);
        assert!((survival[1] - (-0.3f64).exp()).abs() < 1e-14);
        assert_eq!(n_risk, vec![1, 1]);
    }

    #[test]
    fn interpolated_dimensions_blend_adjacent_rates() {
        let factors = [2];
        let dims = [2];
        let cut_values = [0.0, 1.0, 2.0];
        let table = RateTable {
            factors: &factors,
            dims: &dims,
            cuts: vec![&cut_values],
            rates: &[0.1, 0.3],
        };
        let survival = survexp_fit(true, &table, &[1], &[1.0], &[1.0], &[1.0], 1).0;
        assert!((survival[0] - (-0.2f64).exp()).abs() < 1e-14);
    }
}
