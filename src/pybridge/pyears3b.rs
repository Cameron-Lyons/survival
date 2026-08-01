use super::pystep::pystep;
use crate::constants::PYEARS_TIME_EPSILON;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub(crate) struct PyearsExpectedParams<'a> {
    pub dim: usize,
    pub fac: &'a [i32],
    pub dims: &'a [usize],
    pub cut: &'a [f64],
    pub rates: &'a [f64],
    pub data: &'a [f64],
}

pub(crate) struct PyearsObservedParams<'a> {
    pub dim: usize,
    pub fac: &'a [i32],
    pub dims: &'a [usize],
    pub cut: &'a [f64],
    pub data: &'a [f64],
}

pub(crate) struct PyearsOutput<'a> {
    pub pyears: &'a mut [f64],
    pub pn: &'a mut [f64],
    pub pcount: &'a mut [f64],
    pub pexpect: &'a mut [f64],
    pub offtable: &'a mut f64,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn pyears3b(
    n: usize,
    ny: usize,
    doevent: i32,
    y: &[f64],
    weight: &[f64],
    expected: PyearsExpectedParams<'_>,
    observed: PyearsObservedParams<'_>,
    method: i32,
    output: &mut PyearsOutput<'_>,
) {
    let PyearsExpectedParams {
        dim: edim,
        fac: efac,
        dims: edims,
        cut: ecut,
        rates: expect,
        data: edata,
    } = expected;
    let PyearsObservedParams {
        dim: odim,
        fac: ofac,
        dims: odims,
        cut: ocut,
        data: odata,
    } = observed;
    let PyearsOutput {
        pyears,
        pn,
        pcount,
        pexpect,
        offtable,
    } = output;
    let (start, stop, event) = if ny == 3 || (ny == 2 && doevent == 0) {
        let start = &y[0..n];
        let stop = &y[n..2 * n];
        let event = if ny == 3 { &y[2 * n..3 * n] } else { &[] };
        (start, stop, event)
    } else {
        let stop = &y[0..n];
        let event = if doevent == 1 { &y[n..2 * n] } else { &[] };
        (&[] as &[f64], stop, event)
    };
    let mut ecut_slices = Vec::with_capacity(edim);
    let mut ecut_ptr = ecut;
    for j in 0..edim {
        let len = if efac[j] == 0 {
            edims[j]
        } else if efac[j] > 1 {
            1 + (efac[j] - 1) as usize * edims[j]
        } else {
            0
        };
        if len > 0 {
            ecut_slices.push(&ecut_ptr[0..len]);
            ecut_ptr = &ecut_ptr[len..];
        } else {
            ecut_slices.push(&[]);
        }
    }
    let mut ocut_slices = Vec::with_capacity(odim);
    let mut ocut_ptr = ocut;
    for j in 0..odim {
        if ofac[j] == 0 {
            let len = odims[j] + 1;
            ocut_slices.push(&ocut_ptr[0..len]);
            ocut_ptr = &ocut_ptr[len..];
        } else {
            ocut_slices.push(&[]);
        }
    }
    let mut eps = 0.0;
    for i in 0..n {
        let timeleft = if start.is_empty() {
            stop[i]
        } else {
            stop[i] - start[i]
        };
        if timeleft > 0.0 {
            eps = timeleft;
            break;
        }
    }
    for i in 0..n {
        let timeleft = if start.is_empty() {
            stop[i]
        } else {
            stop[i] - start[i]
        };
        if timeleft > 0.0 && timeleft < eps {
            eps = timeleft;
        }
    }
    eps *= PYEARS_TIME_EPSILON;
    **offtable = 0.0;
    let mut data = vec![0.0; odim];
    let mut data2 = vec![0.0; edim];
    for i in 0..n {
        for j in 0..odim {
            if ofac[j] == 1 || start.is_empty() {
                data[j] = odata[j * n + i];
            } else {
                data[j] = odata[j * n + i] + start[i];
            }
        }
        for j in 0..edim {
            if efac[j] == 1 || start.is_empty() {
                data2[j] = edata[j * n + i];
            } else {
                data2[j] = edata[j * n + i] + start[i];
            }
        }
        let mut timeleft = if start.is_empty() {
            stop[i]
        } else {
            stop[i] - start[i]
        };
        let mut cumhaz: f64 = 0.0;
        let mut index = -1isize;

        if timeleft <= eps && doevent == 1 {
            let (_, current_index, _, _) =
                pystep(odim, &data, ofac, odims, &ocut_slices, 1.0, false);
            index = current_index;
        }

        while timeleft > eps {
            let (thiscell, current_index, _, _) =
                pystep(odim, &data, ofac, odims, &ocut_slices, timeleft, false);
            index = current_index;

            if index >= 0 {
                let output_index = index as usize;
                pyears[output_index] += thiscell * weight[i];
                pn[output_index] += 1.0;

                if edim > 0 {
                    let mut etime = thiscell;
                    let mut hazard: f64 = 0.0;
                    let mut temp = 0.0;
                    while etime > 0.0 {
                        let (et2, expected_index, expected_index2, expected_weight) =
                            pystep(edim, &data2, efac, edims, &ecut_slices, etime, true);
                        let expected_index = expected_index as usize;
                        let lambda = if expected_weight < 1.0 {
                            expected_weight * expect[expected_index]
                                + (1.0 - expected_weight) * expect[expected_index2 as usize]
                        } else {
                            expect[expected_index]
                        };
                        if method == 0 {
                            let interval_survival_loss = if lambda == 0.0 {
                                et2
                            } else {
                                -(-lambda * et2).exp_m1() / lambda
                            };
                            temp += (-hazard).exp() * interval_survival_loss;
                        }
                        hazard += lambda * et2;
                        for j in 0..edim {
                            if efac[j] != 1 {
                                data2[j] += et2;
                            }
                        }
                        etime -= et2;
                    }

                    if method == 1 {
                        pexpect[output_index] += hazard * weight[i];
                    } else {
                        pexpect[output_index] += (-cumhaz).exp() * temp * weight[i];
                    }
                    cumhaz += hazard;
                }
            } else {
                **offtable += thiscell * weight[i];
                for j in 0..edim {
                    if efac[j] != 1 {
                        data2[j] += thiscell;
                    }
                }
            }

            for j in 0..odim {
                if ofac[j] == 0 {
                    data[j] += thiscell;
                }
            }
            timeleft -= thiscell;
        }

        if index >= 0 && doevent == 1 && !event.is_empty() {
            pcount[index as usize] += event[i] * weight[i];
        }
    }
}

fn checked_dimension_product(name: &str, dims: &[usize]) -> PyResult<usize> {
    if dims.contains(&0) {
        return Err(PyValueError::new_err(format!(
            "{name} dimensions must be positive"
        )));
    }
    dims.iter().try_fold(1usize, |product, &dim| {
        product
            .checked_mul(dim)
            .ok_or_else(|| PyValueError::new_err(format!("{name} dimension product is too large")))
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

struct TableLayout<'a> {
    name: &'a str,
    n: usize,
    dim: usize,
    factors: &'a [i32],
    dims: &'a [usize],
    cuts: &'a [f64],
    data: &'a [f64],
    observed: bool,
}

fn validate_table_layout(layout: TableLayout<'_>) -> PyResult<usize> {
    let TableLayout {
        name,
        n,
        dim,
        factors,
        dims,
        cuts,
        data,
        observed,
    } = layout;
    if dim == 0
        && !observed
        && factors.is_empty()
        && dims.is_empty()
        && cuts.is_empty()
        && data.is_empty()
    {
        return Ok(0);
    }
    if dim == 0 {
        return Err(PyValueError::new_err(format!(
            "{name}_dim must be positive"
        )));
    }
    if factors.len() != dim || dims.len() != dim {
        return Err(PyValueError::new_err(format!(
            "{name} factors and dimensions must have length {dim}"
        )));
    }
    let table_size = checked_dimension_product(name, dims)?;
    if data.len() != n * dim {
        return Err(PyValueError::new_err(format!(
            "{name}_data must have length {}",
            n * dim
        )));
    }
    validate_finite(&format!("{name}_data"), data)?;

    let mut cut_offset = 0usize;
    for dimension in 0..dim {
        let factor = factors[dimension];
        if factor < 0 || (observed && factor > 1) {
            return Err(PyValueError::new_err(format!(
                "invalid {name} factor type at dimension {dimension}"
            )));
        }
        if factor == 1 {
            for row in 0..n {
                let value = data[dimension * n + row];
                if value.fract() != 0.0 || value < 1.0 || value > dims[dimension] as f64 {
                    return Err(PyValueError::new_err(format!(
                        "{name} factor codes must be integers between 1 and {}",
                        dims[dimension]
                    )));
                }
            }
            continue;
        }

        let cut_count = if observed {
            dims[dimension] + 1
        } else if factor == 0 {
            dims[dimension]
        } else {
            1 + (factor as usize - 1) * dims[dimension]
        };
        let end = cut_offset
            .checked_add(cut_count)
            .ok_or_else(|| PyValueError::new_err(format!("{name} cutpoint count is too large")))?;
        if end > cuts.len() {
            return Err(PyValueError::new_err(format!(
                "{name}_cuts does not contain enough cutpoints"
            )));
        }
        let dimension_cuts = &cuts[cut_offset..end];
        validate_finite(&format!("{name}_cuts"), dimension_cuts)?;
        if dimension_cuts.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(PyValueError::new_err(format!(
                "{name} cutpoints must be strictly increasing"
            )));
        }
        cut_offset = end;
    }
    if cut_offset != cuts.len() {
        return Err(PyValueError::new_err(format!(
            "{name}_cuts contains unused cutpoints"
        )));
    }
    Ok(table_size)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn perform_pyears_calculation(
    time_data: Vec<f64>,
    weights: Vec<f64>,
    expected_dim: usize,
    expected_factors: Vec<i32>,
    expected_dims: Vec<usize>,
    expected_cuts: Vec<f64>,
    expected_rates: Vec<f64>,
    expected_data: Vec<f64>,
    observed_dim: usize,
    observed_factors: Vec<i32>,
    observed_dims: Vec<usize>,
    observed_cuts: Vec<f64>,
    method: i32,
    observed_data: Vec<f64>,
    do_event: Option<i32>,
    ny: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let n = weights.len();
    if n == 0 {
        return Err(PyRuntimeError::new_err("No observations provided"));
    }
    let ny = ny.unwrap_or(2);
    let doevent = do_event.unwrap_or(1);
    if !(1..=3).contains(&ny) {
        return Err(PyValueError::new_err("ny must be 1, 2, or 3"));
    }
    if doevent != 0 && doevent != 1 {
        return Err(PyValueError::new_err("do_event must be 0 or 1"));
    }
    if ny == 1 && doevent == 1 {
        return Err(PyValueError::new_err("ny=1 cannot include an event column"));
    }
    if time_data.len() != n * ny {
        return Err(PyValueError::new_err(format!(
            "time_data must have length {}",
            n * ny
        )));
    }
    if method != 0 && method != 1 {
        return Err(PyValueError::new_err("method must be 0 or 1"));
    }
    validate_finite("time_data", &time_data)?;
    validate_finite("weights", &weights)?;
    if weights.iter().any(|&weight| weight < 0.0) {
        return Err(PyValueError::new_err("weights must be non-negative"));
    }

    let total_expected = validate_table_layout(TableLayout {
        name: "expected",
        n,
        dim: expected_dim,
        factors: &expected_factors,
        dims: &expected_dims,
        cuts: &expected_cuts,
        data: &expected_data,
        observed: false,
    })?;
    if expected_rates.len() != total_expected {
        return Err(PyValueError::new_err(format!(
            "expected_rates must have length {total_expected}"
        )));
    }
    validate_finite("expected_rates", &expected_rates)?;
    let total_observed = validate_table_layout(TableLayout {
        name: "observed",
        n,
        dim: observed_dim,
        factors: &observed_factors,
        dims: &observed_dims,
        cuts: &observed_cuts,
        data: &observed_data,
        observed: true,
    })?;
    let mut pyears = vec![0.0; total_observed];
    let mut pn = vec![0.0; total_observed];
    let mut pcount = vec![0.0; total_observed];
    let mut pexpect = vec![0.0; total_observed];
    let mut offtable = 0.0;
    let expected = PyearsExpectedParams {
        dim: expected_dim,
        fac: &expected_factors,
        dims: &expected_dims,
        cut: &expected_cuts,
        rates: &expected_rates,
        data: &expected_data,
    };
    let observed = PyearsObservedParams {
        dim: observed_dim,
        fac: &observed_factors,
        dims: &observed_dims,
        cut: &observed_cuts,
        data: &observed_data,
    };
    let mut output = PyearsOutput {
        pyears: &mut pyears,
        pn: &mut pn,
        pcount: &mut pcount,
        pexpect: &mut pexpect,
        offtable: &mut offtable,
    };
    pyears3b(
        n,
        ny,
        doevent,
        &time_data,
        &weights,
        expected,
        observed,
        method,
        &mut output,
    );
    Python::attach(|py| {
        let dict = PyDict::new(py);
        dict.set_item("pyears", pyears)?;
        dict.set_item("pn", pn)?;
        dict.set_item("pcount", pcount)?;
        dict.set_item("pexpect", pexpect)?;
        dict.set_item("offtable", offtable)?;
        Ok(dict.into())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_observed(
        n: usize,
        ny: usize,
        y: &[f64],
        weights: &[f64],
        observed: PyearsObservedParams<'_>,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64) {
        let size = observed.dims.iter().product();
        let mut pyears = vec![0.0; size];
        let mut pn = vec![0.0; size];
        let mut pcount = vec![0.0; size];
        let mut pexpect = vec![0.0; size];
        let mut offtable = 0.0;
        let expected_data = vec![1.0; n];
        let expected = PyearsExpectedParams {
            dim: 1,
            fac: &[1],
            dims: &[1],
            cut: &[],
            rates: &[0.0],
            data: &expected_data,
        };
        let mut output = PyearsOutput {
            pyears: &mut pyears,
            pn: &mut pn,
            pcount: &mut pcount,
            pexpect: &mut pexpect,
            offtable: &mut offtable,
        };

        pyears3b(n, ny, 1, y, weights, expected, observed, 1, &mut output);
        (pyears, pn, pcount, offtable)
    }

    #[test]
    fn tcut_person_time_and_events_match_reference_cells() {
        let result = run_observed(
            2,
            2,
            &[25.0, 8.0, 1.0, 0.0],
            &[1.0, 1.0],
            PyearsObservedParams {
                dim: 1,
                fac: &[0],
                dims: &[3],
                cut: &[0.0, 10.0, 20.0, 30.0],
                data: &[0.0, 5.0],
            },
        );

        assert_eq!(result.0, vec![15.0, 13.0, 5.0]);
        assert_eq!(result.1, vec![2.0, 2.0, 1.0]);
        assert_eq!(result.2, vec![0.0, 0.0, 1.0]);
        assert_eq!(result.3, 0.0);
    }

    #[test]
    fn tcut_tracks_off_table_time_without_panicking() {
        let result = run_observed(
            3,
            2,
            &[10.0, 10.0, 10.0, 1.0, 1.0, 1.0],
            &[1.0, 1.0, 1.0],
            PyearsObservedParams {
                dim: 1,
                fac: &[0],
                dims: &[3],
                cut: &[0.0, 10.0, 20.0, 30.0],
                data: &[-5.0, 25.0, 35.0],
            },
        );

        assert_eq!(result.0, vec![5.0, 0.0, 5.0]);
        assert_eq!(result.1, vec![1.0, 0.0, 1.0]);
        assert_eq!(result.2, vec![1.0, 0.0, 0.0]);
        assert_eq!(result.3, 20.0);
    }

    #[test]
    fn tcut_and_factor_dimensions_use_column_major_output_order() {
        let result = run_observed(
            3,
            2,
            &[25.0, 8.0, 12.0, 1.0, 0.0, 1.0],
            &[1.0, 1.0, 1.0],
            PyearsObservedParams {
                dim: 2,
                fac: &[0, 1],
                dims: &[4, 2],
                cut: &[0.0, 10.0, 20.0, 30.0, 40.0],
                data: &[0.0, 5.0, 15.0, 1.0, 2.0, 1.0],
            },
        );

        assert_eq!(result.0, vec![10.0, 15.0, 12.0, 0.0, 5.0, 3.0, 0.0, 0.0]);
        assert_eq!(result.1, vec![1.0, 2.0, 2.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
        assert_eq!(result.2, vec![0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(result.3, 0.0);
    }
}
