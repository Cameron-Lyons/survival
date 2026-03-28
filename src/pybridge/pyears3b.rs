use super::pystep::pystep;
use crate::constants::PYEARS_TIME_EPSILON;
use pyo3::exceptions::PyRuntimeError;
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
        let event = &y[n..2 * n];
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
    for i in 0..n {
        let mut data = vec![0.0; odim];
        let mut data2 = vec![0.0; edim];
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
        let mut cumhaz = 0.0;
        let mut data2_current = vec![0.0; edim];
        while timeleft > eps {
            let (thiscell, idx, _idx2, _lwt) =
                pystep(odim, &mut data, ofac, odims, &ocut_slices, timeleft);
            pyears[idx] += thiscell * weight[i];
            pn[idx] += 1.0;
            if doevent == 1 && !event.is_empty() && event[i] > 0.0 {
                pcount[idx] += weight[i];
            }
            let mut etime = thiscell;
            let mut hazard = 0.0;
            let mut temp = 0.0;
            data2_current.copy_from_slice(&data2);
            while etime > 0.0 {
                let (et2, edx, edx2, elwt) =
                    pystep(edim, &mut data2_current, efac, edims, &ecut_slices, etime);
                let lambda = if elwt < 1.0 {
                    elwt * expect[edx] + (1.0 - elwt) * expect[edx2]
                } else {
                    expect[edx]
                };
                if method == 0 {
                    let neg_hazard: f64 = -hazard;
                    let neg_lambda_et2: f64 = -lambda * et2;
                    temp += neg_hazard.exp() * (1.0 - neg_lambda_et2.exp()) / lambda;
                }
                hazard += lambda * et2;
                for j in 0..edim {
                    if efac[j] != 1 {
                        data2_current[j] += et2;
                    }
                }
                etime -= et2;
            }
            if method == 1 {
                pexpect[idx] += hazard * weight[i];
            } else {
                let neg_cumhaz: f64 = -cumhaz;
                pexpect[idx] += neg_cumhaz.exp() * temp * weight[i];
            }
            cumhaz += hazard;
            timeleft -= thiscell;
        }
    }
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
    let mut total_observed = 1;
    for &dim in &observed_dims {
        total_observed *= dim;
    }
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
