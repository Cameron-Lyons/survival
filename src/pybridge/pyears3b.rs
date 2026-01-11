use super::column_major_index;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub struct PyearsExpectedParams<'a> {
    pub dim: usize,
    pub fac: &'a [i32],
    pub dims: &'a [usize],
    pub cut: &'a [f64],
    pub rates: &'a [f64],
    pub data: &'a [f64],
}

pub struct PyearsObservedParams<'a> {
    pub dim: usize,
    pub fac: &'a [i32],
    pub dims: &'a [usize],
    pub cut: &'a [f64],
    pub data: &'a [f64],
}

pub struct PyearsOutput<'a> {
    pub pyears: &'a mut [f64],
    pub pn: &'a mut [f64],
    pub pcount: &'a mut [f64],
    pub pexpect: &'a mut [f64],
    pub offtable: &'a mut f64,
}

#[allow(clippy::too_many_arguments)]
pub fn pyears3b(
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
    eps *= 1e-8;
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
        if timeleft <= eps && doevent == 1 {
            let (_, _idx, _, _) =
                pystep(odim, &mut data.clone(), ofac, odims, &ocut_slices, timeleft);
        }
        while timeleft > eps {
            let mut data_current = data.clone();
            let (thiscell, idx, _idx2, _lwt) =
                pystep(odim, &mut data_current, ofac, odims, &ocut_slices, timeleft);
            data.copy_from_slice(&data_current);
            pyears[idx] += thiscell * weight[i];
            pn[idx] += 1.0;
            if doevent == 1 && !event.is_empty() && event[i] > 0.0 {
                pcount[idx] += weight[i];
            }
            let mut etime = thiscell;
            let mut hazard = 0.0;
            let mut temp = 0.0;
            let mut data2_current = data2.clone();
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

fn pystep(
    edim: usize,
    data: &mut [f64],
    efac: &[i32],
    edims: &[usize],
    ecut: &[&[f64]],
    tmax: f64,
) -> (f64, usize, usize, f64) {
    let mut et2 = tmax;
    let mut wt = 1.0;
    let mut limiting_dim = None;
    for j in 0..edim {
        if efac[j] != 0 {
            continue;
        }
        let cuts = ecut[j];
        let current = data[j];
        let pos = cuts.partition_point(|&x| x <= current);
        if pos < cuts.len() {
            let next_cut = cuts[pos];
            let delta = (next_cut - current).max(0.0);
            if delta < et2 {
                et2 = delta;
                limiting_dim = Some(j);
            }
        }
    }
    et2 = et2.min(tmax);
    let mut indices_current = vec![0; edim];
    let mut indices_next = vec![0; edim];
    for j in 0..edim {
        if efac[j] == 0 {
            data[j] += et2;
            let cuts = ecut[j];
            let pos = cuts.partition_point(|&x| x <= data[j]) - 1;
            indices_current[j] = pos.min(edims[j] - 1);
            indices_next[j] = (pos + 1).min(edims[j] - 1);
        } else {
            indices_current[j] = data[j] as usize - 1;
            indices_next[j] = indices_current[j];
        }
    }
    let indx = column_major_index(&indices_current, edims);
    let indx2 = column_major_index(&indices_next, edims);
    if let Some(dim) = limiting_dim {
        let current = data[dim] - et2;
        let cuts = ecut[dim];
        let pos = cuts.partition_point(|&x| x <= current) - 1;
        let next_cut = if pos + 1 < cuts.len() {
            cuts[pos + 1]
        } else {
            cuts[pos]
        };
        let prev_cut = cuts[pos];
        let width = next_cut - prev_cut;
        wt = if width > 0.0 {
            (current + et2 - prev_cut) / width
        } else {
            1.0
        };
        wt = wt.clamp(0.0, 1.0);
    }
    (et2, indx, indx2, wt)
}
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn perform_pyears_calculation(
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
    let _total_expected = {
        let mut result = 1;
        for &dim in &expected_dims {
            result *= dim;
        }
        result
    };
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
