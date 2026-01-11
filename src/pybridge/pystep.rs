use super::column_major_index;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn find_interval(cuts: &[f64], x: f64) -> Option<usize> {
    match cuts.binary_search_by(|&cut| cut.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal)) {
        Ok(i) => {
            if i < cuts.len() - 1 {
                Some(i)
            } else {
                None
            }
        }
        Err(i) => {
            if i > 0 && i <= cuts.len() {
                Some(i - 1)
            } else {
                None
            }
        }
    }
}
pub fn pystep(
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
        if cuts.is_empty() {
            continue;
        }
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
            if !cuts.is_empty() {
                let pos = cuts.partition_point(|&x| x <= data[j]) - 1;
                indices_current[j] = pos.min(edims[j] - 1);
                indices_next[j] = (pos + 1).min(edims[j] - 1);
            } else {
                indices_current[j] = 0;
                indices_next[j] = 0;
            }
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
        if !cuts.is_empty() {
            let pos = cuts.partition_point(|&x| x <= current) - 1;
            if pos + 1 < cuts.len() {
                let next_cut = cuts[pos + 1];
                let prev_cut = cuts[pos];
                let width = next_cut - prev_cut;
                if width > 0.0 {
                    wt = (current + et2 - prev_cut) / width;
                    wt = wt.clamp(0.0, 1.0);
                }
            }
        }
    }
    (et2, indx, indx2, wt)
}
pub fn pystep_simple(
    odim: usize,
    data: &[f64],
    ofac: &[i32],
    odims: &[usize],
    ocut: &[&[f64]],
    timeleft: f64,
) -> (f64, i32) {
    let mut maxtime = timeleft;
    let mut intervals = vec![0; odim];
    let mut valid = true;
    for j in 0..odim {
        if ofac[j] == 0 {
            let cuts = ocut[j];
            if cuts.is_empty() {
                valid = false;
                break;
            }
            let x = data[j];
            match find_interval(cuts, x) {
                Some(i) => {
                    let next_cut = cuts[i + 1];
                    let time_to_next = next_cut - x;
                    if time_to_next < maxtime {
                        maxtime = time_to_next;
                    }
                    intervals[j] = i;
                }
                None => {
                    valid = false;
                    break;
                }
            }
        }
    }
    if !valid {
        return (0.0, -1);
    }
    let mut index = 0;
    for j in 0..odim {
        let idx_j = if ofac[j] == 1 {
            data[j] as usize
        } else {
            intervals[j]
        };
        if idx_j >= odims[j] {
            return (maxtime, -1);
        }
        index = index * odims[j] + idx_j;
    }
    (maxtime, index as i32)
}
#[pyfunction]
pub fn perform_pystep_calculation(
    edim: usize,
    data: Vec<f64>,
    efac: Vec<i32>,
    edims: Vec<usize>,
    ecut: Vec<Vec<f64>>,
    tmax: f64,
) -> PyResult<Py<PyAny>> {
    if data.len() != edim {
        return Err(PyRuntimeError::new_err("Data length does not match edim"));
    }
    if efac.len() != edim {
        return Err(PyRuntimeError::new_err("Factor length does not match edim"));
    }
    if edims.len() != edim {
        return Err(PyRuntimeError::new_err(
            "Dimensions length does not match edim",
        ));
    }
    if ecut.len() != edim {
        return Err(PyRuntimeError::new_err(
            "Cutpoints length does not match edim",
        ));
    }
    let mut data_mut = data.clone();
    let ecut_refs: Vec<&[f64]> = ecut.iter().map(|v| v.as_slice()).collect();
    let (time_step, current_index, next_index, weight) =
        pystep(edim, &mut data_mut, &efac, &edims, &ecut_refs, tmax);
    Python::attach(|py| {
        let dict = PyDict::new(py);
        dict.set_item("time_step", time_step)?;
        dict.set_item("current_index", current_index)?;
        dict.set_item("next_index", next_index)?;
        dict.set_item("weight", weight)?;
        dict.set_item("updated_data", data_mut)?;
        Ok(dict.into())
    })
}
#[pyfunction]
pub fn perform_pystep_simple_calculation(
    odim: usize,
    data: Vec<f64>,
    ofac: Vec<i32>,
    odims: Vec<usize>,
    ocut: Vec<Vec<f64>>,
    timeleft: f64,
) -> PyResult<Py<PyAny>> {
    if data.len() != odim {
        return Err(PyRuntimeError::new_err("Data length does not match odim"));
    }
    if ofac.len() != odim {
        return Err(PyRuntimeError::new_err("Factor length does not match odim"));
    }
    if odims.len() != odim {
        return Err(PyRuntimeError::new_err(
            "Dimensions length does not match odim",
        ));
    }
    if ocut.len() != odim {
        return Err(PyRuntimeError::new_err(
            "Cutpoints length does not match odim",
        ));
    }
    let ocut_refs: Vec<&[f64]> = ocut.iter().map(|v| v.as_slice()).collect();
    let (time_step, index) = pystep_simple(odim, &data, &ofac, &odims, &ocut_refs, timeleft);
    Python::attach(|py| {
        let dict = PyDict::new(py);
        dict.set_item("time_step", time_step)?;
        dict.set_item("index", index)?;
        Ok(dict.into())
    })
}
