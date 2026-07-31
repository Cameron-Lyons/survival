use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn find_interval(cuts: &[f64], x: f64) -> Option<usize> {
    match cuts.binary_search_by(|cut| cut.total_cmp(&x)) {
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
pub(crate) fn pystep(
    dim: usize,
    data: &[f64],
    factors: &[i32],
    dims: &[usize],
    cuts: &[&[f64]],
    step: f64,
    extend_edges: bool,
) -> (f64, isize, isize, f64) {
    let mut index = 0isize;
    let mut index2 = 0isize;
    let mut stride = 1isize;
    let mut wt = 1.0;
    let mut shortfall = 0.0;
    let mut max_time = step;

    for dimension in 0..dim {
        let factor = factors[dimension];
        if factor == 1 {
            index += (data[dimension] as isize - 1) * stride;
        } else {
            let dimension_cuts = cuts[dimension];
            let cut_count = if factor > 1 {
                1 + (factor as usize - 1) * dims[dimension]
            } else {
                dims[dimension]
            };
            let mut cut_index =
                dimension_cuts[..cut_count].partition_point(|&cut| data[dimension] >= cut);

            if cut_index == 0 {
                let time_to_first_cut = dimension_cuts[0] - data[dimension];
                if !extend_edges && time_to_first_cut > shortfall {
                    shortfall = time_to_first_cut.min(step);
                }
                max_time = max_time.min(time_to_first_cut);
            } else if cut_index == cut_count {
                if !extend_edges {
                    let time_to_upper_limit = dimension_cuts[cut_count] - data[dimension];
                    if time_to_upper_limit <= 0.0 {
                        shortfall = step;
                    } else {
                        max_time = max_time.min(time_to_upper_limit);
                    }
                }
                if factor > 1 {
                    cut_index = dims[dimension] - 1;
                } else {
                    cut_index -= 1;
                }
            } else {
                max_time = max_time.min(dimension_cuts[cut_index] - data[dimension]);
                cut_index -= 1;
                if factor > 1 {
                    wt = 1.0 - (cut_index % factor as usize) as f64 / factor as f64;
                    cut_index /= factor as usize;
                    index2 = stride;
                }
            }
            index += cut_index as isize * stride;
        }
        stride *= dims[dimension] as isize;
    }

    index2 += index;
    if shortfall == 0.0 {
        (max_time, index, index2, wt)
    } else {
        (shortfall, -1, index2, wt)
    }
}
pub(crate) fn pystep_simple(
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
pub(crate) fn perform_pystep_calculation(
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
    let ecut_refs: Vec<&[f64]> = ecut.iter().map(|v| v.as_slice()).collect();
    let (time_step, current_index, next_index, weight) =
        pystep(edim, &data, &efac, &edims, &ecut_refs, tmax, true);
    let mut updated_data = data;
    for idx in 0..edim {
        if efac[idx] != 1 {
            updated_data[idx] += time_step;
        }
    }
    Python::attach(|py| {
        let dict = PyDict::new(py);
        dict.set_item("time_step", time_step)?;
        dict.set_item("current_index", current_index)?;
        dict.set_item("next_index", next_index)?;
        dict.set_item("weight", weight)?;
        dict.set_item("updated_data", updated_data)?;
        Ok(dict.into())
    })
}
#[pyfunction]
pub(crate) fn perform_pystep_simple_calculation(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pystep_assigns_elapsed_time_to_the_current_cell() {
        let cuts = [0.0, 1.0];
        let (elapsed, index, index2, weight) = pystep(1, &[0.25], &[0], &[2], &[&cuts], 1.0, true);

        assert_eq!(elapsed, 0.75);
        assert_eq!(index, 0);
        assert_eq!(index2, 0);
        assert_eq!(weight, 1.0);
    }

    #[test]
    fn pystep_reports_time_below_and_above_observed_cuts_as_off_table() {
        let cuts = [0.0, 10.0, 20.0, 30.0];
        let below = pystep(1, &[-5.0], &[0], &[3], &[&cuts], 10.0, false);
        let final_cell = pystep(1, &[25.0], &[0], &[3], &[&cuts], 10.0, false);
        let above = pystep(1, &[30.0], &[0], &[3], &[&cuts], 5.0, false);

        assert_eq!(below.0, 5.0);
        assert_eq!(below.1, -1);
        assert_eq!(final_cell.0, 5.0);
        assert_eq!(final_cell.1, 2);
        assert_eq!(above.0, 5.0);
        assert_eq!(above.1, -1);
    }
}
