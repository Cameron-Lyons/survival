use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn walkup(nwt: &[f64], twt: &[f64], index: usize, ntree: usize) -> [f64; 3] {
    let mut sums = [0.0; 3];
    if index >= ntree {
        return sums;
    }

    sums[2] = nwt[index];

    let right_child = 2 * index + 2;
    if right_child < ntree {
        sums[0] += twt[right_child];
    }

    let left_child = right_child - 1;
    if left_child < ntree {
        sums[1] += twt[left_child];
    }

    let mut current = index;
    while current > 0 {
        let parent = (current - 1) / 2;
        let parent_twt = twt[parent];
        let current_twt = twt[current];

        if current % 2 == 1 {
            sums[0] += parent_twt - current_twt;
        } else {
            sums[1] += parent_twt - current_twt;
        }
        current = parent;
    }

    sums
}

#[allow(dead_code)]
fn addin(nwt: &mut [f64], twt: &mut [f64], mut index: usize, wt: f64) {
    if index >= nwt.len() {
        return;
    }

    nwt[index] += wt;
    while index > 0 {
        twt[index] += wt;
        index = (index - 1) / 2;
    }
    twt[0] += wt;
}

fn compute_z2(wt: f64, wsum: &[f64]) -> f64 {
    wt * (wsum[0] * (wt + 2.0 * (wsum[1] + wsum[2]))
        + wsum[1] * (wt + 2.0 * (wsum[0] + wsum[2]))
        + (wsum[0] - wsum[1]).powi(2))
}

fn add_internal(nwt: &mut [f64], twt: &mut [f64], index: usize, wt: f64) {
    nwt[index] += wt;
    let mut current = index;
    while current > 0 {
        let parent = (current - 1) / 2;
        twt[parent] += wt;
        current = parent;
    }
    twt[0] += wt;
}

pub fn concordance3(
    y: &[f64],
    x: &[i32],
    wt: &[f64],
    timewt: &[f64],
    sortstop: &[i32],
    doresid: bool,
) -> (Vec<f64>, Vec<f64>, Option<Vec<f64>>) {
    let n = x.len();
    let ntree = x.iter().map(|&v| v as usize).max().unwrap_or(0) + 1;
    let nevent = y[n..].iter().filter(|&&v| v == 1.0).count();

    let mut nwt = vec![0.0; 4 * ntree];
    let (first, rest) = nwt.split_at_mut(ntree);
    let (second, rest) = rest.split_at_mut(ntree);
    let (third, fourth) = rest.split_at_mut(ntree);
    let nwt_main = first;
    let twt = second;
    let dnwt = third;
    let dtwt = fourth;

    let mut count = vec![0.0; 6];
    let mut imat = vec![0.0; 5 * n];
    let mut resid = if doresid {
        vec![0.0; 3 * nevent]
    } else {
        vec![]
    };

    let mut z2 = 0.0;
    let mut utime = 0;
    let mut i = 0;

    while i < n {
        let ii = sortstop[i] as usize;
        if y[n + ii] != 1.0 {
            let wsum = walkup(dnwt, dtwt, x[ii] as usize, ntree);
            imat[ii] -= wsum[1];
            imat[n + ii] -= wsum[0];
            imat[2 * n + ii] -= wsum[2];

            let wsum_main = walkup(nwt_main, twt, x[ii] as usize, ntree);
            z2 += compute_z2(wt[ii], &wsum_main);

            add_internal(nwt_main, twt, x[ii] as usize, wt[ii]);
            i += 1;
        } else {
            let mut ndeath = 0;
            let mut dwt = 0.0;
            let _dwt2 = 0.0;
            let adjtimewt = timewt[utime];
            utime += 1;

            let mut j = i;
            while j < n && y[j] == y[i] {
                let jj = sortstop[j] as usize;
                ndeath += 1;
                count[3] += wt[jj] * dwt * adjtimewt;
                dwt += wt[jj];

                let wsum_main = walkup(nwt_main, twt, x[jj] as usize, ntree);
                for k in 0..3 {
                    count[k] += wt[jj] * wsum_main[k] * adjtimewt;
                    imat[k * n + jj] += wsum_main[k] * adjtimewt;
                }

                add_internal(dnwt, dtwt, x[jj] as usize, adjtimewt * wt[jj]);
                j += 1;
            }

            for j in i..(i + ndeath) {
                let jj = sortstop[j] as usize;
                let wsum_death = walkup(dnwt, dtwt, x[jj] as usize, ntree);
                imat[jj] -= wsum_death[1];
                imat[n + jj] -= wsum_death[0];
                imat[2 * n + jj] -= wsum_death[2];

                let wsum_main = walkup(nwt_main, twt, x[jj] as usize, ntree);
                z2 += compute_z2(wt[jj], &wsum_main);

                add_internal(nwt_main, twt, x[jj] as usize, wt[jj]);
            }

            if doresid {
                let mut event_idx = 0;
                for j in i..(i + ndeath) {
                    let jj = sortstop[j] as usize;
                    let wsum = walkup(nwt_main, twt, x[jj] as usize, ntree);
                    resid[event_idx * 3 + 0] = wsum[0];
                    resid[event_idx * 3 + 1] = wsum[1];
                    resid[event_idx * 3 + 2] = wsum[2];
                    event_idx += 1;
                }
            }

            count[5] += dwt * adjtimewt * z2 / twt[0];
            i += ndeath;
        }
    }

    for i in 0..n {
        let ii = sortstop[i] as usize;
        let wsum = walkup(dnwt, dtwt, x[ii] as usize, ntree);
        imat[ii] += wsum[1];
        imat[n + ii] += wsum[0];
        imat[2 * n + ii] += wsum[2];
    }

    let resid_opt = if doresid { Some(resid) } else { None };
    (count, imat, resid_opt)
}

#[pyfunction]
pub fn perform_concordance3_calculation(
    time_data: Vec<f64>,
    indices: Vec<i32>,
    weights: Vec<f64>,
    time_weights: Vec<f64>,
    sort_stop: Vec<i32>,
    do_residuals: bool,
) -> PyResult<Py<PyAny>> {
    let n = weights.len();
    if n == 0 {
        return Err(PyRuntimeError::new_err("No observations provided"));
    }

    if time_data.len() != 2 * n {
        return Err(PyRuntimeError::new_err(
            "Time data should have 2*n elements (time, status)",
        ));
    }

    if indices.len() != n {
        return Err(PyRuntimeError::new_err(
            "Indices length does not match observations",
        ));
    }

    if weights.len() != n {
        return Err(PyRuntimeError::new_err(
            "Weights length does not match observations",
        ));
    }

    if time_weights.len() != n {
        return Err(PyRuntimeError::new_err(
            "Time weights length does not match observations",
        ));
    }

    if sort_stop.len() != n {
        return Err(PyRuntimeError::new_err(
            "Sort stop length does not match observations",
        ));
    }

    let (count, imat, resid_opt) = concordance3(
        &time_data,
        &indices,
        &weights,
        &time_weights,
        &sort_stop,
        do_residuals,
    );

    let concordant = count[0];
    let discordant = count[1];
    let tied_x = count[2];
    let tied_y = count[3];
    let tied_xy = count[4];
    let variance = count[5];

    let total_pairs = concordant + discordant + tied_x + tied_y + tied_xy;
    let concordance_index = if total_pairs > 0.0 {
        (concordant + 0.5 * (tied_x + tied_y + tied_xy)) / total_pairs
    } else {
        0.0
    };

    Python::attach(|py| {
        let dict = PyDict::new(py);
        dict.set_item("concordant", concordant).unwrap();
        dict.set_item("discordant", discordant).unwrap();
        dict.set_item("tied_x", tied_x).unwrap();
        dict.set_item("tied_y", tied_y).unwrap();
        dict.set_item("tied_xy", tied_xy).unwrap();
        dict.set_item("variance", variance).unwrap();
        dict.set_item("concordance_index", concordance_index)
            .unwrap();
        dict.set_item("total_pairs", total_pairs).unwrap();
        dict.set_item("information_matrix", imat).unwrap();
        dict.set_item("n_observations", n).unwrap();

        if let Some(resid) = resid_opt {
            dict.set_item("residuals", resid).unwrap();
        }

        Ok(dict.into())
    })
}
