#![allow(clippy::needless_range_loop)]
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

struct FenwickTree {
    tree: Vec<f64>,
}

impl FenwickTree {
    fn new(size: usize) -> Self {
        FenwickTree {
            tree: vec![0.0; size + 1],
        }
    }

    fn update(&mut self, index: usize, value: f64) {
        let mut idx = index + 1;
        while idx < self.tree.len() {
            self.tree[idx] += value;
            idx += idx & (!idx + 1);
        }
    }

    fn prefix_sum(&self, index: usize) -> f64 {
        let mut sum = 0.0;
        let mut idx = index + 1;
        while idx > 0 {
            sum += self.tree[idx];
            idx -= idx & (!idx + 1);
        }
        sum
    }

    fn total(&self) -> f64 {
        self.prefix_sum(self.tree.len() - 2)
    }
}

fn addin(nwt: &mut [f64], fenwick: &mut FenwickTree, x: usize, weight: f64) {
    nwt[x] += weight;
    fenwick.update(x, weight);
}

fn walkup(nwt: &[f64], fenwick: &FenwickTree, x: usize) -> [f64; 3] {
    let sum_less = fenwick.prefix_sum(x.saturating_sub(1));
    let sum_greater = fenwick.total() - fenwick.prefix_sum(x);
    let sum_equal = nwt[x];
    [sum_greater, sum_less, sum_equal]
}

pub fn concordance5(
    y: &[f64],
    x: &[i32],
    wt: &[f64],
    timewt: &[f64],
    sortstart: Option<&[usize]>,
    sortstop: &[usize],
    doresid: bool,
) -> (Vec<f64>, Vec<f64>, Option<Vec<f64>>) {
    let n = x.len();
    let mut ntree = 0;

    for &val in x {
        ntree = ntree.max(val as usize + 1);
    }

    let mut nwt = vec![0.0; ntree];
    let mut fenwick = FenwickTree::new(ntree);
    let mut count = vec![0.0; 6];
    let mut imat = vec![0.0; 3 * n];
    let resid = if doresid {
        let nevent = y[n..].iter().filter(|&&v| v == 1.0).count();
        Some(vec![0.0; 3 * nevent])
    } else {
        None
    };

    let mut utime = 0;
    let i2 = 0;
    let mut i = 0;
    let mut z2 = 0.0;

    while i < n {
        let ii = sortstop[i];
        let current_time = y[ii];

        if (sortstart.is_some() && i2 < n && y[sortstart.unwrap()[i2]] >= current_time)
            || y[ii] == 0.0
        {
            addin(&mut nwt, &mut fenwick, x[ii] as usize, wt[ii]);
            i += 1;
        } else {
            let mut ndeath = 0;
            let mut _dwt = 0.0;
            let mut _dwt2 = 0.0;
            let adjtimewt = timewt[utime];
            utime += 1;

            while i + ndeath < n && y[sortstop[i + ndeath]] == current_time {
                let jj = sortstop[i + ndeath];
                if y[n + jj] == 1.0 {
                    _dwt += wt[jj];
                    _dwt2 += wt[jj] * adjtimewt;
                }
                ndeath += 1;
            }

            for j in i..(i + ndeath) {
                let jj = sortstop[j];
                if y[n + jj] == 1.0 {
                    let wsum = walkup(&nwt, &fenwick, x[jj] as usize);

                    count[0] += wt[jj] * wsum[0] * adjtimewt;
                    count[1] += wt[jj] * wsum[1] * adjtimewt;
                    count[2] += wt[jj] * wsum[2] * adjtimewt;

                    imat[jj] += wsum[1] * adjtimewt;
                    imat[n + jj] += wsum[0] * adjtimewt;
                    imat[2 * n + jj] += wsum[2] * adjtimewt;

                    z2 += compute_z2(wt[jj], &wsum);
                }
            }

            count[4] += (ndeath as f64) * (ndeath as f64 - 1.0) / 2.0;

            for j in i..(i + ndeath) {
                let jj = sortstop[j];
                addin(&mut nwt, &mut fenwick, x[jj] as usize, wt[jj]);
            }

            i += ndeath;
        }
    }

    count[3] = count[4];
    count[4] = 0.0;

    if fenwick.total() > 0.0 {
        count[5] = z2 / fenwick.total();
    }

    (count, imat, resid)
}

fn compute_z2(wt: f64, wsum: &[f64]) -> f64 {
    let total = wsum[0] + wsum[1] + wsum[2];
    if total == 0.0 {
        return 0.0;
    }

    let expected = total / 3.0;
    let observed = wsum[0];
    wt * (observed - expected).powi(2) / expected
}

#[pyfunction]
pub fn perform_concordance_calculation(
    time_data: Vec<f64>,
    predictor_values: Vec<i32>,
    weights: Vec<f64>,
    time_weights: Vec<f64>,
    sort_stop: Vec<usize>,
    sort_start: Option<Vec<usize>>,
    do_residuals: Option<bool>,
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

    if predictor_values.len() != n {
        return Err(PyRuntimeError::new_err(
            "Predictor values length does not match observations",
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

    let doresid = do_residuals.unwrap_or(false);

    let (count, imat, resid) = concordance5(
        &time_data,
        &predictor_values,
        &weights,
        &time_weights,
        sort_start.as_deref(),
        &sort_stop,
        doresid,
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
        if let Some(residuals) = resid {
            dict.set_item("residuals", residuals).unwrap();
        }
        Ok(dict.into())
    })
}
