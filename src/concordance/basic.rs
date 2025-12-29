#![allow(clippy::explicit_counter_loop)]
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
#[pyfunction]
pub fn concordance(
    y: Vec<f64>,
    x: Vec<i32>,
    wt: Vec<f64>,
    timewt: Vec<f64>,
    sortstart: Option<Vec<usize>>,
    sortstop: Vec<usize>,
) -> PyResult<Py<PyDict>> {
    let n = y.len();
    let mut ntree = 0;
    let mut nwt = vec![0.0; n];
    let mut twt = vec![0.0; n];
    let mut count = vec![0.0; 5];
    for val in &x {
        ntree = ntree.max(*val as usize + 1);
    }
    let mut utime = 0;
    let i2 = 0;
    let mut i = 0;
    while i < n {
        let ii = sortstop[i];
        let current_time = y[ii];
        if (sortstart.as_ref().is_some()
            && i2 < n
            && y[sortstart.as_ref().unwrap()[i2]] >= current_time)
            || y[ii] == 0.0
        {
            addin(&mut nwt, &mut twt, x[ii] as usize, wt[ii]);
            i += 1;
        } else {
            let mut ndeath = 0;
            let mut _dwt = 0.0;
            let mut _dwt2 = 0.0;
            let xsave = x[ii];
            let adjtimewt = timewt[utime];
            utime += 1;
            while i + ndeath < n && y[sortstop[i + ndeath]] == current_time {
                let jj = sortstop[i + ndeath];
                if x[jj] == xsave {
                    count[2] += 1.0;
                } else if i > 1000 {
                    let (concordant, discordant): (f64, f64) = (0..i)
                        .into_par_iter()
                        .map(|k| {
                            let kk = sortstop[k];
                            if x[kk] != x[jj] {
                                if (x[kk] < x[jj] && y[kk] > current_time)
                                    || (x[kk] > x[jj] && y[kk] < current_time)
                                {
                                    (1.0, 0.0)
                                } else {
                                    (0.0, 1.0)
                                }
                            } else {
                                (0.0, 0.0)
                            }
                        })
                        .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
                    count[0] += concordant;
                    count[1] += discordant;
                } else {
                    for &kk in &sortstop[..i] {
                        if x[kk] != x[jj] {
                            if (x[kk] < x[jj] && y[kk] > current_time)
                                || (x[kk] > x[jj] && y[kk] < current_time)
                            {
                                count[0] += 1.0;
                            } else {
                                count[1] += 1.0;
                            }
                        }
                    }
                }
                _dwt += wt[jj];
                _dwt2 += wt[jj] * adjtimewt;
                ndeath += 1;
            }
            count[4] += (ndeath as f64) * (ndeath as f64 - 1.0) / 2.0;
            for &jj in &sortstop[i..i + ndeath] {
                addin(&mut nwt, &mut twt, x[jj] as usize, wt[jj]);
            }
            i += ndeath;
        }
    }
    count[3] -= count[4];
    Python::attach(|py| {
        let dict = PyDict::new(py);
        dict.set_item("count", count)?;
        Ok(dict.into())
    })
}
#[inline]
fn addin(nwt: &mut [f64], twt: &mut [f64], x: usize, weight: f64) {
    nwt[x] += weight;
    let mut node_index = x;
    while node_index != 0 {
        let parent_index = (node_index - 1) / 2;
        twt[parent_index] += weight;
        node_index = parent_index;
    }
    twt[x] += weight;
}
