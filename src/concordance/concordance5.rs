use super::common::{build_concordance_result, validate_extended_concordance_inputs};
use crate::constants::PARALLEL_THRESHOLD_SMALL;
use crate::utilities::fenwick::FenwickTree;
use pyo3::prelude::*;
use rayon::prelude::*;
#[inline]
fn addin(nwt: &mut [f64], fenwick: &mut FenwickTree, x: usize, weight: f64) {
    nwt[x] += weight;
    fenwick.update(x, weight);
}
#[inline]
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
        let should_skip = match sortstart {
            Some(ss) if i2 < n => y[ss[i2]] >= current_time,
            _ => false,
        };
        if should_skip || y[ii] == 0.0 {
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
            if ndeath > PARALLEL_THRESHOLD_SMALL {
                let results: Vec<_> = (i..(i + ndeath))
                    .into_par_iter()
                    .filter_map(|j| {
                        let jj = sortstop[j];
                        if y[n + jj] == 1.0 {
                            let wsum = walkup(&nwt, &fenwick, x[jj] as usize);
                            let c0 = wt[jj] * wsum[0] * adjtimewt;
                            let c1 = wt[jj] * wsum[1] * adjtimewt;
                            let c2 = wt[jj] * wsum[2] * adjtimewt;
                            let z2_val = compute_z2(wt[jj], &wsum);
                            Some((jj, wsum, c0, c1, c2, z2_val))
                        } else {
                            None
                        }
                    })
                    .collect();
                for (jj, wsum, c0, c1, c2, z2_val) in results {
                    count[0] += c0;
                    count[1] += c1;
                    count[2] += c2;
                    imat[jj] += wsum[1] * adjtimewt;
                    imat[n + jj] += wsum[0] * adjtimewt;
                    imat[2 * n + jj] += wsum[2] * adjtimewt;
                    z2 += z2_val;
                }
            } else {
                for &jj in &sortstop[i..i + ndeath] {
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
            }
            count[4] += (ndeath as f64) * (ndeath as f64 - 1.0) / 2.0;
            for &jj in &sortstop[i..i + ndeath] {
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
#[inline]
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
#[pyo3(signature = (time_data, predictor_values, weights, time_weights, sort_stop, sort_start=None, do_residuals=None))]
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
    validate_extended_concordance_inputs(
        time_data.len(),
        n,
        predictor_values.len(),
        weights.len(),
        time_weights.len(),
        sort_stop.len(),
    )?;
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
    Python::attach(|py| {
        build_concordance_result(py, &count, Some(&imat), resid.as_deref(), None).map(|d| d.into())
    })
}
