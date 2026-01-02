#![allow(clippy::explicit_counter_loop)]
use super::common::{
    add_to_binary_tree, build_concordance_result, validate_extended_concordance_inputs,
    walkup_binary_tree,
};
use crate::constants::PARALLEL_THRESHOLD_LARGE;
use pyo3::prelude::*;
use rayon::prelude::*;

#[inline]
fn compute_z2(wt: f64, wsum: &[f64]) -> f64 {
    wt * (wsum[0] * (wt + 2.0 * (wsum[1] + wsum[2]))
        + wsum[1] * (wt + 2.0 * (wsum[0] + wsum[2]))
        + (wsum[0] - wsum[1]).powi(2))
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
            let wsum = walkup_binary_tree(dnwt, dtwt, x[ii] as usize, ntree);
            imat[ii] -= wsum[1];
            imat[n + ii] -= wsum[0];
            imat[2 * n + ii] -= wsum[2];
            let wsum_main = walkup_binary_tree(nwt_main, twt, x[ii] as usize, ntree);
            z2 += compute_z2(wt[ii], &wsum_main);
            add_to_binary_tree(nwt_main, twt, x[ii] as usize, wt[ii]);
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
                let wsum_main = walkup_binary_tree(nwt_main, twt, x[jj] as usize, ntree);
                for k in 0..3 {
                    count[k] += wt[jj] * wsum_main[k] * adjtimewt;
                    imat[k * n + jj] += wsum_main[k] * adjtimewt;
                }
                add_to_binary_tree(dnwt, dtwt, x[jj] as usize, adjtimewt * wt[jj]);
                j += 1;
            }
            for &sort_j in &sortstop[i..i + ndeath] {
                let jj = sort_j as usize;
                let wsum_death = walkup_binary_tree(dnwt, dtwt, x[jj] as usize, ntree);
                imat[jj] -= wsum_death[1];
                imat[n + jj] -= wsum_death[0];
                imat[2 * n + jj] -= wsum_death[2];
                let wsum_main = walkup_binary_tree(nwt_main, twt, x[jj] as usize, ntree);
                z2 += compute_z2(wt[jj], &wsum_main);
                add_to_binary_tree(nwt_main, twt, x[jj] as usize, wt[jj]);
            }
            if doresid {
                for (event_idx, &sort_j) in sortstop[i..i + ndeath].iter().enumerate() {
                    let jj = sort_j as usize;
                    let wsum = walkup_binary_tree(nwt_main, twt, x[jj] as usize, ntree);
                    resid[event_idx * 3] = wsum[0];
                    resid[event_idx * 3 + 1] = wsum[1];
                    resid[event_idx * 3 + 2] = wsum[2];
                }
            }
            count[5] += dwt * adjtimewt * z2 / twt[0];
            i += ndeath;
        }
    }
    if n > PARALLEL_THRESHOLD_LARGE {
        let updates: Vec<_> = sortstop
            .par_iter()
            .take(n)
            .map(|&sort_idx| {
                let ii = sort_idx as usize;
                let wsum = walkup_binary_tree(dnwt, dtwt, x[ii] as usize, ntree);
                (ii, wsum)
            })
            .collect();
        for (ii, wsum) in updates {
            imat[ii] += wsum[1];
            imat[n + ii] += wsum[0];
            imat[2 * n + ii] += wsum[2];
        }
    } else {
        for &sort_idx in sortstop.iter().take(n) {
            let ii = sort_idx as usize;
            let wsum = walkup_binary_tree(dnwt, dtwt, x[ii] as usize, ntree);
            imat[ii] += wsum[1];
            imat[n + ii] += wsum[0];
            imat[2 * n + ii] += wsum[2];
        }
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
    validate_extended_concordance_inputs(
        time_data.len(),
        n,
        indices.len(),
        weights.len(),
        time_weights.len(),
        sort_stop.len(),
    )?;
    let (count, imat, resid_opt) = concordance3(
        &time_data,
        &indices,
        &weights,
        &time_weights,
        &sort_stop,
        do_residuals,
    );
    Python::attach(|py| {
        build_concordance_result(py, &count, Some(&imat), resid_opt.as_deref(), Some(n))
            .map(|d| d.into())
    })
}
