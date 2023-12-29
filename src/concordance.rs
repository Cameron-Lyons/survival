use pyo3::prelude::*;
use std::collections::HashMap;

struct ConcordanceResult {
    count: Vec<f64>,
}

#[pyfunction]
fn concordance(
    y: &Vec<f64>,
    x: &Vec<i32>,
    wt: &Vec<f64>,
    timewt: &Vec<f64>,
    sortstart: Option<&Vec<usize>>,
    sortstop: &Vec<usize>,
) -> ConcordanceResult {
    let n = y.len();
    let mut ntree = 0;
    let mut nwt = vec![0.0; n];
    let mut twt = vec![0.0; n];
    let mut count = vec![0.0; 5];

    for &val in x {
        ntree = ntree.max(val as usize + 1);
    }

    let mut utime = 0;
    let mut i2 = 0; // Only used for concordance6
    let mut i = 0;

    while i < n {
        let ii = sortstop[i];
        let current_time = y[ii];

        if (sortstart.is_some() && i2 < n && y[sortstart.unwrap()[i2]] >= current_time)
            || y[ii] == 0.0
        {
            addin(&mut nwt, &mut twt, x[ii] as usize, wt[ii]);
            i += 1;
        } else {
            let mut ndeath = 0;
            let mut dwt = 0.0;
            let mut dwt2 = 0.0;
            let mut xsave = x[ii];
            let adjtimewt = timewt[utime];
            utime += 1;

            // Pass 1
            while i + ndeath < n && y[sortstop[i + ndeath]] == current_time {
                let jj = sortstop[i + ndeath];
                if x[jj] == xsave {
                    // If this is a tied event, update the appropriate count
                    count[2] += 1.0;
                } else {
                    // If this is not a tied event, update the concordance/discordance
                    for k in 0..i {
                        let kk = sortstop[k];
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
                dwt += wt[jj];
                dwt2 += wt[jj] * adjtimewt;
                ndeath += 1;
            }

            // Updating the tied events
            count[4] += (ndeath as f64) * (ndeath as f64 - 1.0) / 2.0;
            // Pass 2
            for j in i..(i + ndeath) {
                let jj = sortstop[j];
                addin(&mut nwt, &mut twt, x[jj] as usize, wt[jj]);
            }

            i += ndeath;
        }
    }

    count[3] -= count[4];

    ConcordanceResult { count }
}

#[pyfunction]
fn walkup(nwt: &Vec<f64>, twt: &Vec<f64>, index: usize, wsum: &mut [f64; 3], ntree: usize) {
    wsum[0] = 0.0; // Greater than
    wsum[1] = 0.0; // Less than
    wsum[2] = 0.0; // Equal

    for i in 0..ntree {
        if i < index {
            wsum[1] += twt[i];
        } else if i > index {
            wsum[0] += nwt[i];
        } else {
            wsum[2] += nwt[i];
        }
    }
}
#[pyfunction]
fn addin(nwt: &mut Vec<f64>, twt: &mut Vec<f64>, x: usize, weight: f64) {
    nwt[x] += weight;
    let mut node_index = x;
    while node_index != 0 {
        let parent_index = (node_index - 1) / 2;
        twt[parent_index] += weight;
        node_index = parent_index;
    }
    twt[x] += weight;
}
