use extendr_api::prelude::*;
use std::cmp::Ordering;

#[extendr]
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

#[extendr]
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

#[extendr]
fn concordance3(
    y: &[f64],
    x: &[i32],
    wt: &[f64],
    timewt: &[f64],
    sortstop: &[i32],
    doresid: bool,
) -> Robj {
    let n = x.len();
    let mut ntree = x.iter().map(|&v| v as usize).max().unwrap_or(0) + 1;
    let nevent = y[n..].iter().filter(|&&v| v == 1.0).count();

    let mut nwt = vec![0.0; 4 * ntree];
    let (twt, dnwt, dtwt) = (
        &mut nwt[ntree..2 * ntree],
        &mut nwt[2 * ntree..3 * ntree],
        &mut nwt[3 * ntree..4 * ntree],
    );

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
            // Censored observation
            let wsum = walkup(dnwt, dtwt, x[ii] as usize, ntree);
            imat[ii] -= wsum[1];
            imat[n + ii] -= wsum[0];
            imat[2 * n + ii] -= wsum[2];

            let wsum_main = walkup(&nwt, twt, x[ii] as usize, ntree);
            z2 += compute_z2(wt[ii], &wsum_main);

            add_internal(&mut nwt, twt, x[ii] as usize, wt[ii]);
            i += 1;
        } else {
            // Process deaths
            let mut ndeath = 0;
            let mut dwt = 0.0;
            let mut dwt2 = 0.0;
            let adjtimewt = timewt[utime];
            utime += 1;

            let mut j = i;
            while j < n && y[j] == y[i] {
                let jj = sortstop[j] as usize;
                ndeath += 1;
                count[3] += wt[jj] * dwt * adjtimewt;
                dwt += wt[jj];

                let wsum_main = walkup(&nwt, twt, x[jj] as usize, ntree);
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

                let wsum_main = walkup(&nwt, twt, x[jj] as usize, ntree);
                z2 += compute_z2(wt[jj], &wsum_main);

                add_internal(&mut nwt, twt, x[jj] as usize, wt[jj]);
            }

            if doresid { //TODO
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

    let mut out = List::new(3);
    out.set_elt(0, count).unwrap();
    out.set_elt(1, imat).unwrap();
    if doresid {
        out.set_elt(2, resid).unwrap();
    }
    out.into_robj()
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

extendr_module! {
    mod survivalutils;
    fn walkup;
    fn addin;
    fn concordance3;
}
