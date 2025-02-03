use extendr_api::prelude::*;

struct FenwickTree {
    tree: Vec<f64>,
}

impl FenwickTree {
    fn new(size: usize) -> Self {
        FenwickTree {
            tree: vec![0.0; size + 1], // 1-based indexing
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

#[extendr]
fn concordance5(
    y: RealVector,
    x: IntVector,
    wt: RealVector,
    timewt: RealVector,
    sortstop: IntVector,
) -> List {
    let n = y.len() / 2;
    let time: Vec<f64> = y.iter().take(n).collect();
    let status: Vec<f64> = y.iter().skip(n).take(n).collect();
    let x: Vec<usize> = x.iter().map(|&xi| xi as usize).collect();
    let wt: Vec<f64> = wt.iter().collect();
    let timewt: Vec<f64> = timewt.iter().collect();
    let sortstop: Vec<usize> = sortstop.iter().map(|&si| (si - 1) as usize).collect();

    let ntree = x.iter().max().map(|&m| m + 1).unwrap_or(0);
    let mut nwt = vec![0.0; ntree];
    let mut fenwick = FenwickTree::new(ntree);
    let mut count = vec![0.0; 5];
    let mut utime = 0;
    let mut i = 0;

    while i < n {
        let ii = sortstop[i];
        if (status[ii] - 0.0).abs() < f64::EPSILON {
            addin(&mut nwt, &mut fenwick, x[ii], wt[ii]);
            i += 1;
        } else {
            let current_time = time[ii];
            let mut j = i;
            let mut ndeath = 0;
            let mut dwt = 0.0;
            let mut xsave = x[ii];
            let adjtimewt = timewt[utime];
            utime += 1;

            while j < n && (time[sortstop[j]] - current_time).abs() < f64::EPSILON {
                let jj = sortstop[j];
                ndeath += 1;
                count[3] += wt[jj] * dwt * adjtimewt;
                dwt += wt[jj];

                if x[jj] != xsave {
                    xsave = x[jj];
                    dwt2 = 0.0;
                }
                count[4] += wt[jj] * dwt2 * adjtimewt;
                dwt2 += wt[jj];

                let wsum = walkup(&nwt, &fenwick, x[jj]);
                for k in 0..3 {
                    count[k] += wt[jj] * wsum[k] * adjtimewt;
                }

                j += 1;
            }

            for j in i..i + ndeath {
                let jj = sortstop[j];
                addin(&mut nwt, &mut fenwick, x[jj], wt[jj]);
            }

            i += ndeath;
        }
    }

    count[3] -= count[4];
    list!(count = count)
}

#[extendr]
fn concordance6(
    y: RealVector,
    x: IntVector,
    wt: RealVector,
    timewt: RealVector,
    sortstart: IntVector,
    sortstop: IntVector,
) -> List {
    let n = y.len() / 3;
    let time1: Vec<f64> = y.iter().take(n).collect();
    let time2: Vec<f64> = y.iter().skip(n).take(n).collect();
    let status: Vec<f64> = y.iter().skip(2 * n).take(n).collect();
    let x: Vec<usize> = x.iter().map(|&xi| xi as usize).collect();
    let wt: Vec<f64> = wt.iter().collect();
    let timewt: Vec<f64> = timewt.iter().collect();
    let sortstart: Vec<usize> = sortstart.iter().map(|&si| (si - 1) as usize).collect();
    let sortstop: Vec<usize> = sortstop.iter().map(|&si| (si - 1) as usize).collect();

    let ntree = x.iter().max().map(|&m| m + 1).unwrap_or(0);
    let mut nwt = vec![0.0; ntree];
    let mut fenwick = FenwickTree::new(ntree);
    let mut count = vec![0.0; 5];
    let mut utime = 0;
    let mut i2 = 0;
    let mut i = 0;

    while i < n {
        let ii = sortstop[i];
        if (status[ii] - 0.0).abs() < f64::EPSILON {
            addin(&mut nwt, &mut fenwick, x[ii], wt[ii]);
            i += 1;
        } else {
            while i2 < n && time1[sortstart[i2]] >= time2[ii] {
                let jj = sortstart[i2];
                addin(&mut nwt, &mut fenwick, x[jj], -wt[jj]);
                i2 += 1;
            }

            let current_time = time2[ii];
            let mut j = i;
            let mut ndeath = 0;
            let mut dwt = 0.0;
            let mut xsave = x[ii];
            let adjtimewt = timewt[utime];
            utime += 1;

            while j < n && (time2[sortstop[j]] - current_time).abs() < f64::EPSILON {
                let jj = sortstop[j];
                ndeath += 1;
                count[3] += wt[jj] * dwt * adjtimewt;
                dwt += wt[jj];

                if x[jj] != xsave {
                    xsave = x[jj];
                    dwt2 = 0.0;
                }
                count[4] += wt[jj] * dwt2 * adjtimewt;
                dwt2 += wt[jj];

                let wsum = walkup(&nwt, &fenwick, x[jj]);
                for k in 0..3 {
                    count[k] += wt[jj] * wsum[k] * adjtimewt;
                }

                j += 1;
            }

            for j in i..i + ndeath {
                let jj = sortstop[j];
                addin(&mut nwt, &mut fenwick, x[jj], wt[jj]);
            }

            i += ndeath;
        }
    }

    count[3] -= count[4];
    list!(count = count)
}

extendr_module! {
    mod concordance;
    fn concordance5;
    fn concordance6;
}
