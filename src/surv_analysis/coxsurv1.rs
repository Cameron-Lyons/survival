#![allow(clippy::needless_range_loop)]
use ndarray::{Array2, ArrayView2};

#[allow(dead_code)]
pub(crate) fn coxsurv1(
    otime: &[f64],
    y: ArrayView2<f64>,
    weight: &[f64],
    sort2: &[usize],
    trans: &[i32],
    xmat: ArrayView2<f64>,
    risk: &[f64],
) -> (usize, Array2<f64>, Array2<f64>, Array2<f64>) {
    let nused = y.nrows();
    let ntime = otime.len();
    let nvar = xmat.ncols();

    let mut ntrans = 1;
    let mut current_trans = trans[sort2[0]];
    for i in 1..nused {
        let i2 = sort2[i];
        if trans[i2] != current_trans {
            ntrans += 1;
            current_trans = trans[i2];
        }
    }

    let irow_total = ntime * ntrans;
    let mut count = Array2::zeros((irow_total, 10));
    let mut xbar = Array2::zeros((irow_total, nvar));
    let mut xsum2_arr = Array2::zeros((irow_total, nvar));

    let mut person2 = nused as isize - 1;

    for ii in 0..ntrans {
        if person2 < 0 {
            break;
        }

        let current_trans = trans[sort2[person2 as usize]];
        let mut n = [0.0; 12];
        let mut xsum1 = vec![0.0; nvar];
        let mut xsum2 = vec![0.0; nvar];

        for jj in (0..ntime).rev() {
            let dtime = otime[jj];

            for k in 3..8 {
                n[k] = 0.0;
            }

            while person2 >= 0 && trans[sort2[person2 as usize]] == current_trans {
                let i2 = sort2[person2 as usize];
                let tstop_i = y[[i2, 0]];
                let status_i = y[[i2, 1]];

                if tstop_i < dtime {
                    break;
                }

                n[0] += 1.0;
                n[1] += weight[i2];
                n[2] += weight[i2] * risk[i2];

                for k in 0..nvar {
                    xsum1[k] += weight[i2] * risk[i2] * xmat[[i2, k]];
                }

                if status_i == 0.0 {
                    n[6] += 1.0;
                    n[7] += weight[i2];
                } else if tstop_i == dtime {
                    n[3] += 1.0;
                    n[4] += weight[i2];
                    n[5] += weight[i2] * risk[i2];

                    for k in 0..nvar {
                        xsum2[k] += weight[i2] * risk[i2] * xmat[[i2, k]];
                    }
                }

                person2 -= 1;
            }

            if n[3] <= 1.0 {
                n[8] = n[2];
                n[9] = n[2].powi(2);
            } else {
                let meanwt = n[5] / (n[3] * n[3]);
                let n_events = n[3] as usize;
                let mut sum_efron = 0.0;
                let mut sum_sq_efron = 0.0;
                for k in 0..n_events {
                    let term = n[2] - (k as f64) * meanwt;
                    sum_efron += term;
                    sum_sq_efron += term.powi(2);
                }
                n[8] = sum_efron / n[3];
                n[9] = sum_sq_efron / n[3];
            }

            let row = (ntrans - 1 - ii) * ntime + jj;

            for k in 0..10 {
                count[[row, k]] = n[k];
            }

            for k in 0..nvar {
                xbar[[row, k]] = if n[0] == 0.0 { 0.0 } else { xsum1[k] / n[3] };
                xsum2_arr[[row, k]] = xsum2[k];
            }
        }

        while person2 >= 0 && trans[sort2[person2 as usize]] == current_trans {
            person2 -= 1;
        }
    }

    (ntrans, count, xbar, xsum2_arr)
}
