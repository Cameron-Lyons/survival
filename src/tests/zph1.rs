use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_stats::QuantileExt;

#[derive(Debug)]
pub struct ZphResult {
    pub u: Array1<f64>,
    pub imat: Array2<f64>,
    pub schoen: Array2<f64>,
    pub used: Array2<i32>,
}

pub fn zph1(
    gt: ArrayView1<f64>,
    y: (ArrayView1<f64>, ArrayView1<f64>),
    covar: ArrayView2<f64>,
    eta: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    strata: ArrayView1<i32>,
    method: i32,
    sort: ArrayView1<usize>,
) -> ZphResult {
    let nused = y.0.len();
    let nvar = covar.ncols();
    let nevent = y.1.iter().filter(|&&s| s != 0.0).count();
    let nstrat = strata.iter().max().map(|&s| s + 1).unwrap_or(0) as usize;

    let mut u = Array1::zeros(2 * nvar);
    let mut imat = Array2::zeros((2 * nvar, 2 * nvar));
    let mut schoen = Array2::zeros((nevent, nvar));
    let mut used = Array2::zeros((nstrat, nvar));

    let mut current_stratum = -1;
    let mut k = 0;
    let mut ndead = 0;
    for (i, &idx) in sort.iter().enumerate() {
        let stratum = strata[idx];
        if stratum != current_stratum {
            if current_stratum != -1 {
                update_used(&mut used, current_stratum, k, i, &covar, &sort);
            }
            current_stratum = stratum;
            k = i;
            ndead = 0;
        }
        ndead += y.1[idx] as usize;
    }
    if current_stratum != -1 {
        update_used(&mut used, current_stratum, k, sort.len(), &covar, &sort);
    }

    let mut centered_covar = covar.to_owned();
    for mut col in centered_covar.columns_mut() {
        let mean = col.mean().unwrap();
        col -= mean;
    }

    let mut cstrat = -1;
    let mut ip = nused - 1;
    let mut denom = 0.0;
    let mut a = Array1::zeros(nvar);
    let mut cmat = Array2::zeros((nvar, nvar));
    let mut a2 = Array1::zeros(nvar);
    let mut cmat2 = Array2::zeros((nvar, nvar));
    let mut nevent_counter = nevent;

    while ip >= 0 {
        let person = sort[ip];
        if strata[person] != cstrat {
            cstrat = strata[person];
            denom = 0.0;
            a.fill(0.0);
            cmat.fill(0.0);
        }

        let dtime = y.0[person];
        let timewt = gt[person];
        let mut ndead_current = 0;
        let mut deadwt = 0.0;
        let mut denom2 = 0.0;

        let mut ip_start = ip;
        while ip >= 0 {
            let p = sort[ip];
            if y.0[p] != dtime || strata[p] != cstrat {
                break;
            }
            let risk = eta[p].exp() * weights[p];
            if y.1[p] == 0.0 {
                denom += risk;
                for i in 0..nvar {
                    a[i] += risk * centered_covar[(p, i)];
                    for j in 0..=i {
                        cmat[(i, j)] += risk * centered_covar[(p, i)] * centered_covar[(p, j)];
                    }
                }
            } else {
                ndead_current += 1;
                deadwt += weights[p];
                denom2 += risk;
                nevent_counter -= 1;
                for i in 0..nvar {
                    schoen[(nevent_counter, i)] = centered_covar[(p, i)];
                    u[i] += weights[p] * centered_covar[(p, i)];
                    u[i + nvar] += timewt * weights[p] * centered_covar[(p, i)];
                    a2[i] += risk * centered_covar[(p, i)];
                    for j in 0..=i {
                        cmat2[(i, j)] += risk * centered_covar[(p, i)] * centered_covar[(p, j)];
                    }
                }
            }
            ip -= 1;
        }

        if ndead_current > 0 {
            match method {

                0 => process_breslow(
                    &mut u,
                    &mut imat,
                    &mut schoen,
                    nevent_counter,
                    ndead_current,
                    deadwt,
                    timewt,
                    &a,
                    &mut a2,
                    &cmat,
                    &mut cmat2,
                    &mut denom,
                    nvar,
                ),
                _ => process_efron(
                    &mut u,
                    &mut imat,
                    &mut schoen,
                    nevent_counter,
                    ndead_current,
                    deadwt,
                    timewt,
                    &a,
                    &mut a2,
                    &cmat,
                    &mut cmat2,
                    &mut denom,
                    nvar,
                ),
            }
        }
    }

    for i in 0..nvar {
        for j in 0..i {
            imat[(i, j)] = imat[(j, i)];
            imat[(i, j + nvar)] = imat[(j, i + nvar)];
            imat[(i + nvar, j + nvar)] = imat[(j + nvar, i + nvar)];
        }
    }

    ZphResult {
        u,
        imat,
        schoen,
        used,
    }
}
