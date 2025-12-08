#![allow(dead_code)]
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

fn update_used(
    used: &mut Array2<i32>,
    stratum: i32,
    start: usize,
    end: usize,
    covar: &ArrayView2<f64>,
    sort: &ArrayView1<usize>,
) {
    let stratum_idx = stratum as usize;
    for i in start..end {
        let person = sort[i];
        for j in 0..covar.ncols() {
            if covar[(person, j)] != 0.0 {
                used[(stratum_idx, j)] += 1;
            }
        }
    }
}

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
    let mut _ndead = 0;
    for (i, &idx) in sort.iter().enumerate() {
        let stratum = strata[idx];
        if stratum != current_stratum {
            if current_stratum != -1 {
                update_used(&mut used, current_stratum, k, i, &covar, &sort);
            }
            current_stratum = stratum;
            k = i;
            _ndead = 0;
        }
        _ndead += y.1[idx] as usize;
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

    #[allow(unused_comparisons)]
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
        let mut _denom2 = 0.0;

        let _ip_start = ip;
        #[allow(unused_comparisons)]
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
                _denom2 += risk;
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

fn process_breslow(
    u: &mut Array1<f64>,
    imat: &mut Array2<f64>,
    _schoen: &mut Array2<f64>,
    _nevent_counter: usize,
    _ndead_current: usize,
    deadwt: f64,
    timewt: f64,
    a: &Array1<f64>,
    a2: &mut Array1<f64>,
    cmat: &Array2<f64>,
    cmat2: &mut Array2<f64>,
    denom: &mut f64,
    nvar: usize,
) {
    let wt = deadwt / *denom;
    for i in 0..nvar {
        u[i] -= wt * a[i];
        u[i + nvar] -= timewt * wt * a[i];
        for j in 0..=i {
            imat[(i, j)] += wt * cmat[(i, j)];
            imat[(i, j + nvar)] += timewt * wt * cmat[(i, j)];
            imat[(i + nvar, j + nvar)] += timewt * timewt * wt * cmat[(i, j)];
        }
    }
    *denom += a2.iter().sum::<f64>();
    for i in 0..nvar {
        a2[i] = 0.0;
        for j in 0..=i {
            cmat2[(i, j)] = 0.0;
        }
    }
}

fn process_efron(
    u: &mut Array1<f64>,
    imat: &mut Array2<f64>,
    _schoen: &mut Array2<f64>,
    _nevent_counter: usize,
    ndead_current: usize,
    deadwt: f64,
    timewt: f64,
    a: &Array1<f64>,
    a2: &mut Array1<f64>,
    cmat: &Array2<f64>,
    cmat2: &mut Array2<f64>,
    denom: &mut f64,
    nvar: usize,
) {
    for k in 0..ndead_current {
        let temp = k as f64 / ndead_current as f64;
        let d2 = *denom - temp * a2.iter().sum::<f64>();
        let wt = deadwt / (ndead_current as f64 * d2);
        for i in 0..nvar {
            u[i] -= wt * (a[i] - temp * a2[i]);
            u[i + nvar] -= timewt * wt * (a[i] - temp * a2[i]);
            for j in 0..=i {
                imat[(i, j)] += wt * (cmat[(i, j)] - temp * cmat2[(i, j)]);
                imat[(i, j + nvar)] += timewt * wt * (cmat[(i, j)] - temp * cmat2[(i, j)]);
                imat[(i + nvar, j + nvar)] +=
                    timewt * timewt * wt * (cmat[(i, j)] - temp * cmat2[(i, j)]);
            }
        }
    }
    *denom += a2.iter().sum::<f64>();
    for i in 0..nvar {
        a2[i] = 0.0;
        for j in 0..=i {
            cmat2[(i, j)] = 0.0;
        }
    }
}
