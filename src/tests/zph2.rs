#![allow(dead_code)]
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

#[allow(clippy::too_many_arguments)]
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

#[allow(clippy::too_many_arguments)]
pub fn zph2(
    _gt: ArrayView1<f64>,
    y: (ArrayView1<f64>, ArrayView1<f64>, ArrayView1<f64>),
    covar: ArrayView2<f64>,
    eta: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    strata: ArrayView1<i32>,
    method: i32,
    sort1: ArrayView1<usize>,
    sort2: ArrayView1<usize>,
) -> Result<ZphResult, &'static str> {
    let nused = y.0.len();
    let nvar = covar.ncols();
    let nevent = y.2.iter().filter(|&&s| s != 0.0).count();
    let nstrat = strata.iter().max().map(|&s| s + 1).unwrap_or(0) as usize;

    let mut u = Array1::zeros(2 * nvar);
    let mut imat = Array2::zeros((2 * nvar, 2 * nvar));
    let mut schoen = Array2::zeros((nevent, nvar));
    let mut used = Array2::zeros((nstrat, nvar));

    let mut current_stratum = -1;
    let mut k = 0;
    let mut _ndead = 0;
    for (i, &idx) in sort2.iter().enumerate() {
        let stratum = strata[idx];
        if stratum != current_stratum {
            if current_stratum != -1 {
                update_used(&mut used, current_stratum, k, i, &covar, &sort2);
            }
            current_stratum = stratum;
            k = i;
            _ndead = 0;
        }
        _ndead += y.2[idx] as usize;
    }
    if current_stratum != -1 {
        update_used(&mut used, current_stratum, k, sort2.len(), &covar, &sort2);
    }

    let mut centered_covar = covar.to_owned();
    for mut col in centered_covar.columns_mut() {
        let mean = col.mean().unwrap();
        col -= mean;
    }

    let cstrat = -1;
    let mut denom = 0.0;
    let mut nrisk = 0;
    let mut etasum = 0.0;
    let mut recenter = 0.0;
    let mut keep = vec![0; nused];

    let mut a = Array1::zeros(nvar);
    let mut cmat = Array2::zeros((nvar, nvar));
    let mut a2 = Array1::zeros(nvar);
    let mut cmat2 = Array2::zeros((nvar, nvar));

    let mut person = 0;
    let mut indx1 = 0;
    let mut nevent_counter = nevent;

    while person < nused {
        let (_dtime, timewt, death_index) =
            match find_next_death(&y, &strata, &sort2, person, cstrat) {
                Some(res) => res,
                None => break,
            };

        update_risk_set(
            &y,
            &strata,
            &sort1,
            &mut keep,
            &mut indx1,
            &mut nrisk,
            &mut etasum,
            &mut denom,
            &mut a,
            &mut cmat,
            &centered_covar,
            &eta,
            &weights,
            recenter,
        );

        let (meanwt, ndead_current) = process_events(
            &mut u,
            &mut schoen,
            &mut a2,
            &mut cmat2,
            &mut nevent_counter,
            death_index,
            person,
            &sort2,
            &y,
            &centered_covar.view(),
            &weights,
            &eta,
            recenter,
            timewt,
        );

        update_scores_and_imat(
            method,
            ndead_current,
            meanwt,
            timewt,
            &mut u,
            &mut imat,
            &mut a,
            &mut cmat,
            &mut a2,
            &mut cmat2,
            &mut denom,
            nvar,
        );

        let mut eta_owned = eta.to_owned();
        recenter = handle_numerics(
            &mut eta_owned,
            &mut a,
            &mut cmat,
            &mut denom,
            nrisk,
            etasum,
            recenter,
        )?;

        person = death_index + 1;
    }

    fill_symmetric_blocks(&mut imat, nvar);

    Ok(ZphResult {
        u,
        imat,
        schoen,
        used,
    })
}

#[allow(clippy::too_many_arguments)]
fn find_next_death(
    y: &(ArrayView1<f64>, ArrayView1<f64>, ArrayView1<f64>),
    strata: &ArrayView1<i32>,
    sort2: &ArrayView1<usize>,
    start: usize,
    cstrat: i32,
) -> Option<(f64, f64, usize)> {
    for i in start..sort2.len() {
        let idx = sort2[i];
        if y.2[idx] > 0.0 && (cstrat == -1 || strata[idx] == cstrat) {
            return Some((y.0[idx], y.1[idx], i));
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn update_risk_set(
    y: &(ArrayView1<f64>, ArrayView1<f64>, ArrayView1<f64>),
    _strata: &ArrayView1<i32>,
    sort1: &ArrayView1<usize>,
    keep: &mut [i32],
    indx1: &mut usize,
    nrisk: &mut usize,
    etasum: &mut f64,
    denom: &mut f64,
    a: &mut Array1<f64>,
    cmat: &mut Array2<f64>,
    centered_covar: &Array2<f64>,
    eta: &ArrayView1<f64>,
    weights: &ArrayView1<f64>,
    _recenter: f64,
) {
    while *indx1 < sort1.len() {
        let idx = sort1[*indx1];
        if y.0[idx] < y.0[sort1[(*indx1).min(sort1.len() - 1)]] {
            break;
        }
        if keep[idx] == 0 {
            keep[idx] = 1;
            *nrisk += 1;
            let risk = eta[idx].exp() * weights[idx];
            *etasum += eta[idx] * weights[idx];
            *denom += risk;
            for i in 0..centered_covar.ncols() {
                a[i] += risk * centered_covar[(idx, i)];
                for j in 0..=i {
                    cmat[(i, j)] += risk * centered_covar[(idx, i)] * centered_covar[(idx, j)];
                }
            }
        }
        *indx1 += 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn process_events(
    u: &mut Array1<f64>,
    schoen: &mut Array2<f64>,
    a2: &mut Array1<f64>,
    cmat2: &mut Array2<f64>,
    nevent_counter: &mut usize,
    death_index: usize,
    person: usize,
    sort2: &ArrayView1<usize>,
    y: &(ArrayView1<f64>, ArrayView1<f64>, ArrayView1<f64>),
    centered_covar: &ArrayView2<f64>,
    weights: &ArrayView1<f64>,
    eta: &ArrayView1<f64>,
    _recenter: f64,
    timewt: f64,
) -> (f64, usize) {
    let mut meanwt = 0.0;
    let mut ndead_current = 0;
    let dtime = y.0[sort2[death_index]];

    for i in person..=death_index {
        let idx = sort2[i];
        if y.0[idx] == dtime && y.2[idx] > 0.0 {
            ndead_current += 1;
            meanwt += weights[idx];
            *nevent_counter -= 1;
            let risk = eta[idx].exp() * weights[idx];
            for j in 0..centered_covar.ncols() {
                schoen[(*nevent_counter, j)] = centered_covar[(idx, j)];
                u[j] += weights[idx] * centered_covar[(idx, j)];
                u[j + centered_covar.ncols()] += timewt * weights[idx] * centered_covar[(idx, j)];
                a2[j] += risk * centered_covar[(idx, j)];
                for k in 0..=j {
                    cmat2[(j, k)] += risk * centered_covar[(idx, j)] * centered_covar[(idx, k)];
                }
            }
        }
    }
    (meanwt / ndead_current.max(1) as f64, ndead_current)
}

#[allow(clippy::too_many_arguments)]
fn update_scores_and_imat(
    method: i32,
    ndead_current: usize,
    meanwt: f64,
    timewt: f64,
    u: &mut Array1<f64>,
    imat: &mut Array2<f64>,
    a: &mut Array1<f64>,
    cmat: &mut Array2<f64>,
    a2: &mut Array1<f64>,
    cmat2: &mut Array2<f64>,
    denom: &mut f64,
    nvar: usize,
) {
    if method == 0 || ndead_current == 1 {
        let wt = meanwt / *denom;
        for i in 0..nvar {
            u[i] -= wt * a[i];
            u[i + nvar] -= timewt * wt * a[i];
            for j in 0..=i {
                imat[(i, j)] += wt * cmat[(i, j)];
                imat[(i, j + nvar)] += timewt * wt * cmat[(i, j)];
                imat[(i + nvar, j + nvar)] += timewt * timewt * wt * cmat[(i, j)];
            }
        }
    } else {
        for k in 0..ndead_current {
            let temp = k as f64 / ndead_current as f64;
            let d2 = *denom - temp * a2.iter().sum::<f64>();
            let wt = meanwt / (ndead_current as f64 * d2);
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
    }
    *denom += a2.iter().sum::<f64>();
    a.fill(0.0);
    a2.fill(0.0);
    cmat.fill(0.0);
    cmat2.fill(0.0);
}

fn handle_numerics(
    eta: &mut Array1<f64>,
    _a: &mut Array1<f64>,
    _cmat: &mut Array2<f64>,
    denom: &mut f64,
    nrisk: usize,
    etasum: f64,
    recenter: f64,
) -> Result<f64, &'static str> {
    if nrisk > 0 {
        let new_recenter = etasum / nrisk as f64;
        let diff = new_recenter - recenter;
        if diff.abs() > 1e-10 {
            *denom *= diff.exp();
            for i in 0..eta.len() {
                eta[i] -= diff;
            }
            Ok(new_recenter)
        } else {
            Ok(recenter)
        }
    } else {
        Ok(recenter)
    }
}

fn fill_symmetric_blocks(imat: &mut Array2<f64>, nvar: usize) {
    for i in 0..nvar {
        for j in 0..i {
            imat[(i, j)] = imat[(j, i)];
            imat[(i, j + nvar)] = imat[(j, i + nvar)];
            imat[(i + nvar, j + nvar)] = imat[(j + nvar, i + nvar)];
        }
    }
}
