use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_stats::QuantileExt;

#[derive(Debug)]
pub struct ZphResult {
    pub u: Array1<f64>,
    pub imat: Array2<f64>,
    pub schoen: Array2<f64>,
    pub used: Array2<i32>,
}

pub fn zph2(
    gt: ArrayView1<f64>,
    y: (ArrayView1<f64>, ArrayView1<f64>, ArrayView1<f64>), // (start, stop, status)
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
    let mut ndead = 0;
    for (i, &idx) in sort2.iter().enumerate() {
        let stratum = strata[idx];
        if stratum != current_stratum {
            if current_stratum != -1 {
                update_used(&mut used, current_stratum, k, i, &covar, &sort2);
            }
            current_stratum = stratum;
            k = i;
            ndead = 0;
        }
        ndead += y.2[idx] as usize;
    }
    if current_stratum != -1 {
        update_used(&mut used, current_stratum, k, sort2.len(), &covar, &sort2);
    }

    let mut centered_covar = covar.to_owned();
    for mut col in centered_covar.columns_mut() {
        let mean = col.mean().unwrap();
        col -= mean;
    }

    let mut cstrat = -1;
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
        let (dtime, timewt, death_index) =
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
            &centered_covar,
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
            &a2,
            &cmat2,
            &mut denom,
            nvar,
        );

        recenter = handle_numerics(
            &mut eta.clone(),
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
