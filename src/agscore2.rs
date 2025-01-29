use extendr_api::prelude::*;
use ndarray::{Array2, Axis};

#[extendr]
fn agscore2(
    y: Vec<f64>,
    covar: Vec<f64>,
    strata: Vec<i32>,
    score: Vec<f64>,
    weights: Vec<f64>,
    method: i32,
) -> Vec<f64> {
    let n = y.len() / 3;
    let nvar = covar.len() / n;
    let method = method as i32;

    let tstart = &y[0..n];
    let tstop = &y[n..2 * n];
    let event = &y[2 * n..3 * n];

    let covar_matrix = Array2::from_shape_vec((nvar, n), covar).unwrap();

    let mut resid_matrix = Array2::zeros((nvar, n));

    let mut a = vec![0.0; nvar];
    let mut a2 = vec![0.0; nvar];
    let mut mean = vec![0.0; nvar];
    let mut mh1 = vec![0.0; nvar];
    let mut mh2 = vec![0.0; nvar];
    let mut mh3 = vec![0.0; nvar];

    let mut person = 0;
    while person < n {
        if event[person] == 0.0 {
            person += 1;
            continue;
        }

        let time = tstop[person];
        let mut denom = 0.0;
        let mut e_denom = 0.0;
        let mut deaths = 0.0;
        let mut meanwt = 0.0;

        a.iter_mut().for_each(|x| *x = 0.0);
        a2.iter_mut().for_each(|x| *x = 0.0);

        // Calculate risk set
        let mut k = person;
        while k < n && strata[k] == strata[person] {
            if tstart[k] < time {
                let risk = score[k] * weights[k];
                denom += risk;

                for i in 0..nvar {
                    a[i] += risk * covar_matrix[[i, k]];
                }

                if tstop[k] == time && event[k] == 1.0 {
                    deaths += 1.0;
                    e_denom += risk;
                    meanwt += weights[k];

                    for i in 0..nvar {
                        a2[i] += risk * covar_matrix[[i, k]];
                    }
                }
            }
            k += 1;
        }

        if deaths < 2.0 || method == 0 {
            // Breslow method
            let hazard = meanwt / denom;
            for i in 0..nvar {
                mean[i] = a[i] / denom;
            }

            let mut k = person;
            while k < n && strata[k] == strata[person] {
                if tstart[k] < time {
                    let risk = score[k];
                    for i in 0..nvar {
                        let diff = covar_matrix[[i, k]] - mean[i];
                        resid_matrix[[i, k]] -= diff * risk * hazard;
                    }

                    if tstop[k] == time && event[k] == 1.0 {
                        for i in 0..nvar {
                            let diff = covar_matrix[[i, k]] - mean[i];
                            resid_matrix[[i, k]] += diff;
                        }
                    }
                }
                k += 1;
            }
        } else {
            // Efron method
            let meanwt_norm = meanwt / deaths;
            let mut temp1 = 0.0;
            let mut temp2 = 0.0;

            mh1.iter_mut().for_each(|x| *x = 0.0);
            mh2.iter_mut().for_each(|x| *x = 0.0);
            mh3.iter_mut().for_each(|x| *x = 0.0);

            for dd in 0..deaths as usize {
                let downwt = dd as f64 / deaths;
                let d2 = denom - downwt * e_denom;
                let hazard = meanwt_norm / d2;

                temp1 += hazard;
                temp2 += (1.0 - downwt) * hazard;

                for i in 0..nvar {
                    mean[i] = (a[i] - downwt * a2[i]) / d2;
                    mh1[i] += mean[i] * hazard;
                    mh2[i] += mean[i] * (1.0 - downwt) * hazard;
                    mh3[i] += mean[i] / deaths;
                }
            }

            let mut k = person;
            while k < n && strata[k] == strata[person] {
                if tstart[k] < time {
                    let risk = score[k];

                    if tstop[k] == time && event[k] == 1.0 {
                        for i in 0..nvar {
                            resid_matrix[[i, k]] += covar_matrix[[i, k]] - mh3[i];
                            resid_matrix[[i, k]] -= risk * covar_matrix[[i, k]] * temp2;
                            resid_matrix[[i, k]] += risk * mh2[i];
                        }
                    } else {
                        for i in 0..nvar {
                            let term = risk * (covar_matrix[[i, k]] * temp1 - mh1[i]);
                            resid_matrix[[i, k]] -= term;
                        }
                    }
                }
                k += 1;
            }
        }

        while person < n && tstop[person] == time && strata[person] == strata[person] {
            person += 1;
        }
    }

    resid_matrix.into_raw_vec()
}

extendr_module! {
    mod coxscore;
    fn agscore2;
}
