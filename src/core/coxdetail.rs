#![allow(dead_code)]
pub fn coxdetail(
    nused: usize,
    nvar: usize,
    ndead: &mut usize,
    center: &[f64],
    y: &mut [f64],
    covar2: &mut [f64],
    strata: &mut [i32],
    score: &mut [f64],
    weights: &mut [f64],
    means2: &mut [f64],
    u2: &mut [f64],
    var: &mut [f64],
    rmat: &mut [i32],
    nrisk2: &mut [f64],
    work: &mut [f64],
) {
    let method = means2[0];
    let rflag = 1 - rmat[0] as i32;

    u2.fill(0.0);
    means2.fill(0.0);
    var.fill(0.0);

    let cmat_size = nvar * nvar;
    let (cmat_work, rest) = work.split_at_mut(cmat_size);
    let (cmat2_work, rest) = rest.split_at_mut(cmat_size);
    let (a_work, rest) = rest.split_at_mut(nvar);
    let (a2_work, _) = rest.split_at_mut(nvar);

    for i in 0..nvar {
        for person in 0..nused {
            let idx = i * nused + person;
            covar2[idx] -= center[i];
        }
    }

    let (start_slice, rest1) = y.split_at_mut(nused);
    let (stop_slice, rest2) = rest1.split_at_mut(nused);
    let event = &mut rest2[0..nused];
    let start = start_slice;
    let stop = stop_slice;

    let mut person = 0;
    let mut ideath = 0;

    while person < nused {
        if event[person] == 0.0 {
            person += 1;
        } else {
            let time = stop[person];
            let mut deaths = 0;
            let mut wdeath = 0.0;
            let mut efron_wt = 0.0;
            let mut meanwt = 0.0;
            let mut nrisk = 0;
            let mut denom = 0.0;

            a_work.fill(0.0);
            a2_work.fill(0.0);
            cmat_work.fill(0.0);
            cmat2_work.fill(0.0);

            let mut k = person;
            while k < nused && strata[k] != 1 {
                k += 1;
            }
            let end_stratum = k;

            for k in person..end_stratum {
                if start[k] < time {
                    nrisk += 1;
                    if rflag == 1 {
                        let idx = ideath * nused + k;
                        rmat[idx] = 1;
                    }

                    let risk = score[k] * weights[k];
                    denom += risk;

                    for i in 0..nvar {
                        let covar_ik = covar2[i * nused + k];
                        a_work[i] += risk * covar_ik;

                        for j in 0..=i {
                            let idx = i * nvar + j;
                            let covar_jk = covar2[j * nused + k];
                            cmat_work[idx] += risk * covar_ik * covar_jk;
                        }
                    }

                    if stop[k] == time && event[k] == 1.0 {
                        deaths += 1;
                        wdeath += weights[k];
                        efron_wt += risk;
                        meanwt += weights[k];

                        for i in 0..nvar {
                            let covar_ik = covar2[i * nused + k];
                            a2_work[i] += risk * covar_ik;

                            for j in 0..=i {
                                let idx = i * nvar + j;
                                let covar_jk = covar2[j * nused + k];
                                cmat2_work[idx] += risk * covar_ik * covar_jk;
                            }
                        }
                    }
                }
            }

            if deaths == 0 {
                person += 1;
                continue;
            }

            meanwt /= deaths as f64;
            let mut hazard = 0.0;
            let mut varhaz = 0.0;
            let mut itemp = -1;

            let mut k = person;
            while k < end_stratum && stop[k] == time {
                if event[k] == 1.0 {
                    itemp += 1;
                    let temp = itemp as f64 * method / deaths as f64;
                    let d2 = denom - temp * efron_wt;

                    hazard += meanwt / d2;
                    varhaz += meanwt.powi(2) / d2.powi(2);

                    for i in 0..nvar {
                        let temp2 = (a_work[i] - temp * a2_work[i]) / d2;
                        let means_idx = i * *ndead + ideath;
                        means2[means_idx] += (center[i] + temp2) / deaths as f64;

                        let u_idx = i * *ndead + ideath;
                        u2[u_idx] += weights[k] * covar2[i * nused + k] - meanwt * temp2;

                        for j in 0..nvar {
                            let cmat_idx = i * nvar + j;
                            let cmat_val = cmat_work[cmat_idx];
                            let cmat2_val = cmat2_work[cmat_idx];
                            let aj = a_work[j] - temp * a2_work[j];

                            let temp3 = ((cmat_val - temp * cmat2_val) - temp2 * aj) / d2;
                            let temp3 = temp3 * meanwt;

                            let var_idx = ideath * nvar * nvar + j * nvar + i;
                            var[var_idx] += temp3;

                            if j != i {
                                let sym_idx = ideath * nvar * nvar + i * nvar + j;
                                var[sym_idx] += temp3;
                            }
                        }
                    }
                }
                k += 1;
            }

            strata[ideath] = person as i32;
            score[ideath] = wdeath;
            start[ideath] = deaths as f64;
            stop[ideath] = nrisk as f64;
            event[ideath] = hazard;
            weights[ideath] = varhaz;
            nrisk2[ideath] = denom;

            ideath += 1;
            person = k;
        }
    }

    *ndead = ideath;
}
