#![allow(clippy::needless_range_loop)]
#[allow(dead_code)]
pub(crate) struct CoxDetailInput<'a> {
    pub center: &'a [f64],
    pub y: &'a mut [f64],
    pub covar2: &'a mut [f64],
    pub strata: &'a mut [i32],
    pub score: &'a mut [f64],
    pub weights: &'a mut [f64],
}

#[allow(dead_code)]
pub(crate) struct CoxDetailOutput<'a> {
    pub ndead: &'a mut usize,
    pub means2: &'a mut [f64],
    pub u2: &'a mut [f64],
    pub var: &'a mut [f64],
    pub rmat: &'a mut [i32],
    pub nrisk2: &'a mut [f64],
}

#[allow(dead_code)]
pub(crate) struct CoxDetailParams {
    pub nused: usize,
    pub nvar: usize,
}

#[allow(dead_code)]
pub(crate) fn coxdetail(
    params: CoxDetailParams,
    input: CoxDetailInput,
    output: CoxDetailOutput,
    work: &mut [f64],
) {
    let method = output.means2[0];
    let rflag = 1 - output.rmat[0];

    output.u2.fill(0.0);
    output.means2.fill(0.0);
    output.var.fill(0.0);

    let cmat_size = params.nvar * params.nvar;
    let (cmat_work, rest) = work.split_at_mut(cmat_size);
    let (cmat2_work, rest) = rest.split_at_mut(cmat_size);
    let (a_work, rest) = rest.split_at_mut(params.nvar);
    let (a2_work, _) = rest.split_at_mut(params.nvar);

    for i in 0..params.nvar {
        for person in 0..params.nused {
            let idx = i * params.nused + person;
            input.covar2[idx] -= input.center[i];
        }
    }

    let (start_slice, rest1) = input.y.split_at_mut(params.nused);
    let (stop_slice, rest2) = rest1.split_at_mut(params.nused);
    let event = &mut rest2[0..params.nused];
    let start = start_slice;
    let stop = stop_slice;

    let mut person = 0;
    let mut ideath = 0;

    while person < params.nused {
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
            while k < params.nused && input.strata[k] != 1 {
                k += 1;
            }
            let end_stratum = k;

            for k in person..end_stratum {
                if start[k] < time {
                    nrisk += 1;
                    if rflag == 1 {
                        let idx = ideath * params.nused + k;
                        output.rmat[idx] = 1;
                    }

                    let risk = input.score[k] * input.weights[k];
                    denom += risk;

                    for i in 0..params.nvar {
                        let covar_ik = input.covar2[i * params.nused + k];
                        a_work[i] += risk * covar_ik;

                        for j in 0..=i {
                            let idx = i * params.nvar + j;
                            let covar_jk = input.covar2[j * params.nused + k];
                            cmat_work[idx] += risk * covar_ik * covar_jk;
                        }
                    }

                    if stop[k] == time && event[k] == 1.0 {
                        deaths += 1;
                        wdeath += input.weights[k];
                        efron_wt += risk;
                        meanwt += input.weights[k];

                        for i in 0..params.nvar {
                            let covar_ik = input.covar2[i * params.nused + k];
                            a2_work[i] += risk * covar_ik;

                            for j in 0..=i {
                                let idx = i * params.nvar + j;
                                let covar_jk = input.covar2[j * params.nused + k];
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

                    for i in 0..params.nvar {
                        let temp2 = (a_work[i] - temp * a2_work[i]) / d2;
                        let means_idx = i * *output.ndead + ideath;
                        output.means2[means_idx] += (input.center[i] + temp2) / deaths as f64;

                        let u_idx = i * *output.ndead + ideath;
                        output.u2[u_idx] +=
                            input.weights[k] * input.covar2[i * params.nused + k] - meanwt * temp2;

                        for j in 0..params.nvar {
                            let cmat_idx = i * params.nvar + j;
                            let cmat_val = cmat_work[cmat_idx];
                            let cmat2_val = cmat2_work[cmat_idx];
                            let aj = a_work[j] - temp * a2_work[j];

                            let temp3 = ((cmat_val - temp * cmat2_val) - temp2 * aj) / d2;
                            let temp3 = temp3 * meanwt;

                            let var_idx = ideath * params.nvar * params.nvar + j * params.nvar + i;
                            output.var[var_idx] += temp3;

                            if j != i {
                                let sym_idx =
                                    ideath * params.nvar * params.nvar + i * params.nvar + j;
                                output.var[sym_idx] += temp3;
                            }
                        }
                    }
                }
                k += 1;
            }

            input.strata[ideath] = person as i32;
            input.score[ideath] = wdeath;
            start[ideath] = deaths as f64;
            stop[ideath] = nrisk as f64;
            event[ideath] = hazard;
            input.weights[ideath] = varhaz;
            output.nrisk2[ideath] = denom;

            ideath += 1;
            person = k;
        }
    }

    *output.ndead = ideath;
}
