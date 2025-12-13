#[allow(dead_code)]
pub(crate) struct CoxSurvResult {
    pub time: Vec<f64>,
    pub strata: Vec<i32>,
    pub count: Vec<[f64; 7]>,
    pub xbar: Vec<Vec<f64>>,
    pub sresid: Vec<Vec<f64>>,
}

#[allow(dead_code)]
pub(crate) fn coxsurv3(
    y: &[(f64, f64)],
    xmat: &[Vec<f64>],
    strata: &[i32],
    risk: &[f64],
    weight: &[f64],
    sort2: &[usize],
    efron: bool,
) -> CoxSurvResult {
    let n_obs = y.len();
    let n_vars = xmat[0].len();

    let mut ntime = 0;
    let mut current_stratum = strata[sort2[0]];
    let mut last_time = y[sort2[0]].0 - 1.0;

    for &idx in sort2 {
        let (time, status) = y[idx];
        let stratum = strata[idx];

        if stratum != current_stratum {
            current_stratum = stratum;
            last_time = time - 1.0;
        }

        if status == 1.0 && (time - last_time).abs() > f64::EPSILON {
            ntime += 1;
            last_time = time;
        }
    }

    let mut result = CoxSurvResult {
        time: Vec::with_capacity(ntime),
        strata: Vec::with_capacity(ntime),
        count: Vec::with_capacity(ntime),
        xbar: vec![vec![0.0; n_vars]; ntime],
        sresid: vec![vec![0.0; n_vars]; n_obs],
    };

    let mut cum_haz = 0.0;
    let mut current_stratum = strata[sort2[n_obs - 1]];
    let mut strat_start = n_obs - 1;
    let mut itime = ntime - 1;

    let mut n = [0.0; 7];
    let mut xsum1 = vec![0.0; n_vars];
    let mut xsum2 = vec![0.0; n_vars];
    let mut xhaz = vec![0.0; n_vars];

    let mut i = n_obs as isize - 1;
    while i >= 0 {
        let idx = sort2[i as usize];
        let (time, _status) = y[idx];
        let stratum = strata[idx];

        if stratum != current_stratum {
            for k in (i + 1)..=strat_start as isize {
                let k_idx = sort2[k as usize];
                for var in 0..n_vars {
                    result.sresid[k_idx][var] +=
                        risk[k_idx] * (xhaz[var] - xmat[k_idx][var] * cum_haz);
                }
            }

            current_stratum = stratum;
            strat_start = i as usize;
            n = [0.0; 7];
            xsum1.iter_mut().for_each(|v| *v = 0.0);
            xhaz.iter_mut().for_each(|v| *v = 0.0);
            cum_haz = 0.0;
        }

        let mut deaths = 0;
        xsum2.iter_mut().for_each(|v| *v = 0.0);
        for j in (3..7).rev() {
            n[j] = 0.0;
        }

        let mut j = i;
        while j >= 0 {
            let j_idx = sort2[j as usize];
            let (j_time, j_status) = y[j_idx];
            let j_stratum = strata[j_idx];

            if j_time != time || j_stratum != current_stratum {
                break;
            }

            let wt = weight[j_idx];
            let rsk = risk[j_idx];

            n[0] += 1.0;
            n[1] += wt;
            n[2] += wt * rsk;

            for var in 0..n_vars {
                xsum1[var] += wt * rsk * xmat[j_idx][var];
                result.sresid[j_idx][var] = rsk * (cum_haz * xmat[j_idx][var] - xhaz[var]);
            }

            if j_status == 1.0 {
                deaths += 1;
                n[3] += 1.0;
                n[4] += wt;
                n[5] += wt * rsk;

                for var in 0..n_vars {
                    xsum2[var] += rsk * wt * xmat[j_idx][var];
                }
            }

            j -= 1;
        }

        if deaths > 0 {
            let mean_wt = n[4] / n[3];
            let mut xmean = vec![0.0; n_vars];

            if deaths == 1 || !efron {
                let hazard = n[4] / n[2];
                cum_haz += hazard;

                for var in 0..n_vars {
                    xmean[var] = xsum1[var] / n[2];
                    xhaz[var] += xmean[var] * hazard;

                    for k in (j + 1)..=i {
                        let k_idx = sort2[k as usize];
                        result.sresid[k_idx][var] += xmat[k_idx][var] - xmean[var];
                    }
                }

                n[6] = n[2];
            } else {
                let mut total_hazard = 0.0;
                let mut total_xhaz = vec![0.0; n_vars];

                for d in 0..deaths {
                    let downwt = d as f64 / deaths as f64;
                    let denominator = n[2] - downwt * n[5];
                    let hazard_step = mean_wt / denominator;
                    total_hazard += hazard_step;

                    for var in 0..n_vars {
                        let tmean = (xsum1[var] - downwt * xsum2[var]) / denominator;
                        xmean[var] += tmean / deaths as f64;
                        total_xhaz[var] += tmean * hazard_step;

                        for k in (j + 1)..=i {
                            let k_idx = sort2[k as usize];
                            let diff = xmat[k_idx][var] - tmean;
                            result.sresid[k_idx][var] += diff / deaths as f64;
                            result.sresid[k_idx][var] += diff * risk[k_idx] * hazard_step * downwt;
                        }
                    }
                }

                cum_haz += total_hazard;
                for var in 0..n_vars {
                    xhaz[var] += total_xhaz[var];
                }
                n[6] = deaths as f64 / (total_hazard / mean_wt);
            }

            result.time.push(time);
            result.strata.push(current_stratum);
            result.count.push(n);

            result.xbar[itime][..n_vars].copy_from_slice(&xmean[..n_vars]);

            itime -= 1;
        }

        i = j;
    }

    for k in 0..=strat_start {
        let k_idx = sort2[k];
        for var in 0..n_vars {
            result.sresid[k_idx][var] += risk[k_idx] * (xhaz[var] - xmat[k_idx][var] * cum_haz);
        }
    }

    result
}
