#[allow(dead_code)]
pub(crate) struct CoxSurvResult {
    pub time: Vec<f64>,
    pub strata: Vec<f64>,
    pub count: Vec<f64>,
    pub xbar1: Vec<f64>,
    pub xbar2: Vec<f64>,
}

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn coxsurv4(
    y: &[f64],
    weight: &[f64],
    sort1: &[usize],
    sort2: &[usize],
    position: &[i32],
    strata: &[i32],
    xmat: &[f64],
    risk: &[f64],
) -> CoxSurvResult {
    let nused = sort2.len();
    let nvar = xmat.len() / nused;

    let (tstart, rest) = y.split_at(nused);
    let (stime, status) = rest.split_at(nused);

    let mut ntime = 1;
    let mut current_stratum = strata[sort2[0]];
    let mut current_time = stime[sort2[0]];

    for &i in &sort2[1..] {
        if strata[i] != current_stratum {
            ntime += 1;
            current_stratum = strata[i];
            current_time = stime[i];
        } else if stime[i] != current_time {
            ntime += 1;
            current_time = stime[i];
        }
    }

    let mut rtime = vec![0.0; ntime];
    let mut rstrat = vec![0.0; ntime];
    let mut rn = vec![0.0; ntime * 12];
    let mut rx1 = vec![0.0; ntime * nvar];
    let mut rx2 = vec![0.0; ntime * nvar];

    let mut person = 0;
    let mut person2 = 0;
    let mut current_stratum = strata[sort2[0]];
    let mut n = [0.0; 12];
    let mut xsum1 = vec![0.0; nvar];
    let mut xsum2 = vec![0.0; nvar];

    for itime in (0..ntime).rev() {
        let i2 = sort2[person];

        if person == 0 || strata[i2] != current_stratum {
            if person > 0 {
                while person2 < nused {
                    let j2 = sort1[person2];
                    if tstart[j2] >= current_time && strata[j2] == current_stratum {
                        n[10] += 1.0;
                        n[11] += weight[j2];
                        person2 += 1;
                    } else {
                        break;
                    }
                }
            }

            current_stratum = strata[i2];
            n = [0.0; 12];
            xsum1 = vec![0.0; nvar];
        }

        let dtime = stime[i2];
        rtime[itime] = dtime;
        rstrat[itime] = current_stratum as f64;

        for k in 3..12 {
            n[k] = 0.0;
        }

        while person < nused {
            let i2 = sort2[person];
            if stime[i2] != dtime || strata[i2] != current_stratum {
                break;
            }

            let wt = weight[i2];
            let r = risk[i2];
            let pos = position[i2];

            n[0] += 1.0;
            n[1] += wt;
            n[2] += wt * r;

            for k in 0..nvar {
                xsum1[k] += wt * r * xmat[k * nused + i2];
            }

            if status[i2] > 0.0 {
                n[3] += 1.0;
                n[4] += wt;

                for k in 0..nvar {
                    xsum2[k] += wt * r * xmat[k * nused + i2];
                }

                if pos > 1 {
                    n[7] += 1.0;
                    n[8] += wt;
                    n[9] += wt * r;
                }
            }

            if pos > 1 {
                n[5] += 1.0;
                n[6] += wt;
            }

            person += 1;
        }

        while person2 < nused {
            let j2 = sort1[person2];
            if tstart[j2] >= dtime && strata[j2] == current_stratum {
                n[0] -= 1.0;
                n[1] -= weight[j2];
                n[2] -= weight[j2] * risk[j2];

                for k in 0..nvar {
                    xsum1[k] -= xmat[k * nused + j2] * weight[j2] * risk[j2];
                }

                if position[j2] == 1 || position[j2] == 3 {
                    n[10] += 1.0;
                    n[11] += weight[j2];
                }

                person2 += 1;
            } else {
                break;
            }
        }

        for k in 0..12 {
            rn[itime * 12 + k] = n[k];
        }

        let denom = if n[3] > 0.0 { n[3] } else { 1.0 };
        for k in 0..nvar {
            rx1[itime * nvar + k] = xsum1[k] / denom;
            rx2[itime * nvar + k] = xsum2[k] / denom;
        }

        xsum2 = vec![0.0; nvar];
    }

    while person2 < nused {
        let j2 = sort1[person2];
        n[10] += 1.0;
        n[11] += weight[j2];
        person2 += 1;
    }
    rn[10] = n[10];
    rn[11] = n[11];

    CoxSurvResult {
        time: rtime,
        strata: rstrat,
        count: rn,
        xbar1: rx1,
        xbar2: rx2,
    }
}
