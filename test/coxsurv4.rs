pub struct CoxSurvResult {
    pub time: Vec<f64>,
    pub strata: Vec<f64>,
    pub count: Vec<[f64; 12]>,
    pub xbar1: Vec<Vec<f64>>,
    pub xbar2: Vec<Vec<f64>>,
}

pub fn coxsurv4(
    tstart: &[f64],
    stime: &[f64],
    status: &[f64],
    weight: &[f64],
    sort1: &[usize],
    sort2: &[usize],
    position: &[i32],
    strata: &[i32],
    xmat: &[Vec<f64>],
    risk: &[f64],
) -> CoxSurvResult {
    let nused = stime.len();
    let nvar = xmat.len();
    let mut ntime = 1;

    if nused > 0 {
        let mut current_stratum = strata[sort2[0]];
        let mut current_time = stime[sort2[0]];
        for i in 1..nused {
            let idx = sort2[i];
            if strata[idx] != current_stratum {
                ntime += 1;
                current_stratum = strata[idx];
                current_time = stime[idx];
            } else if stime[idx] != current_time {
                ntime += 1;
                current_time = stime[idx];
            }
        }
    }

    let mut time = vec![0.0; ntime];
    let mut strata_out = vec![0.0; ntime];
    let mut count = vec![[0.0; 12]; ntime];
    let mut xbar1 = vec![vec![0.0; nvar]; ntime];
    let mut xbar2 = vec![vec![0.0; nvar]; ntime];

    let mut person = 0;
    let mut person2 = 0;
    let mut istrat = if nused > 0 { strata[sort2[0]] } else { 0 };
    let mut dtime;
    let mut n = [0.0; 12];
    let mut xsum1 = vec![0.0; nvar];
    let mut xsum2 = vec![0.0; nvar];

    for itime in (0..ntime).rev() {
        if person >= nused {
            break;
        }

        let initial_idx = sort2[person];
        if person == 0 || strata[initial_idx] != istrat {
            while person2 < sort1.len() {
                let j = sort1[person2];
                if tstart[j] < dtime || strata[j] != istrat {
                    break;
                }
                if position[j] == 1 || position[j] == 3 {
                    n[10] += 1.0;
                    n[11] += weight[j];
                }
                person2 += 1;
            }

            n.iter_mut().for_each(|v| *v = 0.0);
            xsum1.iter_mut().for_each(|v| *v = 0.0);
            xsum2.iter_mut().for_each(|v| *v = 0.0);
            istrat = strata[initial_idx];
        }

        dtime = stime[initial_idx];
        time[itime] = dtime;
        strata_out[itime] = istrat as f64;

        while person < nused {
            let idx = sort2[person];
            if stime[idx] != dtime || strata[idx] != istrat {
                break;
            }

            let wt_val = weight[idx];
            let risk_val = risk[idx];

            n[0] += 1.0;
            n[1] += wt_val;
            n[2] += wt_val * risk_val;

            for k in 0..nvar {
                xsum1[k] += wt_val * risk_val * xmat[k][idx];
            }

            if status[idx] > 0.0 {
                for k in 0..nvar {
                    xsum2[k] += wt_val * risk_val * xmat[k][idx];
                }
                n[3] += 1.0;
                n[4] += wt_val;

                if position[idx] > 1 {
                    n[7] += 1.0;
                    n[8] += wt_val;
                    n[9] += wt_val * risk_val;
                }
            }

            if position[idx] > 1 {
                n[5] += 1.0;
                n[6] += wt_val;
            }

            person += 1;
        }

        while person2 < sort1.len() {
            let j = sort1[person2];
            if tstart[j] < dtime || strata[j] != istrat {
                break;
            }

            n[0] -= 1.0;
            n[1] -= weight[j];
            n[2] -= weight[j] * risk[j];
            for k in 0..nvar {
                xsum1[k] -= weight[j] * risk[j] * xmat[k][j];
            }

            if position[j] == 1 || position[j] == 3 {
                n[10] += 1.0;
                n[11] += weight[j];
            }

            person2 += 1;
        }

        count[itime] = n;
        let event_count = n[3].max(1.0);
        for k in 0..nvar {
            xbar1[itime][k] = xsum1[k] / event_count;
            xbar2[itime][k] = xsum2[k] / event_count;
        }
    }

    while person2 < sort1.len() {
        let j = sort1[person2];
        if position[j] == 1 || position[j] == 3 {
            n[10] += 1.0;
            n[11] += weight[j];
        }
        person2 += 1;
    }

    if ntime > 0 {
        count[0][10] = n[10];
        count[0][11] = n[11];
    }

    CoxSurvResult {
        time,
        strata: strata_out,
        count,
        xbar1,
        xbar2,
    }
}
