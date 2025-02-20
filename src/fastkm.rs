pub fn fastkm1(
    time: &[f64],
    status: &[f64],
    wt: &[f64],
    sort: &[usize],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = time.len();
    assert_eq!(status.len(), n);
    assert_eq!(wt.len(), n);
    assert_eq!(sort.len(), n);

    let mut nevent = 0;
    let mut ncount = vec![0.0; n];
    let mut dcount = vec![0.0; n];
    let mut ccount = vec![0.0; n];

    let mut dtime = time[sort[0]];
    let mut ntemp = 0.0;
    let mut dtemp = 0.0;
    let mut ctemp = 0.0;

    for i in 0..n {
        let p = sort[i];
        let current_time = time[p];
        let current_status = status[p];

        if current_time != dtime {
            if dtemp > 0.0 {
                nevent += 1;
            }
            dtemp = 0.0;
            ctemp = 0.0;
            dtime = current_time;
        }

        ntemp += wt[p];
        if current_status == 0.0 {
            ctemp += wt[p];
        } else {
            dtemp += wt[p];
        }

        ncount[i] = ntemp;
        dcount[i] = dtemp;
        ccount[i] = ctemp;
    }

    if dtemp > 0.0 {
        nevent += 1;
    }

    let mut S = vec![0.0; nevent];
    let mut G = vec![0.0; nevent];
    let mut nrisk = vec![0.0; nevent];
    let mut etime = vec![0.0; nevent];

    let mut k = 0;
    let mut stemp = 1.0;
    let mut gtemp = 1.0;
    let mut dfirst = true;
    let mut cfirst = true;
    let mut dtime_current = f64::NAN;
    let mut ctime_current = f64::NAN;

    for i in (0..n).rev() {
        if k >= nevent {
            break;
        }

        let p = sort[i];
        let current_time = time[p];
        let current_status = status[p];

        if current_status != 0.0 {
            // Death event
            if dfirst || (current_time != dtime_current) {
                dtime_current = current_time;
                dfirst = false;

                S[k] = stemp;
                G[k] = gtemp;
                nrisk[k] = ncount[i];
                etime[k] = dtime_current;

                stemp *= (ncount[i] - dcount[i]) / ncount[i];
                k += 1;
            }
        } else {
            // Censoring event
            if cfirst || (current_time != ctime_current) {
                ctime_current = current_time;
                cfirst = false;
                gtemp *= (ncount[i] - ccount[i]) / ncount[i];
            }
        }
    }

    (S, G, nrisk, etime)
}

pub fn fastkm2(
    tstart: &[f64],
    tstop: &[f64],
    status: &[f64],
    wt: &[f64],
    sort1: &[usize],
    sort2: &[usize],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = tstart.len();
    assert_eq!(tstop.len(), n);
    assert_eq!(status.len(), n);
    assert_eq!(wt.len(), n);
    assert_eq!(sort1.len(), n);
    assert_eq!(sort2.len(), n);

    let mut nevent = 0;
    let mut ncount = vec![0.0; n];
    let mut dcount = vec![0.0; n];

    let mut k = 0;
    let mut ntemp = 0.0;
    let mut i = 0;

    while i < n {
        let p2 = sort2[i];
        let current_time = tstop[p2];

        while k < n {
            let p1 = sort1[k];
            if tstart[p1] >= current_time {
                ntemp -= wt[p1];
                k += 1;
            } else {
                break;
            }
        }

        let mut dtemp = 0.0;
        let mut j = i;

        while j < n && tstop[sort2[j]] == current_time {
            let p = sort2[j];
            ntemp += wt[p];
            if status[p] == 1.0 {
                dtemp += wt[p];
            }
            j += 1;
        }

        for l in i..j {
            ncount[l] = ntemp;
            dcount[l] = dtemp;
        }

        if dtemp > 0.0 {
            nevent += 1;
        }

        i = j;
    }

    let mut S = vec![0.0; nevent];
    let mut nrisk = vec![0.0; nevent];
    let mut etime = vec![0.0; nevent];

    let mut k_out = 0;
    let mut stemp = 1.0;
    let mut dfirst = true;
    let mut dtime_current = f64::NAN;

    for i in (0..n).rev() {
        let p2 = sort2[i];
        let current_time = tstop[p2];
        let current_status = status[p2];

        if current_status == 1.0 {
            if dfirst || current_time != dtime_current {
                dtime_current = current_time;
                dfirst = false;

                S[k_out] = stemp;
                nrisk[k_out] = ncount[i];
                etime[k_out] = dtime_current;

                stemp *= (ncount[i] - dcount[i]) / ncount[i];
                k_out += 1;

                if k_out >= nevent {
                    break;
                }
            }
        }
    }

    (S, nrisk, etime)
}
