pub struct CoxCountOutput {
    pub time: Vec<f64>,
    pub nrisk: Vec<i32>,
    pub index: Vec<i32>,
    pub status: Vec<i32>,
}

pub fn coxcount1(time: &[f64], status: &[f64], strata: &[i32]) -> CoxCountOutput {
    let n = time.len();
    let mut ntime = 0;
    let mut nrow = 0;
    let mut stratastart = 0;
    let mut nrisk = 0;

    let mut i = 0;
    while i < n {
        if strata[i] == 1 {
            stratastart = i;
            nrisk = 0;
        }
        nrisk += 1;

        if status[i] == 1.0 {
            let dtime = time[i];
            let mut j = i + 1;
            while j < n
                && (time[j] - dtime).abs() < f64::EPSILON
                && status[j] == 1.0
                && strata[j] == 0
            {
                nrisk += 1;
                j += 1;
            }
            ntime += 1;
            nrow += nrisk;
            i = j - 1;
        }
        i += 1;
    }

    let mut time_vec = Vec::with_capacity(ntime);
    let mut nrisk_vec = Vec::with_capacity(ntime);
    let mut index_vec = Vec::with_capacity(nrow);
    let mut status_vec = Vec::with_capacity(nrow);

    let mut stratastart = 0;
    let mut i = 0;

    while i < n {
        if strata[i] == 1 {
            stratastart = i;
        }

        if status[i] == 1.0 {
            let dtime = time[i];
            let mut j = i + 1;
            while j < n
                && (time[j] - dtime).abs() < f64::EPSILON
                && status[j] == 1.0
                && strata[j] == 0
            {
                j += 1;
            }

            for k in stratastart..i {
                status_vec.push(0);
                index_vec.push((k + 1) as i32);
            }

            for k in i..j {
                status_vec.push(1);
                index_vec.push((k + 1) as i32);
            }

            time_vec.push(dtime);
            nrisk_vec.push((j - stratastart) as i32);
            i = j - 1;
        }
        i += 1;
    }

    CoxCountOutput {
        time: time_vec,
        nrisk: nrisk_vec,
        index: index_vec,
        status: status_vec,
    }
}

pub fn coxcount2(
    time1: &[f64],
    time2: &[f64],
    status: &[f64],
    sort1: &[usize],
    sort2: &[usize],
    strata: &[i32],
) -> CoxCountOutput {
    let n = time1.len();
    let mut ntime = 0;
    let mut nrow = 0;
    let mut j = 0;
    let mut i = 0;
    let mut nrisk = 0;

    while i < n {
        let iptr = sort2[i];
        if strata[i] == 1 {
            nrisk = 0;
            j = i;
        }

        if status[iptr] == 1.0 {
            let dtime = time2[iptr];

            while j < i && time1[sort1[j]] >= dtime {
                nrisk -= 1;
                j += 1;
            }

            nrisk += 1;
            i += 1;

            while i < n && strata[i] == 0 && (time2[sort2[i]] - dtime).abs() < f64::EPSILON {
                nrisk += 1;
                i += 1;
            }

            nrow += nrisk;
            ntime += 1;
        } else {
            nrisk += 1;
            i += 1;
        }
    }

    let mut time_vec = Vec::with_capacity(ntime);
    let mut nrisk_vec = Vec::with_capacity(ntime);
    let mut index_vec = Vec::with_capacity(nrow);
    let mut status_vec = Vec::with_capacity(nrow);

    let mut atrisk = vec![None; n];
    let mut who = Vec::with_capacity(n);
    let mut j = 0;
    let mut i = 0;

    while i < n {
        let iptr = sort2[i];
        if strata[i] == 1 {
            atrisk.iter_mut().for_each(|x| *x = None);
            who.clear();
            j = i;
        }

        if status[iptr] == 0.0 {

            if atrisk[iptr].is_none() {
                atrisk[iptr] = Some(who.len());
                who.push(iptr);
            }
            i += 1;
        } else {
            let dtime = time2[iptr];

            while j < i {
                let jptr = sort1[j];
                if time1[jptr] >= dtime {
                    if let Some(pos) = atrisk[jptr] {
                        if pos < who.len() {
                            let last = who.pop().unwrap();
                            if pos < who.len() {
                                who[pos] = last;
                                atrisk[last] = Some(pos);
                            }
                            atrisk[jptr] = None;
                        }
                    }
                    j += 1;
                } else {
                    break;
                }
            }

            for &k in &who {
                status_vec.push(0);
                index_vec.push((k + 1) as i32);
            }

            let mut events = vec![iptr];
            i += 1;
            while i < n && strata[i] == 0 && (time2[sort2[i]] - dtime).abs() < f64::EPSILON {
                events.push(sort2[i]);
                i += 1;
            }

            for &k in &events {
                status_vec.push(1);
                index_vec.push((k + 1) as i32);

                if atrisk[k].is_none() {
                    atrisk[k] = Some(who.len());
                    who.push(k);
                }
            }

            time_vec.push(dtime);
            nrisk_vec.push(who.len() as i32);
        }
    }

    CoxCountOutput {
        time: time_vec,
        nrisk: nrisk_vec,
        index: index_vec,
        status: status_vec,
    }
}
