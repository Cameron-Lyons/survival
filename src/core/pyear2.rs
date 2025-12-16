#[allow(dead_code)]
fn find_interval(cuts: &[f64], x: f64) -> Option<usize> {
    if cuts.is_empty() {
        return None;
    }
    if x < cuts[0] || x >= cuts[cuts.len() - 1] {
        return None;
    }
    match cuts.binary_search_by(|probe| probe.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal)) {
        Ok(i) => {
            if i < cuts.len() - 1 {
                Some(i)
            } else {
                None
            }
        }
        Err(i) => {
            if i > 0 && i <= cuts.len() {
                Some(i - 1)
            } else {
                None
            }
        }
    }
}

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn pystep(
    odim: usize,
    data: &[f64],
    ofac: &[i32],
    odims: &[usize],
    ocut: &[&[f64]],
    timeleft: f64,
) -> (f64, i32) {
    let mut maxtime = timeleft;
    let mut intervals = vec![0; odim];
    let mut valid = true;

    for j in 0..odim {
        if ofac[j] == 0 {
            let cuts = ocut[j];
            if cuts.is_empty() {
                valid = false;
                break;
            }
            let x = data[j];
            match find_interval(cuts, x) {
                Some(i) => {
                    let next_cut = cuts[i + 1];
                    let time_to_next = next_cut - x;
                    if time_to_next < maxtime {
                        maxtime = time_to_next;
                    }
                    intervals[j] = i;
                }
                None => {
                    valid = false;
                    break;
                }
            }
        }
    }

    if !valid {
        return (0.0, -1);
    }

    let mut index = 0;
    for j in 0..odim {
        let idx_j = if ofac[j] == 1 {
            data[j] as usize
        } else {
            intervals[j]
        };

        if idx_j >= odims[j] {
            return (maxtime, -1);
        }

        index = index * odims[j] + idx_j;
    }

    (maxtime, index as i32)
}

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn pyears2(
    n: usize,
    ny: usize,
    doevent: bool,
    y: &[f64],
    wt: &[f64],
    odim: usize,
    ofac: &[i32],
    odims: &[usize],
    socut: &[f64],
    sodata: &[f64],
    pyears: &mut [f64],
    pn: &mut [f64],
    pcount: &mut [f64],
    offtable: &mut f64,
) {
    let dostart = ny == 3 || (ny == 2 && !doevent);

    let (start_slice, stop_slice, event_slice) = if dostart {
        let stop_start = n;
        let event_start = 2 * n;
        (
            &y[0..n],
            &y[stop_start..stop_start + n],
            &y[event_start..event_start + n],
        )
    } else {
        let event_start = n;
        (
            &[] as &[f64],
            &y[0..n],
            if doevent {
                &y[event_start..event_start + n]
            } else {
                &[]
            },
        )
    };

    let mut ocut = Vec::with_capacity(odim);
    let mut current = 0;
    for j in 0..odim {
        if ofac[j] == 0 {
            let len = odims[j] + 1;
            ocut.push(&socut[current..current + len]);
            current += len;
        } else {
            ocut.push(&[]);
        }
    }

    let odata: Vec<_> = (0..odim).map(|j| &sodata[j * n..(j + 1) * n]).collect();

    let mut eps = 0.0;
    for i in 0..n {
        let start_i = if dostart { start_slice[i] } else { 0.0 };
        let timeleft = stop_slice[i] - start_i;
        if timeleft > 0.0 {
            eps = timeleft;
            break;
        }
    }
    for i in 0..n {
        let start_i = if dostart { start_slice[i] } else { 0.0 };
        let timeleft = stop_slice[i] - start_i;
        if timeleft > 0.0 && timeleft < eps {
            eps = timeleft;
        }
    }
    eps *= 1e-8;

    *offtable = 0.0;
    for i in 0..n {
        let start_i = if dostart { start_slice[i] } else { 0.0 };
        let stop_i = stop_slice[i];
        let event_i = if doevent && !event_slice.is_empty() {
            event_slice[i]
        } else {
            0.0
        };
        let weight = wt[i];

        let mut data = vec![0.0; odim];
        for j in 0..odim {
            data[j] = if ofac[j] == 1 || !dostart {
                odata[j][i]
            } else {
                odata[j][i] + start_i
            };
        }

        let mut timeleft = stop_i - start_i;
        let mut last_index = -1;

        if timeleft <= eps && doevent {
            let (_, index) = pystep(odim, &data, ofac, odims, &ocut, timeleft);
            last_index = index;
        }

        while timeleft > eps {
            let (thiscell, index) = pystep(odim, &data, ofac, odims, &ocut, timeleft);
            last_index = index;

            if index >= 0 {
                let idx = index as usize;
                pyears[idx] += thiscell * weight;
                pn[idx] += 1.0;
            } else {
                *offtable += thiscell * weight;
            }

            for j in 0..odim {
                if ofac[j] == 0 {
                    data[j] += thiscell;
                }
            }

            timeleft -= thiscell;
        }

        if doevent && last_index >= 0 {
            let idx = last_index as usize;
            pcount[idx] += event_i * weight;
        }
    }
}
