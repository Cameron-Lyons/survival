use extendr_api::prelude::*;
use itertools::izip;

    let mut index = 0;
    let mut stride = 1;
    for (&i, &dim) in izip!(indices, dims) {
        index += i * stride;
        stride *= dim;
    }
    index
}

fn pystep(
    edim: usize,
    data: &mut [f64],
    efac: &[i32],
    edims: &[usize],
    ecut: &[&[f64]],
    tmax: f64,
) -> (f64, usize, usize, f64) {
    let mut et2 = tmax;
    let mut wt = 1.0;
    let mut limiting_dim = None;

    for j in 0..edim {
        if efac[j] != 0 {
            continue;
        }
        let cuts = ecut[j];
        let current = data[j];
        let pos = cuts.partition_point(|&x| x <= current);

        if pos < cuts.len() {
            let next_cut = cuts[pos];
            let delta = (next_cut - current).max(0.0);
            if delta < et2 {
                et2 = delta;
                limiting_dim = Some(j);
            }
        }
    }

    et2 = et2.min(tmax);
    let mut indices_current = vec![0; edim];
    let mut indices_next = vec![0; edim];

    for j in 0..edim {
        if efac[j] == 0 {
            data[j] += et2;
            let cuts = ecut[j];
            let pos = cuts.partition_point(|&x| x <= data[j]) - 1;
            indices_current[j] = pos.min(edims[j] - 1);
            indices_next[j] = (pos + 1).min(edims[j] - 1);
        } else {
            indices_current[j] = data[j] as usize - 1;
            indices_next[j] = indices_current[j];
        }
    }

    let indx = column_major_index(&indices_current, edims);
    let indx2 = column_major_index(&indices_next, edims);

    if let Some(dim) = limiting_dim {
        let current = data[dim] - et2;
        let cuts = ecut[dim];
        let pos = cuts.partition_point(|&x| x <= current) - 1;
        let next_cut = if pos + 1 < cuts.len() {
            cuts[pos + 1]
        } else {
            cuts[pos]
        };
        let prev_cut = cuts[pos];
        let width = next_cut - prev_cut;
        wt = if width > 0.0 {
            (current + et2 - prev_cut) / width
        } else {
            1.0
        };
        wt = wt.min(1.0).max(0.0);
    }

    (et2, indx, indx2, wt)
}

#[extendr]
fn pyears3b(
    death: Logical,
    efac: Integers,
    edims: Integers,
    ecut: Doubles,
    expect: Doubles,
    grpx: Integers,
    x: Doubles,
    y: Doubles,
    times: Doubles,
    ngrp: i32,
) -> List {
    let death = death.is_true();
    let edim = efac.len();
    let n = y.len();
    let ngrp = ngrp as usize;
    let ntime = times.len();
    let expect = expect.as_slice();
    let times = times.as_slice();
    let edims_rust: Vec<usize> = edims.iter().map(|&d| d as usize).collect();

    let mut ecut_slices = Vec::with_capacity(edim);
    let ecut_all = ecut.as_slice();
    let mut pos = 0;
    for (&ef, &dim) in izip!(efac.iter(), edims.iter()) {
        let len = if ef == 0 {
            dim as usize
        } else if ef > 1 {
            1 + (ef as usize - 1) * dim as usize
        } else {
            0
        };
        ecut_slices.push(&ecut_all[pos..pos + len]);
        pos += len;
    }

    let mut esurv = vec![0.0; ntime * ngrp];
    let mut nsurv = vec![0; ntime * ngrp];
    let mut wvec = vec![0.0; ntime * ngrp];

    for i in 0..n {
        let mut data: Vec<f64> = x.iter()
            .skip(i * edim)
            .take(edim)
            .cloned()
            .collect();
        let mut timeleft = y[i];
        let group = (grpx[i] - 1) as usize;
        let mut cumhaz = 0.0;
        let mut current_time = 0.0;

        for j in 0..ntime {
            if timeleft <= 0.0 {
                break;
            }

            let time_next = times[j];
            let thiscell = (time_next - current_time).min(timeleft);
            if thiscell <= 0.0 {
                continue;
            }

            let mut hazard = 0.0;
            let mut etime = thiscell;
            let mut current_data = data.clone();

            while etime > 0.0 {
                let (et2, indx, indx2, wt) = pystep(
                    edim,
                    &mut current_data,
                    efac.as_slice(),
                    &edims_rust,
                    &ecut_slices,
                    etime,
                );

                let rate = if wt < 1.0 && indx2 < expect.len() {
                    et2 * (wt * expect[indx] + (1.0 - wt) * expect[indx2])
                } else {
                    et2 * expect[indx]
                };
                hazard += rate;
                etime -= et2;
            }

            let index = j + ntime * group;
            if time_next == 0.0 {
                wvec[index] = 1.0;
                esurv[index] = if death { 0.0 } else { 1.0 };
            } else if !death {
                esurv[index] += (-(cumhaz + hazard)).exp() * thiscell;
                wvec[index] += (-cumhaz).exp() * thiscell;
            } else {
                esurv[index] += hazard * thiscell;
                wvec[index] += thiscell;
            }
            nsurv[index] += 1;

            cumhaz += hazard;
            current_time += thiscell;
            timeleft -= thiscell;
        }
    }

    for i in 0..(ntime * ngrp) {
        if wvec[i] > 0.0 {
            if death {
                esurv[i] = (-esurv[i] / wvec[i]).exp();
            } else {
                esurv[i] /= wvec[i];
            }
        } else if death {
            esurv[i] = (-esurv[i]).exp();
        }
    }

    list!(
        surv = esurv,
        n = nsurv
    )
}

extendr_module! {
    mod pyears;
    fn pyears3b;
}
