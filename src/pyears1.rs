pub fn pyears1(
    n: usize,
    ny: usize,
    doevent: i32,
    y: &[f64],
    weight: &[f64],
    edim: usize,
    efac: &[i32],
    edims: &[usize],
    ecut: &[f64],
    expect: &[f64],
    edata: &[f64],
    odim: usize,
    ofac: &[i32],
    odims: &[usize],
    ocut: &[f64],
    method: i32,
    odata: &[f64],
    pyears: &mut [f64],
    pn: &mut [f64],
    pcount: &mut [f64],
    pexpect: &mut [f64],
    offtable: &mut f64,
) {
    let (start, stop, event) = if ny == 3 || (ny == 2 && doevent == 0) {
        let start = &y[0..n];
        let stop = &y[n..2 * n];
        let event = if ny == 3 { &y[2 * n..3 * n] } else { &[] };
        (start, stop, event)
    } else {
        let stop = &y[0..n];
        let event = &y[n..2 * n];
        (&[].as_slice(), stop, event)
    };

    let mut ecut_slices = Vec::with_capacity(edim);
    let mut ecut_ptr = ecut;
    for j in 0..edim {
        let len = if efac[j] == 0 {
            edims[j]
        } else if efac[j] > 1 {
            1 + (efac[j] - 1) as usize * edims[j]
        } else {
            0
        };
        if len > 0 {
            ecut_slices.push(&ecut_ptr[0..len]);
            ecut_ptr = &ecut_ptr[len..];
        } else {
            ecut_slices.push(&[]);
        }
    }

    let mut ocut_slices = Vec::with_capacity(odim);
    let mut ocut_ptr = ocut;
    for j in 0..odim {
        if ofac[j] == 0 {
            let len = odims[j] + 1;
            ocut_slices.push(&ocut_ptr[0..len]);
            ocut_ptr = &ocut_ptr[len..];
        } else {
            ocut_slices.push(&[]);
        }
    }

    let mut eps = 0.0;
    for i in 0..n {
        let timeleft = if start.is_empty() {
            stop[i]
        } else {
            stop[i] - start[i]
        };
        if timeleft > 0.0 {
            eps = timeleft;
            break;
        }
    }
    for i in 0..n {
        let timeleft = if start.is_empty() {
            stop[i]
        } else {
            stop[i] - start[i]
        };
        if timeleft > 0.0 && timeleft < eps {
            eps = timeleft;
        }
    }
    eps *= 1e-8;

    *offtable = 0.0;

    for i in 0..n {
        let mut data = vec![0.0; odim];
        let mut data2 = vec![0.0; edim];

        for j in 0..odim {
            if ofac[j] == 1 || start.is_empty() {
                data[j] = odata[j * n + i];
            } else {
                data[j] = odata[j * n + i] + start[i];
            }
        }
        for j in 0..edim {
            if efac[j] == 1 || start.is_empty() {
                data2[j] = edata[j * n + i];
            } else {
                data2[j] = edata[j * n + i] + start[i];
            }
        }

        let mut timeleft = if start.is_empty() {
            stop[i]
        } else {
            stop[i] - start[i]
        };

        let mut cumhaz = 0.0;
        let mut index = -1;

        if timeleft <= eps && doevent == 1 {
            let (_, idx, _, _) = pystep(
                odim,
                &mut data.clone(),
                ofac,
                odims,
                &ocut_slices,
                timeleft,
                false,
            );
            index = idx;
        }

        while timeleft > eps {
            let mut data_current = data.clone();
            let (thiscell, idx, idx2, lwt) = pystep(
                odim,
                &mut data_current,
                ofac,
                odims,
                &ocut_slices,
                timeleft,
                false,
            );

            data.copy_from_slice(&data_current);

            if idx >= 0 {
                let idx = idx as usize;
                pyears[idx] += thiscell * weight[i];
                pn[idx] += 1.0;

                let mut etime = thiscell;
                let mut hazard = 0.0;
                let mut temp = 0.0;
                let mut data2_current = data2.clone();

                while etime > 0.0 {
                    let (et2, edx, edx2, elwt) = pystep(
                        edim,
                        &mut data2_current,
                        efac,
                        edims,
                        &ecut_slices,
                        etime,
                        true,
                    );

                    let lambda = if elwt < 1.0 {
                        elwt * expect[edx as usize] + (1.0 - elwt) * expect[edx2 as usize]
                    } else {
                        expect[edx as usize]
                    };

                    if method == 0 {
                        temp += (-hazard).exp() * (1.0 - (-lambda * et2).exp()) / lambda;
                    }
                    hazard += lambda * et2;

                    for j in 0..edim {
                        if efac[j] != 1 {
                            data2_current[j] += et2;
                        }
                    }
                    etime -= et2;
                }

                if method == 1 {
                    pexpect[idx] += hazard * weight[i];
                } else {
                    pexpect[idx] += (-cumhaz).exp() * temp * weight[i];
                }
                cumhaz += hazard;
            } else {
                *offtable += thiscell * weight[i];
                for j in 0..edim {
                    if efac[j] != 1 {
                        data2[j] += thiscell;
                    }
                }
            }

            for j in 0..odim {
                if ofac[j] == 0 {
                    data[j] += thiscell;
                }
            }

            timeleft -= thiscell;
            index = idx;
        }

        if index >= 0 && doevent == 1 {
            pcount[index as usize] += event[i] * weight[i];
        }
    }
}

fn pystep(
    dim: usize,
    data: &mut [f64],
    fac: &[i32],
    dims: &[usize],
    cut: &[&[f64]],
    timeleft: f64,
    is_lower: bool,
) -> (f64, i32, i32, f64) {
    let mut thiscell = timeleft;
    let mut index = 0;
    let mut indx2 = -1;
    let mut lwt = 1.0;
    let mut stride = 1;

    for j in 0..dim {
        if fac[j] == 1 {
            let cell = data[j] as i32;
            index += cell as usize * stride;
            stride *= dims[j];
        } else if fac[j] == 0 {
            let cuts = cut[j];
            if cuts.is_empty() {
                continue;
            }

            let pos = data[j];
            let mut k = 0;
            while k < cuts.len() - 1 && pos >= cuts[k] {
                k += 1;
            }
            k = k.saturating_sub(1);

            let next_edge = cuts[k + 1];
            let time_here = (next_edge - pos).max(0.0);

            if time_here < thiscell {
                thiscell = time_here;
            }

            data[j] += thiscell;

            index += k * stride;
            stride *= dims[j];
        } else {
            // Special handling (calendar year etc.) - placeholder
        }
    }

    (thiscell.min(timeleft), index as i32, indx2, lwt)
}
