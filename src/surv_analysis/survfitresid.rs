#[allow(dead_code)]
pub(crate) struct SurvivalResult {
    pub influence_pstate: Vec<Vec<f64>>,
    pub influence_auc: Option<Vec<Vec<f64>>>,
}

#[derive(Debug)]
pub(crate) struct SurvfitResidError {
    pub message: String,
}

impl std::fmt::Display for SurvfitResidError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for SurvfitResidError {}

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn survfitresid(
    y: &[f64],
    sort1: &[usize],
    sort2: &[usize],
    cstate: &[i32],
    wt: &[f64],
    p0: &[f64],
    i0: &[f64],
    otime: &[f64],
    starttime: f64,
    doauc: bool,
) -> Result<SurvivalResult, SurvfitResidError> {
    let ncoly = if sort1.is_empty() {
        return Err(SurvfitResidError {
            message: "sort1 array cannot be empty".to_string(),
        });
    } else if y.len().is_multiple_of(sort1.len()) {
        y.len() / sort1.len()
    } else {
        return Err(SurvfitResidError {
            message: format!(
                "Invalid Y matrix dimensions: y.len()={} is not divisible by sort1.len()={}",
                y.len(),
                sort1.len()
            ),
        });
    };
    let nrowy = sort1.len();
    let (entry, etime, status) = if ncoly == 2 {
        (vec![], y.to_vec(), y[nrowy..].to_vec())
    } else {
        (
            y[..nrowy].to_vec(),
            y[nrowy..2 * nrowy].to_vec(),
            y[2 * nrowy..].to_vec(),
        )
    };

    let nobs = sort2.len();
    let nstate = i0.len() / nobs;
    let nout = otime.len();
    let mut starttime = starttime;
    let mut itime = 0;

    let mut influence_pstate = vec![vec![0.0; nout * nstate]; nobs];
    let mut influence_auc = if doauc {
        Some(vec![vec![0.0; nout * nstate]; nobs])
    } else {
        None
    };

    #[allow(clippy::needless_range_loop)]
    for k in 0..nstate {
        #[allow(clippy::needless_range_loop)]
        for i in 0..nobs {
            influence_pstate[i][k] = i0[i * nstate + k];
        }
    }

    let mut atrisk = if ncoly == 3 {
        vec![false; nobs]
    } else {
        vec![true; nobs]
    };
    let mut ws = vec![0.0; nstate];
    let mut nrisk = vec![0; nstate];
    let mut pstate = p0.to_vec();
    let mut eptr = 0;

    if ncoly == 3 {
        for &idx in sort1 {
            if entry[idx] < starttime {
                atrisk[idx] = true;
                let s = cstate[idx] as usize;
                ws[s] += wt[idx];
                nrisk[s] += 1;
                eptr += 1;
            }
        }
    } else {
        #[allow(clippy::needless_range_loop)]
        for i in 0..nobs {
            let s = cstate[i] as usize;
            ws[s] += wt[i];
            nrisk[s] += 1;
        }
    }

    let mut i = 0;
    while i < nobs && itime < nout {
        let p2 = sort2[i];
        let ctime = etime[p2];

        while itime < nout && otime[itime] < ctime {
            if doauc {
                let delta = otime[itime] - starttime;
                #[allow(clippy::needless_range_loop)]
                for k in 0..nstate {
                    #[allow(clippy::needless_range_loop)]
                    for j in 0..nobs {
                        influence_auc.as_mut().unwrap()[j][itime * nstate + k] +=
                            influence_pstate[j][itime * nstate + k] * delta;
                    }
                }
                starttime = otime[itime];
            }
            itime += 1;
        }

        if itime >= nout {
            break;
        }

        if ncoly == 3 {
            while eptr < nobs {
                let p1 = sort1[eptr];
                if entry[p1] < ctime {
                    atrisk[p1] = true;
                    let s = cstate[p1] as usize;
                    ws[s] += wt[p1];
                    nrisk[s] += 1;
                    eptr += 1;
                } else {
                    break;
                }
            }
        }

        let mut cmat = vec![vec![0.0; nstate]; nstate];
        let mut nevent = 0;
        let mut _wevent = 0.0;
        let mut transitions = Vec::new();

        #[allow(clippy::needless_range_loop)]
        for j in i..nobs {
            let p2j = sort2[j];
            if etime[p2j] > ctime {
                break;
            }
            if status[p2j] != 0.0 {
                let oldstate = cstate[p2j] as usize;
                let newstate = (status[p2j] as usize) - 1;
                if oldstate != newstate {
                    transitions.push((p2j, oldstate, newstate));
                    cmat[oldstate][newstate] += wt[p2j] / ws[oldstate];
                    cmat[oldstate][oldstate] -= wt[p2j] / ws[oldstate];
                    nevent += 1;
                    _wevent += wt[p2j];
                }
            }
        }

        if nevent > 0 {
            if doauc {
                let delta = ctime - starttime;
                #[allow(clippy::needless_range_loop)]
                for k in 0..nstate {
                    #[allow(clippy::needless_range_loop)]
                    for j in 0..nobs {
                        influence_auc.as_mut().unwrap()[j][itime * nstate + k] +=
                            influence_pstate[j][itime * nstate + k] * delta;
                    }
                }
                starttime = ctime;
            }

            if nevent == 1 {
                let (psave, oldstate, newstate) = transitions[0];
                let temp = -cmat[oldstate][oldstate];
                let temp2 = pstate[oldstate] / ws[oldstate];

                #[allow(clippy::needless_range_loop)]
                for j in 0..nobs {
                    let inf_old = influence_pstate[j][itime * nstate + oldstate];
                    influence_pstate[j][itime * nstate + newstate] += temp * inf_old;
                    influence_pstate[j][itime * nstate + oldstate] -= temp * inf_old;
                }

                influence_pstate[psave][itime * nstate + newstate] += temp2;
                influence_pstate[psave][itime * nstate + oldstate] -= temp2;

                #[allow(clippy::needless_range_loop)]
                for j in i..nobs {
                    let p2j = sort2[j];
                    if atrisk[p2j] && cstate[p2j] as usize == oldstate {
                        influence_pstate[p2j][itime * nstate + oldstate] += temp * temp2;
                        influence_pstate[p2j][itime * nstate + newstate] -= temp * temp2;
                    }
                }
            } else if nevent > 1 {
                let mut tempvec = vec![0.0; nstate];
                #[allow(clippy::needless_range_loop)]
                for j in 0..nobs {
                    #[allow(clippy::needless_range_loop)]
                    for k in 0..nstate {
                        tempvec[k] = 0.0;
                        #[allow(clippy::needless_range_loop)]
                        for kk in 0..nstate {
                            tempvec[k] += influence_pstate[j][itime * nstate + kk] * cmat[kk][k];
                        }
                    }
                    #[allow(clippy::needless_range_loop)]
                    for k in 0..nstate {
                        influence_pstate[j][itime * nstate + k] += tempvec[k];
                    }
                }

                #[allow(clippy::needless_range_loop)]
                for j in i..nobs {
                    let p2j = sort2[j];
                    if atrisk[p2j] {
                        let oldstate = cstate[p2j] as usize;
                        let temp2 = pstate[oldstate] / ws[oldstate];
                        #[allow(clippy::needless_range_loop)]
                        for k in 0..nstate {
                            influence_pstate[p2j][itime * nstate + k] -= cmat[oldstate][k] * temp2;
                        }
                    }
                }

                for (p2j, oldstate, newstate) in transitions {
                    let temp2 = pstate[oldstate] / ws[oldstate];
                    influence_pstate[p2j][itime * nstate + oldstate] -= temp2;
                    influence_pstate[p2j][itime * nstate + newstate] += temp2;
                }
            }
        }

        let mut new_pstate = vec![0.0; nstate];
        for j in 0..nstate {
            for k in 0..nstate {
                new_pstate[j] += pstate[k] * cmat[k][j];
            }
        }
        for j in 0..nstate {
            pstate[j] += new_pstate[j];
        }

        while i < nobs {
            let p2i = sort2[i];
            if etime[p2i] > ctime {
                break;
            }
            let oldstate = cstate[p2i] as usize;
            ws[oldstate] -= wt[p2i];
            nrisk[oldstate] -= 1;
            atrisk[p2i] = false;
            i += 1;
        }
    }

    while itime < nout {
        if doauc {
            let delta = otime[itime] - starttime;
            #[allow(clippy::needless_range_loop)]
            for k in 0..nstate {
                #[allow(clippy::needless_range_loop)]
                for j in 0..nobs {
                    influence_auc.as_mut().unwrap()[j][itime * nstate + k] +=
                        influence_pstate[j][itime * nstate + k] * delta;
                }
            }
            starttime = otime[itime];
        }
        itime += 1;
    }

    Ok(SurvivalResult {
        influence_pstate,
        influence_auc,
    })
}
