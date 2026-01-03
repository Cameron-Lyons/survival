use ndarray::{Array1, Array2, s};
use pyo3::prelude::*;
use std::error::Error;
#[pyclass]
#[derive(Debug, Clone)]
pub struct SurvFitAJ {
    #[pyo3(get)]
    pub n_risk: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub n_event: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub n_censor: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub pstate: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub cumhaz: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub std_err: Option<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub std_chaz: Option<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub std_auc: Option<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub influence: Option<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub n_enter: Option<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub n_transition: Vec<Vec<f64>>,
}
#[derive(Debug)]
struct SurvFitAJComputed {
    pub n_risk: Array2<f64>,
    pub n_event: Array2<f64>,
    pub n_censor: Array2<f64>,
    pub pstate: Array2<f64>,
    pub cumhaz: Array2<f64>,
    pub std_err: Option<Array2<f64>>,
    pub std_chaz: Option<Array2<f64>>,
    pub std_auc: Option<Array2<f64>>,
    pub influence: Option<Array2<f64>>,
    pub n_enter: Option<Array2<f64>>,
    pub n_transition: Array2<f64>,
}
impl SurvFitAJComputed {
    fn into_python_result(self) -> SurvFitAJ {
        let array2_to_vec = |arr: Array2<f64>| -> Vec<Vec<f64>> {
            arr.outer_iter().map(|row| row.to_vec()).collect()
        };
        let option_array2_to_vec =
            |opt: Option<Array2<f64>>| -> Option<Vec<Vec<f64>>> { opt.map(array2_to_vec) };
        SurvFitAJ {
            n_risk: array2_to_vec(self.n_risk),
            n_event: array2_to_vec(self.n_event),
            n_censor: array2_to_vec(self.n_censor),
            pstate: array2_to_vec(self.pstate),
            cumhaz: array2_to_vec(self.cumhaz),
            std_err: option_array2_to_vec(self.std_err),
            std_chaz: option_array2_to_vec(self.std_chaz),
            std_auc: option_array2_to_vec(self.std_auc),
            influence: option_array2_to_vec(self.influence),
            n_enter: option_array2_to_vec(self.n_enter),
            n_transition: array2_to_vec(self.n_transition),
        }
    }
}

pub struct SurvFitAJData<'a> {
    pub y: &'a [f64],
    pub sort1: &'a [usize],
    pub sort2: &'a [usize],
    pub utime: &'a [f64],
    pub cstate: &'a [usize],
    pub wt: &'a [f64],
    pub grp: &'a [usize],
    pub position: &'a [usize],
}

pub struct SurvFitAJParams<'a> {
    pub ngrp: usize,
    pub p0: &'a [f64],
    pub i0: &'a [f64],
    pub sefit: i32,
    pub entry: bool,
    pub hindx: &'a Array2<usize>,
    pub trmat: &'a Array2<usize>,
    pub t0: f64,
}

fn compute_survfitaj(
    data: &SurvFitAJData<'_>,
    params: &SurvFitAJParams<'_>,
) -> Result<SurvFitAJComputed, Box<dyn Error>> {
    let y = data.y;
    let sort1 = data.sort1;
    let sort2 = data.sort2;
    let utime = data.utime;
    let cstate = data.cstate;
    let wt = data.wt;
    let grp = data.grp;
    let position = data.position;
    let ngrp = params.ngrp;
    let p0 = params.p0;
    let i0 = params.i0;
    let sefit = params.sefit;
    let entry = params.entry;
    let hindx = params.hindx;
    let trmat = params.trmat;
    let t0 = params.t0;
    let ntime = utime.len();
    let _n = y.len() / 3;
    let nused = sort1.len();
    let nstate = p0.len();
    let nhaz = trmat.nrows();
    let mut n_risk = Array2::zeros((ntime, 2 * nstate));
    let mut n_event = Array2::zeros((ntime, nstate));
    let mut n_censor = Array2::zeros((ntime, 2 * nstate));
    let mut n_transition = Array2::zeros((ntime, 2 * nhaz));
    let mut pstate = Array2::zeros((ntime, nstate));
    let mut cumhaz = Array2::zeros((ntime, nhaz));
    let mut n_enter = if entry {
        Some(Array2::zeros((ntime, 2 * nstate)))
    } else {
        None
    };
    let mut ntemp = Array1::zeros(2 * nstate);
    let mut phat = Array1::from_vec(p0.to_vec());
    let mut chaz = Array1::zeros(nhaz);
    let mut person1 = nused - 1;
    let mut person2 = nused - 1;
    for i in (0..ntime).rev() {
        let ctime = utime[i];
        while person1 > 0 && y[sort1[person1] * 3] >= ctime {
            let idx = sort1[person1];
            let cs = cstate[idx];
            ntemp[cs] -= wt[idx];
            ntemp[cs + nstate] -= 1.0;
            if entry
                && (position[idx] & 0x1) != 0
                && let Some(ref mut ne) = n_enter
            {
                ne[[i, cs]] += wt[idx];
                ne[[i, cs + nstate]] += 1.0;
            }
            person1 -= 1;
        }
        while person2 > 0 && y[sort2[person2] * 3 + 1] >= ctime {
            let idx = sort2[person2];
            let cs = cstate[idx];
            ntemp[cs] += wt[idx];
            ntemp[cs + nstate] += 1.0;
            let state = y[idx * 3 + 2] as usize;
            if state > 0 {
                let trans = hindx[[cs, state - 1]];
                n_transition[[i, trans]] += wt[idx];
                n_transition[[i, trans + nhaz]] += 1.0;
                n_event[[i, state - 1]] += wt[idx];
            } else if position[idx] > 1 {
                n_censor[[i, cs]] += wt[idx];
                n_censor[[i, cs + nstate]] += 1.0;
            }
            person2 -= 1;
        }
        n_risk.row_mut(i).assign(&ntemp);
    }
    let _person1 = 0;
    let _person2 = 0;
    let mut u = if sefit > 0 {
        Some(Array2::from_shape_vec((ngrp, nstate), i0.to_vec())?)
    } else {
        None
    };
    for i in 0..ntime {
        for jk in 0..nhaz {
            if n_transition[[i, jk]] > 0.0 {
                let j = trmat[[jk, 0]];
                let k = trmat[[jk, 1]];
                let haz = n_transition[[i, jk]] / n_risk[[i, j]];
                chaz[jk] += haz;
                let pj = phat[j];
                phat[j] -= pj * haz;
                phat[k] += pj * haz;
            }
        }
        pstate.row_mut(i).assign(&phat);
        cumhaz.row_mut(i).assign(&chaz);
    }
    let (std_err, std_auc, std_chaz, influence) = if sefit > 0 {
        let Some(u) = u.as_mut() else {
            return Err("Internal error: u should be Some when sefit > 0".into());
        };
        let mut std_err_arr: Array2<f64> = Array2::zeros((ntime, nstate));
        let mut std_auc_arr: Array2<f64> = Array2::zeros((ntime, nstate));
        let mut std_chaz_arr: Array2<f64> = Array2::zeros((ntime, nhaz));
        let mut influence_arr: Option<Array2<f64>> = if sefit > 1 {
            Some(Array2::zeros((ngrp * nstate, ntime)))
        } else {
            None
        };
        let mut ua: Array2<f64> = Array2::zeros((ngrp, nstate));
        let mut c: Array2<f64> = Array2::zeros((ngrp, nhaz));
        let mut wg: Array2<f64> = Array2::zeros((ngrp, nstate));
        let _h: Array2<f64> = Array2::zeros((nstate, nstate));
        let mut ucopy: Array2<f64> = Array2::zeros((ngrp, nstate));
        let mut se1 = Array1::zeros(nstate);
        let mut se2 = Array1::zeros(nhaz);
        let mut se3 = Array1::zeros(nstate);
        for j in 0..nstate {
            se1[j] = u.column(j).mapv(|x| x.powi(2)).sum().sqrt();
        }
        let mut person1_wg = 0;
        let mut person2_wg = 0;
        for i in 0..ntime {
            let delta = if i > 0 {
                utime[i] - utime[i - 1]
            } else {
                utime[i] - t0
            };
            if sefit > 0 {
                for j in 0..nstate {
                    let mut ua_col = ua.column_mut(j);
                    ua_col += &(u.column(j).mapv(|x| x * delta));
                    se3[j] = ua_col.mapv(|x| x.powi(2)).sum().sqrt();
                }
            }
            while person1_wg < nused {
                let idx = sort1[person1_wg];
                if y[idx * 3] >= utime[i] {
                    break;
                }
                let cs = cstate[idx];
                wg[[grp[idx], cs]] += wt[idx];
                person1_wg += 1;
            }
            while person2_wg < nused {
                let idx = sort2[person2_wg];
                if y[idx * 3 + 1] >= utime[i] {
                    break;
                }
                let cs = cstate[idx];
                wg[[grp[idx], cs]] -= wt[idx];
                person2_wg += 1;
            }
            let mut h: Array2<f64> = Array2::zeros((nstate, nstate));
            let mut tdeath = 0;
            #[allow(clippy::needless_range_loop)]
            for p in person2_wg..nused {
                let idx = sort2[p];
                if y[idx * 3 + 1] != utime[i] {
                    break;
                }
                if y[idx * 3 + 2] > 0.0 {
                    tdeath += 1;
                    let j = cstate[idx];
                    let k = y[idx * 3 + 2] as usize - 1;
                    let jk = hindx[[j, k]];
                    let g = grp[idx];
                    c[[g, jk]] += wt[idx] / n_risk[[i, j]];
                    if j != k {
                        h[[j, j]] -= wt[idx] / n_risk[[i, j]];
                        h[[j, k]] += wt[idx] / n_risk[[i, j]];
                    }
                }
            }
            if tdeath == 0 {
                continue;
            }
            ucopy.assign(u);
            for j in 0..nstate {
                if h[[j, j]] != 0.0 {
                    for k in 0..nstate {
                        if k != j && h[[j, k]] != 0.0 {
                            for g in 0..ngrp {
                                u[[g, k]] += ucopy[[g, j]] * h[[j, k]];
                            }
                        }
                    }
                    for g in 0..ngrp {
                        u[[g, j]] += ucopy[[g, j]] * h[[j, j]];
                    }
                }
            }
            #[allow(clippy::needless_range_loop)]
            for p in person2_wg..nused {
                let idx = sort2[p];
                if y[idx * 3 + 1] != utime[i] {
                    break;
                }
                if y[idx * 3 + 2] > 0.0 {
                    let j = cstate[idx];
                    let k = y[idx * 3 + 2] as usize - 1;
                    let g = grp[idx];
                    let term = wt[idx] * phat[j] / n_risk[[i, j]];
                    u[[g, j]] -= term;
                    u[[g, k]] += term;
                }
            }
            for jk in 0..nhaz {
                if n_transition[[i, jk]] > 0.0 {
                    let j = trmat[[jk, 0]];
                    let k = trmat[[jk, 1]];
                    let haz = n_transition[[i, jk]] / n_risk[[i, j]];
                    let htemp = haz / n_risk[[i, j]];
                    for g in 0..ngrp {
                        if wg[[g, j]] > 0.0 {
                            c[[g, jk]] -= wg[[g, j]] * htemp;
                        }
                    }
                    if j != k {
                        for g in 0..ngrp {
                            if wg[[g, j]] > 0.0 {
                                let term = wg[[g, j]] * phat[j] * htemp;
                                u[[g, j]] += term;
                                u[[g, k]] -= term;
                            }
                        }
                    }
                }
            }
            for j in 0..nstate {
                se1[j] = u.column(j).mapv(|x| x.powi(2)).sum().sqrt();
            }
            for jk in 0..nhaz {
                se2[jk] = c.column(jk).mapv(|x: f64| x.powi(2)).sum().sqrt();
            }
            for j in 0..nstate {
                std_err_arr[[i, j]] = se1[j];
                std_auc_arr[[i, j]] = se3[j];
            }
            for jk in 0..nhaz {
                std_chaz_arr[[i, jk]] = se2[jk];
            }
            if let Some(ref mut influence_data) = influence_arr {
                let mut influence_slice = influence_data.slice_mut(s![.., i]);
                for j in 0..nstate {
                    for g in 0..ngrp {
                        influence_slice[[g + j * ngrp]] = u[[g, j]];
                    }
                }
            }
        }
        (
            Some(std_err_arr),
            Some(std_auc_arr),
            Some(std_chaz_arr),
            influence_arr,
        )
    } else {
        (None, None, None, None)
    };
    Ok(SurvFitAJComputed {
        n_risk,
        n_event,
        n_censor,
        pstate,
        cumhaz,
        std_err,
        std_chaz,
        std_auc,
        influence,
        n_enter,
        n_transition,
    })
}
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn survfitaj(
    y: Vec<f64>,
    sort1: Vec<usize>,
    sort2: Vec<usize>,
    utime: Vec<f64>,
    cstate: Vec<usize>,
    wt: Vec<f64>,
    grp: Vec<usize>,
    ngrp: usize,
    p0: Vec<f64>,
    i0: Vec<f64>,
    sefit: i32,
    entry: bool,
    position: Vec<usize>,
    hindx: Vec<Vec<usize>>,
    trmat: Vec<Vec<usize>>,
    t0: f64,
) -> PyResult<SurvFitAJ> {
    let hindx_array = Array2::from_shape_vec(
        (hindx.len(), hindx[0].len()),
        hindx.into_iter().flatten().collect(),
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid hindx array: {}", e)))?;
    let trmat_array = Array2::from_shape_vec(
        (trmat.len(), trmat[0].len()),
        trmat.into_iter().flatten().collect(),
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid trmat array: {}", e)))?;
    let data = SurvFitAJData {
        y: &y,
        sort1: &sort1,
        sort2: &sort2,
        utime: &utime,
        cstate: &cstate,
        wt: &wt,
        grp: &grp,
        position: &position,
    };
    let fit_params = SurvFitAJParams {
        ngrp,
        p0: &p0,
        i0: &i0,
        sefit,
        entry,
        hindx: &hindx_array,
        trmat: &trmat_array,
        t0,
    };
    let result = compute_survfitaj(&data, &fit_params).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("survfitaj failed: {}", e))
    })?;
    Ok(result.into_python_result())
}
