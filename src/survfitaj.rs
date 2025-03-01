use ndarray::{Array1, Array2, Axis, s};
use std::error::Error;

#[derive(Debug)]
pub struct SurvFitAJ {
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

pub fn survfitaj(
    y: &[f64],
    sort1: &[usize],
    sort2: &[usize],
    utime: &[f64],
    cstate: &[usize],
    wt: &[f64],
    grp: &[usize],
    ngrp: usize,
    p0: &[f64],
    i0: &[f64],
    sefit: i32,
    entry: bool,
    position: &[usize],
    hindx: &Array2<usize>,
    trmat: &Array2<usize>,
    t0: f64,
) -> Result<SurvFitAJ, Box<dyn Error>> {
    let ntime = utime.len();
    let n = y.len() / 3;
    let nused = sort1.len();
    let nstate = p0.len();
    let nhaz = trmat.nrows();

    let mut n_risk = Array2::zeros((ntime, 2 * nstate));
    let mut n_event = Array2::zeros((ntime, nstate));
    let mut n_censor = Array2::zeros((ntime, 2 * nstate));
    let mut n_transition = Array2::zeros((ntime, 2 * nhaz));
    let mut pstate = Array2::zeros((ntime, nstate));
    let mut cumhaz = Array2::zeros((ntime, nhaz));
    let mut n_enter = if entry { Some(Array2::zeros((ntime, 2 * nstate))) } else { None };

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
            
            if entry && (position[idx] & 0x1) != 0 {
                if let Some(ref mut ne) = n_enter {
                    ne[[i, cs]] += wt[idx];
                    ne[[i, cs + nstate]] += 1.0;
                }
            }
            person1 -= 1;
        }
        
        while person2 > 0 && y[sort2[person2] * 3 + 1 >= ctime {
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

    let mut person1 = 0;
    let mut person2 = 0;
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

        if sefit > 0 {
        }
    }

    Ok(SurvFitAJ {
        n_risk,
        n_event,
        n_censor,
        pstate,
        cumhaz,
        std_err: None,
        std_chaz: None,
        std_auc: None,
        influence: None,
        n_enter,
        n_transition,
    })
}

