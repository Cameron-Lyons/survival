use pyo3::prelude::*;

pub(crate) struct CoxScoreData<'a> {
    pub y: &'a [f64],
    pub strata: &'a [i32],
    pub covar: &'a [f64],
    pub score: &'a [f64],
    pub weights: &'a [f64],
}

pub(crate) struct CoxScoreParams {
    pub method: i32,
    pub n: usize,
    pub nvar: usize,
}

pub(crate) fn cox_score_residuals_internal(data: CoxScoreData, params: CoxScoreParams) -> Vec<f64> {
    let time = &data.y[0..params.n];
    let status = &data.y[params.n..2 * params.n];
    let mut resid = vec![0.0; params.n * params.nvar];
    let mut a = vec![0.0; params.nvar];
    let mut a2 = vec![0.0; params.nvar];
    let mut xhaz = vec![0.0; params.nvar];
    let mut denom = 0.0;
    let mut cumhaz = 0.0;
    let mut stratastart = params.n as i32 - 1;
    let mut currentstrata = if params.n > 0 {
        data.strata[params.n - 1]
    } else {
        0
    };
    let mut i = stratastart;

    while i >= 0 {
        let i_usize = i as usize;
        let newtime = time[i_usize];
        let mut deaths_count = 0;
        let mut e_denom = 0.0;
        let mut meanwt = 0.0;
        a2.fill(0.0);

        let mut j = i;

        while j >= 0 {
            let j_usize = j as usize;
            if time[j_usize] != newtime || data.strata[j_usize] != currentstrata {
                break;
            }

            let risk = data.score[j_usize] * data.weights[j_usize];
            denom += risk;

            for (var, (resid_elem, xhaz_elem)) in resid
                .iter_mut()
                .skip(j_usize * params.nvar)
                .take(params.nvar)
                .zip(xhaz.iter())
                .enumerate()
            {
                let idx = j_usize * params.nvar + var;
                let covar_val = data.covar[idx];
                *resid_elem = data.score[j_usize] * (covar_val * cumhaz - xhaz_elem);
            }

            for (var, a_elem) in a.iter_mut().enumerate().take(params.nvar) {
                *a_elem += risk * data.covar[j_usize * params.nvar + var];
            }

            if status[j_usize] == 1.0 {
                deaths_count += 1;
                e_denom += risk;
                meanwt += data.weights[j_usize];
                for (var, a2_elem) in a2.iter_mut().enumerate().take(params.nvar) {
                    *a2_elem += risk * data.covar[j_usize * params.nvar + var];
                }
            }

            j -= 1;
        }

        let processed_start = j + 1;
        let processed_end = i;

        i = j;

        if deaths_count > 0 {
            let deaths = deaths_count as f64;
            if deaths < 2.0 || params.method == 0 {
                let hazard = meanwt / denom;
                cumhaz += hazard;

                for (var, (a_elem, xhaz_elem)) in
                    a.iter().zip(xhaz.iter_mut()).enumerate().take(params.nvar)
                {
                    let xbar = a_elem / denom;
                    *xhaz_elem += xbar * hazard;

                    for k in processed_start..=processed_end {
                        let k_usize = k as usize;
                        let idx = k_usize * params.nvar + var;
                        resid[idx] += data.covar[idx] - xbar;
                    }
                }
            } else {
                let meanwt_per_death = meanwt / deaths;
                for dd in 0..deaths_count {
                    let downwt = dd as f64 / deaths;
                    let temp = denom - downwt * e_denom;
                    let hazard = meanwt_per_death / temp;
                    cumhaz += hazard;

                    for var in 0..params.nvar {
                        let xbar = (a[var] - downwt * a2[var]) / temp;
                        xhaz[var] += xbar * hazard;

                        for k in processed_start..=processed_end {
                            let k_usize = k as usize;
                            let idx = k_usize * params.nvar + var;
                            let temp2 = data.covar[idx] - xbar;
                            resid[idx] += temp2 / deaths;
                            resid[idx] += temp2 * data.score[k_usize] * hazard * downwt;
                        }
                    }
                }
            }
        }

        if i < 0 || data.strata[i as usize] != currentstrata {
            for k in (i + 1)..=stratastart {
                let k_usize = k as usize;
                for (var, (resid_elem, xhaz_elem)) in resid
                    .iter_mut()
                    .skip(k_usize * params.nvar)
                    .take(params.nvar)
                    .zip(xhaz.iter())
                    .enumerate()
                {
                    let idx = k_usize * params.nvar + var;
                    *resid_elem += data.score[k_usize] * (xhaz_elem - data.covar[idx] * cumhaz);
                }
            }

            denom = 0.0;
            cumhaz = 0.0;
            a.fill(0.0);
            xhaz.fill(0.0);

            stratastart = i;
            if i >= 0 {
                currentstrata = data.strata[i as usize];
            }
        }
    }

    resid
}

/// Calculate Cox score (dfbeta) residuals.
///
/// Score residuals measure the influence of each observation on the coefficient estimates.
/// They are useful for identifying influential observations and assessing model fit.
///
/// # Arguments
/// * `y` - Survival data as [time..., status...] (length 2*n)
/// * `strata` - Stratum indicators for each observation
/// * `covar` - Covariate matrix in row-major order (n * nvar)
/// * `score` - Risk scores (exp(linear predictor))
/// * `weights` - Observation weights
/// * `nvar` - Number of covariates
/// * `method` - Efron (1) or Breslow (0) method for ties
///
/// # Returns
/// Score residuals matrix in row-major order (n * nvar)
#[pyfunction]
#[pyo3(signature = (y, strata, covar, score, weights, nvar, method=0))]
pub fn cox_score_residuals(
    y: Vec<f64>,
    strata: Vec<i32>,
    covar: Vec<f64>,
    score: Vec<f64>,
    weights: Vec<f64>,
    nvar: usize,
    method: i32,
) -> PyResult<Vec<f64>> {
    let n = score.len();

    if y.len() < 2 * n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "y array must have length >= 2 * n (time, status)",
        ));
    }
    if strata.len() < n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "strata array length must match n",
        ));
    }
    if covar.len() < n * nvar {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "covar array must have length >= n * nvar",
        ));
    }
    if weights.len() < n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "weights array length must match n",
        ));
    }

    let data = CoxScoreData {
        y: &y,
        strata: &strata,
        covar: &covar,
        score: &score,
        weights: &weights,
    };

    let params = CoxScoreParams { method, n, nvar };

    Ok(cox_score_residuals_internal(data, params))
}
