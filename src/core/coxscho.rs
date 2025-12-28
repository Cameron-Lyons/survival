use pyo3::prelude::*;
pub(crate) struct CoxSchoInput<'a> {
    pub y: &'a [f64],
    pub score: &'a [f64],
    pub strata: &'a [i32],
}
pub(crate) struct CoxSchoParams {
    pub nused: usize,
    pub nvar: usize,
    pub method: i32,
}
pub(crate) fn coxscho(
    params: CoxSchoParams,
    input: CoxSchoInput,
    covar: &mut [f64],
    work: &mut [f64],
) {
    assert!(input.y.len() >= 3 * params.nused, "y array too short");
    assert!(
        covar.len() >= params.nvar * params.nused,
        "covar array too short for nvar and nused"
    );
    assert!(input.score.len() >= params.nused, "score array too short");
    assert!(input.strata.len() >= params.nused, "strata array too short");
    assert!(
        work.len() >= 3 * params.nvar,
        "work array must be at least 3 * nvar in length"
    );
    let start = &input.y[0..params.nused];
    let stop = &input.y[params.nused..2 * params.nused];
    let event = &input.y[2 * params.nused..3 * params.nused];
    let mut covar_cols = Vec::with_capacity(params.nvar);
    let mut remaining = covar;
    for _ in 0..params.nvar {
        let (col, rest) = remaining.split_at_mut(params.nused);
        covar_cols.push(col);
        remaining = rest;
    }
    let (a, rest) = work.split_at_mut(params.nvar);
    let (a2, mean) = rest.split_at_mut(params.nvar);
    let mut person = 0;
    while person < params.nused {
        if event[person] != 1.0 {
            person += 1;
            continue;
        }
        let time = stop[person];
        let mut deaths = 0.0;
        let mut denom = 0.0;
        let mut efron_wt = 0.0;
        #[allow(clippy::needless_range_loop)]
        for i in 0..params.nvar {
            a[i] = 0.0;
            a2[i] = 0.0;
        }
        let mut k = person;
        while k < params.nused {
            if start[k] < time {
                let weight = input.score[k];
                denom += weight;
                #[allow(clippy::needless_range_loop)]
                for i in 0..params.nvar {
                    a[i] += weight * covar_cols[i][k];
                }
                if stop[k] == time && event[k] == 1.0 {
                    deaths += 1.0;
                    efron_wt += weight;
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..params.nvar {
                        a2[i] += weight * covar_cols[i][k];
                    }
                }
            }
            if input.strata[k] == 1 {
                break;
            }
            k += 1;
        }
        #[allow(clippy::needless_range_loop)]
        for i in 0..params.nvar {
            mean[i] = 0.0;
        }
        if deaths > 0.0 {
            for k_death in 0..(deaths as usize) {
                let temp = if params.method == 1 {
                    (k_death as f64) / deaths
                } else {
                    0.0
                };
                #[allow(clippy::needless_range_loop)]
                for i in 0..params.nvar {
                    let denominator = deaths * (denom - temp * efron_wt);
                    if denominator != 0.0 {
                        mean[i] += (a[i] - temp * a2[i]) / denominator;
                    }
                }
            }
        }
        let mut k = person;
        while k < params.nused && stop[k] == time {
            if event[k] == 1.0 {
                #[allow(clippy::needless_range_loop)]
                for i in 0..params.nvar {
                    covar_cols[i][k] -= mean[i];
                }
            }
            person += 1;
            if input.strata[k] == 1 {
                break;
            }
            k += 1;
        }
    }
}
#[pyfunction]
#[pyo3(signature = (y, score, strata, covar, nvar, method=0))]
pub fn schoenfeld_residuals(
    y: Vec<f64>,
    score: Vec<f64>,
    strata: Vec<i32>,
    covar: Vec<f64>,
    nvar: usize,
    method: i32,
) -> PyResult<Vec<f64>> {
    let nused = score.len();
    if y.len() < 3 * nused {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "y array must have length >= 3 * n (start, stop, event)",
        ));
    }
    if strata.len() < nused {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "strata array length must match score length",
        ));
    }
    if covar.len() < nvar * nused {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "covar array must have length >= nvar * n",
        ));
    }
    let mut covar_copy = covar.clone();
    let mut work = vec![0.0; 3 * nvar];
    let params = CoxSchoParams {
        nused,
        nvar,
        method,
    };
    let input = CoxSchoInput {
        y: &y,
        score: &score,
        strata: &strata,
    };
    coxscho(params, input, &mut covar_copy, &mut work);
    Ok(covar_copy)
}
