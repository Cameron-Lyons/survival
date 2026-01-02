use super::common::{
    apply_deltas_add, apply_deltas_set, build_score_result, validate_scoring_inputs,
};
use ndarray::{Array2, ArrayView2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[inline]
pub fn agscore3(
    y: &[f64],
    covar: &[f64],
    strata: &[i32],
    score: &[f64],
    weights: &[f64],
    method: i32,
    sort1: &[i32],
) -> Result<Vec<f64>, String> {
    let n = y.len() / 3;
    let nvar = covar.len() / n;
    let tstart = &y[0..n];
    let tstop = &y[n..2 * n];
    let event = &y[2 * n..3 * n];

    let covar_matrix = ArrayView2::from_shape((nvar, n), covar).map_err(|e| {
        format!(
            "Failed to create covariate view with shape ({}, {}): {}",
            nvar, n, e
        )
    })?;

    let mut resid_matrix = Array2::zeros((nvar, n));
    let mut a = vec![0.0; nvar];
    let mut a2 = vec![0.0; nvar];
    let mut mean = vec![0.0; nvar];
    let mut mh1 = vec![0.0; nvar];
    let mut mh2 = vec![0.0; nvar];
    let mut mh3 = vec![0.0; nvar];
    let mut xhaz = vec![0.0; nvar];

    let mut cumhaz = 0.0;
    let mut denom = 0.0;
    let mut current_stratum = *strata.last().unwrap_or(&0);
    let mut i1 = n - 1;
    let sort1: Vec<usize> = sort1.iter().map(|&x| (x - 1) as usize).collect();

    let mut person = n - 1;
    while person > 0 {
        let dtime = tstop[person];

        if strata[person] != current_stratum {
            let mut exit_indices = Vec::new();
            while i1 > 0 && sort1[i1] > person {
                exit_indices.push(sort1[i1]);
                i1 -= 1;
            }

            apply_deltas_add(&exit_indices, nvar, &mut resid_matrix, |k| {
                (0..nvar)
                    .map(|j| -score[k] * (cumhaz * covar_matrix[[j, k]] - xhaz[j]))
                    .collect()
            });

            cumhaz = 0.0;
            denom = 0.0;
            a.fill(0.0);
            xhaz.fill(0.0);
            current_stratum = strata[person];
        } else {
            let mut exit_indices = Vec::new();
            while i1 > 0 && tstart[sort1[i1]] >= dtime {
                let k = sort1[i1];
                if strata[k] != current_stratum {
                    break;
                }
                exit_indices.push(k);
                let risk = score[k] * weights[k];
                denom -= risk;
                for j in 0..nvar {
                    a[j] -= risk * covar_matrix[[j, k]];
                }
                i1 -= 1;
            }

            apply_deltas_add(&exit_indices, nvar, &mut resid_matrix, |k| {
                (0..nvar)
                    .map(|j| -score[k] * (cumhaz * covar_matrix[[j, k]] - xhaz[j]))
                    .collect()
            });
        }

        let mut e_denom = 0.0;
        let mut deaths = 0.0;
        let mut meanwt = 0.0;
        a2.fill(0.0);

        let mut processed_indices = Vec::new();
        while person > 0 && tstop[person] == dtime {
            if strata[person] != current_stratum {
                break;
            }
            processed_indices.push(person);

            let risk = score[person] * weights[person];
            denom += risk;
            for j in 0..nvar {
                a[j] += risk * covar_matrix[[j, person]];
            }

            if event[person] > 0.5 {
                deaths += 1.0;
                e_denom += risk;
                meanwt += weights[person];
                for j in 0..nvar {
                    a2[j] += risk * covar_matrix[[j, person]];
                }
            }
            person -= 1;
        }

        apply_deltas_set(&processed_indices, nvar, &mut resid_matrix, |p| {
            (0..nvar)
                .map(|j| (covar_matrix[[j, p]] * cumhaz - xhaz[j]) * score[p])
                .collect()
        });

        if deaths > 0.0 {
            if deaths < 2.0 || method == 0 {
                let hazard = meanwt / denom;
                cumhaz += hazard;
                for j in 0..nvar {
                    mean[j] = a[j] / denom;
                    xhaz[j] += mean[j] * hazard;
                }

                apply_deltas_add(&processed_indices, nvar, &mut resid_matrix, |p| {
                    (0..nvar).map(|j| covar_matrix[[j, p]] - mean[j]).collect()
                });
            } else {
                mh1.fill(0.0);
                mh2.fill(0.0);
                mh3.fill(0.0);
                let meanwt_norm = meanwt / deaths;

                for dd in 0..deaths as i32 {
                    let downwt = dd as f64 / deaths;
                    let d2 = denom - downwt * e_denom;
                    let hazard = meanwt_norm / d2;
                    cumhaz += hazard;
                    for j in 0..nvar {
                        mean[j] = (a[j] - downwt * a2[j]) / d2;
                        xhaz[j] += mean[j] * hazard;
                        mh1[j] += hazard * downwt;
                        mh2[j] += mean[j] * hazard * downwt;
                        mh3[j] += mean[j] / deaths;
                    }
                }

                apply_deltas_add(&processed_indices, nvar, &mut resid_matrix, |p| {
                    (0..nvar)
                        .map(|j| {
                            (covar_matrix[[j, p]] - mh3[j])
                                + score[p] * (covar_matrix[[j, p]] * mh1[j] - mh2[j])
                        })
                        .collect()
                });
            }
        }
    }

    let mut final_indices = Vec::new();
    while i1 > 0 {
        final_indices.push(sort1[i1]);
        i1 -= 1;
    }

    apply_deltas_add(&final_indices, nvar, &mut resid_matrix, |k| {
        (0..nvar)
            .map(|j| -score[k] * (cumhaz * covar_matrix[[j, k]] - xhaz[j]))
            .collect()
    });

    Ok(resid_matrix.into_raw_vec_and_offset().0)
}

#[pyfunction]
pub fn perform_agscore3_calculation(
    time_data: Vec<f64>,
    covariates: Vec<f64>,
    strata: Vec<i32>,
    score: Vec<f64>,
    weights: Vec<f64>,
    method: i32,
    sort1: Vec<i32>,
) -> PyResult<Py<PyAny>> {
    let n = weights.len();
    validate_scoring_inputs(
        n,
        time_data.len(),
        covariates.len(),
        strata.len(),
        score.len(),
        weights.len(),
    )?;
    if sort1.len() != n {
        return Err(PyRuntimeError::new_err(
            "Sort1 length does not match observations",
        ));
    }
    let residuals = agscore3(
        &time_data,
        &covariates,
        &strata,
        &score,
        &weights,
        method,
        &sort1,
    )
    .map_err(PyRuntimeError::new_err)?;
    let nvar = covariates.len() / n;
    Python::attach(|py| build_score_result(py, residuals, n, nvar, method).map(|d| d.into()))
}
