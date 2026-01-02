use super::common::{apply_deltas_add, build_score_result, validate_scoring_inputs};
use ndarray::{Array2, ArrayView2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[inline]
pub fn agscore2(
    y: &[f64],
    covar: &[f64],
    strata: &[i32],
    score: &[f64],
    weights: &[f64],
    method: i32,
) -> Result<Vec<f64>, String> {
    let n = y.len() / 3;
    let nvar = covar.len() / n;
    let tstart = &y[0..n];
    let tstop = &y[n..2 * n];
    let event = &y[2 * n..3 * n];

    let covar_matrix: ArrayView2<f64> = ArrayView2::from_shape((nvar, n), covar).map_err(|e| {
        format!(
            "Failed to create covariate matrix view with shape ({}, {}): {}",
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

    let mut person = 0;
    while person < n {
        if event[person] == 0.0 {
            person += 1;
            continue;
        }

        let time = tstop[person];
        let mut denom = 0.0;
        let mut e_denom = 0.0;
        let mut deaths = 0.0;
        let mut meanwt = 0.0;
        a.fill(0.0);
        a2.fill(0.0);

        let mut at_risk_indices: Vec<usize> = Vec::new();
        let mut k = person;
        while k < n && strata[k] == strata[person] {
            if tstart[k] < time {
                at_risk_indices.push(k);
                let risk = score[k] * weights[k];
                denom += risk;
                for i in 0..nvar {
                    a[i] += risk * covar_matrix[[i, k]];
                }
                if tstop[k] == time && event[k] == 1.0 {
                    deaths += 1.0;
                    e_denom += risk;
                    meanwt += weights[k];
                    for i in 0..nvar {
                        a2[i] += risk * covar_matrix[[i, k]];
                    }
                }
            }
            k += 1;
        }

        if deaths < 2.0 || method == 0 {
            let hazard = meanwt / denom;
            for i in 0..nvar {
                mean[i] = a[i] / denom;
            }

            apply_deltas_add(&at_risk_indices, nvar, &mut resid_matrix, |k| {
                let risk = score[k];
                let is_event = tstop[k] == time && event[k] == 1.0;
                (0..nvar)
                    .map(|i| {
                        let diff = covar_matrix[[i, k]] - mean[i];
                        let mut delta = -diff * risk * hazard;
                        if is_event {
                            delta += diff;
                        }
                        delta
                    })
                    .collect()
            });
        } else {
            let meanwt_norm = meanwt / deaths;
            let mut temp1 = 0.0;
            let mut temp2 = 0.0;
            mh1.fill(0.0);
            mh2.fill(0.0);
            mh3.fill(0.0);

            for dd in 0..deaths as usize {
                let downwt = dd as f64 / deaths;
                let d2 = denom - downwt * e_denom;
                let hazard = meanwt_norm / d2;
                temp1 += hazard;
                temp2 += (1.0 - downwt) * hazard;
                for i in 0..nvar {
                    mean[i] = (a[i] - downwt * a2[i]) / d2;
                    mh1[i] += mean[i] * hazard;
                    mh2[i] += mean[i] * (1.0 - downwt) * hazard;
                    mh3[i] += mean[i] / deaths;
                }
            }

            apply_deltas_add(&at_risk_indices, nvar, &mut resid_matrix, |k| {
                let risk = score[k];
                let is_event = tstop[k] == time && event[k] == 1.0;
                (0..nvar)
                    .map(|i| {
                        if is_event {
                            (covar_matrix[[i, k]] - mh3[i]) - risk * covar_matrix[[i, k]] * temp2
                                + risk * mh2[i]
                        } else {
                            -risk * (covar_matrix[[i, k]] * temp1 - mh1[i])
                        }
                    })
                    .collect()
            });
        }

        while person < n && tstop[person] == time {
            person += 1;
        }
    }

    Ok(resid_matrix.into_raw_vec_and_offset().0)
}

#[pyfunction]
pub fn perform_score_calculation(
    time_data: Vec<f64>,
    covariates: Vec<f64>,
    strata: Vec<i32>,
    score: Vec<f64>,
    weights: Vec<f64>,
    method: i32,
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
    let residuals = agscore2(&time_data, &covariates, &strata, &score, &weights, method)
        .map_err(PyRuntimeError::new_err)?;
    let nvar = covariates.len() / n;
    Python::attach(|py| build_score_result(py, residuals, n, nvar, method).map(|d| d.into()))
}
