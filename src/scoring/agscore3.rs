use ndarray::{Array2, ArrayView2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub fn agscore3(
    y: &[f64],
    covar: &[f64],
    strata: &[i32],
    score: &[f64],
    weights: &[f64],
    method: i32,
    sort1: &[i32],
) -> Vec<f64> {
    let n = y.len() / 3;
    let nvar = covar.len() / n;

    let tstart = &y[0..n];
    let tstop = &y[n..2 * n];
    let event = &y[2 * n..3 * n];

    let covar_matrix = ArrayView2::from_shape((nvar, n), covar).unwrap();
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
            while i1 > 0 && sort1[i1] > person {
                let k = sort1[i1];
                for j in 0..nvar {
                    resid_matrix[[j, k]] -= score[k] * (cumhaz * covar_matrix[[j, k]] - xhaz[j]);
                }
                i1 -= 1;
            }

            cumhaz = 0.0;
            denom = 0.0;
            a.iter_mut().for_each(|x| *x = 0.0);
            xhaz.iter_mut().for_each(|x| *x = 0.0);
            current_stratum = strata[person];
        } else {
            while i1 > 0 && tstart[sort1[i1]] >= dtime {
                let k = sort1[i1];
                if strata[k] != current_stratum {
                    break;
                }

                let risk = score[k] * weights[k];
                denom -= risk;

                for j in 0..nvar {
                    resid_matrix[[j, k]] -= score[k] * (cumhaz * covar_matrix[[j, k]] - xhaz[j]);
                    a[j] -= risk * covar_matrix[[j, k]];
                }
                i1 -= 1;
            }
        }

        let mut e_denom = 0.0;
        let mut deaths = 0.0;
        let mut meanwt = 0.0;
        a2.iter_mut().for_each(|x| *x = 0.0);

        let mut processed = 0;
        while person > 0 && tstop[person] == dtime {
            if strata[person] != current_stratum {
                break;
            }

            for j in 0..nvar {
                resid_matrix[[j, person]] =
                    (covar_matrix[[j, person]] * cumhaz - xhaz[j]) * score[person];
            }

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
            processed += 1;
        }

        if deaths > 0.0 {
            if deaths < 2.0 || method == 0 {
                let hazard = meanwt / denom;
                cumhaz += hazard;

                for j in 0..nvar {
                    mean[j] = a[j] / denom;
                    xhaz[j] += mean[j] * hazard;

                    for k in person + 1..=person + processed {
                        resid_matrix[[j, k]] += covar_matrix[[j, k]] - mean[j];
                    }
                }
            } else {
                mh1.iter_mut().for_each(|x| *x = 0.0);
                mh2.iter_mut().for_each(|x| *x = 0.0);
                mh3.iter_mut().for_each(|x| *x = 0.0);
                meanwt /= deaths;

                for dd in 0..deaths as i32 {
                    let downwt = dd as f64 / deaths;
                    let d2 = denom - downwt * e_denom;
                    let hazard = meanwt / d2;
                    cumhaz += hazard;

                    for j in 0..nvar {
                        mean[j] = (a[j] - downwt * a2[j]) / d2;
                        xhaz[j] += mean[j] * hazard;
                        mh1[j] += hazard * downwt;
                        mh2[j] += mean[j] * hazard * downwt;
                        mh3[j] += mean[j] / deaths;
                    }
                }

                for k in person + 1..=person + processed {
                    for j in 0..nvar {
                        resid_matrix[[j, k]] += (covar_matrix[[j, k]] - mh3[j])
                            + score[k] * (covar_matrix[[j, k]] * mh1[j] - mh2[j]);
                    }
                }
            }
        }
    }

    while i1 > 0 {
        let k = sort1[i1];
        for j in 0..nvar {
            resid_matrix[[j, k]] -= score[k] * (cumhaz * covar_matrix[[j, k]] - xhaz[j]);
        }
        i1 -= 1;
    }

    resid_matrix.into_raw_vec_and_offset().0
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
    if n == 0 {
        return Err(PyRuntimeError::new_err("No observations provided"));
    }

    if time_data.len() != 3 * n {
        return Err(PyRuntimeError::new_err(
            "Time data should have 3*n elements (start, stop, event)",
        ));
    }

    if covariates.len() % n != 0 {
        return Err(PyRuntimeError::new_err(
            "Covariates length should be divisible by number of observations",
        ));
    }

    if strata.len() != n {
        return Err(PyRuntimeError::new_err(
            "Strata length does not match observations",
        ));
    }

    if score.len() != n {
        return Err(PyRuntimeError::new_err(
            "Score length does not match observations",
        ));
    }

    if weights.len() != n {
        return Err(PyRuntimeError::new_err(
            "Weights length does not match observations",
        ));
    }

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
    );

    let nvar = covariates.len() / n;
    let mut summary_stats = Vec::new();

    for i in 0..nvar {
        let start_idx = i * n;
        let end_idx = (i + 1) * n;
        let var_residuals = &residuals[start_idx..end_idx];

        let mean = var_residuals.iter().sum::<f64>() / n as f64;
        let variance = var_residuals
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / (n - 1) as f64;

        summary_stats.push(mean);
        summary_stats.push(variance);
    }

    Python::attach(|py| {
        let dict = PyDict::new(py);
        dict.set_item("residuals", residuals).unwrap();
        dict.set_item("n_observations", n).unwrap();
        dict.set_item("n_variables", nvar).unwrap();
        dict.set_item("method", if method == 0 { "breslow" } else { "efron" })
            .unwrap();
        dict.set_item("summary_stats", summary_stats).unwrap();
        Ok(dict.into())
    })
}
