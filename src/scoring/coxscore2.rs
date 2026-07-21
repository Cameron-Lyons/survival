use pyo3::prelude::*;
use rayon::prelude::*;

use crate::internal::validation::{
    ValidationError, validate_binary_f64, validate_finite, validate_non_negative,
};

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

fn validation_err_to_py(err: ValidationError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(err.to_string())
}

fn validate_cox_score_inputs(
    y: &[f64],
    strata: &[i32],
    covar: &[f64],
    score: &[f64],
    weights: &[f64],
    nvar: usize,
    method: i32,
) -> PyResult<usize> {
    let n = score.len();
    let expected_y_len = 2usize.checked_mul(n).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("2 * n exceeds supported array size")
    })?;
    let expected_covar_len = n.checked_mul(nvar).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("n * nvar exceeds supported array size")
    })?;
    if y.len() < expected_y_len {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "y array must have length >= 2 * n (time, status)",
        ));
    }
    if strata.len() < n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "strata array length must match n",
        ));
    }
    if covar.len() < expected_covar_len {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "covar array must have length >= n * nvar",
        ));
    }
    if weights.len() < n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "weights array length must match n",
        ));
    }
    if method != 0 && method != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "method must be 0 (Breslow) or 1 (Efron)",
        ));
    }

    validate_finite(&y[..expected_y_len], "y").map_err(validation_err_to_py)?;
    validate_binary_f64(&y[n..expected_y_len], "status").map_err(validation_err_to_py)?;
    validate_finite(&covar[..expected_covar_len], "covar").map_err(validation_err_to_py)?;
    validate_finite(score, "score").map_err(validation_err_to_py)?;
    validate_non_negative(score, "score").map_err(validation_err_to_py)?;
    validate_finite(&weights[..n], "weights").map_err(validation_err_to_py)?;
    validate_non_negative(&weights[..n], "weights").map_err(validation_err_to_py)?;

    Ok(n)
}

fn find_strata_boundaries(strata: &[i32], n: usize) -> Vec<(usize, usize)> {
    if n == 0 {
        return vec![];
    }
    let mut boundaries = Vec::new();
    let mut start = 0;
    let mut current = strata[0];
    for (i, &stratum) in strata.iter().enumerate().skip(1).take(n - 1) {
        if stratum != current {
            boundaries.push((start, i - 1));
            start = i;
            current = stratum;
        }
    }
    boundaries.push((start, n - 1));
    boundaries
}

#[inline]
fn death_rows_in_bucket(status: &[f64], start: i32, end: i32) -> impl Iterator<Item = usize> + '_ {
    (start..=end)
        .map(|row| row as usize)
        .filter(|&row| status[row] == 1.0)
}

#[allow(clippy::too_many_arguments)]
fn process_stratum(
    start: usize,
    end: usize,
    time: &[f64],
    status: &[f64],
    covar: &[f64],
    score: &[f64],
    weights: &[f64],
    nvar: usize,
    method: i32,
) -> Vec<(usize, Vec<f64>)> {
    let mut results = Vec::new();
    let mut a = vec![0.0; nvar];
    let mut a2 = vec![0.0; nvar];
    let mut xhaz = vec![0.0; nvar];
    let mut denom = 0.0;
    let mut cumhaz = 0.0;
    let mut partial_resid: Vec<Vec<f64>> = (start..=end).map(|_| vec![0.0; nvar]).collect();
    let mut i = end as i32;
    while i >= start as i32 {
        let i_usize = i as usize;
        let newtime = time[i_usize];
        let mut deaths_count = 0;
        let mut e_denom = 0.0;
        let mut meanwt = 0.0;
        a2.fill(0.0);
        let mut j = i;
        while j >= start as i32 {
            let j_usize = j as usize;
            if time[j_usize] != newtime {
                break;
            }
            let risk = score[j_usize] * weights[j_usize];
            denom += risk;
            let local_idx = j_usize - start;
            for var in 0..nvar {
                let covar_val = covar[j_usize * nvar + var];
                partial_resid[local_idx][var] = score[j_usize] * (covar_val * cumhaz - xhaz[var]);
            }
            for var in 0..nvar {
                a[var] += risk * covar[j_usize * nvar + var];
            }
            if status[j_usize] == 1.0 {
                deaths_count += 1;
                e_denom += risk;
                meanwt += weights[j_usize];
                for var in 0..nvar {
                    a2[var] += risk * covar[j_usize * nvar + var];
                }
            }
            j -= 1;
        }
        let processed_start = j + 1;
        let processed_end = i;
        i = j;
        if deaths_count > 0 {
            let deaths = deaths_count as f64;
            if deaths < 2.0 || method == 0 {
                let hazard = meanwt / denom;
                cumhaz += hazard;
                for var in 0..nvar {
                    let xbar = a[var] / denom;
                    xhaz[var] += xbar * hazard;
                    for k_usize in death_rows_in_bucket(status, processed_start, processed_end) {
                        let local_idx = k_usize - start;
                        partial_resid[local_idx][var] += covar[k_usize * nvar + var] - xbar;
                    }
                }
            } else {
                let meanwt_per_death = meanwt / deaths;
                for dd in 0..deaths_count {
                    let downwt = dd as f64 / deaths;
                    let temp = denom - downwt * e_denom;
                    let hazard = meanwt_per_death / temp;
                    cumhaz += hazard;
                    for var in 0..nvar {
                        let xbar = (a[var] - downwt * a2[var]) / temp;
                        xhaz[var] += xbar * hazard;
                        for k_usize in death_rows_in_bucket(status, processed_start, processed_end)
                        {
                            let local_idx = k_usize - start;
                            let temp2 = covar[k_usize * nvar + var] - xbar;
                            partial_resid[local_idx][var] += temp2 / deaths;
                            partial_resid[local_idx][var] +=
                                temp2 * score[k_usize] * hazard * downwt;
                        }
                    }
                }
            }
        }
    }
    for k in start..=end {
        let local_idx = k - start;
        for var in 0..nvar {
            partial_resid[local_idx][var] +=
                score[k] * (xhaz[var] - covar[k * nvar + var] * cumhaz);
        }
        results.push((k, partial_resid[local_idx].clone()));
    }
    results
}
pub(crate) fn compute_cox_score_residuals(data: CoxScoreData, params: CoxScoreParams) -> Vec<f64> {
    let time = &data.y[0..params.n];
    let status = &data.y[params.n..2 * params.n];
    let boundaries = find_strata_boundaries(data.strata, params.n);
    if boundaries.len() > 1 {
        let all_results: Vec<Vec<(usize, Vec<f64>)>> = boundaries
            .par_iter()
            .map(|&(start, end)| {
                process_stratum(
                    start,
                    end,
                    time,
                    status,
                    data.covar,
                    data.score,
                    data.weights,
                    params.nvar,
                    params.method,
                )
            })
            .collect();
        let mut resid = vec![0.0; params.n * params.nvar];
        for stratum_results in all_results {
            for (idx, values) in stratum_results {
                for (var, &val) in values.iter().enumerate() {
                    resid[idx * params.nvar + var] = val;
                }
            }
        }
        return resid;
    }
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
                    for k_usize in death_rows_in_bucket(status, processed_start, processed_end) {
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
                        for k_usize in death_rows_in_bucket(status, processed_start, processed_end)
                        {
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
    let n = validate_cox_score_inputs(&y, &strata, &covar, &score, &weights, nvar, method)?;
    let data = CoxScoreData {
        y: &y,
        strata: &strata,
        covar: &covar,
        score: &score,
        weights: &weights,
    };
    let params = CoxScoreParams { method, n, nvar };
    Ok(compute_cox_score_residuals(data, params))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repeated_mixed_tie_residuals(method: i32, strata_count: usize) -> Vec<f64> {
        let n = 4 * strata_count;
        let mut time = Vec::with_capacity(n);
        let mut status = Vec::with_capacity(n);
        let mut strata = Vec::with_capacity(n);
        let mut covar = Vec::with_capacity(n);
        for stratum in 0..strata_count {
            time.extend([1.0, 1.0, 1.0, 2.0]);
            status.extend([1.0, 1.0, 0.0, 0.0]);
            strata.extend([stratum as i32; 4]);
            covar.extend([0.0, 2.0, 4.0, 6.0]);
        }
        let mut y = time;
        y.extend(status);
        let score = vec![1.0; n];
        let weights = vec![1.0; n];

        compute_cox_score_residuals(
            CoxScoreData {
                y: &y,
                strata: &strata,
                covar: &covar,
                score: &score,
                weights: &weights,
            },
            CoxScoreParams { method, n, nvar: 1 },
        )
    }

    fn assert_values_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-12,
                "value {idx}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn single_stratum_output_length() {
        let n = 4;
        let nvar = 1;
        let y = vec![1.0, 2.0, 3.0, 4.0, 1.0, 0.0, 1.0, 0.0];
        let strata = vec![0, 0, 0, 0];
        let covar = vec![0.5, 1.0, 1.5, 2.0];
        let score = vec![1.0, 1.0, 1.0, 1.0];
        let weights = vec![1.0, 1.0, 1.0, 1.0];
        let data = CoxScoreData {
            y: &y,
            strata: &strata,
            covar: &covar,
            score: &score,
            weights: &weights,
        };
        let params = CoxScoreParams { method: 0, n, nvar };
        let result = compute_cox_score_residuals(data, params);
        assert_eq!(result.len(), n * nvar);
    }

    #[test]
    fn multiple_strata_output_length() {
        let n = 4;
        let nvar = 1;
        let y = vec![1.0, 2.0, 3.0, 4.0, 1.0, 0.0, 1.0, 0.0];
        let strata = vec![0, 0, 1, 1];
        let covar = vec![0.5, 1.0, 1.5, 2.0];
        let score = vec![1.0, 1.0, 1.0, 1.0];
        let weights = vec![1.0, 1.0, 1.0, 1.0];
        let data = CoxScoreData {
            y: &y,
            strata: &strata,
            covar: &covar,
            score: &score,
            weights: &weights,
        };
        let params = CoxScoreParams { method: 0, n, nvar };
        let result = compute_cox_score_residuals(data, params);
        assert_eq!(result.len(), n * nvar);
    }

    #[test]
    fn breslow_vs_efron() {
        let n = 4;
        let nvar = 1;
        let y = vec![1.0, 1.0, 2.0, 3.0, 1.0, 1.0, 0.0, 0.0];
        let strata = vec![0, 0, 0, 0];
        let covar = vec![1.0, 2.0, 3.0, 4.0];
        let score = vec![1.0, 1.0, 1.0, 1.0];
        let weights = vec![1.0, 1.0, 1.0, 1.0];
        let data_b = CoxScoreData {
            y: &y,
            strata: &strata,
            covar: &covar,
            score: &score,
            weights: &weights,
        };
        let params_b = CoxScoreParams { method: 0, n, nvar };
        let breslow = compute_cox_score_residuals(data_b, params_b);
        let data_e = CoxScoreData {
            y: &y,
            strata: &strata,
            covar: &covar,
            score: &score,
            weights: &weights,
        };
        let params_e = CoxScoreParams { method: 1, n, nvar };
        let efron = compute_cox_score_residuals(data_e, params_e);
        let differs = breslow
            .iter()
            .zip(efron.iter())
            .any(|(a, b)| (a - b).abs() > 1e-15);
        assert!(differs);
    }

    #[test]
    fn breslow_excludes_tied_censor_from_event_contribution() {
        let residuals = repeated_mixed_tie_residuals(0, 1);

        assert_values_close(&residuals, &[-1.5, -0.5, -0.5, -1.5]);
        assert!((residuals.iter().sum::<f64>() + 4.0).abs() < 1e-12);
    }

    #[test]
    fn efron_excludes_tied_censor_from_event_and_death_adjustments() {
        let residuals = repeated_mixed_tie_residuals(1, 1);
        let expected = [-71.0 / 36.0, -29.0 / 36.0, -13.0 / 36.0, -55.0 / 36.0];

        assert_values_close(&residuals, &expected);
        assert!((residuals.iter().sum::<f64>() + 14.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn stratified_mixed_ties_preserve_each_stratum_score_sum() {
        for (method, expected, expected_score) in [
            (0, vec![-1.5, -0.5, -0.5, -1.5], -4.0),
            (
                1,
                vec![-71.0 / 36.0, -29.0 / 36.0, -13.0 / 36.0, -55.0 / 36.0],
                -14.0 / 3.0,
            ),
        ] {
            let residuals = repeated_mixed_tie_residuals(method, 2);
            for stratum_residuals in residuals.chunks_exact(4) {
                assert_values_close(stratum_residuals, &expected);
                assert!((stratum_residuals.iter().sum::<f64>() - expected_score).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn weighted_residual_sum_matches_partial_likelihood_score() {
        let n = 4;
        let covar = vec![0.0, 2.0, 4.0, 6.0];
        let score = vec![1.0, 1.25, 0.8, 1.1];
        let weights = vec![1.5, 0.75, 2.0, 0.5];
        let y = vec![1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.0, 0.0];
        let strata = vec![0; n];
        let risk: Vec<f64> = score
            .iter()
            .zip(&weights)
            .map(|(&score, &weight)| score * weight)
            .collect();
        let denom: f64 = risk.iter().sum();
        let risk_x: f64 = risk.iter().zip(&covar).map(|(&risk, &x)| risk * x).sum();
        let death_denom = risk[0] + risk[1];
        let death_risk_x = risk[0] * covar[0] + risk[1] * covar[1];
        let death_weight = weights[0] + weights[1];
        let observed = weights[0] * covar[0] + weights[1] * covar[1];

        for method in [0, 1] {
            let residuals = compute_cox_score_residuals(
                CoxScoreData {
                    y: &y,
                    strata: &strata,
                    covar: &covar,
                    score: &score,
                    weights: &weights,
                },
                CoxScoreParams { method, n, nvar: 1 },
            );
            let expected_score = if method == 0 {
                observed - death_weight * risk_x / denom
            } else {
                let mean_death_weight = death_weight / 2.0;
                observed
                    - mean_death_weight * risk_x / denom
                    - mean_death_weight * (risk_x - 0.5 * death_risk_x)
                        / (denom - 0.5 * death_denom)
            };
            let residual_score: f64 = residuals
                .iter()
                .zip(&weights)
                .map(|(&residual, &weight)| residual * weight)
                .sum();

            assert!((residual_score - expected_score).abs() < 1e-12);
        }
    }

    #[test]
    fn no_events_all_zero() {
        let n = 4;
        let nvar = 1;
        let y = vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0];
        let strata = vec![0, 0, 0, 0];
        let covar = vec![0.5, 1.0, 1.5, 2.0];
        let score = vec![1.0, 1.0, 1.0, 1.0];
        let weights = vec![1.0, 1.0, 1.0, 1.0];
        let data = CoxScoreData {
            y: &y,
            strata: &strata,
            covar: &covar,
            score: &score,
            weights: &weights,
        };
        let params = CoxScoreParams { method: 0, n, nvar };
        let result = compute_cox_score_residuals(data, params);
        for &r in &result {
            assert_eq!(r, 0.0);
        }
    }

    #[test]
    fn wrapper_accepts_strata_labels() {
        let result = cox_score_residuals(
            vec![1.0, 2.0, 1.5, 2.5, 1.0, 0.0, 1.0, 0.0],
            vec![2, 2, 4, 4],
            vec![0.5, 1.0, 1.5, 2.0],
            vec![1.0, 1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0],
            1,
            0,
        )
        .expect("strata are labels, not binary flags");

        assert_eq!(result.len(), 4);
    }

    #[test]
    fn wrapper_rejects_invalid_public_inputs() {
        let y = vec![1.0, 1.0, 2.0, 3.0, 1.0, 1.0, 0.0, 0.0];
        let strata = vec![0, 0, 0, 0];
        let covar = vec![1.0, 2.0, 3.0, 4.0];
        let score = vec![1.0, 1.0, 1.0, 1.0];
        let weights = vec![1.0, 1.0, 1.0, 1.0];

        let method_err = cox_score_residuals(
            y.clone(),
            strata.clone(),
            covar.clone(),
            score.clone(),
            weights.clone(),
            1,
            2,
        )
        .expect_err("unsupported method should fail");
        assert!(
            method_err
                .to_string()
                .contains("method must be 0 (Breslow) or 1 (Efron)")
        );

        let mut y_nan = y.clone();
        y_nan[0] = f64::NAN;
        let y_err = cox_score_residuals(
            y_nan,
            strata.clone(),
            covar.clone(),
            score.clone(),
            weights.clone(),
            1,
            0,
        )
        .expect_err("NaN y value should fail");
        assert!(y_err.to_string().contains("y contains non-finite"));

        let mut y_bad_status = y.clone();
        y_bad_status[5] = 0.5;
        let status_err = cox_score_residuals(
            y_bad_status,
            strata.clone(),
            covar.clone(),
            score.clone(),
            weights.clone(),
            1,
            0,
        )
        .expect_err("non-binary status should fail");
        assert!(
            status_err
                .to_string()
                .contains("status values must be 0 or 1")
        );

        let mut score_bad = score.clone();
        score_bad[1] = -1.0;
        let score_err = cox_score_residuals(
            y.clone(),
            strata.clone(),
            covar.clone(),
            score_bad,
            weights.clone(),
            1,
            0,
        )
        .expect_err("negative score should fail");
        assert!(
            score_err
                .to_string()
                .contains("score contains negative value")
        );

        let mut weights_bad = weights;
        weights_bad[2] = -1.0;
        let weight_err = cox_score_residuals(y, strata, covar, score, weights_bad, 1, 0)
            .expect_err("negative weight should fail");
        assert!(
            weight_err
                .to_string()
                .contains("weights contains negative value")
        );
    }
}
