use super::common::{build_score_result, validate_scoring_inputs};
use crate::internal::validation::{PermutationIndexError, validate_one_based_i32_permutation};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

fn validate_one_based_sort1(sort1: &[i32], n: usize) -> Result<Vec<usize>, String> {
    if sort1.len() != n {
        return Err("Sort1 length does not match observations".to_string());
    }
    match validate_one_based_i32_permutation(sort1, n) {
        Ok(normalized) => Ok(normalized),
        Err(PermutationIndexError::Negative { position, value }) => Err(format!(
            "Sort1 value {value} at position {position} is outside 1..={n}"
        )),
        Err(PermutationIndexError::OutOfBounds { position, value }) => Err(format!(
            "Sort1 value {value} at position {position} is outside 1..={n}"
        )),
        Err(PermutationIndexError::Duplicate { position, value }) => Err(format!(
            "Sort1 must be a permutation of 1..={n}; duplicate index {value} at position {position}"
        )),
    }
}

#[inline]
fn finish_residual(
    row: usize,
    n: usize,
    covar: &[f64],
    score: &[f64],
    cumhaz: f64,
    xhaz: &[f64],
    residuals: &mut [f64],
) {
    for (var, &xhaz_value) in xhaz.iter().enumerate() {
        let idx = var * n + row;
        residuals[idx] -= score[row] * (cumhaz * covar[idx] - xhaz_value);
    }
}

#[inline]
pub(crate) fn agscore3(
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

    let mut residuals = vec![0.0; nvar * n];
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
    let mut entry_pos = n;
    let sort1 = validate_one_based_sort1(sort1, n)?;
    let mut death_rows = Vec::new();

    let mut person = n;
    while person > 0 {
        let current_row = person - 1;
        let dtime = tstop[current_row];

        if strata[current_row] != current_stratum {
            while entry_pos > 0 && sort1[entry_pos - 1] > current_row {
                entry_pos -= 1;
                finish_residual(
                    sort1[entry_pos],
                    n,
                    covar,
                    score,
                    cumhaz,
                    &xhaz,
                    &mut residuals,
                );
            }

            cumhaz = 0.0;
            denom = 0.0;
            a.fill(0.0);
            xhaz.fill(0.0);
            current_stratum = strata[current_row];
        } else {
            while entry_pos > 0 && tstart[sort1[entry_pos - 1]] >= dtime {
                let k = sort1[entry_pos - 1];
                if strata[k] != current_stratum {
                    break;
                }
                entry_pos -= 1;
                finish_residual(k, n, covar, score, cumhaz, &xhaz, &mut residuals);
                let risk = score[k] * weights[k];
                denom -= risk;
                for var in 0..nvar {
                    a[var] -= risk * covar[var * n + k];
                }
            }
        }

        let mut e_denom = 0.0;
        let mut meanwt = 0.0;
        a2.fill(0.0);

        death_rows.clear();
        while person > 0 && tstop[person - 1] == dtime {
            let row = person - 1;
            if strata[row] != current_stratum {
                break;
            }
            person -= 1;

            for (var, &xhaz_value) in xhaz.iter().enumerate() {
                let idx = var * n + row;
                residuals[idx] = (covar[idx] * cumhaz - xhaz_value) * score[row];
            }
            let risk = score[row] * weights[row];
            denom += risk;
            for var in 0..nvar {
                a[var] += risk * covar[var * n + row];
            }

            if event[row] == 1.0 {
                death_rows.push(row);
                e_denom += risk;
                meanwt += weights[row];
                for var in 0..nvar {
                    a2[var] += risk * covar[var * n + row];
                }
            }
        }

        if !death_rows.is_empty() {
            let deaths = death_rows.len() as f64;
            if death_rows.len() == 1 || method == 0 {
                let hazard = meanwt / denom;
                cumhaz += hazard;
                for var in 0..nvar {
                    mean[var] = a[var] / denom;
                    xhaz[var] += mean[var] * hazard;
                }

                for &row in &death_rows {
                    for (var, &mean_value) in mean.iter().enumerate() {
                        let idx = var * n + row;
                        residuals[idx] += covar[idx] - mean_value;
                    }
                }
            } else {
                mh1.fill(0.0);
                mh2.fill(0.0);
                mh3.fill(0.0);
                let meanwt_norm = meanwt / deaths;

                for dd in 0..death_rows.len() {
                    let downwt = dd as f64 / deaths;
                    let d2 = denom - downwt * e_denom;
                    let hazard = meanwt_norm / d2;
                    cumhaz += hazard;
                    for var in 0..nvar {
                        mean[var] = (a[var] - downwt * a2[var]) / d2;
                        xhaz[var] += mean[var] * hazard;
                        mh1[var] += hazard * downwt;
                        mh2[var] += mean[var] * hazard * downwt;
                        mh3[var] += mean[var] / deaths;
                    }
                }

                for &row in &death_rows {
                    for var in 0..nvar {
                        let idx = var * n + row;
                        residuals[idx] += (covar[idx] - mh3[var])
                            + score[row] * (covar[idx] * mh1[var] - mh2[var]);
                    }
                }
            }
        }
    }

    while entry_pos > 0 {
        entry_pos -= 1;
        finish_residual(
            sort1[entry_pos],
            n,
            covar,
            score,
            cumhaz,
            &xhaz,
            &mut residuals,
        );
    }

    Ok(residuals)
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
    let (n, nvar) =
        validate_scoring_inputs(&time_data, &covariates, &strata, &score, &weights, method)?;
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
    Python::attach(|py| build_score_result(py, residuals, n, nvar, method).map(|d| d.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-12,
                "value {idx} differed: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn includes_an_event_in_the_first_row() {
        let n = 3;
        let y = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 1.0, 0.0, 0.0];
        let covar = vec![0.5, 1.0, 1.5];
        let strata = vec![0, 0, 0];
        let score = vec![1.0, 1.0, 1.0];
        let weights = vec![1.0, 1.0, 1.0];
        let sort1 = vec![1, 2, 3];
        let result = agscore3(&y, &covar, &strata, &score, &weights, 0, &sort1).unwrap();
        assert_eq!(result.len(), n);
        assert_close(&result, &[-1.0 / 3.0, 0.0, -1.0 / 6.0]);
    }

    #[test]
    fn two_covariates_output_length() {
        let n = 3;
        let nvar = 2;
        let y = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0];
        let covar = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let strata = vec![0, 0, 0];
        let score = vec![1.0, 1.0, 1.0];
        let weights = vec![1.0, 1.0, 1.0];
        let sort1 = vec![1, 2, 3];
        let result = agscore3(&y, &covar, &strata, &score, &weights, 0, &sort1).unwrap();
        assert_eq!(result.len(), n * nvar);
    }

    #[test]
    fn matches_weighted_stratified_tied_event_reference() {
        let y = vec![
            0.0, 0.0, 0.0, 1.0, 1.5, 0.0, 0.0, 1.0, 0.0, // start
            1.0, 2.0, 2.0, 2.0, 3.0, 1.0, 2.0, 2.0, 3.0, // stop
            1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, // event
        ];
        let covar = vec![
            -1.0, 0.2, 1.1, 0.5, -0.4, 0.8, -0.7, 0.3, 1.4, // x1
            0.5, -1.0, 0.7, 1.2, -0.2, -0.6, 0.9, -1.3, 0.1, // x2
        ];
        let strata = vec![0, 0, 0, 0, 0, 1, 1, 1, 1];
        let score = vec![1.0, 1.2, 0.8, 1.5, 0.7, 1.1, 0.9, 1.4, 0.6];
        let weights = vec![0.5, 2.0, 1.25, 0.75, 1.5, 1.0, 0.4, 2.2, 0.8];
        let sort1 = vec![1, 2, 3, 4, 5, 6, 7, 9, 8];
        let cases = [
            (
                0,
                vec![
                    -1.11321499013807,
                    -0.0209051282804873,
                    0.337701743846123,
                    -0.167045385992077,
                    0.289310060528062,
                    1.1927614710623e-17,
                    -0.42033527696793,
                    -0.00306122448979592,
                    -0.42069970845481,
                    0.760026298487837,
                    -0.173621914405268,
                    0.317217447427648,
                    -1.13716342576766,
                    0.0406241830722516,
                    1.12952823548651e-17,
                    0.736203665139525,
                    -0.02667638483965,
                    -0.408517284464806,
                ],
            ),
            (
                1,
                vec![
                    -1.11321499013807,
                    -0.0146311379877521,
                    0.412169374387006,
                    -0.246665252829545,
                    0.332715384062158,
                    2.41063196411902e-17,
                    -0.481287473194709,
                    -0.0151917692696914,
                    -0.544267209599306,
                    0.760026298487837,
                    -0.305463552399534,
                    0.311359764467586,
                    -1.27634446002223,
                    0.100990551473691,
                    4.10204557001015e-18,
                    0.769461336701994,
                    -0.0552638122545356,
                    -0.528506576116701,
                ],
            ),
        ];

        for (method, expected) in cases {
            let actual = agscore3(&y, &covar, &strata, &score, &weights, method, &sort1).unwrap();
            assert_close(&actual, &expected);
        }
    }

    #[test]
    fn rejects_invalid_sort1_values() {
        let y = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0];
        let covar = vec![0.5, 1.0, 1.5];
        let strata = vec![0, 0, 0];
        let score = vec![1.0, 1.0, 1.0];
        let weights = vec![1.0, 1.0, 1.0];

        let zero = agscore3(&y, &covar, &strata, &score, &weights, 0, &[1, 0, 3])
            .expect_err("zero sort1 index should fail");
        assert!(zero.contains("outside 1..=3"));

        let duplicate = agscore3(&y, &covar, &strata, &score, &weights, 0, &[1, 1, 3])
            .expect_err("duplicate sort1 index should fail");
        assert!(duplicate.contains("Sort1 must be a permutation"));
    }
}
