use super::agscore3::agscore3;
use super::common::{build_score_result, validate_scoring_inputs};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[inline]
pub(crate) fn agscore2(
    y: &[f64],
    covar: &[f64],
    strata: &[i32],
    score: &[f64],
    weights: &[f64],
    method: i32,
) -> Result<Vec<f64>, String> {
    let n = y.len() / 3;
    let tstart = &y[0..n];
    let mut stratum_blocks = vec![0usize; n];
    for idx in 1..n {
        stratum_blocks[idx] = stratum_blocks[idx - 1] + usize::from(strata[idx] != strata[idx - 1]);
    }
    let mut start_order: Vec<usize> = (0..n).collect();
    start_order.sort_by(|&left, &right| {
        stratum_blocks[left]
            .cmp(&stratum_blocks[right])
            .then_with(|| tstart[left].total_cmp(&tstart[right]))
            .then_with(|| left.cmp(&right))
    });
    let sort1: Vec<i32> = start_order
        .into_iter()
        .map(|idx| {
            i32::try_from(idx + 1)
                .map_err(|_| "Observation count exceeds supported sort index range".to_string())
        })
        .collect::<Result<_, _>>()?;

    agscore3(y, covar, strata, score, weights, method, &sort1)
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
    let (n, nvar) =
        validate_scoring_inputs(&time_data, &covariates, &strata, &score, &weights, method)?;
    let residuals = agscore2(&time_data, &covariates, &strata, &score, &weights, method)
        .map_err(PyRuntimeError::new_err)?;
    Python::attach(|py| build_score_result(py, residuals, n, nvar, method).map(|d| d.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_event_breslow() {
        let n = 3;
        let nvar = 1;
        let y = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0];
        let covar = vec![0.5, 1.0, 1.5];
        let strata = vec![0, 0, 0];
        let score = vec![1.0, 1.0, 1.0];
        let weights = vec![1.0, 1.0, 1.0];
        let result = agscore2(&y, &covar, &strata, &score, &weights, 0).unwrap();
        assert_eq!(result.len(), n * nvar);
        assert_eq!(result, vec![0.0, -0.125, -0.125]);
    }

    #[test]
    fn no_events_all_zero() {
        let y = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0];
        let covar = vec![0.5, 1.0, 1.5];
        let strata = vec![0, 0, 0];
        let score = vec![1.0, 1.0, 1.0];
        let weights = vec![1.0, 1.0, 1.0];
        let result = agscore2(&y, &covar, &strata, &score, &weights, 0).unwrap();
        for &r in &result {
            assert_eq!(r, 0.0);
        }
    }

    #[test]
    fn accepts_stratum_labels_outside_numeric_order() {
        let y = vec![
            0.0, 0.0, 0.0, 0.0, // start
            1.0, 2.0, 1.0, 2.0, // stop
            1.0, 0.0, 1.0, 0.0, // event
        ];
        let covar = vec![0.0, 1.0, 2.0, 3.0];
        let strata = vec![4, 4, 2, 2];
        let score = vec![1.0; 4];
        let weights = vec![1.0; 4];
        let result = agscore2(&y, &covar, &strata, &score, &weights, 0).unwrap();
        assert_eq!(result, vec![-0.25; 4]);
    }

    #[test]
    fn breslow_vs_efron_tied_deaths() {
        let y = vec![0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 3.0, 1.0, 1.0, 0.0, 0.0];
        let covar = vec![1.0, 2.0, 3.0, 4.0];
        let strata = vec![0, 0, 0, 0];
        let score = vec![1.0, 1.0, 1.0, 1.0];
        let weights = vec![1.0, 1.0, 1.0, 1.0];
        let breslow = agscore2(&y, &covar, &strata, &score, &weights, 0).unwrap();
        let efron = agscore2(&y, &covar, &strata, &score, &weights, 1).unwrap();
        let differs = breslow
            .iter()
            .zip(efron.iter())
            .any(|(a, b)| (a - b).abs() > 1e-15);
        assert!(differs);
    }

    #[test]
    fn multiple_covariates_output_length() {
        let n = 3;
        let nvar = 2;
        let y = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0];
        let covar = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let strata = vec![0, 0, 0];
        let score = vec![1.0, 1.0, 1.0];
        let weights = vec![1.0, 1.0, 1.0];
        let result = agscore2(&y, &covar, &strata, &score, &weights, 0).unwrap();
        assert_eq!(result.len(), n * nvar);
    }
}
