//! SAS-style type III tests from the overparameterized indicator design.

use super::{YatesContrast, estimates, quadratic_form, transpose};
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::numpy_utils::{FloatRows, FloatVec};
use crate::internal::qr::{LinpackLeastSquares, LinpackQr};
use crate::internal::validation::{validate_finite, validate_length};
use pyo3::prelude::*;

/// Inputs to [`yates_sgtt`]. Formula expansion stays with the caller.
pub struct YatesSgttInput<'a> {
    /// Full indicator model matrix, in R's SAS column order.
    pub x: &'a [Vec<f64>],
    /// Term number of every full indicator column (zero for the intercept).
    pub assign: &'a [usize],
    /// Later categorical terms sharing a variable, indexed by term minus one.
    pub adjustment_terms: &'a [Vec<usize>],
    pub beta: &'a [f64],
    pub vmat: &'a [Vec<f64>],
    /// Term number of each non-aliased fitted coefficient.
    pub coefficient_assign: &'a [usize],
    /// Term numbers and labels to test.
    pub test_terms: &'a [(usize, String)],
    pub sigma2: Option<f64>,
    /// False for a Cox model, whose baseline absorbs the intercept.
    pub include_intercept: bool,
}

/// The estimable SAS hypothesis matrix and one test per requested term.
#[derive(Debug, Clone)]
#[pyclass(from_py_object, get_all)]
pub struct YatesSgttResult {
    pub sas: Vec<Vec<f64>>,
    /// Original full-design column numbers retained by the QR (zero based).
    pub columns: Vec<usize>,
    pub test: Vec<YatesContrast>,
}

/// R's SGTT branch of `yates`: solve `X D = X`, residualize each categorical
/// term against later related terms, and test the retained estimable rows.
pub fn yates_sgtt(input: &YatesSgttInput<'_>) -> SurvivalResult<YatesSgttResult> {
    let n = input.x.len();
    let p = input.assign.len();
    let nterms = input.adjustment_terms.len();
    if n == 0 || p == 0 {
        return Err(SurvivalError::invalid_input(
            "SAS design must have rows and columns",
        ));
    }
    for row in input.x {
        validate_length(p, row.len(), "SAS design columns")?;
        validate_finite(row, "SAS design")?;
    }
    if input
        .assign
        .iter()
        .chain(input.coefficient_assign)
        .any(|&term| term > nterms)
        || input
            .test_terms
            .iter()
            .any(|(term, _)| *term == 0 || *term > nterms)
        || input
            .adjustment_terms
            .iter()
            .enumerate()
            .any(|(i, terms)| terms.iter().any(|&term| term <= i + 1 || term > nterms))
    {
        return Err(SurvivalError::invalid_input("invalid SAS term assignment"));
    }
    validate_length(
        input.beta.len(),
        input.coefficient_assign.len(),
        "coefficient assignments",
    )?;
    validate_finite(input.beta, "beta")?;
    validate_length(input.beta.len(), input.vmat.len(), "variance rows")?;
    for row in input.vmat {
        validate_length(input.beta.len(), row.len(), "variance columns")?;
        validate_finite(row, "variance")?;
    }
    if input
        .sigma2
        .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        return Err(SurvivalError::invalid_input(
            "sigma2 must be finite and non-negative",
        ));
    }
    let xcolumns = transpose(input.x);
    let qr = LinpackLeastSquares::new(xcolumns.clone(), n);
    // Each fitted response is a row of B = t(D). Missing coefficient columns
    // have the same positions for every response.
    let mut b: Vec<Vec<f64>> = xcolumns
        .iter()
        .map(|column| qr.coefficients(column))
        .collect();
    let columns: Vec<usize> = (0..p)
        .filter(|&j| !b[0][j].is_nan() && (input.include_intercept || input.assign[j] != 0))
        .collect();
    validate_length(input.beta.len(), columns.len(), "estimable SAS columns")?;
    for row in &mut b {
        for value in row {
            if value.is_nan() {
                *value = 0.0;
            }
        }
    }
    for (i, adjusters) in input.adjustment_terms.iter().enumerate() {
        if adjusters.is_empty() {
            continue;
        }
        let mut predictors = vec![vec![1.0; p]];
        predictors.extend(
            (0..p)
                .filter(|&j| adjusters.contains(&input.assign[j]))
                .map(|j| b.iter().map(|row| row[j]).collect()),
        );
        let adjustment = LinpackQr::new(predictors, p);
        for j in (0..p).filter(|&j| input.assign[j] == i + 1) {
            let column: Vec<f64> = b.iter().map(|row| row[j]).collect();
            let residual = adjustment.residual(&column);
            for (row, value) in b.iter_mut().zip(residual) {
                row[j] = value;
            }
        }
    }
    let sas: Vec<Vec<f64>> = columns
        .iter()
        .map(|&i| columns.iter().map(|&j| b[j][i]).collect())
        .collect();
    let test = input
        .test_terms
        .iter()
        .map(|(term, name)| {
            let rows: Vec<Vec<f64>> = sas
                .iter()
                .zip(input.coefficient_assign)
                .filter(|(_, assignment)| **assignment == *term)
                .map(|(row, _)| row.clone())
                .collect();
            let (estimate, var) = estimates(&rows, input.beta, input.vmat);
            let (chisq, df) = quadratic_form(&var, &estimate);
            YatesContrast {
                name: name.clone(),
                chisq,
                df: Some(df),
                ss: input.sigma2.map(|s| chisq * s),
            }
        })
        .collect();
    Ok(YatesSgttResult { sas, columns, test })
}

/// Python entry point for the native SAS hypothesis construction.
#[pyfunction(name = "yates_sgtt")]
#[pyo3(signature = (x, assign, adjustment_terms, beta, vmat, coefficient_assign, test_terms, sigma2=None, include_intercept=true))]
#[allow(clippy::too_many_arguments)]
pub fn yates_sgtt_py(
    py: Python<'_>,
    x: FloatRows,
    assign: Vec<usize>,
    adjustment_terms: Vec<Vec<usize>>,
    beta: FloatVec,
    vmat: FloatRows,
    coefficient_assign: Vec<usize>,
    test_terms: Vec<(usize, String)>,
    sigma2: Option<f64>,
    include_intercept: bool,
) -> PyResult<YatesSgttResult> {
    Ok(py.detach(|| {
        yates_sgtt(&YatesSgttInput {
            x: &x,
            assign: &assign,
            adjustment_terms: &adjustment_terms,
            beta: &beta,
            vmat: &vmat,
            coefficient_assign: &coefficient_assign,
            test_terms: &test_terms,
            sigma2,
            include_intercept,
        })
    })?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balanced_factorial_type_three_hypothesis() {
        let x = vec![
            vec![1., 0., 1., 0., 1., 0., 0., 0., 1.],
            vec![1., 1., 0., 0., 1., 0., 0., 1., 0.],
            vec![1., 0., 1., 1., 0., 0., 1., 0., 0.],
            vec![1., 1., 0., 1., 0., 1., 0., 0., 0.],
        ];
        let result = yates_sgtt(&YatesSgttInput {
            x: &x,
            assign: &[0, 1, 1, 2, 2, 3, 3, 3, 3],
            adjustment_terms: &[vec![3], vec![3], vec![]],
            beta: &[2., 3., 4., 5.],
            vmat: &[
                vec![1., 0., 0., 0.],
                vec![0., 1., 0., 0.],
                vec![0., 0., 1., 0.],
                vec![0., 0., 0., 1.],
            ],
            coefficient_assign: &[0, 1, 2, 3],
            test_terms: &[(1, "a".into())],
            sigma2: Some(2.0),
            include_intercept: true,
        })
        .unwrap();
        assert_eq!(result.columns, vec![0, 1, 3, 5]);
        let expected = [
            vec![1., 0., 0., 0.],
            vec![0., 1., 0., 0.5],
            vec![0., 0., 1., 0.5],
            vec![0., 0., 0., 1.],
        ];
        for (row, expected) in result.sas.iter().zip(expected) {
            for (actual, expected) in row.iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-12);
            }
        }
        assert!((result.test[0].chisq - 24.2).abs() < 1e-10);
        assert_eq!(result.test[0].df, Some(1));
    }
}
