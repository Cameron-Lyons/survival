//! Global tests of a Cox model's coefficients: the likelihood ratio, Wald
//! and score tests reported by R's `summary.coxph` (`R/summary.coxph.R`),
//! with the Wald quadratic form computed like `coxph.wtest`
//! (`src/coxph_wtest.c`: generalised Cholesky, redundant columns dropped).

use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::dist::pchisq;
use crate::internal::matrix::{cholesky2, chsolve2};
use crate::internal::validation::{validate_finite, validate_length};
use ndarray::Array2;
use pyo3::prelude::*;

/// R's `coxph.wtest` default `toler.chol`.
const WTEST_TOLERANCE: f64 = 1e-9;

/// A chi-square test statistic with its degrees of freedom and upper-tail
/// p-value.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object, get_all)]
pub struct TestResult {
    pub statistic: f64,
    pub df: usize,
    pub p_value: f64,
    pub test_name: String,
}

fn chi_square_result(statistic: f64, df: usize, test_name: &str) -> TestResult {
    TestResult {
        statistic,
        df,
        p_value: pchisq(statistic, df as f64, false, false),
        test_name: test_name.to_string(),
    }
}

/// Quadratic form `b' V^{-1} b` through R's `coxph.wtest`: the generalised
/// Cholesky factorisation of `var` with tolerance `toler`, redundant
/// (near-singular) columns contributing nothing.  Returns the statistic
/// and the rank of `var`.
fn quadratic_form(var: &[Vec<f64>], b: &[f64], toler: f64) -> SurvivalResult<(f64, usize)> {
    let n = b.len();
    validate_length(n, var.len(), "var rows")?;
    let mut chol = Array2::zeros((n, n));
    for (i, row) in var.iter().enumerate() {
        validate_length(n, row.len(), "var columns")?;
        validate_finite(row, "var")?;
        for (j, &value) in row.iter().enumerate() {
            chol[[i, j]] = value;
        }
    }
    let rank = cholesky2(&mut chol, toler);
    let mut solution = b.to_vec();
    chsolve2(&chol, &mut solution);
    let statistic = b
        .iter()
        .zip(&solution)
        .map(|(&value, &coefficient)| value * coefficient)
        .sum();
    Ok((statistic, rank.unsigned_abs() as usize))
}

/// Likelihood ratio test `2 (loglik_full - loglik_reduced)` on `df`
/// degrees of freedom (R's `logtest`).
pub fn likelihood_ratio_test(
    loglik_full: f64,
    loglik_reduced: f64,
    df: usize,
) -> SurvivalResult<TestResult> {
    if !loglik_full.is_finite() || !loglik_reduced.is_finite() {
        return Err(SurvivalError::invalid_input(
            "log-likelihoods must be finite",
        ));
    }
    if df == 0 {
        return Err(SurvivalError::invalid_input("df must be positive"));
    }
    Ok(chi_square_result(
        2.0 * (loglik_full - loglik_reduced),
        df,
        "LikelihoodRatioTest",
    ))
}

/// Wald test `(beta - init)' V^{-1} (beta - init)` on `length(beta)` degrees
/// of freedom, as `coxph` stores in `fit$wald.test` and `summary.coxph`
/// reports (`coef` and `var` restricted to the non-`NA` coefficients).
pub fn wald_test(
    coef: &[f64],
    var: &[Vec<f64>],
    init: Option<&[f64]>,
) -> SurvivalResult<TestResult> {
    if coef.is_empty() {
        return Err(SurvivalError::invalid_input("coef must not be empty"));
    }
    validate_finite(coef, "coef")?;
    let centred: Vec<f64> = match init {
        Some(init) => {
            validate_length(coef.len(), init.len(), "init")?;
            validate_finite(init, "init")?;
            coef.iter().zip(init).map(|(b, i)| b - i).collect()
        }
        None => coef.to_vec(),
    };
    let (statistic, _) = quadratic_form(var, &centred, WTEST_TOLERANCE)?;
    Ok(chi_square_result(statistic, coef.len(), "WaldTest"))
}

/// Score test `U' I^{-1} U` on `length(U)` degrees of freedom, `U` and `I`
/// being the score vector and information matrix at the initial
/// coefficients (R's `sctest`, computed by `coxfit6` on its first
/// iteration).
pub fn score_test(score: &[f64], information: &[Vec<f64>]) -> SurvivalResult<TestResult> {
    if score.is_empty() {
        return Err(SurvivalError::invalid_input("score must not be empty"));
    }
    validate_finite(score, "score")?;
    let (statistic, _) = quadratic_form(information, score, WTEST_TOLERANCE)?;
    Ok(chi_square_result(statistic, score.len(), "ScoreTest"))
}

#[pyfunction(name = "lrt_test")]
pub fn lrt_test_py(loglik_full: f64, loglik_reduced: f64, df: usize) -> PyResult<TestResult> {
    Ok(likelihood_ratio_test(loglik_full, loglik_reduced, df)?)
}

/// `wald_test(coef, var, init=None)`: `var` is the coefficient variance
/// matrix as nested rows.
#[pyfunction(name = "wald_test")]
#[pyo3(signature = (coef, var, init=None))]
pub fn wald_test_py(
    coef: Vec<f64>,
    var: Vec<Vec<f64>>,
    init: Option<Vec<f64>>,
) -> PyResult<TestResult> {
    Ok(wald_test(&coef, &var, init.as_deref())?)
}

#[pyfunction(name = "score_test")]
pub fn score_test_py(score: Vec<f64>, information: Vec<Vec<f64>>) -> PyResult<TestResult> {
    Ok(score_test(&score, &information)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wald_test_uses_the_full_covariance() {
        // beta' V^-1 beta with V = [[2, 0.5], [0.5, 1]], beta = (1, 2):
        // V^-1 = [[4, -2], [-2, 8]] / 7  ->  (4 - 8 + 32) / 7 = 4
        let result = wald_test(&[1.0, 2.0], &[vec![2.0, 0.5], vec![0.5, 1.0]], None).unwrap();
        assert!((result.statistic - 4.0).abs() < 1e-12);
        assert_eq!(result.df, 2);
        assert!((result.p_value - pchisq(4.0, 2.0, false, false)).abs() < 1e-15);
        let shifted = wald_test(
            &[1.0, 2.0],
            &[vec![2.0, 0.5], vec![0.5, 1.0]],
            Some(&[1.0, 2.0]),
        )
        .unwrap();
        assert_eq!(shifted.statistic, 0.0);
        assert_eq!(shifted.p_value, 1.0);
    }

    #[test]
    fn singular_variance_drops_redundant_columns_like_coxph_wtest() {
        let var = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 6.0],
            vec![3.0, 6.0, 9.0],
        ];
        let result = wald_test(&[1.0, 2.0, 3.0], &var, None).unwrap();
        assert!((result.statistic - 1.0).abs() < 1e-12);
    }

    #[test]
    fn score_test_matches_quadratic_form() {
        let result = score_test(&[1.0, 1.0], &[vec![2.0, 0.0], vec![0.0, 4.0]]).unwrap();
        assert!((result.statistic - 0.75).abs() < 1e-12);
        assert_eq!(result.df, 2);
    }

    #[test]
    fn likelihood_ratio_test_is_twice_the_difference() {
        let result = likelihood_ratio_test(-10.0, -12.0, 1).unwrap();
        assert!((result.statistic - 4.0).abs() < 1e-12);
        assert!((result.p_value - 0.04550026).abs() < 1e-7);
        assert!(likelihood_ratio_test(f64::NAN, -12.0, 1).is_err());
        assert!(likelihood_ratio_test(-10.0, -12.0, 0).is_err());
    }

    #[test]
    fn inputs_are_validated() {
        assert!(wald_test(&[], &[], None).is_err());
        assert!(wald_test(&[f64::INFINITY], &[vec![1.0]], None).is_err());
        assert!(wald_test(&[1.0], &[vec![1.0, 2.0]], None).is_err());
        assert!(wald_test(&[1.0], &[vec![1.0]], Some(&[1.0, 2.0])).is_err());
        assert!(score_test(&[], &[]).is_err());
        assert!(score_test(&[1.0], &[vec![f64::NAN]]).is_err());
        assert!(score_test(&[1.0, 2.0], &[vec![1.0], vec![0.0, 1.0]]).is_err());
    }
}
