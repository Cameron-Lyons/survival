//! Score residual kernels of R's `survival` package (`coxscore2.c`,
//! `agscore3.c`).  Both take the row-major `n x p` covariate matrix and the
//! stratum-label convention of `crate::core::strata_order`, and both return the
//! `n x p` matrix of per-observation score residuals in input order.

pub mod agscore3;
pub mod coxscore2;

pub use agscore3::agscore3;
pub use coxscore2::coxscore2;

use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::{validate_finite, validate_length, validate_non_negative};
use ndarray::ArrayView2;

/// Shared argument checks: the covariate matrix has `n` rows, `score`
/// (`exp(eta)`) and the optional weights/strata have `n` entries.
pub(crate) fn validate_score_inputs(
    n: usize,
    covariates: ArrayView2<'_, f64>,
    score: &[f64],
    weights: Option<&[f64]>,
    strata: Option<&[i32]>,
) -> SurvivalResult<()> {
    if covariates.nrows() != n {
        return Err(SurvivalError::invalid_input(format!(
            "covariates has {} rows but the response has {n}",
            covariates.nrows()
        )));
    }
    if covariates.ncols() == 0 {
        return Err(SurvivalError::invalid_input(
            "covariates must have at least one column",
        ));
    }
    if let Some(value) = covariates.iter().find(|value| !value.is_finite()) {
        return Err(SurvivalError::invalid_input(format!(
            "covariates contains non-finite value {value}"
        )));
    }
    validate_length(n, score.len(), "score")?;
    validate_finite(score, "score")?;
    validate_non_negative(score, "score")?;
    if let Some(weights) = weights {
        validate_length(n, weights.len(), "weights")?;
        validate_finite(weights, "weights")?;
        validate_non_negative(weights, "weights")?;
    }
    if let Some(strata) = strata {
        validate_length(n, strata.len(), "strata")?;
    }
    Ok(())
}
