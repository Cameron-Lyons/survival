pub mod agscore3;
pub mod coxscore2;

pub use agscore3::agscore3;
pub use coxscore2::coxscore2;

use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::{validate_finite, validate_length, validate_non_negative};
use ndarray::{Array2, ArrayView2};

/// The rows of `matrix` in `order` (row `k` is `matrix` row `order[k]`),
/// copied as whole contiguous rows.
pub(crate) fn gather_rows(matrix: ArrayView2<'_, f64>, order: &[usize]) -> Array2<f64> {
    let ncols = matrix.ncols();
    let source = matrix.as_standard_layout();
    let source = source.as_slice().expect("standard layout is contiguous");
    let mut rows = Vec::with_capacity(order.len() * ncols);
    for &row in order {
        rows.extend_from_slice(&source[row * ncols..(row + 1) * ncols]);
    }
    Array2::from_shape_vec((order.len(), ncols), rows).expect("one row per ordered index")
}

/// Undoes [`gather_rows`]: row `order[k]` of the result is row `k` of `sorted`.
pub(crate) fn scatter_rows(sorted: &Array2<f64>, order: &[usize]) -> Array2<f64> {
    let ncols = sorted.ncols();
    let source = sorted.as_standard_layout();
    let source = source.as_slice().expect("standard layout is contiguous");
    let mut rows = Array2::zeros((order.len(), ncols));
    let target = rows.as_slice_mut().expect("a new array is contiguous");
    for (position, &row) in order.iter().enumerate() {
        target[row * ncols..(row + 1) * ncols]
            .copy_from_slice(&source[position * ncols..(position + 1) * ncols]);
    }
    rows
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_gather_and_scatter_copy_rows_of_any_layout() {
        let matrix = Array2::from_shape_fn((3, 4), |(i, j)| (i * 4 + j) as f64 - 0.5);
        let order = [3, 0, 2, 1];
        let columns = matrix.t();
        let gathered = gather_rows(columns, &order);
        for (position, &row) in order.iter().enumerate() {
            assert_eq!(gathered.row(position), columns.row(row));
        }
        assert_eq!(scatter_rows(&gathered, &order), columns);
        let empty = Array2::<f64>::zeros((2, 0));
        assert_eq!(gather_rows(empty.view(), &[1, 0]).dim(), (2, 0));
        assert_eq!(scatter_rows(&empty, &[1, 0]).dim(), (2, 0));
    }
}
