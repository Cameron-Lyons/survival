//! Centring and scaling of the sorted covariates before the iterations, as
//! R's fitters do it.
//!
//! Each column statistic sums over the rows in the order of its column-wise
//! definition, so a few row-major passes over the matrix give the same bits
//! as one strided sweep per column and statistic.

use super::{CoxData, CoxFit, Fitter};
use ndarray::{Array1, Array2};

impl CoxFit {
    /// Centres and scales the `doscale` columns in place, records their
    /// means, scales and predictor bounds, and moves the initial coefficients
    /// onto the scaled columns.
    pub(super) fn scale_center(&mut self, doscale: &[bool]) {
        let CoxData {
            ref mut covar,
            ref weights,
            ref mut predictor_bounds,
            ..
        } = self.data;
        let nused = covar.nrows();
        let (centers, means) = match &self.fitter {
            Fitter::Coxfit6 { .. } => {
                let means = weighted_column_means(covar, weights);
                (means.clone(), means)
            }
            Fitter::Agfit4 { walk, .. } => (
                walk.by_stop.first().map_or_else(
                    || vec![0.0; covar.ncols()],
                    |first| covar.row(first.row).to_vec(),
                ),
                plain_column_means(covar),
            ),
            Fitter::Coxexact | Fitter::Agexact { .. } => {
                let means = plain_column_means(covar);
                (means.clone(), means)
            }
        };
        let scales = match &self.fitter {
            Fitter::Coxfit6 { .. } => mean_abs_scales(covar, weights, &centers, 0..nused),
            Fitter::Agfit4 { walk, .. } => mean_abs_scales(
                covar,
                weights,
                &centers,
                walk.by_stop.iter().map(|joining| joining.row),
            ),
            Fitter::Coxexact => inverse_sds(covar, &centers),
            Fitter::Agexact { .. } => vec![1.0; centers.len()],
        };
        for (i, &scaled) in doscale.iter().enumerate() {
            (self.means[i], self.scale[i]) = if scaled {
                (means[i], scales[i])
            } else {
                (0.0, 1.0)
            };
        }
        predictor_bounds.fill(0.0);
        for mut row in covar.rows_mut() {
            for (i, value) in row.iter_mut().enumerate() {
                if doscale[i] {
                    *value = (*value - centers[i]) * scales[i];
                }
                predictor_bounds[i] = predictor_bounds[i].max(value.abs());
            }
        }
        for (beta, &scale) in self.beta.iter_mut().zip(&self.scale) {
            *beta /= scale;
        }
    }
}

/// Case-weighted column means. The sums fold from -0.0 in row order, as
/// `Iterator::sum` folds, so a column of negative zeros keeps its sign.
fn weighted_column_means(covar: &Array2<f64>, weights: &Array1<f64>) -> Vec<f64> {
    let total_weight: f64 = weights.sum();
    let mut sums = vec![-0.0; covar.ncols()];
    for (row, &weight) in covar.rows().into_iter().zip(weights) {
        for (sum, &value) in sums.iter_mut().zip(row) {
            *sum += weight * value;
        }
    }
    sums.into_iter().map(|sum| sum / total_weight).collect()
}

/// Plain column means. ndarray sums a strided column in row order from 0.0,
/// as this row-major pass does, but a contiguous one (the lone column of a
/// one-covariate matrix) with an unrolled fold, which that case keeps.
fn plain_column_means(covar: &Array2<f64>) -> Vec<f64> {
    let nused = covar.nrows() as f64;
    if covar.ncols() == 1 {
        return vec![covar.column(0).sum() / nused];
    }
    let mut sums = vec![0.0; covar.ncols()];
    for row in covar.rows() {
        for (sum, &value) in sums.iter_mut().zip(row) {
            *sum += value;
        }
    }
    sums.into_iter().map(|sum| sum / nused).collect()
}

/// R's scale `1 / mean(|x - centre|)` for each column, case-weighted over
/// `rows` in order, or 1 for a column without spread.
fn mean_abs_scales(
    covar: &Array2<f64>,
    weights: &Array1<f64>,
    centers: &[f64],
    rows: impl Iterator<Item = usize>,
) -> Vec<f64> {
    let mut weight = 0.0;
    let mut abs_sums = vec![0.0; centers.len()];
    for person in rows {
        let case_weight = weights[person];
        weight += case_weight;
        for ((sum, &value), &center) in abs_sums.iter_mut().zip(covar.row(person)).zip(centers) {
            *sum += case_weight * (value - center).abs();
        }
    }
    abs_sums
        .into_iter()
        .map(|abs_sum| if abs_sum > 0.0 { weight / abs_sum } else { 1.0 })
        .collect()
}

/// coxexact's scale `1 / sd` about the centres. The squares form a contiguous
/// array, which ndarray sums with its unrolled fold.
fn inverse_sds(covar: &Array2<f64>, centers: &[f64]) -> Vec<f64> {
    let nused = covar.nrows() as f64;
    covar
        .columns()
        .into_iter()
        .zip(centers)
        .map(|(column, &center)| {
            let squares = column.mapv(|value| {
                let value = value - center;
                value * value
            });
            let sd = (squares.sum() / (nused - 1.0)).sqrt();
            if sd > 0.0 { 1.0 / sd } else { 1.0 }
        })
        .collect()
}
