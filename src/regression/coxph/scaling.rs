use super::{CoxPHFit, SortedRows};
use crate::regression::cox_optimizer::covariate_anchor_needed;
pub(crate) use crate::regression::cox_optimizer::{add_predictor_parts, relative_covariate_scale};
use crate::regression::log_scale::LogScale;
use ndarray::{ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// Ordinary predictions keep one float per row. Paired storage is allocated
/// only when a row needs compensation before the baseline scale cancels.
pub(super) enum PredictionRisks {
    Scalar(Vec<f64>),
    Scaled(Vec<LogScale>),
}

impl PredictionRisks {
    pub fn risks(&self) -> Vec<f64> {
        match self {
            Self::Scalar(values) => values.iter().map(|value| value.exp()).collect(),
            Self::Scaled(values) => values.iter().map(|value| value.exp_at(0.0)).collect(),
        }
    }

    pub fn needs_logs(&self, stable: bool) -> bool {
        stable
            || match self {
                Self::Scalar(values) => values.iter().any(|v| !(-30.0..=30.0).contains(v)),
                Self::Scaled(values) => values
                    .iter()
                    .any(|v| !(-30.0..=30.0).contains(&v.log_value())),
            }
    }

    pub fn scale(&self, index: usize) -> LogScale {
        match self {
            Self::Scalar(values) => LogScale::from_log(values[index]),
            Self::Scaled(values) => values[index],
        }
    }

    pub fn as_scales(&self) -> Cow<'_, [LogScale]> {
        match self {
            Self::Scalar(values) => {
                Cow::Owned(values.iter().map(|&v| LogScale::from_log(v)).collect())
            }
            Self::Scaled(values) => Cow::Borrowed(values),
        }
    }

    pub fn as_scalar(&self) -> Cow<'_, [f64]> {
        match self {
            Self::Scalar(values) => Cow::Borrowed(values),
            Self::Scaled(values) => Cow::Owned(values.iter().map(|v| v.log_value()).collect()),
        }
    }
}

/// A covariate and offset reference kept separately until prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PredictorOrigin {
    pub offset: f64,
    pub covariates: Option<Vec<f64>>,
}

impl PredictorOrigin {
    pub fn reference(offset: f64) -> Self {
        Self {
            offset,
            covariates: None,
        }
    }
}

/// Retain the resolved predictor alongside its public, rounded presentation.
/// Allocated only for unsafe offsets or dot products.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RiskPredictors {
    pub(crate) values: Vec<f64>,
    pub(crate) scales: Vec<LogScale>,
    origins: Vec<PredictorOrigin>,
    public_origins: Vec<LogScale>,
    pub(crate) public_values: Vec<f64>,
}

impl RiskPredictors {
    pub(crate) fn new(
        origins: Vec<PredictorOrigin>,
        scales: Vec<LogScale>,
        public_origins: Vec<LogScale>,
        public_values: Vec<f64>,
    ) -> Self {
        Self {
            values: scales.iter().map(|scale| scale.log_value()).collect(),
            scales,
            origins,
            public_origins,
            public_values,
        }
    }
}

pub(crate) fn predictor_origins(
    x: ArrayView2<'_, f64>,
    beta: &[f64],
    sorted: &SortedRows,
    unsafe_offset: bool,
) -> Option<Vec<PredictorOrigin>> {
    let covariates = covariate_anchor_needed(x, beta);
    (unsafe_offset || covariates).then(|| {
        sorted
            .bounds
            .iter()
            .map(|&(_, end)| PredictorOrigin {
                // Paired risk sums can retain the absolute offset itself. This avoids
                // overflowing offset-offset_origin for opposite finite offsets.
                offset: 0.0,
                covariates: covariates.then(|| x.row(sorted.order[end - 1]).to_vec()),
            })
            .collect()
    })
}

pub(crate) fn predictor_scales(
    x: ArrayView2<'_, f64>,
    beta: &[f64],
    means: &[f64],
    offset: &[f64],
    sorted: &SortedRows,
    origins: &[PredictorOrigin],
) -> Vec<LogScale> {
    x.rows()
        .into_iter()
        .enumerate()
        .map(|(i, row)| {
            let origin = &origins[sorted.stratum_index[i]];
            let cov =
                relative_covariate_scale(row, origin.covariates.as_deref().unwrap_or(means), beta);
            add_predictor_parts(LogScale::from_log(offset[i] - origin.offset), cov)
        })
        .collect()
}

pub(crate) fn predictor_origin_scales(
    origins: &[PredictorOrigin],
    means: &[f64],
    beta: &[f64],
) -> Vec<LogScale> {
    origins
        .iter()
        .map(|origin| {
            let cov = origin
                .covariates
                .as_ref()
                .map_or(LogScale::from_log(0.0), |anchor| {
                    relative_covariate_scale(ArrayView1::from(anchor.as_slice()), means, beta)
                });
            add_predictor_parts(LogScale::from_log(origin.offset), cov)
        })
        .collect()
}

pub(crate) fn predictor_ranks(scales: &[LogScale], sorted: &SortedRows) -> Vec<f64> {
    let mut result = vec![0.0; scales.len()];
    for &(start, end) in &sorted.bounds {
        let mut rows = sorted.order[start..end].to_vec();
        rows.sort_by(|&a, &b| scales[a].ratio_log(scales[b]).total_cmp(&0.0));
        let mut rank = 0.0;
        for (i, &row) in rows.iter().enumerate() {
            if i > 0 && scales[row].ratio_log(scales[rows[i - 1]]) != 0.0 {
                rank += 1.0;
            }
            result[row] = rank;
        }
    }
    result
}

impl CoxPHFit {
    pub(crate) fn restore_retained_predictors(&mut self, retained: RiskPredictors) {
        self.linear_predictors = retained.public_values.clone();
        self.risk_predictors = Some(retained);
    }

    fn saved_risk_predictors(&self) -> Option<&RiskPredictors> {
        self.risk_predictors.as_ref().filter(|saved| {
            saved.values.len() == self.n
                && saved.scales.len() == self.n
                && saved.origins.len() == self.sorted.nstrata()
                && saved.public_origins.len() == self.sorted.nstrata()
                && saved.public_values.len() == self.linear_predictors.len()
                && saved
                    .public_values
                    .iter()
                    .zip(&self.linear_predictors)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
        })
    }

    pub(crate) fn diagnostic_predictors(&self) -> &[f64] {
        self.saved_risk_predictors()
            .map_or(&self.linear_predictors, |saved| &saved.values)
    }
    pub(crate) fn diagnostic_risk_scales(&self) -> Option<&[LogScale]> {
        self.saved_risk_predictors()
            .map(|saved| saved.scales.as_slice())
    }
    pub(crate) fn diagnostic_origin_scales(&self) -> Option<&[LogScale]> {
        self.saved_risk_predictors()
            .map(|saved| saved.public_origins.as_slice())
    }
    pub(super) fn curve_origins(&self, offset_mean: f64) -> Vec<PredictorOrigin> {
        self.saved_risk_predictors().map_or_else(
            || vec![PredictorOrigin::reference(offset_mean); self.sorted.nstrata()],
            |saved| saved.origins.clone(),
        )
    }
    pub(super) fn has_curve_origins(&self) -> bool {
        self.saved_risk_predictors().is_some()
    }
    pub(super) fn training_risk_scales(&self, origins: &[PredictorOrigin]) -> Vec<LogScale> {
        self.saved_risk_predictors().map_or_else(
            || {
                self.linear_predictors
                    .iter()
                    .enumerate()
                    .map(|(i, &lp)| {
                        add_predictor_parts(
                            LogScale::from_log(-origins[self.sorted.stratum_index[i]].offset),
                            LogScale::from_log(lp),
                        )
                    })
                    .collect()
            },
            |saved| saved.scales.clone(),
        )
    }

    pub(super) fn training_prediction_risks(&self, origins: &[PredictorOrigin]) -> PredictionRisks {
        self.saved_risk_predictors().map_or_else(
            || PredictionRisks::Scalar(self.training_log_risks_at(origins)),
            |saved| PredictionRisks::Scaled(saved.scales.clone()),
        )
    }
    pub(super) fn training_log_risks_at(&self, origins: &[PredictorOrigin]) -> Vec<f64> {
        self.saved_risk_predictors().map_or_else(
            || {
                self.linear_predictors
                    .iter()
                    .enumerate()
                    .map(|(i, &lp)| lp - origins[self.sorted.stratum_index[i]].offset)
                    .collect()
            },
            |saved| saved.values.clone(),
        )
    }
}

/// Keep the usual summation order, rescaling only if the sum overflows.
pub(super) fn stable_mean(values: &[f64]) -> f64 {
    let sum = values.iter().sum::<f64>();
    if sum.is_finite() || values.is_empty() {
        return sum / values.len() as f64;
    }
    let scale = values.iter().map(|value| value.abs()).fold(0.0, f64::max);
    let fraction = values.iter().map(|value| value / scale).sum::<f64>() / values.len() as f64;
    fraction.clamp(-1.0, 1.0) * scale
}

/// An overflow-safe weighted centre for finite values and positive weights.
/// Ordinary inputs retain the original products and addition order.
pub(super) fn stable_weighted_mean(values: &[f64], weights: &[f64]) -> f64 {
    let total: f64 = weights.iter().sum();
    let numerator: f64 = values
        .iter()
        .zip(weights)
        .map(|(value, weight)| value * weight)
        .sum();
    if (total.is_finite() && numerator.is_finite()) || total == 0.0 {
        return numerator / total;
    }
    let value_scale = values.iter().map(|value| value.abs()).fold(0.0, f64::max);
    if value_scale == 0.0 {
        return 0.0;
    }
    let weight_scale = weights.iter().copied().fold(0.0, f64::max);
    let denominator: f64 = weights.iter().map(|weight| weight / weight_scale).sum();
    let numerator: f64 = values
        .iter()
        .zip(weights)
        .map(|(value, weight)| (value / value_scale) * (weight / weight_scale))
        .sum();
    (numerator / denominator).clamp(-1.0, 1.0) * value_scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_and_mixed_large_centres_remain_representable() {
        assert_eq!(stable_mean(&[1e308; 4]), 1e308);
        assert_eq!(stable_mean(&[-1e308; 4]), -1e308);
        assert_eq!(stable_mean(&[1e308, 1e308, -1e308, -1e308]), 0.0);
        assert_eq!(stable_weighted_mean(&[1e308; 4], &[1.0; 4]), 1e308);
        assert_eq!(stable_weighted_mean(&[1e308, -1e308], &[2.0, 2.0]), 0.0);
        assert_eq!(stable_weighted_mean(&[0.0, 1.0], &[1e308; 2]), 0.5);
        assert_eq!(stable_weighted_mean(&[0.0, 0.0], &[1e308; 2]), 0.0);
        let weighted = stable_weighted_mean(&[1e308, 0.0], &[3.0, 1.0]);
        assert!((weighted / 1e308 - 0.75).abs() < 1e-15);
    }

    #[test]
    fn ordinary_centres_keep_their_exact_arithmetic() {
        let values = [0.1, -4.3, 0.7, 2.8];
        let weights = [0.3, 2.1, 1.5, 0.4];
        let expected = values.iter().sum::<f64>() / values.len() as f64;
        assert_eq!(stable_mean(&values).to_bits(), expected.to_bits());
        let expected = values.iter().zip(weights).map(|(v, w)| v * w).sum::<f64>()
            / weights.iter().sum::<f64>();
        assert_eq!(
            stable_weighted_mean(&values, &weights).to_bits(),
            expected.to_bits()
        );
    }
}
