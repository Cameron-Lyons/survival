//! Density evaluation shared by the ordinary and penalized AFT likelihoods.
//!
//! A callback receives every standardized lower endpoint, followed by the
//! upper endpoints of interval-censored rows in input order. This is the
//! batching contract of R's `survregc2`, without any language-runtime types.

use crate::error::{SurvivalError, SurvivalResult};
use crate::regression::survreg_distributions::{SurvregDensity, SurvregDistribution};

#[cfg(test)]
pub(crate) mod test_support;

/// The density part of a location-scale distribution. Initialization,
/// transforms, quantiles and deviance are handled outside the likelihood.
pub(crate) trait SurvregDensitySource: Send + Sync {
    fn density_batch(&self, z: &[f64]) -> SurvivalResult<Vec<SurvregDensity>>;

    /// Built-ins retain the scalar C-kernel formulas and avoid a batch buffer.
    fn builtin(&self) -> Option<&SurvregDistribution> {
        None
    }
}

impl SurvregDensitySource for SurvregDistribution {
    fn density_batch(&self, z: &[f64]) -> SurvivalResult<Vec<SurvregDensity>> {
        if self.family == super::survreg_distributions::SurvregFamily::Custom {
            // The likelihood checks this batch before accumulation. Avoid
            // validating it twice through the public distribution API.
            self.custom_callbacks()?.density(z, &self.parms)
        } else {
            self.density_batch(z)
        }
    }

    fn builtin(&self) -> Option<&SurvregDistribution> {
        (self.callbacks.is_none()
            && self.family != super::survreg_distributions::SurvregFamily::Custom)
            .then_some(self)
    }
}

/// Check callback results before any row indexing or derivative arithmetic.
pub(crate) fn check_density_batch(
    values: &[SurvregDensity],
    expected: usize,
) -> SurvivalResult<()> {
    if values.len() != expected {
        return Err(SurvivalError::invalid_input(format!(
            "density callback returned {} rows; expected {expected}",
            values.len()
        )));
    }
    for (i, d) in values.iter().enumerate() {
        if !d.cdf.is_finite()
            || !(0.0..=1.0).contains(&d.cdf)
            || !d.survival.is_finite()
            || !(0.0..=1.0).contains(&d.survival)
            || !d.pdf.is_finite()
            || d.pdf < 0.0
            || (d.pdf > 0.0 && (!d.score.is_finite() || !d.curvature.is_finite()))
        {
            return Err(SurvivalError::invalid_input(format!(
                "density callback returned invalid probabilities or derivatives at row {i}"
            )));
        }
    }
    Ok(())
}

impl SurvregDensity {
    pub(crate) fn density_kernel(self) -> [f64; 4] {
        [0.0, self.pdf, self.score, self.curvature]
    }

    pub(crate) fn distribution_kernel(self) -> [f64; 4] {
        // A density that underflows to zero may have undefined ratios f'/f
        // and f''/f. Its zero derivative still gives valid censored tails.
        let derivative = if self.pdf == 0.0 {
            0.0
        } else {
            self.pdf * self.score
        };
        [self.cdf, self.survival, self.pdf, derivative]
    }
}
