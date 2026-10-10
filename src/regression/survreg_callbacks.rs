//! Runtime-defined location-scale distributions and response transforms.
//!
//! Callbacks operate on complete batches and return crate errors. They own
//! their state, so fits can retain them for predictions and residuals without
//! retaining a language runtime lock.

use crate::error::{SurvivalError, SurvivalResult};
use crate::regression::survreg_distributions::SurvregDensity;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::any::Any;
use std::fmt;
use std::sync::Arc;

mod distribution;
mod dpqr;
pub use dpqr::DpqrWarning;
pub(crate) use dpqr::QueryParms;
#[cfg(feature = "python")]
pub(crate) mod python;
#[cfg(test)]
mod tests;

/// The `init`, `density`, `quantile`, `deviance` and optional `variance`
/// functions of an R `survreg.distributions` entry. `parms` are the fitted
/// distribution's parameters. Fitting requires returned vectors to match the
/// input length; distribution queries apply R's arithmetic recycling rules.
pub trait SurvregCallbacks: Any + Send + Sync {
    /// Initial location and variance (not standard deviation).
    fn init(&self, y: &[f64], weights: &[f64], parms: &[f64]) -> SurvivalResult<[f64; 2]>;
    /// Standardized CDF, survival, density, f'/f and f''/f at each endpoint.
    fn density(&self, z: &[f64], parms: &[f64]) -> SurvivalResult<Vec<SurvregDensity>>;
    /// Standardized quantiles, including infinite endpoints for p=0 or 1.
    fn quantile(&self, p: &[f64], parms: &[f64]) -> SurvivalResult<Vec<f64>>;
    /// Saturated centers and log likelihoods on the transformed response
    /// scale. Status is 0 right, 1 exact, 2 left, 3 interval; y2 is read only
    /// for intervals. Each observation has its own scale.
    fn deviance(
        &self,
        y1: &[f64],
        y2: &[f64],
        status: &[i32],
        scale: &[f64],
        parms: &[f64],
    ) -> SurvivalResult<(Vec<f64>, Vec<f64>)>;
    /// Effective-sample-size variance used during penalized initialization.
    /// The default is the standardized variance. R adapters can reproduce
    /// `sd$variance(scale_squared)` without changing the ordinary variance.
    fn fitting_variance(&self, _scale_squared: f64, parms: &[f64]) -> SurvivalResult<f64> {
        let value = self.variance(parms)?;
        if !value.is_finite() || value <= 0.0 {
            return Err(SurvivalError::invalid_input(
                "variance callback must return a finite positive value",
            ));
        }
        Ok(value)
    }

    /// Variance of the standardized distribution, used by penalized fits.
    fn variance(&self, _parms: &[f64]) -> SurvivalResult<f64> {
        Err(SurvivalError::invalid_input(
            "custom distribution has no variance callback",
        ))
    }
}

/// An increasing response transform and its derivative and inverse.
/// Fitting checks that transformed responses and the exact-event Jacobian
/// are finite. Prediction permits infinite quantile endpoints.
pub trait SurvregTransformCallbacks: Any + Send + Sync {
    fn transform(&self, y: &[f64]) -> SurvivalResult<Vec<f64>>;
    fn derivative(&self, y: &[f64]) -> SurvivalResult<Vec<f64>>;
    fn inverse(&self, y: &[f64]) -> SurvivalResult<Vec<f64>>;
}

pub(crate) struct RuntimeCallback<T: ?Sized>(pub Arc<T>);

impl<T: ?Sized> Clone for RuntimeCallback<T> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<T: ?Sized> PartialEq for RuntimeCallback<T> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl<T: ?Sized> fmt::Debug for RuntimeCallback<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("RuntimeCallback(..)")
    }
}

/// Raw serde must never silently discard a callable. Python pickle encodes
/// the callables separately and serializes only an explicitly detached copy.
pub(crate) mod callback_serde {
    use super::*;

    pub fn serialize<T: ?Sized, S: Serializer>(
        callback: &Option<RuntimeCallback<T>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        if callback.is_some() {
            Err(serde::ser::Error::custom(
                "runtime distribution callbacks cannot be serialized with serde",
            ))
        } else {
            Option::<()>::None.serialize(serializer)
        }
    }

    pub fn deserialize<'de, T: ?Sized, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<RuntimeCallback<T>>, D::Error> {
        match Option::<()>::deserialize(deserializer)? {
            None => Ok(None),
            Some(()) => Err(serde::de::Error::custom("invalid runtime callback state")),
        }
    }
}

pub(crate) fn check_batch_len(actual: usize, expected: usize, name: &str) -> SurvivalResult<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(SurvivalError::invalid_input(format!(
            "{name} callback returned {actual} values; expected {expected}"
        )))
    }
}
