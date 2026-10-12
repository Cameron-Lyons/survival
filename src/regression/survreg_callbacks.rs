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

pub trait SurvregCallbacks: Any + Send + Sync {
    fn init(&self, y: &[f64], weights: &[f64], parms: &[f64]) -> SurvivalResult<[f64; 2]>;
    fn density(&self, z: &[f64], parms: &[f64]) -> SurvivalResult<Vec<SurvregDensity>>;
    fn quantile(&self, p: &[f64], parms: &[f64]) -> SurvivalResult<Vec<f64>>;
    fn deviance(
        &self,
        y1: &[f64],
        y2: &[f64],
        status: &[i32],
        scale: &[f64],
        parms: &[f64],
    ) -> SurvivalResult<(Vec<f64>, Vec<f64>)>;
    fn fitting_variance(&self, _scale_squared: f64, parms: &[f64]) -> SurvivalResult<f64> {
        let value = self.variance(parms)?;
        if !value.is_finite() || value <= 0.0 {
            return Err(SurvivalError::invalid_input(
                "variance callback must return a finite positive value",
            ));
        }
        Ok(value)
    }

    fn variance(&self, _parms: &[f64]) -> SurvivalResult<f64> {
        Err(SurvivalError::invalid_input(
            "custom distribution has no variance callback",
        ))
    }
}

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
