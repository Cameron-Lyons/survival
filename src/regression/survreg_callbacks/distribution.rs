use super::*;
use crate::regression::survreg_density::check_density_batch;
use crate::regression::survreg_distributions::{
    SurvregDistribution, SurvregFamily, SurvregTransform,
};

impl SurvregDistribution {
    /// Define a location-scale family in native Rust. Fits retain an `Arc`
    /// to these callbacks. Runtime callables have no automatic serde form.
    pub fn from_callbacks(
        name: impl Into<String>,
        callbacks: Arc<dyn SurvregCallbacks>,
        transform: SurvregTransform,
        scale: Option<f64>,
        parms: Vec<f64>,
    ) -> SurvivalResult<Self> {
        let distribution = Self {
            name: name.into(),
            family: SurvregFamily::Custom,
            transform,
            scale,
            parms,
            query_parms: QueryParms::Values,
            callbacks: Some(RuntimeCallback(callbacks)),
            transform_callbacks: None,
        };
        distribution.validate()?;
        Ok(distribution)
    }

    /// Replace a response transform, retaining the base family and parameters.
    pub fn with_transform(mut self, callbacks: Arc<dyn SurvregTransformCallbacks>) -> Self {
        self.transform = SurvregTransform::Custom;
        self.transform_callbacks = Some(RuntimeCallback(callbacks));
        self
    }

    /// Clone a distribution with new fitted parameters, retaining callbacks.
    pub fn with_parms(&self, parms: Vec<f64>) -> SurvivalResult<Self> {
        if self.callbacks.is_some() && parms.len() != self.parms.len() {
            return Err(SurvivalError::invalid_input(
                "wrong number of distribution parameters",
            ));
        }
        let mut distribution = self.clone();
        distribution.parms = parms;
        distribution.validate()?;
        Ok(distribution)
    }

    /// Derive a named distribution using the same base family and callbacks.
    pub fn derived(
        &self,
        name: impl Into<String>,
        transform: SurvregTransform,
        scale: Option<f64>,
    ) -> SurvivalResult<Self> {
        let mut result = self.clone();
        result.name = name.into();
        result.transform = transform;
        result.scale = scale;
        if transform != SurvregTransform::Custom {
            result.transform_callbacks = None;
        }
        result.validate()?;
        Ok(result)
    }

    pub(crate) fn custom_callbacks(&self) -> SurvivalResult<&dyn SurvregCallbacks> {
        self.callbacks
            .as_ref()
            .map(|c| c.0.as_ref())
            .ok_or_else(|| SurvivalError::invalid_input("custom distribution has no callbacks"))
    }

    /// Density summaries for a whole batch of standardized endpoints.
    pub fn density_batch(&self, z: &[f64]) -> SurvivalResult<Vec<SurvregDensity>> {
        if self.family == SurvregFamily::Custom || self.callbacks.is_some() {
            let values = self.custom_callbacks()?.density(z, &self.parms)?;
            check_density_batch(&values, z.len())?;
            Ok(values)
        } else {
            Ok(z.iter().map(|&z| self.builtin_density(z)).collect())
        }
    }

    /// Density summary at one standardized endpoint. Built-ins allocate no batch.
    pub fn density(&self, z: f64) -> SurvivalResult<SurvregDensity> {
        if self.family == SurvregFamily::Custom || self.callbacks.is_some() {
            Ok(self.density_batch(&[z])?[0])
        } else {
            Ok(self.builtin_density(z))
        }
    }

    /// Standardized quantiles evaluated in one callback invocation.
    pub fn quantiles(&self, p: &[f64]) -> SurvivalResult<Vec<f64>> {
        if self.family == SurvregFamily::Custom || self.callbacks.is_some() {
            let values = self.custom_callbacks()?.quantile(p, &self.parms)?;
            check_batch_len(values.len(), p.len(), "quantile")?;
            Ok(values)
        } else {
            Ok(p.iter().map(|&p| self.builtin_quantile(p)).collect())
        }
    }

    /// Standardized quantile at a single probability.
    pub fn quantile(&self, p: f64) -> SurvivalResult<f64> {
        if self.family == SurvregFamily::Custom || self.callbacks.is_some() {
            Ok(self.quantiles(&[p])?[0])
        } else {
            Ok(self.builtin_quantile(p))
        }
    }

    /// Saturated centers and log likelihoods for a complete response batch.
    pub fn deviance_batch(
        &self,
        y1: &[f64],
        y2: &[f64],
        status: &[i32],
        scale: &[f64],
    ) -> SurvivalResult<(Vec<f64>, Vec<f64>)> {
        crate::internal::validation::validate_equal_len(&[
            ("y1", y1.len()),
            ("y2", y2.len()),
            ("status", status.len()),
            ("scale", scale.len()),
        ])?;
        if self.family == SurvregFamily::Custom || self.callbacks.is_some() {
            let values = self
                .custom_callbacks()?
                .deviance(y1, y2, status, scale, &self.parms)?;
            check_batch_len(values.0.len(), y1.len(), "deviance center")?;
            check_batch_len(values.1.len(), y1.len(), "deviance loglik")?;
            Ok(values)
        } else {
            Ok((0..y1.len())
                .map(|i| self.builtin_deviance(y1[i], y2[i], status[i], scale[i]))
                .unzip())
        }
    }

    /// Saturated center and log likelihood for one observation.
    pub fn deviance(
        &self,
        y1: f64,
        y2: f64,
        status: i32,
        scale: f64,
    ) -> SurvivalResult<(f64, f64)> {
        if self.callbacks.is_some() || self.family == SurvregFamily::Custom {
            let (centers, loglik) = self.deviance_batch(&[y1], &[y2], &[status], &[scale])?;
            Ok((centers[0], loglik[0]))
        } else {
            Ok(self.builtin_deviance(y1, y2, status, scale))
        }
    }

    /// Forward response transform, evaluated as one batch.
    pub fn transform_values(&self, values: &[f64]) -> SurvivalResult<Vec<f64>> {
        self.transform_batch(
            values,
            "transform",
            |t, y| t.apply(y),
            |c, y| c.transform(y),
        )
    }

    /// Derivative of the response transform, evaluated as one batch.
    pub fn transform_derivatives(&self, values: &[f64]) -> SurvivalResult<Vec<f64>> {
        self.transform_batch(
            values,
            "transform derivative",
            |t, y| t.derivative(y),
            |c, y| c.derivative(y),
        )
    }

    /// Inverse response transform, evaluated as one batch.
    pub fn inverse_values(&self, values: &[f64]) -> SurvivalResult<Vec<f64>> {
        self.transform_batch(
            values,
            "inverse transform",
            |t, y| t.inverse(y),
            |c, y| c.inverse(y),
        )
    }

    fn transform_batch(
        &self,
        values: &[f64],
        name: &str,
        builtin: impl Fn(SurvregTransform, f64) -> SurvivalResult<f64>,
        callback: impl Fn(&dyn SurvregTransformCallbacks, &[f64]) -> SurvivalResult<Vec<f64>>,
    ) -> SurvivalResult<Vec<f64>> {
        if let Some(c) = &self.transform_callbacks {
            let result = callback(c.0.as_ref(), values)?;
            check_batch_len(result.len(), values.len(), name)?;
            Ok(result)
        } else {
            values.iter().map(|&y| builtin(self.transform, y)).collect()
        }
    }

    pub(crate) fn detached_callbacks(&self) -> Self {
        let mut result = self.clone();
        result.callbacks = None;
        result.transform_callbacks = None;
        result
    }

    pub(crate) fn transformed_endpoints(
        &self,
        time: &[f64],
        time2: Option<&[f64]>,
        status: &[i32],
    ) -> SurvivalResult<(Vec<f64>, Vec<f64>)> {
        let lower = self.transform_values(time)?;
        let mut upper = lower.clone();
        let intervals: Vec<_> = status
            .iter()
            .enumerate()
            .filter_map(|(i, &s)| (s == 3).then_some(i))
            .collect();
        if !intervals.is_empty() {
            let time2 = time2
                .ok_or_else(|| SurvivalError::invalid_input("intervals require upper endpoints"))?;
            let values: Vec<_> = intervals.iter().map(|&i| time2[i]).collect();
            let transformed = self.transform_values(&values)?;
            for (&i, value) in intervals.iter().zip(transformed) {
                upper[i] = value;
            }
        }
        Ok((lower, upper))
    }
}
