//! Shared custom-density fixtures for the two AFT optimizers.
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::dist::{dnorm, pnorm};
use crate::regression::survreg_density::SurvregDensitySource;
use crate::regression::survreg_distributions::SurvregDensity;
use crate::regression::survregc1::SurvregKernel;
use ndarray::Array2;
use serde_json::Value;
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};

pub(crate) fn mixture(z: f64) -> SurvregDensity {
    let v = (z - 1.4) / 0.8;
    let a = 0.65 * dnorm(z, false);
    let b = 0.35 * dnorm(v, false) / 0.8;
    let pdf = a + b;
    SurvregDensity {
        cdf: 0.65 * pnorm(z, true, false) + 0.35 * pnorm(v, true, false),
        survival: 0.65 * pnorm(z, false, false) + 0.35 * pnorm(v, false, false),
        pdf,
        score: (-z * a - v * b / 0.8) / pdf,
        curvature: ((z * z - 1.0) * a + (v * v - 1.0) * b / 0.64) / pdf,
    }
}

#[derive(Default)]
pub(crate) struct Mixture {
    pub(crate) calls: Mutex<Vec<Vec<f64>>>,
}

impl SurvregDensitySource for Mixture {
    fn density_batch(&self, z: &[f64]) -> SurvivalResult<Vec<SurvregDensity>> {
        self.calls.lock().unwrap().push(z.to_vec());
        Ok(z.iter().map(|&z| mixture(z)).collect())
    }
}

pub(crate) struct Data {
    pub(crate) y1: Vec<f64>,
    pub(crate) y2: Vec<f64>,
    pub(crate) status: Vec<i32>,
    pub(crate) weights: Vec<f64>,
    pub(crate) offset: Vec<f64>,
    pub(crate) strata: Vec<usize>,
    pub(crate) x: Array2<f64>,
}

pub(crate) fn reference() -> Value {
    serde_json::from_str(include_str!(
        "../../../python/tests/fixtures/survreg_density_reference.json"
    ))
    .unwrap()
}

pub(crate) fn numbers(value: &Value) -> Vec<f64> {
    serde_json::from_value(value.clone()).unwrap()
}

impl Data {
    pub(crate) fn reference() -> Self {
        let reference = reference();
        let d = &reference["data"];
        let x = numbers(&d["x"]);
        Self {
            y1: numbers(&d["y1"]),
            y2: numbers(&d["y2"]),
            status: serde_json::from_value(d["status"].clone()).unwrap(),
            weights: numbers(&d["weights"]),
            offset: numbers(&d["offset"]),
            strata: serde_json::from_value(d["g"].clone()).unwrap(),
            x: Array2::from_shape_fn((x.len(), 2), |(i, j)| if j == 0 { 1.0 } else { x[i] }),
        }
    }

    pub(crate) fn kernel<'a>(
        &'a self,
        source: &'a dyn SurvregDensitySource,
        nstrat: usize,
    ) -> SurvregKernel<'a> {
        SurvregKernel {
            y1: &self.y1,
            y2: &self.y2,
            status: &self.status,
            covariates: self.x.view(),
            weights: &self.weights,
            offset: &self.offset,
            strata: &self.strata,
            nstrat,
            distribution: source,
        }
    }
}

pub(crate) struct Failure {
    pub(crate) calls: AtomicUsize,
    pub(crate) after: usize,
}
impl SurvregDensitySource for Failure {
    fn density_batch(&self, z: &[f64]) -> SurvivalResult<Vec<SurvregDensity>> {
        if self.calls.fetch_add(1, Ordering::Relaxed) >= self.after {
            Err(SurvivalError::computation("custom density stopped"))
        } else {
            Ok(z.iter().map(|&z| mixture(z)).collect())
        }
    }
}
