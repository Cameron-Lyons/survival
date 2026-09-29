//! Compile and exercise the callback contract as an external crate consumer.
use std::sync::Arc;
use survival::SurvivalResult;
use survival::regression::{
    SurvregCallbacks, SurvregDensity, SurvregDistribution, SurvregTransform,
    SurvregTransformCallbacks,
};

struct Logistic;
impl SurvregCallbacks for Logistic {
    fn init(&self, y: &[f64], weights: &[f64], _: &[f64]) -> SurvivalResult<[f64; 2]> {
        let sum: f64 = weights.iter().sum();
        let mean = y.iter().zip(weights).map(|(y, w)| y * w).sum::<f64>() / sum;
        let variance = y
            .iter()
            .zip(weights)
            .map(|(y, w)| w * (y - mean).powi(2))
            .sum::<f64>()
            / sum;
        Ok([mean, variance])
    }

    fn density(&self, z: &[f64], _: &[f64]) -> SurvivalResult<Vec<SurvregDensity>> {
        Ok(z.iter()
            .map(|&z| {
                let e = (-z.abs()).exp();
                let (cdf, survival) = if z > 0.0 {
                    (1.0 / (1.0 + e), e / (1.0 + e))
                } else {
                    (e / (1.0 + e), 1.0 / (1.0 + e))
                };
                let pdf = cdf * survival;
                SurvregDensity {
                    cdf,
                    survival,
                    pdf,
                    score: 1.0 - 2.0 * cdf,
                    curvature: 1.0 - 6.0 * pdf,
                }
            })
            .collect())
    }

    fn quantile(&self, p: &[f64], _: &[f64]) -> SurvivalResult<Vec<f64>> {
        Ok(p.iter().map(|&p| p.ln() - (-p).ln_1p()).collect())
    }

    fn deviance(
        &self,
        y1: &[f64],
        y2: &[f64],
        status: &[i32],
        scale: &[f64],
        _: &[f64],
    ) -> SurvivalResult<(Vec<f64>, Vec<f64>)> {
        Ok((0..y1.len())
            .map(|i| match status[i] {
                1 => (y1[i], -(4.0 * scale[i]).ln()),
                3 => (
                    (y1[i] + y2[i]) / 2.0,
                    ((y2[i] - y1[i]) / (4.0 * scale[i])).tanh().ln(),
                ),
                _ => (y1[i], 0.0),
            })
            .unzip())
    }

    fn variance(&self, _: &[f64]) -> SurvivalResult<f64> {
        Ok(std::f64::consts::PI.powi(2) / 3.0)
    }
}

struct Asinh;
impl SurvregTransformCallbacks for Asinh {
    fn transform(&self, y: &[f64]) -> SurvivalResult<Vec<f64>> {
        Ok(y.iter().map(|y| y.asinh()).collect())
    }
    fn derivative(&self, y: &[f64]) -> SurvivalResult<Vec<f64>> {
        Ok(y.iter().map(|y| 1.0 / (1.0 + y * y).sqrt()).collect())
    }
    fn inverse(&self, y: &[f64]) -> SurvivalResult<Vec<f64>> {
        Ok(y.iter().map(|y| y.sinh()).collect())
    }
}

#[test]
fn external_rust_callbacks_are_constructible_and_callable() -> SurvivalResult<()> {
    let distribution = SurvregDistribution::from_callbacks(
        "User logistic",
        Arc::new(Logistic),
        SurvregTransform::Identity,
        None,
        vec![],
    )?;
    assert!(distribution.dtest().is_empty());
    let density: SurvregDensity = distribution.density(0.0)?;
    assert_eq!(density.pdf, 0.25);
    assert_eq!(
        distribution.deviance(1.0, 0.0, 1, 0.5)?,
        (1.0, -2.0f64.ln())
    );
    assert_eq!(distribution.quantile(0.5)?, 0.0);
    let distribution = distribution.with_transform(Arc::new(Asinh));
    assert!(distribution.dtest().is_empty());
    assert_eq!(distribution.quantile_at(0.5, 1.0, 0.7)?, 1.0f64.sinh());
    Ok(())
}
