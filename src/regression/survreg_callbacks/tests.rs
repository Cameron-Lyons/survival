//! Public native callbacks, including postfit operations, against R references.
use super::*;
use crate::regression::parametric_survival::{SurvregControl, SurvregData, survreg_fit};
use crate::regression::survreg_density::{SurvregDensitySource, test_support::*};
use crate::regression::survreg_distributions::{SurvregDistribution, SurvregTransform};
use crate::regression::survreg_predict::SurvregPredictType;
use crate::residuals::survreg_resid::SurvregResidType;
use serde_json::Value;

fn root(mut lo: f64, mut hi: f64, f: impl Fn(f64) -> f64) -> f64 {
    let negative = f(lo) < 0.0;
    for _ in 0..65 {
        let mid = (lo + hi) / 2.0;
        if (f(mid) < 0.0) == negative {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}

impl SurvregCallbacks for Mixture {
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
        SurvregDensitySource::density_batch(self, z)
    }
    fn quantile(&self, p: &[f64], _: &[f64]) -> SurvivalResult<Vec<f64>> {
        Ok(p.iter()
            .map(|&p| root(-20.0, 20.0, |z| mixture(z).cdf - p))
            .collect())
    }
    fn deviance(
        &self,
        y1: &[f64],
        y2: &[f64],
        status: &[i32],
        scale: &[f64],
        _: &[f64],
    ) -> SurvivalResult<(Vec<f64>, Vec<f64>)> {
        let mode = root(-2.0, 3.0, |z| mixture(z).score);
        let mut center = y1.to_vec();
        let mut loglik = vec![0.0; y1.len()];
        for i in 0..y1.len() {
            if status[i] == 1 {
                center[i] -= scale[i] * mode;
                loglik[i] = mixture(mode).pdf.ln() - scale[i].ln();
            } else if status[i] == 3 {
                center[i] = root(y1[i] - 8.0 * scale[i], y2[i] + 8.0 * scale[i], |eta| {
                    mixture((y1[i] - eta) / scale[i]).pdf - mixture((y2[i] - eta) / scale[i]).pdf
                });
                loglik[i] = (mixture((y2[i] - center[i]) / scale[i]).cdf
                    - mixture((y1[i] - center[i]) / scale[i]).cdf)
                    .ln();
            }
        }
        Ok((center, loglik))
    }
    fn variance(&self, _: &[f64]) -> SurvivalResult<f64> {
        Ok(1.3199)
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

fn distribution() -> SurvregDistribution {
    SurvregDistribution::from_callbacks(
        "mixture",
        Arc::new(Mixture::default()),
        SurvregTransform::Identity,
        None,
        vec![],
    )
    .unwrap()
}
fn data(case: &Value, transformed: bool, robust: bool) -> SurvregData {
    let mut d = Data::reference();
    if transformed {
        d.y1.iter_mut().for_each(|y| *y = y.sinh());
        d.y2.iter_mut().for_each(|y| *y = y.sinh());
    }
    if robust {
        d.status.iter_mut().for_each(|s| {
            if *s == 3 {
                *s = 1
            }
        });
    }
    SurvregData::try_new(
        d.y1,
        d.status,
        d.x,
        Some(d.y2),
        Some(d.weights),
        Some(d.offset),
        (case["nstrat"] == 2).then_some(d.strata),
        robust.then(|| (0..40).map(|i| i % 10).collect()),
    )
    .unwrap()
}
fn close(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} != {b}");
    }
}
fn matrix(value: &Value) -> Vec<Vec<f64>> {
    serde_json::from_value(value.clone()).unwrap()
}

#[test]
fn public_callbacks_fit_predict_and_compute_residuals() {
    for case in reference()["cases"].as_array().unwrap() {
        for transformed in [false, true] {
            for initialized in [false, true] {
                let d = data(case, transformed, false);
                let dist = if transformed {
                    distribution().with_transform(Arc::new(Asinh))
                } else {
                    distribution()
                };
                assert!(dist.dtest().is_empty());
                let nstrat = case["nstrat"].as_u64().unwrap() as usize;
                let start = numbers(&case["init"]);
                let control = SurvregControl {
                    iter_max: 50,
                    rel_tolerance: 1e-11,
                    ..Default::default()
                };
                let fit = survreg_fit(
                    &d,
                    &dist,
                    initialized.then_some(&start[..2 + nstrat]),
                    if nstrat == 0 { 0.9 } else { 0.0 },
                    &control,
                    false,
                )
                .unwrap();
                assert!(fit.converged);
                close(
                    &fit.coefficients,
                    &numbers(&case["expected"]["parameters"])[..2 + nstrat],
                    2e-7,
                );
                let postfit = if transformed {
                    &case["transformed"]["postfit"]
                } else {
                    &case["postfit"]
                };
                for (kind, key) in [
                    (SurvregPredictType::Response, "response"),
                    (SurvregPredictType::Quantile, "quantile"),
                ] {
                    let pred = fit
                        .predict(None, kind, true, &[0.1, 0.5, 0.9], None, None)
                        .unwrap();
                    for (actual, field) in [
                        (&pred.fit, "fit"),
                        (pred.se_fit.as_ref().unwrap(), "se.fit"),
                    ] {
                        let expected = if kind == SurvregPredictType::Response {
                            numbers(&postfit[key][field])
                                .into_iter()
                                .map(|x| vec![x])
                                .collect()
                        } else {
                            matrix(&postfit[key][field])
                        };
                        assert_eq!(actual.len(), expected.len());
                        for (a, b) in actual.iter().zip(&expected) {
                            close(a, b, 2e-7);
                        }
                    }
                }
                for (kind, key) in [
                    (SurvregResidType::Response, "response"),
                    (SurvregResidType::Deviance, "deviance"),
                    (SurvregResidType::Working, "working"),
                ] {
                    let actual = fit.residuals(kind, true, None, false).unwrap();
                    close(
                        &actual.values.iter().map(|row| row[0]).collect::<Vec<_>>(),
                        &numbers(&postfit["residuals"][key]),
                        2e-7,
                    );
                }
                let actual = fit
                    .residuals(SurvregResidType::Matrix, true, None, false)
                    .unwrap();
                for (i, (a, b)) in actual
                    .values
                    .iter()
                    .zip(matrix(&postfit["residuals"]["matrix"]))
                    .enumerate()
                {
                    let n = if d.status[i] == 3 { 3 } else { 6 };
                    close(&a[..n], &b[..n], 2e-7);
                }
            }
        }
    }
}

#[test]
fn public_callback_robust_variance_matches_r() {
    for case in reference()["cases"].as_array().unwrap() {
        let d = data(case, false, true);
        let nstrat = case["nstrat"].as_u64().unwrap() as usize;
        let start = numbers(&case["init"]);
        let fit = survreg_fit(
            &d,
            &distribution(),
            Some(&start[..2 + nstrat]),
            if nstrat == 0 { 0.9 } else { 0.0 },
            &SurvregControl {
                iter_max: 50,
                rel_tolerance: 1e-11,
                ..Default::default()
            },
            true,
        )
        .unwrap();
        let reference = &case["robust_no_interval"];
        close(
            &fit.coefficients,
            &numbers(&reference["parameters"])[..2 + nstrat],
            1e-8,
        );
        for (row, expected) in fit
            .variance_matrix
            .iter()
            .zip(matrix(&reference["variance"]))
        {
            close(row, &expected, 1e-8);
        }
    }
}

#[test]
fn callbacks_survive_clone_and_cannot_be_silently_serialized() {
    let d = distribution().with_transform(Arc::new(Asinh));
    let cloned = d.clone();
    assert_eq!(d, cloned);
    assert_eq!(
        d.quantile_values(&[0.1, 0.5, 0.9], &[0.4], &[0.9]).unwrap(),
        cloned
            .quantile_values(&[0.1, 0.5, 0.9], &[0.4], &[0.9])
            .unwrap()
    );
    assert!(
        serde_json::to_string(&d)
            .unwrap_err()
            .to_string()
            .contains("runtime distribution callbacks")
    );
    let builtin = SurvregDistribution::from_name("weibull", None).unwrap();
    let restored: SurvregDistribution =
        serde_json::from_str(&serde_json::to_string(&builtin).unwrap()).unwrap();
    assert_eq!(builtin, restored);
}

#[test]
fn native_penalized_callback_fit_matches_r() {
    use crate::regression::penalized::{ModelTerm, PenaltyTerm};
    use crate::regression::survpenal::{SurvpenalData, SurvpenalFit, SurvpenalOptions};
    for case in reference()["cases"].as_array().unwrap() {
        for intervals in [false, true] {
            let mut d = data(case, false, false);
            if !intervals {
                d.status.iter_mut().for_each(|s| {
                    if *s == 3 {
                        *s = 1
                    }
                });
            }
            let d = SurvpenalData::try_new(
                d,
                vec![
                    ModelTerm {
                        columns: vec![0],
                        penalty: None,
                    },
                    ModelTerm {
                        columns: vec![1],
                        penalty: Some(
                            PenaltyTerm::ridge(Some(0.7), None, 0.1, false, None).unwrap(),
                        ),
                    },
                ],
            )
            .unwrap();
            let nstrat = case["nstrat"].as_u64().unwrap() as usize;
            let start = numbers(&case["init"]);
            let fit = SurvpenalFit::fit(
                &d,
                &distribution(),
                &SurvpenalOptions {
                    init: Some(start[..2 + nstrat].to_vec()),
                    scale: if nstrat == 0 { 0.9 } else { 0.0 },
                    control: SurvregControl {
                        iter_max: 50,
                        rel_tolerance: 1e-11,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            )
            .unwrap();
            let expected = &case[if intervals {
                "penalized_interval"
            } else {
                "penalized_no_interval"
            }];
            close(
                &fit.survreg.coefficients,
                &numbers(&expected["parameters"])[..2 + nstrat],
                if intervals { 2e-6 } else { 1e-8 },
            );
            close(
                &[fit.survreg.log_likelihood - fit.penalty[1]],
                &[expected["loglik"].as_f64().unwrap()],
                1e-10,
            );
        }
    }
}
