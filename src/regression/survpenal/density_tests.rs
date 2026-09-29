//! Custom-density coverage of the penalized AFT inner optimizer.
use super::kernel::{Survreg7Fit, survreg7};
use crate::error::{SurvivalError, SurvivalResult};
use crate::regression::penalized::penalty::CoxPenaltyTerms;
use crate::regression::penalized::terms::{PenaltyCallback, PenaltyShape};
use crate::regression::survreg_density::test_support::*;
use crate::regression::survregc1::SurvregKernel;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Ridge(CoxPenaltyTerms);

impl PenaltyCallback for Ridge {
    fn call(&mut self, which: i32, coef: &mut [f64]) -> SurvivalResult<&CoxPenaltyTerms> {
        assert_eq!(which, 2);
        self.0.first[1] = -0.7 * coef[1];
        self.0.second[1] = 0.7;
        self.0.penalty = -0.7 * coef[1] * coef[1] / 2.0;
        Ok(&self.0)
    }
}

fn fit(kernel: &SurvregKernel<'_>, start: Vec<f64>) -> SurvivalResult<Survreg7Fit> {
    let mut penalty = Ridge(CoxPenaltyTerms::zeros(2, 2, 2));
    survreg7(
        kernel,
        None,
        50,
        start,
        1e-11,
        1e-10,
        PenaltyShape {
            sparse: false,
            dense: true,
            full_imat: false,
        },
        &mut penalty,
    )
}

fn close(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() <= tol * expected.abs().max(1.0),
        "{actual} != {expected}"
    );
}

#[test]
fn mixture_ridge_matches_r_survreg_and_independent_interval_optimizer() {
    for case in reference()["cases"].as_array().unwrap() {
        for intervals in [false, true] {
            let mut data = Data::reference();
            if !intervals {
                for status in &mut data.status {
                    if *status == 3 {
                        *status = 1;
                    }
                }
            }
            let source = Mixture::default();
            let kernel = data.kernel(&source, case["nstrat"].as_u64().unwrap() as usize);
            let fit = fit(&kernel, numbers(&case["init"])).unwrap();
            assert!(fit.converged, "{}, intervals={intervals}", case["name"]);
            let expected = &case[if intervals {
                "penalized_interval"
            } else {
                "penalized_no_interval"
            }];
            let params = numbers(&expected["parameters"]);
            assert_eq!(fit.beta.len(), params.len());
            // stats::optim uses numerical gradients, unlike survreg's
            // analytic information. Its coefficients have lower precision.
            let tolerance = if intervals { 2e-6 } else { 1e-8 };
            for (&a, &b) in fit.beta.iter().zip(&params) {
                close(a, b, tolerance);
            }
            close(fit.loglik, expected["loglik"].as_f64().unwrap(), 1e-10);
            if !intervals {
                close(-fit.penalty, expected["penalty"].as_f64().unwrap(), 1e-10);
            }
        }
    }
}

#[test]
fn density_errors_propagate_from_every_penalized_evaluation() {
    let data = Data::reference();
    // A far start exercises the fallback and line-search evaluations as
    // well as the initial and accepted full evaluations.
    let start = vec![-2.0, 1.5, -1.0];
    let source = Mixture::default();
    let baseline = fit(&data.kernel(&source, 1), start.clone()).unwrap();
    let evaluations = source.calls.lock().unwrap().len();
    assert!(
        evaluations > baseline.iter + 1,
        "must exercise extra evaluations"
    );
    for after in 0..evaluations {
        let source = Failure {
            calls: AtomicUsize::new(0),
            after,
        };
        let result = fit(&data.kernel(&source, 1), start.clone());
        assert_eq!(
            result.err(),
            Some(SurvivalError::computation("custom density stopped")),
            "callback {after}"
        );
        assert_eq!(source.calls.load(Ordering::Relaxed), after + 1);
    }
}
