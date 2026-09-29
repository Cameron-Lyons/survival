use super::*;
use crate::error::SurvivalError;
use crate::regression::parametric_survival::survreg6;
use crate::regression::survreg_density::test_support::*;
use crate::regression::survreg_distributions::SurvregDensity;
use std::sync::atomic::{AtomicUsize, Ordering};

fn close(a: f64, b: f64, tolerance: f64) {
    assert!((a - b).abs() <= tolerance * b.abs().max(1.0), "{a} != {b}");
}

#[test]
fn custom_mixture_optimizer_matches_r_survregc2() {
    let data = Data::reference();
    for case in reference()["cases"].as_array().unwrap() {
        let source = Mixture::default();
        let kernel = data.kernel(&source, case["nstrat"].as_u64().unwrap() as usize);
        let fit = survreg6(&kernel, 50, numbers(&case["init"]), 1e-11, 1e-10).unwrap();
        assert!(fit.converged, "{}", case["name"]);
        assert_eq!(
            fit.beta.len(),
            numbers(&case["expected"]["parameters"]).len()
        );
        for (a, b) in fit
            .beta
            .iter()
            .zip(numbers(&case["expected"]["parameters"]))
        {
            close(*a, b, 1e-8);
        }
        close(
            fit.loglik,
            case["expected"]["loglik"].as_f64().unwrap(),
            1e-10,
        );
        let var: Vec<Vec<f64>> =
            serde_json::from_value(case["expected"]["variance"].clone()).unwrap();
        assert_eq!(fit.var.nrows(), var.len());
        for (i, row) in var.iter().enumerate() {
            for (j, &expected) in row.iter().enumerate() {
                close(fit.var[(i, j)], expected, 1e-8);
            }
        }
    }
}

#[test]
fn callback_evaluations_accept_column_major_designs() {
    let mut data = Data::reference();
    let source = Mixture::default();
    let beta = [0.2, -0.1, 0.3, -0.2];
    let row_major = data.kernel(&source, 2).evaluate(&beta, true).unwrap();
    data.x = data.x.t().as_standard_layout().into_owned().reversed_axes();
    assert!(!data.x.is_standard_layout());
    let column_major = data.kernel(&source, 2).evaluate(&beta, true).unwrap();
    assert_eq!(column_major.loglik, row_major.loglik);
    assert_eq!(column_major.u, row_major.u);
    assert_eq!(column_major.imat, row_major.imat);
    assert_eq!(column_major.jj, row_major.jj);
}

#[test]
fn zero_density_tails_keep_valid_one_sided_probabilities() {
    struct Tails;
    impl SurvregDensitySource for Tails {
        fn density_batch(&self, z: &[f64]) -> SurvivalResult<Vec<SurvregDensity>> {
            Ok(z.iter()
                .map(|&z| SurvregDensity {
                    cdf: if z > 0.0 { 1.0 } else { 0.0 },
                    survival: if z > 0.0 { 0.0 } else { 1.0 },
                    pdf: 0.0,
                    score: f64::NAN,
                    curvature: f64::NAN,
                })
                .collect())
        }
    }
    let x = Array2::ones((2, 1));
    let kernel = SurvregKernel {
        y1: &[-40.0, 40.0],
        y2: &[],
        status: &[0, 2],
        covariates: x.view(),
        weights: &[1.0; 2],
        offset: &[0.0; 2],
        strata: &[0; 2],
        nstrat: 1,
        distribution: &Tails,
    };
    let fit = kernel.evaluate(&[0.0, 0.0], true).unwrap();
    assert_eq!(fit.loglik, 0.0);
    assert_eq!(fit.u, [0.0, 0.0]);
    assert!(fit.imat.iter().all(|&v| v == 0.0));
    assert!(fit.jj.unwrap().iter().all(|&v| v == 0.0));
}

#[test]
fn callback_batch_orders_endpoints_and_includes_sparse_effects() {
    let data = Data::reference();
    let source = Mixture::default();
    let kernel = data.kernel(&source, 2);
    let group: Vec<_> = (0..data.y1.len()).map(|i| i % 3).collect();
    let frailty = SparseFrailty {
        group: &group,
        nf: 3,
    };
    let beta = [0.15, -0.25, 0.35, 0.2, -0.1, 0.3, -0.2];
    let mut result = BlockLikelihood::new(3, 4);
    kernel
        .evaluate_blocks(&beta, Some(&frailty), true, &mut result)
        .unwrap();
    let mut expected = Vec::new();
    let mut upper = Vec::new();
    for (i, &g) in group.iter().enumerate() {
        let eta = beta[3] + beta[4] * data.x[(i, 1)] + data.offset[i] + beta[g];
        let scale = beta[5 + data.strata[i]].exp();
        expected.push((data.y1[i] - eta) / scale);
        if data.status[i] == 3 {
            upper.push((data.y2[i] - eta) / scale);
        }
    }
    expected.extend(upper);
    {
        let calls = source.calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].len(), expected.len());
        for (&a, &b) in calls[0].iter().zip(&expected) {
            close(a, b, 1e-15);
        }
    }
    close(
        kernel.loglik_at(&beta, Some(&frailty)).unwrap(),
        result.loglik,
        1e-15,
    );
    assert_eq!(source.calls.lock().unwrap().len(), 2);
}

#[test]
fn mixture_score_and_information_match_finite_differences() {
    let data = Data::reference();
    let source = Mixture::default();
    let kernel = data.kernel(&source, 2);
    let beta = [0.2, -0.1, 0.3, -0.2];
    let base = kernel.evaluate(&beta, true).unwrap();
    assert_eq!(source.calls.lock().unwrap().len(), 1);
    let h = 1e-5;
    for i in 0..beta.len() {
        let mut plus = beta;
        let mut minus = beta;
        plus[i] += h;
        minus[i] -= h;
        let upper = kernel.evaluate(&plus, false).unwrap();
        let lower = kernel.evaluate(&minus, false).unwrap();
        close((upper.loglik - lower.loglik) / (2.0 * h), base.u[i], 1e-7);
        for j in 0..beta.len() {
            close(
                -(upper.u[j] - lower.u[j]) / (2.0 * h),
                base.imat[(i, j)],
                1e-7,
            );
        }
    }
    let mut blocks = BlockLikelihood::new(0, 4);
    kernel
        .evaluate_blocks(&beta, None, true, &mut blocks)
        .unwrap();
    close(blocks.loglik, base.loglik, 1e-15);
    assert_eq!(blocks.u, base.u);
    for i in 0..4 {
        for j in 0..=i {
            assert_eq!(blocks.hmat[(i, j)], base.imat[(i, j)]);
            assert_eq!(blocks.jj[(i, j)], base.jj.as_ref().unwrap()[(i, j)]);
        }
    }
}

#[test]
fn callback_errors_stop_every_evaluation_and_the_optimizer() {
    let data = Data::reference();
    let source = Failure {
        calls: AtomicUsize::new(0),
        after: 0,
    };
    let kernel = data.kernel(&source, 1);
    let beta = [0.2, -0.1, 0.1];
    let expected = Some(SurvivalError::computation("custom density stopped"));
    assert_eq!(kernel.evaluate(&beta, false).err(), expected);
    assert_eq!(kernel.loglik_at(&beta, None).err(), expected);
    assert_eq!(kernel.jj(&beta).err(), expected);
    let mut blocks = BlockLikelihood::new(0, 3);
    blocks.loglik = 123.0;
    assert_eq!(
        kernel.evaluate_blocks(&beta, None, true, &mut blocks).err(),
        expected
    );
    assert_eq!(blocks.loglik, 123.0);
    let source = Failure {
        calls: AtomicUsize::new(0),
        after: 1,
    };
    let kernel = data.kernel(&source, 1);
    assert_eq!(
        survreg6(&kernel, 50, beta.to_vec(), 1e-9, 1e-10).err(),
        expected
    );
    assert_eq!(source.calls.load(Ordering::Relaxed), 2);
}

#[test]
fn malformed_callback_output_is_rejected_and_zero_density_tails_are_allowed() {
    let valid = mixture(0.1);
    for bad in [
        vec![],
        vec![valid; 2],
        vec![SurvregDensity {
            cdf: f64::NAN,
            ..valid
        }],
        vec![SurvregDensity {
            survival: 1.1,
            ..valid
        }],
        vec![SurvregDensity { pdf: -1.0, ..valid }],
        vec![SurvregDensity {
            score: f64::NAN,
            ..valid
        }],
        vec![SurvregDensity {
            curvature: f64::INFINITY,
            ..valid
        }],
    ] {
        assert!(check_density_batch(&bad, 1).is_err());
    }
    let zero = SurvregDensity {
        cdf: 0.0,
        survival: 1.0,
        pdf: 0.0,
        score: f64::NAN,
        curvature: f64::NAN,
    };
    check_density_batch(&[zero], 1).unwrap();
    assert_eq!(zero.distribution_kernel(), [0.0, 1.0, 0.0, 0.0]);
}
