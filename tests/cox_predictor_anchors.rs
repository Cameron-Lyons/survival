use ndarray::array;
use survival::regression::{
    CoxNewData, CoxPHFit, CoxphData, CoxphOptions, PredictReference, SurvfitOptions, TieMethod,
};

fn close(a: f64, b: f64) {
    assert!(a.is_finite() && b.is_finite(), "{a} != {b}");
    assert!((a - b).abs() < 2e-9 * (1. + b.abs()), "{a} != {b}");
}

fn model(shift: f64, method: TieMethod, counting: bool, robust: bool, dynamic: bool) -> CoxPHFit {
    let mut time = vec![1., 1., 2., 3., 1., 1., 2., 3.];
    let mut entry = vec![0.; 8];
    if dynamic {
        for value in &mut time[4..] {
            *value += 10.;
        }
        for value in &mut entry[4..] {
            *value += 10.;
        }
    }
    CoxPHFit::fit(
        CoxphData::try_new(
            time,
            counting.then_some(entry),
            vec![1, 1, 1, 0, 1, 1, 1, 0],
            array![
                [1., -1.],
                [1., 0.],
                [1., 1.],
                [1., 2.],
                [1., -1.],
                [1., 0.],
                [1., 1.],
                [1., 2.]
            ],
            None,
            (!dynamic).then(|| vec![0, 0, 0, 0, 1, 1, 1, 1]),
            Some(if dynamic {
                vec![1e308, 1e308, 1e308, 1e308, -1e308, -1e308, -1e308, -1e308]
            } else {
                vec![0.; 8]
            }),
        )
        .unwrap(),
        CoxphOptions {
            method,
            init: Some(vec![shift, 0.3]),
            iter_max: 0,
            robust: Some(robust),
            cluster: robust.then(|| vec![0, 0, 1, 1, 2, 2, 3, 3]),
            ..CoxphOptions::default()
        },
    )
    .unwrap()
}

fn rows(fit: &CoxPHFit) -> CoxNewData {
    CoxNewData::try_new(
        fit.x.clone(),
        fit.strata.clone(),
        Some(fit.offset.clone()),
        Some(fit.time.clone()),
        fit.entry.clone(),
    )
    .unwrap()
}

#[test]
fn huge_constant_coefficient_preserves_small_covariate_effects_everywhere() {
    for method in [TieMethod::Breslow, TieMethod::Efron, TieMethod::Exact] {
        for counting in [false, true] {
            for robust in [false, true] {
                if robust && method == TieMethod::Exact {
                    continue;
                }
                let reference = model(0., method, counting, robust, false);
                let reference_rows = rows(&reference);
                for shift in [1e12, 1e16, 1e308, -1e308] {
                    let fit = model(shift, method, counting, robust, false);
                    let newdata = rows(&fit);
                    for (a, b) in fit.loglik.iter().zip(reference.loglik) {
                        close(*a, b);
                    }
                    for (a, b) in fit.var.iter().zip(&reference.var) {
                        close(*a, *b);
                    }
                    for (a, b) in fit.residuals.iter().zip(&reference.residuals) {
                        close(*a, *b);
                    }
                    close(
                        fit.concordance.concordance[0],
                        reference.concordance.concordance[0],
                    );
                    for training in [false, true] {
                        let a = fit
                            .predict_expected((!training).then_some(&newdata), true)
                            .unwrap();
                        let b = reference
                            .predict_expected((!training).then_some(&reference_rows), true)
                            .unwrap();
                        for (a, b) in a.fit.iter().zip(b.fit) {
                            close(*a, b);
                        }
                        for (a, b) in a.se_fit.unwrap().iter().zip(b.se_fit.unwrap()) {
                            close(*a, b);
                        }
                    }
                    for stype in [1, 2] {
                        let options = SurvfitOptions {
                            stype,
                            ..SurvfitOptions::default()
                        };
                        let a = fit.survfit(Some(&newdata), options).unwrap();
                        let b = reference.survfit(Some(&reference_rows), options).unwrap();
                        for (a, b) in a.iter().zip(&b) {
                            for (a, b) in a.surv.iter().flatten().zip(b.surv.iter().flatten()) {
                                close(*a, *b);
                            }
                            for (a, b) in a.cumhaz.iter().flatten().zip(b.cumhaz.iter().flatten()) {
                                close(*a, *b);
                            }
                        }
                    }
                    let a = fit.predict_survival_at(&[0., 1., 2., 3.], None).unwrap();
                    let b = reference
                        .predict_survival_at(&[0., 1., 2., 3.], None)
                        .unwrap();
                    for (a, b) in a.iter().zip(&b) {
                        close(*a, *b);
                    }
                    let a = fit
                        .expected_survival(&newdata, &[0; 8], &[1.; 8], None, None, "ederer")
                        .unwrap();
                    let b = reference
                        .expected_survival(&reference_rows, &[0; 8], &[1.; 8], None, None, "ederer")
                        .unwrap();
                    for (a, b) in a.surv.iter().flatten().zip(b.surv.iter().flatten()) {
                        close(*a, *b);
                    }
                }
            }
        }
    }
}

#[test]
fn dynamic_offsets_keep_small_covariates_after_a_risk_set_departure() {
    for method in [TieMethod::Breslow, TieMethod::Efron] {
        let fit = model(1e16, method, true, false, true);
        let newdata = rows(&fit);
        // A separate zero-offset fit of each active block is an independent
        // control for expected counts and the individual counting curves.
        let reference = CoxPHFit::fit(
            CoxphData::try_new(
                vec![1., 1., 2., 3.],
                Some(vec![0.; 4]),
                vec![1, 1, 1, 0],
                array![[1., -1.], [1., 0.], [1., 1.], [1., 2.]],
                None,
                None,
                None,
            )
            .unwrap(),
            CoxphOptions {
                method,
                init: Some(vec![0., 0.3]),
                iter_max: 0,
                ..CoxphOptions::default()
            },
        )
        .unwrap();
        let control = rows(&reference);
        let expected = reference.predict_expected(Some(&control), false).unwrap();
        let actual = fit.predict_expected(Some(&newdata), false).unwrap();
        for (i, &a) in actual.fit.iter().enumerate() {
            close(a, expected.fit[i % 4]);
        }
        let data = CoxNewData::try_new(
            array![[1., 0.2], [1., 0.2]],
            None,
            Some(vec![1e308, -1e308]),
            Some(vec![3., 13.]),
            Some(vec![0., 10.]),
        )
        .unwrap();
        let control = CoxNewData::try_new(
            array![[1., 0.2]],
            None,
            None,
            Some(vec![3.]),
            Some(vec![0.]),
        )
        .unwrap();
        for stype in [1, 2] {
            let options = SurvfitOptions {
                stype,
                ..SurvfitOptions::default()
            };
            let actual = fit.survfit_individual(&data, &[0, 1], options).unwrap();
            let expected = reference
                .survfit_individual(&control, &[0], options)
                .unwrap();
            for curve in actual {
                for (a, b) in curve
                    .surv
                    .iter()
                    .flatten()
                    .zip(expected[0].surv.iter().flatten())
                {
                    close(*a, *b);
                }
                for (a, b) in curve
                    .cumhaz
                    .iter()
                    .flatten()
                    .zip(expected[0].cumhaz.iter().flatten())
                {
                    close(*a, *b);
                }
            }
        }
    }
}

#[test]
fn public_predictors_preserve_small_effects_when_coefficient_and_offset_cancel() {
    let z = [-1.5, -0.5, 0.5, 1.5];
    for method in [TieMethod::Breslow, TieMethod::Efron] {
        for shift in [1e12, 1e16, -1e16, 1e308, -1e308] {
            let fit = CoxPHFit::fit(
                CoxphData::try_new(
                    vec![1., 1., 2., 3.],
                    None,
                    vec![1, 1, 1, 0],
                    array![[1., z[0]], [1., z[1]], [1., z[2]], [1., z[3]]],
                    None,
                    None,
                    Some(vec![-shift; 4]),
                )
                .unwrap(),
                CoxphOptions {
                    method,
                    init: Some(vec![shift, 0.3]),
                    iter_max: 0,
                    ..CoxphOptions::default()
                },
            )
            .unwrap();
            let newdata = rows(&fit);
            for (actual, z) in fit.linear_predictors.iter().zip(z) {
                close(*actual, 0.3 * z);
            }
            for data in [None, Some(&newdata)] {
                let lp = fit
                    .predict_lp(data, false, PredictReference::Sample)
                    .unwrap();
                let risk = fit
                    .predict_risk(data, false, PredictReference::Sample)
                    .unwrap();
                for ((actual_lp, actual_risk), z) in lp.fit.iter().zip(risk.fit).zip(z) {
                    close(*actual_lp, 0.3 * z);
                    close(actual_risk, (0.3 * z).exp());
                }
            }
        }
    }
}

#[test]
fn public_predictors_with_se_preserve_cancellation_with_zero_mean_offset() {
    // R centers offsets only on the SE path. Balance them across strata so
    // both prediction routes have the same reference and a finite answer.
    let z = [-1.5, -0.5, 0.5, 1.5];
    for method in [TieMethod::Breslow, TieMethod::Efron] {
        for shift in [1e16, -1e16, 1e308, -1e308] {
            let fit = CoxPHFit::fit(
                CoxphData::try_new(
                    vec![1., 2., 1., 2.],
                    None,
                    vec![1, 0, 1, 0],
                    array![[1., z[0]], [1., z[1]], [-1., z[2]], [-1., z[3]]],
                    None,
                    Some(vec![0, 0, 1, 1]),
                    Some(vec![-shift, -shift, shift, shift]),
                )
                .unwrap(),
                CoxphOptions {
                    method,
                    init: Some(vec![shift, 0.3]),
                    iter_max: 0,
                    ..CoxphOptions::default()
                },
            )
            .unwrap();
            let newdata = rows(&fit);
            for data in [None, Some(&newdata)] {
                for se in [false, true] {
                    let lp = fit.predict_lp(data, se, PredictReference::Sample).unwrap();
                    let risk = fit
                        .predict_risk(data, se, PredictReference::Sample)
                        .unwrap();
                    for ((actual_lp, actual_risk), z) in lp.fit.iter().zip(risk.fit).zip(z) {
                        close(*actual_lp, 0.3 * z);
                        close(actual_risk, (0.3 * z).exp());
                    }
                    if se {
                        assert!(lp.se_fit.unwrap().iter().all(|value| value.is_finite()));
                        assert!(risk.se_fit.unwrap().iter().all(|value| value.is_finite()));
                    }
                }
            }
        }
    }
}
