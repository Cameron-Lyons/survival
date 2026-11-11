use ndarray::{Array2, array};
use survival::regression::{
    CoxNewData, CoxPHFit, CoxphData, CoxphOptions, SurvfitOptions, TieMethod,
};

fn close(actual: f64, expected: f64) {
    assert!(
        actual.is_finite() && expected.is_finite(),
        "{actual} != {expected}"
    );
    assert!(
        (actual - expected).abs() < 3e-10 * (1.0 + expected.abs()),
        "{actual} != {expected}"
    );
}

fn model(method: TieMethod, shifted: bool, counting: bool, strata: bool, p: usize) -> CoxPHFit {
    let mut time = vec![1., 2., 3., 1., 2., 3.];
    let mut entry = vec![0.; 6];
    if !strata {
        for t in &mut time[3..] {
            *t += 10.;
        }
        for t in &mut entry[3..] {
            *t += 10.;
        }
    }
    CoxPHFit::fit(
        CoxphData::try_new(
            time,
            counting.then_some(entry),
            vec![1, 1, 0, 1, 1, 0],
            Array2::zeros((6, p)),
            (method != TieMethod::Exact).then_some(vec![1., 2., 1., 1., 2., 1.]),
            strata.then_some(vec![1, 1, 1, 2, 2, 2]),
            Some(if shifted {
                vec![1000., 1000., 1000., -1000., -1000., -1000.]
            } else {
                vec![0.; 6]
            }),
        )
        .unwrap(),
        CoxphOptions {
            method,
            ..CoxphOptions::default()
        },
    )
    .unwrap()
}

fn training_data(fit: &CoxPHFit) -> CoxNewData {
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
fn opposite_strata_preserve_curves_expected_counts_and_cohort_survival() {
    for method in [TieMethod::Breslow, TieMethod::Efron, TieMethod::Exact] {
        let shifted = model(method, true, false, true, 0);
        let ordinary = model(method, false, false, true, 0);
        let x = training_data(&shifted);
        let reference_x = training_data(&ordinary);
        for stype in [1, 2] {
            let options = SurvfitOptions {
                stype,
                ..SurvfitOptions::default()
            };
            let actual = shifted.survfit(Some(&x), options).unwrap();
            let reference = ordinary.survfit(Some(&reference_x), options).unwrap();
            for (a, b) in actual.iter().zip(&reference) {
                for (a, b) in a.surv.iter().flatten().zip(b.surv.iter().flatten()) {
                    close(*a, *b);
                }
                for (a, b) in a.cumhaz.iter().flatten().zip(b.cumhaz.iter().flatten()) {
                    close(*a, *b);
                }
                if stype == 2 {
                    for (a, b) in a
                        .std_err
                        .as_ref()
                        .unwrap()
                        .iter()
                        .flatten()
                        .zip(b.std_err.as_ref().unwrap().iter().flatten())
                    {
                        close(*a, *b);
                    }
                }
            }
        }
        for newdata in [false, true] {
            let actual = shifted
                .predict_expected(newdata.then_some(&x), true)
                .unwrap();
            let reference = ordinary
                .predict_expected(newdata.then_some(&reference_x), true)
                .unwrap();
            for (a, b) in actual.fit.iter().zip(&reference.fit) {
                close(*a, *b);
            }
            for (a, b) in actual.se_fit.unwrap().iter().zip(reference.se_fit.unwrap()) {
                close(*a, b);
            }
        }
        let actual = shifted
            .predict_survival_at(&[0., 1., 2., 4.], Some(&x))
            .unwrap();
        let reference = ordinary
            .predict_survival_at(&[0., 1., 2., 4.], Some(&reference_x))
            .unwrap();
        for (a, b) in actual.iter().zip(&reference) {
            close(*a, *b);
        }
        for method in ["ederer", "hakulinen", "conditional"] {
            let actual = shifted
                .expected_survival(&x, &[0; 6], &[1.; 6], Some(&[4.; 6]), None, method)
                .unwrap();
            let reference = ordinary
                .expected_survival(
                    &reference_x,
                    &[0; 6],
                    &[1.; 6],
                    Some(&[4.; 6]),
                    None,
                    method,
                )
                .unwrap();
            assert_eq!(actual.time, reference.time);
            assert_eq!(actual.n_risk, reference.n_risk);
            for (a, b) in actual
                .surv
                .iter()
                .flatten()
                .zip(reference.surv.iter().flatten())
            {
                close(*a, *b);
            }
        }
    }
}

#[test]
fn disjoint_counting_intervals_keep_late_events_and_variance() {
    for method in [TieMethod::Breslow, TieMethod::Efron] {
        let shifted = model(method, true, true, false, 0);
        let ordinary = model(method, false, true, false, 0);
        let x = training_data(&shifted);
        let reference_x = training_data(&ordinary);
        for newdata in [false, true] {
            let actual = shifted
                .predict_expected(newdata.then_some(&x), true)
                .unwrap();
            let reference = ordinary
                .predict_expected(newdata.then_some(&reference_x), true)
                .unwrap();
            for (a, b) in actual.fit.iter().zip(&reference.fit) {
                close(*a, *b);
            }
            for (a, b) in actual.se_fit.unwrap().iter().zip(reference.se_fit.unwrap()) {
                close(*a, b);
            }
        }
        let subject = CoxNewData::try_new(
            Array2::zeros((2, 0)),
            None,
            Some(vec![1000., -1000.]),
            Some(vec![3., 13.]),
            Some(vec![0., 10.]),
        )
        .unwrap();
        let reference_subject = CoxNewData::try_new(
            Array2::zeros((2, 0)),
            None,
            Some(vec![0.; 2]),
            Some(vec![3., 13.]),
            Some(vec![0., 10.]),
        )
        .unwrap();
        for stype in [1, 2] {
            let options = SurvfitOptions {
                stype,
                ..SurvfitOptions::default()
            };
            let actual = shifted
                .survfit_individual(&subject, &[1, 1], options)
                .unwrap();
            let reference = ordinary
                .survfit_individual(&reference_subject, &[1, 1], options)
                .unwrap();
            assert_eq!(actual[0].time, reference[0].time);
            for (a, b) in actual[0]
                .surv
                .iter()
                .flatten()
                .zip(reference[0].surv.iter().flatten())
            {
                close(*a, *b);
            }
            for (a, b) in actual[0]
                .cumhaz
                .iter()
                .flatten()
                .zip(reference[0].cumhaz.iter().flatten())
            {
                close(*a, *b);
            }
            if stype == 2 {
                for (a, b) in actual[0]
                    .std_err
                    .as_ref()
                    .unwrap()
                    .iter()
                    .flatten()
                    .zip(reference[0].std_err.as_ref().unwrap().iter().flatten())
                {
                    close(*a, *b);
                }
            }
        }
        // A late range must retain its own events even after an unrepresentably
        // large earlier baseline cumulative hazard.
        close(
            shifted.predict_expected(Some(&x), false).unwrap().fit[4],
            1. / 4. + 2. / 3.,
        );
    }
}

#[test]
fn right_censored_late_tiny_risks_reappear_after_large_risks_depart() {
    let fit = model(TieMethod::Breslow, true, false, false, 0);
    let x = training_data(&fit);
    let prediction = fit.predict_expected(Some(&x), true).unwrap();
    close(prediction.fit[3], 1. / 4.);
    close(prediction.fit[4], 1. / 4. + 2. / 3.);
    close(
        prediction.se_fit.as_ref().unwrap()[4],
        (1.0_f64 / 16. + 2. / 9.).sqrt(),
    );
    let survival = fit.predict_survival_at(&[11., 12.], Some(&x)).unwrap();
    close(survival[(1, 4)], (-1.0_f64 / 4. - 2. / 3.).exp());
}

#[test]
fn basehaz_applies_center_transform_before_exponentiating() {
    let mut fit = model(TieMethod::Breslow, true, false, true, 1);
    fit.coefficients[0] = 1.;
    fit.means[0] = 1000.;
    let curve = fit.basehaz(false).unwrap();
    assert!(curve.hazard.iter().all(|h| !h.is_nan()));
    close(curve.hazard[0], 0.);
    close(curve.hazard[3], 1. / 4.);
    close(curve.hazard[4], 1. / 4. + 2. / 3.);
}

#[test]
fn prediction_se_scales_a_normalized_quadratic_before_sqrt() {
    let data = CoxphData::try_new(
        vec![1., 2., 3., 4., 5., 6.],
        None,
        vec![1, 1, 0, 1, 1, 0],
        array![[0., 0.], [1., 0.], [0., 1.], [1., 2.], [2., 1.], [2., 3.]],
        None,
        None,
        None,
    )
    .unwrap();
    let mut fit = CoxPHFit::fit(
        data,
        CoxphOptions {
            init: Some(vec![0., 0.]),
            iter_max: 0,
            ..CoxphOptions::default()
        },
    )
    .unwrap();
    fit.var = array![[1., -0.5], [-0.5, 1.]];
    let newdata = CoxNewData::try_new(
        array![[4., 5.], [4., 5.]],
        None,
        Some(vec![500., 1000.]),
        Some(vec![5., 5.]),
        None,
    )
    .unwrap();
    let curves = fit
        .survfit(Some(&newdata), SurvfitOptions::default())
        .unwrap();
    assert!(
        curves[0]
            .std_err
            .as_ref()
            .unwrap()
            .iter()
            .all(|row| row[0].is_finite() && row[1] == f64::INFINITY)
    );
    let prediction = fit.predict_expected(Some(&newdata), true).unwrap();
    assert!(prediction.se_fit.as_ref().unwrap()[0].is_finite());
    assert_eq!(prediction.se_fit.as_ref().unwrap()[1], f64::INFINITY);
    let survival = fit.predict_survival(Some(&newdata), true).unwrap();
    assert_eq!(survival.fit, [0., 0.]);
    assert_eq!(survival.se_fit.unwrap(), [0., 0.]);
}

#[test]
fn no_events_remain_zero_even_for_generated_infinite_prediction_risk() {
    let fit = CoxPHFit::fit(
        CoxphData::try_new(
            vec![1., 2.],
            None,
            vec![0, 0],
            Array2::zeros((2, 0)),
            None,
            None,
            None,
        )
        .unwrap(),
        CoxphOptions::default(),
    )
    .unwrap();
    let data = CoxNewData::try_new(
        Array2::zeros((1, 0)),
        None,
        Some(vec![1000.]),
        Some(vec![2.]),
        None,
    )
    .unwrap();
    let curve = fit.survfit(Some(&data), SurvfitOptions::default()).unwrap();
    assert_eq!(curve[0].cumhaz, vec![vec![0.], vec![0.]]);
    assert_eq!(curve[0].surv, vec![vec![1.], vec![1.]]);
    assert_eq!(
        curve[0].std_err.as_ref().unwrap(),
        &vec![vec![0.], vec![0.]]
    );
    let prediction = fit.predict_expected(Some(&data), true).unwrap();
    assert_eq!(prediction.fit, [0.]);
    assert_eq!(prediction.se_fit.unwrap(), [0.]);
}

#[test]
fn kp_does_not_reconstruct_a_rounded_baseline_survival_increment() {
    for lp in [-100., 100.] {
        let mut fit = CoxPHFit::fit(
            CoxphData::try_new(
                vec![1., 2.],
                None,
                vec![1, 0],
                Array2::zeros((2, 1)),
                None,
                None,
                None,
            )
            .unwrap(),
            CoxphOptions::default(),
        )
        .unwrap();
        fit.x.fill(1.);
        fit.means[0] = 0.;
        fit.coefficients[0] = lp;
        fit.linear_predictors.fill(lp);
        let data = CoxNewData::try_new(array![[1.]], None, None, None, None).unwrap();
        let curve = fit
            .survfit(
                Some(&data),
                SurvfitOptions {
                    stype: 1,
                    ..SurvfitOptions::default()
                },
            )
            .unwrap();
        close(curve[0].surv[0][0], 0.5);
        // R's vartype=1 formula is scale-dependent: at +/-100 it tends
        // to n_event / risk_sum^2, while the zero-LP value is 1/2.
        close(curve[0].std_err.as_ref().unwrap()[0][0], 0.5);
    }
}

#[test]
fn small_late_hazard_is_not_subtracted_from_an_infinite_prefix() {
    let mut fit = model(TieMethod::Breslow, true, true, false, 0);
    for value in &mut fit.offset {
        *value = -*value;
    }
    for value in &mut fit.linear_predictors {
        *value = -*value;
    }
    let data = training_data(&fit);
    let prediction = fit.predict_expected(Some(&data), true).unwrap();
    close(prediction.fit[4], 1. / 4. + 2. / 3.);
    close(
        prediction.se_fit.as_ref().unwrap()[4],
        (1.0_f64 / 16. + 2. / 9.).sqrt(),
    );
}

fn anchored_model(shift: f64, method: TieMethod, counting: bool) -> CoxPHFit {
    CoxPHFit::fit(
        CoxphData::try_new(
            vec![1., 1., 2., 3., 1., 1., 2., 3.],
            counting.then(|| vec![0.; 8]),
            vec![1, 1, 1, 0, 1, 1, 1, 0],
            array![[-1.], [0.], [1.], [2.], [-1.], [0.], [1.], [2.]],
            None,
            Some(vec![1, 1, 1, 1, 2, 2, 2, 2]),
            Some(vec![
                shift, shift, shift, shift, -shift, -shift, -shift, -shift,
            ]),
        )
        .unwrap(),
        CoxphOptions {
            method,
            init: Some(vec![0.4]),
            iter_max: 0,
            ..CoxphOptions::default()
        },
    )
    .unwrap()
}

#[test]
fn stratum_origins_preserve_small_covariate_effects_at_large_offsets() {
    for method in [TieMethod::Breslow, TieMethod::Efron, TieMethod::Exact] {
        for counting in [false, true] {
            let reference = anchored_model(0., method, counting);
            let reference_data = training_data(&reference);
            let reference_prediction = reference
                .predict_expected(Some(&reference_data), true)
                .unwrap();
            let reference_survival = reference
                .predict_survival_at(&[0., 1., 2., 3.], None)
                .unwrap();
            for shift in [1000., 1e12, 1e16, 1e308] {
                let fit = anchored_model(shift, method, counting);
                let data = training_data(&fit);
                for (a, b) in fit.coefficients.iter().zip(&reference.coefficients) {
                    close(*a, *b);
                }
                for (a, b) in fit.var.iter().zip(&reference.var) {
                    close(*a, *b);
                }
                for (a, b) in fit.loglik.iter().zip(&reference.loglik) {
                    close(*a, *b);
                }
                let prediction = fit.predict_expected(Some(&data), true).unwrap();
                for (a, b) in prediction.fit.iter().zip(&reference_prediction.fit) {
                    close(*a, *b);
                }
                for (a, b) in prediction
                    .se_fit
                    .as_ref()
                    .unwrap()
                    .iter()
                    .zip(reference_prediction.se_fit.as_ref().unwrap())
                {
                    close(*a, *b);
                }
                let survival = fit.predict_survival_at(&[0., 1., 2., 3.], None).unwrap();
                for (a, b) in survival.iter().zip(&reference_survival) {
                    close(*a, *b);
                }
                for stype in [1, 2] {
                    let options = SurvfitOptions {
                        stype,
                        ..SurvfitOptions::default()
                    };
                    let actual = fit.survfit(Some(&data), options).unwrap();
                    let expected = reference.survfit(Some(&reference_data), options).unwrap();
                    for (actual, expected) in actual.iter().zip(&expected) {
                        for (a, b) in actual
                            .surv
                            .iter()
                            .flatten()
                            .zip(expected.surv.iter().flatten())
                        {
                            close(*a, *b);
                        }
                        for (a, b) in actual
                            .cumhaz
                            .iter()
                            .flatten()
                            .zip(expected.cumhaz.iter().flatten())
                        {
                            close(*a, *b);
                        }
                        if stype == 2 {
                            for (a, b) in actual
                                .std_err
                                .as_ref()
                                .unwrap()
                                .iter()
                                .flatten()
                                .zip(expected.std_err.as_ref().unwrap().iter().flatten())
                            {
                                close(*a, *b);
                            }
                        }
                    }
                }
                for method in ["ederer", "hakulinen", "conditional"] {
                    let actual = fit
                        .expected_survival(&data, &[0; 8], &[1.; 8], Some(&[4.; 8]), None, method)
                        .unwrap();
                    let expected = reference
                        .expected_survival(
                            &reference_data,
                            &[0; 8],
                            &[1.; 8],
                            Some(&[4.; 8]),
                            None,
                            method,
                        )
                        .unwrap();
                    for (a, b) in actual
                        .surv
                        .iter()
                        .flatten()
                        .zip(expected.surv.iter().flatten())
                    {
                        close(*a, *b);
                    }
                }
            }
        }
    }
}

#[test]
fn unstratified_newdata_broadcast_uses_each_curves_origin() {
    let reference = anchored_model(0., TieMethod::Efron, false);
    let reference_data = CoxNewData::try_new(array![[0.2]], None, None, None, None).unwrap();
    let expected = reference
        .survfit(Some(&reference_data), SurvfitOptions::default())
        .unwrap();
    for shift in [1000., 1e12, 1e16, 1e308] {
        let fit = anchored_model(shift, TieMethod::Efron, false);
        for (offset, matching) in [(shift, 0), (-shift, 1)] {
            let data =
                CoxNewData::try_new(array![[0.2]], None, Some(vec![offset]), None, None).unwrap();
            let actual = fit.survfit(Some(&data), SurvfitOptions::default()).unwrap();
            for (a, b) in actual[matching]
                .surv
                .iter()
                .flatten()
                .zip(expected[matching].surv.iter().flatten())
            {
                close(*a, *b);
            }
            for (a, b) in actual[matching]
                .std_err
                .as_ref()
                .unwrap()
                .iter()
                .flatten()
                .zip(
                    expected[matching]
                        .std_err
                        .as_ref()
                        .unwrap()
                        .iter()
                        .flatten(),
                )
            {
                close(*a, *b);
            }
            let other = 1 - matching;
            assert!(
                actual[other]
                    .surv
                    .iter()
                    .flatten()
                    .all(|s| *s == if matching == 0 { 0. } else { 1. })
            );
        }
        assert!(
            fit.basehaz(true)
                .unwrap()
                .hazard
                .iter()
                .all(|h| !h.is_nan())
        );
        assert!(
            fit.basehaz(false)
                .unwrap()
                .hazard
                .iter()
                .all(|h| !h.is_nan())
        );
    }
}

#[test]
fn cohort_survival_stays_zero_after_extreme_risk_extinction() {
    let fit = CoxPHFit::fit(
        CoxphData::try_new(
            vec![1., 2.],
            None,
            vec![1, 1],
            Array2::zeros((2, 0)),
            None,
            None,
            None,
        )
        .unwrap(),
        CoxphOptions::default(),
    )
    .unwrap();
    let data =
        CoxNewData::try_new(Array2::zeros((1, 0)), None, Some(vec![1000.]), None, None).unwrap();
    for method in ["ederer", "hakulinen", "conditional"] {
        let result = fit
            .expected_survival(&data, &[0], &[1.], Some(&[3.]), None, method)
            .unwrap();
        assert_eq!(result.surv, vec![vec![0.], vec![0.]]);
    }
}

#[test]
fn kp_preserves_point_estimates_when_baseline_increments_round_to_zero_or_one() {
    for (lp, tied) in [(-7., false), (19., true), (20., true)] {
        let n = if tied { 4 } else { 2 };
        let fit = CoxPHFit::fit(
            CoxphData::try_new(
                if tied {
                    vec![1., 1., 2., 2.]
                } else {
                    vec![1., 2.]
                },
                None,
                if tied { vec![1, 1, 0, 0] } else { vec![1, 0] },
                Array2::ones((n, 1)),
                None,
                None,
                None,
            )
            .unwrap(),
            CoxphOptions {
                init: Some(vec![lp]),
                iter_max: 0,
                ..CoxphOptions::default()
            },
        )
        .unwrap();
        let data =
            CoxNewData::try_new(array![[1.]], None, None, Some(vec![2.]), Some(vec![0.])).unwrap();
        let options = SurvfitOptions {
            stype: 1,
            ..SurvfitOptions::default()
        };
        let curve = fit.survfit(Some(&data), options).unwrap();
        close(curve[0].surv[0][0], 0.5);
        close(curve[0].surv[1][0], 0.5);
        let individual = fit.survfit_individual(&data, &[1], options).unwrap();
        close(individual[0].surv[0][0], 0.5);
        close(individual[0].surv[1][0], 0.5);
    }
}

#[test]
fn event_free_extreme_offsets_do_not_require_representable_centered_risks() {
    for counting in [false, true] {
        let fit = CoxPHFit::fit(
            CoxphData::try_new(
                vec![1., 2., 3., 4.],
                counting.then(|| vec![0.; 4]),
                vec![0; 4],
                Array2::zeros((4, 0)),
                None,
                Some(vec![0, 0, 0, 1]),
                Some(vec![1e308, 1e308, 1e308, -1e308]),
            )
            .unwrap(),
            CoxphOptions::default(),
        )
        .unwrap();
        // Keep the established public no-event centering, rather than repairing
        // this edge case by changing the stored fit's meaning.
        assert_eq!(fit.linear_predictors, vec![5e307, 5e307, 5e307, -1.5e308]);
        let mut data = training_data(&fit);
        data.entry = Some(vec![0.; 4]);
        for centered in [false, true] {
            assert!(
                fit.basehaz(centered)
                    .unwrap()
                    .hazard
                    .iter()
                    .all(|h| *h == 0.)
            );
        }
        for newdata in [None, Some(&data)] {
            for stype in [1, 2] {
                for curve in fit
                    .survfit(
                        newdata,
                        SurvfitOptions {
                            stype,
                            ..SurvfitOptions::default()
                        },
                    )
                    .unwrap()
                {
                    assert!(curve.surv.iter().flatten().all(|s| *s == 1.));
                    assert!(curve.cumhaz.iter().flatten().all(|h| *h == 0.));
                    assert!(curve.std_err.unwrap().iter().flatten().all(|se| *se == 0.));
                }
            }
            let prediction = fit.predict_expected(newdata, true).unwrap();
            assert!(prediction.fit.iter().all(|h| *h == 0.));
            assert!(prediction.se_fit.unwrap().iter().all(|se| *se == 0.));
            assert!(
                fit.predict_survival_at(&[0., 2., 4.], newdata)
                    .unwrap()
                    .iter()
                    .all(|s| *s == 1.)
            );
        }
        for stype in [1, 2] {
            for curve in fit
                .survfit_individual(
                    &data,
                    &[0, 1, 2, 3],
                    SurvfitOptions {
                        stype,
                        ..SurvfitOptions::default()
                    },
                )
                .unwrap()
            {
                assert!(curve.surv.iter().flatten().all(|s| *s == 1.));
                assert!(curve.cumhaz.iter().flatten().all(|h| *h == 0.));
                assert!(curve.std_err.unwrap().iter().flatten().all(|se| *se == 0.));
            }
        }
        for method in ["ederer", "hakulinen", "conditional"] {
            let cohort = fit
                .expected_survival(
                    &data,
                    &[0; 4],
                    &[1.; 4],
                    Some(&[5.; 4]),
                    Some(&[0., 4.]),
                    method,
                )
                .unwrap();
            assert!(cohort.surv.iter().flatten().all(|s| *s == 1.));
        }
    }
}

#[test]
fn ordinary_kp_ties_satisfy_independent_weighted_likelihood_solutions() {
    for (offset, weights, expected) in [
        (
            vec![2.0_f64.ln(), 2.0_f64.ln(), 0., 4.0_f64.ln()],
            vec![1., 3., 2., 0.5],
            1. / 3.,
        ),
        (
            vec![0., 2.0_f64.ln(), 3.0_f64.ln(), 4.0_f64.ln()],
            vec![1., 0.5, 2., 1.],
            20. / (1. + 481.0_f64.sqrt()),
        ),
    ] {
        for counting in [false, true] {
            let fit = CoxPHFit::fit(
                CoxphData::try_new(
                    vec![1., 1., 2., 2.],
                    counting.then(|| vec![0.; 4]),
                    vec![1, 1, 0, 0],
                    Array2::zeros((4, 0)),
                    Some(weights.clone()),
                    None,
                    Some(offset.clone()),
                )
                .unwrap(),
                CoxphOptions::default(),
            )
            .unwrap();
            let data = CoxNewData::try_new(
                Array2::zeros((1, 0)),
                None,
                Some(vec![offset[0]]),
                None,
                None,
            )
            .unwrap();
            let curve = fit
                .survfit(
                    Some(&data),
                    SurvfitOptions {
                        stype: 1,
                        ..SurvfitOptions::default()
                    },
                )
                .unwrap();
            close(curve[0].surv[0][0], expected);
        }
    }
}

#[test]
fn kp_keeps_a_tiny_surviving_risk_without_denominator_subtraction() {
    for counting in [false, true] {
        let fit = CoxPHFit::fit(
            CoxphData::try_new(
                vec![1., 2.],
                counting.then(|| vec![0.; 2]),
                vec![1, 0],
                Array2::zeros((2, 0)),
                None,
                None,
                Some(vec![29.3, 0.1]),
            )
            .unwrap(),
            CoxphOptions::default(),
        )
        .unwrap();
        let data = CoxNewData::try_new(
            Array2::zeros((1, 0)),
            None,
            Some(vec![25.559318370144094]),
            None,
            None,
        )
        .unwrap();
        let curve = fit
            .survfit(
                Some(&data),
                SurvfitOptions {
                    stype: 1,
                    ..SurvfitOptions::default()
                },
            )
            .unwrap();
        close(curve[0].surv[0][0], 0.5);
    }
}

#[test]
fn empty_expected_intervals_preserve_missing_values_and_their_payloads() {
    let fit = CoxPHFit::fit(
        CoxphData::try_new(
            vec![1., 2.],
            None,
            vec![1, 0],
            Array2::zeros((2, 0)),
            None,
            None,
            None,
        )
        .unwrap(),
        CoxphOptions::default(),
    )
    .unwrap();
    for missing in [f64::NAN, f64::from_bits(0x7ff0_0000_0000_07a2)] {
        let data = CoxNewData::try_new_prediction(
            Array2::zeros((1, 0)),
            None,
            Some(vec![missing]),
            Some(vec![0.5]),
            None,
        )
        .unwrap();
        for prediction in [
            fit.predict_expected(Some(&data), true).unwrap(),
            fit.predict_survival(Some(&data), true).unwrap(),
        ] {
            for value in [prediction.fit[0], prediction.se_fit.unwrap()[0]] {
                assert!(value.is_nan());
                assert_eq!(
                    value.to_bits() & 0x0007_ffff_ffff_ffff,
                    missing.to_bits() & 0x0007_ffff_ffff_ffff
                );
            }
        }
    }
}

#[test]
fn kp_counting_falls_back_when_future_entrants_erase_the_active_denominator() {
    let n = 1001;
    let mut time = vec![3.; n];
    time[0] = 1.;
    let mut entry = vec![2.; n];
    entry[0] = 0.;
    let mut status = vec![0; n];
    status[0] = 1;
    let mut weights = vec![1.0; n];
    weights[0] = 1.1;
    let mut fit = CoxPHFit::fit(
        CoxphData::try_new(
            time,
            Some(entry),
            status,
            Array2::zeros((n, 1)),
            Some(weights),
            None,
            None,
        )
        .unwrap(),
        CoxphOptions::default(),
    )
    .unwrap();
    fit.coefficients[0] = 22.8;
    for row in 1..n {
        fit.x[(row, 0)] = 1.;
        fit.linear_predictors[row] = 22.8;
    }
    let data = CoxNewData::try_new(array![[0.]], None, None, None, None).unwrap();
    let curve = fit
        .survfit(
            Some(&data),
            SurvfitOptions {
                stype: 1,
                ..SurvfitOptions::default()
            },
        )
        .unwrap();
    close(curve[0].cumhaz[0][0], 1.);
    close(
        curve[0].std_err.as_ref().unwrap()[0][0],
        (1.0_f64 / 1.1).sqrt(),
    );
    close(curve[0].surv[0][0], 0.);
}

#[test]
fn terminal_kp_increments_stay_terminal_for_a_generated_negative_infinite_lp() {
    let fit = CoxPHFit::fit(
        CoxphData::try_new(
            vec![1., 1.],
            None,
            vec![1, 1],
            Array2::zeros((2, 0)),
            None,
            Some(vec![0, 1]),
            Some(vec![1e308, -1e308]),
        )
        .unwrap(),
        CoxphOptions::default(),
    )
    .unwrap();
    let data =
        CoxNewData::try_new(Array2::zeros((1, 0)), None, Some(vec![-1e308]), None, None).unwrap();
    let curves = fit
        .survfit(
            Some(&data),
            SurvfitOptions {
                stype: 1,
                ..SurvfitOptions::default()
            },
        )
        .unwrap();
    for curve in curves {
        assert_eq!(curve.surv, vec![vec![0.]]);
    }
}
