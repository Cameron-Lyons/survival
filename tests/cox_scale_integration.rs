use ndarray::{Array2, array};
use survival::regression::{
    CoxNewData, CoxPHFit, CoxphData, CoxphOptions, SurvfitOptions, TieMethod,
};

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 2e-10 * (1.0 + expected.abs()),
        "expected {expected}, got {actual}"
    );
}

fn null_fit(method: TieMethod, counting: bool) -> CoxPHFit {
    CoxPHFit::fit(
        CoxphData::try_new(
            vec![1.0, 2.0, 1.0, 2.0],
            counting.then(|| vec![0.0; 4]),
            vec![1, 0, 1, 0],
            Array2::zeros((4, 0)),
            None,
            Some(vec![1, 1, 2, 2]),
            Some(vec![1000.0, 1000.0, -1000.0, -1000.0]),
        )
        .unwrap(),
        CoxphOptions {
            method,
            iter_max: 0,
            ..CoxphOptions::default()
        },
    )
    .unwrap()
}

#[test]
fn fit_diagnostics_and_all_prediction_routes_share_the_same_risk_scale() {
    for counting in [false, true] {
        for method in [TieMethod::Breslow, TieMethod::Efron, TieMethod::Exact] {
            let fit = null_fit(method, counting);
            for likelihood in fit.loglik {
                close(likelihood, -2.0 * 2.0_f64.ln());
            }
            for (&actual, expected) in fit.residuals.iter().zip([0.5, -0.5, 0.5, -0.5]) {
                close(actual, expected);
            }
            let newdata = CoxNewData::try_new(
                Array2::zeros((4, 0)),
                Some(vec![1, 1, 2, 2]),
                Some(fit.offset.clone()),
                Some(vec![2.0; 4]),
                counting.then(|| vec![0.0; 4]),
            )
            .unwrap();
            for newdata in [None, Some(&newdata)] {
                let expected = fit.predict_expected(newdata, true).unwrap();
                for value in expected.fit {
                    close(value, 0.5);
                }
                for value in expected.se_fit.unwrap() {
                    close(value, 0.5);
                }
                let survival = fit.predict_survival_at(&[0.0, 1.0, 2.0], newdata).unwrap();
                for column in 0..4 {
                    close(survival[[0, column]], 1.0);
                    close(survival[[1, column]], (-0.5_f64).exp());
                    close(survival[[2, column]], (-0.5_f64).exp());
                }
            }
            for stype in [1, 2] {
                let options = SurvfitOptions {
                    stype,
                    ..SurvfitOptions::default()
                };
                let expected_survival = if stype == 1 { 0.5 } else { (-0.5_f64).exp() };
                let mut individual_data = newdata.clone();
                individual_data.entry = Some(vec![0.0; 4]);
                for curves in [
                    fit.survfit(Some(&newdata), options).unwrap(),
                    fit.survfit_individual(&individual_data, &[1, 2, 3, 4], options)
                        .unwrap(),
                ] {
                    assert_eq!(curves.len(), 4);
                    for curve in curves {
                        for row in curve.surv {
                            close(row[0], expected_survival);
                        }
                        for row in curve.cumhaz {
                            close(row[0], 0.5);
                        }
                        assert!(
                            curve
                                .std_err
                                .unwrap()
                                .iter()
                                .flatten()
                                .all(|value| value.is_finite())
                        );
                    }
                }
            }
            for method in ["ederer", "hakulinen", "conditional"] {
                let expected = fit
                    .expected_survival(
                        &newdata,
                        &[0; 4],
                        &[1.0; 4],
                        Some(&[2.0; 4]),
                        Some(&[1.0, 2.0]),
                        method,
                    )
                    .unwrap();
                for row in expected.surv {
                    close(row[0], (-0.5_f64).exp());
                }
            }
            let baseline = fit.basehaz(true).unwrap();
            assert_eq!(
                baseline.hazard,
                vec![0.0, 0.0, f64::INFINITY, f64::INFINITY]
            );
        }
    }
}

fn common_offset_fit(offset: f64, method: TieMethod) -> CoxPHFit {
    CoxPHFit::fit(
        CoxphData::try_new(
            vec![1.0, 2.0, 1.0, 2.0],
            None,
            vec![1, 0, 1, 0],
            array![[0.0], [1.0], [0.0], [1.0]],
            None,
            Some(vec![1, 1, 2, 2]),
            Some(vec![offset; 4]),
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
fn common_offsets_keep_small_covariate_contributions_after_serialization() {
    for method in [TieMethod::Breslow, TieMethod::Efron, TieMethod::Exact] {
        let control = common_offset_fit(0.0, method);
        for offset in [-1e308, 1e308] {
            let fit = common_offset_fit(offset, method);
            #[cfg(feature = "python")]
            let restored = {
                let encoded =
                    bincode::serde::encode_to_vec(&fit, bincode::config::standard()).unwrap();
                let (restored, read): (CoxPHFit, usize) =
                    bincode::serde::decode_from_slice(&encoded, bincode::config::standard())
                        .unwrap();
                assert_eq!(read, encoded.len());
                restored
            };
            #[cfg(not(feature = "python"))]
            let restored = fit.clone();
            for fit in [&fit, &restored] {
                for (&actual, expected) in fit.loglik.iter().zip(control.loglik) {
                    close(actual, expected);
                }
                for (&actual, &expected) in fit.residuals.iter().zip(&control.residuals) {
                    close(actual, expected);
                }
                let actual = fit.predict_survival_at(&[1.0, 2.0], None).unwrap();
                let expected = control.predict_survival_at(&[1.0, 2.0], None).unwrap();
                for (&actual, &expected) in actual.iter().zip(&expected) {
                    close(actual, expected);
                }
                if method != TieMethod::Exact {
                    let actual = fit.score_residuals(false, None).unwrap();
                    let expected = control.score_residuals(false, None).unwrap();
                    for (&actual, &expected) in actual.iter().zip(&expected) {
                        close(actual, expected);
                    }
                }
            }
        }
    }
}

#[test]
fn future_counting_intervals_do_not_erase_the_current_risk_set() {
    for method in [TieMethod::Breslow, TieMethod::Efron, TieMethod::Exact] {
        for shift in [20.0, 30.0] {
            let n = 1001;
            let mut time = vec![3.0; n];
            time[0] = 1.0;
            let mut entry = vec![2.0; n];
            entry[0] = 0.0;
            let mut offset = vec![shift; n];
            offset[0] = 0.0;
            let mut status = vec![0; n];
            status[0] = 1;
            let fit = CoxPHFit::fit(
                CoxphData::try_new(
                    time,
                    Some(entry),
                    status,
                    Array2::zeros((n, 0)),
                    None,
                    None,
                    Some(offset),
                )
                .unwrap(),
                CoxphOptions {
                    method,
                    ..CoxphOptions::default()
                },
            )
            .unwrap();
            let newdata = CoxNewData::try_new(
                Array2::zeros((1, 0)),
                None,
                Some(vec![0.0]),
                Some(vec![1.0]),
                Some(vec![0.0]),
            )
            .unwrap();
            let expected = fit.predict_expected(Some(&newdata), true).unwrap();
            close(expected.fit[0], 1.0);
            close(expected.se_fit.unwrap()[0], 1.0);
            let curves = fit
                .survfit(Some(&newdata), SurvfitOptions::default())
                .unwrap();
            close(curves[0].cumhaz[0][0], 1.0);
            close(curves[0].surv[0][0], (-1.0_f64).exp());
            close(curves[0].std_err.as_ref().unwrap()[0][0], 1.0);
        }
    }
}
