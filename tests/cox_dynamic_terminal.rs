use ndarray::Array2;
use survival::regression::{
    CoxNewData, CoxPHFit, CoxphData, CoxphOptions, SurvfitOptions, TieMethod,
};

#[test]
fn successive_terminal_kalbfleisch_prentice_increments_keep_survival_zero() {
    for method in [TieMethod::Breslow, TieMethod::Efron, TieMethod::Exact] {
        for origin in [1000., 1e16, 1e308] {
            let model = CoxPHFit::fit(
                CoxphData::try_new(
                    vec![1., 1., 3., 3.],
                    Some(vec![0., 0., 2., 2.]),
                    vec![1; 4],
                    Array2::zeros((4, 0)),
                    None,
                    None,
                    Some(vec![0., 0., origin, origin]),
                )
                .unwrap(),
                CoxphOptions {
                    method,
                    ..CoxphOptions::default()
                },
            )
            .unwrap();
            let newdata = CoxNewData::try_new(
                Array2::zeros((2, 0)),
                None,
                Some(vec![0., origin]),
                None,
                None,
            )
            .unwrap();
            for predictions in [None, Some(&newdata)] {
                let curves = model
                    .survfit(
                        predictions,
                        SurvfitOptions {
                            stype: 1,
                            se_fit: false,
                            ..SurvfitOptions::default()
                        },
                    )
                    .unwrap();
                assert_eq!(curves.len(), 1);
                assert_eq!(curves[0].time, [1., 3.]);
                // Every active subject dies at each time. The second terminal
                // increment must not turn an already zero survival into NaN.
                assert!(curves[0].surv.iter().flatten().all(|&value| value == 0.));
            }
        }
    }
}

#[test]
fn heterogeneous_kalbfleisch_prentice_deaths_keep_the_small_risk_root() {
    for method in [TieMethod::Breslow, TieMethod::Efron, TieMethod::Exact] {
        for origin in [1000., 1e16, 1e308] {
            let model = CoxPHFit::fit(
                CoxphData::try_new(
                    vec![1.; 3],
                    None,
                    vec![1, 1, 0],
                    Array2::zeros((3, 0)),
                    None,
                    None,
                    Some(vec![origin, 0., 0.]),
                )
                .unwrap(),
                CoxphOptions {
                    method,
                    ..CoxphOptions::default()
                },
            )
            .unwrap();
            let newdata =
                CoxNewData::try_new(Array2::zeros((1, 0)), None, Some(vec![0.]), None, None)
                    .unwrap();
            let curves = model
                .survfit(
                    Some(&newdata),
                    SurvfitOptions {
                        stype: 1,
                        se_fit: false,
                        ..SurvfitOptions::default()
                    },
                )
                .unwrap();
            // The enormous-risk death has negligible likelihood contribution
            // at q=ln(2). The remaining one death and one survivor require
            // survival 1/2, independently of that departed risk's magnitude.
            assert!(
                (curves[0].surv[0][0] - 0.5).abs() < 2e-12,
                "method {method:?}, origin {origin}, survival {}",
                curves[0].surv[0][0]
            );
        }
    }
}

#[test]
fn kalbfleisch_prentice_finite_opposite_origins_are_not_a_terminal_hazard() {
    for origin in [1e16, 1e308] {
        for reverse in [false, true] {
            let death = if reverse { -origin } else { origin };
            let initial = if reverse && origin == 1e308 {
                origin
            } else {
                death
            };
            let mut model = CoxPHFit::fit(
                CoxphData::try_new(
                    vec![1.; 2],
                    None,
                    vec![1, 0],
                    Array2::zeros((2, 0)),
                    None,
                    None,
                    Some(vec![initial, -initial]),
                )
                .unwrap(),
                CoxphOptions::default(),
            )
            .unwrap();
            if initial != death {
                // This curve is defined even though the reversed model's
                // log likelihood (-2e308) exceeds f64. Set its public risks
                // before caching curves, invalidating the fitted snapshot.
                model.linear_predictors = vec![death, -death];
                model.offset = vec![death, -death];
            }
            let newdata = CoxNewData::try_new(
                Array2::zeros((2, 0)),
                None,
                Some(vec![death, -death]),
                None,
                None,
            )
            .unwrap();
            let curves = model
                .survfit(
                    Some(&newdata),
                    SurvfitOptions {
                        stype: 1,
                        se_fit: false,
                        ..SurvfitOptions::default()
                    },
                )
                .unwrap();
            if reverse {
                assert_eq!(curves[0].surv[0][0], 1.);
                assert!((curves[0].surv[0][1] - (-1_f64).exp()).abs() < 1e-15);
            } else {
                assert_eq!(curves[0].surv[0][0], 0.);
                assert_eq!(curves[0].surv[0][1], 1.);
            }
        }
    }
}
