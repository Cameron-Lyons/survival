use super::*;
use crate::regression::{
    ModelTerm, PenaltyTerm, SurvpenalData, SurvpenalFit, SurvpenalFitResult, SurvpenalOptions,
};

fn data() -> SurvregData {
    SurvregData::try_new(
        vec![1.2, 2.5, 0.9, 3.0, 1.8, 2.7, 3.9, 1.1],
        vec![1, 0, 1, 1, 1, 0, 1, 1],
        Array2::from_shape_fn((8, 2), |(i, j)| if j == 0 { 1.0 } else { (i % 3) as f64 }),
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap()
}

#[test]
fn public_aft_offset_is_checked_before_initial_fit() {
    let mut data = data();
    data.offset = Some(vec![0.0]);
    let distribution = SurvregDistribution::from_name("gaussian", None).unwrap();
    assert!(
        survreg_fit(
            &data,
            &distribution,
            None,
            0.0,
            &SurvregControl::default(),
            false
        )
        .is_err()
    );
}

#[test]
fn public_aft_empty_design_is_checked_before_rescaling() {
    let mut data = data();
    data.covariates = Array2::zeros((data.n(), 0));
    let distribution = SurvregDistribution::from_name("gaussian", None).unwrap();
    assert!(
        SurvregFitResult::fit(
            &data,
            &distribution,
            None,
            0.0,
            &SurvregControl::default(),
            None
        )
        .is_err()
    );
}

fn terms() -> Vec<ModelTerm> {
    vec![
        ModelTerm {
            columns: vec![0],
            penalty: None,
        },
        ModelTerm {
            columns: vec![1],
            penalty: Some(PenaltyTerm::ridge(Some(1.0), None, 0.1, false, None).unwrap()),
        },
    ]
}

fn malformed() -> Vec<(&'static str, SurvregData)> {
    let mut cases = vec![];
    for length in [0, 7, 9] {
        cases.extend([
            (
                "status",
                SurvregData {
                    status: vec![1; length],
                    ..data()
                },
            ),
            (
                "covariates",
                SurvregData {
                    covariates: Array2::ones((length, 2)),
                    ..data()
                },
            ),
            (
                "time2",
                SurvregData {
                    time2: Some(vec![4.0; length]),
                    ..data()
                },
            ),
            (
                "weights",
                SurvregData {
                    weights: Some(vec![1.0; length]),
                    ..data()
                },
            ),
            (
                "offset",
                SurvregData {
                    offset: Some(vec![0.0; length]),
                    ..data()
                },
            ),
            (
                "strata",
                SurvregData {
                    strata: Some(vec![0; length]),
                    ..data()
                },
            ),
            (
                "cluster",
                SurvregData {
                    cluster: Some(vec![0; length]),
                    ..data()
                },
            ),
        ]);
    }
    cases.push((
        "empty",
        SurvregData {
            time: vec![],
            ..data()
        },
    ));
    cases.push((
        "no columns",
        SurvregData {
            covariates: Array2::zeros((8, 0)),
            ..data()
        },
    ));
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut x = data();
        x.covariates[(3, 1)] = value;
        cases.push(("nonfinite covariate", x));
        cases.push((
            "time",
            SurvregData {
                time: vec![value; 8],
                ..data()
            },
        ));
        cases.push((
            "offset",
            SurvregData {
                offset: Some(vec![value; 8]),
                ..data()
            },
        ));
        cases.push((
            "weights",
            SurvregData {
                weights: Some(vec![value; 8]),
                ..data()
            },
        ));
        cases.push((
            "interval stop",
            SurvregData {
                status: vec![3; 8],
                time2: Some(vec![value; 8]),
                ..data()
            },
        ));
    }
    for weight in [0.0, -1.0] {
        cases.push((
            "nonpositive weight",
            SurvregData {
                weights: Some(vec![weight; 8]),
                ..data()
            },
        ));
    }
    for code in [-1, 4] {
        cases.push((
            "invalid status",
            SurvregData {
                status: vec![code; 8],
                ..data()
            },
        ));
    }
    cases.push((
        "missing interval stop",
        SurvregData {
            status: vec![3; 8],
            ..data()
        },
    ));
    cases.push((
        "reversed interval",
        SurvregData {
            status: vec![3; 8],
            time2: Some(vec![0.0; 8]),
            ..data()
        },
    ));
    cases.push((
        "stratum overflow",
        SurvregData {
            strata: Some(vec![usize::MAX; 8]),
            ..data()
        },
    ));
    cases
}

fn assert_fit_errors(case: &str, input: SurvregData, distribution: &SurvregDistribution) {
    let control = SurvregControl::default();
    let penalized = SurvpenalData {
        survreg: input,
        terms: terms(),
    };
    let options = SurvpenalOptions::default();
    let results = [
        survreg_fit(&penalized.survreg, distribution, None, 0.0, &control, false).map(|_| ()),
        SurvregFitResult::fit(&penalized.survreg, distribution, None, 0.0, &control, None)
            .map(|_| ()),
        SurvpenalFit::fit(&penalized, distribution, &options).map(|_| ()),
        SurvpenalFitResult::fit(&penalized, distribution, &options, None).map(|_| ()),
    ];
    for (index, result) in results.into_iter().enumerate() {
        assert!(result.is_err(), "fit {index} accepted {case}");
    }
}

#[test]
fn all_aft_fitters_revalidate_public_input_fields() {
    for name in ["gaussian", "weibull"] {
        let distribution = SurvregDistribution::from_name(name, None).unwrap();
        for (case, input) in malformed() {
            assert!(
                SurvpenalData::try_new(input.clone(), terms()).is_err(),
                "{case}"
            );
            assert_fit_errors(case, input, &distribution);
        }
    }
}

#[test]
fn aft_fitters_reject_unrepresentable_covariance_sizes() {
    let distribution = SurvregDistribution::from_name("gaussian", None).unwrap();
    for size in [usize::MAX, usize::MAX / 2, u32::MAX as usize] {
        assert_fit_errors(
            "covariance size",
            SurvregData {
                strata: Some(vec![size - 1; 8]),
                ..data()
            },
            &distribution,
        );
        let penalized = SurvpenalData::try_new(data(), terms()).unwrap();
        assert!(
            SurvregFitResult::fit(
                &penalized.survreg,
                &distribution,
                None,
                0.0,
                &SurvregControl::default(),
                Some(size)
            )
            .is_err()
        );
        assert!(
            SurvpenalFitResult::fit(
                &penalized,
                &distribution,
                &SurvpenalOptions::default(),
                Some(size)
            )
            .is_err()
        );
    }
}

#[test]
fn aft_ignored_upper_endpoints_and_unused_strata_remain_supported() {
    let mut input = data();
    input.status[0] = 3;
    input.time2 = Some(vec![2.0; 8]);
    input.strata = Some(vec![0, 2, 0, 2, 0, 2, 0, 2]);
    let distribution = SurvregDistribution::from_name("gaussian", None).unwrap();
    let expected = SurvregFitResult::fit(
        &input,
        &distribution,
        None,
        0.0,
        &SurvregControl::default(),
        Some(4),
    )
    .unwrap();
    assert_eq!(expected.coefficients.len(), 6);
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        input.time2.as_mut().unwrap()[1..].fill(value);
        input.validate().unwrap();
        let actual = SurvregFitResult::fit(
            &input,
            &distribution,
            None,
            0.0,
            &SurvregControl::default(),
            Some(4),
        )
        .unwrap();
        assert_eq!(actual.coefficients, expected.coefficients);
        assert_eq!(actual.loglik, expected.loglik);
    }
    // Preserve the constructor's existing equal-endpoint semantics.
    input.time2.as_mut().unwrap()[0] = input.time[0];
    input.validate().unwrap();
}

#[test]
fn penalized_aft_checks_mutated_term_assignments() {
    let distribution = SurvregDistribution::from_name("gaussian", None).unwrap();
    for columns in [vec![], vec![0], vec![2], vec![1, 1]] {
        let mut input = SurvpenalData::try_new(data(), terms()).unwrap();
        input.terms[1].columns = columns;
        assert!(SurvpenalFit::fit(&input, &distribution, &SurvpenalOptions::default()).is_err());
        assert!(
            SurvpenalFitResult::fit(&input, &distribution, &SurvpenalOptions::default(), None)
                .is_err()
        );
    }
}

#[test]
fn malformed_aft_data_is_rejected_before_user_transforms() {
    use crate::regression::survreg_callbacks::SurvregTransformCallbacks;
    use std::sync::Arc;
    struct Unreachable;
    impl SurvregTransformCallbacks for Unreachable {
        fn transform(&self, _: &[f64]) -> SurvivalResult<Vec<f64>> {
            panic!("transform called before validation")
        }
        fn derivative(&self, _: &[f64]) -> SurvivalResult<Vec<f64>> {
            panic!("derivative called before validation")
        }
        fn inverse(&self, _: &[f64]) -> SurvivalResult<Vec<f64>> {
            panic!("inverse called before validation")
        }
    }
    let distribution = SurvregDistribution::from_name("gaussian", None)
        .unwrap()
        .with_transform(Arc::new(Unreachable));
    for (case, input) in malformed() {
        assert_fit_errors(case, input, &distribution);
    }
}
