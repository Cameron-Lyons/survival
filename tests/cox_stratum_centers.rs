use ndarray::Array2;
use survival::regression::{
    CoxNewData, CoxPHFit, CoxPrediction, CoxphData, CoxphOptions, PredictReference, TieMethod,
};

const STRATA: [i32; 8] = [-4, 19, -4, 19, -4, 19, -4, 19];

fn model(constants: [f64; 2], weighted: bool, counting: bool, method: TieMethod) -> CoxPHFit {
    let x = Array2::from_shape_fn((8, 2), |(row, column)| {
        if column == 0 {
            constants[row % 2]
        } else {
            (row / 2) as f64
        }
    });
    let data = CoxphData::try_new(
        vec![1., 1., 2., 2., 3., 3., 4., 4.],
        counting.then(|| vec![0.; 8]),
        vec![1, 1, 1, 1, 1, 1, 0, 0],
        x,
        weighted.then(|| vec![1., 2., 2., 1., 1., 2., 2., 1.]),
        Some(STRATA.to_vec()),
        None,
    );
    // The first covariate is constant within each stratum; its zero
    // coefficient leaves only the ordinary second covariate's contribution.
    CoxPHFit::fit(
        data.unwrap(),
        CoxphOptions {
            method,
            init: Some(vec![0., 0.25]),
            nocenter: Some(constants.to_vec()),
            iter_max: 0,
            robust: Some(false),
            ..Default::default()
        },
    )
    .unwrap()
}

fn prediction_rows(constants: [f64; 2]) -> CoxNewData {
    let z = [2.5, -1., 5., 0.25];
    CoxNewData::try_new(
        Array2::from_shape_fn((4, 2), |(row, column)| {
            if column == 0 {
                constants[1 - row % 2]
            } else {
                z[row]
            }
        }),
        Some(vec![19, -4, 19, -4]),
        None,
        None,
        None,
    )
    .unwrap()
}

fn close(actual: f64, expected: f64) {
    assert!(actual.is_finite() && expected.is_finite());
    assert!(
        (actual - expected).abs() < 2e-12 * (1. + expected.abs()),
        "{actual} != {expected}"
    );
}

fn equal_prediction(actual: CoxPrediction, expected: CoxPrediction) {
    assert_eq!(actual.fit.len(), expected.fit.len());
    for (actual, expected) in actual.fit.into_iter().zip(expected.fit) {
        close(actual, expected);
    }
    assert_eq!(actual.se_fit.is_some(), expected.se_fit.is_some());
    if let (Some(actual), Some(expected)) = (actual.se_fit, expected.se_fit) {
        for (actual, expected) in actual.into_iter().zip(expected) {
            close(actual, expected);
        }
    }
}

#[test]
fn stratum_prediction_is_invariant_to_large_constant_covariates() {
    let ordinary = [1., -1.];
    for method in [TieMethod::Breslow, TieMethod::Efron, TieMethod::Exact] {
        for weighted in [false, true] {
            if weighted && method == TieMethod::Exact {
                continue; // Exact partial likelihood does not support case weights.
            }
            for counting in [false, true] {
                let reference = model(ordinary, weighted, counting, method);
                let reference_rows = prediction_rows(ordinary);
                for constants in [[5e307, -5e307], [-1e308, 1e308], [1e308, -1.]] {
                    let fit = model(constants, weighted, counting, method);
                    let newdata = prediction_rows(constants);
                    for (rows, expected_rows) in
                        [(None, None), (Some(&newdata), Some(&reference_rows))]
                    {
                        equal_prediction(
                            fit.predict_lp(rows, true, PredictReference::Strata)
                                .unwrap(),
                            reference
                                .predict_lp(expected_rows, true, PredictReference::Strata)
                                .unwrap(),
                        );
                        equal_prediction(
                            fit.predict_risk(rows, true, PredictReference::Strata)
                                .unwrap(),
                            reference
                                .predict_risk(expected_rows, true, PredictReference::Strata)
                                .unwrap(),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn stratum_prediction_centers_allow_overflowing_case_weight_totals() {
    let mut fit = model([1., -1.], true, false, TieMethod::Efron);
    let expected = fit
        .predict_lp(None, true, PredictReference::Strata)
        .unwrap();
    // Public fitted weights are used for centering at prediction time. A
    // common positive scaling cannot change any stratum's weighted centre.
    for weight in &mut fit.weights {
        *weight *= 5e307;
    }
    equal_prediction(
        fit.predict_lp(None, true, PredictReference::Strata)
            .unwrap(),
        expected,
    );
}
