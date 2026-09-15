use crate::ml::{
    Activation, DeepSurv, DeepSurvConfig, GBSurvLoss, GradientBoostSurvival,
    GradientBoostSurvivalConfig, SurvivalForest, SurvivalForestConfig, SurvivalForestInput,
    deep_surv, gradient_boost_survival, survival_forest,
};
use pyo3::prelude::*;

fn deep_config() -> DeepSurvConfig {
    DeepSurvConfig {
        hidden_layers: vec![2],
        activation: Activation::Tanh,
        dropout_rate: 0.0,
        learning_rate: 0.01,
        batch_size: 6,
        n_epochs: 1,
        l2_reg: 0.0,
        seed: Some(42),
        early_stopping_patience: None,
        validation_fraction: 0.0,
    }
}

fn boost_config() -> GradientBoostSurvivalConfig {
    GradientBoostSurvivalConfig {
        n_estimators: 2,
        learning_rate: 0.1,
        max_depth: 2,
        min_samples_split: 2,
        min_samples_leaf: 1,
        subsample: 1.0,
        max_features: None,
        loss: GBSurvLoss::CoxPH,
        dropout_rate: 0.0,
        seed: Some(42),
    }
}

fn forest_config() -> SurvivalForestConfig {
    SurvivalForestConfig {
        n_trees: 2,
        max_depth: Some(2),
        min_node_size: 1,
        sample_fraction: 1.0,
        seed: Some(42),
        oob_error: false,
        ..SurvivalForestConfig::default()
    }
}

fn training_input() -> SurvivalForestInput {
    SurvivalForestInput {
        x: vec![1.0, 0.5, 0.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 1.5, 0.5],
        n_obs: 6,
        n_vars: 2,
        time: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        status: vec![1, 1, 0, 1, 0, 1],
    }
}

fn assert_value_error<T>(result: PyResult<T>, expected: &str) {
    let error = match result {
        Ok(_) => panic!("expected a dimension error containing {expected:?}"),
        Err(error) => error,
    };
    #[cfg(feature = "python")]
    Python::attach(|py| {
        assert!(error.is_instance_of::<pyo3::exceptions::PyValueError>(py));
    });
    assert!(
        error.to_string().contains(expected),
        "expected {expected:?} in {error}"
    );
}

fn malformed_inputs() -> Vec<(SurvivalForestInput, &'static str)> {
    let valid = training_input();
    let mut cases = Vec::new();
    for n_vars in [0, 2] {
        cases.push((
            SurvivalForestInput {
                x: vec![],
                n_obs: 0,
                n_vars,
                time: vec![],
                status: vec![],
            },
            "n_obs must be positive",
        ));
    }
    for n_obs in [usize::MAX / 2 + 1, usize::MAX] {
        // The first product wraps to zero in release builds, so an ordinary
        // equality check would accept the empty feature buffer.
        cases.push((
            SurvivalForestInput {
                x: vec![],
                n_obs,
                n_vars: 2,
                time: vec![],
                status: vec![],
            },
            "n_obs * n_vars overflows",
        ));
    }
    for len in [valid.x.len() - 1, valid.x.len() + 1] {
        cases.push((
            SurvivalForestInput {
                x: vec![0.0; len],
                ..valid.clone()
            },
            "x length must equal n_obs * n_vars",
        ));
    }
    for len in [valid.n_obs - 1, valid.n_obs + 1] {
        cases.push((
            SurvivalForestInput {
                time: vec![1.0; len],
                ..valid.clone()
            },
            "time and status must have length n_obs",
        ));
        cases.push((
            SurvivalForestInput {
                status: vec![1; len],
                ..valid.clone()
            },
            "time and status must have length n_obs",
        ));
    }
    cases
}

#[test]
fn public_fits_reject_malformed_training_dimensions() {
    Python::initialize();
    Python::attach(|py| {
        let deep = deep_config();
        let boost = boost_config();
        let forest = forest_config();
        for (input, message) in malformed_inputs() {
            assert_value_error(
                DeepSurv::fit(
                    py,
                    input.x.clone(),
                    input.n_obs,
                    input.n_vars,
                    input.time.clone(),
                    input.status.clone(),
                    &deep,
                ),
                message,
            );
            assert_value_error(
                deep_surv(
                    py,
                    input.x.clone(),
                    input.n_obs,
                    input.n_vars,
                    input.time.clone(),
                    input.status.clone(),
                    Some(&deep),
                ),
                message,
            );
            assert_value_error(
                GradientBoostSurvival::fit(
                    py,
                    input.x.clone(),
                    input.n_obs,
                    input.n_vars,
                    input.time.clone(),
                    input.status.clone(),
                    &boost,
                ),
                message,
            );
            assert_value_error(
                gradient_boost_survival(
                    py,
                    input.x.clone(),
                    input.n_obs,
                    input.n_vars,
                    input.time.clone(),
                    input.status.clone(),
                    Some(&boost),
                ),
                message,
            );
            assert_value_error(
                SurvivalForest::fit(
                    py,
                    input.x.clone(),
                    input.n_obs,
                    input.n_vars,
                    input.time.clone(),
                    input.status.clone(),
                    &forest,
                ),
                message,
            );
            assert_value_error(
                survival_forest(
                    py,
                    input.x,
                    input.n_obs,
                    input.n_vars,
                    input.time,
                    input.status,
                    Some(&forest),
                ),
                message,
            );
        }
    });
}

#[test]
fn typed_forest_inputs_cannot_bypass_dimension_validation() {
    Python::initialize();
    Python::attach(|py| {
        let config = forest_config();
        for (input, message) in malformed_inputs() {
            // Rust callers can construct or mutate the public fields without
            // going through new(), so fit_typed must recheck them.
            assert_value_error(SurvivalForest::fit_typed(py, &input, &config), message);
            assert_value_error(
                SurvivalForestInput::new(
                    input.x,
                    input.n_obs,
                    input.n_vars,
                    input.time,
                    input.status,
                ),
                message,
            );
        }
    });
}

#[test]
fn deep_surv_rejects_training_without_features() {
    Python::initialize();
    Python::attach(|py| {
        let input = training_input();
        let config = deep_config();
        assert_value_error(
            DeepSurv::fit(
                py,
                vec![],
                input.n_obs,
                0,
                input.time.clone(),
                input.status.clone(),
                &config,
            ),
            "n_vars must be positive",
        );
        assert_value_error(
            deep_surv(
                py,
                vec![],
                input.n_obs,
                0,
                input.time,
                input.status,
                Some(&config),
            ),
            "n_vars must be positive",
        );
    });
}

macro_rules! check_prediction_dimensions {
    ($model:expr) => {{
        let model = $model;
        for (x, n_new, message) in [
            (vec![0.0], 0, "x_new dimensions don't match"),
            (vec![0.0], 1, "x_new dimensions don't match"),
            (vec![0.0; 5], 2, "x_new dimensions don't match"),
            (vec![], usize::MAX / 2 + 1, "n_new * n_vars overflows"),
            (vec![], usize::MAX, "n_new * n_vars overflows"),
        ] {
            assert_value_error(model.predict_risk(x.clone(), n_new), message);
            assert_value_error(model.predict_survival(x.clone(), n_new), message);
            assert_value_error(model.predict_cumulative_hazard(x.clone(), n_new), message);
            assert_value_error(model.predict_survival_time(x.clone(), n_new, 0.5), message);
            assert_value_error(model.predict_median_survival_time(x, n_new), message);
        }
        assert!(model.predict_risk(vec![], 0).unwrap().is_empty());
        assert!(model.predict_survival(vec![], 0).unwrap().is_empty());
        assert!(
            model
                .predict_cumulative_hazard(vec![], 0)
                .unwrap()
                .is_empty()
        );
        assert!(
            model
                .predict_survival_time(vec![], 0, 0.5)
                .unwrap()
                .is_empty()
        );
        assert!(
            model
                .predict_median_survival_time(vec![], 0)
                .unwrap()
                .is_empty()
        );

        let x = vec![0.0, 1.0, 0.5, 0.5];
        let risk = model.predict_risk(x.clone(), 2).unwrap();
        assert_eq!(risk.len(), 2);
        assert!(risk.iter().all(|value| value.is_finite()));
        let survival = model.predict_survival(x, 2).unwrap();
        assert_eq!(survival.len(), 2);
        assert!(survival.iter().all(|curve| {
            !curve.is_empty()
                && curve
                    .iter()
                    .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
        }));
    }};
}

#[test]
fn deep_surv_prediction_dimensions_are_checked_before_inference() {
    Python::initialize();
    Python::attach(|py| {
        let input = training_input();
        let model = DeepSurv::fit(
            py,
            input.x,
            input.n_obs,
            input.n_vars,
            input.time,
            input.status,
            &deep_config(),
        )
        .unwrap();
        check_prediction_dimensions!(model);
    });
}

#[test]
fn gradient_boost_prediction_dimensions_are_checked_before_inference() {
    Python::initialize();
    Python::attach(|py| {
        let input = training_input();
        let model = GradientBoostSurvival::fit(
            py,
            input.x,
            input.n_obs,
            input.n_vars,
            input.time,
            input.status,
            &boost_config(),
        )
        .unwrap();
        check_prediction_dimensions!(model);
    });
}

#[test]
fn survival_forest_prediction_dimensions_are_checked_before_inference() {
    Python::initialize();
    Python::attach(|py| {
        let model = SurvivalForest::fit_typed(py, &training_input(), &forest_config()).unwrap();
        check_prediction_dimensions!(model);
    });
}

#[test]
fn tree_models_preserve_training_and_predictions_without_features() {
    Python::initialize();
    Python::attach(|py| {
        let input = training_input();
        let forest_input = SurvivalForestInput::new(
            vec![],
            input.n_obs,
            0,
            input.time.clone(),
            input.status.clone(),
        )
        .unwrap();
        let forest = SurvivalForest::fit_typed(py, &forest_input, &forest_config()).unwrap();
        let boost = gradient_boost_survival(
            py,
            vec![],
            input.n_obs,
            0,
            input.time,
            input.status,
            Some(&boost_config()),
        )
        .unwrap();

        assert!(forest.variable_importance.is_empty());
        assert!(boost.feature_importance.is_empty());
        for risk in [
            forest.predict_risk(vec![], 3).unwrap(),
            boost.predict_risk(vec![], 3).unwrap(),
        ] {
            assert_eq!(risk.len(), 3);
            assert!(risk.iter().all(|value| value.is_finite()));
            assert!(risk.windows(2).all(|pair| pair[0] == pair[1]));
        }
        for survival in [
            forest.predict_survival(vec![], 3).unwrap(),
            boost.predict_survival(vec![], 3).unwrap(),
        ] {
            assert_eq!(survival.len(), 3);
            assert!(!survival[0].is_empty());
            assert!(survival.windows(2).all(|pair| pair[0] == pair[1]));
            assert!(survival[0].iter().all(|value| (0.0..=1.0).contains(value)));
            assert!(survival[0].windows(2).all(|pair| pair[1] <= pair[0]));
        }
        assert!(forest.predict_risk(vec![], 0).unwrap().is_empty());
        assert!(boost.predict_risk(vec![], 0).unwrap().is_empty());
        assert_value_error(
            forest.predict_risk(vec![1.0], 3),
            "x_new dimensions don't match",
        );
        assert_value_error(
            boost.predict_risk(vec![1.0], 3),
            "x_new dimensions don't match",
        );
        for n_new in [isize::MAX as usize / size_of::<Vec<f64>>() + 1, usize::MAX] {
            // Zero input columns make the feature product zero, but the
            // prediction rows still require an addressable output buffer.
            let message = "n_new is too large for prediction output";
            assert_value_error(forest.predict_risk(vec![], n_new), message);
            assert_value_error(forest.predict_survival(vec![], n_new), message);
            assert_value_error(forest.predict_cumulative_hazard(vec![], n_new), message);
            assert_value_error(forest.predict_survival_time(vec![], n_new, 0.5), message);
            assert_value_error(forest.predict_median_survival_time(vec![], n_new), message);
            assert_value_error(boost.predict_risk(vec![], n_new), message);
            assert_value_error(boost.predict_survival(vec![], n_new), message);
            assert_value_error(boost.predict_cumulative_hazard(vec![], n_new), message);
            assert_value_error(boost.predict_survival_time(vec![], n_new, 0.5), message);
            assert_value_error(boost.predict_median_survival_time(vec![], n_new), message);
        }
    });
}
