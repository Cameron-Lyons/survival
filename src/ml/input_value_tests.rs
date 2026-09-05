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

macro_rules! fit_input {
    ($fit:path, $py:expr, $input:ident, $config:expr) => {
        $fit(
            $py,
            $input.x.clone(),
            $input.n_obs,
            $input.n_vars,
            $input.time.clone(),
            $input.status.clone(),
            $config,
        )
    };
}

fn assert_value_error<T>(result: PyResult<T>, field: &str, index: usize) -> String {
    let error = match result {
        Ok(_) => panic!("expected an input error for {field} at index {index}"),
        Err(error) => error,
    };
    #[cfg(feature = "python")]
    Python::attach(|py| {
        assert!(error.is_instance_of::<pyo3::exceptions::PyValueError>(py));
    });
    let message = error.to_string();
    assert!(
        message.contains(field),
        "missing field {field:?}: {message}"
    );
    assert!(
        message.contains(&format!("index {index}")),
        "missing offending index: {message}"
    );
    message
}

#[test]
fn public_fits_and_typed_forest_inputs_reject_invalid_values() {
    Python::initialize();
    Python::attach(|py| {
        let deep = deep_config();
        let boost = boost_config();
        let forest = forest_config();
        let mut cases = Vec::new();
        for value in [f64::NAN, f64::NEG_INFINITY, f64::INFINITY] {
            let mut input = training_input();
            input.x[11] = value;
            cases.push((input, "x", 11));
            let mut input = training_input();
            input.time[3] = value;
            cases.push((input, "time", 3));
        }
        for value in [-1, 2, i32::MIN, i32::MAX] {
            let mut input = training_input();
            input.status[4] = value;
            cases.push((input, "status", 4));
        }
        for (input, field, index) in cases {
            assert_value_error(fit_input!(DeepSurv::fit, py, input, &deep), field, index);
            assert_value_error(fit_input!(deep_surv, py, input, Some(&deep)), field, index);
            assert_value_error(
                fit_input!(GradientBoostSurvival::fit, py, input, &boost),
                field,
                index,
            );
            assert_value_error(
                fit_input!(gradient_boost_survival, py, input, Some(&boost)),
                field,
                index,
            );
            assert_value_error(
                fit_input!(SurvivalForest::fit, py, input, &forest),
                field,
                index,
            );
            assert_value_error(
                fit_input!(survival_forest, py, input, Some(&forest)),
                field,
                index,
            );
            // Public fields bypass new(), so typed fitting must validate them too.
            assert_value_error(SurvivalForest::fit_typed(py, &input, &forest), field, index);
            assert_value_error(
                SurvivalForestInput::new(
                    input.x,
                    input.n_obs,
                    input.n_vars,
                    input.time,
                    input.status,
                ),
                field,
                index,
            );
        }
    });
}

#[test]
fn deep_training_checks_f32_conversion_before_splitting_rows() {
    Python::initialize();
    Python::attach(|py| {
        let config = DeepSurvConfig {
            validation_fraction: 0.5,
            ..deep_config()
        };
        // Cover every row so narrowing is checked in both training and validation.
        for row in 0..6 {
            for value in [-1e100, 1e100] {
                let mut input = training_input();
                let index = row * input.n_vars + 1;
                input.x[index] = value;
                for message in [
                    assert_value_error(fit_input!(DeepSurv::fit, py, input, &config), "x", index),
                    assert_value_error(fit_input!(deep_surv, py, input, Some(&config)), "x", index),
                ] {
                    assert!(
                        message.contains("f32"),
                        "missing conversion detail: {message}"
                    );
                }
            }
        }
    });
}

macro_rules! check_prediction_values {
    ($model:expr) => {{
        let model = &$model;
        for value in [f64::NAN, f64::NEG_INFINITY, f64::INFINITY] {
            let x = vec![0.0, 1.0, 0.5, value];
            assert_value_error(model.predict_risk(x.clone(), 2), "x", 3);
            assert_value_error(model.predict_survival(x.clone(), 2), "x", 3);
            assert_value_error(model.predict_cumulative_hazard(x.clone(), 2), "x", 3);
            assert_value_error(model.predict_survival_time(x.clone(), 2, 0.5), "x", 3);
            assert_value_error(model.predict_median_survival_time(x, 2), "x", 3);
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
    }};
}

#[test]
fn fitted_models_check_features_on_every_prediction_route() {
    Python::initialize();
    Python::attach(|py| {
        let input = training_input();
        check_prediction_values!(fit_input!(DeepSurv::fit, py, input, &deep_config()).unwrap());
        check_prediction_values!(
            fit_input!(GradientBoostSurvival::fit, py, input, &boost_config()).unwrap()
        );
        check_prediction_values!(SurvivalForest::fit_typed(py, &input, &forest_config()).unwrap());
    });
}

#[test]
fn finite_signed_times_and_all_censored_samples_remain_valid() {
    Python::initialize();
    Python::attach(|py| {
        for all_censored in [false, true] {
            let mut input = training_input();
            input.time.iter_mut().for_each(|time| *time -= 4.0);
            // Ordinary rounding and underflow to f32 remain valid neural inputs.
            input.x[0] = 1e-300;
            input.x[11] = 1.0 / 3.0;
            if all_censored {
                input.status.fill(0);
            }
            let deep = fit_input!(deep_surv, py, input, Some(&deep_config())).unwrap();
            let boost =
                fit_input!(gradient_boost_survival, py, input, Some(&boost_config())).unwrap();
            let forest = fit_input!(survival_forest, py, input, Some(&forest_config())).unwrap();
            for times in [
                &deep.unique_times,
                &boost.unique_times,
                &forest.get_unique_times(),
            ] {
                assert_eq!(times, &input.time);
            }
            for risk in [
                deep.predict_risk(input.x.clone(), input.n_obs).unwrap(),
                boost.predict_risk(input.x.clone(), input.n_obs).unwrap(),
                forest.predict_risk(input.x.clone(), input.n_obs).unwrap(),
            ] {
                assert_eq!(risk.len(), input.n_obs);
                assert!(risk.iter().all(|value| value.is_finite()));
            }
            for survival in [
                deep.predict_survival(input.x.clone(), input.n_obs).unwrap(),
                boost
                    .predict_survival(input.x.clone(), input.n_obs)
                    .unwrap(),
                forest
                    .predict_survival(input.x.clone(), input.n_obs)
                    .unwrap(),
            ] {
                assert_eq!(survival.len(), input.n_obs);
                assert!(survival.iter().all(|curve| !curve.is_empty()
                    && curve.iter().all(|value| (0.0..=1.0).contains(value))));
                if all_censored {
                    assert!(survival.iter().flatten().all(|&value| value == 1.0));
                }
            }
        }
    });
}

#[test]
fn deep_predictions_preserve_finite_features_beyond_the_training_dtype() {
    Python::initialize();
    Python::attach(|py| {
        let input = training_input();
        // Tanh keeps this fitted model's f64 inference numerically bounded.
        let model = fit_input!(DeepSurv::fit, py, input, &deep_config()).unwrap();
        let x = vec![1e100, 0.0, -1e100, 0.0];
        let risk = model.predict_risk(x.clone(), 2).unwrap();
        assert_eq!(risk.len(), 2);
        assert!(risk.iter().all(|value| value.is_finite()));
        let survival = model.predict_survival(x.clone(), 2).unwrap();
        assert!(
            survival
                .iter()
                .flatten()
                .all(|value| (0.0..=1.0).contains(value))
        );
        let hazard = model.predict_cumulative_hazard(x.clone(), 2).unwrap();
        assert!(hazard.iter().flatten().all(|value| value.is_finite()));
        assert_eq!(
            model
                .predict_survival_time(x.clone(), 2, 0.5)
                .unwrap()
                .len(),
            2
        );
        assert_eq!(model.predict_median_survival_time(x, 2).unwrap().len(), 2);
    });
}

#[test]
fn tree_fits_preserve_large_finite_features_and_zero_feature_models() {
    Python::initialize();
    Python::attach(|py| {
        for n_vars in [0, 2] {
            let mut input = training_input();
            input.n_vars = n_vars;
            input.x = (0..input.n_obs * n_vars)
                .map(|index| if index % 2 == 0 { -f64::MAX } else { f64::MAX })
                .collect();
            let boost = GradientBoostSurvivalConfig {
                max_depth: 0,
                ..boost_config()
            };
            let forest = SurvivalForestConfig {
                max_depth: Some(0),
                ..forest_config()
            };
            let boost = fit_input!(GradientBoostSurvival::fit, py, input, &boost).unwrap();
            let forest = SurvivalForest::fit_typed(py, &input, &forest).unwrap();
            for risk in [
                boost.predict_risk(input.x.clone(), input.n_obs).unwrap(),
                forest.predict_risk(input.x.clone(), input.n_obs).unwrap(),
            ] {
                assert_eq!(risk.len(), input.n_obs);
                assert!(risk.iter().all(|value| value.is_finite()));
                assert!(risk.windows(2).all(|pair| pair[0] == pair[1]));
            }
        }
    });
}
