use crate::ml::{
    Activation, DeepHitConfig, DeepSurv, DeepSurvConfig, GBSurvLoss, GradientBoostSurvival,
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

fn assert_config_error<T>(result: PyResult<T>, parameter: &str) {
    let error = match result {
        Ok(_) => panic!("expected a configuration error naming {parameter:?}"),
        Err(error) => error,
    };
    #[cfg(feature = "python")]
    Python::attach(|py| {
        assert!(error.is_instance_of::<pyo3::exceptions::PyValueError>(py));
    });
    assert!(
        error.to_string().contains(parameter),
        "expected {parameter:?} in {error}"
    );
}

fn construct_deep(config: &DeepSurvConfig) -> PyResult<DeepSurvConfig> {
    DeepSurvConfig::new(
        Some(config.hidden_layers.clone()),
        config.activation,
        config.dropout_rate,
        config.learning_rate,
        config.batch_size,
        config.n_epochs,
        config.l2_reg,
        config.seed,
        config.early_stopping_patience,
        config.validation_fraction,
    )
}

fn construct_boost(config: &GradientBoostSurvivalConfig) -> PyResult<GradientBoostSurvivalConfig> {
    GradientBoostSurvivalConfig::new(
        config.n_estimators,
        config.learning_rate,
        config.max_depth,
        config.min_samples_split,
        config.min_samples_leaf,
        config.subsample,
        config.max_features,
        config.loss,
        config.dropout_rate,
        config.seed,
    )
}

fn construct_forest(config: &SurvivalForestConfig) -> PyResult<SurvivalForestConfig> {
    SurvivalForestConfig::new(
        config.n_trees,
        config.max_depth,
        config.min_node_size,
        config.mtry,
        config.sample_fraction,
        config.split_rule,
        config.n_random_splits,
        config.seed,
        config.oob_error,
    )
}

macro_rules! invalid_values {
    ($cases:ident, $base:ident, $field:ident, $values:expr) => {
        for value in $values {
            let mut config = $base.clone();
            config.$field = value;
            $cases.push((config, stringify!($field)));
        }
    };
}

#[test]
fn deep_surv_construction_and_fits_validate_public_configuration_fields() {
    Python::initialize();
    Python::attach(|py| {
        let base = construct_deep(&deep_config()).unwrap();
        let input = training_input();
        let mut cases = Vec::new();
        let nonfinite = [f64::NAN, f64::NEG_INFINITY, f64::INFINITY];
        invalid_values!(cases, base, learning_rate, nonfinite);
        invalid_values!(
            cases,
            base,
            learning_rate,
            [-0.1, 0.0, 1e-300, f32::MAX as f64 * 2.0]
        );
        invalid_values!(cases, base, l2_reg, nonfinite);
        invalid_values!(cases, base, l2_reg, [-0.1, f32::MAX as f64 * 2.0]);
        invalid_values!(cases, base, dropout_rate, nonfinite);
        invalid_values!(cases, base, dropout_rate, [-0.1, 1.0, 1.1]);
        invalid_values!(cases, base, validation_fraction, nonfinite);
        invalid_values!(cases, base, validation_fraction, [-0.1, 1.0, 1.1]);
        invalid_values!(cases, base, batch_size, [0]);
        invalid_values!(cases, base, n_epochs, [0, usize::MAX]);
        let product_overflow_width = 1usize << (usize::BITS / 2);
        invalid_values!(
            cases,
            base,
            hidden_layers,
            [
                vec![0],
                vec![2, 0],
                vec![0, 2],
                vec![usize::MAX],
                vec![product_overflow_width, product_overflow_width],
                vec![2, product_overflow_width, product_overflow_width],
            ]
        );

        for (config, parameter) in cases {
            assert_config_error(construct_deep(&config), parameter);
            // A valid constructor does not protect a later fit: all fields
            // above were changed directly through the public Rust API.
            assert_config_error(
                DeepSurv::fit(
                    py,
                    input.x.clone(),
                    input.n_obs,
                    input.n_vars,
                    input.time.clone(),
                    input.status.clone(),
                    &config,
                ),
                parameter,
            );
            assert_config_error(
                deep_surv(
                    py,
                    input.x.clone(),
                    input.n_obs,
                    input.n_vars,
                    input.time.clone(),
                    input.status.clone(),
                    Some(&config),
                ),
                parameter,
            );
        }
    });
}

#[test]
fn gradient_boost_construction_and_fits_validate_public_configuration_fields() {
    Python::initialize();
    Python::attach(|py| {
        let base = construct_boost(&boost_config()).unwrap();
        let input = training_input();
        let mut cases = Vec::new();
        let invalid_rates = [f64::NAN, f64::NEG_INFINITY, f64::INFINITY, -0.1, 0.0, 1.1];
        invalid_values!(cases, base, n_estimators, [0, usize::MAX]);
        invalid_values!(cases, base, min_samples_leaf, [0]);
        invalid_values!(cases, base, learning_rate, invalid_rates);
        invalid_values!(cases, base, subsample, invalid_rates);

        for (config, parameter) in cases {
            assert_config_error(construct_boost(&config), parameter);
            assert_config_error(
                GradientBoostSurvival::fit(
                    py,
                    input.x.clone(),
                    input.n_obs,
                    input.n_vars,
                    input.time.clone(),
                    input.status.clone(),
                    &config,
                ),
                parameter,
            );
            assert_config_error(
                gradient_boost_survival(
                    py,
                    input.x.clone(),
                    input.n_obs,
                    input.n_vars,
                    input.time.clone(),
                    input.status.clone(),
                    Some(&config),
                ),
                parameter,
            );
        }
    });
}

#[test]
fn survival_forest_construction_and_fits_validate_public_configuration_fields() {
    Python::initialize();
    Python::attach(|py| {
        let base = construct_forest(&forest_config()).unwrap();
        let input = training_input();
        let mut cases = Vec::new();
        invalid_values!(cases, base, n_trees, [0, usize::MAX]);
        invalid_values!(cases, base, n_random_splits, [0]);
        invalid_values!(
            cases,
            base,
            sample_fraction,
            [f64::NAN, f64::NEG_INFINITY, f64::INFINITY, -0.1, 0.0, 1.1]
        );

        for (config, parameter) in cases {
            assert_config_error(construct_forest(&config), parameter);
            assert_config_error(SurvivalForest::fit_typed(py, &input, &config), parameter);
            assert_config_error(
                SurvivalForest::fit(
                    py,
                    input.x.clone(),
                    input.n_obs,
                    input.n_vars,
                    input.time.clone(),
                    input.status.clone(),
                    &config,
                ),
                parameter,
            );
            assert_config_error(
                survival_forest(
                    py,
                    input.x.clone(),
                    input.n_obs,
                    input.n_vars,
                    input.time.clone(),
                    input.status.clone(),
                    Some(&config),
                ),
                parameter,
            );
        }
    });
}

#[test]
fn deep_surv_rejects_network_dimensions_that_depend_on_training_data() {
    Python::initialize();
    Python::attach(|py| {
        let f32_capacity = isize::MAX as usize / size_of::<f32>();
        for (n_vars, width, batch_size, validation_fraction) in [
            // The first weight matrix exceeds capacity despite a valid width.
            (3, f32_capacity / 3 + 1, 1, 0.0),
            // Weights fit, but a six-row training activation matrix does not.
            (2, f32_capacity / 6 + 1, 6, 0.0),
            // One-row batches fit, but the three-row validation matrix does not.
            (2, f32_capacity / 3 + 1, 1, 0.5),
        ] {
            let mut input = training_input();
            input.n_vars = n_vars;
            input.x = vec![0.0; input.n_obs * n_vars];
            let config = construct_deep(&DeepSurvConfig {
                hidden_layers: vec![width],
                batch_size,
                validation_fraction,
                ..deep_config()
            })
            .unwrap();

            // These checks must happen before any network buffer is allocated.
            assert_config_error(
                DeepSurv::fit(
                    py,
                    input.x.clone(),
                    input.n_obs,
                    input.n_vars,
                    input.time.clone(),
                    input.status.clone(),
                    &config,
                ),
                "hidden_layers",
            );
            assert_config_error(
                deep_surv(
                    py,
                    input.x,
                    input.n_obs,
                    input.n_vars,
                    input.time,
                    input.status,
                    Some(&config),
                ),
                "hidden_layers",
            );
        }
    });
}

#[test]
fn deephit_sigma_requires_a_finite_positive_value() {
    Python::initialize();
    Python::attach(|_| {
        for sigma in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.1] {
            assert_config_error(
                DeepHitConfig::new(
                    Some(vec![2]),
                    Some(vec![2]),
                    2,
                    1,
                    0.0,
                    0.2,
                    sigma,
                    0.01,
                    6,
                    1,
                    0.0,
                    Some(42),
                    None,
                    0.0,
                    false,
                ),
                "sigma",
            );
        }
    });
}

#[test]
fn deep_surv_preserves_empty_hidden_layers_and_zero_patience() {
    Python::initialize();
    Python::attach(|py| {
        let input = training_input();
        let config = construct_deep(&DeepSurvConfig {
            hidden_layers: vec![],
            early_stopping_patience: Some(0),
            l2_reg: 1e-300,
            ..deep_config()
        })
        .unwrap();
        construct_deep(&DeepSurvConfig {
            learning_rate: f32::from_bits(1) as f64,
            l2_reg: 1e-300,
            ..deep_config()
        })
        .unwrap();
        let model = deep_surv(
            py,
            input.x.clone(),
            input.n_obs,
            input.n_vars,
            input.time,
            input.status,
            Some(&config),
        )
        .unwrap();
        assert!(model.get_hidden_layers().is_empty());
        assert_eq!(model.get_n_features(), input.n_vars);
        let risk = model.predict_risk(input.x, input.n_obs).unwrap();
        assert_eq!(risk.len(), input.n_obs);
        assert!(risk.iter().all(|value| value.is_finite()));

        // These bounds are properties of the backend representation, not an
        // arbitrary upper limit such as one for the neural learning rate.
        construct_deep(&DeepSurvConfig {
            learning_rate: f32::MAX as f64,
            l2_reg: f32::MAX as f64,
            ..deep_config()
        })
        .unwrap();
    });
}

#[test]
fn tree_configuration_boundaries_preserve_constant_models() {
    Python::initialize();
    Python::attach(|py| {
        let input = training_input();
        for (max_depth, minimum) in [(0, 1), (2, usize::MAX)] {
            let boost = construct_boost(&GradientBoostSurvivalConfig {
                max_depth,
                min_samples_leaf: minimum,
                learning_rate: 1.0,
                subsample: 1.0,
                ..boost_config()
            })
            .unwrap();
            let boost = GradientBoostSurvival::fit(
                py,
                input.x.clone(),
                input.n_obs,
                input.n_vars,
                input.time.clone(),
                input.status.clone(),
                &boost,
            )
            .unwrap();
            let forest = construct_forest(&SurvivalForestConfig {
                max_depth: Some(max_depth),
                min_node_size: minimum,
                sample_fraction: 1.0,
                ..forest_config()
            })
            .unwrap();
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

#[test]
fn tree_configurations_preserve_permissive_split_and_node_minimums() {
    Python::initialize();
    Python::attach(|py| {
        let input = training_input();
        for min_samples_split in [0, 1] {
            let config = construct_boost(&GradientBoostSurvivalConfig {
                min_samples_split,
                ..boost_config()
            })
            .unwrap();
            let model = GradientBoostSurvival::fit(
                py,
                input.x.clone(),
                input.n_obs,
                input.n_vars,
                input.time.clone(),
                input.status.clone(),
                &config,
            )
            .unwrap();
            let risk = model.predict_risk(input.x.clone(), input.n_obs).unwrap();
            assert_eq!(risk.len(), input.n_obs);
            assert!(risk.iter().all(|value| value.is_finite()));
        }
        let config = construct_forest(&SurvivalForestConfig {
            min_node_size: 0,
            ..forest_config()
        })
        .unwrap();
        let model = SurvivalForest::fit_typed(py, &input, &config).unwrap();
        let risk = model.predict_risk(input.x, input.n_obs).unwrap();
        assert_eq!(risk.len(), input.n_obs);
        assert!(risk.iter().all(|value| value.is_finite()));
    });
}

#[test]
fn checked_tree_configurations_preserve_zero_feature_fits() {
    Python::initialize();
    Python::attach(|py| {
        let mut input = training_input();
        input.x.clear();
        input.n_vars = 0;
        let forest = survival_forest(
            py,
            vec![],
            input.n_obs,
            0,
            input.time.clone(),
            input.status.clone(),
            Some(&construct_forest(&forest_config()).unwrap()),
        )
        .unwrap();
        let boost = gradient_boost_survival(
            py,
            vec![],
            input.n_obs,
            0,
            input.time,
            input.status,
            Some(&construct_boost(&boost_config()).unwrap()),
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
    });
}
