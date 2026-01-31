import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from helpers import setup_survival_import

survival = setup_survival_import()

HAS_NUMPY = False
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    pass

HAS_SKLEARN_COMPAT = False
try:
    from survival.sklearn_compat import DeepSurvEstimator

    HAS_SKLEARN_COMPAT = True
except ImportError:
    pass

x = [
    1.0,
    0.5,
    0.2,
    0.8,
    0.3,
    0.1,
    0.6,
    0.7,
    0.4,
    0.4,
    0.2,
    0.8,
    0.9,
    0.1,
    0.3,
    0.3,
    0.8,
    0.5,
    0.7,
    0.4,
    0.6,
    0.2,
    0.6,
    0.9,
    0.5,
    0.9,
    0.1,
    1.0,
    0.0,
    0.7,
    0.1,
    0.5,
    0.2,
    0.6,
    0.3,
    0.4,
]
n_obs = 12
n_vars = 3
time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
status = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1]


def _make_config(**overrides):
    defaults = {
        "hidden_layers": [8, 4],
        "activation": survival.Activation("relu"),
        "dropout_rate": 0.0,
        "learning_rate": 0.01,
        "batch_size": 12,
        "n_epochs": 10,
        "l2_reg": 0.0,
        "seed": 42,
        "early_stopping_patience": None,
        "validation_fraction": 0.0,
    }
    defaults.update(overrides)
    return survival.DeepSurvConfig(**defaults)


def test_deepsurv_config_defaults():
    config = survival.DeepSurvConfig()
    assert config.hidden_layers == [64, 32]
    assert config.dropout_rate == pytest.approx(0.2)
    assert config.learning_rate == pytest.approx(0.001)


def test_deepsurv_config_custom():
    custom_config = survival.DeepSurvConfig(
        hidden_layers=[32, 16, 8],
        activation=survival.Activation("relu"),
        dropout_rate=0.1,
        learning_rate=0.01,
        batch_size=64,
        n_epochs=50,
        l2_reg=0.001,
        seed=42,
        early_stopping_patience=5,
        validation_fraction=0.2,
    )
    assert custom_config.hidden_layers == [32, 16, 8]
    assert custom_config.dropout_rate == pytest.approx(0.1)
    assert custom_config.early_stopping_patience == 5


def test_activation_enum():
    survival.Activation("relu")
    survival.Activation("selu")
    survival.Activation("tanh")


def test_activation_invalid():
    with pytest.raises(ValueError, match="Unknown activation"):
        survival.Activation("invalid")


def test_deepsurv_training():
    config = _make_config()
    model = survival.DeepSurv.fit(x, n_obs, n_vars, time, status, config)

    assert model.n_features == n_vars
    assert model.hidden_layers == [8, 4]
    assert len(model.train_loss) == 10
    assert len(model.unique_times) > 0
    assert len(model.baseline_hazard) > 0


def test_deepsurv_predict_risk():
    config = _make_config()
    model = survival.DeepSurv.fit(x, n_obs, n_vars, time, status, config)
    risk_scores = model.predict_risk(x, n_obs)
    assert len(risk_scores) == n_obs


def test_deepsurv_predict_survival():
    config = _make_config()
    model = survival.DeepSurv.fit(x, n_obs, n_vars, time, status, config)
    survival_probs = model.predict_survival(x, n_obs)
    assert len(survival_probs) == n_obs
    assert len(survival_probs[0]) == len(model.unique_times)
    for _i, surv in enumerate(survival_probs):
        assert all(0.0 <= s <= 1.0 for s in surv)
        assert surv == sorted(surv, reverse=True)


def test_deepsurv_predict_cumulative_hazard():
    config = _make_config()
    model = survival.DeepSurv.fit(x, n_obs, n_vars, time, status, config)
    cumhaz = model.predict_cumulative_hazard(x, n_obs)
    assert len(cumhaz) == n_obs
    for _i, ch in enumerate(cumhaz):
        assert all(c >= 0.0 for c in ch)
        assert ch == sorted(ch)


def test_deepsurv_predict_median_survival_time():
    config = _make_config()
    model = survival.DeepSurv.fit(x, n_obs, n_vars, time, status, config)
    median_times = model.predict_median_survival_time(x, n_obs)
    assert len(median_times) == n_obs


def test_deepsurv_predict_survival_time():
    config = _make_config()
    model = survival.DeepSurv.fit(x, n_obs, n_vars, time, status, config)
    percentile_times = model.predict_survival_time(x, n_obs, 0.75)
    assert len(percentile_times) == n_obs


def test_deepsurv_validation_and_early_stopping():
    config_with_val = survival.DeepSurvConfig(
        hidden_layers=[8, 4],
        activation=survival.Activation("selu"),
        dropout_rate=0.1,
        learning_rate=0.01,
        batch_size=6,
        n_epochs=50,
        l2_reg=0.001,
        seed=42,
        early_stopping_patience=5,
        validation_fraction=0.25,
    )
    model_val = survival.DeepSurv.fit(x, n_obs, n_vars, time, status, config_with_val)
    assert len(model_val.val_loss) > 0
    assert len(model_val.val_loss) == len(model_val.train_loss)


def test_deepsurv_mismatched_x_length():
    config = _make_config()
    bad_x = [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match=".*"):
        survival.DeepSurv.fit(bad_x, 10, 3, time, status, config)


def test_deepsurv_mismatched_time_length():
    config = _make_config()
    bad_time = [1.0, 2.0]
    with pytest.raises(ValueError, match=".*"):
        survival.DeepSurv.fit(x, n_obs, n_vars, bad_time, status, config)


def test_deepsurv_mismatched_predict_x():
    config = _make_config()
    model = survival.DeepSurv.fit(x, n_obs, n_vars, time, status, config)
    with pytest.raises(ValueError, match=".*"):
        model.predict_risk([1.0, 2.0], 5)


def test_deep_surv_convenience_function():
    model_func = survival.deep_surv(x, n_obs, n_vars, time, status)
    assert model_func.n_features == n_vars


@pytest.mark.skipif(
    not (HAS_NUMPY and HAS_SKLEARN_COMPAT),
    reason="numpy or survival.sklearn_compat not available",
)
def test_deepsurv_estimator():
    x_array = np.array(x).reshape(n_obs, n_vars)
    y = np.column_stack([time, status])

    estimator = DeepSurvEstimator(
        hidden_layers=[8, 4],
        activation="relu",
        dropout_rate=0.0,
        learning_rate=0.01,
        batch_size=12,
        n_epochs=10,
        seed=42,
        early_stopping_patience=None,
        validation_fraction=0.0,
    )

    estimator.fit(x_array, y)
    assert estimator.n_features_in_ == n_vars

    risk = estimator.predict(x_array)
    assert risk.shape == (n_obs,)

    times, surv = estimator.predict_survival_function(x_array)
    assert len(times) > 0
    assert surv.shape[0] == n_obs

    median = estimator.predict_median_survival_time(x_array)
    assert median.shape == (n_obs,)

    score = estimator.score(x_array, y)
    assert 0.0 <= score <= 1.0

    train_loss = estimator.train_loss
    assert len(train_loss) == 10
