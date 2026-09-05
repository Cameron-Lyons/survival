import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
core = survival._survival
MODEL_NAMES = ("DeepSurv", "GradientBoostSurvival", "SurvivalForest")
pytestmark = pytest.mark.skipif(
    not all(hasattr(core, name) and hasattr(core, f"{name}Config") for name in MODEL_NAMES),
    reason="native input-value tests require the Rust ml feature",
)

FIT_ROUTES = tuple((name, route) for name in MODEL_NAMES for route in ("class", "function"))
if hasattr(core, "SurvivalForestInput") and hasattr(
    getattr(core, "SurvivalForest", None), "fit_typed"
):
    FIT_ROUTES += (("SurvivalForest", "typed"),)
VALIDATION_ROUTES = FIT_ROUTES + (
    (("SurvivalForest", "input"),) if hasattr(core, "SurvivalForestInput") else ()
)
PREDICTION_ROUTES = (
    "predict_risk",
    "predict_survival",
    "predict_cumulative_hazard",
    "predict_survival_time",
    "predict_median_survival_time",
)


def _data(n_vars=2):
    x = [0.1, 0.4, 0.7, 0.2, 0.2, 0.9, 0.9, 0.3, 0.3, 0.6, 0.8, 0.1, 0.4, 0.8, 0.6, 0.5]
    time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    status = [1, 1, 0, 1, 0, 1, 1, 0]
    return x if n_vars else [], len(time), n_vars, time, status


def _config(name, **overrides):
    if name == "DeepSurv":
        options = {
            "hidden_layers": [2],
            "activation": core.Activation("tanh"),
            "dropout_rate": 0.0,
            "batch_size": 4,
            "n_epochs": 1,
            "l2_reg": 0.0,
            "seed": 1,
            "validation_fraction": 0.0,
        }
    elif name == "GradientBoostSurvival":
        options = {
            "n_estimators": 2,
            "max_depth": 2,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "seed": 1,
        }
    else:
        options = {
            "n_trees": 2,
            "max_depth": 2,
            "min_node_size": 1,
            "seed": 1,
            "oob_error": False,
        }
    options.update(overrides)
    return getattr(core, f"{name}Config")(**options)


def _fit(name, route, data, config=None):
    if route == "input":
        return core.SurvivalForestInput(*data)
    config = _config(name) if config is None else config
    if route == "typed":
        return core.SurvivalForest.fit_typed(core.SurvivalForestInput(*data), config)
    if route == "class":
        return getattr(core, name).fit(*data, config)
    function = {
        "DeepSurv": core.deep_surv,
        "GradientBoostSurvival": core.gradient_boost_survival,
        "SurvivalForest": core.survival_forest,
    }[name]
    return function(*data, config)


@pytest.mark.parametrize(("name", "route"), VALIDATION_ROUTES)
@pytest.mark.parametrize(("field", "index"), [("x", 5), ("time", 3)])
@pytest.mark.parametrize(
    "invalid", [np.nan, np.inf, -np.inf], ids=["nan", "infinity", "negative-infinity"]
)
def test_training_rejects_nonfinite_input_with_field_and_index(name, route, field, index, invalid):
    x, n_obs, n_vars, time, status = _data()
    values = x if field == "x" else time
    values[index] = invalid

    with pytest.raises(ValueError, match=rf"{field}.*index {index}\b") as error:
        _fit(name, route, (x, n_obs, n_vars, time, status))
    if name == "DeepSurv" and field == "x":
        assert "f32" in str(error.value)


@pytest.mark.parametrize(("name", "route"), VALIDATION_ROUTES)
@pytest.mark.parametrize(
    "invalid", [-1, 2, -(2**31), 2**31 - 1], ids=["negative", "two", "i32-min", "i32-max"]
)
def test_training_rejects_nonbinary_status_with_index(name, route, invalid):
    x, n_obs, n_vars, time, status = _data()
    status[4] = invalid

    with pytest.raises(ValueError, match=r"status.*index 4\b"):
        _fit(name, route, (x, n_obs, n_vars, time, status))


@pytest.fixture(scope="module", params=MODEL_NAMES)
def fitted_model(request):
    return _fit(request.param, "class", _data())


@pytest.mark.parametrize("method", PREDICTION_ROUTES)
@pytest.mark.parametrize(
    "invalid", [np.nan, np.inf, -np.inf], ids=["nan", "infinity", "negative-infinity"]
)
def test_prediction_rejects_nonfinite_features_with_index(fitted_model, method, invalid):
    x = [0.1, 0.2, 0.3, invalid]

    with pytest.raises(ValueError, match=r"x_new.*index 3\b"):
        getattr(fitted_model, method)(x, 2)


@pytest.mark.parametrize("method", PREDICTION_ROUTES)
def test_native_predictions_preserve_empty_input(fitted_model, method):
    assert getattr(fitted_model, method)([], 0) == []


@pytest.mark.parametrize("route", ["class", "function"])
@pytest.mark.parametrize("row", range(8))
@pytest.mark.parametrize("invalid", [1e100, -1e100], ids=["positive", "negative"])
def test_deep_training_rejects_f32_overflow_in_training_or_validation_rows(route, row, invalid):
    x, n_obs, n_vars, time, status = _data()
    index = row * n_vars + 1
    x[index] = invalid
    config = _config("DeepSurv", validation_fraction=0.5)

    with pytest.raises(ValueError, match=rf"x.*index {index}\b") as error:
        _fit("DeepSurv", route, (x, n_obs, n_vars, time, status), config)
    assert "f32" in str(error.value)


@pytest.mark.parametrize("route", ["class", "function"])
def test_deep_training_preserves_ordinary_f32_rounding_and_underflow(route):
    x, n_obs, n_vars, time, status = _data()
    x[0], x[1] = 1e-300, 0.1
    model = _fit("DeepSurv", route, (x, n_obs, n_vars, time, status))

    assert np.isfinite(model.predict_risk(x, n_obs)).all()


@pytest.mark.parametrize("route", ["class", "function"])
def test_deep_inference_preserves_finite_f64_features_outside_training_range(route):
    model = _fit("DeepSurv", route, _data())
    x = [1e100, 0.2, -1e100, 0.3]

    for method in PREDICTION_ROUTES:
        assert len(getattr(model, method)(x, 2)) == 2
    assert np.isfinite(model.predict_risk(x, 2)).all()
    assert np.isfinite(model.predict_survival(x, 2)).all()


@pytest.mark.parametrize(("name", "route"), FIT_ROUTES)
@pytest.mark.parametrize("policy", ["signed-times", "all-censored"])
def test_native_training_preserves_supported_time_and_censoring_values(name, route, policy):
    x, n_obs, n_vars, time, status = _data()
    if policy == "signed-times":
        time = [value - 4.0 for value in time]
    else:
        status = [0] * n_obs
    model = _fit(name, route, (x, n_obs, n_vars, time, status))

    assert model.unique_times == time
    assert np.isfinite(model.predict_risk(x, n_obs)).all()
    curves = model.predict_survival(x, n_obs)
    assert np.isfinite(curves).all()
    if policy == "all-censored":
        np.testing.assert_array_equal(curves, np.ones((n_obs, len(time))))


TREE_ROUTES = tuple((name, route) for name, route in FIT_ROUTES if name != "DeepSurv")


@pytest.mark.parametrize(("name", "route"), TREE_ROUTES)
def test_native_tree_models_preserve_zero_features(name, route):
    data = _data(n_vars=0)
    model = _fit(name, route, data)

    for method in PREDICTION_ROUTES:
        values = getattr(model, method)([], 3)
        assert len(values) == 3
        assert values == [values[0]] * 3
        assert getattr(model, method)([], 0) == []


@pytest.mark.parametrize(("name", "route"), TREE_ROUTES)
@pytest.mark.parametrize("large", [1e100, np.finfo(np.float64).max], ids=["beyond-f32", "f64-max"])
def test_native_trees_accept_large_finite_features(name, route, large):
    x, n_obs, n_vars, time, status = _data()
    x[0] = large
    # Constant trees exercise the input contract independently of split midpoint arithmetic.
    model = _fit(name, route, (x, n_obs, n_vars, time, status), _config(name, max_depth=0))
    prediction_x = [large, 0.2, -large, 0.3]

    assert np.isfinite(model.predict_risk(prediction_x, 2)).all()
    assert np.isfinite(model.predict_survival(prediction_x, 2)).all()
