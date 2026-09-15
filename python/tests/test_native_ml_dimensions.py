import struct

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
core = survival._survival
MODEL_NAMES = ("DeepSurv", "GradientBoostSurvival", "SurvivalForest")
HAS_ML_BINDINGS = all(
    hasattr(core, name)
    for name in (
        *MODEL_NAMES,
        "DeepSurvConfig",
        "GradientBoostSurvivalConfig",
        "SurvivalForestConfig",
    )
)
pytestmark = pytest.mark.skipif(
    not HAS_ML_BINDINGS,
    reason="native ML dimension tests require the Rust ml feature",
)

USIZE_MAX = (1 << (8 * struct.calcsize("P"))) - 1
OVERFLOW_HALF = USIZE_MAX // 2 + 1
FIT_ROUTES = tuple((name, route) for name in MODEL_NAMES for route in ("class", "function"))
if hasattr(core, "SurvivalForestInput"):
    FIT_ROUTES += (("SurvivalForest", "input"),)

PREDICTION_ROUTES = (
    "predict_risk",
    "predict_survival",
    "predict_cumulative_hazard",
    "predict_survival_time",
    "predict_median_survival_time",
)


def _data():
    x = [0.1, 0.4, 0.7, 0.2, 0.2, 0.9, 0.9, 0.3, 0.3, 0.6, 0.8, 0.1, 0.4, 0.8, 0.6, 0.5]
    time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    status = [1, 1, 0, 1, 0, 1, 1, 0]
    return x, len(time), 2, time, status


def _config(name):
    if name == "DeepSurv":
        return core.DeepSurvConfig(
            hidden_layers=[2],
            dropout_rate=0.0,
            batch_size=8,
            n_epochs=1,
            seed=1,
            validation_fraction=0.0,
        )
    if name == "GradientBoostSurvival":
        return core.GradientBoostSurvivalConfig(
            n_estimators=2,
            max_depth=2,
            min_samples_split=2,
            min_samples_leaf=1,
            seed=1,
        )
    return core.SurvivalForestConfig(n_trees=2, min_node_size=1, seed=1, oob_error=False)


def _fit(name, route, x, n_obs, n_vars, time, status):
    if route == "input":
        return core.SurvivalForestInput(x, n_obs, n_vars, time, status)
    if route == "class":
        return getattr(core, name).fit(x, n_obs, n_vars, time, status, _config(name))
    function = {
        "DeepSurv": core.deep_surv,
        "GradientBoostSurvival": core.gradient_boost_survival,
        "SurvivalForest": core.survival_forest,
    }[name]
    return function(x, n_obs, n_vars, time, status, _config(name))


@pytest.mark.parametrize(("name", "route"), FIT_ROUTES)
def test_training_rejects_zero_rows(name, route):
    with pytest.raises(ValueError, match="n_obs|observation|row|sample"):
        _fit(name, route, [], 0, 2, [], [])


@pytest.mark.parametrize(("name", "route"), FIT_ROUTES)
def test_training_rejects_row_dimension_overflow(name, route):
    # On the previous release this product wraps to zero, then the short time
    # vector fails safely. Requiring the overflow error proves shape validation
    # happens before any potentially enormous training allocation.
    with pytest.raises(ValueError, match="overflow|too large"):
        _fit(name, route, [], OVERFLOW_HALF, 2, [1.0, 2.0], [1, 0])


@pytest.mark.parametrize(("name", "route"), FIT_ROUTES)
def test_training_rejects_column_dimension_overflow(name, route):
    with pytest.raises(ValueError, match="overflow|too large"):
        _fit(name, route, [], 2, OVERFLOW_HALF, [1.0, 2.0], [1, 0])


@pytest.mark.parametrize("route", ["class", "function"])
def test_deep_training_rejects_zero_features(route):
    _, n_obs, _, time, status = _data()

    with pytest.raises(ValueError, match="n_vars|feature"):
        _fit("DeepSurv", route, [], n_obs, 0, time, status)


@pytest.mark.parametrize(("name", "route"), FIT_ROUTES)
@pytest.mark.parametrize(
    "field",
    ["short-matrix", "long-matrix", "short-time", "long-time", "short-status", "long-status"],
)
def test_training_rejects_inconsistent_lengths(name, route, field):
    x, n_obs, n_vars, time, status = _data()
    if field.endswith("matrix"):
        x = x[:-1] if field.startswith("short") else x + [0.0]
    elif field.endswith("time"):
        time = time[:-1] if field.startswith("short") else time + [9.0]
    else:
        status = status[:-1] if field.startswith("short") else status + [1]

    with pytest.raises(ValueError, match="length|dimension"):
        _fit(name, route, x, n_obs, n_vars, time, status)


@pytest.mark.parametrize(("name", "route"), FIT_ROUTES)
@pytest.mark.parametrize("field", ["n_obs", "n_vars"])
@pytest.mark.parametrize("invalid", [-1, USIZE_MAX + 1], ids=["negative", "beyond-usize"])
def test_training_rejects_dimensions_outside_python_binding_range(name, route, field, invalid):
    x, n_obs, n_vars, time, status = _data()
    if field == "n_obs":
        n_obs = invalid
    else:
        n_vars = invalid

    with pytest.raises(OverflowError, match="convert|large|range|negative"):
        _fit(name, route, x, n_obs, n_vars, time, status)


@pytest.fixture(scope="module", params=MODEL_NAMES)
def fitted_model(request):
    return _fit(request.param, "class", *_data())


@pytest.mark.parametrize("method", PREDICTION_ROUTES)
def test_prediction_rejects_dimension_product_overflow(fitted_model, method):
    with pytest.raises(ValueError, match="overflow|too large"):
        getattr(fitted_model, method)([], OVERFLOW_HALF)


@pytest.mark.parametrize("method", PREDICTION_ROUTES)
@pytest.mark.parametrize(
    ("x", "n_obs"),
    [([], 1), ([0.1], 1), ([0.1, 0.2, 0.3], 1), ([0.1], 0)],
    ids=["missing-row", "short-row", "long-row", "values-with-zero-rows"],
)
def test_prediction_rejects_inconsistent_matrix_lengths(fitted_model, method, x, n_obs):
    with pytest.raises(ValueError, match="length|dimension"):
        getattr(fitted_model, method)(x, n_obs)


@pytest.mark.parametrize("method", PREDICTION_ROUTES)
@pytest.mark.parametrize("invalid", [-1, USIZE_MAX + 1], ids=["negative", "beyond-usize"])
def test_prediction_rejects_dimensions_outside_python_binding_range(fitted_model, method, invalid):
    with pytest.raises(OverflowError, match="convert|large|range|negative"):
        getattr(fitted_model, method)([], invalid)


@pytest.mark.parametrize("method", PREDICTION_ROUTES)
def test_native_prediction_preserves_empty_batches(fitted_model, method):
    assert getattr(fitted_model, method)([], 0) == []


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_convenience_fit_preserves_valid_dimensions(name):
    x, n_obs, n_vars, time, status = _data()
    model = _fit(name, "function", x, n_obs, n_vars, time, status)

    for method in PREDICTION_ROUTES:
        assert len(getattr(model, method)(x, n_obs)) == n_obs
    assert np.isfinite(model.predict_risk(x, n_obs)).all()


TREE_FIT_ROUTES = (
    ("GradientBoostSurvival", "class"),
    ("GradientBoostSurvival", "function"),
    ("SurvivalForest", "class"),
    ("SurvivalForest", "function"),
)


@pytest.mark.parametrize(("name", "route"), TREE_FIT_ROUTES)
def test_tree_models_preserve_zero_feature_training_and_prediction(name, route):
    _, n_obs, _, time, status = _data()
    model = _fit(name, route, [], n_obs, 0, time, status)

    for method in PREDICTION_ROUTES:
        prediction = getattr(model, method)([], 3)
        assert len(prediction) == 3
        assert prediction == [prediction[0]] * 3
        assert getattr(model, method)([], 0) == []
        with pytest.raises(ValueError, match="too large|output"):
            getattr(model, method)([], USIZE_MAX)
    assert np.isfinite(model.predict_risk([], 3)).all()


@pytest.mark.parametrize("n_vars", [0, 2])
def test_forest_typed_fit_preserves_valid_dimensions(n_vars):
    if not hasattr(core, "SurvivalForestInput") or not hasattr(core.SurvivalForest, "fit_typed"):
        pytest.skip("typed forest input is not exposed by this build")
    x, n_obs, _, time, status = _data()
    x = x if n_vars else []
    typed_input = core.SurvivalForestInput(x, n_obs, n_vars, time, status)
    model = core.SurvivalForest.fit_typed(typed_input, _config("SurvivalForest"))

    assert typed_input.n_obs == n_obs
    assert typed_input.n_vars == n_vars
    for method in PREDICTION_ROUTES:
        assert len(getattr(model, method)(x, n_obs)) == n_obs
        assert getattr(model, method)([], 0) == []
    if hasattr(core, "survival_forest_typed"):
        convenience_model = core.survival_forest_typed(typed_input, _config("SurvivalForest"))
        np.testing.assert_allclose(
            convenience_model.predict_risk(x, n_obs), model.predict_risk(x, n_obs)
        )
