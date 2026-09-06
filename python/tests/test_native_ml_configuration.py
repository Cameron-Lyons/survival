import struct

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
core = survival._survival
MODEL_NAMES = ("DeepSurv", "GradientBoostSurvival", "SurvivalForest")
pytestmark = pytest.mark.skipif(
    not all(hasattr(core, name) and hasattr(core, f"{name}Config") for name in MODEL_NAMES),
    reason="native configuration tests require the Rust ml feature",
)
USIZE_MAX = (1 << (8 * struct.calcsize("P"))) - 1
F32_VEC_CAPACITY = (USIZE_MAX // 2) // 4
WIDE_HIDDEN_LAYER = 1 << (4 * struct.calcsize("P"))


def _cases(name, field, values):
    return [(name, field, value, f"{name}-{field}-{label}") for label, value in values]


INVALID_CONFIGURATIONS = (
    _cases("DeepSurv", "batch_size", [("zero", 0)])
    + _cases("DeepSurv", "n_epochs", [("zero", 0), ("impossible-capacity", USIZE_MAX)])
    + _cases(
        "DeepSurv",
        "hidden_layers",
        [
            ("zero-width", [0]),
            ("zero-last-width", [2, 0]),
            ("zero-first-width", [0, 2]),
            ("impossible-width", [USIZE_MAX]),
            ("adjacent-width-overflow", [WIDE_HIDDEN_LAYER, WIDE_HIDDEN_LAYER]),
        ],
    )
    + _cases(
        "DeepSurv",
        "learning_rate",
        [
            ("zero", 0.0),
            ("negative", -0.1),
            ("nan", np.nan),
            ("infinity", np.inf),
            ("negative-infinity", -np.inf),
            ("float32-overflow", 1e100),
            ("float32-underflow", 1e-300),
        ],
    )
    + _cases(
        "DeepSurv",
        "l2_reg",
        [
            ("negative", -0.1),
            ("nan", np.nan),
            ("infinity", np.inf),
            ("negative-infinity", -np.inf),
            ("float32-overflow", 1e100),
        ],
    )
    + [
        item
        for field in ("dropout_rate", "validation_fraction")
        for item in _cases(
            "DeepSurv",
            field,
            [
                ("negative", -0.1),
                ("one", 1.0),
                ("nan", np.nan),
                ("infinity", np.inf),
                ("negative-infinity", -np.inf),
            ],
        )
    ]
    + _cases(
        "GradientBoostSurvival", "n_estimators", [("zero", 0), ("impossible-capacity", USIZE_MAX)]
    )
    + _cases("GradientBoostSurvival", "min_samples_leaf", [("zero", 0)])
    + [
        item
        for field in ("learning_rate", "subsample")
        for item in _cases(
            "GradientBoostSurvival",
            field,
            [
                ("zero", 0.0),
                ("negative", -0.1),
                ("above-one", 1.1),
                ("nan", np.nan),
                ("infinity", np.inf),
                ("negative-infinity", -np.inf),
            ],
        )
    ]
    + _cases("SurvivalForest", "n_trees", [("zero", 0), ("impossible-capacity", USIZE_MAX)])
    + _cases("SurvivalForest", "n_random_splits", [("zero", 0)])
    + _cases(
        "SurvivalForest",
        "sample_fraction",
        [
            ("zero", 0.0),
            ("negative", -0.1),
            ("above-one", 1.1),
            ("nan", np.nan),
            ("infinity", np.inf),
            ("negative-infinity", -np.inf),
        ],
    )
)


def _config(name, **overrides):
    if name == "DeepSurv":
        options = {
            "hidden_layers": [2],
            "dropout_rate": 0.0,
            "learning_rate": 0.001,
            "batch_size": 8,
            "n_epochs": 1,
            "l2_reg": 0.0,
            "validation_fraction": 0.0,
            "seed": 1,
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
        options = {"n_trees": 2, "min_node_size": 1, "seed": 1, "oob_error": False}
    options.update(overrides)
    return getattr(core, f"{name}Config")(**options)


def _data(n_vars=2):
    x = [0.1, 0.4, 0.7, 0.2, 0.2, 0.9, 0.9, 0.3, 0.3, 0.6, 0.8, 0.1, 0.4, 0.8, 0.6, 0.5]
    time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    status = [1, 1, 0, 1, 0, 1, 1, 0]
    if n_vars == 3:
        x = [value for i in range(len(time)) for value in (x[2 * i], x[2 * i + 1], 0.5)]
    return x if n_vars else [], len(time), n_vars, time, status


def _routes(name):
    if name == "SurvivalForest" and hasattr(getattr(core, "SurvivalForest", None), "fit_typed"):
        return ("class", "function", "typed")
    return ("class", "function")


def _fit(name, route, config, n_vars=2):
    arguments = _data(n_vars)
    if route == "typed":
        return core.SurvivalForest.fit_typed(core.SurvivalForestInput(*arguments), config)
    if route == "class":
        return getattr(core, name).fit(*arguments, config)
    function = {
        "DeepSurv": core.deep_surv,
        "GradientBoostSurvival": core.gradient_boost_survival,
        "SurvivalForest": core.survival_forest,
    }[name]
    return function(*arguments, config)


@pytest.mark.parametrize(
    ("name", "field", "invalid"),
    [
        pytest.param(name, field, value, id=label)
        for name, field, value, label in INVALID_CONFIGURATIONS
    ],
)
def test_constructor_rejects_invalid_configuration(name, field, invalid):
    with pytest.raises(ValueError, match=field):
        _config(name, **{field: invalid})


@pytest.mark.parametrize(
    ("name", "route", "field", "invalid"),
    [
        pytest.param(name, route, field, value, id=f"{label}-{route}")
        for name, field, value, label in INVALID_CONFIGURATIONS
        for route in _routes(name)
    ],
)
def test_fit_rechecks_mutated_configuration_before_training(name, route, field, invalid):
    config = _config(name)
    setattr(config, field, invalid)

    with pytest.raises(ValueError, match=field):
        _fit(name, route, config)


@pytest.mark.parametrize("sigma", [np.nan, np.inf], ids=["nan", "infinity"])
def test_deephit_constructor_rejects_nonfinite_sigma(sigma):
    if not hasattr(core, "DeepHitConfig"):
        pytest.skip("DeepHit is not exposed by this build")

    with pytest.raises(ValueError, match="sigma"):
        core.DeepHitConfig(sigma=sigma)


@pytest.mark.parametrize(
    ("name", "route"), [(name, route) for name in MODEL_NAMES for route in _routes(name)]
)
def test_valid_mutated_configuration_remains_usable(name, route):
    config = _config(name)
    if name == "DeepSurv":
        changes = {
            "hidden_layers": [],
            "learning_rate": 0.01,
            "dropout_rate": 0.0,
            "batch_size": 4,
            "n_epochs": 2,
            "l2_reg": 0.001,
            "validation_fraction": 0.0,
        }
    elif name == "GradientBoostSurvival":
        changes = {"n_estimators": 3, "learning_rate": 0.2, "subsample": 0.75, "max_depth": 0}
    else:
        changes = {"n_trees": 3, "sample_fraction": 1.0, "n_random_splits": 1, "max_depth": 0}
    for field, value in changes.items():
        setattr(config, field, value)

    model = _fit(name, route, config)
    x, n_obs, _, _, _ = _data()

    assert np.isfinite(model.predict_risk(x, n_obs)).all()
    assert np.isfinite(model.predict_survival(x, n_obs)).all()
    if name == "DeepSurv":
        assert model.hidden_layers == []
        assert len(model.train_loss) == 2
    elif name == "GradientBoostSurvival":
        assert model.n_estimators == 3
    else:
        assert model.n_trees == 3
    for field, value in changes.items():
        assert getattr(config, field) == value


@pytest.mark.parametrize("route", ["class", "function"])
def test_deep_empty_hidden_layers_remain_supported(route):
    model = _fit("DeepSurv", route, _config("DeepSurv", hidden_layers=[]))
    x, n_obs, _, _, _ = _data()

    assert model.hidden_layers == []
    assert np.isfinite(model.predict_risk(x, n_obs)).all()


TREE_ROUTES = [
    (name, route) for name in ("GradientBoostSurvival", "SurvivalForest") for route in _routes(name)
]


@pytest.mark.parametrize(("name", "route"), TREE_ROUTES)
@pytest.mark.parametrize("n_vars", [0, 2])
def test_tree_depth_zero_and_zero_feature_models_remain_supported(name, route, n_vars):
    model = _fit(name, route, _config(name, max_depth=0), n_vars=n_vars)
    x, n_obs, _, _, _ = _data(n_vars)
    risk = model.predict_risk(x, n_obs)

    assert np.isfinite(risk).all()
    assert risk == [risk[0]] * n_obs
    assert np.isfinite(model.predict_survival(x, n_obs)).all()


@pytest.mark.parametrize(("name", "route"), TREE_ROUTES)
@pytest.mark.parametrize(
    "minimum", [USIZE_MAX // 2 + 1, USIZE_MAX], ids=["half-usize", "max-usize"]
)
def test_large_positive_tree_leaf_thresholds_produce_stumps(name, route, minimum):
    field = "min_samples_leaf" if name == "GradientBoostSurvival" else "min_node_size"
    config = _config(name, **{field: minimum})
    model = _fit(name, route, config)
    x, n_obs, _, _, _ = _data()
    risk = model.predict_risk(x, n_obs)

    assert np.isfinite(risk).all()
    assert risk == [risk[0]] * n_obs
    assert getattr(config, field) == minimum


@pytest.mark.parametrize("route", ["class", "function"])
@pytest.mark.parametrize(
    ("width", "n_vars", "batch_size", "validation_fraction"),
    [
        pytest.param(F32_VEC_CAPACITY // 3 + 1, 3, 1, 0.0, id="first-layer-weights"),
        pytest.param(F32_VEC_CAPACITY // 4 + 1, 2, 8, 0.0, id="training-activations"),
        pytest.param(F32_VEC_CAPACITY // 3 + 1, 2, 1, 0.5, id="validation-activations"),
    ],
)
def test_deep_fit_rejects_impossible_tensor_capacity(
    route, width, n_vars, batch_size, validation_fraction
):
    config = _config("DeepSurv", batch_size=batch_size, validation_fraction=validation_fraction)
    config.hidden_layers = [width]

    with pytest.raises(ValueError, match="hidden_layers"):
        _fit("DeepSurv", route, config, n_vars=n_vars)


@pytest.mark.parametrize("route", ["class", "function"])
@pytest.mark.parametrize("mutated", [False, True], ids=["constructor", "mutated"])
def test_small_nonnegative_deep_regularization_remains_supported(route, mutated):
    config = _config("DeepSurv", l2_reg=0.0 if mutated else 1e-300)
    if mutated:
        config.l2_reg = 1e-300
    model = _fit("DeepSurv", route, config)
    x, n_obs, _, _, _ = _data()

    assert np.isfinite(model.predict_risk(x, n_obs)).all()
    assert config.l2_reg == 1e-300


@pytest.mark.parametrize("route", ["class", "function"])
@pytest.mark.parametrize("minimum", [0, 1])
def test_small_boost_split_thresholds_remain_supported(route, minimum):
    config = _config("GradientBoostSurvival", min_samples_split=minimum)
    model = _fit("GradientBoostSurvival", route, config)
    x, n_obs, _, _, _ = _data()

    assert np.isfinite(model.predict_risk(x, n_obs)).all()


@pytest.mark.parametrize("route", _routes("SurvivalForest"))
def test_forest_zero_node_threshold_remains_supported(route):
    config = _config("SurvivalForest", min_node_size=0)
    model = _fit("SurvivalForest", route, config)
    x, n_obs, _, _, _ = _data()

    assert np.isfinite(model.predict_risk(x, n_obs)).all()
