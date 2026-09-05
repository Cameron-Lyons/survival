import importlib
import warnings
from decimal import Decimal
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
sklearn_compat = importlib.import_module("survival.sklearn_compat")

BASE_ESTIMATORS = (
    "CoxPHEstimator",
    "AFTEstimator",
    "DeepSurvEstimator",
    "GradientBoostSurvivalEstimator",
    "SurvivalForestEstimator",
)
STREAMING_ESTIMATORS = tuple(f"Streaming{name}" for name in BASE_ESTIMATORS)
ALL_ESTIMATORS = BASE_ESTIMATORS + STREAMING_ESTIMATORS
CLASSICAL_ESTIMATORS = (
    "CoxPHEstimator",
    "AFTEstimator",
    "StreamingCoxPHEstimator",
    "StreamingAFTEstimator",
)
INVALID_FEATURES = ("nan", "inf", "negative-inf", "complex", "masked")
FEATURE_ERROR = r"(?i)feature|input|finite|nan|infin|complex|mask|float|sample|row"


def _data():
    x = np.array(
        [
            [-2, 0],
            [0, -1],
            [-1, 2],
            [1, 0],
            [0, 1],
            [2, -2],
            [-2, 1],
            [1, -1],
            [2, 2],
            [-1, 0],
            [0, -2],
            [2, 1],
        ],
        dtype=np.float64,
    )
    y = np.column_stack(
        ([4, 8, 1, 11, 6, 3, 10, 2, 12, 5, 9, 7], [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1])
    )
    return x, y


def _invalid_features(kind):
    x, _ = _data()
    if kind == "complex":
        return x.astype(np.complex128) + 2j
    if kind == "masked":
        x = np.ma.array(x, mask=False)
        x.mask[0, 0] = True
        return x
    x[0, 0] = {"nan": np.nan, "inf": np.inf, "negative-inf": -np.inf}[kind]
    return x


def _unexpected_work(*args, **kwargs):
    pytest.fail("Invalid features reached model training, prediction, or concordance")


def _invoke(estimator, method, x, y=None, **kwargs):
    result = (
        getattr(estimator, method)(x, y, **kwargs)
        if method in {"fit", "score"}
        else getattr(estimator, method)(x, **kwargs)
    )
    return list(result) if method.endswith("batched") else result


def _boundary_estimator(name):
    estimator = getattr(sklearn_compat, name)()
    estimator.is_fitted_ = True
    estimator.n_features_in_ = 2
    estimator.coef_ = np.array([0.2, -0.1])
    estimator.intercept_ = 0.1
    estimator.scale_ = 1.0
    estimator.event_times_ = np.array([1.0, 2.0])
    estimator._baseline_times_ = np.array([1.0, 2.0])
    estimator._baseline_hazard_ = np.array([0.1, 0.2])
    estimator._center_ = 0.0
    estimator.model_ = SimpleNamespace(
        predict=_unexpected_work,
        predict_risk=_unexpected_work,
        predict_survival=_unexpected_work,
        predict_median_survival_time=_unexpected_work,
        unique_times=[1.0, 2.0],
    )
    return estimator


@pytest.fixture
def reject_backend_work(monkeypatch):
    for symbol in (
        "coxph_fit",
        "survreg",
        "concordance_index",
        "Activation",
        "GradientBoostSurvivalConfig",
        "SurvivalForestConfig",
    ):
        monkeypatch.setattr(survival._survival, symbol, _unexpected_work, raising=False)


@pytest.mark.parametrize("name", ALL_ESTIMATORS)
@pytest.mark.parametrize("method", ["fit", "score", "predict"])
@pytest.mark.parametrize("invalid", INVALID_FEATURES)
def test_estimators_reject_invalid_features_before_work(reject_backend_work, name, method, invalid):
    estimator = _boundary_estimator(name)
    x = _invalid_features(invalid)
    _, y = _data()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=FEATURE_ERROR):
            _invoke(estimator, method, x, y)


PREDICTION_ROUTES = tuple(
    (name, method)
    for name in ALL_ESTIMATORS
    for method in (
        ("predict_median", "predict_quantile")
        if "AFT" in name
        else ("predict_survival_function", "predict_median_survival_time")
    )
)


@pytest.mark.parametrize(("name", "method"), PREDICTION_ROUTES)
@pytest.mark.parametrize("invalid", INVALID_FEATURES)
def test_prediction_routes_reject_invalid_features_before_work(name, method, invalid):
    estimator = _boundary_estimator(name)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=FEATURE_ERROR):
            getattr(estimator, method)(_invalid_features(invalid))


STREAMING_ROUTES = tuple(
    (name, method)
    for name in STREAMING_ESTIMATORS
    for method in (
        ("predict_batched", "predict_to_array")
        if "AFT" in name
        else ("predict_batched", "predict_survival_batched", "predict_to_array")
    )
)


@pytest.mark.parametrize(("name", "method"), STREAMING_ROUTES)
@pytest.mark.parametrize("invalid", INVALID_FEATURES)
def test_streaming_routes_reject_invalid_features_before_work(name, method, invalid):
    estimator = _boundary_estimator(name)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=FEATURE_ERROR):
            _invoke(estimator, method, _invalid_features(invalid), batch_size=3)


@pytest.mark.parametrize(("name", "method"), STREAMING_ROUTES)
def test_empty_streaming_predictions_remain_empty(name, method):
    estimator = _boundary_estimator(name)
    result = getattr(estimator, method)(np.empty((0, 2)), batch_size=3)

    if method.endswith("batched"):
        assert list(result) == []
    else:
        assert result.shape == (0,)


def _classical_estimator(name):
    kwargs = {"n_iters": 20} if "CoxPH" in name else {"max_iter": 100}
    return getattr(sklearn_compat, name)(**kwargs)


@pytest.mark.parametrize("name", CLASSICAL_ESTIMATORS)
@pytest.mark.parametrize(
    "storage", ["bool", "int", "float", "numeric-string", "unmasked", "readonly", "noncontiguous"]
)
def test_valid_feature_storage_preserves_predictions_and_inputs(name, storage):
    x, y = _data()
    if storage in {"bool", "int", "float", "numeric-string"}:
        x = x.astype(
            {"bool": bool, "int": np.int64, "float": np.float32, "numeric-string": str}[storage]
        )
    elif storage == "unmasked":
        x = np.ma.array(x, mask=False)
    elif storage == "readonly":
        x.setflags(write=False)
    else:
        x, y = x[::-1], y[::-1]
        assert not x.flags.c_contiguous
    original = x.copy()
    expected_x = np.asarray(x, dtype=np.float64)
    expected = _classical_estimator(name).fit(expected_x, y)
    actual = _classical_estimator(name).fit(x, y)

    np.testing.assert_allclose(
        actual.predict(x), expected.predict(expected_x), rtol=1e-12, atol=1e-12
    )
    assert actual.score(x, y) == pytest.approx(expected.score(expected_x, y), abs=1e-15)
    np.testing.assert_array_equal(x, original)
    if np.ma.isMaskedArray(x):
        np.testing.assert_array_equal(x.mask, original.mask)


@pytest.mark.parametrize("name", CLASSICAL_ESTIMATORS)
def test_classical_models_support_zero_features(name):
    _, y = _data()
    x = np.empty((len(y), 0))
    estimator = _classical_estimator(name).fit(x, y)

    assert estimator.coef_.shape == (0,)
    prediction = estimator.predict(x)
    assert prediction.shape == (len(y),)
    assert np.isfinite(prediction).all()
    np.testing.assert_array_equal(prediction, np.repeat(prediction[0], len(y)))
    assert 0.0 <= estimator.score(x, y) <= 1.0


class _LazyMaskedRows:
    def __init__(self, data):
        self.data = data
        self.full_materializations = 0
        self.slices = []

    @property
    def shape(self):
        return self.data.shape

    def __array__(self, dtype=None, copy=None):
        self.full_materializations += 1
        return np.asarray(self.data, dtype=dtype)

    def __getitem__(self, key):
        self.slices.append((key.start, key.stop))
        return self.data[key]


@pytest.mark.parametrize(
    "route",
    [
        "predict_batched",
        "predict_survival_batched",
        "predict_to_array",
        "predict_large_dataset",
        "survival_curves_to_disk",
    ],
)
def test_streaming_rejects_masked_lazy_slice_without_materializing_all_rows(tmp_path, route):
    x, _ = _data()
    masked = np.ma.array(x, mask=False)
    masked.mask[3, 0] = True
    rows = _LazyMaskedRows(masked)
    estimator = _boundary_estimator("StreamingCoxPHEstimator")
    predicted_rows = []

    def predict_valid_rows(covariates):
        predicted_rows.append(len(covariates))
        return [0.0] * len(covariates)

    estimator.model_.predict = predict_valid_rows

    def run_prediction():
        if route == "predict_large_dataset":
            sklearn_compat.predict_large_dataset(estimator, rows, batch_size=3)
        elif route == "survival_curves_to_disk":
            sklearn_compat.survival_curves_to_disk(
                estimator, rows, str(tmp_path / "survival.dat"), batch_size=3
            )
        else:
            _invoke(estimator, route, rows, batch_size=3)

    with pytest.raises(ValueError, match=FEATURE_ERROR):
        run_prediction()

    assert rows.full_materializations == 0
    assert rows.slices == [(0, 3), (3, 6)]
    assert predicted_rows == [3]
    assert masked.mask[3, 0]


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        (
            "GradientBoostSurvivalEstimator",
            {"n_estimators": 2, "max_depth": 2, "min_samples_leaf": 1, "seed": 1},
        ),
        (
            "SurvivalForestEstimator",
            {"n_trees": 2, "min_node_size": 1, "seed": 1, "oob_error": False},
        ),
    ],
)
def test_fitted_trees_reject_nonfinite_features_instead_of_returning_plausible_outputs(
    name, kwargs
):
    if not hasattr(survival._survival, name.removesuffix("Estimator")):
        pytest.skip("tree survival estimators require the Rust ml feature")
    x, y = _data()
    estimator = getattr(sklearn_compat, name)(**kwargs).fit(x, y)

    for invalid in ("nan", "inf", "negative-inf"):
        for method in ("predict", "predict_survival_function", "score"):
            with pytest.raises(ValueError, match=FEATURE_ERROR):
                _invoke(estimator, method, _invalid_features(invalid), y)


@pytest.mark.parametrize("name", ALL_ESTIMATORS)
@pytest.mark.parametrize("method", ["fit", "score", "predict"])
def test_direct_estimator_methods_reject_zero_rows_before_work(reject_backend_work, name, method):
    estimator = _boundary_estimator(name)

    with pytest.raises(ValueError, match=FEATURE_ERROR):
        _invoke(estimator, method, np.empty((0, 2)), np.empty((0, 2)))


@pytest.mark.parametrize("name", ["DeepSurvEstimator", "StreamingDeepSurvEstimator"])
def test_deep_models_reject_zero_features_before_training(reject_backend_work, name):
    _, y = _data()
    estimator = getattr(sklearn_compat, name)()

    with pytest.raises(ValueError, match="feature"):
        estimator.fit(np.empty((len(y), 0)), y)


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        (
            "GradientBoostSurvivalEstimator",
            {"n_estimators": 2, "max_depth": 2, "min_samples_leaf": 1, "seed": 1},
        ),
        (
            "SurvivalForestEstimator",
            {"n_trees": 2, "min_node_size": 1, "seed": 1, "oob_error": False},
        ),
    ],
)
def test_tree_models_support_zero_features(name, kwargs):
    if not hasattr(survival._survival, name.removesuffix("Estimator")):
        pytest.skip("tree survival estimators require the Rust ml feature")
    _, y = _data()
    x = np.empty((len(y), 0))
    estimator = getattr(sklearn_compat, name)(**kwargs).fit(x, y)

    prediction = estimator.predict(x)
    assert prediction.shape == (len(y),)
    assert np.isfinite(prediction).all()
    np.testing.assert_array_equal(prediction, np.repeat(prediction[0], len(y)))
    assert np.isfinite(estimator.predict_survival_function(x)[1]).all()
    assert 0.0 <= estimator.score(x, y) <= 1.0


@pytest.mark.parametrize(
    "case",
    [
        "numpy-complex-object",
        "huge-integer",
        "wide-float-overflow",
        "datetime-nat",
        "timedelta-nat",
        "datetime-nat-object",
        "timedelta-nat-object",
        "masked-object",
        "none-object",
        "nan-string",
        "infinity-string",
    ],
)
def test_feature_conversion_rejects_invalid_scalar_values_without_warnings(case):
    common = importlib.import_module("survival._sklearn_common")
    if case == "wide-float-overflow":
        with np.errstate(over="ignore"):
            x = np.array([[np.longdouble(np.finfo(np.float64).max) * 2]], dtype=np.longdouble)
    elif case in {"datetime-nat", "timedelta-nat"}:
        dtype = "datetime64[D]" if case == "datetime-nat" else "timedelta64[D]"
        x = np.array([["NaT"]], dtype=dtype)
    else:
        scalar = {
            "numpy-complex-object": np.complex128(1 + 2j),
            "huge-integer": 2**10000,
            "datetime-nat-object": np.datetime64("NaT", "D"),
            "timedelta-nat-object": np.timedelta64("NaT", "D"),
            "masked-object": np.ma.masked,
            "none-object": None,
            "nan-string": "nan",
            "infinity-string": "inf",
        }[case]
        x = np.array([[scalar]], dtype=object)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=FEATURE_ERROR):
            common.check_array(x, dtype=np.float64, ensure_2d=True)


@pytest.mark.parametrize("unmasked", [False, True])
def test_real_numeric_objects_and_numeric_strings_remain_supported(unmasked):
    common = importlib.import_module("survival._sklearn_common")
    x = np.array(
        [[Decimal("1.5"), Fraction(1, 2)], [True, np.int64(2)], ["3.25", -4]], dtype=object
    )
    if unmasked:
        x = np.ma.array(x, mask=False)
    original = x.copy()

    checked = common.check_array(x, dtype=np.float64, ensure_2d=True)

    np.testing.assert_array_equal(checked, [[1.5, 0.5], [1.0, 2.0], [3.25, -4.0]])
    assert checked.dtype == np.float64
    np.testing.assert_array_equal(x, original)
    if unmasked:
        np.testing.assert_array_equal(x.mask, original.mask)


@pytest.mark.parametrize("x", [np.array(1), np.ones(2), np.ones((2, 1, 1))])
def test_feature_validation_rejects_malformed_dimensions(x):
    common = importlib.import_module("survival._sklearn_common")

    with pytest.raises(ValueError, match="2D"):
        common.check_array(x, dtype=np.float64, ensure_2d=True)
