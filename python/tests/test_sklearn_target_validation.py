import importlib
import warnings
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
sklearn_compat = importlib.import_module("survival.sklearn_compat")

ESTIMATORS = (
    "CoxPHEstimator",
    "AFTEstimator",
    "StreamingCoxPHEstimator",
    "StreamingAFTEstimator",
)
ML_ESTIMATORS = (
    "DeepSurvEstimator",
    "GradientBoostSurvivalEstimator",
    "SurvivalForestEstimator",
    "StreamingDeepSurvEstimator",
    "StreamingGradientBoostSurvivalEstimator",
    "StreamingSurvivalForestEstimator",
)


def _data(dtype=np.float64):
    x = np.array([[0.1], [0.7], [0.2], [0.9], [0.3], [0.8], [0.4], [0.6]])
    y = np.array([[1, 1], [2, 1], [3, 0], [4, 1], [5, 0], [6, 1], [7, 1], [8, 0]], dtype=dtype)
    return x, y


def _estimator(name, **kwargs):
    options = {"n_iters": 10} if "CoxPH" in name else {"max_iter": 50}
    options.update(kwargs)
    return getattr(sklearn_compat, name)(**options)


def _unexpected_work(*args, **kwargs):
    pytest.fail("Invalid survival targets reached model training or prediction")


@pytest.fixture(scope="module", params=ESTIMATORS)
def fitted_estimator(request):
    x, y = _data()
    return _estimator(request.param).fit(x, y)


INVALID_STATUSES = (
    pytest.param(0.9, np.float64, id="fractional-censor"),
    pytest.param(1.9, np.float64, id="fractional-event"),
    pytest.param(-0.1, np.float64, id="negative-fraction"),
    pytest.param(-1, np.int64, id="negative-integer"),
    pytest.param(2, np.int64, id="left-censoring-is-not-binary"),
    pytest.param(np.nan, np.float64, id="nan"),
    pytest.param(np.inf, np.float64, id="infinity"),
    pytest.param(-np.inf, np.float64, id="negative-infinity"),
    pytest.param(2**32, np.int64, id="int32-overflow-censor"),
    pytest.param(2**32 + 1, np.int64, id="int32-overflow-event"),
    pytest.param(2**63 + 1, np.uint64, id="uint64-overflow-event"),
    pytest.param(2**1000 + 1, object, id="large-python-integer"),
    pytest.param(Decimal("1.00000000000000000001"), object, id="decimal-above-one"),
    pytest.param(Fraction(2**60 + 1, 2**60), object, id="fraction-above-one"),
    pytest.param(np.finfo(np.float64).max, np.float64, id="float-overflow"),
    pytest.param(np.nextafter(1.0, 2.0), np.float64, id="next-float-above-one"),
    pytest.param(
        np.nextafter(np.longdouble(1), np.longdouble(2)),
        np.longdouble,
        id="next-longdouble-above-one",
    ),
    pytest.param(np.nextafter(0.0, 1.0), np.float64, id="smallest-positive-float"),
    pytest.param(1 + 2j, np.complex128, id="complex-array"),
    pytest.param(1 + 2j, object, id="complex-object"),
)


@pytest.mark.parametrize("name", ESTIMATORS)
@pytest.mark.parametrize(("invalid_status", "dtype"), INVALID_STATUSES)
def test_fit_rejects_invalid_status_before_training(monkeypatch, name, invalid_status, dtype):
    x, y = _data(dtype)
    y[0, 1] = invalid_status
    monkeypatch.setattr(survival._survival, "coxph_fit", _unexpected_work)
    monkeypatch.setattr(survival._survival, "survreg", _unexpected_work)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="status|real"):
            _estimator(name).fit(x, y)


@pytest.mark.parametrize(("invalid_status", "dtype"), INVALID_STATUSES)
def test_score_rejects_invalid_status_before_prediction(
    monkeypatch, fitted_estimator, invalid_status, dtype
):
    x, y = _data(dtype)
    y[0, 1] = invalid_status
    monkeypatch.setattr(fitted_estimator, "predict", _unexpected_work)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="status|real"):
            fitted_estimator.score(x, y)


MALFORMED_TARGETS = (
    pytest.param(np.array(1), "2D", id="scalar"),
    pytest.param(np.ones(8), "2D", id="one-dimensional"),
    pytest.param(np.ones((8, 2, 1)), "2D", id="three-dimensional"),
    pytest.param(np.ones((8, 1)), "2 columns", id="one-column"),
    pytest.param(np.ones((8, 3)), "2 columns", id="three-columns"),
    pytest.param(np.ones((7, 2)), "samples", id="row-mismatch"),
)


@pytest.mark.parametrize("name", ESTIMATORS)
@pytest.mark.parametrize(("y", "message"), MALFORMED_TARGETS)
def test_fit_rejects_malformed_target_before_training(monkeypatch, name, y, message):
    x, _ = _data()
    monkeypatch.setattr(survival._survival, "coxph_fit", _unexpected_work)
    monkeypatch.setattr(survival._survival, "survreg", _unexpected_work)

    with pytest.raises(ValueError, match=message):
        _estimator(name).fit(x, y)


@pytest.mark.parametrize(("y", "message"), MALFORMED_TARGETS)
def test_score_rejects_malformed_target_before_prediction(
    monkeypatch, fitted_estimator, y, message
):
    x, _ = _data()
    monkeypatch.setattr(fitted_estimator, "predict", _unexpected_work)

    with pytest.raises(ValueError, match=message):
        fitted_estimator.score(x, y)


@pytest.mark.parametrize("name", ESTIMATORS)
@pytest.mark.parametrize("invalid_time", [np.nan, np.inf, -np.inf, 1 + 2j])
def test_fit_rejects_invalid_time_before_training(monkeypatch, name, invalid_time):
    x, y = _data(object if isinstance(invalid_time, complex) else np.float64)
    y[0, 0] = invalid_time
    monkeypatch.setattr(survival._survival, "coxph_fit", _unexpected_work)
    monkeypatch.setattr(survival._survival, "survreg", _unexpected_work)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="time|real"):
            _estimator(name).fit(x, y)


@pytest.mark.parametrize("invalid_time", [np.nan, np.inf, -np.inf, 1 + 2j])
def test_score_rejects_invalid_time_before_prediction(monkeypatch, fitted_estimator, invalid_time):
    x, y = _data(object if isinstance(invalid_time, complex) else np.float64)
    y[0, 0] = invalid_time
    monkeypatch.setattr(fitted_estimator, "predict", _unexpected_work)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="time|real"):
            fitted_estimator.score(x, y)


@pytest.mark.parametrize("name", ML_ESTIMATORS)
@pytest.mark.parametrize("invalid_status", [0.9, 2, np.nan])
def test_ml_fit_rejects_invalid_status_before_model_setup(monkeypatch, name, invalid_status):
    x, y = _data()
    y[0, 1] = invalid_status
    for symbol in ("Activation", "GradientBoostSurvivalConfig", "SurvivalForestConfig"):
        monkeypatch.setattr(survival._survival, symbol, _unexpected_work, raising=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="status"):
            getattr(sklearn_compat, name)().fit(x, y)


@pytest.mark.parametrize("name", ESTIMATORS)
@pytest.mark.parametrize(
    "status_type", [bool, np.int32, np.int64, np.float32, np.float64, Decimal, Fraction]
)
def test_binary_status_types_preserve_fitted_results(name, status_type):
    x, y = _data()
    # Object storage preserves the status scalar types alongside floating-point times.
    typed_y = y.astype(object)
    typed_y[:, 1] = [status_type(value) for value in y[:, 1]]
    expected = _estimator(name).fit(x, y)
    actual = _estimator(name).fit(x, typed_y)

    np.testing.assert_allclose(actual.predict(x), expected.predict(x), rtol=1e-12, atol=1e-12)
    assert actual.score(x, typed_y) == pytest.approx(expected.score(x, y), abs=1e-15)


@pytest.mark.parametrize("name", ["AFTEstimator", "StreamingAFTEstimator"])
@pytest.mark.parametrize("distribution", ["gaussian", "logistic"])
def test_identity_aft_families_pass_negative_and_zero_times_to_model(
    monkeypatch, name, distribution
):
    x, y = _data()
    y[:, 0] -= 4.0

    class ReachedModelError(Exception):
        pass

    def check_signed_times(**kwargs):
        assert kwargs["distribution"] == distribution
        np.testing.assert_array_equal(kwargs["time"], y[:, 0])
        np.testing.assert_array_equal(kwargs["status"], y[:, 1])
        raise ReachedModelError

    monkeypatch.setattr(survival._survival, "survreg", check_signed_times)
    # Time support belongs to the chosen distribution, beyond the shared target guard.
    with pytest.raises(ReachedModelError):
        _estimator(name, distribution=distribution).fit(x, y)


@pytest.mark.parametrize("dtype", [str, bytes, "datetime64[D]", "timedelta64[D]"])
@pytest.mark.parametrize("object_storage", [False, True], ids=["native-dtype", "object-dtype"])
def test_target_validation_rejects_nonnumeric_storage(dtype, object_storage):
    common = importlib.import_module("survival._sklearn_common")
    x, y = _data(dtype)
    if object_storage:
        y = y.astype(object)
    original = y.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="real numeric"):
            common._validate_survival_data(x, y)

    np.testing.assert_array_equal(y, original)


@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.uint64, object])
def test_target_validation_preserves_readonly_noncontiguous_inputs(dtype):
    common = importlib.import_module("survival._sklearn_common")
    x, y = _data(dtype)
    x, y = x[::-1], y[::-1]
    x.flags.writeable = False
    y.flags.writeable = False
    original_x, original_y = x.copy(), y.copy()

    checked_x, time, status = common._validate_survival_data(x, y)

    assert checked_x.dtype == np.float64
    assert time.dtype == np.float64
    assert status.dtype == np.int32
    np.testing.assert_array_equal(checked_x, original_x)
    np.testing.assert_array_equal(time, original_y[:, 0])
    np.testing.assert_array_equal(status, original_y[:, 1])
    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(y, original_y)


@pytest.mark.parametrize("name", ESTIMATORS)
@pytest.mark.parametrize("masked_column", [0, 1], ids=["time", "status"])
def test_fit_rejects_masked_target_before_training(monkeypatch, name, masked_column):
    x, y = _data()
    y = np.ma.array(y, mask=False)
    y.mask[0, masked_column] = True
    original = y.copy()
    monkeypatch.setattr(survival._survival, "coxph_fit", _unexpected_work)
    monkeypatch.setattr(survival._survival, "survreg", _unexpected_work)

    with pytest.raises(ValueError, match="mask"):
        _estimator(name).fit(x, y)

    np.testing.assert_array_equal(y.data, original.data)
    np.testing.assert_array_equal(y.mask, original.mask)


@pytest.mark.parametrize("masked_column", [0, 1], ids=["time", "status"])
def test_score_rejects_masked_target_before_prediction(
    monkeypatch, fitted_estimator, masked_column
):
    x, y = _data()
    y = np.ma.array(y, mask=False)
    y.mask[0, masked_column] = True
    original = y.copy()
    monkeypatch.setattr(fitted_estimator, "predict", _unexpected_work)

    with pytest.raises(ValueError, match="mask"):
        fitted_estimator.score(x, y)

    np.testing.assert_array_equal(y.data, original.data)
    np.testing.assert_array_equal(y.mask, original.mask)


@pytest.mark.parametrize("name", ESTIMATORS)
@pytest.mark.parametrize("mask", [np.ma.nomask, False], ids=["no-mask", "all-false-mask"])
def test_unmasked_targets_preserve_fitted_results_and_inputs(name, mask):
    x, y = _data()
    unmasked = np.ma.array(y.copy(), mask=mask)
    original_mask = np.array(unmasked.mask, copy=True)
    expected = _estimator(name).fit(x, y)
    actual = _estimator(name).fit(x, unmasked)

    np.testing.assert_allclose(actual.predict(x), expected.predict(x), rtol=1e-12, atol=1e-12)
    assert actual.score(x, unmasked) == pytest.approx(expected.score(x, y), abs=1e-15)
    np.testing.assert_array_equal(unmasked.data, y)
    np.testing.assert_array_equal(unmasked.mask, original_mask)


@pytest.mark.parametrize(
    "invalid_time",
    [
        pytest.param(Decimal("NaN"), id="decimal-nan"),
        pytest.param(Decimal("sNaN"), id="decimal-signaling-nan"),
        pytest.param(Decimal("Infinity"), id="decimal-infinity"),
        pytest.param(Decimal("1e9999"), id="decimal-overflow"),
        pytest.param(Fraction(2**10000, 1), id="fraction-overflow"),
    ],
)
def test_target_validation_rejects_nonfinite_or_unrepresentable_object_times(invalid_time):
    common = importlib.import_module("survival._sklearn_common")
    x, y = _data(object)
    y[0, 0] = invalid_time

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="time"):
            common._validate_survival_data(x, y)
