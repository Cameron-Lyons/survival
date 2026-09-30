"""The R state envelope carries the same complete model as native Python pickle."""

import io
import pickle

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
from survival.pybridge import _serialize_r_object, _unserialize_r_object  # noqa: E402


@pytest.mark.parametrize(
    ("fitter", "formula"),
    [
        ("coxph", "Surv(time, status) ~ age + sex"),
        ("coxph", "Surv(time, status) ~ ridge(age, theta=1) + sex"),
        ("survreg", "Surv(time, status) ~ age + strata(sex)"),
        ("survreg", "Surv(time, status) ~ pspline(age, df=3) + sex"),
    ],
)
def test_r_envelope_preserves_fitted_methods(fitter, formula):
    fit = getattr(survival.r, fitter)(formula, survival.datasets.load_lung())
    state = _serialize_r_object(fit)
    assert state["version"] == 1
    assert isinstance(state["pickle"], bytearray)
    assert state["callbacks"] == []
    restored = _unserialize_r_object(state)
    assert type(restored) is type(fit)
    np.testing.assert_array_equal(survival.r.coef(restored), survival.r.coef(fit))
    np.testing.assert_array_equal(survival.r.vcov(restored), survival.r.vcov(fit))
    np.testing.assert_array_equal(
        survival.r.predict(restored, type="lp"), survival.r.predict(fit, type="lp")
    )
    np.testing.assert_array_equal(
        survival.r.residuals(restored, type="deviance"),
        survival.r.residuals(fit, type="deviance"),
    )


@pytest.mark.parametrize(
    ("state", "message"),
    [
        (None, "version"),
        ({"version": 2}, "version"),
        ({"version": 1}, "byte payload"),
        ({"version": 1, "pickle": "bytes", "callbacks": []}, "byte payload"),
        ({"version": 1, "pickle": b"", "callbacks": None}, "R callbacks"),
        ({"version": 1, "pickle": b"", "callbacks": [1]}, "R callbacks"),
    ],
)
def test_invalid_r_envelope_is_rejected(state, message):
    with pytest.raises(ValueError, match=message):
        _unserialize_r_object(state)


@pytest.mark.parametrize("key", [("wrong", 0), ("r_callback", -1), ("r_callback", True), 0])
def test_invalid_persistent_callback_reference_is_rejected(key):
    sentinel = object()

    class PersistentPickler(pickle.Pickler):
        def persistent_id(self, value):
            return key if value is sentinel else None

    stream = io.BytesIO()
    PersistentPickler(stream).dump(sentinel)
    with pytest.raises(pickle.UnpicklingError, match="invalid R callback reference"):
        _unserialize_r_object({"version": 1, "pickle": stream.getvalue(), "callbacks": []})


def test_repeated_persistent_callback_references_keep_identity():
    sentinel = object()

    class PersistentPickler(pickle.Pickler):
        def persistent_id(self, value):
            return ("r_callback", 0) if value is sentinel else None

    stream = io.BytesIO()
    PersistentPickler(stream).dump([sentinel, sentinel])
    restored = _unserialize_r_object(
        {"version": 1, "pickle": stream.getvalue(), "callbacks": [np.add]}
    )
    assert restored[0] is restored[1] is np.add


def test_r_envelope_captures_current_state_each_time():
    value = {"coefficients": [1, 2]}
    before = _serialize_r_object(value)
    value["coefficients"][0] = 3
    after = _serialize_r_object(value)
    assert _unserialize_r_object(before)["coefficients"] == [1, 2]
    assert _unserialize_r_object(after)["coefficients"] == [3, 2]
