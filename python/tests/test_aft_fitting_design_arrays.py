"""Owned array designs preserve AFT fits, retained model data and public methods."""

import importlib
import pickle
import warnings

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
survreg = importlib.import_module("survival.r._survreg")
formula = importlib.import_module("survival.r._formula")


def _data(layout="lists"):
    data = dict(survival.datasets.load_ovarian())
    n = len(data["futime"])
    data.update(
        weight=[1.0 + (row % 3) / 2 for row in range(n)],
        subject=[row // 2 for row in range(n)],
        off=[0.01 * value for value in data["ecog.ps"]],
    )
    if layout == "iterators":
        return {name: iter(values) for name, values in data.items()}
    if layout == "strided":
        return {name: np.repeat(np.asarray(values), 2)[::2] for name, values in data.items()}
    if layout == "readonly":
        result = {name: np.asarray(values) for name, values in data.items()}
        for values in result.values():
            values.flags.writeable = False
        return result
    return data


def _fit(spec, data, **options):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = r.survreg(spec, data, **options)
    return fit, [(message.category, str(message.message)) for message in caught]


@pytest.mark.parametrize("layout", ["lists", "strided", "readonly", "iterators"])
@pytest.mark.parametrize("dist", ["weibull", "gaussian", "loglogistic"])
@pytest.mark.parametrize(
    "terms",
    ["1", "age * rx + offset(off)", "age * factor(rx)", "age + strata(ecog.ps)"],
)
def test_fits_and_methods_match_row_designs(monkeypatch, layout, dist, terms):
    options = {
        "dist": dist,
        "weights": "weight",
        "cluster": "subject",
        "model": True,
        "x": True,
        "score": True,
    }
    spec = f"Surv(futime, fustat) ~ {terms}"
    actual, messages = _fit(spec, _data(layout), **options)
    with monkeypatch.context() as patch:
        patch.setattr(survreg, "_design_array_from_spec", formula._design_rows_from_spec)
        reference, expected_messages = _fit(spec, _data(layout), **options)
    assert messages == expected_messages
    assert pickle.dumps(actual.fit) == pickle.dumps(reference.fit)
    assert actual.coefficient_names == reference.coefficient_names
    assert actual.assign == reference.assign
    assert actual.strata_levels == reference.strata_levels
    assert r.model_frame(actual) == r.model_frame(reference)
    assert r.model_matrix(actual) == r.model_matrix(reference)
    assert isinstance(actual.x, list)
    assert all(isinstance(row, list) for row in actual.x)
    assert actual.x == reference.x
    for result in (r.model_summary, r.predict, r.residuals):
        assert pickle.dumps(result(actual)) == pickle.dumps(result(reference))
    restored = pickle.loads(pickle.dumps(actual))
    assert restored.x == actual.x
    assert pickle.dumps(restored.fit) == pickle.dumps(actual.fit)


@pytest.mark.parametrize("terms", ["ridge(age, theta=1)", "pspline(age, df=3)"])
def test_penalized_designs_match_row_designs(monkeypatch, terms):
    spec = f"Surv(futime, fustat) ~ {terms} + rx"
    actual, messages = _fit(spec, _data("strided"), x=True)
    with monkeypatch.context() as patch:
        patch.setattr(survreg, "_design_array_from_spec", formula._design_rows_from_spec)
        reference, expected_messages = _fit(spec, _data("strided"), x=True)
    assert messages == expected_messages
    assert pickle.dumps(actual.penalized) == pickle.dumps(reference.penalized)
    assert actual.x == reference.x


@pytest.mark.parametrize("action", ["omit", "exclude"])
def test_missing_repeated_and_reordered_rows_match_row_designs(monkeypatch, action):
    data = _data()
    data["age"][2] = float("nan")
    options = {"na_action": action, "subset": [5, 4, 3, 2, 1, 0, 5], "iter_max": 0}
    spec = "Surv(futime, fustat) ~ age + rx"
    actual, messages = _fit(spec, data, **options)
    with monkeypatch.context() as patch:
        patch.setattr(survreg, "_design_array_from_spec", formula._design_rows_from_spec)
        reference, expected_messages = _fit(spec, data, **options)
    assert messages == expected_messages
    assert pickle.dumps(actual.fit) == pickle.dumps(reference.fit)
    assert actual.na_action == reference.na_action
    np.testing.assert_equal(r.predict(actual), r.predict(reference))
    np.testing.assert_equal(r.residuals(actual), r.residuals(reference))


def test_formula_frame_owns_a_contiguous_matrix():
    data = _data("readonly")
    frame = survreg._formula_frame(
        "Surv(futime, fustat) ~ age * rx",
        data,
        weights=None,
        subset=None,
        na_action="omit",
        offset=None,
        cluster=None,
        keep_model=False,
    )
    assert isinstance(frame.x, np.ndarray)
    assert frame.x.flags.owndata
    assert frame.x.flags.c_contiguous
    assert frame.x.flags.writeable
    assert not np.shares_memory(frame.x, data["age"])


def test_infinite_predictor_error_precedes_distribution_validation():
    data = _data("strided")
    data["age"][0] = float("inf")
    with pytest.raises(ValueError, match="data contains an infinite predictor"):
        r.survreg("Surv(futime, fustat) ~ age", data, dist="invalid")
