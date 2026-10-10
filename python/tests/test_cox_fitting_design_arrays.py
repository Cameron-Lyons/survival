"""Array-backed Cox fitting preserves the complete row-list fit and public methods."""

import importlib
import pickle
import warnings

import numpy as np
import pytest

from .helpers import setup_survival_import
from .r_fixture_support import RFactor

survival = setup_survival_import()
r = survival.r
coxph = importlib.import_module("survival.r._coxph")
model_frame = importlib.import_module("survival.r._fit")


def _data(layout="lists"):
    data = dict(survival.datasets.load_ovarian())
    n = len(data["futime"])
    data.update(
        start=[0.0] * n,
        off=[707.0 + value for value in data["ecog.ps"]],
        weight=[1.0 + (row % 3) / 2 for row in range(n)],
        subject=[row // 2 for row in range(n)],
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


def _fit(formula, data, **options):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = r.coxph(formula, data, **options)
    return result, [(item.category, str(item.message)) for item in caught]


def _list_model_frame(*args, **kwargs):
    kwargs["as_array"] = False
    return model_frame._model_frame(*args, **kwargs)


@pytest.mark.parametrize("layout", ["lists", "strided", "readonly", "iterators"])
@pytest.mark.parametrize("method", ["efron", "breslow", "exact"])
@pytest.mark.parametrize(
    ("formula", "options"),
    [
        ("Surv(futime, fustat) ~ 1", {}),
        (
            "Surv(futime, fustat) ~ age * rx + offset(off)",
            {"init": [0.0, 0.0, 0.0], "weights": "weight", "cluster": "subject"},
        ),
        (
            "Surv(start, futime, fustat) ~ age + rx + offset(off)",
            {"weights": "weight", "cluster": "subject", "init": [0.01, -0.1]},
        ),
        (
            "Surv(futime, fustat) ~ log(age) * I(rx + 0.5) + strata(ecog.ps)",
            {"subset": list(range(25, 0, -1)), "iter_max": 1},
        ),
    ],
)
def test_numeric_fits_and_public_methods_match_row_lists(
    monkeypatch, layout, method, formula, options
):
    # Exact fits do not support non-unit weights or robust variance.
    if method == "exact" and "weights" in options:
        options = {
            **{name: value for name, value in options.items() if name != "weights"},
            "robust": False,
        }
    actual, actual_warnings = _fit(formula, _data(layout), method=method, model=True, **options)
    with monkeypatch.context() as patch:
        patch.setattr(coxph, "_model_frame", _list_model_frame)
        reference, reference_warnings = _fit(
            formula, _data(layout), method=method, model=True, **options
        )
    assert actual_warnings == reference_warnings
    # Native serialization includes every numerical fit field, not just coefficients.
    assert pickle.dumps(actual.fit) == pickle.dumps(reference.fit)
    assert actual.coef_names == reference.coef_names
    assert actual.assign == reference.assign
    assert actual.strata_levels == reference.strata_levels
    assert actual.na_action == reference.na_action
    assert r.model_frame(actual) == r.model_frame(reference)
    assert r.model_matrix(actual) == r.model_matrix(reference)
    assert isinstance(actual.x, list)
    assert all(isinstance(row, list) for row in actual.x)
    assert actual._frame.x == []
    for kind in ("lp", "risk", "expected"):
        assert pickle.dumps(r.predict(actual, type=kind, se_fit=True)) == pickle.dumps(
            r.predict(reference, type=kind, se_fit=True)
        )
    assert pickle.dumps(r.residuals(actual)) == pickle.dumps(r.residuals(reference))
    restored = pickle.loads(pickle.dumps(actual))  # noqa: S301 - local model round trip
    assert r.model_matrix(restored) == r.model_matrix(reference)
    assert pickle.dumps(restored.fit) == pickle.dumps(reference.fit)


@pytest.mark.parametrize("action", ["na.omit", "na.exclude"])
def test_array_fits_preserve_missing_rows_labels_and_repeated_subset(monkeypatch, action):
    from survival.pybridge import _r_data_frame

    data = _data()
    n = len(data["futime"])
    data["age"][4] = None
    frame = _r_data_frame(data, n, [f"row {row}" for row in range(n)])
    options = {"na_action": action, "subset": [5, 4, 3, 2, 1, 0, 5], "iter_max": 0}
    actual, actual_warnings = _fit("Surv(futime, fustat) ~ age + rx", frame, **options)
    with monkeypatch.context() as patch:
        patch.setattr(coxph, "_model_frame", _list_model_frame)
        reference, reference_warnings = _fit("Surv(futime, fustat) ~ age + rx", frame, **options)
    assert actual_warnings == reference_warnings
    assert pickle.dumps(actual.fit) == pickle.dumps(reference.fit)
    assert actual.na_action == reference.na_action
    assert r.model_matrix(actual) == r.model_matrix(reference)
    assert pickle.dumps(r.predict(actual, _with_row_names=True)) == pickle.dumps(
        r.predict(reference, _with_row_names=True)
    )
    assert pickle.dumps(r.residuals(actual)) == pickle.dumps(r.residuals(reference))


@pytest.mark.parametrize(
    ("events", "predictor", "init"),
    [
        ([0, 0, 0, 0], [1, float("inf"), 3, 4], [1, 2]),
        ([1, 0, 1, 0], [1, float("inf"), 3, 4], None),
        ([1, 0, 1, 0], [1, 2, 3, 4], [1, 2]),
        ([1, 0, 1, 0], [1, 2, 3, 4], [1000]),
        ([1, 0, 1, 0], [1, 2, 3, 4], [float("inf")]),
    ],
)
def test_no_events_and_input_errors_match_row_lists(monkeypatch, events, predictor, init):
    data = {"time": [1, 2, 3, 4], "event": events, "x": predictor}

    def outcome():
        try:
            fit, messages = _fit("Surv(time, event) ~ x", data, init=init)
        except (TypeError, ValueError) as error:
            return type(error), str(error)
        return pickle.dumps(fit.fit), messages

    actual = outcome()
    with monkeypatch.context() as patch:
        patch.setattr(coxph, "_model_frame", _list_model_frame)
        assert actual == outcome()


def test_numeric_model_frame_owns_matrix_and_take_preserves_empty_columns():
    data = _data("readonly")
    frame = model_frame._model_frame("Surv(futime, fustat) ~ age * rx", data, as_array=True)
    assert isinstance(frame.x, np.ndarray)
    assert frame.x.flags.owndata
    assert frame.x.flags.c_contiguous
    assert frame.x.flags.writeable
    assert not np.shares_memory(frame.x, data["age"])
    subset = frame.take([2, 0, 2])
    np.testing.assert_array_equal(subset.x, frame.x[[2, 0, 2]])
    assert not np.shares_memory(subset.x, frame.x)
    null = model_frame._model_frame("Surv(futime, fustat) ~ 1", data, as_array=True)
    assert null.x.shape == (26, 0)
    repeated = null.take([2, 0, 2])
    assert repeated.x.shape == (3, 0)
    assert repeated.n == 3
    assert repeated.matrix_rows() == [[], [], []]


@pytest.mark.parametrize(
    "term", ["factor(rx)", "age * factor(rx)", "ridge(age, theta=1)", "tt(age)"]
)
def test_special_designs_keep_row_lists(term):
    frame = model_frame._model_frame(
        f"Surv(futime, fustat) ~ {term}", _data(), as_array=True, defer_tt=True
    )
    assert isinstance(frame.x, list)
    assert all(isinstance(row, list) for row in frame.x)


def test_multistate_and_other_model_consumers_keep_row_lists():
    data = {"time": [1, 2, 3, 4], "state": RFactor(["a", "b", "a", "b"], ["a", "b"])}
    frame = model_frame._model_frame("Surv(time, state) ~ time", data, as_array=True)
    assert frame.y.type == "mright"
    assert isinstance(frame.x, list)
    ordinary = model_frame._model_frame("Surv(futime, fustat) ~ age", _data())
    assert isinstance(ordinary.x, list)
    assert ordinary.matrix_rows() is ordinary.x
