"""Model-frame shortcuts: column lengths, the response NA scan and categorical codes.

Row counts read a numeric column's length without copying it, ``is.na(Surv)`` skips
the array scan of a column whose sum shows it complete, and ``factor()`` of a pandas
categorical takes its codes.  Each must agree exactly with the per-element path (kept
here as the reference), including its errors.
"""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
r_coerce = importlib.import_module("survival.r._coerce")
r_formula = importlib.import_module("survival.r._formula")
r_surv = importlib.import_module("survival.r._surv")

NAN = math.nan


def reference_missing_rows(x):
    columns = (x.time, x.status) if isinstance(x, r.Surv2) else (x.start, x.time, x.time2, x.event)
    missing = np.zeros(len(x), dtype=bool)
    for column in columns:
        if column is not None:
            missing |= np.isnan(np.array(column, dtype=np.float64))
    return missing


def reference_factor(values, name="values"):
    declared = r_coerce._categories(values)
    levels = [level for level in declared if not r_coerce._is_missing_value(level)]
    index = {level: code for code, level in enumerate(levels)}
    materialized = r_coerce._materialize_labels(values, name)
    codes = list(map(index.get, materialized))
    if None in codes:
        for value, code in zip(materialized, codes, strict=True):
            if code is None and not r_coerce._is_missing_value(value):
                raise ValueError(f"{name} contains a value outside the declared categories")
    return codes, [r_coerce._as_character(level) for level in levels]


def _outcome(function, *args):
    try:
        return ("value", function(*args))
    except Exception as exc:
        return ("error", type(exc), str(exc))


def _length_sources():
    yield [1.0, 2.0, 3.0]
    yield (1, 2)
    yield np.arange(4.0)
    yield np.arange(5, dtype=np.int16)
    yield np.array([True, False])
    yield np.array([], dtype=float)
    yield np.arange(6.0).reshape(3, 2)
    yield np.array(["a", "b"])
    yield np.array([1.0, None], dtype=object)
    yield np.ma.masked_invalid([1.0, NAN, 3.0])
    yield [[1, 2], [3, 4]]
    yield "abc"
    yield 3
    try:
        import pandas as pd
    except ImportError:
        return
    yield pd.Series([1.0, NAN])
    yield pd.Series([1, 2, 3], dtype="Int64")
    yield pd.Series(pd.Categorical(["a", "b"]))
    yield pd.Series(["a", None], dtype=object)
    yield pd.Series(pd.date_range("2020-01-01", periods=3))


def test_column_length_matches_the_materialized_length():
    for source in _length_sources():
        for materialize in (r_coerce._materialize_1d, r_coerce._coerce_array_like):
            expected = _outcome(lambda s, m=materialize: len(m(s, "x")), source)
            actual = _outcome(lambda s, m=materialize: r_formula._column_length(s, "x", m), source)
            assert actual == expected, (type(source), materialize.__name__)


def _responses():
    rng = np.random.default_rng(2)
    pool = [0.5, 1.0, 2.0, NAN, math.inf, -math.inf]
    for _ in range(300):
        n = int(rng.integers(0, 8))
        time = [pool[i] for i in rng.integers(len(pool), size=n).tolist()]
        start = [pool[i] for i in rng.integers(4, size=n).tolist()]
        event = [(0, 1, None)[i] for i in rng.integers(3, size=n).tolist()]
        yield r.Surv._from_normalized(
            time=time, event=event, start=None, time2=None, surv_type="right"
        )
        yield r.Surv._from_normalized(
            time=time, event=event, start=start, time2=None, surv_type="counting"
        )
        yield r.Surv._from_normalized(
            time=time, event=event, start=None, time2=start, surv_type="interval"
        )
    yield r.Surv([1.0, NAN, 3.0], [1, 0, 1])
    yield r.Surv2([1.0, NAN, 2.0], [1, 0, 1])


def test_response_missing_rows_match_the_array_scan():
    for response in _responses():
        expected = reference_missing_rows(response)
        actual = r_surv._missing_rows(response)
        assert actual.dtype == expected.dtype
        assert actual.tolist() == expected.tolist(), (response.time, response.event)


def test_omitted_response_rows_are_the_missing_ones():
    rng = np.random.default_rng(3)
    n = 50
    time = np.round(rng.exponential(size=n), 2) + 0.1
    status = rng.integers(0, 2, n).astype(float)
    time[[2, 9, 30]] = NAN
    status[[9, 14]] = NAN
    time[20] = math.inf
    data = {"time": time, "status": status, "x": rng.normal(size=n)}
    for formula in ("Surv(time, status) ~ x", "Surv(time, status) ~ 1"):
        _rows, _aligned, removed = r_formula._apply_formula_na_action(formula, data, "na.omit")
        assert removed == [2, 9, 14, 30]
    time[20] = 1.0
    fit = r.coxph("Surv(time, status) ~ x", data)
    assert [row - 1 for row in fit.na_action.rows] == [2, 9, 14, 30]


def _categoricals(pd):
    values = ["b", "a", None, "c", "a"]
    yield pd.Series(pd.Categorical(values))
    yield pd.Series(pd.Categorical(values, categories=["c", "b", "a", "unused"]))
    yield pd.Series(pd.Categorical(values, categories=["c", "a", "b"], ordered=True))
    yield pd.Categorical(values, categories=["a", "b", "c"])
    yield pd.CategoricalIndex(values)
    yield pd.Series(pd.Categorical([3, 1, None, 2]))
    yield pd.Series(pd.Categorical([1.5, NAN, 0.5]))
    yield pd.Series(pd.Categorical([True, False, None]))
    yield pd.Series(pd.Categorical([1, "1", None]))
    yield pd.Series(pd.Categorical([None, None], categories=["x"]))
    yield pd.Series(pd.Categorical(pd.date_range("2020-01-01", periods=3)))
    yield pd.Series(pd.Categorical([], categories=["a"]))


def test_categorical_factor_codes_match_the_category_lookup():
    pd = pytest.importorskip("pandas")
    for values in _categoricals(pd):
        expected = _outcome(reference_factor, values)
        actual = _outcome(r_coerce._factor, values)
        assert actual == expected, values
        if actual[0] == "value":
            assert all(code is None or type(code) is int for code in actual[1][0])


def test_strata_of_a_categorical_column_keep_their_codes(monkeypatch):
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(4)
    n = 60
    frame = pd.DataFrame(
        {
            "time": np.round(rng.exponential(size=n), 2) + 0.1,
            "status": rng.integers(0, 2, n),
            "g": pd.Categorical(rng.choice(["lo", "mid", "hi"], n), categories=["mid", "hi", "lo"]),
            "x": rng.normal(size=n),
        }
    )
    frame.loc[[5, 17], "g"] = None

    def fits():
        km = r.survfit("Surv(time, status) ~ g", frame)
        cox = r.coxph("Surv(time, status) ~ x + strata(g)", frame)
        return km.strata, list(km.surv), cox.coefficients, cox.strata_levels, cox.na_action.rows

    actual = fits()
    with monkeypatch.context() as patched:
        patched.setattr(r_coerce, "_category_codes", lambda values, categories: None)
        expected = fits()
    assert actual == expected
