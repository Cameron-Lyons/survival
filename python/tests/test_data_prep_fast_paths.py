"""The plain-scalar fast paths of the missing-value test and of tmerge.

Missing-value tests, storage modes and numeric views decide plain Python scalars from
their types.  Each is checked against the per-element algorithm it replaces (kept
here as the reference) on randomized inputs, and end to end.
"""

from __future__ import annotations

import decimal
import importlib
import math
import numbers
import pickle

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
r_coerce = importlib.import_module("survival.r._coerce")
r_data_prep = importlib.import_module("survival.r._data_prep")
factor_module = importlib.import_module("survival._factor")
Factor = factor_module.Factor

NAN = math.nan


def reference_is_missing(value):
    if value is None or type(value).__name__ in {"NAType", "NaTType"}:
        return True
    if isinstance(value, np.ma.MaskedArray) and value.size == 1 and np.ma.is_masked(value):
        return True
    try:
        return bool(value != value)
    except Exception:
        return False


def reference_storage_mode(values):
    if values is None:
        return "integer"
    observed = [value for value in values if not reference_is_missing(value)]
    if any(isinstance(value, str) for value in observed):
        return "character"
    if not observed:
        return "double" if any(isinstance(value, numbers.Real) for value in values) else "logical"
    if all(r_coerce._is_bool_like(value) for value in observed):
        return "logical"
    if all(isinstance(value, numbers.Integral) for value in observed):
        return "integer"
    return "double"


def reference_numeric_values(values):
    result = []
    for value in values:
        if reference_is_missing(value):
            result.append(NAN)
        elif isinstance(value, str):
            return None
        else:
            try:
                result.append(float(value))
            except (TypeError, ValueError):
                return None
    return result


def reference_numeric_or_nan(values, name):
    result = []
    for value in r_coerce._materialize_1d(values, name):
        if reference_is_missing(value):
            result.append(NAN)
            continue
        try:
            result.append(float(value))
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be numeric") from exc
    return result


def _outcome(function, *args):
    try:
        return ("value", _typed(function(*args)))
    except Exception as exc:
        return ("error", type(exc), str(exc))


def _typed(value):
    """A value comparable bit for bit: element types, NaN, factor levels and order."""

    if isinstance(value, Factor):
        return ("Factor", _typed(list(value)), value.categories, value.ordered)
    if isinstance(value, list | tuple):
        return (type(value), [_typed(item) for item in value])
    if isinstance(value, float):
        return (float, value.hex())
    return (type(value), value)


class FloatSub(float):
    pass


class IntSub(int):
    pass


SCALARS = [
    0.0,
    -0.0,
    1.5,
    NAN,
    -NAN,
    math.inf,
    -math.inf,
    0,
    7,
    -3,
    10**30,
    True,
    False,
    "",
    "a",
    "NA",
    None,
]
SPECIAL = [
    np.float64(NAN),
    np.float64(2.0),
    np.float32(NAN),
    np.int64(4),
    np.bool_(True),
    np.str_("b"),
    np.datetime64("NaT"),
    np.datetime64("2020-01-01"),
    np.ma.masked_array([1.0], mask=[True])[0],
    np.ma.masked_array([1.0], mask=[True]),
    decimal.Decimal("NaN"),
    decimal.Decimal("1.5"),
    FloatSub(NAN),
    FloatSub(1.0),
    IntSub(2),
    [1, 2],
    (NAN,),
    object(),
]


def _special_values():
    values = list(SPECIAL)
    try:
        import pandas as pd
    except ImportError:
        return values
    return [*values, pd.NA, pd.NaT, pd.Timestamp("2020-01-02")]


def test_missing_value_test_matches_the_general_test():
    for value in SCALARS + _special_values():
        assert r_coerce._is_missing_value(value) == reference_is_missing(value), repr(value)


def _picks(rng, pool, size):
    """*size* draws from *pool*, a list of mixed Python objects kept as they are."""

    return [pool[i] for i in rng.integers(len(pool), size=size).tolist()]


def _random_lists(pool, count, seed):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        yield _picks(rng, pool, int(rng.integers(0, 6)))


def test_storage_mode_matches_the_per_value_rules():
    plain = [True, False, 0, 3, 1.5, NAN, None, "x"]
    cases = [
        [True, False, NAN],
        [1, 2, NAN],
        [1, True, NAN],
        [NAN, None],
        [None],
        [NAN],
        [],
        None,
    ]
    cases += list(_random_lists(plain, 3000, 1))
    cases += list(_random_lists([*plain, np.float64(1.0), np.int64(2), np.bool_(True)], 500, 2))
    for values in cases:
        assert r_data_prep._storage_mode(values) == reference_storage_mode(values), values


def test_numeric_views_match_the_per_value_conversion():
    pool = [True, False, 0, -4, 2.5, NAN, -NAN, math.inf, None, 10**400, "1", np.float64(NAN)]
    for values in _random_lists(pool, 3000, 3):
        assert _outcome(r_data_prep._numeric_values, values) == _outcome(
            reference_numeric_values, values
        ), values
        assert _outcome(r_data_prep._numeric_or_nan, values, "x") == _outcome(
            reference_numeric_or_nan, values, "x"
        ), values


def test_tmerge_keeps_logical_and_integer_updates_with_missing_values():
    base = {"id": [1, 2, 3], "futime": [5.0, 6.0, 7.0]}
    first = r.tmerge(base, base, id="id", death=r.event("futime"))
    labs = {
        "id": [1, 1, 2, 3],
        "day": [1.0, 3.0, 2.0, 4.0],
        "flag": [NAN, True, False, True],
        "count": [1, NAN, 2, 3],
        "mixed": [1, True, NAN, 0],
    }
    merged = r.tmerge(
        first,
        labs,
        id="id",
        fl=r.tdc("day", "flag"),
        ct=r.tdc("day", "count"),
        mx=r.tdc("day", "mixed"),
    )
    columns = merged.columns
    assert _typed(columns["fl"]) == _typed([NAN, NAN, True, NAN, False, NAN, True])
    assert _typed(columns["ct"]) == _typed([NAN, 1, 1, NAN, 2, NAN, 3])
    assert _typed(columns["mx"]) == _typed([NAN, 1, True, NAN, NAN, NAN, 0])


def test_tmerge_chain_matches_the_per_value_helpers(monkeypatch):
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(8)
    n = 80
    base = pd.DataFrame(
        {
            "id": np.arange(n) * 3,
            "futime": np.round(rng.exponential(size=n) * 10, 2) + 1.0,
            "status": rng.integers(0, 2, n),
            "g": pd.Categorical(rng.choice(["a", "b"], n)),
        }
    )
    labs = pd.DataFrame(
        {
            "id": np.repeat(base.id.to_numpy(), 2),
            "day": np.round(base.futime.to_numpy().repeat(2) * rng.uniform(0.1, 0.9, 2 * n), 3),
            "value": rng.normal(size=2 * n),
            "flag": rng.integers(0, 2, 2 * n).astype(bool),
            "count": rng.integers(0, 4, 2 * n),
        }
    )
    labs.loc[::7, "value"] = NAN
    labs["flag_na"] = labs["flag"].astype(object)
    labs.loc[::5, "flag_na"] = NAN

    def chain():
        first = r.tmerge(base, base, id="id", death=r.event("futime", "status"))
        second = r.tmerge(first, labs, id="id", lab=r.tdc("day", "value"))
        third = r.tmerge(
            second,
            labs,
            id="id",
            fl=r.tdc("day", "flag"),
            fn=r.tdc("day", "flag_na"),
            total=r.cumtdc("day", "count"),
            visits=r.cumevent("day"),
        )
        return r.tmerge(third, labs, id="id", seen=r.event("day"), lab2=r.tdc("day", "value"))

    def summary(frame):
        return (
            {name: _typed(values) for name, values in frame.columns.items()},
            frame.tname,
            frame.tevent,
            frame.tdcvar,
            frame.tcount,
        )

    actual = summary(chain())
    with monkeypatch.context() as patched:
        patched.setattr(r_data_prep, "_is_missing_value", reference_is_missing)
        patched.setattr(r_data_prep, "_storage_mode", reference_storage_mode)
        patched.setattr(r_data_prep, "_numeric_values", reference_numeric_values)
        patched.setattr(r_data_prep, "_numeric_or_nan", reference_numeric_or_nan)
        expected = summary(chain())
    assert pickle.dumps(actual) == pickle.dumps(expected)
