"""The plain-scalar fast paths of survSplit, tmerge and survcondense.

Missing-value tests, storage modes, numeric views, factor validation and row
selection decide plain Python scalars from their types and read pandas numeric and
categorical columns as arrays.  Each is checked against the per-element algorithm it
replaces (kept here as the reference) on randomized inputs, and end to end.
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


def reference_factor_values(values, levels):
    declared = set(levels)
    return [
        None if factor_module._missing(value) or value not in declared else value
        for value in values
    ]


def reference_select_rows(values, rows):
    selected = list(map(values.__getitem__, rows))
    categories = r_coerce._categories(values)
    if categories is None:
        return selected
    ordered = getattr(values, "ordered", getattr(getattr(values, "dtype", None), "ordered", False))
    return Factor(selected, categories, bool(ordered))


def reference_split_data(data, rows):
    sources = r_data_prep._data_column_sources(data, "data")
    index = np.asarray(rows, dtype=np.intp)
    frame = {}
    for column, source in sources.items():
        if (
            type(source) is np.ndarray
            and source.ndim == 1
            and source.dtype.kind in "biuf"
            and len(source) > (index.max() if len(index) else -1)
        ):
            frame[column] = source[index].tolist()
        else:
            frame[column] = reference_select_rows(
                r_coerce._rows_of(source, r_coerce._materialize_1d(source, column)), rows
            )
    if len({len(values) for values in frame.values()}) > 1:
        raise ValueError("data columns must have equal lengths")
    return frame


def reference_row_codes(columns):
    keyed = [
        [None if value != value else value for value in column]
        if set(map(type, column)) <= {float, int}
        else [None if reference_is_missing(value) else value for value in column]
        for column in columns
    ]
    codes = {}
    return [codes.setdefault(row, len(codes)) for row in zip(*keyed, strict=True)]


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


def test_factor_validation_matches_the_missing_aware_test():
    rng = np.random.default_rng(4)
    level_pool = ["a", "b", 1, 2.0, True, None, NAN, np.float64(NAN)]
    value_pool = ["a", "b", "c", 1, 1.0, 2, 2.0, True, None, NAN, np.float64(NAN), np.float64(2.0)]
    for _ in range(2000):
        chosen = rng.choice(len(level_pool), int(rng.integers(0, 4)), replace=False)
        levels = [level_pool[i] for i in chosen.tolist()]
        values = _picks(rng, value_pool, int(rng.integers(0, 8)))
        factor = Factor(values, levels)
        assert _typed(list(factor)) == _typed(reference_factor_values(values, levels))
        assert factor.categories == tuple(levels)
    with pytest.raises(TypeError, match="unhashable"):
        Factor(["a", ["b"]], ["a"])


def test_row_selection_matches_python_indexing():
    rows = [3, 0, 0, -1, 2]
    sources = [
        ["a", "b", "c", "d"],
        ("a", "b", "c", "d"),
        Factor(["a", None, "b", "a"], ["b", "a"], ordered=True),
        r_coerce._RFactorVector(["x", "y", "x", None], ["y", "x"], ordered=False),
        np.array([1.0, NAN, 3.0, 4.0]),
    ]
    for source in sources:
        assert _typed(r_coerce._select_rows(source, rows)) == _typed(
            reference_select_rows(source, rows)
        )
    with pytest.raises(IndexError):
        r_coerce._select_rows(Factor(["a"], ["a"]), [1])


def test_row_codes_match_with_strings_logicals_and_missing_values():
    pool = ["a", "b", "", True, False, None, NAN, 1, 1.0, 2.5]
    rng = np.random.default_rng(5)
    for _ in range(500):
        n = int(rng.integers(0, 12))
        columns = [_picks(rng, pool, n) for _ in range(int(rng.integers(1, 4)))]
        assert r_data_prep._row_codes(columns) == reference_row_codes(columns), columns


def _same_frame(actual, expected, label=""):
    assert list(actual) == list(expected), label
    for name in expected:
        assert type(actual[name]) is type(expected[name]), (label, name)
        assert _typed(actual[name]) == _typed(expected[name]), (label, name)


def _split_inputs(pd, n=60):
    rng = np.random.default_rng(6)
    frame = pd.DataFrame(
        {
            "time": np.round(rng.exponential(size=n), 2) + 0.05,
            "status": rng.integers(0, 2, n),
            "x": rng.normal(size=n),
            "count": rng.integers(-3, 4, n).astype(np.int32),
            "flag": rng.integers(0, 2, n).astype(bool),
            "big": rng.integers(0, 2**62, n, dtype=np.uint64),
            "small": rng.normal(size=n).astype(np.float32),
            "g": pd.Categorical(rng.choice(["a", "b", "c"], n), categories=["c", "a", "b", "z"]),
            "grade": pd.Categorical(
                rng.choice(["low", "high"], n), categories=["low", "high"], ordered=True
            ),
            "code": pd.Categorical(rng.choice([10, 20], n)),
            "label": [f"s{i % 7}" for i in range(n)],
            "when": pd.date_range("2021-01-01", periods=n),
        }
    )
    frame.loc[::9, "x"] = NAN
    frame.loc[::11, "g"] = None
    frame.loc[::13, "code"] = None
    masked = frame.copy()
    masked["count"] = masked["count"].astype("Int64")
    masked.loc[::5, "count"] = pd.NA
    masked["x"] = masked["x"].astype("Float64")
    masked.loc[::4, "x"] = pd.NA
    masked["flag"] = masked["flag"].astype("boolean")
    masked.loc[::6, "flag"] = pd.NA
    yield "numpy_dtypes", frame
    yield "masked", masked
    yield "filtered", frame[frame.time > 0.2]
    yield "dict", {name: frame[name].to_list() for name in frame.columns if name != "when"}


def test_split_data_matches_the_per_value_selection(monkeypatch):
    pd = pytest.importorskip("pandas")
    for name, data in _split_inputs(pd):
        for formula, options in (
            ("Surv(time, status) ~ .", {"cut": [0.3, 0.8, 1.5], "episode": "ep"}),
            ("Surv(time, status) ~ x + g", {"cut": [0.5], "id": "sid", "zero": 0.0}),
        ):
            actual = r.survSplit(formula, data, **options)
            with monkeypatch.context() as patched:
                patched.setattr(r_data_prep, "_split_data", reference_split_data)
                expected = r.survSplit(formula, data, **options)
            _same_frame(actual, expected, f"{name}: {formula}")
    masked = dict(_split_inputs(pd))["masked"]
    split = r.survSplit("Surv(time, status) ~ .", masked, cut=[0.5])
    assert any(value is pd.NA for value in split["count"])
    assert any(value is pd.NA for value in split["x"])


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


def test_survcondense_matches_the_per_value_row_codes(monkeypatch):
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(9)
    n = 50
    data = pd.DataFrame(
        {
            "time": np.round(rng.exponential(size=n) * 4, 1) + 0.5,
            "status": rng.integers(0, 2, n),
            "x": rng.integers(0, 2, n).astype(float),
            "s": rng.choice(["u", "v"], n).astype(object),
            "b": rng.integers(0, 2, n).astype(bool),
        }
    )
    split = pd.DataFrame(r.survSplit("Surv(time, status) ~ .", data, cut=[1.0, 2.0], id="sid"))
    split.loc[::6, "s"] = None
    actual = r.survcondense("Surv(tstart, time, status) ~ x + s + b", split, id="sid")
    with monkeypatch.context() as patched:
        patched.setattr(r_data_prep, "_row_codes", reference_row_codes)
        expected = r.survcondense("Surv(tstart, time, status) ~ x + s + b", split, id="sid")
    _same_frame(actual, expected)
    assert len(actual["sid"]) < len(split)
