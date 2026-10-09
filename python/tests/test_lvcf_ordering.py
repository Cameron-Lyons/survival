"""Last-value carry-forward uses one subject/time order for initialization and copying."""

import importlib
import math

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
_r_factor = importlib.import_module("survival.r._coerce")._r_factor


@pytest.mark.parametrize(
    ("ids", "values", "times", "expected"),
    [
        ([1, 1, 1], [None, None, 1], [math.nan, 1, 2], [1, 0, 1]),
        ([1, 1, 1], [None, None, True], [None, 1, 2], [True, False, True]),
        ([1, 1, 1], [None, None, 1], [2, math.nan, 1], [1, 1, 1]),
        ([1, 1, 1], [None, None, None], [math.nan, math.nan, 1], [False, False, False]),
        ([1, 1, 1], [None, None, 2.5], [math.nan, 1, 2], [2.5, None, 2.5]),
        ([1, 1, 1], [None, None, "a"], [math.nan, 1, 2], ["a", None, "a"]),
        ([1, 1, 1], [None, 1, None], [1, 1, math.nan], [0, 1, 1]),
        (
            [1, 1, 1, 1, 1],
            [None, None, None, None, 1],
            [math.nan, math.inf, -math.inf, 1, 1],
            [1, 1, 0, 0, 1],
        ),
        (
            [2, 1, 2, 1, 2, 1],
            [None, 1, None, None, 1, None],
            [math.nan, 2, 1, math.nan, 2, 1],
            [1, 1, 0, 1, 1, 0],
        ),
    ],
)
def test_initial_values_follow_r_missing_times_last(ids, values, times, expected):
    # R/xtras.R initializes the first observation after order(id, time), whose
    # default puts missing times last. Equal times keep their input order.
    assert r.lvcf(ids, values, time=times) == expected


@pytest.mark.parametrize("first", [True, False])
@pytest.mark.parametrize("storage", ["list", "numpy", "pandas", "polars"])
def test_carry_forward_preserves_input_rows_and_storage(first, storage):
    ids = ["b", "a", "b", "a", "b", "a"]
    values = [None, 1, None, None, 1, None]
    times = [math.nan, 2, 1, math.nan, 2, 1]
    if storage == "numpy":
        ids = np.asarray(ids)
        values = np.asarray(values, dtype=object)
        times = np.asarray(times)
    elif storage == "pandas":
        pd = pytest.importorskip("pandas")
        ids, values, times = (pd.Series(column) for column in (ids, values, times))
    elif storage == "polars":
        pl = pytest.importorskip("polars")
        ids, values, times = (pl.Series(column) for column in (ids, values, times))
    expected = [1, 1, 0 if first else None, 1, 1, 0 if first else None]
    actual = r.lvcf(ids, values, time=times, first=first)
    assert all(
        observed == reference if reference is not None else observed is None or math.isnan(observed)
        for observed, reference in zip(actual, expected, strict=True)
    )
    assert values[0] is None or math.isnan(values[0])


def test_first_false_keeps_unknown_initial_values():
    assert r.lvcf([1, 1, 1], [None, None, 1], time=[math.nan, 1, 2], first=False) == [
        1,
        None,
        1,
    ]


def test_initialization_uses_native_subject_identity():
    # Numeric and character IDs remain distinct in the Python native API;
    # integral floating-point IDs have the same identity as integers.
    assert r.lvcf([1, "1", 1.0, "1"], [None, None, 1, 1]) == [0, 0, 1, 1]


@pytest.mark.parametrize("storage", ["r_factor", "pandas_categorical", "pandas_series"])
@pytest.mark.parametrize("observed", [1, True])
def test_factor_values_keep_unknown_initial_levels(storage, observed):
    values = [None, observed, None]
    if storage == "r_factor":
        values = _r_factor(values, [0, 1] if observed is not True else [False, True])
    else:
        pd = pytest.importorskip("pandas")
        values = pd.Categorical(values)
        if storage == "pandas_series":
            values = pd.Series(values)
    actual = r.lvcf([1, 1, 1], values)
    assert actual[0] is None or math.isnan(actual[0])
    assert actual[1:] == [observed, observed]


def test_empty_data_remains_empty():
    assert r.lvcf([], [], time=[]) == []
