"""Last-value carry-forward uses one subject/time order for initialization and copying."""

import importlib
import math
import sys
from datetime import date, datetime

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
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
    assert r.lvcf([1, "1", 1.0, "1"], [None, None, 1, 1]) == [0, 0, 1, 1]


@pytest.mark.parametrize("base", [2**63, -(2**100)])
def test_wide_subject_ids_keep_separate_histories(base):
    ids = [base, base + 1, float(base), base + 1, str(base)]
    assert r.lvcf(ids, [10, None, None, 20, None]) == [10, None, 10, 20, None]


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


@pytest.mark.parametrize(
    ("times", "expected"),
    [
        (["10", "2", None], [10, 10, 20]),
        (_r_factor(["b", "a", "c"], ["c", "b", "a"]), [10, 10, 20]),
        ([date(2020, 1, 2), date(2020, 1, 1), None], [10, None, 20]),
        ([0.0, -0.0, 0.0], [10, 10, 20]),
        ([2**53 + 1, 2**53, 2**53 + 2], [10, None, 20]),
        (np.array([2**53 + 1, 2**53, 2**53 + 2]).view("datetime64[ns]"), [10, None, 20]),
    ],
)
def test_visit_order_uses_exact_times_and_factor_levels(times, expected):
    assert r.lvcf([1] * 3, [10, None, 20], time=times) == expected


def test_typed_missing_columns_keep_their_initial_value_type():
    pd = pytest.importorskip("pandas")
    for dtype, expected in [("Int64", 0), ("Float64", 0), ("boolean", False), ("string", pd.NA)]:
        result = r.lvcf([1, 1], pd.Series([pd.NA, pd.NA], dtype=dtype))
        for value in result:
            assert type(value) is type(expected)
            assert value is pd.NA or value == expected


@pytest.mark.parametrize("library", ["pandas", "polars"])
def test_nullable_integer_visit_times_keep_their_exact_order(library):
    module = pytest.importorskip(library)
    times = [2**53 + 1, 2**53, None]
    column = module.Series(times, dtype="Int64" if library == "pandas" else module.UInt64)
    assert r.lvcf([1] * 3, [10, None, 20], column) == [10, None, 20]


def test_direct_nullable_array_preserves_integer_visits_and_measurements():
    pd = pytest.importorskip("pandas")
    base = 2**53
    column = pd.array([base + 1, base, pd.NA], dtype="Int64")
    assert r.lvcf([1] * 3, [10, None, 20], time=column) == [10, None, 20]
    assert r.lvcf([1] * 3, column) == [base + 1, base, base]


@pytest.mark.parametrize("dtype", ["datetime64[ns]", "timedelta64[ns]"])
def test_direct_pandas_temporal_array_keeps_nanosecond_scalars(dtype):
    pd = pytest.importorskip("pandas")
    column = pd.array([None, 1, None], dtype=dtype)
    initial, observed, carried = r.lvcf([1] * 3, column)
    assert initial is pd.NaT
    assert type(observed) is type(column[1])
    assert observed == carried == column[1]


def test_calendar_measurements_keep_their_values_and_unknown_initial_visit():
    values = np.array(["NaT", "2020-01-01T00:00:00.000000001", "NaT"], dtype="datetime64[ns]")
    initial, observed, carried = r.lvcf([1] * 3, values)
    assert np.isnat(initial)
    assert isinstance(carried, np.datetime64)
    assert observed == carried == values[1]


@pytest.mark.parametrize("missing", [np.datetime64("NaT"), np.timedelta64("NaT")])
def test_untyped_calendar_measurements_keep_missing_values(missing):
    assert all(np.isnat(value) for value in r.lvcf([1, 1], [missing, missing]))


@pytest.mark.parametrize("kind", ["Datetime", "Duration"])
def test_polars_temporal_measurements_retain_nanoseconds(kind):
    pl = pytest.importorskip("polars")
    column = pl.Series([None, 1, None], dtype=getattr(pl, kind)("ns"))
    initial, observed, carried = r.lvcf([1] * 3, column)
    assert np.isnat(initial)
    assert observed == carried == column.to_numpy()[1]


def test_polars_microsecond_aligned_zoned_measurements_keep_python_datetimes():
    pl = pytest.importorskip("polars")
    column = pl.Series([None, 1000, None], dtype=pl.Datetime("ns", "America/Chicago"))
    initial, observed, carried = r.lvcf([1] * 3, column)
    assert initial is None
    assert type(observed) is datetime
    assert observed == carried == column.to_list()[1]
    assert str(observed.tzinfo) == "America/Chicago"


def test_polars_zoned_measurements_keep_exact_instants_and_zone():
    pl = pytest.importorskip("polars")
    pd = pytest.importorskip("pandas")
    column = pl.Series([None, -1, None], dtype=pl.Datetime("ns", "America/Chicago"))
    initial, observed, carried = r.lvcf([1] * 3, column)
    assert initial is None
    assert isinstance(observed, pd.Timestamp)
    assert observed.value == -1
    assert str(observed.tzinfo) == "America/Chicago"
    assert carried == observed


def test_polars_zoned_measurements_without_pandas_keep_exact_utc_instants(monkeypatch):
    pl = pytest.importorskip("polars")
    column = pl.Series([None, 1, None], dtype=pl.Datetime("ns", "America/Chicago"))
    monkeypatch.setitem(sys.modules, "pandas", None)
    initial, observed, carried = r.lvcf([1] * 3, column)
    assert initial is None
    assert isinstance(observed, np.datetime64)
    assert observed == carried == np.datetime64(1, "ns")


def test_polars_zoned_measurement_at_local_timestamp_limit_keeps_utc_instant():
    pl = pytest.importorskip("polars")
    ticks = 2**63 - 1
    column = pl.Series([ticks, None], dtype=pl.Datetime("ns", "Asia/Tokyo"))
    observed, carried = r.lvcf([1, 1], column)
    assert isinstance(observed, np.datetime64)
    assert observed == carried == np.datetime64(ticks, "ns")
