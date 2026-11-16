"""Vectorised paths of the population front end.

Date rate variables skip the per-row ``Timestamp`` work (the ``NaT`` scan of the
``na.action`` and ``ratetableDate``) and the match summary reads whole columns.
Every result must be bit for bit the one of the per-element path each replaces,
which the tests run by switching the fast paths off.
"""

from __future__ import annotations

import importlib
import math
import struct
from datetime import date

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
r_coerce = importlib.import_module("survival.r._coerce")
r_pyears = importlib.import_module("survival.r._pyears")

pd = pytest.importorskip("pandas")


def _bits(value):
    """``value`` with every float replaced by its IEEE bytes, so NaN and -0 compare."""

    if isinstance(value, float):
        return struct.pack("<d", value)
    if isinstance(value, dict):
        return {key: _bits(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_bits(item) for item in value]
    return value


def _slow_paths(monkeypatch):
    monkeypatch.setattr(r_coerce, "_datetime_missing_mask", lambda values: None)
    monkeypatch.setattr(r_pyears, "_timestamp_days", lambda values: None)


def _dates(n, seed=3, tz=None):
    rng = np.random.default_rng(seed)
    ns = rng.integers(-25 * 365 * 86_400 * 10**9, 40 * 365 * 86_400 * 10**9, n)
    dates = pd.Series(pd.to_datetime(ns))
    if tz is not None:
        dates = dates.dt.tz_localize("UTC").dt.tz_convert(tz)
    return dates


@pytest.mark.parametrize("tz", [None, "America/Chicago", "Asia/Tokyo"])
@pytest.mark.parametrize("unit", ["ns", "us", "s"])
def test_timestamp_days_match_ratetable_day(tz, unit):
    dates = _dates(500, tz=tz).dt.as_unit(unit)
    dates.iloc[[3, 17]] = pd.NaT
    column = dates.tolist()
    days = r_pyears._timestamp_days(column)
    assert days is not None
    assert _bits(days) == _bits([r_pyears._ratetable_day(value) for value in column])


def test_timestamp_days_leave_other_columns_to_the_per_value_path():
    early = pd.Timestamp("1500-01-01").as_unit("s")
    late = pd.Timestamp("2400-06-30 13:00").as_unit("s")
    assert r_pyears._timestamp_days([early, pd.NaT, late]) is None
    assert r_pyears._timestamp_days([pd.Timestamp("2000-01-01"), date(2001, 1, 1)]) is None
    assert r_pyears._timestamp_days([date(2001, 1, 1)]) is None
    assert r_pyears._timestamp_days([]) is None
    table = r.survexp_us()
    matched = r.match_ratetable(
        {"age": [20000.0, 21000.0], "sex": [1, 2], "year": [early, late]}, table
    )
    assert matched.r[0][2] == r_pyears._ratetable_day(early)
    assert matched.r[1][2] == r_pyears._ratetable_day(late)


@pytest.mark.parametrize(
    "values",
    [
        lambda: _dates(40),
        lambda: _dates(40, tz="America/Chicago"),
        lambda: _dates(40).dt.as_unit("s"),
        lambda: pd.Series(pd.to_timedelta(np.arange(40), unit="D")),
        lambda: pd.DatetimeIndex(_dates(40)),
        lambda: _dates(40).to_numpy().copy(),
        lambda: _dates(40).to_numpy().astype("datetime64[D]"),
        lambda: np.arange(40).astype("timedelta64[h]"),
    ],
)
def test_nat_scan_matches_the_per_element_scan(monkeypatch, values):
    column = values()
    if isinstance(column, np.ndarray):
        nat = np.datetime64("NaT") if column.dtype.kind == "M" else np.timedelta64("NaT")
        column[[2, 31]] = nat
    elif isinstance(column, pd.Index):
        column = column.where(~np.isin(np.arange(40), [2, 31]))
    else:
        column = column.copy()
        column.iloc[[2, 31]] = pd.NaT
    fast = r_coerce._missing_row_indices([("year", column)], 40)
    with pytest.raises(ValueError, match="year must have length 41"):
        r_coerce._missing_row_indices([("year", column)], 41)
    _slow_paths(monkeypatch)
    assert fast == r_coerce._missing_row_indices([("year", column)], 40) == {2, 31}


def _population_data(n, tz=None):
    rng = np.random.default_rng(11)
    year = _dates(n, seed=5, tz=tz)
    year.iloc[[4, 9]] = pd.NaT
    return pd.DataFrame(
        {
            "ftime": np.round(rng.exponential(900.0, n)),
            "status": rng.integers(0, 2, n),
            "g": pd.Categorical(np.array(["a", "b", "c"])[rng.integers(0, 3, n)]),
            "age": rng.uniform(30, 80, n) * 365.25,
            "sex": rng.integers(1, 3, n),
            "race": rng.integers(1, 3, n),
            "year": year,
            "w": rng.uniform(0.5, 2.0, n),
        }
    )


_RMAP = {"age": "age", "sex": "sex", "year": "year"}
_POPULATION_CALLS = {
    "ederer": lambda d, table: r.survexp(
        "~ g", d, ratetable=table, rmap=_RMAP, times=[0, 365, 730, 1825]
    ),
    "hakulinen": lambda d, table: r.survexp(
        "Surv(ftime, status) ~ g", d, ratetable=table, rmap=_RMAP, method="hakulinen"
    ),
    "conditional": lambda d, table: r.survexp(
        "Surv(ftime, status) ~ 1", d, ratetable=table, rmap=_RMAP, method="conditional"
    ),
    "individual": lambda d, table: r.survexp(
        "ftime ~ 1", d, ratetable=table, rmap=_RMAP, cohort=False, na_action="na.exclude"
    ),
    "pyears": lambda d, table: r.pyears(
        "Surv(ftime, status) ~ g", d, ratetable=table, rmap=_RMAP, weights=d["w"]
    ),
}


def _population_key(result):
    fields = (
        "time",
        "surv",
        "n_risk",
        "method",
        "n",
        "strata",
        "pyears",
        "event",
        "expected",
        "offtable",
        "observations",
        "summary",
    )
    return _bits({name: getattr(result, name, None) for name in fields})


@pytest.mark.parametrize("tz", [None, "America/Chicago"])
@pytest.mark.parametrize("table", ["us", "usr"])
@pytest.mark.parametrize("call", sorted(_POPULATION_CALLS))
def test_population_fits_with_date_columns_match_the_per_element_path(monkeypatch, tz, table, call):
    data = _population_data(300, tz=tz)
    rate_table = r.survexp_us() if table == "us" else r.survexp_usr()
    if table == "usr":
        monkeypatch.setitem(_RMAP, "race", "race")
    fast = _POPULATION_CALLS[call](data, rate_table)
    _slow_paths(monkeypatch)
    slow = _POPULATION_CALLS[call](data, rate_table)
    assert _population_key(fast) == _population_key(slow)


def _summary_reference(table, positions):
    """``_population_match_summary`` before it read whole columns: one row at a time."""

    source = table.source
    if source not in {"survexp.us", "survexp.usr", "survexp.mn"}:
        return None
    age_low = year_low = math.inf
    age_high = year_high = -math.inf
    male = female = white = black = 0
    for row in positions:
        age_low, age_high = min(age_low, row[0]), max(age_high, row[0])
        year_low, year_high = min(year_low, row[-1]), max(year_high, row[-1])
        male += row[1] == 1
        female += row[1] == 2
        if source == "survexp.usr":
            white += row[2] == 1
            black += row[2] == 2
    if positions:
        dates = [survival._survival.days_to_date(math.floor(v)) for v in (year_low, year_high)]
        first, last = [f"{day.year:04d}-{day.month:02d}-{day.day:02d}" for day in dates]
    else:
        first, last = "Inf", "-Inf"
    low, high = [
        r_coerce._r_format_number(round(value / 365.25, 1), 7) for value in (age_low, age_high)
    ]
    indent = "  " if source == "survexp.mn" else "    "
    text = (
        f" age ranges from {low} to {high} years\n"
        f"{indent}male: {male}  female: {female} \n"
        f"{indent}date of entry from {first} to {last} \n"
    )
    if source == "survexp.usr":
        text += f"    white: {white}  black: {black} \n"
    return text


@pytest.mark.parametrize("table", ["us", "usr", "mn"])
def test_match_summary_reads_columns_as_the_row_loop(table):
    rate_table = {"us": r.survexp_us, "usr": r.survexp_usr, "mn": r.survexp_mn}[table]()
    width = len(rate_table.dimid)
    rng = np.random.default_rng(2)
    rows = [
        [float(rng.uniform(0, 30000)), float(rng.integers(1, 3)), *[1.0] * (width - 3), 7000.0]
        for _ in range(50)
    ]
    if width == 4:
        for row in rows:
            row[2] = float(rng.integers(1, 3))
    signed_zero = [[0.0, 1.0, *[2.0] * (width - 3), 0.0], [-0.0, 2.0, *[1.0] * (width - 3), -0.0]]
    for positions in (rows, signed_zero, signed_zero[::-1], []):
        expected = _summary_reference(rate_table, positions)
        assert r_pyears._population_match_summary(rate_table, positions) == expected
