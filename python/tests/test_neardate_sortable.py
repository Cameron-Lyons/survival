"""Sortable neardate dates against stock R 4.5.3 / survival 3.8-12.

The character/date reference rows come from survival::neardate with nomatch=-1;
successful rows are converted from R's one-based indices to Python positions.
"""

from datetime import UTC, date, datetime, timedelta, timezone

import numpy as np
import pytest

from .helpers import setup_survival_import

r = setup_survival_import().r_api


@pytest.mark.parametrize(
    ("best", "expected"),
    [
        ("after", [1, 2, -1, -1, 4, 2]),
        ("prior", [1, 3, 1, -1, 3, 3]),
    ],
)
@pytest.mark.parametrize("array", [False, True])
def test_character_dates_retain_lexical_order_and_tie_direction(best, expected, array):
    query = ["2", "10", "3", None, "11", "10"]
    reference = ["1", "2", "10", "10", "12", None]
    if array:
        query, reference = np.array(query, dtype=object), np.array(reference, dtype=object)
    assert r.neardate([1] * 6, [1] * 6, query, reference, best=best, nomatch=-1) == expected


@pytest.mark.parametrize(
    ("best", "expected"),
    [("after", [2, 2, 1, -1, -1]), ("prior", [0, 0, 3, -1, -1])],
)
def test_iso_character_dates_preserve_groups_and_missing_matches(best, expected):
    assert (
        r.neardate(
            ["a", "a", "b", "b", "c"],
            ["a", "b", "a", "b", "d"],
            ["2020-01-02", "2020-01-10", "2020-01-03", None, "2020-01-01"],
            ["2020-01-01", "2020-01-03", "2020-01-12", "2020-01-03", "2020-01-01"],
            best=best,
            nomatch=-1,
        )
        == expected
    )


@pytest.mark.parametrize(("best", "expected"), [("after", [1, 2, -1]), ("prior", [1, 2, 1])])
def test_numeric_and_character_dates_use_r_common_character_type(best, expected):
    assert (
        r.neardate([1] * 3, [1] * 3, [2, 10, 3], ["1", "2", "10"], best=best, nomatch=-1)
        == expected
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.longdouble])
@pytest.mark.parametrize("best", ["after", "prior"])
def test_numpy_floating_scalars_use_r_numeric_labels_with_character_dates(dtype, best):
    assert r.neardate([1, 1], [1, 1], [dtype(1), dtype(2.5)], ["1", "2.5"], best=best) == [0, 1]


@pytest.mark.parametrize(
    ("best", "expected"),
    [("after", [None, -1, -1]), ("prior", [None, -1, 1])],
)
def test_missing_identifiers_remain_missing_independently_of_nomatch(best, expected):
    assert (
        r.neardate(
            [None, 1, 1], [None, 1, 1], ["a", None, "z"], ["a", "b", None], best=best, nomatch=-1
        )
        == expected
    )


@pytest.mark.parametrize("best", ["after", "prior"])
@pytest.mark.parametrize("kind", ["date", "datetime", "numpy-D", "numpy-ns", "pandas"])
def test_calendar_date_containers_agree_with_stock_date_references(best, kind):
    query = ["2020-01-02", "2020-01-10", "2020-01-03", None, "1960-01-01", "2020-01-03"]
    reference = ["2020-01-01", "2020-01-03", "2020-01-12", "2020-01-03", None, "1970-01-01"]
    if kind in {"date", "datetime"}:
        convert = date.fromisoformat if kind == "date" else datetime.fromisoformat
        query, reference = (
            [None if value is None else convert(value) for value in vector]
            for vector in (query, reference)
        )
    elif kind.startswith("numpy"):
        unit = kind.split("-")[1]
        query, reference = (
            np.array(vector, dtype=f"datetime64[{unit}]") for vector in (query, reference)
        )
    else:
        pd = pytest.importorskip("pandas")
        query, reference = (pd.Series(pd.to_datetime(vector)) for vector in (query, reference))
    expected = [1, 2, 1, -1, 5, 1] if best == "after" else [0, 3, 3, -1, -1, 3]
    assert r.neardate([1] * 6, [1] * 6, query, reference, best=best, nomatch=-1) == expected


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns", "ps", "fs", "as"])
def test_numpy_timestamp_matching_retains_exact_resolution(unit):
    # A float conversion can merge adjacent large integer timestamps. Ranking
    # the union preserves them before the native matcher reads double ranks.
    query = np.array([2**53 + 1, 2**53 + 3], dtype=np.int64).view(f"datetime64[{unit}]")
    reference = np.array([2**53, 2**53 + 2], dtype=np.int64).view(f"datetime64[{unit}]")
    assert r.neardate([1, 1], [1, 1], query, reference, nomatch=-1) == [1, -1]
    assert r.neardate([1, 1], [1, 1], query, reference, best="prior") == [0, 1]


def test_mixed_numpy_date_units_do_not_overflow_distant_calendar_dates():
    query = [np.datetime64("1500-01-01", "D"), np.datetime64("2020-01-01", "ns")]
    reference = [np.datetime64("1600-01-01", "D"), np.datetime64("2020-01-02", "ns")]
    assert r.neardate([1, 1], [1, 1], query, reference) == [0, 1]


@pytest.mark.parametrize("unit", ["Y", "M", "W", "2D"])
def test_numpy_coarse_calendar_units_keep_their_epoch_alignment(unit):
    query = np.array(["2020-01-01", "2021-02-01"], dtype=f"datetime64[{unit}]")
    reference = query.astype("datetime64[D]")
    assert r.neardate([1, 1], [1, 1], query, reference) == [0, 1]
    assert r.neardate([1, 1], [1, 1], query, reference, best="prior") == [0, 1]


def test_python_calendar_and_aware_datetimes_match_absolute_instants():
    offset = timezone(timedelta(hours=-6))
    query = [date(2020, 1, 1), datetime(2019, 12, 31, 18, tzinfo=offset)]
    reference = [np.datetime64("2020-01-01", "D"), datetime(2020, 1, 1, tzinfo=UTC)]
    assert r.neardate([1, 1], [1, 1], query, reference) == [0, 0]
    assert r.neardate([1, 1], [1, 1], query, reference, best="prior") == [1, 1]


def test_masked_numpy_datetimes_do_not_lose_type_or_missingness():
    query = np.ma.array(
        np.array(["2020-01-01", "2020-01-01"], dtype="datetime64[ns]"), mask=[False, True]
    )
    reference = np.ma.array(
        np.array(["2020-01-01", "2020-01-01"], dtype="datetime64[ns]"), mask=[True, False]
    )
    assert r.neardate([1, 1], [1, 1], query, reference, nomatch=-1) == [1, -1]


@pytest.mark.parametrize("which", [0, 1])
def test_factor_dates_raise_r_sortability_error(which):
    pd = pytest.importorskip("pandas")
    dates = [[1, 2], [1, 2]]
    dates[which] = pd.Categorical([1, 2])
    with pytest.raises(ValueError, match="y1 and y2 must be sortable"):
        r.neardate([1, 1], [1, 1], *dates)


@pytest.mark.parametrize("which", [0, 1])
def test_all_missing_character_reference_dates_raise_r_error(which):
    query = [None] if which else ["a"]
    with pytest.raises(ValueError, match="No valid entries in data set 2"):
        r.neardate([1], [1], query, [None])
