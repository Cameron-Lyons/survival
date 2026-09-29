"""Public population matching, including unused factors and native ambiguity checks."""

import datetime
import json
import math
import pickle
import re
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "rate_matching_reference.json").read_text()
)


def decode(column):
    from survival.r._coerce import _RFactorVector

    kind, values = column["kind"], column["values"]
    if kind == "factor":
        return _RFactorVector(values, column["levels"])
    conversions = {
        "date": datetime.date.fromisoformat,
        "datetime": datetime.datetime.fromisoformat,
        "timedelta": lambda value: datetime.timedelta(days=value),
    }
    return [conversions[kind](value) for value in values] if kind in conversions else values


@pytest.fixture(scope="module")
def tables():
    return {
        name: getattr(r, spec["builtin"].replace(".", "_"))()
        if "builtin" in spec
        else r.RateTable(
            spec["dims"],
            spec["dimid"],
            spec["dimnames"],
            spec["cutpoints"],
            spec["types"],
            spec["rates"],
        )
        for name, spec in REFERENCE["tables"].items()
    }


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_match_ratetable_against_r(case, tables):
    data = {name: decode(column) for name, column in case["data"].items()}
    expected = case["expected"]
    if "error" in expected:
        with pytest.raises(ValueError, match=re.escape(expected["error"])):
            r.match_ratetable(data, tables[case["table"]])
        return
    result = r.match_ratetable(data, tables[case["table"]])
    assert result.r == expected["r"]
    assert result.cutpoints == expected["cutpoints"]
    assert result.summary == expected["summary"]
    frame = r.as_data_frame(result)
    assert list(frame) == tables[case["table"]].dimid
    assert list(zip(*frame.values(), strict=True)) == [tuple(row) for row in result.r]


def test_full_result_ownership_and_numpy_dates(tables):
    table = tables["survexp.us"]
    data = {
        "age": [18000.125, 22000.375],
        "sex": ["m", "f"],
        "year": np.array(["1996-01-01", "1997-01-01"], dtype="datetime64[D]"),
    }
    result = r.match_ratetable(data, table)
    assert result.r == [[18000.125, 1.0, 9496.0], [22000.375, 2.0, 9862.0]]
    assert pickle.loads(pickle.dumps(result)) == result  # noqa: S301 - own test data
    frame = r.as_data_frame(result)
    frame["age"][0] = 0
    result.cutpoints[0][0] = -999
    data["age"][0] = -5
    assert result.r[0][0] == 18000.125
    assert table.cutpoints[0][0] == 0


def test_pandas_declared_levels_and_duplicate_columns(tables):
    pd = pytest.importorskip("pandas")
    data = pd.DataFrame(
        {
            "age": [18000],
            "sex": pd.Categorical(["male"], categories=["male", "other"]),
            "year": [9500],
        }
    )
    with pytest.raises(ValueError, match="Levels do not match"):
        r.match_ratetable(data, tables["survexp.us"])
    data = pd.DataFrame([[18000, 1, 2, 9500]], columns=["age", "sex", "sex", "year"])
    with pytest.raises(ValueError, match="appears twice"):
        r.match_ratetable(data, tables["survexp.us"])


@pytest.mark.parametrize("function", [r.pyears, r.survexp])
@pytest.mark.parametrize("selection", ["all", "subset", "omit", "constant"])
def test_population_formulas_validate_unused_categories(function, selection):
    from survival.r._coerce import _RFactorVector

    data = {
        "time": [100, 200],
        "age": [18000, 22000],
        "year": [9500, 9500],
        "sex": _RFactorVector(["male", "female"], ["male", "female", "other"]),
    }
    kwargs = {}
    if selection == "subset":
        kwargs["subset"] = [0]
    elif selection == "omit":
        data["time"][1] = None
    elif selection == "constant":
        kwargs["rmap"] = {"sex": _RFactorVector(["male"], ["male", "female", "other"])}
    with pytest.raises(ValueError, match="Levels do not match"):
        function("time ~ 1", data, ratetable=r.survexp_us(), **kwargs)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        ([None], "missing"),
        ([math.inf], "finite"),
        ([1.5], "out of range"),
        ([0], "out of range"),
        (["unknown"], "Levels do not match"),
    ],
)
def test_invalid_positions(values, match, tables):
    with pytest.raises(ValueError, match=match):
        r.match_ratetable({"age": [18000], "sex": values, "year": [9500]}, tables["survexp.us"])


def test_invalid_shape_and_object(tables):
    with pytest.raises(TypeError, match="Invalid rate table"):
        r.match_ratetable({}, None)
    with pytest.raises(TypeError, match="named columns"):
        r.match_ratetable([[18000, 1, 9500]], tables["survexp.us"])
    with pytest.raises(ValueError, match="same length"):
        r.match_ratetable({"age": [18000, 22000], "sex": [1], "year": [9500]}, tables["survexp.us"])


def test_declared_level_native_api(tables):
    table = tables["survexp.us"]
    assert table.match_levels(1, ["f", "m", "female"]) == [2, 1, 2]
    with pytest.raises(ValueError, match="dimension out of range"):
        table.match_levels(3, [])
    with pytest.raises(ValueError, match="Non-unique"):
        tables["duplicate"].match_levels(0, ["male"])
    with pytest.raises(ValueError, match="Non-unique"):
        survival.population.match_ratetable(tables["duplicate"], ["group"], [["male"]])


def test_ratetable_date_uses_utc_for_aware_datetimes():
    value = datetime.datetime(
        1996, 1, 1, 23, 30, tzinfo=datetime.timezone(datetime.timedelta(hours=-6))
    )
    assert r.ratetableDate(value) == 9497.0


def test_many_levels_and_mixed_exact_prefix_cache_entries():
    levels = [f"group{i:02d}.long" for i in range(32)]
    table = r.RateTable([32], ["group"], [levels], [None], [1], [0.01] * 32)
    labels = [
        levels[i % 32].upper() if i % 3 else levels[i % 32].split(".")[0] for i in range(2000)
    ]
    result = survival.population.match_ratetable(table, ["group"], [labels])
    assert result.r == [[float(i % 32 + 1)] for i in range(2000)]
