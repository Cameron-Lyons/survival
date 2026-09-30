"""Direct person-years preparation shares formula validation and native tabulation."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import
from .r_fixture_support import RFactor

survival = setup_survival_import()
r = survival.r_api
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/pyears_matrix_reference.json").read_text()
)
CASES = [case for case in REFERENCE["cases"] if "tcut" not in case["formula"]]


def _table():
    return r.RateTable([3], ["age"], [["0", "5", "10"]], [[0, 5, 10]], [2], [0.01, 0.02, 0.03])


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
@pytest.mark.parametrize("layout", ["list", "array"])
def test_direct_matrices_match_r_reference_tables_and_retained_response(case, layout):
    data = case["data"]
    lhs = case["formula"].split("~")[0].strip()
    response = np.column_stack([data[name.strip()] for name in lhs[6:-1].split(",")]).astype(float)
    if layout == "list":
        response = response.tolist()
    output = r.pyears(
        response,
        data,
        ratetable=_table() if case["ratetable"] else None,
        expect=case["expect"],
        weights="weight",
        subset=case["subset"],
        group="group" if case["formula"].endswith("~ group") else None,
        na_action="exclude",
        scale=2,
        x=True,
        y=True,
    )
    for name in ("pyears", "n", "event", "expected"):
        actual, expected = getattr(output, name), case[name]
        if expected is None:
            assert actual is None
        else:
            np.testing.assert_allclose(np.asarray(actual).ravel(order="F"), expected, rtol=1e-12)
    assert output.offtable == pytest.approx(case["offtable"])
    assert output.observations == case["observations"]
    assert output.dim == case["dim"]
    assert list(output.dimnames.values()) == list((case["dimnames"] or {}).values())
    assert (list(output.na_action.rows) if output.na_action else []) == case["na_action"]
    np.testing.assert_allclose(output.y, case["y"])
    np.testing.assert_allclose(output.x, case["x"])


@pytest.mark.parametrize("expect", ["event", "pyears"])
@pytest.mark.parametrize("kind", ["time", "right", "counting", "entry"])
def test_direct_vectors_match_formula_rate_predictions(kind, expect):
    d = {
        "time": [2, 6, 9, 3],
        "start": [0, 1, 3, 1],
        "event": [1, 0, 1, 1],
        "age": [1, 3, 2, 6],
        "group": ["b", "a", "b", "a"],
        "weight": [1, 2, 0.5, 1],
    }
    if kind == "time":
        args = {"time": "time"}
        lhs = "time"
    elif kind == "right":
        args = {"formula": r.Surv(d["time"], d["event"])}
        lhs = "Surv(time, event)"
    elif kind == "counting":
        args = {"formula": r.Surv(d["start"], d["time"], d["event"])}
        lhs = "Surv(start, time, event)"
    else:
        args = {"start": "start", "stop": "time"}
        lhs = "cbind(start, time)"
    options = {
        "data": d,
        "ratetable": _table(),
        "expect": expect,
        "scale": 1,
        "weights": "weight",
        "subset": [3, 1, 0, 3],
    }
    actual = r.pyears(**args, **options, group="group")
    reference = r.pyears(f"{lhs} ~ group", **options)
    for field in ("pyears", "n", "event", "expected", "offtable"):
        if getattr(reference, field) is None:
            assert getattr(actual, field) is None
        else:
            np.testing.assert_allclose(
                getattr(actual, field), getattr(reference, field), rtol=1e-12
            )


def test_rate_positions_advance_from_entry_and_expected_pyears_are_honored():
    event = r.pyears(start=[2], stop=[6], rmap={"age": [2]}, ratetable=_table(), scale=1)
    # Age 4 at entry: one time unit at .01 and three at .02.
    assert event.expected == pytest.approx(0.07)
    py = r.pyears(
        start=[2], stop=[6], rmap={"age": [2]}, ratetable=_table(), scale=1, expect="pyears", y=True
    )
    expected = -math.expm1(-0.01) / 0.01 + math.exp(-0.01) * -math.expm1(-0.06) / 0.02
    assert py.expected == pytest.approx(expected)
    assert py.event is None
    assert py.y == [[2, 6]]


def test_prepared_columns_do_not_shadow_mapping_or_weight_sources():
    data = {
        "unused": [0],
        "time": [1, 2],
        "_time": [8, 9],
        "event": [3, 4],
        "group": [0, 1],
        "age": [1, 2],
    }
    before = {name: list(values) for name, values in data.items()}
    actual = r.pyears(
        [2, 2],
        data,
        event=[1, 0],
        group=["a", "a"],
        weights="time",
        ratetable=_table(),
        rmap={"age": "time + event"},
        scale=1,
        model=True,
    )
    assert actual.pyears == [6]
    assert actual.expected == pytest.approx([0.11])
    assert actual.event == [1]
    assert data == before
    assert actual.model["time"] == [1, 2]
    assert actual.model["event"] == [3, 4]
    assert actual.model["(weights)"] == [1, 2]
    assert "Surv(__time, _event)" in actual.model


def test_direct_factors_and_time_cuts_preserve_group_metadata():
    group = RFactor(["b", "a", "b"], ["b", "a", "empty"])
    out = r.pyears([2, 3, 4], group=group, scale=1, model=True)
    assert out.dimnames == {"group": ["b", "a", "empty"]}
    assert out.pyears == [6, 3, 0]
    assert list(out.model["group"].categories) == ["b", "a", "empty"]
    cut = r.tcut([1, 4], [0, 5, 10], labels=["early", "late"])
    out = r.pyears([6, 2], group=cut, scale=1, model=True)
    assert out.pyears == [5, 3]
    assert out.tcut
    assert out.model["group"].labels == ["early", "late"]


def test_missing_direct_followup_and_mapping_keep_subsets_aligned():
    out = r.pyears(
        time=[2, None, 4, 5],
        group=["a", "b", "a", "b"],
        rmap={"age": [1, 2, None, 3]},
        ratetable=_table(),
        weights=[1, 2, 3, 4],
        subset=[3, 1, 0, 2, 3],
        na_action="exclude",
        scale=1,
        y=True,
    )
    assert out.na_action.rows == (2, 4)
    assert out.pyears == [2, 40]
    assert out.y == [[5], [2], [5]]
    with pytest.raises(ValueError, match="missing"):
        r.pyears(time=[2, None], na_action="fail")
    with pytest.raises(ValueError, match="finite"):
        r.pyears(time=[2, None], na_action="pass")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rmap": {"age": 2}}, "No rate table"),
        ({"ratetable": {}}, "Invalid rate table"),
        ({"ratetable": _table(), "rmap": {"bogus": 1}}, "Variable not found"),
        ({"time": [1]}, "not more than one"),
        ({"group": [1, 2]}, "same length"),
    ],
)
def test_direct_calls_validate_every_supplied_option(kwargs, message):
    with pytest.raises(ValueError, match=message):
        r.pyears([2], **kwargs)


def test_ambiguous_response_sources_are_rejected():
    with pytest.raises(ValueError, match="only one"):
        r.pyears(time=[2], stop=[3])
    with pytest.raises(ValueError, match="already supplies"):
        r.pyears(r.Surv([2], [1]), event=[0])
    with pytest.raises(ValueError, match="cannot be combined"):
        r.pyears([[2, 1]], event=[0])


@pytest.mark.parametrize("argument", ["time", "start", "stop", "event", "group"])
def test_formula_calls_reject_unused_direct_arguments(argument):
    with pytest.raises(ValueError, match="cannot be combined"):
        r.pyears("time ~ 1", {"time": [2]}, **{argument: [1]})


def test_direct_dataframe_and_column_names_support_rate_data():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {
            "follow up": [2, 3],
            "age": [1, 2],
            "weight": [1, 2],
            "category": pd.Categorical(["b", "a"], categories=["b", "a", "c"]),
        }
    )
    out = r.pyears(
        data=frame,
        time="follow up",
        group="category",
        weights="weight",
        ratetable=_table(),
        scale=1,
        data_frame=True,
    )
    assert out.data == {
        "group": ["b", "a"],
        "pyears": [2, 6],
        "n": [1, 1],
        "expected": pytest.approx([0.02, 0.06]),
    }
