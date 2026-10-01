"""Numeric matrix responses against R survival 3.8-12 and independent totals."""

import json
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/pyears_matrix_reference.json").read_text()
)


def _table():
    return r.RateTable([3], ["age"], [["0", "5", "10"]], [[0, 5, 10]], [2], [0.01, 0.02, 0.03])


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("matrix_column", [False, True])
def test_numeric_response_matches_complete_r_tables_and_retained_data(case, matrix_column):
    data = {name: list(values) for name, values in case["data"].items()}
    formula = case["formula"]
    if matrix_column:
        lhs, rhs = formula.split("~")
        names = lhs.strip()[6:-1].split(",")
        data["Y"] = np.column_stack([data[name.strip()] for name in names]).astype(float)
        formula = "Y ~" + rhs
    kwargs = {
        "weights": "weight",
        "subset": case["subset"],
        "na_action": "exclude",
        "scale": 2,
        "expect": case["expect"],
    }
    if case["ratetable"]:
        kwargs["ratetable"] = _table()
    output = r.pyears(formula, data, x=True, y=True, **kwargs)
    for name in ("pyears", "n", "event", "expected"):
        actual, expected = getattr(output, name), case[name]
        if expected is None:
            assert actual is None
        else:
            np.testing.assert_allclose(np.asarray(actual).ravel(order="F"), expected, rtol=1e-12)
    assert output.offtable == pytest.approx(case["offtable"])
    assert output.observations == case["observations"]
    assert output.tcut == case["tcut"]
    assert output.dim == case["dim"]
    assert output.dimnames == (case["dimnames"] or {})
    assert (list(output.na_action.rows) if output.na_action else []) == case["na_action"]
    np.testing.assert_allclose(output.y, case["y"])
    np.testing.assert_allclose(output.x, case["x"])
    model = r.pyears(formula, data, model=True, **kwargs).model
    names = list(case["model_names"])
    if matrix_column:
        names[0] = "Y"
    assert list(model) == names
    np.testing.assert_allclose(model[names[0]], case["model_y"])


@pytest.mark.parametrize("layout", ["list", "c", "f", "strided", "readonly", "float32", "object"])
def test_matrix_rows_keep_missingness_alignment_and_ownership(layout):
    matrix = np.array([[2, 1], [5, np.nan], [3, 4], [9, 1]], dtype=float)
    if layout == "list":
        matrix = matrix.tolist()
        matrix[1][1] = None
    elif layout == "f":
        matrix = np.asfortranarray(matrix)
    elif layout == "strided":
        matrix = np.repeat(matrix, 2, axis=1)[:, ::2]
    elif layout == "readonly":
        matrix.flags.writeable = False
    elif layout == "float32":
        matrix = matrix.astype(np.float32)
    elif layout == "object":
        matrix = matrix.astype(object)
        matrix[1, 1] = None
    data = {"Y": matrix, "group": ["b", "a", "b", "a"], "weight": [2, 1, 0.5, 0]}
    out = r.pyears(
        "Y ~ group",
        data,
        subset=[2, 1, 0, 2],
        weights="weight",
        scale=1,
        na_action="exclude",
        y=True,
    )
    assert out.na_action.rows == (2,)
    assert out.pyears == [7]
    assert out.event == [6]
    assert out.y == [[3, 4], [2, 1], [3, 4]]
    before = np.array(matrix, dtype=float)
    out.y[0][0] = -100
    np.testing.assert_equal(np.array(matrix, dtype=float), before)
    frame = r.model_frame("Y ~ group", data, subset=[2, 0])
    assert frame["Y"] == [[3, 4], [2, 1]]
    frame["Y"][0][0] = -100
    np.testing.assert_equal(np.array(matrix, dtype=float), before)


def test_cbind_arithmetic_constants_nested_calls_and_missing_results():
    data = {"time": [0, 2, 6, 8], "denom": [1, 0, 2, 2], "event": [1, 1, 2, 3]}
    formula = "I(cbind(follow = time / denom, events = cbind(event + 1))) ~ 1"
    # An infinite follow-up is present but invalid at the numerical boundary.
    with pytest.raises(ValueError, match="finite"):
        r.pyears(formula, data, scale=1)
    data["time"][1] = 0  # 0/0 is missing, and is omitted with its event value.
    out = r.pyears(formula, data, scale=1, y=True)
    assert out.na_action.rows == (2,)
    assert out.pyears == 7
    assert out.event == 9
    assert out.y == [[0, 2], [3, 3], [4, 4]]
    assert r.pyears("cbind(time, 2) ~ 1", data, scale=1).event == 8
    assert r.pyears("cbind(2, 3) ~ 1", data, scale=1).event == 3


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        ([[1, 2, 3]], "too many columns"),
        ([[1, -1]], "Negative follow up"),
        ([[1], [2, 3]], "must be numeric"),
        ([[1, float("inf")]], "finite"),
        (np.empty((2, 0)), "at least one column"),
        (np.empty((0, 2)), "0 observations"),
    ],
)
def test_numeric_matrix_rejects_invalid_response(matrix, message):
    with pytest.raises(ValueError, match=message):
        r.pyears("Y ~ 1", {"Y": matrix})


def test_missing_matrices_respect_fail_and_pass_without_affecting_survexp():
    data = {"Y": [[1, 1], [2, None]], "age": [1, 2]}
    with pytest.raises(ValueError, match="missing"):
        r.pyears("Y ~ 1", data, na_action="fail")
    with pytest.raises(ValueError, match="finite"):
        r.pyears("Y ~ 1", data, na_action="pass")
    with pytest.raises(ValueError, match="Illegal response value"):
        r.survexp("Y ~ 1", data, ratetable=_table())


def test_matrix_event_counts_are_numeric_totals_not_surv_status_codes():
    out = r.pyears(
        "cbind(time, event) ~ 1",
        {"time": [0, 2, 4], "event": [2, 0.5, 3]},
        weights=[2, 0.5, 1],
        scale=1,
    )
    assert out.pyears == 5
    assert out.event == 7.25
    assert out.n == 2


def test_dot_expansion_excludes_the_whole_matrix_response():
    out = r.pyears("Y ~ .", {"Y": [[2, 1], [3, 2]], "group": ["a", "b"]}, scale=1)
    assert out.term_labels == ["group"]
    assert out.pyears == [2, 3]
    assert out.event == [1, 2]


def test_rate_mapping_can_read_data_with_a_matrix_as_its_first_column():
    data = {"Y": [[0, 2], [1, 4]], "age": [1, 2]}
    out = r.pyears("Y ~ 1", data, ratetable=_table(), scale=1, model=True)
    assert out.pyears == 5
    assert out.expected == pytest.approx(0.06)
    assert out.event is None
    frame = r.model_frame(out)
    assert frame["Y"] == data["Y"]
    frame["Y"][0][0] = -100
    assert out.model["Y"] == data["Y"]


def test_matrix_model_response_can_use_a_reserved_frame_column_name():
    out = r.model_frame("group ~ 1", {"group": [[2, 1], [3, 0]]})
    assert out == {"group": [[2, 1], [3, 0]]}
