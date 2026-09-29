"""Surv row operations and formatting against R survival 3.8-12."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
REFERENCE = json.loads((Path(__file__).parent / "fixtures/surv_vector_reference.json").read_text())
METHODS = {"c": "concat_surv", "t": "transpose_surv", "as.character": "as_character_surv"}


def response(snapshot):
    rows = snapshot["matrix"]
    kind = snapshot["type"]

    def numeric(col):
        return [math.nan if row[col] is None else row[col] for row in rows]

    return r.Surv._from_normalized(
        time=numeric(1 if "counting" in kind else 0),
        event=[row[-1] for row in rows],
        start=numeric(0) if "counting" in kind else None,
        time2=numeric(1) if kind == "interval" else None,
        surv_type=kind,
        states=snapshot["states"] or (),
        clabel=snapshot["clabel"],
    )


def assert_response(actual, expected):
    assert isinstance(actual, r.Surv)
    assert actual.type == expected["type"]
    assert list(actual.states) == (expected["states"] or [])
    assert actual.clabel == expected["clabel"]
    np.testing.assert_equal(
        np.asarray(actual.as_matrix(), dtype=float), np.asarray(expected["matrix"], dtype=float)
    )


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize(
    "operation", REFERENCE["operations"], ids=lambda operation: operation["name"]
)
def test_surv_vector_operations_match_r(case, operation):
    x = response(case["response"])
    method = METHODS.get(operation["method"], operation["method"] + "_surv")
    function = getattr(r, method)
    assert getattr(survival, method) is function
    actual = function(x, **dict(operation["args"]))
    expected = case["results"][operation["name"]]
    if isinstance(expected, dict):
        assert_response(actual, expected)
    elif method == "transpose_surv":
        np.testing.assert_equal(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float))
    else:
        assert actual == expected


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_concatenation_preserves_type_states_and_normalized_codes(case):
    x = response(case["response"])
    result = r.concat_surv(x.subset([2, 0]), x.subset([6, 1]))
    assert_response(result, case["concat"])
    # A status code of 2 in a multistate or interval response must not be
    # reinterpreted through the constructor's ordinary 1/2 event coding.
    assert result.event == (x.event[2], x.event[0], x.event[6], x.event[1])


def test_concatenation_rejects_incompatible_objects_and_states():
    x = r.Surv([1, 2], [1, 0])
    with pytest.raises(ValueError, match="at least one"):
        r.concat_surv()
    with pytest.raises(TypeError, match="class Surv"):
        r.concat_surv(x, [1, 2])
    with pytest.raises(ValueError, match="same Surv type"):
        r.concat_surv(x, r.Surv([1, 2], [1, 0], type="left"))
    left = response(REFERENCE["cases"][4]["response"])
    right = r.Surv._from_normalized(
        time=[1], event=[1], start=None, time2=None, surv_type="mright", states=["different"]
    )
    with pytest.raises(ValueError, match="same list of states"):
        r.concat_surv(left, right)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_empty_responses_keep_column_shape_and_metadata(case):
    x = response(case["response"]).subset([])
    for function in (
        r.rep_surv,
        r.rev_surv,
        r.unique_surv,
        r.head_surv,
        r.tail_surv,
        r.concat_surv,
    ):
        result = function(x)
        assert len(result) == 0
        assert result.type == x.type
        assert result.states == x.states
    assert r.transpose_surv(x) == [[] for _ in range(x.ncol)]
    assert r.duplicated_surv(x) == []
    assert r.as_character_surv(x) == []
    assert r.format_surv(x) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"times": -1},
        {"times": [1, 2, 3]},
        {"times": math.inf},
        {"each": -1},
        {"each": 0, "length_out": 2},
        {"length_out": -1},
    ],
)
def test_repetition_rejects_invalid_counts(kwargs):
    with pytest.raises(ValueError, match="invalid"):
        r.rep_surv(r.Surv([1, 2], [1, 0]), **kwargs)


def test_character_conversion_preserves_leading_padding_only():
    x = r.Surv([1, 20], [1, 0])
    assert r.as_character_surv(x) == [" 1", "20+"]
    assert r.format_surv(x) == [" 1 ", "20+"]
    x2 = r.Surv2([1, 20], [1, 0])
    assert r.as_character_surv(x2) == [" 1", "20+"]


def test_missing_rows_deduplicate_by_column_without_merging_statuses():
    x = r.Surv([math.nan, math.nan, math.nan, 0, -0.0], [0, 1, 0, 1, 1])
    assert r.duplicated_surv(x) == [False, False, True, False, True]
    assert r.duplicated_surv(x, from_last=True) == [True, False, False, True, False]
