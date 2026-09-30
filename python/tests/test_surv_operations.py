"""Survival responses reject R's numeric operation groups and implicit NumPy coercion."""

import copy
import json
import math
import operator
import pickle
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/surv_operations_reference.json").read_text()
)


def response(case):
    rows = case["matrix"]
    if case["class"] == "Surv2":
        times = [row[0] for row in rows]
        status = [row[1] for row in rows]
        if case["states"]:
            levels = [case["clabel"] or "censor", *case["states"]]
            status = survival.r_api._r_factor(
                [None if value is None else levels[int(value)] for value in status], levels
            )
        return r.Surv2(times, status, repeated=case["repeated"])
    kind = case["type"]

    def column(index):
        return [math.nan if row[index] is None else row[index] for row in rows]

    return r.Surv._from_normalized(
        time=column(1 if "counting" in kind else 0),
        event=[row[-1] for row in rows],
        start=column(0) if "counting" in kind else None,
        time2=column(1) if kind == "interval" else None,
        surv_type=kind,
        states=case["states"],
        clabel=case["clabel"],
    )


OPERATIONS = {
    "add": lambda x: x + 1,
    "radd": lambda x: 1 + x,
    "sub": lambda x: x - 1,
    "rsub": lambda x: 1 - x,
    "mul": lambda x: x * 2,
    "rmul": lambda x: 2 * x,
    "div": lambda x: x / 2,
    "rdiv": lambda x: 2 / x,
    "floor_div": lambda x: x // 2,
    "mod": lambda x: x % 2,
    "power": lambda x: x**2,
    "rpower": lambda x: 2**x,
    "eq": lambda x: x == x,
    "ne": lambda x: x != x,
    "lt": lambda x: x < 2,
    "le": lambda x: x <= 2,
    "gt": lambda x: x > 2,
    "ge": lambda x: x >= 2,
    "logical_and": lambda x: x & True,
    "logical_or": lambda x: x | False,
    "invert": operator.invert,
    "negative": operator.neg,
    "positive": operator.pos,
    "abs": abs,
    "sqrt": np.sqrt,
    "log": np.log,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "trunc": math.trunc,
    "cumsum": np.cumsum,
    "cumprod": np.cumprod,
    "sum": np.sum,
    "prod": np.prod,
    "min": np.min,
    "max": np.max,
    "all": np.all,
    "any": np.any,
}


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("operation", REFERENCE["operations"])
def test_invalid_operations_match_r(case, operation):
    with pytest.raises(TypeError, match=f"^{case['errors'][operation]}$"):
        OPERATIONS[operation](response(case))


@pytest.mark.parametrize("cls", [r.Surv, r.Surv2])
@pytest.mark.parametrize(
    "operation",
    [
        np.asarray,
        lambda x: np.asarray(x, dtype=object),
        lambda x: np.asarray([x]),
        lambda x: np.asarray([x], dtype=object),
        lambda x: np.add(1, x),
        lambda x: np.add.reduce(x),
        lambda x: np.add.accumulate(x),
        lambda x: np.equal(x, x),
        lambda x: np.equal(np.ones(2), x),
        lambda x: np.minimum.outer(x, [1]),
        np.mean,
        np.median,
        lambda x: np.quantile(x, 0.5),
        np.round,
        np.nansum,
        np.nanmin,
        np.logical_not,
        sum,
        min,
        max,
        all,
        any,
        math.prod,
        list,
        bool,
        float,
        int,
        complex,
        lambda x: True & x,
        lambda x: False | x,
        lambda x: 2 // x,
        lambda x: 2 % x,
        lambda x: x and True,
        lambda x: x or False,
    ],
)
def test_numpy_and_python_cannot_bypass_response_guards(cls, operation):
    with pytest.raises(TypeError, match="Invalid operation on a survival time"):
        operation(cls([1, 2], [1, 0]))


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_explicit_values_and_structural_comparisons_remain_available(case):
    x = response(case)
    assert x.equals(response(case))
    assert x.equals(copy.copy(x))
    assert x.equals(copy.deepcopy(x))
    assert x.equals(pickle.loads(pickle.dumps(x)))  # noqa: S301
    assert not x.equals(None)
    assert not x.equals(x.as_matrix())
    assert len(x) == len(case["matrix"])
    expected = np.array(case["matrix"], dtype=float)
    np.testing.assert_equal(np.array(x.as_matrix(), dtype=float), expected)
    # Mutating an extracted matrix cannot mutate the response.
    matrix = x.as_matrix()
    if matrix:
        matrix[0][0] = 99
    assert x.equals(response(case))


@pytest.mark.parametrize("cls", [r.Surv, r.Surv2])
def test_structural_equality_compares_values_and_metadata(cls):
    x = cls([1, None, 3], [1, 0, None])
    assert x.equals(cls([1, math.nan, 3], [1, 0, None]))
    assert not x.equals(cls([1, 2, 3], [1, 0, None]))
    assert not x.equals(cls([1, None, 3], [0, 0, None]))
    assert not x.equals(cls([1, None], [1, 0]))
    if cls is r.Surv:
        assert not x.equals(r.Surv([1, None, 3], [1, 0, None], type="left"))
    else:
        assert not x.equals(r.Surv2([1, None, 3], [1, 0, None], repeated=True))
    with pytest.raises(TypeError, match="unhashable"):
        hash(x)


@pytest.mark.parametrize("name", ["multistate", "timeline_multistate"])
@pytest.mark.parametrize("field", ["states", "clabel"])
def test_structural_equality_distinguishes_labels_with_identical_numeric_codes(name, field):
    case = next(case for case in REFERENCE["cases"] if case["name"] == name)
    changed = copy.deepcopy(case)
    if field == "states":
        changed[field][0] = "renamed state"
    else:
        changed[field] = "other baseline"
    left, right = response(case), response(changed)
    assert left.as_matrix() == right.as_matrix()
    assert not left.equals(right)


def test_failed_numpy_operations_do_not_write_output_or_mutate_responses():
    x = r.Surv([1, 2], [1, 0])
    out = np.full((2, 2), 17.0)
    with pytest.raises(TypeError, match="Invalid operation"):
        np.add(x, 1, out=out)
    np.testing.assert_array_equal(out, 17)
    assert x.equals(r.Surv([1, 2], [1, 0]))
    with pytest.raises(TypeError, match="Invalid operation"):
        x += 1


def test_survival_aware_summaries_still_work():
    x = r.Surv([1, 2, 3], [1, 0, 1])
    for result in (r.median(x), r.quantile(x, probs=[0.5])):
        assert result.probs == [0.5]
        np.testing.assert_equal(result.quantile, [[3.0]])
        np.testing.assert_equal(result.lower, [[1.0]])
        np.testing.assert_equal(result.upper, [[math.nan]])
