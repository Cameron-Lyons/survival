"""Bulk construction of list-column frames, ``finegray`` rows and ``tt()`` risk sets.

Column inference decides string, logical and integer storage from the distinct value
types instead of testing every value; ``finegray``, counting-process ``Surv`` and the
``tt()`` expansion build their row-aligned columns with NumPy. Each path is checked
against the per-element algorithm it replaces (kept here as the reference), including
the Python types of the values it returns.
"""

from __future__ import annotations

import importlib
import math
import numbers
import warnings

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
core = importlib.import_module("survival._survival")
r_coerce = importlib.import_module("survival.r._coerce")
r_expression = importlib.import_module("survival.r._expression")


def reference_kind_and_storage(values):
    """``_expression_source``'s inference for an undeclared list, value by value."""

    if r_expression._all_logical(values):
        kind = "logical"
    elif any(isinstance(value, str) for value in values):
        kind = "character"
    else:
        kind = "numeric"
    storage = "double"
    if kind == "numeric":
        integer = False
        for value in values:
            if r_coerce._is_missing_value(value):
                continue
            if (
                not isinstance(value, numbers.Integral)
                or r_coerce._is_bool_like(value)
                or not (-(2**31) < int(value) < 2**31)
            ):
                integer = False
                break
            integer = True
        if integer:
            storage = "integer"
    return kind, storage


POOLS = {
    "ints": [0, 1, -7, 42, 2**31 - 1, -(2**31) + 1],
    "wide ints": [0, 2**31, -(2**31), 2**40, 3],
    "floats": [0.5, -1.25, 3.0, math.nan, math.inf],
    "mixed numbers": [1, 2.5, np.int64(3), np.float64(4.0), None, math.nan],
    "numpy ints": [np.int64(1), np.int32(-4), 7, np.uint8(2)],
    "logicals": [True, False, None, np.bool_(True), math.nan],
    "strings": ["a", "b", None, np.str_("c"), 1],
    "string subclass": [np.str_("x"), np.str_("y")],
    "bools and ints": [True, 1, 0, False],
}


@pytest.mark.parametrize("pool", sorted(POOLS))
def test_list_column_inference_matches_the_per_value_scan(pool):
    rng = np.random.default_rng(len(pool))
    values = POOLS[pool]
    for size in (1, 2, 5, 40):
        sample = [values[i] for i in rng.integers(0, len(values), size)]
        inferred = r_expression._expression_source(list(sample), "x")
        assert (inferred.kind, inferred.storage or "double") == reference_kind_and_storage(
            sample
        ), sample
        assert list(inferred) == sample or any(
            isinstance(value, float) and math.isnan(value) for value in sample
        )


def test_integer_storage_needs_every_exact_int_inside_the_int32_range():
    assert r_expression._expression_source([1, 2, 2**31 - 1], "x").storage == "integer"
    assert r_expression._expression_source([1, 2**31], "x").storage == "double"
    assert r_expression._expression_source([-(2**31), 1], "x").storage == "double"
    assert r_expression._expression_source([1, None, 3], "x").storage == "integer"
    assert r_expression._expression_source([1, 2.0], "x").storage == "double"


def test_list_and_array_columns_fit_identically():
    rng = np.random.default_rng(4)
    n = 300
    data = {
        "time": rng.exponential(size=n).round(2).tolist(),
        "status": rng.integers(0, 2, n).tolist(),
        "x": rng.normal(size=n).tolist(),
        "g": rng.integers(0, 3, n).tolist(),
        "s": [str(v) for v in rng.choice(["u", "v"], n)],
    }
    arrays = {name: np.asarray(values) for name, values in data.items()}
    formula = "Surv(time, status) ~ x + g + s"
    from_lists = r.coxph(formula, data)
    from_arrays = r.coxph(formula, arrays)
    assert from_lists.coefficients == from_arrays.coefficients
    assert from_lists.loglik == from_arrays.loglik
    assert r.coef_names(from_lists) == ["x", "g", "sv"]


def reference_backwards(start, stop):
    return [
        not (math.isnan(a) or math.isnan(b)) and a >= b for a, b in zip(start, stop, strict=True)
    ]


def test_counting_surv_marks_the_same_backwards_intervals():
    rng = np.random.default_rng(9)
    pool = [0.0, -0.0, 1.0, 2.5, math.nan, math.inf, -math.inf]
    for _ in range(50):
        start = [pool[i] for i in rng.integers(0, len(pool), 12)]
        stop = [pool[i] for i in rng.integers(0, len(pool), 12)]
        status = rng.integers(0, 2, 12).tolist()
        bad = reference_backwards(start, stop)
        expected = [math.nan if flag else value for value, flag in zip(start, bad, strict=True)]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            response = r.Surv(start, stop, status)
        assert [str(value) for value in response.start] == [str(value) for value in expected]
        assert all(type(value) is float for value in response.start)
        assert any("Stop time must be > start time" in str(w.message) for w in caught) == any(bad)


def reference_finegray_columns(split, status, enum):
    source = [row - 1 for row in split.row]
    return source, {
        "fgstart": split.start,
        "fgstop": split.end,
        "fgstatus": [int(status[idx] == enum) for idx in source],
        "fgwt": split.wt,
        "count": split.add,
    }


def test_finegray_rows_match_the_per_row_assembly():
    rng = np.random.default_rng(2)
    n = 120
    time = rng.integers(1, 25, n).astype(float).tolist()
    codes = rng.integers(0, 3, n)
    labels = np.array(["censor", "a", "b"])[codes].tolist()
    data = {
        "time": time,
        "ev": r_coerce._RFactorVector(labels, ["censor", "a", "b"]),
        "x": rng.normal(size=n).tolist(),
        "w": rng.uniform(0.5, 2.0, n).tolist(),
    }
    for etype in ("a", "b"):
        frame = r.finegray("Surv(time, ev) ~ x", data, etype=etype, weights=data["w"], count="k")
        enum = ["a", "b"].index(etype) + 1
        status = codes.tolist()
        split = core.finegray_expand(time, status, enum, weights=data["w"])
        source, expected = reference_finegray_columns(split, status, enum)
        assert frame["x"] == [data["x"][idx] for idx in source]
        assert frame["(weights)"] == [data["w"][idx] for idx in source]
        for name in ("fgstart", "fgstop", "fgstatus", "fgwt"):
            assert frame[name] == expected[name]
            assert [type(value) for value in frame[name]] == [
                type(value) for value in expected[name]
            ]
        assert frame["k"] == expected["count"]
        assert all(type(value) is int for value in frame["k"])


def test_tt_expansion_passes_the_same_risk_set_columns():
    rng = np.random.default_rng(5)
    n = 80
    data = {
        "time": rng.integers(1, 15, n).astype(float).tolist(),
        "status": rng.integers(0, 2, n).tolist(),
        "x": rng.normal(size=n).tolist(),
        "w": rng.uniform(0.5, 2.0, n).tolist(),
        "g": rng.integers(0, 2, n).tolist(),
    }
    seen = []

    def transform(x, t, riskset, weights):
        seen.append((x, t, riskset, weights))
        return np.asarray(x) * np.log(t)

    fit = r.coxph(
        "Surv(time, status) ~ x + tt(x) + strata(g)",
        data,
        tt=transform,
        weights=data["w"],
    )
    counts = core.coxcount1(core.SurvivalData(data["time"], data["status"]), data["g"])
    tindex = [int(idx) for idx in counts.index]
    nrisk = [int(value) for value in counts.nrisk]
    new_time = [time for time, size in zip(counts.time, nrisk, strict=True) for _ in range(size)]
    riskset = [group + 1 for group, size in enumerate(nrisk) for _ in range(size)]
    x, t, sets, weights = seen[-1]
    assert list(x) == [data["x"][idx] for idx in tindex]
    assert t == new_time
    assert all(type(value) is float for value in t)
    assert sets == riskset
    assert all(type(value) is int for value in sets)
    assert weights == [data["w"][idx] for idx in tindex]
    assert fit.n == n
    assert len(fit.coefficients) == 2
