"""Stock single-curve selection and explicit stale-uncertainty correction."""

import json
import pickle
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from .helpers import setup_survival_import

r = setup_survival_import().r_api
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/cox_survfit_single_reference.json").read_text()
)


def data():
    return pd.DataFrame(
        {
            "time": range(1, 13),
            "status": [1, 1, 0] * 4,
            "x": [0, 1, 2] * 4,
            "z": [0, 1, 0.5, -1, 2, 0.7, 1, 0, -1, 0.5, 2, 1],
            "g": pd.Categorical(["a"] * 6 + ["b"] * 6),
        }
    )


@cache
def fit(grouped=False):
    formula = "Surv(time,status) ~ x + z" + (" + strata(g)" if grouped else "")
    return r.coxph(formula, data(), init=[0.25, -0.125], iter_max=0)


def newdata():
    return pd.DataFrame(
        {
            "x": [-1, 0, 2],
            "z": [1, 0, -1],
            "tag": pd.Categorical(["b", "a", "b"], categories=["a", "unused", "b"]),
        },
        index=["two", "one", "three"],
    )


@cache
def source(name):
    if name == "default":
        return r.survfit(fit())
    if name == "aggregated":
        return r.aggregate_survfit(r.survfit(fit(), newdata=newdata()))
    if name == "selected_one":
        return r.survfit(fit(True), newdata=newdata()).subset(strata=[0], data=[0])
    return r.survfit(
        fit(),
        newdata=newdata().iloc[[0]],
        **({"start_time": 4.5} if name == "starts_late" else {}),
    )


def check_values(actual, expected):
    for name in ["time", "surv", "cumhaz", "std.err", "n"]:
        value = getattr(actual, name.replace(".", "_"))
        if expected[name] is None:
            # The native aggregate represents cleared cumulative hazards as [].
            assert value is None or name == "cumhaz" and np.size(value) == 0
        else:
            np.testing.assert_allclose(
                np.asarray(value).ravel(), np.asarray(expected[name]).ravel(), rtol=2e-12
            )
    assert actual.start_time == expected["start.time"]
    assert actual.logse is expected["logse"]
    assert actual.dim == (expected["dim"] or {})
    assert (actual.newdata is None) == (expected["newdata"] is None)


@pytest.mark.parametrize(
    "case",
    [case for case in REFERENCE["single"]["cases"] if "error" not in case["expected"]],
    ids=lambda case: case["name"],
)
def test_implicit_single_curve_selection_and_serialization_match_stock(case):
    original = source(case["source"])
    noop = case["selector"] in {"missing", "null"}
    selected = r._subset_cox_survfit(original, _implicit=not noop)
    assert (selected is original) is noop
    check_values(selected, case["expected"]["value"])
    check_values(pickle.loads(pickle.dumps(selected)), case["expected"]["serialized"])  # noqa: S301
    if not noop:
        assert selected.newdata is None
        assert selected.start_time is None
    assert (original.newdata is None) == (
        REFERENCE["single"]["sources"][case["source"]]["newdata"] is None
    )


@pytest.mark.parametrize("name", REFERENCE["start_time"])
def test_margin_and_linear_selection_drop_start_time_except_noops(name):
    original = r.survfit(fit(True), newdata=newdata()[["x", "z"]], start_time=4.5)
    options = {
        "missing": {},
        "null": {},
        "both_missing": {},
        "full_margins": {"strata": [0, 1], "data": [0, 1, 2], "drop": False},
        "reorder_margins": {"strata": [1, 0], "data": [2, 1, 0], "drop": False},
        "repeat_margins": {"strata": [1, 0, 1], "data": [2, 0], "drop": False},
        "single_margin": {"strata": [0], "drop": False},
        "data_margin": {"data": [0], "drop": False},
        "single_both": {"strata": [0], "data": [0]},
        "linear_full": {"curves": list(range(6))},
        "linear_reorder": {"curves": list(range(5, -1, -1))},
        "linear_repeat": {"curves": [3, 0, 3]},
        "linear_single": {"curves": [1]},
    }
    selected = r._subset_cox_survfit(original, **options[name])
    expected = REFERENCE["start_time"][name]
    assert selected.start_time == expected["start.time"]
    assert selected.dim == (expected["dim"] or {})
    np.testing.assert_allclose(selected.time, expected["time"], rtol=2e-12)
    np.testing.assert_allclose(selected.surv, expected["surv"], rtol=2e-12)
    assert original.start_time == 4.5


@pytest.mark.parametrize("case", ["none", "constant", "two_groups"])
def test_aggregation_clears_stock_stale_cumulative_hazard_uncertainty(case):
    queries = pd.DataFrame({"x": [-1, 0, 2, 3], "z": [1, 0, -1, 0.5]})
    original = r.survfit(fit(True), newdata=queries)
    stock = REFERENCE["std_chaz"]["cases"][case]
    by = None if case == "none" else ["same"] * 4 if case == "constant" else ["b", "a", "b", "a"]
    selected = r.aggregate_survfit(original, by=by)
    np.testing.assert_allclose(
        np.asarray(selected.surv).ravel(),
        np.asarray(stock["aggregate"]["surv"]).ravel(),
        rtol=2e-12,
    )
    raw = REFERENCE["std_chaz"]["cases"]["original"]["std.chaz"]
    np.testing.assert_allclose(original.std_chaz, raw, rtol=2e-12)
    assert stock["identical_uncertainty"]
    assert stock["aggregate"]["std.chaz_dim"] == [12, 4]
    assert stock["aggregate"]["surv_dim"] == ([12, 2] if case == "two_groups" else None)
    assert selected.std_chaz is stock["corrected"]["std.chaz"] is None
    if case == "two_groups":
        assert stock["aggregate"]["newdata"][0]["aggregate"] == "a"
        # Group a consists of source columns 2 and 4, but stock assigns column 1.
        np.testing.assert_array_equal(stock["selected_first"]["std.chaz"], np.asarray(raw)[:6, 0])
        np.testing.assert_array_equal(stock["selected_second"]["std.chaz"], np.asarray(raw)[:6, 1])
