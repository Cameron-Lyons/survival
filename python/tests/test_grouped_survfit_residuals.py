"""Grouped residual and pseudo arrays and row order from unmodified stock R."""

import json
import warnings
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from survival import r

from .r_fixture_support import RFactor

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/grouped_residual_reference.json").read_text()
)


@cache
def fitted(name):
    case = REFERENCE["datasets"][name]
    data = dict(case["data"])
    data["group"] = RFactor(data["group"], case["levels"])
    if case["states"] is not None:
        data["event"] = RFactor(data["event"], case["states"])
    return r.survfit(
        case["formula"],
        data,
        id="subject",
        cluster="cluster",
        weights="weight",
        start_time=case["start_time"],
        na_action="na.omit",
        timefix=False,
    )


def call(case, data_frame=False):
    options = {
        "times": REFERENCE["times"],
        "type": case["type"],
        "collapse": case["collapse"],
        "data_frame": data_frame,
    }
    if case["operation"] == "residual":
        options["weighted"] = case["weighted"]
        return r.survfit_residuals(fitted(case["dataset"]), **options)
    return r.pseudo(fitted(case["dataset"]), **options)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_grouped_residual_and_pseudo_arrays_match_r(case):
    result = call(case)
    actual = result.resid if case["operation"] == "residual" else result
    encoded = case["expected"]
    expected = np.asarray(encoded["values"], dtype=float).reshape(encoded["dim"], order="F")
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=3e-13)
    if case["operation"] == "residual":
        assert result.time == REFERENCE["times"]
        assert result.id == case["id"]
        assert result.curve == case["curve"]
        assert result.columns == case["columns"]


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_grouped_residual_and_pseudo_long_tables_match_r(case):
    actual = call(case, data_frame=True)
    expected = case["data_frame"]
    assert list(actual) == list(expected)
    for name, values in actual.items():
        if name in {"resid", "pseudo", "time"}:
            np.testing.assert_allclose(values, expected[name], rtol=2e-11, atol=3e-13)
        else:
            assert values == expected[name]


@pytest.mark.parametrize("multistate", [False, True], ids=["km", "aj"])
def test_ragged_curve_endpoint_warning_and_values_follow_each_stratum(multistate):
    # Stratum lengths 2/3/4/5 and endpoints 2/3/4/5 require both the row
    # offsets and the per-curve warning comparison to follow curve order.
    time, event, group = [], [], []
    for label, size in zip(["z", "a", "m", "c"], [2, 3, 4, 5], strict=True):
        time.extend(range(1, size + 1))
        event.extend([1, *([0] * (size - 1))])
        group.extend([label] * size)
    order = np.random.default_rng(722).permutation(len(time))
    data = {
        "time": [time[i] for i in order],
        "event": [event[i] for i in order],
        "group": [group[i] for i in order],
    }
    if multistate:
        data["event"] = RFactor(
            ["a" if value else "censor" for value in data["event"]], ["censor", "a"]
        )
    fit = r.survfit("Surv(time,event) ~ group", data, timefix=False)
    assert list(fit.strata.values()) == [3, 5, 4, 2]
    times = [1.5, 3.5, 4.5]
    with pytest.warns(UserWarning, match="beyond the end of one or more curves") as caught:
        actual = np.asarray(r.pseudo(fit, times=times))
    assert len(caught) == 1
    expected = np.empty_like(actual)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for label in set(group):
            rows = [i for i, value in enumerate(data["group"]) if value == label]
            separate = {
                "time": [data["time"][i] for i in rows],
                "event": [data["event"][i] for i in rows],
            }
            if multistate:
                separate["event"] = RFactor(separate["event"], ["censor", "a"])
            curve = r.survfit("Surv(time,event) ~ 1", separate, timefix=False)
            expected[rows] = r.pseudo(curve, times=times)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
    # Exactly at the earliest endpoint needs no warning, including duplicate
    # and unsorted query times that the public residual/pseudo facade normalizes.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        r.pseudo(fit, times=[2, 1.5, 2])
