"""Survival residual row omission/exclusion against stock R survival 3.8-12."""

import copy
import json
import pickle
import warnings
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from survival import r

from .r_fixture_support import RFactor

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/survfit_missing_residual_reference.json").read_text(
        encoding="utf-8"
    )
)


def data_for(name):
    dataset = REFERENCE["datasets"][name]
    data = dict(dataset["data"])
    if dataset["states"] is not None:
        data["event"] = RFactor(data["event"], dataset["states"])
    return data


@cache
def fitted(name, action):
    dataset = REFERENCE["datasets"][name]
    return r.survfit(
        dataset["formula"],
        data_for(name),
        id="subject" if dataset["id"] else None,
        weights="weight",
        na_action=action,
        timefix=False,
    )


def expected_array(encoded):
    values = np.asarray(encoded["values"], dtype=float)
    return values if encoded["dim"] is None else values.reshape(encoded["dim"], order="F")


def assert_frame(actual, expected):
    assert list(actual) == list(expected)
    for name, values in actual.items():
        if name in {"resid", "pseudo", "time"}:
            np.testing.assert_allclose(values, expected[name], rtol=2e-11, atol=3e-13)
        else:
            assert values == expected[name]


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_survfit_missing_residual_rows_match_r(case):
    fit = fitted(case["dataset"], case["na_action"])
    assert fit.na_action == r.NaAction(tuple(case["na_rows"]), case["na_action"][3:])
    options = {"times": case["times"], "type": case["type"], "collapse": case["collapse"]}
    result = r.survfit_residuals(fit, **options)
    actual = np.asarray(result.resid)
    expected = expected_array(case["residual"])
    if actual.ndim == 3 and expected.ndim == 2:
        actual = actual[:, :, 0]
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=3e-13)
    np.testing.assert_allclose(result.curve, np.asarray(case["curve"], dtype=float), rtol=0, atol=0)
    row_names = case["residual"]["row_names"]
    if row_names is not None:
        assert [str(value) for value in result.id] == row_names
    assert_frame(r.survfit_residuals(fit, data_frame=True, **options), case["residual_frame"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        values = r.pseudo(fit, **options)
        frame = r.pseudo(fit, data_frame=True, **options)
    np.testing.assert_allclose(values, expected_array(case["pseudo"]), rtol=2e-11, atol=3e-13)
    assert_frame(frame, case["pseudo_frame"])


@pytest.mark.parametrize("multistate", [False, True], ids=["km", "aj"])
@pytest.mark.parametrize("action", ["na.omit", "na.exclude"])
def test_bare_response_missing_rows_follow_formula_and_survive_survfit0(multistate, action):
    data = data_for("aj_right" if multistate else "km_right")
    event = data["event"] if multistate else data["status"]
    formula = "Surv(time, event) ~ group" if multistate else "Surv(time, status) ~ group"
    fit = r.survfit(formula, data, weights="weight", na_action=action, timefix=False)
    bare = r.survfit(
        r.Surv(data["time"], event),
        group=data["group"],
        weights=data["weight"],
        na_action=action,
        timefix=False,
    )
    assert bare.na_action == fit.na_action == r.NaAction((2, 6), action[3:])
    expected = r.survfit_residuals(fit, times=[2.5, 6.5])
    restored = pickle.loads(pickle.dumps(bare))  # noqa: S301 -- this test's own serialized object
    for transformed in (bare, r.survfit0(bare), copy.deepcopy(bare), restored):
        assert transformed.na_action == bare.na_action
        actual = r.survfit_residuals(transformed, times=[2.5, 6.5])
        assert actual.id == ([1, 3, 4, 5, 7, 8] if action == "na.omit" else list(range(1, 9)))
        np.testing.assert_allclose(actual.resid, expected.resid, rtol=0, atol=0)
        np.testing.assert_allclose(actual.curve, expected.curve, rtol=0, atol=0)


def test_subset_missing_positions_are_relative_to_selected_rows():
    data = data_for("km_right")
    fit = r.survfit(
        "Surv(time, status) ~ group",
        data,
        subset=[7, 1, 0, 6, 5, 2],
        weights="weight",
        na_action="na.exclude",
        timefix=False,
    )
    assert fit.na_action == r.NaAction((2, 5), "exclude")
    result = r.survfit_residuals(fit, times=[2.5])
    assert result.id == list(range(1, 7))
    np.testing.assert_array_equal(
        np.isnan(np.asarray(result.resid)[:, 0]), [False, True, False, False, True, False]
    )
    frame = r.survfit_residuals(fit, times=[2.5], data_frame=True)
    assert frame["(id)"] == [1, 3, 4, 6]


def test_restored_multistate_missing_rows_keep_independent_state_time_dimensions():
    fit = fitted("aj_right", "na.exclude")
    residual = r.survfit_residuals(fit, times=[2.5, 6.5])
    assert np.shape(residual.resid) == (8, 3, 2)
    assert np.isnan(residual.resid[1]).all()
    assert np.isnan(residual.resid[5]).all()
    residual.resid[1][0][0] = 123
    assert np.isnan(residual.resid[1][1][0])
    assert np.isnan(residual.resid[5][0][0])
