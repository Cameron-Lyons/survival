"""Curve summary data frames against stock R, including Cox prediction margins."""

import importlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
FIXTURES = Path(__file__).parent / "fixtures"
REFERENCE = json.loads((FIXTURES / "curve_summary_frame_reference.json").read_text())
COUNTS = json.loads((FIXTURES / "summary_counts_reference.json").read_text())


@pytest.fixture(scope="module")
def fits():
    data = dict(COUNTS["data"])
    data["event"] = r.Factor(data["event"], COUNTS["event_levels"])
    result = {}
    for name, spec in REFERENCE["fits"].items():
        options = {key.replace(".", "_"): value for key, value in spec.get("fit_args", {}).items()}
        if spec["kind"] == "formula":
            result[name] = r.survfit(spec["formula"], data, **options)
        else:
            model = r.coxph(
                spec["formula"], data, **({"id": "id"} if spec["kind"] == "coxms" else {})
            )
            result[name] = r.survfit(model, newdata=spec.get("newdata"), **options)
    return result


def assert_frame(actual, expected):
    assert list(actual) == list(expected)
    for name, values in expected.items():
        if any(isinstance(value, str) for value in values):
            assert actual[name] == values, name
        else:
            np.testing.assert_allclose(
                actual[name],
                np.asarray(values, dtype=float),
                rtol=1e-9,
                atol=1e-12,
                equal_nan=True,
                err_msg=name,
            )


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_summary_data_frame_matches_stock_r(case, fits):
    actual = r.summary_survfit(fits[case["fit"]], data_frame=True, **dict(case["arguments"]))
    assert_frame(actual, case["expected"])


@pytest.mark.parametrize("kind", ["km", "cox", "coxms"])
def test_summary_generic_returns_the_same_data_frame(kind, fits):
    expected = r.summary_survfit(fits[kind], times=[2, 4], data_frame=True)
    assert_frame(r.model_summary(fits[kind], times=[2, 4], data_frame=True), expected)


def test_formula_data_frame_does_not_compute_unused_mean_table(monkeypatch, fits):
    def unexpected(*args, **kwargs):
        raise AssertionError("data-frame summaries do not need a compact mean/quantile table")

    module = importlib.import_module("survival.r._survfit")
    monkeypatch.setattr(module._core, "survmean", unexpected)
    result = r.summary_survfit(fits["km"], times=[2, 4], data_frame=True)
    assert result["time"] == [2.0, 4.0]
    result = r.summary_survfit(fits["cox"], times=[2, 4], data_frame=True)
    assert result["time"] == [2.0, 4.0, 2.0, 4.0]


def test_empty_multistate_data_frames_keep_every_column(fits):
    for kind in ("aj", "coxms"):
        expected = r.summary_survfit(fits[kind], times=[2, 4], data_frame=True)
        actual = r.summary_survfit(fits[kind], times=[20], data_frame=True)
        assert actual == dict.fromkeys(expected, [])


def test_summary_data_frame_validates_rmean_before_rendering(fits):
    with pytest.raises(ValueError, match="Invalid value for rmean"):
        r.summary_survfit(fits["cox"], data_frame=True, rmean="invalid")


@pytest.mark.parametrize("kind", ["km", "aj", "cox", "coxms"])
@pytest.mark.parametrize("data_frame", [False, True])
@pytest.mark.parametrize("scale", [0, -1])
def test_summary_checks_scale_without_requiring_a_mean_table(kind, data_frame, scale, fits):
    with pytest.raises(ValueError, match="scale must be a positive number"):
        r.summary_survfit(fits[kind], scale=scale, data_frame=data_frame)


def test_multistate_prediction_columns_cannot_overwrite_state_or_time(fits):
    fit = fits["coxms"]
    newdata = {"x": [0, 2], "state": [17, 29], "time": [101, 202], "state_": [31, 43]}
    collision = replace(fit, newdata=newdata)
    original = r.summary_survfit(fit, times=[2, 4], data_frame=True)
    for frame in (
        r.summary_survfit(collision, times=[2, 4], data_frame=True),
        r.as_data_frame(r.summary_survfit(collision, times=[2, 4])),
    ):
        assert frame["state"] == original["state"]
        assert frame["time"] == original["time"]
        assert frame["pstate"] == original["pstate"]
        assert frame["x_"] == original["x"]
        assert frame["state_"] == [17, 17, 29, 29] * len(fit.states)
        assert frame["time_"] == [101, 101, 202, 202] * len(fit.states)
        assert frame["state__"] == [31, 31, 43, 43] * len(fit.states)
        assert len(frame) == len(original) + 3
