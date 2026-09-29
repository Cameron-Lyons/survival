"""Reference coordinates captured from R's actual plot.cox.zph method."""

from __future__ import annotations

import copy
import json
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
plotting = survival.plotting
REFERENCE = json.loads((Path(__file__).parent / "fixtures/cox_zph_plot_reference.json").read_text())
for case in REFERENCE["cases"]:
    case["options"] = dict(case["options"])


def fitted(name):
    values = REFERENCE["fits"][name]
    return survival.r.CoxZPHResult(
        table=[],
        x=values["x"],
        time=values["time"],
        y=np.asarray(values["y"], dtype=float).tolist(),
        var=values["variance"],
        names=values["names"],
        transform=values["transform"],
    )


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_diagnostic_coordinates_match_r(case):
    options = {key: value for key, value in case["options"].items() if key != "resid"}
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        data = plotting.cox_zph_plot_data(fitted(case["fit"]), **options)
    assert len(captured) == len(case["expected"]["warnings"])
    assert len(data.curves) == len(case["expected"]["plots"])
    for curve, expected in zip(data.curves, case["expected"]["plots"], strict=True):
        assert data.xlog == ("x" in expected["log"])
        assert data.ylog == ("y" in expected["log"])
        for values, line in zip(
            (curve.estimate, curve.upper, curve.lower), expected["lines"], strict=False
        ):
            np.testing.assert_allclose(curve.x, line["x"], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(values, line["y"], rtol=2e-9, atol=2e-11)
        if case["options"].get("resid", True):
            np.testing.assert_allclose(curve.residual_x, expected["points"]["x"], rtol=1e-12)
            np.testing.assert_allclose(curve.residual_y, expected["points"]["y"], rtol=1e-12)
        if expected["ticks"] is not None:
            np.testing.assert_allclose(
                data.tick_positions, expected["ticks"], rtol=1e-12, equal_nan=True
            )
            assert [float(label) for label in data.tick_labels] == [
                float(label) for label in expected["labels"]
            ]


def test_real_fitted_model_has_the_reference_smoothing():
    result = survival.r.cox_zph(
        survival.r.coxph("Surv(time,status) ~ age + sex", survival.datasets.load_lung())
    )
    actual = plotting.cox_zph_plot_data(result)
    expected = plotting.cox_zph_plot_data(fitted("km"))
    for a, b in zip(actual.curves, expected.curves, strict=True):
        np.testing.assert_allclose(a.estimate, b.estimate, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(a.lower, b.lower, rtol=1e-9, atol=1e-11)


def test_preparation_does_not_mutate_fit_or_share_curve_arrays():
    result = fitted("identity")
    original = copy.deepcopy(result)
    data = plotting.cox_zph_plot_data(result)
    data.curves[0].x[:] = -1
    data.curves[0].residual_y[:] = 0
    assert result == original
    assert np.all(data.curves[1].x > 0)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"df": 1}, "at least 2"),
        ({"nsmo": 1}, "at least 2"),
        ({"hr": 1}, "boolean"),
        ({"se": "yes"}, "boolean"),
        ({"var": 0}, "invalid variable"),
        ({"var": []}, "invalid variable"),
        ({"var": "missing"}, "term names"),
        ({"var": 1.5}, "term names"),
        ({"var": 3}, "invalid variable"),
    ],
)
def test_options_are_validated(options, message):
    with pytest.raises((ValueError, TypeError), match=message):
        plotting.cox_zph_plot_data(fitted("km"), **options)


def test_malformed_results_and_unusable_times_are_rejected():
    fit = fitted("km")
    with pytest.raises(TypeError, match="cox_zph"):
        plotting.cox_zph_plot_data(None)
    with pytest.raises(ValueError, match="inconsistent shapes"):
        plotting.cox_zph_plot_data(replace(fit, time=[]))
    with pytest.raises(ValueError, match="distinct"):
        plotting.cox_zph_plot_data(replace(fit, x=[1.0] * len(fit.x)))
    with pytest.raises(ValueError, match="finite"):
        plotting.cox_zph_plot_data(replace(fit, x=[np.nan] * len(fit.x)))


@pytest.mark.parametrize("layout", ["c", "fortran", "strided", "list"])
def test_native_boundary_accepts_matrix_layouts(layout):
    x = np.linspace(0, 1, 50)
    y = np.column_stack((np.sin(x), np.cos(x)))
    expected = survival.regression.cox_zph_smooth(x, y, [1, 2])
    if layout == "fortran":
        y = np.asfortranarray(y)
    elif layout == "strided":
        y = np.repeat(y, 2, axis=0)[::2]
    elif layout == "list":
        y = y.tolist()
    actual = survival.regression.cox_zph_smooth(x, y, [1, 2])
    np.testing.assert_array_equal(actual.y, expected.y)
    np.testing.assert_array_equal(actual.std_err, expected.std_err)


def test_native_validation_and_no_observed_residuals():
    smooth = survival.regression.cox_zph_smooth
    x = [0.0, 1.0, 2.0]
    y = np.full((3, 1), np.nan)
    result = smooth(x, y, [1.0])
    assert result.skipped == [0]
    assert np.isnan(result.y).all()
    assert np.isnan(result.std_err).all()
    assert smooth(x, np.ones((3, 1)), [np.nan], df=2, se=False).std_err is None
    with pytest.raises(ValueError, match="nonnegative"):
        smooth(x, y, [-1.0])
    with pytest.raises(ValueError, match="finite or NaN"):
        smooth(x, np.full((3, 1), np.inf), [1.0])
    with pytest.raises(ValueError, match="rows must match"):
        smooth(x[:2], y, [1.0])
