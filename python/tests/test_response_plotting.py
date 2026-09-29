"""R response/expected-survival graphics and model-specific dispatch."""

import copy
import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
plotting, r = survival.plotting, survival.r
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/response_plot_reference.json").read_text()
)
for case in REFERENCE["cases"]:
    case["options"] = dict(case["options"])


def response(name):
    encoded = REFERENCE["responses"][name]
    values = np.asarray(encoded["values"], dtype=float)
    kind = encoded["type"]
    if kind == "mright":
        from survival.r._coerce import _r_factor

        labels = ["censor", *encoded["states"]]
        event = _r_factor([labels[int(code)] for code in values[:, 1]], labels)
        return r.Surv(values[:, 0], event)
    return r.Surv(*(values[:, i] for i in range(values.shape[1])), type=kind)


def expected_fit(name):
    encoded = REFERENCE["expected"][name]
    return r.SurvExpResult(
        time=encoded["time"],
        surv=encoded["surv"],
        n_risk=encoded["n_risk"],
        strata=encoded["names"],
        method=encoded["method"],
        n=6,
    )


def test_expected_data_preserves_group_names_and_never_changes_fit():
    fit = expected_fit("groups")
    original = copy.deepcopy(fit)
    data = plotting.survfit_plot_data(fit)
    assert [curve.label for curve in data.curves] == fit.strata
    assert all(curve.lower is None and not curve.event.any() for curve in data.curves)
    data.curves[0].estimate[:] = 0
    data.curves[0].time[:] = -1
    assert fit == original


def test_expected_hazard_is_computed_from_survival_and_bands_are_not_inferred():
    fit = expected_fit("groups")
    data = plotting.survfit_plot_data(fit, cumhaz=True)
    expected = -np.log(fit.surv)
    for col, curve in enumerate(data.curves):
        np.testing.assert_allclose(curve.estimate, np.r_[0, expected[:, col]])
    with pytest.raises(ValueError, match="does not have standard errors"):
        plotting.survfit_plot_data(fit, conf_int=True)
    assert all(
        len(curve.censor_time) == 0
        for curve in plotting.survfit_plot_data(fit, mark_time=True).curves
    )


def test_actual_population_fit_has_the_reference_expected_curves():
    frame = {
        "time": [100, 300, 500, 800, 1200, 1500],
        "age": np.asarray([50, 60, 70, 80, 55, 65]) * 365.25,
        "sex": [1, 2, 1, 2, 1, 2],
        "year": [date(2000, 1, 1)] * 6,
    }
    fit = r.survexp(
        "~ sex",
        frame,
        rmap={"age": "age", "sex": "sex", "year": "year"},
        times=REFERENCE["expected"]["groups"]["time"],
    )
    actual = plotting.survfit_plot_data(fit)
    expected = plotting.survfit_plot_data(expected_fit("groups"))
    for a, b in zip(actual.curves, expected.curves, strict=True):
        np.testing.assert_allclose(a.estimate, b.estimate, rtol=1e-12)


@pytest.mark.parametrize("method", ["plot", "lines", "points"])
def test_surv2_methods_have_rs_explicit_refusal(method):
    with pytest.raises(ValueError, match="method not defined for a Surv2 object"):
        getattr(plotting, method)(r.Surv2([1, 2], [0, 1]))


@pytest.mark.parametrize("method", ["lines", "points"])
def test_raw_response_overlay_methods_have_rs_explicit_refusal(method):
    with pytest.raises(ValueError, match="method not defined for a Surv object"):
        getattr(plotting, method)(response("right"))


def test_invalid_explicit_method_inputs_fail_before_rendering():
    with pytest.raises(TypeError, match="Surv object"):
        plotting.plot_surv(expected_fit("single"))
    with pytest.raises(TypeError, match="survexp"):
        plotting.lines_survexp(response("right"))
    with pytest.raises(TypeError, match="survfit result"):
        plotting.plot(None)


@pytest.fixture
def pyplot():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    from matplotlib import pyplot as plt

    yield plt
    plt.close("all")


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_graphics_dispatch_coordinates_match_r(case, pyplot):
    value = response(case["key"]) if case["source"] == "response" else expected_fit(case["key"])
    options = {
        key.replace(".", "_"): value for key, value in case["options"].items() if key != "col"
    }
    if "type" in options:
        options["drawstyle"] = {"s": "steps-post", "l": "default"}[options.pop("type")]
    result = getattr(plotting, case["method"])(value, **options)
    calls = []
    for artist in result.axes.lines:
        x, y = artist.get_xdata(), artist.get_ydata()
        if len(x):
            calls.append(
                {"kind": "point" if artist.get_linestyle() == "None" else "line", "x": x, "y": y}
            )
    expected = [call for call in case["expected"]["calls"] if call["x"]]
    assert len(calls) == len(expected)
    # R draws censor marks before confidence lines; Matplotlib adds them after.
    # Compare coordinates in their original order within each artist kind.
    calls.sort(key=lambda call: call["kind"])
    expected.sort(key=lambda call: call["kind"])
    for actual, wanted in zip(calls, expected, strict=True):
        assert actual["kind"] == wanted["kind"]
        for field in ("x", "y"):
            np.testing.assert_allclose(
                actual[field], np.asarray(wanted[field], dtype=float), rtol=2e-10, atol=2e-12
            )
    result.axes.figure.canvas.draw()


def test_expected_overlay_defaults_to_straight_lines_and_keeps_axes(pyplot):
    observed = plotting.plot(response("right"), xscale=365.25)
    before = observed.axes.get_xlim(), observed.axes.get_ylim()
    result = plotting.lines(expected_fit("groups"), ax=observed.axes, xscale=365.25)
    assert (observed.axes.get_xlim(), observed.axes.get_ylim()) == before
    assert len(result.lines[0].get_xdata()) == len(expected_fit("groups").time) + 1
    assert result.lines[0].get_drawstyle() == "default"
    explicit = plotting.lines_survfit(expected_fit("single"), ax=observed.axes)
    assert len(explicit.lines[0].get_xdata()) > len(result.lines[0].get_xdata())


def test_generic_model_diagnostics_and_fitted_point_methods(pyplot):
    lung = survival.datasets.load_lung()
    cox = r.cox_zph(r.coxph("Surv(time,status) ~ age + sex", lung))
    assert isinstance(plotting.plot(cox, var="age"), plotting.CoxDiagnosticPlot)
    aalen = r.aareg("Surv(futime,fustat) ~ age + ecog.ps", survival.datasets.load_ovarian())
    assert isinstance(plotting.plot(aalen, var="age"), plotting.AalenPlot)
    assert isinstance(plotting.lines(aalen, var="age"), plotting.AalenPlot)
    curve = r.survfit(response("right"))
    assert plotting.points(curve).lines


def test_straight_and_alternative_steps_keep_uncompressed_band_coordinates(pyplot):
    fit = r.survfit(r.Surv([1, 2, 3, 4], [1, 0, 0, 1]))
    for style in ("default", "steps-pre", "steps-mid"):
        result = plotting.plot_survfit(fit, drawstyle=style, conf_style="band")
        curve = result.data.curves[0]
        np.testing.assert_array_equal(result.lines[0].get_xdata(), curve.time)
        assert result.lines[0].get_drawstyle() == style
        assert 3 in result.confidence[0].get_paths()[0].vertices[:, 0]
    with pytest.raises(ValueError, match="drawstyle"):
        plotting.plot_survfit(fit, drawstyle="diagonal")
