"""Rendered artists and figure integration for survival graphics."""

import builtins
import io

import numpy as np
import pytest
from survival import plotting, r

from .test_survfit_plot_data import REFERENCE, options_for, reference_fit

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_rendered_artists_match_prepared_coordinates(case):
    options = options_for(case)
    result = getattr(plotting, case["method"] + "_survfit")(reference_fit(case["fit"]), **options)
    assert isinstance(result, plotting.SurvivalPlot)
    result.axes.figure.canvas.draw()
    if case["method"] == "points":
        for artist, curve in zip(result.lines, result.data.curves, strict=True):
            keep = np.ones(len(curve.time), dtype=bool) if options.get("censor") else curve.event
            np.testing.assert_array_equal(artist.get_xdata(), curve.time[keep])
            np.testing.assert_array_equal(artist.get_ydata(), curve.estimate[keep])
    elif result.data.plot_estimate:
        for artist, curve in zip(result.lines, result.data.curves, strict=True):
            xx, yy = curve.step()
            np.testing.assert_array_equal(artist.get_xdata(), xx)
            np.testing.assert_array_equal(artist.get_ydata(), yy)
    else:
        assert not result.lines
    if case["method"] == "plot":
        assert result.axes.get_xscale() == ("log" if result.data.xlog else "linear")
        assert result.axes.get_yscale() == ("log" if result.data.ylog else "linear")


def test_rendered_censor_marks_and_decreasing_confidence_band():
    result = plotting.plot_survfit(
        reference_fit("km"), fun="event", conf_style="band", mark_time=True
    )
    curve = result.data.curves[0]
    assert np.all(curve.lower >= curve.upper)
    assert len(result.confidence) == 1
    assert result.marks[0].get_color() == result.lines[0].get_color()
    np.testing.assert_array_equal(result.marks[0].get_ydata(), curve.censor_value)
    result.axes.figure.canvas.draw()


def test_scaled_axes_labels_and_overlays_preserve_existing_view():
    fit = reference_fit("groups")
    result = plotting.plot_survfit(
        fit,
        xscale=2,
        yscale=100,
        colors=["navy", "orange"],
        linestyles=["-", ":"],
        xlabel="Years",
        ylabel="Percent",
        xlim=(0, 7),
        ylim=(0.1, 1),
        mark_time=True,
    )
    assert result.axes.get_xlabel() == "Years"
    assert result.axes.get_ylabel() == "Percent"
    np.testing.assert_allclose(result.axes.get_xlim(), [0, 3.5])
    np.testing.assert_allclose(result.axes.get_ylim(), [10, 100])
    before = result.axes.get_xlim(), result.axes.get_ylim()
    added = plotting.lines_survfit(reference_fit("km"), ax=result.axes, xscale=2, yscale=100)
    assert added.axes is result.axes
    assert not added.confidence
    assert before == (result.axes.get_xlim(), result.axes.get_ylim())
    for artist, curve in zip(result.lines, result.data.curves, strict=True):
        xx, yy = curve.step()
        np.testing.assert_allclose(artist.get_xdata(), xx / 2)
        np.testing.assert_allclose(artist.get_ydata(), yy * 100)


def test_confidence_bars_use_r_next_time_interpolation():
    from survival._plot_data import step_at

    query = np.asarray([1.5, 2, 4.5])
    result = plotting.plot_survfit(reference_fit("km"), conf_times=query, conf_offset=0, conf_cap=0)
    curve = result.data.curves[0]
    segments = np.asarray(result.confidence[0].get_segments())
    np.testing.assert_array_equal(segments[:, :, 0], np.column_stack((query, query)))
    np.testing.assert_allclose(
        segments[:, 0, 1], step_at(curve.time, curve.lower, query, right=False)
    )
    np.testing.assert_allclose(
        segments[:, 1, 1], step_at(curve.time, curve.upper, query, right=False)
    )


def test_graphics_export_and_real_cox_fit():
    data = {"time": np.arange(1, 21), "status": [1, 1, 0, 1, 0] * 4, "x": [0, 2, 1, 3, 0] * 4}
    fit = r.survfit(r.coxph("Surv(time,status) ~ x", data), newdata={"x": [0, 2]})
    result = plotting.plot_survfit(fit, conf_int=True, conf_style="band")
    assert len(result.lines) == len(result.confidence) == 2
    for file_format in ("svg", "png", "pdf"):
        stream = io.BytesIO()
        result.axes.figure.savefig(stream, format=file_format)
        assert stream.tell() > 1000


def test_rgb_colors_custom_labels_and_confidence_only_legends():
    result = plotting.plot_survfit(reference_fit("km"), colors=(0.2, 0.4, 0.6), label="Study")
    assert result.lines[0].get_label() == "Study"
    assert result.lines[0].get_color() == (0.2, 0.4, 0.6)
    bars = plotting.plot_survfit(
        reference_fit("groups"), conf_int="only", conf_times=[2, 4], conf_offset=0, conf_cap=0
    )
    assert not bars.lines
    assert len(bars.confidence) == 2
    assert len(bars.axes.get_legend().get_texts()) == 2
    assert bars.axes.get_xlim()[1] >= 8


def test_grouped_confidence_offsets_use_one_shared_axis_range():
    result = plotting.plot_survfit(
        reference_fit("groups"), conf_times=[2], conf_offset=0.02, conf_cap=0, xlim=(0, 8)
    )
    positions = [artist.get_segments()[0][0, 0] for artist in result.confidence]
    np.testing.assert_allclose(positions, [2 - 0.08, 2 + 0.08])


def test_multistate_event_points_and_log_probabilities():
    fit = reference_fit("aj")
    result = plotting.points_survfit(fit)
    for line in result.lines:
        np.testing.assert_array_equal(line.get_xdata(), [1, 2, 3, 5, 6])
    log_plot = plotting.plot_survfit(fit, fun="log")
    assert log_plot.axes.get_yscale() == "linear"
    log_plot.axes.figure.canvas.draw()


def test_constant_confidence_bands_render_without_one_vertex_per_observation():
    fit = r.survfit(r.Surv(np.arange(1, 10001), np.zeros(10000, dtype=int)))
    result = plotting.plot_survfit(fit, conf_style="band")
    assert len(result.lines[0].get_xdata()) == 2
    assert len(result.confidence[0].get_paths()[0].vertices) < 20


def test_missing_optional_renderer_has_actionable_message(monkeypatch):
    original = builtins.__import__

    def without_matplotlib(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_matplotlib)
    assert plotting.survfit_plot_data(reference_fit("km")).curves
    with pytest.raises(ImportError, match=r"pip install survival\[plot\]"):
        plotting.plot_survfit(reference_fit("km"))


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"xscale": 0}, "positive"),
        ({"yscale": float("nan")}, "positive"),
        ({"xlim": [1, 0]}, "increasing"),
        ({"ylim": [0, 1, 2]}, "increasing"),
        ({"conf_style": "unknown"}, "conf_style"),
        ({"conf_times": [[1, 2]]}, "finite vector"),
        ({"conf_cap": -1}, "nonnegative"),
        ({"conf_offset": []}, "finite values"),
        ({"colors": []}, "empty"),
    ],
)
def test_invalid_plot_options(options, message):
    with pytest.raises(ValueError, match=message):
        plotting.plot_survfit(reference_fit("km"), **options)
