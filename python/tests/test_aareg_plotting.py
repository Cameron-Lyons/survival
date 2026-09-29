"""Aalen renderer checks, including multi-panel plots and overlays."""

import numpy as np
import pytest

from .test_aareg_plot_data import REFERENCE, options_for, plotting, reference_fit

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_aalen_artists_match_r(case):
    function = plotting.lines_aareg if case["method"] == "lines" else plotting.plot_aareg
    options = options_for(case)
    drawstyle = "default" if case["options"].get("type") == "l" else "steps-post"
    result = function(reference_fit(case["fit"]), drawstyle=drawstyle, **options)
    for col, line in enumerate(result.lines):
        np.testing.assert_array_equal(line.get_xdata(), result.data.time)
        np.testing.assert_array_equal(line.get_ydata(), result.data.coefficient[:, col])
        assert line.get_drawstyle() == drawstyle
    for col in range(len(result.data.names)):
        if options.get("se", True):
            np.testing.assert_array_equal(
                result.confidence[2 * col].get_ydata(), result.data.upper[:, col]
            )
            np.testing.assert_array_equal(
                result.confidence[2 * col + 1].get_ydata(), result.data.lower[:, col]
            )
    for axis in result.axes:
        axis.figure.canvas.draw()


def test_existing_axes_labels_selection_and_exports(tmp_path):
    fig, axes = plt.subplots(1, 2)
    result = plotting.plot_aareg(
        reference_fit("robust"),
        var=["age", "ecog.ps"],
        ax=axes,
        colors=["navy", "orange"],
        xlabel="Days",
        alpha=0.7,
    )
    assert result.axes == tuple(axes)
    assert axes[0].get_ylabel() == "age"
    assert axes[1].get_ylabel() == "ecog.ps"
    assert result.lines[0].get_color() == "navy"
    assert result.confidence[2].get_color() == "orange"
    for extension in ("svg", "png", "pdf"):
        path = tmp_path / f"aalen.{extension}"
        fig.savefig(path)
        assert path.stat().st_size > 1000


def test_overlay_keeps_limits_scales_and_adds_all_selected_curves():
    axis = plt.subplots()[1]
    axis.plot([10, 1000], [1, 2])
    axis.set_xscale("log")
    before = axis.get_xlim(), axis.get_ylim()
    result = plotting.lines_aareg(reference_fit("ovarian"), ax=axis, var=2, se=True)
    assert result.axes == (axis,)
    assert axis.get_xscale() == "log"
    assert (axis.get_xlim(), axis.get_ylim()) == before
    assert len(result.lines) == 1
    assert len(result.confidence) == 2


def test_no_bands_share_one_panel_with_named_legend():
    result = plotting.plot_aareg(reference_fit("ovarian"), se=False)
    assert len(result.axes) == 1
    assert len(result.lines) == 3
    assert [t.get_text() for t in result.axes[0].get_legend().get_texts()] == list(
        result.data.names
    )
    assert result.confidence == ()
    with pytest.raises(ValueError, match="one axis"):
        plotting.plot_aareg(reference_fit("ovarian"), ax=result.axes[0])
