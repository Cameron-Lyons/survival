"""Cox diagnostic artists and optional-renderer integration."""

import warnings

import numpy as np
import pytest

from .test_cox_zph_plot_data import REFERENCE, fitted, plotting

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_artists_match_r_coordinates(case):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = plotting.plot_cox_zph(fitted(case["fit"]), **case["options"])
    assert len(result.axes) == len(case["expected"]["plots"])
    for axis, expected in zip(result.axes, case["expected"]["plots"], strict=True):
        assert axis.get_xlabel() == "Time"
        assert axis.get_ylabel() == expected["ylab"]
        assert axis.get_xscale() == ("log" if "x" in expected["log"] else "linear")
        assert axis.get_yscale() == ("log" if "y" in expected["log"] else "linear")
        offset = int(case["options"].get("resid", True))
        for actual, line in zip(axis.lines[offset:], expected["lines"], strict=True):
            np.testing.assert_allclose(actual.get_xdata(), line["x"], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(actual.get_ydata(), line["y"], rtol=2e-9, atol=2e-11)
        axis.figure.canvas.draw()


def test_existing_axes_styles_and_exports(tmp_path):
    fig, ax = plt.subplots(1, 2)
    result = plotting.plot_cox_zph(
        fitted("km"),
        ax=ax,
        var=[2, 1],
        colors=["navy", "orange"],
        linestyles=["-", ":"],
        resid=False,
        xlabel="Follow-up",
        alpha=0.7,
    )
    assert result.axes == tuple(ax)
    assert not result.residuals
    assert result.lines[0].get_color() == "navy"
    assert result.confidence[0].get_color() == "orange"
    assert result.confidence[0].get_linestyle() == ":"
    assert ax[0].get_ylabel() == "Beta(t) for sex"
    assert ax[0].get_xlabel() == "Follow-up"
    for extension in ("png", "svg", "pdf"):
        path = tmp_path / f"diagnostic.{extension}"
        fig.savefig(path)
        assert path.stat().st_size > 1000


def test_one_selected_term_and_invalid_axes():
    axis = plt.subplots()[1]
    result = plotting.plot_cox_zph(
        fitted("km"), ax=axis, var="age", se=False, colors=(0.1, 0.3, 0.5), ylabel=""
    )
    assert result.axes == (axis,)
    assert result.confidence == ()
    assert axis.get_ylabel() == ""
    with pytest.raises(ValueError, match="one axis"):
        plotting.plot_cox_zph(fitted("km"), ax=axis)
    with pytest.raises(TypeError, match="boolean"):
        plotting.plot_cox_zph(fitted("km"), resid="yes")


def test_all_singular_terms_warn_without_creating_a_figure():
    with pytest.warns(RuntimeWarning, match="sex skipped"):
        result = plotting.plot_cox_zph(fitted("singular"), var="sex")
    assert result.axes == ()
    assert result.data.curves == ()
    assert result.data.skipped == ("sex",)
    assert not plt.get_fignums()
