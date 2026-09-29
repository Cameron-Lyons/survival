"""Survival curves and model diagnostics with optional Matplotlib rendering.

Install ``survival[plot]`` to render. Numerical data helpers support other
graphics libraries without importing Matplotlib or creating a figure.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._aalen_plot import AalenPlot, AalenPlotData, aareg_plot_data, lines_aareg, plot_aareg
from ._cox_plot_data import CoxDiagnosticCurve, CoxDiagnosticData, cox_zph_plot_data
from ._plot_data import (
    SurvivalCurve,
    SurvivalFit,
    SurvivalPlotData,
    step_at,
    survfit_plot_data,
)
from ._plot_helpers import axes as _axes
from ._plot_helpers import panel_axes
from ._plot_helpers import styles as _styles
from .r._surv import Surv, Surv2
from .r._types import AaregModelResult, CoxZPHResult, SurvExpResult

__all__ = [
    "SurvivalCurve",
    "SurvivalPlotData",
    "SurvivalPlot",
    "survfit_plot_data",
    "plot_survfit",
    "lines_survfit",
    "points_survfit",
    "CoxDiagnosticCurve",
    "CoxDiagnosticData",
    "CoxDiagnosticPlot",
    "cox_zph_plot_data",
    "plot_cox_zph",
    "AalenPlot",
    "AalenPlotData",
    "aareg_plot_data",
    "plot_aareg",
    "lines_aareg",
    "plot_surv",
    "lines_survexp",
    "plot",
    "lines",
    "points",
]


@dataclass(frozen=True)
class SurvivalPlot:
    """Rendered curves, their axes, and the numerical data used to draw them.

    ``lines``, ``confidence`` and ``marks`` contain Matplotlib artists that can
    be styled or removed. Export with ``result.axes.figure.savefig(...)``.
    """

    axes: Any
    data: SurvivalPlotData
    lines: tuple[Any, ...]
    confidence: tuple[Any, ...]
    marks: tuple[Any, ...]


def _limits(values: Any, name: str) -> tuple[float, float] | None:
    if values is None:
        return None
    result = np.asarray(values, dtype=float)
    if result.shape != (2,) or not np.isfinite(result).all() or result[0] >= result[1]:
        raise ValueError(f"{name} must contain two finite increasing values")
    return float(result[0]), float(result[1])


def _plot_extent(data: SurvivalPlotData, *, time: bool, logarithmic: bool) -> tuple[float, float]:
    limits: list[float] = []
    for curve in data.curves:
        arrays = (curve.time,) if time else (curve.estimate, curve.lower, curve.upper)
        for values in arrays:
            if values is not None:
                keep = np.isfinite(values)
                if logarithmic:
                    keep &= values > 0
                if np.any(keep):
                    limits.extend((values[keep].min(), values[keep].max()))
    if not limits:
        raise ValueError("no finite coordinates available for the requested axis scale")
    return min(limits), max(limits)


def _draw(
    fit: SurvivalFit,
    *,
    ax: Any = None,
    overlay: bool = False,
    points: bool = False,
    censor: bool = False,
    conf_int: Any = None,
    conf_type: str | None = None,
    mark_time: Any = False,
    fun: Any = None,
    cumhaz: Any = False,
    cumprob: Any = False,
    noplot: Any = "(s0)",
    log: bool | str = False,
    xscale: float = 1,
    yscale: float = 1,
    xlim: Any = None,
    ylim: Any = None,
    xmax: float | None = None,
    conf_times: Any = None,
    conf_cap: float = 0.005,
    conf_offset: Any = 0.012,
    conf_style: str = "lines",
    colors: Any = None,
    linestyles: Any = None,
    linewidth: float = 1.5,
    marker: str = "+",
    markersize: float = 6,
    legend: bool = True,
    xlabel: str = "Time",
    ylabel: str | None = None,
    drawstyle: str = "steps-post",
    **line_kwargs: Any,
) -> SurvivalPlot:
    if drawstyle not in {"default", "steps-post", "steps-pre", "steps-mid"}:
        raise ValueError("drawstyle must be default, steps-post, steps-pre, or steps-mid")
    if not np.isfinite([xscale, yscale]).all() or xscale <= 0 or yscale <= 0:
        raise ValueError("xscale and yscale must be finite and positive")
    xlim, ylim = _limits(xlim, "xlim"), _limits(ylim, "ylim")
    if xlim is not None:
        xmax = xlim[1]
    if conf_style not in {"lines", "band"}:
        raise ValueError("conf_style must be 'lines' or 'band'")
    if conf_times is not None:
        conf_times = np.atleast_1d(np.asarray(conf_times, dtype=float))
        if conf_times.ndim != 1 or not np.isfinite(conf_times).all():
            raise ValueError("conf_times must be a finite vector")
        if conf_int is None:
            conf_int = True
    if not np.isfinite(conf_cap) or conf_cap < 0:
        raise ValueError("conf_cap must be finite and nonnegative")
    offsets = np.atleast_1d(np.asarray(conf_offset, dtype=float))
    if offsets.ndim != 1 or not len(offsets) or not np.isfinite(offsets).all():
        raise ValueError("conf_offset must contain finite values")
    data = survfit_plot_data(
        fit,
        conf_int=False if points else conf_int,
        conf_type=conf_type,
        mark_time=mark_time,
        fun=fun,
        cumhaz=cumhaz,
        cumprob=cumprob,
        noplot=noplot,
        log=log,
        xmax=xmax,
    )
    styles = _styles(linestyles, "-")
    ax = _axes(ax, overlay)
    from matplotlib.colors import is_color_like

    colors = [colors] if colors is not None and is_color_like(colors) else _styles(colors, None)
    old_limits = (ax.get_xlim(), ax.get_ylim()) if overlay and ax.has_data() else None
    if not overlay:
        ax.set_xscale("log" if data.xlog else "linear")
        ax.set_yscale("log" if data.ylog else "linear")
        domain = np.asarray(_plot_extent(data, time=True, logarithmic=data.xlog)) / xscale
        values = np.asarray(_plot_extent(data, time=False, logarithmic=data.ylog)) * yscale
        ax.update_datalim(np.column_stack((domain, values)))
        ax.autoscale_view()
    span = (xlim[1] - xlim[0]) if xlim else np.ptp(ax.get_xlim()) * xscale
    lines, confidence, marks = [], [], []
    for i, curve in enumerate(data.curves):
        style = {
            "color": colors[i % len(colors)],
            "linewidth": linewidth,
            "linestyle": styles[i % len(styles)],
            "drawstyle": "default" if drawstyle == "steps-post" else drawstyle,
            **line_kwargs,
        }
        label = style.pop("label", curve.label)
        if not data.plot_estimate and conf_times is not None and style["color"] is None:
            style["color"] = f"C{i}"
        if points:
            keep = np.ones(len(curve.time), dtype=bool) if censor else curve.event
            artist = ax.plot(
                curve.time[keep] / xscale,
                curve.estimate[keep] * yscale,
                **{
                    **style,
                    "linestyle": "none",
                    "marker": marker,
                    "markersize": markersize,
                    "label": label,
                },
            )[0]
            lines.append(artist)
            continue
        if data.plot_estimate:
            xx, yy = curve.step() if drawstyle == "steps-post" else (curve.time, curve.estimate)
            artist = ax.plot(xx / xscale, yy * yscale, label=label, **style)[0]
            lines.append(artist)
            style["color"] = artist.get_color()
        if curve.lower is not None and curve.upper is not None and conf_times is None:
            if conf_style == "band":
                keep = np.r_[
                    True,
                    (curve.lower[1:] != curve.lower[:-1]) | (curve.upper[1:] != curve.upper[:-1]),
                ]
                keep[-1] = True
                if drawstyle != "steps-post":
                    keep[:] = True
                artist = ax.fill_between(
                    curve.time[keep] / xscale,
                    curve.lower[keep] * yscale,
                    curve.upper[keep] * yscale,
                    step=drawstyle.removeprefix("steps-") if drawstyle != "default" else None,
                    color=style["color"],
                    alpha=0.2,
                    label="_nolegend_" if data.plot_estimate else label,
                )
                confidence.append(artist)
                if style["color"] is None:
                    style["color"] = artist.get_facecolor()[0]
            else:
                for bound in (curve.lower, curve.upper):
                    xx, yy = curve.step(bound) if drawstyle == "steps-post" else (curve.time, bound)
                    artist = ax.plot(
                        xx / xscale,
                        yy * yscale,
                        **{
                            **style,
                            "linestyle": "--",
                            "label": "_nolegend_" if data.plot_estimate else label,
                        },
                    )[0]
                    confidence.append(artist)
                    style["color"] = artist.get_color()
                    label = "_nolegend_"
        if data.plot_estimate and len(curve.censor_time):
            marks.append(
                ax.plot(
                    curve.censor_time / xscale,
                    curve.censor_value * yscale,
                    color=style["color"],
                    linestyle="none",
                    marker=marker,
                    markersize=markersize,
                    label="_nolegend_",
                )[0]
            )
        if curve.lower is not None and curve.upper is not None and conf_times is not None:
            # Use the current device's range, as R does, and f=1 interpolation
            # for interval bars (censor marks use f=0 instead).
            offset = (
                (i - (len(data.curves) - 1) / 2) * offsets[0]
                if len(offsets) == 1
                else offsets[i % len(offsets)]
            )
            query = conf_times + offset * span
            lo = step_at(curve.time, curve.lower, query, right=False)
            hi = step_at(curve.time, curve.upper, query, right=False)
            artist = ax.vlines(
                query / xscale,
                lo * yscale,
                hi * yscale,
                colors=style["color"],
                linewidths=linewidth,
                label="_nolegend_" if data.plot_estimate else label,
            )
            confidence.append(artist)
            if conf_cap:
                for bound in (lo, hi):
                    confidence.append(
                        ax.hlines(
                            bound * yscale,
                            (query - conf_cap * span) / xscale,
                            (query + conf_cap * span) / xscale,
                            colors=style["color"],
                            linewidths=linewidth,
                        )
                    )
    if old_limits is not None:
        ax.set_xlim(old_limits[0])
        ax.set_ylim(old_limits[1])
    if not overlay:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(data.ylabel if ylabel is None else ylabel)
        if xlim is not None:
            ax.set_xlim(np.asarray(xlim) / xscale)
        if ylim is not None:
            ax.set_ylim(np.asarray(ylim) * yscale)
        elif data.ylabel == "Survival probability" and not data.ylog:
            ax.set_ylim(bottom=0)
    if legend and len(data.curves) > 1:
        ax.legend()
    return SurvivalPlot(ax, data, tuple(lines), tuple(confidence), tuple(marks))


def plot_survfit(fit: SurvivalFit, *, ax: Any = None, **kwargs: Any) -> SurvivalPlot:
    """Plot fitted survival, event, cumulative-hazard or multistate curves.

    ``conf_int`` defaults to bands for a single curve; use ``True`` to include
    them for grouped curves, a numeric level to recompute them, or ``"only"``.
    ``conf_style="band"`` shades intervals; the default draws dashed lines.
    ``conf_times`` draws interval bars at selected times instead of full bands.
    ``mark_time=True`` adds censor marks, or supply a vector of marker times.
    ``fun`` accepts R's transformations (``event``, ``cumhaz``, ``cloglog``,
    ``pct``, ``log``, ``logpct``, ``identity``) or a NumPy-compatible callable.
    ``cumprob`` accumulates selected multistate probabilities; ``noplot`` hides
    the initial ``(s0)`` state by default. See ``survfit_plot_data`` for ordering.

    ``xscale`` divides displayed times and ``yscale`` multiplies displayed values.
    Limits and ``xmax`` are in unscaled coordinates. ``colors`` and ``linestyles``
    recycle over curves. Remaining keywords go to Matplotlib's ``Axes.plot``.
    The function never calls ``show`` or changes Matplotlib's backend.
    """

    return _draw(fit, ax=ax, **kwargs)


def plot_surv(response: Surv, *, ax: Any = None, **kwargs: Any) -> SurvivalPlot:
    """Fit one ungrouped curve to a raw ``Surv`` response and plot it, as R does.

    Plot options are passed to ``plot_survfit``; fitting uses ``survfit`` defaults.
    To change fitting options, fit explicitly before plotting.
    """

    if isinstance(response, Surv2):
        raise ValueError("method not defined for a Surv2 object")
    if not isinstance(response, Surv):
        raise TypeError("response must be a Surv object")
    from .r._survfit import survfit

    return plot_survfit(survfit(response), ax=ax, **kwargs)


def lines_survexp(fit: SurvExpResult, *, ax: Any = None, **kwargs: Any) -> SurvivalPlot:
    """Overlay expected-survival curves, joining their time grid with straight lines.

    Accepts the options of ``lines_survfit``; override ``drawstyle`` to draw steps.
    No confidence intervals or observed censor counts are inferred.
    """

    if not isinstance(fit, SurvExpResult):
        raise TypeError("fit must be returned by survival.r.survexp")
    kwargs.setdefault("drawstyle", "default")
    return lines_survfit(fit, ax=ax, **kwargs)


def plot(
    value: SurvivalFit | Surv | Surv2 | AaregModelResult | CoxZPHResult,
    *,
    ax: Any = None,
    **kwargs: Any,
) -> SurvivalPlot | AalenPlot | CoxDiagnosticPlot:
    """Dispatch a response, survival curve, or model diagnostic to its plot method."""

    if isinstance(value, Surv2):
        raise ValueError("method not defined for a Surv2 object")
    if isinstance(value, Surv):
        return plot_surv(value, ax=ax, **kwargs)
    if isinstance(value, AaregModelResult):
        return plot_aareg(value, ax=ax, **kwargs)
    if isinstance(value, CoxZPHResult):
        return plot_cox_zph(value, ax=ax, **kwargs)
    return plot_survfit(value, ax=ax, **kwargs)


def lines(
    value: SurvivalFit | Surv | Surv2 | AaregModelResult,
    *,
    ax: Any = None,
    **kwargs: Any,
) -> SurvivalPlot | AalenPlot:
    """Add fitted curves to existing axes, using each model's default line method."""

    if isinstance(value, Surv | Surv2):
        raise ValueError(f"method not defined for a {type(value).__name__} object")
    if isinstance(value, AaregModelResult):
        return lines_aareg(value, ax=ax, **kwargs)
    if isinstance(value, SurvExpResult):
        return lines_survexp(value, ax=ax, **kwargs)
    return lines_survfit(value, ax=ax, **kwargs)


def points(
    value: SurvivalFit | Surv | Surv2,
    *,
    ax: Any = None,
    **kwargs: Any,
) -> SurvivalPlot:
    """Add fitted event-time estimates; raw response point methods are undefined."""

    if isinstance(value, Surv | Surv2):
        raise ValueError(f"method not defined for a {type(value).__name__} object")
    return points_survfit(value, ax=ax, **kwargs)


def lines_survfit(
    fit: SurvivalFit, *, ax: Any = None, conf_int: Any = False, **kwargs: Any
) -> SurvivalPlot:
    """Add survival curves to existing axes, preserving their limits and scales.

    Accepts the same options as ``plot_survfit``. Confidence intervals are off
    by default, as in R's ``lines.survfit``. Repeat axis-unit scalings when adding
    curves to a plot made with ``xscale`` or ``yscale``.
    """

    return _draw(fit, ax=ax, overlay=True, conf_int=conf_int, **kwargs)


def points_survfit(
    fit: SurvivalFit, *, ax: Any = None, censor: bool = False, marker: str = "o", **kwargs: Any
) -> SurvivalPlot:
    """Add event-time estimates as points; ``censor=True`` includes every time.

    Supports the transformations, state/transition selections and styles of
    ``plot_survfit``. Existing axis limits and scales are preserved.
    """

    return _draw(fit, ax=ax, overlay=True, points=True, censor=censor, marker=marker, **kwargs)


@dataclass(frozen=True)
class CoxDiagnosticPlot:
    """One subplot per selected Cox diagnostic term and its numerical data."""

    axes: tuple[Any, ...]
    data: CoxDiagnosticData
    lines: tuple[Any, ...]
    confidence: tuple[Any, ...]
    residuals: tuple[Any, ...]


def plot_cox_zph(
    result: CoxZPHResult,
    *,
    ax: Any = None,
    resid: bool = True,
    se: bool = True,
    df: int = 4,
    nsmo: int = 40,
    var: str | int | Sequence[str | int] | None = None,
    hr: bool = False,
    colors: Any = None,
    linestyles: Any = ("-", "--"),
    linewidth: float = 1.5,
    marker: str = "o",
    markersize: float = 4,
    xlabel: str = "Time",
    ylabel: str | None = None,
    **line_kwargs: Any,
) -> CoxDiagnosticPlot:
    """Plot R's Cox proportional-hazards diagnostics, one subplot per term.

    ``var`` selects term names or one-based indices. ``df`` controls the natural
    spline and ``nsmo`` the prediction grid. Bands use two standard errors.
    ``hr=True`` displays hazard ratios on a log y axis. ``resid=False`` hides
    scaled Schoenfeld residuals. Pass one axis per selected nonsingular term
    through ``ax``; otherwise new subplots are created. ``colors`` and
    ``linestyles`` specify the estimate and confidence styles, recycling if
    needed. Remaining line properties apply to the smoothed curve and bands.
    Export via ``result.axes[0].figure.savefig(...)``.
    """

    if not isinstance(resid, bool | np.bool_):
        raise TypeError("resid must be boolean")
    data = cox_zph_plot_data(result, df=df, nsmo=nsmo, var=var, se=se, hr=hr)
    if not data.curves:
        return CoxDiagnosticPlot((), data, (), (), ())
    axes = panel_axes(ax, len(data.curves))
    from matplotlib.colors import is_color_like

    colors = [colors] if colors is not None and is_color_like(colors) else _styles(colors, "black")
    styles = _styles(linestyles, "-")
    lines, confidence, residuals = [], [], []
    for axis, curve in zip(axes, data.curves, strict=True):
        axis.set_xscale("log" if data.xlog else "linear")
        axis.set_yscale("log" if data.ylog else "linear")
        axis.set_xlabel(xlabel)
        axis.set_ylabel(
            f"{'HR' if hr else 'Beta'}(t) for {curve.name}" if ylabel is None else ylabel
        )
        if data.tick_positions is not None and data.tick_labels is not None:
            keep = np.isfinite(data.tick_positions)
            axis.set_xticks(
                data.tick_positions[keep],
                [label for label, ok in zip(data.tick_labels, keep, strict=True) if ok],
            )
        if resid:
            residuals.append(
                axis.plot(
                    curve.residual_x,
                    curve.residual_y,
                    linestyle="none",
                    marker=marker,
                    markersize=markersize,
                    markerfacecolor="none",
                    color=colors[0],
                )[0]
            )
        style = {"linewidth": linewidth, **line_kwargs}
        lines.append(
            axis.plot(curve.x, curve.estimate, color=colors[0], linestyle=styles[0], **style)[0]
        )
        for bound in (curve.upper, curve.lower):
            if bound is not None:
                confidence.append(
                    axis.plot(
                        curve.x,
                        bound,
                        color=colors[1 % len(colors)],
                        linestyle=styles[1 % len(styles)],
                        **style,
                    )[0]
                )
    return CoxDiagnosticPlot(axes, data, tuple(lines), tuple(confidence), tuple(residuals))
