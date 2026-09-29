"""Cumulative Aalen coefficients and optional Matplotlib rendering."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._plot_data import FloatArray
from ._plot_helpers import axes as get_axes
from ._plot_helpers import panel_axes, styles, variables
from .r._types import AaregModelResult


@dataclass(frozen=True)
class AalenPlotData:
    """Cumulative coefficients on unique event times, with R's zero origin.

    Matrices have one row per time and one column per selected coefficient.
    ``std_err`` uses stored influences when available, otherwise coefficient
    increments. ``lower``/``upper`` are pointwise estimate ± 1.96 standard errors.
    """

    time: FloatArray
    coefficient: FloatArray
    std_err: FloatArray | None
    lower: FloatArray | None
    upper: FloatArray | None
    names: tuple[str, ...]
    robust: bool


def _influence_variance(
    values: Any, width: int, times: int, keep: int, selected: list[int]
) -> FloatArray:
    """Reduce groups in bounded blocks, without materializing or squaring a cube."""

    variance = np.zeros((keep, len(selected)))
    if keep == 0:
        return variance
    if isinstance(values, np.ndarray):
        if values.ndim != 3 or values.shape[1:] != (width, times):
            raise ValueError("dfbeta must be group by coefficient by unique event time")
        # Basic slicing preserves the original cube. Bound the finite-value
        # check too; otherwise even a boolean group-by-time array can be large.
        block_size = max(1, 131072 // keep)
        for col, source in enumerate(selected):
            for start in range(0, len(values), block_size):
                part = np.asarray(values[start : start + block_size, source, :keep], dtype=float)
                if not np.isfinite(part).all():
                    raise ValueError("dfbeta must contain finite values")
                variance[:, col] += np.einsum("gt,gt->t", part, part)
        return variance
    # About 1 MiB of floating-point input per block. Lists already hold the
    # fitted influences, so converting the entire cube would duplicate it.
    block_size = max(1, 131072 // max(1, keep * len(selected)))
    for start in range(0, len(values), block_size):
        block = []
        for group in values[start : start + block_size]:
            if len(group) != width or any(len(column) != times for column in group):
                raise ValueError("dfbeta must be group by coefficient by unique event time")
            block.append([group[col] if keep == times else group[col][:keep] for col in selected])
        array = np.asarray(block, dtype=float)
        if not np.isfinite(array).all():
            raise ValueError("dfbeta must contain finite values")
        variance += np.einsum("gpt,gpt->tp", array, array)
    return variance


def aareg_plot_data(
    fit: AaregModelResult,
    *,
    se: bool = True,
    maxtime: float | None = None,
    var: str | int | Sequence[str | int] | None = None,
) -> AalenPlotData:
    """Prepare R's Aalen cumulative-coefficient curves without a renderer.

    Tied events contribute their individual increments before retaining the
    final value at each time. Stored per-group influences determine standard
    errors when present. ``maxtime`` includes all events at the cutoff and does
    not extend the curve to it. ``var`` selects names or one-based indices.
    """

    if not isinstance(fit, AaregModelResult):
        raise TypeError("fit must be returned by survival.r.aareg")
    if not isinstance(se, bool | np.bool_):
        raise TypeError("se must be boolean")
    selected = variables(var, fit.coefficient_names)
    time = np.asarray(fit.times, dtype=float)
    coefficient = np.asarray(fit.coefficient, dtype=float)
    width = len(fit.coefficient_names)
    if (
        time.ndim != 1
        or not len(time)
        or not np.isfinite(time).all()
        or np.any(time[1:] < time[:-1])
    ):
        raise ValueError("fitted times must be finite and ordered")
    if coefficient.shape != (len(time), width) or not np.isfinite(coefficient).all() or not width:
        raise ValueError("coefficient must have a finite value for each time and name")
    if maxtime is not None and not np.isfinite(maxtime):
        raise ValueError("maxtime must be finite")
    count = len(time) if maxtime is None else int(np.searchsorted(time, maxtime, side="right"))
    all_ends = np.flatnonzero(np.r_[time[1:] != time[:-1], True])
    ends = all_ends[all_ends < count]
    increments = coefficient[:count, selected]
    estimate = np.vstack((np.zeros(len(selected)), np.cumsum(increments, axis=0)[ends]))
    error = lower = upper = None
    robust = se and fit.dfbeta is not None
    if se:
        if robust:
            variances = _influence_variance(fit.dfbeta, width, len(all_ends), len(ends), selected)
            cumulative = np.cumsum(variances, axis=0)
        else:
            cumulative = np.cumsum(increments * increments, axis=0)[ends]
        error = np.vstack((np.zeros(len(selected)), np.sqrt(cumulative)))
        lower, upper = estimate - 1.96 * error, estimate + 1.96 * error
    times = np.r_[0.0, time[ends]]
    if np.any(time[ends] == 0):
        # Keep the last value at zero, including its event, for every matrix.
        times, estimate = times[1:], estimate[1:]
        if error is not None and lower is not None and upper is not None:
            error, lower, upper = error[1:], lower[1:], upper[1:]
    return AalenPlotData(
        times,
        estimate,
        error,
        lower,
        upper,
        tuple(fit.coefficient_names[i] for i in selected),
        bool(robust),
    )


@dataclass(frozen=True)
class AalenPlot:
    """Rendered cumulative coefficient curves and their prepared data."""

    axes: tuple[Any, ...]
    data: AalenPlotData
    lines: tuple[Any, ...]
    confidence: tuple[Any, ...]


def _draw_aareg(
    fit: AaregModelResult,
    *,
    ax: Any = None,
    overlay: bool = False,
    se: bool = True,
    maxtime: float | None = None,
    var: str | int | Sequence[str | int] | None = None,
    colors: Any = None,
    linewidth: float = 1.5,
    drawstyle: str = "steps-post",
    legend: bool = True,
    xlabel: str = "Time",
    ylabel: str | None = None,
    **line_kwargs: Any,
) -> AalenPlot:
    data = aareg_plot_data(fit, se=se, maxtime=maxtime, var=var)
    axes = (get_axes(ax, True),) if overlay else panel_axes(ax, len(data.names) if se else 1)
    from matplotlib.colors import is_color_like

    palette = [colors] if colors is not None and is_color_like(colors) else styles(colors, None)
    old_limits = (
        (axes[0].get_xlim(), axes[0].get_ylim()) if overlay and axes[0].has_data() else None
    )
    lines, confidence = [], []
    for col, name in enumerate(data.names):
        axis = axes[col] if se and not overlay else axes[0]
        style = {
            "color": palette[col % len(palette)],
            "linewidth": linewidth,
            "drawstyle": drawstyle,
            **line_kwargs,
        }
        label = style.pop("label", name)
        line = axis.plot(data.time, data.coefficient[:, col], label=label, **style)[0]
        lines.append(line)
        if data.lower is not None and data.upper is not None:
            for bound in (data.upper, data.lower):
                confidence.append(
                    axis.plot(
                        data.time,
                        bound[:, col],
                        **{
                            **style,
                            "color": line.get_color(),
                            "linestyle": "--",
                            "label": "_nolegend_",
                        },
                    )[0]
                )
        if not overlay:
            axis.set_xlabel(xlabel)
            axis.set_ylabel(
                (name if se else "Cumulative coefficient") if ylabel is None else ylabel
            )
    if old_limits is not None:
        axes[0].set_xlim(old_limits[0])
        axes[0].set_ylim(old_limits[1])
    if legend and len(data.names) > 1 and (overlay or not se):
        axes[0].legend()
    return AalenPlot(axes, data, tuple(lines), tuple(confidence))


def plot_aareg(fit: AaregModelResult, *, ax: Any = None, **kwargs: Any) -> AalenPlot:
    """Plot cumulative Aalen coefficients, with pointwise 1.96-SE bands by default.

    With ``se=True`` each coefficient has its own panel; without bands, all
    coefficients share one panel, as in R. ``var`` selects names or one-based
    indices; ``maxtime`` truncates to fitted event times. Pass existing axes
    through ``ax``. ``drawstyle`` defaults to ``"steps-post"``; other line
    properties (including markers) pass to Matplotlib. The result contains
    axes, artists and numerical data; the function never calls ``show()``.
    """

    return _draw_aareg(fit, ax=ax, **kwargs)


def lines_aareg(
    fit: AaregModelResult, *, ax: Any = None, se: bool = False, **kwargs: Any
) -> AalenPlot:
    """Add Aalen coefficient curves to one axis, preserving its scales and limits.

    Accepts the same options as ``plot_aareg``; confidence bands are off by
    default, as in R's ``lines.aareg``.
    """

    return _draw_aareg(fit, ax=ax, overlay=True, se=se, **kwargs)
