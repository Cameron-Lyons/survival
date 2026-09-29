"""Numerical preparation for proportional-hazards diagnostic graphics."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from operator import index

import numpy as np

from . import _survival as _core
from ._plot_data import FloatArray
from ._plot_helpers import variables
from .r._types import CoxZPHResult


@dataclass(frozen=True)
class CoxDiagnosticCurve:
    """One coefficient's smoothed curve, bands and scaled residuals."""

    name: str
    x: FloatArray
    estimate: FloatArray
    lower: FloatArray | None
    upper: FloatArray | None
    residual_x: FloatArray
    residual_y: FloatArray


@dataclass(frozen=True)
class CoxDiagnosticData:
    """Plot coordinates, scales and time labels; arrays are independent of the fit."""

    curves: tuple[CoxDiagnosticCurve, ...]
    xlog: bool
    ylog: bool
    tick_positions: FloatArray | None
    tick_labels: tuple[str, ...] | None
    skipped: tuple[str, ...]


def _interpolate(x: FloatArray, y: FloatArray, query: FloatArray) -> FloatArray:
    """R approx: sorted abscissae, mean at ties, NA outside the range."""

    unique, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    averaged = np.bincount(inverse, weights=y) / counts
    return np.interp(query, unique, averaged, left=np.nan, right=np.nan)


def cox_zph_plot_data(
    result: CoxZPHResult,
    *,
    df: int = 4,
    nsmo: int = 40,
    var: str | int | Sequence[str | int] | None = None,
    se: bool = True,
    hr: bool = False,
) -> CoxDiagnosticData:
    """Prepare R's ``plot.cox.zph`` curves using the Rust natural-spline kernel.

    ``var`` selects names or one-based columns. ``df`` controls the spline basis;
    knots use both event times and the ``nsmo`` prediction times, as in R.
    Bands use two standard errors. ``hr=True`` exponentiates coefficients and
    residuals and selects a log y axis. Singular terms warn and are skipped.
    This function does not import Matplotlib.
    """

    if not isinstance(result, CoxZPHResult):
        raise TypeError("result must be returned by survival.r.cox_zph")
    if not isinstance(hr, bool | np.bool_) or not isinstance(se, bool | np.bool_):
        raise TypeError("hr and se must be boolean")
    df, nsmo = index(df), index(nsmo)
    if df < 2 or nsmo < 2:
        raise ValueError("df and nsmo must both be at least 2")
    selected = variables(var, result.names)
    x, time = np.asarray(result.x, dtype=float), np.asarray(result.time, dtype=float)
    y, variance = np.asarray(result.y, dtype=float), np.asarray(result.var, dtype=float)
    if (
        y.ndim != 2
        or y.shape != (len(x), len(result.names))
        or variance.shape != (len(result.names), len(result.names))
        or time.shape != x.shape
        or not np.isfinite(time).all()
    ):
        raise ValueError("diagnostic times, residuals, names and variance have inconsistent shapes")
    smoothed = _core.cox_zph_smooth(
        x,
        y[:, selected],
        np.diag(variance)[selected],
        df=df,
        nsmo=nsmo,
        se=bool(se),
    )
    grid = np.asarray(smoothed.x, dtype=float)
    smooth = np.asarray(smoothed.y, dtype=float)
    errors = None if smoothed.std_err is None else np.asarray(smoothed.std_err, dtype=float)
    ticks, labels = None, None
    if result.transform == "log":
        x, grid = np.exp(x), np.exp(grid)
    elif result.transform != "identity":
        # R first keeps the first time at each transformed value, then labels
        # eight evenly spaced positions with rounded original event times.
        _, first = np.unique(x, return_index=True)
        positions = np.linspace(x.min(), x.max(), 17)[1::2]
        times = _interpolate(x[first], time[first], positions)
        rounded = np.asarray([float(f"{value:.2g}") for value in times])
        ticks = _interpolate(time[first], x[first], rounded)
        labels = tuple(f"{value:g}" for value in rounded)
    curves = []
    skipped = []
    for col, original in enumerate(selected):
        name = result.names[original]
        if col in smoothed.skipped:
            warnings.warn(
                f"spline fit is singular, variable {name} skipped", RuntimeWarning, stacklevel=2
            )
            skipped.append(name)
            continue
        keep = ~np.isnan(y[:, original])
        estimate, residual = smooth[:, col].copy(), y[keep, original].copy()
        lower = None if errors is None else estimate - 2 * errors[:, col]
        upper = None if errors is None else estimate + 2 * errors[:, col]
        if hr:
            with np.errstate(over="ignore", under="ignore"):
                estimate, residual = np.exp(estimate), np.exp(residual)
                if lower is not None and upper is not None:
                    lower, upper = np.exp(lower), np.exp(upper)
        curves.append(
            CoxDiagnosticCurve(name, grid.copy(), estimate, lower, upper, x[keep].copy(), residual)
        )
    return CoxDiagnosticData(
        tuple(curves), result.transform == "log", bool(hr), ticks, labels, tuple(skipped)
    )
