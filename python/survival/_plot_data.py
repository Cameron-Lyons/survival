"""Numerical preparation for survival graphics, without importing a renderer."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from . import _survival as _core
from .r._types import (
    CoxSurvfitMultiStateResult,
    CoxSurvfitResult,
    SurvExpResult,
    SurvfitMultiStateResult,
    SurvfitResult,
)

FloatArray = NDArray[np.float64]
SurvivalFit = (
    SurvfitResult
    | CoxSurvfitResult
    | SurvfitMultiStateResult
    | CoxSurvfitMultiStateResult
    | SurvExpResult
)
Transform = str | Callable[[FloatArray], Any] | None
_MULTISTATE = (SurvfitMultiStateResult, CoxSurvfitMultiStateResult)


@dataclass(frozen=True)
class SurvivalCurve:
    """One plotted curve in original time units, after the requested transformation.

    ``lower`` and ``upper`` retain their original confidence-limit identities;
    a decreasing transformation such as ``event`` reverses their numeric order.
    Arrays belong to this result, never to the fitted model.
    """

    label: str
    time: FloatArray
    estimate: FloatArray
    lower: FloatArray | None
    upper: FloatArray | None
    censor_time: FloatArray
    censor_value: FloatArray
    event: NDArray[np.bool_]

    def step(self, values: FloatArray | None = None) -> tuple[FloatArray, FloatArray]:
        """Finite, right-continuous step coordinates with constant runs compressed."""

        y = self.estimate if values is None else values
        keep = np.isfinite(self.time) & np.isfinite(y)
        x, y = self.time[keep], y[keep]
        if len(x) < 2:
            return x, y
        if len(x) == 2:
            return x[[0, 1, 1]], y[[0, 0, 1]]
        drops = np.flatnonzero(y[1:] != y[:-1]) + 1
        indices = np.r_[0, np.repeat(drops, 2)]
        previous = np.r_[0, np.column_stack((drops - 1, drops)).ravel()]
        if not len(drops) or drops[-1] != len(x) - 1:
            indices = np.r_[indices, len(x) - 1]
            previous = np.r_[previous, len(x) - 1]
        return x[indices], y[previous]


@dataclass(frozen=True)
class SurvivalPlotData:
    """Curve coordinates shared by the plotting functions and custom renderers."""

    curves: tuple[SurvivalCurve, ...]
    xlog: bool
    ylog: bool
    ylabel: str
    plot_estimate: bool

    @property
    def xend(self) -> FloatArray:
        return np.asarray([curve.time[-1] for curve in self.curves])

    @property
    def yend(self) -> FloatArray:
        return np.asarray([curve.estimate[-1] for curve in self.curves])


def _array(values: Any, rows: int, name: str) -> FloatArray:
    if values is None:
        raise ValueError(f"{name} is missing from the fitted curves")
    result = np.asarray(values, dtype=np.float64)
    if result.ndim not in (1, 2, 3) or result.shape[0] != rows:
        raise ValueError(f"{name} must have one row per fitted time")
    return result.reshape((rows, -1), order="F") if rows else result.reshape((0, 1))


def _selection(value: Any, size: int, name: str) -> NDArray[np.intp]:
    indices = np.atleast_1d(np.asarray(value, dtype=float))
    if (
        indices.ndim != 1
        or not len(indices)
        or not np.isfinite(indices).all()
        or np.any(indices != np.floor(indices))
        or np.any((indices < 1) | (indices > size))
    ):
        raise ValueError(f"{name} must contain one-based indices between 1 and {size}")
    return indices.astype(np.intp) - 1


def _transform(fun: Transform, multistate: bool, hazard: bool) -> tuple[Callable, str]:
    if callable(fun):
        return fun, "Transformed estimate"
    name = "identity" if fun is None else str(fun).lower()
    if hazard:
        if name not in {"identity", "log", "cumhaz"}:
            raise ValueError("invalid function for cumulative hazards")
        return lambda y: y, "Cumulative hazard"
    if name in {"pct", "logpct"} and (name != "logpct" or not multistate):
        return lambda y: 100 * y, "State probability (%)" if multistate else "Survival (%)"
    if name == "cloglog":
        if multistate:
            return lambda y: np.log(-np.log1p(-y)), "log(-log(1 - probability))"
        return lambda y: np.log(-np.log(y)), "log(-log(survival))"
    if name in {"event", "f"} and not multistate:
        return lambda y: 1 - y, "Event probability"
    if name == "log" and multistate:
        return np.log, "Log state probability"
    valid = {"identity", "event", "cumhaz"} if multistate else {"identity", "log", "s", "surv"}
    if name not in valid:
        raise ValueError(f"unrecognized function argument: {fun!r}")
    return lambda y: y, "State probability" if multistate else "Survival probability"


def step_at(time: FloatArray, values: FloatArray, query: FloatArray, *, right: bool) -> FloatArray:
    """R's constant interpolation: f=0 (right=True) or f=1 (right=False)."""

    indices = np.searchsorted(time, query, side="right" if right else "left")
    if right:
        indices -= 1
    result = values[np.clip(indices, 0, len(time) - 1)].copy()
    result[(query < time[0]) | (query > time[-1]) | ~np.isfinite(query)] = np.nan
    return result


def _log_zeros(curves: list[SurvivalCurve]) -> None:
    """R displays zero probabilities below the smallest positive plotted bound."""

    smallest = np.inf
    for curve in curves:
        arrays = (curve.estimate,) if curve.lower is None else (curve.lower, curve.upper)
        for values in arrays:
            if values is None:
                continue
            positive = values[np.isfinite(values) & (values > 0)]
            if len(positive):
                smallest = min(smallest, positive.min())
    if np.isfinite(smallest):
        for curve in curves:
            curve.estimate[curve.estimate == 0] = smallest * 0.8
            if curve.lower is not None:
                curve.lower[curve.lower == 0] = smallest * 0.8


def survfit_plot_data(
    fit: SurvivalFit,
    *,
    conf_int: bool | float | str | None = None,
    conf_type: str | None = None,
    mark_time: bool | Sequence[float] = False,
    fun: Transform = None,
    cumhaz: bool | Sequence[int] = False,
    cumprob: bool | Sequence[int] = False,
    noplot: str | Sequence[str] = "(s0)",
    log: bool | str = False,
    xmax: float | None = None,
) -> SurvivalPlotData:
    """Prepare R-style survival graphics without Matplotlib or model refitting.

    Accept Kaplan-Meier, Turnbull, Cox, multistate and expected-survival results.
    Curves are ordered with strata varying fastest, then prediction rows, then
    states or transitions. ``cumhaz``/``cumprob`` numeric selections are one-based,
    as in R. ``mark_time=True`` marks censored times; a sequence requests specific
    times. Simultaneous censoring and events place the mark halfway down the jump.
    Confidence limits default to the fitted bands for a single curve. A numeric
    ``conf_int`` requests a confidence level; ``"only"`` suppresses the estimate.
    """

    if not isinstance(fit, (SurvfitResult, CoxSurvfitResult, SurvExpResult, *_MULTISTATE)):
        raise TypeError("plotting requires a fitted survfit result")
    time = np.asarray(fit.time, dtype=float)
    if time.ndim != 1 or not len(time) or not np.isfinite(time).all():
        raise ValueError("survival curves must contain finite fitted times")
    expected_fit = fit if isinstance(fit, SurvExpResult) else None
    groups = (
        list(fit.strata.items())
        if not isinstance(fit, SurvExpResult) and fit.strata
        else [("", len(time))]
    )
    if any(size < 1 for _, size in groups) or sum(size for _, size in groups) != len(time):
        raise ValueError("strata must partition the fitted times into nonempty curves")
    multi_fit = fit if isinstance(fit, _MULTISTATE) else None
    multi = multi_fit is not None
    numeric_hazard = not isinstance(cumhaz, bool | np.bool_)
    hazard = numeric_hazard or bool(cumhaz) or fun == "cumhaz"
    cumulative = not isinstance(cumprob, bool | np.bool_) or bool(cumprob)
    if cumulative and not multi:
        raise ValueError("cumprob requires multistate curves")
    source = getattr(
        fit, "cumhaz" if hazard and expected_fit is None else "pstate" if multi else "surv"
    )
    if source is None:
        raise ValueError("survfit object does not contain a cumulative hazard")
    estimate = _array(source, len(time), "estimate")
    if expected_fit is not None and hazard:
        with np.errstate(divide="ignore", invalid="ignore"):
            estimate = -np.log(estimate)
    if estimate.shape[1] == 0:
        raise ValueError("no fitted curves to plot")
    raw_std = getattr(fit, "std_chaz" if hazard else "std_err", None)
    ndata = np.shape(source)[1] if np.ndim(source) == 3 else 1
    states = list(multi_fit.states) if multi_fit is not None else []
    names = (
        list(getattr(fit, "cumhaz_names", getattr(fit, "hazard_names", [])))
        if hazard and multi
        else states
        if multi
        else list(expected_fit.strata or [])
        if expected_fit is not None
        else list(getattr(fit, "colnames", None) or [])
    )
    if multi:
        names = [
            ", ".join(filter(None, (str(i + 1) if ndata > 1 else "", name)))
            for name in names
            for i in range(ndata)
        ]
    if not names:
        names = [str(i + 1) if estimate.shape[1] > 1 else "" for i in range(estimate.shape[1])]
    if len(names) != estimate.shape[1]:
        raise ValueError("curve labels must match the fitted columns")
    original_columns = estimate.shape[1] if not hazard or not multi else len(states) * ndata
    selection = np.arange(estimate.shape[1])
    if numeric_hazard:
        if not multi and np.any(np.asarray(cumhaz) != 1):
            raise ValueError("numeric cumhaz argument only applies to multistate curves")
        # Select transitions, retaining all prediction rows of each transition.
        selected = _selection(cumhaz, estimate.shape[1] // ndata, "cumhaz")
        selection = (selected[:, None] * ndata + np.arange(ndata)).ravel()
    elif multi and not hazard:
        if cumulative:
            selected = (
                np.arange(len(states))
                if isinstance(cumprob, bool | np.bool_)
                else _selection(cumprob, len(states), "cumprob")
            )
        else:
            excluded = [noplot] if isinstance(noplot, str) else list(noplot)
            selected = np.asarray([i for i, state in enumerate(states) if state not in excluded])
            if not len(selected):
                selected = np.arange(len(states))
        selection = (selected[:, None] * ndata + np.arange(ndata)).ravel()
    estimate = estimate[:, selection]
    names = [names[i] for i in selection]
    if cumulative and not hazard:
        estimate = np.cumsum(estimate.reshape(len(time), ndata, -1, order="F"), axis=2).reshape(
            len(time), -1, order="F"
        )
    ci_type = conf_type or getattr(fit, "conf_type", None) or "log"
    if ci_type not in {"log", "log-log", "plain", "logit", "arcsin", "none"}:
        raise ValueError("invalid confidence type")
    plot_estimate = conf_int != "only"
    level = 0.95
    changed_level = False
    if conf_int is None:
        show_ci = raw_std is not None and len(groups) * original_columns == 1
    elif isinstance(conf_int, str):
        if conf_int not in {"none", "only"}:
            raise ValueError("conf_int must be a boolean, confidence level, 'only', or 'none'")
        show_ci = conf_int == "only"
    elif isinstance(conf_int, bool | np.bool_):
        show_ci = bool(conf_int)
    else:
        level = float(conf_int)
        if not np.isfinite(level) or not 0 < level < 1:
            raise ValueError("confidence level must lie strictly between 0 and 1")
        show_ci = True
        changed_level = level != getattr(fit, "conf_int", None)
    show_ci = show_ci and ci_type != "none"
    if cumulative and not hazard and show_ci:
        raise ValueError("confidence intervals not available when cumprob=True")
    if show_ci and raw_std is None:
        raise ValueError("object does not have standard errors, CI not possible")
    lower = upper = None
    if show_ci and not hazard and not changed_level:
        raw_lower, raw_upper = getattr(fit, "lower", None), getattr(fit, "upper", None)
        if raw_lower is not None and raw_upper is not None:
            lower = _array(raw_lower, len(time), "lower")[:, selection]
            upper = _array(raw_upper, len(time), "upper")[:, selection]
    recomputed_ci = show_ci and lower is None
    if recomputed_ci:
        std = _array(raw_std, len(time), "std_err")[:, selection]
        bands = _core.survfit_confint(
            # FloatVec accepts NumPy directly; its generated stub lists Sequence.
            cast(Sequence[float], estimate.ravel()),
            cast(Sequence[float], std.ravel()),
            logse=False if hazard else bool(getattr(fit, "logse", False)),
            conf_type=ci_type,
            conf_int=level,
            ulimit=False,
        )
        lower = np.asarray(bands.lower).reshape(estimate.shape)
        upper = np.asarray(bands.upper).reshape(estimate.shape)
    if not show_ci:
        lower = upper = None
    transform, ylabel = _transform(fun, multi, hazard)
    if isinstance(log, bool | np.bool_):
        xlog, ylog = False, bool(log)
    elif log in {"", "x", "y", "xy"}:
        xlog, ylog = "x" in log, "y" in log
    else:
        raise ValueError("log must be a boolean, 'x', 'y', or 'xy'")
    if isinstance(fun, str):
        xlog = xlog or fun == "cloglog"
        ylog = ylog or (not multi and fun in {"log", "logpct"})
    if xmax is not None and not np.isfinite(xmax):
        raise ValueError("xmax must be finite")
    requested_marks = (
        None
        if isinstance(mark_time, bool | np.bool_)
        else np.sort(np.atleast_1d(np.asarray(mark_time, dtype=float)))
    )
    if requested_marks is not None and requested_marks.ndim != 1:
        raise ValueError("mark_time must be a boolean or vector of times")
    event_counts = (
        np.zeros((len(time), 1))
        if isinstance(fit, SurvExpResult)
        else _array(fit.n_event, len(time), "n_event")
    )
    events = event_counts[:, 0]
    censored = (
        np.zeros(len(time))
        if isinstance(fit, SurvExpResult)
        else (
            events == 0
            if fit.n_censor is None
            else _array(fit.n_censor, len(time), "n_censor")[:, 0]
        )
    )
    origin = getattr(fit, "t0", min(0.0, float(time.min())))
    if not np.isfinite(origin):
        raise ValueError("curve origin must be finite")
    initial = (
        np.zeros((len(groups), estimate.shape[1]))
        if hazard
        else np.ones((len(groups), estimate.shape[1]))
    )
    if multi_fit is not None and not hazard:
        initial = np.repeat(np.asarray(multi_fit.p0, dtype=float), ndata, axis=1)[:, selection]
        if cumulative:
            initial = np.cumsum(initial.reshape(len(groups), ndata, -1, order="F"), axis=2).reshape(
                len(groups), -1, order="F"
            )
    initial_lower = initial_upper = np.full_like(initial, 0.0 if multi or hazard else 1.0)
    if recomputed_ci:
        bands = _core.survfit_confint(
            cast(Sequence[float], initial.ravel()),
            cast(Sequence[float], np.zeros(initial.size)),
            logse=False if hazard else bool(getattr(fit, "logse", False)),
            conf_type=ci_type,
            conf_int=level,
            ulimit=False,
        )
        initial_lower = np.asarray(bands.lower).reshape(initial.shape)
        initial_upper = np.asarray(bands.upper).reshape(initial.shape)
    curves, censor_masks = [], []
    for column, name in enumerate(names):
        start = 0
        for group, (label, size) in enumerate(groups):
            end = start + size
            xx = time[start:end].copy()
            if np.any(np.diff(xx) < 0):
                raise ValueError("fitted times must be ordered within each stratum")
            yy = estimate[start:end, column].copy()
            lo = None if lower is None else lower[start:end, column].copy()
            hi = None if upper is None else upper[start:end, column].copy()
            event = events[start:end] > 0
            event_time = np.any(event_counts[start:end] > 0, axis=1)
            censor = censored[start:end] > 0
            if xx[0] != origin and not getattr(fit, "time0", False):
                xx = np.r_[origin, xx]
                yy = np.r_[initial[group, column], yy]
                event, censor = np.r_[False, event], np.r_[False, censor]
                event_time = np.r_[False, event_time]
                if lo is not None and hi is not None:
                    lo, hi = (
                        np.r_[initial_lower[group, column], lo],
                        np.r_[initial_upper[group, column], hi],
                    )
            with np.errstate(divide="ignore", invalid="ignore"):
                yy = np.asarray(transform(yy.copy()), dtype=float)
                if yy.shape != xx.shape:
                    raise ValueError("fun must return one value per curve time")
                if lo is not None and hi is not None:
                    lo, hi = (
                        np.asarray(transform(lo.copy()), dtype=np.float64),
                        np.asarray(transform(hi.copy()), dtype=np.float64),
                    )
                    if lo.shape != xx.shape or hi.shape != xx.shape:
                        raise ValueError("fun must preserve confidence-limit shape")
            if xmax is not None and xx[-1] > xmax:
                at = np.searchsorted(xx, xmax, side="right")
                if at == 0:
                    raise ValueError("xmax must not precede the curve origin")
                xx = np.r_[xx[:at], xmax]
                yy = np.r_[yy[:at], yy[at - 1]]
                event, censor = np.r_[event[:at], False], np.r_[censor[:at], False]
                event_time = np.r_[event_time[:at], False]
                if lo is not None and hi is not None:
                    lo, hi = np.r_[lo[:at], lo[at - 1]], np.r_[hi[:at], hi[at - 1]]
            curves.append(
                SurvivalCurve(
                    ", ".join(filter(None, (label, name))) or "Survival",
                    xx,
                    yy,
                    lo,
                    hi,
                    np.empty(0),
                    np.empty(0),
                    event_time,
                )
            )
            censor_masks.append((event, censor))
            start = end
    if ylog:
        _log_zeros(curves)
    for i, (curve, (event, censor)) in enumerate(zip(curves, censor_masks, strict=True)):
        if requested_marks is not None:
            marks_x = requested_marks.copy()
            marks_y = step_at(curve.time, curve.estimate, marks_x, right=True)
        elif mark_time:
            marks_x, marks_y = curve.time[censor].copy(), curve.estimate[censor].copy()
            tied = event[censor]
            previous = np.maximum(0, np.flatnonzero(censor) - 1)
            marks_y[tied] = (marks_y[tied] + curve.estimate[previous[tied]]) / 2
        else:
            continue
        curves[i] = replace(curve, censor_time=marks_x, censor_value=marks_y)
    return SurvivalPlotData(tuple(curves), xlog, ylog, ylabel, plot_estimate)
