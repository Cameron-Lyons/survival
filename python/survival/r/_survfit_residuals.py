"""``residuals.survfit`` and ``pseudo``: the infinitesimal-jackknife residuals of a curve.

Both re-read the model frame of the fit, as R does through ``model.frame(object)``, and hand
the data to the Rust ports of ``rsurvpart1`` / ``rsurvpart2`` (``survfitresid``,
``survfitresid_aj``) and of ``pseudo`` (``pseudo``, ``pseudo_aj``).
"""

from __future__ import annotations

import math
import warnings
from typing import Any

from .. import _survival as _core
from ._coerce import _float_vector, _is_bool_like, _match_string_arg, _pop_dotted_keyword
from ._survfit import _survfit_data_from_fit, _SurvfitData
from ._types import (
    CoxSurvfitResult,
    SurvfitMultiStateResult,
    SurvfitResidualsResult,
    SurvfitResult,
)

_RESIDUAL_TYPES = ("pstate", "cumhaz", "sojourn", "survival", "chaz", "rmst", "rmts", "auc")
_CANONICAL_TYPE = {
    "pstate": "pstate",
    "survival": "pstate",
    "cumhaz": "cumhaz",
    "chaz": "cumhaz",
    "sojourn": "auc",
    "rmst": "auc",
    "rmts": "auc",
    "auc": "auc",
}


def _residual_type(type_: Any) -> str:
    """``match.arg(casefold(type), ...)`` folded to ``pstate`` / ``cumhaz`` / ``auc``."""

    if not isinstance(type_, str):
        raise TypeError("type must be a string")
    quoted = ", ".join(f'"{choice}"' for choice in _RESIDUAL_TYPES)
    matched = _match_string_arg(
        type_.lower(), "type", _RESIDUAL_TYPES, f"'type' should be one of {quoted}"
    )
    return _CANONICAL_TYPE[matched]


def _logical(value: Any, name: str) -> bool:
    if not _is_bool_like(value):
        raise ValueError(f"{name} must be TRUE/FALSE")
    return bool(value)


def _residual_times(times: Any | None) -> list[float]:
    if times is None:
        raise ValueError("the times argument is required")
    try:
        values = [float(times)] if isinstance(times, int | float) else _float_vector(times, "times")
    except (TypeError, ValueError) as exc:
        raise ValueError("times must be a numeric vector") from exc
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("times must be a numeric vector")
    return sorted(set(values))


def _check_survfit_object(fit: Any) -> SurvfitResult | SurvfitMultiStateResult:
    if isinstance(fit, CoxSurvfitResult):
        raise TypeError("residuals method for coxph survival curve not found")
    if not isinstance(fit, SurvfitResult | SurvfitMultiStateResult):
        raise TypeError("argument must be a survfit object")
    if fit.type == "interval":
        raise ValueError("residuals for interval-censored data are not available")
    if fit.call.start_time is not None:
        raise NotImplementedError("residuals of a curve fitted with start.time are not available")
    return fit


def _row_labels(frame: _SurvfitData, codes: list[int]) -> list[Any]:
    """The id of each residual row: the id values, or R's ``seq(n)`` when there is none."""

    if frame.id is None:
        return [code + 1 for code in codes]
    levels = list(dict.fromkeys(frame.id))
    return [levels[code] for code in codes]


def _kernel_residuals(
    fit: SurvfitResult | SurvfitMultiStateResult,
    frame: _SurvfitData,
    times: list[float],
    type_: str,
    *,
    collapse: bool | None,
    weighted: bool | None,
) -> Any:
    """``rsurvpart1`` / ``rsurvpart2`` (residuals) or ``pseudo`` (``collapse=None``)."""

    y, call = frame.y, fit.call
    common: dict[str, Any] = {
        "start": None if y.start is None else list(y.start),
        "weights": frame.weights,
        "strata": frame.strata_codes,
        "id": frame.id_codes(),
        "type_": type_,
        "timefix": call.timefix,
    }
    if collapse is not None:
        common.update(collapse=collapse, weighted=weighted)
    if isinstance(fit, SurvfitMultiStateResult):
        istate, istate_levels = frame.istate_labels()
        kernel = _core.survfitresid_aj if collapse is not None else _core.pseudo_aj
        return kernel(
            list(y.time),
            [int(value) for value in y.event],
            list(y.states),
            times,
            istate=istate,
            istate_levels=istate_levels,
            cluster=frame.cluster_codes(),
            p0=call.p0,
            **common,
        )
    kernel = _core.survfitresid if collapse is not None else _core.pseudo
    return kernel(
        list(y.time),
        [int(value) for value in y.event],
        times,
        stype=call.stype,
        ctype=call.ctype,
        **common,
    )


def _residuals_result(
    fit: SurvfitResult | SurvfitMultiStateResult,
    frame: _SurvfitData,
    result: Any,
    type_: str,
) -> SurvfitResidualsResult:
    multistate = isinstance(fit, SurvfitMultiStateResult)
    return SurvfitResidualsResult(
        resid=result.values,
        time=result.times,
        id=_row_labels(frame, result.id),
        curve=[int(code) + 1 for code in result.curve] if fit.strata is not None else None,
        columns=list(result.columns) if multistate else None,
        column_name=("transition" if type_ == "cumhaz" else "state") if multistate else None,
        id_name=fit.call.id if frame.id is not None else None,
    )


def _residual_frame(result: SurvfitResidualsResult) -> dict[str, list[Any]]:
    """R's ``data.frame = TRUE`` layout: one row per (id, column, time), times slowest."""

    n = len(result.id)
    columns = result.columns or [None]
    frame: dict[str, list[Any]] = {
        result.id_name or "(id)": [
            value for _time in result.time for _column in columns for value in result.id
        ]
    }
    if result.columns is not None:
        frame[result.column_name or "state"] = [
            column for _time in result.time for column in columns for _row in range(n)
        ]
    frame["time"] = [time for time in result.time for _column in columns for _row in range(n)]
    if result.columns is None:
        frame["resid"] = [row[j] for j in range(len(result.time)) for row in result.resid]
    else:
        frame["resid"] = [
            row[k][j]
            for j in range(len(result.time))
            for k in range(len(columns))
            for row in result.resid
        ]
    if result.curve is not None:
        frame["curve"] = [
            curve for _time in result.time for _column in columns for curve in result.curve
        ]
    return frame


def survfit_residuals(
    object: Any,
    times: Any | None = None,
    type: str = "pstate",
    collapse: Any = False,
    weighted: Any | None = None,
    data_frame: Any = False,
    extra: Any = False,
    **kwargs: Any,
) -> Any:
    """R's ``residuals.survfit``: the influence of each observation on the curve at ``times``.

    ``type`` is ``"pstate"`` (``"survival"``), ``"cumhaz"`` (``"chaz"``) or the area under the
    curve (``"sojourn"``, ``"rmst"``, ``"rmts"``, ``"auc"``).  ``collapse`` sums the rows of each
    subject (``weighted`` multiplies them by the case weights; it defaults to ``collapse``).
    The result carries the residual matrix (or array), the row ids and the curve of each row;
    ``extra`` is accepted for R compatibility (the curve is always reported) and
    ``data_frame = True`` returns the long-format columns of R's data frame.
    """

    data_frame = _pop_dotted_keyword(kwargs, "data.frame", "data_frame", data_frame, False)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"survfit_residuals got unexpected keyword argument(s): {unexpected}")
    fit = _check_survfit_object(object)
    collapse = _logical(collapse, "collapse")
    weighted = collapse if weighted is None else _logical(weighted, "weighted")
    data_frame = _logical(data_frame, "data.frame")
    _logical(extra, "extra")
    type_ = _residual_type(type)
    times = _residual_times(times)

    frame = _survfit_data_from_fit(fit)
    cluster = frame.cluster if frame.cluster is not None else frame.id
    if cluster is None or len(set(cluster)) == len(cluster):
        collapse = False
    if collapse and not weighted:
        raise ValueError("invalid combination of options: collapse=TRUE and weighted=FALSE")
    # R sets weighted <- FALSE when there are no case weights, which only skips the
    # (unit) resid * casewt product; the kernel already treats absent weights as 1.
    if collapse and frame.id is not None and frame.strata_codes is not None:
        seen: dict[Any, int] = {}
        for value, code in zip(frame.id, frame.x_codes, strict=True):
            if seen.setdefault(value, code) != code:
                raise ValueError("same id appears in multiple curves, cannot collapse")
    result = _kernel_residuals(fit, frame, times, type_, collapse=collapse, weighted=weighted)
    residuals = _residuals_result(fit, frame, result, type_)
    return _residual_frame(residuals) if data_frame else residuals


def _drop(values: list[Any], multistate: bool, n_times: int) -> Any:
    """R's ``drop()`` on the pseudo-value matrix / array."""

    if n_times > 1:
        return values
    if multistate:
        return [[column[0] for column in row] for row in values]
    return [row[0] for row in values]


def pseudo(
    fit: Any,
    times: Any | None = None,
    type: str | None = None,
    collapse: Any = True,
    data_frame: Any = False,
    **kwargs: Any,
) -> Any:
    """R's ``pseudo``: jackknife pseudo values of the curve at ``times``, from the IJ residuals.

    Rows are the subjects of the fit (observations without an id); the result is the matrix
    of pseudo values (``rows x times``, or ``rows x states x times`` for a multi-state curve),
    dropped to a vector for a single time as R does.  ``data_frame = True`` returns the
    long-format columns with the residual and the pseudo value.
    """

    data_frame = _pop_dotted_keyword(kwargs, "data.frame", "data_frame", data_frame, False)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"pseudo got unexpected keyword argument(s): {unexpected}")
    fit = _check_survfit_object(fit)
    collapse = _logical(collapse, "collapse")
    data_frame = _logical(data_frame, "data.frame")
    type_ = "pstate" if type is None else _residual_type(type)
    times = _residual_times(times)

    frame = _survfit_data_from_fit(fit)
    if not collapse and frame.id is not None and len(set(frame.id)) < len(frame.id):
        raise NotImplementedError(
            "pseudo values per observation (collapse = FALSE) of a subject with several rows "
            "are not available"
        )
    n_curves = len(fit.strata) if fit.strata else 1
    sizes = list(fit.strata.values()) if fit.strata else [len(fit.time)]
    ends = [fit.time[sum(sizes[: k + 1]) - 1] for k in range(n_curves) if sizes[k] > 0]
    if any(end < max(times) for end in ends):
        warnings.warn(
            "requested time points are beyond the end of one or more curves", stacklevel=2
        )
    multistate = isinstance(fit, SurvfitMultiStateResult)
    result = _kernel_residuals(fit, frame, times, type_, collapse=None, weighted=None)
    if not data_frame:
        return _drop(result.values, multistate, len(times))
    residuals = _kernel_residuals(fit, frame, times, type_, collapse=True, weighted=True)
    frame_columns = _residual_frame(_residuals_result(fit, frame, residuals, type_))
    columns = result.columns if multistate else [None]
    frame_columns["pseudo"] = [
        (row[k][j] if multistate else row[j])
        for j in range(len(times))
        for k in range(len(columns))
        for row in result.values
    ]
    return frame_columns
