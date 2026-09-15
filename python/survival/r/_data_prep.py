"""Data utilities: ``tmerge``, ``survSplit``, ``survcondense``, ``neardate``, ``tcut``,
``aeqSurv``, ``lvcf``, ``nostutter`` and ``rttright``.

Each function does what its R counterpart's R code does (``R/tmerge.R``,
``R/survSplit.R``, ``R/survcondense.R``, ``R/neardate.R``, ``R/tcut.R``,
``R/aeqSurv.R``, ``R/xtras.R``, ``R/rttright.R``): build the model frame, check
the arguments, call the kernel and label the result.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .. import _survival as _core
from ._coerce import (
    _as_character,
    _categories,
    _finite_float,
    _float_vector,
    _is_bool_like,
    _is_missing_value,
    _match_string_arg,
    _materialize_1d,
    _materialize_labels,
    _normalize_bool_option,
    _normalize_positive_scale,
    _pop_dotted_keyword,
    _scalar_or_vector,
)
from ._formula import (
    _column,
    _column_source,
    _data_column_names,
    _data_row_count,
    _formula_name,
    _model_strata,
    _model_variables,
    _response_spec,
    _unsupported_formula_name,
    model_frame,
)
from ._surv import Surv, Surv2
from ._types import ModelFrame, TcutResult, TMergeFrame, TMergeOperation

# ---------------------------------------------------------------------------
# tcut, neardate, lvcf, nostutter
# ---------------------------------------------------------------------------


def _numeric_or_nan(values: Any, name: str) -> list[float]:
    """A numeric vector with ``NaN`` for missing values (dates arrive as day counts)."""

    result: list[float] = []
    for value in _materialize_1d(values, name):
        if _is_missing_value(value):
            result.append(math.nan)
            continue
        try:
            result.append(float(value))
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be numeric") from exc
    return result


def tcut(x: Any, breaks: Any, labels: Any | None = None, scale: Any = 1) -> TcutResult:
    """R's ``tcut``: a time-dependent categorisation for ``pyears``."""

    label_values = (
        None if labels is None else [str(value) for value in _materialize_1d(labels, "labels")]
    )
    return _core.tcut(
        _numeric_or_nan(x, "x"),
        _numeric_or_nan(_scalar_or_vector(breaks, "breaks"), "breaks"),
        label_values,
        _normalize_positive_scale(scale),
    )


def neardate(
    id1: Any,
    id2: Any,
    y1: Any,
    y2: Any,
    best: str = "after",
    nomatch: int | None = None,
) -> list[int | None]:
    """R's ``neardate``: for each ``(id1, y1)`` the zero-based row of the closest match.

    The row numbers are Python (zero-based) indices into ``id2``/``y2``; ``nomatch``
    replaces the ``None`` of rows without a match, as R's ``nomatch`` argument does.
    """

    best_value = _match_string_arg(best, "best", ("after", "prior"), "best must be after or prior")
    id1_values = _materialize_labels(id1, "id1")
    id2_values = _materialize_labels(id2, "id2")
    y1_values = _numeric_or_nan(y1, "y1")
    y2_values = _numeric_or_nan(y2, "y2")
    if len(id1_values) != len(y1_values):
        raise ValueError("id1 and y1 have different lengths")
    if len(id2_values) != len(y2_values):
        raise ValueError("id2 and y2 have different lengths")
    present1 = [not _is_missing_value(value) for value in id1_values]
    keep2 = [not _is_missing_value(value) for value in id2_values]
    matched = _core.neardate(
        [value for value, keep in zip(id1_values, present1, strict=True) if keep],
        [value for value, keep in zip(y1_values, present1, strict=True) if keep],
        [value for value, keep in zip(id2_values, keep2, strict=True) if keep],
        [value for value, keep in zip(y2_values, keep2, strict=True) if keep],
        best_value,
    )
    rows2 = [row for row, keep in enumerate(keep2) if keep]
    result: list[int | None] = []
    cursor = iter(matched)
    for keep in present1:
        match = next(cursor) if keep else None
        result.append(nomatch if match is None else rows2[match])
    return result


def lvcf(id: Any, x: Any, time: Any | None = None, first: bool = True) -> list[Any]:
    """R's ``lvcf``: last value carried forward within each ``id``.

    With ``first`` (R's default) a missing first observation of a logical or 0/1
    variable becomes ``False``/``0`` before the values are carried forward.
    """

    id_values = _materialize_labels(id, "id")
    values = list(_materialize_1d(x, "x"))
    if len(values) != len(id_values):
        raise ValueError("x must have the same length as id")
    if any(_is_missing_value(value) for value in id_values):
        raise ValueError("id must not contain missing values")
    times = None if time is None else _numeric_or_nan(time, "time")
    if times is not None and len(times) != len(values):
        raise ValueError("time must have the same length as id")
    if first:
        seen: set[Any] = set()
        observed = [value for value in values if not _is_missing_value(value)]
        logical = all(_is_bool_like(value) for value in observed)
        binary = not logical and all(
            not isinstance(value, str) and float(value) in (0.0, 1.0) for value in observed
        )
        order = sorted(
            range(len(values)),
            key=lambda idx: (
                _as_character(id_values[idx]),
                math.inf if times is None else times[idx],
            ),
        )
        for idx in order:
            key = _as_character(id_values[idx])
            if key in seen:
                continue
            seen.add(key)
            if _is_missing_value(values[idx]) and (logical or binary):
                values[idx] = False if logical else 0
    missing = [_is_missing_value(value) for value in values]
    source = _core.lvcf(id_values, missing, times)
    return [values[row] for row in source]


def nostutter(id: Any, x: Any, censor: Any = 0, single: bool = False) -> list[Any]:
    """R's ``nostutter``: repeated adjacent states within an ``id`` become ``censor``."""

    id_values = _materialize_labels(id, "id")
    values = _materialize_1d(x, "x")
    if len(values) != len(id_values):
        raise ValueError("wrong length for x or id")
    if any(_is_missing_value(value) for value in id_values):
        raise ValueError("id must not contain missing values")
    if any(
        not _is_missing_value(value) and not isinstance(value, int | float | str)
        for value in values
    ):
        raise ValueError("invalid variable type")
    states = [None if _is_missing_value(value) else _as_character(value) for value in values]
    replaced = _core.nostutter(id_values, states, _as_character(censor), bool(single))
    return [censor if flag else value for flag, value in zip(replaced, values, strict=True)]


# ---------------------------------------------------------------------------
# aeqSurv
# ---------------------------------------------------------------------------


def aeqSurv(x: Any, tolerance: Any | None = None) -> Surv:
    """R's ``aeqSurv``: snap near-tied times of a ``Surv`` object."""

    if tolerance is not None:
        try:
            tolerance_value = float(tolerance)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid value for tolerance") from exc
        if not math.isfinite(tolerance_value):
            raise ValueError("invalid value for tolerance")
        if tolerance_value <= 0.0:
            return x
    else:
        tolerance_value = None
    if not isinstance(x, Surv):
        raise TypeError("argument is not a Surv object")
    if x.start is not None:
        result = _core.aeq_surv(list(x.start), list(x.time), tolerance_value)
        return x.replace_times(start=result.time, time=result.time2 or [])
    if x.time2 is not None:
        result = _core.aeq_surv(list(x.time), list(x.time2), tolerance_value)
        return x.replace_times(time=result.time, time2=result.time2 or [])
    return x.replace_times(time=_core.aeq_surv(list(x.time), None, tolerance_value).time)


# ---------------------------------------------------------------------------
# survSplit
# ---------------------------------------------------------------------------


def _surv_argument_name(argument: str) -> str | None:
    """A ``Surv()`` argument that is a plain variable name (R's ``is.name``)."""

    name, quoted = _formula_name(argument)
    if not name or (not quoted and _unsupported_formula_name(name, quoted)):
        return None
    return name


def _surv_argument_names(mf: ModelFrame) -> tuple[str | None, str | None, str | None]:
    """R's ``match.call(Surv, formula[[2]])``: the ``time``, ``time2`` and ``event`` names."""

    spec = mf.spec
    if spec is None or not spec.surv:
        return None, None, None
    names = [_surv_argument_name(argument) for argument in spec.arguments]
    if len(names) == 2:
        return names[0], names[1], None
    if len(names) == 3:
        return names[0], names[1], names[2]
    return names[0], None, None


def _status_labels(states: Sequence[str], status: Sequence[Any]) -> list[Any]:
    """R's ``factor(status, 0:length(states), labels = c("censor", states))``."""

    labels = ["censor", *states] if states else None
    result: list[Any] = []
    for value in status:
        if _is_missing_value(value):
            result.append(None)
        else:
            result.append(labels[int(value)] if labels else int(value))
    return result


def _cut_points(cut: Any) -> list[float]:
    values = _float_vector(_scalar_or_vector(cut, "cut"), "cut")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("cut must be a vector of finite numbers")
    return sorted(set(values))


def _output_name(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{name}' must be a variable name")
    return value.strip()


def _split_frame(columns: Mapping[str, Sequence[Any]], rows: Sequence[int]) -> dict[str, list[Any]]:
    return {name: [values[row] for row in rows] for name, values in columns.items()}


def _data_columns(data: Any, name: str) -> dict[str, list[Any]]:
    """Every column of a mapping or data frame as a list, in data order."""

    if isinstance(data, TMergeFrame):
        return {column: list(values) for column, values in data.columns.items()}
    names = _data_column_names(data)
    if names is None:
        raise TypeError(f"{name} must be a mapping or data frame")
    if any(not isinstance(column, str) or not column for column in names):
        raise ValueError(f"{name} column names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError(f"{name} column names must be unique")
    columns = {column: _column(data, column) for column in names}
    if len({len(values) for values in columns.values()}) > 1:
        raise ValueError(f"{name} columns must have equal lengths")
    return columns


def _split_kernel(response: Any, cut: list[float], zero: float, timefix: bool, id: Any) -> Any:
    """Run the ``survsplit`` kernel on a ``Surv``/``Surv2`` response."""

    if isinstance(response, Surv2):
        if id is None:
            raise ValueError("an id statement is required")
        status = [math.nan if value is None else float(value) for value in response.status]
        return _core.survsplit(
            list(response.time), status, cut, None, _materialize_labels(id, "id"), zero, timefix
        )
    if not isinstance(response, Surv):
        raise ValueError("the model must have a Surv or Surv2 object as the response")
    if response.type not in {"right", "mright", "counting", "mcounting"}:
        raise ValueError(f"not valid for {response.type} censored survival data")
    status = [math.nan if value is None else float(value) for value in response.event]
    start = None if response.start is None else list(response.start)
    return _core.survsplit(list(response.time), status, cut, start, None, zero, timefix)


def survSplit(
    formula: Any = None,
    data: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = "na.pass",
    id: Any | None = None,
    *,
    cut: Any,
    zero: Any = 0,
    episode: str | None = None,
    start: str | None = None,
    end: str | None = None,
    event: str | None = None,
    added: str | None = None,
    timefix: bool = True,
    response: Any | None = None,
    **kwargs: Any,
) -> dict[str, list[Any]]:
    """R's ``survSplit``: split survival records at the ``cut`` times.

    ``formula`` is ``Surv(...) ~ terms``; a ``Surv`` (or ``Surv2``) object may be
    given as ``response`` with ``data`` holding the covariates instead (R's old-style
    call).  ``id`` names the subject column to add for ``(time, status)`` data, or is
    the subject vector of ``Surv2`` timeline data.
    """

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "na.pass")
    if kwargs:
        raise TypeError(
            f"survSplit got unexpected keyword argument(s): {', '.join(sorted(kwargs))}"
        )
    if not isinstance(timefix, bool):
        raise ValueError("invalid value for timefix option")
    cut_values = _cut_points(cut)
    zero_value = _finite_float(zero, "zero")
    if isinstance(formula, Surv | Surv2) and response is None:
        response, formula = formula, None
    if response is not None:
        return _survsplit_object(
            response, data, cut_values, zero_value, timefix, id, start, end, event, episode, added
        )
    if not isinstance(formula, str):
        raise ValueError("either a formula or the end and event arguments are required")
    if data is None:
        raise ValueError("a data argument is required")
    idname = id if isinstance(id, str) else None
    names = _data_column_names(data) or []
    if idname is not None and idname not in names:
        spec = _response_spec(formula)
        if spec is not None and spec.surv and len(spec.arguments) == 2:
            n = _data_row_count(data, formula)
            data = {str(name): _column_source(data, str(name)) for name in names}
            data[idname] = list(range(1, n + 1))
    mf = model_frame(formula, data, subset=subset, na_action=na_action, id=None if idname else id)
    split = _split_kernel(mf.response, cut_values, zero_value, timefix, None)
    rows = list(split.row)
    right_dot = formula.partition("~")[2].strip() == "." and mf.n == len(
        next(iter(_data_columns(data, "data").values()), [])
    )
    if right_dot:
        newdata = _split_frame(_data_columns(data, "data"), rows)
    else:
        newdata = _split_frame(dict(_model_variables(mf)), rows)
        if idname is not None and idname in (_data_column_names(mf.data) or []):
            newdata[idname] = [_column(mf.data, idname)[row] for row in rows]
    states = () if mf.response is None else mf.response.states
    time_name, time2_name, event_name = _surv_argument_names(mf)
    if mf.response is None or mf.response.ncol == 2:
        end = end or time_name or "tstop"
        event = event or time2_name or event_name or "event"
        start = start or "tstart"
    else:
        end = end or time2_name or "tstop"
        event = event or event_name or "event"
        start = start or time_name or "tstart"
    newdata[_output_name(start, "start")] = list(split.start)
    newdata[_output_name(end, "end")] = list(split.end)
    newdata[_output_name(event, "event")] = _status_labels(states, split.status)
    if episode is not None:
        newdata[_output_name(episode, "episode")] = [value + 1 for value in split.interval]
    if added is not None:
        newdata[_output_name(added, "added")] = list(split.censor)
    return newdata


def _survsplit_object(
    response: Any,
    data: Any | None,
    cut: list[float],
    zero: float,
    timefix: bool,
    id: Any | None,
    start: str | None,
    end: str | None,
    event: str | None,
    episode: str | None,
    added: str | None,
) -> dict[str, list[Any]]:
    """``survSplit`` for a ``Surv``/``Surv2`` object plus a frame of covariates."""

    split = _split_kernel(response, cut, zero, timefix, id)
    two_columns = isinstance(response, Surv2) or response.start is None
    rows = list(split.row)
    columns = {} if data is None else _data_columns(data, "data")
    if columns and len(next(iter(columns.values()))) != len(response):
        raise ValueError("data must have one row per response observation")
    newdata = _split_frame(columns, rows)
    if isinstance(id, str) and two_columns and id not in newdata:
        newdata[id] = [row + 1 for row in rows]
    newdata[_output_name(start or "tstart", "start")] = list(split.start)
    if not isinstance(response, Surv2):
        newdata[_output_name(end or "tstop", "end")] = list(split.end)
    newdata[_output_name(event or "event", "event")] = _status_labels(response.states, split.status)
    if episode is not None:
        newdata[_output_name(episode, "episode")] = [value + 1 for value in split.interval]
    if added is not None:
        newdata[_output_name(added, "added")] = list(split.censor)
    return newdata


# ---------------------------------------------------------------------------
# survcondense
# ---------------------------------------------------------------------------


def _row_codes(columns: Sequence[Sequence[Any]]) -> list[int]:
    """One code per row for the combination of values across *columns* (``NA`` matches ``NA``)."""

    codes: dict[tuple[Any, ...], int] = {}
    result: list[int] = []
    for row in zip(*columns, strict=True):
        key = tuple("NA" if _is_missing_value(value) else _as_character(value) for value in row)
        result.append(codes.setdefault(key, len(codes)))
    return result


def survcondense(
    formula: str,
    data: Any | None = None,
    subset: Any | None = None,
    weights: Any | None = None,
    na_action: str | None = "na.pass",
    *,
    id: Any,
    start: str | None = None,
    end: str | None = None,
    event: str | None = None,
    **kwargs: Any,
) -> dict[str, list[Any]]:
    """R's ``survcondense``: merge adjacent ``(start, stop]`` rows with equal covariates."""

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "na.pass")
    id_name = kwargs.pop("_id_name", None)
    weights_name = kwargs.pop("_weights_name", None)
    if kwargs:
        raise TypeError(
            f"survcondense got unexpected keyword argument(s): {', '.join(sorted(kwargs))}"
        )
    if id is None:
        raise ValueError("id is required")
    if not isinstance(formula, str):
        raise ValueError("A formula argument is required")
    mf = model_frame(formula, data, subset=subset, na_action=na_action, weights=weights, id=id)
    if mf.terms.clusters:
        raise ValueError("function does not handle cluster() terms")
    response = mf.response
    if response is None:
        raise ValueError("the response must be a Surv object")
    if response.type not in {"counting", "mcounting"}:
        raise ValueError("invalid survival type")
    if any(math.isnan(value) for value in (*response.time, *(response.start or ()))) or any(
        value is None for value in response.event
    ):
        raise ValueError("response cannot have missing values")
    if mf.id is None:
        raise ValueError("id is required")
    variables = _model_variables(mf)
    comparison = [values for _name, values in variables]
    if mf.weights is not None:
        comparison.append(mf.weights)
    comparison.append(mf.id)
    condensed = _core.survcondense(
        mf.id, list(response.start or ()), list(response.time), _row_codes(comparison)
    )
    keep = list(condensed.keep)
    output: dict[str, list[Any]] = {}
    for name, values in variables:
        output.setdefault(name, [values[row] for row in keep])
    if mf.weights is not None:
        weights_column = weights_name or (weights if isinstance(weights, str) else "(weights)")
        output.setdefault(str(weights_column), [mf.weights[row] for row in keep])
    id_column = id_name or (id if isinstance(id, str) else "id")
    output.setdefault(str(id_column), [mf.id[row] for row in keep])
    time_name, time2_name, event_name = _surv_argument_names(mf)
    output[_output_name(start or time_name or "tstart", "start")] = [
        condensed.start[row] for row in keep
    ]
    output[_output_name(end or time2_name or "tstop", "end")] = [response.time[row] for row in keep]
    output[_output_name(event or event_name or "event", "event")] = _status_labels(
        response.states, [response.event[row] for row in keep]
    )
    return output


# ---------------------------------------------------------------------------
# rttright
# ---------------------------------------------------------------------------


def rttright(
    formula: Any = None,
    data: Any | None = None,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = None,
    times: Any | None = None,
    id: Any | None = None,
    timefix: bool = True,
    renorm: bool = True,
    **kwargs: Any,
) -> list[float] | list[list[float]]:
    """R's ``rttright``: redistribute-to-the-right weights.

    Returns one weight per observation, or one row per observation with a column
    per requested ``times`` value (R's matrix) when several times are given.
    """

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, None)
    response = kwargs.pop("response", None)
    warn_offset = kwargs.pop("_warn_offset", True)
    if kwargs:
        raise TypeError(f"rttright got unexpected keyword argument(s): {', '.join(sorted(kwargs))}")
    if response is not None:
        formula = response
    if not isinstance(timefix, bool):
        raise ValueError("invalid value for timefix option")
    if not isinstance(renorm, bool):
        raise ValueError("invalid value for renorm option")
    if not isinstance(formula, str):
        raise ValueError("a formula argument is required")
    mf = model_frame(formula, data, subset=subset, na_action=na_action, weights=weights, id=id)
    surv = mf.response
    if surv is None:
        raise ValueError("response must be a Surv object")
    if surv.type not in {"right", "mright", "counting", "mcounting"}:
        raise ValueError("response must be right censored")
    casewt = None
    if mf.weights is not None:
        casewt = []
        for value in mf.weights:
            try:
                weight = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("weights must be numeric") from exc
            if not math.isfinite(weight):
                raise ValueError("weights must be finite")
            if weight < 0.0:
                raise ValueError("weights must be non-negative")
            casewt.append(weight)
    if mf.terms.offsets and warn_offset:
        warnings.warn("Offset term ignored", RuntimeWarning, stacklevel=2)
    groups = _model_strata(mf)
    strata_codes = None
    if groups is not None:
        if any(code is None for code in groups.codes):
            raise ValueError("missing values in the strata")
        strata_codes = [int(code) for code in groups.codes if code is not None]
    if surv.ncol == 3 and mf.id is None:
        raise ValueError("id is required for start-stop data")
    if any(value is None for value in surv.event):
        raise ValueError("missing values in the response")
    query = None if times is None else _float_vector(_scalar_or_vector(times, "times"), "times")
    result = _core.rttright(
        list(surv.time),
        [int(value) for value in surv.event],
        None if surv.start is None else list(surv.start),
        strata_codes,
        casewt,
        None if mf.id is None else mf.id,
        query,
        timefix,
        renorm,
    )
    if query is not None and len(query) > 1:
        return [list(row) for row in result.weights]
    return [row[0] for row in result.weights]


# ---------------------------------------------------------------------------
# tmerge
# ---------------------------------------------------------------------------

_TMERGE_COUNT_NAMES = (
    "early",
    "late",
    "gap",
    "within",
    "boundary",
    "leading",
    "trailing",
    "tied",
    "missid",
)


def tdc(time: Any, value: Any | None = None, init: Any | None = None) -> TMergeOperation:
    """A time-dependent covariate argument for :func:`tmerge`."""

    return TMergeOperation("tdc", time, value=value, default=init)


def cumtdc(time: Any, value: Any | None = None, init: Any | None = None) -> TMergeOperation:
    """A cumulative time-dependent covariate argument for :func:`tmerge`."""

    return TMergeOperation("cumtdc", time, value=value, default=init)


def event(time: Any, value: Any | None = None, censor: Any | None = None) -> TMergeOperation:
    """An event argument for :func:`tmerge`."""

    return TMergeOperation("event", time, value=value, censor=censor)


def cumevent(time: Any, value: Any | None = None, censor: Any | None = None) -> TMergeOperation:
    """A cumulative event argument for :func:`tmerge`."""

    return TMergeOperation("cumevent", time, value=value, censor=censor)


def _tmerge_operation(value: Any, name: str) -> TMergeOperation:
    if isinstance(value, TMergeOperation):
        operation = value
    elif isinstance(value, Mapping):
        kind = value.get("kind", value.get("type", value.get("class")))
        operation = TMergeOperation(
            kind=str(kind),
            time=value.get("time"),
            value=value.get("value"),
            default=value.get("default", value.get("init")),
            censor=value.get("censor"),
        )
    else:
        raise ValueError(f"argument(s) {name} not a recognized type")
    kind = operation.kind.lower()
    if kind not in {"tdc", "cumtdc", "event", "cumevent"}:
        raise ValueError(f"argument(s) {name} not a recognized type")
    if operation.time is None:
        raise ValueError(f"argument {name} requires a time")
    return replace(operation, kind=kind)


def _tmerge_control(options: Mapping[str, Any] | None, tname: Mapping[str, str] | None) -> dict:
    """R's ``tmerge.control``: the options, over the names retained from the first call."""

    raw = {} if options is None else dict(options)
    normalized = {("na_rm" if key == "na.rm" else str(key)): value for key, value in raw.items()}
    allowed = {"idname", "tstartname", "tstopname", "delay", "na_rm", "tdcstart"}
    unexpected = sorted(set(normalized) - allowed)
    if unexpected:
        raise ValueError(f"unrecognized option(s):{', '.join(unexpected)}")
    retained = {} if tname is None else dict(tname)
    control = {
        "idname": normalized.get("idname", retained.get("idname", "id")),
        "tstartname": normalized.get("tstartname", retained.get("tstartname", "tstart")),
        "tstopname": normalized.get("tstopname", retained.get("tstopname", "tstop")),
    }
    for key, label in (("idname", "idname"), ("tstartname", "tstart"), ("tstopname", "tstop")):
        if not isinstance(control[key], str) or not control[key]:
            raise ValueError(f"{label} option must be a valid variable name")
    delay = _finite_float(normalized.get("delay", 0.0), "delay")
    if delay < 0.0:
        raise ValueError("delay option must be a number >= 0")
    control["delay"] = delay
    control["na_rm"] = _normalize_bool_option(normalized.get("na_rm", True), "na.rm")
    tdcstart = normalized.get("tdcstart", math.nan)
    control["tdcstart"] = math.nan if _is_missing_value(tdcstart) else tdcstart
    return control


def _tmerge_retained(
    data1: Any, metadata: Mapping[str, Any] | None
) -> tuple[dict[str, str] | None, dict[str, Any], list[str], dict[str, dict[str, int]]]:
    """The ``tm.retain`` attribute of a previous call: names, event censors, tdc names, tcount."""

    if isinstance(data1, TMergeFrame):
        return (
            dict(data1.tname),
            dict(data1.tevent),
            list(data1.tdcvar),
            {name: dict(counts) for name, counts in data1.tcount.items()},
        )
    if metadata is None:
        return None, {}, [], {}
    tname = metadata.get("tname")
    tevent = metadata.get("tevent") or {}
    return (
        None if tname is None else {str(key): str(value) for key, value in dict(tname).items()},
        {str(key): value for key, value in dict(tevent).items()},
        [str(value) for value in metadata.get("tdcvar", ()) or ()],
        {
            str(name): {str(kind): int(count) for kind, count in dict(counts).items()}
            for name, counts in dict(metadata.get("tcount", {}) or {}).items()
        },
    )


def _tmerge_vector(value: Any, data2: Any, n2: int, name: str) -> list[Any]:
    """Evaluate a ``tmerge`` argument in ``data2``: a column name, a scalar or a vector."""

    if isinstance(value, str) and value in (_data_column_names(data2) or []):
        values = _column(data2, value)
    elif isinstance(value, str | bytes) or not hasattr(value, "__iter__"):
        values = [value] * n2
    else:
        values = _materialize_1d(value, name)
        if len(values) == 1 and n2 != 1:
            values = values * n2
    if len(values) != n2:
        raise ValueError(f"argument {name} is not the same length as id")
    return values


def _first_call_frame(
    columns1: dict[str, list[Any]],
    base_ids: list[Any],
    id2: list[Any],
    tstart: list[float] | None,
    tstop: list[float],
    control: Mapping[str, Any],
) -> dict[str, list[Any]]:
    """R's first ``tmerge`` call: ``data1`` rows in ``data2`` order with the time range."""

    if any(_is_missing_value(value) for value in tstop):
        raise ValueError("missing time value, when that variable defines the span")
    keys1 = [_as_character(value) for value in base_ids]
    keys2 = [_as_character(value) for value in id2]
    if tstart is None:
        for value in tstop:
            if value <= 0.0:
                raise ValueError(
                    f"found an ending time of {_as_character(value)}, "
                    "the default starting time of 0 is invalid"
                )
        tstart = [0.0] * len(tstop)
    if any(a >= b for a, b in zip(tstart, tstop, strict=True)):
        raise ValueError("tstart must be < tstop")
    order = list(range(len(id2)))
    if len(set(keys2)) != len(keys2):
        first_seen = {key: idx for idx, key in enumerate(dict.fromkeys(keys2))}
        order.sort(key=lambda idx: (first_seen[keys2[idx]], tstop[idx]))
    position = {key: idx for idx, key in enumerate(keys1)}
    rows = [position[keys2[idx]] for idx in order]
    newdata = {name: [values[row] for row in rows] for name, values in columns1.items()}
    newdata[str(control["tstartname"])] = [tstart[idx] for idx in order]
    newdata[str(control["tstopname"])] = [tstop[idx] for idx in order]
    ids = [keys2[idx] for idx in order]
    starts = newdata[str(control["tstartname"])]
    stops = newdata[str(control["tstopname"])]
    for idx in range(1, len(stops)):
        if ids[idx] == ids[idx - 1] and stops[idx - 1] > starts[idx]:
            raise ValueError("first call has created overlapping or duplicated time intervals")
    return newdata


def _censor_value(values: Sequence[Any], declared: Sequence[Any] | None) -> Any:
    """R's ``tcens`` for a new event variable: the type's censoring value."""

    if declared:
        return declared[0]
    sample = next((value for value in values if not _is_missing_value(value)), 0)
    if _is_bool_like(sample):
        return False
    if isinstance(sample, str):
        return ""
    if isinstance(sample, float):
        return 0.0
    return 0


def _numeric_values(values: Sequence[Any]) -> list[float] | None:
    """The numeric view of an argument's values (``NaN`` for ``NA``), or ``None``."""

    result: list[float] = []
    for value in values:
        if _is_missing_value(value):
            result.append(math.nan)
        elif isinstance(value, str):
            return None
        else:
            try:
                result.append(float(value))
            except (TypeError, ValueError):
                return None
    return result


@dataclass(frozen=True)
class _TmergeArgument:
    """One ``name = kind(time, value)`` argument evaluated in ``data2``.

    ``values`` are the update values (numeric ones as floats with ``NaN`` for
    ``NA``), ``numeric`` their float view when every value is a number.
    """

    name: str
    kind: str
    time: list[float]
    values: list[Any] | None
    numeric: list[float] | None
    default: Any
    censor: Any
    levels: list[Any] | None

    @property
    def missing(self) -> list[bool] | None:
        return None if self.values is None else [_is_missing_value(v) for v in self.values]


def _tmerge_argument(
    name: str, operation: TMergeOperation, data2: Any, n2: int, control: Mapping[str, Any]
) -> _TmergeArgument:
    time = _numeric_or_nan(_tmerge_vector(operation.time, data2, n2, name), f"{name} time")
    values = None if operation.value is None else _tmerge_vector(operation.value, data2, n2, name)
    numeric = None if values is None else _numeric_values(values)
    if operation.kind in {"cumtdc", "cumevent"} and values is not None and numeric is None:
        raise ValueError("invalid increment for cumtdc or cumevent")
    default = control["tdcstart"] if operation.default is None else operation.default
    source = operation.value
    if isinstance(source, str) and source in (_data_column_names(data2) or []):
        source = _column_source(data2, source)
    return _TmergeArgument(
        name=name,
        kind=operation.kind,
        time=time,
        values=numeric if numeric is not None else values,
        numeric=numeric,
        default=default,
        censor=operation.censor,
        levels=None if source is None else _categories(source),
    )


def _tdc_values(step: Any, argument: _TmergeArgument, prior: list[Any] | None) -> list[Any]:
    """R's ``tdc`` update: the value of the last update at or before each interval start."""

    values = argument.values
    if prior is None:
        if values is None:
            return [0 if source is None else 1 for source in step.source]
        return [argument.default if source is None else values[source] for source in step.source]
    if values is None:
        if any(not (_is_missing_value(v) or v in (0, 1, False, True)) for v in prior):
            raise ValueError(f"tdc update does not match prior variable type: {argument.name}")
        return [
            1 if source is not None else v for source, v in zip(step.source, prior, strict=True)
        ]
    return [
        values[source] if source is not None else v
        for source, v in zip(step.source, prior, strict=True)
    ]


def _event_values(
    step: Any, argument: _TmergeArgument, prior: list[Any] | None, censor: Any, n_out: int
) -> list[Any]:
    """R's ``event``/``cumevent`` update: events land on the interval they end."""

    values = [censor] * n_out if prior is None else list(prior)
    for row, source, value in zip(step.event_row, step.event_source, step.event_value, strict=True):
        if argument.kind == "cumevent":
            values[row] = value
        elif argument.values is None:
            values[row] = 1
        else:
            values[row] = argument.values[source]
    return values


def _apply_tmerge_argument(
    newdata: dict[str, list[Any]],
    data2: Any,
    id2: list[Any],
    name: str,
    operation: TMergeOperation,
    control: Mapping[str, Any],
    tevent: dict[str, Any],
    tdcvar: list[str],
    first_call: bool,
) -> dict[str, int]:
    """One ``tmerge`` argument applied to ``newdata`` in place; returns its ``tcount`` row."""

    idname, startname, stopname = (
        str(control["idname"]),
        str(control["tstartname"]),
        str(control["tstopname"]),
    )
    argument = _tmerge_argument(name, operation, data2, len(id2), control)
    kind = argument.kind
    if kind in {"tdc", "cumtdc"} and name in tevent:
        raise ValueError(f"attempt to turn event variable {name} into a {kind}")
    if kind in {"event", "cumevent"} and name in tdcvar:
        raise ValueError(f"attempt to turn time-dependent covariate {name} into an event")
    prior = newdata.get(name)
    if kind == "tdc" and name not in tdcvar and prior is not None:
        warnings.warn(f"replacement of variable '{name}'", stacklevel=3)
        prior = None
    if kind in {"event", "cumevent"} and name not in tevent:
        prior = None
    prior_numeric = None
    if kind == "cumtdc" and prior is not None:
        prior_numeric = _numeric_values(prior)
        if prior_numeric is None:
            raise ValueError("data and starting value do not agree on data type")
    step = _core.tmerge_step(
        newdata[idname],
        [float(value) for value in newdata[startname]],
        [float(value) for value in newdata[stopname]],
        id2,
        argument.time,
        kind,
        argument.numeric,
        argument.missing,
        prior_numeric,
        float(argument.default)
        if kind == "cumtdc" and not _is_missing_value(argument.default)
        else math.nan,
        float(control["delay"]),
        bool(control["na_rm"]),
        not first_call,
    )
    rows = list(step.row)
    expanded = {column: [values[row] for row in rows] for column, values in newdata.items()}
    for column, censor in tevent.items():
        for row in step.censor_rows:
            expanded[column][row] = censor
    expanded[startname] = list(step.start)
    expanded[stopname] = list(step.stop)
    prior_expanded = None if prior is None else [prior[row] for row in rows]
    if kind in {"tdc", "cumtdc"}:
        expanded[name] = (
            _tdc_values(step, argument, prior_expanded) if kind == "tdc" else list(step.cumulative)
        )
        if name not in tdcvar:
            tdcvar.append(name)
    else:
        if name not in tevent:
            tevent[name] = (
                _censor_value([1] if argument.values is None else argument.values, argument.levels)
                if argument.censor is None
                else argument.censor
            )
        expanded[name] = _event_values(step, argument, prior_expanded, tevent[name], len(rows))
    newdata.clear()
    newdata.update(expanded)
    return dict(zip(_TMERGE_COUNT_NAMES, step.tcount, strict=True))


def _tmerge_first_call(
    data1: Any,
    data2: Any,
    id: Any,
    id2: list[Any],
    tstart: Any | None,
    tstop: Any | None,
    first_operation: TMergeOperation | None,
    control: Mapping[str, Any],
) -> dict[str, list[Any]]:
    """R's first ``tmerge`` call: ``data1`` with the ``(tstart, tstop]`` range of each subject.

    Without ``tstop`` the first argument must be an ``event`` whose first time per
    subject sets the range.
    """

    if not isinstance(id, str):
        raise ValueError("on the first call 'id' must be a single variable name")
    columns1 = _data_columns(data1, "data1")
    if id not in columns1:
        raise ValueError("id variable not found in data1")
    base_ids = list(columns1[id])
    keys1 = {_as_character(value) for value in base_ids}
    if len(keys1) != len(base_ids):
        raise ValueError(
            "for the first call (that establishes the time range) data1 must have no "
            "duplicate identifiers"
        )
    idname = str(control["idname"])
    if idname != id:
        columns1[idname] = list(base_ids)
    keys2 = {_as_character(value) for value in id2}
    if keys2 - keys1:
        raise ValueError("setting the range, and data2 has id values not in data1")
    if keys1 - keys2:
        raise ValueError("setting the range, and data1 has id values not in data2")
    n2 = len(id2)
    if tstop is None:
        if first_operation is None or first_operation.kind != "event":
            raise ValueError("neither a tstop argument nor an initial event argument was found")
        times = _numeric_or_nan(_tmerge_vector(first_operation.time, data2, n2, "tstop"), "tstop")
        seen: dict[str, int] = {}
        for row, value in enumerate(id2):
            seen.setdefault(_as_character(value), row)
        range_ids = [id2[row] for row in seen.values()]
        range_stop = [times[row] for row in seen.values()]
    else:
        range_ids = id2
        range_stop = _numeric_or_nan(_tmerge_vector(tstop, data2, n2, "tstop"), "tstop")
    start_values = (
        None
        if tstart is None
        else _numeric_or_nan(_tmerge_vector(tstart, data2, len(range_ids), "tstart"), "tstart")
    )
    return _first_call_frame(columns1, base_ids, range_ids, start_values, range_stop, control)


def tmerge(
    data1: Any,
    data2: Any,
    id: Any,
    *,
    tstart: Any | None = None,
    tstop: Any | None = None,
    options: Mapping[str, Any] | None = None,
    operations: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    **args: Any,
) -> TMergeFrame:
    """R's ``tmerge``: build ``(tstart, tstop]`` data with time-dependent covariates.

    ``id`` is the name of the subject column (on the first call it must exist in
    both data sets; later calls may pass a vector aligned with ``data2``).  The
    ``tdc``/``cumtdc``/``event``/``cumevent`` arguments follow as keywords whose
    ``time``/``value`` are column names of ``data2`` or vectors.
    """

    if data1 is None or data2 is None or id is None:
        raise ValueError("the data1, data2, and id arguments are required")
    named = {} if operations is None else dict(operations)
    duplicated = set(named) & set(args)
    if duplicated:
        raise TypeError(f"duplicate tmerge argument(s): {', '.join(sorted(duplicated))}")
    named.update(args)
    if any(not isinstance(name, str) or not name for name in named):
        raise ValueError("all additional argments must have a name")
    parsed = {name: _tmerge_operation(value, name) for name, value in named.items()}
    tname, tevent, tdcvar, tcount = _tmerge_retained(data1, metadata)
    first_call = tname is None
    if first_call and isinstance(id, str):
        control = _tmerge_control({"idname": id, **(dict(options) if options else {})}, None)
        if options and "idname" in options:
            control["idname"] = str(options["idname"])
    else:
        control = _tmerge_control(options, tname)
    columns2 = _data_columns(data2, "data2")
    n2 = len(next(iter(columns2.values()), []))
    if isinstance(id, str):
        if id not in columns2:
            raise ValueError("id variable not found in data2")
        id2 = list(columns2[id])
    else:
        id2 = _materialize_labels(id, "id")
        if len(id2) != n2:
            raise ValueError("id variable not found in data2")
    if any(_is_missing_value(value) for value in id2):
        raise ValueError("id variable cannot have missing values")

    if first_call:
        newdata = _tmerge_first_call(
            data1, data2, id, id2, tstart, tstop, next(iter(parsed.values()), None), control
        )
    else:
        if tstart is not None or tstop is not None:
            raise ValueError("tstart and tstop arguments only apply to the first call")
        newdata = _data_columns(data1, "data1")
        for key in ("idname", "tstartname", "tstopname"):
            if str(control[key]) not in newdata:
                raise ValueError("tmerge object has been modified, missing variables")

    for name, operation in parsed.items():
        tcount[name] = _apply_tmerge_argument(
            newdata, data2, id2, name, operation, control, tevent, tdcvar, first_call
        )
    return TMergeFrame(
        columns=newdata,
        tname={key: str(control[key]) for key in ("idname", "tstartname", "tstopname")},
        tevent=tevent,
        tdcvar=tuple(tdcvar),
        tcount=tcount,
    )
