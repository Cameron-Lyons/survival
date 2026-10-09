"""Data utilities: ``tmerge``, ``survSplit``, ``survcondense``, ``neardate``, ``tcut``,
``aeqSurv``, ``lvcf``, ``nostutter`` and ``rttright``.

Each function does what its R counterpart's R code does (``R/tmerge.R``,
``R/survSplit.R``, ``R/survcondense.R``, ``R/neardate.R``, ``R/tcut.R``,
``R/aeqSurv.R``, ``R/xtras.R``, ``R/rttright.R``): build the model frame, check
the arguments, call the kernel and label the result.
"""

from __future__ import annotations

import math
import numbers
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, TypeVar, cast

from .. import _survival as _core
from ._coerce import (
    _DEFAULT_NA_ACTION,
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
    _warn_outside_package,
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
    _timeline_counting,
    _unsupported_formula_name,
    model_frame,
)
from ._surv import Surv, Surv2
from ._types import ModelFrame, TcutResult, TMergeFrame, TMergeOperation, _SurvResponseSpec

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
    # R's match() pairs a missing id with a missing id, so such rows stay in both sets
    # (their answer is NA whatever nomatch says); rows of set 2 without a date go.
    missing1 = [_is_missing_value(value) for value in id1_values]
    id1_labels = [
        "\x00NA" if missing else value for value, missing in zip(id1_values, missing1, strict=True)
    ]
    id2_labels = ["\x00NA" if _is_missing_value(value) else value for value in id2_values]
    keep2 = [not math.isnan(value) for value in y2_values]
    matched = _core.neardate(
        id1_labels,
        y1_values,
        [value for value, keep in zip(id2_labels, keep2, strict=True) if keep],
        [value for value, keep in zip(y2_values, keep2, strict=True) if keep],
        best_value,
    )
    rows2 = [row for row, keep in enumerate(keep2) if keep]
    result: list[int | None] = []
    for match, missing in zip(matched, missing1, strict=True):
        if missing:
            result.append(None)
        else:
            result.append(nomatch if match is None else rows2[match])
    return result


def lvcf(id: Any, x: Any, time: Any | None = None, first: bool = True) -> list[Any]:
    """R's ``lvcf``: last value carried forward within each ``id``.

    With ``first`` (R's default) a missing first observation of a logical or 0/1
    variable becomes ``False``/``0`` before the values are carried forward.
    Factor-valued columns keep their unknown initial levels. Supplied times are
    ordered within each subject with missing times last; results retain input order.
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
    missing = [_is_missing_value(value) for value in values]
    source = _core.lvcf(id_values, missing, times)
    if first and _categories(x) is None:
        # A missing row refers to itself only when it is the subject's first
        # observation. Reuse the native ordering (including missing times last)
        # instead of sorting the same rows again in Python.
        initial_missing = [
            row for row, origin in enumerate(source) if row == origin and missing[row]
        ]
        if initial_missing:
            observed = [value for value, absent in zip(values, missing, strict=True) if not absent]
            logical = all(_is_bool_like(value) for value in observed)
            binary = not logical and all(
                not isinstance(value, str) and float(value) in (0.0, 1.0) for value in observed
            )
            if logical or binary:
                for row in initial_missing:
                    values[row] = False if logical else 0
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


_SurvT = TypeVar("_SurvT", Surv, Surv2)


def aeqSurv(x: _SurvT, tolerance: Any | None = None) -> _SurvT:
    """R's ``aeqSurv``: snap near-tied times of a ``Surv`` or ``Surv2`` object."""

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
    if isinstance(x, Surv2):
        return x.replace_times(time=_core.aeq_surv(list(x.time), None, tolerance_value).time)
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


def _status_labels(states: Sequence[str], status: Sequence[float | int | None]) -> list[Any]:
    """R's ``factor(status, 0:length(states), labels = c("censor", states))``.

    ``status`` holds the integer codes, with ``None`` or ``NaN`` (the kernels'
    missing value) for ``NA``.
    """

    if not states:
        return [None if code is None or code != code else int(code) for code in status]
    labels = ["censor", *states]
    return [None if code is None or code != code else labels[int(code)] for code in status]


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
    return {name: list(map(values.__getitem__, rows)) for name, values in columns.items()}


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


def _old_style_formula(data: Any, start: Any, end: Any, event: Any) -> str:
    """The formula of R's old-style ``survSplit`` call: ``Surv([start, ]end, event) ~ .``."""

    if end is None or event is None:
        raise ValueError("either a formula or the end and event arguments are required")
    names = _data_column_names(data) or []
    if not (isinstance(event, str) and event in names):
        raise ValueError("'event' must be a variable name in the data set")
    if not (isinstance(end, str) and end in names):
        raise ValueError("'end' must be a variable name in the data set")
    start = "tstart" if start is None else start
    if not isinstance(start, str):
        raise ValueError("'start' must be a variable name")
    columns = [start, end, event] if start in names else [end, event]
    return f"Surv({', '.join(f'`{name}`' for name in columns)}) ~ ."


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

    ``formula`` is ``Surv(...) ~ terms`` or, for timeline data, ``Surv2(...) ~ terms``,
    whose cut rows are inserted into the timeline.  R's old-style call gives no formula
    (or the data frame in its place) and names the ``end`` and ``event`` columns
    instead, splitting ``Surv([start, ]end, event) ~ .``.  ``id`` names the
    subject column to add for ``(time, status)`` data, or gives the subjects (a vector
    or a column name) of ``Surv2`` timeline data.  A ``Surv`` (or ``Surv2``) object may
    also be given as ``response`` (or ``formula``) with ``data`` holding the covariates,
    one row per observation.
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
    if formula is None or _data_column_names(formula) is not None:
        if data is None:
            if formula is None:
                raise ValueError("a data frame is required")
            data = formula
        formula = _old_style_formula(data, start, end, event)
    elif not isinstance(formula, str):
        raise ValueError("either a formula or the end and event arguments are required")
    if data is None:
        raise ValueError("a data argument is required")
    idname = id if isinstance(id, str) else None
    names = _data_column_names(data) or []
    added_id = False
    if idname is not None and idname not in names:
        spec = _response_spec(formula)
        if spec is not None and spec.surv and not spec.timeline and len(spec.arguments) == 2:
            n = _data_row_count(data, formula)
            data = {str(name): _column_source(data, str(name)) for name in names}
            data[idname] = list(range(1, n + 1))
            added_id = True
    # a character id names a data column, which the model frame subsets with the rows
    id_column = idname if added_id or idname in names else None
    mf = model_frame(
        formula,
        data,
        subset=subset,
        na_action=na_action,
        id=id if idname is None else id_column,
        timeline=True,
    )
    # R only invents the id column for right-censored (time, status) data
    if added_id and (not isinstance(mf.response, Surv) or mf.response.type != "right"):
        data = {name: values for name, values in data.items() if name != idname}
        added_id = False
    # a Surv2 timeline keeps its (time, event) form: cut rows are inserted into it
    timeline = isinstance(mf.response, Surv2)
    split = _split_kernel(mf.response, cut_values, zero_value, timefix, mf.id if timeline else None)
    rows = split.row
    # R's rightdot: with ``~ .`` and no rows dropped the data itself is split, so
    # every column keeps its place
    if formula.partition("~")[2].strip() == "." and mf.n == _data_row_count(data):
        newdata = _split_frame(_data_columns(data, "data"), rows)
    else:
        newdata = _split_frame(dict(_model_variables(mf)), rows)
        if timeline:
            if mf.id is None:
                raise ValueError("id is required for timeline data")
            newdata["(id)"] = [mf.id[row] for row in rows]
        elif idname is not None and (added_id or idname in names):
            if mf.id is None:
                raise ValueError("id column is missing from the model frame")
            newdata[idname] = [mf.id[row] for row in rows]
    states = () if mf.response is None else mf.response.states
    time_name, time2_name, event_name = _surv_argument_names(mf)
    if isinstance(mf.response, Surv2):
        # the Surv2 arguments that are variable names name the columns, whatever
        # start and event say
        start = time_name or start or "tstart"
        event = time2_name or event or "event"
    elif mf.response is None or mf.response.ncol == 2:
        end = end or time_name or "tstop"
        event = event or time2_name or event_name or "event"
        start = start or "tstart"
    else:
        end = end or time2_name or "tstop"
        event = event or event_name or "event"
        start = start or time_name or "tstart"
    newdata[_output_name(start, "start")] = split.start
    if not timeline:
        newdata[_output_name(end, "end")] = split.end
    newdata[_output_name(event, "event")] = _status_labels(states, split.status)
    return _split_extras(newdata, split, episode, added)


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
    rows = split.row
    columns = {} if data is None else _data_columns(data, "data")
    if columns and len(next(iter(columns.values()))) != len(response):
        raise ValueError("data must have one row per response observation")
    newdata = _split_frame(columns, rows)
    if isinstance(id, str) and two_columns and id not in newdata:
        newdata[id] = [row + 1 for row in rows]
    newdata[_output_name(start or "tstart", "start")] = split.start
    if not isinstance(response, Surv2):
        newdata[_output_name(end or "tstop", "end")] = split.end
    newdata[_output_name(event or "event", "event")] = _status_labels(response.states, split.status)
    return _split_extras(newdata, split, episode, added)


def _split_extras(
    newdata: dict[str, list[Any]], split: Any, episode: str | None, added: str | None
) -> dict[str, list[Any]]:
    """survSplit's optional ``episode`` (interval number) and ``added`` (inserted row) columns."""

    if episode is not None:
        newdata[_output_name(episode, "episode")] = [value + 1 for value in split.interval]
    if added is not None:
        newdata[_output_name(added, "added")] = split.censor
    return newdata


# ---------------------------------------------------------------------------
# fromtimeline
# ---------------------------------------------------------------------------


def fromtimeline(
    formula: str,
    data: Any,
    subset: Any | None = None,
    id: Any | None = None,
    repeated: Any = False,
    lvcf: Any = True,
    yname: Any | None = None,
) -> dict[str, list[Any]]:
    """R's ``fromtimeline`` (R/fromtimeline.R): timeline data as counting-process data.

    ``formula`` is ``Surv2(time, event) ~ terms`` (or ``Surv(time, event) ~ terms``) and
    ``id`` (a column name or a vector) gives the subjects, whose consecutive rows become
    intervals as :func:`coxph` makes them; with ``lvcf`` a missing variable takes the
    subject's last value.  The result holds the model variables, ``istate`` when every
    subject starts in a state, and the response columns: named ``yname`` or, by default,
    after the response's arguments (``t1``, ``t2`` and ``s`` for ``Surv2(t, s)``).
    """

    if not isinstance(formula, str):
        raise ValueError("a formula argument is required")
    if data is None:
        raise ValueError("the data argument is required")
    if id is None:
        raise ValueError("the id argument is required")
    spec = _response_spec(formula)
    if spec is None or not spec.surv:
        raise ValueError("response must be a survival object")
    counting_formula, rows, arguments = _timeline_counting(
        formula,
        data,
        subset,
        {"id": id},
        repeated=repeated,
        lvcf=_normalize_bool_option(lvcf, "lvcf"),
        require_repeats=True,
    )
    mf = model_frame(counting_formula, rows, na_action=None)
    new = {name: _materialize_1d(values, name) for name, values in _model_variables(mf)}
    if arguments.get("istate") is not None:
        new["(istate)" if "istate" in new else "istate"] = list(arguments["istate"])
    response = cast(Surv, mf.response)
    counting = response.start is not None
    status = _status_labels(response.states, response.event)
    times = (
        [list(response.start), list(response.time)]
        if response.start is not None
        else [list(response.time)]
    )
    taken = {spec.name, *new}
    if yname is None:
        names = _timeline_response_names(spec, counting)
        names = [f"_{name}_" if name in taken else name for name in names]
    else:
        names = [str(name) for name in _scalar_or_vector(yname, "yname")]
        if any(name in taken for name in names):
            raise ValueError("element of yname conflicts with an existing name in the data")
        if len(names) != 3 and (counting or len(names) != 2):
            raise ValueError("wrong length for yname")
        if not counting:
            names = [names[0], names[-1]]
    return {**new, **dict(zip(names, [*times, status], strict=True))}


def _timeline_response_names(spec: _SurvResponseSpec, counting: bool) -> list[str]:
    """fromtimeline's default response names: ``tstart``, ``tstop`` and ``status``, or
    the response's time argument (with 1 and 2 appended for intervals) and its event
    argument when they are variable names."""

    time_name, event_name = (_surv_argument_name(argument) for argument in spec.arguments[:2])
    if counting:
        names = [f"{time_name}1", f"{time_name}2"] if time_name else ["tstart", "tstop"]
    else:
        names = [time_name or "tstart"]
    return [*names, event_name or "status"]


# ---------------------------------------------------------------------------
# survcondense
# ---------------------------------------------------------------------------


def _row_codes(columns: Sequence[Sequence[Any]]) -> list[int]:
    """One code per row for the combination of values across *columns*.

    Values are compared as R's ``==`` does, so ``0.3`` and ``0.1 + 0.2`` differ;
    ``NA`` matches ``NA``.
    """

    codes: dict[tuple[Any, ...], int] = {}
    result: list[int] = []
    for row in zip(*columns, strict=True):
        key = tuple(None if _is_missing_value(value) else value for value in row)
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
    if not isinstance(response, Surv):
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
    new_start = condensed.start
    output: dict[str, list[Any]] = {}
    for name, values in variables:
        output.setdefault(name, [values[row] for row in keep])
    if mf.weights is not None:
        weights_column = weights_name or (weights if isinstance(weights, str) else "(weights)")
        output.setdefault(str(weights_column), [mf.weights[row] for row in keep])
    id_column = id_name or (id if isinstance(id, str) else "id")
    output.setdefault(str(id_column), [mf.id[row] for row in keep])
    time_name, time2_name, event_name = _surv_argument_names(mf)
    output[_output_name(start or time_name or "tstart", "start")] = [new_start[row] for row in keep]
    output[_output_name(end or time2_name or "tstop", "end")] = [response.time[row] for row in keep]
    output[_output_name(event or event_name or "event", "event")] = _status_labels(
        response.states, [response.event[row] for row in keep]
    )
    return output


# ---------------------------------------------------------------------------
# rttright
# ---------------------------------------------------------------------------


def _rttright_survcheck(response: Surv, id_values: Sequence[Any]) -> None:
    """R's ``survcheck2`` gate on the ``id`` data: any flagged row is an error."""

    levels = list(dict.fromkeys(id_values))
    codes = {value: code for code, value in enumerate(levels, start=1)}
    check = _core.survcheck(
        [codes[value] for value in id_values],
        list(response.time),
        response._event_codes(),
        list(response.states) or ["event"],
        time1=None if response.start is None else list(response.start),
    )
    flags = check.flag
    if flags.overlap or flags.gap or flags.jump or flags.teleport or flags.duplicate:
        raise ValueError("one or more flags are >0 in survcheck")


def rttright(
    formula: Any = None,
    data: Any | None = None,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = _DEFAULT_NA_ACTION,
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

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, _DEFAULT_NA_ACTION)
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
    if not isinstance(surv, Surv):
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
    if mf.id is not None:
        _rttright_survcheck(surv, mf.id)
    query = None if times is None else _float_vector(_scalar_or_vector(times, "times"), "times")
    result = _core.rttright(
        list(surv.time),
        surv._event_codes(),
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


def summary_tmerge(object: TMergeFrame, **_kwargs: Any) -> dict[str, list[Any]]:
    """R's ``summary.tmerge`` count matrix as a column-oriented table.

    ``term`` names each operation; the remaining columns count early, late,
    gap, within-interval, boundary and unmatched-ID updates.
    """

    if not isinstance(object, TMergeFrame):
        raise TypeError("summary_tmerge requires a TMergeFrame")
    return {
        "term": list(object.tcount),
        **{name: [row[name] for row in object.tcount.values()] for name in _TMERGE_COUNT_NAMES},
    }


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


def _tmerge_vector(
    value: Any,
    data2: Any,
    n2: int,
    name: str,
    *,
    recycle: bool = False,
    mismatch: str | None = None,
) -> list[Any]:
    """Evaluate a ``tmerge`` argument in ``data2``, as R does: a string names a column.

    The values must line up with ``id`` (the ``mismatch`` error otherwise); only
    ``tstart`` recycles a single value (``recycle``).
    """

    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str):
        if value not in (_data_column_names(data2) or []):
            raise ValueError(f"object '{value}' not found in data2")
        values = _column(data2, value)
    else:
        values = _materialize_1d(value, name) if hasattr(value, "__iter__") else [value]
        if recycle and len(values) == 1:
            values = values * n2
    if len(values) != n2:
        raise ValueError(mismatch or f"argument {name} is not the same length as id")
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


def _storage_mode(values: Sequence[Any] | None) -> str:
    """R's storage mode of an update vector: logical, integer, double or character.

    Without values the updates are R's ``1L``. Missing values alone are R's bare
    ``NA`` (logical) unless they are ``NaN``, which R stores as double.
    """

    if values is None:
        return "integer"
    observed = [value for value in values if not _is_missing_value(value)]
    if any(isinstance(value, str) for value in observed):
        return "character"
    if not observed:
        return "double" if any(isinstance(value, numbers.Real) for value in values) else "logical"
    if all(_is_bool_like(value) for value in observed):
        return "logical"
    if all(isinstance(value, numbers.Integral) for value in observed):
        return "integer"
    return "double"


_CENSOR_VALUES = {"logical": False, "integer": 0, "double": 0.0, "character": ""}
# R's coercion order: combining two modes gives the later one
_STORAGE_MODES = ("logical", "integer", "double", "character")


def _as_mode(value: Any, mode: str) -> Any:
    """``value`` stored in an R vector of storage ``mode`` (``NA`` stays as given)."""

    if _is_missing_value(value) or mode == "logical":
        return value
    if mode == "integer":
        return int(value)
    if mode == "double":
        return float(value)
    return _as_character(value)


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

    ``values`` are the update values as given (``NaN`` for a missing number,
    floats in a double vector), ``numeric`` their float view when every value is
    a number and ``mode`` R's storage mode of the values, which the new variable
    keeps unless a ``tdc`` default changes it.
    """

    name: str
    kind: str
    time: list[float]
    values: list[Any] | None
    numeric: list[float] | None
    mode: str
    default: Any
    censor: Any
    levels: list[Any] | None

    @property
    def missing(self) -> list[bool] | None:
        return None if self.values is None else [_is_missing_value(v) for v in self.values]

    def censor_value(self) -> Any:
        """R's ``tcens`` for a new event variable: the censoring value of its type.

        A factor censors at its first level; ``cumevent`` sums logical events as numbers.
        """

        if self.censor is not None:
            return self.censor
        if self.levels:
            return self.levels[0]
        if self.kind == "cumevent" and self.mode == "logical":
            return _CENSOR_VALUES["double"]
        return _CENSOR_VALUES[self.mode]


def _tmerge_argument(
    name: str, operation: TMergeOperation, data2: Any, n2: int, control: Mapping[str, Any]
) -> _TmergeArgument:
    time = _numeric_or_nan(_tmerge_vector(operation.time, data2, n2, name), f"{name} time")
    values = None if operation.value is None else _tmerge_vector(operation.value, data2, n2, name)
    numeric = None if values is None else _numeric_values(values)
    if operation.kind in {"cumtdc", "cumevent"} and values is not None and numeric is None:
        raise ValueError("invalid increment for cumtdc or cumevent")
    mode = _storage_mode(values)
    if values is not None and numeric is not None:
        values = (
            numeric
            if mode == "double"
            else [math.nan if _is_missing_value(value) else value for value in values]
        )
    default = control["tdcstart"] if operation.default is None else operation.default
    source = operation.value
    if isinstance(source, str):
        source = _column_source(data2, source)
    return _TmergeArgument(
        name=name,
        kind=operation.kind,
        time=time,
        values=values,
        numeric=numeric,
        mode=mode,
        default=default,
        censor=operation.censor,
        levels=None if source is None else _categories(source),
    )


def _tdc_default(argument: _TmergeArgument) -> tuple[Any, str]:
    """R's ``newvar[index == 0] <- default`` for a new ``tdc`` (tmerge.R): the default
    as stored and the storage mode the variable then has.

    Numeric values take ``as.numeric(default)`` and become double; logical and
    character values take the higher of their mode and the default's.
    """

    default, mode = argument.default, argument.mode
    if _is_missing_value(default) or (argument.numeric is None and mode != "character"):
        return default, mode
    if mode in {"integer", "double"}:
        try:
            return float(default), "double"
        except ValueError:
            _warn_outside_package("NAs introduced by coercion")
            return math.nan, "double"
    mode = max(mode, _storage_mode([default]), key=_STORAGE_MODES.index)
    return _as_mode(default, mode), mode


def _tdc_values(step: Any, argument: _TmergeArgument, prior: list[Any] | None) -> list[Any]:
    """R's ``tdc`` update: the value of the last update at or before each interval start."""

    values = argument.values
    sources = step.source
    if prior is None:
        if values is None:
            return [0 if source is None else 1 for source in sources]
        default, mode = _tdc_default(argument)
        if mode != argument.mode and None in sources:
            # R converts the whole variable only when some interval takes the default
            values = [_as_mode(value, mode) for value in values]
        return [default if source is None else values[source] for source in sources]
    if values is None:
        if any(not (_is_missing_value(v) or v in (0, 1, False, True)) for v in prior):
            raise ValueError(f"tdc update does not match prior variable type: {argument.name}")
        return [1 if source is not None else v for source, v in zip(sources, prior, strict=True)]
    return [
        values[source] if source is not None else v
        for source, v in zip(sources, prior, strict=True)
    ]


def _event_values(
    step: Any, argument: _TmergeArgument, prior: list[Any] | None, censor: Any, n_out: int
) -> list[Any]:
    """R's ``event``/``cumevent`` update: events land on the interval they end."""

    values = [censor] * n_out if prior is None else list(prior)
    for row, source, value in zip(step.event_row, step.event_source, step.event_value, strict=True):
        if argument.kind == "cumevent":
            if argument.numeric is not None and math.isnan(argument.numeric[source]):
                # R's newvar[indx2[keep]] <- yinc[keep] stops on the NA that yinc != 0 puts in keep
                raise ValueError(
                    f"argument {argument.name} has a missing cumevent increment at an event time"
                )
            values[row] = (
                int(value) if argument.mode == "integer" and not math.isnan(value) else value
            )
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
            tevent[name] = argument.censor_value()
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
    first: tuple[str, TMergeOperation] | None,
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
        if first is None or first[1].kind != "event":
            raise ValueError("neither a tstop argument nor an initial event argument was found")
        name, operation = first
        times = _numeric_or_nan(_tmerge_vector(operation.time, data2, n2, name), "tstop")
        seen: dict[str, int] = {}
        for row, value in enumerate(id2):
            seen.setdefault(_as_character(value), row)
        range_ids = [id2[row] for row in seen.values()]
        range_stop = [times[row] for row in seen.values()]
    else:
        range_ids = id2
        range_stop = _numeric_or_nan(
            _tmerge_vector(
                tstop, data2, n2, "tstop", mismatch="tstop and id must be the same length"
            ),
            "tstop",
        )
    start_values = (
        None
        if tstart is None
        else _numeric_or_nan(
            _tmerge_vector(
                tstart,
                data2,
                len(range_ids),
                "tstart",
                recycle=True,
                mismatch="tstart and id must be the same length",
            ),
            "tstart",
        )
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
            data1, data2, id, id2, tstart, tstop, next(iter(parsed.items()), None), control
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
