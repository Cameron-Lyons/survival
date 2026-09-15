"""``Surv``/``Surv2`` responses, ``strata``, and the timeline conversions.

Ports of ``R/Surv.R``, ``R/Surv2.R``, ``R/strata.R`` and the data side of
``R/fromtimeline.R``: the Python layer builds the response columns the way R's
``Surv`` does (status coding, ``origin``, the ``interval2`` to ``interval``
conversion, multi-state factors) and hands every kernel (``strata``,
``surv2counting``, ``totimeline``) R's inputs.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .. import _survival as _core
from ._coerce import (
    _SURV_RESPONSE_TYPES,
    _SURV_TYPES,
    _as_character,
    _categories,
    _factor,
    _factor_levels,
    _finite_float,
    _is_bool_like,
    _is_missing_value,
    _keep_rows_after_na_action,
    _match_string_arg,
    _materialize_1d,
    _materialize_labels,
    _missing_row_indices,
    _normalize_na_action,
    _r_format_numbers,
    _subset_sequence,
)
from ._types import _MISSING, StrataFactor, Surv2Data, Timeline

# ---------------------------------------------------------------------------
# Surv
# ---------------------------------------------------------------------------


def _normalize_surv_type(value: Any) -> str:
    """R's ``match.arg(type)`` for ``Surv`` (``"mstate"`` accepted, deprecated in R)."""

    return _match_string_arg(
        value,
        "type",
        _SURV_TYPES,
        "Surv type must be 'right', 'left', 'counting', 'interval', 'interval2', or 'mstate'",
    )


def _formula_response_argument_name(name: str) -> str | None:
    """The canonical ``Surv`` argument for a named formula argument (R's names only)."""

    key = name.strip()
    return key if key in {"time", "time2", "event"} else None


def _ordered_named_response_arguments(named_arguments: dict[str, Any]) -> list[Any]:
    """Positional order of ``time=``, ``time2=``, ``event=`` arguments.

    As in R, ``Surv(time, event)`` matches the second argument to ``event``, so a
    call without ``time2`` keeps the two-argument form.
    """

    if "time" not in named_arguments:
        raise ValueError("Must have a time argument")
    arguments = [named_arguments["time"]]
    if "time2" in named_arguments:
        arguments.append(named_arguments["time2"])
    if "event" in named_arguments:
        arguments.append(named_arguments["event"])
    return arguments


def _time_column(values: Any, name: str, message: str) -> list[float]:
    """A numeric time column with ``NaN`` for missing values."""

    result: list[float] = []
    for value in _materialize_1d(values, name):
        if _is_missing_value(value):
            result.append(math.nan)
            continue
        if isinstance(value, str) or _is_bool_like(value):
            raise ValueError(message)
        try:
            result.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(message) from exc
    return result


def _binary_status(values: Any, name: str) -> list[int | None]:
    """R's status coding for right/left/counting data: logical, 0/1 or 1/2."""

    raw = _materialize_1d(values, name)
    if all(_is_bool_like(value) or _is_missing_value(value) for value in raw):
        return [None if _is_missing_value(value) else int(bool(value)) for value in raw]
    numeric: list[float] = []
    for value in raw:
        if _is_missing_value(value):
            numeric.append(math.nan)
            continue
        if isinstance(value, str):
            raise ValueError("Invalid status value, must be logical or numeric")
        try:
            numeric.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid status value, must be logical or numeric") from exc
    observed = [value for value in numeric if not math.isnan(value)]
    if observed and max(observed) == 2.0:
        numeric = [value - 1.0 for value in numeric]
    status: list[int | None] = []
    invalid = False
    for value in numeric:
        if math.isnan(value):
            status.append(None)
        elif value in (0.0, 1.0):
            status.append(int(value))
        else:
            status.append(None)
            invalid = True
    if invalid:
        warnings.warn("Invalid status value, converted to NA", stacklevel=3)
    return status


def _mstate_status(values: Any) -> tuple[list[int | None], tuple[str, ...]]:
    """``as.numeric(as.factor(event)) - 1`` and the states (every level but the first)."""

    codes, labels = _factor(values, "event")
    states = tuple(labels[1:])
    if any(state == "" or state == "NA" for state in states):
        raise ValueError("each state must have a non-blank name")
    return codes, states


def _interval_status(values: Any) -> list[int | None]:
    raw = _materialize_1d(values, "event")
    if any(isinstance(value, str) for value in raw):
        raise ValueError("Invalid status value, must be logical or numeric")
    status: list[int | None] = []
    invalid = False
    for value in raw:
        if _is_missing_value(value):
            status.append(None)
        elif float(value) in (0.0, 1.0, 2.0, 3.0):
            status.append(int(float(value)))
        else:
            status.append(None)
            invalid = True
    if invalid:
        warnings.warn("Status must be 0, 1, 2 or 3; converted to NA", stacklevel=3)
    return status


def _is_factor_like(values: Any) -> bool:
    return _categories(values) is not None


@dataclass(frozen=True, init=False)
class Surv:
    """R's ``Surv`` response object.

    ``time`` and ``event`` are the last two R columns (``NaN``/``None`` for ``NA``);
    counting-process data adds ``start``, interval data ``time2`` (R's dummy ``1``
    where the status is not 3).  ``type`` is one of R's ``right``, ``left``,
    ``interval``, ``counting``, ``mright`` and ``mcounting``; ``states`` lists the
    multi-state levels after the censoring level.
    """

    time: tuple[float, ...]
    event: tuple[int | None, ...]
    start: tuple[float, ...] | None
    time2: tuple[float, ...] | None
    type: str
    states: tuple[str, ...]

    def __init__(
        self,
        *args: Any,
        type: str | None = None,
        origin: Any = 0.0,
        time: Any = _MISSING,
        time2: Any = _MISSING,
        event: Any = _MISSING,
    ) -> None:
        named = {
            name: value
            for name, value in (("time", time), ("time2", time2), ("event", event))
            if value is not _MISSING
        }
        if args and named:
            raise TypeError("Surv(...) must not mix positional and named arguments")
        if named:
            args = tuple(_ordered_named_response_arguments(named))
        if not args:
            raise ValueError("Must have a time argument")
        if len(args) > 3:
            raise TypeError("Surv expects (time), (time, event), or (time, time2, event)")
        ng = len(args)
        mtype = None if type is None else _normalize_surv_type(type)
        if mtype is None or mtype == "mstate":
            surv_type = "counting" if ng == 3 else "right"
        else:
            surv_type = mtype
            if ng != 3 and surv_type in {"interval", "counting"}:
                raise ValueError("Wrong number of args for this type of survival data")
            if ng != 2 and surv_type in {"right", "left", "interval2"}:
                raise ValueError("Wrong number of args for this type of survival data")
        origin_value = _finite_float(origin, "origin")
        columns = self._build(args, surv_type, mtype == "mstate", origin_value)
        for name, value in columns.items():
            object.__setattr__(self, name, value)

    @staticmethod
    def _build(args: tuple[Any, ...], surv_type: str, mstate: bool, origin: float) -> dict:
        time = _time_column(args[0], "time", "Time variable is not numeric")
        nn = len(time)
        start: list[float] | None = None
        time2: list[float] | None = None
        states: tuple[str, ...] = ()
        if len(args) == 1:
            status: list[int | None] = [1] * nn
            surv_type = "right"
        elif surv_type in {"right", "left"}:
            event = args[1]
            if len(_materialize_1d(event, "event")) != nn:
                raise ValueError("Time and status are different lengths")
            if mstate or _is_factor_like(event):
                status, states = _mstate_status(event)
                surv_type = "mright"
            else:
                status = _binary_status(event, "event")
        elif surv_type == "counting":
            start = time
            time = _time_column(args[1], "time2", "Stop time is not numeric")
            if len(time) != nn:
                raise ValueError("Start and stop are different lengths")
            if len(_materialize_1d(args[2], "event")) != nn:
                raise ValueError("Start and event are different lengths")
            backwards = [
                not (math.isnan(a) or math.isnan(b)) and a >= b
                for a, b in zip(start, time, strict=True)
            ]
            if any(backwards):
                start = [
                    math.nan if bad else value for value, bad in zip(start, backwards, strict=True)
                ]
                warnings.warn("Stop time must be > start time, NA created", stacklevel=4)
            if mstate or _is_factor_like(args[2]):
                status, states = _mstate_status(args[2])
                surv_type = "mcounting"
            else:
                status = _binary_status(args[2], "event")
        elif surv_type == "interval2":
            time, time2, status = _interval2_columns(time, args[1])
            surv_type = "interval"
        else:  # interval
            status = _interval_status(args[2])
            if len(status) != nn:
                raise ValueError("Time and status are different lengths")
            time2 = _interval_time2(time, args[1], status)
        if start is not None:
            start = [value - origin for value in start]
        time = [value - origin for value in time]
        if time2 is not None:
            time2 = [
                value - origin if code == 3 else 1.0
                for value, code in zip(time2, status, strict=True)
            ]
        return {
            "time": tuple(time),
            "event": tuple(status),
            "start": None if start is None else tuple(start),
            "time2": None if time2 is None else tuple(time2),
            "type": surv_type,
            "states": states,
        }

    def __len__(self) -> int:
        return len(self.time)

    @property
    def status(self) -> tuple[int | None, ...]:
        return self.event

    @property
    def ncol(self) -> int:
        """R's ``ncol(Surv)``: 2 for (time, status) data, 3 otherwise."""

        return 2 if self.start is None and self.time2 is None else 3

    def as_matrix(self) -> list[list[Any]]:
        """R's ``as.matrix(Surv)``: one row per observation in R's column order."""

        if self.start is not None:
            return [list(row) for row in zip(self.start, self.time, self.event, strict=True)]
        if self.time2 is not None:
            return [list(row) for row in zip(self.time, self.time2, self.event, strict=True)]
        return [list(row) for row in zip(self.time, self.event, strict=True)]

    def replace_times(
        self,
        *,
        time: Sequence[float] | None = None,
        start: Sequence[float] | None = None,
        time2: Sequence[float] | None = None,
    ) -> Surv:
        """The same response with one or more time columns replaced (``aeqSurv``)."""

        return Surv._from_normalized(
            time=self.time if time is None else time,
            event=self.event,
            start=self.start if start is None else start,
            time2=self.time2 if time2 is None else time2,
            surv_type=self.type,
            states=self.states,
        )

    def subset(self, indices: Sequence[int]) -> Surv:
        """R's ``x[i]`` on a ``Surv`` object."""

        rows = list(indices)
        return Surv._from_normalized(
            time=[self.time[idx] for idx in rows],
            event=[self.event[idx] for idx in rows],
            start=None if self.start is None else [self.start[idx] for idx in rows],
            time2=None if self.time2 is None else [self.time2[idx] for idx in rows],
            surv_type=self.type,
            states=self.states,
        )

    @classmethod
    def _from_normalized(
        cls,
        *,
        time: Sequence[float],
        event: Sequence[int | None],
        start: Sequence[float] | None,
        time2: Sequence[float] | None,
        surv_type: str,
        states: Sequence[str] = (),
    ) -> Surv:
        if surv_type not in _SURV_RESPONSE_TYPES:
            raise ValueError(f"unsupported Surv type {surv_type!r}")
        result = object.__new__(cls)
        object.__setattr__(result, "time", tuple(time))
        object.__setattr__(result, "event", tuple(event))
        object.__setattr__(result, "start", None if start is None else tuple(start))
        object.__setattr__(result, "time2", None if time2 is None else tuple(time2))
        object.__setattr__(result, "type", surv_type)
        object.__setattr__(result, "states", tuple(states))
        return result


def _interval2_columns(
    time: list[float], right: Any
) -> tuple[list[float], list[float], list[int | None]]:
    """R's ``interval2`` branch: infer the status and convert to ``interval`` columns."""

    time2 = _time_column(right, "time2", "Time2 must be numeric")
    if len(time2) != len(time):
        raise ValueError("time and time2 are different lengths")
    backwards = [
        not (math.isnan(a) or math.isnan(b)) and a > b for a, b in zip(time, time2, strict=True)
    ]
    time = [value if math.isfinite(value) else math.nan for value in time]
    time2 = [value if math.isfinite(value) else math.nan for value in time2]
    status: list[int | None] = []
    for left, right_value, bad in zip(time, time2, backwards, strict=True):
        if bad or (math.isnan(left) and math.isnan(right_value)):
            status.append(None)
        elif math.isnan(left):
            status.append(2)
        elif math.isnan(right_value):
            status.append(0)
        elif left == right_value:
            status.append(1)
        else:
            status.append(3)
    if any(backwards):
        warnings.warn("Invalid interval: start > stop, NA created", stacklevel=4)
    time = [
        right_value if code == 2 else left
        for left, right_value, code in zip(time, time2, status, strict=True)
    ]
    return time, time2, status


def _interval_time2(time: list[float], right: Any, status: list[int | None]) -> list[float]:
    """R's ``interval`` branch: ``time2`` is only read where the status is 3."""

    if not any(code == 3 for code in status):
        return [1.0] * len(time)
    time2 = _time_column(right, "time2", "Time2 must be numeric")
    if len(time2) != len(time):
        raise ValueError("time and time2 are different lengths")
    backwards = [
        code == 3 and not (math.isnan(a) or math.isnan(b)) and a > b
        for a, b, code in zip(time, time2, status, strict=True)
    ]
    if any(backwards):
        for idx, bad in enumerate(backwards):
            if bad:
                status[idx] = None
        warnings.warn("Invalid interval: start > stop, NA created", stacklevel=4)
    return time2


def is_surv(value: Any) -> bool:
    """R's ``is.Surv``."""

    return isinstance(value, Surv)


def is_na_surv(x: Any) -> list[bool]:
    """R's ``is.na.Surv``/``is.na.Surv2``: rows with a missing entry in any column."""

    if isinstance(x, Surv2):
        return [
            math.isnan(time) or status is None
            for time, status in zip(x.time, x.status, strict=True)
        ]
    if not isinstance(x, Surv):
        raise TypeError("argument is not a Surv object")
    return [
        any(value is None or (isinstance(value, float) and math.isnan(value)) for value in row)
        for row in x.as_matrix()
    ]


def _pad(labels: list[str]) -> list[str]:
    """R's ``format()`` of a character vector: pad to a common width."""

    width = max((len(label) for label in labels), default=0)
    return [label.ljust(width) for label in labels]


def _format_times(values: Sequence[float]) -> list[str]:
    """R's ``format()`` of a numeric column: common decimals and width."""

    return _r_format_numbers(values, 7)


def _event_suffixes(x: Surv | Surv2, censor: str) -> list[str]:
    if x.states:
        return ["+", *(f":{state}" for state in x.states)]
    return [censor, ""]


def format_surv(x: Any) -> list[str]:
    """R's ``format(Surv)`` / ``as.character.Surv``."""

    if isinstance(x, Surv2):
        suffixes = _event_suffixes(x, "+")
        return _pad(
            [
                f"{time}{'?' if status is None else suffixes[status]}"
                for time, status in zip(_format_times(x.time), x.status, strict=True)
            ]
        )
    if not isinstance(x, Surv):
        raise TypeError("argument is not a Surv object")
    if x.type == "interval":
        times = _format_times(x.time)
        times2 = _format_times(x.time2 or ())
        labels = []
        for left, right, status in zip(times, times2, x.event, strict=True):
            if status is None:
                labels.append("NA")
            elif status == 3:
                labels.append(f"[{left}, {right}]")
            else:
                labels.append(f"{left}{['+', '', '-'][status]}")
        return _pad(labels)
    suffixes = _event_suffixes(x, "-" if x.type == "left" else "+")
    marks = ["?" if status is None else suffixes[status] for status in x.event]
    if x.start is None:
        return _pad(
            [f"{time}{mark}" for time, mark in zip(_format_times(x.time), marks, strict=True)]
        )
    return _pad(
        [
            f"({start},{stop}{mark}]"
            for start, stop, mark in zip(
                _format_times(x.start), _format_times(x.time), marks, strict=True
            )
        ]
    )


def _subset_surv(response: Surv, indices: list[int]) -> Surv:
    """Alias of :meth:`Surv.subset` kept for the modules that import it."""

    return response.subset(indices)


def _apply_surv_na_action(
    response: Surv,
    na_action: str | None,
    context: str,
    **row_aligned: Any,
) -> tuple[Surv, dict[str, Any]]:
    """Apply an ``na.action`` to a ``Surv`` response and its row-aligned vectors."""

    action = _normalize_na_action(na_action)
    if action == "pass":
        return response, row_aligned
    columns: list[tuple[str, Any]] = [("response", response.as_matrix())]
    columns.extend((name, values) for name, values in row_aligned.items() if values is not None)
    keep = _keep_rows_after_na_action(
        _missing_row_indices(columns, len(response)), len(response), action, context
    )
    if keep is None:
        return response, row_aligned
    filtered = {
        name: _subset_sequence(values, keep, name) if values is not None else None
        for name, values in row_aligned.items()
    }
    return response.subset(keep), filtered


# --- helpers other modules build on ------------------------------------------


def _survfit_response_with_etype(response: Surv, etype: Any) -> Surv:
    """``survfit``'s old ``etype`` argument: a right/counting response plus event types."""

    if response.type not in {"right", "counting"}:
        raise ValueError(
            "etype can only be used with a right-censored or counting-process Surv response"
        )
    raw = _materialize_labels(etype, "etype")
    if len(raw) != len(response):
        raise ValueError("etype must have the same length as the Surv response")
    levels = _factor_levels(etype, "etype")
    observed = {
        _as_character(value)
        for value, status in zip(raw, response.event, strict=True)
        if status == 1 and not _is_missing_value(value)
    }
    states = tuple(_as_character(level) for level in levels if _as_character(level) in observed)
    index = {state: code + 1 for code, state in enumerate(states)}
    events: list[int | None] = []
    for value, status in zip(raw, response.event, strict=True):
        if status is None or _is_missing_value(value):
            events.append(None)
        elif status == 0:
            events.append(0)
        else:
            events.append(index[_as_character(value)])
    return Surv._from_normalized(
        time=response.time,
        event=events,
        start=response.start,
        time2=None,
        surv_type="mright" if response.start is None else "mcounting",
        states=states,
    )


def _turnbull_intervals(response: Surv) -> tuple[list[float], list[float]]:
    """The ``(left, right]`` intervals of a left/interval-censored response."""

    left: list[float] = []
    right: list[float] = []
    if response.type == "left":
        for time, event in zip(response.time, response.event, strict=True):
            left.append(time if event == 1 else 0.0)
            right.append(time)
        return left, right
    if response.type != "interval":
        raise TypeError("Turnbull intervals require left or interval-censored Surv responses")
    for time, time2, status in zip(
        response.time, response.time2 or (), response.event, strict=True
    ):
        if status == 0:
            left.append(time)
            right.append(math.inf)
        elif status == 1:
            left.append(time)
            right.append(time)
        elif status == 2:
            left.append(0.0)
            right.append(time)
        else:
            left.append(time)
            right.append(time2)
    return left, right


def _survreg_response_arrays(
    response: Surv,
) -> tuple[list[float], list[float], list[float] | None]:
    """R's ``survreg`` response columns: time, status code and (interval data) time2."""

    if response.type == "right":
        return list(response.time), [float(value or 0) for value in response.event], None
    if response.type == "left":
        return (
            list(response.time),
            [1.0 if value == 1 else 2.0 for value in response.event],
            None,
        )
    if response.type == "interval":
        return (
            list(response.time),
            [float(value or 0) for value in response.event],
            list(response.time2 or ()),
        )
    raise NotImplementedError("survreg supports right, left, and interval Surv responses")


# ---------------------------------------------------------------------------
# strata
# ---------------------------------------------------------------------------


def _strata_arguments(variables: tuple[Any, ...]) -> tuple[list[Any], list[str] | None]:
    """R's ``allf`` and its names: the ``...`` arguments, or the one list argument.

    A mapping or data frame supplies both the variables and their names.
    """

    if not variables:
        raise ValueError("all arguments must be vectors")
    if len(variables) == 1:
        (single,) = variables
        if isinstance(single, dict):
            return list(single.values()), [str(key) for key in single]
        columns = getattr(single, "columns", None)
        if columns is not None and hasattr(single, "__getitem__"):
            names = [str(column) for column in columns]
            return [single[column] for column in columns], names
        if not isinstance(single, str | bytes) and not _is_factor_like(single):
            items = list(single) if hasattr(single, "__iter__") else []
            if items and all(
                not isinstance(item, str | bytes)
                and not _is_bool_like(item)
                and hasattr(item, "__iter__")
                for item in items
            ):
                return items, None
    return list(variables), None


def strata(
    *variables: Any,
    na_group: bool = False,
    shortlabel: bool | None = None,
    sep: str = ", ",
    labels: Sequence[str] | None = None,
) -> StrataFactor:
    """R's ``strata(..., na.group, shortlabel, sep)``.

    Python cannot recover the argument expressions R uses as labels, so unnamed
    arguments are called ``v1``, ``v2``, ...; pass ``labels`` (or a mapping) to name
    them.  As in R, ``shortlabel`` defaults to ``True`` when every argument is
    character or factor and no argument is named.
    """

    if not isinstance(na_group, bool):
        raise TypeError("na.group must be TRUE or FALSE")
    if shortlabel is not None and not isinstance(shortlabel, bool):
        raise TypeError("shortlabel must be TRUE, FALSE, or missing")
    if not isinstance(sep, str):
        raise TypeError("sep must be a string")
    columns, names = _strata_arguments(variables)
    nterms = len(columns)
    if labels is not None:
        names = [str(label) for label in _materialize_1d(labels, "labels")]
        if len(names) != nterms:
            raise ValueError("labels must have one entry per strata variable")
    if shortlabel is None:
        shortlabel = names is None and all(
            _is_factor_like(column)
            or all(
                isinstance(value, str) or _is_missing_value(value)
                for value in _materialize_labels(column, "strata")
            )
            for column in columns
        )
    if names is None:
        names = [f"v{idx + 1}" for idx in range(nterms)]
    lengths = {len(_materialize_labels(column, "strata")) for column in columns}
    if len(lengths) > 1:
        raise ValueError("all arguments must be the same length")
    codes: list[list[int | None]] = []
    levels: list[list[str]] = []
    for column in columns:
        column_codes, column_levels = _factor(column, "strata")
        codes.append(column_codes)
        levels.append(column_levels)
    result = _core.strata(names, levels, codes, na_group, shortlabel, sep)
    return StrataFactor(
        codes=list(result.codes),
        levels=list(result.levels),
        labels=[None if code is None else result.levels[code] for code in result.codes],
        counts=list(result.counts),
    )


# ---------------------------------------------------------------------------
# Surv2 and the timeline conversions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, init=False)
class Surv2:
    """R's ``Surv2``: a timeline response of ``(time, event)`` rows per subject."""

    time: tuple[float, ...]
    status: tuple[int | None, ...]
    states: tuple[str, ...]
    repeated: bool | str

    def __init__(self, time: Any, event: Any, repeated: Any = False) -> None:
        time_values = _time_column(time, "time", "Time variable is not numeric")
        if isinstance(repeated, str):
            if repeated.lower() != "first":
                raise ValueError("invalid value for repeated option")
            repeated_value: bool | str = "first"
        elif _is_bool_like(repeated):
            repeated_value = bool(repeated)
        else:
            raise ValueError("invalid value for repeated option")
        if len(_materialize_1d(event, "event")) != len(time_values):
            raise ValueError("Time and event are different lengths")
        states: tuple[str, ...] = ()
        if _is_factor_like(event):
            status, states = _mstate_status(event)
        else:
            status = _binary_status(event, "event")
        object.__setattr__(self, "time", tuple(time_values))
        object.__setattr__(self, "status", tuple(status))
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "repeated", repeated_value)

    def __len__(self) -> int:
        return len(self.time)


def _repeated_option(repeated: Any) -> str:
    if isinstance(repeated, str) and repeated.lower() == "first":
        return "first"
    if _is_bool_like(repeated):
        return "true" if repeated else "false"
    raise ValueError("invalid value for repeated option")


def Surv2data(
    time: Any,
    status: Any,
    *,
    states: Any | None = None,
    repeated: Any = False,
    id: Any,
) -> Surv2Data:
    """The data side of R's ``surv2counting``: timeline rows to counting-process rows.

    ``status`` holds R's integer codes (0 censored, otherwise the state number) and
    ``states`` the state names of a multi-state timeline; the result's ``row`` gives
    the input row each interval starts from.
    """

    time_values = _time_column(time, "time", "Time variable is not numeric")
    status_values: list[int | None] = []
    for value in _materialize_1d(status, "status"):
        if _is_missing_value(value):
            status_values.append(None)
            continue
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError("Surv2 status values must be integer codes")
        status_values.append(int(numeric))
    id_values = _materialize_labels(id, "id")
    if len(status_values) != len(time_values) or len(id_values) != len(time_values):
        raise ValueError("id statement is required")
    if any(_is_missing_value(value) for value in id_values) or any(
        math.isnan(value) for value in time_values
    ):
        raise ValueError("id and time cannot be missing")
    state_names = (
        [] if states is None else [str(value) for value in _materialize_1d(states, "states")]
    )
    result = _core.surv2counting(
        id_values, time_values, status_values, bool(state_names), _repeated_option(repeated)
    )
    kind = "counting" if result.counting else "right"
    return Surv2Data(
        row=list(result.row),
        start=list(result.tstart),
        stop=list(result.tstop),
        status=list(result.status),
        istate=None if result.istate is None else list(result.istate),
        states=state_names,
        type=f"m{kind}" if state_names else kind,
    )


def fromtimeline(
    time: Any,
    status: Any,
    *,
    id: Any,
    states: Any | None = None,
    repeated: Any = False,
) -> Surv2Data:
    """R's ``fromtimeline`` data side: :func:`Surv2data` under its exported name."""

    return Surv2data(time, status, states=states, repeated=repeated, id=id)


def totimeline(
    start: Any,
    stop: Any,
    status: Any,
    *,
    states: Any,
    id: Any,
    istate: Any | None = None,
    istate_levels: Any | None = None,
) -> Timeline:
    """R's ``totimeline`` (draft): counting-process rows to timeline rows.

    Rows of a subject must be consecutive and in time order.  ``status`` holds the
    ``Surv`` codes, ``states`` the response states, and ``istate`` (with its
    ``istate_levels``) the state each subject starts in, ``(s0)`` by default.
    """

    start_values = _time_column(start, "start", "Start time is not numeric")
    stop_values = _time_column(stop, "stop", "Stop time is not numeric")
    status_values = [int(value) for value in _materialize_1d(status, "status")]
    id_values = _materialize_labels(id, "id")
    n = len(start_values)
    if len(stop_values) != n or len(status_values) != n or len(id_values) != n:
        raise ValueError("start, stop, status, and id must have the same length")
    event_states = [str(value) for value in _materialize_1d(states, "states")]
    if not event_states:
        raise ValueError("states must contain at least one event state")
    if istate is None:
        istate_labels = ["(s0)"] * n
        levels = ["(s0)"]
    else:
        istate_labels = [str(value) for value in _materialize_1d(istate, "istate")]
        if len(istate_labels) != n:
            raise ValueError("istate must have the same length as the Surv response")
        levels = (
            [str(level) for level in _factor_levels(istate_labels, "istate")]
            if istate_levels is None
            else [str(value) for value in _materialize_1d(istate_levels, "istate_levels")]
        )
    check_states = [level for level in levels if level not in event_states] + event_states
    code_of = {state: idx + 1 for idx, state in enumerate(check_states)}
    try:
        istate_codes = [code_of[label] for label in istate_labels]
    except KeyError as exc:
        raise ValueError(f"istate level {exc.args[0]!r} is not a recognized state") from exc
    if any(value < 0 or value > len(event_states) for value in status_values):
        raise ValueError("status code is outside the event state range")
    state_codes = [0, *(code_of[state] for state in event_states)]
    result = _core.totimeline(
        id_values,
        start_values,
        stop_values,
        [state_codes[value] for value in status_values],
        istate_codes,
    )
    censor = "(censor)" if "censor" in check_states else "censor"
    return Timeline(
        time=list(result.time),
        status=list(result.state),
        data_row=list(result.covariate_row),
        state_levels=[censor, *check_states],
    )
