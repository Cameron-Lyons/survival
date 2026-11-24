"""``Surv``/``Surv2`` responses, ``strata``/``cluster``, and the timeline conversions.

Ports of ``R/Surv.R``, ``R/Surv2.R``, ``R/strata.R``, ``R/cluster.R`` and ``totimeline``
of ``R/fromtimeline.R``: the Python layer builds the response columns the way R's
``Surv`` does (status coding, ``origin``, the ``interval2`` to ``interval``
conversion, multi-state factors) and hands every kernel (``strata``, ``totimeline``)
R's inputs.  The ``Surv2`` formulas of the model functions and ``fromtimeline`` are
converted in :mod:`survival.r._formula`.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar, cast, overload

import numpy as np

from .. import _survival as _core
from ._coerce import (
    _SURV_RESPONSE_TYPES,
    _SURV_TYPES,
    _as_character,
    _categories,
    _factor,
    _factor_codes,
    _finite_float,
    _first_argument,
    _is_bool_like,
    _is_missing_value,
    _keep_rows_after_na_action,
    _match_string_arg,
    _materialize_1d,
    _materialize_labels,
    _missing_row_indices,
    _normalize_na_action,
    _numeric_factor_array,
    _numeric_ndarray,
    _r_factor,
    _r_format_numbers,
    _subset_sequence,
    _warn_outside_package,
)
from ._surv_ops import _index_rows, _SurvivalOperations
from ._types import _MISSING, StrataFactor


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


def _match_response_arguments(positional: Sequence[Any], named: dict[str, Any]) -> list[Any]:
    """R's matching of ``Surv(time, time2, event)``: the named arguments first, then the
    positional ones fill the remaining formals in order."""

    remaining = [name for name in ("time", "time2", "event") if name not in named]
    if len(positional) > len(remaining):
        raise TypeError("Surv expects (time), (time, event), or (time, time2, event)")
    matched = dict(zip(remaining, positional, strict=False))
    return _ordered_named_response_arguments({**named, **matched})


def _time_column(values: Any, name: str, message: str) -> list[float]:
    """A numeric time column with ``NaN`` for missing values."""

    array = _numeric_ndarray(values)
    if array is not None:
        if array.dtype.kind == "b":
            raise ValueError(message)
        return array.astype(np.float64).tolist()
    raw = _materialize_1d(values, name)
    types = set(map(type, raw))
    if types <= {float, int}:
        return list(map(float, raw))
    if types <= {float, int, type(None)}:
        return [math.nan if value is None else float(value) for value in raw]
    result: list[float] = []
    for value in raw:
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

    array = _numeric_ndarray(values)
    if array is not None:
        if array.dtype.kind == "b":
            return array.astype(np.int64).tolist()
        if array.dtype.kind in "iu":
            if array.size == 0:
                return []
            lo = int(array.min())
            hi = int(array.max())
            if lo >= 0 and hi <= 1:
                return array.tolist()
            if lo >= 1 and hi == 2:
                return (array - 1).tolist()
        coded = array.astype(np.float64, copy=False)
        has_nan = bool(np.isnan(coded).any()) if coded.size else False
        if not has_nan:
            if coded.size and float(coded.max()) == 2.0:
                coded = coded - 1.0
            valid = (coded == 0.0) | (coded == 1.0)
            if valid.all():
                return coded.astype(np.int64).tolist()
            out_codes: list[int | None] = np.where(valid, coded, 0.0).astype(np.int64).tolist()
            for row in np.flatnonzero(~valid).tolist():
                out_codes[row] = None
            _warn_outside_package("Invalid status value, converted to NA")
            return out_codes
        present = ~np.isnan(coded)
        if present.any() and coded[present].max() == 2.0:
            coded = coded - 1.0
        valid = (coded == 0.0) | (coded == 1.0)
        codes: list[int | None] = np.where(valid, coded, 0.0).astype(np.int64).tolist()
        if valid.all():
            return codes
        for row in np.flatnonzero(~valid).tolist():
            codes[row] = None
        if (present & ~valid).any():
            _warn_outside_package("Invalid status value, converted to NA")
        return codes
    raw = _materialize_1d(values, name)
    if raw and set(map(type, raw)) <= {float, int}:
        return _binary_status(np.asarray(raw, dtype=np.float64), name)
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
        _warn_outside_package("Invalid status value, converted to NA")
    return status


def _event_input(values: Any, rows: int, message: str) -> tuple[Any, list[Any] | None]:
    """Inspect event length once, retaining its numeric dtype and declared levels."""

    levels = _categories(values)
    if levels is not None:
        levels = [level for level in levels if not _is_missing_value(level)]
    array = _numeric_ndarray(values) if levels is None or isinstance(values, np.ndarray) else None
    prepared = array if array is not None else _materialize_1d(values, "event")
    if len(prepared) != rows:
        raise ValueError(message)
    return prepared, levels


def _mstate_status(
    values: Any, *, levels: Sequence[Any] | None = None
) -> tuple[list[int | None], tuple[str, ...], str | None]:
    """``as.numeric(as.factor(event)) - 1`` and the states (every level but the first)."""

    codes, labels = _factor(values, "event", levels=levels)
    states = tuple(labels[1:])
    if any(state == "" for state in states):
        raise ValueError("each state must have a non-blank name")
    return codes, states, labels[0] if labels else None


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
        _warn_outside_package("Status must be 0, 1, 2 or 3; converted to NA")
    return status


def _is_factor_like(values: Any) -> bool:
    return _categories(values) is not None


@dataclass(frozen=True, init=False, eq=False)
class Surv(_SurvivalOperations):
    """R's ``Surv`` response object.

    ``time`` and ``event`` are the last two R columns (``NaN``/``None`` for ``NA``);
    counting-process data adds ``start``, interval data ``time2`` (R's dummy ``1``
    where the status is not 3).  ``type`` is one of R's ``right``, ``left``,
    ``interval``, ``counting``, ``mright`` and ``mcounting``; ``states`` lists the
    multi-state levels after the censoring level.
    ``clabel`` preserves the censoring factor level used by R 3.8-12's labels.
    """

    time: tuple[float, ...]
    event: tuple[int | None, ...]
    start: tuple[float, ...] | None
    time2: tuple[float, ...] | None
    type: str
    states: tuple[str, ...]
    clabel: str | None

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
        if named:
            args = tuple(_match_response_arguments(args, named))
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
        columns = self._build(args, surv_type, mtype == "mstate", _origin_values(origin))
        for name, value in columns.items():
            object.__setattr__(self, name, value)

    @staticmethod
    def _build(
        args: tuple[Any, ...], surv_type: str, mstate: bool, origin: float | list[float]
    ) -> dict:
        time = _time_column(args[0], "time", "Time variable is not numeric")
        nn = len(time)
        start: list[float] | None = None
        time2: list[float] | None = None
        states: tuple[str, ...] = ()
        clabel: str | None = None
        if len(args) == 1:
            status: list[int | None] = [1] * nn
            surv_type = "right"
        elif surv_type in {"right", "left"}:
            event, levels = _event_input(args[1], nn, "Time and status are different lengths")
            if mstate or levels is not None:
                status, states, clabel = _mstate_status(event, levels=levels)
                surv_type = "mright"
            else:
                status = _binary_status(event, "event")
        elif surv_type == "counting":
            start = time
            time = _time_column(args[1], "time2", "Stop time is not numeric")
            if len(time) != nn:
                raise ValueError("Start and stop are different lengths")
            event, levels = _event_input(args[2], nn, "Start and event are different lengths")
            # A comparison with NaN is false, so missing endpoints are never backwards.
            start_values = np.asarray(start, dtype=np.float64)
            backwards = start_values >= np.asarray(time, dtype=np.float64)
            if backwards.any():
                start = np.where(backwards, np.nan, start_values).tolist()
                _warn_outside_package("Stop time must be > start time, NA created")
            if mstate or levels is not None:
                status, states, clabel = _mstate_status(event, levels=levels)
                surv_type = "mcounting"
            else:
                status = _binary_status(event, "event")
        elif surv_type == "interval2":
            time, time2, status = _interval2_columns(time, args[1])
            surv_type = "interval"
        else:
            status = _interval_status(args[2])
            if len(status) != nn:
                raise ValueError("Time and status are different lengths")
            time2 = _interval_time2(time, args[1], status)
        if isinstance(origin, list):
            if len(origin) != nn:
                raise ValueError("origin must have length 1 or the length of time")
            if any(origin):
                if start is not None:
                    start = [value - shift for value, shift in zip(start, origin, strict=True)]
                time = [value - shift for value, shift in zip(time, origin, strict=True)]
            if time2 is not None:
                time2 = [
                    value - shift if code == 3 else 1.0
                    for value, code, shift in zip(time2, status, origin, strict=True)
                ]
        else:
            if origin != 0.0:
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
            "clabel": clabel,
        }

    def __len__(self) -> int:
        return len(self.time)

    def __getitem__(self, index: Any) -> Surv:
        """R's ``x[i]`` with Python's zero-based positions, slices or a logical mask."""

        return self.subset(_index_rows(index, len(self)))

    @property
    def status(self) -> tuple[int | None, ...]:
        return self.event

    @property
    def ncol(self) -> int:
        """R's ``ncol(Surv)``: 2 for (time, status) data, 3 otherwise."""

        return 2 if self.start is None and self.time2 is None else 3

    def _event_codes(self) -> list[int]:
        """Owned, complete status codes for kernels that cannot accept missing rows."""
        if None in self.event:
            raise ValueError("missing values in the response")
        return list(cast(tuple[int, ...], self.event))

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
            clabel=self.clabel,
        )

    def subset(self, indices: Sequence[int]) -> Surv:
        """R's ``x[i]`` on a ``Surv`` object."""

        rows = list(indices)
        return Surv._from_normalized(
            time=tuple(map(self.time.__getitem__, rows)),
            event=tuple(map(self.event.__getitem__, rows)),
            start=None if self.start is None else tuple(map(self.start.__getitem__, rows)),
            time2=None if self.time2 is None else tuple(map(self.time2.__getitem__, rows)),
            surv_type=self.type,
            states=self.states,
            clabel=self.clabel,
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
        clabel: str | None = None,
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
        object.__setattr__(result, "clabel", clabel)
        return result


def _origin_values(origin: Any) -> float | list[float]:
    """``Surv``'s ``origin``: one number, or one per observation."""

    if isinstance(origin, str) or not hasattr(origin, "__iter__"):
        return _finite_float(origin, "origin")
    values = [_finite_float(value, "origin") for value in _materialize_1d(origin, "origin")]
    return values[0] if len(values) == 1 else values


def _interval2_columns(
    time: list[float], right: Any
) -> tuple[list[float], list[float], list[int | None]]:
    """R's ``interval2`` branch: infer the status and convert to ``interval`` columns."""

    time2 = _time_column(right, "time2", "Time2 must be numeric")
    if len(time2) != len(time):
        raise ValueError("time and time2 are different lengths")
    backwards = False
    status: list[int | None] = []
    for row, (left, upper) in enumerate(zip(time, time2, strict=True)):
        finite_upper = math.isfinite(upper)
        if not finite_upper:
            time2[row] = math.nan
        if not math.isfinite(left):
            time[row] = time2[row]
            code = 2 if finite_upper else None
        elif not finite_upper:
            code = 0
        else:
            code = 1 if left == upper else 3
        if left > upper:
            backwards = True
            code = None
        status.append(code)
    if backwards:
        _warn_outside_package("Invalid interval: start > stop, NA created")
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
        _warn_outside_package("Invalid interval: start > stop, NA created")
    return time2


def is_surv(value: Any = _MISSING, **kwargs: Any) -> bool:
    """R's ``is.Surv`` (R's argument name ``x`` is accepted)."""

    value = _first_argument(value, kwargs, "value", "x", "is_surv")
    if kwargs:
        raise TypeError(f"is_surv got unexpected keyword argument(s): {', '.join(sorted(kwargs))}")
    return isinstance(value, Surv)


def _missing_rows(x: Surv | Surv2) -> np.ndarray:
    """``is.na(x)`` as a boolean array: a missing entry in any column of the response."""

    columns: tuple[Sequence[Any] | None, ...] = (
        (x.time, x.status) if isinstance(x, Surv2) else (x.start, x.time, x.time2, x.event)
    )
    missing = np.zeros(len(x), dtype=bool)
    for column in columns:
        if column is None:
            continue
        try:
            # A NaN entry makes the sum NaN, so a sum that is not NaN proves the column
            # complete; a missing status (None) or inf - inf takes the array scan.
            if not math.isnan(sum(column)):
                continue
        except (TypeError, OverflowError):
            pass
        missing |= np.isnan(np.array(column, dtype=np.float64))
    return missing


def is_na_surv(x: Any) -> list[bool]:
    """R's ``is.na.Surv``/``is.na.Surv2``: rows with a missing entry in any column."""

    if not isinstance(x, Surv | Surv2):
        raise TypeError("argument is not a Surv object")
    return _missing_rows(x).tolist()


def _pad(labels: list[str]) -> list[str]:
    """R's ``format()`` of a character vector: pad to a common width."""

    width = max((len(label) for label in labels), default=0)
    return [label.ljust(width) for label in labels]


def _format_times(values: Sequence[float]) -> list[str]:
    """R's ``format()`` of a numeric column: common decimals and width."""

    return _r_format_numbers(values, 7)


def _event_suffixes(x: Surv | Surv2, censor: str) -> list[str]:
    if x.states or x.clabel is not None:
        return ["+" if x.clabel is None else f":{x.clabel}", *(f":{state}" for state in x.states)]
    return [censor, ""]


def format_surv(x: Any) -> list[str]:
    """R's ``format(Surv)``: character labels padded to a common width."""

    return _pad(as_character_surv(x))


def as_character_surv(x: Any) -> list[str]:
    """R's ``as.character.Surv`` / ``as.character.Surv2``.

    Numeric time columns use R's common precision and width, but the resulting
    labels have no extra right padding (unlike :func:`format_surv`).
    """

    if isinstance(x, Surv2):
        suffixes = _event_suffixes(x, "+")
        return [
            f"{time}{'?' if status is None else suffixes[status]}"
            for time, status in zip(_format_times(x.time), x.status, strict=True)
        ]
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
        return labels
    suffixes = _event_suffixes(x, "-" if x.type == "left" else "+")
    marks = ["?" if status is None else suffixes[status] for status in x.event]
    if x.start is None:
        return [f"{time}{mark}" for time, mark in zip(_format_times(x.time), marks, strict=True)]
    return [
        f"({start},{stop}{mark}]"
        for start, stop, mark in zip(
            _format_times(x.start), _format_times(x.time), marks, strict=True
        )
    ]


@overload
def _subset_surv(response: Surv, indices: list[int]) -> Surv: ...


@overload
def _subset_surv(response: Surv2, indices: list[int]) -> Surv2: ...


def _subset_surv(response: Surv | Surv2, indices: list[int]) -> Surv | Surv2:
    """Internal row selection, with -1 for an R model-frame missing row."""

    if isinstance(response, Surv2):
        if any(row < 0 for row in indices):
            return Surv2._from_normalized(
                time=[math.nan if row < 0 else response.time[row] for row in indices],
                status=[None if row < 0 else response.status[row] for row in indices],
                states=response.states,
                repeated=response.repeated,
                clabel=response.clabel,
            )
        return response.subset(indices)
    if any(row < 0 for row in indices):
        return Surv._from_normalized(
            time=[math.nan if row < 0 else response.time[row] for row in indices],
            event=[None if row < 0 else response.event[row] for row in indices],
            start=(
                None
                if response.start is None
                else [math.nan if row < 0 else response.start[row] for row in indices]
            ),
            time2=(
                None
                if response.time2 is None
                else [math.nan if row < 0 else response.time2[row] for row in indices]
            ),
            surv_type=response.type,
            states=response.states,
            clabel=response.clabel,
        )
    return response.subset(indices)


def _apply_surv_na_action(
    response: Surv,
    na_action: str | None,
    context: str,
    **row_aligned: Any,
) -> tuple[Surv, dict[str, Any], list[int]]:
    """Filter a response and aligned vectors; return omitted zero-based row indices."""

    action = _normalize_na_action(na_action)
    if action == "pass":
        return response, row_aligned, []
    columns = [(name, values) for name, values in row_aligned.items() if values is not None]
    missing = _missing_row_indices(columns, len(response))
    missing.update(np.flatnonzero(_missing_rows(response)).tolist())
    keep = _keep_rows_after_na_action(missing, len(response), action, context)
    if keep is None:
        return response, row_aligned, []
    filtered = {
        name: _subset_sequence(values, keep, name) if values is not None else None
        for name, values in row_aligned.items()
    }
    return response.subset(keep), filtered, sorted(missing)


def _survreg_response_arrays(
    response: Surv,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """R's ``survreg`` response columns: time, status code and (interval data) time2.

    The columns are float64 and int32 arrays, which the native fit reads without a
    per-element conversion; a missing (``None``) status cannot be stored as int32.
    """

    try:
        event = np.fromiter(response.event, np.int32, count=len(response))
    except TypeError:
        raise ValueError("missing values in the response") from None
    time = np.fromiter(response.time, np.float64, count=len(response))
    if response.type == "right":
        return time, event, None
    if response.type == "left":
        return time, np.where(event == 1, 1, 2).astype(np.int32), None
    if response.type == "interval":
        return time, event, np.fromiter(response.time2 or (), np.float64)
    raise NotImplementedError("survreg supports right, left, and interval Surv responses")


def _strata_iterator(values: Any) -> Any:
    """Materialize a one-shot column once, retaining its declared factor levels."""

    if not isinstance(values, Iterator):
        return values
    levels = _categories(values)
    materialized = list(values)
    return materialized if levels is None else _r_factor(materialized, levels)


def _strata_arguments(variables: tuple[Any, ...]) -> tuple[list[Any], list[str] | None]:
    """R's ``allf`` and its names: the ``...`` arguments, or the one list argument.

    A mapping or data frame supplies both the variables and their names.
    """

    if not variables:
        raise ValueError("all arguments must be vectors")
    if len(variables) == 1:
        (single,) = variables
        single = _strata_iterator(single)
        variables = (single,)
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
    arguments are called ``v1``, ``v2``, ...; pass ``labels`` to stand in for those
    expressions, or a mapping to name the arguments (R's ``strata(a = x)``).  As in
    R, ``shortlabel`` defaults to ``True`` when every argument is character or
    factor and no argument is named.
    """

    if not isinstance(na_group, bool):
        raise TypeError("na.group must be TRUE or FALSE")
    if shortlabel is not None and not isinstance(shortlabel, bool):
        raise TypeError("shortlabel must be TRUE, FALSE, or missing")
    if not isinstance(sep, str):
        raise TypeError("sep must be a string")
    columns, names = _strata_arguments(variables)
    nterms = len(columns)
    if shortlabel is None and names is not None:
        shortlabel = False
    if labels is not None:
        names = [str(label) for label in _materialize_1d(labels, "labels")]
        if len(names) != nterms:
            raise ValueError("labels must have one entry per strata variable")
    if names is None:
        names = [f"v{idx + 1}" for idx in range(nterms)]
    return _strata(
        list(zip(names, columns, strict=True)),
        shortlabel=shortlabel,
        na_group=na_group,
        sep=sep,
        _lazy=False,
    )


def _is_character(values: Any) -> bool:
    """``is.character(x) | is.factor(x)``, with missing values allowed."""

    if _is_factor_like(values):
        return True
    if _numeric_ndarray(values) is not None:
        return False
    return all(
        isinstance(value, str) or _is_missing_value(value)
        for value in _materialize_labels(values, "strata")
    )


def _strata(
    columns: Sequence[tuple[str, Any]],
    *,
    shortlabel: bool | None = None,
    na_group: bool = False,
    sep: str = ", ",
    _lazy: bool = True,
) -> StrataFactor:
    """R's ``strata()`` (R/strata.R) of the named *columns*, each coded as ``factor()``
    codes it (a factor keeps its declared levels, unused ones included, which the
    ``format()`` padding of the labels counts).

    ``shortlabel=None`` is R's default: bare levels when every column is character or
    a factor.
    """

    names = [name for name, _values in columns]
    prepared: dict[int, tuple[np.ndarray | list[int | None], list[str], bool]] = {}
    coded = []
    character_columns = []
    single_complete_codes: np.ndarray | None = None
    for _name, source in columns:
        # One source may supply several named columns, including a shared iterator.
        # Keep this cache local so later calls see mutations to reusable inputs.
        key = id(source)
        if key not in prepared:
            if len(columns) == 1 and _categories(source) is None:
                array = _numeric_ndarray(source)
                if array is not None:
                    codes_arr, num_levels, counts_arr = _numeric_factor_array(array)
                    levels = [_as_character(level) for level in num_levels.tolist()]
                    if counts_arr is not None:
                        single_complete_codes = codes_arr
                    prepared[key] = codes_arr, levels, False
            if key not in prepared:
                values = _strata_iterator(source)
                codes, levels = _factor_codes(values, "strata")
                character = _is_character(values) if shortlabel is None else False
                prepared[key] = codes, levels, character
        codes, levels, character = prepared[key]
        coded.append((codes, levels))
        character_columns.append(character)
    if len({len(codes) for codes, _levels in coded}) > 1:
        raise ValueError("all arguments must be the same length")
    if shortlabel is None:
        shortlabel = all(character_columns)
    result = _core.strata(
        names,
        [levels for _codes, levels in coded],
        [codes for codes, _levels in coded],
        na_group,
        shortlabel,
        sep,
    )
    if _lazy and single_complete_codes is not None:
        return StrataFactor(
            codes=None,
            levels=result.levels,
            labels=None,
            counts=result.counts,
            _codes_array=single_complete_codes,
        )
    codes_out = result.codes
    levels_out = result.levels
    labels_out = (
        None if _lazy else [None if code is None else levels_out[code] for code in codes_out]
    )
    return StrataFactor(
        codes=codes_out,
        levels=levels_out,
        labels=labels_out,
        counts=result.counts,
    )


def _complete_codes(factor: StrataFactor, message: str) -> list[int]:
    """The codes of *factor*, which a model needs without ``NA`` (*message* otherwise)."""

    arr = factor.__dict__.get("_codes_array")
    if arr is not None and "codes" not in factor.__dict__:
        if (arr < 0).any():
            raise ValueError(message)
        return cast(list[int], factor.codes)
    if None in factor.codes:
        raise ValueError(message)
    return cast(list[int], factor.codes)


def _complete_codes_fast(factor: StrataFactor, message: str) -> Any:
    """The complete codes of *factor*, returning a contiguous ``int64`` ndarray when available."""

    arr = factor.__dict__.get("_codes_array")
    if arr is not None:
        if (arr < 0).any():
            raise ValueError(message)
        return arr
    return _complete_codes(factor, message)


_T = TypeVar("_T")


def cluster(x: _T) -> _T:
    """R's ``cluster`` (R/cluster.R): the identity.  Its meaning comes from the model
    formula, where ``cluster(id)`` names the groups of a robust variance."""

    return x


@dataclass(frozen=True, init=False, eq=False)
class Surv2(_SurvivalOperations):
    """R's ``Surv2``: a timeline response of ``(time, event)`` rows per subject."""

    time: tuple[float, ...]
    status: tuple[int | None, ...]
    states: tuple[str, ...]
    repeated: bool | str
    clabel: str | None

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
        event, levels = _event_input(
            event, len(time_values), "Time and event are different lengths"
        )
        states: tuple[str, ...] = ()
        clabel: str | None = None
        if levels is not None:
            status, states, clabel = _mstate_status(event, levels=levels)
        else:
            status = _binary_status(event, "event")
        object.__setattr__(self, "time", tuple(time_values))
        object.__setattr__(self, "status", tuple(status))
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "repeated", repeated_value)
        object.__setattr__(self, "clabel", clabel)

    def __len__(self) -> int:
        return len(self.time)

    def __getitem__(self, index: Any) -> Surv2:
        """R's ``x[i]`` with Python's zero-based positions, slices or a logical mask."""

        return self.subset(_index_rows(index, len(self)))

    def as_matrix(self) -> list[list[float | int | None]]:
        """Extract the time and normalized status columns for explicit numeric use."""
        return [list(row) for row in zip(self.time, self.status, strict=True)]

    def subset(self, indices: Sequence[int]) -> Surv2:
        """R's ``x[i]`` on a ``Surv2`` object."""

        rows = list(indices)
        return Surv2._from_normalized(
            time=tuple(map(self.time.__getitem__, rows)),
            status=tuple(map(self.status.__getitem__, rows)),
            states=self.states,
            repeated=self.repeated,
            clabel=self.clabel,
        )

    @classmethod
    def _from_normalized(
        cls,
        *,
        time: Sequence[float],
        status: Sequence[int | None],
        states: Sequence[str] = (),
        repeated: bool | str = False,
        clabel: str | None = None,
    ) -> Surv2:
        result = object.__new__(cls)
        object.__setattr__(result, "time", tuple(time))
        object.__setattr__(result, "status", tuple(status))
        object.__setattr__(result, "states", tuple(states))
        object.__setattr__(result, "repeated", repeated)
        object.__setattr__(result, "clabel", clabel)
        return result

    def replace_times(self, *, time: Sequence[float]) -> Surv2:
        """The same response with its time column replaced (``aeqSurv``)."""

        result = copy.copy(self)
        object.__setattr__(result, "time", tuple(time))
        return result


def _repeated_option(repeated: Any) -> str:
    """R's ``repeated`` option (``FALSE``, ``TRUE`` or ``"first"``) as the
    ``surv2counting`` kernel spells it."""

    if isinstance(repeated, str) and repeated.lower() == "first":
        return "first"
    if _is_bool_like(repeated):
        return "true" if repeated else "false"
    raise ValueError("invalid value for repeated option")
