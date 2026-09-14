"""``Surv``/``Surv2`` responses, timeline conversion, formatting, and ``strata``."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby
from typing import Any

from .. import _survival as _core
from ._coerce import (
    _SURV_RESPONSE_TYPES,
    _SURV_TYPES,
    _encode_labels,
    _event_vector,
    _finite_float,
    _float_vector,
    _hashable_group_value,
    _int_vector,
    _interval_endpoint_vector,
    _interval_status_vector,
    _is_bool_like,
    _is_missing_value,
    _keep_rows_after_na_action,
    _materialize_1d,
    _materialize_labels,
    _missing_row_indices,
    _mstate_categories,
    _mstate_event_label,
    _mstate_event_vector,
    _mstate_levels,
    _normalize_bool_option,
    _normalize_na_action,
    _strata_level_sort_key,
    _strata_value_label,
    _subset_sequence,
    _surv_format_number,
)
from ._types import _MISSING, StrataFactor


def _survfit_response_with_etype(response: Surv, etype: Any) -> Surv:
    if response.type not in {"right", "counting"}:
        raise ValueError(
            "etype can only be used with a right-censored or counting-process Surv response"
        )
    raw, levels = _mstate_levels(etype, "etype")
    if len(raw) != len(response):
        raise ValueError("etype must have the same length as the Surv response")

    event_labels = {
        _mstate_event_label(value)
        for value, status in zip(raw, response.event, strict=True)
        if status == 1 and not _is_missing_value(value)
    }
    states = tuple(level for level in levels if level in event_labels)
    state_index = {state: idx + 1 for idx, state in enumerate(states)}
    events: list[int | None] = []
    for value, status in zip(raw, response.event, strict=True):
        if status is None or _is_missing_value(value):
            events.append(None)
        elif status == 0:
            events.append(0)
        else:
            events.append(state_index[_mstate_event_label(value)])
    return Surv._from_normalized(
        time=response.time,
        event=events,
        start=response.start,
        time2=None,
        surv_type="mright" if response.start is None else "mcounting",
        states=states,
    )


def _normalize_surv_type(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("Surv type must be a string")
    normalized = value.strip().lower()
    if normalized in _SURV_TYPES:
        return normalized
    matches = [choice for choice in _SURV_TYPES if choice.startswith(normalized)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError("Surv type is ambiguous; use a full type name")
    raise ValueError(
        "Surv type must be 'right', 'left', 'counting', 'interval', 'interval2', or 'mstate'"
    )


_FORMULA_RESPONSE_ARGUMENT_ALIASES = {
    "event": "event",
    "start": "time",
    "status": "event",
    "stop": "time2",
    "time": "time",
    "time1": "time",
    "time2": "time2",
}


def _formula_response_argument_name(name: str) -> str | None:
    return _FORMULA_RESPONSE_ARGUMENT_ALIASES.get(name.strip().lower())


def _ordered_named_response_arguments(named_arguments: dict[str, str]) -> list[str]:
    if "time" not in named_arguments:
        raise ValueError("named Surv(...) formula response requires time=")
    arguments = [named_arguments["time"]]
    if "time2" in named_arguments:
        arguments.append(named_arguments["time2"])
    if "event" in named_arguments:
        arguments.append(named_arguments["event"])
    return arguments


def _ordered_named_surv_arguments(named_arguments: Mapping[str, Any]) -> tuple[Any, ...]:
    if "time" not in named_arguments:
        raise ValueError("named Surv(...) requires time=, time1=, or start=")
    arguments = [named_arguments["time"]]
    if "time2" in named_arguments:
        arguments.append(named_arguments["time2"])
    if "event" in named_arguments:
        arguments.append(named_arguments["event"])
    return tuple(arguments)


def _collect_named_surv_arguments(arguments: Mapping[str, Any]) -> tuple[Any, ...] | None:
    named_arguments: dict[str, Any] = {}
    for option, value in arguments.items():
        if value is _MISSING:
            continue
        argument_name = _formula_response_argument_name(option)
        if argument_name is None:
            raise TypeError(f"Surv got an unexpected keyword argument {option!r}")
        if argument_name in named_arguments:
            raise ValueError(f"Surv(...) contains multiple {argument_name}= arguments")
        named_arguments[argument_name] = value
    if not named_arguments:
        return None
    return _ordered_named_surv_arguments(named_arguments)


def _derive_interval2_status(left: list[float], right: list[float]) -> list[int]:
    if len(left) != len(right):
        raise ValueError("Surv inputs must have the same length")

    status: list[int] = []
    for idx, (lo, hi) in enumerate(zip(left, right, strict=True)):
        lo_missing = math.isinf(lo) and lo < 0.0
        hi_missing = math.isinf(hi) and hi > 0.0
        if lo_missing and hi_missing:
            raise ValueError("interval2 observations cannot have both endpoints missing")
        if lo_missing:
            status.append(2)
        elif hi_missing:
            status.append(0)
        elif hi < lo:
            raise ValueError(f"interval2 right endpoint is less than left endpoint at index {idx}")
        elif hi == lo:
            status.append(1)
        else:
            status.append(3)
    return status


def _validate_surv_intervals(
    time: list[float],
    time2: list[float] | None,
    event: list[int],
    surv_type: str,
) -> None:
    if surv_type in {"right", "left"}:
        if any(value not in {0, 1} for value in event):
            raise ValueError(f"{surv_type} Surv status must contain only 0/1 values")
        return
    if surv_type == "interval":
        if time2 is None:
            raise ValueError("interval Surv requires time2")
        for idx, status in enumerate(event):
            if status == 3 and time2[idx] < time[idx]:
                raise ValueError(
                    f"interval right endpoint is less than left endpoint at index {idx}"
                )
        return
    if surv_type == "interval2" and time2 is None:
        raise ValueError("interval2 Surv requires time2")


def _validate_surv_time_values(name: str, values: list[float]) -> None:
    for idx, value in enumerate(values):
        if math.isnan(value):
            continue
        if not math.isfinite(value):
            raise ValueError(f"{name} contains non-finite value at index {idx}")


def _validate_surv_time_structure(
    time: list[float],
    time2: list[float] | None,
    event: list[int],
    start: list[float] | None,
    surv_type: str,
) -> None:
    if surv_type == "interval2":
        return

    _validate_surv_time_values("stop" if start is not None else "time", time)
    if time2 is not None:
        if surv_type == "interval":
            for idx, (status, value) in enumerate(zip(event, time2, strict=True)):
                if status == 3 and not (math.isnan(value) or math.isfinite(value)):
                    raise ValueError(f"time2 contains non-finite value at index {idx}")
        else:
            _validate_surv_time_values("time2", time2)
    if start is None:
        return

    _validate_surv_time_values("start", start)
    for idx, (start_value, stop_value) in enumerate(zip(start, time, strict=True)):
        if math.isnan(start_value) or math.isnan(stop_value):
            continue
        if start_value >= stop_value:
            raise ValueError(f"start[{idx}] must be less than stop[{idx}]")


def _turnbull_intervals(response: Surv) -> tuple[list[float], list[float]]:
    left: list[float] = []
    right: list[float] = []
    if response.type == "left":
        for time, event in zip(response.time, response.event, strict=True):
            if event == 1:
                left.append(time)
                right.append(time)
            else:
                left.append(0.0)
                right.append(time)
        return left, right

    if response.type == "interval":
        if response.time2 is None:
            raise ValueError("interval Surv response is missing time2")
        for time, time2, status in zip(
            response.time,
            response.time2,
            response.event,
            strict=True,
        ):
            if status == 0:
                left.append(time)
                right.append(float("inf"))
            elif status == 1:
                left.append(time)
                right.append(time)
            elif status == 2:
                left.append(0.0)
                right.append(time)
            elif status == 3:
                left.append(time)
                right.append(time2)
        return left, right

    if response.type == "interval2":
        if response.time2 is None:
            raise ValueError("interval2 Surv response is missing time2")
        for time, time2, status in zip(
            response.time,
            response.time2,
            response.event,
            strict=True,
        ):
            if status == 2:
                left.append(0.0)
                right.append(time2)
            else:
                left.append(time)
                right.append(time2)
        return left, right

    raise TypeError("Turnbull intervals require left or interval-censored Surv responses")


def _survreg_response_arrays(response: Surv) -> tuple[list[float], list[float], list[float] | None]:
    if response.type == "right":
        return list(response.time), [float(value) for value in response.event], None

    if response.type == "left":
        return (
            list(response.time),
            [1.0 if value == 1 else 2.0 for value in response.event],
            None,
        )

    if response.type == "interval":
        if response.time2 is None:
            raise ValueError("interval Surv response is missing time2")
        return (
            list(response.time),
            [float(value) for value in response.event],
            list(response.time2),
        )

    if response.type == "interval2":
        if response.time2 is None:
            raise ValueError("interval2 Surv response is missing time2")
        time: list[float] = []
        time2: list[float] = []
        for left, right, status in zip(
            response.time,
            response.time2,
            response.event,
            strict=True,
        ):
            if status == 2:
                time.append(right)
                time2.append(right)
            elif status == 0:
                time.append(left)
                time2.append(left)
            else:
                time.append(left)
                time2.append(right)
        return time, [float(value) for value in response.event], time2

    raise NotImplementedError(
        "survreg currently supports right, left, interval, and interval2 Surv responses"
    )


def _strata_is_vector_sequence(value: Any) -> bool:
    if isinstance(value, str | bytes | Mapping):
        return False
    try:
        items = list(value)
    except TypeError:
        return False
    if not items or isinstance(items[0], str | bytes | Mapping):
        return False
    try:
        list(items[0])
    except TypeError:
        return False
    return True


def _strata_legacy_core_call(
    variables: tuple[Any, ...],
    na_group: bool,
    shortlabel: Any,
    sep: str,
    labels: Any,
) -> bool:
    if na_group or shortlabel is not None or sep != ", " or labels is not None:
        return False
    if len(variables) != 1 or not _strata_is_vector_sequence(variables[0]):
        return False
    try:
        columns = [list(column) for column in variables[0]]
    except TypeError:
        return False
    if not columns:
        return False
    for column in columns:
        for value in column:
            if isinstance(value, bool) or _is_missing_value(value):
                return False
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return False
            if not math.isfinite(numeric) or not numeric.is_integer():
                return False
    return True


def _strata_variables_from_args(variables: tuple[Any, ...]) -> list[list[Any]]:
    if not variables:
        raise ValueError("strata requires at least one variable")
    if len(variables) == 1 and _strata_is_vector_sequence(variables[0]):
        variables = tuple(variables[0])
    columns = [
        _materialize_1d(variable, f"variable {idx + 1}") for idx, variable in enumerate(variables)
    ]
    n = len(columns[0])
    if any(len(column) != n for column in columns):
        raise ValueError("all arguments must be the same length")
    return columns


def _strata_column_levels(
    column: Sequence[Any],
    na_group: bool,
) -> tuple[list[Any], list[str], list[int | None]]:
    values: dict[Any, None] = {}
    saw_missing = False
    missing = object()
    normalized: list[Any] = []
    for value in column:
        if _is_missing_value(value):
            saw_missing = True
            normalized.append(missing)
            continue
        try:
            values.setdefault(value, None)
        except TypeError as exc:
            raise TypeError("strata variables must contain hashable values") from exc
        normalized.append(value)
    levels = sorted(values, key=_strata_level_sort_key)
    labels = [_strata_value_label(value) for value in levels]
    if na_group and saw_missing:
        levels.append(None)
        labels.append("NA")
    level_map = {value: idx for idx, value in enumerate(levels)}
    try:
        codes = [
            (level_map[None] if na_group else None) if value is missing else level_map[value]
            for value in normalized
        ]
    except KeyError as exc:
        raise ValueError("missing strata level could not be encoded") from exc
    return levels, labels, codes


def _strata_default_labels(n_terms: int) -> list[str]:
    return [f"v{idx + 1}" for idx in range(n_terms)]


def _strata_normalize_labels(labels: Any, n_terms: int) -> list[str]:
    if labels is None:
        return _strata_default_labels(n_terms)
    result = [str(label) for label in _materialize_1d(labels, "labels")]
    if len(result) != n_terms:
        raise ValueError("labels must have one entry per strata variable")
    return result


def _strata_default_shortlabel(columns: Sequence[Sequence[Any]], labels: Any) -> bool:
    if labels is not None:
        return False
    return all(
        all(_is_missing_value(value) or isinstance(value, str) for value in column)
        for column in columns
    )


def _strata_combination_labels(
    term_labels: Sequence[str],
    column_level_labels: Sequence[Sequence[str]],
    short: bool,
) -> list[list[str]]:
    """Match ``survival::strata`` formatting of component labels."""

    result: list[list[str]] = []
    for term_idx, level_labels in enumerate(column_level_labels):
        pieces = (
            list(level_labels)
            if short
            else [f"{term_labels[term_idx]}={level_label}" for level_label in level_labels]
        )
        if not short and term_idx > 0 and pieces:
            width = max(map(len, pieces))
            pieces = [piece.ljust(width) for piece in pieces]
        result.append(pieces)
    return result


def strata(
    *variables: Any,
    na_group: bool = False,
    shortlabel: bool | None = None,
    sep: str = ", ",
    labels: Any | None = None,
) -> Any:
    """Create R-style strata factor codes and labels."""

    if _strata_legacy_core_call(variables, na_group, shortlabel, sep, labels):
        return _core.strata([[int(value) for value in column] for column in variables[0]])
    if not isinstance(na_group, bool):
        raise TypeError("na_group must be True or False")
    if shortlabel is not None and not isinstance(shortlabel, bool):
        raise TypeError("shortlabel must be True, False, or None")
    if not isinstance(sep, str):
        raise TypeError("sep must be a string")

    columns = _strata_variables_from_args(variables)
    term_labels = _strata_normalize_labels(labels, len(columns))
    short = _strata_default_shortlabel(columns, labels) if shortlabel is None else shortlabel

    column_levels: list[list[Any]] = []
    column_level_labels: list[list[str]] = []
    column_codes: list[list[int | None]] = []
    for column in columns:
        levels, level_labels, codes = _strata_column_levels(column, na_group)
        column_levels.append(levels)
        column_level_labels.append(level_labels)
        column_codes.append(codes)

    codes, observed_parts, counts = _core.strata_compact(
        column_codes,
        [len(levels) for levels in column_levels],
    )
    combination_labels = _strata_combination_labels(
        term_labels,
        column_level_labels,
        short,
    )
    levels: list[str] = []
    for parts in observed_parts:
        pieces = [combination_labels[term_idx][part_idx] for term_idx, part_idx in enumerate(parts)]
        levels.append(sep.join(pieces))
    row_labels = [None if code is None else levels[code - 1] for code in codes]
    return StrataFactor(codes=codes, levels=levels, labels=row_labels, counts=counts)


@dataclass(frozen=True, init=False)
class Surv:
    """Survival response container, like R's Surv."""

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
        time1: Any = _MISSING,
        time2: Any = _MISSING,
        event: Any = _MISSING,
        status: Any = _MISSING,
        start: Any = _MISSING,
        stop: Any = _MISSING,
    ) -> None:
        named_options = {
            "time": time,
            "time1": time1,
            "time2": time2,
            "event": event,
            "status": status,
            "start": start,
            "stop": stop,
        }
        if args and any(value is not _MISSING for value in named_options.values()):
            raise TypeError(
                "Surv(...) must not mix positional and named time/time2/event arguments"
            )
        named_args = _collect_named_surv_arguments(named_options)
        if named_args is not None:
            args = named_args

        surv_type = _normalize_surv_type(type) if type is not None else None
        categorical_event = len(args) in {2, 3} and _mstate_categories(args[-1]) is not None
        if categorical_event and (
            (len(args) == 2 and surv_type in {None, "right", "left", "mstate"})
            or (len(args) == 3 and surv_type in {None, "counting", "mstate"})
        ):
            surv_type = "mstate"
        states: tuple[str, ...] = ()
        origin_value = _finite_float(origin, "origin")
        if len(args) == 1:
            if surv_type is not None:
                raise ValueError("one-argument Surv does not accept an explicit type")
            start = None
            time = [value - origin_value for value in _float_vector(args[0], "time")]
            time2 = None
            event = [1] * len(time)
            surv_type = "right"
        elif len(args) == 2:
            start = None
            if surv_type == "interval2":
                time = [
                    value - origin_value
                    for value in _interval_endpoint_vector(args[0], "time", float("-inf"))
                ]
                time2 = [
                    value - origin_value
                    for value in _interval_endpoint_vector(args[1], "time2", float("inf"))
                ]
                event = _derive_interval2_status(time, time2)
            elif surv_type == "mstate":
                time = [value - origin_value for value in _float_vector(args[0], "time")]
                time2 = None
                event, states = _mstate_event_vector(args[1], "event")
                surv_type = "mright"
            else:
                time = [value - origin_value for value in _float_vector(args[0], "time")]
                time2 = None
                event = _event_vector(args[1], "event")
                if surv_type not in {None, "right", "left"}:
                    raise ValueError(
                        "two-argument Surv supports type='right', 'left', or 'interval2'"
                    )
                surv_type = surv_type or "right"
        elif len(args) == 3:
            if surv_type == "interval":
                start = None
                time = [value - origin_value for value in _float_vector(args[0], "time")]
                time2 = [value - origin_value for value in _float_vector(args[1], "time2")]
                event = _interval_status_vector(args[2], "event")
            elif surv_type == "mstate":
                start = [value - origin_value for value in _float_vector(args[0], "start")]
                time = [value - origin_value for value in _float_vector(args[1], "stop")]
                time2 = None
                event, states = _mstate_event_vector(args[2], "event")
                surv_type = "mcounting"
            else:
                start = [value - origin_value for value in _float_vector(args[0], "start")]
                time = [value - origin_value for value in _float_vector(args[1], "stop")]
                time2 = None
                event = _event_vector(args[2], "event")
                if surv_type not in {None, "counting"}:
                    raise ValueError("three-argument Surv supports type='counting' or 'interval'")
                surv_type = surv_type or "counting"
        else:
            raise TypeError("Surv expects (time), (time, event), or (start, stop, event)")

        if (
            len(time) != len(event)
            or (start is not None and len(start) != len(time))
            or (time2 is not None and len(time2) != len(time))
        ):
            raise ValueError("Surv inputs must have the same length")
        if not time:
            raise ValueError("Surv inputs must not be empty")
        if surv_type not in _SURV_RESPONSE_TYPES:
            raise ValueError(
                "Surv type must be 'right', 'left', 'counting', 'interval', 'interval2', "
                "or 'mstate'"
            )
        _validate_surv_intervals(time, time2, event, surv_type)
        _validate_surv_time_structure(time, time2, event, start, surv_type)

        object.__setattr__(self, "time", tuple(time))
        object.__setattr__(self, "event", tuple(event))
        object.__setattr__(self, "start", tuple(start) if start is not None else None)
        object.__setattr__(self, "time2", tuple(time2) if time2 is not None else None)
        object.__setattr__(self, "type", surv_type)
        object.__setattr__(self, "states", states)

    def __len__(self) -> int:
        return len(self.time)

    @property
    def status(self) -> tuple[int | None, ...]:
        return self.event

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
        result = object.__new__(cls)
        object.__setattr__(result, "time", tuple(time))
        object.__setattr__(result, "event", tuple(event))
        object.__setattr__(result, "start", None if start is None else tuple(start))
        object.__setattr__(result, "time2", None if time2 is None else tuple(time2))
        object.__setattr__(result, "type", surv_type)
        object.__setattr__(result, "states", tuple(states))
        return result


@dataclass(frozen=True, init=False)
class Surv2:
    """Multi-state response container, like R's ``Surv2``."""

    time: tuple[float, ...]
    status: tuple[int | None, ...]
    states: tuple[str, ...]
    repeated: bool | str

    def __init__(self, time: Any, event: Any, repeated: Any = False) -> None:
        time_values = [
            math.nan if _is_missing_value(value) else float(value)
            for value in _materialize_1d(time, "time")
        ]
        event_values = _materialize_1d(event, "event")
        if len(event_values) != len(time_values):
            raise ValueError("Time and event are different lengths")
        repeated_is_first = isinstance(repeated, str) and repeated.lower() == "first"
        if not (_is_bool_like(repeated) or repeated_is_first):
            raise ValueError("invalid value for repeated option")
        repeated_value: bool | str = "first" if repeated_is_first else bool(repeated)

        levels = _surv2_levels(event)
        states = levels[1:]
        if any(state == "" for state in states):
            raise ValueError("each state must have a non-blank name")
        level_index = {level: idx for idx, level in enumerate(levels)}
        status = [
            None if _is_missing_value(value) else level_index[_surv2_event_label(value)]
            for value in event_values
        ]

        object.__setattr__(self, "time", tuple(time_values))
        object.__setattr__(self, "status", tuple(status))
        object.__setattr__(self, "states", tuple(states))
        object.__setattr__(self, "repeated", repeated_value)

    def __len__(self) -> int:
        return len(self.time)


def _surv2_event_label(value: Any) -> str:
    if isinstance(value, bool):
        return "FALSE" if not value else "TRUE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _surv_format_number(value)
    return str(value)


def _surv2_level_sort_key(label: str) -> tuple[int, Any]:
    try:
        numeric = float(label)
    except ValueError:
        return (1, label)
    if math.isfinite(numeric):
        return (0, numeric)
    return (1, label)


def _surv2_levels(events: Any) -> list[str]:
    categories = _mstate_categories(events)
    if categories is not None:
        return [
            _surv2_event_label(value)
            for value in _materialize_1d(categories, "event categories")
            if not _is_missing_value(value)
        ]
    levels: dict[str, None] = {}
    for value in _materialize_1d(events, "event"):
        if not _is_missing_value(value):
            levels.setdefault(_surv2_event_label(value), None)
    return sorted(levels, key=_surv2_level_sort_key)


def _surv2data_status_values(status: Any) -> list[int | None]:
    values: list[int | None] = []
    for value in _materialize_1d(status, "status"):
        if _is_missing_value(value):
            values.append(None)
            continue
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError("Surv2 status values must be integer codes")
        values.append(int(numeric))
    return values


def Surv2data(
    time: Any,
    status: Any,
    *,
    states: Any | None = None,
    repeated: Any = False,
    id: Any,
) -> dict[str, Any]:
    """Convert R ``Surv2`` timeline rows into start-stop transition rows."""

    time_values = _float_vector(time, "time")
    status_values = _surv2data_status_values(status)
    id_values = _materialize_labels(id, "id")
    if len(status_values) != len(time_values) or len(id_values) != len(time_values):
        raise ValueError("time, status, and id must have the same length")
    if any(_is_missing_value(value) for value in id_values) or any(
        math.isnan(value) for value in time_values
    ):
        raise ValueError("id and time cannot be missing")
    state_values = (
        [str(value) for value in _materialize_1d(states, "states")] if states is not None else []
    )

    id_codes = _encode_labels(id_values, "id")
    repeated_is_first = isinstance(repeated, str) and repeated.lower() == "first"
    if repeated_is_first:
        seen_by_id: dict[int, set[int]] = {}
        for row_idx in sorted(
            range(len(time_values)),
            key=lambda idx: (id_codes[idx], time_values[idx]),
        ):
            status_value = status_values[row_idx]
            if status_value in (None, 0):
                continue
            seen = seen_by_id.setdefault(id_codes[row_idx], set())
            if status_value in seen:
                status_values[row_idx] = 0
            else:
                seen.add(status_value)
        repeated_value = True
    else:
        repeated_value = _normalize_bool_option(repeated, "repeated")
    if not state_values:
        order = sorted(
            range(len(time_values)),
            key=lambda idx: (id_codes[idx], time_values[idx]),
        )
        intervals: list[tuple[int, float, float, int, Any]] = []
        for _, grouped_rows_iter in groupby(order, key=lambda idx: id_codes[idx]):
            grouped_rows = list(grouped_rows_iter)
            for current_row, next_row in zip(grouped_rows[:-1], grouped_rows[1:], strict=True):
                next_status = status_values[next_row]
                intervals.append(
                    (
                        current_row,
                        time_values[current_row],
                        time_values[next_row],
                        0 if next_status is None else next_status,
                        id_values[current_row],
                    )
                )
        intervals.sort(key=lambda interval: interval[0])
        rows = [interval[0] for interval in intervals]
        starts = [interval[1] for interval in intervals]
        stops = [interval[2] for interval in intervals]
        output_status = [interval[3] for interval in intervals]
        output_ids = [interval[4] for interval in intervals]
        response_type = "right" if starts and all(value == 0.0 for value in starts) else "counting"
        return {
            "row": rows,
            "start": starts,
            "stop": stops,
            "status": output_status,
            "id": output_ids,
            "istate": [0] * len(rows),
            "states": [],
            "type": response_type,
        }
    result = _core.surv2data_timeline(
        id_codes,
        time_values,
        status_values,
        repeated_value,
    )
    rows = [int(value) for value in result.row_index]
    starts = [float(value) for value in result.start]
    response_type = (
        "mright"
        if state_values and starts and all(value == 0.0 for value in starts)
        else "mcounting"
        if state_values
        else "right"
        if starts and all(value == 0.0 for value in starts)
        else "counting"
    )
    return {
        "row": rows,
        "start": starts,
        "stop": [float(value) for value in result.stop],
        "status": [int(value) for value in result.status],
        "id": [id_values[row] for row in rows],
        "istate": [None if value is None else int(value) for value in result.istate],
        "states": state_values,
        "type": response_type,
    }


def _totimeline_state_values(states: Any | None) -> list[str]:
    return [str(value) for value in _materialize_1d(states, "states")] if states is not None else []


def _totimeline_check_states(
    event_states: Sequence[str],
    istate_levels: Any | None,
) -> list[str]:
    if istate_levels is None:
        return ["(s0)", *event_states]
    levels = [str(value) for value in _materialize_1d(istate_levels, "istate_levels")]
    return [level for level in levels if level not in event_states] + list(event_states)


def _totimeline_istate_codes(
    istate: Any | None,
    check_states: Sequence[str],
    n: int,
) -> list[int]:
    if istate is None:
        return [1] * n
    labels = [
        None if _is_missing_value(value) else str(value)
        for value in _materialize_1d(istate, "istate")
    ]
    if len(labels) != n:
        raise ValueError("istate must have the same length as the Surv response")
    code_by_state = {state: idx + 1 for idx, state in enumerate(check_states)}
    result: list[int] = []
    for label in labels:
        if label is None:
            raise ValueError("istate contains missing values")
        try:
            result.append(code_by_state[label])
        except KeyError as exc:
            raise ValueError(f"istate level {label!r} is not a recognized state") from exc
    return result


def totimeline(
    start: Any,
    stop: Any,
    status: Any,
    *,
    states: Any,
    id: Any,
    istate: Any | None = None,
    istate_levels: Any | None = None,
) -> dict[str, Any]:
    """Convert start-stop multi-state rows into R ``totimeline`` rows."""

    start_values = _float_vector(start, "start")
    stop_values = _float_vector(stop, "stop")
    status_values = [int(value) for value in _int_vector(status, "status")]
    id_values = _materialize_labels(id, "id")
    n = len(start_values)
    if len(stop_values) != n or len(status_values) != n or len(id_values) != n:
        raise ValueError("start, stop, status, and id must have the same length")
    if any(not math.isfinite(value) for value in [*start_values, *stop_values]):
        raise ValueError("start and stop times must be finite")
    event_states = _totimeline_state_values(states)
    if not event_states:
        raise ValueError("states must contain at least one event state")
    check_states = _totimeline_check_states(event_states, istate_levels)
    istate_codes = _totimeline_istate_codes(istate, check_states, n)
    event_code_by_status = {
        status_idx + 1: check_states.index(state) + 1
        for status_idx, state in enumerate(event_states)
    }

    first = []
    seen: set[Any] = set()
    for id_value in id_values:
        key = _hashable_group_value(id_value)
        first.append(key not in seen)
        seen.add(key)

    last = [False] * n
    seen.clear()
    for row_idx in range(n - 1, -1, -1):
        key = _hashable_group_value(id_values[row_idx])
        last[row_idx] = key not in seen
        seen.add(key)

    times: list[float] = []
    state_codes: list[int] = []
    data_rows: list[int] = []
    for row_idx in range(n):
        if first[row_idx]:
            times.append(start_values[row_idx])
            state_codes.append(istate_codes[row_idx])
            data_rows.append(row_idx)

        times.append(stop_values[row_idx])
        status_value = status_values[row_idx]
        if status_value < 0 or status_value > len(event_states):
            raise ValueError("status code is outside the event state range")
        state_codes.append(0 if status_value == 0 else event_code_by_status[status_value])
        data_rows.append(row_idx if last[row_idx] else row_idx + 1)

    state_levels = (
        ["(censor)", *check_states]
        if any(state == "censor" for state in check_states)
        else ["censor", *check_states]
    )
    return {
        "time": times,
        "status": state_codes,
        "data_row": data_rows,
        "state_levels": state_levels,
    }


def _fromtimeline_data_columns(data: Any | None, n: int) -> tuple[list[str], list[list[Any]]]:
    if data is None:
        return [], []
    if not isinstance(data, Mapping):
        raise TypeError("data must be mapping-like")
    names = [str(name) for name in data]
    columns = [_materialize_1d(data[name], str(name)) for name in data]
    for name, column in zip(names, columns, strict=True):
        if len(column) != n:
            raise ValueError(f"{name} must have the same length as the Surv response")
    return names, columns


def _fromtimeline_static_columns(
    columns: Sequence[Sequence[Any]],
    id_values: Sequence[Any],
    column_names: Sequence[str],
    id_name: str,
) -> list[bool]:
    result: list[bool] = []
    for name, column in zip(column_names, columns, strict=True):
        if name == id_name:
            result.append(True)
            continue
        if any(_is_missing_value(value) for value in column):
            result.append(False)
            continue
        first_by_id: dict[Any, Any] = {}
        static = True
        for value, id_value in zip(column, id_values, strict=True):
            key = _hashable_group_value(id_value)
            if key not in first_by_id:
                first_by_id[key] = value
            elif value != first_by_id[key]:
                static = False
                break
        result.append(static)
    return result


def fromtimeline(
    time: Any,
    status: Any,
    *,
    id: Any,
    states: Any | None = None,
    data: Any | None = None,
    id_name: Any = "id",
) -> dict[str, Any]:
    """Convert right-censored timeline rows into R ``fromtimeline`` intervals."""

    time_values = _float_vector(time, "time")
    status_values = [int(value) for value in _int_vector(status, "status")]
    id_values = _materialize_labels(id, "id")
    n = len(time_values)
    if len(status_values) != n or len(id_values) != n:
        raise ValueError("time, status, and id must have the same length")
    if any(not math.isfinite(value) for value in time_values):
        raise ValueError("time values must be finite")
    id_name_value = str(id_name)
    column_names, columns = _fromtimeline_data_columns(data, n)
    static_columns = _fromtimeline_static_columns(columns, id_values, column_names, id_name_value)
    id_codes = _encode_labels(
        [_hashable_group_value(value) for value in id_values],
        "id",
    )
    plan = _core.from_timeline_rows(id_codes, time_values, status_values)
    removed_ids = [id_values[int(row)] for row in plan.removed_row]

    state_values = _totimeline_state_values(states) if states is not None else []
    if state_values:
        state_levels = ["censor", *state_values]
        istate_levels = state_values
    else:
        state_levels = []
        istate_levels = []

    return {
        "start": plan.start,
        "stop": plan.stop,
        "status": plan.status,
        "istate": plan.istate,
        "static": static_columns,
        "static_row": plan.static_row,
        "dynamic_row": plan.dynamic_row,
        "state_levels": state_levels,
        "istate_levels": istate_levels,
        "removed_id": removed_ids,
    }


def is_surv(value: Any) -> bool:
    """Return whether *value* is a survival response object, like R's is.Surv."""

    return isinstance(value, Surv)


def _surv_missing_row(response: Surv, idx: int) -> bool:
    if response.event[idx] is None:
        return True
    if response.start is not None and math.isnan(response.start[idx]):
        return True
    if math.isnan(response.time[idx]):
        return True
    return response.time2 is not None and math.isnan(response.time2[idx])


def is_na_surv(x: Any) -> list[bool]:
    """Return row-wise missingness for a ``Surv`` response, like R's ``is.na.Surv``."""

    if isinstance(x, Surv2):
        return [
            math.isnan(time) or status is None
            for time, status in zip(x.time, x.status, strict=True)
        ]
    if not isinstance(x, Surv):
        raise TypeError("argument is not a Surv object")
    return [_surv_missing_row(x, idx) for idx in range(len(x))]


def _format_surv_right_or_left(response: Surv) -> list[str]:
    suffix = "+" if response.type == "right" else "-"
    times = [_surv_format_number(value) for value in response.time]
    width = max(len(value) for value in times)
    return [
        f"{time.rjust(width)}{' ' if event else suffix}"
        for time, event in zip(times, response.event, strict=True)
    ]


def _format_surv_counting(response: Surv) -> list[str]:
    if response.start is None:
        raise ValueError("counting Surv response is missing start times")
    starts = [_surv_format_number(value) for value in response.start]
    stops = [_surv_format_number(value) for value in response.time]
    start_width = max(len(value) for value in starts)
    stop_width = max(len(value) for value in stops)
    labels = [
        f"({start.rjust(start_width)}, {stop.rjust(stop_width)}{'' if event else '+'}]"
        for start, stop, event in zip(starts, stops, response.event, strict=True)
    ]
    width = max(len(value) for value in labels)
    return [value.ljust(width) for value in labels]


def _format_surv_mstate(response: Surv) -> list[str]:
    suffixes = ["+", *(f":{state}" for state in response.states)]

    def suffix(event: int | None) -> str:
        return "?" if event is None else suffixes[event]

    if response.type == "mright":
        labels = [
            f"{_surv_format_number(time)}{suffix(event)}"
            for time, event in zip(response.time, response.event, strict=True)
        ]
    else:
        if response.start is None:
            raise ValueError("mcounting Surv response is missing start times")
        labels = [
            f"({_surv_format_number(start)},{_surv_format_number(stop)}{suffix(event)}]"
            for start, stop, event in zip(
                response.start,
                response.time,
                response.event,
                strict=True,
            )
        ]
    width = max(len(value) for value in labels) if labels else 0
    return [value.ljust(width) for value in labels]


def _format_surv_interval(response: Surv) -> list[str]:
    if response.time2 is None:
        raise ValueError(f"{response.type} Surv response is missing time2")
    labels: list[str] = []
    for left, right, status in zip(response.time, response.time2, response.event, strict=True):
        left_label = _surv_format_number(left)
        right_label = _surv_format_number(right)
        if status == 0:
            labels.append(f"{left_label}+")
        elif status == 1:
            labels.append(left_label)
        elif status == 2:
            labels.append(f"{right_label}-")
        else:
            labels.append(f"[{left_label}, {right_label}]")
    width = max(len(value) for value in labels)
    return [value.ljust(width) for value in labels]


def _format_surv2(response: Surv2) -> list[str]:
    labels: list[str] = []
    suffixes = ["+", *(f":{state}" for state in response.states)]
    for time, status in zip(response.time, response.status, strict=True):
        suffix = "?" if status is None else suffixes[status]
        labels.append(f"{_surv_format_number(time)}{suffix}")
    width = max(len(value) for value in labels) if labels else 0
    return [value.ljust(width) for value in labels]


def format_surv(x: Any) -> list[str]:
    """Return R-style display strings for a ``Surv`` response."""

    if isinstance(x, Surv2):
        return _format_surv2(x)
    if not isinstance(x, Surv):
        raise TypeError("argument is not a Surv object")
    if x.type in {"right", "left"}:
        return _format_surv_right_or_left(x)
    if x.type == "counting":
        return _format_surv_counting(x)
    if x.type in {"mright", "mcounting"}:
        return _format_surv_mstate(x)
    if x.type in {"interval", "interval2"}:
        return _format_surv_interval(x)
    raise ValueError(f"unsupported Surv type {x.type!r}")


def _subset_surv(response: Surv, indices: list[int]) -> Surv:
    times = [response.time[idx] for idx in indices]
    events = [response.event[idx] for idx in indices]
    if response.type in {"mright", "mcounting"}:
        return Surv._from_normalized(
            time=times,
            event=events,
            start=(None if response.start is None else [response.start[idx] for idx in indices]),
            time2=None,
            surv_type=response.type,
            states=response.states,
        )
    if response.type in {"right", "left"}:
        return Surv(times, events, type=response.type)
    if response.type == "interval":
        if response.time2 is None:
            raise ValueError("interval Surv response is missing time2")
        return Surv(
            times,
            [response.time2[idx] for idx in indices],
            events,
            type="interval",
        )
    if response.type == "interval2":
        if response.time2 is None:
            raise ValueError("interval2 Surv response is missing time2")
        return Surv(times, [response.time2[idx] for idx in indices], type="interval2")
    if response.start is None:
        raise ValueError("counting Surv response is missing start times")
    return Surv([response.start[idx] for idx in indices], times, events)


def _apply_surv_na_action(
    response: Surv,
    na_action: str | None,
    context: str,
    **row_aligned: Any,
) -> tuple[Surv, dict[str, Any]]:
    action = _normalize_na_action(na_action)
    if action == "pass":
        return response, row_aligned

    columns: list[tuple[str, Any]] = [("time", response.time), ("event", response.event)]
    if response.start is not None:
        columns.append(("start", response.start))
    if response.time2 is not None:
        columns.append(("time2", response.time2))
    columns.extend((name, values) for name, values in row_aligned.items() if values is not None)

    keep = _keep_rows_after_na_action(
        _missing_row_indices(columns, len(response)),
        len(response),
        action,
        context,
    )
    if keep is None:
        return response, row_aligned

    filtered = {
        name: _subset_sequence(values, keep, name) if values is not None else None
        for name, values in row_aligned.items()
    }
    return _subset_surv(response, keep), filtered
