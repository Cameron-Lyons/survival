"""R's row operations on ``Surv`` responses, preserving censoring metadata."""

from __future__ import annotations

import math
from itertools import chain
from typing import Any

from ._coerce import (
    _normalize_bool_option,
    _pop_dotted_keyword,
    _scalar_or_vector,
)
from ._surv import Surv


def _response(x: Any) -> Surv:
    if not isinstance(x, Surv):
        raise TypeError("argument is not a Surv object")
    return x


def _columns(x: Surv) -> tuple[tuple[Any, ...], ...]:
    if x.start is not None:
        return x.start, x.time, x.event
    if x.time2 is not None:
        return x.time, x.time2, x.event
    return x.time, x.event


def concat_surv(*objects: Surv) -> Surv:
    """R's ``c.Surv``: join rows with matching censoring type and state levels."""

    if not objects:
        raise ValueError("at least one Surv object is required")
    if any(not isinstance(x, Surv) for x in objects):
        raise TypeError("all elements must be of class Surv")
    first = objects[0]
    if any(x.type != first.type for x in objects[1:]):
        raise ValueError("all elements must be of the same Surv type")
    if any(x.states != first.states for x in objects[1:]):
        raise ValueError("all elements must have the same list of states")
    return Surv._from_normalized(
        time=tuple(chain.from_iterable(x.time for x in objects)),
        event=tuple(chain.from_iterable(x.event for x in objects)),
        start=None if first.start is None else tuple(chain.from_iterable(x.start for x in objects)),
        time2=None if first.time2 is None else tuple(chain.from_iterable(x.time2 for x in objects)),
        surv_type=first.type,
        states=first.states,
        clabel=first.clabel,
    )


def _count(value: Any, name: str) -> int:
    try:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError
        return int(numeric)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid '{name}' argument") from exc


def _scalar(value: Any, name: str) -> float:
    values = _scalar_or_vector(value, name)
    if len(values) != 1:
        raise ValueError(f"'{name}' must have length 1")
    try:
        return float(values[0])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid '{name}' argument") from exc


def rep_surv(
    x: Surv, times: Any = 1, *, each: Any = 1, length_out: Any = None, **kwargs: Any
) -> Surv:
    """R's ``rep.Surv``: repeat rows, retaining type and state levels.

    ``each`` repeats each row before ``times`` repeats the whole sequence (or
    repeats its entries by a vector of counts). ``length_out``/``length.out``
    recycles that sequence to the given length and takes precedence over times.
    Counts truncate towards zero. Empty responses remain empty.
    """

    x = _response(x)
    length_out = _pop_dotted_keyword(kwargs, "length.out", "length_out", length_out, None)
    if kwargs:
        raise TypeError(f"rep_surv got unexpected keyword argument(s): {', '.join(sorted(kwargs))}")
    each_value = _scalar(each, "each")
    each_count = 1 if math.isnan(each_value) else _count(each_value, "each")
    n = len(x)
    length = None if length_out is None else _scalar(length_out, "length.out")
    if length is not None and not math.isnan(length):
        size = _count(length, "length.out")
        if size and not each_count:
            raise ValueError("invalid 'each' argument")
        indices = [(index // each_count) % n for index in range(size)] if n and size else []
    else:
        counts = [_count(value, "times") for value in _scalar_or_vector(times, "times")]
        if len(counts) == 1:
            indices = [row for _ in range(counts[0]) for row in range(n) for _ in range(each_count)]
        elif len(counts) == n * each_count:
            indices = [
                index // each_count for index, count in enumerate(counts) for _ in range(count)
            ]
        else:
            raise ValueError("invalid 'times' argument")
    return x.subset(indices)


def rev_surv(x: Surv) -> Surv:
    """R's ``rev.Surv``: reverse row order without changing censoring codes."""

    x = _response(x)
    return x.subset(range(len(x) - 1, -1, -1))


def duplicated_surv(x: Surv, from_last: bool = False, **kwargs: Any) -> list[bool]:
    """R's ``duplicated.Surv`` for rows, with ``fromLast`` also accepted.

    Missing values compare equal in matching columns, as do signed zeros.
    The scan uses a hash set and takes expected linear time in the row count.
    """

    x = _response(x)
    from_last = _pop_dotted_keyword(kwargs, "fromLast", "from_last", from_last, False)
    if kwargs:
        raise TypeError(
            f"duplicated_surv got unexpected keyword argument(s): {', '.join(sorted(kwargs))}"
        )
    backward = _normalize_bool_option(from_last, "fromLast")
    columns = _columns(x)
    rows = (
        zip(*(reversed(column) for column in columns), strict=True)
        if backward
        else zip(*columns, strict=True)
    )
    seen: set[tuple[Any, ...]] = set()
    result = []
    for row in rows:
        key = tuple(None if value is None or value != value else value for value in row)
        result.append(key in seen)
        seen.add(key)
    if backward:
        result.reverse()
    return result


def unique_surv(x: Surv, from_last: bool = False, **kwargs: Any) -> Surv:
    """R's ``unique.Surv``: retain the first (or last) occurrence of each row."""

    duplicates = duplicated_surv(x, from_last=from_last, **kwargs)
    return x.subset([row for row, duplicate in enumerate(duplicates) if not duplicate])


def transpose_surv(x: Surv) -> list[list[Any]]:
    """R's ``t.Surv``: a plain column-by-observation matrix."""

    return [list(column) for column in _columns(_response(x))]


def levels_surv(x: Surv) -> list[str] | None:
    """R's ``levels.Surv``: multistate event levels, or None for ordinary responses."""

    x = _response(x)
    return list(x.states) if x.type in {"mright", "mcounting"} else None


def _end_count(x: Surv, n: Any, *, tail: bool) -> int:
    value = _scalar(n, "n")
    if math.isnan(value):
        raise ValueError("invalid 'n' argument")
    size = max(len(x) + value, 0) if value < 0 else min(value, len(x))
    # seq_len() truncates for head; seq.int(length.out=) rounds up for tail.
    return math.ceil(size) if tail else int(size)


def head_surv(x: Surv, n: Any = 6) -> Surv:
    """R's ``head.Surv``: the first n rows; negative n drops the last abs(n)."""

    x = _response(x)
    return x.subset(range(_end_count(x, n, tail=False)))


def tail_surv(x: Surv, n: Any = 6) -> Surv:
    """R's ``tail.Surv``: the last n rows; negative n drops the first abs(n)."""

    x = _response(x)
    count = _end_count(x, n, tail=True)
    return x.subset(range(len(x) - count, len(x)))
