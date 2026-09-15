"""``finegray``: the Fine-Gray weighted data set for competing risks.

A port of the R code of ``R/finegray.R``: the model frame, the multi-state
response checks, the censoring distribution (``survfitkm`` on the ranked
times) and the per-stratum split (the ``finegray`` kernel).
"""

from __future__ import annotations

import math
import re
import warnings
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .. import _survival as _core
from ._coerce import (
    _finite_float,
    _materialize_1d,
    _normalize_bool_option,
    _pop_dotted_keyword,
)
from ._data_prep import aeqSurv
from ._formula import _column, _model_variables, model_frame
from ._surv import Surv, strata
from ._types import FineGrayFrame, ModelFrame


@dataclass(frozen=True)
class _CensoringCurve:
    """One stratum of R's ``Gsurv``/``Hsurv``: the event times and survival of a KM fit."""

    time: list[float]
    surv: list[float]


def _censoring_curves(
    start: Sequence[float], stop: Sequence[float], event: Sequence[bool], group: Sequence[int]
) -> list[_CensoringCurve]:
    """``survfit(Surv(start, stop, event) ~ group)`` restricted to its event times."""

    fit = _core.survfitkm(
        list(stop),
        [int(flag) for flag in event],
        start=list(start),
        strata=list(group),
        se_fit=False,
        timefix=False,
    )
    counts = fit.strata or [len(fit.time)]
    curves: list[_CensoringCurve] = []
    offset = 0
    for count in counts:
        rows = range(offset, offset + count)
        curves.append(
            _CensoringCurve(
                time=[fit.time[row] for row in rows if fit.n_event[row] > 0],
                surv=[fit.surv[row] for row in rows if fit.n_event[row] > 0],
            )
        )
        offset += count
    return curves


def _subject_layout(
    response: Surv, id_values: Sequence[Any] | None
) -> tuple[list[bool], list[bool], bool]:
    """R's ``first``/``last`` row flags of each subject and whether entry is delayed."""

    n = len(response)
    if response.type != "mcounting":
        return [False] * n, [True] * n, False
    if id_values is None:
        raise ValueError("(start, stop] data requires a subject id")
    order = sorted(range(n), key=lambda idx: (str(id_values[idx]), response.time[idx]))
    first = [False] * n
    last = [False] * n
    start = response.start or ()
    delay = False
    minimum_stop = min(response.time)
    previous: int | None = None
    for position, idx in enumerate(order):
        subject_start = previous is None or str(id_values[order[position - 1]]) != str(
            id_values[idx]
        )
        if subject_start:
            first[idx] = True
            if start[idx] > minimum_stop:
                delay = True
            if previous is not None:
                last[previous] = True
        else:
            if response.event[previous] != 0:
                raise ValueError("a subject has a transition before their last time point")
            if response.time[previous] != start[idx]:
                raise ValueError("a subject has gaps in time")
        previous = idx
    if previous is not None:
        last[previous] = True
    return first, last, delay


def _make_names(value: str) -> str:
    """R's ``make.names`` for one name."""

    name = re.sub(r"[^A-Za-z0-9._]", ".", value)
    if not name or not (name[0].isalpha() or (name[0] == "." and not name[1:2].isdigit())):
        name = f"X{name}"
    reserved = {
        "if",
        "else",
        "repeat",
        "while",
        "function",
        "for",
        "next",
        "break",
        "TRUE",
        "FALSE",
        "NULL",
        "Inf",
        "NaN",
        "NA",
        "NA_integer_",
        "NA_real_",
        "NA_character_",
        "NA_complex_",
    }
    return f"{name}." if name in reserved else name


def _etype_index(states: Sequence[str], etype: Any) -> int:
    """R's ``match(etype, states)[1]`` (one-based) with its checks and warning."""

    if etype is None:
        return 1
    requested = (
        [etype] if isinstance(etype, str) else [str(v) for v in _materialize_1d(etype, "etype")]
    )
    index = [states.index(value) + 1 if value in states else None for value in requested]
    if any(value is None for value in index) or not index:
        raise ValueError("etype argument has a state that is not in the data")
    if len(index) > 1:
        warnings.warn("only the first endpoint was used", stacklevel=3)
    return int(index[0] or 1)


def _finegray_inputs(mf: ModelFrame, timefix: bool) -> tuple[Surv, list[int], list[float]]:
    """The checked response, the stratum of each row and the user weights."""

    response = mf.response
    if response is None:
        raise ValueError("Response must be a survival object")
    if response.type not in {"mright", "mcounting"}:
        raise ValueError("Fine-Gray model requires a multi-state survival")
    if len(response.states) < 2:
        raise ValueError("survival time has only a single state")
    if any(value is None for value in response.event) or any(
        math.isnan(value) for value in (*response.time, *(response.start or ()))
    ):
        raise ValueError("missing values in the response")
    if timefix:
        response = aeqSurv(response)
    if mf.terms.clusters:
        raise ValueError("a cluster() term is not valid")
    if mf.terms.strata:
        factor = strata(
            *[_column(mf.data, column) for column in mf.terms.strata],
            labels=list(mf.terms.strata),
            shortlabel=True,
        )
        if any(code is None for code in factor.codes):
            raise ValueError("strata must not contain missing values")
        istrat = [int(code) for code in factor.codes if code is not None]
    else:
        istrat = [0] * mf.n
    weights = (
        [1.0] * mf.n if mf.weights is None else [_finite_float(v, "weights") for v in mf.weights]
    )
    return response, istrat, weights


def _stratum_split(
    rows: list[int],
    start: Sequence[float],
    stop: Sequence[float],
    status: Sequence[int],
    last: Sequence[bool],
    censoring: _CensoringCurve,
    entry: _CensoringCurve | None,
    utime: Sequence[float],
    enum: int,
) -> Any | None:
    """R's ``stratfun``: the ``finegray`` kernel call for one stratum, or ``None``."""

    times = sorted({stop[idx] for idx in rows if status[idx] == enum})
    if not times:
        return None
    maxtime = max(stop[idx] for idx in rows)
    if entry is not None:
        dtime = [-value for value in reversed(entry.time)]
        dprob = [*list(reversed(entry.surv))[1:], 1.0]
        combined = sorted(set(dtime) | set(censoring.time))
        gprob = [1.0, *censoring.surv]
        ctime = [utime[int(value) - 1] for value in combined]
        cprob = [
            dprob[max(bisect_right(dtime, value) - 1, 0)]
            * gprob[bisect_right(censoring.time, value)]
            for value in combined
        ]
    else:
        ctime = [utime[int(value) - 1] for value in censoring.time]
        cprob = list(censoring.surv)
    ct2 = [*ctime, maxtime]
    cp2 = [1.0, *cprob]
    keep = [False] * len(ct2)
    for value in times:
        position = bisect_left(ct2, value)
        if position < len(keep):
            keep[position] = True
    keep[0] = True
    return _core.finegray(
        [start[idx] for idx in rows],
        [stop[idx] for idx in rows],
        ct2,
        cp2,
        [status[idx] != 0 and status[idx] != enum and last[idx] for idx in rows],
        keep,
    )


def finegray(
    formula: str,
    data: Any | None = None,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = "na.pass",
    etype: Any | None = None,
    prefix: str = "fg",
    count: str | None = None,
    id: Any | None = None,
    timefix: bool = True,
    **kwargs: Any,
) -> FineGrayFrame:
    """R's ``finegray``: expand a multi-state ``Surv`` response into Fine-Gray weighted rows.

    The result carries the model-frame columns plus ``<prefix>start``,
    ``<prefix>stop``, ``<prefix>status``, ``<prefix>wt`` (and ``count``), with the
    selected endpoint as its ``event`` attribute.
    """

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "na.pass")
    if kwargs:
        raise TypeError(f"finegray got unexpected keyword argument(s): {', '.join(sorted(kwargs))}")
    if not isinstance(formula, str):
        raise ValueError("A formula argument is required")
    if not isinstance(prefix, str):
        raise TypeError("prefix must be a string")
    fix_time = _normalize_bool_option(timefix, "timefix")
    mf = model_frame(formula, data, subset=subset, na_action=na_action, weights=weights, id=id)
    if mf.n == 0:
        raise ValueError("No (non-missing) observations")
    response, istrat, user_weights = _finegray_inputs(mf, fix_time)
    first, last, delay = _subject_layout(response, mf.id)
    enum = _etype_index(list(response.states), etype)
    count_name = None if count is None else _make_names(str(count))
    output_names = [f"{prefix}{suffix}" for suffix in ("start", "stop", "status", "wt")]

    stop = list(response.time)
    if response.start is not None:
        start = list(response.start)
    else:
        minimum = min(stop)
        start = [0.0 if minimum > 0.0 else 2.0 * minimum - 1.0] * len(stop)
    status = [int(value) for value in response.event]
    utime = sorted(set(start) | set(stop))
    rank1 = [float(bisect_right(utime, value)) for value in start]
    rank2 = [
        float(bisect_right(utime, value)) - (0.2 if code != 0 else 0.0)
        for value, code in zip(stop, status, strict=True)
    ]
    censoring = _censoring_curves(
        rank1,
        rank2,
        [is_last and code == 0 for is_last, code in zip(last, status, strict=True)],
        istrat,
    )
    entry = (
        _censoring_curves([-v for v in rank2], [-v for v in rank1], first, istrat)
        if delay
        else None
    )

    variables = [
        (name, values) for name, values in _model_variables(mf) if not name.startswith("strata(")
    ]
    if mf.weights is not None:
        variables.append(("(weights)", list(mf.weights)))
    columns: dict[str, list[Any]] = {name: [] for name, _values in variables}
    for name in output_names:
        columns[name] = []
    if count_name is not None:
        columns[count_name] = []
    for stratum in range(max(istrat) + 1):
        rows = [idx for idx, value in enumerate(istrat) if value == stratum]
        split = _stratum_split(
            rows,
            start,
            stop,
            status,
            last,
            censoring[stratum],
            None if entry is None else entry[stratum],
            utime,
            enum,
        )
        if split is None:
            continue
        source = [rows[int(row) - 1] for row in split.row]
        for name, values in variables:
            columns[name].extend(values[idx] for idx in source)
        columns[output_names[0]].extend(float(value) for value in split.start)
        columns[output_names[1]].extend(float(value) for value in split.end)
        columns[output_names[2]].extend(1 if status[idx] == enum else 0 for idx in source)
        # R indexes the user weights by the row number within the stratum
        # (``user.weights[split$row]``), so with strata the weights follow the
        # position of the row in the whole data set; reproduced as is.
        columns[output_names[3]].extend(
            float(weight) * user_weights[int(row) - 1]
            for weight, row in zip(split.wt, split.row, strict=True)
        )
        if count_name is not None:
            columns[count_name].extend(int(value) for value in split.add)
    if not columns[output_names[0]]:
        raise ValueError("selected endpoint has no events")
    return FineGrayFrame(columns, event=response.states[enum - 1])
