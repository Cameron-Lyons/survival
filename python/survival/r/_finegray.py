"""``finegray`` competing-risks data expansion."""

from __future__ import annotations

import warnings
from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from numbers import Real
from typing import Any

from .. import _survival as _core
from ._coerce import (
    _finite_float,
    _group_indices,
    _is_missing_value,
    _materialize_1d,
    _materialize_labels,
    _normalize_bool_option,
    _normalize_na_action,
    _pop_dotted_keyword,
    _r_formula_ordered_levels,
    _strata_value_label,
)
from ._data_prep import _aeq_adjust_time_columns, _raise_if_aeq_zero_interval
from ._formula import (
    _apply_formula_na_action,
    _column,
    _column_or_values,
    _combine_aligned_columns,
    _covariate_term_name,
    _dot_terms,
    _formula_response_spec,
    _formula_response_values,
    _numeric_term_values,
    _response_operand,
    _response_rep_call,
    _split_terms,
    _subset_formula_inputs,
    _term_raw_values,
    _term_values,
    _top_level_comparison,
    _unwrap_response_identity,
)
from ._types import (
    FineGrayFrame,
    FineGrayOutput,
    _CovariateTerm,
    _FormulaModelTerm,
    _FormulaTerms,
    _InteractionTerm,
    _ModelCovariateTerm,
    _ModelOffsetTerm,
    _SurvResponseSpec,
)


@dataclass(frozen=True)
class _FineGrayResponse:
    start: list[float]
    stop: list[float]
    status: list[int]
    states: tuple[Any, ...]
    counting: bool


@dataclass(frozen=True)
class _FineGrayInputs:
    response: _FineGrayResponse
    model_columns: list[tuple[str, list[Any]]]
    user_weights: list[Any] | None
    numeric_weights: list[float]
    id_values: list[Any] | None
    strata: list[Any] | None
    strata_levels: tuple[Any, ...] | None


def _finegray_raw_column(data: Any, name: str) -> Any:
    if isinstance(data, Mapping):
        try:
            return data[name]
        except KeyError as exc:
            raise KeyError(f"column {name!r} not found in data") from exc
    try:
        return data[name]
    except Exception as exc:
        raise KeyError(f"column {name!r} not found in data") from exc


def _finegray_categorical_levels(values: Any) -> tuple[Any, ...] | None:
    categories = getattr(getattr(values, "dtype", None), "categories", None)
    if categories is None:
        try:
            categories = values.categories
        except (AttributeError, TypeError):
            categories = None
    if categories is None:
        try:
            accessor = values.cat
        except (AttributeError, TypeError):
            accessor = None
        if accessor is not None:
            categories = getattr(accessor, "categories", None)
            if categories is None:
                get_categories = getattr(accessor, "get_categories", None)
                if callable(get_categories):
                    categories = get_categories()
    if categories is None:
        return None
    return tuple(_materialize_1d(categories, "event categories"))


def _finegray_declared_levels(data: Any, spec: _SurvResponseSpec) -> tuple[Any, ...] | None:
    if len(spec.arguments) < 2:
        return None
    expression = _unwrap_response_identity(spec.arguments[-1])
    if _top_level_comparison(expression) is not None or _response_rep_call(expression) is not None:
        return None
    try:
        operand = _response_operand(expression, allow_literal=False)
    except ValueError:
        return None
    if operand.column is None:
        return None
    raw_values = _finegray_raw_column(data, operand.column)
    declared = _finegray_categorical_levels(raw_values)
    if _finegray_factor_response(spec):
        if declared is None:
            declared = _finegray_factor_levels(
                [
                    value
                    for value in _materialize_1d(raw_values, "event")
                    if not _is_missing_value(value)
                ]
            )
        else:
            declared = tuple(_finegray_factor_label(value) for value in declared)
    elif declared is None:
        observed = [
            value for value in _materialize_1d(raw_values, "event") if not _is_missing_value(value)
        ]
        if observed and all(isinstance(value, str) for value in observed):
            declared = _finegray_event_levels(observed, None)
    return declared


def _finegray_factor_response(spec: _SurvResponseSpec) -> bool:
    if len(spec.arguments) < 2:
        return False
    expression = spec.arguments[-1].strip()
    return any(
        expression.startswith(f"{wrapper}(") and expression.endswith(")")
        for wrapper in ("factor", "as.factor")
    )


def _finegray_factor_label(value: Any) -> str:
    return _strata_value_label(value)


def _finegray_factor_levels(values: list[Any]) -> tuple[str, ...]:
    levels = _finegray_unique_levels(values, "event")
    if all(isinstance(value, str) for value in levels) or all(
        isinstance(value, bool) for value in levels
    ):
        ordered = sorted(levels)
    elif all(isinstance(value, Real) and not isinstance(value, bool) for value in levels):
        ordered = sorted(levels, key=float)
    else:
        ordered = sorted(levels, key=lambda value: _finegray_factor_label(value))
    return tuple(_finegray_factor_label(value) for value in ordered)


def _finegray_unique_levels(values: Sequence[Any], name: str) -> tuple[Any, ...]:
    levels: dict[Any, None] = {}
    for value in values:
        if _is_missing_value(value):
            raise ValueError(f"{name} must not contain missing values")
        try:
            levels.setdefault(value, None)
        except TypeError as exc:
            raise TypeError(f"{name} must contain hashable labels") from exc
    return tuple(levels)


def _finegray_event_levels(
    event: list[Any],
    declared_levels: tuple[Any, ...] | None,
) -> tuple[Any, ...]:
    if declared_levels is not None:
        levels = _finegray_unique_levels(declared_levels, "event categories")
    else:
        if any(not isinstance(value, str) for value in event if not _is_missing_value(value)):
            raise ValueError(
                "Fine-Gray formula status must be categorical; use string labels or an "
                "ordered categorical column"
            )
        levels = _finegray_unique_levels(event, "event")
        censor_names = {
            "(censor)",
            "0",
            "censor",
            "censored",
            "censoring",
            "no event",
            "no_event",
        }
        censor = next(
            (
                value
                for value in levels
                if isinstance(value, str) and value.strip().lower() in censor_names
            ),
            None,
        )
        if censor is not None and levels[0] != censor:
            levels = (censor, *(value for value in levels if value != censor))
    if len(levels) < 3:
        raise ValueError("survival time has only a single state")
    for state in levels[1:]:
        if _is_missing_value(state) or (isinstance(state, str) and not state):
            raise ValueError("each state must have a non-blank name")
    return levels


def _finegray_response_from_formula(
    data: Any,
    spec: _SurvResponseSpec,
    declared_levels: tuple[Any, ...] | None,
    *,
    timefix: bool,
) -> _FineGrayResponse:
    response_columns = _formula_response_values(data, spec)
    if len(response_columns) not in {2, 3}:
        raise ValueError("Fine-Gray formula response must contain time and categorical status")
    counting = len(response_columns) == 3
    expected_type = "counting" if counting else "right"
    if spec.type in {"right", "counting"} and spec.type != expected_type:
        raise ValueError("Wrong number of args for this type of survival data")
    if spec.type not in {None, "mstate", expected_type}:
        raise ValueError("Fine-Gray model requires a multi-state survival")
    if counting:
        raw_start, raw_stop, raw_event = response_columns
    else:
        raw_stop, raw_event = response_columns
        raw_start = []
    n = len(raw_stop)
    if n == 0:
        raise ValueError("No (non-missing) observations")
    if len(raw_event) != n or (counting and len(raw_start) != n):
        raise ValueError("formula response columns must have the same length")

    stop = [_finite_float(value, "time") - spec.origin for value in raw_stop]
    start = [_finite_float(value, "start") - spec.origin for value in raw_start] if counting else []
    if counting:
        for idx, (left, right) in enumerate(zip(start, stop, strict=True)):
            if left >= right:
                raise ValueError(f"stop must be greater than start at index {idx}")
        if timefix:
            adjusted_start, adjusted_stop = _aeq_adjust_time_columns((start, stop), None)
            _raise_if_aeq_zero_interval(start, stop, adjusted_start, adjusted_stop)
            start, stop = adjusted_start, adjusted_stop
    elif timefix:
        (stop,) = _aeq_adjust_time_columns((stop,), None)

    event = (
        [
            value if _is_missing_value(value) else _finegray_factor_label(value)
            for value in raw_event
        ]
        if _finegray_factor_response(spec)
        else list(raw_event)
    )
    levels = _finegray_event_levels(event, declared_levels)
    try:
        level_codes = {value: idx for idx, value in enumerate(levels)}
        status = [level_codes[value] for value in event]
    except TypeError as exc:
        raise TypeError("event must contain hashable labels") from exc
    except KeyError as exc:
        raise ValueError(f"event contains undeclared category {exc.args[0]!r}") from exc
    return _FineGrayResponse(
        start=start,
        stop=stop,
        status=status,
        states=levels[1:],
        counting=counting,
    )


def _finegray_model_columns(
    data: Any,
    terms: _FormulaTerms,
    n: int,
) -> list[tuple[str, list[Any]]]:
    columns: list[tuple[str, list[Any]]] = []
    seen: set[str] = set()

    def append_term(term: _CovariateTerm) -> None:
        name = _covariate_term_name(term)
        if name not in seen:
            seen.add(name)
            values = (
                _term_raw_values(data, term, n)
                if term.transform in {"I", "identity"} and term.arithmetic is None
                else _term_values(data, term, n)
            )
            columns.append((name, values))

    model_terms: Sequence[_FormulaModelTerm]
    model_terms = terms.model_terms or [_ModelCovariateTerm(term) for term in terms.covariates]
    for model_term in model_terms:
        if isinstance(model_term, _ModelCovariateTerm):
            if isinstance(model_term.term, _InteractionTerm):
                for factor in model_term.term.factors:
                    append_term(factor)
            else:
                append_term(model_term.term)
        elif isinstance(model_term, _ModelOffsetTerm):
            name = f"offset({_covariate_term_name(model_term.term)})"
            if name not in seen:
                seen.add(name)
                values = _numeric_term_values(
                    _term_raw_values(data, model_term.term, n),
                    model_term.term,
                )
                columns.append((name, values))
    return columns


def _finegray_declared_strata_levels(
    data: Any,
    terms: _FormulaTerms,
) -> list[tuple[Any, ...] | None]:
    return [
        _finegray_categorical_levels(_finegray_raw_column(data, column)) for column in terms.strata
    ]


def _finegray_strata_values_and_levels(
    data: Any,
    terms: _FormulaTerms,
    n: int,
    declared_levels: list[tuple[Any, ...] | None],
) -> tuple[list[Any] | None, tuple[Any, ...] | None]:
    if not terms.strata:
        return None, None
    columns = [_column(data, term) for term in terms.strata]
    if any(len(column) != n for column in columns):
        raise ValueError("formula columns must have the same length as the Surv response")
    if any(_is_missing_value(value) for column in columns for value in column):
        raise ValueError("strata must not contain missing values")

    column_levels: list[tuple[Any, ...]] = []
    for column, declared in zip(columns, declared_levels, strict=True):
        observed = _finegray_unique_levels(column, "strata")
        levels = declared or _r_formula_ordered_levels(column, "finegray strata")
        if any(value not in levels for value in observed):
            raise ValueError("strata contains a value outside the declared categories")
        column_levels.append(tuple(value for value in levels if value in observed))

    strata = _combine_aligned_columns(columns, n)
    levels = column_levels[0] if len(column_levels) == 1 else tuple(product(*column_levels))
    observed_strata = _finegray_unique_levels(strata, "strata")
    return strata, tuple(level for level in levels if level in observed_strata)


def _finegray_formula_inputs(
    formula: str,
    data: Any,
    weights: Any | None,
    subset: Any | None,
    na_action: str | None,
    id_values: Any | None,
    *,
    timefix: bool,
) -> _FineGrayInputs:
    if data is None:
        raise ValueError("data is required when finegray response is a formula")
    spec = _formula_response_spec(formula)
    declared_levels = _finegray_declared_levels(data, spec)
    _lhs, _sep, rhs = formula.partition("~")
    initial_terms = _split_terms(rhs, _dot_terms(data, list(spec.columns)))
    declared_strata_levels = _finegray_declared_strata_levels(data, initial_terms)
    weights = _column_or_values(data, weights, "weights") if weights is not None else None
    id_values = _column_or_values(data, id_values, "id") if id_values is not None else None
    if subset is not None:
        data, aligned = _subset_formula_inputs(
            formula,
            data,
            subset,
            weights=weights,
            id=id_values,
        )
        weights = aligned["weights"]
        id_values = aligned["id"]
    data, aligned = _apply_formula_na_action(
        formula,
        data,
        na_action,
        weights=weights,
        id=id_values,
    )
    weights = aligned["weights"]
    id_values = aligned["id"]
    response = _finegray_response_from_formula(
        data,
        spec,
        declared_levels,
        timefix=timefix,
    )
    n = len(response.stop)
    terms = _split_terms(rhs, _dot_terms(data, list(spec.columns)))
    if terms.clusters:
        raise ValueError("a cluster() term is not valid")
    model_columns = _finegray_model_columns(data, terms, n)

    user_weights = None if weights is None else _materialize_1d(weights, "weights")
    if user_weights is not None and len(user_weights) != n:
        raise ValueError("weights must have the same length as the response")
    numeric_weights = (
        [1.0] * n
        if user_weights is None
        else [_finite_float(value, "weights") for value in user_weights]
    )

    materialized_id = None if id_values is None else _materialize_labels(id_values, "id")
    if materialized_id is not None and len(materialized_id) != n:
        raise ValueError("id must have the same length as the response")
    strata, strata_levels = _finegray_strata_values_and_levels(
        data,
        terms,
        n,
        declared_strata_levels,
    )
    return _FineGrayInputs(
        response=response,
        model_columns=model_columns,
        user_weights=user_weights,
        numeric_weights=numeric_weights,
        id_values=materialized_id,
        strata=strata,
        strata_levels=strata_levels,
    )


def _finegray_selected_state(states: tuple[Any, ...], etype: Any | None) -> tuple[int, Any]:
    def state_index(value: Any) -> int | None:
        for idx, state in enumerate(states):
            try:
                equal = bool(value == state)
            except (TypeError, ValueError):
                equal = False
            if equal:
                return idx
        return None

    if etype is None:
        selected = [states[0]]
    elif isinstance(etype, str | bytes) or (
        isinstance(etype, tuple) and state_index(etype) is not None
    ):
        selected = [etype]
    else:
        try:
            selected = _materialize_1d(etype, "etype")
        except TypeError:
            selected = [etype]

    selected_indices: list[int] = []
    for value in selected:
        match = state_index(value)
        if match is None:
            raise ValueError("etype argument has a state that is not in the data")
        selected_indices.append(match)
    if not selected:
        raise ValueError("etype argument has a state that is not in the data")
    if len(selected) > 1:
        warnings.warn("only the first endpoint was used", RuntimeWarning, stacklevel=2)
    endpoint = selected[0]
    return selected_indices[0] + 1, endpoint


def _finegray_counting_layout(
    response: _FineGrayResponse,
    id_values: list[Any] | None,
) -> tuple[list[bool], list[bool], bool]:
    n = len(response.stop)
    if not response.counting:
        return [False] * n, [True] * n, False
    if id_values is None:
        raise ValueError("(start, stop] data requires a subject id")
    groups: dict[Any, list[int]] = {}
    for idx, value in enumerate(id_values):
        if _is_missing_value(value):
            raise ValueError("id must not contain missing values")
        try:
            groups.setdefault(value, []).append(idx)
        except TypeError as exc:
            raise TypeError("id must contain hashable labels") from exc

    first = [False] * n
    last = [False] * n
    first_rows: list[int] = []
    for rows in groups.values():
        ordered = sorted(rows, key=lambda idx: (response.stop[idx], idx))
        first[ordered[0]] = True
        last[ordered[-1]] = True
        first_rows.append(ordered[0])
        for idx in ordered[:-1]:
            if response.status[idx] != 0:
                raise ValueError("a subject has a transition before their last time point")
        for left, right in zip(ordered, ordered[1:], strict=False):
            if response.stop[left] != response.start[right]:
                raise ValueError("a subject has gaps in time")
    minimum_stop = min(response.stop)
    delayed = any(response.start[idx] > minimum_stop for idx in first_rows)
    return first, last, delayed


def _finegray_censor_curve(
    start: list[float],
    stop: list[float],
    event: list[bool],
) -> tuple[list[float], list[float]]:
    event_counts: dict[float, int] = {}
    for time, observed in zip(stop, event, strict=True):
        if observed:
            event_counts[time] = event_counts.get(time, 0) + 1
    if not event_counts:
        return [], []
    sorted_start = sorted(start)
    sorted_stop = sorted(stop)
    survival = 1.0
    curve_time: list[float] = []
    curve_survival: list[float] = []
    for time in sorted(event_counts):
        risk = bisect_left(sorted_start, time) - bisect_left(sorted_stop, time)
        deaths = event_counts[time]
        if risk <= 0 or deaths > risk:
            raise ValueError("invalid censoring risk set in Fine-Gray expansion")
        survival *= 1.0 - deaths / risk
        curve_time.append(time)
        curve_survival.append(survival)
    return curve_time, curve_survival


def _finegray_rank_time(utime: list[float], rank: float) -> float:
    index_value = int(rank)
    if float(index_value) != rank or not 1 <= index_value <= len(utime):
        raise ValueError("invalid ranked time in Fine-Gray expansion")
    return utime[index_value - 1]


def _finegray_censoring_grid(
    start_rank: list[float],
    stop_rank: list[float],
    status: list[int],
    first: list[bool],
    last: list[bool],
    utime: list[float],
    maximum_time: float,
    target_times: list[float],
    *,
    delayed: bool,
) -> tuple[list[float], list[float], list[bool]]:
    gtime, gsurv = _finegray_censor_curve(
        start_rank,
        stop_rank,
        [is_last and value == 0 for is_last, value in zip(last, status, strict=True)],
    )
    if delayed:
        htime, hsurv = _finegray_censor_curve(
            [-value for value in stop_rank],
            [-value for value in start_rank],
            first,
        )
        dtime = list(reversed([-value for value in htime]))
        dprob = list(reversed(hsurv))[1:] + [1.0]
        combined = sorted(set(dtime + gtime))
        ctime = [_finegray_rank_time(utime, value) for value in combined]
        gprob = [1.0, *gsurv]
        cprob = []
        for value in combined:
            dindex = bisect_right(dtime, value)
            hvalue = dprob[max(dindex - 1, 0)]
            gvalue = gprob[bisect_right(gtime, value)]
            cprob.append(hvalue * gvalue)
    else:
        ctime = [_finegray_rank_time(utime, value) for value in gtime]
        cprob = list(gsurv)

    cut_times = [*ctime, maximum_time]
    cut_probabilities = [1.0, *cprob]
    keep = [False] * len(cut_times)
    keep[0] = True
    for target_time in target_times:
        position = bisect_left(cut_times, target_time)
        if position < len(keep):
            keep[position] = True
    return cut_times, cut_probabilities, keep


def _finegray_split_intervals(
    start: list[float],
    stop: list[float],
    cut_time: list[float],
    cut_probability: list[float],
    extend: list[bool],
    keep: list[bool],
) -> FineGrayOutput:
    return _core.finegray(start, stop, cut_time, cut_probability, extend, keep)


_R_RESERVED_NAMES = {
    "break",
    "else",
    "FALSE",
    "for",
    "function",
    "if",
    "Inf",
    "NA",
    "NA_character_",
    "NA_complex_",
    "NA_integer_",
    "NA_real_",
    "NaN",
    "next",
    "NULL",
    "repeat",
    "TRUE",
    "while",
}


def _finegray_make_name(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("count must be a string or None")
    name = "".join(char if char.isalnum() or char in {".", "_"} else "." for char in value)
    if not name or not (name[0].isalpha() or (name[0] == "." and not name[1:2].isdigit())):
        name = f"X{name}"
    if name in _R_RESERVED_NAMES:
        name += "."
    return name


def finegray(
    formula: str,
    data: Any | None = None,
    *,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = "pass",
    etype: Any | None = None,
    prefix: str = "fg",
    count: str | None = None,
    id: Any | None = None,
    timefix: bool = True,
    **kwargs: Any,
) -> FineGrayFrame:
    """Create Fine-Gray weighted interval data from a multi-state ``Surv`` formula."""

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "pass")
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"finegray got unexpected keyword argument(s): {unexpected}")

    if not isinstance(formula, str):
        raise TypeError("finegray formula must be a string")

    fix_time = _normalize_bool_option(timefix, "timefix")
    if not isinstance(prefix, str):
        raise TypeError("prefix must be a string")
    inputs = _finegray_formula_inputs(
        formula,
        data,
        weights,
        subset,
        _normalize_na_action(na_action),
        id,
        timefix=fix_time,
    )
    response = inputs.response
    endpoint_code, endpoint = _finegray_selected_state(response.states, etype)
    first, last, delayed = _finegray_counting_layout(response, inputs.id_values)

    if response.counting:
        start = list(response.start)
    else:
        minimum_stop = min(response.stop)
        zero = 0.0 if minimum_stop > 0.0 else 2.0 * minimum_stop - 1.0
        start = [zero] * len(response.stop)
    stop = list(response.stop)
    utime = sorted(set(start + stop))
    start_rank = [float(bisect_right(utime, value)) for value in start]
    stop_rank = [float(bisect_right(utime, value)) for value in stop]
    for idx, value in enumerate(response.status):
        if value != 0:
            stop_rank[idx] -= 0.2

    n = len(stop)
    if inputs.strata is None:
        groups = [(None, list(range(n)))]
    else:
        groups = list(_group_indices(inputs.strata, n, levels=inputs.strata_levels).items())

    source_rows: list[int] = []
    expanded_start: list[float] = []
    expanded_stop: list[float] = []
    expanded_status: list[float] = []
    expanded_weight: list[float] = []
    expanded_add: list[int] = []
    for _level, rows in groups:
        target_times = sorted({stop[idx] for idx in rows if response.status[idx] == endpoint_code})
        if not target_times:
            continue
        local_start = [start[idx] for idx in rows]
        local_stop = [stop[idx] for idx in rows]
        local_status = [response.status[idx] for idx in rows]
        local_first = [first[idx] for idx in rows]
        local_last = [last[idx] for idx in rows]
        cut_time, cut_probability, keep = _finegray_censoring_grid(
            [start_rank[idx] for idx in rows],
            [stop_rank[idx] for idx in rows],
            local_status,
            local_first,
            local_last,
            utime,
            max(local_stop),
            target_times,
            delayed=delayed,
        )
        raw = _finegray_split_intervals(
            local_start,
            local_stop,
            cut_time,
            cut_probability,
            [
                value != 0 and value != endpoint_code and is_last
                for value, is_last in zip(local_status, local_last, strict=True)
            ],
            keep,
        )
        for raw_row, left, right, weight, add in zip(
            raw.row,
            raw.start,
            raw.end,
            raw.wt,
            raw.add,
            strict=True,
        ):
            local_idx = int(raw_row) - 1
            source_idx = rows[local_idx]
            source_rows.append(source_idx)
            expanded_start.append(float(left))
            expanded_stop.append(float(right))
            expanded_status.append(float(response.status[source_idx] == endpoint_code))
            expanded_weight.append(float(weight) * inputs.numeric_weights[source_idx])
            expanded_add.append(int(add))

    if not source_rows:
        raise ValueError("selected endpoint has no events")
    columns: dict[str, list[Any]] = {
        name: [values[idx] for idx in source_rows] for name, values in inputs.model_columns
    }
    if inputs.user_weights is not None:
        columns["(weights)"] = [inputs.user_weights[idx] for idx in source_rows]
    output_names = [f"{prefix}{suffix}" for suffix in ("start", "stop", "status", "wt")]
    columns[output_names[0]] = expanded_start
    columns[output_names[1]] = expanded_stop
    columns[output_names[2]] = expanded_status
    columns[output_names[3]] = expanded_weight
    if count is not None:
        columns[_finegray_make_name(count)] = expanded_add
    return FineGrayFrame(columns, event=endpoint)
