"""``pyears``, ``survexp`` and rate-table helpers."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from datetime import date as _Date
from datetime import datetime as _DateTime
from itertools import product
from typing import Any

from .. import _survival as _core
from ._coerce import (
    _event_vector,
    _finite_float,
    _float_vector,
    _int_vector,
    _integer_scalar,
    _is_missing_value,
    _materialize_1d,
    _materialize_labels,
    _normalize_bool_option,
    _normalize_bool_option_with_default,
    _normalize_positive_scale,
    _r_formula_ordered_levels,
    _subset_indices,
    _subset_optional_sequence,
)
from ._data_prep import _survcondense_strata_values
from ._formula import (
    _apply_formula_na_action,
    _column,
    _combine_aligned_columns,
    _parse_formula,
    _subset_formula_inputs,
    _term_values,
)
from ._surv import Surv
from ._types import (
    _MISSING,
    PyearsResult,
    RateTable,
    SurvExpResult,
    TcutResult,
    _FormulaTerms,
    _InteractionTerm,
    _ModelClusterTerm,
    _ModelCovariateTerm,
    _ModelStrataTerm,
)


def is_ratetable(
    x: Any,
    has_rates: Any | None = None,
    has_dims: Any | None = None,
    verbose: Any = False,
) -> bool:
    """Return whether *x* is a population rate table, like R's ``is.ratetable``."""

    _normalize_bool_option(verbose, "verbose")
    if has_rates is not None or has_dims is not None:
        if has_rates is None or has_dims is None:
            raise TypeError("has_rates and has_dims must be supplied together")
        return _core.is_ratetable(
            _integer_scalar(x, "ndim"),
            _normalize_bool_option(has_rates, "has_rates"),
            _normalize_bool_option(has_dims, "has_dims"),
        )
    return isinstance(x, RateTable)


def _ratetable_date_from_components(
    year: Any,
    month: Any,
    day: Any,
    origin_year: Any,
) -> float:
    result = _core.ratetable_date(
        _integer_scalar(year, "year"),
        _integer_scalar(month, "month"),
        _integer_scalar(day, "day"),
        _integer_scalar(origin_year, "origin_year"),
    )
    return float(result.days)


def _ratetable_date_value(value: Any, origin_year: Any) -> float:
    if value is None or _is_missing_value(value):
        return math.nan
    if isinstance(value, _DateTime):
        return _ratetable_date_from_components(
            value.year,
            value.month,
            value.day,
            origin_year,
        )
    if isinstance(value, _Date):
        return _ratetable_date_from_components(
            value.year,
            value.month,
            value.day,
            origin_year,
        )
    if isinstance(value, str):
        parsed = _Date.fromisoformat(value[:10])
        return _ratetable_date_from_components(
            parsed.year,
            parsed.month,
            parsed.day,
            origin_year,
        )
    return float(value)


def ratetableDate(
    x: Any,
    month: Any | None = None,
    day: Any | None = None,
    *,
    origin_year: Any = 1970,
) -> Any:
    """Convert dates to rate-table day counts, matching R's ``ratetableDate``."""

    if month is not None or day is not None:
        if month is None or day is None:
            raise TypeError("month and day must be supplied together")
        return _ratetable_date_from_components(x, month, day, origin_year)
    if isinstance(x, Sequence) and not isinstance(x, str | bytes | bytearray):
        return [_ratetable_date_value(value, origin_year) for value in x]
    return _ratetable_date_value(x, origin_year)


def _survexp_ratetable(ratetable: Any | None) -> RateTable:
    if ratetable is None:
        return _core.survexp_us()
    if not isinstance(ratetable, RateTable):
        raise TypeError("ratetable must be a RateTable")
    return ratetable


def _normalize_survexp_method(
    method: Any | None,
    cohort: Any,
    conditional: Any,
) -> str:
    cohort_value = _normalize_bool_option_with_default(cohort, "cohort", True)
    conditional_value = _normalize_bool_option(conditional, "conditional")
    if method is None:
        if conditional_value:
            return "conditional"
        if not cohort_value:
            return "individual.s"
        return "hakulinen"
    if not isinstance(method, str):
        raise TypeError("method must be a string")
    value = method.strip().lower().replace("_", ".")
    aliases = {
        "ederer": "ederer",
        "hakulinen": "hakulinen",
        "conditional": "conditional",
        "individual": "individual",
        "individual.h": "individual.h",
        "individual.s": "individual.s",
    }
    if value not in aliases:
        raise ValueError(
            "method must be 'ederer', 'hakulinen', 'conditional', "
            "'individual.h', 'individual.s', or 'individual'"
        )
    return aliases[value]


def _survexp_result_from_core(result: Any, scale: float) -> SurvExpResult:
    return SurvExpResult(
        time=[float(value) / scale for value in result.time],
        surv=[float(value) for value in result.surv],
        n_risk=[float(value) for value in result.n_risk],
        cumhaz=[float(value) for value in result.cumhaz],
        method=str(result.method),
        n=int(result.n),
    )


def survexp(
    time: Any,
    age: Any,
    year: Any,
    ratetable: Any | None = None,
    sex: Any | None = None,
    times: Any | None = None,
    method: Any | None = None,
    *,
    cohort: Any = True,
    conditional: Any = False,
    scale: Any = 1.0,
    se_fit: Any | None = None,
) -> SurvExpResult | list[float]:
    """Compute expected survival from direct vectors and a population rate table."""

    if se_fit is not None and _normalize_bool_option(se_fit, "se_fit"):
        warnings.warn("se_fit value ignored", RuntimeWarning, stacklevel=2)
    method_value = _normalize_survexp_method(method, cohort, conditional)
    scale_value = _normalize_positive_scale(scale)
    table = _survexp_ratetable(ratetable)
    time_values = _float_vector(time, "time")
    age_values = _float_vector(age, "age")
    year_values = _float_vector(year, "year")
    sex_values = None if sex is None else _int_vector(sex, "sex")

    if method_value in {"individual.h", "individual.s"}:
        individual = _core.survexp_individual(
            time_values,
            age_values,
            year_values,
            table,
            sex_values,
        )
        values = [float(value) for value in individual]
        if method_value == "individual.s":
            return values
        return [-math.log(value) if value > 0.0 else math.inf for value in values]

    result = _core.survexp(
        time_values,
        age_values,
        year_values,
        table,
        sex_values,
        None if times is None else _float_vector(times, "times"),
        method_value,
    )
    return _survexp_result_from_core(result, scale_value)


def survexp_individual(
    time: Any,
    age: Any,
    year: Any,
    ratetable: Any | None = None,
    sex: Any | None = None,
) -> list[float]:
    """Return per-subject expected survival from direct vectors."""

    return [
        float(value)
        for value in _core.survexp_individual(
            _float_vector(time, "time"),
            _float_vector(age, "age"),
            _float_vector(year, "year"),
            _survexp_ratetable(ratetable),
            None if sex is None else _int_vector(sex, "sex"),
        )
    ]


def _pyears_response_from_direct(
    response: Any,
    *,
    time: Any,
    start: Any,
    stop: Any,
    event: Any,
) -> tuple[list[float], list[float], list[float] | None, int, bool]:
    if response is not None:
        if isinstance(response, Surv):
            if response.type == "right":
                return [], list(response.time), list(response.event), 2, True
            if response.type == "counting":
                if response.start is None:
                    raise ValueError("counting Surv response is missing start times")
                return list(response.start), list(response.time), list(response.event), 3, True
            raise ValueError("pyears supports only right-censored and counting Surv responses")
        if time is not _MISSING or stop is not _MISSING:
            raise ValueError("use either response or explicit time/start/stop inputs")
        stop_values = _float_vector(response, "time")
        event_values = None if event is _MISSING or event is None else _event_vector(event, "event")
        return [], stop_values, event_values, 2, True

    if stop is not _MISSING:
        if start is _MISSING:
            raise TypeError("start must be supplied with stop")
        start_values = _float_vector(start, "start")
        stop_values = _float_vector(stop, "stop")
        event_values = None if event is _MISSING or event is None else _event_vector(event, "event")
        return (
            start_values,
            stop_values,
            event_values,
            3 if event_values is not None else 2,
            (event_values is not None),
        )
    if start is not _MISSING:
        raise TypeError("stop must be supplied with start")
    if time is _MISSING:
        raise TypeError("pyears requires a response or time vector")
    stop_values = _float_vector(time, "time")
    event_values = None if event is _MISSING or event is None else _event_vector(event, "event")
    return [], stop_values, event_values, 2, True


def _pyears_validate_time_columns(
    start: list[float],
    stop: list[float],
    event: list[float] | None,
) -> None:
    n = len(stop)
    if start and len(start) != n:
        raise ValueError("start and stop must have the same length")
    if event is not None and len(event) != n:
        raise ValueError("event must have the same length as time")
    if n == 0:
        raise ValueError("pyears requires at least one observation")
    for idx, value in enumerate(stop):
        if not math.isfinite(value):
            raise ValueError(f"time contains non-finite value at index {idx}")
        if value < 0.0:
            raise ValueError(f"time contains negative value at index {idx}")
    for idx, value in enumerate(start):
        if not math.isfinite(value):
            raise ValueError(f"start contains non-finite value at index {idx}")
        if value < 0.0:
            raise ValueError(f"start contains negative value at index {idx}")
        if stop[idx] < value:
            raise ValueError(f"stop must be greater than or equal to start at index {idx}")


def _pyears_weights(weights: Any | None, n: int) -> list[float]:
    if weights is None:
        return [1.0] * n
    values = [_finite_float(value, "weights") for value in _materialize_1d(weights, "weights")]
    if len(values) != n:
        raise ValueError("weights must have the same length as the response")
    for idx, value in enumerate(values):
        if value < 0.0:
            raise ValueError(f"weights contains negative value at index {idx}")
    return values


def _pyears_group_codes(
    group: Any | None,
    n: int,
    *,
    levels: Sequence[Any] | None = None,
) -> tuple[list[float], list[str]]:
    if group is None:
        return [1.0] * n, ["(all)"]
    values = _materialize_labels(group, "group")
    if len(values) != n:
        raise ValueError("group must have the same length as the response")
    if levels is None:
        group_levels: list[Any] = []
        labels: dict[Any, int] = {}
        codes: list[float] = []
        for value in values:
            try:
                code = labels.get(value)
                if code is None:
                    code = len(labels) + 1
                    labels[value] = code
                    group_levels.append(value)
            except TypeError as exc:
                raise TypeError("group contains unhashable labels") from exc
            codes.append(float(code))
        return codes, [str(value) for value in group_levels]
    group_levels = tuple(levels)
    labels = {value: idx + 1 for idx, value in enumerate(group_levels)}
    try:
        codes = [float(labels[value]) for value in values]
    except KeyError as exc:
        raise ValueError("group contains a value outside the supplied levels") from exc
    return codes, [str(value) for value in group_levels]


def _pyears_formula_group_and_levels(
    data: Any,
    terms: _FormulaTerms,
    n: int,
) -> tuple[list[Any] | None, tuple[Any, ...] | None]:
    columns: list[list[Any]] = []
    if terms.model_terms:
        for model_term in terms.model_terms:
            if isinstance(model_term, _ModelCovariateTerm):
                columns.append(_term_values(data, model_term.term, n))
            elif isinstance(model_term, _ModelStrataTerm):
                columns.append(_survcondense_strata_values(data, model_term.columns, n))
            elif isinstance(model_term, _ModelClusterTerm):
                columns.append(_column(data, model_term.column))
    else:
        columns = [
            *[_column(data, term) for term in terms.strata],
            *[_term_values(data, term, n) for term in terms.covariates],
            *[_column(data, term) for term in terms.clusters],
        ]
    if not columns:
        return None, None
    group = _combine_aligned_columns(columns, n)
    column_levels = [
        _r_formula_ordered_levels(column, "pyears formula groups") for column in columns
    ]
    if len(column_levels) == 1:
        return group, column_levels[0]
    levels = tuple(tuple(reversed(parts)) for parts in product(*reversed(column_levels)))
    return group, levels


def _pyears_formula_inputs(
    formula: str,
    data: Any,
    weights: Any | None,
    subset: Any | None,
    na_action: str | None,
) -> tuple[Surv, list[Any] | None, tuple[Any, ...] | None, list[float] | None]:
    if data is None:
        raise ValueError("data is required when pyears response is a formula")
    aligned_weights = None if weights is None else _materialize_1d(weights, "weights")
    if subset is not None:
        data, aligned = _subset_formula_inputs(formula, data, subset, weights=aligned_weights)
        aligned_weights = aligned["weights"]
    data, aligned = _apply_formula_na_action(formula, data, na_action, weights=aligned_weights)
    aligned_weights = aligned["weights"]
    response, terms = _parse_formula(formula, data)
    if any(isinstance(term, _InteractionTerm) for term in terms.covariates):
        raise ValueError("pyears formula does not support interaction terms")
    group, group_levels = _pyears_formula_group_and_levels(data, terms, len(response))
    return (
        response,
        group,
        group_levels,
        None if aligned_weights is None else [float(value) for value in aligned_weights],
    )


def _pyears_result_frame(result: PyearsResult) -> dict[str, list[Any]]:
    frame: dict[str, list[Any]] = {
        "group": result.group,
        "pyears": result.pyears,
        "n": result.n,
    }
    if result.expected is not None:
        frame["expected"] = result.expected
    if result.event is not None:
        frame["event"] = result.event
    return frame


def _finegray_frame(result: Any) -> dict[str, list[Any]]:
    return {
        "row": [int(value) for value in result.row],
        "start": [float(value) for value in result.start],
        "end": [float(value) for value in result.end],
        "wt": [float(value) for value in result.wt],
        "add": [int(value) for value in result.add],
    }


def pyears(
    response: Any = None,
    data: Any | None = None,
    *,
    time: Any = _MISSING,
    start: Any = _MISSING,
    stop: Any = _MISSING,
    event: Any = _MISSING,
    group: Any | None = None,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = None,
    scale: Any = 365.25,
    data_frame: Any = False,
) -> PyearsResult | dict[str, list[Any]]:
    """Tabulate person-years for direct ``Surv``/formula inputs, like R's ``pyears``."""

    direct_inputs_ready = False
    if isinstance(response, str) and "~" in response:
        response, formula_group, formula_group_levels, formula_weights = _pyears_formula_inputs(
            response,
            data,
            weights,
            subset,
            na_action,
        )
        if group is not None:
            raise ValueError("group must not be supplied separately when response is a formula")
        group = formula_group
        weights = formula_weights
    else:
        formula_group_levels = None
        start_values, stop_values, event_values, ny, do_event = _pyears_response_from_direct(
            response,
            time=time,
            start=start,
            stop=stop,
            event=event,
        )
        n_direct = len(stop_values)
        if subset is not None:
            keep = _subset_indices(subset, n_direct)
            response = (
                Surv([stop_values[idx] for idx in keep], [event_values[idx] for idx in keep])
                if not start_values and event_values is not None
                else None
            )
            start = [start_values[idx] for idx in keep] if start_values else _MISSING
            stop = [stop_values[idx] for idx in keep]
            event = [event_values[idx] for idx in keep] if event_values is not None else _MISSING
            if isinstance(group, TcutResult):
                group = _core.tcut(
                    [float(group.values[idx]) for idx in keep],
                    list(group.breaks),
                    list(group.levels),
                )
            else:
                group = _subset_optional_sequence(group, keep, "group")
            weights = _subset_optional_sequence(weights, keep, "weights")
        else:
            direct_inputs_ready = True
    if not direct_inputs_ready:
        start_values, stop_values, event_values, ny, do_event = _pyears_response_from_direct(
            response,
            time=time,
            start=start,
            stop=stop,
            event=event,
        )
    n = len(stop_values)
    event_for_core = [0.0] * n if event_values is None else [float(value) for value in event_values]
    _pyears_validate_time_columns(start_values, stop_values, event_values)
    weight_values = _pyears_weights(weights, n)
    if isinstance(group, TcutResult):
        if formula_group_levels is not None:
            raise ValueError("formula group levels cannot be combined with a tcut group")
        if len(group.values) != n:
            raise ValueError("group must have the same length as the response")
        observed_factors = [0]
        observed_dims = [len(group.levels)]
        observed_cuts = [float(value) for value in group.breaks]
        observed_data = [float(value) for value in group.values]
        group_labels = [str(value) for value in group.levels]
        has_tcut = True
    else:
        group_codes, group_labels = _pyears_group_codes(
            group,
            n,
            levels=formula_group_levels,
        )
        observed_factors = [1]
        observed_dims = [len(group_labels)]
        observed_cuts = []
        observed_data = group_codes
        has_tcut = False
    scale_value = _normalize_positive_scale(scale)
    _normalize_bool_option(data_frame, "data_frame")

    time_data = (
        [*start_values, *stop_values, *event_for_core]
        if start_values and event_values is not None
        else [*start_values, *stop_values]
        if start_values
        else [*stop_values, *event_for_core]
    )
    raw = _core.perform_pyears_calculation(
        time_data,
        weight_values,
        0,
        [],
        [],
        [],
        [],
        [],
        1,
        observed_factors,
        observed_dims,
        observed_cuts,
        1,
        observed_data,
        1 if do_event else 0,
        ny,
    )
    result = PyearsResult(
        pyears=[float(value) / scale_value for value in raw["pyears"]],
        n=[float(value) for value in raw["pn"]],
        offtable=float(raw["offtable"]) / scale_value,
        group=group_labels,
        observations=n,
        event=[float(value) for value in raw["pcount"]] if event_values is not None else None,
        expected=None,
        tcut=has_tcut,
    )
    return _pyears_result_frame(result) if data_frame else result


def survexp_us() -> RateTable:
    """Return the bundled US population mortality rate table."""

    return _core.survexp_us()


def survexp_mn() -> RateTable:
    """Return the bundled Minnesota population mortality rate table."""

    return _core.survexp_mn()


def survexp_usr() -> RateTable:
    """Return the rural US population mortality rate table alias."""

    return _core.survexp_usr()
